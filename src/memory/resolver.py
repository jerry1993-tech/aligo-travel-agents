# -*- coding: utf-8 -*-
"""把记忆接进动态 Prompt —— 包一层 :data:`ContextResolver`。

═══ 接缝在哪 ═══

``ContextInjectionMiddleware``（:mod:`src.orchestration.context`）每次拼
system prompt 时调用一个 ``ContextResolver``，拿回一个
:class:`~src.orchestration.context.PromptContext`。那个对象里有一个
``profile_summary`` 字段，从 P3 起就一直空着，注释写着「P4 的记忆模块提供」。

本模块就是那个提供者：它**包住**原有的解析器，在它给出的结果上补一个
``profile_summary``，其余部分原样透传。

⚠️ 用「包一层」而不是「重写一个解析器」，是因为原解析器（读阶段、读路由
决策）与记忆是**两件不相干的事**，改动它意味着记忆模块要懂
:mod:`src.orchestration.lane` 的内部结构。包一层之后，记忆模块只需要
认识 ``PromptContext`` 这一个类型。

═══ ⚠️ 为什么是 async ═══

记忆要查库（画像）和查向量库（语义召回），两者都是 I/O。
``ContextResolver`` 原来被声明成同步的 —— 那是 P3 时的实情
（默认解析器只读 agent 内存里的状态，不做 I/O）。

这里把它放宽成「可以返回 awaitable」，由
``ContextInjectionMiddleware._resolve`` 负责 ``await``。
⚠️ 放宽是**向后兼容**的：既有的同步解析器（以及它们的单测）一行都不用改。

═══ ⚠️ 查询文本从哪来 ═══

语义召回需要一句「用户现在在问什么」。可用的来源有三个，取舍如下：

  · ``agent.state.context`` —— **选它**。它就在内存里，同步可读，
    不需要任何 I/O；
  · 框架传给 ``on_model_call`` 的 ``messages`` —— 拿不到，
    那个钩子和 ``on_system_prompt`` 是两条独立的链路；
  · 存进 ``middle_context`` 供后续读取 —— 那要把用户原文写进会话状态，
    多一份留存，而它并没有换来更好的召回。

⚠️ ``state.context`` 里可能有**很多**条历史消息，取最后一条用户消息的
逻辑与 :func:`src.orchestration.lane._last_user_text` 是同一套
（从后往前找、用 ``get_text_content()``），原因见那里的注释。
这里不复用那个函数，是因为它是 ``LaneRouterMiddleware`` 的
``@staticmethod``，为一个纯文本提取去 import 中间件类会把
路由模块拖进记忆模块的依赖图 —— 而它们之间确实没有关系。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from src.orchestration.context import PromptContext

if TYPE_CHECKING:
    from collections.abc import Callable

    from .service import TravelerMemory

logger = logging.getLogger(__name__)

#: 送去语义召回的查询文本上限（字符）。
#:
#: ⚠️ 截断是必须的：一句 2000 字的输入会变成一次 2000 字的 embedding
#: 请求，而多出来的那 1900 字对「召回哪几条画像笔记」几乎没有影响 ——
#: 用户真正在问的那件事通常在最前面。
_MAX_QUERY_CHARS = 512


def last_user_text(agent: Any) -> str:
    """取 agent 当前上下文里**最后一条用户消息**的纯文本。

    ⚠️ **必须从后往前找**。框架会在上下文末尾放一条空的 assistant 占位
    消息等着填输出，所以 ``context[-1]`` 通常**不是**用户消息 ——
    只看最后一个元素的写法在这里恒为失败（同样的坑在
    :func:`src.orchestration.lane.LaneRouterMiddleware._last_user_text`
    里已经踩过一次）。

    ⚠️ 任何异常都降级为空串。本函数跑在主链路上，而调用方对
    「拿不到查询文本」有明确的兜底（只召回结构化画像）。

    Args:
        agent (`Any`): 框架传入的 agent 实例。

    Returns:
        `str`: 用户文本；没有则空串。
    """
    try:
        messages = getattr(getattr(agent, "state", None), "context", None) or []
        for message in reversed(list(messages)):
            if getattr(message, "role", None) != "user":
                continue
            getter = getattr(message, "get_text_content", None)
            if getter is None:
                return ""
            # ⚠️ 只取**最后一条**用户消息。多轮对话里更早的用户消息属于
            # 历史上下文，拿它们当查询会让记忆一直被同一句旧话牵着走。
            return (getter() or "").strip()
    except Exception:  # noqa: BLE001 —— 见文档，主链路不抛
        logger.debug("读取用户消息文本失败，本轮只召回结构化画像。", exc_info=True)
    return ""


def make_memory_resolver(
    base: "Callable[[Any], PromptContext | Awaitable[PromptContext]]",
    memory: "TravelerMemory",
    user_id: str,
) -> "Callable[[Any], Awaitable[PromptContext]]":
    """把 ``base`` 解析器包成「带长期画像」的版本。

    ⚠️ ``user_id`` 是**装配时捕获**的，不是运行时从 agent 上读的。
    中间件工厂（``AgentMiddlewareFactory``）被框架调用时会把
    ``user_id`` 作为参数传进来（``agentscope/app/_service/_chat.py:994-1000``），
    那是**唯一**可信的来源。从 agent 上猜（比如读它的名字或某个
    context 字段）会得到「看起来能用、偶尔串到别人画像」的行为 ——
    而串号在多租户系统里是事故。

    ⚠️ 整个函数体包在 try 里，**永不抛**。它跑在 ``on_system_prompt``
    的主链路上，而 ``TravelerMemory`` 内部已经保证读路径不抛；
    这里再兜一层，是为了挡住「memory 对象本身是坏的」这种情况
    （比如某个测试传了个假的）。

    Args:
        base: 原解析器（同步或异步皆可）。
        memory (`TravelerMemory`): 记忆门面。
        user_id (`str`): 当前用户标识。

    Returns:
        `Callable[[Any], Awaitable[PromptContext]]`: 异步解析器。
    """

    async def resolver(agent: Any) -> PromptContext:
        """解析出带画像的上下文。"""
        try:
            # ⚠️ ``base`` 可能是同步的也可能是异步的（``ContextResolver``
            # 放宽后两种都允许），所以先拿到结果再看它是不是 awaitable。
            # 判断用 ``isinstance(..., Awaitable)`` 而不是
            # ``inspect.isawaitable`` —— 前者不 import inspect，
            # 而两者对「协程 / Task / 自定义 __await__ 对象」的判断一致。
            context = base(agent)
            if isinstance(context, Awaitable):
                context = await context

            if not user_id:
                return context

            section = await memory.render_prompt_section(
                user_id,
                last_user_text(agent)[:_MAX_QUERY_CHARS],
            )
            if not section:
                # ⚠️ 没有画像时**返回原对象**，而不是
                # ``replace(context, profile_summary="")`` ——
                # 两者语义相同，但前者不新建对象，而这条路径
                # （没有画像的新用户）恰恰是最常走的一条。
                return context

            # ``PromptContext`` 是 ``frozen=True`` 的，所以用 ``replace``
            # 而不是属性赋值（后者会抛 FrozenInstanceError）。
            return replace(context, profile_summary=section)
        except Exception:  # noqa: BLE001 —— 见文档，主链路不抛
            logger.warning("注入长期画像失败，本轮沿用无画像的 prompt。", exc_info=True)
            try:
                fallback = base(agent)
                if isinstance(fallback, Awaitable):
                    fallback = await fallback
                return fallback
            except Exception:  # noqa: BLE001
                return PromptContext()

    return resolver


__all__ = ["last_user_text", "make_memory_resolver"]
