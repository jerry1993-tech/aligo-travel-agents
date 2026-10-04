# -*- coding: utf-8 -*-
"""**拦下 HintBlock 事件** —— 提示块是给模型看的，不该出现在用户界面上。

文件职责：
    实现 ``HintSuppressionMiddleware``。它只做一件事：把回复流里的
    ``HintBlockEvent`` 吃掉，不放行给下游。

上下游依赖：
    - 上游：``agentscope.middleware.MiddlewareBase`` 的 ``on_reply`` 钩子、
      ``agentscope.event.HintBlockEvent``。
    - 下游：``src/server/agents_factory.py`` 的 ``build_middlewares_factory``
      把它装配进每一条回复链路。

═══ 为什么需要这个中间件（这是踩出来的，不是设计出来的） ═══

2026-10-03 的对抗审计顺着「用户能看到什么」这条线一路查到：**框架注入的
每一个 HintBlock 都会原样出现在用户界面上**，而且刷新页面之后还在。

事实链（均已核实，附证据位置）：

1. ``Agent._inject_runtime_state`` 把一段运行时状态包进 HintBlock 追加到
   上下文（``agentscope/agent/_agent.py:1629-1632``），随后 yield 一个
   ``HintBlockEvent``（``agentscope/agent/_agent.py:1633-1639``）。
   那段文本的默认模板是**英文提示词**（``agent/_config.py:285-292``）::

       <system-reminder>Treat the following as the ground truth at this point
       of the conversation. Anything stated earlier is outdated, ...

   也就是说，用户在界面上展开「运行状态」，读到的是一句对模型下的英文指令。

2. ``ChatService`` 把 agent 的**每一个**事件无条件
   ``publish_session_event(...)``（``app/_service/_chat.py:1250-1269``），
   没有任何按类型的过滤 —— 事件直达 message bus。

3. ``GET /sessions/{id}/stream`` 把总线上的事件原样写成 SSE 帧
   （``app/_router/_session.py:907-915``），前端 ``case 'hint'`` 把它渲染成
   一个可折叠的「系统消息 - 运行状态」（``ASMessageBubble.tsx:670-729``）。

4. **刷新也躲不掉**：``Msg.append_event`` 对 ``HINT_BLOCK`` 的处理是把整块
   追加进消息 content，注释写明 "for persistence and replay"
   （``agentscope/message/_base.py:372-382``）；``ChatService`` 随后
   ``upsert_message(reply_msg)``（``app/_service/_chat.py:1456-1463``），于是
   ``GET /sessions/{id}/messages`` 每次都把它带回来。

═══ 为什么在**中间件**这一层拦，而不是别的三处 ═══

- **不是前端过滤**：数据仍然发到浏览器，只是不画。信息已经出网，
  而且换任何别的客户端（CLI、第三方集成）就漏了。
- **不是改 ``InjectionConfig.emit_hint_event``**：那是 ``InjectionConfig``
  的字段（``agent/_config.py:353-359``），而**装配 agent 的不是我们**
  —— 框架在 ``app/_service/`` 内部构造 agent，从头到尾没给我们传
  ``injection_config`` 的口子。况且那个开关只覆盖「运行时状态」一种来源，
  RAG（``middleware/_rag.py:995``）、收件箱（``app/middleware/_inbox_middleware.py:128``）、
  团队成员回报（``app/middleware/_team_member_middleware.py:215``）各自
  还会发自己的 HintBlockEvent。
- **不是改 ``/sessions/{id}/messages`` 的返回**：那是框架路由。

中间件是唯一「我们自己的、且在事件离开 agent 之后、进入总线之前」的位置。
吃掉事件之后，第 2、3、4 条链路**同时**断掉 —— 因为第 4 条的持久化也依赖
这个事件（``append_event`` 没被调用，content 里就没有这一块）。

═══ 为什么可以放心吃掉：HintBlock 本来就是**智能体之间**的通道 ═══

按 :mod:`src.server.agents_factory` 的装配说明，子智能体的产出**就是**
以 HintBlock 的形式回到主智能体的。也就是说这些块有两个读者：模型（通过
上下文）与其他智能体。**用户不在其中** —— 用户该看的是主智能体组织好的
答复，以及 :mod:`src.chains.events` 那套专门写好的中文进度文案
（「正在判断该走哪条处理流程」），而不是原始提示块。

吃掉事件**不影响模型**：HintBlock 是通过 ``state.append_context(...)``
进上下文的，与事件是两条路。事件只喂 UI。

═══ ⚠️ 不要顺手也吃掉别的事件类型 ═══

``on_reply`` 流里还有 ``ReplyEndEvent``（回复守卫用它触发重说）、
``ToolCallBlock``、``TextBlock*`` 等。本中间件**只**匹配
:class:`~agentscope.event.HintBlockEvent`，其余一律 ``yield`` 原样放行 ——
写成「白名单放行」的话，框架将来新增一种事件类型就会被静默吞掉，
而症状是「某个新功能在前端永远不出现」，极难定位。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Callable
from typing import Any

from agentscope.event import HintBlockEvent
from agentscope.middleware import MiddlewareBase

#: 本模块的日志器。
logger = logging.getLogger(__name__)


class HintSuppressionMiddleware(MiddlewareBase):
    """把 ``HintBlockEvent`` 挡在用户界面之外。

    ⚠️ **不实现** ``on_system_prompt`` / ``on_model_call`` / ``on_acting``
    等钩子：它只关心「事件往哪走」，不关心模型看到什么。少实现一个钩子
    就少一处可能改变行为的地方。

    ⚠️ 与 :class:`~src.orchestration.reply_guard.ReplyGuardMiddleware` 的
    关系：**互不干扰**。回复守卫自己也会往上下文里塞 HintBlock
    （重说指令，见其 ``_request_retry``），但它是直接
    ``agent.state.append_context(...)``，**不 yield 事件** —— 所以本中间件
    根本看不到它，也就不存在「把守卫的纠正指令吃掉」这回事。
    这一点是**必须**成立的：若哪天有人把守卫改成 yield 事件，
    纠正指令会被这里吞掉，重说机制会静默失效。
    """

    def __init__(self, *, enabled: bool = True) -> None:
        """初始化。

        Args:
            enabled (`bool`): 关掉时**完全透明**（所有事件原样放行）。
                留这个开关是为了排障：怀疑「某个块本该显示却没了」时，
                可以把它关掉对比，而不必改代码。
        """
        self._enabled = enabled
        #: 被吃掉的事件计数。⚠️ 只用于日志与测试断言，不对外暴露成指标
        #: —— 它的量级完全由框架内部行为决定，做成监控指标会误导容量规划。
        self._suppressed = 0

    @property
    def suppressed(self) -> int:
        """已被拦下的事件数（进程内累计）。"""
        return self._suppressed

    async def on_reply(
        self,
        agent: Any,
        input_kwargs: dict[str, Any],
        next_handler: Callable[..., AsyncGenerator[Any, None]],
    ) -> AsyncGenerator[Any, None]:
        """过滤回复流里的事件。

        Args:
            agent (`Any`): 正在回复的 agent。
            input_kwargs (`dict[str, Any]`): 钩子入参（本中间件不使用）。
            next_handler (`Callable[..., AsyncGenerator]`): 链上的下一个
                处理者。

        Yields:
            `Any`: 除 ``HintBlockEvent`` 之外的一切事件，原样放行。
        """
        if not self._enabled:
            async for event in next_handler(**input_kwargs):
                yield event
            return

        async for event in next_handler(**input_kwargs):
            if isinstance(event, HintBlockEvent):
                self._suppressed += 1
                # ⚠️ INFO 级而不是 DEBUG：这条日志是排查「某个块为什么没显示」
                # 的唯一线索。DEBUG 在生产默认不开，等于没有。
                # ⚠️ 只记来源与长度，**不记 hint 正文** —— 正文里可能有
                # 检索到的制度片段、子智能体结论等内容，而它本来就不该
                # 出现在日志里（用户没要求留档，留了反而是数据扩散）。
                logger.info(
                    "已拦下提示块（agent=%s，来源=%s，长度=%d）—— 提示块只给模型看。",
                    getattr(agent, "name", "?"),
                    _safe_source(event),
                    _hint_length(event),
                )
                continue
            yield event


def _safe_source(event: HintBlockEvent) -> str:
    """取事件来源，取不到就返回占位符。

    Args:
        event (`HintBlockEvent`): 事件。

    Returns:
        `str`: 来源字符串；为空时返回 ``"?"``。

    ⚠️ ``source`` 是 ``str | None``（``event/_event.py:310-311``），
    直接进 ``%s`` 会打出 ``None``，读日志时要多绕一下才知道是「没来源」
    还是「来源就是个叫 None 的字符串」。统一成 ``"?"``。
    """
    return event.source or "?"


def _hint_length(event: HintBlockEvent) -> int:
    """算提示块正文的长度（字符数）。

    Args:
        event (`HintBlockEvent`): 事件。

    Returns:
        `int`: 字符数；``hint`` 是多模态块列表时按各块 ``text`` 累加。

    ⚠️ ``hint`` 是 ``str | list[TextBlock | DataBlock]`` 两种形态
    （``event/_event.py:313-314``）。只处理 ``str`` 的话，列表形态会
    在 ``len()`` 上抛类型错误 —— 而这行日志在**事件循环里**，抛出去
    会把这轮回复打断。宁可返回 0 也不抛。
    """
    hint = event.hint
    if isinstance(hint, str):
        return len(hint)
    try:
        return sum(len(getattr(block, "text", "") or "") for block in hint)
    except TypeError:  # pragma: no cover —— 防御性分支，见 docstring
        return 0


__all__ = ["HintSuppressionMiddleware"]
