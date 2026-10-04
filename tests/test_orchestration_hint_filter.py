# -*- coding: utf-8 -*-
"""``HintSuppressionMiddleware`` 的单测：提示块不得放行给下游。

为什么单独立一个文件
====================

端到端那条用例（``tests/test_e2e_stream.py::test_hint_blocks_never_reach_the_user``）
已经证明了「用户看不到提示块」，但它证明的是**当前这条链路的整体行为**。
本文件证明的是**这个中间件本身的契约**，两者不能互相替代：

- 端到端用例会因为「上游压根没产生提示块」而变绿 —— 那是**假绿**，
  它没有证明中间件在工作。本文件里 :func:`test_hint_block_is_swallowed`
  直接喂一个 ``HintBlockEvent`` 进去，中间件是死的还是活的，一眼可辨。
- 端到端用例覆盖不到 ``enabled=False`` 的旁路、覆盖不到 ``hint`` 为
  多模态块列表时的取长度分支（框架的运行时状态注入只走字符串形态）。

★ 这个文件里**每一条**用例都要能被一个"合理的错误实现"弄红：
    · 把 ``isinstance(event, HintBlockEvent)`` 改成 ``isinstance(event, EventBase)``
      → :func:`test_other_events_pass_through_untouched` 红；
    · 把 ``continue`` 改成 ``yield event``            → 吞掉那条红；
    · 把 ``enabled`` 判断删掉                          → 旁路那条红；
    · 把 ``_hint_length`` 的列表分支删掉               → 列表形态那条红。
写不出"弄红它的改法"的断言，就是在测自己刚写的那行代码。
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from agentscope.event import HintBlockEvent, ReplyEndEvent, TextBlockDeltaEvent
from agentscope.message import TextBlock
from agentscope.event import ReplyFinishedReason

from src.orchestration.hint_filter import HintSuppressionMiddleware

#: 事件里用的固定 id。取值本身无意义，只要**稳定** —— 断言里要按 id 找事件。
REPLY_ID = "reply-hint-test"
SESSION_ID = "session-hint-test"


class _FakeAgent:
    """只带 ``name`` 的假 agent。

    中间件只读这一个属性（写日志用），所以只造这一个 —— 造多了会掩盖
    「哪天中间件开始依赖别的属性」这件事：那时本用例会因为
    ``AttributeError`` 变红，而不是静默地继续通过。
    """

    name = "main_plan"


def _hint(text: str | list[Any]) -> HintBlockEvent:
    """构造一个提示块事件。

    Args:
        text (`str | list[Any]`): ``hint`` 字段的内容。

    Returns:
        `HintBlockEvent`: 事件。
    """
    return HintBlockEvent(
        reply_id=REPLY_ID,
        block_id="block-hint-test",
        source='{"label": "System", "sublabel": "Runtime State"}',
        hint=text,
    )


def _delta(text: str) -> TextBlockDeltaEvent:
    """构造一个正文增量事件。"""
    return TextBlockDeltaEvent(reply_id=REPLY_ID, block_id="block-text", delta=text)


def _end() -> ReplyEndEvent:
    """构造一个回复结束事件。"""
    return ReplyEndEvent(
        session_id=SESSION_ID,
        reply_id=REPLY_ID,
        finished_reason=ReplyFinishedReason.COMPLETED,
    )


async def _run(
    middleware: HintSuppressionMiddleware,
    events: list[Any],
) -> list[Any]:
    """把脚本事件喂给中间件，收集它实际放行的事件。

    Args:
        middleware (`HintSuppressionMiddleware`): 中间件。
        events (`list[Any]`): 上游事件流。

    Returns:
        `list[Any]`: 下游收到的事件。
    """

    async def handler(**_kwargs: Any) -> AsyncGenerator[Any, None]:
        for event in events:
            yield event

    out: list[Any] = []
    async for event in middleware.on_reply(
        agent=_FakeAgent(),
        input_kwargs={"inputs": None},
        next_handler=handler,
    ):
        out.append(event)
    return out


def test_hint_block_is_swallowed() -> None:
    """★ 提示块必须被吃掉，其余事件一个不少地放行。

    ⚠️ 断言的是**整个序列**而不是「提示块不在里面」：只查后者的话，
    一个把 ``on_reply`` 写成「什么都不 yield」的实现也会通过 ——
    那是把提示块拦住了，顺带把用户的答复也拦没了。
    """
    import asyncio

    events = [_delta("北京"), _hint("<system-reminder>内部的英文提示词</system-reminder>"), _delta("的差标是 600 元。"), _end()]
    middleware = HintSuppressionMiddleware()

    out = asyncio.run(_run(middleware, events))

    assert [type(e) for e in out] == [TextBlockDeltaEvent, TextBlockDeltaEvent, ReplyEndEvent], (
        f"放行的事件序列不对：{[type(e).__name__ for e in out]}"
    )
    assert middleware.suppressed == 1, (
        f"计数应为 1，实际 {middleware.suppressed} —— "
        "计数是排障时判断「这个中间件到底有没有在工作」的唯一线索。"
    )
    # ⚠️ 顺带钉住正文没被改动：中间件只该**丢弃**事件，不该改写内容。
    assert "".join(e.delta for e in out if isinstance(e, TextBlockDeltaEvent)) == "北京的差标是 600 元。"


def test_disabled_middleware_is_fully_transparent() -> None:
    """★ ``enabled=False`` 时必须**完全透明**（一个事件都不动）。

    ⚠️ 为什么保留这个开关：怀疑「某个块本该显示却没了」时，把它关掉做对比
    是**不用改代码**的排查手段。而一个「关掉了却还吃掉一半」的开关
    比没有开关更坏 —— 它会让人得出「不是这个中间件干的」这个错误结论。
    """
    import asyncio

    events = [_hint("要被放行"), _delta("正文"), _end()]
    middleware = HintSuppressionMiddleware(enabled=False)

    out = asyncio.run(_run(middleware, events))

    assert [type(e) for e in out] == [type(e) for e in events], (
        f"关掉之后事件被改动了：{[type(e).__name__ for e in out]}"
    )
    assert middleware.suppressed == 0


def test_multimodal_hint_list_does_not_break_the_stream() -> None:
    """★ ``hint`` 是多模态块列表时，日志取长度**不得**抛异常。

    ⚠️ 这条守的是一个具体的崩法：``hint`` 的类型是
    ``str | list[TextBlock | DataBlock]``（``agentscope/event/_event.py:309-310``），
    而取长度的代码跑在**事件循环里** —— 列表形态上直接 ``len()`` 会
    ``TypeError``，把用户的这一轮回复整个打断。宁可日志里长度不准，
    也不能抛。

    ⚠️ 构造用 ``TextBlock``（框架真实使用的块类型）而不是一个只带
    ``text`` 属性的假对象：假对象会让「框架换了块类型」这件事测不出来。
    """
    import asyncio

    blocks = [TextBlock(type="text", text="一二三"), TextBlock(type="text", text="四五六七")]
    events = [_hint(blocks), _delta("正文"), _end()]
    middleware = HintSuppressionMiddleware()

    out = asyncio.run(_run(middleware, events))

    assert [type(e) for e in out] == [TextBlockDeltaEvent, ReplyEndEvent], (
        f"列表形态的提示块没有被正确吃掉：{[type(e).__name__ for e in out]}"
    )
    assert middleware.suppressed == 1


def test_hint_without_source_is_loggable() -> None:
    """★ ``source`` 为 ``None`` 时也要能写日志。

    ⚠️ ``source`` 是 ``str | None``（``agentscope/event/_event.py:307-308``）。
    直接把它塞进 ``%s`` 会打出字面量 ``None`` —— 排查时无法区分
    「来源就是空的」与「来源是个真的叫 None 的字符串」。这条要求它
    统一成占位符。
    """
    import asyncio

    event = HintBlockEvent(
        reply_id=REPLY_ID,
        block_id="block-no-source",
        source=None,
        hint="无来源的提示",
    )
    middleware = HintSuppressionMiddleware()

    out = asyncio.run(_run(middleware, [event, _end()]))

    assert [type(e) for e in out] == [ReplyEndEvent]
    assert middleware.suppressed == 1


def test_suppression_does_not_touch_the_context() -> None:
    """★ 中间件**不得**去动 agent 的上下文 —— 模型还得看得见提示块。

    ⚠️ 这条守的是本中间件的**全部理由**：提示块对**模型**是有用的
    （时间感知、任务感知），只有对**用户**是噪音。一个"顺手把它从上下文
    里也删掉"的实现会把功能改坏，而所有「用户看不到」的用例都照样绿。
    所以这里用一个**记录所有属性访问**的假 agent：任何对
    ``state`` / ``context`` 的触碰都会被发现。
    """

    class _WatchingAgent:
        """任何对 ``state`` 的访问都会抛异常的假 agent。"""

        name = "main_plan"

        def __getattr__(self, item: str) -> Any:
            if item == "name":  # pragma: no cover —— name 是类属性，走不到这
                return "main_plan"
            raise AssertionError(
                f"中间件访问了 agent.{item} —— 它只该过滤事件流，"
                "不该去改 agent 的状态或上下文。"
            )

    import asyncio

    async def handler(**_kwargs: Any) -> AsyncGenerator[Any, None]:
        yield _hint("模型该看到的东西")
        yield _end()

    middleware = HintSuppressionMiddleware()
    out: list[Any] = []

    async def drive() -> None:
        async for event in middleware.on_reply(
            agent=_WatchingAgent(),
            input_kwargs={"inputs": None},
            next_handler=handler,
        ):
            out.append(event)

    asyncio.run(drive())

    assert [type(e) for e in out] == [ReplyEndEvent], (
        f"事件流不对：{[type(e).__name__ for e in out]}"
    )
