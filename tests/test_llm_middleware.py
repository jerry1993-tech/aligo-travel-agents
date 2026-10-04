# -*- coding: utf-8 -*-
"""熔断中间件（``src/llm/middleware.py``）的测试。

═══ 这张表守的是什么 ═══

熔断器本身的状态机在 ``tests/test_llm_factory.py``（或它自己的文件）里
穷举过了。本文件只测**这一层**：把熔断器挂到 ``on_model_call`` 钩子上
之后，三个分支的语义有没有搞混。

三个分支里有两个是「写错了也照样跑」的：

1. **短路时不记账**。记的话，每一次被拒绝的请求都把熔断窗口往后推，
   「冷却 30 秒后放一个探针进来」永远不会发生 —— 熔断器**永久断开**。
   这是个只在**下游真的挂了**的时候才触发的 bug，也就是最难撞上的时刻。

2. **异常要原样抛出去**。吞掉它，框架自己的重试与降级机制就全部失效，
   而用户会拿到一个我们编的、看起来像正常回复的东西。

⚠️ 还有一条 API 契约值得单独钉住：``next_handler`` 必须用**关键字**展开调用
（``next_handler(**input_kwargs)``）。写成位置参数传字典，会在框架更深处
报一个与本模块毫无关系的 ``TypeError``。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import pytest
from agentscope.model import ChatResponse, ChatUsage

from src.llm.breaker import CircuitBreaker, CircuitState
from src.llm.middleware import (
    BreakerMiddleware,
    ModelCallTimeout,
    _model_label,
    _usage_tokens,
    build_breaker_middleware,
    build_model_timeout_middleware,
)
from src.observability.metrics import REGISTRY

INPUT_KWARGS: dict[str, Any] = {
    "current_model": object(),
    "messages": [],
    "tools": [],
    "tool_choice": None,
}
"""一次模型调用的参数，形状与 ``agent/_agent.py`` 传下来的一致。"""


# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------
class SpyHandler:
    """一个记录被调用次数的 ``next_handler``。

    ⚠️ 断言「有没有被调用」是这些用例的核心：短路分支与记账分支的区别
    **只能**靠这个计数看出来。

    Attributes:
        calls (`int`): 被调用的次数。
        seen (`list[dict]`): 每次收到的参数。
        raises (`BaseException | None`): 非 ``None`` 时抛它。
        response (`ChatResponse`): 非流式时返回的响应。
        stream_chunks (`tuple[Any, ...] | None`): 非 ``None`` 时改为返回
            一个异步生成器，先吐这些分片。
        stream_boom (`BaseException | None`): 流吐完后抛的异常。

    ⚠️ 流式能力做成``__init__`` 的参数、而不是事后往实例上挂一个
    ``__call__``：**Python 对特殊方法走的是类型查找**，
    往实例上赋 ``handler.__call__ = ...`` 对 ``handler(...)``
    一点影响都没有（连个警告都没有，只会静默走类上那个）。
    这是踩过一次的坑。
    """

    def __init__(
        self,
        *,
        raises: BaseException | None = None,
        stream_chunks: tuple[Any, ...] | None = None,
        stream_boom: BaseException | None = None,
    ) -> None:
        """初始化。

        Args:
            raises (`BaseException | None`): 要抛的异常。
            stream_chunks (`tuple[Any, ...] | None`): 非 ``None`` 时走流式。
            stream_boom (`BaseException | None`): 流尾要抛的异常。
        """
        self.calls = 0
        self.seen: list[dict[str, Any]] = []
        self.raises = raises
        self.response = ChatResponse(content=[], is_last=True)
        self.stream_chunks = stream_chunks
        self.stream_boom = stream_boom

    async def __call__(self, **kwargs: Any) -> Any:
        """被调用。

        Args:
            **kwargs: 框架传来的参数。

        Returns:
            `Any`: 非流式时是 ``ChatResponse``；流式时是**未被迭代**的
                异步生成器（与 ``agentscope/model/_base.py:254-290`` 的形状一致）。

        Raises:
            BaseException: ``raises`` 非 ``None`` 时抛出它。
        """
        self.calls += 1
        self.seen.append(kwargs)
        if self.raises is not None:
            raise self.raises
        if self.stream_chunks is not None:
            return _stream(*self.stream_chunks, boom=self.stream_boom)
        return self.response


def run(middleware: BreakerMiddleware, handler: SpyHandler, **kwargs: Any) -> Any:
    """跑一次钩子。

    Args:
        middleware (`BreakerMiddleware`): 中间件。
        handler (`SpyHandler`): 下一个处理者。
        **kwargs: 覆盖 ``INPUT_KWARGS`` 的字段。

    Returns:
        `Any`: 钩子返回值。
    """
    payload = {**INPUT_KWARGS, **kwargs}
    return asyncio.run(middleware.on_model_call(object(), payload, handler))


def breaker(**kwargs: Any) -> CircuitBreaker:
    """构造一个熔断器。

    Args:
        **kwargs: 覆盖默认参数。

    Returns:
        `CircuitBreaker`: 熔断器。
    """
    params: dict[str, Any] = {"failure_threshold": 3, "recovery_seconds": 30.0, "name": "test"}
    params.update(kwargs)
    return CircuitBreaker(**params)


# ---------------------------------------------------------------------------
# 一、正常路径
# ---------------------------------------------------------------------------
def test_a_successful_call_is_forwarded_verbatim() -> None:
    """★★ 正常调用把响应**原样**返回，且参数**原封不动**传下去。

    ⚠️ 参数原样这一条比看起来重要：中间件在链上，它「顺手」补个字段或
    改个默认值，链尾那个真正的模型调用收到的就不是上游本意了 ——
    而这类改动在中间件里极难被注意到，因为中间件看起来只该做保护。

    ⚠️ 用关键字展开（``**input_kwargs``）是本模块唯一的正确写法。
    写成 ``next_handler(input_kwargs)`` 会把整个字典当成一个位置参数，
    报出来的 ``TypeError`` 在框架更深处，与本模块毫无关系。
    """
    h = SpyHandler()
    result = run(BreakerMiddleware(breaker=breaker()), h)

    assert result is h.response
    assert h.calls == 1
    assert h.seen[0] == INPUT_KWARGS


def test_a_successful_call_resets_the_failure_streak() -> None:
    """★ 成功一次就把连续失败计数清零。

    ⚠️ 不清零的话，失败计数会跨会话累积 —— 一个稳定运行的系统会因为
    「历史上失败过 3 次」而熔断，而当前下游完全健康。
    """
    br = breaker()
    h = SpyHandler()
    mw = BreakerMiddleware(breaker=br)

    h.raises = ConnectionError("超时")
    for _ in range(2):
        with pytest.raises(ConnectionError):
            run(mw, h)
    assert br.snapshot()["consecutive_failures"] == 2

    h.raises = None
    run(mw, h)

    assert br.snapshot()["consecutive_failures"] == 0
    assert br.state is CircuitState.CLOSED


def test_an_empty_response_is_still_a_success() -> None:
    """★★ 「成功」的判据是**没有抛异常**，不是「回复内容好看」。

    ⚠️ 这条守的是一个很容易顺手加上的错误判据：内容质量不是熔断器能
    判断的，把它算进去会让一个「只会说车轱辘话但服务完全正常」的模型
    被熔断 —— 而熔断之后连车轱辘话都没有了。
    """
    br = breaker()
    h = SpyHandler()
    h.response = ChatResponse(content=[], is_last=True)

    run(BreakerMiddleware(breaker=br), h)

    assert br.state is CircuitState.CLOSED
    assert br.snapshot()["consecutive_failures"] == 0


# ---------------------------------------------------------------------------
# 二、异常路径
# ---------------------------------------------------------------------------
def test_a_failure_is_recorded_and_re_raised() -> None:
    """★★★ 模型抛异常时记一次失败，然后**原样抛出去**。

    ⚠️ 两条缺一不可，而且它们服务于**不同**的目的：

    - **记账**：不记的话熔断器永远数不到失败，退化成一个纯粹的中转层。
    - **上抛**：吞掉的话框架自己的重试与降级机制全部失效，用户会拿到一个
      我们编的、看起来像正常回复的东西 —— 那比一个诚实的错误糟糕得多，
      因为他会照着它继续操作。

    ⚠️ 断言异常**类型**，不是「抛了异常」。包一层自定义异常同样会让
    框架按未知错误处理，而它对上游超时与配置错误本来有不同策略。
    """
    br = breaker()
    h = SpyHandler(raises=ConnectionError("上游超时"))

    with pytest.raises(ConnectionError, match="上游超时"):
        run(BreakerMiddleware(breaker=br), h)

    assert br.snapshot()["consecutive_failures"] == 1
    assert br.snapshot()["total_failures"] == 1


def test_failures_add_up_to_an_open_circuit() -> None:
    """★★ 连续失败到阈值后熔断器开路，之后的调用**不再碰真实模型**。

    ⚠️ 这条把「记账」与「短路」接起来验证。分开测的话，一个「记了账但
    从不开路」或者「开了路但仍然调用模型」的实现都能通过。
    """
    br = breaker(failure_threshold=3)
    h = SpyHandler(raises=ConnectionError("挂了"))
    mw = BreakerMiddleware(breaker=br)

    for _ in range(3):
        with pytest.raises(ConnectionError):
            run(mw, h)
    assert h.calls == 3

    result = run(mw, h)
    assert h.calls == 3, "熔断器已经开路，真实模型仍然被调用了"
    assert isinstance(result, ChatResponse)


def test_cancellation_is_not_recorded_as_a_model_failure() -> None:
    """★★★ 用户取消**不算**下游故障。

    ⚠️ 这条守的是一个很容易顺手写错的地方：``except Exception`` 捕不到
    ``CancelledError``（它继承 ``BaseException``），所以中间件天然不会
    把它记成失败。但如果哪天有人为了「更保险」改成
    ``except BaseException``，用户每点一次「停止」就会给熔断器记一笔失败
    —— 几个手快的用户就能把一个完全健康的下游熔断掉。

    ⚠️ 同时断言它**继续向上传播**：吞掉 ``CancelledError`` 的后果是
    「点了停止但没停下来」。
    """
    br = breaker()
    h = SpyHandler(raises=asyncio.CancelledError())

    with pytest.raises(asyncio.CancelledError):
        run(BreakerMiddleware(breaker=br), h)

    assert br.snapshot()["consecutive_failures"] == 0, "用户取消被记成了下游故障"
    assert br.state is CircuitState.CLOSED


# ---------------------------------------------------------------------------
# 三、短路路径
# ---------------------------------------------------------------------------
def test_an_open_circuit_short_circuits_before_the_model() -> None:
    """★★★ 开路的熔断器**不调用** ``next_handler``。

    ⚠️ 这才是熔断的意义。只返回一个错误响应但仍然调用模型的话，下游
    已经挂了还要每次请求都去撞一遍，熔断器就退化成了一个错误格式化器。
    """
    br = breaker(failure_threshold=1)
    h = SpyHandler(raises=ConnectionError("挂了"))
    mw = BreakerMiddleware(breaker=br)

    with pytest.raises(ConnectionError):
        run(mw, h)
    assert br.state is CircuitState.OPEN

    h.raises = None  # 下游恢复了，但冷却期还没过
    run(mw, h)

    assert h.calls == 1, "短路分支仍然调用了真实模型"


def test_the_short_circuit_response_is_last() -> None:
    """★★ 短路响应的 ``is_last`` 是 ``True``，且内容非空。

    ⚠️ 这条是**防御性**的，不是承重的 —— 写这条注释的人踩过一次，
    值得说清楚：框架只在**流式**分支上读 ``is_last``
    （``agentscope/agent/_agent.py:1746-1748``），而本中间件返回的是裸
    ``ChatResponse``，走的是 ``isinstance(res, ChatResponse)`` 那一支
    （``:1759-1760``），在那里 ``is_last`` **根本不会被读**。
    所以「漏掉它整轮回复会卡住」这个说法在**当前**这条路径上是错的。

    仍然断言它，是因为在**流式**分支上它确实必需：框架拿不到
    ``is_last=True`` 的分片会 ``raise RuntimeError``（``:1790-1796``）。
    哪天本中间件的返回值换了形状、或短路响应被复用，这条就是那条防线。

    ⚠️ 同时断言内容非空：一个空的短路响应会让用户看到一片空白，
    与「卡住」在观感上没有区别。
    """
    br = breaker(failure_threshold=1)
    h = SpyHandler(raises=ConnectionError("挂了"))
    mw = BreakerMiddleware(breaker=br)

    with pytest.raises(ConnectionError):
        run(mw, h)

    result = run(mw, h)

    assert isinstance(result, ChatResponse)
    assert result.is_last is True, "少了 is_last，整轮回复会卡住不结束"
    assert result.content, "短路响应是空的，用户会看到一片空白"


def test_the_short_circuit_message_tells_the_user_what_to_check() -> None:
    """★★ 短路文案要提醒用户**确认刚才的操作有没有生效**。

    ⚠️ 这是产品判断，不是文案偏好：用户看到报错时的第一反应是
    「那我刚才提交的申请算提交了吗」。不回答这个问题，他要么重复提交，
    要么去问客服 —— 两种都比在提示里加一句话贵得多。

    ⚠️ 同时断言它**不含**内部术语（熔断/失败率/下游/上游）。用户无法
    处理这些信息，而它们出现在提示里只会让他觉得系统出了大问题。
    """
    br = breaker(failure_threshold=1)
    h = SpyHandler(raises=ConnectionError("挂了"))
    mw = BreakerMiddleware(breaker=br)

    with pytest.raises(ConnectionError):
        run(mw, h)
    result = run(mw, h)

    text = "".join(getattr(block, "text", "") for block in result.content)
    assert "重试" in text or "再试" in text
    assert "订单" in text, "没有提醒用户确认刚才的操作是否生效"
    for jargon in ("熔断", "失败率", "下游", "上游", "circuit"):
        assert jargon not in text, f"短路文案里出现了内部术语 {jargon!r}"


def test_the_short_circuit_does_not_record_a_failure() -> None:
    """★★★ 被拒绝的请求**不再记账** —— 否则熔断器永久断开。

    ⚠️ 这条是本文件里最容易写错、后果最严重的一条。短路时记一笔失败的话：

        每次被拒绝 → 连续失败 +1 → 冷却窗口被重置
        ⇒ 「冷却 30 秒后放一个探针进来」**永远不会发生**
        ⇒ 熔断器再也不会恢复，即使下游早就好了

    ⚠️ 而它只在**下游真的挂了**的时候才触发 —— 也就是最难在测试里撞上的
    时刻。这条用例存在的唯一价值就是把它提前撞出来。

    ⚠️ 断言方式：短路若干次之后，连续失败数**没有变化**，
    且总拒绝数在涨（说明确实走了短路分支，不是用例写错了路径）。
    """
    br = breaker(failure_threshold=1)
    h = SpyHandler(raises=ConnectionError("挂了"))
    mw = BreakerMiddleware(breaker=br)

    with pytest.raises(ConnectionError):
        run(mw, h)

    before = br.snapshot()
    for _ in range(10):
        run(mw, h)
    after = br.snapshot()

    assert after["consecutive_failures"] == before["consecutive_failures"], (
        "短路时记了账：每次被拒绝都会重置冷却窗口，熔断器将永远不会恢复"
    )
    assert after["total_rejections"] > before["total_rejections"], "用例没走到短路分支"
    assert br.state is CircuitState.OPEN, "熔断器自己恢复了？冷却期不该被短路重置"


# ---------------------------------------------------------------------------
# 三·B、流式路径（本项目**生产默认**的那条）
# ---------------------------------------------------------------------------
async def _stream(*chunks: Any, boom: BaseException | None = None) -> Any:
    """造一个异步生成器：先吐完 ``chunks``，再（可选）抛 ``boom``。

    ⚠️ 用异步生成器而不是返回一个列表，是因为要紧贴框架的真实形状：
    ``ChatModelBase.__call__`` 在 ``stream=True`` 时返回的就是一个
    **未被迭代**的异步生成器（``agentscope/model/_base.py:254-290``），
    而 ``execute_chain`` 是 ``return await current_model(...)``
    （``agentscope/agent/_agent.py:3339-3343``）—— 也就是说，
    ``await`` 它只负责**建出**生成器，一点 I/O 都不做。

    Args:
        *chunks: 先吐出的分片。
        boom (`BaseException | None`): 吐完后要抛的异常。

    Yields:
        `Any`: 分片。
    """
    for chunk in chunks:
        yield chunk
    if boom is not None:
        raise boom


def streaming(handler: SpyHandler, *chunks: Any, boom: BaseException | None = None) -> SpyHandler:
    """把 ``SpyHandler`` 变成「返回一个异步生成器」的形态。

    Args:
        handler (`SpyHandler`): 替身。
        *chunks: 生成器先吐的分片。
        boom (`BaseException | None`): 吐完后要抛的异常。

    Returns:
        `SpyHandler`: 同一个替身，但返回值换成了生成器。
    """
    handler.stream_chunks = chunks
    handler.stream_boom = boom
    return handler


async def _drain(result: Any) -> list[Any]:
    """把 ``on_model_call`` 的返回值消费干净。

    ⚠️ 必须消费 —— 流式路径上记账发生在**迭代**的时候，
    只 ``await`` 不迭代等于什么都没发生（那正是当年那个 bug 的成因）。

    Args:
        result (`Any`): ``on_model_call`` 的返回值。

    Returns:
        `list[Any]`: 收到的分片。
    """
    return [chunk async for chunk in result]


def call_stream(middleware: BreakerMiddleware, handler: SpyHandler) -> Any:
    """跑一次钩子并返回**未消费**的返回值（供调用方自己决定怎么迭代）。

    Args:
        middleware (`BreakerMiddleware`): 中间件。
        handler (`SpyHandler`): 替身。

    Returns:
        `Any`: 钩子返回值。
    """
    return asyncio.run(middleware.on_model_call(object(), dict(INPUT_KWARGS), handler))


def test_a_failed_stream_is_recorded_as_a_failure() -> None:
    """★★★ 流**在消费过程中**抛异常，熔断器要记一笔失败。

    ⚠️⚠️ 这是本文件最重要的一条，因为它守的是一个**已经在生产代码里
    真实存在过**的 bug，而且它在非流式路径上完全看不出来。

    背景：流式（``stream=True``）是本项目的生产默认。在这个形状下，
    「调用」与「拿到结果」是两次操作 —— ``await next_handler(...)``
    只是**建出**一个异步生成器（不做 I/O），真正的网络请求发生在框架
    迭代它的时候（``agentscope/agent/_agent.py:1744-1757``）。所以
    「``await`` 没抛异常」**不等于**「模型调用成功」：
    上游挂掉、超时、鉴权失败全都发生在中间件 ``try`` 块**之外**。

    ⚠️ 修之前实测到的症状：用真实调用约定连打 5 次「迭代到一半抛
    ``ConnectionError``」，熔断器仍然是 ``CLOSED``、
    ``consecutive_failures`` 是 **0**；同样 5 次非流式失败会让它 ``OPEN``。
    也就是说**熔断器在生产路径上从来没生效过**。

    ⚠️ 断言方式：必须真的把流**迭代完**（``_drain``）——
    只 ``await`` 不迭代的话，一个「建出生成器就记成功」的错误实现
    照样能通过。
    """
    br = breaker(failure_threshold=3)
    h = SpyHandler()
    mw = BreakerMiddleware(breaker=br)

    for _ in range(3):
        h2 = streaming(SpyHandler(), "第一片", boom=ConnectionError("流中断了"))
        with pytest.raises(ConnectionError, match="流中断了"):
            asyncio.run(_drain(call_stream(mw, h2)))

    assert br.state is CircuitState.OPEN, (
        "流式失败没有被记账，熔断器永远开不了 —— 而流式是生产默认路径"
    )
    assert br.snapshot()["consecutive_failures"] == 3


def test_a_stream_that_completes_records_a_success() -> None:
    """★★ 流**完整消费完**才记成功，且分片原样透传。

    ⚠️ 分片原样这一条与正常路径同理：中间件「顺手」改一下分片内容，
    链下游收到的东西就不是模型本意了，而这类改动极难被注意到。

    ⚠️ 同时断言**消费完之后**才记账：正常流里 ``total_calls`` 与
    连续失败计数在迭代前就该是「还没结论」的状态。
    """
    br = breaker()
    h = streaming(SpyHandler(), "第一片", "第二片", "第三片")
    result = call_stream(BreakerMiddleware(breaker=br), h)

    assert br.snapshot()["consecutive_failures"] == 0
    assert asyncio.run(_drain(result)) == ["第一片", "第二片", "第三片"]
    assert br.snapshot()["consecutive_failures"] == 0
    assert br.snapshot()["total_failures"] == 0


def test_a_completed_stream_clears_the_failure_streak() -> None:
    """★★ 一次成功的流同样把连续失败清零 —— 不能只清非流式那一半。

    ⚠️ 清零与非流式路径分开测：两条路的记账点不同（一个在 ``await`` 后、
    一个在迭代完后），实现时很容易只改了一条。
    """
    br = breaker()
    mw = BreakerMiddleware(breaker=br)

    for _ in range(2):
        with pytest.raises(ConnectionError):
            asyncio.run(_drain(call_stream(mw, streaming(SpyHandler(), boom=ConnectionError("挂了")))))
    assert br.snapshot()["consecutive_failures"] == 2

    asyncio.run(_drain(call_stream(mw, streaming(SpyHandler(), "好了"))))

    assert br.snapshot()["consecutive_failures"] == 0
    assert br.state is CircuitState.CLOSED


def test_a_cancellation_inside_a_stream_is_not_a_model_failure() -> None:
    """★★★ 流内被取消**不算**下游故障。

    ⚠️ ``CancelledError`` 继承 ``BaseException``，所以一条
    ``except Exception`` 天然捕不到它 —— 但流式路径是新加的代码，
    改写成 ``except BaseException``（为了「顺手把 GeneratorExit 也收掉」）
    是很容易发生的一步。那样用户每点一次「停止」就记一笔失败。

    ⚠️ 同时断言它继续向上传播（``asyncio.CancelledError``），
    吞掉的后果是「点了停止没停下来」。
    """
    br = breaker()
    mw = BreakerMiddleware(breaker=br)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_drain(call_stream(mw, streaming(SpyHandler(), "半片", boom=asyncio.CancelledError()))))

    assert br.snapshot()["consecutive_failures"] == 0, "用户取消被记成了下游故障"


def test_an_abandoned_stream_records_neither_outcome() -> None:
    """★★ 流被**中途放弃**时两边都不记。

    ⚠️ 把它算成「成功」会让一个「连接建得上、但一读就断」的下游
    永远熔不断：每次请求都建出一个生成器、记一次成功，
    连续失败计数被不断清零。这正是流式路径最阴险的失效模式。

    ⚠️ 也不该记失败 —— 消费方提前 ``break`` 是正常行为
    （比如界面切走了），把它算成下游故障会误熔。

    ⚠️ 断言的是「连续失败数没变」，且这不是因为压根没记账 ——
    所以同时确认熔断器仍然是 ``CLOSED``（没被推过去）。
    """
    br = breaker(failure_threshold=2)
    mw = BreakerMiddleware(breaker=br)

    for _ in range(5):
        result = call_stream(mw, streaming(SpyHandler(), "第一片", "第二片"))
        first = asyncio.run(_drain_one(result))
        assert first == "第一片"
        # 拿到第一片就走人，剩下的不消费。

    assert br.snapshot()["consecutive_failures"] == 0
    assert br.snapshot()["total_failures"] == 0
    assert br.state is CircuitState.CLOSED


async def _drain_one(result: Any) -> Any:
    """只取异步生成器的第一个分片。

    Args:
        result (`Any`): 异步生成器。

    Returns:
        `Any`: 第一个分片。
    """
    async for chunk in result:
        return chunk
    return None


def test_a_short_circuit_never_builds_a_stream() -> None:
    """★★ 熔断打开时短路，**不**建生成器 —— 短路必须发生在建流之前。

    ⚠️ 这条守的是新加的流式分支有没有把短路顺序搞反：若先
    ``await next_handler(...)`` 再判熔断，那「短路」就没有省掉任何东西
    —— 而省掉那次调用正是熔断的全部意义。
    """
    br = breaker(failure_threshold=1)
    mw = BreakerMiddleware(breaker=br)
    h = streaming(SpyHandler(), boom=ConnectionError("挂了"))

    with pytest.raises(ConnectionError):
        asyncio.run(_drain(call_stream(mw, h)))
    assert h.calls == 1

    h2 = streaming(SpyHandler(), "不该被拿到")
    result = call_stream(mw, h2)

    assert h2.calls == 0, "短路分支仍然建了流"
    assert isinstance(result, ChatResponse), "短路应返回 ChatResponse，不是生成器"


class _HangingHandler:
    """一个**永远不返回**的 ``next_handler``（模拟上游把连接挂住）。

    Attributes:
        calls (`int`): 被调用次数。
    """

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, **kwargs: Any) -> Any:
        """挂住，直到被取消。

        Args:
            **kwargs: 框架传来的参数。

        Returns:
            `Any`: 永远不会走到这里。

        Raises:
            asyncio.CancelledError: 被取消时（内层不吞取消的普通实现）。
        """
        self.calls += 1
        await asyncio.Event().wait()  # 永远等下去
        raise AssertionError("不可达")


class _CancellationSwallowingHandler:
    """被取消时**返回一个正常响应**的替身 —— 复刻框架自己的行为。

    ⚠️ 框架的模型层就是这么写的（``agentscope/model/_base.py:219-224``：捕
    ``CancelledError`` 并返回一个 ``finished_reason=INTERRUPTED`` 的响应）。
    用它来区分「靠取消传播判定超时」（``wait_for``，会被骗过）与
    「按时钟判定」（本实现，不依赖下层行为）。

    Attributes:
        swallowed (`bool`): 是否真的经历过一次被吞掉的取消。
    """

    def __init__(self) -> None:
        self.swallowed = False

    async def __call__(self, **kwargs: Any) -> Any:
        """挂住；被取消时吞掉取消并返回一个空响应。

        Args:
            **kwargs: 框架传来的参数。

        Returns:
            `Any`: 被取消时的空响应。
        """
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.swallowed = True
            return ChatResponse(content=[], is_last=True)
        raise AssertionError("不可达")


class _SlowHandler:
    """慢，但会返回的替身。

    Attributes:
        delay (`float`): 每次调用的耗时（秒）。
        calls (`int`): 被调用次数。
        response (`ChatResponse`): 返回值。
    """

    def __init__(self, *, delay: float) -> None:
        """初始化。

        Args:
            delay (`float`): 每次调用的耗时（秒）。
        """
        self.delay = delay
        self.calls = 0
        self.response = ChatResponse(content=[], is_last=True)

    async def __call__(self, **kwargs: Any) -> Any:
        """睡 ``delay`` 之后返回。

        Args:
            **kwargs: 框架传来的参数。

        Returns:
            `ChatResponse`: 固定响应。
        """
        self.calls += 1
        await asyncio.sleep(self.delay)
        return self.response


class _StallingStreamHandler:
    """返回一个「吐几个分片之后卡住」的异步生成器。

    Attributes:
        delivered (`list`): 已经吐出去的分片（用于断言卡住之前的部分照常送达）。
        closed (`bool`): 底层生成器有没有被关闭（``finally`` 里置真）。
    """

    def __init__(
        self,
        *,
        first_chunks: tuple[Any, ...] = (),
        stall_after: int | None = None,
        gap: float = 0.0,
    ) -> None:
        """初始化。

        Args:
            first_chunks (`tuple`): 先吐出的分片。
            stall_after (`int | None`): 吐到第几个之后卡住；``None`` 表示吐完
                就正常结束（用于「慢但活着」的用例）。
            gap (`float`): 分片之间的间隔（秒）。
        """
        self.delivered: list[Any] = []
        self.closed = False
        self._first_chunks = first_chunks
        self._stall_after = stall_after
        self._gap = gap

    async def __call__(self, **kwargs: Any) -> Any:
        """返回（未被迭代的）异步生成器 —— 与框架的真实形状一致。

        Args:
            **kwargs: 框架传来的参数。

        Returns:
            `Any`: 异步生成器。
        """
        return self._stream()

    async def _stream(self) -> Any:
        """先吐分片，再按需卡住。

        Yields:
            `Any`: 分片。
        """
        try:
            for index, chunk in enumerate(self._first_chunks, start=1):
                if self._gap:
                    await asyncio.sleep(self._gap)
                self.delivered.append(chunk)
                yield chunk
                if self._stall_after is not None and index >= self._stall_after:
                    await asyncio.Event().wait()  # 卡在这里不再吐
        finally:
            self.closed = True


class _Bound:
    """把一个中间件绑到它的 ``next_handler`` 上，形成链上的一节。"""

    def __init__(self, middleware: Any, next_handler: Any) -> None:
        """初始化。

        Args:
            middleware (`Any`): 中间件。
            next_handler (`Any`): 链上的下一个处理者。
        """
        self._mw = middleware
        self._next = next_handler

    async def __call__(self, **kwargs: Any) -> Any:
        """把调用交给中间件。

        Args:
            **kwargs: 框架传来的参数。

        Returns:
            `Any`: 中间件返回值。
        """
        return await self._mw.on_model_call(object(), kwargs, self._next)


def _chain(middlewares: list[Any], handler: Any) -> Any:
    """把中间件按「下标 0 最外层」串成一条链（与框架的装配顺序一致）。

    Args:
        middlewares (`list`): 中间件（下标 0 最外层）。
        handler (`Any`): 链尾的真实调用。

    Returns:
        `Any`: 链头，``await chain(**kwargs)`` 即可。
    """
    node = handler
    for middleware in reversed(middlewares):
        node = _Bound(middleware, node)
    return node


async def _run_chain(middlewares: list[Any], handler: Any) -> Any:
    """跑一次链，返回 ``on_model_call`` 的返回值（不消费流）。

    Args:
        middlewares (`list`): 中间件（下标 0 最外层）。
        handler (`Any`): 链尾的真实调用。

    Returns:
        `Any`: 链头返回值。
    """
    return await _chain(middlewares, handler)(**dict(INPUT_KWARGS))


# ---------------------------------------------------------------------------
# 四、开关
# ---------------------------------------------------------------------------
def test_disabled_middleware_is_completely_transparent() -> None:
    """★★ ``enabled=False`` 时中间件**完全透明**：不查熔断、不记账、直接透传。

    ⚠️ 这个开关是排障用的：怀疑「回复变慢/变差是不是熔断引起的」时，
    关掉它做对照。所以关掉之后必须连**熔断判定**都不做 —— 只是不记账
    但仍然拦截，对照实验就没有意义了（慢的可能是拦截本身）。

    ⚠️ 断言即便是开路状态也照样放行。这一条把「透明」这个语义钉死。
    """
    br = breaker(failure_threshold=1)
    mw = BreakerMiddleware(breaker=br, enabled=False)

    h = SpyHandler(raises=ConnectionError("挂了"))
    for _ in range(5):
        with pytest.raises(ConnectionError):
            run(mw, h)

    assert br.snapshot()["total_calls"] == 0, "关掉之后仍然在记账"
    assert br.state is CircuitState.CLOSED

    h.raises = None
    result = run(mw, h)
    assert result is h.response
    assert h.calls == 6


def test_a_disabled_middleware_warns_about_nothing_it_cannot_see() -> None:
    """★ 关掉时短路日志不出现（它压根不走那条分支）。"""
    br = breaker(failure_threshold=1)
    br_state_before = br.state
    mw = BreakerMiddleware(breaker=br, enabled=False)

    run(mw, SpyHandler())

    assert br.state is br_state_before


# ---------------------------------------------------------------------------
# 五、日志
# ---------------------------------------------------------------------------
def test_the_short_circuit_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """★★ 短路**要留日志**。

    ⚠️ 短路对用户表现为一句「模型服务暂时不可用」，与一次普通的模型报错
    长得一样。不留日志的话，「今天下午系统为什么一直提示不可用」就查不出
    原因 —— 而熔断器的 `total_opens` 只说明它开过，不说明它拒了多少次。
    """
    br = breaker(failure_threshold=1)
    h = SpyHandler(raises=ConnectionError("挂了"))
    mw = BreakerMiddleware(breaker=br)

    with pytest.raises(ConnectionError):
        run(mw, h)

    with caplog.at_level(logging.WARNING):
        run(mw, h)

    assert any("熔断" in record.message for record in caplog.records), caplog.text


# ---------------------------------------------------------------------------
# 六、构造
# ---------------------------------------------------------------------------
def test_the_builder_requires_a_breaker() -> None:
    """★★ 构造器**强制**显式传入熔断器。

    ⚠️ 这是刻意的：熔断器必须进程内唯一（见 ``src/llm/factory.py`` 的
    ``get_breaker``）。给参数一个默认值，就等于允许「顺手 new 了一个」——
    而每个实例各自失败 N 次才熔断，N 就是并发数，保护形同虚设。

    ⚠️ 断言的是 ``TypeError``（缺必填关键字参数），不是别的异常。
    """
    with pytest.raises(TypeError):
        build_breaker_middleware()  # type: ignore[call-arg]


def test_the_builder_passes_the_switch_through() -> None:
    """⚠️ ``enabled`` 透传到中间件（不是被构造器吞掉）。"""
    br = breaker()
    assert build_breaker_middleware(breaker=br, enabled=False)._enabled is False
    assert build_breaker_middleware(breaker=br)._enabled is True


def test_a_model_call_has_a_deadline() -> None:
    """★★★ 卡住不返回的模型调用必须在预算内失败，而不是一直挂着。

    ⚠️ 这条守的是熔断器**管不到**的那一半：熔断器要「失败先发生」，
    而卡住的调用永远不抛异常 —— 没有这个截止时间，它会一直挂着，
    用户在页面上看到的是转圈转到天荒地老。

    ⚠️ 判据里带**墙钟上限**：一个「抛了超时异常但其实是等上游自己超时」
    的实现能骗过 ``pytest.raises``，骗不过 ``elapsed``。
    """
    h = _HangingHandler()

    started = time.monotonic()
    with pytest.raises(ModelCallTimeout):
        asyncio.run(_run_chain([build_model_timeout_middleware(timeout=0.05)], h))
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"预算 0.05s，却等了 {elapsed:.2f}s —— 截止时间没有生效"
    assert h.calls == 1


def test_the_timeout_is_also_a_plain_timeout_error() -> None:
    """``ModelCallTimeout`` 必须能被 ``except TimeoutError`` 接住。

    ⚠️ 与 ``VectorSearchTimeout`` 同一个理由：上层（框架的降级链路、
    我们的兜底）用宽口径捕获，新调用点未必要认识本模块的类型。
    """
    h = _HangingHandler()

    with pytest.raises(TimeoutError):
        asyncio.run(_run_chain([build_model_timeout_middleware(timeout=0.02)], h))


def test_a_swallowed_cancellation_still_counts_as_a_timeout() -> None:
    """★★★ 即便下层**吞掉取消**，超时判定也照样成立。

    ⚠️⚠️ 这是本组用例里最重要的一条，因为它挡的是一个「用
    ``asyncio.wait_for`` 写就必然踩中」的陷阱：

    框架的模型层**故意**吞掉 ``CancelledError``（把它转成一次优雅的
    「被打断」—— 非流式在 ``agentscope/model/_base.py:219-224`` 返回一个
    ``INTERRUPTED`` 空响应，流式在 ``:283-288`` 转成末尾分片）。
    而 ``wait_for`` 判定超时的唯一依据就是「取消有没有传播出来」——
    被吞掉之后它看到的是「正常返回」，超时**永远不成立**：
    护栏写在代码里，一次都不会触发。

    所以本中间件把调用放进独立任务、按**时钟**判定。这个用例用一个
    「被取消时返回一个正常响应」的替身把两者区分开：用 ``wait_for``
    的实现会**返回那个响应**，而正确实现必须抛 ``ModelCallTimeout``。
    """
    h = _CancellationSwallowingHandler()

    with pytest.raises(ModelCallTimeout):
        asyncio.run(_run_chain([build_model_timeout_middleware(timeout=0.05)], h))

    assert h.swallowed, "用例没有走到「吞掉取消」那条分支，测的不是它"


def test_a_hung_stream_between_chunks_fails_within_the_budget() -> None:
    """★★★ 流式响应在**分片之间**卡住时，同样要在预算内失败。

    ⚠️ 流式是本项目的生产默认，而它的全部 I/O 都在**迭代**里发生：
    ``await next_handler(...)`` 只是建出生成器（框架 ``agentscope/model/_base.py:254-290``），
    所以「建流没抛异常」什么都不能说明。只箍建流那一步的中间件，
    对生产路径等于不存在。
    """
    h = _StallingStreamHandler(first_chunks=("第一片",), stall_after=1)

    async def scenario() -> list[Any]:
        result = await _run_chain([build_model_timeout_middleware(timeout=0.05)], h)
        return await _drain(result)

    started = time.monotonic()
    with pytest.raises(ModelCallTimeout):
        asyncio.run(scenario())
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"预算 0.05s，却等了 {elapsed:.2f}s"
    assert h.delivered == ["第一片"], "卡住之前的分片应当照常送达"


def test_a_slow_but_alive_stream_is_not_cut_off() -> None:
    """★★ 只要分片还在持续到达，慢的流**不该**被掐断。

    ⚠️ 这条钉住超时的语义是「分片间**不活动**超时」，不是「整个流的总时长」：
    总时长语义会把一个正常的、只是话多的回答拦腰砍断 —— 那是一种
    「护栏把功能搞坏了」的故障，而且只在回答长的时候出现，极难归因。
    """
    chunks = ("一", "二", "三", "四", "五", "六", "七", "八")
    h = _StallingStreamHandler(
        first_chunks=chunks,
        # ⚠️ 参数刻意留出**大**余量：预算 0.3s，分片间隔 0.05s（6 倍余量），
        # 总时长约 0.4s > 预算。间隔取得太贴近预算的话，本机负载一高
        # （这是一台 4 vCPU 的开发机）就会偶发假红 —— 而一条会偶发假红的
        # 用例，最后一定会被人加个 ``skip`` 了事。
        gap=0.05,
    )

    async def scenario() -> list[Any]:
        result = await _run_chain([build_model_timeout_middleware(timeout=0.3)], h)
        return await _drain(result)

    assert asyncio.run(scenario()) == list(chunks)


def test_a_model_error_passes_through_unchanged() -> None:
    """★ 模型自己抛的异常原样上抛，不被包装成超时。

    ⚠️ 两种失败的处置完全不同：超时该被熔断器记一笔「下游不可用」，
    而一个业务错误（比如内容审核拒绝）不是。把它们混成一种类型，
    熔断器就会在模型其实是**活着**的时候开路。
    """
    h = SpyHandler(raises=ConnectionError("连接被重置"))

    with pytest.raises(ConnectionError, match="连接被重置"):
        asyncio.run(_run_chain([build_model_timeout_middleware(timeout=1.0)], h))


def test_the_deadline_feeds_the_breaker() -> None:
    """★★★ 超时 → 熔断器记账 → 连续几次之后开路（两个中间件的接缝）。

    ⚠️ 这条是**组合**验证：单看任何一个中间件，这条链路都是对的 ——
    超时中间件自顾自地抛 ``ModelCallTimeout``，熔断中间件自顾自地记账。
    接缝处的错误（比如把超时异常吞掉、或记成成功）只有在把两者串起来
    时才暴露，而线上它们本来就是一前一后串着的。
    """
    br = breaker(failure_threshold=3)
    h = _HangingHandler()
    chain = [
        BreakerMiddleware(breaker=br),
        build_model_timeout_middleware(timeout=0.02),
    ]

    for _ in range(3):
        with pytest.raises(ModelCallTimeout):
            asyncio.run(_run_chain(chain, h))

    assert br.state is CircuitState.OPEN, "连续超时没有把熔断器打开"
    assert h.calls == 3

    # 开路之后：一次都不再调用下游（这正是熔断对「卡住」的意义）。
    asyncio.run(_run_chain(chain, h))
    assert h.calls == 3, "熔断器已经开路，仍然调用了模型"


def test_a_disabled_timeout_middleware_is_transparent() -> None:
    """★ ``enabled=False`` 时完全透明：不架时钟、原样透传。

    ⚠️ 排障开关：怀疑「回复被截断是不是这个中间件干的」时关掉它做对照。
    """
    h = _SlowHandler(delay=0.05)
    mw = build_model_timeout_middleware(timeout=0.01, enabled=False)

    result = asyncio.run(_run_chain([mw], h))

    assert result is h.response
    assert h.calls == 1


def test_a_non_positive_timeout_is_refused() -> None:
    """★ 超时取 0 或负数必须当场报错（与检索护栏同一个判据）。"""
    for bad in (0, -1.0):
        with pytest.raises(ValueError):
            build_model_timeout_middleware(timeout=bad)


def test_an_abandoned_stream_is_closed() -> None:
    """★★ 消费方中途放弃时，底层流要被关掉（连接不能挂着等 GC）。

    ⚠️ 我们返回的是一个包装过的生成器，消费方 ``break`` 时被关闭的是
    **我们这一层**。不把关闭动作透传下去，底层那条到上游的 HTTP 连接
    会一直挂到垃圾回收 —— 在压测或高频取消的场景下这就是连接泄漏。
    """
    h = _StallingStreamHandler(first_chunks=("一", "二", "三"), gap=0.0)

    async def scenario() -> bool:
        result = await _run_chain([build_model_timeout_middleware(timeout=1.0)], h)
        await _drain_one(result)  # 只取第一片就走人
        await result.aclose()
        return h.closed

    assert asyncio.run(scenario()) is True, "底层流没有被关闭"


def test_one_breaker_can_serve_many_middleware_instances() -> None:
    """★★★ 一个熔断器被多个中间件共享时，失败计数**合在一起**。

    ⚠️ 这条是整个设计的前提：框架为**每一个 agent 装配**都调一次
    中间件工厂（``agentscope/app/_service/_chat.py:994-1000``），所以生产上必然存在
    很多个 ``BreakerMiddleware`` 实例。它们必须共享同一个熔断器，
    否则「用全体调用者的失败共同判断下游是否可用」这个前提就不成立了。

    ⚠️ 断言的是「3 个实例各失败 1 次 = 熔断器开路（阈值 3）」。若每个
    实例各持一个熔断器，这里会看到 3 个各自只有 1 次失败的熔断器，
    一个都不会开路 —— 而线上表现是「下游挂了，但系统还要多扛 N 倍流量
    才熔断」。
    """
    br = breaker(failure_threshold=3)
    h = SpyHandler(raises=ConnectionError("挂了"))

    for _ in range(3):
        mw = BreakerMiddleware(breaker=br)  # 每次都是新实例，模拟三次 agent 装配
        with pytest.raises(ConnectionError):
            run(mw, h)

    assert br.state is CircuitState.OPEN, "多个中间件实例没有共享失败计数"


# ---------------------------------------------------------------------------
# 六、指标埋点
# ---------------------------------------------------------------------------
# 这一节守的是「指标是不是真的活着」。
#
# ⚠️ 三个指标曾经**只打印 HELP+TYPE、series 数 = 0**：`metrics.py` 里定义得
# 很完整、辅助函数 `observe_model_call` 也写好了，但**没有任何调用方**。
# `/metrics` 上看起来像「模型一次都没被调用过」，而 `/readyz` 同时报着
# total_calls=160。这种坏法不会让任何功能测试变红 —— 面板只是安静地空着。
#
# ⚠️ 一律用**增量**（调用前后各读一次相减），不读绝对值：REGISTRY 是
# **进程级**的，其它用例（以及将来并行跑的用例）会往同一个计数器上加数字，
# 断言绝对值等于「这条用例恰好是第一个跑的那个」，换个顺序就红。
# 这不是假想的洁癖，而是本仓库 `test_makefile_contract.py` 那类用例踩过的坑。

#: 本文件用的模型名标签。取一个不可能与真实模型重名的值，避免和别的用例
#: （或真跑过一次的模型调用）共用同一条时序。
_TEST_MODEL = "unit-test-model"


class _FakeModel:
    """只带 ``model`` 属性的模型替身。

    ⚠️ 框架真实传下来的是 ``ChatModelBase`` 实例（``.model`` 是模型名，
    见 ``agentscope/model/_base.py:93``）。这里只需要那一个属性，所以**不做**鸭子类型
    的完整仿真 —— 少一个属性就会让 :func:`_model_label` 退回 ``"unknown"``，
    而那样本节的断言会以「标签对不上」的形式失败，不会静默通过。
    """

    def __init__(self, name: str) -> None:
        """初始化。

        Args:
            name (`str`): 模型名。
        """
        self.model = name


def _sample(name: str, **labels: str) -> float:
    """读一条时序的值；该时序还不存在时返回 ``0.0``。

    Args:
        name (`str`): 指标名（计数器要带 ``_total``，直方图要带 ``_count``）。
        **labels: 标签键值。

    Returns:
        `float`: 当前值。
    """
    value = REGISTRY.get_sample_value(name, labels or None)
    return 0.0 if value is None else float(value)


def _calls(outcome: str, model: str = _TEST_MODEL) -> float:
    """:data:`MODEL_CALLS_TOTAL` 上某个 ``(model, outcome)`` 的值。"""
    return _sample("aligo_model_calls_total", model=model, outcome=outcome)


def _duration_count(model: str = _TEST_MODEL) -> float:
    """:data:`MODEL_CALL_DURATION_SECONDS` 上某个模型的观测次数。"""
    return _sample("aligo_model_call_duration_seconds_count", model=model)


def _tokens(direction: str, model: str = _TEST_MODEL) -> float:
    """:data:`MODEL_TOKENS_TOTAL` 上某个方向的累计 token 数。"""
    return _sample("aligo_model_tokens_total", model=model, direction=direction)


def _usage(*, input_tokens: int, output_tokens: int) -> ChatUsage:
    """造一个 ``ChatUsage``。

    ⚠️ ``time`` 是 ``ChatUsage`` 的**必填字段**（``agentscope/model/_model_usage.py:19``），
    造替身时漏掉它会得到一个 ``TypeError``，而不是一个「没带耗时」的对象。
    这里给 0：本文件断言的是 token 数，耗时由中间件自己测，
    两者不是同一个数（前者来自上游，后者来自本地时钟）。

    Args:
        input_tokens (`int`): 输入 token 数。
        output_tokens (`int`): 输出 token 数。

    Returns:
        `ChatUsage`: 用量对象。
    """
    return ChatUsage(input_tokens=input_tokens, output_tokens=output_tokens, time=0.0)


def test_a_successful_call_is_counted_as_ok() -> None:
    """★★★ 非流式成功 ⇒ ``outcome="ok"`` 加一，并观测一次耗时。

    ⚠️ 这条用例是「指标是不是活的」的**总闸**：它要是红了，说明埋点被摘掉了，
    面板上所有的模型调用曲线都会变成空白 —— 而系统本身**一点问题都没有**。
    """
    mw = BreakerMiddleware(breaker=breaker())
    h = SpyHandler()
    before, before_duration = _calls("ok"), _duration_count()

    run(mw, h, current_model=_FakeModel(_TEST_MODEL))

    assert _calls("ok") - before == 1
    assert _duration_count() - before_duration == 1, "耗时没有被观测"


def test_a_failed_call_is_counted_as_error() -> None:
    """★★★ 抛异常 ⇒ ``outcome="error"`` 加一，且异常照旧抛给调用方。

    ⚠️ 两个断言缺一不可：只断言计数会漏掉「吞异常」，只断言抛异常会漏掉
    「没记账」—— 而这两件事曾经分别由别的用例守着，加指标时**最容易被顺手
    改坏**（比如把 ``raise`` 挪到 ``observe_model_call`` 上面去）。
    """
    mw = BreakerMiddleware(breaker=breaker())
    h = SpyHandler(raises=ConnectionError("挂了"))
    before = _calls("error")

    with pytest.raises(ConnectionError):
        run(mw, h, current_model=_FakeModel(_TEST_MODEL))

    assert _calls("error") - before == 1


def test_a_short_circuit_is_counted_as_rejected_and_not_as_a_duration() -> None:
    """★★★ 熔断打开 ⇒ ``rejected`` 加一，且**不观测耗时**。

    ⚠️ 这一条是 ``rejected`` 单列的全部意义：它和 ``error`` 的处置方式完全
    不同 —— 「error 高」是下游挂了，「rejected 高」是**我们自己在拦**。
    合成一条 ``error`` 会让面板把「上游故障」和「熔断器在保护上游」说成
    同一件事，于是**在最该看出「我们正在放弃请求」的时刻看不出来**。

    ⚠️ 断言直方图**不动**：耗时传 0 也会被
    :func:`src.observability.metrics.observe_model_call` 跳过
    （``outcome == "rejected"`` 时整段跳过）。如果哪天有人在短路分支上
    改传真实耗时，P50 会被这些 0 拉低，看起来像「模型突然变快了」——
    正好是反的。
    """
    br = breaker(failure_threshold=1)
    br_open = BreakerMiddleware(breaker=br)
    with pytest.raises(ConnectionError):
        run(br_open, SpyHandler(raises=ConnectionError("挂了")))

    before_rejected, before_duration = _calls("rejected"), _duration_count()
    h = SpyHandler()
    result = run(br_open, h, current_model=_FakeModel(_TEST_MODEL))

    assert h.calls == 0, "熔断了却仍然调用了模型"
    assert result.is_last is True
    assert _calls("rejected") - before_rejected == 1
    assert _duration_count() == before_duration, "被拒绝的调用不该进耗时直方图"


def test_a_completed_stream_is_counted_once_as_ok() -> None:
    """★★★ 流式跑完 ⇒ ``ok`` **只加一**，不是每个分片加一。

    ⚠️ 计数据点在 ``_guard_stream`` 里、循环之外，就是为了这个。写进循环里
    会让「模型调用次数」变成「分片数」，而分片数与回复长度成正比 ——
    面板上会看到一个随用词长短起伏的、完全没有意义的曲线。
    """
    mw = BreakerMiddleware(breaker=breaker())
    h = SpyHandler(stream_chunks=(ChatResponse(content=[], is_last=False), ChatResponse(content=[], is_last=True)))
    before, before_duration = _calls("ok"), _duration_count()

    asyncio.run(_drain(run(mw, h, current_model=_FakeModel(_TEST_MODEL))))

    assert _calls("ok") - before == 1
    assert _duration_count() - before_duration == 1


def test_a_failed_stream_is_counted_as_error() -> None:
    """★★★ 流迭代到一半抛 ⇒ ``error`` 加一。"""
    mw = BreakerMiddleware(breaker=breaker())
    h = SpyHandler(
        stream_chunks=(ChatResponse(content=[], is_last=False),),
        stream_boom=ConnectionError("流断了"),
    )
    before = _calls("error")

    with pytest.raises(ConnectionError):
        asyncio.run(_drain(run(mw, h, current_model=_FakeModel(_TEST_MODEL))))

    assert _calls("error") - before == 1


def test_an_abandoned_stream_is_counted_as_nothing() -> None:
    """★★ 流被中途放弃 ⇒ 三个 outcome **都不加**（与熔断器的记账一致）。

    ⚠️ 这条钉的是「指标不撒谎」：被放弃的调用确实没有结果。把它记成 ``ok``
    会让一个「连接建得上、但一读就断」的下游在面板上永远是绿的 ——
    这正是「熔断器在生产路径上完全失效」那个 bug 的指标版本。
    """
    mw = BreakerMiddleware(breaker=breaker())
    h = SpyHandler(
        stream_chunks=(ChatResponse(content=[], is_last=False), ChatResponse(content=[], is_last=True)),
    )
    before = {outcome: _calls(outcome) for outcome in ("ok", "error", "rejected")}

    async def abandon() -> None:
        """拿到流就关掉，一个分片都不消费。"""
        stream = await mw.on_model_call(object(), {**INPUT_KWARGS, "current_model": _FakeModel(_TEST_MODEL)}, h)
        await stream.aclose()

    asyncio.run(abandon())

    assert {o: _calls(o) - before[o] for o in before} == {"ok": 0, "error": 0, "rejected": 0}


def test_the_tokens_of_a_stream_come_from_the_chunk_that_has_them() -> None:
    """★★ 流式 token 数取自**带 usage 的那一片**（本项目实测：只有末片带）。

    ⚠️ 先给一片**带 usage 的非末片**，再给一片**不带 usage 的末片**。
    这个顺序专门用来抓「只看最后一片」的写法 —— 那样会记成 0 而不是 7。
    真实上游恰好是反过来的（只有末片带），所以只按真实顺序测**测不出**
    这个 bug，得故意造一个反例。
    """
    with_usage = ChatResponse(
        content=[],
        is_last=False,
        usage=_usage(input_tokens=7, output_tokens=3),
    )
    without_usage = ChatResponse(content=[], is_last=True)
    mw = BreakerMiddleware(breaker=breaker())
    h = SpyHandler(stream_chunks=(with_usage, without_usage))
    before_in, before_out = _tokens("input"), _tokens("output")

    asyncio.run(_drain(run(mw, h, current_model=_FakeModel(_TEST_MODEL))))

    assert _tokens("input") - before_in == 7
    assert _tokens("output") - before_out == 3


def test_a_response_without_usage_does_not_report_zero_tokens() -> None:
    """★★ 拿不到 usage ⇒ token 时序**压根不出现**，而不是记一笔 0。

    ⚠️ 区别在面板上是「没配」和「坏了」：记 0 会让成本面板显示
    「今天消耗了 0 个 token」——在一个明明在跑的模型上，这个数字是**假的**，
    而且没人会去查一个看起来正常的 0。
    """
    mw = BreakerMiddleware(breaker=breaker())
    h = SpyHandler()  # 默认 response 的 usage 是 None
    before_in, before_out = _tokens("input"), _tokens("output")

    run(mw, h, current_model=_FakeModel(_TEST_MODEL))

    assert _tokens("input") == before_in
    assert _tokens("output") == before_out


def test_a_disabled_switch_also_silences_the_metrics() -> None:
    """★ ``enabled=False`` ⇒ 指标也不记（这是「完全透明」的代价，已写进文档）。

    ⚠️ 这条用例存在的意义不是「保证行为正确」，而是**把这个取舍钉住**：
    关掉开关做 A/B 对照时，面板上这一段会变成空白，看起来像「没有任何模型
    调用」。哪天要让关掉开关也记指标，就必须先改这条用例 —— 也就是
    **被迫读一遍上面那段文档**，而不是在排障时才发现面板空了。
    """
    mw = BreakerMiddleware(breaker=breaker(), enabled=False)
    before = {outcome: _calls(outcome) for outcome in ("ok", "error", "rejected")}

    run(mw, SpyHandler(), current_model=_FakeModel(_TEST_MODEL))

    assert {o: _calls(o) - before[o] for o in before} == {"ok": 0, "error": 0, "rejected": 0}


# ---------------------------------------------------------------------------
# 七、指标标签的取值（两个纯函数）
# ---------------------------------------------------------------------------
def test_the_model_label_falls_back_when_the_kwargs_have_no_model() -> None:
    """★★ 取不到模型名 ⇒ ``"unknown"``，而不是抛异常。

    ⚠️ 用 ``object()`` 而不是删掉那个键：``INPUT_KWARGS`` 里的
    ``current_model`` **就是** ``object()``，也就是本文件其它 30 多条用例
    真实传下去的东西。这里断言的是「本文件其余用例在跑的时候，指标上落的是
    `unknown` 而不是把 KeyError 抛进真实请求里」。
    """
    assert _model_label({"current_model": object()}) == "unknown"
    assert _model_label({}) == "unknown"
    assert _model_label({"current_model": _FakeModel("")}) == "unknown"
    assert _model_label({"current_model": _FakeModel("qwen-max")}) == "qwen-max"


def test_the_token_reader_refuses_to_guess() -> None:
    """★★ ``_usage_tokens`` 对「没有 usage」「usage 为 None」一律返回 ``(0, 0)``。

    ⚠️ ``(0, 0)`` 在这里是**「未知」的编码**，不是「消耗了 0 个」—— 之所以
    能这么用，是因为 :func:`observe_model_call` 对 0 是跳过而不是记账。
    两处必须成对理解，改一处就要改另一处。
    """
    assert _usage_tokens(ChatResponse(content=[], is_last=True)) == (0, 0)
    assert _usage_tokens(object()) == (0, 0)
    assert _usage_tokens(ChatResponse(content=[], is_last=True, usage=_usage(input_tokens=11, output_tokens=4))) == (11, 4)
