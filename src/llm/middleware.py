# -*- coding: utf-8 -*-
"""把熔断器包成 **``on_model_call`` 中间件**。

文件职责：
    给每一次真实的模型调用套上熔断保护。熔断器本身在
    :mod:`src.llm.breaker`（纯逻辑，可穷举单测），本模块只负责把它挂到
    框架的钩子上。

上下游依赖：
    - 上游：``agentscope.middleware.MiddlewareBase``、:mod:`src.llm.breaker`。
    - 下游：``src/server/agents_factory.py`` 在每个 agent 装配时把它加进
      中间件列表。

═══ 为什么是中间件，而不是「在工厂里包一层 model」 ═══

``build_chat_model`` 返回的模型对象确实可以包一层代理来做熔断。但那样会
丢掉两样东西：

1. **装配点的可见性**。熔断器是一个**跨请求共享**的对象（见
   :func:`src.llm.factory.get_breaker` 的说明）。包在模型里，它就成了模型
   的私有状态 —— 每个 agent 装配各造一个模型，也就各造一个熔断器，
   熔断点被推迟 N 倍（N = 并发数）。
2. **短路的位置**。中间件能在**不调用模型**的前提下直接返回一个响应，
   这才是熔断的意义；包一层代理只能选择「抛异常」，而异常会以失败的
   形式一路冒到用户面前。

═══ 用的是 7 个 hook 里的哪一个 ═══

``MiddlewareBase`` 一共暴露 **7 个 hook**（口径：``middleware/_base.py`` 里
以 ``on_`` 开头的 ``async def``；``list_tools`` 与 ``get_middleware_key``
是普通方法，不算钩子）。其中 6 个是洋葱式（自带 ``next_handler``，
可以「包住」下游），1 个是转换式（``on_system_prompt``，只做字符串改写）。

本模块选 ``on_model_call``，理由是**只有它**同时满足两个条件：
能包住真实模型调用（于是能计失败），且能在不调用模型的前提下返回响应
（于是能短路）。

⚠️ 权限判断**不在** ``on_acting`` 上做，尽管名字看起来更像。
``on_acting`` 只包住 ``call_tool`` 这个纯 I/O 层（``agentscope/agent/_agent.py:2719``），
在它上面拦工具调用会连框架自己的记账与事件都一起绕过。

═══ ⚠️ ``next_handler`` 必须被调用，且必须用关键字传参 ═══

已核实（``agentscope/agent/_agent.py:3365-3371``）：``on_model_call`` 的返回值就是
整条链的返回值。**不调用 ``next_handler`` 就等于真实模型永远不会被调用**
（快车道正是利用这一点短路的）。

``next_handler`` 的签名是 ``async def next_handler(**kwargs)`` —— 它把
kwargs **合并**到捕获的 ``input_kwargs`` 上再往下传。所以必须写成
``await next_handler(**input_kwargs)``，写成 ``next_handler(input_kwargs)``
会把整个字典当成一个位置参数，在更深的地方报一个看不懂的 ``TypeError``。

═══ ⚠️ 熔断打开时**不再记账** ═══

短路返回时**不**调用 ``record_failure``。记的话，每一次被拒绝的请求都会
把熔断窗口往后推，于是「冷却 30 秒后放一个探针进来」永远不会发生 ——
熔断器变成永久断开。这是一个只在**下游真的挂了**的时候才会触发的 bug，
也就是最难在测试里撞上的时刻。

═══ 本模块的第二个中间件：单次模型调用的**截止时间** ═══

:class:`ModelTimeoutMiddleware` 补的是熔断器**管不到**的那一半：

    · 熔断器管「连续失败多少次之后不再调用」—— 它需要失败**先发生**；
    · 但一次卡住的调用根本不会失败：它会一直挂着。于是熔断器永远数不到
      那一笔账，请求也永远不返回 —— 用户在页面上看到的是转圈转到天荒地老。

这不是假想的。本项目**离线链路**（`build_chat_model`）会给 SDK 显式设
`timeout`（`src/llm/factory.py` 的 `client_kwargs`），但**框架的 app 链路
不走那个函数** —— `agentscope/app/_service/_model.py:54-58` 直接用
`model_cls(credential=…, model=…, parameters=…)` 构造，于是：

    · 没有 `client_kwargs` ⇒ openai SDK 用**默认超时 600s**；
    · 没有降 `max_retries` ⇒ openai SDK 默认再自己重试 2 次（3 次尝试）；
    · 框架 `ChatModelBase.__call__` 自己又重试 `max_retries=3`
      （`agentscope/model/_base.py:208` 的 `range(self.max_retries + 1)` ⇒ 4 轮）。

最坏情况 ≈ 4 × 3 × 600s = **2 小时**：一次卡住的模型调用可以让请求挂到
天荒地老。中间件挂在链上，是 app 链路唯一能收口的地方（离线脚本由
`build_chat_model` 的 SDK 超时兜住）。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Final

from agentscope.message import TextBlock
from agentscope.middleware import MiddlewareBase
from agentscope.model import ChatResponse

from src.llm.breaker import CircuitBreaker, CircuitBreakerOpen
from src.observability.metrics import observe_model_call

#: 本模块的日志器。
logger = logging.getLogger(__name__)

#: 熔断打开时给用户看的话。
#:
#: ⚠️ 不提「熔断」「失败率」「下游」这些词。用户无法处理这些信息，
#: 他只需要知道「现在用不了、等一下再试」，以及**他刚才的操作没有被执行**——
#: 最后这一点最要紧：用户看到报错时的第一反应是「那我的申请提交了吗」。
_OPEN_MESSAGE = (
    "模型服务暂时不可用，这一步没有执行成功。"
    "请稍等一会儿再试一次；如果你刚才是在提交申请或取消订单，"
    "请先查一下订单状态确认它没有被重复提交。"
)


#: 取不到模型名时给指标用的标签值。
#:
#: ⚠️ 刻意**不写 ``"unknown"`` 之外的任何东西**：Prometheus 的标签值必须是
#: 有界的。若把 ``repr(model)`` 或异常信息塞进标签，一次故障就能造出成千上万
#: 条时序，把 Prometheus 拖垮 —— 这比「指标没数据」严重得多。
_UNKNOWN_MODEL: Final = "unknown"


def _model_label(input_kwargs: dict[str, Any]) -> str:
    """从 ``input_kwargs`` 里取出模型名，供 Prometheus 标签使用。

    ⚠️ 为什么用 ``getattr`` 兜底而不是直接 ``input_kwargs["current_model"].model``：

    1. 键可能不在。``on_model_call`` 的 ``input_kwargs`` 由框架组装
       （``agentscope/agent/_agent.py:3347-3352``），但本中间件也可以被别的调用方挂到
       别的链上 —— 那时少一个键就是 ``KeyError``，
       **一个纯观测动作绝不该让真实请求失败**；
    2. 值可能不是模型。已核实：框架传的是 ``ChatModelBase`` 实例，但单测里
       惯常用 ``object()`` 占位（``tests/test_llm_middleware.py`` 的
       ``INPUT_KWARGS``），它没有 ``.model``。

    两处都退化成 :data:`_UNKNOWN_MODEL`，于是「模型名取不到」这件事在面板上
    表现为一条 ``model="unknown"`` 的时序 —— **看得见**，而不是悄悄丢失。

    Args:
        input_kwargs (`dict[str, Any]`): ``on_model_call`` 收到的参数字典。

    Returns:
        `str`: 模型名；取不到时是 :data:`_UNKNOWN_MODEL`。
    """
    name = getattr(input_kwargs.get("current_model"), "model", None)
    if isinstance(name, str) and name:
        return name
    return _UNKNOWN_MODEL


def _usage_tokens(response: Any) -> tuple[int, int]:
    """从一次响应里取 ``(input_tokens, output_tokens)``。

    ⚠️ 取不到就返回 ``(0, 0)``，而 :func:`observe_model_call` 对 0 是**跳过
    计数**而不是记一笔 0 —— 两者在面板上完全不同：记 0 会让「token 消耗」
    看起来是 0（像坏了），跳过则是这条时序压根不出现（像没配）。
    这里选择后者，因为「拿不到 token」与「用了 0 个 token」是两件事，
    把它俩混起来是**在成本面板上说谎**。

    ⚠️ 流式响应里**只有最后一个分片**带 ``usage``（本项目实测：非末片的
    ``usage`` 是 ``None``），所以调用方需要**逐片**取、取到非空就覆盖 ——
    见 :meth:`BreakerMiddleware._guard_stream`。

    Args:
        response (`Any`): 一个 ``ChatResponse``（或形状相同的对象）。

    Returns:
        `tuple[int, int]`: 输入、输出 token 数；未知时都是 0。
    """
    usage = getattr(response, "usage", None)
    if usage is None:
        return (0, 0)
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


class BreakerMiddleware(MiddlewareBase):
    """``on_model_call`` 钩子上的熔断保护。

    ⚠️ **必须传进程内唯一的那个熔断器**（
    :func:`src.llm.factory.get_breaker`）。每个 agent 各造一个的话，
    「用全体调用者的失败共同判断下游是否可用」这个前提就不成立了 ——
    熔断点被推迟到「每个实例各自失败 N 次」，N 就是并发数。

    ⚠️ 同一个熔断器实例被多个中间件实例共享是**安全**的：本类自己
    没有可变字段，所有状态都在熔断器里，而熔断器自己处理并发
    （见 :mod:`src.llm.breaker` 的状态机说明）。

    ═══ 指标埋点 ═══

    本类**同时是** ``aligo_model_calls_total`` / ``aligo_model_call_duration_seconds``
    / ``aligo_model_tokens_total`` 三个指标的唯一埋点处
    （:func:`src.observability.metrics.observe_model_call`）。放这里而不是
    放进 ``factory.build_chat_model``，是因为**只有这条链能看见全部三种结果**：
    离线脚本那条路压根没有熔断器，于是永远看不到 ``rejected`` 那一笔 ——
    而「error 高」与「我们在自己熔断」必须能分开，这正是 ``rejected`` 单列的
    理由（见 ``metrics.py`` 对 ``MODEL_CALLS_TOTAL`` 的注释）。

    ⚠️ 三个结果的对应关系是**逐分支**的，不是「成功/失败」两分：
    ``rejected`` ⇔ 熔断打开短路（没发出去）、``error`` ⇔ 抛了异常、
    ``ok`` ⇔ 正常返回（含「回复内容是废话但服务正常」—— 内容质量不是
    熔断器能判断的，把它算成 error 会让一个只会说车轱辘话的模型被熔断）。

    ⚠️ ``enabled=False`` 时**连指标也不记**。这是有代价的：关掉开关做 A/B
    对照时，面板上这一段会变成空白，看起来像「没有任何模型调用」。
    仍然这么定，是因为这个开关的语义已经由
    ``test_disabled_middleware_is_completely_transparent`` 钉死为
    「完全透明」（它断言连熔断器的账本都不动），而「只在观测上不透明」
    是一个更难向排障的人解释的状态。**关掉它是排查手段，不是运行模式** ——
    要长期关，改配置里的 ``enabled`` 默认值，别把这个开关当运行开关用。

    Attributes:
        _breaker (`CircuitBreaker`): 共享的熔断器。
        _enabled (`bool`): 总开关（排障用）。
    """

    def __init__(self, *, breaker: CircuitBreaker, enabled: bool = True) -> None:
        """初始化。

        Args:
            breaker (`CircuitBreaker`): 熔断器。
            enabled (`bool`): 是否启用。``False`` 时本中间件完全透明
                （直接透传），用于对照排查「回复变慢/变差是不是熔断引起的」。
        """
        self._breaker = breaker
        self._enabled = enabled

    async def on_model_call(
        self,
        agent: Any,
        input_kwargs: dict[str, Any],
        next_handler: Any,
    ) -> Any:
        """包裹一次真实的模型调用。

        ⚠️ 四个分支的语义各不相同，别合并：

        - **熔断打开**：合成一个 ``ChatResponse`` 短路返回。**不调用**
          ``next_handler``（那才是短路），也**不记账**（见模块文档）。
        - **调用抛异常**：记一次失败，然后**原样抛出去**。不吞掉 ——
          框架对模型异常有自己的处置（重试、降级），吞掉会让那些机制失效，
          而用户会拿到一个我们编的、看起来像正常回复的东西。
        - **返回的不是流**（``ChatResponse``）：记一次成功。
        - **返回的是流**（``AsyncGenerator``）：**把记账推迟到流被消费完**，
          见下。⚠️ 「成功」的判据是**没有抛异常**，不是「回复内容正确」——
          内容质量问题不是熔断器能判断的，把它算进去会让一个只会说
          车轱辘话但服务正常的模型被熔断。

        Args:
            agent (`Any`): 框架传入的 agent 实例（本中间件不用，但签名要求）。
            input_kwargs (`dict[str, Any]`): 模型调用参数
                （``current_model`` / ``messages`` / ``tools`` / ``tool_choice``）。
            next_handler (`Any`): 链上的下一个处理者。

        Returns:
            `Any`: 真实模型响应（或它的记账包装），或熔断时合成的响应。
        """
        if not self._enabled:
            return await next_handler(**input_kwargs)

        model = _model_label(input_kwargs)

        try:
            await self._breaker.allow_request()
        except CircuitBreakerOpen as exc:
            logger.warning(
                "熔断器处于打开状态，本轮跳过真实模型调用（%s）。",
                exc.retry_after_seconds,
            )
            # ⚠️ 耗时传 0：这一笔**根本没发出去**，把它算进延迟分位数会
            # 让「模型变快了」这种假象出现在最该报警的时刻（下游正挂着）。
            observe_model_call(model, "rejected", 0.0)
            return _open_response()

        started = time.monotonic()
        try:
            response = await next_handler(**input_kwargs)
        except Exception:
            # ⚠️ 先记账再抛。顺序反过来的话，``record_failure`` 自己抛异常
            # （比如它内部有 bug）会**盖掉**原始的模型异常，
            # 而那个原始异常才是排查时真正需要的。
            observe_model_call(model, "error", time.monotonic() - started)
            await self._breaker.record_failure()
            raise

        if inspect.isasyncgen(response):
            # ⚠️ 流式响应**不能在这里记成功**，见 :meth:`_guard_stream`。
            # 同理也**不能在这里记指标**：此刻只是建出了生成器，
            # 网络 I/O 一次都没发生，耗时刻度从这里起算才有意义
            # （所以把 started 传下去，而不是在那边重新取一次时间）。
            return self._guard_stream(response, model=model, started=started)

        input_tokens, output_tokens = _usage_tokens(response)
        observe_model_call(
            model,
            "ok",
            time.monotonic() - started,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        await self._breaker.record_success()
        return response

    async def _guard_stream(
        self,
        stream: Any,
        *,
        model: str,
        started: float,
    ) -> Any:
        """把模型返回的异步生成器包一层，**在流被消费时**记账。

        ═══ ⚠️ 为什么必须有这一层 ═══

        流式（``stream=True``）是本项目的**生产默认**，而在这个模型上，
        「调用」与「拿到结果」是分开的两次操作：

        1. ``await current_model(...)`` 只是**建出**一个异步生成器，
           不做任何 I/O（已核实：``agentscope/model/_base.py:254-290`` 返回 ``_stream()``；
           ``agentscope/agent/_agent.py:3339-3343`` 是 ``return await current_model(...)``）；
        2. 真正的网络请求发生在框架**迭代**这个生成器的时候
           （``agentscope/agent/_agent.py:1744-1757`` 的 ``async for chunk in res``）。

        于是「``await next_handler(...)`` 没抛异常」**不等于**「模型调用成功」
        —— 上游挂掉、超时、鉴权失败，全都发生在上面的第 2 步，
        也就是本模块 ``try`` 块之外。

        ⚠️ 实测过的后果（这是本方法存在的唯一理由）：用真实的调用约定
        （``next_handler`` 返回一个「迭代到一半才抛 ``ConnectionError``」的
        生成器）连打 5 次，熔断器仍然是 ``CLOSED``、``consecutive_failures``
        是 **0**；而同样 5 次非流式失败会让它 ``OPEN``。
        也就是说：**熔断器在生产路径上是完全失效的** —— 下游真的挂了，
        它一次都不会开。这个 bug 不会让任何功能测试变红，
        因为非流式路径的表现完全正确。

        ⚠️ 只捕 ``Exception``，不捕 ``BaseException``：
        - 用户取消（``CancelledError``）与生成器被提前关闭（``GeneratorExit``）
          都继承 ``BaseException``，它们**不是**下游故障。捕了的话，
          用户每点一次「停止」就给熔断器记一笔失败。

        ⚠️ 流被中途放弃（消费方 ``break`` 或压根不迭代）时两边都不记：
        这次调用确实没有结果，把它算成成功会让一个「连接建得上、
        但一读就断」的下游永远熔不断。**指标同理**，所以
        ``rejected`` / ``ok`` / ``error`` 三种结果之外不会再出现第四种
        「没结果」—— 它在这条路径上就是不可观测的，别在面板上假装能看见。

        ⚠️ 逐片取 ``usage`` 而不是只看最后一片：本项目实测，流式响应里
        **只有末片带 usage**，但这是上游的行为、不是契约，所以写成
        「谁带就记谁」，最后取到的那一份生效。

        Args:
            stream (`Any`): ``next_handler`` 返回的异步生成器。
            model (`str`): 模型名标签，由 :func:`_model_label` 取出。
            started (`float`): :func:`time.monotonic` 的起点，取在
                ``next_handler`` **被调用之前** —— 于是这里的耗时是
                「建流 + 真实网络往返 + 消费完整个流」的总时长，
                也就是用户实际等待的时间。

        Yields:
            `Any`: 原样透传的每个模型分片。
        """
        input_tokens = 0
        output_tokens = 0
        try:
            async for chunk in stream:
                chunk_input, chunk_output = _usage_tokens(chunk)
                if chunk_input or chunk_output:
                    input_tokens, output_tokens = chunk_input, chunk_output
                yield chunk
        except Exception:
            observe_model_call(model, "error", time.monotonic() - started)
            await self._breaker.record_failure()
            raise

        observe_model_call(
            model,
            "ok",
            time.monotonic() - started,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        await self._breaker.record_success()


def _open_response() -> ChatResponse:
    """构造熔断打开时的合成响应。

    ⚠️ 用 ``ChatResponse``（dataclass）而不是伪造一个模型对象：
    ``on_model_call`` 的返回类型是 ``ChatResponse | AsyncGenerator[ChatResponse, None]``
    （注解在 ``agentscope/agent/_agent.py:3336``，调用点在 ``:3365-3371``）。
    我们取前一支 —— 返回别的东西（比如一个鸭子类型的假对象），
    框架会在这个钩子之后的某一行崩掉，而堆栈指向框架内部而非本模块。

    ⚠️ ``is_last=True`` 在这条路径上是**防御性**的，不是承重的。
    已核实，框架只在**流式**分支上读它
    （``agentscope/agent/_agent.py:1746-1748`` 的 ``async for chunk in res: if chunk.is_last``），
    而一个裸 ``ChatResponse`` 走的是 ``isinstance(res, ChatResponse)``
    （``:1759-1760``）那一支，**根本不会读** ``is_last``。
    所以漏掉它在这里不会「卡住不结束」—— 那种说法是错的，别照着它推理。

    仍然显式写上，是因为这个对象将来可能被别处复用（或本中间件被排到
    别的 ``on_model_call`` 之间），而**在流式分支上**它是必需的：
    框架拿不到 ``is_last=True`` 的分片会直接
    ``raise RuntimeError("Model returned an empty streaming response...")``
    （``:1790-1796``）。写一个正确的值比省略它便宜。

    Returns:
        `ChatResponse`: 只含一段提示文本的响应。
    """
    return ChatResponse(
        content=[TextBlock(text=_OPEN_MESSAGE)],
        is_last=True,
    )


class ModelCallTimeout(TimeoutError):
    """一次模型调用超过预算仍未返回（或流式响应迟迟不出下一个分片）。

    ⚠️ 继承内置 ``TimeoutError``（与 ``VectorSearchTimeout`` 同一个理由）：
    调用方写 ``except TimeoutError`` 也能接住它，不必先认识本模块 ——
    框架的降级链路与我们的兜底处理都是宽口径捕获。

    Attributes:
        timeout (`float`): 触发的超时预算（秒）。
        streaming (`bool`): 是「等流式分片」超时，还是「整次调用」超时。
            两者的运维含义不同：前者多半是上游把流挂住了，
            后者多半是连接/首包就没回来。
    """

    def __init__(self, timeout: float, *, streaming: bool) -> None:
        """初始化。

        Args:
            timeout (`float`): 超时预算（秒）。
            streaming (`bool`): 是否发生在流式消费阶段。
        """
        self.timeout = timeout
        self.streaming = streaming
        where = "等待下一个流式分片" if streaming else "整次调用"
        super().__init__(
            f"模型调用超过 {timeout:g}s 仍未返回（{where}），已放弃本次调用"
            f"（详见 src/llm/middleware.py 的模块文档）。",
        )


def _consume_outcome(task: "asyncio.Task[Any]") -> None:
    """取走一个我们已不再关心的任务的结局，避免 "never retrieved" 噪音。

    ⚠️ 只对**已结束**的任务调用（做成 done callback 就是为了保证这一点）：
    对还在跑的任务调 ``exception()`` 会抛 ``InvalidStateError``，
    而为一次清理动作引入一个可能上抛的异常，是纯粹的负收益。

    Args:
        task (`asyncio.Task`): 已结束的任务。
    """
    if task.cancelled():
        return
    try:
        task.exception()
    except Exception:  # noqa: BLE001 —— 消费结局本身不该影响任何控制流
        pass


class ModelTimeoutMiddleware(MiddlewareBase):
    """``on_model_call`` 钩子上的**单次调用截止时间**。

    ⚠️ 与 :class:`BreakerMiddleware` 是上下游关系，缺一不可：

        · 熔断器需要失败**先发生**才会开路；而卡住不返回的调用永远不会失败；
        · 本中间件把「卡住」变成一次**有界**的失败（``ModelCallTimeout``），
          这个异常再被链上的熔断器记进失败计数 —— 于是连续卡住几次之后，
          熔断器会把下游判成不可用，请求连卡都不再卡。

    超时预算用 ``settings.llm.timeout_seconds``（默认 60s）：与离线链路
    给 SDK 的那个 ``timeout`` 是**同一个数**，即「一次模型调用最多允许多久」
    在两套入口上口径一致。

    ⚠️ 流式（本项目生产默认）的语义是**分片间不活动超时**：每等一个分片
    给一整个预算。这与 httpx/openai SDK 对流式响应的 read timeout 语义一致
    —— 一个持续在吐字的回答不会被掐断，而一个卡住不动的流会在预算内失败。

    Attributes:
        _timeout (`float`): 单次调用（或分片间）的预算秒数。
        _enabled (`bool`): 总开关（排障用）。
    """

    def __init__(self, *, timeout: float, enabled: bool = True) -> None:
        """初始化。

        Args:
            timeout (`float`): 超时预算（秒），必须 > 0。
            enabled (`bool`): ``False`` 时完全透明（直接透传）。

        Raises:
            ValueError: ``timeout`` 非正时。取 0 会让**每次**调用都超时，
                等于把模型能力静默关掉 —— 这种配置必须当场炸。
        """
        if timeout <= 0:
            raise ValueError(
                f"模型调用超时必须 > 0，实际为 {timeout!r}；"
                f"取 0 会让每次调用都超时（等价于静默关闭模型能力）。",
            )
        self._timeout = float(timeout)
        self._enabled = enabled

    async def on_model_call(
        self,
        agent: Any,
        input_kwargs: dict[str, Any],
        next_handler: Any,
    ) -> Any:
        """在预算内等模型调用返回；流式响应则继续在预算内等每一个分片。

        Args:
            agent (`Any`): 框架传入的 agent 实例（本中间件不用，签名要求）。
            input_kwargs (`dict[str, Any]`): 模型调用参数。
            next_handler (`Any`): 链上的下一个处理者。

        Returns:
            `Any`: 模型响应，或**包了一层截止时间**的异步生成器。

        Raises:
            ModelCallTimeout: 超过预算仍未返回时。
        """
        if not self._enabled:
            return await next_handler(**input_kwargs)

        response = await self._await_within_budget(
            next_handler(**input_kwargs),
            streaming=False,
        )
        if inspect.isasyncgen(response):
            # ⚠️ 非流式到这里就结束了；流式的 I/O 全在**迭代**里发生
            # （见 :meth:`BreakerMiddleware._guard_stream` 的说明），
            # 所以必须把截止时间一路带到分片上，而不是在这里判「没抛异常 = 成功」。
            return self._guard_stream(response)
        return response

    async def _guard_stream(self, stream: Any) -> Any:
        """把异步生成器包一层：每等一个分片的时间都受预算约束。

        Args:
            stream (`Any`): ``next_handler`` 返回的异步生成器。

        Yields:
            `Any`: 原样透传的每个分片。

        Raises:
            ModelCallTimeout: 某个分片超过预算仍未到达时。
        """
        iterator = stream.__aiter__()
        try:
            while True:
                try:
                    chunk = await self._await_within_budget(
                        iterator.__anext__(),
                        streaming=True,
                    )
                except StopAsyncIteration:
                    return
                yield chunk
        finally:
            # ⚠️ 无论是正常读完、超时、还是消费方中途 ``break``（我们会被
            # ``aclose()``，这里的 ``finally`` 照跑），都要把底层流关掉 ——
            # 否则那条到上游的 HTTP 连接会一直挂着，直到 GC 才回收。
            await _aclose_quietly(stream)

    async def _await_within_budget(self, awaitable: Any, *, streaming: bool) -> Any:
        """在预算内等一个可等待对象出结果，超时就放弃。

        ⚠️⚠️ **不能用 ``asyncio.wait_for``** —— 这是本方法存在的唯一理由。
        ``wait_for`` 靠**取消**来中断，再根据「取消有没有传播出来」判定超时；
        而框架的模型层**故意吞掉** ``CancelledError``（把它转成一次优雅的
        「被打断」）：

          · 非流式：``agentscope/model/_base.py:219-224`` 返回一个
            ``finished_reason=INTERRUPTED`` 的空响应；
          · 流式：``agentscope/model/_base.py:283-288`` 把取消转成一个末尾分片。

        于是超时取消会被优雅地接住，``wait_for`` 看到的是「正常返回」，
        超时判定**永远不成立** —— 护栏在纸面上存在、实际一次都不触发。
        所以这里把调用放进**独立任务**，按 ``asyncio.wait`` 的**时钟**
        判定，超时由我们自己抛。

        ⚠️ 超时后仍要 ``cancel()`` 内层任务：结果是不要了，但那条到上游的
        连接得尽快断掉（否则它会一直占着连接池，正是要防的堆积）。

        Args:
            awaitable (`Any`): 被等待的协程/可等待对象。
            streaming (`bool`): 是否发生在流式分片之间（只影响异常与日志的措辞）。

        Returns:
            `Any`: 内层结果（若在预算内结束）。

        Raises:
            ModelCallTimeout: 超过预算时。
            BaseException: 内层自己抛出的异常，原样上抛。
        """
        task = asyncio.ensure_future(awaitable)
        try:
            done, _ = await asyncio.wait({task}, timeout=self._timeout)
        except BaseException:
            # 外层被取消（用户点了「停止生成」）：内层也要收掉，
            # 否则它会继续占着到上游的连接，而我们已经在返回了。
            task.cancel()
            task.add_done_callback(_consume_outcome)
            raise

        if task in done:
            return task.result()

        task.cancel()
        task.add_done_callback(_consume_outcome)
        # 让出一次，给「取消」一个被投递的机会：否则内层任务可能带着一次
        # 尚未投递的取消活到事件循环关闭，留下
        # "Task was destroyed but it is pending" 的噪音日志。
        await asyncio.sleep(0)
        where = "等待流式分片" if streaming else "整次调用"
        logger.warning(
            "模型调用超过 %.1fs 仍未返回（%s），已放弃本次调用；"
            "该异常会计入链上熔断器的失败计数。",
            self._timeout,
            where,
        )
        raise ModelCallTimeout(self._timeout, streaming=streaming)


async def _aclose_quietly(stream: Any) -> None:
    """尽力关闭一个异步生成器，绝不因为它失败而影响主流程。

    Args:
        stream (`Any`): 待关闭的对象（没有 ``aclose`` 时是 no-op）。
    """
    close = getattr(stream, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except Exception:  # noqa: BLE001 —— 清理失败只是一条日志
        logger.debug("关闭模型响应流失败", exc_info=True)


def build_model_timeout_middleware(
    *,
    timeout: float,
    enabled: bool = True,
) -> ModelTimeoutMiddleware:
    """构造模型调用超时中间件。

    Args:
        timeout (`float`): 超时预算（秒），通常取 ``settings.llm.timeout_seconds``。
        enabled (`bool`): 是否启用。

    Returns:
        `ModelTimeoutMiddleware`: 构造好的中间件。
    """
    return ModelTimeoutMiddleware(timeout=timeout, enabled=enabled)


def build_breaker_middleware(
    *,
    breaker: CircuitBreaker,
    enabled: bool = True,
) -> BreakerMiddleware:
    """构造熔断中间件。

    ⚠️ 参数**没有默认值**（``enabled`` 除外）。调用方手上一定有熔断器，
    强制显式传入，避免出现「顺手 new 了一个」这种把共享语义破坏掉的写法。

    Args:
        breaker (`CircuitBreaker`): 进程内共享的熔断器。
        enabled (`bool`): 是否启用。

    Returns:
        `BreakerMiddleware`: 构造好的中间件。
    """
    return BreakerMiddleware(breaker=breaker, enabled=enabled)


__all__ = [
    "BreakerMiddleware",
    "ModelCallTimeout",
    "ModelTimeoutMiddleware",
    "build_breaker_middleware",
    "build_model_timeout_middleware",
]
