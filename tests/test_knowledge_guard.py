# -*- coding: utf-8 -*-
"""检索护栏（``src/knowledge/guard.py``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    框架的 ``MilvusLiteStore.search`` 没有任何超时（它把
    ``self.get_client().search`` 丢进 ``asyncio.to_thread``），而
    ``pymilvus`` 又把 ``timeout=None`` 原样交给 gRPC —— 于是
    **Milvus 卡住 = 检索永不返回 = 对话请求永不返回**。
    2026-10-03 在本机 12 容器 tracing 档上真实发生过：``make smoke`` 的
    端到端对话只收到 ``REPLY_START``，60 秒预算耗尽。

    本文件的用例盯着六件事，每一件都对应一个**只在故障时才会暴露**的
    失败形态：

      1. **有截止时间**。慢检索必须在超时预算内失败，而不是挂着。
         判据里带**墙钟断言** —— 一个「抛了超时异常但等了 5 秒才抛」的
         实现能骗过 ``pytest.raises``，骗不过 ``elapsed``。
      2. **超时不堆积**。连续超时之后熔断器要开路，而且开路时必须
         **不再调用底层**。这一条是护栏的真正价值：``to_thread`` 的线程
         追不回来，唯一能做的就是别再产生新的卡死线程。
      3. **记账要准**。只有「下游不可用」（超时 / 非调用方错误的 pymilvus
         异常）才计入熔断；调用方自己的错误（集合不存在、filter 非法…）
         不计 —— 否则熔断器会指错方向，把我们的 bug 记成 Milvus 的账。
      4. **写路径与取消不受影响**。``insert`` / ``delete`` 原样透传
         （离线脚本要能暴露问题），``CancelledError`` 不算下游失败。
         ⚠️ 但「不记账」≠「可以不给熔断器销账」：被取消 / 调用方错误的
         那次调用若正握着 HALF_OPEN 的唯一试探名额，不归还就会把熔断器
         **永久**卡死在半开态（见下面第二节末尾两条用例）。
      5. **读操作的覆盖面**。``_GUARDED_OPERATIONS`` 里每个名字都必须
         真的被实现、且真的被箍住 —— 尤其是 ``has_collection``：它挂在
         框架 ``KnowledgeBase.search() → ensure_collection()`` 的必经之路上
         （``agentscope/rag/_knowledge.py:179``），漏掉它 = 漏掉整条检索路径。
      6. **建连不许占着事件循环**。``get_client()`` 是 ``to_thread`` 的
        实参，默认在事件循环线程上求值；它一旦同步阻塞，``wait_for``
         的超时回调排不上队，护栏在纸面上存在、实际整台服务假死。

    ⚠️ 本文件**不重复测熔断器的状态机**（那是
    ``tests/test_llm_factory.py`` 的活），只测「护栏有没有把熔断器
    接对位置」。
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import pytest

from src.config import Settings
from src.knowledge import guard as guard_mod
from src.knowledge.guard import (
    _GUARDED_OPERATIONS,
    _MAX_IN_FLIGHT,
    GuardedVectorStore,
    VectorSearchTimeout,
    guard_vector_store,
)
from src.knowledge.store import build_vector_store
from src.llm.breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState


# ==============================================================================
# 测试替身
# ==============================================================================
class _FakeStore:
    """一个可控的向量库替身：想慢就慢，想炸就炸，并记录被调了几次。

    ⚠️ 与框架的 ``MilvusLiteStore`` 只保持**鸭子类型**一致：护栏是代理，
    不继承任何东西，所以替身也不需要继承。这反过来也验证了
    「护栏不依赖被包裹对象的具体类型」。
    """

    def __init__(
        self,
        *,
        delay: float = 0.0,
        raises: BaseException | None = None,
        result: Any = "ok",
    ) -> None:
        self.delay = delay
        self.raises = raises
        self.result = result
        self.calls: list[str] = []
        #: 此刻正在 ``_work`` 里的调用数，以及历史峰值。用来验证
        #: ``src/knowledge/guard.py`` 的在飞闸门（``_gate()``）真的在限流 ——
        #: 只断言「调用次数」是测不出闸门的：闸门放开还是收紧，
        #: 调用次数都一样，只有**同时在飞的数量**会变。
        self.in_flight = 0
        self.peak_in_flight = 0

    async def _work(self, name: str) -> Any:
        self.calls.append(name)
        self.in_flight += 1
        self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.raises is not None:
                raise self.raises
            return self.result
        finally:
            self.in_flight -= 1

    async def search(self, *args: Any, **kwargs: Any) -> Any:
        return await self._work("search")

    async def list_documents(self, *args: Any, **kwargs: Any) -> Any:
        return await self._work("list_documents")

    async def list_chunks(self, *args: Any, **kwargs: Any) -> Any:
        return await self._work("list_chunks")

    # ⚠️ 它们也在护栏的覆盖清单里：框架的 ``search`` 每次先问
    # ``has_collection`` 集合在不在，答「不在」就直接 ``create_collection``
    # （``agentscope/rag/_knowledge.py:179``）—— 也就是说这两个名字上「不像读」的方法
    # 实际上会在检索路径上被调用。替身必须实现，否则参数化的
    # 「每个被声明的操作都要有界」用例会因为替身缺方法而失败，
    # 掩盖真正的问题。
    async def has_collection(self, *args: Any, **kwargs: Any) -> Any:
        return await self._work("has_collection")

    async def create_collection(self, *args: Any, **kwargs: Any) -> Any:
        return await self._work("create_collection")

    # ---- 纯离线写路径：刻意慢得到、也炸得出，用来证明它们**没有**被护栏箍住 ----
    async def insert(self, *args: Any, **kwargs: Any) -> Any:
        return await self._work("insert")

    async def delete(self, *args: Any, **kwargs: Any) -> Any:
        return await self._work("delete")


def _fresh_breaker(**kwargs: Any) -> CircuitBreaker:
    """造一个**本次用例专用**的熔断器。

    ⚠️ 必须每次新建：``src.knowledge.guard`` 里那个熔断器是**进程级共享**的
    （这是设计约束，见模块文档），若用例之间共用它，前一个用例留下的
    失败计数会让后一个用例莫名收到 ``CircuitBreakerOpen`` —— 而失败信息
    会指向被测代码，与真正的原因无关。
    """
    params: dict[str, Any] = {"failure_threshold": 2, "recovery_seconds": 60.0}
    params.update(kwargs)
    return CircuitBreaker(**params)


# ==============================================================================
# 一、截止时间：慢检索必须有界地失败
# ==============================================================================
async def test_a_hung_search_is_abandoned_within_the_budget() -> None:
    """★★★ 检索卡住时，护栏在超时预算内**放弃**，而不是陪着一起挂。

    ⚠️ 断言里那个墙钟上限是本用例的核心。只断言「抛了
    ``VectorSearchTimeout``」是不够的：一个把 ``wait_for`` 用在错误位置
    （比如把整个 ``to_thread`` 的**创建**包起来而不是 await 包起来）的
    实现照样会抛超时异常，但会先老老实实等完那 5 秒 —— 对使用者而言，
    那仍然是「对话卡死 5 秒」，与没有护栏的差别只是「最后会报错」。
    """
    inner = _FakeStore(delay=5.0)
    store = GuardedVectorStore(inner, timeout=0.05, breaker=_fresh_breaker())

    started = time.monotonic()
    with pytest.raises(VectorSearchTimeout) as excinfo:
        await store.search("aligo_travel_policy", query_vector=[0.0] * 8)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, (
        f"超时预算 0.05s，却等了 {elapsed:.2f}s 才返回 —— 护栏没有真正生效。"
    )
    assert "search" in str(excinfo.value)
    assert "0.05" in str(excinfo.value)


async def test_the_timeout_is_also_a_plain_timeout_error() -> None:
    """``VectorSearchTimeout`` 必须能被 ``except TimeoutError`` 接住。

    ⚠️ 这不是形式主义：调用方（我们的长期记忆、框架的 RAG 中间件）
    处理的是宽口径异常，但**新的**调用点未必要认识本模块。
    继承内置 ``TimeoutError`` 让「忘了 import 本模块」也不至于漏掉超时。
    """
    inner = _FakeStore(delay=1.0)
    store = GuardedVectorStore(inner, timeout=0.02, breaker=_fresh_breaker())

    with pytest.raises(TimeoutError):
        await store.search("c", query_vector=[0.0])


async def test_a_prompt_read_is_returned_unchanged() -> None:
    """正常路径必须**逐字**返回底层结果，护栏不改变任何语义。"""
    inner = _FakeStore(result=[{"id": 1}])
    store = GuardedVectorStore(inner, timeout=1.0, breaker=_fresh_breaker())

    assert await store.search("c", query_vector=[0.0]) == [{"id": 1}]
    assert await store.list_documents("c") == [{"id": 1}]
    assert await store.list_chunks("c", "doc") == [{"id": 1}]
    # 替身对所有操作返回同一个值 —— 这里看的正是「原样返回」。
    assert await store.has_collection("c") == [{"id": 1}]
    assert await store.create_collection("c", dimensions=1024) == [{"id": 1}]
    assert inner.calls == [
        "search",
        "list_documents",
        "list_chunks",
        "has_collection",
        "create_collection",
    ]


# ⚠️ 参数化直接吃 ``_GUARDED_OPERATIONS``：这份元组是**声明的覆盖清单**，
# 而本用例把它与**实际实现**对齐。两处最容易出的错都能被抓住：
#   1. 往清单里加了名字却忘了写方法（方法缺失 → 断言 __dict__ 失败）；
#   2. 写了名字、但忘了加进清单（本条抓不到 —— 反过来由「清单里的每个
#      名字都必须有界」+ 代码评审覆盖；清单短了不会让谁静默失效，
#      因为方法在类里是显式的）。
@pytest.mark.parametrize("operation", _GUARDED_OPERATIONS)
async def test_every_declared_guarded_operation_is_actually_wrapped(
    operation: str,
) -> None:
    """★★★ 清单里每个读操作都必须**真的**被箍住，一个都不能漏。

    ⚠️ 这条用例的存在理由是一次真实审计发现的洞：``has_collection``
    只写在模块文档的「不箍」列表里，而它恰恰挂在框架
    ``KnowledgeBase.search() → ensure_collection()`` 的必经之路上
    （``agentscope/rag/_knowledge.py:179``）—— 卡住它，用户看到的现象与卡住
    ``search`` **完全一样**，但护栏对它视而不见。
    """
    assert operation in GuardedVectorStore.__dict__, (
        f"{operation!r} 列在 _GUARDED_OPERATIONS 里，但 GuardedVectorStore "
        f"没有显式实现它 —— 它会经 __getattr__ 裸透传出去，等于没有护栏。"
    )

    inner = _FakeStore(delay=5.0)
    store = GuardedVectorStore(inner, timeout=0.05, breaker=_fresh_breaker())

    started = time.monotonic()
    with pytest.raises(VectorSearchTimeout) as excinfo:
        await getattr(store, operation)("aligo_travel_policy")
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"{operation} 没有被截止时间箍住（等了 {elapsed:.2f}s）。"
    assert operation in str(excinfo.value), "异常信息里应能看出是哪个操作超时。"
    assert inner.calls == [operation]


async def test_a_hung_create_collection_is_bounded_on_the_read_path() -> None:
    """★★★ 集合不存在时，「先建集合」这一步也必须在预算内失败。

    ⚠️ 这条守的是 ``_GUARDED_OPERATIONS`` 里**最反直觉**的那个名字：
    框架的 ``KnowledgeBase.ensure_collection()`` 在 ``has_collection``
    答「没有」时会调 ``create_collection``（``agentscope/rag/_knowledge.py:179``），
    而那是由 ``search()`` 触发的（``:237``）。只箍读、不箍写的版本
    在这里会把「集合不存在」这个**可预期的**状态，变成整个服务在
    唯一没有护栏的那一步上假死。

    断言分两半：既要有界地失败，也要让熔断器**数得到这一笔账** ——
    否则连续失败永远攒不够，检索会一直往一个卡死的 Milvus 上重试。
    """
    inner = _FakeStore(delay=5.0)
    breaker = _fresh_breaker(failure_threshold=1)
    store = GuardedVectorStore(inner, timeout=0.05, breaker=breaker)

    started = time.monotonic()
    with pytest.raises(VectorSearchTimeout):
        await store.create_collection("aligo_travel_policy", dimensions=1024)
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, f"create_collection 没有截止时间（等了 {elapsed:.2f}s）。"
    assert breaker.state is CircuitState.OPEN, (
        "建集合卡住没有被记成下游失败 —— 熔断器数不到账，就不会停止重试。"
    )


async def test_the_in_flight_gate_caps_how_many_calls_run_concurrently() -> None:
    """★★★ 同时**在飞**的受护栏调用数有上界（``_MAX_IN_FLIGHT``）。

    ⚠️ 为什么这条与「超时有界」是两件事：超时放弃的是 await，线程还在跑，
    而 ``asyncio`` 的默认执行器只有 ``min(32, CPU+4)`` 个工位 ——
    本地 ONNX 向量化（``src/web_embedding/local.py``）也走它。
    没有闸门，一次「Milvus 刚挂 + 若干用户同时提问」就能在一瞬间把工位
    占满，把向量化一起饿死。

    断言的是**峰值并发**而不是调用次数：闸门放宽或收紧，调用次数都一样，
    只有在飞数量会变 —— 所以只数调用次数的用例测不出闸门是否存在。

    ⚠️⚠️ **这条用例测的是协程，不是线程** —— 别把它读成「遗留线程 ≤ 4」。
    ``_FakeStore`` 数的 ``in_flight`` 是「进了 ``_work`` 还没出来的协程数」，
    而闸门对协程数天然封顶 4，所以**它在结构上不可能发现线程侧的超限**：
    真实的闸门在超时时归还名额，而那时线程还在下游卡着，下一批请求会立刻
    占满名额再起线程。测试替身没有真线程，这个差异在这里看不见。
    留给后来者的话：想验证线程上界，得用真的执行器（``to_thread`` + 计数
    线程），别在这个替身上加断言 —— 它证明不了。
    """
    inner = _FakeStore(delay=5.0)
    # 阈值给得足够高，免得熔断器在闸门生效之前就把后续调用拒了 ——
    # 那会让「并发被限住」与「调用被熔断挡下」变成同一个现象。
    store = GuardedVectorStore(
        inner,
        timeout=0.05,
        breaker=_fresh_breaker(failure_threshold=100),
    )

    attempts = _MAX_IN_FLIGHT + 4
    results = await asyncio.gather(
        *(store.search("c", query_vector=[0.0]) for _ in range(attempts)),
        return_exceptions=True,
    )

    assert all(isinstance(r, VectorSearchTimeout) for r in results), results
    assert inner.peak_in_flight <= _MAX_IN_FLIGHT, (
        f"同时在飞的调用达到 {inner.peak_in_flight}，超过闸门上限 "
        f"{_MAX_IN_FLIGHT} —— 一次突发就能占满默认执行器的工位。"
    )
    # 反例：闸门不能把并发压到 1（那会把检索变成串行，属于功能退化）。
    assert inner.peak_in_flight == _MAX_IN_FLIGHT, (
        f"峰值并发只有 {inner.peak_in_flight}，闸门上限是 {_MAX_IN_FLIGHT} —— "
        "限得过狠，检索被串行化了。"
    )


async def test_the_in_flight_gate_is_returned_when_a_call_is_abandoned() -> None:
    """★★★ 超时/取消必须**归还**名额，否则一次抖动会永久废掉检索。

    ⚠️ 这是闸门最危险的一种写法：名额代表「同时在飞」。若在取消路径上
    不归还，那么连续几次超时之后名额就被永久耗尽，此后**每一次**检索都在
    闸门前等到超时 —— 包括 Milvus 已经恢复之后，表现为「向量库好好的，
    但检索永远失败，只有重启进程才行」。
    """
    inner = _FakeStore(delay=5.0)
    store = GuardedVectorStore(
        inner,
        timeout=0.05,
        breaker=_fresh_breaker(failure_threshold=100),
    )

    # 先把名额全部用光（每一次都会超时并被放弃）。
    for _ in range(_MAX_IN_FLIGHT * 3):
        with pytest.raises(VectorSearchTimeout):
            await store.search("c", query_vector=[0.0])

    # 现在让底层变快：若名额被归还了，这一次应当**立刻成功**。
    inner.delay = 0.0
    result = await asyncio.wait_for(store.search("c", query_vector=[0.0]), 1.0)

    assert result == "ok", "名额没有归还 —— 恢复之后的检索仍然拿不到工位。"


# ==============================================================================
# 二、熔断：超时之后**不再**产生新的卡死调用
# ==============================================================================
async def test_consecutive_timeouts_open_the_breaker_and_stop_calling() -> None:
    """★★★ 连续超时后开路，且开路期间**一次底层调用都不发**。

    ⚠️ 只断言「抛了 ``CircuitBreakerOpen``」会漏掉最要紧的一半：
    必须同时断言底层被调用的**次数没有增加**。护栏存在的意义就是
    「不再往一个已经卡死的 Milvus 上堆线程」（``asyncio.to_thread``
    的线程取消不掉，只能不再新建），所以「有没有真的不打网络」是
    这条用例的全部价值。
    """
    inner = _FakeStore(delay=5.0)
    breaker = _fresh_breaker(failure_threshold=2)
    store = GuardedVectorStore(inner, timeout=0.02, breaker=breaker)

    for _ in range(2):
        with pytest.raises(VectorSearchTimeout):
            await store.search("c", query_vector=[0.0])
    assert inner.calls == ["search", "search"]
    assert breaker.state is CircuitState.OPEN

    with pytest.raises(CircuitBreakerOpen):
        await store.search("c", query_vector=[0.0])

    assert inner.calls == ["search", "search"], (
        "熔断器已经开路，护栏却仍然调用了底层 —— "
        "这正是「Milvus 挂了之后每个请求都还在往上堆卡死线程」的原因。"
    )


async def test_a_success_resets_the_failure_count() -> None:
    """成功一次即清零：熔断看的是**连续**失败，不是累计失败。

    ⚠️ 反过来（累计）的后果是：一个偶发慢请求的 Milvus，跑上一天之后
    总会被熔断，而每次熔断都停掉 30 秒的检索 —— 用户看到的是
    「知识库时灵时不灵」，且没有任何一条日志指向真正的原因。
    """
    inner = _FakeStore(delay=0.05)
    breaker = _fresh_breaker(failure_threshold=3)
    store = GuardedVectorStore(inner, timeout=0.02, breaker=breaker)

    # 失败、失败、**成功**、失败、失败 —— 成功那一次把计数清零，
    # 于是此刻只有 2 次连续失败，还不到 3，熔断器必须仍然放行。
    for _ in range(2):
        with pytest.raises(VectorSearchTimeout):
            await store.search("c", query_vector=[0.0])

    inner.delay = 0.0
    assert await store.search("c", query_vector=[0.0]) == "ok"

    inner.delay = 0.05
    for _ in range(2):
        with pytest.raises(VectorSearchTimeout):
            await store.search("c", query_vector=[0.0])

    assert breaker.state is CircuitState.CLOSED, (
        "成功一次之后计数没有清零 —— 熔断器把「累计失败」当成了「连续失败」。"
    )
    inner.delay = 0.0
    assert await store.search("c", query_vector=[0.0]) == "ok"


# ==============================================================================
# 三、记账：只有「下游不可用」才计入熔断
# ==============================================================================
async def test_a_caller_side_error_is_not_counted_as_a_failure() -> None:
    """集合不存在之类的**我们自己**的错误，不计入熔断，且原样上抛。

    ⚠️ 两类错误的处置完全相反：下游不可用要熔断（再试也没用），
    调用方错误要暴露（那是 bug）。把它们记成同一笔账，会让熔断器
    在「我们的代码写错了」时把 Milvus 判成故障 —— 运维去查 Milvus，
    什么也查不到。
    """
    inner = _FakeStore(raises=ValueError("集合 aligo_travel_policy 不存在"))
    breaker = _fresh_breaker(failure_threshold=2)
    store = GuardedVectorStore(inner, timeout=1.0, breaker=breaker)

    for _ in range(5):
        with pytest.raises(ValueError):
            await store.search("c", query_vector=[0.0])

    assert breaker.state is CircuitState.CLOSED, (
        "调用方错误被记成了下游失败 —— 熔断器会在我们自己的 bug 上误熔断。"
    )
    assert len(inner.calls) == 5, "异常被吞掉或被改写成了别的类型。"


async def test_a_milvus_error_counts_as_a_downstream_failure() -> None:
    """pymilvus 自己的异常（连不上、超时…）**要**计入熔断。"""
    pytest.importorskip("pymilvus")
    from pymilvus.exceptions import MilvusException

    inner = _FakeStore(raises=MilvusException(message="Fail connecting to server"))
    breaker = _fresh_breaker(failure_threshold=2)
    store = GuardedVectorStore(inner, timeout=1.0, breaker=breaker)

    for _ in range(2):
        with pytest.raises(MilvusException):
            await store.search("c", query_vector=[0.0])

    assert breaker.state is CircuitState.OPEN
    with pytest.raises(CircuitBreakerOpen):
        await store.search("c", query_vector=[0.0])


async def test_a_milvus_input_error_is_not_counted_as_a_downstream_failure() -> None:
    """服务端**判定为调用方输入错误**的 Milvus 异常不计入熔断。

    ⚠️ 「pymilvus 异常 ⇒ Milvus 挂了」是错的：``MilvusException`` 带一个
    ``is_input_error`` 标志（由 ``from_status`` 从服务端 Status 的
    ``extra_info`` 解出），集合不存在、维度不匹配、filter 非法都属于这一类。
    它们说明对端**活着并回了话** —— 记成下游失败会让熔断器指错方向，
    正是模块文档约束 2 要防的事。
    """
    pytest.importorskip("pymilvus")
    from pymilvus.exceptions import MilvusException

    inner = _FakeStore(
        raises=MilvusException(
            code=1100,
            message="invalid filter expression",
            is_input_error=True,
        ),
    )
    breaker = _fresh_breaker(failure_threshold=2)
    store = GuardedVectorStore(inner, timeout=1.0, breaker=breaker)

    for _ in range(5):
        with pytest.raises(MilvusException):
            await store.search("c", query_vector=[0.0])

    assert breaker.state is CircuitState.CLOSED, (
        "服务端明确标了 is_input_error 的异常被记成了下游失败 —— "
        "熔断器会在「我们传错了参数」时把 Milvus 判成故障。"
    )


async def test_a_cancelled_search_is_not_counted_as_a_failure() -> None:
    """被取消的检索不算下游失败 —— 那是调用方不想要了，不是 Milvus 的错。

    ⚠️ ``asyncio.CancelledError`` 继承自 ``BaseException``，用
    ``except Exception`` 是抓不到的；本用例挡住的是「护栏把所有
    BaseException 都记成失败」——那会让用户在页面上按「停止生成」
    三次就把检索熔断掉。
    """
    inner = _FakeStore(delay=5.0)
    breaker = _fresh_breaker(failure_threshold=1)
    store = GuardedVectorStore(inner, timeout=30.0, breaker=breaker)

    task = asyncio.create_task(store.search("c", query_vector=[0.0]))
    await asyncio.sleep(0.02)  # 让它真的进到 await 里
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert breaker.state is CircuitState.CLOSED


async def test_a_cancelled_probe_returns_its_half_open_slot() -> None:
    """★★★ 被取消的**试探请求**必须把 HALF_OPEN 的唯一名额还回去。

    ⚠️ 这条用例盯着一个「一旦发生就无法自愈」的故障（审计发现的洞）：
    ``allow_request()`` 放行试探请求时把 ``_probe_in_flight`` 置为 True，
    而只有 ``record_success`` / ``record_failure`` 会清掉它。取消属于
    「不记账」的路径 —— 于是名额**永远**被占着，熔断器卡死在 HALF_OPEN：
    此后每个请求都被 ``CircuitBreakerOpen(name, 0.0)`` 拒绝，
    检索能力被永久关闭，直到进程重启。

    ⚠️ 断言不是「state 是 HALF_OPEN」这种描述现状的话，而是
    「下一次调用能不能真的成功」—— 后者才是用户看得见的东西。
    """
    inner = _FakeStore(delay=5.0)
    breaker = _fresh_breaker(failure_threshold=1, recovery_seconds=0.05)
    store = GuardedVectorStore(inner, timeout=0.02, breaker=breaker)

    # 第一步：制造一次超时 → 熔断开路。
    with pytest.raises(VectorSearchTimeout):
        await store.search("c", query_vector=[0.0])
    assert breaker.state is CircuitState.OPEN

    # 第二步：等冷却期过，再起一个请求 —— 它是**试探请求**，握着唯一名额，
    # 然后被取消（用户点了「停止生成」/ 客户端断开都会走这条路）。
    await asyncio.sleep(0.06)
    task = asyncio.create_task(store.search("c", query_vector=[0.0]))
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # 第三步（关键）：下一个请求必须被**放行**并成功，把熔断器带回 CLOSED。
    # 若名额没归还，这里收到的是 retry_after=0.0 的 CircuitBreakerOpen，
    # 且此后永远是它。
    inner.delay = 0.0
    assert await store.search("c", query_vector=[0.0]) == "ok", (
        "被取消的试探没有归还 HALF_OPEN 名额 —— 熔断器被永久卡死，"
        "此后每次检索都会被拒绝，且没有任何自愈路径。"
    )
    assert breaker.state is CircuitState.CLOSED


async def test_a_caller_error_during_a_probe_returns_its_half_open_slot() -> None:
    """同样地，试探请求抛出**调用方错误**（不记账）时也要归还名额。

    ⚠️ 「不记账」与「不归还」是两件事：前者是**计数的正确性**（不能把
    我们的 bug 记成 Milvus 挂了），后者是**状态机的活性**（名额是独占
    资源，借了就得还）。审计时这两件事很容易被当成一件。
    """
    inner = _FakeStore(delay=5.0)
    breaker = _fresh_breaker(failure_threshold=1, recovery_seconds=0.05)
    store = GuardedVectorStore(inner, timeout=0.02, breaker=breaker)

    with pytest.raises(VectorSearchTimeout):
        await store.search("c", query_vector=[0.0])
    assert breaker.state is CircuitState.OPEN

    await asyncio.sleep(0.06)
    inner.delay = 0.0
    inner.raises = ValueError("filter 表达式非法")  # 我们自己传错的参数
    with pytest.raises(ValueError):
        await store.search("c", query_vector=[0.0])

    inner.raises = None
    assert await store.search("c", query_vector=[0.0]) == "ok"
    assert breaker.state is CircuitState.CLOSED


# ==============================================================================
# 四、写路径与透传：护栏不越界
# ==============================================================================
async def test_writes_bypass_the_guard_entirely() -> None:
    """``insert`` 等写操作不受超时与熔断影响，原样透传。

    ⚠️ 写路径只出现在初始化 / 灌数据脚本里（离线、有人盯着）。
    把它也箍住，症状会变成「灌了一半的数据 + 一个被悄悄吞掉的超时」，
    比直接卡住更难查。
    """
    inner = _FakeStore(delay=5.0)
    breaker = _fresh_breaker(failure_threshold=1)
    store = GuardedVectorStore(inner, timeout=0.02, breaker=breaker)

    # 先把熔断器打开（读路径超时一次即可）。
    with pytest.raises(VectorSearchTimeout):
        await store.search("c", query_vector=[0.0])
    assert breaker.state is CircuitState.OPEN

    # 写路径：既要能穿透熔断，也要**不受 timeout 限制**。
    # ⚠️ delay 必须留在「远大于 timeout、但用例等得起」的值上（0.2s vs 0.02s）。
    # 早期版本在这里先把 delay 归零再断言 —— 那样一来「写操作被超时掐断」
    # 这个 bug 根本不可能被测出来，用例只是走了个过场。
    inner.delay = 0.2
    started = time.monotonic()
    assert await store.insert("c", records=[]) == "ok"
    elapsed = time.monotonic() - started

    assert elapsed >= 0.2, (
        f"insert 只用了 {elapsed:.3f}s，没跑满它自己的 0.2s —— "
        "它被护栏的超时（0.02s）掐断了；写路径必须原样透传。"
    )
    assert inner.calls[-1] == "insert"


async def test_unknown_attributes_are_transparent() -> None:
    """未被箍住的属性一律透传 —— 护栏对调用方是**透明**的。

    ⚠️ 这是 :class:`GuardedVectorStore` 用代理而不是继承换来的代价：
    必须保证「没显式实现的东西照常可用」。本用例盯的是几类真实用法：
    框架的私有属性（``_metric_type``）、探针用的 ``get_client()``、
    以及显式调用的 ``__aexit__``（``src/knowledge/store.py`` 的探针
    就是这么关客户端的）。
    """

    class _WithExtras(_FakeStore):
        _metric_type = "COSINE"
        _uri = "http://milvus:19530"

        def get_client(self) -> str:
            return "client"

        async def __aexit__(self, *exc: Any) -> None:
            self.calls.append("aexit")

    inner = _WithExtras()
    store = GuardedVectorStore(inner, timeout=1.0, breaker=_fresh_breaker())

    assert store._metric_type == "COSINE"
    assert store._uri == "http://milvus:19530"
    assert store.get_client() == "client"
    await store.__aexit__(None, None, None)
    assert inner.calls == ["aexit"]


async def test_async_with_yields_the_guarded_store_not_the_bare_one() -> None:
    """``async with store`` 必须可用，且拿到的是**护栏自己**。

    ⚠️ 两个坑各对应一条断言：
      · ``async with`` 的特殊方法查找走**类型**、不经过 ``__getattr__``，
        纯代理会直接 ``TypeError``（护栏必须显式实现 ``__aenter__``）；
      · ``__aenter__`` 若把底层的返回值（往往是裸 store）原样返回，
        一进 ``with`` 块就把护栏绕过去了 —— 那是最坏的一种「静默失效」。
    """

    class _WithLifecycle(_FakeStore):
        async def __aenter__(self) -> "_WithLifecycle":
            self.calls.append("aenter")
            return self

        async def __aexit__(self, *exc: Any) -> None:
            self.calls.append("aexit")

    inner = _WithLifecycle()
    store = GuardedVectorStore(inner, timeout=1.0, breaker=_fresh_breaker())

    async with store as entered:
        assert entered is store, "async with 拿到的是裸 store —— 护栏被绕过了。"
        assert await entered.search("c") == "ok"

    assert inner.calls == ["aenter", "search", "aexit"]


async def test_the_breaker_is_process_wide() -> None:
    """两个 store（知识库 / 长期记忆）共享同一个熔断器。

    ⚠️ 应用启动时**构造了两个** store（``src/knowledge/__init__.py`` 与
    ``src/memory/__init__.py``）。若各自持一个熔断器，同一场 Milvus 故障
    要被分别数够阈值才熔断 —— 熔断点被推迟，而这段时间里每个请求
    都还在卡。本用例用一个共享的熔断器打开后，验证**另一个** store
    立刻拒绝且不调用自己的底层。
    """
    inner_a = _FakeStore(delay=5.0)
    inner_b = _FakeStore(result="b")
    shared = _fresh_breaker(failure_threshold=1)
    store_a = GuardedVectorStore(inner_a, timeout=0.02, breaker=shared)
    store_b = GuardedVectorStore(inner_b, timeout=0.02, breaker=shared)

    with pytest.raises(VectorSearchTimeout):
        await store_a.search("c", query_vector=[0.0])

    with pytest.raises(CircuitBreakerOpen):
        await store_b.search("c", query_vector=[0.0])

    assert inner_b.calls == [], "另一个 store 在熔断期间仍然调用了底层。"


async def test_the_default_breaker_is_the_shared_module_level_one() -> None:
    """不传 ``breaker`` 时用的就是模块级那一个（生产路径）。

    ⚠️ 这条断言把「共享实例」从**口头约定**变成**可验证的事实**：
    只要有人把 ``guarded_vector_store`` 改成「每个实例自带一个熔断器」，
    这里立刻红。
    """
    store = guard_vector_store(_FakeStore(), timeout=1.0)

    assert store._breaker is guard_mod.get_vector_store_breaker()


# ==============================================================================
# 五、建连：预热必须离开事件循环，并且一样有截止时间
# ==============================================================================
async def test_the_client_is_warmed_on_a_worker_thread() -> None:
    """★★★ ``get_client()`` 必须在**非事件循环**的线程里被调用。

    ⚠️ 这是「护栏有没有用」的前提条件，不是性能优化：框架的读方法写成
    ``asyncio.to_thread(self.get_client().search, ...)`` —— ``get_client()``
    是 ``to_thread`` 的实参，在事件循环线程上求值。它第一次调用会同步
    建连（Milvus 不可达时约 10s），这期间事件循环连 ``wait_for`` 的
    超时回调都排不上队 —— 于是**护栏形同虚设，整台服务假死**，
    与 2026-10-03 记录的症状一模一样。
    """
    loop_thread = threading.get_ident()

    class _ClientfulStore(_FakeStore):
        def __init__(self, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            # 复刻框架的缓存字段（``agentscope/rag/_vdb/_milvus_lite.py:75``）。
            self._client: object | None = None
            self.client_threads: list[int] = []

        def get_client(self) -> object:
            self.client_threads.append(threading.get_ident())
            if self._client is None:
                self._client = object()
            return self._client

    inner = _ClientfulStore(result="ok")
    store = GuardedVectorStore(inner, timeout=1.0, breaker=_fresh_breaker())

    assert await store.search("c") == "ok"
    assert inner.client_threads, (
        "护栏没有预热客户端 —— get_client() 会在事件循环上同步建连。"
    )
    assert all(tid != loop_thread for tid in inner.client_threads), (
        "get_client() 是在事件循环线程里被调用的 —— 建连会阻塞整个进程。"
    )

    # 已建连之后不该再派线程（快路径）：每次读都额外派一次线程是没必要的
    # 开销，而且会在卡死线程堆积时占用线程池名额。
    inner.client_threads.clear()
    assert await store.list_documents("c") == "ok"
    assert inner.client_threads == []


async def test_a_hung_client_connect_is_also_bounded() -> None:
    """建连本身卡住时，护栏同样在超时预算内放弃（且不去执行检索）。

    ⚠️ 预热与真正的调用共用一个超时预算：只箍检索、不箍建连的话，
    最开始的（也是最容易卡住的）那一步反而是无界的。
    """
    release = threading.Event()

    class _UnconnectableStore(_FakeStore):
        def __init__(self) -> None:
            super().__init__(result="ok")
            self._client: object | None = None

        def get_client(self) -> object:
            # 同步阻塞 —— 复刻 pymilvus 建连卡住的样子。
            # ⚠️ 用 Event 而不是 sleep 常值：线程拦不住（Python 无法中断
            # 线程），用例结束时得放它走，否则事件循环关闭时会等它。
            release.wait(5.0)
            return object()

    inner = _UnconnectableStore()
    store = GuardedVectorStore(inner, timeout=0.05, breaker=_fresh_breaker())

    started = time.monotonic()
    with pytest.raises(VectorSearchTimeout):
        await store.search("c")
    elapsed = time.monotonic() - started

    assert elapsed < 1.0
    assert inner.calls == [], "建连都超时了，检索不该被执行。"
    release.set()
    await asyncio.sleep(0.05)  # 放走那个仍在阻塞的线程


# ==============================================================================
# 六、装配：配置 → 护栏
# ==============================================================================
def test_a_non_positive_timeout_is_refused() -> None:
    """超时取 0 或负数必须**当场报错**。

    ⚠️ 0 的效果是「每次检索都超时」，也就是静默地把检索能力关掉。
    这种配置必须让进程起不来，而不是在线上表现为「知识库好像没数据」。
    """
    for bad in (0, -1.0):
        with pytest.raises(ValueError):
            GuardedVectorStore(_FakeStore(), timeout=bad)


def test_the_built_store_takes_its_timeout_from_the_config(
    settings: Settings,
) -> None:
    """``build_vector_store`` 造出来的 store 必须带上配置里的超时。

    ⚠️ 这条用例挡的是「配置项加了、schema 加了、但装配时忘了传」——
    那种情况下 ``search_timeout_seconds`` 改到 0.1 也不会有任何效果，
    而使用者会以为自己已经调过参数了。

    ⚠️ 所以这里刻意把配置值改成**与默认值不同**的一个数：若断言只用
    默认的 5.0，一个「写死 5.0、根本没读配置」的实现也能全绿 ——
    用例必须能抓住它声称要抓的那个 bug。
    """
    settings.milvus.search_timeout_seconds = 0.75  # 默认是 5.0，刻意区分开
    store = build_vector_store(settings)

    assert isinstance(store, GuardedVectorStore)
    assert store._timeout == 0.75, (
        f"装配出来的超时是 {store._timeout!r}，而配置说 0.75 —— "
        "要么没读配置，要么读错了字段。"
    )
    # 装配路径必须用共享熔断器，而不是临时新建一个。
    assert store._breaker is guard_mod.get_vector_store_breaker()


def test_the_gate_survives_a_second_event_loop() -> None:
    """★★★ 闸门不能绑死在**第一个**事件循环上（真跑两遍 ``asyncio.run``）。

    ⚠️ 这是一条**回归用例**。曾经的写法是模块级
    ``_GATE = asyncio.Semaphore(_MAX_IN_FLIGHT)``，而 ``asyncio`` 原语在
    **第一次发生竞争**（需要挂 waiter）时会惰性绑定当时那个 loop：
    之后再换一个 loop 竞争，直接抛
    ``RuntimeError: ... is bound to a different event loop``。

    一个进程里出现第二个 loop 是常态 —— 每个 asyncio 用例一个、
    脚本里每次 ``asyncio.run`` 一个。而那个 ``RuntimeError`` 在
    ``_bounded`` 里走 ``except BaseException``（不算下游失败、原样上抛），
    再被调用方宽口径的 ``except Exception`` 接住 ⇒ 表现为
    **「检索能力静默消失」**，而不是一条能看出原因的报错 ——
    这正是为什么它值得一条专门的用例。

    ⚠️ 必须**饱和**闸门（并发数 > ``_MAX_IN_FLIGHT``）才能重现：
    并发数没超过名额时 ``Semaphore`` 走快路径、根本不碰 loop 绑定，
    照着写一条不饱和的用例，在旧代码上也会是绿的 ——
    用例必须能抓住它声称要抓的那个 bug。

    ⚠️ 本用例是**同步**函数，因为它要自己开两个事件循环；在
    ``async def`` 里没法再 ``asyncio.run``。
    """

    async def saturate() -> list[Any]:
        """在一个新 loop 里把闸门压到饱和，返回每个调用的结果。"""
        store = GuardedVectorStore(
            _FakeStore(delay=5.0),
            timeout=0.05,
            breaker=_fresh_breaker(failure_threshold=100),
        )
        return await asyncio.gather(
            *(
                store.search("c", query_vector=[0.0])
                for _ in range(_MAX_IN_FLIGHT + 2)
            ),
            return_exceptions=True,
        )

    # 第一遍：把闸门绑到 loop1，随后 loop1 随 asyncio.run 结束而关闭。
    first = asyncio.run(saturate())
    # 第二遍：**关键的一步**。旧写法在这里抛 RuntimeError。
    second = asyncio.run(saturate())

    for label, results in (("第一个 loop", first), ("第二个 loop", second)):
        assert all(isinstance(r, VectorSearchTimeout) for r in results), (
            f"{label} 的调用没有全部以超时返回：{results!r} —— "
            "闸门跨事件循环失效时，这里会混进 RuntimeError"
            "（它会被上层宽口径的 except 吞掉，表现为检索静默消失）。"
        )


class _LazyClientStore:
    """复刻框架 ``MilvusLiteStore`` 的**惰性建连**语义。

    只做两件事，且都是框架真实行为：

      · ``_client`` 是缓存字段（框架在 ``agentscope/rag/_vdb/_milvus_lite.py:75`` 定义），
        护栏会读它做快路径判断；
      · ``get_client()`` 是**同步**函数，且**自己不是并发安全的** ——
        ``if self._client is None: self._client = MilvusClient(...)``。
        护栏把它丢进线程池预热，正是为了不让它阻塞事件循环。

    另外记录被真正执行了几次建连（``clients_created``）—— 这是本用例的
    断言对象：**建连只能启动一次**。
    """

    def __init__(self, *, connect_delay: float = 1.0) -> None:
        self._client: Any = None
        self.connect_delay = connect_delay
        #: 造出了几个「客户端」。>1 就意味着有客户端被覆盖掉、连同
        #: 它自己的连接与后台线程一起泄漏（框架的 ``__aexit__``
        #: 只关当前那个 ``_client``）。
        self.clients_created = 0

    def get_client(self) -> Any:
        """同步建连（在 ``to_thread`` 的工作线程里执行）。"""
        if self._client is None:
            self.clients_created += 1
            time.sleep(self.connect_delay)
            self._client = object()
        return self._client

    async def search(self, *args: Any, **kwargs: Any) -> Any:
        """检索本体。⚠️ 本用例里**不该被走到** —— 建连都没成功。

        但仍必须存在：护栏是 ``_bounded("search", self._inner.search, ...)``，
        它会在调用前就取一次 ``self._inner.search``，缺了属性会直接
        AttributeError，用例就测不到建连那条路径了（这正是第一次写这条
        用例时踩到的坑）。
        """
        raise AssertionError(
            "建连都没完成就走到检索本体了 —— 预热没有生效。",
        )


async def test_a_hung_client_warmup_is_started_only_once() -> None:
    """★★★ 建连**卡住**时，后续请求必须等同一个建连，而不是再起一个。

    ⚠️ 这是一条**回归用例**，挡的是「加了锁但锁不住」这个很隐蔽的写法：
    超时取消的是 ``await``，``async with`` 随之退栈把锁**还回去**，
    可线程里的建连还在跑。下一个调用者拿到锁、发现 ``_client`` 仍是
    ``None``，于是**再建一个**。三个连续超时的检索就能建出三个客户端，
    只有最后一个被留下 —— 前两个连同各自的连接与后台线程一起泄漏，
    而框架的 ``__aexit__`` 只关当前那个 ``_client``。

    ⚠️ 必须**串行**地发两次请求才能稳定重现：并发的两次会一起被
    ``wait_for`` 取消（都堵在锁上），建连次数仍是 1，旧写法也能绿 ——
    那样写出来的用例抓不住它声称要抓的 bug。

    ⚠️ ``connect_delay`` 取 1s 而不是几十秒：它只用来保证「第二次请求
    到达时第一次建连还没结束」，测试结束时要等这个线程收尾，取值过大
    会白白拖慢整套用例。
    """
    inner = _LazyClientStore(connect_delay=1.0)
    store = GuardedVectorStore(
        inner,
        # 超时远小于建连耗时，保证两次请求都「建连还没好就放弃」。
        timeout=0.05,
        breaker=_fresh_breaker(failure_threshold=100),
    )

    # 第一次：建连被卡住，检索超时放弃 —— 但建连线程还在跑。
    first = await asyncio.gather(
        store.search("c", query_vector=[0.0]),
        return_exceptions=True,
    )
    # 让出控制权，确保锁已经释放（旧写法就是在这里失守的）。
    await asyncio.sleep(0)

    # 第二次：又一个请求进来。旧写法会在这里**再建一个**客户端。
    second = await asyncio.gather(
        store.search("c", query_vector=[0.0]),
        return_exceptions=True,
    )

    assert all(isinstance(r, VectorSearchTimeout) for r in first + second), (
        first,
        second,
    )
    assert inner.clients_created == 1, (
        f"建连被启动了 {inner.clients_created} 次 —— 单飞失效。"
        "每次超时都再建一个客户端，只有最后一个被留下，"
        "其余连同它们的连接和后台线程一起泄漏。"
    )
