# -*- coding: utf-8 -*-
"""检索路径的护栏：给「可能永远不返回」的向量库调用加**超时**与**熔断**。

文件职责：
    把 :class:`agentscope.rag.MilvusLiteStore` 的**读操作**包一层，
    让一次卡住的检索变成一次**有界**的失败（超时 + 降级），
    而不是一个永不返回的请求。

上下游依赖：
    - 上游：由 ``src/knowledge/store.py::build_vector_store`` 装配，
      于是**所有**拿到向量库的地方（知识库检索、长期画像语义召回）
      都自动带上护栏 —— 不需要每个调用点自己记得加。
    - 下游：``src.llm.breaker.CircuitBreaker``（复用，不重写第二份熔断器）、
      ``src.config.schema.MilvusSettings.search_timeout_seconds``。

------------------------------------------------------------------------------
为什么需要它：一次真实故障（2026-10-03，本机 12 容器 tracing 档实测）
------------------------------------------------------------------------------
    Milvus standalone 在资源不足时开始丢 etcd 租约（日志里是
    ``Slow etcd operation save`` 8.19s → ``clock offset is huge`` →
    ``the session is expired without activing closing``），进程随后以
    exit 80 退出。**在它退出的前后那几分钟里**，应用侧的对话请求走到了
    「把用户消息向量化 → 拿向量去 Milvus 检索」这一步：

        10:28:09  dashscope 返回 embedding（1024 维，正常）
        10:28:09  之后 —— 应用**再无任何日志**，``/healthz`` 20s 超时，
                  ``make smoke`` 的端到端对话在 60s 预算内只收到
                  ``REPLY_START``，拿不到 ``REPLY_END``

    根因不在我们的检索代码里，而在**它没有截止时间**：

      · ``MilvusLiteStore.search`` 是 ``await asyncio.to_thread(
        self.get_client().search, ...)``（``_vdb/_milvus_lite.py:303-311``），
        **没有 timeout 参数**；
      · 它最终落到 ``pymilvus`` 的 gRPC 调用上，而 ``pymilvus`` 自己在
        建连路径上留了一句注释：``grpc.Future.result(timeout=None) blocks
        indefinitely``（``pymilvus/client/grpc_handler.py:248``）—— 它只在
        **建连**时把 ``None`` 归一到 10s，**检索 RPC 不归一**；
      · 于是 Milvus 卡住 ⇒ search 永远不返回 ⇒ 请求永远不返回。

    2026-10-03 实测（宿主直连一个不可路由地址）：``MilvusClient(uri=...,
    timeout=2.0)`` 把失败时间从 10.5s 压到 2.0s —— 但那**只证明了建连**受
    构造参数约束；检索 RPC 是否也受它约束没有得到证实，故本模块
    **不依赖**该行为，改成在 asyncio 层自己兜底（:func:`asyncio.wait_for`），
    这条路径可单测、可证伪。

------------------------------------------------------------------------------
三条设计约束
------------------------------------------------------------------------------
1. **箍住「会在检索路径上被调用」的操作，不管它名字像读还是像写。**
   被箍的是 ``search`` / ``list_documents`` / ``list_chunks`` /
   ``has_collection`` / ``create_collection`` 五个；只有 ``insert`` /
   ``delete`` 这类**纯离线**的写操作原样透传。理由是**失败要响**：
   灌数据脚本离线运行、有人盯着，那里卡住应该暴露出来让人处理；
   把它悄悄超时掉，会让「灌了一半的数据」变成一个静默状态。

   ⚠️ 但「读 / 写」这个划分**不足以判断该不该箍** —— 这是本模块
   2026-10-03 自查时发现的一处真实缺口。``has_collection`` 名字上像
   「探询」，``create_collection`` 名字上像「写」，可它们俩**都在检索
   路径上**：框架的 ``KnowledgeBase.search()`` 每次都先走
   ``ensure_collection()``（``rag/_knowledge.py:237`` → ``:167``），
   而它是 ``has_collection`` 问一句、**没有就 ``create_collection``**
   （``rag/_knowledge.py:179``），成功之后才把 ``_collection_ready``
   置真（``:128``）。两个都是 Milvus RPC、两个都没有截止时间，因此
   **卡住其中任何一个，与卡住 ``search`` 对使用者完全等价** ——
   而这是我们 2026-10-03 实测到的同一形态。

   只箍前四个的版本还多一层危害：集合**不存在**时（例如有人删了集合、
   或换了个新库地址），第一次检索会走进 ``create_collection`` ——
   恰恰是唯一没有护栏的那一次，于是「集合不存在」这个可预期的状态
   变成了整个服务假死。本项目自己的 ``store.ensure_collection``
   （``src/knowledge/store.py:195``）也走这两个方法。
   （对离线脚本而言，被箍住只意味着失败方式从「挂住」变成
   ``VectorSearchTimeout``，或者在连续失败后被熔断器以
   ``CircuitBreakerOpen`` 拦下 —— 仍然是响亮的失败。）
   ⚠️ ``create_collection`` 被箍住之后，``ensure_collection`` 里的
   ``_collection_ready`` 会停在 ``False``，于是后续每次检索都重试一次
   —— 这正是我们要的：让熔断器数得到账，并在连续失败后停止再发。

2. **只把「下游不可用」计入熔断。** 超时与 pymilvus 的连接类错误算失败；
   调用方自己的错误（集合不存在、过滤条件非法）**不算** —— 后者是我们的
   bug，把它算成「Milvus 挂了」会让熔断器指错方向，还会掩盖真正的故障。
   这也正是本模块不复用 ``CircuitBreaker.guard()`` 的原因：那个上下文
   管理器把所有异常一视同仁地记成失败（对本模块的需求来说太粗）。

3. **熔断器是全进程共享的一个实例。** 不是为了省内存，是正确性：
   知识库与长期记忆**各构造了一个 store**（``src/knowledge/__init__.py``
   与 ``src/memory/__init__.py``），若各自持一个熔断器，同一场 Milvus
   故障要被分别数够三次才熔断；共享一个则全体调用者共同贡献失败计数。

------------------------------------------------------------------------------
一件不做就会前功尽弃的事：客户端预热必须离开事件循环
------------------------------------------------------------------------------
    框架的读方法都是 ``await asyncio.to_thread(self.get_client().search,
    ...)`` 的形状（``_vdb/_milvus_lite.py:304``、``:187``）—— 注意
    ``self.get_client()`` 是 ``to_thread`` 的**实参**，在**事件循环线程**
    上求值；而它第一次调用会真的建连（``_vdb/_milvus_lite.py:78-84``）。
    那是一次**同步**阻塞：Milvus 不可达时它会把整个事件循环按住
    （pymilvus 在**建连**路径上自己把 ``timeout=None`` 归一到 10s，
    ``pymilvus/client/grpc_handler.py:248``），于是 ``asyncio.wait_for``
    的截止时间**根本来不及触发** —— 循环被占着，连定时器回调都没机会跑，
    「有超时」只写在纸上。所以护栏在调用目标方法**之前**先把
    ``get_client()`` 本身丢进线程预热（也在同一个超时预算内），
    并保证同一时刻只有一个协程在建连（否则会建出两个客户端，泄漏一个）。

------------------------------------------------------------------------------
遗留线程：超时**不会**取消线程，所以这里只能压速率 + 限并发
------------------------------------------------------------------------------
    ``asyncio.wait_for`` 取消的是 ``asyncio.to_thread`` 的 await，而线程里
    正在跑的 pymilvus 调用不受影响（Python 无法中断线程）。所以「遗留线程」
    是这种护栏**固有**的代价，只能控制它涨得多快、同时留几个，压不到零。
    三层机制各管一段：

      · **闸门**（:func:`_gate`，同时在飞 ≤ 4）—— 管**瞬时并发**。
        ⚠️ 它限的是**同时在飞的协程数**，**不是**线程数 —— 别把这两件事
        混起来（这里曾经写错过：原文声称「一次突发最多遗留 4 个卡死线程」，
        不成立）。原因就是上面那句：超时会**归还**名额（名额代表「同时在飞
        的 await」，不代表「线程结束了」），所以 t=5s 那一批被放弃后，
        排在闸门后面、自己 deadline 还没到的请求立刻又占满 4 个名额、
        再起 4 个卡死线程。持续故障下线程就是这样一批批涨上去的，
        闸门压缩的是**涨的速率**，不是总量。
        那为什么还要这道闸门：它把「一瞬间并发涌入 N 个请求」削成
        「同时最多 4 个在下游」，于是**遗留速率**从「随请求量线性」变成
        「随超时窗口线性」。为什么不能在超时时扣住名额直到线程结束：
        那样 Milvus 长挂时名额永不归还，闸门会变成一道**永久关闭**的门，
        检索再也不会自动恢复 —— 比多遗留几个线程更坏。
        这事很要紧是因为：``asyncio`` 默认执行器只有 ``min(32, CPU+4)``
        个工位（本机 4 核 ⇒ **8 个**），而它是**全进程共享**的 ——
        本地 ONNX 向量化（``src/web_embedding/local.py``）也走它。
        不过**每个事件循环各有一套默认执行器**，所以闸门也是按 loop 各一份
        （见 :data:`_GATES`）。
      · **熔断器**（连续 3 次失败 → 开路 30s，之后单探针）—— 管**持续**。
        稳态下遗留速度被压到约「每 30 秒一个线程」。
      · **超时预算**（``ALIGO__MILVUS__SEARCH_TIMEOUT_SECONDS``）——
        管**单次**。只要 Milvus 正常，工位在毫秒级就还回来。

    ⚠️ 三段加起来仍然**不是**「线程数有上界」（理由同上：闸门管的是协程）：
    Milvus 持续挂几小时的话，
    遗留线程会以约 120 个/小时的速度在默认执行器里堆积，最终吃光那 8 个工位，
    连本地向量化一起饿死（此时向量化的
    ``src/web_embedding/bounded.py`` 会让它**按时失败**而不是挂住，
    所以表现是「检索整体降级」而不是「服务假死」）。真正把线程收回来
    需要能中断线程，Python 给不了。**这一条是已知的、留在明面上的取舍**，
    不是被忘掉的缺口 —— 写在这里是因为「护栏 = 线程数也受控」是个很容易
    被默认成立、实际上不成立的假设。
    真正的解药仍然是让 Milvus 活着，以及别让它挂上几小时
    （``docs/06-部署与运维.md`` 第十节）。
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from typing import Any, Callable

from src.llm.breaker import CircuitBreaker, CircuitBreakerOpen

logger = logging.getLogger(__name__)

#: 连续失败多少次后熔断。取 3 而不是 5（模型调用用的是 5）：检索是
#: **可选**能力 —— 熔断错了（其实只是抖动）代价是多等一个冷却期再自动恢复，
#: 而熔断晚了代价是每个请求都白等一整个超时。两者不对称，故取小。
_FAILURE_THRESHOLD = 3

#: 熔断后冷却多久。30s 与模型侧一致：够 Milvus 重启或从负载尖峰里缓过来，
#: 又不至于让一次短暂抖动把检索停掉太久。
_RECOVERY_SECONDS = 30.0

#: 被护栏箍住的**会在检索路径上被调用**的操作名（判据不是读/写，见模块文档
#: 约束 1）。``insert`` / ``delete`` 这类纯离线写操作刻意不在其中。
#: ⚠️ 这份元组是给**测试**用的单一事实来源：``tests/test_knowledge_guard.py``
#: 会断言其中每个名字都真的在 :class:`GuardedVectorStore` 上显式实现了
#: （而不是靠 ``__getattr__`` 透传出去），并对每个名字各跑一遍「卡住必须
#: 有界失败」。往这里加名字而不实现方法，测试立刻红。
_GUARDED_OPERATIONS = (
    "search",
    "list_documents",
    "list_chunks",
    "has_collection",
    "create_collection",
)

#: 同时**在飞**的受护栏调用上限。见模块文档「遗留线程」一节：
#: 超时放弃的是 await，线程还在跑，而线程池的工位数是有限的
#: （``asyncio`` 默认执行器是 ``min(32, CPU+4)`` —— 本机 4 核 ⇒ **只有 8 个**）。
#: 没有这道闸门，一次「Milvus 刚挂 + 若干用户同时提问」就能在**一瞬间**
#: 占满全部工位，而工位是**全进程共享**的：本地 ONNX 向量化
#: （``src/web_embedding/local.py``）也走它，于是向量化跟着一起饿死。
#:
#: 取 4 而不是更大：正常时一次检索是毫秒级，4 个并发工位远超需要；
#: 异常时它是「一瞬最多遗留几个卡死线程」的上界，越小越好。
_MAX_IN_FLIGHT = 4

#: 在飞闸门的**按事件循环**注册表。⚠️ 与 :data:`_BREAKER` 一样，闸门必须是
#: **共享**的：知识库与长期记忆各建了一个 store（``src/knowledge/__init__.py`` /
#: ``src/memory/__init__.py``），按实例各发一份就等于把上限乘了实例数 ——
#: 那正是「共享一个熔断器」要避免的同一类错误。
#:
#: ⚠️ 但「共享」的粒度只能是**每个事件循环一份**，不能是一个模块级
#: ``asyncio.Semaphore``：``asyncio`` 原语在第一次发生竞争（需要挂 waiter）时
#: 会**惰性绑定**当时那个 loop，之后换一个 loop 再竞争直接抛
#: ``RuntimeError: ... is bound to a different event loop``。
#: 实测（同一进程内 ``asyncio.run`` 跑两遍饱和调用）：第一遍正常，第二遍
#: 必抛。而一个进程里出现第二个 loop 是常态 —— 每个 asyncio 用例一个 loop，
#: 脚本里每次 ``asyncio.run`` 一个 loop。那个 ``RuntimeError`` 会被
#: :meth:`GuardedVectorStore._bounded` 的 ``except BaseException`` 原样上抛，
#: 再被调用方宽口径的 ``except Exception`` 接住 ⇒ 表现为**检索能力静默消失**，
#: 而不是一个能看出原因的报错。
#:
#: 按 loop 分开在语义上也更准：``asyncio.to_thread`` 用的是**该 loop 自己的**
#: 默认执行器（``loop._default_executor``），工位本来就是每 loop 一份的。
#: 用 ``WeakKeyDictionary`` 而不是 ``id(loop)`` 作键，是为了让结束的 loop
#: 能被回收、注册表不会随进程寿命无限增长。
_GATES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def _consume_task_exception(task: "asyncio.Future[Any]") -> None:
    """把一个 future 的异常取走，避免 GC 时打 "never retrieved" 噪声。

    只用于 :meth:`GuardedVectorStore._warm_client` 记下的那个建连 task：
    它可能**没有任何人** await 过（调用方自己先超时走了），
    而 asyncio 对「有异常但没人取」的 future 会在析构时打一条 ERROR。
    那条日志会把真正的错误信息淹没，且看起来像是泄漏。

    Args:
        task (`asyncio.Future`): 已完成的建连 future。
    """
    if not task.cancelled():
        task.exception()  # 取走即可；真正的抛出在 _warm_client 里做


def _gate() -> asyncio.Semaphore:
    """取**当前事件循环专属**的在飞闸门（没有就建一个）。

    Returns:
        `asyncio.Semaphore`: 当前 loop 的闸门，名额为 :data:`_MAX_IN_FLIGHT`。
    """
    loop = asyncio.get_running_loop()
    gate = _GATES.get(loop)
    if gate is None:
        gate = asyncio.Semaphore(_MAX_IN_FLIGHT)
        _GATES[loop] = gate
    return gate

#: 全进程共享的熔断器。⚠️ 见模块文档约束 3：**不要**改成按实例创建。
_BREAKER = CircuitBreaker(
    failure_threshold=_FAILURE_THRESHOLD,
    recovery_seconds=_RECOVERY_SECONDS,
    name="vector_store",
)


class VectorSearchTimeout(TimeoutError):
    """向量库读操作超时。

    ⚠️ 继承内置的 :class:`TimeoutError`（而不是自定义一个 ``Exception``）
    是刻意的：调用方写 ``except TimeoutError`` 也能接住它。
    ``src/memory/semantic.py`` 与框架的 RAG 中间件都是宽口径 ``except
    Exception``，两条路都能降级；继承 ``TimeoutError`` 让**新**的调用点
    不必知道本模块的存在也能正确处理超时。
    """

    def __init__(self, operation: str, timeout: float) -> None:
        self.operation = operation
        self.timeout = timeout
        super().__init__(
            f"向量库 {operation} 超过 {timeout:g}s 未返回，已放弃本次检索"
            f"（底层调用可能仍在进行，见 src/knowledge/guard.py 的模块文档）。",
        )


def get_vector_store_breaker() -> CircuitBreaker:
    """取全进程共享的向量库熔断器（供 ``/readyz`` 与 ``/metrics`` 读取状态）。"""
    return _BREAKER


def _is_downstream_failure(exc: BaseException) -> bool:
    """判断一个异常是否说明**下游（Milvus）不可用**。

    ⚠️ 判据刻意收窄（模块文档约束 2）：只有超时，以及**不是调用方错误**的
    pymilvus 异常算数。返回值是 ``False`` 时，熔断器**不记账** ——
    那个异常会照常抛给调用方，只是不会影响熔断状态。

    Args:
        exc (`BaseException`): 被捕获的异常。

    Returns:
        `bool`: 是否应计入「下游失败」。
    """
    if isinstance(exc, TimeoutError):
        # 含本模块的 VectorSearchTimeout 与 asyncio 自己的超时。
        return True

    # ⚠️ 延迟 import：``guard`` 被 ``store`` 导入，而 ``store`` 在
    # ``/readyz`` 与启动路径上都会被构造 —— 那两个地方不该强依赖
    # pymilvus 能 import 成功（框架自己也是这么处理的，见
    # ``_vdb/_milvus_lite.py`` 里对 pymilvus 的惰性导入）。
    try:
        from pymilvus.exceptions import MilvusException
    except Exception:  # noqa: BLE001 —— 取不到就只认超时
        return False
    if not isinstance(exc, MilvusException):
        return False

    # ⚠️ 不是所有 ``MilvusException`` 都是「Milvus 挂了」：pymilvus 会把
    # **服务端判定为调用方输入错误**的那些标出来（``is_input_error``，
    # 由 ``MilvusException.from_status`` 从 Status 的
    # ``extra_info["is_input_error"]`` 解出）—— 集合不存在、维度不匹配、
    # filter 非法都属于这一类。它们说明对端**活着并且回了话**，
    # 把它记成下游失败会让熔断器指错方向（正是约束 2 要防的事）。
    # 取不到该属性时按「不是调用方错误」处理 —— 宁可多熔断一次
    # （代价是 30s 后自动恢复），也不要漏掉一次真实的下游故障。
    return not bool(getattr(exc, "is_input_error", False))


class GuardedVectorStore:
    """给向量库的**读**操作加超时与熔断的透明代理。

    实现方式是**代理**而不是继承 ``MilvusLiteStore``：

      · 继承会把几十个方法一起继承过来，其中绝大多数**没有**护栏，
        看代码的人无法一眼看出「到底哪些操作被箍住了」；
      · 代理把被箍住的五个方法（``_GUARDED_OPERATIONS``）显式写在类里，
        其余一律 :meth:`__getattr__` 透传 —— 「哪些有护栏」是读出来的，
        不是猜出来的。

    ⚠️ 框架侧对 ``vector_store`` 只做鸭子类型使用（``KnowledgeBase`` 是
    普通类，构造时不校验类型，``rag/_knowledge.py:44``），所以代理不会
    被 ``isinstance`` 拦住。本项目的 ``store.get_client()`` /
    ``store._client`` 这类访问也照常透传。

    ⚠️ ``__aenter__`` / ``__aexit__`` 必须**显式**实现（见类尾）：Python 的
    特殊方法查找走类型而不是实例字典，**不经过** :meth:`__getattr__`，
    所以 ``async with store:`` 在纯 ``__getattr__`` 代理上会直接 TypeError，
    尽管 ``store.__aexit__(...)`` 这种显式写法看起来能用。
    """

    def __init__(
        self,
        inner: Any,
        *,
        timeout: float,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        """装配护栏。

        Args:
            inner (`Any`): 被包裹的向量库（通常是框架的 ``MilvusLiteStore``）。
            timeout (`float`): 单次读操作的超时（秒），必须 > 0。
            breaker (`CircuitBreaker | None`, optional): 熔断器；``None`` 时
                用全进程共享的那个（正常路径都走 ``None``，测试才显式传）。

        Raises:
            ValueError: ``timeout`` 非正时。取 0 会让每次检索**必然**超时，
                那等于把检索能力静默关掉 —— 这种配置必须当场炸，不能等到
                线上表现为「知识库好像没数据」。
        """
        if timeout <= 0:
            raise ValueError(
                f"检索超时必须 > 0，实际为 {timeout!r}；"
                f"取 0 会让每次检索都超时（等价于静默关闭检索能力）。",
            )
        self._inner = inner
        self._timeout = float(timeout)
        self._breaker = breaker if breaker is not None else _BREAKER
        #: **正在跑的那次建连**（单飞用，见 :meth:`_warm_client`）。
        #: 记下它是为了「后来者等同一个建连」，而不是各自再起一个 ——
        #: 光靠锁挡不住重复建连（超时会把锁放掉、线程还在跑），
        #: 于是会有多个 ``MilvusClient`` 被建出来，只有最后一个留下，
        #: 其余连同它们的后台线程一起泄漏。
        #: ⚠️ 它属于**创建它的那个事件循环**（asyncio 的 Future 不能跨 loop），
        #: 且由本对象独占 —— 这与 :data:`_GATES` 那种「按 loop 共享」的需求不同。
        self._warm_task: asyncio.Future[Any] | None = None

    # ------------------------------------------------------------------
    # 被护栏箍住的读操作
    # ------------------------------------------------------------------
    async def search(self, *args: Any, **kwargs: Any) -> Any:
        """带超时的 ``search``（语义与参数完全透传给底层 store）。"""
        return await self._bounded("search", self._inner.search, *args, **kwargs)

    async def list_documents(self, *args: Any, **kwargs: Any) -> Any:
        """带超时的 ``list_documents``。"""
        return await self._bounded(
            "list_documents",
            self._inner.list_documents,
            *args,
            **kwargs,
        )

    async def list_chunks(self, *args: Any, **kwargs: Any) -> Any:
        """带超时的 ``list_chunks``。"""
        return await self._bounded(
            "list_chunks",
            self._inner.list_chunks,
            *args,
            **kwargs,
        )

    async def has_collection(self, *args: Any, **kwargs: Any) -> Any:
        """带超时的 ``has_collection``。

        ⚠️ 它看着不像读操作，但**在检索路径上**：框架的
        ``KnowledgeBase.search()`` 每次都先经 ``ensure_collection()``
        调它（``rag/_knowledge.py:179``），卡住它等于卡住检索。
        理由详见模块文档约束 1。
        """
        return await self._bounded(
            "has_collection",
            self._inner.has_collection,
            *args,
            **kwargs,
        )

    async def create_collection(self, *args: Any, **kwargs: Any) -> Any:
        """带超时的 ``create_collection``。

        ⚠️ 它名字上像「写」，但它**在检索路径上**：框架的
        ``KnowledgeBase.ensure_collection()`` 在 ``has_collection`` 回答
        「没有」时会调它（``rag/_knowledge.py:179``），而那一步由
        ``search()`` 触发（``:237``）。集合缺失时，它就是唯一没有护栏的
        那一次调用 —— 理由详见模块文档约束 1。
        """
        return await self._bounded(
            "create_collection",
            self._inner.create_collection,
            *args,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # 内部：一次带截止时间的调用
    # ------------------------------------------------------------------
    async def _bounded(
        self,
        operation: str,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """执行一次被箍住的调用。

        Args:
            operation (`str`): 操作名（只用于日志与异常信息）。
            func (`Callable`): 真正要调用的底层方法。
            *args: 透传。
            **kwargs: 透传。

        Returns:
            `Any`: 底层返回值。

        Raises:
            CircuitBreakerOpen: 熔断器处于开路/试探且未放行时。
            VectorSearchTimeout: 超过 ``timeout`` 仍未返回时。
            Exception: 底层抛出的其它异常，原样上抛。
        """
        # ⚠️ 手写 allow_request / record_* 而**不用** ``breaker.guard()``：
        # guard() 把任何异常都记成失败，而这里只有「下游不可用」才该记账
        # （模块文档约束 2）。allow_request 在开路时抛 CircuitBreakerOpen，
        # 此时**不打网络**，这正是熔断的意义。
        #
        # ⚠️ 它在 try **外面**是刻意的：没被放行的请求不该走下面任何一个
        # 「归还试探名额」的分支 —— 名额此刻握在别人手里，替别人归还
        # 会让 HALF_OPEN 一下放出多个试探请求。
        await self._breaker.allow_request()

        started = time.monotonic()
        try:
            # ⚠️ 预热与真正的调用**共用一个**超时预算：建连本身也可能
            # 卡住（见模块文档「客户端预热」一节），它必须和检索一样有界。
            result = await asyncio.wait_for(self._invoke(func, args, kwargs), self._timeout)
        except CircuitBreakerOpen:
            # 理论上到不了这里（allow_request 已经拦过），但保留它是为了
            # 让「熔断异常不计失败」这条规则在**任何**控制流下都成立。
            await self._breaker.release_probe()
            raise
        except TimeoutError as exc:
            # ⚠️ 这里同时兜住两种超时：asyncio.wait_for 自己抛的，
            # 以及底层自己抛的（比如某天 pymilvus 开始支持 timeout 了）。
            await self._breaker.record_failure()
            elapsed = time.monotonic() - started
            logger.warning(
                "向量库 %s 超时（%.2fs ≥ 上限 %.2fs），本次检索降级为「无结果」；"
                "连续失败 %d/%d 次后将熔断 %.0fs。",
                operation,
                elapsed,
                self._timeout,
                self._breaker.snapshot().get("consecutive_failures", 0),
                _FAILURE_THRESHOLD,
                _RECOVERY_SECONDS,
            )
            raise VectorSearchTimeout(operation, self._timeout) from exc
        except BaseException as exc:
            # BaseException 而不是 Exception：``asyncio.CancelledError``
            # 继承自 BaseException，而它**不该**被记成下游失败（那是调用方
            # 主动取消，不是 Milvus 的错）—— 下面的判据会把它排除掉。
            if _is_downstream_failure(exc):
                await self._breaker.record_failure()
            else:
                # ⚠️ 不记账 ≠ 可以就这么走人：若本次调用握着一个 HALF_OPEN
                # 试探名额（被取消、或调用方错误），不归还就会把熔断器
                # **永久**卡在 HALF_OPEN（后续每个请求都被
                # ``CircuitBreakerOpen(name, 0.0)`` 拒绝，直到进程重启）。
                # 详见 ``src/llm/breaker.py::release_probe``。
                await self._breaker.release_probe()
            raise
        else:
            await self._breaker.record_success()
            return result

    async def _invoke(
        self,
        func: Callable[..., Any],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """抢工位 → 预热客户端 → 执行一次底层调用。

        ⚠️ 三步都在**同一个**超时预算内（本方法整个被 ``wait_for`` 包着）：
        抢工位也要有界，否则「向量库挂了」会退化成「所有请求在闸门前排队
        排到天荒地老」—— 那是把一种假死换成了另一种。

        ⚠️ 闸门用 ``async with``，因此**超时/取消时名额会被归还**。
        ⚠️⚠️ 归还的后果必须说清楚：**它限制的是同时在飞的协程数，不是线程数。**
        超时归还名额时，那个线程还在 Milvus 里卡着 —— 下一个调用者立刻
        可以占走名额再起一个线程。所以「一瞬最多遗留 4 个卡死线程」这句
        话**是错的**（曾在模块文档与注释里出现过，已更正）：
        持续故障下线程会一批批增加，闸门压住的只是**涨的速度**。
        真正把线程数压到个位数的是熔断器（连续 3 次失败后每 30s 才放一个
        试探）。为什么不能扣住名额直到线程结束：那样 Milvus 长挂时名额
        永不归还，闸门变成一道**永久关闭**的门，检索再也不会恢复 ——
        比「多遗留几个线程」更坏。取舍见模块文档「遗留线程」一节。
        """
        async with _gate():
            await self._warm_client()
            return await func(*args, **kwargs)

    async def _warm_client(self) -> None:
        """在**线程**里触发底层建连，避免 ``get_client()`` 阻塞事件循环。

        ⚠️ 这不是优化，是「超时能不能生效」的前提：框架的读方法写成
        ``asyncio.to_thread(self.get_client().search, ...)``，``get_client()``
        是在**事件循环线程**上求值的同步调用。Milvus 不可达时它阻塞的
        那几秒里，事件循环连 ``wait_for`` 的超时回调都排不上 ——
        护栏看上去在，实际上整台服务已经假死（这正是要防的症状）。
        在协程里把它换成 ``to_thread`` 之后，事件循环始终是空的，
        超时才能真正触发。

        ⚠️ **单飞（single-flight）**：``get_client()`` 自己不是并发安全的
        （``if self._client is None: ... = MilvusClient(...)``），
        两个协程同时预热会建出两个客户端、泄漏一个。

        而**光加锁是不够的** —— 这是本方法最容易写错的地方（曾经就写错过）：
        超时取消的是 ``await``，``async with`` 随之退栈把锁**还回去**，
        可线程里的建连还在跑。下一个调用者拿到锁、发现 ``_client`` 仍是
        ``None``，于是**再建一个**。三个连续超时的检索就能建出三个客户端，
        只有最后一个被留下，前两个连 ``MilvusClient`` 的后台线程一起泄漏
        （框架的 ``__aexit__`` 只关当前那个 ``_client``）。

        正确做法是记下**正在跑的那个建连 future**，后来者等它、而不是另起一个：
        任何时刻每个 store 最多只有一次建连在飞。

        ⚠️ 这里**刻意不用锁**：asyncio 是单线程协作式调度，「读
        :attr:`_warm_task` → 判断 → 建 task → 写回」中间**没有任何 await**，
        因此这段天生是原子的，别的协程插不进来。反倒是 ``asyncio.Lock``
        会带来两个额外麻烦：它自己也有跨事件循环绑定的问题（同
        :data:`_GATES` 那条注释），而且如上一段所说，它并不能真正
        挡住重复建连。

        ⚠️ 等它用的是 ``asyncio.wait``，**不是** ``await task``：
        ``await`` 在被取消时会**连带取消那个 task**（而 task 取消并不能
        停下线程，只会让它变成「跑完了也没人要」）。``asyncio.wait``
        只等结果，调用方被取消时不动它。

        ⚠️ 对没有 ``get_client`` 的对象（测试替身、别的 store 实现）
        整个方法是个 no-op —— 代理是鸭子类型的，不假设底层一定长这样。
        """
        get_client = getattr(self._inner, "get_client", None)
        if get_client is None:
            return

        # 常见的快路径：客户端早就建好了，连判断都不必做。
        # ⚠️ 这里读 ``_client`` 是探头，不是依赖：它是框架自己的缓存字段
        # （``_vdb/_milvus_lite.py:75``，随包 vendored、版本固定）。
        # 判断错了也不影响正确性 —— 大不了多绕一次线程预热。
        if getattr(self._inner, "_client", None) is not None:
            return

        # ⚠️ 以下到赋值结束**不得插入任何 await**（见 docstring 的原子性说明）。
        task = self._warm_task
        if task is None or task.done():
            task = asyncio.ensure_future(asyncio.to_thread(get_client))
            # 挂一个「取走异常」的回调：调用方可能自己先超时走了，
            # 于是这个 future 永远没人 await，而没被取走的异常会在
            # 它被 GC 时打成 "Task exception was never retrieved" 噪声。
            task.add_done_callback(_consume_task_exception)
            self._warm_task = task

        await asyncio.wait({task})
        if task.done() and not task.cancelled():
            # 把内层真实错误（认证失败 / 连不上 / 库不存在）**原样**抛出去：
            # 上层要靠它记熔断，而且它的类型比「超时」有信息量得多。
            exception = task.exception()
            if exception is not None:
                raise exception

    # ------------------------------------------------------------------
    # 生命周期（必须显式实现，见类文档的说明）
    # ------------------------------------------------------------------
    async def __aenter__(self) -> "GuardedVectorStore":
        """进入异步上下文：委托给底层，但**返回代理自己**。

        ⚠️ 返回 ``self`` 而不是底层的返回值：否则 ``async with store as s``
        拿到的会是**没有护栏**的裸 store，一次手滑就把护栏整个绕过去了。

        ⚠️ 必须显式定义：``async with`` 走的是**类型**上的特殊方法查找，
        不经过 :meth:`__getattr__` —— 只靠透传的话这里是 TypeError。
        """
        enter = getattr(self._inner, "__aenter__", None)
        if enter is not None:
            await enter()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> Any:
        """退出异步上下文：原样委托给底层（关连接、释放 Lite server 都在那里）。

        ⚠️ 返回值（是否吞掉异常）也照原样返回，护栏不对异常做任何裁决。
        """
        exit_ = getattr(self._inner, "__aexit__", None)
        if exit_ is None:
            return None
        return await exit_(exc_type, exc_val, exc_tb)

    # ------------------------------------------------------------------
    # 透传
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """把未被箍住的一切属性透传给底层 store。

        ⚠️ 只在正常属性查找失败时被调用，所以 ``_inner`` / ``_timeout`` /
        ``_breaker`` 不会走这里，不存在递归风险。

        Args:
            name (`str`): 属性名。

        Returns:
            `Any`: 底层 store 上的同名属性。
        """
        return getattr(self._inner, name)


def guard_vector_store(inner: Any, *, timeout: float) -> GuardedVectorStore:
    """把向量库包上护栏。

    Args:
        inner (`Any`): 被包裹的向量库。
        timeout (`float`): 单次读操作超时（秒）。

    Returns:
        `GuardedVectorStore`: 带超时与熔断的代理；写入与连接方法照常透传。
    """
    return GuardedVectorStore(inner, timeout=timeout)


__all__ = [
    "GuardedVectorStore",
    "VectorSearchTimeout",
    "get_vector_store_breaker",
    "guard_vector_store",
]
