# -*- coding: utf-8 -*-
"""给**重排**那一次模型调用加截止时间 —— 检索路径上最后一处无界等待。

═══ ⚠️ 为什么需要它 ═══

``src/knowledge/guard.py`` 箍住了向量库，``src/web_embedding/bounded.py``
箍住了向量模型，但开了重排（``ALIGO__RERANK__ENABLED=true``）之后，一次检索
还有**第三段**调用：

    框架 ``_rerank_results``（``middleware/_rag.py:454``）::

        response = await rerank_model.generate_structured_output(   # ← 本模块覆盖这里
            messages=[...],
            structured_model=_RerankOutput,
        )

它的调用**在返回给用户的那条链路上**（``RAGMiddleware.on_reasoning``，
``_rag.py:319``）。没有任何一层给它设截止时间：

  · ``ModelTimeoutMiddleware`` 只管 ``on_model_call`` —— 那是**agent 自己**的
    模型调用，重排是中间件在 ``on_reasoning`` 里另起的一次调用，不经过那个钩子；
  · 框架给重排的 ``except Exception``（``_rag.py:433-445``）只在**抛异常**时
    才回退到向量序，而「卡住不返回」永远不抛异常；
  · SDK 层的 ``timeout``（``src/llm/factory.py`` 的 ``client_kwargs``）箍的是
    **单次 HTTP 请求**，而 ``generate_structured_output`` 内部是一条
    **策略阶梯 + 重试**（``model/_base.py:457-490``：forced → auto →
    no_think → none 四种策略，每种各带 ``max_retries`` 次重试）。
    最坏情况是 4 种策略 × 3 次尝试 × SDK 超时 —— 配置写着 60s，
    实际能挂十几分钟。对要回应用户的服务来说，这就是「无界」。

═══ 与 ``ModelTimeoutMiddleware`` 同一个坑：必须按钟判断 ═══

**不能用 ``asyncio.wait_for``**。对话模型基类会**吞掉** ``CancelledError``
（``model/_base.py:224-230`` 非流式转成空响应、``:255-270`` 流式转成末尾分片），
于是 ``wait_for`` 看到的永远是「正常返回」，超时判定一次都不成立 ——
护栏在纸面上存在、实际是死的。``src/llm/middleware.py``
的 ``ModelTimeoutMiddleware._await_within_budget`` 已经踩过这个坑，
本模块用同一套写法（``asyncio.wait`` 计时 + 超时后自己抛 + ``cancel()`` 内层
尽快断开上游连接）。⚠️ 两处**刻意重复**而不是抽公共函数：那边的调用点是
中间件钩子（要区分流式/非流式两种措辞与两条不同的中断路径），这边是一次
普通协程调用；把它们硬塞进一个抽象里，会让「哪个护栏在报错」变难判断。

═══ 为什么是**鸭子类型代理**而不是继承 ``ChatModelBase`` ═══

``src/web_embedding/bounded.py`` 选择继承，是因为框架会读向量模型的
``dimensions`` / ``supports_multimodal`` **类型属性**，纯代理会丢。重排这条路
已经核实**只用两个东西**（全项目内 grep 过 ``middleware/_rag.py``）：

    ``_rag.py:484``   ``rerank_model.model``                       —— 打日志
    ``_rag.py:518``   ``await rerank_model.generate_structured_output(...)``

没有 ``isinstance(..., ChatModelBase)``、没有读 ``credential`` / ``stream`` /
``parameters``。所以这里做纯代理（与 ``src/knowledge/guard.py`` 同一种权衡）：
继承 ``ChatModelBase`` 反而要跟着基类的构造签名与抽象方法走，
而本类**根本不使用**基类的 ``__call__``（那条路会吞取消，正是我们要绕开的）。
⚠️ 若将来有代码把本对象传给一个会 ``isinstance`` 检查的地方，
它会不通过 —— 那种地方必须显式 :func:`unwrap_chat_model`。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agentscope.model import ChatModelBase

__all__ = [
    "BoundedChatModel",
    "ChatCallTimeout",
    "bound_chat_model",
    "unwrap_chat_model",
]


logger = logging.getLogger(__name__)


class ChatCallTimeout(TimeoutError):
    """一次对话模型调用超过预算仍未返回。

    ⚠️ 与 :class:`~src.llm.middleware.ModelCallTimeout`、
    :class:`~src.knowledge.guard.VectorSearchTimeout`、
    :class:`~src.web_embedding.bounded.EmbeddingCallTimeout` 一样继承内置
    ``TimeoutError``：调用方的宽口径 ``except Exception``（框架的重排兜底
    ``_rag.py:433``、我们自己的降级路径）不必先认识本模块，
    就能把「超时」正确地当成一次可降级的失败。

    Attributes:
        timeout (`float`): 触发的超时预算（秒）。
        model (`str`): 被箍住的模型名。⚠️ 只放模型名，不放 prompt ——
            重排的 prompt 里**装着检索到的知识库正文**，把它带进异常、
            再被日志或探针打出来，等于把租户数据泄露到运维面。
    """

    def __init__(self, timeout: float, *, model: str) -> None:
        """初始化。

        Args:
            timeout (`float`): 超时预算（秒）。
            model (`str`): 模型名。
        """
        self.timeout = timeout
        self.model = model
        super().__init__(
            f"模型 {model!r} 超过 {timeout:g}s 未返回，已放弃本次调用"
            f"（详见 src/llm/bounded.py 的模块文档）。",
        )


class BoundedChatModel:
    """给任意对话模型加「整次调用截止时间」的透明代理。

    ⚠️ 截止时间包住**整次** ``generate_structured_output``，而不是它内部的
    每一次 HTTP 尝试：内部那条策略阶梯（4 种策略）与重试对调用方是**不可见**的
    —— 调用方真正关心的语义是「我这次等多久能拿到结果」，把超时按「次」算
    会让最坏情况变成一个乘出来的数（策略数 × 重试数 × 预算），
    那正是要消灭的东西。

    ⚠️ 超时**不会**让底层那次 HTTP 立刻停下：我们取消的是**等待**。
    与 ``guard.py`` / ``web_embedding/bounded.py`` 的说明同源，这里不重复，
    但结论一致 —— 调用方立刻拿到失败，比陪着一起等更重要。
    """

    def __init__(self, inner: ChatModelBase, *, timeout: float) -> None:
        """装配护栏。

        Args:
            inner (`ChatModelBase`): 被包裹的对话模型。
            timeout (`float`): 单次 ``generate_structured_output`` 的
                截止时间（秒），必须 > 0。

        Raises:
            ValueError: ``timeout`` 非正时。取 0 会让每次调用**必然**超时，
                等价于静默关掉重排能力 —— 这种配置必须当场炸
                （与其余三个护栏对「非正超时」的处理方式一致）。
        """
        if timeout <= 0:
            raise ValueError(
                f"重排调用超时必须 > 0，实际为 {timeout!r}；"
                f"取 0 会让每次重排都超时（等价于静默关闭重排阶段）。",
            )
        # ⚠️ 先登记内层引用，**再**做别的：``__getattr__`` 在属性查找失败时
        # 会读 ``self._inner``，构造途中若有任何未赋值属性被访问，
        # 就会掉进「找 _inner → 又走 __getattr__」的无限递归。
        self._inner = inner
        self._timeout = float(timeout)

    # ------------------------------------------------------------------
    # 身份
    # ------------------------------------------------------------------
    @property
    def inner(self) -> ChatModelBase:
        """被包裹的对话模型（供 :func:`unwrap_chat_model` 与测试使用）。"""
        return self._inner

    @property
    def timeout(self) -> float:
        """单次调用的截止时间（秒）。"""
        return self._timeout

    @property
    def model(self) -> str:
        """模型名。

        ⚠️ **显式**提供而不是靠 ``__getattr__`` 兜：框架的重排日志
        （``_rag.py:484``）会读它，而一个会在运维面上出现的字段
        值得在类上看得见。
        """
        return str(getattr(self._inner, "model", "<未知>"))

    # ------------------------------------------------------------------
    # 被箍住的调用
    # ------------------------------------------------------------------
    async def generate_structured_output(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """在线内调用内层模型的结构化输出，超时即放弃。

        ⚠️ 签名用 ``*args, **kwargs`` 全透传，而不是照抄框架的
        ``(messages, structured_model, **kwargs)``：框架改一次参数名，
        照抄的写法就会在**运行期**以 ``TypeError`` 的形式炸出来，
        而那种错误看起来像「调用方传错了」，指不到本层。
        照抄成位置参数还会让 ``structured_model`` 被误当成第二个位置实参 ——
        透传则与框架之间只隔一层，谁改了都看得出来。

        Args:
            *args: 透传给内层模型的位置参数。
            **kwargs: 透传给内层模型的关键字参数。

        Returns:
            `Any`: 内层模型返回的 ``StructuredResponse``。

        Raises:
            ChatCallTimeout: 超过 :attr:`timeout` 仍未返回。
            BaseException: 内层自己抛出的异常，原样上抛（**不**包装成超时）。
        """
        return await self._await_within_budget(
            self._inner.generate_structured_output(*args, **kwargs),
        )

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """在线内调用内层模型，超时即放弃。

        ⚠️ 重排路径**不用**这个方法（它走 ``generate_structured_output``）。
        提供它是为了「套上护栏之后对象仍然是一个能用的对话模型」——
        否则任何一次顺手 ``await model(msgs)`` 都会绕开截止时间，
        而绕开是**静默**的：没有任何迹象说明这次调用没被保护。

        ⚠️ 若内层返回**流式**异步生成器，截止时间只覆盖到「拿到生成器」为止，
        迭代过程中的等待不受保护 —— 与 ``ModelTimeoutMiddleware``
        把截止时间一路带到分片上的做法不同。这是刻意收窄的：
        本类服务于重排（非流式、结构化输出），把流式也一并管起来
        需要引入 ``_guard_stream`` 那一整套，而**没有任何调用方需要它**。
        需要流式护栏的路径请走 ``ModelTimeoutMiddleware``。
        """
        return await self._await_within_budget(self._inner(*args, **kwargs))

    async def _await_within_budget(self, awaitable: Any) -> Any:
        """在预算内等一个可等待对象出结果，超时就放弃。

        ⚠️⚠️ **不能用 ``asyncio.wait_for``** —— 理由见模块文档：
        框架的对话模型基类吞掉 ``CancelledError``，``wait_for`` 的超时
        判定永远不成立。这里把调用放进**独立任务**，按 ``asyncio.wait``
        的钟判定，超时由我们自己抛。

        ⚠️ 外层被取消（用户点了「停止生成」）时要 ``cancel()`` 内层：
        我们已经在返回了，那条到上游的连接不该继续挂着。

        Args:
            awaitable (`Any`): 被等待的协程。

        Returns:
            `Any`: 内层结果（若在预算内结束）。

        Raises:
            ChatCallTimeout: 超过预算时。
            BaseException: 内层自己抛出的异常，原样上抛。
        """
        # ⚠️ 已经是 Task 的不要再包（``ensure_future`` 对 Task 是恒等），
        # 但协程必须包 —— 否则超时后我们没有任何句柄可以取消。
        task = asyncio.ensure_future(awaitable)
        try:
            done, _ = await asyncio.wait({task}, timeout=self._timeout)
        except BaseException:
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
        logger.warning(
            "模型 %r 超过 %.1fs 仍未返回，已放弃本次调用；"
            "重排是尽力而为的，框架会退回向量序（不会让检索整体失败）。",
            self.model,
            self._timeout,
        )
        raise ChatCallTimeout(self._timeout, model=self.model)

    # ------------------------------------------------------------------
    # 透传
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """把未被本层接管的属性透传给内层模型。

        ⚠️ 覆盖不到的是 ``model`` / ``inner`` / ``timeout`` —— 那三个已在
        本层显式定义，正常属性查找会先命中，不会走到这里。

        ⚠️ 构造期由 ``self.__dict__`` 直接兜一道：内层引用尚未就绪时
        抛 ``AttributeError``，而不是无限递归。

        Args:
            name (`str`): 属性名。

        Returns:
            `Any`: 内层模型上的同名属性。

        Raises:
            AttributeError: 内层引用尚未就绪，或内层也没有这个属性。
        """
        inner = self.__dict__.get("_inner")
        if inner is None:  # pragma: no cover —— 只有构造期异常路径会走到
            raise AttributeError(name)
        return getattr(inner, name)

    def __repr__(self) -> str:
        """返回一个能直接看出「被包着」的表示。

        ⚠️ 默认的 ``object.__repr__`` 只会给出内存地址，线上排障时
        看不出这个对象还带着一层截止时间。
        """
        return (
            f"BoundedChatModel(model={self.model!r}, "
            f"timeout={self._timeout:g}s)"
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


def bound_chat_model(inner: ChatModelBase, *, timeout: float) -> Any:
    """给对话模型套上截止时间（幂等：已套过的直接返回）。

    ⚠️ 幂等是为了挡住**双重包装**：套两层不会算错，但会让「超时到底是
    哪一层报的」在日志里失去唯一答案，而且超时值会变成两层的较小者 ——
    排查时看到 60s 的配置却 30s 就超时，方向会被引向错误的地方。

    Args:
        inner (`ChatModelBase`): 被包裹的对话模型。
        timeout (`float`): 单次调用的截止时间（秒），必须 > 0。

    Returns:
        `BoundedChatModel`: 带护栏的对话模型。
    """
    if isinstance(inner, BoundedChatModel):
        return inner
    return BoundedChatModel(inner, timeout=timeout)


def unwrap_chat_model(model: Any) -> Any:
    """剥掉本模块的包装，返回**真正的实现**。

    ⚠️ 需要 ``isinstance(..., ChatModelBase)`` 的地方必须经过它 ——
    本类刻意是纯代理（理由见模块文档），包装后**不是** ``ChatModelBase``
    的实例。

    Args:
        model (`Any`): 任意对象（对话模型、包装层、或别的东西）。

    Returns:
        `Any`: 未被包装的内层对象；没有包装时原样返回。
    """
    current = model
    # ⚠️ 用循环而不是单次 ``isinstance``：将来若再叠一层包装
    # （例如给重排模型也加熔断），单次判断会停在中间那层上。
    while isinstance(current, BoundedChatModel):
        current = current.inner
    return current
