# -*- coding: utf-8 -*-
"""给向量模型的**调用**加超时 —— 检索路径上第二处无界等待的护栏。

═══ ⚠️ 为什么需要它：检索路径上有**两处**无界等待 ═══

``src/knowledge/guard.py`` 给向量库的读操作箍上了超时与熔断，但一次检索
在到达向量库**之前**还有一步：

    框架 ``KnowledgeBase.search``（``agentscope/rag/_knowledge.py:238``）::

        await self.ensure_collection()                       # ← 已被 guard 覆盖
        response = await self._embedding_model(queries)      # ← 本模块覆盖的是这里
        results_per_query = await asyncio.gather(
            *(... self._vector_store.search(...)),           # ← 已被 guard 覆盖
        )

也就是说：**向量模型卡住 = 整个检索卡住**，而 :class:`~src.knowledge.guard.GuardedVectorStore`
完全没有机会介入 —— 它保护的调用压根还没开始。

这不是假想的故障。两条真实路径都是**无界**的：

  · ``dashscope`` 档（``agentscope/embedding/_dashscope/_model.py:371``）走
    ``asyncio.to_thread(dashscope.embeddings.TextEmbedding.call, ...)`` ——
    一个同步 HTTP 调用，超时值由 dashscope SDK 自己决定，我们控制不到；
    且框架基类还会重试（``agentscope/embedding/_embedding_base.py:330``：
    ``max_retries=3`` ⇒ 最多 4 轮 × 每轮 ``retry_delay=1.0s``）。
  · ``local`` 档（``src/web_embedding/local.py``）走 ``asyncio.to_thread``
    把 ONNX 推理丢进线程池 —— 推理本身有界，但线程池**排队**没有：
    若干个并发请求 + 一次权重加载就足以让它长时间拿不到线程。

═══ 与模型调用中间件的关键区别：这里**可以**用 ``asyncio.wait_for`` ═══

``src/llm/middleware.py`` 里的 ``ModelTimeoutMiddleware`` 被迫用「按钟判断」
（``asyncio.wait``）而不是 ``wait_for``，因为框架的**对话**模型基类会**吞掉**
``CancelledError``（``agentscope/model/_base.py:219-224`` / ``:283-288``），
于是 ``wait_for`` 的超时永远触发不了。

向量模型**没有**这个毛病，而且是可核验的：``agentscope/embedding/_embedding_base.py:330`` 的重试
只捕获 ``Exception``（``except Exception as e``），而 ``CancelledError``
自 3.8 起继承自 ``BaseException`` —— 它会穿过重试循环，让 ``wait_for``
正常判定超时。**所以这里刻意用更简单的 ``wait_for``**：
它能顺带把取消传播进内层（线程池里的等待会被放弃），比按钟判断更干净。
⚠️ 改动框架版本后要重新核验这一条 —— 它是本模块选择实现方式的前提。

═══ ⚠️ 超时**不会**让底层那次 HTTP / 推理停下来 ═══

与 ``guard.py`` 的诚实说明同源：``asyncio.wait_for`` 取消的是**等待**，
线程池里那个线程会把它手上的活干完（线程不可取消）。
危害是「卡死的那几轮会一直占着线程」，好处是**调用方立刻拿到失败**
而不是陪着一起等 —— 对一个要回应用户的服务来说，后者才是要命的那个。
"""

from __future__ import annotations

from typing import Any

from agentscope.embedding import EmbeddingModelBase

__all__ = [
    "BoundedEmbeddingModel",
    "EmbeddingCallTimeout",
    "bound_embedding_model",
    "unwrap_embedding_model",
]


class EmbeddingCallTimeout(TimeoutError):
    """向量模型调用超时。

    ⚠️ 与 :class:`~src.knowledge.guard.VectorSearchTimeout` 一样继承内置的
    :class:`TimeoutError`：上层的宽口径 ``except Exception``（框架的 RAG
    中间件、:mod:`src.knowledge.rag` 的降级路径）不必知道本模块存在，
    也能把「超时」正确地当成一次可降级的检索失败。
    """

    def __init__(self, timeout: float, *, model: str) -> None:
        self.timeout = timeout
        self.model = model
        super().__init__(
            f"向量模型 {model!r} 超过 {timeout:g}s 未返回，已放弃本次向量化"
            f"（底层调用可能仍在进行，见 src/web_embedding/bounded.py 的模块文档）。",
        )


class BoundedEmbeddingModel(EmbeddingModelBase):
    """给任意向量模型加「单次调用截止时间」的透明包装。

    ⚠️ 刻意**继承** ``EmbeddingModelBase`` 而不是做纯代理：框架对向量模型的
    使用不只是「调一下」——``KnowledgeBase`` 读它的
    ``dimensions``（``agentscope/rag/_knowledge.py:182``，建集合时用）与
    ``supports_multimodal``（``:232-233``，决定要不要丢掉 ``DataBlock``），
    这两个属性必须在包装层上**真的存在**。
    ``src/knowledge/guard.py`` 那边框架只做鸭子类型使用、可以纯代理；
    这里不行，所以老老实实继承。

    ⚠️ 包装之后 ``type(model).__name__`` 会变成 ``BoundedEmbeddingModel``，
    而 :mod:`src.knowledge.rag` 的探针正是靠类名回答「实际用的是哪一档」、
    靠 ``isinstance(..., MockEmbeddingModel)`` 回答「是不是掉进了没有语义的
    假向量档」。**那种诊断不能因为加了一层包装就失效** ——
    它恰恰是唯一能发现「线上悄悄在用假向量」的线索。
    所以本模块提供 :func:`unwrap_embedding_model`，读实现类一律经过它。
    """

    def __init__(
        self,
        inner: EmbeddingModelBase,
        *,
        timeout: float,
    ) -> None:
        """装配护栏。

        Args:
            inner (`EmbeddingModelBase`): 被包裹的向量模型。
            timeout (`float`): 单次 ``__call__`` 的截止时间（秒），必须 > 0。

        Raises:
            ValueError: ``timeout`` 非正时。取 0 会让每次向量化**必然**超时，
                等价于静默关掉检索能力 —— 这种配置必须当场炸
                （与 :class:`~src.knowledge.guard.GuardedVectorStore`
                对 ``milvus.search_timeout_seconds`` 的处理方式一致）。
        """
        if timeout <= 0:
            raise ValueError(
                f"向量模型超时必须 > 0，实际为 {timeout!r}；"
                f"取 0 会让每次向量化都超时（等价于静默关闭检索能力）。",
            )

        # ⚠️ 先登记内层引用，**再**调基类构造：:meth:`__getattr__` 会在属性
        # 查找失败时访问 ``self._inner``，而基类构造中途若访问到任何尚未
        # 赋值的属性，就会掉进「找 ``_inner`` → 又走 ``__getattr__``」的
        # 无限递归（RecursionError 的信息完全指不出真正的原因）。
        self._inner = inner
        self._timeout = float(timeout)

        # ⚠️ 把内层的身份属性**复制**到本层，而不是「用到时再透传」：
        # ``dimensions`` / ``model`` 会被框架直接读取并参与决策
        # （建集合的维度、日志与错误信息里的模型名），
        # 靠 ``__getattr__`` 兜底虽然也能拿到值，但读代码的人无法一眼
        # 看出这些属性是「有的」；显式复制让契约立在类上。
        super().__init__(
            credential=inner.credential,
            model=inner.model,
            dimensions=inner.dimensions,
            parameters=inner.parameters,
            context_size=inner.context_size,
            batch_size=inner.batch_size,
            max_retries=inner.max_retries,
            retry_delay=inner.retry_delay,
        )
        # ⚠️ ``supports_multimodal`` 是**实例**属性（子类按模型名在
        # ``__init__`` 里决定，见 ``agentscope/embedding/_dashscope/_model.py:160``），
        # 基类的类默认值恒为 False —— 不复制的话，多模态档经过包装
        # 会被当成纯文本模型，``KnowledgeBase.search`` 会**静默丢掉**
        # 所有 ``DataBlock`` 输入，表现为「图片检索永远没有结果」。
        self.supports_multimodal: bool = bool(
            getattr(inner, "supports_multimodal", False),
        )

    @property
    def inner(self) -> EmbeddingModelBase:
        """被包裹的向量模型（供 :func:`unwrap_embedding_model` 与测试使用）。"""
        return self._inner

    @property
    def timeout(self) -> float:
        """单次调用的截止时间（秒）。"""
        return self._timeout

    async def __call__(self, inputs: Any, **kwargs: Any) -> Any:
        """调用内层模型，但**最多等** :attr:`timeout` 秒。

        ⚠️ 整体包住一次 ``__call__``，而不是去包内层的 ``_call_api``：
        后者的调用次数取决于批切分（``batch_size``）与重试次数，
        按「批」计时会让「一次请求最多等多久」变成一个乘出来的数
        （批数 × 超时），那正是本次要消灭的「不可预测的最坏情况」。
        包在 ``__call__`` 外层，语义就是调用方真正关心的那个：
        **我这次等多久能拿到结果**。

        Args:
            inputs: 与内层模型相同的输入（``str`` / ``TextBlock`` / ``DataBlock``）。
            **kwargs: 透传给内层模型。

        Returns:
            内层模型返回的 ``EmbeddingResponse``。

        Raises:
            EmbeddingCallTimeout: 超过 ``timeout`` 仍未返回。
        """
        import asyncio

        try:
            return await asyncio.wait_for(
                self._inner(inputs, **kwargs),
                self._timeout,
            )
        except TimeoutError as exc:
            # ⚠️ 只转换**超时**这一种，其余异常原样透出：
            # 把内层的真实错误（缺 key、模型下线、维度不符）包成超时，
            # 会让排查方向整个错掉 —— 那是最贵的一种信息损失。
            raise EmbeddingCallTimeout(self._timeout, model=self.model) from exc

    async def _call_api(self, inputs: Any, **kwargs: Any) -> Any:
        """透传到内层的 ``_call_api``（基类的抽象方法，必须实现）。

        ⚠️ 本类的正常路径是 :meth:`__call__`（已带截止时间）。
        这里保留透传是为了让本类成为一个**完整**的 ``EmbeddingModelBase``
        —— 万一有代码直接走基类的 ``__call__`` 语义（它按 ``batch_size``
        切片后调 ``_call_api``），行为与包装前一致，而不是抛
        ``NotImplementedError`` 那种「包装一下就废了一半」的坑。
        那条路**没有**超时护栏，这是已知且接受的：本项目自己的装配
        （``src/web_embedding/factory.py``、``src/knowledge/manager.py``）
        一律走 :meth:`__call__`。

        Args:
            inputs (`Any`): 单批输入。
            **kwargs: 透传。

        Returns:
            内层 ``_call_api`` 的返回值。
        """
        return await self._inner._call_api(inputs, **kwargs)  # noqa: SLF001 —— 见上面的说明

    # ------------------------------------------------------------------
    # 透传
    # ------------------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        """把未被本层显式接管的属性透传给内层模型。

        ⚠️ ``__init__`` 已经**复制**了框架真正会读的那几个
        （``dimensions`` / ``model`` / ``supports_multimodal`` …），
        这里兜的是剩下的、以及**将来**框架新增的属性 ——
        例如 dashscope 档的 ``embedding_cache``、各档自己的 ``Parameters``。
        没有它的话，任何一处 ``model.某个内层属性`` 都会变成
        ``AttributeError``，而症状会出现在离包装层很远的地方。

        ⚠️ 只在正常属性查找失败时被调用，所以 ``_inner`` / ``_timeout``
        不会走这里；构造期由 ``self.__dict__`` 直接兜一道，
        保证「尚未赋值就访问」时抛的是 ``AttributeError`` 而不是无限递归。

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


def bound_embedding_model(
    inner: EmbeddingModelBase,
    *,
    timeout: float,
) -> BoundedEmbeddingModel:
    """给向量模型套上截止时间（幂等：已套过的直接返回）。

    ⚠️ 幂等是为了挡住**双重包装**：``manager.get_knowledge`` 每次调用都会
    构造一次向量模型，而它构造出来的对象可能已经被 ``factory`` 包过一层。
    套两层不会算错，但会让「超时到底是哪一层报的」在日志里失去唯一答案，
    而且超时值会变成两层的较小者 —— 排查时看到 30s 的配置却 5s 就超时，
    方向会被引向错误的地方。

    Args:
        inner (`EmbeddingModelBase`): 被包裹的向量模型。
        timeout (`float`): 单次调用的截止时间（秒），必须 > 0。

    Returns:
        `BoundedEmbeddingModel`: 带护栏的向量模型。
    """
    if isinstance(inner, BoundedEmbeddingModel):
        return inner
    return BoundedEmbeddingModel(inner, timeout=timeout)


def unwrap_embedding_model(model: Any) -> Any:
    """剥掉本模块的包装，返回**真正的实现**。

    ⚠️ 所有「这是哪一档」的判断都必须经过它 —— 类名与 ``isinstance``
    是 :mod:`src.knowledge.rag` 用来发现「降级到了假向量」的唯一线索
    （见 :class:`BoundedEmbeddingModel` 的说明）。加一层包装就让它失效，
    等于把一条安全告警静默关掉。

    Args:
        model (`Any`): 任意对象（向量模型、包装层、或别的东西）。

    Returns:
        `Any`: 未被包装的内层对象；没有包装时原样返回。
    """
    current = model
    # ⚠️ 用循环而不是 ``if isinstance`` 单次判断：将来若再叠一层包装
    # （比如给向量模型也加熔断），单次判断会停在中间那层上，
    # 症状与「包装后就失效」完全一样，却更难看出来。
    while isinstance(current, BoundedEmbeddingModel):
        current = current.inner
    return current
