# -*- coding: utf-8 -*-
"""把文本变成向量的**三合一**降级链。

对外只暴露一个构造函数::

    from src.web_embedding import build_embedding_model

    model = build_embedding_model(settings)   # -> EmbeddingModelBase

═══ 为什么是「一条链」而不是「一个开关」 ═══

检索能力要在三种环境下都能跑起来，而这三种环境的**能力不同**：

    · 正式环境   有 DashScope key，走云端向量模型（质量最好）
    · 开发机     没有 key，但可以下载本地 ONNX 模型
    · CI / 冒烟  既没 key 也不该下载 100MB 模型

如果让调用方自己 if/else 选，这三种环境会长出三套不同的代码路径，
而它们的差异恰恰在**最容易出错的地方**（维度、归一化、批量大小）。
做成一条链之后，调用方拿到的永远是一个 :class:`EmbeddingModelBase`，
三种环境跑的是同一段上层代码。

═══ ⚠️ 三档的产物维度必须**完全一致** ═══

Milvus 的集合维度在**建集合那一刻**定死。三档若输出不同维度，
症状是「换个环境就写不进去」或更糟的「写进去了但召回全错」。
所以 :class:`~src.config.schema.EmbeddingSettings` 的 ``dimension``
会在启动时与 ``milvus.dimension`` 比对（见 ``Settings`` 的 validator），
而这里的三档**都必须**用它构造，不许各写各的默认值。

═══ ⚠️ Mock 档是**危险**的，默认只该出现在测试里 ═══

:class:`~src.web_embedding.mock.MockEmbeddingModel` 产出的是**确定性假向量**：
同一个字符串永远得到同一个向量（所以测试可复现），但
**语义相近的两段文本得到的向量没有任何关系**。

也就是说，Mock 档下检索会「正常工作」—— 不报错、有结果、有分数 ——
只是那些结果**没有意义**。这比检索直接失败难发现得多。
生产环境请把 ``embedding.allow_fallback`` 设为 ``false``：
宁可起不来，也不要一个安静的、看起来正常的错误答案。

═══ 与框架的分工 ═══

云端那一档**完全**用框架的实现
（``agentscope.embedding.DashScopeEmbeddingModel``，见 :mod:`src.web_embedding.factory`），
我们一行都不重写。只有框架**没有提供**的两档（本地 ONNX、确定性假向量）
才是本模块自己的代码，且它们继承框架的
``EmbeddingModelBase`` —— 只实现 ``_call_api`` 一个方法，
批切分、重试、并发、``TextBlock`` 解包全部由基类的 ``__call__`` 完成。

═══ ⚠️ 无论哪一档，出厂前都会套上**截止时间** ═══

:func:`build_embedding_model` 的**每一条**出口都经过
:func:`~src.web_embedding.bounded.bound_embedding_model`：
检索路径是「先向量化、再查库」，而框架的向量模型调用是无界的
（``dashscope`` 档是同步 HTTP 丢进线程池，本地档是 ONNX 推理排队）。
只箍住查库那一段，向量模型卡住时请求照样假死 ——
完整论证见 :mod:`src.web_embedding.bounded` 的模块文档。
"""

from __future__ import annotations

from .bounded import (
    BoundedEmbeddingModel,
    EmbeddingCallTimeout,
    bound_embedding_model,
    unwrap_embedding_model,
)
from .factory import (
    EmbeddingUnavailableError,
    build_embedding_model,
    describe_embedding,
)
from .local import LocalOnnxEmbeddingModel
from .mock import MockEmbeddingModel

__all__ = [
    "BoundedEmbeddingModel",
    "EmbeddingCallTimeout",
    "EmbeddingUnavailableError",
    "LocalOnnxEmbeddingModel",
    "MockEmbeddingModel",
    "bound_embedding_model",
    "build_embedding_model",
    "describe_embedding",
    "unwrap_embedding_model",
]
