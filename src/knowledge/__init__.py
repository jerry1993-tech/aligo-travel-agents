# -*- coding: utf-8 -*-
"""知识库层：Milvus 接入 + 单集合隔离策略 + 就绪探针。

对外三个入口::

    from src.knowledge import build_knowledge_manager        # 应用装配
    from src.knowledge import probe_collection               # /readyz 探针
    from src.knowledge import ensure_collection              # 初始化脚本

═══ 本层在整体架构里的位置 ═══

::

    src/server/app.py
        │  create_app(knowledge_base_manager=...)
        ▼
    SingleCollectionKbManager  ←── 本层
        │  ├ storage（KB 记录与凭据）
        │  └ VectorStoreBase（MilvusLiteStore，全体 KB 共用）
        ▼
    KnowledgeBase（框架的运行时句柄，带 metadata_filter 作用域）

═══ 一条贯穿本层的约束 ═══

**Milvus 不可用不能拖垮服务。** 对话、订单、审批、鉴权全都不经过向量库；
只有「查差旅政策」这一条路径需要它。所以：

    · 构造向量库客户端不发网络请求（惰性连接，``store.py`` 的模块文档）；
    · 知识库管理器进入生命周期时**不**连接（``VectorStoreBase.__aenter__``
      是 no-op，``MilvusLiteStore`` 没有覆盖它）；
    · ``/readyz`` 走 :func:`probe_collection`，它有超时且**永不抛**。

这三条合起来的效果是：Milvus 挂掉时，服务照常启动、照常服务其它请求，
``/readyz`` 报出一条 Milvus 相关的警告而不是把自己也拖成不健康。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .manager import KB_ID_KEY, USER_ID_KEY, SingleCollectionKbManager
from .rag import RagBridgeStatus, build_rag_middlewares, clear_handle_cache
from .store import (
    CONSISTENCY_LEVEL,
    build_vector_store,
    describe_collection,
    describe_vector_store,
    ensure_collection,
    probe_collection,
    request_strong_consistency,
    verify_collection_shape,
)

if TYPE_CHECKING:
    from agentscope.app.storage import StorageBase

    from src.config.schema import Settings

__all__ = [
    "CONSISTENCY_LEVEL",
    "KB_ID_KEY",
    "USER_ID_KEY",
    "RagBridgeStatus",
    "SingleCollectionKbManager",
    "build_knowledge_manager",
    "build_rag_middlewares",
    "build_vector_store",
    "clear_handle_cache",
    "describe_collection",
    "describe_vector_store",
    "ensure_collection",
    "probe_collection",
    "request_strong_consistency",
    "verify_collection_shape",
]


def build_knowledge_manager(
    settings: "Settings",
    storage: "StorageBase",
    *,
    access_policy_provider: "Callable[[], Any] | None" = None,
) -> SingleCollectionKbManager:
    """装配知识库管理器。

    ⚠️ 本函数**不做任何 I/O**：向量库客户端是惰性连接的
    （``MilvusLiteStore.__init__`` 只存参数），所以它可以在应用装配期
    安全调用，不会因为 Milvus 没起来而失败、也不会拖慢启动。

    ⚠️ ``storage`` 由调用方传入而不是在这里构造 —— 因为
    ``create_app`` 需要的**就是同一个** storage 实例。各造一个的后果是
    「KB 记录写进了 A，对话链路从 B 里读」，症状是「刚建的知识库
    列表里看得见、检索时说找不到」。

    Args:
        settings (`Settings`): 配置。
        storage (`StorageBase`): 应用级存储（必须与 ``create_app`` 的是同一个）。
        access_policy_provider (`Callable[[], Any] | None`): 延迟取资源访问
            策略的函数 —— 它让「KB 绑定的 embedding 凭据来自**共享**」
            这件事能被解析出来（详见
            :meth:`~src.knowledge.manager.SingleCollectionKbManager._resolve_embedding_credential`）。
            传函数而不是策略对象，是因为策略由 ``create_app`` 在
            **本函数之后**写进 ``app.state``；默认 ``None`` 时行为与框架
            自带管理器一致（只认属主），离线脚本用这个默认值即可。

    Returns:
        `SingleCollectionKbManager`: 可直接交给 ``create_app`` 的管理器。
    """
    return SingleCollectionKbManager(
        storage=storage,
        vector_store=build_vector_store(settings),
        settings=settings,
        access_policy_provider=access_policy_provider,
    )
