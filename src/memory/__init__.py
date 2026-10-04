# -*- coding: utf-8 -*-
"""长期记忆层：自研差旅画像（结构化 + 语义）+ 可选的框架 ReMe 适配。

═══ 本层在整体架构里的位置 ═══

::

    AgentMiddlewareFactory(user_id, agent_id, session_id)   ← 框架传入 user_id
        │
        ├─ ContextInjectionMiddleware(resolver=…)
        │      └─ make_memory_resolver(base, memory, user_id)   ← resolver.py
        │             └─ TravelerMemory.render_prompt_section()
        │                    ├─ SqlProfileRepository  → Postgres business schema
        │                    └─ SemanticMemory        → Milvus {集合}_memory
        ▼
    PromptContext.profile_summary  →  build_system_prompt()  →  system prompt

═══ 为什么是**自研**，而不是直接用框架的三套中间件 ═══

框架提供了 ``AgenticMemoryMiddleware`` / ``Mem0Middleware`` / ``ReMeMiddleware``
（:mod:`agentscope.middleware._longterm_memory`）。本项目**以自研为主**，
理由写在 ``MemorySettings`` 的文档里，并在 :mod:`src.memory.reme` 里
展开。一句话：那三套各自带着自己的存储形态（ReMe 落文件、mem0 落它自己的
向量库），一旦成为主路径，用户画像就不再是我方可查可改的数据 ——
而审批流、成本中心、常旅客号全都建立在「画像是一条我们能读写的记录」之上。

═══ ⚠️ 读路径永不抛，写路径照常抛 ═══

这是本层最要紧的一条不变量，贯穿三个模块：

  · 读（召回画像注入 prompt）失败 ⇒ 降级为「没有画像」，对话继续；
  · 写（用户说「记住这个」）失败 ⇒ 原样抛出，让调用方知道没记住。

理由写在 :meth:`src.memory.semantic.SemanticMemory.recall` 的文档里。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from .profile import (
    InMemoryProfileRepository,
    ProfilePatch,
    ProfileRepository,
    TravelerProfile,
)
from .reme import ReMeBundle, build_reme, reme_available
from .repository import SqlProfileRepository, ensure_profile_table, profile_table
from .resolver import last_user_text, make_memory_resolver
from .semantic import (
    MEMORY_KIND_KEY,
    MEMORY_USER_KEY,
    MemoryKind,
    MemoryNote,
    SemanticMemory,
    SemanticRecall,
    memory_collection,
    note_id,
    render_profile_section,
)
from .service import MemoryContext, TravelerMemory

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from src.config.schema import Settings

__all__ = [
    "MEMORY_KIND_KEY",
    "MEMORY_USER_KEY",
    "InMemoryProfileRepository",
    "MemoryContext",
    "MemoryKind",
    "MemoryNote",
    "ProfilePatch",
    "ProfileRepository",
    "ReMeBundle",
    "SemanticMemory",
    "SemanticRecall",
    "SqlProfileRepository",
    "TravelerMemory",
    "TravelerProfile",
    "build_memory",
    "build_reme",
    "ensure_profile_table",
    "last_user_text",
    "make_memory_resolver",
    "memory_collection",
    "note_id",
    "profile_table",
    "reme_available",
    "render_profile_section",
]

logger = logging.getLogger(__name__)


def build_memory(
    settings: "Settings",
    engine: "AsyncEngine | Callable[[], AsyncEngine | None] | None" = None,
) -> TravelerMemory:
    """装配长期记忆门面。

    ⚠️ 本函数**不做任何 I/O**，因此可以在应用装配期安全调用：

      · ``build_vector_store`` 只存参数、不连 Milvus（见 :mod:`src.knowledge.store`）；
      · 画像表的创建是**另一个**函数（:func:`ensure_profile_table`），
        由 lifespan 在数据库就绪之后调用 —— 装配期建表会让
        「数据库没起来」变成「应用起不来」。

    ⚠️ ``engine`` 为 ``None`` 时退回**内存**画像仓储。这不是「还没做完」：
    ``make test`` 与不接数据库的离线脚本走的正是这条路，
    而它们需要的是**确定性**而不是持久化（见
    :class:`~src.memory.profile.InMemoryProfileRepository` 的文档）。

    ⚠️ ``engine`` 也可以传一个**无参可调用对象**。生产路径必须用这个形态 ——
    业务库引擎是在 lifespan 里创建的，而本函数在 ``create_app`` **之前**
    就被调用（中间件工厂只被读一次），装配期拿不到那个引擎。
    详见 :class:`~src.memory.repository.SqlProfileRepository` 的类文档。

    ⚠️ 语义那半**失败也不致命**。它需要向量模型，而向量模型可能在
    降级链上全部失败（``EmbeddingUnavailableError``）。那时候
    记忆退化成「只有结构化画像」—— 比整个应用起不来好得多，
    而 ``describe()`` 会把这件事报出来。

    Args:
        settings (`Settings`): 配置。
        engine (`AsyncEngine | None`, optional): 业务库引擎。
            传入则用 PostgreSQL 画像仓储；否则用内存实现。

    Returns:
        `TravelerMemory`: 可直接交给中间件工厂的门面。
    """
    if not settings.memory.enabled:
        # ⚠️ 关闭时仍然返回一个**完整的对象**（而不是 None）——
        # 让调用方不必到处写 ``if memory is not None``，
        # 而 ``render_prompt_section`` 会自然地返回空串。
        logger.info("长期记忆已关闭（ALIGO__MEMORY__ENABLED=false）。")
        return TravelerMemory(settings)

    repository: ProfileRepository
    if engine is not None:
        from src.storage.engine import business_schema_for

        repository = SqlProfileRepository(engine, business_schema_for(settings))
    else:
        repository = InMemoryProfileRepository()

    semantic = _build_semantic(settings)

    memory = TravelerMemory(
        settings,
        repository=repository,
        semantic=semantic,
    )
    logger.info("长期记忆已装配：%s", memory.describe())
    return memory


def _build_semantic(settings: "Settings") -> SemanticMemory | None:
    """尽力构造语义记忆；失败时返回 None 并告警。

    ⚠️ 这里**必须**吞异常，与「写路径照常抛」并不矛盾：
    被吞掉的是**装配**失败（向量模型不可用 ⇒ 这个进程没有语义记忆），
    而用户主动要求的「记住这句话」走的是另一条路径，那里照样会抛。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `SemanticMemory | None`: 语义记忆；不可用时 None。
    """
    try:
        # ⚠️ 延迟 import：``src.knowledge`` 与 ``src.web_embedding`` 都会
        # 拉起 agentscope 的 rag / embedding 子系统，而
        # ``src/memory.profile`` 单测不需要它们。
        from src.knowledge.store import build_vector_store
        from src.web_embedding import build_embedding_model

        return SemanticMemory(
            vector_store=build_vector_store(settings),
            embedding_model=build_embedding_model(settings),
            settings=settings,
        )
    except Exception as exc:  # noqa: BLE001 —— 见文档
        from src.observability.redaction import safe_error

        logger.warning(
            "⚠️ 向量模型不可用，长期记忆降级为「仅结构化画像」"
            "（语义召回失效，但画像字段照常工作）：%s",
            safe_error(exc),
        )
        return None
