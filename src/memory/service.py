# -*- coding: utf-8 -*-
"""长期画像的门面 —— 把「结构化的权威事实」与「语义的零散提及」合成一段提示。

═══ 这个模块要解决的问题不是「怎么存」，而是「怎么合并」 ═══

存的两半分别在 :mod:`src.memory.profile`（字段、可校验、可更新）与
:mod:`src.memory.semantic`（向量、模糊、只作提示）。本模块负责把
两半在**每一次对话**里合成一个字符串，交给
:class:`~src.orchestration.context.PromptContext` 的 ``profile_summary``。

═══ ⚠️ 三条合并原则（每一条都对应一类会真实发生的错） ═══

  1. **结构化永远赢**。两者冲突时以结构化为准，并且在渲染出来的
     文本里**明确写出**这条优先级。不写的话，模型看到两段互相
     矛盾的话（「偏好舱位：ECONOMY」与「他上次说要坐商务舱」），
     会挑一个**更显眼**的，而不是更权威的。

  2. **读失败必须降级，不能中断对话**。语义召回失败时，
     结构化画像照常返回 —— 用户拿到的是一个「少了点个性化」
     但**能用**的助手，而不是一个报错的助手。

  3. **没有信息就什么都不注入**。见
     :func:`~src.memory.semantic.render_profile_section`：没有画像时
     返回空串。往系统提示里塞「用户画像：无」不只是浪费 token，
     它还会让模型以为「这个用户是特意说过自己没有任何偏好的」。

═══ ⚠️ 关于「写」的不对称 ═══

本模块的读路径（:meth:`TravelerMemory.recall`）**永不抛异常**；
写路径（:meth:`TravelerMemory.remember` / :meth:`TravelerMemory.forget` /
:meth:`TravelerMemory.forget_all` / :meth:`TravelerMemory.update_profile` /
:meth:`TravelerMemory.put_profile`）**照常抛**。这个不对称是刻意的，理由写在
:meth:`~src.memory.semantic.SemanticMemory.recall` 的文档里：
读失败只影响锦上添花，写失败是欺骗。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.config.schema import Settings

from .profile import ProfilePatch, ProfileRepository, TravelerProfile
from .semantic import (
    MemoryKind,
    MemoryNote,
    SemanticMemory,
    SemanticRecall,
    render_profile_section,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryContext:
    """一次召回的全部结果。

    ⚠️ 把「结构化画像」与「语义笔记」分成两个字段而不是先合并成一个
    字符串，是为了让**调用方**（中间件、评测、调试接口）能分别使用：
    例如评测脚本要断言「画像里确实有常旅客号」，而那是
    语义笔记里**永远不该**出现的东西（它是硬事实，见
    :mod:`src.memory.profile` 的模块文档）。
    """

    #: 结构化画像；没有画像记录时为 None（注意：与「空画像」不同）。
    profile: TravelerProfile | None = None
    #: 语义召回结果（可能失败，见 :attr:`SemanticRecall.error`）。
    recall: SemanticRecall = SemanticRecall()

    @property
    def semantic_error(self) -> str | None:
        """语义那半失败时的原因；成功时为 None。"""
        return self.recall.error

    @property
    def is_empty(self) -> bool:
        """两半都没有可用信息。"""
        has_profile = self.profile is not None and not self.profile.is_empty()
        return not has_profile and not self.recall.notes

    def render(self, max_notes: int = 3) -> str:
        """渲染成可以塞进 Prompt 的一段文本。

        Args:
            max_notes (`int`, optional): 最多渲染几条语义笔记。

        Returns:
            `str`: 文本段；没有信息时是空串。
        """
        return render_profile_section(self.profile, self.recall, max_notes=max_notes)


class TravelerMemory:
    """差旅长期记忆的门面。

    ⚠️ 构造时**两个依赖都可以是 None**：

      · ``repository=None`` —— 本进程不读写结构化画像；
      · ``semantic=None`` —— 本进程不做语义召回（比如
        ``ALIGO__MEMORY__ENABLED=false``，或者向量模型起不来）。

    这两种 None 都不是「忘了传参数」，而是**合法的部署形态**：
    一个只跑评测、不碰画像的进程不该因为有 None 就崩掉。
    """

    def __init__(
        self,
        settings: Settings,
        repository: ProfileRepository | None = None,
        semantic: SemanticMemory | None = None,
    ) -> None:
        """初始化。

        Args:
            settings (`Settings`): 配置。
            repository (`ProfileRepository | None`, optional): 结构化画像仓储。
            semantic (`SemanticMemory | None`, optional): 语义记忆。
        """
        self._settings = settings
        self._repository = repository
        self._semantic = semantic

    @property
    def enabled(self) -> bool:
        """总开关状态（来自 ``ALIGO__MEMORY__ENABLED``）。"""
        return self._settings.memory.enabled

    @property
    def has_semantic(self) -> bool:
        """本实例是否具备语义召回能力。"""
        return self._semantic is not None

    @property
    def has_repository(self) -> bool:
        """本实例是否具备结构化画像能力。"""
        return self._repository is not None

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    async def recall(
        self,
        user_id: str,
        query: str = "",
        top_k: int | None = None,
    ) -> MemoryContext:
        """取回这个用户的全部记忆。

        ⚠️⚠️ **永不抛异常。** 这是本模块最重要的性质，理由见模块文档
        的第 2 条原则。注意下面**两个** try/except 是分开的：
        仓储读失败不该影响语义召回，反之亦然 —— 两个依赖是两个
        独立的故障域（一个是 PostgreSQL，一个是 Milvus + 向量模型），
        共用一个 try 会让「一边挂了」变成「两边都没有」。

        Args:
            user_id (`str`): 员工 id。
            query (`str`, optional): 本轮查询，用于语义召回。空串时
                只返回结构化画像。
            top_k (`int | None`, optional): 语义召回条数；``None`` 时
                取 ``settings.memory.top_k``。⚠️ 这个参数是给
                ``GET /api/v1/memory/notes`` 准备的 —— **显式**接口上
                调用方可以自己决定要几条；对话链路不传它，
                用的是运营者配的默认值。

        Returns:
            `MemoryContext`: 两半的结果（各自可能为空）。
        """
        if not self._settings.memory.enabled:
            # ⚠️ 关掉时返回**空的** MemoryContext，而不是「跳过调用」——
            # 让调用方（中间件）永远拿到一个可以 ``.render()`` 的对象，
            # 不必到处写 ``if memory:``。
            return MemoryContext()

        profile: TravelerProfile | None = None
        if self._repository is not None:
            try:
                profile = await self._repository.get(user_id)
            except Exception as exc:  # noqa: BLE001 —— 见本方法的文档
                from src.observability.redaction import safe_error

                logger.warning(
                    "读结构化画像失败（已降级为「没有画像」）：%s",
                    safe_error(exc),
                )

        recall = SemanticRecall()
        if self._semantic is not None and query.strip():
            recall = await self._semantic.recall(user_id, query, top_k=top_k)

        return MemoryContext(profile=profile, recall=recall)

    async def render_prompt_section(
        self,
        user_id: str,
        query: str = "",
        max_notes: int = 3,
    ) -> str:
        """取回并渲染成一段 Prompt 文本。

        这是 ``ContextInjectionMiddleware`` 实际调用的方法 ——
        它把上面那串「可能出错的 I/O」收成一个**永远不会炸**的
        ``str`` 返回。

        Args:
            user_id (`str`): 员工 id。
            query (`str`, optional): 本轮查询。
            max_notes (`int`, optional): 最多渲染几条语义笔记。

        Returns:
            `str`: 可直接拼进系统提示的文本段；没有信息时是空串。
        """
        context = await self.recall(user_id, query)
        return context.render(max_notes=max_notes)

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    async def remember(
        self,
        user_id: str,
        text: str,
        kind: str = MemoryKind.PREFERENCE,
    ) -> MemoryNote:
        """记住一句话（写语义记忆）。

        ⚠️ 与读路径相反，这里**照常抛异常**。用户明确说了「记住这个」
        的时候，静默失败是一次欺骗 —— 他会在下周发现助手并没有记住，
        而且没有任何地方告诉过他。

        Raises:
            RuntimeError: 本实例没有语义记忆能力。
            Exception: 向量库/模型故障，原样透出。
        """
        if self._semantic is None:
            raise RuntimeError(
                "本实例没有配置语义记忆（ALIGO__MEMORY__ENABLED=false "
                "或向量模型不可用），无法记住新内容。",
            )
        return await self._semantic.remember(user_id, text, kind=kind)

    async def forget(self, user_id: str, text: str) -> None:
        """忘掉一句话（按原文精确匹配）。

        ⚠️ 与 :meth:`remember` 同一条不对称原则：**照常抛异常**。
        「忘记」是一个用户主动要求的动作，静默失败比记住失败更糟 ——
        他以为那条记录没了，而它仍然躺在下一次对话的召回窗口里。

        ⚠️ 只能按**原文**忘（见
        :meth:`~src.memory.semantic.SemanticMemory.forget`）：
        调用方要先 :meth:`recall` 拿到原文。这不是不方便，而是刻意的 ——
        「按相似度删」会误删用户没说过的内容，而删除是不可撤销的。

        Args:
            user_id (`str`): 员工 id。
            text (`str`): 要忘掉的原文（与当初 :meth:`remember` 传入的一致）。

        Raises:
            RuntimeError: 本实例没有语义记忆能力。
            Exception: 向量库故障，原样透出。
        """
        if self._semantic is None:
            raise RuntimeError(
                "本实例没有配置语义记忆（ALIGO__MEMORY__ENABLED=false "
                "或向量模型不可用），无法忘记内容。",
            )
        await self._semantic.forget(user_id, text)

    async def forget_all(self, user_id: str) -> int:
        """忘掉这个用户的**全部**笔记，返回删掉的条数。

        ⚠️ 仍然照常抛异常（理由同上）。返回条数是必要的：调用方
        （``DELETE /api/v1/memory/notes?all=true``）要能回答
        「到底删掉了几条」—— 只回一个 200 的话，用户无法区分
        「清空了 3 条」与「本来就没有任何记录」。

        ⚠️ 命名里带 ``all`` 但作用域是**当前身份**：底层
        :meth:`~src.memory.semantic.SemanticMemory.forget_all` 按
        ``user_id`` 过滤后逐条删，**绝不**碰整个集合 ——
        那个集合里住着所有用户的记忆（见其文档）。

        Args:
            user_id (`str`): 员工 id。

        Returns:
            `int`: 删掉的笔记条数。

        Raises:
            RuntimeError: 本实例没有语义记忆能力。
            Exception: 向量库故障，原样透出。
        """
        if self._semantic is None:
            raise RuntimeError(
                "本实例没有配置语义记忆（ALIGO__MEMORY__ENABLED=false "
                "或向量模型不可用），无法忘记内容。",
            )
        return await self._semantic.forget_all(user_id)

    async def update_profile(
        self,
        user_id: str,
        patch: ProfilePatch,
    ) -> TravelerProfile:
        """部分更新结构化画像。

        Raises:
            RuntimeError: 本实例没有结构化画像能力。
            Exception: 存储故障，原样透出。
        """
        if self._repository is None:
            raise RuntimeError("本实例没有配置结构化画像仓储。")
        return await self._repository.merge(user_id, patch)

    async def get_profile(self, user_id: str) -> TravelerProfile | None:
        """读结构化画像（无记录时返回 None）。

        ⚠️ 与 :meth:`recall` 不同，这里**不吞异常**：它是给
        ``/api/v1/memory/profile`` 这类**显式**接口用的，
        调用方正在等一个答案，把它变成「None」等于把一次故障
        伪装成「这个用户没有画像」。
        """
        if self._repository is None:
            return None
        return await self._repository.get(user_id)

    async def put_profile(self, profile: TravelerProfile) -> TravelerProfile:
        """整体覆盖结构化画像。

        Raises:
            RuntimeError: 本实例没有结构化画像能力。
        """
        if self._repository is None:
            raise RuntimeError("本实例没有配置结构化画像仓储。")
        return await self._repository.upsert(profile)

    # ------------------------------------------------------------------
    # 诊断
    # ------------------------------------------------------------------
    def describe(self) -> dict[str, object]:
        """给 ``/readyz`` 与启动日志用的**纯配置**描述（零 I/O）。

        ⚠️ 不含任何用户数据，只报「这个进程具备哪些能力」。
        与 ``describe_vector_store`` 同一取舍：它会在 Milvus 挂着时
        被调用，所以它自己绝不能碰 Milvus。

        Returns:
            `dict[str, object]`: 开关与能力描述。
        """
        return {
            "enabled": self._settings.memory.enabled,
            "structured": self.has_repository,
            "semantic": self.has_semantic,
            "reme_enabled": self._settings.memory.reme_enabled,
            "top_k": self._settings.memory.top_k,
        }


__all__ = ["MemoryContext", "TravelerMemory"]
