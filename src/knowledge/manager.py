# -*- coding: utf-8 -*-
"""单集合知识库管理器 —— 把「所有知识库放一个集合」这件事做对。

═══ ⚠️ 为什么不能用框架自带的 ``CollectionPerKbManager`` ═══

框架唯一的 KB 管理器是 :class:`~agentscope.app.rag.CollectionPerKbManager`，
**每个知识库一个集合**，集合名 ``kb_<uuid>``（``agentscope/app/rag/knowledge_base_manager/_collection_per_kb.py:89``）。
本项目用的是**单集合**策略：契约写死了
``ALIGO__MILVUS__COLLECTION=aligo_travel_policy_dev``，所有知识库共用它，
靠 ``metadata_filter`` 做 KB / 租户隔离。

把 ``ALIGO__MILVUS__COLLECTION`` 换成 per-KB 是**不行的** —— 它是契约里
逐字指定的值，运维、初始化脚本、Grafana 面板都指着它。

═══ ⚠️⚠️ 这里有一个会造成数据丢失的陷阱 ═══

单集合策略下，``delete_knowledge_base`` **绝不能**沿用框架的写法。
``CollectionPerKbManager.delete_knowledge_base`` 的实现是
``has_collection`` → ``delete_collection``（``:139-144``）——
在 per-KB 策略下那是对的（那个集合只属于这一个 KB），
在单集合策略下那是**把所有人的数据一起删掉**。

所以本类的 ``delete_knowledge_base`` 是**按 metadata 限定范围**逐个文档删，
并且**永不**调用 ``delete_collection``。这条差异必须留在代码里，
不能靠「继承后记得覆盖」这种口头约定 —— 见本类的测试。

═══ 隔离靠什么 ═══

``KnowledgeBase`` 的 ``metadata_filter`` 是**双向**的（``rag/_knowledge.py``）：

    · 检索与列举时，结果被限定在 filter 命中的记录里；
    · **写入时**，filter 里的键被强制写给每一个 chunk，覆盖调用方给的值。

第二条是关键：即使某个解析器有 bug、或某个工具传了别家的 ``kb_id``，
记录也**进不了**别人的范围。这是多租户的纵深防御，不是可选项。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from agentscope.app.access import ResourceKind
from agentscope.app.rag.knowledge_base_manager import (
    DimensionPolicy,
    DimensionPolicyKind,
    DimensionPolicyError,
    KnowledgeBaseManagerBase,
    KnowledgeBaseNotFoundError,
)
from agentscope.app.storage import (
    CredentialRecord,
    KnowledgeBaseData,
    KnowledgeBaseRecord,
)
from agentscope.rag import KnowledgeBase

from src.config.schema import Settings
from src.web_embedding.bounded import bound_embedding_model

from .store import ensure_collection

if TYPE_CHECKING:
    from agentscope.app.storage import (
        ChunkerConfig,
        EmbeddingModelConfig,
        StorageBase,
    )
    from agentscope.rag import VectorStoreBase

logger = logging.getLogger(__name__)


#: 写进每个 chunk metadata 的租户键。
#:
#: ⚠️ 名字里带前缀（``aligo_``）不是洁癖：这个字段和框架自己的字段
#: 共用同一个 JSON 列，将来框架新增一个同名键时会**静默冲突** ——
#: 我们的 filter 会开始命中（或漏掉）不该命中的记录，
#: 而那种 bug 表现为「检索结果偶尔不对」，极难归因。
KB_ID_KEY = "aligo_kb_id"
USER_ID_KEY = "aligo_user_id"


class SingleCollectionKbManager(KnowledgeBaseManagerBase):
    """把**所有**知识库放进同一个集合，用 metadata 做隔离。

    Attributes:
        _settings (`Settings`): 配置（取集合名与维度）。
        _access_policy_provider (`Callable[[], Any] | None`): 延迟取资源访问
            策略的函数，用于解析**跨属主共享**的 embedding 凭据
            （见 :meth:`_resolve_embedding_credential`）。
    """

    def __init__(
        self,
        storage: "StorageBase",
        vector_store: "VectorStoreBase",
        settings: Settings,
        *,
        access_policy_provider: "Callable[[], Any] | None" = None,
    ) -> None:
        """初始化。

        Args:
            storage (`StorageBase`): 应用级存储，用于持久化 KB 记录与凭据。
            vector_store (`VectorStoreBase`): 应用级向量库（全体 KB 共用）。
            settings (`Settings`): 配置，提供集合名与维度。
            access_policy_provider (`Callable[[], Any] | None`): 返回
                ``ResourceAccessPolicyBase`` 的**取值函数**（不是策略对象本身）。

                ⚠️ 为什么是函数而不是对象：策略由框架的 ``create_app`` 在
                装配期写进 ``app.state.resource_access_policy``，而本管理器
                在它**之前**就构造好了（``build_knowledge_manager`` 的结果
                要当参数传给它）。直接传对象就意味着「构造顺序决定谁能看见
                什么」，而这正是最难排查的一类 bug。传一个延迟求值的取值函数，
                装配顺序就不再是契约的一部分。

                ``None``（默认）表示没有接上策略 —— 单测、以及
                ``scripts/seed_data.py`` 这类离线脚本就是这种情况，
                此时行为与框架自带管理器**逐字一致**（只按属主查）。
        """
        super().__init__(storage=storage, vector_store=vector_store)
        self._settings = settings
        self._access_policy_provider = access_policy_provider
        #: 单次向量化的截止时间（秒）。在这里取一次而不是每次构造 KB 时读配置：
        #: 配置是只读的，重复读取只会让「到底用的哪个值」多出几个可能。
        self._embedding_timeout = settings.embedding.timeout_seconds

    @property
    def collection(self) -> str:
        """契约集合名。

        Returns:
            `str`: ``settings.milvus.collection``。
        """
        return self._settings.milvus.collection

    # ------------------------------------------------------------------
    # 能力声明
    # ------------------------------------------------------------------
    async def get_dimension_policy(self) -> DimensionPolicy:
        """报出 **FIXED** 策略：维度由服务端钉死。

        ⚠️ 这里必须是 ``FIXED`` 而不是框架 per-KB 策略的 ``ANY``。
        单集合策略下，集合的维度在建集合那一刻就定死了，之后每一个
        KB 都必须用同一个维度 —— 让用户「自由选择维度」是**假的自由**：
        他选了 768，写入时才会炸，而那时集合已经按 1024 建好了。

        ``FIXED`` 会让前端**提前**把不兼容的模型/维度筛掉
        （``DimensionPolicy.filter_card``），把错误挡在提交之前。

        Returns:
            `DimensionPolicy`: ``(FIXED, settings.milvus.dimension)``。
        """
        return DimensionPolicy(
            kind=DimensionPolicyKind.FIXED,
            dimension=self._settings.milvus.dimension,
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------
    async def create_knowledge_base(
        self,
        user_id: str,
        name: str,
        description: str,
        embedding_model_config: "EmbeddingModelConfig",
        chunker_config: "ChunkerConfig | None" = None,
    ) -> KnowledgeBaseRecord:
        """建知识库：核验维度 → 确保集合存在 → 落记录。

        ⚠️ 顺序是有讲究的，**不能**先把记录落库再建集合：
        那样一旦集合创建失败，storage 里就留下一条指向不存在集合的记录，
        而后续每一次检索都会以一个「看起来配好了」的 KB 的身份失败。

        ⚠️ 与框架的 per-KB 实现不同，这里**没有**「失败时补偿删除集合」
        那一段。因为那个补偿在单集合策略下会把**所有** KB 的数据删掉 ——
        一个失败的创建请求不该有能力清空整个知识库。这里的补偿是
        「什么都不做」：集合留着是对的（它是共享的），记录没落库即可。

        Args:
            user_id (`str`): 属主用户 id。
            name (`str`): 展示名。
            description (`str`): 描述。
            embedding_model_config (`EmbeddingModelConfig`): 向量模型配置。
            chunker_config (`ChunkerConfig | None`, optional): 分块配置。

        Returns:
            `KnowledgeBaseRecord`: 落库后的记录。

        Raises:
            DimensionPolicyError: 维度不等于 ``settings.milvus.dimension``。
        """
        requested = embedding_model_config.dimensions
        policy = await self.get_dimension_policy()
        if not policy.accepts(requested):
            # ⚠️ 用框架的 DimensionPolicyError（而不是 HTTPException）：
            # 路由层会把它映射成 409。管理器不该知道 HTTP 的存在 ——
            # 它还要被 CLI 脚本与离线评测复用。
            raise DimensionPolicyError(
                f"维度 {requested} 与本项目的单集合策略不符："
                f"共享集合 {self.collection!r} 建在 "
                f"{self._settings.milvus.dimension} 维上，"
                f"所有知识库必须一致。\n"
                f"修法：改用 {self._settings.milvus.dimension} 维的向量模型，"
                f"或改 ALIGO__MILVUS__DIMENSION（需重建集合并重灌数据）。",
                requested_dimension=requested,
                policy_dimension=policy.dimension,
            )

        # ⚠️ 幂等：第一次建 KB 时真的建集合，之后每次都是 no-op。
        # 放在这里而不是启动期，是因为「库里有 KB 记录」与
        # 「集合已建好」这两件事必须一致 —— 让创建路径同时负责两者，
        # 就没有「记录在而集合不在」的窗口。
        await ensure_collection(self._vector_store, self._settings)

        record = KnowledgeBaseRecord(
            user_id=user_id,
            data=KnowledgeBaseData(
                name=name,
                description=description,
                embedding_model_config=embedding_model_config,
                chunker_config=chunker_config,
                # ⚠️ 不是 ``f"kb_{record.id}"``（框架那样）。所有记录
                # 指向**同一个**集合 —— 这是本策略的定义。
                collection_name=self.collection,
            ),
        )
        return await self._storage.upsert_knowledge_base(user_id, record)

    async def delete_knowledge_base(
        self,
        user_id: str,
        knowledge_base_id: str,
    ) -> bool:
        """删知识库：**只删这个 KB 的向量**，集合本身保留。

        ⚠️⚠️ 这是与框架 per-KB 实现最要命的一处差异。框架的实现是
        ``delete_collection(record.data.collection_name)``
        （``agentscope/app/rag/knowledge_base_manager/_collection_per_kb.py:142-144``）—— 在单集合策略下，
        ``collection_name`` 对**每一个** KB 都是同一个值，
        于是「删掉一个知识库」= 「删掉所有人的知识库」。

        所以这里走的是「列出本 KB 范围内的文档 → 逐个删」。
        代价是 N 次往返（N = 该 KB 的文档数），但那是正确的代价：
        少一次往返换来清空全库，不是一笔值得做的交易。

        Args:
            user_id (`str`): 属主用户 id。
            knowledge_base_id (`str`): 要删的知识库 id。

        Returns:
            `bool`: 记录存在并被删除返回 True；不存在返回 False。
        """
        record = await self._storage.get_knowledge_base(
            user_id,
            knowledge_base_id,
        )
        if record is None:
            # ⚠️ 直接返回 False，不做任何清理 —— 分不清「不存在」与
            # 「不属于你」是刻意的（框架的 KnowledgeBaseNotFoundError
            # 文档里写明了同一条理由：不要泄露他人知识库的存在性）。
            return False

        await self._delete_scope(record)

        return await self._storage.delete_knowledge_base(
            user_id,
            knowledge_base_id,
        )

    async def _delete_scope(self, record: KnowledgeBaseRecord) -> int:
        """删掉 ``record`` 这个 KB 范围内的全部向量。

        ⚠️ 先列举再逐个删，而不是「按 filter 批量删」——
        ``VectorStoreBase`` 的接口里**没有**按 filter 批量删这一项
        （``delete(collection, document_id)`` 只按单个 document_id 删）。
        绕开接口直接调 pymilvus 的 ``delete(expr=...)`` 是能更快，
        但那就把 backend 细节漏进了管理器，而管理器要能被换成
        Qdrant / Elasticsearch —— 见 ``VectorStoreBase`` 的模块文档。

        Args:
            record (`KnowledgeBaseRecord`): 目标知识库记录。

        Returns:
            `int`: 实际删除的文档数（用于日志与「删了没」的判断）。
        """
        summaries = await self._vector_store.list_documents(
            self.collection,
            metadata_filter=self._scope(record),
        )
        for summary in summaries:
            await self._vector_store.delete(
                self.collection,
                summary.document_id,
            )

        if summaries:
            logger.info(
                "知识库 %s 已删除 %d 篇文档（集合 %s 未受影响）。",
                record.id,
                len(summaries),
                self.collection,
            )
        return len(summaries)

    # ------------------------------------------------------------------
    # 凭据解析
    # ------------------------------------------------------------------
    async def _resolve_embedding_credential(
        self,
        owner_id: str,
        credential_id: str,
    ) -> CredentialRecord | None:
        """解析 KB 绑定的 embedding 凭据：先按属主，**再回落到访问策略**。

        ═══ 为什么必须有第二跳 ═══

        框架的实现只查属主（``agentscope/app/rag/knowledge_base_manager/_collection_per_kb.py:187-195``），并在那里
        写下了它所依赖的前提：「KB 管理器是属主内部路径，凭据与知识库
        同属一个属主」。这个前提在本项目里**不再成立**：

            ``GET /knowledge_bases/embedding_models`` 明确地把**共享凭据**
            也列进可选列表（它的 docstring 原文：「own + shared … so KB
            creation works against shared credentials too」），而
            ``GET /credential/`` 里那条 ``aligo-system-model`` 同样在列
            （见 ``src/llm/system_credential.py``）。

        用户据此建出来的 KB，记录里钉的是一个**别人的** credential_id。
        到 ``get_knowledge`` 时按属主一查 —— 查不到 ⇒
        ``KnowledgeBaseNotFoundError`` ⇒ 对话链路把它记成
        「Skipping knowledge base」（``agentscope/app/_service/_chat.py:1048-1059``，
        异常被吞、只留一条日志）⇒ **知识库静默失效**，而上传、索引、
        前端列表全都显示正常。这正是本项目最想消灭的那类故障：
        探针全绿、界面正常、功能不在。

        所以第二跳补的是框架那两处假设之间的缝：属主查不到时，
        去问**资源访问策略**「这个属主能不能用这条凭据」——
        与框架运行期解析模型凭据走的是同一条规则
        （``app/_service/_access.py::resolve_credential``，那里同样
        「先按属主、再按可访问的 refs」）。**能不能用**只有一个权威，
        我们不在这里另立一套。

        ═══ 为什么这不会放宽权限 ═══

        ``get_knowledge`` 的 ``owner_id`` **始终是知识库记录的属主**
        （框架在两处调用点都做过这个改写：``_service/_knowledge_base.py``
        的 ``_resolve_knowledge`` 与 ``_service/_chat.py``），
        而下面查的是「**该属主**可访问的凭据」—— 调用者无法通过这个函数
        获得它本来就拿不到的东西：策略不给它，这里就照样解析不出来。

        Args:
            owner_id (`str`): 知识库记录的属主。
            credential_id (`str`): 记录里钉的凭据 id。

        Returns:
            `CredentialRecord | None`: 解析到的原始记录（**未打码** ——
            它只用于构造 embedding 模型，绝不进 HTTP 响应）；解析不到时
            ``None``，由调用方翻译成 ``KnowledgeBaseNotFoundError``。

        Raises:
            Exception: 策略抛出的异常**原样向上抛**，与框架的
                ``resolve_credential`` 一致。吞掉它只会让策略的 bug
                变成「凭据不存在」这一个假答案，而真正的故障点
                连一行日志都不会留下。
        """
        record = await self._storage.get_credential(owner_id, credential_id)
        if record is not None:
            return record

        provider = self._access_policy_provider
        if provider is None:
            # 没接上策略：行为与框架自带管理器逐字一致（只认属主）。
            return None
        policy = provider()
        if policy is None:
            # 框架回落到了默认的 deny-all —— 「没有跨属主可访问的东西」。
            return None

        refs = await policy.list_accessible(
            owner_id,
            ResourceKind.CREDENTIAL,
            self._storage,
        )
        for ref in refs:
            # ⚠️ ``ref.kind`` 也要核。``list_accessible`` 的 ``kind`` 入参是
            # **请求**，不是**承诺** —— 策略返回什么由它的实现决定，一个
            # 宽松（或写错）的策略完全可能把别的资源类型也一并倒出来。
            # 框架自己就在 ``app/_service/_access.py::_list_refs`` 里逐条
            # 过滤 ``ref.kind == kind``，我们照做：id 空间是各种资源共用
            # 一张表的，若不看 kind，一条 id 恰好等于某凭据 id 的
            # **别的类型**的共享记录，就能把那条凭据顶到解析结果里。
            if ref.kind != ResourceKind.CREDENTIAL:
                continue
            if ref.resource_id != credential_id:
                continue
            # ⚠️ 用 ref 里的 owner_id 去取，**不是**用 owner_id：
            # 跨属主引用的价值就在于记录在别人名下。
            return await self._storage.get_credential(
                ref.owner_id,
                ref.resource_id,
            )
        return None

    # ------------------------------------------------------------------
    # 运行时句柄
    # ------------------------------------------------------------------
    async def get_knowledge(
        self,
        user_id: str,
        knowledge_base_id: str,
    ) -> KnowledgeBase:
        """解析出一个绑定了**本 KB 作用域**的运行时句柄。

        ⚠️ ``metadata_filter`` 不是「优化」，是**隔离**。见模块文档的
        「隔离靠什么」：它在写入方向也会强制生效。

        Args:
            user_id (`str`): 属主用户 id。
            knowledge_base_id (`str`): 知识库 id。

        Returns:
            `KnowledgeBase`: 绑定了集合与作用域过滤的运行时句柄。

        Raises:
            KnowledgeBaseNotFoundError: 记录不存在或不属于该用户；
                或记录指向的凭据**解析不出来** —— 既不属于该属主，
                也不在访问策略授予它的范围内（见
                :meth:`_resolve_embedding_credential`）。
        """
        record = await self._storage.get_knowledge_base(
            user_id,
            knowledge_base_id,
        )
        if record is None:
            raise KnowledgeBaseNotFoundError(
                f"知识库 {knowledge_base_id!r} 不存在。",
            )

        credential_id = record.data.embedding_model_config.credential_id
        credential_record = await self._resolve_embedding_credential(
            record.user_id,
            credential_id,
        )
        if credential_record is None:
            raise KnowledgeBaseNotFoundError(
                f"知识库 {knowledge_base_id!r} 绑定的凭据 "
                f"{credential_id!r} 不存在。",
            )

        # ⚠️ 复用框架的构造函数，不自己拼 —— 维度校验、参数字典化、
        # context_size 查找都在里面（``_service/_embedding.py``）。
        from agentscope.app._service._embedding import build_embedding_model

        embedding_model = build_embedding_model(
            credential_record=credential_record,
            config=record.data.embedding_model_config,
        )
        # ⚠️ 这里构造出来的模型**不是**来自 ``src/web_embedding`` 的降级链
        # （走框架的 ``CredentialFactory``，见 :mod:`src.knowledge.rag` 的说明），
        # 所以它不会自动带上截止时间 —— 必须在这里显式套。
        # 否则这条路上「向量模型卡住 ⇒ 检索假死」的洞仍然开着，
        # 而 ``GuardedVectorStore`` 保护的那次 ``search`` 调用压根还没开始。
        # 超时值取共享配置（``embedding.timeout_seconds``）：无论模型是谁造的，
        # 「一次向量化最多等多久」都该是同一个数。
        embedding_model = bound_embedding_model(
            embedding_model,
            timeout=self._embedding_timeout,
        )

        return KnowledgeBase(
            name=record.data.name,
            description=record.data.description,
            embedding_model=embedding_model,
            vector_store=self._vector_store,
            collection=self.collection,
            metadata_filter=self._scope(record),
        )

    def _scope(self, record: KnowledgeBaseRecord) -> dict[str, str]:
        """算出某个知识库的 metadata 作用域。

        ⚠️ 同时带 ``kb_id`` 与 ``user_id`` 是**冗余**的 —— ``record.id``
        已经全局唯一，单靠它就能定位。之所以两个都放，是因为
        ``metadata_filter`` 的价值在于「纵深防御」：它要能挡住
        **上层逻辑写错**的情况。假如哪天有代码把另一个用户的 kb_id
        传了进来，只有 kb_id 的 filter 会照常放行；带上 user_id 之后，
        这条越权读取会在存储层就被挡住。

        ⚠️ 两个键都必须在**写入**方向也生效（框架保证），
        否则这条防御是单向的、形同虚设。

        Args:
            record (`KnowledgeBaseRecord`): 目标知识库记录。

        Returns:
            `dict[str, str]`: 两个扁平键值对。
        """
        return {
            KB_ID_KEY: record.id,
            USER_ID_KEY: record.user_id,
        }


__all__ = ["KB_ID_KEY", "USER_ID_KEY", "SingleCollectionKbManager"]
