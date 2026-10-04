# -*- coding: utf-8 -*-
"""单集合知识库管理器（``src/knowledge/manager.py``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    单集合策略把「每个知识库一个集合」换成了「所有知识库共用一个集合」。
    这个替换本身很简单，但它悄悄改变了**每一个 CRUD 操作的含义** ——
    而其中有一个改动错了会**丢数据**：

      ``delete_knowledge_base``

    在 per-KB 策略下它是 ``delete_collection(kb_xxx)``，删的只是那一个 KB。
    在单集合策略下，``collection_name`` 对每个 KB 都是同一个值，
    同一行代码会变成「删掉所有人的知识库」—— 不报错、不回滚、不可逆。

    所以本文件里最重要的用例是
    :func:`test_deleting_one_kb_never_drops_the_shared_collection`，
    它断言的是**没有发生**某件事（``delete_collection`` 未被调用）。
    这类「断言一个调用没发生」的用例平时显得多余，但在这里它是
    唯一能挡住数据丢失的东西 —— 因为那个 bug 不会让任何断言变红，
    只会让某个用户的政策库在某天突然空了。

    其余三组分别覆盖：维度策略（FIXED 而非 ANY）、作用域隔离
    （filter 同时约束读与写）、以及记录/集合的一致性。
"""

from __future__ import annotations

import pytest

from agentscope.app.access import (
    ResourceKind,
    ResourcePermission,
    ResourceRef,
)
from agentscope.app.rag.knowledge_base_manager import (
    DimensionPolicyKind,
    DimensionPolicyError,
    KnowledgeBaseNotFoundError,
)
from agentscope.app.storage import (
    CredentialRecord,
    EmbeddingModelConfig,
    KnowledgeBaseRecord,
)
from agentscope.credential import DashScopeCredential
from agentscope.message import TextBlock
from agentscope.rag import Chunk, DocumentSummary, VectorSearchResult
from src.config import Settings
from src.knowledge.manager import (
    KB_ID_KEY,
    USER_ID_KEY,
    SingleCollectionKbManager,
)


# ==============================================================================
# 测试替身
# ==============================================================================
class FakeVectorStore:
    """内存版向量库替身。

    ⚠️ 自己写一个而不是用框架的 ``MilvusLiteStore``，是因为这些用例要断言的
    是**管理器调了什么、没调什么**，而不是 Milvus 的行为。用真 Milvus 的话
    用例会依赖 Docker，而最关键的那条断言（「没有调用 delete_collection」）
    反而更难看清。

    ⚠️ ``delete_collection`` **必须**记录调用并抛错（而不是默默什么都不做）：
    它是本文件要挡的那个 bug 的现场。一个「删了就删了」的替身会让
    :func:`test_deleting_one_kb_never_drops_the_shared_collection` 变成空断言。
    """

    def __init__(self) -> None:
        self.collections: dict[str, int] = {}
        self.rows: list[dict] = []
        self.delete_collection_calls: list[str] = []
        self.create_collection_calls: list[tuple[str, int]] = []

    async def create_collection(self, name: str, dimensions: int) -> None:
        """幂等地建集合（与 ``MilvusLiteStore`` 同语义）。"""
        self.create_collection_calls.append((name, dimensions))
        self.collections.setdefault(name, dimensions)

    async def delete_collection(self, name: str) -> None:
        """记录并真的删掉 —— 这正是单集合策略下**不该被走到**的那条路。"""
        self.delete_collection_calls.append(name)
        self.collections.pop(name, None)

    async def has_collection(self, name: str) -> bool:
        """集合是否存在。"""
        return name in self.collections

    async def insert(self, collection: str, records: list) -> None:
        """写入。真实实现会强制附带 filter 的键，替身保持同样的语义。"""
        for record in records:
            meta = dict(record.chunk.metadata or {})
            self.rows.append(
                {
                    "collection": collection,
                    "document_id": record.document_id,
                    "chunk_index": record.chunk.chunk_index,
                    "metadata": meta,
                },
            )

    async def delete(self, collection: str, document_id: str) -> None:
        """按 document_id 删（接口如此，见 ``VectorStoreBase.delete``）。"""
        self.rows = [
            row
            for row in self.rows
            if not (
                row["collection"] == collection
                and row["document_id"] == document_id
            )
        ]

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list[VectorSearchResult]:
        """不做真的相似度计算 —— 这些用例不测检索质量。"""
        del query_vector, top_k
        return [
            VectorSearchResult(
                score=1.0,
                document_id=row["document_id"],
                chunk=Chunk(
                    content="替身内容",
                    source="fake",
                    chunk_index=row["chunk_index"],
                    total_chunks=1,
                    metadata=row["metadata"],
                ),
            )
            for row in self._match(collection, metadata_filter)
        ]

    async def list_documents(
        self,
        collection: str,
        metadata_filter: dict | None = None,
    ) -> list[DocumentSummary]:
        """按 filter 列出文档。"""
        seen: dict[str, int] = {}
        for row in self._match(collection, metadata_filter):
            seen[row["document_id"]] = seen.get(row["document_id"], 0) + 1
        return [
            DocumentSummary(
                document_id=doc_id,
                source="fake",
                chunk_count=count,
                metadata={},
            )
            for doc_id, count in seen.items()
        ]

    def _match(self, collection: str, metadata_filter: dict | None) -> list[dict]:
        """按**扁平 key == value** 过滤 —— 与 Milvus 后端的语义一致。

        ⚠️ 只支持相等，不支持 ``$gt`` / ``$in`` / ``$or`` ——
        ``_milvus_lite.py:504-515`` 就是这么实现的。替身跟着一起窄，
        是为了让「用了一个 Milvus 不支持的 filter」在单测里就暴露，
        而不是等到连上真库才报错。
        """
        rows = [r for r in self.rows if r["collection"] == collection]
        if not metadata_filter:
            return rows
        return [
            r
            for r in rows
            if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
        ]


class FakeStorage:
    """内存版存储替身，只实现管理器用到的四个方法。"""

    def __init__(self) -> None:
        self.kbs: dict[tuple[str, str], KnowledgeBaseRecord] = {}
        self.credentials: dict[tuple[str, str], object] = {}
        self.upsert_calls = 0

    async def upsert_knowledge_base(self, user_id: str, record):
        """落库。"""
        self.upsert_calls += 1
        self.kbs[(user_id, record.id)] = record
        return record

    async def get_knowledge_base(self, user_id: str, knowledge_base_id: str):
        """按属主查 —— 不是属主就是 None（不泄露存在性）。"""
        record = self.kbs.get((user_id, knowledge_base_id))
        return record if record is not None else None

    async def delete_knowledge_base(self, user_id: str, knowledge_base_id: str):
        """删记录。"""
        return self.kbs.pop((user_id, knowledge_base_id), None) is not None

    async def get_credential(self, user_id: str, credential_id: str):
        """查凭据。"""
        return self.credentials.get((user_id, credential_id))


def _config(dimensions: int = 1024) -> EmbeddingModelConfig:
    """构造一份向量模型配置。"""
    return EmbeddingModelConfig(
        type="dashscope_credential",
        credential_id="cred-1",
        model="text-embedding-v4",
        dimensions=dimensions,
    )


def _credential(user_id: str = "u1") -> CredentialRecord:
    """构造一条**真的**凭据记录。

    ⚠️ 用真实的 :class:`~agentscope.credential.DashScopeCredential` 而不是
    一个 `Mock` 对象，是因为 ``build_embedding_model`` 会走
    ``CredentialFactory.from_dict`` + ``get_embedding_model_class()``
    这两步**真实的工厂查表**。一个假凭据会让这两步被绕过，
    于是用例测的是「我的替身能跑」，而不是「框架的装配路径能跑」。

    ⚠️ ``data`` 存的是 **dict**（``CredentialRecord.data`` 在框架里就声明成
    字典），而 ``model_dump()`` 会带上 ``type`` 字段 —— 那正是
    ``CredentialFactory.from_dict`` 用来查表的东西。存模型对象本身
    会在构造 ``CredentialRecord`` 时就被 pydantic 拒掉。

    ⚠️ 里面的 api_key 是假的（``sk-test``），且**永远**用不到 ——
    ``DashScopeEmbeddingModel`` 的构造不会发请求（HTTP 客户端是惰性的）。
    """
    return CredentialRecord(
        user_id=user_id,
        data=DashScopeCredential(
            api_key="sk-test-not-a-real-key",
        ).model_dump(mode="json"),
    )


@pytest.fixture
def store() -> FakeVectorStore:
    """向量库替身。"""
    return FakeVectorStore()


@pytest.fixture
def storage() -> FakeStorage:
    """存储替身。"""
    return FakeStorage()


@pytest.fixture
def manager(settings: Settings, storage: FakeStorage, store: FakeVectorStore):
    """被测的管理器。"""
    return SingleCollectionKbManager(
        storage=storage,
        vector_store=store,
        settings=settings,
    )


async def _make_kb(manager, user_id: str = "u1", name: str = "政策库"):
    """建一个知识库，返回记录。"""
    return await manager.create_knowledge_base(
        user_id=user_id,
        name=name,
        description="差旅政策",
        embedding_model_config=_config(),
    )


# ==============================================================================
# 一、⚠️⚠️ 删一个知识库绝不能删掉共享集合
# ==============================================================================
async def test_deleting_one_kb_never_drops_the_shared_collection(
    manager: SingleCollectionKbManager,
    storage: FakeStorage,
    store: FakeVectorStore,
) -> None:
    """★★★ 删掉一个 KB 时，**没有**发生 ``delete_collection``。

    ⚠️ 这是整个 P4 里最重要的一条断言，因为它挡的是一个**不可逆的数据丢失**。

    框架的 ``CollectionPerKbManager.delete_knowledge_base`` 走的是
    ``has_collection`` → ``delete_collection``（``_collection_per_kb.py:139-144``）。
    在 per-KB 策略下那是对的：那个集合只属于这一个 KB。
    但在本项目的单集合策略下，``record.data.collection_name``
    对**每一个** KB 都是 ``settings.milvus.collection`` ——
    同一行代码会删掉**所有人**的知识库。

    而它不会报错、不会回滚、也没有任何断言会因此变红（除非像本用例这样
    显式盯着「有没有调用 delete_collection」）。

    三个知识库的场景是刻意的：只有存在**别的** KB 时，
    「删掉共享集合」与「删掉这一个 KB」才区分得开。
    """
    a = await _make_kb(manager, "u1", "政策库A")
    b = await _make_kb(manager, "u2", "政策库B")
    c = await _make_kb(manager, "u3", "政策库C")

    assert store.collections[manager.collection] == 1024

    deleted = await manager.delete_knowledge_base("u1", a.id)

    assert deleted is True
    assert store.delete_collection_calls == [], (
        f"删除知识库时调用了 delete_collection({store.delete_collection_calls})！\n"
        f"⚠️ 单集合策略下这是**把所有人的知识库一起删掉**。\n"
        f"框架的 per-KB 实现确实这么写，但本项目不能沿用 —— "
        f"详见 src/knowledge/manager.py 的模块文档。"
    )
    # 集合还在，另外两个 KB 的记录也还在。
    assert manager.collection in store.collections
    assert await storage.get_knowledge_base("u2", b.id) is not None
    assert await storage.get_knowledge_base("u3", c.id) is not None


async def test_deleting_a_kb_only_removes_that_kbs_vectors(
    manager: SingleCollectionKbManager,
    storage: FakeStorage,
    store: FakeVectorStore,
) -> None:
    """删一个 KB 只清掉**它自己**的向量，别人的一行不动。

    ⚠️ 这条与上一条是两个不同的失败方向：
    上一条挡「删太多」（删整个集合），这条挡「删错对象」
    （把 filter 写成了不含 kb_id 的形式，于是删掉全部文档、
    但集合还在 —— 症状更隐蔽，因为集合还在，看起来一切正常）。
    """
    a = await _make_kb(manager, "u1", "政策库A")
    b = await _make_kb(manager, "u2", "政策库B")

    await store.insert(
        manager.collection,
        [
            _record(a, "doc-a-1", {KB_ID_KEY: a.id, USER_ID_KEY: "u1"}),
            _record(a, "doc-a-2", {KB_ID_KEY: a.id, USER_ID_KEY: "u1"}),
            _record(b, "doc-b-1", {KB_ID_KEY: b.id, USER_ID_KEY: "u2"}),
        ],
    )

    await manager.delete_knowledge_base("u1", a.id)

    remaining = {row["document_id"] for row in store.rows}
    assert remaining == {"doc-b-1"}, (
        f"删完 A 之后剩下的文档是 {remaining}，应当只有 B 的 doc-b-1。"
    )
    assert await storage.get_knowledge_base("u2", b.id) is not None


async def test_deleting_a_non_existent_kb_is_a_silent_no_op(
    manager: SingleCollectionKbManager,
    store: FakeVectorStore,
) -> None:
    """删不存在的 KB 返回 False，且**不碰**任何数据。

    ⚠️ 「不存在」与「不属于你」返回同一个结果（都是 False）——
    这是刻意的，框架的 ``KnowledgeBaseNotFoundError`` 文档里写明了同一条理由：
    不要泄露他人知识库的存在性。
    """
    await _make_kb(manager, "u1", "政策库A")

    assert await manager.delete_knowledge_base("u1", "不存在的id") is False
    assert await manager.delete_knowledge_base("别人", "随便") is False
    assert store.delete_collection_calls == []


def _record(kb: KnowledgeBaseRecord, document_id: str, metadata: dict):
    """构造一条待写入的向量记录（只为驱动替身）。

    ⚠️ ``Chunk.content`` 的类型是 ``TextBlock | DataBlock``，**不是** ``str``
    —— 传裸字符串会让 pydantic 在校验时报一个字段路径很长的错。
    这里顺手用一个极简的对象而不是 ``VectorRecord``，因为
    替身的 ``insert`` 只读 ``document_id`` 与 ``chunk`` 三个字段，
    用完整的 ``VectorRecord`` 还要先造一个向量，纯属噪音。
    """

    class _R:
        chunk = Chunk(
            content=TextBlock(type="text", text="内容"),
            source=document_id,
            chunk_index=0,
            total_chunks=1,
            metadata=metadata,
        )

    _R.document_id = document_id
    return _R()


# ==============================================================================
# 二、维度策略：FIXED，不是 ANY
# ==============================================================================
async def test_the_dimension_policy_is_fixed_to_the_configured_dimension(
    manager: SingleCollectionKbManager,
    settings: Settings,
) -> None:
    """维度策略必须是 ``FIXED`` 且等于 ``milvus.dimension``。

    ⚠️ 不能是 ``ANY``（框架 per-KB 策略报的就是 ANY）。单集合策略下集合维度
    在建集合那一刻定死，让用户「自由选维度」是假的自由：他选了 768，
    写入时才炸 —— 而那时集合已经按 1024 建好了，错误信息还指向集合配置
    （那里其实是对的）。
    """
    policy = await manager.get_dimension_policy()

    assert policy.kind is DimensionPolicyKind.FIXED
    assert policy.dimension == settings.milvus.dimension


async def test_creating_a_kb_with_a_mismatched_dimension_is_rejected(
    manager: SingleCollectionKbManager,
    settings: Settings,
) -> None:
    """维度不符时抛 ``DimensionPolicyError``，且**在建集合之前**。

    ⚠️ 「在建集合之前」是这条用例真正在测的东西：如果先建集合再校验，
    一次失败的创建会留下一个按错维度建好的集合 —— 而
    ``create_collection`` 是幂等的、**永远不会修正**它。
    于是这个错误会一直藏着，直到某次真的写入。

    判据：抛错之后，集合**没有**被创建。
    """
    with pytest.raises(DimensionPolicyError) as excinfo:
        await manager.create_knowledge_base(
            user_id="u1",
            name="错维度的库",
            description="",
            embedding_model_config=_config(dimensions=768),
        )

    assert excinfo.value.requested_dimension == 768
    assert excinfo.value.policy_dimension == settings.milvus.dimension
    # ⚠️ 错误信息要给出**修法**，不能只说「维度不对」。
    assert "768" in str(excinfo.value)
    assert str(settings.milvus.dimension) in str(excinfo.value)


async def test_a_rejected_create_leaves_no_collection_behind(
    manager: SingleCollectionKbManager,
    store: FakeVectorStore,
) -> None:
    """被拒的创建**不能**留下集合，更不能留下记录。

    ⚠️ 这条与上一条分开写，是因为「抛了错」和「没留下副作用」是两件事。
    一个先建集合再校验的实现会通过上一条、通不过这一条。
    """
    with pytest.raises(DimensionPolicyError):
        await manager.create_knowledge_base(
            user_id="u1",
            name="错维度的库",
            description="",
            embedding_model_config=_config(dimensions=512),
        )

    assert store.create_collection_calls == [], (
        f"被拒的创建建了集合：{store.create_collection_calls}"
    )
    assert store.collections == {}


async def test_creating_a_kb_creates_the_collection_exactly_once(
    manager: SingleCollectionKbManager,
    store: FakeVectorStore,
) -> None:
    """重复建 KB 时集合只被真正创建一次（幂等）。

    ⚠️ 断言的是「没炸」+「维度没变」。``MilvusLiteStore.create_collection``
    在集合已存在时是 no-op，所以这里用替身的 ``collections`` 字典
    （``setdefault``）来复现同样的语义。
    """
    await _make_kb(manager, "u1", "A")
    await _make_kb(manager, "u2", "B")

    assert store.collections == {manager.collection: 1024}
    # 两次都调了（幂等由后端保证），但维度始终是配置值。
    assert {dim for _, dim in store.create_collection_calls} == {1024}


# ==============================================================================
# 三、作用域：filter 必须同时约束读与写
# ==============================================================================
async def test_the_knowledge_handle_carries_a_two_key_scope(
    manager: SingleCollectionKbManager,
    storage: FakeStorage,
) -> None:
    """运行时句柄必须带上 ``kb_id`` 与 ``user_id`` 两个键。

    ⚠️ 为什么两个都要（``record.id`` 已经全局唯一了）：
    ``metadata_filter`` 的价值在于**纵深防御** —— 它要能挡住上层逻辑写错。
    假如哪天有代码把另一个用户的 kb_id 传了进来，只有 kb_id 的 filter
    会照常放行；带上 user_id 之后，这条越权读取会在存储层就被挡住。
    """
    kb_record = await _make_kb(manager, "u1", "政策库")
    storage.credentials[("u1", "cred-1")] = _credential("u1")

    handle = await manager.get_knowledge("u1", kb_record.id)

    assert handle.collection == manager.collection
    assert handle.metadata_filter == {
        KB_ID_KEY: kb_record.id,
        USER_ID_KEY: "u1",
    }


async def test_getting_another_users_knowledge_raises_not_found(
    manager: SingleCollectionKbManager,
    storage: FakeStorage,
) -> None:
    """别人的知识库必须报「找不到」，而不是「无权限」。

    ⚠️ 两种说法看起来只差语气，实际是一条信息泄漏：
    「无权限」等于确认了「这个 id 存在，只是不属于你」。
    而 id 是 uuid，能猜中的概率为零 —— 所以真正的风险不是枚举，
    而是**日志与错误响应里泄露他人资源的存在性**。
    """
    kb_record = await _make_kb(manager, "u1", "政策库")
    storage.credentials[("u1", "cred-1")] = _credential("u1")

    with pytest.raises(KnowledgeBaseNotFoundError):
        await manager.get_knowledge("u2", kb_record.id)


async def test_a_kb_whose_credential_vanished_raises_not_found(
    manager: SingleCollectionKbManager,
    storage: FakeStorage,
) -> None:
    """记录在、凭据没了 ⇒ 报「找不到」，且错误信息指向**凭据**。

    ⚠️ 这个场景是真实会发生的：凭据被单独删掉时，KB 记录不会跟着走。
    如果这里不给出明确信息，症状会是 ``AttributeError: 'NoneType'``
    出现在 ``build_embedding_model`` 内部 —— 离真正的原因很远。
    """
    kb_record = await _make_kb(manager, "u1", "政策库")
    # 刻意不往 storage.credentials 里放凭据。

    with pytest.raises(KnowledgeBaseNotFoundError) as excinfo:
        await manager.get_knowledge("u1", kb_record.id)

    assert "cred-1" in str(excinfo.value), (
        f"错误信息里没有点出缺失的凭据 id：{excinfo.value}"
    )


# ==============================================================================
# 四、记录与集合的一致性
# ==============================================================================
async def test_every_record_points_at_the_one_contract_collection(
    manager: SingleCollectionKbManager,
    storage: FakeStorage,
    settings: Settings,
) -> None:
    """所有 KB 记录的 ``collection_name`` 都等于契约集合名。

    ⚠️ 框架的 per-KB 实现写的是 ``f"kb_{record.id}"``（``:89``）。
    照抄那一行的后果是「配置里的集合名从来没被用过」——
    而它是契约里逐字指定的值，初始化脚本、Grafana、运维手册都指着它。
    """
    a = await _make_kb(manager, "u1", "A")
    b = await _make_kb(manager, "u2", "B")

    assert a.data.collection_name == settings.milvus.collection
    assert b.data.collection_name == settings.milvus.collection
    assert a.id != b.id, "两个知识库的记录 id 必须不同（隔离靠它）"


async def test_the_record_is_persisted_after_the_collection_exists(
    manager: SingleCollectionKbManager,
    storage: FakeStorage,
    store: FakeVectorStore,
) -> None:
    """落库时集合已经建好了 —— 不存在「记录在而集合不在」的窗口。

    ⚠️ 顺序反过来的话，一次集合创建失败会留下一条指向不存在集合的记录，
    而之后每一次检索都会以一个「看起来配好了」的 KB 的身份失败。
    """
    calls_before = len(storage.kbs)
    assert calls_before == 0

    await _make_kb(manager, "u1", "A")

    assert manager.collection in store.collections
    assert len(storage.kbs) == 1


# ==============================================================================
# 七、跨属主共享的 embedding 凭据（「用得了」与「查得到」之间的那道缝）
# ==============================================================================
# 背景：框架里有两处互相矛盾的假设。
#
#   · GET /knowledge_bases/embedding_models 把**共享凭据**也列进可选列表
#     （它的 docstring 原文：「own + shared … so KB creation works against
#     shared credentials too」）；
#   · 而 CollectionPerKbManager.get_knowledge 只按**属主**查凭据，并在那里
#     写下前提：「凭据与知识库同属一个属主」。
#
#   本项目的系统凭据（src/llm/system_credential.py，属主 aligo-system）
#   让第二条假设不成立：用户据此建出来的 KB，检索时按属主一查就查不到 ⇒
#   KnowledgeBaseNotFoundError ⇒ 对话链路把异常吞成一行
#   「Skipping knowledge base」⇒ **知识库静默失效**，
#   而上传、索引、前端列表全都显示正常。
#
# 修复见 SingleCollectionKbManager._resolve_embedding_credential：
# 属主查不到时，回落到资源访问策略 —— 与框架解析模型凭据走同一条规则。


class _FakePolicy:
    """只实现 ``list_accessible`` 的资源访问策略替身。"""

    def __init__(self, refs: list[ResourceRef]) -> None:
        """记录要返回的 refs。

        Args:
            refs (`list[ResourceRef]`): 本策略「授予」的跨属主资源。
        """
        self._refs = refs
        #: ``(viewer_id, kind)`` 调用明细 —— 用来钉住「问的是谁」。
        self.calls: list[tuple[str, ResourceKind]] = []

    async def list_accessible(self, viewer_id: str, kind: ResourceKind, storage):
        """返回预先设定的 refs。"""
        self.calls.append((viewer_id, kind))
        return list(self._refs)


#: 共享凭据的属主与 id。字面量**刻意手抄**自 src/llm/system_credential.py：
#: 这一节要钉住的正是「属主不是 KB 的属主」这个形状，跟着常量漂移就失去意义。
SHARED_OWNER = "aligo-system"
SHARED_CREDENTIAL_ID = "aligo-system-model"


def _shared_config() -> EmbeddingModelConfig:
    """一份钉在**别人的**凭据上的向量模型配置。"""
    return EmbeddingModelConfig(
        type="dashscope_credential",
        credential_id=SHARED_CREDENTIAL_ID,
        model="text-embedding-v4",
        dimensions=1024,
    )


def _manager_with_policy(
    settings: Settings,
    storage: FakeStorage,
    store: FakeVectorStore,
    policy,
) -> SingleCollectionKbManager:
    """造一个接上了访问策略的管理器。"""
    return SingleCollectionKbManager(
        storage=storage,
        vector_store=store,
        settings=settings,
        access_policy_provider=lambda: policy,
    )


async def test_a_kb_bound_to_a_shared_credential_resolves_through_the_policy(
    settings: Settings,
    storage: FakeStorage,
    store: FakeVectorStore,
) -> None:
    """★ 用共享凭据建的 KB 必须能解析出运行时句柄。

    这是修复前**静默失效**的那条路径：KB 建得出来、文档传得上去、
    列表里看得见，只有检索永远是空的 —— 因为每次对话都在
    「跳过这个知识库」，而异常被吞掉、只留一行日志。
    """
    policy = _FakePolicy(
        [
            ResourceRef(
                kind=ResourceKind.CREDENTIAL,
                owner_id=SHARED_OWNER,
                resource_id=SHARED_CREDENTIAL_ID,
                permission=ResourcePermission.READ,
            ),
        ],
    )
    manager = _manager_with_policy(settings, storage, store, policy)

    kb = await manager.create_knowledge_base(
        user_id="u1",
        name="共享凭据政策库",
        description="",
        embedding_model_config=_shared_config(),
    )
    # 凭据在**别人**名下 —— 这就是全部问题所在。
    storage.credentials[(SHARED_OWNER, SHARED_CREDENTIAL_ID)] = _credential(
        SHARED_OWNER,
    )

    handle = await manager.get_knowledge("u1", kb.id)

    assert handle is not None
    # 策略被问的是「KB 属主能不能用这条凭据」，不是别的身份。
    assert policy.calls == [("u1", ResourceKind.CREDENTIAL)], policy.calls
    # 作用域过滤仍然是这一份 —— 解析凭据不该顺带放宽隔离。
    assert handle.metadata_filter[USER_ID_KEY] == "u1"
    assert handle.metadata_filter[KB_ID_KEY] == kb.id


async def test_a_credential_the_policy_does_not_grant_is_still_not_found(
    settings: Settings,
    storage: FakeStorage,
    store: FakeVectorStore,
) -> None:
    """★ 回落**不是**一把万能钥匙：策略不授予，就照样解析不出来。

    ⚠️ 这条用例是上面那条的护栏。回落逻辑若写成「查不到就去系统属主那里
    拿一条」（而不是「去问策略」），这个用例会失败 —— 而那意味着
    任何用户都能把别人的凭据拖进自己的 KB 使用，属于越权。
    """
    policy = _FakePolicy(
        [
            ResourceRef(
                kind=ResourceKind.CREDENTIAL,
                owner_id=SHARED_OWNER,
                resource_id="some-other-credential",  # ← 不是这一条
                permission=ResourcePermission.READ,
            ),
        ],
    )
    manager = _manager_with_policy(settings, storage, store, policy)

    kb = await manager.create_knowledge_base(
        user_id="u1",
        name="政策库",
        description="",
        embedding_model_config=_shared_config(),
    )
    storage.credentials[(SHARED_OWNER, SHARED_CREDENTIAL_ID)] = _credential(
        SHARED_OWNER,
    )

    with pytest.raises(KnowledgeBaseNotFoundError) as excinfo:
        await manager.get_knowledge("u1", kb.id)

    assert SHARED_CREDENTIAL_ID in str(excinfo.value)
    # ⚠️ 不仅要「没解析出来」，还要证明**问过策略了**：一个压根不问策略、
    # 直接返回 None 的实现同样能通过上面的断言，但它把「策略说不行」
    # 与「我们没看策略」变成了同一个结果 —— 而这两者对排查的含义完全相反。
    assert policy.calls == [("u1", ResourceKind.CREDENTIAL)], policy.calls


async def test_a_ref_of_the_wrong_kind_cannot_unlock_the_credential(
    settings: Settings,
    storage: FakeStorage,
    store: FakeVectorStore,
) -> None:
    """候选 ref 的 ``kind`` 不匹配时不得采用（纵深防御）。

    ⚠️ ``list_accessible(viewer_id, kind, storage)`` 的 ``kind`` 是**请求**，
    不是**承诺**：策略返回什么由它的实现决定。一个宽松或写错的策略完全
    可能把别的资源类型也一并倒出来，而 id 空间是各种资源共用一张表的 ——
    一条 id 恰好等于某凭据 id 的**别的类型**共享记录，若不核对 kind，
    就能把那条凭据顶进解析结果里。

    ⚠️ 框架自己就在 ``app/_service/_access.py::_list_refs`` 里逐条过滤
    ``ref.kind == kind``。这里布置的场景刻意做成「id 与属主都对得上、
    只有 kind 不对」：若过滤被删掉，凭据会被本当用，用例随即变红。
    """
    policy = _FakePolicy(
        [
            ResourceRef(
                kind=ResourceKind.KNOWLEDGE_BASE,  # ← 唯一的问题所在
                owner_id=SHARED_OWNER,
                resource_id=SHARED_CREDENTIAL_ID,  # ← id 与属主都是对的
                permission=ResourcePermission.READ,
            ),
        ],
    )
    manager = _manager_with_policy(settings, storage, store, policy)

    kb = await manager.create_knowledge_base(
        user_id="u1",
        name="政策库",
        description="",
        embedding_model_config=_shared_config(),
    )
    storage.credentials[(SHARED_OWNER, SHARED_CREDENTIAL_ID)] = _credential(
        SHARED_OWNER,
    )

    with pytest.raises(KnowledgeBaseNotFoundError):
        await manager.get_knowledge("u1", kb.id)


async def test_without_a_policy_the_behaviour_stays_owner_scoped(
    settings: Settings,
    storage: FakeStorage,
    store: FakeVectorStore,
) -> None:
    """没有接上策略时，行为与框架自带管理器**逐字一致**（只认属主）。

    ⚠️ 这是 ``access_policy_provider`` 默认 ``None`` 的语义，也是
    ``scripts/seed_data.py`` 等离线脚本所处的形态。写坏了的回落
    （例如无条件去查系统属主）会在离线脚本里表现为「凭据凭空出现」，
    而那种环境里没有策略、也就没有日志能解释它从哪来。
    """
    manager = _manager_with_policy(settings, storage, store, None)

    kb = await manager.create_knowledge_base(
        user_id="u1",
        name="政策库",
        description="",
        embedding_model_config=_shared_config(),
    )
    storage.credentials[(SHARED_OWNER, SHARED_CREDENTIAL_ID)] = _credential(
        SHARED_OWNER,
    )

    with pytest.raises(KnowledgeBaseNotFoundError):
        await manager.get_knowledge("u1", kb.id)


async def test_the_owner_scoped_lookup_does_not_touch_the_policy(
    settings: Settings,
    storage: FakeStorage,
    store: FakeVectorStore,
) -> None:
    """自己名下的凭据照旧直接命中 —— **不必**去问策略。

    这不是性能优化，是语义：策略只描述**跨属主**的可见性。
    每次取凭据都先问一遍策略，会让 `ListAgents` 之外的所有
    「策略坏了」都变成 KB 不可用，而策略本来跟这条路径无关。
    """
    policy = _FakePolicy([])
    manager = _manager_with_policy(settings, storage, store, policy)

    kb = await _make_kb(manager, "u1", "政策库")
    storage.credentials[("u1", "cred-1")] = _credential("u1")

    await manager.get_knowledge("u1", kb.id)

    assert policy.calls == [], "自己的凭据不该走策略"
