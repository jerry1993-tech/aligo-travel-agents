# -*- coding: utf-8 -*-
"""``scripts/seed_data.py`` 的离线用例：数据构造必须**确定、合法、幂等**。

==============================================================================
为什么这一组用例全部离线
==============================================================================
    真写路径要连 Milvus 与 PostgreSQL，而 ``make test`` 的硬约束是「不依赖
    Docker」（P1 验收）。所以这里测的不是「数据有没有写进去」，而是三件
    在没有外部服务时**依然能判定**的事：

        1. **确定性** —— 同一份代码跑两次，构造出的数据必须逐字节相同。
           用 ``random`` / ``hash()`` 会破坏它，而那种破坏很难在别处发现
           （数据「看起来」总是对的，只是每次不一样）。
        2. **字段合法** —— 实体字段齐备、日期是 ``YYYY-MM-DD``、状态是枚举成员、
           外键指向真实存在的用户。这些能在构造期判定，不必等落库报错。
        3. **幂等键稳定** —— 键相同且唯一；重复跑只补缺、不重复插。

    另有两条用例用 ``monkeypatch`` 造「连不上」的失败，断言错误信息的**形状**
    （中文、含下一步建议、不泄露凭据）—— 这条路径平时要真断网才走得到，
    正因如此它最容易被写成一条裸 traceback 而无人察觉。

    唯一会碰数据库的是一条 sqlite 内存库用例 —— sqlite 不需要任何外部服务，
    它把「先查后插」的幂等逻辑真正跑了两遍，而不是只断言键相等。
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest

from scripts import seed_data
from src.domain.enums import OrderStatus
from src.domain.rules import check_transition
from src.knowledge.manager import KB_ID_KEY, USER_ID_KEY
from src.llm.mock import MOCK_CREDENTIAL_TYPE
from src.storage.engine import build_business_engine

#: ``YYYY-MM-DD``。
_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

#: 合法的行程阶段（与 seed_data.trips 里的取值一致）。
_TRIP_STATUSES = {"PLANNED", "ONGOING", "DONE"}

#: 合法的订单类型（与 TravelOrder.kind 的约定一致）。
_ORDER_KINDS = {"flight", "train", "hotel"}


# ==============================================================================
# 一、确定性
# ==============================================================================
def test_build_plan_is_deterministic() -> None:
    """两次构造必须**完全相等**。

    ⚠️ 断言用 ``==``（dataclass 的逐字段比较）而不是「主键集合相等」：
    后者漏得掉「同一条记录的金额变了」—— 而那正是 ``random``/``hash()``
    污染数据时最先出现、也最隐蔽的症状。
    """
    assert seed_data.build_plan() == seed_data.build_plan()


def test_render_dry_run_is_deterministic() -> None:
    """dry-run 的文本逐字可复现（文本里不能混进时间戳、随机数）。"""
    first = seed_data.render_dry_run(seed_data.build_plan())
    second = seed_data.render_dry_run(seed_data.build_plan())
    assert first == second


async def test_dry_run_never_touches_external_services(settings, monkeypatch) -> None:
    """dry-run 必须**一次都不碰**外部服务。

    做法：把两个「会不会连服务」的入口换成一被调用就抛 ``AssertionError``
    的桩。若 dry-run 偷偷调了其中任何一个，用例立刻红 —— 这比「跑通就行」
    强得多，因为 dry-run 的全部价值就在于**它能在没有 Docker 的机器上跑通**。
    """

    def _boom(*_args, **_kwargs):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("dry-run 不得连接任何外部服务")

    monkeypatch.setattr(seed_data, "build_business_engine", _boom)
    monkeypatch.setattr(seed_data, "build_vector_store", _boom)

    code = await seed_data._run(settings, only="all", dry_run=True)
    assert code == 0


# ==============================================================================
# 二、政策文档：条数与结构
# ==============================================================================
def test_policy_documents_count_and_uniqueness() -> None:
    """文档数量足够多、标识唯一。"""
    docs = seed_data.policy_documents()
    assert len(docs) >= 8, f"政策文档只有 {len(docs)} 篇，细则覆盖不足"
    ids = [doc.doc_id for doc in docs]
    assert len(ids) == len(set(ids)), f"doc_id 有重复：{ids}"


def test_policy_document_structure() -> None:
    """每篇文档都非空，且 ``chunk_count`` 与真实小节数一致。"""
    for doc in seed_data.policy_documents():
        assert doc.title, f"{doc.doc_id} 缺标题"
        assert doc.category, f"{doc.doc_id} 缺分类"
        assert doc.sections, f"{doc.doc_id} 没有任何小节"
        assert doc.chunk_count == len(doc.sections)
        for section in doc.sections:
            assert section.heading, f"{doc.doc_id} 有小节缺标题"
            assert section.body, f"{doc.doc_id}/{section.heading} 正文为空"


def test_policy_documents_have_differentiating_detail() -> None:
    """细则要「够细」—— 具体数字是检索有区分度的前提。

    ⚠️ 断言的是**内容里出现具体数值**（里程阈值、住宿限额），而不是「字数够多」。
    一段长而空洞的总则对检索毫无帮助：任何 query 都会召回它。
    """
    docs = seed_data.policy_documents()
    total_chunks = sum(doc.chunk_count for doc in docs)
    assert total_chunks >= 30, f"总 chunk 数只有 {total_chunks}，检索粒度太粗"

    bodies = "\n".join(section.body for doc in docs for section in doc.sections)
    for number in ("1200", "600", "7", "15"):
        assert number in bodies, f"正文里找不到关键数值 {number}，条款可能不够具体"

    # 分类要有区分度，不能全挤在一类里。
    assert len({doc.category for doc in docs}) >= 6


# ==============================================================================
# 三、业务数据：字段合法性
# ==============================================================================
def test_users_are_valid() -> None:
    """用户标识唯一、字段非空。"""
    users = seed_data.users()
    assert users
    ids = [user.user_id for user in users]
    assert len(ids) == len(set(ids)), f"user_id 有重复：{ids}"
    for user in users:
        assert user.display_name and user.department
        assert user.employee_level and user.cost_center


def test_trips_reference_real_users_and_valid_fields() -> None:
    """行程日期格式、天数、状态、外键都合法。"""
    user_ids = {user.user_id for user in seed_data.users()}
    for trip in seed_data.trips():
        assert _DATE_RE.fullmatch(trip.depart_date), f"{trip.trip_id} 日期格式错"
        assert trip.days >= 1, f"{trip.trip_id} 天数非法"
        assert trip.status in _TRIP_STATUSES, f"{trip.trip_id} 状态非法：{trip.status}"
        assert trip.user_id in user_ids, f"{trip.trip_id} 指向不存在的用户 {trip.user_id}"


def test_orders_are_valid() -> None:
    """订单状态是枚举成员、金额为正、类型合法、归属用户存在。"""
    user_ids = {user.user_id for user in seed_data.users()}
    for order in seed_data.orders():
        assert isinstance(order.status, OrderStatus), f"{order.order_id} 状态不是枚举"
        assert order.amount > 0, f"{order.order_id} 金额非正"
        assert order.kind in _ORDER_KINDS, f"{order.order_id} 类型非法：{order.kind}"
        assert order.order_id and order.title
        assert order.user_id in user_ids, f"{order.order_id} 指向不存在的用户"
        assert isinstance(order.detail, dict)


def test_approvals_are_valid() -> None:
    """申请单状态是枚举成员、金额非负、日期合法、归属用户存在。"""
    user_ids = {user.user_id for user in seed_data.users()}
    for request in seed_data.approvals():
        assert isinstance(request.status, OrderStatus)
        assert request.amount >= 0
        assert _DATE_RE.fullmatch(request.depart_date), f"{request.request_id} 日期格式错"
        assert request.days >= 1
        assert request.user_id in user_ids, f"{request.request_id} 指向不存在的用户"


def _reachable_from_draft() -> set[OrderStatus]:
    """用**真实的状态机**算出「从 DRAFT 出发能到达哪些状态」。

    ⚠️ 不 import ``_ALLOWED_TRANSITIONS``（那是私有实现细节），而是用公开的
    :func:`~src.domain.rules.check_transition` 做一次广度优先。这样测的是
    「状态机的**行为**」，而不是「那张表长什么样」—— 表被重构时本用例照常有效。

    Returns:
        `set[OrderStatus]`: 从 ``DRAFT`` 可达的状态集合（含 ``DRAFT`` 自身）。
    """
    seen: set[OrderStatus] = {OrderStatus.DRAFT}
    frontier: list[OrderStatus] = [OrderStatus.DRAFT]
    while frontier:
        current = frontier.pop()
        for target in OrderStatus:
            if target in seen or target == current:
                continue
            if check_transition(current, target).allowed:
                seen.add(target)
                frontier.append(target)
    return seen


def test_seeded_statuses_are_reachable_in_the_state_machine() -> None:
    """种子里每个订单/申请单的状态都必须是状态机里**真实可达**的。

    ⚠️ 这条比「是枚举成员」强：枚举成员可以是任何人手写上去的一个值，而
    「可达」保证它确实能由合法迁移走到。一个不可达的状态（例如把 ``PAID``
    写成 ``REFUNDING``）在库里看着正常，却会让任何按状态筛选的查询都漏掉它。
    """
    reachable = _reachable_from_draft()
    for record in (*seed_data.orders(), *seed_data.approvals()):
        assert record.status in reachable, (
            f"{record.order_id if hasattr(record, 'order_id') else record.request_id} "
            f"的状态 {record.status} 从 DRAFT 不可达"
        )


# ==============================================================================
# 四、幂等键
# ==============================================================================
def _idempotency_keys(plan: seed_data.SeedPlan) -> dict[str, list[str]]:
    """把一份 plan 里的全部幂等键汇总出来（供下面的用例断言）。

    Args:
        plan (`seed_data.SeedPlan`): 待提取的 plan。

    Returns:
        `dict[str, list[str]]`: 每一类数据的幂等键列表。
    """
    return {
        "policy": [
            seed_data.policy_document_id(doc.doc_id) for doc in plan.policy_documents
        ],
        "users": [user.user_id for user in plan.users],
        "trips": [trip.trip_id for trip in plan.trips],
        "orders": [order.order_id for order in plan.orders],
        "approvals": [request.request_id for request in plan.approvals],
    }


def test_idempotency_keys_are_stable() -> None:
    """两次构造出的幂等键必须一致 —— 否则重复跑会写成新数据。"""
    assert _idempotency_keys(seed_data.build_plan()) == _idempotency_keys(
        seed_data.build_plan()
    )


def test_idempotency_keys_are_unique() -> None:
    """每一类数据的幂等键在类内唯一。"""
    keys = _idempotency_keys(seed_data.build_plan())
    for name, values in keys.items():
        assert len(values) == len(set(values)), f"{name} 的幂等键有重复：{values}"


def test_policy_keys_use_the_documented_prefix() -> None:
    """政策文档的 Milvus 幂等键必须带前缀 —— 便于把种子数据与用户知识区分开。"""
    keys = _idempotency_keys(seed_data.build_plan())
    assert keys["policy"]
    for key in keys["policy"]:
        assert key.startswith(seed_data.POLICY_DOC_ID_PREFIX)


# ==============================================================================
# 五、连不上外部服务时的错误信息形状
#
# 这一节用 monkeypatch 直接跳过「真的去连」这一步，只验证失败时**说什么**。
# 断言三件事：是中文、给了下一步、不泄露凭据。
# ==============================================================================
class _FakeSettings:
    """带凭据的假配置，用于验证错误信息**脱敏**。

    ⚠️ 只填 ``db.url`` / ``milvus.uri``：``_open_*`` 在探测失败时只会读这两个
    字段来拼消息，其余字段根本走不到。
    """

    class db:  # noqa: N801
        """假 db 段。"""

        url = "postgresql+asyncpg://aligo:s3cretpw@postgres:5432/aligo"

    class milvus:  # noqa: N801
        """假 milvus 段。"""

        uri = "http://aligo:s3cretpw@milvus:19530"
        collection = "aligo_travel_policy_dev"


async def test_business_connection_error_shape(monkeypatch) -> None:
    """业务库连不上时：中文说明 + 下一步，且不出现参数原文与裸 traceback。"""

    def _boom(_settings):  # noqa: ANN001
        raise OSError("connection refused")

    monkeypatch.setattr(seed_data, "build_business_engine", _boom)

    with pytest.raises(seed_data.SeedConnectionError) as excinfo:
        await seed_data.seed_business_data(
            _FakeSettings(),  # type: ignore[arg-type]
            seed_data.build_plan(),
        )

    message = str(excinfo.value)
    assert "无法连接业务库" in message
    assert "下一步" in message
    assert "Traceback" not in message
    # 凭据必须被脱敏：用户与密码都不该出现在日志里。
    assert "s3cretpw" not in message
    assert "aligo:" not in message


async def test_milvus_connection_error_shape(settings, monkeypatch) -> None:
    """Milvus 连不上时：中文说明 + 指向 milvus_init 的下一步。"""

    class _FailingStore:
        """``has_collection`` 直接抛错的假向量库。"""

        async def has_collection(self, _name: str) -> bool:
            raise RuntimeError("connection refused")

        async def __aexit__(self, *_args) -> None:  # noqa: ANN002
            return None

    monkeypatch.setattr(
        seed_data,
        "build_vector_store",
        lambda _settings: _FailingStore(),
    )

    with pytest.raises(seed_data.SeedConnectionError) as excinfo:
        await seed_data.seed_policy_documents(settings, seed_data.build_plan())

    message = str(excinfo.value)
    assert "无法连接 Milvus" in message
    assert "下一步" in message
    assert "make milvus_init" in message
    assert "Traceback" not in message


# ==============================================================================
# 六、幂等的**真**验证：在 sqlite 内存库上跑两遍
#
# 前面几条断的是「键相同」。这一条真的写两遍，断「第二遍新增为 0」——
# 前者证明不了 INSERT 逻辑本身幂等，后者能。
# ==============================================================================
async def test_business_write_is_idempotent(settings) -> None:
    """同一份 plan 写两遍，第二遍每张表新增都必须是 0。

    ⚠️ 复用**同一个** engine：sqlite 内存库由 StaticPool 支撑，同一 engine
    内表与数据才持久；换 engine 会得到两个互相看不见的库，也就测不了幂等。
    """
    engine = build_business_engine(settings)
    try:
        plan = seed_data.build_plan()
        first = await seed_data.seed_business_data(settings, plan, engine=engine)
        second = await seed_data.seed_business_data(settings, plan, engine=engine)
    finally:
        await engine.dispose()

    assert first == {"users": 5, "trips": 4, "orders": 4, "approvals": 3}
    assert second == {"users": 0, "trips": 0, "orders": 0, "approvals": 0}


# ==============================================================================
# 七、政策文档的正向用例：写入必须**带租户作用域键**，且重复播种幂等
#
# 前面几节只测「连不上时说什么」。这一节反过来测「连得上时写什么」——
# 用替身 storage + 替身向量库，在完全离线的情况下把
# ``seed_policy_documents`` 真正跑两遍，断言服务端检索所依赖的那两个键
# 确实进了 chunk metadata，且第二遍不会长出第二条 KB 记录。
#
# ⚠️ 这条用例存在的理由：本脚本此前的缺陷正是「写进去了却检索不到」。
# 只断言「函数没抛异常」是不够的 —— 数据确实写进去了，只是缺了作用域键，
# 服务端一过滤就全部落空。所以这里断言的是**metadata 的内容**，不是写没写。
# ==============================================================================


class _FakeVectorStore:
    """够跑通「探连通 → 建集合检查 → 写记录 → 删文档」的替身向量库。

    ⚠️ ``has_collection`` 必须返回 True：新的 ``_open_vector_store`` 把
    「集合不存在」当成播种失败（``SeedConnectionError``），返回 False 会让
    用例在写数据之前就退出。``insert`` 把每次写入的 ``VectorRecord`` 收集起来，
    metadata 断言就从这里取。
    """

    def __init__(self) -> None:
        self.inserted: list[list[object]] = []
        self.deleted: list[str] = []

    async def has_collection(self, _name: str) -> bool:
        return True

    async def create_collection(self, _name: str, _dimensions: int) -> None:
        return None

    async def insert(self, _collection: str, records: list[object]) -> None:
        self.inserted.append(list(records))

    async def delete(self, _collection: str, document_id: str) -> None:
        self.deleted.append(document_id)

    async def __aexit__(self, *_args) -> None:  # noqa: ANN002
        return None


class _FakeStorage:
    """内存替身 storage：保存凭据与 KB 记录，够管理器的句柄解析跑完。

    ⚠️ 只实现本用例真正走到的几个方法：``upsert_credential`` /
    ``get_credential``（``get_knowledge`` 解析 embedding 凭据用）与
    ``upsert_knowledge_base`` / ``get_knowledge_base``（KB 记录幂等）。
    """

    def __init__(self) -> None:
        self.credentials: dict[str, object] = {}
        self.knowledge_bases: dict[str, object] = {}

    async def __aenter__(self) -> "_FakeStorage":
        return self

    async def __aexit__(self, *_args) -> None:  # noqa: ANN002
        return None

    async def upsert_credential(self, _user_id: str, credential: object) -> str:
        self.credentials[credential.id] = credential  # type: ignore[attr-defined]
        return credential.id  # type: ignore[attr-defined]

    async def get_credential(self, user_id: str, credential_id: str) -> object | None:
        credential = self.credentials.get(credential_id)
        if credential is None:
            return None
        # 框架的 build_embedding_model 只读 ``.data``（它是凭据的 model_dump）。
        return SimpleNamespace(
            id=credential_id,
            user_id=user_id,
            data=credential.model_dump(),  # type: ignore[attr-defined]
        )

    async def upsert_knowledge_base(self, _user_id: str, record: object) -> object:
        self.knowledge_bases[record.id] = record  # type: ignore[attr-defined]
        return record

    async def get_knowledge_base(self, _user_id: str, kb_id: str) -> object | None:
        return self.knowledge_bases.get(kb_id)


async def test_seed_policy_documents_scopes_chunks_and_is_idempotent(
    settings,
    monkeypatch,
) -> None:
    """政策 chunk 必须带 ``aligo_kb_id`` / ``aligo_user_id``，且重复播种幂等。

    - (a) 每个写进去的 chunk，metadata 里都要有这两个键 —— 这正是服务端
      ``get_knowledge`` 的 ``metadata_filter`` 用来命中的键；缺了它们，
      数据在库里但检索永远空手而归。
    - (b) 跑两遍只留**一条** KB 记录、id 固定不变（用 ``--kb-user`` 之外的
      默认身份），证明固定 id + upsert 的幂等性成立。
    """
    store = _FakeVectorStore()
    monkeypatch.setattr(
        seed_data,
        "build_vector_store",
        lambda _settings: store,
    )
    storage = _FakeStorage()
    plan = seed_data.build_plan()

    first = await seed_data.seed_policy_documents(settings, plan, storage=storage)
    second = await seed_data.seed_policy_documents(settings, plan, storage=storage)

    # (a) 写入的每个 chunk 都带租户作用域键。
    assert store.inserted, "没有任何 chunk 被写进向量库"
    for batch in store.inserted:
        for record in batch:
            metadata = record.chunk.metadata  # type: ignore[attr-defined]
            assert metadata[KB_ID_KEY] == seed_data.KB_ID
            assert metadata[USER_ID_KEY] == seed_data.KB_USER_ID

    # (b) 幂等：两次播种只留一条 KB 记录，id 固定。
    assert list(storage.knowledge_bases) == [seed_data.KB_ID]
    assert first["knowledge_base_created"] is True
    assert second["knowledge_base_created"] is False
    assert first["knowledge_base_id"] == second["knowledge_base_id"] == seed_data.KB_ID

    # (c) 报出去的向量模型名必须是**剥掉包装后的真实现**。
    #     ⚠️ 这条不是「打印好看一点」：句柄上的模型被截止时间包装套着，直接取
    #     ``type(...).__name__`` 得到的是 "BoundedEmbeddingModel"，于是
    #     ``_run`` 里那句「本次用的是确定性假向量」的告警**永远不会触发** ——
    #     而它是唯一能发现「悄悄降级到 Mock」的地方（从检索输出上看不出来）。
    #     所以这里断言的是实现类名，包装名会让本用例变红。
    assert first["embedding_model"] == "MockEmbeddingModel", (
        "期望剥壳后的实现类名，实际拿到 "
        f"{first['embedding_model']!r} —— 包装层没有被 unwrap 掉。"
    )

    # (d) KB 记录里钉的凭据必须**就是**刚刚播种的那条。
    #     这条断言看着像同义反复，其实钉的是一处真实缺陷的形状：
    #     曾经无论配置如何，记录里都写死 MOCK_CREDENTIAL_TYPE ——
    #     于是「配好密钥后重灌」这条 remediation 永远修不好降级。
    #     记录与凭据是**两次**独立的写入，任何一边被写死都该在这里变红。
    record = storage.knowledge_bases[seed_data.KB_ID]
    config = record.data.embedding_model_config  # type: ignore[attr-defined]
    assert config.type == MOCK_CREDENTIAL_TYPE
    assert config.credential_id in storage.credentials, (
        "KB 记录指向的凭据不在 storage 里 —— 框架解析句柄时会以"
        "「凭据不存在」失败，而那条失败在 rag.py 里是逐条吞掉的。"
    )
    assert config.dimensions == settings.milvus.dimension


# ==============================================================================
# 六、知识库向量凭据：有密钥走真模型，没密钥才走 Mock
# ==============================================================================
async def test_kb_embedding_credential_is_mock_without_a_key(settings) -> None:
    """零密钥（测试档）下钉 Mock —— 这是唯一还能跑通的档。"""
    storage = _FakeStorage()

    credential_type, credential_id, model_name = (
        await seed_data.ensure_kb_embedding_credential(
            storage,
            settings,
            seed_data.KB_USER_ID,
        )
    )

    assert credential_type == MOCK_CREDENTIAL_TYPE
    assert model_name == seed_data.MOCK_EMBEDDING_MODEL_NAME
    assert credential_id in storage.credentials


async def test_kb_embedding_credential_is_the_real_provider_with_a_key(
    settings,
) -> None:
    """有密钥时必须钉**真**凭据 —— 否则「配了密钥也仍是假向量」。

    ⚠️ 这条用例针对的是一处已修复的真实缺陷：本函数曾经**无条件**写
    Mock 凭据，而结尾那句告警让运维「配置 DASHSCOPE_API_KEY 后重灌」——
    重灌只会再写一次 Mock。检测到了降级，却给了一条**永远修不好它**的
    补救办法；而 KB 记录的向量模型在创建时就被钉死（框架的
    ``PATCH /knowledge_bases`` 明确不接受改 embedding 配置），
    没有第二条路可以绕。

    ⚠️ ``model_copy`` 绕过了 pydantic 校验（见
    ``tests/test_server_agents_factory.py`` 的同类提醒），这里只改
    ``api_key`` 一个值、不动维度，因此不会掩盖任何校验失败。
    置的是一个**假 key**：本函数只做分支判定与凭据构造，不发任何请求。
    """
    keyed = settings.model_copy(deep=True)
    keyed.llm.api_key = "sk-not-a-real-key"
    storage = _FakeStorage()

    credential_type, credential_id, model_name = (
        await seed_data.ensure_kb_embedding_credential(
            storage,
            keyed,
            seed_data.KB_USER_ID,
        )
    )

    assert credential_type == "dashscope_credential"
    assert credential_id == seed_data.KB_EMBEDDING_CREDENTIAL_ID
    # 模型名取配置里的 embedding.model，而不是 Mock 那个占位名 ——
    # 名字对不上不会报错，只会让写入与检索用上「两个不同模型身份」的向量。
    assert model_name == keyed.embedding.model

    # 框架必须能按记录里的 type 解出向量模型类。解不出来的话，
    # KB 句柄解析会在 ``src/knowledge/rag.py`` 里被**逐条吞掉**，
    # 症状退化成「知识库列表看得见、问答就是不带出处」——
    # 从检索结果上完全看不出是凭据类型写错了。
    from agentscope.credential import CredentialFactory

    credential_cls = CredentialFactory.get_credential_class(credential_type)
    assert credential_cls is not None, f"类型 {credential_type!r} 未注册"
    assert credential_cls.get_embedding_model_class() is not None, (
        f"类型 {credential_type!r} 不支持 embedding —— KB 记录指过去等于没有模型。"
    )

    # 凭据必须写在**知识库归属者**名下：框架的 KB 链路按 owner-internal
    # 解析（``get_credential(kb.user_id, ...)``），写到别人名下会解析不到，
    # 而这条失败同样是静默的。
    stored = storage.credentials[credential_id]
    assert stored.type == credential_type  # type: ignore[attr-defined]
    assert stored.id == credential_id  # type: ignore[attr-defined]


__all__: list[str] = []
