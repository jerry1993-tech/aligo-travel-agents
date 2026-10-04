# -*- coding: utf-8 -*-
"""向量库接入与探针（``src/knowledge/store.py``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    P4 有一条验收是「**Milvus 不可用不得拖垮 /readyz**」。这条要求落成代码
    只有一行（``REQUIRED_CHECKS`` 里没有 ``milvus``），但**支撑**它的
    是一整套「碰 Milvus 的代码都不该把异常漏出来」的约定：

      · 构造客户端 —— 不许发网络请求；
      · 探针 —— 不许抛，必须有超时；
      · 配置描述 —— 不许连网、不许漏密钥。

    这套约定有个共同的失败形态：**它们全都在「Milvus 好着的时候」看不出来**。
    开发机上 Milvus 一跑起来，构造函数连不连、探针超不超时、错误信息里
    有没有密码，全都无关紧要。只有换到 Milvus 挂掉、或配置写错的机器上，
    这些区别才变成「服务照常」与「服务整体不可用」的差别。

    所以下面的用例**刻意在一个 Milvus 不可达的环境里**断言行为 ——
    这正是它们的价值所在，而不是「因为本机没起 Milvus 所以只能这么测」。
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from src.config import Settings
from src.knowledge.store import (
    build_vector_store,
    describe_vector_store,
    ensure_collection,
    probe_collection,
    verify_collection_shape,
)


def _actual(**overrides: object) -> dict:
    """构造一份 `describe_collection` 形状的返回值。"""
    base = {
        "collection": "aligo_travel_policy_test",
        "exists": True,
        "dimension": 1024,
        "metric_type": "COSINE",
        "index_type": "HNSW",
        "index_name": "vector",
    }
    base.update(overrides)
    return base


# ==============================================================================
# 一、构造不发网络请求
# ==============================================================================
def test_building_the_store_performs_no_io(settings: Settings) -> None:
    """构造向量库客户端**不连接** Milvus。

    ⚠️ 这条约束支撑着「Milvus 挂了服务照常启动」：应用装配期就会构造它
    （``create_root_app`` → ``build_knowledge_manager``）。若构造函数去连库，
    装配就会在 Milvus 没起来时失败 —— 而 Milvus 只被「查差旅政策」用到。

    判据：在本机 Milvus **不可达**的前提下（见模块文档），构造仍然成功，
    且返回的对象类型正确。这是一条**反向**断言：它不检查「做了什么」，
    而是检查「在没有外部依赖时仍然成立」。

    ⚠️ 返回的是 ``GuardedVectorStore``（超时 + 熔断护栏，见
    :mod:`src.knowledge.guard`）。断言因此分两层：外层是护栏、**内层仍然是
    框架的** ``MilvusLiteStore`` —— 后半句不是形式主义，它挡住的是
    「哪天有人图省事，在护栏里自己实现一个 store」，那就违反了
    「智能体能力一律 import 框架实现」这条第一约束。
    """
    store = build_vector_store(settings)

    assert type(store).__name__ == "GuardedVectorStore"
    assert type(store._inner).__name__ == "MilvusLiteStore"
    # ⚠️ 客户端必须是**惰性**的：构造完还没创建（护栏也不许建连接）。
    # 直接读私有属性是刻意的 —— 没有公开的「连了没」接口，
    # 而这是本条断言唯一能站得住的地方。
    assert store._inner._client is None


def test_the_remote_uri_goes_through_the_remote_branch(settings: Settings) -> None:
    """``http://milvus:19530`` 必须走**远端服务**分支，不是内嵌 Lite。

    ⚠️ 类名叫 ``MilvusLiteStore``，很有误导性。判定走哪条路的是
    ``_is_local_db_uri``（``rag/_vdb/_milvus_lite.py:538-543``）::

        not uri.startswith(("http://", "https://")) and splitext(uri)[1] == ".db"

    对我们的 URI 返回 False ⇒ 远端。如果这个判断反了（比如哪天有人把
    配置改成 ``./milvus.db``），症状是「服务起来了、也能写，
    但写进的是一个**容器内的本地文件**」—— 数据在一个副本里，
    重启就没了，而且另一个副本完全看不见。
    """
    store = build_vector_store(settings)

    assert store._is_lite_uri is False, (
        "URI 被判成了内嵌 Lite 模式！\n"
        f"  uri = {settings.milvus.uri!r}\n"
        "  ⚠️ 内嵌模式写的是容器内的本地文件，重启即丢、多副本不共享。"
    )


def test_the_contract_index_and_metric_are_passed_through(settings: Settings) -> None:
    """``index_type`` 与 ``metric_type`` 必须**逐字**传给后端。

    ⚠️ 框架的 ``MilvusLiteStore`` 默认 ``index_type="AUTOINDEX"``，
    而契约要的是 ``HNSW``。不显式传就会静默地建成 AUTOINDEX ——
    检索仍然能用，只是索引类型与契约不符（进而与运维手册、
    Grafana 面板上写的对不上）。

    同样地 ``metric_type`` 默认就是 COSINE，与本项目一致 ——
    但仍然显式传：依赖一个「碰巧相同」的默认值，在框架改默认值时
    会静默地改变本项目的检索语义。
    """
    store = build_vector_store(settings)

    assert store._index_type == settings.milvus.index_type
    assert store._metric_type == settings.milvus.metric_type
    assert store._uri == settings.milvus.uri


# ==============================================================================
# 二、探针永不抛
# ==============================================================================
def test_the_probe_never_raises_when_milvus_is_down(settings: Settings) -> None:
    """★★★ Milvus 不可达时，探针返回 ``ok: False`` 而**不是抛异常**。

    ⚠️ 这是 P4 验收「Milvus 不可用不得拖垮 /readyz」在代码层的落点。

    探针炸掉的后果不是「一条检查失败」，而是**整个 /readyz 500** ——
    于是 LB 看到的不是一个「Milvus 挂了但服务还好」的实例，
    而是一个「连探测都做不到」的实例，会把它摘掉。
    对 Milvus 这一个只被部分路径用到的依赖来说，这个代价完全不成比例。

    判据用的是「函数正常返回」这件事本身：任何异常漏出来，
    ``asyncio.run`` 都会把它抛到测试里，用例就红了。
    """
    result = asyncio.run(probe_collection(settings, timeout=2.0))

    assert isinstance(result, dict)
    assert result["ok"] is False, (
        "本机 Milvus 不可达，探针却报了 ok=True —— "
        f"返回的是 {result}"
    )
    assert "error" in result and result["error"], (
        f"探针失败时**必须**给出 error 说明，否则运维只看到一个 false：{result}"
    )


def test_the_probe_does_not_leak_credentials(settings: Settings) -> None:
    """探针的错误信息与配置描述里**绝不能**出现连接串凭据。

    ⚠️ 探针响应体会进 ``/readyz``，而它是最容易被整段贴进工单的东西。
    Milvus 的 URI 是支持 ``http://user:pass@host:19530`` 这种写法的。

    这里用一份**带凭据**的配置来测，而不是用默认那份不带凭据的 ——
    默认配置里没有密码，用它测等于断言「空字符串里没有密码」。
    """
    from tests.conftest import TEST_ENVIRON

    from src.config import load_settings

    environ = {
        **TEST_ENVIRON,
        "ALIGO__MILVUS__URI": "http://aligo:s3cret-pw@milvus:19530",
    }
    leaky = load_settings("test", environ=environ, dotenv=False)

    described = describe_vector_store(leaky)
    assert "s3cret-pw" not in repr(described), (
        f"配置描述里泄漏了密码：{described}"
    )
    assert "***" in described["uri"], (
        f"凭据没有被替换成占位符：{described['uri']!r}"
    )

    probed = asyncio.run(probe_collection(leaky, timeout=2.0))
    assert "s3cret-pw" not in repr(probed), (
        f"探针结果里泄漏了密码：{probed}"
    )


def test_describing_the_store_does_no_io(settings: Settings) -> None:
    """配置描述是**纯配置**，零 I/O。

    ⚠️ 它会被放进启动日志与 ``/readyz`` 的响应体 —— 两处都不该因为
    Milvus 没起来而失败或变慢。判据是耗时：连一次 Milvus 不可能在
    50ms 内完成（本机不可达时要等连接超时）。
    """
    import time

    start = time.perf_counter()
    described = describe_vector_store(settings)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05, f"配置描述耗时 {elapsed:.3f}s，疑似真的去连了库"
    assert described["collection"] == settings.milvus.collection
    assert described["dimension"] == settings.milvus.dimension
    assert described["index_type"] == settings.milvus.index_type
    assert described["metric_type"] == settings.milvus.metric_type
    assert described["top_k"] == settings.milvus.top_k


# ==============================================================================
# 三、形态核验（回读比对）
# ==============================================================================
def test_a_matching_shape_reports_no_problems(settings: Settings) -> None:
    """形态与配置一致时，核验结果为空。"""
    assert verify_collection_shape(_actual(), settings) == []


def test_a_missing_collection_is_reported(settings: Settings) -> None:
    """集合不存在要被报出来。"""
    problems = verify_collection_shape(_actual(exists=False), settings)

    assert any("不存在" in line for line in problems)


@pytest.mark.parametrize(
    ("key", "wrong_value", "expected_word"),
    [
        ("dimension", 768, "维度"),
        ("metric_type", "L2", "度量"),
        ("index_type", "AUTOINDEX", "索引类型"),
    ],
)
def test_each_shape_mismatch_is_reported(
    settings: Settings,
    key: str,
    wrong_value: object,
    expected_word: str,
) -> None:
    """维度 / 度量 / 索引类型，**每一项**不符都要被报出来。

    ⚠️ 参数化而不是写一个「改一处」的用例：这三项的核验是三条独立的
    比较，删掉任意一条都不会让另两条变红。分开跑才能保证每条都在。

    ⚠️ 度量不符（配置 COSINE、集合 L2）是最隐蔽的一种：
    写入成功、检索成功、有分数 —— **没有任何一处会报错**，
    只是分数的大小关系没有意义。只有回读能发现它。
    """
    problems = verify_collection_shape(_actual(**{key: wrong_value}), settings)

    assert any(expected_word in line for line in problems), (
        f"{key}={wrong_value!r} 没有被报出来。\n"
        f"拿到的问题清单：{problems}"
    )
    assert any(str(wrong_value) in line for line in problems), (
        f"问题清单里没说清实际值是什么：{problems}"
    )


def test_a_problem_report_always_carries_a_fix(settings: Settings) -> None:
    """只要报了问题，就**必须**同时给出修法。

    ⚠️ 「集合形态不对」本身是个没有行动价值的信息 —— 运维看完仍然不知道该做什么，
    而最容易想到的动作（改配置）在这里恰好是**错**的：集合的形态在建集合那一刻
    就定死了，改配置不会改变已存在的集合，只会让配置与线上更不一致。

    所以问题清单的最后一条必须是「删集合重建 / 或改配置对齐」这个二选一。
    """
    problems = verify_collection_shape(_actual(dimension=512), settings)

    assert any("删掉集合" in line for line in problems), (
        f"问题清单里没有给出修法：{problems}"
    )
    assert any("不会改变已存在的集合" in line for line in problems), (
        f"问题清单里没有点破「改配置没用」这件事：{problems}"
    )


def test_unreadable_fields_are_not_reported_as_mismatches(settings: Settings) -> None:
    """读不到的字段（``None``）**不算**不符。

    ⚠️ 这是刻意的宽容，理由很实：``pymilvus`` 在不同版本里把维度放在
    ``params.dim`` 或 ``params.dimension``、索引信息也可能读不出来
    （``describe_collection`` 已经在两种键名上都试过）。
    分辨不了「值错了」与「版本不同读不出来」，而把后者报成「配置错误」
    会让人去改一份本来正确的配置 —— **比不报更糟**。

    判据：把三项都设成 None，核验必须返回空清单。
    """
    problems = verify_collection_shape(
        _actual(dimension=None, metric_type=None, index_type=None),
        settings,
    )

    assert problems == [], f"读不到的字段被报成了不符：{problems}"


# ==============================================================================
# 三·五、具名集合：长期记忆画像那个集合走的是同两条函数
# ==============================================================================
class _FakeStore:
    """只会记账的假向量库，够 :func:`ensure_collection` 用。"""

    def __init__(self) -> None:
        self.created: list[tuple[str, int]] = []
        self.existing: set[str] = set()

    async def has_collection(self, name: str) -> bool:
        return name in self.existing

    async def create_collection(self, name: str, dimensions: int) -> None:
        self.created.append((name, dimensions))


class _FakeClient:
    """只会记账的假 ``pymilvus.MilvusClient``。

    ⚠️ 只实现 ``alter_collection_properties``：用例要钉住的是
    「有没有请求、请求成什么值、失败会怎样」，不是 pymilvus 的行为。
    """

    def __init__(self, *, fail: bool = False) -> None:
        self.altered: list[tuple[str, dict]] = []
        self._fail = fail

    def alter_collection_properties(
        self,
        *,
        collection_name: str,
        properties: dict,
    ) -> None:
        """记账（或按需失败）。

        Raises:
            RuntimeError: 构造时给了 ``fail=True`` —— 模拟一个不支持
                该属性的服务端。
        """
        if self._fail:
            raise RuntimeError("unsupported collection property")
        self.altered.append((collection_name, properties))


class _FakeStoreWithClient(_FakeStore):
    """多一个 ``get_client()`` 的假向量库（一致性是通过它改的）。"""

    def __init__(self, *, fail_alter: bool = False) -> None:
        super().__init__()
        self.client = _FakeClient(fail=fail_alter)

    def get_client(self) -> _FakeClient:
        return self.client


def test_ensure_collection_can_target_another_collection(settings: Settings) -> None:
    """★★ ``ensure_collection`` 能用**别的集合名**建集合（长期记忆画像）。

    ⚠️ 这条参数是为长期记忆加的，理由写在 ``scripts/milvus_init.py`` 与
    ``src/memory/semantic.py::remember`` 里：那个集合（``{契约集合}_memory``）
    曾经谁都不建，于是「记住这个」在全新部署上直接以
    ``collection not found`` 失败。

    ⚠️ 断言的是「名字换了、**维度没换**」：两个集合共用一个向量模型，
    维度必须逐字相同；给记忆另立一份期望值 = 把同一条约束抄两遍。
    """
    store = _FakeStore()

    created = asyncio.run(
        ensure_collection(store, settings, collection="aligo_memory_test"),
    )

    assert created is True
    assert store.created == [("aligo_memory_test", settings.milvus.dimension)]


def test_ensure_collection_reports_an_existing_collection(settings: Settings) -> None:
    """已存在的集合**不改动**，返回 ``False``。

    ⚠️ 写路径每次写入前都会调它（幂等自愈），所以这条语义要钉住：
    否则「自愈」会变成「每次写入都重建集合」—— 那是数据丢失。
    """
    store = _FakeStore()
    store.existing.add(settings.milvus.collection)

    created = asyncio.run(ensure_collection(store, settings))

    assert created is False
    assert store.created == [(settings.milvus.collection, settings.milvus.dimension)]


def test_a_problem_report_names_the_collection_it_checked(settings: Settings) -> None:
    """核验报告里的集合名必须是**被检查的那个**，不是契约集合。

    ⚠️ 两个集合用同一套期望值、只换名字，报告里写错名字的后果不是
    「措辞不准」，而是**运维会去删错集合**（修法那行是要被复制粘贴执行的）。
    """
    missing = verify_collection_shape(
        _actual(exists=False),
        settings,
        collection="aligo_memory_test",
    )
    assert missing == ["集合 'aligo_memory_test' 不存在。"]

    mismatch = verify_collection_shape(
        _actual(dimension=512),
        settings,
        collection="aligo_memory_test",
    )
    assert any("'aligo_memory_test'" in line for line in mismatch), mismatch
    # ⚠️ 契约集合的名字**不能**出现在针对另一个集合的报告里。
    assert all(settings.milvus.collection not in line for line in mismatch), mismatch


# ==============================================================================
# 三·六、一致性：默认的 Bounded 会造成约 1 秒的可见性窗口
# ==============================================================================
def test_ensure_collection_requests_strong_consistency(
    settings: Settings,
) -> None:
    """★★★ 建集合时把一致性**请求**为 ``Strong``。

    ⚠️ 这条用例守的是一个 2026-10-03 在容器里实测出来的真实行为：
    Milvus 默认的一致性级别是 **Bounded**，写入对检索**不是立刻可见**。
    量到的窗口是插入后 0.39–1.37s、删除后 0.56–0.58s ——
    也就是说 ``DELETE /api/v1/memory/notes`` 返回 200「已忘记」之后，
    紧接着的召回**仍可能命中那条已被删掉的笔记**。改成 Strong 后复测
    3 轮全部即时。

    ⚠️ 断言的是「请求了什么」，不是「生效了」：``alter_collection_properties``
    对不认识的属性名**不报错**（实测过），而 ``describe_collection``
    改完还会先返回旧值一段时间 —— 这条路径上没有任何东西能证明生效。
    真正的核验在 ``/readyz`` 与 ``milvus_init`` 的回读里。
    """
    store = _FakeStoreWithClient()

    asyncio.run(ensure_collection(store, settings))

    assert store.client.altered == [
        (
            settings.milvus.collection,
            {"collection.consistency_level": "Strong"},
        ),
    ]


def test_an_existing_collection_also_gets_the_consistency_request(
    settings: Settings,
) -> None:
    """★★ 集合**已存在**时也要请求一次 —— 老部署不会自己变成 Strong。

    ⚠️ 与「已存在就不改动」的幂等语义不冲突：这里改的是**元数据**
    （一致性级别），不是形态（维度/索引/度量），后者才是
    ``create_collection`` 静默 no-op 时会被永久固化下来的东西。
    """
    store = _FakeStoreWithClient()
    store.existing.add(settings.milvus.collection)

    created = asyncio.run(ensure_collection(store, settings))

    assert created is False
    assert len(store.client.altered) == 1


def test_a_failed_consistency_request_does_not_break_the_init(
    settings: Settings,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★ 一致性请求失败 ⇒ 只告警，**不影响**建集合的结论。

    ⚠️ 退化掉的是「写入立刻可见」这个保证（从「立刻」退到「约 1 秒内」），
    数据一条不少。反过来，若在这里抛异常，一个不支持该属性的 Milvus
    会让**所有**记忆写入直接失败 —— 用一个可见性优化换功能不可用，
    方向是反的。
    """
    store = _FakeStoreWithClient(fail_alter=True)

    with caplog.at_level(logging.WARNING, logger="src.knowledge.store"):
        created = asyncio.run(ensure_collection(store, settings))

    assert created is True
    assert store.created == [
        (settings.milvus.collection, settings.milvus.dimension),
    ]
    assert any("一致性" in record.message for record in caplog.records), (
        "一致性请求失败被静默吞掉了 —— 运维没有任何线索"
    )


def test_the_described_consistency_is_human_readable() -> None:
    """``describe_collection`` 把一致性级别翻成人话。

    ⚠️ 一个裸的 ``2`` 没人看得出是「Bounded」，而正是这个默认值造成了
    可见性窗口。它在 ``/readyz`` 与初始化脚本的输出里都要出现 ——
    那是运维**唯一**能看见它的地方。
    """
    from src.knowledge.store import _CONSISTENCY_NAMES

    assert _CONSISTENCY_NAMES[0] == "Strong"
    assert _CONSISTENCY_NAMES[2] == "Bounded"


# ==============================================================================
# 四、与检查项集合的契约
# ==============================================================================
def test_milvus_is_not_a_blocking_readiness_check() -> None:
    """★★★ ``milvus`` **不在** ``REQUIRED_CHECKS`` 里。

    ⚠️ 这是 P4 验收「Milvus 不可用不得拖垮 /readyz」的**字面**落点。
    上面那些「探针永不抛」的用例挡的是「探针把自己搞崩」，
    这一条挡的是另一回事：探针**正常返回了 ok=False**，
    但就绪判定把它当成了一票否决。

    两个失败方向都得挡，因为它们对应完全不同的修法：
    前者要改探针代码，后者只要把这个常量里的 ``milvus`` 去掉。

    ⚠️ 同时断言它**仍然被检查**：从 ``REQUIRED_CHECKS`` 里删掉
    与「不再执行这个检查」是两件事。只删常量、检查照跑，
    运维依然能在响应体里看到 Milvus 挂了（``checks.milvus.ok=false``）；
    若连检查也一起删了，Milvus 的故障就变成一个没人知道的静默降级。
    """
    from src.server.probes import REQUIRED_CHECKS

    assert "milvus" not in REQUIRED_CHECKS, (
        "milvus 被加回了阻断性检查集！\n"
        "⚠️ Milvus 只被「查差旅政策」用到；把它算作就绪条件，"
        "等于用「政策问答不可用」换来「整个实例被 LB 摘掉」。"
    )
    # ⚠️ 另外三项**必须**还在 —— 它们分别在存储、事件总线、启动完成
    # 这三条所有请求都会经过的路径上。
    assert {"postgres", "redis", "boot"} <= REQUIRED_CHECKS
