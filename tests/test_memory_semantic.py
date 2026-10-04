# -*- coding: utf-8 -*-
"""语义记忆（``src/memory/semantic.py``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    语义记忆跑在**每一次对话**的上下文注入路径上。它有两种失败，
    后果完全不对称：

      · **读**失败（Milvus 挂了、模型超时）—— 绝不能因此让对话失败。
        用户应当拿到一个「少了点个性化」但**能用**的助手。
      · **写**失败（用户说「记住这个」）—— 必须让调用方知道。
        静默丢掉是一次欺骗：他下周才会发现助手没记住，
        而中间没有任何一处告诉过他。

    这两条合起来决定了本文件里**最大的那一组**断言：同一个类里，
    ``recall`` 永不抛而 ``remember`` 照常抛。

    另一组是**隔离**：``recall`` 的 ``metadata_filter`` 漏掉
    ``user_id`` 的后果不是「多召回几条」，而是 A 员工读到 B 员工的
    差旅记录 —— 一次数据泄露。这里用一个会**真的执行** filter 的
    假向量库把它挡住（而不是断言「调用了 filter」这种形似而神不似的检查）。
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

import pytest
from agentscope.embedding import EmbeddingResponse, EmbeddingUsage
from agentscope.rag import Chunk
from agentscope.message import TextBlock

from src.config import Settings
from src.memory.semantic import (
    MEMORY_KIND_KEY,
    MEMORY_USER_KEY,
    MemoryKind,
    SemanticMemory,
    memory_collection,
    note_id,
    render_profile_section,
)
from src.memory.profile import TravelerProfile


# ==============================================================================
# 测试替身
# ==============================================================================
class FakeEmbedding:
    """确定性的假向量模型：字符袋哈希。

    ⚠️ 不用随机向量。随机的后果是「同一段文本两次得到不同向量」，
    于是「记住 → 召回」这条最基本的路径会**偶发**失败 ——
    排查时会先怀疑检索逻辑，浪费大量时间。

    ⚠️ 也不要求它「语义好」。本文件的用例要验证的是**管道**
    （作用域、幂等、降级），不是模型质量。真模型的召回质量由
    评测集（``tests/evaluation/``）负责，不是单测。
    """

    def __init__(self, dimensions: int = 16) -> None:
        self.dimensions = dimensions
        self.calls: list[list[str]] = []

    async def __call__(self, inputs: list[str], **kwargs: object) -> EmbeddingResponse:
        self.calls.append(list(inputs))
        vectors = []
        for text in inputs:
            vector = [0.0] * self.dimensions
            for char in text:
                vector[ord(char) % self.dimensions] += 1.0
            # 归一化：余弦相似度才是有意义的。
            norm = sum(v * v for v in vector) ** 0.5 or 1.0
            vectors.append([v / norm for v in vector])
        return EmbeddingResponse(
            embeddings=vectors,
            usage=EmbeddingUsage(tokens=len(inputs), time=0.0),
        )


class FakeMilvusClient:
    """假 ``pymilvus.MilvusClient``：只记「谁被请求过改成 Strong」。

    ⚠️ 记忆的写路径会通过 ``store.get_client()`` 请求把集合的一致性
    改成 Strong（理由与实测数据见 ``src/knowledge/store.py``）。
    这里只记账 —— 用例要钉住的是「有没有请求、请求了几次」。
    """

    def __init__(self) -> None:
        self.altered: list[tuple[str, dict]] = []

    def alter_collection_properties(
        self,
        *,
        collection_name: str,
        properties: dict,
    ) -> None:
        """记下这次请求。"""
        self.altered.append((collection_name, properties))


class FakeVectorStore:
    """内存向量库，**真的执行**扁平等值过滤。

    ⚠️ 「真的执行」是刻意的。只断言「``search`` 收到了 ``metadata_filter``」
    是形似而神不似的检查 —— 它会在「filter 传对了但后端没拿它过滤」
    这个真实故障上照样变绿。这里把过滤逻辑实现出来（且**只**支持
    ``key == value``，与 Milvus 的能力一致），于是越权读取会
    **真的**读到别人的记录，用例才会红。
    """

    def __init__(self) -> None:
        self.collections: dict[str, dict[str, dict]] = {}
        self.delete_collection_calls: list[str] = []
        #: 每一次 ``create_collection`` 的 ``(集合名, 维度)`` —— 按调用顺序。
        self.create_collection_calls: list[tuple[str, int]] = []
        #: 假 pymilvus 客户端（一致性级别的修改从它走，见
        #: ``src/knowledge/store.py::request_strong_consistency``）。
        self.client = FakeMilvusClient()
        self.fail_with: Exception | None = None

    def get_client(self) -> "FakeMilvusClient":
        """把假客户端交出去（与框架 ``MilvusLiteStore.get_client`` 同形）。"""
        return self.client

    def _bucket(self, collection: str) -> dict[str, dict]:
        return self.collections.setdefault(collection, {})

    async def create_collection(self, name: str, dimensions: int) -> None:
        """幂等建集合 —— 与框架 ``MilvusLiteStore.create_collection`` 同形。

        ⚠️ 已存在时是 **no-op**（``agentscope/rag/_vdb/_milvus_lite.py:129-131``）。
        用例靠这个语义断言「每次写入都调一次 ensure」既是安全的、
        也不会把已有数据清掉。
        """
        if self.fail_with is not None:
            raise self.fail_with
        self.create_collection_calls.append((name, dimensions))
        if name in self.collections:
            return
        self.collections[name] = {}

    async def has_collection(self, name: str) -> bool:
        """集合是否已存在。"""
        return name in self.collections

    async def insert(self, collection: str, records: list) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        bucket = self._bucket(collection)
        for record in records:
            bucket[record.document_id] = {
                "vector": record.vector,
                "chunk": record.chunk,
            }

    async def delete(self, collection: str, document_id: str) -> None:
        if self.fail_with is not None:
            raise self.fail_with
        # ⚠️ 匹配不到时是 no-op（Milvus 的 ``delete(filter=...)`` 就是这样），
        # 这与「抛 KeyError」在幂等性上是两回事。
        self._bucket(collection).pop(document_id, None)

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list:
        if self.fail_with is not None:
            raise self.fail_with
        from agentscope.rag import VectorSearchResult

        hits = []
        for document_id, entry in self._bucket(collection).items():
            metadata = entry["chunk"].metadata
            if metadata_filter and any(
                metadata.get(key) != value for key, value in metadata_filter.items()
            ):
                continue
            score = sum(a * b for a, b in zip(query_vector, entry["vector"]))
            hits.append(
                VectorSearchResult(
                    score=score,
                    document_id=document_id,
                    chunk=entry["chunk"],
                ),
            )
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    async def list_documents(self, collection: str, metadata_filter: dict | None = None) -> list:
        from agentscope.rag import DocumentSummary

        summaries = []
        for document_id, entry in self._bucket(collection).items():
            metadata = entry["chunk"].metadata
            if metadata_filter and any(
                metadata.get(key) != value for key, value in metadata_filter.items()
            ):
                continue
            summaries.append(
                DocumentSummary(
                    document_id=document_id,
                    source=entry["chunk"].source,
                    chunk_count=1,
                    metadata=metadata,
                ),
            )
        return summaries

    async def delete_collection(self, name: str) -> None:
        self.delete_collection_calls.append(name)
        self.collections.pop(name, None)


@pytest.fixture
def store() -> FakeVectorStore:
    """一个空的假向量库。"""
    return FakeVectorStore()


@pytest.fixture
def memory(settings: Settings, store: FakeVectorStore) -> SemanticMemory:
    """接了假向量库与假模型的语义记忆。"""
    return SemanticMemory(
        vector_store=store,
        embedding_model=FakeEmbedding(),
        settings=settings,
    )


# ==============================================================================
# 一、集合名与 id：确定性
# ==============================================================================
def test_the_memory_collection_is_derived_and_distinct(settings: Settings) -> None:
    """★★★ 画像笔记**不能**住进政策知识库那个集合。

    ⚠️ 理由写在模块文档里，这里用断言把它钉住：
    ``scripts/milvus_init.py`` 的**恢复路径**就是「删掉契约集合重跑」
    （那是修复「维度建错了」的唯一办法）。两者共用一个集合的话，
    一次为了修政策库而做的删除会**顺手抹掉所有员工的长期记忆** ——
    不报错，只会在几天后表现为「助手怎么变笨了」。
    """
    derived = memory_collection(settings)

    assert derived != settings.milvus.collection, (
        "画像笔记与政策知识库共用了同一个集合！\n"
        "⚠️ 删政策库集合的恢复操作会连带清空所有用户记忆。"
    )
    assert derived.startswith(settings.milvus.collection), (
        "派生规则变了 —— 运维只需要记住一个集合名，"
        "派生名必须一眼看得出它从哪来。"
    )


def test_the_note_id_is_deterministic_across_processes() -> None:
    """★★★ ``note_id`` 在不同进程里必须**一模一样**。

    ⚠️ 用 ``hash()`` 实现的话，Python 对 str 的哈希默认加了随机盐
    （``PYTHONHASHSEED``，抵御哈希碰撞攻击）。症状是「服务重启后
    同一条笔记换了个 id」—— 于是「先删后写」删不掉旧的那份，
    Top-K 里出现两条一样的笔记，而它们还会挤掉别的记忆。

    ⚠️ 判据必须是**子进程**。同进程内断言相等是无效的：
    盐在一次进程里是固定的，无论怎么改实现都会通过。
    """
    script = (
        "import sys; sys.path.insert(0, '.');"
        "from src.memory.semantic import note_id;"
        "print(note_id('u1', '我一般坐靠窗'))"
    )
    outputs = set()
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            check=True,
        )
        outputs.add(result.stdout.strip())

    assert len(outputs) == 1, (
        f"note_id 在不同 PYTHONHASHSEED 下不一致：{outputs}\n"
        "⚠️ 八成是用了内置 hash()。"
    )


def test_the_note_id_fits_the_primary_key_width() -> None:
    """id 必须是 64 位十六进制。

    ⚠️ Milvus 的主键是 ``VARCHAR(64)``（``agentscope/rag/_vdb/_milvus_lite.py:133-160``）。
    超长的 id 会在**写入时**才被拒绝，而那条路径是用户明确要求
    「记住这个」的时候 —— 失败信息还是一条 Milvus 的参数错误。
    ``sha256`` 的十六进制摘要正好 64 位，是刻意的选择。
    """
    identifier = note_id("u1", "任意内容")

    assert len(identifier) == 64
    assert all(char in "0123456789abcdef" for char in identifier)


def test_different_users_with_the_same_text_get_different_ids() -> None:
    """同一句话、不同用户 ⇒ 不同 id。

    ⚠️ 反过来的话，B 员工说一句和 A 员工一样的话，会**覆盖**掉
    A 的那条笔记 —— 一次静默的跨租户数据破坏。
    """
    assert note_id("u1", "我坐靠窗") != note_id("u2", "我坐靠窗")


# ==============================================================================
# 二、写
# ==============================================================================
def test_remembering_then_recalling_returns_the_note(
    memory: SemanticMemory,
) -> None:
    """写入 → 召回命中（P4 验收的前半条）。

    ⚠️ 断言里检查 ``note_id`` 相等而不只是文本相等：文本相等只说明
    「召回了一条内容一样的东西」，而 ``note_id`` 相等才说明
    「召回的就是刚写进去的那一条」。
    """
    written = asyncio.run(memory.remember("u1", "我一般坐靠窗", MemoryKind.PREFERENCE))
    recall = asyncio.run(memory.recall("u1", "我一般坐靠窗"))

    assert recall.ok, recall.error
    assert [note.note_id for note in recall.notes] == [written.note_id]
    assert recall.notes[0].text == "我一般坐靠窗"
    assert recall.notes[0].kind == MemoryKind.PREFERENCE


def test_remembering_the_same_text_twice_is_idempotent(
    memory: SemanticMemory,
    store: FakeVectorStore,
) -> None:
    """同一句原文记两次，库里仍然只有**一条**。

    ⚠️ 幂等的理由不是省空间，而是 Top-K 的名额：两条一模一样的笔记
    会挤掉别的记忆的位置，于是「用户越是反复提到某件事，
    它越把其他记忆挤出召回窗口」—— 恰好与直觉相反。
    """
    asyncio.run(memory.remember("u1", "我一般坐靠窗"))
    asyncio.run(memory.remember("u1", "我一般坐靠窗"))

    stored = store.collections[memory.collection]
    assert len(stored) == 1, f"库里留下了 {len(stored)} 条重复笔记"


def test_remembering_a_blank_note_is_rejected(memory: SemanticMemory) -> None:
    """空内容必须被拒，而且是在**碰向量库之前**。

    ⚠️ 放进去的话，它会成为一个「什么也不匹配、但占一个 Top-K 名额」
    的条目，且没人能解释它为什么在那儿。
    """
    with pytest.raises(ValueError):
        asyncio.run(memory.remember("u1", "   "))


def test_remembering_an_unknown_kind_is_rejected(memory: SemanticMemory) -> None:
    """非法类别被拒。

    ⚠️ ``kind`` 会写进 metadata 并参与 ``metadata_filter`` 的拼接。
    放任意值进去的后果不是「分类不准」，而是拼出来的 Milvus 表达式
    在某个取值上语法错误 —— 那会让**整次检索**失败，而不是这一条。
    """
    with pytest.raises(ValueError):
        asyncio.run(memory.remember("u1", "内容", kind="不存在的类别"))


def test_the_written_metadata_is_flat_and_scoped(memory: SemanticMemory, store: FakeVectorStore) -> None:
    """写进去的 metadata 只有**两个扁平的字符串键**。

    ⚠️ ``metadata_filter`` 走到 Milvus 那边是 ``metadata["k"] == "v"``
    拼出来的表达式（``agentscope/rag/_vdb/_milvus_lite.py:504-515``），**只支持等值**。
    塞一个列表或嵌套 dict 进去不会当场报错 —— 它会在检索时
    拼出一句非法表达式，那次搜索整个失败。
    """
    asyncio.run(memory.remember("u1", "我一般坐靠窗"))

    stored = next(iter(store.collections[memory.collection].values()))
    metadata = stored["chunk"].metadata

    assert set(metadata) == {MEMORY_USER_KEY, MEMORY_KIND_KEY}
    assert all(isinstance(value, str) for value in metadata.values())


def test_the_chunk_carries_a_text_block_not_a_string(memory: SemanticMemory, store: FakeVectorStore) -> None:
    """``Chunk.content`` 是 ``TextBlock``，不是 ``str``。

    ⚠️ 传裸字符串会直接 ``ValidationError``（``agentscope/rag/_document.py:80-82``）。
    这条断言写下来是因为它**已经犯过一次**，而报错信息指向 pydantic
    的联合类型，不看文档很难反应过来。
    """
    asyncio.run(memory.remember("u1", "我一般坐靠窗"))

    stored = next(iter(store.collections[memory.collection].values()))
    assert isinstance(stored["chunk"].content, TextBlock)


# ==============================================================================
# 三、隔离
# ==============================================================================
def test_recall_never_returns_another_users_notes(
    memory: SemanticMemory,
) -> None:
    """★★★ 召回**只能**拿到自己的笔记。

    ⚠️ 这是本文件里后果最重的一条。``metadata_filter`` 漏掉
    ``user_id`` 的话，没有任何报错、没有任何异常日志 ——
    只是 A 员工的对话里出现了 B 员工的差旅记录。
    """
    asyncio.run(memory.remember("u1", "我的常旅客号是 CZ123"))
    asyncio.run(memory.remember("u2", "我住希尔顿"))

    recall = asyncio.run(memory.recall("u2", "常旅客号"))

    assert [note.text for note in recall.notes] == ["我住希尔顿"], (
        "u2 召回到了 u1 的笔记！"
    )


def test_forget_all_only_touches_one_user(
    memory: SemanticMemory,
    store: FakeVectorStore,
) -> None:
    """★★★ 清空某个人的记忆时，**绝不**碰集合本身。

    ⚠️ 与 ``SingleCollectionKbManager`` 里那条是同一种事故的另一副面孔：
    这个集合里住着**所有**用户的记忆。用 ``delete_collection``
    实现「清空我的记忆」，等于让一个用户的隐私操作清空全公司。
    """
    asyncio.run(memory.remember("u1", "我的常旅客号是 CZ123"))
    asyncio.run(memory.remember("u2", "我住希尔顿"))

    removed = asyncio.run(memory.forget_all("u1"))

    assert removed == 1
    assert store.delete_collection_calls == [], (
        "「清空我的记忆」调用了 delete_collection —— 那会清掉**所有人**的记忆！"
    )
    assert memory.collection in store.collections, "集合本身不该被删掉"
    remaining = asyncio.run(memory.recall("u2", "我住希尔顿"))
    assert [note.text for note in remaining.notes] == ["我住希尔顿"]


def test_forget_removes_exactly_one_note(memory: SemanticMemory) -> None:
    """``forget`` 按原文精确删一条。"""
    asyncio.run(memory.remember("u1", "我一般坐靠窗"))
    asyncio.run(memory.remember("u1", "我不吃辣"))

    asyncio.run(memory.forget("u1", "我不吃辣"))

    recall = asyncio.run(memory.recall("u1", "靠窗"))
    assert [note.text for note in recall.notes] == ["我一般坐靠窗"]


def test_forgetting_something_never_stored_is_a_no_op(memory: SemanticMemory) -> None:
    """忘掉一条从未存在过的笔记不报错。

    ⚠️ 它支撑「忘记」这个动作的幂等：用户连说两次「忘掉它」，
    第二次不该收到一个错误。
    """
    asyncio.run(memory.forget("u1", "从来没说过的话"))


# ==============================================================================
# 四、读永不抛 / 写照常抛
# ==============================================================================
def test_recall_never_raises_when_the_store_is_down(
    memory: SemanticMemory,
    store: FakeVectorStore,
) -> None:
    """★★★ 向量库挂了，``recall`` 返回 ``error`` 而**不是**抛异常。

    ⚠️ 这是本文件的核心断言。召回跑在每一次对话的上下文注入路径上，
    它抛异常的后果是**整轮对话**因为「画像笔记暂时读不出来」而失败 ——
    用户看到「助手挂了」，而实际上订票链路完全正常。
    """
    store.fail_with = RuntimeError("Milvus 连不上（模拟）")

    recall = asyncio.run(memory.recall("u1", "随便问点什么"))

    assert isinstance(recall.notes, list)
    assert recall.notes == []
    assert recall.ok is False
    assert recall.error and "Milvus" in recall.error


def test_recall_never_raises_when_the_embedding_model_is_down(
    settings: Settings,
    store: FakeVectorStore,
) -> None:
    """向量模型挂了同样只降级，不抛。"""

    class BrokenEmbedding:
        async def __call__(self, inputs: list[str], **kwargs: object) -> object:
            raise RuntimeError("向量服务 503")

    memory = SemanticMemory(
        vector_store=store,
        embedding_model=BrokenEmbedding(),
        settings=settings,
    )

    recall = asyncio.run(memory.recall("u1", "随便问点什么"))

    assert recall.ok is False
    assert recall.error


def test_remember_does_raise_when_the_store_is_down(
    memory: SemanticMemory,
    store: FakeVectorStore,
) -> None:
    """★★★ 但 ``remember`` **照常抛**。

    ⚠️ 这是与 :func:`test_recall_never_raises_when_the_store_is_down`
    刻意的不对称，也是本文件要钉住的第二条不变量：

      · 读失败只影响锦上添花 ⇒ 降级；
      · 写失败是**欺骗** —— 用户明确说了「记住这个」，静默丢掉的话，
        他下周才会发现，而中间没有任何一处告诉过他。

    ⚠️ 两条用例必须**同时**存在。只写其中一条，实现者很容易
    把「永不抛」当成整个类的性质，从而把写路径也一起吞掉。
    """
    store.fail_with = RuntimeError("Milvus 连不上（模拟）")

    with pytest.raises(RuntimeError):
        asyncio.run(memory.remember("u1", "记住我喜欢靠窗"))


def test_an_empty_query_is_not_reported_as_an_error(memory: SemanticMemory) -> None:
    """空查询返回空结果，``error`` 保持为 None。

    ⚠️ 把它算成失败的话，日志里会出现大量「召回失败」——
    而它们其实来自空对话，会把真正的失败淹掉。
    """
    recall = asyncio.run(memory.recall("u1", "   "))

    assert recall.ok is True
    assert recall.notes == []


def test_the_min_score_threshold_filters_weak_hits(memory: SemanticMemory) -> None:
    """低于 ``min_score`` 的命中被丢弃。

    ⚠️ 默认是 0.0（COSINE 下「正交」，即毫无关系）。定得更高会让
    召回率骤降且很难解释为什么 —— 所以它是一个**显式**参数，
    不是隐藏的默认。
    """
    asyncio.run(memory.remember("u1", "我一般坐靠窗"))

    # 一句与笔记几乎不相关的话。⚠️ 不能用**原文**当查询 ——
    # 那会得到 1.0 的相似度，无论门槛定多高都过滤不掉，
    # 用例会变成一条永远为真的断言。
    weak = asyncio.run(memory.recall("u1", "酒店的发票抬头怎么开", min_score=0.99))
    assert weak.notes == [], "高于实际相似度的门槛没有把这些命中滤掉"

    # ⚠️ 同一个查询在默认门槛（0.0）下**必须**回来 —— 否则这条用例
    # 可能只是因为「检索本身什么都没召回到」而变绿，
    # 那样它就完全没有在测门槛。
    kept = asyncio.run(memory.recall("u1", "酒店的发票抬头怎么开"))
    assert [note.text for note in kept.notes] == ["我一般坐靠窗"], (
        "默认门槛下应当召回 —— 否则上一条断言测的不是门槛，而是「检索坏了」"
    )


def test_recall_respects_the_configured_top_k(memory: SemanticMemory, settings: Settings) -> None:
    """召回条数受 ``ALIGO__MEMORY__TOP_K`` 约束。"""
    for index in range(settings.memory.top_k + 3):
        asyncio.run(memory.remember("u1", f"偏好第 {index} 条"))

    recall = asyncio.run(memory.recall("u1", "偏好"))

    assert len(recall.notes) <= settings.memory.top_k


# ==============================================================================
# 五、集合自愈：第一次写入前幂等地把集合建出来
# ==============================================================================
class MilvusLikeStore(FakeVectorStore):
    """像**真** Milvus 一样行事：集合不存在时，读写删一律报错。

    ⚠️ 这是本组用例的关键。``FakeVectorStore`` 的 ``_bucket`` 会在写入时
    顺手把桶建出来（对用例方便，但掩盖了「集合不存在」这个真实状态）；
    而真 Milvus 的行为是 ``MilvusException: collection not found`` ——
    2026-10-03 实测到的那个故障正是它。用这个子类，回归用例才会
    在「忘了建集合」的实现上**变红**。
    """

    def _require(self, collection: str) -> dict[str, dict]:
        if collection not in self.collections:
            raise RuntimeError(f"collection not found: {collection}")
        return self.collections[collection]

    async def insert(self, collection: str, records: list) -> None:
        self._require(collection)
        await super().insert(collection, records)

    async def delete(self, collection: str, document_id: str) -> None:
        self._require(collection)
        await super().delete(collection, document_id)

    async def search(
        self,
        collection: str,
        query_vector: list[float],
        top_k: int = 5,
        metadata_filter: dict | None = None,
    ) -> list:
        self._require(collection)
        return await super().search(collection, query_vector, top_k, metadata_filter)

    async def list_documents(
        self,
        collection: str,
        metadata_filter: dict | None = None,
    ) -> list:
        self._require(collection)
        return await super().list_documents(collection, metadata_filter)


def test_remembering_creates_the_collection_before_the_first_write(
    settings: Settings,
) -> None:
    """★★★ 集合不存在时，``remember`` 先建它再写 —— 而不是直接失败。

    ⚠️ 这条钉住的是 2026-10-03 实测到的一个真实故障：``milvus_init``
    当时只建政策知识库那个集合，于是全新部署上「记住这个」的第一条 RPC
    就以 ``collection not found`` 失败 —— 用户明确要求的动作一次都没成功过。
    修法写在 :meth:`SemanticMemory.remember` 里；这里用会**真的**在
    缺集合时拒绝读写的假向量库把它挡住。

    ⚠️ 断言维度而不是只断言「调过 create_collection」：维度错了的集合
    建出来才是更坏的失败（写入时才报错，或者分数没有意义）——
    见 ``src/knowledge/store.py`` 的模块文档。
    """
    store = MilvusLikeStore()
    memory = SemanticMemory(
        vector_store=store,
        embedding_model=FakeEmbedding(),
        settings=settings,
    )

    note = asyncio.run(memory.remember("u1", "我一般坐靠窗"))

    assert store.create_collection_calls == [
        (memory_collection(settings), settings.milvus.dimension),
    ], "remember 没有在写入前把集合建出来（或缺了维度）"
    # 建完必须真能写能读 —— 只断言调用会发生，就在「建了但写不进去」时照样绿。
    recall = asyncio.run(memory.recall("u1", "座位偏好"))
    assert [hit.text for hit in recall.notes] == ["我一般坐靠窗"]
    assert recall.notes[0].note_id == note.note_id


def test_ensuring_the_collection_is_idempotent(settings: Settings) -> None:
    """★★ 第二次写入不会因为「重复建集合」丢掉第一条笔记。

    ⚠️ 幂等性的真正考验不是「第二次不报错」，而是**不覆盖数据**：
    框架的 ``create_collection`` 已存在时是 no-op（``agentscope/rag/_vdb/_milvus_lite.py:129-131``），
    本用例把这条语义变成断言 —— 若哪天有人把实现换成
    ``delete_collection`` + ``create_collection``，这里立刻变红。
    """
    store = MilvusLikeStore()
    memory = SemanticMemory(
        vector_store=store,
        embedding_model=FakeEmbedding(),
        settings=settings,
    )

    asyncio.run(memory.remember("u1", "我一般坐靠窗"))
    asyncio.run(memory.remember("u1", "我对花生过敏"))

    assert len(store.create_collection_calls) == 2, "每次写入都应 ensure 一次（幂等）"
    recall = asyncio.run(memory.recall("u1", "偏好"))
    assert {hit.text for hit in recall.notes} == {"我一般坐靠窗", "我对花生过敏"}, (
        "重复 ensure 把已有笔记弄丢了"
    )


def test_the_first_write_requests_strong_consistency_once(
    settings: Settings,
) -> None:
    """★★★ 首次写入时把集合的一致性请求为 Strong，且**每个实例只请求一次**。

    ⚠️ 为什么这件事落在写路径上：记忆那个集合是 ``remember`` **惰性创建**的
    （见上一条用例），所以「集合刚被建出来」的时刻只有这里知道。而
    Milvus 建集合时的默认一致性是 Bounded —— 实测「插入后 0.39–1.37s
    才可见、删除后 0.56–0.58s 才不可见」，即 ``POST /notes`` 返回 200
    之后紧接着的 ``GET /notes`` 可能读不到刚写的那条。数据与理由见
    ``src/knowledge/store.py::CONSISTENCY_LEVEL``。

    ⚠️ 「只请求一次」是这条用例的另一半，同样重要：改一致性是一次 DDL
    （服务端要落元数据），每次写入都发一遍既浪费又与别的写者抢同一把
    DDL 锁。第二次写入必须**不再**请求 —— 断言 ``== 1`` 而不是 ``>= 1``。
    """
    store = MilvusLikeStore()
    memory = SemanticMemory(
        vector_store=store,
        embedding_model=FakeEmbedding(),
        settings=settings,
    )

    asyncio.run(memory.remember("u1", "我一般坐靠窗"))
    asyncio.run(memory.remember("u1", "我对花生过敏"))

    assert store.client.altered == [
        (
            memory_collection(settings),
            {"collection.consistency_level": "Strong"},
        ),
    ], "要么没请求一致性，要么每次写入都重复请求了一次 DDL"


def test_recall_does_not_create_the_collection(settings: Settings) -> None:
    """★ 读路径**不**建集合 —— 这是刻意的，不是遗漏。

    ⚠️ ``recall`` 在每一轮对话的上下文注入路径上跑。让它去建集合，
    等于每轮对话白花一次 RPC；而读失败本来就已经降级成「没有记忆」
    （:meth:`SemanticMemory.recall` 的文档），不需要集合存在。
    建集合的职责只在写路径与 ``scripts/milvus_init.py``。
    """
    store = MilvusLikeStore()
    memory = SemanticMemory(
        vector_store=store,
        embedding_model=FakeEmbedding(),
        settings=settings,
    )

    recall = asyncio.run(memory.recall("u1", "随便问点什么"))

    # ⚠️ 缺集合时读**允许**降级：不抛异常、notes 为空、error 有值
    #（与 ``test_recall_never_raises_when_the_store_is_down`` 同一条不变量），
    # 但绝不能顺手把集合建出来 —— 那是写路径与 milvus_init 的职责。
    assert recall.notes == []
    assert recall.ok is False and recall.error
    assert store.create_collection_calls == [], "读路径不该建集合"


# ==============================================================================
# 六、渲染
# ==============================================================================
def test_rendering_an_empty_state_produces_nothing() -> None:
    """★★★ 没有信息时渲染出**空串**。

    ⚠️ 不这么做的话，系统提示里会出现「用户画像：无」——
    它既占 token，又让模型以为「这个用户特意说过自己没有任何偏好」。
    对绝大多数新用户来说，这是错误的第一印象。
    """
    from src.memory.semantic import SemanticRecall

    assert render_profile_section(None, SemanticRecall()) == ""
    assert render_profile_section(TravelerProfile(user_id="u1"), SemanticRecall()) == ""


def test_structured_facts_win_and_the_prompt_says_so() -> None:
    """★★★ 结构化画像与语义笔记同时出现时，提示里必须写明优先级。

    ⚠️ 只把两段并排放进去是不够的。模型看到「偏好舱位：ECONOMY」
    与「他上次说要坐商务舱」这两句互相矛盾的话，会挑一个**更显眼**的，
    而不是更权威的 —— 而「更显眼」通常意味着更长、更具体的那句，
    也就是语义笔记。所以优先级必须**显式写出来**。
    """
    from src.memory.semantic import MemoryNote, SemanticRecall

    rendered = render_profile_section(
        TravelerProfile(user_id="u1", preferred_cabin="ECONOMY"),
        SemanticRecall(
            notes=[MemoryNote(text="他上次说要坐商务舱", kind="observation", score=0.9, note_id="x")],
        ),
    )

    assert "ECONOMY" in rendered
    assert "他上次说要坐商务舱" in rendered
    assert "以上面为准" in rendered, (
        "渲染出来的文本没有写明二者冲突时以结构化为准 —— "
        "模型会挑更显眼的那句，而不是更权威的那句。"
    )


def test_the_frequent_flyer_number_is_rendered_verbatim() -> None:
    """★★★ 常旅客号必须**原样**输出，一个字都不能改。

    ⚠️ 它是硬事实。任何「美化」（加空格、分段、取后四位）都会让
    它在真实下单时订不上 —— 而那是用户看不见的一次静默失败，
    只在几个月后发现里程没累积时才暴露。
    """
    from src.memory.semantic import SemanticRecall

    rendered = render_profile_section(
        TravelerProfile(user_id="u1", frequent_flyer_numbers={"CZ": "CZ1234567890"}),
        SemanticRecall(),
    )

    assert "CZ1234567890" in rendered
    assert "不得改写" in rendered


def test_only_a_bounded_number_of_notes_is_rendered() -> None:
    """★★★ 塞进 Prompt 的笔记条数是**有上限**的。

    ⚠️ 召回的 ``top_k`` 与塞进 prompt 的条数是两个数：多召回几条
    是为了让调用方有得挑，但全塞进去会让当前这一轮真正该做的事
    （订一张去北京的票）被历史细节淹没。
    """
    from src.memory.semantic import MemoryNote, SemanticRecall

    recall = SemanticRecall(
        notes=[
            MemoryNote(text=f"笔记 {index}", kind="preference", score=0.9, note_id=str(index))
            for index in range(10)
        ],
    )

    rendered = render_profile_section(None, recall, max_notes=3)

    assert "笔记 0" in rendered
    assert "笔记 2" in rendered
    assert "笔记 3" not in rendered
