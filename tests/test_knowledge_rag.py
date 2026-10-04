# -*- coding: utf-8 -*-
"""知识库桥接层（``src/knowledge/rag.py``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    桥接层把「用户的知识库」解析成运行时句柄，再包成框架的
    ``RAGMiddleware``。它很短，但里面有五处错了**不会报错、只会悄悄变坏**：

      1. **多租户串号** —— 拿错用户的句柄。检索会正常返回，
         只是返回的是别人的资料。这类事故在结果上完全看不出来，
         所以必须有专门一条用例盯着「另一个用户的 KB 不出现在结果里」。

      2. **每回合重建向量模型** —— ``get_knowledge`` 会构造向量模型
         （dashscope 新建连接池 / local 加载权重，秒级）。忘了缓存不会
         报错，只会让每个用户的每句话都慢几秒。用**调用计数**把它钉住 ——
         计数是不会骗人的。

      3. **知识库故障拖垮对话** —— 向量库抖动时若把异常抛出去，
         整条对话链路 500。本层的决策是「吞掉并降级为空」，这条决策
         必须有一条用例钉死，否则很容易被后来者「顺手改成抛出去」。

      4. **缓存的 TTL 退化成滑动窗口** —— 命中时顺手刷新 ``stored_at``
         （看起来很像「保活」）会让 TTL 只在用户空闲时才到点。而
         ``agents_factory.py`` 是每请求装配一次的，于是持续对话的用户
         永远不重解析，被撤销的凭据一直用到进程重启。它**不会报错**，
         只会让「撤销迟早生效」这句承诺默默失效 —— 用**两个**时间点
         （250 与 400，TTL=300）把「绝对窗口」与「空闲窗口」区分开。

      5. **并发各解析一遍** —— 缓存只在解析**完成之后**才起作用，而解析
         中途有真实的挂起点，于是同一个用户的两个并发请求会双双判定
         「没命中」、各自把每个 KB 解析一遍（各建一次连接池 / 各加载一次
         权重）。同样是「只贵不错」，同样从外观上看不出来 —— 用闸门把
         解析者卡在中途，再用**调用计数**证明后来者没有重复解析。

    ⚠️ 这些用例**不碰** Milvus / 网络 / 数据库：管理器是鸭子类型的替身，
    只要求 ``list_knowledge_bases`` 与 ``get_knowledge`` 两个方法。
    这正是本层「可以被测试替身驱动」这条设计的价值所在。
"""

from __future__ import annotations

import asyncio

import pytest

from agentscope.middleware import RAGMiddleware
from src.config import Settings
from src.knowledge.rag import (
    RagBridgeStatus,
    build_rag_middlewares,
    clear_handle_cache,
)
from src.web_embedding import MockEmbeddingModel


# ==============================================================================
# 测试替身
# ==============================================================================
class _FakeData:
    """``KnowledgeBaseRecord.data`` 的替身：桥接层只读 name / description。"""

    def __init__(self, name: str, description: str) -> None:
        self.name = name
        self.description = description


class _FakeRecord:
    """``KnowledgeBaseRecord`` 的替身：桥接层只读 ``id`` 与 ``data``。"""

    def __init__(self, kb_id: str, name: str, description: str = "") -> None:
        self.id = kb_id
        self.data = _FakeData(name, description)


class _FakeHandle:
    """``KnowledgeBase`` 运行时句柄的替身。

    ⚠️ 只带桥接层会读到的字段：``name`` / ``description``（进 agentic 工具
    描述）与 ``embedding_model``（进只读痕迹）。**不**实现 ``search`` ——
    这些用例不测检索质量，实现它反而会诱导后来者在这里造一个假的向量库。
    """

    def __init__(
        self,
        kb_id: str,
        name: str,
        user_id: str,
        embedding_model: object | None = None,
    ) -> None:
        self.name = name
        self.description = f"{name} 的描述"
        self.embedding_model = embedding_model or _NamedModel()
        # 真实的 KnowledgeBase 会在这里带租户作用域，替身保持同样的形状，
        # 以便将来有代码去读它时不会因为「替身没有」而误报。
        self.metadata_filter = {"aligo_kb_id": kb_id, "aligo_user_id": user_id}


class _NamedModel:
    """一个「看着像正常向量模型」的替身（类名会进痕迹）。"""


class _FakeClock:
    """可拨动的单调时钟替身。

    ⚠️ 只在**一个**用例里用（TTL 的滑动窗口那条），因为只有它需要
    「时间精确地走 N 秒」而不真的 ``sleep``。实现刻意只有一个
    ``monotonic()``：多一个方法就多一分「测试替身比被测代码还复杂」的风险，
    而 ``time`` 在 rag.py 里只被用来取这一个值（见该文件的 ``import time``）。
    """

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        """返回当前（被我们摆布的）时刻。"""
        return self.now


class FakeKbManager:
    """鸭子类型的知识库管理器替身。

    ⚠️ 自己写一个而不是用真管理器：真管理器的 ``get_knowledge`` 会去构造
    真的向量模型（读凭据、建 HTTP 连接池），而本文件要断言的是
    **「调了几次 get_knowledge」**，不是「模型能不能构造」。
    用真管理器会让计数断言被一堆无关行为淹没，还会引入对凭据的依赖。

    Attributes:
        records: ``user_id -> 记录列表``。**按用户分桶**正是「用户隔离」
            这条用例的现场。
        list_calls: ``list_knowledge_bases`` 的调用次数。
        get_calls: ``get_knowledge`` 的调用次数（缓存命中的核心断言）。
        fail_list: 为真时 ``list_knowledge_bases`` 抛异常。
        fail_get_ids: 这些 kb_id 的 ``get_knowledge`` 抛异常（模拟凭据被删）。
        gate: 非 ``None`` 时，``get_knowledge`` 先等这个闸门 —— 让用例把
            解析者**卡在解析中途**，从而在它完成之前制造出真实的并发交错
            （真实现里这一步是 I/O，一定会让出事件循环，替身必须显式补上，
            否则第一个协程会一路跑完，第二个根本挤不进来）。
    """

    def __init__(self) -> None:
        self.records: dict[str, list[_FakeRecord]] = {}
        self.handles: dict[tuple[str, str], _FakeHandle] = {}
        self.list_calls = 0
        self.get_calls = 0
        self.resolved_ids: list[str] = []
        self.fail_list = False
        self.fail_get_ids: set[str] = set()
        self.gate: asyncio.Event | None = None

    def add(self, user_id: str, record: _FakeRecord, handle: _FakeHandle) -> None:
        """登记一个用户的知识库。"""
        self.records.setdefault(user_id, []).append(record)
        self.handles[(user_id, record.id)] = handle

    async def list_knowledge_bases(self, user_id: str) -> list[_FakeRecord]:
        """只返回**该用户**的记录。"""
        self.list_calls += 1
        if self.fail_list:
            raise RuntimeError("存储不可用（模拟）")
        return list(self.records.get(user_id, []))

    async def get_knowledge(self, user_id: str, kb_id: str) -> _FakeHandle:
        """解析句柄；记录是否存在与用户绑定（越权即「找不到」）。"""
        self.get_calls += 1
        if self.gate is not None:
            await self.gate.wait()
        if kb_id in self.fail_get_ids:
            raise RuntimeError(f"凭据缺失（模拟）：{kb_id}")
        handle = self.handles.get((user_id, kb_id))
        if handle is None:
            raise RuntimeError(f"知识库不存在（模拟）：{kb_id}")
        self.resolved_ids.append(kb_id)
        return handle


@pytest.fixture(autouse=True)
def _clean_cache():
    """每个用例前后清空进程级句柄缓存。

    ⚠️ 缓存本来按**管理器对象**分桶，用例各自新建管理器就天然隔离了。
    仍然显式清一次，是因为「隔离」依赖 ``WeakKeyDictionary`` 的弱引用语义，
    而那条语义不值得让每个用例都去推理一遍 —— 清掉最省心，也让
    :func:`~src.knowledge.rag.clear_handle_cache` 这个公开 API 有测试覆盖。
    """
    clear_handle_cache()
    yield
    clear_handle_cache()


def _manager_with(user_id: str, kb_id: str, name: str = "政策库") -> FakeKbManager:
    """造一个只含一个知识库的管理器替身。"""
    manager = FakeKbManager()
    manager.add(
        user_id,
        _FakeRecord(kb_id, name),
        _FakeHandle(kb_id, name, user_id),
    )
    return manager


# ==============================================================================
# 一、没有知识库 ⇒ 空列表，且不去解析
# ==============================================================================
async def test_no_knowledge_bases_yields_no_middleware(
    settings: Settings,
) -> None:
    """用户没有知识库时返回 ``[]``，且**不**调用 ``get_knowledge``。

    ⚠️ 为什么不是「挂一个检索永远为空的中间件」：agentic 模式下模型会
    看到「已装备 0 个知识库」的描述，却仍可能去调那个工具，然后把
    「没查出来」读成「知识库里没有」—— 而真相是「压根没接检索」。
    这两种情况对用户的处置完全不同，不能被一个空中间件抹平。
    """
    manager = FakeKbManager()

    middlewares = await build_rag_middlewares("u1", settings, manager)

    assert middlewares == []
    assert manager.get_calls == 0, "没有知识库却去解析句柄，是白造向量模型"


# ==============================================================================
# 二、有知识库 ⇒ 句柄被原样传进中间件
# ==============================================================================
async def test_the_handle_is_passed_into_the_middleware(
    settings: Settings,
) -> None:
    """有知识库时返回 ``[RAGMiddleware]``，且句柄对象被传了进去。"""
    manager = _manager_with("u1", "kb-a")
    handle = manager.handles[("u1", "kb-a")]

    middlewares = await build_rag_middlewares("u1", settings, manager)

    assert len(middlewares) == 1
    assert isinstance(middlewares[0], RAGMiddleware)
    assert middlewares[0]._knowledge_bases == [handle], (
        "中间件拿到的不是管理器解析出来的那个句柄"
    )


async def test_parameters_default_to_the_repository_settings(
    settings: Settings,
) -> None:
    """``top_k`` 默认取自 ``settings.milvus.top_k``，且可被入参覆盖。

    ⚠️ 这条钉的是「默认值取仓库既有 settings」这条约定：本轮刻意
    **不**新增配置段（``extra="forbid"`` 下加键风险高），所以桥接层的
    默认值必须来自既有字段，而不是自己编一个魔数。
    """
    manager = _manager_with("u1", "kb-a")

    default = await build_rag_middlewares("u1", settings, manager)
    assert default[0]._parameters.top_k == settings.milvus.top_k

    overridden = await build_rag_middlewares(
        "u1", settings, manager, top_k=3,
    )
    assert overridden[0]._parameters.top_k == 3


async def test_a_non_positive_top_k_is_rejected(settings: Settings) -> None:
    """显式传入 ``top_k=0`` 直接报错 —— 0 条召回不是检索，是静默失效。"""
    manager = _manager_with("u1", "kb-a")

    with pytest.raises(ValueError):
        await build_rag_middlewares("u1", settings, manager, top_k=0)


# ==============================================================================
# 三、⚠️ 用户隔离：只解析本人的知识库
# ==============================================================================
async def test_only_the_owners_knowledge_bases_are_resolved(
    settings: Settings,
) -> None:
    """为 u1 装配时，u2 的知识库**不**出现在结果里，也**不**被解析。

    ⚠️ 这是本文件最重要的一条。串号不会报错、检索照常返回，
    只是返回的是别人的资料 —— 在多租户系统里这是事故。
    两个用户各有一个知识库的场景是刻意的：只有存在**别人的** KB，
    「只查本人的」与「查全部」才区分得开。
    """
    manager = FakeKbManager()
    manager.add("u1", _FakeRecord("kb-a", "A 的政策"), _FakeHandle("kb-a", "A 的政策", "u1"))
    manager.add("u2", _FakeRecord("kb-b", "B 的政策"), _FakeHandle("kb-b", "B 的政策", "u2"))

    middlewares = await build_rag_middlewares("u1", settings, manager)

    assert len(middlewares) == 1
    kb_names = [kb.name for kb in middlewares[0]._knowledge_bases]
    assert kb_names == ["A 的政策"], f"u1 的中间件里混进了别人的库：{kb_names}"
    assert "kb-b" not in manager.resolved_ids, (
        f"u1 的装配解析了 u2 的知识库：{manager.resolved_ids}"
    )


# ==============================================================================
# 四、缓存：第二次调用不再重复解析
# ==============================================================================
async def test_a_second_identical_call_does_not_re_resolve(
    settings: Settings,
) -> None:
    """同样入参第二次调用，``get_knowledge`` **不再**被调用（缓存命中）。

    ⚠️ ``get_knowledge`` 是「构造向量模型」的代名词（新建连接池 /
    加载权重，秒级）。这里断言的是**调用次数**，因为它是唯一不会骗人的
    证据 —— 缓存有没有生效，从返回的中间件外观上完全看不出来。

    ⚠️ ``list_knowledge_bases`` 仍然会被调用：它是廉价的存储查询，
    而且**必须**每次调用才能发现「知识库列表变了」。两者一个贵一个便宜，
    区别对待正是缓存策略的要点。
    """
    manager = _manager_with("u1", "kb-a")

    first = await build_rag_middlewares("u1", settings, manager)
    assert manager.get_calls == 1

    second = await build_rag_middlewares("u1", settings, manager)

    assert manager.get_calls == 1, "第二次调用又重新解析了句柄 —— 缓存没生效"
    assert manager.list_calls == 2, "应当每次都核对一次列表以发现失效"
    # 复用的一定是**同一个**句柄对象，而不是「内容相同」的新对象：
    # 后者仍意味着重建过向量模型。
    assert second[0]._knowledge_bases[0] is first[0]._knowledge_bases[0]


async def test_the_cache_is_invalidated_when_a_kb_is_added(
    settings: Settings,
) -> None:
    """知识库列表变了 ⇒ 签名不匹配 ⇒ 重新解析。

    ⚠️ 不失效的后果是「新建的知识库永远不生效」，直到进程重启 ——
    而用户看到的是「我明明建好了，它却说没有」。
    """
    manager = _manager_with("u1", "kb-a")
    await build_rag_middlewares("u1", settings, manager)
    assert manager.get_calls == 1

    manager.add("u1", _FakeRecord("kb-b", "B 库"), _FakeHandle("kb-b", "B 库", "u1"))

    middlewares = await build_rag_middlewares("u1", settings, manager)

    assert manager.get_calls == 3, "新增知识库后没有重新解析（kb-a 也应重建）"
    assert {kb.name for kb in middlewares[0]._knowledge_bases} == {"政策库", "B 库"}


async def test_the_cache_expires_so_a_revoked_credential_takes_effect(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """★★★ 句柄缓存必须在 TTL 后失效 —— 否则被撤销的凭据会一直用到进程重启。

    ⚠️ 这条守的是一个**签名看不住**的口子，值得逐句说明为什么签名不够用：

        句柄是从**凭据**构造出来的（``get_knowledge`` → ``build_embedding_model``
        → ``CredentialFactory``），而签名只看 ``KnowledgeBaseRecord`` 的
        ``(id, name, description)``。管理员撤销或轮换那个向量凭据时，
        **记录一个字段都不会变** —— 记录里存的是 ``credential_id``，
        不是 key 本身。于是签名照样相等，缓存一直命中，``get_knowledge``
        再也不被调用（这正是缓存存在的意义），那个已经作废的凭据就这样
        被一路用下去。

        没有 TTL 的话，唯一的出路是**重启进程** —— 而症状是「凭据明明撤销了，
        检索却还在按旧配置走」，在管理界面上完全看不出来。

    ⚠️ 用 ``monkeypatch`` 把 TTL 按到 0 来模拟「时间走过了 TTL」。不 ``sleep``：
    那会把 300 秒变成用例的运行时间。这里要验证的是**判据读了这个常量**
    这件事，而不是时钟本身（``time.monotonic`` 无需我们测）。
    """
    manager = _manager_with("u1", "kb-a")

    # 第一次：解析成功并写入缓存。
    await build_rag_middlewares("u1", settings, manager)
    assert manager.get_calls == 1

    # TTL 之内：命中缓存，不再解析。
    await build_rag_middlewares("u1", settings, manager)
    assert manager.get_calls == 1

    # 管理员撤销了这个库的向量凭据（记录本身没变 ⇒ 签名仍然相等）。
    manager.fail_get_ids = {"kb-a"}

    # 时间走过 TTL。
    monkeypatch.setattr("src.knowledge.rag._HANDLE_TTL_SECONDS", 0.0)

    middlewares = await build_rag_middlewares("u1", settings, manager)

    assert manager.get_calls == 2, (
        "TTL 到点后没有重新解析 —— 缓存里的句柄会一直用下去，"
        "被撤销的凭据要等到进程重启才生效。"
    )
    assert middlewares == [], (
        "凭据已经撤销，这个知识库应当被摘掉（与「解析失败只丢它自己」一致），"
        "而不是继续拿着旧句柄检索。"
    )


async def test_hitting_the_cache_does_not_push_the_ttl_forward(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """★★★ TTL 是**绝对窗口**：命中缓存不得刷新 ``stored_at``。

    ⚠️ 这条防的是「TTL 悄悄退化成空闲窗口」—— 上面那条用例（把 TTL 按到 0）
    抓不到它，因为按到 0 之后**每次**都算过期，写不写 ``stored_at``
    都一样。两条用例合起来才把 TTL 钉死：一条证明「到点会失效」，
    这条证明「不命中才重新计时」。

        时间线（TTL = 300）：t=0 首解析 → t=250 命中 → t=400 必须重解析。
        若命中时把 ``stored_at`` 刷新成 250，则 400 那一刻离上次刷新才 150 秒，
        仍被判为新鲜 —— 于是**每一个**持续对话的用户都够不到 TTL：
        ``agents_factory.py`` 是每请求装配一次的，请求之间的间隔远小于 300 秒。
        症状就是「凭据撤销了，检索却一直按旧的走」，直到进程重启。

    ⚠️ 用假时钟而不是真的等 300 秒，也不是把 TTL 按小：这两个数字
    （250 与 400）是**相对 TTL 的比值**才有意义，按小 TTL 会让用例
    依赖具体取值，调 TTL 时一起变红。
    """
    clock = _FakeClock()
    monkeypatch.setattr("src.knowledge.rag.time", clock)
    manager = _manager_with("u1", "kb-a")

    # t=0：首解析，写入缓存（stored_at = 0）。
    await build_rag_middlewares("u1", settings, manager)
    assert manager.get_calls == 1

    # t=250：TTL 未到（250 < 300），命中缓存。
    clock.now = 250.0
    await build_rag_middlewares("u1", settings, manager)
    assert manager.get_calls == 1, "250 秒时不该过期 —— 否则缓存等于没有"

    # t=400：距**首次**解析已 400 秒（> 300），必须重解析；
    # 距**上一次调用**只有 150 秒 —— 这正是滑动窗口会误判成「还新鲜」的点。
    clock.now = 400.0
    await build_rag_middlewares("u1", settings, manager)

    assert manager.get_calls == 2, (
        "命中缓存时刷新了 stored_at —— TTL 变成了滑动窗口，"
        "只要用户一直在说话就永远不过期。被撤销的凭据会一直用到进程重启，"
        "而 agents_factory.py 每请求都调用本函数，所以这个窗口永远够不到。"
    )


# ==============================================================================
# 五、⚠️ 知识库故障不能拖垮对话
# ==============================================================================
async def test_a_listing_failure_is_swallowed_and_yields_no_middleware(
    settings: Settings,
) -> None:
    """``list_knowledge_bases`` 抛异常时，返回 ``[]`` 而**不**向上抛。

    ⚠️ 这是刻意的降级，不是偷懒。``src/knowledge/__init__.py`` 写明
    「Milvus 不可用不能拖垮服务」。若这里把异常放出去，一次向量库抖动
    就会让**所有人**的对话 500 —— 而他们说的很可能只是一句「你好」。
    检索是增强，不是主链路。这条用例把该决策钉死，防止后来者
    「顺手改成抛出去」。
    """
    manager = _manager_with("u1", "kb-a")
    manager.fail_list = True

    middlewares = await build_rag_middlewares("u1", settings, manager)

    assert middlewares == []


async def test_one_broken_kb_does_not_take_down_the_others(
    settings: Settings,
) -> None:
    """某个 KB 解析失败只丢它自己，其余照常可用；且本轮结果**不**入缓存。

    ⚠️ 两件事一起断言，因为它们耦合：如果失败的结果被缓存了，一次瞬时
    故障就会固化成「永远少一个知识库」，而用户无从察觉。用第二次调用的
    ``get_calls`` 增加来证明「没入缓存、下一回合重试了」。
    """
    manager = FakeKbManager()
    manager.add("u1", _FakeRecord("kb-ok", "好库"), _FakeHandle("kb-ok", "好库", "u1"))
    manager.add("u1", _FakeRecord("kb-bad", "坏库"), _FakeHandle("kb-bad", "坏库", "u1"))
    manager.fail_get_ids = {"kb-bad"}

    first = await build_rag_middlewares("u1", settings, manager)

    assert [kb.name for kb in first[0]._knowledge_bases] == ["好库"]
    calls_after_first = manager.get_calls
    assert calls_after_first == 2  # 两条都试过

    # 第二次：坏库仍然坏，但因为第一轮没入缓存，好库会被**重新**解析。
    await build_rag_middlewares("u1", settings, manager)
    assert manager.get_calls > calls_after_first, (
        "部分失败的结果被缓存了 —— 一次瞬时故障会被固化成永久状态"
    )


# ==============================================================================
# 六、只读痕迹：向量模型落到 Mock 时留下线索
# ==============================================================================
async def test_a_mock_embedding_model_is_surfaced_as_degraded(
    settings: Settings,
) -> None:
    """句柄用了 Mock 向量 ⇒ 中间件上挂的痕迹 ``degraded`` 为真。

    ⚠️ Mock 向量之间没有语义：检索会「成功」（有结果、有分数）但结果
    没有意义，从结果本身完全看不出来。这条痕迹是唯一能让人事后把
    「检索结果莫名不对」与「这台机器少装了什么」对上号的线索。
    """
    manager = FakeKbManager()
    manager.add(
        "u1",
        _FakeRecord("kb-a", "假向量库"),
        _FakeHandle(
            "kb-a",
            "假向量库",
            "u1",
            embedding_model=MockEmbeddingModel(dimensions=8),
        ),
    )

    middlewares = await build_rag_middlewares("u1", settings, manager)

    status = middlewares[0].rag_bridge_status
    assert isinstance(status, RagBridgeStatus)
    assert status.degraded is True
    assert "MockEmbeddingModel" in status.embedding_models
    assert status.knowledge_base_ids == ("kb-a",)


async def test_a_normal_embedding_model_is_not_degraded(
    settings: Settings,
) -> None:
    """正常向量模型不该被误报成降级 —— 痕迹要能区分，否则等于没有。"""
    manager = _manager_with("u1", "kb-a")

    middlewares = await build_rag_middlewares("u1", settings, manager)

    status = middlewares[0].rag_bridge_status
    assert status.degraded is False
    assert status.embedding_models == ("_NamedModel",)
    assert status.user_id == "u1"


# ==============================================================================
# 七、⚠️ 并发：同一个用户的两个请求只解析一次
# ==============================================================================
async def test_two_concurrent_assemblies_resolve_the_handles_once(
    settings: Settings,
) -> None:
    """同一个用户的两次**并发**装配，``get_knowledge`` 只被调用**一次**。

    ⚠️ 防的是什么：缓存只在解析**完成之后**才起作用。解析中途有真实的挂起
    点（``await kb_manager.get_knowledge(...)`` —— 它内部还要读凭据、建向量
    模型），于是两个请求会双双判定「缓存里没有」，各自把每个 KB 解析一遍。
    用户开两个标签页、或前端同时发一条消息与一次会话刷新，就能凑出这个交错。

    这不报错、结果也等价，只是**贵**：一次多余的解析 = 一次多余的连接池
    构造或模型权重加载（秒级），而这正是本模块存在的理由。

    ⚠️ 用**闸门**而不是 ``sleep`` 来制造交错：``gate`` 让第一个协程卡在
    解析中途（真实现里那一步是 I/O，一定会让出事件循环），第二个协程因此
    必然在「缓存空、有人在解析」这个瞬间进场 —— 不依赖调度顺序的运气。

    ⚠️ 断言 ``get_calls == 1`` 而不是 ``>= 1``：计数是唯一不会骗人的证据，
    两个协程拿到的是**同一个句柄对象**这一点看不出来谁解析的。
    """
    manager = _manager_with("u1", "kb-a")
    manager.gate = asyncio.Event()

    first_task = asyncio.create_task(build_rag_middlewares("u1", settings, manager))
    await asyncio.sleep(0)  # 让第一个跑到闸门上（此时它已经登记为解析者）
    second_task = asyncio.create_task(build_rag_middlewares("u1", settings, manager))
    await asyncio.sleep(0)  # 让第二个跑到「有人在解析」那条分支上

    manager.gate.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert manager.get_calls == 1, (
        f"两个并发装配把句柄解析了 {manager.get_calls} 次 —— 单飞闸门没生效，"
        "同一个用户的并发请求会各建一次连接池 / 各加载一次模型权重。"
    )
    assert manager.resolved_ids == ["kb-a"]
    # 后来者拿到的必须是**同一个**句柄对象 —— 内容相同的新对象仍然意味着
    # 重建过向量模型。
    assert second[0]._knowledge_bases[0] is first[0]._knowledge_bases[0]


async def test_a_waiter_does_not_borrow_a_result_with_another_signature(
    settings: Settings,
) -> None:
    """等待者**重查缓存**，不复用解析者那份「已经签不上名」的结果。

    ⚠️ 这条钉的是一个设计选择，而不是实现细节。等待者手里有自己的
    ``records`` 与 ``signature``，解析者手里有它的 —— 两者可能不同（两次
    ``list_knowledge_bases`` 之间管理员改了知识库）。若等待者直接复用解析者的
    结果，调用方就会拿**自己的**签名去写**别人的**句柄：缓存项里的签名与
    句柄从此对不上，之后一次命中会按错的 id 取句柄（``zip`` 静默截断，
    表现为「某个知识库莫名其妙不在检索范围里」，直到 TTL 到点）。

    等待者改为重查缓存之后，这条不变量由既有判据保证：签名不匹配自然不命中，
    等待者自己解析一遍，缓存里绝不会出现签名与句柄不匹配的项。
    """
    manager = _manager_with("u1", "kb-a")
    manager.gate = asyncio.Event()

    first_task = asyncio.create_task(build_rag_middlewares("u1", settings, manager))
    await asyncio.sleep(0)  # 解析者卡在 kb-a 上，签名里只有 kb-a
    # 解析者干活期间，管理员新建了一个知识库 —— 等待者的签名与它不同了。
    manager.add("u1", _FakeRecord("kb-b", "B 库"), _FakeHandle("kb-b", "B 库", "u1"))
    second_task = asyncio.create_task(build_rag_middlewares("u1", settings, manager))
    await asyncio.sleep(0)  # 等待者进场，此刻解析者仍在解析

    manager.gate.set()
    first, second = await asyncio.gather(first_task, second_task)

    assert [kb.name for kb in second[0]._knowledge_bases] == ["政策库", "B 库"], (
        "等待者复用了签名不同的那份结果 —— 新建的知识库被静默丢掉了"
    )
    # kb-a 被解析了两次：解析者一次，等待者自己一次（它没有借用别人的句柄）。
    assert manager.resolved_ids.count("kb-a") == 2
    assert manager.resolved_ids.count("kb-b") == 1


async def test_a_cancelled_waiter_does_not_strand_the_others(
    settings: Settings,
) -> None:
    """等待者被取消，**不该**把「解析完成」这个信号本身也一起取消掉。

    ⚠️ 防的是什么：等待者 ``await`` 的是解析者登记的 ``Future``。直接
    ``await fut`` 时，取消等待者会**连那个 Future 一起取消**（这是 asyncio
    的既定语义：取消一个 Task 会取消它正在等的 Future）。于是解析者照常
    做完、却没人收到信号，**其他**等待者醒来时 ``await`` 一个已取消的
    Future，直接吃到 ``CancelledError`` —— 一次客户端断开会波及别的请求。

    真实现里等待者被取消是很平常的：用户在 SSE 流上关掉页面、请求超时、
    服务优雅停机。所以 ``shield`` 在这里不是装饰。
    """
    manager = _manager_with("u1", "kb-a")
    manager.gate = asyncio.Event()

    resolver = asyncio.create_task(build_rag_middlewares("u1", settings, manager))
    await asyncio.sleep(0)  # 解析者已登记，卡在闸门上
    doomed = asyncio.create_task(build_rag_middlewares("u1", settings, manager))
    await asyncio.sleep(0)  # 它在等解析者的信号
    doomed.cancel()
    with pytest.raises(asyncio.CancelledError):
        await doomed

    # 又被取消了一个等待者 —— 它必须仍然能等到解析者的信号。
    latecomer = asyncio.create_task(build_rag_middlewares("u1", settings, manager))
    await asyncio.sleep(0)

    manager.gate.set()
    resolved, survived = await asyncio.gather(resolver, latecomer)

    assert len(resolved) == 1
    assert len(survived) == 1, "前一个等待者被取消，把后来的等待者也带崩了"
    assert manager.get_calls == 1, "带崩之后它只能自己重解析一遍"
