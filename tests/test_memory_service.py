# -*- coding: utf-8 -*-
"""长期记忆门面（``src/memory/service.py``）与解析器（``resolver.py``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    门面的全部价值在**降级**上。两半依赖（Postgres 与 Milvus+向量模型）
    是两个独立的故障域，而它们各自的失败都**不能**中断对话：

      · 画像库连不上 ⇒ 本轮没有画像，但订票照常；
      · 向量库连不上 ⇒ 没有语义召回，但画像字段照常。

    最容易写错的实现是「一个 try 包住两半」—— 那样一边挂了会连带
    另一边也不可用，而这在开发机上永远看不出来（两个依赖都活着）。

    解析器那一半防的是另一件事：``user_id`` 必须来自**装配时捕获**的
    参数，而不是运行时从 agent 上猜。猜出来的东西在单会话测试里
    永远是对的，只在多租户下串号。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.config import Settings
from src.memory.profile import InMemoryProfileRepository, ProfilePatch
from src.memory.resolver import last_user_text, make_memory_resolver
from src.memory.service import TravelerMemory
from src.orchestration.context import PromptContext

# ⚠️ 用绝对的 ``tests.xxx`` 而不是 ``from .xxx``：``tests/`` 没有
# ``__init__.py``（pytest.ini 的 ``pythonpath = .`` 让它以命名空间包形式可见），
# 相对导入在收集阶段会直接 ImportError。同目录的 ``test_evaluation_judge.py``
# 等文件用的也是这种写法。
from tests.test_memory_semantic import FakeEmbedding, FakeVectorStore


def _semantic(settings: Settings, store: FakeVectorStore) -> Any:
    """造一个接了假向量库的 ``SemanticMemory``。"""
    from src.memory.semantic import SemanticMemory

    return SemanticMemory(
        vector_store=store,
        embedding_model=FakeEmbedding(),
        settings=settings,
    )


@dataclass
class FakeMessage:
    """最小可用的 ``Msg`` 替身。"""

    role: str
    text: str

    def get_text_content(self) -> str:
        """返回文本内容。"""
        return self.text


@dataclass
class FakeAgent:
    """最小可用的 agent 替身：只需要 ``state.context``。"""

    context: list[Any] = field(default_factory=list)

    @property
    def state(self) -> Any:
        """伪装出框架的 ``agent.state``。"""
        return type("State", (), {"context": self.context, "middle_context": {}})()


# ==============================================================================
# 一、门面：降级
# ==============================================================================
def test_a_disabled_memory_renders_nothing(settings: Settings) -> None:
    """总开关关掉时，读路径返回空，且**不抛**。

    ⚠️ 关掉 `ALIGO__MEMORY__ENABLED` 是运维的一个合法选择
    （比如画像库还没建好）。它不该让对话报错，只该让画像段落消失。
    """
    disabled = settings.model_copy(
        update={"memory": settings.memory.model_copy(update={"enabled": False})},
    )
    memory = TravelerMemory(disabled, repository=InMemoryProfileRepository())

    assert asyncio.run(memory.render_prompt_section("u1", "你好")) == ""


def test_a_broken_profile_repository_does_not_break_the_semantic_half(
    settings: Settings,
) -> None:
    """★★★ 画像库挂了，语义召回**照常**工作。

    ⚠️ 这条挡的是「一个 try 包住两半」的实现。那样写的话，
    PostgreSQL 抖一下会连带让 Milvus 那半也不可用 —— 而这两个依赖
    毫无关系，一个挂了不该拖垮另一个。开发机上两个都活着，
    这个 bug 永远看不出来。
    """

    class BrokenRepository:
        async def get(self, user_id: str) -> Any:
            raise RuntimeError("Postgres 连不上（模拟）")

        async def upsert(self, profile: Any) -> Any:
            raise RuntimeError("Postgres 连不上（模拟）")

        async def merge(self, user_id: str, patch: Any) -> Any:
            raise RuntimeError("Postgres 连不上（模拟）")

    store = FakeVectorStore()
    memory = TravelerMemory(
        settings,
        repository=BrokenRepository(),
        semantic=_semantic(settings, store),
    )
    asyncio.run(memory.remember("u1", "我一般坐靠窗"))

    rendered = asyncio.run(memory.render_prompt_section("u1", "我一般坐靠窗"))

    assert "我一般坐靠窗" in rendered, (
        "画像库挂了之后，语义那半也跟着不可用了 —— 两个 try 被合成一个了？"
    )


def test_a_broken_semantic_half_does_not_break_the_profile(
    settings: Settings,
) -> None:
    """★★★ 反过来也成立：向量库挂了，画像字段照常注入。

    ⚠️ 这正是 P4 验收里「Milvus 不可用不得拖垮服务」在**请求路径**上的
    同一条原则 —— 不只是 ``/readyz`` 那一处。
    """
    store = FakeVectorStore()
    store.fail_with = RuntimeError("Milvus 连不上（模拟）")
    repository = InMemoryProfileRepository()
    asyncio.run(repository.merge("u1", ProfilePatch(preferred_cabin="BUSINESS")))

    memory = TravelerMemory(
        settings,
        repository=repository,
        semantic=_semantic(settings, store),
    )

    rendered = asyncio.run(memory.render_prompt_section("u1", "帮我订票"))

    assert "BUSINESS" in rendered, "向量库挂了之后画像也不注入了 —— 降级写错了"
    context = asyncio.run(memory.recall("u1", "帮我订票"))
    assert context.semantic_error, "语义那半的失败没有被记录下来"
    assert context.profile is not None, "结构化那半不该被语义的失败波及"


def test_everything_broken_still_renders_nothing_and_does_not_raise(
    settings: Settings,
) -> None:
    """两半全挂时返回空串，仍然不抛。

    ⚠️ 这是最坏情况，也是**必须**能撑住的一种：它就是
    「Postgres 与 Milvus 同时不可用」—— 而那时候正是最需要
    「服务别整体趴下」的时刻。
    """

    class BrokenRepository:
        async def get(self, user_id: str) -> Any:
            raise RuntimeError("Postgres 挂了")

        async def upsert(self, profile: Any) -> Any:
            raise RuntimeError("Postgres 挂了")

        async def merge(self, user_id: str, patch: Any) -> Any:
            raise RuntimeError("Postgres 挂了")

    store = FakeVectorStore()
    store.fail_with = RuntimeError("Milvus 挂了")
    memory = TravelerMemory(
        settings,
        repository=BrokenRepository(),
        semantic=_semantic(settings, store),
    )

    assert asyncio.run(memory.render_prompt_section("u1", "你好")) == ""


def test_a_memory_with_no_dependencies_at_all_still_works(settings: Settings) -> None:
    """两半都没配（``None``）时也能构造、也能读。

    ⚠️ 这不是「忘传参数」的兜底，而是一个合法部署形态：
    离线评测进程不需要画像，也不该因为有 None 就崩掉。
    """
    memory = TravelerMemory(settings)

    assert memory.has_repository is False
    assert memory.has_semantic is False
    assert asyncio.run(memory.render_prompt_section("u1", "你好")) == ""


# ==============================================================================
# 二、门面：读写的不对称
# ==============================================================================
def test_writing_without_a_semantic_half_raises(settings: Settings) -> None:
    """没有语义能力时 ``remember`` 抛 ``RuntimeError``，而不是静默丢弃。

    ⚠️ 静默丢弃是这个接口最糟的实现：用户说了「记住这个」，
    界面上一切正常，而他下周才发现助手没记住。
    """
    memory = TravelerMemory(settings, repository=InMemoryProfileRepository())

    with pytest.raises(RuntimeError):
        asyncio.run(memory.remember("u1", "记住我喜欢靠窗"))


def test_forgetting_without_a_semantic_half_raises(settings: Settings) -> None:
    """没有语义能力时 ``forget`` / ``forget_all`` 同样抛 ``RuntimeError``。

    ⚠️ 「忘记」失败比「记住」失败更糟：用户以为那条记录已经没了，
    而它仍然躺在下一次对话的召回窗口里 —— 一个「删了但还在」的
    隐私问题，而界面上一切正常。
    """
    memory = TravelerMemory(settings, repository=InMemoryProfileRepository())

    with pytest.raises(RuntimeError):
        asyncio.run(memory.forget("u1", "记住我喜欢靠窗"))
    with pytest.raises(RuntimeError):
        asyncio.run(memory.forget_all("u1"))


def test_forget_and_forget_all_reach_the_store(settings: Settings) -> None:
    """★★ ``forget`` 删一条、``forget_all`` 删光并**返回条数**。

    ⚠️ 断言的是「召回不到了」而不是「没抛异常」：一个只计算了
    note_id 却没真的 ``delete`` 的实现，在「不抛异常」的判据下
    照样全绿 —— 而那正是删除类接口最典型的假成功。

    ⚠️ 条数必须真的数出来（这里 2 条）：调用方（
    ``DELETE /api/v1/memory/notes?all=true``）要能区分
    「清空了 2 条」与「本来就没有任何记录」。
    """
    store = FakeVectorStore()
    memory = TravelerMemory(
        settings,
        repository=InMemoryProfileRepository(),
        semantic=_semantic(settings, store),
    )
    for text in ("我一般坐靠窗", "我不吃辣"):
        asyncio.run(memory.remember("u1", text))

    asyncio.run(memory.forget("u1", "我不吃辣"))
    left = asyncio.run(memory.recall("u1", "坐靠窗不吃辣"))

    assert [note.text for note in left.recall.notes] == ["我一般坐靠窗"]

    removed = asyncio.run(memory.forget_all("u1"))

    assert removed == 1, "forget_all 返回的条数与实际删掉的对不上"
    assert asyncio.run(memory.recall("u1", "坐靠窗")).recall.notes == []


def test_recall_forwards_top_k(settings: Settings) -> None:
    """★★ ``recall(top_k=...)`` 真的**传下去**（不是被本地切片冒充的）。

    ⚠️ 这个参数是给 ``GET /api/v1/memory/notes`` 用的。若门面把它
    丢掉、由调用方在本地 ``[:top_k]`` 切片，调用方要 10 条而配置是 3 条时
    只会拿到 3 条 —— 一个「参数看起来生效了，其实没有」的假象。

    ⚠️⚠️ 请求数必须**大于**配置的上限，否则这条用例证明不了任何事
    （对抗性审核 2026-10-03 的发现）：库里只有 3 条、配置 ``top_k=5`` 时，
    「真转发」与「本地切片」两种实现的结果**完全一样**（都是 3 条），
    用例会在真正要防的那个 bug 上照样变绿。这里把配置压到 2、
    存 3 条、请求 3 条 —— 本地切片最多只能给出 2 条，于是两者可分。
    """
    small = settings.model_copy(
        update={"memory": settings.memory.model_copy(update={"top_k": 2})},
    )
    memory = TravelerMemory(
        small,
        repository=InMemoryProfileRepository(),
        semantic=_semantic(small, FakeVectorStore()),
    )
    for text in ("我一般坐靠窗", "我不吃辣", "我喜欢住全季"):
        asyncio.run(memory.remember("u1", text))

    asked = asyncio.run(memory.recall("u1", "坐靠窗不吃辣全季", top_k=3))
    default = asyncio.run(memory.recall("u1", "坐靠窗不吃辣全季"))

    assert len(asked.recall.notes) == 3, (
        "要 3 条只拿到 2 条 —— top_k 没传到语义那半（被配置的上限截断了）"
    )
    assert len(default.recall.notes) == 2, (
        "不带 top_k 时应当取 settings.memory.top_k（这里刻意设成 2）"
    )


def test_update_profile_merges_rather_than_replaces(settings: Settings) -> None:
    """``update_profile`` 走的是 ``merge``（部分更新）。"""
    repository = InMemoryProfileRepository()
    memory = TravelerMemory(settings, repository=repository)
    asyncio.run(memory.update_profile("u1", ProfilePatch(seat_preference="靠窗")))

    merged = asyncio.run(memory.update_profile("u1", ProfilePatch(cost_center="CC-1")))

    assert merged.cost_center == "CC-1"
    assert merged.seat_preference == "靠窗"


def test_get_profile_does_not_swallow_errors(settings: Settings) -> None:
    """``get_profile`` **不吞**异常。

    ⚠️ 与 ``render_prompt_section`` 相反，这是刻意的：
    它是给 ``/api/v1/memory/profile`` 这类**显式**接口用的，
    调用方正在等一个答案。把它变成 ``None`` 等于把一次故障
    伪装成「这个用户没有画像」—— 一个会让人查错方向的谎。
    """

    class BrokenRepository:
        async def get(self, user_id: str) -> Any:
            raise RuntimeError("Postgres 挂了")

        async def upsert(self, profile: Any) -> Any:
            raise RuntimeError("Postgres 挂了")

        async def merge(self, user_id: str, patch: Any) -> Any:
            raise RuntimeError("Postgres 挂了")

    memory = TravelerMemory(settings, repository=BrokenRepository())

    with pytest.raises(RuntimeError):
        asyncio.run(memory.get_profile("u1"))


def test_describe_is_pure_configuration(settings: Settings) -> None:
    """``describe()`` 只报能力，不含用户数据、不做 I/O。

    ⚠️ 它会在 Milvus 挂着时被调用（放进 ``/readyz``），
    所以它自己绝不能碰 Milvus。判据是它不需要 await ——
    一个同步函数不可能等待网络。
    """
    store = FakeVectorStore()
    memory = TravelerMemory(
        settings,
        repository=InMemoryProfileRepository(),
        semantic=_semantic(settings, store),
    )

    described = memory.describe()

    assert described["structured"] is True
    assert described["semantic"] is True
    assert described["reme_enabled"] is False


# ==============================================================================
# 三、解析器
# ==============================================================================
def test_last_user_text_skips_the_trailing_assistant_placeholder() -> None:
    """★★★ 最后一条用户消息**不是**列表的最后一个元素。

    ⚠️ 框架会先往上下文末尾塞一条空的 assistant 占位消息，等着模型的
    输出填进去。所以 ``context[-1]`` 永远不是用户消息 ——
    「检查最后一个元素」这个看起来完全合理的写法会让召回**一次都不触发**，
    而症状（「记忆不生效」）看起来像是检索坏了，排查方向会整个跑偏。
    """
    agent = FakeAgent(
        context=[
            FakeMessage(role="system", text="你是差旅助手。"),
            FakeMessage(role="user", text="帮我订去北京的票"),
            FakeMessage(role="assistant", text=""),
        ],
    )

    assert last_user_text(agent) == "帮我订去北京的票"


def test_last_user_text_picks_the_most_recent_one() -> None:
    """多轮对话里取的是**最近**那句。"""
    agent = FakeAgent(
        context=[
            FakeMessage(role="user", text="第一句"),
            FakeMessage(role="assistant", text="好的"),
            FakeMessage(role="user", text="第二句"),
            FakeMessage(role="assistant", text=""),
        ],
    )

    assert last_user_text(agent) == "第二句"


def test_last_user_text_survives_a_broken_agent() -> None:
    """拿不到就返回空串，绝不抛。

    ⚠️ 它跑在 ``on_system_prompt`` 的主链路上，抛异常会让整轮回复失败 ——
    而「这一轮拿不到查询文本」的代价只是「不做语义召回」。
    """
    assert last_user_text(object()) == ""
    assert last_user_text(FakeAgent(context=[FakeMessage(role="user", text="")])) == ""


def test_the_resolver_fills_in_the_profile_summary(settings: Settings) -> None:
    """解析器把画像填进 ``profile_summary``，其余字段原样透传。"""
    repository = InMemoryProfileRepository()
    asyncio.run(repository.merge("u1", ProfilePatch(preferred_cabin="BUSINESS")))
    memory = TravelerMemory(settings, repository=repository)

    resolver = make_memory_resolver(
        lambda agent: PromptContext(stage=PromptContext().stage),
        memory,
        "u1",
    )
    agent = FakeAgent(context=[FakeMessage(role="user", text="订票")])

    context = asyncio.run(resolver(agent))

    assert "BUSINESS" in context.profile_summary


def test_the_resolver_uses_the_captured_user_id(settings: Settings) -> None:
    """★★★ ``user_id`` 来自**装配时捕获**的参数，不是从 agent 上猜的。

    ⚠️ 这是多租户隔离的支点。``AgentMiddlewareFactory`` 被框架调用时
    把 ``user_id`` 作为参数传进来（``app/_service/_chat.py:240-253``），
    那是唯一可信的来源。从 agent 上猜（名字、某个 context 字段）会得到
    「看起来能用、偶尔串到别人画像」的行为 —— 而串号是事故。
    """
    repository = InMemoryProfileRepository()
    asyncio.run(repository.merge("u1", ProfilePatch(preferred_cabin="BUSINESS")))
    asyncio.run(repository.merge("u2", ProfilePatch(preferred_cabin="ECONOMY")))
    memory = TravelerMemory(settings, repository=repository)

    resolver = make_memory_resolver(lambda agent: PromptContext(), memory, "u2")
    agent = FakeAgent(context=[FakeMessage(role="user", text="订票")])

    context = asyncio.run(resolver(agent))

    assert "ECONOMY" in context.profile_summary
    assert "BUSINESS" not in context.profile_summary, (
        "解析器拿到了别人的画像 —— user_id 是从 agent 上猜的？"
    )


def test_the_resolver_works_with_a_sync_base(settings: Settings) -> None:
    """★ 同步的原解析器仍然可用（P4 放宽 ``ContextResolver`` 后的兼容性）。

    ⚠️ 这条挡的是「无条件 ``await`` 解析结果」的实现 ——
    对一个普通对象 ``await`` 会抛 ``TypeError``，而
    「解析器是同步的」恰恰是**默认**情况（``default_resolver``）。
    """
    memory = TravelerMemory(settings, repository=InMemoryProfileRepository())

    resolver = make_memory_resolver(
        lambda agent: PromptContext(stage=PromptContext().stage),
        memory,
        "u1",
    )

    context = asyncio.run(resolver(FakeAgent()))

    assert isinstance(context, PromptContext)


def test_the_resolver_survives_a_broken_memory(settings: Settings) -> None:
    """记忆整个坏掉时，解析器仍然返回一个可用的 ``PromptContext``。"""

    class Exploding:
        async def render_prompt_section(self, *args: Any, **kwargs: Any) -> str:
            raise RuntimeError("记忆模块炸了")

    resolver = make_memory_resolver(
        lambda agent: PromptContext(),
        Exploding(),  # type: ignore[arg-type]
        "u1",
    )

    context = asyncio.run(resolver(FakeAgent()))

    assert isinstance(context, PromptContext)
