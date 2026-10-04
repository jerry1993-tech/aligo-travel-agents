# -*- coding: utf-8 -*-
"""画像的 SQL 实现（``src/memory/repository.py``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    这一层的三道坎，每一道都**只在生产里**才发作：

      1. **并发丢更新**。两个请求同时改同一个人的画像（一个改座位、
         一个改成本中心），"读出来→改→写回去"的实现会让后写的那个
         把前一个的改动整份覆盖掉。单线程测试**永远**是绿的。

      2. **旧记录读不出来**。库里那条 JSON 是上个版本的应用写的，
         缺今天新增的字段。用 ``model_construct`` 的实现会造出一个
         缺属性的对象，错误推迟到第一次访问该字段时才炸 ——
         离这里十万八千里。

      3. **引擎还没就绪**。业务库引擎在 lifespan 里创建，而画像仓储在
         ``create_app`` 之前装配。中间那段时间调 ``get`` 必须**响亮地**
         失败，而不是静默退化成「这个用户没有画像」。

    测试库是 **sqlite 内存库**（与 ``make test`` 同一条路），
    所以这里同时验证了「sqlite 分支不写 ``FOR UPDATE``」——
    那个子句一加上，本文件当场全红。
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import StaticPool

from src.memory.profile import ProfilePatch, ProfileRepository, TravelerProfile
from src.memory.repository import (
    SqlProfileRepository,
    ensure_profile_table,
    profile_table,
)

#: 与 ``tests/conftest.py`` 的 ``TEST_DB_URL`` 同一条串。
#:
#: ⚠️ 用内存 sqlite 而不是 PostgreSQL：``make test`` 必须在没有 Docker 的
#: 机器上跑绿（README 的硬性约定之一）。``StaticPool`` 是必须的 ——
#: 内存库默认每个连接一份**独立**的库，不共用连接的话
#: 「建完表就查不到了」。
#:
#: 这里**不**断言 PostgreSQL 分支（``ON CONFLICT`` / ``FOR UPDATE``）：
#: 那需要真的起一个 PG，属于 ``make smoke`` 的范畴。本文件覆盖的是
#: 两条分支**共用**的那部分语义。
DB_URL = "sqlite+aiosqlite:///:memory:"


def make_engine() -> object:
    """造一个内存 sqlite 异步引擎。"""
    return create_async_engine(DB_URL, poolclass=StaticPool)


@pytest.fixture
def engine():
    """函数级引擎，每个用例一份全新的空库。"""
    eng = make_engine()
    try:
        yield eng
    finally:
        asyncio.run(eng.dispose())


@pytest.fixture
def repo(engine) -> SqlProfileRepository:
    """接好引擎的仓储（``schema=None``，sqlite 不支持 schema）。"""
    return SqlProfileRepository(engine, None)


# ==============================================================================
# 一、建表
# ==============================================================================
def test_ensure_profile_table_is_idempotent(engine) -> None:
    """★★ 建表跑两次不报错。

    ⚠️ 这不是「顺手测一下」：``ensure_profile_table`` 在**每次**
    应用启动时都会跑（lifespan 的进入段），而容器是会重启的。
    第二次启动就崩的话，服务再也起不来了 —— 而这条路径在
    开发机上只会在「第二次跑」时暴露，最容易被漏掉。
    """
    asyncio.run(ensure_profile_table(engine, None))
    asyncio.run(ensure_profile_table(engine, None))

    table = profile_table(None)
    assert table.name == "traveler_profiles"


def test_the_table_is_built_for_the_requested_schema() -> None:
    """``profile_table`` 每次按入参构造，schema 不被缓存。

    ⚠️ 若把 ``Table`` 做成模块级常量，schema 就被钉死在 import 那一刻 ——
    而 PostgreSQL 要 ``business``、sqlite 要 ``None``，两者跑在
    同一份代码里。缓存的症状是「本机测试全绿，容器里报
    relation "business.traveler_profiles" does not exist」。
    """
    assert profile_table("business").schema == "business"
    assert profile_table(None).schema is None


def test_the_repository_satisfies_the_protocol(repo: SqlProfileRepository) -> None:
    """SQL 实现与内存实现满足**同一个**协议。"""
    assert isinstance(repo, ProfileRepository)


# ==============================================================================
# 二、读
# ==============================================================================
def test_get_on_a_missing_row_returns_none(repo: SqlProfileRepository, engine) -> None:
    """没有记录时返回 None（不是空画像）。"""
    asyncio.run(ensure_profile_table(engine, None))

    assert asyncio.run(repo.get("nobody")) is None


def test_a_record_from_an_older_version_still_reads(repo, engine) -> None:
    """★★★ 缺字段的旧记录要能被**补默认值**读出来。

    ⚠️ 这是「一个 JSON 列」这笔交易能否兑现的关键。库里那条记录是
    上个版本的应用写的，只有 ``user_id`` 与 ``seat_preference`` ——
    今天新增的九个字段一个都不在。用 ``model_construct`` 的实现会造出
    一个**缺属性**的对象，不报错，直到某段代码访问
    ``profile.cost_center`` 时才 ``AttributeError``。

    ⚠️ 所以断言的是**字段真的存在且是默认值**，而不是「没抛异常」——
    后者对上面那个 bug 完全无效。
    """
    asyncio.run(ensure_profile_table(engine, None))
    asyncio.run(_raw_insert(engine, "u-old", {"user_id": "u-old", "seat_preference": "靠窗"}))

    loaded = asyncio.run(repo.get("u-old"))

    assert loaded is not None
    assert loaded.seat_preference == "靠窗"
    assert loaded.cost_center is None, "旧记录缺的新字段应当被补成默认值"
    assert loaded.preferred_airlines == []


def test_a_record_with_an_unknown_field_still_reads(repo, engine) -> None:
    """★ 反过来：记录里有**今天不存在**的字段（回滚了版本）也要能读。

    ⚠️ 回滚一次不该让整个画像读不出来。pydantic 默认忽略多余字段，
    这条用例把这个默认行为**钉住** —— 因为 ``extra="forbid"`` 是个
    看起来很合理、改起来只需要一个字的改动，而它的后果是
    「回滚后所有老用户读画像都 500」。
    """
    asyncio.run(ensure_profile_table(engine, None))
    asyncio.run(
        _raw_insert(
            engine,
            "u-new",
            {"user_id": "u-new", "seat_preference": "靠窗", "favourite_colour": "蓝"},
        ),
    )

    loaded = asyncio.run(repo.get("u-new"))

    assert loaded is not None and loaded.seat_preference == "靠窗"


# ==============================================================================
# 三、写
# ==============================================================================
def test_upsert_inserts_then_updates(repo, engine) -> None:
    """``upsert`` 第一次是插入，第二次是**覆盖**，且不会留下两行。"""
    asyncio.run(ensure_profile_table(engine, None))
    asyncio.run(repo.upsert(TravelerProfile(user_id="u1", seat_preference="靠窗")))
    asyncio.run(repo.upsert(TravelerProfile(user_id="u1", cost_center="CC-2")))

    loaded = asyncio.run(repo.get("u1"))

    assert loaded is not None
    assert loaded.cost_center == "CC-2"
    assert loaded.seat_preference is None, "upsert 是整体覆盖"
    assert asyncio.run(_row_count(engine)) == 1, "主键冲突被写成了两行"


def test_merge_creates_a_profile_for_a_brand_new_user(repo, engine) -> None:
    """新用户的第一次 ``merge`` 直接建一份，不报错。"""
    asyncio.run(ensure_profile_table(engine, None))

    merged = asyncio.run(repo.merge("u-new", ProfilePatch(seat_preference="靠窗")))

    assert merged.seat_preference == "靠窗"
    assert merged.user_id == "u-new"


def test_merge_keeps_the_fields_the_patch_does_not_name(repo, engine) -> None:
    """★★★ 部分更新：patch 没提的字段必须**原样留着**。

    ⚠️ 在 SQL 实现里这条比内存实现更容易写错，因为写回的是**整份 JSON**：
    只要 ``patch.apply`` 的对象是「从库里读出来的那一份」，
    没提的字段就自然留着；若图省事用了 ``TravelerProfile(user_id=...)``
    （一份全新的空画像）再套 patch，用户的座位偏好在更新成本中心时
    就静默消失了 —— 而 SQL 本身完全正确，没有任何报错。
    """
    asyncio.run(ensure_profile_table(engine, None))
    asyncio.run(
        repo.merge("u1", ProfilePatch(seat_preference="靠窗", preferred_airlines=["国航"])),
    )

    merged = asyncio.run(repo.merge("u1", ProfilePatch(cost_center="CC-1")))

    assert merged.cost_center == "CC-1"
    assert merged.seat_preference == "靠窗", "没提的字段被清掉了！"
    assert merged.preferred_airlines == ["国航"]

    reloaded = asyncio.run(repo.get("u1"))
    assert reloaded is not None and reloaded.seat_preference == "靠窗"


def test_merge_on_an_existing_row_round_trips_through_the_orm(repo, engine) -> None:
    """★★★ 「第二次更新同一个用户」这条路径必须真的走得通。

    ⚠️ 这是本文件里唯一一条**曾经真的挂过**的用例，所以单独立一条。
    首次写入走 INSERT 分支、压根不读库；只有第二次更新才走
    「读出来 → 改 → 写回去」，而那里读出来的东西会被直接喂给
    ``TravelerProfile.model_validate``。若读取函数返回的是 SQLAlchemy 的
    ``Row`` 而不是载荷 ``dict``，pydantic 会抛
    ``Input should be a valid dictionary or instance of TravelerProfile``
    —— 而它的触发条件是「同一个人被更新两次」，恰好是**最常见的**
    用法，却不在任何一条只写一次的用例的覆盖范围内。
    """
    asyncio.run(ensure_profile_table(engine, None))
    asyncio.run(repo.merge("u1", ProfilePatch(seat_preference="靠窗")))

    second = asyncio.run(repo.merge("u1", ProfilePatch(cost_center="CC-1")))
    third = asyncio.run(repo.merge("u1", ProfilePatch(default_approver="zhang")))

    assert (second.seat_preference, second.cost_center) == ("靠窗", "CC-1")
    assert third.default_approver == "zhang"


def test_merge_accumulates_across_calls(repo, engine) -> None:
    """连续多次 ``merge`` 是累积的（每次都基于库里当前那一份）。"""
    asyncio.run(ensure_profile_table(engine, None))

    asyncio.run(repo.merge("u1", ProfilePatch(cost_center="CC-1")))
    asyncio.run(repo.merge("u1", ProfilePatch(seat_preference="过道")))
    asyncio.run(repo.merge("u1", ProfilePatch(default_approver="zhang")))

    loaded = asyncio.run(repo.get("u1"))

    assert loaded is not None
    assert (loaded.cost_center, loaded.seat_preference, loaded.default_approver) == (
        "CC-1",
        "过道",
        "zhang",
    )


def test_two_users_never_see_each_other(repo, engine) -> None:
    """不同 ``user_id`` 的行完全隔离。"""
    asyncio.run(ensure_profile_table(engine, None))
    asyncio.run(repo.merge("u1", ProfilePatch(seat_preference="靠窗")))
    asyncio.run(repo.merge("u2", ProfilePatch(seat_preference="过道")))

    first = asyncio.run(repo.get("u1"))
    second = asyncio.run(repo.get("u2"))

    assert first is not None and first.seat_preference == "靠窗"
    assert second is not None and second.seat_preference == "过道"


def test_an_unsupported_dialect_is_refused_loudly(repo) -> None:
    """★★ 未知方言上 ``upsert`` 直接拒绝，不退化成一个非原子实现。

    ⚠️ 「先删后插」在所有库上都能跑通，所以它看起来是个安全的兜底 ——
    但它**不是**原子的：两次写之间有一个窗口，此时并发读会看到
    「这个用户没有画像」。与其在一个从没被验证过的方言上悄悄丢掉
    并发保证，不如在配置阶段就响亮地失败。
    """

    class FakeDialect:
        name = "mysql"

    class FakeEngine:
        dialect = FakeDialect()

    broken = SqlProfileRepository(FakeEngine(), None)  # type: ignore[arg-type]

    with pytest.raises(NotImplementedError):
        asyncio.run(broken.upsert(TravelerProfile(user_id="u1")))


# ==============================================================================
# 四、引擎的两种形态
# ==============================================================================
def test_the_callable_engine_form_works(engine) -> None:
    """★ 传「取引擎的函数」与传引擎本身，行为完全一致。

    ⚠️ 这个形态**必须**支持，理由是一条真实的时序约束：
    业务库引擎在 lifespan 的进入段才创建，而画像仓储在
    ``create_app`` 之前就要装配好（中间件工厂只被读走一次）。
    用取函数把「哪台引擎」推迟到第一次真正查库那一刻，
    装配期就拿不到引擎的问题就不存在了。
    """
    asyncio.run(ensure_profile_table(engine, None))
    lazy = SqlProfileRepository(lambda: engine, None)

    asyncio.run(lazy.merge("u1", ProfilePatch(seat_preference="靠窗")))

    loaded = asyncio.run(lazy.get("u1"))
    assert loaded is not None and loaded.seat_preference == "靠窗"


def test_the_callable_engine_form_follows_a_rebuilt_engine() -> None:
    """★★ 引擎被重建后，取函数自然指向**新的**那台。

    ⚠️ 这条挡的是「第一次取到就缓存起来」的实现。应用被重建
    （测试里反复进出 lifespan、或进程内重建 app）之后，缓存的
    那台引擎已经 dispose 了，而症状是「第一个用例好、后面全挂」。
    """
    first, second = make_engine(), make_engine()
    current = {"engine": first}
    repo = SqlProfileRepository(lambda: current["engine"], None)

    asyncio.run(ensure_profile_table(first, None))
    asyncio.run(repo.merge("u1", ProfilePatch(seat_preference="靠窗")))

    # 换掉引擎（模拟应用重建）；新库是空的，所以 u1 应当查不到。
    current["engine"] = second
    asyncio.run(ensure_profile_table(second, None))

    assert asyncio.run(repo.get("u1")) is None, "仓储缓存了旧引擎"

    asyncio.run(first.dispose())
    asyncio.run(second.dispose())


def test_using_the_repository_before_the_engine_exists_fails_loudly() -> None:
    """★★★ 引擎还没就绪时**抛**，不静默退化成内存实现。

    ⚠️ 这是本文件里后果最重的一条。业务库引擎在 lifespan 里创建，
    而画像仓储在 ``create_app`` 之前装配 —— 中间确实存在一段
    「能拿到仓储对象、但它还不该被用」的窗口。

    若这里退化成「返回 None / 一份内存画像」，症状是
    **「写进去的画像重启就没了」**：没有异常、没有日志，
    用户只会觉得助手总是记不住他。宁可 500。
    """
    repo = SqlProfileRepository(lambda: None, None)

    with pytest.raises(RuntimeError):
        asyncio.run(repo.get("u1"))

    with pytest.raises(RuntimeError):
        asyncio.run(repo.merge("u1", ProfilePatch(seat_preference="靠窗")))


# ==============================================================================
# 辅助
# ==============================================================================
async def _raw_insert(engine: object, user_id: str, payload: dict) -> None:
    """绕过仓储直接写一行 —— 模拟「上个版本的应用写下的记录」。"""
    table = profile_table(None)
    async with engine.begin() as conn:  # type: ignore[attr-defined]
        await conn.execute(table.insert().values(user_id=user_id, data=payload))


async def _row_count(engine: object) -> int:
    """数一下表里有几行。"""
    from sqlalchemy import func, select

    table = profile_table(None)
    async with engine.connect() as conn:  # type: ignore[attr-defined]
        return int((await conn.execute(select(func.count()).select_from(table))).scalar_one())
