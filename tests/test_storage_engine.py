# -*- coding: utf-8 -*-
"""业务库引擎的单元测试（``src/storage/engine.py``）。

覆盖的核心契约：
    1. **方言分流**：PostgreSQL 拿到连接池参数，sqlite 拿到 StaticPool
       且**绝不**拿到 pool_size —— 后者会让 ``make test`` 直接报
       ``TypeError: Invalid argument(s) 'pool_size','max_overflow'``。
    2. **schema 判定**：只有 PostgreSQL 才返回 ``business``；
       sqlite 必须返回 ``None``（否则模型上的 ``schema="business"``
       会让 sqlite 生成 ``CREATE TABLE business.xxx`` 而语法错误）。
    3. **建 schema 是幂等的**，且非 PostgreSQL 时是**彻底的 no-op**。
    4. **``storage_engine_kwargs`` 仍可从 ``src.server.app`` 导入** ——
       ``tests/conftest.py`` 依赖这个再导出的别名，删掉它会让整套用例
       在收集阶段就 ImportError。

为什么不连真实 PostgreSQL 来测：
    ``make test`` 的硬性约束是**不依赖 Docker**。因此这里只测
    「参数推导」这条纯函数逻辑（它的正确性不依赖数据库是否在跑），
    真实连通性由 ``tests/test_e2e_stream.py`` 与 ``make smoke`` 覆盖。
    把单测绑到 PG 上的代价是：CI 没有 PG 时整套用例红，
    而红的原因与本次改动无关 —— 那种红会训练人忽略测试结果。
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import StaticPool

from src.config import Settings
from src.storage.engine import (
    BUSINESS_SCHEMA,
    build_business_engine,
    business_schema_for,
    ensure_business_schema,
    ping_business_engine,
    storage_engine_kwargs,
)


class _FakeDB:
    """``settings.db`` 的最小替身（只带 :func:`storage_engine_kwargs` 读的字段）。"""

    def __init__(
        self,
        url: str,
        *,
        echo: bool = False,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_recycle_seconds: int = 1800,
        pool_timeout_seconds: float = 5.0,
        statement_timeout_seconds: float = 10.0,
    ) -> None:
        """记录各项连接参数。

        Args:
            url (`str`): 连接串。
            echo (`bool`): 是否回显 SQL。
            pool_size (`int`): 池常驻连接数。
            max_overflow (`int`): 池溢出上限。
            pool_recycle_seconds (`int`): 连接回收秒数。
            pool_timeout_seconds (`float`): 等空闲连接的秒数上限。
            statement_timeout_seconds (`float`): 单条语句的秒数上限。
        """
        self.url = url
        self.echo = echo
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle_seconds = pool_recycle_seconds
        self.pool_timeout_seconds = pool_timeout_seconds
        self.statement_timeout_seconds = statement_timeout_seconds


def _fake_settings(url: str, **overrides: Any) -> Any:
    """造一个只满足本模块需要的 settings 替身。

    Args:
        url (`str`): 数据库连接串。
        **overrides (`Any`): 透传给 :class:`_FakeDB` 的字段覆盖。

    Returns:
        `Any`: 带 ``db`` 属性的对象。
    """

    class _S:
        db = _FakeDB(url, **overrides)

    return _S()


def _fake_settings_with_timeout(
    url: str,
    *,
    statement_timeout_seconds: float,
) -> Any:
    """只改语句超时的 settings 替身（供参数化用例用）。

    Args:
        url (`str`): 数据库连接串。
        statement_timeout_seconds (`float`): 语句超时秒数。

    Returns:
        `Any`: 带 ``db`` 属性的对象。
    """
    return _fake_settings(
        url,
        statement_timeout_seconds=statement_timeout_seconds,
    )


# ==============================================================================
# 一、方言分流
# ==============================================================================
def test_sqlite_memory_uses_static_pool() -> None:
    """内存 sqlite 必须用 StaticPool，且**不带**任何 QueuePool 参数。

    ⚠️ 这条用例保护的是 ``make test`` 本身能不能跑起来。
    NullPool（默认）会为每个连接新建一个独立的 ``:memory:`` 数据库，
    症状是「框架明明建了表、查的时候却说 no such table」。
    """
    kwargs = storage_engine_kwargs(_fake_settings("sqlite+aiosqlite:///:memory:"))

    assert kwargs["poolclass"] is StaticPool
    # 这三项传给非 QueuePool 的引擎会直接抛 TypeError，必须是「压根没传」。
    assert "pool_size" not in kwargs
    assert "max_overflow" not in kwargs
    assert "pool_recycle" not in kwargs


def test_sqlite_file_does_not_get_static_pool() -> None:
    """文件型 sqlite **不**该用 StaticPool。

    StaticPool 只保留一个连接，对文件库而言等于把并发退化成串行；
    它只是内存库「必须共享同一份数据」的无奈之举，不该外溢。
    """
    kwargs = storage_engine_kwargs(_fake_settings("sqlite+aiosqlite:///tmp/x.db"))

    assert "poolclass" not in kwargs


def test_postgres_gets_full_pool_kwargs() -> None:
    """PostgreSQL 必须拿到完整的连接池参数与 pre_ping。"""
    kwargs = storage_engine_kwargs(
        _fake_settings("postgresql+asyncpg://u:p@db:5432/aligo"),
    )

    assert kwargs["pool_size"] == 10
    assert kwargs["max_overflow"] == 20
    assert kwargs["pool_recycle"] == 1800
    # pre_ping 是 pool_recycle 的补充而非替代：recycle 有周期性，
    # pre_ping 覆盖周期之间那段已被中间设备掐断的连接。
    assert kwargs["pool_pre_ping"] is True
    # 等空闲连接的上界。⚠️ 少了它，SQLAlchemy 的默认值 30s 会让
    # 「池满」表现成「每个请求卡半分钟」，与「数据库挂了」几乎分不开。
    assert kwargs["pool_timeout"] == 5.0


# ==============================================================================
# 一·B、语句级截止时间（三种「等数据库」的等待之一）
# ==============================================================================
def test_postgres_gets_both_server_and_client_statement_timeouts() -> None:
    """★★★ PostgreSQL 必须**同时**拿到服务端与客户端两道语句超时。

    ⚠️ 两道都要，因为它们挡的是不同故障：

      · **服务端**（``server_settings.statement_timeout``）挡慢查询与锁等待，
        由 PG 自己执行 —— 杀掉查询、返回明确错误、连接干净地还回池里。
        这是首选路径。
      · **客户端**（``command_timeout``）挡「服务端根本没机会回答」：
        网络黑洞、TCP 半开、PG 进程被 SIGSTOP。此时服务端定时器在同一个
        进程里，帮不上忙。

    只有一道时，另一种故障仍然是无界等待 —— 而两种故障在用户那里
    都是「请求不返回」。
    """
    kwargs = storage_engine_kwargs(
        _fake_settings("postgresql+asyncpg://u:p@db:5432/aligo"),
    )

    connect_args = kwargs["connect_args"]
    # 服务端单位是**毫秒**（PostgreSQL 的约定），客户端单位是秒（asyncpg）。
    # ⚠️ 这两个单位不能想当然 —— 传成秒的话，PG 会把它当成 10 毫秒，
    # 于是**每一条**语句都被立刻取消，症状是「数据库什么都没干就报超时」。
    assert connect_args["server_settings"]["statement_timeout"] == "10000"
    assert connect_args["command_timeout"] == 15.0


def test_the_client_timeout_is_strictly_larger_than_the_server_one() -> None:
    """★★ 客户端必须比服务端**晚**放弃，且留出余量。

    ⚠️ 方向反了会制造一个很难查的故障：asyncpg 先超时并放弃等待，
    而 PG 侧那条查询**还在跑**，连接已经带着它回到池里 ——
    下一个拿到这条连接的请求，要么被那条查询拖慢，要么直接踩上
    ``another command is already in progress``。而报错点在下一次查询上，
    与真正的原因隔了一整个连接生命周期。
    """
    for statement_timeout in (1.0, 10.0, 60.0):
        kwargs = storage_engine_kwargs(
            _fake_settings_with_timeout(
                "postgresql+asyncpg://u:p@db:5432/aligo",
                statement_timeout_seconds=statement_timeout,
            ),
        )

        client = kwargs["connect_args"]["command_timeout"]
        server_ms = int(kwargs["connect_args"]["server_settings"]["statement_timeout"])
        assert client > statement_timeout, (
            f"客户端超时 {client}s 没有大于服务端 {statement_timeout}s"
        )
        assert server_ms == int(statement_timeout * 1000)


def test_non_asyncpg_drivers_do_not_get_asyncpg_connect_args() -> None:
    """★★ 非 asyncpg 驱动**不能**拿到 ``server_settings`` / ``command_timeout``。

    ⚠️ 这两个是 asyncpg 专有参数。把它们传给别的驱动（psycopg / aiopg）
    会让 ``create_async_engine`` 直接抛「未知连接参数」—— 引擎**建不出来**，
    应用启动即崩。少一层护栏是「可查的降级」（下面那句 warning 就是线索），
    而启动崩溃是「不可用」，两者不能混为一谈。
    """
    kwargs = storage_engine_kwargs(
        _fake_settings("postgresql+psycopg://u:p@db:5432/aligo"),
    )

    assert "connect_args" not in kwargs
    # 连接池参数仍然要设 —— 它们与驱动无关。
    assert kwargs["pool_timeout"] == 5.0


def test_sqlite_gets_neither_pool_nor_statement_timeouts() -> None:
    """sqlite（``make test`` 的内存库）不能拿到任何 PG 专有参数。

    ⚠️ 这条保护的是**测试本身能不能跑起来**：``make test`` 全程用
    ``sqlite+aiosqlite``，一旦这里多传一个 ``connect_args``，
    整套用例会在建引擎时就红，而红的原因与本次改动无关。
    """
    kwargs = storage_engine_kwargs(_fake_settings("sqlite+aiosqlite:///:memory:"))

    assert "pool_timeout" not in kwargs
    assert "connect_args" not in kwargs


def test_file_sqlite_gets_the_pool_kwargs_too() -> None:
    """★★ 文件型 sqlite 必须**照常**拿到池参数（回归用例）。

    ⚠️ 这条挡的是一个「理由已经过期、代码却还在」的分支：早先所有
    sqlite 都提前 return，理由是「文件库默认 NullPool，传池参数会
    TypeError」。那个理由在 SQLAlchemy 2.0 上**不成立** —— 实测
    2.0.54 给文件型 ``sqlite+aiosqlite`` 的默认池是
    ``AsyncAdaptedQueuePool``，且接受 pool_size / pool_timeout。

    后果不是「少一层优化」而是**配置被静默忽略**：
    ``ALIGO__DB__POOL_TIMEOUT_SECONDS=1.0`` 不生效，池满时仍按默认
    30s 排队，且没有任何告警 —— 与这个配置项的存在意义恰好相反。

    ⚠️ 内存库那一支仍然**必须**提前 return（StaticPool 不接受池参数），
    所以这条只针对文件库，两者不能合并。
    """
    kwargs = storage_engine_kwargs(_fake_settings("sqlite+aiosqlite:///tmp/x.db"))

    assert kwargs["pool_timeout"] == 5.0
    assert kwargs["pool_size"] == 10
    assert kwargs["pool_pre_ping"] is True
    # StaticPool 是内存库专用的；文件库拿到它就等于把并发退化成串行。
    assert "poolclass" not in kwargs
    # 但 sqlite 没有语句超时的概念，异步驱动参数一个都不能传。
    assert "connect_args" not in kwargs


def test_asyncpg_connect_timeout_is_bounded() -> None:
    """★★★ 建连本身也要有截止时间（``connect_args["timeout"]``）。

    ⚠️ 它和 ``command_timeout`` 是**两回事**，很容易被合并成一个：
    ``command_timeout`` 管的是「已建好的连接上跑语句」，
    而**建连那一刻还没有连接**，它无从约束。不显式传 ``timeout`` 时
    asyncpg 用自己的默认值 **60s**，于是「PG 被防火墙 DROP / 进程被
    SIGSTOP」这类连不上的故障要等 60s —— 而文档承诺的是 15s。

    ⚠️ 它也**不在** ``pool_timeout`` 的覆盖范围内：QueuePool 只在等
    **队列槽位**时受 pool_timeout 约束，新建连接发生在计时等待之外。
    所以池里没有空闲连接时（冷启动首个请求、PG 重启后）单次请求
    最坏就是 60s —— 这两个配置项都救不了，只有这个参数能救。
    """
    kwargs = storage_engine_kwargs(
        _fake_settings("postgresql+asyncpg://u:p@db:5432/aligo"),
    )

    connect_timeout = kwargs["connect_args"]["timeout"]
    assert isinstance(connect_timeout, (int, float))
    assert connect_timeout <= 15.0, (
        f"建连超时是 {connect_timeout}s，太长了 —— asyncpg 的默认值是 60s，"
        "不显式传就等于没设。"
    )
    # 与客户端的语句超时同预算：两者都是「一次客户端等待」的上界。
    assert connect_timeout == kwargs["connect_args"]["command_timeout"]


def test_a_sub_millisecond_statement_timeout_is_not_truncated_to_zero() -> None:
    """★★★ 毫秒换算必须**四舍五入**，不能截断成 "0"（回归用例）。

    ⚠️ ``int(x * 1000)`` 是截断：任何小于 1ms 的合法配置（schema 只要求
    ``> 0``）都会变成字符串 ``"0"``，而 PostgreSQL 把
    ``statement_timeout = 0`` 解释为**不限制** —— 一个配置项就这样把
    服务端那道闸门静默关掉了，正是 ``DBSettings`` 里那句
    「不接受 0」要防的事。

    ⚠️ 兜底方向不能反：兜到 1ms 仍是「有闸门」，而放行成 0 是「无闸门」。
    宁可把用户设的 0.5ms 变成 1ms，也不能变成无限。
    """
    kwargs = storage_engine_kwargs(
        _fake_settings_with_timeout(
            "postgresql+asyncpg://u:p@db:5432/aligo",
            statement_timeout_seconds=0.0005,
        ),
    )

    server_value = kwargs["connect_args"]["server_settings"]["statement_timeout"]
    assert server_value != "0", (
        "小于 1ms 的语句超时被截断成了 \"0\"，而 PG 把 0 解释为**不限制** —— "
        "服务端闸门被静默关掉了。"
    )
    assert int(server_value) >= 1


def test_a_normal_statement_timeout_is_not_rounded_away() -> None:
    """★★ 常规取值必须原样换算成毫秒（别为了兜底把正常值也改了）。

    ⚠️ 与上一条是一对：只测「不变成 0」的话，一个把值写死成 1 的实现
    也能全绿。10.0s 必须得到 ``"10000"``。
    """
    kwargs = storage_engine_kwargs(
        _fake_settings_with_timeout(
            "postgresql+asyncpg://u:p@db:5432/aligo",
            statement_timeout_seconds=10.0,
        ),
    )

    assert kwargs["connect_args"]["server_settings"]["statement_timeout"] == "10000"


# ==============================================================================
# 二、schema 判定
# ==============================================================================
def test_postgres_maps_to_business_schema() -> None:
    """PostgreSQL 系返回 ``business``。"""
    assert business_schema_for(_fake_settings("postgresql+asyncpg://u:p@db:5432/x")) == BUSINESS_SCHEMA


def test_sqlite_has_no_schema() -> None:
    """sqlite 返回 ``None``。

    ⚠️ 这条不是「顺便」，而是**正确性**要求：若返回 ``"business"``，
    模型上的 ``__table_args__ = {"schema": "business"}`` 会让 sqlite 生成
    ``CREATE TABLE business.orders`` —— 语法错误。
    症状恰恰是最坏的一种：**本地测试全挂、生产 PostgreSQL 完全正常**。
    """
    assert business_schema_for(_fake_settings("sqlite+aiosqlite:///:memory:")) is None


def test_unparsable_url_is_treated_as_no_schema() -> None:
    """无法解析的连接串按「无 schema」处理，而不是抛异常。

    ``business_schema_for`` 只被用于「要不要建 schema」这个判断，
    为一个解析问题让调用方崩掉是不划算的 —— 真正连不上时，
    后面的 ping 会给出准确得多的错误。
    """
    assert business_schema_for(_fake_settings("这不是一个 URL")) is None


# ==============================================================================
# 三、构建与建 schema
# ==============================================================================
def test_build_business_engine_does_not_connect(settings: Settings) -> None:
    """构造引擎**不得**建立任何连接。

    这是「数据库不可达时服务仍能启动、``/healthz`` 仍能返回 200」的前提。
    连接是惰性的：``create_async_engine`` 只建池对象，第一次执行语句才拨号。
    单测里传入的是内存 sqlite，若哪天有人在构造里加了一句 ping，
    这条用例会因为「必须 await」而立刻暴露。
    """
    engine = build_business_engine(settings)

    assert isinstance(engine, AsyncEngine)
    assert engine.dialect.name == "sqlite"


@pytest.mark.asyncio
async def test_ensure_business_schema_is_noop_without_schema(settings: Settings) -> None:
    """``schema=None`` 时必须是彻底的 no-op —— **连连接都不该开**。

    断言方式是给 ``engine.connect`` 装一个探针，而不是检查池的计数：
    sqlite 内存库用的是 ``StaticPool``，它**没有** ``checkedout()`` /
    ``checkedin()`` 这些 QueuePool 才有的方法，用池计数写断言会直接
    ``AttributeError``（本用例的第一版就是这么挂的）。而且「连接计数器为 0」
    只是「没留下打开的连接」，与「压根没去连」是两回事 ——
    后者才是这里要保证的性质。
    """
    engine = build_business_engine(settings)

    try:
        # ⚠️ 只能在**类**上打补丁：``AsyncEngine`` 用了 ``__slots__``，
        # 实例属性是只读的，``engine.connect = spy`` 会抛
        # ``AttributeError: attribute 'connect' is read-only``。
        # 补丁的作用域仅限这个 with 块，且下面的 dispose 在块外 ——
        # dispose 走的是 ``pool.dispose()``，不经过 ``connect``，不受影响。
        with mock.patch.object(
            AsyncEngine,
            "connect",
            side_effect=AssertionError("schema 为 None 时不应建立任何连接"),
        ):
            await ensure_business_schema(engine, None)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ensure_business_schema_rejects_bad_identifier(settings: Settings) -> None:
    """非法标识符必须在拼 SQL **之前**被拦下。

    ``CREATE SCHEMA`` 的 schema 名不能用绑定参数传（DDL 不支持），
    只能拼字符串。当前调用点传的是常量，但校验写在拼串处，
    是为了让将来把它改成可配置项时，这里仍然是安全的。
    """
    engine = build_business_engine(settings)

    try:
        with pytest.raises(ValueError, match="非法的 SQL 标识符"):
            await ensure_business_schema(engine, "business; DROP TABLE users--")
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ping_works_against_sqlite(settings: Settings) -> None:
    """ping 在可用的库上不抛异常。"""
    engine = build_business_engine(settings)

    try:
        await ping_business_engine(engine)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_ping_raises_on_unreachable_database() -> None:
    """ping 在库不可达时**必须抛**，而不是返回 False。

    调用方（``/readyz`` 之类）需要拿到异常里的真实原因来写进报告；
    一个布尔值会把「认证失败」「库不存在」「网络不通」压成同一个
    ``ok=False``，而这三者的处置完全不同。
    """
    engine = build_business_engine(
        _fake_settings("postgresql+asyncpg://u:p@127.0.0.1:1/none"),
    )

    try:
        with pytest.raises(Exception):  # noqa: B017 —— 具体异常类型由驱动决定
            await ping_business_engine(engine)
    finally:
        await engine.dispose()


# ==============================================================================
# 四、向后兼容（这是测试侧的契约，不是「顺手加的」）
# ==============================================================================
def test_engine_kwargs_reexported_from_server_app() -> None:
    """``src.server.app._storage_engine_kwargs`` 必须仍然可导入。

    ⚠️ 为什么值得一条专门的用例：``tests/conftest.py`` 直接
    ``from src.server.app import _storage_engine_kwargs``。P2 把实现搬到了
    ``src.storage.engine``（理由见那里的模块文档字符串），只留了一个再导出的
    别名。将来有人清理「看起来没人用的私有函数」时删掉它，
    后果是**全套用例在收集阶段 ImportError** —— 而报错信息指向 conftest，
    与「删了哪个函数」相距甚远，排查方向会被完全带偏。
    """
    from src.server.app import _storage_engine_kwargs
    from src.storage.engine import storage_engine_kwargs as impl

    assert _storage_engine_kwargs is impl


# ==============================================================================
# 五、hide_parameters：不允许密钥出现在异常文本里
# ==============================================================================
# 为什么值得两条用例（而不是「顺手加个 kwarg 断言」）：
#
#   ``credentials.data`` 里装的是运营者的真实 API key（见
#   src/llm/system_credential.py）。播种那条 INSERT 一旦失败，
#   SQLAlchemy 默认会把语句**连同绑定参数**渲染进异常消息，
#   而 lifespan 会 ``logger.exception`` 它 —— key 落进容器日志、
#   日志采集系统，以及任何一次 ``make logs`` 的终端回滚缓冲。
#
#   下面第一条钉住配置（快、但只是代理指标）；第二条钉住**行为**
#   （慢一点，但它才是真正要保住的那件事）。只留第一条的话，
#   某个未来版本的 SQLAlchemy 改了 ``hide_parameters`` 的作用范围，
#   用例照样全绿。

#: 一个形似真实密钥的值。**不是**真密钥 —— 它只用于观察它会不会被打印。
_BIND_SECRET = "sk-not-a-real-key-only-for-this-test"


@pytest.mark.parametrize(
    "url",
    [
        "sqlite+aiosqlite:///:memory:",
        "sqlite+aiosqlite:///./tmp-test.db",
        "postgresql+asyncpg://aligo:pw@pg:5432/aligo",
    ],
)
def test_hide_parameters_is_on_for_every_dialect(url: str) -> None:
    """两种方言分支都要打开 ``hide_parameters``。

    ⚠️ 参数化三个 URL 是刻意的：``sqlite`` 分支在函数里**提前 return**，
    只在 postgres 那段加这个开关的话，测试环境（内存 sqlite）不生效 ——
    而测试环境恰恰是最常被 ``logger.exception`` 观察到的环境。
    """
    kwargs = storage_engine_kwargs(_fake_settings(url))
    assert kwargs["hide_parameters"] is True, f"{url} 没有关闭参数渲染"


@pytest.mark.asyncio
async def test_an_engine_built_from_these_kwargs_never_prints_bind_values() -> None:
    """★ 行为断言：用真实引擎跑一条会失败的语句，密钥不得出现在异常里。

    ⚠️ 用例**自带反向证据**（下面那段 ``without``）：先证明「不设这个开关
    时确实能观察到泄漏」。没有这一步，一条永远绿的用例比没有用例更糟 ——
    它会在开关被删掉之后继续通过，因为那时的失败模式变成了
    「SQLAlchemy 不渲染参数了」，而不是「我们关掉了它」。
    """
    url = "sqlite+aiosqlite:///:memory:"

    async def _failure_message(**engine_kwargs: Any) -> str:
        """用给定参数建引擎、跑一条注定失败的语句，返回异常文本。"""
        engine = create_async_engine(url, **engine_kwargs)
        try:
            async with engine.begin() as conn:
                await conn.execute(
                    text("INSERT INTO no_such_table (v) VALUES (:v)"),
                    {"v": _BIND_SECRET},
                )
        except Exception as exc:  # noqa: BLE001 —— 要的就是这条消息
            return str(exc)
        finally:
            await engine.dispose()
        raise AssertionError("那条语句本应失败")

    # 反向证据：默认（hide_parameters=False）时参数**确实**会被打印出来。
    without = await _failure_message()
    assert _BIND_SECRET in without, (
        "SQLAlchemy 不再把绑定参数渲染进异常文本了 —— 本用例的两条断言"
        "都失去了判别力，请重新设计（例如改用 echo 日志观察）。"
    )

    # 正题：用本项目真正的 kwargs 建引擎，同样的语句，参数必须被隐去。
    kwargs = storage_engine_kwargs(_fake_settings(url))
    with_hiding = await _failure_message(**kwargs)
    assert _BIND_SECRET not in with_hiding, (
        f"绑定参数泄漏进了异常文本：{with_hiding[:300]}"
    )
    # 隐去不等于丢信息：语句本身、SQLSTATE、提示都还在。
    assert "no_such_table" in with_hiding


@pytest.mark.asyncio
async def test_the_framework_storage_honours_our_engine_kwargs() -> None:
    """★ 链条上最容易断、又最没人看的一环：kwargs 真的传到引擎了吗？

    上面两条用例证明的是"我们算出的 kwargs 是对的"，而真正决定会不会漏的
    是**框架有没有把它交给 SQLAlchemy** —— ``src/server/app.py`` 那行
    ``AsyncSQLAlchemyStorage(..., engine_kwargs=_storage_engine_kwargs(settings))``
    一旦被谁改掉（比如换个参数名），前面所有断言照绿，而密钥开始进日志。

    ⚠️ 所以这条用例又是**自带反向证据**的：先用默认 kwargs 复现泄漏
    （证明这次构造真的能观察到差异），再用我们的 kwargs 证明它被隐去。
    只留后半句的话，一个"框架忽略了 engine_kwargs"的世界里它照样通过。

    ⚠️ 用**真实的** ``AsyncSQLAlchemyStorage`` + 真实的主键冲突来触发异常，
    而不是造一条假语句：这条路正是 ``ensure_system_credential`` 播种失败时
    会走的那条（见 src/llm/system_credential.py 的 Raises 段）。
    """
    from agentscope.app.storage._sql import AsyncSQLAlchemyStorage
    from agentscope.credential import DashScopeCredential

    url = "sqlite+aiosqlite:///:memory:"

    async def _conflict_message(**engine_kwargs: Any) -> str:
        """制造一次真实的凭据主键冲突，返回异常文本。"""
        storage = AsyncSQLAlchemyStorage(
            url=url,
            create_tables=True,
            auto_migrate=False,
            engine_kwargs=engine_kwargs,
        )
        async with storage:
            credential = DashScopeCredential(
                api_key=_BIND_SECRET,
                id="conflict-probe-id",
                name="泄漏探针",
            )
            # 属主 alice 先占住这个预设 id。
            await storage.upsert_credential("alice", credential)
            try:
                # 属主 bob 抢同一个 id —— 框架有意用 INSERT 撞主键（反跨租户覆盖）。
                await storage.upsert_credential("bob", credential)
            except Exception as exc:  # noqa: BLE001 —— 要的就是这条消息
                return str(exc)
            raise AssertionError("跨属主抢同一个预设 id 本应失败")

    # 反向证据：不设开关时，密钥**确实**会出现在异常文本里。
    without = await _conflict_message()
    assert _BIND_SECRET in without, (
        "在这个版本的框架/SQLAlchemy 上已观察不到泄漏 —— 本用例失去判别力，"
        "请重新设计（例如改为断言 echo 日志）。"
    )

    # 正题：走本项目真正的 kwargs 路径。
    with_hiding = await _conflict_message(**storage_engine_kwargs(_fake_settings(url)))
    assert _BIND_SECRET not in with_hiding, (
        f"框架没有把 engine_kwargs 交给引擎（或开关失效）：{with_hiding[:300]}"
    )


# ==============================================================================
# 六、契约：**每一个** engine 构造点都必须带 hide_parameters
# ==============================================================================
# 为什么需要这一节（它是审计发现的，不是设想出来的）：
#
#   上面那两条用例钉住的是 ``storage_engine_kwargs()`` 这个**函数**，
#   而真正决定「会不会漏」的是**每个调用点有没有用它**。
#   审计时确实找到了一个漏网的：``src/server/probes.py`` 的探针 engine ——
#   它刻意不复用那个函数（探针要 NullPool + 短命连接，与业务引擎的池化参数
#   正好相反），于是「加了新 engine 却忘了这个开关」这条路是**真实存在**的。
#
#   ``hide_parameters`` 的性质决定了这类遗漏不会被任何单点用例发现：
#   漏掉的那个 engine 只有在**恰好执行一条带绑定参数的语句并抛异常**时才显形。
#   那可能是一年后某个人在探针里加了个带租户参数的查询 —— 而那时
#   密钥会明文进 ``/readyz`` 响应体。所以这里用一条**静态**契约把它焊死：
#   扫源码里所有 engine 构造点，没有这道开关的当场红。
#
# 扫描范围**只有** src/ 与 scripts/，不含 tests/：
#   · src/ 是会被部署的东西，scripts/ 是离线跑但**接触真密钥**的东西；
#   · tests/ 里恰恰有一个 engine **必须不带**这个开关
#     （上面那段 ``without`` 反向证据，它靠泄漏才能证明自己有效），
#     把它纳进来只会逼着人给那条用例开豁免 —— 豁免一多，闸门就形同虚设。

#: 扫这两棵子树（相对仓库根）。
_CONTRACT_SCAN_DIRS = ("src", "scripts")

#: 视为「建 engine」的函数名。只认名字，不解析 import 来源：
#: 本仓库没有同名的自定义函数，而多一层解析只会多一处会腐化的假设。
_ENGINE_FACTORIES = frozenset({"create_async_engine", "create_engine"})


def _engine_call_sites() -> list[tuple[str, int, Any]]:
    """列出 ``src/`` 与 ``scripts/`` 下所有的 engine 构造调用。

    Returns:
        `list[tuple[str, int, Any]]`: ``(相对路径, 行号, ast.Call)``。
    """
    root = Path(__file__).resolve().parents[1]
    sites: list[tuple[str, int, Any]] = []
    for sub in _CONTRACT_SCAN_DIRS:
        for path in sorted((root / sub).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else func.attr
                    if isinstance(func, ast.Attribute)
                    else None
                )
                if name in _ENGINE_FACTORIES:
                    sites.append((str(path.relative_to(root)), node.lineno, node))
    return sites


def test_every_engine_call_site_disables_parameter_rendering() -> None:
    """★ 契约：没有一处 engine 构造能绕过 ``hide_parameters``。

    ⚠️ 两种放行方式，都有理由：

        · 显式写 ``hide_parameters=...``（如 ``src/server/probes.py``）
          —— 那个 engine 需要与业务引擎**不同**的池参数，不能复用共享函数；
        · ``**kwargs`` 解包（如 ``src/storage/engine.py`` 的
          ``**storage_engine_kwargs(settings)``）—— kwargs 由那个函数决定，
          而它本身有上面两条用例钉着。

    ⚠️ 用例**自带体检**（下面的 ``assert sites``）：如果扫描逻辑哪天因为
    目录改名而一个调用点都找不到，它会以「没找到」的方式失败，
    而不是心满意足地通过一个空集合。
    """
    sites = _engine_call_sites()
    assert sites, (
        f"在 {_CONTRACT_SCAN_DIRS} 下一个 engine 构造点都没扫到 —— "
        "扫描逻辑或目录结构变了，本契约实际上没在检查任何东西。"
    )

    offenders: list[str] = []
    for rel_path, lineno, call in sites:
        has_explicit = any(kw.arg == "hide_parameters" for kw in call.keywords)
        has_unpack = any(kw.arg is None for kw in call.keywords)
        if not (has_explicit or has_unpack):
            offenders.append(f"{rel_path}:{lineno}")

    assert not offenders, (
        "以下 engine 构造点既没写 hide_parameters、也没有解包共享 kwargs：\n"
        + "\n".join(f"    · {item}" for item in offenders)
        + "\n    绑定参数会被渲染进异常文本，而 credentials.data 里就是 API key。\n"
        "    改法二选一：复用 src/storage/engine.py 的 storage_engine_kwargs()，\n"
        "    或像 src/server/probes.py 那样显式写上 hide_parameters=True。"
    )
