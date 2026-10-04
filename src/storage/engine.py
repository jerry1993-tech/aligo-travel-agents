# -*- coding: utf-8 -*-
"""业务库引擎：与框架存储**完全独立**的一套 SQLAlchemy 连接。

文件职责：
    · 推导 ``create_async_engine`` 的参数（按方言分流，见
      :func:`storage_engine_kwargs`）；
    · 构造业务库引擎（:func:`build_business_engine`）；
    · 引导 ``business`` schema（:func:`ensure_business_schema`）；
    · 连通性探测（:func:`ping_business_engine`），供 ``/readyz`` 使用。

上下游依赖：
    - 上游：``src/server/app.py::create_root_app`` 在 lifespan 里构造并释放；
      ``tests/conftest.py`` 复用 :func:`storage_engine_kwargs`。
    - 下游：``sqlalchemy.ext.asyncio``。**不导入任何 agentscope**——
      这正是「业务库与框架存储解耦」在依赖图上的体现。

==============================================================================
为什么参数推导函数住在这里，而不是留在 app.py
==============================================================================
    它原本是 ``src/server/app.py`` 的私有函数 ``_storage_engine_kwargs``。
    搬过来的原因是：现在有**两个** engine 要用它 —— 框架的
    ``AsyncSQLAlchemyStorage`` 和本模块的业务 engine。留在 app.py 里，
    storage 层就得反过来 import server 层，依赖方向颠倒（storage 是被
    server 使用的下层）。搬完之后 ``app.py`` 仍然以原名再导出一次，
    见该文件顶部的 ``_storage_engine_kwargs = storage_engine_kwargs``——
    ``tests/conftest.py`` 直接 import 那个私有名，那是既有的测试契约。

==============================================================================
为什么业务库要单独一个 engine，而不是从框架的 storage 里掏
==============================================================================
    见 :mod:`src.storage` 的包文档字符串（框架那 13 张表与本项目业务表
    会撞名，且框架不允许传入自定义 metadata）。这里补一条**运维**上的理由：

    两个 engine 的失效特征完全不同。框架 storage 挂了 ⇒ 会话/消息读写失败，
    表现为「对话用不了」；业务 engine 挂了 ⇒ 订单/审批读写失败，
    表现为「查询用不了」。共用一个 engine 时，任一方的连接泄漏
    （拿连接忘了还）都会同时打爆两边，而排障时你分不清是谁漏的。
    分开之后，``pool_size`` 也是各自独立的额度，互不挤占。

==============================================================================
连接池算总账（与 config/base.yaml 的 db 段呼应）
==============================================================================
    单 engine 峰值 = ``pool_size + max_overflow`` = 10 + 20 = **30**。
    现在有两个 engine ⇒ 单进程峰值 **60**。若 ``workers`` 开到 4，
    就是 4 × 60 = 240 —— PostgreSQL 默认 ``max_connections=100``，
    **直接连不上**。

    这是保持 ``WORKERS=1`` 的又一条理由（第一条是调度器会重复触发，
    见 config/base.yaml）。扩 worker 前必须先算这个账，
    并把 pg 的 ``max_connections`` 与 pgbouncer 一起规划。

==============================================================================
三种「等数据库」的等待，各有各的闸门
==============================================================================
    「数据库慢」在用户那里是同一个现象（请求不返回），成因却有三层，
    而它们**互相不能替代** —— 只有一条闸门时，另外两层仍然无界：

      · **等一条空闲连接**（池满）—— ``pool_timeout``。
        数据库本身很快也可能卡在这里：慢查询堆积、连接泄漏、
        PG 侧 ``max_connections`` 打满。默认 30s 太长，本项目取
        ``db.pool_timeout_seconds``（默认 5s）。
      · **一条语句执行太久**（慢查询 / 锁等待）—— ``statement_timeout``。
        由 PostgreSQL **自己**执行，杀掉查询并把连接干净地还回池里。
        客户端同时设一个更大的 ``command_timeout`` 兜底
        （见 :data:`_COMMAND_TIMEOUT_MARGIN_SECONDS`）。
      · **连不上 / 没有回应**（网络黑洞、PG 被暂停）—— 同样由
        ``command_timeout`` 兜住：此时服务端的定时器在同一个进程里，
        帮不上忙。

    ⚠️ 这三条都只在 **PostgreSQL** 生效。sqlite（``make test`` 用的内存库）
    没有连接池也没有语句超时概念，强行传参会让引擎**建不出来** ——
    那正是 ``storage_engine_kwargs`` 按方言分流的原因。
"""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

logger = logging.getLogger(__name__)

#: 业务表所在的 schema。
#:
#: **是硬契约，不做成配置项。** 理由：schema 名会被写进 alembic 的
#: ``version_table_schema``、写进每个模型的 ``__table_args__``、
#: 写进运维的手工排查 SQL 里。把它做成 ``ALIGO__DB__SCHEMA`` 这样的可配置项，
#: 并不会带来任何实际好处（没有哪个环境需要换一个 schema 名），
#: 却制造出「配置改了、模型没改，运行时报 relation does not exist」这类
#: 只在启动后才暴露的故障。写死成常量，改名是一次全局搜索替换，编译器帮忙。
BUSINESS_SCHEMA = "business"

#: 客户端命令超时相对服务端 ``statement_timeout`` 的**余量**（秒）。
#:
#: ⚠️ 存在的唯一理由是让服务端**先**超时：PostgreSQL 杀掉查询后会返回一个
#: 明确的错误且连接保持可用；若客户端先放弃，那条查询还在 PG 侧跑着，
#: 而连接已经带着它回到池子里 —— 下一个拿到它的请求要么被拖慢，
#: 要么踩上 "another command is already in progress"。
#: 5s 是「一次本地往返 + PG 撤销查询」的量级，不需要精确：
#: 它只在服务端那道闸门**失灵**时才会被用到。
_COMMAND_TIMEOUT_MARGIN_SECONDS = 5.0


def storage_engine_kwargs(settings: Any) -> dict[str, Any]:
    """按数据库方言给出合适的 SQLAlchemy engine 参数。

    ⚠️ 为什么必须区分方言：``pool_size`` / ``max_overflow`` / ``pool_recycle``
    这三项**只对 QueuePool 有效**。而 ``sqlite+aiosqlite`` 默认用的是
    ``NullPool``（文件库）或 ``StaticPool``（内存库），把 QueuePool 的参数传给它
    会直接抛
    ``TypeError: Invalid argument(s) 'pool_size','max_overflow' sent to create_engine()``。

    测试环境（``make test``）用的正是内存 sqlite，因此这一分支不是「以防万一」，
    而是**测试能否跑起来**的前提。

    ⚠️ 与此相对，``hide_parameters`` 是**跨方言**的 Engine 级参数，两种方言
    都要设，因此它不是方言分支的一部分，而是在分支之前就写进 ``kwargs``。
    它的取值只有一个正确答案（``True``），理由见下方代码块里的长注释 ——
    一句话：``credentials.data`` 里装着运营者的真实 API key，
    而 SQLAlchemy 默认会把它渲染进异常文本。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `dict`: 传给 ``create_async_engine`` 的额外参数。
    """
    url = settings.db.url
    kwargs: dict[str, Any] = {
        "echo": settings.db.echo,
        # hide_parameters：**把绑定参数从异常文本与 echo 日志里抹掉**。
        #
        # 这不是「日志洁癖」，是本项目唯一一处「加密钥会自己跑进日志」的地方。
        # SQLAlchemy 默认把出错语句连同参数一起渲染进异常消息：
        #
        #     (asyncpg.exceptions.UniqueViolationError) ...
        #     [SQL: INSERT INTO credentials (id, user_id, data) VALUES ($1,$2,$3)]
        #     [parameters: ('aligo-system-model', 'aligo-system',
        #                   '{"api_key": "sk-...", ...}')]
        #
        # 而 ``credentials.data`` 里装的就是**运营者的真实 API key**
        # （见 ``src/llm/system_credential.py`` 的播种逻辑）。只要那一句
        # INSERT 撞上一次主键冲突/连接中断，整个 key 就会原样出现在
        # ``logger.exception`` 的栈里 —— 落进容器日志、日志采集系统、
        # 以及任何一次 ``make logs`` 的终端回滚缓冲里。
        #
        # 打开它之后同样的异常仍然会被抛出（类型、语句、SQLSTATE 都在），
        # 只是参数位置变成 ``(...)``，排障信息几乎不损失，而秘密不再外泄。
        #
        # ⚠️ 顺带管住 echo：``echo=true`` 时 SQLAlchemy 打印的是**渲染后**的
        # 语句，同样带参数。``echo`` 与 ``hide_parameters`` 是两套开关，
        # 只设 ``echo`` 不设它，「开了 echo 调试一下」就等于把 key 打到了
        # stdout。放在同一个 dict 里，是为了让这两项永远一起被注意到。
        "hide_parameters": True,
    }

    # ------------------------------------------------------------------
    # 池参数（QueuePool 系）—— PostgreSQL 与**文件型** sqlite 都吃这一套
    # ------------------------------------------------------------------
    pool_kwargs: dict[str, Any] = {
        "pool_size": settings.db.pool_size,
        "max_overflow": settings.db.max_overflow,
        # pool_recycle：在连接被中间设备静默掐断之前主动回收。
        # 取值方向是「回收周期 < 中间设备的空闲超时」才有效，
        # 详见 config/base.yaml 的 db.pool_recycle_seconds 注释。
        "pool_recycle": settings.db.pool_recycle_seconds,
        # pool_pre_ping：从池里取连接时先发一个轻量探测。
        # 它能把「取到了一条已死的连接」变成一次透明的重连，
        # 是 pool_recycle 的补充而不是替代 —— recycle 有周期性，
        # pre_ping 则覆盖了周期之间的那一段。
        "pool_pre_ping": True,
        # pool_timeout：**等一条空闲连接**的上界。
        # ⚠️ 它管的是与「语句慢」完全不同的一种等待：池被占满时
        # （慢查询堆积、连接泄漏、PG 侧 max_connections 打满），
        # 新请求会在这里排队。SQLAlchemy 的默认值是 30s ——
        # 于是「数据库其实很快、只是池满了」会表现成每个请求都卡半分钟。
        # 见 config/base.yaml 的 db.pool_timeout_seconds 注释。
        "pool_timeout": settings.db.pool_timeout_seconds,
    }

    if url.startswith("sqlite"):
        if ":memory:" in url or "mode=memory" in url:
            # 内存库必须用 StaticPool：默认的 NullPool 会为**每个连接**新建一个
            # 独立的 ``:memory:`` 数据库，于是「建表」和「查表」落在两个不同的库上，
            # 表现为「表明明建了却报 no such table」。
            # StaticPool 让所有连接复用同一个底层连接，内存库才真的是一份。
            # ⚠️ 也正因为是 StaticPool，这里的池参数**一个都不能传**
            # （它不接受 pool_size 等）—— 所以这一支提前 return 是对的。
            from sqlalchemy.pool import StaticPool

            kwargs["poolclass"] = StaticPool
            return kwargs

        # ⚠️ **文件型 sqlite 要照常吃池参数**，不能跟着内存库一起提前 return。
        # 这里曾经对所有 sqlite 一律 return，理由是「文件库默认 NullPool，
        # 传池参数会 TypeError」—— 那个理由在 SQLAlchemy 2.0 上**已经不成立**：
        # 实测 2.0.54 给文件型 ``sqlite+aiosqlite`` 的默认池是
        # ``AsyncAdaptedQueuePool``，而且**接受** pool_size / pool_timeout。
        # 于是那段代码的后果是：文件库部署下 ``ALIGO__DB__POOL_TIMEOUT_SECONDS=1.0``
        # 被静默忽略，实际仍在池满时排队 30s（默认值）—— 与这个配置项
        # 「消除过长等待」的存在意义恰好相反，而且没有任何告警。
        # ⚠️ 语句级超时那一段仍然跳过：sqlite 没有这个概念，也没有
        # connect_args 的位置（见下面按驱动分流的那一段）。
        kwargs.update(pool_kwargs)
        return kwargs

    kwargs.update(pool_kwargs)

    # ------------------------------------------------------------------
    # 语句级截止时间：**必须**服务端与客户端各设一道
    # ------------------------------------------------------------------
    # 两道都要，因为它们挡的是两种不同的故障：
    #   · 服务端的 statement_timeout 由 PostgreSQL 自己执行 —— 超时后
    #     PG **主动杀掉**那条查询并把连接干净地还回池里，报一个明确的
    #     "canceling statement due to statement timeout"。这是首选路径。
    #   · 客户端的 command_timeout 由 asyncpg 执行 —— 它挡的是
    #     「服务端根本没机会回答」的情况（网络黑洞、TCP 半开、PG 进程被
    #     SIGSTOP）。此时服务端的定时器也在同一个进程里，帮不上忙。
    #
    # ⚠️ 客户端的预算必须**大于**服务端，且要大出一个余量：反过来（客户端
    # 先超时）会让 asyncpg 在 PG 还没执行完杀掉动作时就放弃，
    # 连接带着一条**仍在运行**的查询回到池里 —— 下个请求拿到它，
    # 要么被那条查询拖慢，要么直接踩上 "another command is already in progress"。
    statement_timeout = settings.db.statement_timeout_seconds
    if make_url(url).drivername.endswith("+asyncpg"):
        kwargs["connect_args"] = {
            # asyncpg 的 ``command_timeout`` 单位是秒。
            "command_timeout": statement_timeout + _COMMAND_TIMEOUT_MARGIN_SECONDS,
            # ⚠️ ``timeout`` = **建连本身**的截止时间，与上面那个是两回事，
            # 少了它本节开头那句「连不上由 command_timeout 兜住」就是错的。
            # 不传的话 asyncpg 用自己默认的 **60s**（实测
            # ``inspect.signature(asyncpg.connect).parameters["timeout"].default == 60``），
            # 于是「PG 被防火墙 DROP / 进程被 SIGSTOP」这类**连不上**的故障
            # 实际要等 60s：``command_timeout`` 管的是**已建好的连接上跑语句**，
            # 建连那一刻还没有连接，它无从约束。
            # ⚠️ 它也**不在** ``pool_timeout`` 的覆盖范围内：QueuePool 只在等
            # **队列槽位**时受 pool_timeout 约束，新建连接发生在计时等待之外。
            # 所以池里没有空闲连接时（冷启动首个请求、PG 重启后），
            # 单次请求最坏是 60s，而不是本节承诺的 5s / 15s。
            "timeout": statement_timeout + _COMMAND_TIMEOUT_MARGIN_SECONDS,
            # ``server_settings`` 由 asyncpg 在建连时作为启动参数发给 PG，
            # 单位是**毫秒**（PostgreSQL 的约定）。
            "server_settings": {
                # ⚠️ 先 round 再兜底到 1：``int(x * 1000)`` 是**截断**，
                # 任何小于 1ms 的合法配置（schema 只要求 > 0）都会被截成
                # 字符串 "0"，而 PG 把 ``statement_timeout = 0`` 解释为
                # **不限制** —— 一个配置项就这样静默关掉了服务端闸门，
                # 正是 DBSettings 那句「不接受 0」要防的事。
                # 兜到 1ms 而不是报错：1ms 仍是一道**存在**的闸门，方向安全。
                "statement_timeout": str(
                    max(1, round(statement_timeout * 1000)),
                ),
            },
        }
    else:
        # ⚠️ 只给 asyncpg 传：``server_settings`` / ``command_timeout`` 是
        # asyncpg 专有参数，换成 psycopg / aiopg 会以「未知连接参数」直接
        # 抛在 ``create_async_engine`` 上（连引擎都建不出来）。
        # 不猜别的驱动的等价参数 —— 猜错是启动期崩溃，不猜只是少一层护栏，
        # 而少一层护栏会在日志里留下「没有语句超时」这个可查的事实。
        logger.warning(
            "数据库驱动是 %r（非 asyncpg），**未设置语句级超时**；"
            "慢查询将不受约束地占着连接。生产建议使用 postgresql+asyncpg。",
            make_url(url).drivername,
        )

    return kwargs


def business_schema_for(settings: Any) -> str | None:
    """给出该连接串下业务表应落的 schema 名。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `str | None`: PostgreSQL 系返回 :data:`BUSINESS_SCHEMA`；
        sqlite 等**不支持 schema** 的方言返回 ``None``。

    ⚠️ 为什么 sqlite 返回 ``None`` 而不是照样返回 ``"business"``：
        sqlite 没有 schema 概念（``ATTACH`` 的别名是另一回事，语义不同）。
        如果照样返回 ``"business"``，那么模型上的
        ``__table_args__ = {"schema": "business"}`` 会让 sqlite 生成
        ``CREATE TABLE business.orders``，直接语法错误 —— 于是
        **本地测试全挂，而生产 PostgreSQL 完全正常**。
        返回 ``None`` 是明确告诉调用方「这个库没有 schema 这一层，
        表就建在默认位置」，由调用方自己决定模型要不要带 schema。
    """
    try:
        backend = make_url(settings.db.url).get_backend_name()
    except Exception:  # noqa: BLE001 —— 连接串解析失败不该让调用方崩在探测上
        logger.warning("无法解析数据库连接串方言，按「无 schema」处理。")
        return None
    # asyncpg / psycopg 的 backend name 都是 ``postgresql``（驱动在 drivername 里）。
    return BUSINESS_SCHEMA if backend == "postgresql" else None


def build_business_engine(settings: Any) -> AsyncEngine:
    """构造业务库引擎。

    ⚠️ ``create_async_engine`` **不建立任何连接** —— 连接是惰性的，
    第一次执行语句时才真正拨号。这是「零密钥 / 数据库不可达时服务仍能启动」
    的前提：``/healthz`` 要能回答「进程活着」，而这件事不该依赖 PG 是否可达。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `AsyncEngine`: 业务库引擎。**调用方负责 dispose**（见
        ``src/server/app.py`` 的 lifespan 收尾）。
    """
    url = settings.db.url
    engine = create_async_engine(url, **storage_engine_kwargs(settings))
    logger.debug(
        "业务库引擎已构造：方言=%s，schema=%s（尚未建立连接）。",
        engine.dialect.name,
        business_schema_for(settings),
    )
    return engine


def _quote_ident(name: str) -> str:
    """校验并引用一个 SQL 标识符。

    ``CREATE SCHEMA`` 的 schema 名**不能**用绑定参数传（DDL 不支持），
    只能拼进 SQL 字符串。因此这里必须自己把关：只接受
    「Python 标识符」形态（字母/数字/下划线、不以数字开头）的名字。

    当前唯一的调用点传的是常量 :data:`BUSINESS_SCHEMA`，不存在注入风险；
    但把校验写在这里，是为了让**将来**有人把它改成配置项时，
    拼字符串这一步仍然是安全的 —— 安全边界要守在所有可变输入可能进入的地方，
    而不是守在当前恰好是常量的那个值上。

    Args:
        name (`str`): 待校验的标识符。

    Returns:
        `str`: 双引号包裹后的标识符。

    Raises:
        ValueError: 名字不是合法的标识符形态。
    """
    if not name.isidentifier():
        raise ValueError(f"非法的 SQL 标识符：{name!r}")
    return f'"{name}"'


async def ensure_business_schema(engine: AsyncEngine, schema: str | None) -> None:
    """确保业务 schema 存在（幂等）。

    Args:
        engine (`AsyncEngine`): 业务库引擎。
        schema (`str | None`): 目标 schema；为 ``None``（sqlite）时直接返回。

    ⚠️ 三条必须写下来的理由：

        1. **为什么是这里建，而不是交给 alembic 或 ``scripts/postgres/init``**
           postgres 官方镜像的 ``/docker-entrypoint-initdb.d/*.sh`` **只在数据卷
           为空时执行一次**。而本项目的命名卷 ``pg_data`` 是长期存在的 ——
           一个跑了半年的环境，加一个新的 init 脚本进去是**不会执行**的。
           把创建动作放在应用启动时，才真正幂等且与部署方式无关。

        2. **为什么不用 ``MetaData.create_all``**
           它**不会**发出 ``CREATE SCHEMA``（SQLAlchemy 的已知行为：
           假定 schema 已存在）。对着一张带 ``schema="business"`` 的表调
           ``create_all``，得到的是一句
           ``relation "business.orders" does not exist`` 的建表失败 ——
           错误信息指向表，真实原因却在 schema。

        3. **为什么用 IF NOT EXISTS 而不是先查 ``information_schema``**
           先查后建是两次往返，且中间存在竞态（两个 worker 同时查、同时建）。
           ``CREATE SCHEMA IF NOT EXISTS`` 是单条语句、由数据库保证原子性。
           本项目虽为单 worker，但测试可能并发跑多个 app 实例。

    Raises:
        ValueError: ``schema`` 不是合法标识符（见 :func:`_quote_ident`）。
        其他数据库异常原样抛出 —— 启动期建不出 schema 是**致命**的，
        静默吞掉只会让故障推迟到第一次写业务表时才爆。
    """
    if schema is None:
        return

    quoted = _quote_ident(schema)
    async with engine.begin() as conn:
        await conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {quoted}"))
    logger.debug("业务 schema 已就绪：%s", schema)


async def ping_business_engine(engine: AsyncEngine) -> None:
    """探测业务库连通性。

    Args:
        engine (`AsyncEngine`): 业务库引擎。

    Raises:
        Exception: 数据库不可达、认证失败、库不存在等，原样抛给调用方 ——
            ``/readyz`` 需要据此把状态标成未就绪，不该被这里吞成布尔值。
    """
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


__all__ = [
    "BUSINESS_SCHEMA",
    "build_business_engine",
    "business_schema_for",
    "ensure_business_schema",
    "ping_business_engine",
    "storage_engine_kwargs",
]
