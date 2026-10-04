# -*- coding: utf-8 -*-
"""画像的 **PostgreSQL 实现** —— 让结构化画像活在业务库里。

═══ ⚠️ 为什么表结构是「一个 JSON 列」而不是「每个偏好一个字段」 ═══

直觉上应该 ``seat_preference TEXT`` / ``preferred_cabin TEXT`` … 一列一个字段。
本项目**没有**这么做，理由是三条具体的事实：

  1. **画像的字段会长**。P4 有 9 个，用户在用的过程中一定会有人要
     「记住我的护照有效期」「记住我要开发票的抬头」。一列一个字段意味着
     **每加一个偏好就要一次 DDL 迁移**，而迁移在灰度期间是最容易出事的东西
     （新旧两个版本的应用同时跑，一个认识新列、一个不认识）。

  2. **我们从来不按画像字段查询**。所有查询都是
     ``WHERE user_id = ?``（见 :meth:`SqlProfileRepository.get`）。
     没有任何一条 SQL 会写 ``WHERE preferred_cabin = 'BUSINESS'``。
     也就是说，拆成列能换来的东西（索引、约束、按字段聚合）我们**一样都不用**。

  3. **校验已经在 pydantic 里了**。``TravelerProfile`` 的 validator 是
     权威的（它还要被内存实现、被 API 层、被评测脚本共用）。
     在数据库里再表达一遍同样的约束，等于把同一条规则写两遍 ——
     而两份规则迟早会不一致，之后以哪份为准就成了一个没人能回答的问题。

代价必须说清楚：**数据库层对画像内容是没有任何约束的**。
一个绕过 pydantic 的写入（比如有人手工 ``INSERT``）能塞进任何东西，
而 :meth:`get` 会直接炸在 ``model_validate`` 上。这是这笔交易的
明码标价 —— 换来的是「加字段不用迁移」。

═══ ⚠️ 为什么它住在这里，而不是 ``src/storage/`` ═══

``src/storage/`` 放的是**业务实体**（订单、审批、行程）的持久化。
画像的 Protocol 定义在 :mod:`src.memory.profile`，模型也在那里；
把它拆到另一个包只会制造一条 ``src.memory ↔ src.storage`` 的
反向依赖，而依赖方向正是本项目一直在守的东西
（见 ``src/storage/engine.py`` 模块文档里那段搬迁史）。

⚠️ 但它**必须**用 ``business`` schema，与业务表同一套引擎、
同一个 ``ensure_business_schema``，不另起炉灶。
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from sqlalchemy import JSON, Column, DateTime, MetaData, String, Table, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine

from .profile import ProfilePatch, TravelerProfile

logger = logging.getLogger(__name__)


#: 画像表的元数据。
#:
#: ⚠️ **必须是本模块自己的 ``MetaData`` 实例**，不能借用框架的
#: ``_Base.metadata``（那是 ``_tables.py`` 里的私有全局单例）。理由与
#: ``src/storage/__init__.py`` 里那段完全相同：``create_all`` 是按
#: metadata 建表的，共用它意味着「框架升级新增的表」会与我们的表
#: 混在一次建表里，而回滚应用版本回滚不了 schema。
_METADATA = MetaData()

#: 画像列的类型。
#:
#: ⚠️ PostgreSQL 上用 **JSONB** 而不是 JSON：JSONB 以解析后的二进制存储，
#: 因此 ``data->>'cost_center'`` 这类取值不需要每次重新解析文本。
#: 今天我们不用它，但把列建成 JSONB 是**无法事后便宜地改**的决定
#: （要重写整张表），而代价只是写入时多一次解析。先选对的那个。
#:
#: ⚠️ ``with_variant`` 是为了 sqlite（``make test`` 用的内存库）——
#: 它没有 JSONB，直接建会报「unknown type」。
_PROFILE_JSON = JSON().with_variant(JSONB(), "postgresql")


def profile_table(schema: str | None) -> Table:
    """按 schema 构造画像表对象。

    ⚠️ 做成函数而不是模块级常量：``schema`` 必须能在运行时决定
    （PostgreSQL 是 ``business``、sqlite 是 ``None``，见
    :func:`src.storage.engine.business_schema_for`）。
    模块级常量会把「哪个方言」这件事钉死在 import 那一刻，
    而 SQLAlchemy 的 ``Table`` 一旦建好就不能换 schema。

    Args:
        schema (`str | None`): 目标 schema；``None`` 表示库不支持 schema。

    Returns:
        `Table`: 表对象（每次都新建，因此调用方不必担心被别处改过）。
    """
    return Table(
        "traveler_profiles",
        _METADATA,
        Column(
            "user_id",
            String(128),
            primary_key=True,
            comment="员工 id，与鉴权层注入的 X-User-ID 同源。",
        ),
        Column(
            "data",
            _PROFILE_JSON,
            nullable=False,
            comment="TravelerProfile 的 model_dump()，字段随代码演进。",
        ),
        Column(
            "updated_at",
            DateTime(timezone=True),
            server_default=func.now(),
            onupdate=func.now(),
            nullable=False,
            comment="最后一次写入时间，由数据库生成（不信任应用时钟）。",
        ),
        # ⚠️ ``metadata`` 注册表里可能已经有同名的表（第二次调用本函数时），
        # 用 ``extend_existing`` 避免 ``InvalidRequestError``。
        extend_existing=True,
        schema=schema,
        comment="差旅长期画像（结构化部分）。一条记录 = 一个人的全部偏好。",
    )


async def ensure_profile_table(engine: AsyncEngine, schema: str | None) -> None:
    """建画像表（幂等）。

    ⚠️ 用 ``CREATE TABLE IF NOT EXISTS``（``Table.create`` 默认
    ``checkfirst=True``）而不是「先查 information_schema 再建」——
    与 :func:`src.storage.engine.ensure_business_schema` 同一取舍：
    单条语句、由数据库保证原子性、没有查与建之间的竞态。

    ⚠️⚠️ **必须建单张表（``table.create``），不能建整个 metadata
    （``table.metadata.create_all``）。** 这两者在生产里看起来一模一样
    （进程只见过一个 schema），但 ``_METADATA`` 是**模块级**的：
    同一个进程里只要有人同时要过 ``business`` 与 ``None`` 两个 schema
    的表（测试就是这么干的，而且 ``tests/`` 全在一个进程里跑），
    metadata 里就会同时存在两张同名不同 schema 的表，于是
    ``create_all`` 会在一台 sqlite 上去建 ``business.traveler_profiles``
    —— 报 ``unknown database business``，而错误指向的却是那句
    ``CREATE TABLE``，跟「谁往 metadata 里加了表」看不出关系。

    ⚠️ **也不在这里做迁移。** 加字段走的是应用层的 ``model_validate``
    （老记录缺新字段时用默认值补上，见 :meth:`get`），
    所以这张表几乎不需要 DDL 迁移 —— 这正是「一个 JSON 列」的收益。
    真的需要改表结构时（比如加索引）再引入 alembic，
    而不是现在就为一张两列的表搭一套迁移框架。

    ⚠️ 它**不**建 schema 本身：PostgreSQL 上 ``business`` 由
    :func:`src.storage.engine.ensure_business_schema` 先建好
    （``create_all`` 会顺手 ``CREATE SCHEMA``，但我们不依赖那个副作用）。

    Args:
        engine (`AsyncEngine`): 业务库引擎。
        schema (`str | None`): 目标 schema。
    """
    table = profile_table(schema)
    async with engine.begin() as conn:
        await conn.run_sync(lambda sync_conn: table.create(sync_conn, checkfirst=True))
    logger.debug("画像表已就绪：%s", table.fullname)


class SqlProfileRepository:
    """画像的 PostgreSQL 实现。

    ⚠️ 它满足 :class:`~src.memory.profile.ProfileRepository` 协议，
    但**不继承**它 —— 协议是结构子类型，继承只会多一条 import。

    ═══ ⚠️ 为什么引擎可以是「一个取引擎的函数」 ═══

    ``engine`` 参数接受两种形态：一个 ``AsyncEngine``，或一个**无参可调用
    对象**（返回引擎或 None）。

    需要后者的原因是一条真实的时序约束：业务库引擎是在
    **应用 lifespan 的进入段**里创建、退出段里 dispose 的
    （``src/server/app.py``，那里写着为什么必须逐次进入新建 ——
    「测试里反复进出 lifespan 会共用同一个已 close 的引擎」）。
    而长期记忆必须在 ``create_app`` **之前**装配（中间件工厂只在
    那一次被读走）。两者一个在装配期、一个在运行期，中间隔着
    ``create_app`` 本身 —— 装配期根本拿不到那个引擎。

    用取函数把它推迟到**第一次真正查库**那一刻，问题就没有了：
    那时 lifespan 早已进入，取函数返回的就是当前活着的那个引擎；
    应用若被重建，取函数自然返回新的那个，不需要任何「换引擎」的代码。

    ⚠️ 另一个形态（直接传引擎）保留给测试：单测里引擎是现成的，
    让它们多包一层 lambda 只是噪声。
    """

    def __init__(
        self,
        engine: "AsyncEngine | Callable[[], AsyncEngine | None]",
        schema: str | None,
    ) -> None:
        """初始化。

        Args:
            engine: 业务库引擎（与订单/审批**共用**同一个，见
                ``src/storage/engine.py`` 的连接池账）；
                或一个返回它的无参可调用对象。
            schema (`str | None`): 目标 schema。
        """
        self._engine_ref = engine
        self._schema = schema

    def _engine(self) -> AsyncEngine:
        """取出当前业务库引擎。

        Returns:
            `AsyncEngine`: 当前引擎。

        Raises:
            RuntimeError: 取函数返回 None，即**应用生命周期尚未进入**。
                ⚠️ 这个错误必须响亮：它意味着有人在 lifespan 之外
                调用了画像接口。静默退回内存实现的话，症状会变成
                「写进去的画像重启就没了」，而那是数据丢失。
        """
        engine = (
            self._engine_ref()
            if callable(self._engine_ref)
            else self._engine_ref
        )
        if engine is None:
            raise RuntimeError(
                "业务库引擎尚不可用 —— 画像仓储只能在应用 lifespan "
                "进入之后使用（写入与读取都在请求路径上，不该发生这种事）。",
            )
        return engine

    @property
    def _table(self) -> Table:
        """当前 schema 下的表对象。"""
        return profile_table(self._schema)

    async def get(self, user_id: str) -> TravelerProfile | None:
        """读画像。

        ⚠️ 走 ``model_validate`` 而不是 ``model_construct``：库里那条
        记录可能是**旧版本代码**写的，缺少今天新增的字段。
        ``model_validate`` 会用字段默认值补上（``TravelerProfile``
        所有字段都有默认值，见其文档），而 ``model_construct``
        会造出一个缺属性的对象 —— 那个错误会在**第一次访问该字段时**
        才炸，离这里十万八千里。

        ⚠️ 反过来，如果记录里有一个**今天不存在**的字段
        （回滚了应用版本），pydantic 默认忽略它。这是对的选择：
        回滚一次不该让整个画像读不出来。

        Args:
            user_id (`str`): 员工 id。

        Returns:
            `TravelerProfile | None`: 画像；没有记录时为 None。
        """
        table = self._table
        async with self._engine().connect() as conn:
            row = (
                await conn.execute(
                    select(table.c.data).where(table.c.user_id == user_id),
                )
            ).first()
        if row is None:
            return None
        return TravelerProfile.model_validate(row[0])

    async def upsert(self, profile: TravelerProfile) -> TravelerProfile:
        """整体覆盖写。

        ⚠️ 用方言原生的 ``ON CONFLICT DO UPDATE``（"upsert"），而**不是**
        「先 SELECT 再决定 INSERT / UPDATE」。后者在并发下会同时通过
        SELECT 的存在性检查，然后一个 INSERT 撞主键 ——
        而那个错误只在并发时出现，本地怎么测都是绿的。

        Args:
            profile (`TravelerProfile`): 新画像。

        Returns:
            `TravelerProfile`: 落库后的画像（与入参一致）。
        """
        table = self._table
        payload = profile.model_dump(mode="json")

        dialect = self._engine().dialect.name
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert as pg_insert

            stmt = pg_insert(table).values(user_id=profile.user_id, data=payload)
            stmt = stmt.on_conflict_do_update(
                index_elements=[table.c.user_id],
                set_={"data": stmt.excluded.data, "updated_at": func.now()},
            )
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert as lite_insert

            stmt = lite_insert(table).values(user_id=profile.user_id, data=payload)
            stmt = stmt.on_conflict_do_update(
                index_elements=[table.c.user_id],
                set_={"data": lite_insert(table).excluded.data},
            )
        else:
            # ⚠️ 未知方言退化成「先删后插」。它**不是**原子的，所以
            # 这里明确拒绝 —— 与其在一个没被验证过的方言上悄悄丢掉
            # 并发保证，不如在这里响亮地失败。真要用别的库，
            # 请先为它写一版 upsert 并补测试。
            raise NotImplementedError(
                f"方言 {dialect!r} 上没有实现画像的原子 upsert。"
                "请为它补一版实现（或在配置里换回 postgresql）。",
            )

        async with self._engine().begin() as conn:
            await conn.execute(stmt)
        return profile

    async def merge(
        self,
        user_id: str,
        patch: ProfilePatch,
    ) -> TravelerProfile:
        """**原子地**部分更新画像。

        ⚠️⚠️ 实现的关键是 ``SELECT ... FOR UPDATE``（行锁）。
        没有它的话，「读出来 → 改 → 写回去」在并发下会丢更新：
        两个请求分别改了座位偏好和成本中心，最后一个写的那个
        会把另一个的改动**整个覆盖掉**（因为写回的是整份 JSON）。
        症状是「用户刚设的偏好，下一秒又变回去了」，且只在并发时出现。

        ⚠️ 行锁只对**已存在**的行有效。新用户的第一次 merge 会走到
        INSERT 分支，两个并发请求可能同时 INSERT，其中一个撞主键。
        这里用 ``begin_nested()``（SAVEPOINT）捕获它并退回去走 UPDATE
        分支 —— 直接捕获 IntegrityError 是不行的：在 PostgreSQL 里
        事务一旦出错就**整个作废**，后续语句一律拒绝执行，
        除非中间有一个 SAVEPOINT 把状态回滚回来。

        Args:
            user_id (`str`): 员工 id。
            patch (`ProfilePatch`): 要改的字段。

        Returns:
            `TravelerProfile`: 合并后的画像。
        """
        table = self._table

        async with self._engine().begin() as conn:
            existing = await self._select_locked(conn, table, user_id)

            if existing is not None:
                merged = patch.apply(TravelerProfile.model_validate(existing))
                await conn.execute(
                    table.update()
                    .where(table.c.user_id == user_id)
                    .values(data=merged.model_dump(mode="json"), updated_at=func.now()),
                )
                return merged

            # 新用户：画像不存在时凭空建一份（不是报错）——
            # 用户第一次说「我靠窗」时不该收到「请先创建画像」。
            merged = patch.apply(TravelerProfile(user_id=user_id))
            try:
                async with conn.begin_nested():
                    await conn.execute(
                        table.insert().values(
                            user_id=user_id,
                            data=merged.model_dump(mode="json"),
                        ),
                    )
                return merged
            except IntegrityError:
                # 另一个请求抢先插入了。它插的那份已经带着它自己的改动，
                # 所以这里必须**重新**读一次再加自己的 patch —— 用
                # 上面那份 ``merged`` 会把它刚写的东西抹掉。
                logger.debug("画像 %s 并发首写冲突，退回合并路径。", user_id)
                fresh = await self._select_locked(conn, table, user_id)
                if fresh is None:
                    # 抢插的那个事务回滚了（比如它后续失败了）。
                    # 重试一次直插；再失败就让它抛 —— 无限重试更糟。
                    await conn.execute(
                        table.insert().values(
                            user_id=user_id,
                            data=merged.model_dump(mode="json"),
                        ),
                    )
                    return merged
                again = patch.apply(TravelerProfile.model_validate(fresh))
                await conn.execute(
                    table.update()
                    .where(table.c.user_id == user_id)
                    .values(data=again.model_dump(mode="json"), updated_at=func.now()),
                )
                return again

    async def _select_locked(
        self,
        conn: object,
        table: Table,
        user_id: str,
    ) -> dict | None:
        """带行锁地读出这一行的 ``data`` 载荷。

        ⚠️⚠️ 返回的是**载荷本身**（``dict``），不是 ``Row``。
        这一点必须写死在这个函数里：``merge`` 拿到的东西会直接喂给
        ``TravelerProfile.model_validate``，而 pydantic **不认**
        SQLAlchemy 的 ``Row``（它既不是 dict 也没有被注册成
        dataclass），会抛 ``Input should be a valid dictionary``。
        这个错误只在「第二次更新同一个用户」时才出现 ——
        首次写入走的是 INSERT 分支，压根不经过这里。

        ⚠️ ``FOR UPDATE`` **只在 PostgreSQL 上加**。sqlite 不支持这个
        子句（SQLAlchemy 的 sqlite 方言会把它渲染成 ``FOR UPDATE``
        然后被数据库拒绝），而 sqlite 走的又是 ``make test`` 那条路 ——
        加了它测试当场就挂。sqlite 不需要它：它的写事务本来就是
        库级串行的。

        Args:
            conn (`object`): 已开启的 ``AsyncConnection``。
            table (`Table`): 画像表。
            user_id (`str`): 员工 id。

        Returns:
            `dict | None`: 该行的画像载荷；没有这一行时是 None。
        """
        stmt = select(table.c.data).where(table.c.user_id == user_id)
        if self._engine().dialect.name == "postgresql":
            stmt = stmt.with_for_update()
        row = (await conn.execute(stmt)).first()
        return None if row is None else row[0]


__all__ = [
    "SqlProfileRepository",
    "ensure_profile_table",
    "profile_table",
]
