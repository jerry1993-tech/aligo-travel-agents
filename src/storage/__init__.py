# -*- coding: utf-8 -*-
"""业务存储层：**独立于** AgentScope 框架存储的数据库访问。

==============================================================================
为什么必须是独立的一套，而不是复用框架的 storage
==============================================================================
    AgentScope 的 ``AsyncSQLAlchemyStorage`` 管的是**它的**表：
    ``sessions`` / ``messages`` / ``agents`` / ``credentials`` / ``schedules`` /
    ``teams`` / ``knowledge_bases`` / ``mcps`` / ``skills`` / ``channels`` /
    ``sops`` / ``sop_runs`` / ``knowledge_documents`` —— 共 13 张，全部由框架的
    ``_Base.metadata`` 建在默认 schema 里，且**框架不支持传入自定义 metadata**
    （``_Base`` 是 ``_tables.py`` 里的私有全局单例，没有任何参数能换掉它）。

    本项目的业务域要建的是另一批表：用户画像、行程、订单、申请单、审批流。
    它们的名字（``orders`` / ``trips`` / ``users`` …）与框架那 13 张一样通用，
    挤在同一个 schema 里迟早撞名；而 ``_Base.metadata.create_all`` 是
    「按 metadata 建表」，一旦两边共用 metadata，框架升级时新增的表
    会与我们的表混在一起，回滚应用版本却回滚不了 schema。

    因此：**业务表单独一套 engine + 单独一个 ``business`` schema + 单独一套
    alembic 迁移**（见 :mod:`src.storage.engine`）。

==============================================================================
本包当前包含
==============================================================================
    engine.py  业务库引擎的构造、``business`` schema 的引导、连通性探测。
               P2 只到「引擎可用」；真正的业务表在 P3 随第一个业务实体加入。

从 :mod:`src.storage` 直接导入的内容会转发到 :mod:`src.storage.engine`。
"""

from .engine import (
    BUSINESS_SCHEMA,
    build_business_engine,
    business_schema_for,
    ensure_business_schema,
    ping_business_engine,
    storage_engine_kwargs,
)

__all__ = [
    "BUSINESS_SCHEMA",
    "build_business_engine",
    "business_schema_for",
    "ensure_business_schema",
    "ping_business_engine",
    "storage_engine_kwargs",
]
