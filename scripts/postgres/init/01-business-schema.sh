#!/bin/bash
# ==============================================================================
# 文件职责：在业务库里创建 **business** schema（本项目的业务表全部落在这里）。
#
# 执行时机：**仅容器首启、且 PGDATA 为空时**由 postgres 官方入口脚本执行。
#           它被 compose 挂载到 /docker-entrypoint-initdb.d（见 docker-compose.yaml
#           的 postgres.volumes）。之后重启容器不会再跑 —— 这一点极其重要，
#           见下方「为什么不能靠改这个文件来补救」。
#
# 上下游依赖：
#   - 上游：docker-compose.yaml 注入的 POSTGRES_USER / POSTGRES_DB / LANGFUSE_DB_PASSWORD。
#   - 下游：src/storage/engine.py 的业务表全部声明
#           `__table_args__ = {"schema": "business"}`；alembic 的
#           version_table_schema 也指向它。schema 不存在 ⇒ 建表直接失败。
#
# ------------------------------------------------------------------------------
# 为什么业务表要单独一个 schema（而不是和 AgentScope 的表混在 public）
# ------------------------------------------------------------------------------
#   AgentScope 自带的表名**极其通用**：sessions / messages / agents / credentials /
#   schedules / teams / knowledge_bases …（见已安装 agentscope 包的
#   app/storage/_sql/_tables.py）。我们的业务域里同样会有 sessions 概念、
#   同样会有 credentials 概念 —— 放在同一个 schema 里，撞名是迟早的事。
#
#   撞名的后果不是「报错」，而是**静默地共用一张表**：框架往 sessions 里写会话，
#   我们的业务代码往「同一个」sessions 里写业务会话，两边都觉得自己独占它。
#   直到某天一次框架升级改了列定义，业务数据被连带改坏 —— 而那时已经很难
#   追溯是「从哪一天开始共用的」。
#
#   隔离到 business schema 之后，两边的表在物理上就不可能撞上。
#
# ------------------------------------------------------------------------------
# ⚠️ 为什么不能靠改这个文件来补救（一个真实踩过的坑）
# ------------------------------------------------------------------------------
#   postgres 官方镜像的入口脚本只在 **PGDATA 为空**时执行 initdb 与
#   /docker-entrypoint-initdb.d 下的脚本。若 pg_data 卷里已经有了数据
#   （比如上一次启动失败前已经建好了集群），**本文件根本不会被读**。
#
#   于是会产生一个很反直觉的现象：改了本文件、`make up`、容器起来了、
#   日志里一句相关的话都没有 —— 但改动**完全没有生效**。
#   表现为「我明明加了 schema，程序还是报 relation does not exist」。
#
#   遇到这种情况的唯一解法是删卷重来（`make clean` 或
#   `docker volume rm aligo-travel-agents_pg_data`），而不是继续改脚本。
# ==============================================================================

set -euo pipefail

# 使用 postgres 官方入口脚本提供的两个变量：
#   POSTGRES_DB   —— 默认业务库名（compose 里是 ${POSTGRES_DB}）
#   POSTGRES_USER —— 超级用户名
# 这里刻意**不写默认值兜底**（不用 ${VAR:-aligo}）：万一 compose 没把变量传进来，
# 我们要的是立刻失败，而不是连到一个「猜出来的」库上 —— 后者会在一个错误的库上
# 建出 schema，而这个错误要到应用建表时才会暴露，中间隔了好几层。
: "${POSTGRES_DB:?POSTGRES_DB 未设置：请检查 docker-compose.yaml 的 postgres.environment}"
: "${POSTGRES_USER:?POSTGRES_USER 未设置：请检查 docker-compose.yaml 的 postgres.environment}"

echo "[init] 在数据库 ${POSTGRES_DB} 中创建 business schema ..."

psql \
  --username "${POSTGRES_USER}" \
  --dbname "${POSTGRES_DB}" \
  --set ON_ERROR_STOP=1 \
  --no-psqlrc \
  <<'SQL'
-- IF NOT EXISTS 是必需的（而不是可选的美化）：
--   本脚本虽然只在首启时执行一次，但在**重建容器但保留卷**的排障过程中，
--   有人会手工再跑一次。没有 IF NOT EXISTS 时它会以
--   「schema already exists」失败 —— 而因为脚本开头是 set -e，
--   整个初始化会中断，连带后面的 langfuse 建库也不执行。
CREATE SCHEMA IF NOT EXISTS business;

-- 把新建 schema 的权限交给业务用户，好让应用（以及 alembic 迁移）能在这里建表。
-- 不写这一步时，若将来把应用换成非超级用户角色，会在建表时收到
-- 「permission denied for schema business」。
GRANT ALL ON SCHEMA business TO CURRENT_USER;
SQL

echo "[init] business schema 就绪。"
