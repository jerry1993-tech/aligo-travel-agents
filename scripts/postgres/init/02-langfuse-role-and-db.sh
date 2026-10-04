#!/bin/bash
# ==============================================================================
# 文件职责：为本项目的 trace 后端 **Langfuse v3** 创建**专用角色 + 专用 database**。
#
# 执行时机：与 01-business-schema.sh 相同 —— **仅容器首启、且 PGDATA 为空时**
#           由 postgres 官方入口脚本执行（挂载自 docker-compose.yaml 的
#           postgres.volumes）。重启容器不会再跑，理由见那个文件的注释。
#
# 上下游依赖：
#   - 上游：docker-compose.yaml 的 postgres.environment 注入的
#           LANGFUSE_DB_PASSWORD（以及 POSTGRES_USER / POSTGRES_DB）。
#   - 下游：langfuse / langfuse-worker 两个服务用
#               postgresql://langfuse:${LANGFUSE_DB_PASSWORD}@postgres:5432/langfuse
#           连接（见 docker-compose.yaml 的 x-langfuse-env 锚点）。
#           这两条字符串必须**逐字对上** —— 对不上的症状是 langfuse 容器
#           反复重启并报 FATAL: password authentication failed。
#
# ------------------------------------------------------------------------------
# 为什么 Langfuse 用独立角色而不是复用超级用户
# ------------------------------------------------------------------------------
#   观测库与业务库的生命周期完全不同：观测数据可以随时清空、可以整体迁移到
#   另一台机器，而业务数据不能。角色分开之后，两套凭据可以**分别轮换** ——
#   泄漏了观测库的密码，业务库不受影响。共用一个超级用户则做不到这一点：
#   任何一处需要改密码，两边的服务都得同时重启。
#
#   还有一个更实际的理由：Langfuse 会自己执行 schema 迁移（它内部跑 prisma
#   migrate）。让一个会自动改表结构的第三方组件拿着**超级用户**凭据，
#   等于把「它不会误删业务库」这件事完全寄托在它的实现正确性上。
#   它的连接串里指定的 database 是 langfuse，但超级用户能跨库操作 ——
#   把权限收窄到只够用它自己的库，是一道成本几乎为零的纵深防御。
#
# ------------------------------------------------------------------------------
# ⚠️ 本脚本在「不启用 tracing 档」时也会执行，这是**刻意**的
# ------------------------------------------------------------------------------
#   core 档（make up）并不启动 langfuse，因此这次建的角色与库暂时没人用。
#   保留它的理由：这两样东西**只在首启时能建**（见 01 脚本的说明）。
#   如果改成「只在 tracing 档建」，那么一个先跑过 `make up`、
#   后来才想 `make up-tracing` 的人会撞上「库不存在，且卷已非空、脚本不再执行」——
#   唯一的出路是删掉全部数据卷重来，把一次「想看看 trace」变成一次数据清空。
#   建一个空库的代价是几 MB，远低于上面那条路的代价。
#
# ------------------------------------------------------------------------------
# ⚠️ 一个刻意避开的写法：不要在 DO $$ ... $$ 里用 psql 变量
# ------------------------------------------------------------------------------
#   第一版写成 `DO $$ ... :'langfuse_role' ... $$`，想用 DO 块拿到
#   「不存在才创建」的幂等性。问题是：psql 的变量插值作用在它自己解析的
#   SQL 文本上，而**美元引用（$$ ... $$）之间是一段不被解析的原文** ——
#   插值是否发生取决于 psql 的版本与具体上下文，不能指望。
#   一旦没插值，DO 块里留给服务端的就是字面量 ':' 'langfuse_role'，
#   报错指向一段看起来完全正常的 SQL，排查方向完全被带偏。
#
#   这里改用 **`SELECT format(...) WHERE NOT EXISTS (...)` + `\gexec`**：
#   · 所有 `:'name'` / `:"name"` 都在**顶层**（不在任何引用体内），插值必然发生；
#   · `format('%I'/'%L')` 在**服务端**做标识符/字面量的转义，
#     因此密码里带单引号也不会破坏语句 —— 随机密码里出现单引号是常事，
#     而没有 format 时那会变成一个难以理解的语法错误；
#   · 条件不成立时 SELECT 不返回行，`\gexec` 什么也不执行 ⇒ 天然幂等。
# ==============================================================================

set -euo pipefail

: "${POSTGRES_DB:?POSTGRES_DB 未设置：请检查 docker-compose.yaml 的 postgres.environment}"
: "${POSTGRES_USER:?POSTGRES_USER 未设置：请检查 docker-compose.yaml 的 postgres.environment}"
: "${LANGFUSE_DB_PASSWORD:?LANGFUSE_DB_PASSWORD 未设置：请检查 docker-compose.yaml 的 postgres.environment}"

# langfuse 的角色名与库名在本项目里是**固定的**（不来自环境变量）。
# 理由：它们被 docker-compose.yaml 的 DATABASE_URL 写死成字面量
# （`postgresql://langfuse:...@postgres:5432/langfuse`），而那段注释明确说了
# 「用户名必须写死 langfuse，不可写成 ${POSTGRES_USER}」。
# 若这里改成可配置，就会出现「脚本建成 A、连接串去连 B」的静默错配，
# 症状是 langfuse 无限重启。两处都写死反而消除了这一类问题。
LANGFUSE_DB="langfuse"
LANGFUSE_ROLE="langfuse"

echo "[init] 创建 Langfuse 专用角色与数据库（${LANGFUSE_ROLE} / ${LANGFUSE_DB}）..."

# ------------------------------------------------------------------------------
# 第一步：在**业务库**里建角色与数据库（两者都是实例级对象，连哪个库都能建）
# ------------------------------------------------------------------------------
psql \
  --username "${POSTGRES_USER}" \
  --dbname "${POSTGRES_DB}" \
  --set ON_ERROR_STOP=1 \
  --no-psqlrc \
  --set lf_role="${LANGFUSE_ROLE}" \
  --set lf_db="${LANGFUSE_DB}" \
  --set lf_password="${LANGFUSE_DB_PASSWORD}" \
  <<'SQL'
-- 角色：不存在则建（\gexec 执行 SELECT 产生的语句；没有行就什么都不做）。
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'lf_role', :'lf_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'lf_role')
\gexec

-- 无论上一步是「新建」还是「已存在」，都重置一次密码。
-- ⚠️ 这一步不是多余的：常见情形是「卷还在、.env 里的密码改了」，
-- 此时若不重置，langfuse 会一直拿旧密码连不上 ——
-- 而初始化日志看起来一切正常，排障会从 langfuse 的配置一路查到网络。
ALTER ROLE :"lf_role" WITH LOGIN PASSWORD :'lf_password';

-- 数据库：CREATE DATABASE 不能出现在事务块里，因此同样用 \gexec。
-- OWNER 指定为该角色，省掉后面一堆跨库授权。
SELECT format('CREATE DATABASE %I OWNER %I', :'lf_db', :'lf_role')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'lf_db')
\gexec
SQL

# ------------------------------------------------------------------------------
# 第二步：连到 langfuse 库，授予 public schema 的建表权限
# ------------------------------------------------------------------------------
# 为什么必须**另起一次 psql**：`GRANT ... ON SCHEMA` 作用于**当前库**的 schema，
# 而 PostgreSQL 没有跨库操作这回事（除 dblink / postgres_fdw 那类扩展）。
# 在业务库里写 `ON SCHEMA public` 改的是业务库的 public —— 看起来执行成功，
# 对 langfuse 库毫无影响，属于最难发现的一类「静默无效」。
#
# ⚠️ 这一段真正防的是 PostgreSQL 15 的一个行为变更：
#    从 PG15 起，public schema **不再**默认授予 PUBLIC 建表权限
#    （CVE-2018-1058 的加固）。Langfuse 靠 prisma migrate 自己建表，
#    缺了这条授权会以
#        permission denied for schema public
#    启动失败 —— 而报错里不会提到「这是 PG15 的变更」，
#    只会让人以为 Langfuse 的镜像或权限配置坏了。
#    本项目用 postgres:16-alpine，正落在受影响范围内。
#    这也是为什么不能照抄一份 PG14 时代的初始化脚本。
echo "[init] 授予 ${LANGFUSE_ROLE} 在 ${LANGFUSE_DB}.public 上的建表权限 ..."

psql \
  --username "${POSTGRES_USER}" \
  --dbname "${LANGFUSE_DB}" \
  --set ON_ERROR_STOP=1 \
  --no-psqlrc \
  --set lf_role="${LANGFUSE_ROLE}" \
  <<'SQL'
GRANT ALL ON SCHEMA public TO :"lf_role";
SQL

# ------------------------------------------------------------------------------
# 第三步：把两个库的 CONNECT 权限从 PUBLIC 上收回
# ------------------------------------------------------------------------------
# ⚠️ 这一步不是「顺手加固」，它补的是一个**已验证存在**的缺口。
#    PostgreSQL 出厂就给 PUBLIC 授予了所有数据库的 CONNECT 权限，
#    因此「给 langfuse 建了一个专用角色」这件事**本身并不构成隔离** ——
#    实测（本项目验收时跑过）：
#        psql -U langfuse -d aligo -c 'SELECT 1'   → 成功
#    也就是说，一旦 langfuse 侧被攻破，攻击者拿到的凭据可以直接连上业务库。
#    这不等于能读到业务数据（business schema 没有授予它任何权限），
#    但「能不能连」与「能连上后能做什么」是两道独立的防线，
#    而这一道的成本是两行 SQL。
#
#    收回之后，langfuse 的凭据**只能连它自己的库** —— 与上面
#    「把权限收窄到只够用它自己的库」那句注释名副其实。
#
# ⚠️ 为什么这样做是安全的（写下来以免后人不敢加）：
#    · 两个库的属主都是超级用户/该角色本身，属主与超级用户**不受**
#      CONNECT 授权的影响（超级用户绕过一切权限检查）；
#    · 本栈里连业务库的只有应用（用 POSTGRES_USER，超级用户）
#      与 psql 排障（同上），没有任何第三方角色依赖 PUBLIC 的 CONNECT；
#    · PostgreSQL 的连接串里 database 是必填项，不存在「连不上 A 库就走 B 库」
#      的隐式回退 —— 因此这条改动不会把故障从一个库转移到另一个库，
#      只会让「不该连的库」明确地连不上。
#
# ⚠️ 顺序要紧：必须**先**完成上面的 GRANT，**再**收回 CONNECT。
#    反过来写时，下面这条 psql 会因为连不上 langfuse 库而失败 ——
#    而它失败的表现是一个连接错误，看不出「只是顺序写反了」。
echo "[init] 收回两个数据库对 PUBLIC 的 CONNECT 权限 ..."

psql \
  --username "${POSTGRES_USER}" \
  --dbname "${POSTGRES_DB}" \
  --set ON_ERROR_STOP=1 \
  --no-psqlrc \
  --set lf_db="${LANGFUSE_DB}" \
  <<'SQL'
-- 业务库：只有超级用户（应用）能连。
-- ⚠️ 库名用 current_database() 取，而不是写 CURRENT_DATABASE 之类的关键字：
--    GRANT/REVOKE 的库名位置要的是**标识符**，PostgreSQL 在那里并不认
--    CURRENT_DATABASE（它只在少数几个语法位置可用）。写错了会得到一个
--    语法错误，而那条语句看起来相当合理。用 format('%I', current_database())
--    既避开这个坑，也顺带完成了标识符转义。
SELECT format('REVOKE CONNECT ON DATABASE %I FROM PUBLIC', current_database())
\gexec

-- langfuse 库：只有它自己的角色能连。
-- 用 format() 是因为库名要作为**标识符**参与，不能写成字符串字面量。
SELECT format('REVOKE CONNECT ON DATABASE %I FROM PUBLIC', :'lf_db')
\gexec
SELECT format('GRANT CONNECT ON DATABASE %I TO %I', :'lf_db', 'langfuse')
\gexec
SQL

echo "[init] Langfuse 角色与数据库就绪。"
