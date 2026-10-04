# ==============================================================================
# 文件职责：AliGo 差旅助手的一键操作入口。把「起服务 / 关服务 / 跑测试 / 冒烟 / 看日志」
#           等高频操作收敛为固定目标，避免每个开发者各写一套命令。
#
# 上下游依赖：
#   - 上游：调用 `docker compose`（读取 docker-compose.yaml 与 .env）与本地 `python3`。
#   - 下游：`make test` 依赖 tests/ 与 pytest.ini；`make smoke` 依赖 scripts/smoke.py。
#
# 使用前提：本机 Docker 守护进程已启动，且已执行 `cp .env.example .env`。
# ==============================================================================

# 用 bash 执行 recipe（配方里用到 [[ ]]、进程替换 <() 、set -o pipefail 等 bash 语法）。
#
# ⚠️ 这里**刻意不写 fail-fast 开关**，因为「设一个变量就让行内命令失败即中断」在本机行不通，
#    而按惯例去写它会得到一个**假绿**：
#      · 本机 /usr/bin/make 是 **GNU Make 3.81**（Apple 随 macOS 附带的那一版）；
#      · `.SHELLFLAGS` 是 **3.82 才引入**的，在 3.81 下**被静默忽略** —— 命令照旧只以 `-c` 执行，
#        旧的 `SHFLAGS` 同样不被采纳。两项都已实测排除：把 SHELL 指向一个记录 argv 的脚本后，
#        make 实际传的是 `-c echo hi`，与有没有设那两个变量无关。
#    make 的中断粒度是**整条 recipe 行**：行内用 `;` 串联的命令，前面失败后面照样跑。
#    所以需要 fail-fast 的配方必须**自己**写 `set -eu -o pipefail;`（3.81 下实测有效，
#    失败时 make 会以 Error 中断该目标）。目前只有 check-env 与 preflight 这么做，
#    其余配方是「逐行失败即停、行内不保证」的既有语义 —— 这是已知取舍，不是疏漏。
SHELL := /bin/bash

# 默认目标：`make` 等价于 `make help`，避免误触发破坏性操作。
.DEFAULT_GOAL := help

# 统一变量：compose 命令前缀，便于整体替换。
COMPOSE      := docker compose
# 应用镜像名**不在这里定义** —— 它的真值只有一处：docker-compose.yaml 里 app 服务的
# `image: aligo-travel-app:local`。此处原先有一个 APP_IMAGE 变量，但全仓库除它自己的
# 定义行外**没有任何引用**，而注释写着「便于整体替换」：改它零效果，是典型的假配置
# （会让人以为改名成功了），故删除。真要改镜像名，改 docker-compose.yaml 那一行。
# 冒烟脚本访问的地址（宿主机侧）。默认 8000，与 .env 里 ALIGO__APP__PORT 的**默认值**一致。
# ⚠️ 它**不是**从 compose 推导出来的（早先这里写「与 compose 的映射保持一致」，是错的）：
#    compose 的映射是 `${ALIGO__APP__PORT:-8000}:8000`，改了 .env 里那个键，宿主侧端口就变了，
#    而本变量纹丝不动 —— `make smoke` 会去打一个没人监听的端口，把健康服务报成「冒烟未通过」。
#    改过那个键时请一并覆盖： make smoke SMOKE_URL=http://localhost:8010
SMOKE_URL    ?= http://localhost:8000
# 负载测试脚本访问的地址（宿主机侧）。默认与 SMOKE_URL 同值，但**刻意分开一个变量**：
# 负载测试会持续加压，通常不希望它与冒烟共用同一个可覆盖点 —— 若共用一个变量，
# `make smoke SMOKE_URL=http://staging` 会连带把 `make loadtest` 也指向 staging，
# 而压测打到哪台机器本该是**显式**的选择。改了 .env 的 ALIGO__APP__PORT 时两者都要跟着改。
LOADTEST_URL ?= http://localhost:8000
# loadtest 目标的附加参数。默认是**纯基础设施基线**：只打 /healthz、跑 10 秒
# （免鉴权、零 I/O、不调模型），因此 `make loadtest` 不会误触业务链路或花钱调模型。
# 要压业务端点请整体覆盖，例如：
#     make loadtest LOADTEST_ARGS="--endpoint /api/v1/health --requests 5000 --concurrency 50"
LOADTEST_ARGS ?= --health-only --duration 10
# 并发正确性测试脚本访问的地址（宿主机侧）。与上面两个**同样刻意分开**：
# 它会真的建智能体、建会话、调模型（三个用户同时），因此「打到哪台机器」必须
# 是显式选择，不能被 `make smoke SMOKE_URL=...` 顺手带偏。
CONCURRENCY_URL ?= http://localhost:8000
# concurrency 目标的附加参数。默认即任务要求的形态：3 个用户 × 1 轮。
#     make concurrency CONCURRENCY_ARGS="--users 6 --rounds 2"     # 加压
#     make concurrency CONCURRENCY_ARGS="--keep"                   # 留现场排查
CONCURRENCY_ARGS ?=
# 本地开发用的解释器；容器内不使用该变量。
PYTHON       ?= python3

# `provision-agent` 要预置智能体的**用户身份**。
#
# ⚠️ 变量名刻意不叫 `USER`：`USER` 是 POSIX 环境变量，make 会把环境里的值
#    直接当成变量值 —— 于是 `make provision-agent`（不带参数）会用**当前
#    登录的操作系统用户名**，在一个谁都没听说过的工作区里建智能体，
#    然后打印「✅ 已创建」。目标用户 `alice` 反而什么都没拿到。
#    带 `PROVISION_` 前缀的名字不在环境里，缺参数时用的是下面这个默认值。
PROVISION_USER ?= alice

# `docker compose up --wait` 的等待上限（秒）。**必须显式给出**，否则该命令会无限等待。
# 为什么不设会出事：app 的就绪依赖 Redis 消息总线，若 Redis 的密码未正确注入，
# SchedulerManager 会卡在「等订阅就绪」上永不返回（详见 config/base.yaml 的踩坑记录），
# 此时容器既非 healthy 也非 exited，`--wait` 会**一直挂住**，终端看起来像卡死，
# 排查时完全看不到「到底谁没起来」。
# 给出上限后，超时会以非零退出码失败并打印未就绪的服务，问题立刻可见。
# 取值依据：Milvus 冷启动最慢（其自身 start_period 为 90s，且要连 etcd 与 MinIO 初始化元数据。
#          注意：它首启**不会**创建业务集合 —— 集合需 P4 用 pymilvus 显式建，见 docs/06 第九节），
#          叠加镜像构建耗时，需要留足余量。
# ⚠️ 本机实测：首次 `make up` 里单是构建 app 镜像（apt 装 build-essential + pip 装依赖，
#    容器内下载约 180–240 kB/s）就会吃掉大半预算。
#    因此默认值取 **480**（= 实测通过的那个值：本轮即以 `make up WAIT_TIMEOUT=480` 成功，
#    退出码 0）。此前默认是 300，而文档与验收记录里所有**成功样本**用的都是 **480**
#    （600 只出现在「怎么放宽上限」的建议里，**从未**出现在任何一条成功记录中），
#    300 在全部记录里**一次都没有出现过**。我们只能陈述这个事实，不能反推「300 必然失败」
#    —— 那需要一个 300 的失败样本，我们没有，也**不再补测**。补测的真实代价（别夸大成
#    「要删镜像」）：用 `make build`（compose `build --no-cache`）就能在不删镜像的前提下
#    复现同样的慢速 pip 构建，或者 `make down`（保留镜像与卷）后直接
#    `docker compose --profile core up -d --wait --wait-timeout 300` 只量运行时冷启动。
#    两条路都**不碰任何卷**，丢的只是 pip 层缓存与 5 分钟以上（README 有记）；
#    收益仅是把一句注释说得更硬，故不做。改成 480 是为了让「不传任何参数」这条路径用的就是
#    已被验证过的值；镜像建好后重跑很快，觉得等太久的可自行调小：`make up WAIT_TIMEOUT=300`。
WAIT_TIMEOUT ?= 480

# 离线评测（make eval）的综合得分阈值。取值依据见 scripts/eval.py 的 DEFAULT_MIN_SCORE：
# 正常工作的系统在这份黄金数据集上应接近 1.0，0.8 留出个位数用例失败的余量。
# ⚠️ 改这里只影响 `make eval`；直接跑 `python scripts/eval.py` 用的是脚本里的默认值。
MIN_SCORE    ?= 0.8
# `make eval` 产出的机器可读报告的落盘路径（相对仓库根；已在 .gitignore 里忽略）。
EVAL_OUTPUT  ?= eval_report.json

# 声明伪目标：这些名字不是真实文件，避免与同名目录冲突导致目标被跳过。
.PHONY: help up up-core up-obs up-tracing up-optional down down-core logs ps build pull test smoke \
        loadtest concurrency eval milvus_init seed_data provision-agent check-docs check-env preflight \
        clean-gen clean clean-all

# 全部档位，供 down / ps / logs / pull / clean 一次性带上。
#
# 为什么需要：`down` 与 `clean` 必须「看得见」所有档的服务，否则未启用档的容器与卷会被
# **静默留下** —— 命令照样以退出码 0 结束，看起来清干净了，实际没有（本项目最忌讳的
# 「报告照绿、覆盖面缩水」）。`pull` 与 `logs` 同理需要完整的服务清单。
#
# ⚠️ 但 `ps` **实测不需要**它：不带任何 --profile 的 `docker compose ps` 与带齐四档的输出
#    **逐行完全一致**（12 个容器，2026-09-24 实测，Compose v2.31.0-desktop.2）——
#    ps 是按项目标签找容器的，不做 profile 过滤。此处仍然带上，只为这几个目标写法统一。
#    推论：别把「profile 决定 compose 看得见哪些服务」当成通用规则 —— `logs` 上它就不成立
#    （见 logs 目标的注释：不带档时 SVC=langfuse 会失败，但同为 core 档的 app 却成功）。
#
# ⚠️ 下面四个档名必须与 docker-compose.yaml 里的 `profiles:` 集合**完全一致**。
#    写错一个字母不会报任何错，只会让 down/clean 静默漏掉一批容器与卷。
#    校验方法：`grep -oE 'profiles: \[[a-z]+' docker-compose.yaml | sort -u` 应得同四个名字。
ALL_PROFILES := --profile core --profile observability --profile tracing --profile optional

# ------------------------------------------------------------------------------
# help —— 列出全部可用目标（默认目标）
# ------------------------------------------------------------------------------
# ⚠️ 下面这份清单必须与 Makefile 里**真实定义的目标**保持一致：
#    tests/test_makefile_contract.py 会断言「每个真实目标都在 help 输出里出现（help 自身除外）」，
#    漏列会让 make test 失败。
#    加这个闸门的原因：此前 help 漏列了 7 个真实存在的目标，而 help 是 `make` 的默认动作、
#    也就是使用者接触到的**唯一入口** —— 漏列等于这些目标对使用者不存在。
#
# 说明里的 ⚠️ 不是装饰：`down-core` 是 stop 而非 down、`build` 带 --no-cache、
# `check-docs` 只查「行号超没超界 + 数字对不对」而**不查内容** —— 这三处都是
# 「只看名字会理解错」的地方，故在入口处点明。
help:
	@echo "AliGo 差旅助手 —— 可用命令："
	@echo ""
	@echo "  启动与停止"
	@echo "    make up            启动 core 档（postgres/redis/etcd/minio/milvus/app）—— 日常开发用这个"
	@echo "    make up-obs        启动 core + observability（prometheus/grafana），可观测的最小完整形态"
	@echo "    make up-tracing    启动 core + observability + tracing（加 langfuse v3 四容器），内存最紧张的一档"
	@echo "                       ⚠️ 该目标会注入 OTLP 变量而**重建 app 容器**（运行中的 app 会被停掉重建）"
	@echo "    make up-optional   启动 optional 档（maxkb 知识库）"
	@echo "    make up-core       同 up，但**不等待**健康检查（只构建并后台启动，快速迭代用）"
	@echo "    make down          停止并移除全部容器（保留数据卷）"
	@echo "    make down-core     只停 core 档容器（⚠️ 是 stop 不是 down：容器保留，仅暂停，重启即可用）"
	@echo "                       ⚠️ 它**只停 core**：tracing 档的 clickhouse/langfuse 一个都不停，"
	@echo "                       所以跨档切换要腾内存时请用 make down，别用这个"
	@echo "    make ps            查看容器状态与健康情况"
	@echo ""
	@echo "  验证"
	@echo "    make test          跑 pytest 单元/集成测试"
	@echo "    make smoke         对已启动的服务发冒烟请求，验证核心链路"
	@echo "    make loadtest      对已启动的服务施加受控并发，量吞吐与 p99（⚠️ 会持续打压，"
	@echo "                       默认只打 /healthz；越线即非零退出，会打印是哪条阈值）"
	@echo "                       ⚠️ 地址变量是 LOADTEST_URL，与 SMOKE_URL 分开，避免误压到非目标环境"
	@echo "    make concurrency   多用户**同时**发**不同**请求，验证互不干扰、全部成功"
	@echo "                       （3 个用户各走一条不同链路；⚠️ 会真的建智能体/会话并调模型，"
	@echo "                        跑完自己删掉；地址变量是 CONCURRENCY_URL）"
	@echo "    make eval          离线跑黄金数据集，产出可 diff 的评测报告（不依赖 Docker）"
	@echo "                       （⚠️ 未配置模型密钥时自动退化为**规则判分**，报告里会标注 judge_mode=rule）"
	@echo "    make check-docs    校验文档里的两类硬伤：①「文件:行号」引用是否超出该文件行数；"
	@echo "                       ② 写死的计数（如「N 个用例」）是否等于机器实测值"
	@echo "                       （⚠️ 只查超界与数字，不查那一行的内容是否与文档描述相符）"
	@echo "    make logs          实时跟踪容器日志（默认 app；SVC 可换任意档的服务，如 SVC=langfuse）"
	@echo "    make check-env     环境前置检查：.env 存在、Docker 守护进程可达、.env 无违规键，"
	@echo "                       外加一行 .env 与 .env.example 的键名差集警告（只警告，不阻塞）"
	@echo "                       （up* / build 自动依赖它；只读**键名**，不打印任何值）"
	@echo ""
	@echo "  数据（Milvus / 业务库）"
	@echo "    make milvus_init   幂等建 Milvus 集合（政策库 + 长期记忆），并回读核验维度/索引/度量"
	@echo "                       （⚠️ 只建集合、不灌数据；与 seed_data 分工，先跑它）"
	@echo "    make seed_data     构造差旅政策文档与业务演示数据并灌进 Milvus / 业务库"
	@echo "                       （幂等，重复跑不产生重复数据；⚠️ 需要 Milvus / PostgreSQL 可达）"
	@echo "                       先离线预览将要写入的内容（不连任何服务）："
	@echo "                         make seed_data ARGS=\"--dry-run\""
	@echo "    make provision-agent  给一个用户预置 AliGo 主智能体（幂等），让浏览器里**能打字**"
	@echo "                       （⚠️ 没有智能体/会话时，前端的输入框是灰的、点不动；"
	@echo "                        理由见 scripts/provision_agent.py 的模块文档）"
	@echo "                          make provision-agent PROVISION_USER=bob"
	@echo ""
	@echo "  维护"
	@echo "    make pull          预拉取全部镜像（首次部署/换机时先跑，跳过需本地构建的 app）"
	@echo "    make build         重建应用镜像 —— ⚠️ 带 --no-cache 全量重建，pip 层要跑数分钟；"
	@echo "                       只是改了源码/文档时请用 make up（保留缓存，快得多）"
	@echo "    make preflight     启动前修复：补齐 postgres 初始化脚本的可执行位（up* 自动依赖它）"
	@echo "    make clean         停止服务并删除数据卷（⚠️ 数据库/向量库/日志全部清空，不可恢复）"
	@echo "    make clean-gen     清理本地缓存（__pycache__/.pytest_cache/.coverage/htmlcov），"
	@echo "                       不动容器与卷；**跳过 third_party/**（那份源码树按约定不改动）"
	@echo "    make clean-all     clean + clean-gen —— ⚠️ **含 clean，会不可恢复地删掉全部数据卷**"
	@echo "                       （等于先跑一次 make clean 再跑 clean-gen；不清镜像）"
	@echo ""
	@echo "  提示：本机 Docker VM 仅 3.82 GiB 内存，core 与观测栈同跑时请优先用 up-obs（只加指标），"
	@echo "        确认内存有余量再加 langfuse（up-tracing）。用 'docker stats' 观察真实占用。"
	@echo "        注意 langfuse v3 依赖 ClickHouse，内存占用远高于 v2，务必先 'make down' 再切档。"
	@echo "        机器偏慢导致启动超时，可放宽等待上限： make up WAIT_TIMEOUT=600"

# ------------------------------------------------------------------------------
# check-env —— 环境前置检查（**真的查环境**，不只是查 .env 在不在）
# ------------------------------------------------------------------------------
# 与 preflight 的分工：**check-env 只读**（只报错退出，绝不修改任何东西），
# **preflight 才做修复**（补可执行位）。两者在 up* 的依赖里按「先只读校验、后修复」排列。
#
# 为什么从「只判 .env 存在」扩到这里：目标名叫 check-env，读起来就该是「检查环境」，
# 而此前它只判一个文件在不在 —— 名实不符。更要紧的是下面几件事**原本哪里都没查**，
# 出问题时的暴露方式都极难归因：
#   1) .env 存在 —— 否则 compose 会抛晦涩的变量未定义错误（原有行为，原样保留）；
#   2) Docker 守护进程可达 —— 否则每个目标都只报一句「连不上 socket」（此前无处检查）；
#   3) .env 里**不得**出现 ALIGO__IMAGE_REGISTRY —— 见 .env.example 的红字警告：
#      它属于**编排层**（compose 的镜像前缀），一旦写进 .env 就会被 env_file 注入 app
#      容器，而 app 的配置校验器不允许该键，容器会以
#          image_registry  Extra inputs are not permitted
#      直接崩并进入 Restarting 循环；
#   4) .env 与 .env.example 的键名差集 —— **只警告不阻塞**：缺的键都有默认值兜底，
#      但「静默取默认值」是最难察觉的一类偏差，值得在启动前说一句。
# ⚠️ 本目标只读 .env 的**键名**，任何情况下都不打印值。
check-env:
	@set -eu -o pipefail; \
	if [ ! -f .env ]; then \
		echo "❌ 未找到 .env 文件。请先执行：  cp .env.example .env"; \
		echo "   （.env 已被 .gitignore 忽略，绝不入库）"; \
		exit 1; \
	fi; \
	if ! docker info >/dev/null 2>&1; then \
		echo "❌ Docker 守护进程不可达（docker info 失败）。"; \
		echo "   全部 make 目标底层都是 docker compose，请先启动 Docker Desktop 再重试。"; \
		exit 1; \
	fi; \
	if grep -qE '^[[:space:]]*ALIGO__IMAGE_REGISTRY[[:space:]]*=' .env; then \
		echo "❌ .env 里出现了 ALIGO__IMAGE_REGISTRY —— 这个键不能放在 .env。"; \
		echo "   它是编排层变量（compose 的镜像前缀），写进 .env 会被注入 app 容器，"; \
		echo "   而 app 的配置校验器不允许该键，容器会直接崩进 Restarting 循环："; \
		echo "       image_registry  Extra inputs are not permitted"; \
		echo "   请从 .env 删掉它，改用命令行传入，例如："; \
		echo "       make up ALIGO__IMAGE_REGISTRY=docker.m.daocloud.io/"; \
		exit 1; \
	fi; \
	keys() { sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=.*/\2/p' "$$1" | sort -u; }; \
	miss=$$(comm -13 <(keys .env) <(keys .env.example)); \
	if [ -n "$$miss" ]; then \
		n=$$(printf '%s\n' "$$miss" | wc -l | tr -d ' '); \
		echo "⚠️  .env 比 .env.example 少 $$n 个键（缺的都有默认值兜底，不阻塞启动）："; \
		if [ "$${SHOW_KEYS:-0}" = "1" ]; then \
			printf '     %s\n' $$miss; \
		else \
			echo "     加 SHOW_KEYS=1 可列出键名： make check-env SHOW_KEYS=1"; \
		fi; \
	fi

# ------------------------------------------------------------------------------
# preflight —— 启动前自动修复：确保挂进容器的 SQL 初始化脚本带可执行位
# ------------------------------------------------------------------------------
# 为什么必须自动修，而不是只在文档里提醒（依据是一次真实故障）：
#   scripts/postgres/init/*.sh 是**挂载**进 postgres 容器的（不是 COPY 进镜像），
#   而 postgres 官方入口脚本对 *.sh 的处置是「可执行就 "$f" 直接执行，否则用 . 引入」。
#   一旦可执行位丢失（本机就发生过），执行会以
#       /bin/sh: bad interpreter: Permission denied
#   结束，postgres 随即以**退出码 126** 死亡 —— 报错信息里完全不提"权限位"，
#   极易被误判成镜像坏、脚本语法错或磁盘权限问题。
#   更麻烦的是连带伤害：那次失败会在 pg_data 卷里留下一个**只初始化了一半**的集群
#   （langfuse 角色与库都没建），而 PGDATA 非空会让下次启动直接跳过 initdb ——
#   该库从此永远不会被补建，只能删卷重来。
# 因此这里在启动前自动补齐可执行位：只对**缺位**的文件动手（已可执行的跳过），
# 正常情况下是空操作，出问题时省掉整整一轮排查。
#
# ⚠️ 路径必须锚到**本 Makefile 所在目录**，不能写成相对路径：配方里那句
#    `[ -e "$f" ] || continue` 会把「脚本没找到」当成「这个文件不用处理」而**静默跳过** ——
#    若执行时 CWD 不在仓库根（`make -f 绝对路径/Makefile`、某些 IDE 的 make 集成），
#    相对 glob 匹配不到任何文件，本目标就会**假装成功**，恰好放行它要防的那个退出码 126。
#    锚定之后，glob 的位置不再取决于 CWD。
#    注：`make -C 目录` **不**属于这一类 —— 它先 chdir，相对 glob 照样命中。
#    实测（dry-run，只打印）：`make -C <repo> -n test` 里 $(CURDIR) 正是仓库根；
#    而 `cd /tmp && make -f <repo>/Makefile -n test` 里 $(CURDIR) 是 /tmp。
preflight:
	@set -eu -o pipefail; \
	here="$(dir $(lastword $(MAKEFILE_LIST)))"; \
	for f in "$${here}"scripts/postgres/init/*.sh; do \
		[ -e "$$f" ] || continue; \
		if [ ! -x "$$f" ]; then \
			echo "🔧 补齐缺失的可执行位：$$f"; \
			echo "   （缺失会导致 postgres 以退出码 126 启动失败，原因见 Makefile 本目标注释）"; \
			chmod +x "$$f"; \
		fi; \
	done

# ------------------------------------------------------------------------------
# up —— 启动 core 档（业务主链路），并等待健康检查通过
# ------------------------------------------------------------------------------
# --wait 会阻塞到所有容器 healthy，这样 `make up` 一返回就能确认环境可用，
# 不必再手工 `docker ps` 观察；与 P1 验收标准「make up 全部容器健康」直接对应。
# --wait-timeout 是它的强制配对项（理由见文件上方 WAIT_TIMEOUT 的注释）：
# 没有它就等于允许无限挂起，失败时没有任何可读信息。
# 注意：这里的「healthy」对 app 而言是 /readyz（已真实连过 PG/Redis/Milvus），
# 对中间件而言是各自的健康检查 —— 详见 docker-compose.yaml 中 app.healthcheck 的说明。
up: check-env preflight
	@echo "▶ 启动 core 档服务（postgres / redis / etcd / minio / milvus / app）..."
	@echo "  等待上限 ${WAIT_TIMEOUT}s（超时会失败并列出未就绪的服务；可用 make up WAIT_TIMEOUT=600 放宽）"
	$(COMPOSE) --profile core up -d --build --wait --wait-timeout $(WAIT_TIMEOUT)
	@echo ""
	@echo "✅ core 档已就绪。健康检查： curl -s $(SMOKE_URL)/healthz | $(PYTHON) -m json.tool"
	@echo "   建议接着跑： make smoke   （验证就绪探针、trace 透传与指标端点）"
	@echo "   Milvus: http://localhost:19530   MinIO 控制台: http://localhost:9001"

# 仅构建/启动、不等待，便于快速迭代
up-core: check-env preflight
	$(COMPOSE) --profile core up -d --build

# ------------------------------------------------------------------------------
# up-obs —— 启动 core + 指标档（Prometheus + Grafana）
# ------------------------------------------------------------------------------
# 必须同时带上 core：Prometheus 的抓取目标就是 core 里的 app（app:8000/metrics），
# 只起 observability 会得到一个「抓不到任何目标」的空壳，指标面板全为 No data。
up-obs: check-env preflight
	@echo "▶ 启动 core + observability 档（app / prometheus / grafana）..."
	$(COMPOSE) --profile core --profile observability up -d --build --wait --wait-timeout $(WAIT_TIMEOUT)
	@echo ""
	@echo "✅ 已就绪。Grafana: http://localhost:3000（admin 密码见 .env）"
	@echo "   Prometheus 目标页: http://localhost:9090/targets （aligo-app 应为 UP）"

# ------------------------------------------------------------------------------
# up-tracing —— 在 up-obs 基础上再加 Langfuse v3（内存最紧张的一档）
# ------------------------------------------------------------------------------
# Langfuse v3 不是一个容器，而是一组四个：langfuse(web) + langfuse-worker +
# clickhouse + langfuse-redis（外加复用 core 档既有的 minio 与 postgres）。
# 其中 clickhouse 是主要内存开销，故这一档显著重于 up-obs。
#
# 下面两行环境变量是**整档唯一的 trace 开关**：
#   · 命令行环境变量优先于 .env，所以只在执行本条 make 目标时生效；
#   · app 容器的配置因此发生变化，compose 会**自动重建**它（无需先 down）；
#   · 反过来，`make up` / `make up-obs` 不注入它们，app 就取 compose 里的默认值 none，
#     不会去连一个并不存在的 langfuse 而刷错误日志。
up-tracing: check-env preflight
	@echo "▶ 启动 core + observability + tracing 档（含 langfuse v3 四个容器）..."
	@echo "  ⚠️ 本机 Docker VM 仅 3.82 GiB，这是最重的一档；若健康检查随机失败请先 'make down'。"
	@ALIGO__OBSERVABILITY__TRACE_EXPORTER=otlp \
	 ALIGO__OBSERVABILITY__OTLP_ENDPOINT=http://langfuse:3000/api/public/otel/v1/traces \
	 $(COMPOSE) --profile core --profile observability --profile tracing up -d --build --wait --wait-timeout $(WAIT_TIMEOUT)
	@echo ""
	@echo "✅ 已就绪。Langfuse: http://localhost:3001（登录账号见 .env 的 LANGFUSE_INIT_USER_*）"
	@echo "   trace 已同时开向 Langfuse 的 OTLP 端点；若密钥留空，app 日志会有一条"
	@echo "   '未配置 LANGFUSE_*_KEY，跳过导出器装配' 的告警 —— 那是设计行为，不是故障。"

# ------------------------------------------------------------------------------
# up-optional —— 启动可选档（MaxKB），甲方 #5 拍板默认关停
# ------------------------------------------------------------------------------
up-optional: check-env preflight
	@echo "▶ 启动 optional 档（maxkb）..."
	$(COMPOSE) --profile optional up -d --wait --wait-timeout $(WAIT_TIMEOUT)
	@echo "✅ MaxKB 已就绪： http://localhost:8080"

# ------------------------------------------------------------------------------
# down —— 停止并移除容器与网络；**保留数据卷**，下次启动数据仍在
# ------------------------------------------------------------------------------
# ⚠️ 这里的 $(ALL_PROFILES) **是必需的、不是装饰**。`down` 与 `ps` 相反：它**按档过滤**。
#    实测（2026-09-24，12 个容器全在跑）：
#        docker compose down --dry-run                  → rc=0，**一行输出都没有**（什么都不删）
#        docker compose <四档> down --dry-run           → 12 个容器 + 网络 aligo-travel_aligo-net 全列出来
#    也就是说漏掉档位不会报错，只会**静默什么都不做** —— 比报错更难发现。改这行前先跑一遍
#    上面那条 `--dry-run`（dry-run 不产生副作用，可放心跑）。
#    另外：`down` 不带 `-v`，**卷一律保留**（dry-run 输出里没有任何 volume 行）。
down:
	@echo "▶ 停止全部档位的容器（数据卷保留）..."
	$(COMPOSE) $(ALL_PROFILES) down

# 只停 core（stop 而非 down：容器与卷都保留），用于 core 档自身的重启。
# ✅ 实测（2026-09-24，`docker compose --profile core stop --dry-run`）：只会停
#    app / redis / postgres / milvus / minio / etcd 这 **6 个**，其余 6 个（clickhouse、langfuse、
#    langfuse-worker、langfuse-redis、prometheus、grafana）不在列表里。所以 `stop` **是**按档过滤的
#    —— 别被 `ps` 误导：`ps` 不按档过滤（见下方 ps 目标），两者行为**不一致**，这是 compose 的现状。
# ⚠️ 即便它确实只停 core，也**不要**拿它当「切档腾内存」的手段：从 up-tracing 切回 up-obs 时，
#    真正占内存的恰是 tracing 档的 clickhouse / langfuse / langfuse-worker / langfuse-redis 四个，
#    它们一个都不会停。跨档切换请用 `make down`（同样保留数据卷）。
down-core:
	$(COMPOSE) --profile core stop

# ------------------------------------------------------------------------------
# ps —— 查看容器状态（含健康状态与端口）
# ------------------------------------------------------------------------------
# 注意：`ps` **不按 profile 过滤**，加不加 $(ALL_PROFILES) 输出完全一样（实测见文件上方
# ALL_PROFILES 处）。此处带上只为与其他目标写法统一，不带有也不会漏看容器。
ps:
	@$(COMPOSE) $(ALL_PROFILES) ps

# ------------------------------------------------------------------------------
# logs —— 跟踪日志；`make logs SVC=langfuse` 可看任意档的服务
# ------------------------------------------------------------------------------
# ⚠️ 这里**必须**带 $(ALL_PROFILES)（本目标此前漏了，是全仓库唯一一处）。
#    实测（2026-09-24，12 个容器全在跑、COMPOSE_PROFILES 为空），逐个服务试的结果：
#      · 13 个服务里**只有 2 个**失败：langfuse 与 langfuse-worker，均为**退出码 1**，
#        stderr 只给一句 `no such service: minio`
#        —— 一个你没问、却健康跑着的服务名，极易把排查带向 minio（对象存储/桶/凭据）。
#      · 其余 11 个**都成功**（退出码 0），包括同样不在 core 档的
#        clickhouse、langfuse-redis、maxkb、grafana。所以**不是**「不在当前档就失败」。
#    再加一组对照，说明带档确实能救：
#        docker compose logs --tail=0 langfuse                        → rc=1  （上一条的错误）
#        docker compose --profile core logs --tail=0 langfuse         → rc=0
#        docker compose --profile tracing logs --tail=0 langfuse      → rc=15        ← 反而更糟
#        docker compose --profile tracing config -q                   → rc=15  ← 同因，整份项目非法
#        docker compose $(ALL_PROFILES) logs --tail=0 langfuse        → rc=0  ← 本目标采用
#    ⚠️ **别把 rc=15 那条的报错文本当规范抄下来**：它是**不稳定的**。同一条命令连跑 20~30 次，
#       出现且仅出现 4 种组合，各约 3~9 次（实测 20 次：9 / 4 / 4 / 3）：
#           langfuse|langfuse-worker 作主语  ×  postgres|minio 作依赖
#       —— 这**正是两个维度的笛卡尔积**：报错的两个服务各有**两个** core 档依赖
#       （postgres、minio），而「取集合里第一个」这一步遍历的是**无序**集合，两个维度各自随机。
#       对照：`logs` 那条路径的查找是**确定**的，所以同一件事它**稳定**只报 minio
#       （实测 20/20 与 10/10 两次逐字相同）；且它报的是 minio 而**不是**声明顺序更靠前的
#       postgres，说明那也不是按声明顺序取 —— 具体排序规则未深究，只需知道它确定。
#       两者说的其实是同一组跨档依赖，只是一个稳定、一个不稳定。
#       所以任何「报的是 X」的单一样本都不具有规范意义（本注释早期版本犯过这个错）。
#    ✅ **机制已查明（2026-09-24 实测：7 条新预测全中，且为唯一区分性判据）**：
#       compose 对 `logs` 这类命令会**自动启用「本命令目标服务自身所属的那个档」**，
#       但**不会**顺带启用该服务**依赖**所在的档。自动启用本身有两处独立铁证：
#         · `logs langfuse` 报的是 **minio**（langfuse 的依赖），不是 langfuse —— 说明 langfuse
#           明明写着 `profiles: [tracing]` 却**已被启用**，否则报错该是 `no such service: langfuse`；
#         · `logs maxkb`（optional 档、**零依赖**）→ rc=0，同理。
#       于是判据就是「**直接**依赖跨档」：全仓**只有** langfuse / langfuse-worker 有这种边
#       （两者的 depends_on 里都有 core 档的 minio 与 postgres），它们恰好是唯一失败的 2 个；
#       其余 11 个的直接依赖要么同档、要么没有 → rc=0。
#    ⚠️ 本注释早期版本用「app 的 depends_on **闭包**里同样含 minio（app → milvus → minio）」
#       去反驳上面这条 —— 那是**粒度用错了**，不是反例：app 的**直接**依赖是
#       postgres / redis / milvus，**三个全在 core 档**，core 一启用整条链自然都通；
#       minio 对 app 只是**传递**依赖，不构成跨档的**直接**边。
#       同理 `--profile core logs langfuse` → rc=0（core 一进来，那两个依赖就都有了），与机制一致。
#       而 `--profile tracing`（**显式**给档）走的是另一条会校验整份项目的代码路径，直接 rc=15 ——
#       两条路径的错误码不同，不需要用同一个解释去套，但缺的都是同一组跨档依赖。
#    实践结论（这也是唯一需要记住的一条）：本目标一律带上 $(ALL_PROFILES)，
#    就不必区分服务属于哪一档，上面所有不确定性全部绕开。
logs:
	$(COMPOSE) $(ALL_PROFILES) logs -f --tail=200 $(or $(SVC),app)

# ------------------------------------------------------------------------------
# build —— 仅重建应用镜像（改了 Dockerfile 或 requirements.txt 时用）
# ------------------------------------------------------------------------------
build: check-env
	$(COMPOSE) --profile core build --no-cache app

# ------------------------------------------------------------------------------
# pull —— 预拉取全部镜像（首次部署/换机时先跑，把「拉镜像」与「起服务」分开）
# ------------------------------------------------------------------------------
# ⚠️ 必须带 --ignore-buildable，原因是一次实测踩到的坑：
#   compose 里的 app 服务**同时**写了 `image: aligo-travel-app:local` 与 `build:`，
#   它是要本地构建的，registry 上根本没有这个镜像。不带该开关时 compose 会照样去
#   registry 找它，报：
#       pull access denied for aligo-travel-app, repository does not exist ...
#   整条命令以**退出码 18** 结束 —— 而其余镜像其实全都拉好了。
#   （是 11 个：docker-compose.yaml 里共 13 个 `image:` 行，去掉本地构建的 app，再按 tag 去重
#     —— core 的 redis 与 langfuse 的 redis 共用 `redis:7-alpine`，故 12-1=11。
#     此处原写「5 个」，是升级到 ALL_PROFILES 之前只覆盖 core 档时的遗留数字。）
#   这个非零退出码会让人误判成「镜像拉取失败了」从而去查网络。
#   --ignore-buildable 让 compose 跳过所有带 build 段的服务，退出码才如实反映结果。
pull:
	@echo "▶ 预拉取全部档位的镜像（自动跳过需本地构建的 app，见本目标上方注释）..."
	$(COMPOSE) $(ALL_PROFILES) pull --ignore-buildable
	@echo ""
	@echo "✅ 镜像已就绪。下一步： make up"

# ------------------------------------------------------------------------------
# test —— 跑 pytest。不依赖 Docker：健康检查测试用 sqlite 内存库替代 PostgreSQL。
# ------------------------------------------------------------------------------
# PYTHONPATH 只给仓库根（让 `import src.*` 可用），**刻意不再指向 third_party**：
#   agentscope / reme 已装在当前 Python 环境里，直接 import 即可。
#   由安装元数据决定解析结果，比在命令行里硬编码一条路径更稳 —— 硬编码的路径
#   在换机器/进容器后会静默失效。
#   ⚠️ 注意：这不等于"本地与容器必然同代码"。实测各有错位（本机 reme 是 PyPI 的
#   0.4.1.12，容器装的是仓库源码的 0.4.1.13），详见 README「三条硬性约定」第 2、3 条。
test:
	@echo "▶ 运行 pytest（本地模式，无需 Docker）..."
	PYTHONPATH=$(CURDIR) \
		$(PYTHON) -m pytest tests/ -q

# ------------------------------------------------------------------------------
# check-docs —— 校验文档里的两类硬伤：① 行号引用是否指得住；② 写死的计数是否正确
# 为什么需要：本项目文档里满是**手工维护、不会自动更新**的东西，它们失效时文档看起来
#   毫无异常：一类是行号引用（如 `src/server/probes.py:262`），改一次源码就整体漂移；
#   另一类是**写死的数字**（如「N 个用例」「N passed」），加一个测试用例就静默变错。
#   两者都由本目标守：改动 src/ / config/ / tests/ / 文档后请跑一次。
# 两个脚本各管一类，**互不越界**（别指望任何一个去干另一个的活）：
#   scripts/check_doc_refs.py   查「文件在不在 + 行号超没超界」，不看数字；
#   scripts/check_doc_counts.py 查「数字对不对（== 机器实测值）」，不看行号。
# 判定范围（refs）:**自有文件与 vendored 代码都判**——vendored 代码我们不改，但文档引它的
#   行号同样会漂（换 pin 的版本就会），故一并判定；口径与未判定的缺口见该脚本头部注释。
# ⚠️ 与别的多命令目标不同，这里刻意**不用 `set -e` 短路**：两道闸门相互独立，
#   第一道红了也应当把第二道的报告一并打出来，免得修一个跑一次。
#   故用 `|| rc=1` 累计退出码，最后统一以该码退出（任一失败即失败）。
# ------------------------------------------------------------------------------
check-docs:
	@set -u; \
		rc=0; \
		echo "▶ 校验文档中的硬编码行号引用（文件在不在、行号超没超界）..."; \
		$(PYTHON) scripts/check_doc_refs.py || rc=1; \
		echo ""; \
		echo "▶ 校验文档中写死的计数声明（数字是否等于机器实测值）..."; \
		$(PYTHON) scripts/check_doc_counts.py || rc=1; \
		exit $$rc

# ------------------------------------------------------------------------------
# smoke —— 对**已启动**的服务发冒烟请求，验证健康检查与核心接口
# ------------------------------------------------------------------------------
smoke:
	@echo "▶ 对 $(SMOKE_URL) 执行冒烟测试..."
	$(PYTHON) scripts/smoke.py --base-url $(SMOKE_URL)

# ------------------------------------------------------------------------------
# loadtest —— 对**已启动**的服务施加受控并发，量吞吐与尾延迟（p99）
# ------------------------------------------------------------------------------
# 与 smoke 的分工见 scripts/loadtest.py 的模块注释：smoke 各打一发问「活着吗」，
# loadtest 持续打压问「并发下什么表现」—— 后者是前者推不出来的结论。
# ⚠️ 默认只打 /healthz（免鉴权、零 I/O、不调模型），是纯基础设施基线；
#    要压业务端点请显式给参数，例如：
#        make loadtest LOADTEST_ARGS="--endpoint /api/v1/health --requests 5000 --concurrency 50"
#    或直接在命令行跑（本目标只是最常见形态的快捷方式）：
#        python scripts/loadtest.py --base-url $(LOADTEST_URL) --health-only --duration 20
# ⚠️ 服务没起来时**不要**指望它报错退出 —— 它会如实报「失败率 100%」并以退出码 1 结束，
#    那是正确结论（压测一个连不上的端点），不是脚本故障。
loadtest:
	@echo "▶ 对 $(LOADTEST_URL) 施加并发负载（默认 /healthz；越线即失败退出）..."
	$(PYTHON) scripts/loadtest.py --base-url $(LOADTEST_URL) $(LOADTEST_ARGS)

# ------------------------------------------------------------------------------
# concurrency —— 多用户**同时**发**不同**请求，验证互不干扰、全部成功
# ------------------------------------------------------------------------------
# 与 smoke / loadtest 的分工见 scripts/concurrency_test.py 的模块注释：
# smoke 串行各打一发问「能不能用」；loadtest 只发 GET、问「并发下的分布」；
# 本目标问的是「同时用的时候数据会不会串号」—— 前两者都推不出这一条。
# ⚠️ 它会**真的调模型**、真的建智能体与会话（三个用户同时），因此比 loadtest 贵。
#    跑完自己删掉（与 smoke 同一条纪律）；要留现场用 CONCURRENCY_ARGS="--keep"。
# ⚠️ 用户是 concurrency-user-1..N 这些**一次性工作区**，不会碰 alice 的数据。
#    若其中某个用户已存在 main_plan 智能体，本脚本复用它而不删它。
concurrency:
	@echo "▶ 对 $(CONCURRENCY_URL) 执行并发正确性测试（3 个用户同时发不同请求）..."
	$(PYTHON) scripts/concurrency_test.py --base-url $(CONCURRENCY_URL) $(CONCURRENCY_ARGS)

# ------------------------------------------------------------------------------
# eval —— 对黄金数据集跑离线评测，产出可 diff 的评测报告（**不依赖 Docker**）
# ------------------------------------------------------------------------------
# 与 `make test` 的分工见 scripts/eval.py 的模块注释：test 回答「代码逻辑对不对」，
# eval 回答「改动之后系统整体是变好还是变差了」。两者都需要能在任何机器上直接跑。
# ⚠️ 未配置模型密钥时会自动退化为**规则判分**（只比车道/意图集/工具集），
#    报告里会如实标注 judge_mode=rule —— 它不会假装做过语义评估。
# 阈值与输出路径可用变量覆盖： make eval MIN_SCORE=0.9 EVAL_OUTPUT=/tmp/r.json
eval:
	@echo "▶ 对黄金数据集执行离线评测（阈值 $(MIN_SCORE)）..."
	$(PYTHON) scripts/eval.py --min-score $(MIN_SCORE) --output $(EVAL_OUTPUT)

# ------------------------------------------------------------------------------
# milvus_init —— 幂等建 Milvus 集合，并**回读核验**其形态（**只建集合、不灌数据**）
# ------------------------------------------------------------------------------
# 与 seed_data 的分工见 scripts/milvus_init.py 的模块注释：init 负责「集合建对了吗」，
# seed_data 负责「库里有没有内容」。分开是为了让「集合参数以谁为准」只有一个答案 ——
# 若不建集合也算 seed_data 的活，两处都会写 create_collection，而它们的维度/索引
# 一旦不一致，先跑的那个静默赢（create_collection 已存在即 no-op），没人会发现。
# ⚠️ 建的是**两个**集合：{MILVUS__COLLECTION}（政策知识库）与它派生出的
#    {MILVUS__COLLECTION}_memory（长期记忆画像）。后者曾漏建，导致「记住这个」
#    在全新部署上直接失败（2026-10-03 实测，见 scripts/milvus_init.py 模块注释）。
# ⚠️ 因此灌数据前请先跑本目标；脚本连不上 Milvus 时会以非零码退出并说明原因。
milvus_init:
	@echo "▶ 初始化 Milvus 集合并回读核验（幂等）..."
	$(PYTHON) scripts/milvus_init.py

# ------------------------------------------------------------------------------
# seed_data —— 构造并灌入差旅政策文档与业务演示数据（幂等）
# ------------------------------------------------------------------------------
# ⚠️ 需要 Milvus 与 PostgreSQL 可达（core 档含它们：make up）。连不上时脚本会打印
# 清晰的中文错误与下一步，而不是裸 traceback。
# 幂等：重复跑不产生重复数据（Milvus 侧幂等键是 document_id = seed-<doc_id>，
# 业务库侧是各表主键）。不加任何参数即真的写库；只预览不写库用：
#     make seed_data ARGS="--dry-run"
# dry-run **不连任何外部服务**，可在没有 Docker 的机器上跑通。
seed_data:
	@echo "▶ 灌入种子数据（幂等；只预览加 ARGS=\"--dry-run\"）..."
	$(PYTHON) scripts/seed_data.py $(ARGS)

# ------------------------------------------------------------------------------
# provision-agent —— 给一个用户预置 AliGo 主智能体（幂等）
# ------------------------------------------------------------------------------
# ⚠️ 为什么它必须存在：`make up` 全绿 + `make seed_data` 灌完数据之后，
#    浏览器里的输入框**仍然是灰的**。框架前端的输入框由会话的
#    `chat_model_config` 解禁，而会话必须挂在智能体下 —— 没有智能体，
#    页面上只有一句「请先选择一个智能体」，而下拉里一个都没有。
#    seed_data 灌的是**数据**（政策库/画像），不建智能体；两者分工不同，
#    所以这里是独立的一步，而不是塞进 seed_data。
provision-agent:
	@echo "▶ 预置智能体（用户 $(PROVISION_USER)，幂等）..."
	$(PYTHON) scripts/provision_agent.py --user $(PROVISION_USER)

# ------------------------------------------------------------------------------
# clean —— ⚠️ 停止服务并删除全部数据卷（数据库、向量、日志全部清空）
# ------------------------------------------------------------------------------
# ⚠️ $(ALL_PROFILES) 同样是**必需**的，理由与 down 完全相同（`down` 按档过滤，漏档 = 静默
#    什么都不做，却照样打印下面的「✅ 已清理」。那是最坏的一种假绿灯：以为清干净了，
#    其实卷一个没删，下次启动看到的还是旧数据）。改动本行前请用
#    `docker compose <四档> down -v --dry-run` 核对。
clean:
	@echo "⚠️  即将删除全部数据卷（数据库/向量库/日志将被清空），5 秒内可 Ctrl-C 取消..."
	@sleep 5
	$(COMPOSE) $(ALL_PROFILES) down -v
	@echo "✅ 已清理。"

# clean-gen —— 清理本地构建产物与缓存（不动容器与卷，也**不碰 third_party/**）
# ------------------------------------------------------------------------------
# 两条约束都是被真实隐患逼出来的，不是洁癖：
#   1) 路径锚到**本 Makefile 所在目录**（与 preflight 同理）。下面两条原本是相对 CWD 的，
#      若执行时 CWD 不在仓库根（`make -f 绝对路径/Makefile`、某些 IDE 的 make 集成），
#      它们删的就是**调用者所在目录**里的东西 —— 在 $HOME 下跑一次就会静默递归删掉
#      $HOME 子树里所有 __pycache__，而执行者的预期是「清本仓库的缓存」。
#   2) `-path ./third_party -prune`：third_party/ 是 vendored 源码树，README 与 docs/01
#      都把它定为「只读、不改动」。不豁免时实测 126 个 __pycache__ 里有 **118 个在
#      third_party/ 下** —— 等于每次「清缓存」都在那棵树里写改动，会被误判成
#      「有人改了 vendored 源码」（进而怀疑踩了 README 里「PyPI 同名不同码」的坑）。
clean-gen:
	@set -eu -o pipefail; \
	here="$(dir $(lastword $(MAKEFILE_LIST)))"; \
	cd "$$here"; \
	find . -path ./third_party -prune -o -type d -name __pycache__ -prune -exec rm -rf {} +; \
	rm -rf .pytest_cache .coverage htmlcov; \
	echo "✅ 已清理本地缓存（已跳过 third_party/）。"

# 兼容别名：一次性全清
clean-all: clean clean-gen
