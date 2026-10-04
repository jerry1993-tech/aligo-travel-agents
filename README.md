# AliGo 差旅助手 — 企业级多智能体差旅助手

基于 [AgentScope](https://agentscope.io/blog/alibaba-business-travel/) 构建的多智能体差旅助手。
把「出差事项收集 → 行程规划 → 一键下单」做成一条可流式观测的智能体链路：
独立意图识别、快慢车道分流、动态 Prompt 组装、政策知识库检索、长期画像，全部通过
**直接 import 本地已安装的 `agentscope` 包**调用框架能力实现 —— 不自行重造框架已有的轮子。

> **本 README 的读者**：接手这个仓库的开发者。
> 它重点回答两件事：**怎么把它跑起来**，以及**哪些坑是踩过的、不要再踩一遍**。
> 不写"项目简介"式的空话 —— 那些在 `docs/` 里。

---

## 目录

- [一、当前状态（先看这个）](#一当前状态先看这个)
- [二、三条硬性约定](#二三条硬性约定)
- [三、前置依赖](#三前置依赖)
- [四、快速开始](#四快速开始)
- [五、架构总览](#五架构总览)
- [六、目录结构](#六目录结构)
- [七、配置体系](#七配置体系)
- [八、端口与档位](#八端口与档位)
- [九、零密钥降级](#九零密钥降级)
- [十、Embedding 三档](#十embedding-三档)
- [十一、可观测](#十一可观测)
- [十二、测试与闸门](#十二测试与闸门)
- [十三、已知限制](#十三已知限制)

---

## 一、当前状态（先看这个）

**分阶段交付，每个阶段独立可验证。** 当前处于 **P1–P5 均已完成**
（P5 的最后一项「Langfuse 里有完整 trace」于 2026-10-03 对 12 容器 tracing 档实测通过，见下方「已验证」）。

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| **P1** | 骨架与闸门：配置加载、根应用装配、健康检查三件套、LLM 工厂与熔断、可观测基座、两个文档校验脚本 | ✅ 已完成 |
| **P2** | 会话 + SSE 流式 + 鉴权中间件链 + `/api/v1/**` 业务路由骨架 + 业务库引擎 | ✅ 已完成 |
| **P3** | 意图识别 / 快慢车道 / 动态 Prompt / 差旅工具集 / 思考链 | ✅ 已完成 |
| **P4** | Milvus + 单集合知识库 + Embedding 三合一 + 长期画像 | ✅ 已完成 |
| **P5** | 全链路 trace + 评测集 + 前端 + `docs/` 全套 | ✅ 已完成 |

**P2 的验收线**（`tests/test_e2e_stream.py` 逐条钉住）：

- 一句话端到端流式返回：`POST /chat/` → `GET /sessions/{id}/stream` 收到
  `REPLY_START` + 多个 `TEXT_BLOCK_DELTA`，以 `finished_reason="completed"` 收尾；
- 空闲连接每 30 秒收到心跳帧 `:\n\n`（用例把间隔压到 0.25 秒来验证）；
- 两个不同 `X-User-ID` 互相看不到对方的会话（`/messages` `/stream` `/status` 全 404）。

**这意味着**：现在 `make up` 起来的是一个**能跑通、能被探活、能被观测、能对话并看到逐字流式输出**的服务。
聊天链路（`/chat/`、`/sessions/**`）由框架自带并已随根应用暴露，鉴权与限流由我们的中间件补在前面。
P3/P4 之后它背后**已经有差旅业务能力** —— 快慢车道分流、动态 Prompt、7 个差旅工具、
单集合政策知识库、长期画像与思考链；P5 补齐了 trace 面板、评测集与前端。
关于"哪些代码已经存在"，以 `src/` 下的实际文件为准，不要以本文档的阶段表反推。

**已验证 / 未验证**：

- ✅ `make test` 绿（**2025 个用例，不需要 Docker**，用 sqlite 内存库；实测 `2025 passed in 132.74s`）
- ✅ P2 的流式验收**对真实 uvicorn + 真实 TCP** 跑过（不是 `httpx.ASGITransport` ——
  它会缓冲整个响应体，SSE 在它上面一个字节都读不到，见 `tests/test_e2e_stream.py` 的模块文档字符串）
- ✅ `make check-docs` 绿（两道闸门都通过；具体检查了多少条由脚本自己打印 ——
  这里**刻意不写死那个条数**，因为它每次改文档都会变，写了就必然漂）
- ✅ 零密钥（不配任何 API key）下服务能正常启动并让 `/healthz` `/readyz` 通过
- ✅ 观测栈（Prometheus/Grafana）的配置已对着真实容器验证过：Prometheus 配置过了 `promtool`，Grafana 的面板与数据源在 Grafana 11.1.0 里能正常加载
- ✅ `make smoke` 对**真实核心栈**（`make up-core` 的 6 个容器：app / postgres / redis / milvus / etcd / minio）
  **8/8 全过** —— 含一轮**真实的端到端对话**（配了 DashScope key，回复 45 字符正常收尾）。
- ✅ `make smoke` 对**完整非可选栈**（`make up-tracing` 的 12 容器 = 上面这 6 个核心 + 追踪档 4 个
  （langfuse / langfuse-worker / clickhouse / langfuse-redis）+ 观测档 2 个（prometheus / grafana））
  同样 **8/8 全过**，且那一轮对话在 Langfuse 里留下了**完整 trace**：
  `invoke_agent` 根 span（21.3s）+ 2 个 `GENERATION`（`chat qwen3.8-flash`，2.9s / 1.6s）+
  1 个 `TOOL`（`aligo_route_intent`）—— 即「模型调用与工具调用都被埋点」，
  而不只是一根孤立的 span。第 13 个服务 maxkb 属可选档（`make up-optional`，默认不起），
  不被 smoke 的任何一项触达。
- ✅ **检索护栏**（2026-10-03 加的 P4 加固）：向量库的**读**操作有 5s 截止时间与熔断
  （连续 3 次下游失败开路 30s），写路径刻意不受限；`/readyz` 里
  `vector_store_breaker` 与 `breaker`（模型）**并列**暴露，`/metrics` 里有
  `aligo_model_breaker_state{name="vector_store"}`。起因是一次真实故障 ——
  Milvus 丢 etcd 租约退出，而框架的 `search` 没有任何超时，于是对话请求**永不返回**；
  完整经过、观测点与恢复步骤见 `docs/06-部署与运维.md` 第十节。
- ✅ 全部 13 个服务都带 `restart: unless-stopped`（同一次故障的第二个教训：
  compose 默认 `restart: no` 会让崩掉的 Milvus **永远不回来**，
  后续每次冒烟都失败且失败信息指向「连不上 Milvus」）。已用
  `docker inspect` 对 12 个运行中的容器逐个核验过策略确实生效。
- ✅ 前端已在 `web/frontend/` 下二次开发完成并**已构建进 `src/server/static/`**
  （vite 的 `outDir` 直指该目录，见 `web/frontend/vite.config.ts`；`pnpm build` 退出码 0，
  产物 391 个 asset / 15 MB，index.html 的 `<title>` 为 `AliGo 差旅助手`）。
  差旅专属的工具卡片渲染器（路由决策/交通/酒店/差标/订单/申请）在
  `web/frontend/src/components/chat/tool-renderers/aligo/`，用法是**展开合并**进渲染器表。
  事件类型**不重抄**：前端 `import { EventType } from '@agentscope-ai/agentscope/event'`，
  用的是包自己那份枚举 —— 于是"前端类型与框架的 **28** 种 `EventType` 是否对齐"这个问题
  由**构造方式**回答，不由人工比对回答（`tsc -b` 通过即为对齐；它一旦对不上，
  编译期就红，而不是等到某个事件在运行时静默漏渲染）。
  静态托管是**条件式**的（目录/产物存在才挂到 `/`），见 `src/server/app.py:_mount_static_if_present`
- ✅ **长期记忆的显式读写面 `/api/v1/memory/**`**（2026-10-03 补上）：`GET/PUT …/profile`
  与 `POST/GET/DELETE …/notes` 共五个端点。补之前 `remember` / `update_profile` 在 `src/` 里
  **一个调用者都没有**（7 个差旅工具与全部业务路由里都没有记忆写入），
  而 `src/server/constants.py` 的 `MEMORY_ATTR`、`TravelerMemory.get_profile` 的 docstring
  与 `docs/03` 已经三处把它当成存在的东西在描述 —— 现在三处都是真的。
  身份只来自鉴权（请求体里带 `user_id` 一律 422），写失败从**不**静默丢弃
  （503 且明说「没有生效」，原因经 `safe_error` 脱敏），召回失败是 `200 + error`
  而不是空列表。接口契约与三种 503 的区分见 `docs/01-功能接口.md` 2.5。

---

## 二、三条硬性约定

这三条被 `Makefile`、`pytest.ini`、`Dockerfile` 的注释多处引用（搜「三条硬性约定」）。
**它们不是偏好，是踩过之后定下来的边界。**

### 约定 1：`third_party/` 只读，不修改

`third_party/agentscope`（`src/agentscope`，version `2.0.10dev`）与 `third_party/ReMe`（`reme`，version `0.4.1.13`）
是两份 **vendored 源码树**。它们同时扮演两个角色：

1. **本机 Python 环境的安装来源** —— `import agentscope` 直接解析到 `third_party/agentscope/src/agentscope`（editable 安装）；
2. **容器镜像的安装来源** —— `Dockerfile:54-56` 把它们 `COPY` 进构建阶段，用 `pip install --no-deps` 装进 `/opt/venv`。

**约定**：不在这两棵树里改任何一行。要覆写上游行为，用 **middleware / 继承 / 组合** 在自己的代码里做。
理由不只是"保持干净"：`make clean-gen` 会对 `third_party/` 做 prune（见 `Makefile:472`），
而如果那棵树里有我们的改动，**每次清缓存都像有人动过 vendored 源码** —— 排查方向会被彻底带偏。

### 约定 2：**本机跑的不等于容器里跑的**，且两者都不保证是同一份

这是最容易误判的一条。`agentscope` 两边一致（本机是 editable 安装，`.pth` 恰好也指向 `third_party/agentscope/src`），
但 **`reme` 不一致**：

| | 版本 | 来源 |
| --- | --- | --- |
| 本机 Python 环境 | `0.4.1.12` | PyPI 的 `reme_ai` 发行版，装在 `site-packages/reme` |
| 容器 `/opt/venv` | `0.4.1.13` | `third_party/ReMe/reme`（仓库里的源码） |

实测两份源码有 **24 处 `.py` 文件差异**（20 个两边都有但内容不同 + 4 个只存在于一边）。
这是个可复现的事实，不是估计：

```bash
# 本机版本
python -c "import reme; print(reme.__version__, reme.__file__)"
# 容器版本
docker compose exec app python -c "import reme; print(reme.__version__, reme.__file__)"
# 差异数（忽略 __pycache__）
diff -rq --exclude='__pycache__' third_party/ReMe/reme \
  "$(python -c 'import reme,os;print(os.path.dirname(reme.__file__))')" | wc -l
```

**后果**：一个"本机测试全绿、容器里行为不同"的 bug，**可能是环境差异而不是代码 bug**。
排查这类问题前，先确认两边跑的是不是同一份代码。

### 约定 3：`PYTHONPATH` 只放仓库根，不硬编码 `third_party`

`pytest.ini` 的 `pythonpath` 与 `Makefile:414` 都只加仓库根（让 `import src.*` 可用），
**刻意不把 `third_party/*/src` 写进去**。

理由是约定 2 的另一面：靠**安装元数据**决定 `import agentscope` 解析到哪儿，
比在配置里硬编码一条路径更稳 —— 硬编码的路径只在"仓库就地跑"时成立，
换台机器或进容器立刻失效，而失效形式是一句很难定位的 `ImportError`。

⚠️ 但请**不要**因此推断"既然都是靠安装元数据，那两边必然同代码" —— 约定 2 已经证伪了这一点。

---

## 三、前置依赖

| 依赖 | 版本 | 说明 |
| --- | --- | --- |
| Docker | 含 Compose v2 | `make` 的绝大多数目标都走 `docker compose`。实测记录基于 Compose `v2.31.0-desktop.2` |
| Python | 3.11 | 容器基镜像是 `python:3.11-slim`；本机需一个**装了 `agentscope` 与 `reme`** 的环境才能跑 `make test` |
| GNU Make | 任意 | 入口 |
| Node.js | ≥ 18 | **仅 P5 前端需要**，P1 用不到 |

⚠️ **本机目前未装 `pnpm`**（有 `node v24.14.1`）。P5 开始前端之前需要先
`corepack enable pnpm` 或 `npm i -g pnpm`。

⚠️ **磁盘**：全套镜像约 10 GB。若空间紧张，`docker system prune` 通常能回收数 GB 构建缓存。

---

## 四、快速开始

```bash
# 1) 准备环境变量（首次）
cp .env.example .env
#    必须改掉其中的口令类占位值（POSTGRES_PASSWORD / REDIS_PASSWORD / MINIO_ROOT_PASSWORD / ...）
#    ⚠️ 但**不要**把 ALIGO__IMAGE_REGISTRY 写进 .env —— 理由见第七节

# 2) 起全栈（core 档：postgres / redis / etcd / minio / milvus / app）
make up

# 3) 首次还要建 Milvus 集合并灌演示数据（幂等，可重复跑）
#    ⚠️ Milvus 首启**不会**自动建集合；本步建**两个**：政策知识库 + 长期记忆画像。
#    跳过它的话：政策检索会失败，长期记忆的语义召回每轮打一条 collection not found
make milvus_init
make seed_data                   # 灌政策文档与业务演示数据（不建集合）

# 4) 给要用的身份预置智能体（幂等）
#    ⚠️ 不做这一步，浏览器里的输入框是**灰的、点不动** —— 见下面「第一次打开网页」
make provision-agent                       # 默认给 alice
make provision-agent PROVISION_USER=bob    # 换一个身份

# 5) 验证
curl -s localhost:8000/healthz   # {"status":"ok"}
curl -s localhost:8000/readyz    # 各依赖的真实连通性
curl -s localhost:8000/metrics   # Prometheus 文本
make smoke                       # 端到端冒烟，退出码 0 = 通过

# 6) 日常开发
make test                        # 单测，不需要 Docker
make check-docs                  # 文档闸门（行号引用 + 计数断言）
make help                        # 全部命令（不记得别的目标时就敲这个）
```

> ⚠️ **`make up` 成功 ≠ 集合已建**：`/readyz` 只探 Milvus 的 TCP 连通性，
> 不探集合是否存在（见 `docs/06-部署与运维.md` 第九节）。

### 第一次打开网页：网址、身份、以及「为什么输入框是灰的」

网址是 **<http://localhost:8000/>** —— 前端产物由 app 容器自己托管，没有第二个端口。

第一次打开会看到「连接到服务器」引导页，两个输入框：**服务器地址**填
`http://localhost:8000`（要带 `http://`，这一栏不会自动填当前地址），**用户名**填
`alice`（第 4 步预置的那个身份）。点「开始使用」后前端会先请求一次 `/healthz`
验证地址可达，通过才把这两项存进浏览器 localStorage。

> ⚠️ 这里的「用户名」**不是账号，也没有密码**：它只是每次请求带上的一个
> `X-User-ID`。任何能给这个端口发请求的人，填 `alice` 就是 `alice`。
> 所以这套部署只适合本机或纯内网演示，不要直接挂到公网。
>
> ⚠️ **大小写敏感**：`alice` 与 `Alice` 是**两个互不相通的工作区**（身份只做
> `strip()`，从不 `lower()`）。本机实测这两者各自有一个独立的 `main_plan`、
> 各自的会话。所以「我明明预置过了，怎么还是灰的」常见原因就是**登录时首字母
> 打成了大写** —— 此时 `make provision-agent` 补的是另一个工作区。

**「输入框点不动」是这个项目最容易卡住人的一处**，它不是故障，而是一条硬性前置条件：

```text
有智能体  →  有会话  →  有模型  →  输入框解禁
```

框架前端的输入框由「当前会话的 `chat_model_config`」决定启用与否，而会话必须挂在
智能体下。全新部署里 `alice` 一个智能体都没有，页面上只有一句「请先选择一个智能体。」，
下拉框却是空的 —— 于是**没有任何可点的东西**。第 4 步的 `make provision-agent`
就是为了消掉这一步。

> ⚠️ 智能体名必须是 **`main_plan`**（脚本里取的 `MAIN_AGENT_NAME` 常量）。
> 自己在前端点「+」随手建一个叫「我的助手」的智能体也能聊天，但快慢车道
> （`LaneRouterMiddleware`）与动态 Prompt 注入都**按 `agent.name` 收窄**，
> 于是这些能力静默不生效 —— 界面上完全看不出区别。
> 要自己建的话，名字请填 `main_plan`，提示词可留空（留空则用兜底提示词）。

**加观测栈**（Prometheus + Grafana，浏览器看指标）：

```bash
make up-obs        # core + prometheus + grafana
# Grafana:    http://localhost:3000   （面板在 AliGo 目录下，已自动配好数据源）
# Prometheus: http://localhost:9090
```

**加全链路 trace**（Langfuse v3，另起 clickhouse 等 4 个容器，**内存占用最高**的一档）：

```bash
make down          # ⚠️ 切档前先 down。up-tracing 会重建 app 容器
make up-tracing
# Langfuse: http://localhost:3001
```

> ⚠️ 本机 Docker VM 内存约 3.82 GiB。core 与观测栈同跑时优先用 `make up-obs`（只加指标），
> 确认有余量再加 Langfuse。用 `docker stats` 看真实占用。

---

## 五、架构总览

```text
浏览器（前端，P5；基于 examples/web_ui 二次开发）
   │  X-User-ID  或  Bearer JWT（由我们的 ASGI 中间件解析后注入 X-User-ID）
   ▼
┌────────────────────────────────────────────────────────────────────┐
│ src/server/app.py   模块级 app —— 入口是 uvicorn src.server.app:app │
│                                                                    │
│   app = create_app(...)     ← 直接复用返回值作根应用，**绝不 mount** │
│   ├ 框架自带并已暴露：/chat/  /sessions/**(SSE)  /agent             │
│   │                    /knowledge_bases  /mcp  /schedule  /workspace│
│   │                    /hub  + HITL / 权限 / 多租户                 │
│   ├ 我们追加：/healthz  /readyz  /metrics                          │
│   ├ 我们追加：/api/v1/**  业务命名空间（与框架路由**并行**，不重叠）  │
│   ├ 我们追加：中间件 TraceContext → HttpMetrics → Auth → RateLimit   │
│   └ 最后：mount("/", StaticFiles(...))  前端产物（**存在才挂**）     │
└────────────────────────────────────────────────────────────────────┘
   │                    │                    │
   ▼                    ▼                    ▼
PostgreSQL           Redis               Milvus
business schema      RedisMessageBus     单集合政策知识库
+ AgentScope 表      (SSE/锁/取消广播)    + 画像语义召回
```

⚠️ **框架路由在根路径上、不带前缀**（`/chat/` 而不是 `/api/v1/chat/`），这是既定契约：
官方前端与 `scripts/smoke.py` 都按根路径调用它。`/api/v1/**` 是**并行**的另一个命名空间，
不是给框架路由套的外壳 —— 给框架的 `include_router` 加上前缀会让 `/chat/` 直接消失。
`tests/test_api_v1.py::test_business_routes_do_not_shadow_framework_routes` 钉住了这一点。

### 为什么"复用 `create_app` 的返回值"而不是 `mount` —— 一个已核实的致命陷阱

`create_app` 把 lifespan 挂在它自己创建的 FastAPI 实例上（`app/_app.py:292`），
而 **Starlette 的 `mount()` 不会触发子应用的 lifespan**。AgentScope 作者本人在健康检查的
docstring 里就把这件事写成了警告。

若按官方 docstring 的 `root.mount("/agentscope", agentscope_app)` 写：

1. `chat_service` / `session_service` / `scheduler_manager` **全在 lifespan 里**才写入 `app.state`，
   于是取 `app.state.chat_service` 会 `AttributeError` ⇒ **所有业务端点 500**；
2. 前缀 mount 还会打断官方前端（它用 `new URL(path, baseUrl)` 且调用处传绝对路径，前缀会被丢掉）。

**对策**：`app = create_app(...)` 的返回值**直接作根应用**，再 `include_router` /
`add_middleware` / 包一层 lifespan。三个好处：lifespan 天然执行；无前缀问题；
`StaticFiles` 最后 mount 到 `/` 不会遮蔽 API。

### 另一处陷阱：中间件顺序与直觉**相反**

`Starlette.add_middleware` 内部是 `user_middleware.insert(0, ...)`，而构建时按
`reversed(...)` 迭代 ⇒ **列表里越靠后 = 越外层**。所以本项目的写法是：

```python
for middleware in [Middleware(RateLimitMiddleware, settings=settings),  # 先加 ⇒ 最内层
                   Middleware(AuthMiddleware, settings=settings),
                   Middleware(HttpMetricsMiddleware),
                   Middleware(TraceContextMiddleware)]:                 # 后加 ⇒ 最外层
    app.add_middleware(middleware.cls, **middleware.kwargs)
```

目标链（由外到内）：`TraceContext → HttpMetrics → Auth → RateLimit → 应用`。

四条各有各的理由，其中两条特别容易搞错：

- **`HttpMetrics` 在 `Auth` 之外** —— 放里面的话，被 401/429 拒绝的请求不进指标，
  而鉴权失败率与限流率恰恰是最需要看板的两条曲线。更糟的是它看起来完全正常：
  有流量、有延迟，只是永远看不到被拒的那部分。
- **`RateLimit` 在 `Auth` 之内** —— 它要读 `Auth` 写进 `scope["state"]` 的身份当限流键。
  顺序反了不会报错，只会让所有请求退化成按客户端 IP 限流，多人共用出口 IP 时互相误伤。

外层是 `TraceContextMiddleware` 才能保证**所有**日志与指标都带上 trace id。
这处曾经写反过（trace id 在某些路径上丢失），`tests/test_server_assembly.py` 现在把这个顺序钉住了；
完整的四元素链条由 `tests/test_api_v1.py::test_middleware_stack_order` 逐字断言。

### 鉴权：中间件**注入** `X-User-ID`，而不是替换框架的 `get_current_user_id`

框架的 `get_current_user_id` 自称是临时方案（`deps.py`：「will be replaced by JWT auth.」），
但它在 **13 个路由文件、85 个 `Depends` 点**被使用。替换它意味着把这 85 处全改掉，
且框架每次升级都可能加新的调用点 —— **漏掉的那一处就是一个没有鉴权的口子**，且不会报错。

我们的做法是「一处修改、全链路生效」：中间件解析凭据后，**先删光客户端自带的
`x-user-id` 再注入**可信值，框架读到的仍是它熟悉的那个头。

> ⚠️ 「先删光」不是洁癖。Starlette 的 `Headers` 取**第一个**匹配项，
> 只追加不删除的话，客户端伪造的 `X-User-ID` 会排在前面并**胜出** ——
> JWT 通道的全部安全价值就在这一行 `remove_header` 上。

两条通道由 `ALIGO__AUTH__JWT_ENABLED` 切换：

| 通道 | 配置 | 凭据 | 校验 |
| --- | --- | --- | --- |
| `X-User-ID` 直连（默认） | `jwt_enabled=false` | `X-User-ID: alice` | **不校验真伪**，适合本地与内网可信环境 |
| JWT Bearer | `jwt_enabled=true` | `Authorization: Bearer <JWT>` | 签名 / 有效期 / 受众 / 签发者，取 `sub` 当身份 |

本机没有 IdP，所以仓库里带了一个 dev 签发脚本：

```bash
python scripts/mint_token.py --user alice --ttl 600    # 打印 token 与可直接粘贴的 curl
TOKEN=$(python scripts/mint_token.py --user alice --raw)
```

它有几条硬性约束：**生产环境拒签**（没有 `--force`）、**保留身份拒签**（`--user aligo-system`
同样直接退出——它是系统凭据的属主，中间件对声明它的请求一律 403，签出来也只是一枚注定被拒的
token，而脚本的自检只跑解码、抓不到这种情况）、**绝不打印密钥**（只打印 sha256 指纹前缀）、
以及签发后**用运行时的 `AuthMiddleware._decode` 回验一次** ——
自己签、自己验是循环论证，只有走运行时那段代码才能证明「签出来的东西服务端真的收」。
见 `tests/test_mint_token.py`。

---

## 六、目录结构

```text
src/
├── server/          # app.py(入口) + probes.py + middleware/ + static/(前端产物，P5)
├── config/          # loader.py + schema.py —— 配置的唯一真值入口
├── llm/             # factory.py(模型装配) + breaker.py(熔断) + mock.py(零密钥降级)
├── observability/   # tracing.py + metrics.py + logging.py + context.py(trace id)
├── agents/          # 各子智能体 + 版本化注册表                             (P3)
├── orchestration/   # 快慢车道 / 路由交接 / 上下文栈 / 动态 Prompt           (P3)
├── tools/           # 交通·酒店·价格·合规·下单（FunctionTool）              (P3)
├── knowledge/       # Milvus 接入 + 单集合 KBManager + 混合检索 + 检索护栏    (P4)
├── memory/          # 短期会话 + 自研画像(Postgres+Milvus) + ReMe 可选适配   (P4)
├── chains/          # 思考链（消费 agentscope.event 的 28 种事件）          (P3)
├── domain/          # 实体 / 仓储 / 业务规则引擎                            (P2)
└── storage/         # 业务库 engine + alembic（business schema）            (P2)

config/              # base.yaml + dev/test/prod.yaml（全中文注释）
scripts/             # smoke / check_doc_* / postgres 初始化 / prometheus / grafana
tests/               # 单测（不依赖 Docker）
docs/                # 编号文档（01 功能接口 / 02 技术架构 / 03 模块关系 / 06 部署运维）+ 实施计划.md + 博客原文
web/                 # 前端（基于 examples/web_ui 二次开发）
```

`src/` 下标注 `(P2)`–`(P5)` 的目录是**按阶段填入**的，P1–P5 的代码均已就位。
以 `src/` 下的**实际代码**为准，不要以目录是否存在反推功能。

---

## 七、配置体系

### 优先级（从低到高）

```text
config/base.yaml                    全环境默认值
  ↓ 深度合并
config/{env}.yaml                   env 由 ALIGO__APP__ENV 决定，默认 dev
  ↓ 逐键覆盖
ALIGO__<段>__<键> 环境变量           例：ALIGO__LLM__MODEL=qwen3.8-flash
```

- **段名与键名都是大写**，用 `__`（双下划线）分隔，例如 `ALIGO__DB__POOL_SIZE`。
- **未知键直接报错**（pydantic `extra_forbidden`），不会静默忽略。
  这条是刻意的：配置写错时立刻崩，比带着一份"以为生效了"的默认值跑起来强。
- `${VAR}` / `${VAR:-default}` 可在 YAML 值里做展开（用于 compose 侧或需要默认值的场合）。
  百炼的端点就是这么写的 —— `base_url: "${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"`，
  于是 `.env` 里**不需要**出现 `DASHSCOPE_BASE_URL`，而需要指向代理时又能覆盖它。

### 模型 Provider：两档可选

`llm.provider` 决定用框架的哪个模型类。**provider 与 model 是一组值，必须成对**：

| provider | 框架模型类 | 默认 model | 密钥变量 | 什么时候用 |
| --- | --- | --- | --- | --- |
| `dashscope`（默认） | `DashScopeChatModel` | `qwen3.8-flash` | `DASHSCOPE_API_KEY` | 阿里云百炼。原生档，有 `thinking_enable` / `thinking_budget` / `top_k` |
| `openai` | `OpenAIChatModel` | 无默认 | 需在 `.env` 覆盖 `ALIGO__LLM__API_KEY` | 任何 OpenAI 兼容端点（vLLM / SGLang / 网关 / 其它厂商） |

选 `dashscope` 而不是"拿 OpenAI 兼容协议打百炼"，是因为**原生档多出的
`thinking_enable` / `thinking_budget` 正是 P3「显示推理」的基础** ——
兼容协议档拿不到这些字段，走它等于把思考链这条路提前堵死。

> ⚠️ 两档的 `Parameters` 字段并不完全相同，但 `max_tokens` / `temperature` 是公共子集，
> 因此 `build_chat_model` 用 `model_cls.Parameters(...)` 走同一段构造，不必分支。
> 把 provider 配成 `openai` 却填 `qwen*` 的模型名会得到 404 ——
> 错误信息看起来像密钥或额度问题，实际是这一对没配齐。

### 45 个 `ALIGO__` 键

`.env.example` 里列了 **45** 个生效的 `ALIGO__` 键。它们必须与 `src/config/schema.py` 的字段一一对应 ——
**`.env` 里多一个 schema 不认识的键，容器就会启动即崩**，而不是忽略它。

> 这条不是一个「请注意」式的提醒，而是有机器化的守卫：用例
> `tests/test_config_loader.py::test_env_example_keys_are_all_accepted_by_the_loader`
> 会把 `.env.example` 里每个生效的键喂给真正的 `load_settings`，有一个对不上就红。
> 数量本身也由 `make check-docs` 的计数闸门看住。
> ⚠️ 被 `#` 注释掉的示例（例如停用的 `# LLM_MODEL=…`）不算数 —— 它们不会进入容器环境。

### 🔴 `ALIGO__IMAGE_REGISTRY` 绝不能写进 `.env`

本机实测：写进去会让 app 容器**崩溃重启**。原因是 `.env` 被两个消费者读取：

1. **compose 自身** —— 用于镜像名的 `${...}` 插值；
2. **app 容器** —— 因为 app 服务经公共片段继承了 `env_file: .env`。

而 app 的配置加载器对 `ALIGO__` 前缀做严格校验，它没有 `image_registry` 这个设置项，
于是启动即抛 `extra_forbidden` 并进入 `Restarting (1)` 循环。

**正确用法是把前缀放在命令行上**（只影响 make 进程与其子进程 compose，不进容器）：

```bash
ALIGO__IMAGE_REGISTRY=docker.m.daocloud.io/ make up
```

> 为什么需要它：部分镜像（如 `milvusdb/milvus`）只发布在 Docker Hub，而某些环境
> `registry-1.docker.io` 会时通时断。给它一个镜像站前缀即可整体改走该站，无需 `docker tag`。
> 注意 `~/.docker/daemon.json` 的 `registry-mirrors` 在本机**实测无效**。
> 作用范围与 9 个受影响镜像的完整说明见 `.env.example` 的「镜像源前缀」段。

### `.env` 与 `.env.example`

`.gitignore` 忽略 `.env` / `.env.*`，但用 `!.env.example` **显式放行**模板。
所以 `.env.example` 是**注定要提交进仓库**的那一个 —— **里面不得有任何真实密钥**。
需要密钥时放进 `.env`。

---

## 八、端口与档位

### 端口映射

| 服务 | 宿主端口 | 说明 |
| --- | --- | --- |
| app | `8000` | 可用 `ALIGO__APP__PORT` 覆盖 |
| postgres | 5432 | 业务库 + Langfuse 元数据库（同实例、不同 database，省内存） |
| redis | 6379 | 会话状态、幂等锁、SSE 事件总线 pub/sub |
| minio | 9000 / 9001 | 对象存储（Milvus 依赖）/ 控制台 |
| milvus | 19530 / 9091 | 向量库 / metrics |
| prometheus | 9090 | 观测档 |
| grafana | 3000 | 观测档 |
| langfuse | 3001 | trace 档 |
| maxkb | 8080 | optional 档 |

### 四个档位

compose 里共 **13 个服务**，全部显式挂档，没有"默认档"服务。命令按档过滤。

| 档位 | 服务数 | 服务 |
| --- | --- | --- |
| `core` | 6 | app / postgres / redis / etcd / minio / milvus |
| `tracing` | 4 | langfuse / langfuse-worker / langfuse-redis / clickhouse |
| `observability` | 2 | prometheus / grafana |
| `optional` | 1 | maxkb |

默认全栈（core + tracing + observability）= **12 个**。`maxkb` 单列在 optional 档，
因为自建 Milvus RAG 是主路径，MaxKB 只作可选对照。

> ⚠️ **`down` / `clean` / `pull` 必须带齐档位**，否则 compose 会**静默什么都不做**
> 却照样退出 0。`Makefile` 用 `ALL_PROFILES` 统一处理，并有测试
> （`tests/test_makefile_contract.py::test_all_profiles_matches_compose`）钉住
> 「档位名集合必须与 compose 的 `profiles:` 完全一致」。

---

## 九、零密钥降级

**不配任何 API key，服务也应当能起来并让 `/healthz` `/readyz` 通过。** 这是 P1 的验收条件之一。

降级发生在模型装配层（`src/llm/factory.py`）：

- `llm.api_key` 为空（或全是空白）且 `use_mock_when_no_key=true` ⇒ 返回**确定性 Mock 模型**
  （`src/llm/mock.py`）。它的输出是**可预测的**：不联网、不花钱，因此可以直接写断言。
- 若 `use_mock_when_no_key=false` 却没给 key ⇒ **启动即报错**，并同时给出两条补救路径。
  刻意不"默默地用 mock 顶上" —— 生产环境里静默降级成 mock 是最坏的一种失败。
- 任何日志、探针响应、`describe_*` 输出里都**只有** `api_key_configured` 这样的布尔值，
  **绝不出现密钥本身**；探针的错误信息会剥掉 `scheme://user:pass@` 形式的口令。

同一层还有**熔断器**（`src/llm/breaker.py`，**自研** —— 框架完全没有这个能力）。
它与重试分工明确：**重试对付偶发失败，熔断对付持续失败**；全体智能体共享同一个实例。

**熔断器管不到「卡住不返回」**，所以中间件链上还有一道 `ModelTimeoutMiddleware`
（`src/llm/middleware.py`，超时取自 `llm.timeout_seconds`）：
框架的 `app` 链路构造模型时**不传** `client_kwargs`（`app/_service/_model.py:54-58`），
于是 openai SDK 的默认 600s 超时 × SDK 自身重试 2 次 × 框架 `max_retries=3`
（`model/_base.py:207` 的 `range(max_retries + 1)` = 4 轮）⇒ 最坏情况约 **2 小时**，
而在这两小时里调用既不返回、也不抛异常，熔断器连账都记不上。

> ⚠️ 它**不能用** `asyncio.wait_for` 实现：框架的模型基类会**吞掉** `CancelledError`
> （非流式返回一个 `finished_reason=INTERRUPTED` 的空响应 `model/_base.py:224-230`，
> 流式则发一个最终块 `:255-270`），而 `wait_for` 只有在取消**真的抛出来**时才判定超时 ——
> 于是超时永远检测不到。它改成**按钟判断**（`asyncio.wait` + 超时后显式 `cancel()`）。
> 这条差异在向量模型那一侧**不成立**（那边的重试只捕获 `Exception`），
> 所以 `src/web_embedding/bounded.py` 用的是更简单的 `wait_for` —— 两处实现不同是**刻意的**，
> 各自的理由都写在对应模块的文档里。

### ⚠️ 另一半：配了密钥也**不等于**用户就能用上模型

上面说的是"没密钥怎么办"。反过来的那一半更反直觉，而且是部署时最容易卡住的地方：

> `.env` 里配好了密钥、`/readyz` 全绿、`curl` 直接 `POST /chat/` 也正常，
> **但浏览器里"可用模型"是空的 ⇒ 发送按钮永远是灰的。**

原因是框架的对话链路要求会话的 `chat_model_config.credential_id` 指向一条**调用者可解析**的凭据记录，
而框架默认的 `DenyAllResourceAccessPolicy` 是严格的 owner 隔离 —— 运营者的密钥写在自己名下，用户看不见。

本项目为此实现了**系统凭据共享**（`src/llm/system_credential.py`），与上面的 Mock 降级**互补且互斥**：

| 形态 | 判据 | 每个用户看到什么 |
| --- | --- | --- |
| Mock 降级 | `should_use_mock(settings)` 为真（没密钥 + 允许降级） | 一条**属于他自己**的 Mock 凭据（不含任何秘密） |
| 系统凭据共享 | `sharing_enabled` = 降级**不**生效 **且** 确实有密钥 | 那条共用的系统凭据，**能用但读不到** |

**没有"共享开关"这个配置项** —— 判据完全由既有配置派生（`src/llm/system_credential.py::sharing_enabled`）。
运维去找一个不存在的开关是这一块的常见误区。

共享的语义由框架既有链路兑现，我们只提供"谁能用"这一条规则（`ResourceAccessPolicyBase`）：
非属主在 `GET /credential/` 里**看得见**这条记录，但 `data` 被打码成 `{name, type}`、`editable=false`
（已对真实服务核验）；运行期解析时框架返回原始记录供模型构造使用。权限是 **READ 不是 EDIT** ——
EDIT 等于"任何人都能把全公司的密钥删掉或改掉"。

⚠️ **正因为存在这条全员可用的凭据，`aligo-system` 这个身份必须不可被客户端声明**：
框架里"属主读自己的凭据"是**明文**且 `editable: true`，于是伪造 `X-User-ID: aligo-system`
一度可以读出密钥明文、甚至 DELETE 掉那条凭据让所有人当场失效。
现在 `src/server/middleware/auth.py` 在**三条通道**（X-User-ID 直连 / JWT 的 `sub` / 匿名模式）
统一拒绝这个保留身份，返回 **403**（不是 401 —— 没有任何登录能变成系统身份，返回 401 只会让客户端徒劳地重试）。
判据是**精确比较**（先 `strip()`），与框架逐字节一致：`ALIGO-SYSTEM` 在框架眼里本就是另一个用户。

用户到底"能用什么"，由 `GET /api/v1/default-model` 回答（四种 mode：`mock` / `configured` / `shared` / `missing`）——
细节见 [docs/01-功能接口.md](docs/01-功能接口.md)。

---

## 十、Embedding 三档

Embedding 是**可插拔三合一**的，按可用性依次降级 —— 任何一档缺失都不会让服务起不来：

| 档 | 触发条件 | 说明 |
| --- | --- | --- |
| 1. DashScope | `DASHSCOPE_API_KEY` 非空 | 走百炼的向量接口（与 LLM 共用同一个 key），维度需与 `milvus.dimension` 一致 |
| 2. 本地 BGE | 档 1 不可用 | 通过 **ONNX / fastembed** 跑 —— **刻意不引入 torch**，镜像只多约 150 MB |
| 3. 确定性 Mock | 前两档都不可用 | 用于测试与离线开发；同样维度、同样接口，但不保证语义质量 |

**维度必须全局一致**：`milvus.dimension`（默认 `1024`）与所选模型的实际输出维度必须对齐，
否则写入/检索会直接失败。

> ⚠️ 上面这条链是**本项目自己的**向量化路径（长期记忆、业务检索用）。**知识库（KB）链路只有两档**：
> 框架从 KB 记录里的「凭据类型」构造向量模型，而本地 ONNX 档是本项目自己的类、不挂在任何凭据上，
> 框架侧认不出来。因此 `scripts/seed_data.py` 的行为是：**有密钥就钉真实提供方凭据，没密钥才钉 Mock**
> （`ensure_kb_embedding_credential`）—— KB 的向量模型在**写记录那一刻**定死，框架的
> `PATCH /knowledge_bases` 不接受改它，所以换模型 = 改配置 + **重灌**（重跑会重钉记录并重算全部分块向量）。
> 反过来说，重灌也是「配好密钥后把 Mock 换成真模型」**唯一**的路 —— 这条建议以前写过但当时并不成立。
>
> 三档的降级链写在 `src/web_embedding/factory.py` 的 `_CHAIN = ("dashscope", "local", "mock")`，
> 与 LLM 共用同一个 key（`settings.llm.api_key`，展开自 `DASHSCOPE_API_KEY`）。
> ⚠️ 降级是**逐档构造、构造成功才停**，不是"先探测再选"—— 因此某一档的失败会被下一条兜住，
> 而不是让整个服务起不来。走到链尾（mock）时没有任何外部依赖，所以链尾**必定**构造得出来。

**每一档出厂前都会套上调用截止时间**（`ALIGO__EMBEDDING__TIMEOUT_SECONDS`，默认 30s）：
检索路径是「先向量化、再查库」，而向量模型的调用本身是无界的
（云端档是同步 HTTP 丢进线程池，本地档是 ONNX 推理排队）。只箍住查库那一段的话，
向量模型卡住时请求会**卡在向量化那一步** —— 向量库那边的超时与熔断连介入的机会都没有。
完整论证见 `src/web_embedding/bounded.py` 的模块文档。

> ⚠️ 包装层会**改变 `type(model).__name__` 与 `isinstance` 的结果**。
> 因此「实际用的是哪一档」一律经 `unwrap_embedding_model()` 判断 ——
> 绕过它会让 `src/knowledge/rag.py` 里那条「正在用没有语义的假向量」的告警**静默消失**
> （告警是发现该故障的唯一线索）。这一点在 `tests/test_web_embedding.py` 里有专门的用例钉住。

---

## 十一、可观测

### 健康检查三件套

| 端点 | 语义 | 关键性质 |
| --- | --- | --- |
| `GET /healthz` | 存活探针 | **零 I/O** —— 不连任何依赖。它回答的是"进程还在吗"，不是"能干活吗" |
| `GET /readyz` | 就绪探针 | **真实连** Postgres / Redis / Milvus。逐项返回 `{ok, detail, duration_ms, required}` |
| `GET /metrics` | Prometheus 指标 | 含 `aligo_ready`、熔断器状态、请求速率/延迟分位、模型调用结果与 token 吞吐 |

两条设计原则：

1. **`/healthz` 与 `/readyz` 判断的是两件独立的事。** 外部依赖挂了 ⇒ `/readyz` 503 但 `/healthz` 仍 200，
   k8s 不该因此重启进程（重启治不好一个挂掉的数据库，只会让日志更难读）。
2. **`/readyz` 的响应体是自解释的** —— 它直接告诉你哪一项不 ok、耗时多少、是不是必需项，
   不必再去翻日志。且它**永不泄露凭据**（错误信息经过 `_redact` / `_safe_error`）。

### Trace

trace id 由最外层中间件生成（或沿用上游传入的），并贯穿日志与指标。
请求头里的 trace id 会被校验，恶意/超长的值会被替换成新生成的 id。

OTLP → Langfuse 的装配是**条件式**的：`trace_exporter == "otlp"` 且 `otlp_endpoint` 非空才装。
⚠️ 这个条件必须显式满足 —— **未配 `TracerProvider` 时框架的 `TracingMiddleware` 会静默短路**，
表现为"trace 功能看起来开了，但 Langfuse 里什么都没有"，极易被误判成网络问题。

### Prometheus / Grafana

配置以**文件**形式进版本库（`scripts/prometheus/`、`scripts/grafana/provisioning/`），
`make up-obs` 起来就是配好的，不需要进界面点两下。当前只有 **2 个抓取目标**
（app 与 prometheus 自身）；Milvus 的 9091 与 Redis/PG exporter **未接**，
缺口清单写在 `scripts/prometheus/prometheus.yml` 末尾。

---

## 十二、测试与闸门

### `make test` —— 不需要 Docker

全部用例（当前数量以 `make check-docs` 的断言为准，它会把 README 里的数字与
`pytest --collect-only` 的实测值对拍）跑在 **sqlite 内存库** + 内存消息总线 + 临时工作区上，
不需要 Docker。
实现方式见 `tests/conftest.py`：它不 import `src.server.app`（那会用真实 dev 配置装配一个连 PG 的应用），
而是直接调用 `create_root_app(...)` 并注入替身。

⚠️ **为什么必须能在无 Docker 环境跑**：这保证任何人 clone 下来就能验证代码，
不必先拉 10 GB 镜像。这也是 `create_root_app` 的 `storage` / `message_bus` /
`workspace_manager` / `enable_scheduler` 四个参数存在的原因 —— 它们是**参数注入**，
而**不是配置开关**。做成开关的话，生产里一旦被误设，症状是"SSE 单进程内正常、
多副本之间收不到"，只在扩容时才暴露，且看起来与"配置写错了"毫无关系。

### `make check-docs` —— 文档的两道闸门

本项目的注释里有大量**手工维护、不会自动更新**的东西，它们失效时文档看起来毫无异常。
两个脚本各管一类，**互不越界**：

| 脚本 | 查什么 | 不查什么 |
| --- | --- | --- |
| `scripts/check_doc_refs.py` | 「文件在不在 + 行号超没超界」 | ✗ 数字对不对 |
| `scripts/check_doc_counts.py` | 「数字对不对（== 机器实测值）」 | ✗ 行号 |

两类问题的失效方式都**无声无息**：

- 往被引用的文件里加一行注释，后面所有引用的行号整体偏移 1 —— 文字仍通顺，链接仍"看起来"有效。
  最坏的不是引错行，而是**让人误以为已经看过了那一段**。
- 计数断言（如「共 13 个 `image:` 行」）不会因为 compose 里加了个服务而报错，
  它只是变成了一句**读起来像认真结论的假话**。

`check_doc_counts.py` 的两个设计要点：

1. **脚本里没有 13、没有 9、没有 28。** 数字只从被校验的文本里读，再与实测值比对 ——
   否则脚本自己就成了第二份真值：改了代码忘了改脚本，闸门依然绿。
2. **「断言被删掉」也算失败。** 把一句带数字的注释整段删掉，是让闸门变绿最省事的办法，
   而它同时也删掉了那句话提供的保障。所以锚点一处都匹配不上时，报的是**失败**而不是跳过。

⚠️ `check-docs` 里最慢的一环是**数用例数**：它要真跑一次 `pytest --collect-only`
（本机约 7 秒），因为静态数 `def test_` 会被 `parametrize` / `skipif` 数错 ——
而"少一点"恰好是那种看起来合理、于是没人核对的错。整条 `make check-docs` 约 15 秒。

### `make smoke`

对**已启动**的服务发真实请求：`/healthz` → `/readyz` → `/metrics` → trace id 生成与回传 → 业务接口，
最后**真的发一句话**并等到 `REPLY_END`。退出码 0 表示通过。`--base-url` 可指向任意实例。

⚠️ **它会自己收拾干净。** 最后那项对话检查必须真的建一个智能体 + 一条会话
（这是「证明它真能对话」的代价），跑完会**删掉它们并回读确认** ——
成功、失败、异常路径都会删，删除结果写在该项检查的详情里
（绿行上那句「🧹 已清理本轮产物」就是它）。这条不是一开始就有的：
2026-10-03 实测 `smoke-user` 名下有 **11 个**智能体，其中 **10 个**正是
这项检查历次留下的「冒烟测试助手 <时间戳>」，而没有任何地方告诉运维这件事 ——
一个「跑一次多一个」的脚本，最终会让人不敢再跑它，而没人跑的冒烟测试等于没有冒烟测试。

要留现场排查时用 `python scripts/smoke.py --keep-artifacts`：它会打印 id、什么都不删。

### `make eval` —— 离线评测（回答「改完之后是变好还是变差」）

`make test` 回答「代码逻辑对不对」，`make eval` 回答「系统整体是变好还是变差」——
它对 `tests/evaluation/golden_dataset.yaml` 里的 34 条黄金样例跑一遍**真实的编排链路**
（车道判定 / 意图识别 / 工具选择），把结果与标注对拍，并输出可 diff 的报告 `eval_report.json`。

- 判分默认走 **LLM-as-judge**（需要模型密钥）；**没配密钥时会自动退化为规则判分**，
  并在报告里如实写 `judge_mode=rule` —— 它不会假装做过语义评估。
- 阈值与输出路径可覆盖：`make eval MIN_SCORE=0.9 EVAL_OUTPUT=/tmp/r.json`，低于阈值非零退出。
- 本机实测（2026-10-03，`judge_mode=llm`）：**overall_score 0.9778** —— 其中车道判定 34/34、
  意图识别 F1 0.9333、工具选择精确匹配 16/16。

### `make loadtest` —— 压测

对**已启动**的服务施加并发负载（默认打 `/healthz`，可用 `LOADTEST_URL=` / `LOADTEST_ARGS=` 覆盖），
越过延迟阈值即以非零码退出。⚠️ 本机 Docker VM 只有 3.82 GiB / 4 vCPU：压测前先确认
没有同时在跑 `make up-tracing`（12 容器），否则压测与被压测的两端会互相抢资源，
看到的数字既不代表服务能力、也不代表宿主机能力（见 `docs/06-部署与运维.md` 第十节）。

### `make concurrency` —— 并发**正确性**（不是并发性能）

`make loadtest` 压的是**一个端点、一个人、几千发**，量的是延迟分布；它**只发 GET**
（见 `scripts/loadtest.py` 的模块注释），所以「三个用户各建各的会话、各发各的话」
这条路径它一毫秒都没覆盖过 —— 而线上一次真正的并发，几乎必然发生在这里。
`make smoke` 则是串行的，对「同时」这个维度覆盖为零。

`make concurrency` 补的就是这一段：**3 个用户（一次性工作区 `concurrency-user-1..3`）
在同一个栅栏释放后发出三条不同的问句**，然后跑 12 条判据。三条问句刻意走不同的链路：

| 用户 | 问句 | 链路 |
| --- | --- | --- |
| 1 | 住宿标准 | 快车道（规则精确命中，不经意图识别模型） |
| 2 | 我上个月订的那张去广州的机票现在是什么状态 | 慢车道 QUERY_ORDER → `query_orders` |
| 3 | 去北京出差住宿费能报多少 | 慢车道 QUERY_POLICY → `check_travel_policy` |

> ⚠️ 用户 2、3 调的两个工具读的都是**内存仓储** —— **不是 Postgres，更不是 Milvus**。
> `src/server/agents_factory.py:162-168` 给的是一整包内存实现（该文件自己写着
> 「P3 阶段是内存实现，P4 换成 Postgres」），`policy=StaticPolicyRepository()` 还是
> **无参**构造 ⇒ 所有用户一视同仁拿 `DEFAULT_POLICY_LIMIT`。本表早先写成
> 「业务库 / 政策库（Milvus 向量检索那条链路）」，**那是错的**：实测 21 条轨迹里每条
> 恰好 1 次 embedding、且全部是消息记忆召回，`knowledge_documents` 为 0 ——
> **RAG 链路的并发行为根本没被这个脚本覆盖**。别照着旧说法以为它测过了。

其中两条是**互为补角**的串号判据，缺一条就漏一类缺陷：**流不串号**（每个 SSE 事件
带的 `session_id` 都是自己的 —— 防「推错了人」）与**记载不串号**（事后读回的会话历史里
恰好一条 user 消息、且逐字等于本人的问句 —— 防「存错了」）。

判据 2 与 2b 也是一对补角：**回复非空**管「有没有字」，**回复是答复而非占位**管
「那些字是不是答复」。后者来自一次真实的假绿灯：三个用户全部 `completed`、其余判据全绿，
而其中一个人拿到的全部回复只有 30 个字（含一个空格）——
「已让政策问答智能体检索制度原文，稍等。 等待政策检索结果中。」。
`REPLY_END` 有了、消息也非空，用户却什么都没拿到。

> ⚠️ 这条先例只有作者记录在案，**没有日志可复核**（服务端日志不记录回复正文，当时的日志已丢）。
> 它能证明的是「这种形状真实发生过」，不能用来推算发生频率 —— 频率看下面连跑统计表。

2b 内部还分两级，因为实测撞到的「不是答复」有**两种**：

| 形状 | 例子（实测原文） | 级别 |
| --- | --- | --- |
| 占位 —— 用户**什么都没拿到** | `已让政策问答智能体检索制度原文，稍等。等待政策检索结果中。` | ❌ 失败 |
| 内心独白 —— 答案在，但开头是模型草稿 | `用户想查已有的机票订单，我直接查一下。查不到这张票。…`<br>`I'll look up your flight orders. 我查了一下，你名下目前没有…` | ⚠️ 告警 |

**频次（可复核）：** 一次默认运行（3 用户 × 1 轮）+ 两次 `--users 6 --rounds 2` 加压运行，
合计 27 轮（**不是**三次 6×2 —— 那是 36 轮），独白命中 **7 条，全部在慢车道**
（两个加压档分别 2/8、5/8，默认档 0/2）。7 是**下界**——探测器只认 9 种开头写法。
（早先这里写过「合计 4 次」之类的历史数字，无法复核且与其他文件互相矛盾；
服务端日志根本不记录回复正文，重跑一遍才是可用的证据。）

分级的理由是**不让并发结论取决于模型文风** —— 一个会因为文风随机变红的判据，
在真实项目里的下场是被关掉，连旁边那个真正重要的「占位」判定也一起失效。
告警照常打印、照常出现在末尾那行「另有 N 条告警，见上」里，只是不改退出码。
通用判定要靠 `scripts/eval.py` 的 LLM-as-judge。

> ⚠️ **两件不能从上面这段读出来的事：**
>
> 1. **它不是「文风问题」那么轻。** 有一例泄漏的是**内部实现**：回复正文里出现了
>    工具名 `check_travel_policy`、参数名 `kind/price/cabin`、智能体名 `policy_rag`。
>    那是信息外泄，只是本判据按「开头措辞」判定，抓不抓得到看那一轮怎么起头。
> 2. **它没有硬闸门。** 告警不改退出码，而 `eval.py` 不在 `make concurrency`
>    的路径上 —— 「用户看到模型草稿」这件事目前**在 CI 里拦不下来**。
>    warn 的取舍成立，但**不等于这个缺陷已经被防住了**。
>
> 另外，「慢车道」这个定位也**不是**「快车道免疫」：快车道只跳过**意图识别**那一次
> 模型调用（`src/orchestration/lane.py:16-27`，另有 `tests/test_orchestration_lane.py`
> 断言快车道模型调用数 **== 1，不是 0**），最终答复照样由模型写。实测里快车道
> 撞到过 2 条**占位**（`正在为您查询公司的住宿差标规定，稍等。`），把整轮打成退出码 1。

还有一条是整套测试的底线：**确实同时**。它不看「我用了 threading」这句话，而是记录每人
`POST /chat/` 与收到 `REPLY_END` 的时刻，判定 `max(t_chat) < min(t_end)` ——
即最后一个人的请求发出时，第一个人的回复还没结束。没有这条，
一个**把请求顺序发出**的脚本照样会打印「3/3 通过」，而那验的是排队。

`--users N`（默认 3）、`--rounds R`（默认 1，第 k 轮全员同时发）、`--keep`（留现场）。
与 `smoke` 同一条纪律：**跑完自己删掉**；若某用户已有 `main_plan`，**复用且不删**它。

本机实测（2026-10-03，`make up-tracing` 未起、只有 core 6 容器）。
**下表是连跑多次统计出来的通过率，不是一次幸运的结果** —— 单看一次会得出错误印象：

| 配置 | 通过 | 未通过 | 未通过的原因 |
| --- | --- | --- | --- |
| `make concurrency`（3 用户 × 1 轮） | **4/4** | 0 | — |
| `--users 6 --rounds 2` | **5/7** | 2 | 都是**模型输出**问题，见下 |

> ⚠️ **两次未通过都不是并发缺陷。** 12 条判据里与并发正确性有关的那几条
> （流不串号、记载不串号、跨用户不可见、无服务端错误）**每一次都是绿的** ——
> 3~6 个用户、共 96 条会话（4 次三用户 × 1 轮 = 12 条，7 次六用户 × 2 轮 = 84 条，
> 每用户每轮一条新会话 —— 见 `scripts/concurrency_test.py` 的 `_run_user`）、
> 几百次越权探测里，没有一次串号、没有一条 5xx。
> 红的是判据 2b 和判据 1：
>
> - **判据 2b（占位）** 2 次：模型把「正在查制度原文，稍等」当成最终答复交出去，
>   用户什么都没拿到。**两次都发生在快车道**（`用户 4 · 快车道 · 政策词条`）——
>   快车道只省掉意图识别那一次模型调用，最终答复照样是模型写的。
> - **判据 1（流没结束）** 1 次：某轮推流在 `TOOL_RESULT_START` 之后停了，
>   脚本等满 120 秒仍未收到 `REPLY_END`。判据详情会区分「脚本等超时」与
>   「流被结束了却没给 REPLY_END」——这一例是前者，**不能据此断定服务端有错**，
>   但它说明该轮确实卡了 120 秒以上（同批正常轮次 5~40 秒）。
>
> 换句话说：**`make concurrency` 在高并发档位下约 30% 的概率退出非零，
> 原因不在并发，而在模型不肯好好给答复。** 要让这条命令稳定当闸门用，
> 得先修输出质量（提示词或输出侧剥离），不是放宽判据。

### `tests/test_makefile_contract.py`

`make help` 是使用者的**唯一入口**（它是默认目标）。这个测试断言三件事：

- 每个真实目标都在 help 输出里（本项目曾漏列 **7 个**目标，而 `make help` 照样退出 0）；
- help 里提到的每个 `make xxx` 都真实存在（不承诺不存在的东西）；
- `.PHONY` 与真实目标集合**互相**包含（否则目标会与同名文件撞车而**被静默跳过**）。

---

## 十三、已知限制

**故意的、已记录的边界**（不是"还没发现的 bug"）：

- **`ALIGO__APP__WORKERS=1`** —— 默认单进程。多进程下框架的 `SchedulerManager` 会**每个进程各触发一次** cron，
  且连接池峰值是 `pool_size + max_overflow = 30` 每进程。要扩容需一并解决这两件事，不能只调 worker 数。
- **消息总线必须是 `RedisMessageBus`** —— `InMemoryMessageBus` 在多副本下会让 SSE 事件收不到，
  框架自己的注释里就这么写着。测试里用内存总线只是因为测试是单进程且不测跨副本。
- **`download_secret` 必须显式设为固定值** —— 否则每个进程随机生成，LB 后面的签名 token 会失效。
- **`auto_migrate` 恒为 `False`** —— 框架明确警告多副本下它不安全。
- **MCP 工具调用的超时由注册方自己填**。框架的 `MCPClientConfig` 有
  `execution_timeout` 字段（`mcp/_mcp_client.py:122`），但它**默认 `None` = 无界**，
  而本项目不代它设默认值：注册 MCP 服务器时请显式填这一项。
  卡住的后果是「那一次工具调用永不返回 ⇒ 该轮对话假死」——
  模型调用超时中间件**管不到它**（卡的是工具，不是模型）。
  ⚠️ 不给工具调用统一加超时是**刻意的**：本项目自己的差旅工具里有下单/改签这类
  **非幂等**操作，超时后我们并不知道对端到底做没做成 —— 把「失败」变成「不知道」
  比多等一会儿更糟。MCP 那一侧由注册方按自己工具的性质决定。
- **知识库写路径里的 `insert` / `delete` 不经超时与熔断**（`src/knowledge/guard.py`
  的 `_GUARDED_OPERATIONS` 只含**检索路径上会被调用**的那几个方法，
  既不是「所有读方法」，也不是「所有写方法」）。受影响的是
  `DELETE /knowledge_bases/{id}`（逐文档删）与上传文档的**写入**那一段
  （向量化那一段另有 30s 预算）—— Milvus 卡住时它们会一直挂着。
  理由同上（非幂等写不静默放弃），且对话/检索路径不受影响。
  ⚠️ 判据是**「会不会在检索路径上被调用」而不是「名字像读还是像写」**：
  `create_collection` 名字像写操作，但集合缺失时 `KnowledgeBase.search()` 会
  通过 `ensure_collection()` 直接调它 —— 所以它在名单里。**这也意味着
  `POST /knowledge_bases`（显式建集合）走的是同一个被箍住的方法，它是有界的**
  （5s 超时，且失败计入那个共享熔断器）—— 「名字像写 ⇒ 无界」这个直觉在这里是错的，
  本项目就此写过一次错的文档（见 `docs/06-部署与运维.md` 第 10.2 节）。
  ⚠️ 另外，Milvus 持续卡死几小时的话，被放弃的 `to_thread` 线程仍会以约
  120 个/小时堆积并最终吃光默认执行器的 8 个工位 —— **"有超时"不等于"线程数有上界"**，
  见 `docs/06-部署与运维.md` 第 10.2 节。
- **`docs/` 的编号文档已就位**（`01-功能接口` / `02-技术架构` / `03-模块关系与调用逻辑` /
  `06-部署与运维`），另含 [实施计划.md](docs/实施计划.md)（计划书原件，含 P1–P5 的决策与理由）
  与抓取来的博客原文。
  ⚠️ 计划书里的「写死的数字」是**当时的实测值**（如「95 个用例」），不随仓库更新 ——
    `docs/` 在 `scripts/check_doc_counts.py` 的 `SKIP_DIRS` 里，故不受计数闸门约束；
    但它在 `scripts/check_doc_refs.py` 的扫描范围内，**行号引用会**被 `make check-docs` 校验。
- **用户自建的凭据只会得到「按类型自动命名」的名字**。框架的
  `_generate_credential_name`（`app/storage/_sql/_storage.py:679`）在 `name` 为空时
  从**凭据类型**推导显示名：`dashscope_credential` → `Dashscope`，同类型再来一条就是
  `Dashscope (2)` / `Dashscope (3)`，**与模型名和用途无关**。
  ⚠️ 于是「同一个用户的两条 DashScope 凭据分别对应不同模型」时，从列表上**分不出谁是谁** ——
  建凭据时请自己填一个能认出来的 `name`。
  本项目自己播种的那条系统凭据是**显式覆写**名字的（`src/llm/system_credential.py`），
  不受这条影响；受影响的是用户通过 `/credentials` 接口自建的那些。
- **前端没有测试套件**。`web/frontend/` 下只有 eslint 与 `tsc -b`（后者随 `pnpm build` 一起跑），
  没有 vitest/jest 一类的单测。因此前端能保证的是"类型与构建过得去"，
  **运行时行为**（事件流怎么渲染、卡片在什么边界条件下退化成默认渲染器）没有自动化覆盖。
- **构建产物进版本库**。前端产物落在 `src/server/static/`（这样不用改 `Dockerfile`，
  镜像里随 `COPY src` 进去；开发期被 `./src:/app/src:ro` 只读挂载覆盖）。
  ⚠️ 代价是 `src/` 下带着 **14.69 MiB / 393 个打包文件**（构建上下文的过半体积），
  且**每次 `pnpm build` 都会换掉哈希文件名**，diff 会非常吵。
  这是"镜像里必须有产物"与"仓库里别塞产物"之间的取舍，当前选了前者。

**待验证**：

- ✅ `make smoke` 对完整非可选栈（`make up-tracing` 的 12 容器）**8/8 通过**，且同一次冒烟
  在 Langfuse 里留下完整 trace —— 这一项已从「待验证」转为「已验证」，见本文开头「已验证」。
- ✅ `.dockerignore` 里关于构建上下文体积的注释已用真实 `docker build` 量过并写入实数
  （**27.43 MiB / 2013 个文件**，量法与被淘汰的三种错误量法都写在 `.dockerignore` 顶部）。
  这一项已从「待验证」转为「已验证」。

**构建耗时**：`make build` 带 `--no-cache` 全量重建，其中 **pip 层占绝大部分** ——
本机容器内 pip 下载速度实测约 180–240 kB/s，因此是全流程里最慢的一段（数分钟量级）。
`make up` 保留缓存，快得多。只是改了源码或文档时请用 `make up`，别用 `make build`。

### ⚠️ 构建踩坑：pip 报「找不到某个版本」时，先怀疑镜像而不是版本号

`make up` 曾两次在 `pip install -r requirements.txt` 阶段失败，两次的根因完全不同，
且**第二次的报错极具误导性**，记在这里省下再次排查的时间：

1. **依赖冲突（真·版本问题）** —— `fastmcp==4.0.5` 的元包无条件依赖
   `fastmcp-slim[client,server]==4.0.5`，而后者要求 `mcp>=2.0.0,<3.0.0`；但
   `agentscope` 要求 `mcp<2.0.0`。二者区间不相交，`ResolutionImpossible`。
   已改为 `fastmcp==3.4.7`（其 slim 要求 `mcp<2.0,>=1.24.0`，与 agentscope 相容）。
   完整推导见 `requirements.txt` 第 93 行附近的注释。

2. **镜像缓存陈旧（假·版本问题）** —— 报错是
   `Could not find a version that satisfies the requirement watchfiles==1.3.0`，
   看起来像版本号钉错了。实际该版本在 PyPI 上**存在且未被 yank**。
   真因：阿里云镜像索引页的 **gzip 缓存副本陈旧**，而 pip 无条件发
   `Accept-Encoding: gzip`，于是拿到不含 1.3.0 的旧副本。同一时刻同一 URL：

   | 请求头 | 响应大小 | 含 `watchfiles-1.3.0` |
   | --- | --- | --- |
   | `Accept-Encoding: identity`（curl / 浏览器） | 744,530 B | ✅ 66 处 |
   | `Accept-Encoding: gzip`（**pip 的默认**） | 464,851 B | ❌ 0 处 |

   **误判陷阱**：用 `curl` 或浏览器打开索引页会看到版本明明在 —— 因为它们不发
   `Accept-Encoding`。所以「curl 能看到」**不能**证明 pip 能看到。
   复现命令：

   ```bash
   # 会看到 1.3.0（假象）
   curl -s https://mirrors.aliyun.com/pypi/simple/watchfiles/ | grep -c 'watchfiles-1\.3\.0'
   # 会看不到 1.3.0（pip 的真实处境）
   curl -s -H 'Accept-Encoding: gzip' --compressed \
     https://mirrors.aliyun.com/pypi/simple/watchfiles/ | grep -c 'watchfiles-1\.3\.0'
   ```

   对策：`Dockerfile` / `docker-compose.yaml` 给 pip 配了
   `PIP_EXTRA_INDEX_URL=https://pypi.org/simple/` 作为**备用源** —— 主源缺的版本由
   PyPI 补齐。这是正确性保障，不是可选优化。详见 Dockerfile 顶部的 pip 源说明段。

---

## 附：与本项目相关的上游资料

- [阿里商旅 AliGo 多智能体实践（博客）](https://agentscope.io/blog/alibaba-business-travel/) ——
  业务架构与产品经验的来源。**注意它是 AgentScope 1.x 时代的写法**
  （`ReActAgent` / `register_class_hook` / `@tool` / `agent(user_msg)` 在本地版本**全部不存在**），
  只借它的**业务设计**，API 一律以本地安装的包为准。原文见 `docs/博客原文-Alibaba-Business-Travel.md`。
- 本地包与博客写法的对应关系：`Agent` + `MiddlewareBase` + `FunctionTool` + `reply_stream`。
