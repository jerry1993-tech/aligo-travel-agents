# ==============================================================================
# 文件职责：AliGo 差旅助手应用镜像的**多阶段构建**定义。
#           构建阶段负责把 Python 依赖装进独立虚拟环境；运行阶段只带走该虚拟环境，
#           从而把编译器与构建缓存留在构建层，减小最终镜像体积与攻击面。
#
# 上下游依赖：
#   - 上游：被 `docker-compose.yaml` 的 `app` 服务以 `build.context: .` 引用；
#           依赖根目录 `requirements.txt`（第三方依赖清单）与 `third_party/`（源码库）。
#   - 下游：容器内以非 root 用户运行 `uvicorn`，对外暴露 8000，健康检查打 `/healthz`。
#
# 关键约定（勿擅改）：
# ==============================================================================


# ------------------------------------------------------------------------------
# 阶段一：builder —— 只做「装依赖」，产物是一个自包含的虚拟环境 /opt/venv
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS builder

# 少数依赖带 C 扩展，slim 镜像没有编译器；这些包仅存在于构建层，不会进入运行镜像。
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# 建立虚拟环境。用虚拟环境而非全局 site-packages，是为了让「构建层」能干净地整体
# 拷贝到运行层，无需在运行层重装依赖、也不会带上构建工具。
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# ------------------------------------------------------------------------------
# pip 源：主源 + 备用源（**两个都要，缺一不可**）
# ------------------------------------------------------------------------------
# ⚠️ 位置约定：本段必须待在**所有 apt-get 之后、COPY requirements.txt 之前**。
#    这不是风格偏好，是构建耗时。理由是 Docker 的缓存键规则：
#      **ARG 的取值会成为它之后每一层的缓存键的一部分**（ENV 同理）。
#    本段原先写在 `apt-get install build-essential` **之前** —— 于是「改一次 pip 源」
#    会让 apt 层缓存失效，哪怕 apt 与 pip 毫无关系。实测代价：apt-get update +
#    install 重下一遍约 15 分钟。而它**看起来完全合理**（"环境变量当然写在最前面"），
#    只有当你为了改 pip 而动了它、然后眼看着 apt 又跑一遍时才会发现。
#    边界的两侧各有代价：再往**上**会拖累 apt 层；再往**下**（比如挪到 pip 命令本身
#    之前但已经被 COPY 过）则会被 COPY 的上下文变更牵连。
ARG PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ARG PIP_EXTRA_INDEX_URL=https://pypi.org/simple/

# 🔴 为什么必须配 PIP_EXTRA_INDEX_URL 指向 pypi.org：阿里云镜像的**简单索引页存在
#    gzip 缓存陈旧**问题，会让「新发布不久的版本」对它不可见 —— 而 pip 恰恰总是请求
#    gzip 变体。2026-10-01 实测（同一 URL、同一时刻、同一台机器、仅改请求头）：
#        Accept-Encoding: identity  → 744530 字节，**含** watchfiles-1.3.0
#        Accept-Encoding: gzip      → 464851 字节，**不含** watchfiles-1.3.0
#    而 pip 无条件发送 `Accept-Encoding: gzip, deflate`，于是只能拿到缓存陈旧的那一份，
#    报错为：
#        ERROR: Could not find a version that satisfies the requirement watchfiles==1.3.0
#               (from versions: ... 1.2.0)
#    —— 看起来像「版本号钉错了 / 该版本被 yank 了」，实际是镜像缓存问题：该版本在 PyPI
#    上确实存在（2026-09-21 发布，未被 yank，含 cp310-abi3-manylinux_2_17_x86_64 轮子）。
#    ⚠️ 排查时**极易误判**：用 curl 或浏览器打开同一个索引页会看到 1.3.0 明明在列表里
#    （它们不发 Accept-Encoding，拿到的是 identity 那份新鲜内容）。所以「curl 能看到」
#    不能用来证明 pip 能看到 —— 必须带 `Accept-Encoding: gzip` 复现。
#    ⇒ 备用源不是「优化项」而是**正确性**保障：主源查不到的版本由 pypi.org 补齐。
#    代价：pip 会对两个源各查一次索引，解析阶段略慢；对本项目可接受。
#    安全：两个源都是可信公共源，且 requirements.txt 全部 `==` 精确钉版，
#    不存在依赖混淆（dependency confusion）的取值空间。
# 由 docker-compose.yaml 的 build.args 传入，便于在受限网络下一行切换。
#
# 构建阶段的环境变量：
#   PIP_INDEX_URL       主源（快，默认阿里云）
#   PIP_EXTRA_INDEX_URL 备用源（新，默认 pypi.org）—— 见上方说明，**不要删**
#   PIP_NO_CACHE_DIR  不落 pip 缓存，避免构建层体积膨胀
#   PIP_DISABLE_PIP_VERSION_CHECK 跳过版本自检的网络请求，加快构建
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 先只拷贝依赖清单再安装：只要 requirements.txt 未变，这一层就能命中 Docker 构建缓存，
# 改业务代码时不必重装依赖（显著缩短迭代时间）。
COPY requirements.txt /tmp/requirements.txt
# setuptools 与 wheel 必须显式装上：下面的本地源码包用 `--no-build-isolation` 构建，
# 该开关意味着「用当前环境里的构建后端，不去 PyPI 下载隔离环境」—— 若缺 setuptools，
# 构建会以 `BackendUnavailable: Cannot import 'setuptools.build_meta'` 失败。
RUN pip install --upgrade pip setuptools wheel \
    && pip install -r /tmp/requirements.txt

# ---- 安装本地源码包：agentscope 与 ReMe ----
# 这一步做了两件事：把两个库装成 site-packages 里真正的包（于是运行期 `import agentscope`
# 直接可用），以及让本项目**不再需要** PYTHONPATH 指向 third_party/。
#   那会拉进与 requirements.txt 不一致的版本，把已钉好的组合打散。
COPY third_party/agentscope/ /tmp/src/agentscope/
COPY third_party/ReMe/ /tmp/src/ReMe/
RUN pip install --no-deps --no-build-isolation /tmp/src/agentscope /tmp/src/ReMe \
    && rm -rf /tmp/src


# ------------------------------------------------------------------------------
# 阶段二：runtime —— 最终运行镜像，只包含运行时必需物
# ------------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# 运行期环境变量：
#   PYTHONUNBUFFERED     日志实时输出，不缓冲（否则 `docker logs` 看不到实时日志）
#   PYTHONDONTWRITEBYTECODE 不生成 .pyc，保持挂载目录干净
#   PYTHONPATH           现在**只需** /app，让 `import src.*` 可用。
#                        agentscope / reme 已在 builder 阶段装进 /opt/venv，
#                        由 site-packages 解析 —— 不再需要、也不应再写上
#                        third_party 的 src 路径（那会让"到底加载的是哪份代码"
#                        变得不确定，且与本地开发环境的行为分叉）。
#   TZ                   容器时区，与业务侧 InjectionConfig.timezone 保持一致
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TZ=Asia/Shanghai

# 运行期系统依赖：
#   curl      —— compose healthcheck 探活用（必需，缺失会导致容器永远 unhealthy）
#   tzdata    —— 让 TZ 生效，保证日志与业务时间正确
#   ca-certificates —— 访问 DeepSeek 等 HTTPS 接口时的根证书
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl tzdata ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段整体搬来虚拟环境；--chown 直接指定属主，避免多一次改权限的层。
COPY --from=builder --chown=1000:1000 /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# 创建非 root 用户运行服务（uid/gid 固定 1000）。以 root 运行容器是安全红线：
# 一旦应用被攻破，攻击者将直接获得容器内的最高权限。
RUN groupadd --gid 1000 aligo \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash aligo

WORKDIR /app

# 拷贝项目源码。--chown 保证 aligo 用户对代码有读权限。
#
# ⚠️ 这里**仍然**要拷 third_party/，但它已经**不再参与模块导入**（agentscope / reme 走
#   /opt/venv 的 site-packages，见上方 PYTHONPATH 说明）。留着它是为了**运行期资产**：
#   `third_party/ReMe/skills/<技能名>/SKILL.md` 位于 ReMe 仓库根级、**不在 `reme` 包内**，
#   因此 `pip install` 不会把它带进 site-packages，而 ReMe 运行期要求技能目录下存在
#   SKILL.md（缺失即抛 FileNotFoundError）。.dockerignore 已专门保留这类 SKILL.md，
#   详见该文件第三、四节的说明。
#   体量（**实测值**，取自镜像内实物 `du -sb`，不是估算、也不是 BuildKit 的进度行）：
#   .dockerignore 过滤后整个构建上下文是 **7.02 MB**（7,017,657 字节），
#   而 third_party/ 在磁盘上是 174 MB（其中 52 MB 是 ReMe 的 .git）。
#   即 .dockerignore 挡掉了约 **96%**，而留下的这 7.02 MB 几乎全是**构建必需**的源码：
#   agentscope/src 6.09 MB 是那两个 `pip install` 的输入，**动不得**。
#   ⚠️ 早先这里写的是"179.75 kB、挡掉 99.9%"，**那两个数都是错的**：179.75 kB 是
#      BuildKit 在缓存命中时报的**增量传输**，被误当成了上下文总量。辨别方法已写在
#      .dockerignore 末尾（同一份上下文连跑两次，13.10MB → 137.68kB）。
#   所以：本文件与 .dockerignore 是配套的，改任何一个前请**先读 .dockerignore 的注释**；
#   但也不要为了"再压掉一点体积"去动它——真正必须排除的理由是**密钥入层**与**代码重复**，
#   不是体积。这张上下文清单已压到接近下限，再往下砍就会切到构建输入。
COPY --chown=1000:1000 third_party/ /app/third_party/
COPY --chown=1000:1000 src/ /app/src/
COPY --chown=1000:1000 config/ /app/config/
COPY --chown=1000:1000 scripts/ /app/scripts/
COPY --chown=1000:1000 requirements.txt /app/requirements.txt

# 准备运行期需要的可写目录：
#   workspace —— LocalWorkspaceManager 的文件根目录（智能体读写中间产物）
#   logs      —— 结构化日志落盘目录
# 先建好并授权，否则非 root 用户在挂载卷时可能无写权限而启动失败。
RUN mkdir -p /app/workspace /app/logs && chown -R 1000:1000 /app/workspace /app/logs

# 切换为非 root 用户，之后所有指令与进程均以 aligo 身份运行。
# ⚠️ 注释必须**单独成行**，绝不能写成 `USER aligo  # 说明`：Dockerfile 不支持行尾注释，
#    指令行里的 `#` 及之后的内容会被当作**参数的一部分**。这里的后果是把用户名解析成
#    "aligo        # 切换为非 root 用户，…" 这一整串，构建期不报错（语法合法），
#    直到 `docker compose up` 创建容器时才失败：
#      unable to find user aligo        # 切换为非 root 用户…: no matching entries in passwd file
#    `RUN` 之所以能安全地写尾注释，是因为 `#` 由 shell 处理；`USER`/`EXPOSE`/`WORKDIR`/
#    `COPY` 等没有 shell 介入，一律不行。（同类提醒见下方 EXPOSE 上方的说明。）
USER aligo

# 声明服务端口（文档作用，不实际发布；真实端口映射见 docker-compose.yaml 的 ports）。
# 注释单独成行而不是写在 EXPOSE 后面：部分 Docker 语法检查器会把同一行的尾注释
# 当成额外的端口号解析，报出 "Invalid containerPort" 的假告警。
EXPOSE 8000

# 镜像级健康检查：探本项目的 /healthz，它**不做任何 I/O**，是纯存活（liveness）探针。
#
# 为什么镜像里用 /healthz 而不是 /readyz：
#   HEALTHCHECK 写在镜像里，意味着 `docker run` 单跑也生效。而单跑时 PostgreSQL /
#   Redis / Milvus 本来就不在（也未必需要），此时 /readyz 必然 503 —— 容器会被标成
#   unhealthy，传达的信息是"镜像坏了"，而真相是"依赖没起"。用 liveness 探针才不会误导。
#   要看真实的就绪状态，请用 compose：那里的 app 服务把探针覆盖成了 /readyz
#   （见 docker-compose.yaml），因为 compose 环境下三个依赖都在，就绪才有意义。
#
# ⚠️ 必须说清楚它**不**检查什么：/healthz 只证明"进程活着、事件循环没卡住"，
#    它**不**会去连 PG/Redis/Milvus。若把它当成"依赖都通了"的证据，就是就绪假阳性。
HEALTHCHECK --interval=15s --timeout=10s --start-period=60s --retries=10 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

# 启动命令：uvicorn 承载 FastAPI 应用。
#   --host 0.0.0.0 让容器外可访问（绑 127.0.0.1 会只在容器内可见）
#   --workers 由环境变量控制，默认 1：本机 Docker VM 仅 3.82 GiB 内存，
#             多 worker 会成倍占用内存；扩容场景再调大（见 docs/06-部署与运维.md）。
ENV ALIGO__APP__WORKERS=1
CMD ["sh", "-c", "uvicorn src.server.app:app --host 0.0.0.0 --port 8000 --workers ${ALIGO__APP__WORKERS}"]
