# -*- coding: utf-8 -*-
"""服务装配入口 —— 全项目的**根应用**。

本文件是硬契约：``Dockerfile`` 的 ``CMD`` 与 ``docker-compose.yaml`` 都按
``uvicorn src.server.app:app`` 启动，因此模块级必须存在一个名为 ``app`` 的
ASGI 应用对象。改这个约定等于改部署方式，必须同步改那两个文件。

上下游依赖：
    - 上游：``src/config/loader.py``（配置）、``agentscope.app.create_app``（框架服务层）。
    - 下游：``uvicorn`` 加载 :data:`app`；``scripts/smoke.py`` 与 compose 的健康
      检查打 :mod:`src.server.probes` 的路由。

==============================================================================
核心决策一：**复用 create_app 的返回值作根应用，绝不 mount**
==============================================================================
    ``create_app`` 把 lifespan 挂在它自己创建的 FastAPI 实例上
    （``agentscope/app/_app.py:292``），而 **Starlette 的 ``mount()`` 不会触发子应用的
    lifespan**（只处理 http/websocket，见 ``starlette/routing.py`` 的
    ``Router.app``）。AgentScope 的作者本人也在健康检查路由的 docstring 里
    把这一点写成了警告（``app/_router/_health.py``）。

    若按官方 docstring 的 ``root.mount("/agentscope", agentscope_app)`` 写
    （``agentscope/app/_app.py:121-129``），后果是：
        · ``chat_service`` / ``session_service`` / ``scheduler_manager`` 等
          **全部在 lifespan 里才写入** ``app.state``（``app/_lifespan.py``）；
        · lifespan 没跑 ⇒ 这些属性不存在 ⇒ 依赖注入取 ``app.state.chat_service``
          时 **AttributeError** ⇒ **所有业务端点 500**；
        · 且失败方式极具迷惑性：应用能启动、``/docs`` 能打开、
          ``/healthz`` 返回 200，只有真正调业务接口才炸。

    另外，前缀 mount 还会打断官方前端：``examples/web_ui`` 的
    ``client.ts`` 用 ``new URL(path, baseUrl)`` 而调用处传的是**绝对路径**，
    前缀会被 URL 解析规则丢掉。

    **对策**：把 ``create_app`` 的返回值**直接当作根应用**，
    再往上 ``include_router`` / ``add_middleware`` / 包一层 lifespan。
    好处有三个：lifespan 天然执行；**没有前缀问题**（官方前端填根地址即可直连）；
    ``StaticFiles`` 最后挂到 ``/`` 也不会遮蔽 API 路由。

==============================================================================
核心决策二：**自己设置 ``boot_completed``**
==============================================================================
    框架的 lifespan 不设这个属性（它只写 chat_service 等）。没有它，
    ``/readyz`` 就只能靠「PG 通、Redis 通、Milvus 通」去**推断**应用是否
    就绪 —— 而这个推断不成立是完全可以发生的：那三个依赖由**三个独立
    进程**提供，它们全通、而 ``app.state.chat_service`` 尚不存在，
    两者之间没有任何因果关系。此时探针报「就绪」，第一个真实请求却会因为
    取不到 ``chat_service`` 而 500。

    ⚠️ 顺带更正一条曾经写在这里、但**已核实为假**的理由：
    「FastAPI 在 lifespan 的进入段执行完之前就已经开始接受连接」。
    实测 uvicorn 0.53 的 ``Server.startup``：``await self.lifespan.startup()``
    **在前**，真正开始 accept 的 ``loop.create_server(...)`` 在**后**；
    而启动完成的消息要等我们的 lifespan ``__aenter__`` 整个跑完才会发出。
    所以在本项目部署的这套 uvicorn 下，那个「窗口期」**不存在**。
    照原来的说法推理，会得出「boot_completed 是为了堵 uvicorn 的时序」
    这个错误结论，进而觉得换台服务器就没必要留着它 —— 正好反了。

    留着这个标志真正值得，理由是另外两条：

        · 它把「进入段跑完了」变成**可断言的事实**，而不是靠「资源探针
          恰好覆盖了全部前置条件」去推断。今天覆盖得全，是因为我们
          记得同时探 PG/Redis/Milvus；明天框架多一个启动期资源，
          推断就会静默地少一项。
        · 它同时标记**退出段**（退出时置回 ``False``）。退出期的就绪性
          没有别的信号能表达：框架正在拆资源，而三个外部依赖照样是通的
          —— 探针会一直绿到进程消失。

    另外，若哪天换成启动顺序不同的服务器（Hypercorn、Gunicorn 的自定义
    worker、或 uvicorn 的 ``lifespan="off"``），这个标志就是唯一还能
    挡住误报的东西。

    :data:`boot_completed` 把这个窗口期显式标出来，供 ``/readyz`` 判定。
    它由 :func:`_wrap_lifespan` 在框架 lifespan 的进入段**全部完成之后**置为
    ``True``，并在退出段置回 ``False``（退出期间实例正在拆除，不该再接流量）。

==============================================================================
核心决策三：**零密钥可启动**
==============================================================================
    本模块不要求任何密钥即可完成装配：
        · 模型：``build_chat_model`` 在无 key 时降级为 MockLLM
          （见 ``src/llm/factory.py``）；
        · 存储：``AsyncSQLAlchemyStorage`` 与 ``RedisStorage`` 的构造函数
          **不建立任何连接**（连接发生在 ``__aenter__``，即 lifespan 里）；
        · trace：``trace_exporter=none`` 时不注册任何导出器。
    因此「clone 下来直接 ``make up``」是成立的。这条性质很容易在后续开发中
    被无意破坏（比如有人在模块级加一句 ``await storage.ping()``），
    所以单测里有一条专门断言「无密钥也能 import 本模块」。
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator
from urllib.parse import urlparse

from fastapi import FastAPI
from fastapi.middleware import Middleware

from ..config import Settings, get_settings, repo_root
from ..llm.mock import MockCredential
from ..observability.logging import configure_logging
from ..observability.tracing import setup_tracing, shutdown_tracing
from ..storage.engine import (
    build_business_engine,
    business_schema_for,
    ensure_business_schema,
    storage_engine_kwargs,
)
from .constants import (
    BOOT_COMPLETED_ATTR,
    BUSINESS_ENGINE_ATTR,
    MEMORY_ATTR,
    RESOURCE_ACCESS_POLICY_ATTR,
    SETTINGS_ATTR,
)
from .middleware import (
    AuthMiddleware,
    HttpMetricsMiddleware,
    MockCredentialSeedMiddleware,
    RateLimitMiddleware,
    TraceContextMiddleware,
)
from .probes import router as probes_router
from .routers import api_v1_router
from .spa import SpaStaticFiles, static_root_public_paths

logger = logging.getLogger(__name__)

#: ``app.state`` 上的属性名常量（``boot_completed`` / ``business_engine``）。
#:
#: ⚠️ 定义在 :mod:`src.server.constants`，不在这里 —— 本模块第 96 行就
#: ``import probes``，而 ``probes.py`` 要读 ``BOOT_COMPLETED_ATTR``。
#: 名字若定义在下面，``probes.py`` 拿到的会是一个**执行到一半**的本模块，
#: 直接 ImportError。详见那个模块的文档字符串。
#:
#: 这里仍然 import 回来并放进 ``__all__``：调用方照旧写
#: ``from src.server.app import BOOT_COMPLETED_ATTR`` 也能拿到，
#: 但**新代码应当直接从 ``src.server.constants`` 取**，免得再长出循环。

#: 前端静态资源的落盘目录（相对仓库根）。
#: P5 会把 vite 的构建产物输出到这里（``src/server/static/``），
#: 这样**不需要改 Dockerfile** —— 镜像里的 ``COPY src/`` 已经会带上它，
#: 而开发期它被 compose 的 ``./src:/app/src:ro`` 覆盖，改完刷新即可见。
STATIC_DIR_NAME = "static"

#: 工作区根目录（``LocalWorkspaceManager`` 的 basedir）。
#: compose 把它挂成了命名卷 ``app_workspace``，见 docker-compose.yaml 的 app.volumes。
#: 它也是**唯一**保证「容器里可写」的目录 —— 框架的 blob 存储默认根
#: （``./blobs``，相对 CWD ``/app``）就落在这个目录下，见 :func:`build_blob_store`。
WORKSPACE_DIR = "/app/workspace"

#: 知识库原始文件的落盘子目录（相对 :data:`WORKSPACE_DIR`）。
BLOB_DIR_NAME = "blobs"

#: 见上：与 :data:`BOOT_COMPLETED_ATTR` 一样来自 :mod:`src.server.constants`。


# ==============================================================================
# 存储与消息总线
# ==============================================================================
# 引擎参数推导的实现已搬到 ``src.storage.engine`` —— 业务库的 engine 也要用它，
# 而 storage 是被 server 使用的**下层**，依赖方向不允许它反过来 import server。
# 这里保留同名再导出，是因为 ``tests/conftest.py`` 直接
# ``from src.server.app import _storage_engine_kwargs`` —— 那是测试侧的既有契约，
# 删掉这个别名会让一堆用例在收集阶段就 ImportError。
_storage_engine_kwargs = storage_engine_kwargs


def build_storage(settings: Settings) -> Any:
    """构造 AgentScope 的会话/消息/凭据存储。

    ⚠️ 构造函数**不建立任何连接**（连接发生在 lifespan 的 ``__aenter__``）。
    这是「零密钥可启动」得以成立的原因之一，也是本模块能在 import 期安全调用它的前提。

    ``auto_migrate`` 恒为 ``False``（不对外暴露成配置项）：框架源码明确警告
    多副本并发迁移不安全（``agentscope/app/storage/_sql/_storage.py:137-142``）。
    框架侧靠 ``create_tables`` 建表，业务 schema 的迁移由 alembic 负责。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `AsyncSQLAlchemyStorage`: 未进入上下文的存储实例。
    """
    from agentscope.app.storage._sql import AsyncSQLAlchemyStorage

    return AsyncSQLAlchemyStorage(
        url=settings.db.url,
        create_tables=settings.db.create_tables,
        auto_migrate=False,
        engine_kwargs=_storage_engine_kwargs(settings),
    )


def _redis_connection_params(settings: Settings) -> dict[str, Any]:
    """把 ``redis.url`` 拆成 ``RedisStorage`` / ``RedisMessageBus`` 需要的参数。

    框架这两个类的签名收的是**分项参数**（host / port / db / password），
    而不是一条 URL。因此必须在这里解析。

    ⚠️ **超时参数必须在这里补上，否则整条会话链路是无界的。**

    框架那两个类只是把 ``**kwargs`` 原样转交给
    ``redis.asyncio.ConnectionPool``（``agentscope/app/message_bus/_redis_message_bus.py:142-149``、
    ``agentscope/app/storage/_redis_storage.py:233-239``），而 redis-py 的 ``socket_timeout``
    **默认是 None**（= 永不到期）。也就是说：只传 host/port/db/password 的话，
    一个「接受连接但不再回包」的 Redis（网络分区、阻塞、主从切换）
    会让每一次 ``SET`` / ``XADD`` / ``PUBLISH`` 无限等待。

    这不是理论风险，后果在源码里可以直接读出来：

      · ``RedisMessageBus.acquire_lock`` 是 ``while True: await self._client.set(...)``
        （``agentscope/app/message_bus/_redis_message_bus.py:666-670``），框架没有任何外层截止时间；
      · 而框架的每轮对话都在这把锁里面跑（``agentscope/app/_service/_chat.py:797``）

    ⇒ Redis 卡住 = 所有对话永远拿不到锁 = SSE 一条事件都不再推送，
    而 ``/healthz`` 依然是 200（进程活着）。这正是 2026-10-03 Milvus
    那次「服务假死」的同一种形态，只是换了一个依赖。

    ⚠️ ``socket_timeout`` 会不会误伤长连接订阅？不会 —— 框架的 pub/sub
    读循环显式接住了读超时并 ``continue``（``agentscope/app/message_bus/_redis_message_bus.py:600-605``，
    它甚至就是为「连接设了 socket_timeout」这一情形写的）。
    ``socket_connect_timeout`` 单独设置，是为了让「连不上」与「连上了不回话」
    都快速失败。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `dict`: 含 ``host`` / ``port`` / ``db`` / ``password``，以及
        ``max_connections`` / ``socket_timeout`` / ``socket_connect_timeout``。

    Raises:
        ValueError: URL 里没有主机名时。这一条必须显式报错而不是静默用默认值 ——
            静默回落到 localhost 会让容器连到「自己」，症状是「Redis 明明起着，
            应用却读不到别人的会话」，排查方向完全被带偏。

    ⚠️ 报错文案里**绝不能**回显那条 URL。本项目里它形如
    ``redis://:${REDIS_PASSWORD}@redis:6379/0``（``config/base.yaml`` 的
    ``redis`` 段），也就是**密码就在字符串里**。而这段代码在
    ``import src.server.app`` 时就会执行（``create_root_app()`` →
    ``build_message_bus``），异常会带着消息打进容器日志与
    ``docker compose logs`` —— 一次配置写错就把生产密码留在了日志里。

    所以只回显**协议名**这一个安全片段：它足以区分
    「写成了 ``http://``」「压根没写协议」，又不含任何凭据。
    密码本身不需要出现在这里 —— 有密码的 URL 与没密码的 URL
    在「解析不出主机名」这件事上给出的是同一个结论。
    """
    parsed = urlparse(settings.redis.url)
    if not parsed.hostname:
        raise ValueError(
            f"redis.url 解析不出主机名（scheme={parsed.scheme or '空'}）。\n"
            f"预期形如 redis://:密码@redis:6379/0（见 config/base.yaml 的 redis 段）。\n"
            f"⚠️ 这里刻意不回显该 URL：它内嵌密码，回显会把它写进容器日志。",
        )
    # 路径形如 "/0"；无路径时用 0 号库。
    db_index = 0
    if parsed.path and parsed.path.strip("/"):
        try:
            db_index = int(parsed.path.strip("/"))
        except ValueError:
            raise ValueError(
                f"redis.url 的库号不是整数：{parsed.path!r}（预期形如 /0）。",
            ) from None
    return {
        "host": parsed.hostname,
        "port": parsed.port or 6379,
        "db": db_index,
        # password 可能是 None（URL 里没写密码）。框架接受 None，
        # 传空串反而会被当成「密码是空串」而触发 AUTH。
        "password": parsed.password,
        # ⚠️ 下面三项一起走 ``**kwargs`` → ``ConnectionPool``。见 docstring
        # 里「超时参数必须在这里补上」一段：漏掉它们 = 会话链路上有
        # 一处永远不返回的等待。``max_connections`` 也一并补上，
        # 否则 ``redis.max_connections`` 会变成一个**没有任何效果**的配置
        # （SSE 长连接会长期占着池，池子无上限则占用无上限）。
        "max_connections": settings.redis.max_connections,
        "socket_timeout": settings.redis.socket_timeout_seconds,
        "socket_connect_timeout": settings.redis.socket_timeout_seconds,
    }


def build_message_bus(settings: Settings) -> Any:
    """构造消息总线（SSE 事件广播、跨进程取消通知的载体）。

    必须是 ``RedisMessageBus`` 而不是 ``InMemoryMessageBus``：后者的实现注释
    自述「仅适用于单进程」，在容器化部署里会让 SSE 事件丢失 ——
    表现为「聊天页偶尔不刷新，刷新一下又好了」。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `RedisMessageBus`: 未进入上下文的总线实例。
    """
    from agentscope.app.message_bus import RedisMessageBus

    return RedisMessageBus(**_redis_connection_params(settings))


def build_workspace_manager(settings: Settings) -> Any:
    """构造工作区管理器（智能体读写文件的根目录）。

    用 ``LocalWorkspaceManager``（本地目录隔离），理由是 AgentScope 内置的容器/
    云沙箱后端都要额外的 Docker-in-Docker 或云凭据，而本项目跑在 compose 里，
    本地目录隔离已经足够，且便于 ``make logs`` 与人工查看产物。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `LocalWorkspaceManager`: 未进入上下文的工作区管理器。
    """
    from agentscope.app.workspace_manager import LocalWorkspaceManager

    return LocalWorkspaceManager(basedir=WORKSPACE_DIR)


def build_blob_store(workspace_dir: str = WORKSPACE_DIR) -> Any:
    """构造知识库原始文件的 blob 存储（本地文件系统后端）。

    ★ 这是一个**真实事故的修复**，不是「顺手显式化」。

    框架的 ``create_app`` 在「配了知识库管理器」这条分支上会给
    ``app.state.blob_store`` 兜一个默认值::

        app.state.blob_store = blob_store if blob_store is not None \\
            else LocalBlobStore(root_dir="./blobs")      # 相对**进程 CWD**

    而容器的 CWD 是 ``/app``，属主是 root、权限 755，进程却以 uid 1000
    （``aligo``）运行。于是 ``LocalBlobStore.__aenter__`` 里的
    ``makedirs("/app/blobs")`` 抛 **PermissionError**，它发生在 lifespan
    的进入段 ⇒ ``Application startup failed. Exiting.`` ⇒ 容器进入**崩溃循环**。

    为什么 ``make test`` 全绿却漏掉了它：pytest 的 CWD 是**仓库根目录**，
    既可写、又刚好什么都没写坏 —— ``./blobs`` 被静默创建在仓库里。
    也就是说这个缺陷在测试环境里表现为「多了一个空目录」，
    在容器里表现为「服务起不来」。**换个 CWD 就换一种命运**，
    这正是「默认值相对 CWD」这类写法最坏的地方。

    修法是把这个路径钉死在**唯一保证可写的目录**（命名卷 ``app_workspace``，
    属主 aligo）下面：``/app/workspace/blobs`` —— 与工作区管理器同根，
    备份/清理时只需要处理一个目录。

    Args:
        workspace_dir (`str`): 工作区根目录，见 :data:`WORKSPACE_DIR`。

    Returns:
        `LocalBlobStore`: 尚未进入上下文的 blob 存储（目录在 lifespan 里创建）。
    """
    from agentscope.app.rag.blob_store import LocalBlobStore

    return LocalBlobStore(root_dir=str(Path(workspace_dir) / BLOB_DIR_NAME))


def build_storage_backend(settings: Settings) -> Any:
    """构造 Redis 存储（会话状态、锁、缓存）。

    ⚠️ 与 :func:`build_storage` 的区别：本函数建的是 Redis 后端的
    ``StorageBase`` 实现，而 :func:`build_storage` 建的是 PostgreSQL 后端的。
    本项目**用 PostgreSQL 作为 AgentScope 的存储**（会话/消息/凭据表与业务表
    同库不同 schema，省一个组件），因此这里保留 Redis 版本仅供将来切换；
    当前不在装配路径上。

    保留它是刻意的：切换存储后端是一次真实可能发生的需求，
    而把「怎么连 Redis」这段 URL 解析逻辑放在这里，切换时不用重写。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `RedisStorage`: 未进入上下文的存储实例。
    """
    from agentscope.app.storage import RedisStorage

    return RedisStorage(**_redis_connection_params(settings))


# ==============================================================================
# lifespan 包装
# ==============================================================================
async def _seed_system_credential(
    app: FastAPI,
    settings: Settings,
    system_box: dict[str, Any],
) -> None:
    """把运营者配置的真实凭据写成一条**全员只读可用**的系统凭据。

    这是「探针全绿但一个字都发不出去」在**有密钥**部署上的那一半解法：
    密钥配了、模型调得通，但框架默认的 owner 隔离让任何用户都看不见
    运营者那条凭据，于是浏览器里「可用模型」是空的。完整论证见
    ``src/llm/system_credential.py``。

    本函数**从不抛异常**（除了 ``BaseException``）。理由与代价都写下来：

        · 不抛的理由 —— 共享凭据是可用性增强，不是启动前提。让一次
          数据库抖动把整个服务变成崩溃循环，是拿一个次要功能去换主功能。
        · 代价说清楚 —— 失败时 ``/api/v1/default-model`` 会返回 ``missing``
          而不是 ``shared``。这是**如实**的：策略的 ``seeded`` 没置位，
          用户此刻确实用不了那条凭据。不会出现「说可以用、点了报 404」。
        · 日志给到 ERROR 且带栈（``logger.exception``）：这一条**不能**
          只留一行 warning —— 它的症状（所有人都在 ``missing``）
          与「运营者压根没配密钥」完全一样，没有栈就无从区分。

    ⚠️ 每次进入 lifespan 都**先置回 False** 再尝试。不重置的话，
    第二次进入（测试里常见）若播种失败，盒子还留着上一轮的 ``True``，
    策略会继续分享一条**本次启动根本没写成功**的凭据。

    Args:
        app (`FastAPI`): 当前应用（用于取框架挂好的 storage）。
        settings (`Settings`): 全量配置。
        system_box (`dict[str, Any]`): 与策略对象共享的状态盒子。
    """
    # 局部 import：``src.llm.system_credential`` 会拉进 ``agentscope.app.access``，
    # 而本模块的 import 期就调用了 :func:`create_root_app`（模块级 ``app = ...``）。
    # 放在模块级会把「零密钥可启动」这条路径的导入面扩大到框架的鉴权子包 ——
    # 多一个在 import 期出岔子的地方，而收益只是省下一次函数内查找。
    from ..llm.system_credential import (
        SYSTEM_USER_ID,
        ensure_system_credential,
        sharing_enabled,
    )

    system_box["seeded"] = False
    if not sharing_enabled(settings):
        # 零密钥部署：没有可共享的东西。这是**正常**路径，不打日志。
        return

    storage = getattr(app.state, "storage", None)
    if storage is None:
        logger.error(
            "系统凭据播种被跳过：app.state 上没有 storage（应用未按预期装配）。"
            "所有用户都会看到 mode=missing。",
        )
        return

    try:
        credential_id = await ensure_system_credential(storage, settings)
    except Exception:  # noqa: BLE001 —— 见 docstring：绝不阻断启动
        logger.exception(
            "系统凭据播种失败：所有用户都会看到 mode=missing（与「运营者没配密钥」"
            "表现完全相同，请按本条栈排查）。服务继续启动，其余功能不受影响。",
        )
        return

    if credential_id is None:
        # sharing_enabled 与 ensure_system_credential 用了同一个判据，
        # 走到这里说明两者分叉了（本文件刚判过 True）。不静默吞掉。
        logger.error("系统凭据播种返回 None，但共享判据为真 —— 判据分叉，请检查代码。")
        return

    system_box["seeded"] = True
    # ⚠️ INFO 级别、且把「只读」写进文案：这条日志是运维**唯一**会看到
    # 「公司密钥正在被全员使用」的地方。等级太高会被忽略，太低则淹没在
    # 启动噪音里；文案里不带任何密钥片段（只有记录 id）。
    logger.info(
        "已播种系统模型凭据 %s（属主 %s）：全体已鉴权用户可**只读使用**"
        "（列表里打码为 {type, name}，运行期由框架解析原始记录）。"
        # ⚠️ 这里**只能**给「清空 api_key」这一条路：判据是
        # ``not should_use_mock and has_api_key``，而
        # ``should_use_mock = use_mock_when_no_key and not has_api_key`` ——
        # 只要 key 非空，``use_mock_when_no_key`` 取任何值都得到
        # ``should_use_mock=False``，共享照旧。早先的文案把该开关列为
        # 第二条路（照抄了直觉），照着做会发现「改了没用」。
        "如需关闭，清空 llm.api_key（或覆盖 ALIGO__LLM__API_KEY=）——"
        "⚠️ 单改 llm.use_mock_when_no_key 无效：它只在 api_key 为空时才参与判据。",
        credential_id,
        SYSTEM_USER_ID,
    )


def _wrap_lifespan(
    framework_lifespan: Any,
    settings: Settings,
    engine_box: dict[str, Any] | None = None,
    system_box: dict[str, Any] | None = None,
) -> Any:
    """把框架的 lifespan 包一层，插入我们自己的启动/关闭逻辑。

    执行顺序（外层到内层）::

        配置日志
        setup_tracing()                 ← 先于框架启动，才能采到启动阶段的 span
        ┌─ 框架 lifespan 进入段 ─────────────────────────────┐
        │  storage / message_bus / workspace / scheduler ... │
        └────────────────────────────────────────────────────┘
        boot_completed = True           ← 关键：此刻所有资源才真正就绪
        ---- yield：开始对外服务 ----
        boot_completed = False          ← 退出段开始，不该再接新流量
        ┌─ 框架 lifespan 退出段 ─────────────────────────────┐
        └────────────────────────────────────────────────────┘
        shutdown_tracing()              ← 冲刷尚未上报的 span

    ⚠️ ``boot_completed`` 置位的位置是这段代码的**全部要点**：
    必须严格在 ``async with framework_lifespan(app)`` 的**进入段之后**，
    不能放在它之前（那时资源还没就绪），也不能放进 yield 之后（那时已经开始服务了）。

    Args:
        framework_lifespan (`Any`): 框架原始的 lifespan 上下文管理器函数。
        settings (`Settings`): 全量配置。
        engine_box (`dict[str, Any] | None`, optional): 长期画像仓储用来
            取「当前业务库引擎」的盒子。由 :func:`create_root_app` 创建并
            同时交给 :func:`src.memory.build_memory` —— 两者必须拿到
            **同一个**字典对象，否则画像会一直以为引擎还没建好。
            ``None`` 时本函数不填它（单测直接调用时的路径）。
        system_box (`dict[str, Any] | None`, optional): 系统凭据共享的
            状态盒子（键 ``seeded``）。由 :func:`create_root_app` 创建并与
            交给 ``create_app`` 的策略对象**共享同一个字典** ——
            策略要读它、本函数要写它，两边必须是同一个对象，
            否则策略永远读不到 ``True``（共享静默失效）。
            ``None`` 时本函数不播种（单测直接调用、或运营者没配密钥）。

    Returns:
        `Any`: 包装后的 lifespan 函数（供 ``app.router.lifespan_context`` 使用）。
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """包装后的应用生命周期。

        Args:
            app (`FastAPI`): 应用实例。

        Yields:
            `None`: 让 FastAPI 在 yield 期间对外服务。
        """
        configure_logging(settings)
        setup_tracing(settings)

        logger.info(
            "AliGo 差旅助手启动中：env=%s workers=%s",
            settings.app.env,
            settings.app.workers,
        )

        # 业务库引擎。**构造它不建立任何连接**（惰性拨号），所以放在这里
        # 与放在模块级没有连接上的区别；放在 lifespan 里是为了让它的生命周期
        # 与本次「进入—退出」严格配对 —— 每次进入新建、每次退出 dispose。
        # 放模块级的话，测试里反复进出 lifespan 会共用同一个已 close 的引擎，
        # 症状是第二个用例报 "Event loop is closed" 或连接已失效。
        business_engine = build_business_engine(settings)
        setattr(app.state, BUSINESS_ENGINE_ATTR, business_engine)
        # ⚠️ 同步填进盒子：长期画像仓储通过它拿到**当前**引擎。
        # 这一行与下面退出段的 ``engine_box["engine"] = None`` 必须成对 ——
        # 否则 dispose 之后盒子还指着那个已关闭的引擎，
        # 而症状是「重启 lifespan 后所有画像读写报 Event loop is closed」。
        if engine_box is not None:
            engine_box["engine"] = business_engine

        try:
            async with framework_lifespan(app):
                # 业务 schema 的引导**必须在 boot_completed 置位之前**完成。
                # 反过来的话，/readyz 会在 schema 还没建好的那几十毫秒里
                # 报「就绪」，而此刻打进来的请求会撞上
                # ``relation "business.xxx" does not exist``。
                # 就绪探针的语义是「现在打进来能成功」，不是「马上就能成功」。
                await ensure_business_schema(
                    business_engine,
                    business_schema_for(settings),
                )

                # 画像表同理：它建在 ``business`` schema 里，因此必须在
                # ``ensure_business_schema`` **之后**；也必须在
                # boot_completed 之前 —— 否则就绪探针会在一段
                # 「画像表还没建好」的窗口里报 200。
                #
                # ⚠️ 建表失败**不让启动失败**：画像是对话的增强，不是前提。
                # 代价说清楚 —— 此时 ``/readyz`` 报的就绪是真的，
                # 但「记住偏好」这类写入会失败，而它会以异常的形式
                # 出现在用户面前（不是静默），所以不会有人被蒙在鼓里。
                if getattr(app.state, MEMORY_ATTR, None) is not None:
                    # ⚠️⚠️ 这个 import **必须留在这里**，它是一个真实事故的修复。
                    #
                    # 原本 ``ensure_profile_table`` 只在 :func:`create_root_app`
                    # 的函数体里 import 过一次（那里也要用它来给
                    # :func:`src.memory.build_memory` 传参）。但本函数
                    # （:func:`_wrap_lifespan` 内的 lifespan）是**另一个作用域** ——
                    # 它看不见那个名字，于是这一行每次都抛 ``NameError``。
                    #
                    # 而 ``NameError`` 是 ``Exception`` 的子类，正好落在下面那个
                    # ``except Exception`` 里：**画像表从来没被建过**，
                    # 启动日志里只有一行被吞掉的 ERROR 栈，``/readyz`` 照样 200。
                    # 症状要等到第一次写画像时才出现，那时离病因已经很远了。
                    #
                    # 教训写在这里：`except Exception` 兜住的异常一旦包含
                    # **代码自身的错误**（NameError / AttributeError / TypeError），
                    # 「降级」就变成了「掩盖」。降级只该兜住环境类失败
                    # （连不上、没权限、表被别人改了）。
                    from ..memory import ensure_profile_table

                    try:
                        await ensure_profile_table(
                            business_engine,
                            business_schema_for(settings),
                        )
                    except Exception:  # noqa: BLE001 —— 见上
                        logger.exception(
                            "画像表初始化失败，长期记忆的**写入**将不可用"
                            "（读取会正常降级为「没有画像」）。",
                        )

                # ---- 系统凭据：把运营者配的真密钥写成**一条**共享凭据 -------
                # 必须在 boot_completed **之前**：播种完成后用户第一次打
                # /credential/ 才看得见它。反过来的话，就绪探针已经报 200，
                # 而这段时间里进来的请求会得到「可用模型为空」——正是本机制
                # 要消灭的那个状态。
                #
                # ⚠️ 顺序的**另一半**：必须在 ``async with framework_lifespan``
                # 的进入段**之后** —— storage 是在那里面才进入异步上下文的，
                # 放到外面写库会撞上「存储尚未启动」。
                #
                # ⚠️ 失败**不让启动失败**：共享凭据是可用性增强，不是前提。
                # 唯一的对外影响是 ``/api/v1/default-model`` 返回 ``missing``
                # （而不是 ``shared``），那是**如实**的答复 ——
                # 策略的 ``seeded`` 没置位，用户也确实用不了。
                if system_box is not None:
                    await _seed_system_credential(app, settings, system_box)

                # ★ 框架的全部生命周期资源此刻才真正进入异步上下文。
                setattr(app.state, BOOT_COMPLETED_ATTR, True)
                logger.info(
                    "启动完成：storage / message_bus / workspace / scheduler 均已就绪，"
                    "就绪探针 /readyz 现在会返回 200。",
                )
                try:
                    yield
                finally:
                    # 退出段一开始就把标志清掉：此时框架正在拆除资源，
                    # 继续对外声明「就绪」会让负载均衡把新请求送进一个
                    # 正在拆连接的实例，表现为一批 5xx。
                    setattr(app.state, BOOT_COMPLETED_ATTR, False)
        finally:
            # 放在 finally 里：即使框架启动失败（比如 PG 连不上），
            # 也要把已经注册的 TracerProvider 关掉并冲刷。
            # 否则一个启动失败的进程会留下一个后台导出线程，
            # 而它的日志会混进真正的启动错误里，干扰判断。
            #
            # 业务引擎的 dispose 同样放在 finally 里，顺序**先于**
            # shutdown_tracing。
            #
            # ⚠️ 这里曾经给出的理由是「调换顺序会让最后几条驱动的告警日志
            # 失去 trace_id」—— **已核实为假**。日志里的 ``trace_id`` 来自
            # ``src/observability/context.py`` 的 ContextVar，由
            # ``src/observability/logging.py`` 的过滤器读出来，与
            # TracerProvider 存不存在、关没关**没有任何关系**。
            # 先关 tracing 不会让日志少任何一个字段。
            #
            # 真正的理由只有一条：dispose 期间**仍可能产生 span**。
            # 今天业务引擎没有接 instrumentation，所以是潜在的；一旦接上
            # SQLAlchemy 埋点就必然有。provider 关掉之后新建的 span 会被
            # 直接丢弃，于是 trace 里恰好缺掉「关闭时到底发生了什么」那一段
            # —— 而那一段正是最需要证据的地方。
            # 代价为零（两次调用之间不共享任何状态），顺序就按这个来。
            # ⚠️ 先把盒子清空**再** dispose。顺序反了的话，从 dispose 开始
            # 到清空之间有一小段窗口，此时盒子还指着一个正在关闭的引擎，
            # 恰好落在窗口里的画像查询会拿到一个半死的连接。
            if engine_box is not None:
                engine_box["engine"] = None
            try:
                await business_engine.dispose()
            except Exception:  # noqa: BLE001
                # 关闭失败不该掩盖真正的启动错误：dispose 抛异常会替换掉
                # 正在向上传播的那个异常（Python 的异常链会给出一堆
                # "During handling of the above exception"，把根因埋掉）。
                logger.exception("业务库引擎关闭时出错（不影响进程退出）。")

            shutdown_tracing()
            logger.info("AliGo 差旅助手已停止。")

    return lifespan


# ==============================================================================
# 应用装配
# ==============================================================================
def create_root_app(
    settings: Settings | None = None,
    *,
    storage: Any | None = None,
    message_bus: Any | None = None,
    workspace_manager: Any | None = None,
    blob_store: Any | None = None,
    enable_scheduler: bool = True,
    agent_wiring: Any | None = None,
) -> FastAPI:
    """装配并返回根 FastAPI 应用。

    本函数是**纯装配**：不建立任何网络连接、不读 .env 之外的外部状态，
    因此可以在单测里安全调用。

    Args:
        settings (`Settings | None`): 配置；``None`` 时取进程内单例。
        storage (`Any | None`): 覆盖存储实现。**仅供测试注入**，见下方说明。
        message_bus (`Any | None`): 覆盖消息总线实现。**仅供测试注入**。
        workspace_manager (`Any | None`): 覆盖工作区管理器。**仅供测试注入**。
        blob_store (`Any | None`): 覆盖知识库 blob 存储。**仅供测试注入** ——
            测试的 CWD 与工作区都不是容器里的那两个路径，注入一份指向
            ``tmp_path`` 的实现才能既写进临时目录、又不留下垃圾。
            生产默认值见 :func:`build_blob_store`（**必须是绝对路径**）。
        enable_scheduler (`bool`): 是否启用调度器。
        agent_wiring (`Any | None`): 智能体装配产物
            （:class:`~src.server.agents_factory.AgentWiring`）；
            ``None`` 时按配置构造。**仅供测试注入** —— 传它可以绕开
            「构造真实模型」这一步，让应用装配的单测不需要模型凭据。

    Returns:
        `FastAPI`: 可直接交给 uvicorn 的应用。

    ⚠️ 为什么开放这几个覆盖参数（以及为什么不做成配置开关）
        生产环境**永远不要**传它们 —— 默认值就是从 ``Settings`` 构造的那一套。

        需要它们的原因：框架的 ``SchedulerManager`` 会订阅消息总线上的调度事件，
        而订阅失败时它**会无限重试**（日志表现为反复的
        ``schedule lifecycle subscription lost``）。这在生产是正确行为
        （Redis 短暂抖动应当自愈），但会让「不依赖 Docker 的 ``make test``」
        永远卡在启动阶段。

        刻意**不做成** ``ALIGO__REDIS__USE_IN_MEMORY`` 之类的配置开关：
        那种开关一旦被误设到生产，症状是「SSE 事件在单进程内正常、
        多副本之间收不到」—— 一个只在扩容时才暴露、且与「配置写错了」
        这件事看起来毫无关系的故障。用**参数注入**则不可能被配置误触。
    """
    settings = settings or get_settings()

    from agentscope.app import create_app

    # ---- P4：长期记忆 --------------------------------------------------------
    # ⚠️ 必须在 ``create_app`` **之前**装配（与下面 P3 同一条理由：中间件
    # 工厂只在 create_app 内部被读一次），而它要用的业务库引擎却在
    # lifespan 进入段才创建。两者之间隔着 create_app 本身。
    #
    # 解法是把「取引擎」推迟成一个**函数**（``engine_box`` 是本次
    # create_root_app 调用私有的，不是模块级全局 —— 每次装配一个盒子，
    # 因此不同的应用实例之间不会互相串）。这样：
    #   · 装配期：不需要引擎，盒子是空的；
    #   · lifespan 进入：盒子被填上；
    #   · 第一次查画像：取到的是**当时活着**的那个引擎。
    # 应用实例被重建时（测试里常见），取函数自然返回新引擎，
    # 不需要任何「换引擎」的代码，也就没有换漏的可能。
    from src.memory import build_memory, ensure_profile_table

    engine_box: dict[str, Any] = {"engine": None}
    memory = build_memory(settings, engine=lambda: engine_box["engine"])

    # ---- P4：知识库（Milvus 单集合） -----------------------------------------
    # ⚠️ **必须排在智能体装配之前** —— 这是本次调整顺序的**全部理由**。
    # RAG 中间件要用它：装配工厂把 ``kb_manager`` 交给
    # ``src.knowledge.rag.build_rag_middlewares`` 去解析「这个用户的 KB 句柄」。
    # 若还按原来的顺序（先 wiring、后 KB），``build_agent_wiring`` 只能拿到
    # ``None``，检索中间件压根装不上 —— 而症状是「知识库建好了、问答却不带
    # 任何出处」，全程不报错，只在用户抱怨「它怎么没查资料」时才显形。
    #
    # ⚠️ storage 在这里**先落一个本地变量**，因为 create_app、知识库管理器
    # 与 RAG 桥接层必须拿到**同一个实例**。各造一个的后果是「KB 记录写进了 A，
    # 对话链路从 B 里读」，症状是「刚建的知识库列表里看得见、
    # 检索时说找不到」—— 一个看起来像缓存问题的数据一致性问题。
    resolved_storage = storage if storage is not None else build_storage(settings)

    # ⚠️ 这里**不连接** Milvus。``build_vector_store`` 只存参数，
    # 真正的 pymilvus 客户端是惰性创建的；知识库管理器进入生命周期时
    # 也不会连接（``VectorStoreBase.__aenter__`` 是 no-op）。
    # 所以 Milvus 没起来**不影响启动** —— 这正是 P4 验收里
    # 「Milvus 不可用不得拖垮服务」这一条在装配期的落点。
    from src.knowledge import build_knowledge_manager

    # ⚠️ 第三个「盒子」，与上面的 ``engine_box`` 同一个形状，理由也一样：
    # **装配期拿到、就绪期才成立**。
    #
    # 知识库管理器要能在解析 embedding 凭据时回落到资源访问策略（否则
    # 「用共享凭据建的知识库」会在检索时静默失效，见
    # ``SingleCollectionKbManager._resolve_embedding_credential``），
    # 而策略是**框架**在 ``create_app`` 里才写进 ``app.state`` 的 ——
    # 也就是说，本行执行时那个对象还不存在。
    #
    # 所以传进去的不是策略对象，而是一个「到时候再取」的函数。取不到时
    # 返回 ``None``，管理器退化成「只按属主查」= 框架自带管理器的行为，
    # 这是安全的默认值：少解析出一条共享凭据，代价是那个知识库报
    # 「凭据不存在」，而不是错误地解析出**别人的**凭据。
    app_box: dict[str, Any] = {"app": None}

    def _access_policy_provider() -> Any:
        """延迟取资源访问策略；应用尚未建好时返回 ``None``。"""
        app = app_box["app"]
        if app is None:
            return None
        # 用常量而不是字面量：读不到时的表现是**静默**回落到不共享，
        # 与「框架回落到 deny-all」一模一样（见 constants.py 的说明）。
        return getattr(app.state, RESOURCE_ACCESS_POLICY_ATTR, None)

    kb_manager = build_knowledge_manager(
        settings,
        resolved_storage,
        access_policy_provider=_access_policy_provider,
    )

    # ---- P3：智能体装配 ------------------------------------------------------
    # ⚠️ 在 ``create_app`` **之前**做。装配会构造模型（真实调用时含 HTTP
    # 客户端连接池）与工具集；放在 create_app 之后的话，应用已经带着
    # 「没有业务能力」的状态构造完了，再补参数也补不进去 ——
    # ``extra_agent_tools`` 这类参数只在 ``create_app`` 内部被读一次。
    #
    # ⚠️ ``kb_manager`` 传进去，让中间件工厂能装配 RAG（见上）。它**必须是
    # 上面那一个实例** —— 传给 create_app 与传给 wiring 的必须是同一份。
    if agent_wiring is None:
        from src.server.agents_factory import build_agent_wiring

        agent_wiring = build_agent_wiring(
            settings,
            memory=memory,
            kb_manager=kb_manager,
        )

    # ---- 系统凭据共享（配了真密钥的部署才启用） ------------------------------
    # ⚠️ 又是「装配期需要、就绪期才成立」的那一类，与上面 engine_box 同形：
    # 策略对象必须在 ``create_app`` **之前**交给框架（它只在内部读一次），
    # 而「共享到底成没成」要等到 lifespan 里那条凭据写成功才知道。
    # 解法同样是一个**每次装配私有的**盒子：策略读它、lifespan 写它。
    #
    # 没配密钥时 ``build_access_policy`` 返回 None，create_app 回落到框架
    # 默认的 ``DenyAllResourceAccessPolicy``（owner 隔离）—— 那正是零密钥
    # 部署应有的行为：此时由 Mock 降级那条线负责，两条线不重叠。
    from src.llm.system_credential import build_access_policy

    system_box: dict[str, Any] = {"seeded": False}
    resource_access_policy = build_access_policy(settings, system_box)

    app = create_app(
        storage=resolved_storage,
        message_bus=(
            message_bus if message_bus is not None else build_message_bus(settings)
        ),
        workspace_manager=(
            workspace_manager
            if workspace_manager is not None
            else build_workspace_manager(settings)
        ),
        # 知识库管理器：本项目用**单集合**策略（所有 KB 共用
        # ALIGO__MILVUS__COLLECTION，靠 metadata_filter 做 KB/租户隔离）。
        # ⚠️ 不能用框架的 CollectionPerKbManager —— 它给每个 KB 建一个集合，
        # 那会把契约指定的集合名变成死配置。详见 src/knowledge/manager.py。
        #
        # ⚠️ 用的是上面那个 ``kb_manager`` 本地变量（而不是在这里再调一次
        # ``build_knowledge_manager``）：RAG 桥接层的句柄缓存挂在管理器对象上
        # （``src/knowledge/rag.py`` 的 WeakKeyDictionary），两个实例各带一份
        # 缓存、各连一个向量库 —— 检索路径与这里必须共享**同一个**管理器。
        knowledge_base_manager=kb_manager,
        # blob_store：知识库文档的原始文件落盘处（上传 → 解析 → 分块 → 入向量库）。
        # ⚠️ 必须显式传：走 knowledge_base_manager 这条分支时，框架会**兜一个
        # 相对 CWD 的默认值** ``./blobs``，而容器 CWD ``/app`` 不可写
        # （uid 1000 对 root:root 755）⇒ makedirs 抛 PermissionError ⇒
        # lifespan 进入段失败 ⇒ 容器崩溃循环。完整事故链见 build_blob_store。
        blob_store=blob_store if blob_store is not None else build_blob_store(),
        # enable_scheduler：APScheduler 的 jobstore 在内存里，每个 worker 进程
        # 都会独立触发一次 cron。本项目保持 WORKERS=1（见 config/base.yaml），
        # 因此这里是 True。若将来要开多 worker，**必须**在除一个之外的所有进程
        # 上传 False，否则定时任务会被执行 N 次。
        enable_scheduler=enable_scheduler,
        # enable_channel_worker：本项目不接钉钉/飞书等外部 IM（channels 未传），
        # 关掉它省一个长连接工作线程。
        enable_channel_worker=False,
        download_secret=settings.app.download_secret.strip() or None,
        # resource_access_policy：跨属主的凭据可见性规则。我们传的是
        # 「系统凭据全員只读」这一条（见 src/llm/system_credential.py）；
        # 没配密钥时是 None，框架回落到 deny-all（owner 隔离）。
        resource_access_policy=resource_access_policy,
        title="AliGo 差旅助手",
        version="1.0.0",
        # extra_credentials：注册本项目的自定义凭据类型。
        # ``create_app`` 是**唯一**的注册入口（``app/_app.py`` 里调
        # ``CredentialFactory.register_credential``），不在这里传，
        # 框架的 ``CredentialFactory._classes`` 里就只有它自带的 10 种
        # （Anthropic / DashScope / DeepSeek / Gemini / MiniMax / Moonshot /
        # Ollama / OpenAI / XAI / Volcengine），**没有** MockCredential。
        #
        # 后果不是「Mock 用不了」这么轻：零密钥启动时（README 承诺的开箱即用路径）
        # 应用要做到「起来并给出可解释的降级回复」，而框架的
        # ``CredentialFactory.from_dict`` 遇到未注册的 type 会直接抛
        # ValidationError —— 也就是**整个模型链路 500**，而不是降级。
        extra_credentials=[MockCredential],
        # ---- P3：智能体层扩展点 ------------------------------------------
        # 三个扩展点把本项目的业务能力接进框架的对话链路。装配逻辑全部在
        # ``src/server/agents_factory.py``，这里只负责把结果展开。
        #
        # ⚠️ 这三者的**类型形态完全不同**（两个 async 工厂 + 一个静态列表），
        # 传错任何一个的报错点都离原因很远。详见该模块的文档。
        **agent_wiring.as_kwargs(),
    )

    # ---- 替换 lifespan ------------------------------------------------------
    # create_app 不接受自定义 lifespan 参数，它把框架的 lifespan 写死在
    # FastAPI(...) 构造里（agentscope/app/_app.py:292）。因此这里直接替换路由器上的
    # ⚠️ 这一行让上面那个 ``_access_policy_provider`` 从此刻起能取到策略。
    # 必须在 ``create_app`` **之后**：策略对象是**框架**在 create_app 内部
    # 写进 ``app.state`` 的（我们传进去的只是「用哪一条规则」）。写在之前
    # 会永远取到 None，而症状是「用共享凭据建的知识库检索时静默失效」——
    # 没有任何报错，只表现为「它怎么没查资料」。
    # （``app_box`` 的定义与完整理由见 kb_manager 构造处的注释。）
    app_box["app"] = app

    # lifespan_context —— 这是 Starlette 读取 lifespan 的唯一位置。
    from agentscope.app._lifespan import lifespan as framework_lifespan

    app.router.lifespan_context = _wrap_lifespan(
        framework_lifespan,
        settings,
        engine_box,
        # 只有在真的注册了共享策略时才把盒子交下去：没注册策略却播种，
        # 会写出一条**没有任何人能看见**的凭据（策略没生效 ⇒ 列表里不出现），
        # 却要付一次数据库写入。传 None 让这条路径在装配期就断掉。
        system_box if resource_access_policy is not None else None,
    )

    # ---- 初始状态 -----------------------------------------------------------
    # 显式置 False（而不是依赖 getattr 的默认值）：让 boot_completed 在
    # 应用对象一造出来就存在，语义明确，也让单测可以直接断言。
    setattr(app.state, BOOT_COMPLETED_ATTR, False)

    # 全量配置：/api/v1/default-model 要按 settings.llm 判断零密钥降级是否生效
    # （见 src/server/constants.py::SETTINGS_ATTR 的说明 —— 框架不会替我们写，
    # 它自己只往 app.state 写 storage / message_bus 这些运行期对象）。
    # ⚠️ 必须挂**本次装配用的那一份** settings，而不是让读取方自己再
    # get_settings() 一次：单测用显式传入的 settings 构造应用，两者可能不同源，
    # 症状是「测试里明明关掉了降级，接口却按开启处理」。
    setattr(app.state, SETTINGS_ATTR, settings)

    # 长期记忆门面：lifespan 要读它来决定建不建画像表，/api/v1 的画像接口
    # 与单测也要读。**在 create_app 之后挂上去**（而不是之前）是因为那之前
    # 还没有 ``app`` 对象 —— 而中间件工厂拿到的是一个闭包，
    # 闭包里捕获的是 ``memory`` 本身，与它挂在哪无关。
    setattr(app.state, MEMORY_ATTR, memory)

    # ---- 我们自己的运维路由 -------------------------------------------------
    # 框架自带的是 /health（见 app/_router/_health.py），与本项目的
    # /healthz / /readyz / /metrics **不冲突**。保留框架的 /health 不做处理:
    # 它由框架维护，语义（是否含就绪判定）由框架决定，我们不去干预。
    app.include_router(probes_router)

    # ---- 我们自己的业务路由（/api/v1/**）------------------------------------
    # ⚠️ 与框架路由**并存**，不是替换。框架把 /chat/ /sessions/** /agent 等
    # 直接注册在根路径上（``app/_app.py`` 里 ``include_router`` 未加 prefix），
    # 那是本项目 P2 流式验收要用的通道，我们照用。
    # ``/api/v1`` 是**并行的**命名空间，放差旅业务域（订单/行程/申请单）。
    # 两套路径不重叠，因此不存在遮蔽问题 —— 但注册顺序仍要在
    # ``_mount_static_if_present`` 之前（见该函数），否则挂到 "/" 的
    # StaticFiles 会把它们全部吃掉。
    app.include_router(api_v1_router)

    # ---- 中间件 -------------------------------------------------------------
    # ⚠️⚠️ 顺序陷阱（本项目最容易写反的一处）：
    #     add_middleware 内部是 `user_middleware.insert(0, ...)`
    #     （starlette/applications.py），而 build_middleware_stack 是
    #     `for cls, ... in reversed(middleware): app = cls(app)`
    #     ⇒ **下标 0 = 最外层**，即 **列表中越靠后 = 越外层**。
    #
    #     换句话说，下面这个列表的**书写顺序与执行顺序完全相反**：
    #         列表 [HttpMetrics, TraceContext]
    #         执行 HttpMetrics 之后是 TraceContext？不 ——
    #         下标 0 是 TraceContext（它最后被 add，插到了最前）
    #         ⇒ 实际执行 TraceContext → HttpMetrics → 应用。
    #
    # 为什么必须是 TraceContext 在外层：trace_id 必须在**进入任何其他中间件
    # 之前**就绑好。它在内层的话，外层中间件（P2 的鉴权、限流）产生的日志
    # ——恰恰是最需要 trace_id 的那几条「401 / 429」—— 会打出一串 `[-]`，
    # 而功能完全正常，只在排障时才发现。
    # 目标执行顺序（由外到内，P2 完整版）：
    #
    #     TraceContext → HttpMetrics → Auth → RateLimit → 应用
    #
    # 因此**书写顺序**（也就是实际 add 的顺序）完全相反：
    #
    #     [RateLimit, Auth, HttpMetrics, TraceContext]
    #      ^最后执行                ^最先执行
    #
    # 逐条理由见 src/server/middleware/__init__.py 的模块文档字符串，
    # 其中两条最容易写反、也最值得在这里重述：
    #   · HttpMetrics 在 Auth 之外 ⇒ 401/429 也进指标。放里面的话，
    #     鉴权失败率与限流率这两条最该盯的曲线会恒为 0 —— 而且看起来
    #     完全正常（有流量、有延迟），只是永远看不到被拒绝的那部分。
    #   · TraceContext 在最外层 ⇒ 401/429 的日志也带 trace_id。
    #
    # 依赖注入：Auth 与 RateLimit 都需要 settings。
    # RateLimit **必须**在 Auth 之内（即先于它执行的只有 TraceContext/HttpMetrics），
    # 因为它要读 Auth 写进 scope["state"] 的身份来当限流键；顺序反了的话
    # 所有请求都会退化成按 IP 限流，多人共用出口 IP 时会互相误伤。
    #
    # Auth 还要一份「静态产物根目录下有哪些文件」—— 即 extra_public_paths。
    # 这份清单必须在**装配期**算好：浏览器取 index.html 引用的子资源
    # （/agentscope.svg 之类）时不带任何凭据，漏一个就是页面上一个碎图标，
    # 而漏掉的原因（产物里多了个文件）在服务端日志里完全看不出来。
    # 放在这里而不是让中间件自己去读盘：静态目录的位置是装配根节点的知识，
    # 中间件保持「不读盘、不猜」。⚠️ 它必须在下面的 StaticFiles 挂载**之前**
    # 算出来 —— 顺序反了不会报错，只会得到一份空的清单。
    extra_public_paths = static_root_public_paths(_static_dir())
    # ★ 「这个路径有没有后端端点」的判据，由**路由表**算出来，交给鉴权中间件。
    #
    # ⚠️ 必须在写中间件列表**之前**、且在 ``_mount_static_if_present`` **之前**
    # 算：静态目录那个挂载会匹配一切路径（``Mount("/")``），把它算进路由表的话
    # 每个前端路由都会被判成「有端点」，浏览器导航再也不放行 ——
    # 症状是刷新 /chat 直接 401。``build_api_path_matcher`` 只收 ``Route``、
    # 不收 ``Mount``，因此这里的顺序天然安全（``app.routes`` 此刻还没有那个挂载）。
    #
    # ⚠️ 它修的是一个**真实的越权入口**：见 build_api_path_matcher 的文档字符串。
    from .spa import build_api_path_matcher

    api_path_matcher = build_api_path_matcher(app.routes)
    for middleware in [
        Middleware(RateLimitMiddleware, settings=settings),  # 先加 ⇒ 下标最大 ⇒ 最内层
        # 降级凭据播种：**必须在 Auth 之内**（它要读 Auth 写进 scope 的身份），
        # 又必须在 RateLimit 之外（限流不该被一次数据库写入拖慢）。
        # 零密钥降级关闭时它整条快路径直接透传，见该中间件模块的文档。
        Middleware(
            MockCredentialSeedMiddleware,
            settings=settings,
            storage=resolved_storage,
        ),
        Middleware(
            AuthMiddleware,
            settings=settings,
            extra_public_paths=extra_public_paths,
            # 见上：浏览器导航只对「没有后端端点」的路径放行。
            api_path_matcher=api_path_matcher,
        ),
        Middleware(HttpMetricsMiddleware),
        Middleware(TraceContextMiddleware),  # 后加 ⇒ 下标 0 ⇒ 最外层
    ]:
        app.add_middleware(middleware.cls, **middleware.kwargs)

    # ---- 前端静态资源（**必须最后**）-----------------------------------------
    # 挂到 "/" 的 StaticFiles 会匹配所有未命中的路径，所以它必须在所有 API
    # 路由注册完之后再挂 —— 否则先挂它会把 API 路由全部遮蔽。
    _mount_static_if_present(app)

    return app


def _static_dir() -> Path:
    """前端产物目录（**可能不存在**：P1~P4 阶段前端还没构建）。

    Returns:
        `Path`: ``<repo>/src/server/static``。
    """
    return repo_root() / "src" / "server" / STATIC_DIR_NAME


def _mount_static_if_present(app: FastAPI) -> None:
    """若前端产物目录存在，则把它挂到 ``/``。

    **不做「目录不存在就报错」**：P1~P4 阶段前端还没构建，目录可能不存在，
    而「目录不存在」不该阻止整个服务启动 —— 后端 API 与 SSE 完全不依赖前端。
    只有 P5 构建出产物后，``/`` 才会返回前端页面。

    挂的是 :class:`~src.server.spa.SpaStaticFiles`（而不是裸 ``StaticFiles``）：
    ``html=True`` **只**在「请求的是一个已存在的目录」时补 ``index.html``，
    **不做** SPA 回退 —— ``/chat``、``/schedule`` 这些前端路由刷新时会拿到
    404（浏览器表现为白屏），而它们恰恰是用户最常刷新的地址。
    回退的判据与边界见 ``src/server/spa.py`` 的模块文档字符串。

    ⚠️ 这里也是路由表的**最后**一步：``/`` 上的挂载会匹配所有没被前面
    路由吃掉的路径，所以它必须在所有 API 路由注册完之后再挂。

    Args:
        app (`FastAPI`): 目标应用。
    """
    static_dir = _static_dir()
    if not static_dir.is_dir():
        # ⚠️ 这里是 WARNING 而不是 INFO，理由是**它描述的是一个残缺状态**：
        # 「没挂上前端」的症状是首页 404，而 404 本身不会在日志里留下任何
        # 与「前端没构建」相关的线索；用 INFO 很容易被过滤规则筛掉，
        # 排查的人会从「前端构建」一路怀疑到「路由被吃掉」。
        #
        # ⚠️ 这条注释曾经写着「日志还没配好，装配期的 INFO 一条都看不到」——
        # 那个**曾经的事实**已经修掉了（见模块级 ``configure_logging()``
        # 上面那段说明）：现在装配期的 INFO 会正常落盘，级别选 WARNING
        # 纯粹是因为这条信息的重要度，不是因为「INFO 会丢」。
        logger.warning(
            "未发现前端静态资源目录 %s：跳过挂载（后端 API 不受影响，首页会 404）。",
            static_dir,
        )
        return

    app.mount(
        "/",
        SpaStaticFiles(directory=str(static_dir), html=True),
        name="static",
    )
    logger.info("已挂载前端静态资源：/ ← %s", static_dir)


# ==============================================================================
# 模块级应用（uvicorn 的加载目标）
# ==============================================================================
# ⚠️ 这里在 **import 期**就完成装配。这么做的理由：
#    uvicorn 以 `--workers N`（多进程）启动时会 fork worker，每个 worker 都会
#    import 本模块。装配放在模块级，能保证**每个 worker 各自持有自己的
#    storage / message_bus / 熔断器实例** —— 若改成在 lifespan 里懒装配，
#    会有多个 worker 共享同一个对象的风险（asyncio 原语不可跨进程共享），
#    那会是极难复现的并发故障。
#
# ⚠️ 代价：import 失败 = 启动失败。这是**刻意的**：配置写错（比如多了一个
#    ALIGO__ 未知键）就应该在启动时立刻失败并给出清晰报错，而不是让服务
#    「看起来起来了」然后在第一次请求时报错。
#
# ⚠️⚠️ 日志必须**先于**装配配置好，否则装配期打的 INFO 一条都看不到。
#    这曾经是一个真实的自相矛盾：装配在 import 期完成，而
#    ``configure_logging`` 只在 lifespan 里调用 —— 于是
#    「智能体装配完成：… 快车道 开、动态 Prompt 开、重排 关」这类
#    **装配期唯一的开关清单**（``agents_factory.py``）永远进不了日志：
#    没有 handler 时 Python 只把 WARNING 及以上交给 lastResort 输出到 stderr，
#    INFO 被直接丢掉。后果不是「少几行日志」，而是「有一个开关没生效」
#    这件事在容器里**没有任何可查的事实**，只能靠读代码去猜。
#
#    ⚠️ 顺带说明为什么这一段不会重复配置：``configure_logging`` 自带幂等
#    标记，lifespan 里那次调用会直接返回（那里仍然是**权威**的一次 ——
#    它用的是 lifespan 拿到的 ``settings``）。反过来，若将来有人把这里的
#    调用去掉，lifespan 那次**补不回来**：records 在 import 期就已经被丢弃了。
configure_logging()
app = create_root_app()


__all__ = [
    "BLOB_DIR_NAME",
    "BOOT_COMPLETED_ATTR",
    "STATIC_DIR_NAME",
    "WORKSPACE_DIR",
    "app",
    "build_blob_store",
    "build_message_bus",
    "build_storage",
    "build_storage_backend",
    "build_workspace_manager",
    "create_root_app",
]
