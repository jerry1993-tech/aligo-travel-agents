# -*- coding: utf-8 -*-
"""pytest 全局夹具。

==============================================================================
本文件存在的唯一理由：**让 `make test` 不依赖 Docker**
==============================================================================
    P1 的验收里有明确一条：「``make test`` 绿（不依赖 Docker，用 sqlite 内存库）」。
    这不是「顺便也能跑」，而是一条硬性约束 —— 它保证任何人在任何一台干净的
    机器上 clone 下来就能验证代码，而不必先拉 10 GB 镜像。

    要做到这一点，必须绕开三件在装配期就会发生的真实 I/O：

    ==================  ==========================================  ==========================
    组件                默认实现                                    测试替身
    ==================  ==========================================  ==========================
    存储                PostgreSQL（``postgres:5432``）              ``sqlite+aiosqlite:///:memory:``
    消息总线            ``RedisMessageBus``（``redis:6379``）       ``InMemoryMessageBus``
    工作区              ``LocalWorkspaceManager('/app/workspace')``  ``tmp_path``（pytest 临时目录）
    ==================  ==========================================  ==========================

    ⚠️ 为什么消息总线必须换掉（而不是「连不上就算了」）
        框架的 ``SchedulerManager`` 会订阅总线上的调度事件，**订阅失败时它会
        无限重试**（指数退避，日志表现为反复的
        ``schedule lifecycle subscription lost``）。因此没有 Redis 时，
        lifespan 的进入段**永远不会返回**，测试会一直挂到超时 ——
        表现为「pytest 卡住不动」，而不是「一个清晰的连接错误」。
        这是本项目踩过的真实坑，见 :func:`app` 里的 ``enable_scheduler=False``。

    ⚠️ 为什么不用配置开关（``ALIGO__REDIS__USE_IN_MEMORY`` 之类）
        那种开关一旦被误设到生产，症状是「SSE 事件单进程内正常、多副本之间
        收不到」—— 只在扩容时才暴露，且与「配置写错了」看起来毫无关系。
        改用**参数注入**（见 ``src/server/app.py::create_root_app``）则
        不可能被配置误触。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.config import Settings, load_settings, repo_root

# ==============================================================================
# 常量
# ==============================================================================

#: 测试用的内存库连接串。
#:
#: ``:memory:`` + asyncio 有一个隐蔽的坑：``sqlite+aiosqlite`` 默认用 NullPool，
#: 而 NullPool 会为**每个连接**新建一个独立的 ``:memory:`` 数据库 ——
#: 于是「建表」和「查表」落在两个不同的库上，症状是「表明明建了却报
#: ``no such table``」。``src/server/app.py::_storage_engine_kwargs`` 里
#: 检测到 ``:memory:`` 会自动改用 StaticPool（所有连接复用同一个底层连接），
#: 内存库才真的是一份。**不要**在这里自己传 poolclass 绕开那段逻辑 ——
#: 那样就测不到它了。
TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

#: 测试档的环境变量视图。
#:
#: ⚠️ 这里**显式列出每一个变量**，而不是增量覆盖 ``os.environ``。
#: 传给 ``load_settings(environ=...)`` 的映射是**完全替代**：
#: 既不读真实进程环境、也不读仓库根的 ``.env``。
#: 这正是让配置类用例可重复的关键 —— 开发者本机 ``.env`` 里有没有 key、
#: CI 机上有没有 export，都不会改变测试结果。
#:
#: 尤其注意 ``ALIGO__LLM__API_KEY`` 与 ``ALIGO__LLM__BASE_URL`` 被钉成空串：
#: 哪怕跑测试的机器上恰好配了真 key，也必须走 MockLLM。
#: 否则用例会变成「联网、花钱、且断言随模型输出浮动」的三重不稳定。
TEST_ENVIRON: dict[str, str] = {
    "ALIGO__APP__ENV": "test",
    "ALIGO__APP__LOG_LEVEL": "WARNING",
    # 内存库（见 TEST_DB_URL 的注释）。
    "ALIGO__DB__URL": TEST_DB_URL,
    # Redis 的 URL 保留默认值即可 —— 总线已被替换成 InMemoryMessageBus，
    # 这个 URL 在测试里**没有任何代码会去连它**。留着它是为了证明
    # 「总线替身」与「URL 配置」确实是两条独立的路，而不是碰巧连上了。
    "ALIGO__REDIS__URL": "redis://:test-password@localhost:6379/0",
    # 零密钥：强制降级到 MockLLM。
    "ALIGO__LLM__API_KEY": "",
    "ALIGO__LLM__BASE_URL": "",
    # trace 导出器必须关掉：导出是**异步批处理**的，进程退出时可能在后台
    # 线程里留下未发送的批次，既拖慢测试又刷一堆无意义的网络错误日志。
    "ALIGO__OBSERVABILITY__TRACE_EXPORTER": "none",
    "ALIGO__OBSERVABILITY__OTLP_ENDPOINT": "",
}


# ==============================================================================
# 夹具：配置
# ==============================================================================
@pytest.fixture(scope="session")
def repo_dir() -> Path:
    """返回仓库根目录。

    用例里凡是需要引用 ``config/`` ``scripts/`` 之类的真实文件，
    都应当从这里出发拼路径，**不要**写 ``Path.cwd()`` 或相对路径 ——
    pytest 的 cwd 取决于是从哪个目录调用的，不可靠。

    Returns:
        `Path`: 仓库根目录。
    """
    return repo_root()


@pytest.fixture
def settings() -> Settings:
    """构造一份**确定性的**测试配置。

    每个用例拿到的是全新对象（``scope="function"``），
    因此用例之间不可能通过修改配置互相影响。

    Returns:
        `Settings`: 指向内存 sqlite、MockLLM、无 trace 导出的配置。
    """
    return load_settings("test", environ=TEST_ENVIRON, dotenv=False)


@pytest.fixture
def clean_settings_cache() -> Iterator[None]:
    """在用例前后清空 ``get_settings()`` 的进程内单例缓存。

    ⚠️ 为什么需要它：``get_settings()`` 会把结果缓存到模块级变量，
    而该缓存**跨用例存活**。任何调用过 ``get_settings()`` 的用例
    （比如 import ``src.server.app`` 触发的装配）都会留下一个
    指向「默认 dev 档」的缓存，污染后续用例。

    需要这份干净的用例请显式声明本夹具；没声明的用例不受影响 ——
    这是刻意的，避免让所有用例都背上一次重复加载 YAML 的开销。

    Yields:
        `None`: 用例执行期间缓存为空。
    """
    from src.config import set_settings

    set_settings(None)
    yield
    set_settings(None)


# ==============================================================================
# 夹具：应用与 HTTP 客户端
# ==============================================================================
@pytest_asyncio.fixture
async def app(settings: Settings, tmp_path: Path) -> AsyncIterator[FastAPI]:
    """装配一个**可以真正启动**的测试应用（不依赖任何外部服务）。

    这里刻意不去 import ``src.server.app``（那会触发模块级的
    ``create_root_app()``，用真实 dev 配置装配一个连 PostgreSQL 的应用），
    而是直接调用 :func:`~src.server.app.create_root_app` 并注入替身。

    与生产装配的差异（也是本夹具存在的全部内容）::

        storage          → sqlite 内存库
        message_bus      → InMemoryMessageBus（见模块 docstring 的 ⚠️）
        workspace_manager→ tmp_path（生产是 /app/workspace，本机不可写）
        blob_store       → tmp_path（同上，见下）
        enable_scheduler → False

    ⚠️ ``blob_store`` 也必须换掉，而且这一条是**踩过坑**的：
    生产默认根是 ``/app/workspace/blobs``（绝对路径，见
    :func:`~src.server.app.build_blob_store`），本机没有 ``/app`` 这一层，
    lifespan 一进去就会 ``makedirs`` 失败、整个夹具炸在启动阶段。
    换成 ``tmp_path`` 之后，用例既写得进去、也不会在仓库里留下垃圾
    （框架自己兜的默认值是相对 CWD 的 ``./blobs``，曾经真的在仓库根
    建出过一个目录，而那个默认值在生产里会让容器**崩溃循环**）。

    ⚠️ ``enable_scheduler=False`` 不只是「省点资源」：调度器会订阅消息总线，
    而本项目**不去测调度**（P1 没有定时任务）。关掉它可以让用例失败时
    的栈更短、更容易定位；同时避免 APScheduler 在事件循环里留下未清理的
    定时器而触发「Event loop is closed」告警。

    Args:
        settings (`Settings`): 测试配置。
        tmp_path (`Path`): pytest 提供的临时目录，用作工作区根。

    Yields:
        `FastAPI`: 已装配、但**尚未进入 lifespan** 的应用对象。
    """
    from agentscope.app.message_bus import InMemoryMessageBus
    from agentscope.app.rag.blob_store import LocalBlobStore
    from agentscope.app.storage._sql import AsyncSQLAlchemyStorage
    from agentscope.app.workspace_manager import LocalWorkspaceManager

    # 复用生产代码的 engine 参数推导（而不是在这里写死 StaticPool）：
    # 这样 TEST_DB_URL 注释里说的那条「:memory: 自动改用 StaticPool」
    # 才真的被测到 —— 若哪天那段逻辑被改坏，这里会跟着一起失败。
    from src.server.app import _storage_engine_kwargs, create_root_app

    storage = AsyncSQLAlchemyStorage(
        url=settings.db.url,
        create_tables=settings.db.create_tables,
        auto_migrate=False,
        engine_kwargs=_storage_engine_kwargs(settings),
    )

    yield create_root_app(
        settings,
        storage=storage,
        message_bus=InMemoryMessageBus(),
        workspace_manager=LocalWorkspaceManager(basedir=str(tmp_path / "workspace")),
        blob_store=LocalBlobStore(root_dir=str(tmp_path / "blobs")),
        enable_scheduler=False,
    )


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[AsyncClient]:
    """返回一个**已进入 lifespan** 的 HTTP 客户端。

    ⚠️ 这里手动 ``async with app.router.lifespan_context(app)``，
    而不是用 ``httpx.ASGITransport`` 直接发请求 —— 因为
    **ASGITransport 不会触发 lifespan**（它只把请求转发给 ASGI 应用，
    不扮演服务器的生命周期职责）。

    这个区别在本项目里是**致命的**，不是理论问题：框架把
    ``chat_service`` / ``session_service`` 等全部资源写在 lifespan 里，
    跳过 lifespan 的应用能启动、``/healthz`` 也返回 200，
    但任何业务路由都会因为 ``app.state.xxx`` 不存在而 500。
    换句话说，一个不跑 lifespan 的测试客户端会让「服务是否真的能起来」
    这件事完全测不到。

    手动进入 lifespan 还有个附带好处：它会顺带验证
    ``src/server/app.py::_wrap_lifespan`` 是否正确置位了 ``boot_completed``。

    Args:
        app (`FastAPI`): 已装配的应用。

    Yields:
        `AsyncClient`: 绑定到该应用的 HTTP 客户端（base_url 为 ``http://test``）。
    """
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


# ==============================================================================
# 夹具：环境变量隔离
# ==============================================================================
@pytest.fixture
def isolated_environ() -> Iterator[dict[str, str]]:
    """提供一个与真实进程环境**互相隔离**的字典。

    用于测「环境变量覆盖配置」这类行为：直接改 ``os.environ`` 会污染同进程
    的其它用例（pytest 不隔离环境变量），而把字典传给
    ``load_settings(environ=...)`` 则完全没有副作用。

    Yields:
        `dict[str, str]`: 空的、可随意读写的环境变量视图。
    """
    yield {}


__all__: list[str] = []
