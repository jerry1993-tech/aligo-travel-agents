# -*- coding: utf-8 -*-
"""服务装配（``src/server/app.py``）的契约测试。

==============================================================================
这些用例在防什么
==============================================================================
    ``src/server/app.py`` 里藏着三个**不会在本地开发时暴露**的陷阱，
    它们的共同特征是：代码看起来完全正确，服务也确实「起来了」，
    但真正的业务请求会 500，或者某条路由永远收不到流量。

      1. **lifespan 被 mount 吞掉** —— 若有人按框架 docstring 写成
         ``root.mount("/agentscope", app)``，应用能启动、``/healthz`` 返回 200、
         ``/docs`` 能打开，但所有业务端点都会因为 ``app.state.chat_service``
         不存在而 500。本地手工点两下**测不出来**。
      2. **中间件顺序反了** —— ``add_middleware`` 是 ``insert(0)``，
         列表越靠后越外层。顺序反了的症状是「访问日志里没有 trace_id」，
         一个不影响功能、只影响排障的缺陷 —— 因此最容易长期潜伏。
      3. **静态资源挂早了** —— 先挂 ``StaticFiles("/")`` 会把后续所有 API
         路由遮蔽掉，症状是「接口全 404，但首页能打开」。

    这三条都不可能靠「跑起来看一眼」发现，只能靠断言。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI

from src.server.app import (
    BOOT_COMPLETED_ATTR,
    STATIC_DIR_NAME,
    create_root_app,
)

# ==============================================================================
# 一、装配期：不连接任何外部服务
# ==============================================================================
def test_assembly_does_not_require_any_secret(clean_settings_cache: None) -> None:
    """**零密钥**下也要能完成装配。

    这是「clone 下来直接 ``make up``」这条验收的机器化版本。
    装配路径上的三个组件都必须是**惰性**的：
    ``AsyncSQLAlchemyStorage`` / ``RedisMessageBus`` 的构造函数不建连接
    （连接发生在 lifespan 的 ``__aenter__``），trace 在
    ``trace_exporter=none`` 时不注册导出器。

    ⚠️ 这条性质很容易被无意破坏 —— 比如有人在模块级加一句
    ``await storage.ping()`` 做「启动自检」。那会让本用例变红，
    而它变红的方式恰好就是「用户的机器上没有任何密钥」这个场景。

    Args:
        clean_settings_cache (`None`): 清空配置单例，避免读到别的用例留下的缓存。
    """
    app = create_root_app()

    assert isinstance(app, FastAPI)


def test_blob_store_root_is_absolute_and_inside_the_workspace(
    clean_settings_cache: None,
) -> None:
    """★★★ blob 存储的根必须是**绝对**路径，且落在工作区里 —— 否则容器崩溃循环。

    这是一次**真实事故**的回归用例。框架的 ``create_app`` 在配了知识库管理器
    的那条分支上，会给 ``app.state.blob_store`` 兜一个相对 CWD 的默认值
    （``LocalBlobStore(root_dir="./blobs")``）。容器里 CWD 是 ``/app``
    （root:root 755），进程却是 uid 1000 ⇒ lifespan 进入段
    ``makedirs("/app/blobs")`` 抛 PermissionError ⇒
    ``Application startup failed`` ⇒ 容器**崩溃循环**。

    而 ``make test`` 当时是**全绿**的：pytest 的 CWD 是仓库根目录、可写，
    于是同一个默认值只是静默创建了一个空目录 —— 一个缺陷在两种 CWD 下
    表现为「多一个目录」和「服务起不来」两种完全不同的现象。

    ⚠️ 所以这里断言的是**路径的形状**（绝对 + 在工作区内），而不是
    「能不能写」。断言可写性会让这条用例在开发机上通过（仓库根可写）、
    在容器里失败 —— 那就成了又一条「只有上生产才知道」的用例。
    形状断言不依赖当前 CWD 是否可写，因此两种环境下都成立，
    且只要有人删掉 ``create_root_app`` 里那个显式的 ``blob_store=``，
    它立刻变红。

    Args:
        clean_settings_cache (`None`): 清空配置单例，确保走的是默认装配路径。
    """
    from pathlib import Path

    from src.server.app import BLOB_DIR_NAME, WORKSPACE_DIR

    app = create_root_app()
    root = Path(app.state.blob_store._root)  # noqa: SLF001 —— 框架未提供公开访问器

    assert root.is_absolute(), (
        f"blob 存储根是相对路径 {root}：它会跟着进程 CWD 走，"
        f"在容器里（CWD=/app，不可写）会直接让启动失败。"
    )
    assert root == Path(WORKSPACE_DIR) / BLOB_DIR_NAME, (
        f"blob 存储根是 {root}，期望 {Path(WORKSPACE_DIR) / BLOB_DIR_NAME}。"
        f" 它必须落在工作区（compose 里唯一保证可写的命名卷）下面。"
    )
    assert Path.cwd() not in root.parents, (
        f"blob 存储根 {root} 落在了当前工作目录下面 —— 这正是框架默认值"
        f"（./blobs）的形状，生产和测试会各拿到一份不同的东西。"
    )


def test_boot_flag_starts_false(app: FastAPI) -> None:
    """应用对象一造出来，``boot_completed`` 就必须存在且为 ``False``。

    ⚠️ 断言「存在」与断言「为 False」是两件事：
    若只是 ``getattr(app.state, BOOT_COMPLETED_ATTR, False)`` 这样的读法，
    属性**不存在**与属性**为 False** 会得到同一个结果，
    于是「忘了设置」这个缺陷永远测不出来。这里要求属性真实存在。
    """
    assert hasattr(app.state, BOOT_COMPLETED_ATTR)
    assert getattr(app.state, BOOT_COMPLETED_ATTR) is False


async def test_boot_flag_flips_inside_lifespan(app: FastAPI) -> None:
    """进入 lifespan 后 ``boot_completed`` 变 ``True``，退出后变回 ``False``。

    ⚠️ 这是本文件最重要的一条。它验证的是**那个窗口期**：
    FastAPI 在 lifespan 的进入段执行完之前就已经开始接受连接，
    此时进程在响应 HTTP，但 storage / message_bus / scheduler 都还没就绪。
    若 ``/readyz`` 只查 PG/Redis/Milvus（它们都通），就会误报「就绪」，
    而第一个真实请求会因为 ``app.state.chat_service`` 尚不存在而 500。

    同时也验证了框架的资源**确实**被启动了 —— 这正是「不许 mount」那条
    决策要保证的东西（见 ``src/server/app.py`` 的核心决策一）。
    """
    assert getattr(app.state, BOOT_COMPLETED_ATTR) is False

    async with app.router.lifespan_context(app):
        assert getattr(app.state, BOOT_COMPLETED_ATTR) is True
        # 框架在 lifespan 里写入的资源必须真的存在 —— 这是「不 mount」的收益。
        assert hasattr(app.state, "chat_service")
        assert hasattr(app.state, "session_service")

    assert getattr(app.state, BOOT_COMPLETED_ATTR) is False


# ==============================================================================
# 二、路由注册顺序
# ==============================================================================
def test_static_mount_is_registered_last(app: FastAPI) -> None:
    """挂在 ``/`` 的静态资源必须是**最后一个**注册的路由。

    ``StaticFiles`` 挂到 ``/`` 会匹配所有未命中的路径。若它排在 API 路由
    之前注册，Starlette 会先命中它，于是**所有接口 404，而首页正常打开** ——
    一个「服务看起来没坏」的故障。

    ⚠️ 断言方式：找 ``Mount`` 实例在 ``app.routes`` 里的下标，
    要求它等于最后一位。不去断言「没有别的 Mount」——
    框架自己也可能挂子应用，那种情况不构成风险。
    """
    from starlette.routing import Mount

    indices = [
        i
        for i, route in enumerate(app.routes)
        if isinstance(route, Mount) and getattr(route, "name", None) == STATIC_DIR_NAME
    ]
    assert indices, (
        f"没有找到名为 {STATIC_DIR_NAME!r} 的 Mount —— "
        f"静态资源目录不存在，或挂载时的 name 改了？"
    )

    static_index = indices[-1]
    assert static_index == len(app.routes) - 1, (
        f"静态资源挂在第 {static_index} 位，共 {len(app.routes)} 条路由。"
        f"它必须是最后一个，否则会遮蔽 API 路由。"
    )


def test_static_directory_ships_with_the_repo() -> None:
    """静态资源目录必须随仓库一起存在。

    用例的目标不是「目录在不在」，而是：**若哪天有人把它删了**，
    ``_mount_static_if_present`` 会静默跳过挂载（那是刻意的设计，
    好让 P1~P4 阶段前端还没构建时后端照样能起）——
    而静默跳过意味着「上一条用例悄悄失效了」。
    这条用例保证上一条用例永远是**有效**的。
    """
    from src.config import repo_root

    static_dir = repo_root() / "src" / "server" / STATIC_DIR_NAME
    assert static_dir.is_dir(), f"缺少静态资源目录：{static_dir}"
    assert (static_dir / "index.html").is_file(), f"缺少首页：{static_dir}/index.html"


async def test_the_built_frontend_is_actually_served(client: Any) -> None:
    """★★★ 前端真的能被浏览器打开：``GET /`` 的 HTML 与它引用的资源都可达。

    ⚠️ 只断言 ``index.html`` 存在是**不够**的 —— 一个手工拷进去的占位页
    同样满足它。这条用例沿着浏览器真正会走的那条路再走一遍：
    取 ``/`` → 从 HTML 里解析出脚本与样式的 URL → 逐个取回来。

    ⚠️ 为什么必须逐条取回：vite 的产物是**带内容哈希**的
    （``assets/index-a1b2c3d4.js``），于是「HTML 里写的文件名」与
    「磁盘上的文件名」是两份数据。它们可以不同步 —— 只重新构建了一半、
    或把产物拷到了别的目录、或构建产物的解析路径不对。任一种情况下
    浏览器打开都是一片白，而 ``/healthz`` 照样 200、``make test`` 照样绿。

    ⚠️ 这条用例在 P1~P4 阶段（前端还没构建）就会是红的 —— 这是**刻意**的：
    P5 的验收标准之一就是「前端能打开」，把它变成机器可判的断言，
    比在 README 里写一句「已验证」可靠得多。
    """
    import re

    response = await client.get("/")
    assert response.status_code == 200, f"GET / 返回 {response.status_code}"
    assert "text/html" in response.headers.get("content-type", ""), (
        "GET / 返回的不是 HTML —— 静态挂载被别的路由吃掉了？"
    )

    html = response.text
    # 同时抓 <script src> 与 <link href>：样式丢了页面能开，但样式丢了
    # 说明产物不完整，而「产物不完整」正是这条用例要抓的东西。
    assets = re.findall(r'(?:src|href)="(/[^"]+)"', html)
    assert assets, "首页里没有任何 / 开头的资源引用 —— 这不像一份 vite 产物"

    for asset in assets:
        asset_response = await client.get(asset)
        assert asset_response.status_code == 200, (
            f"首页引用了 {asset}，但取回来是 {asset_response.status_code}。"
            f" HTML 与磁盘上的产物不同步（重新构建一次前端试试）。"
        )


#: 浏览器导航请求的 ``Accept``（Chrome/Safari/Firefox 的地址栏与 F5 都是这个形状）。
BROWSER_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"


def _root_static_files() -> list[str]:
    """列出前端产物**根目录**下的文件名（形如 ``/agentscope.svg``）。

    Returns:
        `list[str]`: 排序后的路径列表。
    """
    from src.config import repo_root

    static_dir = repo_root() / "src" / "server" / STATIC_DIR_NAME
    return sorted("/" + p.name for p in static_dir.iterdir() if p.is_file())


@pytest.mark.parametrize("path", _root_static_files())
async def test_every_static_root_file_is_public(client: Any, path: str) -> None:
    """★ 产物根目录下的每个文件都必须**免鉴权**可达。

    这是线上最容易漏、又最难排查的一类缺陷：``index.html`` 里
    ``<img src="/agentscope.svg">`` 这类**子资源由浏览器自己发起**，
    它**不会**携带页面上任何自定义请求头（``X-User-ID`` 是 fetch 才能带的），
    于是鉴权中间件看到的是一个不带凭据的请求 → 401 → 页面上一个碎图标。
    服务端日志里只有一条 401，看不出跟前端有任何关系。

    ⚠️ 参数化列表在**收集期**从磁盘算出来，因此新增一个 ``public/`` 文件
    会自动被覆盖到 —— 这正是这条用例存在的意义：白名单必须跟着产物走。

    Args:
        client (`Any`): 测试客户端。
        path (`str`): 待验证的静态文件路径。
    """
    response = await client.get(path)

    assert response.status_code == 200, (
        f"{path} 是产物根目录下的文件，但匿名取回来是 {response.status_code}。"
        f" 它必须出现在鉴权白名单里（见 src/server/spa.py::static_root_public_paths）。"
    )


@pytest.mark.parametrize("path", ["/chat", "/chat/agent-1", "/schedule", "/setup"])
async def test_spa_routes_survive_a_browser_refresh(client: Any, path: str) -> None:
    """★ 前端路由刷新时必须返回外壳（``index.html``），而不是 401/404。

    前端是 ``createBrowserRouter``（history 路由）：用户在 ``/chat`` 页面
    按 F5，浏览器发的是 ``GET /chat``。这一步**不带凭据**（导航请求带不了
    自定义头），而 ``/chat`` 也不是后端端点 —— 若没有 SPA 回退，
    用户看到的是白屏或一段 JSON 404，而应用本身完全正常。

    Args:
        client (`Any`): 测试客户端。
        path (`str`): 前端路由路径。
    """
    response = await client.get(path, headers={"Accept": BROWSER_ACCEPT})

    assert response.status_code == 200, (
        f"浏览器导航 {path} 返回 {response.status_code}；"
        f" 前端路由刷新会白屏（SPA 回退没生效？）"
    )
    assert "text/html" in response.headers.get("content-type", "")
    assert "<!doctype html" in response.text.lower(), f"{path} 返回的不是首页 HTML"


async def test_a_browser_accept_header_cannot_forge_an_identity(
    tmp_path: Path,
) -> None:
    """★★★ 在 **JWT 模式**下，``Accept: text/html`` + 伪造 ``X-User-ID`` 也必须被拒。

    这一条是本文件里最贵的一条用例，因为它守的是一个**静默的越权**：
    鉴权中间件的 SPA 放行分支在 JWT 校验**之前**短路（见
    ``src/server/middleware/auth.py`` 的 ``__call__``）。也就是说，
    只要一个请求被判成「浏览器导航」，它就连 JWT 都不用出示 ——
    而判成导航的信号只有两个：``Accept: text/html``（谁都能填）与
    「路径不在 API 路由表里」（由 ``build_api_path_matcher`` 判定）。

    所以这条链上的**任何一个**环节写错，结果都是
    ``curl -H 'Accept: text/html' -H 'X-User-ID: victim' /agent/schema``
    → 200，而这条 curl 不需要任何签名、不需要猜 token。实测确实如此过。

    ⚠️ 为什么必须开 JWT 才测得到：在默认的 ``X-User-ID`` 直连模式里，
    这个头**本身就是凭据**（``require_user_header`` 默认 True），
    伪造值被接受是**设计如此**，断言不出问题。JWT 模式才把
    「导航放行」与「身份校验」摆成互斥的两条路 —— 放行了就必然没有校验。

    ⚠️ 同时也别用 ``/mcp`` 之外的路径来「顺手」验证白名单剥离：
    本用例只断言**拒绝**（401）。白名单路径上剥离 ``X-User-ID``
    那件事属于中间件的单元行为，由
    ``tests/test_auth_middleware.py::test_browser_navigation_is_public_and_strips_the_identity_header``
    用记录式假应用精确断言 —— 那里能直接看到转发出去的 scope，
    比在装配层靠状态码反推可靠得多。

    Args:
        tmp_path (`Path`): pytest 临时目录，用作工作区与 blob 根。
    """
    from agentscope.app.message_bus import InMemoryMessageBus
    from agentscope.app.rag.blob_store import LocalBlobStore
    from agentscope.app.storage._sql import AsyncSQLAlchemyStorage
    from agentscope.app.workspace_manager import LocalWorkspaceManager
    from httpx import ASGITransport, AsyncClient

    from src.config import load_settings
    from src.server.app import _storage_engine_kwargs
    from tests.conftest import TEST_ENVIRON

    # 在测试档上**只**打开 JWT 这一项；其余全部沿用 TEST_ENVIRON
    #（内存 sqlite / MockLLM / 无 trace 导出），所以这条用例仍然不依赖 Docker。
    settings = load_settings(
        "test",
        environ={
            **TEST_ENVIRON,
            "ALIGO__AUTH__JWT_ENABLED": "true",
            "ALIGO__AUTH__JWT_SECRET": "assembly-test-secret-not-a-real-one",
        },
        dotenv=False,
    )
    storage = AsyncSQLAlchemyStorage(
        url=settings.db.url,
        create_tables=settings.db.create_tables,
        auto_migrate=False,
        engine_kwargs=_storage_engine_kwargs(settings),
    )
    app = create_root_app(
        settings,
        storage=storage,
        message_bus=InMemoryMessageBus(),
        workspace_manager=LocalWorkspaceManager(basedir=str(tmp_path / "workspace")),
        blob_store=LocalBlobStore(root_dir=str(tmp_path / "blobs")),
        enable_scheduler=False,
    )

    # 这几个端点自身不依赖身份（路由里没有 Depends(get_current_user_id)），
    # 所以「剥掉 X-User-ID」这条防线对它们无效 —— 唯一的防线就是别放行。
    api_paths = ["/agent/schema", "/agent/schema/v2", "/mcp", "/hub/mcp", "/credential/schemas"]

    async with app.router.lifespan_context(app):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            for path in api_paths:
                response = await client.get(
                    path,
                    headers={"Accept": BROWSER_ACCEPT, "X-User-ID": "victim"},
                )
                assert response.status_code == 401, (
                    f"JWT 模式下，{path} 带着伪造的 X-User-ID 与 Accept: text/html "
                    f"返回了 {response.status_code}（应为 401）。"
                    f" 这条若变成 200，说明 SPA 放行分支又抢在 JWT 校验前面短路了 —— "
                    f"任何人都能用一条 curl 匿名读走这个端点。"
                )


async def test_non_navigation_requests_are_still_denied_by_default(client: Any) -> None:
    """接口客户端（``Accept`` 不是 text/html）走的是**另一个**分支：仍然 401。

    回退规则必须窄到只放行浏览器导航。这里用同一个路径 ``/chat`` 对照：
    带 ``Accept: text/html`` 是 200（上一条用例），不带就是 401 ——
    证明放行的确是「导航」这个信号，而不是「路径不在 API 里」。

    Args:
        client (`Any`): 测试客户端。
    """
    response = await client.get("/chat", headers={"Accept": "*/*"})

    assert response.status_code == 401, (
        f"非导航请求返回 {response.status_code}，应为 401。"
        f" 若放行条件写成「路径不是 API」就会变成 404/200 —— 那等于把 "
        f"鉴权白名单扩大到所有未知路径。"
    )


#: 框架里**自身不依赖身份**的 GET 端点 —— 它们的路由声明里没有
#: ``Depends(get_current_user_id)``。这正是下面两条用例存在的理由：
#: 鉴权的 SPA 放行分支只做「剥掉客户端的 ``X-User-ID``」，
#: 而对这些端点剥头**毫无作用**，端点照样执行。若它们还能被放行，
#: 就等于匿名可读。
IDENTITY_FREE_ENDPOINTS: list[str] = [
    "/agent/schema",
    "/agent/schema/v2",
    "/hub/mcp",
    "/hub/skill",
    "/credential/schemas",
    "/sop/schema",
    "/channels/types",
]

#: 真实装配出来的前端路由（路由表里**没有**对应 API 的路径）。
SPA_ROUTES: list[str] = ["/chat", "/chat/agent-1", "/schedule", "/setup", "/"]


def test_the_api_path_matcher_sees_the_framework_routes(app: FastAPI) -> None:
    """★ ``build_api_path_matcher`` 必须在**真实路由表**上认出框架的 API 路由。

    这一条是一次**真实的越权**的回归用例（不是理论问题）：

        ``AuthMiddleware`` 曾把 ``is_spa_navigation`` 当成无条件放行规则。
        实测 ``curl -H 'Accept: text/html' /agent/schema`` → **200**，
        换成 ``Accept: */*`` → 401。``Accept`` 谁都能填，所以那个口子
        等于没有鉴权（JWT 模式同样中招 —— SPA 分支在 JWT 校验之前短路）。

        修法是把放行收窄成「**没有任何 API 路由匹配这个路径**」才算导航，
        判据由本函数给出。

    ⚠️ 而本函数**第一版就是错的**，错法值得记下来：它按
    ``isinstance(route, starlette.routing.Route)`` 过滤路由表，于是
    ``api_routes`` 里只剩下 ``/openapi.json``、``/docs``、``/redoc``
    三条 —— 框架的 API 路由**不是** ``Route``，而是
    ``fastapi.routing._IncludedRouter``（也是 ``BaseRoute`` 子类）。结果
    ``matcher("/agent/schema")`` 返回 **False**，越权原样保留。

    这正是本用例的价值：**只对着手写的假路由表测，永远发现不了它**。
    所以它拿的是 ``create_root_app()`` 装配出来的真表。

    Args:
        app (`FastAPI`): 真实装配的应用对象。
    """
    from src.server.spa import build_api_path_matcher

    matcher = build_api_path_matcher(app.routes)

    for path in IDENTITY_FREE_ENDPOINTS + ["/api/v1/me", "/healthz", "/readyz", "/metrics"]:
        assert matcher(path), (
            f"{path} 是一条真实的 API 路由，但判据说「它不是」，"
            f"于是导航放行会把它放给匿名请求 —— 越权。"
            f" 若新增了路由类型，请检查 build_api_path_matcher 的过滤条件"
            f"（框架用的是 fastapi.routing._IncludedRouter，不是 Route）。"
        )

    for path in SPA_ROUTES:
        assert not matcher(path), (
            f"{path} 是前端路由，但判据说「它是 API」——"
            f" 浏览器的 F5 会被 401 挡掉，用户看到白屏。"
        )


@pytest.mark.parametrize("path", IDENTITY_FREE_ENDPOINTS)
async def test_a_browser_accept_header_cannot_reach_an_api_path(
    client: Any, path: str
) -> None:
    """★★★ 一个 ``Accept: text/html`` 头**不能**把 API 路径变成匿名可读的。

    这是上一条用例的端到端版本 —— 判据写对了，还得确认中间件真的在用它。
    这些端点自身不依赖身份，所以「剥掉 X-User-ID」对它们完全无效；
    唯一的防线就是**别放行它们**。

    ⚠️ 断言具体的 ``401``（而不是 ``!= 200``）：401 精确地说明请求在
    **鉴权中间件**就被挡下了，根本没有进入路由分发。若返回 200，
    就是越权；若返回 404/422，则说明请求走到了框架里 —— 同样是漏。

    Args:
        client (`Any`): 真实装配的测试客户端。
        path (`str`): 待验证的框架端点。
    """
    response = await client.get(path, headers={"Accept": BROWSER_ACCEPT})

    assert response.status_code == 401, (
        f"{path} 带 Accept: text/html 匿名访问返回 {response.status_code}（应为 401）。"
        f" 这个端点的路由声明里没有身份依赖，剥 X-User-ID 拦不住它 —— "
        f"一旦被导航规则放行就是真越权（见 src/server/middleware/auth.py::_is_spa_route）。"
    )


def test_ops_routes_are_registered(app: FastAPI) -> None:
    """``/healthz`` ``/readyz`` ``/metrics`` 三条运维路由必须存在。

    它们由我们的 ``probes_router`` 提供，与框架自带的 ``/health`` **不冲突**
    （后者由框架维护，我们不干预）。用例把这三个路径钉死 ——
    它们是 compose 健康检查与 ``scripts/smoke.py`` 的契约，
    改名会同时打断部署与冒烟测试，且失败发生在容器编排层而非应用层，
    定位成本很高。

    ⚠️ 断言方式不能是遍历 ``app.routes`` 找 ``APIRoute.path``：
    FastAPI 0.141 起 ``include_router`` 会把子路由包成一层
    ``_IncludedRouter`` 对象塞进 ``app.routes``，**不会**摊平成
    ``APIRoute`` 列表。直接遍历 ``app.routes`` 只会看到 ``/docs``
    ``/openapi.json`` 这几条框架自建的路径，然后误判「路由没注册上」——
    而它们其实好好的（HTTP 请求打得通）。

    因此这里读 ``app.openapi()['paths']``：那是 FastAPI 自己摊平后的
    路由表，也是 ``/docs`` 页面渲染的依据 —— 用它断言等于断言
    「用户能在文档里看到这条路由」。
    """
    paths = set(app.openapi()["paths"])

    for expected in ("/healthz", "/readyz", "/metrics"):
        assert expected in paths, f"缺少运维路由 {expected}；现有路径：{sorted(paths)}"


# ==============================================================================
# 三、中间件顺序
# ==============================================================================
def test_trace_middleware_is_outermost(app: FastAPI) -> None:
    """``TraceContextMiddleware`` 必须比 ``HttpMetricsMiddleware`` **更外层**。

    ⚠️ 为什么顺序有语义（而不是随便排排）：``add_middleware`` 内部是
    ``user_middleware.insert(0, ...)``，因此**列表里越靠后 = 越外层**。
    我们要的是::

        TraceContext → HttpMetrics → … → 应用

    即 TraceContext 在最外层。这样 trace_id 才在**进入应用之前**绑好，
    访问日志里才能带上它。若顺序反了，访问日志会打出一串 ``[-]``，
    而功能完全正常 —— 一个只在排障时才被发现、且发现者会以为
    「trace 功能没实现」的缺陷。

    ⚠️ 本行原有的一版示意图写的是 ``HttpMetrics → TraceContext → 应用``，
        与紧跟其后的「即 TraceContext 在外层」正好相反 —— 箭头方向写反了。
        照那张图去改代码，会把 trace_id 绑到应用内部，正好造成上面描述的
        那个缺陷。图已按断言的实际方向修正。

    ⚠️ 断言方式：``app.user_middleware`` 里下标 **0 是最外层**
    （因为每次 insert(0)）。所以要求 TraceContext 的下标
    **小于** HttpMetrics 的下标。

    ⚠️ 这里只断言这两个的相对顺序，**不**断言完整链条。
        完整链条（含 P2 新增的 Auth / RateLimit）由
        ``tests/test_api_v1.py::test_middleware_stack_order`` 逐字断言。
        刻意不在这里再写一遍：那样每加一个中间件都要改两处，
        而漏改的那一处不会报错，只会让这里变成一个「看起来还在守着什么」
        的过期断言。
    """
    names = [m.cls.__name__ for m in app.user_middleware]

    assert "TraceContextMiddleware" in names, f"未挂载 TraceContextMiddleware：{names}"
    assert "HttpMetricsMiddleware" in names, f"未挂载 HttpMetricsMiddleware：{names}"

    trace_index = names.index("TraceContextMiddleware")
    metrics_index = names.index("HttpMetricsMiddleware")

    assert trace_index < metrics_index, (
        f"中间件顺序反了：{names}。\n"
        f"下标 0 是最外层；TraceContext({trace_index}) 必须小于 "
        f"HttpMetrics({metrics_index})，否则访问日志里不会有 trace_id。"
    )


# ==============================================================================
# 四、装配参数（这些是配置里没有、但必须传对的东西）
# ==============================================================================
def test_the_knowledge_base_manager_is_the_single_collection_one(
    app: FastAPI,
    settings: Settings,
) -> None:
    """P4 起知识库管理器必须是**本项目的**单集合实现。

    ⚠️ 这条替换掉了 P1 时代的 ``..._is_not_wired_in_p1``（它断言的是
    ``is None``）。那次改动正是这条用例存在的意义 —— 一条变红的用例
    比一句 TODO 注释可靠得多，P1 那版就是专门为「提醒 P4 改这里」写的。

    断言的是**类型**而不是「非 None」，因为这里有两个都会让
    ``is not None`` 通过的错法：

      1. 接上框架自带的 ``CollectionPerKbManager`` —— 它给每个 KB 建一个
         集合，会让契约里逐字指定的 ``ALIGO__MILVUS__COLLECTION``
         变成死配置；
      2. 接上一个什么都没覆盖的基类子类 —— 它的
         ``delete_knowledge_base`` 会把**共享集合整个删掉**。

    两者都不会报错，只会在使用中造成数据错乱或丢失。
    """
    from src.knowledge.manager import SingleCollectionKbManager

    manager = getattr(app.state, "knowledge_base_manager", None)
    assert isinstance(manager, SingleCollectionKbManager), (
        f"知识库管理器是 {type(manager).__name__}，"
        f"本项目要求 SingleCollectionKbManager。"
    )

    # ⚠️ 集合名必须**逐字**等于契约值 —— 单集合策略的全部意义就在这里。
    assert manager.collection == settings.milvus.collection, (
        f"管理器用的集合是 {manager.collection!r}，"
        f"配置里是 {settings.milvus.collection!r}。"
    )


def test_the_knowledge_base_manager_shares_the_app_storage(app: FastAPI) -> None:
    """知识库管理器必须与 ``create_app`` 用**同一个** storage 实例。

    ⚠️ 这是个很容易写错、且症状很绕的约束：如果各造一个 storage，
    KB 记录会写进 A，而对话链路从 B 里读 —— 表现出来是
    「刚建的知识库在列表里看得见、检索时却说找不到」，
    看起来像缓存不一致或事务问题，实际是装配错了。

    判据直接比对象身份（``is``），不是 ``==``：两个独立的
    ``AsyncSQLAlchemyStorage`` 即使参数相同也不是同一个连接池，
    而这里要挡的正是「参数相同所以看起来没问题」的那种错法。
    """
    manager = app.state.knowledge_base_manager

    assert manager._storage is app.state.storage, (
        "知识库管理器与应用的 storage 不是同一个实例 —— "
        "KB 记录与检索会落在两个不同的存储上。"
    )


def test_the_kb_manager_reads_the_policy_from_the_app(app: FastAPI) -> None:
    """知识库管理器必须取到**应用上的那个**资源访问策略。

    这条守的是「用共享凭据建的知识库能不能检索」这件事的**装配那一半**：
    管理器解析 embedding 凭据时，属主查不到就回落到策略（原因见
    ``SingleCollectionKbManager._resolve_embedding_credential``），
    而策略是**框架**在 ``create_app`` 内部才写进 ``app.state`` 的 ——
    我们只能给它一个「到时候再取」的函数。若那个盒子永远取到 ``None``
    （例如赋值那一行被挪到了 ``app = create_app(...)`` 之前），
    回落就静默失效：表现是「共享凭据建的 KB 检索永远为空」，
    而上传、索引、列表全都正常。函数能取到、且取到的就是应用上那一个，
    是这条链路唯一的证据。
    """
    manager = app.state.knowledge_base_manager

    assert manager._access_policy_provider is not None, (
        "装配期没有把 access_policy_provider 传给知识库管理器 —— "
        "共享凭据建的 KB 会静默检索不到东西。"
    )
    assert manager._access_policy_provider() is app.state.resource_access_policy, (
        "管理器取到的策略不是应用上的那一个（或取不到）。"
        "检查 create_root_app 里 app_box['app'] = app 的位置。"
    )


def test_the_kb_manager_reaches_the_agent_wiring(
    monkeypatch: pytest.MonkeyPatch,
    clean_settings_cache: None,
) -> None:
    """★★★ 知识库管理器必须**在智能体装配之前**建好，并传进 wiring。

    ⚠️ 这条守的是本次调整装配顺序的**全部理由**。RAG 中间件要用管理器去解析
    「这个用户的 KB 句柄」；若还按原来的顺序（先 ``build_agent_wiring``、
    后建 KB 管理器），wiring 只能拿到 ``None``，检索中间件压根装不上 ——
    症状是「知识库建好了、问答却不带任何出处」，全程不报错。

    ⚠️ 断言「同一个实例」而不是「非 None」：传给 wiring 与传给
    ``create_app`` 的必须是同一份管理器 —— 桥接层的句柄缓存挂在管理器对象
    上（WeakKeyDictionary），两份实例各带一份缓存、各连一个向量库。

    Args:
        monkeypatch (`pytest.MonkeyPatch`): 用来旁路 ``build_agent_wiring`` 抓参数。
        clean_settings_cache (`None`): 清空配置单例，避免读到别的用例的缓存。
    """
    from src.server import agents_factory

    captured: dict[str, Any] = {}
    real = agents_factory.build_agent_wiring

    def spy(settings: Any, **kwargs: Any) -> Any:
        """记录 kwargs 后转调真正的装配。

        Args:
            settings (`Any`): 配置。
            **kwargs: 装配参数。

        Returns:
            `Any`: 真正的装配产物。
        """
        captured.update(kwargs)
        return real(settings, **kwargs)

    monkeypatch.setattr(agents_factory, "build_agent_wiring", spy)

    app = create_root_app()

    assert captured.get("kb_manager") is not None, (
        "build_agent_wiring 没拿到 kb_manager —— RAG 中间件将装不上。"
    )
    assert captured["kb_manager"] is getattr(
        app.state, "knowledge_base_manager", None
    ), "传给 wiring 与传给 create_app 的 kb_manager 不是同一个实例。"


def test_a_broken_redis_url_never_echoes_the_password() -> None:
    """★★★ 解析失败时抛出的错误消息里**绝不能**出现 Redis 密码。

    ⚠️ 这条守的是一类**不可逆**的泄漏。本项目的 ``redis.url`` 形如
    ``redis://:${REDIS_PASSWORD}@redis:6379/0``（``config/base.yaml`` 的
    ``redis`` 段）—— **密码就在这条字符串里**。而解析它的代码在
    ``import src.server.app`` 时就会执行（``create_root_app()`` →
    ``build_message_bus``），异常会带着消息进容器日志、
    ``docker compose logs``、以及任何一份被贴出去的排障记录。

    也就是说：一次「URL 少写了个主机名」的配置错误，就足以把生产密码
    永久留在日志里。密码一旦进了日志就只能靠轮换解决，
    而**回显它对排查毫无帮助** —— 有密码的 URL 与没密码的 URL
    在「解析不出主机名」这件事上给出的结论完全相同。

    ⚠️ 断言方式：拿一个**真的带密码**的 URL 去触发，然后查密码子串。
    只断言「报错了」是不够的 —— 那正是修复前的行为。

    ⚠️ 用 ``load_settings`` 现造一份配置，而不是改环境变量后重载模块：
    后者会污染同进程里其它用例看到的环境。
    """
    from tests.conftest import TEST_ENVIRON

    from src.config import load_settings
    from src.server.app import _redis_connection_params

    secret = "Sup3rS3cret-Redis-Pw"
    settings = load_settings(
        "test",
        environ={**TEST_ENVIRON, "ALIGO__REDIS__URL": f"redis://:{secret}@/0"},
        dotenv=False,
    )

    with pytest.raises(ValueError) as excinfo:
        _redis_connection_params(settings)

    message = str(excinfo.value)
    assert secret not in message, (
        f"Redis 密码被写进了异常消息，它会随堆栈进容器日志：\n{message}"
    )
    assert "主机名" in message, f"错误消息没说清哪儿错了：{message}"


def test_the_message_bus_is_built_with_bounded_redis_timeouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★★ 消息总线必须拿到 ``socket_timeout`` 等三项，否则会话链路无界。

    ⚠️ 这是一条**只有读源码才看得出来**的缺陷：``build_message_bus`` 里
    ``RedisMessageBus(**params)`` 长得完全正确，服务也照常启动、照常对话
    —— 在 Redis 正常时，少传 ``socket_timeout`` 与传了它**没有任何区别**。
    差别只在 Redis「接受连接但不再回包」的那一刻（网络分区、阻塞、
    主从切换）：漏传 ⇒ 每次 ``SET`` / ``XADD`` 无限等待 ⇒
    所有对话卡在 ``acquire_lock`` 的 ``while True`` 上
    （``_redis_message_bus.py:666-670``，框架没有外层截止时间），
    而 ``/healthz`` 仍是 200。

    ⚠️ 断言的是**传进去的值**而不是「函数返回的字典好看」：
    真正决定超时生效与否的，是框架构造函数收到了什么
    （它把 ``**kwargs`` 原样转交 ``redis.asyncio.ConnectionPool``）。
    所以这里替换掉 ``RedisMessageBus`` 抓取实参 —— 这正是装配契约
    该被断言的地方。

    ⚠️ 超时值刻意用 ``1.25``（非默认的 5.0）且与默认值不等：
    硬编码成任何常数都过不了「配置改了就跟着变」这一条。
    ``max_connections`` 同理（默认 50，这里用 7）。
    """
    from tests.conftest import TEST_ENVIRON

    from src.config import load_settings
    from src.server import app as app_module

    captured: dict[str, Any] = {}

    class _RecordingBus:
        """替身：只记录构造实参，不建连接（真身也不建，但会 import redis）。"""

        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(
        "agentscope.app.message_bus.RedisMessageBus",
        _RecordingBus,
    )
    settings = load_settings(
        "test",
        environ={
            **TEST_ENVIRON,
            "ALIGO__REDIS__SOCKET_TIMEOUT_SECONDS": "1.25",
            "ALIGO__REDIS__MAX_CONNECTIONS": "7",
        },
        dotenv=False,
    )

    app_module.build_message_bus(settings)

    assert captured.get("socket_timeout") == 1.25, (
        "socket_timeout 没传给 RedisMessageBus —— Redis「连得上但不回包」时，"
        f"每次 SET/XADD/PUBLISH 都会无限等待。实际收到的参数：{sorted(captured)}"
    )
    assert captured.get("socket_connect_timeout") == 1.25, (
        "socket_connect_timeout 没传 —— 「连不上」与「连上了不回话」要都能快速失败。"
    )
    assert captured.get("max_connections") == 7, (
        "max_connections 没传 —— redis.max_connections 会变成一个没有效果的配置，"
        "SSE 长连接会无上限地占用 Redis 连接。"
    )
    # 顺带钉住连接身份：补参数时最容易顺手写错的就是这几项。
    assert captured["port"] == 6379
    assert captured["db"] == 0
    assert captured["password"] == "test-password"


def test_probes_read_the_boot_flag_through_the_constant() -> None:
    """★★ ``probes.py`` 读 ``boot_completed`` 走的是常量，不是字面量。

    ⚠️ 这两处（``app.py`` 写、``probes.py`` 读）中间隔着一次进程启动，
    所以拼写错误**不会报错** —— ``getattr(state, "boot_completd", False)``
    只是安静地返回默认值 ``False``，症状是「探针永远报未就绪」，
    而排查方向会被引到 lifespan 上去。

    ⚠️ 断言方式是**读源码**而不是跑一遍：跑一遍测不出「用的是常量还是
    字面量」—— 两者在那条路径上行为完全一样，差别只在改名字的那一天。
    这正是本文件存在的理由（「跑起来看一眼」发现不了的那一类）。

    ⚠️ 同时断言 ``probes.py`` 没有从 ``src.server.app`` import 那个常量 ——
    那会构成循环导入（``app.py`` 在第 96 行就 import probes，
    而常量现在定义在 ``src/server/constants.py``）。真长回去了的话，
    一个干净的进程 import 会直接炸，而这里的用例是同一进程内跑的，
    可能因为 ``sys.modules`` 里已经有 app 而**假装通过**。
    """
    probes_source = (
        Path(__file__).resolve().parent.parent / "src" / "server" / "probes.py"
    ).read_text(encoding="utf-8")

    assert "BOOT_COMPLETED_ATTR" in probes_source, "probes.py 没在用那个常量"
    assert 'getattr(request.app.state, "boot_completed"' not in probes_source, (
        "probes.py 又退回字面量了 —— 改名字时这里不会有任何提示"
    )
    assert "from .app import" not in probes_source, (
        "probes.py 从 app.py 取常量了 —— 那是循环导入，换个入口就会炸"
    )


# ==============================================================================
# 二、lifespan 进入段：那些「被 except 吞掉就没人知道」的启动步骤
# ==============================================================================
async def test_the_profile_table_is_actually_created_at_startup(
    app: FastAPI,
    client: Any,
    settings: Any,
) -> None:
    """★★★ 进入 lifespan 之后，画像表**真的**在库里。

    ⚠️ 反例来自一次真实事故，它值得原文记在这里。

    ``ensure_profile_table`` 曾经只在 :func:`~src.server.app.create_root_app`
    的函数体里 import 过一次（那里也要用它给 :func:`src.memory.build_memory`
    传参），而调用它的地方在 ``_wrap_lifespan`` 返回的 **lifespan 闭包**里 ——
    另一个作用域。于是那一行每次启动都抛 ``NameError``。

    偏偏它被 ``except Exception`` 兜住、只打一行日志（那段代码的注释还写着
    「建表失败不让启动失败」，本意是兜住「库连不上」这类环境问题）。
    结果：``/healthz`` 200、``/readyz`` 200、``make test`` 全绿，
    而**画像表从来没有被创建过**。症状要等到第一次写画像才出现，
    那时离病因已经很远，而且日志早被滚动掉了。

    这条用例把「建表这件事真的发生了」变成**可观测**的：它不查日志、
    不查那个被吞掉的异常，而是直接问数据库「这张表在不在」。
    一个被吞掉的启动步骤，只有查它的**结果**才测得出来。

    ⚠️ 顺带钉住 ``app.state`` 上的长期记忆确实装配了 —— 如果
    ``memory`` 是 ``None``，上面那段建表代码根本不会执行，
    而本用例会因为「表不存在」而变红，方向是对的但原因会被误导。
    所以先断言它非 ``None``。
    """
    from sqlalchemy import inspect as sa_inspect

    from src.memory import profile_table
    from src.server.app import BUSINESS_ENGINE_ATTR, MEMORY_ATTR
    from src.storage.engine import business_schema_for

    # 前置条件：长期记忆确实装上了，否则下面的建表分支压根不会跑。
    assert getattr(app.state, MEMORY_ATTR, None) is not None, (
        "app.state 上没有长期记忆 —— 建表那段代码根本不会执行"
    )

    # ★ 用 client 夹具：它手动进入 lifespan，建表就发生在那里面。
    assert client is not None

    engine = getattr(app.state, BUSINESS_ENGINE_ATTR)
    schema = business_schema_for(settings)
    table = profile_table(schema)

    async with engine.connect() as conn:
        present = await conn.run_sync(
            lambda sync_conn: sa_inspect(sync_conn).has_table(
                table.name,
                schema=schema,
            ),
        )

    assert present, (
        f"启动之后 {table.fullname} 仍然不存在 —— 建表那一步被吞掉了？"
        " 去看启动日志里有没有一行被 except Exception 兜住的栈。"
    )
