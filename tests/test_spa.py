# -*- coding: utf-8 -*-
"""``src/server/spa.py`` 的单元测试：SPA 回退的判据、边界与静态根文件清单。

为什么这些用例值得单独存在（而不是只靠 ``test_server_assembly.py`` 的
端到端用例）：端到端用例跑的是**真实产物**，产物里有哪些文件、前端路由
叫什么名字，都会随前端代码变化 —— 它验证的是「现在这套产物是好的」，
不验证「判据本身是对的」。这里的三组用例钉住的恰恰是判据：

    1. ``is_spa_navigation`` 的取值边界（哪种请求才算「浏览器导航」）；
    2. ``SpaStaticFiles`` 在 404 / 非导航 / 带点号路径上的行为；
    3. ``static_root_public_paths`` 的「只放行文件、不放行目录」。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.applications import Starlette
from starlette.routing import Mount

from src.server.spa import (
    SpaStaticFiles,
    is_spa_navigation,
    static_root_public_paths,
)

HTML_ACCEPT = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"


def _scope(
    path: str,
    *,
    method: str = "GET",
    accept: str | None = "*/*",
    scope_type: str = "http",
) -> dict[str, Any]:
    """拼一个最小的 ASGI http scope。

    Args:
        path (`str`): 请求路径。
        method (`str`): HTTP 方法。
        accept (`str | None`): ``Accept`` 头；``None`` 表示不带这个头。
        scope_type (`str`): scope 类型。

    Returns:
        `dict`: ASGI scope。
    """
    headers: list[tuple[bytes, bytes]] = []
    if accept is not None:
        headers.append((b"accept", accept.encode()))
    return {
        "type": scope_type,
        "method": method,
        "path": path,
        "headers": headers,
    }


# ==============================================================================
# 一、is_spa_navigation —— 「这是浏览器导航吗」
# ==============================================================================
@pytest.mark.parametrize(
    ("path", "method", "accept", "scope_type", "expected"),
    [
        # 浏览器地址栏 / F5：这才应该命中。
        ("/chat", "GET", HTML_ACCEPT, "http", True),
        ("/chat/abc/def", "GET", HTML_ACCEPT, "http", True),
        ("/", "GET", HTML_ACCEPT, "http", True),
        ("/schedule", "HEAD", "text/html", "http", True),
        # 只是大小写不同，仍然命中（Accept 是大小写不敏感的）。
        ("/chat", "GET", "TEXT/HTML", "http", True),
        # fetch() 的默认 Accept 是 */*，SSE 是 text/event-stream：
        # 都**不**命中，否则接口会收到一份 HTML 而不是 JSON 404。
        ("/chat", "GET", "*/*", "http", False),
        ("/chat", "GET", None, "http", False),
        ("/chat", "GET", "text/event-stream", "http", False),
        ("/api/v1/me", "GET", "application/json", "http", False),
        # POST 永远不走回退：写请求收到一份 HTML 只会更难排查。
        ("/chat/", "POST", HTML_ACCEPT, "http", False),
        ("/chat", "DELETE", HTML_ACCEPT, "http", False),
        # 末段带点号 ⇒ 用户要的是一个具体文件，取不到就该 404，
        # 而不是把 index.html 当成图片返回。
        ("/logo.png", "GET", HTML_ACCEPT, "http", False),
        ("/assets/index-a1b2c3.js", "GET", HTML_ACCEPT, "http", False),
        # 非 http scope 一律不碰（lifespan/websocket 没有 path 语义）。
        ("/chat", "GET", HTML_ACCEPT, "websocket", False),
    ],
)
def test_is_spa_navigation(
    path: str,
    method: str,
    accept: str | None,
    scope_type: str,
    expected: bool,
) -> None:
    """导航判据的取值边界（见 ``src/server/spa.py`` 模块文档字符串）。"""
    scope = _scope(path, method=method, accept=accept, scope_type=scope_type)

    assert is_spa_navigation(scope) is expected


def test_is_spa_navigation_accepts_any_header_containing_text_html() -> None:
    """同一次请求可以有多个 ``Accept`` 头：任何一个含 text/html 就算命中。"""
    scope = _scope("/chat", accept="application/json")
    scope["headers"].append((b"accept", b"text/html"))

    assert is_spa_navigation(scope) is True


# ==============================================================================
# 二、SpaStaticFiles —— 404 时补 index.html
# ==============================================================================
def _asgi_client(static_dir: Path) -> httpx.AsyncClient:
    """把「挂载在 ``/`` 上的静态目录」包成一个 HTTP 客户端。

    ⚠️ 这里**不能**直接拿 ``SpaStaticFiles`` 当应用喂给 ``ASGITransport``：
    裸的 ``StaticFiles`` 遇到 404 是 ``raise HTTPException``，而把
    ``HTTPException`` 翻译成 404 响应的是 ``ExceptionMiddleware`` ——
    它属于外层应用，不在静态应用自己身上。少这一层，异常会直接冒到
    测试里（表现为 ``starlette.exceptions.HTTPException: 404``），
    于是「404」这件事根本测不到。
    因此这里照生产的样子组装：一个 ``Starlette`` 应用 + ``Mount("/", ...)``。

    Args:
        static_dir (`Path`): 前端产物目录。

    Returns:
        `httpx.AsyncClient`: 指向该应用的客户端（无需 lifespan，静态应用没有）。
    """
    app = Starlette(routes=[Mount("/", app=SpaStaticFiles(directory=str(static_dir), html=True))])
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://spa.test",
    )


@pytest.fixture
def static_dir(tmp_path: Path) -> Path:
    """构造一份最小的「前端产物」。

    Args:
        tmp_path (`Path`): pytest 提供的临时目录。

    Returns:
        `Path`: 产物目录。
    """
    (tmp_path / "index.html").write_text("<!doctype html><title>shell</title>", encoding="utf-8")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "agentscope.svg").write_text("<svg/>", encoding="utf-8")
    return tmp_path


@pytest.fixture
def static_client(static_dir: Path) -> httpx.AsyncClient:
    """指向「最小前端产物」的客户端。

    Args:
        static_dir (`Path`): 产物目录。

    Returns:
        `httpx.AsyncClient`: 客户端。
    """
    return _asgi_client(static_dir)


async def test_real_file_is_served(static_client: httpx.AsyncClient) -> None:
    """磁盘上真有这个文件就照常返回它。"""
    async with static_client as client:
        response = await client.get("/agentscope.svg")

    assert response.status_code == 200
    assert response.text == "<svg/>"


async def test_browser_navigation_falls_back_to_the_shell(
    static_client: httpx.AsyncClient,
) -> None:
    """★ 浏览器刷新 ``/chat`` 必须拿到外壳，而不是 404。

    这是本模块存在的理由：``html=True`` 只在「请求的是一个**已存在的目录**」
    时补 ``index.html``，``/chat`` 不是目录，于是它返回 404 ——
    浏览器表现为白屏，而服务端日志里只是一条平平无奇的 404。
    """
    async with static_client as client:
        response = await client.get("/chat", headers={"Accept": HTML_ACCEPT})

    assert response.status_code == 200, "浏览器导航没有回退到 index.html"
    assert "text/html" in response.headers["content-type"]
    assert "shell" in response.text


async def test_fallback_needs_the_html_accept_header(static_client: httpx.AsyncClient) -> None:
    """不带 ``Accept: text/html`` 时**不**回退 —— 接口调用要继续拿到 404。

    这是回退规则的安全边界：接口客户端（``fetch()`` / curl / SDK）的默认
    ``Accept`` 不是 text/html，所以「路径不存在」对它们始终是 404。
    """
    async with static_client as client:
        response = await client.get("/chat")

    assert response.status_code == 404


async def test_fallback_does_not_swallow_missing_assets(static_client: httpx.AsyncClient) -> None:
    """末段带点号 ⇒ 仍然 404（否则一张图会收到一份 HTML）。"""
    async with static_client as client:
        response = await client.get("/logo.png", headers={"Accept": HTML_ACCEPT})

    assert response.status_code == 404


async def test_fallback_does_not_apply_to_writes(static_client: httpx.AsyncClient) -> None:
    """POST 到不存在的路径仍然是 405/404，不会收到 HTML。"""
    async with static_client as client:
        response = await client.post("/chat", headers={"Accept": HTML_ACCEPT})

    assert response.status_code in (404, 405)
    assert "text/html" not in response.headers.get("content-type", "")


@pytest.fixture
def static_dir_with_404_page(static_dir: Path) -> Path:
    """在产物根目录里**额外**放一个 ``404.html``。

    这不是假想场景：``404.html`` 是静态托管的常见产物（Vite 插件、
    Netlify / Vercel 的模板都可能生成它），运维「顺手」拷进产物目录
    是完全正常的动作。

    Args:
        static_dir (`Path`): 一个已经带 ``index.html`` 的产物目录。

    Returns:
        `Path`: 同一个目录（已补上 ``404.html``）。
    """
    (static_dir / "404.html").write_text(
        "<!doctype html><title>not-found-page</title>", encoding="utf-8"
    )
    return static_dir


async def test_a_404_page_in_the_bundle_does_not_disable_the_fallback(
    static_dir_with_404_page: Path,
) -> None:
    """★★★ 产物里存在 ``404.html`` 时，SPA 回退**仍然**必须生效。

    这是一个**静默失效**的陷阱，本模块的
    :meth:`~src.server.spa.SpaStaticFiles.get_response` 专门为它写了一段
    注释 —— 因为它的两个分支看起来都「没抛异常」，代码读起来是对的：

        ``StaticFiles.get_response`` 在静态根目录下存在 ``404.html`` 时，
        走的**不是** ``raise HTTPException(404)``，而是
        ``return FileResponse(..., status_code=404)`` —— 一个正常返回的
        404 响应。于是 ``except HTTPException`` 那段整个变成死代码，
        回退不触发，用户刷新 ``/chat`` 拿到的是那份 ``404.html``。

    ⚠️ 症状的迷惑性正是它值得一条专门用例的原因：**首页正常、前端路由
    一刷新就白屏**，而且 ``tests/test_spa.py`` 的其它夹具目录里没有
    ``404.html``，所以**它们全都照样绿**。本地不会有任何提示，
    要等到有人往产物里加了这个文件才在线上发作。
    """
    async with _asgi_client(static_dir_with_404_page) as client:
        response = await client.get("/chat", headers={"Accept": HTML_ACCEPT})

    assert response.status_code == 200, (
        f"产物里有 404.html 时，浏览器导航 /chat 返回了 {response.status_code}（应为 200）。"
        f" 说明回退被 StaticFiles 返回的 404 响应绕过了（它不抛异常）。"
    )
    assert "shell" in response.text, (
        "回退拿到的不是 index.html —— 很可能是那份 404.html 被当成了页面。"
    )
    assert "not-found-page" not in response.text


async def test_a_404_page_in_the_bundle_still_serves_api_style_404s(
    static_dir_with_404_page: Path,
) -> None:
    """★ 上一条的对照组：**非导航**请求仍要拿到产物里那份 ``404.html``。

    回退只能吃掉「浏览器导航取不到页面」这一种 404；接口客户端的 404
    必须原样透出，包括产物里那份自定义页面。否则回退就从「补首页」
    扩大成了「把所有 404 都变成 200」。
    """
    async with _asgi_client(static_dir_with_404_page) as client:
        response = await client.get("/chat")

    assert response.status_code == 404
    assert "not-found-page" in response.text, (
        "非导航请求应当原样拿到产物里的 404.html，而不是被回退成首页。"
    )


async def test_missing_index_html_is_not_papered_over(tmp_path: Path) -> None:
    """⚠️ 产物里没有 ``index.html`` 时，回退**不能**把 404 变成 200。

    否则「构建没跑完」会被伪装成一个正常页面，排查方向直接跑到前端去。
    """
    (tmp_path / "assets").mkdir()

    async with _asgi_client(tmp_path) as client:
        response = await client.get("/chat", headers={"Accept": HTML_ACCEPT})

    assert response.status_code == 404


# ==============================================================================
# 三、static_root_public_paths —— 静态根目录的免鉴权清单
# ==============================================================================
def test_root_files_are_public_paths(tmp_path: Path) -> None:
    """根目录下的**文件**逐个转成 ``/文件名``。"""
    (tmp_path / "index.html").write_text("x", encoding="utf-8")
    (tmp_path / "agentscope.svg").write_text("x", encoding="utf-8")
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "app.js").write_text("x", encoding="utf-8")

    paths = static_root_public_paths(tmp_path)

    assert paths == frozenset({"/index.html", "/agentscope.svg"})


def test_missing_dir_is_an_empty_set(tmp_path: Path) -> None:
    """前端还没构建时目录不存在，属正常状态：返回空集合，不报错。"""
    assert static_root_public_paths(tmp_path / "nope") == frozenset()


def test_directory_is_not_turned_into_a_prefix(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★ 子目录**不**会被当作前缀放行，且必须留下一条 warning。

    「把目录当前缀放行」是静默越权：产物里若有一个叫 ``api`` 的目录，
    放行 ``/api/`` 就等于把整套接口打开。所以只放行文件；
    没被 ``PUBLIC_PREFIXES`` 覆盖的子目录要显式告警，让人看得见。
    """
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "logo.png").write_text("x", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="src.server.spa"):
        paths = static_root_public_paths(tmp_path)

    assert paths == frozenset(), "目录被误当成免鉴权路径了"
    assert any("images" in record.getMessage() for record in caplog.records), (
        "未覆盖的子目录必须告警，否则引用它的资源会静默 401"
    )


def test_assets_dir_is_covered_by_prefix_and_not_warned(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``assets/`` 已经由 ``PUBLIC_PREFIXES`` 覆盖：不重复告警。"""
    (tmp_path / "assets").mkdir()

    with caplog.at_level(logging.WARNING, logger="src.server.spa"):
        static_root_public_paths(tmp_path)

    assert not caplog.records
