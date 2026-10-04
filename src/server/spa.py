# -*- coding: utf-8 -*-
"""前端单页应用（SPA）的静态托管：vite 产物挂到 ``/``，并补齐两处缺失的放行规则。

文件职责：
    ``src/server/static/`` 里是 vite 的构建产物（``index.html`` + ``assets/`` +
    若干根级文件）。本模块提供三样东西，它们解决的是**同一个现象的三张面孔**：

        1. :class:`SpaStaticFiles` —— ``StaticFiles`` 的 SPA 回退版。
        2. :func:`is_spa_navigation` —— 「这是浏览器导航，不是接口调用」的判据。
        3. :func:`static_root_public_paths` —— 静态根目录下的文件清单（免鉴权）。

==============================================================================
现象：浏览器加载页面时**不可能带凭据**，而它要拿的东西不止一个
==============================================================================
    一次页面加载在服务端看来是这样的::

        GET /                       ← 外壳（已在 auth.PUBLIC_PATHS 里）
        GET /assets/index-xxxx.js   ← 外壳引用的 JS/CSS（/assets/ 前缀已放行）
        GET /agentscope.svg         ← 外壳引用的**根级**文件（❗没有任何前缀覆盖它）
        GET /chat                   ← 用户刷新页面时，地址栏里的就是这条（❗404）

    前两条本来就通。第三条被鉴权中间件挡成 401（``/agentscope.svg`` 既不在
    ``PUBLIC_PATHS`` 里，也不以 ``/assets/`` 开头），浏览器只能显示一个碎图标；
    第四条拿到了 JSON 404 —— 因为 ``StaticFiles(html=True)`` 的 ``html`` 参数
    **不**做 SPA 回退，它只在一个**已存在的目录** URL 上补 ``index.html``
    （见 :meth:`SpaStaticFiles.get_response` 的注释），刷新即白屏。

    本模块把这两件事分别补上：根级文件按磁盘实际内容放行（①），
    浏览器导航在 404 时补 ``index.html``（②）。

==============================================================================
为什么回退的判据是「Accept 里有 text/html」而不是「路径不在 API 里」
==============================================================================
    静态挂载位于路由表的**最后**（``app.mount("/", ...)`` 是 ``create_root_app``
    的最后一步），所以一个请求能走到 ``StaticFiles``，就意味着**前面没有任何
    API 路由匹配它** —— 这一点由 Starlette 的路由分发保证（第一个 FULL 匹配
    直接分发并返回）。于是这里不需要、也不应该再去判断「这个路径归不归 API」：
    真正的 API 请求根本到不了这里。

    剩下唯一的风险是「把本该 404 的请求变成 200 的 HTML」，所以判据要尽量窄：

        · 方法必须是 GET/HEAD（POST /chat/ 这类永远走不到这里）；
        · ``Accept`` 必须包含 ``text/html`` —— 这是浏览器地址栏/F5 的特征，
          ``fetch()`` 与 curl 的默认值是 ``*/*``，SSE 是 ``text/event-stream``；
        · 末段**不含点号** —— 保证 ``/logo.png`` 这种取不到的资源仍然是 404，
          而不是收到一份 HTML（浏览器拿到 HTML 当图片用只会更迷惑）。

    三条都满足才回退。判据同时被 :func:`is_spa_navigation` 复用于鉴权中间件
    （导航请求要放进鉴权，否则它连 ``StaticFiles`` 都到不了），两处共用一个
    函数是刻意的：**放行集合与回退集合必须完全一致**，否则会出现「鉴权放行了、
    回退不认」或者反过来的缝。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.routing import Match, Mount
from starlette.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

#: 浏览器导航请求的 ``Accept`` 特征片段（小写比较）。
#: ``fetch()`` 默认 ``*/*``、SSE 是 ``text/event-stream``、curl 是 ``*/*``，
#: 都不会命中它 —— 这让回退只对「人在地址栏里敲/按 F5」生效。
HTML_ACCEPT = "text/html"

#: 回退时返回的文件名（相对静态根目录）。
INDEX_HTML = "index.html"


def is_spa_navigation(scope: dict[str, Any]) -> bool:
    """判断这次请求是不是「浏览器在导航到某个前端路由」。

    Args:
        scope (`dict`): ASGI scope。

    Returns:
        `bool`: 是浏览器导航返回 True。

    判据（三条**同时**满足，理由见模块文档字符串）：

        1. ``type == "http"`` 且方法为 GET/HEAD；
        2. ``Accept`` 含 ``text/html``；
        3. 路径末段不含 ``.``（不像是要取一个具体文件）。
    """
    if scope.get("type") != "http":
        return False
    if scope.get("method") not in ("GET", "HEAD"):
        return False

    accept = ""
    for name, value in scope.get("headers") or ():
        if name == b"accept":
            # 同一次请求可能有多个 Accept 头（罕见但合法）：任何一个含
            # text/html 就算命中，所以这里**不 break**。
            accept += " " + value.decode("latin-1", "replace")
    if HTML_ACCEPT not in accept.lower():
        return False

    path = str(scope.get("path") or "")
    last_segment = path.rsplit("/", 1)[-1]
    return "." not in last_segment


def build_api_path_matcher(routes: Any) -> Callable[[str], bool]:
    """由路由表构造「这个路径是不是被某条 API 路由处理」的判据。

    存在的理由是一个**真实的越权入口**（不是理论问题）：

        :func:`is_spa_navigation` 曾被当作鉴权中间件里的一条**无条件放行规则**
        （``auth.py`` 的放行 ``or`` 链）。它在 ``spa.py`` 里成立的前提是
        「请求能走到 StaticFiles，说明前面没有 API 路由匹配它」—— 但鉴权
        中间件看到的是**全部**请求，这个前提在那里根本不成立。

        放行分支只做一件事：剥掉客户端自带的 ``X-User-ID``。对于**自身
        不依赖身份**的端点（框架里有一批：``GET /agent/schema``、
        ``/hub/mcp``、``/credential/schemas``、``/sop/schema``、
        ``/channels/types``），剥头毫无作用 —— 端点照样执行。

        实测：``curl -H 'Accept: text/html' /agent/schema`` → **200**，
        换成 ``Accept: */*`` → 401。而 ``Accept`` 是普通请求头，谁都能填。
        开启 JWT 后行为不变（SPA 分支在 JWT 校验之前短路）。

    修法就是这一行：**先问路由表**。只有「没有任何 API 路由匹配这个路径」
    才允许走浏览器导航放行 —— 也就是回到 ``spa.py`` 本来想表达的语义。

    ⚠️ 收集「有 ``matches`` 的顶层路由」，但**排除 ``Mount``**：静态资源自己
    就是挂在 ``/`` 上的 ``Mount``，它会匹配**一切**路径。把它算进来，判据就恒
    为真，SPA 回退永远不生效（每个前端路由都变成「命中 API」⇒ 401 白屏）。

    ⚠️⚠️ 不能只收 ``starlette.routing.Route``。框架的 API 路由**不是** ``Route``：
    ``create_app`` 把它们装进了 ``fastapi.routing._IncludedRouter``（也是
    ``BaseRoute`` 子类，同样实现 ``matches``）。只按 ``Route`` 过滤的后果是
    实测可复现的 —— ``api_routes`` 里只剩 ``/openapi.json``、``/docs``、
    ``/redoc`` 三条，**每一条真实 API 路径都判为 False**，于是 ``_is_spa_route``
    反向得出「这是前端路由」，越权原样保留：

        matcher("/agent/schema") → False   ← 错，它明明是一条 API

    所以这里按**鸭子类型**收（``hasattr(route, "matches")``），不按具体类。
    附带地，这也让本函数对 Starlette / FastAPI 的版本变化更耐受。

    Args:
        routes (`Any`): ``app.routes``（Starlette 的路由列表）。

    Returns:
        `Callable[[str], bool]`: 吃掉一个路径，返回是否被某条 API 路由处理。
    """
    api_routes = [
        route
        for route in routes
        if isinstance(route, Mount) is False and hasattr(route, "matches")
    ]

    def matches(path: str) -> bool:
        """``path`` 是否被某条 API 路由处理。

        ⚠️ 用 ``route.matches(scope)`` 而不是读 ``route.path_regex``：
        后者在 Starlette 1.7 里不是公开属性。构造一个**最小 scope** 即可 ——
        ``matches`` 只读 ``type`` / ``path`` / ``method`` / ``root_path``。

        ``FULL`` 与 ``PARTIAL`` 都算命中：``PARTIAL`` 的典型来源是**方法不匹配**
        （``/chat/`` 只有 POST，探针用 GET 去问它），路径本身仍归 API —— 这种
        路径绝不能被当成前端路由放行，否则一个 ``Accept: text/html`` 的 POST
        就能绕过鉴权直达端点。
        """
        # 方法给 GET：`matches` 只拿它判 FULL/PARTIAL（即「方法对不对」），
        # 而我们关心的只是「路径在不在路由表里」，两种都不是 NONE。
        probe = {
            "type": "http",
            "path": path,
            "method": "GET",
            "root_path": "",
            "headers": [],
        }
        return any(route.matches(probe)[0] is not Match.NONE for route in api_routes)

    return matches


class SpaStaticFiles(StaticFiles):
    """``StaticFiles`` + SPA 回退：404 且是浏览器导航时返回 ``index.html``。

    只重写 :meth:`get_response`，不动 :meth:`__call__` —— 后者负责按 ASGI
    协议发送响应，``StaticFiles`` 的实现已经把 ``HTTPException`` 转成响应，
    我们只要在**抛之前**把 404 换掉即可。
    """

    async def get_response(self, path: str, scope: dict[str, Any]) -> Response:
        """取静态文件；取不到且是浏览器导航时退回 ``index.html``。

        ⚠️ 这里依赖一个**事实**，写代码时容易想当然：``StaticFiles(html=True)``
        的 ``html`` **不**提供 SPA 回退。它只在「请求的是一个目录」时补该目录
        下的 ``index.html``（``/chat`` 不是一个存在的目录，所以补不上），
        外加把 ``404.html`` 当 404 响应体返回。真正的 SPA 回退要靠下面这段。

        Args:
            path (`str`): 相对静态根目录的路径（Starlette 已做过去穿越校验）。
            scope (`dict`): ASGI scope。

        Returns:
            `Response`: 文件响应，或回退的 ``index.html``。

        Raises:
            HTTPException: 非导航请求取不到文件时原样抛出（保持不变 → 404 JSON）。
        """
        try:
            response = await super().get_response(path, scope)
        except HTTPException as exc:
            if exc.status_code != 404 or not is_spa_navigation(scope):
                raise
            return await self._index_response(scope)

        # ⚠️⚠️ 「没抛异常」**不等于**「文件存在」—— 这是 html=True 下的一个陷阱，
        # 它会让整个回退静默失效：
        #
        #   ``StaticFiles.get_response`` 在静态根目录里存在 ``404.html`` 时，
        #   是 ``return FileResponse(full_path, status_code=404)`` —— 一个**正常
        #   返回**的 404 响应，而不是 raise。于是上面那个 ``except`` 永远进不去，
        #   SPA 回退变成死代码。
        #
        #   ``404.html`` 是静态托管的常见产物（vite 插件、Netlify/Vercel 模板
        #   都可能生成）。本仓库当前的产物里没有它，所以测试全绿、线上也暂时
        #   不发作 —— 而一旦有人加进去，症状是「首页正常、前端路由 F5 白屏」，
        #   且 ``tests/test_spa.py`` 照样全绿（它的夹具目录里没有 404.html）。
        #
        # 因此这里**必须**再看一眼状态码：真正要回退的是「这个路径没有对应的
        # 文件」，而 404 就是这件事的结论 —— 不管它是抛出来的还是返回的。
        if response.status_code == 404 and is_spa_navigation(scope):
            return await self._index_response(scope)
        return response

    async def _index_response(self, scope: dict[str, Any]) -> Response:
        """SPA 回退：返回静态根目录下的 ``index.html``。

        ⚠️ 回退**只有一层**：``index.html`` 自己取不到时让它的 404 照常抛出，
        不在这里兜底成 200 —— 那会把「产物压根没构建好」伪装成一个正常页面，
        症状从「首页 404，日志里一句话都没有」变成「首页一片空白，什么都不说」。
        两者都难查，但前者至少指对了方向。

        Args:
            scope (`dict`): ASGI scope（只用于日志）。

        Returns:
            `Response`: ``index.html`` 的文件响应。

        Raises:
            HTTPException: ``index.html`` 取不到时原样抛出。
        """
        logger.debug("SPA 回退：%s → %s", scope.get("path"), INDEX_HTML)
        return await super().get_response(INDEX_HTML, scope)

def static_root_public_paths(static_dir: Path) -> frozenset[str]:
    """列出静态根目录下的**文件**，转成免鉴权路径。

    ``/agentscope.svg``、``/favicon.ico`` 这类由 ``web/frontend/public/`` 拷进
    产物根目录的文件，是 ``index.html`` 直接引用的子资源；浏览器取它们时同样
    不带任何凭据（见模块文档字符串），必须在鉴权白名单里。

    为什么按磁盘实际内容算、而不是在 ``PUBLIC_PATHS`` 里硬编码文件名：
    新增一个 ``public/`` 文件就要记得同步改白名单，忘了的表现是「构建产物里
    明明有、页面上却是个碎图标」—— 一个不会报错、只在浏览器里看得见的故障。
    按目录算则天然跟着产物走。

    ⚠️ 只放行**文件**，不放行**目录**：目录会被当成前缀放行（``/images/``），
    而一个叫 ``api`` 或 ``sessions`` 的目录会把真接口一并打开 —— 这是**静默**
    的越权。子目录下的资源请显式加进 ``auth.PUBLIC_PREFIXES``；发现未覆盖的
    子目录时本函数会打一条 warning，不会让它悄悄漏掉。

    Args:
        static_dir (`Path`): 前端产物根目录（可能不存在）。

    Returns:
        `frozenset[str]`: 形如 ``{"/agentscope.svg", "/favicon.ico"}`` 的路径集合；
        目录不存在时返回空集合（前端还没构建，属正常状态）。
    """
    if not static_dir.is_dir():
        return frozenset()

    # 延迟 import：constants 与 middleware 都不依赖本模块，这里只是为了避免
    # 模块级循环（middleware.auth 反过来要 import 本模块的 is_spa_navigation）。
    from .middleware.auth import PUBLIC_PREFIXES

    paths: set[str] = set()
    for entry in sorted(static_dir.iterdir()):
        if entry.is_file():
            paths.add("/" + entry.name)
        elif entry.is_dir() and ("/" + entry.name + "/") not in PUBLIC_PREFIXES:
            logger.warning(
                "静态资源目录 %s 下的子目录 %s 不在鉴权白名单前缀 %s 里："
                "其中被 index.html 引用的资源会返回 401。"
                "请把“/%s/”加入 src/server/middleware/auth.py 的 PUBLIC_PREFIXES。",
                static_dir,
                entry.name,
                PUBLIC_PREFIXES,
                entry.name,
            )
    return frozenset(paths)
