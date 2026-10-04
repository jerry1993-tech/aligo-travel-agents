# -*- coding: utf-8 -*-
"""鉴权中间件：把「凭据」翻译成框架认得的 ``X-User-ID``。

文件职责：
    在请求进入应用**之前**确定调用者是谁，并把结论写进 ``scope``，供两处使用：

        1. ``scope["headers"]`` 里的 ``x-user-id`` —— **框架**读的就是它
           （``agentscope/app/deps.py::get_current_user_id``，全仓 85 个
           ``Depends`` 点都从这里取身份）；
        2. ``scope["state"]`` —— 我们自己的路由用 ``request.state.xxx`` 读，
           不必再把请求头解析一遍。

上下游依赖：
    - 上游：``src/server/app.py`` 注册，位于中间件链的
      ``TraceContext → HttpMetrics → Auth → RateLimit → 应用`` 中的第三层。
    - 下游：``agentscope.app.deps`` 的依赖注入；``rate_limit.py`` 从
      ``scope["state"]`` 取已解析好的身份来当限流键。

==============================================================================
为什么是「注入 X-User-ID」而不是替换框架的 get_current_user_id
==============================================================================
    框架的 ``get_current_user_id`` 自称是临时方案
    （``deps.py``：「Temporary header-based identity; will be replaced by JWT auth.」），
    但它在 **13 个路由文件、85 个 ``Depends`` 点**被使用。替换它意味着
    把这 85 处全部改掉，且框架每次升级都可能加新的调用点 —— 漏掉的那一处
    就是一个**没有鉴权的口子**，而且不会有任何报错。

    注入请求头则是「一处修改、全链路生效」：框架读到的仍然是它熟悉的那个头，
    只是头的值已经由我们替换成了可信来源。代价是我们必须彻底删掉客户端
    自带的同名头 —— 见 :func:`~src.server.middleware._asgi.set_header` 的警告。

==============================================================================
两条通道
==============================================================================
    1. **``X-User-ID`` 直连**（默认，``auth.jwt_enabled=false``）：
       要求请求带非空的 ``X-User-ID``；它**就是**身份，不做真伪校验。
       适合本地开发与内网可信环境 —— 此时前端的每一次 fetch 都会带上它。

    2. **JWT Bearer**（``auth.jwt_enabled=true``）：
       要求 ``Authorization: Bearer <token>``，校验签名/有效期/受众/签发者，
       取 ``sub`` 作为 user_id。**先删光客户端自带的 x-user-id 再注入** ——
       这是这条通道唯一的、也是全部的安全价值所在。

==============================================================================
白名单
==============================================================================
    下列路径**不要求凭据**（取值见 ``AUTH_MODE_PUBLIC``）：

        · 探针 ``/healthz`` ``/readyz`` ``/metrics`` —— 被鉴权拦住的探针会让
          编排系统误判容器已死并重启它（经典的「健康检查把自己打挂」）；
        · ``/api/v1/health`` —— 同上，业务侧的浅探活；
        · 前端静态资源（``PUBLIC_PATHS`` 的根级文件、``PUBLIC_PREFIXES`` 的
          前缀、以及装配期传入的 ``extra_public_paths``）与 ``/docs``
          ``/openapi.json`` —— 浏览器加载页面时**还没有**任何凭据，
          拦住它们的结果是白屏而不是 401 提示；
        · **浏览器导航**（:func:`~src.server.spa.is_spa_navigation`）
          —— 前端是 SPA，``/chat``、``/schedule`` 这类地址刷新时地址栏就是它，
          而它们是前端路由、没有对应的后端端点。判据只对 ``Accept: text/html``
          的 GET/HEAD 生效，``fetch()`` / curl / SSE 都不命中。

    ⚠️ **白名单路径上仍然会剥离客户端自带的 ``X-User-ID``**（见
    :meth:`AuthMiddleware.__call__`）。这是白名单规则的**安全前提**：
    放行只意味着「我们这一层不拦」，而框架的 ``get_current_user_id``
    仍然会去读这个头 —— 若放行时原样带着客户端给的值过去，
    一个 ``Accept: text/html`` 的请求就能凭空冒用任意用户身份。
    因此「免鉴权」的准确含义是「不要求凭据」，不是「不碰请求头」。

    ⚠️ **框架的 ``/health`` 不在白名单里**，这是刻意的：它自带
    ``Depends(get_current_user_id)``（``app/_router/_health.py``），
    本来就是一个需要身份的端点，今天如此、加了中间件之后仍然如此。
    把它放进白名单反而会造成「白名单放行了、框架又拦住」的割裂。

==============================================================================
保留身份：唯一一条「凭据合法但必须拒绝」的规则
==============================================================================
    两条通道解析出来的身份，在**放行之前**都要过一遍
    :func:`~src.llm.identity.is_reserved_user_id`，命中即 403。

    要拦的原因在本项目里是具体的：运营者的真实 API key 被写成一条属主为
    ``aligo-system`` 的系统凭据（``src/llm/system_credential.py``），
    而框架的可见性规则是「属主读自己的凭据是明文、且 ``editable: true``」。
    一个 ``X-User-ID: aligo-system`` 的请求因此可以：

        · ``GET /credential/`` 拿到 API key 明文；
        · ``DELETE /credential/aligo-system-model`` 删掉那条全员共享的凭据
          —— 所有用户的模型当场失效。

    这不是「提权」（``X-User-ID`` 直连模式本来就不校验真伪），而是
    **把内部身份当成了可声明的身份**。系统身份从来不是给人用的登录名，
    它只是存储层的一个隔离维度；共享策略那边用 ``viewer_id == SYSTEM_USER_ID``
    保证「属主不会在共享列表里再看见一次」，这里用同一条规则的另一个
    方向保证「没有人能假装成那个属主」，两处必须同时成立。

    ⚠️ 三条通道**都要拦**，一处漏掉就等于没有拦：直连通道（身份即请求头）、
    JWT 通道（``sub`` 由签发方填写，但一个带保留身份的合法 token 同样不该
    放行），以及 ``require_user_header=false`` 的匿名通道（它不重写请求头，
    若只拦前两条，关掉那个开关就成了绕过规则的捷径）。
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from ...llm.identity import is_reserved_user_id
from ...observability import metrics as metrics_mod
from ..spa import is_spa_navigation
from ._asgi import (
    Message,
    Receive,
    Scope,
    Send,
    header_value,
    remove_header,
    send_json,
    set_header,
    state_of,
)

# ==== scope["state"] 的键名 =====================================================
#: 解析出来的 user id。消费方（限流中间件、业务路由）统一从这里读。
USER_ID_STATE_KEY = "aligo_user_id"

#: 本次请求的身份来源。取值见下面 ``AUTH_MODE_*``。
AUTH_MODE_STATE_KEY = "aligo_auth_mode"

#: 身份来源：JWT Bearer 解析而来。
AUTH_MODE_JWT = "jwt"
#: 身份来源：客户端直传的 ``X-User-ID``（未校验真伪）。
AUTH_MODE_HEADER = "header"
#: 放行的匿名请求（白名单路径）。
AUTH_MODE_PUBLIC = "public"
#: 允许匿名（``require_user_header=false``）且确实没带身份。
AUTH_MODE_ANONYMOUS = "anonymous"

#: 鉴权失败原因 —— **同时是指标标签的取值域**。
#: 新增原因时必须一起改这里，否则 ``aligo_auth_failures_total`` 上会出现
#: 一条只出现过一次的时序，面板分组随之失去意义。
FAILURE_MISSING_BEARER = "missing_bearer"
FAILURE_INVALID_TOKEN = "invalid_token"
FAILURE_MISSING_USER_HEADER = "missing_user_header"
#: 客户端（或 token 的 ``sub``）声明了一个**保留身份**。见模块文档字符串
#: 的「保留身份」一节 —— 这是本中间件唯一一条「身份格式完全合法但必须
#: 拒绝」的规则，也是最容易在重构中被整条删掉的规则（它的表现是 403 计数
#: 掉到 0，而其它一切看起来完全正常）。
FAILURE_RESERVED_IDENTITY = "reserved_identity"

#: 免鉴权的**精确**路径。
PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/healthz",
        "/readyz",
        "/metrics",
        "/api/v1/health",
        # 前端外壳：浏览器加载页面时不可能带凭据（见模块文档字符串）。
        "/",
        "/index.html",
        "/favicon.ico",
        "/manifest.webmanifest",
        "/robots.txt",
        # 接口文档。⚠️ 生产若不希望公开，在这一行注释掉即可 ——
        # 但注意拦住 /docs 的后果是 Swagger UI 打不开，而不是一条 401 提示。
        "/docs",
        "/docs/oauth2-redirect",
        "/redoc",
        "/openapi.json",
    },
)

#: 免鉴权的**前缀**（静态资源带哈希文件名，无法逐个列举）。
PUBLIC_PREFIXES: tuple[str, ...] = ("/assets/", "/static/")

#: JWT 头里 Bearer 方案的标识（RFC 6750 的大小写不敏感）。
_BEARER_PREFIX = "bearer "

#: 请求头名（ASGI 规范要求小写 bytes）。
_USER_ID_HEADER = b"x-user-id"
_AUTHORIZATION_HEADER = b"authorization"


def _is_public(path: str) -> bool:
    """判断路径是否在免鉴权白名单里。

    Args:
        path (`str`): 请求路径。

    Returns:
        `bool`: 免鉴权返回 True。
    """
    return path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES)


class AuthMiddleware:
    """纯 ASGI 鉴权中间件。

    刻意**不用** ``BaseHTTPMiddleware``：后者会缓冲流式响应，会破坏
    ``/sessions/{id}/stream`` 的实时性（详见 ``http_trace.py`` 的模块文档字符串）。
    """

    def __init__(
        self,
        app: Any,
        *,
        settings: Any,
        extra_public_paths: Iterable[str] = (),
        api_path_matcher: Any = None,
    ) -> None:
        """保存下游应用与鉴权配置。

        Args:
            app (`Any`): 下游 ASGI 应用（FastAPI 中间件的约定签名）。
            settings (`Settings`): 全量配置，只读取 ``settings.auth``。
                传整个 ``Settings`` 而不是拆开的几个参数，是为了让
                「配置只有一个来源」这件事在构造函数签名上就看得出来。
            extra_public_paths (`Iterable[str]`): 额外的**精确**免鉴权路径。
                装配期由 ``src/server/app.py`` 按静态产物的实际内容算出来
                （见 :func:`~src.server.spa.static_root_public_paths`）——
                白名单的取值依赖磁盘上的产物，这份知识只有装配根节点有，
                中间件自己不去猜、也不去读盘。
            api_path_matcher (`Any`): ``path -> bool``，「这个路径是否被某条
                **API 路由**处理」。装配期由 ``src/server/app.py`` 用
                :func:`~src.server.spa.build_api_path_matcher` 从 ``app.routes``
                构造。它是浏览器导航放行规则的**前提条件**（见下）。

                ⚠️ **默认 ``None`` = 不放行任何导航**（安全默认值）。
                不知道路由表时，唯一正确的做法是「不知道就按 API 处理」：
                误判成 API 的代价是某个前端路由刷新返回 401（立刻可见、
                立刻有人报），误判成导航的代价是**任意匿名用户可读的接口**
                （静默的越权）。两者不对称，默认值必须倒向安全的那一边。

        Raises:
            RuntimeError: 开启了 JWT 但没有安装 PyJWT 时。**在装配期就报错**，
                而不是等到第一个请求：那时失败会长成「所有请求 500」，
                与「少装一个包」相距甚远。
        """
        self.app = app
        self._extra_public_paths: frozenset[str] = frozenset(extra_public_paths)
        self._api_path_matcher = api_path_matcher
        auth = settings.auth
        self._jwt_enabled: bool = bool(auth.jwt_enabled)
        self._require_user_header: bool = bool(auth.require_user_header)
        self._jwt_secret: str = auth.jwt_secret
        self._jwt_algorithm: str = auth.jwt_algorithm
        self._jwt_audience: str = auth.jwt_audience.strip()
        self._jwt_issuer: str = auth.jwt_issuer.strip()

        self._jwt: Any = None
        if self._jwt_enabled:
            try:
                import jwt as pyjwt
            except ImportError as exc:  # pragma: no cover - 依赖已固定版本
                raise RuntimeError(
                    "auth.jwt_enabled 为 true 但未安装 PyJWT。"
                    "请执行 `pip install PyJWT`（requirements.txt 已固定版本），"
                    "或把 ALIGO__AUTH__JWT_ENABLED 设为 false 走 X-User-ID 直连模式。",
                ) from exc
            self._jwt = pyjwt

    def _is_spa_route(self, scope: Scope, path: str) -> bool:
        """这次请求是不是「浏览器在加载一个**没有后端端点**的前端路由」。

        ★ 这是本中间件最容易写错、代价最高的一处，两个条件**缺一不可**：

            1. :func:`~src.server.spa.is_spa_navigation` —— 形状上像浏览器导航
               （GET/HEAD + ``Accept: text/html`` + 路径末段无点）；
            2. **没有任何 API 路由匹配这个路径** —— 它确实只能由 StaticFiles
               托管，也就是一次真正的页面加载。

        ⚠️ 只判第 1 条就是一条**越权入口**，而且它看起来完全合理：

            ``is_spa_navigation`` 在 ``spa.py`` 里的前提是「请求能走到
            StaticFiles，说明前面没有 API 路由匹配它」。但鉴权中间件看到的
            是**全部**请求，那个前提在这里不成立。

            放行分支只做一件事：剥掉客户端自带的 ``X-User-ID``。对于
            **自身不依赖身份**的端点（框架里有一批：``GET /agent/schema``、
            ``/hub/mcp``、``/credential/schemas``、``/sop/schema``、
            ``/channels/types``），剥头毫无作用 —— 端点照常执行。

            实测：``curl -H 'Accept: text/html' /agent/schema`` → **200**，
            换成 ``Accept: */*`` → 401。而 ``Accept`` 是普通请求头，谁都能填；
            开启 JWT 之后行为不变（这条分支在 JWT 校验之前短路）。

        ⚠️ 判据 2 的默认值（``api_path_matcher is None``）是**不放行**。
        理由见构造函数：误判成 API 只是某个页面刷新 401（吵、但可见），
        误判成导航是一条静默的匿名可读接口。默认值倒向安全的那一边。

        Args:
            scope (`Scope`): ASGI scope。
            path (`str`): 已解析出的请求路径。

        Returns:
            `bool`: 是「纯前端路由的页面加载」返回 True。
        """
        if not is_spa_navigation(scope):
            return False
        matcher = self._api_path_matcher
        if matcher is None:
            # 见 docstring：不知道路由表时按 API 处理。
            return False
        return not matcher(path)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """处理一次 ASGI 调用。

        Args:
            scope (`Scope`): ASGI scope。
            receive (`Receive`): 接收可调用对象（本中间件不消费请求体）。
            send (`Send`): 发送可调用对象。
        """
        # lifespan / websocket 原样透传：lifespan 不是请求；
        # websocket 没有本中间件依赖的 path/headers 语义（框架的 SSE 走 HTTP，
        # 不走 websocket），贸然处理只会引入一条没人测过的分支。
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path") or ""
        state = state_of(scope)

        if (
            _is_public(path)
            or path in self._extra_public_paths
            or self._is_spa_route(scope, path)
        ):
            state[AUTH_MODE_STATE_KEY] = AUTH_MODE_PUBLIC
            # ⚠️⚠️ 放行之前**必须**剥掉客户端自带的 ``X-User-ID``。
            # 放行只代表「我们这一层不拦」，而框架的 ``get_current_user_id``
            # 照样会读这个头并按它认身份（``app/deps.py``）。
            # 不剥的后果：`curl -H 'X-User-ID: victim' -H 'Accept: text/html' /mcp`
            # 就绕过了整套鉴权 —— 而 SPA 导航这条放行规则对**任何**客户端
            # 都成立（Accept 是个普通请求头，谁都能填），所以这是一个
            # 「不用伪造签名、不用猜 token」的越权入口。
            # 剥离之后，白名单路径的真实含义才和文档一致：
            # **不要求**凭据，而不是**不看**凭据。
            remove_header(scope, _USER_ID_HEADER)
            await self.app(scope, receive, send)
            return

        if self._jwt_enabled:
            if not await self._authenticate_jwt(scope, state, send):
                return
        elif self._require_user_header:
            if not await self._authenticate_header(scope, state, send):
                return
        else:
            # 允许匿名：带了身份就记下来（限流要用），没带就标成匿名放行。
            # ⚠️ 此时框架的业务路由仍会拒绝（它们自己 Depends(get_current_user_id)，
            #    缺头返回 422）。这个开关只影响**我们这一层**放不放行，
            #    关掉它并不会让框架变得可以匿名访问 —— 这一点必须说清楚，
            #    否则运维会以为「关掉鉴权就能匿名压测」，然后在业务接口上撞 422。
            # ⚠️ 这条分支**不重写请求头**（带了身份就原样透传给框架），
            # 因此保留身份必须在这里也拦一次 —— 否则「把 require_user_header
            # 关掉」就成了绕过上一条规则的开关，而它们看起来毫不相干。
            #
            # 归一化方式与 _authenticate_header 保持一致（``strip()``）：
            # HTTP 服务器解析请求时本来就会去掉字段值两侧的空白，
            # ``X-User-ID: " aligo-system "`` 到框架那里就是 ``aligo-system``。
            # 判据用同一套归一化，才不会出现「我们看到的」与「框架看到的」分叉。
            user_id = (header_value(scope, _USER_ID_HEADER) or "").strip()
            if is_reserved_user_id(user_id):
                await self._reject_reserved_identity(send)
                return
            if user_id:
                state[USER_ID_STATE_KEY] = user_id
                state[AUTH_MODE_STATE_KEY] = AUTH_MODE_HEADER
            else:
                state[AUTH_MODE_STATE_KEY] = AUTH_MODE_ANONYMOUS

        await self.app(scope, receive, send)

    # --------------------------------------------------------------------------
    # 通道一：X-User-ID 直连
    # --------------------------------------------------------------------------
    async def _authenticate_header(
        self,
        scope: Scope,
        state: dict[str, Any],
        send: Send,
    ) -> bool:
        """校验 ``X-User-ID`` 直连模式。

        Args:
            scope (`Scope`): ASGI scope。
            state (`dict`): 该请求的状态字典（出参）。
            send (`Send`): ASGI 发送可调用对象。

        Returns:
            `bool`: 通过返回 True；已返回 401 返回 False。
        """
        user_id = (header_value(scope, _USER_ID_HEADER) or "").strip()
        if is_reserved_user_id(user_id):
            # ⚠️ 这个头**就是**身份（本通道不做真伪校验，见模块文档字符串），
            # 所以「声明一个保留身份」在这条通道上等价于「已是那个身份」。
            # 拒绝理由不是「你缺少凭据」而是「这个身份不由客户端声明」，
            # 因此是 403 而不是 401 —— 见 _reject_reserved_identity。
            await self._reject_reserved_identity(send)
            return False
        if not user_id:
            # ⚠️ 这里是 **401**，不是框架的 422。两者都「拒绝」，但含义不同：
            #    401 = 「你没给凭据」，422 = 「请求格式不对」。客户端据此决定
            #    「去登录」还是「改请求体」，混淆会让前端在缺 token 时去修 JSON。
            #    框架侧实测是 422（它的 Header(...) 没有默认值，FastAPI 在进入
            #    函数体前就拦下了），所以**不能**照抄框架的行为 ——
            #    我们这一层挡在前面，就有责任给出语义正确的状态码。
            await self._reject(
                send,
                status=401,
                reason=FAILURE_MISSING_USER_HEADER,
                detail="缺少 X-User-ID 请求头。",
            )
            return False

        state[USER_ID_STATE_KEY] = user_id
        state[AUTH_MODE_STATE_KEY] = AUTH_MODE_HEADER
        return True

    # --------------------------------------------------------------------------
    # 通道二：JWT Bearer
    # --------------------------------------------------------------------------
    async def _authenticate_jwt(
        self,
        scope: Scope,
        state: dict[str, Any],
        send: Send,
    ) -> bool:
        """校验 ``Authorization: Bearer`` 并把 ``sub`` 注入成 ``X-User-ID``。

        Args:
            scope (`Scope`): ASGI scope。
            state (`dict`): 该请求的状态字典（出参）。
            send (`Send`): ASGI 发送可调用对象。

        Returns:
            `bool`: 通过返回 True；已返回 401 返回 False。
        """
        raw = header_value(scope, _AUTHORIZATION_HEADER) or ""
        if not raw.lower().startswith(_BEARER_PREFIX):
            await self._reject(
                send,
                status=401,
                reason=FAILURE_MISSING_BEARER,
                detail="缺少 Authorization: Bearer <token>。",
            )
            return False

        token = raw[len(_BEARER_PREFIX):].strip()
        claims = self._decode(token)
        if claims is None:
            # 错误详情**刻意不细分**（过期 / 签名错 / 受众不符…）：把它回给调用方
            # 等于告诉攻击者「签名是对的，只是过期了」，为爆破提供了反馈信号。
            # 细分信息进日志与指标，不进响应体。
            await self._reject(
                send,
                status=401,
                reason=FAILURE_INVALID_TOKEN,
                detail="token 无效或已过期。",
            )
            return False

        if is_reserved_user_id(claims):
            # JWT 通道同样要拦：签名合法不等于这个身份可以被用来登录。
            # ``sub`` 是**签发方**填的，而保留身份（``aligo-system``）根本
            # 不该有对应的登录主体 —— 带它的合法 token 只可能来自签发侧：
            # IdP 配置写错，或有人拿真密钥手搓了一枚。scripts/mint_token.py
            # 已明确拒签它（那里的「约束四」），所以这里拦的是那些旁路。
            await self._reject_reserved_identity(send)
            return False

        # 「先删光再注入」——见 _asgi.set_header 的警告：只追加的话，
        # Starlette 取首个匹配项，客户端伪造的 x-user-id 会胜出。
        set_header(scope, _USER_ID_HEADER, claims)
        state[USER_ID_STATE_KEY] = claims
        state[AUTH_MODE_STATE_KEY] = AUTH_MODE_JWT
        return True

    def _decode(self, token: str) -> str | None:
        """校验并解析 token，返回 ``sub``。

        Args:
            token (`str`): Bearer 之后的原始 token。

        Returns:
            `str | None`: 校验通过且 ``sub`` 为非空字符串时返回它，否则 ``None``
            （调用方据此返回 401 —— 具体原因只进日志，不进响应体）。
        """
        options: dict[str, Any] = {
            # require：**强制**这两个声明必须存在。
            #   不带 exp 的 token 永不过期 —— 一次泄漏就是永久后门，
            #   而它只比正常 token 少一个字段，肉眼极难发现。
            #   不带 sub 的 token 则没有身份可言，放行等于匿名。
            "require": ["exp", "sub"],
        }

        kwargs: dict[str, Any] = {
            "algorithms": [self._jwt_algorithm],
        }

        # audience / issuer 为空串时**不传**这个参数 ——
        # 传 None 是「不校验」，传 "" 是「要求 aud 恰好等于空串」，语义完全不同，
        # 而后者会让所有正常 token 都失败。
        if self._jwt_audience:
            kwargs["audience"] = self._jwt_audience
        else:
            # ⚠️ 未配置受众时必须**显式关掉** aud 校验，光「不传 audience」是不够的。
            #
            # PyJWT 的规则是：token 里**有** ``aud`` 而调用方**没给** ``audience``
            # 参数时，抛 ``InvalidAudienceError`` —— 而不是「跳过校验」。
            # 于是默认配置（``jwt_audience: ""``，即「不校验受众」）会与
            # 事实相反：它拒绝了**所有**带 aud 的 token。而现实里的 IdP
            # （Auth0 / Keycloak / 阿里云 IDaaS…）几乎都会签 aud，
            # 也就是说默认配置下**一个 token 都过不了**。
            #
            # 这个坑的隐蔽之处在于：自签的测试 token 若不带 aud 就完全正常，
            # 只有在接上真实 IdP 时才炸 —— 而那时已经在联调环境了。
            options["verify_aud"] = False
        if self._jwt_issuer:
            kwargs["issuer"] = self._jwt_issuer

        kwargs["options"] = options

        try:
            payload = self._jwt.decode(token, self._jwt_secret, **kwargs)
        except Exception:  # noqa: BLE001 - PyJWT 的异常层级较深，统一按「无效」处理
            # 不把异常原文回给调用方（见 _authenticate_jwt 的注释），
            # 但仍要进日志，否则运维只能看到一个光秃秃的 401 计数。
            import logging

            logging.getLogger(__name__).warning(
                "JWT 校验失败（原因不回传给调用方，避免为爆破提供反馈）。",
                exc_info=True,
            )
            return None

        sub = payload.get("sub")
        if not isinstance(sub, str) or not sub.strip():
            return None
        return sub.strip()

    # --------------------------------------------------------------------------
    # 保留身份
    # --------------------------------------------------------------------------
    async def _reject_reserved_identity(self, send: Send) -> None:
        """拒绝一个保留身份（见模块文档字符串的「保留身份」一节）。

        三条通道共用这一处文案与状态码，理由有二：

            · 文案只有一处 —— 它同时是运维看到的唯一解释。写岔了会变成
              「有的入口说保留身份、有的入口说无效凭据」，排障的人先怀疑
              自己的请求，而问题其实在规则本身；
            · 状态码是 **403 而不是 401** 是刻意的。401 的语义是「你没提供
              可用的凭据」，客户端据此会去走登录流程 —— 但没有任何一次
              登录能让人变成系统身份，于是它会陷入「登了再试、再 401」。
              403 说的是「凭据本身没问题，但这件事实在不能做」，这正是
              这里要表达的，也告诉调用方**别再重试了**。

        ⚠️ 响应体**不回显**被拒绝的那个 id，但理由**不是**「名字要保密」——
        恰恰相反，它压根不是秘密：常量就在 ``src/llm/identity.py:42``，
        ``docs/`` 与 README 都在讲它，配了密钥的部署里 ``GET /credential/``
        返回的共享凭据记录上也带着这个属主。把「名字没泄漏」当成一层防御
        是危险的：它会在名字早已公开的今天让人误以为防线破了个洞，而真正
        拦下这次请求的从来只是下面那行 ``is_reserved_user_id`` 判断。

        真正的理由更朴素：本函数的响应体是**固定文案**（``_reject`` 只发
        ``{"detail": detail}``），把一个调用方自己填进来的字符串再映回去，
        对排查没有半点帮助 —— 它本来就知道自己发了什么。

        Args:
            send (`Send`): ASGI 发送可调用对象。
        """
        await self._reject(
            send,
            status=403,
            reason=FAILURE_RESERVED_IDENTITY,
            detail="该身份为系统保留身份，客户端不得声明。",
        )

    # --------------------------------------------------------------------------
    async def _reject(
        self,
        send: Send,
        *,
        status: int,
        reason: str,
        detail: str,
    ) -> None:
        """统一构造拒绝响应并记指标。

        Args:
            send (`Send`): ASGI 发送可调用对象。
            status (`int`): HTTP 状态码。
            reason (`str`): 指标标签取值，见 ``FAILURE_*`` 常量。
            detail (`str`): 回给调用方的说明。
        """
        metrics_mod.observe_auth_failure(reason)
        headers: list[tuple[bytes, bytes]] = []
        if status == 401:
            # RFC 6750 §3 要求 401 带上这条响应头，客户端据此知道该用哪种方案。
            # 它不是形式主义：缺了它，某些客户端会退化成「一直重试同样的请求」。
            headers.append(
                (b"www-authenticate", b'Bearer realm="aligo"'),
            )
        await send_json(send, status, {"detail": detail}, extra_headers=headers)


__all__ = [
    "AUTH_MODE_ANONYMOUS",
    "AUTH_MODE_HEADER",
    "AUTH_MODE_JWT",
    "AUTH_MODE_PUBLIC",
    "AUTH_MODE_STATE_KEY",
    "PUBLIC_PATHS",
    "PUBLIC_PREFIXES",
    "USER_ID_STATE_KEY",
    "AuthMiddleware",
]
