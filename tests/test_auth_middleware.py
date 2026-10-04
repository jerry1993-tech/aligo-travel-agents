# -*- coding: utf-8 -*-
"""鉴权中间件的单元测试（``src/server/middleware/auth.py``）。

==============================================================================
为什么这一层要直接喂 ASGI scope，而不是通过 httpx 打整个应用
==============================================================================
    走整个应用时，**框架也会参与**：``get_current_user_id`` 对缺失的
    ``X-User-ID`` 返回的是 **422**（它的 ``Header(...)`` 没有默认值，
    FastAPI 在进入函数体前就拦下了），而本中间件返回的是 **401**。
    两者都「拒绝」，但测出来的数字取决于哪一层先说话 ——
    于是「中间件到底返回了什么」这件事就被掩盖了。

    直接喂 scope 则能精确断言：状态码、响应头、指标标签、以及
    **下游看到的请求头到底是什么**。本文件的核心断言
    （伪造的 ``X-User-ID`` 有没有被删掉）只有在这一层才看得见 ——
    它发生在应用被调用**之前**。

==============================================================================
本文件保护的头号性质
==============================================================================
    ``set_header`` 必须**先删光同名头再追加**。

    Starlette 的 ``Headers`` 取的是**第一个**匹配值。只追加不删除的话，
    客户端自己带一个 ``X-User-ID: 别人`` 就会盖过我们注入的值 ——
    JWT 模式形同虚设，而且**功能测试全绿**：
    请求成功地访问到了「别人」的数据，返回 200，没有任何异常。
    这是一条只会在安全审计中被发现、日常测试中完全隐形的缺陷。
"""

from __future__ import annotations

import json
from typing import Any
from unittest import mock

import jwt as pyjwt
import pytest

from src.server.middleware import auth as auth_mod
from src.server.middleware.auth import (
    AUTH_MODE_ANONYMOUS,
    AUTH_MODE_HEADER,
    AUTH_MODE_JWT,
    AUTH_MODE_PUBLIC,
    AUTH_MODE_STATE_KEY,
    FAILURE_INVALID_TOKEN,
    FAILURE_MISSING_BEARER,
    FAILURE_MISSING_USER_HEADER,
    FAILURE_RESERVED_IDENTITY,
    USER_ID_STATE_KEY,
    AuthMiddleware,
)

#: 保留身份的字面值。**刻意手抄**而不是 ``from src.llm.identity import
#: SYSTEM_USER_ID``：这一条规则的要害是「它必须等于系统凭据的属主」，
#: 而单测若跟着常量一起漂移，就永远测不出漂移。见
#: ``test_the_reserved_identity_is_the_system_credential_owner``。
RESERVED_ID = "aligo-system"

#: 测试用的签名密钥。**只存在于测试里** —— 真实密钥由
#: ``ALIGO__AUTH__JWT_SECRET`` 提供，且绝不写进任何受版本控制的文件。
#:
#: 长度刻意超过 32 字节：PyJWT 对 HMAC-SHA256 的密钥短于 32 字节会发
#: ``InsecureKeyLengthWarning``。测试输出里堆满无关警告的代价是真实的 ——
#: 真正的告警会被埋掉，而人一旦开始习惯「反正有一堆警告」，
#: 就等于没有告警。
TEST_SECRET = "test-secret-for-unit-tests-only-0123456789abcdef"


class _FakeAuth:
    """``settings.auth`` 的最小替身。"""

    def __init__(
        self,
        *,
        jwt_enabled: bool = False,
        require_user_header: bool = True,
        jwt_secret: str = TEST_SECRET,
        jwt_algorithm: str = "HS256",
        jwt_audience: str = "",
        jwt_issuer: str = "",
    ) -> None:
        """记录各项鉴权配置。

        Args:
            jwt_enabled (`bool`): 是否启用 JWT 校验。
            require_user_header (`bool`): 是否要求 ``X-User-ID``。
            jwt_secret (`str`): 签名密钥。
            jwt_algorithm (`str`): 签名算法。
            jwt_audience (`str`): 受众；空串表示不校验。
            jwt_issuer (`str`): 签发者；空串表示不校验。
        """
        self.jwt_enabled = jwt_enabled
        self.require_user_header = require_user_header
        self.jwt_secret = jwt_secret
        self.jwt_algorithm = jwt_algorithm
        self.jwt_audience = jwt_audience
        self.jwt_issuer = jwt_issuer


def _settings(**kwargs: Any) -> Any:
    """造一个只带 ``auth`` 的 settings 替身。

    Args:
        **kwargs: 透传给 :class:`_FakeAuth`。

    Returns:
        `Any`: 带 ``auth`` 属性的对象。
    """

    class _S:
        auth = _FakeAuth(**kwargs)

    return _S()


class _Recorder:
    """一个只记录「收到了什么」的下游 ASGI 应用。"""

    def __init__(self) -> None:
        """初始化记录器。"""
        self.called = False
        self.scope: dict[str, Any] = {}

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """记录 scope 并返回一个 200。

        Args:
            scope (`Any`): ASGI scope。
            receive (`Any`): 接收可调用对象（不使用）。
            send (`Any`): 发送可调用对象。
        """
        self.called = True
        self.scope = scope
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            },
        )
        await send({"type": "http.response.body", "body": b"{}"})

    def header(self, name: bytes) -> list[bytes]:
        """取出下游看到的所有同名请求头。

        Args:
            name (`bytes`): 小写的头名。

        Returns:
            `list[bytes]`: 全部匹配值，按顺序。
        """
        return [v for k, v in self.scope.get("headers", []) if k == name]


def _scope(
    path: str = "/api/v1/me",
    *,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> dict[str, Any]:
    """造一个最小可用的 HTTP scope。

    Args:
        path (`str`): 请求路径。
        headers (`list[tuple[bytes, bytes]] | None`): 请求头。

    Returns:
        `dict`: ASGI scope。
    """
    return {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": list(headers or []),
        "client": ("127.0.0.1", 12345),
    }


async def _drive(
    middleware: AuthMiddleware,
    scope: dict[str, Any],
) -> tuple[int, list[tuple[bytes, bytes]], dict[str, Any]]:
    """驱动一次中间件调用，收集响应状态、响应头与响应体。

    Args:
        middleware (`AuthMiddleware`): 待测中间件。
        scope (`dict`): ASGI scope。

    Returns:
        `tuple[int, list, dict]`: ``(状态码, 响应头, 解析后的响应体)``。
    """
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        """收集下游发出的 ASGI 消息。"""
        sent.append(message)

    async def receive() -> dict[str, Any]:
        """本中间件不消费请求体。"""
        return {"type": "http.request", "body": b"", "more_body": False}

    await middleware(scope, receive, send)

    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(
        m.get("body", b"") for m in sent if m["type"] == "http.response.body"
    )
    return (
        int(start["status"]) if start else 0,
        list(start.get("headers", [])) if start else [],
        json.loads(body) if body else {},
    )


def _token(**claims: Any) -> str:
    """签发一个测试 token。

    Args:
        **claims: 放进 payload 的声明。未提供 ``exp`` 时默认给一个未来的时间。

    Returns:
        `str`: 编码后的 JWT。
    """
    payload = {"exp": 4_102_444_800, **claims}
    return pyjwt.encode(payload, TEST_SECRET, algorithm="HS256")


# ==============================================================================
# 一、通往应用的两条直连路径
# ==============================================================================
@pytest.mark.asyncio
async def test_valid_user_header_passes_and_records_identity() -> None:
    """带合法 ``X-User-ID`` 时放行，并把身份写进 scope["state"]。"""
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings())

    status, _, _ = await _drive(
        mw,
        _scope(headers=[(b"x-user-id", b"alice")]),
    )

    assert status == 200
    assert recorder.called
    assert recorder.scope["state"][USER_ID_STATE_KEY] == "alice"
    assert recorder.scope["state"][AUTH_MODE_STATE_KEY] == AUTH_MODE_HEADER


@pytest.mark.asyncio
async def test_missing_user_header_is_401_not_422() -> None:
    """缺 ``X-User-ID`` 时返回 **401**，并带上 ``WWW-Authenticate``。

    ⚠️ 框架对同一件事返回 **422**（它的 ``Header(...)`` 没有默认值，
    在进入函数体前就被 FastAPI 拦下）。两者都「拒绝」但含义不同：
    401 = 你没给凭据（去登录），422 = 请求格式不对（改请求体）。
    混淆会让前端在缺 token 时去修 JSON。本中间件挡在最前面，
    有责任给出语义正确的那个 —— 这条用例把这个选择固定下来。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings())

    status, headers, body = await _drive(mw, _scope())

    assert status == 401
    assert recorder.called is False, "未通过鉴权的请求不该到达应用"
    assert body["detail"] == "缺少 X-User-ID 请求头。"
    # RFC 6750 §3：401 必须告诉客户端该用哪种方案。
    assert (b"www-authenticate", b'Bearer realm="aligo"') in headers


@pytest.mark.asyncio
async def test_blank_user_header_is_rejected() -> None:
    """空串的 ``X-User-ID`` 等同于没给。

    与框架的行为刻意不同：框架那里「头存在但为空」会走到函数体里再判空，
    「头不存在」则在更早的校验层被拦下 —— 同一个语义走两条路径。
    这里统一成一种结果。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings())

    status, _, _ = await _drive(mw, _scope(headers=[(b"x-user-id", b"   ")]))

    assert status == 401


# ==============================================================================
# 二、白名单
# ==============================================================================
@pytest.mark.parametrize(
    "path",
    ["/healthz", "/readyz", "/metrics", "/api/v1/health", "/", "/docs", "/openapi.json"],
)
@pytest.mark.asyncio
async def test_public_paths_are_not_authenticated(path: str) -> None:
    """白名单路径免鉴权，且**不碰请求头**。

    「不碰请求头」是刻意的：探针与静态资源不需要身份，
    在这里注入一个空身份只会让下游多一条需要判断的分支。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings())

    status, _, _ = await _drive(mw, _scope(path))

    assert status == 200
    assert recorder.scope["state"][AUTH_MODE_STATE_KEY] == AUTH_MODE_PUBLIC


@pytest.mark.asyncio
async def test_framework_health_is_not_whitelisted() -> None:
    """框架的 ``/health`` **不在**白名单里。

    它自带 ``Depends(get_current_user_id)``，本来就是一个需要身份的端点。
    把它放进白名单会造成「我们放行了、框架又拦住」的割裂：
    响应仍然是 401/422，而白名单里写着它免鉴权 —— 下一个人排查时会
    先怀疑白名单没生效。让两层的判断保持一致才是对的。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings())

    status, _, _ = await _drive(mw, _scope("/health"))

    assert status == 401


#: 浏览器导航请求的 ``Accept``（地址栏与 F5）。
_BROWSER_ACCEPT = (b"accept", b"text/html,application/xhtml+xml,*/*;q=0.8")


@pytest.mark.asyncio
async def test_extra_public_paths_are_exact_matches() -> None:
    """``extra_public_paths`` 是**精确**路径，不会被当成前缀。

    装配期传进来的是「产物根目录下确有哪些文件」，文件名本身不含通配语义。
    若实现里写成 ``startswith``，``/agentscope.svg`` 就会顺带放行
    ``/agentscope.svg/../../admin`` 这类构造出来的路径 —— 一个只有
    在白名单里出现「像前缀的字符串」时才会显形的越权。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(
        recorder,
        settings=_settings(),
        extra_public_paths={"/agentscope.svg"},
    )

    status_hit, _, _ = await _drive(mw, _scope("/agentscope.svg"))
    assert status_hit == 200
    assert recorder.scope["state"][AUTH_MODE_STATE_KEY] == AUTH_MODE_PUBLIC

    recorder_miss = _Recorder()
    mw_miss = AuthMiddleware(
        recorder_miss,
        settings=_settings(),
        extra_public_paths={"/agentscope.svg"},
    )
    status_miss, _, _ = await _drive(mw_miss, _scope("/agentscope.svg.bak"))
    assert status_miss == 401, "extra_public_paths 被当成了前缀匹配"


#: 一个「/chat 是纯前端路由、/agent/schema 是 API」的最小路由判据。
#: 真实实现是 ``src/server/spa.py::build_api_path_matcher``（按 ``app.routes``
#: 逐条 ``Route.matches`` 构造），本文件不重复那一套 —— 它的行为由
#: ``tests/test_server_assembly.py`` 在**真装配**上验证（那里才有真实路由表）。
def _fake_api_matcher(path: str) -> bool:
    """测试用的路由判据：把 ``/agent/`` 前缀当作 API。

    Args:
        path (`str`): 请求路径。

    Returns:
        `bool`: 是该由 API 处理返回 True。
    """
    return path.startswith("/agent/")


def _nav_middleware(recorder: Any) -> AuthMiddleware:
    """构造一个「知道路由表」的鉴权中间件。

    真实装配（``create_root_app``）**总是**传 ``api_path_matcher``；
    只有直接构造的用例需要显式给一个，否则浏览器导航一律不放行
    （那是刻意的安全默认值，见中间件的构造函数）。

    Args:
        recorder (`Any`): 下游记录器。

    Returns:
        `AuthMiddleware`: 可放行浏览器导航的中间件。
    """
    return AuthMiddleware(
        recorder,
        settings=_settings(),
        api_path_matcher=_fake_api_matcher,
    )


@pytest.mark.asyncio
async def test_browser_navigation_is_public_and_strips_the_identity_header() -> None:
    """★ 浏览器导航放行，**并且**剥掉客户端自带的 ``X-User-ID``。

    这是 SPA 回退规则的安全前提。导航请求（``Accept: text/html`` 的 GET）
    被放行是必须的 —— 前端路由刷新时地址栏里就是 ``/chat``，而它没有凭据
    可带；但放行**不等于**把客户端给的身份原样交给下游：框架的
    ``get_current_user_id`` 只认 ``X-User-ID`` 这个头，不看它是谁给的。
    不剥的后果是一条 curl 就能冒用任意用户（``Accept`` 是普通头，谁都能填）。

    ⚠️ 中间件必须拿到 ``api_path_matcher`` 才会放行 —— 见
    :func:`test_api_paths_are_never_reachable_by_spoofing_the_accept_header`。
    """
    recorder = _Recorder()
    mw = _nav_middleware(recorder)

    status, _, _ = await _drive(
        mw,
        _scope("/chat", headers=[_BROWSER_ACCEPT, (b"x-user-id", b"victim")]),
    )

    assert status == 200, "浏览器导航应被放行"
    assert recorder.scope["state"][AUTH_MODE_STATE_KEY] == AUTH_MODE_PUBLIC
    assert recorder.header(b"x-user-id") == [], (
        "放行时没有剥掉客户端自带的 X-User-ID —— 下游会把它当成可信身份"
    )


@pytest.mark.asyncio
async def test_api_paths_are_never_reachable_by_spoofing_the_accept_header() -> None:
    """★ 带 ``Accept: text/html`` 的 **API 路径**照样要凭据。

    这条是同一个缺陷的第二半，也是更严重的一半：

        「浏览器导航」曾经是一条**无条件**放行规则。放行分支只剥掉
        ``X-User-ID``，对**自身不依赖身份**的端点（框架里有一批：
        ``GET /agent/schema``、``/hub/mcp``、``/credential/schemas``、
        ``/sop/schema``、``/channels/types``）剥头毫无作用 —— 端点照常执行。

        于是 ``curl -H 'Accept: text/html' /agent/schema`` 返回 200，
        而 ``Accept`` 是普通请求头，谁都能填。开启 JWT 之后同样中招
        （这条分支在 JWT 校验之前短路）。

    ⚠️ 修法是「先问路由表」：只有**没有任何 API 路由匹配**的路径才算页面加载。
    删掉 ``api_path_matcher`` 这个条件（或把它的默认值改成放行），本用例会红。
    """
    recorder = _Recorder()
    mw = _nav_middleware(recorder)

    status, _, _ = await _drive(
        mw,
        _scope("/agent/schema", headers=[_BROWSER_ACCEPT]),
    )

    assert status == 401, (
        "带 Accept: text/html 的 API 路径被放行了 —— 匿名可读的越权入口"
    )
    assert recorder.scope.get("state", {}).get(AUTH_MODE_STATE_KEY) != AUTH_MODE_PUBLIC


@pytest.mark.asyncio
async def test_unknown_route_table_fails_closed() -> None:
    """★ 没拿到路由表时**不放行**任何导航（安全默认值）。

    两种误判的代价是不对称的：把页面加载误判成 API，代价是某个前端路由
    刷新返回 401 —— 吵、立刻可见、立刻有人报；把 API 误判成页面加载，
    代价是一条**静默的**匿名可读接口。因此默认值必须倒向安全的那一边。

    这条用例同时钉住了「将来有人在别处直接构造 AuthMiddleware」时的行为：
    他拿到的是一个更严的中间件，而不是一个更松的。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings())

    status, _, _ = await _drive(mw, _scope("/chat", headers=[_BROWSER_ACCEPT]))

    assert status == 401, (
        "没有路由表时仍然放行了浏览器导航 —— 安全默认值被改成了「放行」"
    )


@pytest.mark.asyncio
async def test_non_navigation_request_is_not_public() -> None:
    """没有 ``Accept: text/html`` 的同一个路径仍然要凭据（默认拒绝）。"""
    recorder = _Recorder()
    mw = _nav_middleware(recorder)

    status, _, _ = await _drive(mw, _scope("/chat", headers=[(b"accept", b"*/*")]))

    assert status == 401


@pytest.mark.asyncio
async def test_non_http_scope_passes_through() -> None:
    """lifespan 等非 http scope 原样透传。

    ⚠️ 这里必须给一个**真的** send，不能图省事传 ``None``：
    ``_Recorder`` 会调用 ``await send(...)`` 回一个 200，
    而 ``'NoneType' object is not callable`` 会伪装成
    「中间件没透传」，把注意力引到错误的模块上。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings())
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        """收集下游发出的 ASGI 消息。"""
        sent.append(message)

    await mw({"type": "lifespan"}, None, send)  # type: ignore[arg-type]

    assert recorder.called, "非 http scope 必须原样透传给下游"


# ==============================================================================
# 三、JWT 模式
# ==============================================================================
@pytest.mark.asyncio
async def test_jwt_sub_is_injected_as_user_header() -> None:
    """JWT 模式下把 ``sub`` 注入成 ``X-User-ID``。"""
    recorder = _Recorder()
    mw = AuthMiddleware(
        recorder,
        settings=_settings(jwt_enabled=True),
    )

    status, _, _ = await _drive(
        mw,
        _scope(headers=[(b"authorization", f"Bearer {_token(sub='bob')}".encode())]),
    )

    assert status == 200
    assert recorder.header(b"x-user-id") == [b"bob"]
    assert recorder.scope["state"][AUTH_MODE_STATE_KEY] == AUTH_MODE_JWT


@pytest.mark.asyncio
async def test_forged_user_header_is_stripped_in_jwt_mode() -> None:
    """★ JWT 模式下，客户端自带的 ``X-User-ID`` 必须被**删掉**。

    这是整个鉴权链路的**安全核心**，也是本文件里最重要的用例。

    Starlette 的 ``Headers`` 取**第一个**匹配值。若只追加不删除，
    伪造的头排在我们注入的头前面，它会赢 —— 于是「拿 bob 的 token
    访问 alice 的数据」会成功，且返回 200、没有任何异常。
    攻击者只需要多发一个请求头。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(
        recorder,
        settings=_settings(jwt_enabled=True),
    )

    status, _, _ = await _drive(
        mw,
        _scope(
            headers=[
                (b"x-user-id", b"alice"),  # 伪造
                (b"authorization", f"Bearer {_token(sub='bob')}".encode()),
            ],
        ),
    )

    assert status == 200
    # 关键断言：下游**只能**看到 bob，且只看到一个值。
    assert recorder.header(b"x-user-id") == [b"bob"], (
        f"伪造的 x-user-id 没有被清掉：{recorder.header(b'x-user-id')}。"
        f"Starlette 取首个匹配值，客户端伪造的身份会因此生效。"
    )


@pytest.mark.asyncio
async def test_jwt_missing_bearer_is_401() -> None:
    """没有 ``Authorization`` 头时返回 401（原因 ``missing_bearer``）。"""
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings(jwt_enabled=True))

    status, _, body = await _drive(mw, _scope())

    assert status == 401
    assert "Bearer" in body["detail"]


@pytest.mark.asyncio
async def test_jwt_bearer_prefix_is_case_insensitive() -> None:
    """``bearer`` 的大小写不敏感（RFC 6750 明确要求）。"""
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings(jwt_enabled=True))

    status, _, _ = await _drive(
        mw,
        _scope(headers=[(b"authorization", f"bearer {_token(sub='bob')}".encode())]),
    )

    assert status == 200
    assert recorder.header(b"x-user-id") == [b"bob"]


@pytest.mark.asyncio
async def test_jwt_wrong_signature_is_401() -> None:
    """签名不对的 token 被拒。"""
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings(jwt_enabled=True))
    # 密钥同样是 48 字节：这里要测的是「密钥**不对**」，不是「密钥太短」——
    # 短密钥会让 PyJWT 先发一条 InsecureKeyLengthWarning，
    # 于是失败时你得先分辨是哪一类问题。
    forged = pyjwt.encode(
        {"sub": "mallory", "exp": 4_102_444_800},
        "a-completely-different-secret-0123456789abcdef",
        algorithm="HS256",
    )

    status, _, _ = await _drive(
        mw,
        _scope(headers=[(b"authorization", f"Bearer {forged}".encode())]),
    )

    assert status == 401
    assert recorder.called is False


@pytest.mark.asyncio
async def test_jwt_without_exp_is_rejected() -> None:
    """**没有 ``exp`` 的 token 必须被拒**。

    ⚠️ 这条不是形式主义：不带 ``exp`` 的 token **永不过期** ——
    一次泄漏就是一个永久后门。而它只比正常 token 少一个字段，
    肉眼审阅时极难发现。``options={"require": ["exp", "sub"]}``
    把这个判断交给库，比我们自己解析 payload 再判要可靠。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings(jwt_enabled=True))
    no_exp = pyjwt.encode({"sub": "bob"}, TEST_SECRET, algorithm="HS256")

    status, _, _ = await _drive(
        mw,
        _scope(headers=[(b"authorization", f"Bearer {no_exp}".encode())]),
    )

    assert status == 401, "不带 exp 的 token 永不过期，必须拒绝"


@pytest.mark.asyncio
async def test_jwt_without_sub_is_rejected() -> None:
    """没有 ``sub`` 的 token 没有身份可言，必须拒。"""
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings(jwt_enabled=True))
    no_sub = pyjwt.encode({"exp": 4_102_444_800}, TEST_SECRET, algorithm="HS256")

    status, _, _ = await _drive(
        mw,
        _scope(headers=[(b"authorization", f"Bearer {no_sub}".encode())]),
    )

    assert status == 401


@pytest.mark.asyncio
async def test_jwt_rejection_detail_does_not_leak_reason() -> None:
    """401 的响应体**不得**区分「过期」「签名错」「受众不符」。

    细分开来等于给攻击者一个反馈信号：「签名是对的，只是过期了」——
    这直接告诉爆破者该继续调什么。细分信息进日志与指标，不进响应体。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings(jwt_enabled=True))
    expired = pyjwt.encode({"sub": "bob", "exp": 1}, TEST_SECRET, algorithm="HS256")

    _, _, body = await _drive(
        mw,
        _scope(headers=[(b"authorization", f"Bearer {expired}".encode())]),
    )

    assert body["detail"] == "token 无效或已过期。"
    # 响应体里不该出现任何一种具体原因。
    for leak in ("expired", "expire", "signature", "audience", "issuer"):
        assert leak not in body["detail"].lower()


@pytest.mark.asyncio
async def test_jwt_audience_is_checked_when_configured() -> None:
    """配置了 audience 时，受众不符的 token 被拒。"""
    recorder = _Recorder()
    mw = AuthMiddleware(
        recorder,
        settings=_settings(jwt_enabled=True, jwt_audience="aligo-api"),
    )

    status, _, _ = await _drive(
        mw,
        _scope(headers=[(b"authorization", f"Bearer {_token(sub='bob', aud='other')}".encode())]),
    )

    assert status == 401


@pytest.mark.asyncio
async def test_jwt_empty_audience_does_not_check_audience() -> None:
    """audience 为空串时**不校验**受众 —— 带 ``aud`` 的 token 必须能通过。

    ★ 这条用例在写出来的时候**立刻抓到了一个真实的 bug**，值得把经过记下来：

        ``_decode`` 最初只做了「audience 为空就不传 ``audience`` 参数」。
        看起来等价于「不校验」，实际不是 —— PyJWT 的规则是：
        **token 里带 ``aud`` 而调用方没给 ``audience`` 时，抛
        ``InvalidAudienceError``**，而不是跳过校验。

        于是默认配置（``jwt_audience: ""``，字面意思是「不校验受众」）
        会让**所有**带 aud 的 token 被拒。现实中的 IdP（Auth0 / Keycloak /
        阿里云 IDaaS）几乎都会签 aud，所以默认配置下一个 token 都过不了。

        更隐蔽的是：自签的测试 token 如果**不带** aud，一切正常 ——
        问题只在接上真实 IdP 时才暴露，而那时人已经在联调环境里，
        很容易先怀疑「IdP 配错了」而不是「我们的默认值反了」。

    修法是显式 ``options["verify_aud"] = False``，而不是「不传参数」。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings(jwt_enabled=True))

    status, _, _ = await _drive(
        mw,
        _scope(headers=[(b"authorization", f"Bearer {_token(sub='bob', aud='whatever')}".encode())]),
    )

    assert status == 200, (
        "未配置 ALIGO__AUTH__JWT_AUDIENCE 时不应校验受众；"
        "这里失败通常意味着 verify_aud 没有被显式关掉，"
        "而 PyJWT 对「token 有 aud、调用方没给 audience」的默认行为是**抛异常**。"
    )
    assert recorder.header(b"x-user-id") == [b"bob"]


@pytest.mark.asyncio
async def test_jwt_mode_fails_fast_without_pyjwt() -> None:
    """启用 JWT 但没装 PyJWT 时，**装配期**就报错。

    放在第一个请求时失败的话，症状是「所有请求 500」——
    与「少装一个包」相距甚远，而且没装 JWT 的环境通常也跑不到
    「所有请求」这一步（本地开发大多只用 X-User-ID）。
    """
    import builtins

    real_import = builtins.__import__

    def _fail_jwt(name: str, *args: Any, **kwargs: Any) -> Any:
        """拦截 ``import jwt``。"""
        if name == "jwt":
            raise ImportError("模拟未安装 PyJWT")
        return real_import(name, *args, **kwargs)

    with mock.patch.object(builtins, "__import__", _fail_jwt):
        with pytest.raises(RuntimeError, match="未安装 PyJWT"):
            AuthMiddleware(_Recorder(), settings=_settings(jwt_enabled=True))


# ==============================================================================
# 四、匿名模式与指标
# ==============================================================================
@pytest.mark.asyncio
async def test_anonymous_mode_passes_without_identity() -> None:
    """两个开关都关时放行，并标成 anonymous。

    ⚠️ 此时**框架的业务路由仍然会拒绝**（它们自己 ``Depends``，
    缺头返回 422）。这个开关只影响我们这一层 —— 文档里必须说清楚，
    否则运维会以为「关掉鉴权就能匿名压测」，然后在业务接口上撞 422。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings(require_user_header=False))

    status, _, _ = await _drive(mw, _scope())

    assert status == 200
    assert recorder.scope["state"][AUTH_MODE_STATE_KEY] == AUTH_MODE_ANONYMOUS


@pytest.mark.asyncio
async def test_anonymous_mode_still_records_provided_identity() -> None:
    """匿名模式下若带了身份，仍要记下来 —— 限流要用它当键。"""
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings(require_user_header=False))

    await _drive(mw, _scope(headers=[(b"x-user-id", b"alice")]))

    assert recorder.scope["state"][USER_ID_STATE_KEY] == "alice"


@pytest.mark.parametrize(
    ("kwargs", "headers", "reason"),
    [
        ({}, [], FAILURE_MISSING_USER_HEADER),
        ({"jwt_enabled": True}, [], FAILURE_MISSING_BEARER),
        (
            {"jwt_enabled": True},
            [(b"authorization", b"Bearer not-a-token")],
            FAILURE_INVALID_TOKEN,
        ),
        (
            {},
            [(b"x-user-id", RESERVED_ID.encode())],
            FAILURE_RESERVED_IDENTITY,
        ),
    ],
)
@pytest.mark.asyncio
async def test_auth_failures_are_counted_by_reason(
    kwargs: dict[str, Any],
    headers: list[tuple[bytes, bytes]],
    reason: str,
) -> None:
    """鉴权失败必须按**原因**计入指标。

    只有总数是不够的：``missing_bearer`` 涨说明客户端没升级到 JWT，
    ``invalid_token`` 涨说明有人在爆破或密钥轮换出了岔子 ——
    两者的处置完全不同，而它们在「失败总数」里长得一模一样。
    """
    mw = AuthMiddleware(_Recorder(), settings=_settings(**kwargs))
    before = auth_mod.metrics_mod.AUTH_FAILURES_TOTAL.labels(reason=reason)._value.get()

    await _drive(mw, _scope(headers=headers))

    after = auth_mod.metrics_mod.AUTH_FAILURES_TOTAL.labels(reason=reason)._value.get()
    assert after == before + 1, f"原因 {reason} 的计数没有增加"


# ==============================================================================
# 五、保留身份：唯一一条「凭据合法但必须拒绝」的规则
# ==============================================================================
# 背景（完整论证见 src/llm/identity.py 与 auth.py 的模块文档字符串）：
#
#   ``aligo-system`` 是那条**全员只读共享**的模型凭据的属主
#   （src/llm/system_credential.py）。而框架的可见性规则是
#   「属主读自己的凭据是明文，且 editable: true」——
#   于是一个自称 ``aligo-system`` 的请求可以读出运营者的真实 API key，
#   或者 ``DELETE`` 掉那条凭据让所有人的模型当场失效。
#
#   实测过的越权（修复前）：
#       X-User-ID: alice        → GET /credential/ 打码 {type,name}
#       X-User-ID: aligo-system → GET /credential/ data 里有 api_key 明文
#       X-User-ID: aligo-system → DELETE /credential/aligo-system-model → 204
#
# 下面五条覆盖三条通道 + 两个方向（该拦的拦住、不该拦的别误伤）。


@pytest.mark.asyncio
async def test_reserved_identity_is_rejected_in_header_mode() -> None:
    """``X-User-ID: aligo-system`` 必须被拦下，且**不能到达下游**。"""
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings())

    status, _, body = await _drive(
        mw,
        _scope(headers=[(b"x-user-id", RESERVED_ID.encode())]),
    )

    assert status == 403, f"保留身份应当 403，实际 {status}"
    assert recorder.called is False, "被拒绝的请求竟然到达了下游应用"
    # 响应体不回显被拒绝的那个 id（见 _reject_reserved_identity 的说明）。
    assert RESERVED_ID not in json.dumps(body, ensure_ascii=False)


@pytest.mark.asyncio
async def test_reserved_identity_with_surrounding_whitespace_is_rejected() -> None:
    """两侧带空白的写法同样要拦。

    ⚠️ 这一条不是「防御过度」：HTTP 服务器解析请求行时本来就会去掉
    字段值两侧的空白，``X-User-ID: " aligo-system "`` 到框架那里
    就是 ``aligo-system``。若只在中间件里做逐字节比较而不先 ``strip()``，
    这个头就会**穿过**我们的检查、在框架那里命中属主 —— 一条完整的绕过。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings())

    status, _, _ = await _drive(
        mw,
        _scope(headers=[(b"x-user-id", f"  {RESERVED_ID}\t".encode())]),
    )

    assert status == 403
    assert recorder.called is False


@pytest.mark.asyncio
async def test_reserved_identity_is_rejected_in_jwt_mode() -> None:
    """JWT 的 ``sub`` 是保留身份时同样要拦。

    签名合法 ≠ 这个身份可以被用来登录。保留身份根本没有对应的登录主体 ——
    一个带它的合法 token，要么是签发侧配置写错了，要么是有人拿我们
    ``scripts/mint_token.py`` 签的运维 token 想当系统用。两种都不该放行。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings(jwt_enabled=True))

    status, _, _ = await _drive(
        mw,
        _scope(
            headers=[(b"authorization", f"Bearer {_token(sub=RESERVED_ID)}".encode())],
        ),
    )

    assert status == 403
    assert recorder.called is False


@pytest.mark.asyncio
async def test_reserved_identity_is_rejected_when_the_header_is_optional() -> None:
    """``require_user_header=false`` 的匿名通道也必须拦。

    ⚠️ 这条分支**不重写请求头**（带了身份就原样透传给框架），所以它是
    最容易漏掉的一条：只拦前两条的话，「把 ``require_user_header`` 关掉」
    就成了绕过这条规则的开关 —— 而这两个配置项在运维眼里毫不相干。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings(require_user_header=False))

    status, _, _ = await _drive(
        mw,
        _scope(headers=[(b"x-user-id", RESERVED_ID.encode())]),
    )

    assert status == 403
    assert recorder.called is False


@pytest.mark.parametrize(
    "user_id",
    [
        "aligo-system-2",  # 前缀相同
        "aligo-systems",  # 只差一个字母
        "ALIGO-SYSTEM",  # 大小写不同
        "aligo.system",  # 只差一个分隔符
    ],
)
@pytest.mark.asyncio
async def test_a_lookalike_identity_is_not_rejected(user_id: str) -> None:
    """形似但不相等的身份**不是**保留身份，必须照常放行。

    判据必须与框架**逐字节一致**：框架是拿请求头的值去和
    ``record.user_id`` 做 ``==`` 的，所以 ``ALIGO-SYSTEM`` 在它眼里
    本就是另一个用户（读到的凭据照样打码）。把比较写成大小写不敏感
    之类的「更严格」，换来的只是「一个本来安全的写法被拒」——
    而这种行为极难解释，也极难排查。
    """
    recorder = _Recorder()
    mw = AuthMiddleware(recorder, settings=_settings())

    status, _, _ = await _drive(
        mw,
        _scope(headers=[(b"x-user-id", user_id.encode())]),
    )

    assert status == 200, f"{user_id!r} 不该被当成保留身份"
    assert recorder.called is True


@pytest.mark.asyncio
async def test_the_reserved_identity_is_the_system_credential_owner() -> None:
    """钉住两个真相源：中间件拦的那个 id，就是系统凭据的属主。

    ⚠️ 这条断言存在的唯一理由是**防止静默失效**：若哪天有人把
    ``SYSTEM_USER_ID`` 改成别的值（或改了中间件的常量），
    鉴权会照常工作、测试会照常全绿，而那条共享凭据的属主
    又变成了可被客户端声明的身份 —— 越权原样回来，没有任何提示。
    """
    from src.llm.identity import RESERVED_USER_IDS, SYSTEM_USER_ID
    from src.llm.system_credential import SYSTEM_USER_ID as CREDENTIAL_OWNER

    assert SYSTEM_USER_ID == RESERVED_ID
    assert CREDENTIAL_OWNER == RESERVED_ID
    assert RESERVED_ID in RESERVED_USER_IDS
    assert len(RESERVED_USER_IDS) == 1, (
        "保留身份多了一个 —— 请一并确认中间件的三条通道都拦住了它，"
        "以及它确实不该由客户端声明。"
    )
