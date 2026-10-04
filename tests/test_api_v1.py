# -*- coding: utf-8 -*-
"""业务命名空间 ``/api/v1/**`` 的接口测试。

==============================================================================
这一层测的是「装配」，不是「逻辑」
==============================================================================
    中间件各自的逻辑已由 ``test_auth_middleware.py`` /
    ``test_rate_limit_middleware.py`` 用裸 ASGI 精确覆盖。本文件要回答的是
    另一个问题：**把它们装进真实应用之后，行为还对吗？**

    这个区别不是学术性的。中间件的单元测试可以全绿，而真实应用里
    请求依然走不通 —— 顺序写反（限流跑在鉴权前面，于是拿到的是 IP 而不是身份）、
    路由注册在 ``StaticFiles`` 之后（被挂到 ``/`` 的静态目录吃掉）、
    ``include_router`` 的前缀写错……这些**都只在装配层看得见**，
    而且症状全都是「404 或 401」，光看现象分不清是哪一处。

==============================================================================
刻意用 httpx 而不是 TestClient
==============================================================================
    与 ``tests/conftest.py`` 的 ``client`` 夹具一致：它手动进入
    ``app.router.lifespan_context(app)``，因此**真的跑过启动流程**。
    不跑 lifespan 的客户端能启动、``/healthz`` 也返回 200，
    但任何业务路由都会因为 ``app.state.xxx`` 不存在而 500 ——
    那会让「服务是否真的能起来」这件事完全测不到。
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from src.server.middleware import AUTH_MODE_STATE_KEY

#: 测试用的身份。取值本身无意义，只要稳定即可。
USER_ALICE = "alice"


# ==============================================================================
# 一、路由树本身
# ==============================================================================
def _route_paths(app: FastAPI) -> set[str]:
    """列出应用里**真实可用**的全部路由路径。

    ⚠️ 为什么用 ``app.openapi()["paths"]`` 而不是遍历 ``app.routes``：
        这一版 FastAPI 的 ``include_router`` **不展平**子路由，而是在
        ``app.routes`` 里放一个 ``_IncludedRouter`` 对象。那个对象**没有
        ``.path``**，于是 ``{getattr(r, "path", "") for r in app.routes}``
        会得到一堆空串加几条框架默认路由 —— 看着像「路由全丢了」，
        实际请求一切正常。（本测试的第一版就是这么写的，断言全红。）

        OpenAPI schema 是**应用自己算出来的**真实路由表，83 条路径一条不少，
        还顺带覆盖了「路由存在但没进 schema」这种更细的问题。
        （83 是实测值，不是约定值 —— 框架升级加了路由，这个数字就会变；
        它写在这里只是为了让下面「只筛 /api/v1」的那一步有据可依。）

    Args:
        app (`FastAPI`): 目标应用。

    Returns:
        `set[str]`: 路径模板集合，如 ``{"/api/v1/health", ...}``。
    """
    return set(app.openapi()["paths"])


def test_api_v1_routes_are_registered_under_the_prefix(app: FastAPI) -> None:
    """``/api/v1`` 前缀在**最终路径**上生效。

    ⚠️ 断言的是最终路径而不是 ``api_v1_router.prefix``：
    子路由若用了绝对路径（以 ``/`` 开头的字符串），会**覆盖**前缀 ——
    一个静默的失效方式：路由照样注册，只是挂到了根上，
    于是 ``/api/v1/me`` 变成 ``/me``。断言前缀字段是看不出来的。
    """
    # 只看业务命名空间：整个应用的 schema 有 83 条路径（含框架的 77 条），
    # 断言全集会把「框架升级加了一条路由」也报成我们的失败。
    business = {p for p in _route_paths(app) if p.startswith("/api/v1")}

    # ⚠️ 这是一个**恰好等于**的断言：新增一条业务路由就必须同步改这里。
    # 新增时不要图省事改成 ``>=``（子集判断）—— 那样「某个子路由用了绝对路径
    # 把前缀覆盖掉、于是少挂了一条」这种失效就再也测不出来了，而那正是本用例
    # 存在的首要理由（见上面 _route_paths 的说明）。
    assert business == {
        "/api/v1/health",
        "/api/v1/me",
        "/api/v1/default-model",
        "/api/v1/memory/profile",
        "/api/v1/memory/notes",
        "/api/v1/sessions/{session_id}/chains",
    }, (
        f"业务路由集合与预期不符：{sorted(business)}。\n"
        f"多出来的路径通常意味着某个子路由用了绝对路径，把前缀覆盖掉了。\n"
        f"少掉的路径则可能是 include_router 漏挂了。"
    )


def test_business_routes_do_not_shadow_framework_routes(app: FastAPI) -> None:
    """业务路由与框架路由**并存**，没有互相遮蔽。

    框架把 ``/chat/`` ``/sessions/**`` ``/agent/`` 注册在根路径上
    （``include_router`` 未加 prefix），那是 P2 流式验收要用的通道，
    我们照用不改。``/api/v1`` 是**并行**的命名空间。

    这条用例防的是一种具体的破坏：为了「统一前缀」而给框架的
    ``include_router`` 也加上 ``/api/v1``。那样 ``/chat/`` 会消失、
    前端与冒烟脚本全部 404，而改动本身看起来像是**整理**。
    """
    paths = _route_paths(app)

    for framework_path in ("/chat/", "/sessions/", "/agent/"):
        assert framework_path in paths, (
            f"框架的 {framework_path} 不见了。\n"
            f"它必须在根路径上 —— 框架的 include_router 不加前缀是既定契约，"
            f"给它加前缀会同时打断官方前端与 scripts/smoke.py。"
        )

    assert "/api/v1/health" in paths
    assert "/api/v1/me" in paths
    assert "/api/v1/sessions/{session_id}/chains" in paths


# ==============================================================================
# 二、公开路径
# ==============================================================================
@pytest.mark.asyncio
async def test_api_v1_health_is_public(client: AsyncClient) -> None:
    """``/api/v1/health`` 免鉴权即返回 200。

    这是「探针不能被自己的鉴权拦住」这条性质的回归测试。
    一旦它开始返回 401，容器编排会判定实例不健康并反复重启 ——
    一个自指的故障，而根因是一个白名单条目被误删。
    """
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_api_v1_health_does_not_leak_configuration(client: AsyncClient) -> None:
    """公开探针的响应体**不得**包含配置信息。

    它是一个任何人都能打的公开端点。把 ``db.url`` 之类的信息放进来
    等于给扫描器送情报 —— 而这类字段往往是「顺手加上去方便排障」的，
    加的时候没人会想到它落在公开路径上。
    """
    body = (await client.get("/api/v1/health")).text

    for leak in ("password", "secret", "postgres", "redis", "@", "key"):
        assert leak not in body.lower(), f"公开探针的响应里出现了 {leak!r}：{body}"


@pytest.mark.asyncio
async def test_health_carries_trace_id_header(client: AsyncClient) -> None:
    """每个响应都带 ``X-Trace-ID``（由 TraceContext 中间件补上）。"""
    response = await client.get("/api/v1/health")

    assert response.headers.get("x-trace-id")


@pytest.mark.asyncio
async def test_upstream_trace_id_is_reused(client: AsyncClient) -> None:
    """上游带了 ``X-Trace-ID`` 时沿用，不新生成。

    跨服务排查时，一个请求在两个服务里必须是**同一个** id；
    各自生成的话，日志里就是两条毫不相干的记录。
    """
    response = await client.get(
        "/api/v1/health",
        headers={"X-Trace-ID": "trace-from-upstream"},
    )

    assert response.headers["x-trace-id"] == "trace-from-upstream"


# ==============================================================================
# 三、鉴权生效
# ==============================================================================
@pytest.mark.asyncio
async def test_me_without_identity_is_401(client: AsyncClient) -> None:
    """缺身份访问 ``/api/v1/me`` 返回 **401**（我们的中间件，不是框架的 422）。"""
    response = await client.get("/api/v1/me")

    assert response.status_code == 401, (
        f"期望 401，实际 {response.status_code}。"
        f"返回 422 说明请求穿过了 Auth 中间件、被框架的依赖拦下了 —— "
        f"即鉴权中间件没有生效。"
    )
    assert "www-authenticate" in {k.lower() for k in response.headers}


@pytest.mark.asyncio
async def test_me_returns_resolved_identity(client: AsyncClient) -> None:
    """带身份访问 ``/api/v1/me`` 返回解析出的身份与鉴权模式。

    ★ 这是鉴权链路的**端到端证明**：``/api/v1/me`` 用的是**框架自己的**
    ``Depends(get_current_user_id)``（那 85 个既有依赖点用的同一个），
    所以它返回正确的 ``user_id`` 就意味着「我们注入的请求头
    与框架的期望完全兼容」—— 而不只是「我们的中间件自己觉得对」。
    """
    response = await client.get("/api/v1/me", headers={"X-User-ID": USER_ALICE})

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == USER_ALICE
    assert body["auth_mode"] == "header"
    # consistent：框架依赖取到的身份 == 中间件写进 scope 的身份。
    # 为假说明中间层有人改了请求头 —— 那会导致「某些接口按 A 用户过滤、
    # 另一些按 B 用户」，是极难定位的一类不一致。
    assert body["consistent"] is True


@pytest.mark.asyncio
async def test_me_reports_trace_id(client: AsyncClient) -> None:
    """``/api/v1/me`` 把 trace_id 也回给调用方，方便直接拿去查日志。"""
    response = await client.get(
        "/api/v1/me",
        headers={"X-User-ID": USER_ALICE, "X-Trace-ID": "trace-me-123"},
    )

    assert response.json()["trace_id"] == "trace-me-123"
    # 响应头与响应体必须是同一个 id —— 两者不一致时，排障的人
    # 会拿着响应体里的 id 去搜日志，然后搜不到。
    assert response.headers["x-trace-id"] == "trace-me-123"


@pytest.mark.asyncio
async def test_unknown_api_v1_path_requires_auth_before_404(client: AsyncClient) -> None:
    """未定义路径在**无身份**时返回 401 而不是 404。

    ⚠️ 这是刻意的，且值得写下来，因为它看起来像个 bug：

        鉴权中间件在**路由匹配之前**执行（它是 ASGI 中间件，包在整个
        FastAPI 应用外面）。因此一个不存在的 ``/api/v1/nope`` 也会先被
        鉴权拦下，返回 401 —— 只有带上身份之后才暴露 404。

        另一个可选设计是「先路由、后鉴权」，那样未定义路径返回 404。
        两者都自洽。这里选前者，理由是：**未认证的调用方不该能通过
        404/401 的差异来探测哪些接口存在**（接口枚举）。
        代价是「路径拼错」与「没带身份」在无身份时长得一样 ——
        但这个代价是给调用方的，而枚举风险是给攻击者的。
    """
    response = await client.get("/api/v1/definitely-not-a-route")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_unknown_api_v1_path_is_404_with_identity(client: AsyncClient) -> None:
    """带上身份后，未定义路径正常返回 404。"""
    response = await client.get(
        "/api/v1/definitely-not-a-route",
        headers={"X-User-ID": USER_ALICE},
    )

    assert response.status_code == 404


# ==============================================================================
# 四、中间件链在真实应用里的顺序
# ==============================================================================
def test_middleware_stack_order(app: FastAPI) -> None:
    """★ 完整中间件链的顺序（这是装配层最容易写反的一处）。

    ``add_middleware`` 内部是 ``insert(0, ...)``，因此
    ``app.user_middleware`` 的**下标 0 是最外层**（最先看到请求）。

    目标链（由外到内）::

        TraceContext → HttpMetrics → Auth → 降级凭据播种 → RateLimit → 应用

    三条最容易搞错、也最值得重复的理由：

        · **HttpMetrics 在 Auth 之外** —— 放里面的话，被 401/429 拒绝的请求
          不进指标，而鉴权失败率与限流率恰恰是最需要看板的两条曲线。
          更糟的是它看起来完全正常：有流量、有延迟，只是永远看不到被拒的那部分。
        · **降级凭据播种在 Auth 之内、RateLimit 之外** —— 它要读 Auth 写进
          ``scope["state"]`` 的身份才知道该给谁播种；而它自己会写库，
          应当排在被限流拦下的请求**之后**（一个注定 429 的请求不该产生写）。
          放到 Auth 之外是本层最阴的失效：读不到身份 ⇒ 永远不播种，
          而它是**静默**的（没日志、没异常），症状与「零密钥下发送按钮是灰的」
          一模一样，根本分不出是没播种还是播种了没用。
        · **RateLimit 在 Auth 之内** —— 它要读 Auth 写进 ``scope["state"]``
          的身份当限流键。顺序反了不会报错，只会让所有请求退化成按 IP 限流，
          多人共用出口 IP 时互相误伤。
    """
    names = [m.cls.__name__ for m in app.user_middleware]
    expected = [
        "TraceContextMiddleware",
        "HttpMetricsMiddleware",
        "AuthMiddleware",
        "MockCredentialSeedMiddleware",
        "RateLimitMiddleware",
    ]

    assert names == expected, (
        f"中间件链顺序不对：{names}\n"
        f"期望 {expected}（下标 0 为最外层）。\n"
        f"注意 add_middleware 是 insert(0)，所以 app.py 里的**书写顺序是反的**。"
    )


def test_auth_and_ratelimit_receive_settings(app: FastAPI) -> None:
    """Auth 与 RateLimit 必须拿到 settings。

    漏传的症状是**装配期 TypeError**（构造函数要求关键字参数），
    看起来很明显 —— 但只有在有人把 ``Middleware(AuthMiddleware)``
    写成不带 kwargs 时才会发生，而那正是「照抄上面两行」时最容易发生的错误。
    """
    by_name = {m.cls.__name__: m for m in app.user_middleware}

    assert "settings" in by_name["AuthMiddleware"].kwargs
    assert "settings" in by_name["RateLimitMiddleware"].kwargs


def test_seed_middleware_shares_the_storage_instance(app: FastAPI) -> None:
    """播种中间件拿到的 storage 必须与 ``app.state.storage`` 是**同一个实例**。

    ``create_app`` 在装配期就把传入的 storage 挂到了 ``app.state.storage``
    （``app/_app.py``），框架的 lifespan 随后从那里读它 ——
    因此这两者本来**就该是同一个对象**，断言 ``is`` 是在钉住
    「我们传给 create_app 的」与「我们传给中间件的」没有分家。

    分家的症状是「刚播种的凭据下一秒就消失」：凭据写进了 A，
    而对话链路从 B 里读。两者都是合法连接、都是同一张表，
    只有在用不同数据库（或不同 schema）时才暴露 ——
    本地跑永远看不出来，因为默认配置下 A 与 B 恰好指向同一个库。
    """
    by_name = {m.cls.__name__: m for m in app.user_middleware}
    seeded = by_name["MockCredentialSeedMiddleware"].kwargs

    assert "settings" in seeded and "storage" in seeded, (
        "播种中间件少了构造参数 —— 装配期应当直接 TypeError；"
        "若它被放宽成可选参数，这里就会变成「静默不播种」。"
    )
    assert seeded["storage"] is app.state.storage, (
        "播种中间件与 create_app 用的不是同一个 storage 实例 —— "
        "播种写进去的凭据，对话链路读不到。"
    )


# ==============================================================================
# 五、业务库引擎挂上了
# ==============================================================================
@pytest.mark.asyncio
async def test_business_engine_is_exposed_on_app_state(
    client: AsyncClient,
    app: FastAPI,
) -> None:
    """lifespan 必须把业务库引擎挂到 ``app.state``。

    请求 ``/api/v1/health`` 只是为了确保 lifespan 已经跑过 ——
    引擎是在 lifespan 里构造的，而 ``app`` 夹具产出的对象在进入
    lifespan 之前并没有它。
    """
    from src.server.app import BUSINESS_ENGINE_ATTR

    await client.get("/api/v1/health")

    engine: Any = getattr(app.state, BUSINESS_ENGINE_ATTR, None)
    assert engine is not None, (
        f"app.state.{BUSINESS_ENGINE_ATTR} 不存在 —— "
        f"业务库引擎没有在 lifespan 里被构造。"
    )
    assert engine.dialect.name == "sqlite"
