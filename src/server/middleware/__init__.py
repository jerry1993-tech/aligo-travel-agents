# -*- coding: utf-8 -*-
"""ASGI 中间件包：请求从进入到返回所经过的每一层横切逻辑。

当前包含::

    _asgi.py          —                       纯 ASGI 共用工具（头读写 / 直接构造响应）
    http_trace.py     TraceContextMiddleware  trace_id 分配/传播 + 访问日志
    http_metrics.py   HttpMetricsMiddleware   Prometheus 请求指标
    auth.py           AuthMiddleware          鉴权：凭据 → 可信 X-User-ID
    rate_limit.py     RateLimitMiddleware     按身份的令牌桶限流

------------------------------------------------------------------------------
⚠️ 中间件的**执行顺序**（改这里前必读）
------------------------------------------------------------------------------
    Starlette 的 ``add_middleware`` 内部是 ``user_middleware.insert(0, ...)``，
    也就是**后加的在外层**。而 ``agentscope.app.create_app`` 是按我们传入的
    ``extra_middlewares`` 列表**顺序**逐个 ``add_middleware`` 的
    （``agentscope/app/_app.py:428-429``），于是：

        传入 ``[A, B, C]``  ⇒  实际执行顺序 ``C → B → A → 应用``

    **列表顺序与执行顺序相反**。这一点必须写下来：它违反了绝大多数人的直觉，
    而搞错它的症状是「鉴权中间件跑在限流后面」之类只有压测才暴露的问题。

    本项目当前的目标链（由外到内）::

        TraceContext → HttpMetrics → Auth → RateLimit → 应用

    因此 ``src/server/app.py`` 里的**书写顺序**是反过来的::

        [RateLimit, Auth, HttpMetrics, TraceContext]

    ⚠️ 两点容易搞错的地方：

    1. **HttpMetrics 在 Auth 之外**（即先于 Auth 执行）。这是刻意的：
       放里面的话，被 401/429 拒绝的请求**不会进入指标** ——
       而鉴权失败率与限流率恰恰是最需要看板的两条曲线。指标应当统计
       「到达了服务器的全部请求」，而不只是「通过了鉴权的那些」。

    2. **TraceContext 必须在最外层**：它要在鉴权失败（401/429）时也已经分配好
       trace_id，否则「谁在打我的鉴权接口」这类请求在日志里没有关联标识，
       而这恰恰是安全排查最需要的一类记录。

    ⚠️ 注册时机：Starlette 在**第一次请求之后**会缓存 middleware_stack，
    此后 ``add_middleware`` 抛 ``Cannot add middleware after an application
    has started``（``starlette/applications.py``）。新增中间件必须在装配期
    （``create_root_app`` 内）完成，测试里若重复进出 lifespan 不受影响，
    但**不要**在请求处理过程中动态注册。

------------------------------------------------------------------------------
⚠️ 本包内一律使用「纯 ASGI」写法，禁止 ``BaseHTTPMiddleware``
------------------------------------------------------------------------------
    见 :mod:`src.server.middleware.http_trace` 的模块文档字符串：
    ``BaseHTTPMiddleware`` 会把下游应用放进 anyio 任务组，额外引入一层
    「发送/接收」队列，**会缓冲流式响应** —— 对 ``/sessions/{id}/stream``
    这条 SSE 通道是致命的（逐块刷新会变成攒批）。

    这类 bug 的可怕之处在于：**功能测试全过**。非流式端点完全正常，
    只有真正走 SSE 时才表现为「回复半天不出来、然后一次性全出来」，
    而那时已经很难把它与「中间件写法」联系起来。
"""

from ._asgi import Receive, Scope, Send
from .auth import (
    AUTH_MODE_STATE_KEY,
    USER_ID_STATE_KEY,
    AuthMiddleware,
)
from .http_metrics import UNMATCHED_ROUTE, HttpMetricsMiddleware
from .mock_credential import MockCredentialSeedMiddleware
from .http_trace import MAX_TRACE_ID_LENGTH, TRACE_HEADER, TraceContextMiddleware
from .rate_limit import RateLimitMiddleware

__all__ = [
    "AUTH_MODE_STATE_KEY",
    "MAX_TRACE_ID_LENGTH",
    "TRACE_HEADER",
    "UNMATCHED_ROUTE",
    "USER_ID_STATE_KEY",
    "AuthMiddleware",
    "HttpMetricsMiddleware",
    "MockCredentialSeedMiddleware",
    "RateLimitMiddleware",
    "Receive",
    "Scope",
    "Send",
    "TraceContextMiddleware",
]
