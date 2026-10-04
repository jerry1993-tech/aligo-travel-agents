# -*- coding: utf-8 -*-
"""请求指标中间件：把每个 HTTP 请求记进 Prometheus。

文件职责：
    在请求结束时记录 ``aligo_http_requests_total`` 与
    ``aligo_http_request_duration_seconds``，并维护 ``aligo_http_in_flight_requests``。

上下游依赖：
    - 上游：由 ``src/server/app.py`` 注册在中间件链中。
    - 下游：写在 :mod:`src.observability.metrics` 里的指标，由
      ``/metrics`` 渲染、被 ``scripts/prometheus/prometheus.yml`` 抓取。

------------------------------------------------------------------------------
核心难点：路由**模板** vs 真实路径
------------------------------------------------------------------------------
    必须用 ``/api/v1/orders/{order_id}`` 这样的模板，**绝不能**用
    ``/api/v1/orders/8f3a-...`` 这样的真实路径。

    原因不是「好看」，而是 Prometheus 的**基数（cardinality）**：
    每一条不同的标签组合都是一条独立时序，全部驻留在内存里。
    用真实路径时，一次 1 万用户的压测就会产生 1 万条时序 ——
    轻则 Prometheus 内存暴涨、查询变慢，重则直接 OOM 崩掉抓取端。
    这是一个「本地测不出来、上线才炸」的问题，因为本地压测的 id 往往只有几个。

    路由模板从 ``scope["route"]`` 取（Starlette 在路由匹配成功后写入，
    见 ``starlette/routing.py:695``）。取不到时回落到固定字符串 ``<unmatched>``，
    而不是回落到路径本身 —— 那等于把上面的问题原样放回来。
    回落到固定串还能顺带暴露一件事：如果 ``<unmatched>`` 的计数很高，
    说明有大量请求打到了不存在的路径（被扫端口 / 前端路由配错）。

------------------------------------------------------------------------------
仍未匹配的请求：404 也要被记录
------------------------------------------------------------------------------
    404（没匹配到任何路由）时 ``scope["route"]`` 不存在，但这恰恰是
    **最需要被记录**的一类流量。因此本中间件不跳过它们，而是统计到
    ``<unmatched>`` 上。
"""

from __future__ import annotations

import time
from typing import Any

from ...observability import metrics as metrics_mod
from ._asgi import Message, Receive, Scope, Send

#: 未能匹配到路由时使用的占位标签值。见模块文档字符串。
UNMATCHED_ROUTE = "<unmatched>"


def _route_template(scope: Scope) -> str:
    """取出本次请求命中的路由模板。

    Args:
        scope (`Scope`): ASGI scope（此时路由匹配已完成）。

    Returns:
        `str`: 路由模板；未匹配到路由时为 :data:`UNMATCHED_ROUTE`。
    """
    route = scope.get("route")
    # ``scope["route"]`` 在挂载子应用的情况下是 ``functools.partial``
    # （``starlette/routing.py:707``），它没有 ``.path``。用 getattr 兜底，
    # 避免为了统计指标而抛 AttributeError —— 那会让指标中间件自己变成故障源。
    path = getattr(route, "path", None)
    return path if isinstance(path, str) and path else UNMATCHED_ROUTE


class HttpMetricsMiddleware:
    """纯 ASGI 中间件：统计请求数与耗时。

    与 :class:`~src.server.middleware.http_trace.TraceContextMiddleware` 一样
    使用纯 ASGI 而非 ``BaseHTTPMiddleware`` —— 后者会缓冲流式响应，
    对本项目的 SSE 通道是致命的（原因见 http_trace.py 的模块文档字符串）。
    """

    def __init__(self, app: Any) -> None:
        """保存下游 ASGI 应用。

        Args:
            app (`Any`): 下游 ASGI 应用。
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """处理一次 ASGI 调用。

        Args:
            scope (`Scope`): ASGI scope。
            receive (`Receive`): 接收可调用对象。
            send (`Send`): 发送可调用对象。
        """
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        method = scope.get("method", "-")
        # in_flight 在**请求开始时**递增、结束时递减，因此它反映的是「此刻
        # 正在处理的请求数」。用它减去 QPS，可以区分「流量涨了」与「变慢了」——
        # 两者都需要扩容，但前者要加水、后者要先查慢查询，处置完全不同。
        metrics_mod.HTTP_IN_FLIGHT.inc()
        started = time.perf_counter()
        status_holder = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            """记下响应状态码后原样转发。"""
            if message.get("type") == "http.response.start":
                status_holder["status"] = int(message.get("status", 500))
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            # 无论成功、异常还是客户端断开，都必须记录并递减 in_flight ——
            # 少减一次会让 in_flight 单调上涨，最终在面板上表现为
            # 「永远有 N 个请求卡住」的假故障，而实际系统是健康的。
            metrics_mod.HTTP_IN_FLIGHT.dec()
            metrics_mod.observe_http_request(
                method=method,
                route=_route_template(scope),
                status=status_holder["status"],
                duration_seconds=time.perf_counter() - started,
            )


__all__ = [
    "UNMATCHED_ROUTE",
    "HttpMetricsMiddleware",
]
