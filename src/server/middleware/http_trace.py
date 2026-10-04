# -*- coding: utf-8 -*-
"""请求链路中间件：为每个请求分配/继承 ``trace_id``，并把访问日志打出来。

文件职责：
    · **分配** trace_id：请求进来时若带了 ``X-Trace-ID`` 就沿用，否则新生成一个；
    · **传播**：放进 :mod:`src.observability.context` 的 ContextVar，
      使这次请求产生的**所有**日志都自动带上它；
    · **回传**：在响应头里返回 ``X-Trace-ID``，让调用方（前端 / smoke 脚本 /
      上游网关）能拿着它去日志里反查这一条请求；
    · **记一条访问日志**：方法、路径、状态码、耗时、trace_id。

上下游依赖：
    - 上游：由 ``src/server/app.py::create_root_app`` 注册（见该文件里的中间件
      列表），位于整个中间件链的**最外层**（最先看到请求，最后看到响应）。
    - 下游：``src/observability/logging.py`` 的过滤器读取 ContextVar 的值。

      ⚠️ 本行早先写的是「通过 ``create_app(extra_middlewares=[...])`` 注册」——
      **与代码不符**：``create_root_app`` 从未传过 ``extra_middlewares``，
      它是在拿到 ``create_app`` 的返回值之后直接 ``app.add_middleware`` 的。
      两处 docstring（本文件与 ``middleware/__init__.py``）都曾这么写，
      照抄的后果是**重复注册一层 TraceContextMiddleware**，
      同时把 Auth/RateLimit 挤到 HttpMetrics 的内侧。已按代码更正。

------------------------------------------------------------------------------
纯 ASGI 中间件，而不是 ``BaseHTTPMiddleware``
------------------------------------------------------------------------------
    Starlette 的 ``BaseHTTPMiddleware`` 写起来更短，但它有一个众所周知的代价：
    它把下游应用放进 ``anyio`` 的任务组里跑，额外引入一层「发送/接收」队列。
    后果是：
        · **SSE / 流式响应会被缓冲**，逐块刷新变成攒批 —— 对本项目「实时思考链」
          这个核心卖点来说是不可接受的（P2 的 /sessions/{id}/stream 正是流式）；
        · 它会在异常时把 traceback 包装成 ``ExceptionGroup``，掩盖真实异常类型。
    因此本项目一律使用**纯 ASGI 中间件**：多写十几行，换来流式语义不被破坏。

------------------------------------------------------------------------------
为什么用 ContextVar 而不是把 trace_id 塞进 request.state
------------------------------------------------------------------------------
    业务代码（``src/agents/`` / ``src/tools/``）深处的一行 ``logger.info`` 拿不到
    ``request`` 对象。而 ContextVar 是「隐式但协程安全」的传递方式 ——
    这正是日志上下文该有的形态：不需要每层函数都多一个参数。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ...observability.context import new_trace_id, reset_trace_id, set_trace_id
from ._asgi import Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

#: 传递 trace id 的请求/响应头名。
#: 用 ``X-Trace-ID`` 而不是 W3C 的 ``traceparent``：后者是一套完整的
#: 分布式追踪协议（含采样位、span id 等），本项目在 P1 只需要一个可检索的
#: 关联标识；引入 traceparent 而不完整实现它，只会制造出「看起来支持 W3C 追踪、
#: 实际上下游都不认」的假象。P5 接入 OTel 后由 OTel 自己管传播，两者不冲突。
TRACE_HEADER = "x-trace-id"

#: 上游传来的 trace id 的长度上限。
#: **必须限制**：这个值会被原样写进日志与响应头。不设限时，一个恶意的超长
#: 请求头（比如 1 MB）会让每一条日志都被撑爆，甚至把日志系统打挂 ——
#: 这是一条典型的「输入未校验」导致的可用性问题。
MAX_TRACE_ID_LENGTH = 128

def _pick_trace_id(scope: Scope) -> str:
    """决定本次请求用哪个 trace id。

    优先沿用上游传入的值（这样跨服务的一整条链路可以共用一个 id），
    但必须经过长度与字符校验 —— 上游的值是不可信输入。

    Args:
        scope (`Scope`): ASGI scope。

    Returns:
        `str`: 沿用的或新生成的 trace id。
    """
    for raw_name, raw_value in scope.get("headers") or []:
        # ASGI 规范规定 header 名是小写的 bytes。
        if raw_name == b"x-trace-id":
            candidate = raw_value.decode("latin-1").strip()
            # 只接受「长度合理」且「全部是可打印 ASCII」的值。
            # 不接受换行等控制字符：它们能伪造出额外的日志行，
            # 也就是经典的日志注入（log injection）。
            if (
                candidate
                and len(candidate) <= MAX_TRACE_ID_LENGTH
                and candidate.isascii()
                and candidate.isprintable()
            ):
                return candidate
            logger.debug("忽略不可用的上游 X-Trace-ID 请求头。")
            break
    return new_trace_id()


class TraceContextMiddleware:
    """纯 ASGI 中间件：绑定 trace_id + 输出访问日志。"""

    def __init__(self, app: Any) -> None:
        """保存下游 ASGI 应用。

        Args:
            app (`Any`): 下游 ASGI 应用（框架 Starlette/FastAPI 的约定签名）。
        """
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """处理一次 ASGI 调用。

        只对 ``http`` 类型的 scope 做处理；``lifespan`` 与 ``websocket``
        原样透传 —— 给 lifespan 分配 trace_id 没有意义（它不是请求），
        而 websocket 的 header 结构与生命周期回调都不同，贸然处理会引入
        与 P2 的 SSE 通道不一致的语义。

        Args:
            scope (`Scope`): ASGI scope。
            receive (`Receive`): 接收可调用对象。
            send (`Send`): 发送可调用对象。
        """
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        trace_id = _pick_trace_id(scope)
        token = set_trace_id(trace_id)
        started = time.perf_counter()
        status_holder = {"status": 500}

        async def send_wrapper(message: Message) -> None:
            """在响应开始时补上 X-Trace-ID 头，并记下状态码。"""
            if message.get("type") == "http.response.start":
                status_holder["status"] = int(message.get("status", 500))
                headers = list(message.get("headers") or [])
                headers.append(
                    (TRACE_HEADER.encode("ascii"), trace_id.encode("ascii")),
                )
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            duration_ms = (time.perf_counter() - started) * 1000
            status = status_holder["status"]
            # 5xx 用 ERROR、4xx 用 WARNING：把「客户端用错了」和「我们坏了」
            # 在日志级别上分开，日志告警规则才能只盯后者。
            if status >= 500:
                log = logger.error
            elif status >= 400:
                log = logger.warning
            else:
                log = logger.info
            log(
                "%s %s → %d（%.1fms）",
                scope.get("method", "-"),
                scope.get("path", "-"),
                status,
                duration_ms,
            )
            # ⚠️ reset 必须放在 finally 里，且必须在**日志打完之后** ——
            # 放在前面的话，这条访问日志自己就没有 trace_id 了。
            reset_trace_id(token)


__all__ = [
    "MAX_TRACE_ID_LENGTH",
    "TRACE_HEADER",
    "TraceContextMiddleware",
]
