# -*- coding: utf-8 -*-
"""请求上下文：把 ``trace_id`` 沿着一次请求的调用链带下去。

文件职责：
    提供一个基于 :class:`contextvars.ContextVar` 的 ``trace_id`` 读写接口。
    任何需要打日志的地方都可以随时取到「当前请求的 trace_id」，而不必把它
    作为参数一路显式传递。

上下游依赖：
    - 上游：``src/server/middleware/http_trace.py`` 在请求进入时**设置**它。
    - 下游：``src/observability/logging.py`` 的日志过滤器**读取**它，
      把 trace_id 注入每一条日志记录。

------------------------------------------------------------------------------
为什么用 ContextVar，而不是 threading.local 或全局变量
------------------------------------------------------------------------------
    · **全局变量**：所有并发请求共用一个值，后到的请求会覆盖先到的 ——
      日志里的 trace_id 会指向别人的请求。这比没有 trace_id 更糟，
      因为它会把人引向一个**错误的**调用链。
    · **threading.local**：在 asyncio 里无效。事件循环会在同一个线程里交替
      执行成百上千个协程，threading.local 的值会在 await 点之间互相串。
    · **ContextVar**：为每个协程任务维护一份独立的值，且能被
      ``asyncio.create_task`` 继承（子任务看得到父任务的 trace_id）。
      这正是「一次请求派生的所有后台任务共享同一个 trace_id」所需要的语义。

------------------------------------------------------------------------------
为什么设置与重置必须成对
------------------------------------------------------------------------------
    :func:`set_trace_id` 返回一个 token，必须交给 :func:`reset_trace_id`。
    ASGI 中间件的处理函数可能在**同一个事件循环任务**里被复用于多个请求
    （取决于服务器实现），只 set 不 reset 会让上一个请求的 trace_id
    泄漏到下一个请求上 —— 而且只在特定的并发时序下才出现，极难复现。
"""

from __future__ import annotations

from contextvars import ContextVar, Token

import shortuuid

#: 当前请求的 trace id。默认空串表示「不在任何请求上下文里」
#: （例如启动阶段、后台调度任务、迁移脚本）。
#:
#: ⚠️ 用 ``default=""`` 而不是 ``default=None``：日志过滤器里要做字符串拼接，
#:    统一成 str 可以省掉一次类型判断，也避免了「None 被格式化成了字符串 'None'
#:    出现在日志里」这种低级但常见的失误。
_trace_id: ContextVar[str] = ContextVar("aligo_trace_id", default="")


def new_trace_id() -> str:
    """生成一个新的 trace id。

    Returns:
        `str`: 形如 ``tr-8f2a1c...`` 的短 id。前缀 ``tr-`` 的价值是**可检索**：
        日志里一眼能把 trace_id 与 session_id、user_id 之类的标识区分开。
    """
    return f"tr-{shortuuid.uuid()}"


def set_trace_id(trace_id: str) -> Token[str]:
    """把 ``trace_id`` 设为当前上下文的取值。

    Args:
        trace_id (`str`): 要设置的 id（通常来自 :func:`new_trace_id`
            或上游传入的 ``X-Trace-ID`` 请求头）。

    Returns:
        `Token[str]`: 重置令牌，必须原样交给 :func:`reset_trace_id`。
    """
    return _trace_id.set(trace_id)


def reset_trace_id(token: Token[str]) -> None:
    """把 ``trace_id`` 恢复成 :func:`set_trace_id` 之前的值。

    Args:
        token (`Token[str]`): :func:`set_trace_id` 返回的令牌。
    """
    _trace_id.reset(token)


def get_trace_id() -> str:
    """取当前上下文的 ``trace_id``。

    Returns:
        `str`: 当前 trace id；不在请求上下文里时为空串。
    """
    return _trace_id.get()


__all__ = [
    "get_trace_id",
    "new_trace_id",
    "reset_trace_id",
    "set_trace_id",
]
