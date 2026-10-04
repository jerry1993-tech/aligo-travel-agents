# -*- coding: utf-8 -*-
"""纯 ASGI 中间件的共用底层工具：类型别名、请求头读写、直接构造响应。

文件职责：
    把「在 ASGI 这一层怎么改请求头」「怎么在中间件里直接返回一个 JSON 错误」
    这两件小事收敛到一处。它们看起来不值得单独成文件，但每一个都有
    **一个不写下来就会踩的坑**，而坑的代价是安全漏洞或难查的时序问题：

    · :func:`set_header` —— 必须**先删光同名头再追加**；
    · :func:`header_value` —— 取**首个**匹配项（与 Starlette 一致）；
    · :func:`send_json` —— 中间件里没有 ``Response`` 对象可用，
      必须手写 ``http.response.start`` + ``http.response.body`` 两帧。

上下游依赖：
    - 上游：无（只依赖标准库）。
    - 下游：``auth.py`` / ``rate_limit.py`` / ``http_trace.py`` / ``http_metrics.py``。

------------------------------------------------------------------------------
为什么必须是纯 ASGI（而不是 BaseHTTPMiddleware）
------------------------------------------------------------------------------
    ``BaseHTTPMiddleware`` 把下游应用放进 anyio 的任务组，额外引入一层
    「发送/接收」队列，**会缓冲流式响应** —— 对本项目实时思考链（SSE）这个
    核心卖点是致命的。详见 :mod:`src.server.middleware.http_trace` 的模块文档字符串。
    本模块里的工具正是为了「不借助 BaseHTTPMiddleware 也能写中间件」而存在。
"""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Iterable

#: ASGI 三件套的类型别名。四个中间件模块共用这一份定义 ——
#: 各写一份的后果不是「多几行」，而是它们迟早会漂移（比如有人给 Send 加了
#: 可选参数），而类型别名漂移是静态检查抓不到的。
Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]

#: 中间件直接构造响应时的响应头。
_JSON_CONTENT_TYPE = b"application/json; charset=utf-8"


def header_value(scope: Scope, name: bytes) -> str | None:
    """读取请求头，返回**首个**匹配项的值。

    ⚠️ 「首个」这个语义是刻意对齐 Starlette 的：它的
    ``Headers.__getitem__`` / ``get`` 都只认第一条匹配的头
    （``starlette/datastructures.py``）。FastAPI 的 ``Header(...)`` 依赖
    最终也走这条路径。所以要「覆盖」一个头时，只追加是不够的 ——
    见 :func:`set_header`。

    Args:
        scope (`Scope`): ASGI scope。
        name (`bytes`): **小写**的请求头名（ASGI 规范要求小写 bytes）。

    Returns:
        `str | None`: 头的值（按 latin-1 解码）；不存在时 ``None``。
            注意空串与不存在是两回事：前者返回 ``""``，后者返回 ``None``。
    """
    for raw_name, raw_value in scope.get("headers") or []:
        if raw_name == name:
            return raw_value.decode("latin-1")
    return None


def remove_header(scope: Scope, name: bytes) -> None:
    """删除**所有**同名请求头。

    Args:
        scope (`Scope`): ASGI scope。
        name (`bytes`): 要删除的头名（小写）。
    """
    headers = scope.get("headers")
    if not headers:
        return
    scope["headers"] = [(k, v) for k, v in headers if k != name]


def set_header(scope: Scope, name: bytes, value: str) -> None:
    """把请求头设为指定值（先删光同名的，再追加）。

    ⚠️⚠️ **不能只 append**。这是本模块存在的首要理由。
        Starlette 取头是「首个匹配胜出」，因此客户端发来的
        `X-User-ID: attacker` 会排在中间件追加的 `X-User-ID: alice` **前面** ——
        框架的 ``get_current_user_id``（只校验非空、不校验真伪）会读到
        ``attacker``。结果就是：JWT 校验全部通过，身份却仍然是伪造的那个，
        而且**没有任何报错**。鉴权中间件会变成一件纯装饰品。

        先删后加，则无论客户端发了几个同名头，最终都只剩下我们注入的那一个。

    Args:
        scope (`Scope`): ASGI scope。
        name (`bytes`): 头名（小写）。
        value (`str`): 新的头值。
    """
    remove_header(scope, name)
    headers = list(scope.get("headers") or [])
    headers.append((name, value.encode("latin-1")))
    scope["headers"] = headers


def state_of(scope: Scope) -> dict[str, Any]:
    """取出（必要时创建）本次请求的 ``scope["state"]`` 字典。

    中间件之间用它传递「已经解析好的结论」，而不是让下游再解析一遍请求头。
    Starlette 的 ``Request.state`` 读的正是 ``scope["state"]``，因此中间件写进去的
    值，路由处理器可以直接用 ``request.state.xxx`` 取到。

    Args:
        scope (`Scope`): ASGI scope。

    Returns:
        `dict[str, Any]`: 该请求的状态字典（已保证存在）。
    """
    state = scope.get("state")
    if not isinstance(state, dict):
        state = {}
        scope["state"] = state
    return state


async def send_json(
    send: Send,
    status: int,
    payload: dict[str, Any],
    *,
    extra_headers: Iterable[tuple[bytes, bytes]] = (),
) -> None:
    """在中间件里直接返回一个 JSON 响应。

    中间件这一层拿不到 FastAPI 的 ``Response``/``HTTPException``（那些要走到
    路由匹配之后才存在），所以必须手写 ASGI 的两帧。

    ⚠️ 这里**不**设置 ``X-Trace-ID`` 响应头：那是
    :class:`~src.server.middleware.http_trace.TraceContextMiddleware` 的职责，
    它在这个中间件的外层，会给**所有** ``http.response.start`` 补上这个头。
    在这里再写一次会得到两个同名响应头，而客户端取首个 —— 反而可能取到旧值。

    Args:
        send (`Send`): ASGI 发送可调用对象。
        status (`int`): HTTP 状态码。
        payload (`dict`): 响应体（会被 JSON 序列化）。
        extra_headers (`Iterable[tuple[bytes, bytes]]`): 额外响应头。
    """
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = [
        (b"content-type", _JSON_CONTENT_TYPE),
        (b"content-length", str(len(body)).encode("ascii")),
        *extra_headers,
    ]
    await send(
        {"type": "http.response.start", "status": status, "headers": headers},
    )
    await send({"type": "http.response.body", "body": body})


__all__ = [
    "Message",
    "Receive",
    "Scope",
    "Send",
    "header_value",
    "remove_header",
    "send_json",
    "set_header",
    "state_of",
]
