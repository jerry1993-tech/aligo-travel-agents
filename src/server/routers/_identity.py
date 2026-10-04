# -*- coding: utf-8 -*-
"""``/api/v1/me`` —— 把「你是谁」显式回答出来。

文件职责：
    返回本次请求最终被认定的身份、身份是怎么来的（鉴权模式）、
    以及本次请求的 trace_id。

上下游依赖：
    - 上游：``agentscope.app.deps.get_current_user_id`` —— **框架自己的**
      身份依赖，本项目 13 个路由文件、85 个 ``Depends`` 点用的都是它。
    - 下游：无。

==============================================================================
这个端点存在的唯一理由：把鉴权链路变成**可观测的**
==============================================================================
    鉴权做对了但看不见，等于没做。没有这个端点时，验证「JWT 有没有被正确
    解析并注入成 X-User-ID」只能靠间接证据：拿一个会话去查、看能不能看到、
    再换一个身份看还看不看得到 —— 一旦结果不对，你分不清是
    **鉴权没生效**（JWT 压根没解析出来）还是**隔离没生效**（解析对了
    但存储层没按身份过滤）。这是两条完全不同的排查方向。

    ``/api/v1/me`` 把第一层单独暴露出来，于是排障变成一句话的事：
    它返回的 ``user_id`` 对得上，说明鉴权链路通；对不上，问题在中间件。

------------------------------------------------------------------------------
为什么用**框架的**依赖，而不是自己从 scope 里读
------------------------------------------------------------------------------
    本模块有两种拿到身份的写法：

        (a) 自己 ``request.scope["state"]["aligo_user_id"]``（读我们中间件写的值）
        (b) ``Depends(get_current_user_id)``（走框架的请求头依赖）

    这里选 (b)，因为 (b) 才是**有意义的证明**：
    它证明了我们的鉴权中间件产出的东西，与框架那 85 个既有依赖点
    期望的输入**完全兼容**。用 (a) 的话，即使中间件只写 scope、没写请求头，
    这个端点照样返回正确身份 —— 测试全绿，而框架的会话接口会全部 422。
    换句话说，(a) 测的是「我们的中间件」，(b) 测的是「整条线」。

    两种模式殊途同归的前提是：Auth 中间件在 JWT 模式下会先**删光**已有的
    ``x-user-id`` 再注入解析出的 ``sub``（见 ``_asgi.set_header`` 的说明）。
    否则客户端只要自己带一个 ``X-User-ID: 别人`` 就能在 JWT 模式下冒充 ——
    Starlette 的 ``Headers`` 取的是**第一个**匹配值，伪造的那一个会赢。

------------------------------------------------------------------------------
``auth_mode`` 的四个取值
------------------------------------------------------------------------------
    ``jwt``       —— 走 JWT 解析，身份来自 token 的 ``sub``。
    ``header``    —— JWT 关闭且要求请求头，身份来自 ``X-User-ID``。
    ``public``    —— 该路径在白名单里，中间件**没做**鉴权（本端点不会走到）。
    ``anonymous`` —— 两个开关都关，中间件放行且不注入身份。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

# 框架自己的身份依赖。**顶层导入**是刻意的：
# ``Depends(...)`` 的参数在**装饰器求值那一刻**就被计算，把它藏进一个
# 「延迟导入」的辅助函数里只是看起来惰性，实际仍在模块导入期执行 ——
# 那种写法会让人以为「本模块不依赖 agentscope」，是比直接导入更坏的误导。
# 导入 ``agentscope.app.deps`` 本身不需要任何密钥、也不建立连接，
# 与「零密钥可启动」不冲突。
from agentscope.app.deps import get_current_user_id

from ...observability.context import get_trace_id
from ..middleware import AUTH_MODE_STATE_KEY, USER_ID_STATE_KEY

router = APIRouter()


def _auth_mode_of(request: Request) -> str:
    """读取本次请求的鉴权模式。

    Args:
        request (`Request`): 当前请求。

    Returns:
        `str`: ``jwt`` / ``header`` / ``public`` / ``anonymous``；
        中间件没有留下标记时返回 ``unknown``（正常装配下不会出现，
        出现即说明本路由被挂到了一个没经过 Auth 中间件的应用上）。
    """
    state = request.scope.get("state")
    if isinstance(state, dict):
        mode = state.get(AUTH_MODE_STATE_KEY)
        if isinstance(mode, str) and mode:
            return mode
    return "unknown"


def _user_id_from_scope(request: Request) -> str | None:
    """读取中间件写进 scope 的原始身份（仅用于**交叉核对**）。

    Args:
        request (`Request`): 当前请求。

    Returns:
        `str | None`: 中间件认定的身份；未注入时为 ``None``。
    """
    state = request.scope.get("state")
    if isinstance(state, dict):
        value = state.get(USER_ID_STATE_KEY)
        if isinstance(value, str) and value:
            return value
    return None


@router.get(
    "/me",
    summary="当前请求的身份与链路信息",
    description=(
        "返回本次请求被认定的用户身份、鉴权模式与 trace_id。"
        "用于验证鉴权链路（JWT/请求头 → 框架的 get_current_user_id）是否生效。"
    ),
    responses={
        200: {"description": "身份解析成功"},
        401: {"description": "凭据缺失或无效（由鉴权中间件返回）"},
        422: {"description": "身份依赖未取到可用的用户标识"},
    },
)
async def api_v1_me(
    request: Request,
    user_id: str = Depends(get_current_user_id),
) -> JSONResponse:
    """返回当前身份信息。

    Args:
        request (`Request`): 当前请求。
        user_id (`str`): 由框架的身份依赖解析出的用户标识。

    Returns:
        `JSONResponse`: 身份、鉴权模式、trace_id 与一致性标记。

    ⚠️ ``consistent`` 字段是刻意放进响应体的：
        它比对「框架依赖解析出的 user_id」与「中间件写进 scope 的
        user_id」。两者不一致意味着**有东西在中间层改了请求头**——
        最可能的是一条新加的中间件或网关注入，而这类不一致若不被显式暴露，
        会表现为「某些接口按 A 用户过滤、另一些按 B 用户」，极难定位。
        正常装配下它恒为 ``true``。
    """
    from_scope = _user_id_from_scope(request)
    return JSONResponse(
        {
            "user_id": user_id,
            "auth_mode": _auth_mode_of(request),
            "trace_id": get_trace_id(),
            "consistent": from_scope == user_id,
        },
    )


__all__ = ["router"]
