# -*- coding: utf-8 -*-
"""``/api/v1/default-model`` —— 「我现在该用哪个模型？」的机器可读答案。

文件职责：
    返回本次身份**当下可用**的模型配置，或在没有可用模型时给出
    「为什么没有」与「该怎么办」。零密钥降级模式下它就是那条
    「开箱即用」的配置来源。

上下游依赖：
    - 上游：``src/llm/degradation.py::resolve_default_model``（判据的唯一实现）、
      ``agentscope.app.deps.get_current_user_id``（框架的身份依赖）。
    - 下游：``scripts/smoke.py`` 的对话检查用它拿到配置后发第一轮消息；
      浏览器端的模型选择器在「可用模型为空」时也应回落到它。

==============================================================================
为什么需要一个新端点（不能只靠前端自己挑）
==============================================================================
    前端的模型选择器是「凭据列表 × 模型列表」拼出来的，两个都是**列表**接口：
    列表为空时它无法区分下面这两种完全不同的处境 ——

        (a) 本来就没有模型可用（需要人去配一条凭据）；
        (b) 有降级模型可用，只是**列表为空这件事本身**是缺陷
            （见 ``src/llm/mock.py::mock_model_card`` 的说明）。

    两者在 UI 上长得一模一样（一个禁用的发送按钮），而处置方式完全相反。
    本端点把处境显式说出来：``mode`` 是三选一的**枚举**（不是布尔值），
    ``hint`` 是一句可以直接交给用户的人话。

------------------------------------------------------------------------------
为什么响应里可以有 credential_id
------------------------------------------------------------------------------
    ``credential_id`` 是**记录 id**，不是密钥；它属于调用者自己
    （查询用的是 ``Depends(get_current_user_id)`` 解析出的身份），
    因此不构成越权信息。响应里**没有**、也不允许出现任何密钥字段 ——
    本端点与它的下游（``resolve_default_model``）任何一行都不读凭据的
    ``data``，只读记录的 id。

    ⚠️ 这条约束是刻意的：一旦将来有人「顺手」在这里把凭据的 ``data`` 也带上
    （方便前端一键填充之类），明文密钥就会进入一个 HTTP 响应体 —— 而它同时
    会进入访问日志、浏览器缓存与任何中间代理。要填凭据，走框架自己的
    ``POST /credential/``，那条路径的返回形态是被单独评审过的。
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from agentscope.app.deps import get_current_user_id

from ...llm.degradation import (
    MODE_CONFIGURED,
    MODE_MISSING,
    MODE_MOCK,
    MODE_SHARED,
    resolve_default_model,
)
from ...observability.context import get_trace_id

router = APIRouter()


@router.get(
    "/default-model",
    summary="当前身份的默认模型配置",
    description=(
        "返回当前用户在**此刻**可用的模型配置。零密钥降级模式下返回一条"
        "可直接使用的 MockLLM 配置（不含任何密钥）；用户已自配凭据时返回 "
        "mode=configured 并且 chat_model_config 为 null（由用户在模型选择器"
        "中自行选择）；用户没有凭据但运营者的系统凭据对其开放时返回 "
        "mode=shared 与一份可用配置；以上皆无时返回 mode=missing 与人话提示。"
    ),
    responses={
        200: {"description": "查询成功（四种 mode 均为 200）"},
        401: {"description": "凭据缺失或无效（由鉴权中间件返回）"},
    },
)
async def api_v1_default_model(
    request: Request,
    user_id: str = Depends(get_current_user_id),
) -> JSONResponse:
    """返回默认模型配置。

    ⚠️ 四种模式一律返回 **200**，而不是「没有模型就 404/503」：
        调用方（前端、smoke、运维的 curl）真正需要的信息是 ``mode`` 与
        ``hint``，而不是一个 HTTP 状态码。用非 200 表达「没有模型」会让
        客户端把「一个正常的业务状态」当成传输层故障 —— 于是各种重试、
        告警、熔断会对着一个「本来就该有人去配凭据」的状态狂响。

    Args:
        request (`Request`): 当前请求（读取 app.state 上的存储实例）。
        user_id (`str`): 由框架的身份依赖解析出的用户标识。

    Returns:
        `JSONResponse`: ``{mode, chat_model_config, hint, trace_id}``。
    """
    settings = getattr(request.app.state, "settings", None)
    storage = getattr(request.app.state, "storage", None)

    # 装配契约：create_app 一定会把这两样写进 app.state（见
    # agentscope/app/_lifespan.py）。取不到说明应用被以非预期的方式装配了
    # （比如某个测试直接构造了裸 FastAPI），此时给一条**能看懂**的 503，
    # 而不是让下面某个 getattr 变成 AttributeError → 500。
    if settings is None or storage is None:
        return JSONResponse(
            status_code=503,
            content={
                "mode": MODE_MISSING,
                "chat_model_config": None,
                "hint": (
                    "应用未完成装配（app.state 上缺少 settings/storage），"
                    "请确认服务是通过 src.server.app:app 启动的。"
                ),
                "trace_id": get_trace_id(),
            },
        )

    # 「系统凭据对谁可用」的唯一权威是访问策略 —— 由 create_app 在装配期
    # 写进 app.state（``app/_app.py``：``app.state.resource_access_policy``，
    # 缺省是框架的 DenyAllResourceAccessPolicy）。
    #
    # ⚠️ 用 getattr 兜默认值而不是直接取属性：测试里存在「裸 FastAPI」
    # 这种装配形态（没有经过 create_app），直接取会 AttributeError → 500，
    # 而那正是上面那段 503 分支想避免的事。取不到时传 None，
    # ``resolve_shared_model`` 会当作「不共享」——也就是框架默认的 deny-all 语义。
    policy = getattr(request.app.state, "resource_access_policy", None)
    resolved = await resolve_default_model(
        storage,
        settings,
        user_id,
        policy=policy,
    )
    return JSONResponse(
        {
            "mode": resolved["mode"],
            "chat_model_config": resolved["chat_model_config"],
            "hint": resolved["hint"],
            "trace_id": get_trace_id(),
        },
    )


__all__ = [
    "MODE_CONFIGURED",
    "MODE_MISSING",
    "MODE_MOCK",
    "MODE_SHARED",
    "router",
]
