# -*- coding: utf-8 -*-
"""``/api/v1/health`` —— 业务命名空间的**存活**探针。

文件职责：
    返回一个极浅的 200，证明「``/api/v1`` 这套路由确实注册上了、
    且请求能穿过鉴权与限流中间件到达业务层」。

上下游依赖：
    - 上游：``src/server/middleware/auth.py`` 的 ``PUBLIC_PATHS`` 必须包含
      本文件的路径（否则探针自己会被 401 拦住，而 compose 的 healthcheck
      会因此判定容器不健康 —— 一个自指的故障）。
    - 下游：``scripts/smoke.py`` 打这个端点。

==============================================================================
为什么它和 ``/healthz`` 是**两个**端点，而不是同一个
==============================================================================
    它们的**读者不同**，因此契约也不同：

        /healthz   给**编排系统**看（compose healthcheck / k8s livenessProbe）。
                   顶层、免鉴权、极浅（不查任何外部依赖）。
        /api/v1/health  给**调用方**看（前端、联调脚本、API 使用者）。
                   它处在业务命名空间里，因此它顺带证明了
                   「鉴权中间件放行了公开路径」「限流没有误伤探针」
                   「/api/v1 的路由树装配正确」这三件事。

    如果只有 ``/healthz``，那么当 ``/api/v1`` 整个路由树因为 include 写错
    而没挂上时，所有探针仍然是绿的 —— 这正是 P1 阶段 ``smoke.py``
    把 ``/api/v1/health`` 的 404 当作「通过」的原因（当时这个路由还没写）。
    P2 起它必须返回 200，``smoke.py`` 的判定也随之收紧。

    ⚠️ 浅是**刻意**的：这里不查 PG / Redis / Milvus。
    要查依赖请打 ``/readyz``。在探针里做 I/O 会让它变成一个
    「依赖抖动就红、红了就重启」的放大器，而依赖抖动本身往往几分钟就自愈。
"""

from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get(
    "/health",
    summary="业务命名空间存活探针",
    description=(
        "返回 ``/api/v1`` 路由树已装配、且请求可穿过中间件链。"
        "**不检查任何外部依赖** —— 依赖检查见 ``/readyz``。"
    ),
    responses={200: {"description": "服务在运行"}},
)
async def api_v1_health() -> JSONResponse:
    """返回业务命名空间的存活状态。

    Returns:
        `JSONResponse`: 固定形状的存活响应。

    ⚠️ 响应体**刻意不含**版本号以外的环境信息（配置值、主机名、依赖地址）。
        公开路径的响应任何人都能拿到，把 ``db.url`` 之类的信息放进来
        等于给扫描器送情报。要排障请用带鉴权的 ``/readyz`` 或日志。
    """
    return JSONResponse(
        {
            "status": "ok",
            "scope": "api/v1",
            "message": "业务接口可用。依赖状态请查看 /readyz。",
        },
    )


__all__ = ["router"]
