# -*- coding: utf-8 -*-
"""业务路由包：``/api/v1/**`` —— 本项目的差旅业务域接口。

当前包含::

    _health.py    GET /api/v1/health   业务命名空间存活探针（公开）
    _identity.py  GET /api/v1/me       当前身份 / 鉴权模式 / trace_id（需鉴权）
    _chains.py    GET /api/v1/sessions/{session_id}/chains
                                       思考链 SSE（需鉴权，逐帧推全量快照）
    _default_model.py GET /api/v1/default-model
                                       当前身份可用的模型配置（需鉴权）
    _memory.py    GET/PUT /api/v1/memory/profile
                  POST/GET/DELETE /api/v1/memory/notes
                                       长期记忆的显式读写面（需鉴权）

P3 起会在这里加：行程（trips）、订单（orders）、申请单（applications）、
审批流（approvals）。届时每个业务域拆一个 ``_<domain>.py``，与上面几个
文件同级。（用户画像本来也在这张待办清单里，它已经落地为
``_memory.py`` —— 画像不是一个独立的业务域，而是长期记忆的**结构化的
那一半**，读写都从 :class:`~src.memory.service.TravelerMemory` 走。）

------------------------------------------------------------------------------
为什么文件名带下划线前缀
------------------------------------------------------------------------------
    与框架的 ``agentscope/app/_router/`` 保持同一约定：**带下划线的模块是
    「实现细节」，不带下划线的是「对外接口」**。本包的对外接口只有
    ``api_v1_router`` 一个符号，业务模块不该被别处直接 import ——
    否则路由的挂载点会散落在多个文件里，改前缀时要满仓库找。

------------------------------------------------------------------------------
路由前缀只在这里声明一次
------------------------------------------------------------------------------
    ``APIRouter(prefix="/api/v1")`` 只写在这一处，子模块里的 ``router``
    一律**不带前缀**。这样做的好处是「/api/v1」这个版本号在代码里只有一个
    真相来源：将来要开 ``/api/v2`` 并行，是把本文件复制一份改前缀，
    而不是在十几个子模块里做正则替换（那种替换一定会漏掉某个
    ``@router.get`` 上的硬编码路径）。

    对比框架的做法：``agentscope/app/_app.py`` 里 ``include_router`` 是
    **不加前缀**的，所以框架的路由全在根路径上（``/chat/``、``/sessions/``）。
    那是框架的既定契约，我们照用不改；我们自己的业务域则统一收进
    ``/api/v1``，两套命名空间并行、互不遮蔽。
"""

from fastapi import APIRouter

from ._chains import router as chains_router
from ._default_model import router as default_model_router
from ._health import router as health_router
from ._identity import router as identity_router
from ._memory import router as memory_router

#: 业务接口的统一路由器。由 ``src/server/app.py::create_root_app`` 挂到根应用。
api_v1_router = APIRouter(
    prefix="/api/v1",
    tags=["业务接口"],
)

api_v1_router.include_router(health_router)
api_v1_router.include_router(identity_router)
api_v1_router.include_router(chains_router)
api_v1_router.include_router(default_model_router)
api_v1_router.include_router(memory_router)

__all__ = ["api_v1_router"]
