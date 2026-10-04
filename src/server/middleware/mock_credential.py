# -*- coding: utf-8 -*-
"""降级凭据播种中间件：让零密钥部署真的能对话。

文件职责：
    在**零密钥降级模式**下，为每个第一次出现的用户按需补一条 Mock 凭据，
    使前端的「可用模型」列表非空、会话能配上 ``chat_model_config``。

上下游依赖：
    - 上游：由 ``src/server/app.py`` 注册在 **Auth 之内、RateLimit 之外**
      （理由见下），身份从 Auth 写进 ``scope["state"]`` 的
      :data:`~src.server.middleware.auth.USER_ID_STATE_KEY` 读。
    - 下游：``src/llm/degradation.py::ensure_mock_credential``。

==============================================================================
为什么必须是中间件（而不是一个依赖项或启动钩子）
==============================================================================
    需要播种的是**框架自己的路由**（``GET /credential/``、``GET /model/``），
    我们无法往它们的 ``Depends`` 里塞东西；而启动那一刻又没有身份可用
    （没有用户表，见 ``src/llm/degradation.py`` 的说明）。

    中间件是唯一同时满足「看得到身份」与「覆盖所有路由」的位置。代价是
    每个请求多一次判断 —— 判断本身只是一次 ``set`` 查表（``self._seeded``），
    真正落库每个用户**每个进程只发生一次**。

------------------------------------------------------------------------------
顺序：必须在 Auth 之内
------------------------------------------------------------------------------
    执行顺序是 ``TraceContext → HttpMetrics → Auth → 本中间件 → RateLimit → 应用``。
    在 Auth **之外**的话读不到身份（此时 ``scope["state"]`` 里还没有 user_id），
    本中间件会退化成「永远什么都不做」—— 而它是**静默**的：没有日志、
    没有异常，症状与「零密钥下发送按钮是灰的」一模一样，根本分不出是
    没播种还是播种了没用。

    ⚠️ 别把它与 ``src/server/app.py`` 里那串 ``add_middleware`` 的**书写顺序**
    搞混：书写顺序与执行顺序**相反**（``insert(0)`` ⇒ 后写的在下标 0、
    最外层），所以本中间件在那份**书写列表里是第 2 项**
    （``[RateLimit, 本中间件, Auth, HttpMetrics, TraceContext]``），
    而**执行**时才落在 ``Auth`` 之后、``RateLimit`` 之前。
    两句话说的都对，但混用就会把中间件插到错误的位置上（见该处详细说明）。

------------------------------------------------------------------------------
失败一律不阻断请求
------------------------------------------------------------------------------
    播种只是「让默认路径更好用」，它不是业务请求的一部分。数据库抖动时
    正确的表现是「这一次请求照常返回（只是可能还没有降级凭据），
    下一个请求再试」，而不是把用户的请求变成 500 —— 那会把一个
    **可用性增强**变成一个**可用性缺陷**。因此这里吞掉所有异常，
    只记一条 warning（同一个进程只记一次，避免刷屏）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.llm.degradation import ensure_mock_credential
from src.llm.factory import should_use_mock

from ._asgi import Receive, Scope, Send, state_of
from .auth import USER_ID_STATE_KEY

logger = logging.getLogger(__name__)


class MockCredentialSeedMiddleware:
    """零密钥降级模式下按需播种 Mock 凭据的纯 ASGI 中间件。

    刻意**不用** ``BaseHTTPMiddleware``：后者会缓冲流式响应体，会破坏
    ``/sessions/{id}/stream`` 的实时性（与 ``http_trace.py`` 一致的理由）。
    """

    def __init__(
        self,
        app: Any,
        *,
        settings: Any,
        storage: Any,
    ) -> None:
        """保存下游应用、判据与存储。

        Args:
            app (`Any`): 下游 ASGI 应用（FastAPI 中间件的约定签名）。
            settings (`Settings`): 全量配置。**只在这里读一次** ——
                装配期算好 :attr:`_enabled`，请求期不再碰配置对象。
                这样做的额外好处是：单测可以断言「开关关掉时中间件完全不工作」。
            storage (`Any`): 框架的存储实现。⚠️ 必须与 ``create_app``
                拿到的是**同一个实例**（``src/server/app.py`` 里那个
                ``resolved_storage``），否则会出现「凭据写进了 A、
                对话链路从 B 里读」——症状是「刚播种的凭据下一秒就消失」。
        """
        self.app = app
        self._enabled: bool = should_use_mock(settings)
        self._storage = storage
        #: 已经成功播种过的用户。**进程内**的记忆，因此：
        #:   · 用户手动删掉降级凭据后，本进程不会再把它补回来（尊重用户选择）；
        #:   · 重启后用户第一次请求时会再补一条（幂等，见 ensure_mock_credential）。
        self._seeded: set[str] = set()
        #: 串行化「首次播种」这件事。并发请求（页面加载时前端会同时打
        #: /credential/、/model/、/sessions/ 等好几个）若各自走到「记录不存在
        #: ⇒ INSERT」，多出来的那些会撞主键。锁把这一窗口收窄到只有一个请求在写。
        self._lock = asyncio.Lock()
        #: 失败只警告一次，避免每次请求都刷一条同样的日志。
        self._warned: bool = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """处理一次 ASGI 调用。

        Args:
            scope (`Scope`): ASGI scope。
            receive (`Receive`): 接收可调用对象（本中间件不消费请求体）。
            send (`Send`): 发送可调用对象。
        """
        # 非 HTTP（lifespan / websocket）与「降级关闭」两条快路径：
        # 直接透传，一个多余的操作都不做。
        if scope.get("type") != "http" or not self._enabled:
            await self.app(scope, receive, send)
            return

        state = state_of(scope)
        user_id = state.get(USER_ID_STATE_KEY)
        # 没有身份有三种正常情况：白名单路径（Auth 放行且剥了头）、
        # 允许匿名模式、以及 Auth 都没配的裸部署。都不是本中间件的事。
        if isinstance(user_id, str) and user_id and user_id not in self._seeded:
            await self._seed(user_id)

        await self.app(scope, receive, send)

    async def _seed(self, user_id: str) -> None:
        """为 ``user_id`` 播种降级凭据，绝不抛异常。

        Args:
            user_id (`str`): 框架认定的用户标识。
        """
        try:
            async with self._lock:
                # 拿锁后再查一次：等锁期间别的请求可能已经播过了。
                if user_id in self._seeded:
                    return
                await ensure_mock_credential(self._storage, user_id)
                self._seeded.add(user_id)
        except Exception:  # noqa: BLE001 —— 见模块文档：绝不阻断请求
            if not self._warned:
                self._warned = True
                # ⚠️ ``make logs`` **不带**服务名：Makefile 的服务名走 ``SVC``
                # 变量（默认 app），写成 ``make logs app`` 会被 make 当成两个目标，
                # 报 ``No rule to make target 'app'`` —— 一条把人带偏的提示。
                logger.warning(
                    "零密钥降级凭据播种失败，已跳过（不影响本次请求）："
                    "后续请求会重试。请检查数据库连通性（make psql / make logs）。",
                    exc_info=True,
                )


__all__ = ["MockCredentialSeedMiddleware"]
