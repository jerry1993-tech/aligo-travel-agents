# -*- coding: utf-8 -*-
"""服务层包：HTTP 应用的装配、路由、中间件与静态资源。

包结构::

    app.py             ★ 硬契约：模块级 ``app``，uvicorn src.server.app:app
    probes.py          /healthz（存活）、/readyz（就绪）、/metrics（指标）
    middleware/        纯 ASGI 中间件（trace_id、请求指标；P2 加鉴权与限流）
    routers/           业务路由（P2 起：/api/v1/**）
    schemas/           请求/响应模型（P2 起）
    static/            前端产物（P5 由 vite 构建到此处）

------------------------------------------------------------------------------
关于 ``app.py`` 的模块级副作用（这是一个需要知道的事实）
------------------------------------------------------------------------------
    :mod:`src.server.app` 在 **import 时**就调用 ``create_root_app()`` 完成装配。
    因此::

        import src.server.app      # ← 这一步就已经读配置、建对象了

    这么设计是为了让多进程启动（``uvicorn --workers N``）下每个 worker 各自
    持有独立的 storage / message_bus / 熔断器 —— 若改成 lifespan 里懒装配，
    会有 asyncio 原语跨进程共享的风险。

    带来的约束：
        · 新增模块**不要**在这里 import —— 那会让「导入 src.server」变成
          「启动整个服务」，模块边界就没了；
        · 需要单独构造应用（比如单测里换一份配置）时，请直接调用
          :func:`src.server.app.create_root_app`，而不要 import 本包。
"""

__all__: list[str] = []
