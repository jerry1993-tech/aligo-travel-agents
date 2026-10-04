# -*- coding: utf-8 -*-
"""可观测包：指标（Prometheus）与链路追踪（OpenTelemetry → Langfuse）。

对外暴露两组入口，分别对应「看趋势」与「看单次请求」::

    from src.observability import render_latest, observe_http_request
    from src.observability import setup_tracing, shutdown_tracing, is_tracing_active

三个模块的分工：

    ``metrics.py``   指标注册表 + 埋点辅助函数 + ``/metrics`` 的渲染。
    ``tracing.py``   TracerProvider 的条件式装配（框架没有 setup_tracing，必须自建）。

⚠️ 两者的**装配时机不同**，这是刻意的：

    · 指标是「用时即注册」—— ``metrics.py`` 在 import 时就建好了注册表和采集器，
      因为 Prometheus 是拉取模型，任何时刻被抓到都必须已经有完整指标表
      （少一条时序，面板上就是一条断线）。
    · trace 是「启动时装配」—— 必须由 lifespan 显式调用 :func:`setup_tracing`，
      因为它要决定「全局 TracerProvider 是谁」，而这个决定进程内只能做一次。

⚠️ 这两个模块都**不打印密钥**。排障时最高频的动作是「把探针响应或启动日志
贴进工单」，密钥一旦从这里出去就收不回来了 —— 因此所有的外部可见输出
（日志、``/readyz``）一律只输出「密钥是否已配置」这类布尔量。
"""

from .metrics import (
    REGISTRY,
    observe_breaker_state,
    observe_http_request,
    observe_model_call,
    render_latest,
    set_ready,
)
from .tracing import (
    describe_tracing,
    is_tracing_active,
    setup_tracing,
    shutdown_tracing,
)

__all__ = [
    "REGISTRY",
    "describe_tracing",
    "is_tracing_active",
    "observe_breaker_state",
    "observe_http_request",
    "observe_model_call",
    "render_latest",
    "set_ready",
    "setup_tracing",
    "shutdown_tracing",
]
