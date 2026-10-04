# -*- coding: utf-8 -*-
"""模型装配包：把配置变成可调用的 AgentScope 模型对象，并加上熔断保护。

对外只暴露下面这些名字，内部实现细节留在各自的模块里::

    from src.llm import build_chat_model, get_breaker, reset_breaker

    model = build_chat_model()          # 全项目唯一的模型构造入口
    breaker = get_breaker()             # 全进程共享的熔断器
    async with breaker.guard():
        reply = await model(messages)

四个模块的分工：

    ``factory.py``  配置 → 模型对象。含「零密钥降级为 Mock」的判据，
                    以及重排模型（``settings.rerank``）的装配。
    ``breaker.py``  自研熔断器（框架没有这个能力）。三态状态机 + 单探针试探。
    ``mock.py``     确定性 Mock 模型，让无密钥环境也能跑通全链路。
    ``bounded.py``  模型调用的截止时间**代理**（``BoundedChatModel``）——
                    给那些不经过 ``on_model_call`` 钩子的调用（重排）兜底，
                    与 ``middleware.py`` 的钩子版是互补而非重复。
    ``middleware.py`` 熔断与截止时间的**中间件**版本（挂在钩子上，
                    作用于 agent 自己的模型调用）。

⚠️ 这里**不做**任何模型实例的缓存。模型对象本身很轻（真正重的是它内部的
HTTP 客户端连接池），而缓存实例会让「配置改了但进程还拿着旧模型」这类问题
变得隐蔽。需要共享的是**熔断器**（它承载跨请求的失败统计），不是模型。
"""

from .bounded import (
    BoundedChatModel,
    ChatCallTimeout,
    bound_chat_model,
    unwrap_chat_model,
)
from .breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState
from .factory import (
    BREAKER_NAME,
    PROVIDER_DASHSCOPE,
    PROVIDER_OPENAI,
    PROVIDER_REGISTRY,
    build_chat_model,
    build_credential,
    build_rerank_model,
    describe_model_target,
    get_breaker,
    reset_breaker,
    resolve_provider,
    should_use_mock,
)
from .middleware import (
    BreakerMiddleware,
    ModelCallTimeout,
    ModelTimeoutMiddleware,
    build_breaker_middleware,
    build_model_timeout_middleware,
)
from .mock import REPLY_PREFIX, MockChatModel, MockCredential

__all__ = [
    "BREAKER_NAME",
    "PROVIDER_DASHSCOPE",
    "PROVIDER_OPENAI",
    "PROVIDER_REGISTRY",
    "REPLY_PREFIX",
    "BoundedChatModel",
    "BreakerMiddleware",
    "ChatCallTimeout",
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "CircuitState",
    "MockChatModel",
    "MockCredential",
    "ModelCallTimeout",
    "ModelTimeoutMiddleware",
    "bound_chat_model",
    "build_breaker_middleware",
    "build_chat_model",
    "build_credential",
    "build_model_timeout_middleware",
    "build_rerank_model",
    "describe_model_target",
    "get_breaker",
    "reset_breaker",
    "resolve_provider",
    "should_use_mock",
    "unwrap_chat_model",
]
