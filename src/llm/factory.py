# -*- coding: utf-8 -*-
"""模型装配工厂 —— 全项目**唯一**构造聊天模型的地方。

文件职责：
    把 :class:`src.config.schema.LLMSettings` 变成可以直接调用的
    :class:`agentscope.model.ChatModelBase` 实例，并处理三件事：

        1. **选协议**：``provider`` → 具体的 ``ChatModelBase`` 子类；
        2. **零密钥降级**：没有密钥时返回 :class:`~src.llm.mock.MockChatModel`；
        3. **共享熔断器**：全进程只造一个 :class:`~src.llm.breaker.CircuitBreaker`。

上下游依赖：
    - 上游：读 :func:`src.config.loader.get_settings` 得到的配置。
    - 下游：``src/server/app.py``（app 链路与探针）、
      ``scripts/eval.py`` / ``scripts/smoke.py`` / 单测 —— 全都从这里拿模型。
      离线脚本与线上服务共用同一个装配函数，是「本地评测的分数能代表线上表现」
      的前提；一旦两条路径各自组装模型，评测就跑在一个和生产不同的模型上。

------------------------------------------------------------------------------
为什么必须是「唯一」的
------------------------------------------------------------------------------
    框架的 app 链路（``agentscope/app/_service/_model.py:12-63``）是从 storage 里的
    credential 记录反查模型类的。本模块的 :func:`build_credential` 与
    :func:`build_chat_model` 只要与那条链路的**构造方式一致**，两种入口就会得到
    同一种模型对象。若不一致（比如我们这边配了 ``temperature=0.3``、app 链路
    用默认值），同一个问题在「脚本里跑」和「页面上问」会得到不同风格的答案，
    而这种偏差极难被归因到「两个地方各写了一份参数」。

------------------------------------------------------------------------------
零密钥降级的判据（与 config/base.yaml 的注释逐字对应）
------------------------------------------------------------------------------
    当且仅当 ``use_mock_when_no_key`` 为真 **且** ``api_key`` 为空 → 返回 MockLLM。

    · 只留空 key、却把 ``use_mock_when_no_key`` 设为 false ⇒ **抛出异常**
      （而不是降级）。这是刻意的：显式关掉降级又不给密钥，是自相矛盾的配置，
      应当在启动时立刻失败，而不是让一个「看起来在跑」的服务在第一次对话时
      才报错。
    · ``api_key`` 非空但 ``use_mock_when_no_key`` 为真 ⇒ 正常走真实模型。
      降级开关只在**确实没有 key** 时才起作用，不会因为开关忘了关就把线上
      流量偷偷导到 Mock 上。
"""

from __future__ import annotations

import logging
from typing import Any

from agentscope.credential import (
    CredentialBase,
    DashScopeCredential,
    OpenAICredential,
)
from agentscope.model import ChatModelBase, DashScopeChatModel, OpenAIChatModel

from ..config import Settings, get_settings
from .bounded import bound_chat_model
from .breaker import CircuitBreaker
from .mock import MockChatModel, MockCredential

logger = logging.getLogger(__name__)

#: 阿里云百炼（DashScope）。**当前默认档**，走框架的 :class:`DashScopeChatModel`。
#: 选它而不是「用 OpenAI 兼容协议打百炼」有实质差别：原生档除基础采样参数外还支持
#: ``thinking_enable`` / ``thinking_budget`` / ``top_k``，并带 DashScope 专属的消息
#: 格式化与多模态块处理。P3 的「显示推理」依赖 thinking 字段 —— 走兼容协议档拿不到。
PROVIDER_DASHSCOPE = "dashscope"

#: 通用 OpenAI 兼容协议。**不是**「旧档」，而是**私有化/多厂商**档：
#: vLLM、SGLang、One-API 之类网关、以及任何声称 OpenAI 兼容的厂商都走它，
#: 换服务商只改 ``base_url``、不改代码。保留它是为了让本项目的部署形态不受限于百炼。
PROVIDER_OPENAI = "openai"

#: provider → (凭据类, 模型类)。**这是 provider 的唯一登记处**：
#: 加一档新协议只需要在这里加一行，:func:`build_credential` 与
#: :func:`build_chat_model` 都从这张表取类，不必各自再加一个 if 分支 ——
#: 那种写法迟早会出现「凭据分支加了、模型分支忘了」的半接线状态，
#: 而它的症状是一个能构造凭据却在装配模型时抛 AttributeError 的启动失败。
#:
#: ⚠️ 为什么两个类要成对登记：框架的 app 链路是靠 credential 记录反查模型类的
#:    （``agentscope/app/_service/_model.py:12-63`` 调 ``get_chat_model_class()``）。
#:    配错对（比如 DashScope 的 key 配 OpenAICredential）时，app 链路会去实例化
#:    一个端点不对的模型类，报错停在「404 / 认证失败」，看起来像密钥问题。
PROVIDER_REGISTRY: dict[str, tuple[type[CredentialBase], type[ChatModelBase]]] = {
    PROVIDER_DASHSCOPE: (DashScopeCredential, DashScopeChatModel),
    PROVIDER_OPENAI: (OpenAICredential, OpenAIChatModel),
}

#: 每个 provider 对应的密钥环境变量名 —— **只用于错误文案**。
#: 写死在这里而不是从 settings 反推，是因为这个函数恰恰在「配置错了」的路径上被调用：
#: 那时能信的只有 provider 本身。文案里给对环境变量名，是把一次配置排障从
#: 「读源码」降成「照抄提示」的唯一办法。
_API_KEY_ENV_HINT: dict[str, str] = {
    PROVIDER_DASHSCOPE: "DASHSCOPE_API_KEY",
    PROVIDER_OPENAI: "OPENAI_API_KEY",
}

#: 熔断器名字（出现在日志、``/metrics`` 与 ``/readyz`` 里）。
BREAKER_NAME = "llm"


# ==============================================================================
# 凭据
# ==============================================================================
def has_api_key(settings: Settings) -> bool:
    """判断配置里是否**真的**给了密钥。

    只判断「非空」是不够的：``.env.example`` 里存在 ``sk-xxx`` 这类占位符，
    而 ``config/base.yaml`` 用的是 ``${DASHSCOPE_API_KEY}`` 展开（未设置时展开成
    空串，见 ``src/config/loader.py::_expand_env``）。因此这里统一做
    ``strip()``，把「全是空格」也当作没有密钥。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `bool`: 有可用密钥返回 True。
    """
    return bool(settings.llm.api_key.strip())


#: ``has_api_key`` 的历史私有名。**保留**（而不是删掉）是因为它出现在
#: 若干既有调用点与单测里，删掉会得到一串 ImportError，而收益只是少一行。
#: 新代码请用公开名 ``has_api_key`` —— 它是本模块「密钥是否存在」这一判据的
#: 唯一实现，``should_use_mock`` 也复用它。
_has_api_key = has_api_key


def should_use_mock(settings: Settings) -> bool:
    """判定本次装配是否应当降级为 MockLLM。

    把判据单独抽成函数（而不是写在 :func:`build_chat_model` 的 if 里），
    是为了让 ``/readyz``、smoke 脚本、单测都能**引用同一份判据**。
    判据一旦分散，最常见的后果是「探针说在用真模型、实际走的是 Mock」。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `bool`: 应当降级返回 True。
    """
    return settings.llm.use_mock_when_no_key and not _has_api_key(settings)


def resolve_provider(settings: Settings) -> str:
    """取出并校验 provider，返回登记表里确实存在的那个名字。

    配置层（``LLMSettings.provider``）已经用 ``Literal`` 挡掉了写错的值，
    因此这里走到 ``raise`` 只可能是「schema 加了新取值、登记表却忘了加」——
    即本项目自己的疏漏。仍然显式报错而不是 ``KeyError``：
    后者会以一个看不出所以然的字典键错误出现在启动日志里。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `str`: 已登记的 provider 名。

    Raises:
        ValueError: provider 未在 :data:`PROVIDER_REGISTRY` 中登记。
    """
    provider = settings.llm.provider
    if provider not in PROVIDER_REGISTRY:
        raise ValueError(
            f"未知的 llm.provider={provider!r}；已登记的有 "
            f"{sorted(PROVIDER_REGISTRY)}。"
            "若这是新加的协议，请同时补上 PROVIDER_REGISTRY 与 schema 的 Literal。",
        )
    return provider


def _effective_base_url(settings: Settings) -> str | None:
    """返回**最终会打到**的端点（配置留空时用凭据类的默认值兜底）。

    为什么不能直接返回 ``settings.llm.base_url``：DashScope 档的端点写在了
    ``config/base.yaml`` 的 ``${DASHSCOPE_BASE_URL:-…}`` 默认值里、OpenAI 档则可能
    整个留空交给 SDK —— 两种情况下「配置里的字符串」与「真正请求的地址」并不一致。
    而 ``/readyz`` 与启动日志里那个 base_url 字段的全部价值就是回答
    「到底打到了哪一家」，一个空字符串或 ``None`` 等于什么都没回答，
    排障的人还是得回去翻配置。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `str | None`: 有效端点；连凭据类都没有默认值时为 ``None``
        （``OpenAICredential`` 就是这种，此时由 SDK 决定官方端点）。
    """
    configured = settings.llm.base_url.strip()
    if configured:
        return configured

    credential_cls, _ = PROVIDER_REGISTRY[resolve_provider(settings)]
    default = credential_cls.model_fields["base_url"].default
    # pydantic 未设默认值时是 PydanticUndefined，它不是字符串 —— 归一成 None。
    return default if isinstance(default, str) and default else None


def build_credential(settings: Settings) -> CredentialBase:
    """按配置构造凭据对象（真实密钥 或 Mock 凭据）。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `CredentialBase`: ``DashScopeCredential`` / ``OpenAICredential``，
        或 :class:`MockCredential`（降级时）。

    Raises:
        ValueError: 未降级、但也没有密钥时（配置自相矛盾，见模块文档字符串）。
    """
    if should_use_mock(settings):
        return MockCredential()

    provider = resolve_provider(settings)

    if not _has_api_key(settings):
        env_hint = _API_KEY_ENV_HINT.get(provider, "OPENAI_API_KEY")
        raise ValueError(
            "配置矛盾：llm.use_mock_when_no_key 为 false，但 llm.api_key 为空。\n"
            "请二选一：\n"
            f"  · 在 .env 里设置 {env_hint}（真实调用）；或\n"
            "  · 把 ALIGO__LLM__USE_MOCK_WHEN_NO_KEY 设为 true（零密钥降级）。",
        )

    credential_cls, _ = PROVIDER_REGISTRY[provider]

    # ⚠️ base_url 为空时要**整个省略这个参数**，而不是传 None 或 ""：
    #   · 传 ""  —— 会被 SDK 当成一个合法的（但无意义的）相对地址，
    #              报错变成「URL 格式非法」，排查方向被带偏；
    #   · 传 None —— OpenAICredential 接受（它的默认值本来就是 None），
    #              但 DashScopeCredential 的 base_url 是**必填 str**，
    #              传 None 会直接抛 ValidationError（已实测）。
    # 省略参数才能让**每个凭据类各自的默认值**生效 —— 那正是它们的官方端点。
    kwargs: dict[str, Any] = {"api_key": settings.llm.api_key}
    base_url = settings.llm.base_url.strip()
    if base_url:
        kwargs["base_url"] = base_url
    return credential_cls(**kwargs)


# ==============================================================================
# 模型
# ==============================================================================
def build_chat_model(
    settings: Settings | None = None,
    *,
    stream: bool = True,
    model_name: str | None = None,
) -> ChatModelBase:
    """按配置构造一个可直接调用的聊天模型。

    这是**全项目唯一的模型构造入口**（硬契约，见 ``.env.example`` 与
    ``config/base.yaml``）。返回的对象对调用方完全透明：``await model(msgs)``
    即可，重试与流式累积由框架的 ``ChatModelBase.__call__`` 负责。

    Args:
        settings (`Settings | None`): 配置；``None`` 时取进程内单例
            :func:`src.config.loader.get_settings`。
        stream (`bool`): 是否请求流式输出。注意这只是**请求**：
            真实模型可能因为服务端原因返回非流式结果，消费方必须两种情况都能处理
            （框架的 ``__call__`` 返回值因此是联合类型）。
        model_name (`str | None`): 覆盖 ``settings.llm.model`` 的模型名；
            ``None``（默认）时用配置里的主对话模型。**凭据、端点、超时、
            重试策略全部沿用同一份配置** —— 它只换「调哪个模型」，
            不换「怎么调」。存在的理由是重排阶段要开第二个模型
            （``settings.rerank.model``，见 ``src/server/agents_factory.py``），
            而那必须是**同一个**构造入口，否则「零密钥降级为 Mock」
            这类关键行为会在第二条路径上悄悄失效。

    Returns:
        `ChatModelBase`: :class:`DashScopeChatModel` / :class:`OpenAIChatModel`
        （按 ``provider`` 选，真实调用）或
        :class:`~src.llm.mock.MockChatModel`（零密钥降级）。

    Raises:
        ValueError: 配置自相矛盾（见 :func:`build_credential`）或 provider 未登记。
    """
    settings = settings or get_settings()
    llm = settings.llm
    # ⚠️ 只在这里做一次覆盖，后面**所有**用到模型名的地方都读 ``effective_model``
    # —— 从前面的降级分支到最后的装配日志。漏掉任何一处的症状都是
    # 「日志说调 A、实际调 B」，那是最难查的一类不一致。
    effective_model = model_name or llm.model
    provider = resolve_provider(settings)
    credential_cls, model_cls = PROVIDER_REGISTRY[provider]

    # ---- 降级分支 --------------------------------------------------------
    if should_use_mock(settings):
        env_hint = _API_KEY_ENV_HINT.get(provider, "OPENAI_API_KEY")
        logger.warning(
            "未配置模型密钥（llm.api_key 为空）且 use_mock_when_no_key=true，"
            "已降级为 MockLLM：模型 %r 不会被调用，回复由本地确定性生成。"
            "这适合本地开发与 CI；生产环境请配置 %s。",
            effective_model,
            env_hint,
        )
        return MockChatModel(
            credential=MockCredential(),
            model=effective_model,
            stream=stream,
            max_retries=llm.max_retries,
            retry_delay=llm.retry_backoff_seconds,
        )

    # ---- 真实分支 --------------------------------------------------------
    credential = build_credential(settings)
    assert isinstance(credential, credential_cls)  # 类型收窄，便于静态检查

    # 两个模型类的 Parameters 都至少含 max_tokens / temperature
    # （已核对 agentscope 2.0.10dev 的两处定义），因此可以走同一段构造。
    # DashScope 档另外还有 thinking_enable / thinking_budget / top_k / voice ——
    # 那些**刻意不在这里设**：本函数是两条链路（app 与离线脚本）的公共装配点，
    # 在这里打开思考会同时改变评测与线上的行为。P3 需要时按会话在中间件里注入。
    parameters = model_cls.Parameters(
        max_tokens=llm.max_tokens,
        temperature=llm.temperature,
    )

    # ⚠️ client_kwargs 里显式把 SDK 自己的重试关掉（max_retries=0）。
    # 不关的话会出现**两层重试相乘**：框架按 llm.max_retries(默认 2) 重试，
    # openai SDK 默认还要再自己重试 2 次 ⇒ 最坏情况 3×3=9 次请求。
    # 后果不只是慢：每一次都要走一遍熔断器的失败计数，
    # 于是一个「下游已经挂了」的判断会被放大成 9 倍的失败量，
    # 熔断阈值形的意义随之失真。这里让**框架作为唯一的重试权威**。
    # 两个模型类都把这个字典透传给底层的 AsyncOpenAI（已核对
    # DashScopeChatModel 的 `AsyncOpenAI(api_key=…, base_url=…, **self.client_kwargs)`
    # 与 OpenAIChatModel 的同名写法），因此这一段对两档同样有效。
    client_kwargs: dict[str, Any] = {
        "timeout": llm.timeout_seconds,
        "max_retries": 0,
    }

    model = model_cls(
        credential=credential,
        model=effective_model,
        parameters=parameters,
        stream=stream,
        max_retries=llm.max_retries,
        retry_delay=llm.retry_backoff_seconds,
        client_kwargs=client_kwargs,
    )

    logger.info(
        "已装配聊天模型：provider=%s model=%s base_url=%s（密钥已配置，值不打印）",
        provider,
        effective_model,
        # base_url 不是密钥，打印它是排障时最有用的一个字段
        # （「打到了哪一家」几乎能解释所有的认证/限流类问题）。
        # 用 _effective_base_url 而不是配置原文：留空时它给出凭据类的默认端点，
        # 因此这里永远能看到一个真实地址，而不是 <SDK 默认> 这种没有信息量的字面量。
        _effective_base_url(settings) or "<由 SDK 决定>",
    )
    return model


def build_rerank_model(
    settings: Settings | None = None,
    *,
    reuse: ChatModelBase | None = None,
) -> Any | None:
    """按配置构造**重排**用的对话模型（关闭时返回 ``None``）。

    ⚠️ 先把一件事说清楚：本项目的「重排」是 **LLM-as-reranker**。
    框架（agentscope 2.0.10dev）**没有**重排模型类 —— ``RAGMiddleware``
    的 ``rerank_model`` 参数收的是一个 ``ChatModelBase``，它按提示词
    （``middleware/_rag.py:85-94``）给候选段落打分排序。
    所以 ``settings.rerank.model`` 填的必须是**对话模型名**，
    写 ``qwen3-rerank`` / ``gte-rerank`` 这类专用重排模型名在本框架里
    必然调用失败（而失败会被 ``_rag.py:433`` 静默吞掉、退回向量序 ——
    症状是「开了重排但排序没变」，非常难发现）。

    ⚠️ 返回值**一律**套了 :class:`~src.llm.bounded.BoundedChatModel`。
    理由与 ``src/web_embedding/bounded.py`` 同源但更严重：重排调用发生在
    回复链路内（``RAGMiddleware.on_reasoning``），却没有 ``on_model_call``
    钩子覆盖它，而它的内部是一条「策略阶梯 + 重试」
    （``model/_base.py:511-534``）。没有这一层，一次卡住的重排会把用户的
    回复一起拖住，而框架那边的 ``except Exception`` 只在**抛异常**时才回退
    —— 卡住不返回的调用永远不抛异常。

    Args:
        settings (`Settings | None`): 配置；``None`` 时取进程内单例。
        reuse (`ChatModelBase | None`): 主对话模型。当
            ``settings.rerank.model`` 为空（=「复用主对话模型」）时直接返回
            **它**包上护栏的版本，而不是另建一个实例 —— 这样
            ``build_agent_wiring(model=...)`` 注入的测试替身也会被沿用，
            装配链在测试里不会偷偷去建一个真实模型。
            ⚠️ 复用实例是安全的：结构化输出走
            ``_call_api_with_structured_output``，基类对**流式**
            响应自己做了累积（``model/_base.py:668-681``），
            所以主模型 ``stream=True`` 不影响重排拿结果。

    Returns:
        `Any | None`: 带截止时间的对话模型；``rerank.enabled=false`` 时
        ``None``（⚠️ 不是「一个永远不生效的模型」——
        传 ``None`` 给 ``RAGMiddleware`` 与「开了但从不调用」在行为上等价，
        但前者让「本轮没开重排」在装配现场就看得出来）。
    """
    settings = settings or get_settings()
    rerank = settings.rerank
    if not rerank.enabled:
        if rerank.model:
            # ⚠️ 配了模型名却没开开关：这几乎一定是「以为自己开了」。
            # 静默忽略会让「重排到底有没有生效」变成一个要靠读配置回答的问题。
            logger.warning(
                "配置里设了 rerank.model=%r，但 rerank.enabled=false —— "
                "重排**不会**生效。要启用请设 ALIGO__RERANK__ENABLED=true。",
                rerank.model,
            )
        return None

    if rerank.model:
        inner = build_chat_model(settings, stream=False, model_name=rerank.model)
    elif reuse is not None:
        inner = reuse
    else:
        # 没有可复用的实例（例如离线脚本直接调本函数）：按主对话模型名建一个。
        inner = build_chat_model(settings, stream=False)

    bounded = bound_chat_model(inner, timeout=settings.llm.timeout_seconds)
    logger.info(
        "已装配重排模型：model=%s timeout=%gs（LLM-as-reranker，"
        "超时即退回向量序）",
        getattr(inner, "model", "<未知>"),
        settings.llm.timeout_seconds,
    )
    return bounded


# ==============================================================================
# 熔断器（全进程单例）
# ==============================================================================
_breaker: CircuitBreaker | None = None


def get_breaker(settings: Settings | None = None) -> CircuitBreaker:
    """返回进程内共享的熔断器（首次调用时创建）。

    **为什么必须是单例**：熔断的意义是「用全体调用者的失败共同判断下游是否可用」。
    如果每个会话、每次装配各造一个，那么下游彻底挂掉时，也要等**每个实例各自**
    连续失败到阈值才会熔断 —— 熔断点被推迟了 N 倍，而 N 正是并发数。
    那等于在最需要保护的时候完全不起保护作用。

    配置在进程内不变（见 ``src/config/loader.py`` 的单例设计），
    因此这里缓存下来的阈值/冷却时长不会与实际生效的配置漂移。

    Args:
        settings (`Settings | None`): 配置；``None`` 时取进程内单例。
            仅在**首次**调用时生效（后续调用复用已创建的实例）。

    Returns:
        `CircuitBreaker`: 进程内唯一的熔断器。
    """
    global _breaker
    if _breaker is None:
        settings = settings or get_settings()
        _breaker = CircuitBreaker(
            failure_threshold=settings.llm.circuit_breaker_failure_threshold,
            recovery_seconds=settings.llm.circuit_breaker_recovery_seconds,
            name=BREAKER_NAME,
        )
    return _breaker


def reset_breaker() -> None:
    """清空熔断器单例。

    **仅供测试使用**：用例之间必须互不影响 —— 上一个用例把熔断器打到 OPEN 之后
    若不清理，下一个用例的第一次调用会直接被拒，表现为「莫名其妙地失败」。
    生产代码不应调用它。
    """
    global _breaker
    _breaker = None


# ==============================================================================
# 只读描述（供日志与探针使用）
# ==============================================================================
def describe_model_target(settings: Settings | None = None) -> dict[str, Any]:
    """导出一份**不含密钥**的模型目标描述。

    给 ``/readyz`` 与启动日志用。刻意不返回 ``api_key`` 本身，只返回
    ``api_key_configured`` 这个布尔量 —— 探针与日志是最容易被截图、被贴进工单、
    被采集进第三方系统的地方，密钥一旦从这里泄出去，收都收不回来。

    Args:
        settings (`Settings | None`): 配置；``None`` 时取进程内单例。

    Returns:
        `dict`: 含 provider / model / base_url / 是否降级 / 密钥是否已配置 / 超时与重试参数。

        ``base_url`` 给的是 :func:`_effective_base_url` 的结果（**真正会打到的地址**），
        而不是配置原文 —— 配置留空时原文是个空串，对排障毫无用处。
    """
    settings = settings or get_settings()
    llm = settings.llm
    return {
        "provider": llm.provider,
        "model": llm.model,
        "base_url": _effective_base_url(settings),
        "using_mock": should_use_mock(settings),
        "api_key_configured": _has_api_key(settings),
        "timeout_seconds": llm.timeout_seconds,
        "max_retries": llm.max_retries,
    }


__all__ = [
    "BREAKER_NAME",
    "PROVIDER_DASHSCOPE",
    "PROVIDER_OPENAI",
    "PROVIDER_REGISTRY",
    "build_chat_model",
    "build_credential",
    "build_rerank_model",
    "describe_model_target",
    "get_breaker",
    "reset_breaker",
    "resolve_provider",
    "should_use_mock",
]
