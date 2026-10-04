# -*- coding: utf-8 -*-
"""按配置选出**一个**可用的向量模型，并在需要时沿降级链下降。

对外只有两个函数：:func:`build_embedding_model`（构造）与
:func:`describe_embedding`（把「实际用了哪一档」说清楚）。
"""

from __future__ import annotations

import logging

from agentscope.embedding import EmbeddingModelBase

from src.config.schema import Settings

from .bounded import bound_embedding_model
from .local import LocalEmbeddingUnavailableError, LocalOnnxEmbeddingModel
from .mock import MockEmbeddingModel

#: 本模块的日志器。
logger = logging.getLogger(__name__)

#: 降级链的顺序：``provider`` 是**起点**，不是终点。
#:
#: ⚠️ 顺序写死在这里，不跟配置走。理由：降级链的每一档都是「能力更弱
#: 但更可能可用」，这个强弱关系是**客观的**（云 > 本地 > 假向量），
#: 让它可以被配置只会制造出「先试假向量再试云」这种没有意义的组合。
#: 配置能决定的是**从哪一档开始**（``provider``）与**要不要降**（``allow_fallback``）。
_CHAIN: tuple[str, ...] = ("dashscope", "local", "mock")


class EmbeddingUnavailableError(RuntimeError):
    """沿降级链走到底仍没有可用实现。

    ⚠️ 只有一种情况会抛它：``allow_fallback=false`` 时首选档不可用。
    这是**刻意的**——生产环境不该因为少一个 key 就悄悄用上假向量。
    错误信息必须说清「哪一档为什么不可用」，否则运维只会看到
    「启动失败」而不知道该装什么、该配什么。
    """


def _build_dashscope(settings: Settings) -> EmbeddingModelBase:
    """构造云端（DashScope）向量模型 —— **完全**用框架的实现。

    ⚠️ 这里一行都没有重写框架的能力：模型类、凭据类、参数对象全部来自
    ``agentscope``。本函数做的只有「把配置翻译成构造参数」这一件事。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `EmbeddingModelBase`: 框架的 ``DashScopeEmbeddingModel`` 实例。

    Raises:
        EmbeddingUnavailableError: 没有配置 DashScope 的 api key。
    """
    from agentscope.credential import DashScopeCredential
    from agentscope.embedding import DashScopeEmbeddingModel

    api_key = settings.llm.api_key
    if not api_key:
        raise EmbeddingUnavailableError(
            "DashScope 向量档需要 api key，但 llm.api_key 为空。\n"
            "在 .env 里设置 DASHSCOPE_API_KEY，"
            "或把 embedding.provider 改成 local / mock。",
        )

    return DashScopeEmbeddingModel(
        credential=DashScopeCredential(api_key=api_key),
        model=settings.embedding.model,
        # ⚠️ dimensions 是**必填**的（``agentscope/embedding/_embedding_base.py:158``：
        # 为 None 且 legacy parameters 里也没有时抛 ValueError）。
        # 传配置值而不是让框架去猜：这三档必须输出同一个维度，
        # 而「同一个维度」这件事只有我们自己的配置知道。
        dimensions=settings.embedding.dimension,
    )


def _build_local(settings: Settings) -> EmbeddingModelBase:
    """构造本地（ONNX）向量模型。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `EmbeddingModelBase`: 本地模型实例。

    Raises:
        LocalEmbeddingUnavailableError: ``fastembed`` 未安装或模型加载失败。
    """
    return LocalOnnxEmbeddingModel(
        dimensions=settings.embedding.dimension,
        model=settings.embedding.local_model,
    )


def _build_mock(settings: Settings) -> EmbeddingModelBase:
    """构造确定性假向量模型。

    ⚠️ 这一档**永不失败**。它是降级链的终点，也是 ``allow_fallback=false``
    时最不该选中的那一档 —— 见 :mod:`src.web_embedding` 的模块文档。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `EmbeddingModelBase`: 假向量模型实例。
    """
    return MockEmbeddingModel(dimensions=settings.embedding.dimension)


#: 档位名 → 构造函数。
_BUILDERS = {
    "dashscope": _build_dashscope,
    "local": _build_local,
    "mock": _build_mock,
}


def build_embedding_model(settings: Settings) -> EmbeddingModelBase:
    """构造向量模型：从 ``embedding.provider`` 起沿降级链找第一个可用的。

    ⚠️ **只在启动期调用一次**，把结果存起来复用。两档的原因不同：

    · ``dashscope`` 档每次构造都会新建一个 HTTP 客户端（连接池）；
    · ``local`` 档每次构造都会**加载模型权重**，是秒级的操作。

    放进请求路径等于按请求付这两笔开销。

    ⚠️ 降级**会记 warning**，而且是逐档记的。把「实际用了哪一档」写进日志
    这件事看似琐碎，实际是这套设计里唯一能发现「线上悄悄在用假向量」的线索 ——
    因为从检索结果上看不出来（见模块文档）。

    ⚠️ 返回前**一律套上** :func:`~src.web_embedding.bounded.bound_embedding_model`
    （超时取 ``embedding.timeout_seconds``）。三条出口（不降级 / 首选可用 /
    降级成功）都必须经过它 —— 检索路径上「先向量化、再查库」，
    两段都得有界，理由全在 :mod:`src.web_embedding.bounded` 的模块文档里。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `EmbeddingModelBase`: 可用的向量模型。

    Raises:
        EmbeddingUnavailableError: ``allow_fallback=false`` 且首选档不可用；
            或 ``provider`` 不是已知档位（正常不会发生，schema 的 Literal 已挡住）。
    """
    provider = settings.embedding.provider
    if provider not in _BUILDERS:
        # ⚠️ 正常到不了这里：schema 把 provider 声明成 Literal，
        # 非法值在**配置加载期**就报错了。留着它是因为本函数可能被
        # 单测直接调用（绕开 schema），那时 KeyError 的信息量远不如这句话。
        raise EmbeddingUnavailableError(
            f"未知的 embedding.provider={provider!r}，"
            f"合法值：{'、'.join(sorted(_BUILDERS))}。",
        )

    # 从 provider 那一档开始，截取降级链的**尾部**。
    # ⚠️ 用 index 截尾而不是 `if provider in ... else` 分支：
    # 后者在往链里插一档时（比如将来加一个 ollama 档）会出现
    # 「新档位没被任何分支覆盖」的静默漏洞。
    start = _CHAIN.index(provider)
    chain = _CHAIN[start:]

    if not settings.embedding.allow_fallback:
        # ⚠️ 不降级时**只试首选档**，而且把它的异常**原样透出** ——
        # 不要包成 EmbeddingUnavailableError 的笼统版本：
        # 上面两档抛出的错误信息已经写清了「装什么 / 配什么」，
        # 再包一层只会把那段最有用的文字埋进 cause 里。
        model = _BUILDERS[provider](settings)
        logger.info(
            "向量模型：%s（provider=%s，allow_fallback=false，不降级）。",
            type(model).__name__,
            provider,
        )
        # ⚠️ 三条出口**都要**包（这里、下面的降级出口、以及不降级出口）——
        # 漏掉任何一条，那条路上的检索就仍然会在向量化那一步无界等待，
        # 而症状是「换个 embedding.provider 就偶发假死」，极难归因。
        return bound_embedding_model(model, timeout=settings.embedding.timeout_seconds)

    failures: list[str] = []
    for index, name in enumerate(chain):
        try:
            model = _BUILDERS[name](settings)
        except Exception as exc:  # noqa: BLE001 —— 见下面关于「为什么这么宽」的说明
            # ⚠️ 捕获**宽**异常是刻意的：这一层的判断是「这一档能不能用」，
            # 而各档不可用的异常类型天生不同（缺 key 是自定义异常、
            # 缺包是 ImportError、模型名写错是 ValueError、
            # 网络不通是 requests 的各种异常）。按类型列举必然漏，
            # 漏掉的那一种就会让降级链在最需要它的时候不起作用。
            #
            # 安全性由**位置**保证而不是由类型保证：这个 try 恰好只包住
            # 「构造」这一件事，构造之外的任何代码都不在这里。
            failures.append(f"{name}: {exc}")
            short = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            logger.warning(
                "向量档 %s 不可用（%s），%s。",
                name,
                short,
                "降级到下一档" if index + 1 < len(chain) else "已无更多档位",
            )
            continue

        if name != provider:
            # ⚠️ 这一行是**运维能看到的唯一提示**：从检索结果里看不出
            # 用了假向量（见模块文档），所以「降级了」必须留下痕迹。
            # 用 warning 而不是 info：它是一个「功能已降级」的事实。
            logger.warning(
                "⚠️ 向量模型已从 %s 降级到 %s。检索仍会返回结果，"
                "但质量与配置预期不符 —— 请检查上一段的不可用原因。",
                provider,
                name,
            )
        logger.info("向量模型：%s（provider=%s）。", type(model).__name__, name)
        return bound_embedding_model(model, timeout=settings.embedding.timeout_seconds)

    # 走到这里说明连 mock 档都没构造出来 —— 理论上不可能（它不依赖任何外部条件）。
    # ⚠️ 仍然显式处理：静默返回 None 或抛一个没上下文的异常，
    # 会让这个「不可能」变成一个极难排查的问题。
    raise EmbeddingUnavailableError(
        "降级链全部失效（这不该发生：mock 档不依赖任何外部条件）。\n"
        + "\n".join(f"  · {item}" for item in failures),
    )


def describe_embedding(settings: Settings) -> dict[str, object]:
    """把向量档的**意图与实际可用性**说清楚（供 /readyz 与日志使用）。

    ⚠️ 这里回答的是「**会不会**降级」，不是「降级了没」——
    要回答后者必须真的构造一次模型，而那是启动期的一次重操作。
    本函数的定位是**探针**，必须是廉价的。

    ⚠️ 因此 ``may_fall_back`` 为真**不代表**真的降了级：
    云端 key 配了但网络不通时，它是真的；key 配了且网络通时，它是假的。
    这个区别必须写在返回值里而不是靠调用方猜 —— 探针把自己说不清的事情
    说成「已降级」，会让一次网络抖动被读成一次配置错误。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `dict[str, object]`: 含 ``provider`` / ``model`` / ``dimension`` /
        ``allow_fallback`` / ``may_fall_back`` / ``local_available`` 的字典。
        ⚠️ **不含任何密钥**。
    """
    from .local import local_embedding_available

    provider = settings.embedding.provider
    local_ok = local_embedding_available()

    may_fall_back = False
    if settings.embedding.allow_fallback:
        if provider == "dashscope":
            # ⚠️ 恒为 True，而且是**确定的 True**，不是「猜不准」：
            # 云端档能不能用取决于这次 HTTP 调用成不成，
            # 而探针不许发请求（见 docstring 的「必须廉价」）。
            # 所以从探针的位置看，「可能降级」是它能给出的**唯一诚实答案**。
            #
            # ⚠️ 不要改成 `not settings.llm.api_key`：
            # 那只覆盖了「缺 key」这一种不可用，漏掉网络不通、配额耗尽、
            # 模型下线 —— 而这些恰恰是线上更常见的。漏报的后果是
            # 运维看到 may_fall_back=false 而放心，同时线上正在跑假向量。
            may_fall_back = True
        elif provider == "local":
            # 本地档的可用性**可以**廉价判定（查包在不在），所以这里给的是
            # 确切的答案而不是保守的 True。
            may_fall_back = not local_ok
        # mock 档是链尾，没有下一档可降 —— 保持 False。

    return {
        "provider": provider,
        "model": (
            settings.embedding.model
            if provider == "dashscope"
            else settings.embedding.local_model
            if provider == "local"
            else "aligo-mock-embedding"
        ),
        "dimension": settings.embedding.dimension,
        "allow_fallback": settings.embedding.allow_fallback,
        "may_fall_back": may_fall_back,
        "local_available": local_ok,
    }


__all__ = [
    "EmbeddingUnavailableError",
    "build_embedding_model",
    "describe_embedding",
]
