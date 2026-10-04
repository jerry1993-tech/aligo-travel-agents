# -*- coding: utf-8 -*-
"""OpenTelemetry trace 装配 —— 把框架的埋点接到 Langfuse 上。

文件职责：
    构造并注册一个**真实的** ``TracerProvider``，把 span 通过 OTLP/HTTP 导出到
    Langfuse。装配是**条件式**的：缺少必要条件时明确告警并跳过，而不是装一个
    注定失败的导出器。

上下游依赖：
    - 上游：由 ``src/server/app.py`` 的 lifespan 在启动时调用
      :func:`setup_tracing`，关闭时调用 :func:`shutdown_tracing`；
      配置来自 :class:`src.config.schema.ObservabilitySettings`。
    - 下游：``agentscope.middleware.TracingMiddleware`` —— 它是**消费方**，
      只调用 ``opentelemetry.trace.get_tracer("agentscope", ...)``
      （见 ``agentscope/middleware/_tracing/_setup.py:11-19``），
      取到的 tracer 来自哪个 provider，完全取决于我们在这里注册了什么。

------------------------------------------------------------------------------
框架没有 setup_tracing —— 这个模块必须自己写
------------------------------------------------------------------------------
    ``TracingMiddleware`` 的注释里提到 "``setup_tracing`` was not called"，
    但**框架里并不存在这个函数**（``middleware/_tracing/`` 下只有
    ``_setup.py::_get_tracer``，它只是 ``trace.get_tracer(...)`` 的一层包装）。
    也就是说：tracer 的提供者是**使用方**的责任，本项目必须自己装配。

------------------------------------------------------------------------------
最危险的失效模式：静默短路（Silent Short-Circuit）
------------------------------------------------------------------------------
    框架在每个钩子开头都做同一个判断（``agentscope/middleware/_tracing/_trace.py:59-70``）::

        return isinstance(otel_trace.get_tracer_provider(), TracerProvider)

    只有当前**全局** provider 是 SDK 的 ``TracerProvider`` 时才真的埋点；
    否则——包括「只装了 opentelemetry-api 没装 SDK」「忘了 set_tracer_provider」
    「set 的时机早于 SDK 导入」——它只是把调用原样透传，**不报错、不打日志**。

    结果是一种最难归因的症状：应用跑得好好的，Langfuse 里一条 trace 都没有，
    而代码里确实写了埋点。因此本模块的硬要求是：

        · 装完之后立即自检（:func:`is_tracing_active`），不成立就打 WARNING；
        · 把真实的装配结果暴露给 ``/readyz``（:func:`describe_tracing`），
          让「trace 到底有没有在发」是一个**可观测**的事实，而不是一个信念。

------------------------------------------------------------------------------
401 不可重试 —— 为什么缺密钥时必须跳过而不是硬装
------------------------------------------------------------------------------
    Langfuse v3 的 OTLP 端点强制 HTTP Basic 认证。缺了认证头会得到 401，
    而 OTLP 导出器**只重试 408 与 5xx**（401 属于「重试也没用」的客户端错误），
    ⇒ 整批 trace 被直接丢弃。

    所以「密钥为空」时的两种做法后果完全不同：
        · 硬装导出器 —— 应用照常跑、日志反复报错、Langfuse 里一条没有；
        · 跳过并告警 —— 日志里一句话说清「为什么没在发」。
    本模块选后者，并在跳过时把原因写进返回值，供 ``/readyz`` 展示。
"""

from __future__ import annotations

import base64
import logging

from opentelemetry import trace as otel_trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

from ..config import Settings, get_settings

logger = logging.getLogger(__name__)

#: 导出批量上报的间隔（毫秒）。
#: 5 秒是「本地看 trace 时不用等太久」与「不要每个 span 一次 HTTP 请求」
#: 之间的折中。刻意**不**复用 ``observability.metrics_interval_seconds`` ——
#: 那个配置项的注释已明确说明它不驱动任何采集循环，拿它来决定 trace 的
#: 上报频率会让「改一个没人消费的配置，trace 行为却变了」这种事发生。
_EXPORT_SCHEDULE_DELAY_MS = 5000

#: 单次导出的超时（毫秒）。导出器在后台线程执行，这个超时不会阻塞请求。
_EXPORT_TIMEOUT_MS = 10000

#: 当前进程已注册的 provider（``None`` 表示未装配）。
#: 用它做幂等：``trace.set_tracer_provider`` 在进程内**只生效一次**，
#: 第二次调用会打一条 "Overriding of current TracerProvider is not allowed"
#: 的告警并**静默忽略**。若不加这层判断，重复装配时会得到
#: 「代码以为换了新配置、实际还是旧的」这种极难发现的偏差。
_provider: TracerProvider | None = None

#: 本进程是否已经关闭过 trace（见 :func:`shutdown_tracing`）。
#: 必须单独记录，不能靠 ``_provider is None`` 判断 —— 关闭之后
#: ``_provider`` 被清空，但**全局** provider 仍是那个已被 shutdown 的实例，
#: 于是框架的判据依旧返回 True、span 依旧被创建，只是全部被丢弃。
#: 这正是「以为在埋点、实际一个字节没发」的第二种成因。
_shutdown = False

#: 最近一次装配的结果说明（供 ``/readyz`` 与日志使用）。
_last_status: dict[str, object] = {
    "configured": False,
    "exporter": None,
    "reason": "尚未装配",
}


# ==============================================================================
# 内部工具
# ==============================================================================
def _basic_auth_header(public_key: str, secret_key: str) -> str:
    """拼出 OTLP 请求要带的 HTTP Basic 认证头。

    Basic 的格式是 ``Basic base64(user:pass)``，Langfuse 规定用户名 = public key、
    密码 = secret key（``.env.example`` 与 ``config/base.yaml`` 均有说明）。

    Args:
        public_key (`str`): Langfuse public key。
        secret_key (`str`): Langfuse secret key。

    Returns:
        `str`: 形如 ``Basic dXNlcjpwYXNz`` 的完整头值。

    Note:
        返回值含密钥，**不要**写进日志、span 属性或探针响应。
        它只作为 ``headers`` 传给导出器。
    """
    raw = f"{public_key}:{secret_key}".encode("utf-8")
    return "Basic " + base64.b64encode(raw).decode("ascii")


def _build_resource(settings: Settings) -> Resource:
    """构造 trace 的 Resource（描述「这些 span 是谁产生的」）。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `Resource`: 含 ``service.name`` 与 ``deployment.environment``。
    """
    return Resource.create(
        {
            # service.name 是 Langfuse / Jaeger 里分组的第一维度，缺了它会
            # 显示为 "unknown_service"，多个服务混在一起无法区分。
            "service.name": settings.observability.service_name,
            # 环境标签：dev / test / prod 的 trace 混在一个面板里几乎没法用，
            # 这个属性让它们可以在查询时被过滤掉。
            "deployment.environment": settings.app.env,
        },
    )


def _build_exporter(settings: Settings) -> SpanExporter | None:
    """按配置构造导出器；条件不满足时返回 ``None``（并已记录原因）。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `SpanExporter | None`: 构造好的导出器；跳过时返回 ``None``。
    """
    obs = settings.observability

    # ---- 1. 关闭档：这是默认值，也是最常见的情况，用 INFO 而不是 WARNING --
    if obs.trace_exporter == "none":
        _set_status(False, None, "trace_exporter=none，未装配导出器（默认行为）")
        logger.info("trace 未启用（trace_exporter=none）。")
        return None

    # ---- 2. 控制台档：本地排障用，不需要任何密钥 -------------------------
    if obs.trace_exporter == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter

        _set_status(True, "console", "导出到标准输出（仅用于本地排障）")
        logger.warning(
            "trace 导出到标准输出（trace_exporter=console）："
            "每批 span 都会打印到应用日志。仅用于本地排障，不要在生产开启。",
        )
        return ConsoleSpanExporter()

    # ---- 3. OTLP 档：需要端点 + 密钥，缺一不可 ---------------------------
    if not obs.otlp_endpoint.strip():
        _set_status(
            False,
            None,
            "trace_exporter=otlp 但 otlp_endpoint 为空，未装配",
        )
        logger.warning(
            "trace_exporter=otlp 但 otlp_endpoint 为空：已跳过 trace 装配。"
            "请设置 ALIGO__OBSERVABILITY__OTLP_ENDPOINT"
            "（Langfuse v3 固定为 http://langfuse:3000/api/public/otel/v1/traces）。",
        )
        return None

    missing_keys = [
        name
        for name, value in (
            ("langfuse_public_key", obs.langfuse_public_key),
            ("langfuse_secret_key", obs.langfuse_secret_key),
        )
        if not value.strip()
    ]
    if missing_keys:
        _set_status(
            False,
            None,
            f"缺少 {'/'.join(missing_keys)}，未装配（OTLP 401 不可重试）",
        )
        # ⚠️ 这是本项目最需要「大声说出来」的一条告警。缺密钥时不装导出器，
        # 后果是**Langfuse 里一条 trace 都没有**；而如果不说清楚原因，
        # 排查方向会被引向「埋点代码是不是没生效」，离真正的原因很远。
        logger.warning(
            "trace_exporter=otlp 但 %s 为空：已**跳过** trace 装配。"
            "原因：Langfuse v3 的 OTLP 端点要求 HTTP Basic 认证，"
            "缺认证头会返回 401，而 OTLP 导出器只重试 408/5xx —— "
            "401 会被直接丢弃且不重试，装上也收不到任何 trace。"
            "请设置 LANGFUSE_INIT_PROJECT_PUBLIC_KEY / "
            "LANGFUSE_INIT_PROJECT_SECRET_KEY（见 .env.example）。",
            "、".join(missing_keys),
        )
        return None

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    endpoint = obs.otlp_endpoint.strip()
    exporter = OTLPSpanExporter(
        endpoint=endpoint,
        headers={
            "Authorization": _basic_auth_header(
                obs.langfuse_public_key,
                obs.langfuse_secret_key,
            ),
        },
        timeout=_EXPORT_TIMEOUT_MS / 1000.0,
    )
    _set_status(True, "otlp", f"OTLP → {endpoint}")
    # 端点地址不是密钥，打印它对排障价值很大（域名写错是最常见的失误）。
    logger.info("trace 已启用：OTLP → %s（认证头已装配，密钥不打印）", endpoint)
    return exporter


def _set_status(configured: bool, exporter: str | None, reason: str) -> None:
    """记录最近一次装配结果（供 :func:`describe_tracing` 读取）。

    Args:
        configured (`bool`): 导出器是否真的装上了。
        exporter (`str | None`): 导出通道名。
        reason (`str`): 人类可读的说明（给人看，不含密钥）。
    """
    _last_status.clear()
    _last_status.update(
        {"configured": configured, "exporter": exporter, "reason": reason},
    )


# ==============================================================================
# 对外入口
# ==============================================================================
def setup_tracing(settings: Settings | None = None) -> TracerProvider | None:
    """装配 trace 导出（幂等）。

    应在应用启动时（lifespan 的进入段）调用一次。重复调用直接返回已有 provider，
    不会重复注册 —— 见 :data:`_provider` 的注释。

    Args:
        settings (`Settings | None`): 配置；``None`` 时取进程内单例。

    Returns:
        `TracerProvider | None`: 装配成功返回 provider；条件不满足（或已装配过）
            返回已有的 / ``None``。调用方**不需要**根据返回值决定后续行为，
            用 :func:`is_tracing_active` 判断更准确 —— 它反映的是框架真正会看到的
            那个全局状态。
    """
    global _provider, _shutdown

    if _provider is not None:
        return _provider

    if _shutdown:
        # 进程内 ``trace.set_tracer_provider`` 只能生效一次，关闭之后无法换新 ——
        # 这一点必须**明确报出来**，否则重启逻辑（比如测试里重建应用）会得到
        # 「装配函数返回了 None，我也不知道为什么」。
        logger.warning(
            "trace 已在本进程关闭过，无法重新装配："
            "OpenTelemetry 的全局 TracerProvider 在进程内只允许设置一次。"
            "如需重新启用 trace，请重启进程。",
        )
        return None

    settings = settings or get_settings()
    exporter = _build_exporter(settings)

    if exporter is None:
        # 没有导出器时**不注册** provider。
        # 理由：注册一个不带 exporter 的 SDK provider 会让
        # ``_check_tracing_enabled()`` 返回 True，于是框架开始真的创建 span
        # 并序列化属性 —— 全部开销照付，却一个字节都发不出去。
        # 不注册时框架走的是「原样透传」分支，开销忽略不计。
        return None

    provider = TracerProvider(resource=_build_resource(settings))
    provider.add_span_processor(
        BatchSpanProcessor(
            exporter,
            schedule_delay_millis=_EXPORT_SCHEDULE_DELAY_MS,
        ),
    )

    otel_trace.set_tracer_provider(provider)
    _provider = provider

    # ---- 装配后自检 ------------------------------------------------------
    # 这一步不是多余的：全局 provider 可能因为别处先 set 过而**没有**变成
    # 我们的 provider（set_tracer_provider 只生效一次且静默忽略后续调用）。
    # 不检查的话，我们会以为装配成功了，而框架的钩子仍然是空转的。
    if not is_tracing_active():
        # 只改「结果」两项，保留 _build_exporter 写好的 exporter 名字 ——
        # 导出器确实造出来了，失败的是「它没成为全局 provider」这一步，
        # 两者都保留才能看清发生了什么。
        _last_status["configured"] = False
        _last_status["reason"] = (
            "已注册 TracerProvider，但全局 provider 仍非 SDK provider"
            "（可能是别处已先行注册），框架埋点将静默跳过"
        )
        logger.warning(
            "trace 装配后自检未通过：全局 TracerProvider 不是本次注册的实例，"
            "agentscope 的 TracingMiddleware 会静默跳过所有埋点。"
            "通常原因是本函数被调用前已有代码注册过 provider。",
        )
    return provider


def shutdown_tracing() -> None:
    """关闭导出器并**冲刷**尚未上报的 span。

    必须在应用关闭时调用。``BatchSpanProcessor`` 是攒批上报的，
    进程直接退出会让缓冲区里最后一批 span 永久丢失 ——
    而这一批往往正是「关闭前发生了什么」的记录，恰恰是最需要的那部分。

    调用后 :func:`is_tracing_active` 转为 ``False``：关闭过的 provider 不会再
    导出任何数据，继续报告「trace 正常」只会误导排障。
    """
    global _provider, _shutdown
    if _provider is None:
        return
    try:
        _provider.shutdown()
    except Exception:  # pylint: disable=broad-except
        # 关闭阶段的异常不应影响进程退出（此时业务已停止，报错也无处可去）。
        # 用 warning 而不是静默吞掉：万一导出器一直失败，这里会留下痕迹。
        logger.warning("关闭 trace provider 时发生异常。", exc_info=True)
    finally:
        _provider = None
        _shutdown = True


def is_tracing_active() -> bool:
    """判断框架的埋点**是否真的会执行**。

    直接调用框架自己的判据 ``_check_tracing_enabled``，而不是在本模块里
    重新实现一遍同样的 ``isinstance`` 检查。理由：这里要回答的是
    「框架会怎么做」，唯一可靠的答案来自框架本身。自己抄一份的话，
    哪天框架改了判据（比如接受 ``ProxyTracerProvider``），我们这边会继续
    报告「已启用」，而实际早已静默短路 —— 那正是本模块最想避免的症状。

    Returns:
        `bool`: 埋点会真正执行返回 True。
    """
    # 关闭过的进程里，框架的判据仍返回 True（全局 provider 还是那个实例），
    # 但 span 已经发不出去了 —— 这种情况必须报 False，否则关停中的实例会在
    # /readyz 里显示「trace 正常」，把真正的症状掩盖成一个假的好消息。
    if _shutdown:
        return False

    try:
        from agentscope.middleware._tracing._trace import _check_tracing_enabled

        return bool(_check_tracing_enabled())
    except Exception:  # pylint: disable=broad-except
        # 私有路径万一变动（框架重构），退回到等价判断，保证探针不因此 500。
        try:
            return isinstance(otel_trace.get_tracer_provider(), TracerProvider)
        except Exception:  # pylint: disable=broad-except
            return False


def describe_tracing(settings: Settings | None = None) -> dict[str, object]:
    """导出 trace 的**只读**状态（供 ``/readyz`` 与启动日志使用）。

    Args:
        settings (`Settings | None`): 配置；``None`` 时取进程内单例。

    Returns:
        `dict`: 含配置意图、实际装配结果、以及框架层面是否真的会埋点。
            刻意不含任何密钥或认证头。
    """
    settings = settings or get_settings()
    obs = settings.observability
    return {
        "trace_exporter": obs.trace_exporter,
        "otlp_endpoint": obs.otlp_endpoint.strip() or None,
        # 「配置希望发」与「实际发得出去」是两件事，分别给出，避免误读。
        "otlp_ready": obs.otlp_ready,
        "credentials_configured": bool(
            obs.langfuse_public_key.strip() and obs.langfuse_secret_key.strip(),
        ),
        "exporter_configured": bool(_last_status.get("configured")),
        # 实际装配的通道。与上面的 trace_exporter（配置**想**用的通道）并列给出：
        # 两者不一致，说明这一行来自另一次装配（例如探针被注入了一份不同的配置），
        # 光看 reason 很难发现这一点。
        "exporter": _last_status.get("exporter"),
        "shutdown": _shutdown,
        "active": is_tracing_active(),
        "reason": str(_last_status.get("reason", "")),
    }


__all__ = [
    "describe_tracing",
    "is_tracing_active",
    "setup_tracing",
    "shutdown_tracing",
]
