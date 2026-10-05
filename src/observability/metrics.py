# -*- coding: utf-8 -*-
"""Prometheus 指标 —— 进程内的指标注册表与埋点辅助函数。

文件职责：
    定义本项目对外暴露的全部 Prometheus 指标，并提供一组**语义化**的埋点
    辅助函数（:func:`observe_http_request` / :func:`observe_model_call` …）。
    业务代码只调用这些辅助函数，不直接碰 ``prometheus_client``。

上下游依赖：
    - 上游：由 ``src/server/probes.py`` 的 ``/metrics`` 路由用
      :func:`render_latest` 实时渲染；由 ``src/server/app.py`` 的中间件调用
      :func:`observe_http_request` 记录每个请求。
    - 下游：``scripts/prometheus/prometheus.yml`` 抓取 ``/metrics``，
      Grafana 面板消费这些指标名。

------------------------------------------------------------------------------
为什么用**独立注册表**而不是全局默认 REGISTRY
------------------------------------------------------------------------------
    1. ``prometheus_client`` 的默认注册表在 import 时就自动挂上了
       ``process_*`` / ``python_gc_*`` / ``python_info`` 等一批采集器。
       其中 GC 采集器会产生上百条时序，对排障几乎没有价值，却会让
       ``/metrics`` 的响应体膨胀数倍、抓取变慢。
    2. 更关键的是**重复注册会抛异常**。测试里反复 import / 重建应用时，
       用全局注册表几乎必然撞上 ``Duplicated timeseries in CollectorRegistry``，
       而这类错误只在测试跑第二遍时出现，非常难查。
       自建注册表后，每个进程只有一份，语义清晰。

------------------------------------------------------------------------------
多 worker 的注意点（本项目为何保持 WORKERS=1）
------------------------------------------------------------------------------
    Prometheus 的拉取模型是「一个进程一份指标」。多 worker 时每个 worker 各持
    自己的计数器，``/metrics`` 只会命中其中一个 ⇒ 数字忽大忽小、看不出趋势。
    正确的多进程方案是 ``prometheus_client.multiprocess`` + 共享目录，代价是需要
    在 Gunicorn 的 ``child_exit`` 钩子里做清理。本项目任务书规定
    ``ALIGO__APP__WORKERS=1``（另有调度器重复触发的理由），因此不需要它。
    ⚠️ 若将来要开多 worker，**必须同时**改用 multiprocess 模式，否则面板会骗人。

------------------------------------------------------------------------------
指标命名约定
------------------------------------------------------------------------------
    一律 ``aligo_<子系统>_<度量>_<单位>``：
        · 计数器以 ``_total`` 结尾（Prometheus 的约定，否则 ``rate()`` 会告警）；
        · 直方图的单位写进名字（``_seconds``），不靠注释 —— 面板上没人看注释。
"""

from __future__ import annotations

import logging
from typing import Final

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)

logger = logging.getLogger(__name__)

#: 本模块专属的注册表（见模块文档字符串「为什么用独立注册表」）。
REGISTRY: Final[CollectorRegistry] = CollectorRegistry(auto_describe=True)

# 进程级采集器：内存、文件描述符、CPU。容器里排查「内存缓慢上涨」
# 「连接数打满」时最先看的就是它们，因此显式挂上。
# 刻意**不**挂 GC 采集器：它产生的时序最多，而 actionable 的信息最少。
ProcessCollector(registry=REGISTRY)
PlatformCollector(registry=REGISTRY)


# ==============================================================================
# HTTP 层
# ==============================================================================
#: 请求总数，按「方法 / 路由模板 / 状态码」三个维度切分。
#: ⚠️ 用的是**路由模板**（如 ``/api/v1/orders/{order_id}``）而不是真实路径 ——
#: 否则每个 order_id 都会变成一条独立的时序，一次压测就能产生上万条，
#: 直接打爆 Prometheus 的内存。这一点在中间件里落实（见 src/server/app.py）。
HTTP_REQUESTS_TOTAL: Final[Counter] = Counter(
    "aligo_http_requests_total",
    "HTTP 请求总数（按方法/路由模板/状态码切分）",
    labelnames=("method", "route", "status"),
    registry=REGISTRY,
)

#: 请求耗时。分桶按 Web 服务的现实分布设计：
#: 50ms 以内是「本地缓存命中」，100–500ms 是「一次数据库往返」，
#: 1–5s 是「一次带工具调用的模型往返」，10s 以上基本就是出问题了。
HTTP_REQUEST_DURATION_SECONDS: Final[Histogram] = Histogram(
    "aligo_http_request_duration_seconds",
    "HTTP 请求耗时（秒）",
    labelnames=("method", "route"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)

#: 当前正在处理中的请求数。用它减去 QPS 可以看出「是流量涨了还是变慢了」——
#: 单看 QPS 无法区分这两者，而处置方式完全不同（扩容 vs 查慢查询）。
HTTP_IN_FLIGHT: Final[Gauge] = Gauge(
    "aligo_http_in_flight_requests",
    "当前正在处理中的 HTTP 请求数",
    registry=REGISTRY,
)


# ==============================================================================
# 安全层（鉴权失败 / 被限流）
# ==============================================================================
#: 鉴权失败次数，按**原因**切分。
#: 原因只有有限的几个取值（见 src/server/middleware/auth.py 的 _AuthFailure），
#: 刻意**不**带上 user id 或 token 片段：那会让一条时序按调用者爆炸式增长，
#: 并且把凭据材料写进指标库 —— 而 /metrics 通常是**不需要鉴权**就能抓的。
#:
#: 为什么必须有这个计数器：鉴权失败在此之前只体现为一条 WARNING 日志，
#: 而日志不会触发任何告警。一个「某客户端配错了 token」和一次
#: 「有人在枚举用户身份」在日志里长得一模一样，只有计数器的**斜率**能区分。
AUTH_FAILURES_TOTAL: Final[Counter] = Counter(
    "aligo_auth_failures_total",
    "鉴权失败总数（按原因切分）",
    labelnames=("reason",),
    registry=REGISTRY,
)

#: 被限流拒绝的请求数，按**身份来源**切分（``user`` / ``ip``）。
#: 区分二者是有用的：``ip`` 占绝大多数说明限流退化成了按来源 IP 限流
#: （即身份没生效，多半是 Auth 中间件没跑到它前面），
#: 而这与「某个用户刷得太凶」是完全不同的两件事。
RATE_LIMITED_TOTAL: Final[Counter] = Counter(
    "aligo_rate_limited_total",
    "被限流拒绝的请求总数（按身份来源切分）",
    labelnames=("key_kind",),
    registry=REGISTRY,
)

#: 回复守卫的动作次数，按**动作**切分。
#:
#: 动作取值（见 :mod:`src.orchestration.reply_guard`）：
#:
#:     clean                  守卫**一字未改**地放过了正文事件（DELTA 拼接）。
#:                            ⚠️ 它描述的是「守卫有没有动手」，不是「上屏文本
#:                            一定像样」：形状表外的候选（见
#:                            ``residual_internal_marker``）与 ``TEXT_BLOCK_END``
#:                            覆盖载荷里的候选都可以与它并存 —— 那两件事各有
#:                            自己的指标，刻意不挤进 clean 的口径
#:                            （2026-10-05 与 ``stripped_internal_marker``
#:                            的互斥是另一回事：两者都描述「改没改」）。
#:     stripped               正文与模型原文不同：剥掉了开头/结尾的草稿段，
#:                            和/或**正文通道**（含 ``TEXT_BLOCK_END`` 覆盖
#:                            载荷）里的内部标记 —— 只洗了标记时本动作与
#:                            ``stripped_internal_marker`` 同时记。
#:                            ⚠️ 它只覆盖正文通道：仅思考链/工具载荷被清洗的
#:                            回复不记它（那两条通道改的不是正文）。
#:     stripped_internal_marker
#:                            混进上屏文本的框架内部标记（如
#:                            ``[tool_result cleared by postprune]``）已剥除
#:                            （2026-10-05 缺陷 E 新增）。**七条上屏且落库的
#:                            通道都记这里**：正文文本块、思考链（``ThinkingBlock*``
#:                            同样实时上屏且落库）、``TEXT_BLOCK_END`` 的
#:                            覆盖载荷、工具的两条增量通道（调用参数 /
#:                            返回文本 —— 按流「留尾」缓冲，跨 delta 分片的
#:                            标记在配对 End 事件处整段结算，见
#:                            ``reply_guard._feed_tool_stream``）、工具名
#:                            （``tool_call_name``：模型可写、前端当执行链标题、
#:                            还随 ``ToolCallBlock``/``ToolResultBlock`` 落库，
#:                            同一个名字在调用块与结果块各记一次）、
#:                            ``DataBlockStartEvent`` 的 ``name``/``media_type``、
#:                            ``HintBlockEvent.hint``（``str`` 或
#:                            ``TextBlock`` 列表都洗；后三条通道由第四轮对抗
#:                            验证的 completeness critic 补上）。剥法是
#:                            接缝级（只吃标记两侧空白，不动行文）。任一通道
#:                            剥过标记的轮次**不再记 clean**（clean 的语义是
#:                            「一个字符都没改」）。工具名与 ``name``/
#:                            ``media_type`` 走**宽判据**：半截标记头也整名
#:                            清空（受控短标识，不存在误伤真内容的问题）。——
#:                            「模型开始复读标记」这件事全靠它可见：标记
#:                            本身被删掉了，正文与日志里都留不下痕。
#:     residual_internal_marker
#:                            **上屏文本里仍有**形状表外的标记候选
#:                            （``[… cleared up …]`` / ``【…】`` 这类词表外、
#:                            括号外变种），**只计数不删**（2026-10-05 对抗
#:                            验证 F7 新增；此前模块文档误称这类变种会落进
#:                            ``draft_paragraph_left_*``，实测不成立）。
#:                            扫描覆盖**五条已发出通道**：正文、思考链、END
#:                            覆盖载荷、工具载荷（含结尾处确信的半截标记头 ——
#:                            载荷属执行链可见性，不扣尾、只计数）、
#:                            ``HintBlockEvent.hint`` 文本；另加两处剥离前
#:                            就已知的补记：内容超长的成对 system 标签、
#:                            回复结束时仍扣在手里的确信族半截前缀。
#:                            ⚠️ 同一个碎片**只记一笔**（随文本上屏的由扫描
#:                            数，显式那一笔只在扫描认不出该形状时补）。
#:                            它抬头 = 该扩形状表了；和 WARNING 日志配套
#:     dropped_tool_round_narration / dropped_tool_round_other
#:                            工具轮的文字整轮丢弃，按**形状**分两类：
#:                            narration = 判据（草稿 / 占位话术）认得它，
#:                            是提示词要治的那一类；other = 判据认不出来，
#:                            **要盯住**（工具轮的文字是静默丢弃的，里面
#:                            可能夹着本该给用户的答案）。
#:                            ⚠️ 2026-10-04 由单一动作 ``dropped_tool_round``
#:                            拆开：一条曲线答不了「提示词改得有没有效」。
#:     kept_confirmation_round 工具轮含写操作，文字按用户复述保留
#:     rejected_empty / rejected_placeholder / rejected_monologue /
#:     rejected_ungrounded_limit
#:                            候选正文不可用，未发给用户
#:     retry_empty / retry_placeholder / retry_monologue /
#:     retry_ungrounded_limit / retry_skipped_structured /
#:     retry_skipped_no_budget
#:                            已要求模型重说（按当时的原因分类；
#:                            retry_skipped_* 是**没能**重说：结构化输出回复
#:                            或无重试预算，此时 ``rejected_*`` 与
#:                            ``retry_*`` 的差值会拉大）
#:     fallback_from_draft    重试用尽，用最后一版草稿的剥离结果兜底
#:     fallback_skipped_ungrounded
#:                            草稿因事实接地闸门被弃（含「已被闸门拦下过」的
#:                            既往记录），改用下一段或固定话术
#:     fallback_generic       重试用尽且草稿也救不回来，用固定话术兜底
#:     draft_paragraph_left_internal / draft_paragraph_left_monologue
#:                            发给用户的正文里仍有可疑段落（本模块未覆盖的
#:                            形状），按判据分类：internal = 含工具名/智能体名
#:                            等内部实现名（用户可见的泄露，该做名字清洗）；
#:                            monologue = 独白 / 过程叙述（该压提示词）。
#:                            ⚠️ 2026-10-04 由单一动作 ``draft_paragraph_left``
#:                            拆开，两类之和 = 旧口径（
#:                            ``reply_guard._classify_suspicious_paragraphs``
#:                            的并集断言钉住了这一点）。
#:     ungrounded_amount      答复里的金额在同轮工具返回中找不到出处
#:
#: 为什么这个计数器是这个模块最重要的产出：**「剥离草稿」的效果无法从日志
#: 里看出来**（被剥掉的文本压根不会出现在回复里），只有这条曲线能回答
#: 「这套闸门到底在拦什么、拦了多少、有没有把真答案也拦掉」。
#: ``rejected_*`` 与 ``retry_*`` 的差值持续为 0 说明模型改不动，该回头看提示词；
#: ``fallback_generic`` 一旦抬头说明发生了「用户什么都没拿到」，需要告警。
REPLY_GUARD_ACTIONS_TOTAL: Final[Counter] = Counter(
    "aligo_reply_guard_actions_total",
    "回复守卫的动作次数（按动作切分）",
    labelnames=("action",),
    registry=REGISTRY,
)


# ==============================================================================
# 模型层
# ==============================================================================
#: 模型调用总数。``outcome`` 取值只有三种，刻意不按异常类型细分：
#:    ok       —— 正常返回
#:    error    —— 调用了但失败（网络/限流/服务端错误）
#:    rejected —— **被熔断器拦下，根本没发出去**
#: 把 rejected 单列是关键：它和 error 的处置方式完全不同，
#: 而「error 数很高」既可能是下游挂了，也可能是我们自己在熔断，必须能区分。
MODEL_CALLS_TOTAL: Final[Counter] = Counter(
    "aligo_model_calls_total",
    "模型调用总数（按模型名与结果切分）",
    labelnames=("model", "outcome"),
    registry=REGISTRY,
)

#: 模型调用耗时。分桶明显比 HTTP 更靠后：一次模型往返几秒是常态。
#: 若沿用 HTTP 的分桶（上限 10s），几乎所有观测都会落进最后一个桶，
#: 直方图退化成计数器，算不出分位数。
MODEL_CALL_DURATION_SECONDS: Final[Histogram] = Histogram(
    "aligo_model_call_duration_seconds",
    "模型调用耗时（秒）",
    labelnames=("model",),
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0),
    registry=REGISTRY,
)

#: token 消耗（按方向切分）。成本面板的基础数据。
MODEL_TOKENS_TOTAL: Final[Counter] = Counter(
    "aligo_model_tokens_total",
    "模型 token 消耗总数（按模型名与方向切分）",
    labelnames=("model", "direction"),
    registry=REGISTRY,
)

#: 熔断器状态：0=closed，1=half_open，2=open。
#: 用**数值**而不是标签值，是因为标签值一变就会产生一条新时序，
#: 而 Grafana 里画「状态随时间变化」需要的是同一条时序上的数值跳变。
MODEL_BREAKER_STATE: Final[Gauge] = Gauge(
    "aligo_model_breaker_state",
    "熔断器状态（0=closed, 1=half_open, 2=open）",
    labelnames=("name",),
    registry=REGISTRY,
)


# ==============================================================================
# 就绪探针
# ==============================================================================
#: 就绪探针结果：1=全部就绪，0=有依赖不可用。
#: 与 ``/readyz`` 的 HTTP 状态码是同一件事的两种暴露方式 ——
#: 状态码给编排系统（K8s / compose）看，这个指标给告警规则看。
READY: Final[Gauge] = Gauge(
    "aligo_ready",
    "就绪探针结果（1=就绪, 0=未就绪）",
    registry=REGISTRY,
)


# ==============================================================================
# 埋点辅助函数
# ==============================================================================
#: 熔断状态 → 数值的映射。放在这里而不是各调用点，保证只有一个真值。
_BREAKER_STATE_VALUES: Final[dict[str, int]] = {
    "closed": 0,
    "half_open": 1,
    "open": 2,
}


def observe_http_request(
    method: str,
    route: str,
    status: int,
    duration_seconds: float,
) -> None:
    """记录一次 HTTP 请求。

    Args:
        method (`str`): HTTP 方法。
        route (`str`): **路由模板**（如 ``/api/v1/orders/{order_id}``），
            不是真实路径 —— 原因见 :data:`HTTP_REQUESTS_TOTAL` 的注释。
        status (`int`): HTTP 状态码。
        duration_seconds (`float`): 处理耗时（秒）。
    """
    HTTP_REQUESTS_TOTAL.labels(
        method=method,
        route=route,
        status=str(status),
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(method=method, route=route).observe(
        duration_seconds,
    )


def observe_model_call(
    model: str,
    outcome: str,
    duration_seconds: float,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> None:
    """记录一次模型调用。

    Args:
        model (`str`): 模型名。
        outcome (`str`): ``ok`` / ``error`` / ``rejected``（含义见
            :data:`MODEL_CALLS_TOTAL` 的注释）。
        duration_seconds (`float`): 耗时（秒）。被熔断拒绝时应传 0 ——
            那种情况根本没发请求，把它算进延迟会污染分位数。
        input_tokens (`int`): 输入 token 数；未知传 0。
        output_tokens (`int`): 输出 token 数；未知传 0。
    """
    MODEL_CALLS_TOTAL.labels(model=model, outcome=outcome).inc()
    # 被熔断拒绝的调用没有耗时可言，跳过直方图观测。
    # 若照记 0 秒，P50 会被大量 0 拉低，看起来「模型变快了」——正好相反。
    if outcome != "rejected":
        MODEL_CALL_DURATION_SECONDS.labels(model=model).observe(
            duration_seconds,
        )
    if input_tokens:
        MODEL_TOKENS_TOTAL.labels(model=model, direction="input").inc(
            input_tokens,
        )
    if output_tokens:
        MODEL_TOKENS_TOTAL.labels(model=model, direction="output").inc(
            output_tokens,
        )


def observe_breaker_state(name: str, state: str) -> None:
    """更新熔断器状态指标。

    Args:
        name (`str`): 熔断器名字。
        state (`str`): ``closed`` / ``half_open`` / ``open``；未知值按 ``closed`` 处理。
    """
    MODEL_BREAKER_STATE.labels(name=name).set(
        _BREAKER_STATE_VALUES.get(state, 0),
    )


def set_ready(ready: bool) -> None:
    """更新就绪指标。

    Args:
        ready (`bool`): 是否就绪。
    """
    READY.set(1 if ready else 0)


def observe_auth_failure(reason: str) -> None:
    """记录一次鉴权失败。

    Args:
        reason (`str`): 失败原因，取值为
            :mod:`src.server.middleware.auth` 里 ``_AuthFailure`` 常量的值。
            传未登记的值不算错（Prometheus 会照记一条新时序），
            但会让面板上的分组失去意义 —— 新增原因时请同时补进那里的常量表。
    """
    AUTH_FAILURES_TOTAL.labels(reason=reason).inc()


def observe_rate_limited(key_kind: str) -> None:
    """记录一次被限流拒绝的请求。

    Args:
        key_kind (`str`): ``user``（按身份限流）或 ``ip``（身份缺失、回落到 IP）。
            二者的比例失衡说明鉴权链路没生效，见 :data:`RATE_LIMITED_TOTAL`。
    """
    RATE_LIMITED_TOTAL.labels(key_kind=key_kind).inc()


def observe_reply_guard(action: str) -> None:
    """记录一次回复守卫的动作。

    Args:
        action (`str`): 动作名，取值见 :data:`REPLY_GUARD_ACTIONS_TOTAL`。
            传未登记的值不算错（Prometheus 会照记一条新时序），
            但会让面板上的分组失去意义 —— 新增动作时请同时补进那里的注释。

    ⚠️ 动作名里**不要**带用户 id、会话 id、回复 id 或任何回复正文片段：
    ``/metrics`` 通常无需鉴权即可抓取，把回复正文写进去等于把用户数据
    暴露在一个没有访问控制的口子上。
    """
    REPLY_GUARD_ACTIONS_TOTAL.labels(action=action).inc()


# ==============================================================================
# 渲染
# ==============================================================================
def render_latest() -> tuple[bytes, str]:
    """渲染 ``/metrics`` 的响应体。

    Returns:
        `tuple[bytes, str]`: ``(响应体, Content-Type)``。
            Content-Type 必须用 prometheus_client 提供的常量 —— 它带有
            ``version=0.0.4`` 的协商参数，Prometheus 依此选择解析器；
            手写 ``text/plain`` 会在某些版本上被拒绝。
    """
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


__all__ = [
    "AUTH_FAILURES_TOTAL",
    "HTTP_IN_FLIGHT",
    "HTTP_REQUESTS_TOTAL",
    "HTTP_REQUEST_DURATION_SECONDS",
    "MODEL_BREAKER_STATE",
    "MODEL_CALLS_TOTAL",
    "MODEL_CALL_DURATION_SECONDS",
    "MODEL_TOKENS_TOTAL",
    "RATE_LIMITED_TOTAL",
    "READY",
    "REGISTRY",
    "observe_auth_failure",
    "observe_breaker_state",
    "observe_http_request",
    "observe_model_call",
    "observe_rate_limited",
    "render_latest",
    "set_ready",
]
