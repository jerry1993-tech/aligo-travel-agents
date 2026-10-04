# -*- coding: utf-8 -*-
"""健康探针与指标端点：``/healthz``（存活）、``/readyz``（就绪）、``/metrics``。

文件职责：
    提供**编排系统**（Docker compose / Kubernetes / 负载均衡）与**监控系统**
    （Prometheus）所需的那组端点。它们是运维面的唯一入口，因此这里的行为
    必须与 ``docker-compose.yaml`` / ``Dockerfile`` 的探针契约**逐字一致**。

上下游依赖：
    - 上游：由 ``src/server/app.py`` 在装配根应用时 ``include_router``。
      配置来自 :class:`src.config.schema.Settings`。
    - 下游：``docker-compose.yaml:431`` 的 app 健康检查打 ``/readyz``；
      ``Dockerfile`` 镜像级 HEALTHCHECK 打 ``/healthz``；
      ``scripts/prometheus/prometheus.yml`` 抓 ``/metrics``；
      ``scripts/smoke.py`` 三个都打。

------------------------------------------------------------------------------
三个端点的语义差别（这是运维面最容易被做错的地方）
------------------------------------------------------------------------------
    /healthz   **存活**。零 I/O，只看「进程还在不在事件循环里」。
               它绝不能去连数据库 —— 若依赖挂了就报不健康，编排系统会把一个
               完全正常的进程反复杀掉重启，而重启解决不了「PG 挂了」，
               只会让故障现场（内存、日志、连接）每次都被清掉。

    /readyz    **就绪**。真实地去连 PostgreSQL / Redis / Milvus，全绿才 200。
               它回答的是「现在把流量打给我，我能正确服务吗」。
               失败时返回 503 + **逐项明细**，让 503 自己说清是哪一项挂了 ——
               「就绪探针失败」这五个字对排障几乎无用。

    /metrics   Prometheus 拉取端点。**每次抓取时实时渲染**，不是预先算好的快照。

------------------------------------------------------------------------------
为什么把依赖检查放在 /readyz 而不是 /healthz
------------------------------------------------------------------------------
    这正是 compose 与镜像**故意用两个不同端点**的原因（见
    ``docker-compose.yaml:412-427``）：
        · 镜像里的 /healthz 服务于 `docker run` 单跑（依赖不在，就绪必然失败）；
        · compose 里的 /readyz 服务于 `make up --wait` 的验收闸门。
    `make up --wait` **只等 healthy**，所以把 compose 探针设成 /readyz，
    就等价于让「make up 返回成功」这件事本身成为「PG + Redis + Milvus 真的都通了」
    的证据。若两边都用 /healthz（零 I/O），验收会变成一张空头支票。

------------------------------------------------------------------------------
超时预算（改这里必须同步改 compose）
------------------------------------------------------------------------------
    每项检查 3 秒上限（:data:`CHECK_TIMEOUT_SECONDS`），三项**并发**执行
    ⇒ ``/readyz`` 最坏耗时约 3 秒。
    因此 ``docker-compose.yaml:433`` 的 healthcheck ``timeout`` 写的是 10s
    （必须 ≥ 3s，且留出 curl 建连与网络开销的余量）。改本文件的超时上界时，
    **必须同步复核那个 10s**，否则探针会先被 docker 掐断，我们精心构造的
    「哪一项失败」的明细就永远送不出去 —— 症状是 compose 里只有一句
    无信息量的 "health check timeout"。
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable
from urllib.parse import urlparse

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

from ..config import Settings, get_settings
from ..observability import metrics as metrics_mod
from ..observability import describe_tracing
from ..observability.redaction import redact, safe_error
from .constants import BOOT_COMPLETED_ATTR

logger = logging.getLogger(__name__)

#: 单项依赖检查的超时上限（秒）。
#: 取值理由：本地容器网络下，一次成功的 PG 连接 + SELECT 1 只要几毫秒；
#: 3 秒足够覆盖「冷启动建连」与「轻微抖动」，又远小于任何编排系统的探针超时。
#: 太大（比如 30s）会让一个真挂了的依赖把探针拖成「一直 pending」。
CHECK_TIMEOUT_SECONDS = 3.0

#: 会**阻断就绪**的检查项。
#:
#: ⚠️ ``milvus`` **刻意不在这里**（P4 验收：「Milvus 不可用不得拖垮 /readyz」）。
#:
#: 理由不是「Milvus 不重要」，而是**就绪的含义**：``/readyz`` 回答的是
#: 「这个实例现在能不能接流量」。而 Milvus 只被「查差旅政策」这一条路径用到 ——
#: 对话、订单、审批、鉴权、会话全部不经过它。Milvus 挂了还把这个实例
#: 从 LB 后面摘掉，等于用「政策问答不可用」的代价换来「整个服务不可用」，
#: 是净损失。
#:
#: ⚠️ 检查本身**仍然执行**，结果仍然出现在 ``/readyz`` 的响应体里 ——
#: 只是不再参与 ready 判定。所以运维依然能一眼看到它挂了：
#: ``{"status": "ok", "ready": true, "checks": {"milvus": {"ok": false}}}``。
#: 「不阻断」不等于「不报告」；把这条区别丢掉，Milvus 的故障就会变成
#: 一个没人知道的静默降级。
#:
#: ⚠️ Postgres / Redis / boot 三者不同：它们分别在存储、事件总线、
#: 启动完成这三条**所有请求都会经过**的路径上，挂了就是真的不能服务。
REQUIRED_CHECKS: frozenset[str] = frozenset(
    {"postgres", "redis", "boot"},
)

#: 脱敏工具。
#:
#: ⚠️ 实现已挪到 :mod:`src.observability.redaction`。原因：P4 的向量库探针
#: （``src/knowledge/store.py``）也要用它，而 ``src/knowledge/`` 依赖
#: ``src/server/`` 是**反的** —— 知识层不该知道有一个 HTTP 服务层存在。
#:
#: 这两个私有别名保留下来，是为了让本模块内已有的 ``_redact(...)`` /
#: ``_safe_error(...)`` 调用点一行都不用改（改调用点是纯粹的噪音，
#: 且容易在改动中漏掉某一处）。新代码请直接 import 那个中立模块。
_redact = redact
_safe_error = safe_error


@dataclass
class CheckResult:
    """单项依赖检查的结果。

    Attributes:
        name (`str`): 检查项名（``postgres`` / ``redis`` / ``milvus`` / ``boot``）。
        ok (`bool`): 是否通过。
        detail (`str`): 人类可读的说明；失败时是原因，成功时是简短佐证。
        duration_ms (`float`): 耗时（毫秒）。**成功时也要记** ——
            依赖「还能连上但已经变得很慢」往往比「直接连不上」更早出现，
            而那正是需要提前扩容的信号。
        required (`bool`): 是否参与就绪判定（见 :data:`REQUIRED_CHECKS`）。
    """

    name: str
    ok: bool
    detail: str = ""
    duration_ms: float = 0.0
    required: bool = True

    def as_dict(self) -> dict[str, object]:
        """转成可直接 JSON 序列化的字典。

        Returns:
            `dict`: 该检查项的可序列化表示。
        """
        return {
            "ok": self.ok,
            "detail": self.detail,
            "duration_ms": round(self.duration_ms, 2),
            "required": self.required,
        }


@dataclass
class ReadinessReport:
    """``/readyz`` 的完整结果。

    Attributes:
        ready (`bool`): 全部**必需**检查是否都通过。
        checks (`list[CheckResult]`): 逐项结果。
        extra (`dict`): 附带的只读运行态信息（模型目标 / trace / 熔断器），
            一律不含密钥。
    """

    ready: bool
    checks: list[CheckResult] = field(default_factory=list)
    extra: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        """转成可直接 JSON 序列化的字典。

        Returns:
            `dict`: 含 ``status`` / ``ready`` / ``checks`` / 以及各段附加信息。
        """
        return {
            # status 只有两个取值，与 HTTP 状态码一一对应（200 / 503）。
            # 不做「degraded」这第三态：编排系统只看状态码，多一个中间态
            # 只会让「到底该不该摘流量」这件事产生分歧。
            "status": "ok" if self.ready else "not_ready",
            "ready": self.ready,
            "checks": {c.name: c.as_dict() for c in self.checks},
            **self.extra,
        }


# ==============================================================================
# 各依赖的检查实现
# ==============================================================================
async def _check_postgres(settings: Settings) -> CheckResult:
    """检查 PostgreSQL：建一条**新**连接并执行 ``SELECT 1``。

    为什么每次都用**短命 engine**（NullPool + 用完 dispose）而不是复用一个长期
    连接池：
        · **探针要测的正是「现在能不能新建连接」**。复用一个池化的连接只能证明
          「曾经连上过」，恰恰漏掉了最常见的故障——连接池里全是陈旧连接、
          PG 已经重启、认证凭据已失效。用 NullPool 每次真实建连才能测到。
        · 探针每 15 秒才跑一次，多一次 TCP + 认证握手的代价可以忽略；
          而一个长期存活的探针 engine 反而是连接泄漏的隐患（探针出异常时
          dispose 不一定被执行到）。
    ⚠️ NullPool 在这里是**必须**的，不是优化：默认池会把连接缓存下来，
    于是「第一次探针成功」之后，后续每次都在测那条被缓存的旧连接。

    Args:
        settings (`Settings`): 全量配置（取 ``db.url``）。

    Returns:
        `CheckResult`: 检查结果。
    """
    started = time.perf_counter()
    engine = None
    try:
        from sqlalchemy import text
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import NullPool

        engine = create_async_engine(
            settings.db.url,
            poolclass=NullPool,
            echo=False,
            # ⚠️ 探针 engine **刻意不复用** ``src/storage/engine.py`` 的
            # ``storage_engine_kwargs()``（探针要的是 NullPool + 短命连接，
            # 与业务引擎的池化参数正好相反），所以它必须自己带上这一条 ——
            # 否则它就是全仓**唯一**没有这道保护的 engine。
            # 今天这里只跑无绑定参数的 ``SELECT 1``，因此还没东西可漏；
            # 但探针的异常文本是直接进 ``/readyz`` 响应体的（见下面的
            # ``_safe_error``），谁将来在这里加一条带参数的语句，
            # 绑定值就会明文进响应体。这条约定由
            # ``tests/test_storage_engine.py`` 的契约用例守着。
            hide_parameters=True,
        )
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return CheckResult(
            name="postgres",
            ok=True,
            detail="SELECT 1 成功",
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:  # pylint: disable=broad-except
        # 捕获所有异常是**刻意**的：探针绝不能自己抛出去。否则 /readyz 会变成
        # 500 而不是 503，而 500 在编排系统看来是「应用有 bug」，
        # 排查方向会被引向代码而不是依赖。
        return CheckResult(
            name="postgres",
            ok=False,
            detail=_safe_error(exc),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    finally:
        if engine is not None:
            # dispose 也必须被保护：连接失败时 engine 可能处于半初始化状态，
            # 它的 dispose 也可能抛错。探针的清理逻辑不能反过来变成故障源。
            try:
                await engine.dispose()
            except Exception:  # pylint: disable=broad-except
                logger.debug("dispose 探针 engine 时抛出异常，已忽略。", exc_info=True)


async def _check_redis(settings: Settings) -> CheckResult:
    """检查 Redis：``PING`` 一次。

    用 ``PING`` 而不是 ``SET/GET``：探针不应在别人的数据里留下垃圾键，
    而 PING 已经足够证明「连接可用 + 认证通过 + 服务在响应」。

    Args:
        settings (`Settings`): 全量配置（取 ``redis.url``）。

    Returns:
        `CheckResult`: 检查结果。
    """
    started = time.perf_counter()
    client = None
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(
            settings.redis.url,
            socket_timeout=settings.redis.socket_timeout_seconds,
            socket_connect_timeout=settings.redis.socket_timeout_seconds,
        )
        pong = await client.ping()
        if not pong:
            # PING 返回假值说明拿到了一个非预期响应（比如打到了别的服务端口）。
            # 把它当失败处理而不是忽略 —— 否则一个「连上了但根本不是 Redis」的
            # 地址会被报成就绪。
            return CheckResult(
                name="redis",
                ok=False,
                detail="PING 未返回真值",
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        return CheckResult(
            name="redis",
            ok=True,
            detail="PING → PONG",
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return CheckResult(
            name="redis",
            ok=False,
            detail=_safe_error(exc),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    finally:
        if client is not None:
            try:
                # redis-py 5+ 用 aclose()；close() 已废弃且会打 DeprecationWarning。
                await client.aclose()
            except Exception:  # pylint: disable=broad-except
                logger.debug("关闭探针 Redis 连接时抛出异常，已忽略。", exc_info=True)


async def _check_milvus(settings: Settings) -> CheckResult:
    """检查 Milvus：对它的 gRPC 端口建一条 TCP 连接后立即关闭。

    为什么用**裸 TCP** 而不是 ``pymilvus.MilvusClient(...).list_collections()``：
        · 探针要回答的是「Milvus 的可达性」，不是「我们的 schema 对不对」。
          schema 问题应当由业务请求暴露，而不是让整个实例被判定为未就绪。
        · ``pymilvus`` 的连接带全局状态与后台线程，在一个每 15 秒就执行一次的
          探针里反复建/拆它，是给自己制造资源泄漏与「偶发卡住」的机会。
        · 裸 TCP 的依赖面最小：只要 asyncio，不需要 pymilvus 正常工作。
          若 pymilvus 自身出了问题（版本不兼容等），我们仍然能区分
          「Milvus 挂了」与「客户端库坏了」这两件事。

    Args:
        settings (`Settings`): 全量配置（取 ``milvus.uri``）。

    Returns:
        `CheckResult`: 检查结果。
    """
    started = time.perf_counter()

    # uri 形如 http://milvus:19530 —— 必须解析出 host/port 再建 TCP，
    # 不能把整个 URL 直接交给 open_connection。
    parsed = urlparse(settings.milvus.uri)
    host = parsed.hostname
    # 未显式给端口时回落到 Milvus 的 gRPC 默认端口（19530）。
    port = parsed.port or 19530

    if not host:
        return CheckResult(
            name="milvus",
            ok=False,
            detail=f"milvus.uri 解析不出主机名：{settings.milvus.uri!r}",
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    writer = None
    try:
        # open_connection 成功即证明「TCP 三次握手完成、端口在监听」。
        # 立即关闭：探针不应持有任何长连接，否则它会自己变成被监控对象。
        _, writer = await asyncio.open_connection(host, port)
        return CheckResult(
            name="milvus",
            ok=True,
            detail=f"TCP {host}:{port} 可连接",
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return CheckResult(
            name="milvus",
            ok=False,
            detail=_safe_error(exc),
            duration_ms=(time.perf_counter() - started) * 1000,
        )
    finally:
        if writer is not None:
            try:
                writer.close()
                # 等 close 真正完成，避免在事件循环结束时留下未关闭的传输层对象
                # （那会在日志里刷 "Unclosed transport" 警告）。
                # wait_closed 在连接已被对端重置时可能抛错，故同样要保护。
                await writer.wait_closed()
            except Exception:  # pylint: disable=broad-except
                logger.debug("关闭探针 Milvus 连接时抛出异常，已忽略。", exc_info=True)


def _check_boot(request: Request) -> CheckResult:
    """检查应用是否已完成启动（``app.state.boot_completed``）。

    这一项**不能省**，它是本项目自己加的、也是最容易漏掉的一环：
    ``/healthz`` 与 ``/readyz`` 都是 FastAPI 路由，而 FastAPI 在 lifespan 的
    **进入段执行之前**就已经开始接受连接了。也就是说存在一个真实的窗口期：
    进程在响应 HTTP，但 storage / message_bus / scheduler 都还没进异步上下文。

    此时若只查 PG/Redis/Milvus（它们都通），探针会报「就绪」，
    而第一个真实请求会因为 ``app.state.chat_service`` 尚不存在而 500。
    ``boot_completed`` 正是把这个窗口期显式地标出来。

    ⚠️ 框架的 ``agentscope.app._lifespan.lifespan`` **不会**设置这个属性
    （它只写 ``chat_service`` / ``session_service`` / ``scheduler_manager`` 等）。
    该标志由 ``src/server/app.py`` 包装 lifespan 时自行设置。

    Args:
        request (`Request`): 当前请求（用于访问 ``app.state``）。

    Returns:
        `CheckResult`: 检查结果。
    """
    started = time.perf_counter()
    # ⚠️ 用常量而不是字面量：这个键由 ``src/server/app.py`` 写、本模块读，
    # 中间隔着一次进程启动。手写字符串在有拼写错误时**不会**报错，
    # 只会让 getattr 的默认值 ``False`` 生效 —— 症状是
    # 「探针永远报未就绪」，而排查方向会被引到 lifespan 上去。
    completed = bool(getattr(request.app.state, BOOT_COMPLETED_ATTR, False))
    return CheckResult(
        name="boot",
        ok=completed,
        detail=(
            "lifespan 已完成启动"
            if completed
            else f"app.state.{BOOT_COMPLETED_ATTR} 为假："
            "应用仍在 lifespan 进入段中，或已进入退出段"
        ),
        duration_ms=(time.perf_counter() - started) * 1000,
    )


# ==============================================================================
# 就绪检查编排
# ==============================================================================
async def _run_check(
    name: str,
    factory: Callable[[], Awaitable[CheckResult]],
) -> CheckResult:
    """带超时地执行一项检查，并把超时也转成一条正常的 :class:`CheckResult`。

    Args:
        name (`str`): 检查项名（用于超时提示）。
        factory (`Callable[[], Awaitable[CheckResult]]`): 无参协程工厂。

    Returns:
        `CheckResult`: 检查结果；超时则 ``ok=False``。
    """
    try:
        return await asyncio.wait_for(factory(), timeout=CHECK_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        # 超时**不是**"未知"，而是一条明确的失败结论：3 秒都连不上就是不可用。
        # 把它与「连接被拒」分开报，因为两者的处置不同 ——
        # 被拒通常是对方没起来，超时通常是网络不通或对端卡死。
        return CheckResult(
            name=name,
            ok=False,
            detail=f"检查超时（>{CHECK_TIMEOUT_SECONDS:.0f}s）",
            duration_ms=CHECK_TIMEOUT_SECONDS * 1000,
        )


async def evaluate_readiness(request: Request) -> ReadinessReport:
    """并发执行全部依赖检查，汇总成就绪报告。

    三项 I/O 检查**并发**执行（总耗时 ≈ 最慢的一项，而不是三项之和），
    这是「单项 3 秒、合计约 3 秒」这一超时预算得以成立的前提。

    Args:
        request (`Request`): 当前请求。

    Returns:
        `ReadinessReport`: 就绪报告。
    """
    settings = get_settings()

    # asyncio.gather 而非顺序 await：三项检查互不依赖，串行执行会让
    # 最坏耗时变成 3×3=9 秒，直接顶穿 compose healthcheck 的 10 秒上限。
    results = await asyncio.gather(
        _run_check("postgres", lambda: _check_postgres(settings)),
        _run_check("redis", lambda: _check_redis(settings)),
        _run_check("milvus", lambda: _check_milvus(settings)),
        # boot 是纯内存判断，不涉及 I/O，但仍然走同一条路径：
        # 这样响应体里的每一项结构完全一致，消费方（smoke 脚本、面板）不需要特例。
        _run_check("boot", lambda: asyncio.sleep(0, result=_check_boot(request))),
    )

    checks = [
        CheckResult(
            name=r.name,
            ok=r.ok,
            detail=r.detail,
            duration_ms=r.duration_ms,
            required=r.name in REQUIRED_CHECKS,
        )
        for r in results
    ]
    ready = all(c.ok for c in checks if c.required)

    return ReadinessReport(
        ready=ready,
        checks=checks,
        extra={
            "model": _describe_model_safely(),
            "tracing": describe_tracing(settings),
            "breaker": _describe_breaker_safely(),
            # ⚠️ 与上面那个**并列而不是合并**：它们是两个独立的下游
            # （模型 / 向量库），熔断状态必须能分开看 —— 合并成一个字段的话，
            # 「模型好着、检索熔断了」这种情况在探针输出里根本读不出来。
            "vector_store_breaker": _describe_vector_store_breaker_safely(),
        },
    )


def _describe_model_safely() -> dict[str, object]:
    """取模型目标描述；失败时返回一条错误说明而不是抛出。

    Returns:
        `dict`: 模型目标描述，或 ``{"error": ...}``。
    """
    try:
        from ..llm import describe_model_target

        return describe_model_target()
    except Exception as exc:  # pylint: disable=broad-except
        # 探针的附加信息绝不能让探针本身失败。模型描述读不到（比如配置异常）
        # 与「依赖不通」是两件事，不该把实例判成未就绪。
        return {"error": _safe_error(exc)}


def _describe_breaker_safely() -> dict[str, object]:
    """取熔断器快照，并**顺带把状态同步进指标**。

    Returns:
        `dict`: 熔断器快照，或 ``{"error": ...}``。
    """
    try:
        from ..llm import get_breaker

        snapshot = get_breaker().snapshot()
        # 在这里（每次探针被调用时）刷新熔断状态指标，而不是在每个模型调用点刷新：
        # 熔断状态是一个**低频**变化的量，每次模型调用都去 set 一次纯属浪费；
        # 而探针本身就是周期性的，天然适合做这类状态的同步点。
        state = snapshot.get("state")
        name = snapshot.get("name")
        if isinstance(state, str) and isinstance(name, str):
            metrics_mod.observe_breaker_state(name, state)
        return snapshot
    except Exception as exc:  # pylint: disable=broad-except
        return {"error": _safe_error(exc)}


def _describe_vector_store_breaker_safely() -> dict[str, object]:
    """取**检索护栏**熔断器的快照，并同步进指标。

    ⚠️ 与模型熔断器分开报告，理由见 :func:`_readiness_report` 里的注释：
    它们保护的是两个互不相干的下游，「模型正常、检索已熔断」是一种**常态**
    （Milvus 挂了而 DashScope 好着），必须能被直接读出来。

    ⚠️ 这一项**不参与就绪判定** —— 与 ``milvus`` 检查项同一条原则
    （``REQUIRED_CHECKS`` 里没有它）：检索熔断只影响「查差旅政策」，
    对话、订单、审批、鉴权都不经过它。它出现在这里是为了**让人看得见**。

    Returns:
        `dict[str, object]`: 熔断器快照，或 ``{"error": ...}``。
    """
    try:
        from ..knowledge.guard import get_vector_store_breaker

        snapshot = get_vector_store_breaker().snapshot()
        # 与模型侧同一套指标（``aligo_model_breaker_state``，按 ``name`` 区分），
        # 于是 Grafana 上一条查询就能同时画两条熔断曲线。
        state = snapshot.get("state")
        name = snapshot.get("name")
        if isinstance(state, str) and isinstance(name, str):
            metrics_mod.observe_breaker_state(name, state)
        return snapshot
    except Exception as exc:  # pylint: disable=broad-except
        return {"error": _safe_error(exc)}


# ==============================================================================
# 路由
# ==============================================================================
router = APIRouter(tags=["ops"])


@router.get("/healthz", summary="存活探针（零 I/O）")
async def healthz() -> JSONResponse:
    """存活探针：只要进程还能响应就返回 200。

    **刻意不做任何 I/O**，理由见模块文档字符串。即使 PG / Redis / Milvus 全部
    不可达，这里也必须返回 200 —— 那三个是「就绪」的判据，不是「存活」的判据。

    Returns:
        `JSONResponse`: 恒为 200，体为 ``{"status": "ok"}``。
    """
    return JSONResponse(status_code=200, content={"status": "ok"})


@router.get("/readyz", summary="就绪探针（真实连 PG / Redis / Milvus）")
async def readyz(request: Request) -> JSONResponse:
    """就绪探针：全部必需依赖可用时 200，否则 503。

    响应体给出**逐项明细**（含失败原因与耗时），使 503 本身就能定位问题。

    Args:
        request (`Request`): 当前请求。

    Returns:
        `JSONResponse`: 就绪 200 / 未就绪 503。
    """
    report = await evaluate_readiness(request)

    # 同步到指标：让告警规则可以基于 aligo_ready 而不是「探针返回码」来写。
    metrics_mod.set_ready(report.ready)

    if not report.ready:
        failed = [c.name for c in report.checks if c.required and not c.ok]
        logger.warning(
            "就绪探针失败，未通过的必需项：%s",
            ", ".join(failed) or "<无>",
        )

    return JSONResponse(
        status_code=200 if report.ready else 503,
        content=report.as_dict(),
    )


@router.get("/metrics", summary="Prometheus 指标")
async def prometheus_metrics() -> Response:
    """Prometheus 拉取端点。

    受 ``observability.metrics_enabled`` 开关控制：关闭时返回 **404**
    （而不是 200 + 空体）。返回 404 会让 Prometheus 把该 target 标成 down，
    这是一个**显式**的信号；若返回 200 空体，抓取会显示成功而所有面板都是空的，
    排查方向被完全带偏。

    Returns:
        `Response`: 指标文本（200）或 404 说明。
    """
    settings = get_settings()
    if not settings.observability.metrics_enabled:
        return JSONResponse(
            status_code=404,
            content={
                "detail": "指标端点已关闭（observability.metrics_enabled=false）",
            },
        )

    body, content_type = metrics_mod.render_latest()
    return Response(content=body, media_type=content_type)


__all__ = [
    "CHECK_TIMEOUT_SECONDS",
    "REQUIRED_CHECKS",
    "CheckResult",
    "ReadinessReport",
    "evaluate_readiness",
    "router",
]
