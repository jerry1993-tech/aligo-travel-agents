#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""负载测试：对**已启动**的 AliGo 服务施加受控并发，量它的吞吐与尾延迟。

==============================================================================
它与 `make test` / `make smoke` / （milvus_init）的分工
==============================================================================
    `make test`         进程内，不连任何服务，测「代码逻辑对不对」
    `make smoke`        对已启动的服务**各打一发**，测「这一套部署能不能用」
    `make loadtest`     对已启动的服务**持续打压**，测「它在并发下什么表现」
    `scripts/milvus_init.py`  直接连 Milvus，测「那个集合按什么参数建出来的」

    冒烟只问「活着吗」（一次请求、是/否）；负载测试问的是**分布的尾部**：
    在 20 并发、几千个请求里，有多少失败、最慢的那 1% 有多慢。
    这两件事没有任何一个是另一个的推论 —— 一个冒烟全绿的服务，
    完全可能在并发下把 p99 拖到几秒，而单发请求永远测不出这一点。

==============================================================================
怎么读这份报告（每个数字的含义）
==============================================================================
    总请求 / 吞吐      吞吐 = 总请求 ÷ 实际耗时，单位 req/s。它回答「这台机器、
                       这个并发下，这个端点每秒能吃多少」。
                       ⚠️ 吞吐**不能跨端点比较**：``/healthz`` 是零 I/O 的，能跑到
                       几千 req/s；带模型调用的 ``/chat`` 被上游 LLM 的往返延迟主导，
                       通常只有个位数。拿两者的吞吐对着看，只会得出
                       「服务很慢」这种毫无信息量的结论。

    p50                「一半的请求比它快」，代表**典型**体感。
    p90 / p95          尾部的开始。p95 常被理解为「用户偶尔会碰到的卡顿」。
    p99                「最差的 1%」，SLO 最常用的那个点 —— 因为**平均值会被大量
                       快请求稀释**：「平均 50ms」完全可能同时意味着
                       「每 100 个请求里有一个卡了 3 秒」。
                       ⚠️ 本脚本用**最近秩（nearest-rank）**算法，直接从排序后的
                       真实样本里取，不做任何插值或估算 —— 样本里不存在的延迟值，
                       报告里就不会出现。（插值法会让 p50 落在两个样本**之间**，
                       比如 1..100 的 p50 会算成 50.5；对一个只做观测的工来说，
                       这只是把「我看不清」包装成了一个精确的假象。）

    失败率             = (非 2xx 样本数 + 传输层失败数) ÷ 总请求。
                       ⚠️ **3xx 计入失败**：负载测试打到重定向，多半意味着地址写错了
                       （少了结尾斜杠、或打到了 http 而非 https）。
                       ⚠️ **传输层失败与 5xx 是两回事**：5xx 是服务端处理不了，
                       传输层失败是**根本没连上**（服务没起、端口不对、或并发太高把
                       连接排队时间拖过了超时）。报告里把前者记成状态码、后者记成
                       「无」并单列，就是为了让这两类故障不被混为一谈。

==============================================================================
★ 单 worker 对吞吐上限意味着什么（别把结果读成"服务就能这么快"）
==============================================================================
    本项目**恒定单 worker**（``config/base.yaml`` 的 ``WORKERS=1``；且
    ``src/server/app.py`` 里 ``enable_scheduler=True`` 的注释写明多 worker
    会让定时任务被重复触发）。uvicorn 单进程 = 一个事件循环 = 单核在跑
    Python 字节码。因此：

      · 对 ``/healthz`` 这类**零 I/O** 端点，事件循环里没有阻塞点，
        吞吐上限由「单核处理速度 + 网络栈」决定，可以很高。
      · 对**任何碰 I/O 的**端点（查 PG / 连 Milvus / 调模型），单 worker 的
        吞吐上限 ≈ 1 ÷ 该请求里**串行**的那段耗时。异步只在「等 I/O」时让出
        执行权，真正占着 CPU 的那一段（JSON 序列化、模板渲染、Pydantic 校验）
        仍是**串行**的。
      · 所以本脚本量出的是**当前部署形态**的产数，不是「这个服务的能力上限」。
        它只有用在两处才有意义：(1) 同一端点改代码前后的对比；(2) 失败率 /
        尾延迟有没有越线。想抬上限，先扩容（多 worker / 多副本），再谈优化单请求。

    ⚠️ 并发参数本身也会改变结论：并发太低测不出瓶颈（事件循环闲着），并发太高会把
    连接排队的时间也算进延迟里（p99 立刻起飞）。健康的上限探测法是从小到大逐档加压
    （如 10 → 50 → 200），看「吞吐在哪里不再涨、而 p99 在哪里开始陡增」——
    那个拐点才是上限。一次跑一个大并发，得到的只是「那一刻的队列深度」。

==============================================================================
零依赖与自检
==============================================================================
    只用 httpx（冒烟脚本已依赖它），不引入任何压测框架。
    没有 Docker / 没有服务也能验证**本脚本自身**的逻辑：
        python scripts/loadtest.py --dry-run
    它会起一个**进程内**的最小 ASGI 应用当靶子，跑一遍真实的采样、分位与
    阈值判定 —— 验的是脚本，不是被压的服务。

退出码：**0 = 未突破任何阈值；1 = 有阈值被突破（会打印是哪一条）**。
    （与 Makefile 的 `loadtest` 目标配合，任一条越线即 `make loadtest` 失败。）
"""

from __future__ import annotations

import argparse
import asyncio
import math
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

# 允许以 `python scripts/loadtest.py` 直接运行（此时 sys.path[0] 是 scripts/）。
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.observability.redaction import safe_error  # noqa: E402

# ==============================================================================
# 常量
# ==============================================================================

#: 默认靶地址。与 Makefile 的 `LOADTEST_URL`（默认 http://localhost:8000）一致。
DEFAULT_BASE_URL = "http://localhost:8000"

#: `--health-only` 打的端点。**零 I/O、免鉴权、不碰模型**，是纯基础设施基线。
HEALTH_ENDPOINT = "/healthz"

#: 默认并发。20 对本地单 worker 服务是一个温和的起点：足以让事件循环不再空闲，
#: 又不会立刻把连接排队时间压进延迟里（那样量到的就主要是队列而不是服务）。
DEFAULT_CONCURRENCY = 20

#: 未显式指定 `--duration` / `--requests` 时的默认模式与时长（秒）。
DEFAULT_DURATION_SECONDS = 10.0

#: `--dry-run` 未指定模式时的默认请求数。刻意用**计数模式**而非计时模式：
#: 自检要的是「跑得完、几秒内结束」，而不是「压满 10 秒」。
DRY_RUN_REQUESTS = 300

#: 单次请求的超时（秒）。与 ``scripts/smoke.py`` 的取值一致 —— 见那里关于
#: /readyz 最坏耗时（约 3s）的说明；10s 留了 3 倍余量。
DEFAULT_REQUEST_TIMEOUT_SECONDS = 10.0

#: 默认阈值：失败率上限。1% 是「健康服务在负载下允许的毛刺」——
#: ⚠️ 对 ``--health-only`` 这种纯基线，可以且应当把它压到 0。
DEFAULT_MAX_FAILURE_RATE = 0.01

#: 默认阈值：p99 上限（毫秒）。1s 是「本地/内网打一个零 I/O 端点」的宽松上界：
#: 一旦越线，几乎必然是排队或线程/事件循环被卡住，而不是网络慢。
DEFAULT_MAX_P99_MS = 1000.0

#: 报告里展示的分位点（百分位）。
PERCENTILES: tuple[int, ...] = (50, 90, 95, 99)

#: 传输层失败在报告里的状态码标签（区别于真实的 4xx/5xx）。
NO_STATUS_LABEL = "无"


# ==============================================================================
# 结果模型
# ==============================================================================
@dataclass(frozen=True)
class Sample:
    """一次请求的观测结果。

    Attributes:
        latency_ms (`float`): 端到端耗时（毫秒），含读取响应体的时间。
        status (`int | None`): HTTP 状态码；``None`` 表示请求在**传输层**就失败了
            （连接被拒 / 超时 / DNS 失败），此时没有状态码可言。
        error (`str`): 传输层失败的原因；成功时为 ``None`` 的占位空串。
    """

    latency_ms: float
    status: int | None
    error: str = ""

    @property
    def ok(self) -> bool:
        """本次是否算成功。

        Returns:
            `bool`: 状态码落在 **2xx** 区间时为 ``True``。

        ⚠️ 3xx 与 4xx 一律算失败：负载测试不是浏览器，不跟随重定向，
            也不把「302 到登录页」当成一次成功的业务调用。
        """
        return self.status is not None and 200 <= self.status < 300


@dataclass
class LoadResult:
    """整轮负载测试的汇总。

    Attributes:
        url (`str`): 完整目标 URL（含端点路径）。
        concurrency (`int`): 并发上限。
        mode (`str`): ``duration``（按时间）或 ``requests``（按数量）。
        planned (`float | int | None`): 计划值（秒数或请求数），仅用于展示。
        samples (`list[Sample]`): 全部观测样本，按完成顺序排列。
        elapsed_seconds (`float`): 整轮实际耗时（秒）。
    """

    url: str
    concurrency: int
    mode: str
    planned: float | int | None
    samples: list[Sample] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    # --------------------------------------------------------------------------
    # 计数
    # --------------------------------------------------------------------------
    @property
    def total(self) -> int:
        """样本总数（即实际发出的请求数）。

        Returns:
            `int`: 样本数。
        """
        return len(self.samples)

    @property
    def successes(self) -> int:
        """成功样本数（2xx）。

        Returns:
            `int`: 成功数。
        """
        return sum(1 for s in self.samples if s.ok)

    @property
    def failures(self) -> int:
        """失败样本数（非 2xx + 传输层失败）。

        Returns:
            `int`: 失败数。
        """
        return self.total - self.successes

    @property
    def failure_rate(self) -> float:
        """失败率。

        Returns:
            `float`: 失败数 ÷ 总数；**没有样本时返回 0.0**。

        ⚠️ 无样本返回 0.0 是刻意的「数值诚实」：真正的「一个请求都没发出去」
            由 :func:`evaluate_thresholds` 单独判为一个必须失败的阈值，
            而不是在这里伪造一个 100% 或 0% 的比率去误导调用方。
        """
        return self.failures / self.total if self.total else 0.0

    @property
    def throughput(self) -> float:
        """吞吐（req/s）。

        Returns:
            `float`: 总请求数 ÷ 实际耗时；耗时为 0 或没样本时返回 0.0。
        """
        if not self.total or self.elapsed_seconds <= 0:
            return 0.0
        return self.total / self.elapsed_seconds

    # --------------------------------------------------------------------------
    # 分类
    # --------------------------------------------------------------------------
    def status_class_counts(self) -> dict[str, int]:
        """按状态码类别统计样本数。

        Returns:
            `dict[str, int]`: 形如 ``{"2xx": 998, "5xx": 2, "无": 1}``；
            ``"无"`` 计的是传输层失败。只包含出现过的类别，且按固定顺序返回。
        """
        counts: Counter[str] = Counter()
        for sample in self.samples:
            if sample.status is None:
                counts[NO_STATUS_LABEL] += 1
            else:
                counts[f"{sample.status // 100}xx"] += 1
        order = ["2xx", "3xx", "4xx", "5xx", NO_STATUS_LABEL]
        return {key: counts[key] for key in order if counts.get(key)}

    def transport_errors(self) -> list[tuple[str, int]]:
        """列出传输层失败的原因及其出现次数。

        Returns:
            `list[tuple[str, int]]`: ``(原因, 次数)`` 列表，按次数降序；无失败时为空。
        """
        counter = Counter(s.error for s in self.samples if s.status is None and s.error)
        return counter.most_common()

    def percentiles(self, percents: Sequence[int] = PERCENTILES) -> dict[int, float]:
        """计算延迟分位数（毫秒）。

        Args:
            percents (`Sequence[int]`): 要计算的分位点（百分位）。

        Returns:
            `dict[int, float]`: 分位点 → 延迟（毫秒）。没有样本时为空字典。
        """
        return latency_percentiles(self.samples, percents)


# ==============================================================================
# 分位数
# ==============================================================================
def percentile(sorted_values: Sequence[float], percent: float) -> float:
    """从**已排序**的样本里取最近秩（nearest-rank）分位数。

    实现是「秩 = ⌈percent% × N⌉，取第秩个（1 起算）样本」，即：
    在 N 个样本里，恰好有 ``rank`` 个样本小于等于返回值。

    ⚠️ 为什么用最近秩而不是线性插值（这是本函数唯一的、也是最容易写错的地方）：
        · 最近秩只返回**样本里真实存在**的延迟值。压测工具的价值在于如实观测，
          不是把没有测到的值「估」出来 —— 插值会让 1..100 的 p50 变成 50.5，
          报告里于是出现一个从未发生过的延迟。
        · 它天然**没有 off-by-one**：1..100 的 p50 必须是 50（第 ⌈0.5×100⌉=50 个），
          既不是 49 也不是 51。``tests/test_loadtest.py`` 用这条把算法钉死。
        · 浮点陷阱已规避：先算 ``percent * n``（整数乘法，精确），**再**除以 100，
          避免 ``0.9 * 100 == 90.00000000000001`` 这类误差把 p90 顶成第 91 个样本。

    Args:
        sorted_values (`Sequence[float]`): 已按升序排好的样本（**调用方保证有序**）。
        percent (`float`): 分位点，取值区间 ``(0, 100]``。

    Returns:
        `float`: 该分位点上的样本值。

    Raises:
        ValueError: 样本为空，或 ``percent`` 不在 ``(0, 100]`` 内。
    """
    if not sorted_values:
        raise ValueError("空样本无法计算分位数。")
    if not 0 < percent <= 100:
        raise ValueError(f"分位点必须落在 (0, 100] 内，收到 {percent!r}。")

    n = len(sorted_values)
    rank = math.ceil(percent * n / 100)
    # 秩至少为 1（percent>0 且 n>=1 时 ⌈percent*n/100⌉ >= 1），至多为 n，故下标恒合法。
    index = min(rank, n) - 1
    return float(sorted_values[index])


def latency_percentiles(
    samples: Sequence[Sample],
    percents: Sequence[int] = PERCENTILES,
) -> dict[int, float]:
    """对一批样本计算延迟分位数。

    这里**先排序再逐个取秩**（而不是对每个分位点各排一次）：排序是 O(N log N)，
    取秩是 O(1)，因此不论算几个分位点，总代价只与样本数有关。

    Args:
        samples (`Sequence[Sample]`): 观测样本。
        percents (`Sequence[int]`): 分位点（百分位）。

    Returns:
        `dict[int, float]`: 分位点 → 延迟（毫秒）。样本为空时返回空字典。
    """
    if not samples:
        return {}
    ordered = sorted(s.latency_ms for s in samples)
    return {p: percentile(ordered, p) for p in percents}


# ==============================================================================
# 阈值判定
# ==============================================================================
@dataclass(frozen=True)
class ThresholdBreach:
    """一条被突破的阈值。

    Attributes:
        name (`str`): 阈值名（如 ``失败率`` / ``p99 延迟``），打印在报告里。
        observed (`float`): 实测值。
        limit (`float`): 允许的上限。
        message (`str`): 面向人的一句话说明，含实测值与上限。
    """

    name: str
    observed: float
    limit: float
    message: str


def evaluate_thresholds(
    result: LoadResult,
    *,
    max_failure_rate: float = DEFAULT_MAX_FAILURE_RATE,
    max_p99_ms: float = DEFAULT_MAX_P99_MS,
) -> list[ThresholdBreach]:
    """判定整轮结果是否越线。

    纯函数：不打印、不退出，只返回结论 —— 因此可以直接被单测覆盖，
    而不必去捕获标准输出或进程退出码。

    Args:
        result (`LoadResult`): 待判定的结果。
        max_failure_rate (`float`): 失败率上限（``0.01`` 表示 1%）。
        max_p99_ms (`float`): p99 延迟上限（毫秒）。

    Returns:
        `list[ThresholdBreach]`: 被突破的阈值列表；**为空即通过**。

    ⚠️ 判定用严格大于（``>``），故「恰好等于上限」算通过 —— 阈值表达的是
        「不得超过」，把它写成 ``>=`` 会让一个 1.00% 的配置把 1.00% 判成失败。

    ★ 最重要的一条：**一个请求都没发出**必须判失败。
        否则会得到一个最坏形状的假绿灯 —— 脚本什么都没压、退出码 0、
        报告一片祥和，而调用者以为服务通过了负载测试。
    """
    breaches: list[ThresholdBreach] = []

    if result.total == 0:
        return [
            ThresholdBreach(
                name="样本数",
                observed=0.0,
                limit=1.0,
                message="一个请求都没有发出 —— 结果不可用（不是「通过」）。",
            ),
        ]

    if result.failure_rate > max_failure_rate:
        breaches.append(
            ThresholdBreach(
                name="失败率",
                observed=result.failure_rate,
                limit=max_failure_rate,
                message=(
                    f"失败率 {result.failure_rate:.2%} > 上限 {max_failure_rate:.2%}"
                    f"（{result.failures}/{result.total} 个请求失败）"
                ),
            ),
        )

    p99 = result.percentiles((99,)).get(99, 0.0)
    if p99 > max_p99_ms:
        breaches.append(
            ThresholdBreach(
                name="p99 延迟",
                observed=p99,
                limit=max_p99_ms,
                message=f"p99 {p99:.1f}ms > 上限 {max_p99_ms:.1f}ms",
            ),
        )

    return breaches


# ==============================================================================
# 执行
# ==============================================================================
def _elapsed_ms(started: float) -> float:
    """把 ``perf_counter`` 的起点换算成已过去的毫秒数。

    ⚠️ 用 ``perf_counter`` 而非 ``time.time``：后者会被系统对时/NTP 校正**回拨**，
        那样一个请求可能算出**负的**延迟，把分位数整体拉偏。
        ``perf_counter`` 是单调时钟，专为测量间隔而设。

    Args:
        started (`float`): 请求发出前一刻的 ``time.perf_counter()``。

    Returns:
        `float`: 已过去的毫秒数。
    """
    return (time.perf_counter() - started) * 1000.0


async def _one_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None,
) -> Sample:
    """发一次请求并把结果压成一个 :class:`Sample`。

    Args:
        client (`httpx.AsyncClient`): 共享客户端。
        method (`str`): HTTP 方法。
        url (`str`): 完整 URL。
        headers (`dict[str, str] | None`): 附加请求头。

    Returns:
        `Sample`: 观测结果（含传输层失败的情形）。
    """
    started = time.perf_counter()
    try:
        response = await client.request(method, url, headers=headers)
    except Exception as exc:  # noqa: BLE001 —— 传输层异常类型繁多，见下
        # ⚠️ 这里宽捕获是刻意的：连不上这件事的可能异常类型很多
        # （ConnectError / ConnectTimeout / ReadTimeout / RemoteProtocolError…），
        # 而**任何**一种都应当被记成一个失败样本、让压测继续，而不是中断整轮 ——
        # 一个连不上的端点，测出来的正该是「失败率 100%」，不是「脚本崩了」。
        return Sample(
            latency_ms=_elapsed_ms(started),
            status=None,
            error=f"{type(exc).__name__}: {exc}",
        )
    return Sample(latency_ms=_elapsed_ms(started), status=response.status_code)


async def _worker(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None,
    semaphore: asyncio.Semaphore,
    lock: asyncio.Lock,
    state: dict[str, int | None],
    deadline: float | None,
    samples: list[Sample],
) -> None:
    """一个工作协程：循环取活、发请求、记录样本，直到本轮结束。

    Args:
        client (`httpx.AsyncClient`): 共享客户端。
        method (`str`): HTTP 方法。
        url (`str`): 完整 URL。
        headers (`dict[str, str] | None`): 附加请求头。
        semaphore (`asyncio.Semaphore`): 在飞请求数上限。
            ⚠️ 它是**冗余**的：本函数被调起恰好 ``concurrency`` 份
            （见 :func:`run_load` 里的 ``workers``），每个协程同时最多
            只有一个请求在飞，所以真正的并发上界是**协程数**，不是这个
            信号量。留着它是因为「并发有界」这件事在压测里太关键，
            多一道显式闸门比少一道安全；但**别把它当成唯一的界** ——
            真要改并发，改的是 ``run_load`` 里那份 ``range(concurrency)``。
            （把这一行删掉不会改变任何行为，属于等价变异。）
        lock (`asyncio.Lock`): 保护 ``state["remaining"]`` 的递减。
        state (`dict[str, int | None]`): 共享状态；``remaining`` 为 ``None`` 时
            表示计时模式（不数数），否则表示还需发出的请求数。
        deadline (`float | None`): 计时模式的截止时刻（``time.monotonic``）；
            计数模式下为 ``None``。
        samples (`list[Sample]`): 汇总列表（追加是同步操作，无需额外加锁）。
    """
    while True:
        # ⚠️⚠️ 这一行 ``sleep(0)`` 不是凑数的节流，它是**可取消性**的支点。
        #
        # 压测的靶子通常是本机的、极快的服务，而这个循环里每一个 await
        # 都可能在**不让出事件循环**的情况下就返回：
        # ``asyncio.Semaphore.acquire`` 在无人争用时直接返回，
        # 而 ASGI/本地回环请求往往也在同一次循环迭代内跑完。
        # 于是这个 ``while True`` 会**饿死事件循环** —— 实测确认过：
        # 一旦剩余量永不归零（一个很容易写出的 bug），
        # ``asyncio.wait_for(..., timeout=3)`` 也**取消不掉**这个协程，
        # 它一直等到超时进程被外部杀掉；``Ctrl-C`` 同理。
        #
        # 对一个压测工具来说这不是小事：跑一次 ``make loadtest`` 发现打错
        # 了地址，想停下来却按不住，只能 kill -9。
        #
        # ``sleep(0)`` 只让出一次调度权、不真正等待（耗时以微秒计，
        # 相对一个 HTTP 往返可以忽略），代价是每个请求多一次循环调度。
        # 换来的是：取消能送达、``--duration`` 的截止时刻能被准时检查、
        # 以及「协程卡死」这类 bug 会**失败**而不是**挂起**。
        await asyncio.sleep(0)

        if deadline is not None and time.monotonic() >= deadline:
            return

        # 计数模式：在锁内认领「第几个请求」。认领与递减是原子的，
        # 否则两个协程会在**同一个**剩余量上各减一次、多发一个请求，
        # 让「我要发 N 个」变成「大约发 N 个」—— 一个只在并发下出现的静默超额。
        if state["remaining"] is not None:
            async with lock:
                remaining = state["remaining"]
                if remaining is None or remaining <= 0:
                    return
                state["remaining"] = remaining - 1

        # ⚠️ 信号量**必须**包住整个请求（发出 + 读体），而不是只包住「发出」。
        #    只包发出的话，读体阶段不在界内，实际在飞请求数会翻倍。
        async with semaphore:
            sample = await _one_request(client, method, url, headers=headers)
        samples.append(sample)


async def run_load(
    *,
    base_url: str,
    endpoint: str,
    method: str = "GET",
    concurrency: int = DEFAULT_CONCURRENCY,
    duration_seconds: float | None = None,
    total_requests: int | None = None,
    timeout_seconds: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    headers: dict[str, str] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> LoadResult:
    """施加受控并发，跑一轮负载测试。

    ``duration_seconds`` 与 ``total_requests`` **二选一**（调用方保证）：
    前者跑满给定秒数，后者发满给定请求数。

    Args:
        base_url (`str`): 服务根地址（结尾斜杠会被忽略）。
        endpoint (`str`): 端点路径（如 ``/healthz``）。
        method (`str`): HTTP 方法。
        concurrency (`int`): 并发上限（同时最多这么多个请求在飞）。
        duration_seconds (`float | None`): 计时模式的时长（秒）。
        total_requests (`int | None`): 计数模式的请求总数。
        timeout_seconds (`float`): 单请求超时（秒）。
        headers (`dict[str, str] | None`): 附加请求头（如鉴权）。
        transport (`httpx.AsyncBaseTransport | None`): 自定义传输层；
            **仅供** ``--dry-run`` 注入进程内 ASGI 传输，正常压测传 ``None``。

    Returns:
        `LoadResult`: 整轮结果。

    Raises:
        ValueError: 既没给时长也没给请求数，或并发数小于 1。
    """
    if (duration_seconds is None) == (total_requests is None):
        raise ValueError("必须且只能指定 duration_seconds 与 total_requests 之一。")
    if concurrency < 1:
        raise ValueError(f"并发数必须 >= 1，收到 {concurrency!r}。")

    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    samples: list[Sample] = []
    state: dict[str, int | None] = {"remaining": total_requests}
    deadline = None if duration_seconds is None else time.monotonic() + duration_seconds

    # ⚠️ 连接池上限**必须**跟并发对齐：httpx 默认 max_connections=100，
    #    并发设为 200 时多出来的请求会在池子里排队，排队时间被算进延迟 ——
    #    量到的就成了「队列深度」而不是「服务耗时」。对齐后，
    #    并发数即真实的在飞请求数上界。
    limits = httpx.Limits(
        max_connections=concurrency,
        max_keepalive_connections=concurrency,
    )
    client_kwargs: dict[str, Any] = {
        "timeout": timeout_seconds,
        "limits": limits,
        # 不跟随重定向：负载测试要观测服务端**返回了什么**，
        # 而不是替它把 302 走完（跟随会掩盖一个坏掉的路由）。
        "follow_redirects": False,
    }
    if transport is not None:
        client_kwargs["transport"] = transport

    async with httpx.AsyncClient(**client_kwargs) as client:
        started = time.monotonic()
        workers = [
            asyncio.create_task(
                _worker(
                    client,
                    method,
                    url,
                    headers=headers,
                    semaphore=semaphore,
                    lock=lock,
                    state=state,
                    deadline=deadline,
                    samples=samples,
                ),
            )
            for _ in range(concurrency)
        ]
        # 计时模式下，工作协程各自看 deadline 自行退出；计数模式下，剩余量归零后退出。
        await asyncio.gather(*workers)
        elapsed = time.monotonic() - started

    mode = "duration" if duration_seconds is not None else "requests"
    planned: float | int | None = duration_seconds if duration_seconds is not None else total_requests
    return LoadResult(
        url=url,
        concurrency=concurrency,
        mode=mode,
        planned=planned,
        samples=samples,
        elapsed_seconds=elapsed,
    )


# ==============================================================================
# 自检用的进程内应用
# ==============================================================================
def _build_self_test_app() -> Any:
    """构造一个**进程内**的最小 ASGI 应用，供 ``--dry-run`` 当靶子。

    ⚠️ 刻意不复用 ``src.server.app.create_root_app``：那条路要拉起
        存储 / 消息总线 / 模型装配，重且**恰好是我们想绕开的依赖**。
        自检验的是**本脚本**（采样、分位、阈值），靶子越轻越好，
        ``/healthz`` 的契约（200 + ``{"status": "ok"}``）也足够代表它的形状。

    Returns:
        `Any`: 一个只实现 ``/healthz`` 与 ``/api/v1/health`` 的 FastAPI 应用。
    """
    # 延迟导入：只有 ``--dry-run`` 才需要 FastAPI，正常压测路径不该为它付导入成本。
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    app = FastAPI()

    @app.get(HEALTH_ENDPOINT)
    async def _healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.get("/api/v1/health")
    async def _api_v1_health() -> JSONResponse:
        return JSONResponse({"status": "ok", "scope": "api/v1"})

    return app


# ==============================================================================
# 报告
# ==============================================================================
def render_report(
    result: LoadResult,
    *,
    health_only: bool,
    max_failure_rate: float,
    max_p99_ms: float,
) -> list[ThresholdBreach]:
    """把结果打印成人读的报告，并返回阈值判定结论。

    Args:
        result (`LoadResult`): 整轮结果。
        health_only (`bool`): 是否 `--health-only` 模式（影响标题措辞）。
        max_failure_rate (`float`): 失败率上限。
        max_p99_ms (`float`): p99 上限（毫秒）。

    Returns:
        `list[ThresholdBreach]`: 被突破的阈值（为空即通过）。
    """
    if health_only:
        print("▶ 模式：纯基础设施基线（--health-only，只打 /healthz，不碰模型与依赖）")

    print(f"▶ 负载目标：{result.url}")
    mode_text = (
        f"持续 {float(result.planned):.1f}s"
        if result.mode == "duration"
        else f"共 {int(result.planned or 0)} 个请求"
    )
    print(f"▶ 并发：{result.concurrency}；施加方式：{mode_text}")
    print(f"▶ 阈值：失败率 ≤ {max_failure_rate:.2%}；p99 ≤ {max_p99_ms:.0f}ms")
    print()

    if result.total == 0:
        print("❌ 一个请求都没有发出 —— 无结果可报告。")
        return evaluate_thresholds(
            result,
            max_failure_rate=max_failure_rate,
            max_p99_ms=max_p99_ms,
        )

    print("结果：")
    print(f"  总请求      {result.total}")
    print(f"  成功        {result.successes}  ({result.successes / result.total:.2%})")
    print(f"  失败        {result.failures}  ({result.failure_rate:.2%})")
    print(f"  吞吐        {result.throughput:.1f} req/s")
    print(f"  耗时        {result.elapsed_seconds:.2f} s")

    classes = result.status_class_counts()
    if classes:
        print()
        print("  状态码分布：")
        for key, count in classes.items():
            print(f"    {key:<10} {count}")

    errors = result.transport_errors()
    if errors:
        print()
        print("  传输层失败（根本原因，降序）：")
        for reason, count in errors[:5]:
            print(f"    {count:>6} × {reason}")

    percentiles = result.percentiles()
    if percentiles:
        print()
        print("  延迟（毫秒）：")
        for point, value in percentiles.items():
            print(f"    p{point:<4}      {value:.2f}")

    breaches = evaluate_thresholds(
        result,
        max_failure_rate=max_failure_rate,
        max_p99_ms=max_p99_ms,
    )

    print()
    print("阈值判定：")
    breached_names = {breach.name for breach in breaches}
    # ⚠️ 这里用「实测值 / 上限」而不是「实测值 ≤ 上限」：后者在**越线**时会打印出
    #    像 `1.5ms ≤ 0ms` 这样一句自相矛盾的话 —— 标记是 ❌，文字却说「小于等于」，
    #    看起来像判定逻辑坏了。措辞要同时适配通过与否两种情况。
    checks = [
        ("失败率", f"{result.failure_rate:.2%} / 上限 {max_failure_rate:.2%}"),
        ("p99 延迟", f"{percentiles.get(99, 0.0):.1f}ms / 上限 {max_p99_ms:.0f}ms"),
    ]
    for name, text in checks:
        mark = "❌" if name in breached_names else "✅"
        print(f"  {mark} {name}  {text}")

    return breaches


# ==============================================================================
# 命令行
# ==============================================================================
@dataclass(frozen=True)
class LoadPlan:
    """由命令行参数解析出的、可执行的压测计划。

    Attributes:
        endpoint (`str`): 目标端点路径。
        duration_seconds (`float | None`): 计时模式时长；计数模式下为 ``None``。
        total_requests (`int | None`): 计数模式请求数；计时模式下为 ``None``。
    """

    endpoint: str
    duration_seconds: float | None
    total_requests: int | None


def resolve_plan(
    *,
    health_only: bool,
    endpoint: str | None,
    duration: float | None,
    requests: int | None,
) -> LoadPlan:
    """把原始命令行参数解释成一个无歧义的 :class:`LoadPlan`。

    与 argparse 解耦，是为了让「参数该怎么解释」这件事能在单测里直接验证，
    而不必拉起真服务或捕获 ``SystemExit``。

    Args:
        health_only (`bool`): 是否 `--health-only`。
        endpoint (`str | None`): ``--endpoint`` 的原值；未指定为 ``None``。
        duration (`float | None`): ``--duration`` 的原值。
        requests (`int | None`): ``--requests`` 的原值。

    Returns:
        `LoadPlan`: 解析结果。

    Raises:
        ValueError: ``--health-only`` 与 ``--endpoint`` 同时出现（含义冲突），
            或 ``--duration`` 与 ``--requests`` 同时出现。
    """
    if health_only and endpoint is not None:
        raise ValueError(
            "--health-only 与 --endpoint 不能同时指定：前者已把端点钉死为 "
            f"{HEALTH_ENDPOINT}，再给 --endpoint 只会让人误以为打的是别的地址。",
        )
    if duration is not None and requests is not None:
        raise ValueError("--duration 与 --requests 只能二选一。")

    resolved_endpoint = HEALTH_ENDPOINT if (health_only or endpoint is None) else endpoint

    if duration is None and requests is None:
        duration = DEFAULT_DURATION_SECONDS

    return LoadPlan(
        endpoint=resolved_endpoint,
        duration_seconds=duration,
        total_requests=requests,
    )


def _positive_int(raw: str) -> int:
    """argparse 的 ``type``：把参数解析成正整数，否则报错。

    Args:
        raw (`str`): 原始参数文本。

    Returns:
        `int`: 解析出的正整数。

    Raises:
        argparse.ArgumentTypeError: 不是正整数。
    """
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"需要一个正整数，收到 {raw!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"必须 >= 1，收到 {value}")
    return value


def _positive_float(raw: str) -> float:
    """argparse 的 ``type``：把参数解析成正浮点数，否则报错。

    Args:
        raw (`str`): 原始参数文本。

    Returns:
        `float`: 解析出的正浮点数。

    Raises:
        argparse.ArgumentTypeError: 不是正数。
    """
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"需要一个正数，收到 {raw!r}") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError(f"必须 > 0，收到 {value}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    Returns:
        `argparse.ArgumentParser`: 解析器。
    """
    parser = argparse.ArgumentParser(
        description="AliGo 差旅助手 —— 对已启动的服务施加受控并发负载。",
        epilog=(
            "示例：\n"
            "  python scripts/loadtest.py --health-only --duration 20\n"
            "  python scripts/loadtest.py --endpoint /api/v1/health --requests 5000 --concurrency 50\n"
            "  python scripts/loadtest.py --dry-run\n"
            "  make loadtest LOADTEST_URL=http://localhost:8010\n"
            "\n"
            "说明：--duration 与 --requests 二选一；都不给时默认 --duration 10。\n"
            "      --health-only 等价于把 --endpoint 钉死为 /healthz（两者不可同时给）。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"服务根地址（默认 {DEFAULT_BASE_URL}，与 Makefile 的 LOADTEST_URL 一致）",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help=f"目标端点路径（默认 {HEALTH_ENDPOINT}）。与 --health-only 互斥。",
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help=f"纯基础设施基线：只打 {HEALTH_ENDPOINT}（免鉴权、零 I/O、不调模型）。",
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=DEFAULT_CONCURRENCY,
        help=f"并发上限（同时最多多少个请求在飞，默认 {DEFAULT_CONCURRENCY}）",
    )
    parser.add_argument(
        "--duration",
        type=_positive_float,
        default=None,
        help=f"按时间压测的秒数（与 --requests 互斥；默认 {DEFAULT_DURATION_SECONDS:.0f}）",
    )
    parser.add_argument(
        "--requests",
        type=_positive_int,
        default=None,
        help="按数量压测的请求总数（与 --duration 互斥）",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=DEFAULT_REQUEST_TIMEOUT_SECONDS,
        help=f"单请求超时秒数（默认 {DEFAULT_REQUEST_TIMEOUT_SECONDS:.0f}）",
    )
    parser.add_argument(
        "--method",
        default="GET",
        help="HTTP 方法（默认 GET；本脚本不发送请求体）",
    )
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=DEFAULT_MAX_FAILURE_RATE,
        help=f"失败率上限，超出即失败退出（默认 {DEFAULT_MAX_FAILURE_RATE:.2%}）",
    )
    parser.add_argument(
        "--max-p99-ms",
        type=float,
        default=DEFAULT_MAX_P99_MS,
        help=f"p99 延迟上限（毫秒），超出即失败退出（默认 {DEFAULT_MAX_P99_MS:.0f}）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="自检模式：对一个**进程内**的 ASGI 应用压测，不需要 Docker 或任何服务。",
    )
    return parser


def _run_once(args: argparse.Namespace, plan: LoadPlan) -> int:
    """执行一轮压测并返回退出码（``main`` 的可测内核）。

    Args:
        args (`argparse.Namespace`): 解析后的命令行参数。
        plan (`LoadPlan`): 已解释的执行计划。

    Returns:
        `int`: 0 = 通过；1 = 有阈值被突破。
    """
    base_url = args.base_url
    transport: httpx.AsyncBaseTransport | None = None

    if args.dry_run:
        # ★ 自检：靶子在进程内，走 ASGITransport 直接进应用，不经过任何 socket。
        base_url = "http://loadtest.local"
        transport = httpx.ASGITransport(app=_build_self_test_app())
        print("▶ 自检模式：靶子是**进程内**的最小 ASGI 应用（不连 Docker，不连网络）。")

    print(f"▶ 压测起点：{base_url.rstrip('/')}/{plan.endpoint.lstrip('/')}")
    print()

    result = asyncio.run(
        run_load(
            base_url=base_url,
            endpoint=plan.endpoint,
            method=args.method,
            concurrency=args.concurrency,
            duration_seconds=plan.duration_seconds,
            total_requests=plan.total_requests,
            timeout_seconds=args.timeout,
            transport=transport,
        ),
    )

    breaches = render_report(
        result,
        health_only=args.health_only,
        max_failure_rate=args.max_failure_rate,
        max_p99_ms=args.max_p99_ms,
    )

    print()
    if breaches:
        print(f"❌ 负载测试未通过：{len(breaches)} 项阈值被突破")
        print("   被突破的阈值：")
        for breach in breaches:
            print(f"     · {breach.message}")
        return 1

    print("✅ 负载测试通过：失败率与 p99 均在阈值内")
    return 0


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv (`list[str] | None`): 命令行参数；``None`` 时取 ``sys.argv``。

    Returns:
        `int`: 0 表示未突破任何阈值，1 表示有阈值被突破。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # 自检模式在参数都没给时改用**计数模式**，让自检几秒内跑完（而不是压满默认的 10 秒）。
    duration, requests = args.duration, args.requests
    if args.dry_run and duration is None and requests is None:
        requests = DRY_RUN_REQUESTS

    try:
        plan = resolve_plan(
            health_only=args.health_only,
            endpoint=args.endpoint,
            duration=duration,
            requests=requests,
        )
    except ValueError as exc:
        # 参数含义冲突属于**用法错误**（退出码 2），与「压测未通过」（1）区分开。
        parser.error(str(exc))
        return 2  # pragma: no cover —— parser.error 已经以 SystemExit(2) 退出

    try:
        return _run_once(args, plan)
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 —— 顶层入口，见下
        # ⚠️ 顶层宽捕获是刻意的（与 scripts/milvus_init.py 一致）：
        # 「连不上」「参数非法」「环境不对」抛出的异常类型各不相同，
        # 让真正的原因（拒绝连接 / 名字不合法）留在消息里，比包一层
        # 只剩「压测失败」的自定义异常有用得多。
        print(f"\n❌ 压测失败：{safe_error(exc)}", file=sys.stderr)
        print(
            f"   提示：确认服务已启动且 {args.base_url} 可达（core 档含它：make up）；\n"
            f"        或先用 `python scripts/loadtest.py --dry-run` 验证脚本自身。",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_CONCURRENCY",
    "DEFAULT_DURATION_SECONDS",
    "DEFAULT_MAX_FAILURE_RATE",
    "DEFAULT_MAX_P99_MS",
    "HEALTH_ENDPOINT",
    "LoadPlan",
    "LoadResult",
    "PERCENTILES",
    "Sample",
    "ThresholdBreach",
    "evaluate_thresholds",
    "latency_percentiles",
    "main",
    "percentile",
    "render_report",
    "resolve_plan",
    "run_load",
]
