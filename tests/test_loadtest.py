# -*- coding: utf-8 -*-
"""``scripts/loadtest.py`` 的**判定表**测试：分位数、阈值、参数解释。

==============================================================================
这些用例在防什么
==============================================================================
    负载测试的价值全在**结论准不准**上：它输出的不是一个是/否，
    而是一组数字（失败率、p99…）外加一条「越没越线」的判定。
    这两处任一写错，都会得到一种**看起来很专业、实际是错的**报告：

      1. **分位数取错**（off-by-one / 插值）——
         「1..100 的 p50」若算成 49 或 51，或插值成 50.5，
         报告里的每个延迟数字都会系统性偏移一点。它不会报错，只会一直错。
         本项目把它钉死为最近秩：p50 of 1..100 **必须**等于 50。

      2. **阈值判反或漏判** —— 比如把 ``>`` 写成 ``>=``（边界误杀），
         或**根本不对「一个请求都没发出」设防**。后一种是本项目最忌讳的
         「假绿灯」：脚本什么都没压、退出码 0、报告一片祥和，
         而调用者以为服务通过了负载测试。

    用例围绕这两条来写，每条判据都配**反例** —— 只测「正常输入通过」的话，
    一个「无脑返回 0」的实现也能全绿。

==============================================================================
为什么这些用例完全离线
==============================================================================
    判定表（:func:`percentile` / :func:`evaluate_thresholds` / :func:`resolve_plan`）
    都是纯函数，直接调、毫秒级。而在飞路径（:func:`run_load`）用
    ``httpx.ASGITransport`` 把靶子放进**本进程**，因此连真服务都不需要 ——
    与 ``scripts/loadtest.py --dry-run`` 走的是同一条路。
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from scripts import loadtest
from scripts.loadtest import (
    PERCENTILES,
    LoadResult,
    Sample,
    evaluate_thresholds,
    latency_percentiles,
    percentile,
    render_report,
    resolve_plan,
    run_load,
)


# ==============================================================================
# 构造工具
# ==============================================================================
def _result(samples: list[Sample], *, elapsed_seconds: float = 1.0) -> LoadResult:
    """把一批样本包成一个 :class:`LoadResult`。

    Args:
        samples (`list[Sample]`): 观测样本。
        elapsed_seconds (`float`): 整轮耗时。

    Returns:
        `LoadResult`: 可直接喂给判定函数的对象。
    """
    return LoadResult(
        url="http://loadtest.test/healthz",
        concurrency=10,
        mode="requests",
        planned=len(samples),
        samples=samples,
        elapsed_seconds=elapsed_seconds,
    )


class _RunawayLoad(BaseException):
    """靶子收到的请求数远超 ``total_requests`` —— 压测协程没有退出。

    ⚠️ 刻意继承 ``BaseException``，而不是 ``Exception``。

    :func:`scripts.loadtest._one_request` 用一个 ``except Exception``
    把**任何**传输层异常都记成一个失败样本、让整轮压测继续跑下去
    （那里的注释解释了为什么这是对的：一个连不上的端点，结论本就该是
    「失败率 100%」而不是「脚本崩了」）。

    但「协程根本不退出」不是传输层故障，它必须**穿出去**、让用例变红。
    那个 ``except Exception`` 会把 ``AssertionError`` 一起吞掉，
    所以这里用 ``BaseException`` 这条唯一的通道 —— 它同时也是
    ``KeyboardInterrupt`` / ``SystemExit`` 所在的层，语义上正好：
    「这不是一次观测失败，这是一次必须中断运行的事故」。
    """


def _capped_transport(
    handler: Any,
    cap: int,
) -> tuple[httpx.MockTransport, dict[str, int]]:
    """把一个 :class:`httpx.MockTransport` 的 handler 包成「最多接受 cap 次请求」。

    Args:
        handler (`Any`): 原 handler，签名 ``(httpx.Request) -> httpx.Response``。
        cap (`int`): 允许的最大请求数。

    Returns:
        `tuple[httpx.MockTransport, dict[str, int]]`:
            包好的传输层，以及一个记录实际请求数的可变字典（键 ``"n"``）。

    Raises:
        _RunawayLoad: 请求数超过 ``cap``。
    """
    seen = {"n": 0}

    def _wrapped(request: httpx.Request) -> httpx.Response:
        seen["n"] += 1
        if seen["n"] > cap:
            raise _RunawayLoad(
                f"靶子已被请求 {seen['n']} 次，上限 {cap}。"
                f"请求数超出 total_requests，已知两种成因："
                f"其一，工作协程没有退出（剩余量没递减）；"
                f"其二，客户端跟随了重定向，一个请求打到了靶子上两次。",
            )
        return handler(request)

    return httpx.MockTransport(_wrapped), seen


class _CappedAsgiApp:
    """把一个 ASGI app 包成「最多接受 cap 次 HTTP 请求」。

    与 :func:`_capped_transport` 同一用意，供 ``httpx.ASGITransport``
    使用 —— 它接的是 ASGI app，不是 handler，包不了同一个壳。

    Args:
        app (`Any`): 被包的 ASGI 应用。
        cap (`int`): 允许的最大请求数。
    """

    def __init__(self, app: Any, cap: int) -> None:
        self._app = app
        self._cap = cap
        self.seen: dict[str, int] = {"n": 0}

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """ASGI 入口。"""
        if scope["type"] == "http":
            self.seen["n"] += 1
            if self.seen["n"] > self._cap:
                raise _RunawayLoad(
                    f"靶子已被请求 {self.seen['n']} 次，上限 {self._cap}。"
                    f"请求数超出 total_requests，已知两种成因："
                    f"其一，工作协程没有退出（剩余量没递减）；"
                    f"其二，客户端跟随了重定向，一个请求打到了靶子上两次。",
                )
        await self._app(scope, receive, send)


#: 计数模式用例的兜底超时（秒）。
#:
#: ⚠️ 给得很宽松 —— 正常路径是毫秒级的。它要抓的是「工作协程**永不退出**」，
#: 不是「跑得慢」。慢在共享 CI 上是常态，把慢当故障会让用例整个不可信。
_LOOP_GUARD_SECONDS = 10.0


async def _run_load_bounded(**kwargs: Any) -> LoadResult:
    """带超时地跑一轮 :func:`run_load`（计数模式用例一律走这里）。

    ⚠️ 这是**兜底**，不是主防线；主防线是 :func:`_capped_transport` /
    :class:`_CappedAsgiApp` 那个请求数上限。两者的分工是实测出来的：

    计数模式的工作协程靠「认领到的剩余量归零」退出（见
    :func:`scripts.loadtest._worker`）。把那一行递减删掉，协程永不退出，
    用例不是**变红**而是**卡死** —— 实测 ``pytest tests/test_loadtest.py``
    连 60 秒都打不出一个进度点。

    于是先加了这里的 ``wait_for``。它**单独跑一条用例时有效**
    （3 次实测：10.3s / 12.6s 干净地报 ``TimeoutError``），但**跑整份文件时
    失效**：前面那条用例已经空转过 10 秒、往 ``samples`` 里堆了上百万个样本，
    内存压力把后面那条的 10 秒定时器也拖住了，实测 180 秒仍未结束。

    结论：**不能靠事件循环自己给自己兜底** —— 定时器本身就是被饿死的那个东西。
    请求数上限则完全不依赖时间：它在下一次请求到达靶子的那一刻就炸，
    毫秒级、确定性地失败，且不会给后续用例留下任何垃圾。
    这里的超时留着，是为了接住「将来出现别的、不走靶子的挂起」。

    Args:
        **kwargs (`Any`): 原样传给 :func:`scripts.loadtest.run_load`。

    Returns:
        `LoadResult`: 压测结果。

    Raises:
        asyncio.TimeoutError: 超过 :data:`_LOOP_GUARD_SECONDS` 仍未结束。
    """
    return await asyncio.wait_for(run_load(**kwargs), timeout=_LOOP_GUARD_SECONDS)


def _ok(status: int = 200, latency_ms: float = 5.0) -> Sample:
    """构造一个成功样本。

    Args:
        status (`int`): 状态码。
        latency_ms (`float`): 延迟（毫秒）。

    Returns:
        `Sample`: 样本。
    """
    return Sample(latency_ms=latency_ms, status=status)


# ==============================================================================
# 一、分位数 —— 最近秩，无 off-by-one
# ==============================================================================
def test_p50_of_1_to_100_is_exactly_50() -> None:
    """1..100 的 p50 ⇒ **恰好 50**（既不是 49，也不是 51）。

    ★ 这条是本文件的锚点。它同时否掉两种错误实现：

      · **向下取整的 off-by-one** ⇒ 49；
      · **向上取整的 off-by-one** ⇒ 51。

    最近秩的定义是「秩 = ⌈p% × N⌉，取第秩个样本」，故 N=100、p=50 时秩为 50，
    取第 50 个样本 = 50。断言里把 49/51 显式列出来，是为了让失败时报出的
    是「算成了 49」这种一眼能懂的话，而不是一个孤零零的 ``50.0 != 49.0``。
    """
    values = [float(v) for v in range(1, 101)]

    got = percentile(values, 50)

    assert got == 50.0, f"p50 应为 50，实际 {got}（51 或 49 都说明取了错的那一秩）"
    assert got != 49.0
    assert got != 51.0


def test_percentile_does_not_interpolate() -> None:
    """1..100 的 p50 ⇒ **不是 50.5**（最近秩而非线性插值）。

    反例用例。若实现用了 ``(N-1)*p/100`` 式的线性插值（numpy 的默认口径），
    p50 会落在第 50 与第 51 个样本之间，得 50.5 —— 一个**样本里从没出现过**
    的延迟值。压测工具报出一个没测到过的数，比报得粗糙更糟。
    """
    values = [float(v) for v in range(1, 101)]

    assert percentile(values, 50) != 50.5


@pytest.mark.parametrize(
    ("percent", "expected"),
    [
        (1, 1.0),
        (50, 50.0),
        (90, 90.0),
        (95, 95.0),
        (99, 99.0),
        (100, 100.0),
    ],
)
def test_percentile_ranks_on_1_to_100(percent: int, expected: float) -> None:
    """1..100 上各分位点 ⇒ 秩恰好等于分位点本身。

    ⚠️ 这条专治**浮点陷阱**：``0.9 * 100 == 90.00000000000001``，若实现先算
        ``percent / 100`` 再乘 N，p90 的秩会被 ``ceil`` 顶成 91。
        实现里先做整数乘法 ``percent * n`` 再除 100，正是为了避开它。
    """
    values = [float(v) for v in range(1, 101)]

    assert percentile(values, percent) == expected


@pytest.mark.parametrize(
    ("count", "percent", "expected"),
    [
        (7, 90, 7.0),  # ⌈6.3⌉ = 7
        (7, 50, 4.0),  # ⌈3.5⌉ = 4
        (7, 1, 1.0),  # ⌈0.07⌉ = 1
        (7, 99, 7.0),  # ⌈6.93⌉ = 7
        (7, 30, 3.0),  # ⌈2.1⌉ = 3
        (3, 50, 2.0),  # ⌈1.5⌉ = 2
        (6, 25, 2.0),  # ⌈1.5⌉ = 2
    ],
)
def test_percentile_rounds_a_fractional_rank_up(
    count: int,
    percent: int,
    expected: float,
) -> None:
    """★★★ 秩**不是整数**时，必须向上取整（``ceil``，不是 ``floor``）。

    ⚠️⚠️ 这条是本文件里唯一能抓住「ceil 写成 floor」的用例，而它此前**一直
    缺着** —— 上面每一条分位用例用的样本数都让 ``percent * n`` 恰好被 100
    整除（N=100 配整数分位、N=4 配 25/50/75/100），在那些点上
    ``ceil`` 与 ``floor`` **恒等**。于是一个 ``math.floor`` 的实现能通过
    全部用例，而它的错误只在小样本上现形：N=3 的 p50 会从第 2 个样本
    变成第 1 个。

    小样本恰恰是压测最常遇到的：``--requests 10 --concurrency 3`` 这种
    冒烟压测，N 就是十几。所以这个「只在小样本上错」的实现，错的正是
    最常被跑的那条路径。

    ⚠️ 每条都同时断言「等于期望值」与「不等于向下取整会得到的值」——
    只断言前者的话，失败信息是一个孤零零的数字，读的人得自己去算
    floor 是多少；把两者都写出来，失败时直接告诉你「它算成了 floor」。
    """
    values = [float(v) for v in range(1, count + 1)]
    # ⚠️ 这里**逐字复刻**实现取下标的那一步，包括它对越界的处理
    # （``min(rank, n) - 1``）：p1 在 N=7 上向下取整得到秩 0，
    # 下标就成了 -1，即**最后一个**样本。自己另写一套「floor 大概会给
    # 第几个」的估算，会在这种边界上算出与真实 floor 实现不同的值，
    # 于是这条反例断言就悄悄失效了 —— 而这正是它要防的东西。
    floor_index = min((percent * count) // 100, count) - 1
    floor_value = float(values[floor_index])

    got = percentile(values, percent)

    assert got == expected, f"N={count} 的 p{percent} 应为 {expected}，实际 {got}"
    assert got != floor_value, (
        f"N={count} 的 p{percent} 算成了 {floor_value}，那是向下取整会得到的结果 —— "
        f"实现里是不是把 math.ceil 写成了 math.floor？"
    )


def test_percentile_p90_of_seven_samples_is_not_the_sixth() -> None:
    """单独钉一条最有代表性的：7 个样本的 p90 ⇒ 第 **7** 个（不是第 6 个）。

    ⚠️ 与上面那条参数化用例重复是**刻意**的：参数化的失败信息是
    「下标 1 的用例挂了」，而这一条在报告里直接写着「p90 of seven」。
    这个具体数字是评审最常拿来复核的一个（``⌈0.9×7⌉ = 7``），
    值得有一条能一眼看懂的红灯。
    """
    values = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]

    assert percentile(values, 90) == 70.0


def test_percentile_on_a_small_hand_checked_sample() -> None:
    """4 个样本 [10,20,30,40] 上逐点核对 ⇒ 秩 = ⌈p%×4⌉。

    小样本上人手可算，用来交叉验证大样本那几条不是碰巧。
    """
    values = [10.0, 20.0, 30.0, 40.0]

    assert percentile(values, 25) == 10.0  # ⌈1⌉ = 1 → 第 1 个
    assert percentile(values, 50) == 20.0  # ⌈2⌉ = 2 → 第 2 个
    assert percentile(values, 75) == 30.0  # ⌈3⌉ = 3 → 第 3 个
    assert percentile(values, 100) == 40.0  # ⌈4⌉ = 4 → 第 4 个


def test_percentile_single_sample() -> None:
    """只有一个样本 ⇒ 任何分位点都返回它自己。"""
    assert percentile([7.0], 50) == 7.0
    assert percentile([7.0], 99) == 7.0


def test_percentile_rejects_empty() -> None:
    """空样本 ⇒ 抛 ``ValueError``（而不是返回 0，那会伪装成一个正常结果）。"""
    with pytest.raises(ValueError):
        percentile([], 50)


@pytest.mark.parametrize("percent", [0, -1, 101, 1000])
def test_percentile_rejects_out_of_range(percent: float) -> None:
    """分位点越界 ⇒ 抛 ``ValueError``。

    ⚠️ 反例用例。少了它，一个把 ``percent`` 直接当下标的实现会在
        ``percent=1000`` 时越界崩溃，而 ``percent=0`` 时静默取到第 0 个样本 ——
        后者是一个**不报错的错**。
    """
    with pytest.raises(ValueError):
        percentile([1.0, 2.0, 3.0], percent)


def test_latency_percentiles_sorts_internally() -> None:
    """``latency_percentiles`` 接受**乱序**样本 ⇒ 结果与排序后一致。

    ⚠️ ``percentile`` 的契约要求输入有序（它自己不做排序，免得每算一个
        分位点都排一遍）。这条用例钉住「排序这一步由 ``latency_percentiles``
        负责」，否则调用方传乱序样本会得到一个静默错误的分位数。
    """
    shuffled = [
        Sample(latency_ms=v, status=200)
        for v in (30.0, 10.0, 40.0, 20.0)
    ]

    got = latency_percentiles(shuffled, (50, 100))

    assert got == {50: 20.0, 100: 40.0}


def test_latency_percentiles_default_keys() -> None:
    """默认分位点 ⇒ 键是 p50/p90/p95/p99。"""
    samples = [_ok(latency_ms=float(v)) for v in range(1, 101)]

    got = latency_percentiles(samples)

    assert tuple(got) == PERCENTILES
    assert got[50] == 50.0
    assert got[99] == 99.0


def test_latency_percentiles_empty() -> None:
    """没有样本 ⇒ 空字典（``percentile`` 会抛，所以这里必须先挡住）。"""
    assert latency_percentiles([]) == {}


# ==============================================================================
# 二、计数与派生指标
# ==============================================================================
def test_success_is_2xx_only() -> None:
    """★ 只有 2xx 算成功；3xx/4xx/5xx 与传输层失败都算失败。

    反例用例。若成功判据写成「状态码 < 400」，一个把所有请求 302 到登录页的
    服务会被判成 100% 成功 —— 那正是必须被拦住的部署事故。
    """
    samples = [
        _ok(200),
        _ok(204),
        _ok(302),
        _ok(404),
        _ok(503),
        Sample(latency_ms=1.0, status=None, error="ConnectError: refused"),
    ]
    result = _result(samples)

    assert result.total == 6
    assert result.successes == 2
    assert result.failures == 4
    assert result.failure_rate == pytest.approx(4 / 6)


def test_status_class_counts_groups_and_labels_transport_errors() -> None:
    """状态码按类别聚合 ⇒ 传输层失败单列为 ``"无"``。

    ⚠️ 把「没连上」混进 5xx 会让报告失去方向：5xx 是服务端处理不了，
        连不上是根本没到服务端。两者的排查路径完全不同。
    """
    samples = [
        _ok(200),
        _ok(200),
        _ok(404),
        _ok(503),
        Sample(latency_ms=1.0, status=None, error="ConnectError"),
    ]
    result = _result(samples)

    assert result.status_class_counts() == {"2xx": 2, "4xx": 1, "5xx": 1, "无": 1}


def test_failure_rate_of_empty_result_is_zero() -> None:
    """没有样本 ⇒ ``failure_rate`` 返回 0.0（**不是** 100%）。

    ⚠️ 这与「空结果应当失败」并不矛盾：失败由 :func:`evaluate_thresholds`
        的「样本数」那一条负责。这里保持 0.0 是为了让 ``failure_rate``
        这个**数值**本身不撒谎 —— 没有请求，就谈不上失败率。
    """
    assert _result([]).failure_rate == 0.0


def test_throughput_uses_elapsed_seconds() -> None:
    """吞吐 = 总数 ÷ 耗时；耗时为 0 ⇒ 返回 0 而不是除零。"""
    result = _result([_ok() for _ in range(100)], elapsed_seconds=2.0)

    assert result.throughput == pytest.approx(50.0)
    assert _result([_ok()], elapsed_seconds=0.0).throughput == 0.0
    assert _result([]).throughput == 0.0


def test_transport_errors_are_counted_and_sorted() -> None:
    """★★★ 传输层失败原因 ⇒ 按**次数降序**聚合。

    ⚠️ 样本的**插入顺序刻意与降序相反**：出现次数少的 ``ConnectTimeout``
    放在前面。原先的用例是反过来的（``ConnectError: refused`` 出现两次、
    又是第一个插入），于是一个**根本不排序**的实现（``list(counter.items())``
    或 ``dict(counter)``，保持首次出现顺序）会给出完全相同的输出，
    用例照样绿。压测报告里「失败原因」那一段的价值就在于**第一条是主因**，
    不排序等于把最有用的信息埋进噪声里。
    """
    samples = [
        Sample(latency_ms=1.0, status=None, error="ConnectTimeout"),  # 1 次，最先出现
        Sample(latency_ms=1.0, status=None, error="ConnectError: refused"),
        Sample(latency_ms=1.0, status=None, error="ConnectError: refused"),  # 2 次
    ]

    assert _result(samples).transport_errors() == [
        ("ConnectError: refused", 2),
        ("ConnectTimeout", 1),
    ]


def test_status_class_counts_are_in_a_fixed_order_not_first_appearance() -> None:
    """★★★ 状态码类别 ⇒ 按**固定顺序**（2xx→3xx→4xx→5xx→无）列出。

    ⚠️ 断言的是 ``list(keys)`` 而不是整个字典：**字典相等在 Python 里
    忽略顺序**，所以 ``{"2xx":2,"4xx":1} == {"4xx":1,"2xx":2}`` 为真。
    原先的用例正是拿整个字典做比较，于是一个直接 ``dict(counts)``
    （顺序 = 首次出现顺序）的实现能通过 —— 而它的报告会随
    请求到达的先后随机排列那几行，两次压测的结果没法并排看。

    样本的首次出现顺序刻意与固定顺序不同（5xx 在前）。
    """
    samples = [
        _ok(503),
        _ok(404),
        _ok(200),
        _ok(200),
    ]

    counts = _result(samples).status_class_counts()

    assert list(counts.keys()) == ["2xx", "4xx", "5xx"], (
        "类别没有按固定顺序输出 —— 报告那几行的次序会变得不可复现"
    )
    assert counts == {"2xx": 2, "4xx": 1, "5xx": 1}


# ==============================================================================
# 三、阈值判定
# ==============================================================================
def test_all_green_passes_with_no_breach() -> None:
    """失败率为 0、p99 远低于上限 ⇒ 无任何突破。"""
    samples = [_ok(latency_ms=5.0) for _ in range(100)]

    breaches = evaluate_thresholds(
        _result(samples),
        max_failure_rate=0.01,
        max_p99_ms=1000.0,
    )

    assert breaches == []


def test_failure_rate_above_limit_is_reported_by_name() -> None:
    """失败率越线 ⇒ 恰好一条，且名字是「失败率」。

    ⚠️ 名字必须可读：任务要求「打印是哪一条阈值被突破」，
        所以判定函数返回的是带名字的对象，而不是一个布尔。
    """
    samples = [_ok() for _ in range(95)] + [_ok(500) for _ in range(5)]

    breaches = evaluate_thresholds(
        _result(samples),
        max_failure_rate=0.01,
        max_p99_ms=1000.0,
    )

    assert [b.name for b in breaches] == ["失败率"]
    assert breaches[0].observed == pytest.approx(0.05)
    assert "失败率" in breaches[0].message


def test_p99_above_limit_is_reported_by_name() -> None:
    """★ p99 越线 ⇒ 一条「p99 延迟」。

    ⚠️ 这条同时钉住最近秩在阈值处的语义：100 个样本里**恰好 5 个**慢样本
        （5% > 1%）才会让第 99 个样本落到慢值上。**恰好 1 个**慢样本时
        p99 仍是快值（见下一条）—— 这是刻意的，不是 bug：
        「最差的 1%」不包含「最差的那一个」。
    """
    slow = 5000.0
    samples = [_ok(latency_ms=5.0) for _ in range(95)] + [_ok(latency_ms=slow) for _ in range(5)]

    breaches = evaluate_thresholds(
        _result(samples),
        max_failure_rate=0.01,
        max_p99_ms=1000.0,
    )

    assert [b.name for b in breaches] == ["p99 延迟"]
    assert breaches[0].observed == pytest.approx(slow)


def test_one_slow_sample_in_a_hundred_does_not_breach_p99() -> None:
    """100 个样本里**只有 1 个**慢 ⇒ p99 **不**越线。

    ⚠️ 反例用例（与上一条配对）。少了它，一个「只要有一个慢样本就报 p99 越线」
        的实现也能通过上一条 —— 而那会让任何一次偶发抖动都变成红灯，
        把噪声当成故障，最终没人再信这份报告。
    """
    samples = [_ok(latency_ms=5.0) for _ in range(99)] + [_ok(latency_ms=5000.0)]

    breaches = evaluate_thresholds(
        _result(samples),
        max_failure_rate=0.01,
        max_p99_ms=1000.0,
    )

    assert breaches == []


def test_both_thresholds_can_breach_together() -> None:
    """失败率与 p99 同时越线 ⇒ 两条都报（不是短路只报一条）。"""
    samples = [_ok(latency_ms=5.0) for _ in range(94)] + [
        _ok(500, latency_ms=5000.0) for _ in range(6)
    ]

    breaches = evaluate_thresholds(
        _result(samples),
        max_failure_rate=0.01,
        max_p99_ms=1000.0,
    )

    assert [b.name for b in breaches] == ["失败率", "p99 延迟"]


def test_threshold_is_strict_greater_than() -> None:
    """恰好等于上限 ⇒ **通过**（判定用 ``>`` 而非 ``>=``）。

    ⚠️ 边界语义必须显式测：写成 ``>=`` 会让一个把上限配成 1% 的人，
        在实测正好 1% 时看到失败 —— 一个与配置字面意思相悖的红灯。
    """
    at_limit = _result([_ok() for _ in range(99)] + [_ok(500)])

    assert evaluate_thresholds(at_limit, max_failure_rate=0.01, max_p99_ms=1000.0) == []
    # 反例：上限压到 0.9% 时，同样这 1% 就必须越线。
    assert [b.name for b in evaluate_thresholds(at_limit, max_failure_rate=0.009)] == ["失败率"]


def test_p99_exactly_at_the_limit_passes() -> None:
    """★★★ p99 **恰好等于**上限 ⇒ 通过（p99 用的也是 ``>`` 而非 ``>=``）。

    ⚠️ 上面那条 ``test_threshold_is_strict_greater_than`` 只钉住了**失败率**
    那条边界（它的样本 p99 是 500ms、上限 1000ms，离边界很远）。
    p99 那条判定是**另一行代码**，写成 ``>=`` 能通过全部用例 ——
    而它的症状与失败率那条一样：把上限配成 800ms 的人，在实测正好
    800ms 时看到一个与配置字面意思相悖的红灯。

    ⚠️ 构造方式必须让 p99 **正好**落在上限上：N=100 时 p99 是第
    ⌈99×100/100⌉ = 99 个样本，所以让第 99 小的样本恰好等于 1000ms、
    第 100 个（最大的那个）高于它。只放一个「1000ms」是不够的 ——
    那样 p99 会取到更大的那个，边界根本没被碰到。
    """
    samples = (
        [_ok(latency_ms=5.0) for _ in range(98)]
        + [_ok(latency_ms=1000.0)]  # 第 99 小 ⇒ 就是 p99
        + [_ok(latency_ms=2000.0)]  # 最大的那个，p100 才用它
    )
    result = _result(samples)

    assert result.percentiles((99,))[99] == 1000.0, "样本没把 p99 摆在边界上，这条用例就白写了"

    assert evaluate_thresholds(result, max_failure_rate=0.01, max_p99_ms=1000.0) == [], (
        "p99 恰好等于上限时被判成了越线 —— p99 的判定写成了 >= 而不是 >？"
    )
    # 反例：上限压到 999ms 时，同样这个 p99 就必须越线。
    assert [
        b.name for b in evaluate_thresholds(result, max_failure_rate=0.01, max_p99_ms=999.0)
    ] == ["p99 延迟"]


def test_zero_samples_is_a_breach_not_a_pass() -> None:
    """★ 一个请求都没发出 ⇒ **必须判失败**。

    ⚠️ 这是本文件最重要的一条反例。空结果下失败率 0%、p99 0ms —— 若不加
        这一条，两种阈值都会「通过」，脚本以退出码 0 结束并打印「✅ 通过」。
        那是最坏的一种假绿灯：什么都没压，报告说一切正常。
        真实成因很常见：地址里 Host 打错导致每个请求都瞬间失败（尚未计入），
        或并发/时长为 0 被上游拦下。
    """
    breaches = evaluate_thresholds(_result([]), max_failure_rate=0.01, max_p99_ms=1000.0)

    assert [b.name for b in breaches] == ["样本数"]
    assert "一个请求都没有发出" in breaches[0].message


def test_render_report_prints_the_breached_threshold_name(capsys: Any) -> None:
    """报告尾部 ⇒ 打印被突破的阈值名（供人一眼看出是哪一条）。"""
    samples = [_ok() for _ in range(95)] + [_ok(500) for _ in range(5)]

    breaches = render_report(
        _result(samples),
        health_only=False,
        max_failure_rate=0.01,
        max_p99_ms=1000.0,
    )
    out = capsys.readouterr().out

    assert [b.name for b in breaches] == ["失败率"]
    assert "失败率" in out
    assert "❌" in out


# ==============================================================================
# 四、参数解释（与 argparse 解耦，直接测语义）
# ==============================================================================
def test_health_only_pins_the_endpoint() -> None:
    """``--health-only`` ⇒ 端点被钉成 ``/healthz``。"""
    plan = resolve_plan(health_only=True, endpoint=None, duration=None, requests=None)

    assert plan.endpoint == loadtest.HEALTH_ENDPOINT
    assert plan.duration_seconds == loadtest.DEFAULT_DURATION_SECONDS
    assert plan.total_requests is None


def test_default_endpoint_when_nothing_given() -> None:
    """端点、模式都不给 ⇒ 默认 ``/healthz`` + 计时模式。"""
    plan = resolve_plan(health_only=False, endpoint=None, duration=None, requests=None)

    assert plan.endpoint == loadtest.HEALTH_ENDPOINT
    assert plan.duration_seconds == loadtest.DEFAULT_DURATION_SECONDS


def test_explicit_endpoint_is_respected() -> None:
    """给了 ``--endpoint`` ⇒ 用它，且不进入默认计时模式判断。"""
    plan = resolve_plan(
        health_only=False,
        endpoint="/api/v1/health",
        duration=None,
        requests=100,
    )

    assert plan.endpoint == "/api/v1/health"
    assert plan.total_requests == 100
    assert plan.duration_seconds is None


def test_health_only_conflicts_with_explicit_endpoint() -> None:
    """``--health-only`` + ``--endpoint`` ⇒ ``ValueError``（含义冲突，必须报错）。

    ⚠️ 反例用例。静默忽略其中一个会让人以为打的是自己指定的地址 ——
        而报告里那个地址看起来完全合理，错误因此被藏起来。
    """
    with pytest.raises(ValueError):
        resolve_plan(health_only=True, endpoint="/api/v1/health", duration=None, requests=None)


def test_duration_conflicts_with_requests() -> None:
    """``--duration`` + ``--requests`` ⇒ ``ValueError``。"""
    with pytest.raises(ValueError):
        resolve_plan(health_only=False, endpoint=None, duration=5.0, requests=100)


def test_parser_rejects_non_positive_concurrency() -> None:
    """``--concurrency 0`` ⇒ argparse 以退出码 2 拒绝（用法错误）。"""
    with pytest.raises(SystemExit) as excinfo:
        loadtest._build_parser().parse_args(["--concurrency", "0"])

    assert excinfo.value.code == 2


def test_main_rejects_conflicting_flags_with_exit_code_2() -> None:
    """端到端：冲突参数 ⇒ ``main`` 以退出码 2 结束（**不是** 1）。

    ⚠️ 2 与 1 的区分是刻意的：1 表示「压测跑了但没达标」（被测服务的问题），
        2 表示「命令写错了」（调用者的问题）。混为一谈会让 CI 把拼错参数
        误报成服务故障。
    """
    with pytest.raises(SystemExit) as excinfo:
        loadtest.main(["--health-only", "--endpoint", "/api/v1/health"])

    assert excinfo.value.code == 2


# ==============================================================================
# 五、在飞路径（进程内 ASGI，无需 Docker）
# ==============================================================================
async def test_run_load_against_in_process_app() -> None:
    """对进程内 ASGI 应用发 50 个请求 ⇒ 全部 2xx，计数与分位可算。

    这条走的是与 ``--dry-run`` 完全相同的路径（``httpx.ASGITransport``），
    因此它证明的是整个采样 → 分位 → 汇总链路能跑通，而不是靶子服务。

    ⚠️ 靶子被套了一层「最多 50 次请求」的壳（见 :class:`_CappedAsgiApp`）。
    它钉的是一个**比 ``result.total`` 更强**的判据：``result.total`` 只能
    证明「计数器自己数到了 50」，而请求数上限证明的是「网络上真的只发了 50 次」。
    ``run_load`` 若多发了请求却漏记，前者照样绿。
    """
    target = _CappedAsgiApp(loadtest._build_self_test_app(), cap=50)
    transport = httpx.ASGITransport(app=target)

    result = await _run_load_bounded(
        base_url="http://loadtest.local",
        endpoint=loadtest.HEALTH_ENDPOINT,
        concurrency=5,
        total_requests=50,
        transport=transport,
    )

    assert result.total == 50
    assert result.successes == 50
    assert result.failures == 0
    assert result.failure_rate == 0.0
    assert result.throughput > 0
    assert result.percentiles()[50] >= 0.0
    assert target.seen["n"] == 50, "发到靶子上的请求数应与 total_requests 相等"


async def test_run_load_counts_transport_failures() -> None:
    """传输层全部失败 ⇒ 失败率 100%、样本状态码为 ``None``、原因被聚合。

    ⚠️ 反例用例。它钉住「连不上不是异常中断，而是一个 100% 失败的结果」——
        压测一个没起来的服务，本就该得到「失败率 100%」这个结论，
        而不是让脚本自己崩掉、留下一句与压测无关的 traceback。
    """

    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport, seen = _capped_transport(_boom, cap=20)

    result = await _run_load_bounded(
        base_url="http://loadtest.local",
        endpoint=loadtest.HEALTH_ENDPOINT,
        concurrency=4,
        total_requests=20,
        transport=transport,
    )

    assert result.total == 20
    assert result.successes == 0
    assert result.failure_rate == 1.0
    assert all(s.status is None for s in result.samples)
    assert result.transport_errors()
    assert seen["n"] == 20, "20 次请求全部失败，但**一共只该发 20 次**"

    breaches = evaluate_thresholds(result, max_failure_rate=0.01, max_p99_ms=1000.0)
    assert [b.name for b in breaches] == ["失败率"]


async def test_run_load_does_not_follow_redirects() -> None:
    """★★★ 目标返回 302 ⇒ 记为**失败**的 302，而不是跟到终点后的 200。

    ⚠️ ``follow_redirects=False`` 此前只被间接验证过（``Sample`` 层面
    断言「3xx 算失败」），**从没有**让一个真的客户端去碰一个真的返回
    重定向的靶子。于是把它改成 ``True`` 能通过全部用例 —— 而它掩盖的
    正是负载测试要发现的东西：一个返回 302 的路由是**坏掉的路由**
    （比如反代的尾斜杠规则配错），跟随它会把每个请求都变成 200，
    报告一片绿，而真实用户拿到的是浏览器跳转或一个 404。

    ⚠️ 靶子必须让「跟」与「不跟」得到**不同**的结果：目标路径返回 302，
    跳转后的路径返回 200。若跳转后的路径也是 302，跟随与否都是失败，
    这条用例就抓不住任何东西。
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == loadtest.HEALTH_ENDPOINT:
            return httpx.Response(
                302,
                headers={"location": "http://loadtest.local/elsewhere"},
                request=request,
            )
        return httpx.Response(200, request=request)

    # ⚠️ 上限 20（= total_requests）在这里有**双重**作用：
    #    · 挡住「协程不退出」把用例挂死（见 :class:`_RunawayLoad`）；
    #    · 顺带证明**没有跟随跳转** —— 一旦 ``follow_redirects=True``，
    #      每个请求会打到靶子两次（一次 302、一次跳转后的 200），
    #      20 次请求就变成 40 次，上限当场炸掉。
    #      下面那条 ``status == 302`` 断言只看得见 `Sample`，看不见网络；
    #      这条上限才真的看得见「请求打了几次」。
    transport, seen = _capped_transport(_handler, cap=20)

    result = await _run_load_bounded(
        base_url="http://loadtest.local",
        endpoint=loadtest.HEALTH_ENDPOINT,
        concurrency=4,
        total_requests=20,
        transport=transport,
    )

    assert all(sample.status == 302 for sample in result.samples), (
        "请求跟到了跳转终点 —— follow_redirects 被改成了 True？"
        " 这会掩盖一个坏掉的路由。"
    )
    assert result.successes == 0
    assert result.failure_rate == 1.0
    assert seen["n"] == 20, "跟随跳转会让请求数翻倍 —— 这里必须恰好是 20"


async def test_run_load_duration_mode_stops_at_deadline() -> None:
    """计时模式 ⇒ 到点即停，且至少发过一次请求。"""
    transport = httpx.ASGITransport(app=loadtest._build_self_test_app())

    result = await run_load(
        base_url="http://loadtest.local",
        endpoint=loadtest.HEALTH_ENDPOINT,
        concurrency=5,
        duration_seconds=0.2,
        transport=transport,
    )

    assert result.mode == "duration"
    assert result.total >= 1
    # ⚠️ 必须带容差：``elapsed_seconds`` 是**实测**墙钟时间，而「到点即停」
    # 是「循环发现已过点就退出」—— 停下的那一刻由调度精度决定，可能比
    # 截止时间早一个亚毫秒级的尾巴（实测 0.19999965699389577，恰好落在
    # 0.2 之下，断言随机变红，而脚本本身没有任何问题）。
    # 这里断言的是「确实跑满了约定期限」，不是「计时精确到微秒」。
    assert result.elapsed_seconds >= 0.2 - 0.02, (
        f"计时模式提前停得太早：{result.elapsed_seconds}s（约定 0.2s）"
    )


async def test_run_load_rejects_two_modes_at_once() -> None:
    """同时给时长与请求数 ⇒ ``ValueError``（调用方须二选一）。"""
    with pytest.raises(ValueError):
        await run_load(
            base_url="http://loadtest.local",
            endpoint=loadtest.HEALTH_ENDPOINT,
            duration_seconds=1.0,
            total_requests=10,
        )


async def test_run_load_rejects_zero_concurrency() -> None:
    """并发为 0 ⇒ ``ValueError``（否则一个工作协程都不会起，静默发出 0 个请求）。"""
    with pytest.raises(ValueError):
        await run_load(
            base_url="http://loadtest.local",
            endpoint=loadtest.HEALTH_ENDPOINT,
            concurrency=0,
            total_requests=10,
        )


# ==============================================================================
# 六、自检端到端：--dry-run 必须能在没有 Docker 的情况下跑通
# ==============================================================================
def _patch_self_test_app(monkeypatch: Any, *, cap: int) -> _CappedAsgiApp:
    """把自检靶子换成「最多接受 ``cap`` 次请求」的那一个。

    ⚠️ 这两条用例走的是 :func:`loadtest.main`，传输层由**被测代码**自己构造，
    没有注入口（也不该为了测试给它开一个）。所以这里 monkeypatch 掉
    :func:`loadtest._build_self_test_app`，让生产代码自己去包那个壳 ——
    生产代码一行不用改，同样拿到请求数上限。

    Args:
        monkeypatch (`Any`): pytest 的 monkeypatch fixture。
        cap (`int`): 允许的最大请求数。

    Returns:
        `_CappedAsgiApp`: 包装后的靶子（``.seen["n"]`` 是实际请求数）。
    """
    target = _CappedAsgiApp(loadtest._build_self_test_app(), cap=cap)
    monkeypatch.setattr(loadtest, "_build_self_test_app", lambda: target)
    return target


def test_dry_run_end_to_end_passes(capsys: Any, monkeypatch: Any) -> None:
    """``--dry-run --requests 40`` ⇒ 退出码 0，报告含目标与判定。

    ★ 这是「脚本自身可测」这条验收的直接体现：整条 CLI（解析 → 采样 →
    分位 → 阈值 → 退出码）在没有 Docker、没有网络的机器上跑完。
    """
    target = _patch_self_test_app(monkeypatch, cap=40)

    code = loadtest.main(["--dry-run", "--requests", "40", "--concurrency", "8"])
    out = capsys.readouterr().out

    assert code == 0
    assert "自检模式" in out
    assert "✅ 负载测试通过" in out
    assert target.seen["n"] == 40, "dry-run 也走计数模式 —— 靶子收到的请求数应当不多不少"


def test_dry_run_defaults_to_request_mode(capsys: Any, monkeypatch: Any) -> None:
    """``--dry-run`` 且不给模式 ⇒ 用**计数模式**（几秒内跑完，不压满 10 秒）。"""
    target = _patch_self_test_app(monkeypatch, cap=loadtest.DRY_RUN_REQUESTS)

    code = loadtest.main(["--dry-run", "--concurrency", "10"])
    out = capsys.readouterr().out

    assert code == 0
    assert f"共 {loadtest.DRY_RUN_REQUESTS} 个请求" in out
    assert target.seen["n"] == loadtest.DRY_RUN_REQUESTS


__all__: list[str] = []
