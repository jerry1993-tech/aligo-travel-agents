# -*- coding: utf-8 -*-
"""限流中间件的单元测试（``src/server/middleware/rate_limit.py``）。

==============================================================================
为什么时间由参数注入，而不是 ``sleep`` 过去
==============================================================================
    测「令牌回满后恢复放行」需要让时间前进。真的 ``await asyncio.sleep(60)``
    会把一个 0.1 秒的用例变成一分钟 —— 而一分钟的用例迟早会被标记成
    ``slow`` 并从默认运行中排除，于是这条保障**在无人察觉的情况下消失**。

    因此 ``RateLimitMiddleware`` 接受一个 ``clock`` 参数。测试里传一个
    自己控制的假时钟，让 60 秒在纳秒内过去。这不是「为了测试而设计的接口」——
    注入时钟本来就是这个中间件唯一能确定性验证的方式。

==============================================================================
本文件保护的头号性质
==============================================================================
    **限流器自己不能成为攻击面。** 限流键来自请求头或客户端地址，
    没有上限时，一个脚本每发一个请求就新建一个桶，字典无限增长 ——
    一次外部扫描就升级成一次内存耗尽。``_prune_if_needed`` 与
    :func:`test_bucket_table_is_bounded` 就是为这条性质存在的。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from src.server.middleware import rate_limit as rl_mod
from src.server.middleware.auth import USER_ID_STATE_KEY
from src.server.middleware.rate_limit import RateLimitMiddleware


class _FakeRateLimit:
    """``settings.ratelimit`` 的最小替身。"""

    def __init__(
        self,
        *,
        enabled: bool = True,
        requests_per_window: int = 5,
        window_seconds: float = 60.0,
        max_keys: int = 10_000,
        exempt_paths: tuple[str, ...] = ("/healthz", "/readyz", "/metrics"),
    ) -> None:
        """记录各项限流配置。

        Args:
            enabled (`bool`): 是否启用限流。
            requests_per_window (`int`): 窗口内允许的请求数（也是桶容量）。
            window_seconds (`float`): 窗口秒数。
            max_keys (`int`): 桶字典的键上限。
            exempt_paths (`tuple[str, ...]`): 免限流路径前缀。
        """
        self.enabled = enabled
        self.requests_per_window = requests_per_window
        self.window_seconds = window_seconds
        self.max_keys = max_keys
        self.exempt_paths = exempt_paths


def _settings(**kwargs: Any) -> Any:
    """造一个只带 ``ratelimit`` 的 settings 替身。

    Args:
        **kwargs: 透传给 :class:`_FakeRateLimit`。

    Returns:
        `Any`: 带 ``ratelimit`` 属性的对象。
    """

    class _S:
        ratelimit = _FakeRateLimit(**kwargs)

    return _S()


class _Clock:
    """可手动推进的单调时钟。"""

    def __init__(self, now: float = 1000.0) -> None:
        """初始化时钟。

        Args:
            now (`float`): 初始时间戳（秒）。
        """
        self.now = now

    def __call__(self) -> float:
        """返回当前时间。

        Returns:
            `float`: 当前时间戳。
        """
        return self.now

    def advance(self, seconds: float) -> None:
        """把时间向前推进。

        Args:
            seconds (`float`): 推进的秒数。
        """
        self.now += seconds


class _Recorder:
    """记录下游是否被调用的 ASGI 应用。"""

    def __init__(self) -> None:
        """初始化记录器。"""
        self.called = False

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        """记录并返回 200。

        Args:
            scope (`Any`): ASGI scope。
            receive (`Any`): 接收可调用对象。
            send (`Any`): 发送可调用对象。
        """
        self.called = True
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            },
        )
        await send({"type": "http.response.body", "body": b"{}"})


def _scope(
    path: str = "/api/v1/me",
    *,
    user_id: str | None = None,
    client: tuple[str, int] | None = ("10.0.0.1", 5000),
) -> dict[str, Any]:
    """造一个 HTTP scope。

    Args:
        path (`str`): 请求路径。
        user_id (`str | None`): 若给出，模拟 Auth 中间件已写入的身份。
        client (`tuple[str, int] | None`): 客户端地址。

    Returns:
        `dict`: ASGI scope。
    """
    scope: dict[str, Any] = {"type": "http", "method": "GET", "path": path, "headers": []}
    if user_id is not None:
        scope["state"] = {USER_ID_STATE_KEY: user_id}
    if client is not None:
        scope["client"] = client
    return scope


async def _call(
    middleware: RateLimitMiddleware,
    scope: dict[str, Any],
) -> tuple[int, list[tuple[bytes, bytes]], dict[str, Any]]:
    """驱动一次中间件调用。

    Args:
        middleware (`RateLimitMiddleware`): 待测中间件。
        scope (`dict`): ASGI scope。

    Returns:
        `tuple[int, list, dict]`: ``(状态码, 响应头, 响应体)``。
    """
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        """收集下游发出的 ASGI 消息。"""
        sent.append(message)

    async def receive() -> dict[str, Any]:
        """本中间件不消费请求体。"""
        return {"type": "http.request", "body": b"", "more_body": False}

    await middleware(scope, receive, send)

    start = next((m for m in sent if m["type"] == "http.response.start"), None)
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return (
        int(start["status"]) if start else 0,
        list(start.get("headers", [])) if start else [],
        json.loads(body) if body else {},
    )


# ==============================================================================
# 一、开关与豁免
# ==============================================================================
@pytest.mark.asyncio
async def test_disabled_passes_everything() -> None:
    """关闭限流后，无论打多少请求都放行。"""
    recorder = _Recorder()
    mw = RateLimitMiddleware(recorder, settings=_settings(enabled=False), clock=_Clock())

    for _ in range(100):
        status, _, _ = await _call(mw, _scope())
        assert status == 200


@pytest.mark.parametrize("path", ["/healthz", "/readyz", "/metrics"])
@pytest.mark.asyncio
async def test_exempt_paths_bypass_the_bucket(path: str) -> None:
    """探针路径**完全绕过**令牌桶。

    ⚠️ 这不是「优化」，是**正确性**要求：被限流的 ``/healthz`` 会让编排系统
    误判容器已死并重启它。而探针通常每几秒打一次，恰恰是最容易吃到限流的
    那类流量 —— 一个「健康检查把自己打挂」的经典自指故障。
    """
    recorder = _Recorder()
    # 容量设为 1：任何进入桶的请求都会在第二次被拒。
    mw = RateLimitMiddleware(
        recorder,
        settings=_settings(requests_per_window=1),
        clock=_Clock(),
    )

    for _ in range(50):
        status, _, _ = await _call(mw, _scope(path))
        assert status == 200

    # 顺带证明桶确实是有容量的 —— 否则上面那段可能只是「限流根本没生效」。
    status, _, _ = await _call(mw, _scope("/api/v1/me"))
    assert status == 200
    status, _, _ = await _call(mw, _scope("/api/v1/me"))
    assert status == 429


@pytest.mark.asyncio
async def test_non_http_scope_passes_through() -> None:
    """非 http scope 原样透传。"""
    recorder = _Recorder()
    mw = RateLimitMiddleware(recorder, settings=_settings(), clock=_Clock())
    sent: list[dict[str, Any]] = []

    async def send(message: dict[str, Any]) -> None:
        """收集 ASGI 消息。"""
        sent.append(message)

    await mw({"type": "lifespan"}, None, send)  # type: ignore[arg-type]

    assert recorder.called


# ==============================================================================
# 二、令牌桶语义
# ==============================================================================
@pytest.mark.asyncio
async def test_capacity_equals_requests_per_window() -> None:
    """前 N 个请求通过，第 N+1 个被拒。"""
    recorder = _Recorder()
    mw = RateLimitMiddleware(
        recorder,
        settings=_settings(requests_per_window=3),
        clock=_Clock(),
    )

    for _ in range(3):
        status, _, _ = await _call(mw, _scope(user_id="alice"))
        assert status == 200

    status, _, body = await _call(mw, _scope(user_id="alice"))
    assert status == 429
    assert body["detail"] == "请求过于频繁，请稍后重试。"


@pytest.mark.asyncio
async def test_rejection_carries_retry_after() -> None:
    """429 必须带 ``Retry-After``（RFC 9110 §10.2.3）与响应体里的秒数。

    返回它比只给 429 有用得多：客户端可以据此退避，而不是立刻重试
    把桶打得更空 —— 后者会让限流变成「谁重试得快谁更吃亏」的恶性循环。
    """
    recorder = _Recorder()
    mw = RateLimitMiddleware(
        recorder,
        # 容量 1、窗口 60 ⇒ 回填速率 1/60 令牌每秒 ⇒ 补满 1 个令牌需 60 秒。
        settings=_settings(requests_per_window=1, window_seconds=60.0),
        clock=_Clock(),
    )

    await _call(mw, _scope(user_id="alice"))
    _, headers, body = await _call(mw, _scope(user_id="alice"))

    assert (b"retry-after", b"60") in headers
    assert body["retry_after_seconds"] == 60


@pytest.mark.asyncio
async def test_retry_after_rounds_up() -> None:
    """Retry-After 必须**向上取整**。

    向下取整会让客户端在令牌还差一点时立刻重试、再吃一个 429 ——
    而每吃一次 429 都让客户端更确信「服务在抽风」。
    """
    clock = _Clock()
    recorder = _Recorder()
    mw = RateLimitMiddleware(
        recorder,
        settings=_settings(requests_per_window=2, window_seconds=1.0),
        clock=clock,
    )

    # 容量 2、速率 2/秒。用掉 2 个令牌，再前进 0.1 秒 → 回填 0.2 个。
    await _call(mw, _scope(user_id="alice"))
    await _call(mw, _scope(user_id="alice"))
    clock.advance(0.1)

    _, _, body = await _call(mw, _scope(user_id="alice"))

    # 还差 0.8 个令牌，速率 2/秒 ⇒ 0.4 秒 ⇒ 向上取整为 1 秒。
    assert body["retry_after_seconds"] == 1


@pytest.mark.asyncio
async def test_tokens_refill_over_time() -> None:
    """令牌随时间回填，空桶会恢复。"""
    clock = _Clock()
    recorder = _Recorder()
    mw = RateLimitMiddleware(
        recorder,
        settings=_settings(requests_per_window=2, window_seconds=1.0),
        clock=clock,
    )

    await _call(mw, _scope(user_id="alice"))
    await _call(mw, _scope(user_id="alice"))
    status, _, _ = await _call(mw, _scope(user_id="alice"))
    assert status == 429

    clock.advance(1.0)  # 回填 2 个令牌（桶容量封顶为 2）

    status, _, _ = await _call(mw, _scope(user_id="alice"))
    assert status == 200


@pytest.mark.asyncio
async def test_bucket_never_exceeds_capacity() -> None:
    """空闲很久后的桶**不会**攒出超过容量的突发额度。

    这是令牌桶相对「固定窗口计数」的核心优势：固定窗口在窗口边界
    可以放行两倍突发，令牌桶严格等于配置值。
    """
    clock = _Clock()
    recorder = _Recorder()
    mw = RateLimitMiddleware(
        recorder,
        settings=_settings(requests_per_window=3, window_seconds=1.0),
        clock=clock,
    )

    await _call(mw, _scope(user_id="alice"))
    clock.advance(3600.0)  # 空转一小时

    allowed = 0
    for _ in range(10):
        status, _, _ = await _call(mw, _scope(user_id="alice"))
        allowed += status == 200

    assert allowed == 3, f"长时间空闲后一次性放行了 {allowed} 个，容量是 3"


@pytest.mark.asyncio
async def test_idle_buckets_do_not_consume_cpu() -> None:
    """空闲的键不消耗 CPU —— 惰性回填，没有后台定时器。

    断言方式是：一个从未被访问过的键，不会因为时间流逝而出现在桶字典里。
    """
    clock = _Clock()
    recorder = _Recorder()
    mw = RateLimitMiddleware(recorder, settings=_settings(), clock=clock)

    await _call(mw, _scope(user_id="alice"))
    assert len(mw._buckets) == 1  # noqa: SLF001 —— 白盒断言，见下

    clock.advance(10_000.0)

    assert len(mw._buckets) == 1  # noqa: SLF001


# ==============================================================================
# 三、限流键的身份来源
# ==============================================================================
@pytest.mark.asyncio
async def test_keying_prefers_identity_over_ip() -> None:
    """有身份时按身份限流，**不是**按 IP。

    按 IP 限流在生产会误伤：一栋办公楼、一个运营商出口 NAT 后面
    可能是几千个真实用户，他们共用一个桶。
    """
    clock = _Clock()
    recorder = _Recorder()
    mw = RateLimitMiddleware(recorder, settings=_settings(requests_per_window=1), clock=clock)

    await _call(mw, _scope(user_id="alice"))
    status, _, _ = await _call(mw, _scope(user_id="alice"))
    assert status == 429

    # 同一个 IP 换一个身份 —— 必须放行（桶是按身份分的）。
    status, _, _ = await _call(mw, _scope(user_id="bob"))
    assert status == 200


@pytest.mark.asyncio
async def test_keying_falls_back_to_ip() -> None:
    """没有身份时按客户端地址限流。"""
    clock = _Clock()
    recorder = _Recorder()
    mw = RateLimitMiddleware(recorder, settings=_settings(requests_per_window=1), clock=clock)

    await _call(mw, _scope(client=("10.0.0.1", 5000)))
    status, _, _ = await _call(mw, _scope(client=("10.0.0.1", 5000)))
    assert status == 429

    status, _, _ = await _call(mw, _scope(client=("10.0.0.2", 5000)))
    assert status == 200


@pytest.mark.asyncio
async def test_user_and_ip_keys_do_not_collide() -> None:
    """★ 身份与 IP 的键**不能**撞在一起。

    不加 ``u:`` / ``ip:`` 前缀时，一个恰好把 user id 起成 ``10.0.0.1``
    的用户会与那个 IP 共享同一个桶 —— 于是「某个用户刷爆自己」
    会连带把整个出口 IP 后面的所有人一起限住。
    这类 bug 的排查成本极高：受害者与肇事者之间没有任何可观察的联系。
    """
    clock = _Clock()
    recorder = _Recorder()
    mw = RateLimitMiddleware(recorder, settings=_settings(requests_per_window=1), clock=clock)

    # 用户 "10.0.0.1" 用掉自己那个桶。
    await _call(mw, _scope(user_id="10.0.0.1"))
    status, _, _ = await _call(mw, _scope(user_id="10.0.0.1"))
    assert status == 429

    # 来自地址 10.0.0.1 的**匿名**请求必须不受影响。
    status, _, _ = await _call(mw, _scope(user_id=None, client=("10.0.0.1", 5000)))
    assert status == 200, (
        "身份键与 IP 键发生了碰撞：一个 user_id 恰为 '10.0.0.1' 的用户"
        "会把该 IP 的匿名流量一起限住。"
    )


@pytest.mark.asyncio
async def test_missing_client_address_does_not_crash() -> None:
    """拿不到客户端地址时用 ``unknown`` 兜底，而不是抛 KeyError。

    ASGI 的 ``client`` 字段在 unix socket 等场景下可以是 ``None``。
    为一个「取不到 IP」而 500，会让整个服务在最不该出问题的地方出问题。
    """
    clock = _Clock()
    recorder = _Recorder()
    mw = RateLimitMiddleware(recorder, settings=_settings(requests_per_window=1), clock=clock)

    status, _, _ = await _call(mw, _scope(user_id=None, client=None))

    assert status == 200


# ==============================================================================
# 四、桶字典的上限（限流器自身不能成为攻击面）
# ==============================================================================
@pytest.mark.asyncio
async def test_bucket_table_is_bounded() -> None:
    """★ 桶字典的规模受 ``max_keys`` 约束。

    限流键来自请求头。没有上限时，一个脚本每发一个请求就伪造一个新身份，
    字典无限增长 —— 一次外部扫描就升级成一次内存耗尽。
    **限流器自己成了攻击面**，这比不设限流更糟：不设限流只是没有保护，
    设错了是主动提供了一个 DoS 入口。
    """
    clock = _Clock()
    recorder = _Recorder()
    mw = RateLimitMiddleware(
        recorder,
        settings=_settings(requests_per_window=10, max_keys=50),
        clock=clock,
    )

    for i in range(500):
        await _call(mw, _scope(user_id=f"attacker-{i}"))

    assert len(mw._buckets) <= 50, (  # noqa: SLF001
        f"桶字典涨到了 {len(mw._buckets)}，max_keys 是 50 —— "
        f"伪造身份的请求会让内存无限增长。"
    )


@pytest.mark.asyncio
async def test_pruning_drops_full_buckets_first() -> None:
    """★ 腾位置时**先丢已回满的桶**，而不是无脑按先进先出。

    只做 FIFO 的话，一个正常的、偶尔来一次的调用方会因为「来得早」
    被淘汰，而一个刚刚刷爆桶的攻击者反而被留下 —— 淘汰策略把
    无辜者清掉、把肇事者保住，方向正好反了。

    ⚠️ 这条用例的构造**必须让两种策略给出不同答案**，否则它测不出任何东西：
        本用例里 ``bob`` 是**先**插入的（FIFO 会淘汰他），
        但 ``alice`` 才是**已回满**的那个（正确策略应淘汰她）。
        断言 bob 仍在、alice 已走 —— 两种策略在这里结论相反。

        第一版用例没有做这个区分（只断言 ``len <= max``），
        两种策略都能让它通过 —— 而正是为了把这条用例写成有区分度的，
        才发现 ``_prune_if_needed`` 当时是**按陈旧字段**判断满桶的，
        那一步实际从未命中。
    """
    clock = _Clock()
    recorder = _Recorder()
    # 容量 10、窗口 60 秒 ⇒ 回填速率 1/6 令牌每秒。
    # ⇒ 从 9 个补到 10 个（满）需要 6 秒；从 0 个补满需要 60 秒。
    mw = RateLimitMiddleware(
        recorder,
        settings=_settings(requests_per_window=10, window_seconds=60.0, max_keys=2),
        clock=clock,
    )

    # bob 先插入：打满 10 个令牌，桶空。
    for _ in range(10):
        await _call(mw, _scope(user_id="bob"))
    # alice 后插入：只用掉 1 个，桶里还剩 9 个。
    await _call(mw, _scope(user_id="alice"))

    # 前进 6 秒：alice 补满（9 + 1 = 10），bob 只补到 1 个。
    clock.advance(6.0)

    # carol 是新键 ⇒ 触发淘汰（此时字典恰好等于 max_keys = 2）。
    await _call(mw, _scope(user_id="carol"))

    assert "u:alice" not in mw._buckets, (  # noqa: SLF001
        "alice 的桶已经回满，与「不存在的桶」完全等价，应当优先被淘汰。"
    )
    assert "u:bob" in mw._buckets, (  # noqa: SLF001
        "bob 的桶还欠着 9 个令牌，是有状态的，不该被淘汰 —— "
        "把有状态的桶丢掉等于凭空发一批令牌（限流出现缺口）；"
        "这里失败说明淘汰退化成了先进先出。"
    )


# ==============================================================================
# 五、指标
# ==============================================================================
@pytest.mark.asyncio
async def test_rejections_are_counted_by_key_kind() -> None:
    """被拒的请求按 ``user`` / ``ip`` 分别计数。

    两个标签的比例本身就是一条诊断信息：``ip`` 占比高说明
    鉴权链路没生效（大量请求以匿名身份到达限流器）——
    而这只在「按来源切分」的指标上才看得出来。
    """
    clock = _Clock()
    mw = RateLimitMiddleware(
        _Recorder(),
        settings=_settings(requests_per_window=1),
        clock=clock,
    )
    before = {
        kind: rl_mod.metrics_mod.RATE_LIMITED_TOTAL.labels(key_kind=kind)._value.get()
        for kind in ("user", "ip")
    }

    await _call(mw, _scope(user_id="alice"))
    await _call(mw, _scope(user_id="alice"))  # 被拒 → user
    await _call(mw, _scope(user_id=None, client=("10.0.0.9", 1)))
    await _call(mw, _scope(user_id=None, client=("10.0.0.9", 1)))  # 被拒 → ip

    after = {
        kind: rl_mod.metrics_mod.RATE_LIMITED_TOTAL.labels(key_kind=kind)._value.get()
        for kind in ("user", "ip")
    }

    assert after["user"] == before["user"] + 1
    assert after["ip"] == before["ip"] + 1
