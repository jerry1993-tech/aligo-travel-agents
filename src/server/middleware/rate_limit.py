# -*- coding: utf-8 -*-
"""限流中间件：按**身份**给请求发令牌，发完了就返回 429。

文件职责：
    用一个进程内的令牌桶拦住两类请求：暴力枚举（拿别人的会话 id 猛试）与
    失控客户端（前端 bug 导致的死循环重试）。拦不住的那一类是分布式刷量 ——
    见下面的「进程内计数器的边界」。

上下游依赖：
    - 上游：``src/server/app.py`` 注册，位于中间件链的
      ``TraceContext → HttpMetrics → Auth → RateLimit → 应用`` 中的第四层 ——
      **必须在 Auth 之内**，因为它要读 Auth 解析好的身份来当限流键。
    - 下游：无（直接构造 429 响应）。

==============================================================================
为什么是令牌桶，不是「每窗口计数」
==============================================================================
    固定窗口计数会在**窗口边界**出问题：窗口末尾打满 120 次、下一秒窗口重置
    再打满 120 次，等于在 0.2 秒里放行了 240 次 —— 突发是配置值的两倍。
    令牌桶没有这个问题：令牌按 ``requests_per_window / window_seconds``
    匀速回填，桶容量就是允许的瞬时突发上限，与配置值严格一致。

==============================================================================
进程内计数器的边界（必须写下来）
==============================================================================
    计数器是 ``__call__`` 里的一个普通 dict。本项目 ``workers=1`` 且单实例，
    这是**正确**的选择：零额外延迟、零新依赖、不需要 Redis 在请求路径上。

    一旦扩到多副本，每个副本各限一份 ⇒ 实际放行量是配置值的 N 倍，
    而且**不会报错**、日志里也看不出异常 —— 与「SSE 必须用 RedisMessageBus」
    是同一类问题：单进程内自洽，多副本下静默失效。
    扩副本前请把 :meth:`RateLimitMiddleware._consume` 换成基于 Redis 的实现，
    那个文件里会需要一段 Lua 或 ``INCR`` + ``EXPIRE`` 的原子操作。

==============================================================================
为什么不需要加锁
==============================================================================
    ``_consume`` 是**同步**函数：从读桶、算令牌、写回桶，中间没有任何 ``await``。
    在 asyncio 的单事件循环里，一个协程不 await 就不会被切走，
    因此「读—改—写」天然是原子的。加一把 ``asyncio.Lock`` 反而会引入
    真实的开销，并让人误以为这个中间件支持多线程 —— 它不支持，
    多线程部署（``workers>1``）要靠上面说的 Redis 方案解决。
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from ...observability import metrics as metrics_mod
from ._asgi import Message, Receive, Scope, Send, send_json
from .auth import USER_ID_STATE_KEY

logger = logging.getLogger(__name__)


class _Bucket:
    """一个身份的令牌桶状态。

    用 ``__slots__`` 而不是 dataclass：这个对象会按身份数量创建，
    数量级是「同时活跃的调用者」，每个实例省下几十字节的 ``__dict__``
    在 :data:`~src.config.schema.RateLimitSettings.max_keys` 达到上限时是有意义的。
    """

    __slots__ = ("tokens", "updated")

    def __init__(self, tokens: float, updated: float) -> None:
        """初始化桶。

        Args:
            tokens (`float`): 当前可用令牌数。
            updated (`float`): 上次回填时的时间戳（来自单调时钟）。
        """
        self.tokens = tokens
        self.updated = updated


class RateLimitMiddleware:
    """纯 ASGI 令牌桶限流中间件。"""

    def __init__(
        self,
        app: Any,
        *,
        settings: Any,
        clock: Any = time.monotonic,
    ) -> None:
        """保存下游应用并展开限流配置。

        Args:
            app (`Any`): 下游 ASGI 应用。
            settings (`Settings`): 全量配置，只读 ``settings.ratelimit``。
            clock (`Callable[[], float]`): 单调时钟。**可注入**，用例据此把
                时间快进过去测「冷却后恢复」，而不必真的 ``sleep(60)``——
                那会让一个 0.1 秒的用例变成一分钟，最终被标记成 slow 而跳过。
                用 ``time.monotonic`` 而不是 ``time.time``：后者会被 NTP 校正、
                夏令时甚至手动改系统时间影响，一次向后的跳变会让令牌
                凭空回满（限流失效），向前跳变则让所有人被误伤。
        """
        self.app = app
        cfg = settings.ratelimit
        self._enabled: bool = bool(cfg.enabled)
        self._capacity: float = float(cfg.requests_per_window)
        # 每秒回填的令牌数。window_seconds 在 schema 里有 gt=0 约束，不会为零。
        self._refill_per_second: float = cfg.requests_per_window / cfg.window_seconds
        self._max_keys: int = int(cfg.max_keys)
        self._exempt_paths: tuple[str, ...] = tuple(cfg.exempt_paths)
        self._buckets: dict[str, _Bucket] = {}
        self._clock = clock

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """处理一次 ASGI 调用。

        Args:
            scope (`Scope`): ASGI scope。
            receive (`Receive`): 接收可调用对象。
            send (`Send`): 发送可调用对象。
        """
        if scope.get("type") != "http" or not self._enabled:
            await self.app(scope, receive, send)
            return

        path = scope.get("path") or ""
        # 探针免限流（见 config/base.yaml 的 ratelimit.exempt_paths 注释）：
        # 被限流的 /healthz 会让编排系统误判容器已死并重启它。
        if path.startswith(self._exempt_paths):
            await self.app(scope, receive, send)
            return

        key, key_kind = self._key_for(scope)
        allowed, retry_after = self._consume(key)

        if not allowed:
            metrics_mod.observe_rate_limited(key_kind)
            # ⚠️ 把**身份本身**写进日志是不行的（那是 PII，且日志会被采集），
            #    所以只记来源类型与路径。要定位到具体是谁，用指标标签 + trace_id。
            logger.warning(
                "请求被限流：key_kind=%s path=%s（容量 %d / %.1fs）",
                key_kind,
                path,
                int(self._capacity),
                1 / self._refill_per_second if self._refill_per_second else 0,
            )
            await send_json(
                send,
                429,
                {
                    "detail": "请求过于频繁，请稍后重试。",
                    "retry_after_seconds": retry_after,
                },
                extra_headers=[
                    # Retry-After 是**秒数**（RFC 9110 §10.2.3）。
                    # 返回它比只给 429 有用得多：客户端可以据此退避，
                    # 而不是立刻重试把桶打得更空 —— 后者会让限流变成
                    # 「谁重试得快谁更吃亏」的恶性循环。
                    (b"retry-after", str(retry_after).encode("ascii")),
                ],
            )
            return

        await self.app(scope, receive, send)

    # --------------------------------------------------------------------------
    def _key_for(self, scope: Scope) -> tuple[str, str]:
        """决定这次请求的限流键与它的来源类型。

        Args:
            scope (`Scope`): ASGI scope。

        Returns:
            `tuple[str, str]`: ``(限流键, 来源类型)``。来源类型是指标标签，
            取值 ``user`` / ``ip`` —— 二者比例失衡说明鉴权链路没生效
            （见 :data:`src.observability.metrics.RATE_LIMITED_TOTAL`）。
        """
        state = scope.get("state")
        if isinstance(state, dict):
            user_id = state.get(USER_ID_STATE_KEY)
            if isinstance(user_id, str) and user_id:
                # 加前缀是**必须**的：身份与 IP 共用一个 dict，
                # 不加前缀时一个恰好叫 ``10.0.0.1`` 的用户会与那个 IP 共享桶，
                # 于是「某个用户刷爆」会连带把整个出口 IP 后面的所有人限住。
                return f"u:{user_id}", "user"

        # 身份缺失时回落到客户端地址。⚠️ **刻意不看 X-Forwarded-For**：
        # 那个头是客户端可以随便写的，直接信它等于让攻击者换一个头就换个桶，
        # 限流形同虚设。要按真实客户端 IP 限流，正确做法是在**可信的反向代理**上
        # 剥离外来的 XFF、再由此处显式配置信任 —— 那是一件需要部署信息才能拍板的事，
        # 不该由这里默默替运维决定。当前的折中是「按直连地址限流」，
        # 在单实例直连场景下它就是真实客户端 IP。
        client = scope.get("client")
        host = client[0] if isinstance(client, (tuple, list)) and client else "unknown"
        return f"ip:{host}", "ip"

    def _consume(self, key: str) -> tuple[bool, int]:
        """从桶里取一个令牌。**同步函数，见模块文档字符串的「为什么不需要加锁」**。

        Args:
            key (`str`): 限流键。

        Returns:
            `tuple[bool, int]`: ``(是否放行, 建议等待秒数)``。放行时秒数为 0。
        """
        now = self._clock()
        bucket = self._buckets.get(key)

        if bucket is None:
            self._prune_if_needed(now)
            bucket = _Bucket(tokens=self._capacity, updated=now)
            self._buckets[key] = bucket
        else:
            # 惰性回填：不跑后台定时器，只在这个键被再次访问时按流逝时间补令牌。
            # 空闲的键因此不消耗任何 CPU —— 这正是选惰性而不是周期扫描的理由。
            elapsed = now - bucket.updated
            if elapsed > 0:
                bucket.tokens = min(
                    self._capacity,
                    bucket.tokens + elapsed * self._refill_per_second,
                )
                bucket.updated = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return True, 0

        # 还差 (1 - tokens) 个令牌，按回填速率折算成秒数。向上取整到整秒 ——
        # Retry-After 只接受整数秒，而向下取整会让客户端在令牌恰好还差一点时
        # 立刻重试、再吃一个 429。
        missing = 1.0 - bucket.tokens
        retry_after = max(1, math.ceil(missing / self._refill_per_second))
        return False, retry_after

    def _prune_if_needed(self, now: float) -> None:
        """在字典达到容量上限时腾位置。

        分两步，顺序不能反：

            1. 先丢**按惰性回填推算已经是满的**桶 —— 它们与「不存在的桶」
               完全等价（下次访问会重新建一个满桶），丢掉的代价是零；
            2. 仍然占满上限时，按**先进先出**淘汰最老的键。

        ⚠️ 这里必须**按回填推算**，而不能直接看 ``b.tokens``。
            ``tokens`` 是**上次访问那一刻**的陈旧值（惰性回填的定义：
            空闲的键不消耗任何 CPU）。一个空转了一小时的桶，
            ``tokens`` 字段里可能还是 ``0``，但它的**实际**额度早已回满。
            照字段直接比，第 1 步几乎永远不会命中 —— 一段看起来在
            「优待空闲者」的逻辑，实际从未生效，而它读起来完全合理。
            （这个缺陷是写 ``test_pruning_drops_full_buckets_first``
            时才暴露出来的：那条用例一开始怎么写都测不出差别。）

        ⚠️ 为什么必须有这一步：限流键来自请求头。没有上限时，一个伪造身份的
        脚本每发一个请求就新建一个桶，字典无限增长 —— 一次外部扫描就
        升级成一次内存耗尽。**限流器自己成了攻击面**，这比不设限流更糟：
        不设限流只是没有保护，设错了是主动提供了一个 DoS 入口。

        用 FIFO 而不是 LRU：字典的迭代顺序就是插入顺序，淘汰 ``next(iter(...))``
        是 O(1) 且确定性的；真做 LRU 需要在每次命中时移动元素，
        而这个字典的读写发生在**每一个请求**的热路径上。

        Args:
            now (`float`): 当前单调时钟读数。
        """
        if len(self._buckets) < self._max_keys:
            return

        full = [
            key
            for key, bucket in self._buckets.items()
            # max(0.0, ...) 防的是时钟倒退（注入的假时钟、或系统时间被改）：
            # 负的流逝时间会让 tokens 被算少，进而把一个「其实还欠着令牌」
            # 的桶误判成满桶丢掉 —— 那等于凭空给它发了一批令牌。
            if bucket.tokens
            + max(0.0, now - bucket.updated) * self._refill_per_second
            >= self._capacity
        ]
        for key in full:
            del self._buckets[key]

        while len(self._buckets) >= self._max_keys:
            oldest = next(iter(self._buckets))
            del self._buckets[oldest]
            logger.debug("限流桶字典已满，淘汰最老的键。")


__all__ = ["RateLimitMiddleware"]
