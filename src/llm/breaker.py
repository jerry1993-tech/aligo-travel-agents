# -*- coding: utf-8 -*-
"""熔断器 —— 本项目**自研**的组件（AgentScope 框架没有这个能力）。

文件职责：
    在下游模型持续失败时**主动切断**调用，避免把已经不堪重负的对端压垮，
    同时让自己的请求快速失败（而不是每个请求都白等一整个超时）。
    冷却期满后放**一个**请求过去试探，成功则恢复、失败则继续熔断。

上下游依赖：
    - 上游：由 ``src/llm/factory.py`` 装配；配置项来自
      ``config/base.yaml`` 的 ``llm.circuit_breaker_*``。
    - 下游：``src/server/app.py`` 把它包成 ``on_model_call`` 中间件注入智能体，
      并用 :meth:`CircuitBreaker.snapshot` 把状态暴露给 ``/metrics`` 与 ``/readyz``。

------------------------------------------------------------------------------
它与「重试」的分工（两者都必要，且不可互相替代）
------------------------------------------------------------------------------
    重试   对付**偶发**失败：一次网络抖动、一个 502。特点是「立刻再试就可能成功」。
    熔断   对付**持续**失败：对端整体不可用、key 失效、限流封禁。
           特点是「再试一万次也没用」，而每次都还要白等一个超时。

框架自带的重试在 ``ChatModelBase.__call__`` 里（固定间隔、无指数退避），
本模块补上它缺的那一半。

------------------------------------------------------------------------------
状态机（三态，转换条件写死在这里，不散落）
------------------------------------------------------------------------------
                   连续失败 >= failure_threshold
        ┌────────┐ ─────────────────────────────► ┌──────┐
        │ CLOSED │                                │ OPEN │
        └────────┘ ◄───────────────────────────── └──────┘
             ▲            试探请求成功                 │
             │                                          │ 冷却期满
             │            ┌───────────┐                 │ (recovery_seconds)
             └─────────── │ HALF_OPEN │ ◄───────────────┘
                试探成功   └───────────┘
                                │ 试探失败 → 回到 OPEN，冷却重新计时

**HALF_OPEN 只放一个请求过去**，这是关键：若冷却期满就把闸门全开，
积压的请求会瞬间涌向一个刚恢复（或根本没恢复）的对端，等于没有熔断。
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from enum import Enum
from typing import AsyncIterator


class CircuitState(str, Enum):
    """熔断器的三种状态。

    继承 ``str`` 是为了让它能直接被序列化进 JSON（``/readyz`` 的响应体里要用），
    而不必在每个输出点手写一次转换。
    """

    #: 正常通行。持续累计失败次数。
    CLOSED = "closed"
    #: 已熔断。所有调用**立即失败**，不打网络。
    OPEN = "open"
    #: 冷却期满后的试探态。只放**一个**请求过去，其余仍然立即失败。
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """熔断器处于开路状态时抛出的异常。

    单独定义一个异常类型（而不是复用 ``RuntimeError``）是为了让调用方
    能**精确区分**「被熔断器拦下」与「模型真的报错了」：
        · 前者是**我们主动不发请求**，通常应立即返回降级内容；
        · 后者是**对端返回了错误**，通常应重试或上报。
    二者的处置方式完全不同，混在一起就只能靠字符串匹配来分辨 —— 那太脆弱。
    """

    def __init__(self, name: str, retry_after_seconds: float) -> None:
        """记录是哪个熔断器打开的、还有多久进入试探态。

        Args:
            name (`str`): 熔断器名字（通常是模型名），用于日志与指标区分。
            retry_after_seconds (`float`): 距离下次允许试探还剩多少秒。
        """
        self.name = name
        self.retry_after_seconds = retry_after_seconds
        super().__init__(
            f"熔断器 {name!r} 处于开路状态，"
            f"约 {retry_after_seconds:.1f}s 后放行试探请求。",
        )


class CircuitBreaker:
    """异步安全的熔断器。

    **全进程共享一个实例**（由 ``src/llm/factory.py`` 保证）。这不是优化，
    而是正确性要求：
        · 每个会话各持一个 ⇒ 下游真的挂了，也要等「每个会话各自失败 5 次」
          才熔断，熔断点被推迟了 N 倍，起不到保护作用；
        · 共享一个 ⇒ 全部调用者共同贡献失败计数，故障被最快识别。

    Attributes:
        failure_threshold (`int`): 连续失败多少次后开路。
        recovery_seconds (`float`): 开路后冷却多久进入试探态。
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_seconds: float = 30.0,
        name: str = "llm",
    ) -> None:
        """初始化熔断器。

        Args:
            failure_threshold (`int`): 连续失败阈值，必须 >= 1。
            recovery_seconds (`float`): 冷却时长（秒），必须 > 0。
            name (`str`): 名字，用于日志与指标。

        Raises:
            ValueError: 阈值或冷却时长非法时。这两个参数直接决定保护是否有效，
                取值荒谬（0 次失败就熔断 / 冷却 0 秒）必须当场暴露，
                而不是留到运行期表现为「熔断器好像没生效」。
        """
        if failure_threshold < 1:
            raise ValueError(
                f"failure_threshold 必须 >= 1，实际为 {failure_threshold}；"
                f"取 0 会让熔断器在任何一次调用前就处于开路状态。",
            )
        if recovery_seconds <= 0:
            raise ValueError(
                f"recovery_seconds 必须 > 0，实际为 {recovery_seconds}；"
                f"取 0 会让每次调用都直接进入试探，等于没有熔断。",
            )

        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds

        # ---- 内部状态 ----
        # 用 asyncio.Lock 而不是 threading.Lock：本模块的所有调用点都在事件循环里，
        # 而 threading.Lock 在协程中阻塞的是**整个线程**（含事件循环），
        # 会造成一次持锁就卡住所有并发请求。这是一类很隐蔽的性能事故。
        self._lock = asyncio.Lock()
        self._state: CircuitState = CircuitState.CLOSED
        #: 连续失败计数（成功一次即清零）。
        self._consecutive_failures = 0
        #: 进入 OPEN 的时刻（单调时钟）。None 表示当前不在 OPEN。
        self._opened_at: float | None = None
        #: HALF_OPEN 下是否已有试探请求在飞行中。
        self._probe_in_flight = False

        # ---- 供 /metrics 读取的累计计数（只增不减，便于算速率）----
        self.total_calls = 0
        self.total_failures = 0
        self.total_rejections = 0
        self.total_opens = 0

    # --------------------------------------------------------------------------
    # 只读视图
    # --------------------------------------------------------------------------
    @property
    def state(self) -> CircuitState:
        """当前状态（**不做**时间推进）。

        ⚠️ 注意它不会把「冷却期已过」的 OPEN 自动变成 HALF_OPEN ——
        状态转换只在 :meth:`allow_request` 里发生。若在这里顺手转换，
        一个只读的观测点就带上了副作用，``/readyz`` 每次被探活都会改变状态机，
        行为会变得极难推理。

        Returns:
            `CircuitState`: 当前状态。
        """
        return self._state

    def snapshot(self) -> dict[str, object]:
        """导出一份可直接序列化的状态快照（供 ``/metrics`` 与 ``/readyz`` 使用）。

        Returns:
            `dict`: 含状态名、连续失败数、累计计数，以及（开路时）剩余冷却秒数。
        """
        remaining = 0.0
        if self._state is CircuitState.OPEN and self._opened_at is not None:
            elapsed = time.monotonic() - self._opened_at
            remaining = max(0.0, self.recovery_seconds - elapsed)
        return {
            "name": self.name,
            "state": self._state.value,
            "consecutive_failures": self._consecutive_failures,
            "failure_threshold": self.failure_threshold,
            "recovery_seconds": self.recovery_seconds,
            "open_remaining_seconds": round(remaining, 3),
            "total_calls": self.total_calls,
            "total_failures": self.total_failures,
            "total_rejections": self.total_rejections,
            "total_opens": self.total_opens,
        }

    # --------------------------------------------------------------------------
    # 状态机
    # --------------------------------------------------------------------------
    async def allow_request(self) -> None:
        """判定是否放行本次调用；不放行则抛 :class:`CircuitBreakerOpen`。

        这是**唯一**会推进状态机的地方（OPEN → HALF_OPEN 的转换发生在这里）。

        Raises:
            CircuitBreakerOpen: 当前应拒绝本次调用时。
        """
        async with self._lock:
            if self._state is CircuitState.CLOSED:
                return

            if self._state is CircuitState.OPEN:
                assert self._opened_at is not None
                elapsed = time.monotonic() - self._opened_at
                if elapsed < self.recovery_seconds:
                    self.total_rejections += 1
                    raise CircuitBreakerOpen(
                        self.name,
                        self.recovery_seconds - elapsed,
                    )
                # 冷却期满：转入试探态，放行**这一个**请求。
                self._state = CircuitState.HALF_OPEN
                self._probe_in_flight = True
                return

            # HALF_OPEN：只放行一个试探请求，其余继续拒绝。
            if self._probe_in_flight:
                self.total_rejections += 1
                raise CircuitBreakerOpen(self.name, 0.0)
            self._probe_in_flight = True

    async def record_success(self) -> None:
        """记录一次成功：清零失败计数并**直接闭合**。

        为什么一次成功就闭合（而不是要求连续成功 N 次）：HALF_OPEN 下本来就
        只放了一个请求过去，它成功即说明对端已恢复。要求更多次成功会让恢复
        变得过于迟钝，而这期间所有请求都在被拒绝 —— 恢复慢的代价并不小。
        """
        async with self._lock:
            self.total_calls += 1
            self._consecutive_failures = 0
            self._probe_in_flight = False
            self._state = CircuitState.CLOSED
            self._opened_at = None

    async def record_failure(self) -> None:
        """记录一次失败：累计并视阈值决定是否开路。

        在 HALF_OPEN 下失败会**立即**回到 OPEN 并重新计时冷却 —— 试探失败说明
        对端没有恢复，此时再放请求过去毫无意义。
        """
        async with self._lock:
            self.total_calls += 1
            self.total_failures += 1
            self._consecutive_failures += 1
            self._probe_in_flight = False

            # 已经在开路状态下的失败（例如试探前就被拒），不重复触发开路。
            if self._state is CircuitState.OPEN:
                self._opened_at = time.monotonic()
                return

            if (
                self._state is CircuitState.HALF_OPEN
                or self._consecutive_failures >= self.failure_threshold
            ):
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self.total_opens += 1

    async def release_probe(self) -> None:
        """归还 HALF_OPEN 的试探名额，**不记账**（既不算成功也不算失败）。

        ⚠️ 为什么必须有这个方法：:meth:`allow_request` 在放行试探请求时把
        ``_probe_in_flight`` 置为 ``True``，而只有 :meth:`record_success` /
        :meth:`record_failure` 会把它清掉。若调用方在试探进行到一半时退出、
        且两个 record_* 都不调（例如请求被 ``asyncio.CancelledError`` 取消，
        或抛出了一个「不该计入熔断」的调用方错误），这个标志就**永远是
        True**：

            HALF_OPEN + _probe_in_flight=True ⇒ 后续每个请求都在
            allow_request 里被 ``CircuitBreakerOpen(name, 0.0)`` 拒绝，
            而 state 永远停在 HALF_OPEN、冷却逻辑也不再推进
            —— 直到**进程重启**为止。

        对全进程共享的检索熔断器（``src/knowledge/guard.py``）而言，那等于
        检索能力被永久关闭，且 ``/readyz`` 上只看得到一个「half_open」。

        语义上「归还」也是正确的处置：试探请求既然没走完，它的结果就不是
        证据 —— 既不能证明对端已恢复（不算成功），也不能证明对端没恢复
        （不算失败）。把名额还回去，让**下一个**请求可以再次试探。

        ⚠️ 在 CLOSED / OPEN 下调用它是无害的 no-op（这两态本来就不持有
        试探名额），所以调用方不必先判断状态 —— 少一个判断就少一处
        写错的机会。
        """
        async with self._lock:
            self._probe_in_flight = False

    # --------------------------------------------------------------------------
    # 便捷上下文管理器
    # --------------------------------------------------------------------------
    @asynccontextmanager
    async def guard(self) -> AsyncIterator[None]:
        """把一段调用包进熔断保护里。

        用法::

            async with breaker.guard():
                result = await model(messages)

        正常结束时自动记账为成功；抛出任何异常（``CircuitBreakerOpen`` 除外）
        都记账为失败并**原样重新抛出** —— 熔断器只做统计与拦截，
        绝不能吞掉或改写业务异常，否则上层的错误处理会看到错误的类型。

        ⚠️ 体里抛出 ``CircuitBreakerOpen`` 时会把试探名额**还回去**：
        此时本次调用没有触达对端、不计失败（见 Raises），但若它就握着
        HALF_OPEN 的唯一名额而不归还，熔断器就再也放不出下一个试探
        —— 详见 :meth:`release_probe` 的说明。

        Yields:
            `None`: 只是一个占位，调用方不需要它。

        Raises:
            CircuitBreakerOpen: 熔断器拒绝放行时（此时不计入失败 ——
                被拒绝的请求根本没有触达对端，把它算作「失败」会让
                连续失败计数虚高，进而延长熔断时间）。
        """
        await self.allow_request()
        try:
            yield
        except CircuitBreakerOpen:
            # ⚠️ allow_request 若在上面就抛了，压根没取到名额，也走不到这里；
            # 能走到这里说明本次调用**是**被放行的那个（可能正握着试探名额）。
            await self.release_probe()
            raise
        except BaseException:
            await self.record_failure()
            raise
        else:
            await self.record_success()


__all__ = [
    "CircuitBreaker",
    "CircuitBreakerOpen",
    "CircuitState",
]
