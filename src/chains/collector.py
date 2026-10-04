# -*- coding: utf-8 -*-
"""**思考链的任务收集器** —— 把事件流投影成一份用户看得懂的任务清单。

文件职责：
    实现博客第 281 行描述的 ``TaskCollector``：「管理任务的完整生命周期
    （PENDING、DOING、DONE、FAILED）」，并对外提供订阅（pub-sub）以便
    界面实时刷新。

上下游依赖：
    - 上游：:mod:`src.domain`（:class:`~src.domain.enums.TaskState`）。
    - 下游：``src/chains/events.py`` 把框架的 ``AgentEvent`` 喂进来；
      ``src/orchestration/`` 的中间件把快照推给前端。

═══ ⚠️ 本模块**刻意不 import** ``agentscope`` ═══

与 :mod:`src.domain`、:mod:`src.orchestration.classifier` 同样的理由，但这里
多一条很实际的：任务清单的状态机（谁能迁到谁、重复结束怎么办、满了丢谁）
是本项目里**最容易写错又最难在集成环境里复现**的一段逻辑 —— 它只在「工具
连续失败」「事件乱序到达」「一轮里调了十几个工具」这些边角下才出问题。
把它做成纯数据结构，就能用普通单测把这些边角穷举掉；混进中间件里，就只能
靠人工造场景。

框架的事件到本模块的映射在 ``src/chains/events.py``，那一层才是必须 import
框架的部分。

═══ 任务清单是**投影**，不是事实来源 ═══

这条决定了本模块的三条设计：

1. **丢了不影响业务**。任务清单只服务于「让用户看见系统在干什么」。
   它被清空、被截断、甚至整个丢失，都不会让任何一笔订单或申请出错。
   所以下面这些取舍都偏向「界面别崩」而不是「数据一条不能少」。
2. **快照是值拷贝**。:meth:`TaskCollector.snapshot` 返回的是不可变记录的
   列表，调用方改不动收集器的内部状态 —— 界面代码是最容易「顺手改一下
   数据」的地方。
3. **监听器抛异常不能反噬**。见 :meth:`TaskCollector._notify`。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from src.domain import TaskState

#: 本模块的日志器。
#:
#: ⚠️ 在**模块级**取一次，而不是在每个函数里 ``import logging`` 现取。
#: 本模块有多处告警（非法迁移、容量丢弃、监听器异常），现取写法会让每个
#: 函数体都多两行噪音，并且一旦有人漏掉一处，那条路径就彻底静默了 ——
#: 而这几条告警恰恰是排查事件流问题的唯一线索。
logger = logging.getLogger(__name__)

#: 任务清单的默认容量上限。
#:
#: ⚠️ 必须有上限。一轮模型调用里工具可能被反复调用（模型在失败后重试、
#: 或在长任务里连续检索十几轮），而清单会**一直增长到回复结束**。
#: 没有上限时，一个卡住的循环就能把内存吃满 —— 而这类循环恰恰是在
#: 「工具一直失败」时最容易出现的，也就是最需要系统保持可用的时刻。
DEFAULT_MAX_TASKS = 200

#: :class:`TaskCollector` 的时钟默认值：单调时钟。
#:
#: ⚠️ 用 ``monotonic`` 而不是 ``time.time``：要算的是**耗时**，
#: 而墙上时钟会被 NTP 校正、夏令时、手工改时间影响 —— 症状是界面上出现
#: 负数耗时或突然跳变几小时的条目。单调时钟不受这些影响，代价是它的
#: 绝对值没有意义（所以只在内部用来相减，绝不对外暴露）。
_DEFAULT_CLOCK: Callable[[], float] = time.monotonic


@dataclass(frozen=True)
class TaskRecord:
    """思考链上的**一个任务**（不可变）。

    ⚠️ 用 ``frozen=True`` 而不是普通 dataclass：这是要交给界面渲染的对象，
    而界面代码最容易「顺手改一下再显示」。冻结之后，任何修改企图都会立刻
    报错，而不是悄悄改变收集器内部的状态。

    Attributes:
        task_id: 任务标识，由收集器分配。
        name: 工具或能力的名字（如 ``query_flights``）——用于日志与排障。
        title: 面向用户的**中文**说明（如「正在查询杭州到北京的航班」）。
            ⚠️ 与 ``name`` 分开是刻意的：``name`` 是给工程师看的、要能
            grep 到代码与日志；``title`` 是给用户看的、要能读懂。合成一个
            字段，就必然要在某一侧将就。
        state: 当前状态。
        arguments: 调用参数（用于界面展开详情）。
        result: 成功时的结果摘要。
        error: 失败时的原因。
        started_at: 开始时刻（单调时钟的读数，**不可直接展示**）。
        finished_at: 结束时刻；未完成为 ``None``。
    """

    task_id: str
    name: str
    title: str = ""
    state: TaskState = TaskState.PENDING
    arguments: dict[str, object] = field(default_factory=dict)
    result: str = ""
    error: str = ""
    started_at: float | None = None
    finished_at: float | None = None

    @property
    def duration_seconds(self) -> float | None:
        """耗时（秒）；未开始或未结束时为 ``None``。

        ⚠️ 未**结束**也返回 ``None``，而不是用「当前时刻减开始时刻」——
        那会算出「到目前为止已耗时」，是个**每次读都在变大**的值。
        界面若照它渲染，就必须自己起定时器重绘；而一个只在某些行上跳动的
        列表比不显示耗时更让人困惑。让「还在跑」明确地表示成 ``None``，
        由界面决定怎么呈现（通常是转圈）。
        """
        if self.started_at is None or self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    @property
    def is_finished(self) -> bool:
        """是否已结束（成功或失败都算）。

        Returns:
            `bool`: 已结束返回 True。
        """
        return self.state in (TaskState.DONE, TaskState.FAILED)


#: 允许的状态迁移：``当前 -> 可迁往``。
#:
#: ⚠️ 与 ``src/domain/rules.py`` 的订单状态机同理，用**显式全量字典**而不是
#: 「一串 if」：新增状态若忘了登记，任何迁移都会被拒（快速失败），而不是
#: 被静默放行。
#:
#: ⚠️ 终态（DONE/FAILED）**没有出边**，包括没有「回到自己」的出边 ——
#: 这一点与订单状态机刻意不同。订单要幂等（网络重试会产生重复的「已处于
#: 目标态」请求），而任务清单是**投影**：重复的结束事件意味着上游事件流
#: 出了问题，此时保持原样并告警比接受它更有价值 —— 见
#: :meth:`TaskCollector.add_result`。
#:
#: ⚠️ ``PENDING -> DONE`` 是**允许**的，别把这条删掉。直觉上「没经过执行
#: 怎么会成功」，但投影层必须容忍两种情况：其一，参数校验失败的任务还没
#: 开始就结束了（``PENDING -> FAILED``，同理）；其二，上游只发了一个
#: 合并事件、没有中间的 DOING 标记。若拒绝这条边，这些任务会**永远停在
#: PENDING** —— 界面上就是一个永远转圈、永不消失的条目，而告警只会在
#: 服务端日志里，用户看不到。这正是「投影层不做裁判」这条原则的具体落点。
_TASK_TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
    TaskState.PENDING: frozenset({TaskState.DOING, TaskState.DONE, TaskState.FAILED}),
    TaskState.DOING: frozenset({TaskState.DONE, TaskState.FAILED}),
    TaskState.DONE: frozenset(),
    TaskState.FAILED: frozenset(),
}


class TaskCollector:
    """思考链的任务收集器：登记任务、记录结果、通知订阅者。

    ⚠️ **非线程安全，也不打算做成线程安全的**。它的生命周期是「一轮 reply
    之内」，而一轮 reply 由单个 asyncio 任务驱动，所有调用天然在同一事件
    循环线程上。加锁会给人一种「可以跨线程用」的错觉，从而掩盖真正的设计
    问题（比如有人想拿它当全局缓存）。

    ⚠️ 但监听器回调会被调用于收集器的**调用栈上**。监听器若做重活（同步写
    网络、渲染大图），会拖慢 agent 的事件循环。契约是「监听器必须快速返回」，
    写在 :meth:`subscribe` 的文档里。

    Attributes:
        _tasks: 任务表，键是 task_id，**保持插入顺序**（Python dict 有序）。
            顺序即界面上的显示顺序，而思考链必须按发生顺序读。
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = _DEFAULT_CLOCK,
        max_tasks: int = DEFAULT_MAX_TASKS,
    ) -> None:
        """初始化。

        Args:
            clock (`Callable[[], float]`): 时钟，默认单调时钟。做成参数是为了
                让测试可以不靠 ``sleep`` 就验证耗时计算 —— 用 ``sleep`` 的
                计时测试既慢又不稳定（在负载高的 CI 上会偶发失败）。
            max_tasks (`int`): 任务数上限，超出后丢弃**最旧**的。

        Raises:
            ValueError: ``max_tasks`` 小于 1。至少得留一个位置，否则收集器
                一登记就丢，行为与「没有收集器」无法区分。
        """
        if max_tasks < 1:
            raise ValueError("max_tasks 至少为 1")
        self._clock = clock
        self._max_tasks = max_tasks
        self._tasks: dict[str, TaskRecord] = {}
        self._listeners: list[Callable[[TaskRecord], None]] = []
        # 自增计数器而非 uuid：task_id 只在本轮 reply 内有意义，而顺序编号
        # 在日志里可读得多（"task-3" 一眼看得出是第三个）。
        self._seq = 0

    # --------------------------------------------------------------------------
    # 登记与推进
    # --------------------------------------------------------------------------
    def plan(self, name: str, *, title: str = "", arguments: dict[str, object] | None = None) -> str:
        """登记一个**尚未开始**的任务（``PENDING``）。

        ⚠️ 这个方法对应「模型已经决定要调某个工具，但还没真正执行」的那一刻
        （框架的 ``TOOL_CALL_START`` 事件）。把它与「开始执行」分开，界面才能
        显示「即将查询航班」→「正在查询航班」两段状态 —— 而这两段在高延迟
        工具上差别很明显，用户对「点了没反应」的忍耐度取决于此。

        Args:
            name (`str`): 工具/能力名。
            title (`str`): 面向用户的中文说明；留空时由调用方在展示层兜底。
            arguments (`dict[str, object] | None`): 调用参数。

        Returns:
            `str`: 新任务的 ``task_id``。
        """
        self._seq += 1
        task_id = f"task-{self._seq}"
        self._put(
            TaskRecord(
                task_id=task_id,
                name=name,
                title=title,
                state=TaskState.PENDING,
                arguments=dict(arguments or {}),
            ),
        )
        return task_id

    def add_use(
        self,
        name: str,
        *,
        title: str = "",
        arguments: dict[str, object] | None = None,
    ) -> str:
        """登记一个**立刻开始执行**的任务（``DOING``）。

        ⚠️ 与 :meth:`plan` 的区别只在于初始状态。有些调用点拿不到「决定要调」
        这个时刻（例如工具在自己的函数体里自报家门），此时 ``add_use`` 一步
        到位。两条入口都保留，是为了让调用方不必为了套用同一个 API 而伪造
        一个 ``PENDING`` 阶段 —— 那会让界面上出现一批「从未 PENDING 过」
        的任务，而 PENDING 段的渲染逻辑就成了死代码。

        Args:
            name (`str`): 工具/能力名。
            title (`str`): 面向用户的中文说明。
            arguments (`dict[str, object] | None`): 调用参数。

        Returns:
            `str`: 新任务的 ``task_id``。
        """
        self._seq += 1
        task_id = f"task-{self._seq}"
        self._put(
            TaskRecord(
                task_id=task_id,
                name=name,
                title=title,
                state=TaskState.DOING,
                arguments=dict(arguments or {}),
                started_at=self._clock(),
            ),
        )
        return task_id

    def retitle(
        self,
        task_id: str,
        *,
        title: str | None = None,
        arguments: dict[str, object] | None = None,
    ) -> None:
        """更新任务的**展示字段**，**不动状态**。

        ⚠️ 为什么需要它（不是「顺便加的一个 setter」）：同一轮回复里，
        同一件事可能被**分批**上报 —— 最典型的是人工确认：框架对每一个
        被挂起的工具调用**各发一条** ``RequireUserConfirmEvent``
        （``agent/_agent.py:2574-2577``，``tool_calls=[tool_call]``，
        长度恒为 1）。此时正确的处置是**更新**已登记的那一条，
        而不是：

        - 新登记一条 —— 界面上会出现好几个「等待你确认」，用户以为要逐个处理；
        - ``mark_doing`` 把它推进到 ``DOING`` —— 「等待你确认」那条会变成
          一个永远转不完的圈（``PENDING → DOING`` 是**合法**迁移，
          所以 ``_transition`` 不会告警，这个错误是**静默**的）。

        ⚠️ 传 ``None`` 表示「这个字段不动」，而**不是**「清空它」。
        把「没传」与「传空」混为一谈的后果是：调用方少写一个关键字参数时，
        一个已有的字段被静默抹掉 —— 而那种缺失在界面上只表现为一片空白。

        ⚠️ 对**不存在**的 ``task_id`` 静默忽略，理由同 :meth:`mark_doing`
        （任务可能因容量上限被丢弃）。

        Args:
            task_id (`str`): 任务标识。
            title (`str | None`): 新的面向用户中文说明；``None`` 表示不改。
            arguments (`dict[str, object] | None`): 新的调用参数；``None`` 表示不改。
        """
        task = self._tasks.get(task_id)
        if task is None:
            return

        changes: dict[str, object] = {}
        if title is not None:
            changes["title"] = title
        if arguments is not None:
            changes["arguments"] = dict(arguments)
        if not changes:
            return

        # ⚠️ 走 ``replace`` 是 ``TaskRecord`` 为 ``frozen=True`` 的**直接后果**
        # （见该类自己的说明）：原地改会抛 ``FrozenInstanceError``，不是「不推荐」。
        # 而冻结本身是为了订阅者 —— 快照对象同时发给界面与日志，
        # 能被追溯修改的话，**已经渲染过**的那条任务说明会凭空变掉。
        updated = replace(task, **changes)  # type: ignore[arg-type]
        self._tasks[task_id] = updated
        self._notify(updated)

    def mark_doing(self, task_id: str) -> None:
        """把任务从 ``PENDING`` 推进到 ``DOING``。

        ⚠️ 对**不存在**的 ``task_id`` 静默忽略，而不是抛 ``KeyError``。
        理由：任务可能因为容量上限被丢弃（见 :meth:`_put`），也可能因为
        事件流乱序而先收到结果、后收到开始。这两种都是**投影层可以容忍**的
        情况，不该让它们把一个正在进行的回复打断。非法**状态**迁移则相反，
        会告警 —— 那说明事件流本身有问题，值得留下痕迹。

        Args:
            task_id (`str`): 任务标识。
        """
        task = self._tasks.get(task_id)
        if task is None:
            return
        self._transition(task, TaskState.DOING, started_at=self._clock())

    def add_result(
        self,
        task_id: str,
        result: str = "",
        *,
        ok: bool = True,
        error: str = "",
    ) -> None:
        """记录任务结果，把它推进到 ``DONE`` 或 ``FAILED``。

        Args:
            task_id (`str`): 任务标识。
            result (`str`): 成功时的结果摘要。
            ok (`bool`): 是否成功。
            error (`str`): 失败原因；``ok=False`` 时必须有值。

        ⚠️ 对**已结束**的任务再次调用会被忽略并告警，而不是当作幂等接受。
        这一点与 ``src/domain/rules.py`` 的订单状态机刻意相反，理由见
        :data:`_TASK_TRANSITIONS` 的说明：任务清单是投影，重复的结束事件
        说明上游事件流有问题，留下告警比默默吞掉更有价值。

        ⚠️ ``ok=False`` 而 ``error`` 为空时**自动补一句兜底文案**。空的失败
        原因在界面上表现为「一个红叉加一片空白」，用户完全不知道发生了什么；
        而失败恰恰是最需要解释的时刻。宁可给一句笼统的「执行失败」，也不要
        什么都不给。
        """
        task = self._tasks.get(task_id)
        if task is None:
            return
        target = TaskState.DONE if ok else TaskState.FAILED
        self._transition(
            task,
            target,
            result=result if ok else "",
            error=(error or "执行失败") if not ok else "",
            finished_at=self._clock(),
        )

    # --------------------------------------------------------------------------
    # 读取与订阅
    # --------------------------------------------------------------------------
    def snapshot(self) -> list[TaskRecord]:
        """取当前任务清单的**值拷贝**。

        Returns:
            `list[TaskRecord]`: 按登记顺序排列的任务列表。

        ⚠️ 返回新列表（而 :class:`TaskRecord` 本身不可变），所以调用方拿到
        之后随便排序、过滤、拼接都不会影响收集器。界面代码是最容易「顺手
        改一下数据」的地方，用值拷贝把这条路堵死。
        """
        return list(self._tasks.values())

    def subscribe(self, listener: Callable[[TaskRecord], None]) -> Callable[[], None]:
        """订阅任务变化。

        Args:
            listener (`Callable[[TaskRecord], None]`): 每次任务有变化时被调用，
                参数是**变化后**的记录。

        Returns:
            `Callable[[], None]`: 退订函数。⚠️ 返回退订函数而不是要求调用方
            自己记住 listener 对象：忘记退订的后果是**内存泄漏**（收集器持有
            listener，而 listener 往往持有整个界面组件），并且会让一次已经结束
            的对话仍在接收回调。提供显式的退订入口，比文档里写一句「记得退订」
            有效得多。

        ⚠️ 监听器**必须快速返回**，且**不得抛异常**。抛出的异常会被
        :meth:`_notify` 捕获并告警，不会中断本轮回复 —— 见那里的说明。
        """
        self._listeners.append(listener)

        def _unsubscribe() -> None:
            # 用 discard 式判断而不是 try/except：重复退订是常见的调用方
            # 失误（比如在 finally 里退订、而正常路径也退订了），不该报错。
            if listener in self._listeners:
                self._listeners.remove(listener)

        return _unsubscribe

    def clear(self) -> None:
        """清空任务清单。

        ⚠️ 不会通知监听器。清空的语义是「这一轮的思考链到此为止，别再显示
        了」，通知反而会让界面先收到一堆「任务消失」的事件。需要界面刷新时，
        由调用方在清空之后自己触发一次重绘。
        """
        self._tasks.clear()

    # --------------------------------------------------------------------------
    # 内部
    # --------------------------------------------------------------------------
    def _put(self, record: TaskRecord) -> None:
        """写入一条新记录，并在超出容量时丢弃最旧的。

        ⚠️ 丢弃策略是**丢最旧的**（FIFO）而不是拒绝新的：思考链上最新的任务
        才是用户此刻最关心的，而最旧的通常已经滚出屏幕。反过来做（拒绝新的）
        会让界面在工具密集的一轮里**卡在最早那几条**，看起来像系统挂了。

        ⚠️ 丢弃会写日志。这是本项目的通用原则（见 ``docs/`` 的「不做静默截断」）：
        一个被无声截断的列表看起来像「这就是全部」，而实际不是。
        """
        self._tasks[record.task_id] = record
        while len(self._tasks) > self._max_tasks:
            # ⚠️ 必须先取出被丢弃的记录再删，日志里报的应该是**被丢的那条**。
            # 写成 `record.name`（刚写入的记录）会报错对象：日志说「丢弃了
            # 任务 X」而 X 其实还在表里，真正消失的是另一条。排查时按日志
            # 去比对表格，会发现对不上，进而怀疑整个收集器 —— 而它是对的。
            oldest_id = next(iter(self._tasks))
            dropped = self._tasks[oldest_id]
            logger.warning(
                "任务清单已达上限 %d，丢弃最旧的任务 %s（name=%s，state=%s）。"
                "思考链是投影，丢弃不影响业务数据；若频繁出现请检查是否存在工具重试循环。",
                self._max_tasks,
                oldest_id,
                dropped.name,
                dropped.state.value,
            )
            del self._tasks[oldest_id]
        self._notify(record)

    def _transition(self, task: TaskRecord, target: TaskState, **changes: object) -> None:
        """校验并执行一次状态迁移。

        Args:
            task (`TaskRecord`): 当前记录。
            target (`TaskState`): 目标状态。
            **changes: 需要一并更新的字段。

        ⚠️ 非法迁移**只告警不抛异常**。理由：本模块是投影层，它的职责是
        如实反映事件流，而不是对事件流的合法性做裁判。抛异常会把一个
        「界面状态有点怪」的问题升级成「整轮回复失败」—— 代价完全不成比例。
        但必须留下日志，否则事件流的顺序问题永远不会被发现。
        """
        allowed = _TASK_TRANSITIONS.get(task.state, frozenset())
        if target not in allowed:
            logger.warning(
                "任务 %s 的非法状态迁移：%s -> %s（已忽略）。"
                "任务清单是事件流的投影，出现这种情况通常意味着上游事件乱序或重复。",
                task.task_id,
                task.state.value,
                target.value,
            )
            return

        updated = replace(task, state=target, **changes)  # type: ignore[arg-type]
        self._tasks[task.task_id] = updated
        self._notify(updated)

    def _notify(self, record: TaskRecord) -> None:
        """通知所有监听器。

        ⚠️ 单个监听器抛异常**不会**影响其它监听器，也不会向上传播。理由：
        监听器是界面回调，而界面代码出错（拿不到某个字段、DOM 已卸载）是
        常态。让一个界面 bug 把 agent 的回复打断，是明显的责任错配 ——
        用户会看到「系统繁忙」，而真正的问题在前端。

        ⚠️ 捕获 ``Exception`` 而不是 ``BaseException``：``CancelledError``
        与 ``KeyboardInterrupt`` 必须继续向上传播，否则用户取消请求时
        会被这里静默吞掉，表现为「点了停止但没停下来」。
        """
        for listener in list(self._listeners):
            try:
                listener(record)
            except Exception:  # noqa: BLE001 —— 见上方说明，刻意宽捕
                logger.exception(
                    "任务清单监听器抛异常（已忽略，不影响本轮回复）：%r",
                    listener,
                )


__all__ = [
    "DEFAULT_MAX_TASKS",
    "TaskCollector",
    "TaskRecord",
]
