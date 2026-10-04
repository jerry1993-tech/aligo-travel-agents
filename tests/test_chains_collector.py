# -*- coding: utf-8 -*-
"""思考链任务收集器的单测。

组织方式按**真实的失败模式**来分，而不是按方法名从头到尾铺一遍：

1. 生命周期 —— 三种入口能否走完，以及各状态是否都有路可达；
2. 状态机的边 —— 谁能迁到谁。用**独立列出**的允许/拒绝清单，而不是从
   ``_TASK_TRANSITIONS`` 推导，否则测试只是在复述实现（同义反复）；
3. 投影层的容错 —— 重复结束、乱序、幽灵 task_id、监听器抛异常。
   这是本模块**唯一**真正复杂的地方，也是最容易在集成环境里被漏掉的；
4. 容量 —— 丢谁、有没有留痕；
5. 耗时 —— 用假时钟，不 ``sleep``；
6. 不可变与订阅 —— 值语义、退订、多监听器互不影响。

⚠️ 本文件**不 import agentscope**，这是 ``src/chains/__init__.py`` 的纯度
设计所保证的。若哪天这里需要 import 框架才能跑，说明有人把
``src.chains.events`` 加进了包的 ``__init__``，那是个需要修回去的回归。
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

import pytest

from src.chains import DEFAULT_MAX_TASKS, TaskCollector, TaskRecord
from src.chains.collector import _TASK_TRANSITIONS
from src.domain import TaskState

# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------


class FakeClock:
    """可手动推进的假时钟。

    ⚠️ 用假时钟而不是 ``time.sleep``：``sleep`` 式的计时断言既慢又不稳定
    （负载高的 CI 上会偶发失败），而且它测的是「机器有多快」而不是
    「耗时算得对不对」。这里推进多少、期望值就是多少，完全确定。
    """

    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        """把时钟往前推 ``seconds`` 秒。"""
        self.now += seconds


@pytest.fixture()
def clock() -> FakeClock:
    """默认的假时钟。"""
    return FakeClock()


@pytest.fixture()
def collector(clock: FakeClock) -> TaskCollector:
    """默认的收集器（挂了假时钟）。"""
    return TaskCollector(clock=clock)


# ---------------------------------------------------------------------------
# 1. 生命周期
# ---------------------------------------------------------------------------


def test_plan_registers_a_pending_task(collector: TaskCollector) -> None:
    """``plan`` 登记出的是 PENDING，且还没有开始计时。"""
    task_id = collector.plan("query_flights", title="准备查询航班", arguments={"from": "HGH"})

    (record,) = collector.snapshot()
    assert record.task_id == task_id
    assert record.state is TaskState.PENDING
    assert record.arguments == {"from": "HGH"}
    assert record.started_at is None
    assert record.finished_at is None


def test_add_use_registers_a_running_task(collector: TaskCollector) -> None:
    """``add_use`` 一步到位进入 DOING，并立刻开始计时。"""
    collector.add_use("query_hotels", title="查询酒店")

    (record,) = collector.snapshot()
    assert record.state is TaskState.DOING
    assert record.started_at is not None


def test_full_lifecycle_plan_doing_done(collector: TaskCollector, clock: FakeClock) -> None:
    """完整链路：PENDING → DOING → DONE，耗时来自两次时钟读数之差。"""
    task_id = collector.plan("query_flights", title="查询航班")
    clock.advance(0.5)
    collector.mark_doing(task_id)
    clock.advance(2.5)
    collector.add_result(task_id, "找到 3 个航班")

    (record,) = collector.snapshot()
    assert record.state is TaskState.DONE
    assert record.result == "找到 3 个航班"
    assert record.duration_seconds == pytest.approx(2.5)


def test_full_lifecycle_plan_doing_failed(collector: TaskCollector) -> None:
    """失败路径同样能走完，且 error 被记下。"""
    task_id = collector.plan("book_flight")
    collector.mark_doing(task_id)
    collector.add_result(task_id, ok=False, error="舱位已售罄")

    (record,) = collector.snapshot()
    assert record.state is TaskState.FAILED
    assert record.error == "舱位已售罄"
    assert record.result == ""


def test_task_ids_are_unique_and_sequential(collector: TaskCollector) -> None:
    """task_id 自增且不重复 —— 顺序编号让日志里一眼看得出先后。"""
    ids = [collector.add_use(f"tool_{i}") for i in range(5)]
    assert ids == ["task-1", "task-2", "task-3", "task-4", "task-5"]
    assert len(set(ids)) == 5


def test_snapshot_preserves_registration_order(collector: TaskCollector) -> None:
    """快照按登记顺序返回：思考链必须按发生顺序读。"""
    collector.add_use("first")
    collector.add_use("second")
    collector.add_use("third")

    assert [r.name for r in collector.snapshot()] == ["first", "second", "third"]


def test_titles_are_kept_separate_from_names(collector: TaskCollector) -> None:
    """``name``（给工程师）与 ``title``（给用户）是两个字段，不互相顶替。"""
    collector.add_use("query_flights", title="正在查询杭州到北京的航班")

    (record,) = collector.snapshot()
    assert record.name == "query_flights"
    assert record.title == "正在查询杭州到北京的航班"


def test_arguments_are_copied_not_aliased(collector: TaskCollector) -> None:
    """传进来的参数字典被拷贝一份，调用方后续修改不影响记录。"""
    args = {"from": "HGH"}
    collector.plan("query_flights", arguments=args)
    args["from"] = "PEK"  # 调用方改自己的字典

    (record,) = collector.snapshot()
    assert record.arguments == {"from": "HGH"}


def test_arguments_default_to_an_empty_dict_per_task(collector: TaskCollector) -> None:
    """未传参数时各自拿到独立的空字典，不共享同一个可变对象。"""
    collector.add_use("a")
    collector.add_use("b")

    first, second = collector.snapshot()
    assert first.arguments == {} and second.arguments == {}
    # 不是同一个对象 —— 若是同一个，一处改动会污染所有任务。
    assert first.arguments is not second.arguments


# ---------------------------------------------------------------------------
# 2. 状态机的边
# ---------------------------------------------------------------------------

#: 应当被接受的迁移。⚠️ **独立列出**，不从 ``_TASK_TRANSITIONS`` 推导 ——
#: 从实现推导的期望值永远与实现一致，测试就成了同义反复，改错了也发现不了。
_ALLOWED_TRANSITIONS = [
    (TaskState.PENDING, TaskState.DOING),
    (TaskState.PENDING, TaskState.DONE),
    (TaskState.PENDING, TaskState.FAILED),
    (TaskState.DOING, TaskState.DONE),
    (TaskState.DOING, TaskState.FAILED),
]

#: 应当被拒绝的迁移，以及为什么值得单独列一条。
#:
#: ⚠️ 只列**能通过公开 API 尝试**的那些。「怎么尝试迁到某个状态」见
#: :func:`_attempt_transition` —— 没有任何公开入口的目标态（比如回溯到
#: ``PENDING``）无法在这里表达，改由
#: :func:`test_the_transition_table_has_no_back_edge_to_pending` 在表层面守住。
_REJECTED_TRANSITIONS = [
    # 终态没有出边：DONE 之后不可能再开始或再失败。
    (TaskState.DONE, TaskState.DOING),
    (TaskState.DONE, TaskState.FAILED),
    (TaskState.DONE, TaskState.DONE),
    (TaskState.FAILED, TaskState.DOING),
    (TaskState.FAILED, TaskState.DONE),
    (TaskState.FAILED, TaskState.FAILED),
]


def _drive_to_state(collector: TaskCollector, state: TaskState) -> str:
    """用**公开 API** 造一个处于 ``state`` 的任务，返回其 task_id。

    ⚠️ 刻意不给收集器加测试专用的后门方法。给生产类加「只有测试用」的
    入口，等于把测试与实现绑在一起：后门能造出的状态，未必是真实调用路径
    能到达的状态，于是测试覆盖的是一个不存在的世界。这里全部经由
    ``plan`` / ``add_use`` / ``mark_doing`` / ``add_result`` 走真实路径。

    Args:
        collector (`TaskCollector`): 目标收集器。
        state (`TaskState`): 想要到达的状态。

    Returns:
        `str`: 该任务的 task_id。
    """
    if state is TaskState.PENDING:
        return collector.plan("driven")
    if state is TaskState.DOING:
        return collector.add_use("driven")

    task_id = collector.add_use("driven")
    if state is TaskState.DONE:
        collector.add_result(task_id, "ok")
    else:  # TaskState.FAILED
        collector.add_result(task_id, ok=False, error="boom")
    return task_id


def _attempt_transition(collector: TaskCollector, task_id: str, target: TaskState) -> None:
    """用公开 API 尝试把任务迁到 ``target``（不管成不成功）。

    ⚠️ ``TaskState.PENDING`` 没有对应的公开入口 —— 状态机里根本不存在
    「迁回 PENDING」这条路，所以这里显式报错而不是静默跳过：真有人往
    :data:`_REJECTED_TRANSITIONS` 里加了 PENDING 目标，应当立刻发现。
    """
    if target is TaskState.DOING:
        collector.mark_doing(task_id)
    elif target is TaskState.DONE:
        collector.add_result(task_id, "试图成功")
    elif target is TaskState.FAILED:
        collector.add_result(task_id, ok=False, error="试图失败")
    else:
        raise AssertionError(f"没有公开入口可以尝试迁移到 {target}；请改用表层面的断言")


def test_every_task_state_is_covered_by_the_transition_table() -> None:
    """迁移表必须覆盖 :class:`TaskState` 的**每一个**成员。

    ⚠️ 这条防的是「新增状态忘了登记」：``_transition`` 用
    ``_TASK_TRANSITIONS.get(state, frozenset())`` 取值，漏登记的状态会
    静默退化成「哪也去不了」，而不是报错。有了这条断言，加状态的人会在
    测试里立刻看到失败。
    """
    assert set(_TASK_TRANSITIONS) == set(TaskState)


@pytest.mark.parametrize(("source", "target"), _ALLOWED_TRANSITIONS)
def test_allowed_transitions_are_accepted(
    collector: TaskCollector,
    source: TaskState,
    target: TaskState,
) -> None:
    """允许的迁移确实被接受（状态真的变了）。"""
    # 先确认这份手写清单与实现一致，否则下面的断言测的是另一回事。
    assert target in _TASK_TRANSITIONS[source]

    task_id = _drive_to_state(collector, source)
    _attempt_transition(collector, task_id, target)

    (record,) = collector.snapshot()
    assert record.state is target


@pytest.mark.parametrize(("source", "target"), _REJECTED_TRANSITIONS)
def test_rejected_transitions_are_ignored_and_warned(
    collector: TaskCollector,
    caplog: pytest.LogCaptureFixture,
    source: TaskState,
    target: TaskState,
) -> None:
    """被拒绝的迁移：状态**原样不动**，但留下告警。"""
    assert target not in _TASK_TRANSITIONS[source]

    task_id = _drive_to_state(collector, source)
    with caplog.at_level(logging.WARNING):
        _attempt_transition(collector, task_id, target)

    (record,) = collector.snapshot()
    # ⚠️ 重点是「状态没变」。投影层不做裁判，但也绝不放行非法迁移 ——
    # 放行的后果是界面上出现一个「从失败恢复到成功」的幽灵条目。
    assert record.state is source
    assert "非法状态迁移" in caplog.text


def test_the_transition_table_has_no_back_edge_to_pending() -> None:
    """没有任何状态能迁回 ``PENDING``：状态机里不存在这条回头路。

    ⚠️ 这条只能在**表层面**守 —— 公开 API 里根本没有「迁回 PENDING」的入口，
    所以写不出对应的行为测试。少了它，「有人给某个状态加了回 PENDING 的边」
    会完全无人察觉。
    """
    for source, targets in _TASK_TRANSITIONS.items():
        assert TaskState.PENDING not in targets, f"{source} 不该能迁回 PENDING"


def test_double_finish_is_rejected_not_treated_as_idempotent(
    collector: TaskCollector,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """对已结束的任务再次 ``add_result``：忽略 + 告警，**不**当作幂等接受。

    ⚠️ 这与 ``src/domain/rules.py`` 的订单状态机刻意相反（那边接受重复的
    「已处于目标态」）。订单是事实、要容忍网络重试；这里是投影、重复的
    结束事件意味着上游事件流有问题，值得留痕。
    """
    task_id = collector.add_use("query_flights")
    collector.add_result(task_id, "第一次结果")

    with caplog.at_level(logging.WARNING):
        collector.add_result(task_id, "第二次结果")

    (record,) = collector.snapshot()
    assert record.result == "第一次结果"  # 第一次的结果没被覆盖
    assert "非法状态迁移" in caplog.text


def test_mark_doing_on_a_finished_task_is_rejected(
    collector: TaskCollector,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """已完成的任务不能被重新标记为执行中。"""
    task_id = collector.add_use("query_flights")
    collector.add_result(task_id, "ok")

    with caplog.at_level(logging.WARNING):
        collector.mark_doing(task_id)

    (record,) = collector.snapshot()
    assert record.state is TaskState.DONE
    assert "非法状态迁移" in caplog.text


# ---------------------------------------------------------------------------
# 2·B、retitle：只改展示字段，不动状态
# ---------------------------------------------------------------------------
def test_retitle_changes_the_title_without_touching_the_state(
    collector: TaskCollector,
) -> None:
    """★★★ ``retitle`` 改标题，**状态一点不动**。

    ⚠️ 这个方法存在的唯一理由是「同一个任务被分批上报时要更新展示，
    但不能推进状态」。所以「状态不动」不是它的一个性质，是它的**全部意义**。

    ⚠️ 反例就是它替代掉的那个写法：调用方本来想要「更新一下」，
    手边只有 ``mark_doing``，于是顺手调了它 —— 状态从 ``PENDING`` 变成
    ``DOING``。而这条迁移是**合法**的，收集器不会告警，日志里一个字都没有。
    症状是界面上一个永远转不完的圈（「等待你确认」那条尤其明显）。

    ⚠️ 断言方式：不光查 ``state``，还要查 ``started_at`` **没被设置** ——
    ``mark_doing`` 会同时写 ``started_at``，只查状态的话，一个
    「状态没改但把 started_at 写了」的实现能混过去，而那个任务的耗时会
    从「等待开始」算起，在界面上显示成一个已经跑了一阵的任务。
    """
    task_id = collector.plan("等待你确认 1 项操作", title="等待你确认 1 项操作")
    before = collector.snapshot()[0]

    collector.retitle(task_id, title="等待你确认 3 项操作", arguments={"pending": 3})

    (after,) = collector.snapshot()
    assert after.title == "等待你确认 3 项操作"
    assert after.arguments == {"pending": 3}
    assert after.state is TaskState.PENDING, "retitle 动了状态"
    assert after.started_at == before.started_at, "retitle 写了 started_at"


def test_retitle_only_changes_what_it_is_given(collector: TaskCollector) -> None:
    """★★ 没传的字段**保持不变**，而不是被清空。

    ⚠️ 把「没传」与「传空」混为一谈的后果：调用方只想改标题，
    结果 ``arguments`` 被抹成空字典 —— 而那种缺失在界面上只表现为
    一片空白，没人会怀疑到这一层。
    """
    task_id = collector.plan("查询航班", title="正在查询航班", arguments={"city": "北京"})

    collector.retitle(task_id, title="正在查询航班（重试）")

    (record,) = collector.snapshot()
    assert record.title == "正在查询航班（重试）"
    assert record.arguments == {"city": "北京"}, "没传 arguments，它却被清空了"
    assert record.name == "查询航班", "retitle 动了名字"


def test_retitle_with_no_fields_does_nothing_and_does_not_notify(
    collector: TaskCollector,
) -> None:
    """⚠️ 一个字段都没给时**不通知订阅者**。

    ⚠️ 通知一次「什么都没变」不只是浪费：订阅者收到通知会推一帧给前端，
    而前端的更新是有成本的（重渲染、滚动位置可能被重置）。
    在一条工具密集的回复里，这类空通知会攒成肉眼可见的抖动。
    """
    task_id = collector.plan("查询航班", title="正在查询航班")
    seen: list[str] = []
    collector.subscribe(lambda record: seen.append(record.task_id))

    collector.retitle(task_id)

    assert seen == []


def test_retitle_on_unknown_task_is_silently_ignored(collector: TaskCollector) -> None:
    """⚠️ 幽灵 ``task_id`` 静默忽略，理由同 ``mark_doing``。

    ⚠️ 任务可能因为容量上限被丢弃，也可能因为事件流乱序而晚到。
    抛 ``KeyError`` 会把一个「界面缺一条」的问题升级成「整轮回复失败」。
    """
    collector.retitle("task-不存在", title="随便什么")

    assert collector.snapshot() == []


def test_retitle_notifies_subscribers_with_the_updated_record(
    collector: TaskCollector,
) -> None:
    """★ 改完之后要通知，且通知里带的是**新**值。

    ⚠️ 通知了但带旧值的实现比不通知更糟：订阅者会拿旧标题覆盖掉自己
    已经渲染好的新标题，表现为「标题闪一下又变回去了」。
    """
    task_id = collector.plan("等待你确认 1 项操作", title="等待你确认 1 项操作")
    seen: list[tuple[str, str]] = []
    collector.subscribe(lambda record: seen.append((record.task_id, record.title)))

    collector.retitle(task_id, title="等待你确认 5 项操作")

    assert seen == [(task_id, "等待你确认 5 项操作")]


# ⚠️ 这里**没有**一条「已发出去的快照不会被追溯修改」的用例，是刻意的：
# ``TaskRecord`` 是 ``frozen=True`` 的（``src/chains/collector.py:73``），
# 原地改会直接抛 ``FrozenInstanceError``，所以那条断言**永远为真** ——
# 一条不可能变红的用例只会让人以为这块被覆盖到了。


# ---------------------------------------------------------------------------
# 3. 投影层的容错 —— 本模块唯一真正复杂的地方
# ---------------------------------------------------------------------------


def test_result_for_unknown_task_is_silently_ignored(collector: TaskCollector) -> None:
    """幽灵 task_id 被静默忽略，不抛 KeyError。

    ⚠️ 任务可能因为容量上限被丢弃，也可能因为事件流乱序而先收到结果、
    后收到开始。这两种都是投影层可以容忍的，不该打断一个正在进行的回复。
    """
    collector.add_result("task-999", "结果")
    collector.mark_doing("task-999")
    assert collector.snapshot() == []


def test_out_of_order_events_do_not_corrupt_the_list(collector: TaskCollector) -> None:
    """乱序到达的事件不会让清单崩掉，也不会产生幽灵条目。"""
    # 先来结果、后来登记：结果落空（task-1 还不存在），随后的登记照常。
    collector.add_result("task-1", "早到的结果")
    collector.add_use("query_flights")

    snapshot = collector.snapshot()
    assert len(snapshot) == 1
    assert snapshot[0].state is TaskState.DOING


def test_pending_task_can_finish_without_passing_through_doing(
    collector: TaskCollector,
) -> None:
    """``PENDING`` 直接结束是合法的 —— 投影层不做裁判。

    ⚠️ 若拒绝这条边，这类任务会**永远停在 PENDING**，界面上就是一个永远
    转圈、永不消失的条目，而告警只在服务端日志里，用户看不到。
    """
    task_id = collector.plan("validate_params", title="校验参数")
    collector.add_result(task_id, ok=False, error="出发日期不能早于今天")

    (record,) = collector.snapshot()
    assert record.state is TaskState.FAILED


def test_empty_failure_reason_gets_a_fallback(collector: TaskCollector) -> None:
    """失败但没给原因时补一句兜底文案。

    ⚠️ 空的失败原因在界面上表现为「一个红叉加一片空白」，而失败恰恰是最
    需要解释的时刻。
    """
    task_id = collector.add_use("query_flights")
    collector.add_result(task_id, ok=False)  # 刻意不给 error

    (record,) = collector.snapshot()
    assert record.state is TaskState.FAILED
    assert record.error == "执行失败"


def test_a_broken_listener_does_not_break_the_reply(collector: TaskCollector) -> None:
    """监听器抛异常被吞掉，不影响本轮回复，也不影响其它监听器。"""
    seen: list[str] = []

    def angry(_record: TaskRecord) -> None:
        raise RuntimeError("界面组件已卸载")

    collector.subscribe(angry)
    collector.subscribe(lambda r: seen.append(r.task_id))

    collector.add_use("query_flights")  # 不应抛异常

    (record,) = collector.snapshot()
    assert record.state is TaskState.DOING
    # ⚠️ 关键：坏监听器排在前面，好监听器**仍然**收到了通知。
    # 若用「遇到异常就 return」，后面的监听器会被静默跳过。
    assert seen == [record.task_id]


def test_listener_errors_are_logged(
    collector: TaskCollector,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """监听器的异常必须留痕，否则前端的 bug 永远不会被发现。"""

    def angry(_record: TaskRecord) -> None:
        raise RuntimeError("界面组件已卸载")

    collector.subscribe(angry)
    with caplog.at_level(logging.ERROR):
        collector.add_use("query_flights")

    assert "监听器抛异常" in caplog.text


def test_cancellation_propagates_out_of_the_listener(
    collector: TaskCollector,
) -> None:
    """``CancelledError`` **不**被吞掉，必须继续向上传播。

    ⚠️ 若宽捕到 ``BaseException``，用户点「停止」时会被这里静默吞掉，
    表现为「点了停止但没停下来」。
    """

    def cancelling(_record: TaskRecord) -> None:
        raise asyncio.CancelledError

    collector.subscribe(cancelling)
    with pytest.raises(asyncio.CancelledError):
        collector.add_use("query_flights")


# ---------------------------------------------------------------------------
# 4. 容量
# ---------------------------------------------------------------------------


def test_default_capacity_is_positive() -> None:
    """默认容量得是个正数 —— 上限为 0 会让收集器一条都留不下。"""
    assert DEFAULT_MAX_TASKS >= 1


def test_zero_capacity_is_rejected() -> None:
    """``max_tasks < 1`` 在构造时就报错，而不是运行到一半才暴露。"""
    with pytest.raises(ValueError, match="max_tasks"):
        TaskCollector(max_tasks=0)


def test_oldest_tasks_are_evicted_first(clock: FakeClock) -> None:
    """超容量时丢**最旧**的，保留最新的。"""
    collector = TaskCollector(clock=clock, max_tasks=3)
    for i in range(5):
        collector.add_use(f"tool_{i}")

    assert [r.name for r in collector.snapshot()] == ["tool_2", "tool_3", "tool_4"]


def test_eviction_is_logged_with_the_dropped_task(
    clock: FakeClock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """丢弃必须留痕，且日志里报的是**被丢的那条**。

    ⚠️ 这条专门盯一个易犯的错：日志写成刚写入的记录名，于是日志说
    「丢弃了 tool_4」而 tool_4 其实还在表里。排查时按日志比对表格会发现
    对不上，进而怀疑收集器 —— 而它是对的。
    """
    collector = TaskCollector(clock=clock, max_tasks=2)
    collector.add_use("tool_0")
    collector.add_use("tool_1")

    with caplog.at_level(logging.WARNING):
        collector.add_use("tool_2")

    assert "丢弃最旧的任务 task-1" in caplog.text
    assert "name=tool_0" in caplog.text
    # 被丢的那条确实不在了，新来的确实在。
    assert [r.name for r in collector.snapshot()] == ["tool_1", "tool_2"]


def test_capacity_one_still_works(clock: FakeClock) -> None:
    """容量为 1 的边界：每一步都只剩最新一条。"""
    collector = TaskCollector(clock=clock, max_tasks=1)
    collector.add_use("a")
    collector.add_use("b")

    (record,) = collector.snapshot()
    assert record.name == "b"


def test_the_latest_task_is_never_dropped(collector: TaskCollector) -> None:
    """刚登记的任务永远在清单里 —— 用户此刻最关心的是它。"""
    for i in range(DEFAULT_MAX_TASKS + 10):
        collector.add_use(f"tool_{i}")

    assert collector.snapshot()[-1].name == f"tool_{DEFAULT_MAX_TASKS + 9}"
    assert len(collector.snapshot()) == DEFAULT_MAX_TASKS


# ---------------------------------------------------------------------------
# 5. 耗时
# ---------------------------------------------------------------------------


def test_duration_is_none_while_still_running(collector: TaskCollector, clock: FakeClock) -> None:
    """执行中的任务耗时为 ``None``，而不是「到目前为止已耗时」。

    ⚠️ 后者是个**每次读都在变大**的值，界面若要显示就得自己起定时器重绘。
    让「还在跑」明确地表示成 ``None``，由界面决定怎么呈现。
    """
    collector.add_use("query_flights")
    clock.advance(10.0)

    (record,) = collector.snapshot()
    assert record.duration_seconds is None


def test_duration_is_none_before_start(collector: TaskCollector) -> None:
    """还没开始的 PENDING 任务同样没有耗时。"""
    collector.plan("query_flights")
    (record,) = collector.snapshot()
    assert record.duration_seconds is None


def test_duration_is_frozen_after_finishing(collector: TaskCollector, clock: FakeClock) -> None:
    """结束之后再推进时钟，耗时**不再变化**。"""
    task_id = collector.add_use("query_flights")
    clock.advance(3.0)
    collector.add_result(task_id, "ok")

    (record,) = collector.snapshot()
    first = record.duration_seconds

    clock.advance(1000.0)  # 时间继续流逝
    (record_again,) = collector.snapshot()
    assert record_again.duration_seconds == first == pytest.approx(3.0)


def test_plan_does_not_count_the_waiting_time(collector: TaskCollector, clock: FakeClock) -> None:
    """PENDING 段的等待时间**不计入**耗时 —— 计时从 ``mark_doing`` 开始。

    ⚠️ 这是刻意的：PENDING 段可能包含用户在权限确认弹窗前的思考时间，
    把它算进「工具耗时」会让监控面板上的耗时指标彻底失真。
    """
    task_id = collector.plan("book_flight")
    clock.advance(60.0)  # 用户在确认弹窗前犹豫了一分钟
    collector.mark_doing(task_id)
    clock.advance(2.0)  # 真正的执行
    collector.add_result(task_id, "ok")

    (record,) = collector.snapshot()
    assert record.duration_seconds == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 6. 不可变与订阅
# ---------------------------------------------------------------------------


def test_task_record_is_immutable() -> None:
    """``TaskRecord`` 是冻结的：界面代码改不动它。"""
    record = TaskRecord(task_id="task-1", name="query_flights")
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.state = TaskState.DONE  # type: ignore[misc]


def test_snapshot_mutation_does_not_affect_the_collector(collector: TaskCollector) -> None:
    """改快照列表（排序、清空）不影响收集器内部状态。"""
    collector.add_use("a")
    collector.add_use("b")

    snap = collector.snapshot()
    snap.clear()
    snap.append(TaskRecord(task_id="fake", name="injected"))

    assert [r.name for r in collector.snapshot()] == ["a", "b"]


def test_snapshot_returns_a_fresh_list_each_time(collector: TaskCollector) -> None:
    """两次 ``snapshot`` 返回不同的列表对象。"""
    collector.add_use("a")
    assert collector.snapshot() is not collector.snapshot()


def test_subscribers_receive_every_transition(collector: TaskCollector) -> None:
    """订阅者收到**每一次**变化：登记、开始、结束。"""
    seen: list[tuple[str, TaskState]] = []
    collector.subscribe(lambda r: seen.append((r.task_id, r.state)))

    task_id = collector.plan("query_flights")
    collector.mark_doing(task_id)
    collector.add_result(task_id, "ok")

    assert seen == [
        (task_id, TaskState.PENDING),
        (task_id, TaskState.DOING),
        (task_id, TaskState.DONE),
    ]


def test_unsubscribe_stops_delivery(collector: TaskCollector) -> None:
    """退订之后不再收到通知。"""
    seen: list[str] = []
    unsubscribe = collector.subscribe(lambda r: seen.append(r.task_id))

    collector.add_use("a")
    unsubscribe()
    collector.add_use("b")

    assert len(seen) == 1


def test_unsubscribing_twice_is_harmless(collector: TaskCollector) -> None:
    """重复退订不报错 —— 调用方常在 finally 与正常路径各退订一次。"""
    unsubscribe = collector.subscribe(lambda r: None)
    unsubscribe()
    unsubscribe()  # 不应抛异常


def test_unsubscribe_does_not_remove_other_listeners(collector: TaskCollector) -> None:
    """退订一个监听器不影响另一个。"""
    seen_a: list[str] = []
    seen_b: list[str] = []
    unsubscribe_a = collector.subscribe(lambda r: seen_a.append(r.task_id))
    collector.subscribe(lambda r: seen_b.append(r.task_id))

    collector.add_use("a")
    unsubscribe_a()
    collector.add_use("b")

    assert len(seen_a) == 1
    assert len(seen_b) == 2


def test_a_listener_subscribing_during_notification_does_not_crash(
    collector: TaskCollector,
) -> None:
    """监听器在回调里**新增**订阅者时，不应因「遍历中被修改」而崩。

    ⚠️ ``_notify`` 用 ``list(self._listeners)`` 做快照遍历正是为了这个。
    直接在原列表上迭代会在这种场景下抛 ``RuntimeError``，而触发条件
    （某个界面组件在收到事件后挂上自己的子监听器）完全不罕见。
    """
    collector.subscribe(lambda r: collector.subscribe(lambda _r: None))
    collector.add_use("a")  # 不应抛 RuntimeError

    assert len(collector.snapshot()) == 1


def test_a_listener_unsubscribing_during_notification_does_not_crash(
    collector: TaskCollector,
) -> None:
    """监听器在回调里**退订自己**时同理。"""
    holder: dict[str, object] = {}

    def self_removing(_record: TaskRecord) -> None:
        holder["unsubscribe"]()  # type: ignore[operator]

    holder["unsubscribe"] = collector.subscribe(self_removing)
    collector.add_use("a")
    collector.add_use("b")  # 不应抛 RuntimeError


def test_clear_empties_the_list_without_notifying(collector: TaskCollector) -> None:
    """``clear`` 清空清单，但**不**通知订阅者。

    ⚠️ 通知会让界面先收到一堆「任务消失」的事件；清空的语义是「这一轮
    到此为止」，由调用方自己触发重绘。
    """
    collector.add_use("a")
    seen: list[str] = []
    collector.subscribe(lambda r: seen.append(r.task_id))

    collector.clear()

    assert collector.snapshot() == []
    assert seen == []


def test_clear_allows_reuse(collector: TaskCollector) -> None:
    """清空后可以继续登记，且 task_id **继续自增**不重用。

    ⚠️ 不重置计数器：若重用 ``task-1``，界面上可能把上一轮的旧记录
    与新记录当成同一条更新掉。
    """
    first = collector.add_use("a")
    collector.clear()
    second = collector.add_use("b")

    assert first == "task-1"
    assert second == "task-2"


def test_is_finished_covers_both_terminal_states(collector: TaskCollector) -> None:
    """``is_finished`` 对 DONE 与 FAILED 都为真，对 PENDING/DOING 为假。"""
    done = collector.add_use("a")
    failed = collector.add_use("b")
    collector.add_result(done, "ok")
    collector.add_result(failed, ok=False, error="boom")
    collector.add_use("c")  # DOING
    collector.plan("d")  # PENDING

    states = {r.name: r.is_finished for r in collector.snapshot()}
    assert states == {"a": True, "b": True, "c": False, "d": False}


def test_collector_does_not_import_the_framework() -> None:
    """纯度探针：import 本收集器**不得**把 ``agentscope`` 拉进来。

    ⚠️ 必须在**子进程**里验证。在进程内断言 ``"agentscope" not in sys.modules``
    是无效的 —— ``conftest.py`` 早已把框架导入了，断言必然假通过。

    这条测试守的是 ``src/chains/__init__.py`` 的设计：一旦有人把
    ``src.chains.events`` 加进包的 ``__init__``，任务清单的纯逻辑测试就会
    被迫依赖框架（装不上框架就跑不了），这里会立刻失败。
    """
    import subprocess
    import sys

    probes = [
        "import sys; import src.chains; "
        "assert 'agentscope' not in sys.modules, 'src.chains 把框架拉进来了'; "
        "print('PURE')",
        "import sys; from src.chains import TaskCollector; "
        "assert 'agentscope' not in sys.modules, 'TaskCollector 把框架拉进来了'; "
        "print('PURE')",
    ]
    for code in probes:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "PURE", result.stdout
