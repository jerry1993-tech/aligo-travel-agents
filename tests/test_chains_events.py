# -*- coding: utf-8 -*-
"""事件适配器（``src/chains/events.py``）的测试。

═══ 这张表守的是什么 ═══

适配器是框架事件流与思考链之间**唯一**的一层。它很薄（就是一堆字段搬运），
但薄不等于安全 —— 框架的事件流有三处反直觉的事实，而每一处的错误表现
都是「功能静默失效」，不是报错：

1. 工具参数**分片到达**，``TOOL_CALL_END`` 里没有参数；
2. 工具结果是**流式**的，``TOOL_RESULT_END`` 里没有内容；
3. 人工确认之后框架会补发一个 **``reply_id`` 相同**的 ``REPLY_START``，
   那是「续接」不是「新回复」。

三条都在下面有用例。第 3 条尤其值得看 —— 它的症状是「用户点了同意，
进度条继续转圈」，而没有任何一处报错。

═══ ⚠️ 两种事件形态都要覆盖 ═══

``reply_stream`` 给的是 pydantic 对象，而总线/SSE 路径给的是
``model_dump`` 出来的字典。两者的差别是**致命**的（``getattr(字典, ...)``
取不到任何字段），所以下面每个用例都跑两遍。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from agentscope.event import (
    ConfirmResult,
    ExceedMaxItersEvent,
    ReplyEndEvent,
    ReplyStartEvent,
    RequireUserConfirmEvent,
    ThinkingBlockDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
    UserConfirmResultEvent,
)
from agentscope.message import ToolCallBlock, ToolResultState
from agentscope.types import ReplyFinishedReason

from src.chains.collector import TaskCollector
from src.chains.events import (
    CONFIRM_TASK_NAME,
    TITLES,
    EventChainAdapter,
    iter_task_states,
    title_for,
)
from src.domain import TaskState

REPLY_ID = "reply-1"


# ---------------------------------------------------------------------------
# 夹具与工具
# ---------------------------------------------------------------------------
def as_dict(event: Any) -> dict[str, Any]:
    """把事件转成服务层在总线上发布的形态。

    ⚠️ 与 ``agentscope/app/_service/_chat.py:1355-1359`` 的做法**一致**
    （``model_dump(mode="json")``），不是随手 ``model_dump()`` ——
    前者会把 ``ToolResultState`` 变成字符串、时间戳变成 ISO 串，
    那正是适配器要面对的形态。
    """
    return event.model_dump(mode="json")


@pytest.fixture(params=["object", "dict"])
def driver(request: pytest.FixtureRequest):
    """按两种事件形态各跑一遍。

    ⚠️ 用 ``params`` 而不是写两个用例：两种形态的**语义要求完全相同**，
    分开写会让人以为它们该有不同行为，而且新增用例时很容易只加一半。

    Returns:
        `Callable[[Any], Any]`: 把事件转成该形态。
    """
    if request.param == "dict":
        return as_dict

    def identity(event: Any) -> Any:
        """原样返回（对象形态）。"""
        return event

    return identity


@pytest.fixture
def collector() -> TaskCollector:
    """一个任务收集器。

    Returns:
        `TaskCollector`: 收集器。
    """
    return TaskCollector()


@pytest.fixture
def adapter(collector: TaskCollector) -> EventChainAdapter:
    """一个接到 ``collector`` 上的适配器。

    Returns:
        `EventChainAdapter`: 适配器。
    """
    return EventChainAdapter(collector)


def begin(adapter: EventChainAdapter, convert, reply_id: str = REPLY_ID) -> None:
    """喂一个 ``REPLY_START``。

    Args:
        adapter (`EventChainAdapter`): 适配器。
        convert (`Callable`): 形态转换。
        reply_id (`str`): 回复标识。
    """
    adapter.consume(convert(ReplyStartEvent(session_id="s", reply_id=reply_id, name="main")))


def announce_call(
    adapter: EventChainAdapter,
    convert,
    call_id: str,
    name: str,
    arguments: dict[str, object] | None = None,
    *,
    fragments: list[str] | None = None,
) -> None:
    """喂一整套 ``TOOL_CALL_START / DELTA* / END``。

    Args:
        adapter (`EventChainAdapter`): 适配器。
        convert (`Callable`): 形态转换。
        call_id (`str`): 工具调用标识。
        name (`str`): 工具名。
        arguments (`dict | None`): 参数；会被切成碎片发出去。
        fragments (`list[str] | None`): 直接指定分片（用来造畸形 JSON）。
    """
    adapter.consume(
        convert(ToolCallStartEvent(reply_id=REPLY_ID, tool_call_id=call_id, tool_call_name=name)),
    )
    pieces = fragments if fragments is not None else _split(json.dumps(arguments or {}))
    for piece in pieces:
        adapter.consume(convert(ToolCallDeltaEvent(reply_id=REPLY_ID, tool_call_id=call_id, delta=piece)))
    adapter.consume(convert(ToolCallEndEvent(reply_id=REPLY_ID, tool_call_id=call_id)))


def _split(raw: str, size: int = 3) -> list[str]:
    """把字符串切成碎片，模拟模型的流式输出。

    ⚠️ 切成 3 个字符一片而不是整串发一次：**「整串一次到达」是流式的
    特例**，用它测不出「分片累积」这条逻辑。切碎才有意义。

    Args:
        raw (`str`): 原始字符串。
        size (`int`): 每片长度。

    Returns:
        `list[str]`: 分片列表。
    """
    return [raw[i : i + size] for i in range(0, len(raw), size)] or [""]


def result_of(adapter: EventChainAdapter, convert, call_id: str, text: str, state) -> None:
    """喂一整套 ``TOOL_RESULT_START / DELTA* / END``。

    Args:
        adapter (`EventChainAdapter`): 适配器。
        convert (`Callable`): 形态转换。
        call_id (`str`): 工具调用标识。
        text (`str`): 结果文本。
        state: 结束状态。
    """
    adapter.consume(
        convert(ToolResultStartEvent(reply_id=REPLY_ID, tool_call_id=call_id, tool_call_name="t")),
    )
    for piece in _split(text):
        adapter.consume(convert(ToolResultTextDeltaEvent(reply_id=REPLY_ID, tool_call_id=call_id, delta=piece)))
    adapter.consume(convert(ToolResultEndEvent(reply_id=REPLY_ID, tool_call_id=call_id, state=state)))


def by_name(collector: TaskCollector, name: str):
    """按名字取任务。

    Args:
        collector (`TaskCollector`): 收集器。
        name (`str`): 任务名。

    Returns:
        匹配的任务记录；没有则为 ``None``。
    """
    for task in collector.snapshot():
        if task.name == name:
            return task
    return None


# ---------------------------------------------------------------------------
# 一、标题
# ---------------------------------------------------------------------------
def test_title_for_returns_a_chinese_title_for_every_real_tool() -> None:
    """★ 每个真实工具名都有中文标题，且不是兜底文案。

    ⚠️ ``title`` 是用户在界面上**唯一**看到的东西（``name`` 是给工程师
    grep 日志用的）。落到兜底文案上意味着用户看到「正在调用
    search_transport」—— 与直接显示函数名没有区别。
    """
    for tool_name, title in TITLES.items():
        assert title_for(tool_name) == title
        assert not title.startswith("正在调用"), f"{tool_name} 的标题像是兜底文案"
        assert title != tool_name


def test_title_for_never_returns_empty() -> None:
    """★★ 认不出的工具名也要给一句人话，**绝不返回空串**。

    ⚠️ 空标题在界面上表现为「一行空白在进行中」。而它恰恰发生在
    「新加了工具、忘了登记标题」的时候 —— 也就是最需要界面还能
    说清楚话的时候。宁可给一句带原始名字的兜底，也不要空白。
    """
    assert title_for("brand_new_tool") == "正在调用 brand_new_tool"
    assert title_for("") == "正在处理"


def test_every_registered_tool_has_a_title() -> None:
    """★★ **真实**工具集里的每一个工具名，都能在 ``TITLES`` 里找到。

    ⚠️ 这条是 ``TITLES`` 存在的代价的兑付：那张表把工具名手抄了一遍
    （工具是闭包，模块级拿不到符号，见 ``src/chains/events.py`` 的说明），
    所以它与真实工具名之间**可能漂移**。这条用例真的构造出工具集，
    逐个比对 —— 改了工具名却忘了改这张表，这里立刻变红。

    ⚠️ 用的是一整套真实的仓储与用户标识，不是打桩的假工具。
    用假工具就只是在测「我给的表和假工具对得上」，证明不了任何事。
    """
    from src.server.agents_factory import build_repositories
    from src.tools import build_business_tools

    repos = build_repositories()
    tools = build_business_tools(
        user_id="u-probe",
        transport_repo=repos.transport,
        hotel_repo=repos.hotel,
        policy_repo=repos.policy,
        order_repo=repos.order,
        approval_repo=repos.approval,
    )

    names = [tool.name for tool in tools]
    assert names, "工具集是空的，这条用例什么都没测到"
    missing = [name for name in names if name not in TITLES]
    assert not missing, (
        f"这些工具没有中文标题：{missing}。"
        f"请在 src/chains/events.py 的 TITLES 里补上——现在它们会显示成"
        f"「正在调用 xxx」。"
    )


# ---------------------------------------------------------------------------
# 二、工具调用的三个阶段
# ---------------------------------------------------------------------------
def test_arguments_are_assembled_from_fragments(adapter, collector, driver) -> None:
    """★★ 参数从 ``TOOL_CALL_DELTA`` 分片拼出来，且解析成字典。

    ⚠️ 参数是**分片**到达的（``agentscope/event/_event.py:313-347``），
    ``TOOL_CALL_END`` 里**没有**参数。把 ``collector.plan()`` 放在
    ``TOOL_CALL_START`` 上（那时参数还没到）会登记出一批 ``arguments={}``
    的任务 —— 界面展开详情时永远是空的，而展开详情正是用户点开
    「系统到底在干什么」时最想看的东西。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_transport", {"from": "杭州", "to": "北京"})

    task = by_name(collector, "search_transport")
    assert task is not None
    assert task.arguments == {"from": "杭州", "to": "北京"}
    assert task.state is TaskState.PENDING


def test_the_task_is_registered_at_call_end_not_call_start(adapter, collector, driver) -> None:
    """★★ 登记发生在 ``TOOL_CALL_END`` —— START 之后、END 之前**还没有**任务。

    ⚠️ 这条与上一条是两面：上一条查「最终参数对」，这条查「中间时刻
    不登记」。只有查中间时刻才能区分 ``plan`` 放在 START 还是 END 上，
    而放错位置时最终参数**是错的**（空字典），两者差别在界面上很明显、
    在断言里却很容易漏掉。
    """
    begin(adapter, driver)
    adapter.consume(
        driver(ToolCallStartEvent(reply_id=REPLY_ID, tool_call_id="c1", tool_call_name="search_hotels")),
    )
    adapter.consume(driver(ToolCallDeltaEvent(reply_id=REPLY_ID, tool_call_id="c1", delta='{"city"')))

    assert collector.snapshot() == [], "工具还没调用完就登记了任务"

    adapter.consume(driver(ToolCallDeltaEvent(reply_id=REPLY_ID, tool_call_id="c1", delta=': "杭州"}')))
    adapter.consume(driver(ToolCallEndEvent(reply_id=REPLY_ID, tool_call_id="c1")))

    assert len(collector.snapshot()) == 1


def test_a_tool_with_no_arguments_gets_an_empty_dict(adapter, collector, driver) -> None:
    """⚠️ 没有参数的工具（模型发了空 JSON）得到 ``{}``，不是 ``None``。"""
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "query_orders", fragments=["{}"])

    assert by_name(collector, "query_orders").arguments == {}


@pytest.mark.parametrize(
    "fragments",
    [
        ["{not json"],
        ['"just a string"'],
        ["[1, 2, 3]"],
        ["null"],
        [""],
        [],
    ],
)
def test_unparseable_arguments_still_register_the_task(
    adapter, collector, driver, fragments: list[str]
) -> None:
    """★★ 参数解析不出来时，任务**照样登记**，参数退化成 ``{}``。

    ⚠️ 这条守的是一个取舍：界面上「有个工具在跑」比「参数好看」重要得多。
    为了参数不合规就放弃登记，等于让用户完全看不到系统在干什么 ——
    而参数不合规恰恰是模型出问题的时候。

    ⚠️ 覆盖六种畸形：半个 JSON、合法但不是对象的 JSON、数组、``null``、
    空串、一个分片都没有。它们走的是不同的分支（``json.loads`` 抛异常 /
    解析出来不是 dict），一种过不代表另一种过。

    ⚠️ ``arguments`` 的契约是 ``dict[str, object]``。塞一个
    ``{"raw": "..."}`` 之类的半成品进去，界面就得为它写特例。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels", fragments=fragments)

    task = by_name(collector, "search_hotels")
    assert task is not None, "参数坏了就不登记任务，用户会完全看不到进度"
    assert task.arguments == {}
    assert isinstance(task.arguments, dict)


def test_a_reused_call_id_does_not_concatenate_old_fragments(adapter, collector, driver) -> None:
    """★★ 同一个 ``tool_call_id`` 被复用两次时，旧分片必须被丢掉。

    ⚠️ ``tool_call_id`` 是**模型生成**的，两次调用未必不同。不清旧分片的话，
    第二次的参数会接在第一次的后面，拼出一段**谁都没生成过**的 JSON。
    更糟的是它有时恰好还是合法 JSON —— 于是任务带着一组
    看起来合理、实际错误的参数显示在界面上。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels", fragments=['{"city": "杭州"}'])
    announce_call(adapter, driver, "c1", "search_hotels", fragments=['{"city": "北京"}'])

    tasks = [t for t in collector.snapshot() if t.name == "search_hotels"]
    assert len(tasks) == 2, "两次调用应该登记两条任务"
    assert tasks[0].arguments == {"city": "杭州"}
    assert tasks[1].arguments == {"city": "北京"}, "新参数粘上了旧分片"


def test_events_without_a_call_id_are_ignored(adapter, collector, driver) -> None:
    """⚠️ 没有 ``tool_call_id`` 的事件被忽略，不会登记出一个匿名任务。"""
    begin(adapter, driver)
    adapter.consume(
        driver(ToolCallStartEvent(reply_id=REPLY_ID, tool_call_id="", tool_call_name="search_hotels")),
    )
    adapter.consume(driver(ToolCallEndEvent(reply_id=REPLY_ID, tool_call_id="")))

    assert collector.snapshot() == []


# ---------------------------------------------------------------------------
# 三、工具结果
# ---------------------------------------------------------------------------
def test_success_marks_the_task_done_with_the_streamed_text(adapter, collector, driver) -> None:
    """★★ 成功时任务变 ``DONE``，且结果是**累积**出来的文本。

    ⚠️ ``TOOL_RESULT_END`` 只带 ``state``，**没有内容**（``event/_event.py``
    的 ``ToolResultEndEvent`` 只有 ``reply_id / tool_call_id / state``）。
    不自己累积分片的话，成功任务的 ``result`` 永远是空串 ——
    界面上「完成了，但什么都没说」。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_transport")
    result_of(adapter, driver, "c1", "找到 3 个航班", ToolResultState.SUCCESS)

    task = by_name(collector, "search_transport")
    assert task.state is TaskState.DONE
    assert task.result == "找到 3 个航班"
    assert task.error == ""


def test_the_result_start_event_moves_the_task_to_doing(adapter, collector, driver) -> None:
    """★★ ``TOOL_RESULT_START`` 把任务从 ``PENDING`` 推到 ``DOING``。

    ⚠️ 这条是「即将查询 → 正在查询」两段式的全部实现。少了它，界面上
    分不出「模型决定要查了」与「真的在查了」—— 而对查酒店、查政策这类
    慢工具，这两者之间隔着好几秒。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels")
    adapter.consume(
        driver(ToolResultStartEvent(reply_id=REPLY_ID, tool_call_id="c1", tool_call_name="search_hotels")),
    )

    assert by_name(collector, "search_hotels").state is TaskState.DOING


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (ToolResultState.ERROR, "工具执行出错"),
        (ToolResultState.DENIED, "该操作未获授权，已跳过"),
        (ToolResultState.INTERRUPTED, "执行被中断"),
    ],
)
def test_failure_states_map_to_readable_reasons(
    adapter, collector, driver, state, expected: str
) -> None:
    """★★ 三种失败状态都收成 ``FAILED``，且各有一句中文原因。

    ⚠️ 兜底文案存在的意义是：空的失败原因在界面上表现为
    「一个红叉加一片空白」，而失败恰恰是最需要解释的时刻。

    ⚠️ 三种分开参数化，是因为 ``DENIED``（用户拒绝）与 ``ERROR``（工具坏了）
    对用户是两件完全不同的事 —— 合并成一句「失败了」会让人把
    「我没同意」误读成「系统坏了」。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "submit_approval")
    result_of(adapter, driver, "c1", "", state)

    task = by_name(collector, "submit_approval")
    assert task.state is TaskState.FAILED
    assert task.error == expected
    assert task.result == ""


def test_the_tools_own_text_wins_over_the_fallback(adapter, collector, driver) -> None:
    """★★ 失败时优先用**工具自己**的输出文本，而不是兜底文案。

    ⚠️ 工具的错误信息通常已经是一句面向用户的中文（见
    ``src/tools/_result.py``），比适配器按状态码编一句更准确 ——
    状态码只知道「失败了」，不知道「为什么」。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_transport")
    result_of(adapter, driver, "c1", "杭州到北京今天没有余票了", ToolResultState.ERROR)

    assert by_name(collector, "search_transport").error == "杭州到北京今天没有余票了"


def test_running_does_not_finish_the_task(adapter, collector, driver) -> None:
    """★★ ``RUNNING`` 状态**不收尾**。

    ⚠️ 框架用它标记「结果未定」。把它当成功或失败都是在说谎 ——
    而用户会照着界面上那个「完成」去理解系统已经拿到了数据，
    进而以为可以下单了。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels")
    result_of(adapter, driver, "c1", "还在查", ToolResultState.RUNNING)

    assert by_name(collector, "search_hotels").state is TaskState.DOING


def test_a_result_without_a_call_is_skipped_with_a_warning(
    adapter, collector, driver, caplog: pytest.LogCaptureFixture
) -> None:
    """★★ 收到没有对应调用的结果时，**跳过并告警**。

    ⚠️ 这条路径是唯一能发现「事件映射漏了一种」的线索。不补登记 ——
    补一个没有名字、没有参数的任务比不显示更让人困惑；但必须留痕，
    否则事件流的问题永远没人发现。

    ⚠️ 断言有 ``WARNING`` 级别的日志。只断言「没崩」是不够的：
    静默跳过与告警跳过的代码在功能上无法区分，而它们的价值差得很远。
    """
    begin(adapter, driver)
    with caplog.at_level("WARNING"):
        adapter.consume(
            driver(
                ToolResultStartEvent(reply_id=REPLY_ID, tool_call_id="ghost", tool_call_name="search_hotels"),
            ),
        )

    assert collector.snapshot() == []
    assert any("没有找到对应的调用登记" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# 四、人工确认（HITL）
# ---------------------------------------------------------------------------
def test_confirmation_registers_one_pending_task(adapter, collector, driver) -> None:
    """★★ 等待确认时登记**一条** ``PENDING`` 任务。

    ⚠️ 停在 ``PENDING`` 而不是 ``DOING``：系统此刻没有在执行任何东西，
    它在等人。标成 ``DOING`` 会让界面显示一个永远转不完的圈 ——
    而用户正在犹豫要不要点确认，那是最容易让人以为系统卡住的时刻。
    """
    begin(adapter, driver)
    blocks = [ToolCallBlock(id=f"c{i}", name="submit_approval", input="{}") for i in range(2)]
    adapter.consume(driver(RequireUserConfirmEvent(reply_id=REPLY_ID, tool_calls=blocks)))

    tasks = collector.snapshot()
    assert len(tasks) == 1
    assert tasks[0].name == CONFIRM_TASK_NAME
    assert tasks[0].state is TaskState.PENDING
    assert "2" in tasks[0].title


def test_a_second_confirmation_request_does_not_register_a_second_task(adapter, collector, driver) -> None:
    """★★★ 一轮里分几批请求确认时：**仍只有一条**任务，且状态停在 ``PENDING``。

    ⚠️ 确认框在界面上是一个整体动作（要么全点要么全不点）。拆成多条会让
    用户以为要逐个处理，而界面上并没有对应的逐个操作入口。

    ⚠️⚠️ 断言状态这一条是本用例存在的**主要**理由，它守的是一个**已经
    真实存在过**的 bug：第二条事件到达时会走「已经登记过」的分支，
    而那个分支曾经调用 ``mark_doing`` 把它推进到 ``DOING``。

    这个错误为什么难发现：``PENDING → DOING`` 是**合法**迁移
    （``_TASK_TRANSITIONS``），所以收集器的 ``_transition`` 不会告警，
    日志里一个字都没有。表现只是「等待你确认」那条任务在界面上转圈，
    而用户正在犹豫要不要点确认 —— 那一刻他最容易以为系统卡住了。
    """
    begin(adapter, driver)
    block = ToolCallBlock(id="c1", name="submit_approval", input="{}")
    adapter.consume(driver(RequireUserConfirmEvent(reply_id=REPLY_ID, tool_calls=[block])))
    adapter.consume(driver(RequireUserConfirmEvent(reply_id=REPLY_ID, tool_calls=[block, block])))

    tasks = collector.snapshot()
    assert [t.name for t in tasks] == [CONFIRM_TASK_NAME]
    assert tasks[0].state is TaskState.PENDING, (
        "等待确认的任务被推进到了 DOING —— 界面上会是一个永远转不完的圈。"
        "注意 PENDING→DOING 是合法迁移，收集器不会为此告警。"
    )


def test_each_confirmation_event_carries_exactly_one_call(adapter, collector, driver) -> None:
    """★★★ 待确认的**项数**是把每条事件累加出来的，不是 ``len(tool_calls)``。

    ⚠️ 这条守的是一个**几乎不可能猜对**的框架细节：框架对每一个被挂起的
    工具调用**各发一条** ``RequireUserConfirmEvent``，且
    ``tool_calls=[tool_call]`` 的长度**恒为 1**
    （``agentscope/agent/_agent.py:2574-2577``）。

    所以「有几项待确认」如果取 ``len(tool_calls)``，它永远是 1 ——
    界面会在有 5 个调用等人确认时写着「等待你确认 1 项操作」。
    数字不对不会报错、不会告警，只是**永远偏小**，
    而用户会照着它判断「哦只有一个，那我点了」。

    ⚠️ 造三条**单元素**事件（就是框架真实的形状），断言标题里是 3。
    写成一条三元素事件的话，这条用例对错误的实现**照样会通过** ——
    那正是它要防的那种实现。
    """
    begin(adapter, driver)
    for i in range(3):
        adapter.consume(
            driver(
                RequireUserConfirmEvent(
                    reply_id=REPLY_ID,
                    tool_calls=[ToolCallBlock(id=f"c{i}", name="submit_approval", input="{}")],
                ),
            ),
        )

    tasks = collector.snapshot()
    assert len(tasks) == 1
    assert "3" in tasks[0].title, (
        f"三条单元素事件应累计成「等待你确认 3 项操作」，实际是 {tasks[0].title!r}"
    )
    assert tasks[0].arguments.get("pending") == 3


def test_the_confirmation_count_restarts_after_a_result(adapter, collector, driver) -> None:
    """★★ 确认结果到达后计数归零，下一批从 1 重新开始。

    ⚠️ 不归零的话计数会跨批次一直往上加：界面在有 2 项待确认时显示
    「等待你确认 7 项操作」。这个数字**看起来只是偏大**，
    所以没人会怀疑它是错的 —— 用户只会觉得「怎么这么多」。
    """
    begin(adapter, driver)
    block = ToolCallBlock(id="c1", name="submit_approval", input="{}")
    adapter.consume(driver(RequireUserConfirmEvent(reply_id=REPLY_ID, tool_calls=[block])))
    adapter.consume(driver(RequireUserConfirmEvent(reply_id=REPLY_ID, tool_calls=[block, block])))
    adapter.consume(
        driver(
            UserConfirmResultEvent(
                reply_id=REPLY_ID,
                confirm_results=[ConfirmResult(confirmed=True, tool_call=block)],
            ),
        ),
    )

    # 新的一批
    adapter.consume(driver(RequireUserConfirmEvent(reply_id=REPLY_ID, tool_calls=[block])))

    tasks = collector.snapshot()
    assert len(tasks) == 2, "新的一批应当**新登记**一条（上一批已经收尾了）"
    assert "1" in tasks[-1].title, f"计数没有归零：{tasks[-1].title!r}"
    assert tasks[-1].state is TaskState.PENDING


def test_a_continuation_reply_start_keeps_the_pending_confirmation(
    adapter, collector, driver
) -> None:
    """★★ 同一 ``reply_id`` 的**续接** ``REPLY_START`` 不得收掉待确认的任务。

    框架在人工确认之后会补发一个 ``reply_id`` **与暂停前相同**的
    ``ReplyStartEvent``（``agentscope/app/_service/_chat.py:1329-1344``），框架的注释
    明确要求消费者「不得因为收到相同 ``reply_id`` 的 ``REPLY_START`` 就清空
    累积的缓冲区」。

    这条与下一条是一对：续接要**保留**，新一轮要**收尾**。判据只有
    ``reply_id`` 一个 —— 把它俩写反的代价是「用户点了同意，进度条继续转圈」
    或者「界面留着一个永远转不完的圈」。
    """
    begin(adapter, driver, "reply-1")
    block = ToolCallBlock(id="c1", name="submit_approval", input="{}")
    adapter.consume(driver(RequireUserConfirmEvent(reply_id="reply-1", tool_calls=[block])))

    # 人工确认之后的续接：reply_id 不变。
    begin(adapter, driver, "reply-1")

    tasks = collector.snapshot()
    assert len(tasks) == 1, "续接的 REPLY_START 把待确认的任务收了 —— 用户刚点的同意会石沉大海"
    assert tasks[0].state is TaskState.PENDING


def test_a_new_reply_round_retires_the_leftover_confirmation(adapter, collector, driver) -> None:
    """★★★ 新一轮回复（``reply_id`` 变了）时，上一轮遗留的「等待你确认」必须收尾。

    ⚠️ 这条守的是一个**静默**的缺陷，它有两个症状，根因却是同一个：

        1. 计数不归零 —— 新一轮的标题在上一轮的数字上继续累加，
           界面上写着「等待你确认 3 项操作」而实际只有 1 项；
        2. 上一轮那条 ``!user_confirmation`` **永远停在 ``PENDING``** ——
           它是**伪任务**，不由任何工具结果收尾，唯一能收掉它的是
           ``_on_confirm_result``，而那个方法的入口判据正是
           ``self._confirm_task is None``。引用一丢，它就没人认领了。

    两个症状都不报错、不告警：前者只是数字偏大（用户只会觉得「怎么这么多」），
    后者只是界面上多一行转不完的圈。

    ⚠️ 断言「两条任务」而不是「一条」：新一轮的确认请求本来就该**新登记**
    一条（上一轮那条已经作废了）。复用同一条会让第二轮的用户看到一个
    混着上一轮数字的标题。
    """
    begin(adapter, driver, "reply-1")
    block = ToolCallBlock(id="c1", name="submit_approval", input="{}")
    adapter.consume(driver(RequireUserConfirmEvent(reply_id="reply-1", tool_calls=[block])))

    # reply_id 变了 ⇒ 这是真正的新一轮，不是续接。
    begin(adapter, driver, "reply-2")
    adapter.consume(driver(RequireUserConfirmEvent(reply_id="reply-2", tool_calls=[block])))

    tasks = [t for t in collector.snapshot() if t.name == CONFIRM_TASK_NAME]
    assert len(tasks) == 2, f"期望两条确认任务（一收尾一新登记），实际 {len(tasks)} 条"
    assert tasks[0].state is TaskState.DONE, (
        f"上一轮的确认请求停在 {tasks[0].state.value} —— 它永远不会有人认领"
        f"（伪任务只能由 _on_confirm_result 收尾，而引用已经被丢掉了）。"
        f" 界面上表现为一行永远转不完的「等待你确认」。"
    )
    assert tasks[1].state is TaskState.PENDING
    assert "1" in tasks[1].title, (
        f"计数没有随新一轮归零：{tasks[1].title!r}（应为「等待你确认 1 项操作」）"
    )


def test_an_empty_confirmation_request_registers_nothing(adapter, collector, driver) -> None:
    """⚠️ 空的确认请求（没有任何待确认调用）不登记任务。"""
    begin(adapter, driver)
    adapter.consume(driver(RequireUserConfirmEvent(reply_id=REPLY_ID, tool_calls=[])))

    assert collector.snapshot() == []


def test_a_user_refusal_is_not_a_failure(adapter, collector, driver) -> None:
    """★★ 用户**拒绝**收成 ``DONE``，不是 ``FAILED``。

    ⚠️ 用户点了「不同意」是系统按预期工作了。标成红色失败会让他以为
    自己做了什么不该做的事 —— 而这个红叉会一直留在他自己的历史记录里。
    """
    begin(adapter, driver)
    block = ToolCallBlock(id="c1", name="submit_approval", input="{}")
    adapter.consume(driver(RequireUserConfirmEvent(reply_id=REPLY_ID, tool_calls=[block])))
    adapter.consume(
        driver(
            UserConfirmResultEvent(
                reply_id=REPLY_ID,
                confirm_results=[ConfirmResult(confirmed=False, tool_call=block)],
            ),
        ),
    )

    task = by_name(collector, CONFIRM_TASK_NAME)
    assert task.state is TaskState.DONE
    assert "未同意" in task.result, "拒绝这件事必须在界面上说出来"


def test_an_approval_says_how_many_were_confirmed(adapter, collector, driver) -> None:
    """★★ 确认文案里的**数字必须对**。

    ⚠️ 这条守的是一个曾经真实存在的字段名错误：判据字段是
    ``ConfirmResult.confirmed``（``agentscope/event/_event.py:468-473``），
    不是 ``approved``。写错名字的后果是 ``getattr(item, "approved", False)``
    永远取到 ``False`` —— 用户点了同意，界面显示「已确认 0/1 项操作」。
    不报错、不告警，只是在最关键的那一步上主动误导人。
    """
    begin(adapter, driver)
    blocks = [ToolCallBlock(id=f"c{i}", name="submit_approval", input="{}") for i in range(3)]
    adapter.consume(driver(RequireUserConfirmEvent(reply_id=REPLY_ID, tool_calls=blocks)))
    adapter.consume(
        driver(
            UserConfirmResultEvent(
                reply_id=REPLY_ID,
                confirm_results=[
                    ConfirmResult(confirmed=True, tool_call=blocks[0]),
                    ConfirmResult(confirmed=True, tool_call=blocks[1]),
                    ConfirmResult(confirmed=False, tool_call=blocks[2]),
                ],
            ),
        ),
    )

    result = by_name(collector, CONFIRM_TASK_NAME).result
    assert "2" in result, f"确认数量不对：{result!r}"
    assert "1" in result, f"未同意数量不对：{result!r}"


def test_a_confirm_result_without_a_pending_task_is_ignored(adapter, collector, driver) -> None:
    """⚠️ 没有等待中的任务时，确认结果被忽略（不抛异常）。"""
    begin(adapter, driver)
    block = ToolCallBlock(id="c1", name="submit_approval", input="{}")
    adapter.consume(
        driver(
            UserConfirmResultEvent(
                reply_id=REPLY_ID,
                confirm_results=[ConfirmResult(confirmed=True, tool_call=block)],
            ),
        ),
    )

    assert collector.snapshot() == []


# ---------------------------------------------------------------------------
# 五、续接 vs 新回复（最容易错的一处）
# ---------------------------------------------------------------------------
def test_a_reply_start_with_the_same_reply_id_keeps_the_state(adapter, collector, driver) -> None:
    """★★★ 同一个 ``reply_id`` 的 ``REPLY_START`` 是**续接**，状态必须留着。

    已核实（``agentscope/app/_service/_chat.py:1329-1344``）：HITL 确认之后，服务层会
    补发一个 ``reply_id`` **与暂停前相同**的 ``ReplyStartEvent``，框架自己的
    注释写着「SSE 处理器**不得**因为收到相同 ``reply_id`` 的 ``REPLY_START``
    就清空累积的缓冲区 —— 这个事件表示续接，不是新回复」。

    ⚠️ 无条件清空的后果非常具体：用户点了「同意」，``submit_approval`` 的
    结果事件回到适配器时登记已经被抹掉，审批任务在界面上**永远停在转圈**。
    而这个 bug 不报错、不告警 —— 唯一的痕迹是那条
    「找不到对应的调用登记」，它把问题伪装成了上游事件乱序。

    ⚠️ 这条用例模拟的正是完整时序：调用 → 执行中 → 请求确认 → 回复结束
    （COMPLETED，**不是**失败）→ 续接 → 结果到达。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "submit_approval", {"order_id": "o-1"})
    adapter.consume(
        driver(ToolResultStartEvent(reply_id=REPLY_ID, tool_call_id="c1", tool_call_name="submit_approval")),
    )
    adapter.consume(
        driver(RequireUserConfirmEvent(
            reply_id=REPLY_ID,
            tool_calls=[ToolCallBlock(id="c1", name="submit_approval", input="{}")],
        )),
    )
    adapter.consume(driver(ReplyEndEvent(session_id="s", reply_id=REPLY_ID)))
    assert by_name(collector, "submit_approval").state is TaskState.DOING

    # 续接：同一个 reply_id 的 REPLY_START。
    begin(adapter, driver)
    result_of(adapter, driver, "c1", "申请已提交，单号 AP-1", ToolResultState.SUCCESS)

    task = by_name(collector, "submit_approval")
    assert task.state is TaskState.DONE, "续接时清了状态，审批任务会永远转圈"
    assert task.result == "申请已提交，单号 AP-1"


def test_a_reply_start_with_a_new_reply_id_clears_the_state(adapter, collector, driver) -> None:
    """★★ 不同 ``reply_id`` 的 ``REPLY_START`` 是**新回复**，中间状态要清掉。

    ⚠️ 与上一条成对。只做「续接不清」而不做「新回复要清」，上一轮的分片
    会拼进这一轮的结果里 —— 而 ``tool_call_id`` 是模型生成的，两次回复里
    未必不同，所以这不是理论问题。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels", {"city": "杭州"})
    adapter.consume(
        driver(ToolResultStartEvent(reply_id=REPLY_ID, tool_call_id="c1", tool_call_name="search_hotels")),
    )

    begin(adapter, driver, reply_id="reply-2")
    # 新回复里复用同一个 ``tool_call_id``，直接送来结果。
    result_of(adapter, driver, "c1", "找到 3 家", ToolResultState.SUCCESS)

    # ⚠️ 旧登记被清掉的**可观测后果**：新回复里 c1 是一次「没有登记的调用」，
    # 那个结果被跳过，旧任务没有跟着被推成 DONE（它停在上一轮的 DOING）。
    # 若中间状态没清，这一行会变成 DONE —— 界面上表现为上一轮那个
    # 早已作废的任务，被这一轮一个碰巧同名的调用「补完了」。
    task = by_name(collector, "search_hotels")
    assert task.state is TaskState.DOING
    assert task.result == ""


def test_reply_start_does_not_clear_the_collector(adapter, collector, driver) -> None:
    """★★ 新回复**只**清适配器的中间状态，不清空收集器。

    ⚠️ 收集器里的清单是「这次会话的思考链」，跨回复连续。清掉它等于把
    界面上正在显示的历史任务抹掉 —— 用户会看到自己刚才的操作记录消失。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels")
    result_of(adapter, driver, "c1", "找到 3 家", ToolResultState.SUCCESS)
    before = [t.task_id for t in collector.snapshot()]

    begin(adapter, driver, reply_id="reply-2")

    assert [t.task_id for t in collector.snapshot()] == before


def test_reset_clears_everything_including_the_reply_id(adapter, collector, driver) -> None:
    """⚠️ ``reset()`` 清中间状态**并**忘掉 ``reply_id``。

    ⚠️ 忘掉 ``reply_id`` 这一点是必须的：否则一次 ``reset()`` 之后再喂
    同一个 ``reply_id`` 的 ``REPLY_START``，会被误判成「续接」而不清理 ——
    调用方明确要求重来，却被当成了续接。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels", {"city": "杭州"})
    adapter.reset()

    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels", fragments=["{}"])

    tasks = [t for t in collector.snapshot() if t.name == "search_hotels"]
    assert tasks[-1].arguments == {}, "reset() 之后旧分片还在"


# ---------------------------------------------------------------------------
# 六、回复结束与强行收尾
# ---------------------------------------------------------------------------
def test_a_completed_reply_leaves_running_tasks_alone(adapter, collector, driver) -> None:
    """★★★ 正常结束（``COMPLETED``）**不**收尾在途任务。

    ⚠️ 这条是整个收尾逻辑的边界。人工确认路径下回复也会结束（而且是
    ``COMPLETED``），那一刻的任务是**正常等待** —— 把它们标成失败等于
    替用户做了决定，而且是在他还没来得及点确认的时候。

    ⚠️ 与 ``test_a_user_refusal_is_not_a_failure`` 是两种不同的「不是失败」：
    那条讲用户拒绝，这条讲系统在等人。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "submit_approval")
    adapter.consume(
        driver(ToolResultStartEvent(reply_id=REPLY_ID, tool_call_id="c1", tool_call_name="submit_approval")),
    )
    adapter.consume(driver(ReplyEndEvent(session_id="s", reply_id=REPLY_ID)))

    assert by_name(collector, "submit_approval").state is TaskState.DOING


def test_reply_end_with_exceed_max_iters_fails_running_tasks(adapter, collector, driver) -> None:
    """★★★ ``EXCEED_MAX_ITERS`` 收尾在途任务 —— 走 ``ReplyEndEvent``。

    ⚠️ 这是**正路**。已核实（``agentscope/event/_event.py:425-431``）：
    ``ExceedMaxItersEvent`` 带 ``@deprecated``，文档写着「仍为向后兼容而
    发出，但**不携带语义**；请改用 ``ReplyEndEvent.finished_reason``」。
    只认那个废弃事件，等于把「任务收尾」押在一个已宣布会消失的事件上 ——
    而它消失的那天，界面上会重新出现永远转圈的任务，且不会有任何报错。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels")
    adapter.consume(
        driver(ToolResultStartEvent(reply_id=REPLY_ID, tool_call_id="c1", tool_call_name="search_hotels")),
    )
    adapter.consume(
        driver(
            ReplyEndEvent(
                session_id="s",
                reply_id=REPLY_ID,
                finished_reason=ReplyFinishedReason.EXCEED_MAX_ITERS,
            ),
        ),
    )

    task = by_name(collector, "search_hotels")
    assert task.state is TaskState.FAILED
    assert "上限" in task.error


def test_the_deprecated_exceed_event_still_works(adapter, collector, driver) -> None:
    """★★ 废弃的 ``EXCEED_MAX_ITERS`` 事件**仍然**收尾（向后兼容）。

    ⚠️ 本地版本两处都发（``agentscope/agent/_agent.py:3599`` 等），旧版本框架只发这一个。
    两条路都要通，否则升级/降级框架时会有一边静默失效。
    """
    import warnings

    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels")
    adapter.consume(
        driver(ToolResultStartEvent(reply_id=REPLY_ID, tool_call_id="c1", tool_call_name="search_hotels")),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        event = ExceedMaxItersEvent(reply_id=REPLY_ID, name="main")
    adapter.consume(driver(event))

    assert by_name(collector, "search_hotels").state is TaskState.FAILED


def test_both_abandonment_signals_together_are_idempotent(adapter, collector, driver) -> None:
    """★★ 两个信号都到（真实的顺序）时，只收尾一次。

    ⚠️ 框架的实际顺序是**先** ``EXCEED_MAX_ITERS`` **后** ``REPLY_END``。
    收尾不幂等的话，第二次会对已失败的任务再调一次 ``add_result`` ——
    收集器会打一条「非法状态迁移」的告警（它是对的，但那条告警会让人
    以为事件流有问题）。更糟的是若哪天收集器改成允许终态自迁移，
    这里就会**重复通知**监听器，界面闪一下。
    """
    import warnings

    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels")
    adapter.consume(
        driver(ToolResultStartEvent(reply_id=REPLY_ID, tool_call_id="c1", tool_call_name="search_hotels")),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        deprecated = ExceedMaxItersEvent(reply_id=REPLY_ID, name="main")
    adapter.consume(driver(deprecated))
    first = by_name(collector, "search_hotels").error
    adapter.consume(
        driver(
            ReplyEndEvent(
                session_id="s",
                reply_id=REPLY_ID,
                finished_reason=ReplyFinishedReason.EXCEED_MAX_ITERS,
            ),
        ),
    )

    assert by_name(collector, "search_hotels").error == first


def test_reply_end_with_error_fails_running_tasks(adapter, collector, driver) -> None:
    """★★ ``ERROR`` 也收尾 —— 与 ``EXCEED_MAX_ITERS`` 同理。

    ⚠️ 回复因错误终止时，那些 ``DOING`` 的任务不会再有结果。不收尾的话
    界面永远转圈，而用户已经在别处看到「系统出错了」—— 两处信息互相矛盾。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels")
    adapter.consume(
        driver(ToolResultStartEvent(reply_id=REPLY_ID, tool_call_id="c1", tool_call_name="search_hotels")),
    )
    adapter.consume(
        driver(
            ReplyEndEvent(
                session_id="s",
                reply_id=REPLY_ID,
                finished_reason=ReplyFinishedReason.ERROR,
            ),
        ),
    )

    assert by_name(collector, "search_hotels").state is TaskState.FAILED


def test_an_error_does_not_touch_a_task_waiting_for_the_user(adapter, collector, driver) -> None:
    """★★★ ``ERROR`` 收尾**只针对 ``DOING``**，审核中的申请单不受影响。

    ⚠️ 等待人工确认的任务停在 ``PENDING``（**不是** ``DOING``），而它
    在下一轮回复里会被同一个 ``tool_call_id`` 认领并正常收尾 ——
    申请单还在审批人手里，我们没有资格替它宣布结束。

    ⚠️ 这条同时把「哪些状态该收」这件事钉死：如果哪天收尾改成
    「所有未结束的任务」，它会变红。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "cP", "submit_approval")
    announce_call(adapter, driver, "cD", "search_hotels")
    adapter.consume(
        driver(ToolResultStartEvent(reply_id=REPLY_ID, tool_call_id="cD", tool_call_name="search_hotels")),
    )
    adapter.consume(
        driver(
            ReplyEndEvent(
                session_id="s",
                reply_id=REPLY_ID,
                finished_reason=ReplyFinishedReason.ERROR,
            ),
        ),
    )

    assert by_name(collector, "submit_approval").state is TaskState.PENDING
    assert by_name(collector, "search_hotels").state is TaskState.FAILED


def test_an_interrupted_reply_does_not_force_finish(adapter, collector, driver) -> None:
    """★★ ``INTERRUPTED`` **不**强行收尾 —— 框架已经补发过工具结果了。

    ⚠️ 已核实（``agentscope/agent/_agent.py:1008-1031``）：中断时框架会为每个在途调用
    补发 ``TOOL_RESULT_START`` + ``INTERRUPTED`` 的 ``TOOL_RESULT_END``。
    适配器照常收成 ``FAILED``，走的是一般路径 —— 不需要、也不该有第二条
    收尾逻辑（两条路都在写同一个任务，会重复通知监听器）。
    """
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels")
    adapter.consume(
        driver(ToolResultStartEvent(reply_id=REPLY_ID, tool_call_id="c1", tool_call_name="search_hotels")),
    )
    # 框架在中断时补发的结果
    adapter.consume(
        driver(ToolResultEndEvent(reply_id=REPLY_ID, tool_call_id="c1", state=ToolResultState.INTERRUPTED)),
    )
    adapter.consume(
        driver(
            ReplyEndEvent(
                session_id="s",
                reply_id=REPLY_ID,
                finished_reason=ReplyFinishedReason.INTERRUPTED,
            ),
        ),
    )

    task = by_name(collector, "search_hotels")
    assert task.state is TaskState.FAILED
    assert task.error == "执行被中断"


# ---------------------------------------------------------------------------
# 七、健壮性与纯度
# ---------------------------------------------------------------------------
def test_consume_never_raises(adapter, collector) -> None:
    """★★ ``consume`` 对**任意**输入都不抛异常。

    ⚠️ 它跑在 ``reply_stream`` 的消费循环里，抛出去会**中断整轮回复** ——
    而思考链是投影，投影坏了不该让业务也坏掉。

    ⚠️ 喂的是各种离谱输入，包括 ``type`` 不是字符串的、``delta`` 是
    整数的、以及 ``None``。这些不该出现在正常事件流里，但适配器是
    外部输入的边界，边界上「不该出现」不是一种保证。
    """
    for weird in (
        None,
        object(),
        {},
        {"type": "NO_SUCH_EVENT"},
        {"type": 123},
        {"type": "TOOL_CALL_DELTA", "delta": 12345},
        {"type": "THINKING_BLOCK_DELTA", "delta": None},
        {"type": "TOOL_CALL_START", "tool_call_id": None, "tool_call_name": None},
        {"type": "TOOL_RESULT_END", "tool_call_id": "x", "state": "no_such_state"},
        {"type": "REPLY_END", "finished_reason": "no_such_reason"},
    ):
        adapter.consume(weird)  # 不抛即通过

    assert collector.snapshot() == []


def test_unknown_event_types_are_silently_ignored(adapter, collector, driver) -> None:
    """★ 认不出的事件静默忽略（不记日志、不登记任务）。

    ⚠️ ``AgentEvent`` 是 28 个成员的联合，绝大多数（``DATA_BLOCK_*``、
    ``MODEL_CALL_*`` 等）与本项目无关。逐条告警会把日志淹掉，
    真正的异常反而被埋掉。
    """
    begin(adapter, driver)
    before = len(collector.snapshot())
    for ignored in (
        {"type": "MODEL_CALL_START", "reply_id": REPLY_ID, "model_name": "qwen"},
        {"type": "MODEL_CALL_END", "reply_id": REPLY_ID},
        {"type": "DATA_BLOCK_DELTA", "reply_id": REPLY_ID, "block_id": "b", "delta": "x"},
        {"type": "TOOL_RESULT_DATA_DELTA", "reply_id": REPLY_ID, "tool_call_id": "c", "delta": {}},
    ):
        adapter.consume(ignored)

    assert len(collector.snapshot()) == before


def test_reasoning_is_accumulated_by_default(driver) -> None:
    """★ 默认累积模型推理文本（思考链的「显示推理」）。"""
    collector = TaskCollector()
    adapter = EventChainAdapter(collector)
    begin(adapter, driver)
    for piece in ("我先看看", "用户的", "出发地"):
        adapter.consume(
            driver(ThinkingBlockDeltaEvent(reply_id=REPLY_ID, block_id="b1", delta=piece)),
        )

    assert adapter.reasoning == "我先看看用户的出发地"


def test_reasoning_is_not_even_stored_when_disabled(driver) -> None:
    """★★ ``expose_reasoning=False`` 时**根本不累积**，不只是不显示。

    ⚠️ 差别不是可有可无的：推理文本可能含用户没说过的个人信息
    （模型从上下文里推断的），让它在内存里多留一份没有收益。
    「不显示」只是不渲染，数据还在。
    """
    adapter = EventChainAdapter(TaskCollector(), expose_reasoning=False)
    begin(adapter, driver)
    adapter.consume(driver(ThinkingBlockDeltaEvent(reply_id=REPLY_ID, block_id="b1", delta="悄悄话")))

    assert adapter.reasoning == ""
    assert adapter._reasoning == [], "关掉之后仍然把推理存进了内存"


def test_snapshot_iteration_yields_flat_dicts(driver) -> None:
    """★ ``iter_task_states`` 产出可 JSON 序列化的扁平字典。

    ⚠️ ``duration_seconds`` 未结束时**保留 ``None``**，不填 0 ——
    前端要靠它区分「还没跑完」和「一瞬间就跑完了」。
    """
    collector = TaskCollector()
    adapter = EventChainAdapter(collector)
    begin(adapter, driver)
    announce_call(adapter, driver, "c1", "search_hotels")
    adapter.consume(
        driver(ToolResultStartEvent(reply_id=REPLY_ID, tool_call_id="c1", tool_call_name="search_hotels")),
    )

    rows = list(iter_task_states(collector))
    assert len(rows) == 1
    assert rows[0]["state"] == TaskState.DOING.value
    assert rows[0]["duration_seconds"] is None
    assert rows[0]["title"] == TITLES["search_hotels"]
    json.dumps(rows)  # 必须可序列化


def test_consume_stream_drains_the_whole_stream(driver) -> None:
    """★★ ``consume_stream`` 把流**读干**，并丢弃 ``Msg``。

    ⚠️ 不读完的后果不是「少处理几个事件」，而是框架内部的 ``asyncio.Queue``
    生产者会在某个 ``put()`` 上**永久阻塞**，那个任务不会被回收。

    ⚠️ 同时断言 ``Msg`` 被丢弃：``reply_stream`` 的产出是
    ``AgentEvent | Msg``，最终那条 ``Msg`` 承载结构化结果，与任务清单无关。
    """
    import asyncio

    from agentscope.message import Msg, TextBlock

    collector = TaskCollector()
    adapter = EventChainAdapter(collector)

    async def stream():
        """产出一条事件、一条消息、又一条事件。"""
        yield ReplyStartEvent(session_id="s", reply_id=REPLY_ID, name="main")
        # ⚠️ ``Msg.content`` **必须**是 list（``message/_base.py``）
        yield Msg(name="main", content=[TextBlock(type="text", text="最终回答")], role="assistant")
        yield ToolCallStartEvent(reply_id=REPLY_ID, tool_call_id="c1", tool_call_name="search_hotels")
        yield ToolCallEndEvent(reply_id=REPLY_ID, tool_call_id="c1")

    asyncio.run(adapter.consume_stream(stream()))

    assert [t.name for t in collector.snapshot()] == ["search_hotels"]


def test_the_adapter_is_not_exported_from_the_package() -> None:
    """★★ 纯度探针：``import src.chains`` **不得**把 ``agentscope`` 拉进来。

    ⚠️ 必须在**子进程**里验证 —— 进程内 ``sys.modules`` 早被 ``conftest.py``
    污染了，断言必然假通过。

    这条守的是 ``src/chains/__init__.py`` 的设计：:mod:`src.chains.events`
    必须 import 框架，而 :mod:`src.chains.collector` 刻意不 import。
    一旦有人把 ``events`` 加进包的 ``__init__``，收集器那 40 多条纯单测
    就全部依赖框架了 —— 改框架导入路径会让整个测试文件在**收集阶段**报错，
    连不相关的用例都跑不了。

    ⚠️ 这个改动**不会让任何功能测试变红**，只有这条会。
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
