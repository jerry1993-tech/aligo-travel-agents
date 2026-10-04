# -*- coding: utf-8 -*-
"""快慢车道路由中间件的测试。

核心是 **P3 的验收断言**：
:func:`test_fast_lane_costs_exactly_one_model_call` 直接数模型被真实调用了
几次 —— 快车道 1 次，慢车道（带工具调用的完整 ReAct）2 次。

⚠️ 本文件与其它单测不同，它**真的构造 Agent 并跑 reply_stream**。
理由：快车道的全部价值在于「省下一次模型调用」，而这件事**只能**在真实的
ReAct 循环里被观测到。给中间件喂养假对象、断言「它返回了 ChatResponse」，
测的是「我写的代码和我以为的一样」，而不是「框架真的会因此少调一次模型」——
后者才是这个设计要证明的事。

代价是这个文件比别的慢（每条用例要构造 Agent、跑完整循环）。可接受：
关键路径的正确性值得几秒钟。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock
from agentscope.tool import FunctionTool, ToolChunk, Toolkit

from src.domain import Intent, LaneName, TripStage
from src.llm.mock import MockChatModel
from src.orchestration.lane import (
    DEFAULT_ROUTE_TOOL,
    INTERNAL_TOOLS,
    LANE_HIDDEN_TOOLS,
    ORCHESTRATION_TOOLS,
    PROCESS_CONTROL_TOOLS,
    ROUTE_DECISION_KEY,
    ROUTE_TOOL_INPUT_FIELDS,
    TOOL_MANAGEMENT_TOOLS,
    WORKSPACE_TOOLS,
    LaneRouterMiddleware,
)
from src.tools.route import ROUTE_TOOL_NAME, aligo_route_intent

# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------


class CountingModel(MockChatModel):
    """会数自己被打了几次的 Mock 模型。

    ⚠️ 数的是 ``_call_api``（真正发请求的那一层），不是 ``__call__``。
    基类的 ``__call__`` 还包含流式累积与重试，而「快车道省掉一次模型调用」
    要省的是**对外的那一次请求**。数 ``__call__`` 会把自己的重试也算进去，
    数字就不再干净。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls = 0

    async def _call_api(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return await super()._call_api(*args, **kwargs)


#: 慢车道对照组用的无害只读工具。
PING_EXECUTED: list[str] = []


def ping(note: str = "") -> ToolChunk:
    """一个什么都不做的只读工具，用于触发第二轮 ReAct。

    Args:
        note (str): 随便什么内容。

    Returns:
        ToolChunk: 固定回复。
    """
    PING_EXECUTED.append(note)
    return ToolChunk(content=[TextBlock(text="pong")])


def make_agent(
    *,
    middleware: LaneRouterMiddleware,
    name: str = "main_plan",
    with_route_tool: bool = True,
) -> Agent:
    """构造一个挂了中间件的 Agent。

    Args:
        middleware (`LaneRouterMiddleware`): 要挂的中间件。
        name (`str`): agent 名字。
        with_route_tool (`bool`): 是否注册路由工具。

    Returns:
        `Agent`: 构造好的 agent。
    """
    tools: list[Any] = [FunctionTool(ping, is_read_only=True)]
    if with_route_tool:
        tools.append(FunctionTool(aligo_route_intent, is_read_only=True))
    return Agent(
        name=name,
        system_prompt="你是差旅助手。",
        model=CountingModel(),
        toolkit=Toolkit(tools=tools),
        middlewares=[middleware],
    )


async def run_reply(
    agent: Agent,
    text: str,
) -> tuple[int, dict[str, Any] | None]:
    """跑完一轮回复。

    Args:
        agent (`Agent`): 目标 agent。
        text (`str`): 用户输入。

    Returns:
        `tuple[int, dict | None]`: (模型调用次数, 记录下来的路由决策)。
    """
    async for _ in agent.reply_stream(
        inputs=Msg(name="user", role="user", content=[TextBlock(text=text)]),
    ):
        pass
    recorded = agent.state.middle_context.get(ROUTE_DECISION_KEY)
    return agent.model.calls, recorded  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# 一、P3 验收：快车道真的省下一次模型调用
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["规划行程", "帮我规划行程", "查询订单", "查询政策", "取消"],
)
def test_fast_lane_costs_exactly_one_model_call(text: str) -> None:
    """**P3 验收断言**：快车道下模型只被调用 **1** 次。

    ⚠️ 这个数字是快车道存在的全部理由。慢车道要 2 次（判断该调什么工具 +
    组织最终答复），快车道把第一次整个跳过。

    若这条挂了，说明 ``on_model_call`` 没有真正短路 —— 而它可能看起来
    「功能正常」（工具照样被调用、回复照样出来），只是**根本没省下任何东西**。
    这正是必须用计数而不是用行为断言的原因。
    """
    agent = make_agent(middleware=LaneRouterMiddleware(agent_names=("main_plan",)))
    calls, recorded = asyncio.run(run_reply(agent, text))

    assert calls == 1, f"快车道应当只调 1 次模型，实际 {calls} 次"
    assert recorded is not None
    assert recorded["lane"] == LaneName.FAST.value


class OneToolCallModel(MockChatModel):
    """第 1 次调用产出一个工具调用，之后正常答一句话。

    ⚠️ 为什么不用 Mock 的 ``#mock-tool:`` 指令来撑起 ReAct 第二轮：
    指令是从**消息历史**里解析的，而那条用户消息在第二轮**仍然在上下文里**
    —— 于是模型每一轮都重新发出同一个工具调用，ReAct 转满 50 轮
    （``exceeds the max iteration numbers 50``）。工具确实跑了，
    但调用次数是 51 而不是 2，这条对照组就没法用来做数量对照了。

    ⚠️ 也不能改成「第二次调用返回文本」来判断「是不是第一轮」——
    那样测的就是 Mock 的行为，而不是框架的。这里用**调用序号**驱动：
    第 1 次给工具调用、第 2 次给文本，恰好构成慢车道最小的一个完整
    ReAct 循环，于是 ``calls == 2`` 是**框架行为**的直接读数。

    ⚠️ 直接继承 ``MockChatModel`` 而**不是** ``CountingModel``。继承后者
    会让计数发生两次：本类自增一次，再经 ``super()._call_api`` 进到
    ``CountingModel`` 又自增一次 —— 于是真实的 2 次调用被读成 3 次，
    而失败信息看起来像「框架多调了一次模型」。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls = 0

    async def _call_api(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            # 已核实：返回一个普通的 ``ChatResponse(is_last=True)`` 即可，
            # 不必构造异步生成器（消费侧对 ``isinstance(res, ChatResponse)``
            # 的分支是「直接当作完整响应使用」）。
            from agentscope.message import ToolCallBlock
            from agentscope.model import ChatResponse

            return ChatResponse(
                content=[ToolCallBlock(id="slow-1", name="ping", input="{}")],
                is_last=True,
            )
        return await super()._call_api(*args, **kwargs)


def test_slow_lane_costs_two_model_calls() -> None:
    """对照组：慢车道下一个**会调工具的**输入要 2 次模型调用。

    ⚠️ 没有这条，「快车道只调 1 次」可能只是因为整个链路本来就只有 1 次
    （比如模型直接答完就结束），那样上面那条断言就什么都没证明。
    两相对照，「省下一次」才有意义 —— 它省掉的是**路由**那一轮，
    不是「这个用例碰巧只有一轮」。
    """
    PING_EXECUTED.clear()
    agent = Agent(
        name="main_plan",
        system_prompt="你是差旅助手。",
        model=OneToolCallModel(),
        toolkit=Toolkit(tools=[FunctionTool(ping, is_read_only=True)]),
        middlewares=[LaneRouterMiddleware(agent_names=("main_plan",))],
    )
    # 输入必须**不命中**快车道规则：一句长句（超过 max_chars=20）。
    text = "我想了解一下公司差旅政策是怎么规定的"
    calls, recorded = asyncio.run(run_reply(agent, text))

    assert recorded is not None
    assert recorded["lane"] == LaneName.SLOW.value
    assert PING_EXECUTED, "对照组必须真的执行过工具，否则 ReAct 第二轮没跑起来"
    assert calls == 2, f"慢车道 + 一次工具调用应当调 2 次模型，实际 {calls} 次"


def test_fast_lane_actually_executes_the_routed_tool() -> None:
    """快车道合成的工具调用会被**真正执行**，不是只发个事件。

    ⚠️ 这条防的是「看起来短路了，其实工具压根没跑」：那种情况下模型调用数
    同样是 1，但业务上什么都没发生 —— 用户点了「查询订单」，系统回了一句话，
    订单却一条没查。
    """
    executed: list[str] = []

    def spy_route(intent: str, matched_rule: str = "", target_agents: list[str] | None = None, reason: str = "") -> ToolChunk:
        """记录一次路由。"""
        executed.append(intent)
        return ToolChunk(content=[TextBlock(text=f"routed:{intent}")])

    agent = Agent(
        name="main_plan",
        system_prompt="你是差旅助手。",
        model=CountingModel(),
        toolkit=Toolkit(tools=[FunctionTool(spy_route, name=ROUTE_TOOL_NAME, is_read_only=True)]),
        middlewares=[LaneRouterMiddleware(agent_names=("main_plan",))],
    )
    asyncio.run(run_reply(agent, "查询订单"))

    assert executed == [Intent.QUERY_ORDER.value]


def test_fast_lane_does_not_fire_twice_in_one_reply() -> None:
    """一次回复里只路由一次 —— 同一个工具不会被合成调用两遍。

    ⚠️ 重复命中的后果不是「多查一次」，而是**下单类操作被执行两次**。
    ``cur_iter`` 与 ``reply_id`` 两个守卫就是为这条服务的。
    """
    executed: list[str] = []

    def spy_route(intent: str, matched_rule: str = "", target_agents: list[str] | None = None, reason: str = "") -> ToolChunk:
        """记录一次路由。"""
        executed.append(intent)
        return ToolChunk(content=[TextBlock(text=f"routed:{intent}")])

    agent = Agent(
        name="main_plan",
        system_prompt="你是差旅助手。",
        model=CountingModel(),
        toolkit=Toolkit(tools=[FunctionTool(spy_route, name=ROUTE_TOOL_NAME, is_read_only=True)]),
        middlewares=[LaneRouterMiddleware(agent_names=("main_plan",))],
    )
    calls, _ = asyncio.run(run_reply(agent, "规划行程"))

    assert executed == [Intent.PLAN_TRIP.value], f"路由工具被调了 {len(executed)} 次"
    assert calls == 1


# ---------------------------------------------------------------------------
# 二、守卫：中间件在什么情况下必须完全透明
# ---------------------------------------------------------------------------


def test_other_agents_are_not_routed() -> None:
    """``agent_names`` 之外的 agent 完全不受影响。

    ⚠️ 这不是可选的限制。快车道规则表描述的是「用户点了什么按钮」，
    而子智能体的输入常常是**内部调用产生的指令文本**（比如「查询政策」
    四个字），与按钮文案一模一样。不加限制的话，子智能体的模型调用会被
    莫名其妙地短路掉。
    """
    agent = make_agent(
        middleware=LaneRouterMiddleware(agent_names=("main_plan",)),
        name="policy_rag",
    )
    calls, recorded = asyncio.run(run_reply(agent, "查询政策"))

    assert recorded is None, "不该被路由的 agent 却记录了路由决策"
    assert calls == 1  # Mock 直接答完，没有工具调用


def test_disabled_middleware_is_transparent() -> None:
    """``enabled=False`` 时中间件不做任何事。"""
    agent = make_agent(middleware=LaneRouterMiddleware(enabled=False))
    calls, recorded = asyncio.run(run_reply(agent, "规划行程"))

    assert recorded is None
    assert calls == 1


def test_slow_lane_decision_is_still_recorded() -> None:
    """慢车道的判定**同样**被记录。

    ⚠️ 只记快车道命中，等于把「为什么没命中」这个更常见的问题丢掉了。
    排障时第一句话就是「用户说了一模一样的话，为什么这次没走快车道」。
    """
    agent = make_agent(middleware=LaneRouterMiddleware())
    _, recorded = asyncio.run(run_reply(agent, "怎么报销？"))

    assert recorded is not None
    assert recorded["lane"] == LaneName.SLOW.value
    assert recorded["reason"], "慢车道的判定必须说明原因"


def test_missing_route_tool_does_not_break_the_reply() -> None:
    """工具表里没有路由工具时，回复不崩 —— 只是快车道失效。

    ⚠️ 真实场景：装配时漏注册了 ``aligo_route_intent``。框架会把
    ``ToolNotFoundError`` 当成一次普通的工具失败结果交给模型（不抛异常），
    所以这条路径很容易带着问题上线。测试确保它至少**不崩**，
    且模型仍然完成了这一轮回复。
    """
    agent = make_agent(
        middleware=LaneRouterMiddleware(agent_names=("main_plan",)),
        with_route_tool=False,
    )
    calls, recorded = asyncio.run(run_reply(agent, "规划行程"))

    # 决策仍然被记录了（中间件记录在合成之前），回复也完成了。
    assert recorded is not None
    assert recorded["lane"] == LaneName.FAST.value
    assert calls == 1


# ---------------------------------------------------------------------------
# 三、消息解析：一个实测踩到的坑
# ---------------------------------------------------------------------------


class _RecordingLane(LaneRouterMiddleware):
    """把每次 ``on_model_call`` 看到的消息列表记下来的中间件。"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.seen: list[list[str]] = []

    async def on_model_call(self, agent: Any, input_kwargs: dict[str, Any], next_handler: Any) -> Any:
        self.seen.append([getattr(m, "role", "?") for m in (input_kwargs.get("messages") or [])])
        return await super().on_model_call(agent, input_kwargs, next_handler)


def test_last_message_is_an_assistant_placeholder() -> None:
    """⚠️ 记录一个**实测事实**：``messages[-1]`` 是空的 assistant 占位消息。

    框架会先往上下文里塞一条空的 assistant 消息等着模型填，再把列表交给
    ``on_model_call``。所以「检查最后一个元素是不是 user」这个写法看起来
    完全合理，实测却让快车道**一次都不会触发**。

    这条断言把该事实钉住：若将来的框架版本改了行为（比如不再塞占位消息），
    这里会红，而写这条注释的人就知道该回来看
    :meth:`LaneRouterMiddleware._last_user_text` 了。
    """
    mw = _RecordingLane(agent_names=("main_plan",))
    agent = make_agent(middleware=mw)
    asyncio.run(run_reply(agent, "规划行程"))

    assert mw.seen, "中间件应当至少被调用一次"
    first = mw.seen[0]
    assert first[-1] == "assistant", f"最后一个消息应当是 assistant 占位，实际 {first}"
    assert "user" in first, f"消息列表里必须有用户消息，实际 {first}"


def test_user_message_is_found_despite_the_trailing_placeholder() -> None:
    """尽管末尾是 assistant 占位，中间件仍能正确拿到用户文本并命中快车道。

    与上一条互为表里：上一条钉住框架的行为，这一条钉住我们的应对。
    """
    mw = _RecordingLane(agent_names=("main_plan",))
    agent = make_agent(middleware=mw)
    _, recorded = asyncio.run(run_reply(agent, "查询订单"))

    assert recorded is not None
    assert recorded["matched_rule"] == "query_order"


def test_no_user_message_means_no_routing() -> None:
    """消息列表里没有用户消息时不路由（而不是拿工具结果去匹配）。"""
    mw = LaneRouterMiddleware()
    decision = mw._decide(_FakeAgent(roles=[]), {"messages": []})  # noqa: SLF001
    assert decision is None


class _FakeAgent:
    """供守卫单测用的极小 agent 替身。"""

    def __init__(self, *, roles: list[str] | None = None, cur_iter: int = 0, name: str = "main_plan") -> None:
        from src.domain import AgentName

        self.name = name
        self.state = _FakeState(cur_iter=cur_iter, roles=roles or [])
        self._agent_name = AgentName.MAIN_PLAN


class _FakeState:
    """``AgentState`` 的最小替身。"""

    def __init__(self, *, cur_iter: int, roles: list[str]) -> None:
        from types import SimpleNamespace

        self.reply_context = SimpleNamespace(cur_iter=cur_iter)
        self.reply_id = "fake-reply"
        self.context: list[Any] = []
        self.middle_context: dict[str, Any] = {}
        self._roles = roles


# ---------------------------------------------------------------------------
# 四、构造参数校验
# ---------------------------------------------------------------------------


def test_non_positive_max_chars_is_rejected() -> None:
    """``max_chars < 1`` 在构造时就报错。

    ⚠️ 非正的上限会让**所有**输入都走慢车道，快车道形同虚设 ——
    这是「配置写错导致功能静默失效」，必须在启动时暴露。
    """
    with pytest.raises(ValueError, match="max_chars"):
        LaneRouterMiddleware(max_chars=0)


def test_max_chars_is_honoured() -> None:
    """超过 ``max_chars`` 的输入一律走慢车道。"""
    agent = make_agent(middleware=LaneRouterMiddleware(max_chars=3))
    _, recorded = asyncio.run(run_reply(agent, "规划行程"))

    assert recorded is not None
    assert recorded["lane"] == LaneName.SLOW.value


def test_stage_is_passed_through_to_the_classifier() -> None:
    """记录里带上阶段与意图，供动态 Prompt 使用。"""
    agent = make_agent(middleware=LaneRouterMiddleware())
    _, recorded = asyncio.run(run_reply(agent, "规划行程"))

    assert recorded is not None
    assert recorded["intent"] == Intent.PLAN_TRIP.value
    assert recorded["target_agents"], "路由决策必须带上目标智能体"


def test_route_tool_name_matches_the_contract() -> None:
    """工具名常量与 ``src/tools/route.py`` 里的实现一致。"""
    assert DEFAULT_ROUTE_TOOL == ROUTE_TOOL_NAME


def test_recorded_decision_has_the_expected_shape() -> None:
    """记录下来的键集合是稳定的 —— 下游（动态 Prompt）按名字读它。"""
    agent = make_agent(middleware=LaneRouterMiddleware())
    _, recorded = asyncio.run(run_reply(agent, "规划行程"))

    assert recorded is not None
    assert set(recorded) == {
        "reply_id",
        "lane",
        "intent",
        "matched_rule",
        "target_agents",
        "reason",
        "stage",
    }


def test_target_agents_is_a_plain_list() -> None:
    """``target_agents`` 是普通 list，不是元组或枚举集合。

    ⚠️ 它会被写进 ``middle_context``，而那可能被序列化成 JSON 存进会话。
    元组序列化后会变成 list，读回来与写进去的类型不一致 —— 这种问题只在
    跨进程/跨重启时才暴露，最难查。
    """
    agent = make_agent(middleware=LaneRouterMiddleware())
    _, recorded = asyncio.run(run_reply(agent, "规划行程"))

    assert recorded is not None
    assert isinstance(recorded["target_agents"], list)
    assert all(isinstance(x, str) for x in recorded["target_agents"])


def test_route_tool_input_fields_are_declared() -> None:
    """入参字段常量与路由工具的实际形参一致（断言在工具契约测试里展开）。"""
    assert ROUTE_TOOL_INPUT_FIELDS == ("intent", "matched_rule", "target_agents", "reason")


def test_stage_hint_is_unused_when_no_decision_recorded() -> None:
    """没有历史决策时取阶段返回 ``None``，分类器自行降级。"""
    mw = LaneRouterMiddleware()
    assert mw._stage_of(_FakeAgent()) is None  # noqa: SLF001


# ---------------------------------------------------------------------------
# 五、阶段（stage）的写入与还原
#
# ⚠️ 这一组守的是一个**曾经真实存在的缺口**：``middle_context["stage"]``
# 被 ``lane._stage_of`` 与 ``context._resolve`` 两处读，却没有任何地方写。
# 「读得到但永远是空」不会报任何错，阶段功能就这么静默失效了。
# ---------------------------------------------------------------------------


def _agent_with_recorded_stage(raw: Any) -> _FakeAgent:
    """造一个 middle_context 里已经记了某个阶段值的 agent。"""
    agent = _FakeAgent()
    agent.state.middle_context[ROUTE_DECISION_KEY] = {"stage": raw}
    return agent


def test_stage_is_recorded_as_a_plain_string() -> None:
    """⚠️ 记进 ``middle_context`` 的是 ``stage.value``（字符串），不是枚举对象。

    会话状态会被序列化；枚举存进去、JSON 读回来就成了字符串，
    「写入时是枚举、读出时是字符串」这种不一致只在重启后才暴露，最难查。
    统一在写入侧降级成字符串，读取侧负责还原。
    """
    agent = _FakeAgent()
    mw = LaneRouterMiddleware()
    mw._remember(agent, _decision(), stage=TripStage.COLLECTING)  # noqa: SLF001

    recorded = agent.state.middle_context[ROUTE_DECISION_KEY]
    assert recorded["stage"] == "COLLECTING"
    assert not isinstance(recorded["stage"], TripStage)


def test_missing_stage_is_recorded_as_none_not_a_string() -> None:
    """⚠️ 阶段未知时记 ``None``，**不是** ``"None"`` 字符串。

    记成字符串会让「未知」变成一个看起来像真实阶段的名字，
    下游 ``TripStage("None")`` 会抛 ValueError —— 一个纯粹由写法造成的异常。
    """
    agent = _FakeAgent()
    LaneRouterMiddleware()._remember(agent, _decision(), stage=None)  # noqa: SLF001

    assert agent.state.middle_context[ROUTE_DECISION_KEY]["stage"] is None


def test_stage_of_restores_the_enum() -> None:
    """⚠️ ``_stage_of`` **必须**把字符串还原成枚举。

    分类器的 ``_with_stage`` 会读 ``stage.value``；把裸字符串递过去会抛
    ``AttributeError: 'str' object has no attribute 'value'`` ——
    而这发生在每一次模型调用的路径上，等于整个对话链路挂掉。
    """
    mw = LaneRouterMiddleware()
    assert mw._stage_of(_agent_with_recorded_stage("COLLECTING")) is TripStage.COLLECTING  # noqa: SLF001


def test_stage_of_passes_an_enum_through() -> None:
    """已经是枚举时原样返回（不重复构造）。"""
    mw = LaneRouterMiddleware()
    assert mw._stage_of(_agent_with_recorded_stage(TripStage.DONE)) is TripStage.DONE  # noqa: SLF001


def test_unknown_stage_degrades_to_none() -> None:
    """⚠️ 不认识的阶段名降级成 ``None``，不抛异常。

    旧会话里遗留的阶段值、或将来枚举改名，都可能让这里读到陌生字符串。
    阶段只是**附加信息**，为它中断一次分类不值得。
    """
    mw = LaneRouterMiddleware()
    assert mw._stage_of(_agent_with_recorded_stage("LEGACY_STAGE")) is None  # noqa: SLF001


def test_stage_round_trips_through_a_real_reply() -> None:
    """端到端：写进去的字符串能被取回来还原成枚举。

    把两半拼起来测 —— 单测两边各自正确、接口却对不上，是这类
    「序列化/反序列化」缺口最典型的形态。
    """
    agent = make_agent(middleware=LaneRouterMiddleware())
    mw = LaneRouterMiddleware()
    mw._remember(agent, _decision(), stage=TripStage.CONFIRMING)  # noqa: SLF001

    assert mw._stage_of(agent) is TripStage.CONFIRMING  # noqa: SLF001


def test_stage_reaches_the_classifier_without_crashing() -> None:
    """阶段是字符串时分类器仍然正常工作，且 ``reason`` 里带上阶段。"""
    agent = make_agent(middleware=LaneRouterMiddleware())
    agent.state.middle_context[ROUTE_DECISION_KEY] = {"stage": "COLLECTING"}

    _, recorded = asyncio.run(run_reply(agent, "怎么报销？"))

    assert recorded is not None
    assert recorded["lane"] == LaneName.SLOW.value
    assert "COLLECTING" in recorded["reason"]


def _decision() -> Any:
    """造一个最小可用的路由判定，供 ``_remember`` 的单测使用。"""
    from src.orchestration.classifier import classify

    return classify("规划行程")


def test_trip_stage_is_importable_for_dynamic_prompt() -> None:
    """阶段枚举能被导入 —— 动态 Prompt 依赖它。"""
    assert TripStage.COLLECTING.value == "COLLECTING"




# ---------------------------------------------------------------------------
# 十一、快车道一次性查询的工具收窄
# ---------------------------------------------------------------------------
# 背景（2026-10-03 实测）：问「住宿标准」，前 7 次 4–7 秒直接作答，
# 第 8 次走了建团队 → 拉成员 → 派任务，用了 50.6 秒，答案还是同一句话。
# 提示词里写「别建团队」压不住这种概率性跑偏（同一句话连问 8 次只出现 1 次），
# 所以加一道确定性的闸门：判过快车道之后，编排类工具直接从工具表里摘掉。
class ToolRecordingModel(CountingModel):
    """记录**每次调用拿到的工具名**的 Mock 模型。

    ``CountingModel`` 只数次数，看不到工具表 —— 而「快车道收窄了工具」
    这件事恰恰只体现在工具表上。这里把每次调用的工具名存下来，
    断言就能精确到「第二轮没有 TeamCreate」。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.tool_names: list[list[str]] = []

    async def _call_api(self, *args: Any, **kwargs: Any) -> Any:
        """记下本次调用的工具名，再交给基类。

        ⚠️ ``tools`` 可能走位置参数也可能走关键字（框架内部用哪种取决于
        调用路径），两种都要收 —— 只认关键字的话，记录会一直是空列表，
        而空列表让所有断言都「通过」。
        """
        tools = kwargs.get("tools")
        if tools is None and len(args) > 2:
            tools = args[2]
        self.tool_names.append(
            [(schema.get("function") or {}).get("name", "") for schema in (tools or [])]
        )
        return await super()._call_api(*args, **kwargs)


#: 框架导出、但**不**属于主智能体编排面、因此刻意不收窄的工具名。
#:
#: ``SubmitHandover`` / ``SubmitVerdict`` 是 SOP 子智能体向 leader 交接用的
#: 工具（``app/_tool/_sop_submit.py``），只在子智能体自己的会话里出现，
#: 主智能体的一次性查询根本碰不到它们。收窄它们没有收益，而把它们写进
#: :data:`ORCHESTRATION_TOOLS` 会误导后来人以为主链路会用到。
_KNOWN_UNNARROWED: frozenset[str] = frozenset({"SubmitHandover", "SubmitVerdict"})


def _team_tool(name: str) -> FunctionTool:
    """造一个**名字等于框架内部工具名**的假工具。

    ⚠️ 必须用真名：收窄是按名字比对的，工具随便叫什么都测不出问题。

    ⚠️ 只借**名字**，不复制框架工具的 ``is_read_only``。收窄是纯名字比对，
    这个替身不该把守卫的判据绑到别的属性上 —— 那会让「名单写错名字」
    这一类缺陷被替身的属性顺手掩盖。
    """

    def _placeholder(note: str = "") -> ToolChunk:
        """占位实现。"""
        return ToolChunk(content=[TextBlock(text="ok")])

    _placeholder.__name__ = name
    return FunctionTool(_placeholder, is_read_only=True)


def make_agent_with_team_tools(*, middleware: LaneRouterMiddleware) -> Agent:
    """构造一个工具表里**同时有**框架内部工具与业务工具的 agent。

    真实的会话工具表就是这样：框架挂上 Task 系列、Team 系列，会话配了模型
    时再加上元工具 ``reset_tools``，启动时挂上工作区六件套与 ``ToolStop``，
    业务工具（差旅查询、路由）混在同一个列表里。

    Args:
        middleware (`LaneRouterMiddleware`): 要挂的中间件。

    Returns:
        `Agent`: 挂好中间件、模型会记录工具表的 agent。

    ⚠️ 用 :data:`LANE_HIDDEN_TOOLS`（编排 + 工具管理 + 进程控制 + 工作区）
    而不是 :data:`ORCHESTRATION_TOOLS`：后者只是前者的一个真子集，
    用子集搭出来的表**测不出**「元工具有没有被收窄」——
    而那个恰恰是后补的一处判据。表要照着**线上真实的那张**搭，
    每一类工具都留一个代表，否则漏收窄的那一类在单测里根本不存在。
    """
    tools: list[Any] = [
        FunctionTool(ping, is_read_only=True),
        FunctionTool(aligo_route_intent, is_read_only=True),
    ]
    for name in sorted(LANE_HIDDEN_TOOLS):
        tools.append(_team_tool(name))
    return Agent(
        name="main_plan",
        system_prompt="你是差旅助手。",
        model=ToolRecordingModel(),
        toolkit=Toolkit(tools=tools),
        middlewares=[middleware],
    )


async def run_and_collect_tools(agent: Agent, text: str) -> list[list[str]]:
    """跑一轮回复，返回「每次模型调用看到的工具名」。

    Args:
        agent (`Agent`): 目标 agent。
        text (`str`): 用户输入。

    Returns:
        `list[list[str]]`: 每次模型调用拿到的工具名列表。

    ⚠️ 快车道下这个列表**只有一条**：第一轮的模型调用被
    ``LaneRouterMiddleware`` 短路掉了（它直接合成 ``ChatResponse``），
    所以 ``_call_api`` 没被调用、也就没记录。想拿「收窄之前」的工具表做
    基准，只能用 :func:`un_narrowed_table`，不能指望 ``calls[0]``。
    """
    async for _ in agent.reply_stream(
        inputs=Msg(name="user", role="user", content=[TextBlock(text=text)]),
    ):
        pass
    return agent.model.tool_names  # type: ignore[attr-defined]


def un_narrowed_table(text: str = "查询政策") -> set[str]:
    """取一份**未收窄**的工具表，作为收窄用例的基准。

    做法：同款 agent、同一句话，只把中间件关掉。这样「基准」与「实测」
    之间只差一个开关，差异一定是收窄造成的 —— 而不是替身搭得不一样。

    Args:
        text (`str`): 用户输入。

    Returns:
        `set[str]`: 最后一次模型调用看到的工具名。
    """
    agent = make_agent_with_team_tools(
        middleware=LaneRouterMiddleware(enabled=False),
    )
    return set(asyncio.run(run_and_collect_tools(agent, text))[-1])


def test_fast_lane_policy_query_narrows_orchestration_tools() -> None:
    """★ 「查询政策」走快车道后，交给模型的工具表里不该再有编排工具。

    快车道只省掉**第一轮**模型调用；第二轮（真正写答案的那轮）仍然要
    调模型。那一轮就是第 8 次实测里跑偏的地方 —— 所以收窄必须落在
    第二轮上，本用例断言的就是它。
    """
    agent = make_agent_with_team_tools(middleware=LaneRouterMiddleware())

    calls = asyncio.run(run_and_collect_tools(agent, "查询政策"))

    assert calls, "一次模型调用都没发生，用例本身失效了"
    # ⚠️ 先钉住「这张表本来是有的」：同款 agent、同一句话、只关掉中间件，
    # 工具表里必须有内部工具。少了这条，下面那条断言在「替身搭错了、
    # 表里压根没有内部工具」时也会通过 —— 而那是假绿。
    assert INTERNAL_TOOLS & un_narrowed_table(), (
        "关掉中间件时工具表里也没有内部工具 —— 替身没搭好，用例失去意义。"
    )
    answer_round = calls[-1]
    leaked = INTERNAL_TOOLS & set(answer_round)
    assert not leaked, f"快车道政策问答里仍能调用 {sorted(leaked)}：{answer_round}"
    assert "aligo_route_intent" in answer_round, "业务工具被误伤了"


def test_fast_lane_also_narrows_the_meta_tool() -> None:
    """★ 工具管理元工具 ``reset_tools`` 也要一起摘掉。

    ⚠️ 它与编排工具**不是**一回事，注入条件也不同：编排工具是「有团队/
    规划能力就有」，元工具的条件是「**工具组多于一个**」
    （``agentscope/tool/_toolkit.py:502-510``，会话配了模型时会出现 ``schedule_tools``
    组 ⇒ 线上确实在）。

    留着它的后果与收窄的初衷相反：模型可以「关掉一组工具、再开另一组、
    再来一轮」，把一次调用能给的结论拖成多轮 —— 而用户只是问了一句
    「住宿标准是多少」。
    """
    agent = make_agent_with_team_tools(middleware=LaneRouterMiddleware())

    calls = asyncio.run(run_and_collect_tools(agent, "查询政策"))

    assert calls
    assert TOOL_MANAGEMENT_TOOLS & un_narrowed_table(), (
        "关掉中间件时工具表里也没有元工具 —— 替身没搭好，用例失去意义。"
    )
    meta_leaked = TOOL_MANAGEMENT_TOOLS & set(calls[-1])
    assert not meta_leaked, (
        f"快车道一次性查询里仍能调用 {sorted(meta_leaked)}：{calls[-1]}"
    )


def test_fast_lane_also_narrows_process_control_tools() -> None:
    """★ 进程控制工具 ``ToolStop`` 也要一起摘掉。

    ⚠️ 它与编排工具、元工具的注入条件**三者各不相同**：编排工具看角色，
    元工具看工具组数量，而 ``ToolStop`` 由
    ``BackgroundTaskManager.list_tools()`` **无条件**挂载
    （``app/_service/_toolkit.py``）。所以它自成一类，不能靠另外两份
    名单顺带覆盖。

    留着它的后果不是延迟而是**答非所问**：用户问一句「住宿标准」，
    模型手里却有个「停掉后台任务」的按钮，就可能先停一下任务、
    再写一句「我先把后台那个还在跑的任务停掉」当答复 —— 而这一轮
    的文字在守卫那边同样是过程独白（反面用例见
    ``test_orchestration_reply_guard.py`` 的同名判据）。
    """
    agent = make_agent_with_team_tools(middleware=LaneRouterMiddleware())

    calls = asyncio.run(run_and_collect_tools(agent, "查询政策"))

    assert calls
    assert PROCESS_CONTROL_TOOLS & un_narrowed_table(), (
        "关掉中间件时工具表里也没有进程控制工具 —— 替身没搭好，用例失去意义。"
    )
    leaked = PROCESS_CONTROL_TOOLS & set(calls[-1])
    assert not leaked, (
        f"快车道一次性查询里仍能调用 {sorted(leaked)}：{calls[-1]}"
    )


def test_fast_lane_also_narrows_the_workspace_tools() -> None:
    """★ 工作区六件套（Bash/Edit/Glob/Grep/Read/Write）也要摘掉。

    ⚠️ 这一条**只**管收窄，不管守卫：工作区工具是真的写工具，
    ``reply_guard._round_needs_recitation`` 对它们**刻意**保持
    「保留本轮文字」的保守判断。两份名单在这里分岔，是有意的 ——
    收窄看的是「一次性业务查询用不用得上文件与进程」，
    守卫看的是「这一轮有没有用户可见的写操作」。把两份名单合并
    （或反过来，让守卫也照这份名单剥）都会打破其中一处的理由。

    ⚠️ 症状不对称，所以宁可摘掉：留在表里，模型有概率不查政策、
    先去 ``Bash`` 敲一条命令看看 —— 用户的等待时间里什么都没得到。
    """
    agent = make_agent_with_team_tools(middleware=LaneRouterMiddleware())

    calls = asyncio.run(run_and_collect_tools(agent, "查询政策"))

    assert calls
    assert WORKSPACE_TOOLS & un_narrowed_table(), (
        "关掉中间件时工具表里也没有工作区工具 —— 替身没搭好，用例失去意义。"
    )
    leaked = WORKSPACE_TOOLS & set(calls[-1])
    assert not leaked, (
        f"快车道一次性查询里仍能调用 {sorted(leaked)}：{calls[-1]}"
    )


def test_narrowing_never_empties_the_tool_table() -> None:
    """★ 收窄后一个工具都不剩时，必须放弃收窄而不是交出空表。

    ⚠️ 这是防御性分支，正常装配到不了（``basic`` 组里一定有业务工具）。
    但真到了就是「模型这一轮什么工具都看不到」——症状是它突然答不出
    「住宿标准」，而日志里只有一条看起来正常的「收窄」记录。
    宁可退回不收窄，也不要制造这种状态。
    """
    middleware = LaneRouterMiddleware()
    fake = _FakeAgent()
    fake.state.middle_context[ROUTE_DECISION_KEY] = {
        "reply_id": fake.state.reply_id,
        "lane": LaneName.FAST.value,
        "intent": Intent.QUERY_POLICY.value,
    }
    kwargs: dict[str, Any] = {
        "tools": [
            {"type": "function", "function": {"name": name}}
            for name in sorted(INTERNAL_TOOLS)
        ],
    }

    assert middleware._narrowed_kwargs(fake, kwargs) is kwargs, (  # noqa: SLF001
        "工具表被摘空了仍然交了出去 —— 模型这一轮将看不到任何工具。"
    )

    # 对照：混进一个业务工具时**应该**收窄（否则上面那条断言在
    # 「收窄整个失效」时也会通过）。
    with_business = {
        "tools": [
            *kwargs["tools"],
            {"type": "function", "function": {"name": "check_travel_policy"}},
        ],
    }
    narrowed = middleware._narrowed_kwargs(fake, with_business)  # noqa: SLF001
    assert [s["function"]["name"] for s in narrowed["tools"]] == [
        "check_travel_policy"
    ]


@pytest.mark.parametrize(
    "text",
    [
        # 慢车道：规则没命中，交给意图识别智能体理解。
        "帮我规划下周去北京的行程",
        # 下面是**快车道但不在收窄名单里**的意图。它们才是这份名单的边界：
        # 慢车道那条由 `lane` 守卫单独兜住，测不出 `intent` 守卫的作用。
        "规划行程",  # PLAN_TRIP
        "提申请",  # APPLY_APPROVAL
        "改签",  # MODIFY_TRIP
        "取消",  # CANCEL
    ],
)
def test_intents_outside_the_narrowing_list_keep_the_full_tool_table(
    text: str,
) -> None:
    """★ 收窄只针对一次性查询，其余意图的工具表必须原样保留。

    ⚠️ 这五条输入覆盖了**两道守卫各自的失效面**，缺一不可：

    - 「帮我规划下周去北京的行程」走慢车道，只有 ``lane`` 守卫兜着；
    - 「规划行程」等四条**是快车道**，``lane`` 守卫对它无效 ——
      挡住它们的是 ``NARROWED_INTENTS`` 名单。删掉名单检查，这四条会红。

    为什么这很重要：把「规划行程」也收窄了，用户就再也建不了团队、
    派不了任务 —— 而规划行程**恰恰**是最需要多线并行的场景。
    收窄是为了治一个概率性延迟，不是为了让主功能消失。
    """
    agent = make_agent_with_team_tools(middleware=LaneRouterMiddleware())

    calls = asyncio.run(run_and_collect_tools(agent, text))

    assert calls, "一次模型调用都没发生，用例本身失效了"
    assert ORCHESTRATION_TOOLS & set(calls[-1]), (
        f"{text!r} 的编排工具被误伤：{calls[-1]}"
    )


def test_stale_decision_from_another_reply_is_ignored() -> None:
    """★ 判定记录属于**别的**回复时，收窄必须整个不生效。

    ⚠️ 为什么这条是**单元**用例而不是端到端用例：端到端跑不出这个状态。
    正常流程里，``_decide`` 在第一轮就会用**本次** ``reply_id`` 覆盖记录
    （``lane.py`` 的 ``_remember``），所以走到 :meth:`_narrowed_kwargs` 时
    记录总是新鲜的 —— 我原本写的端到端版本因此**杀不掉**「删掉 reply_id
    比对」这个变异体（实测 GREEN-BAD）。真正会出现陈旧记录的是
    ``_decide`` 直接返回 ``None`` 的那几条路径（本次回复没有用户消息、
    ``cur_iter`` 非 0 但记录还停在上一次回复），而那几条路径不好在单测里
    稳定复现。

    所以这里直接构造 `middle_context`：这是该守卫**唯一**的判据，
    直接测它，比绕一大圈更有说服力。
    """
    middleware = LaneRouterMiddleware()
    fake = _FakeAgent()
    stale = {
        "reply_id": "★上一次回复的 id★",
        "lane": LaneName.FAST.value,
        "intent": Intent.QUERY_POLICY.value,
    }
    fake.state.middle_context[ROUTE_DECISION_KEY] = stale
    # ⚠️ 表里必须**同时**有内部工具与业务工具：只剩内部工具时收窄会撞上
    # 「不许摘空」的保险丝（见 :func:`test_narrowing_never_empties_the_tool_table`），
    # 那条路径同样返回原对象 —— 断言就分不清「因为陈旧而放过」与
    # 「因为会摘空而放过」了。
    kwargs: dict[str, Any] = {
        "tools": [
            {"type": "function", "function": {"name": "TeamCreate"}},
            {"type": "function", "function": {"name": "check_travel_policy"}},
        ],
    }

    assert middleware._narrowed_kwargs(fake, kwargs) is kwargs, (  # noqa: SLF001
        "陈旧记录（reply_id 对不上）仍然收窄了工具表 —— "
        "用户上一句查政策，这一句的编队能力就被剥夺了。"
    )

    # 对照：同一个 fake，把 reply_id 换成当前的就**应该**收窄。
    # 没有这一步，上面那条断言在「收窄根本没实现」时也会通过。
    fresh = {**stale, "reply_id": fake.state.reply_id}
    fake.state.middle_context[ROUTE_DECISION_KEY] = fresh
    assert [
        schema["function"]["name"]
        for schema in middleware._narrowed_kwargs(fake, kwargs)["tools"]  # noqa: SLF001
    ] == ["check_travel_policy"]


def test_narrowing_does_not_leak_into_the_next_reply() -> None:
    """★ 端到端回归：上一轮查政策，这一轮规划行程，工具表不能少。

    ⚠️ 这条**不**是 reply_id 守卫的守卫者（那是
    :func:`test_stale_decision_from_another_reply_is_ignored` 的活）——
    它守的是更外层的性质：无论内部怎么实现，用户都不该因为
    「上一句问了政策」而在这句话里失去编队能力。
    """
    agent = make_agent_with_team_tools(middleware=LaneRouterMiddleware())

    asyncio.run(run_and_collect_tools(agent, "查询政策"))
    second = asyncio.run(run_and_collect_tools(agent, "帮我规划下周去北京的行程"))

    assert second, "第二次回复没有发生模型调用"
    assert ORCHESTRATION_TOOLS & set(second[-1]), (
        f"上一轮的政策问答把这一轮的编排工具也摘掉了：{second[-1]}"
    )


def test_narrowing_is_transparent_when_disabled() -> None:
    """★ 关掉中间件时工具表原样透传。"""
    agent = make_agent_with_team_tools(
        middleware=LaneRouterMiddleware(enabled=False)
    )

    calls = asyncio.run(run_and_collect_tools(agent, "查询政策"))

    assert calls
    assert ORCHESTRATION_TOOLS & set(calls[-1]), (
        f"关掉中间件后工具表仍被改动：{calls[-1]}"
    )


def test_narrowing_survives_missing_tools_key() -> None:
    """★ 入参里没有 ``tools`` 时不能抛异常。

    ``on_model_call`` 的入参由框架拼装，历史版本或测试替身可能不带
    ``tools``。收窄是个纯优化，它绝不能成为「整条对话链路挂掉」的原因。
    """
    middleware = LaneRouterMiddleware()
    fake = _FakeAgent()
    fake.state.middle_context[ROUTE_DECISION_KEY] = {
        "reply_id": fake.state.reply_id,
        "lane": LaneName.FAST.value,
        "intent": Intent.QUERY_POLICY.value,
    }

    assert middleware._narrowed_kwargs(fake, {}) == {}  # noqa: SLF001
    assert middleware._narrowed_kwargs(  # noqa: SLF001
        fake, {"tools": []}
    ) == {"tools": []}


def test_orchestration_tool_names_match_the_framework() -> None:
    """★ 收窄表里的名字必须与框架真实注入的工具名逐字一致。

    ⚠️ 写错一个名字的症状是「这个工具照样出现在模型面前」，**没有任何报错**。
    这条断言把静默失效变成红灯：不联网、不建团队，只读框架的类定义。

    两组名字来自框架里两个不同的地方，都要覆盖：

    - 团队工具在 ``agentscope.app._tool``（``agentscope/app/_service/_toolkit.py:184-198``
      按会话角色挂载）；
    - 规划工具在 ``agentscope.tool._task``（同文件 ``:143`` **无条件**挂载）。
    """
    import agentscope.app._tool as team_tool_module
    import agentscope.tool._task as task_tool_module

    framework_names: set[str] = set()
    for module, expected_min in (
        (team_tool_module, 5),
        (task_tool_module, 4),
    ):
        found_here = {
            cls.name
            for cls in (getattr(module, exported, None) for exported in module.__all__)
            # 只收**工具类**：``DEFAULT_SUB_AGENT_TEMPLATE`` 这类常量没有
            # ``name`` 字段，混进来只会让断言变得不可读。工具的判据是类且
            # 带字符串 ``name``。
            if isinstance(cls, type) and isinstance(getattr(cls, "name", None), str)
        }
        # ⚠️ 必须断言「这一组至少收到几个」，否则框架把 ``__all__`` 清空、
        # 或把工具改成别的形态时，``framework_names`` 会是空集，
        # 而空集让下面两条断言**全都通过** —— 用例从「守卫」变成「摆设」。
        assert len(found_here) >= expected_min, (
            f"{module.__name__} 只解析出 {sorted(found_here)}，"
            f"少于预期的 {expected_min} 个工具 —— 框架的导出结构变了。"
        )
        framework_names |= found_here

    stale = ORCHESTRATION_TOOLS - framework_names
    assert not stale, (
        f"收窄表里的 {sorted(stale)} 在框架里找不到同名工具 —— "
        "框架可能改名了，收窄对这些工具已经静默失效。"
    )

    # 反向：框架的团队/规划工具必须全在收窄表里，漏一个就有一条慢路径漏网。
    # ⚠️ ``TeamSay`` 对 worker 角色是唯一工具，但它同样属于编排类
    # （它会让 leader 之外的会话参与协作），一并收窄。
    missing = framework_names - ORCHESTRATION_TOOLS - _KNOWN_UNNARROWED
    assert not missing, (
        f"框架的编排工具 {sorted(missing)} 没有被收窄 —— 它们仍会在快车道"
        "一次性查询里露出来。确认无害后加进 _KNOWN_UNNARROWED，否则加进"
        " ORCHESTRATION_TOOLS。"
    )


def test_tool_management_names_match_the_framework() -> None:
    """★ 元工具的名字必须与**真实装配出来的** ``Toolkit`` 逐字一致。

    ⚠️ 名字来源与编排工具完全不同，所以没法并进上一条用例：编排工具是
    在 ``app/_tool`` / ``tool/_task`` 里**导出**的类，而元工具是
    ``Toolkit`` 在装配期**注入**的（``agentscope/tool/_toolkit.py:156``），
    没有任何模块把它写进 ``__all__``。只能问一个造出来的 ``Toolkit``。

    ⚠️ 顺带把「注入条件」也钉住 —— 这正是它与编排工具不该混成一份名单的
    理由：**单组时它不在，多一组时才出现**。哪天框架改成无条件注入，
    这条用例会红，而不是让收窄悄悄失灵。
    """
    import asyncio as _asyncio

    from agentscope.tool import ToolGroup

    def _noop() -> ToolChunk:
        """占位工具。"""
        return ToolChunk(content=[TextBlock(text="ok")])

    single = Toolkit(tools=[FunctionTool(_noop, is_read_only=True)])
    single_names = {
        (schema.get("function") or {}).get("name")
        for schema in _asyncio.run(single.get_tool_schemas(["basic"]))
    }
    assert not (TOOL_MANAGEMENT_TOOLS & single_names), (
        f"只有 basic 一个组时元工具就该被注入：{sorted(single_names)} —— "
        "注入条件变了，收窄与守卫的前提都要重看。"
    )

    two_groups = Toolkit(
        tools=[FunctionTool(_noop, is_read_only=True)],
        tool_groups=[ToolGroup(name="schedule_tools", description="占位", tools=[])],
    )
    two_names = {
        (schema.get("function") or {}).get("name")
        for schema in _asyncio.run(two_groups.get_tool_schemas(["basic", "schedule_tools"]))
    }

    missing = TOOL_MANAGEMENT_TOOLS - two_names
    assert not missing, (
        f"元工具名单里的 {sorted(missing)} 在真实 Toolkit 里找不到 —— "
        f"框架改名了，收窄对它已经静默失效。实际注入的是 {sorted(two_names)}。"
    )


def test_process_control_names_match_the_framework() -> None:
    """★ 进程控制工具名必须与**真实 ``list_tools`` 吐出来的**逐字一致。

    ⚠️ 名字来源是第三处，所以不能并进上面两条：编排工具在模块 ``__all__``
    里，元工具由 ``Toolkit`` 装配期注入，而 ``ToolStop`` 是
    ``BackgroundTaskManager.list_tools(session_id)`` 每次现造的
    （``agentscope/app/_manager/_background_task_manager.py:362-374``）。

    ⚠️ 断言走**真实的** :meth:`list_tools` 而不是读类属性 ``ToolStop.name``：
    前者才是注入路径，将来框架改成「返回别的类」或「多加一个工具」，
    前者会红，后者不会。管理器构造只存一个 bus 引用，替身用 ``object()``
    就够（``list_tools`` 不碰它）。
    """
    from agentscope.app._manager._background_task_manager import (
        BackgroundTaskManager,
    )

    manager = BackgroundTaskManager(object())  # type: ignore[arg-type]
    tools = asyncio.run(manager.list_tools("session-for-contract-test"))
    injected = {tool.name for tool in tools}

    # ⚠️ 先钉住「这条路真的产出工具」：框架把 ``list_tools`` 改成返回空表时，
    # 下面那条断言会**因为空集而通过** —— 用例从守卫变成摆设。
    assert injected, "list_tools 一个工具都没返回 —— 框架的注入路径变了。"

    missing = PROCESS_CONTROL_TOOLS - injected
    assert not missing, (
        f"进程控制名单里的 {sorted(missing)} 在 list_tools 里找不到 —— "
        f"框架改名了，收窄与守卫对它都已静默失效。实际注入的是 {sorted(injected)}。"
    )


def test_workspace_tool_names_match_the_framework() -> None:
    """★ 工作区工具名必须与 ``agentscope.tool`` 真实导出的类逐字一致。

    ⚠️ 名单只给收窄用（守卫刻意不用，理由见
    :data:`src.orchestration.lane.WORKSPACE_TOOLS`），所以这里只验名字。

    ⚠️ 断言两条：名字都能导出，且 ``__all__`` 里**恰好**是这六个 ——
    框架哪天加了第七个工作区工具，这里会红，提醒我们决定它该不该被收窄；
    加一句「至少六个」是本条用例唯一挡不住的静默失效。
    """
    import agentscope.tool as tool_module

    exported = set(getattr(tool_module, "__all__", []))
    missing = WORKSPACE_TOOLS - exported
    assert not missing, (
        f"工作区名单里的 {sorted(missing)} 在 agentscope.tool 里没有导出 —— "
        "框架改名了，收窄对它们已静默失效。"
    )

    # ⚠️ 六个名字同时要是**类**，且自带同名 ``name`` 字段：只比对 ``__all__``
    # 里的字符串，无法发现「类改名、导出别名保留」这种半改法。
    for name in sorted(WORKSPACE_TOOLS):
        cls = getattr(tool_module, name, None)
        assert isinstance(cls, type), f"{name} 不再是类：{cls!r}"
        assert getattr(cls, "name", None) == name, (
            f"{name} 类的 name 字段是 {getattr(cls, 'name', None)!r} —— "
            "与收窄表比对的那个名字对不上了。"
        )


def test_workspace_tools_are_deliberately_absent_from_internal_tools() -> None:
    """★ 两份名单的分岔必须钉住：工作区工具**不**进 :data:`INTERNAL_TOOLS`。

    ⚠️ 这是个**不能合并**的断言，而不是冗余：把 ``WORKSPACE_TOOLS`` 并进
    ``INTERNAL_TOOLS`` 会让 ``reply_guard._round_needs_recitation`` 对
    ``Bash`` / ``Write`` / ``Edit`` 也返回 False（剥掉那一轮的文字）——
    而它们是货真价实的写工具，那一轮的文字很可能是「文件已生成」这类
    用户可见的交代。守卫的取舍一贯是「宁可留一段独白，不可删一段正文」。

    反过来把 ``INTERNAL_TOOLS`` 并进 ``WORKSPACE_TOOLS`` 同样不行：
    收窄表会只剩内部工具那一份，工作区六件套重新露给模型。

    所以两份名单**只能**是「并集关系且各管各的读者」，
    这条用例把关系写成可执行的形式。
    """
    assert not (WORKSPACE_TOOLS & INTERNAL_TOOLS), (
        "工作区工具混进了 INTERNAL_TOOLS —— 守卫会开始剥真实写操作的那轮文字。"
    )
    assert LANE_HIDDEN_TOOLS == INTERNAL_TOOLS | WORKSPACE_TOOLS, (
        "收窄表不再等于两份名单的并集 —— 有一份名单被改了而另一处没跟上。"
    )
