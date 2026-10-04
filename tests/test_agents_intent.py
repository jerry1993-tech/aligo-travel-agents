# -*- coding: utf-8 -*-
"""意图识别智能体（``src/agents/intent.py``）的测试。

核心是 **P3 的验收项之一**：
:func:`test_multi_intent_input_is_split_into_a_structured_breakdown`
—— 「多意图输入返回结构化拆解」。

═══ 为什么这些用例要**真的构造 Agent 并跑 reply** ═══

结构化输出这条链路上有三层我们控制不了的东西：

1. 框架会把 schema 注册成一个叫 ``GenerateStructuredOutput`` 的**工具**，
   而不是用 JSON mode（``agent/_agent.py:1126-1132``）。
2. 交付物在 ``Msg.structured_output``（**dict**），不在消息正文里
   （``agent/_agent.py:3568-3577``）。
3. 校验失败**不抛异常**，而是变成一个 ``state=ERROR`` 的工具结果喂回给
   模型，由模型自己重试；模型一直交不出来时以 ``EXCEED_MAX_ITERS`` 结束，
   ``structured_output`` 是 ``None``。

这三条每一条都可以「按我以为的样子」写出来，然后在一个假模型上跑通 ——
而真实的框架行为不同。所以本文件用**真 Agent + 可控模型**：
模型由我们决定返回什么，其余全走框架真实路径。

═══ 模型是怎么伪造结构化输出的 ═══

直接发一个名为 ``GenerateStructuredOutput`` 的 ``ToolCallBlock``，
``input`` 是符合 schema 的 JSON 串。框架会拿它去调那个内置工具，
工具校验通过后把结果存进 ``structured_output``。

⚠️ 不能用 ``MockChatModel`` 的 ``#mock-tool:`` 指令 —— 那条路要求
指令出现在**最后一条用户消息**里，而意图识别的输入是我们自己拼的
（可能会加背景前缀），且指令会污染「模型看到的文本」这件事，
让「输入原样传给模型」这类断言没法写。
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from agentscope.message import ToolCallBlock
from agentscope.model import ChatResponse

from src.agents.intent import (
    INTENT_AGENT_NAME,
    IntentRecognizer,
    build_intent_recognizer,
)
from src.agents.prompts import PROMPTS
from src.domain import AgentName, Intent
from src.llm.mock import MockChatModel

#: 框架注册结构化输出工具时用的名字。
#:
#: ⚠️ 从框架源码核实（``agent/_structured_output_tool.py:45``），不是猜的。
#: 写错它的后果是「模型永远交不出结构化结果」—— 而这条链路上
#: **没有任何异常**，只是每次识别都降级成追问。
STRUCTURED_TOOL_NAME = "GenerateStructuredOutput"


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------
def make_payload(
    *intents: tuple[str, float],
    reasoning: str = "用户提到了机票和住宿标准。",
    rewritten_query: str = "预订下周三去北京的机票并查询住宿报销标准",
    needs_clarification: bool = False,
    clarification_question: str = "",
) -> dict[str, Any]:
    """拼一份符合 :class:`IntentRecognitionResult` 的载荷。

    Args:
        *intents (tuple[str, float]): ``(意图值, 置信度)`` 对。
        reasoning (str): 推理文本。
        rewritten_query (str): 改写后的查询。
        needs_clarification (bool): 是否要求追问。
        clarification_question (str): 追问内容。

    Returns:
        `dict[str, Any]`: 可直接 ``json.dumps`` 的载荷。
    """
    return {
        "reasoning": reasoning,
        "rewritten_query": rewritten_query,
        "intents": [
            {
                "intent": value,
                "confidence": confidence,
                "slots": {"destination": "北京"},
                "reason": f"判定依据：{value}",
            }
            for value, confidence in intents
        ],
        "needs_clarification": needs_clarification,
        "clarification_question": clarification_question,
    }


class StructuredModel(MockChatModel):
    """第 1 次调用交付一份结构化结果，之后退回正常文本。

    ⚠️ **只交付一次**。每次都交付同一个 ``ToolCallBlock`` 会让框架
    认为「模型在反复调同一个工具」，ReAct 转满 ``max_iters``（50）——
    用例会从 0.1 秒变成几十秒。第 2 次退回正常文本，是因为框架在
    结构化结果**已经满足**之后不会再要求它，正常情况下根本不会有第 2 次调用；
    真有的话说明链路上出了问题，让它以文本结束反而更容易定位。

    ⚠️ 继承 ``MockChatModel`` 而**不是**某个会计数的子类：计数在这里
    由本类自己做（``calls``），继承另一个计数器会让计数发生两次。

    Attributes:
        calls (`int`): 模型被真实调用的次数。
        seen_tools (`list[str]`): 最近一次调用时模型**看得见**的工具名。
    """

    def __init__(self, payload: dict[str, Any] | None, **kwargs: Any) -> None:
        """初始化。

        Args:
            payload (`dict[str, Any] | None`): 要交付的结构化结果；
                ``None`` 表示「什么都不交付」，用来测降级路径。
            **kwargs: 透传给 ``MockChatModel``。
        """
        super().__init__(**kwargs)
        self.calls = 0
        self.seen_tools: list[str] = []
        self._payload = payload

    async def _call_api(self, *args: Any, **kwargs: Any) -> Any:
        """见类文档。"""
        self.calls += 1
        self.seen_tools = _tool_names(kwargs.get("tools"))

        if self.calls == 1 and self._payload is not None:
            return ChatResponse(
                content=[
                    ToolCallBlock(
                        id="structured-1",
                        name=STRUCTURED_TOOL_NAME,
                        input=json.dumps(self._payload, ensure_ascii=False),
                    ),
                ],
                is_last=True,
            )
        return await super()._call_api(*args, **kwargs)


class AlwaysStructuredModel(StructuredModel):
    """**每一次**调用都交付同一份结构化结果。

    ⚠️ 与 :class:`StructuredModel` 的差别只有「交不交第二次」。
    需要它的场景是「一个模型服务多次识别」（共享模型的性质），
    用只交付一次的替身会测不到那件事。
    """

    async def _call_api(self, *args: Any, **kwargs: Any) -> Any:
        """每次都交付。"""
        self.calls += 1
        self.seen_tools = _tool_names(kwargs.get("tools"))
        if self._payload is not None:
            return ChatResponse(
                content=[
                    ToolCallBlock(
                        id=f"structured-{self.calls}",
                        name=STRUCTURED_TOOL_NAME,
                        input=json.dumps(self._payload, ensure_ascii=False),
                    ),
                ],
                is_last=True,
            )
        return await MockChatModel._call_api(self, *args, **kwargs)


class ExplodingModel(MockChatModel):
    """一调用就抛异常，用来测「模型不可用」这条路。

    Attributes:
        calls (`int`): 调用次数。
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化。"""
        super().__init__(**kwargs)
        self.calls = 0

    async def _call_api(self, *args: Any, **kwargs: Any) -> Any:
        """抛一个网络层常见的异常。"""
        self.calls += 1
        raise ConnectionError("模型服务不可用")


def _tool_names(tools: Any) -> list[str]:
    """从模型看到的工具列表里取出工具名。

    ⚠️ 工具 schema 的形态由框架决定（可能是 dict 也可能是对象），
    两种都取一遍。写死一种的话，框架换个形态就会让断言**静默地**
    变成「工具列表为空」—— 而那正好会让「没有业务工具」这条断言假绿。

    Args:
        tools (`Any`): 模型调用参数里的 ``tools``。

    Returns:
        `list[str]`: 工具名列表。
    """
    names: list[str] = []
    for tool in tools or []:
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict) and function.get("name"):
                names.append(str(function["name"]))
            elif tool.get("name"):
                names.append(str(tool["name"]))
        else:
            name = getattr(tool, "name", None)
            if name:
                names.append(str(name))
    return names


def build(**kwargs: Any) -> IntentRecognizer:
    """构造一个识别器，参数取测试默认值。

    Args:
        **kwargs: 覆盖默认参数。

    Returns:
        `IntentRecognizer`: 识别器。
    """
    params: dict[str, Any] = {
        "model": StructuredModel(make_payload(("PLAN_TRIP", 0.9))),
        "max_intents": 5,
        "confidence_threshold": 0.6,
    }
    params.update(kwargs)
    return IntentRecognizer(**params)


# ---------------------------------------------------------------------------
# 一、P3 验收：多意图输入 → 结构化拆解
# ---------------------------------------------------------------------------
async def test_multi_intent_input_is_split_into_a_structured_breakdown() -> None:
    """★★ **P3 验收断言**：一句话里有两件事时，返回两条意图。

    输入照着真实场景写：「帮我订下周三去北京的票，顺便看看住宿标准」——
    这是「规划行程」与「政策问答」两件事。博客第 127 行把「多意图识别和
    分类」列为意图识别智能体的第一项职责。

    ⚠️ 断言的是 ``intents`` 有**两条且内容正确**，而不是「返回值非空」。
    后者在一个只会返回单意图的实现上同样通过 —— 而单意图正是这个功能
    要解决的问题：用户说完两件事，系统只办了一件，且**没有任何提示**。
    """
    model = StructuredModel(
        make_payload(("PLAN_TRIP", 0.93), ("QUERY_POLICY", 0.72)),
    )
    recognizer = build(model=model)

    result = await recognizer.recognize("帮我订下周三去北京的票，顺便看看住宿标准")

    assert len(result.intents) == 2, f"多意图输入只拆出 {len(result.intents)} 条"
    assert {item.intent for item in result.intents} == {
        Intent.PLAN_TRIP,
        Intent.QUERY_POLICY,
    }
    assert result.needs_clarification is False, "识别成功却要求追问"


async def test_intents_are_reordered_by_confidence() -> None:
    """★ 置信度最高的意图排在**第一个**。

    ⚠️ 模型给的是 ``QUERY_POLICY(0.72)`` 在前、``PLAN_TRIP(0.93)`` 在后，
    而我们要的是按置信度降序。

    提示词确实要求模型排序，但**控制流不得依赖模型遵守格式约定** ——
    它没排的话，下游按位置取就会稳定地取到错的那个，而症状看起来像
    「模型判断错了」，排查方向会被完全带偏（见
    :meth:`src.domain.schemas.IntentRecognitionResult.top_intent` 的说明）。
    这里主动重排，让位置不管用。
    """
    model = StructuredModel(
        make_payload(("QUERY_POLICY", 0.72), ("PLAN_TRIP", 0.93)),
    )
    result = await build(model=model).recognize("订票并查标准")

    assert result.intents[0].intent is Intent.PLAN_TRIP
    assert result.top_intent() is Intent.PLAN_TRIP


async def test_rewritten_query_and_reasoning_are_carried_through() -> None:
    """★ 改写后的查询与推理过程要原样带出来。

    ⚠️ 这两个字段不是装饰：``rewritten_query`` 是下游工具与检索**实际使用**
    的输入（博客第 131 行），``reasoning`` 是「显示推理」的数据源。
    在识别器这一层丢掉它们，后面的功能就没有数据可用了。
    """
    model = StructuredModel(
        make_payload(
            ("PLAN_TRIP", 0.9),
            reasoning="用户说了订票，也提到了住宿标准。",
            rewritten_query="预订下周三北京机票；查询住宿报销标准",
        ),
    )
    result = await build(model=model).recognize("订票，顺便看看住宿标准")

    assert result.reasoning == "用户说了订票，也提到了住宿标准。"
    assert result.rewritten_query == "预订下周三北京机票；查询住宿报销标准"


async def test_every_intent_carries_slots_and_a_reason() -> None:
    """⚠️ 每条意图都要带要素与判定依据。

    ``slots`` 是要素抽取的结果（下游拿它填 ``TravelRequest``），
    ``reason`` 是给用户看的依据。模型可能只填其中一部分 —— 那两个字段
    在 schema 里都有默认值，缺了不会报错，只会让下游拿到空的。
    """
    model = StructuredModel(make_payload(("PLAN_TRIP", 0.9)))
    result = await build(model=model).recognize("订票")

    decision = result.intents[0]
    assert decision.slots.get("destination") == "北京"
    assert decision.reason


# ---------------------------------------------------------------------------
# 二、框架契约：结构化输出走的是工具调用
# ---------------------------------------------------------------------------
async def test_structured_output_arrives_via_a_tool_call() -> None:
    """★★ 结构化结果是通过调用 ``GenerateStructuredOutput`` 交付的。

    这条**钉住框架的实现方式**，因为它是本模块多条设计的依据：
    模型不需要 JSON mode（只需要会发工具调用），校验失败不抛异常，
    结果在 ``structured_output`` 而不是消息正文。

    ⚠️ 一并断言「模型看得见的工具**只有**这一个」—— 也就是意图识别
    智能体**没有**业务工具。这是设计决定（见
    :meth:`IntentRecognizer._recognize_with_model`）：给它工具会让它在
    「理解」阶段顺手查一次航班，而查询结果会污染它对意图的判断。
    这个改动**不会让任何功能测试变红**，只会让识别质量悄悄下降。
    """
    model = StructuredModel(make_payload(("PLAN_TRIP", 0.9)))
    await build(model=model).recognize("帮我规划行程")

    assert model.calls == 1, "一次识别只应发生一次模型调用"
    assert model.seen_tools == [STRUCTURED_TOOL_NAME], (
        f"意图识别智能体不该带别的工具，实际看到：{model.seen_tools}"
    )


async def test_the_agent_is_named_and_prompted_from_the_registry() -> None:
    """⚠️ 智能体名与提示词都取自登记处，不是硬写的字符串。

    ⚠️ 名字要进日志与 trace 的 span 名，提示词要给模型一个「不面向用户」
    的身份设定。硬写字面量的话，改名时漏掉这里的后果是
    「日志里一个名字、trace 里另一个名字」。
    """
    assert INTENT_AGENT_NAME == AgentName.INTENT.value
    assert INTENT_AGENT_NAME in PROMPTS

    text = PROMPTS[INTENT_AGENT_NAME]
    assert "不直接回答" in text or "不面向用户" in text or "只输出" in text


# ---------------------------------------------------------------------------
# 三、无状态：两次识别之间不得互相污染
# ---------------------------------------------------------------------------
async def test_each_recognition_uses_a_fresh_agent() -> None:
    """★★ 每次识别都新建 ``Agent``，上下文不跨次累积。

    ⚠️ 这条是本文件里**最难通过肉眼发现**的一类缺陷的守卫。

    ``Agent`` 持有 ``AgentState``，而 ``AgentState.context`` 是**累积的
    对话上下文**。若把 ``Agent`` 缓存在实例上「构造一次、反复用」：

    · 第二次识别时上下文里带着第一次的用户输入与结论，模型会受它影响
      （「上次判了 PLAN_TRIP，这次大概也是」）；
    · ``state.reply_context.structured_output`` 还会残留上一轮的结果 ——
      某次识别失败时读到的会是**上一次**的成功结果，用户拿到一个
      张冠李戴的意图，且完全无迹可循。

    两种表现都**不会报错**，所以只能靠「构造了几次 Agent」来观测。
    这里临时替换 ``Agent.__init__``，既数次数，也顺手记下每次构造出来的
    Agent 的初始上下文长度 —— 后者直接证明「新实例是干净的」。
    """
    from agentscope.agent import Agent

    context_lengths: list[int] = []
    real_init = Agent.__init__

    def capturing_init(self: Agent, *args: Any, **kwargs: Any) -> None:
        """记录每次构造。"""
        real_init(self, *args, **kwargs)
        context_lengths.append(len(getattr(self.state, "context", None) or []))

    recognizer = build(model=StructuredModel(make_payload(("PLAN_TRIP", 0.9))))
    Agent.__init__ = capturing_init  # type: ignore[method-assign]
    try:
        await recognizer.recognize("第一句")
        await recognizer.recognize("第二句")
        await recognizer.recognize("第三句")
    finally:
        Agent.__init__ = real_init  # type: ignore[method-assign]

    assert context_lengths == [0, 0, 0], (
        f"三次识别构造出的 Agent 初始上下文分别是 {context_lengths}，"
        f"应当都是空的。非空说明 Agent 实例被复用了。"
    )


async def test_two_recognitions_do_not_share_results() -> None:
    """★★ 上一次的结果不得泄漏到下一次（失败时尤其明显）。

    ⚠️ 构造的是「第一次成功、第二次模型什么都不交付」的场景。
    若 ``Agent`` 被复用，第二次会从残留的 ``reply_context.structured_output``
    里读到**第一次**的结果 —— 于是用户问了一句完全无关的话，
    系统回的是上一次的意图。这条用例就是为这个场景写的。
    """
    calls = {"n": 0}

    class FlakyModel(MockChatModel):
        """第一次交付结果，第二次什么都不交付。"""

        async def _call_api(self, *args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            if calls["n"] == 1:
                return ChatResponse(
                    content=[
                        ToolCallBlock(
                            id="structured-1",
                            name=STRUCTURED_TOOL_NAME,
                            input=json.dumps(make_payload(("PLAN_TRIP", 0.99))),
                        ),
                    ],
                    is_last=True,
                )
            return await super()._call_api(*args, **kwargs)

    recognizer = build(model=FlakyModel())
    first = await recognizer.recognize("帮我规划行程")
    second = await recognizer.recognize("那个东西怎么样了")

    assert first.top_intent() is Intent.PLAN_TRIP
    assert second.top_intent() is Intent.OTHER, "第二次读到了上一次的结果"
    assert second.needs_clarification is True


# ---------------------------------------------------------------------------
# 四、降级路径：永远返回一个对象，永不抛异常
# ---------------------------------------------------------------------------
async def test_model_that_delivers_nothing_degrades_to_clarification() -> None:
    """★★ 模型交不出结构化结果时**降级为追问**，不抛异常。

    ⚠️ 这条路径在框架里**没有任何异常**：模型在 ``max_iters`` 之内始终没调
    ``GenerateStructuredOutput``，回复就以 ``EXCEED_MAX_ITERS`` 结束，
    ``structured_output`` 是 ``None``（``agent/_agent.py:3580-3653``）。
    照「异常处理」的思路去写这段代码，就会漏掉它 —— 而漏掉的症状是
    ``AttributeError: 'NoneType' object has no attribute 'get'``，
    发生在整轮对话的主链路上，用户直接看到一次失败。

    ⚠️ 降级结果里**必须**有 ``OTHER`` 意图。空列表会让编排层
    「按最高分意图调度」时拿不到任何东西。
    """
    model = StructuredModel(None)
    result = await build(model=model).recognize("那个东西怎么样了")

    assert result.needs_clarification is True
    assert result.top_intent() is Intent.OTHER
    assert result.clarification_question.strip(), "追问内容为空，用户不知道要说什么"


async def test_model_exception_degrades_instead_of_propagating() -> None:
    """★★ 模型调用抛异常时**降级**，不把异常抛给调用方。

    ⚠️ 本方法跑在每一轮对话的主链路上。「意图识别失败」的正确处置是
    降级成追问，不是把整轮回复打断 —— 用户说一句话就吃一个 500，
    而系统本可以说「能说得再具体一点吗」。
    """
    model = ExplodingModel()
    result = await build(model=model).recognize("帮我规划行程")

    assert model.calls == 1, "模型确实被调用过（不是因为别的原因跳过）"
    assert result.needs_clarification is True
    assert result.top_intent() is Intent.OTHER


async def test_the_degraded_question_does_not_leak_internal_reasons() -> None:
    """★★ 降级时给用户看的问题里**不得**出现内部原因。

    ⚠️ 向用户解释「模型没有交付结构化结果」「解析失败」既没有帮助，
    也暴露了内部实现。用户拿这种话没有任何办法处理 ——
    他能做的只有再说一遍，而问题多半不在他。

    ⚠️ 守的是**用户可见的那个字段**，不是日志。日志里必须有原因
    （见 :func:`test_degradation_is_logged`），两者是分开的。
    """
    leaks = ("模型", "结构化", "解析", "schema", "配置", "异常", "错误", "失败")
    for model in (StructuredModel(None), ExplodingModel()):
        result = await build(model=model).recognize("随便说点什么")
        question = result.clarification_question
        for leak in leaks:
            assert leak not in question, f"追问话术泄漏了内部原因 {leak!r}：{question!r}"


async def test_degradation_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """⚠️ 降级时**必须**留下日志。

    ⚠️ 降级的设计目标是「让用户察觉不到」—— 这正是它危险的地方：
    一个意图识别长期失败的系统，用户每次都会被追问一句，
    而他会以为「是我没说清楚」。没有日志的话，没有任何人会知道
    真正的原因是模型交付不了结构化结果。
    """
    import logging

    with caplog.at_level(logging.WARNING, logger="src.agents.intent"):
        await build(model=StructuredModel(None)).recognize("随便说点什么")

    assert any("意图识别" in record.message for record in caplog.records), \
        "降级路径没有留下日志"


async def test_empty_input_never_reaches_the_model() -> None:
    """★★ 空输入**不调模型**，直接降级。

    ⚠️ 空输入里没有任何信息可供判断，模型只能编 —— 而它「编」出来的
    意图会照常进入编排层，被当成真实判断。这比报错糟得多：
    用户什么都没说，系统却开始规划一趟行程。

    ⚠️ 断言 ``calls == 0``。只断言「返回了追问」是不够的 ——
    一个先调模型、失败后再降级的实现同样会返回追问。
    """
    for text in ("", "   ", "\n\t "):
        model = StructuredModel(make_payload(("PLAN_TRIP", 0.9)))
        result = await build(model=model).recognize(text)

        assert model.calls == 0, f"输入 {text!r} 时仍然调用了模型"
        assert result.top_intent() is Intent.OTHER
        assert result.needs_clarification is True


# ---------------------------------------------------------------------------
# 五、追问判定
# ---------------------------------------------------------------------------
async def test_low_confidence_triggers_clarification() -> None:
    """★ 置信度低于阈值时**追问**，而不是按最高分意图往下走。

    ⚠️ 阈值存在的意义：模型在「都不太像」时会挑一个最接近的，
    此时它的 confidence 通常不高。猜错的代价是整条链路白跑 ——
    系统去查了交通、订了酒店，而用户其实想问的是报销标准。
    """
    model = StructuredModel(make_payload(("PLAN_TRIP", 0.2)))
    result = await build(model=model, confidence_threshold=0.6).recognize("嗯，那个")

    assert result.needs_clarification is True
    assert result.top_intent() is Intent.PLAN_TRIP, "低置信度的意图仍然要保留，供人工排查"


async def test_a_model_supplied_question_is_preferred_over_the_fallback() -> None:
    """★ 模型给了追问内容时用模型的，不要用兜底话术。

    ⚠️ 模型刚看过用户说了什么，它的追问比一句通用话术更贴题。
    反过来的话，我们会用「你是想规划行程、查订单…？」盖掉一个
    「你说的『那个』是指上周那张机票吗？」—— 后者明显更有用。
    """
    model = StructuredModel(
        make_payload(
            ("PLAN_TRIP", 0.3),
            clarification_question="你指的是上周那张去上海的机票吗？",
        ),
    )
    result = await build(model=model).recognize("那个还能改吗")

    assert result.clarification_question == "你指的是上周那张去上海的机票吗？"


async def test_other_intent_alone_triggers_clarification() -> None:
    """★ 只有 ``OTHER`` 时算「识别不出来」，即使置信度很高。

    ⚠️ ``OTHER`` 是模型表达「都不像」的出口（见
    :class:`src.domain.enums.Intent` 的说明）。把它当普通意图往下走，
    编排层会去找一个「OTHER 对应的智能体」—— 而那个映射**不存在**。
    结果是一个空的目标列表，用户得不到任何回应。
    """
    model = StructuredModel(make_payload(("OTHER", 0.95)))
    result = await build(model=model).recognize("今天天气不错")

    assert result.needs_clarification is True
    assert result.clarification_question.strip()


async def test_high_confidence_clears_a_stale_clarification_flag() -> None:
    """★ 识别成功时**必须**把 ``needs_clarification`` 清掉。

    ⚠️ 模型偶尔会一边给出高置信度意图、一边把追问标志也置上
    （它把「顺便确认一下」也算成了追问）。留着这个矛盾标志，
    编排层会因为「要追问」而放弃一个本来完全可用的判断 ——
    系统的行为变成「明明认出来了却什么都不做」。
    """
    model = StructuredModel(
        make_payload(
            ("PLAN_TRIP", 0.95),
            needs_clarification=True,
            clarification_question="要帮你订票吗？",
        ),
    )
    result = await build(model=model).recognize("帮我订下周三去北京的票")

    assert result.needs_clarification is False
    assert result.clarification_question == ""


# ---------------------------------------------------------------------------
# 六、意图数量上限
# ---------------------------------------------------------------------------
async def test_intents_are_capped_at_max_intents() -> None:
    """★★ 意图数量受 ``max_intents`` 限制。

    ⚠️ 必须有上限：多意图输入会解析出多条，而无上限时一次请求可能触发
    一串子调用，把延迟与成本都放大到不可控（见
    ``OrchestrationSettings.max_subagent_calls`` 的说明）。

    ⚠️ 截断规则是「保留置信度最高的 N 条」而不是「保留前 N 条」——
    模型给的顺序不一定按置信度，按位置截会砍掉真正重要的那个。
    """
    payload = make_payload(
        ("CHITCHAT", 0.10),
        ("OTHER", 0.20),
        ("CANCEL", 0.30),
        ("QUERY_ORDER", 0.40),
        ("QUERY_POLICY", 0.50),
        ("PLAN_TRIP", 0.90),
    )
    model = StructuredModel(payload)
    result = await build(model=model, max_intents=2).recognize("好多事")

    assert len(result.intents) == 2
    assert [item.intent for item in result.intents] == [
        Intent.PLAN_TRIP,
        Intent.QUERY_POLICY,
    ], "截断时砍掉的不是置信度最低的那些"


async def test_a_shared_model_serves_many_recognizers_independently() -> None:
    """★★ 同一个模型实例服务**多个**识别器时互不影响。

    ⚠️ 模型是被共享的：``ChatModelBase`` 不持有对话状态（状态在 ``Agent``
    上），这正是本类每次新建 ``Agent`` 却复用模型实例的**依据**。
    一旦这条依据不成立（有人在模型上挂了状态），症状是「两个用户的识别
    结果互相干扰」，而它只在并发下出现 —— 极其难复现。

    ⚠️ 用**两次都交付**的模型（:class:`AlwaysStructuredModel`），
    而不是上面那个「只交付一次」的替身：后者会让第二次识别拿不到结果，
    于是断言变成在测「模型只交付一次」，与共享无关。这个错误
    本文件的作者真的犯过一次。
    """
    model = AlwaysStructuredModel(make_payload(("PLAN_TRIP", 0.9)))
    first = build(model=model)
    second = build(model=model)

    a = await first.recognize("帮我规划行程")
    b = await second.recognize("帮我查订单")

    assert a.top_intent() is Intent.PLAN_TRIP
    assert b.top_intent() is Intent.PLAN_TRIP
    assert model.calls == 2, "两次识别应各发生一次模型调用"


# ---------------------------------------------------------------------------
# 七、构造期校验
# ---------------------------------------------------------------------------
def test_max_intents_must_be_positive() -> None:
    """★★ ``max_intents < 1`` 在**构造时**失败。

    ⚠️ 不校验的话，``max_intents=0`` 会让所有识别结果被截成空列表，
    于是每一次识别都降级成追问 —— 系统完全不可用，而**没有任何报错**。
    这种「配置写错导致功能静默失效」必须在启动时暴露。
    """
    with pytest.raises(ValueError, match="max_intents"):
        build(max_intents=0)
    with pytest.raises(ValueError, match="max_intents"):
        build(max_intents=-1)


def test_confidence_threshold_must_be_a_probability() -> None:
    """★★ 阈值必须在 ``[0, 1]`` 内。

    ⚠️ 阈值大于 1 时每一次识别都触发追问（置信度最高就是 1.0，
    而判定是严格小于），系统同样完全不可用且不报错。
    """
    for bad in (-0.1, 1.5, 2.0):
        with pytest.raises(ValueError, match="confidence_threshold"):
            build(confidence_threshold=bad)


def test_threshold_boundaries_are_accepted() -> None:
    """边界值 ``0`` 与 ``1`` 是合法的。

    ⚠️ 与上一条成对：上一条保证非法值被拒，这一条保证**合法值不被误拒**。
    只写前者的测试，会让「把 ``<=`` 写成 ``<``」这种错误溜过去 ——
    而那会让 ``threshold=1`` 这个正当配置在启动时报错。
    """
    build(confidence_threshold=0.0)
    build(confidence_threshold=1.0)


def test_build_intent_recognizer_has_no_defaults_for_config() -> None:
    """⚠️ 工厂函数**不提供**配置默认值。

    ⚠️ 调用方手上一定有 ``OrchestrationSettings``。给默认值的话，
    「配置改了但构造处用了默认值」会让系统行为与配置不符，
    而排查时第一反应是「配置没生效」，方向就偏了。
    """
    import inspect

    signature = inspect.signature(build_intent_recognizer)
    for name in ("max_intents", "confidence_threshold"):
        assert signature.parameters[name].default is inspect.Parameter.empty, \
            f"{name} 不该有默认值"

    recognizer = build_intent_recognizer(
        model=StructuredModel(make_payload(("PLAN_TRIP", 0.9))),
        max_intents=3,
        confidence_threshold=0.5,
    )
    assert isinstance(recognizer, IntentRecognizer)
