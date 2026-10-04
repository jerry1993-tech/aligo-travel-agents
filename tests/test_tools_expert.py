# -*- coding: utf-8 -*-
"""意图识别工具（``src/tools/expert.py``）的测试。

═══ 这张表守的是什么 ═══

把专家智能体包成工具，难点**不在包装**，在两处很容易写反的地方：

1. **工具结果是给模型看的，不是给用户看的。** 摘要写成「你想规划行程」，
   主智能体会把它当成成品直接复述出去 —— 用户看到一句「系统判断你的
   意图是…」，那是内部实现泄漏。所以摘要必须是工程化措辞，且断言它在
   被复述到用户面前时是**不好看**的。

2. **工具名与常量必须一致。** ``FunctionTool`` 默认拿函数名当工具名，
   而提示词、日志、测试引用的是常量。改名漏一处**不报错**，
   只是模型需要它的时候「找不到这个工具」。

⚠️ 这两条都不会让任何功能测试变红 —— 只有本文件会发现。
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from agentscope.message import ToolResultState

from src.domain.enums import Intent
from src.domain.schemas import IntentDecision, IntentRecognitionResult
from src.tools._result import CARD_KEY
from src.tools.expert import INTENT_TOOL_NAME, build_intent_tool
from src.tools.travel import tools_of_kind


# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------
class StubRecognizer:
    """一个不碰模型的识别器，返回预先摆好的结果。

    ⚠️ 这里刻意**不**用真的 ``IntentRecognizer``：那个类跑一轮真实的
    ``Agent.reply``，它的行为已经在 ``tests/test_agents_intent.py`` 里
    逐个断言过了。在这个文件里再跑一遍，只会让「包装层」的用例变慢、
    变脆，而且掩盖掉一处真正的问题 —— 包装层有没有把结果**如实**搬出来。

    Attributes:
        seen (`list[str]`): 每次 ``recognize`` 收到的文本，用来断言
            「原样传入、不改写」。
        raised (`Exception | None`): 非 ``None`` 时 ``recognize`` 直接抛。
    """

    def __init__(self, result: IntentRecognitionResult | None = None) -> None:
        """初始化。

        Args:
            result (`IntentRecognitionResult | None`): 要返回的结果。
        """
        self._result = result or IntentRecognitionResult()
        self.seen: list[str] = []
        self.raised: Exception | None = None

    async def recognize(self, text: str, **_kwargs: Any) -> IntentRecognitionResult:
        """记录输入并返回结果。

        Args:
            text (`str`): 输入文本。
            **_kwargs: 忽略。

        Returns:
            `IntentRecognitionResult`: 预置结果。

        Raises:
            Exception: ``raised`` 非 ``None`` 时抛出它。
        """
        self.seen.append(text)
        if self.raised is not None:
            raise self.raised
        return self._result


def decision(
    intent: Intent = Intent.PLAN_TRIP,
    confidence: float = 0.9,
    **kwargs: Any,
) -> IntentDecision:
    """构造一条意图判定。

    Args:
        intent (`Intent`): 意图。
        confidence (`float`): 置信度。
        **kwargs: 覆盖默认字段。

    Returns:
        `IntentDecision`: 判定。
    """
    params: dict[str, Any] = {
        "intent": intent,
        "confidence": confidence,
        "slots": {},
        "reason": "因为用户说了出差",
    }
    params.update(kwargs)
    return IntentDecision(**params)


def payload_of(chunk: Any) -> dict[str, Any]:
    """把工具返回的 ``ToolChunk`` 解回 JSON 载荷。

    ⚠️ 从**文本**里解 JSON，不去读结构化字段 —— 生产路径就是
    「把 JSON 塞进一个 TextBlock」，测结构化字段等于测一条不存在的路径。

    Args:
        chunk (`Any`): 工具返回。

    Returns:
        `dict[str, Any]`: 载荷。
    """
    text = "".join(getattr(block, "text", "") for block in chunk.content)
    return json.loads(text)


def call(tool: Any, /, **kwargs: Any) -> Any:
    """调用工具（同步等待其异步内部函数）。

    Args:
        tool (`Any`): 工具。
        **kwargs: 关键字参数。

    Returns:
        `Any`: 工具返回。
    """
    return asyncio.run(tool(**kwargs))


def build(result: IntentRecognitionResult | None = None) -> tuple[Any, StubRecognizer]:
    """构造工具与它背后的替身。

    Args:
        result (`IntentRecognitionResult | None`): 替身要返回的结果。

    Returns:
        `tuple[Any, StubRecognizer]`: 工具与替身。
    """
    recognizer = StubRecognizer(result)
    tools = tools_of_kind([build_intent_tool(recognizer)])  # type: ignore[arg-type]
    return tools[INTENT_TOOL_NAME], recognizer


# ---------------------------------------------------------------------------
# 一、契约
# ---------------------------------------------------------------------------
def test_the_tool_name_matches_the_constant() -> None:
    """★★ 注册出来的工具名与 ``INTENT_TOOL_NAME`` **逐字相同**。

    ⚠️ ``FunctionTool`` 用**函数名**当工具名，而提示词、日志、前端契约
    引用的是常量。两者之间没有任何编译期联系 —— 改了函数名不改常量
    （或反过来），系统照常启动、照常跑，只是模型在需要它的时候
    「找不到这个工具」，于是意图识别被静默跳过。

    ⚠️ 生产代码里有一句同样的断言（构造时立刻 ``raise``）。这条用例守的是
    **那句断言本身还在**：它被删掉之后，功能测试仍然全绿。
    """
    tool, _ = build()
    assert tool.name == INTENT_TOOL_NAME


def test_the_tool_is_read_only() -> None:
    """★★ 工具标记为**只读** —— 否则每次调用都要用户点一次「允许」。

    ⚠️ 这不是风格问题，是产品可用性问题：用户点「规划行程」时先被弹一个
    「正在识别你的意图，是否允许？」的确认框，产品没法用。

    ⚠️ 断言的是 ``is_read_only``，**不是**去调 ``check_permissions()`` ——
    后者对只读工具也会返回 ``ASK``（已核实：``FunctionTool`` 不传
    ``permission`` 时一律 ``PermissionDecision(behavior=ASK)``，
    ``agentscope/tool/_adapters.py:116-135``）。只看那个返回值会得出**相反**的结论，
    从而把这条护栏删掉。
    """
    tool, _ = build()
    assert tool.is_read_only is True


def test_the_tool_has_a_documented_signature() -> None:
    """★ 工具的 schema 里能看出参数是什么 —— 模型靠它决定怎么调。

    ⚠️ 函数体里的 docstring 会成为工具描述的一部分，参数名 ``text`` 与
    它的说明会成为 schema。把 docstring 删掉不影响任何功能测试，
    但它正是模型「会不会正确传参」的唯一依据。
    """
    tool, _ = build()
    schema = tool.input_schema
    assert "text" in schema.get("properties", {}), "工具参数没有 text"


# ---------------------------------------------------------------------------
# 二、输入处理
# ---------------------------------------------------------------------------
def test_empty_text_is_an_error_not_a_question_to_the_user() -> None:
    """★★ 空输入返回 ``ERROR``，**不是** ``needs_input``。

    ⚠️ 这条守的是一个很容易搞反的语义。``needs_input`` 的意思是
    「用户还没说清楚，去问用户」；而空 ``text`` 是**模型**没把参数传进来
    —— 该被纠正的是模型自己。用 ``needs_input`` 的话，用户会收到一句
    「请问你想做什么？」，而他刚刚才说过话。

    ⚠️ 同时断言**没有调用识别器**：空文本喂给模型既浪费一次调用，
    又可能让模型对着空串编出一个意图。
    """
    tool, recognizer = build()
    chunk = call(tool, text="   ")

    assert chunk.state is ToolResultState.ERROR
    payload = payload_of(chunk)
    assert payload["ok"] is False
    assert recognizer.seen == [], "空文本不该喂给识别器"


def test_the_text_is_passed_through_verbatim() -> None:
    """★★ 用户原话**原样**传给识别器，不裁剪、不改写。

    ⚠️ 措辞与错别字恰恰是判断意图的关键信息：「帮我订张票」
    与「帮我查一下订票的报销标准」差别全在几个字上。在包装层顺手
    ``strip`` 或补全句子，会让识别器看不到真实输入。
    """
    tool, recognizer = build()
    raw = "  下周三去北京开会，帮我看看能住哪  "
    call(tool, text=raw)

    # 允许去掉首尾空白（那是传输噪音），但**中间一个字符都不能动**。
    assert len(recognizer.seen) == 1
    assert recognizer.seen[0].strip() == raw.strip()
    assert recognizer.seen[0] == raw.strip()


# ---------------------------------------------------------------------------
# 三、成功结果的形状
# ---------------------------------------------------------------------------
def test_a_successful_result_carries_every_intent() -> None:
    """★★ 成功时每条意图都在 ``items`` 里，字段齐全。

    ⚠️ 逐字段查而不是查「非空」：``slots`` 与 ``reason`` 是漏掉之后
    **界面看起来仍然正常**的字段（卡片能渲染，只是详情里少两行），
    所以它们最容易被漏掉，也最需要被钉住。
    """
    tool, _ = build(
        IntentRecognitionResult(
            intents=[
                decision(Intent.PLAN_TRIP, 0.93, slots={"destination": "北京"}),
                decision(Intent.QUERY_POLICY, 0.71, slots={"city": "北京"}),
            ],
        ),
    )
    payload = payload_of(call(tool, text="下周三去北京开会，住宿标准是多少"))

    assert payload["ok"] is True
    assert len(payload["items"]) == 2
    for item in payload["items"]:
        assert set(item) == {"intent", "confidence", "slots", "reason"}

    assert payload["items"][0]["intent"] == Intent.PLAN_TRIP.value
    assert payload["items"][0]["slots"] == {"destination": "北京"}
    assert payload["items"][1]["intent"] == Intent.QUERY_POLICY.value


def test_the_tool_returns_no_card() -> None:
    """★★ 意图识别**不产出卡片** —— ``card`` 是空串。

    ⚠️ 这条是有意为之的产品决定，值得钉住：意图识别的结果会经由主智能体
    变成一句自然语言。若给它一张卡片，用户会在对话里看到一个
    「意图判定表」—— 那是把内部实现直接摆在用户面前。

    ⚠️ 同时这也保证了前端的「自定义卡片」验收项不会被这张表污染：
    前端只认非空的 ``card`` 值来选渲染器。
    """
    tool, _ = build(IntentRecognitionResult(intents=[decision()]))
    payload = payload_of(call(tool, text="我想去北京"))

    assert payload[CARD_KEY] == ""


def test_reasoning_and_rewritten_query_are_exposed() -> None:
    """★★ 「显示推理」的两个字段必须在载荷里，字段名不能变。

    ⚠️ 这两个字段是**前端契约**的一部分（前端读它们渲染推理区块）。
    pydantic 的字段名改动在 Python 侧是安全的（有默认值、不报错），
    但它会让前端静默拿不到数据 —— 推理区块消失，不报错。

    ⚠️ 用 ``rewritten_query`` 这个名字断言而不是从模型上读，是为了让
    改名这件事在这里变红。
    """
    tool, _ = build(
        IntentRecognitionResult(
            reasoning="用户提到出差与住宿标准，涉及规划与政策两类诉求。",
            rewritten_query="下周三从杭州到北京出差，查询北京住宿报销标准",
            intents=[decision()],
        ),
    )
    payload = payload_of(call(tool, text="下周三去北京开会"))

    assert payload["reasoning"].startswith("用户提到出差")
    assert payload["rewritten_query"].startswith("下周三从杭州到北京")
    assert payload["needs_clarification"] is False
    assert payload["clarification_question"] == ""


def test_a_clarification_request_is_carried_through() -> None:
    """⚠️ 需要追问时，标志与问题都要带出来。"""
    tool, _ = build(
        IntentRecognitionResult(
            needs_clarification=True,
            clarification_question="你是想去北京出差，还是查询北京的住宿标准？",
            intents=[decision(Intent.OTHER, 0.3)],
        ),
    )
    payload = payload_of(call(tool, text="北京"))

    assert payload["needs_clarification"] is True
    assert "北京" in payload["clarification_question"]


def test_an_empty_result_still_returns_a_well_formed_payload() -> None:
    """★ 一条意图都没识别出来时，载荷依然完整（不是 ``items`` 缺失）。

    ⚠️ 返回 ``items: []`` 而不是 ``null`` —— 前端与模型都要能区分
    「识别了但结果是空」与「这个工具不产出列表」。缺字段会让它们
    各自写一个兜底分支，而那些分支永远不会被执行到。
    """
    tool, _ = build(IntentRecognitionResult(intents=[]))
    payload = payload_of(call(tool, text="嗯"))

    assert payload["items"] == []
    assert payload["ok"] is True
    assert isinstance(payload["summary"], str) and payload["summary"]


# ---------------------------------------------------------------------------
# 四、摘要必须是「工程化措辞」
# ---------------------------------------------------------------------------
def test_the_summary_is_engineering_flavoured_not_a_reply_to_the_user() -> None:
    """★★★ 摘要写成枚举名 + 置信度，**不能**写成一句像回答的话。

    ⚠️ 这是本文件最重要的一条。工具结果会原样进入主智能体的上下文，
    而模型对「已经像成品」的文本有很强的复述倾向。摘要若是
    「你想规划去北京的行程，同时想了解住宿标准」，模型很可能直接把它
    当答复发出去 —— 用户看到「系统判断你的意图是…」，那是内部实现泄漏，
    而且是**每一轮**都会泄漏。

    ⚠️ 断言方式：摘要里必须有**枚举名**（``PLAN_TRIP`` 这种全大写下划线
    的标识符）。它们在给用户看的句子里是不可能自然出现的，
    所以这条断言实际上锁死了「摘要不能被直接复述」这个性质。
    """
    tool, _ = build(
        IntentRecognitionResult(
            intents=[
                decision(Intent.PLAN_TRIP, 0.93),
                decision(Intent.QUERY_POLICY, 0.71),
            ],
        ),
    )
    summary = payload_of(call(tool, text="下周三去北京开会，住宿标准是多少"))["summary"]

    assert Intent.PLAN_TRIP.value in summary, f"摘要里没有枚举名，模型会直接复述：{summary!r}"
    assert Intent.QUERY_POLICY.value in summary
    assert "2" in summary, f"摘要没有说明识别出几条意图：{summary!r}"


def test_the_summary_reports_the_clarification_flag() -> None:
    """⚠️ 需要追问时，摘要里要**说出来**，别让模型自己从字段里猜。

    ⚠️ 模型看到一堆 ``items`` 不一定会去读 ``needs_clarification``；
    在摘要里点名一句，追问这件事才会真的发生。
    """
    tool, _ = build(
        IntentRecognitionResult(
            needs_clarification=True,
            intents=[decision(Intent.OTHER, 0.2)],
        ),
    )
    summary = payload_of(call(tool, text="嗯"))["summary"]

    assert "追问" in summary, f"摘要没提追问：{summary!r}"


def test_the_summary_handles_an_empty_intent_list() -> None:
    """⚠️ 一条意图都没有时，摘要也要是一句完整的话（不是空串）。

    ⚠️ 先说清楚这条用例**守的是什么**：它守的是 :func:`_summarize` 的**接口
    契约**（「无论给什么列表，都返回一句非空的话」），**不是**一个在跑得到
    的分支。当前识别器**产不出**空列表：``IntentRecognizer._degrade``
    在解析失败时会返回恰好一条 ``OTHER`` 意图（``src/agents/intent.py``），
    所以 ``intents=[]`` 在生产链路上到不了这里。

    这是刻意保留的：``_summarize`` 的入参是鸭子类型的「有 ``intent`` 与
    ``confidence`` 的列表」，将来换一个识别器就可能真的给空列表。
    界面对空摘要的容忍度为零（一行空白在进行中），所以这条边界值得
    有一条用例钉着 —— 但把它当成「覆盖了一个真实分支」来读会高估覆盖率。
    """
    tool, _ = build(IntentRecognitionResult(intents=[]))
    summary = payload_of(call(tool, text="嗯"))["summary"]

    assert summary.strip(), "空摘要在界面上是一行空白"
    assert "没有" in summary or "识别" in summary


# ---------------------------------------------------------------------------
# 五、失败处理
# ---------------------------------------------------------------------------
def test_unexpected_failures_are_left_to_the_framework() -> None:
    """★★★ 不可预期的失败**让它抛出去**，由框架兜底 —— 本层不宽捕。

    ⚠️ 这条守的是一个**设计边界**：只有「可预期的失败」（空输入、参数不对）
    才该被这一层翻译成中文，其余一律上抛。

    理由是 —— 在这里宽捕 ``Exception`` 会把它变成一句笼统的
    「识别失败，请重试」，而真正的根因（模型网关挂了、配置错了）就再也
    不会出现在任何地方：框架那层本来会记一条带堆栈的日志，被本层吞掉之后
    就只剩一句用户看了也做不了什么的提示。

    ⚠️ 三种异常一起测（``ConnectionError`` / ``RuntimeError`` /
    ``ValueError``），因为它们在本项目里代表三种不同的根因：上游不可达、
    配置错误、数据坏了。写一条 ``except Exception`` 会把三种一起吞掉。

    ⚠️ 断言的是**原类型**上抛，不是 ``Exception`` —— 中间层把它包成别的
    类型（比如自定义的 ``ToolError``）同样会让堆栈丢失。
    """
    tool, recognizer = build()
    for boom in (ConnectionError("网关超时"), RuntimeError("配置错了"), ValueError("数据坏了")):
        recognizer.raised = boom
        with pytest.raises(type(boom)):
            call(tool, text="我想去北京")


def test_a_cancellation_is_not_swallowed() -> None:
    """★★ ``CancelledError`` 必须**原样**向上传播。

    ⚠️ 与上一条分开写，是因为它继承 ``BaseException`` 而不是
    ``Exception``：一条 ``except Exception`` 捕不到它，但一条
    ``except BaseException`` 能 —— 而后者是本项目里真实会犯的错
    （收集器的监听器通知处就刻意写了 ``except Exception``，
    见 ``src/chains/collector.py`` 的说明）。
    """
    tool, recognizer = build()
    recognizer.raised = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        call(tool, text="我想去北京")


def test_each_call_uses_the_same_recognizer_instance() -> None:
    """★ 同一个工具被反复调用时复用同一个识别器实例。

    ⚠️ 复用的安全性由识别器自己保证（``src/agents/intent.py`` 里每次都
    新建 ``Agent``，不共享对话状态）。这条用例把它钉住：若哪天识别器改成
    「构造时建一个 Agent 反复用」，这里不会变红 —— 但
    ``tests/test_agents_intent.py`` 的
    ``test_each_recognition_uses_a_fresh_agent`` 会。
    """
    tool, recognizer = build(IntentRecognitionResult(intents=[decision()]))
    for text in ("去北京", "去上海", "去广州"):
        call(tool, text=text)

    assert recognizer.seen == ["去北京", "去上海", "去广州"]
