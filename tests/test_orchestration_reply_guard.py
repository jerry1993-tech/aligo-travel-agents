# -*- coding: utf-8 -*-
"""回复守卫（:class:`ReplyGuardMiddleware`）的测试。

两层证据，各自回答不同的问题：

1. **纯函数层**（``_is_draft`` / ``_strip_drafts`` / ``_is_placeholder`` /
   ``_ungrounded_amounts``）—— 判据本身对不对。用的是 **2026-10-03 实测
   抓到的真实缺陷原文**，不是编出来的例子。
2. **中间件层**（脚本化事件流 + 假 agent）—— 判据接进事件流之后行为对不对：
   草稿真的没发出去、工具调用真的照常透传、重试真的吞掉了 ``ReplyEndEvent``、
   中断/超迭代的结束真的没吞。

⚠️ 为什么中间件层用**假 agent 喂脚本事件**、而不是全部跑真实 ReAct：
因为这里要测的是「守卫怎么改事件流」，需要精确控制「哪一轮有工具调用、
结束原因是什么、迭代余量还剩多少」—— 这些用真实 ReAct 很难构造出**全部**
组合（尤其是 ``EXCEED_MAX_ITERS`` 与迭代余量不足那两条）。真实链路的
端到端验证放在最后一条用例，以及 ``scripts/concurrency_test.py``。

⚠️ 本模块的每一条断言都对应模块文档里写明的一个**实测缺陷或已核实的框架
约束**，注释里都标了出处。不要为了「覆盖率好看」加无关断言 ——
守卫改的是用户能看到的每一个字，测试的价值在于**精确**，不在于多。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, AsyncGenerator

import pytest
from agentscope.agent import Agent
from agentscope.event import (
    EventBase,
    ModelCallEndEvent,
    ModelCallStartEvent,
    ReplyEndEvent,
    ReplyStartEvent,
    TextBlockDeltaEvent,
    TextBlockEndEvent,
    TextBlockStartEvent,
    ToolCallStartEvent,
    ToolResultTextDeltaEvent,
)
from agentscope.message import Msg, TextBlock, ToolCallBlock
from agentscope.tool import Bash, Edit, Glob, Grep, Read, Write
from agentscope.tool._builtin import ResetTools
from agentscope.tool._task import TaskCreate, TaskGet, TaskList, TaskUpdate
from agentscope.model import ChatResponse
from agentscope.tool import FunctionTool, ToolChunk, Toolkit
from agentscope.types import ReplyFinishedReason

from src.domain import TravelRequest
from src.llm.mock import MockChatModel
from src.orchestration.reply_guard import (
    _POLICY_LIMIT_TOOLS,
    ReplyGuardMiddleware,
    _answer_problem,
    _classify_suspicious_paragraph,
    _classify_suspicious_paragraphs,
    _count_suspicious_paragraphs,
    _fold_width,
    _is_derived_total,
    _is_draft,
    _is_placeholder,
    _normalize_amount,
    _strip_drafts,
    _ungrounded_amounts,
    _ungrounded_limit_claims,
)
from src.server.agents_factory import _AgentScopedMiddleware

REPLY_ID = "reply-guard-1"
SESSION_ID = "session-guard-1"

# ==============================================================================
# 一、测试替身
# ==============================================================================


class _FakeTool:
    """只带 ``is_read_only`` 的工具替身。"""

    def __init__(self, *, is_read_only: bool) -> None:
        self.is_read_only = is_read_only


class _FakeToolkit:
    """按名字返回工具替身；查不到返回 ``None``（与框架一致）。"""

    def __init__(self, tools: dict[str, _FakeTool] | None = None) -> None:
        self._tools = tools or {}

    async def get_tool(self, name: str) -> _FakeTool | None:
        """返回工具替身。

        Args:
            name (`str`): 工具名。

        Returns:
            `_FakeTool | None`: 工具替身；未注册时为 ``None``。
        """
        return self._tools.get(name)


class _FakeState:
    """只实现守卫会用到的那部分 agent 状态。"""

    def __init__(self, *, cur_iter: int = 0) -> None:
        self.cur_iter = cur_iter
        #: 当前回复 id —— HITL 续答时由框架沿用（不是空串），
        #: 守卫用它做 ``reply_id`` 的第二级兜底。
        self.reply_id = ""
        #: ``reply_context`` —— 结构化输出相关的判据读它。
        self.reply_context: Any = None
        #: ``middle_context`` —— 框架的 ``AgentState`` 真有这个字段
        #: （``state/_state.py`` 的 ``dict[str, Any]``），是中间件之间的
        #: 交换区。守卫从这里读「动态 Prompt 注入过哪些数字」（缺陷 P1）。
        #: 替身必须带着它，否则那条路径**永远**读不到东西，
        #: 用例会在「忘了接线」的情况下照样变绿。
        self.middle_context: dict[str, Any] = {}
        #: 记录 ``append_context`` 的调用，供「重试必须塞纠正指令」断言用。
        self.context_calls: list[tuple[str, list[Any]]] = []

    def append_context(self, name: str, blocks: list[Any]) -> None:
        """记录塞进上下文的块。

        Args:
            name (`str`): 传进来的名字（守卫传 ``agent.name``）。
            blocks (`list[Any]`): 内容块。
        """
        self.context_calls.append((name, blocks))


class _FakeReactConfig:
    """只带 ``max_iters`` 的 ReAct 配置替身。"""

    def __init__(self, *, max_iters: int = 50) -> None:
        self.max_iters = max_iters


class _FakeAgent:
    """守卫需要的 agent 面：``name`` / ``toolkit`` / ``state`` / ``react_config``。"""

    def __init__(
        self,
        *,
        name: str = "main_plan",
        tools: dict[str, _FakeTool] | None = None,
        cur_iter: int = 0,
        max_iters: int = 50,
    ) -> None:
        self.name = name
        self.toolkit = _FakeToolkit(tools)
        self.state = _FakeState(cur_iter=cur_iter)
        self.react_config = _FakeReactConfig(max_iters=max_iters)


# ------------------------------------------------------------------------------
# 事件构造器
# ------------------------------------------------------------------------------
def _reply_start() -> ReplyStartEvent:
    """构造 ``REPLY_START``。"""
    return ReplyStartEvent(session_id=SESSION_ID, reply_id=REPLY_ID, name="main_plan")


def _call_start(reply_id: str = REPLY_ID) -> ModelCallStartEvent:
    """构造一轮模型调用的开始事件。

    Args:
        reply_id (`str`): 回复 id（续答场景传空串）。

    Returns:
        `ModelCallStartEvent`: 开始事件。
    """
    return ModelCallStartEvent(reply_id=reply_id, model_name="mock")


def _call_end(reply_id: str = REPLY_ID) -> ModelCallEndEvent:
    """构造一轮模型调用的结束事件。

    Args:
        reply_id (`str`): 回复 id（续答场景传空串）。

    Returns:
        `ModelCallEndEvent`: 结束事件。
    """
    return ModelCallEndEvent(reply_id=reply_id, input_tokens=1, output_tokens=1)


def _text(
    text: str,
    block_id: str = "block-1",
    reply_id: str = REPLY_ID,
) -> list[EventBase]:
    """构造一段文本的 START → DELTA → END 事件。

    ⚠️ ``reply_id`` 必须可传：测「续答时从哪里取 id」的用例要求**一个带 id
    的事件都没有**，而默认值会把 id 写死进每一个事件、让那些用例走进另一条
    分支还照样变绿（本文件的 reply_id 用例就这么假绿过一次，靠变异测试发现）。

    Args:
        text (`str`): 文本内容。
        block_id (`str`): 块 id。
        reply_id (`str`): 回复 id。

    Returns:
        `list[EventBase]`: 三个文本事件。
    """
    return [
        TextBlockStartEvent(reply_id=reply_id, block_id=block_id),
        TextBlockDeltaEvent(reply_id=reply_id, block_id=block_id, delta=text),
        TextBlockEndEvent(reply_id=reply_id, block_id=block_id, text=text),
    ]


def _tool_call(name: str, reply_id: str = REPLY_ID) -> ToolCallStartEvent:
    """构造工具调用开始事件。

    Args:
        name (`str`): 工具名。
        reply_id (`str`): 回复 id。

    Returns:
        `ToolCallStartEvent`: 工具调用事件。
    """
    return ToolCallStartEvent(
        reply_id=reply_id,
        tool_call_id=f"call-{name}",
        tool_call_name=name,
    )


def _tool_result(delta: str, reply_id: str = REPLY_ID) -> ToolResultTextDeltaEvent:
    """构造工具返回文本事件。

    Args:
        delta (`str`): 返回文本。
        reply_id (`str`): 回复 id。

    Returns:
        `ToolResultTextDeltaEvent`: 工具返回事件。
    """
    return ToolResultTextDeltaEvent(
        reply_id=reply_id,
        tool_call_id="call-x",
        delta=delta,
    )


def _reply_end(
    reason: ReplyFinishedReason = ReplyFinishedReason.COMPLETED,
    reply_id: str = REPLY_ID,
) -> ReplyEndEvent:
    """构造回复结束事件。

    Args:
        reason (`ReplyFinishedReason`): 结束原因。
        reply_id (`str`): 回复 id。

    Returns:
        `ReplyEndEvent`: 结束事件。
    """
    return ReplyEndEvent(
        session_id=SESSION_ID,
        reply_id=reply_id,
        finished_reason=reason,
    )


async def _run(
    middleware: ReplyGuardMiddleware,
    events: list[EventBase],
    agent: _FakeAgent,
    inputs: Any = None,
) -> list[EventBase]:
    """把脚本事件喂给守卫，收集它实际发出的事件。

    Args:
        middleware (`ReplyGuardMiddleware`): 守卫。
        events (`list[EventBase]`): 脚本事件流。
        agent (`_FakeAgent`): 假 agent。
        inputs (`Any`): 触发本次回复的输入，原样放进 ``input_kwargs["inputs"]``。
            默认 ``None`` —— 与框架在 HITL 续答等场景下传 ``None`` 时一致，
            守卫会把它当成「没有用户原话」（见 ``_user_text_from_inputs``）。

    Returns:
        `list[EventBase]`: 守卫发出的事件。

    ⚠️ ``input_kwargs`` 的键必须与框架一致（``agent/_agent.py:942-945``
    只放 ``inputs`` 与 ``structured_schema`` 两个）。这里保持同样的形状，
    否则「守卫读不到用户原话」这类接线错误在测试里永远不会暴露。
    """

    async def handler(**_kwargs: Any) -> AsyncGenerator[EventBase, None]:
        for event in events:
            yield event

    out: list[EventBase] = []
    async for event in middleware.on_reply(
        agent=agent,
        input_kwargs={"inputs": inputs},
        next_handler=handler,
    ):
        out.append(event)
    return out


def _joined(events: list[EventBase]) -> str:
    """把一段事件里所有文本增量拼起来。

    Args:
        events (`list[EventBase]`): 事件序列。

    Returns:
        `str`: 文本内容。
    """
    return "".join(
        event.delta for event in events if isinstance(event, TextBlockDeltaEvent)
    )


#: 用户实际看到的正文 —— 与生产链路口径一致（服务端按 DELTA 拼）。
def _visible(events: list[EventBase]) -> str:
    """返回「用户会看到什么」。

    Args:
        events (`list[EventBase]`): 守卫发出的事件。

    Returns:
        `str`: 可见正文。
    """
    return _joined(events)


# ==============================================================================
# 二、纯函数：判据本身
# ==============================================================================


#: 2026-10-03 实测抓到的真实草稿段落（来源见模块文档的 A/B/C/D 四类）。
_MEASURED_DRAFTS: list[str] = [
    # A 类：英文内心戏开场
    "I'll wait for the policy agent's report.",
    # A 类变体：中文「打算做什么」
    "我先查一下差标。",
    # B 类：整段内心独白（还带内部实现名字）
    "用户问的是差旅标准规定，我不应该自己凭常识推断金额，"
    "系统里也有专门的 check_travel_policy 工具。",
    # B 类：系统状态外泄
    "当前处于 IDLE 阶段，等待下一轮。",
    # B[6] 第二段：既有数字又是独白 —— 「不含数字」判据抓不住的那个 case
    "现在我看清楚了标准：酒店单晚不超过600元，我应该直接给出这个结论。",
    # B[3] 第七段：第三人称盘算用户 —— 第三条判据（他 + 计划动词）
    "顺便把话题引向出差要素——他既然要去北京，可以接着帮他落地方案。",
    # D 类：内部编排故障外泄
    "已经让 policy_rag 去检索制度原文了。",
]


@pytest.mark.parametrize("paragraph", _MEASURED_DRAFTS)
def test_measured_draft_paragraphs_are_recognized(paragraph: str) -> None:
    """★ 七段**实测**草稿必须全部被判为草稿。

    ⚠️ 用的全是真实抓到的原文，不是编的。少认一段，用户就多看到一句内心戏。
    这七段分别对应四条判据的**每一片**边界：段首独白、过程性措辞、
    第三人称盘算、内部实现名字 —— 只测其中一类，删掉另外三条判据时
    测试照样全绿（这正是要按判据分片取样的理由）。
    """
    assert _is_draft(paragraph), f"这段实测草稿没被认出来：{paragraph!r}"


#: 正经答复 —— 一条都不许误判（误剥真答案比留下草稿更糟）。
_REAL_ANSWERS: list[str] = [
    "住宿标准：**单晚不超过 600 元**，凭发票报销。",
    "我帮你查了一下，北京四环内可选的酒店有两家。",
    "他的订单还没提交，需要先完成审批。",
    "按 600 元一晚算，两晚合计 1200 元。",
    "这个政策涉及差旅标准，具体如下：单晚不超过 600 元。",
    # ---- 英文真答案：早先的 ``^I\b`` / ``^Since\b`` 会把它们判成独白 ----
    # ⚠️ 单段回复被判 monologue 的后果是**整段拒答**（用户拿到道歉话术），
    # 所以这几条是「误剥真答案」这一类里最贵的一种，必须钉住。
    # 判据改成「计划性动词 + 第一人称」之后它们才通过（见 _MONOLOGUE_OPENERS）。
    "I recommend the Beijing hotel at 600 CNY per night.",
    "I checked the policy: the hotel cap is 600 CNY per night.",
    "Since your trip starts Monday, book before Friday to save 200 CNY.",
    "Let me summarize the policy: hotel 600 CNY, economy class only.",
    # ---- 2026-10-04：``_THIRD_PERSON_PLANNERS`` 收窄后才能通过的两句 ----
    # ⚠️ 它们是**状态陈述**，不是「盘算下一步」：旧正则把 ``|已经|是`` 当
    # 计划动词，于是「该用户的申请已经提交」被判成草稿、整段删掉。
    # 收窄（只留既然/可能/会/要/可以/应该/需要/想）之后它们留下，
    # 而实测漏网的那段「他既然要去北京…」仍被抓着（见 _MEASURED_DRAFTS）。
    "该用户的申请已经提交，正在等待审批。",
    "他是本次出差的负责人，报销走他的申请单。",
]


@pytest.mark.parametrize("paragraph", _REAL_ANSWERS)
def test_real_answers_are_not_mistaken_for_drafts(paragraph: str) -> None:
    """★ 五段正经答复**一条都不许**被判成草稿。

    ⚠️ 这几条专挑判据的**近边界**：
    「我帮你查了一下」贴着 ``^我需要``/``^用户`` 那几条；
    「他的订单还没提交」贴着 ``_THIRD_PERSON_PLANNERS``（有「他」但后面
    不是计划性动词）；「这个政策涉及差旅标准」贴着「这是/涉及」那条正则。
    误判的代价是把真答案删掉，所以这条与上一条同等重要。
    """
    assert not _is_draft(paragraph), f"这段正经答复被误判成草稿：{paragraph!r}"


#: 2026-10-04 的分类语料：每一类都要有正例，并与 :func:`_is_draft` 对表。
_CLASSIFICATION_CASES: list[tuple[str, str | None]] = [
    # 内部实现名 → internal（用户可见的泄露，比独白更值得先修）
    ("已经让 policy_rag 去检索制度原文了。", "internal"),
    ("我先调用 check_travel_policy 核对一下。", "internal"),
    # 独白 / 过程叙述 → monologue
    # ⚠️ 别拿「当前处于 IDLE 阶段」当 monologue 的样例：它还含内部名
    # 「IDLE 阶段」，会被**先查内部名**的分支记成 internal（分类顺序有意如此）。
    ("我先查一下差标。", "monologue"),
    ("用户问的是差旅标准规定，不该凭常识推断。", "monologue"),
    # 不可疑 → None
    ("住宿标准：**单晚不超过 600 元**，凭发票报销。", None),
    ("该用户的申请已经提交，正在等待审批。", None),
]


def test_suspicious_paragraphs_are_classified_without_changing_the_predicate() -> None:
    """★★ 分类必须与 ``_is_draft`` **严格等价** —— 它只是分了类，不是第二套判据。

    ⚠️ 分类是给指标用的（``draft_paragraph_left_internal`` /
    ``..._monologue``，2026-10-04 从单条曲线拆开）。两边一旦分叉，会出现
    「判据说不是草稿、分类却说可疑」的段落 —— 这种矛盾**没有任何用户可见的
    症状**，只会在曲线上体现成看不懂的抖动，而曲线上看不出该修哪边。

    ⚠️ 与 :data:`_MEASURED_DRAFTS` / :data:`_REAL_ANSWERS` **对全表**：
    那两张表是实测原文，分类判据必须与它们逐条一致，不只是自造的样例。

    ⚠️ 变异：让 ``_classify_suspicious_paragraph`` 恒返回 ``None``（或恒返回
    ``monologue``、漏掉内部名分支）→ 等价性断言与 :data:`_CLASSIFICATION_CASES`
    立刻变红。
    """
    for paragraph in _MEASURED_DRAFTS + _REAL_ANSWERS:
        kind = _classify_suspicious_paragraph(paragraph)
        assert (kind is None) == (not _is_draft(paragraph)), (
            f"分类与判据分叉：{paragraph!r} -> {kind!r}"
        )
    for paragraph, expected in _CLASSIFICATION_CASES:
        assert _classify_suspicious_paragraph(paragraph) == expected, (
            f"分类结果不对：{paragraph!r}"
        )


def test_suspicious_classification_counts_sum_to_the_total() -> None:
    """★ 分类计数之和 == 旧口径的总数（拆分不许把段落算丢或算重）。

    ⚠️ 论据与 ``_ungrounded_amounts`` 那条同理：两份遍历会各自演化，
    用户看到「总数 3、分类加起来 2」的曲线时，两张图都不可信了。
    这里用一段**三类段落同现**的文本钉住它们的分工。

    ⚠️ 中间那两段（第 2、4 段）刻意是真内容：它们不能被算进任何一类。
    """
    text = (
        "我先查一下差标。\n\n"
        "住宿标准：单晚不超过 600 元。\n\n"
        "已经让 policy_rag 去检索制度原文了。\n\n"
        "凭发票报销。"
    )
    counts = _classify_suspicious_paragraphs(text)

    assert counts == {"monologue": 1, "internal": 1}, counts
    assert _count_suspicious_paragraphs(text) == sum(counts.values()) == 2


def test_the_script_monologue_prefixes_cover_every_gate_opener() -> None:
    """★★ 并发脚本的独白前缀必须**覆盖**闸门的开场词表（方向：闸门 ⊆ 脚本）。

    ⚠️ 两边形态不同（那边是正则、这边是字面前缀），做不到逐字相等，所以钉
    的是**方向性**：闸门认出草稿的样例，脚本也必须认出 —— 否则 `make
    concurrency` 会在闸门已经判红的地方报绿，而这条判据的全部价值就是
    「用户到底看到了什么」。反向不钉：脚本多认几条只会让告警更早出现。

    ⚠️ 反面也有人管：**脚本比闸门严**会让并发结论被模型文风左右，
    ``_MONOLOGUE_PREFIXES`` 的注释里记着英文那几条必须是计划性动词的
    理由（``I recommend…`` 是正经答复）。样例集里因此两条都要有。

    ⚠️ 变异：把 ``scripts/concurrency_test.py`` 的 ``_MONOLOGUE_PREFIXES``
    删回只剩「用户问/想/说/要」→ 本用例变红（``用户在问``/``系统已经``/
    ``当前处于`` 三个样例脚本侧判不出来）。
    """
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path as _Path
    from sys import modules as _modules

    path = _Path("scripts/concurrency_test.py")
    assert path.exists(), f"并发测试脚本不见了：{path}（同源约定无从校验）"
    spec = spec_from_file_location("_concurrency_test_monologue", path)
    module = module_from_spec(spec)
    _modules[spec.name] = module
    spec.loader.exec_module(module)

    #: 闸门两张表里的中文开场词各取一个**样例**（英文侧早已逐字对齐）。
    samples = (
        "用户问的是差旅标准规定。",
        "用户询问差旅标准规定。",
        "用户在问差旅标准。",
        "用户提到要报销住宿费。",
        "用户需要先确认标准。",
        "用户这里还没定日期。",
        "用户既然要报销就先查标准。",
        "现在我看清楚了标准。",
        "系统已经为您核对完成，请稍候。",
        "系统已记录本次申请。",
        "当前处于核对差标的状态。",
        "当前阶段需要先确认要素。",
        "当前我认为需要先查标准。",
    )
    for sample in samples:
        assert _is_draft(sample), f"样例本身不再是闸门判定的草稿，请更新样例：{sample!r}"
        assert module._monologue_reason(sample) is not None, (
            f"闸门判为草稿、脚本却判不出来（并发结论会在这一格报绿）：{sample!r}"
        )

    # 反向的已知代价：正经答复不许被脚本的**中文**前缀误伤。
    for answer in _REAL_ANSWERS:
        assert module._monologue_reason(answer) is None, (
            f"脚本把正经答复判成独白了（误报会污染并发结论）：{answer!r}"
        )


def test_classification_ignores_clean_text() -> None:
    """★ 全干净的正文不产生任何分类计数（空字典，不是 ``{"monologue": 0}``）。

    ⚠️ 上报侧按「有键才记指标」，零值键会让每条回复都多一次无意义的
    ``inc()`` —— 曲线被 0 淹没，看不见真正的那几条。
    """
    assert _classify_suspicious_paragraphs("住宿标准：单晚不超过 600 元。") == {}


def test_strip_never_removes_the_last_paragraph() -> None:
    """★★ 全是草稿时，**最后一段必须留下**。

    ⚠️ 这是模块文档写明的安全底线（``index < len(paragraphs) - 1``）。
    把最后一段也剥掉的后果不是「干净」，是**用户对着一片空白**——
    比留一段草稿严重得多。这条用例就是那条底线本身。
    """
    text = "我先查一下差标。\n\n用户问的是差旅标准。\n\n当前处于 IDLE 阶段。"
    cleaned, removed = _strip_drafts(text)

    assert removed == 2, f"应当只剥掉前两段，实际剥了 {removed} 段"
    assert cleaned == "当前处于 IDLE 阶段。", f"最后一段被剥掉了：{cleaned!r}"


def test_strip_never_touches_the_middle_of_the_text() -> None:
    """★ 段落**中间**夹的草稿不动（保守剥离）。

    ⚠️ 模块文档「能力边界」写明：删中间段落会改变行文结构，风险大于收益。
    这条把该取舍钉住 —— 若哪天有人把循环改成「过滤所有草稿段」，
    这段用例会红，提醒他去看文档里那条取舍的理由。

    ⚠️ 用例名 2026-10-04 改过：原来叫 ``..._only_touches_the_beginning``，
    而那天补上了**结尾**剥离（C3），「只动开头」不再是事实。
    变的只是名字，断言的取舍一字未动 —— 中间段落仍然不碰。
    """
    text = "住宿标准：单晚不超过 600 元。\n\n我先查一下差标。\n\n凭发票报销。"
    cleaned, removed = _strip_drafts(text)

    assert removed == 0
    assert cleaned == text, "中间的草稿段被动了，这超出了当前设计的能力边界"


def test_a_trailing_draft_sentence_does_not_take_the_answer_with_it() -> None:
    """★★★ 草稿句在**段尾**时答案必须留住 —— P5，2026-10-04 补的第二层救回。

    ⚠️ 实测形状（对抗验证逐字复现）：

        「您的订单已确认，出票时间是 10 月 12 日。我应该把行程发给您。」

    整段被 ``_META_COMMENTARY_MARKERS``（``我应该``）判成草稿，而救回函数
    只剥**开头**连续的草稿句 —— 第一句就是真答案，``cut == 0`` 返回空串，
    调用方于是把**整段**删掉：用户连「订单已确认」都看不到了。这是本模块
    最不能接受的失败形态（静默丢答案，且不触发重说）。

    ⚠️ 与 ``test_draft_and_answer_in_one_paragraph_keeps_the_answer`` 是
    同一类缺陷的两个方向：那条是草稿在前、答案在后，这条是答案在前、
    草稿在后。**只修一条不算修** —— 两条各由一个不同的函数兜住。

    ⚠️ 变异：删掉 ``_strip_drafts`` 段首循环里调用
    ``_strip_trailing_draft_sentences`` 的那两行 → 本用例立刻变红
    （输出里「订单已确认」整句消失）。
    """
    text = (
        "您的订单已确认，出票时间是 10 月 12 日。我应该把行程发给您。"
        "\n\n酒店差标是每晚 600 元。"
    )
    cleaned, removed = _strip_drafts(text)

    assert "您的订单已确认，出票时间是 10 月 12 日。" in cleaned, (
        f"段首的真答案被整段删掉了：{cleaned!r}"
    )
    assert "我应该把行程发给您。" not in cleaned, f"段尾的草稿句没剥掉：{cleaned!r}"
    assert removed >= 1, "剥离计数为 0，说明走的是「没变化」分支"


def test_a_pure_draft_paragraph_after_a_rescued_one_is_still_stripped() -> None:
    """★★ 救回一段之后，紧跟的纯草稿段仍要删 —— ``continue`` 而不是 ``break``。

    ⚠️ 这条钉的是**两处修复的相互作用**：上面的第二层救回把段首循环从
    「救回即收手」改成了「救回后继续」。若实现写成 ``break``，下面这个
    形状会把中间那段纯草稿留下来 —— 而那是一句带 ``check_travel_policy``
    的话，等于把内部工具名端给用户（基线本来会删掉它，属于**新引入**的
    泄露）。对抗验证对这个组合形状专门击穿过一次。

    ⚠️ 它同时钉住「段首区不止第一段」：救回的段落之后、第一段真内容之前的
    纯草稿段仍在段首区内。这对 ``test_strip_never_touches_the_middle_of_the_text``
    不是矛盾 —— 那条守的是**真答案之后**的段落。

    ⚠️ 变异：把 ``_strip_drafts`` 段首循环里的 ``continue`` 改回 ``break``，
    本用例变红（``check_travel_policy`` 重新出现在结果里）。
    """
    text = (
        "您的订单已确认，出票时间是 10 月 12 日。我应该把行程发给您。\n\n"
        "我先调用 check_travel_policy 核对一下。\n\n"
        "酒店差标是每晚 600 元。"
    )
    cleaned, removed = _strip_drafts(text)

    assert "check_travel_policy" not in cleaned, f"内部工具名漏给用户了：{cleaned!r}"
    assert cleaned == (
        "您的订单已确认，出票时间是 10 月 12 日。\n\n酒店差标是每晚 600 元。"
    ), f"剥离结果不对：{cleaned!r}"
    assert removed == 2, (
        f"应当剥掉「段尾草稿句」与「中间纯草稿段」各一处，实际 removed={removed}"
    )


def test_strip_removes_tail_drafts() -> None:
    """★★ 结尾的草稿段必须剥掉 —— C3，2026-10-04 补的能力。

    ⚠️ 补之前只剥开头，于是这种形状里最后那句独白**原样发给用户**：

        您的航班 CA1234 已出票，10 月 12 日 08:00 起飞。

        我先调用 check_travel_policy 核对一下。

    ⚠️ 而且这类残留**比开头的更严重**：正文已经发出去了，``_should_retry``
    的第一条（``state.emitted``）就把它挡在重说之外 —— 用户看到的独白
    **不会被修正**，刷新页面还在。开头残留至少还有一次重说的机会。
    """
    text = (
        "您的航班 CA1234 已出票，10 月 12 日 08:00 起飞。\n\n"
        "我先调用 check_travel_policy 核对一下。"
    )
    cleaned, removed = _strip_drafts(text)

    assert removed == 1, f"结尾的草稿段没被剥掉（removed={removed}）：{cleaned!r}"
    assert cleaned == "您的航班 CA1234 已出票，10 月 12 日 08:00 起飞。", (
        f"剥完的结果不对：{cleaned!r}"
    )


def test_strip_removes_tail_drafts_without_losing_the_opening() -> None:
    """★★ 两头都是草稿时，一刀剥干净，正文一段不少。

    ⚠️ 这条守的是「结尾剥离把开头剥离的成果覆盖掉」这类实现错误 ——
    两个循环共用一个 ``paragraphs`` 列表，边界（``index`` / ``end``）
    接错就会出现「剥了尾、把开头的草稿又拼回来」或者反过来。
    """
    text = (
        "我先查一下差标。\n\n"
        "酒店单晚不超过 600 元。\n\n"
        "我先调用 check_travel_policy 核对一下。"
    )
    cleaned, removed = _strip_drafts(text)

    assert removed == 2, f"应当剥掉首尾各一段（removed={removed}）：{cleaned!r}"
    assert cleaned == "酒店单晚不超过 600 元。", f"正文没能留住：{cleaned!r}"


def test_tail_stripping_still_never_empties_the_text() -> None:
    """★★ 结尾剥离**不许**把正文剥空 —— 它没有重说兜底。

    ⚠️ 与 :func:`test_strip_never_removes_the_last_paragraph` 是同一条底线
    的两个方向：那段守「全是草稿时留下最后一段」，这段守「结尾剥离至少
    留一段」。差别在于结尾这一侧**没有救回机制**：正文已经发出去了，
    ``_should_retry`` 不会为它重说，剥空等于用户对着空白。

    ⚠️ 三种形状都要试：单段、尾部连续两段草稿、开头也是草稿 ——
    ``end - 1 > index`` 这个边界在三种形状下分别被不同分支用到。
    """
    for text in (
        "我先调用 check_travel_policy 核对一下。",
        "您的航班 CA1234 已出票。\n\n我先查一下差标。\n\n用户问的是差旅标准。",
        "我先查一下差标。\n\n用户问的是差旅标准。",
    ):
        cleaned, _ = _strip_drafts(text)
        assert cleaned.strip(), f"剥离后用户对着空白了：{text!r} -> {cleaned!r}"


def test_a_normal_closing_line_is_not_a_draft() -> None:
    """★ 正经的收尾句不许被结尾剥离误伤。

    ⚠️ 结尾剥离的误报面比开头**更窄但更值钱**：开头的草稿被误伤，用户
    还能从后文读懂；结尾的收尾句（「需要我帮你订吗？」）是**交互的一部分**，
    剥掉它用户就不知道该接什么话了。

    ⚠️ 这几条专挑贴着判据的收尾句：有第二人称、有问句、有承诺但**不是**
    独白。它们必须原样留下（``removed == 0``）。
    """
    for text in (
        "酒店单晚不超过 600 元。\n\n需要我帮你订吗？",
        "酒店单晚不超过 600 元。\n\n有需要随时告诉我。",
        "酒店单晚不超过 600 元。\n\n这份标准从 10 月 1 日起生效。",
    ):
        cleaned, removed = _strip_drafts(text)
        assert removed == 0, f"正经的收尾句被剥掉了：{text!r} -> {cleaned!r}"
        assert cleaned == text, f"正文被动过：{text!r} -> {cleaned!r}"


def test_tail_status_phrases_are_not_tail_stripped_but_still_rejected() -> None:
    """★★ 「系统已经…」这类**状态句**：结尾不删，整段是它时仍拒（P4 降档）。

    ⚠️ 这条钉的是一个**降档**决定（强档 → 弱档，2026-10-04），两个方向都要
    断言，少一个方向下一个人就会把它改回去：

        1. 它在**结尾**出现时不许删 —— 那一处误删不可逆（正文已发、
           没有重说兜底），而「系统已经为您提交了申请」正是给用户的结论。
        2. 它**自己成篇**时必须仍被 ``_answer_problem`` 拒掉。
           ⚠️ 第二点是命门：若把它从表里**删掉**（而不是降档），
           用户会拿到一句「正在核对」当答复 —— 对抗验证实测过。

    ⚠️ 变异：把它挪回 ``_MONOLOGUE_OPENERS_STRONG`` → 第 1 组断言红；
    从 ``_MONOLOGUE_OPENERS_WEAK`` 里删掉 → 第 2 组断言红。
    """
    phrase = "系统已经为您核对完成，请稍候。"

    cleaned, removed = _strip_drafts(f"酒店差标是每晚 600 元。\n\n{phrase}")
    assert removed == 0, f"结尾的状态句被删了（不可逆）：{cleaned!r}"
    assert phrase in cleaned, f"结尾的状态句没保住：{cleaned!r}"

    assert not _is_draft(phrase, at_tail=True), "结尾剥离不许再认它（降档）"
    assert _is_draft(phrase), "段首剥离与闸门必须仍认得它"
    assert _answer_problem(phrase) is not None, (
        "整篇就是一句状态话术时必须被拒 —— 否则它会被当成答复发出去"
    )


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", True),
        ("   ", True),
        ("稍等", True),
        ("已让政策问答智能体检索制度原文，稍等。", True),
        # 有实质内容（含数字）的「稍等」是**合法答复**，不许拦。
        ("酒店标准 600 元，稍等我再查下班次。", False),
        # 没有等待词的空话不算占位符 —— 它会被「全是草稿」那条判据接住。
        ("按制度执行。", False),
        # ---- 2026-10-03 对抗审计逐条实测：下面五句**原本全部原样发给用户** ----
        # 老词表只有六个词且「请稍候」要求四字紧邻，礼貌插入语一拆就漏。
        ("请您稍候。", True),
        ("正在为您查询，马上回来。", True),
        ("正在处理中。", True),
        ("让我查一下，马上回来。", True),
        ("One moment please, checking.", True),
        # 「预计 N 分钟后」里的数字是**时间估计**，不是带回来的数据。
        # 用裸数字判据时它当场放行一句纯空话。
        ("稍等，预计 2 分钟后给你结果。", True),
        # ``'①'.isdigit()`` 是 True 而 ``'①'.isdecimal()`` 是 False ——
        # 用 ``isdigit`` 会让这个序号字符解除保险丝。
        ("稍等①", True),
        # ---- 反向：带等待词但**确实是答复**，一条都不许拦 ----
        # ⚠️ 这就是「残句保险丝」存在的理由。它含「请稍候」且没有数字，
        # 前两道判据（命中等待词 + 无结果数据）会把它判成占位。
        ("您的航班稍后起飞，请稍候到登机口", False),
        ("请稍候登机，您的座位是 12A。", False),
        ("我帮您把出差申请提交了，等待审批即可。", False),
    ],
)
def test_placeholder_judgment(text: str, expected: bool) -> None:
    """★ 占位符判据必须与 ``scripts/concurrency_test.py`` 同源。

    ⚠️ 两边判据不一致的后果是**测试说红线、线上放行**（模块文档写明
    ``WAITING_PHRASES`` 故意同源）。有数字的「稍等」是两边的**共同边界**：
    它必须被判为「不是占位符」，否则会拦掉真实答复。

    ⚠️ 这张表**两个方向都取样**，因为这道判据的两种错法代价相反：
    漏判（空话发给用户）是产品事故，误判（真答复被拦）会触发重说、
    重试用尽时用户拿到道歉话术。只测一个方向的话，把判据改成恒真或
    恒假都能让用例全绿。
    """
    assert _is_placeholder(text) is expected


def test_waiting_phrases_are_shared_with_the_concurrency_script() -> None:
    """★★ 运行时闸门与并发测试脚本的等待词表必须**逐字相同**。

    ⚠️ 两张表「故意同源」这句话，在 2026-10-03 之前**只写在注释里** ——
    没有任何东西守着它。而它漂移的后果是「测试说红线、线上放行」：
    并发脚本报出来的缺陷，线上闸门根本不会拦。更糟的是漂移可以**静默**
    发生（改一张表不会让任何用例变红）。

    ⚠️ 这条用例真的把脚本 import 进来比对，不是抄一份字面量再比 ——
    抄一份等于把「两处要同步」变成「三处要同步」。

    ⚠️ 顺带钉住保险丝本身（时间估计、剩余字数上限）：它们也是同源判据的
    一部分，只比词表的话，把某一侧的 ``_has_result_data`` 换回裸数字判据
    照样全绿。
    """
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path as _Path
    from sys import modules as _modules

    from src.orchestration.reply_guard import (
        _DELEGATION_MARKERS,
        _PLACEHOLDER_MAX_RESIDUE,
        _TIME_ESTIMATE_PATTERN,
        WAITING_PHRASES,
    )

    path = _Path("scripts/concurrency_test.py")
    assert path.exists(), f"并发测试脚本不见了：{path}（同源约定无从校验）"
    spec = spec_from_file_location("_concurrency_test_contract", path)
    module = module_from_spec(spec)
    _modules[spec.name] = module
    spec.loader.exec_module(module)

    assert tuple(module._WAITING_PHRASES) == tuple(WAITING_PHRASES), (
        "两张等待词表已经漂移：\n"
        f"  只在 reply_guard：{sorted(set(WAITING_PHRASES) - set(module._WAITING_PHRASES))}\n"
        f"  只在并发脚本：{sorted(set(module._WAITING_PHRASES) - set(WAITING_PHRASES))}"
    )
    assert module._PLACEHOLDER_MAX_RESIDUE == _PLACEHOLDER_MAX_RESIDUE, (
        "「剩余字数上限」两边不一致 —— 测试与线上会各说各话。"
    )
    # ⚠️ 转交措辞表同样要逐字相同。它比等待词表**更容易漂移**：新写法的
    # 出现频率取决于模型的文风，而补一条只会发生在「有人刚好碰到」的时候 ——
    # 于是「只补了一边」是常态，两边判据不一致也就成了常态。
    assert tuple(module._DELEGATION_MARKERS) == tuple(_DELEGATION_MARKERS), (
        "两张转交措辞表已经漂移：\n"
        f"  只在 reply_guard：{sorted(set(_DELEGATION_MARKERS) - set(module._DELEGATION_MARKERS))}\n"
        f"  只在并发脚本：{sorted(set(module._DELEGATION_MARKERS) - set(_DELEGATION_MARKERS))}"
    )
    # 时间估计必须**两边都**被排除在「结果数据」之外。
    for sample in ("稍等，预计 2 分钟后给你结果。", "Please wait, 30 seconds."):
        assert module._has_result_data(sample) is False, (
            f"并发脚本仍把 {sample!r} 的时间估计当成结果数据。"
        )
        assert _TIME_ESTIMATE_PATTERN.search(sample), (
            f"运行时的保险丝没覆盖 {sample!r} 里的时间估计。"
        )


#: 跨实现一致性用例的**共享判例表**：``(文本, 是不是占位)``。
#:
#: ⚠️ 这张表比「表本身逐字相同」更强。逐字比的是**实现**，它挡不住
#: 「两边表格一样、但用表格的代码不同」—— 而那正是 2026-10-04 实测到的
#: 漂移形态：脚本侧只写了「残留短」一条分支，没跟上线上的
#: 「或者本身是草稿形状」，于是同一句话在脚本里判绿、在线上被拦。
#: 表一样、行为不一样，逐字比对是看不出来的。
#:
#: ⚠️ 每一条都写清它守的是什么，因为这张表将来还要长 —— 没有理由的
#: 判例会在下一次「这条为什么红」时被直接删掉。
_PLACEHOLDER_AGREEMENT_CASES: tuple[tuple[str, bool], ...] = (
    # —— 真·占位（用户什么都没拿到）——
    ("已让政策问答智能体检索制度原文，稍等。", True),
    ("已让政策问答智能体检索制度原文，稍等。 等待政策检索结果中。", True),
    # ⚠️ 这一条是 2026-10-04 实测的**两边同时漏判**：残留 14 字、不带「已让」
    # 而是「已经交给」，两边的字数阈值和草稿形状都盖不住它。
    ("已经交给政策问答智能体了，请稍候。", True),
    ("等待政策检索结果中。", True),
    ("稍等，预计 2 分钟后给你结果。", True),
    ("One moment please, checking.", True),
    # —— 假·占位（正经答复，只是顺带说了句等）——
    # ⚠️ 这一条是**假阳性基准**：它 13 字，比上面第一条真占位的 16 字还短，
    # 所以「调阈值」永远分不开这两类 —— 想动阈值的人先看这里。
    ("您的航班稍后起飞，请稍候到登机口", False),
    ("请稍候登机，您的座位是 12A。", False),
    ("我帮您把出差申请提交了，等待审批即可。", False),
    ("住宿费每晚上限 600 元。", False),
)


@pytest.mark.parametrize(("text", "is_placeholder"), _PLACEHOLDER_AGREEMENT_CASES)
def test_the_gate_and_the_script_agree_on_every_case(
    text: str,
    is_placeholder: bool,
) -> None:
    """★★★ 同一句话，运行时闸门与并发脚本必须**判得一样**。

    ⚠️ 这是本文件里唯一一条**跨实现**的判据，也是唯一能发现「两边表格
    一致、代码不一致」的判据。漂移的两种方向各有一次真实事故：
    「脚本判红、线上放行」让脚本报出来的缺陷修不掉（因为线上压根不拦）；
    「脚本判绿、线上拦截」更糟 —— 它让线上对着**正确答案**触发重说，
    重试用尽后用户拿到的是兜底道歉。

    ⚠️ 断言必须**双向**（``is_placeholder`` 为真和为假都要有判例）。
    只测真的一侧，一个「永远返回 True」的实现会全绿 —— 而它会把
    「您的航班稍后起飞，请稍候到登机口」这类正经答复全部拦下。
    """
    from importlib.util import module_from_spec, spec_from_file_location
    from pathlib import Path as _Path
    from sys import modules as _modules

    from src.orchestration.reply_guard import _is_placeholder

    path = _Path("scripts/concurrency_test.py")
    spec = spec_from_file_location("_concurrency_test_agreement", path)
    module = module_from_spec(spec)
    _modules[spec.name] = module
    spec.loader.exec_module(module)

    gate = _is_placeholder(text)
    script = module._placeholder_reason(text) is not None

    assert gate is is_placeholder, (
        f"运行时闸门对 {text!r} 判成了 {gate}，判例表要求 {is_placeholder}。"
    )
    assert script is is_placeholder, (
        f"并发脚本对 {text!r} 判成了 {script}，判例表要求 {is_placeholder}。"
    )
    assert gate is script, (
        f"{text!r}：运行时闸门判 {gate}，并发脚本判 {script} —— "
        "两边判据已经漂移，脚本的结论不再代表线上的行为。"
    )


@pytest.mark.parametrize(
    "text",
    [
        "我先查一下差标。\n你的航班 CA1234 已出票。",
        "让我先查一下差标。\n\n住宿标准：单晚不超过 600 元。",
    ],
)
def test_announced_action_fuse_only_looks_at_its_own_sentence(text: str) -> None:
    """★★ 「宣布动作」的保险丝只看**首句**，且「让我…」也算宣布。

    ⚠️ 两个缺陷各有实测原文，别把它们合并成一个「宽松版保险丝」：

    1. 保险丝原本扫**整段**，于是
       ``我先查一下差标。\\n你的航班 CA1234 已出票。`` 里的 ``CA1234``
       （来自**另一句**）把整条判据解除，段首那句空话原样留给用户。
    2. ``_ANNOUNCED_ACTION_OPENERS`` 原本只有第一人称 ``我``，
       ``让我先查一下差标。`` 一个人称换字的同款空话整个漏掉。

    ⚠️ 这里只断言 ``_strip_drafts`` **有没有动手**（``removed >= 1``），
    不断言剥完的正文 —— 正文取决于段落结构，是另一条用例的事。
    """
    _, removed = _strip_drafts(text)
    assert removed >= 1, f"这段开头的空话没有被剥掉：{text!r}"


@pytest.mark.parametrize(
    "text",
    [
        "我先确认了订单和发票信息，都齐全。",
        "我先看过了你的订单，状态是已提交。",
    ],
)
def test_completed_actions_are_real_answers(text: str) -> None:
    """★★ 用完成体说的「我先…了」是**答复**，不是草稿。

    ⚠️ 判据只问「是不是以『我(先)查…』开头」，而「我先确认**了**…」与
    「我先确认**一下**…」的前缀一模一样。前者是完整答复，后者是空话。
    实测把前者整段判成草稿 → ``_answer_problem`` 返回 ``monologue`` →
    触发重说 → 重试用尽时用户拿到的是兜底道歉而不是答案。

    ⚠️ 只用「动词后紧跟 了/过」做代理判据，不做语义理解 ——
    残留风险（未完成体、无数字的真答复仍会被误判）已经在
    ``_announces_an_unfinished_action`` 的 docstring 里如实记下。
    """
    assert not _is_draft(text), f"这段完成陈述被误判成草稿：{text!r}"
    assert _answer_problem(text) is None, f"这段被整段拒答：{text!r}"


def test_draft_and_answer_in_one_paragraph_keeps_the_answer() -> None:
    """★★★ 草稿句与真答案**同段**时，答案必须留住。

    ⚠️ 这是本模块最不能接受的失败形态：**静默丢答案**。
    实测原文（一句英文草稿 + 一句中文答案，中间只有空格没有空行）：

        "I'll wait for the policy agent's report. 住宿标准：单晚不超过 600 元。"

    段粒度的剥离把整段当草稿删掉，用户只看到后面那段「凭发票报销。」——
    而且因为剩下的段落能过 ``_answer_problem``，**连重说都不会触发**。

    ⚠️ 断言的是**答案在结果里**，不是「剥掉了几句」：计数是内部细节，
    用户在意的只有「那句话还在不在」。
    """
    text = (
        "I'll wait for the policy agent's report. 住宿标准：单晚不超过 600 元。"
        "\n\n凭发票报销。"
    )
    cleaned, removed = _strip_drafts(text)

    assert "住宿标准：单晚不超过 600 元。" in cleaned, (
        f"真答案在剥离时被丢掉了：{cleaned!r}"
    )
    assert "I'll wait" not in cleaned, f"草稿句没剥掉：{cleaned!r}"
    assert removed >= 1, "剥离计数为 0，说明走的是「没变化」分支"


@pytest.mark.parametrize(
    "text",
    [
        "I'll wait for the policy agent's report. 住宿标准：单晚不超过 600 元。",
        "我先查一下差标。\n你的航班 CA1234 已出票。",
        "价格 1.5 元一晚。第二句。",
        "Line one\nLine two\n",
    ],
)
def test_sentence_splitting_round_trips(text: str) -> None:
    """★ 切句必须**无损**：拼回去等于原文。

    ⚠️ 第一版用带捕获组的 ``re.split``，把标点切成了独立片段 ——
    于是「剥掉开头连续的草稿句」会留下一个孤零零的句号，
    用户看到的正文以「。」开头（``test_leading_drafts_are_stripped_with_
    synthesized_events`` 实测抓到）。无损性是这条切分器的**硬约束**：
    剥掉若干句之后要能原样拼回剩下的部分。

    ⚠️ ``1.5`` 那一条钉的是英文句号的边界：只在小数点**后面没有空白**时
    不切分，否则价格会被从中间切开。
    """
    from src.orchestration.reply_guard import _split_sentences

    pieces = _split_sentences(text)
    assert "".join(pieces) == text, f"切句有损：{pieces!r}"
    assert all(piece.strip() for piece in pieces), f"切出了空白片段：{pieces!r}"


def test_ungrounded_amounts_reports_only_what_is_missing() -> None:
    """★ 无依据金额只上报、判据要认得出「600 与 600.0 是同一个数」。

    ⚠️ 这里用实测那次编造当样本：工具返回 600，模型写 500。
    漏报会让这条指标失去意义（它存在的全部价值就是把「模型编数」
    变成能画曲线的数字，见模块文档「能力边界」）。
    """
    tool_text = '{"max_hotel_price": 600.0}'

    assert _ungrounded_amounts("单晚上限 500 元。", tool_text) == ["500"]
    assert _ungrounded_amounts("单晚上限 600 元。", tool_text) == []
    assert _ungrounded_amounts("单晚上限 ¥600。", tool_text) == []
    # 用户自己给的金额 / 模型算出来的合计都会误报 —— 这是已知代价，
    # 所以它只上报不拦截。此处把误报**显式**钉住，防止有人拿它去拦截。
    assert _ungrounded_amounts("两晚合计 1200 元。", tool_text) == ["1200"]


# ==============================================================================
# 三、中间件：脚本化事件流
# ==============================================================================


def test_clean_answer_passes_through_with_original_events() -> None:
    """★ 干净的答复必须**原样**透传（复用原事件、保住 block id）。

    ⚠️ 断言的是对象**同一性**（``is``），不是「文本相等」：
    守卫在「剥离后没变化」时走的是 ``list(state.round_events)``，
    为的是让前端拿到的 block id 与模型产出的 id 一致（前端按 id 归并
    文本分片）。若哪天改成无条件重新合成，文本看着一样，但块 id 会变，
    前端的分片归并就会出问题 —— 这条用例专门盯着这件事。
    """
    events = [_reply_start(), _call_start(), *_text("住宿标准 600 元。"), _call_end(), _reply_end()]
    kept = [e for e in events if isinstance(e, TextBlockDeltaEvent)]
    agent = _FakeAgent()

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert _visible(out) == "住宿标准 600 元。"
    assert [id(e) for e in out if isinstance(e, TextBlockDeltaEvent)] == [
        id(e) for e in kept
    ], "干净答复被重新合成了事件（block id 会变）"
    assert agent.state.context_calls == [], "干净答复不该触发重试"


def test_tool_round_text_is_dropped_but_tool_events_pass() -> None:
    """★★ 工具轮里的文字丢掉，**工具调用事件照常透传**。

    ⚠️ 两件事必须同时成立：
    1. 「有工具调用 ⇒ 这轮不是最终答复」是**结构性事实**（框架只在没有
       工具调用时才退出 ReAct，见模块文档），所以文字该丢；
    2. 但工具调用事件**不能丢** —— 它既是用户可见的思考链，也是服务端
       记录「这条回复做过什么」的依据。只测第 1 条会把「整轮丢弃」
       这种更省事的错误实现放过去。

    这个事件流就是实测 A 类缺陷的形状：先写一句打算，再调工具。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("我先查一下差标。"),
        _tool_call("check_travel_policy"),
        _call_end(),
        _call_start(),
        *_text("住宿标准：单晚不超过 600 元。", block_id="block-2"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={"check_travel_policy": _FakeTool(is_read_only=True)},
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert _visible(out) == "住宿标准：单晚不超过 600 元。", (
        f"工具轮的前言漏进了可见正文：{_visible(out)!r}"
    )
    assert any(isinstance(e, ToolCallStartEvent) for e in out), (
        "工具调用事件被丢掉了 —— 思考链会缺一环"
    )


def test_confirmation_round_text_is_kept() -> None:
    """★★ 含**非只读**工具的那一轮，文字必须保留。

    ⚠️ 这是「工具轮文字 = 草稿」的**唯一例外**，也是提示词明确要求的：
    提交申请前要把事由、金额、日期复述给用户确认。判据用工具自身的
    ``is_read_only``，而不是维护一张工具名单（见模块文档「判定规则」）。

    实测过一次「按名单判」的错法：框架的团队工具自报
    ``is_read_only=True``，用名单会漏掉它们；用元数据则自动正确。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("请确认：事由「客户拜访」，金额 1200 元，日期 10 月 8 日。"),
        _tool_call("submit_approval"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(tools={"submit_approval": _FakeTool(is_read_only=False)})

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert "请确认" in _visible(out), (
        f"确认复述被剥掉了，用户在确认框上看不到事由与金额：{_visible(out)!r}"
    )


def _real_tool_stop() -> Any:
    """造一个**真实的** ``ToolStop`` 对象。

    ⚠️ 走真实的构造路径（``BackgroundTaskManager.list_tools``）而不是直接
    ``ToolStop(...)``：前者才是框架注入的那一条路
    （``app/_service/_toolkit.py`` 调的就是它），将来框架换实现，
    这条路径会跟着变，手写的构造调用不会。

    ⚠️ 管理器只存一个 bus 引用，``list_tools`` 不碰它，所以 ``object()``
    当替身足够 —— 别为此起一个真的 ``RedisMessageBus``。
    """
    from agentscope.app._manager._background_task_manager import (
        BackgroundTaskManager,
    )

    manager = BackgroundTaskManager(object())  # type: ignore[arg-type]
    return asyncio.run(manager.list_tools("session-for-tool-stop-test"))[0]


@pytest.mark.parametrize(
    "tool",
    [
        # ⚠️ 用**真实的框架工具对象**，不是替身。这条用例要证明的正是
        # 「框架工具自报的 is_read_only 与我们要的语义不一致」——
        # 用一个 is_read_only=False 的替身，等于把结论当成了前提。
        pytest.param(TaskList(), id="TaskList"),
        pytest.param(TaskCreate(), id="TaskCreate"),
        pytest.param(TaskGet(), id="TaskGet"),
        pytest.param(TaskUpdate(), id="TaskUpdate"),
        pytest.param(
            ResetTools(groups=[], response_template="{groups}"),
            id="reset_tools",
        ),
        # ⚠️ ToolStop 由 ``BackgroundTaskManager.list_tools()`` **无条件**挂载，
        # 自报 ``is_read_only=False``，与上面几个同属「框架机制冒充写工具」。
        # 实测（2026-10-03）它漏在名单外时，``_round_needs_recitation``
        # 返回 True，一轮「我先把后台那个还在跑的任务停掉。」被原样发给用户。
        pytest.param(_real_tool_stop(), id="ToolStop"),
    ],
)
def test_internal_framework_tools_do_not_keep_process_narration(
    tool: Any,
) -> None:
    """★★ 框架的规划/元工具**不是**「需要用户确认」，本轮文字必须剥掉。

    ⚠️ 实测（2026-10-03，用真实 ``Toolkit`` 装配）：

        TaskCreate / TaskList / TaskGet / TaskUpdate  is_read_only=False
        reset_tools                                   is_read_only=False

    它们自报 ``False`` 的含义是「会改动内部状态」，而不是「这一轮的文字是
    给用户看的复述」。守卫早先只有 ``is_read_only`` 一条判据，于是模型
    「我先建个任务清单，再让交通专家去查」这一类**过程独白被原样发给用户**
    —— 而消灭这类文本正是守卫存在的全部理由。

    ⚠️ 判据修正后必须仍然**保留**真正的写工具那一轮（见
    :func:`test_confirmation_round_text_is_kept`）：两条用例是配对的，
    只测一边要么漏掉独白、要么把确认复述也剥掉。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("我先把这件事拆成几个任务。"),
        _tool_call(tool.name),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(tools={tool.name: tool})

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert "我先把这件事拆成几个任务。" not in _visible(out), (
        f"{tool.name} 那一轮的过程独白被发给了用户：{_visible(out)!r}"
    )


def test_a_write_tool_still_wins_in_a_mixed_round() -> None:
    """★ 同一轮里既有内部工具又有写工具时，**保留**复述。

    ⚠️ 名单判断写成「见到内部工具就直接 return False」是很容易犯的错：
    那样一轮里同时调 ``TaskList`` 与 ``submit_approval`` 时，确认复述
    会被剥掉 —— 用户在确认框上看不到事由与金额，而这是提示词明确要求
    必须复述的。判据要落在「有没有真正的写工具」上。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("请确认：事由「客户拜访」，金额 1200 元。"),
        _tool_call("TaskList"),
        _tool_call("submit_approval"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={
            "TaskList": TaskList(),
            "submit_approval": _FakeTool(is_read_only=False),
        },
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert "请确认" in _visible(out), (
        f"确认复述被内部工具带偏着剥掉了：{_visible(out)!r}"
    )


#: 差标工具名。⚠️ 与守卫的 :data:`_POLICY_LIMIT_TOOLS` 同源，下面第一条
#: 断言会在名字对不上时先炸 —— 否则用例会静默走进「本轮没调过差标工具」
#: 的提前返回，接地闸门根本没跑，红/绿都不说明任何事。
_POLICY_TOOL = "check_travel_policy"


def _mixed_write_round(body: str) -> str:
    """跑一轮「差标只读 + 提交写」的混合工具轮，返回用户可见的文本。

    Args:
        body (`str`): 本轮模型写的文字。

    Returns:
        `str`: 过滤后真正发给用户的文本（空串表示一个字都没发）。

    ⚠️ 事件顺序照着真实链路摆：文字 → 只读工具 → 工具返回 → 写工具。
    工具返回里的 ``600`` 是接地闸门的基准值，缺了它闸门会因为
    「本轮没有工具文本」而整体短路（见 ``_ungrounded_limit_offenders``
    的两个先决条件）。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text(body),
        _tool_call(_POLICY_TOOL),
        _tool_result('{"hotel_limit_per_night": 600}'),
        _tool_call("submit_approval"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={
            _POLICY_TOOL: _FakeTool(is_read_only=True),
            "submit_approval": _FakeTool(is_read_only=False),
        },
    )
    return _visible(asyncio.run(_run(ReplyGuardMiddleware(), events, agent)))


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            "我先调用 check_travel_policy 核对，酒店差标是 500 元/晚。",
            id="过程独白",
        ),
        pytest.param(
            "请确认：事由「客户拜访」，酒店差标 500 元/晚，金额 1200 元。",
            id="编造差标",
        ),
    ],
)
def test_confirmation_round_is_not_a_free_pass(body: str) -> None:
    """★★ 写工具轮**不免检**：剥离与接地闸门照样跑。

    ⚠️ 实测缺陷（2026-10-04，已现场复现）：守卫早先一进「本轮有需要用户
    确认的工具」分支就 ``return list(state.round_events)`` —— 剥离与两个
    闸门全跳过。于是一轮里同时调 ``check_travel_policy``（只读）与
    ``submit_approval``（写）时，工具返回 600、正文却写
    「我先调用 check_travel_policy 核对，酒店差标是 500 元/晚。」
    编造的 500 与内部工具名**一起**发给了用户。

    ⚠️ 根因是把「这一轮的文字**可能**是给用户看的复述」当成了
    「这一轮的文字**一定**是对的」。前者是保留的理由，后者不是。

    ⚠️ 与 :func:`test_confirmation_round_text_is_kept` / 下面那条
    ``test_grounded_confirmation_round_still_reaches_the_user`` 是配对的：
    只测一边的修复方向是反的 —— 要么把复述全剥掉，要么把编造全放行。
    """
    assert _POLICY_TOOL in _POLICY_LIMIT_TOOLS, (
        f"差标工具名变了（{_POLICY_TOOL} 不在 {_POLICY_LIMIT_TOOLS} 里），"
        "本用例的前提已失效，改名字时请一并更新。"
    )

    assert _mixed_write_round(body) == "", (
        f"写工具轮绕过了剥离与接地闸门，编造/独白发给了用户："
        f"{_mixed_write_round(body)!r}"
    )


def test_grounded_confirmation_round_still_reaches_the_user() -> None:
    """★ 有依据的确认复述**必须**照常发给用户 —— 上一条的镜像。

    ⚠️ 少了这一条，「把例外轮整轮丢掉」也能让上面两条变绿，
    而那样做的后果正是提示词明确要求的东西没了：用户看不到事由与金额。
    """
    text = _mixed_write_round(
        "请确认：事由「客户拜访」，酒店差标 600 元/晚，金额 1200 元。"
    )

    assert "请确认" in text, f"合规的确认复述被拦下了：{text!r}"
    assert "600" in text, f"工具返回的 600 没有出现在复述里：{text!r}"


def test_the_draft_half_of_a_confirmation_round_is_stripped() -> None:
    """★ 例外轮里「草稿 + 复述」同轮时：剥草稿、留复述。

    ⚠️ 这是最常见的真实形状 —— 模型先自言自语一句「我先核对一下」，
    再写正经的确认复述。整轮放行会把独白发出去，整轮丢弃会把复述弄没，
    两种做法都错。
    """
    text = _mixed_write_round(
        "我先调用 check_travel_policy 核对一下。\n\n"
        "请确认：事由「客户拜访」，酒店差标 600 元/晚，金额 1200 元。"
    )

    assert "请确认" in text, f"复述连同草稿一起被剥掉了：{text!r}"
    assert "我先调用" not in text, f"草稿没能剥掉：{text!r}"


@pytest.mark.parametrize(
    "tool",
    [
        # ⚠️ 用真实的工作区工具对象（``agentscope.tool`` 的六件套），
        # 与上面那条用例同款理由：替身的属性会把结论当成前提。
        pytest.param(Write(), id="Write"),
        pytest.param(Edit(), id="Edit"),
        pytest.param(Bash(), id="Bash"),
    ],
)
def test_workspace_write_tools_still_keep_their_round_text(
    tool: Any,
) -> None:
    """★★ 工作区写工具那一轮的文字**必须保留** —— 与框架机制工具刻意分岔。

    ⚠️ ``Write`` / ``Edit`` / ``Bash`` 自报 ``is_read_only=False``，
    但这个 ``False`` 与 ``TaskList`` 的 ``False`` 含义**不同**：
    它们是真的会改文件系统的写操作，那一轮的文字很可能是
    「文件已生成，路径是 …」这类用户可见的交代。守卫的取舍一贯是
    「宁可留一段独白，不可删一段正文」，所以它们**不**进
    :data:`INTERNAL_TOOLS`（收窄表 ``LANE_HIDDEN_TOOLS`` 里另有它们，
    那是另一个判据：一次性业务查询用不上文件与进程）。

    ⚠️ 这条用例与 :func:`test_internal_framework_tools_do_not_keep_process_narration`
    是一对：只测一边，要么把框架独白漏给用户，要么把真正的交付说明剥掉。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("文件已生成：行程单.pdf。"),
        _tool_call(tool.name),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(tools={tool.name: tool})

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert "文件已生成" in _visible(out), (
        f"{tool.name} 那一轮的交付说明被当成草稿剥掉了：{_visible(out)!r}"
    )


@pytest.mark.parametrize(
    "tool",
    [
        pytest.param(Glob(), id="Glob"),
        pytest.param(Grep(), id="Grep"),
        pytest.param(Read(), id="Read"),
    ],
)
def test_workspace_read_tools_do_not_keep_their_round_text(
    tool: Any,
) -> None:
    """★ 工作区里**只读**的三个（Glob/Grep/Read）照旧剥掉本轮文字。

    ⚠️ 与上一条配对，钉住分岔的**边界落在** ``is_read_only`` 上，
    而不是「凡工作区工具都保留」：只读工具那一轮的文字同样是过程独白
    （「我先看看目录里有什么」），留在正文里没有任何人要。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("我先看看工作目录里有什么。"),
        _tool_call(tool.name),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(tools={tool.name: tool})

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert "我先看看工作目录里有什么。" not in _visible(out), (
        f"{tool.name} 那一轮的过程独白被发给了用户：{_visible(out)!r}"
    )


def test_unknown_tool_is_treated_as_read_only() -> None:
    """★ 查不到的工具按**只读**处理（即允许剥离）。

    ⚠️ 取舍的理由是不对称：查不到的多半是框架的团队/元工具（它们自报只读，
    本来就该剥），而真正的写操作工具是我们自己注册的、一定查得到。
    判错的后果也不对称 —— 多剥一段复述，用户仍能在确认框上看到事由；
    反过来（把工具轮草稿全留下）是**必然**的缺陷。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("我先让团队去查一下。"),
        _tool_call("AgentCreate"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()  # 工具集为空 → get_tool 返回 None

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert "我先让团队去查一下。" not in _visible(out)


def test_leading_drafts_are_stripped_with_synthesized_events() -> None:
    """★★ 正文轮开头的草稿被剥掉，且合成事件的 ``reply_id`` 必须正确。

    ⚠️ ``reply_id`` 错了的话，服务端 ``Msg.append_event`` 会**整条跳过**
    并只记一条 warning（模块文档「两条已核实的框架事实」第 1 条）——
    症状是「守卫剥完草稿，用户也什么都收不到」，而且不报错。
    所以这条既查文本，也查 id，还查 START/DELTA/END 三步齐全
    （缺 START 则 DELTA 找不到块，同样被静默丢弃）。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("我先查一下差标。\n\n住宿标准：单晚不超过 600 元。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))
    synthesized = [
        e for e in out if isinstance(e, (TextBlockStartEvent, TextBlockDeltaEvent, TextBlockEndEvent))
    ]

    assert _visible(out) == "住宿标准：单晚不超过 600 元。"
    assert [type(e).__name__ for e in synthesized] == [
        "TextBlockStartEvent",
        "TextBlockDeltaEvent",
        "TextBlockEndEvent",
    ], "合成的文本事件不完整（缺 START 会让服务端静默丢弃 DELTA）"
    assert {e.reply_id for e in synthesized} == {REPLY_ID}, (
        "合成事件的 reply_id 与回复不一致，服务端会跳过它们"
    )


def test_tail_drafts_are_stripped_and_the_reply_still_ends() -> None:
    """★★ 正文轮**结尾**的草稿也要剥掉，且这一轮**不许**被吞掉重说。

    ⚠️ C3 的端到端版本。三件事一起查：

    1. 结尾的独白不出现在用户可见文本里；
    2. 正文照常发出（合成事件完整、``reply_id`` 正确）；
    3. ``ReplyEndEvent`` **留在输出里** —— 正文已经发出，重说会把同一段
       答复在界面上说两遍（``_should_retry`` 的第一条挡的就是它）。
       反过来，若实现改成「剥完发现 emitted 没置位」而吞掉结束事件，
       用户会看到正文 + 一遍重说，这条会红。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text(
            "住宿标准：单晚不超过 600 元。\n\n"
            "我先调用 check_travel_policy 核对一下。"
        ),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert _visible(out) == "住宿标准：单晚不超过 600 元。", (
        f"结尾的独白漏给了用户：{_visible(out)!r}"
    )
    assert any(isinstance(e, ReplyEndEvent) for e in out), (
        "正文已经发出，结束事件却被吞了 —— 用户会听到两遍答复"
    )


def test_draft_only_reply_is_swallowed_and_retried() -> None:
    """★★ 整轮只有草稿时，``ReplyEndEvent`` 必须被**吞掉**并要求重说。

    ⚠️ 三件事一起查，缺一不可：
    1. ``ReplyEndEvent`` 不在输出里 —— 这就是「吞」，框架因此再跑一轮；
    2. ``append_context`` 被调用，且塞进去的是纠正指令 —— 框架吞掉之后走
       ``Reasoning(hint=None)``，**不会**告诉模型刚才发生了什么，不自己塞
       提示模型只会原样再犯（模块文档「重试机制」第 2 条）；
    3. 用户此时**什么都没收到**（草稿不能漏出去）。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("用户问的是差旅标准，我不应该自己凭常识推断金额。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert _visible(out) == "", f"草稿漏给用户了：{_visible(out)!r}"
    assert not any(isinstance(e, ReplyEndEvent) for e in out), (
        "没有吞掉 ReplyEndEvent，模型不会重说 —— 用户将一文不得"
    )
    assert len(agent.state.context_calls) == 1, "重试前没有把纠正指令塞进上下文"
    _, blocks = agent.state.context_calls[0]
    hint = blocks[0]
    assert "没有发给用户" in hint.hint[0].text, (
        f"纠正指令没有说清「用户什么都没看到」：{hint.hint[0].text!r}"
    )


def test_retry_is_skipped_when_already_answered() -> None:
    """★ 已经答过就不再重说（否则界面上会出现两段回答）。

    ⚠️ 「答过」的判据是 ``state.emitted``：工具轮里被丢掉的文字**不算**
    答过（它本来就不该给用户看），只有真正发出去的正文才算。
    这条用例构造的是「正文轮已发出 → 结束时不该再吞」。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("住宿标准：单晚不超过 600 元。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert any(isinstance(e, ReplyEndEvent) for e in out), "已经答过还吞了结束事件"
    assert agent.state.context_calls == []


@pytest.mark.parametrize(
    "reason",
    [
        ReplyFinishedReason.INTERRUPTED,
        ReplyFinishedReason.EXCEED_MAX_ITERS,
        ReplyFinishedReason.ERROR,
    ],
)
def test_non_completed_endings_are_never_swallowed(
    reason: ReplyFinishedReason,
) -> None:
    """★★ 中断 / 超迭代 / 出错三种结束**绝不允许**吞（哪怕整轮只有草稿）。

    ⚠️ 吞非 COMPLETED 的结束会破坏取消与错误传播 —— 用户点了停止却
    停不下来，或者一次报错被伪装成正常重试。``on_reply`` 的协议文档
    写明了只有 COMPLETED 可以吞（模块文档「重试机制」第 1 条）。

    这条是**回归护栏**：为了修「草稿漏出」而把条件放宽成「只要没答过就吞」
    是很自然的错误，而它的症状只在中断/报错时才出现，人工测很难覆盖。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("用户问的是差旅标准，我不应该自己凭常识推断。"),
        _call_end(),
        _reply_end(reason),
    ]
    agent = _FakeAgent()

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert any(isinstance(e, ReplyEndEvent) for e in out), (
        f"{reason} 的结束被吞了 —— 取消/错误传播会被破坏"
    )
    assert agent.state.context_calls == []


def test_retry_is_skipped_when_iteration_budget_is_short() -> None:
    """★★ 迭代余量不足时**不重试**，直接走兜底。

    ⚠️ 这条是**对抗性评审挑出来的**：重试会消耗 ReAct 迭代，而 ``cur_iter``
    一旦超过上限，框架会把结束原因改成 ``EXCEED_MAX_ITERS`` —— 一次本来
    **成功**的回复会被拖成**失败**（而 :mod:`src.chains.events` 会把失败
    的进行中任务标记出来）。为了修一句不好听的话把整次请求搞失败，
    是纯粹的自伤，所以宁可退回兜底。

    断言的重心是「**没有**吞掉结束事件」：兜底文本照样发给用户，
    但这一轮必须以框架原本的结论收尾。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("用户问的是差旅标准，我不应该自己凭常识推断。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(cur_iter=49, max_iters=50)  # 只剩 1 轮，低于 3 的余量线

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert any(isinstance(e, ReplyEndEvent) for e in out), (
        "迭代余量不足还吞了结束事件，会把成功的回复拖成 EXCEED_MAX_ITERS"
    )
    assert agent.state.context_calls == []
    assert _visible(out).strip(), "既没重试也没兜底 —— 用户会面对空白"
    assert "没能整理出可用的答复" in _visible(out), (
        f"兜底话术不是那段诚实的固定文本：{_visible(out)!r}"
    )


def test_fallback_prefers_the_model_own_last_draft() -> None:
    """★ 兜底优先用模型自己写过的草稿（剥离后），而不是固定话术。

    ⚠️ 实测的 B[3] 就是「七段独白 + 四段正文」——草稿里**有真答案**，
    剥离后通常就能用。直接上固定话术等于把已经生成好的答案扔掉，
    用户还得再问一遍。

    ⚠️ 能走到这条兜底的**只有一种形状**：答案写在**工具轮**里
    （整轮按结构性事实被丢），而这一轮回复就此结束、再也没有正文轮。
    为什么另一种形状（正文轮被整轮拒绝）到不了这里：那些草稿在
    ``_settle_round`` 里已经被 ``_strip_drafts`` 洗过一遍，再洗一次
    结果一样、照样不可用 —— 兜底对它们是无效的。第一版用例就是照
    「正文轮被拒」写的，结果**变异测试**（把草稿回收循环改空）
    照样全绿：它根本没走进这个函数。这个形状才是真正的入口。

    ⚠️ ``max_retries=0`` 让「重试额度用尽」成为触发条件，与上一条
    「迭代余量不足」区分开 —— 两条路径共用同一个兜底函数，
    但入口条件不同，各测各的。
    """
    draft = "我先查一下差标。\n\n住宿标准：单晚不超过 600 元。"
    events = [
        _reply_start(),
        _call_start(),
        *_text(draft),
        _tool_call("check_travel_policy"),
        _call_end(),
        # ⚠️ 没有第二轮模型调用：这轮回复在工具轮之后直接结束。
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={"check_travel_policy": _FakeTool(is_read_only=True)},
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(max_retries=0), events, agent))

    assert "单晚不超过 600 元" in _visible(out), (
        f"兜底没有回收草稿里的真答案：{_visible(out)!r}"
    )
    assert "我先查一下差标。" not in _visible(out), "回收时把草稿段也一起发出来了"
    assert "没能整理出可用的答复" not in _visible(out)


def test_retry_budget_is_respected() -> None:
    """★ 重试次数有上限（模型连续说胡话时不能无限重试）。

    ⚠️ 上限存在的理由见模块文档「重试机制」第 3 条：框架对「连续吞且
    中间没有任何进展」会抛 ``RuntimeError``（``made_progress`` 守卫），
    而且无限重试本身就是把延迟放大给用户看。

    ``max_retries=1`` 下喂**两轮**纯草稿：第一次结束该吞（第 1 次重试），
    第二次结束必须放行（额度已用尽）。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("用户问的是差旅标准，我不应该自己凭常识推断。"),
        _call_end(),
        _reply_end(),
        _call_start(),
        *_text("当前处于 IDLE 阶段，等待下一轮。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()

    out = asyncio.run(_run(ReplyGuardMiddleware(max_retries=1), events, agent))

    ends = [e for e in out if isinstance(e, ReplyEndEvent)]
    assert len(ends) == 1, f"吞的次数不对：发出 {len(ends)} 个结束事件"
    assert len(agent.state.context_calls) == 1, "重试次数超过了上限"


def test_disabled_guard_is_completely_transparent() -> None:
    """★ ``enabled=False`` 时**一个事件都不许改**（连缓冲都不做）。

    ⚠️ 这个开关是模块文档写明的「排障与 A/B 对比」逃生口：UX 上真正的
    代价是「正文不再逐字流式」。它必须在**行为层面**完全等价于「没装
    这个中间件」，否则关掉它之后仍然出问题，排障时就会怀疑到错误的方向
    ——「关了开关但行为仍有细微差别」是最难查的状态。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("用户问的是差旅标准，我不应该自己凭常识推断。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()

    out = asyncio.run(_run(ReplyGuardMiddleware(enabled=False), events, agent))

    assert [id(e) for e in out] == [id(e) for e in events], (
        "关掉开关后事件仍有增删 —— 排障时会被误导"
    )
    assert agent.state.context_calls == []


def test_tool_result_text_is_kept_for_grounding_report() -> None:
    """★ 工具返回文本必须被收下（数值上报靠它做对照）。

    ⚠️ 上报的判据是「答复里的金额在同轮**工具返回**里找不到出处」——
    收不到工具文本，这条指标会恒为「全部无依据」，等于没有。
    这里只查文本没被吃掉，不查指标值（指标走 ``observe_reply_guard``，
    由 ``tests/test_observability_*`` 与运行时 Prometheus 覆盖）。
    """
    events = [
        _reply_start(),
        _call_start(),
        _tool_result('{"max_hotel_price": 600.0}'),
        _tool_call("check_travel_policy"),
        _call_end(),
        _call_start(),
        *_text("住宿标准：单晚不超过 600 元。", block_id="block-2"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={"check_travel_policy": _FakeTool(is_read_only=True)},
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert any(isinstance(e, ToolResultTextDeltaEvent) for e in out), (
        "工具返回事件被丢掉了 —— 用户看不到执行过程"
    )
    assert _visible(out) == "住宿标准：单晚不超过 600 元。"


def test_tool_round_drops_are_classified_by_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """★★ 工具轮被丢弃的文字按**形状**分两类上报（M1，2026-10-04）。

    ⚠️ 为什么必须分：``..._narration`` 是判据（草稿 / 占位话术）认得的那一类
    —— 就是提示词「工具轮别配文字」要治的，它下降 = 提示词在起作用；
    ``..._other`` 是判据**认不出来**的那一类，**要盯住**：工具轮的文字是
    **静默丢弃**的，里面若其实夹着给用户的答案，用户就少了一段内容，
    而且没有任何重说。混在一条曲线上时，一个下降会被另一个上升抵消。

    ⚠️ 这条只能靠**捕获指标调用**来测：文本被丢掉之后，事件流里什么都
    看不到（这正是该指标存在的理由）。``observe_reply_guard`` 是按名字
    导入到本模块的，所以 monkeypatch 打在守卫模块上。

    ⚠️ 变异：把分类判据（``_is_draft(text) or _is_placeholder(text)``）
    改成恒 ``True`` → 第二段断言红；改成恒 ``False`` → 第一段红；
    改回单一动作名 ``dropped_tool_round`` → 两条都红。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    agent = _FakeAgent(tools={"check_travel_policy": _FakeTool(is_read_only=True)})

    def _run_tool_round(text: str) -> None:
        seen.clear()
        events = [
            _reply_start(),
            _call_start(),
            *_text(text, block_id="block-1"),
            _tool_call("check_travel_policy"),
            _call_end(),
            _reply_end(),
        ]
        asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    # 过程叙述形状 → narration
    _run_tool_round("我先查一下差标。")
    assert "dropped_tool_round_narration" in seen, seen
    assert "dropped_tool_round_other" not in seen, seen
    assert "dropped_tool_round" not in seen, "旧标签不该再出现（已按形状拆开）"

    # 占位话术形状 → 也归 narration（同一张判据表的后半张）
    _run_tool_round("正在为您查询，马上回来。")
    assert "dropped_tool_round_narration" in seen, seen

    # 判据认不出来的形状 → other（这一类要盯住）
    _run_tool_round("住宿标准：单晚不超过 600 元。")
    assert "dropped_tool_round_other" in seen, seen
    assert "dropped_tool_round_narration" not in seen, seen


def test_unclassifiable_tool_round_text_is_sampled_at_debug(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★★ 判据**认不出**的工具轮文字，DEBUG 下留一份截断样本（2026-10-04）。

    ⚠️ 为什么非留样不可：这一段文本在本路径外**零留痕** —— 事件被静默丢弃、
    ``drafts`` 随请求结束蒸发、指标 ``dropped_tool_round_other`` 只有计数。
    于是「它到底是不是一段真答案」这个判断（改不改判据的唯一依据）没有任何
    证据可依。2026-10-04 的探针实测里这一类**真的出现过一次**，而当时查不到
    它写的是什么 —— 这正是本条用例的由来。

    ⚠️ 三条边界都要钉住，缺一条这个留样就会长成两种坏样子：
      · 不打 → 回到无从查起（第一条断言）；
      · narration 也打 → 每次工具轮都刷屏，日志被淹（第三条断言）；
      · 不做截断 → 一段几 KB 的正文整段进日志（第二条断言）。

    ⚠️ 变异：删掉 ``logger.debug`` 调用 → 第一条红；把 ``if not observable``
    改成恒真 → 第三条红；把 ``%.200s`` 换成 ``%s`` → 第二条红。
    """
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        lambda _action: None,
    )
    agent = _FakeAgent(tools={"check_travel_policy": _FakeTool(is_read_only=True)})

    def _run_tool_round(text: str) -> None:
        caplog.clear()
        events = [
            _reply_start(),
            _call_start(),
            *_text(text, block_id="block-1"),
            _tool_call("check_travel_policy"),
            _call_end(),
            _reply_end(),
        ]
        asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    def _sampled_texts() -> list[str]:
        return [
            record.getMessage()
            for record in caplog.records
            if "判据认不出的工具轮文字" in record.getMessage()
        ]

    with caplog.at_level("DEBUG"):
        # 判据认不出的形状 → 必须留样，且样本里带得出原文
        _run_tool_round("住宿标准：单晚不超过 600 元。")
        sampled = _sampled_texts()
        assert len(sampled) == 1, sampled
        assert "住宿标准：单晚不超过 600 元。" in sampled[0], sampled

        # 超长正文 → 截断到 200 字符（长度计数仍报**原文**长度）
        long_text = "北京出差住宿标准核对：" + "测" * 500
        _run_tool_round(long_text)
        sampled = _sampled_texts()
        assert len(sampled) == 1, sampled
        assert len(sampled[0]) < 300, len(sampled[0])
        assert f"长度 {len(long_text)}" in sampled[0], sampled[0]

        # 判据认得的形状（过程叙述）→ 刻意不留样：它每次工具轮都会出现
        _run_tool_round("我先查一下差标。")
        assert _sampled_texts() == [], "narration 不留样，否则日志会被刷屏"


def test_resume_path_synthesizes_events_with_the_reply_id() -> None:
    """★★★ HITL 续答没有 ``REPLY_START``，合成事件仍必须带**正确的** reply_id。

    ⚠️ 这是**对抗性评审实测出来的**缺陷，症状极隐蔽：

    · 用户确认订单后，agent 继续回复 —— 这条路径**不发** ``ReplyStartEvent``
      （``agent/_agent.py:1105-1118`` 只在「新回复」分支发它；续答走
      ``_handle_incoming_event``，只发工具结果事件）；
    · 于是守卫的 ``state.reply_id`` 一直是空串，剥离后合成的事件带着
      ``reply_id=''``；
    · 服务端 ``Msg.append_event`` 对 id 不匹配的事件**只记一条 warning 就跳过**
      —— 落库的消息里没有这段正文；
    · 但实时 SSE 是**另走一条路**的（守卫照样把事件发出去），于是出现
      「当场看得见、刷新就没了」，而日志里一条 ERROR 都没有。

    所以这条用例断言两件事：合成事件带对了 id，**并且**把它们喂给真实的
    ``Msg.append_event`` 之后，正文真的进了消息内容 —— 只查 id 字段的话，
    将来某次重构把 id 传错了位置照样能绿。

    ⚠️ 事件流是按**续答的真实形状**造的：没有 ``REPLY_START``，
    只有带 reply_id 的工具结果事件 + 一轮含草稿的正文。
    """
    from agentscope.message import Msg

    events = [
        # 没有 ReplyStartEvent —— 这正是续答的形状。
        _tool_result("已确认"),
        _call_start(),
        *_text("我先查一下差标。\n\n住宿标准：单晚不超过 600 元。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()
    # ⚠️ 刻意**不设** agent.state.reply_id：这条用例只证明「从事件里补记」
    # 那一级兜底。设了它，删掉补记逻辑（变异）用例照样绿 —— 那就成了假绿，
    # 本文件的 reply_id 用例正是这样假绿过一轮，靠变异测试才发现。
    assert agent.state.reply_id == ""

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    synthesized = [
        e for e in out if isinstance(e, TextBlockDeltaEvent)
    ]
    assert synthesized, "续答路径下正文没发出来"
    assert {e.reply_id for e in synthesized} == {REPLY_ID}, (
        f"合成事件的 reply_id 是空的，服务端会把它们静默丢弃："
        f"{[e.reply_id for e in synthesized]}"
    )

    # 更强的证据：喂给真实的 ``Msg.append_event``，看正文是否真的落进消息。
    msg = Msg(name="main_plan", role="assistant", content=[], id=REPLY_ID)
    for event in out:
        msg.append_event(event)
    assert "单晚不超过 600 元" in msg.get_text_content(), (
        f"append_event 之后消息里没有正文：{msg.get_text_content()!r}"
    )


def test_resume_path_falls_back_to_agent_state_reply_id() -> None:
    """★★ 续答时若连一个带 id 的事件都没有，回退到 ``agent.state.reply_id``。

    ⚠️ 与上一条的区别：上一条覆盖「从事件里补记」，这条覆盖「连事件都没有」
    —— 两级兜底缺一不可（见 :meth:`ReplyGuardMiddleware._resolve_reply_id`）。
    没有这一级，``reply_id`` 为空时合成的正文会在落库时被丢掉。
    """
    from agentscope.message import Msg

    events = [
        # ⚠️ 一个带 reply_id 的事件都没有（连文本事件也是空 id），
        # 模拟最坏的续答形状 —— 此时唯一的 id 来源是 agent.state。
        _call_start(reply_id=""),
        *_text("我先查一下差标。\n\n住宿标准：单晚不超过 600 元。", reply_id=""),
        _call_end(reply_id=""),
        _reply_end(reply_id=""),
    ]
    agent = _FakeAgent()
    agent.state.reply_id = REPLY_ID  # 只有这一处能拿到 id

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    msg = Msg(name="main_plan", role="assistant", content=[], id=REPLY_ID)
    for event in out:
        msg.append_event(event)

    assert "单晚不超过 600 元" in msg.get_text_content(), (
        f"没有回退到 agent.state.reply_id，正文丢失：{msg.get_text_content()!r}"
    )


@pytest.mark.parametrize(
    "reason",
    [
        ReplyFinishedReason.INTERRUPTED,
        ReplyFinishedReason.EXCEED_MAX_ITERS,
        ReplyFinishedReason.ERROR,
    ],
)
def test_fallback_text_is_not_injected_on_abnormal_endings(
    reason: ReplyFinishedReason,
) -> None:
    """★★ 非正常结束时**不许**补发兜底话术（尤其是用户主动取消时）。

    ⚠️ 这段兜底话术是「抱歉，这轮我没能整理出可用的答复（后台这次没有返回
    有效内容）……麻烦你再问我一次」。它在**用户自己按了停止**时是
    **错误归因** —— 后台没坏，是用户不想要了；而且它会落进历史消息，
    让用户下次打开会话时看到一句莫名其妙的道歉。

    超迭代/出错同理：那两种情况由框架与 ``src/chains/events.py`` 收尾
    （那里会把进行中的任务标记为失败），我们再插一句只会和它抢屏幕。

    第一版实现**没有**这道门（只在 ``ReplyEndEvent`` 分支无条件调
    ``_fallback_if_silent``），是对抗性评审实测出来的。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("我先查一下差标。"),
        _call_end(),
        _reply_end(reason),
    ]
    agent = _FakeAgent()

    out = asyncio.run(_run(ReplyGuardMiddleware(max_retries=0), events, agent))

    assert _visible(out).strip() == "", (
        f"{reason} 时补发了兜底话术，用户会看到一句错误归因的道歉："
        f"{_visible(out)!r}"
    )
    assert any(isinstance(e, ReplyEndEvent) for e in out)


def test_reply_id_is_recorded_from_the_first_event_carrying_one() -> None:
    """★ ``reply_id`` 只在为空时补记，绝不被后续事件覆盖。

    ⚠️ 覆盖写的后果：一条回复里后续若混进别的 id（例如框架发的事件带的是
    另一个 reply 的 id），已经发出去的事件会和新合成的事件分属两个 id，
    落库时一半进一半丢。所以判据是「只填空、不改写」。
    """
    events = [
        _call_start(),  # 第一个带 id 的事件
        *_text("住宿标准 600 元。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert {e.reply_id for e in out if getattr(e, "reply_id", "")} == {REPLY_ID}


def test_block_ids_are_unique_across_multiple_stripped_rounds() -> None:
    """★ 同一回复里两次剥离必须用**不同**的 block id。

    ⚠️ 撞块的后果不是报错，而是**文本串位**：第二个 ``TEXT_BLOCK_START``
    用同一个 id 时建不出新块，它的 DELTA 会被 ``_find_block`` 归到第一个块上
    （``message/_base.py``）。实测场景：模型先说一段带草稿的答案，
    被剥离发出；再补一句、又被剥离 —— 两句就粘进了同一个块。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("我先查一下差标。\n\n住宿标准：单晚不超过 600 元。"),
        _call_end(),
        _call_start(),
        *_text("我先查一下舱位。\n\n机票限经济舱。", block_id="block-2"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))
    starts = [e for e in out if isinstance(e, TextBlockStartEvent)]

    assert len(starts) == 2, f"应当合成两段正文，实际 {len(starts)} 段"
    assert starts[0].block_id != starts[1].block_id, (
        f"两次剥离用了同一个 block id：{starts[0].block_id}"
    )


def test_retry_is_skipped_when_structured_output_is_satisfied() -> None:
    """★ 结构化输出已满足时不许重试（否则会撞框架的反忙循环守卫）。

    ⚠️ 框架在「schema 已满足」的 ``Exit`` 分支**不调用模型**就能再次发出
    ``ReplyEndEvent``（``agent/_agent.py:3554-3578``），连吞两次且中间没有进展
    就会抛 ``RuntimeError``（``agent/_agent.py:1165-1173``）。重说也改变不了结局 ——
    结构化结果已经定了。

    ⚠️ 当前装配下 ``main_plan`` 不带 ``structured_schema``，这条分支**不可达**；
    写它是为了让将来给主智能体加结构化输出时不会变成一次线上崩溃。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("用户问的是差旅标准，我不应该自己凭常识推断。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()
    agent.state.reply_context = SimpleNamespace(
        structured_schema={"type": "object"},
        structured_output={"intent": "query_policy"},
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert any(isinstance(e, ReplyEndEvent) for e in out), (
        "结构化输出已满足还吞了结束事件 —— 会连吞两次并触发框架的 RuntimeError"
    )
    assert agent.state.context_calls == []


# ==============================================================================
# 四、真实 Agent 端到端：守卫真的接在 ReAct 循环上
# ==============================================================================


def ping(note: str = "") -> ToolChunk:
    """无害的只读工具，用来触发第二轮 ReAct。

    Args:
        note (`str`): 随便什么内容。

    Returns:
        `ToolChunk`: 固定回复。
    """
    del note
    return ToolChunk(content=[TextBlock(text="600")])


class DraftThenAnswerModel(MockChatModel):
    """第 1 轮：一句过程叙述 **+** 一个工具调用；第 2 轮：正常答复。

    ⚠️ 这个形状就是实测 A 类缺陷：模型在同一轮里**既写了字又调了工具**
    （``_agent.py`` 的 reasoning 分支允许两者并存）。用调用序号驱动，
    而不是用 ``#mock-tool:`` 指令 —— 指令来自消息历史，第二轮仍在上下文里，
    会让模型每轮都重发同一个工具调用（见 ``test_orchestration_lane.py``
    里同样的取舍）。
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(stream=False, **kwargs)
        self.calls = 0

    async def _call_api(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        if self.calls == 1:
            return ChatResponse(
                content=[
                    TextBlock(text="我先查一下差标。"),
                    ToolCallBlock(id="slow-1", name="ping", input="{}"),
                ],
                is_last=True,
            )
        return ChatResponse(
            content=[TextBlock(text="住宿标准：单晚不超过 600 元。")],
            is_last=True,
        )


def test_guard_works_inside_a_real_react_loop() -> None:
    """★★★ 端到端：真实 ReAct 循环里，工具轮前言不会出现在可见回复中。

    ⚠️ 上面所有用例喂的都是脚本事件，它们证明「守卫按设计改事件流」，
    但**证不了**「框架真的按这个形状发事件」—— 比如若框架把文字事件
    的 ``reply_id`` 换成消息 id、或在工具轮不产生 ``MODEL_CALL_END``，
    脚本用例会全绿而线上依然漏草稿。这条用例让真实 Agent 跑一遍，
    按生产口径（拼接所有 ``TEXT_BLOCK_DELTA``）读用户可见文本。

    ⚠️ 用的模型是**非流式**的：流式下文本会被切片成多个 DELTA，
    断言仍然成立，但「第一轮前言与第二轮答案各是几个分片」就不确定了。
    这里要钉的是**内容边界**，不是分片数量。

    ⚠️ 守卫**必须**像生产那样包在 :class:`_AgentScopedMiddleware` 里 ——
    这不是「更真实的写法」，而是唯一能发现「包装层把 ``on_reasoning``
    转发给守卫、基类直接抛异常」这个事故的形状：实测中它让 8/8 轮回复
    全部报错，而当时所有单测都是绿的（它们直接调 ``on_reply``）。
    包装层只在 ``agent.name`` 命中时生效，所以这里用 ``main_plan``。
    """

    async def scenario() -> Any:
        """跑一轮真实回复，返回收集到的事件。

        Returns:
            `list[Any]`: 全部事件。
        """
        agent = Agent(
            name="main_plan",
            system_prompt="你是差旅助手。",
            model=DraftThenAnswerModel(),
            toolkit=Toolkit(tools=[FunctionTool(ping, is_read_only=True)]),
            middlewares=[
                _AgentScopedMiddleware(
                    ReplyGuardMiddleware(),
                    agent_names=("main_plan",),
                ),
            ],
        )
        events = []
        async for event in agent.reply_stream(
            inputs=Msg(name="user", role="user", content=[TextBlock(text="差标是多少")]),
        ):
            events.append(event)
        return events, agent

    events, agent = asyncio.run(scenario())
    visible = "".join(
        event.delta for event in events if isinstance(event, TextBlockDeltaEvent)
    )

    assert agent.model.calls == 2, (  # type: ignore[attr-defined]
        f"ReAct 没有跑满两轮（{agent.model.calls} 次），这条用例就没测到东西"  # type: ignore[attr-defined]
    )
    assert "我先查一下差标。" not in visible, (
        f"工具轮的前言漏进了用户可见回复：{visible!r}"
    )
    assert "住宿标准：单晚不超过 600 元。" in visible, (
        f"最终答案没发出来：{visible!r}"
    )


# ==============================================================================
# 九、事实接地闸门（差标上限）
# ==============================================================================
# 背景：2026-10-03 实测，同一个用户、同一个问题「住宿标准」，工具每轮都返回
# 同一个数（酒店单晚上限 600 元），但 8 轮里有 3 轮的正文写成了 500 元，
# 并配上「依据：差旅制度「住宿标准」条款」这种编造出来的出处。
# 那 500 不是凭空来的 —— 知识库里「三类城市 B 档」正是 500，模型用了先验
# 而不是工具返回。提示词压不住它，所以在输出侧加闸门。
#
# 闸门的判据很窄（见 _ungrounded_limit_claims），这些用例就是**钉住这个窄度**：
# 既要拦住编造，也不能误伤正确的回答。
_LIMIT_TOOL_TEXT = (
    '{"max_hotel_price": 600.0, "max_flight_price": 2000.0, '
    '"policy_note": "默认差标：经济舱，酒店单晚不超过 600 元，机票不超过 2000 元。"}'
)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        # ── 该拦的：在断言一个上限，而工具没给过这个数 ──
        ("住宿标准：酒店单晚上限 500 元。", ["500"]),
        ("住宿标准：**每晚上限 500 元**，超出部分不予报销。", ["500"]),
        ("酒店单晚不超过 500 元可以全额报销。", ["500"]),
        ("住宿费最高 500 元。", ["500"]),
        # ── 不该拦的：工具返回里就有这个数 ──
        ("酒店单晚上限 600 元。", []),
        ("酒店单晚不超过 600 元，机票 2000 元封顶。", []),
        # ── 不该拦的：数字来自用户，句子的语气是**核对**不是定义 ──
        # 「上限」在 500 之后 9 个字符处，超出后视窗（4），不命中。
        ("500 元没超过 600 的上限，可以报。", []),
        # ── 不该拦的：附近根本没有定义上限的措辞 ──
        ("二线 450 元、三线 350 元。", []),
        ("这次行程总共 1200 元。", []),
        ("你上次住的酒店 380 元一晚。", []),
        # ── 该拦的：2026-10-03 实测漏放的那一类（真源改口径）──
        # 「差标」「标准」「额度」这三个词是**逐个案例**补进词表的。
        # 下面三条的金额附近**没有**「上限」二字，只能靠这三个词命中 ——
        # 词表缩回去，它们就全部漏放，而那正是线上真实出过的那句话。
        ("酒店差标是 500 元/晚。", ["500"]),
        ("住宿标准写的是 500 元。", ["500"]),
        ("住宿费报销额度 500 元。", ["500"]),
        # ── 不该拦的：房型名里的「标准」不是在说上限 ──
        # ⚠️ 2026-10-03 对抗验证实测到的误伤：这两句都是**正确**回答，
        # 却因为「标准间」里含「标准」被判定成在上限语境，500 被拦下、
        # 重试耗尽后用户只看到一句兜底道歉。
        ("可以订标准间，500 元一晚。", []),
        ("标准间 500 元一晚可以订。", []),
        ("标准双床房 500 元一晚。", []),
        ("标准大床房，500 元。", []),
        # ⚠️ 这一条与上面几条配对，守的是**搭配字表**的边界：把「是」
        # 之类常见的虚词也放进 `_STANDARD_ROOM_HEADS`，「住宿标准是 500」
        # 这类**真编造**就会以「房型」的名义整句漏放。
        ("住宿标准是 500 元/晚。", ["500"]),
    ],
)
def test_ungrounded_limit_claims_pins_the_narrow_judgement(
    answer: str, expected: list[str]
) -> None:
    """★ 闸门的判据必须**窄**：拦编造，但不误伤核对与算账。

    ⚠️ 「不该拦」那几条是重点：它们都是**正确**的答复形态（用户自己报的价、
    知识库里的分档值、模型的算术结果）。判据一放宽，这些就会被拦下来
    改写成别的说法 —— 那比放过一次编造更伤用户信任。

    ⚠️ 「500 元没超过 600 的上限」这条是实测调出来的：对称的 12 字符
    窗口会把用户自己报的 500 判成编造，于是加长窗口反而让闸门变坏。
    见 :data:`_LIMIT_LOOKBEHIND` / :data:`_LIMIT_LOOKAHEAD` 的说明。

    ⚠️ 最后三条与上面那条**是有冲突的解法**，注释里写清楚了取舍：
    靠缩小词表（不收「标准」「差标」）来避免误伤，代价是线上真实编造
    「酒店差标是 500 元/晚」整条漏放 —— 实测就是这样漏掉的。正确的解法
    是把「用户自己说过的数」从源头豁免（见 ``user_text`` 参数），
    而不是让词表替它挨打。这几条一起放着，就是为了钉住这个取舍。
    """
    assert _ungrounded_limit_claims(answer, _LIMIT_TOOL_TEXT) == expected


def test_room_type_standard_does_not_mask_a_real_limit() -> None:
    """★ 房型豁免必须**遍历**而不是**短路**：真编造仍要拦住。

    ⚠️ 守的是实现里很容易写错的一处：豁免「标准间」时若写成
    「窗口里见到房型就直接放行整句」，这句里的 450 会跟着漏网。
    正确写法是跳过这一个「标准」、继续往后扫。

    ⚠️ 句子的构造是**刻意**的，不是随手写的。要同时满足两个条件才能在
    12 字符窗口（前 8 后 4）内观察到「遍历 vs 短路」的差别：
      1. 房型的「标准」落在金额的窗口**内** —— 所以金额要紧跟在
         「标准间」后面；
      2. 窗口里还得有一个在 :data:`_LIMIT_WORDS` **排序上晚于**「标准」
         的上限措辞（差标/标准/额度）—— 否则先扫到的词就已经返回 True，
         两条路径的结果一样，用例杀不掉短路变异体（实测 GREEN-BAD）。
    「订标准间，额度是 450 元。」同时满足这两条：
    窗口 = 「标准间，额度是 450 元。」，先扫到房型「标准」（跳过），
    再扫到「额度」—— 短路实现会在这里返回 False，于是编造漏放。
    """
    answer = "订标准间，额度是 450 元。"

    assert _ungrounded_limit_claims(answer, _LIMIT_TOOL_TEXT) == ["450"], (
        "房型的豁免把同一句里真正的编造也一起放行了"
    )


#: 用户报了一个价、让助手判断能不能报 —— 这个数在正文里出现是**合规**的。
_USER_PRICE_TEXT = "我订了 500 的酒店，能报吗？"
#: 结论正确、但如果不豁免用户数字就会被判成编造的答复。
_ANSWER_QUOTING_USER_PRICE = "按你的差标，500 元一晚可以报销。"


def test_user_supplied_number_is_not_treated_as_fabrication() -> None:
    """★★ 用户自己报过的数字**不算编造** —— 这是闸门最危险的一类误伤。

    ⚠️ 场景是日常的：用户说「我订了 500 的酒店能报吗」，助手复述这个数
    再给结论。这个 500 有合法来源（用户的话），工具返回里当然没有它。
    闸门若不看用户原话，就会把一次**完全正确**的回答拦下、逼模型重说，
    最后可能退回「这轮没答好」的兜底话术 —— 用户会看到系统答不出自己
    刚说过的数字。这比放过一次编造更伤。

    ⚠️ 用例必须**同一段正文跑两遍**：带用户原话 → 放行；不带 → 拦下。
    只测前半段证明不了什么（窗口、词表任何一处改动都可能让它恰好通过），
    后半段才是「豁免真的在起作用」的证据。
    """
    blocked = _ungrounded_limit_claims(
        _ANSWER_QUOTING_USER_PRICE,
        _LIMIT_TOOL_TEXT,
    )
    assert blocked == ["500"], (
        "对照组失败：不带用户原话时这句话本来就该被判成编造，"
        f"否则下面那条断言证明不了豁免生效（实际判定 {blocked}）"
    )

    assert (
        _ungrounded_limit_claims(
            _ANSWER_QUOTING_USER_PRICE,
            _LIMIT_TOOL_TEXT,
            _USER_PRICE_TEXT,
        )
        == []
    ), "用户自己报过的数字被当成了编造 —— 正确的回答会被拦掉"


def test_user_exemption_still_catches_a_second_invented_number() -> None:
    """★ 豁免只对**用户说过的那个数**生效，不放过同一句里的其他编造。

    ⚠️ 豁免写成「本句来自用户就不查」是很容易犯的错：那样模型只要复述
    用户的一个数字，同一句里再编一个上限就没人管了。判据必须落在**每个
    金额**上，而不是整句上。
    """
    claims = _ungrounded_limit_claims(
        "你报的 500 元可以报，住宿差标上限是 800 元。",
        _LIMIT_TOOL_TEXT,
        _USER_PRICE_TEXT,
    )
    assert claims == ["800"], f"同一句里的第二个编造被豁免掉了：{claims}"


def test_user_text_blank_matches_the_old_behaviour() -> None:
    """★ 拿不到用户原话时，判据与「没有这个参数」时**逐字一致**。

    ⚠️ 这条守的是**向后兼容**：``user_text`` 默认空串，且空串时不做任何
    豁免。若有人把默认值写成「豁免一切」或让空串命中子串，所有既有行为
    都会变 —— 上面那批参数化用例会红，但更容易发生的是某个边界悄悄放宽。
    """
    answer = "住宿标准：酒店单晚上限 500 元。"
    assert _ungrounded_limit_claims(answer, _LIMIT_TOOL_TEXT) == ["500"]
    assert _ungrounded_limit_claims(answer, _LIMIT_TOOL_TEXT, "") == ["500"]


@pytest.mark.parametrize(
    "user_text",
    [
        # 2026-10-03 对抗验证给的反例：用户只说了 5000，
        # 答复里的 500 却因为「500 是 5000 的子串」被判成「用户说过的数」。
        "我这次出差预算是 5000 元。",
        "我上个月住的那家是 1500。",
        # 手机号：真实用户不会把它当金额，但子串匹配会。
        "我的手机号是 13800005000。",
        # 千分位写法 —— 归一后是 5000，仍然不能豁免 500。
        "预算 5,000 元。",
    ],
)
def test_a_user_number_only_grounds_its_own_value(user_text: str) -> None:
    """★★ 用户说过「包含 500 的某个数」≠ 用户说过 500。

    ⚠️ 这条守的是**子串匹配**这个具体的错误实现。它对用户原话做过
    ``"500" in user_text`` 之类的判断，于是「预算是 5000」把
    「差标是 500」（工具返回的是 600）这条编造豁免掉了 —— 方向恰好是
    闸门最不该漏的那一侧：模型编了一个**比真实差标更小**的上限。

    ⚠️ 同一个错误在工具侧也发生过，见
    :func:`test_a_tool_number_only_grounds_its_own_value`。两处一起看，
    才能说明「按完整数字比对」是判据而不是巧合。
    """
    claims = _ungrounded_limit_claims(
        "酒店差标是 500 元/晚。",
        _LIMIT_TOOL_TEXT,
        user_text,
    )
    assert claims == ["500"], (
        f"用户说的是「{user_text}」，里面并不含 500 这个数，"
        f"但编造被豁免了：{claims}"
    )


def test_a_tool_number_only_grounds_its_own_value() -> None:
    """★★ 工具返回里的 2000 不能给「上限 200 元」当依据。

    ⚠️ ``check_travel_policy`` 的返回里同时有酒店 600 与机票 **2000**
    （``DEFAULT_POLICY_LIMIT``），于是 ``"200" in tool_text`` 为真。
    子串判据下，「酒店差标是 200 元」这句话不战而胜 —— 判据根本没启动。

    ⚠️ 断言与上面那条**成对**：一条守用户侧、一条守工具侧。只改一侧
    （比如只把 ``user_text`` 改成按数字比）会让另一侧继续漏，而单看
    任何一条用例都是绿的。
    """
    claims = _ungrounded_limit_claims("酒店差标是 200 元/晚。", _LIMIT_TOOL_TEXT)
    assert claims == ["200"], f"2000 里的 200 被当成了依据：{claims}"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        # ── 由工具数字算出来的合计，是正确的回答，必须放行 ──
        # 2026-10-03 对抗验证的反例：这两句被拦后重试耗尽，
        # 用户最终看到的是「抱歉，这轮我没能整理出可用的答复」。
        ("两晚住宿标准合计 1200 元。", []),
        ("住宿费标准内可报 1200 元。", []),
        ("三晚住宿标准合计 1800 元。", []),
        # ── 不是倍数 → 仍然拦 ──
        ("两晚住宿标准合计 1100 元。", ["1100"]),
        ("酒店差标是 900 元/晚。", ["900"]),
    ],
)
def test_a_computed_total_is_not_a_fabricated_limit(
    answer: str, expected: list[str]
) -> None:
    """★★ 1200 = 2×600 是**算出来**的，不是编的。

    ⚠️ 这条与「该拦的」那批用例**方向相反、必须同时成立**：
    闸门要拦的是「工具说 600、正文写 500」这类**无中生有**；
    而「600 元/晚 × 2 晚 = 1200 元」是拿工具给的数算的，属于
    ``_ungrounded_amounts`` 文档里早就承认的合法来源。

    ⚠️ 拦错这一类代价很大：闸门会逼模型重说，重试耗尽后走兜底 ——
    实测用户看到的是「抱歉，这轮我没能整理出可用的答复」，
    也就是**一次正确的回答被系统自己删掉了**。

    ⚠️ 已知取舍（最后两条不是遗漏）：倍数判据会放过「把 2×600 写成
    『差标是 1200』」这种字面上的编造。判据是「这个数算不算得出来」，
    而 1200 确实算得出来 —— 要区分「合计」与「上限断言」得靠语义理解。
    这类漏报仍会被 ``_ungrounded_amounts`` 的**上报**记进指标，
    靠数据决定要不要做更强的方案。
    """
    assert _ungrounded_limit_claims(answer, _LIMIT_TOOL_TEXT) == expected


@pytest.mark.parametrize(
    ("value", "tool_numbers", "expected"),
    [
        # 2×600 —— 实测反例（两晚住宿）
        ("1200", {"600", "2000"}, True),
        # 9×600 是边界内的最大倍数
        ("5400", {"600"}, True),
        # 倍数 1 不是「算出来的合计」，就是原数本身（原数走精确比对那条路）
        ("600", {"600"}, False),
        # 10×600 超出「几晚/几张」的量级
        ("6000", {"600"}, False),
        # 1.5×600：小数倍不是「单价 × 数量」的形态
        ("900", {"600"}, False),
        # 基准数是 0 / 负数时不能当分母（也是为了避免 ZeroDivisionError）
        ("1200", {"0"}, False),
        ("1200", {"-600"}, False),
        # 用户说的数**不参与**倍数判据（调用方只传工具数）——
        # 见 _is_derived_total 的说明
        ("1600", {"600", "2000"}, False),
    ],
)
def test_derived_total_recognizes_only_small_integer_multiples(
    value: str, tool_numbers: set[str], expected: bool
) -> None:
    """★ 倍数判据的边界：2..9 的整数倍，别的都不算。

    ⚠️ 这些边界不是装饰：放宽到任意倍数，``1200/600`` 与 ``1200/1200``
    就没区别了（后者是「工具返回里本来就有」的另一种写法）；
    放宽到小数倍，则任何数除以任何数都可能「算得出来」，
    闸门等于关掉。
    """
    assert _is_derived_total(value, tool_numbers) is expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("600", "600"),
        ("600.0", "600"),
        ("600.00", "600"),
        ("1,200", "1200"),
        ("1200.50", "1200.5"),
        ("0.50", "0.5"),
        ("0", "0"),
        ("60", "60"),
        ("6000", "6000"),
        # ── 全角写法（C2，2026-10-04 补）：同一个数的另一种输入形态 ──
        ("６００", "600"),
        ("１，２００", "1200"),
        ("１２００．５０", "1200.5"),
        ("０", "0"),
    ],
)
def test_normalize_amount_strips_writing_not_digits(raw: str, expected: str) -> None:
    """★ 归一化**只去写法**（全角、千分位、小数尾巴），一个数字都不许动。

    ⚠️ 这条守的是一个真实存在过的实现缺陷形态：把「去掉小数尾巴」写成
    无条件 ``rstrip("0")`` —— ``"600"`` 会变成 ``"6"``。
    它是**双向**错的：工具返回里的 600 也归一成 6，于是「工具说 600、
    答复写 6000」这种差十倍的编造被判成「与工具返回一致」而放行。
    两边同时错，正是它难以被察觉的原因（只测一侧的用例全是绿的）。

    ⚠️ 全角那三条是 2026-10-04 补的（C2）：``\\d`` 在 Python 里匹配
    Unicode 十进制数字，所以 ``６００`` **进得来**；而归一化只认 ASCII 写法，
    于是进来的数一个都对不上 —— 一段**完全正确**的答复被判成编造拦下。
    见 :func:`_normalize_amount` 的 docstring。
    """
    assert _normalize_amount(raw) == expected


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        # 全角写法但数与工具返回**一致** —— 完全正确的答复，必须放行。
        ("酒店差标是 ６００ 元/晚。", []),
        # 全角数字 + **全角千分位**，两个数都在工具返回里（600 / 2000）。
        # ⚠️ 这一条专打「只在 _normalize_amount 里折全角」的半吊子修法：
        # 全角逗号不在 ``[\d,]`` 里，抽取器会从「２０００」开始匹配，
        # 于是 2000 变成残缺的 2000（这里两种切法都得到 2000），
        # 而 1200 那种写法会变成 200 —— 用 1200 才杀得掉这个变异体。
        ("酒店差标是 １，２００ 元/晚。", []),
        # 全角写法且数与工具返回**不一致** —— 仍是编造，必须拦。
        # ⚠️ 上报的是**折过全角**的 ASCII 写法：折全角发生在抽取之前
        # （见 _fold_width），这是有意的 —— 那条字符串会进日志、指标和
        # 给模型的纠正指令，ASCII 更好认。
        ("酒店差标是 ５００ 元/晚。", ["500"]),
    ],
)
def test_full_width_digits_are_the_same_number(
    answer: str, expected: list[str]
) -> None:
    """★★ 全角数字与半角数字是**同一个数** —— C2 的闸门级用例。

    ⚠️ 缺陷方向与别的用例**相反**：这条修的代价是「把正确答案改坏」
    （拦下 → 重说 → 重试用尽时用户拿到一句道歉兜底），不是放走编造。
    中文输入法下全角数字极常见，而工具返回（JSON）永远是半角 ——
    两者对不上时，用户看到的是「模型答对了却被系统拦下」。

    ⚠️ 三条用例各打一个变异体，别删：
    第 1 条杀「完全不折全角」；第 2 条杀「只在 _normalize_amount 里折、
    抽取前不折」（全角千分位会把 token 切断）；第 3 条守「折全角不许
    顺手把编造也放行」。

    ⚠️ 第 2 条的数是 1200：它在工具返回里**以 600 的 2 倍**得到豁免，
    这是 :func:`_is_derived_total` 明写的既有取舍（那里记着「故意放过
    把 2×600 的合计写成差标」）。这里要的是**抽取出完整的 1200**，
    不是「1200 该不该放行」—— 那是另一条用例的事。
    """
    assert _ungrounded_limit_claims(answer, _LIMIT_TOOL_TEXT) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # ── 该折的：全角形式区（U+FF01..U+FF5E）与表意空格 ──
        ("６００", "600"),
        ("１，２００", "1,200"),
        ("６００．５０", "600.50"),
        ("（６００）", "(600)"),
        ("６００　元", "600 元"),
        # ── 不该折的：兼容等价符号（NFKC 会折，折了就是缺陷 F3）──
        ("①600", "①600"),
        ("②600", "②600"),
        ("Ⅻ", "Ⅻ"),
        ("㍿", "㍿"),
    ],
)
def test_width_folding_covers_width_only_not_compatibility(
    text: str, expected: str
) -> None:
    """★★ 折全角只折**全角形式**，不折「兼容等价」—— F3 的判据范围。

    ⚠️ 这里曾经是 ``unicodedata.normalize("NFKC", ...)``。NFKC 折的是
    「兼容等价」，范围**远大于**「全角」：``①``（U+2460）的兼容分解是 ``1``，
    于是一句「①600元/晚是上限」折完变成「1600元/晚是上限」—— 一个模型
    **从来没写过**的数被抽了出来，正确的答复被判成编造（详见
    :func:`_fold_width` 的 docstring）。

    ⚠️ 折全角本身是**必须**的（前两条参数：全角数字与全角千分位都进过
    实测缺陷），所以修法是收窄范围，不是取消折叠 —— 用例里两组参数
    各钉一侧，删掉任何一组，对应的变异体就会活下来。

    ⚠️ ``str.translate`` 的映射是**逐字符 1:1** 的（末条参数里的 ㍿ 长度
    不变即为证）：闸门要在折过的字符串上按 ``match.start()`` 切窗口找
    「上限」措辞，长度一变偏移就会漂。
    """
    assert _fold_width(text) == expected
    assert len(_fold_width(text)) == len(text)


def test_a_circled_number_is_not_glued_into_a_phantom_amount() -> None:
    """★★ 圈号 + 数字不许粘成一个模型没写过的金额 —— F3 的闸门级用例。

    ⚠️ 与上面那条的分工：上面钉 ``_fold_width`` 的**范围**，这条钉
    **后果** —— 圈号是模型列条目时的排版符号，粘出来的 ``1600`` 会让
    :func:`_ungrounded_limit_claims` 判一段正确的答复「无依据」，
    而纠正指令里点名的那个数模型根本没见过，重说只能照着编。

    ⚠️ ``①`` 不是 ``\\d``（Unicode 十进制数字），``_AMOUNT_PATTERN``
    会从「600」开始匹配 —— 这是修复后能从源头对上的原因，
    不是「恰好没报错」。
    """
    assert (
        _ungrounded_limit_claims("酒店差标要点：①600元/晚是上限。", _LIMIT_TOOL_TEXT)
        == []
    ), "圈号被折成了 1，和 600 粘成了 1600 —— 正确的答复会被判成编造"


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        # 差十倍 / 差一位：都**不是**同一个数，都必须拦。
        ("酒店差标是 6000 元。", ["6000"]),
        ("酒店差标是 60 元。", ["60"]),
    ],
)
def test_digit_collapsing_cannot_ground_a_fabrication(
    answer: str, expected: list[str]
) -> None:
    """★★ 归一化塌缩（600/60/6000 → "6"）会让编造白拿依据 —— 端到端钉住。

    ⚠️ 上面那条是单元级；这条走完整闸门，因为**塌缩是两边同时发生的**：
    即使 ``_normalize_amount`` 把 600 变成 6，只要测试只比较「工具说 600、
    答复写 600」，两边一起塌缩，断言照样通过（实测：参数化用例全绿，
    变异体存活）。必须拿一组**本来就不相等**的数去对照，塌缩才现形。

    ⚠️ 这里刻意用**只有酒店 600** 的返回文本，不带 ``_LIMIT_TOOL_TEXT``
    里的机票 2000：6000 恰好是 2000 的 3 倍，会被「合计」豁免放行 ——
    那是另一条判据（见 :func:`test_a_computed_total_is_not_a_fabricated_limit`），
    混进来会让这条用例测不出塌缩。

    ⚠️ 工具侧必须同时给出**裸数字形式**的 600（``policy_note`` 里的
    「不超过 600 元」），只写 ``"max_hotel_price": 600.0`` 测不出塌缩：
    带小数点的 ``"600.0"`` 削零后是 ``"600."`` → 再去点得 ``"600"``，
    塌缩不了，于是「编造 6000」仍被拦住、变异体存活（实测 GREEN-BAD）。
    """
    tool_text = '{"max_hotel_price": 600.0, "policy_note": "酒店单晚不超过 600 元。"}'
    assert _ungrounded_limit_claims(answer, tool_text) == expected


def test_a_total_derived_from_the_users_own_number_is_a_known_gap() -> None:
    """★ 已知缺口：用户自己的数 × 2 **不**算依据。

    ⚠️ 这条用例记录的是一个**有意保留**的窄口，不是修复。理由：
    用户随口说的数一旦进入倍数判据，它的 2..9 倍全部变成「有依据」——
    包括模型编出来的上限（用户说 500，正文写「差标 1500」，正好 3×500）。
    而「差标编造」正是本闸门存在的唯一原因。

    ⚠️ 代价是：用户说「800 一晚」，模型答「两晚合计 1600 元」时会被拦一次。
    这类误伤**在实测中尚未出现过**，而放宽的漏洞是立刻存在的；
    先窄着，等 ``_ungrounded_amounts`` 的指标曲线给出发生率再决定。
    """
    claims = _ungrounded_limit_claims(
        "两晚住宿标准合计 1600 元。",
        _LIMIT_TOOL_TEXT,
        "我这次订的是 800 一晚。",
    )
    assert claims == ["1600"], (
        "倍数判据开始吃用户数字了 —— 用户说 500、模型编「差标 1500」"
        f"也会被放行，请先看 _is_derived_total 的取舍说明。实际：{claims}"
    )


def test_ungrounded_limit_is_blocked_and_retried_with_the_number_named() -> None:
    """★★ 核心场景：正文写了一个工具没给过的上限 → 拦下 + 重说 + 点名。

    ⚠️ 纠正指令**必须点名那个数**。只说「你的数字没依据」时，模型看不到
    自己被拦掉的输出，多半会原样再写一遍 —— 那样重试只是白烧一轮迭代
    和几秒延迟。
    """
    events = [
        _reply_start(),
        _call_start(),
        _tool_call("check_travel_policy"),
        _tool_result(_LIMIT_TOOL_TEXT),
        _call_end(),
        _call_start(),
        # 编造的正文 —— 工具给的是 600。
        *_text("住宿标准：酒店单晚上限 500 元。", block_id="block-2"),
        _call_end(),
        # ⚠️ 这一条 ReplyEnd 是**必须**的：守卫是在「回复要结束了」这一刻
        # 才决定吞不吞的。少了它，模型自己接着往下写的那一轮就会被当成
        # 「它自己改好了」，重试路径根本不会被走到。
        _reply_end(),
        # 重说后模型改正了。
        _call_start(),
        *_text("住宿标准：酒店单晚上限 600 元。", block_id="block-3"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={"check_travel_policy": _FakeTool(is_read_only=True)},
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(max_retries=1), events, agent))

    visible = _visible(out)
    assert "500" not in visible, f"编造的上限被发给了用户：{visible!r}"
    assert "600" in visible, f"改正后的正文没发出去：{visible!r}"
    assert agent.state.context_calls, "重试时必须把纠正指令塞进模型上下文"
    hint = str(agent.state.context_calls[0][1])
    assert "500" in hint, f"纠正指令没有点名被拦下的数值：{hint[:200]!r}"


def test_limit_gate_is_off_when_no_policy_tool_ran() -> None:
    """★ 没查过差标 → 闸门必须关闭。

    知识库里的城市分档值（一类 A 档 600、B 档 800、C 档 1200）**没有**任何
    工具返回作为依据。若闸门不看「调没调过差标工具」，一篇正确的分档答复
    会被整段判成编造 —— 那是把对的改坏。

    ⚠️ 本轮**故意让工具文本非空**（查了酒店），这样才单独验到
    「调过差标工具」这一个条件。否则两个前置条件同时不成立，
    用例会因为「工具文本为空」而通过，删掉政策工具那一条也照样绿 ——
    实测第一版就是这样漏掉一次变异的。
    """
    events = [
        _reply_start(),
        _call_start(),
        _tool_call("search_hotels"),
        _tool_result('{"city": "北京", "price_per_night": 480}'),
        _call_end(),
        _call_start(),
        # ⚠️ 答复里**必须出现「上限」**，否则这个用例证明不了什么：
        # 没有上限措辞时闸门本来就不会启动，把「调过差标工具」这个条件
        # 删掉它照样绿（实测第一版就漏了这次变异）。
        *_text("一类城市住宿上限 600 元/晚，二类城市上限 450 元/晚。", block_id="block-2"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(tools={"search_hotels": _FakeTool(is_read_only=True)})

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert _visible(out) == "一类城市住宿上限 600 元/晚，二类城市上限 450 元/晚。", (
        "没调差标工具时闸门不该启动 —— 知识库的分档值会被误判成编造"
    )


def test_limit_gate_is_off_when_no_tool_text_arrived() -> None:
    """★ 调了工具但没收到返回文本 → 闸门必须关闭。

    「没有证据」必须以「证据通道存在」为前提。工具只回了卡片、或事件流被
    截断时，haystack 是空的 —— 那时**任何**数字都会被判成「工具没给过」，
    闸门就退化成「查过差标就不许提数字」。

    ⚠️ 这条是**踩出来的**：闸门第一版没有这个条件，直接把
    ``test_tool_round_text_is_dropped_but_tool_events_pass`` 里那段
    **正确**的正文拦掉了。
    """
    events = [
        _reply_start(),
        _call_start(),
        _tool_call("check_travel_policy"),
        # ⚠️ 刻意不发 ToolResultTextDeltaEvent。
        _call_end(),
        _call_start(),
        *_text("住宿标准：酒店单晚上限 600 元。", block_id="block-2"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={"check_travel_policy": _FakeTool(is_read_only=True)},
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert _visible(out) == "住宿标准：酒店单晚上限 600 元。"


def test_a_recitation_round_is_gated_even_before_its_tool_returns() -> None:
    """★★ 工具返回**晚于**本轮结算 —— 例外轮仍要过接地闸门（缺陷 F2）。

    ⚠️ 事件顺序是这条用例的全部要点，改一个字它就不测这件事了：

        ModelCallEndEvent          ← 本轮文字在这里结算，闸门也在这里跑
        ToolResultTextDeltaEvent   ← 工具返回**在其后**才到
                                     （``_agent.py``：Reasoning 先于 Acting）

    所以「本轮调过差标工具」时，``state.tool_text`` **必然**还是空的。
    守卫早先把「工具文本为空」直接当成「工具没给过这个数」而整体短路，
    于是本次回复**第一轮**就调差标工具这一最常见的情形，恰好落进空档：
    模型在看到工具返回之前写下的数字（纯粹的先验）原样放行。

    ⚠️ 为什么这里必须是**例外轮**（同时含写工具 ``submit_approval``）：
    普通工具轮的文字整轮丢弃（``dropped_tool_round``），压根走不到闸门；
    只有「文字是给用户看的复述」的例外轮会让这段文字上屏。这正是实测
    缺陷的形状 —— 用户在一句「请确认」里读到「酒店差标 500 元/晚」。

    ⚠️ 与 ``_mixed_write_round`` 的分工：那个辅助函数把工具返回摆在
    ``_call_end()`` **之前**（顺序在真实链路里不可能出现），所以它测的是
    「闸门开着时例外轮不失检」；这条测的是「闸门**开得起来**」——
    少了它，F2 那个空档没有任何用例覆盖（既有用例全绿而线上漏）。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text("请确认：事由「客户拜访」，酒店差标 500 元/晚。", block_id="block-1"),
        _tool_call(_POLICY_TOOL),
        _tool_call("submit_approval"),
        _call_end(),
        # 工具返回在其后 —— 这是真实顺序，也是 F2 的成因。
        _tool_result('{"hotel_limit_per_night": 600}'),
        _tool_result("已提交待确认"),
        # 本轮正文被拦下 → 吞掉 ReplyEnd，要求重说。
        _reply_end(),
        _call_start(),
        *_text("请确认：事由「客户拜访」，酒店差标 600 元/晚。", block_id="block-2"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={
            _POLICY_TOOL: _FakeTool(is_read_only=True),
            "submit_approval": _FakeTool(is_read_only=False),
        },
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(max_retries=1), events, agent))

    visible = _visible(out)
    assert "500" not in visible, f"工具返回之前的先验数字上屏了：{visible!r}"
    assert "600" in visible, f"重说后按工具返回作答的正文没发出去：{visible!r}"
    assert agent.state.context_calls, "重试时必须把纠正指令塞进模型上下文"
    hint = str(agent.state.context_calls[0][1])
    assert "500" in hint, f"纠正指令没有点名被拦下的数值：{hint[:200]!r}"


def test_ungrounded_draft_is_not_revived_by_the_fallback() -> None:
    """★★ 重试用尽后，兜底**不能**把被拦下的草稿原样复活。

    ⚠️ 这是闸门第二版才补上的洞：``_fallback_if_silent`` 会从 ``state.drafts``
    里捞最后一版「还能用」的草稿发出去，而被闸门拦下的那段正文**正是**
    ``drafts`` 里最靠后的一条。没有这条断言时，「拦 → 重试 → 用尽 → 兜底」
    的终点会是同一段编造的正文被发给用户 —— 闸门拦了个寂寞，还白加了延迟。

    期望行为是退回固定话术（承认这轮没答好），而不是发一个会误导用户去
    报销的数字。

    ⚠️ 用 ``max_retries=0`` 把重试额度直接花光，才走得到兜底那一步 ——
    ``ReplyEndEvent`` 分支里 ``_should_retry`` 排在 ``_fallback_if_silent``
    **前面**，只要还能重试就轮不到兜底。
    """
    events = [
        _reply_start(),
        _call_start(),
        _tool_call("check_travel_policy"),
        _tool_result(_LIMIT_TOOL_TEXT),
        _call_end(),
        _call_start(),
        *_text("住宿标准：酒店单晚上限 500 元。", block_id="block-2"),
        _call_end(),
        # 重说一轮，模型**又**写了 500。
        _call_start(),
        *_text("住宿标准：每晚上限 500 元，超出部分自理。", block_id="block-3"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={"check_travel_policy": _FakeTool(is_read_only=True)},
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(max_retries=0), events, agent))

    visible = _visible(out)
    assert "500" not in visible, f"被拦下的草稿经兜底复活了：{visible!r}"
    assert visible.strip(), "兜底不能让用户什么都收不到"


def test_a_blocked_draft_cannot_be_laundered_back_by_a_tool_echo() -> None:
    """★★ 拦过的正文不再翻案：工具回显不算依据（缺陷 F4）。

    ⚠️ 上面那条用例测的是「兜底的即时判据」，这条测的是判据的**非单调**：
    :meth:`_ungrounded_limit_offenders` 拿 ``state.tool_text`` 当事实基准，
    而 tool_text 会随重试轮继续增长 —— 同一段文本先被拦下、后来又能通过。
    实测到的复活链：

        第 1 轮  调差标工具，返回 600
        第 2 轮  例外轮，正文写「酒店差标是 500 元/晚」→ 拦下 ✓
        第 3 轮  模型把 500 当**参数**喂给差标工具，工具在返回里回显了它
        兜底     tool_text 里「有」500 了 → 同一个判据改口 → 原样发给用户 ✗

    「被模型写进参数、又被工具回显」不等于工具**认定**这个数，但对判据
    来说两者长得一样。所以修法是记下**拦过哪段文本**（
    ``_ReplyState.blocked_texts``），兜底只认既往记录 —— 它不受 tool_text
    怎么长的影响。

    ⚠️ 特意把重说轮也摆进来（模型去问工具、工具回显），而不是直接
    空轮 → 兜底：不走这条链的用例只用即时判据就能拦住，
    测不到 F4 要防的东西。
    """
    draft = "酒店差标是 500 元/晚，请确认提交。"
    echo = '{"hotel_limit_per_night": 500, "policy_note": "单晚不超过 500 元"}'
    # ⚠️ 先把前提钉死：兜底那一刻，**即时判据**对这段文本已经改口成
    # 「有依据」。少了这条断言，将来谁把回显文本改一改，用例就会退化成
    # 「即时判据也能拦住」的重复覆盖，而 F4 的复活路径重新失守也没人知道。
    assert _ungrounded_limit_claims(draft, echo) == [], (
        "回显的 500 没能让闸门改口 —— 这条用例就没在测非单调复活，"
        "请检查 echo 里是否真的回显了 500"
    )

    events = [
        _reply_start(),
        _call_start(),
        _tool_call(_POLICY_TOOL),
        _call_end(),
        _tool_result(_LIMIT_TOOL_TEXT),          # 真实顺序：返回在结算之后
        _call_start(),
        _tool_call("submit_approval"),
        *_text(draft, block_id="block-2"),       # 例外轮 → 闸门拦下并记账
        _call_end(),
        _tool_result("已提交待确认"),
        _reply_end(),                            # 被吞 → 重说
        _call_start(),
        _tool_call(_POLICY_TOOL),                # 模型把 500 当参数喂给工具
        _call_end(),
        _tool_result(echo),                      # 工具回显 → tool_text 里有了 500
        _call_start(),                           # 空的一轮
        _call_end(),
        _reply_end(),                            # 重试用尽 → 兜底
    ]
    agent = _FakeAgent(
        tools={
            _POLICY_TOOL: _FakeTool(is_read_only=True),
            "submit_approval": _FakeTool(is_read_only=False),
        },
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(max_retries=1), events, agent))

    visible = _visible(out)
    assert "500" not in visible, (
        f"被拦下的草稿靠工具回显洗白了：{visible!r}"
    )
    assert visible.strip(), "兜底不能让用户什么都收不到"


@pytest.mark.parametrize(
    ("inputs", "expected"),
    [
        # ── 该取到的：普通用户消息 ──
        (Msg(name="user", role="user", content=[TextBlock(text="我订了 500 的酒店")]), "我订了 500 的酒店"),
        (
            [
                Msg(name="user", role="user", content=[TextBlock(text="第一句")]),
                Msg(name="user", role="user", content=[TextBlock(text="第二句")]),
            ],
            "第一句\n第二句",
        ),
        # ── 该取不到的：结构里没有用户文本，返回空串即可（只会更保守）──
        (None, ""),
        ([], ""),
        # 没有文本块的消息（图片/文件类输入）不参与。
        (Msg(name="user", role="user", content=[]), ""),
    ],
)
def test_user_text_extraction_covers_every_input_shape(
    inputs: Any, expected: str
) -> None:
    """★ ``inputs`` 的每一种形状都要有确定行为，且取不到时返回空串。

    ⚠️ **必须包含取不到的那几种**：框架传给 ``on_reply`` 的 ``inputs`` 在
    HITL 续答等场景下压根不是 ``Msg``（``agent/_agent.py:899-906`` 的联合
    类型里有三种事件）。若这里抛异常，守卫会在**用户确认订单之后那一轮**
    崩掉；若这里返回了别的值（比如把事件 ``str()`` 化），用户文本里就会混进
    一堆结构体文本，豁免的判据随之失真。空串是唯一的正确答案。
    """
    from src.orchestration.reply_guard import _user_text_from_inputs

    assert _user_text_from_inputs(inputs) == expected


def test_guard_reads_user_text_from_reply_inputs() -> None:
    """★★ 接线用例：守卫必须真的从 ``input_kwargs`` 里取到用户原话。

    ⚠️ 上面那条纯函数用例证明不了接线 —— 参数默认空串，就算 ``on_reply``
    根本不读 ``inputs``，纯函数也照样绿、``state.user_text`` 也照样是空串，
    唯一的症状是**正确的回答被反复拦掉**。所以这里跑同一条事件流两遍：
    带用户原话 → 放行且不重试；不带 → 拦下并重试。
    """
    events = [
        _reply_start(),
        _call_start(),
        _tool_call("check_travel_policy"),
        _tool_result(_LIMIT_TOOL_TEXT),
        _call_end(),
        _call_start(),
        *_text(_ANSWER_QUOTING_USER_PRICE, block_id="block-2"),
        _call_end(),
        _reply_end(),
    ]

    def run(inputs: Any) -> tuple[str, _FakeAgent]:
        agent = _FakeAgent(
            tools={"check_travel_policy": _FakeTool(is_read_only=True)},
        )
        out = asyncio.run(_run(ReplyGuardMiddleware(max_retries=1), events, agent, inputs))
        return _visible(out), agent

    # ── 对照组：拿不到用户原话 → 这个 500 就是「工具没给过的数」──
    blocked_text, blocked_agent = run(None)
    assert "500" not in blocked_text, (
        "对照组失败：不带用户原话时这句话本该被判成编造，"
        f"否则下面的断言证明不了接线（实际发出 {blocked_text!r}）"
    )
    assert blocked_agent.state.context_calls, "对照组本该触发一次重试"

    # ── 实验组：带上用户原话 → 500 有合法来源，原样发出 ──
    user_inputs = Msg(
        name="user",
        role="user",
        content=[TextBlock(text=_USER_PRICE_TEXT)],
    )
    kept_text, kept_agent = run(user_inputs)
    assert kept_text == _ANSWER_QUOTING_USER_PRICE, (
        f"正确的回答被闸门改坏了：{kept_text!r}"
    )
    assert not kept_agent.state.context_calls, (
        "用户自己报过的数字不该触发重试 —— 白烧一轮迭代和几秒延迟"
    )


def test_a_fabrication_is_still_blocked_when_the_user_mentioned_a_bigger_number() -> None:
    """★★ 端到端复现对抗验证的反例：用户「预算 5000」不能豁免「差标 500」。

    ⚠️ 这是**子串匹配**缺陷最恶劣的形态：用户随口提到一个更大的数
    （预算、手机号），模型编的那个比真实差标更小的上限就自动获得了
    「用户说过」的身份，原样发给用户。用户看到的是「酒店差标是 500 元/晚」
    —— 一个比制度里真实的 600 更小、且写得像官方口径的数字，
    直接会让他订错房。修复前实测：`visible='酒店差标是 500 元/晚。'`、
    `retried=False`（原样发出，连重试都没有）。

    ⚠️ 断言落在**最终发给用户的那段话**上（``_visible``），不是纯函数返回值：
    纯函数用例证明不了接线，而这里的缺陷恰好横跨两处（抽取用户数字 + 判据）。
    """
    events = [
        _reply_start(),
        _call_start(),
        _tool_call("check_travel_policy"),
        _tool_result(_LIMIT_TOOL_TEXT),
        _call_end(),
        _call_start(),
        *_text("酒店差标是 500 元/晚。", block_id="block-2"),
        _call_end(),
        _reply_end(),
        # 被点名后模型改正。
        _call_start(),
        *_text("酒店差标是不超过 600 元/晚。", block_id="block-3"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={"check_travel_policy": _FakeTool(is_read_only=True)},
    )
    user_inputs = Msg(
        name="user",
        role="user",
        content=[TextBlock(text="我这次出差预算是 5000 元，帮我看看住宿标准。")],
    )

    out = asyncio.run(
        _run(ReplyGuardMiddleware(max_retries=1), events, agent, user_inputs)
    )

    visible = _visible(out)
    assert "500" not in visible, f"编造的上限发给了用户：{visible!r}"
    assert "600" in visible, f"改正后的正文没发出去：{visible!r}"
    assert agent.state.context_calls, "这条编造必须触发重说，不能原样放行"


def test_a_computed_total_passes_the_gate_without_a_retry() -> None:
    """★★ 端到端复现另一侧反例：正确的合计**不该**被拦。

    ⚠️ 修复前实测：这句正确的回答连续两轮被拦，重试耗尽后兜底又把它当
    「无依据的草稿」跳过，用户最终看到的是
    「抱歉，这轮我没能整理出可用的答复」——一次**正确**的回答被系统自己删掉了。
    这类伤害（用户什么都没有）比放过一次编造更直接，所以单独立一条端到端用例。

    ⚠️ 用 ``max_retries=0``：闸门若误判，症状就是「正文一个字都不剩」，
    不需要靠重试次数来证明；顺带把「不该重试」也钉住（白烧一轮迭代）。
    """
    answer = "两晚住宿标准合计 1200 元，平均每晚 600 元。"
    events = [
        _reply_start(),
        _call_start(),
        _tool_call("check_travel_policy"),
        _tool_result(_LIMIT_TOOL_TEXT),
        _call_end(),
        _call_start(),
        *_text(answer, block_id="block-2"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={"check_travel_policy": _FakeTool(is_read_only=True)},
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(max_retries=0), events, agent))

    assert _visible(out) == answer, (
        "算出来的合计被闸门拦掉了 —— 正确的回答被系统自己删除"
    )
    assert not agent.state.context_calls, "正确的回答不该触发重说"


def test_limit_gate_is_transparent_when_guard_is_disabled() -> None:
    """★ 关掉守卫时不能留下任何副作用。

    ``ALIGO__ORCHESTRATION__REPLY_GUARD_ENABLED=false`` 是排障开关，
    必须「与没装这个中间件完全等价」—— 包括不新增重试。
    """
    events = [
        _reply_start(),
        _call_start(),
        _tool_call("check_travel_policy"),
        _tool_result(_LIMIT_TOOL_TEXT),
        _call_end(),
        _call_start(),
        *_text("住宿标准：酒店单晚上限 500 元。", block_id="block-2"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={"check_travel_policy": _FakeTool(is_read_only=True)},
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(enabled=False), events, agent))

    assert _visible(out) == "住宿标准：酒店单晚上限 500 元。"
    assert not agent.state.context_calls, "关掉守卫后不该有任何重试"


# ---------------------------------------------------------------------------
# 四·补：动态 Prompt 注入过的数字（缺陷 P1）
# ---------------------------------------------------------------------------
# ⚠️ 这一组用例守的是「**我们自己**写进 prompt 的用户数据」这条合法来源。
# 现场（2026-10-04 实测）：用户在前一轮说过「预算 15000」，此后每一轮的
# 动态 Prompt 都会注入「预算上限：15000 元」；模型照抄这句**完全正确**的话，
# 却因为没有出现在**本轮**用户文本里而被判成编造 —— 拦下 → 重说 →
# 模型坚持引用 → 用户最终收到一句道歉兜底。
INJECTED_BUDGET_TEXT = "预算上限：15000 元"


def test_the_injected_prompt_text_is_a_legal_source() -> None:
    """★ 四个合法来源里的第四个：我们自己注入进 prompt 的数字。

    ⚠️ 两个方向都要钉：
      · **有** ``prompt_text`` 时，15000 必须放行；
      · **没有**时，同一句话必须照旧拦下 —— 少了后者，把来源写成
        「有 prompt_text 就整段放行」的变异体会全绿通过。
    """
    answer = "根据你的预算上限是 15000 元来看，这趟行程够用。"
    user_turn = "这家 550 一晚的酒店能报吗？"

    assert _ungrounded_limit_claims(answer, _LIMIT_TOOL_TEXT, user_turn) == ["15000"], (
        "不传 prompt_text 时行为必须与从前一致（只会更保守）"
    )
    assert (
        _ungrounded_limit_claims(
            answer,
            _LIMIT_TOOL_TEXT,
            user_turn,
            INJECTED_BUDGET_TEXT,
        )
        == []
    ), "动态 Prompt 里注入过的用户数字被当成了编造"


def test_the_prompt_source_exempts_only_the_numbers_it_actually_carries() -> None:
    """★ 来源 4 **只**豁免它自己带的那几个数，不是「有 prompt_text 就放宽」。

    ⚠️ 这是最容易写错的一版：闸门里加一句「prompt_text 非空 → 返回 []」，
    上面那条用例照样绿，而闸门实际上被整个关掉了。所以这里让 prompt 侧
    携带 15000、正文却写 500（工具给的是 600），500 必须**仍然**被拦。
    """
    answer = "住宿标准：酒店单晚上限 500 元。"

    claims = _ungrounded_limit_claims(
        answer,
        _LIMIT_TOOL_TEXT,
        "",
        INJECTED_BUDGET_TEXT,
    )

    assert claims == ["500"], f"豁免范围溢出了它携带的数字：{claims}"


def test_an_injected_budget_reaches_the_user_end_to_end() -> None:
    """★★ 端到端：动态 Prompt 注入 → 守卫放行，用户拿到完整答复。

    ⚠️ 这条用例把**两段真实代码**接在一起跑，而不是直接往 ``middle_context``
    里塞一个字典：``ContextInjectionMiddleware.on_system_prompt``（写登记）
    与 ``ReplyGuardMiddleware``（读登记）。只测其中一半的话，「两边键名
    对不上」这种最常见的接线错误会漏过去 —— 而它的症状正是缺陷 P1 原样
    复现（正确的答复被拦、用户收到道歉）。

    ⚠️ ``agent.state.reply_id`` 要在调用 prompt 中间件**之前**设好：框架在
    回复开始时就把 id 写进状态（``AgentState.reply_id`` 是属性，代理到
    ``reply_context``），两个中间件因此看到同一个 id。登记与读取都按这个
    id 校验新鲜度。
    """
    from src.orchestration.context import ContextInjectionMiddleware, PromptContext
    from src.orchestration.context import recorded_prompt_numbers

    answer = "可以报。酒店 550 元/晚 符合差标；你的预算上限是 15000 元，够用。"
    events = [
        _reply_start(),
        _call_start(),
        _tool_call("check_travel_policy"),
        _tool_result(_LIMIT_TOOL_TEXT),
        _call_end(),
        _call_start(),
        *_text(answer, block_id="block-2"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={"check_travel_policy": _FakeTool(is_read_only=True)},
    )
    agent.state.reply_id = REPLY_ID
    request = TravelRequest(budget=15000.0)
    prompt_mw = ContextInjectionMiddleware(
        resolver=lambda _agent: PromptContext(request=request),
    )

    # 1) 框架的每轮推理：动态 Prompt 拼装 + 登记。
    asyncio.run(prompt_mw.on_system_prompt(agent, "你是差旅助手。"))
    assert "15000" in recorded_prompt_numbers(agent, REPLY_ID), (
        "prompt 中间件没有登记注入过的预算数字 —— 守卫那边会当成编造"
    )

    # 2) 同一轮回复里的结算：守卫必须放行。
    out = asyncio.run(_run(ReplyGuardMiddleware(max_retries=1), events, agent))

    assert _visible(out) == answer, (
        "引用我们自己注入的预算被闸门拦下了（缺陷 P1 复现）"
    )
    assert not agent.state.context_calls, "正确的答复不该触发重说"


def test_a_stale_prompt_record_does_not_exempt_anything() -> None:
    """★★ ``middle_context`` 是跨回复存活的 —— 旧登记**绝不能**当豁免。

    ⚠️ 这条守的是「新鲜度校验」这一侧。``AgentState.middle_context`` 随会话
    持久化，上一条回复登记的可能是完全无关的数字（用户上一轮报的预算、
    另一段行情）。拿旧记录当豁免来源，等于给闸门开一个「随口报数」的口子，
    而且只在「上一条回复恰好也走了动态 Prompt」时出现 —— 最难查的那种
    间歇性漏网。

    ⚠️ 判据往严的一侧倒：对不上就返回空列表（更保守、可能多拦一次正确
    答复），而不是「拿不准就当有效」（会放走编造）。两个方向的代价不对称。
    """
    from src.orchestration.context import ContextInjectionMiddleware, PromptContext

    agent = _FakeAgent(
        tools={"check_travel_policy": _FakeTool(is_read_only=True)},
    )
    # 登记发生在**上一条**回复里。
    agent.state.reply_id = "reply-旧"
    prompt_mw = ContextInjectionMiddleware(
        resolver=lambda _agent: PromptContext(request=TravelRequest(budget=15000.0)),
    )
    asyncio.run(prompt_mw.on_system_prompt(agent, "你是差旅助手。"))
    assert agent.state.middle_context, "前置条件不成立：登记本身就没写进去"

    # 新回复沿用同一个 agent（middle_context 被带过来了）。
    agent.state.reply_id = REPLY_ID
    events = [
        _reply_start(),
        _call_start(),
        _tool_call("check_travel_policy"),
        _tool_result(_LIMIT_TOOL_TEXT),
        _call_end(),
        _call_start(),
        *_text("你的预算上限是 15000 元。", block_id="block-2"),
        _call_end(),
        _reply_end(),
    ]

    out = asyncio.run(_run(ReplyGuardMiddleware(max_retries=0), events, agent))

    assert "15000" not in _visible(out), (
        "上一条回复的登记被当成了本次回复的豁免来源"
    )
