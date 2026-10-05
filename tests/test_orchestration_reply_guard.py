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
import logging
import time
from types import SimpleNamespace
from typing import Any, AsyncGenerator

import pytest
from agentscope.agent import Agent
from agentscope.event import (
    DataBlockStartEvent,
    EventBase,
    HintBlockEvent,
    ModelCallEndEvent,
    ModelCallStartEvent,
    ReplyEndEvent,
    ReplyStartEvent,
    TextBlockDeltaEvent,
    TextBlockEndEvent,
    TextBlockStartEvent,
    ThinkingBlockDeltaEvent,
    ThinkingBlockEndEvent,
    ThinkingBlockStartEvent,
    ToolCallDeltaEvent,
    ToolCallEndEvent,
    ToolCallStartEvent,
    ToolResultEndEvent,
    ToolResultStartEvent,
    ToolResultTextDeltaEvent,
)
from agentscope.message import Msg, TextBlock, ToolCallBlock, ToolResultState
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
    _has_internal_marker,
    _is_derived_total,
    _is_draft,
    _is_placeholder,
    _normalize_amount,
    _residual_marker_count,
    _ReplyState,
    _strip_drafts,
    _strip_internal_markers,
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

    ⚠️ ``input_kwargs`` 的键必须与框架一致（``agentscope/agent/_agent.py:939-942``
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
      （``agentscope/agent/_agent.py:1105-1118`` 只在「新回复」分支发它；续答走
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
    ``ReplyEndEvent``（``agentscope/agent/_agent.py:3554-3578``），连吞两次且中间没有进展
    就会抛 ``RuntimeError``（``agentscope/agent/_agent.py:1165-1173``）。重说也改变不了结局 ——
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
    HITL 续答等场景下压根不是 ``Msg``（``agentscope/agent/_agent.py:899-906`` 的联合
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


# ==============================================================================
# 五、内部标记清洗（2026-10-05 缺陷 E）
# ==============================================================================

#: 2026-10-05 线上工单的原文标记（见模块文档缺陷 E）：
#: 模型在工具轮之后把 agent 框架的「工具结果已清理」标记当正文复读了出来。
_MEASURED_MARKER = "[tool_result cleared by postprune]"


@pytest.mark.parametrize(
    ("text", "expected", "expected_removed"),
    [
        # ① 线上原文：标记自成一段 + 后接正常答复
        (
            f"{_MEASURED_MARKER}\n\n好的，杭州两天出差。还差两个信息：\n\n"
            "**你从哪个城市出发？**",
            "好的，杭州两天出差。还差两个信息：\n\n**你从哪个城市出发？**",
            1,
        ),
        # ① 同族变体：qwen-code 公开存在的标记（形状相同、机制名不同）
        ("[Old tool result content cleared]\n\n订单已确认。", "订单已确认。", 1),
        # ① 变体：带统计的清理标记 / 其他清理动词
        ("[tool output cleared: 4213 chars]", "", 1),
        ("[tool_result pruned]", "", 1),
        ("[tool_result truncated by postprune]", "", 1),
        ("[Old inline media cleared: image/png]", "", 1),
        # ① 词表补漏（2026-10-05 对抗验证 F7）：清理动词变体此前整族漏网
        ("[tool_result cleaned up by the system]", "", 1),
        ("[tool result removed to save space]", "", 1),
        ("[tool_result shortened]", "", 1),
        ("[tool result summarized]", "", 1),
        # ① 全角方括号
        ("［tool_result cleared by postprune］好的。", "好的。", 1),
        # ② 句式兜底：机制名不可穷举。
        # ⚠️ 2026-10-05（F6）：这两个参数是**纯 ②** 命中 —— 名词不在 ① 表里，
        # 删掉 ② 正则它们必红。此前的 ``[context cleared by microcompaction]``
        # 是 ① 顺手命中的（逐模式归因 [1,1,0,0]），把 ② 整条删掉测试仍全绿，
        # 该参数的变异声明失实（见下方函数 docstring）。
        ("[cache cleared by sanitizer]", "", 1),
        ("[history cleared by microcompaction]", "", 1),
        ("[context cleared by microcompaction]\n\n你好。", "你好。", 1),
        # ③ 系统标签（成对）——框架的注入块/背景工具占位符形状
        (
            "<system-reminder>Tool 'x' is running in background (id=1)"
            "</system-reminder>\n\n好的。",
            "好的。",
            1,
        ),
        # ④ 系统标签（残缺）
        ("好的。<system-info>", "好的。", 1),
        # 同段混排：标记与真答案在一段里 —— 只删跨度，不整段删
        (
            "好的。[tool_result cleared by postprune] 你从哪个城市出发？",
            "好的。 你从哪个城市出发？",
            1,
        ),
        # 多处命中
        (
            f"{_MEASURED_MARKER}\n\n好的。\n\n[tool_result pruned]",
            "好的。",
            2,
        ),
        # ── 假阳性防护：这些一个字符都不能动 ──
        ("【重点】酒店差标是 600 元/晚。", "【重点】酒店差标是 600 元/晚。", 0),
        (
            "[订票入口](https://example.com) 在这里。",
            "[订票入口](https://example.com) 在这里。",
            0,
        ),
        ("会议安排在 [14:00]，别迟到。", "会议安排在 [14:00]，别迟到。", 0),
        ("[重要] 请携带身份证。", "[重要] 请携带身份证。", 0),
        ("系统提示：<b>加粗</b>不是标签泄漏。", "系统提示：<b>加粗</b>不是标签泄漏。", 0),
        # ── 对抗验证实测的误删回归（F2/F3，2026-10-05）：初版形状表把这些
        #    真内容整段删过；收紧后一个字符都不许动 ──
        ("[Luggage cleared by customs at 14:30] 已入境。", "[Luggage cleared by customs at 14:30] 已入境。", 0),
        (
            "[Payment cleared by ICBC](https://bank.example.com) 已到账。",
            "[Payment cleared by ICBC](https://bank.example.com) 已到账。",
            0,
        ),
        ("[重要通知：行李已 cleared by 海关 放行] 谢谢。", "[重要通知：行李已 cleared by 海关 放行] 谢谢。", 0),
        ("[Results: clear] 航班准点。", "[Results: clear] 航班准点。", 0),
        ("[Context unclear] 请问从哪个城市出发？", "[Context unclear] 请问从哪个城市出发？", 0),
        ("[Result: visa cleared] 可以出行。", "[Result: visa cleared] 可以出行。", 0),
        ("[Media: photo compressed] 已压缩发送。", "[Media: photo compressed] 已压缩发送。", 0),
        (
            "[Results clearly show option B](https://example.com) 更省。",
            "[Results clearly show option B](https://example.com) 更省。",
            0,
        ),
        ("[Tool clearance required] 请补充授权。", "[Tool clearance required] 请补充授权。", 0),
        # ⚠️ 已知表外变种（刻意不删）：① 的名词表只认 ``tool…`` 与
        # ``inline media`` 全称，**不认裸 media**（与不认裸 results/outputs/
        # contexts 同策：泛化英文名词误伤面太宽，F3 的 ``[Media: photo
        # compressed]`` 是真内容的判定同理）。这条是**边界声明**：某天扩表
        # 把裸 media 加回来，这里会红 —— 扩表的人必须同时改这条注释与
        # residual 计数器的预期。可见性由 residual_internal_marker 兜底
        # （见 test_shape_table_escapees_are_counted_not_silently_clean）。
        ("[Media cleared: 3 images] 已发送。", "[Media cleared: 3 images] 已发送。", 0),
        # ── 多条目/边界覆盖（2026-10-05 第二轮对抗验证 tests-metrics 发现
        #    第 10 条：六个单行删除各自保持全绿 = 这些条目此前零覆盖）──
        # ① 的清理动词表含 redact（完整屈折形式）；删掉该词条这里会红
        ("[tool_result redacted]", "", 1),
        # ① 的名词后缀链含 data|text|blocks?（tool result data 三连）；
        # 删掉后缀链这里会红
        ("[tool result data cleared]", "", 1),
        # ① 的尾部内容上限 {0,80}：动词后 31 字符仍在界内；
        # 上限被改小（≤31）这里会红
        ("[tool_result pruned, annotation padding0123456789]", "", 1),
        # ④ 的属性上限 {0,400}：350 字符属性仍在界内；
        # 上限被改小（≤360）这里会红（③ 管成对、管不到未闭合开标签）
        (
            "好的。<system-reminder data-x='" + "A" * 350 + "'>",
            "好的。",
            1,
        ),
        # ── 接缝的非边界分支（tests-metrics 第 6 条：此前接缝用例全在
        #    文首/文尾，中间接缝的两条分支零覆盖）──
        # 独占一行的标记 = 两侧各一个换行 → 合并成一个（段落不合并、不加空行）；
        # 跨行合成规则被改成 min(1) 这里会红
        (
            "1. 酒店 600 元/晚\n[tool_result pruned]\n2. 机票 1200 元",
            "1. 酒店 600 元/晚\n2. 机票 1200 元",
            1,
        ),
        # 两侧都是空行（l_nl=r_nl=2）→ 至多保留一个空行（段落分隔保住）；
        # min(2,…) 被改成 min(1,…) 这里会红
        ("上文\n\n[tool_result pruned]\n\n下文", "上文\n\n下文", 1),
        # 下一行的缩进是内容（列表续行），不许吞
        (
            "1. 酒店 600 元\n[tool_result pruned]\n   备注：含早餐",
            "1. 酒店 600 元\n   备注：含早餐",
            1,
        ),
        # 上一行的行尾双空格 = markdown 硬换行，不许吞
        ("行一  \n[tool_result pruned]\n行二", "行一  \n行二", 1),
        # 同行接缝：两侧本来就没有空白 → 一个空格都不许插；
        # 「同行恒加空格」的变异这里会红
        ("前。[tool_result pruned]后。", "前。后。", 1),
        # P2b（tests-metrics 之外的独立实测）：文首缩进在**远端**，
        # 与标记无关；`_strip_internal_markers` 结尾的全局 strip() 这里会红
        (
            "    def f():\n        return 1\n\n[tool_result cleared by postprune]\n\n好的。",
            "    def f():\n        return 1\n\n好的。",
            1,
        ),
    ],
)
def test_internal_markers_are_removed_by_shape(
    text: str,
    expected: str,
    expected_removed: int,
) -> None:
    """★★ harness 标记家族一律剥除；中文括号 / markdown 链接一刀不动。

    ⚠️ 为什么必须列假阳性参数：这个函数删的是**用户可见的每一个字**，
    而它的形状表**故意开得窄**（只认方括号标记与 system 标签）——「窄」
    必须被测试钉住，否则下一次扩表时没人知道边界曾经在哪。
    中文括号（``【】``）与 markdown 链接在差旅语境里太常见，误伤的代价
    大于漏掉一个变种标记。

    ⚠️ 变异（2026-10-05 对抗验证 F6 修正过一次**失实声明**；本组参数随后
    经 12 处修复点逐点变异复验，全部哨兵命中）：把形状表放宽成「任何方括号」
    → 后半段假阳性参数红；删掉 ② 正则 → ``[cache cleared by sanitizer]`` /
    ``[history cleared by microcompaction]`` / ``[context cleared by
    microcompaction]`` 三个参数红（此前文档把第三个当 ② 的哨兵，实测它在
    旧表下由 ① 命中 —— 删 ② 后全绿，哨兵失效）；删掉 ① 的名词要求或 ② 的
    机制名要求 → F2/F3 回归参数成片红；裸 ``media`` 加回名词表 →
    ``[Media cleared: 3 images]`` 边界参数红。（F3 的 ``[Results: clear]``
    靠的是**完整屈折动词表**：删掉它、只配对裸词干 ``clear`` 才会红。）
    ⚠️ 2026-10-05 第二轮对抗验证（tests-metrics 第 10 条）又补了六组参数：
    ① 的 redact 动词 / ``data|text|blocks?`` 后缀链 / 尾部 ``{0,80}`` 上限、
    ④ 的属性 ``{0,400}`` 上限、中间接缝的两条非边界分支（跨行合成、
    同行不加空格）、文首缩进的远端保全（P2b）—— 这些此前都是「删掉对应
    单行、245 条仍全绿」的覆盖缺口。
    """
    cleaned, removed = _strip_internal_markers(text)

    assert removed == expected_removed, (text, cleaned, removed)
    assert cleaned == expected, (text, cleaned)


def test_marker_stripping_is_idempotent_and_leaves_clean_text_alone() -> None:
    """剥过的文本再剥一次一字不动；干净文本原样返回（返回同一对象）。

    ⚠️ 幂等是**兜底二洗**可存在的前提（``_fallback_if_silent`` 会再洗
    一遍）；「干净文本原样返回」是 ``clean`` 指标语义（一字未改）的
    函数级保证。
    """
    cleaned, removed = _strip_internal_markers(f"{_MEASURED_MARKER}\n\n好的。")
    assert (cleaned, removed) == ("好的。", 1)

    again, removed_again = _strip_internal_markers(cleaned)
    assert again == cleaned
    assert removed_again == 0

    text = "好的，杭州两天出差。\n\n你从哪个城市出发？"
    untouched, removed_clean = _strip_internal_markers(text)
    assert removed_clean == 0
    assert untouched is text, "干净文本必须原样返回（函数契约：未命中时返回入参本身）"


def test_the_measured_marker_is_blocked_end_to_end() -> None:
    """★★ 缺陷 E 的**端到端**版本：线上那条回复的形状不许再上屏。

    ⚠️ 三件事一起查，缺一不可：

    1. 用户可见文本里**没有** ``postprune``（这是工单本身）；
    2. 答复部分**完整保留**（不能把标记连同答案一起删）；
    3. 事件是**重新合成**的 —— 原始事件里带着标记，复用它们等于没洗
       （``_settle_round`` 末尾 ``cleaned == raw_text`` 判据的靶子；
       ``block_id`` 必须换成 ``guard-`` 前缀的新块）。
    """
    answer = "好的，杭州两天出差。还差两个信息：\n\n**你从哪个城市出发？**"
    events = [
        _reply_start(),
        _call_start(),
        *_text(f"{_MEASURED_MARKER}\n\n{answer}"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert "postprune" not in _visible(out), f"内部标记漏给了用户：{_visible(out)!r}"
    assert _visible(out) == answer, "标记连同真答案被一起删了（静默丢答案）"
    assert any(isinstance(e, ReplyEndEvent) for e in out), (
        "正文已经发出，结束事件却被吞了 —— 用户会听到两遍答复"
    )
    block_ids = {
        e.block_id
        for e in out
        if isinstance(e, (TextBlockStartEvent, TextBlockDeltaEvent, TextBlockEndEvent))
    }
    assert block_ids == {"guard-1-" + REPLY_ID}, (
        "原始（带标记的）事件被复用了 —— 洗了但白洗"
    )


def test_marker_only_reply_is_swallowed_and_retried() -> None:
    """★★ 整轮只剩一个内部标记 = 没有可用内容 → 吞掉重说（不给用户空白）。

    ⚠️ 剥完为空之后必须走既有的 ``empty`` 拒绝路径（重试 → 兜底），而不是
    「把空文本发出去」或「把标记发出去」。这条钉的就是接线顺序：清洗发生
    在 :func:`_answer_problem` **之前**（``_settle_round`` 开头）。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_text(f"{_MEASURED_MARKER}\n\n"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert not any(isinstance(e, ReplyEndEvent) for e in out), (
        "空答复没有被吞掉 —— 框架不会重说，用户将面对空白"
    )
    assert "postprune" not in _visible(out)
    assert agent.state.context_calls, "重说前必须往上下文里塞纠正指令"


def test_fallback_never_resurrects_an_internal_marker() -> None:
    """★★ 兜底路径对内部标记**再洗一遍**（防御性二洗）。

    ⚠️ 今天 ``state.drafts`` 存进来的都是洗过的文本，所以这条二洗按构造
    不可达 —— 但兜底是「重试用尽」的末端路径（比正文更晚、更少人盯），
    一旦将来有别的路径把原文塞进 ``drafts``，这里是最后一道闸。脚本事件
    流构造不出「drafts 里带标记」的形状，所以直接调方法钉住该路径。
    """
    state = _ReplyState()
    state.reply_id = REPLY_ID
    state.drafts = [f"{_MEASURED_MARKER}\n\n订单已确认，10 月 12 日出票。"]

    out = ReplyGuardMiddleware(max_retries=0)._fallback_if_silent(_FakeAgent(), state)

    assert _visible(out) == "订单已确认，10 月 12 日出票。"
    assert "postprune" not in _visible(out)


def test_marker_removal_is_observed_and_never_counted_as_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ 「洗掉标记」必须进指标（``stripped_internal_marker``），且与 ``clean`` 互斥。

    ⚠️ 为什么这个指标不能省：标记被删掉之后，正文和日志里都**留不下痕**
    —— 「模型开始复读标记」这件事唯一的可见性就是这条曲线（以及一条
    WARNING）。而 ``clean`` 的语义是「一个字符都没改」，洗过标记的轮次
    绝不能记成它，否则缺陷 E 复发时监控上什么都看不到。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )

    events = [
        _reply_start(),
        _call_start(),
        *_text(f"{_MEASURED_MARKER}\n\n好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    assert "stripped_internal_marker" in seen, seen
    assert "clean" not in seen, seen
    assert _visible(out) == "好的。"

    # 干净轮次反向：记 ``clean``、绝不记标记指标（互斥的另一半）。
    seen.clear()
    clean_events = [
        _reply_start(),
        _call_start(),
        *_text("住宿标准：单晚不超过 600 元。"),
        _call_end(),
        _reply_end(),
    ]
    asyncio.run(_run(ReplyGuardMiddleware(), clean_events, _FakeAgent()))

    assert "clean" in seen, seen
    assert "stripped_internal_marker" not in seen, seen


# ------------------------------------------------------------------------------
# 对抗验证（2026-10-05，wf_f9b61a1c-1e2）确认缺陷的回归用例
# ------------------------------------------------------------------------------
def test_nested_marker_is_removed_to_a_fixed_point() -> None:
    """★★ 嵌套标记**一次调用**就剥干净（对抗验证 F8）。

    ⚠️ 初版按「找一个、删一个」单趟处理：``[tool_result pruned [tool_result
    pruned]]`` 内层先被删、外层只剩一个空壳 ``[tool_result pruned ]`` ——
    残渣照样上屏，要用户看到第二次调用才干净。现在删到不动点（每删一处
    从头重搜最早匹配），这条钉住收敛性。
    """
    cleaned, removed = _strip_internal_markers(
        "[tool_result pruned [tool_result pruned]]"
    )
    assert (cleaned, removed) == ("", 2)

    # ⚠️ 三层嵌套（tests-metrics 第 10 条）钉的是**结果**：嵌套形状必须
    # 一次调用剥干净。⚠️ 2026-10-05 变异 M1 复测：``while True`` 改成
    # ``for _ in range(2)`` 时**这条不再变红**（批量删除的当前实现两趟就
    # 能收掉三层）—— 趟数敏感度现在由 ``test_nested_marker_cascade_is_not
    # _superlinear``（深度 2000 的级联）承担，M1 的哨兵已改指它。
    three, removed_three = _strip_internal_markers(
        "[tool_result pruned [tool_result pruned [tool_result pruned]]]"
    )
    assert (three, removed_three) == ("", 3)


def test_seam_splicing_preserves_markdown_on_the_far_side() -> None:
    """★★ 删标记是**接缝级**缝合，远端 markdown 一个字符不动（F1）。

    ⚠️ 初版删完标记后对**全文**做 ``[ \\t]{2,}`` → 单空格、``\\n{3,}`` →
    双换行的「整理」：代码围栏里的四空格缩进被吃成 ``x = 1``
    （``ast.parse`` 直接 IndentationError）、嵌套列表三空格缩进塌成一格
    （层级重排）、行尾双空格（markdown 硬换行）消失。这些都是**远端**的
    合法内容，与标记无关 —— 只有标记两侧的接缝空白该被吃掉。
    """
    code = (
        f"{_MEASURED_MARKER}\n\n"
        "```python\ndef f():\n    x = 1\n    return x\n```\n"
    )
    cleaned, removed = _strip_internal_markers(code)
    assert removed == 1
    assert "    x = 1" in cleaned, f"代码围栏缩进被吃了：{cleaned!r}"

    lst = "[tool_result pruned]\n\n- 一级\n  - 二级\n    - 三级\n"
    cleaned_list, _ = _strip_internal_markers(lst)
    assert "  - 二级" in cleaned_list and "    - 三级" in cleaned_list, (
        f"嵌套列表缩进被压平：{cleaned_list!r}"
    )

    hard = f"{_MEASURED_MARKER}\n\n行一  \n行二\n"
    cleaned_hard, _ = _strip_internal_markers(hard)
    assert "行一  \n行二" in cleaned_hard, (
        f"行尾双空格（markdown 硬换行）被吃掉：{cleaned_hard!r}"
    )


def test_system_tag_bounds_cannot_be_evaded_by_long_payloads() -> None:
    """★★ ③ 的内容/属性上限被实测绕过过（F9），加宽后不许再漏。

    ⚠️ 初版 ③ 的属性上限 120、内容上限 600：145 字符的 ``data-x`` 属性或
    601 字的内容让 ③ 整条失配，只剩 ④ 删掉开标签、**内容留在正文里** ——
    比不删更糟（``removed`` 计数还显示「拦下了 1 处」）。这里用 150/700
    字符复现该形状，两个都必须整条消失。
    """
    long_attr = (
        "<system-reminder data-x='" + "A" * 150 + "'>隐藏内容</system-reminder>"
        "\n\n好的。"
    )
    cleaned, removed = _strip_internal_markers(long_attr)
    assert (cleaned, removed) == ("好的。", 1)
    assert "A" * 150 not in cleaned

    long_body = f"<system-info>{'内' * 700}</system-info>\n\n好的。"
    cleaned_body, removed_body = _strip_internal_markers(long_body)
    assert (cleaned_body, removed_body) == ("好的。", 1), (
        f"长内容标签被绕过、内容留在正文：{cleaned_body!r}"
    )

    # ⚠️ 内容里的 ``<``（如 HTML 片段）不许让 ③ 整条失配 —— 内容字符类
    # 必须是 ``[\s\S]`` 而不是 ``[^<]``（F9 同一条修复的另一半，
    # 2026-10-05 tests-metrics 第 10 条复查时发现此处零覆盖）。
    with_html = "<system-info>内容 <b>加粗</b> 继续</system-info>\n\n好的。"
    cleaned_html, removed_html = _strip_internal_markers(with_html)
    assert (cleaned_html, removed_html) == ("好的。", 1), (
        f"内容含 < 的成对标签被绕过：{cleaned_html!r}"
    )


def test_marker_in_the_thinking_chain_never_reaches_the_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ 思考链是**用户可见且逐条落库**的上屏通道（对抗验证 F4 实测旁路）。

    ⚠️ 初版只缓冲文本块，``ThinkingBlockDeltaEvent`` 走泛型 ``yield`` 原样
    放行：同一条复读标记挪进思考链就绕过整道守卫（前端无条件渲染
    ``block.thinking``，服务端对每个事件 ``append_event``），该轮还会被记成
    ``clean``。修复后：思考块按块缓冲、块结束时结算，**只洗标记**（推理本身
    一字不动 —— 「我先查一下」这类措辞在思考链里是正常推理，不是草稿），
    命中时记 ``stripped_internal_marker``、事件重新合成。
    ⚠️ 并且思考链洗过标记的回复**不再记 ``clean``**（2026-10-05 第二轮
    对抗验证 tests-metrics 第 4 条：修前正文这一轮会照记 ``clean``，
    同一条「标记被拦下」的事件在不同通道给出不同动作）。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    answer = "好的，杭州两天出差。还差两个信息：\n\n**你从哪个城市出发？**"
    events = [
        _reply_start(),
        _call_start(),
        ThinkingBlockStartEvent(reply_id=REPLY_ID, block_id="t-1"),
        ThinkingBlockDeltaEvent(
            reply_id=REPLY_ID,
            block_id="t-1",
            delta=f"{_MEASURED_MARKER}\n用户要去杭州，先确认出发地。",
        ),
        ThinkingBlockEndEvent(reply_id=REPLY_ID, block_id="t-1"),
        *_text(answer),
        _call_end(),
        _reply_end(),
    ]

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    thinking = "".join(
        e.delta for e in out if isinstance(e, ThinkingBlockDeltaEvent)
    )
    assert "postprune" not in thinking, f"标记从思考链漏给了用户：{thinking!r}"
    assert "用户要去杭州，先确认出发地。" in thinking, "推理本身被连坐删了"
    assert _visible(out) == answer, "正文被思考链的清洗波及"
    assert "stripped_internal_marker" in seen, seen
    # ⚠️ 正文这一轮确实一字未改，但**整条回复**改过（思考链）——
    # ``clean`` 的语义是「一个字符都没改」，必须被抑制
    # （tests-metrics 第 4 条：修前实测 seen=['stripped_internal_marker',
    # 'clean']，与测试标题自称的互斥语义自相矛盾）。
    assert "clean" not in seen, seen
    block_ids = {
        e.block_id
        for e in out
        if isinstance(
            e,
            (ThinkingBlockStartEvent, ThinkingBlockDeltaEvent, ThinkingBlockEndEvent),
        )
    }
    assert block_ids == {"think-1-" + REPLY_ID}, (
        "洗过的思考链复用了原始事件（洗了但白洗），或块 id 无 think- 前缀"
    )


def test_marker_hidden_in_text_block_end_payload_is_blocked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ ``TEXT_BLOCK_END.text`` 会**覆盖**块内容（对抗验证 F5，潜在旁路）。

    ⚠️ 本轮所有内容判据只看 DELTA 拼接，而 ``Msg.append_event`` 在
    ``TEXT_BLOCK_END`` 上用 ``event.text`` 覆盖块内容：DELTA 干净、END 载荷
    带标记时 ``cleaned == raw_text`` 成立、旧事件原样复用 —— 实时流看着干净，
    **刷新后标记从历史消息里冒出来**。修复后 END 载荷带标记 = 不许复用，
    走重新合成（合成的 END 用洗过的文本），并记 ``stripped_internal_marker``
    （指标口径是「拦下了」，记成 clean 会让曲线骗人）。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    dirty_end = TextBlockEndEvent(
        reply_id=REPLY_ID,
        block_id="block-1",
        text=f"好的。{_MEASURED_MARKER}",
    )
    events = [
        _reply_start(),
        _call_start(),
        TextBlockStartEvent(reply_id=REPLY_ID, block_id="block-1"),
        TextBlockDeltaEvent(reply_id=REPLY_ID, block_id="block-1", delta="好的。"),
        dirty_end,
        _call_end(),
        _reply_end(),
    ]

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    assert "postprune" not in _visible(out)
    ends = [e for e in out if isinstance(e, TextBlockEndEvent)]
    assert ends and all("postprune" not in (e.text or "") for e in ends), (
        f"END 载荷里的标记原样发出：{[e.text for e in ends]!r}"
    )
    assert "stripped_internal_marker" in seen, seen
    assert {e.block_id for e in out if isinstance(e, TextBlockDeltaEvent)} == {
        "guard-1-" + REPLY_ID
    }, "END 载荷脏却没有重新合成 —— 复用等于没洗"


def test_tool_payload_deltas_are_scrubbed_in_place() -> None:
    """★★ 工具事件的文本载荷同样上屏且落库，标记就地洗掉。

    ⚠️ 两条载荷路径：``TOOL_RESULT_TEXT_DELTA``（执行链上屏）与
    ``TOOL_CALL_DELTA``（模型正在写的调用参数，前端会实时显示）。它们不走
    缓冲，只能就地洗「上屏副本」—— 用 ``model_copy`` 换掉 ``delta``，不改
    原始事件对象。工具返回的**其余内容**（如差标金额）必须原样保留：它是
    接地账本的来源。
    """
    tool_answer = f"{_MEASURED_MARKER}\n北京住宿标准：单晚不超过 600 元。"
    events = [
        _reply_start(),
        _call_start(),
        _tool_call("search_flights"),
        _tool_result(tool_answer),
        ToolCallDeltaEvent(
            reply_id=REPLY_ID,
            tool_call_id="call-search_flights",
            delta='{"marker": "[tool_result pruned]"}',
        ),
        *_text("我先查一下航班。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(tools={"search_flights": _FakeTool(is_read_only=True)})

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    tool_deltas = [
        e.delta for e in out if isinstance(e, ToolResultTextDeltaEvent)
    ]
    assert tool_deltas, "工具返回事件没有透传（执行过程必须可见）"
    assert all("postprune" not in d for d in tool_deltas), tool_deltas
    assert any("600 元" in d for d in tool_deltas), "标记清洗把工具返回的差标金额也吃了"

    call_deltas = [e.delta for e in out if isinstance(e, ToolCallDeltaEvent)]
    assert call_deltas and all("pruned" not in d for d in call_deltas), call_deltas

    assert not any(_has_internal_marker(e.delta) for e in out if hasattr(e, "delta"))


def test_shape_table_escapees_are_counted_not_silently_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ 形状表**外**的变种只计数不删，但绝不能无声无息（对抗验证 F7）。

    ⚠️ 模块文档曾承诺这类变种会落进 ``draft_paragraph_left_*`` —— 实测
    不成立：``[things were cleaned up thoroughly]`` 配工具名词缺失时，
    ``_is_draft`` 与 ``_classify_suspicious_paragraphs`` 对它全为假，整轮
    被记成 ``clean``，模型复读标记这件事完全不可见。现在 ``_report`` 扫
    正文 + 思考链的残留候选，记 ``residual_internal_marker`` + WARNING。
    文本本身**不动**（删它没有把握）—— 只保证「看不见」变成「看得见」。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    text = "[things were cleaned up thoroughly] 好的。"
    assert _residual_marker_count(text) == 1, "残留计数器本身没认出这个词表外变种"

    events = [
        _reply_start(),
        _call_start(),
        *_text(text),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    assert _visible(out) == text, "残留候选被删了 —— 这层刻意只计数不删"
    assert "residual_internal_marker" in seen, seen
    assert "stripped_internal_marker" not in seen, (
        "形状表外的变种被记成「已剥除」—— 指标口径错位"
    )


# ==============================================================================
# 六、第二轮对抗验证（2026-10-05，wf_a41ee780-4fb）确认缺陷的回归用例
# ==============================================================================
#: 工具流事件用的调用 id（Start/Delta/End 必须一致才对得上缓冲键）。
_STREAM_CALL_ID = "call-stream-1"


def _tool_result_start(
    call_id: str = _STREAM_CALL_ID,
    name: str = "search_hotels",
) -> ToolResultStartEvent:
    """构造工具返回开始事件。"""
    return ToolResultStartEvent(
        reply_id=REPLY_ID,
        tool_call_id=call_id,
        tool_call_name=name,
    )


def _tool_result_delta(call_id: str, delta: str) -> ToolResultTextDeltaEvent:
    """构造工具返回文本分片事件。"""
    return ToolResultTextDeltaEvent(
        reply_id=REPLY_ID,
        tool_call_id=call_id,
        delta=delta,
    )


def _tool_result_end(call_id: str) -> ToolResultEndEvent:
    """构造工具返回结束事件。"""
    return ToolResultEndEvent(
        reply_id=REPLY_ID,
        tool_call_id=call_id,
        state=ToolResultState.SUCCESS,
    )


def _tool_call_delta(call_id: str, delta: str) -> ToolCallDeltaEvent:
    """构造工具调用参数分片事件。"""
    return ToolCallDeltaEvent(
        reply_id=REPLY_ID,
        tool_call_id=call_id,
        delta=delta,
    )


def _tool_call_end(call_id: str) -> ToolCallEndEvent:
    """构造工具调用结束事件。"""
    return ToolCallEndEvent(reply_id=REPLY_ID, tool_call_id=call_id)


def test_marker_stripping_is_not_superlinear_on_degenerate_input() -> None:
    """★★ P1：清洗成本不许随命中数**超线性**增长（退化输入的时间哨兵）。

    ⚠️ 修前实测（2026-10-05，本机）：每趟只删一处、删一处就全文重扫 ⇒

        · ``"<system-reminder>"`` × 640（10.6 KB，全是开标签、没有闭标签，
          ③ 号模式在每个起点都要把惰性 ``[\\s\\S]{0,4000}?`` 展开到上限）
          —— **21.3 秒**；×1024（17 KB）**67 秒**。这不是慢，是把事件
          循环整个堵死：守卫在 ``on_reply`` 里同步跑，期间整个 worker
          一个事件都发不出去。
        · ``"[tool_result pruned]"`` × 3200（62.5 KB）—— **6.9 秒**
          （每次缝合要做两趟跨全文的接缝正则）。

    修复后同一批输入分别 ~0.1s / ~0.15s / ~0.04s（本机实测）。
    时间阈值取 5s / 3s：距新实现 30 倍以上余量（CI 机器慢十倍也不误报），
    距旧实现 2–4 倍（真退化了就会红）。

    ⚠️ 这是**时间**断言，天然比行为断言脆 —— 所以只在这条路径上放，
    且阈值刻意开得很宽。它守的不是「有多快」，是「不许再退回
    O(命中数 × 全文长度)」。
    """
    open_tags = "<system-reminder>" * 640
    start = time.perf_counter()
    cleaned, removed = _strip_internal_markers(open_tags)
    elapsed = time.perf_counter() - start
    assert (cleaned, removed) == ("", 640)
    assert elapsed < 5.0, (
        f"640 个开标签用了 {elapsed:.2f}s —— 清洗退化回超线性了"
        "（旧实现 21s，批量不动点版 ~0.1s）"
    )

    many_markers = "[tool_result pruned]" * 3200
    start = time.perf_counter()
    cleaned_many, removed_many = _strip_internal_markers(many_markers)
    elapsed_many = time.perf_counter() - start
    assert (cleaned_many, removed_many) == ("", 3200)
    assert elapsed_many < 3.0, (
        f"3200 处标记用了 {elapsed_many:.2f}s —— 每趟仍在全量重扫"
        "（旧实现 6.9s，批量不动点版 ~0.04s）"
    )


def test_two_system_tags_do_not_swallow_the_text_between_them() -> None:
    """★★ ③ 的惰性量词被双标签用例钉住（tests-metrics 第 3 条）。

    ⚠️ ``<system-*>…</system-*>`` 的内容段用的是**惰性** ``[\\s\\S]{0,4000}?``
    —— 里侧那对标签之间的真答案靠它存活。单行变异（惰性 → 贪婪）此前
    245 条全绿：贪婪会让第一个开标签一路吃到**最后一个闭标签**，
    ``真实答案`` 被整段静默删除，而 ``removed`` 仍报 2。
    """
    text = (
        "<system-reminder>a</system-reminder>\n\n真实答案\n\n"
        "<system-reminder>b</system-reminder>"
    )
    cleaned, removed = _strip_internal_markers(text)
    assert (cleaned, removed) == ("真实答案", 2), (
        f"两个泄漏标签之间的真答案被连坐删了：{cleaned!r}（③ 变贪婪了？）"
    )


def test_standalone_marker_line_does_not_arm_the_tail_draft_strip() -> None:
    """★★ P2 端到端：独占一行的标记不许造出段落边界，把真答案喂给尾部剥离。

    ⚠️ 修前实测（workflow 原始复现）：回复 ``出行清单：\\n[tool_result
    pruned]\\nI need to bring my passport.`` 被服务端发出**并落库**为只剩
    ``出行清单：`` —— 接缝把标记那一行换成了空行 ⇒ ``_split_paragraphs``
    切出 ``I need to bring my passport.`` 这个新段落 ⇒ 它命中尾部草稿的
    强开头 ``^I (need|should|…)`` ⇒ ``_strip_drafts(at_tail=True)`` 静默
    吃掉。**没有标记的同样文本一字不动** —— 标记的存在改变用户拿到的
    内容是纯粹的净损失。修复后两者输出必须一致。
    """
    without_marker = "出行清单：\nI need to bring my passport."
    with_marker = "出行清单：\n[tool_result pruned]\nI need to bring my passport."

    out_clean = asyncio.run(
        _run(
            ReplyGuardMiddleware(),
            [
                _reply_start(),
                _call_start(),
                *_text(without_marker),
                _call_end(),
                _reply_end(),
            ],
            _FakeAgent(),
        )
    )
    out_marked = asyncio.run(
        _run(
            ReplyGuardMiddleware(),
            [
                _reply_start(),
                _call_start(),
                *_text(with_marker),
                _call_end(),
                _reply_end(),
            ],
            _FakeAgent(),
        )
    )

    assert _visible(out_clean) == without_marker
    assert _visible(out_marked) == without_marker, (
        f"带标记的回复被剃成了 {_visible(out_marked)!r} —— "
        "接缝又造出段落边界、尾巴草稿剥离吃掉了真答案"
    )
    assert "pruned" not in _visible(out_marked)


def test_punctuation_only_answer_is_rejected_not_shown() -> None:
    """★★ P3：剥完只剩标点的轮次判 ``empty``，不许把「。」端给用户。

    ⚠️ 修前实测：``[tool_result pruned]。`` 剥完剩一个句号，``_answer_problem``
    只查 ``strip()`` 非空 ⇒ 放行 —— 用户收到一个孤零零的「。」。
    判据改成「没有任何 ``\\w`` 字符（Unicode：CJK/字母/数字）即 empty」。
    """
    assert _answer_problem("。") == "empty"
    assert _answer_problem("…。!?、") == "empty"
    assert _answer_problem("好的。") is None, "正常中文答复被误判成 empty"

    seen: list[str] = []
    events = [
        _reply_start(),
        _call_start(),
        *_text("[tool_result pruned]。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent()
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert _visible(out) == "", f"标点孤儿上屏了：{_visible(out)!r}"
    assert not any(isinstance(e, ReplyEndEvent) for e in out), (
        "标点轮没有被吞掉 —— 框架不会重说，用户将面对空白"
    )
    assert agent.state.context_calls, "重说前必须往上下文里塞纠正指令"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 同一处方括号落进两条残留模式（清理动词 + ``by 机制名``）——
        # 修前逐模式相加报 2 处（tests-metrics 第 8 条）
        ("[old data compressed by context manager]", 1),
        ("[caches cleared by microcompaction]", 1),
        # 仅第二条模式命中（清理动词缺失）——第二模式被删时这里会红
        # （tests-metrics 第 10 条：此前被删完全无声）
        ("[caches evicted by microcompaction]", 1),
        # ⚠️ **只**命中第二条模式的形状 —— 2026-10-05 变异 M33 实测：上面
        # ``evicted`` 那条会被第一条模式的宽动词表（``evict(?:ed|ing|s)?``）
        # 兜住，删掉第二条模式它照样是 1 处，对「第二模式被删」完全不敏感。
        # 真正的隔离样本要同时满足两条：①「动词」不在残留动词表里
        # （``hidden``），② ``by`` 后**紧挨着**机制名（第二条模式没有
        # ``the`` 这类冠词的容身之处）。上面的 ``[cache was dropped by the
        # evictor]`` 也不行：``dropp?`` 在动词表里、``evictor`` 不在机制名里。
        ("[snippet hidden by microcompaction]", 1),
        # 两处互不相邻的候选 = 2 处（合并逻辑不许把相邻判定放宽）
        ("[a cleaned up] 和 [b was removed by the pruner]", 2),
    ],
)
def test_residual_count_counts_candidates_not_pattern_hits(
    text: str,
    expected: int,
) -> None:
    """★★ 残留计数按**候选处**计，不按模式命中次数计（tests-metrics 第 8 条）。

    ⚠️ WARNING 文案（以及这条指标）里的数字读作「仍有 N 处候选」；
    同一处方括号同时命中两条宽模式时逐模式相加会把 1 处报成 2 处，
    系统性放大 —— 修复是先把命中区间合并重叠再数不相交区间个数。
    """
    assert _residual_marker_count(text) == expected, text


@pytest.mark.parametrize(
    "text",
    [
        # ``… by <机制名>`` 形状（残留第二模式）——只钉「括号类」这件事
        "【tool_result cleared by postprune】好的。",
        "（tool result cleared by postprune）好的。",
        # ⚠️ 只命中**第一模式**的形状（清理动词、无 by）——不补这两条的话
        # 第二模式会把前两条兜住：把第一模式的括号类改回 ``[［``/``]］``
        # （去掉中文括号）时套件仍全绿，本测试对它不敏感（变异 M34 实测）。
        "【things were cleaned up】好的。",
        "（data was compressed）好的。",
    ],
)
def test_cjk_bracket_variants_are_counted_not_silently_clean(
    text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ P6：中文括号/全角圆括号刻意不剥，但**必须计数**（不许沉默放行）。

    ⚠️ 修前实测：``【tool_result cleared by postprune】`` 既不剥、也不进
    任何计数 —— 指标全为 ``clean``，模型复读标记这件事完全不可见。
    形状表保持刻意窄（差旅正文里中文括号太常见），残留扫描的括号类
    扩到 ``【】（）``：文本不动，但 ``residual_internal_marker`` 抬头。

    ⚠️ ``clean`` 与 ``residual_internal_marker`` 在这条路径上**可以同时
    出现**：前者说「守卫没改正文」，后者说「上屏文本里仍有候选」——
    两个问题，两个指标，刻意不做互斥（互斥的是
    ``clean`` 与 ``stripped_internal_marker``：都描述「改没改」）。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    assert _residual_marker_count(text) == 1, "残留计数器没认出中文括号变种"

    events = [
        _reply_start(),
        _call_start(),
        *_text(text),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    assert _visible(out) == text, "中文括号被剥了 —— 形状表刻意不含它们"
    assert "residual_internal_marker" in seen, seen
    assert "stripped_internal_marker" not in seen, seen


def test_marker_split_across_tool_result_deltas_is_cleaned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ P5：标记被流式分片拦腰截断时，工具通道不许把两半拼给用户。

    ⚠️ 修前实测（服务端 SSE + 落库都复现）：``'{"note": "[tool_result
    clea'`` + ``'red by postprune]"}'`` 两片**各自**干净，逐片就地洗放行，
    拼起来是完整标记 —— 用户屏幕上和刷新后的历史里都有它。
    修复是「留尾缓冲」：保留最近 :data:`_TOOL_STREAM_HOLDBACK` 字符，
    等配对 End 事件到达时整段结算。

    ⚠️ 断言用**拼起来**的文本（单看某一分片干净不算数 —— 那正是旧实现
    的放行依据），并且要求 JSON 结构完好（洗的是跨度，不是整条载荷）。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id=_STREAM_CALL_ID,
            tool_call_name="search_hotels",
        ),
        _tool_result_start(),
        _tool_result_delta(_STREAM_CALL_ID, '{"note": "[tool_result clea'),
        _tool_result_delta(_STREAM_CALL_ID, 'red by postprune]"}'),
        _tool_result_end(_STREAM_CALL_ID),
        *_text("我先查一下。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(tools={"search_hotels": _FakeTool(is_read_only=True)})
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    joined = "".join(
        e.delta for e in out if isinstance(e, ToolResultTextDeltaEvent)
    )
    assert joined == '{"note": ""}', f"分片标记没洗干净：{joined!r}"
    # ⚠️ 副断言原先查的是 ``"pruned"`` —— 它不是本用例标记的子串
    # （标记是 ``cleared by postprune``），**按构造永远为真**，等于没断言
    # （2026-10-05 第三轮对抗验证 D 类的处置）。查标记里真有的词。
    assert "postprune" not in "".join(
        getattr(e, "delta", "") for e in out
    ), "标记从某条通道漏出去了"
    assert "stripped_internal_marker" in seen, seen
    # ⚠️ 原先这里断言 ``"clean" not in seen``，但这个形状**不可能**记 clean
    # （本轮没有可上屏的正文 ⇒ 没有 clean 可记）—— 同样是空断言。
    # clean 的抑制由 test_tool_payload_strip_suppresses_clean_in_production_order
    # 与 test_marker_in_tool_payload_suppresses_the_clean_action 两条钉住。


def test_marker_split_across_tool_call_deltas_is_cleaned() -> None:
    """★★ P5 的另一条通道：``TOOL_CALL_DELTA``（调用参数）同样留尾清洗。

    ⚠️ 这条通道前端会实时显示（模型正在写的参数），修前同样被分片绕过。
    调用参数与返回文本各有一条独立缓冲（键是 ``(kind, call_id)``），
    只修一条 = 另一条仍是旁路。
    """
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id=_STREAM_CALL_ID,
            tool_call_name="search_hotels",
        ),
        _tool_call_delta(_STREAM_CALL_ID, '{"args": "[tool_result clea'),
        _tool_call_delta(_STREAM_CALL_ID, 'red by postprune]"}'),
        _tool_call_end(_STREAM_CALL_ID),
        _tool_result_start(),
        _tool_result_delta(_STREAM_CALL_ID, "共 3 家酒店。"),
        _tool_result_end(_STREAM_CALL_ID),
        *_text("我先查一下。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(tools={"search_hotels": _FakeTool(is_read_only=True)})

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    joined = "".join(e.delta for e in out if isinstance(e, ToolCallDeltaEvent))
    assert joined == '{"args": ""}', f"调用参数里的分片标记没洗干净：{joined!r}"
    assert "共 3 家酒店。" in "".join(
        e.delta for e in out if isinstance(e, ToolResultTextDeltaEvent)
    ), "另一条通道的返回文本被波及"


def test_tool_call_id_reuse_does_not_leak_stale_arguments() -> None:
    """★★ 同一 ``tool_call_id`` 被复用（模型重发同名调用）时清掉旧流缓冲。

    ⚠️ 留尾缓冲引入了「跨事件的累积状态」：第一次调用没走完（End 缺失）
    就重发同名调用时，不清缓冲的话新旧参数会拼成一段**谁都没生成过**的
    文本发给前端（``'{"a": 1'`` + ``'{"b": 2}'``）。框架侧的
    ``src/chains/events.py`` 对同一形状是同策（Start 时清分片），
    守卫必须与它一致 —— 这条钉住 ``_feed_tool_stream`` 之前的 pop。
    """
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id=_STREAM_CALL_ID,
            tool_call_name="search_hotels",
        ),
        _tool_call_delta(_STREAM_CALL_ID, '{"a": 1'),
        # ⚠️ 不结尾就重发同一个 id —— 第二次 Start 必须清掉旧缓冲
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id=_STREAM_CALL_ID,
            tool_call_name="search_hotels",
        ),
        _tool_call_delta(_STREAM_CALL_ID, '{"b": 2}'),
        _tool_call_end(_STREAM_CALL_ID),
        _tool_result_start(),
        _tool_result_delta(_STREAM_CALL_ID, "共 3 家酒店。"),
        _tool_result_end(_STREAM_CALL_ID),
        *_text("我先查一下。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(tools={"search_hotels": _FakeTool(is_read_only=True)})

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    joined = "".join(e.delta for e in out if isinstance(e, ToolCallDeltaEvent))
    assert joined == '{"b": 2}', (
        f"复用 id 时新旧参数被拼在一起：{joined!r}"
    )


def test_tool_stream_is_flushed_when_the_end_event_never_arrives() -> None:
    """★★ 工具流的安全网：配对 End 事件缺失时，缓冲不许无声蒸发。

    ⚠️ 留尾缓冲的代价是文本被扣在内存里 —— 若工具抛异常/上游截断
    （``ToolResultEndEvent`` 永远不来），没有安全网的话已扣的文本随回复
    一起消失，用户看到的执行链凭空少一段。``ModelCallEndEvent`` 与
    ``ReplyEndEvent`` 两处冲刷（:meth:`_flush_tool_streams`）。
    这条用例走 ``ModelCallEnd``（正常链路在此已为空，此刻有内容正是
    「End 缺失」的形状）。

    ⚠️ 冲刷时必须**继续洗**（final 结算走同一清洗），不是原样倒出。
    """
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id=_STREAM_CALL_ID,
            tool_call_name="search_hotels",
        ),
        _tool_result_start(),
        _tool_result_delta(_STREAM_CALL_ID, '{"note": "[tool_result clea'),
        _tool_result_delta(_STREAM_CALL_ID, 'red by postprune]"}'),
        # ⚠️ 刻意不发 _tool_result_end
        *_text("我先查一下。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(tools={"search_hotels": _FakeTool(is_read_only=True)})

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    joined = "".join(
        e.delta for e in out if isinstance(e, ToolResultTextDeltaEvent)
    )
    assert joined == '{"note": ""}', (
        f"End 缺失时缓冲文本要么没冲刷、要么没洗：{joined!r}"
    )


def test_marker_in_tool_payload_suppresses_the_clean_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ 工具通道洗过标记的轮次也不许记 ``clean``（互斥的第三块拼图）。

    ⚠️ 场景取「同一轮里既写复述、又调写工具」（``submit_approval`` 非只读
    ⇒ 文字按复述保留）：正文一字未改（本会走干净快路径记 ``clean``），
    而工具返回里剥掉了分片标记 ⇒ 整条回复改过 ⇒ ``clean`` 必须被抑制
    （``round_marker_stripped`` 标志，2026-10-05 第二轮对抗验证的修复）。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    text = "申请已提交：10 月 12 日出发，酒店 600 元/晚。"
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id=_STREAM_CALL_ID,
            tool_call_name="submit_approval",
        ),
        _tool_result_start(name="submit_approval"),
        _tool_result_delta(_STREAM_CALL_ID, "[tool_result clea"),
        _tool_result_delta(_STREAM_CALL_ID, "red by postprune]"),
        _tool_result_end(_STREAM_CALL_ID),
        *_text(text),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={"submit_approval": _FakeTool(is_read_only=False)},
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert _visible(out) == text, "复述文字被误剥"
    assert "stripped_internal_marker" in seen, seen
    assert "clean" not in seen, seen
    assert "kept_confirmation_round" in seen, seen


def test_end_override_payload_residual_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ P8：``TEXT_BLOCK_END`` 的覆盖载荷是**刷新后**用户看到的文本。

    ⚠️ 修前实测（对抗验证）：覆盖载荷体含形状表外的候选（如
    ``[things were cleaned up thoroughly]``）时，DELTA 拼接干净 ⇒
    走复用路径、指标全为 ``clean`` —— 而刷新后标记从历史消息里冒出来。
    修复后覆盖载荷（``event.text != 原始拼接``）进 ``_report`` 的残留
    扫描（第三条通道），记 ``residual_internal_marker``。

    ⚠️ 这里刻意只计数不删：形状表外的候选删了没有把握（可能是真内容）。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    dirty_payload = "好的。[things were cleaned up thoroughly]"
    events = [
        _reply_start(),
        _call_start(),
        TextBlockStartEvent(reply_id=REPLY_ID, block_id="block-1"),
        TextBlockDeltaEvent(reply_id=REPLY_ID, block_id="block-1", delta="好的。"),
        TextBlockEndEvent(reply_id=REPLY_ID, block_id="block-1", text=dirty_payload),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    assert _visible(out) == "好的。"
    assert "residual_internal_marker" in seen, (
        f"END 覆盖载荷里的候选没进残留计数：{seen}"
    )


def test_thinking_chain_residual_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ ``_report`` 的残留扫描覆盖**思考链**（tests-metrics 第 2 条）。

    ⚠️ 模块文档与 metrics.py 都声称残留计数扫正文 + 思考链，但唯一残留
    用例只喂正文块 —— 删掉 ``_report`` 里 ``state.emitted_thinking``
    那一项（或 ``_settle_thinking`` 里的累积）全套仍绿，思考链里的
    表外变种无声消失。思考链同样实时上屏且落库，不是计数盲区。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    thinking = "[things were cleaned up thoroughly] 先想想。"
    events = [
        _reply_start(),
        _call_start(),
        ThinkingBlockStartEvent(reply_id=REPLY_ID, block_id="t-1"),
        ThinkingBlockDeltaEvent(reply_id=REPLY_ID, block_id="t-1", delta=thinking),
        ThinkingBlockEndEvent(reply_id=REPLY_ID, block_id="t-1"),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    emitted = "".join(
        e.delta for e in out if isinstance(e, ThinkingBlockDeltaEvent)
    )
    assert emitted == thinking, "思考链被改了 —— 残留层刻意只计数不删"
    assert "residual_internal_marker" in seen, (
        f"思考链里的表外候选没进残留计数：{seen}"
    )


def test_tool_payload_residual_is_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ 工具载荷的表外候选也进残留计数（第四条通道）。

    ⚠️ ``tool_residual`` 在流式清洗时**逐段**累积（只数已发出的部分，
    尾巴留到下次）—— 工具载荷上屏/落库，不是盲区。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id=_STREAM_CALL_ID,
            tool_call_name="search_hotels",
        ),
        _tool_result_start(),
        _tool_result_delta(
            _STREAM_CALL_ID,
            "[things were cleaned up thoroughly]",
        ),
        _tool_result_end(_STREAM_CALL_ID),
        *_text("我先查一下。"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(tools={"search_hotels": _FakeTool(is_read_only=True)})

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    emitted = "".join(
        e.delta for e in out if isinstance(e, ToolResultTextDeltaEvent)
    )
    assert emitted == "[things were cleaned up thoroughly]", (
        f"工具载荷被改了 —— 残留层只计数：{emitted!r}"
    )
    assert "residual_internal_marker" in seen, (
        f"工具载荷里的表外候选没进残留计数：{seen}"
    )


def test_stripped_action_is_recorded_when_a_draft_paragraph_is_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ ``stripped`` 动作有测试钉住（tests-metrics 第 1 条）。

    ⚠️ 删掉 ``_settle_round`` 里那一行 ``observe_reply_guard("stripped")``
    此前 245 条全绿 —— 于是「剥掉了头部草稿段」这件事在指标里彻底消失，
    只剩日志。这条曲线是「闸门到底拦了什么」的主要产出，不能无覆盖。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    events = [
        _reply_start(),
        _call_start(),
        *_text("我先查一下差标。\n\n住宿标准：单晚不超过 600 元。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    assert _visible(out) == "住宿标准：单晚不超过 600 元。"
    assert "stripped" in seen, seen
    assert "clean" not in seen, (
        "剥过草稿的轮次同时记 clean —— 两个动作都描述「改没改」，必须互斥"
    )
    assert "stripped_internal_marker" not in seen, (
        "只剥了草稿、没洗标记 —— 记成标记清洗是指标口径错位"
    )


def test_end_payload_marker_is_reported_as_one_in_the_log(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★★ END 载荷分支的 ``marker_removed = 1`` 被日志断言钉住（tests-metrics 第 7 条）。

    ⚠️ 该分支单独记 WARNING 与指标，但 ``marker_removed`` 只喂给随后的
    INFO 汇总行；把它改成 0 此前全套仍绿 —— 于是同一件事在日志里自相
    矛盾：WARNING 说「拦下一处标记」，INFO 说「剥掉 0 处内部标记」。
    断言取 INFO 汇总行里的计数（「1 处内部标记」）。
    """
    caplog.set_level(logging.INFO, logger="src.orchestration.reply_guard")
    events = [
        _reply_start(),
        _call_start(),
        TextBlockStartEvent(reply_id=REPLY_ID, block_id="block-1"),
        TextBlockDeltaEvent(reply_id=REPLY_ID, block_id="block-1", delta="好的。"),
        TextBlockEndEvent(
            reply_id=REPLY_ID,
            block_id="block-1",
            text=f"好的。{_MEASURED_MARKER}",
        ),
        _call_end(),
        _reply_end(),
    ]
    asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    infos = [
        record.getMessage()
        for record in caplog.records
        if record.name == "src.orchestration.reply_guard"
        and record.levelno == logging.INFO
    ]
    joined = "\n".join(infos)
    assert "1 处内部标记" in joined, (
        f"END 载荷里拦下的标记没被计入 INFO 汇总：{infos}"
    )


# ==============================================================================
# 七、第三轮对抗验证（2026-10-05，wf_7e634f25-73e）确认缺陷的回归用例
# ==============================================================================
#: 守卫的 logger 名（`caplog` 按名字取记录）。
_GUARD_LOGGER = "src.orchestration.reply_guard"


def _guard_logs(caplog: pytest.LogCaptureFixture, level: int) -> list[str]:
    """取守卫在本次用例里写下的指定级别日志消息。

    Args:
        caplog (`pytest.LogCaptureFixture`): pytest 的日志夹具。
        level (`int`): 级别（``logging.WARNING`` / ``logging.INFO``）。

    Returns:
        `list[str]`: 消息文本（已格式化）。
    """
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == _GUARD_LOGGER and record.levelno == level
    ]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # ── 带块级前缀的独占标记行：整行连前缀一起摘 ──
        (
            "- 酒店 600 元\n- [tool_result pruned]\n- 机票 1200 元",
            "- 酒店 600 元\n- 机票 1200 元",
        ),
        ("1. 酒店\n2. [tool_result pruned]\n3. 机票", "1. 酒店\n3. 机票"),
        ("1) 酒店\n2) [tool_result pruned]\n3) 机票", "1) 酒店\n3) 机票"),
        ("> 提醒\n> [tool_result pruned]\n> 带身份证", "> 提醒\n> 带身份证"),
        ("## [tool_result pruned]\n正文", "正文"),
        ("### 标题\n## [tool_result pruned]\n正文", "### 标题\n正文"),
        ("  - [tool_result pruned]\n继续", "继续"),
        ("* [tool_result pruned]", ""),
        # ── 控制例：前缀那一行还有别的内容时，只摘标记本身 ──
        ("- [tool_result pruned] 备注", "- 备注"),
        ("前言 [tool_result pruned] 后语", "前言 后语"),
    ],
)
def test_markdown_block_prefix_line_is_removed_with_the_marker(
    source: str,
    expected: str,
) -> None:
    """★★ 标记独占一行、行首是 markdown 块级前缀时，整行（含前缀）都要摘掉。

    ⚠️ 修前实测（2026-10-05 第三轮对抗验证 #1，端到端）：接缝规则只吃标记
    两侧的**空白**，而 ``-`` / ``2.`` / ``>`` / ``##`` 都是非空白内容 ⇒
    用户看到**空列表项**（``'-\\n'-``）、**空引用行**、**空 H2**：结构性损坏
    而不是少一个空格。修复是 :func:`_MARKDOWN_LINE_PREFIX_RE` +
    :func:`_MARKDOWN_LINE_TAIL_RE`：标记独占一行且行首是纯块级前缀时，
    连前缀带换行整行摘除。

    ⚠️ 控制例同样重要：``- [标记] 备注`` 那一行**不是**「只有标记」，
    前缀必须留着（它是用户列表的编号），只摘标记跨度。
    """
    cleaned, removed = _strip_internal_markers(source)
    assert (cleaned, removed) == (expected, 1), (
        f"块级前缀那一行没被整行摘掉：{cleaned!r}（期望 {expected!r}）"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # #2：动词后**紧跟** CJK —— `\b` 词边界失效，整处漏网
        ("[tool_result pruned已从上下文移除] 好的。", "好的。"),
        ("[tool_result cleared已清理] 好的。", "好的。"),
        # #29：下划线分隔的标记族（`_` 不是 `\b` 边界）
        ("[tool_result_pruned] 好的。", "好的。"),
        ("[tool_result_cleared_by_postprune] 好的。", "好的。"),
    ],
)
def test_marker_with_cjk_or_underscore_boundary_is_stripped(
    source: str,
    expected: str,
) -> None:
    """★★ ``\\b`` 换成显式环视后，CJK 紧跟动词 / 下划线分隔也要认得（#2/#29）。

    ⚠️ 修前实测：``\\b`` 在「英文词 + 汉字」「下划线」两侧都不成立
    （两者都是 ``\\w``），``'[tool_result pruned已从上下文移除]'``
    与 ``'[tool_result_pruned]'`` **既不剥离也不计数** —— 用户看到标记、
    监控全 clean。修复改用 :data:`_VERB_LEFT_GUARD` / :data:`_VERB_RIGHT_GUARD`
    显式环视（两侧不是 ``[A-Za-z0-9]`` 即边界）。

    ⚠️ 环视的代价必须由控制例把住：``[Results: unpruned]`` 里的
    ``unpruned`` 是更长英文词的一部分，不许被当动词命中。
    """
    cleaned, removed = _strip_internal_markers(source)
    assert (cleaned, removed) == (expected, 1), (
        f"词边界放宽后这处标记仍漏网：{cleaned!r}（期望 {expected!r}）"
    )


def test_longer_english_word_containing_a_verb_is_not_a_marker() -> None:
    """★★ 控制例：``unpruned`` 不是 ``pruned`` —— 显式环视不许误伤更长的词。

    ⚠️ 这是 #2/#29 修复（``\\b`` → 显式环视）的**反向**钉子：环视一旦
    写成「前缀匹配」，正常英文词就会被整段删掉。``[Results: unpruned]``
    是真实语料里的正常内容。
    """
    text = "[Results: unpruned] 好的。"
    cleaned, removed = _strip_internal_markers(text)
    assert (cleaned, removed) == (text, 0), f"更长的英文词被误当标记：{cleaned!r}"
    assert _residual_marker_count(text) == 0, "正常英文词被误计为残留候选"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        # #4：CRLF 接缝 —— 标记两侧的 `\r\n` 要当成一条分隔吃掉
        ("好的。\r\n[tool_result pruned]\r\n下一段。", "好的。\r\n下一段。"),
        ("[tool_result pruned]\r\n好的。", "好的。"),
        ("好的。\r\n[tool_result pruned]", "好的。"),
    ],
)
def test_crlf_seam_leaves_no_blank_line_or_stray_cr(
    source: str,
    expected: str,
) -> None:
    """★★ ``_SEAM_WS`` 必须含 ``\\r``：CRLF 文本剥标记不许多空行 / 留孤立 ``\\r``。

    ⚠️ 修前实测（#4）：``'好的。\\r\\n[标记]\\r\\n下一段。'`` 剥完是
    ``'好的。\\r\\n\\r\\n下一段。'``（多一个空行）、``'[标记]\\r\\n好的。'``
    剩 ``'\\r\\n好的。'``（前导 CRLF 当内容留着）、行尾那处留下孤立 ``\\r``。
    Windows 客户端 / 复制粘贴的真实文本就是 CRLF，而这些都是**用户可见**
    的排版损坏。
    """
    cleaned, removed = _strip_internal_markers(source)
    assert (cleaned, removed) == (expected, 1), (
        f"CRLF 接缝没洗干净：{cleaned!r}（期望 {expected!r}）"
    )


def test_blank_lines_that_are_not_a_seam_are_left_alone() -> None:
    """★★ 控制例：没有标记时，CRLF 空行原样保留（`\\r` 进接缝空白不许改坏原文）。"""
    text = "好的。\r\n\r\n下一段。"
    cleaned, removed = _strip_internal_markers(text)
    assert (cleaned, removed) == (text, 0), f"干净文本被动了：{cleaned!r}"


def test_cleaned_up_tail_boundary_is_pinned() -> None:
    """★★ ① 号模式的**尾上限 80** 是行为边界，得有输入能红（#24）。

    ⚠️ 变异实测：把 ``[^\\[\\]［］\\n]{0,80}?`` 的上限降到 31/30，277 条
    全绿（注释声称会红）—— 上限被踩到 29 才有参数化输入变红。原因是
    套件里没有**贴着边界**的输入。这条用例把「≤80 剥、>80 不剥」钉死，
    上限往下挪一格就会红。

    ⚠️ 上限不是随便定的：放宽 = 更容易把真内容连坐删掉，收紧 = 真实
    标记（``[tool_result pruned 已清理 12 条记录，剩余 3 条]`` 这类带
    尾注的）漏网。边界值必须显式测。
    """
    at_limit = "[tool_result pruned " + "x" * 79 + "]"  # 动词后 80 字
    over_limit = "[tool_result pruned " + "x" * 80 + "]"  # 动词后 81 字

    cleaned, removed = _strip_internal_markers(at_limit)
    assert (cleaned, removed) == ("", 1), (
        f"尾长正好 80 的标记没被剥掉（上限被收紧了？）：{cleaned!r}"
    )

    cleaned_over, removed_over = _strip_internal_markers(over_limit)
    assert (cleaned_over, removed_over) == (over_limit, 0), (
        f"尾长 81 的文本被剥了（上限被放宽了？）：{cleaned_over!r}"
    )


def test_system_tag_attribute_boundary_is_pinned() -> None:
    """★★ ④ 号模式的**属性上限 400** 同样是行为边界（#24 的另一半）。

    ⚠️ 变异实测：属性量词 ``[^<>]{0,400}`` 降到 360，277 条全绿；降到 359
    才有参数化输入变红。这条用例把边界钉在 400/401 上。
    """
    at_limit = "<system-reminder " + "a" * 399 + ">"
    over_limit = "<system-reminder " + "a" * 400 + ">"

    cleaned, removed = _strip_internal_markers(at_limit)
    assert (cleaned, removed) == ("", 1), (
        f"属性长 400 的孤立标签没被剥掉（上限被收紧了？）：{cleaned!r}"
    )

    cleaned_over, removed_over = _strip_internal_markers(over_limit)
    assert (cleaned_over, removed_over) == (over_limit, 0), (
        f"属性长 401 的文本被剥了（上限被放宽了？）：{cleaned_over!r}"
    )


def test_adjacent_residual_candidates_count_as_two() -> None:
    """★★ 相邻候选是**两处**，不是一处（#22 的变异敏感性钉子）。

    ⚠️ 变异实测：把 :func:`_residual_marker_count` 的区间合并条件从
    ``start >= covered_until`` 放宽成 ``start > covered_until``（相邻区间
    并成一处），原有四条参数化输入输出全不变、测试照样通过 —— 它们的
    候选之间留了 3 个字符的间隔。这条用**真正贴边**的输入把语义钉死。
    """
    assert _residual_marker_count("[a cleaned up][b compressed]") == 2, (
        "贴边的两处候选被合并成了一处（合并条件放宽了？）"
    )
    assert _residual_marker_count("[a cleaned up] [b compressed]") == 2, (
        "有间隔的两处候选也被合并了"
    )


@pytest.mark.parametrize(
    "text",
    [
        # #19：裸清理动词（表外动词，形状表不剥 —— 但要计数）
        "好的。 [tool_result cleaned] 行。",
        "好的。 [tool_result purge] 行。",
        "好的。 [tool_result flushed] 行。",
        # #20：ASCII 圆括号
        "(tool_result cleared by postprune) 好的。",
    ],
)
def test_table_outsider_markers_are_counted_not_silently_clean(text: str) -> None:
    """★★ 形状表**外**的变种：剥不了不要紧，**不许指标全 clean**（#19/#20）。

    ⚠️ 修前实测：裸 ``clean/purge/flush/trim`` 与 ASCII 圆括号既不在剥除
    形状表、也不在残留计数表里 ⇒ ``removed == 0`` 且 ``residual == 0``，
    用户看到标记、监控什么也看不到。残留计数表已补上这两类（只计数
    不删：它们也可能是真内容，删了没有把握）。
    """
    cleaned, removed = _strip_internal_markers(text)
    assert removed == 0, f"表外变种被剥了（形状表放宽了？）：{cleaned!r}"
    assert _residual_marker_count(text) >= 1, (
        "表外变种既不剥也不计数 —— 指标会全 clean，这类泄漏完全不可见"
    )


# ------------------------------------------------------------------------------
# 思考链通道（#5/#7/#8/#9/#10）
# ------------------------------------------------------------------------------
def test_thinking_block_missing_end_is_completed_not_dropped() -> None:
    """★★ 干净思考块缺 ``END`` 也要补全 —— 否则落库块永远悬着（#5/#7）。

    ⚠️ 修前实测：剥过标记的路径一直合成完整三件套，干净路径却**原样
    透传**；上游少发 ``THINKING_BLOCK_END`` 时（handler 中途出错 /
    协议破坏），落库侧 ``finished_at`` 永远为空、块的收尾逻辑永远不跑
    —— 同一个缺陷只修了一半。修复是 :func:`_complete_thinking_blocks`。
    """
    events = [
        _reply_start(),
        _call_start(),
        ThinkingBlockStartEvent(reply_id=REPLY_ID, block_id="t-1"),
        ThinkingBlockDeltaEvent(reply_id=REPLY_ID, block_id="t-1", delta="没结尾。"),
        # ⚠️ 刻意不发 ThinkingBlockEndEvent
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    kinds = [type(e).__name__ for e in out]
    assert "ThinkingBlockEndEvent" in kinds, (
        f"缺 END 的干净思考块被原样透传（落库永远悬着）：{kinds}"
    )
    thinking = "".join(
        e.delta for e in out if isinstance(e, ThinkingBlockDeltaEvent)
    )
    assert thinking == "没结尾。", f"补全把内容改了：{thinking!r}"
    block_ids = {
        e.block_id for e in out if isinstance(e, ThinkingBlockEndEvent)
    }
    assert block_ids == {"t-1"}, f"补全换了块 id：{block_ids}"


def test_thinking_block_missing_start_is_completed_with_original_id() -> None:
    """★★ 干净思考块缺 ``START`` 要补上，且**保留原 id**（#5/#9）。

    ⚠️ 修前实测：缺 ``START`` 时落库侧整块**蒸发**（``Msg.append_event``
    靠 START 创建块，DELTA/END 找不到块就静默丢弃，服务端还打
    ``ThinkingBlock 't1' not found, skipping.``）—— 实时流有、历史里没有。
    补全必须用原 id：换 id 会让前端已经渲染的块对不上（#9）。
    """
    events = [
        _reply_start(),
        _call_start(),
        ThinkingBlockDeltaEvent(reply_id=REPLY_ID, block_id="t-1", delta="没有 START。"),
        ThinkingBlockEndEvent(reply_id=REPLY_ID, block_id="t-1", thinking="没有 START。"),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    kinds = [type(e).__name__ for e in out]
    assert kinds.count("ThinkingBlockStartEvent") == 1, (
        f"缺 START 的思考块没被补全（落库会整块蒸发）：{kinds}"
    )
    thinking = "".join(
        e.delta for e in out if isinstance(e, ThinkingBlockDeltaEvent)
    )
    assert thinking == "没有 START。", f"补全把内容改了：{thinking!r}"
    start = next(e for e in out if isinstance(e, ThinkingBlockStartEvent))
    assert start.block_id == "t-1", f"补全换了块 id：{start.block_id!r}"


def test_thinking_block_settles_at_its_own_end_before_text_events() -> None:
    """★★ 思考块在**自己的 END** 处结算，事件顺序不倒退（#8）。

    ⚠️ 变异实测：把块级结算改成「只在 ``ModelCallEnd`` 结算」，277 条
    全绿 —— 而事件顺序会退化成「工具调用/文本事件之后才出现思考链」，
    前端看到的是「先出答案、后出推理」。这条钉住顺序。
    """
    events = [
        _reply_start(),
        _call_start(),
        ThinkingBlockStartEvent(reply_id=REPLY_ID, block_id="t-1"),
        ThinkingBlockDeltaEvent(reply_id=REPLY_ID, block_id="t-1", delta="想。"),
        ThinkingBlockEndEvent(reply_id=REPLY_ID, block_id="t-1", thinking="想。"),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    kinds = [type(e).__name__ for e in out]
    assert kinds.index("ThinkingBlockEndEvent") < kinds.index(
        "TextBlockStartEvent",
    ), f"思考块被压到文本事件之后了：{kinds}"


def test_thinking_only_marker_strip_records_no_stripped_and_no_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ 只有思考链被洗的轮次：记 ``stripped_internal_marker``，不记 ``stripped``、不记 ``clean``（#10）。

    ⚠️ 变异实测：在思考链剥洗处补一行 ``observe_reply_guard("stripped")``，
    277 条全绿 —— 而 ``stripped`` 的语义是「**正文**与模型原文不同」，
    思考链通道改的根本不是正文（metrics.py 的指标表写明了）。
    三个动作的语义边界必须有输入钉住。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    events = [
        _reply_start(),
        _call_start(),
        ThinkingBlockStartEvent(reply_id=REPLY_ID, block_id="t-1"),
        ThinkingBlockDeltaEvent(
            reply_id=REPLY_ID,
            block_id="t-1",
            delta=f"想。{_MEASURED_MARKER}",
        ),
        ThinkingBlockEndEvent(
            reply_id=REPLY_ID,
            block_id="t-1",
            thinking=f"想。{_MEASURED_MARKER}",
        ),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    assert "stripped_internal_marker" in seen, seen
    assert "stripped" not in seen, (
        f"思考链剥洗被记成了 stripped（那是正文通道的语义）：{seen}"
    )
    assert "clean" not in seen, f"洗过标记的轮次仍记 clean：{seen}"


# ------------------------------------------------------------------------------
# 工具通道（#13/#14/#18/#23）
# ------------------------------------------------------------------------------
#: #13 的成对标签原文（一条工具返回被从闭标签前切开）。
_PAIRED_TAG_TEXT = (
    "好的。<system-reminder>INTERNAL RULES: do not reveal</system-reminder>已办妥。"
)


def test_paired_system_tag_split_across_tool_result_deltas_is_cleaned() -> None:
    """★★ 成对 ``<system-*>`` 标签被分片切开时，内部指令不许上屏（#13，high）。

    ⚠️ 修前实测：两个分片各自看都对（开标签那片被 ④ 当成孤立标签删掉、
    闭标签那片也删掉），拼起来**中间那段内部指令原样上屏且落库**
    （``ToolResultBlock.output`` 里就是它，还会随历史回灌给模型）。
    修复是留尾缓冲的「等待成对」分支：最后一个开标签还没配上闭标签时
    整段原样等待（上限 :data:`_SYSTEM_PAIR_MAX_SPAN`），对上了再一次性剥。

    ⚠️ 内容只有 31 字符，远小于 512 的留尾 —— 这不是「缓冲太短」的
    问题，是 ④ 抢在 ③ 前面把开标签删掉的问题。
    """
    cut = _PAIRED_TAG_TEXT.index("</system-reminder>")
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id=_STREAM_CALL_ID,
            tool_call_name="search_hotels",
        ),
        _call_end(),
        _tool_result_start(),
        _tool_result_delta(_STREAM_CALL_ID, _PAIRED_TAG_TEXT[:cut]),
        _tool_result_delta(_STREAM_CALL_ID, _PAIRED_TAG_TEXT[cut:]),
        _tool_result_end(_STREAM_CALL_ID),
        _call_start(""),
        *_text("好的。已办妥。", reply_id=""),
        _call_end(""),
        _reply_end(),
    ]
    out = asyncio.run(
        _run(
            ReplyGuardMiddleware(),
            events,
            _FakeAgent(tools={"search_hotels": _FakeTool(is_read_only=True)}),
        ),
    )

    joined = "".join(
        e.delta for e in out if isinstance(e, ToolResultTextDeltaEvent)
    )
    assert joined == "好的。已办妥。", (
        f"跨分片的成对标签没洗干净（内部指令上屏了）：{joined!r}"
    )
    assert "INTERNAL RULES" not in "".join(
        getattr(e, "delta", "") for e in out
    ), "内部指令从某条通道漏出去了"


def test_paired_system_tag_split_across_tool_call_deltas_is_cleaned() -> None:
    """★★ #13 的另一条通道：``TOOL_CALL_DELTA``（调用参数）同样等待成对。

    ⚠️ 两条通道各有独立缓冲（键 ``(kind, call_id)``），只修一条 = 另一条
    仍是旁路（与 P5 的拆分标记同一形状）。
    """
    cut = _PAIRED_TAG_TEXT.index("</system-reminder>")
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id=_STREAM_CALL_ID,
            tool_call_name="search_hotels",
        ),
        _tool_call_delta(_STREAM_CALL_ID, _PAIRED_TAG_TEXT[:cut]),
        _tool_call_delta(_STREAM_CALL_ID, _PAIRED_TAG_TEXT[cut:]),
        _tool_call_end(_STREAM_CALL_ID),
        _call_end(),
        _call_start(""),
        *_text("好的。已办妥。", reply_id=""),
        _call_end(""),
        _reply_end(),
    ]
    out = asyncio.run(
        _run(
            ReplyGuardMiddleware(),
            events,
            _FakeAgent(tools={"search_hotels": _FakeTool(is_read_only=True)}),
        ),
    )

    joined = "".join(e.delta for e in out if isinstance(e, ToolCallDeltaEvent))
    assert joined == "好的。已办妥。", (
        f"调用参数里跨分片的成对标签没洗干净：{joined!r}"
    )


def test_residual_candidate_riding_the_holdback_cut_is_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★★ 骑在 512 字留尾切口上的残留候选必须计到数（#14/#18）。

    ⚠️ 修前实测：候选的**起点**落在发出的那段、结尾还在留尾缓冲里时，
    旧实现只扫 ``emit``（看不到结尾）、下一次只扫留尾（看不到开头）——
    两边都不成形状 ⇒ ``tool_residual == 0``、零告警，而标记照常上屏。
    修复按**起点归属**计数：扫描窗口取整段缓冲，起点已发出的当场记账。

    ⚠️ 断言取 WARNING 里的「工具载荷 N 处」：这是**只计一次**的证明 ——
    若下一次结算又数一遍，就会变成 2 处。
    """
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id=_STREAM_CALL_ID,
            tool_call_name="search_hotels",
        ),
        _call_end(),
        _tool_result_start(),
        _tool_result_delta(
            _STREAM_CALL_ID,
            "Z" * 170 + "[things were cleaned up thoroughly]" + "Y" * 494,
        ),
        _tool_result_end(_STREAM_CALL_ID),
        _call_start(""),
        *_text("好的。", reply_id=""),
        _call_end(""),
        _reply_end(),
    ]
    asyncio.run(
        _run(
            ReplyGuardMiddleware(),
            events,
            _FakeAgent(tools={"search_hotels": _FakeTool(is_read_only=True)}),
        ),
    )

    warnings = _guard_logs(caplog, logging.WARNING)
    joined = "\n".join(warnings)
    assert "工具载荷 1 处" in joined, (
        f"骑在留尾切口上的候选没被计数（或计重了）：{warnings}"
    )


def test_oversized_system_pair_in_tool_payload_is_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★★ 超长成对 system 标签：标签剥掉、内容留下 —— 至少要**计数**（#23，工具通道）。

    ⚠️ 修前实测：内容 4200 字（超 ③ 的 4000 上限）时 ③ 不命中、④ 只删两个
    标签，4200 字内部注入内容原样上屏；而那段内容没有标签形状，残留扫描
    （扫的是发出文本）永远认不出来 ⇒ ``tool_residual == 0``、零告警。
    修复在**清洗前**的缓冲上按起点归属计数（
    :func:`_oversize_system_pair_count_in_prefix`）。

    ⚠️ 与「等待超时」那条分支的区别：这里的标签是**完整到达**的
    （一条 delta 就收全），根本走不到等待分支 —— 旧实现连那次记账都没有。
    """
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    oversized = "<system-reminder>" + "内部注入内容。" * 700 + "</system-reminder>"
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id=_STREAM_CALL_ID,
            tool_call_name="search_hotels",
        ),
        _call_end(),
        _tool_result_start(),
        _tool_result_delta(_STREAM_CALL_ID, oversized),
        _tool_result_end(_STREAM_CALL_ID),
        _call_start(""),
        *_text("好的。", reply_id=""),
        _call_end(""),
        _reply_end(),
    ]
    asyncio.run(
        _run(
            ReplyGuardMiddleware(),
            events,
            _FakeAgent(tools={"search_hotels": _FakeTool(is_read_only=True)}),
        ),
    )

    warnings = _log_all(caplog)
    assert "工具载荷 1 处" in warnings, (
        f"超长成对标签的内容上屏了却没计数（监控全绿）：{warnings}"
    )


def test_oversized_system_pair_in_thinking_is_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★★ 超长成对 system 标签在**思考链**里同样要计数（#23 的同形状）。

    ⚠️ 思考链同样实时上屏、同样落库 —— 剥不干净时内容跟着思考链出去。
    正文轮有 :func:`_oversize_system_pair_count` 兜底，思考链此前没有。
    """
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    oversized = "<system-reminder>" + "内部注入内容。" * 700 + "</system-reminder>"
    events = [
        _reply_start(),
        _call_start(),
        ThinkingBlockStartEvent(reply_id=REPLY_ID, block_id="t-1"),
        ThinkingBlockDeltaEvent(reply_id=REPLY_ID, block_id="t-1", delta=oversized),
        ThinkingBlockEndEvent(reply_id=REPLY_ID, block_id="t-1", thinking=oversized),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    warnings = _log_all(caplog)
    assert "剥离前预记 1 处" in warnings, (
        f"思考链里的超长成对标签内容上屏了却没计数：{warnings}"
    )


def _log_all(caplog: pytest.LogCaptureFixture) -> str:
    """把守卫本次写下的所有日志拼成一段文本（断言用）。"""
    return "\n".join(
        record.getMessage()
        for record in caplog.records
        if record.name == _GUARD_LOGGER
    )


def test_tool_stream_is_flushed_when_the_model_call_end_never_arrives() -> None:
    """★★ 工具流的安全网在 ``ReplyEnd`` 那处也要有（D1：两处调用点各钉一次）。

    ⚠️ 变异实测：单独删掉 ``ModelCallEndEvent`` 或 ``ReplyEndEvent`` 处的
    ``_flush_tool_streams`` 调用，277 条全绿 —— 既有的那条用例走的是
    ``ModelCallEnd``，钉不到 ``ReplyEnd``。而 ``ReplyEnd`` 那处正是
    「模型侧直接抛异常/被取消、连 ModelCallEnd 都没有」时的唯一机会。
    """
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id=_STREAM_CALL_ID,
            tool_call_name="search_hotels",
        ),
        _tool_result_start(),
        _tool_result_delta(_STREAM_CALL_ID, "共 3 家酒店。"),
        # ⚠️ 刻意不发 ToolResultEndEvent，也**不发 ModelCallEndEvent**
        _reply_end(),
    ]
    agent = _FakeAgent(tools={"search_hotels": _FakeTool(is_read_only=True)})

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    joined = "".join(
        e.delta for e in out if isinstance(e, ToolResultTextDeltaEvent)
    )
    assert joined == "共 3 家酒店。", (
        f"ReplyEnd 处的冲刷没生效，缓冲文本随回复一起蒸发了：{joined!r}"
    )


def test_tool_payload_strip_suppresses_clean_in_production_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ 生产事件顺序下，工具载荷剥过标记的轮次不许记 ``clean``（#15）。

    ⚠️ 修前实测：既有用例脚本的是「工具结果在 ``ModelCallEnd`` **之前**」
    的**不可达**顺序（框架不可能那样发），所以「剥了标记的轮次仍记 clean」
    在真实顺序下从来没被拦住 —— 实测生产顺序拿到
    ``['kept_confirmation_round', 'clean', 'stripped_internal_marker', ...]``，
    同一个回复既「一字未改」又「剥掉了标记」，指标自相矛盾。
    修复是 ``pending_clean``：``clean`` 的记账推迟到下一个轮次边界，
    那时工具载荷剥没剥过标记已经知道了。

    ⚠️ 这条用例**刻意让回复在工具结果之后直接结束**（没有第二轮模型调用）：
    被剥的那一轮就是最后一轮，``clean`` 若出现必然是它自己的。后续轮次
    真的干净时记 ``clean`` 是正确行为，由
    :func:`test_clean_is_recorded_for_a_later_clean_round_after_a_stripped_one`
    单独钉住 —— 两条合起来才完整，只写一条会把「按轮复位」误判成 bug。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    text = "申请已提交，10 月 12 日出发。"
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id=_STREAM_CALL_ID,
            tool_call_name="submit_approval",
        ),
        *_text(text),
        # ⚠️ 生产顺序：ModelCallEnd **先到**，工具结果事件在它之后
        _call_end(),
        _tool_result_start(name="submit_approval"),
        _tool_result_delta(_STREAM_CALL_ID, "[tool_result clea"),
        _tool_result_delta(_STREAM_CALL_ID, "red by postprune]"),
        _tool_result_end(_STREAM_CALL_ID),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={"submit_approval": _FakeTool(is_read_only=False)},
    )

    out = asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    assert _visible(out) == text, "复述文字被误剥"
    assert "stripped_internal_marker" in seen, seen
    assert "clean" not in seen, (
        f"生产顺序下工具通道剥了标记、这一轮仍被记成 clean：{seen}"
    )


def test_clean_is_recorded_for_a_later_clean_round_after_a_stripped_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ ``round_marker_stripped`` 是**按轮**复位的：后续干净轮该记 ``clean``（#25）。

    ⚠️ 变异实测：删掉 ``begin_round`` 里的 ``self.round_marker_stripped = False``
    那一行，277 条全绿 —— 而行为会变成「第一轮剥过标记，整条回复此后再也
    不记 clean」（跨轮误抑制，指标偏低、告警偏高）。
    反过来，上面 #15 那条要求同一轮里不许记 clean —— 两条一起才把
    「按轮」这个粒度钉死。

    ⚠️ 断言用**完整动作序列**（含顺序）：只断言集合的话，把 ``clean``
    记在剥洗**之前**也照样绿，而那意味着「先记干净、后剥标记」——
    同一个回复的两个动作互相打架。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    events = [
        _reply_start(),
        _call_start(),
        *_text(f"好的。{_MEASURED_MARKER}"),
        _call_end(),
        _call_start(""),
        *_text("下一段。", reply_id=""),
        _call_end(""),
        _reply_end(),
    ]
    asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    assert seen == ["stripped_internal_marker", "stripped", "clean"], (
        f"跨轮复位的动作序列不对（误抑制或多记）：{seen}"
    )


def test_multi_block_end_payloads_are_not_treated_as_overrides(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★★ 多文本块轮次：每块的 END 载荷与本块 DELTA 比，不许整轮重复计数（#16/#27）。

    ⚠️ 修前实测：拿**整轮** DELTA 拼接当基准 ⇒ 每块的载荷都不等于整轮
    文本（真载荷也全被当成「覆盖」），残留候选被重复计数：用户可见 1 处、
    WARNING 报 2 处且「END 覆盖载荷 45 字」虚高。修法是把载荷按 **block_id
    逐块**与**本块**的 DELTA 拼接比 —— 比对基准必须是它自己的块。
    """
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    events = [
        _reply_start(),
        _call_start(),
        TextBlockStartEvent(reply_id=REPLY_ID, block_id="b-1"),
        TextBlockDeltaEvent(
            reply_id=REPLY_ID,
            block_id="b-1",
            delta="[things were cleaned up thoroughly] 好的。",
        ),
        TextBlockEndEvent(
            reply_id=REPLY_ID,
            block_id="b-1",
            text="[things were cleaned up thoroughly] 好的。",
        ),
        TextBlockStartEvent(reply_id=REPLY_ID, block_id="b-2"),
        TextBlockDeltaEvent(reply_id=REPLY_ID, block_id="b-2", delta="第二块。"),
        TextBlockEndEvent(reply_id=REPLY_ID, block_id="b-2", text="第二块。"),
        _call_end(),
        _reply_end(),
    ]
    asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    warnings = "\n".join(_guard_logs(caplog, logging.WARNING))
    assert "仍有 1 处形状表外的标记候选" in warnings, (
        f"多块轮次的残留候选被重复计数（或丢计）：{warnings}"
    )
    assert "END 覆盖载荷 0 字" in warnings, (
        f"非覆盖的真载荷被当成了覆盖载荷：{warnings}"
    )


def test_end_override_superset_of_delta_is_not_double_counted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★★ 载荷是 DELTA 的超集（真覆盖）时，同一处候选只算一次（#21）。

    ⚠️ 修前实测：``_report`` 拿「整轮 DELTA 拼接」与「END 覆盖载荷」两串
    分头扫 ⇒ 载荷 ⊇ 正文时同一处候选被算两遍：用户可见 1 处、WARNING 报
    「2 处（正文 38 字、END 覆盖载荷 44 字）」。修法是**逐块取生效文本**
    （覆盖块取载荷、其余块取拼接），同一段文本只出现一次。
    """
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    events = [
        _reply_start(),
        _call_start(),
        TextBlockStartEvent(reply_id=REPLY_ID, block_id="b-1"),
        TextBlockDeltaEvent(
            reply_id=REPLY_ID,
            block_id="b-1",
            delta="好的 [things were cleaned up thoroughly]",
        ),
        TextBlockEndEvent(
            reply_id=REPLY_ID,
            block_id="b-1",
            text="好的 [things were cleaned up thoroughly] 再加一点。",
        ),
        _call_end(),
        _reply_end(),
    ]
    asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    warnings = "\n".join(_guard_logs(caplog, logging.WARNING))
    assert "仍有 1 处形状表外的标记候选" in warnings, (
        f"覆盖载荷与正文里的同一处候选被算了两遍：{warnings}"
    )


def test_cross_block_join_does_not_fabricate_a_residual_candidate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★★ 多块轮次的残留扫描**逐块**做，不许跨块拼出伪候选（#11 的 P4）。

    ⚠️ 修前实测：把整轮正文拼成一串扫 ⇒ 块 1 结尾 ``[Luggage clea`` +
    块 2 开头 ``red by customs]`` 拼出一个**两块里都不存在**的「标记候选」，
    残留告警虚报（告警是要人去看的，虚报会把信噪比拉垮）。
    逐块扫两头都躲开：真候选（在某一块内部成形）不漏，跨块的拼接不报。
    """
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    events = [
        _reply_start(),
        _call_start(),
        TextBlockStartEvent(reply_id=REPLY_ID, block_id="b-1"),
        TextBlockDeltaEvent(
            reply_id=REPLY_ID,
            block_id="b-1",
            delta="参考价格 [Luggage clea",
        ),
        TextBlockEndEvent(
            reply_id=REPLY_ID,
            block_id="b-1",
            text="参考价格 [Luggage clea",
        ),
        TextBlockStartEvent(reply_id=REPLY_ID, block_id="b-2"),
        TextBlockDeltaEvent(
            reply_id=REPLY_ID,
            block_id="b-2",
            delta="red by customs]",
        ),
        TextBlockEndEvent(
            reply_id=REPLY_ID,
            block_id="b-2",
            text="red by customs]",
        ),
        _call_end(),
        _reply_end(),
    ]
    asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    warnings = "\n".join(_guard_logs(caplog, logging.WARNING))
    assert "标记候选" not in warnings, (
        f"跨块拼接造出了伪候选（残留告警虚报）：{warnings}"
    )


def test_end_payload_marker_count_matches_the_markers_actually_stripped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★★ END 载荷里的标记处数按**实际**剥掉的数报，不许硬编码 1（#12）。

    ⚠️ 修前实测：两块的 END 载荷各带 1 处标记时，WARNING 说「拦下**一**处」
    （硬编码）、INFO 说「剥掉 …**1** 处内部标记」，而实际拦下 2 处 ——
    运维按这条日志估「模型复读标记」的发生率会系统性偏低。
    """
    caplog.set_level(logging.INFO, logger=_GUARD_LOGGER)
    events = [
        _reply_start(),
        _call_start(),
        TextBlockStartEvent(reply_id=REPLY_ID, block_id="b-1"),
        TextBlockDeltaEvent(reply_id=REPLY_ID, block_id="b-1", delta="一。"),
        TextBlockEndEvent(
            reply_id=REPLY_ID,
            block_id="b-1",
            text=f"一。{_MEASURED_MARKER}",
        ),
        TextBlockStartEvent(reply_id=REPLY_ID, block_id="b-2"),
        TextBlockDeltaEvent(reply_id=REPLY_ID, block_id="b-2", delta="二。"),
        TextBlockEndEvent(
            reply_id=REPLY_ID,
            block_id="b-2",
            text=f"二。{_MEASURED_MARKER}",
        ),
        _call_end(),
        _reply_end(),
    ]
    asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    infos = "\n".join(_guard_logs(caplog, logging.INFO))
    assert "2 处内部标记" in infos, (
        f"多处 END 载荷标记被少报（硬编码 1？）：{infos}"
    )


def test_empty_round_keeps_the_unresolved_rejection_reason() -> None:
    """★★ 空轮**刻意不**清 ``pending_problem`` / ``ungrounded_limits``（#17 的处置）。

    ⚠️ 第三轮对抗验证 #17 报的是「空轮不清状态，重试提示引用已翻篇的旧轮
    原因」。处置**不是**清状态，而是让措辞在空轮夹在中间时仍然为真
    （``你上一个候选答复`` 而不是 ``你刚才那段答复``）—— 理由写在
    :meth:`_settle_round` 的注释里：被拒的原因这时**仍未解决**（模型这轮
    什么都没给），清掉只会让提示退化成泛用版、更容易白费一轮。

    ⚠️ 这条用例钉两件事：① 原因确实**留**着（提示里点名那个数）；
    ② 措辞是「上一个候选答复」（空轮形状下依然诚实）。
    只钉 ① 的话，措辞改回「刚才那段」也照样绿 —— 而那时这句话就是假的。
    """
    events = [
        _reply_start(),
        _call_start(),
        # 写工具轮 ⇒ 这段文字按「给用户看的复述」保留 ⇒ 接地闸门跑
        *_text("请确认：酒店差标 500 元/晚，金额 1200 元。"),
        _tool_call("check_travel_policy"),
        _tool_call("submit_approval"),
        _call_end(),
        # 空轮：模型什么都没产出，reply 在这里被重试吞掉
        _call_start(),
        _call_end(),
        _reply_end(),
        # 第二次重试机会：模型这轮给出可用答复
        _call_start(),
        *_text("查到了：酒店差标 600 元/晚。"),
        _tool_call("check_travel_policy"),
        _tool_call("submit_approval"),
        _call_end(),
        _reply_end(),
    ]
    agent = _FakeAgent(
        tools={
            "check_travel_policy": _FakeTool(is_read_only=True),
            "submit_approval": _FakeTool(is_read_only=False),
        },
    )

    asyncio.run(_run(ReplyGuardMiddleware(), events, agent))

    hints = [
        block.text
        for _name, blocks in agent.state.context_calls
        for block in blocks
        for hint in (block.hint or [])
        for block in [hint]
        if getattr(block, "text", "")
    ]
    assert hints, f"一次纠正指令都没塞进去：{agent.state.context_calls!r}"
    assert any("上一个候选答复" in text for text in hints), (
        f"空轮之后提示仍在说「刚才那段答复」（那句话已经不真了）：{hints}"
    )
    assert any("500" in text for text in hints), (
        f"空轮把未解决的原因清掉了，提示不再点名那个数：{hints}"
    )


def test_nested_marker_cascade_is_not_superlinear() -> None:
    """★★ 嵌套/级联形状同样不许退回超线性（#0/#26 的回归钉子）。

    ⚠️ 第三轮对抗验证实测（修前）：``"[tool_result pruned " * d + "]" * d``
    这种**嵌套级联**里，每删一处都会在接缝上露出的新命中原样重扫全文 ——
    d=2000（42 KB）**10.6 秒**、d=800（16.8 KB）**1.5 秒**。既有性能哨兵
    只覆盖**平铺**形状（``"[tool_result pruned]" * 3200``），守不住这条。

    修复后同一批输入：d=2000 约 0.13s、d=4000 约 0.35s（本机实测，
    近线性）。阈值取 3s：距新实现 20 倍以上余量，距旧实现 3 倍以上。
    """
    depth = 2000
    text = "[tool_result pruned " * depth + "]" * depth
    start = time.perf_counter()
    cleaned, removed = _strip_internal_markers(text)
    elapsed = time.perf_counter() - start
    assert (cleaned, removed) == ("", depth), (
        f"嵌套级联没被剥干净：removed={removed}（期望 {depth}）"
    )
    assert elapsed < 3.0, (
        f"嵌套级联 {len(text)} 字用了 {elapsed:.2f}s —— 清洗退化回超线性了"
        "（旧实现 10.6s，批量不动点版 ~0.13s）"
    )


# ==============================================================================
# 八、第三轮对抗验证的两个完整性缺口（2026-10-05，wf_7e634f25-73e 的 critics）
# ==============================================================================
#: 缺口 ① 的实测原文（思考链尾 + 下一段头，拼起来正是线上标记）。
_GAP1_HEAD = _MEASURED_MARKER[: len(_MEASURED_MARKER) // 2]
_GAP1_TAIL = _MEASURED_MARKER[len(_MEASURED_MARKER) // 2 :]


def _thinking_block(block_id: str, text: str) -> list[EventBase]:
    """构造一个思考块的 START → DELTA → END 事件。"""
    return [
        ThinkingBlockStartEvent(reply_id=REPLY_ID, block_id=block_id),
        ThinkingBlockDeltaEvent(reply_id=REPLY_ID, block_id=block_id, delta=text),
        ThinkingBlockEndEvent(reply_id=REPLY_ID, block_id=block_id, thinking=text),
    ]


def _thinking_visible(events: list[EventBase]) -> str:
    """返回用户会看到的思考链文本（拼接 DELTA）。"""
    return "".join(
        event.delta
        for event in events
        if isinstance(event, ThinkingBlockDeltaEvent)
    )


def test_marker_split_across_two_thinking_blocks_is_cleaned() -> None:
    """★★ 缺口 ①：标记被**思考块边界**拦腰截断时也不许上屏（thinking→thinking）。

    ⚠️ 修前实测（第三轮对抗验证的完整性 critic）：思考块按
    ``THINKING_BLOCK_END`` **逐块**结算，标记的前半截拿不到后半截 ——
    两半各自通过清洗，而实时流与落库都是拼着的，用户看到的就是完整标记。
    工具通道有 512 字留尾缓冲挡这个形状，思考链当时没有对应机制。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_thinking_block("t-1", f"先想一下。{_GAP1_HEAD}"),
        *_thinking_block("t-2", f"{_GAP1_TAIL} 继续。"),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    thinking = _thinking_visible(out)
    assert "postprune" not in thinking and "tool_result" not in thinking, (
        f"标记跨思考块边界缝合后原样上屏：{thinking!r}"
    )
    assert "继续。" in thinking, f"清洗把真推理也删了：{thinking!r}"


def test_marker_split_across_thinking_and_text_is_cleaned() -> None:
    """★★ 缺口 ①（更重的变体）：思考链尾 + 正文字块头拼出完整标记。

    ⚠️ 修前实测：这种切法连残留计数都没有（思考链只看自己那块、正文只
    看自己这轮），指标只有 ``clean``、零 WARNING —— 落库后
    ``ThinkingBlock`` + ``TextBlock`` 拼起来就是完整标记。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_thinking_block("t-1", f"先想一下。{_GAP1_HEAD}"),
        *_text(f"{_GAP1_TAIL} 好的。已办妥。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    visible = _visible(out)
    thinking = _thinking_visible(out)
    assert "postprune" not in visible and "tool_result" not in visible, (
        f"标记跨「思考块→正文块」边界缝合后原样上屏：{visible!r}"
    )
    assert "postprune" not in thinking and "tool_result" not in thinking, (
        f"标记前半截留在了思考链里：{thinking!r}"
    )
    assert visible == "好的。已办妥。", f"正文被改坏了：{visible!r}"


def test_uncompleted_carry_is_kept_as_content_not_dropped(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★★ 扣来的半截前缀没等到下半截时：**不删**，原样发出 + WARNING（Round-4）。

    ⚠️ 旧实现在这里把前缀删掉（判据「洗过之后它还在原处」），Round-4 的
    refuter 实测：删的往往不只是碎片 —— ``[tool_result clea`` 后面接着
    真内容时，``startswith`` 成立、整段被切掉、指标还记 ``clean``。修补后
    的取舍是「宁漏不删」：文本原样上屏（碎片由残留扫描的第三模式计数 +
    一条 WARNING），只有回复结束时的**确信族**碎片才丢弃并记账。
    """
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    events = [
        _reply_start(),
        _call_start(),
        *_thinking_block("t-1", f"先想一下。{_GAP1_HEAD}"),
        *_thinking_block("t-2", "换个话题。"),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    thinking = _thinking_visible(out)
    assert thinking == f"先想一下。{_GAP1_HEAD}换个话题。", (
        f"未完成的前缀被删了或者搬错了位置（内容不丢才是新契约）：{thinking!r}"
    )
    assert any(
        "没有等到下半截" in record.getMessage()
        for record in caplog.records
    ), f"未完成的前缀消失了却没留 WARNING：{[r.getMessage() for r in caplog.records]}"


def test_partial_marker_prefix_at_reply_end_is_dropped_and_counted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★★ 扣在手里的半截前缀等不到下半截时：丢弃，但要**看得见**（缺口 ① 收尾）。

    ⚠️ 反向约束：发出去 = 用户屏幕上出现 ``[tool_result clea``；静默丢弃 =
    内容无声消失。选丢弃（扣着的判据只认标记族开头，丢的不是用户内容），
    代价是必须计数 —— 残留总数 + WARNING，两样都钉在这里。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    events = [
        _reply_start(),
        _call_start(),
        *_thinking_block("t-1", f"先想一下。{_GAP1_HEAD}"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    visible = _visible(out) + _thinking_visible(out)
    assert "tool_result" not in visible and "clea" not in visible, (
        f"没写完的半截标记前缀上屏了：{visible!r}"
    )
    assert "residual_internal_marker" in seen, (
        f"半截前缀被静默丢弃 —— 残留没有计数：{seen}"
    )
    assert any(
        "没写完的内部标记前缀" in record.getMessage()
        for record in caplog.records
    ), f"丢弃半截前缀没留 WARNING：{[r.getMessage() for r in caplog.records]}"


@pytest.mark.parametrize(
    "tail",
    [
        "还没算完] 继续。",  # 方括号族，但不是标记族开头
        "继续。",  # 根本没有方括号
    ],
)
def test_non_marker_tail_at_a_block_seam_is_not_moved(tail: str) -> None:
    """★★ 反向约束：块边界上**不是**标记开头的内容，一个字都不许挪（缺口 ① 的护栏）。

    ⚠️ 缝合机制动的是**要发给用户的文本**，所以判据只认标记族自己的开头
    （``[tool…`` / ``[old tool…`` / ``[inline…`` / ``<system…``）。认任意
    方括号的话，正文里 ``[Luggage clea`` + ``red by customs]`` 这种真内容
    会被扣走、在下一块里冒出来 —— 那比漏一个标记更糟。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_thinking_block("t-1", "先想一下。[预算"),
        *_thinking_block("t-2", tail),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    assert _thinking_visible(out) == f"先想一下。[预算{tail}", (
        f"非标记内容被跨块搬家了：{_thinking_visible(out)!r}"
    )


def test_tool_call_name_with_a_marker_is_scrubbed(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★★ 缺口 ②：``ToolCallStartEvent.tool_call_name`` 里的标记必须洗掉。

    ⚠️ 修前实测（第三轮对抗验证的完整性 critic）：这个名字模型可写、
    前端当执行链标题、``Msg.append_event`` 又把它写进 ``ToolCallBlock.name``
    落库 —— 而守卫**根本没看**这个字段（残留计数为 0、零 WARNING）。
    实测名字为 ``[tool_result pruned]`` 时原样透传并落库。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id="c1",
            tool_call_name=_MEASURED_MARKER,
        ),
        _tool_call_delta("c1", '{"city": "杭州"}'),
        _tool_call_end("c1"),
        _text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    names = [
        event.tool_call_name
        for event in out
        if isinstance(event, ToolCallStartEvent)
    ]
    assert names == [""], (
        f"工具名里的标记没洗干净（前端标题与落库的 name 都用它）：{names}"
    )
    assert "stripped_internal_marker" in seen, (
        f"洗了工具名却没记账 —— 曲线看不出模型在复读标记：{seen}"
    )
    assert any(
        "工具名" in record.getMessage() for record in caplog.records
    ), f"洗工具名没留 WARNING：{[r.getMessage() for r in caplog.records]}"


def test_tool_result_name_with_a_marker_is_scrubbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ 缺口 ② 的另一半：``ToolResultStartEvent.tool_call_name`` 同样要洗。

    ⚠️ 上游对同一处标记写**两个**字段（``ToolCallBlock.name`` 与
    ``ToolResultBlock.name``，见 ``agentscope/message/_base.py`` 的两个
    分支），只洗其中一个等于没洗 —— 结果块那个照样上屏、照样落库。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    events = [
        _reply_start(),
        _call_start(),
        _tool_result_start("c1", name=_MEASURED_MARKER),
        _tool_result_delta("c1", "结果。"),
        _tool_result_end("c1"),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    names = [
        event.tool_call_name
        for event in out
        if isinstance(event, ToolResultStartEvent)
    ]
    assert names == [""], f"工具结果块的名字没洗：{names}"
    assert "stripped_internal_marker" in seen, seen


def test_clean_tool_name_is_passed_through_unchanged() -> None:
    """★ 反向约束：干净的工具名必须**原对象**透传（缺口 ② 的护栏）。

    ⚠️ 守卫对工具名做的是「洗标记」而不是「改名字」：合法工具名里不可能
    出现标记形状，所以命中即异常。这条钉住「没命中就一个字都不动」——
    连事件对象都不该换（换了会让下游的身份判断失效）。
    """
    start = ToolCallStartEvent(
        reply_id=REPLY_ID,
        tool_call_id="c1",
        tool_call_name="check_travel_policy",
    )
    events = [
        _reply_start(),
        _call_start(),
        start,
        _tool_call_delta("c1", "{}"),
        _tool_call_end("c1"),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    passed = [event for event in out if isinstance(event, ToolCallStartEvent)]
    assert len(passed) == 1 and passed[0] is start, (
        "干净的工具调用事件被换了对象 —— 判据不该动没命中的事件"
    )


# ==============================================================================
# 九、第四轮对抗验证的修复（2026-10-05，wf_0202c1d5-434：14 survivors + 6 critics）
# ==============================================================================
#: 跨块切分用的 system 标签原文（③ 的形状：开/闭标签 + 内容）。
_SYSTEM_PAIR_SOURCE = "<system-reminder>SECRET</system-reminder>"
#: 孤立闭标签（④ 剥它、留内容）—— 同样会被块边界截断。
_LONE_CLOSE_TAG = "</system-reminder>"
#: 嵌套方括号尾（前一个方括号头未写完、后面又开一个）：宽判据只看得到
#: 第二个 ``[``，要靠**清洗之后**的二次扣尾才能整段扣住。
_NESTED_TAIL = "[tool_result pruned[toolA"


@pytest.mark.parametrize(
    "cut",
    range(len(_SYSTEM_PAIR_SOURCE) + 1),
    ids=lambda n: f"cut@{n}",
)
@pytest.mark.parametrize("second_channel", ["thinking", "text"])
def test_system_tag_split_across_blocks_is_cleaned(
    cut: int,
    second_channel: str,
) -> None:
    """★★ G1：``<system-*>`` 标签跨块时，切点落在**任何**位置都不许上屏。

    ⚠️ Round-4 的 refuter 实测（修前）：旧实现只在**清洗后**的文本上找
    system 开标签 —— 而孤立开标签早被 ④ 删掉了，这条分支是**死代码**。
    跨块的 ``<system-*>SECRET</system-*>` 因此两半各自干净、拼起来完整上屏。
    切在标签名内部（``<system-rem``）时连 ④ 都认不出，两半拼起来把开标签
    **重新拼出来**。修补：判据回到**原文**上求 + 新增「未写完的标签头」臂。

    Args:
        cut (`int`): 切点（第一段 = 切点前的字，第二段 = 其余）。
        second_channel (`str`): 第二段走思考块还是正文轮（两条结算路径）。
    """
    head = _SYSTEM_PAIR_SOURCE[:cut]
    tail = _SYSTEM_PAIR_SOURCE[cut:]
    if second_channel == "thinking":
        second: list[EventBase] = _thinking_block("t-2", f"{tail} B")
    else:
        second = _text(f"{tail} B")
    events = [
        _reply_start(),
        _call_start(),
        *_thinking_block("t-1", f"A{head}"),
        *second,
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    visible = _visible(out) + _thinking_visible(out)
    assert "SECRET" not in visible and "system" not in visible, (
        f"切在第 {cut} 字（{second_channel}）时 system 标签上屏了：{visible!r}"
    )
    assert "A" in visible and "B" in visible, (
        f"标签两侧的真内容被误删了：{visible!r}"
    )


@pytest.mark.parametrize("cut", range(len(_LONE_CLOSE_TAG) + 1), ids=lambda n: f"cut@{n}")
def test_split_lone_close_tag_is_cleaned(cut: int) -> None:
    """★★ G1 的另一半：**孤立闭标签**被块边界截断时也不许上屏。

    ⚠️ 自查穷举探针发现（修前）：扣留判据只认**开**标签头，``'A|</'``
    + ``'system-reminder>…'`` 这种切法两块拼出完整的 ``</system-reminder>``
    —— ④ 本会剥掉孤立闭标签，但它看到的每一块里标签都是碎的。
    """
    head = _LONE_CLOSE_TAG[:cut]
    tail = _LONE_CLOSE_TAG[cut:]
    events = [
        _reply_start(),
        _call_start(),
        *_thinking_block("t-1", f"A|{head}"),
        *_thinking_block("t-2", f"{tail}SECRET|B"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    visible = _thinking_visible(out)
    assert "system-reminder" not in visible, (
        f"切在第 {cut} 字时闭标签拼出来了：{visible!r}"
    )
    assert "SECRET" in visible and visible.startswith("A|"), (
        f"闭标签的内容/上下文被牵连：{visible!r}"
    )


@pytest.mark.parametrize("cut", [0, 1, 2, 3, 4, 5, 12, 19])
def test_marker_split_at_a_short_prefix_is_cleaned(cut: int) -> None:
    """★★ G2：切点落在标记头 1–5 个字（``[`` / ``[t`` / ``[to`` …）时也要干净。

    ⚠️ Round-4 的 seam 镜片实测（修前）：旧扣留判据要求 ``tool`` / ``inline``
    **字面量完整**（``[\\[［【]\\s*(?:old|previous)?(?:tool|inline)``），切在
    ``[t`` / ``[to`` 上时半截头不被扣留 → 两块拼起来完整标记上屏。修补：
    扣留改用**宽判据**（任意未闭合的方括号尾，体为 ASCII 标记字符），
    「删/清」类决定仍只用窄判据 —— 扣留零损失，宽一点没有代价。
    """
    marker = "[tool_result pruned]"
    events = [
        _reply_start(),
        _call_start(),
        *_thinking_block("t-1", f"想一下。{marker[:cut]}"),
        *_thinking_block("t-2", f"{marker[cut:]} 继续。"),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    visible = _thinking_visible(out) + _visible(out)
    assert "pruned" not in visible and "tool_result" not in visible, (
        f"切在第 {cut} 字时完整标记拼出来了：{visible!r}"
    )
    assert "继续。" in visible, f"标记之外的内容被牵连：{visible!r}"


def test_marker_longer_than_the_old_holdback_window_is_cleaned() -> None:
    """★★ CG1：标记体超过旧扣留窗口（80 字）时同样不许上屏。

    ⚠️ Round-4 的 completeness critic 实测（修前）：扣留窗口 ``{0,80}``
    比形状表可剥的最长标记（② 句式 ≈190 字）**更窄** ——
    ``[tool_result cleared by postprune: `` + 55 个 ``x`` + ``]``（91 字，
    清洗器**整处剥得掉**）的头 40 字扣不住，两块拼起来完整标记上屏。
    修补：窗口放到 200（覆盖形状表可剥的最大长度）。
    """
    marker = "[tool_result cleared by postprune: " + "x" * 55 + "]"
    assert len(marker) > 80, "样本必须超过旧窗口，否则测不到这条"
    events = [
        _reply_start(),
        _call_start(),
        *_thinking_block("t-1", f"想一下。{marker[:40]}"),
        *_thinking_block("t-2", f"{marker[40:]} 继续。"),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    visible = _thinking_visible(out) + _visible(out)
    assert "cleared by" not in visible and "xxx" not in visible, (
        f"超长标记跨块后上屏了：{visible!r}"
    )
    assert "继续。" in visible, visible


def test_marker_split_between_thinking_and_tool_payload_is_cleaned() -> None:
    """★★ G5：标记横跨「思考链 → 工具载荷」的缝时，两条通道要缝在一起洗。

    ⚠️ Round-4 的 frontier 镜片实测（修前）：思考块扣下的 ``[tool_result clea``
    与工具调用参数里的 ``red by postprune]}`` 分属两条通道，各自留尾、
    谁也看不到完整标记 —— 拼起来完整标记上屏。修补：工具流入口检测
    「标记起点在扣留里、终点在本条载荷里」时把两半拼起来一起洗。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_thinking_block("t-1", "先查一下。[tool_result clea"),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id="c1",
            tool_call_name="check_travel_policy",
        ),
        _tool_call_delta("c1", "red by postprune]}"),
        _tool_call_delta("c1", "尾部内容"),
        _tool_call_end("c1"),
        _text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    deltas = "".join(
        event.delta
        for event in out
        if isinstance(event, ToolCallDeltaEvent)
    )
    assert "尾部内容" in deltas, f"标记后面的载荷被牵连：{deltas!r}"
    # ⚠️ 断言必须查**所有通道的并集**（2026-10-05 由变异测试 M46 击穿后订正）：
    # 标记横跨思考链与工具载荷两个通道时，逐通道各查一个子串会漏 —— 把缝合
    # 关掉后，前半截进思考链（被回复结束丢弃）、`red by postprune]}` 留在
    # 载荷里，而原来那两条断言各自都成立。查并集才问得出「整处标记有没有
    # 上屏」这个问题。
    visible = _thinking_visible(out) + deltas + _visible(out)
    for half in ("tool_result", "red by", "postprune"):
        assert half not in visible, f"跨通道的标记漏出来了：{visible!r}"


@pytest.mark.parametrize(
    "fragment",
    [
        "[tool_result clea",  # 窄判据也认（[tool…）
        "[Luggage clea",  # 只有**宽**判据认（任意方括号头）—— 名字通道专用
    ],
)
def test_tool_name_with_a_partial_marker_is_cleared(
    monkeypatch: pytest.MonkeyPatch,
    fragment: str,
) -> None:
    """★★ G4：工具名是**半截**标记头时，整名清空 + 记账。

    ⚠️ Round-4 的 toolname 镜片实测（修前）：``_scrub_tool_call_name`` 只认
    **完整**标记（``_has_internal_marker`` 预筛），名字被流式截成
    ``[tool_result clea`` 时判据一个都不命中 → 原样透传 + 落库。名字本该
    是标识符，任何「像标记开头」的形状都不可能是合法工具名（宽判据）。

    ⚠️ 参数里的第二种碎片（``[Luggage clea``）是**只有宽判据认**的形状：
    它同时钉住「名字通道用宽判据、正文通道用窄判据」这条分工 —— 如果名字
    通道退回窄判据（或预筛去掉宽判据），这条就红。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    events = [
        _reply_start(),
        _call_start(),
        ToolCallStartEvent(
            reply_id=REPLY_ID,
            tool_call_id="c1",
            tool_call_name=fragment,
        ),
        _tool_call_delta("c1", "{}"),
        _tool_call_end("c1"),
        _text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    names = [
        event.tool_call_name
        for event in out
        if isinstance(event, ToolCallStartEvent)
    ]
    assert names == [""], f"半截标记工具名没清掉：{names}"
    assert "stripped_internal_marker" in seen, seen


def test_hint_block_marker_is_scrubbed_and_counted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ CG2：``HintBlockEvent.hint`` 是第五条上屏且落库的通道。

    ⚠️ Round-4 的 completeness critic 实测（修前）：``hint``（``str`` 或
    ``TextBlock`` 列表）既不在清洗通道里、也不在残留扫描里 ——
    ``Msg.append_event`` 的 ``HINT_BLOCK`` 分支把它原样落库。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    events = [
        _reply_start(),
        _call_start(),
        HintBlockEvent(
            reply_id=REPLY_ID,
            block_id="h-1",
            hint=f"提示：{_MEASURED_MARKER} 结束",
        ),
        HintBlockEvent(
            reply_id=REPLY_ID,
            block_id="h-2",
            hint=[TextBlock(type="text", text=f"列表{_MEASURED_MARKER}")],
        ),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    hints = [
        event.hint for event in out if isinstance(event, HintBlockEvent)
    ]
    assert len(hints) == 2, hints
    assert hints[0] == "提示： 结束", f"字符串 hint 没洗：{hints[0]!r}"
    assert hints[1][0].text == "列表", f"块列表 hint 没洗：{hints[1]!r}"
    assert "stripped_internal_marker" in seen, seen


def test_data_block_name_and_media_type_are_scrubbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ G6：``DataBlockStartEvent`` 的 ``name`` / ``media_type`` 同类字段要洗。

    ⚠️ Round-4 的 toolname 镜片：这两个字段与 ``tool_call_name`` 一样会被
    ``Msg.append_event``（``DATA_BLOCK_START`` 分支）落库，此前不在任何
    清洗通道里。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    events = [
        _reply_start(),
        _call_start(),
        DataBlockStartEvent(
            reply_id=REPLY_ID,
            block_id="d-1",
            name="[tool_result pruned",
            media_type="[inline cleaned",
        ),
        HintBlockEvent(reply_id=REPLY_ID, block_id="h-1", hint="查完了。"),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    data_starts = [
        event for event in out if isinstance(event, DataBlockStartEvent)
    ]
    assert len(data_starts) == 1, data_starts
    assert data_starts[0].name == "" and data_starts[0].media_type == "", (
        f"data 块的 name/media_type 没清："
        f"{data_starts[0].name!r} / {data_starts[0].media_type!r}"
    )
    assert "stripped_internal_marker" in seen, seen


@pytest.mark.parametrize(
    "tail",
    [
        "[tool_result clea",  # 残留扫描认得（第三模式）—— 只能数一笔
        "[inline clea",  # 扫描认不出（不属于任何残留模式）—— 由显式那一笔补
    ],
)
def test_tool_stream_tail_with_a_confident_head_is_counted(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    tail: str,
) -> None:
    """★★ 工具载荷结尾是**确信**的半截标记时：不改载荷，但必须计数（且只数一笔）。

    ⚠️ 工具载荷属于「执行链可见性」，不能像正文那样扣尾顺延（扣了会改掉
    工具返回值）。半截标记的下半截只可能从别的通道来，而这条流已经收尾
    —— 所以这里的取舍是「原样发出 + 计数 + WARNING」，让运维看得见。

    ⚠️ 反面的坑（2026-10-05 自查）：显式计数与起点归属扫描**同时**命中
    同一处尾碎片时（``[tool_result clea`` 就在残留模式里），日志从
    「工具载荷 1 处」变成「2 处」。所以两条参数各钉一侧：扫描认得的碎片
    由扫描数、显式那笔必须让位；扫描认不出的（``[inline clea``）才由
    显式那笔补上 —— 两种形状最终都恰好 1 处。
    """
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    events = [
        _reply_start(),
        _call_start(),
        _tool_result_start("c1", name="check_travel_policy"),
        _tool_result_delta("c1", f"查到了 {tail}"),
        _tool_result_end("c1"),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    deltas = "".join(
        event.delta
        for event in out
        if isinstance(event, ToolResultTextDeltaEvent)
    )
    assert "查到了" in deltas, f"载荷被改动（本就不该动）：{deltas!r}"
    assert "residual_internal_marker" in seen, (
        f"工具载荷结尾的半截标记没计数：{seen}"
    )
    counted = [
        record.getMessage()
        for record in caplog.records
        if "工具载荷" in record.getMessage() and "标记候选" in record.getMessage()
    ]
    assert counted and "工具载荷 1 处" in counted[-1], (
        f"同一处尾碎片被数了两遍或没数：{counted}"
    )


def test_exception_path_counts_the_dropped_carry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ CG3：没有 ``ReplyEndEvent`` 的异常路径上，扣着的前缀也要记账。

    ⚠️ Round-4 的 completeness critic 实测（修前）：``ReplyEndEvent`` 分支
    是唯一调用「定性扣留前缀」的地方 —— 异常/取消路径上（模型/上游抛错）
    扣着的内容既没上屏、也没记账，**静默消失**。修补：``finally`` 里补一笔。
    """
    seen: list[str] = []
    monkeypatch.setattr(
        "src.orchestration.reply_guard.observe_reply_guard",
        seen.append,
    )
    events = [
        _reply_start(),
        _call_start(),
        *_thinking_block("t-1", "想一下。[tool_result clea"),
        _call_end(),
    ]

    async def handler(**_kwargs: Any) -> AsyncGenerator[EventBase, None]:
        """先把事件发完，再抛（模拟模型侧异常）。"""
        for event in events:
            yield event
        raise RuntimeError("模型侧异常")

    async def drive() -> list[EventBase]:
        out: list[EventBase] = []
        async for event in ReplyGuardMiddleware().on_reply(
            agent=_FakeAgent(),
            input_kwargs={"inputs": None},
            next_handler=handler,
        ):
            out.append(event)
        return out

    with pytest.raises(RuntimeError):
        asyncio.run(drive())

    assert "residual_internal_marker" in seen, (
        f"异常路径上扣着的前缀静默消失了：{seen}"
    )


def test_full_width_bracket_fragment_at_reply_end_is_emitted_as_content() -> None:
    """★★ CG4：全角 ``【`` 的碎片在回复结束时**原样发出**，不许删。

    ⚠️ Round-4 的 completeness critic 实测（修前）：``【`` 被扣留判据接受、
    却不被清洗器接受（①②③④ 都刻意不剥 ``【…】``）—— 回复结束时旧实现
    按「是标记族开头」把它删掉，屏幕上少一段、指标还记 ``clean``。
    修补：扣留判据宽、删除判据窄（``【`` 不在窄判据里），碎片按内容发出。
    """
    events = [
        _reply_start(),
        _call_start(),
        *_thinking_block("t-1", "先想一下。【tool_result clea"),
        _call_end(),
        _reply_end(),
    ]
    out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    visible = _thinking_visible(out) + _visible(out)
    assert "【tool_result clea" in visible, (
        f"全角括号碎片被删了（内容丢失）：{visible!r}"
    )


def test_nested_bracket_tail_is_held_whole_until_the_settlement() -> None:
    """★★ Round-4 自查：清洗**之后**要在新结尾上**再扣一次**尾。

    ⚠️ 宽判据的正则要求 `[` 到行尾之间**没有别的方括号**（体类里不含
    ``[``/``]``）。于是 ``先想一下。[tool_result pruned[toolA`` 这种「前一个
    方括号头还没写完、后面又开了一个」的形状，清洗前的那一次扣尾只会从
    **第二个** ``[`` 起扣 —— 前半截 ``[tool_result pruned`` 留在``body`` 里
    当正文发出去（它是半截标记头，屏幕上不该出现）；而且剩下的 ``[toolA``
    会被判成确信族、回复结束时**丢弃**，等于一处分片被劈成两半各错一次。

    修补：清洗（可能删掉中间那处完整标记）之后再在新结尾上求一次扣尾，
    两段扣留按原文顺序拼接。断言两条：

        · ``thinking`` 里**没有**半截头（它被整段扣住了，没当成正文发）；
        · 回复结束时整段碎片按**宽族**原样发出（不丢内容）。
    """
    for channel in ("thinking", "text"):
        events: list[EventBase] = [_reply_start(), _call_start()]
        if channel == "thinking":
            events += _thinking_block("t-1", f"先想一下。{_NESTED_TAIL}")
        else:
            events += _text(f"先想一下。{_NESTED_TAIL}")
        events += [_call_end(), _reply_end()]
        out = asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

        thinking = _thinking_visible(out)
        visible = _visible(out)
        if channel == "thinking":
            assert thinking == "先想一下。", (
                f"半截标记头当正文发出去了（二次扣尾没生效）：{thinking!r}"
            )
        else:
            assert visible == f"先想一下。{_NESTED_TAIL}", (
                f"正文通道的二次扣尾没生效：{visible!r}"
            )
        assert _NESTED_TAIL in (thinking + visible), (
            f"整段扣留的碎片被删了（内容丢失）：{thinking!r} + {visible!r}"
        )


def test_survived_carry_is_counted_exactly_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """★★ Round-4 自查：扣来的半截前缀**只记一笔**，不许与残留扫描双计。

    ⚠️ 试过的错：在 ``_confident_carry_survived`` 分支里补
    ``residual_precount += 1``。但碎片是**随本段上屏**的 —— 回复结束时的
    残留扫描（第三模式的半截行锚）已经数到它了，再补一笔就是同一个碎片
    数两遍（实测日志从「仍有 1 处」变成「仍有 2 处」）。所以那一笔被删掉，
    只留 WARNING；本用例把「只记一笔」钉死在日志文案上。
    """
    caplog.set_level(logging.WARNING, logger=_GUARD_LOGGER)
    events = [
        _reply_start(),
        _call_start(),
        *_thinking_block("t-1", "想一下。[tool"),
        *_thinking_block("t-2", "结果"),
        *_text("好的。"),
        _call_end(),
        _reply_end(),
    ]
    asyncio.run(_run(ReplyGuardMiddleware(), events, _FakeAgent()))

    messages = [record.getMessage() for record in caplog.records]
    assert any("没有等到下半截" in message for message in messages), messages
    counted = [message for message in messages if "仍有" in message and "标记候选" in message]
    assert len(counted) == 1, f"残留上报条数不对：{messages}"
    assert "仍有 1 处" in counted[0] and "剥离前预记 0 处" in counted[0], (
        f"同一个碎片被数了两遍（预记与扫描双计）：{counted[0]}"
    )
