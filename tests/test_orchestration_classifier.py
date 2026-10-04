# -*- coding: utf-8 -*-
"""快慢车道规则引擎（``src/orchestration/classifier.py``）的测试。

本文件的重心不是「快车道能命中几条短语」——那种用例写起来很热闹，
却挡不住真正的失败模式。真正的失败模式有三类，本文件按这三类组织：

1. **死短语**：写进规则表的短语，其实永远匹配不上（因为没归一化）。
   症状是「规则明明写了却不生效」，排查方向会被带偏到匹配逻辑上。
   → :func:`test_every_rule_phrase_survives_normalization`
2. **误命中**：把用户的**询问**当成**命令**，于是系统去执行用户只是在打听的事。
   快车道的错误比慢车道贵得多（慢车道最多多花一次模型调用）。
   → :func:`test_questions_never_take_the_fast_lane`
3. **规则冲突**：同一条短语落在两条规则里，路由结果取决于表的遍历顺序 ——
   而这不会报错，只会让行为变得「有时对有时错」。
   → :func:`test_no_phrase_is_claimed_by_two_rules`
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src.domain import AgentName, Intent, LaneName, TripStage
from src.orchestration.classifier import (
    FAST_LANE_RULES,
    _INTENT_TARGET_AGENTS,
    _is_question,
    _normalize,
    classify,
    route_for_intent,
    target_agents_for,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


# ==============================================================================
# 一、不变式：规则表本身是自洽的
# ==============================================================================
def test_every_rule_phrase_survives_normalization() -> None:
    """★★ 每条规则短语在归一化后必须**保持原样**。

    这是本文件最重要的一条，它守的是「死短语」。

    机制：``classify`` 比较的是 ``_normalize(用户输入)`` 与规则短语。
    若某条短语本身不是归一化形态（例如写成「请帮我规划行程吧」），那么它
    要求归一化结果**恰好等于**一个带客套词的字符串 —— 而归一化的职责正是
    把这些词删掉。于是这条短语永远不可能相等，一次也命中不了。

    ⚠️ 它的危害不是「少了一条规则」，而是**它看起来像覆盖率**：代码评审时
    看到「请帮我规划行程吧」被收录了，会认为该系统支持这种说法；实际上
    用户说「请帮我规划行程吧」时归一化成了「规划行程」，若表里**同时**有
    「规划行程」还好，若只有带客套词那条，就直接漏到慢车道去了。

    ⚠️ 这条断言是我在写规则表时被自己的用例抓出来的 —— 第一版里
    「我的订单」「我要取消」「为我规划行程」三条全是死的。所以它不是
    假想的风险，是已经发生过一次的真实错误。
    """
    for rule in FAST_LANE_RULES:
        for phrase in rule.phrases:
            assert _normalize(phrase) == phrase, (
                f"规则 {rule.name!r} 的短语 {phrase!r} 不是归一化形态 —— "
                f"归一化后变成 {_normalize(phrase)!r}，该短语永远匹配不上。"
                f"请把它改写成归一化后的形态。"
            )


def test_every_rule_phrase_actually_fires() -> None:
    """★★ 每条规则短语都**真的**能命中它自己那条规则。

    ⚠️ 这条与 ``test_every_rule_phrase_survives_normalization`` 是**两件事**，
    缺一不可。那条只保证短语在归一化后保持原样 —— 但归一化只是判定链上的
    **第一关**。它后面还有问句判定、长度判定等多道闸门，任何一道先拦下来，
    短语就是死的，而归一化检查**完全看不见**这件事。

    真实案例（本项目已发生过一次）：``query_policy`` 规则里曾经有一条
    ``"能报多少"``。它归一化后是稳定的，于是顺利通过了归一化检查；
    但它含疑问词「多少」，被问句判定拦下，永远走慢车道 —— 一次也没命中过。
    规则表看起来覆盖了「能报多少」这个说法，实际上没有。

    ⚠️ 为什么这类错误特别危险：规则表是**唯一**的快车道调优面，读代码的人
    会把它当成系统能力的清单。一条死短语不只是「少一条规则」，而是
    **对系统能力的虚假陈述** —— 评审时看到它，会认为该说法已被支持。

    ⚠️ 遍历**全部**短语而不是抽一两条：死短语的特征就是「看起来跟旁边那些
    一模一样」，抽查抽不到它。
    """
    for rule in FAST_LANE_RULES:
        for phrase in rule.phrases:
            decision = classify(phrase)
            assert decision.lane == LaneName.FAST, (
                f"规则 {rule.name!r} 的短语 {phrase!r} 走的是慢车道"
                f"（原因：{decision.reason}）—— 这是一条死短语，"
                f"它永远不会命中。请从规则表里删掉它，"
                f"或改写成一个不是问句、且足够短的形态。"
            )
            assert decision.matched_rule == rule.name, (
                f"短语 {phrase!r} 命中的是 {decision.matched_rule!r} 而不是它所在的规则 {rule.name!r}"
            )


def test_no_phrase_is_claimed_by_two_rules() -> None:
    """同一条短语不得被两条规则认领。

    机制：``classify`` 命中第一条匹配的规则就返回。若两条规则共用一个短语，
    实际生效的永远是 :data:`FAST_LANE_RULES` 里**靠前**的那条 —— 结果是
    「路由由表的顺序决定」，而调整顺序（比如按字母重排以便阅读）会静默改变
    行为。这类问题不会报错，只会让线上表现「有时对有时错」。
    """
    owners: dict[str, str] = {}
    for rule in FAST_LANE_RULES:
        for phrase in rule.phrases:
            assert phrase not in owners, (
                f"短语 {phrase!r} 同时被 {owners.get(phrase)!r} 与 {rule.name!r} 认领，"
                f"实际生效的是靠前的那条。"
            )
            owners[phrase] = rule.name


def test_every_intent_has_a_target_agent() -> None:
    """★ 每个 :class:`Intent` 成员都必须登记目标智能体。

    ⚠️ 漏登记的后果取决于 :func:`_INTENT_TARGET_AGENTS` 的查询写法：
    直接下标会 ``KeyError``（还算好，快速失败）；``.get(intent, ())`` 则会
    得到**空目标列表** —— 用户发一句话，系统判定出意图，然后**谁也不调**，
    回复是一片空白。这类「静默无响应」是最难排查的形态。
    """
    missing = set(Intent) - set(_INTENT_TARGET_AGENTS)

    assert not missing, f"这些意图没有登记目标智能体：{[i.value for i in missing]}"


def test_every_rule_intent_is_registered() -> None:
    """规则表里用到的意图，必须在目标映射表里有登记（与上一条互补）。

    ⚠️ 两条断言不能合并：上一条遍历 :class:`Intent` 的**成员**，这一条遍历
    规则表里的**取值**。若有人给规则填了一个新意图却忘了加映射，只有这一条
    会红。
    """
    for rule in FAST_LANE_RULES:
        assert rule.intent in _INTENT_TARGET_AGENTS, (
            f"规则 {rule.name!r} 用了未登记目标智能体的意图 {rule.intent.value!r}"
        )


def test_all_target_agents_are_known_names() -> None:
    """目标映射表里的每个值都必须是 :class:`AgentName` 的成员。

    ⚠️ 断言的是**类型**而不是字符串相等：这张表的注解已经是
    ``tuple[AgentName, ...]``，但 Python 不做运行时强制 —— 有人写了一个
    拼错的字符串（``"main_planner"``），类型检查器在没开 strict 的 CI 上
    未必会拦。而下游 :file:`src/agents/registry.py` 按名字查表，
    查不到就是一个只在那条路径上才出现的 ``KeyError``。
    """
    for intent, agents in _INTENT_TARGET_AGENTS.items():
        for agent in agents:
            assert isinstance(agent, AgentName), (
                f"意图 {intent.value!r} 的目标 {agent!r} 不是 AgentName 成员"
            )


# ==============================================================================
# 二、快车道：该命中的必须命中
# ==============================================================================
#: 博客原文里的按钮文案（``docs/博客原文-Alibaba-Business-Travel.md`` 第 141 行起）。
#:
#: ⚠️ 单独列出来测，是为了把「博客的按钮一定走快车道」变成一条**可执行的
#: 事实**。规则表里刻意不收录这些未归一化的原文（收录了就是死短语，见
#: :func:`test_every_rule_phrase_survives_normalization`），于是这个保证
#: 就只能由用例来提供 —— 这也正是它该待的地方。
_BLOG_BUTTONS: list[tuple[str, Intent]] = [
    ("为我规划行程", Intent.PLAN_TRIP),
    ("为我提申请", Intent.APPLY_APPROVAL),
    ("开始规划", Intent.PLAN_TRIP),
]


@pytest.mark.parametrize(("text", "expected"), _BLOG_BUTTONS)
def test_blog_button_texts_route_to_the_fast_lane(text: str, expected: Intent) -> None:
    """博客里的按钮文案必须命中快车道，且落到正确意图。"""
    decision = classify(text)

    assert decision.lane == LaneName.FAST
    assert decision.intent == expected


#: 各种口语变体 —— 都应当在剥掉客套与语气词后命中同一条规则。
_COLLOQUIAL_FORMS: list[tuple[str, str]] = [
    ("请帮我规划行程吧", "plan_trip"),
    ("麻烦安排行程", "plan_trip"),
    ("规划行程！", "plan_trip"),
    ("  规划 行程  ", "plan_trip"),
    ("帮我查订单啊", "query_order"),
    ("我要取消", "cancel_trip"),
    ("报销标准", "query_policy"),
    ("报销标准。", "query_policy"),
]


@pytest.mark.parametrize(("text", "rule_name"), _COLLOQUIAL_FORMS)
def test_colloquial_forms_hit_the_intended_rule(text: str, rule_name: str) -> None:
    """口语化写法在归一化后应命中同一条规则。

    ⚠️ 覆盖的是「客套词 + 语气词 + 标点 + 空格」这四种噪声的**组合**，
    而不是单独一种 —— 实际用户输入里它们总是同时出现，而
    :func:`_strip_affixes` 的不动点循环正是为组合设计的。
    """
    decision = classify(text)

    assert decision.lane == LaneName.FAST
    assert decision.matched_rule == rule_name


def test_fast_lane_decision_carries_rule_name_and_targets() -> None:
    """快车道判定必须带上规则名与目标智能体。

    ⚠️ ``matched_rule`` 不是装饰：它是「为什么这轮走了快车道」的唯一答案。
    线上灰度或异常时，这一条往往就是根因。丢掉它，日志里就只剩「走了快车道」。
    """
    decision = classify("为我规划行程")

    assert decision.lane == LaneName.FAST
    assert decision.matched_rule == "plan_trip"
    assert decision.target_agents == [AgentName.MAIN_PLAN.value]
    assert decision.reason


def test_cancel_is_fast_lane_but_says_it_will_confirm_first() -> None:
    """★ 取消走快车道，但判定文案必须写明**会先确认**。

    ⚠️ 这条守的是「快车道 ≠ 免确认」这个产品约束。快车道只省下「理解
    『取消订单』四个字」的那次模型调用，不省「真的去执行取消」之前的
    人工确认。若哪天有人把取消改成静默执行，这条用例至少会让文案先变得
    自相矛盾，从而在评审时被看见。
    """
    decision = classify("取消订单")

    assert decision.lane == LaneName.FAST
    assert decision.intent == Intent.CANCEL
    assert "确认" in decision.reason


# ==============================================================================
# 三、慢车道：不该命中的绝不能命中
# ==============================================================================
_QUESTION_INPUTS: list[str] = [
    "取消订单？",
    "取消订单?",
    "怎么规划行程",
    "如何提交申请",
    "报销标准是多少",
    "差旅政策是什么",
    "我的订单到哪了",
    "是不是可以取消订单",
    "能报多少呢",
    # ⚠️ 这条是**歧义**输入，刻意判给慢车道：「我的订单呢」既可以是在问
    # 「我的订单怎么样了」，也可以是一句极简的命令「给我看我的订单」。
    # 见 test_ambiguous_inputs_fall_to_the_safe_side。
    "我的订单呢",
]


@pytest.mark.parametrize("text", _QUESTION_INPUTS)
def test_questions_never_take_the_fast_lane(text: str) -> None:
    """★★ 问句一律走慢车道。

    ⚠️ 这是本组里后果最严重的一类误判。快车道命中后会**直接路由到执行链路**
    —— 用户问「取消订单？」，系统若把它当成命令，就会跳过语义理解直接去取消。
    用户是在打听，系统却动了手。

    ⚠️ 注意「是不是可以取消订单」这条：归一化**不会**削掉它（只剥前后缀），
    所以它本来就等于不了「取消订单」。真正兜住它的是问句判定。两条防线
    都要在，因为将来的归一化改动（比如有人加了「去掉『可以』前缀」）会
    让第一道失效 —— 那时第二道仍然是最后一道。
    """
    decision = classify(text)

    assert decision.lane == LaneName.SLOW, f"{text!r} 是问句，不该走快车道"


def test_ambiguous_inputs_fall_to_the_safe_side() -> None:
    """★★ 歧义输入一律判给慢车道。

    机制：「我的订单呢」既可以是在**问**「我的订单怎么样了」，也可以是一句
    极简的**命令**「给我看我的订单」。分类器不做这种消歧 —— 它把判定交给
    慢车道的意图识别智能体，那里有上下文可以做这个判断。

    ⚠️ 为什么这是对的，而不是「偷懒」：分类器能看到的只有这一句话。
    它若自作主张判成命令，用户问「我的订单呢」就会触发一次查询执行；
    判成问句则只是多花一次模型调用 —— 而那次调用恰好**能**看到上下文，
    因此它做出的判断比分类器更可信。**在信息不足时，把决定权交给信息更多的
    那一方**，这是快慢车道分流的基本原则。

    ⚠️ 这条用例的价值在于把「歧义 → 保守」这个取向写下来。它是可争论的
    产品决定，所以需要一个地方记录「我们是有意这样做的」，免得后来的人
    把它当 bug「修」掉。
    """
    decision = classify("我的订单呢")

    assert decision.lane == LaneName.SLOW
    assert decision.target_agents == [AgentName.INTENT.value]


def test_long_sentence_takes_the_slow_lane() -> None:
    """长句子走慢车道 —— 它里面有需要理解的信息。

    ⚠️ 这里的输入**不含任何触发短语**，所以它走慢车道主要靠「匹配不上」。
    ``max_chars`` 是第二道网，单独由下一个用例验证。
    """
    decision = classify("下周三去北京开会，帮我看看住哪儿合适，另外顺便订个高铁")

    assert decision.lane == LaneName.SLOW


def test_max_chars_guard_actually_blocks_matches() -> None:
    """``max_chars`` 真的能拦住本可命中的输入。

    ⚠️ 用一个**极小**的阈值来验证，因为正常阈值下（20）不存在「长到超限
    却还能精确命中」的输入 —— 精确匹配本身就隐含了短。这条参数是为**将来**
    给归一化加规则时引入的意外塌缩准备的（详见 ``classify`` 的 docstring），
    所以要直接测它本身，而不是等某个真实场景去触发它。
    """
    assert classify("规划行程", max_chars=2).lane == LaneName.SLOW
    assert classify("规划行程", max_chars=20).lane == LaneName.FAST


def test_disabling_the_fast_lane_forces_slow_for_every_rule() -> None:
    """★ ``enabled=False`` 时，**任何**触发短语都不得走快车道。

    ⚠️ 遍历**全部**规则的全部短语，而不是抽查一两条：这个开关的用途是
    「线上怀疑快车道出错时一刀切开两种可能」，它必须是一条**绝对的**
    保证。抽查的话，漏掉的那条规则恰好就是没被关掉的那条 —— 排查时
    会得到「关掉了还是有快车道请求」的假象，从而把方向带到完全错误的地方。

    ⚠️ 同时断言 reason 里点出是**被开关关掉的**。否则日志里出现的
    「未命中快车道规则」会让人去翻规则表，而真正的原因在配置里。
    """
    for rule in FAST_LANE_RULES:
        for phrase in rule.phrases:
            decision = classify(phrase, enabled=False)
            assert decision.lane == LaneName.SLOW, f"{phrase!r} 在快车道关闭时仍走了快车道"

    assert "fast_lane_enabled" in classify("规划行程", enabled=False).reason


def test_slow_lane_targets_the_intent_agent_first() -> None:
    """★ 慢车道的第一个目标必须是**意图识别智能体**。

    ⚠️ 这条守的是「思考链上显示的是事实」。慢车道上第一个被调用的确实是
    意图识别智能体；若这里直接写主规划智能体，前端会显示「正在规划行程」，
    而系统其实还在识别意图。用户看到的进度与系统真实状态不符，是观测性的
    根本性缺陷 —— 因为它让界面**无法**用来排查问题。
    """
    decision = classify("下周三去北京开会，帮我看看住哪儿合适")

    assert decision.lane == LaneName.SLOW
    assert decision.target_agents == [AgentName.INTENT.value]


def test_stage_only_enriches_the_reason_never_the_routing() -> None:
    """★ ``stage`` 只影响文案，不影响车道 / 意图 / 规则。

    ⚠️ 这条钉住的是 classifier 的 docstring 里那句「阶段影响的是慢车道内部
    怎么组织提示词，而不是这句话该不该用大模型读」。让阶段参与路由，会造成
    **同一句话在不同阶段走不同车道** —— 用户复现不了，日志里也看不出规律，
    是排查噩梦。

    ⚠️ 用「同一输入 + 多个阶段」做对比，而不是只测一个阶段：只测一个的话，
    「阶段根本没被读」与「阶段被读了但恰好没改变结果」无法区分。
    """
    text = "规划行程"
    baseline = classify(text)

    for stage in TripStage:
        decision = classify(text, stage=stage)
        assert decision.lane == baseline.lane
        assert decision.intent == baseline.intent
        assert decision.matched_rule == baseline.matched_rule
        assert decision.target_agents == baseline.target_agents


def test_stage_is_appended_to_the_reason_including_for_slow_lane() -> None:
    """有阶段时，快慢两条车道的 ``reason`` 都要带上它。

    ⚠️ 慢车道也要带 —— 而慢车道恰好是最需要它的时候：线上出现「同一句话
    有时走快有时走慢」的疑问时，reason 里的阶段是唯一能解释差异的线索。
    """
    assert "COLLECTING" in classify("规划行程", stage=TripStage.COLLECTING).reason
    assert "CONFIRMING" in classify("随便说点什么", stage=TripStage.CONFIRMING).reason


def test_reason_has_no_stage_suffix_when_stage_is_unknown() -> None:
    """没传阶段时，``reason`` 里不得出现 ``None`` 字样。

    ⚠️ 给 None 也拼一句「当前阶段：None」，会让日志里塞满无意义的字串，
    反而淹没真正带阶段信息的那几条。空值应当表现为**没有那句话**。
    """
    assert "None" not in classify("规划行程").reason


# ==============================================================================
# 四、第二跳：意图 → 目标智能体
# ==============================================================================
def test_route_for_intent_always_reports_slow_lane() -> None:
    """★★ :func:`route_for_intent` 返回的车道**恒为 SLOW**，哪怕意图很简单。

    ⚠️ 这条不是形式主义。「这一轮走没走过大模型」是**事实**；若这里因为
    「结果看起来很简单」就返回 FAST，指标与日志里的快车道占比就会失真 ——
    而那个数字正是评估快车道规则覆盖率时唯一的依据。数据一旦不实，
    优化就失去了方向。
    """
    for intent in Intent:
        assert route_for_intent(intent).lane == LaneName.SLOW, (
            f"意图 {intent.value} 的第二跳被标成了快车道"
        )


@pytest.mark.parametrize(
    ("intent", "expected"),
    [
        (Intent.PLAN_TRIP, AgentName.MAIN_PLAN),
        (Intent.APPLY_APPROVAL, AgentName.APPROVAL),
        (Intent.QUERY_POLICY, AgentName.POLICY_RAG),
        (Intent.QUERY_ORDER, AgentName.ORDER_QUERY),
        (Intent.MODIFY_TRIP, AgentName.MAIN_PLAN),
        (Intent.CANCEL, AgentName.MAIN_PLAN),
        (Intent.CHITCHAT, AgentName.MAIN_PLAN),
    ],
)
def test_route_for_intent_maps_to_the_expected_agent(intent: Intent, expected: AgentName) -> None:
    """每个意图落到哪个智能体 —— 逐个钉死。

    ⚠️ 期望值是**独立写出来**的第二份事实，不遍历 ``_INTENT_TARGET_AGENTS``
    求值。遍历那张表来断言它自己，等于同义反复：表改错了，用例跟着一起错，
    照样全绿。
    """
    assert route_for_intent(intent).target_agents == [expected.value]


def test_other_intent_is_never_left_without_a_target() -> None:
    """兜底意图 ``OTHER`` 也必须有人接。

    ⚠️ 指向空列表的话，用户说一句系统听不懂的话，会得到一个**没有任何
    智能体处理**的回复 —— 界面上就是一片空白。兜底要落在「主智能体先接住
    并追问」，而不是「谁也不管」。
    """
    assert route_for_intent(Intent.OTHER).target_agents


def test_target_agents_for_returns_a_fresh_list() -> None:
    """★ :func:`target_agents_for` 每次返回**新**列表。

    ⚠️ 若返回的是共享对象（例如直接返回全局元组转出来的同一份 list），
    某个调用方一句 ``append`` 就会永久污染全局路由表 —— 之后**所有**用户
    的请求都会被路由到多出来的那个智能体。而故障现场离改动点很远，
    几乎不可能靠读代码发现。
    """
    first = target_agents_for(Intent.PLAN_TRIP)
    first.append("污染")

    assert target_agents_for(Intent.PLAN_TRIP) == [AgentName.MAIN_PLAN.value]


# ==============================================================================
# 五、归一化的细节
# ==============================================================================
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("请帮我规划行程吧", "规划行程"),
        ("我的订单呢", "订单"),
        ("我要取消", "取消"),
        ("规划 行程", "规划行程"),
        ("规划行程！", "规划行程"),
        ("  查订单  ", "查订单"),
    ],
)
def test_normalize_strips_politeness_particles_and_punctuation(raw: str, expected: str) -> None:
    """归一化剥掉客套词、语气词、标点、空白。"""
    assert _normalize(raw) == expected


@pytest.mark.parametrize("raw", ["下单", "目的", "下单流程", "目的地"])
def test_normalize_does_not_overstrip(raw: str) -> None:
    """★ 归一化**不得**削掉单字「下」「的」。

    ⚠️ 把「下」「的」收进语气词列表看起来能提高命中率，实际是在**改写用户
    的话**：「下单」会变成「单」、「目的」会变成「目」，两个词直接失去意义。
    而更危险的是长句：过度剥离可能把一句无关的话削成一条触发短语，
    于是快车道被误命中。

    ⚠️ 这条用例的存在本身就是文档：它告诉后来的人「这里少收两个字是刻意的」，
    免得有人出于「补全」的心理把两字加回去。
    """
    assert _normalize(raw) == raw


@pytest.mark.parametrize("raw", ["   ", "。。。", "，。！", "\t\n", "  !?  "])
def test_contentless_input_never_takes_the_fast_lane(raw: str) -> None:
    """★★ 没有任何内容字符的输入**绝不能**走快车道。

    ⚠️ 这是「归一化可能返回空串」（见 :func:`_normalize` 的 docstring）所
    对应的**真正**该被守住的保证。我最初把它写成「``_normalize`` 永不返回
    空串」，结果被这条用例的输入抓出来是假的 —— 空白与纯标点输入本来就会被
    剥成空串。

    ⚠️ 保证的位置很关键：它必须在 :func:`classify` 里**显式**拦截，不能
    依赖「规则表里恰好没有空短语」。空短语一点都不难出现（从配置读进一个
    空字符串就够了），而一旦出现，「用户什么都没说，系统却判定出意图并去
    执行」就会成真 —— 这是本项目里最荒谬的一类故障。

    ⚠️ 断言的是 lane 而不只是 `_normalize` 的返回值：`_normalize` 返回空串
    是**如实描述**（这串输入确实没有可比较的内容），不是缺陷；缺陷是让这个
    空串一路走到规则比对里去。
    """
    assert classify(raw).lane == LaneName.SLOW, f"{raw!r} 是空输入，不该走快车道"


def test_normalize_keeps_content_bearing_affix_only_input() -> None:
    """★ 只由客套词构成的输入**不会**被剥成空串。

    与上一条互补，守的是另一条边界：``_strip_affixes`` 里
    ``len(text) > len(prefix)`` 的条件。没有它的话，「帮我」会被剥成空 ——
    而这正是下一层空串判定的输入。两道边界各自成立，合起来才是完整的。
    """
    for raw in ("帮我", "我", "请", "我的", "请帮我"):
        assert _normalize(raw) != "", f"{raw!r} 被剥成了空串（客套词不该被剥光）"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("取消订单？", True),
        ("怎么规划行程", True),
        ("报销标准是多少", True),
        ("我的订单呢", True),
        ("规划行程", False),
        ("取消订单", False),
        ("查订单", False),
    ],
)
def test_question_detection(text: str, expected: bool) -> None:
    """问句判定 —— 含标记即视为问句。

    ⚠️ 注意最后三条：**不带问号、不带疑问词的祈使句**不是问句。这是快车道
    的主力输入形态，判定必须为 False，否则快车道等于不存在。
    """
    assert _is_question(text) is expected


# ==============================================================================
# 六、纯度：本模块不依赖框架
# ==============================================================================
#: 在**子进程**里验证「导入 classifier 不会连带导入 agentscope」。
#:
#: ⚠️ 为什么必须在子进程里做：pytest 进程自己早就 import 过 ``agentscope``
#: （``conftest.py`` 装配应用时会用到），所以在当前进程里断言
#: ``"agentscope" not in sys.modules`` 永远是 False —— 那样的用例只会
#: 变成一条恒红的噪声，或者更糟：被人改成 ``pytest.skip`` 之后彻底失效。
#: 干净的解释器是唯一能验证「不 import」这件事的地方。
_PURITY_PROBE = """
import sys
from src.orchestration.classifier import classify

leaked = sorted(m for m in sys.modules if m == "agentscope" or m.startswith("agentscope."))
assert not leaked, f"导入 classifier 连带拉进了框架：{leaked}"
print(classify("规划行程").lane.value)
"""


def test_classifier_does_not_pull_in_the_framework() -> None:
    """★★ 单独导入 classifier **不得**把 ``agentscope`` 拉进来。

    这条守的是 :mod:`src.orchestration` 包文档里写的那条性质：快慢车道规则
    是本项目里最该能「脱离框架被验证」的部分（与 :mod:`src.domain` 同理）。
    一旦有人在 ``classifier.py`` 里加一句框架 import，或者往
    ``src/orchestration/__init__.py`` 里加上 ``lane`` 的导出，这条就会红。

    ⚠️ 顺带断言 ``classify`` 真的跑出了结果（``FAST``）：只查「没导入框架」
    的话，一个在导入期就崩掉的模块也能让断言"通过不了"—— 但更隐蔽的情形是
    模块被换成了一个空壳，什么都不做却也不导入框架。跑一句真实分类能同时
    证明「能导入」与「能工作」。
    """
    result = subprocess.run(
        [sys.executable, "-c", _PURITY_PROBE],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"子进程失败：\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert result.stdout.strip() == LaneName.FAST.value
