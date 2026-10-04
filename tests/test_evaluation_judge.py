# -*- coding: utf-8 -*-
"""判分器（``src/evaluation/judge.py``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    判分器是评测里**唯一**给出「分数」的地方，因此它有两条互相拉扯的失败模式：

      1. **判分造假** —— 离线时用规则判分，却把它标成 ``mode="llm"``。
         报告里会出现一个「看起来做过语义评估」的分数，而它其实只比了
         几个集合是否相等。这种错误的危害不是「分数不准」，而是**它系统性地
         高估了报告的可信度**，让人在真正的问题（答非所问）出现时不再去看。
      2. **判分失能** —— 任何输入都返回同一个分数（比如恒 1.0 或恒 0.0）。
         它不会报错，只是让评测彻底失去意义，且与「系统表现稳定」无法区分。

    用例围绕这两条写：既钉住「模式标签必须诚实」，也钉住「分数必须随
    实际吻合度变化」（用一条近似命中拿部分分的用例来证明它不是布尔判定）。
"""

from __future__ import annotations

import pytest

from src.config import Settings, load_settings
from src.domain.enums import Intent, LaneName
from src.evaluation.judge import (
    JUDGE_MODE_LLM,
    JUDGE_MODE_RULE,
    LLMJudge,
    RuleJudge,
    build_judge,
    judge_mode,
)
from src.evaluation.types import EvalCase, JudgeVerdict, Observation
from src.llm.mock import MockChatModel


def _settings(**overrides: str) -> Settings:
    """构造一份测试配置，并允许按 ``ALIGO__`` 环境变量的形式覆盖。

    Args:
        **overrides (`str`): 形如 ``ALIGO__LLM__API_KEY="sk-x"`` 的覆盖项。

    Returns:
        `Settings`: 完整配置对象。
    """
    from tests.conftest import TEST_ENVIRON

    environ = {**TEST_ENVIRON, **overrides}
    return load_settings("test", environ=environ, dotenv=False)


def _case(
    *,
    expected_lane: LaneName | None = None,
    expected_intents: tuple[Intent, ...] = (),
    expected_tools: tuple[str, ...] = (),
) -> EvalCase:
    """构造一条用于判分的用例。

    Args:
        expected_lane (`LaneName | None`): 期望车道。
        expected_intents (`tuple[Intent, ...]`): 期望意图集。
        expected_tools (`tuple[str, ...]`): 期望工具集。

    Returns:
        `EvalCase`: 用例。
    """
    return EvalCase(
        id="case-1",
        input="规划行程",
        expected_lane=expected_lane,
        expected_intents=expected_intents,
        expected_tools=expected_tools,
    )


# ==============================================================================
# 一、规则判分 —— 分数必须随吻合度变化
# ==============================================================================
async def test_rule_judge_full_match_scores_one() -> None:
    """三项全对 ⇒ 得分 1.0。

    正例。它是下面几条反例的对照基准 —— 没有它，「得分 0」也可能是判分器
    压根没工作（而不是真的判错了）。
    """
    case = _case(
        expected_lane=LaneName.FAST,
        expected_intents=(Intent.PLAN_TRIP,),
        expected_tools=("main_plan",),
    )
    observation = Observation(lane="FAST", intents=("PLAN_TRIP",), tools=("main_plan",))

    verdict = await RuleJudge().judge(case, observation)

    assert verdict.score == 1.0
    assert verdict.mode == JUDGE_MODE_RULE
    assert len(verdict.checks) == 3


async def test_rule_judge_lane_mismatch_scores_zero_on_that_dimension() -> None:
    """车道判错 ⇒ 车道这一维得 0，总分随之下降。

    ⚠️ 少了它，一个「无视输入、恒返回 1.0」的判分器能通过上面的正例 ——
    而那正是判分器最坏的一种失能：它让所有用例都「通过」。
    """
    case = _case(expected_lane=LaneName.FAST)
    observation = Observation(lane="SLOW")

    verdict = await RuleJudge().judge(case, observation)

    assert verdict.score == 0.0
    assert verdict.checks[0].name == "车道"


async def test_rule_judge_grades_partial_credit() -> None:
    """近似命中拿**部分分**（不是布尔）。

    ⚠️ 这是「判分不是摆设」最直接的一条证据：期望两条意图、实际只识别出
    一条时，得分必须严格落在 (0, 1) 之间。若实现写成「集合是否相等」，
    这条会得到 0 —— 而 0 会让「只差一点点」与「完全认错」在报告里无法区分，
    而这两者需要完全不同的处置。
    """
    case = _case(expected_intents=(Intent.PLAN_TRIP, Intent.QUERY_POLICY))
    observation = Observation(intents=("PLAN_TRIP",))

    verdict = await RuleJudge().judge(case, observation)

    # F1 = 2 * 1.0 * 0.5 / 1.5 ≈ 0.6667
    assert 0.0 < verdict.score < 1.0
    assert verdict.score == pytest.approx(0.6667, abs=1e-3)


async def test_rule_judge_extra_intent_is_penalised_too() -> None:
    """多识别出一条无关意图 ⇒ 同样扣分（精确率）。

    ⚠️ 反例方向：只惩罚「漏」不惩罚「多」的实现会让一个「把所有意图都报一遍」
    的系统拿满分 —— 那种系统在生产里等于什么都没判断。
    """
    case = _case(expected_intents=(Intent.PLAN_TRIP,))
    observation = Observation(intents=("PLAN_TRIP", "CANCEL"))

    verdict = await RuleJudge().judge(case, observation)

    assert 0.0 < verdict.score < 1.0


async def test_rule_judge_skips_undeclared_dimensions() -> None:
    """没声明的维度**不参与**判分，也不进检查项。

    ⚠️ 这不是细节。离线跑慢车道用例时，意图识别根本拿不到真实意图，于是
    数据集把 ``expected_intents`` 留空。若判分器把「空期望」当成「期望空集」
    去和实际比，这些用例会一律判负 —— 而它们本来就没打算评意图。
    指标由此从「离线也不可判」退化成「离线恒定失败」。
    """
    case = _case(expected_lane=LaneName.SLOW)
    observation = Observation(lane="SLOW", intents=("OTHER",), tools=("intent",))

    verdict = await RuleJudge().judge(case, observation)

    assert verdict.score == 1.0
    assert [check.name for check in verdict.checks] == ["车道"]


async def test_rule_judge_without_expectations_passes_with_reason() -> None:
    """三个维度都没声明 ⇒ 判 1.0，且理由里写明「无可判分内容」。

    ⚠️ 判 1.0 而不是 0，并**明说**理由：这种用例的存在意义只是「跑通链路」，
    判 0 会无缘无故拉低综合得分、掩盖真正的问题。但理由必须写清楚，
    免得读报告的人以为「这条测过了」。
    """
    verdict = await RuleJudge().judge(_case(), Observation())

    assert verdict.score == 1.0
    assert "无可判分" in verdict.rationale


# ==============================================================================
# 二、选路 —— 没有密钥就必须是规则判分
# ==============================================================================
def test_build_judge_uses_rule_when_no_key() -> None:
    """无密钥 ⇒ 规则判分。

    这是「CI 不联网也能评测」这条约束的直接体现（conftest 把 key 钉成空串）。
    """
    judge = build_judge(_settings())

    assert isinstance(judge, RuleJudge)
    assert judge_mode(judge) == JUDGE_MODE_RULE


def test_build_judge_uses_llm_when_key_present() -> None:
    """有密钥 ⇒ 大模型判分。

    反例。少了它，一个「无条件返回 RuleJudge」的实现能通过上面那条 ——
    而那意味着**配好了密钥的评测也永远只比集合**，语义评估形同虚设，
    且没有任何报错。
    """
    judge = build_judge(_settings(**{"ALIGO__LLM__API_KEY": "sk-test-not-a-real-key"}))

    assert isinstance(judge, LLMJudge)
    assert judge_mode(judge) == JUDGE_MODE_LLM


# ==============================================================================
# 三、LLM 判分的降级必须**可见**
# ==============================================================================
async def test_llm_judge_falls_back_to_rule_and_says_so() -> None:
    """LLM 判分拿不到结构化结果 ⇒ 退回规则判分，且 ``mode`` 变成 ``rule``。

    ★ 这是本文件最重要的一条：降级本身没有错（离线、服务抖动时都会发生），
    错的是**把降级藏起来**。若这里返回的仍是 ``mode="llm"``，报告里就会
    出现一批「来源不明」的分数 —— 读报告的人无从知道它们其实是规则的。

    ⚠️ 用一个注定交不出结构化结果的模型（``MockChatModel``）来触发降级，
    因此这条用例**不联网、不花钱**。
    """
    judge = LLMJudge(MockChatModel(model="judge-mock"))
    case = _case(expected_lane=LaneName.FAST)
    observation = Observation(lane="FAST")

    verdict = await judge.judge(case, observation)

    assert isinstance(verdict, JudgeVerdict)
    assert verdict.mode == JUDGE_MODE_RULE, "降级发生了，但模式标签仍写着 llm —— 报告会高估可信度"
    assert "退回规则判分" in verdict.rationale
    assert verdict.score == 1.0  # 规则判分下这条本来就该满分
