# -*- coding: utf-8 -*-
"""**判分器** —— 把「一条用例 + 一次观测」变成一句可比较的结论。

文件职责：
    提供两种判分实现与一个按配置选路的工厂：

        1. :class:`RuleJudge`  —— **规则判分**。只看能核对的东西：
           车道是否一致、意图集是否吻合、工具集是否吻合。不联网、不花钱、
           完全确定性 —— 因此可以在 CI 里对每一次提交跑。
        2. :class:`LLMJudge`   —— **大模型判分**（LLM-as-judge）。把用例与
           观测交给模型，让它给一个 ``0~1`` 的分数与一句理由。它评估的是
           规则看不见的东西（措辞、是否答到点上、是否胡编）。
        3. :func:`build_judge` —— 按配置选路：**没有模型密钥时一律返回
           :class:`RuleJudge`**，并在报告里如实注明。

上下游依赖：
    - 上游：:mod:`src.evaluation.types`（数据契约）、:mod:`src.llm.factory`
      （模型装配与零密钥判据）、:mod:`src.domain.enums`、``agentscope``
      （``Agent.reply`` 的结构化输出机制）。
    - 下游：``src/evaluation/runner.py``（每条用例调一次判分）、
      ``scripts/eval.py``。

＝ 为什么框架里没有 evaluate 模块，这里就得把两条判分路都自己写 ＝

    ``agentscope`` 提供的是**模型与工具**，不是一个评测框架 ——
    它没有一个「拿期望比实际」的组件。所以判分的语义（比什么、怎么算分、
    怎么在离线与在线之间切换）完全是本项目的决定，必须落在我们自己的代码里，
    而不是指望某个 ``import`` 能提供。

═══ ⚠️ 离线判分必须**诚实**，这是本模块最重要的一条约束 ═══

    没有密钥时，我们**不会**让模型去判分（那会是一次注定失败的调用），
    而是退回到规则判分。规则判分能看的东西**就是**那三样集合，
    它看不见「回答得好不好」。于是有两种写法：

        ✗ 把规则判分的结论包装成 ``mode="llm"``，让它看起来像一次语义评估；
        ✓ 把它标成 ``mode="rule"``，并在报告里写明「本次不是 LLM 判分」。

    本模块选后者。理由不是道德：一个被标错的判分模式会让读报告的人
    **系统性地高估**报告的可信度 —— 他看到「判分均分 0.97」会以为
    「语义质量很高」，而那个数字其实只说了「集合对得上」。当真正的问题
    （答非所问但意图判对了）出现时，这个被高估的指标会**掩盖**它。
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase
from pydantic import BaseModel, ConfigDict, Field

from src.config.schema import Settings
from src.domain.enums import LaneName
from src.evaluation.types import (
    DIM_INTENTS,
    DIM_TOOLS,
    EvalCase,
    JudgeCheck,
    JudgeVerdict,
    Observation,
    _ratio,
    _set_f1,
)
from src.llm.factory import build_chat_model, should_use_mock

logger = logging.getLogger(__name__)

#: 判分模式：规则判分（确定性、离线可用、只看可核对项）。
JUDGE_MODE_RULE = "rule"

#: 判分模式：大模型判分（语义质量、需要密钥）。
JUDGE_MODE_LLM = "llm"

#: LLM 判分器的系统提示词。
#:
#: ⚠️ 写成「只输出分数与一句理由」而不是一段开放式要求，是因为这条链路走的是
#: **结构化输出**（框架把 schema 注册成一个工具逼模型按字段交付，见
#: :mod:`src.agents.intent` 的说明）。提示词与 schema 说的必须是同一件事，
#: 否则模型会在「按提示词自由发挥」与「按 schema 填字段」之间摇摆。
_JUDGE_SYSTEM_PROMPT = (
    "你是差旅助手的评测员。你会看到一条用户输入、系统应当给出的期望，"
    "以及系统实际给出的结果。请判断实际结果与期望的吻合程度，"
    "给一个 0 到 1 之间的小数（1 表示完全吻合），并用一句话说明理由。"
    "只依据给出的信息判断，不要臆测系统的内部实现。"
)


class _JudgeDecision(BaseModel):
    """LLM 判分的结构化输出模型。

    ⚠️ 与本项目所有「给模型看的 schema」一样，字段的 ``description``
    是**提示词**而不是注释，因此写得短、具体、无歧义（见
    :mod:`src.domain.schemas` 的模块说明）。

    ``extra="ignore"``（而不是配置层用的 ``forbid``）：这是接收**模型输出**
    的模型，模型偶尔多吐一个自造字段时应当丢弃它而不是让整次判分报废。
    """

    model_config = ConfigDict(extra="ignore")

    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="实际结果与期望的吻合程度，0 到 1 之间的小数，1 表示完全吻合",
    )
    rationale: str = Field(default="", description="给这个分数的一句话理由")


class Judge(Protocol):
    """判分器的统一接口。

    ⚠️ 接口定义成 **async**：两种实现的内部差异很大（一个纯计算、一个要
    联网），但调用方不该因此写两份 ``await``/不 ``await`` 的分支 ——
    那种分支迟早会在某一侧漏掉 ``await``，得到一个「分数是协程对象」的
    诡异报告。让同步实现也返回一个可 ``await`` 的协程，是最便宜的抹平方式
    （:class:`RuleJudge` 里没有真正的等待，开销可以忽略）。
    """

    @property
    def mode(self) -> str:
        """判分模式。

        Returns:
            `str`: ``"rule"`` 或 ``"llm"``。
        """
        ...

    async def judge(self, case: EvalCase, observation: Observation) -> JudgeVerdict:
        """对一条观测判分。

        Args:
            case (`EvalCase`): 黄金用例（含期望）。
            observation (`Observation`): 系统实际给出的结果。

        Returns:
            `JudgeVerdict`: 判分结论。
        """
        ...


# ==============================================================================
# 一、规则判分
# ==============================================================================
class RuleJudge:
    """**确定性规则判分器** —— 只比能核对的东西。

    它判三件事，且**只判用例声明过的**：

        · **车道** —— 实际车道是否等于期望车道；
        · **意图集** —— 实际意图集合与期望集合的 F1；
        · **工具集** —— 实际工具集合与期望集合的 F1。

    综合得分是这三个维度得分的算术平均（只统计被声明过的维度）。
    """

    @property
    def mode(self) -> str:
        """判分模式，恒为 ``"rule"``。

        Returns:
            `str`: :data:`JUDGE_MODE_RULE`。
        """
        return JUDGE_MODE_RULE

    async def judge(self, case: EvalCase, observation: Observation) -> JudgeVerdict:
        """按可核对项给分。

        Args:
            case (`EvalCase`): 黄金用例。
            observation (`Observation`): 实际观测。

        Returns:
            `JudgeVerdict`: 分数、``mode="rule"``、逐项明细。

        ⚠️ 三个维度都要**同时**满足「用例声明了它」与「本次观测真的测到了它」
        才判。第二个条件来自 :attr:`~src.evaluation.types.Observation.unmeasured`，
        它不是保险措施而是**正确性**要求：离线的慢车道意图来自一个确定性
        Mock 模型，本来就识别不出真实意图，把它按 0 分判会让综合得分反映
        「本机没有密钥」而不是「系统变差了」。详见
        :data:`~src.evaluation.types.DIM_INTENTS` 上方那段说明。

        ⚠️ 反过来，**没有任何可判分项**的用例判 **1.0** 并如实写明理由，
        而不是判 0。「没东西可判」不是「判它不及格」；这个 1.0 之所以安全，
        是因为 :attr:`~src.evaluation.types.EvalReport.overall_score` 只在
        :attr:`~src.evaluation.types.EvalMetrics.judge_evaluated` 大于 0 时
        才把判分均分计入 —— 一份全是「无可判分」的数据集，其
        ``judge_evaluated`` 为 0，这些白送的 1.0 一分都进不了综合得分。
        """
        checks: list[JudgeCheck] = []
        skipped: list[str] = []

        if case.expected_lane is not None:
            expected_lane = case.expected_lane.value
            ok = observation.lane == expected_lane
            checks.append(
                JudgeCheck(
                    name="车道",
                    score=1.0 if ok else 0.0,
                    detail=f"期望 {expected_lane}，实际 {observation.lane or '<空>'}",
                ),
            )

        if case.expected_intents:
            if observation.measures(DIM_INTENTS):
                expected = frozenset(intent.value for intent in case.expected_intents)
                actual = frozenset(observation.intents)
                score = _set_f1(expected, actual)
                checks.append(
                    JudgeCheck(
                        name="意图集",
                        score=score,
                        detail=f"期望 {sorted(expected)}，实际 {sorted(actual)}",
                    ),
                )
            else:
                skipped.append("意图集")

        if case.expected_tools:
            if observation.measures(DIM_TOOLS):
                expected = frozenset(case.expected_tools)
                actual = frozenset(observation.tools)
                score = _set_f1(expected, actual)
                checks.append(
                    JudgeCheck(
                        name="工具集",
                        score=score,
                        detail=f"期望 {sorted(expected)}，实际 {sorted(actual)}",
                    ),
                )
            else:
                skipped.append("工具集")

        # ⚠️ 被跳过的维度写进理由，且**不计分**。理由是报告逐条打印的那句话，
        # 不写的话读报告的人只会看到一个比期望高的分数，无从知道
        # 「有一项根本没比」。
        skip_note = f"（{'、'.join(skipped)}未测量，本次不计分）" if skipped else ""

        if not checks:
            return JudgeVerdict(
                score=1.0,
                mode=JUDGE_MODE_RULE,
                rationale=f"该用例没有声明任何期望项（或声明的维度均未测量），"
                f"无可判分内容（判为通过）{skip_note}。",
            )

        score = round(sum(check.score for check in checks) / len(checks), 4)
        rationale = (
            "；".join(f"{check.name} {check.score:.4f}（{check.detail}）" for check in checks)
            + skip_note
        )
        return JudgeVerdict(
            score=score,
            mode=JUDGE_MODE_RULE,
            rationale=rationale,
            checks=tuple(checks),
        )


# ==============================================================================
# 二、大模型判分
# ==============================================================================
class LLMJudge:
    """**大模型判分器** —— 让模型给一个语义层面的分。

    ⚠️ 它只在**配置了模型密钥**时才会被 :func:`build_judge` 选中。没有密钥时
    选中它等于每次判分都发一次注定失败的请求 —— 那既慢又贵（重试），
    还会把「没有密钥」伪装成「模型判分不可用」这个看起来像 bug 的现象。

    ⚠️ 模型调用一旦失败，本类**退回规则判分**，并让返回的 verdict 的
    ``mode`` 变成 ``"rule"``。降级必须是**可见的**：蒙混成一次成功的 llm 判分，
    报告里就会出现一批「来源不明」的分数。

    Attributes:
        _model (`ChatModelBase`): 用于判分的模型。
        _system_prompt (`str`): 系统提示词。
    """

    def __init__(
        self,
        model: ChatModelBase,
        *,
        system_prompt: str | None = None,
    ) -> None:
        """初始化。

        Args:
            model (`ChatModelBase`): 模型实例。复用同一实例是安全的
                （不持有对话状态，见 :mod:`src.agents.intent` 的说明）。
            system_prompt (`str | None`): 覆盖默认提示词，仅供测试与灰度对比。
        """
        self._model = model
        self._system_prompt = system_prompt or _JUDGE_SYSTEM_PROMPT
        self._fallback = RuleJudge()

    @property
    def mode(self) -> str:
        """判分模式，恒为 ``"llm"``。

        Returns:
            `str`: :data:`JUDGE_MODE_LLM`。
        """
        return JUDGE_MODE_LLM

    async def judge(self, case: EvalCase, observation: Observation) -> JudgeVerdict:
        """让模型对一条观测判分（失败时可见地退回规则判分）。

        Args:
            case (`EvalCase`): 黄金用例。
            observation (`Observation`): 实际观测。

        Returns:
            `JudgeVerdict`: 模型判分结论；模型不可用或没交付结构化结果时，
            是**带** ``mode="rule"`` 的规则判分结论。
        """
        try:
            decision = await self._judge_with_model(case, observation)
        except Exception:  # noqa: BLE001 —— 判分失败不该让整轮评测中断
            logger.exception("LLM 判分失败，本条退回规则判分（报告会标为 rule）。")
            decision = None

        if decision is None:
            fallback = await self._fallback.judge(case, observation)
            return JudgeVerdict(
                score=fallback.score,
                mode=JUDGE_MODE_RULE,
                rationale=f"（LLM 判分不可用，已退回规则判分）{fallback.rationale}",
                checks=fallback.checks,
            )

        return JudgeVerdict(
            score=round(float(decision.score), 4),
            mode=JUDGE_MODE_LLM,
            rationale=decision.rationale,
        )

    async def _judge_with_model(
        self,
        case: EvalCase,
        observation: Observation,
    ) -> _JudgeDecision | None:
        """真正调模型的那一步（结构化输出）。

        ⚠️ 每次判分都**新建** ``Agent``，理由与 :mod:`src.agents.intent` 完全
        一致：``AgentState`` 里的上下文是累积的，复用实例会让后一条用例
        读到前一条的判分语境。新建它的成本只是几次对象构造。

        Returns:
            `_JudgeDecision | None`: 解析出的判分；模型没交付时为 ``None``。
        """
        agent = Agent(
            name="eval_judge",
            system_prompt=self._system_prompt,
            model=self._model,
            toolkit=None,
        )
        message = await agent.reply(
            inputs=Msg(name="user", role="user", content=[TextBlock(text=_render_prompt(case, observation))]),
            structured_schema=_JudgeDecision,
        )
        raw = getattr(message, "structured_output", None)
        if not isinstance(raw, dict):
            logger.warning("LLM 判分没有得到结构化结果，本条退回规则判分。")
            return None
        try:
            return _JudgeDecision.model_validate(raw)
        except Exception:  # noqa: BLE001
            logger.exception("LLM 判分的结构化结果无法解析，本条退回规则判分。")
            return None


def _render_prompt(case: EvalCase, observation: Observation) -> str:
    """把用例与观测拼成给判分模型的提示词。

    ⚠️ 提示词里**不含**密钥、不含仓储内容 —— 它只有「输入、期望、实际」三段。
    判分提示词会被模型服务商看到，任何多余的信息都是泄漏面。

    Args:
        case (`EvalCase`): 黄金用例。
        observation (`Observation`): 实际观测。

    Returns:
        `str`: 三段式提示词。
    """
    expected_intents = ", ".join(intent.value for intent in case.expected_intents) or "（未声明）"
    expected_tools = ", ".join(case.expected_tools) or "（未声明）"
    return (
        f"【用户输入】\n{case.input}\n\n"
        f"【期望】\n"
        f"车道：{case.expected_lane.value if case.expected_lane else '（未声明）'}\n"
        f"意图：{expected_intents}\n"
        f"工具：{expected_tools}\n\n"
        f"【实际】\n"
        f"车道：{observation.lane or '（空）'}\n"
        f"意图：{', '.join(observation.intents) or '（空）'}\n"
        f"工具：{', '.join(observation.tools) or '（空）'}\n"
        f"输出：{observation.output or '（空）'}"
    )


# ==============================================================================
# 三、选路
# ==============================================================================
def build_judge(
    settings: Settings,
    model: ChatModelBase | None = None,
) -> Judge:
    """按配置选一个判分器。

    ⚠️ 判据直接复用 :func:`src.llm.factory.should_use_mock`，**不另写一份**。
    这条判据是「有没有密钥」这件事在项目里的唯一真值；在别处再判一次
    （比如 ``if settings.llm.api_key``）就会与它漂移 —— 而漂移的症状是
    「评测用 LLM 判分、生产用 Mock 模型」这种把评测结论彻底带偏的组合，
    且不会报任何错。

    Args:
        settings (`Settings`): 全量配置。
        model (`ChatModelBase | None`): 模型；``None`` 时按配置构造（仅在需要
            LLM 判分时才构造，从而不在离线时白白装配一个 Mock）。

    Returns:
        `Judge`: 无密钥时为 :class:`RuleJudge`，否则为 :class:`LLMJudge`。
    """
    if should_use_mock(settings):
        logger.info(
            "未配置模型密钥，评测将使用**规则判分**（只比车道/意图集/工具集，"
            "不做语义评估）；报告里会如实标注 judge_mode=rule。",
        )
        return RuleJudge()
    return LLMJudge(model if model is not None else build_chat_model(settings))


def judge_mode(judge: Judge) -> str:
    """读出判分器的模式名（供报告使用）。

    Args:
        judge (`Judge`): 判分器。

    Returns:
        `str`: ``"rule"`` 或 ``"llm"``。

    ⚠️ 用一个函数而不是直接取 ``judge.mode``：调用方大多只知道「拿到的是
    某个 Judge」，把取值收在一处，将来若给 Judge 增加模式（比如「人工复核」）
    也只需要改这里。取值失败时退回 :data:`JUDGE_MODE_RULE` ——
    报一个**更保守**的模式，比报错的模式安全：宁可让人以为
    「这次只看集合」，也不要让人误以为「语义被评估过了」。
    """
    return getattr(judge, "mode", JUDGE_MODE_RULE)


__all__ = [
    "JUDGE_MODE_LLM",
    "JUDGE_MODE_RULE",
    "Judge",
    "LLMJudge",
    "RuleJudge",
    "build_judge",
    "judge_mode",
]
