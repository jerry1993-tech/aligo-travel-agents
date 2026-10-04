# -*- coding: utf-8 -*-
"""评测的**数据契约**：黄金用例、单条结果、汇总报告与聚合指标。

文件职责：
    定义评测链路上被传递的那几个结构 —— :class:`EvalCase`（黄金用例）、
    :class:`Observation`（系统的一条实际观测）、:class:`JudgeVerdict`
    （一条判分）、:class:`CaseResult`（用例 + 观测 + 判分）以及
    :class:`EvalReport`（整轮汇总，内含 :class:`EvalMetrics`）。

    这三个层次刻意分开，而不是揉成一个大对象，理由与 ``src/domain`` 把
    「业务事实」与「UI 状态」分开是同一条：**报告要能跨版本 diff**。
    想知道「新版本哪条用例从通过变成失败」时，需要的是「同一份用例、
    两份观测、两份判分」的对比结构；把它们压成一个对象，diff 出来的是
    一整段混杂文本，看不出变化点在哪。

上下游依赖：
    - 上游：:mod:`src.domain.enums`（车道 / 意图枚举，用例的期望值用它表达）、
      仅标准库 ``dataclasses``。
    - 下游：``src/evaluation/judge.py``（产出 :class:`JudgeVerdict`）、
      ``src/evaluation/runner.py``（产出 :class:`Observation` 与
      :class:`EvalReport`）、``scripts/eval.py``（落盘 :meth:`EvalReport.to_dict`）。

═══ ⚠️ 本模块是**纯数据**：不允许出现时间戳与随机数 ═══

    报告里一旦有 ``生成时间``，它就不再是「同一份输入得到同一份输出」——
    两次跑同一份数据集会得到两份字节不同的 JSON，于是：

      · 无法做「只比结论、不比噪声」的 diff；
      · 单测只能断言几个字段，不能断言整个报告；
      · CI 里的「报告没变」这条最便宜的回归信号彻底失效。

    需要「什么时候跑的」这个信息时，让它留在**调用方**（``scripts/eval.py``
    可以在打印时补一行），而不是写进数据契约里。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain.enums import Intent, LaneName

# ==============================================================================
# 数值归一：所有比率都保留 4 位小数
# ==============================================================================
#: 比率的保留位数。
#:
#: ⚠️ 之所以要**统一取整**而不是直接写浮点：``1/3`` 在 JSON 里是
#: ``0.3333333333333333``，两次计算（顺序不同、分母不同）会得到最后几位
#: 不一致的字符串。报告是要进 git、要被 diff 的产物，末位的浮点噪声会把
#: 「结论变了」淹没在「第 16 位小数变了」里。4 位对评测结论绰绰有余。
_RATIO_DIGITS = 4


def _ratio(numerator: int, denominator: int) -> float:
    """算一个安全比率。

    Args:
        numerator (`int`): 分子。
        denominator (`int`): 分母。

    Returns:
        `float`: ``numerator / denominator``，保留 :data:`_RATIO_DIGITS` 位；
        分母为 0 时返回 ``0.0``。

    ⚠️ 分母为 0 返回 0.0 而不是抛异常，也**不是** 1.0：一份没有任何用例的
    数据集，其「车道准确率」既不是满分的证据、也不是失败的证据 —— 它是
    「没有证据」。返回 0.0 会让它在综合得分里拉低分数、从而被注意到；
    返回 1.0 则会让一个**空数据集**看起来完美通过，这是最坏的一种假绿。
    """
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, _RATIO_DIGITS)


def _set_f1(expected: frozenset[str], actual: frozenset[str]) -> float:
    """集合级 F1（用于意图集、工具集这类「无序集合」的吻合度）。

    Args:
        expected (`frozenset[str]`): 期望集合。
        actual (`frozenset[str]`): 实际集合。

    Returns:
        `float`: F1，保留 :data:`_RATIO_DIGITS` 位。

    ⚠️ 用 **F1 而不是「是否相等」**：多意图输入与多工具调用是常态，
    一个「几乎全对、只多了一个」的实现与一个「全错」的实现在布尔判定下
    都是 0 分，而我们恰恰需要区分这两种 —— 前者是微调问题，
    后者是接错了线。F1 把这条距离显式化了。

    ⚠️ 两边都空时返回 1.0：这是「期望为无、实际也为无」的正确吻合。
    但**只有调用方确认了「这条用例确实声明了这一维度」时才该调它** ——
    空期望常常意味着「该维度不适用」（比如离线时模型识别不了意图），
    那种情况应当**整条跳过**，而不是判成满分。
    """
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    tp = len(expected & actual)
    if tp == 0:
        return 0.0
    precision = tp / len(actual)
    recall = tp / len(expected)
    return round(2 * precision * recall / (precision + recall), _RATIO_DIGITS)


# ==============================================================================
# 一、黄金用例
# ==============================================================================
@dataclass(frozen=True)
class EvalCase:
    """一条**黄金用例**：一句输入，加一组「正确的系统应当满足」的期望。

    期望是**逐维度可选**的，而且「没给」与「给了空」是两件事：

        · ``expected_lane=None``    ⇒ 这条用例**不检查**车道；
        · ``expected_intents=()``   ⇒ 这条用例**不检查**意图；
        · ``expected_tools=()``     ⇒ 这条用例**不检查**工具集。

    ⚠️ 这个「可选」是离线评测能成立的关键，不是偷懒。没有模型密钥时
    （CI 与本机常态），慢车道的意图只能由 :class:`~src.llm.mock.MockChatModel`
    给出，而它**识别不出真实意图**（本地确定性生成，见 :mod:`src.llm.mock`）。
    若强行把 ``expected_intents`` 填上真实模型应有的值，离线跑出来的意图
    指标就会恒为 0 —— 那个数字反映的是「没有密钥」，不是「系统变差了」。
    把它留空，该用例就只检查**离线也确定可判**的维度（车道）。

    Attributes:
        id: 用例标识。**全局唯一**，报告与 diff 都按它对齐。
        input: 送给系统的原始用户输入（一句话）。
        expected_lane: 期望车道；``None`` 表示该用例不检查车道。
        expected_intents: 期望意图集（无序号，按集合比较）；空表示不检查。
        expected_tools: 期望被调度到的工具 / 子智能体名集合；空表示不检查。
        tags: 分类标签（如 ``fast_lane`` / ``question``），供分组统计与筛选。
        notes: 给读报告的人看的一句话说明（为什么这条用例在这里）。
    """

    id: str
    input: str
    expected_lane: LaneName | None = None
    expected_intents: tuple[Intent, ...] = ()
    expected_tools: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """序列化成**可 JSON 化**的字典。

        Returns:
            `dict[str, Any]`: 枚举转成其 ``value`` 字符串，元组转成列表。
        """
        return {
            "id": self.id,
            "input": self.input,
            "expected_lane": None if self.expected_lane is None else self.expected_lane.value,
            "expected_intents": [intent.value for intent in self.expected_intents],
            "expected_tools": list(self.expected_tools),
            "tags": list(self.tags),
            "notes": self.notes,
        }


# ==============================================================================
# 二、系统观测与判分
# ==============================================================================
# ==============================================================================
# 一之二、维度名与「测不到」这件事
# ==============================================================================
#: 维度名常量。用常量而不是裸字符串：它们要跨 ``types`` / ``runner`` / ``judge``
#: 三个模块使用，拼错一个字母不会报错，只会让那个维度的「测不到」标记
#: **静默失效** —— 又回到「拿 0 分冒充没测」的老路上。
DIM_INTENTS = "intents"
DIM_TOOLS = "tools"

# ==============================================================================
# ⚠️⚠️ 为什么观测里要有 ``unmeasured`` —— 本项目评测最要紧的一条设计
# ==============================================================================
# 「这一维度没测到」与「这一维度测了、结果是 0 分」在报告里看起来一模一样
# （都是一个 0），但它们是**完全相反**的两件事：
#
#     · 测了、得 0 分 ⇒ 系统坏了，要有人去修；
#     · 没测到       ⇒ 这个数字不存在，不该出现在任何分母里。
#
# 把后者记成 0 有两个后果，两个都很坏：
#
#   1. **分数失真。** 离线评测（无模型密钥）时慢车道的意图由一个确定性
#      Mock 模型给出，它本来就识别不出真实意图。把这 18 条记成 0 分，
#      综合得分被拖到 0.58 —— 而那个数字反映的是「本机没有密钥」，
#      不是「系统变差了」。它会让一次本来正常的改动看起来是回归。
#
#   2. **掩盖真正的坏。** 一旦「没测到」也记 0，就没有任何信号能区分
#      「这个维度全错」和「这个维度根本没跑」。CI 上稳定的 0.58 会让
#      所有人学会无视这个分数，于是**真正的** 0 分被一起无视了。
#
# 所以本模块让被测系统**显式声明**它测不到什么，而不是让评测器去猜。
# 猜的办法（比如「有密钥就评测」）会把评测器和被测系统的内部实现绑死。
#
# ⚠️ 反向的风险也必须堵住：有了这个字段，「什么都声明成没测到」就能
# 得到一个空报告并顺利通关。所以 :attr:`EvalReport.overall_score` 在
# **一个维度都没测到**时返回 0.0（见那边的说明），且报告会打印被跳过的
# 条数 —— 跳过必须是**看得见**的。
# ==============================================================================


@dataclass(frozen=True)
class Observation:
    """**系统对一条用例的实际观测** —— 判分与指标的唯一事实来源。

    ⚠️ 字段一律用**字符串**而不是枚举：观测值可能来自一个连枚举都没认出来的
    实现（比如车道为空串、意图是模型编出来的新词）。用枚举类型接，会在
    「系统给出了一个不在词表里的答案」时抛 ValidationError 而不是如实记录
    —— 而那种情况恰恰是最需要被看见的（它意味着系统跑偏了），
    不该被一个构造期的异常吞掉。

    Attributes:
        lane: 实际判定的车道（``FAST`` / ``SLOW``；出错时为空串）。
        intents: 实际识别出的意图值集合（无序）。
        tools: 实际被调度到的工具 / 子智能体名集合。
        output: 给 LLM 判分用的自然语言输出；规则判分下可为空。
        unmeasured: 本次观测里**根本没测到**的维度名（``"intents"`` /
            ``"tools"``）。见下面那段说明 —— 这个字段是整个评测**诚实性**
            的支点。
    """

    lane: str = ""
    intents: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    output: str = ""
    unmeasured: tuple[str, ...] = ()

    def measures(self, dimension: str) -> bool:
        """本次观测是否**真的测到**了某个维度。

        Args:
            dimension (`str`): ``"intents"`` 或 ``"tools"``。

        Returns:
            `bool`: 测到了为 True。
        """
        return dimension not in self.unmeasured

    def to_dict(self) -> dict[str, Any]:
        """序列化成可 JSON 化的字典。

        Returns:
            `dict[str, Any]`: 元组转列表。
        """
        return {
            "lane": self.lane,
            "intents": list(self.intents),
            "tools": list(self.tools),
            "output": self.output,
            "unmeasured": list(self.unmeasured),
        }


@dataclass(frozen=True)
class JudgeCheck:
    """判分时的一项**可核对**的检查（供报告逐条展示）。

    Attributes:
        name: 检查项名（如「车道」「意图集」「工具集」）。
        score: 该维度的得分，``0.0``～``1.0``。
        detail: 一行说明（期望什么、实际是什么）。
    """

    name: str
    score: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """序列化成可 JSON 化的字典。

        Returns:
            `dict[str, Any]`: 扁平字典。
        """
        return {"name": self.name, "score": self.score, "detail": self.detail}


@dataclass(frozen=True)
class JudgeVerdict:
    """一条用例的判分结论。

    ⚠️ :attr:`mode` 是本项目**诚实性**的落脚点。离线时我们用规则判分，
    此时 :attr:`mode` 必须是 ``"rule"``，报告里也要显式写明「本次不是 LLM
    判分」。让一个规则判分**冒充** LLM 判分（把 mode 写成 ``"llm"``），
    会让报告里那个分数带上它不配有的可信度 —— 读报告的人会据此以为
    「语义质量被评估过了」，而实际上只比了几个集合是否相等。

    Attributes:
        score: 综合得分，``0.0``～``1.0``。
        mode: 判分模式，``"rule"`` 或 ``"llm"``。
        rationale: 一句话依据（面向人；规则判分下逐项列出）。
        checks: 逐项检查明细（LLM 判分下可为空）。
    """

    score: float
    mode: str
    rationale: str = ""
    checks: tuple[JudgeCheck, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """序列化成可 JSON 化的字典。

        Returns:
            `dict[str, Any]`: 含 ``checks`` 的嵌套结构。
        """
        return {
            "score": self.score,
            "mode": self.mode,
            "rationale": self.rationale,
            "checks": [check.to_dict() for check in self.checks],
        }


# ==============================================================================
# 三、单条结果
# ==============================================================================
@dataclass(frozen=True)
class CaseResult:
    """一条用例的完整结果：用例本身 + 实际观测 + 判分。

    ⚠️ 把整个 :class:`EvalCase` 带在结果里，而不是只带 ``case_id``：
    :func:`src.evaluation.runner.compute_metrics` 需要**期望值**才能算
    准确率/召回率，而期望值只存在于用例里。若这里只留 id，聚合函数就得
    再持有一份「id → 用例」的映射，那份映射迟早会与结果列表不同步
    （筛选、截断、去重都会打破它），症状是「指标算的是另一批用例」。

    Attributes:
        case: 对应的黄金用例。
        observation: 系统给出的实际观测。
        verdict: 判分结论。
        error: 观测阶段捕获到的异常摘要；正常时为空串。
    """

    case: EvalCase
    observation: Observation
    verdict: JudgeVerdict
    error: str = ""

    @property
    def case_id(self) -> str:
        """用例标识（便捷访问）。

        Returns:
            `str`: :attr:`EvalCase.id`。
        """
        return self.case.id

    def to_dict(self) -> dict[str, Any]:
        """序列化成可 JSON 化的字典。

        Returns:
            `dict[str, Any]`: 用例 / 观测 / 判分三段。
        """
        return {
            "id": self.case.id,
            "case": self.case.to_dict(),
            "observation": self.observation.to_dict(),
            "verdict": self.verdict.to_dict(),
            "error": self.error,
        }


# ==============================================================================
# 四、聚合指标
# ==============================================================================
@dataclass(frozen=True)
class EvalMetrics:
    """整轮评测的**聚合指标**。

    ⚠️ 每个比率都同时带着它的**分母**（``*_evaluated``）。只报一个
    ``0.5`` 是没有信息量的：它是「2 条里对了 1 条」还是「100 条里对了 50 条」？
    前者说明数据集太小、结论不可信，后者说明系统真的差。把分母写进结构里，
    报告就能打印 ``0.5000 (1/2)`` —— 一个能让人立刻判断「这个数字值不值得当真」的形态。

    Attributes:
        total: 用例总数。
        lane_evaluated: 声明了 ``expected_lane`` 的用例数。
        lane_correct: 其中车道判定正确的条数。
        lane_accuracy: 车道准确率。
        intent_evaluated: 声明了 ``expected_intents`` 的用例数。
        intent_tp / intent_fp / intent_fn: 意图集的微平均 TP / FP / FN。
        intent_precision / intent_recall / intent_f1: 意图微平均 P / R / F1。
        tool_evaluated: 声明了 ``expected_tools`` 的用例数。
        tool_exact_match: 其中工具集**完全相等**的条数。
        tool_exact_match_rate: 工具集完全匹配率。
        judge_mean_score: 全部用例判分得分的均值。
        judge_evaluated: **判分真的有依据可依**的用例数，即
            :class:`~src.evaluation.judge.RuleJudge` 至少产出了一项检查
            （或判分器是大模型判分，见下）。
        intent_skipped / tool_skipped: 因「本次观测没测到这个维度」
            （见 :attr:`Observation.unmeasured`）而被排除出分母的用例数。

    ⚠️ :attr:`judge_evaluated` 与 :attr:`total` 是**两个不同的数**，
    而它们的差就是「判分走过场」的规模。规则判分下，一条没声明任何期望项
    （或声明的那些维度恰好都没测到）的用例会得到「无可判分内容 ⇒ 1.0」，
    它计入 ``total`` 却计入不了 ``judge_evaluated``。若综合得分只看
    ``total > 0``，那么一份**每条用例都无可判分**的数据集会因为一堆
    白送的 1.0 而拿到满分 —— 这正是本模块要堵的那个洞。
    """

    total: int = 0
    lane_evaluated: int = 0
    lane_correct: int = 0
    lane_accuracy: float = 0.0
    intent_evaluated: int = 0
    intent_tp: int = 0
    intent_fp: int = 0
    intent_fn: int = 0
    intent_precision: float = 0.0
    intent_recall: float = 0.0
    intent_f1: float = 0.0
    tool_evaluated: int = 0
    tool_exact_match: int = 0
    tool_exact_match_rate: float = 0.0
    judge_mean_score: float = 0.0
    judge_evaluated: int = 0
    intent_skipped: int = 0
    tool_skipped: int = 0

    def to_dict(self) -> dict[str, Any]:
        """序列化成可 JSON 化的字典。

        Returns:
            `dict[str, Any]`: 扁平字典。
        """
        return {
            "total": self.total,
            "lane_accuracy": self.lane_accuracy,
            "lane_correct": self.lane_correct,
            "lane_evaluated": self.lane_evaluated,
            "intent_precision": self.intent_precision,
            "intent_recall": self.intent_recall,
            "intent_f1": self.intent_f1,
            "intent_tp": self.intent_tp,
            "intent_fp": self.intent_fp,
            "intent_fn": self.intent_fn,
            "intent_evaluated": self.intent_evaluated,
            "intent_skipped": self.intent_skipped,
            "tool_exact_match_rate": self.tool_exact_match_rate,
            "tool_exact_match": self.tool_exact_match,
            "tool_evaluated": self.tool_evaluated,
            "tool_skipped": self.tool_skipped,
            "judge_mean_score": self.judge_mean_score,
            "judge_evaluated": self.judge_evaluated,
        }


# ==============================================================================
# 五、汇总报告
# ==============================================================================
@dataclass(frozen=True)
class EvalReport:
    """一整轮评测的结论：逐条结果 + 聚合指标 + 元信息。

    Attributes:
        dataset: 数据集来源（通常是相对仓库根或绝对的路径字符串）。
        judge_mode: 本轮使用的判分模式（``"rule"`` / ``"llm"``）。
        results: 逐条结果，顺序与数据集一致。
        metrics: 聚合指标。
        notes: 报告级备注（例如「未配置模型密钥，已降级为规则判分」）。
    """

    dataset: str
    judge_mode: str
    results: tuple[CaseResult, ...] = ()
    metrics: EvalMetrics = field(default_factory=EvalMetrics)
    notes: tuple[str, ...] = ()

    def failures(self) -> tuple[CaseResult, ...]:
        """返回**没有满分**的用例（用于报告尾部列要点）。

        Returns:
            `tuple[CaseResult, ...]`: 得分小于 1.0、或观测阶段出错的用例。
        """
        return tuple(
            result
            for result in self.results
            if result.error or result.verdict.score < 1.0
        )

    @property
    def overall_score(self) -> float:
        """综合得分 —— CLI 的退出判据。

        取值口径：对**确实被评测过**的维度求算术平均。一个维度只有在它的
        分母大于 0 时才计入，因此：

            · 没有任何用例的数据集 ⇒ 全部维度被跳过 ⇒ 综合得分 0.0
              （见 :func:`_ratio` 的说明：空数据集不该显示成满分）；
            · 只声明了车道的离线数据集 ⇒ 只把车道准确率与判分均分计入。

        ⚠️ 判分均分只在 :attr:`EvalMetrics.judge_evaluated` 大于 0 时计入，
        而**不是** ``total`` 大于 0 时。这两个条件看起来只差一点，实际
        差的是整个评测有没有意义：

            规则判分下，一条没声明任何期望项的用例拿到的是「无可判分内容
            ⇒ 1.0」（见 :class:`~src.evaluation.judge.RuleJudge`）。于是
            一份**每条用例都没有期望**的数据集，``total`` 是 34、
            ``judge_mean_score`` 是白送的 1.0，而它一个字都没检查过。
            用 ``total > 0`` 当条件，这份数据集会得到综合得分 1.0000、
            退出码 0、报告一片绿 —— **一次什么都没评的评测，看起来
            完美通过**。这正是本项目最需要避免的那种失败。

            ``judge_evaluated`` 数的是「判分真的有依据可依」的用例数，
            上面那份数据集在这里是 0，于是判分均分不参与、四个维度全空、
            综合得分 0.0、退出码 1。

        ⚠️ 判分均分计入时它覆盖全部用例，会把「车道对了但意图/工具不对」
        的那部分损失也带进综合得分 —— 否则一个「只把车道判对、其余全错」
        的实现也能拿到高分。

        Returns:
            `float`: 综合得分，保留 :data:`_RATIO_DIGITS` 位。
        """
        parts: list[float] = []
        if self.metrics.lane_evaluated > 0:
            parts.append(self.metrics.lane_accuracy)
        if self.metrics.intent_evaluated > 0:
            parts.append(self.metrics.intent_f1)
        if self.metrics.tool_evaluated > 0:
            parts.append(self.metrics.tool_exact_match_rate)
        if self.metrics.judge_evaluated > 0:
            parts.append(self.metrics.judge_mean_score)
        if not parts:
            return 0.0
        return round(sum(parts) / len(parts), _RATIO_DIGITS)

    def to_dict(self) -> dict[str, Any]:
        """序列化成可 JSON 化的字典（``scripts/eval.py`` 落盘用）。

        Returns:
            `dict[str, Any]`: 含 ``metrics`` / ``results`` / ``notes`` 的嵌套结构。
        """
        return {
            "dataset": self.dataset,
            "judge_mode": self.judge_mode,
            "notes": list(self.notes),
            "overall_score": self.overall_score,
            "metrics": self.metrics.to_dict(),
            "results": [result.to_dict() for result in self.results],
        }


__all__ = [
    "DIM_INTENTS",
    "DIM_TOOLS",
    "CaseResult",
    "EvalCase",
    "EvalMetrics",
    "EvalReport",
    "JudgeCheck",
    "JudgeVerdict",
    "Observation",
]
