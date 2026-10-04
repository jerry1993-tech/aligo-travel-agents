# -*- coding: utf-8 -*-
"""**评测执行器** —— 跑数据集、算指标、产出报告。

文件职责：
    把「数据集 / 被测系统 / 判分器」三样东西撮合成一份
    :class:`~src.evaluation.types.EvalReport`：

        1. :func:`load_dataset` —— 从 ``.json`` **或** ``.yaml`` 读出用例；
        2. :class:`PipelineSystem` —— 默认的被测系统：跑真实的编排链路
           （``classify`` → 慢车道则意图识别）；
        3. :func:`compute_metrics` —— 聚合车道准确率、意图 P/R/F1、
           工具完全匹配率、判分均分；
        4. :func:`run_dataset` —— 逐条执行 + 判分 + 汇总。

上下游依赖：
    - 上游：:mod:`src.orchestration.classifier`（车道判定与意图→智能体映射）、
      :mod:`src.agents.intent`（慢车道意图识别）、:mod:`src.llm.factory`
      （模型装配）、:mod:`src.evaluation.judge`、:mod:`src.evaluation.types`。
    - 下游：``scripts/eval.py``。

═══ ⚠️ 被测系统是**可注入**的，而且这是刻意的 ═══

    :func:`run_dataset` 接受任意满足 :class:`SystemUnderTest` 协议的对象，
    而不是把 :class:`PipelineSystem` 写死在里面。理由有二：

      · **可测**：单测能用一个「输出完全可控」的假系统去验证指标算术，
        从而证明「指标不是恒为 0 的摆设」—— 这是本模块最重要的一条测试
        （见 ``tests/test_evaluation_runner.py`` 的变异用例）。
      · **可换**：将来要评「完整 Agent 图」而不是「编排决策」时，
        只要换一个实现同一协议的类，指标与报告一行都不用改。

═══ ⚠️ 数据集加载器同时支持 ``.json`` 与 ``.yaml`` ═══

    真实的数据集文件（``tests/evaluation/golden_dataset.*``）由另一个进程
    （可能是并行的另一个协作者）产出，其格式在编写本模块时尚未定稿。
    因此加载器：

      · 按 ``golden_dataset.json`` → ``golden_dataset.yaml`` →
        ``golden_dataset.yml`` 的顺序找**第一个存在**的文件，两种格式都认；
      · 顶层既接受「用例列表」，也接受 ``{"cases": [...]}``；
      · 每个用例的字段名接受若干别名（``text``/``query`` → ``input`` 等），
        但**看不懂的值一律报错**而不是静默跳过 —— 静默跳过会让数据集里
        一条写错的用例变成一个「悄悄没被测到」的空洞。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Protocol

import yaml

from src.agents.intent import build_intent_recognizer
from src.config import Settings
from src.domain.enums import Intent, LaneName
from src.evaluation.judge import Judge, judge_mode
from src.evaluation.types import (
    DIM_INTENTS,
    DIM_TOOLS,
    CaseResult,
    EvalCase,
    EvalMetrics,
    EvalReport,
    Observation,
    _ratio,
)
from src.llm.factory import build_chat_model
from src.llm.mock import MockChatModel
from src.orchestration.classifier import classify, target_agents_for
from src.orchestration.lane import DEFAULT_ROUTE_TOOL

logger = logging.getLogger(__name__)

#: 数据集候选文件名（按优先级）。见模块文档「同时支持 .json 与 .yaml」。
#:
#: ⚠️⚠️ **顺序是 YAML 在前**，这与「按字母序」或「JSON 更常见」的直觉相反，
#: 但它是一条踩过的坑留下的：
#:
#: 本仓库的黄金数据集是 ``tests/evaluation/golden_dataset.yaml``
#: （选它的理由写在那个文件头部 —— JSON 写不了注释，而这份数据集每一条
#: 都带着「防哪个失败模式」的说明）。曾经的顺序把 ``.json`` 排在前面，
#: 于是目录里一份早先的草稿 ``golden_dataset.json`` 一直**盖着**它：
#: ``scripts/eval.py`` 的默认路径正好是这个目录，``find_dataset`` 每次
#: 都返回那份 12 条的旧稿，而它**能正常加载**（不是坏文件），
#: 所以没有任何报错 —— 评测跑了，报告出了，只是评的不是你以为的那份数据集。
#:
#: 教训是：候选顺序把「哪份数据集是权威的」这件事编码进了一个常量，
#: 而它必须与「仓库里实际放了哪份」一致。把权威的那份排前面，
#: 残留的草稿就只能被忽略，而不会静默取代它。
DATASET_CANDIDATES: tuple[str, ...] = (
    "golden_dataset.yaml",
    "golden_dataset.yml",
    "golden_dataset.json",
)

#: ``input`` 字段可接受的别名。
_INPUT_ALIASES: tuple[str, ...] = ("input", "text", "query", "utterance")

#: ``expected_lane`` 字段可接受的别名。
_LANE_ALIASES: tuple[str, ...] = ("expected_lane", "lane", "expectedLane")

#: ``expected_intents`` 字段可接受的别名。
_INTENT_ALIASES: tuple[str, ...] = ("expected_intents", "intents", "expectedIntents")

#: ``expected_tools`` 字段可接受的别名。
_TOOL_ALIASES: tuple[str, ...] = ("expected_tools", "tools", "expected_agents", "agents")


# ==============================================================================
# 一、被测系统
# ==============================================================================
class SystemUnderTest(Protocol):
    """被测系统协议 —— 一条用例进，一个 :class:`Observation` 出。"""

    async def observe(self, case: EvalCase) -> Observation:
        """对一条用例产出实际观测。

        Args:
            case (`EvalCase`): 黄金用例。

        Returns:
            `Observation`: 系统的实际表现。
        """
        ...


class PipelineSystem:
    """默认被测系统：**真实的编排链路**。

    它跑的是线上同一条路：

        1. :func:`src.orchestration.classifier.classify` —— 判快慢车道；
        2. 快车道 ⇒ 意图与目标智能体直接来自规则命中（不调用模型）；
        3. 慢车道 ⇒ 调 :class:`~src.agents.intent.IntentRecognizer` 拿意图，
           再用 :func:`~src.orchestration.classifier.target_agents_for`
           把主意图映射成子智能体。

    ⚠️ 它**不启动 HTTP 服务、不建数据库**：评测的对象是「编排决策」
    （车道 / 意图 / 调度目标），这三样在装配任何存储之前就已确定。
    把评测绑在完整服务上，会让一次评测需要拉起 Milvus 与 Postgres ——
    而它们与这三样决策毫无关系，只会让 CI 变慢、让失败原因变模糊。

    Attributes:
        _settings (`Settings`): 配置。
        _recognizer (`IntentRecognizer`): 慢车道用的意图识别器。
        _measurable (`bool`): 慢车道的意图维度**是否可测**。判据是
            「跑的是不是一个真的模型」—— MockLLM 的回复是本地确定性生成的，
            它给不出真实意图，此时把慢车道意图计入分母就是在评「有没有密钥」。
    """

    def __init__(
        self,
        settings: Settings,
        *,
        model: Any = None,
    ) -> None:
        """初始化。

        Args:
            settings (`Settings`): 配置。
            model (`Any | None`): 聊天模型；``None`` 时按配置构造。零密钥时
                由 :func:`~src.llm.factory.build_chat_model` 自动降级为
                :class:`~src.llm.mock.MockChatModel`。可注入，便于测试。
        """
        self._settings = settings
        orchestrator = settings.orchestration
        resolved = model if model is not None else build_chat_model(settings)
        # ⚠️ 用 ``isinstance`` 认 Mock，而不是问 ``should_use_mock(settings)``：
        # 后者只看配置里的密钥，而 ``model`` 允许被显式注入 —— 一个注入了
        # 真模型、但配置里恰好没密钥的调用方，两者会给出相反的答案，
        # 而这样的评测会把可测的维度也一起标成「没测到」，
        # 于是一个真实的 0 分被伪装成「没测」。
        self._measurable = not isinstance(resolved, MockChatModel)
        self._recognizer = build_intent_recognizer(
            model=resolved,
            max_intents=orchestrator.max_subagent_calls,
            confidence_threshold=orchestrator.intent_confidence_threshold,
        )

    async def observe(self, case: EvalCase) -> Observation:
        """跑一遍编排链路并记录观测。

        Args:
            case (`EvalCase`): 黄金用例。

        Returns:
            `Observation`: 车道 / 意图集 / 目标智能体集 / 一句话输出。

        ⚠️ 快车道分支**不调用模型**（这正是快车道存在的意义，见
        :mod:`src.orchestration.classifier`）。因此离线评测里，快车道用例的
        意图指标是**真实**的 —— 它只依赖确定性的规则表，不依赖模型。

        ⚠️⚠️ 而慢车道的意图与工具**测不到**，必须在观测里如实声明
        （``unmeasured``）。这是本评测器最容易出错的地方，两种写错的方式
        都会让报告说谎：

          · 离线（MockLLM）时慢车道的意图来自一个确定性 Mock 模型，
            它**识别不出真实意图**（本方法下方那条
            「意图识别没有得到结构化结果」的 WARNING 就是它）。
            把这些用例按 0 分判，综合得分反映的是「本机没有密钥」，
            不是「系统变差了」；
          · 慢车道的工具**任何模式下都测不到**：本方法只做到「调度哪个
            子智能体」这一步，真正调用业务工具发生在子智能体的 ReAct
            循环里，而那需要一个真实模型 + 一套完整工具链。拿
            ``target_agents_for()`` 的返回值（**智能体名**）去顶替
            工具名，会让「工具集完全匹配率」恒为 0 —— 一个看起来像
            「系统工具选择全错」的数字，实际是「这个评测器根本没在看工具」。

        ⚠️ 快车道的工具**恰好相反**，是确定性可测的：规则命中后由
        :class:`~src.orchestration.lane.LaneRouterMiddleware` 合成一次对
        :data:`~src.tools.route.ROUTE_TOOL_NAME` 的调用（见
        :data:`~src.orchestration.lane.DEFAULT_ROUTE_TOOL`）。所以这里记
        ``aligo_route_intent``，而**不是** ``decision.target_agents`` ——
        后者是「接下来该交给谁」，不是「本轮调用了什么工具」，两者在
        黄金数据集里是两个不同的字段。
        """
        decision = classify(
            case.input,
            enabled=self._settings.orchestration.fast_lane_enabled,
            max_chars=self._settings.orchestration.fast_lane_max_chars,
        )

        if decision.lane is LaneName.FAST:
            return Observation(
                lane=decision.lane.value,
                intents=(decision.intent.value,),
                tools=(DEFAULT_ROUTE_TOOL,),
                output=decision.reason,
            )

        # ---- 慢车道：调意图识别 ----
        result = await self._recognizer.recognize(case.input)

        # ⚠️ 慢车道的工具**任何模式下**都测不到（见方法文档）。
        unmeasured: tuple[str, ...] = (DIM_TOOLS,)
        if not self._measurable:
            # 离线（MockLLM）：意图同样测不到。
            unmeasured = (DIM_INTENTS, DIM_TOOLS)

        intents = tuple(item.intent.value for item in result.intents)
        if result.needs_clarification:
            # ⚠️ 追问时不调度任何子智能体：这时调度等于「替用户猜」，
            # 而猜错的代价是整条链路白跑（见 classifier 的说明）。
            return Observation(
                lane=decision.lane.value,
                intents=intents,
                tools=(),
                unmeasured=unmeasured,
                output=result.clarification_question or result.reasoning,
            )

        top = result.top_intent()
        tools = tuple(target_agents_for(top)) if top is not None else ()
        return Observation(
            lane=decision.lane.value,
            intents=intents,
            tools=tools,
            unmeasured=unmeasured,
            output=result.reasoning or result.rewritten_query,
        )


# ==============================================================================
# 二、数据集加载
# ==============================================================================
def find_dataset(directory: str | Path) -> Path:
    """在目录里按优先级找数据集文件。

    Args:
        directory (`str | Path`): 待查找目录。

    Returns:
        `Path`: 找到的数据集路径。

    Raises:
        FileNotFoundError: 三个候选名一个都不存在时。异常里列出找过哪些
            文件名 —— 「文件不见了」这个问题，排障时第一件事就是确认
            脚本找的路径与你放的位置是不是同一个。
    """
    base = Path(directory)
    for name in DATASET_CANDIDATES:
        candidate = base / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"在 {base} 下没有找到数据集文件（找过：{', '.join(DATASET_CANDIDATES)}）。\n"
        f"   · 数据集的默认位置是 tests/evaluation/；\n"
        f"   · 也可以用 --dataset 显式指定一个 .json / .yaml 文件。",
    )


def _decode(path: Path) -> Any:
    """按后缀把文件读成 Python 对象。

    Args:
        path (`Path`): 文件路径。

    Returns:
        `Any`: 解析结果（list 或 dict）。

    Raises:
        ValueError: 后缀不认识。
    """
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(text)
    if suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    raise ValueError(
        f"不认识的数据集后缀 {suffix!r}：只支持 .json / .yaml / .yml（文件：{path}）。",
    )


def _pick(entry: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    """从条目里按别名取第一个存在的字段值。

    Args:
        entry (`dict`): 一条用例的原始字典。
        aliases (`tuple[str, ...]`): 候选字段名。

    Returns:
        `Any`: 命中的值；一个都没有时返回 ``None``。
    """
    for name in aliases:
        if name in entry:
            return entry[name]
    return None


#: 期望值所在的**嵌套**字段名。
#:
#: ⚠️ 本仓库的黄金数据集把三个期望维度收在一个 ``expected:`` 映射里
#: （``expected: {lane, intents, tools}``），而不是平铺在用例顶层 ——
#: 这是 ``tests/evaluation/golden_dataset.yaml`` 的字段说明里写死的
#: 契约，``tests/test_evaluation_dataset.py`` 也是按这个形状校验的。
_EXPECTED_KEY = "expected"


def _pick_expected(entry: dict[str, Any], aliases: tuple[str, ...]) -> Any:
    """按别名在**顶层或 ``expected`` 嵌套块**里取期望值。

    ⚠️⚠️ 这个函数存在的唯一理由是：本仓库的数据集是嵌套的，而本模块
    原本只认平铺字段。少了它，``load_dataset`` 会把每一条用例的
    ``expected`` 整块**静默丢弃** —— 每条用例都变成「没声明任何期望」，
    于是 :class:`~src.evaluation.judge.RuleJudge` 对每条都判「无可判分内容」
    并给出 1.0，最后综合得分 1.0000、退出码 0。

    也就是说：**一份一个字都没被检查的数据集会显示成满分。**
    这个错误的可怕之处在于它看起来完全正常 —— 报告有数、有分、
    有阈值比对，只有 ``(0/0)`` 那些分母能透露真相，而没人会盯着看。

    ⚠️ 顶层优先于嵌套：两者同时出现时以更「贴身」的那个为准，
    这与 :func:`_pick` 的取舍一致（靠前的别名优先）。

    Args:
        entry (`dict`): 一条用例的原始字典。
        aliases (`tuple[str, ...]`): 候选字段名。

    Returns:
        `Any`: 命中的值；两处都没有时返回 ``None``。
    """
    found = _pick(entry, aliases)
    if found is not None:
        return found
    nested = entry.get(_EXPECTED_KEY)
    if isinstance(nested, dict):
        return _pick(nested, aliases)
    return None


def _parse_lane(value: Any, case_id: str) -> LaneName | None:
    """把原始值解析成 :class:`LaneName`。

    Args:
        value (`Any`): 原始值（字符串或 ``None``）。
        case_id (`str`): 用例 id（报错时回显）。

    Returns:
        `LaneName | None`: 解析结果；``None`` / 空串表示「不检查车道」。

    Raises:
        ValueError: 值不是合法的车道名。
    """
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    token = str(value).strip().upper()
    try:
        return LaneName(token)
    except ValueError as exc:
        valid = ", ".join(item.value for item in LaneName)
        raise ValueError(
            f"用例 {case_id!r} 的 expected_lane={value!r} 不是合法车道；"
            f"合法取值：{valid}。",
        ) from exc


def _parse_intents(value: Any, case_id: str) -> tuple[Intent, ...]:
    """把原始值解析成意图元组。

    Args:
        value (`Any`): 原始值（列表 / 字符串 / ``None``）。
        case_id (`str`): 用例 id。

    Returns:
        `tuple[Intent, ...]`: 解析结果；空表示「不检查意图」。

    Raises:
        ValueError: 出现不合法的意图名。
    """
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    parsed: list[Intent] = []
    for item in value:
        token = str(item).strip().upper()
        if not token:
            continue
        try:
            parsed.append(Intent(token))
        except ValueError as exc:
            valid = ", ".join(intent.value for intent in Intent)
            raise ValueError(
                f"用例 {case_id!r} 的 expected_intents 里 {item!r} 不是合法意图；"
                f"合法取值：{valid}。",
            ) from exc
    return tuple(parsed)


def _parse_case(entry: dict[str, Any], index: int) -> EvalCase:
    """把一条原始字典解析成 :class:`EvalCase`。

    Args:
        entry (`dict`): 原始条目。
        index (`int`): 在列表中的下标（缺 id 时用它拼一个）。

    Returns:
        `EvalCase`: 解析好的用例。

    Raises:
        ValueError: 条目不是字典、或缺少 ``input`` 时。
    """
    if not isinstance(entry, dict):
        raise ValueError(f"数据集第 {index} 条不是「键: 值」字典，实际是 {type(entry).__name__}。")

    raw_id = entry.get("id")
    case_id = str(raw_id).strip() if raw_id is not None else ""
    if not case_id:
        # ⚠️ 没有 id 就**不生成**一个随机/序号 id：报告要能跨版本 diff，
        # 而一个「按位置生成」的 id 会在用例插入/删除后整体错位，
        # diff 出一堆假变化。这里显式报错，逼数据集作者给一个稳定的标识。
        raise ValueError(f"数据集第 {index} 条缺少稳定 id（id 是报告 diff 的对齐键）。")

    raw_input = _pick(entry, _INPUT_ALIASES)
    if raw_input is None:
        raise ValueError(
            f"用例 {case_id!r} 缺少输入字段（可接受：{', '.join(_INPUT_ALIASES)}）。",
        )

    # ⚠️ 三个期望维度一律走 ``_pick_expected``：本仓库的数据集把它们收在
    # ``expected:`` 块里，用 ``_pick`` 会一块都取不到（见该函数的文档）。
    raw_tools = _pick_expected(entry, _TOOL_ALIASES) or []
    if isinstance(raw_tools, str):
        raw_tools = [raw_tools]

    raw_tags = entry.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = [raw_tags]

    return EvalCase(
        id=case_id,
        input=str(raw_input),
        expected_lane=_parse_lane(_pick_expected(entry, _LANE_ALIASES), case_id),
        expected_intents=_parse_intents(_pick_expected(entry, _INTENT_ALIASES), case_id),
        expected_tools=tuple(str(item) for item in raw_tools),
        tags=tuple(str(item) for item in raw_tags),
        notes=str(entry.get("notes") or ""),
    )


def load_dataset(path: str | Path) -> tuple[EvalCase, ...]:
    """加载数据集文件。

    Args:
        path (`str | Path`): ``.json`` / ``.yaml`` / ``.yml`` 文件路径。

    Returns:
        `tuple[EvalCase, ...]`: 用例列表（顺序与原文件一致）。

    Raises:
        ValueError: 顶层结构不是列表 / ``{"cases": [...]}``，或某条用例非法，
            或出现了重复的 ``id``。
    """
    file_path = Path(path)
    raw = _decode(file_path)

    if isinstance(raw, dict):
        raw = raw.get("cases", [])
    if not isinstance(raw, list):
        raise ValueError(
            f"数据集顶层必须是「用例列表」或 {{\"cases\": [...]}}，"
            f"实际是 {type(raw).__name__}（文件：{file_path}）。",
        )

    cases = tuple(_parse_case(entry, index) for index, entry in enumerate(raw))

    seen: set[str] = set()
    duplicates: list[str] = []
    for case in cases:
        if case.id in seen:
            duplicates.append(case.id)
        seen.add(case.id)
    if duplicates:
        # ⚠️ 重复 id 必须报错而不是覆盖：报告按 id 对齐，重复 id 会让
        # 「两条用例」在指标里变成一个、或让 diff 指错行，且完全静默。
        raise ValueError(f"数据集里出现重复的用例 id：{sorted(set(duplicates))}。")

    return cases


# ==============================================================================
# 三、指标聚合
# ==============================================================================
def compute_metrics(results: tuple[CaseResult, ...]) -> EvalMetrics:
    """把逐条结果聚合成 :class:`EvalMetrics`。

    Args:
        results (`tuple[CaseResult, ...]`): 逐条结果。

    Returns:
        `EvalMetrics`: 聚合指标。

    ⚠️ **只对声明了该维度的用例计分**。一条没声明 ``expected_intents`` 的
    用例，其意图不参与 P/R —— 若把它按「期望空集」算进去，实际识别出
    ``OTHER`` 的用例会被记成一次假阳性，而真相是**这条用例本来就没打算
    评意图**（离线时尤其如此）。指标必须回答「在被评测的那部分上表现如何」，
    而不是「在全部用例上表现如何」，后者会把「未评测」混进分母。
    """
    total = len(results)
    lane_evaluated = 0
    lane_correct = 0
    intent_evaluated = 0
    intent_skipped = 0
    intent_tp = 0
    intent_fp = 0
    intent_fn = 0
    tool_evaluated = 0
    tool_skipped = 0
    tool_exact_match = 0
    judge_sum = 0.0
    judge_evaluated = 0

    for result in results:
        case = result.case
        observation = result.observation

        if case.expected_lane is not None:
            lane_evaluated += 1
            if observation.lane == case.expected_lane.value:
                lane_correct += 1

        if case.expected_intents:
            # ⚠️ 第二个条件与第一个**同样必需**：用例声明了这一维度，
            # 不代表本次观测真的测到了它（离线时慢车道的意图就测不到）。
            # 只看 ``case.expected_intents`` 会把那些用例按 0 分算进分母，
            # 让指标反映「有没有密钥」而不是「系统好不好」。
            if observation.measures(DIM_INTENTS):
                intent_evaluated += 1
                expected = {intent.value for intent in case.expected_intents}
                actual = set(observation.intents)
                intent_tp += len(expected & actual)
                intent_fp += len(actual - expected)
                intent_fn += len(expected - actual)
            else:
                intent_skipped += 1

        if case.expected_tools:
            if observation.measures(DIM_TOOLS):
                tool_evaluated += 1
                if set(case.expected_tools) == set(observation.tools):
                    tool_exact_match += 1
            else:
                tool_skipped += 1

        judge_sum += result.verdict.score

        # ⚠️ 「判分真的有依据可依」= 判分器至少产出了一项检查。规则判分下，
        # 一条没声明任何期望项（或声明的维度恰好都没测到）的用例拿到的是
        # 「无可判分内容 ⇒ 1.0」（见 RuleJudge），那是个**白送**的分，
        # 不能拿来支撑综合得分 —— 否则一份什么都没检查的数据集
        # 会因为一堆白送的 1.0 而拿到满分。
        if result.verdict.checks:
            judge_evaluated += 1

    precision = _ratio(intent_tp, intent_tp + intent_fp)
    recall = _ratio(intent_tp, intent_tp + intent_fn)
    f1 = 0.0 if precision + recall == 0 else round(2 * precision * recall / (precision + recall), 4)

    return EvalMetrics(
        total=total,
        lane_evaluated=lane_evaluated,
        lane_correct=lane_correct,
        lane_accuracy=_ratio(lane_correct, lane_evaluated),
        intent_evaluated=intent_evaluated,
        intent_skipped=intent_skipped,
        intent_tp=intent_tp,
        intent_fp=intent_fp,
        intent_fn=intent_fn,
        intent_precision=precision,
        intent_recall=recall,
        intent_f1=f1,
        tool_evaluated=tool_evaluated,
        tool_skipped=tool_skipped,
        tool_exact_match=tool_exact_match,
        tool_exact_match_rate=_ratio(tool_exact_match, tool_evaluated),
        judge_mean_score=_ratio(round(judge_sum, 4), total) if total else 0.0,
        judge_evaluated=judge_evaluated,
    )


# ==============================================================================
# 四、执行
# ==============================================================================
async def run_case(
    case: EvalCase,
    system: SystemUnderTest,
    judge: Judge,
) -> CaseResult:
    """跑并判一条用例。

    Args:
        case (`EvalCase`): 黄金用例。
        system (`SystemUnderTest`): 被测系统。
        judge (`Judge`): 判分器。

    Returns:
        `CaseResult`: 结果。

    ⚠️ 被测系统抛出的异常在这里被**收成一条结果**（``error`` 非空、
    观测为空），而不是让整轮评测中断。理由：评测是对着**一份数据集**跑的，
    一条用例把进程打挂，等于那一份报告永远拿不到 —— 而恰恰是那条出错的
    用例最需要出现在报告里。异常摘要用 :func:`safe_error` 生成，
    确保不把可能的密钥带进报告。
    """
    error = ""
    try:
        observation = await system.observe(case)
    except Exception as exc:  # noqa: BLE001 —— 见 docstring，绝不中断整轮
        from src.observability.redaction import safe_error

        error = safe_error(exc)
        logger.exception("用例 %s 在被测系统中执行失败。", case.id)
        observation = Observation()

    verdict = await judge.judge(case, observation)
    return CaseResult(case=case, observation=observation, verdict=verdict, error=error)


async def run_dataset(
    cases: tuple[EvalCase, ...],
    system: SystemUnderTest,
    judge: Judge,
    *,
    dataset: str = "",
) -> EvalReport:
    """跑完整份数据集并汇总。

    Args:
        cases (`tuple[EvalCase, ...]`): 用例列表。
        system (`SystemUnderTest`): 被测系统。
        judge (`Judge`): 判分器。
        dataset (`str`): 数据集来源标识（写进报告）。

    Returns:
        `EvalReport`: 汇总报告。

    ⚠️ 用例**串行**执行，不并发。评测不是生产流量，它的第一需求是**可复现**；
    而并发会引入「哪条先跑」的非确定性（对外部模型服务尤其明显），
    让同一份输入的两次结果不同。串行的成本对一份几十条的数据集完全可以接受。
    """
    results = [await run_case(case, system, judge) for case in cases]

    notes: list[str] = []
    if judge_mode(judge) == "rule":
        notes.append(
            "未配置模型密钥：本次使用**规则判分**（只比车道/意图集/工具集），"
            "未做语义评估。",
        )

    return EvalReport(
        dataset=dataset,
        judge_mode=judge_mode(judge),
        results=tuple(results),
        metrics=compute_metrics(tuple(results)),
        notes=tuple(notes),
    )


__all__ = [
    "DATASET_CANDIDATES",
    "PipelineSystem",
    "SystemUnderTest",
    "compute_metrics",
    "find_dataset",
    "load_dataset",
    "run_case",
    "run_dataset",
]
