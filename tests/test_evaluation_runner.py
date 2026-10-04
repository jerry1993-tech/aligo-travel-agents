# -*- coding: utf-8 -*-
"""评测执行器（``src/evaluation/runner.py``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    执行器本身不产生「智能」，它只做两件事：把用例喂给系统、把结果算成指标。
    这两件事各有一个「静默失效」的失败模式：

      1. **指标是空的** —— 分子分母算错、或某个维度被恒等地算成 0，
         报告照样打印得漂漂亮亮，只是那些数字不再随系统表现变化。
         症状是「改了系统，评测分数纹丝不动」，而人会先怀疑系统。
      2. **一条用例把整轮打挂** —— 被测系统抛异常时不接住，整份报告都拿不到，
         而恰恰是那条出错的用例最需要出现在报告里。

    因此本文件的重点是一条 **变异测试**（``test_metrics_change_when_*``）：
    先用一个「完全正确」的假系统拿到满分，再**只改动一个维度**，
    断言对应的指标**必然下降**、而其它指标不动。一个恒返回固定值的实现
    无法通过这组断言 —— 这正是「指标不是摆设」的证明。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import Settings
from src.domain.enums import Intent, LaneName
from src.evaluation import (
    PipelineSystem,
    RuleJudge,
    load_dataset,
    run_case,
    run_dataset,
)
from src.evaluation.runner import find_dataset
from src.evaluation.types import DIM_INTENTS, DIM_TOOLS, EvalCase, Observation


# ==============================================================================
# 测试替身
# ==============================================================================
def _cases() -> tuple[EvalCase, ...]:
    """构造三条覆盖了三种声明形态的用例。

    Returns:
        `tuple[EvalCase, ...]`:
            - ``c1`` 三项全声明；
            - ``c2`` 只声明车道（模拟离线时不可判意图的慢车道用例）；
            - ``c3`` 三项全声明，值不同。
    """
    return (
        EvalCase(
            id="c1",
            input="规划行程",
            expected_lane=LaneName.FAST,
            expected_intents=(Intent.PLAN_TRIP,),
            expected_tools=("main_plan",),
        ),
        EvalCase(
            id="c2",
            input="能报多少？",
            expected_lane=LaneName.SLOW,
        ),
        EvalCase(
            id="c3",
            input="查订单",
            expected_lane=LaneName.FAST,
            expected_intents=(Intent.QUERY_ORDER,),
            expected_tools=("order_query",),
        ),
    )


def _correct_observations() -> dict[str, Observation]:
    """三条用例各自的「完全正确」观测。

    Returns:
        `dict[str, Observation]`: id → 观测。
    """
    return {
        "c1": Observation(lane="FAST", intents=("PLAN_TRIP",), tools=("main_plan",)),
        "c2": Observation(lane="SLOW", intents=("OTHER",), tools=()),
        "c3": Observation(lane="FAST", intents=("QUERY_ORDER",), tools=("order_query",)),
    }


class _FakeSystem:
    """输出完全可控的假系统 —— 变异测试的载体。

    ⚠️ 用它而不是真实的 :class:`~src.evaluation.runner.PipelineSystem`：
    变异测试要的是「只改一个维度，看指标怎么动」。真实系统牵一发动全身，
    改不了单个维度。假系统让「输入 → 输出」是显式的一张表，
    指标算错时能立刻定位到算术，而不是先怀疑模型。
    """

    def __init__(
        self,
        observations: dict[str, Observation],
        *,
        exc: Exception | None = None,
    ) -> None:
        """初始化。

        Args:
            observations (`dict[str, Observation]`): id → 观测。
            exc (`Exception | None`): 非 None 时 ``observe`` 一律抛它
                （用于验证「异常被收成结果」）。
        """
        self._observations = observations
        self._exc = exc

    async def observe(self, case: EvalCase) -> Observation:
        """返回预定观测。

        Args:
            case (`EvalCase`): 用例。

        Returns:
            `Observation`: 观测。

        Raises:
            Exception: 构造时传入了 ``exc`` 时抛出它。
        """
        if self._exc is not None:
            raise self._exc
        return self._observations[case.id]


# ==============================================================================
# 一、变异测试 —— 指标必须**随实际表现变化**
# ==============================================================================
async def test_metrics_are_perfect_when_the_system_is_perfect() -> None:
    """完全正确的系统 ⇒ 全部指标为满分。

    基准。下面三条变异用例都以它为对照：没有它，一个「恒返回 0」的指标
    实现也能让变异断言（"下降了"）通过。
    """
    report = await run_dataset(_cases(), _FakeSystem(_correct_observations()), RuleJudge())

    assert report.metrics.lane_accuracy == 1.0
    assert report.metrics.intent_f1 == 1.0
    assert report.metrics.tool_exact_match_rate == 1.0
    assert report.metrics.judge_mean_score == 1.0
    assert report.overall_score == 1.0


async def test_lane_accuracy_drops_when_only_the_lane_is_wrong() -> None:
    """只把 ``c1`` 的车道判错 ⇒ 只有车道准确率下降，意图/工具指标不动。

    ⚠️ 断言「其它指标不动」与「车道下降」同样重要：只断言前者的话，
    一个「所有指标一起下降」的实现（比如把维度算串了）也能通过。
    """
    observations = _correct_observations()
    observations["c1"] = Observation(lane="SLOW", intents=("PLAN_TRIP",), tools=("main_plan",))

    report = await run_dataset(_cases(), _FakeSystem(observations), RuleJudge())

    assert report.metrics.lane_accuracy == pytest.approx(2 / 3, abs=1e-3)
    assert report.metrics.intent_f1 == 1.0
    assert report.metrics.tool_exact_match_rate == 1.0


async def test_intent_f1_drops_when_only_the_intent_is_wrong() -> None:
    """只把 ``c1`` 的意图判错 ⇒ 只有意图 F1 下降。"""
    observations = _correct_observations()
    observations["c1"] = Observation(lane="FAST", intents=("OTHER",), tools=("main_plan",))

    report = await run_dataset(_cases(), _FakeSystem(observations), RuleJudge())

    assert report.metrics.lane_accuracy == 1.0
    assert report.metrics.intent_f1 == 0.5  # 1 对 1 错：P=R=0.5
    assert report.metrics.tool_exact_match_rate == 1.0


async def test_tool_exact_match_drops_when_only_the_tools_are_wrong() -> None:
    """只把 ``c3`` 的工具集判错 ⇒ 只有工具完全匹配率下降。"""
    observations = _correct_observations()
    observations["c3"] = Observation(lane="FAST", intents=("QUERY_ORDER",), tools=("main_plan",))

    report = await run_dataset(_cases(), _FakeSystem(observations), RuleJudge())

    assert report.metrics.lane_accuracy == 1.0
    assert report.metrics.intent_f1 == 1.0
    assert report.metrics.tool_exact_match_rate == 0.5


async def test_intent_metric_only_counts_cases_that_declare_it() -> None:
    """没声明意图的用例**不进**意图指标的分母。

    ⚠️ 这是离线评测能否成立的关键：``c2`` 没有 ``expected_intents``，即便它
    实际识别出 ``OTHER``，也不该被算成一次假阳性（那会把「没打算评」
    混进「评了但错了」）。
    """
    report = await run_dataset(_cases(), _FakeSystem(_correct_observations()), RuleJudge())

    assert report.metrics.intent_evaluated == 2  # 只有 c1 / c3
    assert report.metrics.intent_tp == 2
    assert report.metrics.intent_fp == 0
    assert report.metrics.intent_fn == 0


# ==============================================================================
# 二、异常与边界
# ==============================================================================
async def test_system_exception_is_captured_as_a_result() -> None:
    """被测系统抛异常 ⇒ 收成该条用例的 ``error``，**不**中断整轮。

    ⚠️ 断言的是「另一条用例仍然出了结果」：若实现让异常向上冒，
    这里会直接抛，整份报告都拿不到 —— 而出错的那条用例恰恰最该被看见。
    """
    cases = _cases()
    result = await run_case(cases[0], _FakeSystem({}, exc=RuntimeError("模拟下游挂了")), RuleJudge())

    assert result.error
    assert "模拟下游挂了" in result.error
    assert result.observation.lane == ""
    assert result.verdict.score == 0.0


async def test_a_dataset_where_nothing_is_checked_scores_zero() -> None:
    """★★★ 一份**什么都没检查**的数据集必须得 0 分，不能得满分。

    ⚠️ 这正是那个真实事故的形状，而且它比「空数据集」隐蔽得多：

        数据集非空（34 条用例、报告有数、有一条条 ✅ 的结果行），
        但每条用例的期望都没被读出来 ⇒ 规则判分对每条判
        「无可判分内容 ⇒ 1.0」⇒ 判分均分 1.0 ⇒ 综合得分 **1.0000**、
        退出码 **0**、报告一片绿。

    一个**一个字都没评**的评测，看起来完美通过。这是本项目评测环节最
    需要避免的失败模式，所以在这里钉死。

    ⚠️ 用例一律写成「声明了期望但观测没测到」而不是「没声明期望」：
    后者（比如数据集里所有用例都只有 id 与 input）走的是另一条路径，
    而这里要挡的是「期望被静默丢弃」那条。
    """

    class _BlindSystem:
        """一个**什么都看不见**的被测系统：三个维度全声明为没测到。"""

        async def observe(self, case: EvalCase) -> Observation:
            """返回一个把三个维度都标为未测量的观测。"""
            return Observation(
                lane="",
                unmeasured=(DIM_INTENTS, DIM_TOOLS),
                output="",
            )

    cases = (
        EvalCase(
            id="c1",
            input="规划行程",
            expected_intents=(Intent.PLAN_TRIP,),
            expected_tools=("search_transport",),
        ),
        EvalCase(
            id="c2",
            input="帮我订票",
            expected_intents=(Intent.PLAN_TRIP,),
            expected_tools=("search_hotels",),
        ),
    )

    report = await run_dataset(cases, _BlindSystem(), RuleJudge())

    assert report.metrics.total == 2
    assert report.metrics.intent_evaluated == 0
    assert report.metrics.tool_evaluated == 0
    assert report.metrics.judge_evaluated == 0, (
        "没有一条用例产出了判分依据，judge_evaluated 必须是 0 —— "
        "否则白送的 1.0 会把综合得分撑起来"
    )
    assert report.metrics.judge_mean_score == 1.0, (
        "逐条判分确实给了 1.0（「无可判分内容」是判通过，不是判不及格）；"
        "要挡的是它**参与综合得分**，所以这条断言必须成立，"
        "而下面那条综合得分的断言才是不变量"
    )
    assert report.overall_score == 0.0, (
        "一份什么都没检查的数据集拿到了非零综合得分 —— "
        "这会让人以为评测跑过了，而实际上一个字都没比"
    )


async def test_empty_dataset_scores_zero() -> None:
    """空数据集 ⇒ 综合得分 0.0（而不是「无数据即满分」）。

    ⚠️ 一个空数据集显示成满分，是最坏的一种假绿：它会让人以为评测跑过了。
    """
    report = await run_dataset((), _FakeSystem({}), RuleJudge())

    assert report.metrics.total == 0
    assert report.overall_score == 0.0


# ==============================================================================
# 三、数据集加载
# ==============================================================================
def test_loader_reads_json(tmp_path: Path) -> None:
    """``.json`` 数据集能被正确读出（含别名与空期望）。"""
    path = tmp_path / "golden_dataset.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "a",
                        "text": "规划行程",
                        "lane": "fast",
                        "intents": ["PLAN_TRIP"],
                        "agents": ["main_plan"],
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_dataset(path)

    assert len(cases) == 1
    assert cases[0].id == "a"
    assert cases[0].input == "规划行程"
    assert cases[0].expected_lane is LaneName.FAST
    assert cases[0].expected_intents == (Intent.PLAN_TRIP,)
    assert cases[0].expected_tools == ("main_plan",)


def test_loader_reads_yaml_the_same_way(tmp_path: Path) -> None:
    """``.yaml`` 与 ``.json`` 必须**语义等价**。

    ⚠️ 协作者可能用任一种格式写数据集（见 runner 的模块说明）。两种格式
    读出来不一样，会让「同一份数据」在两条路径上得到两套指标，而那种偏差
    极难被发现 —— 你只会觉得「这次的结果怎么和上次不同」。
    """
    payload = {
        "cases": [
            {
                "id": "a",
                "input": "规划行程",
                "expected_lane": "FAST",
                "expected_intents": ["PLAN_TRIP"],
                "expected_tools": ["main_plan"],
            },
        ],
    }
    json_path = tmp_path / "golden_dataset.json"
    yaml_path = tmp_path / "golden_dataset.yaml"
    json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    yaml_path.write_text(
        "cases:\n"
        "  - id: a\n"
        "    input: 规划行程\n"
        "    expected_lane: FAST\n"
        "    expected_intents: [PLAN_TRIP]\n"
        "    expected_tools: [main_plan]\n",
        encoding="utf-8",
    )

    assert load_dataset(json_path) == load_dataset(yaml_path)


def test_find_dataset_prefers_yaml(tmp_path: Path) -> None:
    """★ 两种格式都在时取 ``.yaml``（``DATASET_CANDIDATES`` 的顺序即优先级）。

    ⚠️ 这条断言的方向曾经是**反的**，而那个反向的顺序真的造成了事故：
    目录里一份早先的草稿 ``golden_dataset.json`` 一直盖着权威的
    ``golden_dataset.yaml``，``find_dataset`` 每次都返回那份 12 条的旧稿，
    而它**能正常加载**，所以没有任何报错 —— 评测跑了、报告出了，
    只是评的不是你以为的那份数据集。

    这里断言的不是「yaml 比 json 好」，而是**顺序必须与仓库里实际放的那份
    一致**：仓库的黄金数据集是 YAML（理由写在它文件头部），
    所以 YAML 必须排在前面。
    """
    (tmp_path / "golden_dataset.yaml").write_text("cases: []\n", encoding="utf-8")
    (tmp_path / "golden_dataset.json").write_text('{"cases": []}', encoding="utf-8")

    assert find_dataset(tmp_path).name == "golden_dataset.yaml"


def test_find_dataset_still_accepts_a_lone_json(tmp_path: Path) -> None:
    """只有 ``.json`` 时照样能找到（顺序变了不等于不再支持 JSON）。"""
    (tmp_path / "golden_dataset.json").write_text('{"cases": []}', encoding="utf-8")

    assert find_dataset(tmp_path).name == "golden_dataset.json"


def test_loader_rejects_duplicate_ids(tmp_path: Path) -> None:
    """重复 id ⇒ 报错（而不是静默覆盖）。

    ⚠️ 报告按 id 对齐做 diff；重复 id 会让两条用例在指标里变成一个，
    或让 diff 指错行，且完全静默。
    """
    path = tmp_path / "golden_dataset.json"
    path.write_text(
        '{"cases": [{"id": "a", "input": "x"}, {"id": "a", "input": "y"}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="重复"):
        load_dataset(path)


def test_loader_rejects_unknown_intent(tmp_path: Path) -> None:
    """非法的意图名 ⇒ 报错，并列出合法取值。

    反例用例：静默跳过一条写错的用例，会让数据集出现一个「悄悄没被测到」
    的空洞 —— 而空洞是看不出来的。
    """
    path = tmp_path / "golden_dataset.json"
    path.write_text(
        '{"cases": [{"id": "a", "input": "x", "expected_intents": ["NOT_A_INTENT"]}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="合法意图"):
        load_dataset(path)


def test_loader_requires_a_stable_id(tmp_path: Path) -> None:
    """缺 id ⇒ 报错（而不是按位置编一个）。

    ⚠️ 按位置生成的 id 会在插入/删除用例后整体错位，让 diff 出一堆假变化。
    """
    path = tmp_path / "golden_dataset.json"
    path.write_text('{"cases": [{"input": "x"}]}', encoding="utf-8")

    with pytest.raises(ValueError, match="id"):
        load_dataset(path)


# ==============================================================================
# 四、与真实黄金数据集的集成（离线）
# ==============================================================================
def test_golden_dataset_is_well_formed(repo_dir: Path) -> None:
    """仓库里的黄金数据集必须结构完整：每条都声明车道、id 唯一、快慢都有。

    这条守住的是**数据集自身**的质量。它不评测系统 —— 一份写坏的数据集
    会让系统被冤枉（期望值根本达不到），而那种问题会先被误读成「系统变差了」。
    """
    dataset_path = find_dataset(repo_dir / "tests" / "evaluation")
    cases = load_dataset(dataset_path)

    assert len(cases) >= 8, "黄金数据集太小，指标没有统计意义"
    assert all(case.expected_lane is not None for case in cases), "每条用例都应声明车道"
    assert len({case.id for case in cases}) == len(cases)

    lanes = {case.expected_lane for case in cases}
    assert LaneName.FAST in lanes and LaneName.SLOW in lanes, "快慢两条车道都要有用例覆盖"

    fast_cases = [case for case in cases if case.expected_lane is LaneName.FAST]
    assert all(case.expected_intents for case in fast_cases), (
        "快车道用例应当声明期望意图 —— 它不依赖模型，离线也是确定性可判的"
    )


def test_the_real_dataset_loads_with_its_expectations_intact(
    repo_dir: Path,
) -> None:
    """★★★ 仓库里那份数据集经 ``load_dataset`` 读出来，**期望必须还在**。

    ⚠️⚠️ 这是本文件里最要紧的一条，因为少了它，整整一类事故可以完全静默地
    发生，而本文件其余 30 多条用例**一条都抓不住**：

        数据集把期望收在 ``expected: {lane, intents, tools}`` 里，
        而加载器只认平铺字段 ⇒ 每一条用例读出来都是「没声明任何期望」
        ⇒ :class:`~src.evaluation.judge.RuleJudge` 对每条判「无可判分内容
        ⇒ 1.0」⇒ 综合得分 1.0000、退出码 0、报告一片绿。

        **一次一个字都没检查的评测，看起来完美通过。**

    为什么别的用例抓不住：**它们自己构造数据集**（``_cases()`` 里手写
    ``EvalCase(...)``），因此绕过了加载器 —— 加载器坏了它们照样绿。
    唯一能抓住的办法，就是像这样把**真实的那个文件**喂进**真实的那个加载器**，
    再断言读出来的东西不是空的。

    ⚠️ 断言的是「期望非空」而不是「等于某个具体值」：具体值是数据集作者
    在维护的，会变；而「必须读得出来」是不变量。
    """
    cases = load_dataset(find_dataset(repo_dir / "tests" / "evaluation"))

    assert cases, "数据集读出来是空的"
    missing_lane = [case.id for case in cases if case.expected_lane is None]
    missing_intents = [case.id for case in cases if not case.expected_intents]
    assert not missing_lane, (
        f"这些用例的车道期望在加载过程中丢了（预期 nested expected.lane 被读到）："
        f"{missing_lane}。加载器是不是只认平铺字段？"
    )
    assert not missing_intents, f"这些用例的意图期望丢了：{missing_intents}"


async def test_pipeline_system_matches_the_golden_dataset_offline(
    settings: Settings,
    repo_dir: Path,
) -> None:
    """离线跑真实编排链路 ⇒ 可测的维度全对，测不到的维度**明说测不到**。

    ★ 这是「评测链路真的通到系统上」的一次端到端验证：用生产同一条
    :class:`~src.evaluation.runner.PipelineSystem`（MockLLM 降级），
    对仓库里的黄金数据集跑一遍。它同时证明了快车道规则表与数据集是**同步**的
    —— 任何一边改了而另一边没跟上，这里会立刻红。

    ⚠️ 后半段断言（``*_skipped`` 与 ``judge_evaluated``）与前半段同样重要。
    只断言「分数满分」的话，一个把慢车道那些测不到的用例**按 0 分算进分母**
    的实现也会让这里变绿 —— 只要它同时把别的维度都判对。而那个实现产出的
    报告会显示「意图 F1 = 0.66」，读的人会去查意图识别，查半天才发现
    那 0.34 的缺口全部来自「本机没配密钥」。**跳过必须是看得见的。**
    """
    cases = load_dataset(find_dataset(repo_dir / "tests" / "evaluation"))
    report = await run_dataset(cases, PipelineSystem(settings), RuleJudge())
    metrics = report.metrics

    assert metrics.lane_accuracy == 1.0, (
        f"车道判定与数据集的期望不一致：{[r.case_id for r in report.failures()]}"
    )
    assert metrics.intent_f1 == 1.0, (
        f"快车道意图（确定性可判）不该有错：{[r.case_id for r in report.failures()]}"
    )
    assert metrics.tool_exact_match_rate == 1.0, (
        f"快车道工具集（确定性命中 aligo_route_intent）不该有错："
        f"{[r.case_id for r in report.failures()]}"
    )
    assert report.overall_score == 1.0
    assert report.judge_mode == "rule"

    # —— 测不到的维度必须被如实排除，而不是记 0 ——
    slow_cases = [case for case in cases if case.expected_lane is not LaneName.FAST]
    assert metrics.intent_skipped == len(slow_cases), (
        "离线（MockLLM）时慢车道的意图测不到，应当全部跳过 —— "
        "计进分母会让分数反映「有没有密钥」而不是「系统好不好」"
    )
    assert metrics.intent_evaluated == metrics.total - len(slow_cases)

    # ⚠️ 工具那一维的分母不是「全部慢车道用例」：数据集里有一部分慢车道
    # 用例写的是 ``tools: []``，而空列表在 :class:`~src.evaluation.types.EvalCase`
    # 里的语义是**不检查**（见它的文档），不是「期望零个工具」。
    # 所以能进入 `tool_skipped` 的只有那些**声明了非空工具集**的慢车道用例。
    declared_tools = [case for case in slow_cases if case.expected_tools]
    assert metrics.tool_skipped == len(declared_tools)

    # ★ 真正的不变量：**每一条声明了工具集的用例，要么被评测、要么被显式
    # 跳过，没有第三条路。** 这比断言某个具体数字稳，也正是
    # 「静默丢弃」与「如实跳过」的分界线 —— 一个把测不到的用例直接从两个
    # 计数器里都漏掉的实现，会让这条等式不成立。
    assert metrics.tool_evaluated + metrics.tool_skipped == sum(
        1 for case in cases if case.expected_tools
    ), "有用例声明了工具集，却没被评测也没被记为跳过 —— 它凭空消失了"

    assert metrics.judge_evaluated == metrics.total, (
        "每条用例至少有车道的判分依据，因此都该计入 judge_evaluated"
    )
