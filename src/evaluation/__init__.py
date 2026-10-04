# -*- coding: utf-8 -*-
"""**离线评测**：黄金数据集 → 编排链路 → 判分 → 报告。

文件职责：
    本包是博客里「P5 Golden Dataset」那一步的落点。它把「改动之后系统
    到底变好了还是变差了」这个问题，从「凭感觉」变成「一份可 diff 的报告」。

    - :mod:`src.evaluation.types`  数据契约（用例 / 观测 / 判分 / 报告 / 指标）；
    - :mod:`src.evaluation.judge`  判分（规则判分 / 大模型判分 / 选路）；
    - :mod:`src.evaluation.runner` 执行（加载数据集 / 跑链路 / 聚合指标）。

上下游依赖：
    - 上游：:mod:`src.orchestration`、:mod:`src.agents`、:mod:`src.llm`、
      :mod:`src.domain`。
    - 下游：``scripts/eval.py``（CLI）、``tests/test_evaluation_*.py``。

⚠️ 本包**不 import** ``agentscope`` 之外的东西来「跑完整服务」：评测的对象
是编排决策（车道 / 意图 / 调度目标），这些在装配任何存储与 HTTP 层之前
就已确定。因此 ``make eval`` 不需要 Docker。
"""

from __future__ import annotations

from src.evaluation.judge import (
    JUDGE_MODE_LLM,
    JUDGE_MODE_RULE,
    Judge,
    LLMJudge,
    RuleJudge,
    build_judge,
    judge_mode,
)
from src.evaluation.runner import (
    DATASET_CANDIDATES,
    PipelineSystem,
    SystemUnderTest,
    compute_metrics,
    find_dataset,
    load_dataset,
    run_case,
    run_dataset,
)
from src.evaluation.types import (
    CaseResult,
    EvalCase,
    EvalMetrics,
    EvalReport,
    JudgeCheck,
    JudgeVerdict,
    Observation,
)

__all__ = [
    "DATASET_CANDIDATES",
    "JUDGE_MODE_LLM",
    "JUDGE_MODE_RULE",
    "CaseResult",
    "EvalCase",
    "EvalMetrics",
    "EvalReport",
    "Judge",
    "JudgeCheck",
    "JudgeVerdict",
    "LLMJudge",
    "Observation",
    "PipelineSystem",
    "RuleJudge",
    "SystemUnderTest",
    "build_judge",
    "compute_metrics",
    "find_dataset",
    "judge_mode",
    "load_dataset",
    "run_case",
    "run_dataset",
]
