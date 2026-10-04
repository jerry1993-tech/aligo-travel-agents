#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线评测：把黄金数据集跑一遍编排链路，判分并产出可 diff 的报告。

==============================================================================
它与 `make test` / `make smoke` 的分工
==============================================================================
    `make test`     进程内断言「代码逻辑对不对」（确定性，秒级）
    `make smoke`    对已启动的服务发请求，测「这一套部署能不能用」
    本脚本          对**一份固定数据集**跑编排链路，测「改动之后系统是变好还是变差了」

    三者回答的是三个不同的问题，缺一不可。本脚本补的是那个最容易被忽略的：
    **回归**。单测能证明「规则表命中了我写的那条短语」，却不能告诉你
    「改了归一化之后，另外 5 条短语不再命中了」。评测用一份**固定**的用例集
    把系统的整体行为钉住，让「悄悄变差」变成一次红色的 diff。

==============================================================================
⚠️ 零密钥也能跑，但报告会如实说明「判分降级了」
==============================================================================
    没有模型密钥时（本机与 CI 的常态）：

      · 慢车道的意图识别会退化成 :class:`~src.llm.mock.MockChatModel` 的
        确定性回复，识别不出真实意图 —— 因此**慢车道用例的意图维度不评**
        （数据集中那些用例的 ``expected_intents`` 是空的，见 runner 的说明）；
      · 判分器退回**规则判分**，只比车道 / 意图集 / 工具集，报告里
        ``judge_mode`` 会是 ``rule``。

    这是刻意的诚实：报告不会把「只比了三个集合」包装成「语义质量评估过了」。
    配好密钥后重跑同一份数据集，判分模式会变成 ``llm``，报告里那个分数
    才真的包含语义判断。

退出码约定：**0 = 综合得分达到阈值；1 = 未达到或执行失败**。
    （与 Makefile 的 `eval` 目标配合，未达标即 `make eval` 失败。）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# 允许以 `python scripts/eval.py` 直接运行（此时 sys.path[0] 是 scripts/）。
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.config import Settings, get_settings, load_settings, repo_root  # noqa: E402
from src.evaluation import (  # noqa: E402
    build_judge,
    find_dataset,
    judge_mode,
    load_dataset,
    run_dataset,
    PipelineSystem,
    EvalReport,
)
from src.observability.redaction import safe_error  # noqa: E402

#: 默认的阈值。
#:
#: ⚠️ 取值依据（不是随手拍的）：本项目的黄金数据集里，**快车道用例**
#: （车道 + 意图 + 工具三项齐备）是确定性可判的，理应全对；**慢车道用例**
#: 只判车道。也就是说一个正常工作的系统在这份数据集上的综合得分应当接近 1.0。
#: 阈值取 0.8 是留出「个位数用例失败」的余量：再低就挡不住真实回归，
#: 再高则任何一点合理的调整都会误报。要更严或更松，用 --min-score 覆盖。
DEFAULT_MIN_SCORE = 0.8

#: 默认的报告落盘文件名（相对于当前工作目录）。
DEFAULT_OUTPUT = "eval_report.json"


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    Returns:
        `argparse.ArgumentParser`: 解析器。
    """
    parser = argparse.ArgumentParser(
        description="AliGo 差旅助手 —— 对黄金数据集执行离线评测。",
        epilog=(
            "示例：\n"
            "  python scripts/eval.py\n"
            "  python scripts/eval.py --dataset tests/evaluation/golden_dataset.yaml\n"
            "  python scripts/eval.py --min-score 0.9 --output /tmp/report.json\n"
            "  make eval MIN_SCORE=0.9\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="数据集文件或所在目录（.json / .yaml）。默认在 tests/evaluation/ 下自动查找。",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=DEFAULT_MIN_SCORE,
        help=f"综合得分的最低阈值（默认 {DEFAULT_MIN_SCORE}）；低于它则退出码为 1。",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"机器可读 JSON 报告的落盘路径（默认 {DEFAULT_OUTPUT}，相对当前目录）。",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="配置档（dev / test / prod）。默认取 ALIGO__APP__ENV 或 dev。",
    )
    return parser


def _resolve_dataset(raw: str | None) -> Path:
    """把 ``--dataset`` 解析成一个具体的文件路径。

    Args:
        raw (`str | None`): 命令行给的路径或目录；``None`` 时用默认位置。

    Returns:
        `Path`: 数据集文件路径。

    ⚠️ 「给的是目录」与「给的是文件」都要能接受：约定里数据集在
    ``tests/evaluation/`` 下，但协作者可能把文件落在别处，用 ``--dataset``
    指向那个目录即可 —— 不必记住具体文件名。
    """
    target = Path(raw) if raw else repo_root() / "tests" / "evaluation"
    if target.is_dir():
        return find_dataset(target)
    if target.is_file():
        return target
    raise FileNotFoundError(
        f"--dataset 指向的路径不存在：{target}\n"
        f"   · 它是一个目录时，应在其中放 golden_dataset.json / .yaml；\n"
        f"   · 它也可以直接是一个 .json / .yaml 文件。",
    )


def _format_result_line(report: EvalReport, index: int) -> str:
    """渲染一行用例结果。

    Args:
        report (`EvalReport`): 报告。
        index (`int`): 第几条（0 基）。

    Returns:
        `str`: 形如 ``  ✅ id  车道=FAST 得分=1.0000`` 的一行。
    """
    result = report.results[index]
    mark = "❌" if (result.error or result.verdict.score < 1.0) else "✅"
    detail = (
        f"车道={result.observation.lane or '<空>'} "
        f"意图={list(result.observation.intents)} "
        f"工具={list(result.observation.tools)} "
        f"得分={result.verdict.score:.4f}"
    )
    if result.error:
        detail += f" —— 执行出错：{result.error}"
    return f"  {mark} {result.case_id}  {detail}"


def _print_report(report: EvalReport, min_score: float, output: Path, dataset: Path) -> None:
    """打印人类可读的评测报告。

    Args:
        report (`EvalReport`): 评测结果。
        min_score (`float`): 阈值。
        output (`Path`): JSON 报告落盘路径。
        dataset (`Path`): 数据集路径。
    """
    metrics = report.metrics
    print(f"▶ 评测数据集：{dataset}")
    print(
        f"▶ 判分模式：{report.judge_mode}（"
        + ("规则判分：只比车道/意图集/工具集" if report.judge_mode == "rule" else "大模型判分")
        + "）",
    )
    print(f"▶ 用例数：{metrics.total}")
    for note in report.notes:
        print(f"  ⚠️ {note}")
    print()

    for index in range(len(report.results)):
        print(_format_result_line(report, index), flush=True)

    print()
    print("指标（括号内为该维度被评测的用例数）：")
    print(f"    车道准确率      {metrics.lane_accuracy:.4f} ({metrics.lane_correct}/{metrics.lane_evaluated})")
    print(
        f"    意图 P/R/F1     {metrics.intent_precision:.4f} / {metrics.intent_recall:.4f} / "
        f"{metrics.intent_f1:.4f} ({metrics.intent_evaluated})",
    )
    print(
        f"    工具完全匹配    {metrics.tool_exact_match_rate:.4f} "
        f"({metrics.tool_exact_match}/{metrics.tool_evaluated})",
    )
    print(f"    判分均分        {metrics.judge_mean_score:.4f} (有依据可判 {metrics.judge_evaluated})")
    print(f"    综合得分        {report.overall_score:.4f}")
    print()

    # ⚠️ 「被跳过的维度」必须**无条件**打印出来，哪怕是 0。
    #
    # 这是本脚本最重要的一段输出。被跳过意味着那些用例的期望值**没有参与
    # 打分**，而它们在报告里不会留下任何别的痕迹 —— 分数只会比「全都评了」
    # 时更高、更好看。不打印的话，一次「18 条用例的意图根本没比」的评测
    # 与一次「18 条用例的意图全对」的评测，在报告上完全一样。
    #
    # 离线（没有模型密钥）时这一行必然非零：慢车道的意图由 Mock 模型给出，
    # 识别不出真实意图，计入只会让分数反映「本机没配密钥」。
    # 详见 src/evaluation/types.py 里 DIM_INTENTS 上方那段说明。
    print("未参与评测的维度（这些用例的该维度期望值**未**计入任何分数）：")
    print(f"    意图集被跳过    {metrics.intent_skipped} 条")
    print(f"    工具集被跳过    {metrics.tool_skipped} 条")
    if metrics.intent_skipped or metrics.tool_skipped:
        print("    ⚠️ 上述条目的期望值没有被检查，分数比「全都评了」时偏高。")
        if report.judge_mode == "rule":
            print("    ⚠️ 本次是规则判分（未配密钥）：慢车道的意图本来就测不到，属预期。")
    print()

    failures = report.failures()
    if failures:
        print("   未满分用例：")
        for result in failures:
            reason = result.error or result.verdict.rationale
            print(f"     · {result.case_id}：{reason}")
        print()

    print(f"▶ 机器可读报告：{output}")


async def _run(args: argparse.Namespace) -> int:
    """执行一轮评测。

    Args:
        args (`argparse.Namespace`): 命令行参数。

    Returns:
        `int`: 进程退出码。
    """
    dataset_path = _resolve_dataset(args.dataset)
    cases = load_dataset(dataset_path)
    settings: Settings = load_settings(args.env) if args.env else get_settings()

    system = PipelineSystem(settings)
    judge = build_judge(settings)

    print(f"▶ 正在执行 {len(cases)} 条用例...", flush=True)
    report = await run_dataset(
        cases,
        system,
        judge,
        dataset=str(dataset_path),
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    _print_report(report, args.min_score, output, dataset_path)
    print()

    if report.overall_score >= args.min_score:
        print(f"✅ 评测通过：综合得分 {report.overall_score:.4f} ≥ 阈值 {args.min_score:.4f}")
        return 0

    print(
        f"❌ 评测未通过：综合得分 {report.overall_score:.4f} < 阈值 {args.min_score:.4f}\n"
        f"    最可能的三个原因：\n"
        f"      1. 编排链路确实变差了 —— 看上面「未满分用例」的明细，逐条定位；\n"
        f"      2. 数据集本身有问题（期望值与当前的设计意图不符）—— 核对那条用例的 notes；\n"
        f"      3. 阈值定得过高 —— 若是刻意的收紧，用 --min-score 明确表达，别改默认值。",
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv (`list[str] | None`): 命令行参数；``None`` 时取 ``sys.argv``。

    Returns:
        `int`: 0 表示达标，1 表示未达标或执行失败。
    """
    args = _build_parser().parse_args(argv)

    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 —— 顶层入口，见下面的说明
        # ⚠️ 顶层捕获宽异常是刻意的：这个脚本的失败原因五花八门
        # （数据集格式、配置加载、模型装配），每一种的异常类型都不同。
        # 把 traceback 压成一条安全摘要（safe_error 会去掉可能的密钥），
        # 再给一条最可能的方向，比让一屏栈把结论淹没有用得多。
        print(f"\n❌ 评测失败：{safe_error(exc)}", file=sys.stderr)
        print(
            "   提示：先确认数据集格式（--dataset 指向 .json / .yaml），"
            "再确认配置能加载（python -c 'from src.config import get_settings; get_settings()'）。",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
