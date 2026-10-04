# -*- coding: utf-8 -*-
"""评测金标准数据集（``tests/evaluation/golden_dataset.yaml``）的自校验。

本文件**不评测系统**，它评测**数据集本身**——确保那份「期望答案」不是一份
会悄悄过期的文档，而是一份被可执行断言钉住的事实。

═══════════════════════════════════════════════════════════════════════════════
这些用例在防什么
═══════════════════════════════════════════════════════════════════════════════

数据集天生会**漂移**：它引用的意图词表、工具名、快车道规则，全都定义在
别处（``src/domain/enums.py``、``src/tools/``、``src/orchestration/classifier.py``）。
数据集自己不知道它们长什么样，于是任何一处改动都会让它慢慢与现实脱节——
而**脱节不会报错**，只会让评测跑出一堆自我安慰的分数。具体三类：

1. **意图漂移**：``Intent`` 加了新成员、或改了值，数据集里写的还是老词。
   症状是评测把系统判对的答案记成判错。
   → :func:`test_every_expected_intent_is_in_the_real_taxonomy`
   → :func:`test_every_taxonomy_member_is_covered`

2. **工具名漂移**：工具改名（比如 ``search_transport`` → ``search_flight``），
   数据集没跟着改。症状同上一类，但更隐蔽——工具名是字符串，改错了不报错。
   → :func:`test_every_expected_tool_is_a_registered_tool`

3. **快车道期望与规则表不符**：这是最危险的一类。数据集的 ``lane`` 是**人工
   标注**的，而真实判定在 :func:`src.orchestration.classifier.classify`。两者
   若各写一份、又不比对，数据集就变成「系统**曾经**想怎么走」的记录——
   拿它评测只会得出「系统符合它自己昨天的行为」，毫无意义。
   → :func:`test_fast_lane_expectations_match_the_classifier`
   → :func:`test_slow_lane_expectations_match_the_classifier`

⚠️ 本文件的组织方式是「**先验结构，再验内容**」：schema 校验（每个字段都在、
类型都对）放在前面，跨模块的一致性校验（意图、工具、车道）放在后面。这样
一条内容错误的用例会以「哪个字段不对」报出来，而不是在一个 KeyError 里
淹掉真正的原因。

⚠️ 本文件**不 import agentscope**：它只碰 ``src.domain``、``src.orchestration``
与 ``src.tools`` 的构造入口，不构造 Agent、不跑模型。评测数据集的校验不该
依赖框架能跑起来——那会让「数据集坏了」和「环境坏了」两种红灯混在一起。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.config import load_settings
from src.domain import Intent, LaneName
from src.llm.mock import MockChatModel
from src.orchestration.classifier import classify
from src.server.agents_factory import build_repositories, build_tools_factory
from src.tools.route import ROUTE_TOOL_NAME

#: 数据集文件路径。用 ``__file__`` 定位而非 ``Path.cwd()``：pytest 的工作目录
#: 可能是仓库根，也可能是别处（取决于怎么调用），只有基于本文件的位置才稳定。
#: 这与 ``tests/conftest.py`` 的 ``repo_dir`` 夹具是同一个理由。
_DATASET_PATH = Path(__file__).resolve().parent / "evaluation" / "golden_dataset.yaml"

#: 允许的分组取值。
#:
#: ⚠️ 做成集合而不是「随便什么都行」：一个拼错的分组名（比如 ``fast-lane``
#: 写成 ``fastlane``）不会让任何断言失败，只会让「按分组统计通过率」这个
#: 动作漏掉一批用例。在这里拦下，报错信息能带上那个坏名字。
_ALLOWED_CATEGORIES: frozenset[str] = frozenset(
    {"fast_lane", "single_intent", "multi_intent", "adversarial"},
)

#: 允许的车道取值。直接取 :class:`LaneName` 的值，避免又写一份字面量。
_ALLOWED_LANES: frozenset[str] = frozenset(lane.value.lower() for lane in LaneName)

#: 用例数量下限。数据集「够不够用」由这条守着——低于它说明有人删了一批用例
#: 却没补新的，而删除本身不会让任何一致性断言失败。
_MIN_CASES = 30

#: 加载数据集。放在模块级是为了能给 ``pytest.mark.parametrize`` 供数——
#: 参数化需要在**收集期**拿到用例列表。
#:
#: ⚠️ 这里**不吞异常**：文件缺失或 YAML 语法错误会直接在收集期炸开。这是
#: 刻意的——一份读不出来的数据集，唯一有意义的行为就是立刻红灯，而不是
#: 让一堆参数化用例「因为 0 条用例」而全部跳过（跳过是绿的，会被误当成通过）。
_DATASET: dict[str, Any] = yaml.safe_load(_DATASET_PATH.read_text(encoding="utf-8"))
_CASES: list[dict[str, Any]] = _DATASET["cases"]


def _case_ids() -> list[str]:
    """给参数化提供可读的用例 id。

    Returns:
        `list[str]`: 每条用例的 ``id``（缺失时回落到下标）。
    """
    return [str(case.get("id", f"<无 id #{index}>")) for index, case in enumerate(_CASES)]


# ==============================================================================
# 一、结构：文件的形状本身是对的
# ==============================================================================
def test_dataset_file_exists_and_parses() -> None:
    """数据集文件存在且能解析成 YAML 映射。

    ⚠️ 断言的是「解析成一个 dict 且有 cases」，而不是「文件能打开」——
    后者对一份写坏的 YAML 同样成立（文件当然能打开）。
    """
    assert _DATASET_PATH.is_file(), f"数据集文件不存在：{_DATASET_PATH}"
    assert isinstance(_DATASET, dict), "数据集顶层应当是映射（mapping）"
    assert isinstance(_CASES, list), "cases 应当是列表"


def test_dataset_declares_a_version() -> None:
    """★ 数据集带版本号，且是正整数。

    ⚠️ 版本号守的是**兼容性**：将来若改了某个字段的含义（比如 ``tools`` 从
    「期望调用」改成「允许调用」），评测脚本必须能分辨新旧两版格式。没有
    版本号，那种改动只能靠「跑起来发现不对」来发现。
    """
    version = _DATASET.get("version")
    assert isinstance(version, int) and not isinstance(version, bool), \
        f"version 应当是整数，实际为 {version!r}"
    assert version >= 1, f"version 应当为正整数，实际为 {version}"


def test_dataset_has_enough_cases() -> None:
    """★ 用例数不得少于下限。

    ⚠️ 数量不是质量，但没有数量就谈不上覆盖。这条挡的是「数据集被删空了一半
    却没人发现」——删用例不会让任何一致性断言失败，只会让评测悄悄变弱。
    """
    assert len(_CASES) >= _MIN_CASES, (
        f"数据集只有 {len(_CASES)} 条用例，少于下限 {_MIN_CASES} 条"
    )


def test_every_case_id_is_unique() -> None:
    """★ 用例 id 全局唯一。

    ⚠️ 重复 id 的危害在**报告**里：评测结果按 id 索引时，两条用例会互相覆盖，
    于是「哪条挂了」变得无法回答。而重复本身在文件里一眼看不出来。
    """
    ids = _case_ids()
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    assert not duplicates, f"存在重复的用例 id：{duplicates}"


@pytest.mark.parametrize("case", _CASES, ids=_case_ids())
def test_case_schema_is_valid(case: dict[str, Any]) -> None:
    """每条用例的字段齐全、类型正确、取值在允许范围内。

    ⚠️ 逐字段断言而不是「取出来用用看」：字段拼错（``expect`` 写成 ``expected``）
    时，后者只会在**下游某处**抛一个 KeyError，报错点离真正的原因很远。
    """
    case_id = case.get("id")
    assert isinstance(case_id, str) and case_id.strip(), f"id 缺失或为空：{case!r}"

    category = case.get("category")
    assert category in _ALLOWED_CATEGORIES, (
        f"用例 {case_id!r} 的 category={category!r} 不在允许集合 "
        f"{sorted(_ALLOWED_CATEGORIES)} 内"
    )

    text = case.get("input")
    assert isinstance(text, str) and text.strip(), f"用例 {case_id!r} 的 input 为空"

    rubric = case.get("rubric")
    assert isinstance(rubric, str) and rubric.strip(), f"用例 {case_id!r} 缺少 rubric"

    expected = case.get("expected")
    assert isinstance(expected, dict), f"用例 {case_id!r} 缺少 expected 映射"

    lane = expected.get("lane")
    assert lane in _ALLOWED_LANES, (
        f"用例 {case_id!r} 的 expected.lane={lane!r} 不是 {sorted(_ALLOWED_LANES)} 之一"
    )

    intents = expected.get("intents")
    assert isinstance(intents, list) and intents, f"用例 {case_id!r} 的 expected.intents 为空"

    tools = expected.get("tools")
    assert isinstance(tools, list), f"用例 {case_id!r} 的 expected.tools 应当是列表（可为空）"

    # ⚠️ 快车道一条规则只映射一个意图，所以快车道用例的 intents 必须**恰好一条**。
    #    写成两条意味着作者误以为快车道能做多意图拆分——而快车道恰恰**不能**，
    #    那正是它把多意图交给慢车道的原因。
    if lane == LaneName.FAST.value.lower():
        assert len(intents) == 1, (
            f"用例 {case_id!r} 是快车道，但期望了 {len(intents)} 条意图；"
            f"快车道一条规则只产出一个意图"
        )


# ==============================================================================
# 二、意图：期望值必须落在真实的意图词表内（import，不抄第二份）
# ==============================================================================
@pytest.mark.parametrize("case", _CASES, ids=_case_ids())
def test_every_expected_intent_is_in_the_real_taxonomy(case: dict[str, Any]) -> None:
    """★ 每条期望意图都是 :class:`~src.domain.enums.Intent` 的真实成员。

    ⚠️ 判据是**当场 import 的枚举**，不是数据集里另抄的一份清单。抄一份的
    后果是「词表改值、数据集照旧通过」——这与本文件开篇列的失败模式 1 是
    同一件事，也是这份自校验测试存在的根本理由。
    """
    case_id = case["id"]
    for raw in case["expected"]["intents"]:
        try:
            Intent(raw)
        except ValueError:
            valid = sorted(member.value for member in Intent)
            raise AssertionError(
                f"用例 {case_id!r} 的意图 {raw!r} 不在 Intent 词表内；"
                f"合法取值：{valid}",
            ) from None


def test_every_taxonomy_member_is_covered() -> None:
    """★ 意图词表里的**每个**成员都在数据集里至少出现一次。

    ⚠️ 这条与前一条方向相反，缺一不可：前一条防「数据集引用了不存在的意图」，
    这条防「词表新增了一个意图，数据集却一条用例都没覆盖它」。后者不会报错，
    只会让评测在新意图上**零覆盖**——而零覆盖的报告看起来就像满分。
    """
    covered: set[str] = set()
    for case in _CASES:
        covered.update(case["expected"]["intents"])

    missing = sorted(member.value for member in Intent if member.value not in covered)
    assert not missing, f"这些意图在数据集里没有任何用例覆盖：{missing}"


# ==============================================================================
# 三、工具：期望的工具名必须是真实注册的工具（import 装配入口，不抄第二份）
# ==============================================================================
@pytest.fixture(scope="module")
def registered_tool_names() -> frozenset[str]:
    """跑一次真实的工具装配，返回全部已注册工具的名字。

    ⚠️ 走 :func:`src.server.agents_factory.build_tools_factory` 这条**生产
    装配路径**，而不是手工列一个名字列表。列列表等于又抄了一份真值——
    工具改名/增删时它会静默过期，而校验的作用恰好是发现那种过期。

    ⚠️ 用 :class:`~src.llm.mock.MockChatModel` 而不是真实模型：装配只需要一个
    ``ChatModelBase`` 实例（它被意图识别器捕获，构造期**不发起任何调用**）。
    用真实模型会让这个纯数据校验依赖凭据与网络——那是毫无必要的耦合。

    ⚠️ 用 ``TEST_ENVIRON`` 显式构造一份配置，而不是取 ``settings`` 夹具：本夹具
    是 module 作用域，而 ``settings`` 是 function 作用域，两者不兼容。这里
    自己 ``load_settings`` 一次即可，且它不触碰 ``get_settings()`` 的进程单例。
    """
    from tests.conftest import TEST_ENVIRON

    definition = load_settings("test", environ=TEST_ENVIRON, dotenv=False)
    factory = build_tools_factory(
        settings=definition,
        repositories=build_repositories(),
        model=MockChatModel(),
    )
    tools = asyncio.run(factory("eval-user", "agent-1", "session-1"))
    return frozenset(tool.name for tool in tools)


@pytest.mark.parametrize("case", _CASES, ids=_case_ids())
def test_every_expected_tool_is_a_registered_tool(
    case: dict[str, Any],
    registered_tool_names: frozenset[str],
) -> None:
    """★ 每个期望工具名都是当前**真实注册**的工具。

    ⚠️ 工具名是字符串，改错了既不报错、也不影响 YAML 的合法性。唯一能挡住
    「数据集引用了一个不存在的工具」的，就是把它与该工厂的真实输出比对。
    """
    case_id = case["id"]
    for tool_name in case["expected"]["tools"]:
        assert tool_name in registered_tool_names, (
            f"用例 {case_id!r} 期望的工具 {tool_name!r} 未注册；"
            f"当前注册的工具：{sorted(registered_tool_names)}"
        )


def test_fast_lane_cases_route_through_the_route_tool(registered_tool_names: frozenset[str]) -> None:
    """★ 快车道用例的期望工具**只有** ``aligo_route_intent``。

    ⚠️ 这不是可选的约定，而是快车道机制的一部分：规则命中后由
    ``LaneRouterMiddleware`` 合成一次对路由工具的调用，把「系统已替你判定了
    意图」显式告诉模型（见 ``src/tools/route.py``）。快车道**不**直接调业务工具。
    若某条快车道用例期望了别的工具，说明作者误解了快车道做了什么。
    """
    assert ROUTE_TOOL_NAME in registered_tool_names, (
        f"路由工具 {ROUTE_TOOL_NAME!r} 竟然没有注册——快车道会因此静默失效"
    )
    for case in _CASES:
        if case["expected"]["lane"] != LaneName.FAST.value.lower():
            continue
        assert case["expected"]["tools"] == [ROUTE_TOOL_NAME], (
            f"快车道用例 {case['id']!r} 的期望工具是 "
            f"{case['expected']['tools']!r}，应当恰好是 [{ROUTE_TOOL_NAME!r}]"
        )


# ==============================================================================
# 四、车道：期望的 fast/slow 必须与真实分类器逐条一致
# ==============================================================================
@pytest.mark.parametrize(
    "case",
    [c for c in _CASES if c["expected"]["lane"] == LaneName.FAST.value.lower()],
    ids=[c["id"] for c in _CASES if c["expected"]["lane"] == LaneName.FAST.value.lower()],
)
def test_fast_lane_expectations_match_the_classifier(case: dict[str, Any]) -> None:
    """★★ 标注为快车道的用例，**真的**被 :func:`classify` 判成快车道。

    这是本文件最重要的一条。数据集的 ``lane`` 是人工标注的，而真实判定在
    ``classifier.classify``；两者只有在这里被**强制比对**，数据集才不是一份
    「系统曾经的想法」的记录，而是一份可与现实对照的期望。

    ⚠️ 同时断言 ``matched_rule`` 非空与意图一致：只断言 lane 的话，一个
    「所有输入都走快车道」的错误实现也能让本用例全绿——而那种实现会让
    问句、长句、多意图全部被规则误吞，是最糟的一类缺陷。
    """
    expected = case["expected"]
    decision = classify(case["input"])

    assert decision.lane == LaneName.FAST, (
        f"用例 {case['id']!r} 标注为快车道，但 classify 判成了 "
        f"{decision.lane.value}（原因：{decision.reason}）"
    )
    assert decision.matched_rule, (
        f"用例 {case['id']!r} 走了快车道却没有命中规则（matched_rule 为空）"
    )
    assert decision.intent.value == expected["intents"][0], (
        f"用例 {case['id']!r} 的期望意图是 {expected['intents'][0]!r}，"
        f"但 classify 给出的是 {decision.intent.value!r}"
    )


@pytest.mark.parametrize(
    "case",
    [c for c in _CASES if c["expected"]["lane"] == LaneName.SLOW.value.lower()],
    ids=[c["id"] for c in _CASES if c["expected"]["lane"] == LaneName.SLOW.value.lower()],
)
def test_slow_lane_expectations_match_the_classifier(case: dict[str, Any]) -> None:
    """★★ 标注为慢车道的用例，**不能**被 :func:`classify` 判成快车道。

    ⚠️ 这是上一条的**反方向**，而且方向相反、缺一不可。只校验「快车道期望确实
    是快车道」，会漏掉一种回归：某天有人往规则表里加了一条过宽的短语，把一条
    本该走慢车道的用例误吞进快车道——那时那条用例的标注仍是 slow，而系统已经
    悄悄改变了行为。这条断言会当场变红，把「规则表变宽了」这件事摆到明面上。

    ⚠️ 只断言 ``lane == SLOW``，**不**断言意图：慢车道的 ``RouteDecision.intent``
    此刻是 ``OTHER`` 占位，真实意图要等意图识别智能体给出（见
    ``classifier._slow_lane``）。对着占位值断言意图是错的。
    """
    decision = classify(case["input"])
    assert decision.lane == LaneName.SLOW, (
        f"用例 {case['id']!r} 标注为慢车道，但 classify 判成了 "
        f"{decision.lane.value}（命中规则：{decision.matched_rule!r}）——"
        f"这说明规则表里有一条过宽的短语把它误吞了"
    )


__all__ = [
    "test_case_schema_is_valid",
    "test_dataset_declares_a_version",
    "test_dataset_file_exists_and_parses",
    "test_dataset_has_enough_cases",
    "test_every_case_id_is_unique",
    "test_every_expected_intent_is_in_the_real_taxonomy",
    "test_every_expected_tool_is_a_registered_tool",
    "test_every_taxonomy_member_is_covered",
    "test_fast_lane_cases_route_through_the_route_tool",
    "test_fast_lane_expectations_match_the_classifier",
    "test_slow_lane_expectations_match_the_classifier",
]
