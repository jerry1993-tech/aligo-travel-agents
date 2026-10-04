# -*- coding: utf-8 -*-
"""动态 Prompt 组装的单测。

组织方式按**真实的失败模式**来分：

1. 阶段覆盖 —— 有没有哪个阶段静默退化成「没有指令」；
2. 幂等性 —— **本模块最核心的保证**。``on_system_prompt`` 每轮都跑，
   一旦不幂等，prompt 会逐轮膨胀，症状是「聊几句就报上下文超限」；
3. 剥离的边角 —— 未闭合标记、多段落、标记夹在中间；
4. 默认值不是「已确认」 —— 个性化最容易翻车的地方；
5. 已知与待补齐不重叠 —— 重复追问是对话式收集最招人烦的失败模式；
6. 开关能干净关闭；
7. 确定性与不抛异常 —— 它跑在 agent 主链路上；
8. 纯度 —— 不 import 框架（**必须**在子进程验，见下）。

⚠️ 本文件**不 import agentscope**，这是 ``src/orchestration/__init__.py``
的纯度设计所保证的。
"""

from __future__ import annotations

import pytest

from src.domain import CabinClass, TransportMode, TravelRequest, TripStage
from src.orchestration import (
    MARKER_BEGIN,
    MARKER_END,
    MAX_MISSING_PROMPTS,
    build_system_prompt,
    describe_known_slots,
    strip_managed_sections,
)
from src.orchestration.prompt import (
    _CABIN_LABELS,
    _STAGE_DIRECTIVES,
    _STAGE_LABELS,
    _TRANSPORT_LABELS,
)

BASE = "你是 AliGo 差旅助手。"

#: 完整填好的一个出差申请，多数用例直接复用它。
FULL_REQUEST = TravelRequest(
    origin="杭州",
    destination="北京",
    depart_date="2026-10-08",
    return_date="2026-10-10",
    days=3,
    transport_mode=TransportMode.FLIGHT,
    cabin_class=CabinClass.ECONOMY,
    hotel_required=True,
    hotel_area="国贸",
    travelers=2,
    purpose="客户拜访",
    budget=5000.0,
    stage=TripStage.CONFIRMING,
)


# ---------------------------------------------------------------------------
# 1. 阶段覆盖
# ---------------------------------------------------------------------------


def test_every_stage_has_a_directive() -> None:
    """每个阶段都必须有主链路指令。

    ⚠️ 漏登记的阶段会退化成「没有任何动态指令」，表现为该阶段下模型行为
    莫名其妙地回到通用状态 —— 「某些情况下不太对劲」是最难定位的一类问题。
    """
    assert set(_STAGE_DIRECTIVES) == set(TripStage)


def test_every_stage_has_a_label() -> None:
    """每个阶段都必须有中文展示名。"""
    assert set(_STAGE_LABELS) == set(TripStage)


@pytest.mark.parametrize("stage", list(TripStage))
def test_build_works_for_every_stage(stage: TripStage) -> None:
    """每个阶段都能正常组装出非空、且不含空指令的 prompt。"""
    result = build_system_prompt(BASE, stage=stage, request=FULL_REQUEST)

    assert BASE in result
    assert MARKER_BEGIN in result and MARKER_END in result
    assert _STAGE_LABELS[stage] in result


def test_directives_are_distinct_per_stage() -> None:
    """阶段之间的指令必须真的不同。

    ⚠️ 防的是复制粘贴：把 COLLECTING 的指令原样贴给 CONFIRMING，
    表里就多了一条永远不会被区分的分支，而测试若只断言「非空」是发现不了的。
    """
    assert len(set(_STAGE_DIRECTIVES.values())) == len(_STAGE_DIRECTIVES)


def test_labels_are_distinct_per_stage() -> None:
    """阶段展示名也必须互不相同。"""
    assert len(set(_STAGE_LABELS.values())) == len(_STAGE_LABELS)


# ---------------------------------------------------------------------------
# 2. 幂等性 —— 本模块最核心的保证
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stage", list(TripStage))
def test_build_is_idempotent(stage: TripStage) -> None:
    """重复组装得到**完全相同**的结果，不会累积。

    ⚠️ ``on_system_prompt`` 每一轮推理都会被调用，且 ``current_prompt``
    是前序中间件的输出。若本函数不幂等，动态段落会逐轮翻倍，几轮之后
    prompt 被撑爆 —— 症状是「聊几句就报上下文超限」，而根因极难猜。
    """
    once = build_system_prompt(BASE, stage=stage, request=FULL_REQUEST)
    twice = build_system_prompt(once, stage=stage, request=FULL_REQUEST)
    thrice = build_system_prompt(twice, stage=stage, request=FULL_REQUEST)

    assert once == twice == thrice


def test_repeated_build_does_not_grow_the_prompt() -> None:
    """幂等性的量化版本：跑十轮，长度必须**逐字不变**。

    ⚠️ 与上面的相等断言重复，但这条能在失败时直接指出「膨胀了多少」，
    比一个巨大的字符串 diff 好读得多。
    """
    prompt = BASE
    lengths = []
    for _ in range(10):
        prompt = build_system_prompt(prompt, stage=TripStage.COLLECTING, request=FULL_REQUEST)
        lengths.append(len(prompt))

    assert len(set(lengths)) == 1, f"prompt 长度随轮次变化：{lengths}"


def test_idempotent_even_when_the_stage_changes() -> None:
    """换阶段重组装时，**旧阶段的段落被剥掉**，不会两个阶段并存。

    ⚠️ 若旧段落残留，模型会同时看到「还在收集」与「请确认」两条互相
    矛盾的指令，行为变得不可预测。
    """
    collecting = build_system_prompt(BASE, stage=TripStage.COLLECTING, request=FULL_REQUEST)
    confirming = build_system_prompt(collecting, stage=TripStage.CONFIRMING, request=FULL_REQUEST)

    assert _STAGE_LABELS[TripStage.CONFIRMING] in confirming
    assert _STAGE_LABELS[TripStage.COLLECTING] not in confirming
    assert confirming.count(MARKER_BEGIN) == 1


def test_idempotent_when_profile_changes() -> None:
    """画像内容变化时同样只保留一份。"""
    with_profile = build_system_prompt(BASE, stage=TripStage.IDLE, profile_summary="常去上海")
    without = build_system_prompt(with_profile, stage=TripStage.IDLE, profile_summary="")

    assert "常去上海" not in without
    assert without.count(MARKER_BEGIN) == 1


# ---------------------------------------------------------------------------
# 3. 剥离的边角
# ---------------------------------------------------------------------------


def test_strip_removes_a_wellformed_section() -> None:
    """成对标记被完整剥掉。"""
    assert strip_managed_sections(f"base\n\n{MARKER_BEGIN}\nxxx\n{MARKER_END}") == "base"


def test_strip_handles_multiple_sections() -> None:
    """多对标记全部剥掉 —— 万一真的累积了，剥离要能一次清干净。"""
    text = f"a\n{MARKER_BEGIN}\n1\n{MARKER_END}\nb\n{MARKER_BEGIN}\n2\n{MARKER_END}\nc"
    assert strip_managed_sections(text) == "a\n\nb\n\nc"


def test_strip_removes_an_unclosed_section_to_the_end() -> None:
    """未闭合的起始标记（有头无尾）剥到串尾。

    ⚠️ 若只按成对匹配剥离，残留的半个段落会**静默累积**，而且因为它不含
    结束标记，后续每一次剥离都拿它没办法。
    """
    assert strip_managed_sections(f"base\n{MARKER_BEGIN}\n没有结束标记") == "base"


def test_strip_is_a_noop_on_plain_text() -> None:
    """没有标记时原样返回（只做首尾空白清理）。"""
    assert strip_managed_sections("  普通文本  ") == "普通文本"


def test_strip_on_empty_string() -> None:
    """空串不炸。"""
    assert strip_managed_sections("") == ""


def test_strip_keeps_content_around_a_mid_string_section() -> None:
    """标记夹在中间时，前后内容都保留。"""
    text = f"前{MARKER_BEGIN}中{MARKER_END}后"
    assert strip_managed_sections(text) == "前后"


def test_build_strips_before_appending() -> None:
    """组装时先剥后加：传进来的 base 里已有的动态段落不会留下。"""
    dirty = build_system_prompt(BASE, stage=TripStage.IDLE)
    rebuilt = build_system_prompt(dirty, stage=TripStage.DONE)

    assert rebuilt.count(MARKER_BEGIN) == 1
    assert _STAGE_LABELS[TripStage.DONE] in rebuilt
    assert _STAGE_LABELS[TripStage.IDLE] not in rebuilt


# ---------------------------------------------------------------------------
# 4. 默认值不是「已确认」
# ---------------------------------------------------------------------------


def test_empty_request_lists_no_known_slots() -> None:
    """空请求不列出任何「已知要素」。"""
    assert describe_known_slots(TravelRequest()) == []


def test_default_values_are_not_reported_as_known() -> None:
    """默认值不得被当成用户已提供的信息。

    ⚠️ 这是本模块最要紧的一条产品判断。``travelers`` 默认 1，无法区分
    「用户说了我一个人」与「用户根本没提」。若把默认值也列成已知，模型会
    以为人数已确认而不再追问 —— 一旦用户其实是两个人，就成了**静默的
    错误**。宁可多问一句。
    """
    lines = describe_known_slots(
        TravelRequest(origin="杭州", destination="北京", days=2),
    )

    assert "同行人数" not in "\n".join(lines)
    assert "交通方式" not in "\n".join(lines)  # ANY
    assert "舱位等级" not in "\n".join(lines)  # ANY
    assert "酒店" not in "\n".join(lines)  # hotel_required=False
    assert "预算" not in "\n".join(lines)  # budget=0.0
    assert "出差事由" not in "\n".join(lines)


def test_any_transport_mode_is_treated_as_unspecified() -> None:
    """显式传 ``ANY`` 与不传等价 —— 都不算已确认。"""
    explicit = describe_known_slots(TravelRequest(transport_mode=TransportMode.ANY))
    assert explicit == []


def test_one_traveler_is_treated_as_unspecified() -> None:
    """``travelers=1`` 不报告（无法与默认值区分）。"""
    assert describe_known_slots(TravelRequest(travelers=1)) == []


def test_two_travelers_is_reported() -> None:
    """``travelers=2`` 是明确信息，必须报告。"""
    assert "同行人数：2 人" in describe_known_slots(TravelRequest(travelers=2))


def test_explicit_values_are_reported() -> None:
    """明确给出的值都要出现在「已知要素」里。"""
    lines = describe_known_slots(FULL_REQUEST)
    joined = "\n".join(lines)

    for expected in (
        "出发城市：杭州",
        "目的城市：北京",
        "出发日期：2026-10-08",
        "返程日期：2026-10-10",
        "出差天数：3 天",
        "交通方式：飞机",
        "舱位等级：经济舱",
        "需要预订酒店：是",
        "酒店区域偏好：国贸",
        "同行人数：2 人",
        "出差事由：客户拜访",
        "预算上限：5000 元",
    ):
        assert expected in joined, f"缺少：{expected}"


def test_every_transport_mode_has_a_label() -> None:
    """除 ``ANY`` 外每个交通方式都有中文标签。

    ⚠️ ``ANY`` 刻意没有标签 —— 它表示「未指定」，不该被展示。
    """
    assert set(_TRANSPORT_LABELS) == set(TransportMode) - {TransportMode.ANY}


def test_every_cabin_class_has_a_label() -> None:
    """除 ``ANY`` 外每个舱位等级都有中文标签。"""
    assert set(_CABIN_LABELS) == set(CabinClass) - {CabinClass.ANY}


def test_budget_renders_without_a_trailing_zero() -> None:
    """预算渲染成 ``5000`` 而不是 ``5000.0``（模型会原样照抄进正文）。"""
    assert "预算上限：5000 元" in describe_known_slots(TravelRequest(budget=5000.0))


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        (5000.0, "预算上限：5000 元"),
        (15000.0, "预算上限：15000 元"),
        (85000.5, "预算上限：85000.5 元"),
        # ⚠️ 下面三档是缺陷 P2 的现场：``f"{v:g}"`` 会给出
        # ``1.2e+06`` / ``1e+06`` / ``12345.7``（6 位有效数字 + 科学计数法）。
        (1200000.0, "预算上限：1200000 元"),
        (1234567.0, "预算上限：1234567 元"),
        (12345.678, "预算上限：12345.678 元"),
    ],
)
def test_budget_is_injected_verbatim(budget: float, expected: str) -> None:
    """★★ 预算必须**逐字**写进给模型的动态段落（缺陷 P2）。

    ⚠️ 这一节是模型「原样引用」的素材：提示词明写「数字照抄工具返回、不凑整」。
    渲染成科学计数法时，模型照抄会把用户读不懂的 ``1.2e+06 元`` 发给用户；
    而模型若按正常人写法写 ``1200000 元``，回复守卫的接地闸门又会因为
    **来源侧记的是科学计数法**而判它编造 —— 管道自己违反了它对模型提出的
    数字纪律。两处（这里与 ``src/orchestration/amounts.py``）共用同一个
    渲染函数，本用例守的是「确实用上了」。
    """
    lines = describe_known_slots(TravelRequest(budget=budget))
    joined = "\n".join(lines)

    assert expected in joined
    assert "e+" not in joined and "E+" not in joined, f"渲染成了科学计数法：{joined!r}"


def test_user_data_excerpt_carries_slots_and_profile() -> None:
    """``user_data_excerpt`` 摘出的正是「守卫可以豁免的数字」的来源文本。

    ⚠️ 它的每条已知要素必须与 ``describe_known_slots`` **逐字一致** ——
    守卫的豁免判据建立在「同一份渲染」上（见 ``reply_guard`` 的来源 4）。
    两处若各写一份措辞，症状是「同一段正确答复时对时错」。
    """
    from src.orchestration.prompt import user_data_excerpt

    excerpt = user_data_excerpt(FULL_REQUEST, "常年出差，平均每晚 480 元。")

    for line in describe_known_slots(FULL_REQUEST):
        assert line in excerpt, f"摘出的文本里缺少已知要素：{line}"
    assert "平均每晚 480 元" in excerpt, "长期画像里的数字没被摘出来"


def test_user_data_excerpt_is_empty_without_user_data() -> None:
    """没有用户数据时返回空串 —— 空串不会给守卫任何豁免。"""
    from src.orchestration.prompt import user_data_excerpt

    assert user_data_excerpt(None) == ""
    assert user_data_excerpt(TravelRequest()) == ""
    assert user_data_excerpt(None, "   ") == ""


def test_user_data_excerpt_ignores_the_static_directives() -> None:
    """★ 阶段指令这类静态文案**不进**摘要（缺陷 P1 的豁免范围）。

    ⚠️ 动态段落里除了用户数据，还有一小段固定指令（例如「一次最多追问
    两项」）。把整段都摘出来会让静态文案里的数字变成合法来源 ——
    闸门的豁免集合只该包含**用户说过的话**。这里用英文数字做探针：
    摘要里不该出现 ``MAX_MISSING_PROMPTS`` 那个 ``2``。
    """
    from src.orchestration.prompt import user_data_excerpt

    request = TravelRequest(destination="北京", budget=15000.0)  # 必填缺失
    excerpt = user_data_excerpt(request)

    assert "15000" in excerpt
    assert "最多" not in excerpt, "静态指令被摘进了豁免来源"


def test_slot_order_is_stable() -> None:
    """同一份请求重复调用，顺序完全一致（确定性要求）。"""
    first = describe_known_slots(FULL_REQUEST)
    second = describe_known_slots(FULL_REQUEST)
    assert first == second


# ---------------------------------------------------------------------------
# 5. 已知与待补齐不重叠
# ---------------------------------------------------------------------------


def test_missing_slots_are_listed_while_collecting() -> None:
    """COLLECTING 阶段列出待补齐要素。"""
    result = build_system_prompt(BASE, stage=TripStage.COLLECTING, request=TravelRequest())

    assert "## 待补齐要素" in result
    assert "出发城市" in result
    assert "目的城市" in result
    assert "出差时间" in result


def test_known_and_missing_never_overlap() -> None:
    """**同一个字段不能既在「已知」又在「待补齐」里。**

    ⚠️ 重叠意味着模型会重复追问用户已经回答过的问题 ——
    对话式收集最招人烦的失败模式，也是博客里点名要避免的。
    """
    request = TravelRequest(origin="杭州", destination="北京")  # 缺时间
    result = build_system_prompt(BASE, stage=TripStage.COLLECTING, request=request)

    known_part, missing_part = result.split("## 待补齐要素")

    assert "出发城市：杭州" in known_part
    assert "出差时间" in missing_part
    # 已知项不得出现在待补齐段里。
    assert "出发城市" not in missing_part
    assert "目的城市" not in missing_part


def test_at_most_max_missing_prompts_are_asked() -> None:
    """本轮追问项数不超过上限。"""
    result = build_system_prompt(BASE, stage=TripStage.COLLECTING, request=TravelRequest())

    ask_line = next(line for line in result.splitlines() if line.startswith("本轮请优先追问"))
    asked = ask_line.split("：", 1)[1]
    assert len(asked.split("、")) <= MAX_MISSING_PROMPTS


def test_remaining_missing_slots_are_still_named() -> None:
    """本轮不问的剩余项也要写出来。

    ⚠️ 不写的话，模型看到「本轮只问两项」却不知道剩余项是什么，
    可能自行脑补出并不缺的字段来追问。
    """
    result = build_system_prompt(BASE, stage=TripStage.COLLECTING, request=TravelRequest())
    assert "其余待补齐项" in result


def test_missing_section_is_absent_outside_collecting() -> None:
    """非 COLLECTING 阶段不列「待补齐要素」。

    ⚠️ 在 CONFIRMING 阶段列出待补齐项，会让模型倾向于**退回追问**，
    而不是先完成用户当下要的确认动作。
    """
    result = build_system_prompt(BASE, stage=TripStage.CONFIRMING, request=TravelRequest())
    assert "## 待补齐要素" not in result


def test_complete_request_has_no_missing_section() -> None:
    """要素齐全时 COLLECTING 段里也不该有「待补齐要素」。"""
    result = build_system_prompt(BASE, stage=TripStage.COLLECTING, request=FULL_REQUEST)
    assert "## 待补齐要素" not in result
    assert "## 已知要素" in result


def test_none_request_omits_slot_sections_entirely() -> None:
    """拿不到请求对象时不输出要素小节（但阶段指令照常）。"""
    result = build_system_prompt(BASE, stage=TripStage.COLLECTING, request=None)

    assert "## 已知要素" not in result
    assert "## 待补齐要素" not in result
    assert _STAGE_LABELS[TripStage.COLLECTING] in result


def test_directives_do_not_point_at_conditional_sections() -> None:
    """阶段指令**不得点名**那些条件输出的小节标题。

    ⚠️ 「已知要素」「待补齐要素」两节都是条件输出的：要素全齐时没有
    「待补齐」段，一个要素都没收集到时没有「已知」段。指令若写了
    「见上方『已知要素』」，模型就会去找一个不存在的东西 ——
    轻则忽略整条指令，重则凭空编一个字段来追问。

    这条断言把「不许引用小节标题」变成可执行的规则，而不是只写在注释里。
    """
    conditional_titles = ("已知要素", "待补齐要素")
    for stage, directive in _STAGE_DIRECTIVES.items():
        for title in conditional_titles:
            assert title not in directive, f"{stage} 的指令引用了条件小节「{title}」"


def test_plain_string_stage_behaves_like_the_enum() -> None:
    """传等值裸字符串与传枚举的结果**完全一致**。

    ⚠️ 阶段可能来自 session 元数据或请求体，未必是枚举实例。若两者行为
    不同（比如 ``is`` 比较让「待补齐要素」在裸字符串下静默消失），
    就会出现「线上传字符串、测试传枚举，测试全绿而线上不对」。
    """
    for stage in TripStage:
        as_enum = build_system_prompt(BASE, stage=stage, request=TravelRequest())
        as_str = build_system_prompt(BASE, stage=stage.value, request=TravelRequest())
        assert as_enum == as_str, f"{stage} 的枚举与字符串结果不一致"


# ---------------------------------------------------------------------------
# 6. 开关
# ---------------------------------------------------------------------------


def test_disabled_returns_the_bare_base_prompt() -> None:
    """关闭时原样返回基础 prompt，不含任何标记。"""
    result = build_system_prompt(BASE, stage=TripStage.COLLECTING, request=FULL_REQUEST, enabled=False)

    assert result == BASE
    assert MARKER_BEGIN not in result


def test_disabled_still_strips_a_previous_section() -> None:
    """关闭时**仍然执行剥离**，不会留下上一轮的动态段落。

    ⚠️ 半开半关比全开或全关都难排查：这个开关的用途是「怀疑动态 Prompt
    导致行为异常时一键关掉」，它必须关得干净。
    """
    dirty = build_system_prompt(BASE, stage=TripStage.COLLECTING, request=FULL_REQUEST)
    result = build_system_prompt(dirty, stage=TripStage.COLLECTING, enabled=False)

    assert result == BASE
    assert MARKER_BEGIN not in result


def test_disabled_ignores_profile_too() -> None:
    """关闭时画像也不注入。"""
    result = build_system_prompt(BASE, profile_summary="常去上海", enabled=False)
    assert "常去上海" not in result


def test_enabled_is_the_default() -> None:
    """默认开启 —— 调用方不该为了拿到功能而多传一个参数。"""
    assert MARKER_BEGIN in build_system_prompt(BASE, stage=TripStage.IDLE)


# ---------------------------------------------------------------------------
# 7. 确定性与不抛异常
# ---------------------------------------------------------------------------


def test_output_is_deterministic() -> None:
    """同入参同输出，重复一百次也一样。"""
    once = build_system_prompt(BASE, stage=TripStage.COLLECTING, request=FULL_REQUEST)
    for _ in range(100):
        assert build_system_prompt(BASE, stage=TripStage.COLLECTING, request=FULL_REQUEST) == once


def test_empty_base_prompt_does_not_leave_a_leading_blank() -> None:
    """基础 prompt 为空时，结果不以空行开头。"""
    result = build_system_prompt("", stage=TripStage.IDLE)
    assert result.startswith(MARKER_BEGIN)


def test_whitespace_only_base_prompt_behaves_like_empty() -> None:
    """纯空白的基础 prompt 同上。"""
    result = build_system_prompt("   \n\n  ", stage=TripStage.IDLE)
    assert result.startswith(MARKER_BEGIN)


def test_base_prompt_is_preserved_verbatim() -> None:
    """基础 prompt 的内容**逐字**保留，不被改写或压缩。"""
    base = "# 角色\n\n你是差旅助手。\n\n## 纪律\n\n1. 不编造价格\n2. 不越权下单\n"
    result = build_system_prompt(base, stage=TripStage.IDLE)
    assert base.strip() in result


def test_malformed_marker_in_base_does_not_raise() -> None:
    """基础 prompt 里混进半个标记时不抛异常。

    ⚠️ 本函数跑在 agent 主链路上，抛异常会让整轮回复失败 ——
    代价与「prompt 少了一段」完全不成比例。
    """
    result = build_system_prompt(f"{BASE}\n{MARKER_BEGIN}\n没有结束标记", stage=TripStage.IDLE)
    assert MARKER_END in result


def test_build_accepts_an_unknown_stage_gracefully() -> None:
    """阶段值不在枚举里时降级，而不是抛异常。"""
    result = build_system_prompt(BASE, stage="NOT_A_REAL_STAGE")  # type: ignore[arg-type]
    assert BASE in result
    assert "NOT_A_REAL_STAGE" in result  # 至少把阶段名报出来，便于排障


# ---------------------------------------------------------------------------
# 8. 不泄漏开发者信息
# ---------------------------------------------------------------------------


def test_output_contains_no_developer_markers() -> None:
    """产出的 prompt 里不得出现开发者注释标记或源码路径。

    ⚠️ 与本项目 ``src/domain/_doc.py`` 处理的**同一个问题**：本项目要求
    注释详尽（大量 ⚠️ 与「为什么」），而 prompt 是要发给模型的。文档写得
    越细，越容易顺手抄进 prompt，把「给维护者看的理由」当成「给模型的
    指令」发出去 —— 既浪费 token，也会让模型对着自相矛盾的说明推理。
    """
    result = build_system_prompt(BASE, stage=TripStage.COLLECTING, request=FULL_REQUEST)

    for leak in ("⚠️", "src/", "tests/", "_STAGE_DIRECTIVES", "TODO"):
        assert leak not in result, f"prompt 里泄漏了开发者信息：{leak}"


def test_output_contains_no_markdown_code_fences() -> None:
    """产出的 prompt 里不得出现代码围栏 —— 同上，是文档泄漏的典型形态。"""
    result = build_system_prompt(BASE, stage=TripStage.CONFIRMING, request=FULL_REQUEST)
    assert "```" not in result


# ---------------------------------------------------------------------------
# 9. 纯度
# ---------------------------------------------------------------------------


def test_prompt_module_does_not_import_the_framework() -> None:
    """纯度探针：import 本模块**不得**把 ``agentscope`` 拉进来。

    ⚠️ 必须在**子进程**里验证。在进程内断言 ``"agentscope" not in sys.modules``
    是无效的 —— ``conftest.py`` 早已把框架导入了，断言必然假通过。

    这条测试守的是 ``src/orchestration/__init__.py`` 的设计：一旦有人把
    ``src.orchestration.context``（需要框架）加进包的 ``__init__``，
    动态 Prompt 的纯逻辑测试就会被迫依赖框架，这里会立刻失败。
    """
    import subprocess
    import sys

    probes = [
        "import sys; import src.orchestration; "
        "assert 'agentscope' not in sys.modules, 'src.orchestration 把框架拉进来了'; "
        "print('PURE')",
        "import sys; from src.orchestration.prompt import build_system_prompt; "
        "assert 'agentscope' not in sys.modules, 'prompt 模块把框架拉进来了'; "
        "print('PURE')",
        "import sys; from src.orchestration import build_system_prompt; "
        "assert 'agentscope' not in sys.modules; print('PURE')",
    ]
    for code in probes:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "PURE", result.stdout
