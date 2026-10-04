# -*- coding: utf-8 -*-
"""``src/domain`` 的**枚举与结构化模型**测试。

覆盖三件事，每件都对应一类真实会发生的缺陷：

1. **枚举的展示名完整性** —— 新增一个 :class:`Intent` 成员却忘了加中文名，
   用户就会在界面上看到 ``PLAN_TRIP``。这类漏写在代码审查里几乎看不出来
   （新增枚举时注意力都在业务语义上）。
2. **JSON Schema 不被开发注释污染** —— 这是本项目特有的风险：文档写得越详尽，
   泄漏进提示词的噪声越多。见 :func:`test_enum_docstrings_do_not_leak_into_json_schema`。
3. **字段合并不丢数据** —— :meth:`TravelRequest.merged_with` 是「多轮收集」
   的核心，它出错的表现是「用户说过的信息莫名消失」，属于最难复现的一类 bug。
"""

from __future__ import annotations

import json

import pytest

from src.domain import (
    AgentName,
    CabinClass,
    Intent,
    IntentDecision,
    IntentRecognitionResult,
    RouteDecision,
    TransportMode,
    TravelRequest,
    TripStage,
)


# ==============================================================================
# 一、枚举的展示名
# ==============================================================================
@pytest.mark.parametrize(("enum_cls", "table_name"), [(Intent, "_INTENT_DISPLAY_NAMES"), (AgentName, "_AGENT_DISPLAY_NAMES")])
def test_every_member_has_a_chinese_display_name(enum_cls: type, table_name: str) -> None:
    """★ 每个需要展示的枚举成员都必须有中文展示名。

    ⚠️ 这条守的是「新增成员忘了登记中文名」。漏掉时的表现**不是报错**：
    ``display_name`` 会回落到枚举值本身，界面于是显示 ``PLAN_TRIP`` 或
    ``policy_rag`` —— 功能全对，只是用户看不懂。正因为不报错，
    才需要一个遍历所有成员的用例来钉住它。

    ⚠️ 两个枚举一起参数化而不是各写一份：这条断言的逻辑完全相同，复制一份的
    后果是下次给第三个枚举加 display_name 时，只会照抄一份、漏掉一处。
    """
    for member in enum_cls:
        name = member.display_name
        assert name != member.value, (
            f"{enum_cls.__name__}.{member.name} 没有登记中文展示名，"
            f"界面会直接显示英文枚举值。请在 src/domain/enums.py 的 "
            f"{table_name} 里补一条。"
        )


# ==============================================================================
# 二、JSON Schema 的干净程度
# ==============================================================================
@pytest.mark.parametrize(
    "model",
    [IntentRecognitionResult, TravelRequest, RouteDecision],
)
def test_enum_docstrings_do_not_leak_into_json_schema(model: type) -> None:
    """★ 开发注释不得进入交给大模型的 JSON Schema。

    这是本项目的一个**结构性风险**：文档写得越详细（这是项目要求），
    泄漏的噪声就越多。而 schema 里的 ``description`` 在模型眼里是**指令**，
    不是注释 —— 一句写给开发者的「⚠️ OTHER 必须由代码兜底」，会被模型
    读成「多用 OTHER」。

    ⚠️ 判据用**噪声特征**而不是长度阈值：长度阈值会随着正常内容增长而误报，
    而「⚠️」「``」「file:line」这些标记出现在 schema 里一定是泄漏 ——
    它们是本项目注释的排版约定，没有任何一条是写给模型的。

    ⚠️ 另外必须确认 pydantic 真的**调用了**我们的覆盖钩子：若钩子因为
    pydantic 升级而失效，description 会退回整段 docstring，而只测「有没有
    ⚠️」会漏掉「第一行恰好没有 ⚠️」的情形。下面的 ``assert`` 同时覆盖两者。
    """
    schema_text = json.dumps(model.model_json_schema(), ensure_ascii=False)

    for marker in ("⚠️", "``", "src/domain"):
        assert marker not in schema_text, (
            f"{model.__name__} 的 JSON Schema 里出现了开发注释标记 {marker!r}，"
            f"说明枚举的 __doc__ 泄漏进了交给模型的提示词。"
        )


def test_enum_schema_description_is_a_single_line() -> None:
    """枚举在 schema 里的描述**只有一行**（不是整段 docstring）。

    与上一条互补：上一条查「有没有泄漏特征」，这一条查「是不是被压成了一行」。
    两条都需要 —— 一段不含 ⚠️ 的长文档同样会稀释提示词。
    """
    schema = IntentRecognitionResult.model_json_schema()
    intent_schema = schema["$defs"]["Intent"]

    description = intent_schema.get("description", "")
    assert "\n" not in description, f"枚举描述应当是单行，实际是：{description!r}"
    assert len(description) <= 60, (
        f"枚举描述过长（{len(description)} 字），说明压缩逻辑没生效：{description!r}"
    )


# ==============================================================================
# 三、事项收集与合并
# ==============================================================================
def test_missing_required_lists_human_readable_field_names() -> None:
    """缺字段时要返回**中文名**，而不是字段的英文标识。

    ⚠️ 返回值会被直接拼进追问话术（「还需要您补充：出发城市、出差时间」）。
    返回 ``["origin", "days"]`` 的话，这句话就得由调用方再翻译一次 ——
    而调用方迟早会漏掉某个新字段，于是用户看到英文。
    """
    request = TravelRequest()

    missing = request.missing_required()

    assert missing == ["出发城市", "目的城市", "出差时间"]
    assert request.is_complete() is False


def test_date_or_days_satisfies_the_time_requirement() -> None:
    """★ 日期与天数**满足其一**即算时间已给。

    ⚠️ 这条防的是「追问用户已经回答过的问题」：用户说「下周去北京三天」
    （只有天数），系统若坚持要日期，会再问一遍时间 —— 这是对话式收集
    最招人烦的失败模式，而且在开发期很难发现（开发者自己测试时总是
    两个字段都填）。
    """
    with_days = TravelRequest(origin="杭州", destination="北京", days=3)
    with_date = TravelRequest(origin="杭州", destination="北京", depart_date="2026-03-05")

    assert with_days.is_complete() is True
    assert with_date.is_complete() is True
    assert "出差时间" not in with_days.missing_required()


def test_merge_keeps_fields_the_user_did_not_mention() -> None:
    """★ 增量合并不得清空用户此前已经说过的字段。

    场景：第一轮收集到「杭州 → 北京，3 天」，第二轮用户只说「改成上海」。
    这一轮的增量里 ``origin`` 与 ``days`` 都是空值，若按整体覆盖，
    前两轮的信息会一起消失。

    ⚠️ 这类 bug 在真实对话里表现为「聊了几轮之后信息丢了」，用户会以为
    系统没听懂；而在单测里只要构造一个「只带一个字段的 patch」就能钉死。
    """
    collected = TravelRequest(origin="杭州", destination="北京", days=3)

    merged = collected.merged_with(TravelRequest(destination="上海"))

    assert merged.destination == "上海"
    assert merged.origin == "杭州", "用户没说出发地，不该被空值覆盖掉"
    assert merged.days == 3, "用户没说天数，不该被 0 覆盖掉"


def test_merge_does_not_let_any_enum_overwrite_an_explicit_choice() -> None:
    """★★ ``ANY`` 是「有值的未指定」，**不得**覆盖用户明确选过的偏好。

    ⚠️ 这是 :meth:`TravelRequest.merged_with` 最容易写错的一处：
    ``TransportMode.ANY`` 不是空串，按「非空即覆盖」的朴素实现，
    它会覆盖掉用户上一轮明确说的「要坐高铁」。症状是**偏好莫名其妙失效**，
    而且因为 ``ANY`` 看起来无害，排查时几乎不会怀疑到合并逻辑上。

    反向也要成立：用户这一轮**真的**改口说「都可以」，就该覆盖成功。
    """
    collected = TravelRequest(transport_mode=TransportMode.TRAIN)

    kept = collected.merged_with(TravelRequest())
    assert kept.transport_mode == TransportMode.TRAIN, (
        "空 patch 里的 ANY 覆盖了用户明确选择的 TRAIN"
    )

    changed = collected.merged_with(TravelRequest(transport_mode=TransportMode.FLIGHT))
    assert changed.transport_mode == TransportMode.FLIGHT


def test_merge_lets_zero_override_but_not_false() -> None:
    """``0`` 是「未指定」，但 ``False`` 是**有效值**。

    ⚠️ 两者都写成 ``== 0`` 会一起命中 —— ``False == 0`` 在 Python 里为真，
    且 ``bool`` 是 ``int`` 的子类。于是「用户明确说不要订酒店」
    （``hotel_required=False``）会被当成「没提」，从而保留上一轮的 ``True``。
    结果是系统**反着做**用户的要求，而这在界面上只表现为「酒店没去掉」。
    """
    collected = TravelRequest(hotel_required=True, budget=2000.0)

    merged = collected.merged_with(TravelRequest(hotel_required=False))

    assert merged.hotel_required is False, "用户明确说不要酒店，不能被当成「没提」"
    assert merged.budget == 2000.0, "patch 里 budget 是 0，不该覆盖已有预算"


def test_merge_always_takes_the_latest_stage() -> None:
    """阶段（stage）永远以最新为准，即使它是默认值。

    ⚠️ 阶段不是「收集到的要素」，而是**状态机的游标**。若它也走
    「非空才覆盖」，用户从 :attr:`TripStage.CONFIRMING` 回退到
    :attr:`TripStage.IDLE`（例如说「算了重新来」）时，游标会卡在 CONFIRMING
    —— 动态 Prompt 于是继续按「等确认」组装，用户发现怎么说话系统都在催确认。
    """
    collected = TravelRequest(stage=TripStage.CONFIRMING)

    merged = collected.merged_with(TravelRequest(stage=TripStage.IDLE))

    assert merged.stage == TripStage.IDLE


# ==============================================================================
# 四、意图识别结果
# ==============================================================================
def test_top_intent_picks_by_confidence_not_by_position() -> None:
    """★ 取最高置信度意图时**按值取**，不按列表位置取。

    ⚠️ Prompt 里要求模型「按置信度从高到低排列」，但**不能依赖模型遵守格式
    约定**来做控制流：它只要有一次没排序，按位置取就会稳定地取到错的意图，
    而且看起来像是「模型判断错了」，排查方向会被完全带偏。
    这里的输入刻意把低置信度放在前面 —— 正是模型没排序时的样子。
    """
    result = IntentRecognitionResult(
        intents=[
            IntentDecision(intent=Intent.CHITCHAT, confidence=0.2),
            IntentDecision(intent=Intent.PLAN_TRIP, confidence=0.9),
        ],
    )

    assert result.top_intent() == Intent.PLAN_TRIP


def test_top_intent_is_none_when_model_returns_nothing() -> None:
    """没有识别出任何意图时返回 ``None``，而不是抛异常。

    ⚠️ 这是**模型输出的常见形态**（尤其在它认为输入无关紧要时），
    编排层必须能把它当成一种正常输入来处理并追问，而不是 500。
    """
    assert IntentRecognitionResult().top_intent() is None


def test_extra_fields_from_the_model_are_ignored_not_fatal() -> None:
    """模型多吐的字段被**忽略**，不让整轮结构化输出报废。

    ⚠️ 这与 ``src/config/schema.py`` 的 ``extra="forbid"`` 是**刻意的相反
    选择**：配置里多一个键说明人写错了，必须拦；模型输出里多一个键说明
    模型自由发挥了，拦下来只会降低可用性。同一条规则用在不同地方是对是错，
    取决于「谁在写、写错要付什么代价」。
    """
    result = IntentRecognitionResult.model_validate(
        {
            "reasoning": "用户想去北京",
            "intents": [{"intent": "PLAN_TRIP", "confidence": 0.8, "mood": "开心"}],
            "model_invented_field": 42,
        },
    )

    assert result.top_intent() == Intent.PLAN_TRIP
    assert not hasattr(result, "model_invented_field")


def test_slots_are_plain_strings() -> None:
    """``slots`` 的值一律是字符串。

    ⚠️ 这条守的是「嵌套对象」这个诱惑：嵌套结构是结构化输出里最容易
    让模型「填一半」的形状（漏了内层必填字段 → 整条决策校验失败）。
    字符串足够承载抽取到的信息，类型转换交给 :class:`TravelRequest`。

    ⚠️ 注意 pydantic 在 ``dict[str, str]`` 上**不做强制转换**：给一个整数
    会直接 ValidationError，而不是被 str() 掉。所以这条断言同时确认了
    「声明是 str」与「非法类型会被拦」。
    """
    decision = IntentDecision(intent=Intent.PLAN_TRIP, slots={"destination": "北京"})

    assert decision.slots == {"destination": "北京"}

    with pytest.raises(Exception):
        IntentDecision(intent=Intent.PLAN_TRIP, slots={"travelers": 3})


def test_confidence_out_of_range_is_rejected() -> None:
    """置信度超出 0~1 会被拦下，而不是被静默接受。

    ⚠️ 模型偶尔会输出百分比（``85``）而不是小数。若不设范围，这个值会
    一路传到编排层的阈值比较里 —— ``85 > 0.7`` 恒真，于是**任何**意图都
    被判为高置信度，阈值形同虚设。宁可在这里报错（可被
    ``IntentRecognitionResult`` 的调用方捕获重试），也不要让一个错误的
    数值悄悄改变控制流。
    """
    with pytest.raises(Exception):
        IntentDecision(intent=Intent.PLAN_TRIP, confidence=85.0)


def test_cabin_and_transport_enums_expose_the_unspecified_member() -> None:
    """两个偏好枚举都必须有 ``ANY``，且它是默认值。

    ⚠️ 用 ``None`` 表达「未指定」的问题是：结构化输出里模型会直接**省略**
    该字段，于是下游分不清「用户说随便」与「模型忘了填」。显式的 ``ANY``
    让这两种情况可区分，也让合并逻辑（见上文）有一个明确的判据。
    """
    assert CabinClass.ANY in CabinClass
    assert TransportMode.ANY in TransportMode
    assert TravelRequest().transport_mode == TransportMode.ANY
    assert TravelRequest().cabin_class == CabinClass.ANY
