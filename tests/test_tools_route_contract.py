# -*- coding: utf-8 -*-
"""``LaneRouterMiddleware`` ⇄ ``aligo_route_intent`` 的**跨模块契约**测试。

═══ 为什么这个契约需要专门的测试文件 ═══

快车道在 ``on_model_call`` 里手工拼一段 JSON 当作「模型决定调用路由工具」的
输出，框架再拿这段 JSON 去调 :func:`~src.tools.route.aligo_route_intent`。
这条链上有三个**必须逐字一致**的名字：

1. 工具名 —— ``lane.DEFAULT_ROUTE_TOOL`` ⇄ ``route.ROUTE_TOOL_NAME``
2. 入参字段名 —— ``lane.ROUTE_TOOL_INPUT_FIELDS`` ⇄ 工具函数的形参名
3. 参数类型 —— 拼进去的是 ``list[str]``，形参也得接受 ``list[str]``

三处**对不上时全都不会报错**：

- 工具名对不上 → 框架抛 ``ToolNotFoundError``，但它被当成一次普通的
  工具失败结果塞回给模型（实测以 ``TOOL_RESULT_TEXT_DELTA`` 的形式出现），
  日志里只有一条不起眼的工具失败；
- 字段名对不上 → ``got an unexpected keyword argument``，同样变成一条
  工具失败结果；
- 类型对不上 → pydantic 校验失败，还是同一条路径。

三种情况的**表现完全一样**：快车道「看起来跑通了」，但路由信息一个字都
没传进模型。用户点「规划行程」，助手回一句「请问有什么可以帮您」。

所以这里不测行为，测**名字**。一条断言把两边钉死，让静默不一致变成红灯。

⚠️ 本文件与 ``test_orchestration_lane.py`` 的分工：那个文件测「快车道
真的省了一次模型调用」，本文件测「省下来的那次调用传对了东西」。
两者都必要 —— 只测后者，中间件可能压根没短路；只测前者，短路了但传空。
"""

from __future__ import annotations

import asyncio
import json

import pytest
from agentscope.message import TextBlock
from agentscope.tool import FunctionTool

from src.domain import AgentName, Intent
from src.orchestration.classifier import (
    FAST_LANE_RULES,
    classify,
    target_agents_for,
)
from src.orchestration.lane import (
    DEFAULT_ROUTE_TOOL,
    ROUTE_TOOL_INPUT_FIELDS,
    LaneRouterMiddleware,
)
from src.tools.route import ROUTE_TOOL_NAME, _NEXT_STEP_HINTS, aligo_route_intent

# ---------------------------------------------------------------------------
# 一、工具名与入参字段：两条名字契约
# ---------------------------------------------------------------------------


def _route_tool() -> FunctionTool:
    """构造路由工具的 ``FunctionTool`` 包装（与生产装配同款）。"""
    return FunctionTool(aligo_route_intent, is_read_only=True)


def test_tool_name_matches_between_lane_and_route_module() -> None:
    """契约 1：中间件合成的工具名 == 工具注册表里的名字。"""
    assert DEFAULT_ROUTE_TOOL == ROUTE_TOOL_NAME


def test_tool_name_matches_the_wrapped_function_tool() -> None:
    """契约 1（续）：常量与 ``FunctionTool`` 实际注册的名字一致。

    ⚠️ 这条不能省。``FunctionTool`` 默认用**函数名**当工具名，
    而 ``aligo_route_intent`` 的函数名恰好等于 ``ROUTE_TOOL_NAME`` ——
    这是巧合，不是保证。哪天有人把函数名改成 ``route_intent``
    （比如为了少打几个字），``ROUTE_TOOL_NAME`` 常量不会跟着变，
    而工具表里注册的就成了 ``route_intent``：快车道找不到它。
    """
    assert _route_tool().name == DEFAULT_ROUTE_TOOL


def test_input_field_names_match_the_function_signature() -> None:
    """契约 2：中间件拼的 JSON 键 == 工具函数的形参名（顺序也一致）。"""
    schema = _route_tool().input_schema
    assert tuple(schema.get("properties", {})) == ROUTE_TOOL_INPUT_FIELDS


def test_required_field_matches_between_the_two_sides() -> None:
    """契约 2（续）：必填字段两边一致。

    ⚠️ 中间件**总是**把四个字段全填上，所以必填与否不影响快车道。
    但慢车道上模型可能只给 ``intent`` —— 那时必填集合就决定了工具是否被调用。
    把它钉住，是为了让「改必填项」这件事必须同时改测试。
    """
    assert _route_tool().input_schema.get("required") == ["intent"]


def test_synthesized_payload_keys_are_exactly_the_declared_fields() -> None:
    """契约 3（最强的一条）：真的合成一次，逐字比对键集合。

    前面几条比对的是**常量**。这条跑一遍 :meth:`_synthesize`，从它实际
    产出的 ``ToolCallBlock.input`` 里把 JSON 解出来看键 —— 常量对得上、
    而拼 JSON 的代码写了别的键，是这类契约最常见的坏法。
    """
    from src.orchestration.classifier import classify

    decision = classify("规划行程")
    response = LaneRouterMiddleware()._synthesize(decision)  # noqa: SLF001

    block = response.content[0]
    assert block.type == "tool_call"
    assert block.name == DEFAULT_ROUTE_TOOL
    assert tuple(json.loads(block.input)) == ROUTE_TOOL_INPUT_FIELDS


def test_synthesized_payload_can_be_bound_to_the_real_function() -> None:
    """契约 3（续）：把合成出来的 JSON **真的**喂给工具函数，不报错。

    这是端到端的那一步：不比对名字，直接调用。名字全对但类型不对
    （比如把 ``target_agents`` 拼成字符串而不是 list）会在这里暴露。
    """
    from src.orchestration.classifier import classify

    decision = classify("规划行程")
    response = LaneRouterMiddleware()._synthesize(decision)  # noqa: SLF001
    payload = json.loads(response.content[0].input)

    chunk = aligo_route_intent(**payload)

    assert chunk.content, "工具必须返回非空内容"
    text = "".join(getattr(b, "text", "") for b in chunk.content)
    assert "规划行程" in text


def test_synthesized_payload_keeps_chinese_readable() -> None:
    """⚠️ 合成的 JSON **不能**转义中文。

    工具会把这段内容展示给用户与模型。``json.dumps`` 默认
    ``ensure_ascii=True``，中文会变成 ``\\u89c4\\u5212`` 这样的转义串 ——
    模型还能猜，用户看到的却是一串乱码。
    """
    from src.orchestration.classifier import classify

    decision = classify("规划行程")
    response = LaneRouterMiddleware()._synthesize(decision)  # noqa: SLF001

    assert "\\u" not in response.content[0].input


def test_synthesized_tool_call_is_not_pre_completed() -> None:
    """⚠️ 合成调用的状态是 ``PENDING``，不是「已完成」。

    预先标成完成会跳过 ``on_check_permission`` —— 也就是绕过 HITL。
    快车道省的是**路由**那次模型调用，不是权限校验。
    """
    from src.orchestration.classifier import classify

    decision = classify("规划行程")
    response = LaneRouterMiddleware()._synthesize(decision)  # noqa: SLF001

    assert response.content[0].state.value == "pending"
    assert response.is_last is True


# ---------------------------------------------------------------------------
# 二、路由工具自身的行为
# ---------------------------------------------------------------------------


def test_every_intent_has_a_next_step_hint() -> None:
    """⚠️ 每个 :class:`Intent` 都必须在 ``_NEXT_STEP_HINTS`` 里有登记。

    漏一个的后果不是报错，而是模型收到一句**空指引**，行为退化成
    「自己看着办」—— 而快车道存在的意义正是「别让模型自己看着办」。
    这里遍历枚举而不是遍历字典：遍历字典只能证明「表里的项自己没问题」，
    证明不了「没有漏项」，而漏项才是这里唯一的风险。
    """
    missing = [intent for intent in Intent if intent not in _NEXT_STEP_HINTS]
    assert not missing, f"这些意图没有登记下一步指引：{missing}"


def test_next_step_hints_are_non_empty_chinese() -> None:
    """指引文案非空且是中文 —— 它是给模型和用户看的。"""
    for intent, hint in _NEXT_STEP_HINTS.items():
        assert hint.strip(), f"{intent} 的指引是空的"
        assert any("一" <= ch <= "鿿" for ch in hint), f"{intent} 的指引不含中文：{hint!r}"


def test_next_step_hints_have_no_developer_markers() -> None:
    """指引不得泄漏开发口吻或内部路径。

    ⚠️ 与 ``test_orchestration_prompt.py`` 里的同类断言一致：
    这段文字会**原样**进入模型上下文，进而影响用户看到的回复。
    """
    forbidden = ("⚠️", "src/", "tests/", "TODO", "_NEXT_STEP_HINTS")
    for intent, hint in _NEXT_STEP_HINTS.items():
        for token in forbidden:
            assert token not in hint, f"{intent} 的指引里出现了 {token!r}：{hint!r}"


def test_unknown_intent_returns_an_error_chunk_instead_of_raising() -> None:
    """⚠️ 非法意图返回中文错误块，**不抛异常**。

    抛出的异常会被框架吞成一段英文错误文本交给模型
    （``ValueError: 'FOO' is not a valid Intent``）；返回中文说明则能让模型
    知道「路由信息坏了，请自己判断」并继续把这轮回复做完 —— 对用户是
    可恢复的。这条差别在故障时的体感差距很大。
    """
    chunk = aligo_route_intent(intent="NOT_AN_INTENT")

    text = "".join(getattr(b, "text", "") for b in chunk.content)
    assert "路由信息有误" in text


def test_route_tool_is_pure_and_needs_no_io() -> None:
    """路由工具是纯函数：同样入参 → 同样结果。"""
    first = aligo_route_intent(intent="PLAN_TRIP", matched_rule="plan_trip")
    second = aligo_route_intent(intent="PLAN_TRIP", matched_rule="plan_trip")

    assert [b.text for b in first.content] == [b.text for b in second.content]


def test_unknown_target_agents_do_not_break_the_tool() -> None:
    """⚠️ 未注册的目标智能体只记日志，不影响返回。

    路由表是静态的、智能体注册表是随装配变化的，两者短暂不一致很正常。
    把它当致命错误会让一次正常的点击变成报错。
    """
    chunk = aligo_route_intent(
        intent="PLAN_TRIP",
        target_agents=["main_plan", "not_a_real_agent"],
    )

    text = "".join(getattr(b, "text", "") for b in chunk.content)
    assert "规划行程" in text


def test_route_tool_card_contains_the_decision() -> None:
    """卡片里带上完整决策，供前端渲染。"""
    chunk = aligo_route_intent(
        intent="QUERY_ORDER",
        matched_rule="query_order",
        target_agents=["order_query"],
        reason="命中查询订单规则",
    )

    payload = json.loads("".join(getattr(b, "text", "") for b in chunk.content))
    assert payload["ok"] is True
    assert payload["card"] == "route_decision"

    item = payload["items"][0]
    assert item["intent"] == "QUERY_ORDER"
    assert item["matched_rule"] == "query_order"
    assert item["target_agents"] == ["order_query"]
    assert item["reason"] == "命中查询订单规则"


# ---------------------------------------------------------------------------
# 三、快车道规则表 ⇄ 智能体注册表
# ---------------------------------------------------------------------------


def test_every_intent_targets_registered_agents() -> None:
    """⚠️ 意图 → 目标智能体这张表里的名字必须真的在 ``AgentName`` 里。

    ``AgentName`` 是**运行时**枚举，而这张表是静态写死的。表里指向一个
    不存在或改过名的智能体时，路由决策会带一个查不到的名字 ——
    而 ``aligo_route_intent`` 对未知名字只记日志不报错（这是对的，
    见 :func:`~src.tools.route.aligo_route_intent`），于是错误就这么静静地
    流过去了。这条断言把它拦在测试期。

    ⚠️ 遍历 ``Intent`` 而不是遍历 ``FAST_LANE_RULES``：目标智能体是从
    **意图**查出来的（快慢两条车道共用同一张表），与规则无关。遍历规则
    只能覆盖「有规则的那些意图」，漏掉的恰好是最需要兜底的那几个。
    """
    known = {a.value for a in AgentName}
    for intent in Intent:
        targets = target_agents_for(intent)
        assert targets, f"意图 {intent.value} 没有对应的智能体"
        for target in targets:
            assert target in known, f"意图 {intent.value} 指向未注册的智能体 {target!r}"


def test_target_agents_for_returns_a_fresh_list() -> None:
    """⚠️ ``target_agents_for`` 每次返回**新**列表。

    返回共享对象的话，某个调用方「顺手 append 一下」就会**永久**污染
    全局路由表 —— 而那种 bug 的表现是「跑了一段时间之后路由开始出错」，
    排查方向会完全跑偏。
    """
    first = target_agents_for(Intent.PLAN_TRIP)
    first.append("污染")

    assert "污染" not in target_agents_for(Intent.PLAN_TRIP)


def test_fast_lane_rules_have_unique_names() -> None:
    """规则名唯一 —— 它会被写进合成调用的 id 与日志，重名就分不清了。"""
    names = [rule.name for rule in FAST_LANE_RULES]
    assert len(names) == len(set(names)), f"规则名重复：{names}"


def test_every_fast_lane_rule_produces_a_bindable_decision() -> None:
    """⚠️ 每条规则的**每一条短语**都能被绑定到路由工具上。

    ⚠️ 遍历 ``phrases`` 而不是只取第一条。规则表里最容易出的错是**死短语**
    （未归一化的写法永远匹配不上 —— 见 ``FAST_LANE_RULES`` 的说明），
    只测第一条会漏掉后面那些。
    """
    for rule in FAST_LANE_RULES:
        for phrase in rule.phrases:
            decision = classify(phrase)
            assert decision.lane.value == "FAST", f"规则 {rule.name} 的短语 {phrase!r} 没命中快车道"
            assert decision.matched_rule == rule.name

            chunk = aligo_route_intent(
                intent=decision.intent.value,
                matched_rule=decision.matched_rule,
                target_agents=list(decision.target_agents),
                reason=decision.reason,
            )
            assert chunk.content, f"规则 {rule.name} 产出的决策无法被工具接受"


@pytest.mark.parametrize("intent", list(Intent))
def test_every_intent_is_bindable(intent: Intent) -> None:
    """每个意图都能绑定到路由工具上。"""
    chunk = aligo_route_intent(intent=intent.value)
    text = "".join(getattr(b, "text", "") for b in chunk.content)
    assert intent.display_name in text


def test_text_block_type_is_what_the_tool_returns() -> None:
    """工具返回的是 ``TextBlock`` 列表 —— 前端与框架都按这个类型解析。"""
    chunk = aligo_route_intent(intent="CHITCHAT")

    assert all(isinstance(b, TextBlock) for b in chunk.content)
