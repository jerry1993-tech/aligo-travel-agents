# -*- coding: utf-8 -*-
"""差旅查询工具（交通 / 酒店 / 差标）的测试。

═══ 本文件守的是什么 ═══

查询类工具有两类失败**看起来都是「查询失败」，处理方式却完全相反**：

- **「这条线路没有班次」是业务结论** —— 用户应当被告知结论并换个条件，
  不该看到报错样式，更不该反复重试；
- **「仓储抛异常了」是系统故障** —— 应当给一句「稍后重试」并留下排查线索。

把两者混成一种，用户就会「对着一个正常结论反复重试」，或者「对一次真实
故障毫无察觉」。本文件用两组用例把这条界线钉住。

⚠️ 另有一组用例守 ``is_read_only=True``：少了它，用户每点一次「查订单」
都要在确认弹窗里点一次「允许」。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import pytest
from agentscope.message import ToolResultState

from src.domain.entities import HotelOption, TransportOption
from src.domain import PolicyLimit
from src.domain.enums import CabinClass, TransportMode
from src.storage.memory import (
    DEFAULT_POLICY_LIMIT,
    InMemoryHotelRepository,
    InMemoryTransportRepository,
    StaticPolicyRepository,
)
from src.tools._result import CARD_HOTEL, CARD_POLICY, CARD_TRANSPORT
from src.tools.travel import (
    _coerce_price,
    _limit_text,
    build_travel_tools,
    tools_of_kind,
)

USER = "u-test"


# ---------------------------------------------------------------------------
# 测试替身：会按需失败的仓储
# ---------------------------------------------------------------------------
class BoomTransport:
    """``search`` 永远抛异常的交通仓储。"""

    async def search(self, **_kwargs: Any) -> list[TransportOption]:
        raise ConnectionError("上游超时")


class BoomHotel:
    """``search`` 永远抛异常的酒店仓储。"""

    async def search(self, **_kwargs: Any) -> list[HotelOption]:
        raise ConnectionError("上游超时")


class BoomPolicy:
    """``limit_for`` 永远抛异常的差标仓储。"""

    async def limit_for(self, **_kwargs: Any) -> PolicyLimit:
        raise ConnectionError("上游超时")


class FixedPolicy:
    """``limit_for`` 永远返回同一条差标的差标仓储。

    用于构造内存实现里没有的差标形态（比如「不限价」这种 ``0`` 哨兵值）。
    """

    def __init__(self, limit: PolicyLimit) -> None:
        self._limit = limit

    async def limit_for(self, **_kwargs: Any) -> PolicyLimit:
        """返回构造时给定的差标。"""
        return self._limit


def build(
    *,
    transport: Any = None,
    hotel: Any = None,
    policy: Any = None,
) -> dict[str, Any]:
    """构造工具并返回 ``{名字: 工具}``。

    Args:
        transport (`Any`): 交通仓储；``None`` 用内存实现。
        hotel (`Any`): 酒店仓储；``None`` 用内存实现。
        policy (`Any`): 差标仓储；``None`` 用内存实现。

    Returns:
        `dict[str, Any]`: 工具名到工具的映射。
    """
    return tools_of_kind(
        build_travel_tools(
            transport_repo=transport or InMemoryTransportRepository(),
            hotel_repo=hotel or InMemoryHotelRepository(),
            policy_repo=policy or StaticPolicyRepository(),
            user_id=USER,
        ),
    )


def payload_of(chunk: Any) -> dict[str, Any]:
    """把工具返回的 ``ToolChunk`` 解回 JSON 载荷。

    ⚠️ 刻意从**文本**里解 JSON，而不是去读 ``ToolChunk`` 的什么结构化字段 ——
    生产代码走的就是「把 JSON 塞进一个 TextBlock」这条路（框架会把
    ``dict`` 也归一化成文本，见 ``src/tools/_result.py`` 的说明）。
    测结构化字段等于测一条生产上不存在的路径。
    """
    text = "".join(getattr(b, "text", "") for b in chunk.content)
    return json.loads(text)

def amounts_in(text: str) -> set[float]:
    """把一段话里的**金额**解出来，用于断言「某个数没有出现」。

    ⚠️ 不能直接 ``assert "0 元" not in text`` —— 「2000 元」包含「0 元」，
    「600 元/晚」包含「0 元/晚」。子串匹配会让断言在**正确**的实现上也是红的，
    而一条总是失败的用例会被当成噪声删掉，不是被当成回归信号。
    所以这里把金额解出来再比，只有真正的 0 才会被抓住。

    Args:
        text (`str`): 待解析的文本。

    Returns:
        `set[float]`: 文本里出现的金额集合。
    """
    return {
        float(raw.replace(",", ""))
        for raw in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*元", text)
    }


def call(tool: Any, /, **kwargs: Any) -> Any:
    """调用一个 ``FunctionTool``（同步等待其异步内部函数）。"""
    return asyncio.run(tool(**kwargs))


# ---------------------------------------------------------------------------
# 一、交通查询
# ---------------------------------------------------------------------------
def test_transport_returns_a_card_with_the_expected_shape() -> None:
    """正常查询返回 ``transport_options`` 卡片，字段齐全。"""
    chunk = call(build()["search_transport"], origin="杭州", destination="北京", depart_date="2026-10-08")
    payload = payload_of(chunk)

    assert payload["ok"] is True
    assert payload["card"] == CARD_TRANSPORT
    assert set(payload) >= {"ok", "summary", "card", "items"}

    item = payload["items"][0]
    for field in (
        "option_id", "mode", "mode_display", "carrier", "origin", "destination",
        "depart_at", "arrive_at", "duration_minutes", "price", "cabin",
        "seats_left", "sold_out",
    ):
        assert field in item, f"交通卡片缺字段 {field}"


def test_transport_summary_mentions_the_cheapest_option() -> None:
    """⚠️ 摘要必须说出**最低价**，而不是只说「找到 N 个」。

    ``summary`` 是模型组织答复的依据。只说数量的摘要，模型只能自己编价格 ——
    而机票价格是它编不出来的东西。
    """
    chunk = call(build()["search_transport"], origin="杭州", destination="北京", depart_date="2026-10-08")
    payload = payload_of(chunk)
    cheapest = min(item["price"] for item in payload["items"])

    assert str(cheapest) in payload["summary"] or f"{cheapest:g}" in payload["summary"]


def test_transport_items_are_sorted_by_price() -> None:
    """⚠️ 列表按价格升序。

    排序是工具的责任，不是模型或前端的 —— 两边各排一次，用户就可能看到
    「推荐」和「列表第一项」不是同一个。
    """
    chunk = call(build()["search_transport"], origin="杭州", destination="北京", depart_date="2026-10-08")
    prices = [item["price"] for item in payload_of(chunk)["items"]]

    assert prices == sorted(prices)


@pytest.mark.parametrize(
    "missing_field, expected",
    [("origin", "出发城市"), ("destination", "目的城市"), ("depart_date", "出发日期")],
)
def test_transport_asks_for_missing_parameters(missing_field: str, expected: str) -> None:
    """⚠️ 缺要素时返回「需补充」而**不是**错误。

    用户说「帮我订张票」而没说去哪，这是对话的正常一步，不是系统出错。
    标成 ERROR 会让界面出现红色报错样式，而正确表现是继续追问。
    """
    kwargs = {"origin": "杭州", "destination": "北京", "depart_date": "2026-10-08"}
    kwargs[missing_field] = ""

    payload = payload_of(call(build()["search_transport"], **kwargs))

    assert payload["ok"] is True, "缺要素不是错误"
    assert expected in payload["needs"], f"needs 里应当有 {expected}"
    assert payload["card"] == "", "追问时不该给卡片"


def test_transport_missing_parameters_are_all_reported_at_once() -> None:
    """⚠️ 一次把所有缺的都说出来，而不是挤牙膏。

    只报第一个的话，用户补了出发地又被问目的地、补了目的地又被问日期 ——
    三轮往返才能开始查询，而这三轮本可以合成一轮。
    """
    payload = payload_of(call(build()["search_transport"], origin="", destination="", depart_date=""))

    assert set(payload["needs"]) == {"出发城市", "目的城市", "出发日期"}


def test_empty_transport_result_is_a_conclusion_not_an_error() -> None:
    """⚠️ **空结果不是错误** —— 状态是 SUCCESS，``ok`` 是 true。

    这是本文件最要紧的一条。把空结果标成错误，用户会「对着一个正常结论
    反复重试」，而每次都得到同样的失败提示。
    """
    class EmptyTransport:
        async def search(self, **_kwargs: Any) -> list[TransportOption]:
            return []

    chunk = call(build(transport=EmptyTransport())["search_transport"],
                 origin="杭州", destination="北京", depart_date="2026-10-08")

    assert chunk.state is ToolResultState.SUCCESS
    payload = payload_of(chunk)
    assert payload["ok"] is True
    assert payload["items"] == []
    assert payload["card"] == CARD_TRANSPORT, "空结果仍然要卡片，好让前端显示空状态"


def test_empty_transport_summary_suggests_a_way_out() -> None:
    """⚠️ 空结果的摘要要给出**下一步**（换日期 / 换交通方式）。

    只说「没有找到」等于把问题丢回给用户。这是产品文案，但它是模型组织
    答复时唯一能依据的东西 —— 摘要里没有建议，模型也就给不出建议。
    """
    class EmptyTransport:
        async def search(self, **_kwargs: Any) -> list[TransportOption]:
            return []

    payload = payload_of(call(build(transport=EmptyTransport())["search_transport"],
                              origin="杭州", destination="北京", depart_date="2026-10-08"))

    assert "日期" in payload["summary"] or "换" in payload["summary"]


def test_transport_survives_a_broken_repository() -> None:
    """⚠️ 仓储抛异常时给出**中文**说明，而不是把英文异常交给用户。

    框架会把工具抛出的异常吞成 ``ToolChunk(state=ERROR, text=str(e))``，
    用户看到的是 ``ConnectionError: 上游超时``。捕下来至少能给出中文说明，
    而 ``detail`` 里保留原文，排查线索也不丢。
    """
    chunk = call(build(transport=BoomTransport())["search_transport"],
                 origin="杭州", destination="北京", depart_date="2026-10-08")

    assert chunk.state is ToolResultState.ERROR
    payload = payload_of(chunk)
    assert payload["ok"] is False
    assert "稍后重试" in payload["summary"] or "不可用" in payload["summary"]
    assert "ConnectionError" in payload["detail"], "排查线索不该被丢掉"


@pytest.mark.parametrize("raw", ["FLIGHT", "飞机", "航班", "机票", "flight"])
def test_transport_accepts_flight_synonyms(raw: str) -> None:
    """⚠️ 模型把 ``FLIGHT`` 写成「飞机」是常事，不该为此中断查询。"""
    chunk = call(build()["search_transport"], origin="杭州", destination="北京",
                 depart_date="2026-10-08", mode=raw)
    modes = {item["mode"] for item in payload_of(chunk)["items"]}

    assert modes <= {"FLIGHT"}, f"{raw} 应当只查航班，实际 {modes}"


@pytest.mark.parametrize("raw", ["TRAIN", "火车", "高铁", "动车", "列车"])
def test_transport_accepts_train_synonyms(raw: str) -> None:
    """同上，火车方向的别名。"""
    chunk = call(build()["search_transport"], origin="杭州", destination="北京",
                 depart_date="2026-10-08", mode=raw)
    modes = {item["mode"] for item in payload_of(chunk)["items"]}

    assert modes <= {"TRAIN"}, f"{raw} 应当只查火车，实际 {modes}"


@pytest.mark.parametrize("raw", ["", "ANY", "不限", "都可以", "随便", "飞艇"])
def test_unknown_transport_mode_degrades_to_any(raw: str) -> None:
    """⚠️ 认不出的交通方式退化成 ``ANY``，**不报错**。

    ``ANY`` 的语义正是「不限」，退化成「两种都查」对用户无害；
    而报错会让一次本可完成的查询变成一次失败。
    """
    chunk = call(build()["search_transport"], origin="杭州", destination="北京",
                 depart_date="2026-10-08", mode=raw)
    payload = payload_of(chunk)

    assert payload["ok"] is True
    assert len(payload["items"]) > 0, f"mode={raw!r} 应当退化成「两种都查」"


def test_transport_does_not_query_when_parameters_are_missing() -> None:
    """⚠️ 缺要素时**不查仓储**。

    否则「杭州 → 空字符串」这种查询会打到下游，得到一堆无意义的结果；
    更糟的是它会掩盖「模型没传全参数」这个真正的问题。
    """
    queried: list[str] = []

    class SpyTransport:
        async def search(self, **kwargs: Any) -> list[TransportOption]:
            queried.append("called")
            return []

    call(build(transport=SpyTransport())["search_transport"], origin="杭州", destination="", depart_date="")

    assert not queried, "缺要素时不该查仓储"


# ---------------------------------------------------------------------------
# 二、酒店查询
# ---------------------------------------------------------------------------
def test_hotel_returns_a_card_with_the_expected_shape() -> None:
    """正常查询返回 ``hotel_options`` 卡片。"""
    payload = payload_of(call(build()["search_hotels"], city="北京"))

    assert payload["ok"] is True
    assert payload["card"] == CARD_HOTEL
    item = payload["items"][0]
    for field in ("option_id", "name", "city", "area", "price_per_night", "star", "distance_km"):
        assert field in item, f"酒店卡片缺字段 {field}"


def test_hotel_summary_names_the_cheapest() -> None:
    """摘要说出最低价与酒店名 —— 与交通同理。"""
    payload = payload_of(call(build()["search_hotels"], city="北京"))
    cheapest = min(item["price_per_night"] for item in payload["items"])

    assert f"{cheapest:g}" in payload["summary"]


def test_hotel_asks_for_the_city() -> None:
    """缺城市时追问，不报错。"""
    payload = payload_of(call(build()["search_hotels"], city=""))

    assert payload["ok"] is True
    assert "入住城市" in payload["needs"]


def test_hotel_empty_area_result_names_the_area() -> None:
    """⚠️ 指定商圈却查不到时，**明确说出商圈名**。

    用户听到「没有找到酒店」会以为整座城市都没有；说清是「某某附近没有」
    才能引导他放宽条件。这个差别决定了用户下一步会不会换个商圈再试。
    """
    payload = payload_of(call(build()["search_hotels"], city="北京", area="不存在的商圈"))

    assert payload["ok"] is True
    assert "不存在的商圈" in payload["summary"], f"摘要没点出商圈名：{payload['summary']}"


def test_hotel_survives_a_broken_repository() -> None:
    """仓储异常时给中文说明。"""
    chunk = call(build(hotel=BoomHotel())["search_hotels"], city="北京")

    assert chunk.state is ToolResultState.ERROR
    assert "ConnectionError" in payload_of(chunk)["detail"]


# ---------------------------------------------------------------------------
# 三、差标核对
# ---------------------------------------------------------------------------
def test_policy_approves_an_in_policy_hotel() -> None:
    """合规的酒店给「符合」结论，并**说明依据**。"""
    price = DEFAULT_POLICY_LIMIT.max_hotel_price - 100
    payload = payload_of(call(build()["check_travel_policy"], kind="hotel", price=price))

    assert payload["ok"] is True
    assert payload["card"] == CARD_POLICY
    assert payload["items"][0]["compliant"] is True
    assert "符合" in payload["summary"]


def test_policy_rejects_an_over_limit_hotel() -> None:
    """超标的酒店给「不符合」结论，并给出建议。"""
    price = DEFAULT_POLICY_LIMIT.max_hotel_price + 500
    payload = payload_of(call(build()["check_travel_policy"], kind="hotel", price=price))

    item = payload["items"][0]
    assert item["compliant"] is False
    assert "不符合" in payload["summary"]
    assert item["advice"], "不合规时必须给出改进建议"


def test_policy_checks_both_cabin_and_price_for_flights() -> None:
    """⚠️ 机票**舱位与票价两项都要查**。

    只查一项会漏掉最常见的两种违规：「经济舱但超价」与「价格没超但订了
    商务舱」。两种都真实存在，且各自都只被一项检查覆盖。
    """
    limit = DEFAULT_POLICY_LIMIT

    # 舱位合规、价格超标
    over_price = payload_of(call(build()["check_travel_policy"], kind="flight",
                                 price=limit.max_flight_price + 500,
                                 cabin=limit.max_cabin.value))
    assert over_price["items"][0]["compliant"] is False, "价格超标应当被判不合规"

    # 价格合规、舱位超标
    over_cabin = payload_of(call(build()["check_travel_policy"], kind="flight",
                                 price=limit.max_flight_price - 500, cabin="FIRST"))
    assert over_cabin["items"][0]["compliant"] is False, "舱位超标应当被判不合规"


def test_policy_verdict_carries_the_limit_it_used() -> None:
    """⚠️ 结论里带上**所依据的差标数值**。

    只有这样用户才能判断「是标准太严」还是「我选得太贵」。只给一个
    「不符合」，用户无从改进。
    """
    item = payload_of(call(build()["check_travel_policy"], kind="hotel", price=100))["items"][0]

    assert item["max_hotel_price"] == DEFAULT_POLICY_LIMIT.max_hotel_price
    assert item["max_flight_price"] == DEFAULT_POLICY_LIMIT.max_flight_price
    assert item["max_cabin"] == DEFAULT_POLICY_LIMIT.max_cabin.value
    assert item["policy_note"], "必须给出所依据的差标说明"


def test_policy_rejects_an_unknown_kind_in_chinese() -> None:
    """认不出的核对类型给中文错误，不抛异常。"""
    chunk = call(build()["check_travel_policy"], kind="火车票")

    assert chunk.state is ToolResultState.ERROR
    assert "无法识别" in payload_of(chunk)["summary"]


def test_policy_survives_a_broken_repository() -> None:
    """差标仓储异常时给中文说明。"""
    chunk = call(build(policy=BoomPolicy())["check_travel_policy"], kind="hotel", price=100)

    assert chunk.state is ToolResultState.ERROR
    assert "ConnectionError" in payload_of(chunk)["detail"]


# ---------------------------------------------------------------------------
# 四、权限：只读工具必须走快速通道
# ---------------------------------------------------------------------------
def test_all_travel_tools_are_read_only() -> None:
    """★★ **本文件最要紧的一条**：三个查询工具都必须是 ``is_read_only=True``。

    已核实（``agentscope/tool/_adapters.py:116-135``）：``FunctionTool`` 的 ``permission``
    默认是 ``None``，而 ``None`` 被解释成 ``PermissionDecision(behavior=ASK)``
    —— 于是**每一个**工具调用都会弹一次用户确认。

    对查询类工具那是灾难性的：用户问「查下我的订单」，系统弹窗问
    「是否允许查询订单？」。而 ``is_read_only=True`` 会命中权限引擎的
    只读快速通道（``agentscope/permission/_engine.py:659-692``），在工具自身的权限
    判定**之前**直接放行。

    ⚠️ 这条断言反过来也重要：它让「有人为了让演示更顺畅而把写工具也标成
    只读」这个改动必须在**测试**里显式地改掉这一行，而不是悄悄溜过去。
    """
    for name, tool in build().items():
        assert tool.is_read_only is True, f"{name} 不是只读工具，会让每次查询都弹确认窗"


def test_travel_tools_have_no_name_collisions() -> None:
    """⚠️ 工具名唯一。

    ``Toolkit`` 对重名是**静默覆盖**（只打一条 warn），于是「新加的工具
    悄悄顶掉了旧工具」而没有任何报错 —— 表现是「某个功能突然不工作了」。
    """
    tools = build_travel_tools(
        transport_repo=InMemoryTransportRepository(),
        hotel_repo=InMemoryHotelRepository(),
        policy_repo=StaticPolicyRepository(),
        user_id=USER,
    )
    names = [t.name for t in tools]

    assert len(names) == len(set(names)), f"工具名重复：{names}"


def test_every_tool_has_a_description_for_the_model() -> None:
    """⚠️ 每个工具都要有描述。

    工具描述是模型决定「该不该调这个工具」的**唯一**依据。空描述的工具
    等于不可用 —— 模型只能靠名字猜，而它通常猜错。
    """
    for name, tool in build().items():
        description = getattr(tool, "description", "") or ""
        assert len(description.strip()) > 10, f"{name} 的描述太短或为空"


def test_the_user_id_is_bound_at_construction_not_passed_in() -> None:
    """⚠️ ``user_id`` 由闭包绑定，**不是**工具入参。

    让模型传 ``user_id`` 等于把租户隔离交给模型 —— 它完全可能传别人的。
    这条断言守住「工具的参数表里没有 user 相关的字段」。
    """
    for name, tool in build().items():
        properties = (tool.input_schema or {}).get("properties", {})
        leaked = [p for p in properties if "user" in p.lower() or "tenant" in p.lower()]
        assert not leaked, f"{name} 的参数里出现了用户标识字段 {leaked}"


def test_transport_cabin_is_reported_for_each_option() -> None:
    """每个交通选项都带上舱位 —— 差标核对要用。"""
    items = payload_of(call(build()["search_transport"], origin="杭州",
                            destination="北京", depart_date="2026-10-08"))["items"]

    assert all(item["cabin"] == CabinClass.ECONOMY.value for item in items)


def test_transport_mode_filter_actually_reduces_the_result_set() -> None:
    """⚠️ ``mode`` 过滤器真的起作用（而不是被忽略）。

    「参数收下了但没用」是最难发现的一类缺陷：不报错、不告警，只是结果里
    混着用户明确说不想要的东西。
    """
    both = payload_of(call(build()["search_transport"], origin="杭州",
                           destination="北京", depart_date="2026-10-08"))["items"]
    only_train = payload_of(call(build()["search_transport"], origin="杭州",
                                 destination="北京", depart_date="2026-10-08", mode="TRAIN"))["items"]

    assert len(only_train) < len(both), "按火车过滤之后结果没有变少，说明过滤没生效"
    assert all(item["mode"] == TransportMode.TRAIN.value for item in only_train)


# ---------------------------------------------------------------------------
# 三·补、只查标准（不传价格）
# ---------------------------------------------------------------------------
# 背景（2026-10-03 实测）：用户问「住宿标准是多少」，模型调本工具但拿不出
# 单价，price 取默认 0.0，于是工具返回
#     「酒店 0 元/晚 符合差旅标准。（依据：默认差标：…不超过 600 元…）」
# 上限 600 只在一句括注里，正文却赫然写着「0 元/晚」。实测同一句话连问 4 轮，
# 模型分别答成「住宿上限是 0 元/晚」「工具是以 0 元/晚去校验的」
# 「具体金额工具这次没返回数值」—— 工具自己造了一个不存在的数字，再把模型带偏。
#
# 修法：``price <= 0`` 时改走「查标准」分支，正文给上限、依据与适用范围边界。
def test_policy_lookup_returns_the_limit_when_no_price_is_given() -> None:
    """★ 不传价格 = 查标准：正文必须给出上限本身，且**不出现 0 元**。"""
    payload = payload_of(call(build()["check_travel_policy"], kind="hotel"))

    summary = payload["summary"]
    assert payload["ok"] is True
    assert payload["card"] == CARD_POLICY
    assert f"{DEFAULT_POLICY_LIMIT.max_hotel_price:g}" in summary, (
        f"查标准却没给出上限数值：{summary!r}"
    )
    # ⚠️ 这一条是这次修复的**核心**断言：正文里不能再出现「0 元」这个额度。
    #
    # 不能写成 ``assert "0 元" not in summary`` —— 那会被「2000 元」里的
    # 子串命中，于是用例在**正确**的实现上也是红的。一条总是失败的用例
    # 会被当成噪声删掉，而不是被当成回归信号。
    #
    # 也不能只写成 ``"0 元/晚" not in summary`` —— 「600 元/晚」同样包含它。
    # 所以这里把金额**解出来**再判：只有一个真正的 0 才会被抓住。
    amounts = amounts_in(summary)
    assert amounts, f"正文里一个金额都没有，用例本身失效了：{summary!r}"
    assert 0.0 not in amounts, f"查标准时又冒出了「0 元」：{summary!r}"
    assert "符合" not in summary, (
        f"查标准不该给出「符合/不符合」的结论 —— 那是校验分支的话术：{summary!r}"
    )


def test_policy_lookup_states_the_scope_boundary() -> None:
    """★ 查标准的返回必须**明说**没有按城市/职级分档。

    这是本系统的真实状态（``PolicyLimit`` 只有三个字段，域里没有职级概念）。
    不写出来，模型就会按记忆里的行业惯例补一张分档表 —— 实测出现过
    「一类城市 600、其他 450」这类编造。

    ⚠️ 断言的是**否定句本身**（「没有按城市分档」），不是「城市」这两个字
    出现在正文里。后者太弱：删掉否定句之后，后半句
    「某个城市或某个职级的特别额度」里照样有这两个词，用例会继续绿 ——
    实测就是这么漏掉的（变异测试 GREEN-BAD）。而我要守的恰恰是
    「说清楚了**没有**分档」，不是「提到了城市」。
    """
    summary = payload_of(call(build()["check_travel_policy"], kind="hotel"))["summary"]

    assert "没有按城市分档" in summary, (
        f"没有说清适用范围的边界，模型会自己补一张分档表：{summary!r}"
    )
    assert "没有按职级分档" in summary, f"只说了城市、漏了职级：{summary!r}"


def test_policy_lookup_flight_renders_cabin_in_chinese() -> None:
    """★ 机票查标准时舱位必须是中文。

    ⚠️ 直接写 ``limit.max_cabin.value`` 会得到 ``"ECONOMY"`` —— 模型很可能
    原样复述给用户。用户不认识这个词，而且它暴露了内部枚举。
    """
    summary = payload_of(call(build()["check_travel_policy"], kind="flight"))["summary"]

    assert "ECONOMY" not in summary, f"把内部枚举名说给用户了：{summary!r}"
    assert "经济舱" in summary


@pytest.mark.parametrize("raw_kind", ["hotel", "酒店", "住宿", "flight", "机票", "飞机"])
def test_policy_lookup_accepts_the_same_kind_synonyms(raw_kind: str) -> None:
    """★ 查标准与校验走**同一套** kind 别名，不能只认英文。"""
    payload = payload_of(call(build()["check_travel_policy"], kind=raw_kind))

    assert payload["ok"] is True
    assert "无法识别" not in payload["summary"]


def test_policy_lookup_rejects_an_unknown_kind_too() -> None:
    """★ 查标准分支也要挡住认不出的 kind，且同样是中文错误。

    ⚠️ 这条守的是「两个分支的 kind 判定不会漂移」：查标准的分支是**另写**的
    一段 if/elif，如果只在校验分支里挡未知 kind，那么
    ``check_travel_policy(kind="火车票")`` 会从「中文报错」变成
    「返回一份酒店标准」—— 用户问火车票，系统答酒店。
    """
    chunk = call(build()["check_travel_policy"], kind="火车票")

    assert chunk.state is ToolResultState.ERROR
    assert "无法识别" in payload_of(chunk)["summary"]


def test_policy_lookup_payload_is_marked_as_a_lookup() -> None:
    """★★ 查标准的负载必须带 ``lookup: True``，且**不含** ``compliant``。

    ⚠️ 这是给前端用的判别字段。``policy_verdict`` 卡片承载**两种语义**
    （查标准 / 核对结论），而前端原先只写 ``compliant === true ? 绿 : 红``
    —— 查标准的负载没有 ``compliant``，``undefined`` 被判成 false，
    于是用户问一句最普通的「住宿标准是多少」，卡片会打出一个红色
    「不符合」徽标。实测（2026-10-03）在
    ``PolicyVerdictCard.tsx`` 与预构建包里确认了这条渲染路径。

    ⚠️ 同时断言 ``compliant`` **不在**负载里（而不是给它一个 ``None``）：
    留着它会让「三态」退化成「两态 + 一个永远 falsy 的值」，
    下一个人很容易又写出 ``compliant === true ? 绿 : 红``。
    """
    for kind in ("hotel", "flight"):
        payload = payload_of(call(build()["check_travel_policy"], kind=kind))
        item = payload["items"][0]
        assert item["lookup"] is True, f"{kind} 查标准没被标记：{item}"
        assert "compliant" not in item, (
            f"{kind} 查标准的负载里出现了 compliant —— 卡片会把它当核对结论渲染：{item}"
        )


def test_policy_check_payload_is_marked_as_a_check() -> None:
    """★ 核对负载必须带 ``lookup: False`` 与 ``compliant``。

    ⚠️ 与上一条配对：只钉一边的话，把 ``lookup`` 写成常量 True
    也能让查标准那条绿 —— 而那时**所有**核对结论都会被渲染成中性徽章。
    """
    payload = payload_of(call(build()["check_travel_policy"], kind="hotel", price=300))
    item = payload["items"][0]

    assert item["lookup"] is False, item
    assert item["compliant"] is True, item


def test_policy_flight_cabin_without_price_still_gets_a_verdict() -> None:
    """★★ 只给舱位、不给价格时，必须给出**确定性的舱位结论**，而不是查标准。

    ⚠️ 早先的判据是 ``if price <= 0:`` 就走查标准，于是「我订商务舱，
    符合标准吗」被吞成查标准：系统手里握着确定性规则（``check_cabin``），
    却只把标准念了一遍，把判定推回给模型 —— 与本项目「能用代码判定的
    绝不交给模型」的原则相悖。

    ⚠️ 同时钉住主语里**不出现价格**：写 ``{price:g}`` 会得到
    「机票 0 元（商务舱）」，把「没提供价格」说成「零元机票」。
    """
    payload = payload_of(
        call(build()["check_travel_policy"], kind="flight", cabin="商务舱")
    )
    item = payload["items"][0]
    summary = payload["summary"]

    assert item["lookup"] is False, f"只给舱位被吞成了查标准：{item}"
    assert item["compliant"] is False, f"商务舱超出经济舱差标，应当不合规：{item}"
    assert "不符合" in summary, summary
    assert "商务舱" in summary, f"结论里没说是哪个舱位不合规：{summary!r}"
    assert "经济舱" in summary, f"结论里没给改订建议：{summary!r}"
    assert 0.0 not in amounts_in(summary), f"「0 元」又被生产出来了：{summary!r}"


def test_policy_flight_cabin_without_price_is_compliant_when_within_limit() -> None:
    """★ 对照组：舱位在差标之内时，只给舱位也要给出「符合」。"""
    payload = payload_of(
        call(build()["check_travel_policy"], kind="flight", cabin="经济舱")
    )
    item = payload["items"][0]

    assert item["lookup"] is False, item
    assert item["compliant"] is True, item


@pytest.mark.parametrize("kind", ["flight", "机票", "飞机"])
def test_policy_flight_without_price_or_cabin_is_still_a_lookup(kind: str) -> None:
    """★ 边界：机票既没价格也没舱位时仍走查标准 —— 没有可核对的对象。"""
    payload = payload_of(call(build()["check_travel_policy"], kind=kind))

    assert payload["items"][0]["lookup"] is True, payload["items"][0]
    assert "符合" not in payload["summary"], payload["summary"]


def test_policy_cabin_verdict_never_prints_english_codes() -> None:
    """★ 舱位结论里不能出现英文枚举码。

    ⚠️ ``reasons``/``advice`` 是直接印在卡片与正文上的话，
    ``check_cabin`` 早先写的是 ``.value``，于是用户看到
    「BUSINESS 超出差标允许的最高舱位 ECONOMY」—— 中英夹杂，
    最关键的结论（到底哪个舱位不合规）反而看不懂。
    """
    payload = payload_of(
        call(build()["check_travel_policy"], kind="flight", price=1200, cabin="BUSINESS")
    )
    item = payload["items"][0]
    rendered = " ".join(
        [*item["reasons"], item.get("advice") or "", payload["summary"]]
    )

    assert item["compliant"] is False, "用例前提失效：这单应当不合规"
    leaked = [
        code
        for code in ("PREMIUM_ECONOMY", "ECONOMY", "BUSINESS", "FIRST", "ANY")
        if code in rendered
    ]
    assert not leaked, f"英文舱位码漏给了用户：{rendered!r}"


def test_policy_zero_price_is_treated_as_lookup_not_a_free_hotel() -> None:
    """★ 显式传 0 与不传，行为一致（都是查标准）。

    ⚠️ 这条把「0 是哨兵」这个约定钉死。若将来有人改成 ``price is None`` 判定，
    显式传 0 会掉回校验分支，于是「0 元/晚 符合差旅标准」这句又被生产出来 ——
    而调用方（模型）恰恰经常显式传 0。

    ⚠️ 不能只断言「显式 0 与不传的结果相同」：两者**一起**掉回校验分支时
    它们依然相同，用例照样绿（实测变异测试 GREEN-BAD）。所以这里断言的是
    性质本身 —— 显式传 0 得到的结果必须**是一条标准，不是一句合规判定**。
    """
    explicit = payload_of(call(build()["check_travel_policy"], kind="hotel", price=0))
    implicit = payload_of(call(build()["check_travel_policy"], kind="hotel"))

    summary = explicit["summary"]
    assert explicit["summary"] == implicit["summary"]
    assert "符合" not in summary, f"显式传 0 掉回了校验分支：{summary!r}"
    assert 0.0 not in amounts_in(summary), f"「0 元/晚」又被生产出来了：{summary!r}"


def test_policy_negative_price_is_treated_as_lookup_too() -> None:
    """★ 负价与 0 一样走查标准 —— **不是**「-1 元/晚 符合差旅标准」。

    ⚠️ 这条用例的存在是为了钉死判据里的 ``<=``。2026-10-03 变异测试实测：
    把 ``price <= 0`` 改成 ``price == 0`` 后**整套用例全绿**（存活变异体），
    而真实的工具会返回

        「酒店 -1 元/晚 符合差旅标准。（依据：默认差标：…）」

    —— 一句把荒谬输入说成合规的话，恰好是这次修复要消灭的那类输出。
    负价在任何差标下都不可能是一笔真实消费，它和 0 一样属于
    「调用方拿不出价格」，只能走查标准。
    """
    payload = payload_of(call(build()["check_travel_policy"], kind="hotel", price=-1))

    summary = payload["summary"]
    assert "符合" not in summary, f"负价掉进了校验分支：{summary!r}"
    assert "差标" in summary and "不超过" in summary, (
        f"负价没有拿到标准本身：{summary!r}"
    )
    assert not any(amount < 0 for amount in amounts_in(summary)), (
        f"正文里出现了负的金额：{summary!r}"
    )


@pytest.mark.parametrize(
    "raw_price",
    [
        pytest.param(None, id="null"),
        pytest.param("免费", id="non-numeric-string"),
        pytest.param("", id="empty-string"),
        pytest.param("大概八百", id="chinese-numeral"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="inf"),
        pytest.param(True, id="bool"),
    ],
)
def test_policy_unreadable_price_is_treated_as_lookup(raw_price: Any) -> None:
    """★ 读不出的价格一律当「没给价格」，走查标准 —— 绝不抛异常。

    ⚠️ 这是 2026-10-03 对抗验证提出的缺口：框架**不校验、不转换**工具入参
    （``tool/_adapters.py`` 把 ``**kwargs`` 原样传进来），而模型完全可能用
    JSON 的 ``null`` 表达「这一项留空」。修复前 ``None <= 0`` 直接抛
    ``TypeError``，异常被框架吞成一段**英文**错误文本交给模型 ——
    模型不知道发生了什么，用户什么标准都拿不到。

    ⚠️ ``nan`` / ``inf`` 单独列出来是因为它们**不抛异常**、更隐蔽：
    ``nan <= 0`` 为 False，于是溜进校验分支，产出
    「酒店 nan 元/晚 符合差旅标准」。这条比崩溃更难发现。

    ⚠️ 判据是「有没有给出**标准**」，不是「有没有崩」：只断言不抛异常的
    用例在「返回一句 0 元合规」时照样是绿的。
    """
    payload = payload_of(
        call(build()["check_travel_policy"], kind="hotel", price=raw_price)
    )

    summary = payload["summary"]
    assert payload["ok"] is True, f"读不出的价格把工具打挂了：{payload!r}"
    assert "符合" not in summary, (
        f"读不出的价格（{raw_price!r}）掉进了校验分支：{summary!r}"
    )
    assert "不超过" in summary, f"没拿到标准本身：{summary!r}"


@pytest.mark.parametrize(
    ("raw_price", "expected"),
    [
        pytest.param(800, 800.0, id="int"),
        pytest.param(800.5, 800.5, id="float"),
        pytest.param("800", 800.0, id="digit-string"),
        pytest.param("800 元", 800.0, id="string-with-unit"),
        pytest.param("800元/晚", 800.0, id="string-with-unit-and-slash"),
        pytest.param("1,200", 1200.0, id="thousand-separator"),
        pytest.param(" 800 ", 800.0, id="whitespace"),
    ],
)
def test_coerce_price_accepts_the_shapes_models_actually_send(
    raw_price: Any, expected: float
) -> None:
    """★ 模型真会送来的几种「价格」写法必须读得出来。

    ⚠️ 字符串里**只去掉包装**（千分位、单位、空白），绝不去里面抠数字：
    ``"大概八百"`` 抠不出东西就只能当没给（见上面的参数化用例），
    而 ``"800元/晚"`` 的去掉包装恰好等于 800。判据是「这串是不是**就是**
    一个数」，不是「这串里有没有数」—— 后者会把「13800005000」这种
    手机号也读成金额。
    """
    assert _coerce_price(raw_price) == expected


@pytest.mark.parametrize(
    "raw_price",
    [None, "免费", "", "大概八百", float("nan"), float("inf"), True, [], {}],
)
def test_coerce_price_maps_everything_unreadable_to_zero(raw_price: Any) -> None:
    """★ 读不出的输入统一归到 ``0.0``（= 没给价格），不留第二个分支。

    ⚠️ 归到 0 而不是「抛异常」或「返回 None」：调用方只需要看到
    ``price <= 0`` 这一个判据，多一个哨兵值就多一条没人测的路径。
    """
    assert _coerce_price(raw_price) == 0.0


def test_policy_a_numeric_string_price_still_gets_checked() -> None:
    """★ ``"800"`` 这种字符串价格要**照常核对**，不是被当成「没给」。

    ⚠️ 与上面「读不出就走查标准」是一条分界线：读得出就必须用。
    如果实现图省事把字符串一律归零，用户问「这间 800 的能报吗」会得到
    一份标准而不是结论 —— 工具看着没坏，但它答的是另一个问题。
    """
    payload = payload_of(
        call(build()["check_travel_policy"], kind="hotel", price="800")
    )

    summary = payload["summary"]
    assert "不符合" in summary, f"字符串价格没有被真的核对：{summary!r}"
    assert 800.0 in amounts_in(summary), f"结论里没提到用户给的价格：{summary!r}"


def test_policy_unlimited_limit_says_unlimited_not_zero() -> None:
    """★ 差标为「不限」（0）时说「不限」，不能说「上限 0 元」。

    ⚠️ ``PolicyLimit`` 的字段说明写明 ``0`` 表示**不限**。直接格式化数值
    会把「不限」说成「不超过 0 元」—— 意思**恰好相反**：用户会以为一分钱
    都不能报，然后放弃一次本来完全合规的预订。

    ⚠️ fixture 的 note 里**不能**含「不限」二字 —— 那会让断言被自己的输入
    满足（实测变异测试 GREEN-BAD）：真正的输出是「不超过 0 元/晚」，
    而 `「不限」 in summary` 却被 note 里的字骗过。
    """
    from src.domain import PolicyLimit

    payload = payload_of(
        call(
            build(policy=FixedPolicy(PolicyLimit(max_hotel_price=0.0, note="按实报销。")))[
                "check_travel_policy"
            ],
            kind="hotel",
        )
    )

    summary = payload["summary"]
    assert "不限" in summary, f"「不限」被说成了数值：{summary!r}"
    assert 0.0 not in amounts_in(summary), f"把「不限」渲染成了 0 元：{summary!r}"


def test_cabin_text_covers_every_cabin_class() -> None:
    """★ 舱位中文表必须覆盖**全部** :class:`CabinClass` 成员。

    ⚠️ 漏一个的后果是 ``KeyError`` —— 那会让整个差标工具崩掉。这条断言把
    「加枚举成员时忘了改表」变成一个**测试期**红灯，而不是线上的 500。

    ⚠️ 断言的落点是**枚举自己的** ``display_name``，不是本模块那份
    ``_CABIN_TEXT``：后者 2026-10-03 起是前者的派生视图，拿它当断言对象
    会变成同义反复（表错了它会跟着一起错，用例永远绿）。真正的判据是
    「每个成员都有中文名，且中文名不是英文码本身」—— 这条守着它。
    """
    from src.domain.enums import CabinClass

    missing = [m.value for m in CabinClass if not m.display_name]
    assert not missing, f"这些舱位没有中文名：{sorted(missing)}"

    untranslated = [
        m.value for m in CabinClass if m.display_name == m.value
    ]
    assert not untranslated, (
        f"这些舱位的中文名就是英文枚举码本身（漏登记了）：{sorted(untranslated)}"
    )

    # 工具侧那份派生表要与枚举一致 —— 它是快车道与卡片直接取用的入口。
    from src.tools.travel import _CABIN_TEXT

    assert set(_CABIN_TEXT) == set(CabinClass), (
        f"工具侧舱位表与枚举不一致：{sorted(set(CabinClass) ^ set(_CABIN_TEXT))}"
    )


# ---------------------------------------------------------------------------
# 六·补：金额渲染不得改写成科学计数法（缺陷 P2）
# ---------------------------------------------------------------------------
# ⚠️ 这一组守的是「管道自己不许违反它对模型提出的数字纪律」。
# 工具返回是模型「原样引用」的唯一事实来源；一旦里面出现 ``1.2e+06``，
# 模型要么照抄给用户（用户读不懂），要么按正常写法写（回复守卫的接地闸门
# 又会因为来源侧记的是科学计数法而判它编造）。两处都是把**正确**的行为改坏。
class FixedHotel:
    """``search`` 永远返回同一个高价酒店的酒店仓储。

    内存实现的价格上限约 1060 元，够不到 ``:g`` 开始改写数值的那一档
    （6 位有效数字 / 1e6），所以这里自造数据把边界顶出来。
    """

    def __init__(self, price: float) -> None:
        self._price = price

    async def search(self, **_kwargs: Any) -> list[HotelOption]:
        """返回一个单价为构造值的酒店。"""
        return [
            HotelOption(
                option_id="HT-XX-1",
                name="测试酒店",
                city="北京",
                area="市中心",
                price_per_night=self._price,
                star=5,
                distance_km=1.0,
            )
        ]


def test_limit_text_never_uses_scientific_notation() -> None:
    """★ 差标上限的渲染必须逐字 —— 它是接地闸门比对的基准之一。"""
    text = _limit_text(1200000.0, "元/晚")

    assert text == "不超过 1200000 元/晚", f"差标上限被改写：{text!r}"
    assert "e+" not in text.lower()


def test_a_high_hotel_price_is_reported_verbatim() -> None:
    """★ 工具返回里的价格是模型照抄的来源 —— 不得出现科学计数法。

    ⚠️ 1_200_000 是实测里 ``f"{v:g}"`` 会渲染成 ``1.2e+06`` 的最小现场
    （六位有效数字之外，且恰好能被科学计数法整除）。
    """
    payload = payload_of(
        call(build(hotel=FixedHotel(1_200_000.0))["search_hotels"], city="北京")
    )

    summary = payload["summary"]
    assert "1200000 元/晚" in summary, f"价格被改写成了非十进制写法：{summary!r}"
    assert amounts_in(summary) == {1_200_000.0}
