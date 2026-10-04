# -*- coding: utf-8 -*-
"""结构化画像（``src/memory/profile.py``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    画像这一半的失败有个共同特征：**它们都不会报错**。

      · 舱位写成了 ``"business"``（小写）—— 存进去了，检索也对，
        只有真正下单时才发现航司系统不认；
      · 偏好航司列表被 ``set`` 去重 —— 结果还是那三个航司，
        只是「首选国航」这件事没了；
      · ``merge`` 把没传的字段一并清空 —— 用户的座位偏好
        在更新成本中心的时候静默消失。

    三条都是「看起来一切正常」的错。所以下面的用例断言的都不是
    「有没有抛异常」，而是**值的形状与顺序**。
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from src.memory.profile import (
    InMemoryProfileRepository,
    ProfilePatch,
    ProfileRepository,
    ProfileValidationError,
    TravelerProfile,
)


# ==============================================================================
# 一、模型：校验与规范化
# ==============================================================================
def test_an_empty_profile_is_recognisable() -> None:
    """除 ``user_id`` 外全默认值时 ``is_empty()`` 为真。

    ⚠️ 这条支撑着「别再渲染一段空话」：空画像渲染出来的是
    「用户画像：无」，既占 token 又让模型以为「这个用户特意说过
    自己没有任何偏好」。
    """
    assert TravelerProfile(user_id="u1").is_empty() is True
    assert TravelerProfile(user_id="u1", seat_preference="靠窗").is_empty() is False


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("business", "BUSINESS"),
        ("  ECONOMY  ", "ECONOMY"),
        ("First", "FIRST"),
    ],
)
def test_cabin_input_is_normalised(raw: str, expected: str) -> None:
    """舱位大小写与空白都要被规范化。

    ⚠️ 不做这件事的后果不是「显示难看」：``"business"`` 与
    ``"BUSINESS"`` 会被下游当成两个不同的值，于是「按舱位查差标」
    的匹配失败，而差标查不到时的兜底通常是「按最低标准」——
    用户莫名其妙被降舱，且没有任何报错。
    """
    assert TravelerProfile(user_id="u1", preferred_cabin=raw).preferred_cabin == expected


def test_an_unknown_cabin_is_rejected() -> None:
    """非法舱位必须**当场**被拒。

    ⚠️ 拒在建画像这一刻，而不是等到下单。前者是一个可以立刻修的
    输入错误；后者是一次已经花掉的钱。
    """
    with pytest.raises(ValidationError):
        TravelerProfile(user_id="u1", preferred_cabin="超级经济舱")


def test_the_airline_priority_order_survives() -> None:
    """偏好航司是**有序**的，且顺序必须原样保留。

    ⚠️ 这是本文件最要紧的一条。``preferred_airlines`` 的语义是
    「按优先级排序」，一个 ``sorted(set(...))`` 的实现会把
    ``["国航", "南航"]`` 变成 ``["南航", "国航"]`` ——
    列表里还是那两个航司，但用户最在意的那一点（先看国航）
    没了，而且没有任何地方会报错。
    """
    profile = TravelerProfile(
        user_id="u1",
        preferred_airlines=["国航", "南航", "国航", "  东航  ", ""],
    )

    assert profile.preferred_airlines == ["国航", "南航", "东航"], (
        "去重必须**保序**（保留第一次出现的位置），且要丢弃空项与空白"
    )


@pytest.mark.parametrize(
    "field",
    ["preferred_airlines", "preferred_hotel_brands", "dietary_needs"],
)
def test_every_list_field_is_normalised(field: str) -> None:
    """三个列表字段走的是**同一个** validator，逐项验证它真的挂上了。

    ⚠️ 参数化而不是只测一个：``@field_validator`` 的字段名列表
    漏写一个不会有任何报错，症状是「这个字段的空白没被去掉」——
    而它在渲染进 Prompt 时会变成一行带引号的怪东西。
    """
    profile = TravelerProfile(user_id="u1", **{field: [" a ", "a", ""]})

    assert getattr(profile, field) == ["a"]


def test_the_profile_round_trips_through_json() -> None:
    """画像必须能 JSON 往返。

    ⚠️ 这不是形式主义：它要存进 Postgres 的 JSONB 列、要经
    ``model_dump(mode="json")`` 落库再 ``model_validate`` 读回。
    任何一个自定义类型（比如一个非 ``str`` 的枚举）都会在这条路上
    断掉，而断点离定义处很远。
    """
    original = TravelerProfile(
        user_id="u1",
        preferred_cabin="BUSINESS",
        preferred_airlines=["国航"],
        frequent_flyer_numbers={"CZ": "1234567890"},
    )

    restored = TravelerProfile.model_validate(original.model_dump(mode="json"))

    assert restored == original


# ==============================================================================
# 二、Patch：部分更新的语义
# ==============================================================================
def test_a_patch_only_touches_the_fields_it_names() -> None:
    """★★★ 没传的字段**不许**被改。

    ⚠️ 这是本文件里最容易出错、后果最隐蔽的一条。把「没传」
    与「传了 None」混为一谈的实现在这里会把 ``seat_preference``
    清成 None —— 症状是「用户更新成本中心时，座位偏好没了」，
    而两次操作看起来毫不相干，没人会往这个方向查。
    """
    profile = TravelerProfile(
        user_id="u1",
        seat_preference="靠窗",
        preferred_airlines=["国航"],
        dietary_needs=["素食"],
    )

    merged = ProfilePatch(cost_center="CC-001").apply(profile)

    assert merged.cost_center == "CC-001"
    assert merged.seat_preference == "靠窗", "没传的字段被清掉了！"
    assert merged.preferred_airlines == ["国航"]
    assert merged.dietary_needs == ["素食"]


def test_a_patch_does_not_mutate_the_original() -> None:
    """``apply`` 必须返回**新**画像，不改入参。

    ⚠️ 就地改的后果是「调用方手上那份画像在你不知道的时候变了」——
    而画像会被并发读取，于是一个请求的更新会悄悄改变另一个请求
    看到的内容（只在那一个请求里，无法复现）。
    """
    original = TravelerProfile(user_id="u1", seat_preference="靠窗")

    ProfilePatch(seat_preference="过道").apply(original)

    assert original.seat_preference == "靠窗"


def test_a_patch_merges_frequent_flyer_numbers() -> None:
    """常旅客号是**合并**，其余列表是**替换**。

    ⚠️ 两种语义刻意不同，且都有理由：

      · 常旅客号 —— 用户说「我的南航卡号是 X」时**不该**把国航的删掉，
        它们是完全独立的两个事实；
      · 偏好航司列表 —— 「我现在偏好国航」是一个**完整**的表达，
        往里追加会让列表无限增长，而旧偏好永远清不掉。
    """
    profile = TravelerProfile(
        user_id="u1",
        frequent_flyer_numbers={"CZ": "111"},
        preferred_airlines=["国航"],
    )

    merged = ProfilePatch(
        frequent_flyer_numbers={"MU": "222"},
        preferred_airlines=["南航"],
    ).apply(profile)

    assert merged.frequent_flyer_numbers == {"CZ": "111", "MU": "222"}
    assert merged.preferred_airlines == ["南航"], "列表字段应当是**替换**而非追加"


def test_a_patch_still_validates_its_values() -> None:
    """patch 不能成为绕过校验的旁路。

    ⚠️ 若 ``apply`` 直接改属性而不走 ``model_validate``，那么
    「接口层校验、patch 层不校验」就成了一个真实存在的漏洞 ——
    而唯一的入口恰好是 patch。

    ⚠️ 断言的是 :class:`ProfileValidationError` 而不是 pydantic 的
    ``ValidationError``：``apply`` 会把后者**翻译**成前者，因为同一种
    异常在两条语义相反的路上都会出现（这里=调用方的值不合法⇒400；
    仓储 ``get`` 里=**库里的记录**校验不过⇒数据损坏⇒503）。
    测试跟着改，是在钉住「这条路上的分类是我们做的」。
    """
    with pytest.raises(ProfileValidationError):
        ProfilePatch(preferred_cabin="超级经济舱").apply(
            TravelerProfile(user_id="u1"),
        )


# ==============================================================================
# 三、仓储：内存实现
# ==============================================================================
def test_the_in_memory_repository_satisfies_the_protocol(settings: object) -> None:
    """内存实现满足 ``ProfileRepository`` 协议。

    ⚠️ 用 ``isinstance`` 而不是「看着像」：``runtime_checkable``
    的协议检查只比对方法名，因此它能在**装配期**就挡住
    「少实现了一个方法」这种错 —— 而那种错本来要等到生产实现被换进来
    才暴露。
    """
    assert isinstance(InMemoryProfileRepository(), ProfileRepository)


def test_get_returns_none_for_an_unknown_user() -> None:
    """没记录时返回 None，而不是一份空画像。

    ⚠️ 两者语义不同：「None」= 从没见过这个人；「空画像」=
    见过，但他什么都没说过。调用方对前者的处理是「走通用兜底」，
    对后者是「不注入画像段落」—— 混为一谈会让新用户的第一次对话
    拿到一段「已知偏好：无」的噪声。
    """
    repo = InMemoryProfileRepository()

    assert asyncio.run(repo.get("nobody")) is None


def test_merge_creates_a_profile_for_a_brand_new_user() -> None:
    """画像不存在时 ``merge`` 凭空建一份，不报错。

    ⚠️ 用户第一次说「我靠窗」时**不该**收到「请先创建画像」——
    那是把内部数据模型的约束泄露给了用户。
    """
    repo = InMemoryProfileRepository()

    merged = asyncio.run(repo.merge("u-new", ProfilePatch(seat_preference="靠窗")))

    assert merged.seat_preference == "靠窗"
    assert merged.user_id == "u-new"


def test_the_repository_hands_out_copies() -> None:
    """仓储交出去的画像必须与内部状态**不相干**。

    ⚠️ 直接返回内部对象的话，调用方一次属性赋值就改掉了仓储里的
    状态 —— 而那条写入没有经过 ``merge``，于是所有并发保证失效，
    且没有任何日志、没有任何事务。
    """
    repo = InMemoryProfileRepository()
    asyncio.run(repo.merge("u1", ProfilePatch(seat_preference="靠窗")))

    handed_out = asyncio.run(repo.get("u1"))
    assert handed_out is not None
    handed_out.seat_preference = "过道"

    again = asyncio.run(repo.get("u1"))
    assert again is not None
    assert again.seat_preference == "靠窗", "调用方改到了仓储的内部状态！"


def test_upsert_replaces_wholesale() -> None:
    """``upsert`` 是**整体覆盖**，与 ``merge`` 的语义相反。

    ⚠️ 它们在同一个协议里并存是有意的：``merge`` 服务于「用户改了
    一个偏好」，``upsert`` 服务于「管理后台导入了一份完整画像」。
    后者若被实现成合并，导入操作就没法删除任何字段。
    """
    repo = InMemoryProfileRepository()
    asyncio.run(repo.merge("u1", ProfilePatch(seat_preference="靠窗", cost_center="CC")))

    asyncio.run(repo.upsert(TravelerProfile(user_id="u1", cost_center="CC-2")))

    after = asyncio.run(repo.get("u1"))
    assert after is not None
    assert after.cost_center == "CC-2"
    assert after.seat_preference is None, "upsert 应当是整体覆盖"


def test_two_users_never_see_each_other() -> None:
    """不同 ``user_id`` 的画像完全隔离。"""
    repo = InMemoryProfileRepository()
    asyncio.run(repo.merge("u1", ProfilePatch(seat_preference="靠窗")))
    asyncio.run(repo.merge("u2", ProfilePatch(seat_preference="过道")))

    first = asyncio.run(repo.get("u1"))
    second = asyncio.run(repo.get("u2"))

    assert first is not None and first.seat_preference == "靠窗"
    assert second is not None and second.seat_preference == "过道"
