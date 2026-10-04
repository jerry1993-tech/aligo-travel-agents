# -*- coding: utf-8 -*-
"""``src/domain/rules.py`` 的**业务规则引擎**测试。

这组用例的性质与 ``tests/test_domain_schemas.py`` 不同：那边测的是「数据装得对
不对」，这边测的是「业务判定对不对」。规则引擎的全部价值在于**确定性** ——
同样的输入永远得到同样的输出，因此可以像数学命题一样被穷举验证。

用例分四组：

1. **订单状态机** —— 合法迁移、幂等自迁移、终端态拒绝。
2. **一致性不变式** —— 「:attr:`OrderStatus.is_terminal`」与「迁移表里没有出边」
   这两份事实必须永远相等。它们分别在两个文件里，靠人工同步迟早出错。
3. **差标校验** —— 舱位、房价、票价三类上限，以及「0 表示不限」这个约定。
4. **规则的纯函数性质** —— 不读配置、不调模型、结果不可变。
"""

from __future__ import annotations

import pytest

from src.domain import (
    CabinClass,
    OrderStatus,
    PolicyLimit,
    TransportMode,
    check_cabin,
    check_flight_price,
    check_hotel_price,
    check_transition,
    check_transport_mode,
)
from src.domain.rules import _ALLOWED_TRANSITIONS, _CABIN_RANK


# ==============================================================================
# 一、订单状态机
# ==============================================================================
#: 应当被允许的迁移（主流程 + 各环节取消）。
_ALLOWED_CASES: list[tuple[OrderStatus, OrderStatus]] = [
    (OrderStatus.DRAFT, OrderStatus.PENDING_APPROVAL),
    (OrderStatus.DRAFT, OrderStatus.CANCELLED),
    (OrderStatus.PENDING_APPROVAL, OrderStatus.APPROVED),
    (OrderStatus.PENDING_APPROVAL, OrderStatus.REJECTED),
    (OrderStatus.APPROVED, OrderStatus.PAID),
    (OrderStatus.PAID, OrderStatus.COMPLETED),
    (OrderStatus.PAID, OrderStatus.CANCELLED),  # 退票
]


@pytest.mark.parametrize(("current", "target"), _ALLOWED_CASES)
def test_main_flow_transitions_are_allowed(
    current: OrderStatus,
    target: OrderStatus,
) -> None:
    """主流程与各环节的取消都放行。

    ⚠️ 逐个列出而不是遍历迁移表来断言 —— 用表验表是**同义反复**：
    迁移表写错时，遍历它的用例会跟着一起错，全绿。这里的常量列表是**独立
    于实现的第二份事实**，两份对不上才会红。
    """
    assert check_transition(current, target).allowed is True


def test_self_transition_is_allowed_for_every_status() -> None:
    """★ **任何**状态迁移到自己都放行（幂等），包括终端态。

    ⚠️ 这条挡的是分布式系统里最常见的一类伪故障：网络重试、用户连点两次
    「提交」，都会产生一次「已处于目标态」的请求。若拒绝它，一次**本该成功**
    的操作会因为重试而报错，用户看到的是「点了两次就失败」—— 而这类问题
    在单机手工测试时几乎不会出现。
    """
    for status in OrderStatus:
        result = check_transition(status, status)
        assert result.allowed is True, f"{status.value} → 自身 被拒绝了，重试会误报失败"


#: 应当被拒绝的迁移：回退、跨级、从终端态出发。
_REJECTED_CASES: list[tuple[OrderStatus, OrderStatus]] = [
    # 跨级：不能跳过审批直达支付。
    (OrderStatus.DRAFT, OrderStatus.PAID),
    (OrderStatus.DRAFT, OrderStatus.APPROVED),
    # 回退：审批通过后不能退回待审批。
    (OrderStatus.APPROVED, OrderStatus.PENDING_APPROVAL),
    (OrderStatus.PAID, OrderStatus.APPROVED),
    # 从终端态出发。
    (OrderStatus.REJECTED, OrderStatus.PENDING_APPROVAL),
    (OrderStatus.CANCELLED, OrderStatus.PAID),
    (OrderStatus.COMPLETED, OrderStatus.CANCELLED),
]


@pytest.mark.parametrize(("current", "target"), _REJECTED_CASES)
def test_illegal_transitions_are_rejected(current: OrderStatus, target: OrderStatus) -> None:
    """跨级、回退、从终端态出发的迁移一律拒绝。"""
    assert check_transition(current, target).allowed is False


def test_rejection_always_carries_a_reason() -> None:
    """★ 每一次拒绝都必须带上**可读的原因**。

    ⚠️ 原因文案会被直接展示给用户（「订单已是『已完成』终态，不能再变更为
    『已取消』」）。若允许返回空原因，调用方就只能兜一句「操作失败」——
    用户拿到这四个字完全无从下手，而这正是 :class:`TransitionCheck`
    用结果对象而不是 bool 的全部理由。
    """
    result = check_transition(OrderStatus.COMPLETED, OrderStatus.CANCELLED)

    assert result.allowed is False
    assert result.reason.strip(), "拒绝时必须给用户一句能看懂的原因"


def test_terminal_rejection_is_distinguishable_from_illegal_transition() -> None:
    """★ 「终端态」与「非法跳转」的拒绝原因**文案不同**。

    ⚠️ 两者对用户的含义完全不同：前者是「这条订单已经结束，请新建」，
    后者是「你这个顺序不对，下一步应该是 XX」。合并成同一句话会丢掉
    「有没有别的路可走」这一信息 —— 而这恰恰是用户下一步动作的依据。
    """
    terminal = check_transition(OrderStatus.CANCELLED, OrderStatus.PAID)
    illegal = check_transition(OrderStatus.DRAFT, OrderStatus.PAID)

    assert terminal.reason != illegal.reason
    assert "终态" in terminal.reason
    # 非法跳转要报出「当前允许的下一步」，用户才知道怎么继续。
    assert OrderStatus.PENDING_APPROVAL.value in illegal.reason


# ==============================================================================
# 二、一致性不变式（两份事实必须相等）
# ==============================================================================
def test_transition_table_covers_every_status() -> None:
    """★ 迁移表必须为**每一个** :class:`OrderStatus` 成员登记一行。

    ⚠️ 漏登记的后果不是报错，而是**静默放宽**：``check_transition`` 取不到
    条目时用 ``frozenset()`` 兜底，于是那个状态的所有迁移都被拒绝 ——
    或者更糟，如果实现改成了「取不到即放行」，就变成全部允许。本项目的
    实现选了「拒绝」（快速失败），但无论如何这都该在测试里被发现，
    而不是等线上用户撞上。
    """
    missing = set(OrderStatus) - set(_ALLOWED_TRANSITIONS)

    assert not missing, f"迁移表缺少这些状态的登记：{[s.value for s in missing]}"


def test_terminal_flag_matches_the_transition_table() -> None:
    """★★ :attr:`OrderStatus.is_terminal` 与「迁移表里没有出边」必须**等价**。

    ⚠️ 这是本文件最重要的一条：这两份事实存在**两个不同的文件**里 ——
    终端态集合在 ``enums.py``，出边在 ``rules.py``。它们描述同一件事，
    却没有任何机制强制同步。一旦不一致，症状是「系统说订单已完成，
    却又允许它被取消」这种自相矛盾的行为，而两处代码单看都「对」。

    ⚠️ 两个方向都要验：只验一个方向会漏掉「表里有出边但标记为终端」
    （该状态永远走不出去）或「标记非终端但表里没出边」（用户被告知可以
    继续，实际每次都被拒）中的一种。
    """
    for status in OrderStatus:
        has_outgoing = bool(_ALLOWED_TRANSITIONS.get(status))
        assert status.is_terminal is (not has_outgoing), (
            f"{status.value} 的两份事实不一致："
            f"is_terminal={status.is_terminal}，"
            f"迁移表出边数={len(_ALLOWED_TRANSITIONS.get(status, frozenset()))}"
        )


def test_every_status_is_reachable_from_draft() -> None:
    """★ 从 :attr:`OrderStatus.DRAFT` 出发必须能到达**每一个**状态。

    ⚠️ 这条防的是「孤儿状态」：新增一个状态却忘了给它加入边，于是它永远
    不可能出现在任何订单上 —— 代码里为它写的分支全是死代码，而测试若只
    覆盖「已列出的迁移」，这个状态就是用例盲区。
    """
    reachable = {OrderStatus.DRAFT}
    frontier = [OrderStatus.DRAFT]
    while frontier:
        for nxt in _ALLOWED_TRANSITIONS.get(frontier.pop(), frozenset()):
            if nxt not in reachable:
                reachable.add(nxt)
                frontier.append(nxt)

    unreachable = set(OrderStatus) - reachable
    assert not unreachable, (
        f"这些状态从 DRAFT 出发无法到达，说明迁移表缺边："
        f"{[s.value for s in unreachable]}"
    )


# ==============================================================================
# 三、差标校验
# ==============================================================================
def test_cabin_within_limit_is_compliant() -> None:
    """同级或降级都符合差标。"""
    limit = PolicyLimit(max_cabin=CabinClass.BUSINESS)

    assert check_cabin(CabinClass.ECONOMY, limit).compliant is True
    assert check_cabin(CabinClass.BUSINESS, limit).compliant is True


def test_cabin_above_limit_is_rejected_with_advice() -> None:
    """超出差标要拒绝，并给出**降级建议**。

    ⚠️ 只报「不符合」而不说「改成什么才符合」，用户就得自己反推差标 ——
    而这正是他来问系统的原因。

    ⚠️ 断言的是**中文展示名**，不是 ``.value``。``reasons``/``advice``
    是直接印在卡片与正文上的话，早先它们写着
    「FIRST 超出差标允许的最高舱位 ECONOMY」—— 中英夹杂，用户看不懂
    哪条结论落在自己身上。这条用例钉住的是「这两句里不出现英文枚举码」。
    """
    limit = PolicyLimit(max_cabin=CabinClass.ECONOMY)

    verdict = check_cabin(CabinClass.FIRST, limit)

    assert verdict.compliant is False
    assert verdict.reasons
    assert CabinClass.ECONOMY.display_name in verdict.advice

    rendered = " ".join([*verdict.reasons, verdict.advice])
    leaked = [
        cabin.value
        for cabin in (CabinClass.FIRST, CabinClass.ECONOMY)
        if cabin.value in rendered
    ]
    assert not leaked, f"英文舱位码漏进了给用户看的文案：{rendered!r}"


def test_unspecified_cabin_is_always_compliant() -> None:
    """★ :attr:`CabinClass.ANY` 视为符合，**永远不报超差标**。

    ⚠️ ``ANY`` 的含义是「用户还没选」。若这时候报「超出差标」，系统就是在
    对一个尚未做选择的人报错 —— 用户会以为自己的行程有问题，而真正该发生
    的是系统先推荐一个合规的舱位。舱位判定要等到选定具体航班时才有意义。
    """
    strict = PolicyLimit(max_cabin=CabinClass.ECONOMY)

    assert check_cabin(CabinClass.ANY, strict).compliant is True


def test_cabin_rank_is_independent_of_enum_declaration_order() -> None:
    """舱位等级序是**业务事实**，与枚举的声明顺序无关。

    ⚠️ 若实现改成「用 ``list(CabinClass).index(...)`` 比大小」，那么有人
    为了可读性把 ``ECONOMY`` 挪到 ``FIRST`` 后面，差标判定就会**静默反转**
    —— 经济舱被判为超出公务舱差标。这类改动看起来纯属排版，审查时不会
    有人留意。这条用例把两者钉死为互不相关。
    """
    assert _CABIN_RANK[CabinClass.ECONOMY] < _CABIN_RANK[CabinClass.PREMIUM_ECONOMY]
    assert _CABIN_RANK[CabinClass.PREMIUM_ECONOMY] < _CABIN_RANK[CabinClass.BUSINESS]
    assert _CABIN_RANK[CabinClass.BUSINESS] < _CABIN_RANK[CabinClass.FIRST]
    # ANY 不在序里 —— 它由 check_cabin 单独短路，不该有排名。
    assert CabinClass.ANY not in _CABIN_RANK


def test_zero_limit_means_unlimited_not_zero() -> None:
    """★★ ``0`` 表示**不限**，不是「上限为零元」。

    ⚠️ 这是本组里最容易写反、后果最严重的一处。反了之后，一条「该城市
    不设上限」的差标会让用户订不了**任何**酒店，而错误信息会是
    「房价 800 元/晚 超出差标 0 元/晚」—— 看起来像数据录入错误，
    排查的人会去查数据库，而问题其实在比较符号上。

    ⚠️ 票价与房价走的是两套独立实现，必须**分别**验证。只测其中一个的话，
    另一个复制粘贴时写反不会被发现。
    """
    unlimited = PolicyLimit(max_hotel_price=0.0, max_flight_price=0.0)

    assert check_hotel_price(9999.0, unlimited).compliant is True
    assert check_flight_price(9999.0, unlimited).compliant is True


def test_price_boundary_is_inclusive() -> None:
    """上限**含等于**（``<=`` 而非 ``<``）。

    ⚠️ 差标写「800 元」时，800 元的酒店显然应当可以订。用 ``<`` 会让
    恰好卡在标准线上的选项被拒 —— 而这恰恰是员工最容易选中的那一档
    （大家都会贴着标准订）。这类差一错误的投诉率远高于它的技术含量。
    """
    limit = PolicyLimit(max_hotel_price=800.0, max_flight_price=1500.0)

    assert check_hotel_price(800.0, limit).compliant is True
    assert check_flight_price(1500.0, limit).compliant is True
    assert check_hotel_price(800.01, limit).compliant is False
    assert check_flight_price(1500.01, limit).compliant is False


def test_over_budget_message_quantifies_the_excess() -> None:
    """超标时要**报出差额**，而不是只说「超了」。

    ⚠️ 差额是用户决定「换个酒店」还是「走特批」的关键信息。只说「超出
    差标」，用户还得自己算一遍。
    """
    verdict = check_hotel_price(1000.0, PolicyLimit(max_hotel_price=800.0))

    assert verdict.compliant is False
    assert "200" in verdict.reasons[0], "应报出超出金额（1000 - 800 = 200）"


# ------------------------------------------------------------------------------
# 交通方式：建议性规则
# ------------------------------------------------------------------------------
def test_short_haul_flight_gets_advice_but_is_not_blocked() -> None:
    """★ 短途选飞机只给**建议**，不拒绝。

    ⚠️ 把建议做成拒绝是过度管控：员工有正当理由坐飞机（要当天往返、
    高铁无合适车次、携带设备）。系统该做的是提示并留痕交审批判断，
    而不是堵死。因此 ``compliant`` 必须恒为 True —— 若哪天有人「顺手」
    把它改成 False，这条用例会红。
    """
    verdict = check_transport_mode(TransportMode.FLIGHT, distance_km=300.0)

    assert verdict.compliant is True, "建议性规则不该拒绝用户"
    assert verdict.reasons == [], "建议性规则不该往 reasons 里写拒绝理由"
    assert "高铁" in verdict.advice


def test_long_haul_flight_gets_no_advice() -> None:
    """长途选飞机是合理的，不该有多余提示。"""
    verdict = check_transport_mode(TransportMode.FLIGHT, distance_km=2000.0)

    assert verdict.compliant is True
    assert verdict.advice == ""


def test_train_never_gets_the_switch_advice() -> None:
    """本来就要坐高铁的人，不该收到「建议改坐高铁」。"""
    verdict = check_transport_mode(TransportMode.TRAIN, distance_km=300.0)

    assert verdict.advice == ""


def test_train_preferred_threshold_is_configurable() -> None:
    """阈值由调用方传入 —— 规则引擎自己不读配置。

    规则是「里程**不超过**阈值就建议改高铁」，所以固定的 900 公里在两个
    阈值下结论相反：阈值 1500 时建议改高铁，阈值 500 时不建议。

    ⚠️ 这条守的是模块文档里那句「本模块必须是纯函数」。若有人为了图方便
    在这里 ``import`` 配置对象，用例会红；而那个改动会同时毁掉
    「可以脱离框架被验证」这条性质。

    ⚠️ 判据方向容易写反（写这条用例时第一版就写反了）：**小阈值 = 严格**，
    严格意味着对 900 公里**不再**建议高铁，而不是「更强烈地建议」。
    """
    strict = check_transport_mode(
        TransportMode.FLIGHT,
        distance_km=900.0,
        train_preferred_km=500.0,
    )
    loose = check_transport_mode(
        TransportMode.FLIGHT,
        distance_km=900.0,
        train_preferred_km=1500.0,
    )

    assert strict.advice == "", "900 公里已超出 500 公里的建议区间，不该提示改高铁"
    assert "高铁" in loose.advice, "900 公里在 1500 公里以内，应当建议改高铁"


# ==============================================================================
# 四、纯函数性质
# ==============================================================================
def test_verdicts_are_immutable() -> None:
    """判定结果**不可变**。

    ⚠️ 调用方拿到的结论会被层层上抛给用户与日志。若它是可变的，某一层
    「顺手修一下 reasons」就会让日志与用户看到的对不上 —— 而排查时
    日志是唯一可信的东西。冻住它，这类改动在写入时就直接报错。
    """
    verdict = check_hotel_price(1000.0, PolicyLimit(max_hotel_price=800.0))

    with pytest.raises(Exception):
        verdict.compliant = True  # type: ignore[misc]


def test_rules_are_deterministic() -> None:
    """同样的输入永远得到同样的输出（含列表内容，不只是结论）。

    这条看着像废话，但它是「规则引擎」与「让模型拿捏」的分界线：
    模型每次可能给出不同答案，规则引擎不允许。
    """
    limit = PolicyLimit(max_cabin=CabinClass.ECONOMY, max_hotel_price=500.0)

    first = check_hotel_price(900.0, limit)
    second = check_hotel_price(900.0, limit)

    assert first == second
    assert check_cabin(CabinClass.FIRST, limit) == check_cabin(CabinClass.FIRST, limit)
