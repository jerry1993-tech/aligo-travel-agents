# -*- coding: utf-8 -*-
"""差旅**业务规则引擎**：订单状态迁移、差标校验、审批判定。

文件职责：
    把「业务上允许/不允许什么」写成**确定性代码**。这是本项目对博客第
    361-373 行那条经验最直接的落地 —— 原文把「把业务流程与规则全部嵌入
    Prompt」列为准确率停在 50% 的根因，转而采用「工程确定性与 AI 灵活性
    之间的平衡」：**能用代码判定的，绝不交给模型**。

上下游依赖：
    - 上游：:mod:`src.domain.enums`、:mod:`src.domain.schemas`。
    - 下游：``src/tools/``（下单/取消工具调用前先校验）、
      ``src/storage/``（落库前校验状态迁移）、P4 的 RAG 子智能体
      （差标问答的答案要与这里的判定一致，否则「模型说能报、系统不让报」）。

═══ 为什么规则要「先算后说」而不是「让模型自己拿捏」 ═══

一条具体对比：用户问「我订 1200 元的酒店能报吗」。若把差标写在 Prompt 里让
模型推理，模型会给出**看起来合理**的答案，而且它每次可能不一样；一旦它答
「可以」而系统在下单时拒绝，用户看到的是自相矛盾的产品。正确的分工是：
规则引擎给出**唯一的判定**，模型只负责把判定**组织成人话**。

⚠️ 本模块**必须是纯函数**：同样的输入永远得到同样的输出，不读配置、
   不查库、不调模型。这条约束让它可以被普通单测穷举覆盖 —— 而这正是
   「确定性」三个字的价值所在。需要配置（如企业差标上限）时由**调用方**
   通过参数传入。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain.enums import CabinClass, OrderStatus, TransportMode

# ==============================================================================
# 一、订单状态机
# ==============================================================================
#: 合法的状态迁移表：``当前状态 -> 允许迁往的状态集合``。
#:
#: ⚠️ 用**显式的全量字典**而不是「一串 if」或「默认允许」：
#: 后者的问题不是写起来麻烦，而是**新增状态时会静默放开**——加一个
#: ``REFUNDING`` 之后，所有未列出的迁移对它都是放行的。全量字典的相反
#: 性质正是我们要的：新增状态若忘了登记，**任何**迁移都会被拒（快速失败、
#: 立刻被测试发现），而不是被悄悄放行。
#:
#: 终端态在表中**没有出边**（空 frozenset），与
#: :attr:`src.domain.enums.OrderStatus.is_terminal` 的定义互为表里。
#: 两者的一致性由 ``tests/test_domain_rules.py`` 的遍历用例守护。
_ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.DRAFT: frozenset(
        {
            OrderStatus.PENDING_APPROVAL,
            OrderStatus.CANCELLED,
        },
    ),
    OrderStatus.PENDING_APPROVAL: frozenset(
        {
            OrderStatus.APPROVED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
        },
    ),
    OrderStatus.APPROVED: frozenset(
        {
            OrderStatus.PAID,
            OrderStatus.CANCELLED,
        },
    ),
    OrderStatus.PAID: frozenset(
        {
            OrderStatus.COMPLETED,
            OrderStatus.CANCELLED,
        },
    ),
    # ---- 以下三个是终端态：没有出边 ------------------------------------------
    OrderStatus.REJECTED: frozenset(),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.COMPLETED: frozenset(),
}


@dataclass(frozen=True)
class TransitionCheck:
    """一次状态迁移的判定结果。

    ⚠️ 用「结果对象」而不是「返回 bool + 抛异常」：调用方（工具层）需要把
    **拒绝原因**告诉用户，而「能/不能」这个布尔值本身信息量为零。
    返回结构化结果让「拒绝原因」成为编译期就必须处理的东西 ——
    拿 bool 的写法很容易变成 ``if not ok: return "操作失败"``，
    而用户看到「操作失败」四个字时完全无从下手。

    Attributes:
        allowed: 是否允许该迁移。
        reason: 面向用户的中文说明（允许时为空）。
    """

    allowed: bool
    reason: str = ""


def check_transition(current: OrderStatus, target: OrderStatus) -> TransitionCheck:
    """判定订单能否从 ``current`` 迁移到 ``target``。

    Args:
        current (`OrderStatus`): 当前状态。
        target (`OrderStatus`): 目标状态。

    Returns:
        `TransitionCheck`: 判定结果。

    ⚠️ 「迁移到自己」被**显式允许**（幂等）。理由：网络重试、用户连点两次
    提交，都会产生一次「已处于目标态」的请求。拒绝它会让一次成功的操作
    因为重试而报错 —— 这是分布式系统里最常见的伪故障来源之一。
    """
    if current == target:
        return TransitionCheck(allowed=True)

    allowed_targets = _ALLOWED_TRANSITIONS.get(current, frozenset())
    if target in allowed_targets:
        return TransitionCheck(allowed=True)

    if current.is_terminal:
        return TransitionCheck(
            allowed=False,
            reason=(
                f"订单已是「{current.value}」终态，不能再变更为「{target.value}」。"
                f"如需继续，请新建订单。"
            ),
        )

    return TransitionCheck(
        allowed=False,
        reason=(
            f"订单不能从「{current.value}」直接变更为「{target.value}」。"
            f"当前状态允许的下一步是："
            f"{'、'.join(sorted(s.value for s in allowed_targets)) or '无'}。"
        ),
    )


# ==============================================================================
# 二、差标校验
# ==============================================================================
#: 舱位等级 —— 用于「只能同级或降级比较」的判定。
#:
#: ⚠️ 单独定义序而不是复用枚举定义顺序：``CabinClass`` 的成员顺序是**声明
#: 顺序**，随时可能因为可读性调整而变；而「公务舱高于经济舱」是**业务事实**，
#: 不该随排版变化。两者分开，改动声明顺序就不会悄悄改掉差标判定。
_CABIN_RANK: dict[CabinClass, int] = {
    CabinClass.ECONOMY: 0,
    CabinClass.PREMIUM_ECONOMY: 1,
    CabinClass.BUSINESS: 2,
    CabinClass.FIRST: 3,
}


@dataclass(frozen=True)
class PolicyLimit:
    """一条差标上限。

    Attributes:
        max_cabin: 允许的最高舱位。
        max_hotel_price: 酒店单晚上限（元）；``0`` 表示不限。
        max_flight_price: 机票上限（元）；``0`` 表示不限。
        note: 该条差标的说明（进 RAG 引用来源与用户提示）。
    """

    max_cabin: CabinClass = CabinClass.ECONOMY
    max_hotel_price: float = 0.0
    max_flight_price: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class PolicyVerdict:
    """一次差标校验的结论。

    Attributes:
        compliant: 是否符合差标。
        reasons: 不符合的原因列表（符合时为空）。
        advice: 面向用户的**可执行建议**（如「改订 600 元以内的酒店」）。
    """

    compliant: bool
    reasons: list[str]
    advice: str = ""


def check_cabin(cabin: CabinClass, limit: PolicyLimit) -> PolicyVerdict:
    """校验舱位是否超出差标。

    Args:
        cabin (`CabinClass`): 用户选择的舱位。
        limit (`PolicyLimit`): 适用的差标。

    Returns:
        `PolicyVerdict`: 校验结论。

    ⚠️ :attr:`CabinClass.ANY` 视为**符合**：它的含义是「用户没指定」，
    此时不该报「超出差标」——那会让系统对一个尚未做选择的用户报错。
    真正的舱位判定发生在选定具体航班时。
    """
    if cabin == CabinClass.ANY:
        return PolicyVerdict(compliant=True, reasons=[])

    selected_rank = _CABIN_RANK.get(cabin, 0)
    allowed_rank = _CABIN_RANK.get(limit.max_cabin, 0)
    if selected_rank <= allowed_rank:
        return PolicyVerdict(compliant=True, reasons=[])

    # ⚠️ 原因与建议里必须用 ``display_name`` 而不是 ``.value``。
    # ``reasons`` 是**给用户看的话**（卡片与正文直接引用它），
    # 而 ``.value`` 是大写下划线的英文枚举码 —— 实测过用户看到
    # 「BUSINESS 超出差标允许的最高舱位 ECONOMY」这种中英夹杂的结论，
    # 关键信息（到底哪个舱位不合规）反而看不懂。
    return PolicyVerdict(
        compliant=False,
        reasons=[
            f"{cabin.display_name}超出差标允许的最高舱位"
            f"{limit.max_cabin.display_name}",
        ],
        advice=f"请改订{limit.max_cabin.display_name}或更低舱位",
    )


def check_hotel_price(price_per_night: float, limit: PolicyLimit) -> PolicyVerdict:
    """校验酒店房价是否超出差标。

    Args:
        price_per_night (`float`): 单晚房价（元）。
        limit (`PolicyLimit`): 适用的差标。

    Returns:
        `PolicyVerdict`: 校验结论。

    ⚠️ ``limit.max_hotel_price <= 0`` 表示**不限**，直接放行。这个约定
    必须与 :class:`PolicyLimit` 的字段说明一致 —— 若这里写成「<=0 即全部
    拒绝」，则一条「该城市无上限」的差标会让用户订不了任何酒店，而且
    报错信息（「超出上限 0 元」）看起来像数据错误，排查方向会被带偏。
    """
    if limit.max_hotel_price <= 0:
        return PolicyVerdict(compliant=True, reasons=[])

    if price_per_night <= limit.max_hotel_price:
        return PolicyVerdict(compliant=True, reasons=[])

    over = price_per_night - limit.max_hotel_price
    return PolicyVerdict(
        compliant=False,
        reasons=[
            f"房价 {price_per_night:.0f} 元/晚 超出差标 "
            f"{limit.max_hotel_price:.0f} 元/晚（超出 {over:.0f} 元）",
        ],
        advice="请选择差标范围内的酒店，或提交特批申请",
    )


def check_flight_price(price: float, limit: PolicyLimit) -> PolicyVerdict:
    """校验机票价格是否超出差标。

    Args:
        price (`float`): 票价（元）。
        limit (`PolicyLimit`): 适用的差标。

    Returns:
        `PolicyVerdict`: 校验结论。
    """
    if limit.max_flight_price <= 0:
        return PolicyVerdict(compliant=True, reasons=[])

    if price <= limit.max_flight_price:
        return PolicyVerdict(compliant=True, reasons=[])

    return PolicyVerdict(
        compliant=False,
        reasons=[
            f"票价 {price:.0f} 元 超出差标 {limit.max_flight_price:.0f} 元",
        ],
        advice="请选择价格更低的航班，或提交特批申请",
    )


def check_transport_mode(
    mode: TransportMode,
    *,
    distance_km: float,
    train_preferred_km: float = 1200.0,
) -> PolicyVerdict:
    """按里程校验交通方式是否符合差旅惯例。

    规则：**约 1200 公里以内建议高铁**（含安检与往返机场的时间，短途高铁
    往往总耗时更短且更便宜），超出则不限。

    Args:
        mode (`TransportMode`): 用户选择的交通方式。
        distance_km (`float`): 两地直线距离（公里）。
        train_preferred_km (`float`): 「建议高铁」的里程阈值。

    Returns:
        `PolicyVerdict`: 校验结论。

    ⚠️ 这是**建议性**规则，因此 :attr:`PolicyVerdict.compliant` 恒为 True ——
    它只往 ``advice`` 里写建议，从不拒绝。把建议做成拒绝是过度管控：员工
    有正当理由坐飞机（如必须当天往返且高铁无合适车次），系统不该堵死，
    而应记录并交由审批判断。

    ⚠️ 阈值作为**参数**而不是常量硬编码在这里：不同企业的管控线不同，
    而规则引擎本身不读配置（见模块文档）。由调用方从配置传入。
    """
    if mode == TransportMode.FLIGHT and distance_km <= train_preferred_km:
        return PolicyVerdict(
            compliant=True,
            reasons=[],
            advice=(
                f"两地约 {distance_km:.0f} 公里，在 {train_preferred_km:.0f} 公里以内，"
                f"建议优先考虑高铁"
            ),
        )
    return PolicyVerdict(compliant=True, reasons=[])


__all__ = [
    "PolicyLimit",
    "PolicyVerdict",
    "TransitionCheck",
    "check_cabin",
    "check_flight_price",
    "check_hotel_price",
    "check_transition",
    "check_transport_mode",
]
