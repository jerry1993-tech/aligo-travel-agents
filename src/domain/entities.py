# -*- coding: utf-8 -*-
"""差旅**业务实体**：交通工具、酒店、订单、申请单。

文件职责：
    定义业务数据的**形状**。这些是「事实」——一旦产生就要落库、要对账、
    要能被审计，与 :mod:`src.domain.schemas` 里那些「给模型看的中间结构」
    （意图、路由决策）有本质区别。

上下游依赖：
    - 上游：:mod:`src.domain.enums`。
    - 下游：``src/domain/repository.py``（存储协议）、``src/tools/``
      （工具读写的对象）、``src/storage/``（落库实现）。

═══ 为什么与 schemas.py 分开 ═══

两个原因，都很实际：

1. **生命周期不同**。``IntentDecision`` 这类结构活不过一次回复；``TravelOrder``
   要活到对账那天。混在一个文件里，很容易有人给订单加个字段只为「让模型
   看得更清楚」，而那个字段根本不该入库。
2. **序列化要求相反**。schemas 里的模型要**尽量少的字段和描述**（它们每轮
   都要进 prompt，见 ``src/domain/_doc.py`` 的说明）；实体的字段则是越全越好
   （排障、对账都要用）。一个追求「说得少」，一个追求「记全」，放在一起会
   互相拉扯。

═══ 统一用 ``dataclass(frozen=True)`` 而不是 pydantic ═══

与 :mod:`src.domain.rules` 的 ``PolicyLimit`` / ``PolicyVerdict`` 一致。
实体不需要 pydantic 的校验/序列化能力（那是给「来自模型或 HTTP 的不可信
输入」准备的），而它们要的是**不可变**：订单一旦创建就不该被就地修改，
状态迁移必须走 :mod:`src.domain.rules` 的校验。冻结之后，「顺手改一下」
会在运行时立刻报错，而不是悄悄绕过状态机。

⚠️ 因此**更新实体要构造新对象**（``dataclasses.replace``）。这是刻意的
摩擦：状态迁移本来就该是一件事、一次校验，而不是「读出来、改字段、写回去」
三步里悄悄漏掉第二步。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.domain.enums import CabinClass, OrderStatus, TransportMode

# ==============================================================================
# 一、交通与住宿（搜索结果）
# ==============================================================================


@dataclass(frozen=True)
class TransportOption:
    """一条交通选项（飞机或火车）。

    ⚠️ 航班与车次**合成一个类型**，用 :attr:`mode` 区分，而不是分成
    ``FlightOption`` / ``TrainOption`` 两个类。理由：工具返回给用户的是
    「从杭州到北京的所有走法」这一条**混合榜单**，用户按价格或时间挑，
    而不是先决定坐飞机再看航班。两个类会逼着每个下游（排序、渲染、
    差标校验）都写一遍「先判断类型再分别处理」，而它们的字段**本来就
    高度重合**。真正不同的只有编号的叫法（航班号 vs 车次），那个用
    :attr:`carrier` 一个字段就够了。

    Attributes:
        option_id: 选项标识，用于下单时回指。
        mode: 交通方式（只可能是 FLIGHT 或 TRAIN；``ANY`` 不是一种实际交通）。
        carrier: 承运方编号 —— 航班号（``CA1701``）或车次（``G20``）。
        origin: 出发城市。
        destination: 目的城市。
        depart_at: 出发时刻，``YYYY-MM-DD HH:MM``。
        arrive_at: 到达时刻，同格式。
        price: 票价（元）。
        cabin: 舱位/座席等级。
        seats_left: 余票数；``-1`` 表示未知（有些渠道不返回余票）。
    """

    option_id: str
    mode: TransportMode
    carrier: str
    origin: str
    destination: str
    depart_at: str
    arrive_at: str
    price: float
    cabin: CabinClass = CabinClass.ECONOMY
    seats_left: int = -1

    @property
    def duration_minutes(self) -> int:
        """行程时长（分钟）；时刻解析失败时返回 ``0``。

        ⚠️ 解析失败返回 ``0`` 而不是抛异常。本属性会被**排序**用到，而排序
        里抛异常会让整个搜索结果列表都拿不到 —— 一条脏数据毁掉全部结果。
        返回 ``0`` 的后果只是这条排到最前面，用户看得见、能忽略，
        属于**可恢复**的降级。

        ⚠️ ``0`` 同时也是「真的是零分钟」的值，两者无法区分。接受这个
        模糊性：真实的交通选项不可能零分钟，所以它实际上是个「无效」标记。

        Returns:
            `int`: 分钟数。
        """
        from datetime import datetime

        try:
            start = datetime.strptime(self.depart_at, "%Y-%m-%d %H:%M")
            end = datetime.strptime(self.arrive_at, "%Y-%m-%d %H:%M")
        except (ValueError, TypeError):
            return 0
        return max(0, int((end - start).total_seconds() // 60))

    @property
    def is_sold_out(self) -> bool:
        """是否已无余票。

        ⚠️ ``seats_left == -1``（未知）**不算**售罄。渠道不返回余票是常见的，
        把「不知道」当成「没有」会让大量本来可订的选项被过滤掉。
        这与本项目「歧义一律落到安全一侧」的其它判断方向一致 —— 只是这里
        「安全」的意思是「不要让用户少看到选项」，因为真正的拦截在下单时
        还会发生一次（那里才是权威判定）。

        Returns:
            `bool`: 售罄返回 True。
        """
        return self.seats_left == 0


@dataclass(frozen=True)
class HotelOption:
    """一条酒店选项。

    Attributes:
        option_id: 选项标识。
        name: 酒店名。
        city: 所在城市。
        area: 商圈/区域（如「国贸」）。
        price_per_night: 单晚价格（元）。
        star: 星级（0 表示未知）。
        distance_km: 距市中心的距离（公里）；``-1`` 表示未知。
    """

    option_id: str
    name: str
    city: str
    area: str
    price_per_night: float
    star: int = 0
    distance_km: float = -1.0


# ==============================================================================
# 二、订单与申请单
# ==============================================================================


@dataclass(frozen=True)
class TravelOrder:
    """一笔差旅订单（机票 / 火车票 / 酒店）。

    ⚠️ 三种订单**合成一个类型**，用 :attr:`kind` 区分。理由与
    :class:`TransportOption` 融合航班火车相同：它们共享**同一个状态机**
    （:class:`~src.domain.enums.OrderStatus`，见
    :mod:`src.domain.rules` 的迁移表），也共享同一套「查订单 / 取消订单」
    的操作。拆成三个类会让状态机要维护三份，而它们必须始终一致 ——
    这正是最容易长期腐烂的地方。

    Attributes:
        order_id: 订单号。
        user_id: 归属用户（多租户隔离的依据）。
        kind: 订单类型（``flight`` / ``train`` / ``hotel``）。
        status: 当前状态。
        title: 面向用户的摘要（如「杭州 → 北京 CA1701」）。
        amount: 金额（元）。
        detail: 明细（舱位、日期等），结构随 ``kind`` 而变。
        created_at: 创建时刻，ISO 字符串。
        updated_at: 最后更新时刻，ISO 字符串。
    """

    order_id: str
    user_id: str
    kind: str
    status: OrderStatus
    title: str
    amount: float
    detail: dict[str, str] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    @property
    def is_cancellable(self) -> bool:
        """当前状态下是否允许取消。

        ⚠️ 这里的判断**只回答「状态机允不允许」**，不回答「业务上该不该」
        （比如已出票的机票可能有退票费）。后者需要查具体规则，不该塞进
        一个属性里。属性保持廉价、无副作用，调用方再叠加业务判断。

        Returns:
            `bool`: 可取消返回 True。
        """
        from src.domain.rules import check_transition

        return check_transition(self.status, OrderStatus.CANCELLED).allowed


@dataclass(frozen=True)
class ApprovalRequest:
    """一张出差申请单。

    ⚠️ 与 :class:`TravelOrder` **刻意分成两个类型**，尽管字段高度相似。
    它们看着像，生命周期却不同：申请单走的是「提交 → 审批 → 通过/驳回」，
    订单走的是「草稿 → 已支付 → 已完成」。它们**共用** ``OrderStatus``
    这套状态值（由 :mod:`src.domain.rules` 的迁移表统一约束），但
    「什么算终态」「谁能改」并不相同。合并成一个类会让审批逻辑和下单逻辑
    挤在一起，将来任何一方要加字段都得考虑另一方。

    Attributes:
        request_id: 申请单号。
        user_id: 申请人。
        title: 事由摘要（如「10 月北京客户拜访」）。
        status: 当前状态。
        amount: 预估金额（元）。
        destination: 目的地。
        depart_date: 出发日期，``YYYY-MM-DD``。
        days: 天数。
        created_at: 提交时刻，ISO 字符串。
        updated_at: 最后更新时刻，ISO 字符串。
        approver_note: 审批意见（驳回时通常是原因）。
    """

    request_id: str
    user_id: str
    title: str
    status: OrderStatus
    amount: float = 0.0
    destination: str = ""
    depart_date: str = ""
    days: int = 0
    created_at: str = ""
    updated_at: str = ""
    approver_note: str = ""


__all__ = [
    "ApprovalRequest",
    "HotelOption",
    "TransportOption",
    "TravelOrder",
]
