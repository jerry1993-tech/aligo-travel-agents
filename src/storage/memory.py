# -*- coding: utf-8 -*-
"""差旅业务数据的**内存实现**：让 P3 的工具集在没有数据库时也能跑通。

文件职责：
    实现 :mod:`src.domain.repository` 里那几个协议的内存版本，并附带一套
    **确定性**的模拟数据生成器。

上下游依赖：
    - 上游：:mod:`src.domain.repository` / :mod:`src.domain.entities`。
    - 下游：``src/tools/``（工具默认用它）、``tests/``（几乎所有工具单测）。

═══ 为什么需要它 ═══

P4 才会接 Postgres。但 P3 的工具集不能因此空转 —— 那样「工具可用」这件事
就要等到 P4 才能验证，而 P4 又依赖 P3。更实际的是：``make test`` 必须能在
没有 Docker 的机器上跑绿（这是 P1 定下的硬约束），而工具的单测若都要连库，
这条约束当场失效。

═══ ⚠️ 模拟数据必须是**确定性**的 ═══

这是本模块最要紧的一条。三条具体要求：

1. **不许用 ``random``**。用随机数会让「同一份输入跑两次得到不同价格」，
   而价格会进快照测试、进演示录屏、进评测集 —— 任何一处不稳定，排查时
   都会先怀疑业务逻辑，浪费大量时间。
2. **不许用 ``hash()``**。``hash("北京")`` 在不同进程里**结果不同**：
   Python 对 str 的哈希默认加了随机盐（``PYTHONHASHSEED``），这是抵御
   哈希碰撞攻击的设计。用它来生成价格，症状是「重启服务后所有报价都变了」。
   本模块统一用 :func:`_stable_int`（基于 md5）取稳定散列。
3. **不许用系统时间**。时间戳由调用方传入，或留空。数据本身不带「现在几点」。

这样做的代价是数据「假得很规整」，好处是**任何一处不一致都必然是真的 bug**。
"""

from __future__ import annotations

import hashlib

from src.domain.entities import ApprovalRequest, HotelOption, TransportOption, TravelOrder
from src.domain.enums import CabinClass, OrderStatus, TransportMode
from src.domain.rules import PolicyLimit

#: 默认差标。没配差标的用户按它执行。
DEFAULT_POLICY_LIMIT = PolicyLimit(
    max_cabin=CabinClass.ECONOMY,
    max_hotel_price=600.0,
    max_flight_price=2000.0,
    note="默认差标：经济舱，酒店单晚不超过 600 元，机票不超过 2000 元。",
)

#: 各城市的机场/车站三字码，用于拼出看着像真的班次号。
#:
#: ⚠️ 没收录的城市会退化成一个由城市名派生的**稳定**三字母码（见
#: :func:`_city_code`），而不是固定回退成 ``XXX``。回退成同一个值的后果是
#: 「杭州→XXX」这类看着像 bug 的输出，而且多个未知城市会互相撞码。
_CITY_CODES: dict[str, str] = {
    "北京": "BJS",
    "上海": "SHA",
    "杭州": "HGH",
    "广州": "CAN",
    "深圳": "SZX",
    "成都": "CTU",
    "西安": "SIA",
    "南京": "NKG",
    "武汉": "WUH",
    "重庆": "CKG",
}

#: 各城市常见酒店商圈，用于生成酒店数据的区域字段。
_HOTEL_AREAS: dict[str, tuple[str, ...]] = {
    "北京": ("国贸", "中关村", "望京"),
    "上海": ("陆家嘴", "静安寺", "虹桥"),
    "杭州": ("西湖", "滨江", "未来科技城"),
    "广州": ("珠江新城", "天河", "琶洲"),
    "深圳": ("福田", "南山", "宝安"),
}


def _stable_int(*parts: str, modulo: int) -> int:
    """由若干字符串算出一个**跨进程稳定**的整数。

    ⚠️ 这是本模块确定性契约的实现。**不要用内置 ``hash()`` 替代** ——
    Python 对 str 的哈希带随机盐，同一份输入在不同进程里结果不同
    （见模块文档）。md5 在这里不是密码学用途，只是要一个稳定的散列函数，
    用它是图它够快、够短、标准库里就有。

    Args:
        *parts (`str`): 参与散列的字符串片段。
        modulo (`int`): 取模的模数（决定结果范围）。

    Returns:
        `int`: ``[0, modulo)`` 区间内的稳定整数。

    Raises:
        ValueError: ``modulo`` 小于 1 时。取模 0 会直接 ``ZeroDivisionError``，
        但那个报错完全看不出是哪来的，不如在这里拦下。
    """
    if modulo < 1:
        raise ValueError(f"modulo 必须 >= 1，实际为 {modulo}")
    digest = hashlib.md5("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _city_code(city: str) -> str:
    """取城市的三字码；未知城市由城市名**稳定**派生一个。

    ⚠️ 派生而不是回退成常量：回退会让所有未知城市显示同一个码，
    看起来像数据串了。派生至少能保证不同城市不同码。

    Args:
        city (`str`): 城市名。

    Returns:
        `str`: 三个大写字母。
    """
    known = _CITY_CODES.get(city)
    if known:
        return known
    if not city:
        return "XXX"
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    seed = _stable_int(city, modulo=26 * 26 * 26)
    return "".join(
        letters[(seed // (26**i)) % 26] for i in (2, 1, 0)
    )


# ==============================================================================
# 一、交通
# ==============================================================================


def generate_transport_options(
    *,
    origin: str,
    destination: str,
    depart_date: str,
    mode: TransportMode = TransportMode.ANY,
) -> list[TransportOption]:
    """为一条线路**确定性地**生成若干交通选项。

    ⚠️ 生成的数量与价格只取决于 ``(origin, destination, depart_date)`` ——
    同一组入参永远得到同一份结果（确定性契约，见模块文档）。

    ⚠️ ``depart_date`` 参与散列，所以同一条线路**不同日期价格不同**。
    这看着像多余，其实很必要：若价格与日期无关，演示时「换个日期查一下」
    会得到一模一样的结果，评测集里「比较不同日期」的用例也就失去了意义。

    Args:
        origin (`str`): 出发城市。
        destination (`str`): 目的城市。
        depart_date (`str`): 出发日期，``YYYY-MM-DD``。
        mode (`TransportMode`): 交通方式；``ANY`` 表示航班与车次都生成。

    Returns:
        `list[TransportOption]`: 按价格升序排列；航线两端任一为空时返回空列表。
    """
    if not origin.strip() or not destination.strip():
        # ⚠️ 缺要素时返回**空列表**而不是抛异常：这是业务结论
        # （信息不足，查不了），不是系统故障。工具层会把它翻译成
        # 「请先告诉我出发城市」这类话术。
        return []

    origin, destination = origin.strip(), destination.strip()
    options: list[TransportOption] = []

    wants_flight = mode in (TransportMode.ANY, TransportMode.FLIGHT)
    wants_train = mode in (TransportMode.ANY, TransportMode.TRAIN)

    if wants_flight:
        # 3 个航班，价格 780~1780，出发时刻 07:30 / 12:15 / 18:40。
        for slot, hour in enumerate((7, 12, 18)):
            price = 780 + _stable_int(origin, destination, depart_date, f"f{slot}", modulo=1000)
            fly_minutes = 120 + _stable_int(origin, destination, "dur", modulo=40)
            depart_hour = hour
            depart_minute = 30 if slot == 0 else (15 if slot == 1 else 40)
            options.append(
                TransportOption(
                    option_id=f"FL-{_city_code(origin)}{_city_code(destination)}-{slot + 1}",
                    mode=TransportMode.FLIGHT,
                    carrier=f"CA{1700 + _stable_int(origin, destination, f'f{slot}', modulo=800)}",
                    origin=origin,
                    destination=destination,
                    depart_at=f"{depart_date} {depart_hour:02d}:{depart_minute:02d}",
                    arrive_at=_add_minutes(
                        depart_date, depart_hour, depart_minute, fly_minutes,
                    ),
                    price=float(price),
                    cabin=CabinClass.ECONOMY,
                    seats_left=_stable_int(origin, destination, f"seat{slot}", modulo=40),
                ),
            )

    if wants_train:
        # 2 趟高铁，价格 420~760，比飞机便宜。
        for slot, hour in enumerate((8, 14)):
            price = 420 + _stable_int(origin, destination, depart_date, f"t{slot}", modulo=340)
            trip_minutes = 270 + _stable_int(origin, destination, "tdur", modulo=90)
            depart_hour = hour
            depart_minute = 5 if slot == 0 else 20
            options.append(
                TransportOption(
                    option_id=f"TR-{_city_code(origin)}{_city_code(destination)}-{slot + 1}",
                    mode=TransportMode.TRAIN,
                    carrier=f"G{1 + _stable_int(origin, destination, f't{slot}', modulo=300)}",
                    origin=origin,
                    destination=destination,
                    depart_at=f"{depart_date} {depart_hour:02d}:{depart_minute:02d}",
                    arrive_at=_add_minutes(
                        depart_date, depart_hour, depart_minute, trip_minutes,
                    ),
                    price=float(price),
                    cabin=CabinClass.ECONOMY,
                    seats_left=_stable_int(origin, destination, f"tseat{slot}", modulo=60),
                ),
            )

    # ⚠️ 排序键带上 option_id 兜底：两条价格相同时，若只按 price 排，
    # 顺序取决于生成顺序（而这在实现变动时会变），会让快照测试不稳定。
    return sorted(options, key=lambda o: (o.price, o.option_id))


def _add_minutes(date: str, hour: int, minute: int, delta: int) -> str:
    """给一个 ``YYYY-MM-DD HH:MM`` 时刻加上若干分钟。

    ⚠️ 用 ``datetime`` 做真正的日期算术而不是「小时相加」：跨零点的班次
    （比如 23:50 起飞、飞 2 小时）必须把日期也推进一天，否则会算出
    ``2026-10-08 25:50`` 这种非法时刻，而它在界面上会直接显示成乱码。

    Args:
        date (`str`): 日期部分。
        hour (`int`): 小时。
        minute (`int`): 分钟。
        delta (`int`): 要加的分钟数。

    Returns:
        `str`: ``YYYY-MM-DD HH:MM``；日期本身解析失败时原样返回一个保守值。
    """
    from datetime import datetime, timedelta

    try:
        start = datetime.strptime(f"{date} {hour:02d}:{minute:02d}", "%Y-%m-%d %H:%M")
    except ValueError:
        return f"{date} {hour:02d}:{minute:02d}"
    return (start + timedelta(minutes=delta)).strftime("%Y-%m-%d %H:%M")


# ==============================================================================
# 二、酒店
# ==============================================================================


def generate_hotel_options(*, city: str, area: str = "") -> list[HotelOption]:
    """为一座城市**确定性地**生成若干酒店选项。

    Args:
        city (`str`): 城市。
        area (`str`): 区域偏好；空串表示不限。

    Returns:
        `list[HotelOption]`: 按价格升序；城市为空时返回空列表。

    ⚠️ ``area`` 作为**过滤条件**参与：传了就只返回该区域的酒店。若该区域
    没有数据，返回**空列表**而不是「忽略区域返回全部」—— 后者会让用户
    以为系统听懂了他的偏好，实际给了别的地方的酒店，是更糟的体验。
    """
    if not city.strip():
        return []
    city = city.strip()

    areas = _HOTEL_AREAS.get(city, ("市中心", "商务区", "高铁站"))
    if area.strip():
        wanted = area.strip()
        # ⚠️ 用**包含**匹配而不是相等：用户会说「国贸附近」「西湖边」，
        # 而数据里只有「国贸」「西湖」。精确匹配会让这些最常见的说法
        # 一律查不到结果。
        areas = tuple(a for a in areas if wanted in a or a in wanted)
        if not areas:
            return []

    options: list[HotelOption] = []
    for index, zone in enumerate(areas):
        for tier, (star, base) in enumerate(((3, 320), (4, 520), (5, 880))):
            price = base + _stable_int(city, zone, str(tier), modulo=180)
            options.append(
                HotelOption(
                    option_id=f"HT-{_city_code(city)}-{index + 1}{tier + 1}",
                    name=f"{city}{zone}{('商务' if star == 3 else '国际' if star == 4 else '大酒店')}",
                    city=city,
                    area=zone,
                    price_per_night=float(price),
                    star=star,
                    distance_km=round(1.0 + _stable_int(city, zone, str(tier), "d", modulo=80) / 10, 1),
                ),
            )

    return sorted(options, key=lambda o: (o.price_per_night, o.option_id))


# ==============================================================================
# 三、仓储实现
# ==============================================================================


class InMemoryTransportRepository:
    """:class:`~src.domain.repository.TransportRepository` 的内存实现。"""

    async def search(
        self,
        *,
        origin: str,
        destination: str,
        depart_date: str,
        mode: TransportMode = TransportMode.ANY,
    ) -> list[TransportOption]:
        """见协议文档。

        ⚠️ 本方法体内没有任何 ``await``，但**必须**声明成 ``async def`` ——
        协议要求异步（见 :mod:`src.domain.repository` 的说明），而 P4 的
        Postgres 实现确实会 ``await``。签名不一致会让两者无法互换。
        """
        return generate_transport_options(
            origin=origin,
            destination=destination,
            depart_date=depart_date,
            mode=mode,
        )


class InMemoryHotelRepository:
    """:class:`~src.domain.repository.HotelRepository` 的内存实现。"""

    async def search(
        self,
        *,
        city: str,
        area: str = "",
        check_in: str = "",
        check_out: str = "",
    ) -> list[HotelOption]:
        """见协议文档。

        ⚠️ ``check_in`` / ``check_out`` 被**刻意忽略**。内存实现没有库存
        概念，假装按日期过滤会让调用方以为「换日期能查到不同结果」，
        而在真实实现接上之前那是假的。参数留在签名里，是因为协议要求 ——
        调用方现在就该按最终形态传参，将来切换实现时不用改调用点。
        """
        del check_in, check_out  # 见上方说明：内存实现不使用
        return generate_hotel_options(city=city, area=area)


class StaticPolicyRepository:
    """差标仓储：按用户返回差标，未配置的用默认值。"""

    def __init__(self, limits: dict[str, PolicyLimit] | None = None) -> None:
        """初始化。

        Args:
            limits (`dict[str, PolicyLimit] | None`): 用户 → 差标；``None``
                表示所有人都用默认差标。
        """
        self._limits = dict(limits or {})

    async def limit_for(self, *, user_id: str) -> PolicyLimit:
        """见协议文档。"""
        return self._limits.get(user_id, DEFAULT_POLICY_LIMIT)


class InMemoryOrderRepository:
    """订单仓储（内存）。"""

    def __init__(self, orders: list[TravelOrder] | None = None) -> None:
        """初始化。

        Args:
            orders (`list[TravelOrder] | None`): 初始订单。
        """
        # ⚠️ 内部用 dict 存而不是 list：``get`` / ``save`` 都要按 id 定位，
        # 用 list 会退化成每次 O(n) 扫描，而 ``save`` 在批量下单时会很频繁。
        self._orders: dict[str, TravelOrder] = {o.order_id: o for o in (orders or [])}

    async def list_for(self, *, user_id: str, kind: str = "") -> list[TravelOrder]:
        """见协议文档。按创建时间倒序。"""
        matched = [
            order
            for order in self._orders.values()
            if order.user_id == user_id and (not kind or order.kind == kind)
        ]
        # ⚠️ 用 ``created_at`` 倒序，且以 ``order_id`` 兜底：同一秒创建的订单
        # 若不兜底，顺序取决于 dict 的插入顺序（即调用顺序），而那是会变的。
        return sorted(matched, key=lambda o: (o.created_at, o.order_id), reverse=True)

    async def get(self, *, order_id: str) -> TravelOrder | None:
        """见协议文档。"""
        return self._orders.get(order_id)

    async def save(self, order: TravelOrder) -> TravelOrder:
        """见协议文档。"""
        self._orders[order.order_id] = order
        return order


class InMemoryApprovalRepository:
    """出差申请单仓储（内存）。"""

    def __init__(self, requests: list[ApprovalRequest] | None = None) -> None:
        """初始化。

        Args:
            requests (`list[ApprovalRequest] | None`): 初始申请单。
        """
        self._requests: dict[str, ApprovalRequest] = {r.request_id: r for r in (requests or [])}

    async def list_for(
        self,
        *,
        user_id: str,
        status: OrderStatus | None = None,
    ) -> list[ApprovalRequest]:
        """见协议文档。按创建时间倒序。"""
        matched = [
            request
            for request in self._requests.values()
            if request.user_id == user_id and (status is None or request.status == status)
        ]
        return sorted(matched, key=lambda r: (r.created_at, r.request_id), reverse=True)

    async def get(self, *, request_id: str) -> ApprovalRequest | None:
        """见协议文档。"""
        return self._requests.get(request_id)

    async def save(self, request: ApprovalRequest) -> ApprovalRequest:
        """见协议文档。"""
        self._requests[request.request_id] = request
        return request


__all__ = [
    "DEFAULT_POLICY_LIMIT",
    "InMemoryApprovalRepository",
    "InMemoryHotelRepository",
    "InMemoryOrderRepository",
    "InMemoryTransportRepository",
    "StaticPolicyRepository",
    "generate_hotel_options",
    "generate_transport_options",
]
