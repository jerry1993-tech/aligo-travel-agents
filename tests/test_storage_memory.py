# -*- coding: utf-8 -*-
"""内存业务数据（``src/storage/memory.py``）的测试。

═══ 本文件的核心是**确定性** ═══

模拟数据必须是确定性的，这在本项目里不是洁癖，而是三条具体的理由：

1. 价格会进快照测试、进演示录屏、进评测集 —— 任何一处不稳定，排查时都会
   先怀疑业务逻辑；
2. ``make test`` 必须能在没有 Docker 的机器上跑绿，所以工具单测都跑在
   这份内存数据上，而单测断言里到处是具体数字；
3. 不确定的数据会让「评测集分数波动」变得无法归因 —— 而 P5 要靠评测集
   来判断改动是好是坏。

⚠️ 最要紧的一条是 :func:`test_generation_is_stable_across_processes`。
它是**唯一**能抓住「有人用 ``hash()`` 替代了 ``_stable_int``」的测试：
``hash("北京")`` 在同一个进程里是稳定的，所以普通的「跑两次对比」会
愉快地通过，只有跨进程跑才会露出马脚。而症状是「重启服务后所有报价都变了」。
"""

from __future__ import annotations

import asyncio
import subprocess
import sys

import pytest

from src.domain import OrderStatus
from src.domain.entities import ApprovalRequest, TravelOrder
from src.domain.enums import CabinClass, TransportMode
from src.storage.memory import (
    DEFAULT_POLICY_LIMIT,
    InMemoryApprovalRepository,
    InMemoryHotelRepository,
    InMemoryOrderRepository,
    InMemoryTransportRepository,
    StaticPolicyRepository,
    _city_code,
    _stable_int,
    generate_hotel_options,
    generate_transport_options,
)

ROUTE = {"origin": "杭州", "destination": "北京", "depart_date": "2026-10-08"}


# ---------------------------------------------------------------------------
# 一、跨进程确定性（本文件最重要的一组）
# ---------------------------------------------------------------------------
def test_generation_is_stable_across_processes() -> None:
    """★★ 换一个**进程**跑，生成的数据必须**逐字相同**。

    ⚠️ 为什么必须是子进程：内置 ``hash("北京")`` 在**同一个进程内**是稳定的
    —— 它只在进程启动时被随机加盐一次。所以「在同一个测试进程里跑两次对比」
    这个看起来很自然的写法，**完全抓不住**用 ``hash()`` 代替
    :func:`_stable_int` 的错误。

    只有换进程（``PYTHONHASHSEED`` 才会重新取值）才能暴露它。而一旦真的
    用了 ``hash()``，症状是「重启服务后所有报价都变了」—— 没有异常、
    没有日志，用户只会觉得「这系统今天报价怎么不一样了」。

    ⚠️ 三个不同的 ``PYTHONHASHSEED`` 而不是两个：两个的时候有 1/2 的概率
    随机撞上同一个盐而假绿；三个把假绿的概率压到 1/6 以下。真正的确定性
    实现三个都过，用 ``hash()`` 的实现几乎必然有一个红。
    """
    code = (
        "from src.storage.memory import generate_transport_options, generate_hotel_options;"
        "from src.domain.enums import TransportMode;"
        "opts = generate_transport_options(origin='杭州', destination='北京',"
        " depart_date='2026-10-08', mode=TransportMode.ANY);"
        "hotels = generate_hotel_options(city='北京');"
        "print('|'.join(f'{o.option_id}:{o.price:g}:{o.carrier}:{o.seats_left}' for o in opts));"
        "print('|'.join(f'{h.option_id}:{h.price_per_night:g}:{h.name}' for h in hotels))"
    )

    outputs = []
    for seed in ("0", "1", "12345"):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            cwd=".",
        )
        assert result.returncode == 0, f"PYTHONHASHSEED={seed} 时探测脚本失败：{result.stderr}"
        outputs.append(result.stdout)

    assert len(set(outputs)) == 1, (
        "不同 PYTHONHASHSEED 下生成了不同的数据 —— 说明某处用了内置 hash()。\n"
        + "\n".join(f"  seed 输出：{o[:120]}" for o in outputs)
    )


def test_stable_int_does_not_use_builtin_hash() -> None:
    """⚠️ :func:`_stable_int` 的结果与 ``PYTHONHASHSEED`` 无关。

    单进程内验证「同入参同结果」，跨进程由上面那条守。两条一起才有意义：
    这条能快速定位到 ``_stable_int`` 本身，而上面那条覆盖整条生成链。
    """
    assert _stable_int("北京", modulo=1000) == _stable_int("北京", modulo=1000)


def test_generation_is_repeatable_within_a_process() -> None:
    """同一组入参连续生成两次，结果逐字相同。"""
    first = generate_transport_options(**ROUTE, mode=TransportMode.ANY)
    second = generate_transport_options(**ROUTE, mode=TransportMode.ANY)

    assert [(o.option_id, o.price, o.carrier) for o in first] == \
           [(o.option_id, o.price, o.carrier) for o in second]


def test_different_dates_produce_different_prices() -> None:
    """⚠️ 价格与日期相关。

    若价格与日期无关，演示时「换个日期查一下」会得到一模一样的结果，
    评测集里「比较不同日期」的用例也就失去了意义 —— 而那正是差旅助手的
    一个核心卖点（错峰更便宜）。
    """
    prices = {
        date: [o.price for o in generate_transport_options(
            origin="杭州", destination="北京", depart_date=date, mode=TransportMode.FLIGHT)]
        for date in ("2026-10-08", "2026-10-09", "2026-10-10")
    }

    assert len({tuple(v) for v in prices.values()}) > 1, f"三个日期价格完全相同：{prices}"


def test_different_routes_produce_different_prices() -> None:
    """不同线路价格不同 —— 否则数据看起来像「所有线路一个价」。"""
    hz_bj = [o.price for o in generate_transport_options(**ROUTE, mode=TransportMode.FLIGHT)]
    sh_bj = [o.price for o in generate_transport_options(
        origin="上海", destination="北京", depart_date="2026-10-08", mode=TransportMode.FLIGHT)]

    assert hz_bj != sh_bj


# ---------------------------------------------------------------------------
# 二、交通数据
# ---------------------------------------------------------------------------
def test_any_mode_returns_both_flights_and_trains() -> None:
    """``mode=ANY`` 时航班与火车都给。"""
    options = generate_transport_options(**ROUTE, mode=TransportMode.ANY)
    modes = {o.mode for o in options}

    assert modes == {TransportMode.FLIGHT, TransportMode.TRAIN}


def test_mode_filter_restricts_the_result() -> None:
    """``mode`` 过滤真的生效。"""
    for mode, expected in ((TransportMode.FLIGHT, {TransportMode.FLIGHT}),
                           (TransportMode.TRAIN, {TransportMode.TRAIN})):
        options = generate_transport_options(**ROUTE, mode=mode)
        assert {o.mode for o in options} == expected


def test_train_is_cheaper_than_flight() -> None:
    """⚠️ 火车整体比飞机便宜 —— 这条不是业务规则，是**数据合理性**。

    数据若违反常识（比如高铁比头等舱还贵），演示时会显得假，而模型也会
    基于这些数字给出反常识的建议。这里守住生成区间的设定。
    """
    options = generate_transport_options(**ROUTE, mode=TransportMode.ANY)
    cheapest_train = min(o.price for o in options if o.mode is TransportMode.TRAIN)
    cheapest_flight = min(o.price for o in options if o.mode is TransportMode.FLIGHT)

    assert cheapest_train < cheapest_flight


def test_options_are_sorted_by_price() -> None:
    """按价格升序；同价时按单号兜底。

    ⚠️ 排序键带上单号：两条价格相同时若只按 price 排，顺序就取决于生成
    顺序，而这在实现变动时会变 —— 快照测试会莫名其妙地红。
    """
    options = generate_transport_options(**ROUTE, mode=TransportMode.ANY)
    keys = [(o.price, o.option_id) for o in options]

    assert keys == sorted(keys)


def test_option_ids_are_unique() -> None:
    """单号唯一 —— 撞号会让「选中第几个」变得无法表达。"""
    options = generate_transport_options(**ROUTE, mode=TransportMode.ANY)
    ids = [o.option_id for o in options]

    assert len(ids) == len(set(ids))


def test_arrival_is_after_departure() -> None:
    """⚠️ 到达时刻晚于出发时刻。"""
    for o in generate_transport_options(**ROUTE, mode=TransportMode.ANY):
        assert o.arrive_at > o.depart_at, f"{o.option_id} 的到达早于出发"


def test_arrival_handles_crossing_midnight() -> None:
    """⚠️ 跨零点的班次要把**日期**也推进一天，不能算出 ``25:50``。

    ``_add_minutes`` 用 ``datetime`` 做真正的日期算术而不是「小时相加」，
    这条守住那个选择。算错的话界面上会直接显示 ``2026-10-08 25:50``。
    """
    from src.storage.memory import _add_minutes

    assert _add_minutes("2026-10-08", 23, 50, 120) == "2026-10-09 01:50"


def test_add_minutes_survives_a_bad_date() -> None:
    """日期解析失败时原样返回一个保守值，不抛异常。"""
    from src.storage.memory import _add_minutes

    assert _add_minutes("not-a-date", 10, 0, 60) == "not-a-date 10:00"


def test_duration_minutes_is_positive_and_sane() -> None:
    """⚠️ 时长是正数且在常识范围内。

    时长直接显示给用户，也会进「哪个更快」的比较。负的或几百小时的值
    会让整个卡片失去可信度。
    """
    for o in generate_transport_options(**ROUTE, mode=TransportMode.ANY):
        assert 0 < o.duration_minutes < 24 * 60, f"{o.option_id} 时长异常：{o.duration_minutes}"


def test_missing_route_returns_empty_instead_of_raising() -> None:
    """⚠️ 缺出发地/目的地时返回**空列表**，不抛异常。

    这是业务结论（信息不足，查不了），不是系统故障。工具层会把它翻译成
    「请先告诉我出发城市」这类话术。
    """
    assert generate_transport_options(origin="", destination="北京", depart_date="2026-10-08") == []
    assert generate_transport_options(origin="杭州", destination="  ", depart_date="2026-10-08") == []


def test_city_codes_are_stable_and_distinct() -> None:
    """⚠️ 未知城市由城市名**稳定**派生三字码，而不是回退成同一个常量。

    回退成 ``XXX`` 的后果是「杭州→XXX」这类看着像 bug 的输出，
    而且多个未知城市会互相撞码。
    """
    unknown_a = _city_code("克拉玛依")
    unknown_b = _city_code("景德镇")

    assert unknown_a != unknown_b, "两个未知城市撞码了"
    assert _city_code("克拉玛依") == unknown_a, "同一个未知城市两次取码不一致"
    assert len(unknown_a) == 3 and unknown_a.isupper()
    assert _city_code("北京") == "BJS", "已收录的城市应当用收录的码"
    assert _city_code("") == "XXX"


# ---------------------------------------------------------------------------
# 三、酒店数据
# ---------------------------------------------------------------------------
def test_hotels_cover_three_star_tiers() -> None:
    """每个商圈都有 3/4/5 星 —— 否则用户想比较档次时无从选起。"""
    stars = {h.star for h in generate_hotel_options(city="北京")}

    assert stars == {3, 4, 5}


def test_hotel_prices_rise_with_star_rating() -> None:
    """⚠️ 星级越高价格越高。

    数据违反常识（五星比三星便宜）时，模型会基于这些数字给出反常识建议。
    """
    hotels = generate_hotel_options(city="北京")
    by_star = {star: min(h.price_per_night for h in hotels if h.star == star) for star in (3, 4, 5)}

    assert by_star[3] < by_star[4] < by_star[5], f"星级与价格不匹配：{by_star}"


def test_hotel_area_filter_narrows_the_result() -> None:
    """``area`` 作为**过滤条件**参与。"""
    all_hotels = generate_hotel_options(city="北京")
    guomao = generate_hotel_options(city="北京", area="国贸")

    assert 0 < len(guomao) < len(all_hotels)
    assert {h.area for h in guomao} == {"国贸"}


def test_hotel_area_matching_is_contains_based() -> None:
    """⚠️ 区域用**包含**匹配而不是相等。

    用户会说「国贸附近」「西湖边」，而数据里只有「国贸」「西湖」。
    精确匹配会让这些最常见的说法一律查不到结果。
    """
    assert generate_hotel_options(city="北京", area="国贸附近")
    assert generate_hotel_options(city="北京", area="国贸")


def test_unknown_area_returns_empty_not_everything() -> None:
    """⚠️ 指定了不存在的区域时返回**空列表**，而不是「忽略区域返回全部」。

    后者会让用户以为系统听懂了他的偏好，实际给了别的地方的酒店 ——
    比明确说「这里没有」更糟，因为这让他基于错误信息做决定。
    """
    assert generate_hotel_options(city="北京", area="完全不存在的地方") == []


def test_unknown_city_gets_fallback_areas() -> None:
    """⚠️ 未收录的城市给一组通用商圈，而不是空结果。

    只收录了 5 个城市，而用户可能查任何城市。空结果会让「未收录」看起来
    像「这个城市没有酒店」。
    """
    hotels = generate_hotel_options(city="克拉玛依")

    assert hotels, "未收录的城市应当仍有数据"
    assert len({h.area for h in hotels}) >= 2


def test_missing_city_returns_empty_instead_of_raising() -> None:
    """缺城市时返回空列表，不抛异常。"""
    assert generate_hotel_options(city="") == []
    assert generate_hotel_options(city="   ") == []


def test_hotel_options_are_sorted_and_unique() -> None:
    """酒店按价格升序，单号唯一。"""
    hotels = generate_hotel_options(city="北京")
    keys = [(h.price_per_night, h.option_id) for h in hotels]

    assert keys == sorted(keys)
    assert len({h.option_id for h in hotels}) == len(hotels)


def test_hotel_option_ids_are_globally_unique_across_cities() -> None:
    """⚠️ 不同城市的单号不能撞。

    单号只在**当前会话**里用于「选中第几个」，但评测与演示里会把多个
    城市的结果放在一起看。撞号会让「HT-BJS-11 到底是哪家」变成一个问题。
    """
    ids = [h.option_id for city in ("北京", "上海", "杭州") for h in generate_hotel_options(city=city)]

    assert len(ids) == len(set(ids)), "跨城市出现了重复单号"


def test_hotel_distance_is_positive_and_plausible() -> None:
    """距离是正数且在一座城市的合理范围内。"""
    for h in generate_hotel_options(city="北京"):
        assert 0 < h.distance_km < 20, f"{h.name} 距离异常：{h.distance_km}"


# ---------------------------------------------------------------------------
# 四、仓储：多租户与基础行为
# ---------------------------------------------------------------------------
def _order(order_id: str, user_id: str, **over: object) -> TravelOrder:
    """造一笔订单。"""
    return TravelOrder(
        order_id=order_id,
        user_id=user_id,
        kind=str(over.pop("kind", "flight")),
        status=over.pop("status", OrderStatus.PAID),  # type: ignore[arg-type]
        title=str(over.pop("title", "杭州 → 北京")),
        amount=float(over.pop("amount", 1088.0)),  # type: ignore[arg-type]
        created_at=str(over.pop("created_at", "2026-09-30T10:00:00")),
    )


def test_order_repository_scopes_by_user() -> None:
    """⚠️ 订单仓储按用户过滤 —— 多租户的第一道防线。"""
    repo = InMemoryOrderRepository([_order("TR-1", "u-a"), _order("TR-2", "u-b")])

    mine = asyncio.run(repo.list_for(user_id="u-a"))

    assert [o.order_id for o in mine] == ["TR-1"]


def test_order_repository_scopes_by_kind() -> None:
    """``kind`` 过滤生效。"""
    repo = InMemoryOrderRepository([
        _order("TR-1", "u-a", kind="flight"),
        _order("HT-1", "u-a", kind="hotel"),
    ])

    assert [o.order_id for o in asyncio.run(repo.list_for(user_id="u-a", kind="hotel"))] == ["HT-1"]


def test_order_repository_get_returns_none_for_a_missing_id() -> None:
    """⚠️ 查不到时返回 ``None`` 而不是抛异常。

    「没有这条记录」是业务结论；抛异常会让调用方把它当成系统故障。
    """
    repo = InMemoryOrderRepository([])

    assert asyncio.run(repo.get(order_id="TR-NOPE")) is None


def test_order_repository_get_does_not_leak_across_users() -> None:
    """⚠️ ``get`` 是按单号查的，**不带用户过滤** —— 所以调用方必须自己校验。

    这条把该事实钉住：它记录的是**当前实现的行为**，而不是「这是对的」。
    把别人的单号猜出来就能查到别人的订单，是真实存在的风险；
    接口层（``/api/v1``）必须靠「单号来自当前用户的列表」来弥补。
    若哪天要给 ``get`` 加上用户过滤，这条测试会红 —— 那时正好回来看这段注释。
    """
    repo = InMemoryOrderRepository([_order("TR-2", "u-b")])

    fetched = asyncio.run(repo.get(order_id="TR-2"))

    assert fetched is not None and fetched.user_id == "u-b"


def test_order_repository_save_is_idempotent_by_id() -> None:
    """⚠️ 同一个单号再存一次是**覆盖**，不是追加。

    追加的话，一次重试会让用户看到两笔一模一样的订单。
    """
    repo = InMemoryOrderRepository([])
    asyncio.run(repo.save(_order("TR-1", "u-a", amount=100)))
    asyncio.run(repo.save(_order("TR-1", "u-a", amount=200)))

    orders = asyncio.run(repo.list_for(user_id="u-a"))
    assert len(orders) == 1
    assert orders[0].amount == 200


def test_approval_repository_scopes_by_user() -> None:
    """申请单仓储按用户过滤。"""
    repo = InMemoryApprovalRepository([
        ApprovalRequest(request_id="AP-1", user_id="u-a", title="我的", status=OrderStatus.DRAFT),
        ApprovalRequest(request_id="AP-2", user_id="u-b", title="别人的", status=OrderStatus.DRAFT),
    ])

    assert [r.request_id for r in asyncio.run(repo.list_for(user_id="u-a"))] == ["AP-1"]


def test_approval_repository_get_returns_none_for_a_missing_id() -> None:
    """查不到时返回 ``None``。"""
    assert asyncio.run(InMemoryApprovalRepository([]).get(request_id="AP-NOPE")) is None


def test_approval_repository_can_filter_by_status() -> None:
    """按状态过滤 —— 审批列表要按状态分组。"""
    repo = InMemoryApprovalRepository([
        ApprovalRequest(request_id="AP-1", user_id="u-a", title="a", status=OrderStatus.DRAFT),
        ApprovalRequest(request_id="AP-2", user_id="u-a", title="b", status=OrderStatus.PENDING_APPROVAL),
    ])

    pending = asyncio.run(repo.list_for(user_id="u-a", status=OrderStatus.PENDING_APPROVAL))

    assert [r.request_id for r in pending] == ["AP-2"]


def test_transport_repository_delegates_to_the_generator() -> None:
    """交通仓储就是生成器的一层异步包装。"""
    repo = InMemoryTransportRepository()

    direct = generate_transport_options(**ROUTE, mode=TransportMode.ANY)
    via_repo = asyncio.run(repo.search(**ROUTE, mode=TransportMode.ANY))

    assert [(o.option_id, o.price) for o in direct] == [(o.option_id, o.price) for o in via_repo]


def test_hotel_repository_delegates_to_the_generator() -> None:
    """酒店仓储同理。"""
    repo = InMemoryHotelRepository()

    direct = generate_hotel_options(city="北京")
    via_repo = asyncio.run(repo.search(city="北京"))

    assert [(h.option_id, h.price_per_night) for h in direct] == \
           [(h.option_id, h.price_per_night) for h in via_repo]


# ---------------------------------------------------------------------------
# 五、差标
# ---------------------------------------------------------------------------
def test_default_policy_is_conservative() -> None:
    """⚠️ 默认差标必须**偏严**，且理由充分。

    默认值会在「用户没有专属差标」时生效。定得宽松，等于给所有人放开；
    定得严苛，则所有人一上来就卡。这里守住「经济舱 + 有明确上限」，
    并要求 note 里说清依据 —— 差标结论要能对用户解释。
    """
    assert DEFAULT_POLICY_LIMIT.max_cabin is CabinClass.ECONOMY
    assert DEFAULT_POLICY_LIMIT.max_hotel_price > 0
    assert DEFAULT_POLICY_LIMIT.max_flight_price > DEFAULT_POLICY_LIMIT.max_hotel_price
    assert DEFAULT_POLICY_LIMIT.note.strip(), "差标必须带一句可对用户解释的说明"


def test_static_policy_repository_falls_back_to_the_default() -> None:
    """⚠️ 没配差标的用户拿到**默认差标**，而不是 ``None`` 或异常。

    返回 ``None`` 会让每个调用点都要判空，而漏判一处就是 AttributeError；
    返回异常则让「新用户第一次查差标」变成一次失败。
    """
    repo = StaticPolicyRepository()

    limit = asyncio.run(repo.limit_for(user_id="never-configured"))

    assert limit == DEFAULT_POLICY_LIMIT


def test_static_policy_repository_honours_a_configured_user() -> None:
    """配了差标的用户拿到自己的那一份。"""
    from src.domain import PolicyLimit

    custom = PolicyLimit(max_cabin=CabinClass.BUSINESS, max_hotel_price=1200.0, note="高管差标")
    repo = StaticPolicyRepository({"u-vip": custom})

    assert asyncio.run(repo.limit_for(user_id="u-vip")) == custom


def test_static_policy_repository_isolates_configured_users() -> None:
    """⚠️ 一个用户的专属差标**不会**泄漏给另一个用户。"""
    from src.domain import PolicyLimit

    repo = StaticPolicyRepository({
        "u-vip": PolicyLimit(max_cabin=CabinClass.BUSINESS, max_hotel_price=1200.0, note="高管差标"),
    })

    assert asyncio.run(repo.limit_for(user_id="u-ordinary")) == DEFAULT_POLICY_LIMIT


# ---------------------------------------------------------------------------
# 六、协议符合性（结构上的，不是行为上的）
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "impl, protocol",
    [
        (InMemoryTransportRepository(), "TransportRepository"),
        (InMemoryHotelRepository(), "HotelRepository"),
        (StaticPolicyRepository(), "PolicyRepository"),
        (InMemoryOrderRepository(), "OrderRepository"),
        (InMemoryApprovalRepository(), "ApprovalRepository"),
    ],
)
def test_implementations_satisfy_their_protocols(impl: object, protocol: str) -> None:
    """⚠️ 每个内存实现都满足对应的 ``@runtime_checkable`` Protocol。

    这是「工具层只依赖协议」这个设计能成立的前提。少了这条，一个拼错方法名
    的实现要等到运行时某次真实调用才暴露 —— 而它很可能只在那一条分支上暴露。
    """
    import src.domain.repository as repo_module

    proto = getattr(repo_module, protocol)
    assert isinstance(impl, proto), f"{type(impl).__name__} 不满足 {protocol}"


def test_repository_methods_are_async() -> None:
    """⚠️ 仓储方法**全部**是 async，哪怕内存实现根本不需要 await。

    理由：接口一旦定成同步，将来换成真数据库时每个调用点都要改。而
    「内存实现不需要 await」这个事实不该泄漏到接口形状上 ——
    那正是「先用内存、后接数据库」这条路径最容易踩的坑。
    """
    import inspect

    import src.domain.repository as repo_module

    for name in dir(repo_module):
        obj = getattr(repo_module, name)
        if not (inspect.isclass(obj) and getattr(obj, "_is_protocol", False)):
            continue
        for method_name, member in inspect.getmembers(obj, inspect.isfunction):
            if method_name.startswith("_"):
                continue
            assert inspect.iscoroutinefunction(member), f"{name}.{method_name} 不是 async"
