# -*- coding: utf-8 -*-
"""差旅**查询类工具**：交通、酒店、差标。

文件职责：
    把 :mod:`src.domain.repository` 的查询能力包装成 AgentScope 的
    ``FunctionTool``，并统一返回值格式（见 :mod:`src.tools._result`）。

上下游依赖：
    - 上游：:mod:`src.domain.repository`、:mod:`src.domain.rules`、
      :mod:`src.tools._result`。
    - 下游：``src/tools/__init__.py`` 的 ``build_toolkit``。

═══ 全部是只读工具，全部走 ``is_read_only=True`` ═══

这一条不是可选的优化，而是**必须**的设置。已核实（``tool/_adapters.py:116-135``）：
``FunctionTool`` 的 ``permission`` 默认是 ``None``，而 ``None`` 会被解释成
``PermissionDecision(behavior=ASK)`` —— 于是**每一个**自定义工具调用都会
弹一次用户确认。

对查询类工具来说那是灾难性的：用户问「查下我的订单」，系统弹窗问
「是否允许查询订单？」。而 ``is_read_only=True`` 会命中权限引擎的只读
快速通道，**在工具自身的权限判定之前**直接放行。

⚠️ 快速通道的准确位置（已逐行核实，先前这里引用错了行号）：
``permission/_engine.py:659-687`` 是共享的 ``_check_read_only_fast_path``，
``_check_default`` 在 ``:170`` 调用它（步骤 3），位于「工具自身的
``check_permissions``」**之前**。该函数的 docstring 明确写着
「auto-allowed in **every** PermissionMode」—— 所以这条快速通道在默认模式下
同样生效，不需要把权限模式调成 ``ACCEPT_EDITS``。

⚠️ 别被 ``tool.check_permissions()`` 骗了：对只读工具直接调它，返回的**仍然**
是 ``behavior=ASK``（"Custom function tools must be explicitly allowed by
the user."）。快速通道发生在**引擎**这一层、在调用工具的 ``check_permissions``
之前，所以「工具的 check_permissions 返回 ASK」不代表它会被拦下来。
只看那个方法的返回值会得出「只读根本没用」的错误结论。

⚠️ 反过来说，**有副作用的工具绝不能设 ``is_read_only=True``**。
``src/tools/orders.py`` 的提交类工具刻意不设，靠默认的 ASK 拿到人工确认 ——
那是 HITL，是特性不是缺陷。
"""

from __future__ import annotations

import math
from typing import Any

from agentscope.tool import FunctionTool, ToolBase, ToolChunk

from src.domain.enums import CabinClass, TransportMode
from src.domain.repository import HotelRepository, PolicyRepository, TransportRepository
from src.domain.rules import check_cabin, check_flight_price, check_hotel_price
# ⚠️ 价格一律走 :func:`render_amount`，不用 ``f"{value:g}"``（缺陷 P2）：``:g``
# 只保留 6 位有效数字，1200000 会渲染成 ``1.2e+06`` 写进工具返回 —— 模型
# 「原样引用」时就会把用户读不懂、提示词也禁止的科学计数法发给用户。
# 该函数与动态 Prompt、回复守卫共用同一份数字规则（见
# :mod:`src.orchestration.amounts` 的模块说明）。
from src.orchestration.amounts import render_amount
from src.tools._result import (
    CARD_HOTEL,
    CARD_POLICY,
    CARD_TRANSPORT,
    error_chunk,
    needs_input_chunk,
    ok_chunk,
)


#: 舱位枚举 → **给模型看的中文名**。现在就是枚举自己的 ``display_name``。
#:
#: ⚠️ 不能用 ``limit.max_cabin.value`` —— 那是 ``"ECONOMY"`` 这样的内部标识符。
#: 把它写进工具返回值，模型很可能原样复述给用户（「最高舱位 ECONOMY」），
#: 而用户不认识这个词。
#:
#: ⚠️ 本表原先在这里重复维护了一份，2026-10-03 合并回
#: :attr:`src.domain.enums.CabinClass.display_name`。理由是**同一份中文名
#: 当时有三个副本**：本表、:func:`src.domain.rules.check_cabin` 的
#: ``reasons`` 文案（它那时直接印 ``.value``，于是用户看到
#: 「BUSINESS 超出差标允许的最高舱位 ECONOMY」），以及
#: ``src/orchestration/prompt.py`` 的 ``_CABIN_LABELS``。前两处的受众
#: 完全相同（都是要展示给用户的正文），再分两份只会有一次改一处漏一处。
#: 第三处是 **prompt 用词**，受众是模型，仍然保留独立。
#: ``tests/test_tools_travel.py`` 有一条遍历断言守着它覆盖全部成员。
_CABIN_TEXT: dict[CabinClass, str] = {
    cabin: cabin.display_name for cabin in CabinClass
}


def _limit_text(value: float, unit: str) -> str:
    """把一条差标上限渲染成给模型看的话。

    ⚠️ 必须处理 ``<= 0`` 这个哨兵值。:class:`~src.domain.rules.PolicyLimit`
    的字段说明写着「``0`` 表示**不限**」（``rules.py:163-164``），
    所以直接渲染数字会把「不限」写成「上限 0 元」—— 那是一个**恰好相反**
    的意思：用户会以为一分钱都不能报。

    ⚠️ 数值走 :func:`render_amount`（缺陷 P2）：早期这里是 ``f"{value:g}"``，
    上限一过 1e6 就会渲染成 ``不超过 1.2e+06 元/晚``，与提示词「原样引用、
    不要凑整」的纪律直接冲突。

    Args:
        value (`float`): 上限数值。
        unit (`str`): 单位后缀（如 ``元/晚``）。

    Returns:
        `str`: 形如 ``「不超过 600 元/晚」`` 或 ``「不限」``。
    """
    if value <= 0:
        return "不限"
    return f"不超过 {render_amount(value)} {unit}"


def _coerce_price(raw: Any) -> float:
    """把模型可能送来的各种「价格」形状收敛成一个浮点数。

    ⚠️ 这个函数存在的原因是**框架不做参数校验，也不做类型转换**
    （``tool/_adapters.py`` 把 ``**kwargs`` 原样传进来）。于是签名上写着
    ``price: float = 0.0``，实际收到的可能是模型用 JSON 里的各种写法：
    ``null``（最常见 —— 模型想表达「这一项留空」）、``"800"``（字符串数字）、
    ``"800 元"``、``NaN``……

    收敛前的实测后果（2026-10-03，逐个用真对象复现）：

        price=None   → ``TypeError: '<=' not supported between 'NoneType' and 'int'``
        price="800"  → 同上，``TypeError``
        price=nan    → ``nan <= 0`` 为 False，溜进核对分支，
                       返回「酒店 nan 元/晚 符合差旅标准」

    这些异常会被框架吞掉、转成一段**英文**错误文本交给模型 —— 模型看不懂
    发生了什么，只会换个说法重试。而它们全部发生在「查标准」这条主干上：
    用户问「住宿标准是多少」，模型递一个 null 表示留空，本该走
    :func:`_limit_text` 那条路，结果什么标准都拿不到。

    判据：**读不出数字就当作「没给价格」**（返回 ``0.0``），也就是走查标准
    那一支。理由是一致性 —— 那条分支的返回话术本来就写着「有具体价格要核对时，
    把价格告诉我」，它会自己把模型引回正轨；而任何「猜一个数」的兜底
    （比如从 ``"800 元"`` 里抠数字）都可能猜出一个用户没说的价格，
    再拿它去下合规结论 —— 那正是本模块最不想要的输出。

    Args:
        raw (`Any`): 模型传来的原始值。

    Returns:
        `float`: 读得出的价格；读不出（``None`` / 非数字字符串 / ``NaN`` /
        ``±inf`` / 其他类型）时返回 ``0.0``，语义是「没给价格」。
    """
    if raw is None or isinstance(raw, bool):
        # bool 是 int 的子类，``True`` 会被 float() 悄悄变成 1.0 ——
        # 一个「是/否」被当成一元钱，这种错误没有任何人会发现。
        return 0.0
    if isinstance(raw, (int, float)):
        value = float(raw)
    elif isinstance(raw, str):
        # 只去掉**包装**，不做提取：数字外的字符全在，说明这串的本意不是数字
        # （「大概八百」这类），按读不出处理，绝不去里面抠数字。
        text = raw.strip()
        for junk in (",", "，", " ", "元", "¥", "￥", "/", "晚"):
            text = text.replace(junk, "")
        try:
            value = float(text)
        except ValueError:
            return 0.0
    else:
        return 0.0
    if not math.isfinite(value):
        # NaN 会让 ``<= 0`` 为 False、``inf`` 会让一切都合规 —— 两者都会
        # 溜进核对分支并渲染出「nan 元/晚 符合差旅标准」这种正文。
        return 0.0
    return value


def _parse_mode(raw: str) -> TransportMode:
    """把模型给的交通方式字符串解析成枚举。

    ⚠️ 解析失败**返回 ``ANY`` 而不是报错**。模型把 ``FLIGHT`` 写成「飞机」
    或 ``flight`` 是常事，为此中断一次查询不值得 —— 而 ``ANY`` 的语义正是
    「不限」，退化成「两种都查」对用户无害。

    Args:
        raw (`str`): 模型给的值。

    Returns:
        `TransportMode`: 解析结果；无法识别时为 ``ANY``。
    """
    text = (raw or "").strip()
    if not text:
        return TransportMode.ANY
    # 先按枚举值精确匹配，再按中文别名兜底。
    try:
        return TransportMode(text.upper())
    except ValueError:
        pass
    aliases = {
        "飞机": TransportMode.FLIGHT,
        "航班": TransportMode.FLIGHT,
        "机票": TransportMode.FLIGHT,
        "火车": TransportMode.TRAIN,
        "高铁": TransportMode.TRAIN,
        "动车": TransportMode.TRAIN,
        "列车": TransportMode.TRAIN,
        "不限": TransportMode.ANY,
        "都可以": TransportMode.ANY,
    }
    return aliases.get(text, TransportMode.ANY)


def _parse_cabin(raw: str) -> CabinClass:
    """把模型给的舱位字符串解析成枚举。

    ⚠️ 与 :func:`_parse_mode` 同理，失败返回 ``ANY``（「未指定」）。

    Args:
        raw (`str`): 模型给的值。

    Returns:
        `CabinClass`: 解析结果。
    """
    text = (raw or "").strip()
    if not text:
        return CabinClass.ANY
    try:
        return CabinClass(text.upper())
    except ValueError:
        pass
    aliases = {
        "经济舱": CabinClass.ECONOMY,
        "经济": CabinClass.ECONOMY,
        "超级经济舱": CabinClass.PREMIUM_ECONOMY,
        "商务舱": CabinClass.BUSINESS,
        "公务舱": CabinClass.BUSINESS,
        "商务": CabinClass.BUSINESS,
        "头等舱": CabinClass.FIRST,
        "头等": CabinClass.FIRST,
    }
    return aliases.get(text, CabinClass.ANY)


def build_travel_tools(
    *,
    transport_repo: TransportRepository,
    hotel_repo: HotelRepository,
    policy_repo: PolicyRepository,
    user_id: str,
) -> list[ToolBase]:
    """构造查询类工具集。

    ⚠️ 用**工厂函数 + 闭包**而不是模块级函数：工具需要 ``user_id``（多租户
    隔离）与仓储实例，而 ``FunctionTool`` 包装的是普通函数，没有地方注入
    这些依赖。闭包让每个会话拿到自己的一套工具实例 —— 若写成模块级函数
    再去全局变量里找仓储，多租户隔离就只剩一层运气。

    Args:
        transport_repo (`TransportRepository`): 交通仓储。
        hotel_repo (`HotelRepository`): 酒店仓储。
        policy_repo (`PolicyRepository`): 差标仓储。
        user_id (`str`): 当前用户标识。

    Returns:
        `list[ToolBase]`: 可直接放进 ``Toolkit(tools=...)`` 的工具列表。
    """

    async def search_transport(
        origin: str,
        destination: str,
        depart_date: str,
        mode: str = "ANY",
    ) -> ToolChunk:
        """查询两地之间的交通选项（航班与火车）。

        当你已经知道出发地、目的地和日期，需要给用户具体班次建议时使用本工具。

        Args:
            origin (str): 出发城市，例如「杭州」。
            destination (str): 目的城市，例如「北京」。
            depart_date (str): 出发日期，格式 YYYY-MM-DD。
            mode (str): 交通方式，可选 FLIGHT（飞机）、TRAIN（火车）、
                ANY（不限，默认）。

        Returns:
            ToolChunk: 按价格升序的选项列表；缺要素时返回需补充提示。
        """
        missing: list[str] = []
        if not (origin or "").strip():
            missing.append("出发城市")
        if not (destination or "").strip():
            missing.append("目的城市")
        if not (depart_date or "").strip():
            missing.append("出发日期")
        if missing:
            return needs_input_chunk(
                f"还需要你提供：{'、'.join(missing)}，我才能查询交通选项。",
                missing=missing,
            )

        try:
            options = await transport_repo.search(
                origin=origin.strip(),
                destination=destination.strip(),
                depart_date=depart_date.strip(),
                mode=_parse_mode(mode),
            )
        except Exception as exc:  # noqa: BLE001 —— 见下
            # ⚠️ 这里刻意宽捕。不捕的话异常会被框架吞成一段英文错误文本
            # 交给模型（见 src/tools/_result.py 的说明），用户看到的是
            # "ConnectionError: ..."。捕了至少能给出中文说明。
            # 但 detail 里保留原文，别把排查线索也一起丢掉。
            return error_chunk(
                "交通查询服务暂时不可用，请稍后重试。",
                detail=f"{type(exc).__name__}: {exc}",
            )

        if not options:
            # ⚠️ 空结果是**业务结论**，不是错误。措辞要说清「这条线路没有」，
            # 而不是「查询失败」——后者会让用户反复重试一个正常的结论。
            return ok_chunk(
                f"没有查到 {origin} 到 {destination} 在 {depart_date} 的可用班次，"
                f"可以换个日期或改乘其它交通方式试试。",
                card=CARD_TRANSPORT,
                items=[],
            )

        items = [
            {
                "option_id": o.option_id,
                "mode": o.mode.value,
                "mode_display": "飞机" if o.mode is TransportMode.FLIGHT else "火车",
                "carrier": o.carrier,
                "origin": o.origin,
                "destination": o.destination,
                "depart_at": o.depart_at,
                "arrive_at": o.arrive_at,
                "duration_minutes": o.duration_minutes,
                "price": o.price,
                "cabin": o.cabin.value,
                "seats_left": o.seats_left,
                "sold_out": o.is_sold_out,
            }
            for o in options
        ]
        cheapest = options[0]
        return ok_chunk(
            f"找到 {len(options)} 个从 {origin} 到 {destination} 的选项，"
            f"最低 {render_amount(cheapest.price)} 元（{cheapest.carrier}，{cheapest.depart_at}）。",
            card=CARD_TRANSPORT,
            items=items,
            origin=origin.strip(),
            destination=destination.strip(),
            depart_date=depart_date.strip(),
        )

    async def search_hotels(
        city: str,
        area: str = "",
        check_in: str = "",
        check_out: str = "",
    ) -> ToolChunk:
        """查询某城市的酒店，可限定商圈。

        当用户需要住宿建议，或需要核对酒店是否超出差标时使用本工具。

        Args:
            city (str): 城市，例如「北京」。
            area (str): 商圈或区域偏好，例如「国贸」。留空表示不限。
            check_in (str): 入住日期，格式 YYYY-MM-DD。可选。
            check_out (str): 离店日期，格式 YYYY-MM-DD。可选。

        Returns:
            ToolChunk: 按价格升序的酒店列表。
        """
        if not (city or "").strip():
            return needs_input_chunk("还需要你提供入住城市，我才能查询酒店。", missing=["入住城市"])

        try:
            hotels = await hotel_repo.search(
                city=city.strip(),
                area=(area or "").strip(),
                check_in=(check_in or "").strip(),
                check_out=(check_out or "").strip(),
            )
        except Exception as exc:  # noqa: BLE001 —— 同 search_transport
            return error_chunk(
                "酒店查询服务暂时不可用，请稍后重试。",
                detail=f"{type(exc).__name__}: {exc}",
            )

        if not hotels:
            # ⚠️ 指定了商圈却查不到时，**明确说出商圈名**。用户听到
            # 「没有找到酒店」会以为整座城市都没有；说清是「国贸附近没有」
            # 才能引导他放宽条件。
            scope = f"{area.strip()}附近的" if (area or "").strip() else ""
            return ok_chunk(
                f"没有查到 {city}{scope}酒店，可以换个区域或放宽条件再试。",
                card=CARD_HOTEL,
                items=[],
            )

        items = [
            {
                "option_id": h.option_id,
                "name": h.name,
                "city": h.city,
                "area": h.area,
                "price_per_night": h.price_per_night,
                "star": h.star,
                "distance_km": h.distance_km,
            }
            for h in hotels
        ]
        cheapest = hotels[0]
        return ok_chunk(
            f"找到 {len(hotels)} 家{city}的酒店，最低 {render_amount(cheapest.price_per_night)} 元/晚"
            f"（{cheapest.name}，{cheapest.area}）。",
            card=CARD_HOTEL,
            items=items,
            city=city.strip(),
        )

    async def check_travel_policy(
        kind: str,
        price: float = 0.0,
        cabin: str = "",
    ) -> ToolChunk:
        """查询差旅标准，或核对某项选择是否符合差旅标准。

        两种用法，按**用户问的是哪一个**选：

        - 用户问「标准是多少」「上限多少」「能报多少」——
          那就**不要传 price**，本工具会返回标准本身（上限、舱位要求、依据）。
        - 用户给了具体价格（「这间 800 的酒店能报吗」）——
          传上 price 再问，本工具会返回能不能报。

        ⚠️ **必须**在给出「能不能报」的结论前调用本工具。差标是确定性规则，
        不要凭常识推断，也不要编造标准数值。

        ⚠️ 工具的返回就是**唯一**依据。标准里没有的维度（城市分档、职级差异）
        不要替它补 —— 返回文本会明说有没有分档，照它说。

        Args:
            kind (str): 检查类型，可选 hotel（酒店）或 flight（机票）。
            price (float): 单价（元）。酒店填单晚价格；
                **只想查标准时留空**，不要传 0 以外的占位值。
                留空的三种写法（省略、``null``、``0``）都会被识别成
                「查标准」—— 见 :func:`_coerce_price`。
            cabin (str): 舱位，仅在 kind=flight 时有意义。

        Returns:
            ToolChunk: 查标准时返回标准本身；核对时返回是否合规、原因与改进建议。
        """
        try:
            limit = await policy_repo.limit_for(user_id=user_id)
        except Exception as exc:  # noqa: BLE001 —— 同上
            return error_chunk(
                "差旅标准查询失败，暂时无法核对，请稍后重试。",
                detail=f"{type(exc).__name__}: {exc}",
            )

        # ⚠️ 先收敛价格再做任何判断。框架不校验参数类型，模型完全可能递来
        # ``null`` / ``"800"``（实测两种都会在下一次比较时抛 TypeError，
        # 见 :func:`_coerce_price`）。收敛放在这里而不是各分支里，
        # 是为了让「price 一定是一个 float」成为本函数后续所有代码的前提。
        price = _coerce_price(price)

        normalized = (kind or "").strip().lower()
        # ⚠️ 舱位要先解析出来：「有没有可核对的对象」得靠它一起判断。
        parsed_cabin = _parse_cabin(cabin)
        # ⚠️ 走「只查标准」的判据是**没有可核对的对象**，而不是「价格缺失」。
        # 机票只给了舱位、没给价格时，舱位本身就是一个确定性的核对对象
        # —— ``check_cabin`` 会给出「不符合」与改签建议。早先这里只看
        # ``price <= 0``，于是一句「我订商务舱，符合标准吗」被吞成了查标准：
        # 系统手里握着确定性规则，却只把标准念了一遍，把判定推回给模型。
        # 这与本项目「能用代码判定的绝不交给模型」的原则相悖。
        cabin_only = (
            normalized in ("flight", "机票", "飞机")
            and parsed_cabin is not CabinClass.ANY
        )
        if price <= 0 and not cabin_only:
            # ── 只查标准，不核对单价 ────────────────────────────────────────
            #
            # ⚠️ 这一支是**必须**的，不是锦上添花。没有它的时候，
            # 「住宿标准是多少」会走进下面那条校验分支：模型没有单价可传，
            # price 取默认值 0.0，于是工具返回
            #     「酒店 0 元/晚 符合差旅标准。（依据：默认差标：…不超过 600 元…）」
            # —— 上限 600 只在一句括注里，而正文赫然写着「0 元/晚」。
            #
            # 实测（2026-10-03，同一句话连问 4 轮）模型的三种反应：
            #   · 「住宿上限是 **0 元/晚**——这个数字明显不对」
            #   · 「工具在核对时是以『0 元/晚』这个输入去校验的」
            #   · 「具体金额工具这次没返回数值」然后拒绝回答
            # 也就是说，**工具自己造出了一个不存在的数字，再把模型带偏**。
            #
            # 判据用 ``price <= 0`` 而不是 ``price is None``：保持
            # ``price: float = 0.0`` 的签名不变，工具 schema 里 price 仍是一个
            # 普通 number（改成 ``float | None`` 会让 schema 变成 anyOf，
            # 而模型对 anyOf 的遵循度参差）。语义上也不冲突 ——
            # 一笔 0 元的消费在任何差标下都合规，那种「校验」本来就没有信息量。
            if normalized in ("hotel", "酒店", "住宿"):
                headline = f"酒店差标：{_limit_text(limit.max_hotel_price, '元/晚')}"
            elif normalized in ("flight", "机票", "飞机"):
                headline = (
                    f"机票差标：最高舱位{_CABIN_TEXT[limit.max_cabin]}，"
                    f"单张{_limit_text(limit.max_flight_price, '元')}"
                )
            else:
                return error_chunk(
                    f"无法识别的核对类型「{kind}」，目前只支持 hotel（酒店）与 flight（机票）。",
                    detail=f"unknown kind={kind!r}",
                )

            # ⚠️ 正文里必须**同时**给出上限、依据、以及适用范围的边界。
            # 「不区分城市与职级」这句话不是免责声明，而是本系统的真实状态：
            # 差标按用户配置（``PolicyLimit`` 只有三个字段），域里根本没有
            # 职级概念。不写出来，模型就会按它记忆里的「行业惯例」补一个
            # 城市分档表 —— 那正是 2026-10-03 实测到的另一处编造。
            lookup_items = [
                {
                    "kind": normalized,
                    # ⚠️ ``lookup`` 是**给前端用的判别字段**，不是装饰。
                    # 查标准与核对走的是同一张 ``policy_verdict`` 卡片，而
                    # 核对的负载里有 ``compliant``、查标准没有。前端原先
                    # 只写 ``compliant === true ? 绿 : 红``，于是「住宿标准是
                    # 多少」这种最普通的问法会打出一个红色「不符合」徽标
                    # —— 用户看到的是一句无中生有的指控。
                    # 有了这个字段，卡片才能把「查标准」渲染成中性态。
                    "lookup": True,
                    "max_hotel_price": limit.max_hotel_price,
                    "max_flight_price": limit.max_flight_price,
                    "max_cabin": limit.max_cabin.value,
                    "max_cabin_text": limit.max_cabin.display_name,
                    "policy_note": limit.note,
                },
            ]
            lookup = (
                f"{headline}。依据：{limit.note}"
                "这是本次查到的一条统一标准，没有按城市分档，也没有按职级分档 —— "
                "如果你需要的是某个城市或某个职级的特别额度，这次查询结果里没有。"
                "有具体价格要核对时，把价格告诉我，我再帮你判断超没超。"
            )
            return ok_chunk(lookup, card=CARD_POLICY, items=lookup_items)

        if normalized in ("hotel", "酒店", "住宿"):
            verdict = check_hotel_price(price, limit)
            subject = f"酒店 {render_amount(price)} 元/晚"
        elif normalized in ("flight", "机票", "飞机"):
            # ⚠️ 舱位与票价**两项都要查**，任一不合规就是不合规。
            # 只查一项会漏掉「经济舱但超价」或「价格没超但订了商务舱」
            # 这两种最常见的违规。
            cabin_verdict = check_cabin(parsed_cabin, limit)
            if price > 0:
                price_verdict = check_flight_price(price, limit)
                compliant = cabin_verdict.compliant and price_verdict.compliant
                reasons = list(cabin_verdict.reasons) + list(price_verdict.reasons)
                advice = price_verdict.advice or cabin_verdict.advice
                subject = (
                    f"机票 {render_amount(price)} 元（{parsed_cabin.display_name}）"
                )
            else:
                # 只给了舱位、没给价格（``cabin_only`` 那条路）。
                # ⚠️ 主语里**不能**出现价格：渲染 ``price`` 会得到
                # 「机票 0 元（商务舱）」，把「没提供价格」说成「零元机票」。
                compliant = cabin_verdict.compliant
                reasons = list(cabin_verdict.reasons)
                advice = cabin_verdict.advice
                subject = f"机票（{parsed_cabin.display_name}）"
            from src.domain.rules import PolicyVerdict

            verdict = PolicyVerdict(compliant=compliant, reasons=reasons, advice=advice)
        else:
            return error_chunk(
                f"无法识别的核对类型「{kind}」，目前只支持 hotel（酒店）与 flight（机票）。",
                detail=f"unknown kind={kind!r}",
            )

        items = [
            {
                # ⚠️ 与查标准分支配对：两张负载共用一张卡片，必须有判别字段。
                # 树里的核对分支**一直是**这个形状（含 ``compliant``），
                # 这里显式写 ``lookup: False`` 是为了让「同一张卡片的两种
                # 语义」在负载里能被穷举出来，而不是靠"有没有 compliant"去猜。
                "lookup": False,
                "kind": normalized,
                "compliant": verdict.compliant,
                "reasons": list(verdict.reasons),
                "advice": verdict.advice,
                "policy_note": limit.note,
                "max_hotel_price": limit.max_hotel_price,
                "max_flight_price": limit.max_flight_price,
                "max_cabin": limit.max_cabin.value,
                "max_cabin_text": limit.max_cabin.display_name,
            },
        ]

        if verdict.compliant:
            summary = f"{subject} 符合差旅标准。（依据：{limit.note}）"
        else:
            reasons = "；".join(verdict.reasons) or "超出差旅标准"
            summary = f"{subject} **不符合**差旅标准：{reasons}。"
            if verdict.advice:
                summary += f"建议：{verdict.advice}"

        return ok_chunk(summary, card=CARD_POLICY, items=items)

    return [
        # ⚠️ ``is_read_only=True`` 对这三个都是必须的，见模块文档。
        FunctionTool(search_transport, is_read_only=True),
        FunctionTool(search_hotels, is_read_only=True),
        FunctionTool(check_travel_policy, is_read_only=True),
    ]


def tools_of_kind(tools: list[ToolBase]) -> dict[str, Any]:
    """把工具列表转成 ``{工具名: 工具}``，便于测试与排障。

    ⚠️ 仅供测试与调试使用，**不要**在生产路径上用它按名字查工具 ——
    那等于绕开 ``Toolkit`` 自己维护索引，而 ``Toolkit`` 还要处理分组、
    激活状态等本函数完全不知道的事。

    Args:
        tools (`list[ToolBase]`): 工具列表。

    Returns:
        `dict[str, Any]`: 工具名到工具的映射。
    """
    return {getattr(tool, "name", ""): tool for tool in tools}


__all__ = [
    "build_travel_tools",
    "tools_of_kind",
]
