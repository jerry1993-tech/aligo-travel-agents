# -*- coding: utf-8 -*-
"""差旅**工具集**：智能体能调用的全部业务能力的入口。

对外只暴露一个构造函数::

    from src.tools import build_toolkit

    toolkit = build_toolkit(
        user_id="u-123",
        transport_repo=..., hotel_repo=..., policy_repo=...,
        order_repo=..., approval_repo=...,
    )

═══ ⚠️ 本包**必须** import ``agentscope`` ═══

与 :mod:`src.domain` / :mod:`src.orchestration.classifier` /
:mod:`src.chains.collector` 那几处刻意的「纯度」相反：工具层是**框架能力的
使用方**，``FunctionTool`` / ``Toolkit`` / ``ToolChunk`` 都是框架类型，
没有它们就没有工具。这里不追求可脱离框架测试 —— 追求的是**薄**：
每个工具函数体只做「参数解析 → 调仓储 → 组装结果」，业务判断全部下沉到
:mod:`src.domain`。

所以本包的测试策略是「仓储打桩 + 直接调用工具函数」，而不是绕开框架。

═══ 工具分组：**全部放 basic** ═══

框架支持把工具分组，非 ``basic`` 的组需要**显式激活**才可见，
而 ``basic`` 组**始终**激活（``agentscope/tool/_toolkit.py:183-185``：
"The ``basic`` group will always be included regardless of the filter"；
强制点在 ``:254-277`` —— 未激活的组，其工具会被包成一个
``ToolGroupInactiveError`` 的 ``ToolChunk`` 返回）。
这里**没有**分组，全部落在 ``basic``。

理由是权衡后的选择：分组省的是每次请求的 token（工具 schema 要进 prompt），
代价是模型**必须**先调用一个「激活某组」的动作才能看见那些工具。对本项目
的工具数量（6 个）来说，省下的 token 有限，而多一步激活会让模型在
「用户想查酒店」时先激活分组、再调用工具 —— 多一次往返，还多一个可能
出错的环节。

⚠️ 何时该重新考虑：工具数超过约 20 个，或某个工具组的 schema 特别大
（比如带枚举参数的检索工具）。那时分组省下的 token 会超过它的复杂度成本。
"""

from __future__ import annotations

from collections.abc import Iterable

from agentscope.tool import FunctionTool, ToolBase, Toolkit

from src.domain.repository import (
    ApprovalRepository,
    HotelRepository,
    OrderRepository,
    PolicyRepository,
    TransportRepository,
)
from src.tools._result import (
    CARD_APPROVAL,
    CARD_HOTEL,
    CARD_KEY,
    CARD_ORDERS,
    CARD_POLICY,
    CARD_ROUTE,
    CARD_TRANSPORT,
    error_chunk,
    needs_input_chunk,
    ok_chunk,
)
from src.tools.expert import INTENT_TOOL_NAME, build_intent_tool
from src.tools.orders import build_order_tools
from src.tools.route import ROUTE_TOOL_NAME, aligo_route_intent
from src.tools.travel import build_travel_tools

__all__ = [
    "CARD_APPROVAL",
    "CARD_HOTEL",
    "CARD_KEY",
    "CARD_ORDERS",
    "CARD_POLICY",
    "CARD_ROUTE",
    "CARD_TRANSPORT",
    "INTENT_TOOL_NAME",
    "ROUTE_TOOL_NAME",
    "aligo_route_intent",
    "build_business_tools",
    "build_intent_tool",
    "build_toolkit",
    "error_chunk",
    "needs_input_chunk",
    "ok_chunk",
]


def build_business_tools(
    *,
    user_id: str,
    transport_repo: TransportRepository,
    hotel_repo: HotelRepository,
    policy_repo: PolicyRepository,
    order_repo: OrderRepository,
    approval_repo: ApprovalRepository,
    extra: Iterable[ToolBase] = (),
) -> list[ToolBase]:
    """构造当前用户可用的**工具列表**（不含 ``Toolkit`` 外壳）。

    ⚠️ 这个函数与 :func:`build_toolkit` 的分工必须分清楚：

        ``build_business_tools``  决定「有哪些工具」——**唯一**的真源
        ``build_toolkit``         把这份列表装进框架的 ``Toolkit``

    框架的服务层（``agentscope.app``）需要的是一个 ``list[ToolBase]``
    —— 注意 ``AgentToolFactory`` 的**返回类型**是
    ``Awaitable[list[ToolBase]]``（``agentscope/app/_types.py:33-36``），
    框架写的是 ``tools += await factory(...)``，也就是它要的是
    **被 await 之后**的那个列表。它不要 ``Toolkit`` ——
    ``Toolkit`` 由框架在自己的装配流程里构造。所以两条路径都要有入口。

    ⚠️ 但它们**绝不能各写一份工具清单**。两份清单的第一个分歧点
    （比如这边加了新工具、那边忘了）不会报错，只会让「离线脚本能用的工具」
    与「线上对话能用的工具」不一样 —— 而这类差异往往是在排查一个
    更深的问题时才发现，那时它已经误导了排查方向。

    Args:
        user_id (`str`): 当前用户标识。
            ⚠️ **必须**由服务端从鉴权结果传入，绝不能来自模型或请求体 ——
            见 :func:`src.tools.orders.build_order_tools` 的说明。
        transport_repo (`TransportRepository`): 交通仓储。
        hotel_repo (`HotelRepository`): 酒店仓储。
        policy_repo (`PolicyRepository`): 差标仓储。
        order_repo (`OrderRepository`): 订单仓储。
        approval_repo (`ApprovalRepository`): 申请单仓储。
        extra (`Iterable[ToolBase]`): 额外追加的工具（服务层装配时传意图
            识别工具进来）。默认空。

    Returns:
        `list[ToolBase]`: 工具列表。

    ⚠️ 路由工具 ``aligo_route_intent`` 也在这里注册。它通常由
    ``LaneRouterMiddleware`` 自动调用，但**必须**出现在工具表里 ——
    否则框架找不到这个名字，会把合成调用变成一个
    ``ToolNotFoundError`` 的结果交给模型（实测：以普通工具结果的形式出现，
    不抛异常），于是快车道静默失效，而日志里只有一条不起眼的工具失败。

    组装顺序上它排在第一位，纯属可读性考虑（让「路由」这个概念在工具表里
    最先出现）；``Toolkit`` 不保证也不承诺保留顺序。
    """
    return [
        # 路由工具：无副作用、无 I/O，纯函数，所以是只读的。
        FunctionTool(aligo_route_intent, is_read_only=True),
        *build_travel_tools(
            transport_repo=transport_repo,
            hotel_repo=hotel_repo,
            policy_repo=policy_repo,
            user_id=user_id,
        ),
        *build_order_tools(
            order_repo=order_repo,
            approval_repo=approval_repo,
            user_id=user_id,
        ),
        *extra,
    ]


def build_toolkit(
    *,
    user_id: str,
    transport_repo: TransportRepository,
    hotel_repo: HotelRepository,
    policy_repo: PolicyRepository,
    order_repo: OrderRepository,
    approval_repo: ApprovalRepository,
) -> Toolkit:
    """构造当前用户可用的完整工具集。

    ⚠️ 本函数是 :func:`build_business_tools` 的**薄外壳**，不自己拼清单 ——
    理由见那个函数的说明。

    Args:
        user_id (`str`): 当前用户标识。
        transport_repo (`TransportRepository`): 交通仓储。
        hotel_repo (`HotelRepository`): 酒店仓储。
        policy_repo (`PolicyRepository`): 差标仓储。
        order_repo (`OrderRepository`): 订单仓储。
        approval_repo (`ApprovalRepository`): 申请单仓储。

    Returns:
        `Toolkit`: 已装配好的工具集。
    """
    return Toolkit(
        tools=build_business_tools(
            user_id=user_id,
            transport_repo=transport_repo,
            hotel_repo=hotel_repo,
            policy_repo=policy_repo,
            order_repo=order_repo,
            approval_repo=approval_repo,
        ),
    )
