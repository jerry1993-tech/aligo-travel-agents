# -*- coding: utf-8 -*-
"""差旅**仓储协议**：工具层访问业务数据的唯一入口。

文件职责：
    用 ``typing.Protocol`` 描述「业务数据能怎么被读写」。工具层只依赖这些
    协议，不依赖任何具体存储。

上下游依赖：
    - 上游：:mod:`src.domain.entities`、:mod:`src.domain.rules`。
    - 下游：``src/storage/memory.py``（P3 用的内存实现）、
      ``src/storage/postgres.py``（P4 的落库实现）、``src/tools/``。

═══ 为什么用 Protocol 而不是 ABC ═══

``Protocol`` 是**结构化**的：实现方不需要 import 本模块、不需要继承任何
基类，只要方法名和签名对得上就算数。这带来两个具体好处：

1. **P3 的内存实现与 P4 的 Postgres 实现可以毫无关系**。它们各自写各自的
   类，测试里也不用为了造个假仓储去继承一堆东西。
2. **测试替身极简**。单测里要一个「永远返回空列表的订单仓储」时，写一个
   只有 ``list_for`` 的小类就够了，不必实现协议里的全部方法 ——
   静态检查器只会对**实际用到**的方法报缺。

═══ ⚠️ 全部方法都是 async，即使内存实现根本不 await ═══

这是刻意的。P4 的 Postgres 实现必然是异步的（``create_async_engine``），
若协议定成同步，到时候要么改协议（所有调用点跟着改），要么在异步代码里
``await`` 一个同步函数（阻塞事件循环）。现在多写一个 ``async`` 的成本，
远小于将来把同步实现逐个改成异步的成本。

内存实现里会出现「``async def`` 但函数体没有一个 ``await``」的方法 ——
那不是疏忽，是为了满足协议。linter 若提示，属误报。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.domain.entities import ApprovalRequest, HotelOption, TransportOption, TravelOrder
from src.domain.enums import OrderStatus, TransportMode
from src.domain.rules import PolicyLimit

# ==============================================================================
# 一、搜索类（只读）
# ==============================================================================


@runtime_checkable
class TransportRepository(Protocol):
    """交通工具查询。"""

    async def search(
        self,
        *,
        origin: str,
        destination: str,
        depart_date: str,
        mode: TransportMode = TransportMode.ANY,
    ) -> list[TransportOption]:
        """查询两城之间的交通选项。

        Args:
            origin (`str`): 出发城市。
            destination (`str`): 目的城市。
            depart_date (`str`): 出发日期，``YYYY-MM-DD``。
            mode (`TransportMode`): 交通方式；``ANY`` 表示不限。

        Returns:
            `list[TransportOption]`: 按价格升序排列的选项；查不到返回空列表。

        ⚠️ 契约规定「查不到返回**空列表**」，不是抛异常、也不是 ``None``。
        空列表与「出错了」在调用方看来必须可区分：空列表是**业务结论**
        （这几条线路今天没有票），要如实告诉用户；异常是**系统故障**，
        要让用户重试。合并成一种，工具层就没法给出正确的措辞。
        """
        ...


@runtime_checkable
class HotelRepository(Protocol):
    """酒店查询。"""

    async def search(
        self,
        *,
        city: str,
        area: str = "",
        check_in: str = "",
        check_out: str = "",
    ) -> list[HotelOption]:
        """查询酒店。

        Args:
            city (`str`): 城市。
            area (`str`): 区域偏好；空串表示不限。
            check_in (`str`): 入住日期，``YYYY-MM-DD``。
            check_out (`str`): 离店日期，``YYYY-MM-DD``。

        Returns:
            `list[HotelOption]`: 按价格升序排列；查不到返回空列表。
        """
        ...


@runtime_checkable
class PolicyRepository(Protocol):
    """差旅标准查询。"""

    async def limit_for(self, *, user_id: str) -> PolicyLimit:
        """取该用户适用的差标。

        Args:
            user_id (`str`): 用户标识。

        Returns:
            `PolicyLimit`: 差标；该用户没有专门差标时返回**默认差标**。

        ⚠️ 没有专门差标时返回默认值，而不是 ``None`` 或抛异常。理由：
        「这个用户没有差标记录」在业务上等价于「按公司默认标准执行」，
        不是错误状态。返回 ``None`` 会逼每个调用方写一遍
        ``if limit is None: limit = DEFAULT``，漏写一处就是一次空指针。
        """
        ...


# ==============================================================================
# 二、订单与申请单（读写）
# ==============================================================================


@runtime_checkable
class OrderRepository(Protocol):
    """订单读写。"""

    async def list_for(self, *, user_id: str, kind: str = "") -> list[TravelOrder]:
        """列出该用户的订单。

        Args:
            user_id (`str`): 用户标识。
            kind (`str`): 订单类型过滤；空串表示全部。

        Returns:
            `list[TravelOrder]`: 按创建时间**倒序**（最新的在前）。

        ⚠️ 排序是契约的一部分，不是实现细节。用户问「我的订单」时想看的是
        最近那笔；若顺序由实现决定，换个存储后端就会变，而前端的渲染逻辑
        会莫名其妙地跟着变。
        """
        ...

    async def get(self, *, order_id: str) -> TravelOrder | None:
        """按订单号取订单。

        Args:
            order_id (`str`): 订单号。

        Returns:
            `TravelOrder | None`: 订单；不存在时返回 ``None``。

        ⚠️ 这里的 ``None`` 与上面的「空列表」是同一类约定，但方向相反：
        单个查询的「不存在」用 ``None`` 表达，因为它确实可能不存在；
        而列表查询的「不存在」用空列表表达，因为它**总有**一个值。
        """
        ...

    async def save(self, order: TravelOrder) -> TravelOrder:
        """写入或更新订单。

        Args:
            order (`TravelOrder`): 要保存的订单（不可变，整条替换）。

        Returns:
            `TravelOrder`: 保存后的订单（实现可能补全 ``updated_at``）。

        ⚠️ 方法名是 ``save`` 而不是 ``update``：它既创建也更新。分成
        ``create`` / ``update`` 会让每个调用点都要先判断「这条存在吗」——
        而那个判断本身就是一次竞态（查的时候不存在，写的时候已被别人创建）。
        存储层做 upsert 才是原子的。
        """
        ...


@runtime_checkable
class ApprovalRepository(Protocol):
    """出差申请单读写。"""

    async def list_for(
        self,
        *,
        user_id: str,
        status: OrderStatus | None = None,
    ) -> list[ApprovalRequest]:
        """列出该用户的申请单。

        Args:
            user_id (`str`): 申请人。
            status (`OrderStatus | None`): 状态过滤；``None`` 表示全部。

        Returns:
            `list[ApprovalRequest]`: 按创建时间倒序。
        """
        ...

    async def get(self, *, request_id: str) -> ApprovalRequest | None:
        """按申请单号取申请单。

        Args:
            request_id (`str`): 申请单号。

        Returns:
            `ApprovalRequest | None`: 申请单；不存在时为 ``None``。
        """
        ...

    async def save(self, request: ApprovalRequest) -> ApprovalRequest:
        """写入或更新申请单。

        Args:
            request (`ApprovalRequest`): 要保存的申请单。

        Returns:
            `ApprovalRequest`: 保存后的申请单。
        """
        ...


__all__ = [
    "ApprovalRepository",
    "HotelRepository",
    "OrderRepository",
    "PolicyRepository",
    "TransportRepository",
]
