# -*- coding: utf-8 -*-
"""差旅**订单与申请单工具**：查询、提交、取消。

文件职责：
    把订单/申请单的读写包装成 ``FunctionTool``。

上下游依赖：
    - 上游：:mod:`src.domain.repository`、:mod:`src.domain.rules`
      （状态迁移校验）、:mod:`src.tools._result`。
    - 下游：``src/tools/__init__.py`` 的 ``build_toolkit``。

═══ ⚠️ 读写权限**必须**分开：只读走快速通道，写入走人工确认 ═══

这是本文件与 ``src/tools/travel.py`` 最要紧的差别，也是整个 P3 里最容易
做错的一处。

已核实（``agentscope/tool/_adapters.py:116-135``）：``FunctionTool`` 未显式传
``permission`` 时，权限引擎收到的是 ``PermissionDecision(behavior=ASK)``，
agent 会发一个 ``RequireUserConfirmEvent`` 并**暂停**，等用户点确认。

⚠️ 只读工具的「不确认」**不是**来自工具的 ``check_permissions`` ——
对它直接调用仍然返回 ASK。放行发生在引擎的只读快速通道
（``agentscope/permission/_engine.py:659-687``，``_check_default`` 在 ``:170`` 调用），
在工具的 ``check_permissions`` **之前**。详见 ``src/tools/travel.py`` 的说明。

于是同一份代码有两种看似合理的写法，**结果却完全相反**：

- 图省事把所有工具都设成 ``is_read_only=True``（或统一传
  ``PermissionDecision(ALLOW)``）→ 查询工具用起来很顺，
  但**下单/提交申请也不用确认了**。用户说一句「帮我提交申请」，
  系统直接就把申请单提上去了 —— 这是事故。
- 反过来全部走默认 ASK → 安全，但用户每点一次「查订单」都要确认一次，
  产品没法用。

所以本模块的分类是**按副作用**严格划的：

    只读（查询订单、查申请单） → ``is_read_only=True``
    写入（提交申请、取消订单） → 默认 ASK，交给 HITL

⚠️ 判据是「这个调用**会不会改变业务数据**」，不是「重不重要」。
一条经验：**问自己「如果用户在没看清的情况下被连续调用十次，会怎样」**。
查询十次最多是浪费；提交十次就是十张申请单。
"""

from __future__ import annotations

from agentscope.tool import FunctionTool, ToolBase, ToolChunk

from src.domain.entities import ApprovalRequest
from src.domain.enums import OrderStatus
from src.domain.repository import ApprovalRepository, OrderRepository
from src.domain.rules import check_transition
from src.tools._result import (
    CARD_APPROVAL,
    CARD_ORDERS,
    error_chunk,
    ok_chunk,
)

#: 允许提交的申请单初始状态。
#:
#: ⚠️ 从 ``DRAFT`` 起步而不是直接 ``PENDING_APPROVAL``：提交动作本身是
#: 一次**状态迁移**，要经过 :func:`~src.domain.rules.check_transition` 的
#: 校验。直接构造终态会让状态机在这条路径上被绕过 —— 而状态机的价值
#: 恰恰在于**所有**路径都经过它。
_INITIAL_STATUS = OrderStatus.DRAFT

#: 提交后要迁移到的状态。
_SUBMITTED_STATUS = OrderStatus.PENDING_APPROVAL


def build_order_tools(
    *,
    order_repo: OrderRepository,
    approval_repo: ApprovalRepository,
    user_id: str,
) -> list[ToolBase]:
    """构造订单与申请单工具集。

    Args:
        order_repo (`OrderRepository`): 订单仓储。
        approval_repo (`ApprovalRepository`): 申请单仓储。
        user_id (`str`): 当前用户标识。⚠️ **不从工具入参取用户**——
            让模型传 ``user_id`` 等于把租户隔离交给模型，它完全可能传别人的。

    Returns:
        `list[ToolBase]`: 工具列表。
    """

    async def query_orders(kind: str = "", limit: int = 5) -> ToolChunk:
        """查询当前用户自己的订单与出差申请单。

        Args:
            kind (str): 过滤类型，可选 flight（机票）、train（火车票）、
                hotel（酒店）、approval（出差申请）或 all（全部，默认）。
            limit (int): 最多返回几条，默认 5。⚠️ 不要传很大的值，
                订单列表是给人看的，一次几十条反而找不到重点。

        Returns:
            ToolChunk: 订单或申请单列表。
        """
        # ⚠️ 把 limit 夹到合理区间，而不是直接信任模型给的值。
        # 模型完全可能传 limit=1000，那会把整个订单历史拉出来塞进上下文，
        # 既慢又挤占 token。夹逼比报错好：报错会让用户看到一次失败，
        # 而夹逼只是「少给几条」。
        safe_limit = max(1, min(int(limit or 5), 20))
        normalized = (kind or "all").strip().lower()

        wants_orders = normalized in ("all", "", "flight", "train", "hotel", "机票", "火车票", "酒店")
        wants_approvals = normalized in ("all", "", "approval", "申请", "申请单", "出差申请")

        if not wants_orders and not wants_approvals:
            return error_chunk(
                f"无法识别的查询类型「{kind}」，可选：flight / train / hotel / approval / all。",
                detail=f"unknown kind={kind!r}",
            )

        items: list[dict[str, object]] = []
        summaries: list[str] = []

        try:
            if wants_orders:
                order_kind = normalized if normalized in ("flight", "train", "hotel") else ""
                orders = await order_repo.list_for(user_id=user_id, kind=order_kind)
                for order in orders[:safe_limit]:
                    items.append(
                        {
                            "type": "order",
                            "order_id": order.order_id,
                            "kind": order.kind,
                            "status": order.status.value,
                            "title": order.title,
                            "amount": order.amount,
                            "created_at": order.created_at,
                            "cancellable": order.is_cancellable,
                        },
                    )
                if orders:
                    summaries.append(f"{len(orders)} 笔订单")

            if wants_approvals and len(items) < safe_limit:
                requests = await approval_repo.list_for(user_id=user_id)
                remaining = safe_limit - len(items)
                for request in requests[:remaining]:
                    items.append(
                        {
                            "type": "approval",
                            "request_id": request.request_id,
                            "status": request.status.value,
                            "title": request.title,
                            "amount": request.amount,
                            "destination": request.destination,
                            "depart_date": request.depart_date,
                            "created_at": request.created_at,
                        },
                    )
                if requests:
                    summaries.append(f"{len(requests)} 张出差申请")
        except Exception as exc:  # noqa: BLE001 —— 见 src/tools/_result.py 的说明
            return error_chunk(
                "订单查询服务暂时不可用，请稍后重试。",
                detail=f"{type(exc).__name__}: {exc}",
            )

        if not items:
            # ⚠️ 「没有订单」是业务结论，措辞要肯定。写成「未查询到」
            # 会让人怀疑是不是查询失败了。
            return ok_chunk("你目前没有任何订单或出差申请。", card=CARD_ORDERS, items=[])

        return ok_chunk(
            f"共找到 {'、'.join(summaries)}（最多显示 {safe_limit} 条）。",
            card=CARD_ORDERS,
            items=items,
        )

    async def submit_approval(
        title: str,
        destination: str = "",
        depart_date: str = "",
        days: int = 0,
        amount: float = 0.0,
    ) -> ToolChunk:
        """提交一张出差申请单。

        ⚠️ 这是一个**有副作用的操作**，调用前系统会请用户确认。
        在用户明确表达要提交之前，不要主动调用本工具。

        Args:
            title (str): 申请事由摘要，例如「10 月北京客户拜访」。
            destination (str): 目的地城市。
            depart_date (str): 出发日期，格式 YYYY-MM-DD。
            days (int): 出差天数。
            amount (float): 预估金额（元）。

        Returns:
            ToolChunk: 申请单号与当前状态。
        """
        if not (title or "").strip():
            return error_chunk(
                "申请事由不能为空，请补充一句说明这次出差做什么。",
                detail="empty title",
            )

        request_id = _new_id("AP")
        # ⚠️ 先造 DRAFT 再迁移，而不是直接造 PENDING_APPROVAL。
        # 这样提交这个动作走的是和别处**同一条**状态机校验路径 ——
        # 见 _INITIAL_STATUS 的说明。
        request = ApprovalRequest(
            request_id=request_id,
            user_id=user_id,
            title=title.strip(),
            status=_INITIAL_STATUS,
            amount=float(amount or 0.0),
            destination=(destination or "").strip(),
            depart_date=(depart_date or "").strip(),
            days=int(days or 0),
            created_at=_now(),
            updated_at=_now(),
        )

        transition = check_transition(request.status, _SUBMITTED_STATUS)
        if not transition.allowed:
            # ⚠️ 理论上不可达（DRAFT → PENDING_APPROVAL 是合法的），
            # 但**必须**留着这个分支。它守的是「状态机表被改坏」这种情况：
            # 若哪天有人调整了迁移表，这里会明确报出原因，而不是
            # 悄悄写进去一条状态非法的记录。
            return error_chunk(
                f"无法提交申请：{transition.reason}",
                detail=f"transition {request.status.value} -> {_SUBMITTED_STATUS.value} rejected",
            )

        from dataclasses import replace

        request = replace(request, status=_SUBMITTED_STATUS, updated_at=_now())

        try:
            saved = await approval_repo.save(request)
        except Exception as exc:  # noqa: BLE001
            return error_chunk(
                "申请提交失败，请稍后重试。",
                detail=f"{type(exc).__name__}: {exc}",
            )

        return ok_chunk(
            f"出差申请已提交，单号 {saved.request_id}，当前状态「{saved.status.value}」，"
            f"等待审批。",
            card=CARD_APPROVAL,
            items=[
                {
                    "request_id": saved.request_id,
                    "title": saved.title,
                    "destination": saved.destination,
                    "depart_date": saved.depart_date,
                    "days": saved.days,
                    "amount": saved.amount,
                    "status": saved.status.value,
                },
            ],
        )

    return [
        # 只读：走权限快速通道，不打扰用户。
        FunctionTool(query_orders, is_read_only=True),
        # ⚠️ 写入：**刻意不传 permission 也不设 is_read_only**。
        # 默认的 ASK 会触发 HITL 确认 —— 这正是我们要的行为，见模块文档。
        # 若将来有人为了「让演示更顺畅」把这里改成 ALLOW，
        # 那就是把人工确认这道闸门拆了。
        FunctionTool(submit_approval),
    ]


def _new_id(prefix: str) -> str:
    """生成业务单号。

    ⚠️ 用 ``shortuuid`` 而不是 ``uuid4().hex[:8]``：后者是 16 进制，
    可读性差且碰撞概率在短前缀下不低。``shortuuid`` 用去掉易混字符
    （``0/O``、``1/l``）的字母表，人工念单号时不会听错。

    ⚠️ 单号里**不含用户标识与时间**。含用户标识会泄漏信息（单号可被
    外部看到）；含时间会让单号可预测 —— 两者都是不必要的风险暴露。

    Args:
        prefix (`str`): 前缀，如 ``AP``（申请单）。

    Returns:
        `str`: 形如 ``AP-7fK2mQ`` 的单号。
    """
    import shortuuid

    return f"{prefix}-{shortuuid.uuid()[:6]}"


def _now() -> str:
    """当前时刻的 ISO 字符串。

    ⚠️ 这是**唯一**允许读系统时间的地方，且只用于「记录事实发生的时刻」。
    业务判定（价格、可选性）一律不得依赖时间 —— 那会让结果不可复现。
    见 ``src/storage/memory.py`` 的确定性契约。

    Returns:
        `str`: 形如 ``2026-10-01T20:03:28`` 的时间戳。
    """
    from datetime import datetime

    return datetime.now().isoformat(timespec="seconds")


__all__ = [
    "build_order_tools",
]
