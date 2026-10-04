# -*- coding: utf-8 -*-
"""订单与申请单工具（``src/tools/orders.py``）的测试。

═══ 本文件守的是什么 ═══

**读写权限必须分开。** 这是整个 P3 里最容易做错、后果最严重的一处：

- 图省事把所有工具都设成 ``is_read_only=True`` → 查询很顺，但
  **下单/提交申请也不用确认了**。用户说一句「帮我提交申请」，系统直接就
  把申请单提上去了 —— 这是事故，不是性能优化。
- 反过来全部走默认 ASK → 安全，但用户每点一次「查订单」都要确认一次。

两种写法**都能跑通**，功能测试也都能过（工具照样被调用、结果照样返回）。
唯一的差别是**权限判定**，所以必须有专门的断言守着它。

⚠️ 最要紧的两条：
  - :func:`test_query_orders_is_read_only`
  - :func:`test_submit_approval_is_not_read_only`

它们是一对。只写其中一条，把两条都改成 ``True`` 或都改成 ``False``
都能让它保持绿色。
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

import pytest
from agentscope.message import ToolResultState

from src.domain import OrderStatus
from src.domain.entities import ApprovalRequest, TravelOrder
from src.storage.memory import InMemoryApprovalRepository, InMemoryOrderRepository
from src.tools._result import CARD_APPROVAL, CARD_ORDERS
from src.tools.orders import build_order_tools

USER = "u-test"
OTHER_USER = "u-other"


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------
class BoomOrders:
    """``list_for`` 永远抛异常的订单仓储。"""

    async def list_for(self, **_kwargs: Any) -> list[TravelOrder]:
        raise ConnectionError("上游超时")

    async def save(self, *_args: Any, **_kwargs: Any) -> Any:
        raise ConnectionError("上游超时")


class BoomApprovals:
    """``list_for`` / ``save`` 永远抛异常的申请单仓储。"""

    async def list_for(self, **_kwargs: Any) -> list[ApprovalRequest]:
        raise ConnectionError("上游超时")

    async def save(self, *_args: Any, **_kwargs: Any) -> Any:
        raise ConnectionError("上游超时")


def build(
    *,
    orders: Any = None,
    approvals: Any = None,
    user_id: str = USER,
) -> dict[str, Any]:
    """构造工具并返回 ``{名字: 工具}``。

    Args:
        orders (`Any`): 订单仓储；``None`` 用内存实现。
        approvals (`Any`): 申请单仓储；``None`` 用内存实现。
        user_id (`str`): 绑定的用户标识。

    Returns:
        `dict[str, Any]`: 工具名到工具的映射。
    """
    tools = build_order_tools(
        order_repo=orders or InMemoryOrderRepository(),
        approval_repo=approvals or InMemoryApprovalRepository(),
        user_id=user_id,
    )
    return {tool.name: tool for tool in tools}


def payload_of(chunk: Any) -> dict[str, Any]:
    """把工具返回的 ``ToolChunk`` 解回 JSON 载荷。"""
    return json.loads("".join(getattr(b, "text", "") for b in chunk.content))


def call(tool: Any, /, **kwargs: Any) -> Any:
    """调用一个 ``FunctionTool``。"""
    return asyncio.run(tool(**kwargs))


def seed_order(repo: Any, *, user_id: str = USER, order_id: str = "TR-1", **over: Any) -> TravelOrder:
    """往仓储里放一笔订单。"""
    order = TravelOrder(
        order_id=order_id,
        user_id=user_id,
        kind=over.pop("kind", "flight"),
        status=over.pop("status", OrderStatus.PAID),
        title=over.pop("title", "杭州 → 北京 CA1746"),
        amount=over.pop("amount", 1088.0),
        created_at=over.pop("created_at", "2026-09-30T10:00:00"),
        **over,
    )
    asyncio.run(repo.save(order))
    return order


# ---------------------------------------------------------------------------
# 一、★ HITL 闸门：读写权限必须分开
# ---------------------------------------------------------------------------
def test_query_orders_is_read_only() -> None:
    """★ 查询工具**必须**是只读的。

    只有只读调用会命中引擎的只读快速通道（``permission/_engine.py:659-687``，
    ``_check_default`` 在 ``:170`` 调用），不再弹确认窗。少了这个标记，
    用户每点一次「查订单」都要点一次「允许」。
    """
    assert build()["query_orders"].is_read_only is True


def test_submit_approval_is_not_read_only() -> None:
    """★★ **本文件最重要的一条**：提交工具**绝不能**是只读的。

    只读调用会被引擎**自动放行**，不经任何确认。把提交工具标成只读，
    等于拆掉人工确认这道闸门 —— 用户说一句「帮我提交申请」，系统立刻
    就把申请单提上去了，中间没有任何一步询问。

    ⚠️ 这个改动**不会让任何功能测试变红**：工具照样被调用、结果照样返回、
    界面看起来更「顺畅」。所以必须有这条断言专门守着它。

    ⚠️ 与 :func:`test_query_orders_is_read_only` 是一对，缺一不可：
    只留一条的话，「两个都改成 True」或「都改成 False」都能让它保持绿色。
    """
    assert build()["submit_approval"].is_read_only is False


def test_the_two_tools_disagree_on_read_only() -> None:
    """★ 把上一对断言合并成一条「两者必须相反」的检查。

    前两条各自独立地钉住了取值，这条钉住的是**关系** —— 有人把两个工具
    的标记一起改掉时，前两条会同时红（说明有人动了这一对），而这条会
    指出「它们本该相反」。三层防护看着冗余，但这一处的代价是不对称的：
    漏判一次就是一张没经过确认的申请单。
    """
    tools = build()
    read_only_flags = {name: tool.is_read_only for name, tool in tools.items()}

    assert len(set(read_only_flags.values())) == 2, (
        f"读写工具必须有**不同**的只读标记，实际全是 {read_only_flags}"
    )
    assert read_only_flags["query_orders"] is True
    assert read_only_flags["submit_approval"] is False


def test_no_tool_explicitly_forces_allow_permission() -> None:
    """⚠️ 没有任何工具显式设了 ``permission=ALLOW``。

    只读标记与显式 ALLOW 的差别：只读标记表达的是「这个调用没有副作用」，
    引擎据此放行；而显式 ALLOW 表达的是「不管有没有副作用都放行」，
    它会**绕过**所有安全判定。对提交类工具用后者，等于把闸门焊死成常开。

    ⚠️ 检查的是 ``_permission`` 这个私有字段 —— 它是 ``FunctionTool``
    存权限决策的地方。碰私有字段不理想，但这里没有公开读取口，
    而「确认没人偷偷设成 ALLOW」这件事值得破一次例。
    """
    for name, tool in build().items():
        permission = getattr(tool, "_permission", None)
        assert permission is None, f"{name} 显式设了权限 {permission!r}，不应当由工具自己决定放行"


# ---------------------------------------------------------------------------
# 二、多租户隔离
# ---------------------------------------------------------------------------
def test_user_id_is_bound_at_construction_not_passed_in() -> None:
    """⚠️ ``user_id`` 由闭包绑定，**不是**工具入参。

    让模型传 ``user_id`` 等于把租户隔离交给模型 —— 它完全可能传别人的，
    而它并没有恶意，只是照着上下文猜的。这条断言守住「参数表里没有
    用户标识字段」。
    """
    for name, tool in build().items():
        properties = (tool.input_schema or {}).get("properties", {})
        leaked = [p for p in properties if "user" in p.lower() or "tenant" in p.lower()]
        assert not leaked, f"{name} 的参数里出现了用户标识字段 {leaked}"


def test_orders_are_scoped_to_the_bound_user() -> None:
    """⚠️ 只返回**绑定用户**的订单。

    这条与上一条互为表里：上一条守住「模型无法指定用户」，这条守住
    「仓储确实按用户过滤了」。少了这条，一个恒返回全部订单的实现
    也能让上一条通过。
    """
    repo = InMemoryOrderRepository()
    seed_order(repo, user_id=USER, order_id="TR-MINE")
    seed_order(repo, user_id=OTHER_USER, order_id="TR-THEIRS")

    payload = payload_of(call(build(orders=repo)["query_orders"]))
    ids = {item.get("order_id") for item in payload["items"]}

    assert "TR-MINE" in ids
    assert "TR-THEIRS" not in ids, "查到了别人的订单 —— 租户隔离失效"


def test_approvals_are_scoped_to_the_bound_user() -> None:
    """申请单同理。"""
    repo = InMemoryApprovalRepository()
    asyncio.run(repo.save(ApprovalRequest(
        request_id="AP-MINE", user_id=USER, title="我的申请", status=OrderStatus.PENDING_APPROVAL,
    )))
    asyncio.run(repo.save(ApprovalRequest(
        request_id="AP-THEIRS", user_id=OTHER_USER, title="别人的申请", status=OrderStatus.PENDING_APPROVAL,
    )))

    payload = payload_of(call(build(approvals=repo)["query_orders"], kind="approval"))
    ids = {item.get("request_id") for item in payload["items"]}

    assert "AP-MINE" in ids
    assert "AP-THEIRS" not in ids, "查到了别人的申请单 —— 租户隔离失效"


# ---------------------------------------------------------------------------
# 三、查询订单
# ---------------------------------------------------------------------------
def test_query_orders_returns_a_card_with_the_expected_shape() -> None:
    """正常查询返回 ``order_list`` 卡片，字段齐全。"""
    repo = InMemoryOrderRepository()
    seed_order(repo, order_id="TR-1")

    payload = payload_of(call(build(orders=repo)["query_orders"]))

    assert payload["ok"] is True
    assert payload["card"] == CARD_ORDERS
    item = payload["items"][0]
    for field in ("type", "order_id", "kind", "status", "title", "amount", "created_at", "cancellable"):
        assert field in item, f"订单卡片缺字段 {field}"


def test_no_orders_is_a_conclusion_not_an_error() -> None:
    """⚠️ 「没有订单」是**业务结论**，措辞要肯定，状态是 SUCCESS。

    ⚠️ 措辞在这里是功能的一部分。写成「未查询到订单」会让人怀疑是不是
    查询失败了，于是用户反复重试 —— 而这是一个正常结论，重试多少次
    都一样。摘要用的是「你目前没有任何订单或出差申请。」这种肯定句。
    """
    chunk = call(build()["query_orders"])
    payload = payload_of(chunk)

    assert chunk.state is ToolResultState.SUCCESS
    assert payload["ok"] is True
    assert payload["items"] == []
    assert "未查询到" not in payload["summary"], "措辞会被读成「查询失败」"
    assert "暂时不可用" not in payload["summary"]


def test_kind_filter_narrows_the_result() -> None:
    """``kind`` 过滤器真的起作用。"""
    repo = InMemoryOrderRepository()
    seed_order(repo, order_id="TR-1", kind="flight")
    seed_order(repo, order_id="HT-1", kind="hotel")

    only_hotel = payload_of(call(build(orders=repo)["query_orders"], kind="hotel"))
    ids = {item.get("order_id") for item in only_hotel["items"]}

    assert ids == {"HT-1"}, f"kind=hotel 应当只返回酒店，实际 {ids}"


def test_kind_accepts_chinese_aliases() -> None:
    """⚠️ 模型把 ``flight`` 写成「机票」是常事，不该为此报错。"""
    repo = InMemoryOrderRepository()
    seed_order(repo, order_id="TR-1", kind="flight")

    payload = payload_of(call(build(orders=repo)["query_orders"], kind="机票"))

    assert payload["ok"] is True
    assert {item.get("order_id") for item in payload["items"]} == {"TR-1"}


def test_unknown_kind_returns_a_chinese_error() -> None:
    """认不出的类型给中文错误并列出可选值，不抛异常。"""
    chunk = call(build()["query_orders"], kind="飞船票")
    payload = payload_of(chunk)

    assert chunk.state is ToolResultState.ERROR
    assert "无法识别" in payload["summary"]
    assert "flight" in payload["summary"], "错误信息应当列出可选值"


@pytest.mark.parametrize("limit", [0, -5, 1000, 999999])
def test_limit_is_clamped_to_a_sane_range(limit: int) -> None:
    """⚠️ ``limit`` 被夹到合理区间，而不是直接信任模型给的值。

    模型完全可能传 ``limit=1000``，那会把整个订单历史拉出来塞进上下文，
    既慢又挤占 token。夹逼比报错好：报错会让用户看到一次失败，
    而夹逼只是「少给几条」。
    """
    repo = InMemoryOrderRepository()
    for i in range(25):
        seed_order(repo, order_id=f"TR-{i}", created_at="2026-09-30T10:00:00")

    payload = payload_of(call(build(orders=repo)["query_orders"], limit=limit))

    assert payload["ok"] is True
    assert len(payload["items"]) <= 20, f"limit={limit} 没有被夹住"
    assert len(payload["items"]) >= 1


def test_limit_actually_caps_the_result() -> None:
    """⚠️ ``limit`` 真的限制了条数（而不是收下参数却忽略它）。"""
    repo = InMemoryOrderRepository()
    for i in range(25):
        seed_order(repo, order_id=f"TR-{i}")

    payload = payload_of(call(build(orders=repo)["query_orders"], limit=3))

    assert len(payload["items"]) == 3


def test_query_orders_survives_a_broken_repository() -> None:
    """⚠️ 仓储异常时给中文说明，而不是把英文异常交给用户。

    框架会把工具抛出的异常吞成 ``ToolChunk(state=ERROR, text=str(e))``，
    用户看到的是 ``ConnectionError: 上游超时``。
    """
    chunk = call(build(orders=BoomOrders())["query_orders"])
    payload = payload_of(chunk)

    assert chunk.state is ToolResultState.ERROR
    assert "ConnectionError" in payload["detail"], "排查线索不该被丢掉"


# ---------------------------------------------------------------------------
# 四、提交申请
# ---------------------------------------------------------------------------
def test_submit_approval_creates_a_pending_request() -> None:
    """提交成功返回 ``approval_result`` 卡片，状态是待审批。"""
    payload = payload_of(call(build()["submit_approval"], title="10 月北京客户拜访",
                              destination="北京", depart_date="2026-10-08", days=2, amount=3200))

    assert payload["ok"] is True
    assert payload["card"] == CARD_APPROVAL
    item = payload["items"][0]
    assert item["status"] == OrderStatus.PENDING_APPROVAL.value
    assert item["title"] == "10 月北京客户拜访"
    assert item["amount"] == 3200
    assert item["days"] == 2


def test_submit_approval_summary_contains_the_request_id() -> None:
    """⚠️ 摘要里要有**单号** —— 那是用户后续查询与沟通的唯一凭据。

    只说「提交成功」的话，用户问「我的申请到哪一步了」时无据可查。
    """
    payload = payload_of(call(build()["submit_approval"], title="出差"))

    assert payload["items"][0]["request_id"] in payload["summary"]


def test_submit_approval_persists_the_record() -> None:
    """⚠️ 提交之后**真的写进仓储**了。

    这条防的是「返回了一张看起来很正常的卡片，但什么都没存」——
    那种情况下用户会以为提交成功，而系统里根本没有这张申请单。
    """
    repo = InMemoryApprovalRepository()
    payload = payload_of(call(build(approvals=repo)["submit_approval"], title="出差"))
    request_id = payload["items"][0]["request_id"]

    stored = asyncio.run(repo.get(request_id=request_id))

    assert stored is not None, "提交后仓储里查不到这张申请单"
    assert stored.user_id == USER, "申请单没有绑到当前用户"
    assert stored.status is OrderStatus.PENDING_APPROVAL


def test_submit_approval_goes_through_the_state_machine() -> None:
    """⚠️ 提交走的是 ``DRAFT → PENDING_APPROVAL`` 这条**经过校验**的迁移。

    直接构造终态会让状态机在这条路径上被绕过 —— 而状态机的价值恰恰在于
    **所有**路径都经过它。这条断言从外部只能观察到「结果是待审批」，
    但它与 ``orders.py`` 里先建 DRAFT 再迁移的写法配合，
    确保「改回直接构造终态」时至少有人会来看一眼这段代码。
    """
    payload = payload_of(call(build()["submit_approval"], title="出差"))
    item = payload["items"][0]

    assert item["status"] == OrderStatus.PENDING_APPROVAL.value
    assert OrderStatus.DRAFT.value != item["status"]


def test_submit_approval_rejects_an_empty_title() -> None:
    """⚠️ 事由为空时拒绝，并告诉用户**该补什么**。

    申请单的事由是审批人唯一的判断依据。空事由的申请单会被打回，
    但那时已经浪费了一轮审批 —— 不如在提交时就拦住。
    """
    for blank in ("", "   ", "\n"):
        chunk = call(build()["submit_approval"], title=blank)
        payload = payload_of(chunk)

        assert chunk.state is ToolResultState.ERROR, f"title={blank!r} 应当被拒绝"
        assert "事由" in payload["summary"]


def test_submit_approval_does_not_persist_on_rejection() -> None:
    """⚠️ 被拒绝的提交**不留下任何记录**。

    先写库再校验的话，仓储里会堆积一堆 title 为空的脏数据，
    而它们还会出现在后续的查询结果里。
    """
    repo = InMemoryApprovalRepository()
    call(build(approvals=repo)["submit_approval"], title="")

    assert asyncio.run(repo.list_for(user_id=USER)) == []


def test_submit_approval_survives_a_broken_repository() -> None:
    """写入失败时给中文说明，不抛异常。"""
    chunk = call(build(approvals=BoomApprovals())["submit_approval"], title="出差")
    payload = payload_of(chunk)

    assert chunk.state is ToolResultState.ERROR
    assert "ConnectionError" in payload["detail"]


def test_submit_approval_binds_the_current_user() -> None:
    """⚠️ 申请单绑的是**闭包里的用户**，不是模型给的。

    与查询侧的隔离同理：允许模型指定 user_id，等于允许它替别人提交申请。
    """
    repo = InMemoryApprovalRepository()
    payload = payload_of(call(build(approvals=repo, user_id=USER)["submit_approval"], title="出差"))

    stored = asyncio.run(repo.get(request_id=payload["items"][0]["request_id"]))
    assert stored is not None
    assert stored.user_id == USER


# ---------------------------------------------------------------------------
# 五、单号的形态
# ---------------------------------------------------------------------------
def test_request_id_does_not_leak_the_user() -> None:
    """⚠️ 单号里**不含用户标识**。

    单号会被用户看到、会被念出来、会出现在截图里。含用户标识（哪怕是
    哈希）等于把一个可用于关联的信息带出系统外。
    """
    payload = payload_of(call(build(user_id="u-secret-12345")["submit_approval"], title="出差"))
    request_id = payload["items"][0]["request_id"]

    assert "secret" not in request_id.lower()
    assert "12345" not in request_id
    assert "u-" not in request_id.lower()


def test_request_id_does_not_leak_the_time() -> None:
    """⚠️ 单号里**不含时间**。

    含时间会让单号可预测 —— 知道一个人单号的人可以推算出同期其他人的
    单号区间。这是不必要的风险暴露，而单号完全不需要可预测。
    """
    payload = payload_of(call(build()["submit_approval"], title="出差"))
    request_id = payload["items"][0]["request_id"]

    assert not re.search(r"20\d{2}", request_id), f"单号里出现了年份：{request_id}"
    assert not re.search(r"\d{6,}", request_id), f"单号里出现了长数字串：{request_id}"


def test_request_ids_are_unique_across_calls() -> None:
    """⚠️ 连续提交拿到的单号不同。

    单号碰撞的后果是两张不同的申请单共用一个号，后续查询与审批全部错位。
    """
    tool = build()["submit_approval"]
    ids = {payload_of(call(tool, title="出差"))["items"][0]["request_id"] for _ in range(20)}

    assert len(ids) == 20, f"20 次提交只产生了 {len(ids)} 个不同的单号"


def test_request_id_uses_an_unambiguous_alphabet() -> None:
    """⚠️ 单号**不含**易混字符（``0/O``、``1/l``）。

    单号是要被人念、被人手抄的。含易混字符时「念错一位」和「听错一位」
    会变成常态，而用户拿一个错号来问「我的申请呢」时，双方都查不到。
    """
    tool = build()["submit_approval"]
    ids = [payload_of(call(tool, title="出差"))["items"][0]["request_id"] for _ in range(30)]

    for request_id in ids:
        body = request_id.split("-", 1)[1]
        ambiguous = set(body) & set("0O1lI")
        assert not ambiguous, f"单号 {request_id} 含易混字符 {ambiguous}"
