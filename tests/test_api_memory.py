# -*- coding: utf-8 -*-
"""长期记忆接口（``/api/v1/memory/**``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    这个接口面是长期记忆**唯一**的写入入口（在此之前 ``remember`` /
    ``update_profile`` 在 ``src/`` 里一个调用者都没有 —— 详情见
    ``src/server/routers/_memory.py`` 的模块文档）。所以它要防的第一类错
    不是「逻辑写错」，而是**越权**与**身份来源**：

      · ``user_id`` 只来自鉴权。请求体里带 ``user_id`` 必须 422，
        不能「静默忽略」也不能「照着改别人的画像」——后者是水平越权，
        症状会出现在**别人的**订单与账单上；
      · A 员工召回不到 B 员工的笔记（隔离在 ``metadata_filter`` 上，
        假向量库会**真的执行**过滤，见 ``tests/test_memory_semantic.py``）。

    第二类是本项目贯穿始终的那条不对称（读永不抛 / 写照常抛）在
    **HTTP 层**的落点：

      · 写失败 ⇒ ``503`` 且**明确说没写进去**。静默丢弃是一次欺骗；
      · 召回失败 ⇒ ``200`` + ``error`` 非空。检索失败是「这次没查到」，
        不是「你没有记忆」——两者在 UI 上必须可区分；
      · 能力缺失（开关关了 / 向量模型起不来）⇒ ``503`` + 说清缺哪一半。
        这时返回空列表就是一句谎话。

    第三类是**错误信息脱敏**：写失败的原文会被放进响应体，而它同时会进
    访问日志、浏览器缓存与工单截图 —— 里面绝不能出现
    ``scheme://user:pass@`` 形式的凭据（Milvus 的 URI 支持这种写法）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import AsyncClient

from src.config import Settings
from src.memory.profile import InMemoryProfileRepository
from src.memory.service import TravelerMemory
from src.server.constants import MEMORY_ATTR

# 与 ``tests/test_memory_service.py`` 同一写法：复用语义记忆那两个测试替身，
# 而不是在本文件里再抄一份（抄一份的下场是两边慢慢分叉）。
from tests.test_memory_semantic import FakeEmbedding, FakeVectorStore

#: 两个**不同**的身份。隔离用例全靠它们，取值本身无意义。
USER_ALICE = "alice"
USER_BOB = "bob"

#: Alice 的请求头。
ALICE = {"X-User-ID": USER_ALICE}


# ==============================================================================
# 夹具：把假向量库接进 app.state，让接口层能真的走一遍「记住 → 召回」
# ==============================================================================
@dataclass
class Wired:
    """换上去的那套记忆设施（用例要能拿到 store 去注入故障）。"""

    memory: TravelerMemory
    store: FakeVectorStore


def _semantic(settings: Settings, store: FakeVectorStore) -> Any:
    """造一个接了假向量库的 ``SemanticMemory``。"""
    from src.memory.semantic import SemanticMemory

    return SemanticMemory(
        vector_store=store,
        embedding_model=FakeEmbedding(),
        settings=settings,
    )


def wire(app: FastAPI, settings: Settings) -> Wired:
    """把门面换成一套**确定的、不连任何外部服务**的实现，挂在 app.state 上。

    ⚠️ 为什么要换掉真实的那个：``create_root_app`` 装出来的门面
    语义那半指向真 Milvus（本机不可达）、结构化那半指向 sqlite。
    本文件要测的是**路由层的协议与映射**（谁能写、失败怎么报、
    越权挡不挡得住），用真 Milvus 会让「Milvus 没起来」和
    「接口写错了」在测试里长得一模一样。

    Args:
        app (`FastAPI`): 测试应用。
        settings (`Settings`): 测试配置。

    Returns:
        `Wired`: 门面与假向量库。
    """
    store = FakeVectorStore()
    memory = TravelerMemory(
        settings,
        repository=InMemoryProfileRepository(),
        semantic=_semantic(settings, store),
    )
    setattr(app.state, MEMORY_ATTR, memory)
    return Wired(memory=memory, store=store)


@pytest.fixture
def wired(app: FastAPI, settings: Settings) -> Wired:
    """默认形态：两半能力都在，外部依赖全是假的。"""
    return wire(app, settings)


# ==============================================================================
# 一、鉴权与身份来源
# ==============================================================================
@pytest.mark.parametrize(
    ("method", "url", "payload"),
    [
        ("GET", "/api/v1/memory/profile", None),
        ("PUT", "/api/v1/memory/profile", {"seat_preference": "靠窗"}),
        ("POST", "/api/v1/memory/notes", {"text": "我一般坐靠窗"}),
        ("GET", "/api/v1/memory/notes?query=靠窗", None),
        ("DELETE", "/api/v1/memory/notes?all=true", None),
    ],
)
@pytest.mark.asyncio
async def test_every_memory_endpoint_requires_an_identity(
    client: AsyncClient,
    wired: Wired,
    method: str,
    url: str,
    payload: dict[str, Any] | None,
) -> None:
    """★★★ 五个端点**全部**要鉴权：没有身份一律 401。

    ⚠️ 参数化而不是抽查一个：漏掉任何一个端点都是一条**无主的记忆通道**
    （匿名请求能写进某个用户的画像），而漏掉的那一个在代码评审里
    看起来与其它四个一模一样。路由是逐条注册的，鉴权必须逐条验证。

    401 而不是 403/404：它由**鉴权中间件**返回（不是路由抛的），
    语义是「你没提供可用的凭据」（见 docs/01 的五、状态码约定）。
    """
    request = client.build_request(method, url, json=payload)
    response = await client.send(request)

    assert response.status_code == 401, (
        f"{method} {url} 在没有身份时返回了 {response.status_code}，"
        f"期望 401。响应体：{response.text[:200]}"
    )


@pytest.mark.parametrize(
    ("method", "url", "payload"),
    [
        (
            "PUT",
            "/api/v1/memory/profile",
            {"seat_preference": "靠窗", "user_id": USER_BOB},
        ),
        (
            "POST",
            "/api/v1/memory/notes",
            {"text": "我一般坐靠窗", "user_id": USER_BOB},
        ),
        (
            "DELETE",
            "/api/v1/memory/notes",
            {"text": "我一般坐靠窗", "user_id": USER_BOB},
        ),
    ],
)
@pytest.mark.asyncio
async def test_a_user_id_in_the_body_is_rejected(
    client: AsyncClient,
    wired: Wired,
    method: str,
    url: str,
    payload: dict[str, Any],
) -> None:
    """★★★ 请求体里出现 ``user_id`` ⇒ **422**，绝不接受。

    ⚠️ 这是本接口最重要的一条边界。``extra="forbid"`` 挡住的
    不是「多余字段」这种洁癖问题，而是**水平越权**：
    若模型是默认的 ``extra="ignore"``，调用方发 ``{"user_id": "别人"}``
    会被静默忽略 —— 他以为自己改的是别人的画像，接口却改了他自己的，
    两边都以为自己是对的，而账单与审批流会按其中一个的假设走。

    ⚠️ 用 422（框架的校验语义）而不是 400：这是**请求形状**问题，
    客户端应该改请求体，不是重试。
    """
    request = client.build_request(method, url, json=payload, headers=ALICE)
    response = await client.send(request)

    assert response.status_code == 422, (
        f"{method} {url} 接受了请求体里的 user_id（返回 {response.status_code}）！\n"
        "⚠️ 身份只能来自鉴权；接受 body 里的 user_id 等于开放水平越权。"
    )


@pytest.mark.asyncio
async def test_the_profile_belongs_to_the_authenticated_user(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★★★ 画像写在**鉴权身份**名下：换一个身份就看不见。

    ⚠️ 这条与上面那条是同一件事的两面：上面挡「写给别人」，
    这条挡「读到别人」。判据必须是**另一个身份真的查不到**，
    而不是「响应里没有 user_id 字段」（那是形似而神不似的检查）。

    ⚠️ 前置的那次写入**必须断言成功**（对抗性审核 2026-10-03 的发现）：
    「Bob 看不到」这个结论的前提是「Alice 那条记录真的存在」。若写入返回
    4xx/5xx、或实现退化成静默 no-op，``profile is None`` 同样为真 ——
    一个把写入整个删掉的实现会让这条用例**照样全绿**，而它正是用来
    证明隔离的那条。所以下面先确认写进去了，再确认别人看不到。
    """
    written = await client.put(
        "/api/v1/memory/profile",
        json={"seat_preference": "靠窗"},
        headers=ALICE,
    )
    assert written.status_code == 200, f"前置写入就没成功：{written.text}"

    mine = await client.get("/api/v1/memory/profile", headers=ALICE)
    assert mine.json()["profile"]["seat_preference"] == "靠窗", (
        "Alice 自己都读不到刚写的画像 —— 下面的「Bob 看不到」不能说明隔离，"
        "只能说明写入没生效"
    )

    bob = await client.get("/api/v1/memory/profile", headers={"X-User-ID": USER_BOB})

    assert bob.status_code == 200
    assert bob.json()["profile"] is None, (
        "Bob 看到了 Alice 的画像！身份隔离失效。\n"
        f"他拿到了：{bob.json()['profile']}"
    )


# ==============================================================================
# 二、结构化画像：读写闭环
# ==============================================================================
@pytest.mark.asyncio
async def test_a_fresh_user_has_no_profile(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """新用户读到 ``profile: null``（200），**不是** 404/503。

    ⚠️ 「没有画像记录」是一个正常的业务状态（所有新员工都如此），
    不是错误。把它报成 404 会让前端的「首次填表」流程变成一条异常分支。
    """
    response = await client.get("/api/v1/memory/profile", headers=ALICE)

    assert response.status_code == 200
    body = response.json()
    assert body["profile"] is None
    assert body["trace_id"]


@pytest.mark.asyncio
async def test_put_then_get_round_trips(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★★ 写进去的东西读得回来，且**经过 validator**（舱位被规范化）。

    ⚠️ 断言 ``"BUSINESS"`` 而不是 ``"business"`` 是有意的：它证明请求穿过
    了 :class:`~src.memory.profile.TravelerProfile` 的校验器，而不是被
    原样存了下来。绕过 validator 的写入是一条能塞进非法值的旁路 ——
    下游（下单、差旅标准校验）会按枚举值匹配，匹配不上的表现是
    「偏好静默失效」，不是报错。
    """
    written = await client.put(
        "/api/v1/memory/profile",
        json={
            "seat_preference": "靠窗",
            "preferred_cabin": "business",
            "preferred_airlines": ["CA", "CA", "MU"],
        },
        headers=ALICE,
    )

    assert written.status_code == 200, written.text
    assert written.json()["profile"]["preferred_cabin"] == "BUSINESS"
    # 去重且保序（见 TravelerProfile._normalize_list）。
    assert written.json()["profile"]["preferred_airlines"] == ["CA", "MU"]

    read = await client.get("/api/v1/memory/profile", headers=ALICE)
    assert read.json()["profile"] == written.json()["profile"], (
        "写入返回的画像与读回来的不一致 —— 两者必须是同一份。"
    )
    assert read.json()["profile"]["user_id"] == USER_ALICE


@pytest.mark.asyncio
async def test_a_patch_only_touches_the_fields_it_names(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★★ 部分更新是**部分**：没提到的字段原样保留。

    ⚠️ 这是 ``ProfilePatch`` 存在的主要理由（见其文档）。写错的表现是
    「用户改了座位偏好，成本中心却没了」——而它只在用户下一次下单时
    才会暴露（订单走了错账），中间没有任何报错。
    """
    await client.put(
        "/api/v1/memory/profile",
        json={"cost_center": "CC-42", "seat_preference": "过道"},
        headers=ALICE,
    )
    second = await client.put(
        "/api/v1/memory/profile",
        json={"seat_preference": "靠窗"},
        headers=ALICE,
    )

    profile = second.json()["profile"]
    assert profile["seat_preference"] == "靠窗"
    assert profile["cost_center"] == "CC-42", (
        "第二次更新把没提到的 cost_center 清掉了 —— patch 语义写错了。"
    )


@pytest.mark.asyncio
async def test_clearing_a_field_takes_an_empty_value_not_null(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★★ 清空字段靠**空值**（`[]` / `""`），`null` 永远只表示「别动它」。

    ⚠️ 这条钉住的是接口**文案**与行为的一致性：早先文档与报错信息都写着
    「清空字段走整体覆盖」，而整体覆盖**根本没有 HTTP 入口**
    （``put_profile`` 在 ``src/`` 里只有测试在调）—— 又是一处「文档指向
    一个不存在的能力」（对抗性审核 2026-10-03 的发现）。现在文案改成
    「给空值」，这条用例证明「给空值」真的有效，而不是又一句空话。

    ⚠️ 后半段同样重要：``null`` **不能**清空 —— 若哪天有人把
    ``ProfilePatch.apply`` 的 ``is not None`` 判断写成「传了就覆盖」，
    所有前端表单（未填字段天然是 ``null``）都会变成静默的清空器。
    """
    await client.put(
        "/api/v1/memory/profile",
        json={"preferred_airlines": ["CA", "MU"], "seat_preference": "靠窗"},
        headers=ALICE,
    )

    cleared = await client.put(
        "/api/v1/memory/profile",
        json={"preferred_airlines": [], "seat_preference": ""},
        headers=ALICE,
    )
    profile = cleared.json()["profile"]
    assert profile["preferred_airlines"] == [], "给 [] 没能清空列表字段"
    assert profile["seat_preference"] == "", "给 \"\" 没能清空字符串字段"

    untouched = await client.put(
        "/api/v1/memory/profile",
        json={"seat_preference": None, "cost_center": "CC-7"},
        headers=ALICE,
    )
    later = untouched.json()["profile"]
    assert later["cost_center"] == "CC-7"
    assert later["seat_preference"] == "", (
        "null 把字段清掉了（或改掉了）—— null 的语义是「不改这个字段」"
    )


@pytest.mark.asyncio
async def test_empty_values_also_clear_cabin_and_a_flyer_number(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★★ 同一条「空值即清空」的约定，在两个**曾经清不掉**的字段上也成立。

    ⚠️ 这两个字段是同一轮对抗性审核（2026-10-03）挑出来的死角，
    而它们各自是**另一种**坏法：

      · ``preferred_cabin``：给了 ``""`` 会**报 400**（``_validate_cabin``
        把空串当非法值）。于是接口文案里那句「字符串给 ``""``」在这个
        字段上是**假的**，而它也就成了唯一一个「给了空值反而报错」的
        字符串字段 —— 想让助手别再给自己挑舱位，无路可走。
      · ``frequent_flyer_numbers``：只合并不删除，``{}`` 与「不传」
        都是「不动」。于是它**只能增不能减**：用户换了航司之后，
        旧卡号会永远留在画像里、被注入 Prompt、被下单工具拿走。

    ⚠️ 修法不是把文案写得更啰嗦（「除舱位与常旅客号外…」），
    而是**把约定补成真的** —— 空值在哪儿都是「清空」。这条用例同时
    钉住四行行为里剩下的两行，``ProfilePatchIn`` 的文档里有那张表。
    """
    await client.put(
        "/api/v1/memory/profile",
        json={
            "preferred_cabin": "BUSINESS",
            "frequent_flyer_numbers": {"CA": "CA-123", "MU": "MU-456"},
        },
        headers=ALICE,
    )

    cleared = await client.put(
        "/api/v1/memory/profile",
        json={"preferred_cabin": "", "frequent_flyer_numbers": {"CA": ""}},
        headers=ALICE,
    )
    assert cleared.status_code == 200, (
        f"给舱位空串被判成了非法值（400）：{cleared.text}"
    )
    profile = cleared.json()["profile"]
    assert profile["preferred_cabin"] is None, (
        "给 `\"\"` 没能清空舱位偏好 —— 它仍然是唯一一个清不掉的字段？"
    )
    assert profile["frequent_flyer_numbers"] == {"MU": "MU-456"}, (
        "`{\"CA\": \"\"}` 没能删掉国航的卡号 —— 常旅客号仍然只增不减？"
    )


@pytest.mark.asyncio
async def test_frequent_flyer_numbers_are_merged_not_replaced(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """常旅客号是**合并**：新航司不挤掉旧航司。

    ⚠️ 这一条是 ``ProfilePatch.apply`` 里唯一一处「合并不是替换」的字段，
    而它恰好是最贵的：用户说「我的南航卡号是 X」时把国航卡号删掉，
    后果是他下一次坐国航时累积不到里程，且**没有任何提示**。
    """
    await client.put(
        "/api/v1/memory/profile",
        json={"frequent_flyer_numbers": {"CA": "111"}},
        headers=ALICE,
    )
    second = await client.put(
        "/api/v1/memory/profile",
        json={"frequent_flyer_numbers": {"CZ": "222"}},
        headers=ALICE,
    )

    assert second.json()["profile"]["frequent_flyer_numbers"] == {
        "CA": "111",
        "CZ": "222",
    }


@pytest.mark.asyncio
async def test_an_empty_patch_is_rejected(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """一个字段都不给的 PUT ⇒ 400（不是一个「看起来成功」的 200）。

    ⚠️ 200 会让调用方以为偏好写进去了（典型来源是前端表单字段名拼错，
    而 ``extra="forbid"`` 挡住了拼错的字段名、却挡不住「一个都没填」）。
    """
    response = await client.put("/api/v1/memory/profile", json={}, headers=ALICE)

    assert response.status_code == 400, response.text
    assert "一个字段都没给" in response.json()["detail"]


@pytest.mark.asyncio
async def test_an_illegal_cabin_is_rejected(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★ 非法舱位 ⇒ **400**（不是 500，也不是静默存下）。

    ⚠️ 400 与 503 的区别在这里很实在：400 告诉客户端「改请求」，
    503 会让它重试一个永远不会成功的请求。校验错误是前者的典型。
    """
    response = await client.put(
        "/api/v1/memory/profile",
        json={"preferred_cabin": "超级头等舱"},
        headers=ALICE,
    )

    assert response.status_code == 400, response.text
    assert "不合法" in response.json()["detail"]


# ==============================================================================
# 三、语义笔记：记住 / 召回 / 忘记
# ==============================================================================
@pytest.mark.asyncio
async def test_remember_then_recall_then_forget(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★★★ 全链路：POST 记住 → GET 召回 → DELETE 忘掉 → 召回不到了。

    ⚠️ 断言的是一个**闭环**而不是三个孤立的状态码：只测「POST 返回 200」
    的话，一个「收下了但没写进向量库」的实现照样全绿 —— 而那正是
    用户最恨的那种失败（他说了「记住」，助手说「好」，下周它不记得）。
    """
    remembered = await client.post(
        "/api/v1/memory/notes",
        json={"text": "我一般坐靠窗，不吃辣"},
        headers=ALICE,
    )
    assert remembered.status_code == 200, remembered.text
    note = remembered.json()["note"]
    assert note["text"] == "我一般坐靠窗，不吃辣"
    assert note["kind"] == "preference"
    assert note["note_id"]

    found = await client.get(
        "/api/v1/memory/notes",
        params={"query": "坐靠窗"},
        headers=ALICE,
    )
    assert found.status_code == 200, found.text
    assert found.json()["error"] is None, found.text
    assert [item["note_id"] for item in found.json()["notes"]] == [note["note_id"]]

    forgotten = await client.request(
        "DELETE",
        "/api/v1/memory/notes",
        json={"text": "我一般坐靠窗，不吃辣"},
        headers=ALICE,
    )
    assert forgotten.status_code == 200, forgotten.text
    assert forgotten.json()["forgotten"] == "我一般坐靠窗，不吃辣"

    after = await client.get(
        "/api/v1/memory/notes",
        params={"query": "坐靠窗"},
        headers=ALICE,
    )
    assert after.json()["notes"] == [], "忘记之后仍然召回得到 —— 删除没有生效。"


@pytest.mark.asyncio
async def test_remembering_the_same_text_twice_does_not_duplicate(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★★ 同一句话记住两次**不产生两条**（note_id 由 (user_id, text) 决定）。

    ⚠️ 幂等不是省空间：Top-K 检索里两条一模一样的笔记会**挤掉**别的笔记的
    位置 —— 用户越是反复提到某件事，它就越把其他记忆挤出召回窗口，
    恰好与直觉相反（见 ``note_id`` 的文档）。这也是本接口用 200 而不是
    201 的理由：重复写入是一次**覆盖**，不是一次创建。
    """
    first = await client.post(
        "/api/v1/memory/notes",
        json={"text": "我不吃辣"},
        headers=ALICE,
    )
    second = await client.post(
        "/api/v1/memory/notes",
        json={"text": "我不吃辣"},
        headers=ALICE,
    )

    assert first.json()["note"]["note_id"] == second.json()["note"]["note_id"]

    found = await client.get(
        "/api/v1/memory/notes",
        params={"query": "不吃辣"},
        headers=ALICE,
    )
    assert len(found.json()["notes"]) == 1, f"同一条笔记出现了两次：{found.json()}"


@pytest.mark.asyncio
async def test_notes_are_scoped_to_the_authenticated_user(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★★★ A 员工的笔记**绝不**出现在 B 员工的召回结果里。

    ⚠️ 这不是「多召回几条」的小问题，是一次数据泄露：差旅笔记里会出现
    行程、住哪家酒店、和谁一起出差。隔离靠的是
    ``SemanticMemory.recall`` 里的 ``metadata_filter={user_id}``，
    而本文件用的假向量库会**真的执行**过滤 —— 于是这条用例能抓住
    「filter 忘了传」与「后端没拿它过滤」两类真实故障。

    ⚠️ 顺带钉住一个语义：Bob 的召回是 ``200 + 空列表``（他确实没有笔记），
    而不是 403 —— 接口不该暴露「Alice 有多少条笔记」这种元信息。

    ⚠️ 前置的写入同样**必须断言成功**（对抗性审核 2026-10-03 的发现）：
    「Bob 召回为空」的前提是「Alice 那条笔记真的写进去了」。
    一个把写入删掉的实现会让两个身份都召回为空，这条用例照样全绿。
    """
    written = await client.post(
        "/api/v1/memory/notes",
        json={"text": "我一般坐靠窗，不吃辣"},
        headers=ALICE,
    )
    assert written.status_code == 200, f"前置写入就没成功：{written.text}"

    mine = await client.get(
        "/api/v1/memory/notes",
        params={"query": "坐靠窗"},
        headers=ALICE,
    )
    assert [note["text"] for note in mine.json()["notes"]] == ["我一般坐靠窗，不吃辣"], (
        "Alice 自己都召回不到刚写的笔记 —— 下面的「Bob 看不到」说明不了隔离"
    )

    bob = await client.get(
        "/api/v1/memory/notes",
        params={"query": "坐靠窗"},
        headers={"X-User-ID": USER_BOB},
    )

    assert bob.status_code == 200
    assert bob.json()["notes"] == [], (
        "Bob 召回到了 Alice 的笔记 —— 用户隔离失效，这是一次数据泄露。\n"
        f"他拿到了：{bob.json()['notes']}"
    )


@pytest.mark.asyncio
async def test_delete_all_removes_every_note_and_reports_the_count(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★★ ``?all=true`` 清空自己的全部笔记，并**返回条数**。

    ⚠️ 返回条数是必要的：只回一个 200 的话，调用方无法区分
    「清空了 3 条」与「本来就没有任何记录」——而这两种处境下用户的
    下一步动作完全不同（前者确认，后者要怀疑是不是自己记错了账号）。
    """
    for text in ("我不吃辣", "我一般坐靠窗"):
        await client.post("/api/v1/memory/notes", json={"text": text}, headers=ALICE)

    cleared = await client.request(
        "DELETE",
        "/api/v1/memory/notes",
        params={"all": "true"},
        headers=ALICE,
    )

    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["removed"] == 2

    after = await client.get(
        "/api/v1/memory/notes",
        params={"query": "坐靠窗"},
        headers=ALICE,
    )
    assert after.json()["notes"] == []


@pytest.mark.asyncio
async def test_delete_without_a_target_is_rejected(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """既不给 ``all=true`` 也不给请求体 ⇒ 400，而不是「什么都不做」。

    ⚠️ 「什么都不做但返回 200」的问题在于：一个把参数拼错的调用方
    （比如写成 ``?all=1`` 或给了一个空 body）会以为删成功了 ——
    而删除是不可撤销的，他不会再去检查。
    """
    response = await client.request("DELETE", "/api/v1/memory/notes", headers=ALICE)

    assert response.status_code == 400, response.text
    assert "二选一" in response.json()["detail"]


@pytest.mark.asyncio
async def test_delete_with_both_targets_is_rejected(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """``all=true`` 与请求体**同时**给出 ⇒ 400，不猜。

    ⚠️ 删除不可撤销，而这两个参数表达的是**完全不同**的意图
    （删一条 vs 清空）。这时候「按优先级挑一个」是最坏的选择：
    调用方的意图形容与执行结果不一致，而他永远不会知道。
    """
    response = await client.request(
        "DELETE",
        "/api/v1/memory/notes",
        params={"all": "true"},
        json={"text": "我不吃辣"},
        headers=ALICE,
    )

    assert response.status_code == 400, response.text
    assert "同时给出" in response.json()["detail"]


@pytest.mark.asyncio
async def test_recall_requires_a_query(client: AsyncClient, wired: Wired) -> None:
    """★ 不带 ``query`` 的召回 ⇒ 422。

    ⚠️ 这是一个**检索**接口，没有查询就没有答案。允许空查询的话，
    它会返回一个空列表，而空列表在调用方眼里等于「你没有任何记忆」——
    一句没有依据的结论。
    """
    response = await client.get("/api/v1/memory/notes", headers=ALICE)

    assert response.status_code == 422, response.text


@pytest.mark.asyncio
async def test_a_whitespace_only_query_is_rejected(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★ 全是空白的 ``query`` ⇒ 400，而不是一个「没有依据的空列表」。

    ⚠️ ``min_length=1`` 对 ``"   "`` 是成立的，而 ``recall`` 对**空白**
    查询会刻意返回空结果且 ``error=None``（那是给对话链路用的降级语义）。
    两者叠起来，``?query=%20%20`` 就会返回 ``{"notes": [], "error": null}``
    —— 与「这个用户确实没有相关记忆」在响应里完全同形。
    POST / DELETE 两个端点都对空白做了先 strip 再判空，这里补上同一步
    （对抗性审核 2026-10-03 的发现）。
    """
    response = await client.get(
        "/api/v1/memory/notes",
        params={"query": "   "},
        headers=ALICE,
    )

    assert response.status_code == 400, response.text
    assert "空" in response.json()["detail"]


# ==============================================================================
# 四、能力缺失：说清缺的是哪一半
# ==============================================================================
@pytest.mark.asyncio
async def test_notes_return_503_when_semantic_memory_is_missing(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
) -> None:
    """★★★ 没有语义能力时，三个笔记端点全 503，且 hint 指向**向量模型**。

    ⚠️ 这正是「读也报 503」的那条取舍：``recall`` 在对话链路上永不抛、
    失败降级，那是对的（锦上添花）；但这里是**显式**接口，调用方在等
    一个答案。返回 ``200 + 空列表`` 等于把「能力缺失」伪装成
    「你没有记忆」——后者是**错误的事实**。

    ⚠️ hint 必须区分「关掉了」与「起不来」：两者的处置完全不同
    （改配置 vs 看启动日志）。混成一句「记忆不可用」的话，
    运维会去改一个本来就对的开关。
    """
    setattr(
        app.state,
        MEMORY_ATTR,
        TravelerMemory(settings, repository=InMemoryProfileRepository()),
    )

    responses = [
        await client.post("/api/v1/memory/notes", json={"text": "我不吃辣"}, headers=ALICE),
        await client.get("/api/v1/memory/notes", params={"query": "不吃辣"}, headers=ALICE),
        await client.request(
            "DELETE",
            "/api/v1/memory/notes",
            params={"all": "true"},
            headers=ALICE,
        ),
    ]

    for response in responses:
        assert response.status_code == 503, response.text
        assert "语义" in response.json()["detail"], response.text


@pytest.mark.asyncio
async def test_profile_is_503_when_memory_is_disabled(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
) -> None:
    """★★ 总开关关掉时，画像接口 503，且 hint 里点名那个开关。

    ⚠️ ``build_memory`` 在关闭时返回的是一个**两半都为 None** 的门面
    （不是 None 本身）。若路由忘了检查 ``enabled``，症状会是
    ``has_repository`` 为假 → 走到「没有仓储」那条 hint ——
    一模一样的状态码，指向完全错误的排查方向。
    """
    disabled = settings.model_copy(deep=True)
    disabled.memory.enabled = False
    setattr(app.state, MEMORY_ATTR, TravelerMemory(disabled))

    read = await client.get("/api/v1/memory/profile", headers=ALICE)
    written = await client.put(
        "/api/v1/memory/profile",
        json={"seat_preference": "靠窗"},
        headers=ALICE,
    )

    for response in (read, written):
        assert response.status_code == 503, response.text
        assert "ALIGO__MEMORY__ENABLED" in response.json()["detail"], response.text


# ==============================================================================
# 五、写失败必须说出来（且脱敏）
# ==============================================================================
@pytest.mark.asyncio
async def test_a_failed_write_is_reported_not_swallowed(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★★★ 向量库写失败 ⇒ 503，且**明确说「没有生效」**。

    ⚠️ 这是「读永不抛 / 写照常抛」在 HTTP 层的落点，也是本接口存在的
    全部意义之一：用户说了「记住这个」，静默丢弃是一次欺骗 ——
    他下周才会发现助手没记住，而中间没有任何一处告诉过他。

    ⚠️ 故障用 ``collection not found`` 这个真实错误（2026-10-03 在容器里
    实测过的那个）：它正是「Milvus 里没有集合」时的原文。
    """
    wired.store.fail_with = RuntimeError("collection not found: aligo_memory")

    response = await client.post(
        "/api/v1/memory/notes",
        json={"text": "我一般坐靠窗"},
        headers=ALICE,
    )

    assert response.status_code == 503, response.text
    detail = response.json()["detail"]
    assert "失败" in detail and "没有" in detail, detail


@pytest.mark.asyncio
async def test_a_failed_write_really_did_not_land(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★★ 503 说「没生效」，那就必须**真的**没生效。

    ⚠️ 与上一条配对：上一条断言「报错了」，这一条断言「报错说的是实话」。
    一个「先写进去、再因为某处异常返回 503」的实现会让用户重试，
    于是同一条笔记被写两遍（幂等能兜住），但若他改了口径重说一次，
    库里就留下两条互相矛盾的记忆 —— 而调用方以为第一条没写进去。
    """
    wired.store.fail_with = RuntimeError("collection not found: aligo_memory")
    await client.post("/api/v1/memory/notes", json={"text": "我不吃辣"}, headers=ALICE)

    wired.store.fail_with = None
    found = await client.get(
        "/api/v1/memory/notes",
        params={"query": "不吃辣"},
        headers=ALICE,
    )

    assert found.json()["notes"] == [], (
        "写失败时返回了 503，但内容其实写进去了 —— 调用方会重试，"
        "于是库里出现两条重复/矛盾的记忆。"
    )


@pytest.mark.asyncio
async def test_a_failed_recall_is_200_with_an_error(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★★ 召回失败 ⇒ **200** + ``error`` 非空、``notes`` 为空。

    ⚠️ 与写失败的映射刻意不同：召回失败只影响「锦上添花」，
    调用方需要的是「这次没查到」这个信息，而不是一个它必须处理的
    传输层故障（那会引出重试、告警、熔断）。但 ``error`` 必须存在 ——
    否则调用方无法把「没有相关记忆」与「这次没查到」区分开，
    而这两者在前端上应当是两种不同的提示。
    """
    wired.store.fail_with = RuntimeError("milvus unreachable")

    response = await client.get(
        "/api/v1/memory/notes",
        params={"query": "靠窗"},
        headers=ALICE,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["notes"] == []
    assert body["error"], "召回失败却没有 error —— 调用方无法区分「没有记忆」与「没查到」"


@pytest.mark.asyncio
async def test_error_messages_do_not_leak_credentials(
    client: AsyncClient,
    wired: Wired,
) -> None:
    """★★★ 写失败的响应体里**绝不能**出现连接串凭据。

    ⚠️ 这类响应是最容易被整段贴进工单与聊天窗口的东西，而 Milvus 的 URI
    支持 ``http://user:pass@host:19530`` 这种写法 —— 框架抛出的原始
    异常里很可能带着整条 URI。``safe_error`` 会把 ``scheme://user:pass@``
    剥掉，这条用例就是它的守门人。
    """
    wired.store.fail_with = RuntimeError(
        "failed to connect http://aligo:s3cret-pw@milvus:19530",
    )

    response = await client.post(
        "/api/v1/memory/notes",
        json={"text": "我不吃辣"},
        headers=ALICE,
    )

    assert "s3cret-pw" not in response.text, (
        f"响应体里出现了口令明文：{response.text[:300]}"
    )


class _ExplodingRepository:
    """写路径上抛 ``ValueError`` 的画像仓储替身。

    ⚠️ 刻意选 ``ValueError``：它是**唯一**会被旧实现误判成 400 的类型
    （pydantic 的 ``ValidationError`` 也是它的子类，而仓储把库里读出来的
    记录反序列化时同样会抛它）。400 的含义是「你的请求有问题，改了再来」
    —— 于是一次存储故障被永久地归咎于调用方：他不会重试，
    而故障会一直持续到他放弃这个功能。

    三个方法都实现，是为了满足 :class:`ProfileRepository` 协议的形状；
    用例只走 ``merge``。
    """

    async def get(self, user_id: str) -> None:  # pragma: no cover - 用例不走
        """读路径（用例不经过）。"""
        return None

    async def merge(self, user_id: str, patch: object) -> None:
        """写路径：模拟一次**驱动层**的连接失败。

        Raises:
            ValueError: 带着一条含口令的 DSN —— 真实驱动（asyncpg /
                SQLAlchemy 的 URL 解析）抛出的原文就长这样。
        """
        raise ValueError(
            "failed to connect postgresql+asyncpg://aligo:s3cret-pw@db:5432/aligo",
        )

    async def upsert(self, profile: object) -> None:  # pragma: no cover - 用例不走
        """整体覆盖（用例不经过）。"""
        raise AssertionError("本替身不该被 upsert 调用")


@pytest.mark.asyncio
async def test_a_storage_value_error_is_a_503_not_a_400(
    client: AsyncClient,
    app: FastAPI,
    settings: Settings,
) -> None:
    """★★★ 存储层抛的 ``ValueError`` ⇒ **503**（不是 400），且不泄漏凭据。

    ⚠️ 这条用例是 2026-10-03 对抗性审核抓到的真实缺陷的守门人。旧写法是
    ``except ValueError: return _error(400, f"画像字段不合法：{exc}")``，
    而那个 ``try`` 包住的是 ``memory.update_profile`` 的**整条 I/O 路径**
    （仓储读库、反序列化、驱动连接），远不止「值不合法」。两个后果：
    「服务端故障」被报成「你的请求有问题」，以及异常原文被拼进响应体 ——
    实测能让 ``postgresql+asyncpg://user:pw@…`` 明文出现在响应里。

    ⚠️ 断言 503 与「不泄漏」缺一不可：只断言状态码的话，一个返回
    503 但把原文带上的实现照样能过。
    """
    setattr(
        app.state,
        MEMORY_ATTR,
        TravelerMemory(
            settings,
            repository=_ExplodingRepository(),  # type: ignore[arg-type]
            semantic=None,
        ),
    )

    response = await client.put(
        "/api/v1/memory/profile",
        json={"seat_preference": "靠窗"},
        headers=ALICE,
    )

    assert response.status_code == 503, response.text
    assert "s3cret-pw" not in response.text, (
        f"响应体里出现了口令明文：{response.text[:300]}"
    )
    assert "失败" in response.json()["detail"]
