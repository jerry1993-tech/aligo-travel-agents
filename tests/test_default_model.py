# -*- coding: utf-8 -*-
"""零密钥降级：默认模型端点、按需播种、以及「发送按钮不再是灰的」那条线。

这些用例存在的理由，是一个真实缺陷：**零密钥部署下探针全绿，
用户却一个字都发不出去**。完整因果链：

    没有凭据 ⇒ ``GET /credential/`` 返回空
      ⇒ 前端「可用模型」分组为空 ⇒ ``getFirstAvailableModel()`` 返回 ``null``
      ⇒ ``selectedModel === null`` ⇒ 发送按钮 ``disabled``

修复由三块拼成，本文件逐块覆盖：

    1. ``MockChatModel.list_models()`` 给出模型卡片（否则分组里元素为空）；
    2. 每个用户按需播一条 Mock 凭据（否则凭据列表是空的）；
    3. ``GET /api/v1/default-model`` 把「我该用哪个模型」显式说出来
       （否则 curl / smoke / 前端都只能去猜）。

⚠️ 第 2 块有**两条**独立的播种路径，两条都必须有用例：
    · 端点路径：``resolve_default_model`` 自己会播种（本文件覆盖）；
    · 中间件路径：**浏览器走的是这条** —— 它压根不请求我们这个端点，
      而是直接打 ``/credential/`` 与 ``/model/``。只测端点的话，
      中间件坏掉时全部用例照样绿（这一点由
      ``test_credential_listing_is_seeded_before_the_first_read`` 挡）。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.config import Settings
from src.llm.degradation import (
    MODE_CONFIGURED,
    MODE_MISSING,
    MODE_MOCK,
    MODE_SHARED,
    mock_chat_model_config,
    resolve_default_model,
)
from src.llm.mock import (
    MOCK_CREDENTIAL_TYPE,
    MOCK_MODEL_NAME,
    MockChatModel,
    mock_credential_id,
    mock_model_card,
)
from src.llm.system_credential import (
    SYSTEM_CREDENTIAL_ID,
    SYSTEM_CREDENTIAL_NAME,
    SYSTEM_USER_ID,
    SystemCredentialAccessPolicy,
    build_access_policy,
    ensure_system_credential,
    resolve_shared_model,
    sharing_enabled,
    system_chat_model_config,
    system_credential_type,
)

#: 一个**明显是假的**密钥。用它而不是真 key：本文件的用例一旦真的调用了
#: 模型就会立刻失败（而不是悄悄花掉别人的钱），而 `sk-not-a-real-key`
#: 在任何一个真实供应商那里都是无效的。
#:
#: ⚠️ 它不是秘密，也**不能**被当成秘密的占位 —— 需要用密钥做断言的地方
#: 一律断言「响应里没有出现它」，见 test_shared_* 那几条。
FAKE_API_KEY = "sk-not-a-real-key"

#: 一个此前从未出现过的身份 —— 用它才能观察到「第一次请求时的播种」。
FRESH_HEADERS = {"X-User-ID": "test-fresh-user"}


class IntegrityError(Exception):
    """与 SQLAlchemy 的 ``IntegrityError`` **同名**的异常替身。

    ⚠️ 类名就是契约：``src/llm/preset_credential.py`` 用
    ``type(exc).__name__ == "IntegrityError"`` 判定「这是主键冲突，
    不是故障」—— 刻意不用 ``isinstance``，以免把 SQLAlchemy 拖进
    「零密钥启动」那条路径的 import 面。所以替身必须真的叫这个名字。
    这里也不 import SQLAlchemy 的实现：本文件测的是**我们的判据**，
    而不是某个驱动的行为。
    """

#: 与 FRESH_HEADERS 不同的第二个新身份，用于验证按用户隔离。
OTHER_HEADERS = {"X-User-ID": "test-other-user"}


# ==============================================================================
# 替身：只记录调用的存储
# ==============================================================================
class _RecordingStorage:
    """只实现本模块用到的两个方法的存储替身。

    ⚠️ 刻意**不**继承框架的 ``StorageBase``：那会要求实现上百个抽象方法，
    而本文件关心的只有「有没有写、写了几次、写的 id 是不是同一个」。
    用真实实现（sqlite）也能测，但那样测出来的是存储层的行为，
    而这里要钉住的是**我们的判据**。
    """

    def __init__(self, credentials: list[Any] | None = None) -> None:
        """初始化。

        Args:
            credentials (`list[Any] | None`): ``list_credentials`` 要返回的既有凭据。
        """
        self._credentials = credentials or []
        self.upserted: list[str] = []
        #: ``(user_id, credential)`` 明细。``upserted`` 只记 id 是为了让
        #: 既有断言读起来短；**归属**（属主是谁）是本文件后半段的核心，
        #: 单独留一份完整记录，免得为了它去改既有断言。
        self.upsert_calls: list[tuple[str, Any]] = []
        self.list_calls = 0
        self.fail_upsert = False
        #: 还要抛几次主键冲突。见 ``upsert_credential``。
        self.conflict_upsert = 0

    async def upsert_credential(self, user_id: str, credential_data: Any) -> str:
        """记录一次播种。

        Raises:
            RuntimeError: ``fail_upsert`` 为真时（模拟一般的存储故障）。
            IntegrityError: ``conflict_upsert`` 还剩次数时
                （模拟「另一个执行者抢先写了同一条」）。
        """
        if self.fail_upsert:
            raise RuntimeError("模拟存储故障")
        if self.conflict_upsert > 0:
            self.conflict_upsert -= 1
            raise IntegrityError("模拟主键冲突")
        self.upserted.append(credential_data.id)
        self.upsert_calls.append((user_id, credential_data))
        return credential_data.id

    async def list_credentials(self, user_id: str) -> list[Any]:
        """返回既有凭据。"""
        self.list_calls += 1
        return list(self._credentials)


# ==============================================================================
# 一、纯单元：确定性 id 与模型卡片
# ==============================================================================
def test_mock_credential_id_is_deterministic_and_distinct() -> None:
    """凭据 id 必须**确定性**且**按用户区分**。

    确定性是幂等播种的前提：id 随机的话，「补一条」会变成「每次调用再加一条」，
    用户的凭据列表里会堆出一串一模一样的降级模型。
    """
    assert mock_credential_id("alice") == mock_credential_id("alice")
    assert mock_credential_id("alice") != mock_credential_id("bob")
    assert mock_credential_id("alice").startswith("aligo-mock-")


def test_mock_credential_id_stays_inside_the_column_limit() -> None:
    """超长 user_id 也必须落在 ``credentials.id`` 的 255 字符上限内。

    ``credentials.id`` 是**全局**主键，超长会被数据库拒绝或截断，
    而截断后可能与另一个用户撞上 —— 症状是某个用户的首次播种以一个
    看不出原因的 500 收场。
    """
    long_user = "u" * 500
    derived = mock_credential_id(long_user)
    assert len(derived) <= 255, f"派生的凭据 id 过长（{len(derived)}）"
    # 两个不同的超长 id 不能撞到同一条哈希上。
    assert derived != mock_credential_id("u" * 499 + "v")


def test_mock_model_card_matches_the_model_name() -> None:
    """``list_models()`` 必须给出**与配置里的模型名一致**的卡片。

    名字对不上时，前端选择器会选中一个「列表里不存在」的模型，
    于是 ``selectedModelCard`` 为 null —— 界面上看不出错，
    但模型参数面板与输入类型推断会退化成空行为。
    """
    cards = MockChatModel.list_models()
    assert len(cards) == 1, f"期望恰好一张卡片，实际 {len(cards)}"
    card = cards[0]
    assert card.name == MOCK_MODEL_NAME == mock_model_card().name
    assert card.label, "卡片没有展示名，前端下拉框会显示空白"
    assert card.context_size > 0 and card.output_size > 0


# ==============================================================================
# 二、纯单元：判据（三种形态 + 失败形态）
# ==============================================================================
async def test_resolve_default_model_seeds_and_returns_usable_config(
    settings: Settings,
) -> None:
    """零密钥 ⇒ 播种一次并返回一份可直接使用的配置。"""
    storage = _RecordingStorage()

    resolved = await resolve_default_model(storage, settings, "alice")

    assert resolved["mode"] == MODE_MOCK
    assert resolved["chat_model_config"] == mock_chat_model_config(
        mock_credential_id("alice"),
    ), "返回的配置与写入的凭据对不上 —— 会话会指向一条不存在的凭据"
    assert storage.upserted == [mock_credential_id("alice")], (
        f"期望恰好播种一次，实际 {storage.upserted}"
    )
    assert resolved["hint"]


async def test_resolve_default_model_is_idempotent(settings: Settings) -> None:
    """重复调用写的是**同一个** id（``upsert`` 因此是就地更新）。"""
    storage = _RecordingStorage()

    first = await resolve_default_model(storage, settings, "alice")
    second = await resolve_default_model(storage, settings, "alice")

    assert first["chat_model_config"] == second["chat_model_config"]
    assert storage.upserted == [mock_credential_id("alice")] * 2


async def test_resolve_default_model_never_seeds_a_mock_when_a_key_is_configured(
    settings: Settings,
) -> None:
    """有真密钥时**绝不能**再播种 Mock。

    这一条是**安全边界**：有密钥的部署里突然多出一条 Mock 凭据，
    会让人误以为「降级在生效」，而真实调用走的又是另一条路 ——
    排查方向会被彻底带偏。同时它也钉住了「不替用户选真实模型」：
    用户已有凭据时只回 ``configured``，不给配置。
    """
    keyed = settings.model_copy(deep=True)
    keyed.llm.api_key = "sk-not-a-real-key"
    storage = _RecordingStorage()

    # 没有凭据 ⇒ missing（需要人来配）。
    missing = await resolve_default_model(storage, keyed, "alice")
    assert missing["mode"] == MODE_MISSING
    assert missing["chat_model_config"] is None

    # 有凭据 ⇒ configured，且**仍然**不播种、不给配置。
    storage_with_cred = _RecordingStorage(credentials=[object()])
    configured = await resolve_default_model(storage_with_cred, keyed, "alice")
    assert configured["mode"] == MODE_CONFIGURED
    assert configured["chat_model_config"] is None

    assert storage.upserted == [] and storage_with_cred.upserted == [], (
        "配置了真实密钥时仍然播种了 Mock 凭据"
    )


async def test_resolve_default_model_reports_storage_failure_without_raising(
    settings: Settings,
) -> None:
    """存储写不进去时给一句可执行的提示，而不是抛异常变成 500。

    本端点的契约是「告诉我该怎么办」。抛出去只会变成一个没有信息量的 500。
    """
    storage = _RecordingStorage()
    storage.fail_upsert = True

    resolved = await resolve_default_model(storage, settings, "alice")

    assert resolved["mode"] == MODE_MISSING
    assert resolved["chat_model_config"] is None
    # 只带异常**类名**：存储层异常里可能含连接串（用户名/口令）。
    assert "RuntimeError" in resolved["hint"]
    assert "模拟存储故障" not in resolved["hint"], "把异常正文带进了对外响应"


# ==============================================================================
# 三、端到端（走 HTTP）：端点形态与「前端那条线」
# ==============================================================================
async def test_default_model_endpoint_returns_a_usable_mock_config(
    client: httpx.AsyncClient,
) -> None:
    """``GET /api/v1/default-model`` 的响应形态。"""
    response = await client.get("/api/v1/default-model", headers=FRESH_HEADERS)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["mode"] == MODE_MOCK, body
    config = body["chat_model_config"]
    assert set(config) == {"type", "credential_id", "model", "parameters"}, (
        f"配置的字段集合变了：{sorted(config)}。"
        "多出来的字段很可能是密钥 —— 它会进入日志与浏览器缓存。"
    )
    assert config["type"] == MOCK_CREDENTIAL_TYPE
    assert config["model"] == MOCK_MODEL_NAME
    assert config["parameters"] == {}
    assert body["trace_id"], "响应里没有 trace_id，排障时会断链"


async def test_default_model_response_carries_no_secret(
    client: httpx.AsyncClient,
) -> None:
    """响应体里不得出现任何像密钥的东西。"""
    response = await client.get("/api/v1/default-model", headers=FRESH_HEADERS)
    text = response.text
    for needle in ("api_key", "sk-", "password", "secret", "Bearer "):
        assert needle not in text, f"响应里出现了 {needle!r}：{text[:400]}"


async def test_default_model_requires_identity(client: httpx.AsyncClient) -> None:
    """不带身份 ⇒ 401（由鉴权中间件拦下，而不是 422/404）。"""
    response = await client.get("/api/v1/default-model")
    assert response.status_code == 401, (
        f"期望 401（鉴权中间件），实际 {response.status_code}：{response.text[:200]}"
    )


async def test_credential_listing_is_seeded_before_the_first_read(
    client: httpx.AsyncClient,
) -> None:
    """★ 浏览器那条线：**第一个**请求是 ``GET /credential/`` 时也必须已经播种。

    这条用例挡的是「只测端点路径」的盲区。浏览器的请求顺序是
    ``/credential/`` + ``/model/``（拼选择器）、``/agent/``、``/sessions/``
    —— 它**从不**调用我们那个默认模型端点。播种因此必须发生在中间件里，
    而不是「端点被调用时顺手播一下」。

    若把中间件摘掉（或它的顺序挪到 Auth 之外），本用例会红，
    而其它所有用例照样绿 —— 那正是这个缺陷最难被发现的地方。
    """
    listing = await client.get("/credential/", headers=FRESH_HEADERS)
    assert listing.status_code == 200, listing.text
    credentials = listing.json().get("credentials", [])

    seeded = [
        item
        for item in credentials
        if (item.get("data") or {}).get("type") == MOCK_CREDENTIAL_TYPE
    ]
    assert seeded, (
        f"凭据列表是空的（{credentials}）—— 前端此刻的发送按钮是灰的。"
        "播种中间件没有生效，或者它的顺序不在 Auth 之内（读不到身份）。"
    )
    assert seeded[0]["id"] == mock_credential_id("test-fresh-user")


async def test_available_model_group_is_not_empty(
    client: httpx.AsyncClient,
) -> None:
    """★ 前端「可用模型」分组非空 —— 发送按钮可点的**充要条件**。

    前端把这一个分组拼成：``GET /credential/``（拿到凭据）×
    ``GET /model/?provider=<type>``（拿到模型卡片）。
    分组里元素的 ``models`` 为空数组时，``getFirstAvailableModel()``
    仍然返回 null（``ChatViewport.tsx``），因此两张表都必须非空。
    """
    credential_id = mock_credential_id("test-fresh-user")
    await client.get("/credential/", headers=FRESH_HEADERS)  # 触发播种

    models = await client.get(
        "/model/",
        params={"provider": MOCK_CREDENTIAL_TYPE},
        headers=FRESH_HEADERS,
    )
    assert models.status_code == 200, models.text
    cards = models.json().get("models", [])
    assert [c["name"] for c in cards] == [MOCK_MODEL_NAME], (
        f"{MOCK_CREDENTIAL_TYPE} 的模型列表是 {cards} —— "
        "前端会得到一个「有凭据但没有模型」的分组，按钮依然是灰的。"
    )
    assert credential_id.startswith("aligo-mock-")


async def test_seeded_credentials_are_isolated_between_users(
    client: httpx.AsyncClient,
) -> None:
    """播种不能破坏租户隔离：每个用户只看得到自己那条。

    两个身份各自第一次请求都会触发播种，而二者的凭据 id 由 user_id 派生 ——
    若派生写错（比如用了常量 id），第二个用户的播种会去更新**第一个用户**的
    记录，隔离就此破掉，症状是「A 的会话用的是 B 的模型配置」。
    """
    await client.get("/credential/", headers=FRESH_HEADERS)
    await client.get("/credential/", headers=OTHER_HEADERS)

    mine = (await client.get("/credential/", headers=FRESH_HEADERS)).json()
    other = (await client.get("/credential/", headers=OTHER_HEADERS)).json()

    my_ids = {item["id"] for item in mine.get("credentials", [])}
    other_ids = {item["id"] for item in other.get("credentials", [])}

    assert my_ids and other_ids, (my_ids, other_ids)
    assert my_ids.isdisjoint(other_ids), (
        f"两个用户的凭据 id 有交集：{my_ids & other_ids} —— 租户隔离破了"
    )


@pytest.mark.parametrize("headers", [FRESH_HEADERS, OTHER_HEADERS])
async def test_default_model_endpoint_is_stable_across_repeated_calls(
    client: httpx.AsyncClient,
    headers: dict[str, str],
) -> None:
    """反复调用返回同一份配置（幂等的对外表现）。"""
    first = (await client.get("/api/v1/default-model", headers=headers)).json()
    second = (await client.get("/api/v1/default-model", headers=headers)).json()
    assert first["chat_model_config"] == second["chat_model_config"]


def test_settings_are_reachable_from_app_state(app: FastAPI) -> None:
    """端点依赖的装配契约：``app.state.settings`` 必须存在。

    它由 ``src/server/app.py`` 挂上（框架不会替我们写）。少了它，
    端点的 503 分支会被走到 —— 那本身是设计好的降级，但部署时
    所有人都会拿到 503，而原因（忘了 setattr）藏得很深。
    """
    assert getattr(app.state, "settings", None) is not None


# ==============================================================================
# 四、系统凭据共享：「配了真密钥的部署」也必须能对话
# ==============================================================================
# 这是同一种病（探针全绿、发不出消息）的**另一半**。零密钥那半由 Mock 播种
# 解决（第一~三节），有密钥这半不能靠播种 —— 那条路会把运营者的 key 复制进
# 用户自己拥有的记录里，而框架对 owner 读凭据是**明文**返回。
#
# 解法是框架的 ResourceAccessPolicyBase：凭据只有一条（属主 aligo-system），
# 策略把它以 READ 权限共享出去，框架随后让用户**能用但读不到**
# （列表打码成 {type, name}，运行期解析返回原始记录）。
#
# 本节钉住这条链路的**每一环**，因为它们各自的失效方式都极其安静：
#   · 判据写错        ⇒ 该共享时不共享（症状 = 运营者以为自己配了密钥）
#   · 策略返回 EDIT   ⇒ 任何用户都能删掉全公司的密钥（安全问题，不是功能问题）
#   · 盒子不是同一个  ⇒ 策略永远读不到 seeded=True（共享静默失效）
#   · 端点不接策略    ⇒ 共享其实生效了，但 smoke / 脚本拿不到配置
# ------------------------------------------------------------------------------
def _keyed(settings: Settings, *, provider: str = "dashscope") -> Settings:
    """把测试配置改造成「配了真实密钥」的那一档。

    ⚠️ ``use_mock_when_no_key`` 保持 True 不动：这只影响「没有密钥时怎么办」，
    而本函数恰恰给了密钥，因此 ``should_use_mock`` 为 False。
    若哪天有人把它一起改成 False，``build_credential`` 在别的路径上会因为
    「配置自相矛盾」抛错 —— 那是**另一条**用例该管的事，这里不去碰它。

    Args:
        settings (`Settings`): 零密钥的测试配置。
        provider (`str`): ``dashscope`` 或 ``openai``。

    Returns:
        `Settings`: 独立副本（改它不会污染 fixture）。
    """
    keyed = settings.model_copy(deep=True)
    keyed.llm.provider = provider  # type: ignore[assignment]
    keyed.llm.api_key = FAKE_API_KEY
    return keyed


@pytest.mark.parametrize(
    ("api_key", "mock_flag", "expected"),
    [
        # 零密钥 + 降级开着 ⇒ 由 Mock 那条线负责，不共享（没有东西可共享）。
        ("", True, False),
        # 零密钥 + 降级关着 ⇒ 配置自相矛盾，共享不该跟着一起炸。
        ("", False, False),
        # 有密钥 + 降级开着 ⇒ use_mock_when_no_key 对「有密钥」不生效 ⇒ 共享。
        (FAKE_API_KEY, True, True),
        # 有密钥 + 降级关着 ⇒ 同样的结论。
        (FAKE_API_KEY, False, True),
    ],
)
def test_sharing_enabled_truth_table(
    settings: Settings,
    api_key: str,
    mock_flag: bool,
    expected: bool,
) -> None:
    """``sharing_enabled`` 必须同时看两个开关，缺一不可。

    只看 ``has_api_key`` 的话，``use_mock_when_no_key=false`` + 空密钥
    这一档会被判成「可共享」，于是启动时去 ``build_credential`` ——
    而那会抛 ValueError（配置自相矛盾）。共享是可用性增强，
    不该把一次**配置错误**变成启动期的一串噪声。
    """
    probe = settings.model_copy(deep=True)
    probe.llm.api_key = api_key
    probe.llm.use_mock_when_no_key = mock_flag
    assert sharing_enabled(probe) is expected


@pytest.mark.parametrize(
    ("provider", "expected_type"),
    [("dashscope", "dashscope_credential"), ("openai", "openai_credential")],
)
def test_system_credential_type_matches_the_provider(
    settings: Settings,
    provider: str,
    expected_type: str,
) -> None:
    """凭据类型必须与 provider **逐字**一致。

    这个字符串会原样进入会话的 ``chat_model_config.type``，由
    ``CredentialFactory`` 反查模型类。写错一个字（或写成一个自造的名字）
    的症状是：凭据在列表里看得见（打码后只剩 type/name，用户看不出真假），
    但一发起对话就 400/422 —— 一个「列表正常、对话报错」的疑难杂症。

    ⚠️ 断言的是字面量而不是 ``DashScopeCredential.model_fields[...]``：
    后者是把被测代码抄一遍，两边同时写错时用例照样绿。
    """
    assert system_credential_type(_keyed(settings, provider=provider)) == expected_type


def test_system_chat_model_config_has_exactly_four_keys_and_no_secret(
    settings: Settings,
) -> None:
    """共享配置的字段集合必须与 Mock 那条**完全一致**，且不含任何密钥。

    四字段是 ``ChatModelConfig`` 的硬要求（少一个 422）；而「不多一个字
    密钥字段」是本模块最重要的安全性质 —— 这份字典会出现在 HTTP 响应里。
    """
    keyed = _keyed(settings)
    config = system_chat_model_config(keyed)

    assert set(config) == {"type", "credential_id", "model", "parameters"}, (
        f"字段集合变了：{sorted(config)}。多出来的很可能是密钥。"
    )
    assert config["credential_id"] == SYSTEM_CREDENTIAL_ID
    assert config["model"] == keyed.llm.model
    assert config["parameters"] == {}

    rendered = repr(config)
    assert FAKE_API_KEY not in rendered, f"配置里带上了密钥：{rendered}"
    assert "api_key" not in rendered and "base_url" not in rendered, rendered


async def test_policy_lists_nothing_until_seeded(settings: Settings) -> None:
    """★ 播种没成功时策略必须返回空 —— 否则「列表里看得见、点开 404」。

    ``seeded`` 是「那条凭据**真的**写进去了」的记号。不认它的策略会在
    数据库抖动、凭据写失败时照样把 ref 交出去，于是框架去取那条并不存在的
    记录 ⇒ ``resolve_credential`` 抛 404。用户看到的是「凭据列表里明明有，
    一选就报错」。
    """
    from agentscope.app.access import ResourceKind

    box: dict[str, Any] = {"seeded": False}
    policy = SystemCredentialAccessPolicy(state=box)

    assert await policy.list_accessible("alice", ResourceKind.CREDENTIAL, None) == []

    box["seeded"] = True
    refs = await policy.list_accessible("alice", ResourceKind.CREDENTIAL, None)
    assert len(refs) == 1, refs


@pytest.mark.parametrize("kind_name", ["AGENT", "KNOWLEDGE_BASE"])
async def test_policy_never_shares_agents_or_knowledge_bases(
    settings: Settings,
    kind_name: str,
) -> None:
    """★ 策略只共享凭据，**绝不**共享智能体与知识库。

    这是多租户系统里最容易失控的一类改动：「顺手多共享一点」。
    智能体与知识库是**业务数据**，跨属主可见等于跨租户读用户内容 ——
    语义上完全是另一件事，而且本项目没有任何「共享」的产品需求。

    门是关着的还不行，得证明它**锁着**：因此这里在 ``seeded=True``
    （也就是共享最活跃的状态）下断言另外两种 kind 返回空。
    """
    from agentscope.app.access import ResourceKind

    policy = SystemCredentialAccessPolicy(state={"seeded": True})
    kind = getattr(ResourceKind, kind_name)

    assert await policy.list_accessible("alice", kind, None) == []


async def test_policy_grants_read_only_and_skips_the_owner() -> None:
    """★ 权限必须是 ``READ``（不是 ``EDIT``），且不列属主自己那一份。

    ``EDIT`` 意味着任何一个通过鉴权的用户都能改掉、或**删掉**全公司共用的
    那条凭据。这不是「共享」，是「开放的写权限」—— 而且它**看起来**
    完全正常：共享生效了、用户能用上模型了，直到有一天密钥不见了。

    属主自己那一份则走 framework 的 owner 路径（``get_credential`` 直接命中），
    策略里再列一遍只会多一次查询。
    """
    from agentscope.app.access import ResourceKind, ResourcePermission

    policy = SystemCredentialAccessPolicy(state={"seeded": True})

    refs = await policy.list_accessible("alice", ResourceKind.CREDENTIAL, None)
    assert [r.permission for r in refs] == [ResourcePermission.READ], (
        "共享权限不是 READ —— 用户将能修改/删除全公司共用的凭据。"
    )
    assert refs[0].owner_id == SYSTEM_USER_ID
    assert refs[0].resource_id == SYSTEM_CREDENTIAL_ID

    owner_refs = await policy.list_accessible(SYSTEM_USER_ID, ResourceKind.CREDENTIAL, None)
    assert owner_refs == [], "属主不该在策略里再被列一遍"


def test_build_access_policy_follows_the_sharing_switch(settings: Settings) -> None:
    """没密钥 ⇒ 不注册策略（框架回落到 deny-all）。

    ``None`` 不是「随便给个空策略」，而是**不传**——
    ``create_app`` 会据此构造框架默认的 ``DenyAllResourceAccessPolicy``。
    返回一个「总是返回空列表」的自定义策略看起来等价，但它会把
    「本项目没有共享需求」这件事伪装成「我们有共享，只是当下没人够格」，
    将来读代码的人会去找那条永远不成立的规则。
    """
    assert build_access_policy(settings, {"seeded": False}) is None
    assert build_access_policy(_keyed(settings), {"seeded": False}) is not None


async def test_ensure_system_credential_writes_one_fixed_record(
    settings: Settings,
) -> None:
    """播种必须落在**固定 id、固定属主**上（幂等的前提）。

    id 随机的话「改了密钥重启一次」就会变成「凭据列表里多一条」，
    而且越积越多；属主不是 ``aligo-system`` 的话，这条凭据会被框架
    当成某个真实用户的私有凭据，策略里那条跨属主规则就永远命不中。
    """
    storage = _RecordingStorage()

    assert await ensure_system_credential(storage, settings) is None, (
        "零密钥部署不该写出任何系统凭据"
    )
    assert storage.upsert_calls == []

    returned = await ensure_system_credential(storage, _keyed(settings))
    assert returned == SYSTEM_CREDENTIAL_ID
    assert len(storage.upsert_calls) == 1
    owner, credential = storage.upsert_calls[0]
    assert owner == SYSTEM_USER_ID
    assert credential.id == SYSTEM_CREDENTIAL_ID
    assert credential.name == SYSTEM_CREDENTIAL_NAME
    # 同一份配置反复播种 ⇒ 仍然是同一条（就地更新）。
    await ensure_system_credential(storage, _keyed(settings))
    assert storage.upserted == [SYSTEM_CREDENTIAL_ID] * 2


async def test_a_concurrent_first_seed_is_not_a_failure(
    settings: Settings,
) -> None:
    """★ 两个副本同时首次播种 ⇒ 后到的那个必须**正常返回**。

    形状：两边都查不到那条记录 ⇒ 都走直接 ``INSERT`` ⇒ 后者撞全局主键。
    框架的语义没错（那个 INSERT 是防跨租户覆盖的），但在这里它是
    「另一个执行者刚好把同一件事做完了」，不是故障。

    ⚠️ 若把它当故障：``_seed_system_credential`` 会把它记成
    ``logger.exception`` 并以「没播种成功」收场（``seeded`` 保持 False），
    于是**策略对所有人返回空** —— 表现是「运营者配了密钥、所有人却没有
    模型可用」，与「根本没配密钥」一模一样，而日志里只有一条被当成
    噪声的主键冲突。
    """
    storage = _RecordingStorage()
    storage.conflict_upsert = 1

    returned = await ensure_system_credential(storage, _keyed(settings))

    assert returned == SYSTEM_CREDENTIAL_ID
    assert storage.upserted == [SYSTEM_CREDENTIAL_ID], "重试那一次没有写成功"


async def test_resolve_shared_model_requires_a_matching_ref(
    settings: Settings,
) -> None:
    """★ 只有 ref **确实指向那条系统凭据**时才给配置。

    ``list_accessible`` 的契约是「列出所有可跨属主访问的凭据」，
    今天只会返回我们那一条；但将来若有人往策略里加了第二条共享，
    「列表非空 ⇒ 用系统默认模型」就会把用户静默地切到另一种凭据上 ——
    模型名对不上、账单记错地方，而日志里什么都看不出来。
    """
    from agentscope.app.access import ResourceKind, ResourcePermission, ResourceRef

    keyed = _keyed(settings)
    storage = _RecordingStorage()

    class _StubPolicy:
        """只回放预设 refs 的策略替身（本用例只关心调用方的判据）。"""

        def __init__(self, refs: list[Any]) -> None:
            self.refs = refs

        async def list_accessible(
            self,
            viewer_id: str,
            kind: Any,
            storage_: Any,
        ) -> list[Any]:
            """返回预设的 refs。"""
            return list(self.refs)

    # 没有策略 ⇒ 不共享（这正是框架 deny-all 的语义）。
    assert await resolve_shared_model(storage, keyed, "alice", None) is None

    # 空列表 ⇒ 不共享。
    assert await resolve_shared_model(storage, keyed, "alice", _StubPolicy([])) is None

    # 指向**别的**凭据 ⇒ 不共享（本用例存在的全部理由）。
    other = ResourceRef(
        kind=ResourceKind.CREDENTIAL,
        owner_id="someone-else",
        resource_id=SYSTEM_CREDENTIAL_ID,
        permission=ResourcePermission.READ,
    )
    assert await resolve_shared_model(storage, keyed, "alice", _StubPolicy([other])) is None

    # 指向那条系统凭据 ⇒ 给出配置。
    exact = ResourceRef(
        kind=ResourceKind.CREDENTIAL,
        owner_id=SYSTEM_USER_ID,
        resource_id=SYSTEM_CREDENTIAL_ID,
        permission=ResourcePermission.READ,
    )
    shared = await resolve_shared_model(storage, keyed, "alice", _StubPolicy([exact]))
    assert shared == system_chat_model_config(keyed)


async def test_own_credentials_win_over_the_shared_one(settings: Settings) -> None:
    """★ 用户自己配了凭据时，共享必须是**备选**而不是默认。

    反过来的话，一个自带了公司密钥的用户会被悄悄切到系统凭据上 ——
    账单与配额都算到了错误的项目上，而用户以为自己用的还是自己那条。
    这一条同时钉住了判定的**顺序**（见 resolve_default_model 的 docstring）。

    ⚠️ 策略在这里恒为真：若实现里把共享分支放到了自己的凭据之前，
    本用例会红，而其它所有用例照样绿。
    """
    from agentscope.app.access import ResourceKind, ResourcePermission, ResourceRef

    keyed = _keyed(settings)
    storage = _RecordingStorage(credentials=[object()])

    class _AlwaysShared:
        """永远说「可以共享」的策略替身。"""

        async def list_accessible(
            self,
            viewer_id: str,
            kind: Any,
            storage_: Any,
        ) -> list[Any]:
            """恒返回那条系统凭据的 ref。"""
            return [
                ResourceRef(
                    kind=ResourceKind.CREDENTIAL,
                    owner_id=SYSTEM_USER_ID,
                    resource_id=SYSTEM_CREDENTIAL_ID,
                    permission=ResourcePermission.READ,
                ),
            ]

    resolved = await resolve_default_model(
        storage,
        keyed,
        "alice",
        policy=_AlwaysShared(),
    )

    assert resolved["mode"] == MODE_CONFIGURED, resolved
    assert resolved["chat_model_config"] is None, (
        "用户已有自己的凭据，却被塞了一份系统默认配置 —— 选择被覆盖了"
    )


async def test_resolve_default_model_reports_shared_mode(settings: Settings) -> None:
    """没有自己的凭据 + 策略给了一条 ref ⇒ ``shared`` + 可用配置。"""
    from agentscope.app.access import ResourceKind, ResourcePermission, ResourceRef

    keyed = _keyed(settings)
    storage = _RecordingStorage()

    class _Shared:
        """只回放那条系统凭据 ref 的策略替身。"""

        async def list_accessible(
            self,
            viewer_id: str,
            kind: Any,
            storage_: Any,
        ) -> list[Any]:
            """返回系统凭据的 ref。"""
            return [
                ResourceRef(
                    kind=ResourceKind.CREDENTIAL,
                    owner_id=SYSTEM_USER_ID,
                    resource_id=SYSTEM_CREDENTIAL_ID,
                    permission=ResourcePermission.READ,
                ),
            ]

    resolved = await resolve_default_model(storage, keyed, "alice", policy=_Shared())

    assert resolved["mode"] == MODE_SHARED, resolved
    assert resolved["chat_model_config"] == system_chat_model_config(keyed)
    assert resolved["hint"]
    # 共享路径**不**播种任何 Mock 凭据（那是零密钥那条线的行为）。
    assert storage.upserted == []


async def test_policy_failure_degrades_instead_of_returning_500(
    settings: Settings,
) -> None:
    """策略抛异常时端点必须给出可解释的答复，而不是 500。

    策略是本项目自己的代码，抛异常说明有 bug。让它冒到 HTTP 层的后果是
    ``/api/v1/default-model`` 变成 500 —— 排障的人会去查**这个端点**，
    方向完全错了。而真正的故障（对话链路也走策略）照样会在对话时报出来，
    所以吞掉它并不会掩盖任何事情。
    """
    keyed = _keyed(settings)
    storage = _RecordingStorage()

    class _Broken:
        """恒抛异常的策略替身。"""

        async def list_accessible(
            self,
            viewer_id: str,
            kind: Any,
            storage_: Any,
        ) -> list[Any]:
            """抛出一个带**敏感正文**的异常，验证正文不会外泄。"""
            raise RuntimeError(f"数据库连接串 redis://:{FAKE_API_KEY}@host/0")

    resolved = await resolve_default_model(storage, keyed, "alice", policy=_Broken())

    assert resolved["mode"] == MODE_MISSING
    assert resolved["chat_model_config"] is None
    assert FAKE_API_KEY not in repr(resolved), (
        f"策略异常的正文泄漏进了对外响应：{resolved}"
    )


# ------------------------------------------------------------------------------
# 五、端到端（真装配、真 lifespan）：共享凭据在**框架自己的**接口上看得见
# ------------------------------------------------------------------------------
# 上面那些用例测的是我们的判据；这一节测的是**框架的行为**是否符合我们的预期 ——
# 具体说，就是「只读共享」这条安全性质的兑现方式：
#   · 列表里看得见（否则前端分组为空、按钮还是灰的）；
#   · data 被打码成 {type, name}（否则全公司的密钥明文躺在任何用户的
#     /credential/ 响应里 —— 这是本设计要避免的**唯一**一件事）。
#
# ⚠️ 这一节必须用**真的** /credential/ 与 /model/，不能再用替身：
# 「打码」是框架 `_build_view` 的行为，我们的任何替身都不会复现它。
# ------------------------------------------------------------------------------
@pytest_asyncio.fixture
async def keyed_client(
    settings: Settings,
    tmp_path: Path,
) -> AsyncIterator[AsyncClient]:
    """一个**配了（假）密钥**、已进入 lifespan 的 HTTP 客户端。

    与 conftest 的 ``client`` 夹具的区别只有一个：``llm.api_key`` 非空。
    于是 ``should_use_mock`` 为 False → 装配期注册系统凭据共享策略 →
    lifespan 里播种那条凭据。

    ⚠️ 密钥用一个**无效**的值（见 FAKE_API_KEY）。若哪天这个夹具真的
    发起了模型调用，用例会以认证失败告终 —— 那是**期望**的行为：
    宁可用例红，也不要悄悄打到真实供应商上。

    Args:
        settings (`Settings`): 零密钥的测试配置。
        tmp_path (`Path`): 临时目录（工作区与 blob 的根）。

    Yields:
        `AsyncClient`: 绑定到该应用的 HTTP 客户端。
    """
    from agentscope.app.message_bus import InMemoryMessageBus
    from agentscope.app.rag.blob_store import LocalBlobStore
    from agentscope.app.storage._sql import AsyncSQLAlchemyStorage
    from agentscope.app.workspace_manager import LocalWorkspaceManager

    from src.server.app import _storage_engine_kwargs, create_root_app

    keyed = _keyed(settings)
    app = create_root_app(
        keyed,
        storage=AsyncSQLAlchemyStorage(
            url=keyed.db.url,
            create_tables=keyed.db.create_tables,
            auto_migrate=False,
            engine_kwargs=_storage_engine_kwargs(keyed),
        ),
        message_bus=InMemoryMessageBus(),
        workspace_manager=LocalWorkspaceManager(basedir=str(tmp_path / "workspace")),
        blob_store=LocalBlobStore(root_dir=str(tmp_path / "blobs")),
        enable_scheduler=False,
    )

    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http:
            yield http


async def test_shared_credential_is_visible_but_masked(
    keyed_client: AsyncClient,
) -> None:
    """★ 核心安全性质：共享凭据**看得见，但读不到密钥**。

    一次断言同时钉住两件相反的事：

        · 列表里必须**有**它（否则前端「可用模型」分组为空、发送按钮是灰的
          —— 这正是整条修复要解决的问题）；
        · ``data`` 里必须**没有**密钥（否则任何通过鉴权的用户都能在
          「凭据」页把全公司共用的密钥抄走）。

    只断言其中一条都会漏掉一半的失败模式：只测「有」，打码破掉时用例照样绿；
    只测「没有」，共享根本没生效时用例也照样绿。
    """
    response = await keyed_client.get("/credential/", headers=FRESH_HEADERS)
    assert response.status_code == 200, response.text
    credentials = response.json().get("credentials", [])

    shared = [item for item in credentials if item.get("id") == SYSTEM_CREDENTIAL_ID]
    assert len(shared) == 1, (
        f"共享凭据没有出现在 /credential/ 里（{credentials}）——"
        "前端此刻的发送按钮是灰的。检查 lifespan 里那次播种是否成功、"
        "以及策略的 seeded 盒子是不是同一个对象。"
    )

    data = shared[0].get("data") or {}
    assert set(data) == {"type", "name"}, (
        f"共享凭据的 data 没有被打码成 {{type, name}}，实际 {sorted(data)} —— "
        "非属主的凭据内容泄漏了。"
    )
    assert data["name"] == SYSTEM_CREDENTIAL_NAME

    # 用整个响应体（而不是只看 data 字段）再堵一次：任何位置出现密钥都不行。
    assert FAKE_API_KEY not in response.text, "响应体里出现了密钥明文"
    assert "api_key" not in response.text, response.text[:400]


async def test_the_shared_credential_cannot_be_reached_by_claiming_its_owner(
    keyed_client: AsyncClient,
) -> None:
    """★ 冒充系统属主必须被挡住 —— 这条曾经是**实测可复现**的越权。

    修复前（真实装配的应用上跑出来的原始输出）::

        X-User-ID: alice         → GET /credential/  data 打码成 {type,name}
        X-User-ID: aligo-system  → GET /credential/  data 里带 api_key 明文，
                                    且 editable: true
        X-User-ID: aligo-system  → DELETE /credential/aligo-system-model → 204

    也就是说，一次请求就能拿走全公司共用的密钥，或者删掉那条凭据
    让所有用户的模型当场失效。而它的成因不是「漏了鉴权」——
    是**内部身份被当成了可声明的身份**（完整论证见
    ``src/llm/identity.py`` 与 ``src/server/middleware/auth.py``）。

    ⚠️ 这里断言的是**中间件层的 403**，而不是「框架打码正确」：
    打码是框架的既有行为、由上面那条用例守着；本用例守的是
    「那个身份根本进不来」。两者是纵深防御的两层，缺一层都不算修好。
    """
    forged = {"X-User-ID": SYSTEM_USER_ID}

    listing = await keyed_client.get("/credential/", headers=forged)
    assert listing.status_code == 403, (
        f"冒充系统属主去列表竟然拿到了 {listing.status_code}：{listing.text[:300]}"
    )

    single = await keyed_client.get(
        f"/credential/{SYSTEM_CREDENTIAL_ID}",
        headers=forged,
    )
    assert single.status_code == 403, single.text[:300]

    deleted = await keyed_client.delete(
        f"/credential/{SYSTEM_CREDENTIAL_ID}",
        headers=forged,
    )
    assert deleted.status_code == 403, deleted.text[:300]

    # 最直接的证据：那条凭据必须**还在**。上面的状态码断言保证「没做」，
    # 这一条保证「确实没做」—— 若将来有人把破坏性操作改成「先做后报 403」
    # （例如中间件顺序颠倒），只有这一条能发现。
    still_there = await keyed_client.get("/credential/", headers=FRESH_HEADERS)
    assert [item["id"] for item in still_there.json()["credentials"]] == [
        SYSTEM_CREDENTIAL_ID,
    ], "共享凭据不见了 —— 冒充属主的 DELETE 生效了"


async def test_shared_credential_fills_the_frontend_model_group(
    keyed_client: AsyncClient,
) -> None:
    """★ 前端拼下拉框的两个接口都必须给出内容。

    ``useAvailableModels`` 的取值路径是
    ``GET /credential/``（拿到 type）× ``GET /model/?provider=<type>``
    （拿到卡片），两张表**都**要非空 —— 分组里的 ``models`` 是空数组时
    ``getFirstAvailableModel()`` 仍然返回 null，按钮依然是灰的。
    """
    listing = await keyed_client.get("/credential/", headers=FRESH_HEADERS)
    shared = [
        item
        for item in listing.json().get("credentials", [])
        if item.get("id") == SYSTEM_CREDENTIAL_ID
    ]
    assert shared, listing.text
    provider = shared[0]["data"]["type"]

    models = await keyed_client.get(
        "/model/",
        params={"provider": provider},
        headers=FRESH_HEADERS,
    )
    assert models.status_code == 200, models.text
    cards = models.json().get("models", [])
    assert cards, (
        f"{provider} 的模型卡片是空的 —— 前端会得到一个「有凭据、没有模型」"
        "的分组，发送按钮依然是灰的。"
    )


async def test_default_model_reports_shared_on_a_keyed_deployment(
    keyed_client: AsyncClient,
) -> None:
    """★ 配了密钥的部署上，默认模型端点不再返回 ``missing``。

    这是本节的**验收断言**：部署形态从「探针全绿、一个字发不出去」
    变成「拿到一份可以直接用的配置」。``scripts/smoke.py`` 的
    ``check_conversation`` 走的正是这条路。
    """
    response = await keyed_client.get("/api/v1/default-model", headers=FRESH_HEADERS)
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["mode"] == MODE_SHARED, body
    config = body["chat_model_config"]
    assert set(config) == {"type", "credential_id", "model", "parameters"}, config
    assert config["credential_id"] == SYSTEM_CREDENTIAL_ID
    assert config["type"] == "dashscope_credential"
    assert FAKE_API_KEY not in response.text, "响应体里出现了密钥明文"
