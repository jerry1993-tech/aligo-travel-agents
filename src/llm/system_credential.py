# -*- coding: utf-8 -*-
"""全局模型凭据共享：让「配好密钥的部署」也开箱即用。

文件职责：
    当运营者在 ``.env`` 里配了真实密钥时，把它**作为一条系统所有的凭据**
    供给全体已鉴权用户使用，使浏览器端的「可用模型」不再为空、
    会话能配上 ``chat_model_config``。零密钥时本模块整体不生效
    （那条路径由 ``src/llm/degradation.py`` 的 Mock 降级负责）。

上下游依赖：
    - 上游：``src/llm/factory.py::build_credential``（按 provider 构造凭据对象）、
      ``has_api_key`` / ``should_use_mock``（判据）。
    - 下游：``src/server/app.py``（装配期构造策略、lifespan 里播种）。
    - 框架侧：``agentscope.app.access.ResourceAccessPolicyBase`` —— 本模块
      实现它，把「谁可以看见/使用哪条凭据」这件事交给框架既有的鉴权链路。

==============================================================================
问题的形状：密钥配了，服务还是不能用
==============================================================================
    框架的对话链路要求 ``chat_model_config.credential_id`` 指向一条
    **调用者可解析**的凭据记录。而框架默认的 ``DenyAllResourceAccessPolicy``
    是严格的 owner 隔离：用户只看得见自己的记录，别人的（包括系统的）
    一律不可见。于是出现一个反直觉的部署形态 ——

        ``.env`` 里配好了密钥、``/readyz`` 绿、模型调用**完全正常**，
        但浏览器里「可用模型」是空的 ⇒ 发送按钮是灰的；
        curl 直接 ``POST /chat/`` 也只会拿到
        ``No model configuration found for agent ...``。

    换句话说：**「服务端有能力调模型」与「用户能用上模型」是两件事**。

==============================================================================
为什么用「共享一条系统凭据」而不是「往每个用户手里塞一份 key」
==============================================================================
    后者（把运营者的 key 复制进用户自己的记录）有一个致命的副作用：
    owner 读自己的凭据是**明文**。于是任何能通过鉴权的用户都能在
    「凭据」页把自己那份 key 抄走 —— 一次共享，全员泄密。

    框架为此留了正规的扩展点：``ResourceAccessPolicyBase``。
    策略返回 ``ResourceRef(permission=READ)``，框架随后：

        · ``list_resource``  —— 列表里带上这条记录，但**打码**成
          ``{type, name}``（见 ``app/_service/_access.py::_build_view``）；
        · ``resolve_credential`` —— 运行期解析时返回**原始记录**，
          供 ``ChatModelBase`` 构造使用（docstring 原文：
          "Credentials remain masked in API responses and can only be
          used by runtime code."）。

    也就是说：**用户能用，但读不到**。这正是共享密钥应有的语义，
    而且整套鉴权/打码逻辑都是框架既有代码，我们只提供「谁能用」这一条规则。

------------------------------------------------------------------------------
本模块只共享凭据，**不**共享智能体与知识库
------------------------------------------------------------------------------
    ``ResourceKind`` 有三个取值，这里对其余两个恒定返回空列表。
    理由：智能体与知识库是**业务数据**，共享它们等于跨租户读写用户内容，
    语义上完全是另一件事（而且本项目还没有任何「共享」的产品需求）。
    在多租户系统里，「顺手多共享一点」是最容易失控的一类改动。

------------------------------------------------------------------------------
开关
------------------------------------------------------------------------------
    共享**只在「配了真实密钥」时启用**（``sharing_enabled``）。
    没配密钥时既然没有东西可共享，策略干脆不注册，框架回到默认的
    deny-all —— 零密钥路径由 Mock 降级那条线负责，两条线不重叠：

        should_use_mock=True   ⇒ Mock 凭据（每人一条，互不可见）
        should_use_mock=False  ⇒ 共享的系统凭据（一条，全员可用）
"""

from __future__ import annotations

from typing import Any

from agentscope.app.access import (
    ResourceAccessPolicyBase,
    ResourceKind,
    ResourcePermission,
    ResourceRef,
)

from .factory import (
    PROVIDER_REGISTRY,
    build_credential,
    has_api_key,
    resolve_provider,
    should_use_mock,
)
# SYSTEM_USER_ID 的**定义**搬到了 ``src/llm/identity.py``（一个零 import 的
# 叶子模块），因为鉴权中间件也要读它，而中间件不该为了一个字符串常量
# 把 ``agentscope.app.access`` 拉进自己的 import 期 —— 完整理由写在那里。
# 这里 re-export 是为了不让调用方（``src/server/app.py``、单测）被迫改 import
# 路径：对它们而言「系统凭据的属主」仍然读作 ``system_credential.SYSTEM_USER_ID``。
from .identity import SYSTEM_USER_ID
from .preset_credential import upsert_preset_credential

#: 系统凭据的 id。**固定值**（不是随机生成）：策略要在装配期就知道该引用谁，
#: 而装配期还没有任何记录。固定 id + ``upsert`` 的幂等语义 ⇒ 每次启动
#: 都是同一条记录（就地更新密钥），不会越长越多。
SYSTEM_CREDENTIAL_ID = "aligo-system-model"

#: 系统凭据在「凭据」页的显示名。写明「系统」二字，用户才知道
#: 这条不是自己创建的、也不能编辑或删除。
SYSTEM_CREDENTIAL_NAME = "系统默认模型凭据（由管理员配置，全员可用）"


def sharing_enabled(settings: Any) -> bool:
    """判断是否应当启用「系统凭据共享」。

    判据与 :func:`~src.llm.degradation.resolve_default_model` 的三分支
    **互补且互斥**：降级生效时不共享（没有密钥可共享），
    降级不生效且**确实有密钥**时才共享。

    ⚠️ 必须同时判 ``should_use_mock`` 与 ``has_api_key``，只看后者不够：
    ``use_mock_when_no_key=false`` 且密钥为空时，``should_use_mock`` 也是
    False —— 那种配置本身自相矛盾（``build_credential`` 会抛 ValueError），
    此时去播种只会把一次配置错误变成启动期的一串噪声日志。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `bool`: 应当共享返回 True。
    """
    return not should_use_mock(settings) and has_api_key(settings)


def system_credential_type(settings: Any) -> str:
    """返回系统凭据在 ``CredentialFactory`` 里的类型标识。

    推导方式**不是**构造一份凭据再读 ``.type``：那会实例化一个
    ``DashScopeCredential(api_key=...)`` 对象，让真密钥多一份内存副本、
    也多一次「被谁不小心打进日志」的机会。这里只读配置与类字段，
    完全不碰密钥。

    Args:
        settings (`Settings`): 全量配置。

    Returns:
        `str`: 例如 ``"dashscope_credential"`` / ``"openai_credential"``。

    Raises:
        ValueError: provider 未在 :data:`PROVIDER_REGISTRY` 中登记
            （由 :func:`~src.llm.factory.resolve_provider` 抛出）。
    """
    credential_cls, _ = PROVIDER_REGISTRY[resolve_provider(settings)]
    # pydantic v2：默认值就是那个 Literal 的字面量。
    return str(credential_cls.model_fields["type"].default)


def system_chat_model_config(settings: Any) -> dict[str, Any]:
    """组装一份指向系统凭据的 ``ChatModelConfig``。

    ⚠️ 与 :func:`~src.llm.degradation.mock_chat_model_config` 一样，
    这里**只有** type / credential_id / model / parameters ——
    不含 api_key、base_url 或任何派生于密钥的字段，因此可以安全地
    出现在 HTTP 响应与日志里。

    ``model`` 取 ``settings.llm.model``：那是运营者配的默认模型，
    也正是 ``docs/`` 里承诺「配好密钥即可对话」所隐含的那一个。
    用户当然可以在前端选择器里换成别的（19 张 DashScope 卡片全部可选），
    本函数回答的只是「什么都不选时的默认」。

    Args:
        settings (`Settings`): 全量配置，需读 ``settings.llm``。

    Returns:
        `dict[str, Any]`: 可直接放进会话 ``chat_model_config`` 的字典。
    """
    return {
        "type": system_credential_type(settings),
        "credential_id": SYSTEM_CREDENTIAL_ID,
        "model": settings.llm.model,
        "parameters": {},
    }


async def resolve_shared_model(
    storage: Any,
    settings: Any,
    user_id: str,
    policy: Any | None,
) -> dict[str, Any] | None:
    """若系统凭据对 ``user_id`` 可用，返回一份可直接使用的配置。

    判定**完全走框架的策略链路**（``policy.list_accessible``），而不是
    「查一下那条记录在不在」：在不在是一回事，**能不能用**是另一回事，
    而后者才是本函数要回答的问题。直接查库还会绕过策略，导致
    「开机时没播种成功（数据库抖了一下）却仍然告诉用户可以用」——
    此时真正的表现是发起对话后 404，而端点的答复完全看不出来。

    Args:
        storage (`Any`): 框架的存储实现（转手给策略）。
        settings (`Settings`): 全量配置。
        user_id (`str`): 框架认定的用户标识。
        policy (`Any | None`): ``app.state.resource_access_policy``。
            ``None``（或框架默认的 deny-all）时本函数恒返回 ``None``。

    Returns:
        `dict[str, Any] | None`: 可用的 ``chat_model_config``；不可用时 ``None``。

    Raises:
        Exception: 策略抛出的异常**原样向上抛**。调用方
            :func:`~src.llm.degradation.resolve_default_model` 负责把它
            变成一句人话 —— 在本函数里吞掉的话，策略的 bug 就再也不会
            在日志里留下痕迹，而症状只是「这个功能时灵时不灵」。
    """
    if policy is None or not sharing_enabled(settings):
        return None

    refs = await policy.list_accessible(user_id, ResourceKind.CREDENTIAL, storage)
    if not refs:
        return None
    # ⚠️ 必须真的核对 ref 指向的是**我们那条**系统凭据。策略是本项目自己
    # 实现的，今天只会返回它；但 ``list_accessible`` 的契约是「列出所有
    # 可跨属主访问的凭据」，将来若有人往策略里加第二条共享，
    # 这一行就会阻止端点自作主张地把「第一条」当成默认模型。
    if not any(
        ref.resource_id == SYSTEM_CREDENTIAL_ID and ref.owner_id == SYSTEM_USER_ID
        for ref in refs
    ):
        return None
    return system_chat_model_config(settings)


async def ensure_system_credential(storage: Any, settings: Any) -> str | None:
    """把配置里的真实凭据写进 storage（属主为 :data:`SYSTEM_USER_ID`）。

    **幂等**：id 固定，``upsert_credential`` 对该 id 是就地更新 ——
    改了密钥重启一次就生效，不需要手工删旧记录。
    多副本同时启动时「首次写入」会撞主键，那由
    :func:`~src.llm.preset_credential.upsert_preset_credential` 消化
    （重试一次即成功，且不算失败）。

    Args:
        storage (`Any`): 框架的存储实现。
        settings (`Settings`): 全量配置。

    Returns:
        `str | None`: 写入成功时返回凭据 id；配置里没有可用密钥时返回 ``None``
        （调用方据此把共享策略关掉）。

    Raises:
        Exception: 存储写入失败时原样抛出（主键冲突除外，见上）。调用方
            （lifespan）**必须**兜住它 —— 一条共享凭据写不进去不该让整个
            服务起不来。⚠️ 框架的存储层异常可能带着 SQL 绑定参数，
            而这条 INSERT 的参数里就是 API key 明文；全项目的 engine
            都开了 ``hide_parameters``（``src/storage/engine.py``），
            因此 ``logger.exception`` 落盘时参数位置是 ``(...)``。
            改动那个开关前请先读那里的注释。
    """
    if not sharing_enabled(settings):
        return None

    credential = build_credential(settings)
    # 覆写 id 与显示名。id 必须是固定值（见 SYSTEM_CREDENTIAL_ID 的说明），
    # 显示名写成中文可读文案（``upsert`` 只在 name 为空时才自动命名，
    # 而自动名字是「DashScope (2)」这种，用户看了不知道这是什么）。
    credential.id = SYSTEM_CREDENTIAL_ID
    credential.name = SYSTEM_CREDENTIAL_NAME
    return await upsert_preset_credential(storage, SYSTEM_USER_ID, credential)


class SystemCredentialAccessPolicy(ResourceAccessPolicyBase):
    """把那条系统凭据共享给**所有**已鉴权用户的访问策略。

    ⚠️ 它的权限是 ``READ``（不是 ``EDIT``）：共享者可以**用**这条凭据，
    但不能改它、更不能删它。用 ``EDIT`` 的话，任何一个用户都能把
    全公司的密钥删掉或改成自己的 —— 这不是「共享」，是「开放的写权限」。
    """

    def __init__(self, *, state: dict[str, Any]) -> None:
        """初始化。

        Args:
            state (`dict[str, Any]`): 一个**可变的**共享状态盒子，键 ``seeded``
                表示「启动时那条系统凭据是否写成功」。用盒子而不是构造参数，
                是因为装配期（构造策略）与就绪期（真正写入）之间隔着整个
                ``create_app``，而策略对象必须在那之前交给框架。

                ⚠️ 运行期若有人删掉那条凭据，盒子不会知道：框架的
                ``list_resource`` 会因取不到记录而跳过它（返回 404/不出现），
                因此**不会**出现「列表里有、点开就报错」的假象。
                代价只是这条共享要等到下次重启才会恢复 —— 与「系统凭据
                不该被手工删除」这个前提相称。
        """
        self._state = state

    async def list_accessible(
        self,
        viewer_id: str,
        kind: ResourceKind,
        storage: Any,
    ) -> list[ResourceRef]:
        """列出 ``viewer_id`` 可跨属主访问的资源。

        Args:
            viewer_id (`str`): 当前查看者的身份。
            kind (`ResourceKind`): 要列举的资源种类。
            storage (`Any`): 存储（本策略不需要它 —— 是否共享只由启动期的
                写入结果决定，见 ``__init__``）。

        Returns:
            `list[ResourceRef]`: 只可能包含那一条系统凭据；其余情况为空列表。
        """
        del storage  # 见 docstring：本策略不查库
        if not self._state.get("seeded"):
            return []
        if kind != ResourceKind.CREDENTIAL:
            # 智能体与知识库**不共享**（见模块文档字符串）。
            return []
        if viewer_id == SYSTEM_USER_ID:
            # 属主自己走 owner 路径，策略里不该再列一遍。
            # （框架的 list_resource 会去重，但把它排除掉更省一次查询，
            # 也让「策略只描述跨属主」这条契约保持字面成立。）
            return []
        return [
            ResourceRef(
                kind=ResourceKind.CREDENTIAL,
                owner_id=SYSTEM_USER_ID,
                resource_id=SYSTEM_CREDENTIAL_ID,
                permission=ResourcePermission.READ,
            ),
        ]


def build_access_policy(
    settings: Any,
    state: dict[str, Any],
) -> ResourceAccessPolicyBase | None:
    """按配置决定要不要注册共享策略。

    Args:
        settings (`Settings`): 全量配置。
        state (`dict[str, Any]`): 传给 :class:`SystemCredentialAccessPolicy`
            的共享状态盒子。

    Returns:
        `ResourceAccessPolicyBase | None`: 返回 ``None`` 表示「不注册」——
        ``create_app`` 会据此回落到框架默认的 ``DenyAllResourceAccessPolicy``
        （owner 隔离），这正是零密钥部署应有的行为。
    """
    if not sharing_enabled(settings):
        return None
    return SystemCredentialAccessPolicy(state=state)


__all__ = [
    "SYSTEM_CREDENTIAL_ID",
    "SYSTEM_CREDENTIAL_NAME",
    "SYSTEM_USER_ID",
    "SystemCredentialAccessPolicy",
    "build_access_policy",
    "ensure_system_credential",
    "resolve_shared_model",
    "sharing_enabled",
    "system_chat_model_config",
    "system_credential_type",
]
