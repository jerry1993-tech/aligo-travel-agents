# -*- coding: utf-8 -*-
"""零密钥降级：把「服务能起来」补成「服务能对话」。

文件职责：
    回答一个问题 —— **一个刚克隆下来、没有任何模型密钥的部署，怎么让用户
    真的发出第一条消息？**

上下游依赖：
    - 上游：``src/llm/factory.py::should_use_mock``（判据的唯一来源）、
      ``src/llm/mock.py``（Mock 凭据/模型与其常量）。
    - 下游：``src/server/middleware/mock_credential.py``（每个已鉴权请求
      按需播种）、``src/server/routers/_default_model.py``（对外回答
      「我该用哪个模型」）、``scripts/smoke.py``（用它做端到端断言）。

==============================================================================
要解决的问题：探针全绿，但一个字都发不出去
==============================================================================
    框架的对话链路（``app/_service/_chat.py``）**不是**靠一个全局的默认模型
    工作的，它要两样东西同时在位：

        (a) 一条**属于调用者**的 credential 记录（``credentials`` 表）；
        (b) 会话的 ``chat_model_config`` 指向那条记录（``type`` +
            ``credential_id`` + ``model`` + ``parameters``）。

    缺 (b) 时 ``app/_service/_chat.py:1097-1101`` 直接 ``HTTPException(404, ...)``；缺 (a) 时
    ``CredentialFactory.from_dict`` 找不到记录。两条都不会让健康检查变红 ——
    ``/healthz`` 只看进程活没活，``/readyz`` 只看存储/总线通不通。于是最坏
    的部署形态出现了：**所有探针绿、SPA 正常渲染、「发送」按钮永远灰着**。

    前端那一半同样是空转：可用模型是 ``GET /credential/`` ×
    ``GET /model/?provider=`` 拼出来的，零凭据 ⇒ 分组为空 ⇒
    ``getFirstAvailableModel()`` 返回 ``null`` ⇒ ``selectedModel === null``
    ⇒ 发送按钮 ``disabled``（``web/frontend/src/pages/chat/ChatViewport.tsx``）。

    ⇒ 只要**按需**给每个用户补一条 Mock 凭据，并把 Mock 模型的卡片
    （``MockChatModel.list_models``）补上，整条链路就自然通了，
    前端**一行都不用改**。这是本模块存在的全部理由。

==============================================================================
为什么是「按需」而不是「启动时播种」
==============================================================================
    用户的身份来自请求（``X-User-ID`` 或 JWT 的 ``sub``），**没有用户表**
    —— 启动那一刻服务并不知道将来会有谁用。所以播种只能挂在「第一次见到
    某个用户」这个时机上，见 ``MockCredentialSeedMiddleware``。

    这也顺带解决了一个隐私问题：没有人会莫名其妙拥有一条凭据记录。

==============================================================================
为什么只播种 Mock、不播种真实密钥
==============================================================================
    ``docs/实施计划.md`` 的 P2 决策 2 曾设想「启动时从 ``OPENAI_API_KEY``
    播种一条 credential 记录」。**这个设想被安全评审否决了**，理由是它必须
    把运营者的真 key 复制进**用户自己拥有**的记录里：框架的
    ``ResourceAccessPolicyBase`` 默认是 owner 隔离，而 owner 读自己的凭据是
    **明文**（``app/_service/_access.py::_build_view`` 只对非 owner 打码）。
    也就是说，任何能通过鉴权的用户都能在「凭据」页把自己 key 的明文抄走。

    于是本模块只做 Mock 播种：Mock 凭据里**没有任何秘密**，复制一万份也
    不损失什么。真实密钥的路径保持「谁用谁配」——这才是框架 owner 隔离模型
    本来想表达的语义。缺 key 时的对外回答由
    ``GET /api/v1/default-model`` 给出（``mode="configured"/"missing"`` +
    一句人话 hint），而不是偷偷替用户做决定。

------------------------------------------------------------------------------
那「配了真密钥的部署」怎么办
------------------------------------------------------------------------------
    这是同一种病（探针全绿、发不出消息）的另一个形态，但**不能靠播种解决**：
    上面那条安全论证恰恰禁止把真 key 复制进用户记录。

    解法是框架留的正规扩展点 —— ``ResourceAccessPolicyBase``：
    凭据仍然只有**一条**（属主是 ``aligo-system``，密钥不复制），
    策略把它以 ``READ`` 权限共享出去。框架随后让用户**能用但读不到**：
    ``list_resource`` 把非 owner 的 ``data`` 打码成 ``{type, name}``
    （``app/_service/_access.py::_build_view``），而运行期解析
    ``resolve_credential`` 返回原始记录供构造模型。详见
    ``src/llm/system_credential.py``。

    于是本模块的判定多了一支：``mode="shared"``（见下方
    :func:`resolve_default_model`）。两条线互斥且互补 ——
    零密钥走 Mock，有密钥走共享，不可能同时生效。
"""

from __future__ import annotations

import logging
from typing import Any

from .factory import should_use_mock
from .mock import (
    MOCK_CREDENTIAL_NAME,
    MOCK_CREDENTIAL_TYPE,
    MOCK_MODEL_NAME,
    MockCredential,
    mock_credential_id,
)
from .preset_credential import upsert_preset_credential

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 对外回答的三种形态（``GET /api/v1/default-model`` 的 ``mode`` 字段）
# ---------------------------------------------------------------------------

#: 零密钥降级生效：返回的配置可以直接用。
MODE_MOCK = "mock"

#: 用户自己配了真实凭据。**不替他选** —— 选哪个模型是用户的决定，
#: 服务端只告诉他「去选择器里挑一个」。
MODE_CONFIGURED = "configured"

#: 用户自己没有凭据，但**运营者配的那条系统凭据对他可用**（只读共享）。
#: 与 ``configured`` 的区别是：这里的默认模型是运营者定的，
#: 用户没有「自己的选择」可以尊重，因此直接给出一份可用配置。
MODE_SHARED = "shared"

#: 既没有降级（开关关着或已有真 key）也没有用户凭据 —— 需要人来配。
MODE_MISSING = "missing"

#: 三种模式共用的解释文案。放常量里是因为 smoke 脚本与文档都要引用同一句话，
#: 手抄一份必然漂移。
HINT_MOCK = (
    "当前处于零密钥降级模式：已为本用户准备了一条 MockLLM 凭据"
    "（不产生任何费用、输出确定）。配置真实密钥后本模式自动关闭。"
)
HINT_CONFIGURED = (
    "检测到本用户已配置模型凭据，请在模型选择器中选择要使用的模型"
    "（零密钥降级不会覆盖你自己的选择）。"
)
HINT_SHARED = (
    "本用户没有自己的模型凭据，但管理员已配置系统默认模型且对全员开放"
    "（只读共享：你可以使用它，但不能查看或修改其中的密钥）。"
    "下面的配置可直接使用，也可以在选择器里换成其它模型。"
)
HINT_MISSING = (
    "未检测到可用的模型凭据，且零密钥降级已关闭"
    "（ALIGO__LLM__USE_MOCK_WHEN_NO_KEY=false，或已配置真实密钥）。"
    "请先在「凭据」页添加一条模型凭据。"
)


def mock_chat_model_config(credential_id: str) -> dict[str, Any]:
    """组装一份指向 Mock 凭据的 ``ChatModelConfig``。

    ````parameters`` 必须是 ``{}`` 而不是省略**：
    ``ChatModelConfig`` 的四个字段全部必填（``storage/_model/_session.py``），
    少一个就是 422；而 ``{}`` 在框架里是「取默认参数」的语义
    （``app/_service/_model.py`` 里 ``parameters`` 为假值时不构造 Parameters 实例）。

    Args:
        credential_id (`str`): 该用户的 Mock 凭据记录 id。

    Returns:
        `dict[str, Any]`: 可直接放进会话 ``chat_model_config`` 的字典。
        ⚠️ 这里**只有** type / credential_id / model / parameters ——
        没有任何密钥字段，因此可以安全地出现在 HTTP 响应与日志里。
    """
    return {
        "type": MOCK_CREDENTIAL_TYPE,
        "credential_id": credential_id,
        "model": MOCK_MODEL_NAME,
        "parameters": {},
    }


async def ensure_mock_credential(storage: Any, user_id: str) -> str:
    """确保 ``user_id`` 名下存在那条确定的 Mock 凭据，返回其 id。

    **幂等**：id 由 :func:`~src.llm.mock.mock_credential_id` 确定性派生，
    而 ``storage.upsert_credential`` 对「属于调用者的预设 id」是就地更新
    （``storage/_sql/_storage.py::upsert_credential``），因此重复调用只会
    刷新 ``updated_at``，不会长出第二条记录。

    Args:
        storage (`Any`): 框架的存储实现（``AsyncSQLAlchemyStorage`` 等）。
        user_id (`str`): 框架认定的用户标识。

    Returns:
        `str`: 凭据记录 id。

    Raises:
        Exception: 存储写入失败时**原样抛出**。调用方（中间件）决定是吞掉
            还是上报；本函数不替它们做决定，也不把异常翻译成一个看起来
            正常的返回值 —— 那会让「播种失败了」在两端都不可见。
            唯一被内部消化的异常是**主键冲突**，因为它代表
            「另一个执行者刚刚做完了同一件事」，不是失败
            （判定与取舍见 :mod:`src.llm.preset_credential`）。
    """
    credential = MockCredential(
        id=mock_credential_id(user_id),
        name=MOCK_CREDENTIAL_NAME,
    )
    # 「首次写入撞主键 = 另一个执行者刚做完了同一件事」这一条判断
    # 与系统凭据的播种**完全同形**，因此抽到
    # :func:`~src.llm.preset_credential.upsert_preset_credential` 里共用一份，
    # 理由与取舍（以及为什么用异常类名而不是 ``isinstance``）写在那里。
    return await upsert_preset_credential(storage, user_id, credential)


async def resolve_default_model(
    storage: Any,
    settings: Any,
    user_id: str,
    *,
    policy: Any | None = None,
) -> dict[str, Any]:
    """回答「这个用户现在能用什么模型」。

    这是 ``GET /api/v1/default-model`` 的全部实现，抽成函数是为了让 smoke
    脚本与单测能绕过 HTTP 直接验证同一份判据（判据一旦分叉，最常见的后果
    就是「探针说可以、实际发不出去」—— 本模块要治的正是这种病）。

    四种形态的判定顺序是**刻意的**：

        1. 零密钥降级开着 ⇒ 播种并返回 Mock 配置。先判它，是因为「开关开着」
           本身就是运营者的明示选择；此时用户列表里有没有别的鉴权记录都
           不影响「服务至少能跑」这个承诺。
        2. 否则看该用户有没有**自己的**凭据 ⇒ ``configured``（不替他选模型）。
           ⚠️ 必须排在共享之前：用户自己配了凭据时，运营者的共享是**备选**
           而不是**默认** —— 反过来的话，一个自带了公司密钥的用户会被
           悄悄切到系统凭据上，账单与配额都算错了地方。
        3. 都没有、但系统凭据对他可用 ⇒ ``shared`` + 一份可用配置。
        4. 都没有 ⇒ ``missing`` + 一句可执行的 hint。

    Args:
        storage (`Any`): 框架的存储实现。
        settings (`Any`): 全量配置，只需读 ``settings.llm``。
        user_id (`str`): 框架认定的用户标识。
        policy (`Any | None`): ``app.state.resource_access_policy`` ——
            「系统凭据对谁可用」这件事的唯一权威。``None`` 表示调用方
            没有接上访问策略（单测、或框架回落到 deny-all），
            此时共享分支整体不生效。

    Returns:
        `dict[str, Any]`: ``{mode, chat_model_config, hint}``。
        ``chat_model_config`` 只在 ``mock`` 与 ``shared`` 两种模式下非空 ——
        这两种模式的共同点是「默认选谁由运营者决定」，而 ``configured``
        是「选谁由用户决定」，因此不替他选。
        四种模式下的返回值都**不含任何密钥**。
    """
    if should_use_mock(settings):
        try:
            credential_id = await ensure_mock_credential(storage, user_id)
        except Exception as exc:  # noqa: BLE001 —— 见下：失败必须可解释，而不是 500
            # ⚠️ 这里不把异常抛给调用方：本端点的契约是「告诉我该怎么办」，
            # 抛出去只会变成一个 500，而 500 恰恰是**没有信息量**的那种答复 ——
            # 运维看到的只有「默认模型接口挂了」，而不是「数据库写不进去」。
            # 只带异常**类名**、绝不带上 str(exc)：存储层异常里可能含连接串。
            # ⚠️ 提示里的 ``make logs`` **不带**服务名：Makefile 的服务名走
            # ``SVC`` 变量（默认 app），``make logs app`` 会被当成两个目标而报
            # ``No rule to make target 'app'`` —— 下面两条提示同此。
            return {
                "mode": MODE_MISSING,
                "chat_model_config": None,
                "hint": (
                    f"零密钥降级凭据写入失败（{type(exc).__name__}）。"
                    "请检查数据库连通性（make psql / make logs）后重试。"
                ),
            }
        return {
            "mode": MODE_MOCK,
            "chat_model_config": mock_chat_model_config(credential_id),
            "hint": HINT_MOCK,
        }

    try:
        credentials = await storage.list_credentials(user_id)
    except Exception as exc:  # noqa: BLE001 —— 理由同上
        return {
            "mode": MODE_MISSING,
            "chat_model_config": None,
            "hint": (
                f"读取凭据列表失败（{type(exc).__name__}）。"
                "请检查数据库连通性（make psql / make logs）后重试。"
            ),
        }

    if credentials:
        return {
            "mode": MODE_CONFIGURED,
            "chat_model_config": None,
            "hint": HINT_CONFIGURED,
        }

    # ---- 系统凭据共享（只在运营者配了真密钥时为非 None） --------------------
    # 用**局部** import 而不是模块级：``src.llm.system_credential`` 会 import
    # ``agentscope.app.access``，而本模块被 ``src/server/middleware/
    # mock_credential.py`` 在装配期 import。放在模块级会让「零密钥启动」
    # 这条路径多背一个框架子包的导入风险 —— 那条路径恰恰是最不该出岔子的
    # （README 承诺的开箱即用路径）。
    from .system_credential import resolve_shared_model

    try:
        shared = await resolve_shared_model(storage, settings, user_id, policy)
    except Exception:  # noqa: BLE001 —— 见下：策略的 bug 不能变成用户的 500
        # ⚠️ 这里与上面两处「吞掉异常」的取舍**不同**，值得说清楚：
        # 会话链路本身也会走策略，所以策略坏掉时**对话**照样会失败 ——
        # 端点吞掉异常并不会掩盖故障，它只是不再制造**第二个**故障点。
        # 反过来，如果让它抛出去，``/api/v1/default-model`` 会变成一个 500，
        # 而排障的人会去查这个端点本身 —— 方向完全错了。
        # 因此：给用户一个可解释的答复，把栈留给日志。
        logger.exception("查询系统凭据共享失败，按「未配置」处理。")
        shared = None

    if shared is not None:
        return {
            "mode": MODE_SHARED,
            "chat_model_config": shared,
            "hint": HINT_SHARED,
        }
    return {
        "mode": MODE_MISSING,
        "chat_model_config": None,
        "hint": HINT_MISSING,
    }


__all__ = [
    "HINT_CONFIGURED",
    "HINT_MISSING",
    "HINT_MOCK",
    "HINT_SHARED",
    "MODE_CONFIGURED",
    "MODE_MISSING",
    "MODE_MOCK",
    "MODE_SHARED",
    "ensure_mock_credential",
    "mock_chat_model_config",
    "resolve_default_model",
]
