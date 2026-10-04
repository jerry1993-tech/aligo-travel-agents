# -*- coding: utf-8 -*-
"""``/api/v1/memory/**`` —— 长期记忆的**显式**读写面。

文件职责：
    把 :class:`~src.memory.service.TravelerMemory` 的门面暴露成五个端点：
    读/改结构化画像，记住/召回/忘记语义笔记。它不实现任何记忆逻辑 ——
    校验、合并、幂等、降级全在 ``src/memory/`` 里，本模块只做
    **协议转换**（HTTP ⇄ 门面）与**错误映射**。

上下游依赖：
    - 上游：``agentscope.app.deps.get_current_user_id``（框架的身份依赖）、
      ``app.state.traveler_memory``（``src/server/constants.py::MEMORY_ATTR``）。
    - 下游：``src/memory/service.py::TravelerMemory``、
      ``src/memory/profile.py::ProfilePatch``、``src/memory/semantic.py``。

==============================================================================
这个 API 面是补上的，不是新加的 —— 它早就被三处代码「引用」着
==============================================================================
    在写下本文件之前，长期记忆的**写入面在产品里不可达**：
    ``remember`` / ``update_profile`` / ``put_profile`` 在 ``src/`` 里
    一个调用者都没有（7 个业务工具里没有记忆工具，业务路由里也没有
    记忆接口）。而与此同时，下面三处都已经把它当成存在的东西在描述：

      · ``src/server/constants.py`` 的 ``MEMORY_ATTR`` 写着「读取方：
        ``/api/v1/memory/**`` 的画像接口」；
      · :meth:`src.memory.service.TravelerMemory.get_profile` 的 docstring
        写着「它是给 ``/api/v1/memory/profile`` 这类**显式**接口用的」；
      · ``docs/03-模块关系与调用逻辑.md`` 写着建表失败时「『记住偏好』
        这类写入会失败，且会以异常形式**出现在用户面前**」。

    前两条是「文档指向一个不存在的接口」，第三条更微妙 ——
    它描述了一个**正确**的故障语义（写失败不静默），但那个写入
    根本没有入口，于是这句话无从兑现。本模块把这三处一起变成真的。

==============================================================================
⚠️ 三条硬约束（每一条都对应一类会真实发生的错）
==============================================================================
  1. **身份只来自鉴权**。``user_id`` **只**取自
     ``Depends(get_current_user_id)``，绝不出现在请求体里 ——
     请求模型一律 ``extra="forbid"``，body 里带 ``user_id`` 直接 422。
     这不是洁癖：能写别人画像的接口等于一次水平越权，
     而它的后果（把 A 的成本中心写进 B 的订单）在账单上才看得出来。

  2. **读也报 503，不降级成空**。``TravelerMemory.recall`` 在**对话链路**上
     永不抛、失败降级，那是对的（锦上添花）。但这里是**显式**接口：
     调用方正在等一个答案，把故障变成「空列表 / 没有画像」等于
     把一次故障伪装成事实（与 :meth:`...TravelerMemory.get_profile`
     docstring 同一条理由）。所以本模块在能力缺失时显式 503，
     且 hint 里说清楚**缺的是哪一半**（关掉了？还是向量模型起不来？）。

  3. **写失败原样透出、经脱敏**。写路径照常抛（``src/memory/`` 的核心约定），
     本模块把它映射成 503 并把 ``safe_error`` 过的原因放进响应 ——
     用户必须知道「没记住」，而错误文本里绝不能出现连接串凭据。
"""

from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, Body, Depends, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

# 框架自己的身份依赖。与 ``_identity.py`` / ``_default_model.py`` 同一写法，
# 理由见那两个文件的模块文档（它才是「鉴权链路整体可用」的证明）。
from agentscope.app.deps import get_current_user_id

from ...memory.profile import ProfilePatch, ProfileValidationError, TravelerProfile
from ...memory.semantic import MemoryKind, MemoryNote
from ...memory.service import TravelerMemory
from ...observability.context import get_trace_id
from ...observability.redaction import redact, safe_error
from ..constants import MEMORY_ATTR

logger = logging.getLogger(__name__)

router = APIRouter()


# ==============================================================================
# 装配读取与错误响应
# ==============================================================================
def _memory_of(request: Request) -> TravelerMemory | None:
    """取 ``app.state`` 上的长期记忆门面。

    ⚠️ 用 ``getattr(..., None)`` 而不是直接取属性：测试里存在「裸 FastAPI」
    这种装配形态，直接取会 ``AttributeError`` → 500，而本模块想给的是一条
    **能看懂**的 503（与 ``_default_model.py`` 的取舍一致）。

    Args:
        request (`Request`): 当前请求。

    Returns:
        `TravelerMemory | None`: 门面；应用未按契约装配时为 ``None``。
    """
    return getattr(request.app.state, MEMORY_ATTR, None)


def _error(status_code: int, detail: str) -> JSONResponse:
    """构造一个**自解释**的错误响应。

    ⚠️ 错误体用 ``{"detail", "trace_id"}`` 两个字段：``detail`` 与
    FastAPI 自己的校验错误（422）同名字段，客户端只需要认一种形状；
    ``trace_id`` 与所有成功响应同源（``src/observability/context.py``），
    排障时能把这一条响应与日志里的整条链路对上。

    Args:
        status_code (`int`): HTTP 状态码。
        detail (`str`): 给调用方看的中文说明（**不得**含任何密钥）。

    Returns:
        `JSONResponse`: 错误响应。
    """
    return JSONResponse(
        status_code=status_code,
        content={"detail": detail, "trace_id": get_trace_id()},
    )


def _unavailable(memory: TravelerMemory | None, need: str) -> JSONResponse:
    """把「能力缺失」翻译成一条说清**缺哪一半**的 503。

    ⚠️ 三种缺失的处置方式完全不同，所以 hint 必须区分开：

      · 应用没装配 —— 启动方式的问题（去查 ``src.server.app:app``）；
      · 记忆被关掉 —— 运营者的选择（改 ``ALIGO__MEMORY__ENABLED``）；
      · 只有语义那半缺失 —— 向量模型起不来（去看启动日志里的告警）。

    把它们混成一句「记忆不可用」的话，运维会先去改配置，而真正的问题
    在启动日志的第二屏。

    Args:
        memory (`TravelerMemory | None`): 门面；``None`` 表示未装配。
        need (`str`): 需要的能力，``"profile"`` 或 ``"semantic"``。

    Returns:
        `JSONResponse`: 503 响应。
    """
    if memory is None:
        return _error(
            503,
            "应用未完成装配（app.state 上没有长期记忆门面），"
            "请确认服务是通过 src.server.app:app 启动的。",
        )
    if not memory.enabled:
        return _error(
            503,
            "长期记忆已被关闭（ALIGO__MEMORY__ENABLED=false），"
            "本接口当前不可用。要恢复请打开该开关并重启服务。",
        )
    if need == "semantic":
        # ⚠️ 走到这里说明开关是开的，那么缺的只可能是语义那一半 ——
        # ``build_memory`` 在向量模型不可用时会把 semantic 置为 None
        # 并打一条 WARNING（而不是让应用起不来）。
        return _error(
            503,
            "本进程没有可用的语义记忆（向量模型不可用，长期记忆已降级为"
            "「仅结构化画像」）。启动日志里有一条说明原因的告警；"
            "结构化画像接口 /api/v1/memory/profile 不受影响。",
        )
    return _error(
        503,
        "本进程没有配置结构化画像仓储（应用未按契约装配）。",
    )


def _write_failed(action: str, exc: Exception) -> JSONResponse:
    """把一次**写失败**映射成 503，并把脱敏后的原因交给调用方。

    ⚠️ 写路径不能吞异常（``src/memory/`` 的核心约定）：用户说了
    「记住这个」，静默丢弃是一次欺骗 —— 他下周才会发现助手没记住，
    而中间没有任何一处告诉过他。所以这里必须把失败**说出来**，
    但只能说脱敏后的内容（``safe_error`` 会剥掉 ``scheme://user:pass@``
    形式的凭据；Milvus 的 URI 是支持这种写法的）。

    Args:
        action (`str`): 失败的动作，用于日志（如「写入笔记」）。
        exc (`Exception`): 原始异常。

    Returns:
        `JSONResponse`: 503 响应（**可重试**：向量库/数据库故障通常
        是暂时的，而 4xx 会告诉客户端「改请求」——那会把它引向错误方向）。
    """
    logger.warning("%s失败：%s", action, safe_error(exc))
    return _error(
        503,
        f"{action}失败：{safe_error(exc)}。"
        "本次写入**没有**生效（记忆的写路径不会静默降级）。",
    )


# ==============================================================================
# 请求模型
# ==============================================================================
class ProfilePatchIn(BaseModel):
    """``PUT /api/v1/memory/profile`` 的请求体：一次**部分**更新。

    ⚠️ 与 :class:`~src.memory.profile.ProfilePatch` 一样，``None`` 表示
    「不改这个字段」，而不是「把它设成 None」（理由见
    :class:`~src.memory.profile.ProfilePatch` 的文档：把「不传」与
    「传了空」混在一起，会造成「更新一个字段时静默清掉另一个」）。

    ⚠️ **要清空字段就给空值，而不是给 ``null``**：列表字段给 ``[]``、
    字符串字段给 ``""`` —— 两者都走「值不是 None ⇒ 覆盖」这条正常路径
    （``ProfilePatch.apply``）。``null`` 永远只表示「别动它」。这条约定
    在**每一个位置**都成立，包括字典的**值**：

    ==============================  =====================================
    想做的事                          怎么写
    ==============================  =====================================
    清空航司/品牌/饮食偏好列表        ``"preferred_airlines": []``
    清空座位偏好、成本中心等字符串     ``"seat_preference": ""``
    清空舱位偏好                      ``"preferred_cabin": ""``
    删掉某一个常旅客号                ``"frequent_flyer_numbers": {"CA": ""}``
    ==============================  =====================================

    ⚠️ 早先这里的文案写的是「清空字段走整体覆盖」，而**整体覆盖根本没有
    HTTP 入口**（``put_profile`` 在 ``src/`` 里仍然只有测试在调）——
    又是一处「文档指向一个不存在的能力」（对抗性审核 2026-10-03 的发现）。
    要么给能力、要么改文案；这里选择改文案，因为清空一个字段用 ``[]`` /
    ``""`` 已经能做，而那比新开一个整体覆盖端点安全得多。

    ⚠️ 但只改文案还不够（同一轮审核的后续发现）：当时 ``preferred_cabin``
    给了 ``""`` 会**报 400**（``_validate_cabin`` 把空串当非法值），而
    ``frequent_flyer_numbers`` 只合并不删除、``{}`` 与「不传」都是「不动」。
    于是文案里那句「字符串给 ``""``」在其中两个字段上**是假的**，而那两个
    字段也就成了**永远清不掉**的死角 —— 换掉航司之后旧卡号会一直留在画像里、
    被注入 Prompt。修法不是把文案写得更啰嗦，而是**把约定补成真的**：
    空值在哪儿都是「清空」。上表四行都有
    ``tests/test_api_memory.py::test_clearing_a_field_takes_an_empty_value_not_null``
    与 ``tests/test_memory_profile.py`` 的用例守着。

    ⚠️ ``extra="forbid"`` 不是可选项：没有它，请求体里的 ``user_id``
    会被**静默忽略** —— 调用方以为自己在改别人的画像，接口却悄悄改了他自己的，
    两边都以为自己是对的。（更糟的方向是有人真的把它接进逻辑。）
    拒绝未知字段让这类误解在 422 上就停住。

    ⚠️ 字段校验**不在这里重复实现**：舱位的合法值、列表的去重保序
    都在 :class:`~src.memory.profile.TravelerProfile` 的 validator 里。
    本模型只负责「形状」，语义校验由 ``update_profile`` 那条路上的
    ``model_validate`` 完成 —— 两处各写一遍必然漂移。
    """

    model_config = ConfigDict(extra="forbid")

    preferred_cabin: str | None = Field(
        default=None,
        description="舱位偏好；合法值 ECONOMY / PREMIUM_ECONOMY / BUSINESS / FIRST。",
    )
    seat_preference: str | None = Field(default=None, description="座位偏好（自由文本）。")
    preferred_airlines: list[str] | None = Field(
        default=None,
        description="偏好航司代码列表（**整体替换**，按优先级排序）。",
    )
    preferred_hotel_brands: list[str] | None = Field(
        default=None,
        description="偏好酒店品牌列表（整体替换）。",
    )
    dietary_needs: list[str] | None = Field(default=None, description="饮食需求列表（整体替换）。")
    cost_center: str | None = Field(default=None, description="成本中心 / 项目号（硬事实）。")
    default_approver: str | None = Field(default=None, description="默认审批人工号（硬事实）。")
    frequent_flyer_numbers: dict[str, str] | None = Field(
        default=None,
        description="航司代码 → 常旅客号；本字段是**合并**（不会删掉没提到的航司）。",
    )
    accessibility_needs: str | None = Field(default=None, description="无障碍需求。")

    def to_patch(self) -> ProfilePatch:
        """转成 :class:`~src.memory.profile.ProfilePatch`。

        Returns:
            `ProfilePatch`: 等价的 patch 值对象。
        """
        return ProfilePatch(**self.model_dump())

    def is_empty(self) -> bool:
        """是否一个字段都没给。

        ⚠️ 用途是把「空 patch」拦成 400，而不是让它变成一个
        **看起来成功、实际什么都没做**的 200：后者会让调用方以为
        偏好已经写进去了（一个典型场景是前端表单提交时字段名拼错，
        而 ``extra="forbid"`` 已经挡住了拼错的情况，这里兜的是
        「确实一个字段都没填」）。

        Returns:
            `bool`: 所有字段都是 ``None`` 时返回 True。
        """
        return all(value is None for value in self.model_dump().values())


class NoteIn(BaseModel):
    """``POST /api/v1/memory/notes`` 的请求体：记住一句话。"""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1,
        max_length=2000,
        description="要记住的原话（召回时按语义相似度匹配，忘记时按**原文**精确匹配）。",
    )
    kind: Literal["preference", "observation", "trip"] = Field(
        default=MemoryKind.PREFERENCE,
        description="笔记类别：偏好 / 观察 / 行程历史（见 src/memory/semantic.py::MemoryKind）。",
    )


class NoteTarget(BaseModel):
    """``DELETE /api/v1/memory/notes`` 的请求体：要忘掉的那句话。"""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(
        min_length=1,
        max_length=2000,
        description="要忘掉的**原文**（与当初 POST 进来的文本一致）。",
    )


def _note_json(note: MemoryNote) -> dict[str, object]:
    """把一条笔记序列化成响应体的形状。

    ⚠️ 手写字段而不是 ``dataclasses.asdict``：``asdict`` 会把将来
    新增的任何内部字段（比如向量、debug 信息）**自动**带进 HTTP 响应，
    而这条响应会进日志与浏览器缓存。显式列举让「多带了一个字段」
    变成一次需要有人主动改代码的决定。

    Args:
        note (`MemoryNote`): 一条笔记。

    Returns:
        `dict[str, object]`: 可直接 json 序列化的字典。
    """
    return {
        "note_id": note.note_id,
        "text": note.text,
        "kind": note.kind,
        "score": note.score,
    }


def _profile_json(profile: TravelerProfile) -> dict[str, object]:
    """把一份画像序列化成响应体的形状。

    ⚠️ ``model_dump()`` 在这里是安全的：画像的字段集合是**契约**
    （前端表单与它一一对应），不存在「内部字段被顺手带出去」的通道。
    与上面的 :func:`_note_json` 手写字段的理由并不矛盾 —— 那里挡的是
    **未来**新增的内部字段，这里用的是 pydantic 模型自己的公开字段表。

    Args:
        profile (`TravelerProfile`): 画像。

    Returns:
        `dict[str, object]`: 可直接 json 序列化的字典。
    """
    return profile.model_dump()


# ==============================================================================
# 一、结构化画像
# ==============================================================================
@router.get(
    "/memory/profile",
    summary="读取当前身份的结构化画像",
    description=(
        "返回当前身份在结构化画像里可查的全部字段（硬事实 + 软偏好）。"
        "没有画像记录时 ``profile`` 为 ``null``（**与「记忆不可用」是不同的处境**，"
        "后者返回 503）。身份只来自鉴权，不接受任何指定 user_id 的参数。"
    ),
    responses={
        200: {"description": "查询成功（没有任何记录时 profile 为 null）"},
        401: {"description": "凭据缺失或无效（由鉴权中间件返回）"},
        503: {"description": "长期记忆被关闭，或本进程没有结构化画像仓储"},
    },
)
async def read_profile(
    request: Request,
    user_id: str = Depends(get_current_user_id),
) -> JSONResponse:
    """读取当前身份的结构化画像。

    Args:
        request (`Request`): 当前请求（读取 ``app.state`` 上的记忆门面）。
        user_id (`str`): 由框架的身份依赖解析出的用户标识。

    Returns:
        `JSONResponse`: ``{profile, trace_id}``；``profile`` 为 ``null``
        表示这个用户还没有任何画像记录。

    ⚠️ 为什么这里不吞异常（``get_profile`` 的契约也是不吞）：
        仓储读失败时如果返回 ``profile: null``，调用方（前端、运维的 curl）
        会看到「这个用户没有画像」——一个**错误的事实**。
    """
    memory = _memory_of(request)
    if memory is None or not memory.enabled or not memory.has_repository:
        return _unavailable(memory, "profile")

    try:
        profile = await memory.get_profile(user_id)
    except Exception as exc:  # noqa: BLE001 —— 见下面的映射说明
        # ⚠️ 读失败也要报错，理由见 docstring。映射成 503（依赖暂时不可用）
        # 而不是 500：这里的失败几乎总是 PostgreSQL 侧的（连接断了、
        # 表被删了），对调用方来说它是可重试的。
        return _error(
            503,
            f"读取画像失败：{safe_error(exc)}。"
            "这不代表「没有画像」——本次查询没有拿到答案。",
        )

    return JSONResponse(
        {
            "profile": _profile_json(profile) if profile is not None else None,
            "trace_id": get_trace_id(),
        },
    )


@router.put(
    "/memory/profile",
    summary="部分更新当前身份的结构化画像",
    description=(
        "只更新请求体里给出的字段（``null`` 表示不改）。"
        "``frequent_flyer_numbers`` 是**合并**（不会删掉没提到的航司），"
        "其余列表字段是**整体替换**。至少要给一个字段，否则返回 400。"
        "清空字段一律给**空值**：列表给 `[]`、字符串给 `\"\"`、"
        "删掉某个常旅客号给 `{\"CA\": \"\"}` —— `null` 永远只表示「别动它」。"
    ),
    responses={
        200: {"description": "更新成功，返回合并后的完整画像"},
        400: {"description": "一个字段都没给，或字段值不合法（如舱位不在枚举里）"},
        401: {"description": "凭据缺失或无效（由鉴权中间件返回）"},
        422: {"description": "请求体形状不对（含未知字段，如 user_id）"},
        503: {"description": "长期记忆被关闭，或画像存储写入失败"},
    },
)
async def update_profile(
    request: Request,
    payload: ProfilePatchIn,
    user_id: str = Depends(get_current_user_id),
) -> JSONResponse:
    """部分更新当前身份的结构化画像。

    Args:
        request (`Request`): 当前请求。
        payload (`ProfilePatchIn`): 要更新的字段。
        user_id (`str`): 由框架的身份依赖解析出的用户标识。

    Returns:
        `JSONResponse`: ``{profile, trace_id}``。

    ⚠️ 这里抓的是 :class:`~src.memory.profile.ProfileValidationError`
        （**我们自己的**异常类型），而不是 ``ValueError``：舱位之类的校验
        错误来自 :class:`~src.memory.profile.TravelerProfile` 的 validator，
        而 :meth:`~src.memory.profile.ProfilePatch.apply` 在值进入系统的
        那道门上把它**翻译**成了这个窄类型。抓 ``ValueError`` 会把下层
        仓储/驱动抛出的、含义完全相反的 ``ValueError``（如
        ``failed to connect postgresql+asyncpg://user:pw@…``）一起报成 400
        —— 既是**把服务故障说成调用方错误**（客户端会去改一个没问题的
        请求），又是**一条绕过 ``safe_error`` 的凭据泄漏通道**。
        仓储侧的 ``ValueError`` 现在落到下面的 ``except Exception`` ⇒ 503。
    """
    memory = _memory_of(request)
    if memory is None or not memory.enabled or not memory.has_repository:
        return _unavailable(memory, "profile")

    if payload.is_empty():
        return _error(
            400,
            "请求体里一个字段都没给（全部为 null），没有可更新的内容。"
            "要清空某个字段就给**空值**：列表给 []、字符串给 \"\"、"
            "删掉某个常旅客号给 {\"CA\": \"\"}。"
            "null 的语义始终是「不改这个字段」，所以它清不掉任何东西。",
        )

    try:
        profile = await memory.update_profile(user_id, payload.to_patch())
    except ProfileValidationError as exc:
        # 值不合法（如 preferred_cabin="随便"）。
        #
        # ⚠️ 这里抓的是**我们自己**的异常类型，不是 ``ValueError``。
        # 2026-10-03 的对抗性审核抓到过旧写法的一个真实缺陷：``except
        # ValueError`` 的覆盖面远大于「值不合法」—— 它同时接住
        # pydantic 的 ``ValidationError``（仓储把**库里读出来的**记录
        # 反序列化时也会抛，那是数据损坏）以及任何驱动碰巧抛出的
        # ``ValueError``。后果是两重的：把服务端故障报成 400（告诉调用方
        # 「改请求」，于是它不会重试），以及把异常原文直接拼进响应体
        # —— 一次实测里响应里出现了 ``postgresql+asyncpg://user:pw@…``
        # 形式的连接串。分类必须由我们做，见 ``ProfileValidationError``。
        # ⚠️ 这里用 ``redact`` 而不是 ``safe_error``：后者的输出形如
        # ``ProfileValidationError: preferred_cabin：…``，会把**我们自己的
        # 异常类名**塞进给调用方看的 400 里。这条消息整个由我们撰写
        # （不是下层驱动抛的），前缀是纯噪音；而脱敏（``redact``）一步不少
        # —— 「自己的值也过一遍脱敏」是刻意留着的第二道保险，
        # 因为 ``ProfilePatch`` 的值最终来自请求体，本就不该假设它干净。
        return _error(400, f"画像字段不合法：{redact(str(exc))}")
    except Exception as exc:  # noqa: BLE001 —— 写失败必须说出来
        # ⚠️ **不**为 RuntimeError 单独开一个分支。门面在「没有这个能力」时
        # 确实抛 RuntimeError，但那种情况上面已经拦住了；反过来，
        # 底层（pymilvus / SQLAlchemy）同样会抛 RuntimeError —— 两者在
        # 这里长得一模一样，为它开分支等于给底层异常开了一条**绕过
        # ``safe_error`` 的通道**（本文件的脱敏用例抓到的就是这件事）。
        # 统一走 _write_failed：同样的 503，但原因一定脱敏过。
        return _write_failed("更新画像", exc)

    return JSONResponse(
        {"profile": _profile_json(profile), "trace_id": get_trace_id()},
    )


# ==============================================================================
# 二、语义笔记
# ==============================================================================
@router.post(
    "/memory/notes",
    summary="记住一句话（写入长期记忆）",
    description=(
        "把一句话写进当前身份的语义记忆。**幂等**：同一句话记住两次"
        "不会产生两条（note_id 由 (user_id, text) 决定，第二次覆盖第一次），"
        "因此成功返回 200 而不是 201。写入失败返回 503 且**不会静默丢弃**。"
    ),
    responses={
        200: {"description": "已记住（重复写入同一句话时是覆盖）"},
        400: {"description": "文本为空（去掉空白后没有内容）"},
        401: {"description": "凭据缺失或无效（由鉴权中间件返回）"},
        422: {"description": "请求体形状不对（含未知字段，如 user_id）"},
        503: {"description": "记忆被关闭 / 向量模型不可用 / 向量库写入失败"},
    },
)
async def remember_note(
    request: Request,
    payload: NoteIn,
    user_id: str = Depends(get_current_user_id),
) -> JSONResponse:
    """记住一句话。

    Args:
        request (`Request`): 当前请求。
        payload (`NoteIn`): 要记住的文本与类别。
        user_id (`str`): 由框架的身份依赖解析出的用户标识。

    Returns:
        `JSONResponse`: ``{note, trace_id}``；``note.score`` 恒为 1.0
        （刚写进去的这条与它自己完全相似，见 ``remember`` 的实现）。

    ⚠️ 文本先 ``strip()`` 再判空：``" "`` 能通过 pydantic 的
        ``min_length=1``，但它在语义上是空的 —— 一条空白笔记会占掉
        召回窗口里的一个位置，而它什么信息都没有。
    """
    memory = _memory_of(request)
    if memory is None or not memory.enabled or not memory.has_semantic:
        return _unavailable(memory, "semantic")

    text = payload.text.strip()
    if not text:
        return _error(400, "文本为空（去掉首尾空白后没有任何内容）。")

    try:
        note = await memory.remember(user_id, text, kind=payload.kind)
    except Exception as exc:  # noqa: BLE001 —— 写失败必须说出来（含脱敏）
        return _write_failed("写入记忆", exc)

    return JSONResponse(
        {"note": _note_json(note), "trace_id": get_trace_id()},
    )


@router.get(
    "/memory/notes",
    summary="召回与查询最相关的笔记",
    description=(
        "按语义相似度召回当前身份的笔记（跨用户的记录**不会**被召回）。"
        "``query`` 必填 —— 这是一个检索接口，没有查询就没有答案。"
        "语义那半失败时返回 200 且 ``error`` 非空（召回是增强，不中断调用方），"
        "但能力整体缺失时返回 503。"
    ),
    responses={
        200: {"description": "检索完成（error 非空表示这次召回失败，notes 为空）"},
        401: {"description": "凭据缺失或无效（由鉴权中间件返回）"},
        422: {"description": "缺少 query，或 top_k 超出范围"},
        503: {"description": "记忆被关闭，或本进程没有语义记忆能力"},
    },
)
async def list_notes(
    request: Request,
    query: str = Query(
        min_length=1,
        description="查询文本（必填）。语义召回按它与笔记的相似度排序。",
    ),
    top_k: int = Query(
        default=5,
        ge=1,
        le=50,
        description="最多返回几条；上限 50 是为了挡住「一次把库拖干」的调用。",
    ),
    user_id: str = Depends(get_current_user_id),
) -> JSONResponse:
    """召回当前身份的笔记。

    Args:
        request (`Request`): 当前请求。
        query (`str`): 查询文本。
        top_k (`int`): 最多返回几条。
        user_id (`str`): 由框架的身份依赖解析出的用户标识。

    Returns:
        `JSONResponse`: ``{notes, error, trace_id}``。

    ⚠️ 这里与「能力缺失」的取舍是本模块最微妙的一处：
        语义**检索失败**（Milvus 抖了一下）返回 ``200 + error``——
        因为 ``recall`` 的契约就是「永不抛、失败降级」，而调用方
        （前端）拿到的 ``error`` 足以区分「没有相关记忆」与「这次没查到」；
        但语义**能力整体不存在**（开关关了 / 模型起不来）返回 ``503``——
        那时连「降级结果」都不成立，返回空列表就是一句谎话。
    """
    memory = _memory_of(request)
    if memory is None or not memory.enabled or not memory.has_semantic:
        return _unavailable(memory, "semantic")

    # ⚠️ ``min_length=1`` 拦不住 ``"   "`` —— 一个空白查询在语义上就是
    # 「没有查询」，而 :meth:`recall` 对空查询还会**刻意**返回空结果且
    # ``error=None``（见 src/memory/semantic.py），于是
    # ``?query=%20%20`` 会得到 ``{"notes": [], "error": null}``：
    # 一次「没有依据的结论」，与「确实没有相关记忆」在响应里长得一模一样。
    # POST / DELETE 两个端点都对空白做了同样的先 strip 再判空，
    # 这里补上同一步（对抗性审核 2026-10-03 的发现）。
    query = query.strip()
    if not query:
        return _error(400, "查询文本为空（去掉首尾空白后没有任何内容）。")

    # ⚠️ ``recall`` 永不抛（见 src/memory/semantic.py 的文档），
    # 因此这里不需要 try/except —— 但也正因为如此，
    # **不能**把 ``recall.notes`` 当成「确实没有相关记忆」的同义词：
    # ``recall.error`` 非空时它表示「这次没查到」。
    #
    # ⚠️ ``top_k`` 往下传而不是在本地 ``[:top_k]`` 切片：切片只能在
    # 默认条数之内**缩小**结果，调用方要 10 条而配置是 5 时它会
    # 静默地只给 5 条 —— 一个「参数看起来生效了，其实没有」的假象。
    context = await memory.recall(user_id, query, top_k=top_k)
    notes = context.recall.notes

    return JSONResponse(
        {
            "notes": [_note_json(note) for note in notes],
            "error": context.recall.error,
            "trace_id": get_trace_id(),
        },
    )


@router.delete(
    "/memory/notes",
    summary="忘掉一条笔记，或清空自己的全部笔记",
    description=(
        "二选一：给请求体 ``{\"text\": \"...\"}`` 按**原文**忘掉一条；"
        "或给 ``?all=true`` 清空当前身份的**全部**笔记（返回删掉的条数）。"
        "「按原文」是刻意的 —— 先 GET 召回拿到原文，再带原文来删，"
        "避免「按相似度删」误删用户没说过的内容。"
    ),
    responses={
        200: {"description": "已删除（all=true 时返回 removed 条数）"},
        400: {"description": "既没给 all=true 也没给请求体，或两者同时给了"},
        401: {"description": "凭据缺失或无效（由鉴权中间件返回）"},
        422: {"description": "请求体形状不对（含未知字段，如 user_id）"},
        503: {"description": "记忆被关闭 / 向量模型不可用 / 向量库删除失败"},
    },
)
async def forget_notes(
    request: Request,
    payload: NoteTarget | None = Body(
        default=None,
        description='要忘掉的原文，形如 {"text": "我一般坐靠窗"}；与 all=true 二选一。',
    ),
    all_notes: bool = Query(
        default=False,
        alias="all",
        description="true 表示清空当前身份的全部笔记（与其他用户的记录无关）。",
    ),
    user_id: str = Depends(get_current_user_id),
) -> JSONResponse:
    """忘掉一条或全部笔记。

    Args:
        request (`Request`): 当前请求。
        payload (`NoteTarget | None`): 要忘掉的原文。
        all_notes (`bool`): 是否清空全部。
        user_id (`str`): 由框架的身份依赖解析出的用户标识。

    Returns:
        `JSONResponse`: ``{removed, trace_id}``（all=true）
        或 ``{forgotten, trace_id}``（单条）。

    ⚠️ 两种模式**互斥**且都要求显式给出：既不给请求体也不给 ``all=true``
        时返回 400 而不是「什么都不做」——后者会让一个拼错参数的调用方
        以为删成功了。两者同时给出时也返回 400：无法判断调用方到底想删什么，
        而删除是不可撤销的，这时候「猜一个」是最坏的选择。
    """
    memory = _memory_of(request)
    if memory is None or not memory.enabled or not memory.has_semantic:
        return _unavailable(memory, "semantic")

    if all_notes and payload is not None:
        return _error(
            400,
            "all=true 与请求体同时给出了，无法判断要删什么。"
            "请二选一：清空全部用 all=true，删单条在请求体里给出 text。",
        )
    if not all_notes and payload is None:
        return _error(
            400,
            "没有给出要删的内容。请二选一：清空全部用 ?all=true，"
            '删单条在请求体里给出 {"text": "..."}。',
        )

    if all_notes:
        try:
            removed = await memory.forget_all(user_id)
        except Exception as exc:  # noqa: BLE001 —— 删除失败必须说出来（含脱敏）
            return _write_failed("清空记忆", exc)
        return JSONResponse(
            {"removed": removed, "trace_id": get_trace_id()},
        )

    assert payload is not None  # 上面两个分支已覆盖：到这里它必定存在
    text = payload.text.strip()
    if not text:
        return _error(400, "文本为空（去掉首尾空白后没有任何内容）。")

    try:
        await memory.forget(user_id, text)
    except Exception as exc:  # noqa: BLE001 —— 删除失败必须说出来（含脱敏）
        return _write_failed("忘记笔记", exc)

    return JSONResponse(
        {"forgotten": text, "trace_id": get_trace_id()},
    )


__all__ = ["router"]
