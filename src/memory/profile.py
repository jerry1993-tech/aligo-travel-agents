# -*- coding: utf-8 -*-
"""差旅画像的**结构化**部分：有名字、有类型、可校验的那些事实。

═══ ⚠️ 为什么这一半必须结构化，不能全塞进向量库 ═══

一个很自然的想法是「把用户说过的话全丢进向量库，检索出来就够了」。
这条路上有三件事会坏掉：

  1. **精确性问题**。员工的常旅客号是 ``CZ123456789``。
     向量检索是**近似**最近邻 —— 它会把一个**相似但不相等**的号码
     排在前面。而错一位的常旅客号在下单时是致命的：
     要么累积不到里程，要么根本订不上。
  2. **覆盖问题**。「我换座位偏好了」是一句**更新**，
     而向量库的语义是「追加一条相似的记录」。检索时两条都会回来，
     谁新谁旧它不知道 —— 于是模型会看到一个自相矛盾的画像。
  3. **可审计性**。「这个员工为什么被推荐了商务舱」这个问题的答案
     必须是一个可以查的字段，而不是一句「检索相似度 0.87」。

所以本模块管的是**权威的、可更新、可校验**的那部分事实；
:mod:`src.memory.semantic` 管的是「用户随口提过的偏好」那类
模糊的、只作为**提示**使用的信息。两者在
:class:`~src.memory.service.TravelerMemory` 里合并，
且结构化部分的优先级**永远**更高。

═══ ⚠️ 关于存储后端 ═══

本模块用 :class:`Protocol` 定义仓储接口（与
:mod:`src.domain.repository` 同一风格），并提供一个**确定性的内存实现**。

⚠️ 内存实现不是「占位符」，它有一个真实用途：让 ``make test`` 不需要
PostgreSQL 就能覆盖画像的全部逻辑（校验、合并、渲染）。
生产环境的持久化实现（写 ``business`` schema）应当遵循
:mod:`src.storage.engine` 的约定单独实现 —— 见该类文档里
「为什么不做成配置开关」那一段，理由与业务仓储完全一致。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError, field_validator

logger = logging.getLogger(__name__)


# ==============================================================================
# 画像数据模型
# ==============================================================================
class CabinPreference(str):
    """舱位偏好。

    ⚠️ 用 ``str`` 的子类而不是 ``Enum``：值会直接进 Prompt 文本与
    API 响应，继承 ``str`` 让它在序列化时不需要任何转换。
    但**校验**仍然是真的 —— 见 :func:`_validate_cabin`。
    """


#: 合法的舱位偏好值。
#:
#: ⚠️ 与 ``src.domain`` 的舱位枚举**必须**保持一致。写死在这里而不是
#: 直接 import，是因为领域层的舱位是「订单上的舱位」，而这里是
#: 「用户想坐什么」—— 两者今天恰好同构，但语义不同，
#: 将来领域层加了「超级经济舱」不意味着用户的偏好里要有它。
#: 两边不一致时由测试（``tests/test_memory_profile.py``）报出来。
_VALID_CABINS: frozenset[str] = frozenset(
    {"ECONOMY", "PREMIUM_ECONOMY", "BUSINESS", "FIRST"},
)


class ProfileValidationError(ValueError):
    """**调用方**给的画像字段不合法 —— HTTP 400 的唯一依据。

    ⚠️ 为什么不直接用 ``ValueError``（它确实是本类的基类）：因为
    400 与 503 的区别是**给调用方的行动指令** —— 400 说「改请求」，
    503 说「服务端的问题，可以重试」。这个判断必须由**我们**做出，
    而 `except ValueError` 做不到：它同时接住 pydantic 的
    ``ValidationError``（校验**库里读出来的**记录时也会抛，那是
    数据损坏⇒503），以及任何驱动/底层库碰巧抛出的 ``ValueError``
    —— 于是「服务端故障」被报成「你的请求有问题」，调用方改了
    半天请求也不会好，还不会重试。

    ⚠️ 反过来，这个类也是**脱敏的边界**：只有我们自己构造的消息
    才允许原样进响应体（它只回显调用方自己传来的值），
    其余异常一律经 ``safe_error``（见 ``src/server/routers/_memory.py``）。
    """


def _validate_cabin(value: str | None) -> str | None:
    """校验舱位偏好。

    Args:
        value (`str | None`): 待校验的值。

    Returns:
        `str | None`: 规范化（大写）后的值。

    Raises:
        ProfileValidationError: 值不在 :data:`_VALID_CABINS` 中。
    """
    if value is None:
        return None
    normalized = value.strip().upper()
    # ⚠️ 空串（含纯空白）= **清空舱位偏好**，不是「非法值」。
    # 这是 patch 的「空值即清空」约定的一部分（与列表给 ``[]`` 同义）。
    #
    # ⚠️ 早先这里对空串直接抛 ``ProfileValidationError`` ⇒ 400，
    # 而当时接口文案写的是「清空字段给空值（列表给 []、字符串给 ""）」——
    # 于是 ``preferred_cabin`` 成了**唯一一个给了空值反而报错的字符串字段**，
    # 也就成了唯一一个**根本没法清空**的字段（``null`` 是「不改」，
    # 整体覆盖没有 HTTP 入口）。而「我不想再指定舱位了」恰恰是这张表上
    # 最自然的一个诉求。对抗性审核（2026-10-03）把它挑了出来。
    if not normalized:
        return None
    if normalized not in _VALID_CABINS:
        raise ProfileValidationError(
            f"舱位偏好 {value!r} 不合法。"
            f"合法值：{'、'.join(sorted(_VALID_CABINS))}。",
        )
    return normalized


def _format_validation_error(exc: ValidationError) -> str:
    """把 pydantic 的校验错误压成**一句**给调用方看的中文。

    ⚠️ 不用 ``str(exc)``：那是给开发者看的多行报告，末尾还带一条
    ``https://errors.pydantic.dev/...`` 的排错链接 —— 把它塞进 HTTP
    响应的 ``detail`` 里，调用方（前端/curl）会拿到一段带换行的
    字符串，而**只有我们自己的消息**才允许原样出门（见
    :class:`ProfileValidationError`）。这里逐条取出 ``loc`` 与 ``msg``，
    拼成一行。

    Args:
        exc (`ValidationError`): pydantic 抛出的校验错误。

    Returns:
        `str`: 形如 ``preferred_cabin：舱位偏好 '随便' 不合法。…``；
        取不到任何条目时退回 ``str(exc)``（不猜）。
    """
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(item) for item in error.get("loc", ()))
        message = str(error.get("msg", "")).strip()
        # ⚠️ pydantic 会给 validator 里抛出的异常统一加上 ``Value error, ``
        # 前缀（它分不清那也是我们自己写的消息）。留着的话，客户端看到的是
        # 「preferred_cabin：Value error, 舱位偏好 '随便' 不合法。」——
        # 一句夹着框架痕迹的半英文。前缀是**已知且固定**的，剥掉它。
        prefix = "Value error, "
        if message.startswith(prefix):
            message = message[len(prefix):].strip()
        if location and message:
            parts.append(f"{location}：{message}")
        elif message:
            parts.append(message)
    return "；".join(parts) or str(exc)


class TravelerProfile(BaseModel):
    """一位出差员工的差旅画像。

    ⚠️ **所有字段都有默认值**，因为画像必然是**逐步**补全的：
    新员工第一次出差时这些字段一个都没有。把任何一个做成必填，
    都会让「第一次使用」变成一条要走完的填表流程 ——
    而用户来这里是为了订票，不是为了填表。

    ⚠️ 字段分两类，注释里标了：

      · **硬事实**（常旅客号、成本中心）—— 错了会导致下单失败或走错账，
        必须精确匹配，**不**参与语义召回；
      · **软偏好**（座位、酒店品牌）—— 错了只是不舒服，
        可以作为提示喂给模型。

    这个分类决定了 :meth:`~src.memory.service.TravelerMemory.recall`
    怎么用它们。
    """

    user_id: str = Field(description="员工 id（与鉴权层注入的 X-User-ID 同源）。")

    # ---- 硬事实：精确、不可模糊匹配 ----
    frequent_flyer_numbers: dict[str, str] = Field(
        default_factory=dict,
        description="航司代码 → 常旅客号。如 {'CZ': '1234567890'}。",
    )
    cost_center: str | None = Field(
        default=None,
        description="成本中心 / 项目号，下单时写入订单。",
    )
    default_approver: str | None = Field(
        default=None,
        description="默认审批人工号；提交申请时的兜底收件人。",
    )

    # ---- 软偏好：可以作为提示喂给模型 ----
    preferred_airlines: list[str] = Field(
        default_factory=list,
        description="偏好航司代码，按优先级排序。",
    )
    preferred_cabin: str | None = Field(
        default=None,
        description="偏好舱位；受差旅标准约束，见 allowed_cabin。",
    )
    seat_preference: str | None = Field(
        default=None,
        description="座位偏好（如 '靠窗' / '过道'）。自由文本。",
    )
    preferred_hotel_brands: list[str] = Field(
        default_factory=list,
        description="偏好酒店品牌。",
    )
    dietary_needs: list[str] = Field(
        default_factory=list,
        description="饮食需求（如 '素食' / '清真'）。",
    )
    accessibility_needs: str | None = Field(
        default=None,
        description="无障碍需求。⚠️ 这类信息敏感，渲染进 Prompt 时要克制。",
    )

    @field_validator("preferred_cabin")
    @classmethod
    def _check_cabin(cls, value: str | None) -> str | None:
        """校验并规范化舱位偏好。"""
        return _validate_cabin(value)

    @field_validator(
        "preferred_airlines",
        "preferred_hotel_brands",
        "dietary_needs",
    )
    @classmethod
    def _normalize_list(cls, value: list[str]) -> list[str]:
        """去空白、去空项、去重并**保序**。

        ⚠️ 保序是必须的：``preferred_airlines`` 的语义是「按优先级排序」，
        用 ``set`` 去重会让「首选国航」和「首选南航」变成同一个画像 ——
        而这恰恰是用户最在意的那一点。

        ⚠️ 去重时**保留第一次出现的位置**（而不是用 ``sorted(set(...))``）：
        后者会把用户明确表达的优先级顺序打乱成字典序。
        """
        seen: set[str] = set()
        result: list[str] = []
        for item in value:
            cleaned = item.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                result.append(cleaned)
        return result

    def is_empty(self) -> bool:
        """画像是否**完全没有**可用信息。

        ⚠️ 用途是「别再渲染一段空话」：一个空画像渲染出来的会是
        「用户画像：无」这种既占 token 又没有任何信息量的文本。
        ``user_id`` 不计入 —— 它是主键，不是画像内容。

        Returns:
            `bool`: 除 ``user_id`` 外所有字段都是默认值时返回 True。
        """
        return not any(
            (
                self.frequent_flyer_numbers,
                self.cost_center,
                self.default_approver,
                self.preferred_airlines,
                self.preferred_cabin,
                self.seat_preference,
                self.preferred_hotel_brands,
                self.dietary_needs,
                self.accessibility_needs,
            ),
        )


# ==============================================================================
# 仓储
# ==============================================================================
@runtime_checkable
class ProfileRepository(Protocol):
    """画像仓储。

    ⚠️ 定义成 ``Protocol`` 而不是抽象基类，与 :mod:`src.domain.repository`
    同一风格：实现方不需要 import 本模块就能满足契约（结构子类型），
    而生产实现与内存实现之间不必有继承关系。
    ``@runtime_checkable`` 让 ``isinstance`` 可用 —— 装配时的类型自检靠它。
    """

    async def get(self, user_id: str) -> TravelerProfile | None:
        """读画像。

        Args:
            user_id (`str`): 员工 id。

        Returns:
            `TravelerProfile | None`: 画像；没有则 None。
        """
        ...

    async def upsert(self, profile: TravelerProfile) -> TravelerProfile:
        """写画像（整体覆盖）。

        Args:
            profile (`TravelerProfile`): 新画像。

        Returns:
            `TravelerProfile`: 落库后的画像。
        """
        ...

    async def merge(
        self,
        user_id: str,
        patch: "ProfilePatch",
    ) -> TravelerProfile:
        """**部分**更新画像。

        ⚠️ 为什么必须有这个方法，而不是「读出来改完再 upsert」：
        后者是一个 read-modify-write，两个并发请求会互相覆盖
        （经典的 lost update）。症状是「用户刚设的座位偏好，
        下一秒又被另一个字段的更新冲掉了」—— 而且只在并发时出现。

        仓储实现负责让这一步是原子的（Postgres 侧可用
        ``INSERT ... ON CONFLICT DO UPDATE`` 配合 ``jsonb`` 合并）。

        Args:
            user_id (`str`): 员工 id。
            patch (`ProfilePatch`): 要改的字段。

        Returns:
            `TravelerProfile`: 合并后的画像。
        """
        ...


@dataclass(frozen=True)
class ProfilePatch:
    """一次**部分**更新。

    ⚠️ ``None`` 表示「不改这个字段」，而不是「把它设成 None」。
    这个区别在「用户想清空自己的座位偏好」时会有歧义 ——
    本模块的取舍是：**清空要给空值**（列表给 ``[]``、字符串给 ``""``），
    ``None`` 永远只表示「别动它」。

    理由：把「不传」与「传了空」混为一谈，是这类更新接口最常见的 bug，
    而它造成的后果是「更新某一个字段时静默清掉了别的字段」。反过来，
    如果连空值都不认（一律「不传就不动」），用户就再没有清空字段的
    办法了 —— ``put_profile``（整体覆盖）在 ``src/`` 里**没有任何
    HTTP 入口**，所以「清空走整体覆盖」是一句指向不存在能力的文案
    （对抗性审核 2026-10-03 的发现，见 ``tests/test_api_memory.py``
    的 ``test_clearing_a_field_takes_an_empty_value_not_null``）。

    ⚠️ ``frozen=True``：patch 是值对象，被改过的 patch 在日志里
    就对不上它实际做了什么。

    Attributes:
        preferred_cabin: 新的舱位偏好。
        seat_preference: 新的座位偏好。
        preferred_airlines: 新的偏好航司列表（整体替换，不是追加）。
        preferred_hotel_brands: 新的偏好酒店品牌。
        dietary_needs: 新的饮食需求。
        cost_center: 新的成本中心。
        default_approver: 新的默认审批人。
        frequent_flyer_numbers: 要**合并**进现有字典的常旅客号。
        accessibility_needs: 新的无障碍需求。
    """

    preferred_cabin: str | None = None
    seat_preference: str | None = None
    preferred_airlines: list[str] | None = None
    preferred_hotel_brands: list[str] | None = None
    dietary_needs: list[str] | None = None
    cost_center: str | None = None
    default_approver: str | None = None
    frequent_flyer_numbers: dict[str, str] | None = None
    accessibility_needs: str | None = None

    def apply(self, profile: TravelerProfile) -> TravelerProfile:
        """把本 patch 作用到一份画像上，返回**新**画像。

        ⚠️ 不修改入参：``TravelerProfile`` 虽然是可变的 pydantic 模型，
        但在本方法里当值对象用。就地改会让「调用方手里那份画像
        什么时候变的」变得不可追踪 —— 而画像会被并发读取。

        ⚠️ ``frequent_flyer_numbers`` 是**合并**而不是替换：
        用户说「我的南航卡号是 X」时不该把国航的卡号删掉。
        其余列表字段则是**整体替换** —— 「我现在偏好国航」是一个
        完整的表达，而不是往已有列表里追加。

        Args:
            profile (`TravelerProfile`): 原画像。

        Returns:
            `TravelerProfile`: 应用本 patch 后的新画像。
        """
        # ``model_dump`` + ``model_validate`` 做一次深拷贝，
        # 避免改到调用方手上那份（也绕开了 dict/list 的浅拷贝陷阱）。
        data = profile.model_dump()
        for key in (
            "preferred_cabin",
            "seat_preference",
            "preferred_airlines",
            "preferred_hotel_brands",
            "dietary_needs",
            "cost_center",
            "default_approver",
            "accessibility_needs",
        ):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        if self.frequent_flyer_numbers is not None:
            # ⚠️ 合并（而不是替换）之后再过一道「空值即清空」的筛子：
            # ``{"CA": ""}`` 表示**删掉国航的卡号**，与列表给 ``[]``、
            # 字符串给 ``""`` 是同一条约定在字典位置上的形态。
            #
            # ⚠️ 没有这道筛子，``frequent_flyer_numbers`` 就是唯一一个
            # **只能增、不能减**的字段：``{}`` 与「不传」都是「不动」，
            # 于是用户换掉航司之后，旧卡号会永远留在画像里、被注入 Prompt、
            # 甚至被下单工具拿错。对抗性审核（2026-10-03）挑出的死胡同之一。
            merged = {**data["frequent_flyer_numbers"], **self.frequent_flyer_numbers}
            # ⚠️ 判空要**同时**认 ``None`` 与纯空白串，且必须先 ``isinstance``
            # 再 ``.strip()``：``ProfilePatch`` 是普通 dataclass，没有 pydantic
            # 的类型校验，直接 ``number.strip()`` 遇到 ``None`` 会抛
            # ``AttributeError`` ⇒ 500（而它本该是一个 400）。非字符串的
            # 其它类型（如 ``{"CA": 123}``）**不在这里拦**，留给末尾的
            # ``model_validate`` ⇒ ``frequent_flyer_numbers.CA：Input should
            # be a valid string`` 的 400（字段定位是中文的，取自 pydantic
            # 的那半句是英文 —— 内置消息不翻译，只保证定位到字段）。
            data["frequent_flyer_numbers"] = {
                key: number
                for key, number in merged.items()
                if not (
                    number is None
                    or (isinstance(number, str) and not number.strip())
                )
            }
        # ⚠️ 走 ``model_validate`` 而不是直接改属性：舱位、列表的
        # 规范化与校验都在 validator 里，绕过去就等于让 patch 成为
        # 一条能塞进非法值的旁路。
        #
        # ⚠️ 把 pydantic 的 ``ValidationError`` **翻译**成
        # :class:`ProfileValidationError`，这不是包装癖：pydantic 的
        # 异常是 ``ValueError`` 的子类，而 ``TravelerProfile.model_validate``
        # 在**两条语义相反**的路上都会被调用 —— 这里（校验调用方刚传进来的
        # 值 ⇒ 400）与 ``SqlProfileRepository.get``（校验**库里读出来的**
        # 记录 ⇒ 数据损坏，是 503）。异常类型一样，含义相反；在**调用方
        # 值进入系统的这一道门**上翻译一次，下游就只剩一种可能。
        try:
            return TravelerProfile.model_validate(data)
        except ValidationError as exc:
            raise ProfileValidationError(_format_validation_error(exc)) from exc


class InMemoryProfileRepository:
    """**确定性的**内存画像仓储。

    ⚠️ 它存在的理由不是「先凑合」：

      · ``make test`` 不依赖 PostgreSQL（与本项目既有的
        ``src/storage/memory.py`` 同一取舍）；
      · 单测需要**可复现**的画像 —— 从数据库里读会比上一个用例写的
        残留状态，让「单独跑绿、一起跑红」。

    ⚠️ 生产请换成走 ``business`` schema 的实现。它必须自己保证
    :meth:`ProfileRepository.merge` 的原子性 —— 见该方法的文档。
    """

    def __init__(self, seed: dict[str, TravelerProfile] | None = None) -> None:
        """初始化。

        Args:
            seed (`dict[str, TravelerProfile] | None`, optional):
                预置画像（供演示与测试使用）。
        """
        self._profiles: dict[str, TravelerProfile] = dict(seed or {})

    async def get(self, user_id: str) -> TravelerProfile | None:
        """读画像。"""
        profile = self._profiles.get(user_id)
        # ⚠️ 返回**拷贝**：直接把内部对象交出去，调用方一次属性赋值
        # 就改掉了仓储里的状态，而 merge 的原子性保证会因此失效。
        return profile.model_copy(deep=True) if profile is not None else None

    async def upsert(self, profile: TravelerProfile) -> TravelerProfile:
        """写画像（整体覆盖）。"""
        self._profiles[profile.user_id] = profile.model_copy(deep=True)
        return profile.model_copy(deep=True)

    async def merge(
        self,
        user_id: str,
        patch: ProfilePatch,
    ) -> TravelerProfile:
        """部分更新。

        ⚠️ 这里**没有**加锁，因为单进程 asyncio 下从 ``get`` 到 ``upsert``
        之间没有 await（``InMemoryProfileRepository`` 的三个方法都是
        纯同步逻辑），所以不会被打断。生产实现**不能**照抄这一点：
        一旦中间有 I/O，就必须用数据库层的事务或行锁来保证原子性。
        """
        existing = self._profiles.get(user_id)
        if existing is None:
            # ⚠️ 画像不存在时**不是**报错，而是凭空建一份 —— 用户第一次
            # 说「我靠窗」时不该收到「请先创建画像」。
            existing = TravelerProfile(user_id=user_id)
        merged = patch.apply(existing)
        self._profiles[user_id] = merged.model_copy(deep=True)
        return merged.model_copy(deep=True)


__all__ = [
    "InMemoryProfileRepository",
    "ProfilePatch",
    "ProfileRepository",
    "TravelerProfile",
]
