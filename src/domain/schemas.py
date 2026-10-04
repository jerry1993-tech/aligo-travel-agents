# -*- coding: utf-8 -*-
"""差旅业务的**结构化数据模型**：事项、意图决策、编排选路。

文件职责：
    定义三类「在模块之间传递的结构化数据」：

    1. :class:`TravelRequest` —— 出差事项（用户要收集的那张表）；
    2. :class:`IntentRecognitionResult` —— 意图识别智能体的**结构化输出**；
    3. :class:`RouteDecision` —— 编排层「这一轮走哪条车道、调哪些子智能体」。

上下游依赖：
    - 上游：:mod:`src.domain.enums`、pydantic。
    - 下游：``src/agents/``（当 ``structured_schema``）、``src/orchestration/``、
      ``src/tools/``。

═══ 关键约束：能当 ``structured_schema`` 的模型必须是「LLM 友好」的 ═══

``Agent.reply(..., structured_schema=...)`` 会把模型转成 JSON Schema 交给大模型，
框架随后注册一个 ``GenerateStructuredOutput`` 工具强制模型按 schema 输出
（注册与换装见 ``agentscope/agent/_agent.py:1126-1132``）。
因此写这类模型时有三条硬约束：

1. **字段类型必须是 JSON Schema 能表达的基本类型**（str / int / float / bool /
   list / 嵌套 BaseModel / 枚举）。``dict[str, Any]``、``datetime``、自定义类
   转出来的 schema 要么表达不了、要么模型看不懂。
2. **不要依赖默认值来表达「可选」**。模型看到 ``description`` 里写着「可留空」
   与看到字段真的 Required，行为完全不同 —— 前者经常被省略，后者会被填。
   本文件的做法是：**能省的字段给显式默认值并在 description 里说明**，
   需要模型必须填的则不给默认值。
3. **字段名与 description 都是 Prompt**。模型是照着 schema 的
   ``description`` 来理解字段含义的，所以这里的每一句话都按「写给模型看的
   提示词」标准写，而不是按「写给同事看的注释」标准。

⚠️ 这也是本文件与本项目其他模块**风格不同**的地方：其他模块的注释写给
   人类读者，这里的 ``description`` 同时是写给模型的提示词 —— 所以它们
   刻意写得短、具体、无歧义，并且**不复用**中文行话。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler
from pydantic.json_schema import JsonSchemaValue

from src.domain._doc import apply_description
from src.domain.enums import (
    CabinClass,
    Intent,
    LaneName,
    TransportMode,
    TripStage,
)


class _Schema(BaseModel):
    """本模块所有结构化模型的基类。

    与 ``src/config/schema.py`` 的 ``_StrictModel`` **刻意不同**：
    这里必须是 ``extra="ignore"`` 而不是 ``"forbid"``。

    理由：这些模型有一个用途是**接收大模型的输出**。模型偶尔会多吐一个
    自己发明的字段（尤其在它见过相似 schema 之后）。用 ``forbid`` 的话，
    这一次多余的字段会让整个结构化输出**校验失败**，于是整轮回复报废 ——
    而那个字段往往无伤大雅。``ignore`` 让它静默丢弃，保住主流程。

    ⚠️ 这与配置模型的选择不矛盾，因为两者的失败代价完全不同：
    配置里多一个键说明**人写错了**，必须拦；模型输出里多一个键说明
    **模型自由发挥了**，拦下来只会降低可用性。同一条规则用在不同地方
    是对是错，取决于「谁在写、写错要付什么代价」。

    ⚠️ 这里**刻意不开** ``use_enum_values=True``（框架的事件模型开了，见
    ``agentscope/event/_event.py:72``）。
    开了之后 ``request.stage`` 在运行时是字符串 ``"IDLE"`` 而不是
    ``TripStage.IDLE`` —— 值相等（StrEnum）但 ``isinstance`` 为假，
    于是 ``is_terminal`` 之类的**属性访问会静默退化成 AttributeError**
    或者被 ``getattr(x, "is_terminal", False)`` 兜成 False。业务代码要的是
    「类型稳定」，框架事件要的是「序列化省事」，两者取舍不同。
    """

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: Any,
        handler: GetCoreSchemaHandler,
    ) -> JsonSchemaValue:
        """把本模型的 JSON Schema 描述压成**一行**。

        ⚠️ 这是**第二条**泄漏路径，与 :class:`src.domain.enums._AligoStrEnum`
        堵的那条不是一回事 —— 这条常被漏掉，因为它不显眼：pydantic 会把
        **模型自己**的 docstring 整段放进它 schema 顶层的 ``description``。

        实测（``tests/test_domain_schemas.py``）就是这样发现漏网的：
        枚举的描述已经压成一行了，但 ``TravelRequest`` / ``IntentDecision``
        / ``RouteDecision`` 三个模型的顶层描述仍是各自一千多字的完整
        docstring，里面全是 ⚠️ 与 ``file:line``。

        ⚠️ 泄漏内容比枚举那条更"毒"：模型级 docstring 的正文正是
        **Attributes 段落**，逐字段解释「这个字段该怎么填」—— 而这些字段
        各自已经有 ``Field(description=...)`` 了。于是同一件事被说两遍，
        而且其中一遍是带着开发口吻的（「⚠️ 为什么本方法不读 stage」），
        模型会把这种口吻当成对自己的强调指令。

        Returns:
            `JsonSchemaValue`: 描述被压成一行的模型 schema。
        """
        return apply_description(handler(core_schema), cls.__doc__)


# ==============================================================================
# 一、出差事项（收集表）
# ==============================================================================
class TravelRequest(_Schema):
    """一次出差需要收集的**全部要素**。

    这是「出差事项收集」这条主链路的载体：多轮对话把字段一个个填满，
    填满后进入 :attr:`TripStage.CONFIRMING`。

    ⚠️ 所有字段**都有默认值**，因此 ``TravelRequest()`` 是合法的
    「什么都没收集到」状态。这是刻意的：收集过程本质上是**逐步逼近**，
    每一步都可能只有一部分字段。若把字段做成必填，代码里就会到处是
    ``if request.destination is None`` 之外还得再处理「对象根本造不出来」，
    徒增分支。

    Attributes:
        origin: 出发城市（中文城市名，如「杭州」）。
        destination: 目的城市。
        depart_date: 出发日期，``YYYY-MM-DD``。**字符串而非 date**：
            LLM 输出的日期格式五花八门，用 str 接住再由
            ``src/orchestration/`` 的日期解析统一规整；用 ``datetime.date``
            会让格式不对的模型输出直接校验失败。
        return_date: 返程日期，``YYYY-MM-DD``；单程或未定时留空。
        days: 出差天数（用户常只说天数而不是日期）。
        transport_mode: 交通方式偏好。
        cabin_class: 舱位偏好。
        hotel_required: 是否需要酒店。
        hotel_area: 酒店区域偏好（如「公司附近」「高铁站附近」）。
        travelers: 同行人数（含本人），默认 1。
        purpose: 出差事由。
        budget: 预算上限（元）；0 表示未指定。
        stage: 当前所处的对话阶段。
    """

    origin: str = Field(default="", description="出发城市，例如：杭州")
    destination: str = Field(default="", description="目的城市，例如：北京")
    depart_date: str = Field(
        default="",
        description="出发日期，格式 YYYY-MM-DD；用户未明确时留空",
    )
    return_date: str = Field(
        default="",
        description="返程日期，格式 YYYY-MM-DD；单程或未明确时留空",
    )
    days: int = Field(default=0, ge=0, description="出差天数；未明确时填 0")
    transport_mode: TransportMode = Field(
        default=TransportMode.ANY,
        description="交通方式偏好；用户未指定时填 ANY",
    )
    cabin_class: CabinClass = Field(
        default=CabinClass.ANY,
        description="舱位/座席等级偏好；用户未指定时填 ANY",
    )
    hotel_required: bool = Field(default=False, description="是否需要预订酒店")
    hotel_area: str = Field(default="", description="酒店区域偏好；未指定时留空")
    travelers: int = Field(default=1, ge=1, description="同行人数（含本人）")
    purpose: str = Field(default="", description="出差事由；未指定时留空")
    budget: float = Field(default=0.0, ge=0, description="预算上限（元）；未指定时填 0")
    stage: TripStage = Field(
        default=TripStage.IDLE,
        description="当前对话阶段",
    )

    # --------------------------------------------------------------------------
    # 必填要素（决定「能不能生成方案」）
    # --------------------------------------------------------------------------
    def missing_required(self) -> list[str]:
        """列出**还差哪些必填要素**。

        必填项是「没有它就无法给出任何有意义的行程方案」的最小集合：
        出发地、目的地、以及时间（日期或天数**二者其一**）。

        ⚠️ 为什么时间要「二者其一」而不是都要求：用户说「下周去北京三天」
        时给的是天数，说「3 号到 5 号去北京」时给的是日期 —— 强行要求两者
        齐备会导致追问用户已经回答过的问题，这是对话式收集最招人烦的失败模式。

        ⚠️ 为什么本方法是**纯函数、不读 stage**：阶段是「用户当前在哪一步」，
        必填性是「业务上缺什么」，两者可以不一致（例如用户从 CONFIRMING
        回头改口说「其实我还没定目的地」，此时阶段是 CONFIRMING 但必填缺失）。
        让方法只回答业务问题，调用方自己决定「阶段该不该回退」。

        Returns:
            `list[str]`: 缺失字段的**中文名**列表（直接可拼进追问话术）。
        """
        missing: list[str] = []
        if not self.origin.strip():
            missing.append("出发城市")
        if not self.destination.strip():
            missing.append("目的城市")
        # 时间：日期与天数满足其一即可。
        has_date = bool(self.depart_date.strip())
        has_days = self.days > 0
        if not has_date and not has_days:
            missing.append("出差时间")
        return missing

    def is_complete(self) -> bool:
        """必填要素是否已齐全。

        Returns:
            `bool`: 齐全返回 True。
        """
        return not self.missing_required()

    def merged_with(self, patch: "TravelRequest") -> "TravelRequest":
        """把新收集到的字段合并进当前事项，**返回新对象**。

        合并规则是「**非空覆盖**」：新值非空才覆盖旧值。这条规则直接对应
        对话中的一句常见话术 —— 用户说「改成上海」，这一句里只带了目的地，
        其余字段是空字符串/默认值；若按「整体覆盖」，出发地与日期会被
        一并清空。这是收集类对话最经典的一类数据丢失。

        ⚠️ 判定「空」时**枚举要特殊处理**：``TransportMode.ANY`` 与
        ``CabinClass.ANY`` 的语义是「用户没指定」，因此它们**不覆盖**已有值。
        若按「非空字符串」一刀切，``ANY`` 会被当成有效值覆盖掉用户上轮
        明确说过的「要坐高铁」。

        Args:
            patch (`TravelRequest`): 本轮新解析出的增量。

        Returns:
            `TravelRequest`: 合并后的**新**对象（原对象不变）。
        """
        current = self.model_dump()
        incoming = patch.model_dump()
        for key, value in incoming.items():
            # 阶段永远以最新为准 —— 它不是「收集到的要素」，而是状态机的游标。
            if key == "stage":
                current[key] = value
                continue
            if _is_unspecified(value):
                continue
            current[key] = value
        return TravelRequest(**current)


def _is_unspecified(value: object) -> bool:
    """判断一个增量字段是否**代表「用户这次没提」**。

    Args:
        value (`object`): ``model_dump()`` 出来的字段值。

    Returns:
        `bool`: 视为「未指定」返回 True。

    ⚠️ 这个函数存在的唯一理由是 ``TransportMode.ANY`` / ``CabinClass.ANY``：
    它们是**有值的「未指定」**，与空字符串这种**无值的「未指定」**语义相同
    但写法不同。把它们区分对待，是 :meth:`TravelRequest.merged_with` 不出
    数据丢失 bug 的关键。
    """
    if isinstance(value, str):
        if value.strip() == "":
            return True
        # 两个 ANY 枚举的值恰好都是 "ANY"，与它们的 ``display_name`` 无关。
        return value == "ANY"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # 0 表示「未指定」（days=0 / budget=0.0）——见各自 Field 的 description。
        # ⚠️ bool 要排除在外：它是 int 的子类，hotel_required=False 是
        # **有效值**（用户明确说不要酒店），不能被当成「未指定」。
        return value == 0
    return False


# ==============================================================================
# 二、意图识别输出（给模型的 structured_schema）
# ==============================================================================
class IntentDecision(_Schema):
    """**单个**意图的识别结果。

    ⚠️ 与 :class:`IntentRecognitionResult` 的关系是「一条 vs 一束」：
    用户一句「下周三去北京开会，帮我看看能住哪」同时含 PLANNING 与
    POLICY/酒店两类诉求，博客第 127 行把「多意图识别和分类」列为意图识别
    智能体的第一项职责，所以顶层是 list 而不是单值。

    Attributes:
        intent: 意图类型。
        confidence: 置信度 0~1。**不是装饰**：编排层对低于阈值的意图
            走澄清而不是直接调度（阈值见 ``config/base.yaml`` 的
            ``orchestration.*``）。
        slots: 该意图下抽取到的关键信息（如 ``{"destination": "北京"}``）。
            ⚠️ 值一律用字符串，不用嵌套对象 —— 嵌套对象是结构化输出最容易
            让模型「填一半」的形状，而这里的信息很快会被
            :class:`TravelRequest` 接走并做类型转换，字符串足够且更稳。
        reason: 判定该意图的**一句话依据**，给用户看的（「你说要开会，
            判断为规划行程」）。博客的「显示推理」正是靠这类字段落地。
    """

    intent: Intent = Field(
        description=(
            "识别出的意图类型，只能取以下值之一："
            "PLAN_TRIP=规划出差行程；"
            "APPLY_APPROVAL=提交出差申请单；"
            "QUERY_POLICY=查询差旅政策或报销制度；"
            "QUERY_ORDER=查询已有订单或申请进度；"
            "MODIFY_TRIP=修改已生成的行程；"
            "CANCEL=取消行程或订单；"
            "CHITCHAT=寒暄闲聊；"
            "OTHER=以上都不是"
        ),
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="置信度，0 到 1 之间的小数；不确定时给低于 0.5 的值",
    )
    slots: dict[str, str] = Field(
        default_factory=dict,
        description="该意图下抽取到的关键信息，例如 目的地=北京、出发日期=2026-03-05",
    )
    reason: str = Field(default="", description="给出该判定的一句话依据")


class IntentRecognitionResult(_Schema):
    """意图识别智能体的**完整输出**（两段式结构）。

    博客第 133 行原文：「**显示推理**，输出的两段式结构（推理过程 + JSON 决策），
    提升意图识别准确度。」

    ⚠️ 这两段的**顺序很重要**，不能把 ``reasoning`` 放到后面：自回归模型是
    从左到右生成的，先写推理再下结论 = 让模型「先想后答」（类似
    chain-of-thought）；反过来则是「先答后编理由」，那时候 ``reasoning``
    只是对已定结论的粉饰，对准确率没有帮助。字段顺序在这个模型里不是排版，
    是算法的一部分。

    Attributes:
        reasoning: 推理过程（自然语言，给用户看，也提升模型准确率）。
        rewritten_query: **改写后**的规范查询。博客第 131 行：口语化输入
            要标准化、补全上下文、重组关键信息。下游工具与检索用这个字段，
            而不是原始输入。
        intents: 识别出的意图列表，按置信度从高到低。
        needs_clarification: 是否需要向用户追问。
        clarification_question: 需要追问时的问题原文。
    """

    reasoning: str = Field(
        default="",
        description="先写推理过程：结合上下文说明你为什么这样判断，一两句话即可",
    )
    rewritten_query: str = Field(
        default="",
        description="把用户口语化输入改写为规范、完整、可检索的一句话",
    )
    intents: list[IntentDecision] = Field(
        default_factory=list,
        description="识别出的意图，按置信度从高到低排列；无法归类时给一条 OTHER",
    )
    needs_clarification: bool = Field(
        default=False,
        description="信息不足或有歧义、需要向用户追问时填 true",
    )
    clarification_question: str = Field(
        default="",
        description="needs_clarification 为 true 时，要问用户的那句话",
    )

    # --------------------------------------------------------------------------
    def top_intent(self) -> Intent | None:
        """取置信度最高的意图。

        ⚠️ 用 ``max`` 而不是 ``intents[0]``：虽然 Prompt 要求模型按置信度
        排序，但**不能依赖模型遵守格式约定**来做控制流 —— 一旦它没排序，
        按位置取就会稳定地取到错的那个，而且这种错看起来像「模型判断错了」，
        排查方向会被完全带偏。取最大值则对这个约定免疫。

        Returns:
            `Intent | None`: 最高置信度意图；``intents`` 为空时返回 ``None``。
        """
        if not self.intents:
            return None
        return max(self.intents, key=lambda item: item.confidence).intent


# ==============================================================================
# 三、编排选路结果
# ==============================================================================
class RouteDecision(_Schema):
    """编排层对**这一轮**的处置决定。

    它把「快慢车道」（:class:`~src.domain.enums.LaneName`）与「调哪些子智能体」
    合成一条记录，是 ``src/orchestration/`` 的输出、``src/chains/``（思考链）
    的输入。

    Attributes:
        lane: 走快车道还是慢车道。
        intent: 本轮主意图。
        matched_rule: 快车道命中的规则名（慢车道时为空）。
            ⚠️ 保留**规则名**而不是布尔值，是为了让日志能回答「为什么这轮
            走了快车道」—— 灰度或线上异常时，这一条往往就是根因。
        target_agents: 要调度的子智能体名列表。
        reason: 决策依据的一句话说明（面向用户，进思考链展示）。
    """

    lane: LaneName = Field(description="本轮走快车道还是慢车道")
    intent: Intent = Field(description="本轮的主意图")
    matched_rule: str = Field(default="", description="快车道命中的规则名；慢车道时为空")
    target_agents: list[str] = Field(
        default_factory=list,
        description="需要调度的子智能体名称列表",
    )
    reason: str = Field(default="", description="本次路由决策的一句话依据")


__all__ = [
    "IntentDecision",
    "IntentRecognitionResult",
    "RouteDecision",
    "TravelRequest",
]
