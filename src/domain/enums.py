# -*- coding: utf-8 -*-
"""差旅业务的**枚举常量**：意图、行程阶段、交通/舱位、订单状态、车道。

文件职责：
    把「系统里出现过一次以上的业务取值」集中成枚举。这些取值同时出现在
    三个地方 —— 结构化输出 schema、工具入参校验、数据库列 —— 枚举是让
    三者**不可能对不上**的唯一办法。

上下游依赖：
    - 上游：仅标准库。
    - 下游：:mod:`src.domain.schemas`、``src/orchestration``、``src/tools``。

═══ 一条贯穿全文件的约定：枚举值一律用**大写下划线英文** ═══

例如 :attr:`Intent.PLAN_TRIP` 的值是 ``"PLAN_TRIP"`` 而不是 ``"规划行程"``。
理由不是审美，是三类具体的故障：

1. **模型输出的稳定性**。结构化输出里让模型选 ``"PLAN_TRIP"``，它是一个
   词表里挑一个 token 的动作；让它输出中文短语，同一语义可以有「规划行程 /
   制定行程 / 安排行程」多种写法，schema 校验会随机失败。中文只出现在
   **给用户看的文案**里（见 :attr:`Intent.display_name`）。

2. **数据库与日志的可检索性**。中文字段值在 ``LIKE`` 查询、URL 路径、
   日志检索里都要处理编码与大小写变体。

3. **与前端 TypeScript 枚举逐字对齐**。前端 ``@agentscope-ai/agentscope``
   的事件与块类型全是英文大写；业务枚举跟着走，前后端就只需要一份词表。

于是本文件里每个枚举都提供 ``display_name`` 属性做**展示层翻译** ——
需要中文的地方调它，而不是把中文写回 ``value``。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic.json_schema import JsonSchemaValue

from src.domain._doc import apply_description


class _AligoStrEnum(StrEnum):
    """本模块所有业务枚举的基类 —— 让 JSON Schema 里**只出现一行**说明。

    为什么需要这个基类（这不是风格问题，是**提示词体积与噪声**问题）：
        pydantic 生成 JSON Schema 时，会把枚举类的 ``__doc__`` **整段**塞进
        ``description``。本项目的文档风格是「长篇解释为什么」，于是一个
        :class:`Intent` 的 docstring 有两千多字，其中含 ⚠️ 标记、``file:line``
        引用、反引号代码片段 —— 这些**全都会**被 :class:`src.domain.schemas.
        IntentRecognitionResult` 带进交给大模型的 JSON Schema 里。

        后果有两层，第二层更要命：

        1. **浪费**。结构化输出的 schema 每轮对话都要随请求发一次。
        2. **误导**。schema 的 ``description`` 在模型眼里是**指令**；里面有
           「⚠️ ``OTHER`` 必须存在且必须由代码兜底」这种写给**开发者**的话，
           模型会把它当成对自己的要求，从而过度使用 ``OTHER``。

        也就是说：给同事看的注释与给模型看的提示词，**必须分开**。

    做法：
        只取 docstring 的**第一行**当 schema 描述（本项目每个枚举的第一行
        都是一句干净的概括），并去掉 Markdown 的 ``**`` 强调标记。
        完整 docstring 原样保留 —— 开发者在源码里、在 ``help()`` 里看到的
        仍然是全量解释。压缩逻辑在 :mod:`src.domain._doc`。

    ⚠️ 为什么用 ``__get_pydantic_json_schema__`` 而不是给每个字段写
        ``Field(description=...)``：枚举是**被复用的类型**，同一个
        :class:`Intent` 会出现在多个模型里。逐字段覆盖会漏 —— 而漏掉的
        那一处正是长文档泄漏的地方，且不会有人发现（schema 照样合法，
        只是胖）。

    ⚠️ 本类只堵住**枚举**这一条泄漏路径。**模型自己**的 docstring 走的是
        另一条路（直接进它 schema 顶层的 ``description``），由
        :class:`src.domain.schemas._Schema` 负责。两条路径都要堵，
        缺一条就等于没堵 —— 详见 :mod:`src.domain._doc`。
    """

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        core_schema: Any,
        handler: GetCoreSchemaHandler,
    ) -> JsonSchemaValue:
        """改写 pydantic 为枚举生成的 JSON Schema。

        Args:
            core_schema (`Any`): pydantic 的核心 schema（原样透传）。
            handler (`GetCoreSchemaHandler`): 默认生成器。

        Returns:
            `JsonSchemaValue`: 描述被压成一行的 schema。
        """
        return apply_description(handler(core_schema), cls.__doc__)


class Intent(_AligoStrEnum):
    """用户一句话背后的**业务意图**。

    这是意图识别智能体的输出词表（见 :class:`src.domain.schemas.IntentDecision`），
    也是编排层选路的依据。

    ⚠️ 用 ``StrEnum`` 而不是 ``Enum``：值本身就是字符串，序列化（JSON、
    日志、SSE 帧）时不需要额外的 ``.value`` 转换；同时它在 ``json.dumps``
    与 pydantic 里都能当作普通 str 使用 —— 少一层转换就少一处「忘了 .value
    结果存进去一个 ``Intent.PLAN_TRIP`` 对象」的隐患。

    ⚠️ ``OTHER`` 必须存在且**必须由代码兜底**。理由：LLM 面对一个词表时
    总会遇到「都不像」的输入，若不给出口，它会挑一个最接近的 —— 也就是
    **编造意图**。给出 ``OTHER`` 相当于把「我不确定」变成一个合法且可观测的
    输出，编排层据此走澄清话术而不是错误地调用工具。
    """

    PLAN_TRIP = "PLAN_TRIP"
    """规划行程：用户要「出差事项收集 → 行程规划」主链路。慢车道典型输入。"""

    APPLY_APPROVAL = "APPLY_APPROVAL"
    """提交申请：用户要发起出差申请单（走审批流）。"""

    QUERY_POLICY = "QUERY_POLICY"
    """政策问答：差旅标准、报销规则、企业制度。走 RAG（P4）。"""

    QUERY_ORDER = "QUERY_ORDER"
    """订单查询：查已有订单/申请的进度与详情。"""

    MODIFY_TRIP = "MODIFY_TRIP"
    """修改行程：对已生成的方案做局部调整（改签、换酒店、改时间）。"""

    CANCEL = "CANCEL"
    """取消：取消行程、订单或申请。**必须走人工确认**，不可静默执行。"""

    CHITCHAT = "CHITCHAT"
    """闲聊/寒暄：不触发任何子智能体。"""

    OTHER = "OTHER"
    """兜底：无法归入以上任何一类。编排层应就此**追问澄清**而非猜测。"""

    @property
    def display_name(self) -> str:
        """面向用户的中文名。

        Returns:
            `str`: 该意图的中文展示名；未登记时回落到枚举值本身
            （而不是抛 KeyError —— 展示层不该因为新增了一个枚举成员就 500）。
        """
        return _INTENT_DISPLAY_NAMES.get(self.value, self.value)


#: 意图 → 中文展示名。
#:
#: 刻意用**模块级字典**而不是 ``match`` 语句或类属性：字典是数据，
#: 可以整体被测试遍历（「每个 Intent 成员都必须在这里有一条」），
#: 而 ``match`` 分支漏一个只会在运行时走到才暴露。
_INTENT_DISPLAY_NAMES: dict[str, str] = {
    "PLAN_TRIP": "规划行程",
    "APPLY_APPROVAL": "提交申请",
    "QUERY_POLICY": "政策问答",
    "QUERY_ORDER": "订单查询",
    "MODIFY_TRIP": "修改行程",
    "CANCEL": "取消行程",
    "CHITCHAT": "闲聊",
    "OTHER": "其他",
}


class LaneName(_AligoStrEnum):
    """快慢车道 —— 博客「第二阶段的主智能体+意图识别架构」的核心概念。

    见 ``docs/博客原文-Alibaba-Business-Travel.md`` 第 137-160 行：
    用户点「为我规划行程」这类界面按钮时，走完整 LLM 分析纯属浪费，
    于是分成两条路。
    """

    FAST = "FAST"
    """快车道：规则引擎命中，**不调用大模型**直接路由。

    判据与规则表见 ``src/orchestration/classifier.py``。
    """

    SLOW = "SLOW"
    """慢车道：交给意图识别智能体做语义理解（多意图、消歧、改写）。"""


class AgentName(_AligoStrEnum):
    """本项目**子智能体的规范名**。

    ⚠️ 为什么这份「智能体名册」放在 domain 而不是 ``src/agents/``：

    名字是**跨层共享词汇**。至少有三处要拼写它们，而且必须逐字一致：

    1. ``src/orchestration/classifier.py`` —— 规则表写「这个意图该调谁」；
    2. ``src/agents/registry.py`` —— 注册表按名字装配出智能体实例；
    3. ``src/chains/collector.py`` —— 思考链按名字给用户显示「正在做什么」。

    名字定义在任何一层内部，另外两层就得反向依赖它。放在 domain 里，
    三方都只依赖 domain —— 与本项目 :class:`Intent` 的处理方式一致。

    ⚠️ 用 ``StrEnum`` 而字段类型仍然声明为 ``list[str]``（见
    :attr:`src.domain.schemas.RouteDecision.target_agents`）：值是字符串，
    所以枚举能直接进列表、能直接序列化；而声明成 ``list[str]`` 让
    「没匹配到任何专门智能体」时留空列表是自然写法，不必造一个 ``NONE`` 成员。

    ⚠️ 成员值用**小写下划线**，与本文件其他枚举的大写风格不同。这是刻意的：
    这些名字会出现在**日志、trace 的 span 名、URL 路径**里，小写形式是这些
    场景的通行写法（``main_plan`` 而非 ``MAIN_PLAN``）。而
    :class:`Intent` 走的是「模型在词表里挑 token」的链路，大写更醒目。
    两种风格各自服务于各自的下游。
    """

    MAIN_PLAN = "main_plan"
    """主规划智能体：出差事项收集 + 行程方案生成的主链路（慢车道的主角）。"""

    INTENT = "intent"
    """意图识别智能体：无状态，一句话进、结构化意图出。"""

    POLICY_RAG = "policy_rag"
    """政策问答智能体：基于知识库检索回答差旅标准与报销规则（P4 接 Milvus）。"""

    APPROVAL = "approval"
    """申请单智能体：把行程转成出差申请单并走审批流。"""

    ORDER_QUERY = "order_query"
    """订单查询智能体：查已有订单/申请的进度与详情。"""

    @property
    def display_name(self) -> str:
        """面向用户的中文名。

        Returns:
            `str`: 中文展示名；未登记时回落到枚举值本身（展示层不该 500）。
        """
        return _AGENT_DISPLAY_NAMES.get(self.value, self.value)


#: 智能体名 → 中文展示名。与 :data:`_INTENT_DISPLAY_NAMES` 同理，
#: 用模块级字典是为了能被测试整体遍历，防止新增成员时漏登记。
_AGENT_DISPLAY_NAMES: dict[str, str] = {
    "main_plan": "行程规划",
    "intent": "意图识别",
    "policy_rag": "政策问答",
    "approval": "出差申请",
    "order_query": "订单查询",
}


class TripStage(_AligoStrEnum):
    """出差**事项收集**的对话阶段（博客所谓的「动态 Prompt 状态机」）。

    博客第 361-373 行把这套机制描述为「为 AI 构建一个状态机：通过工程手段
    结合自然语言理解与程序化状态控制，精准识别用户所处的对话阶段，
    并将模型注意力聚焦于当前主链路」。

    ⚠️ 顺序有意义：:attr:`COLLECTING` 之后的每个阶段都假定前一阶段的字段已填。
    ``src/orchestration/prompt.py`` 会读当前阶段，只把**该阶段需要的字段**
    写进动态 Prompt —— 这正是「把注意力聚焦于当前主链路」的工程实现。
    """

    IDLE = "IDLE"
    """尚未开始：用户还没表达出差意图。"""

    COLLECTING = "COLLECTING"
    """收集中：正在补齐出发地/目的地/时间等必填要素（可多轮）。"""

    CONFIRMING = "CONFIRMING"
    """待确认：要素齐全，已生成方案，等用户点头。"""

    DONE = "DONE"
    """已完成：方案已确认（或已下单）。再做修改需显式发起新意图。"""

    CANCELLED = "CANCELLED"
    """已取消：终端态，不再接受修改。"""


class TransportMode(_AligoStrEnum):
    """交通方式。"""

    FLIGHT = "FLIGHT"
    TRAIN = "TRAIN"
    CAR = "CAR"
    """用车/接送机。"""
    ANY = "ANY"
    """不限 —— 用户没说时的默认值。

    ⚠️ 刻意有 ``ANY`` 而不是用 ``None``：``None`` 在结构化输出里表现为
    「模型可以省略这个字段」，而字段一省略，下游就分不清「用户说随便」
    与「模型忘了填」。给一个显式的 `ANY` 让这两种情况可区分。
    """


class CabinClass(_AligoStrEnum):
    """舱位/座席等级。"""

    ECONOMY = "ECONOMY"
    PREMIUM_ECONOMY = "PREMIUM_ECONOMY"
    BUSINESS = "BUSINESS"
    FIRST = "FIRST"
    ANY = "ANY"

    @property
    def display_name(self) -> str:
        """面向用户的中文名。

        Returns:
            `str`: 中文展示名；未登记时回落到枚举值本身（展示层不该 500）。

        ⚠️ 缺了本属性时，差标核对会直接对用户说出英文枚举 ——
        ``check_cabin`` 的原因文案原本是
        「BUSINESS 超出差标允许的最高舱位 ECONOMY」，
        出现在正文与卡片上。这是 2026-10-03 实测到的可用性缺陷。
        """
        return _CABIN_DISPLAY_NAMES.get(self.value, self.value)


#: 舱位 → 中文展示名。与 :data:`_INTENT_DISPLAY_NAMES` 同理，
#: 用模块级字典是为了能被测试整体遍历，防止新增成员时漏登记。
_CABIN_DISPLAY_NAMES: dict[str, str] = {
    "ECONOMY": "经济舱",
    "PREMIUM_ECONOMY": "超级经济舱",
    "BUSINESS": "商务舱",
    "FIRST": "头等舱",
    "ANY": "不限舱位",
}


class OrderStatus(_AligoStrEnum):
    """订单状态机。

    ⚠️ 状态迁移规则**不在本枚举里**，在 ``src/domain/rules.py``（P3 交付）。
    枚举只定义「有哪些状态」，合法迁移是另一件事 —— 把它们混在一起，
    会让「给枚举加一个成员」这种无害操作看起来像改动了业务规则。
    """

    DRAFT = "DRAFT"
    """草稿：已生成但未提交。"""

    PENDING_APPROVAL = "PENDING_APPROVAL"
    """待审批：已提交申请单，等审批人处理。"""

    APPROVED = "APPROVED"
    """已通过：审批通过，可下单。"""

    REJECTED = "REJECTED"
    """已驳回：审批未通过（终端态）。"""

    PAID = "PAID"
    """已支付：出票成功。"""

    CANCELLED = "CANCELLED"
    """已取消（终端态）。"""

    COMPLETED = "COMPLETED"
    """已完成：行程结束、订单闭环（终端态）。"""

    @property
    def display_name(self) -> str:
        """面向用户的中文名。

        Returns:
            `str`: 中文展示名；未登记时回落到枚举值本身（展示层不该 500）。

        ⚠️ 缺了本属性时，订单卡片与提交回执会把 ``PENDING_APPROVAL``
        这类英文状态码印给用户 —— 2026-10-03 实测到
        「当前状态「PENDING_APPROVAL」」直接出现在回复正文里。
        """
        return _ORDER_STATUS_DISPLAY_NAMES.get(self.value, self.value)

    @property
    def is_terminal(self) -> bool:
        """是否为**终端态**（不可再迁移）。

        Returns:
            `bool`: 终端态返回 True。

        ⚠️ 判据仍然是「代码里写死的集合」而不是「看 rules.py 有没有出边」：
        ``rules.py`` 里的迁移表是**另一份**事实，两者若不一致，应当由
        ``tests/test_domain_rules.py`` 的一致性用例去发现，而不是让本属性
        动态依赖它（那样一个漏写的迁移会让终端态判定也跟着错）。
        """
        return self in _TERMINAL_ORDER_STATUSES


#: 订单状态 → 中文展示名。键同样是 ``value``，理由与 :data:`_CABIN_DISPLAY_NAMES` 一致。
_ORDER_STATUS_DISPLAY_NAMES: dict[str, str] = {
    "DRAFT": "草稿",
    "PENDING_APPROVAL": "待审批",
    "APPROVED": "已通过",
    "REJECTED": "已驳回",
    "PAID": "已支付",
    "CANCELLED": "已取消",
    "COMPLETED": "已完成",
}


#: 订单的终端态集合。
#:
#: ⚠️ 为什么 ``COMPLETED`` 是终端态而 ``PAID`` 不是：``PAID`` 之后还能
#: 走到 ``COMPLETED`` 或 ``CANCELLED``（退票），``COMPLETED`` 之后不能再动。
_TERMINAL_ORDER_STATUSES: frozenset[OrderStatus] = frozenset(
    {
        OrderStatus.REJECTED,
        OrderStatus.CANCELLED,
        OrderStatus.COMPLETED,
    },
)


class TaskState(_AligoStrEnum):
    """思考链里单个任务的执行状态。

    对应博客第 281 行描述的 ``TaskCollector``：
    「管理任务的完整生命周期（PENDING、DOING、DONE、FAILED）」。

    ⚠️ 与 :class:`OrderStatus` 是**两个不同的状态机**，不要合并：
    订单状态是业务事实（持久化、要对账），任务状态是**一次回复内的**UI 状态
    （不落库、回复结束即作废）。合并的后果是「想清掉思考链的临时状态」
    变成了「删业务数据」。
    """

    PENDING = "PENDING"
    """已登记，尚未开始执行。"""

    DOING = "DOING"
    """执行中。"""

    DONE = "DONE"
    """执行成功。"""

    FAILED = "FAILED"
    """执行失败。"""


__all__ = [
    "AgentName",
    "CabinClass",
    "Intent",
    "LaneName",
    "OrderStatus",
    "TaskState",
    "TransportMode",
    "TripStage",
]
