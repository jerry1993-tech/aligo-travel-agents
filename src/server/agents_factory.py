# -*- coding: utf-8 -*-
"""把智能体层**接进框架服务层**的装配模块。

文件职责：
    产出 ``create_app`` 需要的三个扩展参数
    （``extra_agent_tools`` / ``extra_agent_middlewares`` /
    ``custom_subagent_templates``），以及它们依赖的业务仓储。

上下游依赖：
    - 上游：:mod:`src.agents`（注册表、意图识别）、:mod:`src.orchestration`
      （车道、动态 Prompt）、:mod:`src.tools`（业务工具）、
      :mod:`src.llm`（模型与熔断）、:mod:`src.storage.memory`（仓库实现）、
      :mod:`src.config`。
    - 下游：``src/server/app.py`` 的 :func:`~src.server.app.create_root_app`。

═══ ⚠️ 三个扩展点的**调用时机完全不同**，这是本模块最容易搞错的地方 ═══

已逐行核实（``app/_app.py``、``app/_lifespan.py``、``app/_service/_chat.py``）：

    ``extra_agent_tools``         **每次装配工具集**时调用（async）
    ``extra_agent_middlewares``   **每次装配 agent**时调用（async）
    ``custom_subagent_templates`` 应用**构造期**读一次（**不是**可调用对象）

前两个是**工厂**（``AgentToolFactory`` / ``AgentMiddlewareFactory``，
``app/_types.py:21-36``），框架每次请求都会 await 它们；第三个是一个
**静态列表**，框架在 ``create_app`` 里把它转成 dict 存进 ``app.state``。

把静态列表写成工厂（或反过来）在类型上是错的，而且报错点离原因很远：
传一个 dict 给 ``custom_subagent_templates``，框架会去迭代它的**键**
（一堆字符串），然后在 ``t.type`` 上崩 —— 报出来是
``AttributeError: 'str' object has no attribute 'type'``。

═══ 工厂的参数是 ``(user_id, agent_id, session_id)`` ═══

``agent_id`` 是**存储里的 agent id**，不是 ``AgentName`` 的值。
所以「只对主智能体生效」的判定**不能**在这里做 —— 那个判定在
``LaneRouterMiddleware`` 内部，它比较的是 ``agent.name``（运行时名字，
就是 ``main_plan``）。见 :func:`build_middlewares_factory` 的说明。

═══ ⚠️ 额外工具会加进**每一个** agent 的工具集 ═══

``get_toolkit`` 在通用运行路径上被无条件调用，工人（worker）、渠道会话、
定时唤醒都走同一条路（``app/_service/_chat.py:1075-1092``）。也就是说
这里返回的工具**不只是主智能体**能用。

对业务工具来说这是可接受的（子智能体本来也需要查交通、查政策），
但要知道它的存在：给每个 agent 多挂一个工具，就是给每一次模型调用多一段
工具 schema。

═══ ⚠️ 中间件的**作用域**与**顺序**是两件不同的事 ═══

:func:`build_middlewares_factory` 返回的列表被框架加进**每一个** agent
（与工具同一条路径）。所以列表里每一项要么**自带作用域**，要么
**对所有 agent 都安全**：

    · ``LaneRouterMiddleware`` 自带作用域（``agent_names`` 限定主智能体）；
    · :class:`_AgentScopedMiddleware` 把 RAG 收窄到 ``policy_rag``；
    · 熔断 / 动态 Prompt / tracing 对所有 agent 都安全，不需要作用域。

⚠️ 最容易写错的是 RAG：不加作用域就等于给所有 worker 挂上一个它不会用的
``search_knowledge``。而工厂拿到的 ``agent_id`` 是存储 id 而不是 ``AgentName``，
所以作用域只能在**中间件的钩子里**按 ``agent.name`` 判定 —— 取舍见
:class:`_AgentScopedMiddleware`。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from agentscope.app import SubAgentTemplate
from agentscope.middleware import MiddlewareBase, TracingMiddleware
from agentscope.model import ChatModelBase
from agentscope.tool import ToolBase

from src.agents.intent import build_intent_recognizer
from src.agents.prompts import MAIN_AGENT_NAME
from src.agents.registry import build_subagent_templates
from src.config.schema import Settings
from src.domain import AgentName
from src.domain.repository import (
    ApprovalRepository,
    HotelRepository,
    OrderRepository,
    PolicyRepository,
    TransportRepository,
)
from src.knowledge.rag import build_rag_middlewares
from src.llm import (
    build_breaker_middleware,
    build_chat_model,
    build_model_timeout_middleware,
    build_rerank_model,
    get_breaker,
)
from src.memory import TravelerMemory, make_memory_resolver
from src.observability import is_tracing_active
from src.orchestration.context import ContextInjectionMiddleware, default_resolver
from src.orchestration.hint_filter import HintSuppressionMiddleware
from src.orchestration.lane import LaneRouterMiddleware
from src.orchestration.reply_guard import ReplyGuardMiddleware
from src.storage.memory import (
    InMemoryApprovalRepository,
    InMemoryHotelRepository,
    InMemoryOrderRepository,
    InMemoryTransportRepository,
    StaticPolicyRepository,
)
from src.tools import build_business_tools, build_intent_tool

#: 本模块的日志器。
logger = logging.getLogger(__name__)


# ==============================================================================
# 一、业务仓储
# ==============================================================================
@dataclass(frozen=True)
class RepositoryBundle:
    """一组业务仓储。

    ⚠️ 用**一个对象**把五个仓储打包，而不是让调用方分别持有五个引用。
    理由是它们必须**同生共死**：P4 换成 Postgres 实现时，五个仓储要一起
    换成同一个连接池上的实例；分别持有的话，很容易只换了三个，
    而剩下的两个仍然指向内存 —— 那种混用不会报错，
    只会让一部分数据「写进去查不到」。

    ⚠️ 冻结（``frozen=True``）：仓储的**引用**不该在运行期被替换。
    要换实现，重新构造一个 bundle。

    Attributes:
        transport: 交通仓储。
        hotel: 酒店仓储。
        policy: 差标仓储。
        order: 订单仓储。
        approval: 申请单仓储。
    """

    transport: TransportRepository
    hotel: HotelRepository
    policy: PolicyRepository
    order: OrderRepository
    approval: ApprovalRepository


def build_repositories(_settings: Settings | None = None) -> RepositoryBundle:
    """构造业务仓储组。

    ⚠️ P3 阶段是**内存实现**，P4 换成 Postgres。

    ⚠️ 参数 ``_settings`` 现在**用不上**，但刻意留着。Postgres 实现需要
    它（数据库 URL、连接池参数）。留着一个用不上的参数看着多余，
    但它把「换实现时调用方不用改」这件事提前兑现了 —— 否则 P4 会变成
    一次签名变更，而签名变更意味着要同时改装配处与所有测试。

    Args:
        _settings (`Settings | None`): 配置。当前未使用。

    Returns:
        `RepositoryBundle`: 五个内存仓储。
    """
    return RepositoryBundle(
        transport=InMemoryTransportRepository(),
        hotel=InMemoryHotelRepository(),
        policy=StaticPolicyRepository(),
        order=InMemoryOrderRepository(),
        approval=InMemoryApprovalRepository(),
    )


# ==============================================================================
# 二、装配产物
# ==============================================================================
@dataclass(frozen=True)
class AgentWiring:
    """交给 ``create_app`` 的三个扩展参数。

    ⚠️ 三个字段与 ``create_app`` 的三个关键字参数**一一对应**，名字也
    刻意取得一样。这样 ``create_app(**wiring.as_kwargs())`` 一眼能看出
    对应关系，而不用去猜 ``tools_factory`` 到底喂给哪个参数。

    Attributes:
        tools_factory: ``extra_agent_tools`` 的值。
        middlewares_factory: ``extra_agent_middlewares`` 的值。
        subagent_templates: ``custom_subagent_templates`` 的值。
    """

    tools_factory: Any
    middlewares_factory: Any
    subagent_templates: list[SubAgentTemplate] = field(default_factory=list)

    def as_kwargs(self) -> dict[str, Any]:
        """转成 ``create_app`` 的关键字参数。

        ⚠️ 三个键名是 ``create_app`` 的**公开参数名**，不是 ``app.state``
        上的属性名。两者看着像但是两回事 —— ``app.state`` 上的是框架内部
        转出来的形态（比如 ``extra_agent_middlewares`` 存的就是工厂本身，
        而 ``custom_subagent_templates`` 存的是 **dict**）。

        Returns:
            `dict[str, Any]`: 关键字参数字典。
        """
        return {
            "extra_agent_tools": self.tools_factory,
            "extra_agent_middlewares": self.middlewares_factory,
            "custom_subagent_templates": self.subagent_templates,
        }


# ==============================================================================
# 三、工厂
# ==============================================================================
def build_tools_factory(
    *,
    settings: Settings,
    repositories: RepositoryBundle,
    model: ChatModelBase,
) -> Any:
    """构造 ``extra_agent_tools`` 工厂。

    ⚠️ 返回的是一个 **async 可调用对象**，不是列表
    （``AgentToolFactory = Callable[[str, str, str], Awaitable[list[ToolBase]]]``，
    ``app/_types.py:33-36``）。返回列表会让框架在 ``tools += await factory(...)``
    那一行报 ``TypeError: object list can't be used in 'await' expression``。

    ⚠️ 模型在**装配期**构造一次并复用，而不是每次请求构造一次：
    ``ChatModelBase`` 不持有对话状态（状态在 ``Agent`` 上），复用它没有
    串号风险；而每次请求构造会重建内部的 HTTP 客户端连接池 —— 那是最贵
    的一部分，重建等于把连接复用完全浪费掉。

    Args:
        settings (`Settings`): 配置。
        repositories (`RepositoryBundle`): 业务仓储组。
        model (`ChatModelBase`): 模型实例。

    Returns:
        `Any`: ``async (user_id, agent_id, session_id) -> list[ToolBase]``。
    """
    orchestrator = settings.orchestration
    recognizer = build_intent_recognizer(
        model=model,
        max_intents=orchestrator.max_subagent_calls,
        confidence_threshold=orchestrator.intent_confidence_threshold,
    )
    # ⚠️ 意图识别工具在装配期构造**一次**，被所有用户共享。
    #
    # 这与下面那些**按用户构造**的工具（``build_business_tools`` 每次请求
    # 都拿 ``user_id`` 建一批）是刻意的对比，理由分两层：
    #
    # 1. **它不碰任何用户数据**。``build_intent_tool`` 只捕获那个识别器，
    #    不捕获 ``user_id``、也不捕获任何仓储 —— 识别一段文本要什么用户信息？
    #    没有可泄漏的租户数据，就没有「必须按用户隔离」的理由。
    # 2. **识别器本身是无状态的**。``IntentRecognizer`` 只有四个构造期
    #    字段（``_model`` / ``_max_intents`` / ``_confidence_threshold``
    #    / ``_system_prompt``），``recognize()`` **从不写 self**
    #    —— 每次调用现建一个 ``Agent``，对话状态在**那个 Agent** 上，
    #    用完即弃（见 :mod:`src.agents.intent` 的模块文档）。
    #
    # ⚠️ 顺带：``build_intent_tool`` 每次调用都会做一次「注册名与常量一致」
    # 的断言。那个断言属于**启动期检查**，正是「装配期构造一次」的另一个
    # 好处 —— 放进请求路径等于每个请求都重复它一遍。
    intent_tool = build_intent_tool(recognizer)

    async def factory(
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> list[ToolBase]:
        """装配某个用户的工具集。

        ⚠️ ``user_id`` **只**从这里来。它由框架从鉴权结果取出后传入，
        与模型、与请求体都无关 —— 这是多租户隔离的**唯一**支点。
        见 :func:`src.tools.orders.build_order_tools` 的说明。

        ⚠️ ``agent_id`` 与 ``session_id`` 本工厂**不使用**。它们对
        按会话隔离的工具（比如「只在这个会话里可用的临时工具」）有用，
        但本项目的工具都是按**用户**隔离的。留着这两个参数是因为签名由
        框架定死。

        Args:
            user_id (`str`): 当前用户标识。
            agent_id (`str`): 当前 agent 的存储 id。
            session_id (`str`): 当前会话 id。

        Returns:
            `list[ToolBase]`: 该用户可用的工具列表。
        """
        del agent_id, session_id  # 见 docstring：本项目按用户隔离
        return build_business_tools(
            user_id=user_id,
            transport_repo=repositories.transport,
            hotel_repo=repositories.hotel,
            policy_repo=repositories.policy,
            order_repo=repositories.order,
            approval_repo=repositories.approval,
            extra=[intent_tool],
        )

    return factory


#: 是否已经把「trace 到底会不会生效」这条结论打进日志。
#: ⚠️ 中间件工厂是**每个请求**被 await 的，不加这个闸门会把同一条结论
#: 刷满整个日志。进程内说一次就够 —— 它反映的是全局 TracerProvider 的状态，
#: 而那个状态在进程生命周期内只会从「未装配」变成「已装配」（且不可逆，
#: 见 ``src/observability/tracing.py`` 的模块文档）。
_tracing_status_logged = False


def _log_tracing_status(settings: Settings) -> None:
    """把「tracing 会不会真的产生 span」这条结论**说进日志**（进程内一次）。

    ⚠️ 为什么非做不可：``TracingMiddleware`` 在全局 provider 不是 SDK
    ``TracerProvider`` 时会**静默短路** —— 不报错、不打日志、只是把调用
    原样透传。于是「挂了中间件」与「真的有 trace」是两件事，而 Langfuse 里
    一条没有时，光看装配代码会误以为埋点生效了。``/readyz`` 已经能报这件事，
    但**装配现场**也要留一条可追溯的记录。

    ⚠️ 分档而不是一律 WARNING：``trace_exporter=none`` 是本项目的默认档，
    那时「没装」是**设计如此**，用 WARNING 会让每个默认部署都被一条无意义
    的告警刷屏，久而久之没人再看 WARNING。真正需要大声说的是
    「配置想用某个通道，结果没装成」——那才是会让人误判的情形。

    Args:
        settings (`Settings`): 配置（用于区分「默认关闭」与「想开却没装成」）。
    """
    global _tracing_status_logged
    if _tracing_status_logged:
        return
    _tracing_status_logged = True

    if is_tracing_active():
        logger.info(
            "TracingMiddleware 已装配：全局 TracerProvider 就绪，"
            "span 会真实生成并上报。",
        )
        return

    if settings.observability.trace_exporter == "none":
        logger.debug(
            "TracingMiddleware 已装配，但 trace 未启用"
            "（trace_exporter=none，默认档）：框架会静默跳过埋点。",
        )
        return

    # 走到这里 = 配了导出通道（console / otlp）但全局 provider 不是 SDK 的。
    # 后果是**所有**埋点静默失效，Langfuse（或控制台）里一个 span 都没有。
    logger.warning(
        "TracingMiddleware 已装配，但全局 TracerProvider **未就绪**："
        "框架会静默跳过所有埋点，导出目标里不会有任何 span。"
        "配置想用 %r 通道，说明 setup_tracing 没装成 —— "
        "原因见 src/observability/tracing.py 与 /readyz 的 tracing 段。",
        settings.observability.trace_exporter,
    )


class _AgentScopedMiddleware(MiddlewareBase):
    """把**一个**内层中间件的生效范围收窄到指定的 ``agent.name``。

    ⚠️ **为什么需要它**：``extra_agent_middlewares`` 返回的列表会被框架加进
    **每一个** agent 的装配 —— 主智能体、每个 worker、渠道会话、定时唤醒都走
    同一条路（``app/_service/_chat.py:1075-1092``）。RAG 中间件因此会落到所有
    agent 上，而 ``RAGMiddleware`` 的 ``list_tools`` 会给**每个** agent 塞一个
    ``search_knowledge`` 工具：那些根本不查政策的 worker 会看到一个它不会用的
    工具，白白占一段工具 schema，还可能被误调。

    ⚠️ **为什么不在工厂里判定**：工厂拿到的是 ``(user_id, agent_id,
    session_id)``，其中 ``agent_id`` 是**存储 id**、不是 ``AgentName``
    （见模块文档 :33-38）。要把它换成名字，只能每次请求查一次存储 ——
    一笔纯粹为了过滤而付的 I/O；而 ``agent.name`` 在钩子里**现成**就有。

    ⚠️ **取舍一：只放行「能拿到 agent 的钩子」**。``on_reply`` /
    ``on_reasoning`` 的参数里有 ``agent``，能按名字判定；``list_tools``
    **没有** ``agent`` 参数，无从判定。因此本类**不实现** ``list_tools``
    （基类返回空列表），等于把被包装的 RAG 钉在 ``mode="static"`` 上 ——
    静态模式靠 ``on_reasoning`` 注入检索结果、不暴露工具，作用域才真正成立。
    这是刻意的取舍：**宁可要一个能收窄的静态检索，也不要一个会泄漏到
    所有 agent 的 agentic 工具**。

    ⚠️ **取舍二：按名字匹配是「尽力而为」**。worker 的 ``agent.name`` 是主
    智能体在 ``AgentCreate`` 时**指定**的（``app/_tool/_agent_create.py:410``），
    并不是模板类型名。因此名字对不上时 RAG **不生效** —— 这是安全的一侧：
    宁可不查，也不要给一个有知识库的 agent 之外的对象挂上检索。名字集合由
    :func:`build_middlewares_factory` 的 ``rag_agent_names`` 参数给出，需要
    调整时只改那一处。

    Attributes:
        _inner (`MiddlewareBase`): 被包装的中间件（RAG 的产物）。
        _agent_names (`tuple[str, ...]`): 允许生效的 ``agent.name`` 集合。
    """

    def __init__(
        self,
        inner: MiddlewareBase,
        *,
        agent_names: tuple[str, ...],
    ) -> None:
        """初始化。

        Args:
            inner (`MiddlewareBase`): 被包装的中间件。
            agent_names (`tuple[str, ...]`): 允许生效的 ``agent.name`` 集合。
        """
        self._inner = inner
        self._agent_names = tuple(agent_names)

    def _applies(self, agent: Any) -> bool:
        """判断当前 agent 是否在内层中间件的生效范围内。

        ⚠️ 用 ``getattr(agent, "name", None)`` 而不是 ``agent.name``：
        名字缺失时应当**放行到「不生效」**（返回 False 让内层不跑），
        而不是抛 ``AttributeError`` 把整个回复打断。缺名字意味着我们根本
        无法确认它是谁，此时保持「不挂检索」是安全的方向。

        Args:
            agent (`Any`): 当前执行的 agent。

        Returns:
            `bool`: 名字在集合内返回 True。
        """
        return getattr(agent, "name", None) in self._agent_names

    async def on_reply(
        self,
        agent: Any,
        input_kwargs: dict,
        next_handler: Any,
    ) -> Any:
        """按作用域放行 ``on_reply``。范围外**或内层没实现时**直接透传。

        Args:
            agent (`Any`): 当前 agent。
            input_kwargs (`dict`): 回复输入。
            next_handler (`Any`): 下游处理。

        Yields:
            `Any`: 下游产出的事件。
        """
        if not self._applies(agent) or not self._inner.is_implemented("on_reply"):
            async for item in next_handler(**input_kwargs):
                yield item
            return
        async for item in self._inner.on_reply(
            agent=agent,
            input_kwargs=input_kwargs,
            next_handler=next_handler,
        ):
            yield item

    async def on_reasoning(
        self,
        agent: Any,
        input_kwargs: dict,
        next_handler: Any,
    ) -> Any:
        """按作用域放行 ``on_reasoning``。范围外**或内层没实现时**直接透传。

        ⚠️ 这是静态模式 RAG 真正干活的钩子（首次推理步注入检索结果）。
        范围判定**必须在调用内层之前**做 —— 否则每个 agent 都会去检索，
        正是本类要挡住的事。

        ⚠️ ``is_implemented`` 那半条判据是**线上事故换来的**，别删：
        基类的 ``on_reasoning`` 在没有子类覆写时会 ``raise RuntimeError``
        （``middleware/_base.py``），而**本类覆写了它** ⇒ 框架认为这条钩子
        「可用」并调用它 ⇒ 它又无脑转发给内层 ⇒ 一个只实现 ``on_reply``
        的内层中间件（比如回复守卫）会让**每一次回复**都以 error 结束
        （实测症状：`ReplyGuardMiddleware does not implement on_reasoning`，
        8/8 轮全崩，而所有单测全绿 —— 它们直接调 ``on_reply``，从不经过
        这层包装）。框架自己也是用 ``is_implemented`` 过滤钩子的
        （``agent/_agent.py:236``），这里与它保持一致。

        Args:
            agent (`Any`): 当前 agent。
            input_kwargs (`dict`): 推理输入。
            next_handler (`Any`): 下游处理。

        Yields:
            `Any`: 下游产出的事件。
        """
        if not self._applies(agent) or not self._inner.is_implemented(
            "on_reasoning",
        ):
            async for item in next_handler(**input_kwargs):
                yield item
            return
        async for item in self._inner.on_reasoning(
            agent=agent,
            input_kwargs=input_kwargs,
            next_handler=next_handler,
        ):
            yield item


def build_middlewares_factory(
    *,
    settings: Settings,
    memory: "TravelerMemory | None" = None,
    kb_manager: Any | None = None,
    rerank_model: Any | None = None,
    rag_agent_names: tuple[str, ...] = (AgentName.POLICY_RAG.value,),
) -> Any:
    """构造 ``extra_agent_middlewares`` 工厂。

    ⚠️ 同样返回 **async 可调用对象**
    （``AgentMiddlewareFactory``，``app/_types.py:21-27``）。
    框架支持两种签名：三参 ``(user_id, agent_id, session_id)`` 或四参
    （多一个 ``workspace``）。它用 ``inspect.signature().bind()`` 探测
    **一次**（``app/_service/_chat.py:240-253``），所以选一种就固定了。
    本项目用三参 —— 中间件不使用工作区。

    ═══ ⚠️ 返回列表的**顺序就是执行顺序**，这里有硬约束 ═══

    已核实：中间件列表的下标 0 是**最外层**（``agent/_agent.py:219-240``
    按顺序存进各钩子的过滤列表，链式调用按列表顺序）。

    所以顺序是固定的六段（外加可选的 RAG），每一段的位置都有理由
    （第 6 段的「位置无影响」也是理由的一种）：

    1. ``LaneRouterMiddleware`` —— **最外层**。
       它要短路的是**模型调用本身**。排在别的 ``on_model_call`` 后面的话，
       那些中间件仍然会先跑一遍（熔断器照常记账、trace 照常开 span），
       快车道省下的成本被它们的开销吃掉一部分。
    2. ``TracingMiddleware`` —— 在车道**之内**、熔断**之外**。
       ⚠️ 它必须比熔断器更外层：熔断器 open 时会直接抛
       ``CircuitBreakerOpen``，比它外层才能把「这次调用被熔断拒绝了」
       记成一个 error span；躲在熔断内层的话，被拒绝的调用在 trace 里
       完全不存在，而「这段为什么没有 span」恰恰是最需要证据的地方。
       ⚠️ 它又必须比车道更内层：快车道命中时**根本不发生模型调用**，
       若把它排在车道外面，它会为一个不存在的调用开一个 LLM span、
       记下一条「成功的模型调用」—— 一条凭空捏造的记录。
       它同时实现 ``on_reply``（整个回复一个根 span）与 ``on_acting``
       （每次工具执行一个 span），而这两个钩子**只有它**实现，因此无论
       放在哪都不会被别的中间件抢走。
    3. ``BreakerMiddleware`` —— 在 tracing 之内。
       快车道命中时根本不发生模型调用，也就不该记熔断账 ——
       把「没调用」记成「调用成功」会稀释失败率，让熔断点被推迟。
    4. ``ModelTimeoutMiddleware`` —— 在熔断**之内**、上下文注入**之外**。
       ⚠️ 必须在熔断之内：熔断器要「先看到失败」才可能开路，而它短路时
       压根不往下调 —— 把超时放在它外面，等于给一个已经决定不打的调用
       再架一个时钟，纯属多余。反过来，超时发生在它之内时，
       ``ModelCallTimeout`` 会顺着链冒到熔断器，被记成一次下游失败 ——
       这正是「连续卡住几次之后熔断器把下游判成不可用」的成因。
       ⚠️ 它补的是熔断器**管不到**的那一半：卡住不返回的调用永远不会抛
       异常，于是熔断器永远数不到账、请求也永远不返回（详见
       ``src/llm/middleware.py`` 的模块文档）。
    5. ``ContextInjectionMiddleware`` —— **最内层**（``on_system_prompt`` 链尾）。
       ``on_system_prompt`` 是**串行链式**的（不是洋葱式）：后一个中间件
       拿到的 ``current_prompt`` 是前一个的返回值。放链尾意味着它看到的是
       「基础 prompt + 前面所有中间件的产出」，正是
       :func:`~src.orchestration.prompt.build_system_prompt` 期望的输入。
       放前面则会让后续中间件基于「已附加动态段落」的串继续改。
       ⚠️ 「链尾」的准确含义是**最后一个 ``on_system_prompt`` 实现者**，
       而不是「列表最后一格」：框架按 ``is_implemented("on_system_prompt")``
       过滤后才组成串行链（``agent/_agent.py:236``），不实现的中间件
       根本不进去。所以下面两个作用域包装可以安全地排在它后面。
    6. ``HintSuppressionMiddleware`` —— 紧跟上下文注入之后。
       ⚠️ 位置对它**没有语义影响**（它只实现 ``on_reply``，吃事件与位置
       无关），列在这里只是为了让「这一段链上有哪些人」是完整的。
       它拦的是框架与子智能体发往 UI 的 ``HintBlockEvent`` ——
       不加的话，用户会在界面上读到英文 ``<system-reminder>`` 提示词
       （实测，且刷新后仍在）。完整证据链见
       :mod:`src.orchestration.hint_filter`。
    7. 作用域包装段 —— 追加在链尾，由 :class:`_AgentScopedMiddleware` 包住：
       先是**回复守卫**（只给 ``main_plan``，见下），再是（可选的）**RAG**。
       位置对它们**没有语义影响**：两者都只实现 ``on_reply`` /
       ``on_reasoning``，不实现 ``on_system_prompt`` —— 因此不打扰上面
       「ContextInjection 在 on_system_prompt 链尾」那条不变式。
       其余两个钩子上：``on_reasoning`` 只有 RAG 用；``on_reply`` 上
       它们都落在 tracing 的根 span 之内。
       ⚠️ 回复守卫**必须**排在这里（而不是塞进上面那五段）：它要剥掉的
       是「主智能体在工具轮里写的过程话」，而那些话是**模型输出**，
       它无权也不该改动 prompt 与模型调用本身。

    ⚠️ ``LaneRouterMiddleware`` 的 ``agent_names`` **必须**显式传
    ``(MAIN_AGENT_NAME,)``。默认值 ``None`` 是「对所有 agent 生效」，
    而那会让子智能体也去匹配快车道规则表 —— 规则表描述的是「用户点了什么
    按钮」，只有主智能体才会收到这种输入。子智能体被要求去检索政策时，
    它的输入恰好可能是「查询政策」四个字，于是它自己的模型调用被短路，
    而症状是「子智能体什么都没干就返回了」。

    ⚠️ RAG 只在 ``kb_manager`` 非 ``None`` 时装配，且**只对
    ``rag_agent_names`` 里的 agent 生效**（默认 ``policy_rag``）。
    作用域由 :class:`_AgentScopedMiddleware` 在钩子里按 ``agent.name``
    判定 —— 理由与取舍全在那个类的文档里。

    ⚠️ ``rerank_model`` 与 ``rag_agent_names`` 是**两条独立的收窄**：
    前者决定「检索结果要不要重排」，后者决定「哪些 agent 能检索」。
    重排模型本身**不做**作用域收窄 —— 它只被 ``RAGMiddleware`` 在
    ``policy_rag`` 的钩子里调用，而那个中间件已经被 :class:`_AgentScopedMiddleware`
    包住了，再包一层不会多挡住任何东西。

    Args:
        settings (`Settings`): 配置。
        memory (`TravelerMemory | None`): 长期记忆门面（动态 Prompt 用）。
        kb_manager (`Any | None`): 知识库管理器。``None`` 时**不装配 RAG**
            —— 桥接层 :func:`~src.knowledge.rag.build_rag_middlewares` 需要它；
            没有它就是「本进程不接检索」，与 P3 行为一致。
        rerank_model (`Any | None`): 重排模型（``ChatModelBase``，已套
            ``BoundedChatModel`` 截止时间）。``None`` 时**不做重排**，
            检索按向量序返回。⚠️ 由 :func:`build_agent_wiring` 统一构造
            （见 :func:`~src.llm.factory.build_rerank_model`），
            在**这里**不构造：本工厂每个 agent 都会被调一次，
            逐个建模型会造出 N 个 HTTP 客户端，而它们本该是同一个。
        rag_agent_names (`tuple[str, ...]`): RAG 允许生效的 ``agent.name``
            集合。默认只有 ``policy_rag`` —— 见 :class:`_AgentScopedMiddleware`。

    Returns:
        `Any`: ``async (user_id, agent_id, session_id) -> list[MiddlewareBase]``。
    """
    orchestrator = settings.orchestration
    breaker = get_breaker(settings)

    async def factory(
        user_id: str,
        agent_id: str,
        session_id: str,
    ) -> list[MiddlewareBase]:
        """装配一个 agent 的中间件。

        ⚠️ **每次调用都新建中间件实例**。但理由不是「复用不安全」——
        这几个中间件都是**无状态**的，跨轮状态全在
        ``agent.state.middle_context`` 里（见
        :meth:`~src.orchestration.lane.LaneRouterMiddleware` 的类文档），
        复用它们本身是安全的。

        真正的理由有两条：

        1. **配置写在构造参数上**。``_max_chars`` / ``_show_reasoning``
           这些来自 ``settings``，而 ``settings`` 在测试里是逐用例的。
           复用一个装配期建好的实例，就等于把「这次请求用的配置」
           钉死在「进程启动时的配置」上。
        2. 构造几个纯对象**很便宜**（没有 I/O、没有连接池），
           省下来的那点开销买不到任何东西。

        ⚠️ 这段注释曾经写着「``LaneRouterMiddleware`` 按实例缓存
        ``_routed_reply_id``，复用会串号」—— 那个字段**不存在**
        （实测实例属性只有 ``_enabled`` / ``_max_chars`` / ``_route_tool``
        / ``_agent_names`` 四个，全是配置）。留这条更正，是因为照那个
        说法推理会得出「这个类是有状态的」这个错误结论，
        进而给别处也加上「必须每次新建」的伪约束。

        Args:
            user_id (`str`): 当前用户标识。⚠️ **长期画像用它**，见下。
            agent_id (`str`): 当前 agent 的存储 id（本工厂不使用，签名要求）。
            session_id (`str`): 当前会话 id（本工厂不使用，签名要求）。

        Returns:
            `list[MiddlewareBase]`: 顺序有讲究，见 :func:`build_middlewares_factory`。
        """
        del agent_id, session_id  # 见 docstring

        # ⚠️ ``user_id`` **只**在这个闭包里用一次：装配长期画像解析器。
        # 它不是「顺手拿来用」，而是**唯一可信**的来源 ——
        # 它由框架从鉴权结果取出后传入（``app/_service/_chat.py:240-253``），
        # 与请求体、与模型输出都无关。若改成运行时从 agent 上读
        # （名字、某个 context 字段），就会得到「偶尔串到别人画像」的行为，
        # 而串号在多租户系统里是事故。
        resolver: Any = None
        if memory is not None:
            resolver = make_memory_resolver(default_resolver, memory, user_id)

        # ⚠️ trace 会不会真的产生 span，是**装配现场**必须留痕的一件事
        # （静默短路是框架埋点最危险的失效模式）。进程内只打一次。
        _log_tracing_status(settings)

        middlewares: list[MiddlewareBase] = [
            LaneRouterMiddleware(
                enabled=orchestrator.fast_lane_enabled,
                max_chars=orchestrator.fast_lane_max_chars,
                agent_names=(MAIN_AGENT_NAME,),
            ),
            # ⚠️ 位置见 docstring：车道之内（避免为快车道短路的假调用开 span）、
            # 熔断之外（让被熔断拒绝的调用留下 error span）。
            TracingMiddleware(),
            build_breaker_middleware(breaker=breaker),
            # ⚠️ 位置见 docstring：熔断**之内**（熔断器需要先被调用才会记账，
            # 而它短路时压根不该再套一层超时时钟）、上下文注入**之外**
            # （后者只改 prompt，与调用时长无关）。
            build_model_timeout_middleware(timeout=settings.llm.timeout_seconds),
            ContextInjectionMiddleware(
                enabled=orchestrator.dynamic_prompt_enabled,
                show_reasoning=orchestrator.expose_reasoning,
                # ⚠️ ``None`` 会被 ``ContextInjectionMiddleware.__init__``
                # 换成它自己的 ``default_resolver`` —— 也就是说
                # 「没有记忆」时行为与 P3 完全一致，不需要在这里写分支。
                resolver=resolver,
            ),
            # ---- 提示块过滤：把 HintBlockEvent 挡在用户界面之外 ----
            # ⚠️ **不加** ``_AgentScopedMiddleware``：不需要收窄到某个 agent。
            # 提示块对**任何** agent 都是「给模型的」，而每个 agent 的事件
            # 都可能出现在用户的 SSE 流里；只挡 main_plan 的话，
            # 子智能体的提示块照旧漏。
            # ⚠️ 位置**没有语义影响**：它只实现 ``on_reply``，而吃掉事件这件事
            # 与它在链上的位置无关（谁先谁后，事件都是「要么被它吃掉、
            # 要么被它放行」）。放在这里是因为紧邻它要保护的那段
            # —— 「面向用户的输出」这一段。完整理由见
            # :mod:`src.orchestration.hint_filter` 的模块文档。
            HintSuppressionMiddleware(),
            # ---- 回复守卫：只给 main_plan（唯一直接面向用户的 agent）----
            # ⚠️ 必须包在 ``_AgentScopedMiddleware`` 里，理由与 RAG 那条不同：
            # RAG 收窄是因为「子智能体不该去检索」，而这里是因为
            # **子智能体的输出压根不面向用户**（它以 HintBlock 的形式回到
            # 主智能体，由主智能体组织成答复）。让守卫也去处理子智能体，
            # 会把它写给主智能体的「检索结论」当成草稿剥掉 ——
            # 那不是清理，那是把数据删了。
            # ⚠️ 位置：``on_reply`` 这条链上，它在 tracing 的根 span 之内
            # （与 RAG 同理），且**不在** on_model_call 链上 ——
            # 它只搬运事件，不该挡在模型调用前面。
            _AgentScopedMiddleware(
                ReplyGuardMiddleware(
                    enabled=orchestrator.reply_guard_enabled,
                    max_retries=orchestrator.reply_guard_max_retries,
                ),
                agent_names=(MAIN_AGENT_NAME,),
            ),
        ]

        # ---- RAG：只给 policy_rag（作用域由包装中间件在钩子里判定）----
        # ⚠️ ``mode="static"`` 是**刻意**的，不是默认值：agentic 模式把
        # ``search_knowledge`` 工具交给模型，而工具是经 ``list_tools`` 注册的
        # —— 那个方法**拿不到 agent**，无法按 agent 收窄。既然收窄不了，
        # 就不该让每个 worker 都长出一个查询工具（见 _AgentScopedMiddleware）。
        # 静态模式靠 ``on_reasoning`` 注入结果，收窄才真正成立。
        if kb_manager is not None:
            rag_middlewares = await build_rag_middlewares(
                user_id,
                settings,
                kb_manager,
                rerank_model=rerank_model,
                mode="static",
            )
            for inner in rag_middlewares:
                middlewares.append(
                    _AgentScopedMiddleware(inner, agent_names=rag_agent_names),
                )

        return middlewares

    return factory


# ==============================================================================
# 四、总入口
# ==============================================================================
def build_agent_wiring(
    settings: Settings,
    *,
    repositories: RepositoryBundle | None = None,
    model: ChatModelBase | None = None,
    subagent_templates: Iterable[SubAgentTemplate] | None = None,
    memory: "TravelerMemory | None" = None,
    kb_manager: Any | None = None,
) -> AgentWiring:
    """构造完整的智能体装配产物。

    ⚠️ ``model`` 可注入，是为了让**测试**能在不碰真实模型的前提下跑通
    整条装配链（``create_root_app`` 也能被单测调用）。生产路径不传它，
    走 :func:`src.llm.build_chat_model` 的标准构造 —— 那个函数同时负责
    「零密钥降级为 Mock」这个重要行为，绕过它会让测试环境与生产环境
    的模型选择逻辑不一致。

    Args:
        settings (`Settings`): 配置。
        repositories (`RepositoryBundle | None`): 业务仓储组；``None`` 时
            构造内存实现。
        model (`ChatModelBase | None`): 模型；``None`` 时按配置构造。
        subagent_templates (`Iterable[SubAgentTemplate] | None`): 覆盖子
            智能体模板；``None`` 时用注册表生成的默认模板。
        memory (`TravelerMemory | None`): 长期记忆门面。``None`` 时
            **不注入画像** —— 行为与 P3 完全一致（动态 Prompt 只带阶段
            与要素，不带 ``profile_summary``）。
        kb_manager (`Any | None`): 知识库管理器。``None`` 时**不接 RAG**。
            ⚠️ 必须与交给 ``create_app(knowledge_base_manager=...)`` 的是
            **同一个实例**，否则桥接层解析出的 KB 句柄来自另一个存储，
            症状是「列表里看得见、检索时说找不到」。

    Returns:
        `AgentWiring`: 可直接展开给 ``create_app`` 的三个参数。
    """
    repos = repositories if repositories is not None else build_repositories(settings)
    chat_model = model if model is not None else build_chat_model(settings)
    templates = (
        list(subagent_templates)
        if subagent_templates is not None
        else build_subagent_templates()
    )

    # ⚠️ 重排模型在这里**一次性**建好，而不是在 middlewares_factory 里按需建：
    # 那个工厂每个 agent 都会被调用一次，逐个构造会造出 N 个 HTTP 客户端
    # （各自一个连接池），而它们本该是同一个。它也是无状态的，
    # 跨请求复用安全 —— 跨轮状态在 ``agent.state`` 里，不在模型上。
    rerank_model = build_rerank_model(settings, reuse=chat_model)

    logger.info(
        "智能体装配完成：%d 个子智能体模板、快车道 %s、动态 Prompt %s、重排 %s。",
        len(templates),
        "开" if settings.orchestration.fast_lane_enabled else "关",
        "开" if settings.orchestration.dynamic_prompt_enabled else "关",
        "开" if rerank_model is not None else "关",
    )

    return AgentWiring(
        tools_factory=build_tools_factory(
            settings=settings,
            repositories=repos,
            model=chat_model,
        ),
        middlewares_factory=build_middlewares_factory(
            settings=settings,
            memory=memory,
            kb_manager=kb_manager,
            rerank_model=rerank_model,
        ),
        subagent_templates=templates,
    )


__all__ = [
    "AgentWiring",
    "RepositoryBundle",
    "build_agent_wiring",
    "build_middlewares_factory",
    "build_repositories",
    "build_tools_factory",
]
