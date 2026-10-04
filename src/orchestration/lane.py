# -*- coding: utf-8 -*-
"""**快慢车道的执行层** —— 把 :mod:`src.orchestration.classifier` 的判定
变成真正省下模型调用的副作用。

文件职责：
    实现 ``LaneRouterMiddleware``。它是博客「快慢车道」在 AgentScope 2.x 下的
    落地形态：慢车道交给框架的 ReAct 循环正常跑；快车道则在 ``on_model_call``
    里**直接短路掉第一次模型调用**，用一条合成本地响应把工具调用「塞」给
    agent，由框架照常执行。

上下游依赖：
    - 上游：:mod:`src.orchestration.classifier`（纯判定）、``agentscope``
      的 ``MiddlewareBase`` / ``ChatResponse`` / ``ToolCallBlock``。
    - 下游：``src/server/agents_factory.py`` 把它装配进 ``create_app``。

═══ 为什么快车道能省下一次模型调用 ═══

一次普通的 ReAct 回复是**两轮**模型调用：

    第 1 轮：模型读用户输入 → 决定调哪个工具（模型调用 #1）
    执行工具
    第 2 轮：模型读工具结果 → 组织自然语言答复（模型调用 #2）

快车道把第 1 轮**整个跳过**：路由规则已经知道该调哪个工具了，没必要再问
模型一遍。于是总共只剩 **1 次**模型调用。P3 的验收断言
「快车道用例未发生第二次模型调用」说的就是这件事 —— 它是**可观测**的，
不是修辞。

═══ ⚠️ 快车道 ≠ 免确认 ═══

这是最容易搞错的一点。快车道省掉的是**路由**那一次模型调用，**不是**
权限校验。合成出来的工具调用会走和模型产出的工具调用**完全相同**的执行
路径 —— 包括 ``on_check_permission``、包括 HITL 的
``RequireUserConfirmEvent``。

所以「一键下单」这类按钮走快车道时，用户该看到的确认弹窗**一个都不会少**。
若哪天发现快车道把确认弹窗绕过去了，那说明有人图省事直接在中间件里执行了
业务逻辑 —— 那是必须修回去的越权，而不是性能优化。

═══ 用到的两个框架事实（均已核实） ═══

1. ``on_model_call`` 的返回值**直接决定**模型那一层拿到什么
   （``agent/_agent.py:3365-3371`` 是 ``return await mw.on_model_call(...)``，
   而调用点上面就是 ``return await model(...)``）。所以只要**不调用**
   ``next_handler``，真实模型就一次都不会被调到。
2. ``ChatResponse`` 是**普通 dataclass**，构造它不需要真的有模型
   （``model/_model_response.py``）。消费侧（``agent/_agent.py:1744-1762``）
   对 ``isinstance(res, ChatResponse)`` 的分支是「直接当作完整响应使用」，
   因此返回一个 ``is_last=True`` 的普通对象即可，**不必**构造异步生成器。
   ``is_last=True`` 是必须的：消费侧靠它判断「这轮模型输出到此结束」。
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentscope.middleware import MiddlewareBase

from src.domain import Intent, LaneName, TripStage

#: 合成响应的默认工具名。
#:
#: ⚠️ 这个名字**必须**和 ``src/tools/`` 里注册的工具名逐字一致，否则框架
#: 会去找一个不存在的工具，报「未知工具」。这条约束写成了
#: ``tests/test_orchestration_lane.py`` 里的一条断言。
DEFAULT_ROUTE_TOOL = "aligo_route_intent"

#: 把路由决策放进 ``agent.state.middle_context`` 时用的键。
#:
#: ⚠️ 带 ``aligo:`` 前缀是刻意的。``middle_context`` 是个**没有命名空间
#: 约束**的裸 dict，框架自己的中间件习惯用「类名」当键（见
#: ``middleware/_base.py:296-303`` 的 ``get_middleware_key``）。本键是
#: **跨中间件共享**的（:mod:`src.orchestration.context` 要读它来决定阶段），
#: 不属于任何单个中间件，所以不能挂在某个类名下面 —— 但也因此必须取一个
#: 不可能与类名撞车的名字。
ROUTE_DECISION_KEY = "aligo:route_decision"

#: 合成工具调用时写入的入参**字段名**（顺序即序列化顺序）。
#:
#: ⚠️ 这是一份**跨模块契约**：本模块按这些字段拼 JSON，而
#: ``src/tools/route.py`` 里的工具函数按同样的名字声明形参。两边对不上时
#: 框架不会报「契约不匹配」，只会把工具抛出的
#: ``got an unexpected keyword argument`` **当成一次普通的工具失败结果**
#: 塞回给模型（实测：错误文本以 ``TOOL_RESULT_TEXT_DELTA`` 的形式出现，
#: 不是异常）。于是快车道看起来「跑通了」，实际上路由信息一个字都没传进去。
#:
#: 所以这里抽成常量，并配一条断言两边一致的测试
#: （``tests/test_tools_route_contract.py``），把静默不一致变成红灯。
ROUTE_TOOL_INPUT_FIELDS: tuple[str, ...] = (
    "intent",
    "matched_rule",
    "target_agents",
    "reason",
)

#: 多智能体**编排类**工具名 —— 快车道的一次性查询里要收窄掉的那些。
#:
#: ⚠️ 名字来自框架（``agentscope.app._tool``），改动会静默失效：收窄表里
#: 写错一个名字，症状是「这个工具照样出现在模型面前」，而**没有任何报错**。
#: ``tests/test_orchestration_lane.py`` 里有一条断言把它们与运行时真实
#: 注入的工具集对照，防止改名后悄悄失效。
#:
#: 这份名单是**穷举**的，不是「常见的那几个」：框架给团队 leader 挂的是
#: ``TeamCreate / AgentCreate / TeamSay / TeamDelete``（外加用户有可邀请
#: 智能体时的 ``AgentInvite``，见 ``app/_service/_toolkit.py:184-198``），
#: 另外规划类的 ``TaskCreate / TaskList / TaskGet / TaskUpdate`` 四个
#: **无条件**注入（同文件 ``:143``）。
#:
#: ⚠️ 漏一个的症状是不对称的：``TaskGet`` 单独出现在一轮里，只是让模型
#: 多问一次「任务到哪一步了」；但如果漏的是 ``TeamCreate``，
#: 建团队那条 50 秒的路径就整个回来了。所以宁可写全 —— 这里多写一个
#: 名字的代价是「少一个用不上的工具」，少写一个的代价是「用户多等 45 秒」。
ORCHESTRATION_TOOLS: frozenset[str] = frozenset(
    {
        "TeamCreate",
        "TeamDelete",
        "AgentCreate",
        "AgentInvite",
        "TeamSay",
        "TaskCreate",
        "TaskList",
        "TaskGet",
        "TaskUpdate",
    },
)

#: 框架自带的**工具管理元工具**名。
#:
#: 只有 ``reset_tools``（``agentscope.tool._builtin.ResetTools``），它让模型
#: 自己开关工具组。名字同样来自框架，同样会静默失效。
#:
#: ⚠️ 单独立一条、不并进 :data:`ORCHESTRATION_TOOLS`，是因为**注入条件不同**：
#: 编排工具是「有团队/规划能力就有」（``app/_service/_toolkit.py:143,184``），
#: 而元工具的注入条件是「**工具组多于一个**」（``tool/_toolkit.py:502-510``），
#: 也就是会话配了模型时才会出现的 ``schedule_tools`` 组。两者混成一条，
#: 将来核对注入条件时必然对不上号。
#:
#: ⚠️ 实测（2026-10-03，用真实的 ``Toolkit`` 装配）：单组时可用工具是
#: ``['dummy']``，加一个 ``schedule_tools`` 组后变成
#: ``['reset_tools', 'dummy', ...]``。也就是说**线上会话里它确实在**。
TOOL_MANAGEMENT_TOOLS: frozenset[str] = frozenset({"reset_tools"})

#: **进程控制**工具名 —— 目前只有 ``ToolStop``（停掉后台任务）。
#:
#: ⚠️ 它由 ``BackgroundTaskManager.list_tools()`` 提供，在
#: ``app/_service/_toolkit.py`` 里**无条件**挂载（不像编排工具要看角色、
#: 也不像元工具要看工具组数量），所以线上每一轮它都在表里。
#:
#: ⚠️ 它与 ``TaskCreate`` 一类同属「自报 ``is_read_only=False`` 但其实是
#: 框架机制」：模型调它时说的「我先把后台那个还在跑的任务停掉」是**过程
#: 独白**，不是给用户的答复。2026-10-03 对抗验证实测：
#: ``_round_needs_recitation(['ToolStop'])`` 返回 True —— 那一轮的文字会
#: 被原样发给用户，正是守卫要消灭的症状，只是换了个工具名。收进本集合后
#: 判据从「是不是写工具」回到「是不是用户可见的写操作」。
PROCESS_CONTROL_TOOLS: frozenset[str] = frozenset({"ToolStop"})

#: 工作区自带的**文件/进程工具**（``workspace/_base.py:554-563`` 的六件套）。
#:
#: ⚠️ 这份名单**只给快车道收窄用**，刻意**不**并进 :data:`INTERNAL_TOOLS`：
#:
#: - 收窄用：快车道是一次业务查询（查政策/查交通），文件与进程工具没有任何
#:   出场理由，留在 schema 里只会让模型有机会去 ``Bash`` 一下；
#: - 守卫**不用**：这六个是真的有副作用的写工具（``is_read_only=False``），
#:   一旦它们出现在某一轮里，本轮文字很可能是「文件已生成」这类用户可见的
#:   交代。守卫的取舍一贯是「宁可留一段独白，不可删一段正文」，所以这里
#:   保持按 ``is_read_only`` 判断的保守路径。
#:
#: ⚠️ 名字来自 ``workspace/_base.py:576-581`` 的类名，改名同样会静默失效。
WORKSPACE_TOOLS: frozenset[str] = frozenset(
    {"Bash", "Edit", "Glob", "Grep", "Read", "Write"},
)

#: 框架**内部机制**工具 = 编排 + 工具管理 + 进程控制。
#:
#: ⚠️ 这个集合有两个读者，两处的判据都必须是同一份名单：
#:
#: 1. :meth:`LaneRouterMiddleware._narrowed_kwargs` —— 快车道的一次性查询里
#:    把它们摘掉（让模型没法用管理工具绕回多轮编排）；
#: 2. ``src/orchestration/reply_guard.py`` 的 ``_round_needs_recitation``
#:    —— 判定「本轮的文字是不是给用户看的复述」时，先把它们排除掉。
#:
#: ⚠️ 第 2 处是**踩出来的**：早先守卫只按工具自报的 ``is_read_only`` 判断，
#: 并想当然地写了一句注释「框架的团队/元工具自报 is_read_only=True」。
#: 实测这句只对**团队**工具成立（``app/_tool/_team_tool_base.py:42`` 是 True），
#: 而 ``TaskCreate / TaskList / TaskGet / TaskUpdate``（``tool/_task/
#: _task_tool_base.py:23``）、``reset_tools``（``tool/_builtin/_meta.py:43``）
#: 与 ``ToolStop`` 全都自报 ``False`` —— 于是模型建任务清单、停后台任务那几轮的
#: **过程独白被原样发给用户**，正是守卫本该消灭的那类文本。
INTERNAL_TOOLS: frozenset[str] = ORCHESTRATION_TOOLS | TOOL_MANAGEMENT_TOOLS | PROCESS_CONTROL_TOOLS

#: 快车道收窄要摘掉的**全部**框架工具 = 内部机制 + 工作区六件套。
#:
#: ⚠️ 与 :data:`INTERNAL_TOOLS` 的差别就是 :data:`WORKSPACE_TOOLS`，
#: 理由写在那份名单的说明里（收窄看「业务意图用不用得上」，
#: 守卫看「本轮有没有用户可见的写操作」，两个判据不同，所以是两份名单）。
LANE_HIDDEN_TOOLS: frozenset[str] = INTERNAL_TOOLS | WORKSPACE_TOOLS

#: 快车道命中后**不再允许**动用编排工具的那些意图。
#:
#: ⚠️ 这份名单是「一次性查询」的集合，不是「所有快车道意图」：
#: ``PLAN_TRIP`` 那类本来就需要多线并行，收窄它会砍掉真正的能力。
#: 加新意图前先问一句 —— 这个意图的用户会等在那里要**一个**结论，
#: 还是期待系统**并行办几件事**？只有前者该进来。
#:
#: 背景（2026-10-03 实测）：问「住宿标准」，前 7 次 4–7 秒直接作答，
#: 第 8 次走了建团队→拉成员→派任务，用了 **50.6 秒**，答案还是同一句话。
#: 只在提示词里写「别建团队」挡不住这种概率性跑偏 ——
#: 这里给的是**确定性**的闸门：工具根本不在候选列表里，模型没法调。
NARROWED_INTENTS: frozenset[Intent] = frozenset(
    {Intent.QUERY_POLICY, Intent.QUERY_ORDER},
)

#: 本模块的日志器。
logger = logging.getLogger(__name__)


def recorded_stage(agent: Any) -> TripStage | None:
    """从 agent 的 ``middle_context`` 里读出当前对话阶段。

    ⚠️ **这是读 ``ROUTE_DECISION_KEY["stage"]`` 的唯一入口。** 该键有两处
    读者（本模块的 :meth:`LaneRouterMiddleware._stage_of` 与
    :func:`src.orchestration.context.default_resolver`），而写入方只有一处
    （:meth:`LaneRouterMiddleware._remember`）。

    写入的是 ``stage.value``（字符串，理由见 :meth:`_remember`），所以每个
    读者都必须还原成枚举 —— 而还原逻辑一旦各写一份就会**漂移**：事实上
    就漂移过一次，``context.default_resolver`` 里的
    ``isinstance(explicit, TripStage)`` 判断在字符串写入之后**永远为假**，
    显式阶段分支成了死代码，阶段静默退化成「按意图猜」。

    所以这里把「读 + 还原」抽成一个函数，两处共用，让漂移在结构上不可能。

    ⚠️ 无法识别的阶段值（旧会话遗留、枚举改名）降级为 ``None`` 而不是抛异常：
    阶段是**附加信息**，为一个陌生字符串中断整轮回复不划算。

    Args:
        agent (`Any`): agent 实例。

    Returns:
        `TripStage | None`: 解析出的阶段；读不到或无法识别时为 ``None``。
    """
    state = getattr(agent, "state", None)
    if state is None:
        return None
    middle = getattr(state, "middle_context", None)
    if not isinstance(middle, dict):
        # ⚠️ 显式判 dict：``middle_context`` 可能被反序列化成别的容器，
        # 对它调用 ``.get`` 会抛 AttributeError —— 而调用方在主链路上。
        return None
    recorded = middle.get(ROUTE_DECISION_KEY)
    if not isinstance(recorded, dict):
        return None

    raw = recorded.get("stage")
    if raw is None:
        return None
    if isinstance(raw, TripStage):
        # 已经还原过（或有别的写入方直接写了枚举），直接用。
        return raw
    try:
        return TripStage(raw)
    except ValueError:
        logger.warning("无法识别的对话阶段 %r，按「阶段未知」处理。", raw)
        return None


class LaneRouterMiddleware(MiddlewareBase):
    """快慢车道路由中间件。

    ⚠️ **必须挂在中间件列表的靠前位置**（``index`` 越小越外层）。理由：
    它要短路的是**模型调用**本身，若排在别的 ``on_model_call`` 中间件后面，
    那些中间件仍然会先跑一遍（比如熔断器会照常记账、trace 会照常开
    span），快车道省下的成本就被它们的开销吃掉了一部分。功能上不算错，
    但「快车道」的意义会打折。

    ⚠️ 本类**是无状态的**，所以同一个实例跨 agent 复用是**安全**的。
    四个实例属性全是构造期定下来的配置
    （``_enabled`` / ``_max_chars`` / ``_route_tool`` / ``_agent_names``，
    实测 ``vars()`` 就这四个），而所有跨轮状态都放在
    ``agent.state.middle_context`` 里（键见 :data:`ROUTE_DECISION_KEY`）。
    「这一轮路由过没有」也是从那里读的（:meth:`_already_routed`），
    不在实例上。

    ⚠️ 这段注释曾经写着「不能跨 agent 复用，因为它按实例缓存
    ``_routed_reply_id`` 之类的字段」—— 那个字段**不存在**，
    全仓库只在注释里出现过。留这条更正，是因为那个说法会诱导下一个人
    继续往实例上加缓存字段（「反正已经是这个模式了」），
    而那恰恰会把一个真正无状态的类变成有状态的。

    复用的安全性由 :meth:`_remember` 保证：它写的是 ``agent`` 自己的
    ``middle_context``，两个 agent 各写各的。
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        max_chars: int = 20,
        route_tool: str = DEFAULT_ROUTE_TOOL,
        agent_names: tuple[str, ...] | None = None,
    ) -> None:
        """初始化。

        Args:
            enabled (`bool`): 是否启用快车道。``False`` 时本中间件完全透明
                （每个钩子都直接透传），用于排障。
            max_chars (`int`): 超过这个长度的输入一律走慢车道。
                与 :func:`src.orchestration.classifier.classify` 的
                ``max_chars`` 同义。
            route_tool (`str`): 合成响应里要调用的工具名。
            agent_names (`tuple[str, ...] | None`): 只对这些名字的 agent
                生效；``None`` 表示对所有 agent 生效。

                ⚠️ 默认值是 ``None`` 而不是「只对主智能体生效」，因为本类
                不该硬编码任何具体 agent 的名字 —— 名字来自运行时装配
                （见 ``src/agents/registry.py``）。但**生产装配时应当显式传入
                只含主智能体的元组**：快车道规则表描述的是「用户点了什么
                按钮」，只有主智能体才会收到这种输入；让子智能体也去匹配
                同一张表，会把「子智能体被要求去检索政策，而它的输入恰好
                是『查询政策』四个字」这种内部调用误判成用户点击，
                从而短路掉子智能体的模型调用。

        Raises:
            ValueError: ``max_chars`` 小于 1。与分类器保持一致：非正的上限
                会让所有输入都走慢车道，快车道形同虚设，而这种「配置写错
                导致功能静默失效」必须在启动时暴露。
        """
        if max_chars < 1:
            raise ValueError("max_chars 至少为 1")
        self._enabled = enabled
        self._max_chars = max_chars
        self._route_tool = route_tool
        self._agent_names = agent_names

    # --------------------------------------------------------------------------
    # 钩子
    # --------------------------------------------------------------------------
    async def on_model_call(
        self,
        agent: Any,
        input_kwargs: dict[str, Any],
        next_handler: Any,
    ) -> Any:
        """拦截模型调用：快车道命中时合成响应，不调用真实模型。

        ⚠️ 三个 ``return`` 分支的语义各不相同，别合并：

        - **不该路由**（子智能体 / 非首轮 / 已在本次回复路由过）
          → 透传，与没装本中间件完全等价；
        - **走慢车道**（规则没命中、是问句、输入过长……）
          → 透传，但**顺手记下判定**，供动态 Prompt 与可观测使用；
        - **走快车道** → **不调用 ``next_handler``**，直接返回合成响应。
          这是唯一一条真实模型不会被调到的分支。

        Args:
            agent (`Any`): 框架传入的 agent 实例。
            input_kwargs (`dict[str, Any]`): 含 ``current_model`` /
                ``messages`` / ``tools`` / ``tool_choice``。
            next_handler (`Any`): 调用链的下一环（**仅关键字参数**）。

        Returns:
            `Any`: ``ChatResponse`` 或 ``AsyncGenerator[ChatResponse, None]``。
        """
        # ⚠️ 阶段**只取一次**，同时喂给分类器与记录。分别取两次看着无害，
        # 但那意味着「分类时用的阶段」与「记下来的阶段」可能不是同一个值
        # （比如两者之间有别的中间件写了 middle_context），
        # 而记录的唯一用途就是解释分类结果 —— 那两者必须同源。
        stage = self._stage_of(agent)
        decision = self._decide(agent, input_kwargs, stage=stage)

        if decision is None:
            # 不该路由。**但**如果本次回复此前已经判过快车道、且意图是
            # 一次性查询，则收窄工具表 —— 见 :meth:`_narrowed_kwargs`。
            # 除工具表外完全透明地透传。
            return await next_handler(**self._narrowed_kwargs(agent, input_kwargs))

        self._remember(agent, decision, stage=stage)

        if decision.lane is not LaneName.FAST:
            return await next_handler(**input_kwargs)

        logger.info(
            "快车道命中：rule=%s intent=%s agent=%s，本轮跳过路由用的模型调用。",
            decision.matched_rule,
            decision.intent.value,
            getattr(agent, "name", "?"),
        )
        return self._synthesize(decision)

    # --------------------------------------------------------------------------
    # 内部：该不该路由
    # --------------------------------------------------------------------------
    def _decide(
        self,
        agent: Any,
        input_kwargs: dict[str, Any],
        *,
        stage: Any = None,
    ) -> Any:
        """判断本轮要不要路由，要的话给出 :class:`RouteDecision`。

        Args:
            agent (`Any`): agent 实例。
            input_kwargs (`dict[str, Any]`): 含 ``messages``。
            stage (`Any`): 当前对话阶段。默认 ``None`` —— 分类器会自行降级，
                阶段只影响 ``reason`` 文案，不参与判定。

        Returns:
            `RouteDecision | None`: ``None`` 表示「本轮不管，直接透传」。
        """
        if not self._enabled:
            return None

        if self._agent_names is not None and getattr(agent, "name", None) not in self._agent_names:
            return None

        state = getattr(agent, "state", None)
        if state is None:
            return None

        # ── 守卫一：只在**本次回复的第一轮推理**上路由 ──
        #
        # 已核实 ``agent/_agent.py:1112-1116``：每次 ``reply`` 开始时
        # ``reply_context`` 被重置成 ``cur_iter=0``；而 ``agent/_agent.py:1269``
        # 在「本轮所有工具调用都已拿到结果」时 ``cur_iter += 1``。
        # 所以快车道合成的工具调用执行完之后，第二轮模型调用的 ``cur_iter``
        # 是 1，不会再次命中这里。
        reply_ctx = getattr(state, "reply_context", None)
        if reply_ctx is None or getattr(reply_ctx, "cur_iter", 0) != 0:
            return None

        # ── 守卫二：本次回复已经路由过了就跳过 ──
        #
        # 与守卫一重叠，但**不是冗余**：守卫一依赖 ``cur_iter`` 的递增时机
        # （框架若把「工具全部完成」的判定改到别处，或某轮工具全部被 HITL
        # 挂起而没递增，守卫一就会失效）。这条只依赖 ``reply_id`` —— 它在
        # 每次 reply 开始时重新生成（``agent/_agent.py:1112``），语义直接就是
        # 「这是不是同一次回复」，不依赖任何中间状态的时序。
        #
        # 没有它，重复命中会让**同一个工具被合成调用两次**：用户点一次
        # 「规划行程」，系统规划两遍 —— 对查询类工具是浪费，对下单类工具
        # 是事故。
        if self._already_routed(agent):
            return None

        text = self._last_user_text(input_kwargs.get("messages"))
        if text is None:
            return None

        # 延迟 import：本模块在**导入期**必须能拿到 classify，但把 import
        # 放在模块顶部会让「lane 依赖 classifier」这条边在阅读时不够醒目 ——
        # 而这条边正是「判定与执行分离」这个设计的全部意义。放这里不影响
        # 任何行为（模块只被 import 一次）。
        from src.orchestration.classifier import classify

        return classify(
            text,
            enabled=True,
            stage=stage,
            max_chars=self._max_chars,
        )

    def _narrowed_kwargs(
        self,
        agent: Any,
        input_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        """按需把框架内部机制工具从本轮模型调用的工具表里摘掉。

        Args:
            agent (`Any`): agent 实例。
            input_kwargs (`dict[str, Any]`): 框架给 ``on_model_call`` 的入参
                （含 ``tools``，形如 ``[{"type": "function", "function": {...}}]``）。

        Returns:
            `dict[str, Any]`: 可能要改 ``tools`` 的入参副本；不需要收窄时
                原样返回入参对象（保持「与没装中间件完全等价」）。

        ⚠️ 摘的是 :data:`LANE_HIDDEN_TOOLS`（编排 + 元工具 + 进程控制 +
        工作区六件套），不是只有编排那几个。``reset_tools`` 也是在线的
        （会话配了模型就会出现 ``schedule_tools`` 组 ⇒ 元工具被注入），
        留着它等于给模型一个「重新开关工具组、再来一轮」的口子 ——
        与「一次调用给结论」相反；``Bash / Write / ToolStop`` 同样没理由
        出现在「查一下住宿标准」这种一次性查询里。

        ⚠️ 只在**同一回复内已经判过快车道、且意图属于** :data:`NARROWED_INTENTS`
        时收窄。这样：
        - 慢车道不受影响（它本来就需要完整工具集去规划行程）；
        - 用户的**下一次**提问重新开始判定，不会因为上一轮查过政策就被永久
          剥夺编队能力。

        ⚠️ 收窄是**真的把工具摘掉**，不是提示词里劝一句。理由：那条 50.6 秒
        的链路是概率性的 —— 同一句话连问 8 次只出现 1 次。概率性的跑偏只能
        用确定性的手段治：工具不在候选列表里，模型就调不出来。

        ⚠️ 摘掉工具**不会**让框架与工具表失配：``tool_choice`` 是 ``auto``，
        而 ``_synthesize`` 合成的路由工具调用发生在本方法的**上游**
        （快车道那一轮根本不调模型）。这一点是核对过 ``_agent.py`` 的
        调用顺序才敢写死的。
        """
        tools = input_kwargs.get("tools")
        if not tools:
            return input_kwargs

        recorded = getattr(getattr(agent, "state", None), "middle_context", {}).get(
            ROUTE_DECISION_KEY
        ) or {}
        # ⚠️ 必须比对 ``reply_id``：``middle_context`` 里的记录是**上一次**
        # 回复留下的，不比对就会把上一轮的快车道判定套用到这一轮
        # （用户上一句是「住宿标准」、这一句是「帮我规划下周去北京」时，
        # 后者会被无端收窄工具）。
        if recorded.get("reply_id") != getattr(agent.state, "reply_id", None):
            return input_kwargs
        if recorded.get("lane") != LaneName.FAST.value:
            return input_kwargs

        intent = recorded.get("intent")
        if not any(intent == candidate.value for candidate in NARROWED_INTENTS):
            return input_kwargs

        kept = [
            schema
            for schema in tools
            if (schema.get("function") or {}).get("name") not in LANE_HIDDEN_TOOLS
        ]
        if len(kept) == len(tools):
            return input_kwargs
        if not kept:
            # ⚠️ 理论上到不了这里（``basic`` 组里一定有我们的业务工具），
            # 但真到了就是「工具表被摘空」—— 模型会在一轮里连一个工具都
            # 看不到。宁可退回不收窄，也不要制造这种状态：症状是模型突然
            # 答不出「住宿标准」，而日志里只有一条「收窄」记录。
            logger.warning(
                "快车道（%s）收窄后工具表为空，本轮放弃收窄（原本 %d 个）。",
                intent,
                len(tools),
            )
            return input_kwargs

        logger.info(
            "快车道（%s）收窄工具表：摘掉 %d 个框架内部工具。",
            intent,
            len(tools) - len(kept),
        )
        return {**input_kwargs, "tools": kept}

    def _already_routed(self, agent: Any) -> bool:
        """本次回复是否已经路由过。

        Args:
            agent (`Any`): agent 实例。

        Returns:
            `bool`: 已路由返回 True。
        """
        reply_id = getattr(agent.state, "reply_id", None)
        if not reply_id:
            return False
        recorded = agent.state.middle_context.get(ROUTE_DECISION_KEY) or {}
        return recorded.get("reply_id") == reply_id

    def _stage_of(self, agent: Any) -> TripStage | None:
        """取当前对话阶段，供分类器判断（问句等不改变车道，但阶段会影响话术）。

        ⚠️ 委托给 :func:`recorded_stage`，**不要**在这里另写一份读取逻辑 ——
        还原成枚举这件事已经在两处读者之间漂移过一次，见该函数的说明。

        ⚠️ 返回的**必须**是枚举而不是字符串。分类器
        :func:`src.orchestration.classifier._with_stage` 会读 ``stage.value``；
        把裸字符串递过去会抛
        ``AttributeError: 'str' object has no attribute 'value'`` ——
        而这发生在**每一次模型调用**的路径上，等于整个对话链路挂掉。

        Returns:
            `TripStage | None`: 拿不到或无法识别时返回 ``None``，
                分类器会自行降级。
        """
        return recorded_stage(agent)

    @staticmethod
    def _last_user_text(messages: Any) -> str | None:
        """从消息列表里取**最后一条用户消息**的纯文本。

        ⚠️ **必须从后往前找，不能看最后一个元素。** 实测框架传进来的
        ``messages`` 形如::

            [0] role=system     text='你是差旅助手。'
            [1] role=user       text='规划行程'
            [2] role=assistant  text=''        ← 最后一个是**空的助手占位**

        框架会先往上下文里塞一条空的 assistant 消息，等着模型的输出填进去，
        再把这个列表交给 ``on_model_call``。所以 ``messages[-1]`` **永远**
        不是用户消息。

        「检查最后一个元素是不是 user」这个写法看起来完全合理，实测却让
        快车道**一次都不会触发**；而症状（「快车道不生效」）看起来像是规则表
        写错了，排查方向会整个跑偏。这条注释就是为了省下那段时间。

        ⚠️ 但**从后往前找**也带来了新风险：第二轮模型调用时，上下文里
        那条用户消息**还在**，找到的仍是同一句。所以 ``cur_iter == 0`` 与
        ``reply_id`` 两个守卫是**必需**的，不是锦上添花 —— 见 :meth:`_decide`。

        ⚠️ 用 ``get_text_content()`` 而不是直接读 ``content``：``Msg.content``
        是**内容块列表**（实测 ``Msg(name="u", content="hi")`` 会直接报
        ``ValidationError``），只有 ``get_text_content()`` 才做「把所有
        TextBlock 拼起来」这件事。

        Args:
            messages (`Any`): 框架传入的消息列表。

        Returns:
            `str | None`: 最后一条用户消息的纯文本；没有用户消息、或文本为空时
                返回 ``None``（调用方据此走慢车道）。
        """
        for message in reversed(list(messages or [])):
            if getattr(message, "role", None) != "user":
                continue
            getter = getattr(message, "get_text_content", None)
            if getter is None:
                return None
            try:
                # 只取**最后一条**用户消息：多轮对话里更早的用户消息属于
                # 历史上下文，拿它们做路由会命中早已过时的意图。
                return getter() or None
            except Exception:  # noqa: BLE001 —— 解析失败就当没有用户输入，走慢车道
                logger.warning("读取用户消息文本失败，本轮不路由。", exc_info=True)
                return None
        return None

    # --------------------------------------------------------------------------
    # 内部：记录与合成
    # --------------------------------------------------------------------------
    def _remember(self, agent: Any, decision: Any, *, stage: Any = None) -> None:
        """把判定写进 ``middle_context``，供动态 Prompt 与可观测读取。

        ⚠️ 记的是 ``lane`` 而不是「是否快车道」的布尔值。慢车道的判定同样
        有价值（`reason` 会说明为什么没走快车道，比如「是问句」），
        排障时这是第一手线索 —— 只记快车道命中，等于把「为什么没命中」
        这个更常见的问题丢掉了。

        ⚠️ 同时把 ``stage`` 记下来，本方法是 ``middle_context["stage"]``
        **唯一的写入方**。此前 :meth:`_stage_of` 与
        :meth:`src.orchestration.context.ContextInjectionMiddleware._resolve`
        都读这个键，却谁都不写 —— 于是「读得到但永远是空」，
        阶段功能静默失效而没有任何报错。

        ⚠️ 写的是 ``stage.value`` 而不是枚举对象本身：``middle_context``
        是个无命名空间的裸 dict，而会话状态可能被序列化（见
        :data:`ROUTE_DECISION_KEY` 的说明）。枚举在 JSON 里会退化成字符串，
        写进去是枚举、读出来是字符串，这种不一致只在重启后才暴露。
        """
        agent.state.middle_context[ROUTE_DECISION_KEY] = {
            "reply_id": getattr(agent.state, "reply_id", None),
            "lane": decision.lane.value,
            "intent": decision.intent.value,
            "matched_rule": decision.matched_rule,
            "target_agents": list(decision.target_agents),
            "reason": decision.reason,
            # ``None`` 是合法取值（阶段未知），不写成 ``"None"`` 字符串 ——
            # 那会让「未知」变成一个看起来像真实阶段的名字。
            "stage": getattr(stage, "value", None) if stage is not None else None,
        }

    def _synthesize(self, decision: Any) -> Any:
        """构造一条「模型决定调用路由工具」的合成响应。

        ⚠️ ``is_last=True`` 是**必须**的：消费侧
        （``agent/_agent.py:1744-1762``）靠它判断「本轮模型输出到此结束」。
        缺了它，框架会认为模型还要继续输出，行为不确定。

        ⚠️ 工具调用的 ``state`` 保持默认的 ``PENDING``，**不要**自作主张
        标成已完成。框架会把这条调用交给 ``on_check_permission`` 与
        ``on_acting`` 走完整流程；预先标成完成会跳过权限判定 ——
        那就是绕过 HITL，是事故而不是优化。

        Args:
            decision (`RouteDecision`): 分类器的判定。

        Returns:
            `ChatResponse`: 合成的模型响应。
        """
        # 延迟 import：``agentscope.model`` 的导入链不轻，而本模块在
        # 「不路由」的快路径上完全用不到它。
        from agentscope.message import ToolCallBlock
        from agentscope.model import ChatResponse

        payload = json.dumps(
            {
                "intent": decision.intent.value,
                "matched_rule": decision.matched_rule,
                "target_agents": list(decision.target_agents),
                "reason": decision.reason,
            },
            # ⚠️ ensure_ascii=False：工具会把这个 JSON 展示给用户，
            # 转义成 \uXXXX 之后中文就成了一串乱码。
            ensure_ascii=False,
        )

        return ChatResponse(
            content=[
                ToolCallBlock(
                    # ⚠️ ``id`` 里带上规则名：这一轮的思考链上会同时出现
                    # 「快车道合成的调用」与「模型自己发的调用」，id 若都用
                    # 随机串，日志里根本分不清哪一条是合成的。
                    id=f"fastlane-{decision.matched_rule or decision.intent.value}",
                    name=self._route_tool,
                    input=payload,
                ),
            ],
            is_last=True,
        )


__all__ = [
    "DEFAULT_ROUTE_TOOL",
    "INTERNAL_TOOLS",
    "LANE_HIDDEN_TOOLS",
    "NARROWED_INTENTS",
    "ORCHESTRATION_TOOLS",
    "PROCESS_CONTROL_TOOLS",
    "ROUTE_DECISION_KEY",
    "ROUTE_TOOL_INPUT_FIELDS",
    "TOOL_MANAGEMENT_TOOLS",
    "WORKSPACE_TOOLS",
    "LaneRouterMiddleware",
    "recorded_stage",
]
