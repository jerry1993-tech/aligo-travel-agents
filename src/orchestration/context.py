# -*- coding: utf-8 -*-
"""**动态 Prompt 的执行层** —— 用 ``on_system_prompt`` 钩子把
:mod:`src.orchestration.prompt` 组装出的内容真正挂进 agent。

文件职责：
    实现 ``ContextInjectionMiddleware``。它是博客 ``get_prompt_main_plan``
    在 AgentScope 2.x 下的现代形态。

上下游依赖：
    - 上游：:mod:`src.orchestration.prompt`（纯组装）、
      :mod:`src.orchestration.lane`（快车道记录的路由决策）、
      ``agentscope`` 的 ``MiddlewareBase``。
    - 下游：``src/server/agents_factory.py`` 把它装配进 ``create_app``。

═══ 为什么是 ``on_system_prompt`` 而不是 ``custom_agent_cls`` ═══

这是 P3 期间**推翻原实施计划**的一处修正，记在这里以免后人再走一遍弯路。

计划书原本写的是用 ``create_app(custom_agent_cls=...)`` 注入自定义 Agent 类。
侦察后确认这条路**不能走**：

1. ``custom_agent_cls`` 是**全局单例**，在 ``create_app`` 时定下，对主智能体
   与 ``AgentCreate`` 产生的所有子智能体**同时生效**，无法按会话区分；
2. 更糟的是它的失败模式：``__init__`` 签名不兼容时，异常会在
   ``app/_service/_chat.py:1177`` 的宽 ``except`` 里被吞掉，**静默退化成
   一条「回复失败」事件** —— 没有堆栈、没有告警，只有用户看到「系统繁忙」。

``on_system_prompt`` 没有这两个问题：它是**按 agent 实例**装配的（走
``extra_agent_middlewares`` 工厂，参数含 ``user_id`` / ``agent_id`` /
``session_id``），且调用点的异常会正常向上抛，不会静默。

═══ 这个钩子的两条硬约束（均已核实） ═══

1. **每轮推理都会被调用一次**（``agent/_agent.py:1728 → 3216 → 3251``）。
   一次回复里模型推理几轮，本钩子就跑几次。所以解析器必须便宜，
   且**结果要按回复缓存** —— 见 :meth:`_resolve`。
2. **返回值是「整个 system prompt 的最终串」**，不是追加内容
   （``agent/_agent.py:3237`` 是 ``result = await mw.on_system_prompt(self, result)``，
   逐个中间件串行链式调用）。所以必须**基于** ``current_prompt`` 拼装 ——
   直接返回自己那一段会把框架的 base prompt、skills、offloader 全部丢掉。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Final

from agentscope.middleware import MiddlewareBase

from src.domain import Intent, TravelRequest, TripStage
from src.orchestration.amounts import numbers_in
from src.orchestration.lane import ROUTE_DECISION_KEY, recorded_stage
from src.orchestration.prompt import build_system_prompt, user_data_excerpt

#: 本模块的日志器。
logger = logging.getLogger(__name__)

#: ``agent.state.middle_context`` 里登记「本轮动态 Prompt 注入了哪些数字」的键。
#:
#: ⚠️ 登记的是**数字**而不是那段文本。理由有两条：
#:
#:   1. **个人数据驻留**：那段文本里可能有用户的预算、出差事由、长期画像，
#:      而 ``middle_context`` 是**随会话持久化**的（``state/_state.py`` 的
#:      ``AgentState.middle_context``）。把数字单独摘出来登记，落库的是
#:      ``["15000", "3"]`` 这种最小必要信息，而不是一句句原话。
#:   2. 消费方（回复守卫的接地闸门）要的本来就是数字 —— 它拿这些数做
#:      「这个数有没有合法来源」的比对（见
#:      :func:`~src.orchestration.reply_guard._ungrounded_limit_claims`）。
#:
#: ⚠️ 同键的写入方只有本模块。读取方目前只有回复守卫。改结构要两边同时改。
PROMPT_NUMBERS_KEY: Final[str] = "prompt_numbers"


def record_prompt_numbers(agent: Any, reply_id: str, numbers: list[str]) -> None:
    """把「我们自己注入进 prompt 的数字」登记到 ``middle_context``。

    Args:
        agent (`Any`): 框架传入的 agent 实例。
        reply_id (`str`): 本次回复的 id（供读取方校验新鲜度）。
        numbers (`list[str]`): 已归一的数字串。

    ⚠️ 整件事**必须不抛异常**：它跑在 ``on_system_prompt`` 的路径上，
    而那条路径抛异常会让整轮回复失败。代价与收益完全不成比例 ——
    登记失败最多让守卫少一条豁免（更保守），不该有任何别的后果。
    调用方（:meth:`ContextInjectionMiddleware.on_system_prompt`）已经包了
    ``try``，这里再自查一遍是因为它还有别的调用方（测试、排障脚本）。

    ⚠️ 只在 ``middle_context`` **已经是 dict** 时写入。该字段是框架的
    持久化字段（默认 ``{}``），但取不到时**不新建** —— 往一个结构未知的
    属性上塞东西，可能被序列化层拒绝，那才是真的把回复搞挂。
    """
    try:
        state = getattr(agent, "state", None)
        middle = getattr(state, "middle_context", None)
        if not isinstance(middle, dict):
            return
        middle[PROMPT_NUMBERS_KEY] = {
            "reply_id": reply_id,
            "numbers": list(numbers),
        }
    except Exception:  # noqa: BLE001 —— 见上，绝不向上抛
        logger.warning("登记动态 Prompt 数字失败，回复守卫少一条豁免。", exc_info=True)


def recorded_prompt_numbers(agent: Any, reply_id: str) -> list[str]:
    """读回 :func:`record_prompt_numbers` 登记的数字。

    Args:
        agent (`Any`): 框架传入的 agent 实例。
        reply_id (`str`): **当前**回复的 id。

    Returns:
        `list[str]`: 已归一的数字串；取不到或**对不上本次回复**时返回空列表。

    ⚠️ 新鲜度校验是**必须**的：``middle_context`` 跨回复存活，而上一条回复
    登记的可能是完全无关的数字（用户上一轮的预算、另一段行情）。拿旧记录
    当豁免来源，等于给闸门开一个「随口报数」的口子 —— 而且它只在「上一条
    回复恰好也走了动态 Prompt」时出现，是最难查的一类间歇性漏网。

    ⚠️ 记录里的 ``reply_id`` 为空、或与传入的不一致时**一律返回空**：
    空列表只会让闸门更保守（多拦一次正确答复），而误用旧记录会**放走
    编造**。两个方向的代价不对称，所以判据往严的一侧倒。
    """
    try:
        state = getattr(agent, "state", None)
        middle = getattr(state, "middle_context", None)
        if not isinstance(middle, dict):
            return []
        recorded = middle.get(PROMPT_NUMBERS_KEY)
        if not isinstance(recorded, dict):
            return []
        stored_reply_id = recorded.get("reply_id") or ""
        if not stored_reply_id or stored_reply_id != reply_id:
            return []
        numbers = recorded.get("numbers")
        if not isinstance(numbers, list):
            return []
        return [str(item) for item in numbers]
    except Exception:  # noqa: BLE001 —— 读取侧同样绝不上抛
        logger.warning("读取动态 Prompt 数字失败，本轮按无豁免处理。", exc_info=True)
        return []


@dataclass(frozen=True)
class PromptContext:
    """组装动态 Prompt 所需的**全部**外部输入。

    ⚠️ 做成一个值对象而不是三个散装参数，是为了让「解析」与「组装」的边界
    显式：:mod:`src.orchestration.prompt` 的 ``build_system_prompt`` 是纯的、
    不做 I/O；而**取这些值**可能需要查库，那件事由 :data:`ContextResolver`
    负责。两者分开之后，纯的那部分可以被穷举测试，脏的那部分只有薄薄一层。

    Attributes:
        stage: 当前对话阶段。
        request: 已收集的出差要素；``None`` 表示本轮拿不到。
        profile_summary: 用户长期画像摘要（P4 的记忆模块提供）；空串表示没有。
    """

    stage: TripStage = TripStage.IDLE
    request: TravelRequest | None = None
    profile_summary: str = ""


#: 从 agent 实例解析出 :class:`PromptContext` 的可调用对象。
#:
#: ⚠️ 这是本模块与外部世界（会话存储、记忆模块）唯一的接缝。默认实现
#: （:func:`default_resolver`）只读 ``agent.state.middle_context``，**不做
#: 任何 I/O**；生产装配时由 ``src/server/agents_factory.py`` 换成会去查
#: 会话存储与长期画像的实现。
#:
#: ⚠️ 实现必须**便宜或自带缓存**：``on_system_prompt`` 每轮推理都调用它
#: （本中间件按回复做了缓存，但跨回复仍会再调一次）。
#:
#: ⚠️ **返回值可以是 awaitable**（P4 放宽）。这条放宽是为了长期画像：
#: 它要查业务库与向量库，两者都是 I/O，而 ``on_system_prompt`` 本身
#: 就是 ``async def``（``agent/_agent.py:3237`` 是 ``await mw.on_system_prompt(...)``），
#: 所以把等待能力放开给解析器不需要改动框架的任何一处。
#:
#: ⚠️ 放宽是**向后兼容**的：既有的同步解析器（:func:`default_resolver`
#: 与它们的单测）一行都不用改 —— :meth:`ContextInjectionMiddleware._resolve`
#: 只在结果确实可等待时才 ``await``。
ContextResolver = Callable[[Any], "PromptContext | Awaitable[PromptContext]"]


#: 意图 → 阶段 的**兜底**映射。
#:
#: ⚠️ 这是一张启发式表，不是业务规则。真正权威的阶段应当来自会话里持久化的
#: ``TravelRequest.stage``；这张表只在「拿不到持久化阶段」时用来给一个
#: 合理的初值。之所以还要有它：快车道短路之后，agent 的状态里只有一条路由
#: 决策，若不借此把阶段推进一步，动态 Prompt 会一直停在 IDLE，用户点了
#: 「规划行程」却得到一句「请问有什么可以帮您」。
#:
#: ⚠️ 只有 ``PLAN_TRIP`` 与 ``MODIFY_TRIP`` 映射到 COLLECTING。其余意图
#: 刻意**不给映射**（查订单、问政策、闲聊都不改变出差收集阶段），
#: 落到 IDLE。给它们硬编一个阶段反而会把用户的行程收集进度冲掉。
_INTENT_HINTS: dict[Intent, TripStage] = {
    Intent.PLAN_TRIP: TripStage.COLLECTING,
    Intent.MODIFY_TRIP: TripStage.COLLECTING,
}


def default_resolver(agent: Any) -> PromptContext:
    """默认解析器：只读 agent 自身的状态，**不做 I/O**。

    取值顺序：

    1. ``middle_context`` 里若已有显式的 ``stage``（由别的组件写入），用它；
    2. 否则看 :mod:`src.orchestration.lane` 记下的路由决策，按
       :data:`_INTENT_HINTS` 推一个阶段；
    3. 都没有就是 :attr:`TripStage.IDLE`。

    ⚠️ 第 1 步优先于第 2 步：显式写入的阶段是**权威**的（可能来自持久化的
    ``TravelRequest``），而按意图推出来的只是兜底。顺序反了会让「用户已经
    填到 CONFIRMING，这一轮又说了句『规划行程』」被打回 COLLECTING，
    表现为方案被反复要求重填。

    ⚠️ ``request`` 恒为 ``None``。默认解析器**没有**能力拿到会话里持久化的
    ``TravelRequest`` —— 那需要查库，而本函数被明确要求不做 I/O。所以
    「已知要素 / 待补齐要素」两节在默认配置下不会出现，只有阶段指令生效。
    生产装配必须换成真正的解析器，否则动态 Prompt 只兑现了一半。
    这里不假装能拿到，而是留空 —— 假装的话，就得编一个空的 ``TravelRequest``，
    模型会看到「已知要素」一节是空的，反而更困惑。

    Args:
        agent (`Any`): 框架传入的 agent 实例。

    Returns:
        `PromptContext`: 解析结果；任何异常都降级为全默认值。
    """
    try:
        # ⚠️ 阶段**必须**经由 :func:`src.orchestration.lane.recorded_stage` 读，
        # 不能在这里自己 ``recorded.get("stage")``。写进去的是
        # ``stage.value``（字符串），自己写一个 ``isinstance(..., TripStage)``
        # 判断会**永远为假** —— 事实上就犯过这个错：显式阶段分支成了死代码，
        # 阶段静默退化成「按意图猜」，没有任何报错。
        explicit = recorded_stage(agent)
        if explicit is not None:
            return PromptContext(stage=explicit)

        middle = getattr(getattr(agent, "state", None), "middle_context", None) or {}
        recorded = middle.get(ROUTE_DECISION_KEY) or {}
        intent = recorded.get("intent")
        if intent is not None:
            try:
                hinted = _INTENT_HINTS.get(Intent(intent))
            except ValueError:
                hinted = None
            if hinted is not None:
                return PromptContext(stage=hinted)

        return PromptContext()
    except Exception:  # noqa: BLE001 —— 解析器跑在主链路上，绝不向上抛
        logger.warning("解析 PromptContext 失败，回退到默认阶段。", exc_info=True)
        return PromptContext()


class ContextInjectionMiddleware(MiddlewareBase):
    """把动态 Prompt 挂进 ``on_system_prompt`` 的中间件。

    ⚠️ 与 ``LaneRouterMiddleware`` 不同，本中间件**必须挂在链尾**
    （``index`` 最大 = 最内层）。``on_system_prompt`` 是**串行链式**的：
    后一个中间件拿到的 ``current_prompt`` 是前一个的返回值。挂在链尾意味着
    它看到的是「基础 prompt + 前面所有中间件的产出」，这正是
    :func:`~src.orchestration.prompt.build_system_prompt` 期望的输入。
    挂在前面则会让后续中间件基于「已经附加过动态段落」的串继续改，
    那正是 ``prompt.py`` 的幂等性剥离要处理的场景 —— 能work，但没必要。
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        resolver: ContextResolver | None = None,
        show_reasoning: bool = True,
    ) -> None:
        """初始化。

        Args:
            enabled (`bool`): 是否启用动态 Prompt。``False`` 时仅做剥离
                （见 :func:`~src.orchestration.prompt.build_system_prompt`
                的 ``enabled`` 说明），用于排障。
            resolver (`ContextResolver | None`): 上下文解析器；``None``
                时用 :func:`default_resolver`。
            show_reasoning (`bool`): 是否在 prompt 里要求模型显式展示推理。
                对应 ``OrchestrationSettings.expose_reasoning``。
        """
        self._enabled = enabled
        self._resolver: ContextResolver = resolver or default_resolver
        self._show_reasoning = show_reasoning
        # 按回复缓存解析结果。
        #
        # ⚠️ 单槽缓存（只记最近一次 reply_id）而不是字典：一个 agent 实例
        # 同一时刻只服务一次回复，用字典只会让已经结束的回复解析结果一直
        # 挂在内存里 —— 而解析结果里可能含用户画像，那是**不该长期驻留**的
        # 个人数据。单槽缓存顺带把这份数据的最长存活时间压到「一次回复」。
        self._cached_reply_id: str | None = None
        self._cached_context: PromptContext = PromptContext()

    async def on_system_prompt(self, agent: Any, current_prompt: str) -> str:
        """重算 system prompt。

        ⚠️ 签名是**三个参数**（``self`` 不计），且**不是洋葱式** ——
        没有 ``next_handler``。基类里这是唯一一个转换器式钩子。

        ⚠️ 整个方法体包在 ``try`` 里。它跑在 agent 的主链路上，抛异常会
        让整轮回复失败；而「prompt 少了一段」的代价与之完全不成比例。
        失败时返回**原始** ``current_prompt``（不剥离），因为此时我们连
        「哪些是我们自己加的」都不确定，乱剥可能剥掉别人的内容。

        Args:
            agent (`Any`): 框架传入的 agent 实例。
            current_prompt (`str`): 前序中间件链式处理后的 prompt。

        Returns:
            `str`: 处理后的完整 system prompt。
        """
        try:
            context = await self._resolve(agent)
            prompt = build_system_prompt(
                current_prompt,
                stage=context.stage,
                request=context.request,
                profile_summary=context.profile_summary,
                enabled=self._enabled,
            )
            # ⚠️ 登记必须在**拼装之后**：登记的内容正是这次拼进去的东西。
            # 放在拼装之前，一旦 ``build_system_prompt`` 抛异常走了下面的
            # ``except``（原样返回、什么都没拼进去），登记就成了一句空头
            # 支票 —— 守卫会拿一份 prompt 里根本没有的数字当豁免来源。
            self._record_injected_numbers(agent, context)
            return prompt
        except Exception:  # noqa: BLE001 —— 见上，主链路上绝不向上抛
            logger.exception("组装动态 Prompt 失败，本轮沿用原始 system prompt。")
            return current_prompt

    def _record_injected_numbers(self, agent: Any, context: PromptContext) -> None:
        """把本轮动态 Prompt 里**承载用户数据**的数字登记给回复守卫。

        Args:
            agent (`Any`): 框架传入的 agent 实例。
            context (`PromptContext`): 本次拼装用的上下文。

        ⚠️ 为什么要有这一步（2026-10-04 缺陷 P1）：用户在前一轮说过「预算
        15000」，后续每一轮的动态 Prompt 都会把「预算上限：15000 元」写进
        system prompt。模型照抄这句正确的话，回复守卫的接地闸门却因为没有
        「本轮用户文本」里的 15000 而判它编造 —— 一段**正确**的答复被拦下
        重说，模型坚持引用时用户最终只能收到一句道歉兜底。

        ⚠️ ``enabled=False`` 时**不登记**：那种配置下动态段落根本没进
        prompt（``build_system_prompt`` 只做剥离），登记就等于凭空造一条
        豁免 —— 守卫会放行一个 prompt 里从没出现过的数字。
        这种情况不登记只会让闸门更保守，方向是安全的。

        ⚠️ 摘的是 :func:`~src.orchestration.prompt.user_data_excerpt`
        （已知要素 + 长期画像），不含阶段指令 —— 理由见该函数的说明。
        """
        if not self._enabled:
            return
        reply_id = getattr(getattr(agent, "state", None), "reply_id", "") or ""
        excerpt = user_data_excerpt(context.request, context.profile_summary)
        record_prompt_numbers(agent, reply_id, sorted(numbers_in(excerpt)))

    # --------------------------------------------------------------------------
    # 内部
    # --------------------------------------------------------------------------
    async def _resolve(self, agent: Any) -> PromptContext:
        """解析并缓存本轮的 :class:`PromptContext`。

        ⚠️ 是 ``async`` 的，因为解析器可以去查库（长期画像，见
        :data:`ContextResolver` 的说明）。同步解析器不受影响 —— 下面用
        ``isinstance(result, Awaitable)`` 判断，而不是无条件 ``await``：
        对一个普通对象 ``await`` 会抛 ``TypeError``，而这条路径上
        「解析器是同步的」恰恰是**默认**情况。

        ⚠️ 缓存键是 ``reply_id`` 而不是 ``cur_iter``。``on_system_prompt``
        每轮推理都调用，而同一个 ``reply_id`` 下阶段**不应该变**（变的话
        模型会在一次回复中途看到两套互相矛盾的指令）。用 ``reply_id``
        做键，正好把「一次回复内恒定、跨回复重算」这个语义表达出来。

        ⚠️ 缓存的是**解析结果**而不是最终 prompt 串。因为最终串还取决于
        ``current_prompt``，而那个每轮都可能不同（框架会往里注入 runtime
        state、工具列表变化等）。缓存串会导致这些变化被忽略。

        ⚠️ ``reply_id`` 拿不到时**不缓存**，每次重算。宁可多算几次，
        也不要在拿不到标识的时候赌「应该是同一次回复」—— 赌错的后果是把
        上一轮的阶段套到这一轮。
        """
        reply_id = getattr(getattr(agent, "state", None), "reply_id", None)
        if reply_id and reply_id == self._cached_reply_id:
            return self._cached_context

        context = self._resolver(agent)
        if isinstance(context, Awaitable):
            context = await context

        if reply_id:
            self._cached_reply_id = reply_id
            self._cached_context = context
        else:
            # 拿不到标识：清掉缓存，避免下次误命中。
            self._cached_reply_id = None
            self._cached_context = PromptContext()

        return context

    def invalidate(self) -> None:
        """丢弃缓存的解析结果，强制下一次调用重新解析。

        ⚠️ 需要它的场景：**同一轮回复内**阶段真的变了。正常路径下不会发生
        （会话状态在回复开始时已定），但人工确认（HITL）恢复之后，用户在
        确认弹窗期间可能改了要素 —— 此时清一下缓存，下一轮读到的就是新值。

        ⚠️ 只清缓存，不重算。重算需要 agent 实例，而调用方（HITL 恢复路径）
        手上未必有；让下一次 ``on_system_prompt`` 自然重算更简单，也避免了
        「清缓存时顺带查了一次库」这种不需要的开销。
        """
        self._cached_reply_id = None
        self._cached_context = PromptContext()


__all__ = [
    "PROMPT_NUMBERS_KEY",
    "ContextInjectionMiddleware",
    "ContextResolver",
    "PromptContext",
    "default_resolver",
    "record_prompt_numbers",
    "recorded_prompt_numbers",
]
