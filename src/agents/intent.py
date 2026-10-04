# -*- coding: utf-8 -*-
"""**意图识别智能体** —— 一句话进，结构化拆解出。

文件职责：
    用框架的 ``structured_schema`` 机制把「用户这句话想干什么」变成
    :class:`~src.domain.schemas.IntentRecognitionResult`。
    对应博客第 127~133 行描述的意图识别 Agent（多意图识别、query 改写、
    两段式结构化输出）。

上下游依赖：
    - 上游：``agentscope.agent.Agent``（框架）、``agentscope.model.ChatModelBase``、
      :mod:`src.agents.prompts`、:mod:`src.domain`。
    - 下游：``src/orchestration/`` 依据它的输出选路；
      ``src/server/agents_factory.py`` 把它包成一个 ``FunctionTool``
      交给主智能体（无状态专家路线，见下）。

═══ 为什么它是「无状态专家」而不是一个 Team 成员 ═══

P3 的方案里，子智能体分两类：

- **无状态专家**（意图识别、query 改写、RAG 问答）：一句话进、一个结构化
  结果出，不需要自己的会话、不需要 HITL、不需要工作区。这类用
  ``FunctionTool`` 包一层就够了。
- **纵向域**（申请单、订单查询）：需要独立会话、独立权限、独立工具集，
  这类才用 ``SubAgentTemplate`` 走 Team 机制。

意图识别属于前者，而且是最典型的一个。用 Team 机制做它的代价是：主智能体
要先调 ``AgentCreate`` 建一个 worker、worker 起一个**新的完整 ReAct 循环**、
再通过 ``TeamSay`` 异步把结果塞回主智能体的收件箱。那是一个**异步**的往返
（见 ``app/_tool/_agent_create.py`` 与 ``app/_bus_ops.py``），而意图识别的
结果要在**同一轮**里决定怎么回复用户 —— 绕这一圈只会让延迟翻倍。

═══ ⚠️ 结构化输出走的是「工具调用」，不是 JSON mode ═══

已核实（``agent/_agent.py:1126-1132``）：设了 ``structured_schema`` 之后，
框架会把 schema 注册成一个名字叫 ``GenerateStructuredOutput`` 的内置工具
（``agent/_structured_output_tool.py:45``），模型**调用它**来交付结果，
框架校验后把结果存进 ``Msg.structured_output``。

这条事实有三个直接后果，每一条都写在了下面的代码里：

1. 模型**不需要**支持 JSON mode / ``response_format``，只需要会发
   ``ToolCallBlock``。所以 ``MockChatModel`` 也能跑通（见单测）。
2. 校验失败**不会**抛异常，而是变成一个 ``state=ERROR`` 的工具结果喂回给
   模型，由模型自己重试。所以「调用方永远拿不到异常」——拿不到结果时
   只能看到 ``structured_output is None``。
3. 交付物是 **``Msg.structured_output``（一个 dict）**，不是消息正文。
   框架最后那条消息的正文是固定的
   ``"The required structured output is generated."``
   （``agent/_agent.py:3568-3577``），照着正文解析会稳定地解析出一句英文。

═══ ⚠️ 每次识别都新建一个 Agent 实例 ═══

这是本文件**最容易被改错**的一处。看着「构造一次、反复调用」很自然，
实际上会静默地把一个无状态专家变成有状态的：

``Agent`` 持有 ``AgentState``，而 ``AgentState.context`` 是**累积的对话
上下文**（``state/_state.py``）。反复用同一个实例，第二次识别时上下文里
就带着第一次的用户输入与结论；模型会受它影响（「上次判了 PLAN_TRIP，
这次大概也是」），而**没有任何报错**。

新建一个 ``Agent`` 的成本很低（就是几次对象构造，没有网络、没有连接），
而它买回来的是「无状态」这个**可验证**的性质。见
``tests/test_agents_intent.py`` 里的跨次污染用例。

═══ ⚠️ 一条**不成立**的理由，写在这里免得再被人当成依据 ═══

上面那段曾经还写着「复用实例会让 ``reply_context.structured_output``
残留上一轮的结果，于是识别失败时会读到上一次的成功结果」。
**这条是错的**，已核实并实测：

- 框架在每次非人工确认的 ``reply`` 开始时，把整个 ``reply_context``
  **整体替换**掉，``structured_output`` 被硬置回 ``None``
  （``agent/_agent.py:1111-1116``）。它没有「残留」的机会。
- 本模块也从**不**读 ``state.reply_context.structured_output`` ——
  它读的是**返回消息上**的那个字段（``getattr(message,
  "structured_output", None)``，见 ``recognize`` 里那一行）。
  而失败路径下框架构造的那条 ``exit_msg`` **根本不含**这个字段
  （``agent/_agent.py:3614-3621``），所以无论复用与否读到的都是 ``None``。
- 实测：复用同一个 ``Agent`` 连跑两次，第一次成功、第二次失败，
  第二次读到的确实是 ``None``（降级成「追问」），没有拿到上一次的意图。

所以真正的理由**只有上下文累积**那一条。留着这段是因为一条听起来
合理、但实际不成立的论据比没有论据更危险 —— 下一个人会照它去推理
（比如据此认为「只要清掉 ``structured_output`` 就能安全复用」）。
先跑一遍再断言，别照着注释推。见
``tests/test_agents_intent.py`` 里的跨次污染用例。
"""

from __future__ import annotations

import logging
from typing import Any

from agentscope.agent import Agent
from agentscope.message import Msg, TextBlock
from agentscope.model import ChatModelBase

from src.agents.prompts import prompt_for
from src.domain import AgentName, Intent
from src.domain.schemas import IntentDecision, IntentRecognitionResult

#: 本模块的日志器。
logger = logging.getLogger(__name__)

#: 意图识别智能体的名字。
#:
#: ⚠️ 取自 :class:`~src.domain.enums.AgentName` 而**不是**硬写字符串。
#: 这个名字会进日志、trace 的 span 名，也是 ``prompts.PROMPTS`` 的键；
#: 三处各写一份字面量的后果是改名时漏掉一处，而漏掉的那处不报错。
INTENT_AGENT_NAME = AgentName.INTENT.value

#: 识别不出任何意图时，兜底问用户的那句话。
#:
#: ⚠️ 它要**给出选项**，而不是笼统地说「请说清楚一点」。用户之所以落到
#: 这个分支，恰恰是因为不知道该怎么说；再让他自由发挥一次，大概率还是
#: 落到同一分支。列出几个能力范围让他挑，是把「开放题」变成「选择题」。
_DEFAULT_CLARIFICATION = (
    "方便说得再具体一点吗？比如你是想规划一次出差、查看已有的订单和申请、"
    "了解差旅和报销标准，还是要办理出差申请？"
)


class IntentRecognizer:
    """意图识别器：把用户输入变成结构化意图拆解。

    ⚠️ **不要跨请求复用**。本类内部每次都新建 ``Agent``，所以它自身是
    无状态的；但把它做成进程级单例会让人误以为「它有过什么缓存」，
    从而在将来顺手加进去一个缓存 —— 而那个缓存会跨用户泄漏。
    正确的用法是每个请求构造一个（成本就是几次对象构造）。

    Attributes:
        _model (`ChatModelBase`): 模型实例。
        _max_intents (`int`): 最多保留几个意图。
        _confidence_threshold (`float`): 低于此置信度时追问澄清。
    """

    def __init__(
        self,
        *,
        model: ChatModelBase,
        max_intents: int = 5,
        confidence_threshold: float = 0.6,
        system_prompt: str | None = None,
    ) -> None:
        """初始化。

        Args:
            model (`ChatModelBase`): 模型实例。⚠️ 用**同一个**实例给所有
                请求是安全的（``ChatModelBase`` 不持有对话状态，状态在
                ``Agent`` 上），这正是本类每次新建 ``Agent`` 却复用模型的原因。
            max_intents (`int`): 最多保留几个意图，对应
                ``OrchestrationSettings.max_subagent_calls``。
                ⚠️ 必须有上限：多意图输入会解析出多条，无上限时一次请求
                可能触发一串子调用，延迟与成本都被放大。
            confidence_threshold (`float`): 置信度阈值，对应
                ``OrchestrationSettings.intent_confidence_threshold``。
            system_prompt (`str | None`): 覆盖默认提示词，仅供测试与灰度对比。

        Raises:
            ValueError: ``max_intents`` 小于 1，或阈值不在 ``[0, 1]``。
                ⚠️ 这两个都是「配置写错导致功能静默失效」的类型：
                ``max_intents=0`` 会让所有识别结果都被截空，
                阈值 ``>1`` 会让每一次识别都触发追问 —— 两者都不报错，
                只是系统变得没用。放在构造时失败，问题在启动时就暴露。
        """
        if max_intents < 1:
            raise ValueError("max_intents 至少为 1")
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold 必须落在 [0, 1] 区间")

        self._model = model
        self._max_intents = max_intents
        self._confidence_threshold = confidence_threshold
        self._system_prompt = system_prompt or prompt_for(INTENT_AGENT_NAME)

    # --------------------------------------------------------------------------
    # 主入口
    # --------------------------------------------------------------------------
    async def recognize(
        self,
        text: str,
        *,
        context_hint: str = "",
    ) -> IntentRecognitionResult:
        """识别一句话里的意图。

        ⚠️ 本方法**不会抛异常**（除 ``KeyboardInterrupt`` 这类之外）。
        它跑在每一轮对话的主链路上，而「意图识别失败」的正确处置是
        **降级成追问**，不是把整轮回复打断。降级路径见
        :meth:`_degrade`。

        Args:
            text (`str`): 用户这一轮的原始输入。
            context_hint (`str`): 可选的上下文补充（例如「上一轮在收集
                出发日期」）。框架会把整段上下文给模型，这里只是把
                **项目自己的**阶段信息补进去 —— 模型看不到我们的
                ``middle_context``。

        Returns:
            `IntentRecognitionResult`: 结构化识别结果。**永远返回一个对象**
            （失败时是 ``OTHER`` + 追问），调用方不需要判空。
        """
        clean = (text or "").strip()
        if not clean:
            # 空输入不该走模型：没有任何信息可供判断，模型只能编。
            return self._degrade("用户这一轮没有说任何内容。")

        try:
            message = await self._recognize_with_model(clean, context_hint)
        except Exception:  # noqa: BLE001 —— 见 docstring，主链路上绝不向上抛
            logger.exception("意图识别失败，本轮降级为追问。")
            return self._degrade("识别过程中出现了意外情况。")

        if message is None:
            return self._degrade("模型没有按要求交付结构化结果。")

        return self._normalize(message)

    # --------------------------------------------------------------------------
    # 内部：模型调用与结果规整
    # --------------------------------------------------------------------------
    async def _recognize_with_model(
        self,
        text: str,
        context_hint: str,
    ) -> IntentRecognitionResult | None:
        """真正调模型的那一步。

        ⚠️ 每次调用都**新建** ``Agent``，理由见模块文档。这里不再做额外的
        异常处理 —— 由 :meth:`recognize` 统一兜。

        Returns:
            `IntentRecognitionResult | None`: 解析成功的结果；模型没交付时
            为 ``None``。
        """
        agent = Agent(
            name=INTENT_AGENT_NAME,
            system_prompt=self._system_prompt,
            model=self._model,
            # ⚠️ 刻意**不传 toolkit**：意图识别不调用任何业务工具。
            # 给它工具会引入一个真实的失败模式：模型在「理解」阶段顺手
            # 查一次航班，而查询结果会污染它对意图的判断（比如用户只是
            # 随口说了个城市，模型就去查了 —— 于是它倾向于判成「规划行程」）。
            # 再加上模块文档说的「理解阶段执行了就没有回头路」，这条线必须划死。
            toolkit=None,
        )

        prompt = text if not context_hint else f"（背景：{context_hint}）\n{text}"
        message = await agent.reply(
            inputs=Msg(name="user", role="user", content=[TextBlock(text=prompt)]),
            # ⚠️ ``structured_schema`` 是**每次调用**的参数，不是构造参数
            # （``agent/_agent.py:332-341``）。构造时没有这个口子。
            structured_schema=IntentRecognitionResult,
        )

        raw = getattr(message, "structured_output", None)
        if not isinstance(raw, dict):
            # ⚠️ 走到这里说明模型在 ``max_iters`` 之内始终没能调用
            # ``GenerateStructuredOutput``。框架此时**不报错**，
            # 只是回复以 ``EXCEED_MAX_ITERS`` 结束且 ``structured_output``
            # 为 ``None``（``agent/_agent.py:3580-3653``）。
            logger.warning(
                "意图识别没有得到结构化结果（模型可能在迭代上限内未交付），本轮降级为追问。",
            )
            return None

        # ⚠️ 这里是**二次校验**。框架在 ``GenerateStructuredOutput`` 内部
        # 已经用同一个 schema 校验过一次了，照理不该失败。但仍然要接住：
        # 框架允许 ``structured_schema`` 是一个 **JSON schema dict**
        # （``state`` 重载后的形态，见 ``agent/_structured_output_tool.py:141-155``），
        # 那条路径用的是「默认值填充」式的宽松校验，严格程度与本类的
        # pydantic 校验不同。二次校验把「框架认为合法」和「我们认为合法」
        # 这两个标准之间的距离显式化，而不是让它以 ``KeyError`` 的形式
        # 在某个更远的地方炸开。
        try:
            return IntentRecognitionResult.model_validate(raw)
        except Exception:  # noqa: BLE001
            logger.exception("意图识别的结构化结果无法解析，本轮降级为追问。")
            return None

    def _normalize(self, result: IntentRecognitionResult) -> IntentRecognitionResult:
        """把模型给的结果规整成「可以直接拿去选路」的形态。

        做三件事，每一件都对应一类模型**不会遵守**的格式约定：

        1. **重排**。提示词要求按置信度降序，但控制流不得依赖模型遵守
           格式约定 —— 见 :meth:`IntentRecognitionResult.top_intent` 的
           说明。这里主动排序，下游按位置取也不会错。
        2. **限流**。截到 ``max_intents`` 条。
        3. **兜底**。没有意图、或最高置信度低于阈值时，转成「追问」。

        Args:
            result (`IntentRecognitionResult`): 模型交付并经校验的结果。

        Returns:
            `IntentRecognitionResult`: 规整后的结果。
        """
        intents = sorted(
            result.intents,
            key=lambda item: item.confidence,
            reverse=True,
        )[: self._max_intents]

        if not intents:
            return self._degrade(
                "模型没有给出任何意图。",
                reasoning=result.reasoning,
                rewritten_query=result.rewritten_query,
            )

        # ⚠️ 只有 ``OTHER`` 一条且置信度不低时，仍然算「识别不出来」。
        # ``OTHER`` 是模型表达「都不像」的出口，把它当普通意图往下走，
        # 会让编排层去选一个「OTHER 对应的智能体」—— 而那个映射不存在。
        top = intents[0]
        if top.intent is Intent.OTHER or top.confidence < self._confidence_threshold:
            return IntentRecognitionResult(
                reasoning=result.reasoning,
                rewritten_query=result.rewritten_query or "",
                intents=intents,
                needs_clarification=True,
                # ⚠️ 模型给了问题就用它的（它更懂用户刚说了什么），
                # 没给才用兜底。反过来的话，我们会用一个通用问题盖掉
                # 一个针对性更强的追问。
                clarification_question=(
                    result.clarification_question.strip() or _DEFAULT_CLARIFICATION
                ),
            )

        # ⚠️ 走到这里说明识别是成功的。**必须**把 ``needs_clarification``
        # 清掉 —— 模型偶尔会一边给出高置信度意图、一边把追问标志也置上
        # （它把「顺便确认一下」也当成了追问）。留着这个矛盾标志，
        # 编排层会因为「要追问」而放弃一个本来可用的判断。
        return IntentRecognitionResult(
            reasoning=result.reasoning,
            rewritten_query=result.rewritten_query or "",
            intents=intents,
            needs_clarification=False,
            clarification_question="",
        )

    def _degrade(
        self,
        why: str,
        *,
        reasoning: str = "",
        rewritten_query: str = "",
    ) -> IntentRecognitionResult:
        """构造一条「识别不出来」的结果。

        ⚠️ 降级结果里**保留**模型已经给出的 ``reasoning`` 与
        ``rewritten_query``（如果有）。它们在「要追问」这个场景下仍然有用：
        改写后的 query 可以直接拿去检索政策库，先给用户一点有用的东西，
        而不是干问一句「你想干什么」。

        ⚠️ ``why`` 只进日志，**不进**给用户看的问题。向用户解释
        「模型没有交付结构化结果」既没有帮助，也暴露了内部实现。

        Args:
            why (`str`): 降级原因（日志用）。
            reasoning (`str`): 模型给出的推理，若已有。
            rewritten_query (`str`): 模型给出的改写，若已有。

        Returns:
            `IntentRecognitionResult`: ``OTHER`` 意图 + 追问。
        """
        logger.info("意图识别降级：%s", why)
        return IntentRecognitionResult(
            reasoning=reasoning,
            rewritten_query=rewritten_query,
            intents=[
                IntentDecision(
                    intent=Intent.OTHER,
                    confidence=0.0,
                    slots={},
                    reason="未能判断出明确的意图。",
                ),
            ],
            needs_clarification=True,
            clarification_question=_DEFAULT_CLARIFICATION,
        )


def build_intent_recognizer(
    *,
    model: ChatModelBase,
    max_intents: int,
    confidence_threshold: float,
) -> IntentRecognizer:
    """按配置构造一个意图识别器。

    ⚠️ 参数**没有默认值**，与 :class:`IntentRecognizer` 刻意不同。
    调用方（``src/server/agents_factory.py``）手上一定有
    ``OrchestrationSettings``，强制它显式传值，可以避免「配置改了但构造处
    用了默认值」这种改了不生效的情况 —— 那种情况下系统行为与配置不符，
    而排查时第一反应是「配置没生效」，方向就偏了。

    Args:
        model (`ChatModelBase`): 模型实例。
        max_intents (`int`): 最多保留几个意图。
        confidence_threshold (`float`): 置信度阈值。

    Returns:
        `IntentRecognizer`: 构造好的识别器。
    """
    return IntentRecognizer(
        model=model,
        max_intents=max_intents,
        confidence_threshold=confidence_threshold,
    )


__all__ = [
    "INTENT_AGENT_NAME",
    "IntentRecognizer",
    "build_intent_recognizer",
]
