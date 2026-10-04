# -*- coding: utf-8 -*-
"""把框架的 ``AgentEvent`` 流翻译成 :class:`~src.chains.collector.TaskCollector` 的操作。

文件职责：
    这是**唯一**一处把 ``agentscope.event`` 的 28 种事件接进本项目思考链的
    地方。上游是 :meth:`agentscope.agent.Agent.reply_stream`，下游是
    :class:`~src.chains.collector.TaskCollector`（纯数据）。

上下游依赖：
    - 上游：``agentscope.event``（事件类型）、``agentscope.message``
      （``ToolResultState``）、``agentscope.types``（``ReplyFinishedReason``）。
    - 下游：``src/server/routers/_chains.py`` 的思考链 SSE 端点
      （``GET /api/v1/sessions/{session_id}/chains``）—— 它按连接各建一个
      收集器与本适配器，消费总线上的事件，把
      :func:`iter_task_states` 的产物（连同推理文本）推给前端。
      本模块因此**有**生产消费方；``tests/test_chains_events.py`` 是它的
      单元驱动者，两者都依赖下面这些「反直觉之处」。

      ⚠️ 这里曾经写着「下游是 ``src/server/agents_factory.py``」——
      那是**错的**，那个文件**从未** import 过本模块（它的 import 列表里
      只有 ``src.agents`` / ``src.orchestration`` / ``src.tools`` /
      ``src.llm`` / ``src.storage``）。一条指向不存在连线的「下游」，
      比没有这条注释更糟：它会让读者以为接线已经完成，
      从而去别处找一个根本不存在的问题。留着这条更正，是因为照着它
      推理会得出「本模块无人消费、可以随便改」这种危险结论。

═══ ⚠️ 事件有**两种形态**：对象与字典 ═══

``Agent.reply_stream`` 直接产出的是 **pydantic 对象**；而
``agentscope.app`` 的服务层在把事件发到消息总线之前会做一次
``event.model_dump(mode="json")``（``app/_service/_chat.py:1345-1350``），
于是从**总线/SSE 路径**拿到的全是**普通字典**。

本模块两种都收（``_field()`` 是唯一取值入口）。这不是防御性编程：
``getattr(字典, "type", None)`` 返回 ``None``，会让整条分发链一条都
匹配不上，所有事件被静默忽略，症状是「思考链永远是空的」且不报任何错
—— 与「这一轮确实没调工具」的表现一模一样。

═══ ⚠️ 本模块**必须** import ``agentscope`` ═══

与 :mod:`src.chains.collector` 刻意相反：收集器是纯的（可穷举单测），
本模块是框架的适配层，脱离框架没有意义。两者的分工正是为了做到
「易错的那部分能被穷举测试」—— 事件到收集器调用的映射逻辑很薄，
而状态机（谁能迁到谁）在收集器里。

═══ 事件的三个「反直觉」之处（都已在代码里处理）═══

**一、工具参数是**分片**到达的，且结束事件里没有参数。**

``TOOL_CALL_START`` 只带 ``tool_call_name`` 与 ``tool_call_id``；
参数以 JSON 字符串**片段**的形式，一条一条地出现在 ``TOOL_CALL_DELTA``
的 ``delta`` 字段里；``TOOL_CALL_END`` **只有 ``tool_call_id``**
（``event/_event.py:313-347``）。

所以「拿到一个完整的工具调用」= 从 START 收名字、从若干 DELTA 拼参数、
到 END 才**第一次**凑齐。本模块因此把 ``collector.plan()`` 放在
``TOOL_CALL_END`` 上，而不是 START 上 —— 见 :meth:`_on_tool_call_end`。

**二、工具结果是**流式**的，最终事件里没有内容。**

``TOOL_RESULT_TEXT_DELTA`` 携带 ``delta``；``TOOL_RESULT_END`` 只带
``state``（``ToolResultState``）。所以要显示结果摘要，必须自己累积分片。

**三、一个工具调用可能**永远等不到结果**。**

人工确认（HITL）路径下，模型发出 ``submit_approval`` 之后，框架会发一个
``REQUIRE_USER_CONFIRM`` 并**暂停**整轮回复 —— 此时该工具调用没有结果，
任务停在 ``PENDING``。这是**正常状态**（系统在等人），不是异常。

⚠️ 但有两种「回复结束了，而任务还在跑」是**真的**卡住了：
迭代上限（``EXCEED_MAX_ITERS``）与回复出错（``ERROR``）。这两种情况下
那些 ``DOING`` 的任务不会再有结果，留着它们会让界面永远转圈。
强行收尾**只**针对这两种，见 :meth:`_on_reply_end`。

═══ 四、同一个 ``reply_id`` 的 ``REPLY_START`` **不是**新回复 ═══

已核实（``app/_service/_chat.py:1332-1341``）：HITL 确认之后，服务层会
补发一个 **``reply_id`` 与暂停前相同**的 ``ReplyStartEvent``，框架自己的
注释写着「SSE 处理器**不得**因为收到相同 ``reply_id`` 的 ``REPLY_START``
就清空累积的缓冲区 —— 这个事件表示**续接**，不是新回复」。

所以 :meth:`EventChainAdapter._on_reply_start` 会先比对 ``reply_id``：
相同则**保留**中间状态。无条件清空的后果很具体 —— 用户点了「同意」之后，
``submit_approval`` 的结果事件找不到自己那条任务登记，审批任务在界面上
永远停在转圈状态，而用户刚刚才点了确认。

═══ ⚠️ ``EXCEED_MAX_ITERS`` 事件**已被框架标记为废弃** ═══

``event/_event.py:425-431``：``ExceedMaxItersEvent`` 带 ``@deprecated``，
文档字符串写着「仍为向后兼容而发出，但**不携带语义**；请改用
``ReplyEndEvent.finished_reason``」。本地版本目前两处都发，但事件本身
随时可能被移除 —— 只认它就等于把「任务收尾」这件功能押在一个已宣布
会消失的事件上。本模块两条路都处理（``ReplyEndEvent`` 是正路），
且两边都幂等（收尾只针对 ``DOING``，跑过一次就没有 ``DOING`` 了）。

═══ 事件里的 ``type`` 有**两种形态**，所以一律用 ``==`` 判断 ═══

本模块要同时吃两种形状的事件（见上文「两种事件形态」），而 ``type``
字段在两种形状下**不是同一个东西**。已核实：

- **对象形态**（``reply_stream`` 直接 yield 的）：``type`` 是
  ``EventType`` 的**枚举成员**。实测 ``e.type`` 的 repr 是
  ``<EventType.REPLY_START: 'REPLY_START'>``，``isinstance(e.type,
  EventType)`` 为真，``e.type is EventType.REPLY_START`` **也为真**。
  原因是每个事件的 ``type`` 声明成 ``Literal[EventType.X]``
  （``event/_event.py:85`` 等），pydantic 不会把它降级成字符串。

- **字典形态**（框架 SSE 路径上 ``model_dump(mode="json")`` 之后，
  见 ``app/_service/_chat.py``）：``type`` 被序列化成**普通字符串**
  ``"REPLY_START"``，``is`` 在这里必然是假。

⚠️ 所以**两种写法各自只在一种形状下正确**，这就是本模块一律用 ``==``
的唯一原因：``EventType`` 是 ``StrEnum``，成员与它的字符串值相等
（``EventType.REPLY_START == "REPLY_START"``），于是 ``==`` 在两种形状下
都对。写 ``is`` 会在字典形态下**静默**失效 —— 所有事件都被忽略，
思考链永远是空的，而没有任何报错。

⚠️ 这段之前写的是「``type`` 不是枚举成员，``is`` 永远为假」——
那是**错的**（对象形态下 ``is`` 为真）。留着这条更正，是因为照着
那个说法推理会得出「``is`` 反正永远是假，用它也没关系」这种危险结论。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterable
from typing import Any

from agentscope.event import EventType
from agentscope.message import Msg, ToolResultState
from agentscope.types import ReplyFinishedReason

from src.chains.collector import TaskCollector
from src.domain import TaskState

#: 本模块的日志器。
logger = logging.getLogger(__name__)

#: 工具名 → 面向用户的中文说明。
#:
#: ⚠️ 为什么工具名要在这里再写一遍（而不是从 ``src/tools`` import 常量）：
#: 工具函数都是 **``build_*_tools`` 内部的闭包**，模块级没有同名符号 ——
#: 拿不到。可以构造出工具再从 ``FunctionTool.name`` 反查，但那要求本模块
#: 依赖仓储与用户标识，把「事件翻译」这件轻活拖进一整条装配链。
#:
#: 代价是这份表可能与真实工具名**漂移**（改了工具名，忘了改这里）。
#: 守着它的是 ``tests/test_chains_events.py`` 里的
#: ``test_every_registered_tool_has_a_title``：那条用例真的构造出工具集，
#: 遍历每一个工具的 ``name``，断言它在这张表里。漂移会直接变红。
TITLES: dict[str, str] = {
    "aligo_route_intent": "正在判断该走哪条处理流程",
    "search_transport": "正在查询交通方案",
    "search_hotels": "正在查询酒店",
    "check_travel_policy": "正在核对差旅标准",
    "query_orders": "正在查询订单与申请",
    "submit_approval": "正在提交出差申请",
}

#: 工具名 → 标题取不到时的兜底前缀。
_FALLBACK_TITLE_PREFIX = "正在调用"

#: 等待用户确认时登记的任务名。
#:
#: ⚠️ 名字以一个不可能与真实工具撞车的标识起头（不是 ``query_*`` 之类的
#: 工具命名风格），这样在任务清单里一眼能区分「等用户」与「在查数据」。
CONFIRM_TASK_NAME = "!user_confirmation"


def _field(event: Any, name: str, default: Any = None) -> Any:
    """从一个事件上取字段，**同时支持对象与字典**。

    ⚠️ 这个函数不是多余的防御性编程，它对应一个已核实的**事实**：
    ``agentscope.app`` 往消息总线上发的**所有**事件都是
    ``event.model_dump(mode="json")`` 的产物 —— 也就是普通字典
    （``app/_service/_chat.py:1345-1350``，连它自己补发的续接
    ``REPLY_START`` 也是 ``:1340-1344``）。

    两种形态的差别在这里是致命的：``getattr(字典, "type", None)``
    返回 ``None``，于是 :meth:`EventChainAdapter._dispatch` 的整条
    ``if/elif`` 链**一条都匹配不上**，所有事件被静默忽略。
    症状是「思考链永远是空的」，而不报任何错 —— 它的表现与
    「这一轮确实没调工具」完全一样。

    ⚠️ 比较仍然是安全的：``EventType`` 与 ``ToolResultState`` 都是
    ``StrEnum``，成员与自己的字符串值相等且哈希相同
    （已实测），所以 ``"TOOL_CALL_END" == EventType.TOOL_CALL_END`` 为真。

    Args:
        event (`Any`): 事件对象或事件字典。
        name (`str`): 字段名。
        default (`Any`): 取不到时的默认值。

    Returns:
        `Any`: 字段值。
    """
    if isinstance(event, dict):
        return event.get(name, default)
    return getattr(event, name, default)


def title_for(tool_name: str) -> str:
    """取某个工具的面向用户标题。

    ⚠️ 认不出的工具名**不返回空串**，而是给一句「正在调用 xxx」。
    空标题在界面上表现为「一行空白在进行中」，用户完全不知道系统在干什么 ——
    而这恰恰发生在「新加了一个工具但忘了登记标题」的时候，
    也就是最需要界面还能说清楚话的时候。

    Args:
        tool_name (`str`): 工具名。

    Returns:
        `str`: 中文标题。
    """
    known = TITLES.get(tool_name)
    if known is not None:
        return known
    return f"{_FALLBACK_TITLE_PREFIX} {tool_name}" if tool_name else "正在处理"


class EventChainAdapter:
    """把事件流喂给 :class:`~src.chains.collector.TaskCollector`。

    ⚠️ 一个实例对应**一个 agent 的一次回复**。它内部持有按 ``tool_call_id``
    索引的中间状态（参数分片、结果分片），跨回复复用会把上一次的分片
    拼进这一次的结果里 —— 而 ``tool_call_id`` 是模型生成的，
    两次回复里未必不同。

    ⚠️ **非线程安全**，与收集器一致：由单个 asyncio 任务驱动。

    Attributes:
        _collector (`TaskCollector`): 下游收集器。
        _expose_reasoning (`bool`): 是否累积模型推理文本。
        _call_names (`dict[str, str]`): ``tool_call_id`` → 工具名。
        _call_args (`dict[str, list[str]]`): ``tool_call_id`` → 参数 JSON 分片。
        _result_text (`dict[str, list[str]]`): ``tool_call_id`` → 结果文本分片。
        _tasks (`dict[str, str]`): ``tool_call_id`` → 收集器分配的 ``task_id``。
        _reasoning (`list[str]`): 累积的推理文本分片。
        _confirm_task (`str | None`): 等待确认任务的 ``task_id``。
        _confirm_count (`int`): 本轮累计收到多少条「请用户确认」。
        _reply_id (`str | None`): 当前回复的标识，用来识别「续接」。
    """

    def __init__(
        self,
        collector: TaskCollector,
        *,
        expose_reasoning: bool = True,
    ) -> None:
        """初始化。

        Args:
            collector (`TaskCollector`): 任务收集器。
            expose_reasoning (`bool`): 是否累积模型推理文本，对应
                ``OrchestrationSettings.expose_reasoning``。
                ⚠️ 关掉它不只是「不显示」，而是**根本不累积** ——
                推理文本可能含用户没说过的个人信息（模型从上下文里推断的），
                让它在内存里多留一份没有收益。
        """
        self._collector = collector
        self._expose_reasoning = expose_reasoning
        self._call_names: dict[str, str] = {}
        self._call_args: dict[str, list[str]] = {}
        self._result_text: dict[str, list[str]] = {}
        self._tasks: dict[str, str] = {}
        self._reasoning: list[str] = []
        self._confirm_task: str | None = None
        self._confirm_count = 0
        self._reply_id: str | None = None

    # --------------------------------------------------------------------------
    # 对外
    # --------------------------------------------------------------------------
    @property
    def reasoning(self) -> str:
        """累积的模型推理文本（可能为空串）。

        ⚠️ 返回**拼接后**的新字符串，不是内部列表。调用方拿到之后随意处理
        都不会影响累积过程。

        Returns:
            `str`: 推理文本。
        """
        return "".join(self._reasoning)

    def reset(self) -> None:
        """清空本轮的中间状态。

        ⚠️ **不**清空收集器 —— 收集器里的任务清单可能正被界面读着，
        清空它由调用方显式决定（见 :meth:`TaskCollector.clear` 的说明）。
        本方法只处理「下一次回复不该继承上一次的分片」这件事。
        """
        self._call_names.clear()
        self._call_args.clear()
        self._result_text.clear()
        self._tasks.clear()
        self._reasoning.clear()
        self._confirm_task = None
        self._confirm_count = 0
        self._reply_id = None

    def consume(self, event: Any) -> None:
        """处理一个事件。

        ⚠️ 认不出的事件**静默忽略**，不记日志。理由：``AgentEvent`` 是
        28 个成员的联合，其中绝大多数（``DATA_BLOCK_*``、``TOOL_RESULT_DATA_*``
        等）与本项目无关，逐条告警会把日志淹掉；而真正需要留痕的
        「事件流异常」在下面对应的分支里单独告警。

        ⚠️ 整个方法体不抛异常。它跑在 ``reply_stream`` 的消费循环里，
        抛出去会中断整轮回复 —— 而思考链是**投影**，投影坏了不该让业务
        也坏掉（见 :mod:`src.chains.collector` 的模块文档）。

        Args:
            event (`Any`): 框架事件对象。
        """
        try:
            self._dispatch(event)
        except Exception:  # noqa: BLE001 —— 见 docstring
            logger.exception(
                "处理事件 %r 时出错，已跳过。思考链是投影，不影响本轮回复。",
                _field(event, "type", type(event).__name__),
            )

    async def consume_stream(self, events: AsyncIterator[Any]) -> None:
        """消费一整条事件流直到结束。

        ⚠️ 只处理 ``AgentEvent``，**丢弃**流里的 ``Msg``。``reply_stream``
        的产出类型是 ``AgentEvent | Msg``（``agent/_agent.py:288-298``），
        而最终那条 ``Msg`` 承载的是结构化结果，与任务清单无关 ——
        它的消费方是调用方自己，本适配器只负责把它丢掉。

        ⚠️ 这里曾经举 ``src/agents/intent.py`` 为例，那是**错的**：
        它用的是 ``agent.reply(...)``（**非**流式，``src/agents/intent.py:250``），
        整个模块里没有一处 ``reply_stream``，从来不需要从事件流里挑出
        ``Msg``。举这个例子会把读者引向一种本项目并不存在的调用形态。

        ⚠️ 不 break、不提前返回：事件流的消费者**必须**把它读干，
        否则框架内部的 ``asyncio.Queue`` 生产者会在某个 ``put()`` 上
        永久阻塞，而那个任务不会被回收。

        Args:
            events (`AsyncIterator[Any]`): ``reply_stream`` 的返回值。
        """
        async for event in events:
            if isinstance(event, _MESSAGE_TYPES):
                continue
            self.consume(event)

    # --------------------------------------------------------------------------
    # 分发
    # --------------------------------------------------------------------------
    def _dispatch(self, event: Any) -> None:
        """按事件类型分发。

        ⚠️ 用 ``if/elif`` 链而不是 ``dict[type, handler]``：事件类型是
        **字符串**（见模块文档），不能直接当类用；而按字符串建表要在
        import 期就把 28 个成员都列出来，漏一个不会报错，只会静默少处理
        一种事件。``if/elif`` 链的「漏了就是没写」更显眼。

        Args:
            event (`Any`): 框架事件对象。
        """
        event_type = _field(event, "type", None)

        if event_type == EventType.REPLY_START:
            self._on_reply_start(event)
        elif event_type == EventType.REPLY_END:
            self._on_reply_end(event)
        elif event_type == EventType.THINKING_BLOCK_DELTA:
            self._on_thinking_delta(event)
        elif event_type == EventType.TOOL_CALL_START:
            self._on_tool_call_start(event)
        elif event_type == EventType.TOOL_CALL_DELTA:
            self._on_tool_call_delta(event)
        elif event_type == EventType.TOOL_CALL_END:
            self._on_tool_call_end(event)
        elif event_type == EventType.TOOL_RESULT_START:
            self._on_tool_result_start(event)
        elif event_type == EventType.TOOL_RESULT_TEXT_DELTA:
            self._on_tool_result_delta(event)
        elif event_type == EventType.TOOL_RESULT_END:
            self._on_tool_result_end(event)
        elif event_type == EventType.REQUIRE_USER_CONFIRM:
            self._on_require_confirm(event)
        elif event_type == EventType.USER_CONFIRM_RESULT:
            self._on_confirm_result(event)
        elif event_type == EventType.EXCEED_MAX_ITERS:
            self._on_exceed_max_iters(event)

    # --------------------------------------------------------------------------
    # 各分支
    # --------------------------------------------------------------------------
    def _on_reply_start(self, event: Any) -> None:
        """新一轮回复开始：清掉上一轮的中间状态。

        ⚠️ 只清**本适配器**的中间状态，不动收集器。收集器里的清单属于
        「当前这次会话的思考链」，跨回复是连续的（用户能看到历史任务），
        清掉它等于把界面上正在显示的内容抹掉。

        ⚠️⚠️ **``reply_id`` 相同则是「续接」，一切照旧。** 已核实
        （``app/_service/_chat.py:1332-1341``）：人工确认之后服务层会补发
        一个 ``reply_id`` 与暂停前相同的 ``ReplyStartEvent``，框架的注释
        明确要求消费者「不得因为收到相同 ``reply_id`` 的 ``REPLY_START``
        就清空累积的缓冲区」。

        无条件清空的后果很具体：用户点了「同意」，``submit_approval``
        的结果事件回到适配器时，``_tasks`` 里那条登记已经被抹掉 ——
        审批任务在界面上**永远停在转圈**，而恰恰是用户刚刚点了确认的那一刻。
        同时那条「找不到调用登记」的告警会把它伪装成上游乱序。

        Args:
            event (`Any`): ``ReplyStartEvent``。
        """
        reply_id = _field(event, "reply_id", "") or ""
        if reply_id and reply_id == self._reply_id:
            logger.debug("收到同一轮回复（%s）的续接 REPLY_START，保留中间状态。", reply_id)
            return
        self._reply_id = reply_id or None
        self._call_names.clear()
        self._call_args.clear()
        self._result_text.clear()
        self._tasks.clear()
        self._reasoning.clear()
        self._retire_pending_confirm()
        # ⚠️ 计数**必须**与 ``_confirm_task`` 一起归零，两行是一个整体。
        # 只清任务不清计数的话，下一轮的「等待你确认」会在上一轮的数字上
        # 接着累加（见 :meth:`_on_require_confirm` 的累加逻辑），界面上于是
        # 写着一个比实际多出来的数 —— 用户按这个数字去数确认框，怎么数都对不上。
        self._confirm_count = 0

    def _retire_pending_confirm(self) -> None:
        """收掉上一轮遗留的「等待你确认」任务，然后把追踪引用清空。

        ⚠️ 为什么**不能**只写一句 ``self._confirm_task = None``（本方法存在的
        全部理由）：``!user_confirmation`` 是一条**伪任务**，它不由任何工具
        结果收尾 —— 唯一能收掉它的是 :meth:`_on_confirm_result`，而那个方法
        的入口判据就是 ``self._confirm_task is None``。一旦这里把引用丢掉却
        不管收集器里那条记录，它就**永远**停在 ``PENDING``：

            · 界面上留着一个永远转不完的「等待你确认 N 项操作」；
            · 下一轮的 ``REQUIRE_USER_CONFIRM`` 发现 ``_confirm_task is None``，
              于是**又** ``plan`` 一条同名的 —— 两行一模一样的 PENDING。

        ⚠️ 这也**不**与 :meth:`_on_reply_end` 的「不碰 PENDING」矛盾：那里
        说的是 ``submit_approval`` 这类**真工具**任务（下一个回合会被同一个
        ``tool_call_id`` 认领并正常收尾，见该方法的说明）。伪任务没有
        ``tool_call_id``，没人会来认领它。

        ⚠️ 收尾用 ``ok=True`` + 明确的文案，而不是标成红色失败：确认请求失效
        不是错误，是「这一轮已经过去了」。这与 :meth:`_on_confirm_result`
        对**用户拒绝**的处置一致（那里也是 ``ok=True`` + 「另有 N 项未同意」）。
        标红会让用户以为自己点错了什么。
        """
        if self._confirm_task is not None:
            self._collector.add_result(
                self._confirm_task,
                "确认请求已失效（本轮回复已结束）。",
                ok=True,
            )
            self._confirm_task = None

    def _on_thinking_delta(self, event: Any) -> None:
        """累积模型推理分片（思考链的「显示推理」）。

        Args:
            event (`Any`): ``ThinkingBlockDeltaEvent``。
        """
        if not self._expose_reasoning:
            return
        delta = _field(event, "delta", "") or ""
        if delta:
            self._reasoning.append(delta)

    def _on_tool_call_start(self, event: Any) -> None:
        """记下工具名，参数要等后续分片。

        ⚠️ **不**在这里调 ``collector.plan()``：那时参数还没到，
        登记出来的任务会带着一个空的 ``arguments``，而参数是界面上展开
        详情时唯一有用的东西。见 :meth:`_on_tool_call_end`。

        Args:
            event (`Any`): ``ToolCallStartEvent``。
        """
        call_id = _field(event, "tool_call_id", "") or ""
        name = _field(event, "tool_call_name", "") or ""
        if not call_id:
            return
        self._call_names[call_id] = name
        # 同一个 id 复用（模型重新发了一次同名调用）时清掉旧分片，
        # 否则新的参数会接在旧的后面，拼出一段谁都没生成过的 JSON。
        self._call_args[call_id] = []
        self._result_text[call_id] = []

    def _on_tool_call_delta(self, event: Any) -> None:
        """累积参数 JSON 分片。

        Args:
            event (`Any`): ``ToolCallDeltaEvent``。
        """
        call_id = _field(event, "tool_call_id", "") or ""
        delta = _field(event, "delta", "") or ""
        if not call_id or not delta:
            return
        self._call_args.setdefault(call_id, []).append(delta)

    def _on_tool_call_end(self, event: Any) -> None:
        """参数收齐，登记任务（``PENDING``）。

        ⚠️ 这是 ``plan`` 而不是 ``add_use``：此刻模型**刚决定**要调这个工具，
        还没执行（执行由框架的 ``on_acting`` 接管，之后才会有结果事件）。
        用 ``plan`` 才能让界面显示出「即将查询航班 → 正在查询航班」两段，
        而这两段在高延迟工具（查酒店、查政策）上差别很明显。

        ⚠️ 参数解析失败**不影响登记**：界面上「有个工具在跑」比「参数好看」
        重要得多。解析不出来就给空字典，并在日志里留一条。

        Args:
            event (`Any`): ``ToolCallEndEvent``。
        """
        call_id = _field(event, "tool_call_id", "") or ""
        if not call_id:
            return
        name = self._call_names.get(call_id, "")
        arguments = self._parsed_arguments(call_id)
        task_id = self._collector.plan(
            name or "未知工具",
            title=title_for(name),
            arguments=arguments,
        )
        self._tasks[call_id] = task_id

    def _on_tool_result_start(self, event: Any) -> None:
        """工具开始执行：``PENDING → DOING``。

        Args:
            event (`Any`): ``ToolResultStartEvent``。
        """
        call_id = _field(event, "tool_call_id", "") or ""
        task_id = self._tasks.get(call_id)
        if task_id is None:
            # ⚠️ 有结果却没有对应的调用登记。可能是我们漏处理了某个事件，
            # 也可能是事件流乱序。**不**补登记 —— 补一个没有名字、没有参数
            # 的任务比不显示更让人困惑。但一定要留日志：这是唯一能发现
            # 「事件映射漏了一种」的线索。
            logger.warning(
                "收到工具 %r 的结果，但没有找到对应的调用登记，已跳过。",
                _field(event, "tool_call_name", call_id),
            )
            return
        self._collector.mark_doing(task_id)

    def _on_tool_result_delta(self, event: Any) -> None:
        """累积结果文本分片。

        Args:
            event (`Any`): ``ToolResultTextDeltaEvent``。
        """
        call_id = _field(event, "tool_call_id", "") or ""
        delta = _field(event, "delta", "") or ""
        if not call_id or not delta:
            return
        self._result_text.setdefault(call_id, []).append(delta)

    def _on_tool_result_end(self, event: Any) -> None:
        """工具结束：``DOING → DONE / FAILED``。

        ⚠️ ``ToolResultState.RUNNING`` **不收尾**。它表示工具还在跑
        （框架用它标记「结果未定」），把任务标成完成或失败都是在说谎 ——
        而用户会照着界面上那个「完成」去理解系统已经拿到了数据。

        ⚠️ 失败时的 ``error`` 优先取工具自己的输出文本。工具的错误信息
        通常已经是一句面向用户的中文（见 ``src/tools/_result.py``），
        比我们在这里按状态码编一句更准确。

        Args:
            event (`Any`): ``ToolResultEndEvent``。
        """
        call_id = _field(event, "tool_call_id", "") or ""
        task_id = self._tasks.get(call_id)
        if task_id is None:
            return

        state = _field(event, "state", None)
        text = "".join(self._result_text.get(call_id, [])).strip()

        if state == ToolResultState.RUNNING:
            return

        if state == ToolResultState.SUCCESS:
            self._collector.add_result(task_id, text, ok=True)
            return

        failure = _FAILURE_TEXT.get(state, "工具执行失败")
        self._collector.add_result(
            task_id,
            ok=False,
            error=text or failure,
        )

    def _on_require_confirm(self, event: Any) -> None:
        """登记一个「等待用户确认」的任务。

        ⚠️ 停在 ``PENDING`` 而不是 ``DOING``：系统此刻**没有在执行**任何
        东西，它在等人。标成 ``DOING`` 会让界面显示一个永远转不完的圈，
        而用户正在犹豫要不要点确认 —— 那是最容易让人以为系统卡住的时刻。

        ⚠️ 同一轮里可能有多个待确认调用，但**只登记一条**任务。确认框在
        界面上是一个整体动作（要么全点要么全不点），拆成多条会让用户
        以为要逐个处理。

        ⚠️ **每条事件只带一个**调用，这一点很容易猜错。已核实
        （``agent/_agent.py:2574-2577``）：框架是**每个被挂起的工具调用各发
        一条** ``RequireUserConfirmEvent``，且 ``tool_calls=[tool_call]``
        长度**恒为 1**。所以「有几项待确认」**不能**取 ``len(tool_calls)``
        —— 那个数永远是 1，界面于是在有 5 个调用等人确认时写着
        「等待你确认 1 项操作」。计数必须在**本方法里跨事件累加**。

        ⚠️ 累加而不是「取最后一条的值」：两种事件形状（单条 vs 批量）
        都可能出现，累加对两者都成立。

        Args:
            event (`Any`): ``RequireUserConfirmEvent``。
        """
        calls = _field(event, "tool_calls", None) or []
        count = len(calls)
        if count == 0:
            return

        self._confirm_count += count

        if self._confirm_task is None:
            self._confirm_task = self._collector.plan(
                CONFIRM_TASK_NAME,
                title=f"等待你确认 {self._confirm_count} 项操作",
                arguments={"pending": self._confirm_count},
            )
            return

        # ⚠️ 已经登记过就**只更新展示字段**，不重复登记，更不能
        # ``mark_doing`` —— 状态必须停在 ``PENDING``（见本方法的开头）。
        # ``PENDING → DOING`` 是**合法**迁移，所以推进它不会触发
        # ``_transition`` 的告警：这条路径错了是**完全静默**的，
        # 只表现为界面上一个永远转不完的圈。
        self._collector.retitle(
            self._confirm_task,
            title=f"等待你确认 {self._confirm_count} 项操作",
            arguments={"pending": self._confirm_count},
        )

    def _on_confirm_result(self, event: Any) -> None:
        """用户给出了确认结果：收尾那条等待任务。

        ⚠️ 用户的**拒绝**不是失败。用户点了「不同意」是系统按预期工作了 ——
        把拒绝标成红色失败，会让用户以为自己做了什么不该做的事。
        处置是「完成，并说明用户没有同意」，见下面的 ``ok=True``。

        ⚠️ 判据字段是 ``confirmed``，**不是** ``approved``
        （``event/_event.py:468-473`` 的 ``ConfirmResult``）。
        写错字段名的后果很隐蔽：``_field(item, "approved", False)``
        永远取到 ``False``，界面于是在用户明明点了同意之后显示
        「用户已确认 0/1 项操作」—— 一句在关键路径上主动误导人的文案，
        而且它不会报错、不会告警，只是数字永远不对。

        ⚠️ 这条事件经由消息总线时是 ``model_dump`` 出来的**字典**，
        而字段名与模型一致，所以 :func:`_field` 一套读法两边通用。

        Args:
            event (`Any`): ``UserConfirmResultEvent``。
        """
        if self._confirm_task is None:
            return
        results = _field(event, "confirm_results", None) or []
        if not results:
            self._collector.add_result(self._confirm_task, "用户已确认", ok=True)
        else:
            yes = sum(1 for item in results if _field(item, "confirmed", False))
            no = len(results) - yes
            summary = f"用户已确认 {yes} 项操作"
            if no:
                # ⚠️ 拒绝要说清楚，但不能说成失败 —— 界面上的红叉会让人
                # 以为「我点错了」或「系统坏了」。这是一句中性的事实陈述。
                summary += f"，另有 {no} 项未同意"
            self._collector.add_result(self._confirm_task, summary, ok=True)
        self._confirm_task = None
        # ⚠️ 计数一并归零：确认结果到了，这批就结束了。不归零的话，
        # 下一批待确认会从上一批的数字继续往上加 —— 界面显示
        # 「等待你确认 7 项操作」而实际只有 2 项。
        self._confirm_count = 0

    def _on_reply_end(self, event: Any) -> None:
        """回复结束：**只在真的失败时**把仍在执行的任务收成失败。

        ⚠️ 不是所有 ``REPLY_END`` 都该收尾。人工确认路径下回复也会结束，
        而那一刻的任务是**正常等待**（人在犹豫要不要点确认），
        把它们标成失败等于替用户做了决定。判据是
        ``ReplyEndEvent.finished_reason``：

            ``COMPLETED``    —— 正常结束，可能正等着人工确认 → 什么都不做
            ``INTERRUPTED``  —— 用户中断；框架**已经**为在途调用补发了
                                ``INTERRUPTED`` 的工具结果
                                （``agent/_agent.py:1008-1031``）→ 什么都不做
            ``EXCEED_MAX_ITERS`` / ``ERROR`` —— 系统放弃了本轮，那些
                                ``DOING`` 的任务不会再有结果 → 收成失败

        ⚠️ 收尾**只针对 ``DOING``**。停在 ``PENDING`` 的任务（等待人工确认的
        ``submit_approval``）不在其中 —— 这一点很重要：审核中的申请单
        在下一轮回复里会被同一个 ``tool_call_id`` 认领并正常收尾。

        ⚠️ 幂等：跑过一次之后就没有 ``DOING`` 了，所以同一个 ``reply_id``
        上重复的 ``REPLY_END``（或紧随其后的废弃 ``EXCEED_MAX_ITERS``
        事件）不会造成第二次收尾。

        Args:
            event (`Any`): ``ReplyEndEvent``。
        """
        reason = _field(event, "finished_reason", None)
        if reason not in _ABANDONED_REASONS:
            return
        if reason == ReplyFinishedReason.ERROR:
            logger.warning(
                "本轮回复因错误终止（%s），已把未完成的任务标记为失败。",
                _field(event, "error", None) or "未提供错误详情",
            )
        self._finish_stuck(_ABANDONED_TEXT.get(reason, _DEFAULT_ABANDONED_TEXT))

    def _on_exceed_max_iters(self, event: Any) -> None:
        """迭代上限事件：**废弃事件的向后兼容入口**。

        ⚠️ 已核实（``event/_event.py:425-431``）：``ExceedMaxItersEvent``
        带 ``@deprecated``，文档字符串写着「仍为向后兼容而发出，但**不携带
        语义**；请改用 ``ReplyEndEvent.finished_reason``」。

        本地版本目前两处都发（``agent/_agent.py:3599`` 等），且事件顺序是
        先 ``EXCEED_MAX_ITERS`` 后 ``REPLY_END``。保留这个分支，是因为
        旧版本框架只发这一个事件；真正的语义来源是
        :meth:`_on_reply_end`。两边都幂等，先到的那次收尾，后到的那次
        找不到 ``DOING`` 任务、什么都不做。

        Args:
            event (`Any`): ``ExceedMaxItersEvent``。
        """
        self._finish_stuck(_ABANDONED_TEXT[ReplyFinishedReason.EXCEED_MAX_ITERS])

    def _finish_stuck(self, error: str) -> None:
        """把仍在执行的任务标成失败。

        ⚠️ 先 ``snapshot()`` 再逐条 ``add_result``，不边遍历边改 ——
        ``add_result`` 会替换字典里的记录，而 ``snapshot`` 返回的是值拷贝，
        遍历拷贝才是安全的。

        Args:
            error (`str`): 给用户看的失败原因。
        """
        stuck = [task for task in self._collector.snapshot() if task.state == _DOING]
        for task in stuck:
            self._collector.add_result(task.task_id, ok=False, error=error)
        if stuck:
            logger.warning("本轮未正常结束，已把 %d 个未完成的任务标记为失败。", len(stuck))

    # --------------------------------------------------------------------------
    # 内部
    # --------------------------------------------------------------------------
    def _parsed_arguments(self, call_id: str) -> dict[str, object]:
        """把参数分片拼起来并解析成字典。

        ⚠️ 解析失败返回**空字典**而不是抛异常，也不返回 ``{"raw": "..."}``
        这类半成品。``arguments`` 的契约是 ``dict[str, object]``，
        塞一个键名不可预测的字典进去，界面就得为它写特例。

        Args:
            call_id (`str`): 工具调用标识。

        Returns:
            `dict[str, object]`: 解析出的参数字典；失败时为空字典。
        """
        raw = "".join(self._call_args.get(call_id, [])).strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            logger.debug("工具参数不是合法 JSON，已按空参数处理：%.200s", raw)
            return {}
        if not isinstance(parsed, dict):
            # ⚠️ 合法 JSON 但不是对象（比如模型发了一个字符串或数组）。
            # 同样按「没有参数」处理：契约是字典。
            logger.debug("工具参数不是 JSON 对象，已按空参数处理。")
            return {}
        return parsed


#: 结果状态 → 兜底失败文案。
#:
#: ⚠️ 工具自己的输出文本优先（见 :meth:`EventChainAdapter._on_tool_result_end`），
#: 这张表只在工具什么话都没说时兜底。它存在的意义是：空的失败原因在界面上
#: 表现为「一个红叉加一片空白」，而失败恰恰是最需要解释的时刻。
_FAILURE_TEXT: dict[Any, str] = {
    ToolResultState.ERROR: "工具执行出错",
    ToolResultState.DENIED: "该操作未获授权，已跳过",
    ToolResultState.INTERRUPTED: "执行被中断",
}

#: ``reply_stream`` 的产出里属于「消息」而非「事件」的类型。
#:
#: ⚠️ 抽成常量而不是在 :meth:`EventChainAdapter.consume_stream` 里直接写
#: ``isinstance(event, Msg)``：``reply_stream`` 的产出联合类型
#: （``agent/_agent.py:288-298``）里 ``Msg`` 只有一种，但把它单列出来是为了
#: 让「哪些类型的元素会被跳过」这件事有一个可搜索的位置 ——
#: 漏跳过一种类型的症状是「思考链里混进一条普通消息」，
#: 那种 bug 不会报错，只会让界面多出一行看不懂的东西。
_MESSAGE_TYPES: tuple[type, ...] = (Msg,)


def iter_task_states(collector: TaskCollector) -> Iterable[dict[str, Any]]:
    """把任务清单拍平成可序列化的字典（供 SSE 推给前端）。

    ⚠️ ``duration_seconds`` 可能是 ``None``（任务未结束），**保留** ``None``
    而不是填 0 —— 前端要靠它区分「还没跑完」和「一瞬间就跑完了」，
    填 0 会让两种完全不同的情况长得一样。

    ⚠️ **不含** ``arguments`` 里的原始值之外的东西：``TaskRecord`` 是
    ``frozen`` 的，这里只读不写。

    Args:
        collector (`TaskCollector`): 任务收集器。

    Returns:
        `Iterable[dict[str, Any]]`: 每个任务一个字典。
    """
    for task in collector.snapshot():
        yield {
            "task_id": task.task_id,
            "name": task.name,
            "title": task.title,
            "state": task.state.value,
            "arguments": dict(task.arguments),
            "result": task.result,
            "error": task.error,
            "duration_seconds": task.duration_seconds,
        }


#: 「执行中」的任务状态。
#:
#: ⚠️ 只在 :meth:`EventChainAdapter._finish_stuck` 里用它筛出「卡住的
#: 任务」。**只在那里用** —— 别处一律不要按状态筛任务，理由见
#: :meth:`EventChainAdapter._on_reply_end`。
_DOING = TaskState.DOING

#: 回复以这些原因结束 = 系统放弃了本轮，在途任务不会再有结果。
#:
#: ⚠️ ``COMPLETED`` 与 ``INTERRUPTED`` **刻意不在**里面：
#: 前者可能是人工确认的正常暂停；后者的在途调用框架已经补发过结果。
_ABANDONED_REASONS: frozenset[Any] = frozenset(
    {
        ReplyFinishedReason.EXCEED_MAX_ITERS,
        ReplyFinishedReason.ERROR,
    },
)

#: 放弃原因 → 面向用户的失败文案。
#:
#: ⚠️ 与 :data:`_FAILURE_TEXT` 同样：文案要说「这一步没做完」而不是
#: 抛内部术语。「迭代上限」对用户没有意义，他需要知道的是
#: **刚才那件事没办成，要不要再来一次**。
_ABANDONED_TEXT: dict[Any, str] = {
    ReplyFinishedReason.EXCEED_MAX_ITERS: "本轮处理步数已达上限，该步骤未能完成。",
    ReplyFinishedReason.ERROR: "本轮处理出错，该步骤未能完成，请重试。",
}

#: 放弃原因认不出时的兜底文案。
_DEFAULT_ABANDONED_TEXT = "本轮处理未能完成，该步骤已中止。"


__all__ = [
    "CONFIRM_TASK_NAME",
    "TITLES",
    "EventChainAdapter",
    "iter_task_states",
    "title_for",
]
