# -*- coding: utf-8 -*-
"""``/api/v1/sessions/{session_id}/chains`` —— 思考链的 **SSE** 端点。

文件职责：
    把一场会话的事件流转成「这份对话现在在干什么」的实时投影，以 SSE 推出去。
    上游是消息总线上 ``agentscope:session:events:{sid}`` 里的 ``AgentEvent``
    字典（框架的 ``AppService`` 每条事件都做 ``model_dump(mode="json")``）。

上下游依赖：
    - 上游：``agentscope.app.message_bus``（回放日志 + 实时订阅）、
      ``agentscope.app.deps``（身份 / 存储 / 总线三个依赖）。
    - 中转：``src/chains/events.py`` 的 :class:`~src.chains.events.EventChainAdapter`
      （事件 → 收集器）与 :func:`~src.chains.events.iter_task_states`（收集器 → 扁平字典）。

⚠️ **本端点目前没有前端消费方** —— 这一点必须写在最前面，因为它决定了
    这条代码路径的**验证程度**，而验证程度决定了改它时该有多小心：

        · 本仓库交付的前端产物（``src/server/static/``，以及 ``web/`` 下的
          源码）里搜不到 ``chains`` 这个词，一次调用都没有；
        · 界面上的思考链走的是框架自带的 ``/sessions/{id}/stream``
          原始事件流，前端自己渲染。

    也就是说，它是一条**已经实现、已被端到端测试覆盖、但线上无人调用**的
    能力（服务端把投影算完再发，见下节理由）。它的正确性由
    ``tests/test_chains_stream_endpoint.py`` 保证（含超大日志回放、
    先订阅后回放、重连、多租户隔离等用例），**不是**由「有人在用」保证的。

    反过来也要说清楚：**不要**因为「没人用」就顺手删掉。它与框架
    ``/sessions/{id}/stream`` 的取舍（原始事件 vs 加工好的任务清单）是一处
    有意为之的设计，删掉它等于把「前端各自实现一遍状态机」重新变成唯一选项。
    接入方出现之前，它是**可用的备用入口**；接入方出现之后，改它的行为
    要按**线上接口**对待。

==============================================================================
为什么另起一条端点，而不复用框架的 ``/sessions/{id}/stream``
==============================================================================
    框架那条推的是**原始事件**（REPLY_START / TEXT_BLOCK_DELTA / ...），
    前端要自己维护一套「哪些工具正在跑、谁成功了」的状态机 —— 而那套状态机
    才是容易写错的部分：参数是分片到达的、结果也是分片的、人工确认之后还会
    补发一个 ``reply_id`` 相同的续接 ``REPLY_START``（见 ``src/chains/events.py``
    的模块文档）。把它放在前端，等于让每个前端实现各写一遍，且错了不报错、
    只表现为「进度条不消失」。

    本端点把那份状态机（已经在 :mod:`src.chains.collector` 里被穷举测试过）
    放到服务端跑，推的是**已经加工好的任务清单**。代价是多一条连接、
    多一份每连接的投影状态 —— 而这个代价是刻意接受的：

        · 投影是**投影**，算错了不影响业务数据（见收集器的模块文档）；
        · 前端因此可以对这条连接完全无状态，断了重连即可。

==============================================================================
⚠️ 每个连接各建一份 collector + adapter —— 绝不能共享
==============================================================================
    :class:`~src.chains.collector.TaskCollector` 与
    :class:`~src.chains.events.EventChainAdapter` 都是**一轮回复作用域**、
    **非线程安全**的：adapter 内部按 ``tool_call_id`` 索引参数/结果分片，
    跨连接复用会把 A 会话的分片拼进 B 会话的结果里（而 ``tool_call_id``
    是模型生成的，两条会话里未必不同）。

    所以它们在这里**就地**构造（``_sse_generator`` 内），随连接生、随连接灭。
    做成模块级单例是最自然也最坏的一种写法：本地单人联调一切正常，
    两个人同时用就开始串数据，且没有任何报错。

==============================================================================
为什么发**全量快照**，而不是增量（这是刻意的取舍）
==============================================================================
    :func:`iter_task_states` 每次产出的都是**全量**任务清单。两个选择：

        (a) 每来一条事件就重发一遍全量 —— 最省事，但一轮回复里事件可能有
            几百条（每个 token 一条 TEXT_BLOCK_DELTA），绝大多数帧的内容与
            上一帧**完全一样**，纯属刷屏。
        (b) **只在投影真的变了才发**（本端点的选择）。

    这里选 (b)：状态没变就不发，变了就发**新的全量**。不做「发增量」——
    增量要求前端自己维护合并逻辑，一旦丢一个包或重连一次就永久错位，
    而它错位的表现是「界面上少了一个任务」，没人会往「增量丢了」上想。
    全量快照是**自描述**的：前端每次直接整体替换即可，重连也天然自愈。
    代价只是一帧多大几十字节，而任务清单有上限
    （:data:`~src.chains.collector.DEFAULT_MAX_TASKS`），不会无界膨胀。

    「变了才发」用一个上一次已发送的序列化串比对来实现：任务清单是纯数据、
    推理文本是字符串，序列化后逐字节比较就是最直白的「变没变」。

==============================================================================
⚠️ ``expose_reasoning`` 默认 **false**
==============================================================================
    模型的推理文本可能含用户**没说过的**个人信息（模型从上下文里推断出来的
    出发地、职级、偏好……）。默认关掉，而且不只是「不显示」—— 传下去之后
    :class:`~src.chains.events.EventChainAdapter` 会**根本不累积**它
    （见该类的 ``__init__``）。要看推理请显式 ``expose_reasoning=true``，
    这是一个需要调用方**明确表态**的动作。

==============================================================================
回放与订阅之间的窗口：先订阅，后回放
==============================================================================
    框架自带的 ``/sessions/{id}/stream`` 是「先回放、后订阅」。那样在
    ``log_read`` 与订阅真正建立之间发布的事件会**同时错过两处**（回放已经
    读完、订阅还没挂上），表现为「偶尔少几个事件」，而且只在负载高时出现。

    本端点的顺序反过来：**先把订阅挂上（``on_ready`` 回调保证挂上了），
    再读回放日志**。这样窗口内发布的事件一定落在订阅队列里，不会丢。

    代价是「回放日志」与「订阅队列」会有**重叠**：一条事件先 ``log_append``
    再 ``publish``（``agentscope/app/_bus_ops.py:40-66``），若它恰好落在「订阅已挂、
    回放未读」之间，两处都能拿到。所以下面按 ``_entry_id`` 去重 —— 那个 id
    由总线的 ``log_append`` 分配，回放与实时是**同一个值**。去重是必需的：
    重复消费一次 ``TOOL_CALL_END`` 会登记出**重复的任务**。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import deque
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

# 框架自己的三个依赖。与 ``_identity.py`` 同样在**顶层导入**：``Depends(...)``
# 的参数在装饰器求值那一刻就计算，藏进延迟导入只是看着惰性。
from agentscope.app.deps import (
    get_current_user_id,
    get_message_bus,
    get_storage,
)
from agentscope.app.message_bus import MessageBus, MessageBusKeys
from agentscope.app.storage import StorageBase

from src.chains.collector import TaskCollector
from src.chains.events import EventChainAdapter, iter_task_states

logger = logging.getLogger(__name__)

#: SSE 心跳间隔（秒）。
#:
#: ⚠️ 与框架 ``agentscope/app/_router/_session.py:717`` 的 ``_HEARTBEAT_INTERVAL_SECS``
#: 取值一致（30）。**不复用框架那个常量**：它是私有模块属性，导入它等于把
#: 我们绑在框架的内部结构上；而这两个值是否相等本身没有语义约束 ——
#: 各自表达的是「本连接允许多久没有内容」。保持一致只是为了让两条连接在
#: 同一个反向代理下行为相同。
#:
#: ⚠️ 必须在**生成器循环体里**读这个模块全局（而不是当函数默认参数），
#: 否则测试没法把它压到 0.25 秒来验心跳 —— 默认参数在定义时求值一次。
_HEARTBEAT_INTERVAL_SECS = 30

#: 等待「订阅真正建立」的超时（秒）。
#:
#: ⚠️ 这个超时**不是**可选的保险，而是把一类**永久挂起**变成一类**可观测失败**：
#: ``on_ready`` 只在订阅**成功**建立后才被调用（``_bus_ops`` 会先做一次真实往返，
#: 例如 Redis 的 ``SUBSCRIBE``）。因此只要那一步抛异常（Redis 短暂不可达、超时），
#: ``ready`` 就永远不会置位 —— 少了这个超时，``await ready.wait()`` 会**永远**
#: 卡在那里：既没有数据帧、也没有心跳、也不断开。客户端拿到的是一条
#: ``200 text/event-stream`` 的**死连接**（浏览器不会自动重连，因为它认为连接正常），
#: 思考链面板于是永久空白，且服务端**一条日志都没有**。
#:
#: 取值与框架自己的同类实现一致（``app/channel/_stream.py`` 的
#: ``_SUBSCRIBE_TIMEOUT_SECS``）：5 秒对「建一条本地订阅」是极宽松的余量，
#: 真超时说明总线确实不可用。
_SUBSCRIBE_TIMEOUT_SECS = 5.0

#: 回放日志**向前翻页**的页数上限。
#:
#: ⚠️ 为什么需要翻页（见 :func:`_read_replay_tail`）：``log_read`` 是**从头**读最多
#: ``max_count`` 条，而 Redis 侧写入用的是 ``XADD MAXLEN ~N``（**近似**裁剪），
#: 日志长度可以**超过**上限，于是最新的若干条恰好落在单次读取的窗口之外。
#: 8 页 × 1000 条 = 8000 条，远超一轮回复的事件量；到顶只可能是异常情况，
#: 那时会打一条 warning 而不是静默截断。
_REPLAY_MAX_PAGES = 8

#: 一条 SSE 连接上「思考链没变化」时的**初始**序列化值。
#:
#: ⚠️ 必须先塞进 ``last_sent``：不塞的话，第一条事件（比如 ``REPLY_START``，
#: 它不改动任务清单也不带推理）就会因为「和 ``None`` 不同」而推出一帧
#: 空快照。那一帧没有任何信息量，却会让前端以为思考链刷新了。
_EMPTY_SNAPSHOT = json.dumps({"tasks": [], "reasoning": ""}, ensure_ascii=False)

router = APIRouter()


async def _read_replay_tail(
    bus: MessageBus,
    key: str,
    *,
    max_count: int,
) -> list[tuple[str, dict]]:
    """读回放日志**最后** ``max_count`` 条（而不是最旧的 ``max_count`` 条）。

    ⚠️ 为什么不能直接用 ``log_read(key, max_count=N)``：
    ``log_read`` 的语义是「从 ``since`` 之后**按追加顺序**读最多 N 条」，
    两个实现都是**从头**读（内存实现 ``log[:max_count]``、
    Redis 实现 ``XRANGE`` 从最小 id 起数）。而 Redis 侧的写入用的是
    ``XADD MAXLEN ~N``（``_redis_message_bus.py`` 里 ``approximate=True``），
    **近似**裁剪只保证长度不小于上限，实际可以多出若干条（默认多一个
    宏节点，约 100 条）。于是当日志长度 > ``max_count`` 时：

        · 单次读取拿到的是**最旧**的 N 条；
        · 最新的那几条既不在读取窗口里，又因为发布发生在**订阅建立之前**
          而不在订阅队列里 —— 两处都没有，直接丢失。

    丢的恰恰是**最重要**的那几条（例如 ``TOOL_RESULT_END``），后果是任务
    永远停在「进行中」：前端收到的全量快照里那个工具条一直转圈，
    而且**重连也自愈不了**（重连走的还是同一段回放）。

    所以这里按游标**向前翻页**，只保留末尾 N 条。代价可忽略：
    正常情况下第一页就会返回不足 ``max_count`` 条并立即结束，
    与原来一样只有一次读取。

    Args:
        bus (`MessageBus`): 消息总线。
        key (`str`): 回放日志的键。
        max_count (`int`): 需要保留的尾部条目数。

    Returns:
        `list[tuple[str, dict]]`: ``(entry_id, payload)``，按追加顺序，
        最多 ``max_count`` 条。
    """
    tail: deque[tuple[str, dict]] = deque(maxlen=max_count)
    since: str | None = None
    for page in range(_REPLAY_MAX_PAGES):
        entries = await bus.log_read(key, since=since, max_count=max_count)
        if not entries:
            return list(tail)
        tail.extend(entries)
        since = entries[-1][0]
        if len(entries) < max_count:
            return list(tail)
    # 到页数上限：说明日志比 _REPLAY_MAX_PAGES × max_count 还长。
    # 明确打出来而不是静默截断 —— 「少了几帧」是最难归因的那种现象。
    logger.warning(
        "回放日志 %s 超过 %d 页（每页 %d 条），只回放了最后 %d 条；"
        "更早的事件未回放。",
        key,
        _REPLAY_MAX_PAGES,
        max_count,
        max_count,
    )
    return list(tail)


@router.get(
    "/sessions/{session_id}/chains",
    summary="订阅一场会话的思考链（SSE）",
    description=(
        "以 ``text/event-stream`` 持续推送这场会话的思考链**全量快照**"
        "（每帧是一个对象，含 tasks 与 reasoning 两个字段）。"
        "先回放当前一轮的缓冲事件，再续上实时事件；"
        "空闲时约有每 30 秒一次的 ``:\\n\\n`` 心跳帧。"
        "任务清单里的 ``state`` 取值见 ``TaskState``；"
        "``reasoning`` 默认恒为空串（需显式 ``expose_reasoning=true``）。"
    ),
    responses={
        200: {"description": "SSE 流已建立"},
        404: {"description": "会话不存在或不属于当前身份"},
    },
)
async def session_chains(
    session_id: str,
    agent_id: str = Query(description="会话所属的智能体 id（用于属主校验）。"),
    expose_reasoning: bool = Query(
        default=False,
        description=(
            "是否把模型推理文本一并推送。**默认关闭** —— "
            "推理文本可能含用户没提供过的个人信息。"
        ),
    ),
    user_id: str = Depends(get_current_user_id),
    storage: StorageBase = Depends(get_storage),
    message_bus: MessageBus = Depends(get_message_bus),
) -> StreamingResponse:
    """订阅会话思考链的实时 SSE 流。

    Args:
        session_id (`str`): 目标会话 id。
        agent_id (`str`): 会话所属的智能体 id。与框架的 ``/stream`` 一样，
            **必填** —— 属主校验需要它（``get_session`` 的三元组键）。
        expose_reasoning (`bool`): 是否推送模型推理文本。
        user_id (`str`): 由框架身份依赖解析出的用户标识。
        storage (`StorageBase`): 存储后端（只用于属主校验）。
        message_bus (`MessageBus`): 消息总线（回放 + 实时订阅）。

    Returns:
        `StreamingResponse`: ``text/event-stream``，逐帧推送全量快照。

    Raises:
        HTTPException: 会话不存在或不属于 ``user_id`` 时 404。
            ⚠️ 刻意返回 404 而不是 403：403 会告诉调用方「这个会话是存在的，
            只是不属于你」—— 那就把「谁拥有什么」这种信息泄露出去了。
            404 让「不存在」与「不是你的」**不可区分**，这正是隔离应有的样子
            （框架的 ``/messages`` ``/stream`` ``/status`` 全是这个做法）。
    """
    existing = await storage.get_session(user_id, agent_id, session_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    bus_key = MessageBusKeys.session_events(session_id)

    async def _sse_generator() -> AsyncGenerator[str, None]:
        """逐帧产出思考链快照 + 心跳。

        ⚠️ 收集器与适配器在**这里**构造（每次连接各一份），理由见模块文档。

        Yields:
            `str`: SSE 帧文本（``data: {...}\\n\\n`` 或心跳 ``:\\n\\n``）。
        """
        collector = TaskCollector()
        adapter = EventChainAdapter(
            collector,
            expose_reasoning=expose_reasoning,
        )
        # 「上一次已发送的序列化快照」。用字符串比对来判定「变没变」——
        # 见模块文档里「为什么发全量快照」。
        last_sent = _EMPTY_SNAPSHOT

        def _changed_frame() -> str | None:
            """投影有变化时返回一帧，否则返回 ``None``。

            Returns:
                `str | None`: SSE 帧，或 ``None``（与上一帧相同）。
            """
            nonlocal last_sent
            payload = json.dumps(
                {
                    "tasks": list(iter_task_states(collector)),
                    # 关掉推理时 ``adapter.reasoning`` 恒为空串（它根本没累积），
                    # 但字段**始终存在** —— 保持帧的形状稳定，前端不必为
                    # 「开没开推理」写两套解析。
                    "reasoning": adapter.reasoning,
                },
                ensure_ascii=False,
            )
            if payload == last_sent:
                return None
            last_sent = payload
            return f"data: {payload}\n\n"

        # ---- 1. 先把订阅挂上 ------------------------------------------------
        # 用一个后台 feeder 任务跑 ``subscribe``：不能在主循环里直接
        # ``wait_for(__anext__())``，因为取消一个挂起的 ``__anext__`` 会让
        # 异步生成器停在 "running" 状态、``aclose()`` 再也关不掉它
        # （框架 ``_session.py`` 的同一处注释）。
        queue: asyncio.Queue[dict | None] = asyncio.Queue()
        ready = asyncio.Event()

        async def _feeder() -> None:
            """把订阅到的 payload 转发进队列；订阅结束时投一个 ``None``。

            推 ``None`` 作为哨兵：订阅在本项目里只会在总线关闭时结束，
            那时主循环应当退出而不是空转。
            """
            try:
                async for payload in message_bus.subscribe(
                    bus_key,
                    on_ready=ready.set,
                ):
                    # ⚠️ **保留** ``_entry_id``（框架自己会在 /stream 里剥掉它）：
                    # 下面要靠它把「回放日志」与「订阅队列」的重叠去重。
                    await queue.put(dict(payload))
            except asyncio.CancelledError:
                pass
            finally:
                await queue.put(None)

        feeder_task = asyncio.create_task(
            _feeder(),
            name=f"chains-feeder:{session_id}",
        )

        #: 已经消费过的回放条目 id，用来丢弃订阅队列里的重复事件。
        seen_entry_ids: set[str] = set()

        try:
            # ``on_ready`` 在订阅真正建立后、首个 payload 前恰好调用一次，
            # 所以这一行之后「发布出来的事件一定进得了队列」。
            #
            # ⚠️ 必须带超时（理由见 ``_SUBSCRIBE_TIMEOUT_SECS`` 的注释）：
            # 订阅建立失败时 ``on_ready`` 不会被调用，无超时的 ``wait()``
            # 会让这条 SSE **永久挂起**且不留任何日志。
            try:
                await asyncio.wait_for(
                    ready.wait(),
                    timeout=_SUBSCRIBE_TIMEOUT_SECS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "思考链订阅未能在 %.1f 秒内建立（会话 %s）："
                    "消息总线不可用或不响应。本条 SSE 将以断开收尾"
                    "（前端会看到连接结束并可重试）。",
                    _SUBSCRIBE_TIMEOUT_SECS,
                    session_id,
                )
                return

            # ---- 2. 回放当前一轮的缓冲事件 ----------------------------------
            # ⚠️ 用 ``_read_replay_tail`` 而不是直接 ``log_read``：要的是
            # 日志**末尾**的 N 条（原因见该函数的 docstring —— Redis 的近似
            # 裁剪会让日志超过上限，最旧的 N 条恰恰不含最新事件）。
            for entry_id, event in await _read_replay_tail(
                message_bus,
                bus_key,
                max_count=MessageBusKeys.SESSION_REPLAY_MAX_LEN,
            ):
                seen_entry_ids.add(entry_id)
                adapter.consume(event)
                frame = _changed_frame()
                if frame is not None:
                    yield frame

            # ---- 3. 续上实时事件 --------------------------------------------
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(),
                        timeout=_HEARTBEAT_INTERVAL_SECS,
                    )
                except asyncio.TimeoutError:
                    # 超时 = 这段时间没有任何事件。发一帧注释（SSE 规范里
                    # 以 ``:`` 开头的行会被客户端忽略），用来：
                    #   · 穿过会掐空闲连接的反向代理；
                    #   · 让前端能区分「还活着但没事发生」与「连接已经死了」。
                    yield ":\n\n"
                    continue

                if item is None:
                    # 订阅结束 = 消息总线关闭。这**不是**正常的收尾路径
                    # （请求结束时是被取消掉的），留一条日志是因为此时前端
                    # 只会看到「流不动了」，而原因是服务端的总线没了 ——
                    # 没有这条日志就完全无从查起。
                    logger.warning(
                        "思考链订阅已结束（消息总线关闭），会话 %s 的 SSE 将断开。",
                        session_id,
                    )
                    break

                entry_id = item.get("_entry_id")
                if isinstance(entry_id, str):
                    if entry_id in seen_entry_ids:
                        # 回放与订阅的重叠（见模块文档）。重复消费一次
                        # ``TOOL_CALL_END`` 会多登记一个任务，必须丢掉。
                        continue
                    seen_entry_ids.add(entry_id)

                adapter.consume(item)
                frame = _changed_frame()
                if frame is not None:
                    yield frame
        finally:
            feeder_task.cancel()
            try:
                await feeder_task
            except asyncio.CancelledError:
                pass
            except Exception:  # noqa: BLE001
                # ⚠️ 必须在这里吞掉并记录，而不是让它冒到响应任务上：
                # feeder 抛异常 = 订阅在建立或运行期失败（总线不可达、
                # 订阅被关闭）。此时响应**已经开始**（状态码早就发出去了），
                # 异常无法再变成错误响应，只会变成一条「响应任务里的
                # 未处理异常」—— 那既不是错误响应，也是一条容易被人忽略的
                # 噪声栈。收敛成一条 warning，让「为什么这条流断了」
                # 在日志里有迹可循。
                logger.warning(
                    "思考链订阅异常退出（会话 %s）：后续事件不会再推送。",
                    session_id,
                    exc_info=True,
                )

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            # ``no-cache``：中间层不得缓存 SSE（缓存了就等于把一条长连接
            # 变成一张一次性快照）。``X-Accel-Buffering: no``：显式让 Nginx
            # 关掉响应缓冲 —— 否则它会攒够一块才转发，流式就变成了「攒批」。
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


__all__ = ["router"]
