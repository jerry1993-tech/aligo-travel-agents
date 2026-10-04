# -*- coding: utf-8 -*-
"""P2 端到端验收：一句话流式返回 + 多租户隔离。

==============================================================================
这个文件回答的是 P2 的**验收问题**，不是单元问题
==============================================================================
    中间件、存储、路由各自都有单测，但 P2 的验收线是**一句话**：

        「``POST /chat/`` → ``GET /sessions/{id}/stream`` 收到
          ``REPLY_START`` + ``TEXT_BLOCK_DELTA``（含 30s 心跳帧 ``:\\n\\n``）；
          两个不同 ``X-User-ID`` 互相看不到对方会话。」

    这条线上没有任何一个单测能覆盖 —— 它要的是**真的**跑一遍：真的起
    uvicorn、真的开 TCP、真的等模型逐字吐字、真的换一个身份去摸别人的会话。

==============================================================================
为什么必须起真实的 uvicorn（而不是 httpx.ASGITransport）
==============================================================================
    ``ASGITransport`` 会**把整个响应体缓冲完**再交还给调用方。对普通 JSON
    接口这没问题，对 SSE 是致命的：``/sessions/{id}/stream`` 是一个
    **永不结束**的响应，缓冲意味着 ``aiter_lines()`` 一个字节都拿不到，
    用例会一直挂到超时。

    本项目第一次跑 SSE 的症状正是「连状态行都没打印出来」—— 不是报错，
    是**什么都没有**，极难定位。这也是 ``conftest.py`` 里的 ``client``
    夹具（走 ASGITransport）**不能**用来测流式的原因。

    对策：在**同一个事件循环**里起一个真的 ``uvicorn.Server``，绑
    ``127.0.0.1:0``（端口交给内核分配，避免并行跑测试时撞端口），
    用真的 TCP 客户端连它 —— 见 :func:`live_server`。

==============================================================================
★ 时序陷阱：SSE 必须**先于** ``POST /chat/`` 建立连接
==============================================================================
    这是本项目最反直觉、也最容易在联调时踩到的一处，而且它**不报错**。

    ``app/_service/_chat.py`` 在 ``_persist()`` 里调用
    ``await self._message_bus.log_trim(events_key)``，而 ``log_trim(key)``
    的实现是 ``self._logs.pop(key, None)`` —— 它把这场对话的
    **回放日志整个丢掉**。

    于是：一轮对话结束之后再去连 SSE，重新连上来的订阅者既收不到实时事件
    （对话已经结束）、也收不到历史回放（日志已被 trim），
    客户端看到的是一个**永远只有心跳、没有任何内容**的连接。

    所以流式接口的正确用法是：**先连 SSE，再发消息**。前端必须这么做，
    本文件里 :func:`_bootstrap` 之后的每个用例也都这么做，
    并由 :func:`test_reconnect_after_reply_gets_heartbeat_but_no_replay`
    把这个陷阱钉成回归测试。

==============================================================================
隔离性只在**读写会话内容**的端点上断言
==============================================================================
    实测：

        GET /sessions/{sid}/messages   他人 404    ← 断言
        GET /sessions/{sid}/stream     他人 404    ← 断言
        GET /sessions/{sid}/status     他人 404    ← 断言
        GET /sessions/?agent_id=...    他人 404    ← 断言

    而 ``POST /chat/`` 是**发后不理**（fire-and-forget）的：它对
    ``session_id`` 不做同步的属主校验，别人的（甚至根本不存在的）会话 id
    照样返回 ``{"status": "started"}``。这是框架的设计
    （见 ``_router/_schema/_chat.py`` 的 ``ChatTriggerResponse``），
    不是我们的疏漏 —— 但也意味着**不能把 ``/chat/`` 的状态码当作授权信号**。

    本文件因此只把隔离性断言在真正会回内容的端点上，并单独用一条用例
    (:func:`test_chat_trigger_status_is_not_an_authorization_signal`)
    把这个反直觉的事实钉下来，免得后来者把它误当漏洞去「修」——
    修不动，那是框架内部的异步派发语义。
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator

import httpx
import pytest_asyncio
import uvicorn
from agentscope.app.message_bus import MessageBusKeys
from agentscope.event import EventType
from fastapi import FastAPI

# ==============================================================================
# 常量
# ==============================================================================

#: 两个互相看不见对方的身份。取值本身无意义，只要**不同**即可。
ALICE_HEADERS = {"X-User-ID": "alice"}
BOB_HEADERS = {"X-User-ID": "bob"}
#: 第三个身份：专门用于「开箱即用」用例 —— 它**从不**手工建凭据，
#: 只能靠降级播种拿到模型配置。用一个独立身份是为了不受其它用例
#: （它们会给自己建凭据）的影响。
CAROL_HEADERS = {"X-User-ID": "carol"}

#: 假凭据类型。它由 ``src/server/app.py`` 通过 ``extra_credentials`` 注册，
#: 对应 ``src/llm/mock.py::MockCredential`` —— 它保证本文件**零密钥**也能跑：
#: 不联网、不花钱、输出确定。
MOCK_CREDENTIAL_TYPE = "aligo_mock_credential"

#: MockLLM 的模型名。走 ``get_model`` 时 credential 记录里存的就是它。
MOCK_MODEL_NAME = "mock-model"

#: 心跳间隔所在的模块全局。
#:
#: ⚠️ 心跳的 ``timeout=_HEARTBEAT_INTERVAL_SECS`` 是在 SSE 生成器的**循环体里**
#:    每次求值的（``_router/_session.py`` 的 ``_sse_generator``），
#:    所以 monkeypatch 这个模块属性就能改掉它 —— 不必等 30 秒。
#:    这正是「读模块全局而不是把常量内联进函数」带来的可测性；
#:    如果哪天有人把它改成函数默认参数，这条用例会立刻失败（默认参数在
#:    函数定义时求值一次，之后再改模块属性无效），失败信息里也写了原因。
HEARTBEAT_INTERVAL_GLOBAL = "agentscope.app._router._session._HEARTBEAT_INTERVAL_SECS"

#: 心跳用例里把间隔压到的值。0.25s × 若干次 ≈ 1 秒内必定至少来两帧，
#: 比原值 30s 快 120 倍，用例总耗时仍在 1 秒量级。
FAST_HEARTBEAT_SECS = 0.25

#: 默认读超时。SSE 是长连接，读超时只用来兜住「卡死」而不是「等得久」，
#: 因此给它一个远大于单轮 MockLLM 回复时间的值。
READ_TIMEOUT_SECS = 30.0


# ==============================================================================
# 夹具：真实 uvicorn 服务器
# ==============================================================================
@dataclass
class LiveServer:
    """一个跑在真实 TCP 上的测试服务器。

    Attributes:
        base_url (`str`): 形如 ``http://127.0.0.1:53412``，已含真实端口。
        bus (`Any`): 该应用正在使用的消息总线。用例据此**确定性地**等待
            SSE 订阅者挂上（见 :func:`wait_for_subscriber`），而不是
            ``sleep(0.3)`` 赌一把。
    """

    base_url: str
    bus: Any


@pytest_asyncio.fixture
async def live_server(app: FastAPI) -> AsyncIterator[LiveServer]:
    """在与用例**同一个事件循环**里起一个真实的 uvicorn。

    ⚠️ 为什么必须是同一个事件循环：
        ``InMemoryMessageBus`` 的订阅者是 ``asyncio.Queue``，它绑定在创建它的
        那个循环上。若把 uvicorn 放到另一个线程/循环里跑，SSE 的订阅与
        ``POST /chat/`` 的发布就落在两个循环上，事件永远送不到 ——
        症状是「连接建得起来、心跳也有，就是没有内容」。

    ⚠️ 这里**不**用 ``client`` 夹具：它在 lifespan 里跑 ASGITransport，
        而 uvicorn 会自己进入 lifespan。两者同时跑等于把同一个应用的
        启动流程执行两遍（重复建引擎、重复订阅总线）。本文件的用例
        只声明 ``live_server``。

    Args:
        app (`FastAPI`): conftest 装配好的测试应用（**尚未**进入 lifespan）。

    Yields:
        `LiveServer`: 指向真实端口、可被 httpx 直连的服务器句柄。
    """
    # port=0：让内核分配空闲端口。写死端口会在 `pytest -n auto` 或
    # 本机已有一个 dev 服务时直接撞车，而报错是令人困惑的「address in use」。
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="error",  # 用例输出里不需要 uvicorn 的访问日志
        lifespan="on",  # 必须显式开：关掉的话 chat_service 之类全不存在
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())

    # 轮询 started 而不是 sleep 固定值：启动耗时随机器负载浮动，
    # 固定 sleep 要么慢要么偶发失败。
    try:
        async with asyncio.timeout(READ_TIMEOUT_SECS):
            while not server.started:
                if serve_task.done():
                    # serve() 提前返回 = 启动就失败了（多半是 lifespan 抛异常）。
                    # 把异常取出来原样抛，否则只会看到「server 没起来」。
                    serve_task.result()
                    raise AssertionError("uvicorn 在启动完成前就退出了。")
                await asyncio.sleep(0.01)

        port = server.servers[0].sockets[0].getsockname()[1]
        yield LiveServer(base_url=f"http://127.0.0.1:{port}", bus=app.state.message_bus)
    finally:
        server.should_exit = True
        try:
            await asyncio.wait_for(serve_task, timeout=READ_TIMEOUT_SECS)
        except (asyncio.TimeoutError, asyncio.CancelledError):  # pragma: no cover
            serve_task.cancel()


# ==============================================================================
# 工具：HTTP 与 SSE
# ==============================================================================
def _json(response: httpx.Response, *, expect: int = 200) -> dict[str, Any]:
    """断言状态码并返回 JSON。

    ⚠️ 断言里带上响应体：``/credential/`` 这类接口在参数不对时返回 422，
    而 422 的响应体里才有「哪个字段错了」—— 只报状态码等于让人去猜。

    ⚠️ ``expect`` 必须**显式**写出来，不写成「任意 2xx 都行」：
        本文件里各接口的期望码并不一致 —— 框架的三个**创建**端点
        （``/credential/`` ``/agent/`` ``/sessions/``）统一返回 **201 Created**，
        而 ``POST /chat/`` 返回 **200**（它是触发一次对话，不是创建资源）。
        写死具体码，接口悄悄改语义时才会红；放宽成 2xx 就把这层检查丢掉了。

    Args:
        response (`httpx.Response`): 待检查的响应。
        expect (`int`): 期望的状态码。

    Returns:
        `dict`: 解析后的 JSON 对象。
    """
    assert response.status_code == expect, (
        f"{response.request.method} {response.request.url.path} "
        f"返回 {response.status_code}（期望 {expect}）：{response.text[:400]}"
    )
    return response.json()


@dataclass
class AgentFixture:
    """一套「可以开始对话」的最小上下文。"""

    credential_id: str
    agent_id: str
    session_id: str


async def _bootstrap(client: httpx.AsyncClient, headers: dict[str, str]) -> AgentFixture:
    """为一个身份创建「凭据 → 智能体 → 会话」三个对象。

    这三步是框架的既定流程，缺一不可：
        · **凭据**决定用哪个 ``ChatModelBase`` 子类（这里是 Mock）；
        · **智能体**是身份的载体，会话挂在它下面；
        · **会话**才是 ``/chat/`` 与 ``/stream`` 的操作对象。

    Args:
        client (`httpx.AsyncClient`): 指向 live_server 的客户端。
        headers (`dict[str, str]`): 该身份要带的请求头（``X-User-ID``）。

    Returns:
        `AgentFixture`: 三个对象的 id。
    """
    # ``{"data": {...}}`` 这层包装是 ``/credential/`` 的接口约定，不是我们的设计。
    credential = _json(
        await client.post(
            "/credential/",
            headers=headers,
            json={"data": {"type": MOCK_CREDENTIAL_TYPE, "name": "P2 验收 Mock 凭据"}},
        ),
        expect=201,
    )
    agent = _json(
        await client.post(
            "/agent/",
            headers=headers,
            json={"name": "差旅助手", "system_prompt": "你是差旅助手。"},
        ),
        expect=201,
    )
    session = _json(
        await client.post(
            "/sessions/",
            headers=headers,
            json={
                "agent_id": agent["agent_id"],
                "name": "P2 验收会话",
                "chat_model_config": {
                    "type": MOCK_CREDENTIAL_TYPE,
                    "credential_id": credential["credential_id"],
                    "model": MOCK_MODEL_NAME,
                    "parameters": {},
                },
            },
        ),
        expect=201,
    )
    return AgentFixture(
        credential_id=credential["credential_id"],
        agent_id=agent["agent_id"],
        session_id=session["session_id"],
    )


async def wait_for_subscriber(
    bus: Any,
    session_id: str,
    *,
    timeout: float = READ_TIMEOUT_SECS,
) -> None:
    """等到 SSE 的订阅者真的挂到消息总线上。

    ★ 为什么需要这一步（而不是 ``await asyncio.sleep(0.3)``）：

        Starlette 的 ``StreamingResponse`` 先发 ``http.response.start``
        **再**迭代响应体。也就是说：客户端拿到响应头的那一刻，SSE 生成器
        **还没开始执行**，订阅者自然也不存在。

        此时若立刻 ``POST /chat/``，早几个事件（含 ``REPLY_START``）会在
        订阅者挂上之前就被发布出去 —— 而回放日志要等整轮结束才写，
        于是它们**彻底丢失**。用例会表现为「收到了 TEXT_BLOCK_DELTA
        但没有 REPLY_START」，且**时快时慢**：取决于进程调度。

        ``sleep`` 只是把这个概率压低，不是消除。这里改为轮询总线的订阅者
        表，等条件真正成立再往下走 —— 快、且确定性。

    ⚠️ 读了框架的私有属性 ``_subscribers``。这是**测试里**的取舍：
        总线的公开 API 只提供「订阅」与「发布」，没有「当前有几个订阅者」
        这种查询，而我们要等的正是后者。写错时的失败模式是清晰的
        ``AttributeError``（而不是静默给出错误结论），并在下面显式
        断言了属性的存在，让失败信息能自己解释原因。

    Args:
        bus (`Any`): 应用正在使用的消息总线。
        session_id (`str`): 会话 id。
        timeout (`float`): 最长等待秒数。

    Raises:
        AssertionError: 总线换了实现（不再有 ``_subscribers``），
            或超时仍无订阅者。
    """
    subscribers = getattr(bus, "_subscribers", None)
    assert subscribers is not None, (
        f"{type(bus).__name__} 没有 ``_subscribers``，本用例的等待策略需要更新。\n"
        f"总线换实现时请改这里，不要退回 sleep —— 那会让 SSE 用例偶发失败。"
    )

    key = MessageBusKeys.session_events(session_id)
    async with asyncio.timeout(timeout):
        while not subscribers.get(key):
            await asyncio.sleep(0)


async def wait_for_log_trim(
    bus: Any,
    session_id: str,
    *,
    timeout: float = READ_TIMEOUT_SECS,
) -> None:
    """等到这一轮对话的回放日志被**清空**。

    ★ 为什么需要这一步（而不是「回复一结束就重连」）：

        框架的清理**不在**回复路径上同步执行。``_service/_chat.py`` 把落库与
        ``log_trim`` 一起放进一个**后台任务**里::

            persist_task = asyncio.create_task(_persist())   # app/_service/_chat.py:1497
            ...
            await self._message_bus.log_trim(events_key)      # app/_service/_chat.py:1470

        而 ``log_trim`` 在内存总线上的实现是 ``self._logs.pop(key, None)``。

        于是「客户端收到 ``REPLY_END`` 的时刻」与「日志被清掉的时刻」之间
        有一个**真实的窗口**（本机实测约 50ms）。在这个窗口里重连，订阅者
        会**收到整轮回放**（含 ``REPLY_END``）—— 这不是「重连补发历史」的
        功能，而是清理尚未跑完。

        ⚠️ 这一点对前端也是实情：重连**可能**收到一轮完整回放，
        客户端必须按 ``event.id`` 去重，不能假设「重连必定什么都收不到」。
        本项目不打算消除这个窗口（那是改框架语义），只用本函数让用例
        **确定性地**停在窗口之后 —— 靠 sleep 猜长短会随机器负载偶发失败。

    Args:
        bus (`Any`): 应用正在使用的消息总线。
        session_id (`str`): 会话 id。
        timeout (`float`): 最长等待秒数。

    Raises:
        AssertionError: 超时后日志仍未清空。
    """
    key = MessageBusKeys.session_events(session_id)
    async with asyncio.timeout(timeout):
        while await bus.log_read(key):
            await asyncio.sleep(0.01)


class StreamReader:
    """把一条 SSE 连接拆成「事件 / 心跳 / 原始行」三份可断言的数据。

    之所以要单开一个类：``aiter_lines()`` 必须在**另一个任务**里跑，
    否则主协程会卡在读取上、没法同时去发 ``POST /chat/``。
    """

    def __init__(self, response: httpx.Response) -> None:
        """记录响应对象并初始化各收集器。

        Args:
            response (`httpx.Response`): 已建立的流式响应。
        """
        self._response = response
        self.events: list[dict[str, Any]] = []
        self.lines: list[str] = []
        self.heartbeats = 0
        self._task: asyncio.Task[None] | None = None
        self._closed = asyncio.Event()

    def start(self) -> None:
        """在后台任务里开始读取。"""
        self._task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        """持续读取，**在收到 ``REPLY_END`` 时主动停下**。

        ⚠️ 必须自己 ``return``，不能等连接关闭：
            服务端的 SSE 是**永不结束**的流 —— 一轮对话结束后它继续挂在
            那里发心跳（这正是心跳存在的意义）。指望 ``aiter_lines()``
            自然结束，结果是读取任务永远阻塞、``_closed`` 永远不置位，
            用例只能靠超时失败。而失败信息里事件是**齐全**的，
            看起来像「事件都收到了却还超时」，非常费解。

            换句话说：``REPLY_END`` 是**本轮对话**的结束，
            不是**连接**的结束 —— 这两件事在 SSE 里必须分开处理。
        """
        try:
            async for line in self._response.aiter_lines():
                self.lines.append(line)
                if line == ":":
                    # SSE 注释行 ``:\n\n`` —— 框架用它当心跳
                    # （``_router/_session.py`` 的 ``except asyncio.TimeoutError: yield ":\n\n"``）。
                    # 它**不是**事件，客户端按规范应当忽略；这里单独计数，
                    # 是为了把「连接还活着」与「有内容来了」两件事分开断言。
                    self.heartbeats += 1
                elif line.startswith("data:"):
                    event = json.loads(line[len("data:"):])
                    self.events.append(event)
                    if event.get("type") == EventType.REPLY_END.value:
                        return
        finally:
            self._closed.set()

    @property
    def types(self) -> list[str]:
        """按到达顺序返回事件类型列表。"""
        return [event.get("type") for event in self.events]

    def text_of(self, event_type: str) -> str:
        """把某类事件里的文本片段拼起来。

        Args:
            event_type (`str`): 事件类型，取值见 ``EventType``。

        Returns:
            `str`: 拼接结果。该类事件一次都没出现时返回空串。
        """
        parts: list[str] = []
        for event in self.events:
            if event.get("type") != event_type:
                continue
            parts.extend(_walk_text(event))
        return "".join(parts)

    async def wait_for_end(self, timeout: float = READ_TIMEOUT_SECS) -> None:
        """等到收到 ``REPLY_END``（或连接被关闭）。

        Args:
            timeout (`float`): 最长等待秒数。

        Raises:
            AssertionError: 超时或连接先于 ``REPLY_END`` 结束。
                ⚠️ 超时**不是**直接抛 ``TimeoutError``：那样只会得到一句
                「wait_for 超时」，完全看不出服务端到底发没发东西 ——
                而「一个字都没发」与「发了一半」是两种完全不同的故障。
                这里统一转成带事件序列与原始行的断言失败。
        """
        try:
            await asyncio.wait_for(self._closed.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise AssertionError(
                f"等了 {timeout}s 也没等到 REPLY_END。\n"
                f"已收到的事件：{self.types}\n"
                f"心跳帧数：{self.heartbeats}\n"
                f"原始行（末尾 10 条）：{self.lines[-10:]}\n"
                f"—— 若事件为空但心跳在涨，说明 SSE 通道是通的、"
                f"是这一轮对话没有产生任何事件（多半在服务端后台任务里失败了）。",
            ) from None

        assert EventType.REPLY_END.value in self.types, (
            f"SSE 连接在收到 REPLY_END 之前就结束了。已收到的事件：{self.types}\n"
            f"原始行（末尾 10 条）：{self.lines[-10:]}"
        )

    async def stop(self) -> None:
        """取消后台读取任务。"""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


#: 承载文本的字段名。**两个都要**：
#:
#:     · ``TEXT_BLOCK_DELTA`` 用 ``delta`` —— 增量只有「这一小段」；
#:     · ``TEXT_BLOCK_START`` / ``TEXT_BLOCK_END`` 用 ``text``。
#:
#: ⚠️ 一开始只找了 ``text``，于是「事件全收到、断言却说文本为空」——
#:    因为流的正是 ``delta``。这两个名字长得很像，值得在这里写死。
_TEXT_FIELDS = ("text", "delta")


def _walk_text(node: Any) -> list[str]:
    """递归挑出 JSON 节点里所有文本字段的值。

    刻意递归而不是直接取 ``event["delta"]``：事件载荷里文本可能嵌在
    ``content`` 数组之类的结构里，而具体路径随事件类型变化。
    本用例要验的是「有增量文本送到」，不该因为某个字段挪了位置就红 ——
    那种失败没有任何诊断价值。

    Args:
        node (`Any`): 任意 JSON 节点。

    Returns:
        `list[str]`: 按出现顺序排列的文本。``null`` 值会被跳过
        （``TEXT_BLOCK_END`` 的 ``text`` 就是 ``null``）。
    """
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key in _TEXT_FIELDS and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_walk_text(value))
    elif isinstance(node, list):
        for item in node:
            found.extend(_walk_text(item))
    return found


@asynccontextmanager
async def open_stream(
    client: httpx.AsyncClient,
    fixture: AgentFixture,
    headers: dict[str, str],
    *,
    expect_status: int = 200,
) -> AsyncIterator[StreamReader | httpx.Response]:
    """打开一条 SSE 连接，并按需返回读取器或裸响应。

    ⚠️ 注意传参顺序：``agent_id`` 是**必填的查询参数**，不是可选。
        少了它框架直接 422 —— 而 422 的响应体不会告诉你「这是个查询参数」，
        只会说某个 field 缺失。这一条是本项目联调时实际卡过的地方。

    Args:
        client (`httpx.AsyncClient`): HTTP 客户端。
        fixture (`AgentFixture`): 目标会话与智能体。
        headers (`dict[str, str]`): 请求头（身份）。
        expect_status (`int`): 期望的状态码。非 200 时**不**返回读取器，
            因为错误响应是普通 JSON、读成 SSE 只会得到空结果。

    Yields:
        `StreamReader | httpx.Response`: 状态码为 ``expect_status`` 时是已启动的
        读取器，否则是原始响应（供断言状态码用）。
    """
    async with client.stream(
        "GET",
        f"/sessions/{fixture.session_id}/stream",
        params={"agent_id": fixture.agent_id},
        headers=headers,
    ) as response:
        if response.status_code != expect_status:
            # 错误分支要先读完 body：流式响应在上下文退出前不读会被 httpx 判为错误。
            await response.aread()
            yield response
            return

        reader = StreamReader(response)
        reader.start()
        try:
            yield reader
        finally:
            await reader.stop()


def _chat_body(fixture: AgentFixture, text: str) -> dict[str, Any]:
    """构造 ``POST /chat/`` 的请求体。

    ⚠️ ``input`` 必须是一个 **``Msg`` 字典**，且 ``content`` 必须是
        **块组成的列表** —— 传裸字符串会得到
        ``422 model_attributes_type``。这一条同样是本项目实际踩过的：
        报错信息只说「类型不对」，看不出该包成什么形状。

    Args:
        fixture (`AgentFixture`): 目标会话与智能体。
        text (`str`): 用户这一句。

    Returns:
        `dict`: 请求体。
    """
    return {
        "agent_id": fixture.agent_id,
        "session_id": fixture.session_id,
        "input": {
            "name": "user",
            "role": "user",
            "content": [{"type": "text", "text": text}],
        },
    }


def _client(base_url: str) -> httpx.AsyncClient:
    """构造指向 live_server 的客户端。

    Args:
        base_url (`str`): 服务器地址。

    Returns:
        `httpx.AsyncClient`: 调用方负责 ``async with`` 关闭。
    """
    return httpx.AsyncClient(base_url=base_url, timeout=READ_TIMEOUT_SECS)


# ==============================================================================
# 一、P2 验收主场景：一句话端到端流式返回
# ==============================================================================
async def test_single_sentence_streams_end_to_end(live_server: LiveServer) -> None:
    """★ P2 验收线的主场景。

    顺序是刻意固定的 —— **先连 SSE，再发消息**（理由见模块文档字符串）：

        1. 建凭据 / 智能体 / 会话；
        2. 打开 SSE 并等到订阅者真的挂上总线；
        3. ``POST /chat/``（发后不理，立刻返回 ``started``）；
        4. 在 SSE 上等到 ``REPLY_END``；
        5. 断言这一路收到了 ``REPLY_START`` 与逐字增量。
    """
    async with _client(live_server.base_url) as client:
        fixture = await _bootstrap(client, ALICE_HEADERS)

        async with open_stream(client, fixture, ALICE_HEADERS) as stream:
            assert isinstance(stream, StreamReader)
            await wait_for_subscriber(live_server.bus, fixture.session_id)

            trigger = _json(
                await client.post(
                    "/chat/",
                    headers=ALICE_HEADERS,
                    json=_chat_body(fixture, "帮我订明天去杭州的高铁票。"),
                ),
            )
            assert trigger["status"] == "started"

            await stream.wait_for_end()

        types = stream.types

    # ① 应答开始 —— 它是「服务端真的接下了这一轮」的第一个信号。
    assert EventType.REPLY_START.value in types, (
        f"没有收到 REPLY_START。实际事件序列：{types}\n"
        f"若序列里连 TEXT_BLOCK_DELTA 都没有，多半是 SSE 连晚了"
        f"（回放日志已被 log_trim 丢弃，见模块文档字符串）。"
    )
    # ② 逐字增量 —— 流式的**全部意义**就在这里：不是等一句话生成完再返回，
    #    而是边生成边推。少了它，接口依然「能用」，但前端只能看到
    #    「转圈 → 整段出现」，与阻塞式请求毫无区别。
    deltas = [t for t in types if t == EventType.TEXT_BLOCK_DELTA.value]
    assert deltas, f"没有收到任何 TEXT_BLOCK_DELTA。实际事件序列：{types}"

    # ③ 增量拼起来应当是非空的正常文本。
    text = stream.text_of(EventType.TEXT_BLOCK_DELTA.value)
    assert text.strip(), f"TEXT_BLOCK_DELTA 的文本拼起来是空的。事件：{stream.events}"

    # ④ 正常收尾。finished_reason 不是 completed 就说明这一轮实际是**异常结束**
    #    （最常见的是 setup 失败），而异常结束同样会发 REPLY_END ——
    #    只断言「收到了 REPLY_END」会把一个失败当成功。
    end = next(e for e in stream.events if e.get("type") == EventType.REPLY_END.value)
    assert end.get("finished_reason") == "completed", (
        f"这一轮不是正常结束：{end}"
    )


# ==============================================================================
# 二、心跳与「连晚了」的陷阱
# ==============================================================================
async def test_heartbeat_frame_keeps_idle_stream_alive(
    live_server: LiveServer,
    monkeypatch: Any,
) -> None:
    """长时间没有事件时，服务端持续发 ``:\\n\\n`` 心跳。

    ★ 心跳不是装饰，它有两个硬作用：

        1. **穿透中间设备**：Nginx / 云 LB 普遍对「空闲超过 N 秒的连接」超时断开。
           SSE 一轮对话之间可能几十秒没有事件，没有心跳就会被静默掐断，
           客户端表现为「随机地少收最后一段」。
        2. **让客户端能区分「还活着但没内容」与「已经死了」**：
           连接一旦断了，前端的自动重连才有依据。

    这里把心跳间隔 monkeypatch 成 0.25s，避免用例真的等 30 秒。
    """
    # 见 HEARTBEAT_INTERVAL_GLOBAL 的注释：这个常量是在生成器循环体里
    # 求值的，所以改模块属性真的生效。
    monkeypatch.setattr(HEARTBEAT_INTERVAL_GLOBAL, FAST_HEARTBEAT_SECS)

    async with _client(live_server.base_url) as client:
        fixture = await _bootstrap(client, ALICE_HEADERS)

        async with open_stream(client, fixture, ALICE_HEADERS) as stream:
            assert isinstance(stream, StreamReader)
            await wait_for_subscriber(live_server.bus, fixture.session_id)

            # 一个字都不发，纯等心跳。等到至少两帧再断言 —— 只等一帧的话，
            # 「第一帧是别的东西凑巧被当成心跳」这种误判也能通过。
            async with asyncio.timeout(READ_TIMEOUT_SECS):
                while stream.heartbeats < 2:
                    await asyncio.sleep(0.01)

            assert stream.heartbeats >= 2
            # 心跳期间不该有任何业务事件混进来 —— 会话根本没被触发。
            assert stream.events == [], f"空闲连接上收到了事件：{stream.events}"


async def test_reconnect_after_reply_gets_heartbeat_but_no_replay(
    live_server: LiveServer,
    monkeypatch: Any,
) -> None:
    """★ 回归测试：一轮对话结束、**清理完成之后**再连 SSE，收不到回放。

    钉的是 ``_persist()`` 里那句 ``log_trim``（``self._logs.pop(key, None)``）——
    它把这场对话的回放日志整个丢掉。这不是 bug，是框架有意的「一轮一清」；
    但它带来一个必须写进前端契约的结论：**流式接口不可用于补看历史**，
    要看历史请走 ``GET /sessions/{id}/messages``。

    ⚠️ 这条用例的价值在于**防止有人把它当 bug 修**：
        如果哪天有人给 SSE 加上「重连自动补发历史」，这条用例会红，
        而在动手改之前先读到这段注释，就会知道那是在改框架语义、
        而不是在修缺陷。

    ⚠️⚠️ **「一轮一清」是异步的，中间有一个真实窗口。** 落库与 ``log_trim``
        都在 ``asyncio.create_task(_persist())`` 里跑（``_service/_chat.py:1497``），
        所以「客户端收到 REPLY_END」与「日志被清掉」之间约 50ms 内重连，
        会收到**整轮回放**（本机实测，见 :func:`wait_for_log_trim`）。
        那是清理没跑完，不是补发历史的功能 —— 但它对前端是可见的，
        客户端必须按 ``event.id`` 去重。
        本用例因此先 ``await wait_for_log_trim(...)`` 再重连：
        测的是「清理之后」的语义（这才是它要钉的那条），
        而不是去赌那 50ms 的调度。
    """
    monkeypatch.setattr(HEARTBEAT_INTERVAL_GLOBAL, FAST_HEARTBEAT_SECS)

    async with _client(live_server.base_url) as client:
        fixture = await _bootstrap(client, ALICE_HEADERS)

        # 第一轮：正常跑完。
        async with open_stream(client, fixture, ALICE_HEADERS) as stream:
            assert isinstance(stream, StreamReader)
            await wait_for_subscriber(live_server.bus, fixture.session_id)
            await client.post(
                "/chat/",
                headers=ALICE_HEADERS,
                json=_chat_body(fixture, "你好。"),
            )
            await stream.wait_for_end()
            assert stream.events, "第一轮就没收到事件，后面的断言没有意义。"

        # 等清理真正跑完 —— 不等的话测的就是「清理窗口内重连」，
        # 而那件事的结论是另一个（见 wait_for_log_trim 的文档）。
        await wait_for_log_trim(live_server.bus, fixture.session_id)

        # 第二轮：对话已结束，重新连上来。
        async with open_stream(client, fixture, ALICE_HEADERS) as fresh:
            assert isinstance(fresh, StreamReader)
            await wait_for_subscriber(live_server.bus, fixture.session_id)

            # 等到攒下两帧心跳，确认这段时间里连接一直是活的
            # （不是「断了所以没事件」这种假阴性）。
            async with asyncio.timeout(READ_TIMEOUT_SECS):
                while fresh.heartbeats < 2:
                    await asyncio.sleep(0.01)

            assert fresh.events == [], (
                f"重连后收到了历史回放：{fresh.events}\n"
                f"若这是有意加上的功能，请先读本用例的文档字符串 ——"
                f"它记录的是框架「一轮一清」的既定语义，不是缺陷。"
            )


# ==============================================================================
# 三、多租户隔离
# ==============================================================================
async def test_sessions_are_isolated_between_users(live_server: LiveServer) -> None:
    """★ 两个不同 ``X-User-ID`` 互相看不到对方会话。

    断言范围限于**会回内容的**端点（见模块文档字符串）。

    ⚠️ 必须同时断言「本人能读到」：只断言「他人 404」的话，
        一个把所有人都 404 掉的实现也能通过 —— 那种实现当然「隔离」，
        但它同时也什么都做不了。
    """
    async with _client(live_server.base_url) as client:
        alice = await _bootstrap(client, ALICE_HEADERS)
        bob = await _bootstrap(client, BOB_HEADERS)

        # ---- 他人身份：三个端点都必须是 404 ------------------------------------
        for path, params in (
            (f"/sessions/{alice.session_id}/messages", {"agent_id": alice.agent_id}),
            (f"/sessions/{alice.session_id}/stream", {"agent_id": alice.agent_id}),
            (f"/sessions/{alice.session_id}/status", {"agent_id": alice.agent_id}),
        ):
            response = await client.get(path, params=params, headers=BOB_HEADERS)
            assert response.status_code == 404, (
                f"bob 访问 alice 的 {path} 得到 {response.status_code}，期望 404。\n"
                f"非 404 意味着跨租户可读 —— 这是 P2 验收线里明确要求的一条。\n"
                f"响应体：{response.text[:200]}"
            )

        # ---- 他人身份：连 agent 都列不出来 ------------------------------------
        listing = await client.get(
            "/sessions/",
            params={"agent_id": alice.agent_id},
            headers=BOB_HEADERS,
        )
        assert listing.status_code == 404, (
            f"bob 能列出 alice 的 agent 下的会话（{listing.status_code}）。"
        )

        # ---- 本人身份：同样的请求必须成功（否则上一条断言毫无鉴别力）--------
        own = await client.get(
            f"/sessions/{alice.session_id}/messages",
            params={"agent_id": alice.agent_id},
            headers=ALICE_HEADERS,
        )
        assert own.status_code == 200, (
            f"alice 读自己的会话却得到 {own.status_code} —— "
            f"上下两条断言就变成「所有人都读不到」，那样测不出隔离。"
        )

        # ---- 隔离不是「bob 什么都做不了」：他自己的一套必须完全可用 -----------
        bob_own = await client.get(
            f"/sessions/{bob.session_id}/messages",
            params={"agent_id": bob.agent_id},
            headers=BOB_HEADERS,
        )
        assert bob_own.status_code == 200, (
            f"bob 读自己的会话得到 {bob_own.status_code} —— "
            f"要么是他的会话没建起来，要么是隔离写成了「一律拒绝」。"
        )


async def test_stream_without_identity_is_401(live_server: LiveServer) -> None:
    """不带任何身份连 SSE → 401，且由**我们的**中间件拦下。

    断言 401 而不是 404：404 说明请求穿过了鉴权层、被框架当成「找不到会话」
    处理掉了 —— 那意味着鉴权中间件没有覆盖到流式端点。
    """
    async with _client(live_server.base_url) as client:
        alice = await _bootstrap(client, ALICE_HEADERS)

        response = await client.get(
            f"/sessions/{alice.session_id}/stream",
            params={"agent_id": alice.agent_id},
        )

        assert response.status_code == 401, (
            f"期望 401（鉴权中间件），实际 {response.status_code}。"
        )
        assert "www-authenticate" in {k.lower() for k in response.headers}


async def test_chat_trigger_status_is_not_an_authorization_signal(
    live_server: LiveServer,
) -> None:
    """``POST /chat/`` 对会话属主**不做**同步校验 —— 记下这个事实。

    ⚠️ 这条用例断言的是一种「看起来像漏洞」的既定行为，说明如下：

        ``POST /chat/`` 是发后不理的：它校验完请求体的形状就返回
        ``{"status": "started"}``，真正的派发在后台任务里。因此：

            · 别人的会话 id → 依然返回 ``started``；
            · **根本不存在的**会话 id → 也返回 ``started``。

        后者正是「它不是授权信号」的证明：连不存在的资源都返回成功，
        说明这个状态码里**不含任何**属主信息。

        真正的隔离发生在**读**这一侧（``/messages`` / ``/stream`` 全是 404，
        见 :func:`test_sessions_are_isolated_between_users`）。

    ⚠️ 因此本用例**不**断言某个具体的越权行为，只断言那个不变量：
        用别人的会话 id 触发之后，对方的内容依然读不到。
        这样即使哪天框架加了同步校验（返回 404），这条用例也不会误报 ——
        它保护的是结论，不是实现。
    """
    async with _client(live_server.base_url) as client:
        alice = await _bootstrap(client, ALICE_HEADERS)

        # 一个**不存在**的会话 id：能 200 就说明状态码里没有属主信息。
        nonexistent = await client.post(
            "/chat/",
            headers=BOB_HEADERS,
            json={
                "agent_id": alice.agent_id,
                "session_id": "0000000000000000000000000000dead",
                "input": {
                    "name": "user",
                    "role": "user",
                    "content": [{"type": "text", "text": "越权尝试。"}],
                },
            },
        )
        started_means_nothing = nonexistent.status_code == 200

        # 不变量：无论上面那一步返回什么，alice 的会话内容对外人依然不可读。
        leaked = await client.get(
            f"/sessions/{alice.session_id}/messages",
            params={"agent_id": alice.agent_id},
            headers=BOB_HEADERS,
        )
        assert leaked.status_code == 404, (
            f"bob 读到了 alice 的会话内容（{leaked.status_code}）—— 真的漏了。"
        )

        # 顺带把「/chat/ 不是授权信号」这件事显式记录下来：
        # 若哪天它变成 4xx，说明框架加了同步校验，本用例的注释需要更新
        # （但上面的不变量断言依然成立，不需要改）。
        assert started_means_nothing or nonexistent.status_code in (403, 404), (
            f"POST /chat/ 对一个不存在的会话返回了意外的状态码："
            f"{nonexistent.status_code} {nonexistent.text[:200]}"
        )


# ==============================================================================
# 六、开箱即用：不手工建凭据也能对话（零密钥部署的**唯一**验收线）
# ==============================================================================
async def test_conversation_works_with_zero_config(live_server: LiveServer) -> None:
    """★ 本用例挡住的是一整类「探针全绿但一个字都发不出去」的部署。

    真实事故形态（本用例诞生前，本仓库的实际状态）：
        ``make up`` 之后 13 个容器全 healthy、``/healthz`` ``/readyz`` 全 200、
        SPA 正常渲染、``make smoke`` 通过 —— 但浏览器里「发送」按钮是**灰的**，
        而即使用 curl 绕过前端直接 ``POST /chat/``，得到的也是一个
        ``finished_reason="error"`` 的空回复。

    原因是框架的对话链路要两样东西同时在位：一条属于调用者的凭据记录，
    以及会话的 ``chat_model_config`` 指向它。零密钥部署两样都没有 ——
    而健康检查只看进程与存储，看不见「有没有模型可用」。

    本用例刻意**不**调用 ``POST /credential/``（其它用例都调）：
    它要证明的正是「用户什么都不用配，也能发出第一条消息」。
    """
    async with _client(live_server.base_url) as client:
        # ① 问服务端「我现在该用哪个模型」——零密钥下必须给出一条可用的配置。
        resolved = _json(
            await client.get("/api/v1/default-model", headers=CAROL_HEADERS),
        )
        assert resolved["mode"] == "mock", (
            f"零密钥部署下没有给出降级模型：{resolved}。"
            f"前端此刻的发送按钮是灰的（可用模型列表为空）。"
        )
        config = resolved["chat_model_config"]
        assert config, f"mode=mock 却没有给出配置：{resolved}"

        # ② 这条配置必须能被「可用模型」的两个来源看见 ——
        #    前端把它们拼成选择器的分组，任一侧为空都会让按钮变灰。
        credentials = _json(await client.get("/credential/", headers=CAROL_HEADERS))
        assert any(
            (item.get("data") or {}).get("type") == config["type"]
            and item.get("id") == config["credential_id"]
            for item in credentials.get("credentials", [])
        ), f"降级凭据没有出现在凭据列表里：{credentials}"
        models = _json(
            await client.get(
                "/model/",
                params={"provider": config["type"]},
                headers=CAROL_HEADERS,
            ),
        )
        assert any(
            card.get("name") == config["model"] for card in models.get("models", [])
        ), f"降级模型没有出现在 {config['type']} 的模型列表里：{models}"

        # ③ 用这份配置建智能体与会话（**不建凭据**），然后走完整的一轮对话。
        agent = _json(
            await client.post(
                "/agent/",
                headers=CAROL_HEADERS,
                json={"name": "开箱即用助手", "system_prompt": "你是差旅助手。"},
            ),
            expect=201,
        )
        session = _json(
            await client.post(
                "/sessions/",
                headers=CAROL_HEADERS,
                json={
                    "agent_id": agent["agent_id"],
                    "name": "开箱即用会话",
                    "chat_model_config": config,
                },
            ),
            expect=201,
        )
        fixture = AgentFixture(
            credential_id=config["credential_id"],
            agent_id=agent["agent_id"],
            session_id=session["session_id"],
        )

        async with open_stream(client, fixture, CAROL_HEADERS) as stream:
            assert isinstance(stream, StreamReader)
            await wait_for_subscriber(live_server.bus, fixture.session_id)

            trigger = _json(
                await client.post(
                    "/chat/",
                    headers=CAROL_HEADERS,
                    json=_chat_body(fixture, "帮我规划下周去上海出差。"),
                ),
            )
            assert trigger["status"] == "started"
            await stream.wait_for_end()

        types = stream.types

    # ④ 与主场景一样的四个断言：这一轮必须**真的**产出了文本。
    #    只断言「收到了 REPLY_END」是不够的：异常结束同样会发 REPLY_END，
    #    而本用例挡的正是「异常结束」这一种。
    assert EventType.REPLY_START.value in types, f"没有 REPLY_START：{types}"
    text = stream.text_of(EventType.TEXT_BLOCK_DELTA.value)
    assert text.strip(), (
        f"零配置对话没有产出任何文本。事件序列：{types}\n"
        "    最常见的原因：会话的 chat_model_config 没有被框架接受"
        "（_chat.py 找不到模型配置时会直接以 error 结束）。"
    )
    end = next(e for e in stream.events if e.get("type") == EventType.REPLY_END.value)
    assert end.get("finished_reason") == "completed", f"这一轮不是正常结束：{end}"


# ==============================================================================
# 五、提示块不得到达用户（2026-10-03 审计发现的泄漏）
# ==============================================================================
async def test_hint_blocks_never_reach_the_user(live_server: LiveServer) -> None:
    """★★ 框架注入的提示块**不得**出现在 SSE 流与历史消息里。

    这是一条**回归**用例，守的是一个已经实测发生过、且**刷新页面也躲不掉**
    的泄漏。事实链（每一环都核实过，详见
    :mod:`src.orchestration.hint_filter` 的模块文档）：

        1. 框架把运行时状态包成 ``HintBlock`` 追加进上下文，并 yield 一个
           ``HintBlockEvent``（``agent/_agent.py:1629-1639``）；
        2. ``ChatService`` 把每个事件**无条件**转发到消息总线
           （``app/_service/_chat.py:1250-1269``），SSE 再原样推给浏览器；
        3. ``Msg.append_event`` 对 ``HINT_BLOCK`` 的处理是把它**追加进消息
           content**（``message/_base.py:372-382``，注释写明 "for persistence
           and replay"），于是 ``upsert_message`` 之后
           ``GET /sessions/{id}/messages`` 每次都把它带回来。

    用户看到的是一段**英文提示词**——默认模板的第一句是
    ``"Treat the following as the ground truth at this point of the
    conversation. ..."``（``agent/_config.py:285-292``），外面还裹着
    ``<system-reminder>`` 标签。对这些块，用户的身份是**局外人**：
    它们的读者是模型和别的智能体。

    ⚠️ 两条链路都要断言，缺一不可。只断 SSE 的话，「刷新之后还能看到」
    这个更持久的形态会漏掉 —— 而它恰恰是实测中最容易被当成「偶发」
    而放过去的那一种。

    ⚠️ 断言的是**事件类型**而不是关键字。搜 ``<system-reminder>`` 只能
    证明「这一次恰好没出现」，而漏掉的是「这类事件本就不该下发」这条规则 ——
    换个模板、换个语言，关键字断言就瞎了。
    """
    async with _client(live_server.base_url) as client:
        fixture = await _bootstrap(client, ALICE_HEADERS)

        async with open_stream(client, fixture, ALICE_HEADERS) as stream:
            assert isinstance(stream, StreamReader)
            await wait_for_subscriber(live_server.bus, fixture.session_id)

            trigger = _json(
                await client.post(
                    "/chat/",
                    headers=ALICE_HEADERS,
                    json=_chat_body(fixture, "帮我查一下北京的酒店差标。"),
                ),
            )
            assert trigger["status"] == "started"
            await stream.wait_for_end()

        types = stream.types

        # ① 实时链路：SSE 帧里不得有 HINT_BLOCK。
        hints = [
            e for e in stream.events if e.get("type") == EventType.HINT_BLOCK.value
        ]
        assert not hints, (
            f"SSE 流里出现了 {len(hints)} 个提示块 —— "
            f"它们是给模型的，不该给用户看。第一条：{hints[0]}\n"
            "    若这条用例变红，先看 HintSuppressionMiddleware 还在不在 "
            "build_middlewares_factory 的列表里。"
        )

        # ② 历史链路：落库的消息里也不得有。
        #    ⚠️ 这条必须**在 SSE 结束之后**查：持久化发生在 _persist()，
        #    它跑在 REPLY_END 之后；过早查询会读到一个还没有消息的空列表，
        #    用例于是因为「没查到」而变绿 —— 一个静默的假绿。
        messages = _json(
            await client.get(
                f"/sessions/{fixture.session_id}/messages",
                headers=ALICE_HEADERS,
                params={"agent_id": fixture.agent_id},
            ),
        )

    # ⚠️ 先钉住「真的读到了消息」：消息列表为空时，下面的遍历会一条不查地
    # 通过 —— 用例从守卫退化成摆设。这一步与 event/P2 的其它用例同源。
    # ⚠️ 响应体的键是 ``messages`` 而不是 ``data``（与 ``/sessions/`` 列表
    # 接口的 ``{"data": ...}`` 包装**不一样**），写错会得到
    # 「历史消息是空的」这条误导性的失败信息。
    assert messages.get("messages"), (
        f"历史消息是空的，历史链路无从校验（这本身就是缺陷）：{messages}"
    )
    persisted = json.dumps(messages, ensure_ascii=False)

    # ⚠️ 判据是**落库后的形状**，不是事件类型名。这一点是实测校正的：
    # ``Msg.append_event`` 把 HintBlockEvent 转成 content 里的
    # ``{"type": "hint", "hint": ...}``（``message/_base.py:372-382``），
    # 于是 ``"HINT_BLOCK"`` 这个字符串在响应里**根本不出现** ——
    # 照那个去断言是一条永远不会红的摆设。
    # ⚠️ 子串先落到变量里再进 f-string：Python 3.11 的 f-string 表达式部分
    # **不允许出现反斜杠**，而这里要搜的子串本身带引号。
    hint_block_marker = '"type": "hint"'
    assert hint_block_marker not in persisted, (
        "历史消息里出现了提示块（content 里的 \"type\": \"hint\"）—— "
        "它被写进了 reply Msg 并落库，用户刷新页面照样看得到。\n"
        f"    命中位置：{_excerpt_around(persisted, hint_block_marker)}"
    )
    # ⚠️ 兜底关键字：上面那条依赖框架把块标成 ``hint``；这段英文提示词是
    # **框架默认模板原文**（``agent/_config.py:285-292``），它在响应里出现
    # 只可能来自提示块 —— 换了字段名它照样抓得住。
    assert "<system-reminder>" not in persisted, (
        "历史消息里出现了 <system-reminder> 提示词原文 —— 用户能看到英文提示词。\n"
        f"    命中位置：{_excerpt_around(persisted, '<system-reminder>')}"
    )


def _excerpt_around(text: str, needle: str, *, span: int = 120) -> str:
    """截取命中位置附近的片段，便于排查。

    Args:
        text (`str`): 被搜索的文本。
        needle (`str`): 要找的子串。
        span (`int`): 命中点前后各取多少个字符。

    Returns:
        `str`: 形如 ``...前文【命中】后文...`` 的片段。

    ⚠️ 失败信息里**不能直接打印整份 messages**：它可能很长（几十条消息），
    而 pytest 的报错会被截断，真正有用的那一小段反而被淹掉。
    """
    index = text.find(needle)
    if index < 0:  # pragma: no cover —— 只在「断言没命中却调用了本函数」时发生
        return text[: span * 2]
    start = max(0, index - span)
    end = min(len(text), index + len(needle) + span)
    return f"...{text[start:index]}【{needle}】{text[index + len(needle):end]}..."
