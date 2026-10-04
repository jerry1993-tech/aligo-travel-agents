# -*- coding: utf-8 -*-
"""思考链 SSE 端点（``src/server/routers/_chains.py``）的端到端测试。

==============================================================================
为什么必须起真实的 uvicorn（而不是 httpx.ASGITransport）
==============================================================================
    ``ASGITransport`` 会**把整个响应体缓冲完**再交还给调用方。对普通 JSON
    接口没问题，对 SSE 是致命的：本端点的响应**永不结束**（一轮回复结束后
    它继续挂在线上发心跳），缓冲意味着 ``aiter_lines()`` 一个字节都拿不到，
    用例会一直挂到超时。症状不是报错，而是**什么都没有**。

    对策照抄 ``tests/test_e2e_stream.py``：在**同一个事件循环**里起一个真的
    ``uvicorn.Server``，绑 ``127.0.0.1:0``（端口交给内核分配，避免并行跑
    测试时撞端口），用真的 TCP 客户端连它。

    ⚠️ 必须是**同一个事件循环**：``InMemoryMessageBus`` 的订阅者是
    ``asyncio.Queue``，绑定在创建它的循环上。uvicorn 换到别的线程/循环里跑，
    订阅与发布就落在两个循环上，事件永远送不到 —— 表现为「连接建得起来、
    心跳也有，就是没有内容」。

==============================================================================
这里的用例覆盖什么
==============================================================================
    1. :func:`test_chains_stream_pushes_task_states` —— 投影本身：
       往会话的事件通道上发一串 ``AgentEvent`` 字典（工具调用全流程），
       断言端点把它们投影成任务清单推出来，且**只在变化时才发**。
    2. :func:`test_chains_stream_streams_reasoning_from_a_live_reply` ——
       真实的 ``POST /chat/`` 驱动：带 ``#mock-think:`` 指令让 MockLLM 产出
       思考块，断言端点把推理文本推出来（``expose_reasoning=true``）。
    3. :func:`test_chains_stream_is_404_for_non_owner` —— 属主校验。
    4. :func:`test_chains_stream_is_scoped_to_the_sessions_own_channel` ——
       跨租户隔离。
    5. :func:`test_chains_stream_sends_heartbeat_when_idle` —— 没有事件时的心跳。
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

import httpx
import pytest
import pytest_asyncio
import uvicorn
from agentscope.app.message_bus import InMemoryMessageBus, MessageBusKeys
from fastapi import FastAPI

from src.server.routers._chains import _read_replay_tail, session_chains

# ==============================================================================
# 常量
# ==============================================================================

#: 两个互相看不见对方的身份。取值本身无意义，只要**不同**即可。
ALICE_HEADERS = {"X-User-ID": "alice"}
BOB_HEADERS = {"X-User-ID": "bob"}

#: 假凭据类型。由 ``src/server/app.py`` 通过 ``extra_credentials`` 注册，
#: 对应 ``src/llm/mock.py::MockCredential`` —— 保证本文件**零密钥**也能跑。
MOCK_CREDENTIAL_TYPE = "aligo_mock_credential"

#: MockLLM 的模型名。
MOCK_MODEL_NAME = "mock-model"

#: 让 MockLLM 产出思考块的指令（见 ``src/llm/mock.py`` 的模块文档）。
#: 用它而不是 ``#mock-tool:`` 是刻意的：工具指令会让模型**每一轮**都再发出
#: 同一个调用（历史里那条 user 消息带着指令），一直循环到 ``max_iters``；
#: 而思考块只出现在首轮，一轮就结束。
THINK_TEXT = "我在核对差旅标准"
CHAT_WITH_THINK = f"#mock-think: {THINK_TEXT}"

#: 本端点心跳间隔所在的模块全局。
#:
#: ⚠️ 与 ``test_e2e_stream.py`` 同样的取舍：心跳的 ``timeout=`` 是在 SSE
#: 生成器的**循环体里**每次求值的，所以 monkeypatch 这个模块属性就能改掉它
#: —— 不必等 30 秒。若哪天有人把它写成函数默认参数，这条用例会立刻失败。
CHAIN_HEARTBEAT_GLOBAL = "src.server.routers._chains._HEARTBEAT_INTERVAL_SECS"

#: 心跳用例里把间隔压到的值。
FAST_HEARTBEAT_SECS = 0.25

#: 「等待订阅建立」的超时所在的模块全局（同样在生成器循环体里求值）。
CHAIN_SUBSCRIBE_TIMEOUT_GLOBAL = "src.server.routers._chains._SUBSCRIBE_TIMEOUT_SECS"

#: 订阅失败用例里把超时压到的值。
FAST_SUBSCRIBE_TIMEOUT_SECS = 0.2

#: 默认读超时。SSE 是长连接，读超时只用来兜住「卡死」。
READ_TIMEOUT_SECS = 30.0

#: 一个真实工具名（在 ``src/chains/events.py`` 的 ``TITLES`` 里有中文标题）。
TOOL_NAME = "search_hotels"


# ==============================================================================
# 夹具：真实 uvicorn 服务器（照抄 test_e2e_stream.py）
# ==============================================================================
@dataclass
class LiveServer:
    """一个跑在真实 TCP 上的测试服务器。

    Attributes:
        base_url (`str`): 形如 ``http://127.0.0.1:53412``，已含真实端口。
        bus (`Any`): 该应用正在使用的消息总线。用例据此**确定性地**等待
            SSE 订阅者挂上（见 :func:`wait_for_subscriber`）。
    """

    base_url: str
    bus: Any


@pytest_asyncio.fixture
async def live_server(app: FastAPI) -> AsyncIterator[LiveServer]:
    """在与用例**同一个事件循环**里起一个真实的 uvicorn。

    ⚠️ 这里**不**用 ``client`` 夹具：它在 lifespan 里跑 ASGITransport，
        而 uvicorn 会自己进入 lifespan。两者同时跑等于把同一个应用的
        启动流程执行两遍。本文件的用例只声明 ``live_server``。

    Args:
        app (`FastAPI`): conftest 装配好的测试应用（**尚未**进入 lifespan）。

    Yields:
        `LiveServer`: 指向真实端口、可被 httpx 直连的服务器句柄。
    """
    # port=0：让内核分配空闲端口，避免并行跑测试时撞端口。
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="error",
        lifespan="on",  # 必须显式开：关掉的话 app.state.chat_service 之类全不存在
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())

    # 轮询 started 而不是 sleep 固定值：启动耗时随机器负载浮动。
    try:
        async with asyncio.timeout(READ_TIMEOUT_SECS):
            while not server.started:
                if serve_task.done():
                    # serve() 提前返回 = 启动就失败了（多半是 lifespan 抛异常）。
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

    Args:
        client (`httpx.AsyncClient`): 指向 live_server 的客户端。
        headers (`dict[str, str]`): 该身份要带的请求头（``X-User-ID``）。

    Returns:
        `AgentFixture`: 三个对象的 id。
    """
    credential = _json(
        await client.post(
            "/credential/",
            headers=headers,
            json={"data": {"type": MOCK_CREDENTIAL_TYPE, "name": "思考链测试 Mock 凭据"}},
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
                "name": "思考链测试会话",
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
    """等到本端点的订阅者真的挂到消息总线上。

    ★ 为什么需要这一步（而不是 ``await asyncio.sleep(0.3)``）：

        Starlette 的 ``StreamingResponse`` 先发 ``http.response.start``
        **再**迭代响应体。客户端拿到响应头的那一刻，SSE 生成器**还没开始
        执行**，订阅者自然也不存在。此时若立刻发事件，早几个会在订阅者挂上
        之前就丢 —— 而且**时快时慢**，取决于进程调度。

        ``sleep`` 只是把这个概率压低，不是消除。这里改为轮询总线的订阅者
        表，等条件真正成立再往下走 —— 快、且确定性。

    ⚠️ 读了框架的私有属性 ``_subscribers``。这是**测试里**的取舍：总线没有
        「当前有几个订阅者」这种公开查询，而我们要等的正是后者。写错时的
        失败模式是清晰的 ``AttributeError``，下面显式断言了属性的存在。

    Args:
        bus (`Any`): 应用正在使用的消息总线。
        session_id (`str`): 会话 id。
        timeout (`float`): 最长等待秒数。

    Raises:
        AssertionError: 总线换了实现（不再有 ``_subscribers``），或超时无订阅者。
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


class ChainStream:
    """把一条思考链 SSE 连接拆成「快照帧 / 心跳 / 原始行」三份可断言的数据。

    ⚠️ 与 ``test_e2e_stream.py`` 的 ``StreamReader`` 有一处**关键差别**：
        那边在收到 ``REPLY_END`` 时主动停下，而本端点**从不发** ``REPLY_END``
        （它推的是任务清单快照，不是原始事件）。所以这里不设自动停止条件，
        由用例用 :meth:`wait_until` 断言到需要的帧之后主动收尾。

    ``aiter_lines()`` 必须在**另一个任务**里跑，否则主协程会卡在读取上、
    没法同时去发事件或 ``POST /chat/``。
    """

    def __init__(self, response: httpx.Response) -> None:
        """记录响应对象并初始化各收集器。

        Args:
            response (`httpx.Response`): 已建立的流式响应。
        """
        self._response = response
        self.frames: list[dict[str, Any]] = []
        self.lines: list[str] = []
        self.heartbeats = 0
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """在后台任务里开始读取。"""
        self._task = asyncio.create_task(self._pump())

    async def _pump(self) -> None:
        """持续读取直到连接关闭或任务被取消。"""
        async for line in self._response.aiter_lines():
            self.lines.append(line)
            if line == ":":
                # SSE 注释行 ``:\n\n`` —— 本端点用它当心跳。它不是事件，
                # 单独计数是为了把「连接还活着」与「有内容来了」分开断言。
                self.heartbeats += 1
            elif line.startswith("data:"):
                self.frames.append(json.loads(line[len("data:"):]))

    async def wait_until(
        self,
        predicate: Callable[["ChainStream"], bool],
        *,
        timeout: float = READ_TIMEOUT_SECS,
    ) -> None:
        """等到 ``predicate`` 成立。

        ⚠️ 超时**不**直接抛 ``TimeoutError``：那样只会得到一句「wait_for
        超时」，看不出服务端到底发没发东西 —— 而「一个字都没发」与
        「发了一半」是两种完全不同的故障。这里转成带帧序列与原始行的断言失败。

        Args:
            predicate (`Callable[[ChainStream], bool]`): 判定条件。
            timeout (`float`): 最长等待秒数。

        Raises:
            AssertionError: 超时仍未满足条件。
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        while not predicate(self):
            if loop.time() >= deadline:
                raise AssertionError(
                    f"等了 {timeout}s 条件仍未满足。\n"
                    f"已收到的帧：{self.frames}\n"
                    f"心跳帧数：{self.heartbeats}\n"
                    f"原始行（末尾 10 条）：{self.lines[-10:]}",
                )
            await asyncio.sleep(0.01)

    async def stop(self) -> None:
        """取消后台读取任务。"""
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


@asynccontextmanager
async def open_chains(
    client: httpx.AsyncClient,
    fixture: AgentFixture,
    headers: dict[str, str],
    *,
    expose_reasoning: bool = False,
) -> AsyncIterator[ChainStream | httpx.Response]:
    """打开一条到本端点的 SSE 连接。

    ⚠️ 注意传参顺序：``agent_id`` 是**必填的查询参数**，不是可选。
        少了它框架直接 422 —— 而 422 的响应体不会告诉你「这是个查询参数」。

    ⚠️ 判据是「**是不是 200**」而不是「是不是期望的那个码」：只有 200 的
        响应体才是 SSE。写错的话（比如拿 ``expect_status=404`` 去比），
        一个真实的 404 会被当成「成功」而交给读取器 —— 读取器读普通 JSON
        得到零帧，用例在别处以「超时」的形式失败，完全指不到真正的原因。

    Args:
        client (`httpx.AsyncClient`): HTTP 客户端。
        fixture (`AgentFixture`): 目标会话与智能体。
        headers (`dict[str, str]`): 请求头（身份）。
        expose_reasoning (`bool`): 是否请求推理文本。

    Yields:
        `ChainStream | httpx.Response`: 200 时是已启动的读取器，
        其它状态码时是原始响应（供调用方断言状态码用）。
    """
    params: dict[str, Any] = {"agent_id": fixture.agent_id}
    if expose_reasoning:
        params["expose_reasoning"] = "true"
    async with client.stream(
        "GET",
        f"/api/v1/sessions/{fixture.session_id}/chains",
        params=params,
        headers=headers,
    ) as response:
        if response.status_code != 200:
            # 非 200 分支要先读完 body：流式响应在上下文退出前不读会被 httpx 判为错误。
            await response.aread()
            yield response
            return

        reader = ChainStream(response)
        reader.start()
        try:
            yield reader
        finally:
            await reader.stop()


def _chat_body(fixture: AgentFixture, text: str) -> dict[str, Any]:
    """构造 ``POST /chat/`` 的请求体。

    ⚠️ ``input`` 必须是一个 **``Msg`` 字典**，且 ``content`` 必须是
        **块组成的列表** —— 传裸字符串会得到 ``422 model_attributes_type``。

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


def _tool_call_events() -> list[dict[str, Any]]:
    """造一串「工具调用全流程」的事件字典。

    ⚠️ 形状与框架在总线上发布的一致（``model_dump(mode="json")`` 之后）：
        ``type`` 是**字符串**、``state`` 也是**字符串**。适配器两种形态都吃，
        但总线路径上就是字符串，这里必须按总线路径造。

    ⚠️ **不带** ``_entry_id``：这些事件是直接 ``publish`` 的（没有走
        ``log_append``），所以总线上不该有那个字段。本端点按它去重回放与
        订阅的重叠，缺了它就等同于「这是一条不需要去重的实时事件」。

    Returns:
        `list[dict[str, Any]]`: 依次投递的事件。
    """
    reply_id = "reply-chains-1"
    call_id = "call-1"
    return [
        {"type": "REPLY_START", "reply_id": reply_id, "session_id": "s", "name": "main"},
        {
            "type": "TOOL_CALL_START",
            "reply_id": reply_id,
            "tool_call_id": call_id,
            "tool_call_name": TOOL_NAME,
        },
        {
            "type": "TOOL_CALL_DELTA",
            "reply_id": reply_id,
            "tool_call_id": call_id,
            "delta": '{"city": "杭州"}',
        },
        {"type": "TOOL_CALL_END", "reply_id": reply_id, "tool_call_id": call_id},
        {
            "type": "TOOL_RESULT_START",
            "reply_id": reply_id,
            "tool_call_id": call_id,
            "tool_call_name": TOOL_NAME,
        },
        {
            "type": "TOOL_RESULT_TEXT_DELTA",
            "reply_id": reply_id,
            "tool_call_id": call_id,
            "delta": "找到 3 家酒店",
        },
        {
            "type": "TOOL_RESULT_END",
            "reply_id": reply_id,
            "tool_call_id": call_id,
            # ⚠️ ``state`` 的**字符串值是小写**（``ToolResultState.SUCCESS
            # == "success"``）。写成 ``"SUCCESS"`` 不会报错 —— 适配器认不出
            # 它就走了「失败」分支，任务带着结果文本被标成 FAILED，界面上
            # 是一个红叉配一句正常的结果，最难发现的那类错。
            "state": "success",
        },
    ]


def _tasks_of(stream: ChainStream) -> list[dict[str, Any]]:
    """把已收到的所有帧里的任务拍平成一个列表。

    Args:
        stream (`ChainStream`): 读取器。

    Returns:
        `list[dict]`: 所有帧里出现过的任务记录（含重复的中间态）。
    """
    return [task for frame in stream.frames for task in frame["tasks"]]


# ==============================================================================
# 一、投影：事件 → 任务清单快照
# ==============================================================================
async def test_chains_stream_pushes_task_states(live_server: LiveServer) -> None:
    """★ 工具调用的全流程被投影成任务清单，逐帧推出来。

    顺序是刻意固定的 —— **先连 SSE，再发事件**（回复结束后框架的 ``_persist``
    会 ``log_trim`` 掉这一轮回放日志，那时再连就什么都收不到）：

        1. 建凭据 / 智能体 / 会话；
        2. 打开思考链 SSE 并等到订阅者真的挂上总线；
        3. 往该会话的事件通道投递一串工具调用事件；
        4. 等到任务被推成 ``DONE``。

    同时断言「只在有变化时才发」：帧内容两两不同。
    """
    async with _client(live_server.base_url) as client:
        fixture = await _bootstrap(client, ALICE_HEADERS)
        key = MessageBusKeys.session_events(fixture.session_id)

        async with open_chains(client, fixture, ALICE_HEADERS) as stream:
            assert isinstance(stream, ChainStream)
            await wait_for_subscriber(live_server.bus, fixture.session_id)

            for event in _tool_call_events():
                await live_server.bus.publish(key, event)

            await stream.wait_until(
                lambda s: any(
                    task.get("state") == "DONE" for task in _tasks_of(s)
                ),
            )

        tasks = _tasks_of(stream)

    finished = [task for task in tasks if task["state"] == "DONE"]
    assert finished, f"没有收到任何已完成的任务。帧：{stream.frames}"
    task = finished[-1]
    assert task["name"] == TOOL_NAME
    assert task["result"] == "找到 3 家酒店"
    # 标题来自 ``TITLES``，是给用户看的中文说明 —— 断言它非空且不是工具名，
    # 因为「标题退化成工具名」在界面上就等于没翻译。
    assert task["title"] and task["title"] != TOOL_NAME
    # 参数从分片拼出来后解析成了字典 —— 界面展开详情时靠它。
    assert task["arguments"] == {"city": "杭州"}
    # 默认不推推理文本，但字段**始终存在**（保持帧形状稳定）。
    assert all(frame["reasoning"] == "" for frame in stream.frames)

    # 「只在有变化时才发」：相同内容不会重复发。
    serialized = [json.dumps(frame, ensure_ascii=False, sort_keys=True) for frame in stream.frames]
    assert len(serialized) == len(set(serialized)), (
        f"收到了内容完全相同的重复帧，说明没有按「变化才发」去重：{stream.frames}"
    )


# ==============================================================================
# 二、真实回复驱动：推理文本
# ==============================================================================
async def test_chains_stream_streams_reasoning_from_a_live_reply(
    live_server: LiveServer,
) -> None:
    """★ 真实的 ``POST /chat/`` 一轮回复，思考块被推成推理文本。

    ⚠️ 这条用例与上面那条互补：上面直接往总线上发事件（投影逻辑），
        这条走**完整的框架链路**（HTTP → 后台 run → 模型 → 总线 → 端点），
        证明端点在真实事件流上也能用。

    ⚠️ ``expose_reasoning=true`` 必须显式传：默认关闭时适配器**根本不累积**
        推理（见 ``src/chains/events.py`` 的说明），推出来永远是空串。
    """
    async with _client(live_server.base_url) as client:
        fixture = await _bootstrap(client, ALICE_HEADERS)

        async with open_chains(
            client,
            fixture,
            ALICE_HEADERS,
            expose_reasoning=True,
        ) as stream:
            assert isinstance(stream, ChainStream)
            await wait_for_subscriber(live_server.bus, fixture.session_id)

            trigger = _json(
                await client.post(
                    "/chat/",
                    headers=ALICE_HEADERS,
                    json=_chat_body(fixture, CHAT_WITH_THINK),
                ),
            )
            assert trigger["status"] == "started"

            await stream.wait_until(
                lambda s: any(THINK_TEXT in frame["reasoning"] for frame in s.frames),
            )

        assert stream.frames, "一个数据帧都没收到"


# ==============================================================================
# 三、属主校验
# ==============================================================================
async def test_chains_stream_is_404_for_non_owner(live_server: LiveServer) -> None:
    """★ 别人的会话返回 **404**，本人的返回 200。

    ⚠️ 必须同时断言「本人能连上」：只断言「他人 404」的话，一个把所有人都
        404 掉的实现也能通过 —— 那种实现当然「隔离」，但它同时也什么都做不了。

    ⚠️ 断言 404 而不是 403：403 等于告诉调用方「这个会话存在，只是不是你的」，
        把「谁拥有什么」泄露出去了。404 让两者不可区分。
    """
    async with _client(live_server.base_url) as client:
        alice = await _bootstrap(client, ALICE_HEADERS)

        # 他人身份：404。
        async with open_chains(client, alice, BOB_HEADERS) as response:
            assert isinstance(response, httpx.Response)
            assert response.status_code == 404

        # 本人身份：200（否则上一条断言毫无鉴别力）。
        async with open_chains(client, alice, ALICE_HEADERS) as stream:
            assert isinstance(stream, ChainStream)


async def test_chains_stream_is_scoped_to_the_sessions_own_channel(
    live_server: LiveServer,
    monkeypatch: Any,
) -> None:
    """★ 跨租户隔离：往 A 的会话通道发事件，B 的连接上收不到。

    ⚠️ 这条比「他人 404」更进一步：它证明连接是**绑在自己会话的通道**上的，
        而不是绑在一个全局通道上（那样即使 404 挡住了直连，B 也能从自己的
        连接上读到 A 的事件）。断言 B 上**零数据帧、只有心跳**。

    ⚠️ B 必须能连上自己的会话（200）—— 否则「收不到」可能只是因为他的连接
        压根没建起来。

    ⚠️ 把心跳压到 0.25s：这里要等「几帧心跳」来确认 B 的连接一直是活的，
        用默认的 30s 会让用例等半分钟。
    """
    monkeypatch.setattr(CHAIN_HEARTBEAT_GLOBAL, FAST_HEARTBEAT_SECS)

    async with _client(live_server.base_url) as client:
        alice = await _bootstrap(client, ALICE_HEADERS)
        bob = await _bootstrap(client, BOB_HEADERS)
        alice_key = MessageBusKeys.session_events(alice.session_id)

        async with open_chains(client, bob, BOB_HEADERS) as bob_stream:
            assert isinstance(bob_stream, ChainStream)
            await wait_for_subscriber(live_server.bus, bob.session_id)

            # 往 alice 的通道上发一串事件；bob 订阅的是他自己的通道。
            for event in _tool_call_events():
                await live_server.bus.publish(alice_key, event)

            # 给足时间让事件（若会串的话）送达 —— 等 bob 攒下几帧心跳，
            # 确认这段时间里他的连接一直是活的（不是「断了所以没收到」）。
            await bob_stream.wait_until(lambda s: s.heartbeats >= 2, timeout=5.0)

            assert bob_stream.frames == [], (
                f"bob 在自己的思考链连接上收到了 alice 的事件：{bob_stream.frames}"
            )


# ==============================================================================
# 四、心跳
# ==============================================================================
async def test_chains_stream_sends_heartbeat_when_idle(
    live_server: LiveServer,
    monkeypatch: Any,
) -> None:
    """★ 没有任何事件时，服务端持续发 ``:\\n\\n`` 心跳，且不发数据帧。

    ★ 心跳不是装饰，它有两个硬作用：

        1. **穿透中间设备**：Nginx / 云 LB 普遍对「空闲超过 N 秒的连接」超时
           断开。没有心跳就会被静默掐断，客户端表现为「随机地少收最后一段」。
        2. **让客户端能区分「还活着但没内容」与「已经死了」**：连接一旦断了，
           前端的自动重连才有依据。

    这里把心跳间隔 monkeypatch 成 0.25s，避免用例真的等 30 秒。
    """
    # 见 CHAIN_HEARTBEAT_GLOBAL 的注释：这个常量是在生成器循环体里求值的，
    # 所以改模块属性真的生效。
    monkeypatch.setattr(CHAIN_HEARTBEAT_GLOBAL, FAST_HEARTBEAT_SECS)

    async with _client(live_server.base_url) as client:
        fixture = await _bootstrap(client, ALICE_HEADERS)

        async with open_chains(client, fixture, ALICE_HEADERS) as stream:
            assert isinstance(stream, ChainStream)
            await wait_for_subscriber(live_server.bus, fixture.session_id)

            # 一个字都不发，纯等心跳。等到至少两帧再断言 —— 只等一帧的话，
            # 「第一帧是别的东西凑巧被当成心跳」这种误判也能通过。
            await stream.wait_until(lambda s: s.heartbeats >= 2)

            assert stream.heartbeats >= 2
            # 空闲期间不该有任何数据帧混进来 —— 会话根本没被触发。
            assert stream.frames == [], f"空闲连接上收到了数据帧：{stream.frames}"


# ==============================================================================
# 回放取「尾部」而不是「头部」
# ==============================================================================
async def test_read_replay_tail_returns_the_newest_entries_not_the_oldest() -> None:
    """回放必须取日志**末尾**的 N 条 —— 单次 ``log_read`` 取的是**最旧**的 N 条。

    ★ 这条用例同时钉住两件事，缺一不可：

        1. ``log_read(key, max_count=N)`` 确实是**从头**读的（对照组）；
        2. ``_read_replay_tail`` 给出的是**末尾** N 条（被测行为）。

    只断言第 2 点的话，用例在「两者恰好相同」的实现上也会通过 ——
    而那种实现（比如日志长度一直不超过 max_count）根本覆盖不到这条缺陷。

    ⚠️ 刻意**不**给 ``log_append`` 传 ``max_len``：内存总线会按 ``max_len``
    **精确**裁剪，一传就把日志压回上限以内，那正是这条缺陷在测试里
    永远跑不出来的原因（生产用 Redis，走的是 ``XADD MAXLEN ~N`` 的近似裁剪）。
    """
    bus = InMemoryMessageBus()
    key = MessageBusKeys.session_events("s-replay-tail")
    for seq in range(5):
        await bus.log_append(key, {"type": "TEXT_BLOCK_DELTA", "seq": seq})

    # 对照组：单次读取拿到的是最旧的 2 条。
    head_first = await bus.log_read(key, max_count=2)
    assert [payload["seq"] for _, payload in head_first] == [0, 1], (
        "log_read 的语义变了（不再是从头读）——本用例的前提需要重新核对。"
    )

    tail = await _read_replay_tail(bus, key, max_count=2)
    assert [payload["seq"] for _, payload in tail] == [3, 4]


async def test_read_replay_tail_returns_everything_when_the_log_is_short() -> None:
    """日志比窗口短时，尾部就是全部 —— 且只读一页（不引入额外往返）。"""
    bus = InMemoryMessageBus()
    key = MessageBusKeys.session_events("s-replay-short")
    for seq in range(3):
        await bus.log_append(key, {"type": "REPLY_START", "seq": seq})

    tail = await _read_replay_tail(bus, key, max_count=1000)
    assert [payload["seq"] for _, payload in tail] == [0, 1, 2]


async def test_read_replay_tail_warns_instead_of_silently_truncating(
    monkeypatch: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """翻页到上限时**必须留一条 warning**。

    ⚠️ 这条不是「顺手加的日志断言」：回放不到尾部的事件，症状是
    前端某个工具条**永远转圈**，而日志里一片安静 —— 那是最难归因的一类现象。
    """
    monkeypatch.setattr("src.server.routers._chains._REPLAY_MAX_PAGES", 1)
    bus = InMemoryMessageBus()
    key = MessageBusKeys.session_events("s-replay-capped")
    for seq in range(4):
        await bus.log_append(key, {"type": "TOOL_CALL_END", "seq": seq})

    with caplog.at_level(logging.WARNING, logger="src.server.routers._chains"):
        await _read_replay_tail(bus, key, max_count=2)

    assert any(
        "页" in record.getMessage() for record in caplog.records
    ), f"翻页到上限却没有 warning：{[r.getMessage() for r in caplog.records]}"


# ==============================================================================
# 订阅建立失败：必须**断开**，不能永久挂起
# ==============================================================================
class _UnsubscribableBus:
    """订阅一定失败的替身 —— 模拟「总线不可达 / 订阅超时」。

    ⚠️ 用替身而不是 monkeypatch 真实总线的方法，是因为要模拟的正是
    「``subscribe`` 在 ``on_ready`` 之前就抛」这一条路径；真实总线在
    内存实现里不会抛，而在 Redis 实现里要复现得先掐掉 Redis。
    """

    def __init__(self) -> None:
        """初始化计数器。"""
        self.subscribe_calls = 0

    def subscribe(self, key: str, on_ready: Any = None) -> Any:
        """返回一个一执行就抛的异步生成器。

        Args:
            key (`str`): 订阅键（本替身不使用）。
            on_ready (`Any`): 订阅就绪回调（本替身**不会**调用它 ——
                这正是被模拟的故障：``on_ready`` 永远不会被触发）。

        Returns:
            `Any`: 异步生成器。
        """
        self.subscribe_calls += 1

        async def _gen() -> Any:
            """模拟订阅失败。

            Yields:
                `Any`: 永不产出（先抛异常）。
            """
            raise ConnectionError("模拟消息总线不可达：订阅建立前就失败")
            yield  # pragma: no cover —— 让函数成为异步生成器

        return _gen()


async def test_chains_stream_ends_instead_of_hanging_when_the_subscription_fails(
    app: FastAPI,
    client: httpx.AsyncClient,
    monkeypatch: Any,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """订阅建立失败时，SSE 必须**在有限时间内断开**，并留下日志。

    ★ 这是「永久挂起」这一类故障的回归用例：

        ``on_ready`` 只在订阅**成功**后调用。订阅抛异常时它永不触发，
        于是无超时的 ``await ready.wait()`` 会永远卡住 —— 客户端拿到的是
        一条 ``200 text/event-stream`` 的死连接（既无数据、无心跳、也不断开），
        浏览器不会重连，而服务端**一条日志都没有**。

    ⚠️ 这里直接调用端点函数并迭代 ``body_iterator``，而不是起 uvicorn：
        要断言的是「生成器会自己收尾」这件事。走真实 TCP 反而要处理
        连接关闭的时序，把「有没有断开」变成一个时序问题。
    """
    monkeypatch.setattr(CHAIN_SUBSCRIBE_TIMEOUT_GLOBAL, FAST_SUBSCRIBE_TIMEOUT_SECS)
    fixture = await _bootstrap(client, ALICE_HEADERS)
    bus = _UnsubscribableBus()

    with caplog.at_level(logging.WARNING, logger="src.server.routers._chains"):
        response = await session_chains(
            session_id=fixture.session_id,
            agent_id=fixture.agent_id,
            expose_reasoning=False,
            user_id="alice",
            storage=app.state.storage,
            message_bus=bus,
        )
        # 用 wait_for 兜住「真的挂住了」：超时会以 TimeoutError 失败，
        # 而不是把整个测试会话拖到全局超时。
        frames = await asyncio.wait_for(
            _drain(response.body_iterator),
            timeout=READ_TIMEOUT_SECS,
        )

    assert bus.subscribe_calls == 1, "应当只尝试订阅一次"
    assert frames == [], f"订阅都没建立，却推了帧：{frames}"
    messages = [record.getMessage() for record in caplog.records]
    assert any("订阅未能在" in message for message in messages), (
        f"订阅超时没有留日志：{messages}"
    )
    assert any("订阅异常退出" in message for message in messages), (
        f"订阅失败没有被收敛成一条 warning：{messages}"
    )


async def _drain(iterator: Any) -> list[str]:
    """把 SSE 生成器抽干（不设超时 —— 由调用方用 ``wait_for`` 兜住）。

    Args:
        iterator (`Any`): ``StreamingResponse.body_iterator``。

    Returns:
        `list[str]`: 依次产出的 SSE 帧文本。
    """
    return [frame async for frame in iterator]


# ==============================================================================
# 端到端接线：超过回放窗口时，端点回放的必须是**尾部**
# ==============================================================================
#: 回放日志的窗口大小（``MessageBusKeys.SESSION_REPLAY_MAX_LEN``）。
REPLAY_WINDOW = MessageBusKeys.SESSION_REPLAY_MAX_LEN

#: 一批事件里的条数（见 :func:`_batch_events`）。
REPLAY_EVENTS_PER_BATCH = 6

#: 灌进日志的批次数：167 × 6 = 1002 条，**刚好**越过 1000 条的窗口。
#:
#: ⚠️ 「刚好越过」是刻意的，而且**每批的最后一条必须是「任务收尾」事件**
#: （``TOOL_RESULT_END``）：只有这样，被窗口切掉的那一条才会让最后一批的任务
#: 永远停在「进行中」—— 旧实现（读头部）下用例才会红。
#: 若把无关键作用的 ``REPLY_END`` 放在批尾，切掉的只是它，用例在旧实现下
#: 照样通过 —— 那就成了一条没有分辨力的用例。
REPLAY_OVERSIZE_BATCHES = REPLAY_WINDOW // REPLAY_EVENTS_PER_BATCH + 1

#: 等「1002 帧全部回放完」的时间预算。
#:
#: ⚠️ 这里**刻意**不图快。曾经用过 ``5.0``，理由是「本地回放一千条不该要 5 秒」——
#: 那在空载机器上成立，在负载下不成立：整套用例并发跑（或本机同时在编译前端）
#: 时，光是把这 1002 帧读进来、逐行解析就超过 5 秒，于是用例红在一个
#: **与它要挡的缺陷毫无关系**的原因上。实测三次全量跑里红过一次，且红的那次
#: 失败信息把人往「回放读了头部」上引 —— 一条会把排查带偏的用例，比一条慢的用例糟得多。
#:
#: 超时只影响**失败路径**的耗时：``wait_until`` 条件一满足就返回，通过时并不会真的等这么久。
REPLAY_WAIT_TIMEOUT_SECS = READ_TIMEOUT_SECS


def _batch_events(index: int) -> list[dict[str, Any]]:
    """造一批带有**唯一** ``tool_call_id`` 的工具调用事件。

    ⚠️ 不能直接复用 :func:`_tool_call_events`：它的 ``call_id`` 是写死的，
    所有批次会全部落进**同一个**任务上，于是「最后一批收尾事件丢了」这件事
    就观察不到了 —— 前面那些批次早把同一个任务推成了 DONE。

    ⚠️ 批尾**必须**是 ``TOOL_RESULT_END``（见 ``REPLAY_OVERSIZE_BATCHES``
    的注释）：它是让任务变成 DONE 的那一条，也是被窗口切掉时唯一能暴露
    缺陷的那一条。

    Args:
        index (`int`): 批次序号，用来派生唯一 id。

    Returns:
        `list[dict[str, Any]]`: 该批的 ``REPLAY_EVENTS_PER_BATCH`` 条事件。
    """
    reply_id = f"reply-{index}"
    call_id = f"call-{index}"
    return [
        {"type": "REPLY_START", "reply_id": reply_id, "session_id": "s", "name": "main"},
        {
            "type": "TOOL_CALL_START",
            "reply_id": reply_id,
            "tool_call_id": call_id,
            "tool_call_name": TOOL_NAME,
        },
        {
            "type": "TOOL_CALL_END",
            "reply_id": reply_id,
            "tool_call_id": call_id,
        },
        {
            "type": "TOOL_RESULT_START",
            "reply_id": reply_id,
            "tool_call_id": call_id,
            "tool_call_name": TOOL_NAME,
        },
        {
            "type": "TOOL_RESULT_TEXT_DELTA",
            "reply_id": reply_id,
            "tool_call_id": call_id,
            "delta": "找到 3 家酒店",
        },
        {
            "type": "TOOL_RESULT_END",
            "reply_id": reply_id,
            "tool_call_id": call_id,
            "state": "success",
        },
    ]


async def test_chains_stream_replays_the_tail_when_the_log_exceeds_the_window(
    live_server: LiveServer,
) -> None:
    """★ 日志超过回放窗口时，最新的事件**必须**被回放出来。

    ★ 这条用例的构造方式（直接 ``log_append``，不走 ``publish``）：

        连接是在**所有事件写完之后**才建立的，因此订阅队列里一条都没有 ——
        客户端能看到的**只可能**来自回放。这把「回放取的是头部还是尾部」
        变成一个纯粹的、无时序竞争的断言。

    ★ 失败模式（正是本用例要挡住的）：
        取头部 ⇒ 窗口里没有最后一批的 ``TOOL_RESULT_END`` ⇒ 那个任务永远
        停在「进行中」，界面上一条工具永远转圈；而且**重连也自愈不了**，
        因为重连走的还是同一段回放。
    """
    async with _client(live_server.base_url) as client:
        fixture = await _bootstrap(client, ALICE_HEADERS)
        key = MessageBusKeys.session_events(fixture.session_id)

        total = 0
        for index in range(REPLAY_OVERSIZE_BATCHES):
            for event in _batch_events(index):
                await live_server.bus.log_append(key, event)
                total += 1
        assert total > REPLAY_WINDOW, (
            f"只灌了 {total} 条，没有越过回放窗口 {REPLAY_WINDOW} —— "
            "用例的前提不成立，它会在旧实现下也通过。"
        )

        async with open_chains(client, fixture, ALICE_HEADERS) as stream:
            assert isinstance(stream, ChainStream)
            # ⚠️ 任务的 id 由**收集器**分配（``task-<n>``），不是工具调用 id。
            # 批次 i（从 0 数）创建的就是 ``task-{i+1}``。
            last_task_id = f"task-{REPLAY_OVERSIZE_BATCHES}"
            # ⚠️ 超时/失败都**吞掉** ``wait_until`` 的 AssertionError：它会把已
            # 收到的帧全部打进断言消息，而这里有一千多帧（几 MB）。真正的判据
            # 放在下面统一断言，失败信息只有一行。
            timed_out = False
            try:
                await stream.wait_until(
                    lambda s: any(
                        task.get("task_id") == last_task_id
                        and task.get("state") == "DONE"
                        for task in _tasks_of(s)
                    ),
                    timeout=REPLAY_WAIT_TIMEOUT_SECS,
                )
            except AssertionError:  # noqa: PERF203 — 见上：只为压掉巨量帧输出
                # ⚠️ 记下「是等超时了」而不是「等到了但内容不对」——
                # 这两件事的判据在同一条断言里，但排查方向完全相反（见下面的分支）。
                timed_out = True

        states = {
            task["task_id"]: task["state"]
            for task in _tasks_of(stream)
        }

    # ⚠️ 超时与「回放内容不对」是两种故障，失败信息必须分开 ——
    # 混在一起时，一句「收尾事件没有被回放」会把排查引向回放窗口的实现，
    # 而真实原因可能只是这台机器当时很忙。
    assert not timed_out, (
            f"等了 {REPLAY_WAIT_TIMEOUT_SECS}s 也没等到最后一批（{last_task_id}）"
            f"变成 DONE：已收到 {len(stream.frames)} 帧，"
            f"其中 {last_task_id} 的状态是 {states.get(last_task_id)!r}。\n"
            "    先怀疑**慢**，而不是**丢**：这条用例要读 1000+ 帧，"
            "机器同时在忙别的（并发跑测试、编译前端）时读不完是正常的。\n"
            "    只有当它在空载机器上稳定复现时，才去看回放窗口的实现 —— "
            "那时多半是回放取了日志最旧的 N 条而不是最新 N 条。"
        )

    assert states.get(last_task_id) == "DONE", (
        f"最后一批（{last_task_id}）的收尾事件没有被回放："
        f"它的状态是 {states.get(last_task_id)!r}。\n"
        "    最可能的原因：回放读了日志**最旧**的 N 条而不是最新 N 条 —— "
        "Redis 的近似裁剪会让日志超过窗口，最新那几条两处都拿不到。"
    )
