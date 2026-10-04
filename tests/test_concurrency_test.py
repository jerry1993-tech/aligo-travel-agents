# -*- coding: utf-8 -*-
"""``scripts/concurrency_test.py`` 的回归测试。

==============================================================================
这里防的不是「脚本能不能跑」，而是「它报的绿灯是不是真的」
==============================================================================
    并发测试是那种**最容易做成假绿灯**的东西：一个线程池、三发请求、
    打印「3/3 通过」—— 看上去什么都有。而它可能在验的是：
        · 三个人其实在排队（没有并发）；
        · 判据在清理之后才取证，于是全部 404（本次开发实测踩到）；
        · 判据用子串搜索，把「回答里正常出现的词」当成串号（本次开发实测踩到）；
        · 判据空验 —— 事件里根本没有可比对的字段，它照样返回通过。

    上面四条里有两条**在开发过程中真实发生过**，各自的用例就在下面。
    一个不会失败的并发测试，比没有并发测试更糟：它给出的是「验过了」的错觉。

==============================================================================
哪些用例走假服务、哪些直接调函数
==============================================================================
    · 走假服务的（``_MultiUserServer``）：需要走完整 HTTP 流程才能断言的东西 ——
      取证与清理的**先后顺序**、跨用户 404、清理只删自己建的。
    · 直接调函数的：判据本身是纯函数（拿 ``UserRun`` / ``Turn`` 就能算），
      手工造数据比搭一套假服务更能把边界说清楚，也跑得更快。

    ⚠️ 假服务是**有状态**的：删除真的会把记录拿掉。理由与
    ``tests/test_smoke_checks.py`` 那份假服务相同 —— 一个不会变的桩，
    会让「删干净了」与「根本没删」在断言上长得一模一样。
"""

from __future__ import annotations

import argparse
import json
import threading
from typing import Any

import httpx
import pytest

from scripts import concurrency_test as ct

BASE_URL = "http://concurrency.test"

#: 与脚本内一致的两条默认问句，方便组装期望值。
PROMPT_FAST = ct.DEFAULT_CASES[0].prompt
PROMPT_ORDER = ct.DEFAULT_CASES[1].prompt
PROMPT_POLICY = ct.DEFAULT_CASES[2].prompt


# ==============================================================================
# 假服务
# ==============================================================================


class _MultiUserServer:
    """够让并发测试跑完一轮的假服务（多用户、有状态、按 owner 隔离）。

    ⚠️ 按 ``X-User-ID`` 隔离是真的隔离：A 读 B 的智能体名下会话返回 404，
    与框架的 ``DenyAllResourceAccessPolicy`` 一致。不这么做的话，
    「跨用户不可见」那条判据在测试里永远不会失败 —— 而它正是要防串号的。

    Attributes:
        agents (`dict[str, list[str]]`): 各用户名下的智能体 id。
        sessions (`dict[str, list[str]]`): 各用户名下的会话 id。
        owners (`dict[str, str]`): 会话 id → 所属用户。
        agent_owner (`dict[str, str]`): 智能体 id → 所属用户。
        prompts (`dict[str, str]`): 会话 id → 该会话里那句 user 消息的原文。
        calls (`list[tuple[str, str]]`): 按发生顺序记录的 ``(method, path)``。
        deletes (`list[str]`): 被 DELETE 的路径。
    """

    def __init__(
        self,
        *,
        assistant_text: str = "好的。",
        stream_text: str = "好的。",
        chat_status: int = 200,
        delete_status: int = 204,
        stream_ok: bool = True,
        expected_chats: int | None = None,
        sessions_survive_delete: bool = False,
        sessions_list_status: int = 200,
    ) -> None:
        self.agents: dict[str, list[str]] = {}
        self.sessions: dict[str, list[str]] = {}
        self.owners: dict[str, str] = {}
        self.agent_owner: dict[str, str] = {}
        self.prompts: dict[str, str] = {}
        self.calls: list[tuple[str, str]] = []
        self.deletes: list[str] = []
        self.assistant_text = assistant_text
        self.stream_text = stream_text
        self.chat_status = chat_status
        self.delete_status = delete_status
        self.stream_ok = stream_ok
        self.expected_chats = expected_chats
        #: ``True`` = 删会话时**返回 404 但什么也没删** —— 复刻 ``_delete`` 分不清的
        #: 那种情况（「已经没了」vs「压根没删掉」）。判据 10 必须靠回读抓住它。
        self.sessions_survive_delete = sessions_survive_delete
        #: ``GET /sessions/`` 的状态码。非 200 = 回读不到 ⇒ 判据必须报「没验到」。
        self.sessions_list_status = sessions_list_status
        self.chats_seen = 0
        self._chats_ready = threading.Event()
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        #: ``GET /metrics`` 的响应体。默认是**一份没有任何 5xx 的**快照，
        #: 于是判据 8 的服务端那一侧走的是「读到了、增量全是 0」这条分支 ——
        #: 而不是「读不到 ⇒ 未测量」。用例要造服务端 5xx 就改这个字段。
        self.metrics_body = (
            "# HELP aligo_http_requests_total HTTP 请求总数。\n"
            "# TYPE aligo_http_requests_total counter\n"
            'aligo_http_requests_total{method="GET",route="/readyz",status="200"} 5.0\n'
            'aligo_http_requests_total{method="POST",route="/chat/",status="200"} 3.0\n'
        )

    # -- 内部：带锁的计数 ------------------------------------------------------

    def _next(self, prefix: str) -> str:
        """按前缀各自计数。

        ⚠️ 计数**按前缀分开**，不共用一个 ``_seq``。共用的话，agent 会拿到
        1/3/5、session 拿到 2/4/6，于是「第一个会话是 session-1」这句话不成立 ——
        用例里凡是硬编码了 id 的地方都会静默打空（打到一个不存在的对象上，
        得到的是一份「没有异常」的报告）。
        """
        with self._lock:
            self._counters[prefix] = self._counters.get(prefix, 0) + 1
            return f"{prefix}-{self._counters[prefix]}"

    def _sse(self, session_id: str) -> bytes:
        """造一段「正常结束」的 SSE。

        ⚠️ 每个事件的 ``session_id`` 都填**这条流自己的**会话 —— 判据 5
        靠它判串号，填错了会让所有用例的判据 5 都失败。

        ⚠️ 回复文本取 ``self.stream_text``，**不是** ``self.assistant_text``。
        这两个是**两条不同的通路**，不能共用一个字段：

          · ``stream_text`` → SSE 的 ``TEXT_BLOCK_DELTA`` → ``turn.reply_text``
            → 判据 1/2/2b 看的是它；
          · ``assistant_text`` → ``GET /sessions/{id}/messages`` 里的 assistant
            消息 → 判据 6（记载不串号）看的是它。

        早先只有一个 ``assistant_text`` 时，「判据 2b」的用例改了字段却发现
        判据毫无反应 —— 因为那条判据压根不读这个通路。
        """
        events: list[dict[str, Any]] = [
            {"type": "REPLY_START", "session_id": session_id, "reply_id": "r1", "name": "main_plan"},
            {"type": "TEXT_BLOCK_DELTA", "session_id": session_id, "text": self.stream_text},
            {"type": "REPLY_END", "session_id": session_id, "finished_reason": "completed"},
        ]
        return "".join(f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events).encode()

    # -- 请求处理 -------------------------------------------------------------

    def handler(self, request: httpx.Request) -> httpx.Response:
        """``httpx.MockTransport`` 的处理函数。"""
        path = request.url.path
        method = request.method
        user_id = request.headers.get("X-User-ID", "")
        with self._lock:
            self.calls.append((method, path))

        if method == "DELETE":
            with self._lock:
                self.deletes.append(path)
            if self.delete_status not in (200, 204, 404):
                return httpx.Response(self.delete_status, json={"detail": "删不掉"})
            target = path.rsplit("/", 1)[-1]
            if path.startswith("/sessions/"):
                if self.sessions_survive_delete:
                    # 假装删掉了：状态码是 404（``_delete`` 认它），但记录还在。
                    return httpx.Response(404, json={"detail": "Session not found."})
                owner = self.owners.pop(target, None)
                if owner:
                    self.sessions[owner] = [s for s in self.sessions[owner] if s != target]
            elif path.startswith("/agent/"):
                owner = self.agent_owner.pop(target, None)
                if owner:
                    self.agents[owner] = [a for a in self.agents[owner] if a != target]
            return httpx.Response(204)

        if path == "/api/v1/default-model":
            return httpx.Response(
                200,
                json={
                    "chat_model_config": {"type": "dashscope_credential", "model": "qwen"},
                    "mode": "shared",
                },
            )

        if path == "/api/v1/me":
            return httpx.Response(200, json={"user_id": user_id})

        if path == "/agent/" and method == "GET":
            ids = self.agents.get(user_id, [])
            return httpx.Response(200, json={"agents": [{"id": a} for a in ids], "total": len(ids)})

        if path == "/agent/" and method == "POST":
            new_id = self._next("agent")
            with self._lock:
                self.agents.setdefault(user_id, []).append(new_id)
                self.agent_owner[new_id] = user_id
            return httpx.Response(201, json={"agent_id": new_id})

        if path == "/sessions/" and method == "GET":
            # ⚠️ 只在**删过会话之后**才让状态码生效：这个接口在清理之前也被判据
            # 用过，一上来就 500 会把「准备阶段」打挂，测的就不是清理这条路了。
            if self.sessions_list_status != 200 and any(
                d.startswith("/sessions/") for d in self.deletes
            ):
                return httpx.Response(self.sessions_list_status, json={"detail": "读不了"})
            agent_id = request.url.params.get("agent_id")
            # 照抄框架：智能体对调用者不可见 ⇒ 404。
            if self.agent_owner.get(agent_id) != user_id:
                return httpx.Response(404, json={"detail": f"Agent '{agent_id}' not found."})
            rows = [s for s in self.sessions.get(user_id, []) if True]
            return httpx.Response(
                200,
                json={"sessions": [{"session": {"id": s}} for s in rows], "total": len(rows)},
            )

        if path == "/sessions/" and method == "POST":
            new_id = self._next("session")
            with self._lock:
                self.sessions.setdefault(user_id, []).append(new_id)
                self.owners[new_id] = user_id
            return httpx.Response(201, json={"session_id": new_id})

        if path.endswith("/messages"):
            session_id = path.split("/")[2]
            # ⚠️ 别人的会话返回 404 —— 判据 7 全靠这条。
            if self.owners.get(session_id) is None:
                return httpx.Response(404, json={"detail": f"Session '{session_id}' not found."})
            if request.headers.get("X-User-ID") != self.owners[session_id]:
                return httpx.Response(404, json={"detail": f"Session '{session_id}' not found."})
            return httpx.Response(
                200,
                json={
                    "messages": [
                        {
                            "role": "user",
                            "name": "user",
                            "content": [
                                {"type": "text", "text": self.prompts.get(session_id, "")}
                            ],
                        },
                        {
                            "role": "assistant",
                            "name": "main_plan",
                            "content": [{"type": "text", "text": self.assistant_text}],
                        },
                    ],
                    "has_more": False,
                    "is_running": False,
                },
            )

        if path.endswith("/stream"):
            session_id = path.split("/")[2]
            if not self.stream_ok:
                return httpx.Response(404, json={"detail": "本用例不提供流"})
            # ★ 卡住这段流，直到**所有**用户的 /chat/ 都到齐了再放行。
            #
            # ⚠️ 这不是为了「更像真的」，而是为了让「确实同时」那条判据在测试里
            #    **可观测**：假服务的响应是瞬时的（而且 `_quiet` 把 sleep 也去掉了），
            #    于是先发出的那条流可能在后一条请求还没发出去时就结束了 ——
            #    三条**真的同时发出**的请求，时间窗却两两不相交，判据会报
            #    「实际上是串行的」。那是**假失败**，但它指向的问题是真实的：
            #    判据只在「服务端的活儿比线程启动抖动更长」时才有分辨力。
            #    真实的模型调用要几秒，这里就用「等齐」把那个前提补齐。
            if self.expected_chats:
                self._chats_ready.wait(timeout=5.0)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=self._sse(session_id),
            )

        if path == "/chat/" and method == "POST":
            body = json.loads(request.content)
            with self._lock:
                self.prompts[body["session_id"]] = body["input"]["content"][0]["text"]
                self.chats_seen += 1
                if self.expected_chats and self.chats_seen >= self.expected_chats:
                    self._chats_ready.set()
            if self.chat_status not in (200, 202):
                return httpx.Response(self.chat_status, json={"detail": "模型调用失败"})
            return httpx.Response(202, json={"status": "accepted"})

        if path == "/metrics":
            return httpx.Response(
                200,
                headers={"content-type": "text/plain; version=0.0.4"},
                text=self.metrics_body,
            )

        return httpx.Response(404, json={"detail": f"未桩化的路径 {method} {path}"})


# ==============================================================================
# 测试脚手架
# ==============================================================================


def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    """把脚本里的等待全部变成瞬时（补丁只在本用例内有效）。"""
    monkeypatch.setattr(ct.time, "sleep", lambda *_: None)


def _route_everything_through(
    monkeypatch: pytest.MonkeyPatch, server: _MultiUserServer
) -> None:
    """让脚本**内部另建**的每个客户端都走假服务。

    ⚠️ 与 ``tests/test_smoke_checks.py`` 的同名工具同一个理由：``_read_stream``
    与各处 ``_client()`` 都是**自己**建 ``httpx.Client``，不拦它们的话用例会真的
    去连 ``http://concurrency.test`` —— 于是要么空等满超时，要么把「连不上」
    当成结论，而真正要断言的那条分支根本没走到。
    """
    real_client = httpx.Client

    def _factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs.setdefault("transport", httpx.MockTransport(server.handler))
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _factory)


def _run_main(monkeypatch: pytest.MonkeyPatch, server: _MultiUserServer,
              argv: list[str] | None = None) -> int:
    """跑一遍 ``main()``，全程走假服务。"""
    if server.expected_chats is None:
        server.expected_chats = 3
    _quiet(monkeypatch)
    _route_everything_through(monkeypatch, server)
    return ct.main(["--base-url", BASE_URL, "--users", "3", *(argv or [])])


def _turn(user_id: str, *, start: float, end: float, text: str = "好的",
          session_id: str = "s") -> ct.Turn:
    """手工造一个「正常结束」的 Turn（判据是纯函数，不需要真跑对话）。"""
    turn = ct.Turn(round_index=1)
    turn.t_chat = start
    turn.t_end = end
    turn.chat_status = 200
    turn.stream_open = True
    turn.reply_text = text
    turn.finished_reason = "completed"
    turn.events = [
        {"type": "REPLY_START", "session_id": session_id},
        {"type": "TEXT_BLOCK_DELTA", "session_id": session_id, "text": text},
        {"type": "REPLY_END", "session_id": session_id, "finished_reason": "completed"},
    ]
    return turn


def _run(user_id: str, case: ct.Case, turn: ct.Turn, *, agent_id: str = "a",
         session_id: str = "s", created: bool = True) -> ct.UserRun:
    """手工造一个 UserRun。"""
    run = ct.UserRun(
        index=1, user_id=user_id, case=case, agent_id=agent_id,
        agent_created=created, session_ids=[session_id], turns=[turn],
    )
    return run


# ==============================================================================
# ★ 回归 1：取证必须在清理之前
# ==============================================================================


def test_transcript_evidence_is_gathered_before_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ 读会话历史必须发生在 DELETE 之前。

    ⚠️ 这是本脚本开发过程中**真实踩过**的坑：第一版把 ``_cleanup_all``
    放在判据之前，于是判据 6（会话记载不串号）读回来的全是 404 ——
    一条本该通过的判据报「Session not found」，而那个原因看上去与串号
    毫无关系。判据没错，是**取证顺序**错了。

    断言的是**调用顺序**，不是「有没有读」：读也读了，只是读晚了，
    而读晚了与没读的最终表现一样（都是拿不到内容），只有顺序能区分。
    """
    server = _MultiUserServer()
    code = _run_main(monkeypatch, server)

    assert code == 0, "假服务给出的是一轮完美对话，应当通过"
    reads = [i for i, (m, p) in enumerate(server.calls) if m == "GET" and p.endswith("/messages")]
    deletes = [i for i, (m, _) in enumerate(server.calls) if m == "DELETE"]
    assert reads, "压根没读会话历史 —— 那条判据成了空验"
    assert deletes, "压根没清理"
    assert max(reads) < min(deletes), (
        f"取证发生在清理之后：读历史的下标 {reads}，删除的下标 {deletes}。"
        "顺序反了的话，读回来的是 404。"
    )


# ==============================================================================
# ★ 回归 2：子串搜索的假阳性
# ==============================================================================


def test_another_users_prompt_as_a_normal_word_is_not_crosstalk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ 「住宿标准」出现在别人的**回答**里，不是串号。

    ⚠️ 这也是开发过程中**真实踩过**的坑：判据 6 的第一版是「会话历史的 JSON
    里有没有别的用户的问题原文」这种子串搜索。用户 1 的问题是「住宿标准」，
    而用户 3 的回答里写着「依据公司差旅**标准**…」—— 于是它报了一次串号，
    报的是这个系统里最严重的缺陷，而实际上什么都没发生。

    问句越短越容易在别人的回答里自然出现。一条会喊狼来了的判据，
    代价比不判还大：它教人忽略告警，下一次真串号时也一样被忽略。

    这里让**所有**会话的 assistant 文本里都含有用户 1 的问句原文，
    正确实现必须照样通过。
    """
    server = _MultiUserServer(assistant_text=f"依据公司差旅{PROMPT_FAST}：单晚不超过 600 元。")
    code = _run_main(monkeypatch, server)

    assert code == 0, (
        "回答里自然出现了别人的问句字面量，被误判成串号 —— "
        "判据退化成子串搜索了"
    )


def test_real_crosstalk_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """★ 反过来：真的串了必须抓住，不能为了消假阳性把判据做瞎。

    ⚠️ 这条与上一条**必须成对存在**。只修假阳性而不验证真阳性的话，
    把判据改成「永远返回通过」也能让上一条变绿 —— 那样两个用例合起来
    什么都没保证。这里让某个会话的 user 消息变成**别人的**问句。
    """
    server = _MultiUserServer()

    # 在假服务里把「会话的 user 消息」换掉：模拟落库时落错了人。
    original = server.handler

    def _hijack(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/messages") and request.url.path.split("/")[2] == "session-1":
            session_id = "session-1"
            return httpx.Response(
                200,
                json={
                    "messages": [
                        {
                            "role": "user",
                            "name": "user",
                            # ★ 别人的问句出现在了这条会话里
                            "content": [{"type": "text", "text": PROMPT_ORDER}],
                        },
                    ],
                    "has_more": False,
                },
            )
        return original(request)

    monkeypatch.setattr(server, "handler", _hijack)
    code = _run_main(monkeypatch, server)

    assert code == 1, "会话里躺着别人的问句，判据 6 竟然通过了"


def test_extra_user_message_is_caught() -> None:
    """★ 一个会话里有**两条** user 消息 = 有人往里写了东西。

    这正是「精确到消息」的核心：串号必然让「恰好一条 user 消息」不成立。
    """
    run = _run("concurrency-user-1", ct.DEFAULT_CASES[0], _turn("u1", start=0, end=10))
    run.turns[0].events.append({"type": "REPLY_END", "session_id": "s"})

    def _fake_get(url: str, **kwargs: Any) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": PROMPT_FAST}]},
                    {"role": "user", "content": [{"type": "text", "text": PROMPT_ORDER}]},
                    {"role": "assistant", "content": [{"type": "text", "text": "好"}]},
                ]
            },
        )

    import scripts.concurrency_test as module

    class _C:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def get(self, url, **kwargs): return _fake_get(url, **kwargs)

    original_client = module._client
    module._client = lambda *a, **k: _C()
    try:
        check = ct.check_transcript_isolation(BASE_URL, [run])
    finally:
        module._client = original_client

    assert check.ok is False
    assert "2 条" in check.detail, check.detail


# ==============================================================================
# ★ 回归 3：判据不许空验
# ==============================================================================


def test_stream_isolation_fails_when_nothing_carries_a_session_id() -> None:
    """★ 事件里一个 ``session_id`` 都没有时，必须**失败**，不能通过。

    ⚠️ 这是「空验」的典型：判据 5 的实现是「遍历事件，比对 session_id」。
    如果事件结构变了（字段改名、被削平），循环体一次都不执行，
    ``problems`` 是空的 —— 于是它高高兴兴地打印 ✅。
    一个什么都没验的判据打出绿灯，比它直接报错危险得多。
    """
    run = _run("concurrency-user-1", ct.DEFAULT_CASES[0], _turn("u1", start=0, end=10))
    run.turns[0].events = [
        {"type": "REPLY_START"},
        {"type": "TEXT_BLOCK_DELTA", "text": "好的"},
        {"type": "REPLY_END", "finished_reason": "completed"},
    ]
    check = ct.check_stream_isolation([run])
    assert check.ok is False
    assert "什么都没验到" in check.detail


def test_stream_isolation_catches_a_foreign_session_id() -> None:
    """★ 流里出现别人的 ``session_id`` 必须被抓住 —— 这是串号的直接证据。"""
    run = _run("concurrency-user-1", ct.DEFAULT_CASES[0], _turn("u1", start=0, end=10))
    run.turns[0].events.append({"type": "TEXT_BLOCK_DELTA", "session_id": "别人的会话", "text": "x"})
    check = ct.check_stream_isolation([run])
    assert check.ok is False
    assert "不属于自己" in check.detail


# ==============================================================================
# 「确实同时」这条判据本身
# ==============================================================================


def test_overlapping_windows_pass() -> None:
    """三人同时在飞 ⇒ 通过。"""
    runs = [
        _run("u1", ct.DEFAULT_CASES[0], _turn("u1", start=0.0, end=10.0)),
        _run("u2", ct.DEFAULT_CASES[1], _turn("u2", start=0.5, end=12.0)),
        _run("u3", ct.DEFAULT_CASES[2], _turn("u3", start=1.0, end=9.0)),
    ]
    # 最早结束 9.0 > 最晚开始 1.0 ⇒ 重叠
    assert ct.check_genuinely_concurrent(runs).ok is True


def test_sequential_windows_fail() -> None:
    """★ 排队串行 ⇒ **必须失败**，哪怕三条全部都成功了。

    ⚠️ 这条是整套测试的底线。「三个人都成功」可以由一次串行执行给出；
    只有时间窗重叠才能区分「并发跑通了」与「排队跑通了」。
    若这条判据不存在，一个把请求顺序发出的脚本也会打印「3/3 通过」。
    """
    runs = [
        _run("u1", ct.DEFAULT_CASES[0], _turn("u1", start=0.0, end=5.0)),
        _run("u2", ct.DEFAULT_CASES[1], _turn("u2", start=5.0, end=10.0)),
        _run("u3", ct.DEFAULT_CASES[2], _turn("u3", start=10.0, end=15.0)),
    ]
    check = ct.check_genuinely_concurrent(runs)
    assert check.ok is False, "串行执行被判成了并发"
    assert "实际上是串行的" in check.detail


# ==============================================================================
# 端到端：一轮全绿 / 各条判据的失败路径
# ==============================================================================


def test_happy_path_passes_every_check(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """假服务给出的完美一轮：退出码 0，且**每一条**判据都真的跑过。

    ⚠️ 「每一条都真的跑过」这句原先是空头支票 —— 早先只断言了退出码，
    而**把一条判据从列表里摘掉同样会让退出码是 0**（少一条判据 = 少一个
    可能失败的东西）。同一类缺陷在本仓库真实发生过：三个模型指标写好了
    但一个调用方都没有，``/metrics`` 上只有 HELP+TYPE。

    所以这里断言的是**判据名字的全集**。加判据要动这份清单 ——
    这正是本用例的目的：**逼改动的人读一遍它，而不是让新判据悄悄漏挂**。
    断言用的是 ``in``（子集）而不是相等：报告里还打印别的东西，
    而且真实判据名带说明性后缀，逐字相等会很脆。
    """
    server = _MultiUserServer()
    code = _run_main(monkeypatch, server)
    printed = capsys.readouterr().out

    assert code == 0
    assert server.agents and server.sessions, "产物建过"
    assert not server.agents.get("concurrency-user-1"), "自建智能体应当被删掉"

    expected = [
        "准备就绪",
        "全部正常结束",
        "回复非空",
        "回复是答复而非占位",
        "无错误事件",
        "确实同时",
        "流不串号",
        "记载不串号",
        "跨用户不可见",
        "无服务端错误",
        "业务接口身份正确",
        "产物已清理",
    ]
    missing = [name for name in expected if name not in printed]
    assert not missing, f"这些判据没有出现在报告里（漏挂了？）：{missing}"
    assert "12 条判据全部通过" in printed, f"报告里没数对条数：{printed[-400:]!r}"


def _stalled_turn(*, timed_out: bool, last: str = "TOOL_RESULT_START") -> ct.Turn:
    """造一个「没收到 REPLY_END」的轮次，用于测判据 1 的两种报法。"""
    turn = ct.Turn(round_index=1, chat_status=200, elapsed=120.0, timed_out=timed_out)
    turn.events = [{"type": "REPLY_START"}, {"type": last}]
    return turn


def test_a_timed_out_turn_is_not_reported_as_a_protocol_violation() -> None:
    """★ 超时与「流被结束了却没给 REPLY_END」必须分开报。

    ⚠️ 这两者原先共用一句话（「没收到 REPLY_END」），证据完全一样、含义却相反：
    超时**不能断定服务端有错**（可能只是脚本的 120 秒判据太短），而流被结束
    是确定的协议违规。合并成一句的后果是，看报告的人只能靠猜决定去查哪边。
    """
    case = ct.DEFAULT_CASES[0]
    timed = ct.check_completed([_run("u1", case, _stalled_turn(timed_out=True))])
    closed = ct.check_completed([_run("u1", case, _stalled_turn(timed_out=False))])
    assert timed.ok is False and closed.ok is False
    assert "等满" in timed.detail and "不能断定服务端有错" in timed.detail
    assert "协议违规" in closed.detail and "等满" not in closed.detail
    # 两种都要说出卡在哪一步 —— 没有它，「没收到 REPLY_END」只说结果不说原因。
    assert "TOOL_RESULT_START" in timed.detail


def test_chat_failure_fails_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """``POST /chat/`` 报错时整轮必须失败（哪怕产物照样清理干净）。"""
    server = _MultiUserServer(chat_status=500)
    assert _run_main(monkeypatch, server) == 1


def test_cleanup_failure_is_reported_and_fails_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """删不掉时必须报出来，且不能把失败伪装成通过。"""
    server = _MultiUserServer(delete_status=500)
    assert _run_main(monkeypatch, server) == 1


# ==============================================================================
# 判据 10 的会话一侧：DELETE 的 404 不足以证明删掉了
# ==============================================================================


def test_a_session_that_survives_deletion_fails_the_cleanup_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ 删会话只回了 404、实际没删 —— 判据 10 必须靠**回读列表**抓住它。

    ⚠️ 这条用例针对的是一个真实的逻辑漏洞：``_delete`` 把 404 和 204 一视同仁
    （删两次、目标本就不在，都算正常），于是「路由不认这个 id」和「删成功了」
    在状态码上**长得一模一样**。只看状态码的实现会在这条用例下**静默打绿**，
    然后打印「已删 N 条会话」—— 而一条都没删掉。
    """
    server = _MultiUserServer(sessions_survive_delete=True)
    code = _run_main(monkeypatch, server)
    assert code == 1, "会话还在，整轮不该通过"
    assert server.sessions.get("concurrency-user-1"), "构造前提：会话确实还在"


def test_an_unreadable_session_list_is_not_treated_as_clean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """回读不到会话列表 ⇒ 判「没验到」，**不能**当作删干净了。

    ⚠️ 「空集合上的没问题」与「没看」是同一件事（判据 0 起的同一条原则）。
    回读接口 500 时若判通过，这条判据就退化成一句不能证伪的声明。
    """
    server = _MultiUserServer(sessions_list_status=500)
    assert _run_main(monkeypatch, server) == 1


def test_the_session_list_parser_reads_the_real_nested_shape() -> None:
    """真实响应是嵌套的 ``{"session": {"id": …}}``，不是扁平的 ``{"id": …}``。

    ⚠️ 照扁平写 ``item.get("id")`` 会**永远读到 None**，集合恒为空、判据恒绿。
    所以这里钉住真实形状（取自线上 ``GET /sessions/`` 的实测响应）。
    """
    real = {"sessions": [{"session": {"id": "db16a289", "agent_id": "a1"}}], "total": 1}
    assert ct._session_ids_in_list(real) == {"db16a289"}
    assert ct._session_ids_in_list({"sessions": [], "total": 0}) == set()
    # 扁平形状也认（接口将来改平了不至于静默失效）。
    assert ct._session_ids_in_list({"sessions": [{"id": "flat"}]}) == {"flat"}


def test_the_session_list_parser_refuses_to_guess() -> None:
    """解析不出来时返回 ``None``，**不是**空集 —— 「读不懂」≠「列表是空的」。"""
    assert ct._session_ids_in_list({"detail": "nope"}) is None
    assert ct._session_ids_in_list({"sessions": "不是列表"}) is None
    assert ct._session_ids_in_list({"sessions": [{"session": {"agent_id": "x"}}]}) is None
    assert ct._session_ids_in_list({"sessions": [None]}) is None


def test_keep_leaves_artifacts_and_says_so(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--keep`` 时**一个字节都不删**，且报告里要说清楚是「按要求保留」。"""
    server = _MultiUserServer()
    code = _run_main(monkeypatch, server, ["--keep"])
    # 保留时「产物已清理」这条不判失败 —— 但必须留下痕迹说明为何没清。
    assert code == 0
    assert not server.deletes, f"--keep 却发了 DELETE：{server.deletes}"
    assert server.agents.get("concurrency-user-1"), "按要求应当保留"


def test_pre_existing_agent_is_reused_and_never_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ 用户已有的 ``main_plan`` 要复用、且**绝不删**。

    ⚠️ 这是最不能出错的一条：删掉别人正在用的智能体，比留一条垃圾会话严重得多。
    脚本靠 ``agent_created`` 区分「我建的」与「本来就有的」。

    做法：先在假服务里给该用户预置一个 ``main_plan``，把脚本的"按名字查找"
    变成命中，于是它不会新建。
    """
    server = _MultiUserServer()
    for user_index in (1, 2, 3):
        user_id = f"{ct.USER_PREFIX}-{user_index}"
        server.agents.setdefault(user_id, []).append(f"preexisting-{user_index}")
        server.agent_owner[f"preexisting-{user_index}"] = user_id

    # 让 GET /agent/ 把预置的那个带 name 返回，脚本才会认为它已存在。
    original = server.handler

    def _named(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/agent/" and request.method == "GET":
            user_id = request.headers.get("X-User-ID", "")
            ids = server.agents.get(user_id, [])
            return httpx.Response(
                200,
                json={
                    "agents": [
                        {"id": a, "data": {"name": ct.MAIN_AGENT_NAME}} for a in ids
                    ],
                    "total": len(ids),
                },
            )
        return original(request)

    monkeypatch.setattr(server, "handler", _named)
    code = _run_main(monkeypatch, server)

    assert code == 0, "复用已有智能体的一轮应当通过"
    for user_index in (1, 2, 3):
        assert f"preexisting-{user_index}" in server.agents.get(
            f"{ct.USER_PREFIX}-{user_index}", []
        ), "复用来的智能体被删掉了 —— 这比留垃圾严重得多"


# ==============================================================================
# 零观测不得算通过
# ==============================================================================


def _bare_run(user_id: str = "u1") -> ct.UserRun:
    """造一个「什么观测都没有」的 ``UserRun``：没有轮次、没有会话。

    ⚠️ 刻意**不给** ``session_ids``：给了的话，判据会真的发 HTTP，
    测试就从「验空集语义」变成了「验连不上假地址」，走偏了。
    """
    return ct.UserRun(index=0, user_id=user_id, case=ct.DEFAULT_CASES[0])


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("全部正常结束（REPLY_END + completed）", ct.check_completed),
        ("回复非空", ct.check_replies_nonempty),
        ("回复是答复而非占位", ct.check_replies_are_answers),
        ("无错误事件", ct.check_no_error_events),
        # ⚠️ 判据 4 原先不在这张表里，而它恰好有一个**真**的真空绿路径：
        #    循环上界取 ``runs[0].turns``，用户 1 准备阶段失败时 ``range(0)``
        #    让循环体一次都不执行，``problems`` 为空 ⇒ 返回绿。
        ("确实同时（时间窗重叠）", ct.check_genuinely_concurrent),
        # ⚠️ five_xx_before 给 None（=「没读到基线」）是安全的：本用例的 run
        #    一轮对话都没有，判据会在读 /metrics **之前**就返回「零观测」。
        #    这正是那个提前返回存在的意义之一 —— 它让这条用例不必连网。
        (
            "无服务端错误（无 5xx / 无 409）",
            lambda runs: ct.check_no_server_errors(BASE_URL, runs, None),
        ),
        (
            "会话记载不串号",
            lambda runs: ct.check_transcript_isolation(BASE_URL, runs),
        ),
        (
            "跨用户不可见（404）",
            lambda runs: ct.check_cross_user_invisible(BASE_URL, runs),
        ),
    ],
)
def test_a_check_with_nothing_observed_is_not_green(name: str, call: Any) -> None:
    """**零观测 ⇒ 判不通过**，一条都不能例外。

    这条防的是一整类假绿灯。这些判据都写成「遍历全部观测，有问题就记一笔」，
    于是「遍历到空集合」与「全都对」在代码上同形 —— 报告上都是 ✅，
    证据量却差着数量级。判据 5 最早有这道防护，其余几条没有，
    这个不对称本身就是缺陷：**能被空集骗过的判据不止一条，说明漏的是规则，
    不是某一处**。所以这里用 parametrize 把七条一次钉死，
    以后再加判据时，漏掉防护会被这条用例点名。

    正向对照（有观测时确实报绿）由
    :func:`test_happy_path_passes_every_check` 用完整的假服务覆盖 ——
    没有它，把判据改成「永远不通过」也能让本用例变绿。
    """
    check = call([_bare_run()])
    assert check.ok is False, f"{name} 在零观测时报了通过：{check.detail!r}"
    assert "什么都没验到" in check.detail, f"{name} 的说明没讲清是空观测：{check.detail!r}"


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("全部用户准备就绪", ct.check_setup),
        ("确实同时（时间窗重叠）", ct.check_genuinely_concurrent),
        ("业务接口身份正确", lambda runs: ct.check_business_api(BASE_URL, runs)),
        ("产物已清理", lambda runs: ct.check_cleanup(BASE_URL, runs, [], keep=False)),
    ],
)
def test_a_check_with_no_users_at_all_is_not_green(name: str, call: Any) -> None:
    """``runs`` 本身为空时也不能报绿 —— 与上一条是**两种不同的**空集形状。

    ⚠️ 上一条喂的是「有一个用户、但他一轮都没跑成」；这一条喂的是「一个用户都没有」。
    两者在不同判据上各自真空绿：判据 4 是前者（循环上界 = 0），判据 0/9/10 是后者
    （遍历空列表 ⇒ 没有 problem ⇒ 绿）。修了一种不等于修了另一种。

    ⚠️ 判据 10 在这一输入下还会**谎报**：它会打印「按 id 回读确认不在」，
    而回读一次都没发生（没有 id 可读）。所以这里在断言 ``ok is False`` 之外，
    另断言详情里不许出现「回读确认」——「我验过」是这个判据的全部价值。
    """
    check = call([])
    assert check.ok is False, f"{name} 在零用户时报了通过：{check.detail!r}"
    assert "回读确认" not in check.detail, f"{name} 在没回读时声称回读过：{check.detail!r}"


# ==============================================================================
# 判据 8 的服务端一侧：/metrics 上的 5xx 增量
# ==============================================================================
# ⚠️ 这一节补的是一个**证据面缺口**：判据 8 原先只扫客户端自己看到的
# ``POST /chat/`` 响应码。并发下最可能 5xx 的地方恰恰是客户端不直接看的
# —— SSE 流端点、/api/v1/*、会话读写。服务端的 aligo_http_requests_total
# 覆盖每一个路由。
#
# ⚠️ 「读不到」必须是**未测量**，不是 0。这是本仓库评测模块
# ``Observation.unmeasured`` 的同一条原则：把「没看」和「看了没有」说成同一件事，
# 是假绿灯里最难发现的一种（因为它在报告上和一个真的通过长得一模一样）。


def test_the_5xx_reader_parses_the_prometheus_text_format() -> None:
    """★ 解析：只挑 5xx，4xx/2xx 一概不算。"""
    body = (
        "# HELP aligo_http_requests_total …\n"
        "# TYPE aligo_http_requests_total counter\n"
        'aligo_http_requests_total{method="GET",route="/readyz",status="200"} 9.0\n'
        'aligo_http_requests_total{method="GET",route="/sessions/",status="404"} 4.0\n'
        'aligo_http_requests_total{method="POST",route="/chat/",status="500"} 2.0\n'
        'aligo_http_requests_total{method="GET",route="/sessions/{id}/stream",status="503"} 1.0\n'
        "aligo_ready 1.0\n"
    )
    counts = ct._read_5xx_counts_from_text(body)
    assert counts == {
        ("POST", "/chat/", "500"): 2.0,
        ("GET", "/sessions/{id}/stream", "503"): 1.0,
    }


def test_the_5xx_reader_gives_up_rather_than_guessing() -> None:
    """★★ 行解析不了 ⇒ ``None``（未测量），**不是**一条空的「零 5xx」。

    ⚠️ 这条用例守的是本判据最容易被写坏的地方。若解析失败时返回 ``{}``，
    判据 8 会打印「服务端增量全是 0」—— 一句话也**不成立**的绿灯，
    而它的成因（Prometheus 文本格式变了 / 指标被改名了）永远不会被发现。
    """
    body = 'aligo_http_requests_total{method="POST",status="500"} 1.0\n'
    assert ct._read_5xx_counts_from_text(body) is None, "缺 route 标签时竟然猜了一个值"


def test_only_new_5xx_counts() -> None:
    """★★ 判的是**增量**：运行前就存在的 5xx 不该让本次判失败。

    ⚠️ 不这么写的话，app 容器跑过几小时之后这条判据会**永久报红**，
    而人对永久红的告警的处置方式是把它删掉。
    """
    before = {("POST", "/chat/", "500"): 3.0}
    assert ct._diff_5xx(before, before) == []
    after = {("POST", "/chat/", "500"): 3.0, ("GET", "/stream", "500"): 2.0}
    problems = ct._diff_5xx(before, after)
    assert len(problems) == 1 and "/stream" in problems[0]


def test_a_server_side_5xx_the_client_never_saw_fails_the_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★★ 服务端记了 5xx、而客户端一个都没看见 ⇒ 判据 8 **必须红**。

    ⚠️ 这是这一节存在的全部理由。用例造的是：``POST /chat/`` 全部 200/202
    （客户端侧干干净净），但 ``/metrics`` 上多出来一条
    ``GET /sessions/{id}/stream → 500`` —— 也就是**客户端侧永远看不见的那一类**。
    把服务端那一路摘掉，本用例立刻变绿（= 变瞎），这正是修复前的状态。
    """
    runs = [_run("u1", ct.DEFAULT_CASES[0], _turn("u1", start=0, end=1))]
    server = _MultiUserServer()
    _route_everything_through(monkeypatch, server)
    before = ct._read_5xx_counts(BASE_URL)  # 基线：干净的
    assert before == {}, f"基线不该有 5xx：{before}"
    server.metrics_body += (
        'aligo_http_requests_total{method="GET",'
        'route="/sessions/{id}/stream",status="500"} 1.0\n'
    )

    check = ct.check_no_server_errors(BASE_URL, runs, before)

    assert check.ok is False, f"服务端 5xx 被漏掉了：{check.detail!r}"
    assert "/stream" in check.detail


def test_an_unreadable_metrics_endpoint_is_reported_as_unmeasured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ 读不到 ``/metrics`` ⇒ 详情里必须写「未测量」，且**不影响**其余判定。

    ⚠️ 为什么不是判失败：客户端侧的 5xx/409 证据仍然完整有效，
    只因为它读不到一个**增强**证据面就判红，会让人为了让它变绿而去关掉整个检查。
    折中是「照常判、但在详情里说明少看了一面」—— 前提是这句说明真的写出来，
    所以这条用例断言的就是那句话。
    """
    runs = [_run("u1", ct.DEFAULT_CASES[0], _turn("u1", start=0, end=1))]
    server = _MultiUserServer()
    _route_everything_through(monkeypatch, server)

    check = ct.check_no_server_errors(BASE_URL, runs, None)

    assert check.ok is True, "客户端侧没问题时不该因为读不到 /metrics 就判红"
    assert "未测量" in check.detail, f"详情里没说清楚少看了一面：{check.detail!r}"


def test_the_server_side_evidence_is_quoted_when_it_was_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ 读到了 ⇒ 详情里要写明「覆盖全部路由」，而不是只说「没有 5xx」。"""
    runs = [_run("u1", ct.DEFAULT_CASES[0], _turn("u1", start=0, end=1))]
    server = _MultiUserServer()
    _route_everything_through(monkeypatch, server)
    before = ct._read_5xx_counts(BASE_URL)

    check = ct.check_no_server_errors(BASE_URL, runs, before)

    assert check.ok is True
    assert "全部路由" in check.detail, f"没说明证据面覆盖到哪：{check.detail!r}"


# ==============================================================================
# ★ 判据 2b：回复是答复而非占位
# ==============================================================================
# 这条判据的由来是一次**真实的假绿灯**（2026-10-03）：三个用户全部
# ``finished_reason=completed``、其余判据全绿，而用户 1 拿到的全部回复只有
# 30 个字（含一个空格）：「已让政策问答智能体检索制度原文，稍等。 等待政策检索结果中。」
#
# ⚠️ 这一节的用例**必须包含一个正向对照**（真答复 + 带「稍等」字样 ⇒ 绿）。
# 只有负例的话，把判据改成「永远返回 False」也能让全部用例通过 ——
# 那正是「用测试证明自己」而不是「用测试约束实现」。


def test_a_placeholder_reply_is_not_green() -> None:
    """★★★ 30 个字的占位句 ⇒ 判失败（原文照抄自那次真实运行）。

    ⚠️ 这条用例的价值在于它**同时**满足判据 1、2、3：``completed``、
    非空、无错误事件。也就是说，本用例在 :func:`check_completed` /
    :func:`check_replies_nonempty` 下**必须是绿的** —— 所以下面把它们
    也断言了一遍。若哪天有人把这条判据删掉，那三条会全绿而这条会消失，
    报告上看起来「一切正常」。
    """
    placeholder = "已让政策问答智能体检索制度原文，稍等。 等待政策检索结果中。"
    run = _run("concurrency-user-1", ct.DEFAULT_CASES[0], _turn("u1", start=0, end=10, text=placeholder))

    # 正向对照：另外三条判据在这份观测上确实是绿的 —— 所以这条不是重复劳动。
    assert ct.check_completed([run]).ok is True
    assert ct.check_replies_nonempty([run]).ok is True
    assert ct.check_no_error_events([run]).ok is True

    check = ct.check_replies_are_answers([run])
    assert check.ok is False, "占位句被判成了通过 —— 这正是那次假绿灯的形态"
    assert "占位" in check.detail
    assert "稍等" in check.detail, "详情里没给出命中哪个措辞，排障时等于没说"


def test_an_english_inner_monologue_opening_is_not_green() -> None:
    """★★★ 回复以英文内心独白开头 ⇒ 判失败（原文照抄 2026-10-03 的 6 人运行）。

    ⚠️ 这个形状第一次出现（309 字 / 223 事件）时被写成了「本判据抓不到、
    要靠 eval.py」—— 于是它原样又发生了。反复出现说明它不是偶发，而是这条
    链路的**稳定行为**；对稳定行为，确定性判据永远比概率性的 LLM-as-judge 划算。
    （可复核的频次见 ``check_replies_are_answers`` 的文档；别在这里写死次数，
    历史上有过三个文件各写一个数、互相矛盾且无法复核的情况。）

    ⚠️ 它同样满足判据 1/2/3（completed、非空、无错误事件），所以下面一并断言
    —— 与 :func:`test_a_placeholder_reply_is_not_green` 同一个理由：
    本用例的价值全在「其余判据都是绿的」这一点上。
    """
    leaked = (
        "I'll look up your flight orders. 我查了一下，你名下目前没有机票订单记录"
        "（去广州的也没有）。"
    )
    run = _run("concurrency-user-2", ct.DEFAULT_CASES[1], _turn("u2", start=0, end=10, text=leaked))

    assert ct.check_completed([run]).ok is True
    assert ct.check_replies_nonempty([run]).ok is True
    assert ct.check_no_error_events([run]).ok is True

    check = ct.check_replies_are_answers([run])
    assert check.ok is False, "英文内心独白被判成了答复"
    assert "内心独白" in check.detail


def test_a_chinese_third_person_opening_is_not_green() -> None:
    """★★★ 回复以「用户问的是…」开头 ⇒ 判失败（同样照抄实测原文）。

    ⚠️ 这一条**不能**靠「回复里有『用户』两个字」实现 —— 那样会把
    「如果您想问…」这类正常行文也判红。判据只匹配**开头**，本用例与
    下一条 :func:`test_a_polite_sentence_containing_the_word_user_is_green`
    成对存在，一起把边界钉死。
    """
    leaked = (
        "用户问的是差旅标准——酒店住宿费能报多少。这是政策问答类问题。"
        "我先确认一下当前差标。 结论：**北京酒店单晚不超过 600 元**。"
    )
    run = _run("concurrency-user-3", ct.DEFAULT_CASES[2], _turn("u3", start=0, end=10, text=leaked))

    check = ct.check_replies_are_answers([run])
    assert check.ok is False, "中文第三人称独白被判成了答复"
    assert "内心独白" in check.detail


def test_a_polite_sentence_containing_the_word_user_is_green() -> None:
    """★★ 句中命中措辞表 ⇒ **必须绿**（探测器只匹配**开头**）。

    ⚠️ 这条是上一条的假阳性防线，而且它必须**真的命中措辞表**才算数 ——
    第一版用的是「如果用户有特殊情况」，句中只有「用户有」，
    ``_MONOLOGUE_PREFIXES`` 里根本没有这一条，于是把探测器改成全文搜索
    它也照样绿：**一条永远不会变红的防线等于没有防线**。
    换成「如果用户要改签」——「用户要」在表里，位置在句中，
    于是「只匹配开头」这个设计一旦被改坏，本用例立刻变红。

    ⚠️ 这个句子本身是**真实差旅场景里的正常写法**（改签规定），不是硬造的：
    误报面窄不等于零，所以边界必须钉死。
    """
    answer = "改签规定：如果用户要改签，需在起飞前 24 小时办理，手续费 200 元。"
    assert ct._monologue_reason(answer) is None, "前提不成立：这句里没有命中措辞表"
    assert "用户要" in answer, "前提不成立：句中必须真的含有一个表内措辞"
    run = _run("concurrency-user-3", ct.DEFAULT_CASES[2], _turn("u3", start=0, end=10, text=answer))

    check = ct.check_replies_are_answers([run])
    assert check.ok is True, f"正常行文里的『用户要』被误判成独白：{check.detail!r}"


def test_a_monologue_is_a_warning_not_a_failure() -> None:
    """★★★ 独白 ⇒ ``warn=True``、``ok=False``，且**不影响退出码**。

    ⚠️ 这条用例守的是一个**刻意的分级决定**，不是「想把测试弄绿」：

      · **占位回复**（用户什么都没拿到）是功能故障 ⇒ 判失败；
      · **独白开头**（用户拿到了答案，前面多一段模型草稿）是输出质量缺陷
        ⇒ 告警。

    把后者也判成失败，会让「并发正确性」的结论取决于模型的文风 ——
    而**一个会因为文风随机变红的判据，在真实项目里会被关掉**，
    连它旁边那个真正重要的「占位」判定也一起失效了。
    ⚠️ 别在这里写具体次数：2026-10-03 曾有「4 次」「第二次」等几个数字散落在
    三个文件里互相矛盾，而服务端日志根本不记录回复正文，谁也复核不了。
    可复核的频次在 ``check_replies_are_answers`` 的文档里（重跑可复现）。

    ⚠️ 分级**不等于**隐瞒：本用例同时断言详情里写明了这是告警、以及
    它为什么不算失败。把这个断言删掉，分级就退化成「把问题藏起来」。
    """
    leaked = "用户想查已有的机票订单，我直接查一下。 查无此单，你名下没有任何订单记录。"
    run = _run("concurrency-user-2", ct.DEFAULT_CASES[1], _turn("u2", start=0, end=1, text=leaked))

    check = ct.check_replies_are_answers([run])

    assert check.ok is False
    assert check.warn is True, "独白被升级成了硬失败 —— 退出码会随模型文风随机变红"
    assert "输出质量" in check.detail
    assert "LLM-as-judge" in check.detail, "没说通用判定在哪，读的人只会来改这个脚本"


def test_a_placeholder_is_a_hard_failure_not_a_warning() -> None:
    """★★★ 占位 ⇒ ``warn=False``（真失败），与上一条**必须**分开。

    ⚠️ 与上一条成对。少了它，把两种问题都标成 ``warn=True`` 也能让上一条通过
    —— 于是整个脚本永远不会失败，而「通过」两个字就彻底没意义了。
    """
    placeholder = "已让政策问答智能体检索制度原文，稍等。 等待政策检索结果中。"
    run = _run("concurrency-user-1", ct.DEFAULT_CASES[0], _turn("u1", start=0, end=1, text=placeholder))

    check = ct.check_replies_are_answers([run])

    assert check.ok is False
    assert check.warn is False, "占位（用户什么都没拿到）被降级成了告警"


def test_a_run_with_only_warnings_still_exits_zero(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """★★★ 只有告警时退出码是 0，但报告里**必须**写出「另有 N 条告警」。

    ⚠️ 这条是从用户视角写的：他要的是「并发测试通过」，而通不过的原因
    不该是模型的文风。但如果报告只留一句「全部通过」，那次告警就等于
    没发生过 —— 所以最后那行括号是承重的，不是装饰。
    """
    server = _MultiUserServer(
        stream_text="用户想查已有的机票订单，我直接查一下。 查无此单。"
    )
    code = _run_main(monkeypatch, server)
    printed = capsys.readouterr().out

    assert code == 0, f"只有告警时不该判失败：{printed[-300:]!r}"
    assert "⚠️ 回复是答复而非占位（告警，不计入失败）" in printed, "告警没打出来"
    assert "另有 1 条告警" in printed, "通过的那行里没提这次告警"


def test_a_real_answer_that_happens_to_say_please_wait_is_green() -> None:
    """★★ 真答复里顺带说句「稍等」⇒ **必须绿**（这条划的是判据的边界）。

    ⚠️ 这是「含数字」那个附加条件存在的**唯一理由**。没有它，本用例会红，
    而一个会红的判据在真实项目里的下场是被人关掉 —— 然后连真占位也抓不到了。
    三条默认问句的任何真实答复都带数字（金额 / 日期 / 单号），
    所以「有措辞但没数字」才可疑。
    """
    answer = "正在查询你的订单…已找到：2026-09-12 广州，CZ3101，状态已出票，票价 1850 元。"
    run = _run("concurrency-user-2", ct.DEFAULT_CASES[1], _turn("u2", start=0, end=10, text=answer))

    check = ct.check_replies_are_answers([run])
    assert check.ok is True, f"真答复被误判成占位：{check.detail!r}"


def test_a_plain_answer_without_waiting_words_is_green() -> None:
    """★ 干净的答复（无占位措辞）⇒ 绿，且详情里报出被检查的轮次数。"""
    run = _run("concurrency-user-3", ct.DEFAULT_CASES[2], _turn("u3", start=0, end=10, text="住宿费每晚上限 600 元。"))

    check = ct.check_replies_are_answers([run])
    assert check.ok is True
    assert "1 轮" in check.detail, f"详情里没有证据量（检查了几轮）：{check.detail!r}"


def test_one_placeholder_among_many_turns_fails_the_whole_check() -> None:
    """★★ 多轮里**只要有一轮**是占位 ⇒ 整条判失败，且详情指名是哪一轮。

    ⚠️ 这条防的是「聚合掩盖」：若判据写成「统计占位比例、低于阈值就放过」，
    那么 ``--users 6 --rounds 2`` 会把一次真实故障稀释成一次通过。
    ⚠️ 也断言详情里带用户标签与轮次号 —— 只说「有一轮是占位」的详情，
    在 12 条回复里没法定位。
    """
    good = _run("concurrency-user-1", ct.DEFAULT_CASES[0], _turn("u1", start=0, end=10, text="住宿费每晚上限 600 元。"))
    bad_turn = _turn("u2", start=0, end=10, text="已经交给政策问答智能体了，请稍候。")
    bad_turn.round_index = 2
    bad = _run("concurrency-user-2", ct.DEFAULT_CASES[1], bad_turn)

    check = ct.check_replies_are_answers([good, bad])
    assert check.ok is False
    assert "concurrency-user-2" in check.detail, f"没点名是哪个用户：{check.detail!r}"
    assert "第 2 轮" in check.detail, f"没点名是第几轮：{check.detail!r}"


# ==============================================================================
# CLI 与常量
# ==============================================================================


def test_single_user_is_rejected() -> None:
    """``--users 1`` 必须被拒 —— 一个用户谈不上并发。"""
    with pytest.raises(SystemExit):
        ct._build_parser().parse_args(["--users", "1"])


def test_every_default_is_a_value_the_parser_accepts() -> None:
    """**每个默认值都必须是解析器自己肯接受的值。**

    这条听起来像废话，防的却是一个真实发生过的坑：``--rounds`` 曾与
    ``--users`` 共用同一个校验器（下界写死 2），而 ``DEFAULT_ROUNDS`` 是 1
    —— 于是「不写参数」能跑，「照着 --help 把默认值显式写一遍」报错。
    参数默认值不受 ``type`` 校验，所以这个矛盾**只有显式传参时才暴露**，
    平时跑 CI 永远看不见。
    """
    parser = ct._build_parser()
    checked = 0
    for action in parser._actions:  # noqa: SLF001 —— argparse 没有公开的遍历入口
        if action.type is None or action.default is None:
            continue
        if isinstance(action.default, bool):
            continue
        checked += 1
        flag = action.option_strings[0]
        parsed = parser.parse_args([flag, str(action.default)])
        assert getattr(parsed, action.dest) == action.default, (
            f"{flag} 的默认值 {action.default!r} 被它自己的 type 拒绝了"
        )
    assert checked >= 2, f"至少要查到 --users 与 --rounds 两个，实际 {checked} 个"


def test_one_round_is_accepted_because_it_is_the_default() -> None:
    """``--rounds 1`` 必须被接受 —— 它就是默认值，且一轮完全说得通。

    ⚠️ 下界 2 对 ``--users`` 成立（1 个用户谈不上并发），对 ``--rounds``
    不成立：一轮就是「三个用户同时发一次」，正是默认要测的形态。
    把这条业务约束套到轮数上，是纯粹的参数串味。
    """
    assert ct.DEFAULT_ROUNDS == 1
    assert ct._build_parser().parse_args(["--rounds", "1"]).rounds == 1
    assert ct._build_parser().parse_args([]).rounds == ct.DEFAULT_ROUNDS


def test_users_floor_error_blames_users_not_rounds() -> None:
    """``--users 1`` 的报错必须点名 ``--users``。

    原实现两个参数共用一句话，于是校验 ``--rounds`` 失败时提示的是
    「1 个用户谈不上并发」—— 把读的人引向另一个参数。报错指错地方，
    比没有报错更费时间。
    """
    with pytest.raises(argparse.ArgumentTypeError) as excinfo:
        ct._at_least_two_users("1")
    assert "--users" in str(excinfo.value)
    # 而轮数校验器不该有下界 2 这回事：
    assert ct._positive_int("1") == 1


def test_default_is_exactly_three_distinct_prompts() -> None:
    """默认就是任务书要的形态：3 个用户、3 条**互不相同**的问句。"""
    runs = ct._build_runs(3, 1)
    assert len(runs) == 3
    prompts = [run.case.prompt for run in runs]
    assert len(set(prompts)) == 3, f"三条问句必须互不相同：{prompts}"
    assert len({run.user_id for run in runs}) == 3, "三个不同的用户身份"
    assert {run.case.route for run in runs} and len({run.case.route for run in runs}) == 3


def test_more_users_than_cases_cycles_but_stays_distinct() -> None:
    """用户数超过场景数时轮转复用，但前三个仍然互不相同。"""
    runs = ct._build_runs(6, 1)
    assert len(runs) == 6
    assert len({run.case.prompt for run in runs}) == 3
