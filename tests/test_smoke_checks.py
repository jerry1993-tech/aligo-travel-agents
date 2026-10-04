# -*- coding: utf-8 -*-
"""``scripts/smoke.py`` 里两个业务检查的**判定表**测试。

==============================================================================
为什么冒烟脚本里的检查也要测
==============================================================================
    冒烟测试是交付闸门 —— ``make smoke`` 绿是「这套东西能跑」的对外结论。
    而一个判定写错的检查项比没有检查项更糟：它会持续输出一个**错误的绿灯**。

    本项目就有一处：``check_travel_api`` 原先是

        if response.status_code == 404:
            return Check(..., True, "尚未实现（P2 交付）")
        return Check(..., True, f"{response.status_code}")   # ← 任何状态码都通过

    也就是说 **500 也算通过**。它唯一会报错的场景恰恰是被豁免的那一种，
    其余一律放行。这样的检查在报告里长得和真检查一模一样。

    P2 交付了业务路由之后，404 豁免已经删除。本文件就是那段判定表的固化，
    重点是**每一条分支都要有反例** —— 只测「200 通过、404 失败」的话，
    一个「除了 200 全失败」的实现同样能全绿，而那会把正常的
    「服务刚起来还没就绪」误报成故障。

==============================================================================
为什么用 ``httpx.MockTransport`` 而不是起真服务
==============================================================================
    这里要验的是**分支判定**，不是集成。要构造「返回 200 但响应体不是 JSON」
    「返回 401 但没有 WWW-Authenticate」这类畸形响应，真服务做不到 ——
    得先把服务改成会犯错的版本。``MockTransport`` 让每一种响应都随手可得，
    用例因此覆盖得全，而且毫秒级。

    真正的端到端验证在别处：``tests/test_e2e_stream.py`` 起真 uvicorn，
    交付前的手工验证覆盖真实部署。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from scripts import smoke
from scripts.smoke import (
    Check,
    check_conversation,
    check_identity_endpoint,
    check_travel_api,
)

#: 任意 base url —— MockTransport 不解析它，但检查函数会拼进 URL。
BASE_URL = "http://smoke.test"


def _client(handler: Any) -> httpx.Client:
    """构造一个由回调决定响应的客户端。

    Args:
        handler (`Any`): ``(request) -> httpx.Response`` 形式的回调。

    Returns:
        `httpx.Client`: 调用方负责关闭。
    """
    return httpx.Client(transport=httpx.MockTransport(handler))


def _route(routes: dict[str, httpx.Response]) -> Any:
    """按**路径**分派响应。

    比按调用顺序返回的写法更抗改动：往检查里插一次额外请求不会让
    后续断言全部错位（那种失败看起来像「检查逻辑坏了」，实际只是
    测试桩的序号对不上了）。

    Args:
        routes (`dict[str, httpx.Response]`): 路径 → 响应。

    Returns:
        `Any`: 可传给 ``MockTransport`` 的回调。
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        response = routes.get(request.url.path)
        if response is None:  # pragma: no cover - 用例写错路径时才会走到
            return httpx.Response(404, json={"detail": f"未桩化的路径 {request.url.path}"})
        return response

    return _handler


# ==============================================================================
# 一、业务接口检查
# ==============================================================================
def test_travel_api_passes_on_a_real_business_response() -> None:
    """200 且响应体 ``status == "ok"`` ⇒ 通过。"""
    handler = _route(
        {"/api/v1/health": httpx.Response(200, json={"status": "ok", "scope": "api/v1"})},
    )

    with _client(handler) as client:
        result = check_travel_api(client, BASE_URL)

    assert isinstance(result, Check)
    assert result.ok is True


@pytest.mark.parametrize(
    ("status", "label"),
    [
        (404, "路由没注册"),
        (500, "服务端异常"),
        (503, "依赖不可用"),
    ],
)
def test_travel_api_fails_on_any_non_200(status: int, label: str) -> None:
    """★ 除 200 外的状态码一律**失败**。

    ⚠️ 这几条钉的就是原实现的那个洞：它只对 404 敏感，其余一律返回通过 ——
        包括 500。于是「业务接口挂了」这件事冒烟测试永远发现不了。

    ⚠️ 404 单独再测一次（见下一条）是因为它有专门的提示文案；
        这里把 500 / 503 一起参数化，重点在「非 200 即失败」这条性质。
    """
    handler = _route({"/api/v1/health": httpx.Response(status, json={"detail": label})})

    with _client(handler) as client:
        result = check_travel_api(client, BASE_URL)

    assert result.ok is False, f"{status} 被判成通过了 —— 这正是原实现的缺陷。"
    assert str(status) in result.detail


def test_travel_api_404_mentions_the_likely_cause() -> None:
    """404 的失败详情要指向**最可能的原因**，而不是只说「404」。

    404 在本项目里几乎只有一个成因：``include_router`` 没生效，
    或某个子路由用了绝对路径把 ``/api/v1`` 前缀覆盖掉了。
    把这个推断写进详情，省掉一次「从 404 开始查」的过程。
    """
    handler = _route({"/api/v1/health": httpx.Response(404, json={"detail": "Not Found"})})

    with _client(handler) as client:
        result = check_travel_api(client, BASE_URL)

    assert result.ok is False
    assert "include_router" in result.detail


def test_travel_api_fails_when_200_body_is_not_json() -> None:
    """200 但响应体不是 JSON ⇒ 失败。

    中间设备（网关、错误页、登录重定向）会返回 200 + HTML。
    只看状态码的实现会把它当成业务接口正常。
    """
    handler = _route(
        {"/api/v1/health": httpx.Response(200, text="<html>登录页</html>")},
    )

    with _client(handler) as client:
        result = check_travel_api(client, BASE_URL)

    assert result.ok is False
    assert "不是 JSON" in result.detail


def test_travel_api_fails_when_status_field_is_not_ok() -> None:
    """200 且是 JSON，但 ``status`` 不是 ``ok`` ⇒ 失败。

    这条守的是「接口被换成了别的实现」：形状对、语义不对。
    """
    handler = _route({"/api/v1/health": httpx.Response(200, json={"status": "degraded"})})

    with _client(handler) as client:
        result = check_travel_api(client, BASE_URL)

    assert result.ok is False

    result_detail = result.detail
    assert "degraded" in result_detail


def test_travel_api_reports_connection_errors_as_failure() -> None:
    """连不上 ⇒ 失败，且详情里有可执行的下一步。

    ⚠️ 用 ``httpx.ConnectError`` 而不是随便一个异常：这条分支的
        实现是 ``except Exception``，但真正会发生的只有连接类异常。
        用不可能的异常来测会让用例看起来在测一个不存在的场景。
    """

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with _client(_handler) as client:
        result = check_travel_api(client, BASE_URL)

    assert result.ok is False


# ==============================================================================
# 二、鉴权检查
# ==============================================================================
def _unauthorized(*, with_header: bool = True) -> httpx.Response:
    """构造一个 401 响应。

    Args:
        with_header (`bool`): 是否带上 ``WWW-Authenticate``。

    Returns:
        `httpx.Response`: 401 响应。
    """
    headers = {"WWW-Authenticate": 'Bearer realm="aligo"'} if with_header else {}
    return httpx.Response(401, json={"detail": "缺少 X-User-ID 请求头。"}, headers=headers)


def test_identity_check_passes_on_401_with_www_authenticate() -> None:
    """401 且带 ``WWW-Authenticate`` ⇒ 通过（鉴权链真的在跑）。"""
    handler = _route({"/api/v1/me": _unauthorized()})

    with _client(handler) as client:
        result = check_identity_endpoint(client, BASE_URL)

    assert result.ok is True


def test_identity_check_fails_on_401_without_www_authenticate() -> None:
    """401 但缺 ``WWW-Authenticate`` ⇒ 失败（RFC 6750 §3）。

    ⚠️ 为什么值得单独一条：这个头很容易在重构 ``send_json`` 时被顺手丢掉，
        而**功能上完全看不出来** —— 反正都是 401。丢掉的后果是某些客户端
        会退化成「一直重试同样的请求」，而不是去取新凭据。
    """
    handler = _route({"/api/v1/me": _unauthorized(with_header=False)})

    with _client(handler) as client:
        result = check_identity_endpoint(client, BASE_URL)

    assert result.ok is False
    assert "WWW-Authenticate" in result.detail


def test_identity_check_fails_on_422_with_a_pointed_diagnosis() -> None:
    """★ 422 是最有信息量的一种失败：它说明**鉴权中间件没生效**。

    422 是框架的 ``Header(...)`` 依赖给出的 —— 意味着请求已经**穿过**
    ``AuthMiddleware`` 抵达了路由。这正是「中间件漏装 / 顺序写反」
    的现场症状。

    ⚠️ 所以这条检查比「返回了非 200 就算通过」有价值得多：
        后者会把「中间件没生效」判成通过。
    """
    handler = _route(
        {"/api/v1/me": httpx.Response(422, json={"detail": "Field required"})},
    )

    with _client(handler) as client:
        result = check_identity_endpoint(client, BASE_URL)

    assert result.ok is False
    assert "422" in result.detail
    assert "AuthMiddleware" in result.detail


def test_identity_check_fails_on_404() -> None:
    """404 ⇒ 失败，而不是「反正不是 200 就算鉴权生效」。

    ⚠️ 这条防的是最阴的一种假绿：路由没注册时它返回 404，
        而一个「只要不是 200 就通过」的实现会把 404 判成「鉴权生效」——
        那条检查于是变成了一个**永远绿灯的摆设**。
    """
    handler = _route({"/api/v1/me": httpx.Response(404, json={"detail": "Not Found"})})

    with _client(handler) as client:
        result = check_identity_endpoint(client, BASE_URL)

    assert result.ok is False
    assert "404" in result.detail


def test_identity_check_fails_when_endpoint_is_open() -> None:
    """200 ⇒ 失败：无凭据就能读到身份，鉴权形同虚设。

    这是本检查存在的**根本理由** —— 一个把 ``/api/v1/me`` 做成公开端点的
    回归，必须被冒烟测试拦住。
    """
    handler = _route(
        {"/api/v1/me": httpx.Response(200, json={"user_id": "anonymous"})},
    )

    with _client(handler) as client:
        result = check_identity_endpoint(client, BASE_URL)

    assert result.ok is False
    assert str(200) in result.detail


def test_identity_check_reports_connection_errors_as_failure() -> None:
    """连不上 ⇒ 失败。"""

    def _handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    with _client(_handler) as client:
        result = check_identity_endpoint(client, BASE_URL)

    assert result.ok is False


# ==============================================================================
# 三、每一个检查都必须被真的挂进冒烟流程
# ==============================================================================
def test_every_check_is_wired_into_run_smoke() -> None:
    """★ 光实现检查还不够 —— 它必须真的被 ``run_smoke`` 调用。

    ⚠️ 这条守的是一类非常安静的回归：函数写好了、用例也全绿，
        却没有出现在 ``run_smoke`` 的清单里。于是冒烟报告里**根本没有这一项**，
        而「报告里没有」与「检查通过了」在终端上一模一样 —— 都是一片绿。

    实现上直接读 ``run_smoke`` 的源码：它是一段线性的装配代码，
    没有可以断言的数据结构。用 ``inspect.getsource`` 读它比引入
    「注册表」这样一层为了可测而存在的抽象要诚实。

    ⚠️ ``inspect.getsource`` 依赖行号表：**改了 smoke.py 之后要紧跟一次
    重新运行**（同一个进程里先 import 再改文件，会读到「旧行号 + 新正文」，
    报出来的失败信息指向一个毫不相干的函数）。这不是本用例的缺陷，
    而是它必然的代价 —— 换成「注册表」那种抽象只是把问题挪个地方。
    """
    import inspect

    from scripts import smoke

    source = inspect.getsource(smoke.run_smoke)

    # ⚠️ 这是一个**恰好等于**的清单：新增一个检查就必须同步加到这里。
    # 少了任何一项，那个检查就成了「写了但从不运行」的死代码。
    for name in (
        "check_travel_api",
        "check_identity_endpoint",
        "check_conversation",
    ):
        # ⚠️ 只匹配到左括号为止的前缀，**不要求**闭括号紧跟其后：
        #    检查一旦多出一个关键字参数（比如 `keep_artifacts=`），
        #    整串字面量就对不上了 —— 那时这条用例会报「没被调用」，
        #    而它其实调用了，只是签名变了。假失败比不检查更糟：
        #    它会教人直接改断言，而下一次真漏掉一个检查时也一样被改掉。
        assert f"{name}(client, base_url" in source, (
            f"{name} 没有被 run_smoke 调用 —— 它实现了，但冒烟报告里不会有它。"
        )


# ==============================================================================
# 三、本轮产物的清理（「跑完自己删掉」）
# ==============================================================================
#: 一轮**成功**对话的 SSE 事件流：一段文本 + 一个正常结束标记。
def _sse(*events: dict[str, Any]) -> bytes:
    """把事件字典渲染成 SSE 报文。"""
    return "".join("data: " + json.dumps(e) + "\n\n" for e in events).encode()


def _completed_stream() -> bytes:
    """一段能让对话检查判为「通过」的流。"""
    return _sse(
        {"type": "TEXT_BLOCK_DELTA", "delta": "好的"},
        {"type": "REPLY_END", "finished_reason": "completed"},
    )


class _ConversationServer:
    """够让 :func:`check_conversation` 跑完一轮的假服务。

    ⚠️ 与「按路径返回固定响应」的桩不同，这里的**状态是真的会变的**：
    删除必须真的把记录从列表里拿掉。否则回读核验永远失败，
    而那正是本组用例要测的东西 —— 一个不会变的桩会让「删干净了」
    与「根本没删」在断言上长得一模一样。

    ⚠️ ``GET /sessions/`` 按 ``agent_id`` 过滤，且智能体不可见时返回 404 ——
    这两条都是照框架的路由抄的（``_session.py`` 的 ``list_sessions`` 会先
    ``access.resolve_agent``）。抄它们是为了让「先删智能体、再删会话」
    这个顺序错误**在测试里就能暴露**：先删智能体的话，会话的回读会 404，
    于是清理报告会带上一条 ⚠️。

    Attributes:
        agents (`list[str]`): 现存智能体 id。
        sessions (`list[tuple[str, str]]`): 现存会话 ``(session_id, agent_id)``。
        calls (`list[tuple[str, str]]`): 按发生顺序记录的 ``(method, path)``。
        deletes (`list[tuple[str, dict[str, str]]]`): DELETE 的 ``(path, params)``。
    """

    def __init__(
        self,
        *,
        chat_status: int = 500,
        delete_status: int = 204,
        stream_body: bytes | None = None,
    ) -> None:
        self.agents: list[str] = []
        self.sessions: list[tuple[str, str]] = []
        self.calls: list[tuple[str, str]] = []
        self.deletes: list[tuple[str, dict[str, str]]] = []
        self.chat_status = chat_status
        self.delete_status = delete_status
        self.stream_body = stream_body
        self._agent_seq = 0
        self._session_seq = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        """``httpx.MockTransport`` 的处理函数。"""
        path, method = request.url.path, request.method
        self.calls.append((method, path))

        if method == "DELETE":
            self.deletes.append((path, dict(request.url.params)))
            if self.delete_status not in (200, 204, 404):
                return httpx.Response(self.delete_status, json={"detail": "删不掉"})
            target = path.rsplit("/", 1)[-1]
            if path.startswith("/sessions/"):
                self.sessions = [s for s in self.sessions if s[0] != target]
            elif path.startswith("/agent/"):
                self.agents = [a for a in self.agents if a != target]
            return httpx.Response(204)

        if path == "/api/v1/default-model":
            return httpx.Response(
                200,
                json={
                    "chat_model_config": {"type": "openai_chat", "model": "qwen"},
                    "mode": "configured",
                },
            )

        if path == "/agent/" and method == "GET":
            return httpx.Response(
                200,
                json={"agents": [{"id": a} for a in self.agents], "total": len(self.agents)},
            )
        if path == "/agent/" and method == "POST":
            self._agent_seq += 1
            new_id = f"agent-{self._agent_seq}"
            self.agents.append(new_id)
            return httpx.Response(201, json={"agent_id": new_id})

        if path == "/sessions/" and method == "GET":
            agent_id = request.url.params.get("agent_id")
            if agent_id not in self.agents:
                # 照抄框架：智能体不可见 ⇒ 404（先删智能体就会走到这里）。
                return httpx.Response(404, json={"detail": f"智能体 {agent_id} 不存在"})
            rows = [s for s in self.sessions if s[1] == agent_id]
            return httpx.Response(
                200,
                json={
                    "sessions": [{"session": {"id": s[0]}, "status": "idle"} for s in rows],
                    "total": len(rows),
                },
            )
        if path == "/sessions/" and method == "POST":
            self._session_seq += 1
            new_id = f"session-{self._session_seq}"
            self.sessions.append((new_id, json.loads(request.content)["agent_id"]))
            return httpx.Response(201, json={"session_id": new_id})

        if path.endswith("/stream"):
            if self.stream_body is None:
                return httpx.Response(404, json={"detail": "本用例不提供流"})
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=self.stream_body,
            )

        if path == "/chat/" and method == "POST":
            return httpx.Response(self.chat_status, json={"detail": "模型调用失败"})

        return httpx.Response(404, json={"detail": f"未桩化的路径 {method} {path}"})


def _quiet(monkeypatch: pytest.MonkeyPatch) -> None:
    """把检查里的等待全部变成瞬时。

    ⚠️ 补丁打在 ``smoke.time`` 上，也就是**真的** ``time`` 模块 ——
    只影响本用例（monkeypatch 会还原），且这里没有别的线程在睡。
    """
    monkeypatch.setattr(smoke.time, "sleep", lambda *_: None)


def _route_stream_through(
    monkeypatch: pytest.MonkeyPatch, server: _ConversationServer
) -> None:
    """让检查**内部另建**的那个流式客户端也走假服务。

    ⚠️ 这一步不是可有可无的：``_read_stream`` 刻意用了**自己的**
    ``httpx.Client``（独立的读超时，见该函数的说明），它不会继承外面
    注入的客户端。不拦它的话，用例会真的去连 ``http://smoke.test`` ——
    于是要么慢到空等满超时，要么把「连不上」当成检查的结论，
    而真正要断言的那条分支根本没走到。
    """
    real_client = httpx.Client

    def _factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs.setdefault("transport", httpx.MockTransport(server.handler))
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _factory)


def test_failed_conversation_still_cleans_up_its_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ 对话失败的那一轮**同样**要清理 —— 而且它留下的垃圾最多。

    这里让 POST /chat/ 返回 500，走的正是「智能体与会话都建好了、
    在对话阶段提前 return」的路径：修复前，每这样失败一次就多一个残留。
    """
    server = _ConversationServer(chat_status=500)
    _quiet(monkeypatch)

    with _client(server.handler) as client:
        result = check_conversation(client, BASE_URL)

    assert result.ok is False  # 对话确实失败了
    assert server.agents == [], "智能体没有被删掉"
    assert server.sessions == [], "会话没有被删掉"
    assert "已清理本轮产物" in result.detail


def test_session_is_deleted_before_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """★ 先删会话、后删智能体 —— 顺序反了会留下孤儿会话。

    ⚠️ 这不是「看起来更整齐」：会话的删除接口把 ``agent_id`` 声明为
    **必需**查询参数。智能体先没了，那条 DELETE 要么缺参数、要么 404，
    于是会话永远删不掉，而报告上「删过了」。
    假服务按框架的行为实现了这一点（智能体不在 ⇒ 会话列表 404），
    所以顺序一旦写反，下面关于「干净」的断言就会失败。
    """
    server = _ConversationServer(chat_status=500)
    _quiet(monkeypatch)

    with _client(server.handler) as client:
        check_conversation(client, BASE_URL)

    order = [path for method, path in server.calls if method == "DELETE"]
    assert order == ["/sessions/session-1", "/agent/agent-1"], order
    # 并且会话那一条必须带上 agent_id（少了它框架会 422）。
    session_params = server.deletes[0][1]
    assert session_params.get("agent_id") == "agent-1"


def test_cleanup_failure_does_not_flip_the_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ 对话正常、只是没删干净 ⇒ 该项**仍然通过**，问题写在详情里。

    ⚠️ 这条钉的是「两个问题不许合并成一个判据」。把 DELETE 的成败并进
    「能不能对话」，会让一次完全正常的部署变红，把人引去查对话链路 ——
    方向完全错了。但也不能不吭声：详情里必须有 ⚠️ 和出问题的 id。
    """
    server = _ConversationServer(
        chat_status=200, delete_status=500, stream_body=_completed_stream()
    )
    _quiet(monkeypatch)
    _route_stream_through(monkeypatch, server)

    with _client(server.handler) as client:
        result = check_conversation(client, BASE_URL)

    assert result.ok is True, result.detail  # 对话本身是好的
    assert "未清理干净" in result.detail
    assert "agent-1" in result.detail
    assert server.agents == ["agent-1"], "假服务不该真删 —— 它模拟的是「删不掉」"


def test_cleanup_problems_are_reported_not_swallowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ 连不上服务时，清理也必须**报出问题**，不许静静当作删干净了。

    ⚠️ 这是清理逻辑最容易走向的假绿灯：``except: pass``。它会让
    「删干净了」与「压根没连上」在报告上完全一样 —— 于是残留会一直攒，
    而每一轮冒烟都是绿的。
    """

    def _boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom", request=request)

    with _client(_boom) as client:
        problems = smoke._cleanup_artifacts(
            client, BASE_URL, {"agent_id": "a", "session_id": "s"}, keep=False
        )

    assert problems, "连不上时应当报出问题，而不是当作已清理"
    # ⚠️ 断言里必须点到「删除」这个动作本身，不能只说「有问题就行」：
    #    回读那一步也会因为连不上而报错，于是哪怕 DELETE 的异常被
    #    `except: return` 吞掉，`problems` 依旧是满的 —— 断言照样通过。
    #    （这不是假设：本用例的第一版就是这么写的，变异测试当场没抓到。）
    assert any("删除" in p for p in problems), problems


def test_keep_artifacts_skips_the_deletion(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--keep-artifacts`` 时一个字节都不删，并把 id 写给运维。

    ⚠️ 这条同时守着一个反向风险：清理逻辑**不能**在排查场景下也照删不误 ——
    那会把运维正要看的现场毁掉。
    """
    server = _ConversationServer(chat_status=200, stream_body=_completed_stream())
    _quiet(monkeypatch)
    _route_stream_through(monkeypatch, server)

    with _client(server.handler) as client:
        result = check_conversation(client, BASE_URL, keep_artifacts=True)

    assert result.ok is True, result.detail

    assert [c for c in server.calls if c[0] == "DELETE"] == []
    assert server.agents == ["agent-1"]
    assert server.sessions == [("session-1", "agent-1")]
    assert "保留" in result.detail


def test_a_dead_stream_fails_fast_instead_of_waiting_out_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ SSE 一失败就**立刻**报出来，不空等满 60 秒。

    ⚠️ 这条守的是一个真实的浪费（也是写这组用例时踩出来的）：
    读取线程建的是**自己的**客户端，它出错返回后原先不通知主线程，
    于是「连接被拒」这种瞬时失败也要耗满 ``CONVERSATION_TIMEOUT_SECONDS``
    才报出来 —— 一次确定无疑的失败，却拖一分钟才肯说。

    用**耗时**来钉它，而不是只看结论：结论在修复前后都是「失败」，
    只有耗时能区分「立刻知道」与「等满上限才知道」。
    """
    # 没有 stream_body ⇒ 假服务对 /stream 返回 404。
    server = _ConversationServer(chat_status=200)
    _quiet(monkeypatch)
    _route_stream_through(monkeypatch, server)

    clock = smoke.time.monotonic
    started = clock()

    with _client(server.handler) as client:
        result = check_conversation(client, BASE_URL)

    elapsed = clock() - started
    assert result.ok is False
    assert "SSE 读取失败" in result.detail
    assert elapsed < smoke.CONVERSATION_TIMEOUT_SECONDS / 6, (
        f"用了 {elapsed:.1f} 秒才报出 SSE 失败 —— 说明主线程仍在空等上限，"
        "而不是在读取线程收工的那一刻返回。"
    )


def test_nothing_created_means_no_cleanup_note_claiming_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """什么都没建就失败时，详情里**不许**出现「已清理」。

    ⚠️ 那是一句假话。而假话会让「绿色行上有没有这句话」这个信号作废 ——
    运维会开始怀疑这句话本身，而不是去查清理为什么没发生。
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"chat_model_config": None, "mode": "missing", "hint": "没有任何凭据"},
        )

    with _client(handler) as client:
        result = check_conversation(client, BASE_URL)

    assert result.ok is False
    assert "已清理" not in result.detail


def test_cli_threads_keep_artifacts_into_run_smoke(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ ``--keep-artifacts`` 必须真的传进 ``run_smoke``。

    ⚠️ 只测「解析器认得这个参数」是不够的：参数解析对了、却忘了往下传，
    是一个没有任何症状的缺陷 —— 运维以为留了现场，现场其实已经被删了。
    """
    captured: dict[str, Any] = {}

    def _fake_run_smoke(base_url: str, *, keep_artifacts: bool = False) -> Any:
        captured["base_url"] = base_url
        captured["keep_artifacts"] = keep_artifacts
        return smoke.Report()

    monkeypatch.setattr(smoke, "run_smoke", _fake_run_smoke)

    assert smoke.main([]) == 0
    assert captured["keep_artifacts"] is False, "默认必须是清理"

    assert smoke.main(["--keep-artifacts"]) == 0
    assert captured["keep_artifacts"] is True
