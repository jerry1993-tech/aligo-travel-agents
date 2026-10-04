# -*- coding: utf-8 -*-
"""预置智能体脚本（``scripts/provision_agent.py``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    这个脚本存在的唯一理由是消掉一个**实测复现过**的死路：

        全新部署 → 打开浏览器 → 输入框是灰的、点不动，
        页面上只有「请先选择一个智能体。」，而下拉框里一个都没有。

    它的失败方式全都是**静默**的 —— 脚本报「✅ 成功」，浏览器里照样打不了字：

      · 名字写错 —— 智能体建出来了、也能聊天，但快慢车道与动态 Prompt
        按 `agent.name` 收窄，于是这些能力**静默不生效**（见脚本模块文档）。
        所以 :func:`test_agent_name_comes_from_the_registry` 钉的是
        「名字取自常量」，而不是「名字等于某个字面量」。
      · 会话查询没带 `agent_id` —— 会把**别的**智能体名下的会话算成
        「已有会话」，于是预置会话被跳过，用户看到的仍然是灰输入框。
        这条尤其阴：单用户、单智能体时它**永远看不出问题**。
      · 没做回读核验 —— 创建返回 201 但下一跳读不到（多副本写入可见性），
        脚本报成功而浏览器是空的。

    真连服务的验证在 ``make provision-agent`` 与交付前的手工浏览器验证里；
    单测不该依赖一个跑着的服务，所以这里用 ``httpx.MockTransport``。
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from scripts import provision_agent
from src.agents.prompts import MAIN_AGENT_NAME, prompt_for

#: 任意 base url —— MockTransport 不解析它，但脚本会拼进 URL。
BASE_URL = "http://provision.test"

#: 测试用的假用户。
USER = "unit-user"


# ==============================================================================
# 测试替身：一个「记得住自己被怎么用」的假服务
# ==============================================================================


class _FakeServer:
    """按路径与调用顺序作答的假服务，同时记录收到的请求。

    比静态路由强的一点：**创建之后列表要跟着变**，否则回读核验那一步
    永远失败 —— 而回读核验正是本脚本要测的东西之一。
    """

    def __init__(self, *, agents: list[dict[str, Any]] | None = None,
                 sessions: list[dict[str, Any]] | None = None) -> None:
        self.agents = list(agents or [])
        self.sessions = list(sessions or [])
        self.requests: list[httpx.Request] = []
        self.bodies: list[dict[str, Any]] = []
        self._next = 0

    def _make_agent(self, name: str, prompt: str) -> dict[str, Any]:
        self._next += 1
        return {
            "id": f"agent-{self._next}",
            "data": {"name": name, "system_prompt": prompt},
        }

    def client(self) -> httpx.Client:
        """构造一个走本假服务的客户端。"""
        return httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        import json

        self.requests.append(request)
        body = json.loads(request.content) if request.content else None
        if body is not None:
            self.bodies.append(body)

        path, method = request.url.path, request.method

        if path == "/agent/" and method == "GET":
            return httpx.Response(200, json={"agents": self.agents, "total": len(self.agents)})

        if path == "/agent/" and method == "POST":
            agent = self._make_agent(body["name"], body["system_prompt"])
            self.agents.append(agent)
            return httpx.Response(201, json={"agent_id": agent["id"]})

        if path == "/sessions/" and method == "GET":
            # ⚠️ 真实服务按 `agent_id` 过滤；这里**照做** —— 若测试替身
            # 忽略这个参数，那条「忘了带 agent_id」的缺陷就测不出来了。
            agent_id = request.url.params.get("agent_id")
            rows = [s for s in self.sessions if s["agent_id"] == agent_id]
            return httpx.Response(200, json={"sessions": rows, "total": len(rows)})

        if path == "/sessions/" and method == "POST":
            self._next += 1
            session = {"session_id": f"session-{self._next}", "agent_id": body["agent_id"]}
            self.sessions.append(session)
            return httpx.Response(201, json={"session_id": session["session_id"]})

        return httpx.Response(404, json={"detail": f"未桩化的路径 {method} {path}"})


# ==============================================================================
# 主路径
# ==============================================================================


def test_creates_the_agent_when_the_user_has_none() -> None:
    """用户一个智能体都没有时，建一个 —— 这是「输入框解禁」的第一块拼图。"""
    server = _FakeServer()
    with server.client() as client:
        result = provision_agent.provision(BASE_URL, USER, client=client)

    assert result.agent_created is True
    assert result.agent_id == "agent-1"
    posted = [b for b in server.bodies if b.get("name") == MAIN_AGENT_NAME]
    assert posted, "没有发出建智能体的请求"


def test_agent_name_comes_from_the_registry() -> None:
    """名字取自 `main_plan` 常量，且提示词取自**同一份**产品提示词。

    ⚠️ 这条钉的是「不许在脚本里重抄一遍名字或提示词」。重抄的后果是
    两边各自演化：改了 `src/agents/prompts.py`，预置出来的智能体还是旧的。
    也正因为名字来自常量，同名的快慢车道/动态 Prompt 才生效。
    """
    server = _FakeServer()
    with server.client() as client:
        provision_agent.provision(BASE_URL, USER, client=client)

    assert MAIN_AGENT_NAME == "main_plan"  # 常量本身没被改坏
    created = server.bodies[0]
    assert created["name"] == MAIN_AGENT_NAME
    assert created["system_prompt"] == prompt_for(MAIN_AGENT_NAME)


def test_default_user_matches_the_demo_identity_of_seed_data() -> None:
    """CLI 的默认身份必须与 `seed_data` 的演示身份是**同一个**。

    ⚠️ README 的快速开始写的是「第 4 步用 `alice` 预置，浏览器里用户名填 `alice`」。
    两个默认值一旦各改各的，文档就变成了一句错话 ——
    而错的方式是「预置到了 bob、用户却用 alice 登录」，看到的仍然是灰输入框。
    """
    from scripts.seed_data import KB_USER_ID

    assert provision_agent.DEFAULT_USER_ID == KB_USER_ID


def test_session_is_created_with_the_agent_and_without_a_model() -> None:
    """第二条拼图：建一条空会话。

    ⚠️ 刻意断言**没有** `chat_model_config`：模型该由用户在前端选择，
    服务端替他选一个等于把「用哪个模型」钉死在一次脚本调用上。
    """
    server = _FakeServer()
    with server.client() as client:
        result = provision_agent.provision(BASE_URL, USER, client=client)

    assert result.session_created is True
    assert result.session_id == "session-2"
    # 建会话的请求体带 `agent_id`；建智能体的那个只有 name/system_prompt。
    session_body = [b for b in server.bodies if "agent_id" in b]
    assert session_body, "没有发出建会话的请求"
    assert session_body[0]["agent_id"] == result.agent_id
    assert "chat_model_config" not in session_body[0]


# ==============================================================================
# 幂等：跑第二遍不许产生第二份
# ==============================================================================


def test_is_idempotent_when_the_agent_already_exists() -> None:
    """已存在同名智能体时**一个字节都不改**：不建新的，也不更新提示词。

    ⚠️ 理由不是「省一次请求」：`--user` 指向的是**别人的**工作区，
    自动刷新提示词会在对方没要求的时候改掉他正在用的智能体。
    """
    existing = {"id": "agent-existing", "data": {"name": MAIN_AGENT_NAME}}
    server = _FakeServer(agents=[existing])
    with server.client() as client:
        result = provision_agent.provision(BASE_URL, USER, client=client)

    assert result.agent_created is False
    assert result.agent_id == "agent-existing"
    methods = [(r.method, r.url.path) for r in server.requests]
    assert ("POST", "/agent/") not in methods


def test_existing_session_prevents_a_second_one() -> None:
    """该智能体名下已有会话时，不再建第二条。"""
    agent = {"id": "agent-a", "data": {"name": MAIN_AGENT_NAME}}
    server = _FakeServer(
        agents=[agent], sessions=[{"session_id": "s-old", "agent_id": "agent-a"}]
    )
    with server.client() as client:
        result = provision_agent.provision(BASE_URL, USER, client=client)

    assert result.session_created is False
    assert result.session_id is None
    assert ("POST", "/sessions/") not in [(r.method, r.url.path) for r in server.requests]


def test_session_lookup_is_scoped_to_the_agent() -> None:
    """会话查询必须带 `agent_id`。

    ⚠️ 不带的话，**别的**智能体名下的会话会被算成「已有会话」，
    于是预置会话被跳过 —— 而用户看到的仍然是一个灰输入框，脚本却报成功。
    这条缺陷在「一个用户只有一个智能体」时永远测不出来，所以这里
    显式构造了另一个智能体的会话。
    """
    agent = {"id": "agent-a", "data": {"name": MAIN_AGENT_NAME}}
    server = _FakeServer(
        agents=[agent],
        sessions=[{"session_id": "s-other", "agent_id": "agent-OTHER"}],
    )
    with server.client() as client:
        result = provision_agent.provision(BASE_URL, USER, client=client)

    lookups = [
        r for r in server.requests if r.url.path == "/sessions/" and r.method == "GET"
    ]
    assert lookups, "没有查过会话"
    assert lookups[0].url.params.get("agent_id") == "agent-a"
    # 别人的会话不算数 ⇒ 仍然要建一条自己的
    assert result.session_created is True


# ==============================================================================
# 回读核验与失败路径
# ==============================================================================


def test_readback_failure_is_reported() -> None:
    """创建返回成功、但回读看不到时，必须**报错**而不是报成功。

    ⚠️ 这是本脚本最容易假绿灯的一处：写成功、读不到（多副本可见性），
    若只信创建响应，脚本会说「✅ 已创建」而浏览器里仍然什么都没有。
    """

    class _Vanishing(_FakeServer):
        def _handle(self, request: httpx.Request) -> httpx.Response:
            response = super()._handle(request)
            if request.url.path == "/agent/" and request.method == "POST":
                # 创建成功，但列表里不留痕 —— 模拟写入不可见。
                self.agents.clear()
            return response

    server = _Vanishing()
    with server.client() as client:
        with pytest.raises(RuntimeError, match="回读核验失败"):
            provision_agent.provision(BASE_URL, USER, client=client)


def test_service_error_is_surfaced_with_its_detail() -> None:
    """服务返回非 2xx 时，报错里要带上服务给的原因，而不是一句「失败」。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "禁止使用保留身份"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(RuntimeError, match="禁止使用保留身份"):
            provision_agent.provision(BASE_URL, USER, client=client)


def test_cli_returns_one_when_the_service_is_unreachable() -> None:
    """连不上服务时，CLI 退出码必须是 1（让 CI / `make` 能感知失败）。"""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    # 把默认客户端换成「永远连不上」的那个。
    original = provision_agent.httpx.Client

    def _client_factory(*args: Any, **kwargs: Any) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(handler))

    provision_agent.httpx.Client = _client_factory  # type: ignore[assignment]
    try:
        assert provision_agent.main(["--user", USER, "--base-url", BASE_URL]) == 1
    finally:
        provision_agent.httpx.Client = original  # type: ignore[assignment]


def test_no_session_flag_skips_the_session() -> None:
    """`--no-session` 时只建智能体，不建会话。"""
    server = _FakeServer()
    with server.client() as client:
        result = provision_agent.provision(
            BASE_URL, USER, with_session=False, client=client
        )

    assert result.agent_created is True
    assert result.session_id is None
    assert ("POST", "/sessions/") not in [(r.method, r.url.path) for r in server.requests]
