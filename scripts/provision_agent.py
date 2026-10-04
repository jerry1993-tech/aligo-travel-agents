#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**首次使用引导**：为一个用户预置 AliGo 主智能体（必要时再建一条空会话）。

==============================================================================
它为什么存在 —— 一次实测复现出来的「新用户进不去」
==============================================================================
    全新部署、`make up` 全绿、`make seed_data` 灌完数据之后，运维打开浏览器：

        1. 填服务器地址 + 用户名 → 进到 /chat；
        2. 输入框是**灰的**，点不动；
        3. 页面上只有一句「请先选择一个智能体。」，而下拉框里一个智能体都没有。

    卡住的原因不是故障，而是**框架前端的一条硬性前置条件**（已核实）：

        `ChatViewport` 只由 `session.config.chat_model_config` 决定模型
        ⇒ 没有会话就没有模型 ⇒ `disabled={selectedModel === null}` 为真
        ⇒ 输入框永远禁用。

    而会话必须挂在智能体下。所以「能打字」的完整前提是：

        有智能体  →  有会话  →  有模型  →  输入框解禁

    本脚本把前两步变成一条命令。缺了它，每一个新用户都要自己猜：
    先点侧边栏「智能体」旁的 +、建一个、再点「新会话」—— 而**没有任何
    界面文案告诉他这三步**。

==============================================================================
⚠️ 名字必须是 ``main_plan``，这不是风格问题
==============================================================================
    框架把「额外中间件」加进**每一个** agent 的装配，作用域由中间件
    自己按 `agent.name` 判定（见 `src/server/agents_factory.py` 的模块文档）。
    本项目有三处按名字收窄：

        · `LaneRouterMiddleware`   → 只对 ``main_plan`` 生效（快慢车道）
        · `ContextInjectionMiddleware` 的动态段落 → 主智能体
        · `RAGMiddleware`          → 只对 ``policy_rag`` 生效

    也就是说：用户自己随手建一个叫「我的助手」的智能体，**能聊天**，
    但快慢车道不生效、动态 Prompt 不注入 —— 表面上完全正常，功能静默缺失。
    这正是本脚本要用 `MAIN_AGENT_NAME` 常量而不是字面量的原因。

==============================================================================
幂等：本脚本**只创建，不修改**
==============================================================================
    已存在同名智能体时直接返回它的 id，一个字节都不改。
    理由：`--user` 指向的是**别人的**工作区。一个「顺手把提示词刷新成最新」
    的默认行为，会在别人没要求的时候改掉他正在用的智能体；而提示词一改，
    模型行为就变了，且没有任何记录说明「谁在什么时候改的」。
    需要更新提示词时，用前端的编辑对话框（那里有明确的用户动作）。

退出码约定：**0 = 目标状态已达成**（新建或已存在）；**1 = 失败**。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

# 允许以 `python scripts/provision_agent.py` 直接运行（此时 sys.path[0] 是 scripts/）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.prompts import MAIN_AGENT_NAME, prompt_for  # noqa: E402

# ==============================================================================
# 常量
# ==============================================================================

#: 默认服务根地址。与 `scripts/smoke.py` 的 `--base-url` 默认值一致，
#: 也与 Makefile 的 `SMOKE_URL` 默认值一致 —— 三处一致是为了让
#: 「换端口」时只需要改一个地方（见 Makefile 顶部关于 SMOKE_URL 的说明）。
DEFAULT_BASE_URL = "http://localhost:8000"

#: 默认用户身份。``alice`` 是 `scripts/seed_data.py` 灌演示数据时用的身份
#: （政策知识库的属主，见该脚本的 ``KB_USER_ID``），也是 README 演示口径。
DEFAULT_USER_ID = "alice"

#: 单次 HTTP 请求超时（秒）。
#:
#: ⚠️ 建智能体这一步要写 Postgres，但**不调用模型**，所以正常在毫秒级。
#: 给 30 秒是因为它可能与 `make up` 的收尾阶段并发（容器刚起来、连接池
#: 还在建），而不是因为这一步本身慢。
REQUEST_TIMEOUT_SECONDS = 30.0

#: 预置会话的名字。写死一个可辨认的名字，用户想删时一眼能认出来。
DEFAULT_SESSION_NAME = "第一次对话"


# ==============================================================================
# 结果
# ==============================================================================


@dataclass
class ProvisionResult:
    """一次预置的产出，供调用方（CLI / 测试）断言。

    Attributes:
        agent_id (`str`): 目标智能体的存储 id（新建或已存在）。
        agent_created (`bool`): ``True`` = 本次新建；``False`` = 之前就有。
        session_id (`str | None`): 预置会话的 id；``--no-session`` 时为 ``None``。
        session_created (`bool`): ``True`` = 本次新建。
    """

    agent_id: str
    agent_created: bool
    session_id: str | None = None
    session_created: bool = False


# ==============================================================================
# HTTP 小工具
# ==============================================================================


def _headers(user_id: str) -> dict[str, str]:
    """构造请求头。

    ⚠️ 身份靠 ``X-User-ID`` 声明，这是本项目既有的鉴权口径
    （见 `src/server/middleware/auth.py`）。本脚本不引入第二种口径 ——
    它要走的正是浏览器走的那条路，否则「脚本能跑」不能证明「浏览器能用」。

    Args:
        user_id (`str`): 要冒充/代表的用户身份。

    Returns:
        `dict[str, str]`: 请求头。
    """
    return {"X-User-ID": user_id}


def _error_detail(response: httpx.Response) -> str:
    """从失败响应里取一句给人看的说明。

    框架的报错体是 ``{"detail": ...}``（可能是字符串，也可能是校验错误的
    数组）。取不到就退回状态码 + 截断的正文，**绝不吞掉**失败原因 ——
    这个脚本的失败信息是给运维看的，而运维要的是「为什么」。

    Args:
        response (`httpx.Response`): 失败响应。

    Returns:
        `str`: 一行说明。
    """
    try:
        body: Any = response.json()
        detail = body.get("detail") if isinstance(body, dict) else None
        if isinstance(detail, str) and detail:
            return detail
        if detail is not None:
            return str(detail)[:300]
    except Exception:  # noqa: BLE001 —— 解析失败不该掩盖 HTTP 失败本身
        pass
    return f"HTTP {response.status_code}：{response.text[:200]}"


# ==============================================================================
# 三步：列智能体 → 建智能体 → 建会话
# ==============================================================================


def find_agent_by_name(
    client: httpx.Client, base_url: str, user_id: str, name: str
) -> str | None:
    """在**该用户可见**的智能体里找一个同名的，返回它的 id。

    ⚠️ 可见范围由框架的 ``DenyAllResourceAccessPolicy`` 决定（owner 隔离），
    所以这里不会、也不应该看到别人的智能体。同名判定用 ``data.name`` ——
    注意存储里的记录是 ``{"id": ..., "data": {"name": ...}}`` 两层结构，
    外层 ``id`` 是存储 id（会话要绑的就是它），内层 ``data.id`` 是运行时 id，
    **两者不是一回事**（见 `src/server/agents_factory.py` 的模块文档）。

    Args:
        client (`httpx.Client`): 已建好的 HTTP 客户端。
        base_url (`str`): 服务根地址。
        user_id (`str`): 用户身份。
        name (`str`): 要匹配的智能体名。

    Returns:
        `str | None`: 命中则返回存储 id，否则 ``None``。

    Raises:
        httpx.HTTPError: 网络层失败。
        RuntimeError: 服务返回非 2xx。
    """
    response = client.get(f"{base_url}/agent/", headers=_headers(user_id))
    if response.status_code != 200:
        raise RuntimeError(f"列智能体失败：{_error_detail(response)}")

    for entry in response.json().get("agents", []):
        if (entry.get("data") or {}).get("name") == name:
            return str(entry["id"])
    return None


def create_agent(
    client: httpx.Client, base_url: str, user_id: str, name: str, system_prompt: str
) -> str:
    """创建一个智能体，返回存储 id。

    Args:
        client (`httpx.Client`): 已建好的 HTTP 客户端。
        base_url (`str`): 服务根地址。
        user_id (`str`): 用户身份。
        name (`str`): 智能体名（调用方负责传 ``MAIN_AGENT_NAME``）。
        system_prompt (`str`): 基础系统提示词。

    Returns:
        `str`: 新建智能体的存储 id。

    Raises:
        RuntimeError: 服务返回非 2xx（含 422 校验失败）。
    """
    response = client.post(
        f"{base_url}/agent/",
        headers=_headers(user_id),
        json={"name": name, "system_prompt": system_prompt},
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"建智能体失败：{_error_detail(response)}")
    return str(response.json()["agent_id"])


def has_any_session(
    client: httpx.Client, base_url: str, user_id: str, agent_id: str
) -> bool:
    """该智能体下是否已有会话。

    ⚠️ 需要显式传 ``agent_id``：框架的会话列表按 agent 过滤，
    不传的话拿到的是**全部**会话 —— 用它判断「这个智能体有没有会话」
    会得出错误答案（别的智能体的会话会被算进来），于是预置会话被跳过，
    用户打开页面看到的仍然是一个灰输入框。

    Args:
        client (`httpx.Client`): 已建好的 HTTP 客户端。
        base_url (`str`): 服务根地址。
        user_id (`str`): 用户身份。
        agent_id (`str`): 智能体存储 id。

    Returns:
        `bool`: 有会话为 ``True``。

    Raises:
        RuntimeError: 服务返回非 2xx。
    """
    response = client.get(
        f"{base_url}/sessions/",
        headers=_headers(user_id),
        params={"agent_id": agent_id},
    )
    if response.status_code != 200:
        raise RuntimeError(f"列会话失败：{_error_detail(response)}")
    payload = response.json()
    sessions = payload.get("sessions", payload) if isinstance(payload, dict) else payload
    return bool(sessions)


def create_session(
    client: httpx.Client, base_url: str, user_id: str, agent_id: str, name: str
) -> str:
    """建一条空会话，返回会话 id。

    ⚠️ **刻意不写 `chat_model_config`**：模型由前端在选择器里挑
    （`ChatViewport` 在会话没有模型时会自动选第一个可用的并回写）。
    服务端在这里替用户选一个模型，等于把「用户到底用哪个模型」这件事
    钉死在一条脚本调用上；而这个项目里模型选择是**用户可见、可改**的状态。

    Args:
        client (`httpx.Client`): 已建好的 HTTP 客户端。
        base_url (`str`): 服务根地址。
        user_id (`str`): 用户身份。
        agent_id (`str`): 智能体存储 id。
        name (`str`): 会话名。

    Returns:
        `str`: 新建会话的 id。

    Raises:
        RuntimeError: 服务返回非 2xx。
    """
    response = client.post(
        f"{base_url}/sessions/",
        headers=_headers(user_id),
        json={"agent_id": agent_id, "name": name},
    )
    if response.status_code not in (200, 201):
        raise RuntimeError(f"建会话失败：{_error_detail(response)}")
    return str(response.json()["session_id"])


# ==============================================================================
# 编排
# ==============================================================================


def provision(
    base_url: str,
    user_id: str,
    *,
    agent_name: str = MAIN_AGENT_NAME,
    with_session: bool = True,
    session_name: str = DEFAULT_SESSION_NAME,
    client: httpx.Client | None = None,
) -> ProvisionResult:
    """预置一个用户的主智能体（并按需建一条空会话）。

    Args:
        base_url (`str`): 服务根地址。
        user_id (`str`): 用户身份。
        agent_name (`str`): 智能体名；默认取 :data:`~src.agents.prompts.MAIN_AGENT_NAME`。
        with_session (`bool`): 是否顺带保证「至少有一条会话」。
        session_name (`str`): 新建会话时用的名字。
        client (`httpx.Client | None`): 复用外部客户端（测试注入用）。

    Returns:
        `ProvisionResult`: 本次做了什么。

    Raises:
        RuntimeError: 任何一步拿到非 2xx。
        httpx.HTTPError: 网络层失败。
    """
    owns_client = client is None
    http = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        # ---- 1. 智能体：有则复用，无则创建 ----
        existing = find_agent_by_name(http, base_url, user_id, agent_name)
        if existing is not None:
            result = ProvisionResult(agent_id=existing, agent_created=False)
        else:
            # ⚠️ 提示词取自 `src/agents/prompts.py` 的**同一份常量**，
            # 不在本脚本里重抄一遍。重抄的后果是：改了产品提示词，
            # 预置出来的智能体还是旧的，而两处都「看起来是对的」。
            agent_id = create_agent(
                http, base_url, user_id, agent_name, prompt_for(agent_name)
            )
            result = ProvisionResult(agent_id=agent_id, agent_created=True)

        # ---- 2. 会话：输入框解禁的最后一块拼图 ----
        if with_session and not has_any_session(
            http, base_url, user_id, result.agent_id
        ):
            result.session_id = create_session(
                http, base_url, user_id, result.agent_id, session_name
            )
            result.session_created = True

        # ---- 3. 回读核验 ----
        # 与 `scripts/milvus_init.py` 同一条纪律：写完不算数，读回来才算。
        # 这里的回读能抓住一类真实失败 —— 创建返回 201，但下一次列表里
        # 看不到（比如多副本下的写入可见性问题）。没有这一步，
        # 脚本会报「成功」，而用户打开浏览器仍然是空的。
        confirmed = find_agent_by_name(http, base_url, user_id, agent_name)
        if confirmed != result.agent_id:
            raise RuntimeError(
                "回读核验失败：刚创建的智能体没有出现在该用户的列表里"
                f"（创建返回 {result.agent_id}，回读得到 {confirmed}）。"
                "请确认服务是单副本（WORKERS=1）且 Postgres 正常。"
            )

        return result
    finally:
        if owns_client:
            http.close()


# ==============================================================================
# CLI
# ==============================================================================


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。

    Returns:
        `argparse.ArgumentParser`: 解析器。
    """
    parser = argparse.ArgumentParser(
        prog="provision_agent.py",
        description=(
            "为一个用户预置 AliGo 主智能体（幂等）。"
            "缺了它，浏览器里的输入框永远是灰的 —— 理由见本文件的模块文档。"
        ),
        epilog=(
            "例子：\n"
            "  python scripts/provision_agent.py                    # 给 alice 预置\n"
            "  python scripts/provision_agent.py --user bob         # 给 bob 预置\n"
            "  python scripts/provision_agent.py --no-session       # 只建智能体\n"
            "  make provision-agent PROVISION_USER=bob\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--user",
        default=DEFAULT_USER_ID,
        help=f"用户身份（默认 {DEFAULT_USER_ID}，与 seed_data 的演示口径一致）",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"服务根地址（默认 {DEFAULT_BASE_URL}）",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="只建智能体，不建会话（此时前端仍需手动点一次「新会话」才能输入）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv (`list[str] | None`): 命令行参数；``None`` 时取 ``sys.argv``。

    Returns:
        `int`: 0 表示目标状态已达成，1 表示失败。
    """
    args = _build_parser().parse_args(argv)

    try:
        result = provision(
            args.base_url, args.user, with_session=not args.no_session
        )
    except Exception as exc:  # noqa: BLE001 —— 顶层入口，见下
        # ⚠️ 宽捕获是刻意的：连不上、鉴权被拒、422 校验失败各自抛不同的
        # 异常类型，而这条消息是给运维看的第一手线索。包一层只剩
        # 「预置失败」反而丢掉了真正有用的那句。
        from src.observability.redaction import safe_error

        print(f"❌ 预置失败：{safe_error(exc)}", file=sys.stderr)
        print(
            f"   提示：确认服务已启动且 {args.base_url} 可达（make up 后 "
            "curl -s localhost:8000/healthz）。",
            file=sys.stderr,
        )
        return 1

    verb = "已创建" if result.agent_created else "已存在，未改动"
    print(f"✅ 智能体 {MAIN_AGENT_NAME}（{args.user}）：{verb}，id={result.agent_id}")

    if result.session_id:
        print(f"✅ 预置会话「{DEFAULT_SESSION_NAME}」：id={result.session_id}")
    elif args.no_session:
        print("ℹ️  未预置会话（--no-session）：浏览器里仍需手动点一次「新会话」。")
    else:
        print("✅ 该智能体下已有会话，未新建。")

    print()
    print("浏览器里现在应该能直接输入了：")
    print(
        f"  1. 打开 {args.base_url}/ ，服务器地址填 {args.base_url}，"
        f"用户名填 {args.user}"
    )
    # ⚠️ 会话名只在**本次真的建了**时才敢承诺 —— 别的会话名是用户自己起的，
    # 写死一个名字会让运维照着找一个并不存在的条目。
    if result.session_created:
        print(f"  2. 侧边栏会自动落在智能体 {MAIN_AGENT_NAME} 与刚建的会话上")
    else:
        print(
            f"  2. 侧边栏会自动落在智能体 {MAIN_AGENT_NAME} 与**第一条**会话上；"
            "若一条都没有，点一次「新会话」"
        )
    print("  3. 光标落在输入框即可打字，回车发送")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
