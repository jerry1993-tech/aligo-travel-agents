#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""冒烟测试：对**已启动**的 AliGo 服务发一串真实 HTTP 请求，验证它确实活着。

==============================================================================
它与 `make test` 的分工（这是本文件最需要说清楚的一件事）
==============================================================================
    `make test`（pytest）                本脚本（smoke）
    ----------------------------------  ------------------------------------------
    不需要 Docker                         **需要**服务已经起来
    用 ASGI 内存传输，进程内直调          走真实 TCP，跨容器/跨主机
    测「代码逻辑对不对」                  测「这一套部署能不能用」
    用 sqlite 内存库 + InMemory 总线      连真的 PostgreSQL / Redis / Milvus
    秒级，可跑几千次                      十几秒，只在部署后跑

    一句话：pytest 回答「我写对了吗」，smoke 回答「它**在这儿**跑起来了吗」。
    两者缺一不可 —— 一个全绿的 pytest 完全可能对应一个起不来的容器
    （配置写错、卷没挂上、端口冲突、镜像里少了个依赖…），
    那些问题**只有**对真实的地址发请求才会暴露。

==============================================================================
为什么脚本要自己写而不是用一堆 curl
==============================================================================
    · curl 拿不到「响应头里的 X-Trace-ID 是否被我沿用」这类跨请求的断言；
    · curl 的失败信息是「返回了 500」，而这里要的是「哪一步失败、期望什么、
      实际拿到什么、以及最可能的三个原因」；
    · 一个脚本可以给出**唯一的退出码**，直接接进 CI。

==============================================================================
本脚本**不留痕**：跑完自己删掉
==============================================================================
    对话检查会真的建一个智能体 + 一条会话 —— 这是「证明它真能对话」的代价。
    它们在该项检查结束时被删除，**成功、失败、异常路径都会删**
    （见 `check_conversation` 的文档）。要留现场排查时用 `--keep-artifacts`。

    在这之前它们从不清理：2026-10-03 实测 `smoke-user` 名下有 **11 个**
    智能体，其中 **10 个**是本检查历次留下的「冒烟测试助手 <时间戳>」，
    而没有任何地方记录过这件事。
    一个「跑一次多一个」的脚本最终会让人不敢再跑它 —— 而没人跑的冒烟测试
    等于没有冒烟测试。

    ⚠️ **清理的边界，别把它当保证**：清理写在 `finally` 里，所以它能覆盖的
    是「这个进程还活着」的全部情形 —— 包括检查失败、断言炸掉、SSE 断流。
    它覆盖不到的是**进程被外部打死**：`kill -9`（SIGKILL 不可捕获）和默认的
    `kill`（SIGTERM，本例没装信号处理器）都会让 `finally` 根本没机会跑，
    服务端那两行记录就留下了。CI 里超时杀进程属于这一类。
    也就是说「不留痕」是常态而不是铁律 —— 残留仍可能出现，只是从
    「每跑一次多一个」变成了「只有被强杀才可能多一个」。

退出码约定：**0 = 全部通过；1 = 有检查失败**。
    （与 Makefile 的 `smoke` 目标配合，任一失败即 `make smoke` 失败。）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

# ==============================================================================
# 常量
# ==============================================================================

#: 单次 HTTP 请求的超时（秒）。
#:
#: ⚠️ 它必须**大于**服务端 /readyz 自己的最坏耗时，否则冒烟测试会在服务
#: 尚在正常工作时误报失败。服务端的耗时构成：
#:     scripts/postgres/init 之后，/readyz 并发跑 4 项检查，
#:     每项有 3 秒的独立超时（src/server/probes.py::CHECK_TIMEOUT_SECONDS），
#:     且三项 I/O 检查是**并发**跑的 ⇒ 最坏 ≈ 3 秒 + 调度开销。
#: 给到 10 秒留了 3 倍余量：够容忍慢机器，又能在真挂住时及时失败。
REQUEST_TIMEOUT_SECONDS = 10.0

#: 等待服务就绪的最长时间（秒）。
#: 与 docker-compose.yaml 里 app 健康检查的 start_period 相称。
#: `make up` 已经用 `--wait` 等过一轮，这里再等是为了支持
#: 「服务刚重启、还没就绪」这种手工调用场景。
READY_WAIT_SECONDS = 30.0

#: 轮询 /readyz 的间隔（秒）。
READY_POLL_INTERVAL_SECONDS = 1.5

#: 对话检查使用的身份。用一个**专用**身份（而不是复用运维自己的），
#: 是为了让检查可重复：每次都从零开始建智能体与会话，不受既有数据干扰。
#:
#: ⚠️ 用专用身份还有**第二重**作用，是后来才补上的：这一轮建的智能体与会话
#: 用完即删（见 :func:`check_conversation`）。若复用运维自己的身份，
#: 「删掉本轮产物」与「删掉用户真实数据」就落在同一个工作区里 ——
#: 一次逻辑写错（比如删错 id）毁掉的是真数据，而不是一个一次性工作区。
SMOKE_USER_ID = "smoke-user"

#: 一轮对话最多等多久（秒）。模型链路包含「建会话 → 装配智能体 → 调模型」，
#: 真实模型的首字延迟远大于任何一个探针，给足余量。
CONVERSATION_TIMEOUT_SECONDS = 60.0

#: 对话检查里轮询 SSE 的间隔（秒）。
CONVERSATION_POLL_INTERVAL_SECONDS = 0.25


# ==============================================================================
# 结果模型
# ==============================================================================
@dataclass
class Check:
    """一次检查的结果。

    Attributes:
        name (`str`): 检查项的可读名称（打印在报告里）。
        ok (`bool`): 是否通过。
        detail (`str`): 通过时的补充信息，或失败时的原因。
        fatal (`bool`): 失败时是否应当**立即中止**后续检查。
            用于表达「这一步没过，后面的检查没有意义」——
            例如健康检查都失败了，再去查指标只会产生一串误导性的次生失败。
    """

    name: str
    ok: bool
    detail: str = ""
    fatal: bool = False


@dataclass
class Report:
    """整轮冒烟测试的汇总。

    Attributes:
        checks (`list[Check]`): 按执行顺序记录的检查项。
    """

    checks: list[Check] = field(default_factory=list)

    @property
    def failed(self) -> list[Check]:
        """返回所有未通过的检查项。

        Returns:
            `list[Check]`: 失败项（可能为空）。
        """
        return [c for c in self.checks if not c.ok]

    def add(self, check: Check) -> None:
        """记录一项检查并按需打印结果。

        Args:
            check (`Check`): 检查结果。
        """
        self.checks.append(check)
        mark = "✅" if check.ok else "❌"
        line = f"  {mark} {check.name}"
        if check.detail:
            line += f" —— {check.detail}"
        print(line, flush=True)


# ==============================================================================
# 单项检查
# ==============================================================================
def _describe_connection_error(exc: Exception, base_url: str) -> str:
    """把连接层的异常翻译成一段**能直接照着做**的提示。

    ⚠️ 这段翻译不是锦上添花。冒烟测试最常见的失败就是「连不上」，
    而 httpx 的原始报错（``All connection attempts failed``）完全不提
    「服务是不是根本没起来」「端口对不对」「容器健康吗」。
    一条只说「连不上」的报错，会让人从网络配置开始查，而真正的原因
    通常是「你忘了先 make up」。

    Args:
        exc (`Exception`): 捕获到的异常。
        base_url (`str`): 目标地址（用于提示里回显）。

    Returns:
        `str`: 面向人的排障提示。
    """
    return (
        f"连不上 {base_url}（{type(exc).__name__}: {exc}）。\n"
        f"    最可能的三个原因：\n"
        f"      1. 服务还没启动 —— 先跑 `make up`（它会等到所有容器 healthy）；\n"
        f"      2. 端口不对 —— 默认 8000，若改过 compose 的端口映射，\n"
        f"         请用 `make smoke SMOKE_URL=http://localhost:<端口>`；\n"
        f"      3. 容器起来了但没通过健康检查（`docker ps` 看 STATUS 列，\n"
        f"         或 `make logs` 看应用日志）。"
    )


def check_healthz(client: httpx.Client, base_url: str) -> Check:
    """检查 ``/healthz``：进程存活。

    ⚠️ 这一步是后续所有检查的**前提**，因此标了 ``fatal=True``：
    它都过不了的话，后面每一条都会以「连不上」失败，
    那串次生失败会把真正的原因埋在中间。

    Args:
        client (`httpx.Client`): HTTP 客户端。
        base_url (`str`): 服务地址。

    Returns:
        `Check`: 检查结果。
    """
    try:
        response = client.get(f"{base_url}/healthz")
    except Exception as exc:  # pylint: disable=broad-except
        return Check("存活探针 /healthz", False, _describe_connection_error(exc, base_url), fatal=True)

    if response.status_code != 200:
        return Check(
            "存活探针 /healthz",
            False,
            f"期望 200，实际 {response.status_code}；响应体：{response.text[:200]}",
            fatal=True,
        )

    # 校验响应体而不只是状态码：一个「所有路径都返回 200 空体」的假服务
    # （比如某个反代的兜底页）能骗过状态码断言，但骗不过字段断言。
    try:
        payload = response.json()
    except ValueError:
        return Check("存活探针 /healthz", False, f"响应不是 JSON：{response.text[:200]}", fatal=True)

    if payload.get("status") != "ok":
        return Check("存活探针 /healthz", False, f"响应体不符合契约：{payload}", fatal=True)

    return Check("存活探针 /healthz", True, "200 {'status': 'ok'}")


def check_readyz(client: httpx.Client, base_url: str) -> Check:
    """检查 ``/readyz``：全部必需依赖可用。

    ⚠️ 这里**接受 503 并被标记为失败**，而不是直接当成「服务坏了」：
    503 的响应体里带着逐项明细，把它原样打出来，
    比一句「未就绪」有用得多 —— 那正是探针设计成自解释的目的。

    Args:
        client (`httpx.Client`): HTTP 客户端。
        base_url (`str`): 服务地址。

    Returns:
        `Check`: 检查结果。
    """
    try:
        response = client.get(f"{base_url}/readyz")
    except Exception as exc:  # pylint: disable=broad-except
        return Check("就绪探针 /readyz", False, _describe_connection_error(exc, base_url), fatal=True)

    try:
        payload = response.json()
    except ValueError:
        return Check("就绪探针 /readyz", False, f"响应不是 JSON：{response.text[:200]}", fatal=True)

    if response.status_code != 200:
        # 把逐项明细压成一行「哪几项没过」，让报告的第一屏就能看出方向。
        failed = [
            name
            for name, item in (payload.get("checks") or {}).items()
            if isinstance(item, dict) and item.get("required") and not item.get("ok")
        ]
        hint = f"未通过的必需项：{', '.join(failed)}" if failed else "（无必需项失败）"
        return Check(
            "就绪探针 /readyz",
            False,
            f"期望 200，实际 {response.status_code}；{hint}\n"
            f"    完整明细：{json.dumps(payload, ensure_ascii=False)[:500]}",
        )

    # 就绪时必须逐项确认 —— 只信状态码的话，一个「永远返回 200」的实现也能通过。
    checks = payload.get("checks") or {}
    not_ok = [name for name, item in checks.items() if isinstance(item, dict) and not item.get("ok")]
    if not_ok:
        return Check(
            "就绪探针 /readyz",
            False,
            f"返回 200，但明细里有未通过的项：{', '.join(not_ok)}（状态码与明细自相矛盾）",
        )

    return Check("就绪探针 /readyz", True, f"200，必需项全部通过：{', '.join(checks) or '<无>'}")


def check_metrics(client: httpx.Client, base_url: str) -> Check:
    """检查 ``/metrics``：Prometheus 文本格式可达。

    Args:
        client (`httpx.Client`): HTTP 客户端。
        base_url (`str`): 服务地址。

    Returns:
        `Check`: 检查结果。
    """
    try:
        response = client.get(f"{base_url}/metrics")
    except Exception as exc:  # pylint: disable=broad-except
        return Check("指标端点 /metrics", False, _describe_connection_error(exc, base_url))

    if response.status_code == 404:
        # 404 是**合法的配置结果**（observability.metrics_enabled=false），
        # 因此报错信息要说清「这不是坏了，是关掉了」，否则会被当成故障排查半天。
        return Check(
            "指标端点 /metrics",
            False,
            "返回 404 —— 指标端点被关闭了（observability.metrics_enabled=false）。"
            "这是合法的配置选择；若要启用，请把它设为 true 后重启。",
        )

    if response.status_code != 200:
        return Check("指标端点 /metrics", False, f"期望 200，实际 {response.status_code}")

    body = response.text
    if "aligo_ready" not in body:
        return Check(
            "指标端点 /metrics",
            False,
            f"返回 200 但缺少本项目自己的指标 aligo_ready（前 200 字符：{body[:200]}）",
        )

    # 指标端点是**无鉴权**的，因此顺手确认它没把密钥漏出去。
    # 这条检查放在冒烟里而不是只在单测里，是因为只有在这里它才检验**真实部署**
    # 的配置（单测用的是测试配置，可能与部署的不同）。
    for needle in ("sk-", "password", "PASSWORD"):
        if needle in body:
            return Check(
                "指标端点 /metrics",
                False,
                f"响应里出现了疑似敏感串 {needle!r} —— 指标端点无鉴权，不能含密钥",
            )

    return Check("指标端点 /metrics", True, f"200，{len(body.splitlines())} 行 Prometheus 文本")


def check_trace_id(client: httpx.Client, base_url: str) -> Check:
    """检查 trace_id 的生成与沿用（响应头契约）。

    两件事一起验，因为它们互为反例：
      · 不传 ``X-Trace-ID`` ⇒ 响应里必须有一个**非空**的新 id（生成能力）；
      · 传一个已知值 ⇒ 响应里必须是**同一个**值（沿用能力）。

    只验第一条的话，一个「永远随机生成」的实现能通过 ——
    而那种实现会让上游网关的 id 与本服务的 id 对不上，链路在第一个跳点就断了。

    Args:
        client (`httpx.Client`): HTTP 客户端。
        base_url (`str`): 服务地址。

    Returns:
        `Check`: 检查结果。
    """
    try:
        generated = client.get(f"{base_url}/healthz").headers.get("x-trace-id", "")
    except Exception as exc:  # pylint: disable=broad-except
        return Check("trace_id 传播", False, _describe_connection_error(exc, base_url))

    if not generated:
        return Check(
            "trace_id 传播",
            False,
            "响应头里没有 X-Trace-ID —— 日志与链路追踪将无法关联到具体请求",
        )

    upstream = "smoke-upstream-trace-id"
    try:
        echoed = client.get(
            f"{base_url}/healthz",
            headers={"X-Trace-ID": upstream},
        ).headers.get("x-trace-id", "")
    except Exception as exc:  # pylint: disable=broad-except
        return Check("trace_id 传播", False, _describe_connection_error(exc, base_url))

    if echoed != upstream:
        return Check(
            "trace_id 传播",
            False,
            f"上游传入的 trace_id 没有被沿用（传入 {upstream!r}，返回 {echoed!r}）——"
            f"跨服务链路会在本服务断开",
        )

    return Check("trace_id 传播", True, f"生成 {generated!r}，沿用 {upstream!r}")


def check_travel_api(client: httpx.Client, base_url: str) -> Check:
    """检查业务接口 ``/api/v1/**`` 是否可达**且真的返回了业务响应**。

    ⚠️ 本检查在 P2 之前是「404 也算通过」的（那时业务路由确实还没实现）。
        P2 交付了 ``/api/v1/health`` 与 ``/api/v1/me``，因此那条豁免已经删除 ——
        现在 404 是**失败**，它意味着 ``include_router`` 没生效、
        或者前缀被某个子路由的绝对路径覆盖掉了。

    ⚠️ 原实现还有一处更隐蔽的问题：它在非 404 时直接
        ``return Check(..., True, f"{response.status_code}")`` ——
        也就是说 **500 也算通过**。冒烟测试里出现一个「除了 404 什么都放行」
        的检查项，等于安慰剂：它唯一会报错的场景恰恰是被豁免的那一种。

        现在只认 200，并且顺带校验响应体里的 ``status == "ok"`` ——
        一个返回 200 但内容是错误页的响应不该被算作通过。

    Args:
        client (`httpx.Client`): HTTP 客户端。
        base_url (`str`): 服务地址。

    Returns:
        `Check`: 检查结果。
    """
    name = "业务接口 /api/v1/health"
    try:
        response = client.get(f"{base_url}/api/v1/health")
    except Exception as exc:  # pylint: disable=broad-except
        return Check(name, False, _describe_connection_error(exc, base_url))

    if response.status_code == 404:
        return Check(
            name,
            False,
            "404 —— 业务路由未注册。\n"
            "    下一步：确认 src/server/app.py 里调用了 "
            "`app.include_router(api_v1_router)`，"
            "且子路由没有用绝对路径把 `/api/v1` 前缀覆盖掉。\n"
            "    （P2 已交付这两个路由，404 不再是预期状态。）",
        )

    if response.status_code != 200:
        return Check(
            name,
            False,
            f"{response.status_code} —— {_body_excerpt(response)}",
        )

    try:
        body = response.json()
    except ValueError:
        return Check(name, False, f"响应不是 JSON：{_body_excerpt(response)}")

    if body.get("status") != "ok":
        return Check(name, False, f"响应体里的 status 不是 'ok'：{body}")

    return Check(name, True, "200，status=ok")


def check_identity_endpoint(client: httpx.Client, base_url: str) -> Check:
    """检查 ``/api/v1/me`` 被**鉴权中间件**挡住（无凭据时 401）。

    ★ 这是对「鉴权真的装上了」最直接的一次现场验证，而且只能在**真实部署**里做：

        中间件、路由、配置三者各自都对，装配顺序却写反了 —— 这种情况在单元
        测试里由 ``test_middleware_stack_order`` 覆盖，但它覆盖的是**装配结果**；
        这里验证的是**跑起来的进程**里那条链真的在生效。

    ⚠️ 断言 **401 而不是 422**：422 意味着请求穿过鉴权层、
        被框架的 ``Header(...)`` 依赖拦下了 —— 即中间件没起作用。
        两者都是「拒绝」，但含义完全不同（见 src/server/middleware/auth.py）。
        这也正是本检查比「返回了非 200」有价值的地方：
        后者会把「中间件没生效」判成通过。

    Args:
        client (`httpx.Client`): HTTP 客户端。
        base_url (`str`): 服务地址。

    Returns:
        `Check`: 检查结果。
    """
    name = "鉴权生效 /api/v1/me 无凭据返回 401"
    try:
        response = client.get(f"{base_url}/api/v1/me")
    except Exception as exc:  # pylint: disable=broad-except
        return Check(name, False, _describe_connection_error(exc, base_url))

    if response.status_code == 422:
        return Check(
            name,
            False,
            "422 —— 请求穿过了鉴权中间件，被框架的 Header(...) 依赖拦下了，"
            "说明 AuthMiddleware 没有生效。\n"
            "    下一步：检查 src/server/app.py 的 add_middleware 顺序 —— "
            "AuthMiddleware 必须在中间件链里。",
        )

    if response.status_code == 404:
        # 路由不存在时，任何 404 都不构成「鉴权生效」的证据 ——
        # 那样这条检查会变成一个永远绿灯的摆设。
        return Check(name, False, "404 —— /api/v1/me 未注册，本检查无法验证鉴权。")

    if response.status_code != 401:
        return Check(
            name,
            False,
            f"{response.status_code} —— 期望 401。{_body_excerpt(response)}",
        )

    # RFC 6750 §3：401 应当带 WWW-Authenticate，客户端据此知道该用哪种方案。
    if "www-authenticate" not in {k.lower() for k in response.headers}:
        return Check(name, False, "401 但缺少 WWW-Authenticate 响应头（RFC 6750 §3）。")

    return Check(name, True, "401，且带 WWW-Authenticate")


def check_conversation(
    client: httpx.Client, base_url: str, *, keep_artifacts: bool = False
) -> Check:
    """★ 端到端：真的发一句话，并拿到一段**正常结束**的回复。

    ⚠️ 为什么必须有这一项 —— 这是本项目最重要的一条冒烟检查：

        健康检查、就绪检查、指标、鉴权全都只能证明「进程活着、存储通着」。
        它们**完全看不到**「有没有模型可用」：零密钥部署下，框架的对话链路
        因为找不到 ``chat_model_config`` 而直接以 error 结束，
        而所有探针依然全绿、页面依然正常渲染 —— 用户看到的是
        「发送按钮是灰的」或者「问什么都回一句错误」。

        这是真实发生过的部署形态，也是本检查存在的唯一理由：
        用一句话把「部署能不能真的用」钉死。

    检查步骤（与浏览器走的是同一条路）：
        1. 问 ``GET /api/v1/default-model`` 要一份可用的模型配置；
        2. 建智能体、建会话（把配置写进会话）；
        3. 打开 SSE，再 ``POST /chat/``（**先订阅再触发**，理由见下）；
        4. 等到 ``REPLY_END``，断言 ``finished_reason == "completed"``
           且这一轮确实产出了非空文本。

    ⚠️ 为什么「先订阅再触发」不是可选的：SSE 是**推送**通道，事件在
        订阅建立之前发出去就错过了。本项目的 ``/sessions/{id}/stream``
        与 ``/api/v1/sessions/{id}/chains`` 都实现了「先补发当前一轮的
        缓冲事件、再转入实时推送」，因此即使订阅晚了一点也仍能拿到
        （这正是 ``_read_replay_tail`` 存在的意义）。

    ═══ 本轮产物：用完即删 ═══

    ⚠️ 本检查每跑一次就建一个智能体 + 一条会话。**最初它们从不清理**，
    于是每跑一次 `make smoke` 就攒一个。2026-10-03 实测：`smoke-user`
    名下有 **11 个**智能体，其中 **10 个**正是本检查历次留下的
    「冒烟测试助手 <时间戳>」—— 而没有任何地方记录过这件事。

    清理放在 ``finally`` 里而不是成功路径的末尾，理由是**失败的那一轮
    留下的垃圾最多**：检查在「建完智能体、SSE 没等到 REPLY_END」时返回，
    此时产物已经存在。只在成功路径清理，等于专挑运行正常的时候打扫。

    ⚠️ 清理失败**不会**把这一项判成失败。本检查回答的是「部署能不能真的
    对话」，把 DELETE 的成败并进同一个判据，会让一次「对话完全正常、
    只是清理没成功」变成红色 —— 而红色会把人引去查对话链路，方向完全错了。
    但也不能静默：清理结果会写进详情（`Report.add` 对**通过项**也会打印
    detail），所以绿色行上要么写着「已清理本轮产物」，要么写着 ⚠️ 与原因。

    Args:
        client (`httpx.Client`): HTTP 客户端（用于非流式请求）。
        base_url (`str`): 服务地址。
        keep_artifacts (`bool`): ``True`` 时**跳过清理**，并把 id 打出来。

    Returns:
        `Check`: 检查结果。
    """
    created: dict[str, str] = {}
    notes: list[str] = []
    try:
        result = _run_conversation_check(client, base_url, created)
    finally:
        # ⚠️ 必须放进 ``finally``：`_run_conversation_check` 里的每一条提前
        # return 都是一次「产物已存在」的路径，而意料之外的异常同样不该
        # 留下垃圾。
        notes.extend(
            _cleanup_artifacts(client, base_url, created, keep=keep_artifacts)
        )

    result.detail = _append_cleanup_note(result.detail, notes, created, keep_artifacts)
    return result


def _run_conversation_check(
    client: httpx.Client, base_url: str, created: dict[str, str]
) -> Check:
    """:func:`check_conversation` 的实体：跑一轮真实对话。

    ⚠️ 与包装函数分开，唯一目的是让「清理」能挂在 ``finally`` 上：
    本函数有十几处提前 ``return``，逐处补一句清理调用，漏掉任何一处
    就重新开始攒垃圾 —— 而漏掉的那一处不会有任何症状。

    Args:
        client (`httpx.Client`): HTTP 客户端（用于非流式请求）。
        base_url (`str`): 服务地址。
        created (`dict[str, str]`): **出参**。建出产物后立刻写入
            ``agent_id`` / ``session_id``，供调用方清理。刻意用出参而不是
            返回值：产物在函数中途就存在了，而返回值只出现在结尾 ——
            中途失败时返回值根本不存在，清理却必须知道 id。

    Returns:
        `Check`: 检查结果。
    """
    import threading

    name = "端到端对话 / 一句话拿到正常结束的回复"
    headers = {"X-User-ID": SMOKE_USER_ID}

    # ---- 1. 要一份可用的模型配置 ------------------------------------------------
    try:
        resolved = client.get(f"{base_url}/api/v1/default-model", headers=headers)
    except Exception as exc:  # pylint: disable=broad-except
        return Check(name, False, _describe_connection_error(exc, base_url))

    if resolved.status_code == 401:
        return Check(
            name,
            False,
            "401 —— 带了 X-User-ID 仍被拒。检查 ALIGO__AUTH__ 相关配置："
            "开启 JWT 时请求头身份会被忽略，需要改用 Bearer token"
            "（python scripts/mint_token.py --user smoke-user --ttl 600）。",
        )
    if resolved.status_code == 404:
        return Check(
            name,
            False,
            "404 —— /api/v1/default-model 未注册。"
            "检查 src/server/routers/__init__.py 是否 include 了该路由。",
        )
    if resolved.status_code != 200:
        return Check(
            name,
            False,
            f"{resolved.status_code} —— 期望 200。{_body_excerpt(resolved)}",
        )

    body = resolved.json()
    config = body.get("chat_model_config")
    if not config:
        # 这是最需要被说清楚的一种失败：服务是「活着但用不了」。
        #
        # ⚠️ 处置方式**取决于是哪一种「没有」**，而这两种的端口一模一样：
        #   · mode=configured —— 用户自己配了凭据，得由他在选择器里挑一个
        #     （服务端刻意不替他选）。这不是故障。
        #   · mode=missing —— 真的没有任何可用凭据。
        # 报错文案若不加区分，前一种会被当成故障去排查，而后一种会被
        # 当成「正常，只是还没选」，两边都浪费。
        if body.get("mode") == "configured":
            return Check(
                name,
                False,
                "本用户已配置模型凭据，但**尚未选择**要用的模型"
                "（mode=configured，服务端不替用户做这个选择）。"
                "\n    下一步：在界面的模型选择器里挑一个模型后重试。"
                "\n    ⚠️ 这一项是端到端对话检查，它必须能拿到一份"
                "**确定的**配置才能继续 —— 这不是服务故障。",
            )
        return Check(
            name,
            False,
            f"没有可用的模型配置（mode={body.get('mode')!r}）。"
            f"\n    服务端提示：{body.get('hint')}"
            "\n    下一步（按部署形态二选一）："
            "\n      · 配了真密钥：检查启动日志里有没有「系统凭据播种失败」"
            "（种子写库失败时全体用户都会落到这个状态，"
            "与「压根没配密钥」表现相同）；"
            "\n      · 没配密钥：在「凭据」页添加一条模型凭据，"
            "或打开零密钥降级 ALIGO__LLM__USE_MOCK_WHEN_NO_KEY=true。"
            "\n    ⚠️ 此刻浏览器的发送按钮是**灰的**（可用模型列表为空）。",
        )

    # ---- 2. 建智能体与会话 -----------------------------------------------------
    try:
        agent_response = client.post(
            f"{base_url}/agent/",
            headers=headers,
            json={
                "name": f"冒烟测试助手 {int(time.time())}",
                "system_prompt": "你是差旅助手，回答请简短。",
            },
        )
        if agent_response.status_code not in (200, 201):
            return Check(
                name,
                False,
                f"建智能体失败：{agent_response.status_code} "
                f"{_body_excerpt(agent_response)}",
            )
        agent_id = agent_response.json()["agent_id"]
        # ⚠️ **创建成功的那一刻**就登记，而不是等这一轮跑完再统一登记：
        #    下面任何一步提前 return，产物都已经存在于服务端了，
        #    而清理只认这个字典里有的 id。晚登记一次 = 泄漏一个智能体。
        created["agent_id"] = agent_id

        session_response = client.post(
            f"{base_url}/sessions/",
            headers=headers,
            json={
                "agent_id": agent_id,
                "name": "冒烟测试会话",
                "chat_model_config": config,
            },
        )
        if session_response.status_code not in (200, 201):
            return Check(
                name,
                False,
                f"建会话失败：{session_response.status_code} "
                f"{_body_excerpt(session_response)}",
            )
        session_id = session_response.json()["session_id"]
        created["session_id"] = session_id
    except Exception as exc:  # pylint: disable=broad-except
        return Check(name, False, f"建智能体/会话时出错：{exc!r}")

    # ---- 3. 先订阅 SSE，再触发一轮对话 ------------------------------------------
    events: list[dict[str, Any]] = []
    stream_error: list[str] = []
    done = threading.Event()

    def _read_stream() -> None:
        """在后台线程里读 SSE，直到 REPLY_END 或这个流结束。

        ⚠️ 结束时**一定要** ``done.set()``（见下面的 ``finally``）：
        读取线程一结束，这个流就不可能再送来 REPLY_END 了，
        主线程没有理由继续空等。否则「连接被拒」这种瞬时失败也要
        耗满 60 秒才报出来 —— 一次确定无疑的失败，却拖一分钟才肯说。
        """
        try:
            # ⚠️ 流式请求必须用**独立的读超时**：共用那个 10 秒的客户端会让
            # 空闲的 SSE 连接每隔 10 秒被判超时。
            with httpx.Client(
                timeout=httpx.Timeout(
                    REQUEST_TIMEOUT_SECONDS,
                    read=CONVERSATION_TIMEOUT_SECONDS,
                ),
            ) as stream_client:
                with stream_client.stream(
                    "GET",
                    f"{base_url}/sessions/{session_id}/stream",
                    params={"agent_id": agent_id},
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        stream_error.append(f"SSE 返回 {response.status_code}")
                        return
                    for line in response.iter_lines():
                        if done.is_set():
                            return
                        if not line.startswith("data:"):
                            continue
                        try:
                            payload = json.loads(line[len("data:") :].strip())
                        except json.JSONDecodeError:
                            continue
                        events.append(payload)
                        if payload.get("type") == "REPLY_END":
                            done.set()
                            return
        except Exception as exc:  # pylint: disable=broad-except
            stream_error.append(f"{type(exc).__name__}: {exc}")
        finally:
            done.set()

    reader = threading.Thread(target=_read_stream, daemon=True)
    reader.start()

    # 订阅是异步建立的，给它一点时间挂上去再触发 —— 与浏览器「先连流再发消息」
    # 的时序一致。即使这一步没赶上，回放日志也能兜住（见上面的说明）。
    time.sleep(1.0)

    try:
        trigger = client.post(
            f"{base_url}/chat/",
            headers=headers,
            json={
                "agent_id": agent_id,
                "session_id": session_id,
                "input": {
                    "name": "user",
                    "role": "user",
                    "content": [{"type": "text", "text": "帮我规划下周去上海出差。"}],
                },
            },
        )
    except Exception as exc:  # pylint: disable=broad-except
        done.set()
        return Check(name, False, f"触发对话时出错：{exc!r}")

    if trigger.status_code not in (200, 202):
        done.set()
        return Check(
            name,
            False,
            f"POST /chat/ 返回 {trigger.status_code}：{_body_excerpt(trigger)}",
        )

    # ---- 4. 等 REPLY_END 并判定 -------------------------------------------------
    deadline = time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
    while not done.is_set() and time.monotonic() < deadline:
        done.wait(timeout=CONVERSATION_POLL_INTERVAL_SECONDS)
    # ⚠️ 此刻 done 还没置位 ⇒ 是**我们**等不下去了，而不是读取线程收工了。
    #    「没有 REPLY_END」有两种成因，处置方向完全不同，必须分开报（见下）。
    timed_out = not done.is_set()
    done.set()  # 让读取线程尽快退出

    if stream_error:
        return Check(name, False, f"SSE 读取失败：{stream_error[0]}")
    types = [event.get("type") for event in events]
    if "REPLY_END" not in types:
        # ⚠️ 下面两条提示里写的是 ``make logs``，**不带**服务名 ——
        #    这不是省略。Makefile 的 ``logs`` 目标是 ``logs -f ... $(or $(SVC),app)``，
        #    即服务名走 ``SVC`` 变量、默认就是 app；而 ``make logs app`` 会被 make
        #    解析成**两个目标**，仓库里并没有叫 ``app`` 的目标，于是它连日志都不看，
        #    直接抛 ``No rule to make target 'app'``。运维照着提示敲，拿到的是一句
        #    make 报错 —— 比不给提示更糟，因为它看起来像是「连 make 都坏了」。
        # ⚠️ 「我们等超时了」与「流自己断了」是**两件事**，不能共用一句话：
        #    · 超时 → 服务还连着，但模型调用或智能体装配挂住了；
        #    · 提前断 → 连接被中途掐断（反向代理的读超时、容器被 OOM、
        #      服务端进程崩了），而模型可能压根没问题。
        #    写成同一句「60 秒内没有收到 REPLY_END」会把人引去查错的地方 ——
        #    明明是代理掐了连接，却去翻模型的出口网络。
        headline = (
            f"{CONVERSATION_TIMEOUT_SECONDS:.0f} 秒内没有收到 REPLY_END"
            f"（已收到 {len(events)} 个事件：{types[:8]}）。"
            if timed_out
            else (
                f"SSE 流在没有 REPLY_END 的情况下结束了"
                f"（已收到 {len(events)} 个事件：{types[:8]}）—— 连接被中途断开，"
                "不是等待超时。"
            )
        )
        return Check(
            name,
            False,
            headline
            + "\n    最可能的原因：模型调用挂住（检查出口网络与 "
            "ALIGO__LLM__BASE_URL）、智能体装配抛错（make logs），"
            "或连接被代理/负载均衡按空闲超时切断。",
        )

    end = next(e for e in events if e.get("type") == "REPLY_END")
    text = "".join(
        str(e.get("delta") or "")
        for e in events
        if e.get("type") == "TEXT_BLOCK_DELTA"
    )
    if end.get("finished_reason") != "completed":
        return Check(
            name,
            False,
            f"这一轮以 {end.get('finished_reason')!r} 结束（期望 'completed'）—— "
            f"服务是活着的，但这句问答**实际失败了**。"
            f"\n    回复文本：{text[:200]!r}"
            "\n    下一步：make logs 查看模型调用错误；"
            "确认会话的 chat_model_config 指向一条真实存在的凭据。",
        )
    if not text.strip():
        return Check(
            name,
            False,
            "收到了 REPLY_END 但这一轮没有任何文本增量 —— 用户会看到一条空回复。",
        )

    return Check(name, True, f"回复正常结束，收到 {len(text)} 个字符")


#: 删除时视为「这个资源已经不在了」的状态码。
#:   · 200 / 204 —— 删掉了；
#:   · 404       —— 它本来就不在（上一轮删过、或建到一半失败）。
#: 两者对清理而言**同义**，都算干净。把 404 报成问题只会制造噪音，
#: 而清理报告里的噪音会训练人忽略它 —— 那才是真的危险。
_ALREADY_GONE_STATUSES = (200, 204, 404)


def _cleanup_artifacts(
    client: httpx.Client,
    base_url: str,
    created: dict[str, str],
    *,
    keep: bool,
) -> list[str]:
    """删除本轮冒烟在服务端留下的会话与智能体，并回读确认。

    ⚠️ **本函数在任何情况下都不抛异常**（除了 ``KeyboardInterrupt`` 这类
    不该被吞的），这是它最重要的一条约定。它由 :func:`check_conversation`
    的 ``finally`` 调用，此刻很可能正有一个异常在往外冒：清理自己再抛一次，
    就会**顶替掉原始异常** —— 运维看到的是「清理失败」，而真正让这一轮
    失败的原因（模型挂死、建会话报错）就此消失。

    ⚠️ 清理失败**不改判**：本函数只把问题**写成文字**交回调用方，由它贴在
    检查详情里。理由是「能不能对话」与「能不能删干净」是两个问题；
    把 DELETE 的成败并进对话检查的判据，会让一次「对话完全正常、
    只是清理没成功」变红，把人引去查对话链路，方向完全错了。

    顺序是**先会话、后智能体**：
      · 框架的 ``DELETE /agent/{id}`` 确实会级联删掉该智能体名下的会话
        （``_agent.py`` 的 ``delete_agent`` 文档：cascades through every
        session owned by this agent），所以反过来也能删干净；
      · 但**会话**的删除接口把 ``agent_id`` 声明为必需查询参数
        （``_session.py`` 的 ``delete_session``）。智能体先没了，这条 DELETE
        就永远无法执行 —— 于是智能体那一步一旦被拒（403/404、权限策略变化），
        会话就跟着留在库里，而报告上看起来「删过了」。

    Args:
        client (`httpx.Client`): HTTP 客户端。
        base_url (`str`): 服务地址。
        created (`dict[str, str]`): 本轮登记的产物 id；空字典 = 什么都没建。
        keep (`bool`): ``True`` 时**跳过清理**（调用方要求保留现场）。

    Returns:
        `list[str]`: 问题描述；**空列表 = 干净**（含「本来就没产物」）。
    """
    if keep or not created:
        return []

    headers = {"X-User-ID": SMOKE_USER_ID}
    agent_id = created.get("agent_id")
    session_id = created.get("session_id")
    problems: list[str] = []

    def _delete(path: str, what: str, params: dict[str, str] | None = None) -> None:
        """发一条 DELETE；失败只记问题，不抛。"""
        try:
            response = client.delete(
                f"{base_url}{path}", headers=headers, params=params or {}
            )
        except Exception as exc:  # pylint: disable=broad-except
            problems.append(f"删除{what}时出错：{exc!r}")
            return
        if response.status_code not in _ALREADY_GONE_STATUSES:
            problems.append(
                f"删除{what}返回 {response.status_code}：{_body_excerpt(response)}"
            )

    # ---- 1. 会话（先删，理由见 docstring）----------------------------------
    if session_id and agent_id:
        _delete(f"/sessions/{session_id}", f"会话 {session_id}", {"agent_id": agent_id})

        # ---- 2. 回读会话 --------------------------------------------------
        # ⚠️ 这一步必须在删智能体**之前**做：智能体一没，列表接口就 404，
        #    那时再也分辨不出「会话删掉了」与「会话还在、只是智能体没了」。
        if not problems:
            try:
                listed = client.get(
                    f"{base_url}/sessions/",
                    headers=headers,
                    params={"agent_id": agent_id},
                )
                if listed.status_code == 200:
                    # ⚠️ 这里断言的是「这个智能体名下**一条会话都没有了**」，
                    #    而不是「列表里找不到那个 id」。后者看着更精确，实则更弱：
                    #    它要先从响应体里把 id 抠出来（`session.id`），
                    #    哪天框架给这个字段改了名，抠出来的会是一串 None，
                    #    断言照样通过 —— 又是一个假绿灯。
                    #    而这个智能体是本轮**刚建**的、只装了这一条会话，
                    #    「列表为空」既与字段名无关，也严格更强。
                    rows = listed.json().get("sessions", [])
                    if rows:
                        problems.append(
                            f"回读核验失败：会话 {session_id} 删除后，"
                            f"该智能体名下仍能列出 {len(rows)} 条会话"
                        )
                else:
                    problems.append(
                        f"回读会话失败：HTTP {listed.status_code}，"
                        f"无法确认 {session_id} 是否已删除"
                    )
            except Exception as exc:  # pylint: disable=broad-except
                problems.append(f"回读会话时出错：{exc!r}")

    # ---- 3. 智能体 -----------------------------------------------------------
    # ⚠️ 无论会话那一段成功与否都要走到这里：删得掉一个是一个。
    if agent_id:
        _delete(f"/agent/{agent_id}", f"智能体 {agent_id}")
        try:
            listed = client.get(f"{base_url}/agent/", headers=headers)
            if listed.status_code == 200:
                entries = listed.json().get("agents", [])
                remaining = [str(entry.get("id")) for entry in entries]
                if agent_id in remaining:
                    problems.append(f"回读核验失败：智能体 {agent_id} 删除后仍能列出")
                elif entries and all("id" not in entry for entry in entries):
                    # ⚠️ 与上面会话那条同一个坑的另一面：这里**必须**按 id 认人
                    #    （该用户名下还可能有别的智能体，「列表空了」不成立）。
                    #    但万一框架把 id 字段改了名，`remaining` 会安静地
                    #    变成一串 "None"、断言照样通过 —— 于是这里补一句：
                    #    响应体里根本没有 id 字段时，明说「无法确认」，
                    #    而不是假装核验过了。
                    problems.append(
                        "回读智能体失败：/agent/ 的响应体里没有 `id` 字段，"
                        f"无法确认 {agent_id} 是否已删除（框架响应格式变了吗？）"
                    )
            else:
                problems.append(
                    f"回读智能体失败：HTTP {listed.status_code}，"
                    f"无法确认 {agent_id} 是否已删除"
                )
        except Exception as exc:  # pylint: disable=broad-except
            problems.append(f"回读智能体时出错：{exc!r}")

    return problems


def _append_cleanup_note(
    detail: str,
    problems: list[str],
    created: dict[str, str],
    keep: bool,
) -> str:
    """把清理结果接到检查详情后面。

    ⚠️ **成功时也要写一句**（「已清理本轮产物」），不能只在失败时写。
    只写失败的话，「绿色行上没有这句话」就成了一条无法解释的线索：
    是清理成功但没打印？还是这一轮压根没建东西？还是清理根本没被调用？
    写一句确定的话，它的**缺席**才成为一个信号。

    Args:
        detail (`str`): 原有详情。
        problems (`list[str]`): :func:`_cleanup_artifacts` 返回的问题。
        created (`dict[str, str]`): 本轮登记的产物 id。
        keep (`bool`): 是否被要求保留产物。

    Returns:
        `str`: 追加了清理说明的详情。
    """
    if not created:
        # 什么都还没建就失败了（例如没有可用模型）—— 此时说「已清理」是假话。
        return detail
    if keep:
        ids = "；".join(f"{key}={value}" for key, value in created.items())
        return (
            f"{detail}\n    ℹ️ 按要求**保留**了本轮产物（--keep-artifacts）：{ids}"
        )
    if problems:
        return (
            f"{detail}\n    ⚠️ 本轮产物**未清理干净**（对话检查的判定不受影响，"
            "但库里会多出残留）：" + "；".join(problems) +
            f"\n    手工清理：以 {SMOKE_USER_ID} 的身份删除上面列出的 id。"
        )
    return f"{detail}\n    🧹 已清理本轮产物（会话 + 智能体，并已回读确认）"


def _body_excerpt(response: httpx.Response, limit: int = 200) -> str:
    """截取响应体的一段用于展示。

    ⚠️ 截断是必须的：这里打印的是**任意**响应体，包含 500 的栈信息。
        把整段栈打进冒烟报告会把真正的结论淹没在几十行里。

    Args:
        response (`httpx.Response`): 目标响应。
        limit (`int`): 最多保留的字符数。

    Returns:
        `str`: 截断后的响应体（单行化，避免多行栈撑爆报告）。
    """
    text = response.text.replace("\n", " ").strip()
    return text[:limit] + ("…" if len(text) > limit else "")


# ==============================================================================
# 主流程
# ==============================================================================
def wait_until_ready(client: httpx.Client, base_url: str) -> Check:
    """轮询 ``/readyz`` 直到就绪或超时。

    为什么需要它（`make up` 不是已经等过了吗）：
    本脚本也会被手工调用（`make smoke` 在服务重启后、在 CI 里对已部署环境），
    那时服务可能正处在「进程起来了但还在建连」的窗口期。
    不多等这一下的话，一次正常的启动过程会被报成一次失败的冒烟测试 ——
    而假失败会让人开始怀疑脚本本身，最终没人再看它的输出。

    Args:
        client (`httpx.Client`): HTTP 客户端。
        base_url (`str`): 服务地址。

    Returns:
        `Check`: 就绪等待的结果（超时则 ok=False）。
    """
    deadline = time.monotonic() + READY_WAIT_SECONDS
    last_status: Any = "<未发出请求>"

    while time.monotonic() < deadline:
        try:
            response = client.get(f"{base_url}/readyz")
            last_status = response.status_code
            if response.status_code == 200:
                return Check("等待就绪（轮询 /readyz）", True, "已就绪")
        except Exception as exc:  # pylint: disable=broad-except
            last_status = f"{type(exc).__name__}: {exc}"

        time.sleep(READY_POLL_INTERVAL_SECONDS)

    return Check(
        "等待就绪（轮询 /readyz）",
        False,
        f"等待 {READY_WAIT_SECONDS:.0f} 秒后仍未就绪（最后一次：{last_status}）。\n"
        f"    下一步：`curl -s {base_url}/readyz | python -m json.tool` 看是哪一项没过；\n"
        f"    再 `docker compose logs --tail=100 app` 看应用侧的原因。",
    )


def run_smoke(base_url: str, *, keep_artifacts: bool = False) -> Report:
    """执行整轮冒烟测试。

    Args:
        base_url (`str`): 服务根地址（不带结尾斜杠）。
        keep_artifacts (`bool`): 透传给 :func:`check_conversation` ——
            ``True`` 时保留对话检查建出的智能体与会话（默认删掉）。

    Returns:
        `Report`: 全部检查的结果。
    """
    base_url = base_url.rstrip("/")
    report = Report()

    print(f"▶ 冒烟目标：{base_url}")
    print(f"▶ 请求超时：{REQUEST_TIMEOUT_SECONDS:.0f}s；就绪等待上限：{READY_WAIT_SECONDS:.0f}s")
    print()

    with httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS, follow_redirects=False) as client:
        # 第一步：存活。失败即中止 —— 后面的检查都会以「连不上」失败，
        # 那一串次生失败只会淹没真正的原因。
        health = check_healthz(client, base_url)
        report.add(health)
        if not health.ok:
            print("\n（存活探针失败，后续检查已跳过 —— 它们必然同样失败，只会淹没原因。）")
            return report

        # 第二步：等待就绪。同样中止：未就绪时业务接口不可用，
        # 继续检查会得到一堆误导性的 503。
        ready = wait_until_ready(client, base_url)
        if not ready.ok:
            report.add(ready)
            print("\n（服务未就绪，后续检查已跳过。）")
            return report
        report.add(ready)

        # 第三步：逐项验证。这些彼此独立，一项失败不阻止下一项 ——
        # 一次跑完拿到完整清单，比修一个跑一次高效得多。
        report.add(check_readyz(client, base_url))
        report.add(check_metrics(client, base_url))
        report.add(check_trace_id(client, base_url))
        report.add(check_travel_api(client, base_url))
        report.add(check_identity_endpoint(client, base_url))
        # ★ 最后一项、也是最重要的一项：真的发一句话。
        # 放在最后是因为它最慢（要真的走一轮模型调用），
        # 前面的快速检查先跑完，能让失败信息更快浮现。
        # 它建的智能体与会话由自己负责删掉（见 check_conversation）。
        report.add(
            check_conversation(client, base_url, keep_artifacts=keep_artifacts)
        )

    return report


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    Returns:
        `argparse.ArgumentParser`: 解析器。
    """
    parser = argparse.ArgumentParser(
        description="AliGo 差旅助手 —— 对已启动的服务执行冒烟测试。",
        epilog=(
            "示例：\n"
            "  python scripts/smoke.py\n"
            "  python scripts/smoke.py --base-url http://localhost:8010\n"
            "  python scripts/smoke.py --keep-artifacts   # 留现场排查\n"
            "  make smoke SMOKE_URL=http://localhost:8010\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="服务根地址（默认 http://localhost:8000，与 Makefile 的 SMOKE_URL 一致）",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help=(
            "保留对话检查建出的智能体与会话，不删除（默认会用完即删，"
            "并把删除结果写进该项检查的详情）。排查问题时才需要。"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv (`list[str] | None`): 命令行参数；``None`` 时取 ``sys.argv``。

    Returns:
        `int`: 0 表示全部通过，1 表示有失败。
    """
    args = _build_parser().parse_args(argv)
    report = run_smoke(args.base_url, keep_artifacts=args.keep_artifacts)

    print()
    passed = len(report.checks) - len(report.failed)
    if report.failed:
        print(f"❌ 冒烟失败：{passed}/{len(report.checks)} 项通过")
        print("   未通过：")
        for check in report.failed:
            print(f"     · {check.name}")
        return 1

    print(f"✅ 冒烟通过：{passed}/{len(report.checks)} 项全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
