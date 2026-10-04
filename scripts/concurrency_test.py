#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""并发正确性测试：多个用户**同时**发**不同**的请求，验证互不干扰、全部成功。

==============================================================================
它回答的问题，与 smoke / loadtest 是两个不同的
==============================================================================
    ``make smoke``      一个人、一发、是/否          「这套部署能不能用」
    ``make loadtest``   一个人、同一个端点、几千发    「并发下延迟与失败率的**分布**」
    ``make concurrency`` 多人、同一时刻、内容各异     「同时用的时候，数据会不会串、
                                                     有没有人被打断」
    （本脚本。）

    ⚠️ 前两者的绿灯**推不出**后者，这不是理论上的可能，而是有具体机制：

      · ``loadtest.py`` 的模块注释里写死了「**本脚本不发送请求体**」——
        它只会发 GET。于是「三个用户各自建会话、各发各的话」这条路径，
        它**一毫秒都没有覆盖过**。而线上一次真正的并发，几乎必然发生在这里。
      · ``smoke.py`` 是**串行**的。它跑十遍，十遍之间也没有任何重叠，
        对「同时」这个维度覆盖为零。
      · 两者都全绿，仍然可能存在「A 的回复里混进了 B 的问题」这类缺陷。
        而这类缺陷只要发生一次，用户就不会再信任这个系统 ——
        它不是「慢一点」，是**串号**。

==============================================================================
三个用户各发什么，以及为什么是这三条
==============================================================================
    判据只有一条：**三条走的是互不相同的代码路径**。三个用户发同一句话，
    测的是「同一段代码能不能重入」；三条不同路径同时跑，才测得到
    「他们会不会互相看见对方的中间状态」。因此取：

      用户 1  ``住宿标准``
              快车道。规则**精确命中**，不经过意图识别模型 —— 入口就与另两条不同。
      用户 2  ``我上个月订的那张去广州的机票现在是什么状态``
              慢车道单意图 QUERY_ORDER，要调 ``query_orders`` 查订单。
      用户 3  ``去北京出差住宿费能报多少``
              慢车道单意图 QUERY_POLICY，要调 ``check_travel_policy`` 核差标。

    ⚠️ **这两个工具读的都是内存仓储 —— 不是 Postgres，更不是 Milvus。**
    ``src/server/agents_factory.py:162-168`` 给的是一整包内存实现（该文件
    自己写着「P3 阶段是内存实现，P4 换成 Postgres」），而且
    ``policy=StaticPolicyRepository()`` 是**无参**构造 ⇒ 所有用户一视同仁
    拿 ``DEFAULT_POLICY_LIMIT``（经济舱 / 酒店 600 元/晚 / 机票 2000 元）。

    本脚本第一版把用户 3 的链路写成「**政策库**（Milvus 向量检索那条链路）」，
    **那是错的**。实测反证：21 条轨迹每条恰好 1 次 embedding，且全部是
    消息记忆召回（``src/memory/semantic.py``），``knowledge_documents`` 为 0
    —— 没有任何 RAG 查询向量化发生。写错的代价很具体：读的人会以为
    「RAG 链路的并发行为已经测过了」，而它**根本没被这个脚本覆盖**。
    本脚本覆盖的是快车道 / 慢车道订单 / 慢车道差标这三条。

    三条都对**只读**工具，不写业务数据、不触发人工确认 —— 并发测试本身
    不该产生副作用，否则「测出问题」与「测试把它弄坏了」将无法区分。
    （三条问法取自 ``tests/evaluation/golden_dataset.yaml``，与其标注的
    lane/intents/tools 一致。）

    智能体刻意建成 ``main_plan``（``src/agents/prompts.py`` 的常量）：
    快慢车道、动态 Prompt 都只对**这个名字**生效。随便起个名字也能聊天，
    但那是在测一个**功能被静默关掉**的系统。

==============================================================================
「成功」在这里具体指什么（十二条判据，逐条可证伪）
==============================================================================
    0. 全部用户准备就绪  模型配置、智能体、会话都建好了。
                       ⚠️ 单列这一条，是因为「准备失败」与「对话失败」的**处置方向
                       完全不同**：前者是环境/配置问题，后者才是并发问题。
    1. 全部正常结束    每个人都收到 REPLY_END，且 ``finished_reason == completed``。
                       ⚠️ 只判「收到 REPLY_END」是不够的：``ERROR`` /
                       ``exceed_max_iters`` / ``interrupted`` 也都会发 REPLY_END。
    2. 回复非空        每个人都真的收到了文本，不是一串空事件。
  2b. 回复是答复而非占位
                       ★ 有文本 ≠ 有用。两个探测器：
                         · 命中「已交给别人、稍等」这类措辞且通篇没有数字
                           ⇒ **占位**，判**失败**（用户什么都没拿到）；
                         · 回复以内心独白式措辞开头（``用户想…`` / ``I'll…``）
                           ⇒ **告警**，**不**计入失败（用户拿到了答案，
                           多出来的是一段模型的草稿，属输出质量范畴）。
                       判据 2 与 2b 是**互为补角**的：2 管「有没有字」，
                       2b 管「那些字是不是答复」。
                       ⚠️ 两个形状都来自真实的假绿灯，详见
                       :func:`check_replies_are_answers` 的文档。
                       **可复核的频次以那次文档为准**（别再往这里写「撞到 N 次」——
                       历史次数在三个文件里曾经各写一个数、互相矛盾，而原始日志
                       根本不记录回复正文，谁也验不了）。
                       ⚠️ 分级不是为了让测试变绿，是为了**不让并发结论取决于
                       模型文风** —— 理由写在 :class:`Check` 的 ``warn`` 上。
    3. 无错误事件      流里没有 error 字段、没有 ``REQUIRE_USER_CONFIRM``
                       （只读问句不该弹确认）等异常事件。
    4. 确实同时        ★ 见下一节。
    5. 流不串号        ★ 每个事件带的 ``session_id`` 都等于**自己**的会话。
    6. 记载不串号      ★ 事后读回的会话历史里，用户消息**逐字**等于自己的问题；
                       且任何人的历史里都**不出现**别人的问题原文。
    7. 跨用户不可见    用 A 的身份去读 B 的会话/智能体，必须 404
                       （不是 403：403 等于承认「它存在」，对陌生人连存在性
                       都不该确认）。
    8. 无服务端错误    全程没有 5xx、没有 409（409 = 同一会话有对话在飞，
                       三个用户各用各的会话，本不该出现）。
    9. 业务接口身份正确 ``/api/v1/me`` 在并发下也答自己，没有串。
   10. 产物已清理      本脚本建的东西全删了，且**回读确认**过。

    ⚠️ **判据 6、7 必须在清理之前跑**（代码里是 ``_run_live_checks`` 与
    ``_cleanup_all`` 的先后）。它们读的是**服务端现存的**会话记录，删掉之后
    读回来只剩 404 —— 一条本该通过的判据会报「会话不存在」，而那个原因
    看上去与「串号」毫无关系。本脚本的第一版正是如此：清理在前、取证在后，
    实测三条会话全部 404。**判据没错，是取证顺序错了。**

    第 5、6 条是**互为补角**的两件事，缺一条就漏一类缺陷：
      · 只查 5（流）：能抓住「推流时推错了人」，抓不住「落库时落错了人」；
      · 只查 6（库）：能抓住「存错了」，抓不住「存对了但推给了别人」。
    所以两条都要。

    ⚠️ 编号里那个 ``2b`` 是**故意的**，不是笔误。它挤在 2 与 3 之间，
    因为它在语义上就长在那里（「有文本」的下一步是「文本有用」）；
    后面 3~10 的号**一个都没有往后挪**，于是所有已经写进文档/工单里的
    「判据 6」仍然指的是同一件事。**改编号会让别人的记录失效**，
    所以宁可让列表看起来不齐整。

==============================================================================
★ 「确实同时」是怎么证明的（本脚本最容易做成假绿灯的地方）
==============================================================================
    一个用线程池跑三个 HTTP 请求的脚本，完全可能因为 GIL、连接池、或
    单纯写错了顺序，而**实际上是串行**的 —— 而它照样会打印「3/3 通过」。
    那样的报告是假绿灯里最坏的一种：它让人以为并发被验证过了。

    所以本脚本不靠「我用了 threading」这句话，而是记录**每人的时间窗**：

        t_chat  = 发出 ``POST /chat/`` 的时刻
        t_end   = 收到 REPLY_END 的时刻

    然后用一条不等式判定：

        max(t_chat) < min(t_end)

    即**最后一个人的请求发出时，第一个人的回复还没结束** —— 三个人在
    同一时刻都处于「服务端正在为他干活」的状态。这是「同时」的直接证据，
    而不是它的推论。（阈值只判重叠，不判重叠了多久：重叠 1 毫秒和 100 秒
    在这条判据下都算通过，本脚本不假装能测出前者。）

    ⚠️ **这条判据比它看起来弱，界线要说清楚。** 它排除的是「脚本自己按顺序
    发请求」（那才是最容易被写出来的假并发）；它**排除不了**「请求同时到达、
    但服务端串行处理」—— 在一个严格串行的服务端上，三个请求先后进入队列，
    `max(t_chat)` 仍然小于 `min(t_end)`（都没结束），判据照样通过。

    而且栅栏把这一点**推到极致**：栅栏保证三个 `POST /chat/` 几乎同时发出，
    于是 `max(t_chat)` 与 `min(t_end)` 的差距主要来自「服务端要花多久」，
    而不是「有没有重叠」—— 它几乎必然成立。

    ⇒ 真正强的证据在**服务端一侧**，本脚本给不出来（它是个客户端脚本，
    看不到进程内部的调度）。核验办法（实测跑过，6 用户 × 2 轮）：

      1. app 是**单进程 asyncio**（``--workers 1``），先确认这一点：
         ``docker compose exec app cat /proc/7/cmdline`` 一类的办法看到
         ``uvicorn … --workers 1``。多进程的话下面的推理就不成立了。
      2. 从 ``docker compose logs app`` 取每条 ``GET /sessions/*/stream`` 的
         ``→ 200（Nms）``，用**结束时刻 − N** 还原出开始时刻；
      3. 结果：12 条流的服务端耗时合计 **253.1s**，对应的墙钟跨度只有 **45.0s**
         ⇒ **5.6×**；同一时刻打开的流**峰值 6 条**、持续 40 秒。

    串行服务端上倍率只会是 ~1.0×、峰值恒为 1（第二个请求要等第一个流关闭）。
    所以这两条数字一摆出来，「服务端真的在同时干活」就不再是推论。
    跑完自己核一遍，别只信本脚本的绿灯。

==============================================================================
不留痕
==============================================================================
    与 ``smoke.py`` 同一条纪律：本脚本建出的智能体与会话，跑完自己删掉 ——
    成功、失败、异常路径都删（见 ``_cleanup_all``）。要留现场用 ``--keep``。

    ⚠️ 只删**自己建的**。若目标用户已经有 ``main_plan`` 智能体，本脚本
    **复用**它而不是另建一个（``provision_agent.py`` 的同一套幂等语义），
    并且清理时**不碰**它 —— 删掉别人正在用的智能体，比留一条垃圾会话严重得多。

    ⚠️ 清理的边界与 ``smoke.py`` 相同：``finally`` 覆盖不到 ``kill -9``
    与默认 SIGTERM。被强杀时残留仍可能出现。

==============================================================================
一次通过**不**等于「扛得住并发」（别把本脚本当容量证明）
==============================================================================
    它只说明「3~6 个用户、各 1~2 轮、这套负载下没有异常」。已知**未覆盖**的：

      · 连接池上限（单 engine 峰值 30 连接，N≤6 离得很远，耗尽路径没走到）；
      · 限流 429（N≤6 打不出来，「没有 429」是**因为没触发**，不是因为它工作正常）；
      · 同会话并发（每人各建一条会话，``POST /chat/`` 的 409 是**刻意避开**的）；
      · 业务库 / 向量库争用（订单与差标走**内存仓储**，见 ``agents_factory.py``）；
      · 跨用户隔离只验了 GET，**没验**「A 能不能删掉 B 的会话」这类写越权；
      · 进程级崩溃（OOM / 段错误 / 被 kill）——观测到的失败全是「返回了错误码」。

    真正的并发证据在**服务端**，本脚本给不出来。核验办法见模块开头
    「确实同时」一节，以及 ``docs/06-部署与运维.md``。

退出码约定：**0 = 十二条判据全过；1 = 有任何一条没过**。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

# 允许以 `python scripts/concurrency_test.py` 直接运行（此时 sys.path[0] 是 scripts/）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.prompts import MAIN_AGENT_NAME, prompt_for  # noqa: E402

# ==============================================================================
# 常量
# ==============================================================================

#: 默认靶地址。与 Makefile 的 ``CONCURRENCY_URL``（默认 http://localhost:8000）一致。
DEFAULT_BASE_URL = "http://localhost:8000"

#: 默认并发用户数。3 是任务书要求的规模，也是「能看出串号」的最小规模 ——
#: 2 个用户只能验一次配对，3 个用户才有「A 混进 B」与「A 混进 C」两组以上组合。
DEFAULT_USERS = 3

#: 默认轮数。每人依次发 ``--rounds`` 条消息；第 k 轮由**全部用户同时**发出。
DEFAULT_ROUNDS = 1

#: 非流式请求（建智能体、建会话、读历史）的超时。
REQUEST_TIMEOUT_SECONDS = 10.0

#: 一轮对话的等待上限。
#:
#: ⚠️ 比 ``smoke.py`` 的 60 秒宽：并发场景下三个请求**同时在等同一个模型**，
#: 单发时 20 秒能回的对话，三发同时时排队到 40 秒是正常的服务行为，不是故障。
#: 把这里调小只会制造假失败。
CONVERSATION_TIMEOUT_SECONDS = 120.0

#: 等待 REPLY_END 时的轮询间隔。
CONVERSATION_POLL_INTERVAL_SECONDS = 0.25

#: 订阅 SSE 之后、触发对话之前的静默等待。
#:
#: ⚠️ 订阅是**异步**建立的：``stream()`` 返回时连接可能还没在服务端登记完，
#: 此刻就 ``POST /chat/`` 会撞上「事件已经推完，但我还没挂上去」——
#: 表现是流里空空如也。浏览器是先连流、后发消息，这里复刻同一个时序。
#: 与 ``smoke.py`` 取同一个值，不是巧合：两处面对的是同一个竞态。
STREAM_SETTLE_SECONDS = 1.0

#: 清理/回读时，这些状态码都算「目标已经不在」（见 ``_delete``）。
#: ⚠️ 这里的 404 是**故意放宽**的（删两次、或目标本就不在都算正常），代价是它
#: 也掩盖「路由不存在 / id 不被认」—— 所以会话删完必须再回读一次列表，
#: 见 ``_readback_sessions``。只看状态码不足以支撑判据 10 的结论。
_ALREADY_GONE_STATUSES = (200, 204, 404)


@dataclass(frozen=True)
class Case:
    """一个用户这一轮要发的东西，以及它为什么值得发。

    Attributes:
        key (`str`): 短标识，用于报告里的标签。
        label (`str`): 人类可读的场景名。
        prompt (`str`): 发给智能体的原文。
        route (`str`): 这句话预期走哪条链路（**只用于报告**，不做断言 ——
            断言「该走快车道」是 ``classifier`` 单测的职责，在这里断言等于
            把并发测试变成第二份选路测试，它失败时也说明不了是不是并发的问题）。
    """

    key: str
    label: str
    prompt: str
    route: str


#: 三条默认场景。改动前请先读模块文档的「三个用户各发什么」一节。
DEFAULT_CASES: tuple[Case, ...] = (
    Case(
        key="fast",
        label="快车道 · 政策词条",
        prompt="住宿标准",
        route="规则精确命中，不经意图识别模型",
    ),
    Case(
        key="order",
        label="慢车道 · 订单查询",
        prompt="我上个月订的那张去广州的机票现在是什么状态",
        route="慢车道 QUERY_ORDER → query_orders（内存订单仓储）",
    ),
    Case(
        key="policy",
        label="慢车道 · 政策问答",
        prompt="去北京出差住宿费能报多少",
        route="慢车道 QUERY_POLICY → check_travel_policy（内存差标仓储，非向量检索）",
    ),
)

#: 用户名的前缀。刻意用**一次性**工作区：清理逻辑一旦写错，破坏的只是
#: 「并发测试专用」的数据，而不是某个真人用户的工作区。
USER_PREFIX = "concurrency-user"


# ==============================================================================
# 结果模型
# ==============================================================================


@dataclass
class Turn:
    """一个用户的一轮对话（发出问题 → 收到 REPLY_END）的全部观测。"""

    round_index: int
    t_chat: float = 0.0
    """发出 ``POST /chat/`` 的单调时刻。"""
    t_end: float = 0.0
    """收到 REPLY_END 的单调时刻；没收到则保持 0。"""
    chat_status: int | None = None
    """``POST /chat/`` 的状态码。"""
    chat_body: str = ""
    """``POST /chat/`` 非 2xx 时的响应体摘录（已脱敏）。"""
    events: list[dict[str, Any]] = field(default_factory=list)
    stream_error: str = ""
    """SSE 读取线程抛出的异常，或流以非 200 结束时的说明。"""
    stream_open: bool = False
    """SSE 是否成功建立（HTTP 200）。"""
    reply_text: str = ""
    finished_reason: str = ""
    error_info: str = ""
    #: 是否是**客户端等够了 120 秒自己走的**（而不是服务端把流结束了）。
    #: ⚠️ 这两种情况在报告上原先长得一模一样（都是「没收到 REPLY_END」），
    #: 但含义相反：前者可能是脚本判据太短、也可能是服务端真的卡住了；
    #: 后者是服务端明确结束了流却不出 REPLY_END —— 那是确定的协议违规。
    #: 不区分就只能靠猜，而「靠猜」在故障定位里等于没有信息。
    timed_out: bool = False
    #: 这一轮实际等了多久（秒）。超时时它约等于 ``CONVERSATION_TIMEOUT_SECONDS``。
    elapsed: float = 0.0

    @property
    def reached_end(self) -> bool:
        """是否收到了 REPLY_END。"""
        return any(e.get("type") == "REPLY_END" for e in self.events)

    @property
    def last_event_type(self) -> str:
        """流里最后一个事件的类型 —— 卡住时，它说明卡在**哪一步**。

        ``TOOL_RESULT_START`` 而没有 ``TOOL_RESULT_END`` = 卡在工具执行里；
        ``MODEL_CALL_START`` 而没有 ``MODEL_CALL_END`` = 卡在模型调用里。
        没有这个字段时，「没收到 REPLY_END」只能说明「结果不对」，说不出原因。
        """
        return self.event_types[-1] if self.events else "（一个事件都没有）"

    @property
    def event_types(self) -> list[str]:
        """事件类型序列（报告里只打印前若干个）。"""
        return [str(e.get("type")) for e in self.events]


@dataclass
class UserRun:
    """一个用户的全程观测：身份、产物、以及每一轮的 Turn。"""

    index: int
    user_id: str
    case: Case
    agent_id: str = ""
    agent_created: bool = False
    session_ids: list[str] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    setup_error: str = ""
    """建模型配置/智能体/会话阶段失败时的原因；非空则这一轮不会发出对话。"""

    @property
    def label(self) -> str:
        """报告里的行首标签。"""
        return f"用户 {self.index}（{self.user_id}）· {self.case.label}"


# ==============================================================================
# HTTP 小工具
# ==============================================================================


def _headers(user_id: str) -> dict[str, str]:
    """构造带身份的请求头。

    Args:
        user_id (`str`): 用户标识。

    Returns:
        `dict[str, str]`: 请求头。
    """
    return {"X-User-ID": user_id}


def _body_excerpt(response: httpx.Response, limit: int = 200) -> str:
    """截取响应体用于报错。

    ⚠️ 只截前 ``limit`` 个字符。响应体里**可能有凭据**（``/api/v1/default-model``
    就会返回 ``chat_model_config``），所以本函数返回的内容只用于错误行，
    且截断本身就是一层兜底 —— 不要把整段 body 打进日志。

    Args:
        response (`httpx.Response`): 响应。
        limit (`int`): 最大字符数。

    Returns:
        `str`: 摘录（单行）。
    """
    text = " ".join(response.text.split())
    return text[:limit] + ("…" if len(text) > limit else "")


def _client(timeout: float = REQUEST_TIMEOUT_SECONDS) -> httpx.Client:
    """建一个非流式客户端。

    Args:
        timeout (`float`): 总超时秒数。

    Returns:
        `httpx.Client`: 客户端。
    """
    return httpx.Client(timeout=timeout)


# ==============================================================================
# 单用户的准备阶段
# ==============================================================================


def _resolve_model_config(
    client: httpx.Client, base_url: str, user_id: str
) -> tuple[dict[str, Any] | None, str]:
    """取这个用户可用的模型配置。

    Args:
        client (`httpx.Client`): HTTP 客户端。
        base_url (`str`): 服务地址。
        user_id (`str`): 用户身份。

    Returns:
        `tuple[dict | None, str]`: ``(配置, 错误说明)``。成功时错误说明为空串。
    """
    response = client.get(f"{base_url}/api/v1/default-model", headers=_headers(user_id))
    if response.status_code != 200:
        return None, (
            f"``GET /api/v1/default-model`` 返回 {response.status_code}："
            f"{_body_excerpt(response)}"
        )
    body = response.json()
    config = body.get("chat_model_config")
    if not config:
        return None, (
            f"这个用户没有可用的模型配置（mode={body.get('mode')!r}）。"
            f"\n    服务端提示：{body.get('hint')}"
            "\n    ⚠️ 并发测试必须能拿到一份**确定的**配置才能继续，"
            "否则三个人会在同一行上空转。"
        )
    return config, ""


def _find_agent(client: httpx.Client, base_url: str, user_id: str, name: str) -> str:
    """按名字找一个已存在的智能体，找不到返回空串。

    Args:
        client (`httpx.Client`): HTTP 客户端。
        base_url (`str`): 服务地址。
        user_id (`str`): 用户身份。
        name (`str`): 智能体名。

    Returns:
        `str`: 智能体 id；不存在则空串。
    """
    response = client.get(f"{base_url}/agent/", headers=_headers(user_id))
    if response.status_code != 200:
        return ""
    for item in response.json().get("agents", []):
        data = item.get("data") or {}
        if data.get("name") == name:
            return str(item.get("id") or data.get("id") or "")
    return ""


def _prepare_user(
    base_url: str, run: UserRun, barrier: threading.Barrier
) -> None:
    """准备阶段：取模型配置 → 确保有 ``main_plan`` 智能体 → 建一条新会话。

    ⚠️ 这一步也放在**并发**里跑（所有线程同时进来），因为它压的是
    存储层的建对象路径 —— 「三个人同时建智能体」本身就是一个要被测的场景。

    Args:
        base_url (`str`): 服务地址。
        run (`UserRun`): 出参。就地写入 ``agent_id`` / ``agent_created`` /
            ``session_ids``，或 ``setup_error``。
        barrier (`threading.Barrier`): 准备阶段的同步点。
    """
    try:
        with _client() as client:
            config, error = _resolve_model_config(client, base_url, run.user_id)
            if config is None:
                run.setup_error = error
                return

            # ⚠️ 名字必须是 main_plan：快慢车道与动态 Prompt 都只对这个名字生效。
            existing = _find_agent(client, base_url, run.user_id, MAIN_AGENT_NAME)
            if existing:
                # 复用而不是另建 —— 与 provision_agent.py 同一套幂等语义。
                run.agent_id = existing
                run.agent_created = False
            else:
                created = client.post(
                    f"{base_url}/agent/",
                    headers=_headers(run.user_id),
                    json={
                        "name": MAIN_AGENT_NAME,
                        "system_prompt": prompt_for(MAIN_AGENT_NAME),
                    },
                )
                if created.status_code not in (200, 201):
                    run.setup_error = (
                        f"建智能体失败：{created.status_code} {_body_excerpt(created)}"
                    )
                    return
                run.agent_id = str(created.json()["agent_id"])
                run.agent_created = True

            session = client.post(
                f"{base_url}/sessions/",
                headers=_headers(run.user_id),
                json={
                    "agent_id": run.agent_id,
                    "name": "并发测试会话 r1",
                    "chat_model_config": config,
                },
            )
            if session.status_code not in (200, 201):
                run.setup_error = (
                    f"建会话失败：{session.status_code} {_body_excerpt(session)}"
                )
                return
            run.session_ids.append(str(session.json()["session_id"]))
    except Exception as exc:  # pylint: disable=broad-except
        run.setup_error = f"{type(exc).__name__}: {exc}"
    finally:
        # ⚠️ 无论如何都要放行，否则一个线程在准备阶段就失败时，
        #    其余线程会在 barrier 上**永久等待** —— 测试挂死，而不是失败。
        #    「挂死」比「失败」糟得多：它不给出任何可读的原因。
        try:
            barrier.wait(timeout=REQUEST_TIMEOUT_SECONDS * 3)
        except threading.BrokenBarrierError:
            pass


# ==============================================================================
# 单用户的一轮对话
# ==============================================================================


def _chat_once(
    base_url: str, run: UserRun, session_id: str, turn: Turn, barrier: threading.Barrier
) -> None:
    """订阅 SSE → 与其他人对齐 → 触发对话 → 读到 REPLY_END。

    ⚠️ 读取放在后台线程里，与 ``smoke.py`` 同一个理由：``POST /chat/`` 是
    同步调用，主线程必须能同时等它和等流。

    Args:
        base_url (`str`): 服务地址。
        run (`UserRun`): 所属用户。
        session_id (`str`): 这一轮的会话 id。
        turn (`Turn`): 出参，就地写入观测。
        barrier (`threading.Barrier`): 触发时刻的同步点 —— 这是「同时」的实现处。
    """
    headers = _headers(run.user_id)
    done = threading.Event()

    def _read_stream() -> None:
        """在后台线程里读 SSE，直到 REPLY_END 或流结束。"""
        try:
            # ⚠️ 独立读超时：共用那个 10 秒的客户端会让空闲的 SSE 连接
            #    每隔 10 秒被判超时（与 smoke.py 同一个坑）。
            with httpx.Client(
                timeout=httpx.Timeout(REQUEST_TIMEOUT_SECONDS, read=CONVERSATION_TIMEOUT_SECONDS)
            ) as stream_client:
                with stream_client.stream(
                    "GET",
                    f"{base_url}/sessions/{session_id}/stream",
                    params={"agent_id": run.agent_id},
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        turn.stream_error = f"SSE 返回 {response.status_code}"
                        return
                    turn.stream_open = True
                    for line in response.iter_lines():
                        if done.is_set():
                            return
                        if not line.startswith("data:"):
                            continue
                        try:
                            payload = json.loads(line[len("data:") :].strip())
                        except json.JSONDecodeError:
                            continue
                        turn.events.append(payload)
                        if payload.get("type") == "REPLY_END":
                            return
        except Exception as exc:  # pylint: disable=broad-except
            turn.stream_error = f"{type(exc).__name__}: {exc}"
        finally:
            # ⚠️ 与 smoke.py 同一条修复：线程一结束，这个流就不可能再送来
            #    REPLY_END 了，主线程没有理由继续空等满 120 秒。
            done.set()

    reader = threading.Thread(target=_read_stream, daemon=True)
    reader.start()

    # 先把流挂上去，再和所有人对齐后同时发 —— 顺序不能反（见 STREAM_SETTLE_SECONDS）。
    time.sleep(STREAM_SETTLE_SECONDS)

    try:
        barrier.wait(timeout=CONVERSATION_TIMEOUT_SECONDS)
    except threading.BrokenBarrierError:
        done.set()
        turn.stream_error = turn.stream_error or "同步点被打破，本轮未发出"
        return

    # ★ 这就是「同时」的定义处：下面这行由所有线程在同一个栅栏释放后立刻执行。
    turn.t_chat = time.monotonic()
    try:
        with _client() as client:
            trigger = client.post(
                f"{base_url}/chat/",
                headers=headers,
                json={
                    "agent_id": run.agent_id,
                    "session_id": session_id,
                    "input": {
                        "name": "user",
                        "role": "user",
                        "content": [{"type": "text", "text": run.case.prompt}],
                    },
                },
            )
        turn.chat_status = trigger.status_code
        if trigger.status_code not in (200, 202):
            turn.chat_body = _body_excerpt(trigger)
    except Exception as exc:  # pylint: disable=broad-except
        turn.chat_body = f"{type(exc).__name__}: {exc}"

    deadline = time.monotonic() + CONVERSATION_TIMEOUT_SECONDS
    while not done.is_set() and time.monotonic() < deadline:
        done.wait(timeout=CONVERSATION_POLL_INTERVAL_SECONDS)
    turn.t_end = time.monotonic()
    turn.elapsed = turn.t_end - turn.t_chat
    # ⚠️ 「客户端等到 120 秒自己走」与「服务端结束了流却没给 REPLY_END」
    #    是两回事，必须分开记。判据 1 的详情据此给出不同的处置方向。
    turn.timed_out = not done.is_set()
    done.set()  # 让读取线程尽快退出

    _summarize(turn)


def _summarize(turn: Turn) -> None:
    """从事件流里抽取回复文本、结束原因与错误信息。

    Args:
        turn (`Turn`): 出参，就地补齐 ``reply_text`` / ``finished_reason`` / ``error_info``。
    """
    chunks: list[str] = []
    for event in turn.events:
        if event.get("type") == "TEXT_BLOCK_DELTA":
            text = event.get("text") or event.get("delta") or ""
            if isinstance(text, str):
                chunks.append(text)
        elif event.get("type") == "REPLY_END":
            turn.finished_reason = str(event.get("finished_reason") or "")
            if event.get("error"):
                turn.error_info = json.dumps(event["error"], ensure_ascii=False)[:300]
    turn.reply_text = "".join(chunks)


def _run_user(base_url: str, run: UserRun, setup_barrier: threading.Barrier,
              chat_barriers: list[threading.Barrier]) -> None:
    """一个用户的完整线程体：准备 → 逐轮对话。

    Args:
        base_url (`str`): 服务地址。
        run (`UserRun`): 这个用户的观测。
        setup_barrier (`threading.Barrier`): 准备阶段的同步点。
        chat_barriers (`list[threading.Barrier]`): 每一轮的同步点。
    """
    _prepare_user(base_url, run, setup_barrier)
    if run.setup_error:
        return
    for index, barrier in enumerate(chat_barriers):
        # 每一轮用**新会话**：同一会话的第二次 ``POST /chat/`` 会撞上
        # 框架的 409 双提交保护（那是正确行为，但会把并发测试测成「一半人失败」）。
        session_id = run.session_ids[index] if index < len(run.session_ids) else ""
        turn = Turn(round_index=index + 1)
        run.turns.append(turn)
        if not session_id:
            # 准备阶段就少建了会话 —— 补建一条，让这一轮仍然有意义。
            session_id = _create_extra_session(base_url, run, index + 1)
            if not session_id:
                continue
        _chat_once(base_url, run, session_id, turn, barrier)


def _create_extra_session(base_url: str, run: UserRun, round_index: int) -> str:
    """为第 ``round_index`` 轮补建一条会话。

    Args:
        base_url (`str`): 服务地址。
        run (`UserRun`): 所属用户。
        round_index (`int`): 轮次（从 1 开始）。

    Returns:
        `str`: 会话 id；失败返回空串并写入 ``setup_error``。
    """
    try:
        with _client() as client:
            config, error = _resolve_model_config(client, base_url, run.user_id)
            if config is None:
                run.setup_error = run.setup_error or error
                return ""
            response = client.post(
                f"{base_url}/sessions/",
                headers=_headers(run.user_id),
                json={
                    "agent_id": run.agent_id,
                    "name": f"并发测试会话 r{round_index}",
                    "chat_model_config": config,
                },
            )
            if response.status_code not in (200, 201):
                run.setup_error = f"建会话失败：{response.status_code} {_body_excerpt(response)}"
                return ""
            session_id = str(response.json()["session_id"])
            run.session_ids.append(session_id)
            return session_id
    except Exception as exc:  # pylint: disable=broad-except
        run.setup_error = f"{type(exc).__name__}: {exc}"
        return ""


# ==============================================================================
# 并发执行
# ==============================================================================


def _build_runs(users: int, rounds: int) -> list[UserRun]:
    """按用户数与轮数构造观测对象。

    Args:
        users (`int`): 并发用户数。
        rounds (`int`): 每人轮数。

    Returns:
        `list[UserRun]`: 观测对象，顺序即用户编号顺序。
    """
    runs: list[UserRun] = []
    for index in range(1, users + 1):
        # ⚠️ 场景**轮转**而不是复制：用户 1..3 拿到三条不同的问法（任务要求），
        #    再加用户时回到第一条 —— 加压场景下「有几个人发同一句」是刻意的，
        #    它测的是同一段代码的并发重入。
        case = DEFAULT_CASES[(index - 1) % len(DEFAULT_CASES)]
        runs.append(
            UserRun(
                index=index,
                user_id=f"{USER_PREFIX}-{index}",
                case=case,
                turns=[],
            )
        )
    # ⚠️ 这里**不预建** Turn：Turn 由各线程在真正发出那一轮时才创建。
    #    预建再让线程去找「第几轮对应哪个对象」，等于把轮次和对象靠下标绑死，
    #    中途少跑一轮时，报告里的空 Turn 与「没发出去的那一轮」将无法区分。
    return runs


def run_concurrency(base_url: str, users: int, rounds: int) -> list[UserRun]:
    """让 ``users`` 个用户同时跑 ``rounds`` 轮。

    Args:
        base_url (`str`): 服务地址。
        users (`int`): 并发用户数。
        rounds (`int`): 每人轮数。

    Returns:
        `list[UserRun]`: 全部观测。
    """
    runs = _build_runs(users, rounds)
    setup_barrier = threading.Barrier(users)
    chat_barriers = [threading.Barrier(users) for _ in range(rounds)]

    threads = [
        threading.Thread(
            target=_run_user,
            args=(base_url, run, setup_barrier, chat_barriers),
            name=f"concurrency-{run.user_id}",
            daemon=True,
        )
        for run in runs
    ]
    for thread in threads:
        thread.start()
    # ⚠️ 每轮的超时都要留够：join 的时间上限必须 >= 单轮上限 × 轮数，
    #    否则「服务很慢」会被 join 提前掐断，报出来却像是「线程没跑完」。
    for thread in threads:
        thread.join(timeout=(CONVERSATION_TIMEOUT_SECONDS + 20) * rounds)
    return runs


# ==============================================================================
# 判据
# ==============================================================================


@dataclass
class Check:
    """一条判据的结果。

    Attributes:
        name (`str`): 判据名（打印在报告里）。
        ok (`bool`): 是否通过。
        detail (`str`): 通过时的证据摘要，或失败时的原因。
        warn (`bool`): 不通过，但**不因此判整轮失败** —— 报告里打 ⚠️，
            退出码不受它影响。

    ⚠️ ``warn`` 存在的理由，不是「想让测试变绿」，而是**这条判据里混了
    两种性质不同的东西**（判据 2b）：

      · **占位回复**（用户什么都没拿到）—— 这是**功能故障**，必须判失败；
      · **内心独白漏进回复**（用户拿到了答案，但前面多了一段
        「用户想查已有的机票订单，我直接查一下。」这类模型的草稿）——
        这是**输出质量缺陷**，属于 ``scripts/eval.py`` 的职责范围。

    把后者也判成失败，会让「并发正确性测试」的退出码取决于模型的文风，
    而**一个会因为文风随机变红的判据，在真实项目里的下场是被关掉** ——
    连它旁边那个真正重要的「占位」判定也一起失效。所以分开：
    故障判红、质量告警，两者都在报告里，都看得见。

    ⚠️ 与 ``scripts/smoke.py`` 的 ``Check.fatal`` 是同一类设计（严重程度
    挂在这条结果上，而不是散在打印代码里），只是方向相反：
    那个说「失败且别再往下跑」，这个说「有问题但不算失败」。
    """

    name: str
    ok: bool
    detail: str = ""
    warn: bool = False


def _all_turns(runs: list[UserRun]) -> list[tuple[UserRun, Turn]]:
    """把 (用户, 轮次) 摊平，供各判据遍历。

    Args:
        runs (`list[UserRun]`): 全部观测。

    Returns:
        `list[tuple[UserRun, Turn]]`: 摊平后的列表。
    """
    return [(run, turn) for run in runs for turn in run.turns]


def _nothing_observed(name: str, what: str) -> Check:
    """构造一条「零观测 ⇒ 不通过」的判据结果。

    ⚠️ 这条规则是**判据 5（SSE 流不串号）那次修正推广开来的**：一条判据
    如果在「什么都没测到」时返回 ✅，那它的绿灯是**空头支票** —— 报告上
    与「测了很多、全都对」长得一模一样，而两者的证据量差着数量级。

    为什么零观测在这些判据里是**可能的**，不是杞人忧天：本脚本的判据大多
    是「遍历所有观测，有问题就记一笔」。遍历到空集合时，「没有问题」与
    「没有观测」在代码上是同一件事 —— 只能显式区分。

    为什么零观测要判**失败**而不是「不适用」：这些判据的观测对象（轮次 /
    会话 / 越权探测）都产生于本脚本自己要走的流程。它们为空，说明**流程
    本身没走完**，那是必须有人看的问题，不是可以跳过的情形。
    （``--keep`` 造成的「不适用」是另一回事，见 :func:`check_cleanup`：
    那是**用户明确要求**的结果，所以它判通过并说明原因。）

    Args:
        name (`str`): 判据名。
        what (`str`): 为什么它是空的（写清「本该有什么」）。

    Returns:
        `Check`: 恒为 ``ok=False``。
    """
    return Check(
        name,
        False,
        f"⚠️ {what} —— 这条判据**什么都没验到**，不能当作通过。"
        f"（空集合上的「没问题」与「没看」是同一件事。）",
    )


def check_setup(runs: list[UserRun]) -> Check:
    """判据 0：每个用户都成功准备好（模型配置 + 智能体 + 会话）。

    ⚠️ 单列这一条，是因为「准备失败」和「对话失败」的**处置方向完全不同**：
    前者是环境/配置问题（去看 /api/v1/default-model 与建对象接口），
    后者才是并发问题。混在一起报，读的人会去错的地方找。

    Args:
        runs (`list[UserRun]`): 全部观测。

    Returns:
        `Check`: 结果。
    """
    if not runs:
        # 空集合上的「全部就绪」是句废话（同 ``_nothing_observed`` 的原则）。
        return _nothing_observed("全部用户准备就绪", "一个用户都没有被跑起来")
    broken = [f"{run.label}：{run.setup_error}" for run in runs if run.setup_error]
    if broken:
        return Check("全部用户准备就绪", False, "\n    ".join(broken))
    return Check("全部用户准备就绪", True, f"{len(runs)} 个用户各拿到模型配置、智能体与会话")


def check_completed(runs: list[UserRun]) -> Check:
    """判据 1：每个人都收到 REPLY_END，且 ``finished_reason == completed``。

    Args:
        runs (`list[UserRun]`): 全部观测。

    Returns:
        `Check`: 结果。
    """
    turns = _all_turns(runs)
    if not turns:
        return _nothing_observed(
            "全部正常结束（REPLY_END + completed）", "一轮对话都没有被观测到"
        )
    problems: list[str] = []
    for run, turn in turns:
        prefix = f"{run.label} 第 {turn.round_index} 轮"
        if turn.chat_status not in (200, 202):
            problems.append(
                f"{prefix}：``POST /chat/`` 返回 {turn.chat_status} {turn.chat_body}"
            )
            continue
        if not turn.reached_end:
            # ⚠️ 「脚本等满 120 秒自己走的」与「服务端把流结束了却没给 REPLY_END」
            #    必须分开报 —— 处置方向完全不同（前者先查是不是判据太短/服务端真卡住，
            #    后者是确定的协议违规）。原先两者共用一句话，等于没有信息。
            if turn.timed_out:
                why = (
                    f"脚本等满 {CONVERSATION_TIMEOUT_SECONDS:.0f} 秒仍未结束"
                    f"（实际等了 {turn.elapsed:.1f}s）—— **不能断定服务端有错**，"
                    f"要去看它最终有没有完成"
                )
            else:
                why = (
                    f"**流被结束了却没有 REPLY_END**（脚本等了 {turn.elapsed:.1f}s）"
                    f"—— 这是确定的协议违规，不是超时"
                )
            problems.append(
                f"{prefix}：没收到 REPLY_END —— {why}。"
                f"卡在 `{turn.last_event_type}` 之后"
                f"（已收到 {len(turn.events)} 个事件：{turn.event_types[:8]}）"
                + (f"；流错误：{turn.stream_error}" if turn.stream_error else "")
            )
            continue
        if turn.finished_reason != "completed":
            problems.append(
                f"{prefix}：REPLY_END 的 finished_reason={turn.finished_reason!r}"
                + (f"，error={turn.error_info}" if turn.error_info else "")
            )
        elif turn.stream_error:
            problems.append(f"{prefix}：流报错 —— {turn.stream_error}")
    if problems:
        return Check("全部正常结束（REPLY_END + completed）", False, "\n    ".join(problems))
    return Check(
        "全部正常结束（REPLY_END + completed）",
        True,
        f"{len(turns)} 轮对话全部 finished_reason=completed",
    )


def check_replies_nonempty(runs: list[UserRun]) -> Check:
    """判据 2：每个人都真的收到了文本。

    Args:
        runs (`list[UserRun]`): 全部观测。

    Returns:
        `Check`: 结果。
    """
    turns = _all_turns(runs)
    if not turns:
        return _nothing_observed("回复非空", "一轮对话都没有被观测到")
    empty = [
        f"{run.label} 第 {turn.round_index} 轮：回复为空"
        for run, turn in turns
        if not turn.reply_text.strip()
    ]
    if empty:
        return Check("回复非空", False, "\n    ".join(empty))
    sizes = ", ".join(
        f"{run.user_id}:{len(turn.reply_text)}字" for run, turn in turns
    )
    return Check("回复非空", True, sizes)


#: 「还在处理」的措辞。判据 2b 用它识别**答非所问的占位回复**。
#:
#: ⚠️ 这是一张**已知形状表**，不是通用判据 —— 它只能抓住见过的写法。
#: 之所以还是写：这个缺陷真实发生过（见 :func:`check_replies_are_answers`），
#: 而「回复非空」对它完全无感：30 个字的「等待中」也是非空，判据照样绿。
#: 漏网的另一种形状见 :data:`_MONOLOGUE_PREFIXES`（本表只管「占位」那一类）。
#:
#: ⚠️ **与 ``src/orchestration/reply_guard.py`` 的 ``WAITING_PHRASES``
#: 逐字相同**，且现在由
#: ``tests/test_orchestration_reply_guard.py::test_waiting_phrases_are_shared_with_the_concurrency_script``
#: 守住 —— 在那之前，「两处都要改」只是一句注释，没有任何东西拦得住它
#: 悄悄漂移（本文件里那条自相矛盾的旧注释就是这么来的）。
#:
#: ⚠️ 判据本身也同步了：运行时那边除了「命中等待词 + 没有数字」，还有一道
#: 「剔掉等待话术与标点后剩下的字不超过 10 个」的保险丝，用来放过
#: 「您的航班稍后起飞，请稍候到登机口」这类**正经答复**。
#: 两边判据不一致的后果是「测试说红线、线上放行」，所以这里也照搬。
_WAITING_PHRASES: tuple[str, ...] = (
    # 等待本身
    "稍等",
    "稍候",
    "请等",
    "等待",
    # 进行时
    "正在检索",
    "正在查询",
    "正在为您",
    "正在处理",
    "正在查看",
    "检索中",
    "查询中",
    "处理中",
    # 安抚
    "马上回来",
    "马上就好",
    "马上给您",
    "这就为您",
    "即刻为您",
    # 英文形态
    "one moment",
    "please wait",
    "let me check",
    "checking",
    "just a moment",
    "bear with me",
    "looking into",
    "looking up",
    "hang on",
    "one sec",
    "give me a moment",
)

#: 「内心独白漏进用户可见回复」的**开头**措辞。判据 2b 的第二个探测器。
#:
#: ⚠️ 只匹配**回复的最开头**，不做全文搜索。理由：这些词出现在中间多半是
#: 正常行文（「如果您想问报销比例…」），只有出现在**第一个字符**才说明
#: 模型把自己的思考过程当成了回复本身。
#:
#: ⚠️ 两个形状都是**实测撞到的**，不是假想的：
#:   · 英文 —— 2026-10-03 的 6 人 2 轮运行，用户 2 第 1 轮：
#:     ``I'll look up your flight orders. 我查了一下，你名下目前没有机票订单记录…``
#:     （同一形状在此之前还出现过一次，``I'll wait for the policy agent's findings.``）
#:   · 中文 —— 同一环境的 3 人运行，用户 3：
#:     ``用户问的是差旅标准——酒店住宿费能报多少。这是政策问答类问题。我先确认一下当前差标。结论：…``
#:
#: ⚠️ 英文那几条**必须写成具体动词**，不能写成 ``"I "`` / ``"Let me"``：
#: 对抗性评审实测过，「I recommend the Beijing hotel at 600 CNY per night.」
#: 「Let me summarize the policy: …」这类**正经英文答复**会被裸前缀判成独白 ——
#: 那是误报，而这个判据的误报会**污染并发测试的结论**（把好答案记成缺陷）。
#: 现在的写法只认「计划性动词 + 第一人称」：``I need to check…`` 是独白，
#: ``I checked…`` 是答复。缩写两种写法（``'`` 与 ``’``）都要列，模型两种都会输出。
#:
#: ⚠️ 判据必须与运行时闸门 ``src/orchestration/reply_guard.py`` 的
#: ``_MONOLOGUE_OPENERS`` **同源**：这边的测试判据比那边严会误报，
#: 比那边松会放行缺陷（模块文档写明「新增词时两处都要改」）。
#:
#: ⚠️ 假阳性边界：本项目是**中文**差旅助手，回复不该以英文句子或第三人称
#: 「用户…」开头（对用户说话用「您」）。所以这条规则的误报面很窄。
#: 但它是**已知形状表**，不是通用保证 —— 换个模型、换个语种都可能失效。
_MONOLOGUE_PREFIXES: tuple[str, ...] = (
    "I need",
    "I should",
    "I must",
    "I will",
    "I have to",
    "I want to",
    "I am going to",
    "I'm going to",
    "I'd better",
    "I'll",
    "I’ll",
    "Let me check",
    "Let me first",
    "Let me look",
    "Let me see",
    "Let me verify",
    "Let me query",
    "Let me search",
    "Let me find",
    "Let me call",
    "The user",
    # ---- 中文：与闸门 ``_MONOLOGUE_OPENERS``（强档 + 弱档）逐条对齐 ----
    # ⚠️ 2026-10-04 补齐。原先只有「用户问/想/说/要」四条，而闸门那边的
    # 强档还有「在问/询问/提到/需要/这里/既然」，弱档还有「现在我看 /
    # 系统(已经|已) / 当前(处于|阶段|我)」—— 脚本判绿而闸门判红的那半边
    # 从来没有人量过（对抗验证 P6 实测：'用户询问差旅标准。' 闸门是草稿、
    # 脚本 return None）。这条判据是**告警**，不影响退出码，所以补齐只会
    # 让漏网的形状现形，不会把并发结论判红。
    # ⚠️ 这边是**字面前缀**、那边是**正则**，做不到逐字相等；由
    # ``test_the_script_monologue_prefixes_cover_every_gate_opener``
    # 用样例集钉住「闸门抓到的，脚本也抓得到」这个方向。
    "用户问",
    "用户想",
    "用户说",
    "用户要",
    "用户在问",
    "用户询问",
    "用户提到",
    "用户需要",
    "用户这里",
    "用户既然",
    "现在我看",
    "系统已经",
    "系统已",
    "当前处于",
    "当前阶段",
    "当前我",
)


#: 「这段文字里有没有数量信息」的判据 —— **只认阿拉伯数字**。
#:
#: ⚠️ 中文数字（六百元 / 两晚）**刻意不算**，与 ``reply_guard._has_quantity``
#: 逐字同源。理由见那边的注释：「一下 / 一遍」里的中文数字是虚词，
#: 算进来会让「稍等一下」被判成有内容的答复，判据当场失效。
#:
#: ⚠️ 用 ``isdecimal()`` 而不是 ``isdigit()``：``'①'.isdigit()`` 与
#: ``'²'.isdigit()`` 都是 ``True``，会让「稍等①」这种文本解除保险丝。
#: 与 ``reply_guard._NUMBER_TOKEN_PATTERN`` 的 ``\d`` 口径一致。
def _has_quantity(text: str) -> bool:
    """判断文本里有没有阿拉伯数字（即「带回来了具体数据」的弱信号）。

    Args:
        text (`str`): 待检查的文本。

    Returns:
        `bool`: 含任一阿拉伯数字字符时返回 ``True``。
    """
    return any(char.isdecimal() for char in text)


#: 「N 分钟后」这类**时间估计**里的数字不是带回来的数据。
#:
#: ⚠️ 与 ``reply_guard._TIME_ESTIMATE_PATTERN`` **逐字同源**（中英文单位都要），
#: 由 ``tests/test_orchestration_reply_guard.py`` 的同源用例守住。
#: 早先这里只留了中文单位、想着「可读性优先」，那是错的：模型输出英文整句时
#: ``Please wait, 30 seconds.`` 会被这个脚本判成「带回了数据」，
#: 而线上闸门会正确拦下 —— 又一处「测试说绿、线上说红」。
_TIME_ESTIMATE_PATTERN = re.compile(
    r"\d+\s*(?:秒钟|秒|分钟|分|小时|个?小时|min|mins|minute|minutes|sec|secs|second|seconds|hours|hour)",
    re.IGNORECASE,
)


def _has_result_data(text: str) -> bool:
    """判断文本里有没有**带回来的数据**（排掉时间估计后的数字）。

    Args:
        text (`str`): 待检查的文本。

    Returns:
        `bool`: 去掉时间估计后仍含阿拉伯数字时返回 ``True``。
    """
    return _has_quantity(_TIME_ESTIMATE_PATTERN.sub("", text))


#: 标点与空白 —— 判「占位话术之外还剩多少字」时先剔掉它们。
#: ⚠️ 与 ``reply_guard._NON_CONTENT_CHARS`` 同源。
_NON_CONTENT_CHARS = re.compile(r"[\s，。、；：！？~～…—\-.,;:!?()（）\[\]【】\"'“”‘’]+")

#: 占位回复里，「等待话术之外」最多还能剩下几个字。
#: ⚠️ 与 ``reply_guard._PLACEHOLDER_MAX_RESIDUE`` 同源，边界案例见那边的注释。
_PLACEHOLDER_MAX_RESIDUE = 10

#: 「助手把活转交给了别人」的措辞。
#:
#: ⚠️ 与 ``reply_guard._DELEGATION_MARKERS`` **逐字同源**，由
#: ``tests/test_orchestration_reply_guard.py`` 的跨实现一致性用例守着。
#: 判据本身写在那边（为什么用词表、为什么全是多字词），这里不重复。
#:
#: ⚠️ 少了这一条会漏掉实测的真实占位：``已让政策问答智能体检索制度原文，稍等。``
#: 残留 16 字（比假阳性「您的航班稍后起飞，请稍候到登机口」的 13 字还长，
#: 字数阈值分不开），而 ``已经交给政策问答智能体了，请稍候。`` 更极端 ——
#: 它连上一个版本里借用的「草稿形状」都命中不了，于是**脚本判绿、
#: 线上也拦不住**（2026-10-04 同时实测到）。这张表就是把这两种写法收进来。
_DELEGATION_MARKERS = (
    "已让",
    "已经让",
    "已交给",
    "已经交给",
    "交由",
    "转交",
    "转给",
    "已安排",
    "已经安排",
    "已委托",
    "已请求",
    "已通知",
    "让政策问答",
    "让订单",
    "让行程",
    "让意图",
)


def _placeholder_reason(text: str) -> str | None:
    """判定回复是不是「已交给别人、稍等」式的占位。

    Args:
        text (`str`): 已 ``strip()`` 的回复全文。

    Returns:
        `str | None`: 命中的措辞说明；不是占位时 ``None``。
    """
    lowered = text.lower()
    hit = [phrase for phrase in _WAITING_PHRASES if phrase in lowered]
    if not hit:
        return None
    # ⚠️ 三道保险丝，缺一不可，判据与运行时闸门逐字同源：
    #
    #   1. 「带回了数据就放过」：三条默认问句的**任何**真实答复都会带数字
    #      （金额 / 日期 / 单号），而一句纯「稍等」不会。
    #   2. 时间估计里的数字不算数据 —— 否则「稍等，预计 2 分钟后给你结果。」
    #      会让第 1 道失效。
    #   3. 剔掉等待话术与标点后剩下的字不能多 —— 否则
    #      「您的航班稍后起飞，请稍候到登机口」这句**正经答复**会被误判。
    #      但**光靠字数不够**：残留可能更长却仍是占位（是转交交代），
    #      见下面第 4 条的说明。
    #
    # ⚠️ 这里原先写着「中文数字也算数（六百元/两晚）」，与紧邻的
    # ``_has_quantity`` docstring 以及代码**直接矛盾**（代码只认阿拉伯数字）。
    # 那句注释是错的，会让下一个人把 ``_has_quantity`` 改错方向，故删除。
    if _has_result_data(text):
        return None
    residue = text.lower()
    for phrase in _WAITING_PHRASES:
        residue = residue.replace(phrase, "")
    residue = _NON_CONTENT_CHARS.sub("", residue)

    #   4. 残留是在**交代一次转交**（「已让…」「已经交给…」）⇒ 仍是占位，
    #      无论它多长。与运行时闸门 ``reply_guard._is_placeholder`` 同源。
    if any(marker in residue for marker in _DELEGATION_MARKERS):
        return f"命中占位措辞 {hit}，且只剩一句转交交代"
    if len(residue) > _PLACEHOLDER_MAX_RESIDUE:
        return None
    return f"命中占位措辞 {hit}，且通篇没有数字"


def _monologue_reason(text: str) -> str | None:
    """判定回复是不是把**内心独白**当成了答复。

    Args:
        text (`str`): 已 ``strip()`` 的回复全文。

    Returns:
        `str | None`: 命中的开头措辞；不是独白时 ``None``。
    """
    for prefix in _MONOLOGUE_PREFIXES:
        if text.startswith(prefix):
            return f"回复以内心独白式措辞 {prefix!r} 开头"
    return None


def check_replies_are_answers(runs: list[UserRun]) -> Check:
    """判据 2b：回复是一份**答复**，而不是占位或内心独白。

    ★ 这条判据来自一次**真实的假绿灯**。2026-10-03 的一次运行里，三个用户
    全部 ``finished_reason=completed``、其余判据全绿，而用户 1 拿到的
    全部回复只有 30 个字（含一个空格）：

        ``已让政策问答智能体检索制度原文，稍等。 等待政策检索结果中。``

    ⚠️ 这条例外只有作者记录、**没有日志可复核**（服务端日志不记录回复正文，
    当时的日志已丢）。它证明的是「这种形状会发生」，不是「它发生了多少次」。

    ``REPLY_END`` 有了、``completed`` 有了、消息也非空 —— 用户却什么都没
    拿到。**「非空」与「有用」之间的那段距离，就是这条判据。**

    ═══ 两个探测器，各管一类 ═══

    1. **占位**（:func:`_placeholder_reason`）：回复命中
       :data:`_WAITING_PHRASES` 里的措辞，**且**整条回复不含任何阿拉伯数字。
       为什么加「不含数字」：三条问句（差标金额、机票状态、报销标准）的
       **任何**真实答复都会带数字 —— 金额、日期、单号；反过来，一句纯
       「稍等」不会。这个条件让「答复末尾顺带说句稍等」不误报。

    2. **内心独白漏进回复**（:func:`_monologue_reason`）：回复**以**内心
       独白式的措辞开头（:data:`_MONOLOGUE_PREFIXES`）。它原先被写成
       「本判据抓不到、要靠 eval.py」，但同一个形状反复原样出现
       （``I'll look up your flight orders. 我查了一下…``），说明它不是偶发，
       而是这条链路的**稳定行为**。既然是稳定的，就该有确定性判据，
       不该推给概率性的 LLM-as-judge。

    ═══ 频次（可复核的那一份）═══

    2026-10-03 的 **一次默认运行**（3 用户 × 1 轮）+ **两次加压运行**
    （``--users 6 --rounds 2``），合计 27 轮：慢车道共 18 轮，独白命中
    **7 条，全部在慢车道**（两次加压运行分别是 2/8 与 5/8，默认那次 0/2）。

    ⚠️ 别把这组数读成「三次 6×2」—— 3 次 6×2 是 36 轮，不是 27。
    这组频次证据来自仓库外的运行产物（``/tmp/aligo_concurrency_run*.txt``），
    仓库内无法自行复算，所以引用时请连同构成一起写。

    **7 是下界不是总数** —— 探测器只认 :data:`_MONOLOGUE_PREFIXES` 里那
    几十种开头写法，「我查了一下，你名下没有…」这种第一人称开场就不在表里。

    ⚠️ 早先在本文件/``README`` 里写过「共撞到 4 次」之类的历史次数，
    **那些数字互相矛盾且无法复核**（服务端日志根本不记录回复正文，
    容器重建后旧日志也没了）。要引用频次，用上面这组能重跑出来的数。
    ⚠️ 更要紧的是：有一例泄漏的不只是语气，而是**内部实现**
    （回复正文里出现了工具名 ``check_travel_policy``、参数名 ``kind/price/cabin``、
    智能体名 ``policy_rag``）。那已经不是「文风」而是信息外泄 ——
    但本判据只按「开头措辞」判定，抓不抓得到取决于那一轮怎么起头。

    ⚠️ 局限（写清楚，免得被当成通用保证）：两个探测器都是**已知形状表**，
    不是「回复质量」的度量。已知的漏网形状：

    · 独白出现在**句中**而不是开头（探测 2 只匹配开头，故意的 —— 全文搜索
      会把「如果您想问报销比例…」这类正常行文也抓进来）；
    · 用既有措辞表之外的说法表达同一个意思（换模型、换语种都可能）。

    真正通用的输出质量判定要靠 ``scripts/eval.py`` 的 LLM-as-judge，
    不是客户端判据能干的事。

    ⚠️ **本判据没有硬闸门。** 独白一律走 ``warn``，不影响退出码，而
    ``eval.py`` 不在 ``make concurrency`` 的路径上 —— 也就是说，
    「用户看到模型草稿」这件事目前**在 CI 里永远拦不下来**。
    保留 warn 的理由（不让并发结论取决于模型文风）成立，但它**不能**
    被读成「这个缺陷已经被防住了」。

    Args:
        runs (`list[UserRun]`): 全部观测。

    Returns:
        `Check`: 结果。
    """
    turns = _all_turns(runs)
    if not turns:
        return _nothing_observed("回复是答复而非占位", "一轮对话都没有被观测到")
    failures: list[str] = []
    warnings: list[str] = []
    for run, turn in turns:
        text = turn.reply_text.strip()
        head = f"{run.label} 第 {turn.round_index} 轮："
        placeholder = _placeholder_reason(text)
        if placeholder is not None:
            # ⚠️ 占位 ⇒ 失败，不是告警：用户**什么都没拿到**，这是功能故障。
            failures.append(f"{head}{placeholder}。原文：{text[:200]!r}")
            continue
        monologue = _monologue_reason(text)
        if monologue is not None:
            # ⚠️ 独白 ⇒ 告警：用户拿到了答案，多出来的是一段模型的草稿。
            #    理由是输出质量，不是并发正确性 —— 详见 Check.warn 的说明。
            warnings.append(f"{head}{monologue}。原文：{text[:200]!r}")
    if failures:
        detail = "\n    ".join(failures)
        if warnings:
            detail += "\n    " + "\n    ".join(warnings)
        return Check("回复是答复而非占位", False, detail)
    if warnings:
        return Check(
            "回复是答复而非占位",
            False,
            "\n    ".join(warnings)
            + f"\n    ⚠️ 这 {len(warnings)} 条是**输出质量**问题（用户拿到了答案，"
            "但回复开头是模型的草稿），不影响并发结论，故不计入失败。"
            "通用判定见 scripts/eval.py 的 LLM-as-judge。",
            True,
        )
    return Check(
        "回复是答复而非占位",
        True,
        f"{len(turns)} 轮回复里没有占位措辞、也没有以内心独白开头",
    )


def check_no_error_events(runs: list[UserRun]) -> Check:
    """判据 3：流里没有错误/异常事件。

    ``REQUIRE_USER_CONFIRM`` 也算异常：本脚本的三个问句都是**只读**的，
    不该有任何一步需要人工确认。它出现就说明有工具做了一次不该做的写操作。

    Args:
        runs (`list[UserRun]`): 全部观测。

    Returns:
        `Check`: 结果。
    """
    forbidden = {
        "EXCEED_MAX_ITERS",
        "REQUIRE_USER_CONFIRM",
        "REQUIRE_EXTERNAL_EXECUTION",
        "USER_INTERRUPT",
    }
    turns = _all_turns(runs)
    if not turns:
        return _nothing_observed("无错误事件", "一个事件都没有被观测到")
    problems: list[str] = []
    for run, turn in turns:
        hit = sorted(forbidden & set(turn.event_types))
        if hit:
            problems.append(f"{run.label} 第 {turn.round_index} 轮：出现事件 {hit}")
        for event in turn.events:
            if event.get("type") == "REPLY_END" and event.get("error"):
                problems.append(
                    f"{run.label} 第 {turn.round_index} 轮：REPLY_END 带 error "
                    f"{json.dumps(event['error'], ensure_ascii=False)[:200]}"
                )
    if problems:
        return Check("无错误事件", False, "\n    ".join(problems))
    return Check("无错误事件", True, "没有 EXCEED_MAX_ITERS / 确认请求 / error 字段")


def check_genuinely_concurrent(runs: list[UserRun]) -> Check:
    """判据 4：★ 三个人确实同时在飞 —— ``max(t_chat) < min(t_end)``。

    ⚠️ **这条判据能证明什么、不能证明什么，别读串了：**

    · 能证明 —— 请求**不是脚本按顺序发的**。这是本脚本最容易写错的地方
      （写错之后表面上照样是「3/3 通过」），所以它值一条判据；
    · **不能**证明服务端在并行处理。请求同时到达、服务端串行排队时，
      三个人的回复都还没结束，``max(t_chat) < min(t_end)`` **照样成立**。
      栅栏还把这点推到极致：``t_chat`` 的差被压到毫秒级，于是这条不等式
      几乎总是成立，它通过时说明不了多少。

    ⇒ 服务端并行的那份证据在进程内部（span 耗时之和 vs 墙钟），
    客户端脚本拿不到。详见模块文档「确实同时」那一节的末尾。

    Args:
        runs (`list[UserRun]`): 全部观测。

    Returns:
        `Check`: 结果。
    """
    if not runs:
        return Check("确实同时（时间窗重叠）", False, "没有任何用户被跑起来")
    # ⚠️ 必须先挡住「一轮都没观测到」。原先的循环上界取自 ``runs[0].turns`` ——
    # 若用户 1 在准备阶段就失败（``turns`` 为空），``range(0)`` 让循环体一次都不执行，
    # ``problems`` 保持为空，函数顺流而下返回**绿**。于是「一条数据都没有」被报成
    # 「确实同时」，与服务端做了什么毫无关系。
    # 这不是假想：判据 0 失败时正是这个形状。其余 7 条判据都走 ``_nothing_observed``，
    # 本判据原先漏了 —— 同一个坑，8 条里补了 7 条。
    if not _all_turns(runs):
        return _nothing_observed("确实同时（时间窗重叠）", "一轮能用于判定的对话都没有")
    # ⚠️ 轮数取**所有用户的并集**，不取 ``runs[0]``。用户 1 一轮没跑成、别人跑了
    # 两轮时，按 ``runs[0]`` 会照样一轮都不判 —— 那是上面那个坑的第二种形状。
    rounds = max(len(run.turns) for run in runs)
    problems: list[str] = []
    for index in range(rounds):
        window = [
            (run, run.turns[index])
            for run in runs
            if index < len(run.turns) and run.turns[index].reached_end
        ]
        if len(window) < 2:
            problems.append(f"第 {index + 1} 轮：能用于判定的窗口少于 2 个")
            continue
        base = min(turn.t_chat for _, turn in window)
        starts = [(turn.t_chat - base, run) for run, turn in window]
        ends = [(turn.t_end - base, run) for run, turn in window]
        last_start, last_start_run = max(starts)
        first_end, first_end_run = min(ends)
        if last_start >= first_end:
            detail = ", ".join(
                f"{run.user_id}[{(turn.t_chat - base) * 1000:.0f}~"
                f"{(turn.t_end - base) * 1000:.0f}ms]"
                for run, turn in window
            )
            problems.append(
                f"第 {index + 1} 轮：**实际上是串行的**。"
                f"最后发出的是 {last_start_run.user_id}（+{last_start * 1000:.0f}ms），"
                f"而最早结束的是 {first_end_run.user_id}（+{first_end * 1000:.0f}ms）"
                f"\n      各人时间窗：{detail}"
                "\n      ⚠️ 这一条不通过时，其余「通过」都没有意义 —— "
                "没重叠就没有并发，测的只是三个人排队。"
            )
    if problems:
        return Check("确实同时（时间窗重叠）", False, "\n    ".join(problems))

    spans = []
    for index in range(len(runs[0].turns)):
        window = [run.turns[index] for run in runs if index < len(run.turns)]
        if len(window) < 2:
            continue
        base = min(t.t_chat for t in window)
        last_start = max(t.t_chat for t in window) - base
        first_end = min(t.t_end for t in window) - base
        spans.append(f"第 {index + 1} 轮：全员同时在飞 {first_end - last_start:.2f}s")
    return Check("确实同时（时间窗重叠）", True, "；".join(spans))


def check_stream_isolation(runs: list[UserRun]) -> Check:
    """判据 5：★ 每个 SSE 事件带的 ``session_id`` 都是**自己的**。

    ⚠️ **覆盖面的真相：只有一部分事件带 ``session_id``。** 实测一次 3 人
    一轮的运行：三条流合计 215 个事件（99 + 47 + 69），其中带 ``session_id``
    的只有 **6 个**（每人 2 个 —— ``REPLY_START`` 与 ``REPLY_END``）。
    也就是说本判据**逐字校验的是 6 个事件，不是 215 个**，其余 209 个只能
    靠「它们的 session_id 缺失 ⇒ 无从串起」间接保证。

    ⇒ 所以详情里必须写出「X 个带 session_id 的事件」，读的人才知道证据量。
    只写「流不串号 ✅」会让人以为整条流都被验过了 —— 那是本判据最容易
    被读出来的假象。（要真正覆盖全部事件，得让框架在每个事件上都带
    ``session_id``，那不在本脚本能改的范围内。）

    Args:
        runs (`list[UserRun]`): 全部观测。

    Returns:
        `Check`: 结果。
    """
    problems: list[str] = []
    checked = 0
    total_events = 0
    for run, turn in _all_turns(runs):
        total_events += len(turn.events)
        own = {
            sid
            for sid in run.session_ids
        }
        for event in turn.events:
            session_id = event.get("session_id")
            if session_id is None:
                continue
            checked += 1
            if session_id not in own:
                problems.append(
                    f"{run.label} 第 {turn.round_index} 轮："
                    f"流里出现了**不属于自己**的 session_id={session_id!r}"
                    f"（自己的是 {sorted(own)}，事件类型 {event.get('type')}）"
                )
                break
    if problems:
        return Check("SSE 流不串号", False, "\n    ".join(problems))
    if not checked:
        return Check(
            "SSE 流不串号",
            False,
            "⚠️ 没有任何事件带 session_id —— 这条判据**什么都没验到**。"
            "不能当作通过（改过一次的事件结构就可能是这个症状）。",
        )
    return Check(
        "SSE 流不串号",
        True,
        f"{checked}/{total_events} 个事件带 session_id，它们全部指向自己的会话"
        f"（其余 {total_events - checked} 个事件不带这个字段，本判据管不到）",
    )


def _text_of(message: dict[str, Any]) -> str:
    """把一条消息里所有 ``type == "text"`` 块的 ``text`` 拼起来。

    ⚠️ 只取 text 块。消息的 ``content`` 是一个**块列表**，里面还混着
    ``hint`` / ``tool_call`` / ``tool_result`` 等 —— 把它们一起拼进字符串，
    会让「这段文本是不是我的问题」变成一个无法回答的问题。

    Args:
        message (`dict`): ``GET /sessions/{id}/messages`` 返回的一条消息。

    Returns:
        `str`: 拼接后的纯文本；没有 text 块时为空串。
    """
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


def check_transcript_isolation(base_url: str, runs: list[UserRun]) -> Check:
    """判据 6：★ 每条会话里**有且只有自己的那一句**，没有别人的。

    ⚠️ 与判据 5 互为补角：流不串号只能说明「推」对了，读回历史才能说明
    「存」也对了。两者都做，是因为它们的失败模式不同。

    ★ 判定必须精确到**消息**，不能退化成「历史 JSON 里有没有别人的问题原文」
      这种子串搜索。本脚本第一版就是这么写的，然后它报了一次假串号：

          用户 3 问「去北京出差住宿费能报多少」，模型答「依据公司差旅标准…」
          而用户 1 的问题是「住宿标准」—— 于是用户 3 的历史里**字面上**
          含有用户 1 的问题原文。

      那次报的是「串号」—— 这个系统里最严重的缺陷 —— 而实际上什么都没发生。
      这就是子串搜索的根本问题：**问句越短，越容易在别人的回答里自然出现**。
      而一条会喊狼来了的判据，代价比不判还大：它教人忽略告警，下一次真串号
      时也一样被忽略。

      所以这里改成问一个**能回答的问题**：「这条会话里的 *user 消息*，
      是不是恰好一条、且逐字等于本人问的那句？」串号必然让这句话不成立
      （多出一条 user 消息），而正常的词汇重合不会。

    Args:
        base_url (`str`): 服务地址。
        runs (`list[UserRun]`): 全部观测。

    Returns:
        `Check`: 结果。
    """
    problems: list[str] = []
    owner_of = {run.case.prompt: run.user_id for run in runs}
    total_user_messages = 0

    for run in runs:
        for session_id in run.session_ids:
            try:
                with _client() as client:
                    response = client.get(
                        f"{base_url}/sessions/{session_id}/messages",
                        headers=_headers(run.user_id),
                        params={"agent_id": run.agent_id},
                    )
            except Exception as exc:  # pylint: disable=broad-except
                problems.append(f"{run.label}：读会话历史出错 {type(exc).__name__}: {exc}")
                continue
            if response.status_code != 200:
                problems.append(
                    f"{run.label}：读会话历史返回 {response.status_code} "
                    f"{_body_excerpt(response)}"
                )
                continue

            messages = response.json().get("messages", [])
            user_texts = [
                _text_of(message)
                for message in messages
                if message.get("role") == "user"
            ]
            total_user_messages += len(user_texts)

            if len(user_texts) != 1:
                problems.append(
                    f"{run.label}：会话里有 **{len(user_texts)} 条** user 消息，期望 1 条。"
                    f"\n      各条内容：{[t[:40] for t in user_texts]}"
                    "\n      ⚠️ 多出来的那条就是串号的直接证据。"
                )
                continue

            text = user_texts[0].strip()
            if text != run.case.prompt:
                owner = owner_of.get(text)
                if owner is not None and owner != run.user_id:
                    problems.append(
                        f"{run.label}：会话里的 user 消息是 **{owner}** 的问句"
                        f"「{text}」—— 这是串号。"
                    )
                else:
                    problems.append(
                        f"{run.label}：会话里的 user 消息与本人问句**逐字不符**。"
                        f"\n      期望：{run.case.prompt!r}"
                        f"\n      实际：{text!r}"
                    )

            if not any(m.get("role") == "assistant" for m in messages):
                problems.append(f"{run.label}：会话里没有 assistant 消息 —— 回复没落库。")

    if problems:
        return Check("会话记载不串号", False, "\n    ".join(problems))
    sessions = sum(len(run.session_ids) for run in runs)
    if not sessions:
        return _nothing_observed(
            "会话记载不串号", "一条会话都没有被读到（所以没有任何一份记载被比对过）"
        )
    return Check(
        "会话记载不串号",
        True,
        f"{sessions} 条会话各自恰好 1 条 user 消息，且逐字等于本人的问句"
        f"（共 {total_user_messages} 条，无一多余）",
    )


def check_cross_user_invisible(base_url: str, runs: list[UserRun]) -> Check:
    """判据 7：用 A 的身份读 B 的会话/智能体，必须 404。

    ⚠️ 「必须 404」而不是「必须非 200」：403 意味着「你知道它存在，但你不能看」，
    那也是一次信息泄漏（对陌生人应该连存在性都不确认）。框架默认的
    ``DenyAllResourceAccessPolicy`` 正好给 404，本判据把这个行为钉死。

    Args:
        base_url (`str`): 服务地址。
        runs (`list[UserRun]`): 全部观测。

    Returns:
        `Check`: 结果。
    """
    problems: list[str] = []
    probes = 0
    for run in runs:
        for other in runs:
            if other.user_id == run.user_id or not other.session_ids:
                continue
            with _client() as client:
                # (a) 用 A 的身份列 B 的智能体名下的会话
                listed = client.get(
                    f"{base_url}/sessions/",
                    headers=_headers(run.user_id),
                    params={"agent_id": other.agent_id},
                )
                probes += 1
                if listed.status_code != 404:
                    problems.append(
                        f"{run.user_id} 能看到 {other.user_id} 的智能体下的会话："
                        f"HTTP {listed.status_code} {_body_excerpt(listed, 120)}"
                    )
                # (b) 用 A 的身份读 B 的会话消息
                messages = client.get(
                    f"{base_url}/sessions/{other.session_ids[0]}/messages",
                    headers=_headers(run.user_id),
                    params={"agent_id": other.agent_id},
                )
                probes += 1
                if messages.status_code != 404:
                    problems.append(
                        f"{run.user_id} 能读到 {other.user_id} 的会话消息："
                        f"HTTP {messages.status_code} {_body_excerpt(messages, 120)}"
                    )
    if not probes:
        return _nothing_observed(
            "跨用户不可见（404）", "一次越权探测都没发出去（至少要两个用户各有会话）"
        )
    if problems:
        return Check("跨用户不可见（404）", False, "\n    ".join(problems))
    return Check("跨用户不可见（404）", True, f"{probes} 次越权探测全部 404")


def _read_5xx_counts(base_url: str) -> dict[tuple[str, str, str], float] | None:
    """读服务端**自己**记的 5xx 计数，按 ``(method, route, status)`` 分组。

    ⚠️ 为什么需要这个函数（判据 8 原先只看 ``POST /chat/`` 的响应码）：
    客户端只看得到**自己发的那些请求**，而并发下最可能出 5xx 的地方恰恰是
    客户端不直接看的 —— SSE 流端点、``/api/v1/*``、会话读写、以及**它根本
    没发过**的路径。服务端的 ``aligo_http_requests_total`` 覆盖每一个路由，
    这才是「全程没有 5xx」这句话的完整证据面。

    ⚠️ 返回 ``None``（读不到）与返回 ``{}``（读到了、一条 5xx 都没有）
    **是两回事**，调用方必须分开处置。这是本仓库评测模块里
    ``Observation.unmeasured`` 的同一条原则：**「没测」不能被当成「测到 0」**。

    ⚠️ 标签顺序**不写死**。Prometheus 文本格式里的标签顺序就是注册时的顺序，
    现在恰好是 ``(method, route, status)``，但把它写进正则意味着
    ``src/observability/metrics.py`` 里改一次 ``labelnames`` 的次序，这个检查
    就会**静默失效**（正则一条都不匹配 ⇒ 看起来像「零 5xx」）。所以用通用解析。

    Args:
        base_url (`str`): 服务地址。

    Returns:
        `dict[tuple[str, str, str], float] | None`: 键是
        ``(method, route, status)``，值是累计次数；读不到时是 ``None``。
    """
    try:
        with _client() as client:
            response = client.get(f"{base_url}/metrics")
    except httpx.HTTPError:
        return None
    if response.status_code != 200:
        return None
    return _read_5xx_counts_from_text(response.text)


def _read_5xx_counts_from_text(text: str) -> dict[tuple[str, str, str], float] | None:
    """从 Prometheus 文本格式里挑出 5xx 时序。

    ⚠️ 抽成独立函数是为了**让解析逻辑可单测** —— 它埋在 ``_read_5xx_counts``
    里的话，要测它就得起一个 HTTP 服务，于是最该被钉死的那条规则
    （「解析不了就返回 ``None``，绝不返回 ``{}``」）反而没人测。

    Args:
        text (`str`): ``/metrics`` 的响应体。

    Returns:
        `dict[tuple[str, str, str], float] | None`: 键是
        ``(method, route, status)``；任何一行 5xx 解析不出必要标签时返回
        ``None``（= 未测量）。
    """
    counts: dict[tuple[str, str, str], float] = {}
    for line in text.splitlines():
        if not line.startswith("aligo_http_requests_total{"):
            continue
        # ⚠️ 必须 rpartition（**最后**一个 ``}``），不能 partition。
        # 路由标签里就带着花括号：``route="/sessions/{id}/stream"`` ——
        # partition 会在 ``{id}`` 的那个 ``}`` 上断开，把标签串截成
        # ``route="/sessions/{id``，于是 ``status`` 标签读不到、
        # 整行被 ``continue`` 静默丢掉。后果是这条 5xx **永远算不进增量**，
        # 而这正是本判据唯一要抓的东西（客户端看不见的那一类）。
        # 这个 bug 是被 tests/test_concurrency_test.py 里那条
        # 「解析 Prometheus 文本格式」的用例抓出来的 —— 手写解析器里
        # 最容易错的就是「拿第一个分隔符去切一个内容里含分隔符的串」。
        labels_text, _, value_text = line.rpartition("}")
        parsed = dict(re.findall(r'(\w+)="([^"]*)"', labels_text))
        if not parsed.get("status", "").startswith("5"):
            continue
        try:
            counts[(parsed["method"], parsed["route"], parsed["status"])] = float(
                value_text.strip()
            )
        except (KeyError, ValueError):
            # 行格式看不懂 ⇒ 报「没测到」，绝不报「0」。见上面的 ⚠️。
            return None
    return counts


def _diff_5xx(
    before: dict[tuple[str, str, str], float],
    after: dict[tuple[str, str, str], float],
) -> list[str]:
    """比对两次快照，返回**本次运行新增**的 5xx。

    ⚠️ 用增量而不是绝对值：app 容器已经跑了几个小时，之前任何一次调试留下的
    5xx 都会算进来 —— 那样这条判据会永久报红，而人会开始忽略它。

    Args:
        before (`dict`): 运行前抓的快照。
        after (`dict`): 运行后抓的快照。

    Returns:
        `list[str]`: 每条新增的 ``(method, route, status)`` 一条说明。
    """
    problems: list[str] = []
    for key, count in sorted(after.items()):
        delta = count - before.get(key, 0.0)
        if delta <= 0:
            continue
        method, route, status = key
        problems.append(
            f"服务端 ``/metrics`` 显示本次运行新增 {int(delta)} 次 "
            f"``{method} {route}`` → {status} —— 客户端没有直接看到这一笔"
            "（可能是 SSE 端点或业务接口）。"
        )
    return problems


def check_no_server_errors(
    base_url: str,
    runs: list[UserRun],
    five_xx_before: dict[tuple[str, str, str], float] | None,
) -> Check:
    """判据 8：全程没有 5xx、没有 409 —— **客户端与服务端两侧都要看**。

    409 单列：它是框架的**双提交保护**（同一会话已有对话在飞）。三个人各用
    各的会话，本不该触发；一旦触发，说明会话隔离出了问题 —— 那比一个 500
    更值得停下来看。

    ⚠️ 5xx 有**两个**证据面，缺一个就漏一类：
      · 客户端侧：``POST /chat/`` 的响应码（本判据的第一版只有这个）；
      · 服务端侧：``/metrics`` 的 ``aligo_http_requests_total{status=5xx}``
        增量 —— 覆盖**全部路由**，包括客户端根本没发的那些（SSE 端点、
        ``/api/v1/*``、会话读写…）。
    服务端那侧是可选的：读不到就**在详情里写明「未测量」**，而不是当它通过。

    Args:
        base_url (`str`): 服务地址。
        runs (`list[UserRun]`): 全部观测。
        five_xx_before (`dict | None`): 运行**前**抓的 5xx 快照
            （:func:`_read_5xx_counts`）；``None`` 表示当时没读到。

    Returns:
        `Check`: 结果。
    """
    turns = _all_turns(runs)
    if not turns:
        return _nothing_observed("无服务端错误（无 5xx / 无 409）", "一次请求都没有被观测到")
    problems: list[str] = []
    for run, turn in turns:
        if turn.chat_status == 409:
            problems.append(
                f"{run.label} 第 {turn.round_index} 轮：``POST /chat/`` 返回 409 —— "
                "同一会话有对话在飞。三个用户各用各的会话，这不该发生。"
            )
        elif turn.chat_status is not None and turn.chat_status >= 500:
            problems.append(
                f"{run.label} 第 {turn.round_index} 轮：``POST /chat/`` 返回 "
                f"{turn.chat_status} {turn.chat_body}"
            )

    after = _read_5xx_counts(base_url)
    if five_xx_before is None or after is None:
        server_note = (
            "⚠️ 服务端 ``/metrics`` 的 5xx 计数**没能读到**（未测量）——"
            "本次只验了 ``POST /chat/`` 与业务接口的响应码。"
        )
    else:
        server_delta = _diff_5xx(five_xx_before, after)
        problems.extend(server_delta)
        if server_delta:
            server_note = ""
        elif not after:
            # ⚠️ 「一条 5xx 时序都没有」与「有若干条、但增量是 0」是两种不同的
            #    证据，别用同一句话带过：前者说明这个进程从启动起就没出过 5xx，
            #    后者说明出过（在本次运行之前）、但本次没新增。
            server_note = "服务端从启动至今**一条 5xx 时序都没有**（覆盖全部路由）"
        else:
            server_note = f"服务端 {len(after)} 条 5xx 时序本次增量全为 0（覆盖全部路由）"

    if problems:
        return Check("无服务端错误（无 5xx / 无 409）", False, "\n    ".join(problems))
    return Check(
        "无服务端错误（无 5xx / 无 409）",
        True,
        f"客户端侧没有 5xx、也没有 409；{server_note}",
    )


def check_business_api(base_url: str, runs: list[UserRun]) -> Check:
    """判据 9：业务接口在并发下也答得对（身份没有串）。

    Args:
        base_url (`str`): 服务地址。
        runs (`list[UserRun]`): 全部观测。

    Returns:
        `Check`: 结果。
    """
    # ⚠️ 这里每一处 I/O 都必须**自己兜住异常**。判据函数是在 ``main`` 的
    # ``try`` 之外被调用的：httpx 抛出来会直接冒出 ``main``，于是报告不打印、
    # **``finally`` 里的清理也不执行** —— 一次跑挂会留下满地的会话与智能体。
    # 实测踩过：探活时服务正好不可达，脚本只吐了一段 traceback 就退出，
    # 产物一件没删。判据函数不抛异常，是本脚本能保证「跑完自己删掉」的前提。
    if not runs:
        return _nothing_observed("业务接口身份正确", "一个身份都没有被验过")
    problems: list[str] = []
    for run in runs:
        try:
            with _client() as client:
                me = client.get(f"{base_url}/api/v1/me", headers=_headers(run.user_id))
                if me.status_code != 200:
                    problems.append(
                        f"{run.user_id}：``GET /api/v1/me`` 返回 {me.status_code}"
                    )
                    continue
                body = json.dumps(me.json(), ensure_ascii=False)
        except Exception as exc:  # pylint: disable=broad-except
            problems.append(
                f"{run.user_id}：``GET /api/v1/me`` 请求失败 "
                f"{type(exc).__name__}: {exc}"
            )
            continue
        if run.user_id not in body:
            problems.append(
                f"{run.user_id}：``/api/v1/me`` 回答里没有这个身份 —— "
                f"身份串了。响应：{body[:160]}"
            )
    if problems:
        return Check("业务接口身份正确", False, "\n    ".join(problems))
    return Check("业务接口身份正确", True, f"{len(runs)} 个身份的 /api/v1/me 都答自己")


# ==============================================================================
# 清理
# ==============================================================================


def _delete(client: httpx.Client, base_url: str, path: str,
            headers: dict[str, str], params: dict[str, str] | None = None) -> str:
    """发一次 DELETE，返回问题描述（空串 = 成功或目标已不在）。

    Args:
        client (`httpx.Client`): HTTP 客户端。
        base_url (`str`): 服务地址。
        path (`str`): 路径。
        headers (`dict[str, str]`): 请求头。
        params (`dict[str, str] | None`): 查询参数。

    Returns:
        `str`: 问题描述；空串表示没出问题。
    """
    try:
        response = client.delete(f"{base_url}{path}", headers=headers, params=params)
    except Exception as exc:  # pylint: disable=broad-except
        return f"DELETE {path} 抛异常：{type(exc).__name__}: {exc}"
    if response.status_code not in _ALREADY_GONE_STATUSES:
        return (
            f"DELETE {path} 返回 {response.status_code} "
            f"（期望 204，或 404 = 已经不在）：{_body_excerpt(response, 120)}"
        )
    return ""


def _cleanup_all(base_url: str, runs: list[UserRun], keep: bool) -> list[str]:
    """删掉本脚本建出的会话与智能体，并回读确认。

    ⚠️ 顺序必须是**先会话、后智能体**：会话的 DELETE 要求带 ``agent_id``，
    先把智能体删了，会话就再也无法单独删除。反过来虽然也能删干净
    （智能体的 DELETE 会级联），但那是靠副作用兜底，不是靠把每一步做对。

    ⚠️ **只删自己建的智能体**（``agent_created``）。复用来的 ``main_plan``
    是别人的工作区，删它等于清空用户正在用的东西。

    ⚠️ 本函数**不抛异常**：清理失败不能让一次成功的并发测试变成失败，
    但也不能被吞掉 —— 失败会作为问题列表返回，由调用方写进报告。

    Args:
        base_url (`str`): 服务地址。
        runs (`list[UserRun]`): 全部观测。
        keep (`bool`): ``True`` = 按要求保留现场，不做任何删除。

    Returns:
        `list[str]`: 问题描述列表；空列表 = 全部清理并回读通过。
    """
    if keep:
        return []
    problems: list[str] = []
    for run in runs:
        headers = _headers(run.user_id)
        try:
            with _client() as client:
                for session_id in run.session_ids:
                    problem = _delete(
                        client,
                        base_url,
                        f"/sessions/{session_id}",
                        headers,
                        {"agent_id": run.agent_id},
                    )
                    if problem:
                        problems.append(f"{run.user_id} 删会话：{problem}")
                # ⚠️ 会话必须**回读**，不能只看 DELETE 的状态码。见 `_readback_sessions`。
                problems.extend(_readback_sessions(client, base_url, run, headers))
                if run.agent_created and run.agent_id:
                    problem = _delete(client, base_url, f"/agent/{run.agent_id}", headers)
                    if problem:
                        problems.append(f"{run.user_id} 删智能体：{problem}")
        except Exception as exc:  # pylint: disable=broad-except
            problems.append(f"{run.user_id} 清理时出错：{type(exc).__name__}: {exc}")
    return problems


def _readback_sessions(
    client: httpx.Client, base_url: str, run: UserRun, headers: dict[str, str]
) -> list[str]:
    """回读该用户的会话列表，确认真删掉了（返回问题描述，空 = 干净）。

    ⚠️ **为什么非回读不可。** ``_delete`` 把 200/204/404 一视同仁（``_ALREADY_GONE_STATUSES``），
    于是一条 DELETE 返回 404 时，它分不清两件完全相反的事：

      · 「删成功了，本来就没了」—— 正常；
      · 「这接口根本不认这个 id / 路由压根不存在」—— **一个都没删掉**。

    两种情况都不报问题，判据 10 却会打印「已删 N 条会话」。这条判据是**有先决条件的**：
    这些会话在本轮里被**成功读写过**（判据 6 读的就是它们），所以「存在」是**已证**的，
    不是假设的。于是「删完还能列出来」才构成证据，而不是「404 所以没事」。

    回读的时机是**删完会话、还没删智能体**：智能体一旦删掉，``GET /sessions/`` 的
    ``agent_id`` 查找就会 404（实测），那之后的 404 是「智能体没了」，证明不了会话没了。

    Args:
        client (`httpx.Client`): HTTP 客户端。
        base_url (`str`): 服务地址。
        run (`UserRun`): 该用户的观测。
        headers (`dict[str, str]`): 该用户的请求头。

    Returns:
        `list[str]`: 问题描述列表。
    """
    if not run.session_ids:
        return []
    try:
        response = client.get(
            f"{base_url}/sessions/", headers=headers, params={"agent_id": run.agent_id}
        )
    except Exception as exc:  # pylint: disable=broad-except
        return [f"{run.user_id}：回读 /sessions/ 失败 {type(exc).__name__}: {exc}"]
    if response.status_code != 200:
        return [
            f"{run.user_id}：回读 /sessions/ 返回 {response.status_code} —— "
            f"删没删掉**没验到**（不能因为不是 200 就当它删干净了）"
        ]
    try:
        listed = _session_ids_in_list(response.json())
    except Exception as exc:  # pylint: disable=broad-except
        return [f"{run.user_id}：/sessions/ 响应不是预期形状 {type(exc).__name__}: {exc}"]
    if listed is None:
        return [
            f"{run.user_id}：/sessions/ 的条目没有嵌套的 session.id，"
            f"解析不出来 —— 删没删掉**没验到**（宁可报未测量，也不要静默读成空集）"
        ]
    left = [sid for sid in run.session_ids if sid in listed]
    return [f"{run.user_id}：会话 {sid} 删完仍在列表里" for sid in left]


def _session_ids_in_list(payload: Any) -> set[str] | None:
    """从 ``GET /sessions/`` 的响应里取出全部会话 id；取不出来返回 ``None``。

    ⚠️ 真实响应是**嵌套**的 —— 实测一条长这样::

        {"sessions": [{"session": {"id": "db16a2…", "agent_id": …, …}}], "total": 1}

    不是 ``{"sessions": [{"id": …}]}``。照后者写 ``item.get("id")`` 会**永远读到
    ``None``**，于是集合恒为空、``left`` 恒为空、判据恒绿 —— 一个因为解析写错
    而永远通不过的判据，比没有判据更糟（它会给出「验过了」的错觉）。
    这里因此返回 ``None`` 而不是空集：**解析失败 ≠ 列表为空**。

    Args:
        payload (`Any`): 响应 JSON。

    Returns:
        `set[str] | None`: 会话 id 集合；解析不出来时为 ``None``。
    """
    items = payload.get("sessions") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None
    ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            return None
        nested = item.get("session")
        # 兼容两种：嵌套的 {"session": {"id": …}} 与扁平的 {"id": …}。
        node = nested if isinstance(nested, dict) else item
        value = node.get("id")
        if not isinstance(value, str) or not value:
            return None
        ids.add(value)
    return ids


def check_cleanup(
    base_url: str, runs: list[UserRun], problems: list[str], *, keep: bool
) -> Check:
    """判据 10：本脚本建出的东西已经全部删掉，且**回读确认**过。

    ⚠️ 两类产物的回读方式**不同**，因为容器不同：
      · **智能体**：回读 ``GET /agent/`` 这个**列表**，看 id 还在不在里面。
        列表里没有 = 真的没了（列表是正面证据，不是 404）。
      · **会话**：回读在 ``_cleanup_all`` 里、删完会话之后立刻做（见
        ``_readback_sessions``）—— 只能这样，因为本判据被调用时智能体已经删了，
        那时 ``GET /sessions/`` 必然 404，"404 证明删干净了" 是个循环论证。

    ⚠️ ``keep=True`` 时本判据**必须直接通过**，不能去回读「东西还在不在」——
    东西当然还在，因为那是用户明确要求的 ``--keep``。拿「还在」判失败，
    等于把用户自己的选择报成缺陷：一次 ``make concurrency CONCURRENCY_ARGS="--keep"``
    会以退出码 1 结束，而它做对了每一件事。
    这条实测踩过（``tests/test_concurrency_test.py`` 有对应用例）。

    Args:
        base_url (`str`): 服务地址。
        runs (`list[UserRun]`): 全部观测。
        problems (`list[str]`): ``_cleanup_all`` 返回的问题。
        keep (`bool`): 是否是「按要求保留」。

    Returns:
        `Check`: 结果。
    """
    if not runs:
        # 「一个用户都没有」不是「清理干净了」，是这次调用本身没跑起来。
        # 与 ``keep=True`` 的「判据不适用」不同：那个是**用户明确要求**的结果。
        return _nothing_observed("产物已清理", "一个用户都没有被跑起来")
    if keep:
        kept = sum(len(run.session_ids) for run in runs) + sum(
            1 for run in runs if run.agent_created
        )
        return Check(
            "产物已清理",
            True,
            f"ℹ️ 按要求**保留**了本轮产物（--keep）：{kept} 件未删除，判据不适用。",
        )
    if problems:
        return Check("产物已清理", False, "\n    ".join(problems))

    created = sum(1 for run in runs if run.agent_created)
    sessions = sum(len(run.session_ids) for run in runs)
    if not created and not sessions:
        # ⚠️ 没有产物可回读时，**不能**沿用下面那句「按 id 回读确认不在」——
        # 回读根本没发生过，句子却断言它发生了。这条判据的价值全在「我验过」，
        # 一句没验过却写着「已确认」的详情，比不写更糟。
        return Check(
            "产物已清理",
            True,
            "ℹ️ 本轮没有建出任何东西（0 条会话 / 0 个自建智能体），无产物可回读。",
        )

    leftovers: list[str] = []
    for run in runs:
        if not run.agent_created:
            continue
        with _client() as client:
            try:
                response = client.get(f"{base_url}/agent/", headers=_headers(run.user_id))
            except Exception as exc:  # pylint: disable=broad-except
                leftovers.append(f"{run.user_id}：回读失败 {type(exc).__name__}: {exc}")
                continue
        if response.status_code != 200:
            leftovers.append(f"{run.user_id}：回读 /agent/ 返回 {response.status_code}")
            continue
        ids = {str(item.get("id")) for item in response.json().get("agents", [])}
        if run.agent_id in ids:
            leftovers.append(f"{run.user_id}：智能体 {run.agent_id} 仍在列表里")
    if leftovers:
        return Check("产物已清理", False, "\n    ".join(leftovers))
    return Check(
        "产物已清理",
        True,
        f"已删 {sessions} 条会话 + {created} 个自建智能体；"
        f"会话按 id 回读列表确认不在、智能体按 id 回读列表确认不在",
    )


# ==============================================================================
# 报告
# ==============================================================================


def _print_runs(runs: list[UserRun]) -> None:
    """打印每个用户这一轮实际发生了什么。

    ⚠️ 回复正文只截 60 字：报告是给人读的，不是把模型输出原样搬一遍。
    要点是「三条回答确实是三个不同问题的答案」，不是全文转录。

    Args:
        runs (`list[UserRun]`): 全部观测。
    """
    print("▶ 各用户这一轮实际发生了什么")
    print()
    for run in runs:
        print(f"  {run.label}")
        print(f"      问：{run.case.prompt}")
        print(f"      路径：{run.case.route}")
        if run.setup_error:
            print(f"      ❌ 准备阶段失败：{run.setup_error}")
            continue
        for turn in run.turns:
            reply = " ".join(turn.reply_text.split())
            print(
                f"      第 {turn.round_index} 轮：chat={turn.chat_status} "
                f"REPLY_END={'有' if turn.reached_end else '无'} "
                f"({turn.finished_reason or '—'}) "
                f"{len(turn.events)} 个事件"
            )
            print(f"          答：{reply[:60]}{'…' if len(reply) > 60 else ''}")
        print()


def _run_live_checks(
    base_url: str,
    runs: list[UserRun],
    five_xx_before: dict[tuple[str, str, str], float] | None,
) -> list[Check]:
    """跑**需要现场还在**的判据 —— 必须在清理之前调用。

    ⚠️ 这个函数的调用位置不是风格问题，是正确性问题。判据 6（会话记载不串号）
    和判据 7（跨用户不可见）读的都是**服务端现存的**会话记录；一旦
    ``_cleanup_all`` 跑过，它们读回来的是 404 —— 于是一条本该通过的判据会报
    「会话不存在」，而那个原因看上去与「串号」毫无关系。

    （这不是假设：本脚本的第一版就是把清理放在这些判据之前，实测三条会话
    全部 404。判据本身没错，是**取证顺序**错了 —— 证据删掉了才去取证。）

    Args:
        base_url (`str`): 服务地址。
        runs (`list[UserRun]`): 全部观测。
        five_xx_before (`dict | None`): 运行**前**抓的 5xx 快照，由 ``main``
            在发出任何请求之前采集并传进来（只有那里知道「运行前」是什么时候）。

    Returns:
        `list[Check]`: 判据结果，顺序即报告顺序（清理前那一组）。
    """
    return [
        check_setup(runs),
        check_completed(runs),
        check_replies_nonempty(runs),
        check_replies_are_answers(runs),
        check_no_error_events(runs),
        check_genuinely_concurrent(runs),
        check_stream_isolation(runs),
        check_transcript_isolation(base_url, runs),
        check_cross_user_invisible(base_url, runs),
        check_no_server_errors(base_url, runs, five_xx_before),
        check_business_api(base_url, runs),
    ]


def _print_report(checks: list[Check]) -> int:
    """打印判据结果并返回退出码。

    ⚠️ 三种图标对应三种严重程度，别把它们合并成两种（理由见 :class:`Check`）：
      · ✅ 通过；
      · ⚠️ **告警** —— 有问题，但不是并发/功能故障（目前只有判据 2b 的
        「内心独白漏进回复」）。它**不**影响退出码，但必须在报告里看得见；
      · ❌ 失败 —— 退出码 1。

    Args:
        checks (`list[Check]`): 判据结果。

    Returns:
        `int`: 0 = 全过（可有告警）；1 = 有失败。
    """
    print("▶ 判据")
    print()
    failed = 0
    warned = 0
    for check in checks:
        if check.ok:
            print(f"  ✅ {check.name}")
        elif check.warn:
            warned += 1
            print(f"  ⚠️ {check.name}（告警，不计入失败）")
        else:
            failed += 1
            print(f"  ❌ {check.name}")
        if check.detail:
            # ⚠️ 通过的判据也要打印 detail（与 smoke.py 同一个决定）：
            #    「通过」的信息量全在细节里 —— 比如「同时在飞 3.4 秒」。
            #    只打一行 ✅，读的人无从判断它到底验了什么。
            print(f"      {check.detail}")
    print()
    if failed:
        print(f"❌ 并发测试未通过：{len(checks)} 条判据里有 {failed} 条没过")
        return 1
    if warned:
        # ⚠️ 这一行必须存在。「全部通过」后面跟一个不带说明的告警，
        #    读的人会以为告警是装饰 —— 而它记的是一个真实缺陷。
        print(
            f"✅ 并发测试通过：{len(checks)} 条判据全部通过"
            f"（另有 {warned} 条告警，见上）"
        )
        return 0
    print(f"✅ 并发测试通过：{len(checks)} 条判据全部通过")
    return 0


# ==============================================================================
# CLI
# ==============================================================================


def _positive_int(raw: str) -> int:
    """argparse 的 ``type``：正整数（``>= 1``）。

    Args:
        raw (`str`): 原始参数文本。

    Returns:
        `int`: 解析结果。

    Raises:
        argparse.ArgumentTypeError: 不是正整数。

    ⚠️ 下界刻意是 **1**，不是 2。本函数曾**同时**给 ``--users`` 与
    ``--rounds`` 用，下界写死成 2，于是 ``--rounds 1`` 被拒 —— 而
    ``DEFAULT_ROUNDS`` **就是 1**：默认值是一个解析器自己不肯接受的值，
    谁照着 ``--help`` 把默认值显式写一遍，谁就报错；报错文案还说的是
    「1 个用户谈不上并发」，把问题指向了另一个参数。

    现在两个参数各用各的校验器：本函数管「正整数」这件事本身，
    「至少要 2 个用户」这条业务约束归 :func:`_at_least_two_users`。
    """
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"需要一个正整数，收到 {raw!r}") from exc
    if value < 1:
        raise argparse.ArgumentTypeError(f"必须是正整数，收到 {value}")
    return value


def _at_least_two_users(raw: str) -> int:
    """argparse 的 ``type``：``--users`` 专用，至少 2 个用户。

    Args:
        raw (`str`): 原始参数文本。

    Returns:
        `int`: 解析结果。

    Raises:
        argparse.ArgumentTypeError: 不是正整数，或小于 2。

    下界为什么是 2：一个用户**谈不上并发** —— 本脚本的全部判据
    （同时在飞、互不串号、互相看不见）都需要至少两个当事人才成立。
    这条约束只对 ``--users`` 有效；``--rounds`` 为 1 完全合理。
    """
    value = _positive_int(raw)
    if value < 2:
        raise argparse.ArgumentTypeError(
            f"--users 必须 >= 2，收到 {value}（1 个用户谈不上并发）"
        )
    return value


def _build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    Returns:
        `argparse.ArgumentParser`: 解析器。
    """
    parser = argparse.ArgumentParser(
        description="AliGo 差旅助手 —— 多用户同时发不同请求的并发正确性测试。",
        epilog=(
            "示例：\n"
            "  python scripts/concurrency_test.py\n"
            "  python scripts/concurrency_test.py --users 6 --rounds 2\n"
            "  python scripts/concurrency_test.py --keep      # 留现场排查\n"
            "  make concurrency\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"服务根地址（默认 {DEFAULT_BASE_URL}）",
    )
    parser.add_argument(
        "--users",
        type=_at_least_two_users,
        default=DEFAULT_USERS,
        help=f"并发用户数（默认 {DEFAULT_USERS}）。超过 3 个时场景轮转复用。",
    )
    parser.add_argument(
        "--rounds",
        type=_positive_int,
        default=DEFAULT_ROUNDS,
        help=f"每人轮数（默认 {DEFAULT_ROUNDS}）。第 k 轮由全部用户同时发出。",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="按要求保留本轮产物（默认跑完自己删掉）。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv (`list[str] | None`): 命令行参数；``None`` 时取 ``sys.argv[1:]``。

    Returns:
        `int`: 退出码。
    """
    args = _build_parser().parse_args(argv)

    print(
        f"▶ 并发正确性测试：{args.users} 个用户 × {args.rounds} 轮，同时发不同请求"
    )
    print(f"  靶机：{args.base_url}")
    print()

    # ★ 5xx 基线必须在**发出任何请求之前**抓 —— 判据 8 用的是增量，
    #   基线晚一步就会把本脚本自己（或环境准备阶段）造的 5xx 算漏。
    five_xx_before = _read_5xx_counts(args.base_url)
    if five_xx_before is None:
        print("  ⚠️ 读不到 /metrics 的 5xx 计数，判据 8 只会验客户端侧。")
        print()

    runs = run_concurrency(args.base_url, args.users, args.rounds)
    _print_runs(runs)

    # ★ 取证必须在清理之前 —— 见 _run_live_checks 的文档。
    checks = _run_live_checks(args.base_url, runs, five_xx_before)

    cleanup_problems = _cleanup_all(args.base_url, runs, keep=args.keep)
    if args.keep:
        print("  ℹ️ 按要求**保留**了本轮产物（--keep），未做任何删除。")
        print()
    checks.append(check_cleanup(args.base_url, runs, cleanup_problems, keep=args.keep))

    return _print_report(checks)


if __name__ == "__main__":
    raise SystemExit(main())
