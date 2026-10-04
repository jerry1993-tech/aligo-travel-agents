# -*- coding: utf-8 -*-
"""trace 装配（``src/observability/tracing.py``）的行为测试。

==============================================================================
这份测试为什么长这样：一句话——**全局 TracerProvider 是进程级单例且只能设一次**
==============================================================================
    ``tracing.py`` 里最关键的一句是 ``otel_trace.set_tracer_provider(provider)``。
    它在 OpenTelemetry 里由一把一次性的锁保护：**同一进程内第二次调用会被静默忽略**
    （只打一句 "Overriding of current TracerProvider is not allowed"）。
    后果是：一旦某条用例真的把 provider 装上去了，同一进程里**后面所有用例**
    看到的就是那个旧的 provider ——
    「本该跳过」的用例会因为全局已被设过而表现异常，反之亦然。

    因此这里的用例分成两类，边界划得很清楚：

      · **不注册 provider 的分支**（``none`` / ``otlp`` 缺密钥 / ``otlp`` 端点为空）
        在**同进程**里跑。它们根本不碰全局 provider，只用 fixture 复原模块级状态
        （``_provider`` / ``_shutdown`` / ``_last_status``）即可。

      · **真的会注册 provider 的分支**（``console`` / ``otlp`` 成功档）在**独立子进程**
        里跑。子进程一退出，它动过的全局状态随进程一起消失，主测试进程的
        OTel 全局状态**一个字节都没变**。这不是为了「隔离得漂亮」，而是唯一
        能同时测「装上之后 ``is_tracing_active()`` 真的是 True」与「不污染其它用例」
        的办法 —— 恢复模块级 ``_provider`` 变量**救不了**全局 provider。

    ⚠️ 这也是为什么本文件在跑 ``console``/``otlp`` 用例时**不**直接调用
    ``setup_tracing``：那样做能让 ``make test`` 里排在后面的
    ``test_probes.py`` / ``test_server_assembly.py`` 全部拿到一个「已经装好的 trace」，
    在不相关的用例里引入 console 导出器的后台线程与输出。别图省事改回去。

==============================================================================
子进程用例的另一处刻意设计：在快照里多取两格
==============================================================================
    子进程除了回传 ``describe_tracing`` 的完整输出，还额外塞了三个派生字段：

      · ``provider_returned``    —— 装配函数的返回值是否为 provider 实例；
      · ``active_after_setup``   —— 装配后的 ``is_tracing_active()``；
      · ``active_after_shutdown``—— ``shutdown_tracing()`` **之后**的 ``is_tracing_active()``。

    最后一格尤其重要：关闭过的 provider，全局状态**仍是那个 SDK 实例**，
    所以框架自己的 ``_check_tracing_enabled()`` 依旧返回 True —— 若不额外记录
    ``_shutdown``，``/readyz`` 会在关停中的实例上报告「trace 正常」，
    把真正的症状盖成一个假的好消息。这条用例把那个陷阱钉死在测试里。
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from collections.abc import Iterator

import pytest

from src.config import Settings, load_settings, repo_root
from src.observability import tracing

# ==============================================================================
# 常量与工具
# ==============================================================================

#: 子进程与同进程用例共用的基准环境视图。
#:
#: 对齐 ``tests/conftest.py::TEST_ENVIRON`` 的取舍：**逐项显式列出**，
#: 让配置类用例可重复（不受开发机 ``.env`` / 真实进程环境影响），
#: 并把 LLM key 钉成空串以强制走 MockLLM。这里没有导出器——每条用例
#: 再叠自己的 observability 覆盖项。
_BASE_ENVIRON: dict[str, str] = {
    "ALIGO__APP__ENV": "test",
    "ALIGO__APP__LOG_LEVEL": "WARNING",
    "ALIGO__DB__URL": "sqlite+aiosqlite:///:memory:",
    "ALIGO__REDIS__URL": "redis://:test-password@localhost:6379/0",
    "ALIGO__LLM__API_KEY": "",
    "ALIGO__LLM__BASE_URL": "",
    "ALIGO__OBSERVABILITY__TRACE_EXPORTER": "none",
    "ALIGO__OBSERVABILITY__OTLP_ENDPOINT": "",
}

#: 子进程回传快照时输出行的前缀。用一个足够独特的标记，
#: 避免把子进程里别的日志（tracing.py 自己的 WARNING）当成数据行。
_SNAPSHOT_MARKER = "__ALIGO_TRACING_SNAPSHOT__"

#: Langfuse v3 的 OTLP 端点（与 compose / 文档里的一致）。
_OTLP_ENDPOINT = "http://langfuse:3000/api/public/otel/v1/traces"

#: 用于「不得泄漏」断言的哨兵密钥。取一个别的用例不可能碰巧命中的串。
_CANARY_PUBLIC_KEY = "pk-canary-observability-0001"
_CANARY_SECRET_KEY = "sk-canary-observability-0002"

#: 子进程内执行的脚本：装配 → 取快照 → 关闭 → 再取一次活跃状态。
#:
#: 通过环境变量拿到基准环境与本次覆盖项（都是 JSON），而不是拼字符串——
#: 这样密钥/端点里即便含引号也不会把脚本拼坏。
#: ``<MARKER>`` 占位符在 :func:`_run_subprocess_case` 里替换。
_SUBPROCESS_CODE = """
import json, os

from src.config import load_settings
from src.observability import (
    describe_tracing,
    is_tracing_active,
    setup_tracing,
    shutdown_tracing,
)

base = json.loads(os.environ["ALIGO_TEST_BASE_ENVIRON"])
base.update(json.loads(os.environ["ALIGO_TEST_ENV_OVERRIDES"]))
settings = load_settings("test", environ=base, dotenv=False)

provider = setup_tracing(settings)
snapshot = describe_tracing(settings)
snapshot["provider_returned"] = provider is not None
snapshot["active_after_setup"] = is_tracing_active()

shutdown_tracing()
snapshot["active_after_shutdown"] = is_tracing_active()

print("<MARKER>" + json.dumps(snapshot))
"""


def _obs_settings(**overrides: str) -> Settings:
    """构造一份只改 observability 相关项的测试配置。

    Args:
        **overrides (`str`): 形如 ``ALIGO__OBSERVABILITY__TRACE_EXPORTER="otlp"``
            的环境变量覆盖项。

    Returns:
        `Settings`: 通过严格校验的配置对象。
    """
    env = dict(_BASE_ENVIRON)
    env.update(overrides)
    return load_settings("test", environ=env, dotenv=False)


def _run_subprocess_case(overrides: dict[str, str]) -> dict[str, object]:
    """在**独立子进程**里装配 trace 并回传状态快照。

    为什么必须是子进程：见本文件顶部的长注释。被装配的全局 TracerProvider
    无法在同进程内复原，只有让改动随子进程一起消失才是干净的。

    Args:
        overrides (`dict[str, str]`): 在基准环境之上叠加的 observability 覆盖项。

    Returns:
        `dict[str, object]`: ``describe_tracing`` 的输出，外加三个派生字段
            （``provider_returned`` / ``active_after_setup`` / ``active_after_shutdown``）。
    """
    repo = repo_root()
    env = os.environ.copy()
    # 让子进程能 `import src` —— 它不经过 pytest 的 pythonpath 配置。
    env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
    env["ALIGO_TEST_BASE_ENVIRON"] = json.dumps(_BASE_ENVIRON)
    env["ALIGO_TEST_ENV_OVERRIDES"] = json.dumps(overrides)

    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_CODE.replace("<MARKER>", _SNAPSHOT_MARKER)],
        cwd=str(repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"子进程装配失败（rc={proc.returncode}）：\nSTDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr[-3000:]}"
    )

    for line in proc.stdout.splitlines():
        if line.startswith(_SNAPSHOT_MARKER):
            return json.loads(line[len(_SNAPSHOT_MARKER):])

    raise AssertionError(
        f"子进程没有输出快照行（标记 {_SNAPSHOT_MARKER!r}）。\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr[-3000:]}",
    )


@pytest.fixture(autouse=True)
def _restore_tracing_module_state() -> Iterator[None]:
    """每条用例前后复原 ``tracing`` 模块级的可变状态。

    ⚠️ 它**只**复原模块变量（``_provider`` / ``_shutdown`` / ``_last_status``），
    **复原不了** OTel 的全局 provider —— 那正是 ``console`` / ``otlp`` 用例
    必须走子进程的原因。对不注册 provider 的用例而言，这份 fixture 足够：
    没有它时，某条用例写下的 ``_last_status``（用来描述「上一次装配结果」）
    会串到下一条用例的 ``describe_tracing`` 里，制造出难以复现的假失败。

    Yields:
        `None`: 用例执行期间可自由改动模块状态。
    """
    saved_provider = tracing._provider
    saved_shutdown = tracing._shutdown
    saved_status = dict(tracing._last_status)
    yield
    tracing._provider = saved_provider
    tracing._shutdown = saved_shutdown
    # 就地恢复，保持字典对象 identity 不变（别的地方持有的引用要能看到复原）。
    tracing._last_status.clear()
    tracing._last_status.update(saved_status)


# ==============================================================================
# 一、四条分支
# ==============================================================================
def test_none_branch_skips_and_reports_why() -> None:
    """``trace_exporter=none``：不装配导出器，且原因可自解释。

    ``none`` 是**默认档也是 core 档的常态**，所以它不该发出任何告警级别噪音，
    但 ``describe_tracing`` 仍要说清「没装是因为你没开」，而不是留一个空白的 ``reason``
    —— 否则排障时看到 ``configured=False`` 会分不清「没配」与「配了但坏了」。
    """
    settings = _obs_settings()

    provider = tracing.setup_tracing(settings)

    assert provider is None, "none 档不应返回 provider"
    assert tracing.is_tracing_active() is False

    described = tracing.describe_tracing(settings)
    assert described["trace_exporter"] == "none"
    assert described["otlp_ready"] is False
    assert described["credentials_configured"] is False
    assert described["exporter_configured"] is False
    assert described["exporter"] is None
    assert described["shutdown"] is False
    assert described["active"] is False
    assert "none" in str(described["reason"])


def test_console_branch_registers_real_provider() -> None:
    """``trace_exporter=console``：真的注册了 SDK provider，框架埋点会执行。

    这条跑在子进程里（理由见文件顶部）。它钉住三件事：
    装配函数返回 provider、``is_tracing_active()`` 变 True、关闭后转回 False。
    """
    snap = _run_subprocess_case(
        {"ALIGO__OBSERVABILITY__TRACE_EXPORTER": "console"},
    )

    assert snap["provider_returned"] is True
    assert snap["active_after_setup"] is True, "console 档装上后框架埋点必须真的执行"
    assert snap["exporter"] == "console"
    assert snap["exporter_configured"] is True
    # 关闭之后必须转为不活跃 —— 否则关停中的实例会在 /readyz 里谎报「trace 正常」。
    assert snap["active_after_shutdown"] is False


def test_otlp_branch_registers_otlp_exporter() -> None:
    """``otlp`` + 端点 + 密钥齐全：装配成功，导出通道为 otlp。

    这条跑在子进程里。它同时是「密钥不泄漏」在**真实装配路径**上的检查点：
    快照（含 ``reason`` 与 ``otlp_endpoint``）里不得出现任何一个密钥串，
    也不得出现 Basic 认证头的 base64 形态。
    """
    snap = _run_subprocess_case(
        {
            "ALIGO__OBSERVABILITY__TRACE_EXPORTER": "otlp",
            "ALIGO__OBSERVABILITY__OTLP_ENDPOINT": _OTLP_ENDPOINT,
            "ALIGO__OBSERVABILITY__LANGFUSE_PUBLIC_KEY": _CANARY_PUBLIC_KEY,
            "ALIGO__OBSERVABILITY__LANGFUSE_SECRET_KEY": _CANARY_SECRET_KEY,
        },
    )

    assert snap["provider_returned"] is True
    assert snap["active_after_setup"] is True
    assert snap["active_after_shutdown"] is False
    assert snap["exporter"] == "otlp"
    assert snap["exporter_configured"] is True
    assert snap["otlp_ready"] is True
    assert snap["credentials_configured"] is True
    # 端点本身不是密钥，展示它对排障价值很大（写错域名是最常见的失误）。
    assert snap["otlp_endpoint"] == _OTLP_ENDPOINT
    assert "OTLP" in str(snap["reason"])

    blob = json.dumps(snap, ensure_ascii=False)
    assert _CANARY_PUBLIC_KEY not in blob, "describe/快照里泄漏了 public key"
    assert _CANARY_SECRET_KEY not in blob, "describe/快照里泄漏了 secret key"
    auth_header = base64.b64encode(
        f"{_CANARY_PUBLIC_KEY}:{_CANARY_SECRET_KEY}".encode(),
    ).decode()
    assert auth_header not in blob, "快照里出现了 Basic 认证头的 base64 形态"


def test_otlp_without_keys_is_skipped() -> None:
    """``otlp`` 但密钥为空：**跳过装配**（而不是装一个必然 401 的导出器）。

    ⚠️ 这条钉的是本项目最需要「大声说出来」的一个决策：Langfuse v3 的 OTLP
    端点强制 Basic 认证，缺认证头会得 401，而 OTLP 导出器只重试 408/5xx ——
    401 直接被丢弃且不重试。所以「缺密钥」时的正确做法是干脆不装、并把原因写进
    ``reason``，让 ``/readyz`` 能回答「为什么一条 trace 都没有」。

    同时它验证了 ``describe_tracing`` 把「配置**想**发」（``otlp_ready=True``）
    与「实际发得出去」（``exporter_configured=False``）分成两个字段 ——
    两者混为一谈时，「开了 otlp 却没数据」这种最常见的困惑就无法从探针上读出来。
    """
    settings = _obs_settings(
        ALIGO__OBSERVABILITY__TRACE_EXPORTER="otlp",
        ALIGO__OBSERVABILITY__OTLP_ENDPOINT=_OTLP_ENDPOINT,
        # 两个密钥保持为空。
    )

    provider = tracing.setup_tracing(settings)

    assert provider is None, "缺密钥时必须跳过装配、不注册 provider"
    assert tracing.is_tracing_active() is False

    described = tracing.describe_tracing(settings)
    # 「想发」为真、「发得出去」为假 —— 这一对不一致本身就是最有用的排障信号。
    assert described["otlp_ready"] is True
    assert described["exporter_configured"] is False
    assert described["exporter"] is None
    # 配置意图仍是 otlp：说明「这一行来自一次 otlp 配置，只是没装配成功」。
    assert described["trace_exporter"] == "otlp"
    assert described["credentials_configured"] is False
    reason = str(described["reason"])
    assert "缺少" in reason
    assert "langfuse_public_key" in reason
    assert "langfuse_secret_key" in reason


def test_otlp_with_only_one_key_is_skipped() -> None:
    """"只配了一半密钥"同样必须跳过 —— 半套凭据带不来认证成功。

    这是一个很容易漏的边界：认证头是 ``public:secret`` 拼出来的，
    少任何一半都只会得到 401。若代码只判「两个里有任意一个非空就装」，
    症状会和完全没配密钥一样（Langfuse 里一条 trace 都没有），却少了一句告警。
    """
    settings = _obs_settings(
        ALIGO__OBSERVABILITY__TRACE_EXPORTER="otlp",
        ALIGO__OBSERVABILITY__OTLP_ENDPOINT=_OTLP_ENDPOINT,
        ALIGO__OBSERVABILITY__LANGFUSE_PUBLIC_KEY=_CANARY_PUBLIC_KEY,
        # secret 故意留空。
    )

    provider = tracing.setup_tracing(settings)

    assert provider is None
    assert tracing.is_tracing_active() is False

    described = tracing.describe_tracing(settings)
    assert described["exporter_configured"] is False
    assert described["credentials_configured"] is False
    reason = str(described["reason"])
    assert "langfuse_secret_key" in reason, "应点名缺的那一个是 secret key"
    assert "langfuse_public_key" not in reason


def test_otlp_with_empty_endpoint_is_skipped() -> None:
    """``otlp`` 但端点为空：跳过，并点名 ``otlp_endpoint``。

    端点为空的跳过发生在「检查密钥」**之前**（见 ``_build_exporter`` 的分支顺序）——
    这是有意的：没有端点时连「往哪发」都不知道，先报端点缺失比报密钥缺失更贴近
    用户手上的实际问题（``.env`` 里忘了填 ``OTLP_ENDPOINT``）。
    """
    settings = _obs_settings(
        ALIGO__OBSERVABILITY__TRACE_EXPORTER="otlp",
        ALIGO__OBSERVABILITY__OTLP_ENDPOINT="",
        # 密钥给全，确保跳过是「因为端点空」而不是「因为密钥缺」。
        ALIGO__OBSERVABILITY__LANGFUSE_PUBLIC_KEY=_CANARY_PUBLIC_KEY,
        ALIGO__OBSERVABILITY__LANGFUSE_SECRET_KEY=_CANARY_SECRET_KEY,
    )

    provider = tracing.setup_tracing(settings)

    assert provider is None
    assert tracing.is_tracing_active() is False

    described = tracing.describe_tracing(settings)
    assert described["otlp_ready"] is False, "端点为空的 otlp 不具备装配条件"
    assert described["exporter_configured"] is False
    assert described["exporter"] is None
    # 密钥齐全，但端点缺失 ⇒ 凭据项应为 True，与「缺密钥」那一档区分开。
    assert described["credentials_configured"] is True
    assert "otlp_endpoint" in str(described["reason"])


def test_otlp_endpoint_is_stripped_in_describe() -> None:
    """端点两侧的空白必须被 strip —— 否则会变成一个永远连不上的地址。

    ``.env`` 里手抄 URL 时尾部带一个空格是极常见的失误；若原样拿去建导出器，
    得到的请求会打到 ``"…/traces "`` 而失败。``describe_tracing`` 展示的应是
    真正会被使用的那个值。
    """
    settings = _obs_settings(
        ALIGO__OBSERVABILITY__TRACE_EXPORTER="otlp",
        ALIGO__OBSERVABILITY__OTLP_ENDPOINT=f"  {_OTLP_ENDPOINT}  ",
        ALIGO__OBSERVABILITY__LANGFUSE_PUBLIC_KEY=_CANARY_PUBLIC_KEY,
        ALIGO__OBSERVABILITY__LANGFUSE_SECRET_KEY=_CANARY_SECRET_KEY,
    )

    # 只读 describe，不真的装配（装配会注册全局 provider，须留给子进程用例）。
    described = tracing.describe_tracing(settings)

    assert described["otlp_endpoint"] == _OTLP_ENDPOINT
    assert described["otlp_ready"] is True


# ==============================================================================
# 二、is_tracing_active 的判据
# ==============================================================================
def test_is_tracing_active_is_false_before_any_setup() -> None:
    """没有任何装配时，``is_tracing_active()`` 必须是 **False**。

    ⚠️ 这是本模块存在的核心动机。默认的全局 provider 是 ``ProxyTracerProvider``，
    框架的 ``_check_tracing_enabled()`` 对它返回 False，于是 ``TracingMiddleware``
    **静默透传**（不报错、不打日志）。把这条默认状态测出来，等于给
    「最危险的静默短路」设了一道基线：如果哪天有人误注册了一个不带导出器的
    provider，本用例会立刻变红，而不是等到「Langfuse 里一条都没有」时才发现。
    """
    # 前提断言：本进程（即主测试进程）从未装配过真实的 provider。
    # 若这条失败，说明有别的用例在同进程里注册了 provider，污染了全局状态。
    from opentelemetry import trace as otel_trace

    assert not isinstance(otel_trace.get_tracer_provider(), tracing.TracerProvider), (
        "主测试进程里出现了 SDK TracerProvider —— 说明有用例在**同进程**装配了 trace，"
        "破坏了本文件「注册 provider 一律走子进程」的约定。"
    )
    assert tracing.is_tracing_active() is False


# ==============================================================================
# 三、describe_tracing 的字段完整性
# ==============================================================================
def test_describe_tracing_has_exactly_the_expected_fields() -> None:
    """``describe_tracing`` 的键集合必须**恰好**是契约里那几个。

    它有两个消费者：``/readyz`` 的 ``extra.tracing`` 与运维脚本。
    少一个字段会让下游客端 KeyError；多一个字段则常常是「顺手把密钥也算进去了」
    的开始。因此这里用 `==` 钉死整张键表，而不是逐项 `in`。
    """
    settings = _obs_settings()

    described = tracing.describe_tracing(settings)

    assert set(described) == {
        "trace_exporter",
        "otlp_endpoint",
        "otlp_ready",
        "credentials_configured",
        "exporter_configured",
        "exporter",
        "shutdown",
        "active",
        "reason",
    }
    # 类型约定：三个布尔、一个可空字符串。
    assert isinstance(described["active"], bool)
    assert isinstance(described["exporter_configured"], bool)
    assert isinstance(described["shutdown"], bool)
    assert described["exporter"] is None or isinstance(described["exporter"], str)


def test_describe_active_reflects_is_tracing_active() -> None:
    """``describe_tracing()["active"]`` 必须与 ``is_tracing_active()`` 完全一致。

    两者若漂移，``/readyz`` 展示的就会是另一个问题的答案 ——
    例如「探针说 active 但框架其实在空转」，正是本模块最想消灭的那类幻觉。
    """
    settings = _obs_settings()

    described = tracing.describe_tracing(settings)

    assert described["active"] is tracing.is_tracing_active()


# ==============================================================================
# 四、不泄漏密钥
# ==============================================================================
def test_describe_never_leaks_credentials_even_when_configured() -> None:
    """即使 otlp 密钥**已配置**，``describe_tracing`` 的输出里也不得出现它们的子串。

    ⚠️ 这是一个真实的泄漏面：``/readyz`` 是**无鉴权**的（编排系统不能带凭据探活），
    而 ``describe_tracing`` 的整个输出会原样挂进它的响应体。密钥一旦从这里出去，
    任何能访问该端口的人都能读到，而且 ``/readyz`` 的结果常被粘贴进工单与聊天记录。

    ``describe_tracing`` 因此**刻意**只给出布尔量（``credentials_configured``）
    与不含密钥的说明文本。本用例把这条性质钉死：把两个哨兵密钥配全，
    断言 public key、secret key、以及 ``Basic base64`` 认证头三者**都不出现**。
    """
    settings = _obs_settings(
        ALIGO__OBSERVABILITY__TRACE_EXPORTER="otlp",
        ALIGO__OBSERVABILITY__OTLP_ENDPOINT=_OTLP_ENDPOINT,
        ALIGO__OBSERVABILITY__LANGFUSE_PUBLIC_KEY=_CANARY_PUBLIC_KEY,
        ALIGO__OBSERVABILITY__LANGFUSE_SECRET_KEY=_CANARY_SECRET_KEY,
    )

    blob = json.dumps(tracing.describe_tracing(settings), ensure_ascii=False)

    assert _CANARY_PUBLIC_KEY not in blob
    assert _CANARY_SECRET_KEY not in blob
    auth_header = base64.b64encode(
        f"{_CANARY_PUBLIC_KEY}:{_CANARY_SECRET_KEY}".encode(),
    ).decode()
    assert auth_header not in blob, "输出里出现了 Basic 认证头的 base64 形态"
    # 反而应当能看到「凭据已配置」这个布尔事实 —— 否则排障又没了着力点。
    assert tracing.describe_tracing(settings)["credentials_configured"] is True
