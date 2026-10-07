# -*- coding: utf-8 -*-
"""模型装配（``src/llm/factory.py``）与熔断器（``src/llm/breaker.py``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    模型层有两个「只在生产才咬人」的失败模式，本地开发几乎碰不到：

      1. **降级判据失效** —— 机器上恰好有 key 时一切正常，
         换到没有 key 的 CI/客户机器上就崩。症状是 ImportError 或 401，
         而配置看起来完全正确（``use_mock_when_no_key: true`` 明明写着）。
      2. **熔断器不生效** —— 熔断的整个意义是「下游挂了就别再打它」。
         写错的熔断器（比如每个会话各持一个实例、或阈值判断写反）
         在正常流量下**没有任何异常表现**，只有下游真的挂掉时才暴露 ——
         而那时它本该救场，实际却让故障雪上加霜。

    用例围绕这两条来写：每条判据都有一条「正例 + 反例」，
    否则一个「无脑返回 Mock」的实现也能通过全部用例。
"""

from __future__ import annotations

import asyncio

import pytest

from src.config import Settings, load_settings
from src.llm.breaker import CircuitBreaker, CircuitBreakerOpen, CircuitState
from src.llm.factory import (
    build_chat_model,
    build_credential,
    describe_model_target,
    get_breaker,
    reset_breaker,
    should_use_mock,
)


def _settings(**overrides: str) -> Settings:
    """构造一份测试配置，并允许按 ``ALIGO__`` 环境变量的形式覆盖。

    Args:
        **overrides (`str`): 形如 ``ALIGO__LLM__API_KEY="sk-x"`` 的覆盖项。

    Returns:
        `Settings`: 完整配置对象。
    """
    from tests.conftest import TEST_ENVIRON

    environ = {**TEST_ENVIRON, **overrides}
    return load_settings("test", environ=environ, dotenv=False)


# ==============================================================================
# 一、零密钥降级
# ==============================================================================
def test_mock_is_used_when_no_api_key() -> None:
    """没有密钥 + 开关打开 ⇒ 降级为 Mock。

    这是「零密钥可运行」这条验收的直接体现。conftest 把 api_key 钉成了空串，
    因此这份配置在任何机器上（包括开发者的、恰好配了真 key 的机器）
    都会走 Mock —— 用例本身不受本机环境变量影响。
    """
    settings = _settings()

    assert should_use_mock(settings) is True
    assert build_chat_model(settings).__class__.__name__ == "MockChatModel"


def test_mock_is_used_when_key_is_whitespace_only() -> None:
    """密钥**全是空白字符**也要算作「没有密钥」。

    ⚠️ 这不是吹毛求疵：``.env`` 里写 ``OPENAI_API_KEY=`` 后面跟几个空格，
    或从某个系统复制粘贴时带上了不可见字符，都会产生「非空但无效」的值。
    如果判据只写 ``bool(api_key)``，这种情况会被判成「有密钥」⇒
    不降级 ⇒ 拿这段空白去请求 ⇒ 401。

    症状是最难归因的那一类：「配置看起来配了 key，也确实是 key 的位置，
    就是认证失败」。
    """
    settings = _settings(**{"ALIGO__LLM__API_KEY": "   "})

    assert should_use_mock(settings) is True


def test_mock_is_not_used_when_real_key_is_present() -> None:
    """有密钥时**必须**不降级。

    反例用例。少了它，一个「无条件返回 Mock」的实现也能通过上面两条 ——
    而那意味着生产环境永远在用假模型，且**没有任何报错**。
    """
    settings = _settings(**{"ALIGO__LLM__API_KEY": "sk-test-not-a-real-key"})

    assert should_use_mock(settings) is False
    assert build_chat_model(settings).__class__.__name__ != "MockChatModel"


@pytest.mark.parametrize(
    ("provider", "expected_env_hint"),
    [("dashscope", "DASHSCOPE_API_KEY"), ("openai", "OPENAI_API_KEY")],
)
def test_contradictory_config_raises_at_assembly(
    provider: str,
    expected_env_hint: str,
) -> None:
    """显式关闭降级却又不给密钥 ⇒ **装配期**就报错。

    ⚠️ 这里要的是「尽早失败」而不是「运行时失败」：
    若等到第一次对话才发现，那么一个配置写错的实例会**正常启动、
    正常通过健康检查、正常接收流量**，然后在用户面前报错。
    装配期报错则会让容器启动失败 —— 编排系统立刻能看见，也不会有人用到它。

    用例同时断言错误文案给出了**两条出路**（设 key 或开降级）：
    一件只能靠读源码才能解决的配置问题，等于没被解决。

    ⚠️ 这里对两个 provider 各跑一遍，断言的是**提示里的环境变量名跟着 provider 走**。
    写死一个名字的版本在只有一档 provider 时看着没问题，加档之后就会把
    DashScope 的用户指向 `OPENAI_API_KEY` —— 一个他压根没配、也不该配的变量。
    这类「文案错误」不会让任何测试变红，只会让排障的人多绕一圈。

    Args:
        provider (`str`): 被测 provider。
        expected_env_hint (`str`): 该 provider 对应的密钥环境变量名。
    """
    settings = _settings(
        **{
            "ALIGO__LLM__USE_MOCK_WHEN_NO_KEY": "false",
            "ALIGO__LLM__PROVIDER": provider,
        },
    )

    with pytest.raises(ValueError) as excinfo:
        build_credential(settings)

    message = str(excinfo.value)
    assert expected_env_hint in message
    assert "USE_MOCK_WHEN_NO_KEY" in message


# ==============================================================================
# 一之二、Provider 登记表 —— 凭据类与模型类必须成对
# ==============================================================================
@pytest.mark.parametrize(
    ("provider", "credential_cls", "model_cls", "default_base_url"),
    [
        # DashScope 的默认端点是**凭据类自带的**，配置留空时必须仍能拿到它 ——
        # 这是本项目不再需要 DASHSCOPE_BASE_URL 的原因。
        ("dashscope", "DashScopeCredential", "DashScopeChatModel",
         "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        # OpenAI 兼容档没有默认端点（由 SDK 决定官方地址），因此这里是 None。
        ("openai", "OpenAICredential", "OpenAIChatModel", None),
    ],
)
def test_provider_selects_the_matching_credential_and_model(
    provider: str,
    credential_cls: str,
    model_cls: str,
    default_base_url: str | None,
) -> None:
    """``provider`` 必须同时决定**凭据类**与**模型类**，且两者的内建端点一致。

    ⚠️ 为什么这条值得单独钉住：这两个类在框架的 app 链路里是**互相反查**的
    （credential 的 ``get_chat_model_class()`` 决定用哪个模型类）。配错对的症状是
    一个「用 A 家的密钥去请求 B 家端点」的 404/401 —— 从错误信息看像是密钥问题，
    而真正的错因是登记表里搭错了行。这类错误不会在正常调用路径上暴露。

    用例同时覆盖 ``base_url`` 留空的分支：那是**唯一的**会走到
    「省略 kwarg 让凭据类默认值生效」这条路径的写法，而它正是 DashScope 档的常态。
    写错的版本（传 None 或 ""）会让 DashScopeCredential 抛 ValidationError，
    症状是启动即崩、且错误信息指向 pydantic 而不是配置。
    """
    settings = _settings(
        **{
            "ALIGO__LLM__PROVIDER": provider,
            "ALIGO__LLM__API_KEY": "sk-test-not-a-real-key",
            "ALIGO__LLM__BASE_URL": "",  # ← 留空，走凭据类默认值
        },
    )

    credential = build_credential(settings)
    assert type(credential).__name__ == credential_cls

    model = build_chat_model(settings)
    assert type(model).__name__ == model_cls

    described = describe_model_target(settings)
    # 留空时 base_url 必须报出**真实有效**的端点，而不是空串/None
    # （openai 档由 SDK 决定，故为 None）。
    assert described["base_url"] == default_base_url


def test_unknown_provider_is_rejected_at_load_time() -> None:
    """未登记的 provider 必须在**配置加载期**被挡下，而不是等到装配模型。

    schema 用 ``Literal`` 声明取值，因此从环境变量写一个不存在的 provider
    会在 pydantic 校验阶段失败 —— 这正是我们想要的：
    容器启动失败远好过一个「起来了但每次对话都报 AttributeError」的实例。

    ⚠️ 断言的是 ``ValueError`` 而非 pydantic 的 ``ValidationError``：
    ``load_settings`` 会把 pydantic 的报错**包一层**，附上「哪几个配置文件参与、
    覆盖前缀是什么」的上下文（见 ``src/config/loader.py:575-579``）。
    直接断言 ValidationError 会穿透这层包装去测一个内部类型 ——
    而使用者真正看到的、也是真正该保证的，是那条带着配置文件路径的 ValueError。
    用例因此同时断言字段名出现在文案里：只说「校验失败」而不说是哪个字段，
    等于把定位工作原样丢回给用户。
    """
    with pytest.raises(ValueError) as excinfo:
        _settings(**{"ALIGO__LLM__PROVIDER": "not-a-real-provider"})

    message = str(excinfo.value)
    assert "llm.provider" in message
    # 合法取值要出现在文案里 —— 否则用户知道错了却不知道能填什么。
    assert "dashscope" in message
    assert "openai" in message


def test_model_target_description_never_contains_the_key() -> None:
    """``describe_model_target`` 的结果里**绝不能**有密钥。

    ⚠️ 这个函数的输出会进 ``/readyz`` 的响应体 —— 而 ``/readyz`` 是
    **无鉴权**的（编排系统不能带凭据来探活）。因此它一旦带上密钥，
    任何能访问该端口的人都能读到，且这个泄漏点看起来完全无害
    （函数名字里只有「describe」）。

    用例断言的是「密钥字符串不出现在序列化结果里」，而不是
    「某个字段为空」—— 后者挡不住「把它拼进了 model 名」这种写法。
    """
    secret = "sk-super-secret-value-should-never-appear"
    settings = _settings(
        **{
            "ALIGO__LLM__API_KEY": secret,
            "ALIGO__LLM__BASE_URL": "https://api.example.com/v1",
        },
    )

    described = describe_model_target(settings)

    assert secret not in repr(described), f"describe_model_target 泄漏了密钥：{described}"
    # 但要能看出「配没配 key」—— 否则这个描述函数就没有存在价值。
    assert described["api_key_configured"] is True
    assert described["base_url"] == "https://api.example.com/v1"


def test_model_target_description_reports_mock_mode() -> None:
    """降级状态下，描述里要能一眼看出「正在用 Mock」。

    排查「为什么回答这么奇怪」时，第一个要排除的就是「其实跑的是 Mock」。
    这个字段就是给那一刻准备的。
    """
    described = describe_model_target(_settings())

    assert described["api_key_configured"] is False
    assert described["using_mock"] is True


# ==============================================================================
# 二、熔断器 —— 状态机
# ==============================================================================
async def _outcome(breaker: CircuitBreaker) -> str:
    """试着通过一次熔断器，把结果归一成 ``"passed"`` / ``"rejected"``。

    ``allow_request()`` 的契约是「拒绝时**抛异常**」而不是「返回 False」——
    这是刻意的（详见 ``src/llm/breaker.py`` 的 docstring）：调用方必须显式
    处理拒绝，而不是靠一个容易被忽略的布尔返回值。
    这个 helper 把「异常」翻译回「取值」，好让用例能写出可读的断言。

    Args:
        breaker (`CircuitBreaker`): 被测熔断器。

    Returns:
        `str`: ``"passed"`` 或 ``"rejected"``。
    """
    try:
        await breaker.allow_request()
    except CircuitBreakerOpen:
        return "rejected"
    return "passed"


async def test_breaker_starts_closed() -> None:
    """新熔断器处于「闭合」状态，请求放行。"""
    breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=60.0)

    assert breaker.state is CircuitState.CLOSED
    assert await _outcome(breaker) == "passed"


async def test_breaker_opens_after_threshold_failures() -> None:
    """连续失败达到阈值 ⇒ 开路。

    ⚠️ 「连续」是关键：一次失败后成功一次，计数必须归零。
    否则一个「偶尔失败」的下游（失败率 5%）会在跑得足够久之后
    必然开路 —— 哪怕它从来没真正挂过。那样熔断器就从保护变成了故障源。
    """
    breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=60.0)

    for _ in range(3):
        await breaker.record_failure()

    assert breaker.state is CircuitState.OPEN
    assert await _outcome(breaker) == "rejected"


async def test_breaker_success_resets_the_failure_streak() -> None:
    """中间成功一次 ⇒ 失败计数归零，不会累积到阈值。

    这条与上一条配对，专门挡「计数只加不减」的实现。
    """
    breaker = CircuitBreaker(failure_threshold=3, recovery_seconds=60.0)

    await breaker.record_failure()
    await breaker.record_failure()
    await breaker.record_success()  # ← 清零
    await breaker.record_failure()
    await breaker.record_failure()

    assert breaker.state is CircuitState.CLOSED, (
        "失败计数没有在成功时归零 —— 偶发失败会被累积成一次熔断"
    )


async def test_breaker_rejects_fast_while_open() -> None:
    """开路期间 ``guard()`` 立刻抛 ``CircuitBreakerOpen``，**不打网络**。

    ⚠️ 「立刻」是全部要点：熔断的意义是把「等 60 秒超时」换成
    「微秒级失败」。用例同时断言异常里带了 ``retry_after_seconds``，
    好让调用方（和日志）知道还要等多久。
    """
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=60.0)
    await breaker.record_failure()

    with pytest.raises(CircuitBreakerOpen) as excinfo:
        async with breaker.guard():
            pytest.fail("熔断器开路时不应执行被保护的代码块")

    assert excinfo.value.name == "llm"
    assert 0 < excinfo.value.retry_after_seconds <= 60.0


async def test_breaker_guard_records_failure_on_exception() -> None:
    """``guard()`` 保护的代码块抛异常 ⇒ 自动记一次失败。

    这条覆盖的是「熔断器真的接在调用链上」：若 ``guard()`` 只做放行不做计数，
    它永远不会开路 —— 而所有其它用例都会通过，因为它们手动调了
    ``record_failure()``。**这是最容易漏掉的一条**：
    状态机测得很全，接线却是断的。
    """
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=60.0)

    with pytest.raises(RuntimeError):
        async with breaker.guard():
            raise RuntimeError("模拟下游调用失败")

    assert breaker.state is CircuitState.OPEN


async def test_breaker_guard_records_success_on_clean_exit() -> None:
    """``guard()`` 保护的代码块正常退出 ⇒ 自动记一次成功（失败计数归零）。"""
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=60.0)
    await breaker.record_failure()

    async with breaker.guard():
        pass

    assert breaker.state is CircuitState.CLOSED
    assert breaker.snapshot()["consecutive_failures"] == 0


async def test_breaker_closes_after_cooldown() -> None:
    """冷却期满后，下一次请求被放行（进入半开），成功则闭合。

    用例用一个极短的冷却期（0.05s）来避免真的等 30 秒 ——
    这是唯一需要「等待」的用例，因此刻意把它压到毫秒级。
    """
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0.05)
    await breaker.record_failure()
    assert await _outcome(breaker) == "rejected"

    await asyncio.sleep(0.06)

    assert await _outcome(breaker) == "passed", "冷却期满后应放行一次试探请求"
    assert breaker.state is CircuitState.HALF_OPEN

    await breaker.record_success()
    assert breaker.state is CircuitState.CLOSED


async def test_half_open_admits_exactly_one_probe() -> None:
    """半开状态下只放**一个**请求过去，其余一律拒绝。

    ⚠️ 这是熔断器最容易写错、且错误后果最严重的一处。
    半开时如果放行全部请求，等于「冷却期一到就把积压的流量
    全部砸向刚恢复的下游」—— 那正是把它再次打挂的最有效方式。
    熔断器本该保护下游，写错的版本反而成了压垮它的最后一击。

    正确语义：只放一个探针过去试探；它的结果决定是闭合还是重新开路，
    在它返回之前，其余请求继续快速失败。
    """
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0.05)
    await breaker.record_failure()
    await asyncio.sleep(0.06)

    outcomes = [await _outcome(breaker) for _ in range(3)]

    assert outcomes == ["passed", "rejected", "rejected"], (
        f"半开时放行了多个请求：{outcomes}。冷却期一到就把流量全砸回去，"
        f"会把刚恢复的下游再次打挂。"
    )


async def test_failed_probe_reopens_and_restarts_the_cooldown() -> None:
    """半开时的试探若失败 ⇒ 立刻回到开路，并**重新计时**冷却期。

    ⚠️ 「重新计时」这一点常被漏掉：若不重置 ``_opened_at``，
    冷却期会从**上一次**开路时算起 —— 于是一个持续失败的下游会被
    持续试探（每次都能立刻通过冷却检查），退化成「每来一个请求就打一次下游」。
    熔断器看起来在工作（状态确实是 OPEN），实际完全没起到保护作用。
    """
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0.1)
    await breaker.record_failure()
    await asyncio.sleep(0.11)

    assert await _outcome(breaker) == "passed"  # 试探
    await breaker.record_failure()  # 试探失败

    assert breaker.state is CircuitState.OPEN
    assert await _outcome(breaker) == "rejected", "试探失败后冷却期没有重新计时"


async def test_a_released_probe_lets_the_next_request_try() -> None:
    """★★★ ``release_probe()`` 必须把半开的名额还回去（否则**永久**卡死）。

    ⚠️ 这条用例挡的是一种无法自愈的故障：``allow_request()`` 放行试探时
    置 ``_probe_in_flight=True``，而只有 ``record_success`` / ``record_failure``
    会清掉它。调用方若在试探中途被取消、或抛出「不记账」的调用方错误，
    两个 record_* 都不会被调用 —— 名额被永久占住，此后每个请求都收到
    ``CircuitBreakerOpen(name, 0.0)``，且状态永远停在 HALF_OPEN、
    连冷却逻辑都不再推进，**只能靠重启进程恢复**。
    """
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0.05)
    await breaker.record_failure()
    await asyncio.sleep(0.06)

    assert await _outcome(breaker) == "passed"  # 试探名额被取走
    assert await _outcome(breaker) == "rejected"  # 其余请求仍然被拒

    await breaker.release_probe()  # 试探没有走完 ⇒ 归还名额

    assert await _outcome(breaker) == "passed", (
        "试探名额没有被归还 —— 熔断器会永久卡在 HALF_OPEN，"
        "此后所有请求都被拒绝且没有任何自愈路径。"
    )
    assert breaker.state is CircuitState.HALF_OPEN, "归还名额不该改变状态。"


async def test_release_probe_is_a_harmless_noop_outside_half_open() -> None:
    """在 CLOSED / OPEN 下调 ``release_probe`` 必须无害，且不改状态。

    ⚠️ 这条性质让调用方**不必先判断状态**再决定要不要归还 ——
    少一个判断就少一处「判断写反了」的机会（而写反的后果正是永久卡死）。
    """
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=60.0)

    await breaker.release_probe()
    assert breaker.state is CircuitState.CLOSED
    assert await _outcome(breaker) == "passed"

    await breaker.record_failure()  # ⇒ OPEN
    await breaker.release_probe()
    assert breaker.state is CircuitState.OPEN, "归还名额不该把开路状态改掉。"
    assert await _outcome(breaker) == "rejected"


async def test_guard_returns_the_probe_when_the_body_raises_breaker_open() -> None:
    """``guard()`` 体内抛出 ``CircuitBreakerOpen`` 时也要归还名额。

    ⚠️ 这是 ``guard()`` 里唯一「不记账就退出」的分支，也恰恰是漏掉归还
    最容易发生的地方：它看着像「原样重抛」的一行，而名额是隐式状态，
    不还回去不会有任何报错 —— 只会在下一次请求时表现为永久拒绝。
    """
    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=0.05)
    await breaker.record_failure()
    await asyncio.sleep(0.06)

    with pytest.raises(CircuitBreakerOpen):
        async with breaker.guard():
            raise CircuitBreakerOpen("来自内层的拒绝", 3.0)

    assert await _outcome(breaker) == "passed", (
        "guard() 退出时没有归还试探名额 —— 熔断器从此只能靠重启恢复。"
    )


async def test_breaker_snapshot_reports_counters() -> None:
    """快照要给出计数字段，供 ``/metrics`` 与 ``/readyz`` 使用。"""
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=60.0, name="llm")
    await breaker.record_failure()
    await breaker.record_failure()
    assert await _outcome(breaker) == "rejected"  # ⇒ total_rejections +1

    snapshot = breaker.snapshot()

    assert snapshot["name"] == "llm"
    assert snapshot["state"] == CircuitState.OPEN.value
    assert snapshot["total_failures"] == 2
    assert snapshot["total_rejections"] >= 1
    assert snapshot["total_opens"] == 1


@pytest.mark.parametrize(
    ("threshold", "recovery"),
    [(0, 60.0), (-1, 60.0), (3, 0.0), (3, -1.0)],
)
def test_breaker_rejects_nonsensical_parameters(threshold: int, recovery: float) -> None:
    """不合理的参数必须在**构造时**报错。

    ⚠️ ``failure_threshold=0`` 是个特别隐蔽的写法：若实现用
    ``failures >= threshold`` 判断，阈值 0 会让熔断器**永远处于开路**，
    服务彻底不可用；而若用 ``failures > threshold``，它又变成永不熔断。
    两种都不是调用者想要的，因此应当直接拒绝，而不是挑一种「猜」。
    同理 ``recovery_seconds=0`` 会让冷却期立即结束（熔断形同虚设），
    负数则会让 ``retry_after`` 变成负值。

    Args:
        threshold (`int`): 失败阈值。
        recovery (`float`): 冷却秒数。
    """
    with pytest.raises(ValueError):
        CircuitBreaker(failure_threshold=threshold, recovery_seconds=recovery)


# ==============================================================================
# 三、熔断器 —— 进程级共享（这是设计约束，不是实现细节）
# ==============================================================================
def test_breaker_is_shared_across_callers() -> None:
    """``get_breaker()`` 必须返回**同一个**实例。

    ⚠️ 这不是「缓存一下省点开销」，而是熔断器能否工作的前提。
    若每个会话/每个智能体各持一个实例，那么一个下游挂了之后，
    熔断计数会按会话数摊薄 —— 100 个会话就是 100 个独立的计数器，
    每个都离阈值差得很远，于是**永远不会熔断**。
    症状是「配了熔断器，下游挂掉时服务照样被打爆」。

    用例同时验证 ``settings`` 参数不影响返回结果 ——
    保证「谁先调用，谁决定参数」这个语义是稳定的。
    """
    reset_breaker()
    try:
        first = get_breaker()
        second = get_breaker(_settings())

        assert first is second, "get_breaker() 每次返回了新实例 —— 熔断器将无法生效"
    finally:
        reset_breaker()


def test_reset_breaker_clears_the_singleton() -> None:
    """``reset_breaker()`` 之后拿到的是全新实例。

    它存在是为了让用例之间互不影响；生产代码不该调用它。
    用例把它钉住，是因为「重置没生效」会让**另一个**用例以莫名其妙的方式失败，
    那种跨用例的偶发失败极难定位。
    """
    reset_breaker()
    before = get_breaker()
    reset_breaker()

    assert get_breaker() is not before
    reset_breaker()
