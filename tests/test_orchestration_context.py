# -*- coding: utf-8 -*-
"""动态 Prompt 执行层（``ContextInjectionMiddleware``）的测试。

═══ 本文件守的是什么 ═══

动态 Prompt 的价值在于「模型每次推理都看到与当前进度相符的指令」。它有
三种典型的坏法，本文件各有一组用例：

1. **挂上去了但没生效** —— ``on_system_prompt`` 的返回值**替换**整个
   system prompt，写错就把框架自己的 base prompt 丢了；
2. **丢掉框架的前序内容** —— 必须**基于** ``current_prompt`` 拼装；
3. **阶段读不到** —— 写进去的是字符串、读的人按枚举判，分支永远为假，
   于是阶段静默退化成「按意图猜」。这条**已经真实发生过一次**。

⚠️ 第 3 条是本文件最要紧的部分，见 :func:`test_resolver_sees_the_stage_written_by_lane`。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.domain import Intent, TravelRequest, TripStage
from src.orchestration.context import (
    PROMPT_NUMBERS_KEY,
    ContextInjectionMiddleware,
    PromptContext,
    default_resolver,
    record_prompt_numbers,
    recorded_prompt_numbers,
)
from src.orchestration.lane import (
    ROUTE_DECISION_KEY,
    LaneRouterMiddleware,
    recorded_stage,
)
from src.orchestration.prompt import MARKER_BEGIN, MARKER_END

#: 框架在链式调用里传进来的 prompt —— 模拟「base prompt + 别的中间件的产出」。
BASE = "你是 AliGo 差旅助手。\n\n当前时间由框架注入。"


# ---------------------------------------------------------------------------
# 测试替身
# ---------------------------------------------------------------------------
class _Decision:
    """``RouteDecision`` 的最小替身，供 ``_remember`` 使用。"""

    lane = SimpleNamespace(value="FAST")
    intent = SimpleNamespace(value="PLAN_TRIP")
    matched_rule = "plan_trip"
    target_agents = ["main_plan"]
    reason = "识别为规划行程指令"


def make_agent(*, reply_id: str | None = "reply-1", middle: dict[str, Any] | None = None) -> Any:
    """造一个带 ``state`` 的极小 agent 替身。

    Args:
        reply_id (`str | None`): 回复标识；``None`` 表示拿不到。
        middle (`dict | None`): ``middle_context`` 的初始内容。

    Returns:
        `Any`: agent 替身。
    """
    return SimpleNamespace(
        name="main_plan",
        state=SimpleNamespace(reply_id=reply_id, middle_context=dict(middle or {})),
    )


def write_stage(agent: Any, stage: TripStage) -> None:
    """按 ``lane`` 的方式把阶段写进 ``middle_context``。

    ⚠️ 走 :meth:`LaneRouterMiddleware._remember` 而不是手写字典 —— 手写的话
    就绕过了「写进去的到底是什么形态」这个问题，而本文件一半的用例
    正是为了守那个形态。
    """
    LaneRouterMiddleware()._remember(agent, _Decision(), stage=stage)  # noqa: SLF001


# ---------------------------------------------------------------------------
# 一、钩子的基本契约
# ---------------------------------------------------------------------------
def test_hook_signature_has_no_next_handler() -> None:
    """⚠️ ``on_system_prompt`` 是**三个**参数，**不是**洋葱式钩子。

    它是 ``MiddlewareBase`` 里唯一的「转换器」式钩子：没有 ``next_handler``，
    返回的就是最终串。写成洋葱式（多一个 ``next_handler`` 参数）会让框架
    按参数个数匹配不上，钩子被**静默忽略** —— 动态 Prompt 一点效果都没有，
    而日志干干净净。
    """
    import inspect

    params = list(inspect.signature(ContextInjectionMiddleware.on_system_prompt).parameters)
    assert params == ["self", "agent", "current_prompt"], f"钩子签名是 {params}"
    assert "next_handler" not in params


@pytest.mark.asyncio
async def test_output_is_based_on_the_incoming_prompt() -> None:
    """⚠️ 返回值必须**包含** ``current_prompt`` 的内容。

    消费侧是 ``result = await mw.on_system_prompt(self, result)`` —— 返回值
    **替换**整个 system prompt。直接返回自己那一段，会把框架的 base prompt、
    skills、offloader 全部丢掉：模型失去身份设定，行为立刻退化。
    """
    mw = ContextInjectionMiddleware()
    out = await mw.on_system_prompt(make_agent(), BASE)

    assert BASE in out, "原始 prompt 被丢掉了"
    assert MARKER_BEGIN in out and MARKER_END in out


@pytest.mark.asyncio
async def test_disabled_still_strips_our_own_sections() -> None:
    """⚠️ ``enabled=False`` 是「只剥离、不追加」，不是「什么都不做」。

    排障时关掉动态 Prompt，若旧段落还留在串里，会得到「关了但还是有」
    的假象 —— 那正是排障时最不想要的结果。
    """
    mw = ContextInjectionMiddleware()
    dirty = await mw.on_system_prompt(make_agent(), BASE)
    assert MARKER_BEGIN in dirty

    mw_off = ContextInjectionMiddleware(enabled=False)
    clean = await mw_off.on_system_prompt(make_agent(), dirty)

    assert MARKER_BEGIN not in clean
    assert BASE in clean


@pytest.mark.asyncio
async def test_applying_twice_is_idempotent() -> None:
    """⚠️ 同一个 prompt 处理两次，结果与处理一次相同。

    ``on_system_prompt`` 每轮推理都跑。若处理不幂等，段落会随轮次不断
    堆叠 —— 一轮回复跑三轮，模型就会看到三段几乎一样但阶段不同的指令，
    互相矛盾。
    """
    mw = ContextInjectionMiddleware()
    once = await mw.on_system_prompt(make_agent(), BASE)
    twice = await mw.on_system_prompt(make_agent(), once)

    assert once == twice


# ---------------------------------------------------------------------------
# 二、绝不抛异常
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_a_broken_resolver_does_not_fail_the_reply() -> None:
    """⚠️ 解析器抛异常时返回**原始** prompt，不向上抛。

    本钩子跑在 agent 的主链路上；抛出去会让整轮回复失败。而「prompt 少了
    一段」的代价与之完全不成比例。

    ⚠️ 返回的是**原始** ``current_prompt`` 而不是剥离过的：此时连「哪些是
    我们加的」都不确定，乱剥可能剥掉别人的内容。
    """
    def boom(_agent: Any) -> PromptContext:
        raise RuntimeError("解析器炸了")

    mw = ContextInjectionMiddleware(resolver=boom)
    out = await mw.on_system_prompt(make_agent(), BASE)

    assert out == BASE


@pytest.mark.asyncio
async def test_a_resolver_returning_garbage_does_not_fail_the_reply() -> None:
    """解析器返回 ``None``（而不是 ``PromptContext``）时也不崩。

    ⚠️ 这类「返回了但返回错了」比「抛异常」常见得多：一个查库失败后
    ``return None`` 的实现、一个忘了 ``await`` 的实现，都会走到这里。
    """
    mw = ContextInjectionMiddleware(resolver=lambda _a: None)  # type: ignore[arg-type,return-value]
    out = await mw.on_system_prompt(make_agent(), BASE)

    assert out == BASE


# ---------------------------------------------------------------------------
# 三、按回复缓存
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_resolver_is_called_once_per_reply() -> None:
    """⚠️ 同一 ``reply_id`` 内只解析一次。

    本钩子**每轮推理**都调用（一次回复可能三轮），而解析可能要查库。
    不缓存的话，一次回复会打出三次相同的查询。
    """
    calls: list[str] = []

    def counting(_agent: Any) -> PromptContext:
        calls.append("call")
        return PromptContext(stage=TripStage.COLLECTING)

    mw = ContextInjectionMiddleware(resolver=counting)
    agent = make_agent(reply_id="r-1")

    for _ in range(3):
        await mw.on_system_prompt(agent, BASE)

    assert len(calls) == 1, f"解析器被调了 {len(calls)} 次，应当只调 1 次"


@pytest.mark.asyncio
async def test_a_new_reply_id_recomputes() -> None:
    """⚠️ ``reply_id`` 变了必须重算 —— 否则上一轮的阶段会套到这一轮。

    这条与上一条互为表里：只测「同一回复只算一次」会被「永远只算一次」
    的实现骗过，而那正是最坏的实现。
    """
    calls: list[str] = []

    def counting(_agent: Any) -> PromptContext:
        calls.append("call")
        return PromptContext()

    mw = ContextInjectionMiddleware(resolver=counting)
    agent = make_agent(reply_id="r-1")
    await mw.on_system_prompt(agent, BASE)

    agent.state.reply_id = "r-2"
    await mw.on_system_prompt(agent, BASE)

    assert len(calls) == 2


@pytest.mark.asyncio
async def test_without_a_reply_id_it_recomputes_every_time() -> None:
    """⚠️ 拿不到 ``reply_id`` 时**不缓存**，每次重算。

    宁可多算几次，也不要在拿不到标识的时候赌「应该是同一次回复」——
    赌错的后果是把上一轮的阶段套到这一轮，而用户会看到方案被反复要求重填。
    """
    calls: list[str] = []

    def counting(_agent: Any) -> PromptContext:
        calls.append("call")
        return PromptContext()

    mw = ContextInjectionMiddleware(resolver=counting)
    agent = make_agent(reply_id=None)

    for _ in range(3):
        await mw.on_system_prompt(agent, BASE)

    assert len(calls) == 3


@pytest.mark.asyncio
async def test_invalidate_forces_a_recompute() -> None:
    """``invalidate()`` 清掉缓存，下一次调用重新解析。

    用途是 HITL 恢复路径：用户在确认弹窗期间可能改了要素，此时阶段真的变了。
    """
    stages = [TripStage.COLLECTING, TripStage.CONFIRMING]
    seen: list[TripStage] = []

    def resolver(_agent: Any) -> PromptContext:
        return PromptContext(stage=stages[min(len(seen), len(stages) - 1)])

    def counting(a: Any) -> PromptContext:
        ctx = resolver(a)
        seen.append(ctx.stage)
        return ctx

    mw = ContextInjectionMiddleware(resolver=counting)
    agent = make_agent(reply_id="r-1")

    first = await mw.on_system_prompt(agent, BASE)
    mw.invalidate()
    second = await mw.on_system_prompt(agent, BASE)

    assert len(seen) == 2
    assert first != second, "清缓存后应当解析出不同的阶段"


@pytest.mark.asyncio
async def test_the_cache_holds_only_the_latest_reply() -> None:
    """⚠️ 缓存是**单槽**的，只留最近一次回复的解析结果。

    解析结果里可能含用户画像，那是**不该长期驻留**的个人数据。单槽缓存
    顺带把这份数据的最长存活时间压到「一次回复」。
    """
    mw = ContextInjectionMiddleware(resolver=lambda _a: PromptContext(stage=TripStage.COLLECTING))
    agent = make_agent(reply_id="r-1")
    await mw.on_system_prompt(agent, BASE)

    agent.state.reply_id = "r-2"
    await mw.on_system_prompt(agent, BASE)

    # 只有 r-2 被缓存；r-1 的结果已经被覆盖掉了。
    assert mw._cached_reply_id == "r-2"  # noqa: SLF001


# ---------------------------------------------------------------------------
# 四、默认解析器：阶段从哪来
# ---------------------------------------------------------------------------
def test_resolver_sees_the_stage_written_by_lane() -> None:
    """★★ **回归测试**：``lane`` 写进去的阶段，默认解析器必须读得到。

    ⚠️ 这是一条真实的回归。``lane._remember`` 写入的是 ``stage.value``
    （**字符串**，因为 ``middle_context`` 可能被序列化），而
    ``default_resolver`` 原先自己写了一个
    ``isinstance(recorded.get("stage"), TripStage)`` 判断 —— 在字符串写入
    之后**永远为假**。

    后果：显式阶段分支成了**死代码**，阶段静默退化成「按意图猜」。
    没有异常、没有日志、没有任何可见的坏味道 —— 用户只是偶尔发现
    「我已经填到确认阶段了，它怎么又问我要出发地」。

    ⚠️ 这条用例是**跨模块**的：写入方在 ``lane``，读取方在 ``context``。
    两边各自单测都通过，只有把它们接起来才暴露问题。
    """
    agent = make_agent()
    write_stage(agent, TripStage.CONFIRMING)

    # 先确认写入的确实是字符串 —— 否则这条用例会在某天悄悄失去意义。
    raw = agent.state.middle_context[ROUTE_DECISION_KEY]["stage"]
    assert isinstance(raw, str), f"写入的不是字符串而是 {type(raw).__name__}，本用例的前提变了"

    assert default_resolver(agent).stage is TripStage.CONFIRMING


def test_explicit_stage_wins_over_the_intent_hint() -> None:
    """⚠️ 显式阶段**优先于**按意图猜的阶段。

    顺序反了会让「用户已经填到 CONFIRMING，这一轮又说了句『规划行程』」
    被打回 COLLECTING —— 表现为方案被反复要求重填。
    """
    agent = make_agent()
    # 意图是 PLAN_TRIP（会推出 COLLECTING），但显式阶段是 CONFIRMING。
    write_stage(agent, TripStage.CONFIRMING)
    agent.state.middle_context[ROUTE_DECISION_KEY]["intent"] = Intent.PLAN_TRIP.value

    assert default_resolver(agent).stage is TripStage.CONFIRMING


def test_intent_hint_kicks_in_when_no_explicit_stage() -> None:
    """⚠️ 没有显式阶段时，按意图推一个 —— 否则快车道短路后阶段永远停在 IDLE。

    快车道下 agent 的状态里只有一条路由决策。若不借此把阶段推进一步，
    动态 Prompt 会一直停在 IDLE，用户点了「规划行程」却得到一句
    「请问有什么可以帮您」。
    """
    for intent in (Intent.PLAN_TRIP, Intent.MODIFY_TRIP):
        agent = make_agent(middle={ROUTE_DECISION_KEY: {"intent": intent.value}})
        assert default_resolver(agent).stage is TripStage.COLLECTING, f"{intent} 没有推进到 COLLECTING"


@pytest.mark.parametrize(
    "intent",
    [Intent.QUERY_ORDER, Intent.QUERY_POLICY, Intent.CHITCHAT, Intent.APPLY_APPROVAL, Intent.CANCEL],
)
def test_non_collecting_intents_stay_idle(intent: Intent) -> None:
    """⚠️ 其余意图**不得**映射到 COLLECTING。

    查订单、问政策、闲聊都不改变出差收集阶段。给它们硬编一个阶段，会把
    用户已有的行程收集进度冲掉 —— 用户查个订单回来，发现刚填的要素没了。
    """
    agent = make_agent(middle={ROUTE_DECISION_KEY: {"intent": intent.value}})
    assert default_resolver(agent).stage is TripStage.IDLE


def test_resolver_survives_an_unknown_intent() -> None:
    """⚠️ 认不出的意图值降级为 IDLE，不抛异常。

    路由记录可能来自旧版本（枚举改名），而解析器跑在主链路上。
    """
    agent = make_agent(middle={ROUTE_DECISION_KEY: {"intent": "NOT_AN_INTENT"}})
    assert default_resolver(agent).stage is TripStage.IDLE


def test_resolver_survives_a_missing_state() -> None:
    """拿不到 ``state`` / ``middle_context`` 时返回全默认值，不抛异常。"""
    assert default_resolver(SimpleNamespace()).stage is TripStage.IDLE
    assert default_resolver(SimpleNamespace(state=None)).stage is TripStage.IDLE
    assert default_resolver(SimpleNamespace(state=SimpleNamespace(middle_context=None))).stage is TripStage.IDLE


def test_resolver_never_reports_a_request() -> None:
    """⚠️ 默认解析器的 ``request`` **恒为 None**。

    它被明确要求不做 I/O，而拿到持久化的 ``TravelRequest`` 需要查库。
    这里不假装能拿到，而是留空 —— 假装的话就得编一个空的 ``TravelRequest``，
    模型会看到「已知要素」一节是空的，反而更困惑。
    """
    agent = make_agent()
    write_stage(agent, TripStage.COLLECTING)

    assert default_resolver(agent).request is None


# ---------------------------------------------------------------------------
# 五、共享读取入口（防止两处读者再次漂移）
# ---------------------------------------------------------------------------
def test_recorded_stage_is_the_single_reader() -> None:
    """⚠️ ``lane`` 与 ``context`` 读阶段走的是**同一个**函数。

    两处各写一份读取逻辑，就会漂移 —— 而这件事**已经发生过一次**
    （见 :func:`test_resolver_sees_the_stage_written_by_lane`）。
    这条断言从结构上守住「只有一个读取入口」。
    """
    agent = make_agent()
    write_stage(agent, TripStage.DONE)

    assert recorded_stage(agent) is TripStage.DONE
    assert LaneRouterMiddleware()._stage_of(agent) is TripStage.DONE  # noqa: SLF001
    assert default_resolver(agent).stage is TripStage.DONE


def test_recorded_stage_degrades_on_a_foreign_value() -> None:
    """⚠️ 读不出阶段时返回 ``None``，**不抛异常**。

    旧会话里遗留的阶段值、或将来枚举改名，都会让 ``TripStage(raw)`` 抛
    ValueError。而两个调用方都在主链路上。
    """
    for raw in ("LEGACY_STAGE", 42, ["COLLECTING"]):
        agent = make_agent(middle={ROUTE_DECISION_KEY: {"stage": raw}})
        assert recorded_stage(agent) is None, f"{raw!r} 应当降级为 None"


def test_recorded_stage_survives_a_non_dict_record() -> None:
    """⚠️ 路由记录本身不是 dict 时也不崩。

    会话状态会被序列化；一个被写坏的记录不该让整轮回复挂掉。
    """
    for middle in ({"aligo:route_decision": None}, {"aligo:route_decision": "字符串"}, {}):
        assert recorded_stage(make_agent(middle=middle)) is None


# ---------------------------------------------------------------------------
# 五、登记「本轮注入了哪些数字」（缺陷 P1 的写侧）
# ---------------------------------------------------------------------------
# ⚠️ 这一组用例守的是回复守卫「事实接地闸门」的**第四个合法来源**：
# 我们自己在动态 Prompt 里注入给模型的、承载用户数据的数字。
# 读侧在 tests/test_orchestration_reply_guard.py，写侧在这里 ——
# 两边合起来才是一条完整的接线。
def _request_with_budget() -> TravelRequest:
    """造一个「用户说过 3 天、预算 15000」的出差要素。

    ⚠️ 字段刻意只填这几项：登记的数字集合必须能被逐项列出来，
    多一个字段就会多一组数字，断言就变成「包含」而不再是「相等」。
    """
    return TravelRequest(destination="北京", days=3, budget=15000.0)


@pytest.mark.asyncio
async def test_records_the_numbers_of_the_injected_user_data() -> None:
    """动态 Prompt 里的用户数字必须被登记（供守卫做溯源）。

    ⚠️ 断言用**相等**而不是包含：登记集合就是「守卫会豁免的数字」，
    多出来一个都是在悄悄放宽闸门。这条把「只登记用户数据」钉死。
    """
    mw = ContextInjectionMiddleware(
        resolver=lambda _a: PromptContext(request=_request_with_budget()),
    )
    agent = make_agent(reply_id="r-1")

    await mw.on_system_prompt(agent, BASE)

    assert recorded_prompt_numbers(agent, "r-1") == ["15000", "3"], (
        "登记的数字与用户提供的要素不一致 —— 多了会放宽闸门，"
        "少了会让正确的答复被拦（缺陷 P1）"
    )


@pytest.mark.asyncio
async def test_recording_excludes_the_static_prompt_wording() -> None:
    """★ 阶段/追问指令里的数字（「最多 2 项」）**不得**进豁免集合。

    ⚠️ 这条钉的是「摘的是哪一段」。把整个动态段落都登记进去是最省事的
    写法，它会让 ``MAX_MISSING_PROMPTS``（=2）变成合法来源 ——
    模型写「酒店差标上限 2 元」就再也拦不住了。虽然荒唐，但闸门的
    豁免集合只该包含**用户说过的话**，这是它的语义。
    """
    request = TravelRequest(destination="北京", budget=15000.0)  # 必填项缺失 → 会列「最多 2 项」
    mw = ContextInjectionMiddleware(
        resolver=lambda _a: PromptContext(stage=TripStage.COLLECTING, request=request),
    )
    agent = make_agent(reply_id="r-1")

    prompt = await mw.on_system_prompt(agent, BASE)

    assert "最多 2 项" in prompt, "前置条件不成立：静态指令里的 2 没进 prompt"
    recorded = recorded_prompt_numbers(agent, "r-1")
    assert "2" not in recorded, f"静态文案里的数字被登记成了豁免：{recorded}"


@pytest.mark.asyncio
async def test_disabled_records_nothing() -> None:
    """``enabled=False`` 时**不登记**：那种配置下动态段落根本没进 prompt。

    ⚠️ 登记等于凭空造一条豁免 —— 守卫会放行一个 prompt 里从没出现过的
    数字。不登记只会让它更保守，方向是安全的。
    """
    mw = ContextInjectionMiddleware(
        enabled=False,
        resolver=lambda _a: PromptContext(request=_request_with_budget()),
    )
    agent = make_agent(reply_id="r-1")

    await mw.on_system_prompt(agent, BASE)

    assert recorded_prompt_numbers(agent, "r-1") == []
    assert PROMPT_NUMBERS_KEY not in agent.state.middle_context


@pytest.mark.asyncio
async def test_a_failed_build_records_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ 拼装抛异常时**不能**留下登记 —— 否则守卫豁免一份没进 prompt 的数字。

    ⚠️ 这条守的是登记步骤的**位置**：它必须在 ``build_system_prompt``
    **之后**。写成「先登记、再拼装」，一旦拼装抛异常走了 ``except``
    （原样返回、什么都没拼进去），登记就成了一句空头支票 ——
    而这种错序在正常路径上完全看不出来。
    """
    def boom(*_args: Any, **_kwargs: Any) -> str:
        raise RuntimeError("拼装炸了")

    monkeypatch.setattr("src.orchestration.context.build_system_prompt", boom)
    mw = ContextInjectionMiddleware(
        resolver=lambda _a: PromptContext(request=_request_with_budget()),
    )
    agent = make_agent(reply_id="r-1")

    out = await mw.on_system_prompt(agent, BASE)

    assert out == BASE, "拼装失败时必须原样返回 incoming prompt"
    assert recorded_prompt_numbers(agent, "r-1") == []


@pytest.mark.asyncio
async def test_recorded_numbers_require_a_matching_reply_id() -> None:
    """★★ 登记按 ``reply_id`` 校验新鲜度 —— 旧记录一律作废。

    ⚠️ ``middle_context`` 是**随会话持久化**的，上一条回复登记的可能是
    完全无关的数字。拿旧记录当豁免，等于给闸门开一个「随口报数」的口子，
    而且只在「上一条回复恰好也走了动态 Prompt」时出现 —— 最难查的那类
    间歇性漏网。判据往严的一侧倒（对不上就当没有）。
    """
    mw = ContextInjectionMiddleware(
        resolver=lambda _a: PromptContext(request=_request_with_budget()),
    )
    agent = make_agent(reply_id="r-1")
    await mw.on_system_prompt(agent, BASE)

    assert recorded_prompt_numbers(agent, "r-1") == ["15000", "3"]
    assert recorded_prompt_numbers(agent, "r-2") == [], "旧登记被当成了本轮豁免"
    assert recorded_prompt_numbers(agent, "") == [], "空 reply_id 不该命中任何登记"


def test_reading_survives_a_hostile_state() -> None:
    """状态缺失、字段被写坏、``middle_context`` 不是 dict —— 一律返回空表。

    ⚠️ 读取跑在守卫的结算路径上（agent 的主链路），抛异常会让整轮回复
    失败；而降级为「没有豁免」只会让它更保守。两个方向的代价不对称。
    """
    assert recorded_prompt_numbers(SimpleNamespace(), "r-1") == []
    assert recorded_prompt_numbers(
        SimpleNamespace(state=SimpleNamespace(middle_context=None)), "r-1"
    ) == []
    assert recorded_prompt_numbers(
        SimpleNamespace(state=SimpleNamespace(middle_context={PROMPT_NUMBERS_KEY: "坏了"})),
        "r-1",
    ) == []
    assert recorded_prompt_numbers(
        SimpleNamespace(
            state=SimpleNamespace(
                middle_context={PROMPT_NUMBERS_KEY: {"reply_id": "r-1", "numbers": 42}}
            )
        ),
        "r-1",
    ) == []


def test_writing_survives_a_hostile_state() -> None:
    """登记失败不抛异常（它跑在 ``on_system_prompt`` 的主链路上）。"""
    record_prompt_numbers(SimpleNamespace(), "r-1", ["15000"])  # 不该抛
    record_prompt_numbers(
        SimpleNamespace(state=SimpleNamespace(middle_context=None)), "r-1", ["15000"]
    )  # 不该抛，也不该新建字段
