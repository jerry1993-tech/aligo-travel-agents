# -*- coding: utf-8 -*-
"""重排模型的截止时间护栏（``src/llm/bounded.py``）与它的装配入口
（``src/llm/factory.py::build_rerank_model``）的测试。

═══ 这些用例在防什么 ═══

开了重排（``ALIGO__RERANK__ENABLED=true``）之后，一次检索多出一段调用：

    ``RAGMiddleware.on_reasoning`` → ``_rerank_results`` → ``generate_structured_output``

它在**回复链路里**，却没有任何一层给它设截止时间：``ModelTimeoutMiddleware``
只管 ``on_model_call``（agent 自己的模型调用），而框架给重排的
``except Exception``（``agentscope/middleware/_rag.py:439``）
只在**抛异常**时才回退到向量序 —— 卡住不返回的调用永远不抛异常。
换句话说，一个挂住的重排会把用户的回复一起拖住，而配置看起来完全正常。

所以本文件里最重要的那条用例是
:func:`test_timeout_fires_even_though_the_inner_model_swallows_cancellation`
—— 它构造一个**故意吞掉取消**的内层模型。这是本项目里唯一能区分
「真护栏」与「纸面护栏」的用例：

    ``asyncio.wait_for`` 靠取消来中断，再把「取消有没有传播出来」当成超时判据。
    框架的对话模型基类**刻意**把 ``CancelledError`` 转成一个优雅的
    「已打断」响应（``agentscope/model/_base.py:219-224``），
    于是 ``wait_for`` 看到的永远是「正常返回」，超时一次都不触发。
    ``src/llm/middleware.py`` 的 ``ModelTimeoutMiddleware`` 踩过同一个坑，
    本模块用同一套「按钟判断」的写法绕开它。

用一个**吞取消**的替身而不是 ``asyncio.sleep`` 来测，就是因为
``sleep`` 版本的用例对这两种实现**都能通过** —— 它测不出差别。
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from src.config import Settings, load_settings
from src.llm.bounded import (
    BoundedChatModel,
    ChatCallTimeout,
    bound_chat_model,
    unwrap_chat_model,
)
from src.llm.mock import MockChatModel
from src.llm.factory import build_rerank_model


def _settings(**overrides: str) -> Settings:
    """构造一份测试配置，并允许按 ``ALIGO__`` 环境变量的形式覆盖。

    Args:
        **overrides (`str`): 形如 ``ALIGO__RERANK__ENABLED="true"`` 的覆盖项。

    Returns:
        `Settings`: 完整配置对象。
    """
    from tests.conftest import TEST_ENVIRON

    environ = {**TEST_ENVIRON, **overrides}
    return load_settings("test", environ=environ, dotenv=False)


class SwallowingModel(MockChatModel):
    """一个**吞掉取消**的假模型 —— 框架对话模型基类的行为复刻。

    ``generate_structured_output`` 在收到 ``CancelledError`` 之后不往外抛，
    而是稍等一下再**正常返回**。这正是
    ``agentscope/model/_base.py:219-224``
    对 ``CancelledError`` 的处理方式（转成一次「已打断」的响应），
    也是 ``asyncio.wait_for`` 在本项目里失效的原因。

    Attributes:
        entered (`asyncio.Event`): 内层已经进入等待。用例靠它确定
            「调用确实开始了」，而不是靠 ``sleep`` 猜时间。
        started (`bool`): 是否被调用过（没有它的话，
            「守卫根本没被触发」与「守卫触发后立刻返回」这两种情况
            在断言里长得一样）。
        cancelled (`bool`): 内层是否收到过取消。
    """

    def __init__(self, **kwargs: Any) -> None:
        """初始化替身。

        Args:
            **kwargs: 透传给 :class:`~src.llm.mock.MockChatModel`。
        """
        super().__init__(**kwargs)
        self.entered = asyncio.Event()
        self.started = False
        self.cancelled = False

    async def generate_structured_output(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """记录调用、然后**吞掉取消**并正常返回。

        Args:
            *args: 忽略。
            **kwargs: 忽略。

        Returns:
            `str`: 一个可辨识的哨兵值。
        """
        self.started = True
        self.entered.set()
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            # ⚠️ 刻意**不**重新抛出 —— 这就是被复刻的那个行为。
            self.cancelled = True
        return "swallowed"


# ==============================================================================
# 一、截止时间真的会触发（本文件的核心）
# ==============================================================================
def test_timeout_fires_even_though_the_inner_model_swallows_cancellation() -> None:
    """内层吞掉取消时，超时判定**仍然**成立。

    这是「按钟判断」相对 ``asyncio.wait_for`` 的唯一区别，
    也是这个模块存在的全部理由。若把它改成 ``wait_for``，
    本用例会**挂 30 秒后拿到 'swallowed'**，而不是抛 ``ChatCallTimeout``。
    """
    inner = SwallowingModel(credential=None, model="fake-rerank")
    bounded = BoundedChatModel(inner, timeout=0.05)

    async def scenario() -> None:
        with pytest.raises(ChatCallTimeout):
            await bounded.generate_structured_output(
                messages=[],
                structured_model=dict,
            )
        # 取消已经投递给内层了（不是「任务还挂着」）。
        await asyncio.sleep(0)
        assert inner.cancelled is True

    asyncio.run(scenario())

    assert inner.started is True


def test_a_hung_call_does_not_block_the_caller_beyond_the_budget() -> None:
    """卡住 30s 的内层，调用方在**预算内**就拿到失败。

    ⚠️ 断言的是**墙钟**，不是「抛了异常」：一个「先超时抛异常、
    但把内层任务留在后台」的实现也能通过上一条用例，
    而它并不能阻止请求堆积。
    """
    inner = SwallowingModel(credential=None, model="fake-rerank")
    bounded = BoundedChatModel(inner, timeout=0.2)

    async def scenario() -> float:
        loop = asyncio.get_running_loop()
        started = loop.time()
        with pytest.raises(ChatCallTimeout):
            await bounded.generate_structured_output()
        return loop.time() - started

    elapsed = asyncio.run(scenario())

    # 留 5 倍余量：CI 上时钟抖动是常事，这里要证明的是
    # 「不是等了 30 秒」，而不是「精确地在 0.2 秒返回」。
    assert elapsed < 1.0, f"超时后 {elapsed:.2f}s 才返回，护栏没有真正生效"


# ==============================================================================
# 二、正常路径与错误透传（反例：护栏不能把一切都变成超时）
# ==============================================================================
def test_a_result_within_the_budget_passes_through_unchanged() -> None:
    """预算内返回时，结果**原样**透出，不被包装。

    反例用例。没有它的话，一个「永远抛 ChatCallTimeout」的实现
    能通过上面两条。
    """

    class Fast:
        """立即返回固定值的假模型。"""

        model = "fast"

        async def generate_structured_output(self, *a: Any, **k: Any) -> str:
            """返回哨兵值。"""
            return "ok"

    async def scenario() -> str:
        return await BoundedChatModel(Fast(), timeout=5).generate_structured_output()

    assert asyncio.run(scenario()) == "ok"


def test_inner_errors_are_not_rewritten_into_timeouts() -> None:
    """内层的真实错误必须**原样**上抛，不能被包成超时。

    ⚠️ 这是最贵的一种信息损失：把 401 / 模型下线 / 维度不符这类错误
    报成「超过 N 秒未返回」，排查方向会被整个带偏 ——
    而重排的失败还是**静默**的（框架退回向量序），
    所以日志里那句话往往是唯一的线索。
    """

    class Broken:
        """抛出一类可辨识错误的假模型。"""

        model = "broken"

        async def generate_structured_output(self, *a: Any, **k: Any) -> Any:
            """抛出 ValueError。"""
            raise ValueError("模型名不存在")

    async def scenario() -> None:
        with pytest.raises(ValueError, match="模型名不存在"):
            await BoundedChatModel(Broken(), timeout=5).generate_structured_output()

    asyncio.run(scenario())


# ==============================================================================
# 三、身份与包装的幂等性
# ==============================================================================
def test_model_name_is_readable_through_the_wrapper() -> None:
    """``.model`` 在包装层上直接可读。

    框架的重排日志（``agentscope/middleware/_rag.py:484``）
    会读它；读不到就会以 ``AttributeError`` 的形式在检索路径上炸出来。
    用例同时钉住 :class:`ChatCallTimeout` 的 ``model`` 字段 ——
    它是排障时唯一能回答「哪个模型卡住了」的字段。
    """
    inner = MockChatModel(
        credential=None,
        model="qwen-plus",
        stream=False,
    )
    bounded = BoundedChatModel(inner, timeout=1)

    assert bounded.model == "qwen-plus"
    assert bounded.timeout == 1
    assert bounded.inner is inner
    assert "qwen-plus" in repr(bounded)


def test_unknown_attributes_fall_through_to_the_inner_model() -> None:
    """未接管的属性透传到内层。

    ⚠️ 少了它，任何一处 ``bounded.某个内层属性`` 都会变成
    ``AttributeError``，而报错点离包装层很远。
    """
    inner = MockChatModel(credential=None, model="qwen-plus", stream=False)
    bounded = BoundedChatModel(inner, timeout=1)

    assert bounded.stream == inner.stream
    assert bounded.parameters is inner.parameters


def test_bounding_twice_is_a_no_op() -> None:
    """重复套护栏不会叠成两层。

    ⚠️ 叠两层不会算错，但会让「超时是哪一层报的」失去唯一答案，
    而且实际超时值会变成两层的较小者 —— 排查时看到 60s 的配置
    却 30s 就超时，方向会被引向错误的地方。
    """
    inner = MockChatModel(credential=None, model="qwen-plus", stream=False)

    once = bound_chat_model(inner, timeout=30)
    twice = bound_chat_model(once, timeout=5)

    assert twice is once
    assert twice.timeout == 30


def test_unwrap_strips_every_layer() -> None:
    """剥包装要一直剥到底，不能停在中途那层。"""
    inner = MockChatModel(credential=None, model="qwen-plus", stream=False)
    once = bound_chat_model(inner, timeout=30)

    assert unwrap_chat_model(once) is inner
    assert unwrap_chat_model(inner) is inner


@pytest.mark.parametrize("timeout", [0, -1, 0.0])
def test_non_positive_timeout_is_rejected_at_construction(timeout: float) -> None:
    """超时必须 > 0。

    ⚠️ 取 0 会让**每次**重排都超时，等价于静默关闭重排能力 ——
    这种配置必须当场炸，而不是变成一个「开了重排但排序从没变过」的谜题。
    """
    inner = MockChatModel(credential=None, model="qwen-plus", stream=False)

    with pytest.raises(ValueError, match="必须 > 0"):
        BoundedChatModel(inner, timeout=timeout)


# ==============================================================================
# 四、装配入口：build_rerank_model
# ==============================================================================
def test_rerank_model_is_none_when_disabled() -> None:
    """``rerank.enabled=false`` ⇒ ``None``。

    ⚠️ 返回 ``None`` 而不是「一个永不生效的模型」：``RAGMiddleware``
    只在 ``rerank_model is None`` 时才不扩大召回（
    ``agentscope/middleware/_rag.py:405``），
    塞一个假模型进去会让每次检索多做一轮无用的召回。
    """
    settings = _settings(**{"ALIGO__RERANK__ENABLED": "false"})

    assert build_rerank_model(settings) is None


def test_a_model_name_without_the_switch_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """配了模型名却没开开关 ⇒ 仍然不生效，但**必须**留下一条告警。

    ⚠️ 这是本项目真实踩过的坑：``.env`` 里写了
    ``ALIGO__RERANK__MODEL=...`` 却没有 ``ENABLED=true``，
    而当时的 schema 连 ``rerank`` 段都没有 —— 容器启动即崩。
    现在键合法了，但「配了名字没开开关」仍然是一个
    只能靠读配置回答的问题，所以要在日志里明说。
    """
    settings = _settings(
        **{
            "ALIGO__RERANK__ENABLED": "false",
            "ALIGO__RERANK__MODEL": "qwen-plus",
        },
    )

    with caplog.at_level("WARNING", logger="src.llm.factory"):
        result = build_rerank_model(settings)

    assert result is None
    assert any("rerank.enabled" in record.getMessage() for record in caplog.records)


def test_enabled_with_a_blank_name_reuses_the_main_model_instance() -> None:
    """``enabled=true`` + ``model`` 留空 ⇒ 复用传进来的主对话模型**实例**。

    ⚠️ 复用实例而不是另建一个，是为了让
    ``build_agent_wiring(model=...)`` 注入的测试替身也被沿用 ——
    否则测试环境会偷偷去建一个真实模型（并尝试连网）。
    """
    settings = _settings(
        **{
            "ALIGO__RERANK__ENABLED": "true",
            "ALIGO__RERANK__MODEL": "",
        },
    )
    main = MockChatModel(credential=None, model="qwen-plus", stream=True)

    rerank = build_rerank_model(settings, reuse=main)

    assert isinstance(rerank, BoundedChatModel)
    assert rerank.inner is main
    # 默认预算 = 主对话模型的单次调用预算（两套入口口径一致）。
    assert rerank.timeout == settings.llm.timeout_seconds


def test_enabled_with_a_name_builds_a_dedicated_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``enabled=true`` + 配了模型名 ⇒ 用**那个**名字建一个独立实例。

    ⚠️ 用 monkeypatch 盯住 ``build_chat_model`` 的调用参数，
    而不是断言返回对象的类名：这里要证明的是
    「重排走的是同一个构造入口、并且把模型名透传下去了」。
    断言类名的话，一个「忽略配置、直接复用主模型」的实现也能通过。
    """
    from src.llm import factory as factory_module

    settings = _settings(
        **{
            "ALIGO__RERANK__ENABLED": "true",
            "ALIGO__RERANK__MODEL": "qwen-turbo",
        },
    )
    sentinel = MockChatModel(credential=None, model="qwen-turbo", stream=False)
    calls: list[dict[str, Any]] = []

    def fake_build(settings_arg: Any = None, **kwargs: Any) -> Any:
        """记录调用参数并返回哨兵模型。"""
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(factory_module, "build_chat_model", fake_build)

    rerank = build_rerank_model(settings)

    assert calls == [{"stream": False, "model_name": "qwen-turbo"}]
    assert isinstance(rerank, BoundedChatModel)
    assert rerank.inner is sentinel


def test_a_garbage_model_name_is_still_wrapped_and_bounded() -> None:
    """专用重排模型名（框架用不了的那种）**不**在装配期被拦下。

    ⚠️ 这一条记录的是一个**刻意的取舍**，不是遗漏：
    ``settings.rerank.model`` 填 ``qwen3-rerank`` 这类专用重排模型名时，
    本框架会调用失败（它要的是 ``ChatModelBase``，见模块文档），
    但失败是**运行期**的，而且被框架的 ``except Exception``
    静默接住、退回向量序。

    我们不在装配期做「这个名字像不像对话模型」的启发式校验：
    那种校验只能靠名字字符串猜，而模型名的命名规则是上游随时会变的
    —— 猜错的代价是**拦住一个本来能用的配置**，比放行一个坏配置更糟
    （前者是可用性事故，后者只是一个排序没生效）。
    这里只保证：无论名字是什么，护栏都在。
    """
    settings = _settings(
        **{
            "ALIGO__RERANK__ENABLED": "true",
            "ALIGO__RERANK__MODEL": "qwen3-rerank",
        },
    )

    rerank = build_rerank_model(settings)

    assert isinstance(rerank, BoundedChatModel)
    assert rerank.timeout > 0
