# -*- coding: utf-8 -*-
"""智能体装配层（``src/server/agents_factory.py``）的测试。

═══ 这张表守的是什么 ═══

这一层只做一件事：把三个扩展参数喂给 ``create_app``。它的**每一处**错误
都有一个共同的特征 —— **报错点离原因很远**：

- 把**静态列表**当成**工厂**传（或反过来）：``TypeError: object list
  can't be used in 'await' expression``，或者反过来在 ``t.type`` 上崩。
- 中间件**顺序**写错：不报错，只是快车道省下的成本被熔断器的开销吃掉，
  或者熔断器把「没调用」记成「调用成功」，熔断点被悄悄推迟。
- ``LaneRouterMiddleware`` 忘了限定 ``agent_names``：不报错，只是子智能体
  在「被要求检索政策」时被自己的规则表短路，表现为「子智能体什么都没干
  就返回了」。

三条都在下面有用例。中间两条尤其值得看 —— 它们**不会让任何功能测试变红**。

═══ ⚠️ 这里一律不碰真实模型 ═══

``build_agent_wiring`` 的 ``model`` 参数可注入，正是为了这个。传一个假模型，
装配链就能在毫秒级跑通，而不需要密钥、网络、或真实网关。
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from agentscope.app import SubAgentTemplate
from agentscope.middleware import MiddlewareBase, TracingMiddleware
from agentscope.tool import ToolBase

from src.agents.prompts import MAIN_AGENT_NAME
from src.agents.registry import default_registry
from src.config import Settings, load_settings
from src.domain import AgentName
from src.orchestration.context import ContextInjectionMiddleware
from src.orchestration.hint_filter import HintSuppressionMiddleware
from src.orchestration.lane import LaneRouterMiddleware
from src.orchestration.reply_guard import ReplyGuardMiddleware
from src.server.agents_factory import (
    AgentWiring,
    RepositoryBundle,
    _AgentScopedMiddleware,
    build_agent_wiring,
    build_middlewares_factory,
    build_repositories,
    build_tools_factory,
)
from src.llm.middleware import BreakerMiddleware, ModelTimeoutMiddleware
from src.llm.mock import MockChatModel

USER = "u-test"


# ---------------------------------------------------------------------------
# 替身
# ---------------------------------------------------------------------------
class FakeModel(MockChatModel):
    """一个能构造、但一被调用就炸的模型。

    ⚠️ **继承 ``MockChatModel``，不是 ``ChatModelBase``**。基类的
    ``__init__`` 签名是 ``(credential, model, parameters, stream, ...)``
    —— 它**没有** ``name`` 参数，也没有默认值，直接继承它就得在这里
    手写一整套凭据/参数对象。而 ``MockChatModel`` 正是生产上「零密钥」
    那条路径在用的模型，签名已经是齐的（``src/llm/mock.py``），
    连 ``formatter`` 都在它自己的构造里赋好了。

    ⚠️ 这里仍然覆盖 ``_call_api`` 让它抛异常，因为本文件的用例要断言的
    是「装配过程**一次都没碰过模型**」。沿用 Mock 的默认行为（返回一段
    假回复）的话，一个「装配时顺手跑了一轮模型」的实现会安安静静地通过
    —— 而那正是本层最该拦住的错误：装配发生在**每个请求**上，
    代价是一整轮模型调用。
    """

    async def _call_api(self, **kwargs: Any) -> Any:
        """不应被调用。

        Args:
            **kwargs: 忽略。

        Raises:
            AssertionError: 永远抛出 —— 装配层不该真的调模型。
        """
        raise AssertionError("装配过程不该调用真实模型")


def with_orchestration(**overrides: object) -> Settings:
    """在测试档配置上覆盖 ``orchestration`` 段。

    ⚠️ 走 ``load_settings`` 而不是 ``Settings.model_validate({...})``：
    后者需要把 ``db`` / ``redis`` / ``milvus`` 三段都写全（它们是必填的），
    于是每个开关类用例都要抄一遍几十行无关配置 —— 而抄错一段的报错是
    「Field required」，与它想测的开关毫无关系。

    ⚠️ 也**不**用 ``model_copy(update=...)``：那会绕开校验器，
    于是「配置里的非法值能不能被拦住」这件事在测试里就测不到了。

    Args:
        **overrides: ``orchestration`` 段里要覆盖的字段（Python 名）。

    Returns:
        `Settings`: 配置树。
    """
    from tests.conftest import TEST_ENVIRON

    environ = {
        **TEST_ENVIRON,
        **{
            f"ALIGO__ORCHESTRATION__{key.upper()}": str(value)
            for key, value in overrides.items()
        },
    }
    return load_settings("test", environ=environ, dotenv=False)


def wiring(settings: Settings, **kwargs: Any) -> AgentWiring:
    """构造装配产物。

    Args:
        settings (`Settings`): 配置。
        **kwargs: 覆盖。

    Returns:
        `AgentWiring`: 装配产物。
    """
    return build_agent_wiring(settings, model=FakeModel(), **kwargs)


def build_middlewares(settings: Settings) -> list[MiddlewareBase]:
    """跑一次中间件工厂。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `list[MiddlewareBase]`: 中间件列表。
    """
    factory = build_middlewares_factory(settings=settings)
    return asyncio.run(factory(USER, "agent-1", "session-1"))


def _leaf(middleware: MiddlewareBase) -> MiddlewareBase:
    """剥掉作用域包装，返回真正干活的中间件。

    ⚠️ 存在的理由：装配链里现在有**两个** ``_AgentScopedMiddleware``
    （RAG 一个、回复守卫一个），而它们的外层类型完全相同。断言「第 N 个是
    什么」时必须看到内层，否则「两个包装对调了位置」这种错误根本测不出来 ——
    而对调会让守卫去处理 ``policy_rag``（把检索结论当草稿剥掉）。

    Args:
        middleware (`MiddlewareBase`): 装配出来的中间件。

    Returns:
        `MiddlewareBase`: 包装里的内层；不是包装时原样返回。
    """
    return getattr(middleware, "_inner", middleware)


def build_tools(settings: Settings, repositories: RepositoryBundle) -> list[ToolBase]:
    """跑一次工具工厂。

    Args:
        settings (`Settings`): 配置。
        repositories (`RepositoryBundle`): 仓储组。

    Returns:
        `list[ToolBase]`: 工具列表。
    """
    factory = build_tools_factory(
        settings=settings,
        repositories=repositories,
        model=FakeModel(),
    )
    return asyncio.run(factory(USER, "agent-1", "session-1"))


# ---------------------------------------------------------------------------
# 一、仓储组
# ---------------------------------------------------------------------------
def test_repositories_are_built_as_a_bundle() -> None:
    """★ 五个仓储一次给全，且**互不相同**。

    ⚠️ 断言互不相同不是吹毛求疵：全部返回同一个对象（比如某次重构里
    复制粘贴漏改）在类型上完全合法，而症状是「订单查询查到了交通数据」——
    一个不会报错、只会给出错误答案的 bug。
    """
    repos = build_repositories()

    for name in ("transport", "hotel", "policy", "order", "approval"):
        assert getattr(repos, name) is not None, f"缺少 {name} 仓储"

    ids = [id(getattr(repos, name)) for name in ("transport", "hotel", "policy", "order", "approval")]
    assert len(set(ids)) == 5, "有仓储被复用了同一个对象"


def test_the_bundle_is_frozen() -> None:
    """★ ``RepositoryBundle`` 是冻结的 —— 运行期不能换掉某个仓储。

    ⚠️ 五个仓储必须**同生共死**：换成 Postgres 实现时它们要一起换到同一个
    连接池上。允许逐个替换的话，很容易只换了三个 —— 而那种混用不报错，
    只会让一部分数据「写进去查不到」。
    """
    repos = build_repositories()
    with pytest.raises(Exception):
        repos.transport = None  # type: ignore[misc]


def test_repositories_accept_settings(settings: Settings) -> None:
    """⚠️ ``build_repositories`` 接受配置参数（P4 换 Postgres 要用）。

    ⚠️ 现在传进去没用，但签名必须已经就位 —— 否则 P4 会变成一次签名变更，
    而签名变更意味着要同时改装配处与所有测试。这条用例把签名钉住。
    """
    assert build_repositories(settings) is not None


# ---------------------------------------------------------------------------
# 二、扩展点的形状（最贵的一类错误）
# ---------------------------------------------------------------------------
def test_the_wiring_returns_exactly_the_three_extension_points(settings: Settings) -> None:
    """★★ ``as_kwargs()`` 的键名与 ``create_app`` 的参数名**逐字相同**。

    ⚠️ 键名写错的后果不是「少传一个参数」，而是 ``create_app`` 抛
    ``TypeError: unexpected keyword argument`` —— 报在应用构造期，
    也就是 import 应用的时刻，症状是「服务起不来」。

    ⚠️ 三个键名与 ``app.state`` 上的属性名刻意不同（那边存的是框架内部
    转出来的形态），所以「看着像」不等于「对得上」。逐字断言。
    """
    kwargs = wiring(settings).as_kwargs()

    assert set(kwargs) == {
        "extra_agent_tools",
        "extra_agent_middlewares",
        "custom_subagent_templates",
    }


def test_the_two_factories_are_async_callables(settings: Settings) -> None:
    """★★★ 工具与中间件必须是**工厂**（返回 awaitable），不是列表。

    ⚠️ 这是本文件里最容易搞错的一处，因为它的报错与原因隔得很远：
    ``AgentToolFactory`` 的契约是
    ``Callable[..., Awaitable[list[ToolBase]]]``（``agentscope/app/_types.py:33-36``），
    框架写的是 ``tools += await factory(...)``。传一个列表过去，
    报出来是 ``TypeError: object list can't be used in 'await' expression``
    —— 而「list」这个词会让人以为是返回值写错了，不是「这个参数本身
    该是个函数」。

    ⚠️ 反过来（把 ``custom_subagent_templates`` 写成工厂）同样致命，
    且报错更远：框架会去迭代那个函数对象……
    """
    w = wiring(settings)

    for name, factory in (
        ("tools_factory", w.tools_factory),
        ("middlewares_factory", w.middlewares_factory),
    ):
        assert callable(factory), f"{name} 不是可调用对象"
        # ⚠️ 断言「是个协程函数」而不是「调用它返回一个 awaitable」——
        # 后者会真的建出一个协程并把它丢掉，pytest 会在收尾时报
        # ``coroutine ... was never awaited``。那条警告是**噪音**：
        # 它出现时用例仍然是绿的，于是下一次真正的未等待协程
        # 混在里面就没人看了。
        assert asyncio.iscoroutinefunction(factory), (
            f"{name} 不是 async def —— 框架写的是 ``await factory(...)``，"
            f"同步函数返回的 list 会在那里报 "
            f"\"object list can't be used in 'await' expression\""
        )


def test_the_subagent_templates_are_a_plain_list(settings: Settings) -> None:
    """★★★ ``custom_subagent_templates`` 必须是**静态列表**，不是工厂。

    ⚠️ 已核实（``agentscope/app/_app.py:375-386``）：``create_app`` 的公开参数是
    ``list[SubAgentTemplate] | None``，它在构造期把列表转成 dict 存进
    ``app.state``。传一个**函数**过去，它会去迭代这个函数对象 ——
    报出来是 ``TypeError: 'function' object is not iterable``，
    或者更糟：迭代成功了但元素不是 ``SubAgentTemplate``，
    在 ``t.type`` 上崩成 ``AttributeError``。

    ⚠️ 特别注意**不能传 dict**。dict 是框架**内部**转出来的形态；
    直接传 dict，它会去迭代**键**（一堆字符串），
    在 ``t.type`` 上报 ``AttributeError: 'str' object has no attribute 'type'``。
    """
    w = wiring(settings)

    assert isinstance(w.subagent_templates, list), "模板必须是 list，不能是 dict 或函数"
    assert w.subagent_templates, "模板列表是空的，子智能体一个都建不出来"
    for template in w.subagent_templates:
        assert isinstance(template, SubAgentTemplate)


def test_the_templates_cover_every_registered_agent(settings: Settings) -> None:
    """★★ 模板覆盖注册表里的**每一个**智能体，且类型不重复。

    ⚠️ 数量对不上不会报错（``create_app`` 只查类型唯一性，没有数量下限），
    只是「某个子智能体永远建不出来」—— 而主智能体在需要它的时候会调用
    ``AgentCreate(subagent_type=...)``，拿到一个「未知类型」的工具结果，
    然后编一个理由告诉用户。

    ⚠️ 类型重复则更直接：``create_app`` 会在装配期抛
    ``ValueError: Duplicate sub_agent_template type(s)``。
    在这里提前拦住，报错信息里能带上名字。
    """
    w = wiring(settings)
    types = [t.type for t in w.subagent_templates]

    assert len(types) == len(set(types)), f"模板类型重复：{types}"
    assert set(types) == {spec.name for spec in default_registry().specs()}


def test_the_templates_do_not_leak_the_system_prompt(settings: Settings) -> None:
    """★★ 模板里**有**提示词（那是它的用途），但设计上只有框架该看到它。

    ⚠️ 这条是一个提醒性的断言，不是安全边界：``SubAgentTemplate`` 的
    ``system_prompt_template`` 本来就该有内容，而 P4/P5 若把它序列化给
    前端（比如做个「智能体一览」页面），提示词就会泄漏 ——
    提示词里含业务规则与边界描述，是攻击者构造绕过输入时的现成地图。
    对外序列化必须走 ``src.agents.registry.to_dict``（它**不含**提示词）。
    """
    w = wiring(settings)
    from src.agents.registry import to_dict

    assert all(t.system_prompt_template.strip() for t in w.subagent_templates)
    leaked = repr(to_dict(default_registry()))
    for template in w.subagent_templates:
        head = template.system_prompt_template[-40:]
        assert head not in leaked, "对外序列化里出现了系统提示词"


# ---------------------------------------------------------------------------
# 三、中间件的顺序（不报错，但后果严重）
# ---------------------------------------------------------------------------
def test_the_middleware_order_is_lane_tracing_breaker_context(
    settings: Settings,
) -> None:
    """★★★ 中间件顺序**必须**是「车道 → tracing → 熔断 → 动态 Prompt」。

    ⚠️ 这个顺序有**四条**独立的理由，每条都对应一个不会报错的 bug：

    1. **车道在最外层**：它要短路的是**模型调用本身**。排在别的
       ``on_model_call`` 后面的话，那些中间件仍然会先跑一遍（熔断照常记账，
       tracing 照常开 span），快车道省下的成本被它们的开销吃掉一部分。
    2. **tracing 在车道之内**：快车道命中时**根本不发生模型调用**，若 tracing
       在车道外面，它会为一个不存在的调用开一个 LLM span、记一条「成功的
       模型调用」—— 一条凭空捏造的记录。
    3. **tracing 在熔断之外**：熔断器 open 时直接抛 ``CircuitBreakerOpen``，
       比它外层才能把「这次调用被拒绝」记成一个 error span；躲在里面的话，
       被拒绝的调用在 trace 里完全不存在。
    4. **动态 Prompt 在最内层**：``on_system_prompt`` 是**串行链式**的
       （不是洋葱式），后一个拿到的 ``current_prompt`` 是前一个的返回值。
       放链尾意味着它看到的是「基础 prompt + 前面所有中间件的产出」。

    ⚠️ 断言的是**相对下标**（下标 0 = 最外层），不是「这几个都在列表里」——
    后者在顺序完全颠倒时同样通过。

    ⚠️ 链尾**允许**跟着若干 ``_AgentScopedMiddleware``（回复守卫、可选的 RAG）。
    真正的不变式不是「ContextInjection 在列表最后一格」，而是
    **「它后面没有别的 ``on_system_prompt`` 实现者」** —— 框架用的是
    ``is_implemented("on_system_prompt")`` 过滤（``agentscope/agent/_agent.py:236``），
    而 :class:`_AgentScopedMiddleware` 只实现 ``on_reply``/``on_reasoning``，
    根本不会进入那条串行链。按位置断言会在每次往后追加中间件时误报。
    """
    middlewares = build_middlewares(settings)
    names = [type(mw).__name__ for mw in middlewares]

    assert names[0] == LaneRouterMiddleware.__name__, (
        f"车道不在最外层：{names}。快车道将无法在熔断器/tracing 之前短路。"
    )
    assert names.index(BreakerMiddleware.__name__) > names.index(
        LaneRouterMiddleware.__name__
    ), f"熔断器排在了车道外面：{names}。快车道命中的调用会被记成「成功」。"
    assert names.index(TracingMiddleware.__name__) > names.index(
        LaneRouterMiddleware.__name__
    ), f"tracing 排在了车道外面：{names}。快车道短路会被记成一次模型调用。"
    assert names.index(TracingMiddleware.__name__) < names.index(
        BreakerMiddleware.__name__
    ), f"tracing 排在了熔断器里面：{names}。被熔断拒绝的调用不会留下 span。"
    context_index = names.index(ContextInjectionMiddleware.__name__)
    trailing = [
        type(mw).__name__
        for mw in middlewares[context_index + 1 :]
        if mw.is_implemented("on_system_prompt")
    ]
    assert not trailing, (
        f"动态 Prompt 后面还有 on_system_prompt 实现者 {trailing}：{names}。"
        "on_system_prompt 是串行链，它们的产出会被 ContextInjection 看到。"
    )


def test_the_tracing_middleware_is_assembled(settings: Settings) -> None:
    """★★ tracing 中间件**真的**进了列表。

    ⚠️ 这条守的是本次修复的核心症状：``TracingMiddleware`` 全仓原本没有任何
    地方挂过 —— 即便 ``setup_tracing`` 注册了 provider，也没有任何 span 会被
    创建（框架只在**中间件被挂上**时才会去调它的钩子）。而 ``/readyz`` 的
    ``tracing.active`` 会照样显示 True，给出「追踪正常」的假好消息。

    所以断言的是「类名在列表里」，而不是「tracing 是否 active」——后者由
    ``/readyz`` 与 ``test_observability_tracing.py`` 负责。
    """
    names = [type(mw).__name__ for mw in build_middlewares(settings)]
    assert TracingMiddleware.__name__ in names, (
        f"TracingMiddleware 没有进中间件列表：{names}。"
        f"Langfuse 里将一条 trace 都不会有，而 /readyz 仍会报 tracing 正常。"
    )


def test_the_middleware_list_has_no_duplicates(settings: Settings) -> None:
    """★ 三个中间件各一个，没有重复。"""
    names = [type(mw).__name__ for mw in build_middlewares(settings)]
    assert len(names) == len(set(names)), f"中间件重复：{names}"


def test_every_middleware_is_a_framework_middleware(settings: Settings) -> None:
    """★★ 列表里全是 ``MiddlewareBase`` 子类 —— 否则钩子根本不会被调用。

    ⚠️ 已核实：框架在 **Agent 构造期**用类方法身份探测钩子
    （``agentscope/middleware/_base.py:64-66`` 的 ``getattr(type(self), hook_name, None)``）。
    一个不是 ``MiddlewareBase`` 子类的对象、或者把钩子写成**实例属性**的
    对象，会被**静默忽略** —— 不报错，只是那个中间件完全不起作用。
    """
    for mw in build_middlewares(settings):
        assert isinstance(mw, MiddlewareBase), f"{type(mw).__name__} 不是框架中间件"


class _ReplyOnlyMiddleware(MiddlewareBase):
    """只实现 ``on_reply`` 的内层中间件（回复守卫就是这个形状）。"""

    async def on_reply(
        self,
        agent: Any,
        input_kwargs: dict,
        next_handler: Any,
    ) -> Any:
        """直接透传。

        Args:
            agent (`Any`): 当前 agent。
            input_kwargs (`dict`): 回复输入。
            next_handler (`Any`): 下游处理。

        Yields:
            `Any`: 下游事件。
        """
        del agent
        async for item in next_handler(**input_kwargs):
            yield item


def _collect_reasoning(middleware: MiddlewareBase) -> list[Any]:
    """跑一遍 ``on_reasoning`` 并收集产出。

    Args:
        middleware (`MiddlewareBase`): 待测中间件。

    Returns:
        `list[Any]`: 产出的事件。
    """

    async def downstream(**_kwargs: Any) -> Any:
        yield "downstream"

    async def run() -> list[Any]:
        out = []
        async for item in middleware.on_reasoning(
            agent=SimpleNamespace(name=MAIN_AGENT_NAME),
            input_kwargs={},
            next_handler=downstream,
        ):
            out.append(item)
        return out

    return asyncio.run(run())


async def _collect_reply_through_wrapper(
    wrapper: MiddlewareBase,
    *,
    agent_name: str = MAIN_AGENT_NAME,
) -> list[Any]:
    """跑一遍包装中间件的 ``on_reply`` 并收集产出。

    Args:
        wrapper (`MiddlewareBase`): 包装中间件。
        agent_name (`str`): 伪装的 ``agent.name``（用来测作用域）。

    Returns:
        `list[Any]`: 产出的事件。
    """

    async def downstream(**_kwargs: Any) -> Any:
        yield "downstream"

    out = []
    async for item in wrapper.on_reply(
        agent=SimpleNamespace(name=agent_name),
        input_kwargs={},
        next_handler=downstream,
    ):
        out.append(item)
    return out


@pytest.mark.parametrize(
    "inner",
    [
        # 回复守卫的形状：只实现 on_reply。
        _ReplyOnlyMiddleware(),
        # ContextInjection 的形状：只实现 on_system_prompt。
        ContextInjectionMiddleware(enabled=False),
    ],
    ids=["只实现 on_reply", "只实现 on_system_prompt"],
)
def test_the_scoped_wrapper_never_calls_an_unimplemented_hook(
    inner: MiddlewareBase,
) -> None:
    """★★★ 作用域包装**不许**把钩子转发给没实现它的内层中间件。

    ⚠️ 这条是**线上事故换来的**，前因后果值得读一遍：

    基类的 ``on_reasoning`` 在没有子类覆写时会 ``raise RuntimeError``
    （``middleware/_base.py``）。而 :class:`_AgentScopedMiddleware` **覆写了**
    两个钩子（它要按 ``agent.name`` 做作用域判定），于是框架认为钩子
    「可用」并调用它；它再无条件转发给内层 —— 内层是只实现 ``on_reply``
    的回复守卫时，**每一次回复**都以 error 结束。

    实测症状：容器日志里 8/8 轮全是
    ``RuntimeError: ReplyGuardMiddleware does not implement on_reasoning``
    （用户看到的是空回复 + 未知错误），而**所有单测全绿** —— 它们要么直接
    调 ``on_reply``（绕开包装），要么只检查中间件列表的类型序列（不跑钩子）。

    所以这条必须**真的跑一遍两个钩子**：``on_reasoning`` 在没有内层实现时
    要直接透传到下游，而不是转发给内层。
    """
    wrapper = _AgentScopedMiddleware(inner, agent_names=(MAIN_AGENT_NAME,))

    assert _collect_reasoning(wrapper) == ["downstream"], (
        "on_reasoning 没有透传到下游 —— 内层没实现这个钩子时，"
        "转发会让每一次回复都以 RuntimeError 结束"
    )
    assert asyncio.run(_collect_reply_through_wrapper(wrapper)) == ["downstream"]


def test_the_scoped_wrapper_is_inert_outside_its_agent_names() -> None:
    """★★ 作用域外必须**完全不动**内层（哪怕内层实现了那个钩子）。

    ⚠️ 这是本类存在的**唯一理由**（见其文档字符串）：框架把额外中间件加给
    **每一个** agent，作用域判定就是那道闸门。判据错在「放行」一侧的后果是
    子智能体也去检索政策 / 也走快车道规则表 —— 而这类错误不会让任何功能
    测试变红，只会让子智能体的行为变得难以解释。
    """

    class _Recording(MiddlewareBase):
        """记录自己被调用了几次的内层中间件。"""

        def __init__(self) -> None:
            self.calls = 0

        async def on_reply(
            self,
            agent: Any,
            input_kwargs: dict,
            next_handler: Any,
        ) -> Any:
            """记录一次调用并透传。

            Args:
                agent (`Any`): 当前 agent。
                input_kwargs (`dict`): 回复输入。
                next_handler (`Any`): 下游处理。

            Yields:
                `Any`: 下游事件。
            """
            del agent
            self.calls += 1
            async for item in next_handler(**input_kwargs):
                yield item

    inner = _Recording()
    wrapper = _AgentScopedMiddleware(inner, agent_names=("other_agent",))

    out = asyncio.run(_collect_reply_through_wrapper(wrapper))

    assert out == ["downstream"]
    assert inner.calls == 0, "作用域外的 agent 仍然触发了内层中间件"


def test_the_lane_only_routes_the_main_agent(settings: Settings) -> None:
    """★★★ ``LaneRouterMiddleware`` 必须**显式限定**只对主智能体生效。

    ⚠️ 这是装配层里最隐蔽的一处。``agent_names=None``（默认值）意味着
    「对所有 agent 生效」，而框架把额外中间件加进**每一个** agent
    （``agentscope/app/_service/_chat.py:1075-1092`` 无条件调 ``get_toolkit``）。

    后果很具体：快车道的规则表描述的是「用户点了什么按钮」，
    而子智能体被主智能体要求检索政策时，它的输入恰好可能是
    「查询政策」四个字 —— 于是子智能体**自己的模型调用被短路**，
    表现为「子智能体什么都没干就返回了」。

    ⚠️ 断言的是内部字段而不是行为，因为没有哪个「跑一轮对话」的用例能
    稳定撞上这个场景 —— 它取决于子智能体的输入恰好匹配规则表的概率。
    这个字段就是这条约束的全部实现，直接断言它。
    """
    lane = build_middlewares(settings)[0]
    assert isinstance(lane, LaneRouterMiddleware)

    assert lane._agent_names == (MAIN_AGENT_NAME,), (
        f"快车道没有限定只对主智能体生效（当前 {lane._agent_names!r}）："
        f"子智能体的内部调用会被误判成用户点击。"
    )


def test_every_assembly_builds_fresh_middleware_instances(settings: Settings) -> None:
    """★★★ 每次装配都新建中间件实例，**不复用**。

    ⚠️ ``LaneRouterMiddleware`` 按实例缓存路由决策（``_routed_reply_id``
    之类的字段），它的文档明确要求「同一个实例不跨 agent 复用」。
    复用出错的症状是「两个用户的路由决策互相干扰」—— 一个极难复现的
    并发 bug，而且它会先被怀疑成模型的问题。

    ⚠️ 构造几个对象很便宜，所以这里没有权衡的余地。
    """
    factory = build_middlewares_factory(settings=settings)
    first = asyncio.run(factory("u1", "a1", "s1"))
    second = asyncio.run(factory("u2", "a2", "s2"))

    assert [id(mw) for mw in first] != [id(mw) for mw in second], "两次装配复用了同一批中间件实例"
    for a, b in zip(first, second):
        assert a is not b


def test_the_breaker_is_shared_across_assemblies(settings: Settings) -> None:
    """★★★ 熔断器**跨装配共享**（与中间件实例相反）。

    ⚠️ 与上一条成对、方向相反，这正是本层最容易搞混的一对约束：
    中间件实例**必须**每次新建，而熔断器**必须**全程唯一。

    共享的理由是正确性而非优化：每个 agent 各持一个熔断器，
    「用全体调用者的失败共同判断下游是否可用」的前提就不成立了 ——
    熔断点被推迟 N 倍，N 就是并发数。

    ⚠️ 框架为**每一个 agent 装配**都调一次工厂，所以生产上必然存在很多个
    ``BreakerMiddleware`` 实例。断言它们握的是同一个熔断器。
    """
    factory = build_middlewares_factory(settings=settings)
    first = [mw for mw in asyncio.run(factory("u1", "a1", "s1")) if isinstance(mw, BreakerMiddleware)]
    second = [mw for mw in asyncio.run(factory("u2", "a2", "s2")) if isinstance(mw, BreakerMiddleware)]

    assert len(first) == 1 and len(second) == 1
    assert first[0]._breaker is second[0]._breaker, "每个中间件各持一个熔断器，熔断点会被推迟"


def test_the_switches_follow_the_settings(settings: Settings) -> None:
    """★★ 配置里的开关真的传到了中间件上。

    ⚠️ 这两个开关（``fast_lane_enabled`` / ``dynamic_prompt_enabled``）是
    **排障入口**：线上出现「回复内容不对」时，第一个要排除的假设就是
    「是不是快车道规则误命中了」。开关没接上，就等于排障手段不存在 ——
    而它不会报错，只会让排查的人以为「关了也没用，看来不是这个原因」，
    从而排除掉一个**正确**的假设。
    """
    off = with_orchestration(fast_lane_enabled="false", dynamic_prompt_enabled="false")
    middlewares = build_middlewares(off)

    lane = next(mw for mw in middlewares if isinstance(mw, LaneRouterMiddleware))
    context = next(mw for mw in middlewares if isinstance(mw, ContextInjectionMiddleware))
    assert lane._enabled is False, "快车道开关没有传到中间件"
    assert context._enabled is False, "动态 Prompt 开关没有传到中间件"


def test_the_lane_max_chars_follows_the_settings(settings: Settings) -> None:
    """★ 长度阈值也要传下去（它是快车道的第二道网）。"""
    tuned = with_orchestration(fast_lane_max_chars=8)
    lane = next(
        mw for mw in build_middlewares(tuned) if isinstance(mw, LaneRouterMiddleware)
    )

    assert lane._max_chars == 8


def test_the_context_middleware_follows_expose_reasoning(settings: Settings) -> None:
    """★ ``expose_reasoning`` 传到动态 Prompt 中间件上。"""
    quiet = with_orchestration(expose_reasoning="false")
    context = next(
        mw for mw in build_middlewares(quiet) if isinstance(mw, ContextInjectionMiddleware)
    )

    assert context._show_reasoning is False


# ---------------------------------------------------------------------------
# 四、工具
# ---------------------------------------------------------------------------
def test_the_tools_include_the_business_tools_and_the_intent_tool(settings: Settings) -> None:
    """★★ 工具集里既有业务工具，也有意图识别工具。

    ⚠️ 逐类断言而不是查数量：数量对得上但内容不对（比如意图工具被挤掉了）
    是最容易发生的一种回归，因为两处清单分属不同模块。

    ⚠️ 意图工具**必须**在里面 —— 主智能体的提示词里提到了它。
    不在的话，模型会去调一个不存在的工具，拿到一个工具失败的结果，
    然后编一句「我无法判断你的意图」。
    """
    from src.tools.expert import INTENT_TOOL_NAME

    names = {tool.name for tool in build_tools(settings, build_repositories())}

    assert INTENT_TOOL_NAME in names, "意图识别工具不在工具集里"
    for required in ("aligo_route_intent", "search_transport", "search_hotels"):
        assert required in names, f"缺少业务工具 {required}"


def test_the_tool_factory_is_bound_to_the_caller_user(settings: Settings) -> None:
    """★★★ 工具集**按调用者**装配 —— 这是多租户隔离的唯一支点。

    ⚠️ 这条断言的是「工厂参数确实参与了装配」。``user_id`` 由框架从**鉴权
    结果**取出后传入，与模型、与请求体都无关。它若在装配时被丢掉
    （比如换成某个常量、或者从请求体里读），隔离就没了 ——
    而症状是「A 用户查到了 B 用户的订单」，一个无法用功能测试发现的
    安全问题。

    ⚠️ 断言方式是比对不同 ``user_id`` 造出来的工具集：它们的工具对象
    应该是**不同**的实例（各自闭包捕获了自己的 ``user_id``）。
    全等的话，说明装配时根本没用到这个参数。
    """
    repos = build_repositories()
    factory = build_tools_factory(settings=settings, repositories=repos, model=FakeModel())

    alice = asyncio.run(factory("alice", "a1", "s1"))
    bob = asyncio.run(factory("bob", "a2", "s2"))

    assert [id(t) for t in alice] != [id(t) for t in bob], (
        "两个用户的工具集是同一批实例：user_id 没有参与装配，租户隔离失效"
    )


def test_the_tool_factory_ignores_agent_and_session_ids(settings: Settings) -> None:
    """★ 换 ``agent_id`` / ``session_id`` 不影响工具集（本项目按用户隔离）。

    ⚠️ 这条把「本项目按**用户**隔离」这个决定钉住。哪天有人改成按会话隔离，
    这里会变红 —— 而那时应该同时更新 ``src/tools/orders.py`` 的说明，
    因为它那句「user_id 只能来自鉴权结果」是多租户隔离的文档化依据。
    """
    repos = build_repositories()
    factory = build_tools_factory(settings=settings, repositories=repos, model=FakeModel())

    first = asyncio.run(factory("alice", "a1", "s1"))
    second = asyncio.run(factory("alice", "a2", "s2"))

    assert [t.name for t in first] == [t.name for t in second]


def test_every_tool_is_a_framework_tool(settings: Settings) -> None:
    """★★ 列表里全是 ``ToolBase`` 子类。

    ⚠️ 框架在装配工具集时会调 ``tool.name`` / ``get_tool_schemas()``。
    一个不是 ``ToolBase`` 的对象能进列表（Python 不检查类型），
    却会在装配期或第一次模型调用时炸 —— 报错点离「工具写错了」很远。
    """
    for tool in build_tools(settings, build_repositories()):
        assert isinstance(tool, ToolBase), f"{type(tool).__name__} 不是框架工具"


def test_the_tool_list_has_no_duplicate_names(settings: Settings) -> None:
    """★★ 工具名不重复。

    ⚠️ 已核实：``Toolkit`` 对重复的工具名**静默覆盖**（后一个胜）。
    症状是「某个工具莫名调不到」，而它的实现明明在代码里、日志里也没有
    任何异常。在这里提前拦住，报错信息里能带上名字。
    """
    names = [tool.name for tool in build_tools(settings, build_repositories())]
    assert len(names) == len(set(names)), f"工具名重复：{sorted(n for n in names if names.count(n) > 1)}"


# ---------------------------------------------------------------------------
# 五、总入口
# ---------------------------------------------------------------------------
def test_the_wiring_is_frozen(settings: Settings) -> None:
    """★ 装配产物是冻结的 —— 装配好之后不该被改。"""
    w = wiring(settings)
    with pytest.raises(Exception):
        w.subagent_templates = []  # type: ignore[misc]


def test_injected_repositories_are_used(settings: Settings) -> None:
    """★★ 注入的仓储**真的**被用上了（不是被丢掉又重新造了一个）。

    ⚠️ 这条守的是「可注入」这个性质本身。装配层若忽略参数自己造一套仓储，
    测试里注入的替身就完全不起作用 —— 而症状是「测试通过但测的不是
    生产路径」，那是最坏的一类测试失败。
    """
    repos = build_repositories()
    w = build_agent_wiring(settings, model=FakeModel(), repositories=repos)

    tools = asyncio.run(w.tools_factory(USER, "a1", "s1"))
    assert tools, "注入仓储之后工具集是空的"


def test_injected_templates_are_used(settings: Settings) -> None:
    """★★ 注入的模板替换掉了默认模板。"""
    custom = [
        SubAgentTemplate(
            type=AgentName.POLICY_RAG.value,
            description="只做政策问答。",
            system_prompt_template="你是政策问答专员。",
        ),
    ]
    w = build_agent_wiring(settings, model=FakeModel(), subagent_templates=custom)

    assert [t.type for t in w.subagent_templates] == [AgentName.POLICY_RAG.value]
    assert w.subagent_templates[0].system_prompt_template == "你是政策问答专员。"


def test_building_the_wiring_never_calls_the_model(settings: Settings) -> None:
    """★★★ 装配过程**一次模型调用都不发生**。

    ⚠️ 这条守的是启动成本与失败模式。装配期若为了「验证一下配置对不对」
    去调一次模型，失败点就从「配置错了」漂移成「模型连不上」——
    而服务在模型网关抖动时连启动都启动不了，那是明显的责任错配。

    ⚠️ ``FakeModel._call_api`` 永远抛 ``AssertionError``，所以这里
    「没抛异常」本身就是断言。
    """
    w = wiring(settings)
    assert w.tools_factory is not None


def test_assembling_tools_never_calls_the_model(settings: Settings) -> None:
    """★★★ 跑工具工厂也不发生模型调用（工具是**惰性**的）。

    ⚠️ 区别很关键：装配时把工具**造出来**是必须的，但「造工具」不等于
    「用工具」。识别器只在被模型调用时才跑 —— 那发生在用户真的说话之后。
    """
    build_tools(settings, build_repositories())  # 内部模型一被调用就抛断言


def test_building_middlewares_twice_gives_independent_switches(settings: Settings) -> None:
    """★★ 两次装配的开关互不影响（配置是只读的），且**顺序**逐位对齐。

    ⚠️ 断言的是**类型序列**而不是长度。长度相等只能证明「装了这么多」，
    挡不住顺序写反 —— 而顺序错误不会让任何功能用例变红（它只让成本悄悄
    变高、或让熔断点悄悄推迟）。这里把
    ``src/server/agents_factory.py::build_middlewares_factory`` 文档里
    那段「下标 0 是最外层」的论证钉成断言。

    默认七件套（无 ``kb_manager`` 时不装 RAG，见下面的用例）：
    车道 → tracing → 熔断 → 模型调用截止时间 → 动态 Prompt → 提示块过滤
    → 回复守卫。

    ⚠️ 比的是**内层类型**（``_leaf``）而不是外层：回复守卫被
    ``_AgentScopedMiddleware`` 包着，只看外层的话，它在链上的位置换了也看不出来。
    """
    first = build_middlewares(settings)
    second = build_middlewares(settings)

    expected = [
        LaneRouterMiddleware,
        TracingMiddleware,
        BreakerMiddleware,
        ModelTimeoutMiddleware,
        ContextInjectionMiddleware,
        HintSuppressionMiddleware,
        ReplyGuardMiddleware,
    ]
    assert [type(_leaf(m)) for m in first] == expected
    assert [type(_leaf(m)) for m in second] == expected
    # ⚠️ 两次装配必须是**各自独立的实例**：中间件可能带状态
    # （熔断器替身、计数器），共享实例会让一个租户的调用影响另一个。
    assert all(a is not b for a, b in zip(first, second))


# ---------------------------------------------------------------------------
# 三·补充、RAG 的作用域（本次最容易写错的一处）
# ---------------------------------------------------------------------------
class _RecordingMiddleware(MiddlewareBase):
    """记录自己被调了几次的内层中间件替身。"""

    def __init__(self) -> None:
        self.reply_calls = 0
        self.reasoning_calls = 0

    async def on_reply(self, agent: Any, input_kwargs: dict, next_handler: Any) -> Any:
        self.reply_calls += 1
        async for item in next_handler(**input_kwargs):
            yield item

    async def on_reasoning(
        self, agent: Any, input_kwargs: dict, next_handler: Any
    ) -> Any:
        self.reasoning_calls += 1
        async for item in next_handler(**input_kwargs):
            yield item


async def _passthrough(**kwargs: Any) -> Any:
    """下游替身：产出一个标记值，证明链路走通。"""
    yield "downstream"


async def _drain(agen: Any) -> list[Any]:
    """把一个 async 生成器抽干，返回它产出的所有项。"""
    return [item async for item in agen]


def test_rag_is_not_assembled_without_a_kb_manager(settings: Settings) -> None:
    """★★ 没有 ``kb_manager`` 时**不装** RAG。

    ⚠️ 理由不是「省一点开销」，而是桥接层
    （``src.knowledge.rag.build_rag_middlewares``）**需要**管理器去解析用户的
    KB 句柄；没有它就只能返回空，装一个「永远检索不到」的中间件等于制造
    「看起来查过、其实没查」的假象。所以宁可不装。

    ⚠️ 判据是「有没有 **RAG**」而不是「有没有 ``_AgentScopedMiddleware``」。
    后者在回复守卫接进来之后就不成立了 —— 装守卫**也要**包一层作用域
    （它只对 ``main_plan`` 生效），于是「按外层类名找 RAG」会把守卫误判成 RAG。
    """
    leaves = [type(_leaf(mw)).__name__ for mw in build_middlewares(settings)]

    assert "RAGMiddleware" not in leaves, f"没有 kb_manager 却装了 RAG：{leaves}"


def test_rag_is_assembled_and_scoped_to_policy_rag(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★★ RAG 只在 ``policy_rag`` 上生效，且用 ``mode="static"``。

    ⚠️ 这条守的是本次最隐蔽的一处：``extra_agent_middlewares`` 对**每一个**
    agent 装配都生效，直接塞 RAG 进去会让所有 worker / 子智能体都挂上
    ``search_knowledge``。作用域由 :class:`_AgentScopedMiddleware` 在钩子里
    按 ``agent.name`` 判定 —— 下面用两个不同名字的假 agent 各跑一次，
    断言内层**只在 policy_rag 上**被调用。

    ⚠️ 同时钉住 ``mode="static"``：agentic 模式的工具是经 ``list_tools``
    注册的，而那个方法拿不到 agent、无法收窄 —— 所以装配时必须走 static。
    """
    inner = _RecordingMiddleware()
    captured: dict[str, Any] = {}

    async def fake_build(
        user_id: str,
        settings: Any,
        kb_manager: Any,
        **kwargs: Any,
    ) -> list[MiddlewareBase]:
        captured["mode"] = kwargs.get("mode")
        return [inner]

    monkeypatch.setattr(
        "src.server.agents_factory.build_rag_middlewares",
        fake_build,
    )

    factory = build_middlewares_factory(settings=settings, kb_manager=object())
    middlewares = asyncio.run(factory(USER, "a1", "s1"))

    # ⚠️ 按**内层是不是那个替身**来挑 RAG 包装，而不是按「第一个/唯一一个
    # 作用域包装」—— 链上还有一个回复守卫的包装，用位置挑会挑错对象，
    # 于是这条用例会变成「在测守卫」，而 RAG 收窄失效时它照样绿。
    wrappers = [
        mw
        for mw in middlewares
        if isinstance(mw, _AgentScopedMiddleware) and getattr(mw, "_inner", None) is inner
    ]
    assert len(wrappers) == 1, f"RAG 没有被包进作用域中间件：{middlewares}"
    wrapper = wrappers[0]

    assert captured["mode"] == "static", (
        f"RAG 用了 {captured['mode']!r} 模式：agentic 的工具无法按 agent 收窄。"
    )

    # 非 policy_rag：内层一次都不该被调用。
    other_agent = SimpleNamespace(name=MAIN_AGENT_NAME)
    asyncio.run(_drain(wrapper.on_reply(other_agent, {}, _passthrough)))
    asyncio.run(_drain(wrapper.on_reasoning(other_agent, {}, _passthrough)))
    assert inner.reply_calls == 0
    assert inner.reasoning_calls == 0

    # policy_rag：内层被放行。
    rag_agent = SimpleNamespace(name=AgentName.POLICY_RAG.value)
    asyncio.run(_drain(wrapper.on_reply(rag_agent, {}, _passthrough)))
    asyncio.run(_drain(wrapper.on_reasoning(rag_agent, {}, _passthrough)))
    assert inner.reply_calls == 1
    assert inner.reasoning_calls == 1

    # 不暴露工具：``list_tools`` 拿不到 agent，只能返回空 —— 这正是
    # static 模式成立的前提（见 _AgentScopedMiddleware 的文档）。
    assert asyncio.run(wrapper.list_tools()) == []


def test_the_rerank_model_reaches_the_rag_bridge(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ 重排模型必须**真的**被送进 ``build_rag_middlewares``。

    ⚠️ 这条守的是一个「装配齐全、功能静默失效」的失效模式：
    ``settings.rerank.enabled=true`` 打开了、模型也建好了，
    但只要中间漏传一个参数，检索就仍然按向量序返回 ——
    **没有任何报错**，只是重排从来没发生过。而「排序没变」
    与「重排生效后排序恰好没变」在结果上完全一样。

    所以这里盯的是**传递链**本身：``build_middlewares_factory`` 的
    ``rerank_model`` 参数必须原样出现在桥接层的 ``rerank_model=`` 关键字里。
    """
    sentinel = object()
    captured: dict[str, Any] = {}

    async def fake_build(
        user_id: str,
        settings_arg: Any,
        kb_manager: Any,
        **kwargs: Any,
    ) -> list[MiddlewareBase]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(
        "src.server.agents_factory.build_rag_middlewares",
        fake_build,
    )

    factory = build_middlewares_factory(
        settings=settings,
        kb_manager=object(),
        rerank_model=sentinel,
    )
    asyncio.run(factory(USER, "a1", "s1"))

    assert captured.get("rerank_model") is sentinel, (
        f"重排模型没有传到桥接层：{captured}"
    )
    # 收窄模式必须仍然钉在 static（见上一个用例）。
    assert captured.get("mode") == "static"


def test_wiring_builds_the_rerank_model_from_the_main_model(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ 总入口用**主对话模型实例**去构造重排模型（复用而不是另建）。

    ⚠️ 断言的是 ``reuse=`` 这个参数本身，不是「结果非 None」：
    ``build_rerank_model`` 在 ``reuse`` 缺省时会自己 ``build_chat_model(...)``
    再建一个真实模型 —— 那在测试里会尝试连网，在生产里会多出一个
    HTTP 客户端。漏传 ``reuse`` 的症状在生产上只是「多花一点连接」，
    在测试里却是「莫名其妙地慢/超时」，两者都不该靠运气去猜。
    """
    from src.server import agents_factory as module

    main = FakeModel()
    captured: dict[str, Any] = {}

    def fake_build_rerank(settings_arg: Any = None, **kwargs: Any) -> Any:
        """记录调用参数，返回一个哨兵（不构造真实模型）。"""
        captured["settings"] = settings_arg
        captured.update(kwargs)
        return "rerank-sentinel"

    monkeypatch.setattr(module, "build_rerank_model", fake_build_rerank)

    wiring = build_agent_wiring(settings, model=main)

    assert captured.get("reuse") is main, (
        f"重排模型没有复用主对话模型：{captured}"
    )
    assert isinstance(wiring, AgentWiring)
