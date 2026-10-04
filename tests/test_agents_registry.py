# -*- coding: utf-8 -*-
"""智能体注册表（``src/agents/registry.py``）的测试。

═══ 这张表守的是什么 ═══

「有哪些智能体」这个事实被至少四处消费：提示词表、意图到智能体的路由表、
思考链的显示名、框架的子智能体模板。四处各写一份的后果不是「重复」，
而是**漂移** —— 加了第六个智能体，四处里改了三处，剩下那处不报错，
只是那个智能体永远拿不到提示词或永远不被调度。

所以本文件的一半用例在守「两个方向都要对得上」，另一半在守
**框架的模板契约**（``.format()`` 只有五个占位符），后者是这一层里
最容易踩、报错又最不直观的一处。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.agents.prompts import PROMPTS
from src.agents.registry import (
    MEMBER_HEADER,
    PLACEHOLDERS,
    AgentRegistry,
    AgentSpec,
    UnknownAgentError,
    build_subagent_templates,
    default_registry,
    to_dict,
)
from src.domain import AgentName


def spec(name: str = "demo", version: int = 1, **kwargs: object) -> AgentSpec:
    """构造一个测试用规格。

    Args:
        name (str): 智能体名。
        version (int): 版本号。
        **kwargs: 覆盖默认字段。

    Returns:
        `AgentSpec`: 规格。
    """
    params: dict[str, object] = {
        "name": name,
        "version": version,
        "description": "测试用智能体",
        "system_prompt": "你是一个测试用的助手。",
    }
    params.update(kwargs)
    return AgentSpec(**params)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 一、名册完整性（两个方向都要查）
# ---------------------------------------------------------------------------
def test_registry_covers_every_agent_name() -> None:
    """★★ 每个 :class:`AgentName` 成员都注册了，且表里没有多余的名字。

    ⚠️ **两个方向都要查**。只查「注册表里的项自己没问题」证明不了
    「没有漏项」—— 而漏项是这里唯一的真实风险：那个智能体拿到兜底提示词，
    它**不会报错**，只是专业约束（政策问答的「必须有出处」）悄悄消失。

    ⚠️ 反向查同样重要：``PROMPTS`` 里有已不存在的名字，是「有人改了枚举
    却忘了改提示词表」的信号。它当下无害，但会让下一个加智能体的人
    以为那个名字还在用。
    """
    registry = default_registry()
    declared = {name.value for name in AgentName}

    assert set(registry.names()) == declared


def test_default_registry_refuses_when_a_prompt_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ 有智能体没登记提示词时，**构造注册表就失败**。

    ⚠️ 这条比上一条更本质：上一条守的是「当前状态是对的」，
    这一条守的是「将来错了会有人喊」。少了它，漏登记提示词的后果是
    那个智能体静默拿到兜底提示词 —— 而兜底提示词是**故意写成**
    「一个正常的差旅助手」的（让用户察觉不到），所以没有任何迹象
    会提示你出错了。

    ⚠️ 报错信息里要有**具体是哪个**智能体。只说「配置不完整」的话，
    六个智能体要一个个去看。
    """
    import src.agents.registry as registry_module

    broken = dict(PROMPTS)
    broken.pop(AgentName.POLICY_RAG.value)
    monkeypatch.setattr(registry_module, "PROMPTS", broken)

    with pytest.raises(ValueError, match="policy_rag"):
        registry_module.default_registry()


def test_default_registry_refuses_a_stale_prompt_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★★ 提示词表里有已不存在的智能体名时，同样失败。

    ⚠️ 与上一条成对，缺一不可。只查前者的话，改名后遗留的旧条目会一直
    躺在表里 —— 而它会让人以为那个名字还在用。
    """
    import src.agents.registry as registry_module

    broken = dict(PROMPTS)
    broken["legacy_agent"] = "我是一个早就被删掉的智能体。"
    monkeypatch.setattr(registry_module, "PROMPTS", broken)

    with pytest.raises(ValueError, match="legacy_agent"):
        registry_module.default_registry()


def test_only_the_main_agent_is_user_facing() -> None:
    """★★ 只有主智能体标记为「面向用户」。

    ⚠️ 这个标记不是装饰：子智能体拿到的提示词里会带上一段
    「你不是直接面向用户的那一个」（见 :data:`MEMBER_HEADER`）。
    标错的后果是子智能体以为自己在跟用户说话，于是输出客服腔、
    或者直接给出「你可以在差旅平台上提交申请」这类**面向用户的操作指引**
    —— 而那段话会经由主智能体原样到达用户。

    ⚠️ 断言的是「恰好一个」而不是「主智能体是 True」：后者在
    「两个都标了 True」时同样通过。
    """
    registry = default_registry()
    facing = [item.name for item in registry.specs() if item.is_user_facing]

    assert facing == [AgentName.MAIN_PLAN.value], f"面向用户的智能体不止一个：{facing}"


def test_every_prompt_comes_from_the_prompt_table() -> None:
    """⚠️ 注册表里的提示词与 ``PROMPTS`` **逐字相同**。

    ⚠️ 注册表不做提示词的二次加工（不裁剪、不加前缀）—— 前缀是
    **模板**阶段才拼的（见 :func:`~src.agents.registry._render_template`）。
    在这里顺手加工一下，会让「改提示词」这件事在一个不起眼的地方失效。
    """
    for item in default_registry().specs():
        assert item.system_prompt == PROMPTS[item.name]


# ---------------------------------------------------------------------------
# 二、框架模板契约
# ---------------------------------------------------------------------------
def test_subagent_template_renders_with_the_framework_placeholders() -> None:
    """★★ 每个模板都能用框架提供的**五个**占位符渲染出来。

    ⚠️ 已核实（``agentscope/app/_tool/_agent_create.py:399-405``）：框架对模板做的
    唯一处理是 ``str.format``，且**只**提供
    ``team_name / team_description / member_name / member_description /
    leader_name``。

    多写一个 ``{xxx}`` 的后果不对称：``.format()`` 抛 ``KeyError``，
    而这个调用被包在 ``AgentCreate`` 的 broad ``try/except`` 里 ——
    最终以一句 ``AgentCreate failed: ...`` 的工具结果回到模型手上，
    再由模型转述给用户。也就是说：

        一个花括号的笔误 → 一个**运行时**、**面向模型**、**语焉不详**的失败

    这条用例把那个笔误提前到测试期。
    """
    probe = dict.fromkeys(PLACEHOLDERS, "示例")

    for template in build_subagent_templates():
        rendered = template.system_prompt_template.format(**probe)
        assert rendered.strip(), f"{template.type} 的模板渲染结果为空"
        # ⚠️ 渲染完不该再剩花括号 —— 剩下的会被模型当成字面内容看到。
        assert "{" not in rendered and "}" not in rendered, (
            f"{template.type} 的模板渲染后仍有花括号：{rendered[:200]}"
        )


def test_the_placeholder_list_matches_the_framework_call_site() -> None:
    """★★★ ``PLACEHOLDERS`` 与框架 ``.format(...)`` 的关键字**逐字相同**。

    ⚠️ 这是**唯一**能守住那份名单的用例，而它之所以必须存在，是因为
    上一条测试（渲染一遍）**证明不了名单是对的**：
    :func:`_render_template` 的探针是 ``dict.fromkeys(PLACEHOLDERS, ...)``
    —— **自己给自己供货**。名单里多一个框架根本不提供的键，
    探针照样把它填上，``.format()`` 照样成功，校验毫无反应。
    也就是说：名单漂了，渲染校验是**最后一个**会告诉你的人。

    ⚠️ 断言方式：去**读框架的源码**，把那句
    ``template.system_prompt_template.format(...)`` 的关键字参数抠出来比对。
    这是刻意选的重手段 —— 换成任何「跑一遍看看行不行」的写法都测不出来，
    因为跑一遍用的还是我们自己的名单。

    ⚠️ 两个方向都要查：

    - **我们多写了** → 探针会喂一个框架不给的键，模板里用了它就渲染不出来，
      而框架会在 ``AgentCreate`` 的 broad ``except`` 里把它吞成一句
      ``AgentCreate failed: ...``（面向模型、语焉不详）。
    - **我们少写了** → 我们的探针喂不出那个键，于是**我们**报
      ``ValueError``，而框架其实是能渲染的 —— 一次误报会让人去改一个
      本来没问题的提示词，或者干脆把这条校验删掉。

    ⚠️ 读不到框架源码时**直接失败**，不 skip：``agentscope`` 是
    ``requirements.txt`` 第 零 节钉死的 pip 依赖，它不在才是异常情况，
    而一条会自己消失的护栏等于没有护栏。

    调用点的定位用 ``agentscope.__file__``（当前解释器真正 import 的那份），
    而不是任何写死的路径 —— 换虚拟环境 / 换 Python 版本后仍然准确。
    """
    import re

    import agentscope

    call_site = (
        Path(agentscope.__file__).resolve().parent
        / "app"
        / "_tool"
        / "_agent_create.py"
    )
    assert call_site.is_file(), f"找不到框架的调用点：{call_site}"
    source = call_site.read_text(encoding="utf-8")

    match = re.search(
        r"system_prompt_template\.format\((?P<args>[^)]*)\)",
        source,
    )
    assert match, "框架里找不到 system_prompt_template.format(...) 调用点"

    provided = set(re.findall(r"(\w+)\s*=", match.group("args")))
    assert provided, f"抠不出关键字参数：{match.group('args')!r}"

    assert provided == set(PLACEHOLDERS), (
        f"PLACEHOLDERS 与框架的调用点不一致。\n"
        f"  框架提供：{sorted(provided)}\n"
        f"  我们写着：{sorted(PLACEHOLDERS)}\n"
        f"  我们多写了：{sorted(set(PLACEHOLDERS) - provided)}\n"
        f"  我们少写了：{sorted(provided - set(PLACEHOLDERS))}\n"
        f"⚠️ 渲染校验发现不了「多写」—— 它的探针就是这份名单本身。"
    )


def test_an_unknown_placeholder_fails_when_templates_are_built() -> None:
    """★★ 提示词里混进花括号时，**生成模板**就报错，且指出是哪个智能体。

    ⚠️ 这条守的是「校验发生在对的时刻」。校验若推迟到 ``AgentCreate``
    被调用的那一刻（也就是用户真的让系统建一个子智能体的时候），
    错误就变成了一个运行时的、面向模型的模糊失败（``AgentCreate
    failed: ...``）—— 而它本可以在启动装配时就被拦住。

    ⚠️ 断言报错信息里有智能体名。只说「模板渲染失败」的话，
    六个智能体要一个个去试。

    ⚠️ 顺带断言**别的智能体没被牵连**：校验是按 spec 逐个做的，
    一个坏模板不该让整个列表都生不出来 —— 否则报错信息里的名字
    反而成了「随机挑的一个」。
    """
    registry = AgentRegistry()
    registry.register(spec(name=AgentName.ORDER_QUERY.value))
    registry.register(
        spec(
            name=AgentName.POLICY_RAG.value,
            system_prompt="回答时请引用 {citation} 里的条款。",
        ),
    )

    with pytest.raises(ValueError, match="policy_rag"):
        registry.subagent_templates()

    # 反面：把坏的摘掉，同一个注册表立刻能生成模板。
    registry.register(
        spec(name=AgentName.POLICY_RAG.value, version=2, system_prompt="只答差旅政策。"),
    )
    assert [t.type for t in registry.subagent_templates()] == [
        AgentName.ORDER_QUERY.value,
        AgentName.POLICY_RAG.value,
    ]


def test_member_header_uses_only_framework_placeholders() -> None:
    """⚠️ 成员前缀里出现的占位符必须**全部**是框架提供的。

    ⚠️ 与上一条是两层：上一条查的是「拼接后的整串」，这条查的是
    「我们自己写的那一段」。分开查的价值在于：前缀是**我们**控制的样板，
    提示词是**别人**写的文案，出问题时定位方向完全不同。
    """
    import re

    used = set(re.findall(r"\{(\w+)\}", MEMBER_HEADER))
    assert used, "成员前缀里没有任何占位符，它就不是模板了"
    assert used <= set(PLACEHOLDERS), (
        f"成员前缀用了框架不提供的占位符：{sorted(used - set(PLACEHOLDERS))}"
    )


def test_member_header_puts_the_identity_first() -> None:
    """★★ 成员前缀必须排在基础提示词**之前**。

    ⚠️ 顺序反了会很隐蔽：基础提示词开头是「你是 AliGo 差旅助手」，
    模型先认定自己是主助手，再把后面的「你是团队成员」当成补充说明，
    于是它可能尝试去做只有主智能体能做的事 —— 直接给用户操作指引、
    承诺下单。这条不会让任何功能测试变红。
    """
    for template in build_subagent_templates():
        assert template.system_prompt_template.startswith(MEMBER_HEADER), \
            f"{template.type} 的模板没有以成员前缀开头"


def test_member_header_tells_the_agent_it_is_not_user_facing() -> None:
    """★ 前缀里必须说清楚「你的输出会由队长整理后回复用户」。"""
    assert "队长" in MEMBER_HEADER
    assert "用户" in MEMBER_HEADER


def test_subagent_template_types_are_unique_and_match_names() -> None:
    """⚠️ 模板类型唯一，且等于智能体名。

    ⚠️ ``create_app`` 对重复的 ``type`` 会抛 ``ValueError``
    （``agentscope/app/_app.py:375-386``）。这个错误发生在**应用构造期**，
    也就是 import 应用的时刻 —— 症状是「服务起不来」，而根因是
    「两个智能体重名」，中间隔着一整条调用链。
    """
    templates = build_subagent_templates()
    types = [template.type for template in templates]

    assert len(types) == len(set(types)), f"模板类型有重复：{types}"
    assert set(types) == {item.name for item in default_registry().specs()}


def test_templates_are_a_list_not_a_dict() -> None:
    """★★ 交给 ``create_app`` 的必须是 **list**，不是 dict。

    ⚠️ ``create_app`` 的公开参数是 ``list[SubAgentTemplate] | None``；
    dict 是它**内部**转出来的形态（``agentscope/app/_app.py:386``）。传 dict 过去，
    它会去迭代键（一堆字符串），然后在 ``t.type`` 上崩 ——
    报出来是 ``AttributeError: 'str' object has no attribute 'type'``，
    与「参数类型传错了」这个真实原因隔得很远。

    ⚠️ 同时断言它的返回值是**新的**列表：``create_app`` 会把列表存进
    ``app.state``（进程级共享），返回内部缓存的话，某个调用方「顺手
    排个序」就能改变整个进程的行为。
    """
    first = build_subagent_templates()
    second = build_subagent_templates()

    assert isinstance(first, list)
    assert first is not second, "两次调用返回了同一个列表对象"


def test_template_descriptions_are_written_for_the_leader_model() -> None:
    """★ 模板描述要写「什么时候该用我」，而不是内部实现。

    ⚠️ ``SubAgentTemplate.description`` 是**主智能体选型时看到的唯一依据**
    —— 它会变成 ``AgentCreate`` 工具 schema 里 ``subagent_type`` 枚举的
    说明文字。写成「我用了 Milvus 做检索」对选型毫无帮助，
    还会占掉本该描述适用场景的位置。
    """
    internal = ("Milvus", "FunctionTool", "MiddlewareBase", "src/", "AgentScope")

    for template in build_subagent_templates():
        assert template.description.strip(), f"{template.type} 没有描述"
        for token in internal:
            assert token not in template.description, \
                f"{template.type} 的描述里出现了内部实现 {token!r}"


# ---------------------------------------------------------------------------
# 三、版本语义
# ---------------------------------------------------------------------------
def test_registering_the_same_spec_twice_is_idempotent() -> None:
    """★ 完全相同的规格重复注册是**幂等**的，不报错。

    ⚠️ 这不是宽容，是必需的：``default_registry()`` 可能在 import 期被
    调用多次（测试里反复 import、或 reload）。「同样的东西注册两次就炸」
    会让测试的 import 顺序变成一个必须小心维护的东西 ——
    那种脆弱性迟早以「加了某个测试文件之后别的不相关的测试开始报错」
    的形式暴露出来。
    """
    registry = AgentRegistry()
    first = registry.register(spec(name="a", version=1))
    second = registry.register(spec(name="a", version=1))

    assert first == second
    assert len(registry) == 1


def test_same_version_with_different_content_is_rejected() -> None:
    """★★ 同名同版本但内容不同 → 报错。

    ⚠️ 守的是「谁后注册谁生效」这个不确定性。若允许它，
    系统行为就取决于 import 顺序 —— 而 import 顺序在不同入口
    （uvicorn、pytest、离线脚本）下是不一样的。症状是
    「跑测试时是一个智能体、跑生产时是另一个」，而两者的日志长得一样。
    """
    registry = AgentRegistry()
    registry.register(spec(name="a", version=1))

    with pytest.raises(ValueError, match="v1"):
        registry.register(spec(name="a", version=1, description="改了一句话"))


def test_a_lower_version_is_rejected() -> None:
    """★★ **降级注册**被拒绝。

    ⚠️ 这条看起来多余（谁会写个更低的版本号？），但它挡的是一个很具体
    的场景：**多个装配点各注册一次**。某个测试替身注册了 v1、
    生产装配注册的是 v2；谁后注册谁生效 —— 于是「生产上跑的是哪个版本」
    变成了一个没有答案的问题。

    ⚠️ 报错信息里要有当前版本与试图注册的版本，否则看不出是降级。
    """
    registry = AgentRegistry()
    registry.register(spec(name="a", version=2))

    with pytest.raises(ValueError, match="v1"):
        registry.register(spec(name="a", version=1))


def test_a_higher_version_replaces_the_spec() -> None:
    """★ 更高版本正常替换。"""
    registry = AgentRegistry()
    registry.register(spec(name="a", version=1, description="旧"))
    registry.register(spec(name="a", version=2, description="新"))

    assert registry.get("a").version == 2
    assert registry.get("a").description == "新"
    assert len(registry) == 1


def test_a_spec_needs_a_positive_version() -> None:
    """⚠️ 版本号必须是正整数。"""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="版本"):
            spec(name="a", version=bad)


# ---------------------------------------------------------------------------
# 四、查询与遍历
# ---------------------------------------------------------------------------
def test_unknown_agent_error_lists_the_known_names() -> None:
    """★★ 查不到时的报错要**列出已知的名字**。

    ⚠️ 一个打错的名字（``policy_rag`` 写成 ``policy_rags``）与一个
    「智能体真的没注册」是两件不同的事，但两者的报错都是
    ``KeyError: 'policy_rags'`` —— 而后者的排查要从「系统里到底有哪些
    智能体」开始。把已知名字列出来，一次就能分辨。

    ⚠️ 继承 ``KeyError`` 是刻意的：调用方大多在做「按名字取配置」，
    ``KeyError`` 是这类动作的惯用异常，已有的 ``except KeyError``
    兜底仍然生效。
    """
    registry = default_registry()

    with pytest.raises(UnknownAgentError) as excinfo:
        registry.get("no_such_agent")

    message = str(excinfo.value)
    assert "no_such_agent" in message
    for name in registry.names():
        assert name in message, f"报错信息里没有列出已知的 {name}"


def test_unknown_agent_error_is_a_key_error() -> None:
    """⚠️ 与上一条分开：**类型**必须是 ``KeyError`` 的子类。"""
    registry = default_registry()
    with pytest.raises(KeyError):
        registry.get("no_such_agent")


def test_maybe_get_returns_none_instead_of_raising() -> None:
    """★ ``maybe_get`` 取不到时返回 ``None``。

    ⚠️ 提供它是为了让调用方**不必**用 ``try/except KeyError`` 表达
    「可能没有」—— 那种写法的副作用是 ``try`` 块里其他代码抛的
    ``KeyError`` 也会被同一个 ``except`` 接住，于是真正的缺陷
    变成了「这个智能体没注册」。
    """
    registry = default_registry()

    assert registry.maybe_get("no_such_agent") is None
    assert registry.maybe_get(AgentName.MAIN_PLAN.value) is not None


def test_names_are_sorted() -> None:
    """⚠️ 名字按字典序返回，不按插入顺序。

    ⚠️ 调用方多用于展示（日志、报错信息、前端下拉）。稳定的顺序让两次
    运行的结果可比；插入顺序会随 import 顺序变化，那种「顺序变了」的
    diff 没有信息量。
    """
    names = default_registry().names()
    assert list(names) == sorted(names)


def test_registry_is_iterable_over_specs() -> None:
    """⚠️ 可以直接遍历规格，且遍历顺序与 ``names()`` 一致。"""
    registry = default_registry()
    iterated = [item.name for item in registry]
    assert iterated == list(registry.names())
    assert len(iterated) == len(registry)


def test_to_dict_does_not_leak_the_system_prompt() -> None:
    """★★ 对外序列化**不得**包含系统提示词。

    ⚠️ 提示词里含业务规则与边界描述（比如「超过 X 元要给审批人打电话」）
    —— 那是攻击者构造绕过输入时的现成地图。任何能调接口的人都能拿到它，
    等于把「系统怎么被指令的」公开。
    """
    payload = to_dict(default_registry())
    dumped = repr(payload)

    for prompt in PROMPTS.values():
        assert prompt[:40] not in dumped, "对外接口里泄漏了系统提示词"

    assert payload["count"] == len(default_registry())
    assert len(payload["agents"]) == payload["count"]


def test_to_dict_exposes_the_fields_the_frontend_needs() -> None:
    """⚠️ 序列化结果里要有前端与排障需要的字段。

    ⚠️ 逐个字段查而不是查「非空」：漏掉 ``version`` 的后果是
    「线上跑的是哪个版本的智能体」变成一个查不到的问题。
    """
    payload = to_dict(default_registry())
    for entry in payload["agents"]:
        assert set(entry) == {
            "name",
            "version",
            "description",
            "subagent_type",
            "is_user_facing",
        }


# ---------------------------------------------------------------------------
# 五、纯度
# ---------------------------------------------------------------------------
def test_agents_package_does_not_import_the_framework() -> None:
    """★★ 纯度探针：``import src.agents`` **不得**把 ``agentscope`` 拉进来。

    ⚠️ 必须在**子进程**里验证。在进程内断言 ``"agentscope" not in sys.modules``
    是无效的 —— ``conftest.py`` 早已把框架导入了，断言必然假通过。

    这条守的是 ``src/agents/__init__.py`` 的设计：``registry`` 与 ``intent``
    都 import 了框架，一旦有人把它们加进包的 ``__init__``，提示词的纯单测
    就会被迫依赖框架（装不上框架就跑不了，框架改了导入路径则整个测试文件
    在**收集阶段**就报错，连不相关的用例都跑不了）。

    ⚠️ 这个改动**不会让任何功能测试变红** —— 只有这条会。
    """
    import subprocess
    import sys

    probes = [
        "import sys; import src.agents; "
        "assert 'agentscope' not in sys.modules, 'src.agents 把框架拉进来了'; "
        "print('PURE')",
        "import sys; from src.agents import PROMPTS, prompt_for; "
        "assert 'agentscope' not in sys.modules, 'prompts 把框架拉进来了'; "
        "print('PURE')",
    ]
    for code in probes:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "PURE", result.stdout


def test_registry_module_stays_a_data_module() -> None:
    """★★ 注册表只**消费**框架的一个数据类，不把模型层/服务层拉进来。

    ⚠️ 这也必须在子进程里跑：进程内 ``sys.modules`` 早被 ``conftest.py``
    污染了。

    ⚠️ 断言的对象是 ``src.llm`` / ``src.server`` / ``src.storage``，
    **不是** ``agentscope.agent`` —— 实测 ``import agentscope.app`` 会顺带
    把 ``agentscope.agent`` 拉进来，断言它「没被导入」是一条永远为假的
    测试。这条踩过一次，写在这里免得下次再踩。

    真实的风险是另一回事：注册表在装配期被调用，一旦它开始**构造** ``Agent``
    （哪怕只是为了「验证一下能不能建起来」），就会顺带构造模型、连接池、
    工具集 —— 装配期的「顺手验证」于是变成启动成本，而且失败点从
    「配置错了」漂移成「模型连不上」，排查方向完全不同。
    """
    import subprocess
    import sys

    code = (
        "import sys; import src.agents.registry as r; "
        "assert 'src.llm' not in sys.modules, '注册表拉了模型层'; "
        "assert 'src.server' not in sys.modules, '注册表拉了服务层'; "
        "assert 'src.storage' not in sys.modules, '注册表拉了存储层'; "
        "assert r.SubAgentTemplate.__module__.startswith('agentscope'), '模板类型不是框架的'; "
        "print('OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OK", result.stdout
