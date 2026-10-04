# -*- coding: utf-8 -*-
"""**智能体注册表** —— 名字、版本、提示词、以及给框架的子智能体模板。

文件职责：
    把「本项目有哪些智能体」这件事集中成一份**可校验**的数据，并据此生成
    框架需要的 ``SubAgentTemplate`` 列表。对应 P3 计划里的「版本化注册表」。

上下游依赖：
    - 上游：:mod:`src.agents.prompts`（提示词正文）、:mod:`src.domain`
      （:class:`~src.domain.enums.AgentName` 名册）、
      ``agentscope.app.SubAgentTemplate``（框架类型）。
    - 下游：``src/server/agents_factory.py`` 取 ``subagent_templates()``
      传给 ``create_app(custom_subagent_templates=...)``。

═══ 为什么需要一张表，而不是在装配处直接写 ═══

「有哪些智能体」这个事实被至少四处消费：提示词表、意图到智能体的路由表
（``src/orchestration/classifier.py``）、思考链的显示名、框架的子智能体模板。
四处各写一份的后果不是「重复」，而是**漂移**：加了第六个智能体，四处里
改了三处，剩下那处不报错，只是那个智能体永远拿不到提示词或永远不被调度。
把名册收敛到一处、其余三处都**遍历**它，漂移就变成了不可能。

═══ ⚠️ 版本号守的是「降级」而不是「升级」 ═══

:class:`AgentSpec` 带 ``version``，注册同名但版本更低的 spec 会直接
``ValueError``。这条规则看起来多余（谁会写个更低的版本号？），但它挡的是
一个很具体的场景：**多个装配点各注册一次**。比如某个测试替身注册了
``policy_rag`` 的 v1，而生产装配注册的是 v2；谁后注册谁生效，
于是「跑测试时是一个智能体、跑生产时是另一个」，而两者的日志长得一样。

允许同版本重复注册**完全相同的** spec（幂等），因为模块级装配在
import 时可能会被执行多次（测试里反复 import、或 reload）。这一条是
为了让幂等路径不必报错，而不是为了让「内容不同也算重复」蒙混过关。

═══ ⚠️ ``system_prompt_template`` 只有五个占位符 ═══

已核实（``app/_tool/_agent_create.py:399-405``）：框架对模板做的唯一处理是
``str.format(team_name=..., team_description=..., member_name=...,
member_description=..., leader_name=...)``。

代价不对称：多写一个 ``{xxx}``，``.format()`` 会抛 ``KeyError`` ——
而这个调用被包在 ``AgentCreate`` 的 broad ``try/except`` 里，最终以一句
``AgentCreate failed: ...`` 的工具结果回到模型手上。也就是说：

    一个花括号的笔误 → 一个**运行时**、**面向模型**、**语焉不详**的失败。

所以 :func:`_render_template` 在**生成模板时**就把模板完整渲染一遍，
把这类错误提前到启动期，并且报出是哪个智能体的模板、哪个占位符有问题。

⚠️ 校验的时机是 :meth:`AgentRegistry.subagent_templates`，**不是**
``register``。注册表允许装着一份暂时渲染不出来的 spec（比如某个装配点
想先注册、稍后用更高版本覆盖），校验留到「真的要交给框架」那一刻。

⚠️ 这也是为什么 :data:`MEMBER_HEADER` 与基础提示词是**拼接**关系而不是
嵌套关系：基础提示词正文里若出现花括号，渲染同样会炸。拼接处的校验见
:func:`_render_template`。
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from agentscope.app import SubAgentTemplate

from src.agents.prompts import PROMPTS
from src.domain import AgentName

#: 本模块的日志器。
logger = logging.getLogger(__name__)

#: 框架渲染 ``system_prompt_template`` 时提供的**全部**占位符。
#:
#: ⚠️ 这份名单是从框架的 ``.format()`` 调用点抄下来的
#: （``app/_tool/_agent_create.py:399-405``），不是我们的约定。
#: 少写一个 → 框架渲染时 ``KeyError``，被它吞成一个内容不透明的
#: ``AgentCreate failed: ...`` 工具结果（症状是「子智能体建不出来」，
#: 但日志里只有一句笼统的失败）。
#:
#: ⚠️ **多写一个我们测不出来**，这一点很容易想反：
#: :func:`_render_template` 的「渲染一次就是校验」用的探针是
#: ``dict.fromkeys(PLACEHOLDERS, ...)`` —— **自己给自己供货**。
#: 名单里多一个键，探针照样把它填上，``.format()`` 照样成功，
#: 校验一点反应都没有。
#:
#: 真正守住这份名单的是 ``tests/test_agents_registry.py`` 里那条
#: **直接去读框架调用点**的用例（把 ``.format(...)`` 的关键字参数抠出来
#: 与本名单比对）。所以改这份名单时，那条用例才是判据，不是这里的注释。
PLACEHOLDERS: tuple[str, ...] = (
    "team_name",
    "team_description",
    "member_name",
    "member_description",
    "leader_name",
)

#: 团队成员提示词的**前缀**。
#:
#: ⚠️ 它必须排在基础提示词**之前**。worker 拿到的 system prompt 是这一段
#: 加上基础提示词；把基础提示词（「你是 AliGo 差旅助手…」）放前面、
#: 这段放后面，模型会先认定自己是「主助手」，再把后面的身份说明当成补充，
#: 于是它可能尝试去做只有主智能体能做的事（比如直接回复用户）。
#: 身份说明必须是最先看到的。
MEMBER_HEADER = """\
你是「{member_name}」，团队「{team_name}」的成员，队长是 {leader_name}。
团队目标：{team_description}
你的职责：{member_description}

你不是直接面向用户的那一个：你的结论会由队长整理后再回复用户，
所以输出要**结构化、有依据、不含寒暄**。

"""


class UnknownAgentError(KeyError):
    """查了一个未注册的智能体名。

    ⚠️ 继承 ``KeyError`` 而不是自定义 ``Exception``：调用方（路由、思考链）
    大多是在做「按名字取配置」这件事，``KeyError`` 是这类动作的惯用异常；
    继承它意味着已有的 ``except KeyError`` 兜底仍然生效，不会因为引入本类
    而让某处的兜底静默失效。

    ⚠️ ``__str__`` 覆写成一句**可读的**话。``KeyError.__str__`` 会给参数
    加一对引号（``'policy_rag'``），在日志里看着像「引号是值的一部分」；
    更重要的是这里要顺便把**已知的名字**列出来 —— 打错名字时，看到
    「已知有：a、b、c」比看到一句「没有这个键」有用得多。
    """

    def __init__(self, name: str, known: Iterable[str]) -> None:
        """初始化。

        Args:
            name (`str`): 被查询的名字。
            known (`Iterable[str]`): 当前已注册的名字。
        """
        self.name = name
        self.known = tuple(sorted(known))
        super().__init__(name)

    def __str__(self) -> str:
        """返回可读的报错信息。

        Returns:
            `str`: 形如 ``未注册的智能体 'xxx'；已注册：a、b``。
        """
        listed = "、".join(self.known) if self.known else "（一个都没有）"
        return f"未注册的智能体 {self.name!r}；已注册：{listed}"


@dataclass(frozen=True)
class AgentSpec:
    """一个智能体的**全部静态定义**。

    ⚠️ 冻结（``frozen=True``）：注册表里的 spec 会被多个消费者共享
    （装配、提示词、思考链显示名）。可变的话，某个消费者「顺手改一下」
    就会影响其他消费者，而且是在**另一个**请求里才显形。

    Attributes:
        name: 智能体的规范名，取自 :class:`~src.domain.enums.AgentName`。
        version: 版本号，正整数。
        description: 一句话职责说明。⚠️ **会进模型上下文** ——
            ``SubAgentTemplate.description`` 是主智能体选型时的依据
            （``AgentCreate`` 的 ``subagent_type`` 枚举说明来自它）。
            所以要写成「什么时候该用我」，而不是内部实现说明。
        system_prompt: 渲染好的基础提示词（不含团队成员前缀）。
        subagent_type: 注册给框架的模板类型名。默认等于 ``name``。
        is_user_facing: 是否直接面向用户。⚠️ 只有主智能体是 ``True``：
            它决定「谁的话会被用户直接看到」，而子智能体的输出要先回到
            主智能体整理 —— 见 :data:`MEMBER_HEADER` 的说明。
    """

    name: str
    version: int
    description: str
    system_prompt: str
    subagent_type: str = ""
    is_user_facing: bool = False

    def __post_init__(self) -> None:
        """校验并补全默认值。

        ⚠️ 用 ``object.__setattr__`` 而不是去掉 ``frozen``：冻结是本类的
        核心保证（见类文档），而 ``subagent_type`` 的默认值推导必须在
        初始化时做一次。``dataclasses`` 的官方做法就是这一句。

        Raises:
            ValueError: 名字为空、版本小于 1、或提示词为空。
        """
        if not self.name.strip():
            raise ValueError("智能体名不能为空")
        if self.version < 1:
            raise ValueError(f"智能体 {self.name!r} 的版本号必须为正整数")
        if not self.system_prompt.strip():
            raise ValueError(f"智能体 {self.name!r} 的提示词不能为空")
        if not self.subagent_type:
            object.__setattr__(self, "subagent_type", self.name)


class AgentRegistry:
    """智能体注册表：名字 → 规格，并派生框架子智能体模板。

    ⚠️ **不是线程安全的，也不打算做成线程安全的**。它的构造发生在应用
    装配期（``src/server/app.py`` 的 import 阶段），之后只读。给它加锁会
    给人一种「运行期可以改」的错觉，而运行期改注册表意味着「同一个进程里
    前后两个请求用了不同的智能体定义」—— 那是不该被支持的用法。

    Attributes:
        _specs (`dict[str, AgentSpec]`): 名字 → 当前生效的规格（最高版本）。
    """

    def __init__(self, specs: Iterable[AgentSpec] = ()) -> None:
        """初始化。

        Args:
            specs (`Iterable[AgentSpec]`): 初始规格。
        """
        self._specs: dict[str, AgentSpec] = {}
        for spec in specs:
            self.register(spec)

    # --------------------------------------------------------------------------
    # 写入
    # --------------------------------------------------------------------------
    def register(self, spec: AgentSpec) -> AgentSpec:
        """注册一个智能体规格。

        ⚠️ 重复注册**完全相同**的 spec 是幂等的（不报错、返回已有的那份）。
        这条不是宽容，是必需的：``default_registry()`` 可能在 import 期被
        调用多次，而「同样的东西注册两次就炸」会让测试收集阶段的 import
        顺序变成一个必须小心维护的东西 —— 那种脆弱性迟早会以
        「加了某个测试文件之后别的不相关的测试开始报错」的形式暴露出来。

        Args:
            spec (`AgentSpec`): 待注册的规格。

        Returns:
            `AgentSpec`: 注册后生效的那一份（可能是已有的同版本 spec）。

        Raises:
            ValueError: 同名同版本但内容不同，或版本比现有的更低。
        """
        current = self._specs.get(spec.name)

        if current is None:
            self._specs[spec.name] = spec
            return spec

        if current.version == spec.version:
            if current == spec:
                # 幂等：内容一字不差，直接返回已有的。
                return current
            raise ValueError(
                f"智能体 {spec.name!r} 的 v{spec.version} 已经注册过，"
                f"但这次的定义与已有的不同。同名同版本必须是同一份定义 —— "
                f"否则「谁后注册谁生效」会让系统行为取决于 import 顺序。",
            )

        if current.version > spec.version:
            raise ValueError(
                f"智能体 {spec.name!r} 当前是 v{current.version}，"
                f"不允许注册更低的 v{spec.version}。降级注册几乎总是"
                f"「某个装配点忘了改版本号」，而它的症状是行为取决于注册顺序。",
            )

        self._specs[spec.name] = spec
        logger.info(
            "智能体 %r 从 v%d 升级到 v%d。",
            spec.name,
            current.version,
            spec.version,
        )
        return spec

    # --------------------------------------------------------------------------
    # 读取
    # --------------------------------------------------------------------------
    def get(self, name: str) -> AgentSpec:
        """按名字取规格。

        Args:
            name (`str`): 智能体名。

        Returns:
            `AgentSpec`: 对应的规格。

        Raises:
            UnknownAgentError: 名字未注册。
        """
        spec = self._specs.get(name)
        if spec is None:
            raise UnknownAgentError(name, self._specs)
        return spec

    def maybe_get(self, name: str) -> AgentSpec | None:
        """按名字取规格，取不到返回 ``None``。

        ⚠️ 提供这个方法是为了让调用方**不必**用 ``try/except KeyError``
        来表达「可能没有」。那种写法的副作用是：``try`` 块里的其他代码
        抛出的 ``KeyError`` 也会被同一个 ``except`` 接住，
        于是真正的缺陷变成了「这个智能体没注册」。

        Args:
            name (`str`): 智能体名。

        Returns:
            `AgentSpec | None`: 规格；未注册时为 ``None``。
        """
        return self._specs.get(name)

    def names(self) -> tuple[str, ...]:
        """列出全部已注册的名字（排序后）。

        ⚠️ 排序而不是保持插入顺序：调用方多用于展示（日志、报错信息、
        前端下拉），稳定的顺序让两次运行的结果可比。插入顺序会随
        import 顺序变化，那种「顺序变了」的 diff 没有信息量。

        Returns:
            `tuple[str, ...]`: 名字元组。
        """
        return tuple(sorted(self._specs))

    def specs(self) -> tuple[AgentSpec, ...]:
        """列出全部规格，按名字排序。

        Returns:
            `tuple[AgentSpec, ...]`: 规格元组。
        """
        return tuple(self._specs[name] for name in self.names())

    def __len__(self) -> int:
        """已注册的智能体数量。

        Returns:
            `int`: 数量。
        """
        return len(self._specs)

    def __iter__(self) -> Iterator[AgentSpec]:
        """遍历全部规格（按名字排序）。

        Returns:
            `Iterator[AgentSpec]`: 迭代器。
        """
        return iter(self.specs())

    # --------------------------------------------------------------------------
    # 派生：框架子智能体模板
    # --------------------------------------------------------------------------
    def subagent_templates(self) -> list[SubAgentTemplate]:
        """生成交给 ``create_app(custom_subagent_templates=...)`` 的模板列表。

        ⚠️ 返回的是 **list** 而不是 dict。``create_app`` 的公开参数是
        ``list[SubAgentTemplate] | None``；dict 是它**内部**转出来的形态
        （``app/_app.py:386``）。传 dict 过去，它会去迭代键（一堆字符串），
        然后在 ``t.type`` 上崩掉 —— 这个错误发生在装配期，
        但报出来的是 ``AttributeError: 'str' object has no attribute 'type'``，
        与「参数类型传错了」这个真实原因隔得很远。

        ⚠️ 顺带守住数量上限：``create_app`` 只校验 ``type`` 唯一性，
        **没有数量上限**。模板全部会变成 ``AgentCreate`` 工具 schema 里的
        一个枚举值，进每一次模型调用。智能体多到几十个时那是可观的开销 ——
        但更重要的是：枚举越长，模型选错的概率越高。真到那时候应该先分域，
        而不是继续往这张表里加。

        Returns:
            `list[SubAgentTemplate]`: 每个已注册智能体一个模板。
        """
        return [
            SubAgentTemplate(
                type=spec.subagent_type,
                description=spec.description,
                system_prompt_template=_render_template(spec),
            )
            for spec in self.specs()
        ]


def _render_template(spec: AgentSpec) -> str:
    """把规格渲染成框架的模板串，并**当场校验**它渲染得出来。

    ⚠️ 渲染一次**就是**校验：模板里若有框架未提供的占位符，这一步会抛
    ``KeyError``。提前到装配期抛，好处见模块文档。
    ⚠️ 但校验完要返回**模板串**（含花括号），不是渲染结果 —— 花括号要留给
    框架在真正创建 worker 时填。

    Args:
        spec (`AgentSpec`): 智能体规格。

    Returns:
        `str`: ``成员前缀 + 基础提示词``。

    Raises:
        ValueError: 模板渲染失败（多半是提示词里混进了花括号）。
    """
    template = MEMBER_HEADER + spec.system_prompt
    probe = dict.fromkeys(PLACEHOLDERS, "示例")
    try:
        template.format(**probe)
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError(
            f"智能体 {spec.name!r} 的提示词模板无法渲染（{type(exc).__name__}: {exc}）。"
            f"框架只提供这五个占位符：{'、'.join(PLACEHOLDERS)}；"
            f"提示词正文里出现花括号同样会导致渲染失败。",
        ) from exc
    return template


# ==============================================================================
# 默认注册表
# ==============================================================================
#: 各智能体的当前版本。
#:
#: ⚠️ 集中在一处而不是写在每个 ``AgentSpec`` 里：改版本号是一个**整体**
#: 动作（一次提示词大改通常涉及多个智能体），分散写的话很容易只改了几个，
#: 于是「哪些智能体是 v2」这件事就没有单一答案了。
_VERSION = 1

#: 各智能体的职责说明。
#:
#: ⚠️ 这些文字会进 ``SubAgentTemplate.description``，而它是**主智能体选型
#: 时看到的唯一依据**（``AgentCreate`` 的 ``subagent_type`` 枚举说明）。
#: 所以要写「什么时候该用我」，不要写「我内部用了什么技术」——
#: 后者对选型毫无帮助，还会占掉本该用来描述适用场景的位置。
_DESCRIPTIONS: dict[str, str] = {
    AgentName.MAIN_PLAN.value: "主规划智能体：面向用户，收集出差要素、生成行程方案、协调其他智能体。",
    AgentName.INTENT.value: "意图识别：把用户一句话拆成结构化意图与要素，不直接回复用户。",
    AgentName.POLICY_RAG.value: "政策问答：依据公司差旅制度回答报销标准与规定，答案必须带出处。",
    AgentName.APPROVAL.value: "出差申请：把确定的行程转成申请单并提交审批，提交前需用户确认。",
    AgentName.ORDER_QUERY.value: "订单查询：查已有订单与出差申请单的进度、详情与可取消性。",
}


def default_registry() -> AgentRegistry:
    """构造本项目的默认注册表。

    ⚠️ 遍历 :class:`~src.domain.enums.AgentName` 而不是遍历 ``PROMPTS``。
    遍历字典只能证明「表里的项都有提示词」，证明不了「名册里的智能体都在表里」
    —— 而漏项是这里唯一的真实风险，且后果是那个智能体拿到兜底提示词，
    它**不会报错**，只是专业约束悄悄消失。两个方向都查，见下面的断言。

    Returns:
        `AgentRegistry`: 含全部 :class:`AgentName` 成员。

    Raises:
        ValueError: ``AgentName`` 有成员没有登记提示词，或提示词表里有
            已经不存在的智能体名。
    """
    declared = {name.value for name in AgentName}
    missing = declared - set(PROMPTS)
    if missing:
        raise ValueError(
            f"这些智能体没有登记提示词：{'、'.join(sorted(missing))}。"
            f"请检查 src/agents/prompts.py 的 PROMPTS。",
        )
    stale = set(PROMPTS) - declared
    if stale:
        raise ValueError(
            f"PROMPTS 里有已不存在的智能体名：{'、'.join(sorted(stale))}。"
            f"多半是改名后忘了清理。",
        )

    registry = AgentRegistry()
    for name in AgentName:
        registry.register(
            AgentSpec(
                name=name.value,
                version=_VERSION,
                description=_DESCRIPTIONS[name.value],
                system_prompt=PROMPTS[name.value],
                is_user_facing=name is AgentName.MAIN_PLAN,
            ),
        )
    return registry


def build_subagent_templates() -> list[SubAgentTemplate]:
    """便利函数：直接拿到默认注册表的模板列表。

    ⚠️ 每次调用都**重新构造**注册表。代价可以忽略（几次对象构造），
    换来的是调用方不可能拿到一份被别人改过的注册表 —— 而
    ``create_app(custom_subagent_templates=...)`` 拿到的列表会被框架存进
    ``app.state``，那是**进程级共享**的。

    Returns:
        `list[SubAgentTemplate]`: 框架可用的子智能体模板列表。
    """
    return default_registry().subagent_templates()


def to_dict(registry: AgentRegistry) -> dict[str, Any]:
    """把注册表拍平成可序列化的字典（供 ``/api/v1/agents`` 之类的接口用）。

    ⚠️ **不含** ``system_prompt``。提示词是内部实现，对外暴露它等于把
    「系统怎么被指令的」公开给任何能调接口的人 —— 提示词里往往含有
    业务规则与边界描述，那是攻击者构造绕过输入时的现成地图。

    Args:
        registry (`AgentRegistry`): 注册表。

    Returns:
        `dict[str, Any]`: ``{"agents": [...], "count": n}``。
    """
    return {
        "count": len(registry),
        "agents": [
            {
                "name": spec.name,
                "version": spec.version,
                "description": spec.description,
                "subagent_type": spec.subagent_type,
                "is_user_facing": spec.is_user_facing,
            }
            for spec in registry.specs()
        ],
    }


__all__ = [
    "MEMBER_HEADER",
    "PLACEHOLDERS",
    "AgentRegistry",
    "AgentSpec",
    "UnknownAgentError",
    "build_subagent_templates",
    "default_registry",
    "to_dict",
]
