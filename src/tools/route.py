# -*- coding: utf-8 -*-
"""``aligo_route_intent`` —— 快车道把路由决策交给 agent 的那个工具。

文件职责：
    接收 :class:`~src.orchestration.lane.LaneRouterMiddleware` 合成出来的
    路由决策，校验它，并把「系统已经知道用户想干什么」这件事**显式地**
    告诉模型。

上下游依赖：
    - 上游：:mod:`src.domain.enums`（:class:`Intent` / :class:`AgentName`）、
      :mod:`src.tools._result`。
    - 下游：``src/orchestration/lane.py``（按 :data:`ROUTE_TOOL_INPUT_FIELDS`
      拼入参）、``web/`` 前端（``route_decision`` 卡片）。

═══ 这个工具为什么存在 ═══

快车道省掉的是「模型自己判断该调哪个工具」这一步。但省掉之后有个副作用：
**模型不知道系统已经替它做了决定**。它只看到一条「某个工具被调用了、返回了
这样一段文字」的记录。

若不把这个决策明确告诉模型，它会看到用户说「规划行程」却没有任何工具被
调用的**意图线索**，于是可能重新推理一遍「用户想干嘛」，甚至反问用户。
那快车道省下的那次调用就白省了。

所以本工具的核心产出是**一句给模型看的中文说明**：用户要干什么、
系统已经做了什么、接下来该做什么。

═══ ⚠️ 入参字段是与 ``lane.py`` 的硬契约 ═══

形参名必须与 :data:`src.orchestration.lane.ROUTE_TOOL_INPUT_FIELDS` **逐字
一致**，工具名必须与 :data:`~src.orchestration.lane.DEFAULT_ROUTE_TOOL`
一致。

对不上的后果不是报错，而是**静默失效**：框架把
``got an unexpected keyword argument`` 当成一次普通的工具失败结果塞回给
模型（实测以 ``TOOL_RESULT_TEXT_DELTA`` 出现，不抛异常），于是快车道
看起来跑通了，路由信息却一个字都没传进来。

``tests/test_tools_route_contract.py`` 用一条断言把两边钉在一起。
"""

from __future__ import annotations

from agentscope.tool import ToolChunk

from src.domain.enums import AgentName, Intent
from src.tools._result import CARD_ROUTE, error_chunk, ok_chunk

#: 工具名。⚠️ 必须与 ``lane.DEFAULT_ROUTE_TOOL`` 一致，见模块文档。
ROUTE_TOOL_NAME = "aligo_route_intent"

#: 每种意图 → 告诉模型「下一步该做什么」。
#:
#: ⚠️ 这张表是**给模型的行为指引**，不是业务规则。业务规则在
#: :mod:`src.domain.rules`。放在这里是因为它描述的是「模型该怎么做」，
#: 而这个问题的答案随 prompt 与产品形态变化，不该污染领域层。
#:
#: ⚠️ 每个 :class:`Intent` 成员都必须有登记，由
#: ``tests/test_tools_route_contract.py`` 的遍历用例守护。漏掉的后果是
#: 模型收到一句空指引，行为退化成「自己看着办」。
_NEXT_STEP_HINTS: dict[Intent, str] = {
    Intent.PLAN_TRIP: "用户要规划一次出差。请检查还缺哪些要素，缺则追问，齐全则给出方案。",
    Intent.APPLY_APPROVAL: "用户要提交出差申请。请确认申请事由与预估金额后走审批流程。",
    # ⚠️ 这条指引到达模型时，**检索还没发生** —— 路由工具是每轮第一个被调用的，
    # 它不取任何制度数据。早先的措辞是「请依据制度检索结果回答」，等于告诉模型
    # 「结果已经在那儿了」，于是模型会回一句「已让政策问答智能体检索制度原文，
    # 稍等。」然后结束回合，用户拿到一句空话（实测 8 轮里出现过）。
    # 所以这里必须写成**祈使句**：先拿到数据，再开口。
    Intent.QUERY_POLICY: (
        "用户想了解差旅标准。请先调用 check_travel_policy 拿到标准数值"
        "（问的是「标准是多少」时**不要传价格**，工具会直接返回上限与依据），"
        "再作答并注明依据；"
        "**在拿到结果之前不要回答，也不要说「已让某某去查」「请稍等」**。"
        "标准里有没有城市分档、职级差异，**以工具的返回为准** —— "
        "工具没说有，就不要替它补一张分档表。"
    ),
    Intent.QUERY_ORDER: "用户要查询自己的订单或申请单。请调用查询工具取真实数据，不要凭记忆作答。",
    Intent.MODIFY_TRIP: "用户要修改已有安排。请先确认改哪一项，再重新走收集流程。",
    Intent.CANCEL: "用户要取消。请先确认取消对象，再走取消流程（取消可能产生费用，务必说明）。",
    Intent.CHITCHAT: "用户只是寒暄或闲聊。请简短回应，不要主动推销功能。",
    Intent.OTHER: "用户意图不明确。请用一句话复述你的理解并向用户确认，不要贸然动作。",
}


def aligo_route_intent(
    intent: str,
    matched_rule: str = "",
    target_agents: list[str] | None = None,
    reason: str = "",
) -> ToolChunk:
    """记录并确认一条**由系统路由规则**给出的意图判定。

    ⚠️ 这个工具通常**由系统自动调用**，不由你（模型）决定。当它出现时，
    说明路由规则已经根据用户的输入（多为一键按钮）确定了意图，你不需要
    再重新判断用户想干什么，只需按返回的「下一步」执行。

    Args:
        intent (`str`): 意图标识，取值来自 :class:`~src.domain.enums.Intent`
            （如 ``PLAN_TRIP``）。
        matched_rule (`str`): 命中的快车道规则名，用于排障。
        target_agents (`list[str] | None`): 应当接手的智能体列表。
        reason (`str`): 命中该规则的依据。

    Returns:
        `ToolChunk`: 含 ``route_decision`` 卡片与面向模型的中文指引。

    ⚠️ 意图值非法时返回 :func:`~src.tools._result.error_chunk` 而不是抛异常。
    抛出的异常会被框架吞成一段英文错误文本（见 :mod:`src.tools._result`
    的说明），模型只能看到 ``ValueError: 'FOO' is not a valid Intent`` ——
    而返回中文说明能让模型知道「路由信息坏了，请自己判断」，
    这对用户是可恢复的。
    """
    try:
        parsed = Intent(intent)
    except ValueError:
        return error_chunk(
            f"路由信息有误（无法识别的意图「{intent}」），请自行判断用户意图后再作答。",
            detail=f"unknown intent={intent!r}",
        )

    agents = [a for a in (target_agents or []) if a]
    # ⚠️ 未知的智能体名**只记日志不报错**。智能体注册表是随装配变化的
    # （见 ``src/agents/registry.py``），而路由表是静态的；两者短暂不一致
    # 时，把它当成致命错误会让一次正常的点击变成报错。
    unknown = [a for a in agents if a not in set(AgentName)]
    if unknown:
        import logging

        logging.getLogger(__name__).warning(
            "路由目标里有未注册的智能体：%s（已忽略，不影响本轮回复）",
            unknown,
        )

    hint = _NEXT_STEP_HINTS.get(parsed, _NEXT_STEP_HINTS[Intent.OTHER])
    # ⚠️ ``summary`` 里**只放「发生了什么」**，一个字都不放 ``hint``。
    #
    # 早先的写法是 ``f"...。{hint}"``，理由是「模型更容易注意 summary」。
    # 2026-10-03 的对抗审计指出这条理由买错了东西：``summary`` 按
    # :mod:`src.tools._result` 的契约是「给**模型和用户**看的自然语言」，
    # 而 ``hint`` 是纯模型的第二人称祈使句（带 ``check_travel_policy``
    # 这种工具名与 ``**`` 加粗标记）。两处实测症状：
    #   1. 模型把 summary 直接抄进正文，用户看到「请先调用
    #      check_travel_policy 拿到标准数值」—— 回复守卫为此专门加了
    #      ``_META_COMMENTARY_MARKERS`` 里的「让政策问答」「去检索制度原文」
    #      两条补丁来兜；
    #   2. 卡片数据源 ``items`` 也带过一份，前端把它渲染成了「下一步」。
    # 正确的分工是：**指引只走顶层 ``instruction``**（它存在的唯一目的
    # 就是给模型看），``summary`` 与 ``items`` 都只描述事实。
    # 补丁（守卫关键词）与病根（指引混进用户可见字段）同时存在时，
    # 修病根，别再加关键词。
    # ⚠️ 不再往 summary 里塞一句「请按 instruction 继续」之类的指路话：
    # 那仍然是写给模型看的元话语，模型照样会抄进正文（用户于是看到
    # 「顶层 instruction 字段」这种天书）。指引在顶层字段里，
    # 模型读得到 —— 靠的是字段本身，不是靠 summary 里再喊一遍。
    summary = (
        f"已由路由规则判定用户意图为「{parsed.display_name}」"
        f"（{parsed.value}，命中规则：{matched_rule or '无'}）。"
    )

    return ok_chunk(
        summary,
        card=CARD_ROUTE,
        items=[
            {
                "intent": parsed.value,
                "intent_display": parsed.display_name,
                "matched_rule": matched_rule,
                "target_agents": agents,
                "reason": reason,
            },
        ],
        # 这三个字段是给**模型**读的顶层提示，不进卡片。
        # ⚠️ 与 items 里的内容有意重复：模型更容易注意顶层字段，
        # 而前端只用 items。重复一点点 token，换模型行为的确定性。
        # ⚠️ ``instruction``（= ``_NEXT_STEP_HINTS`` 的值）是**给模型的行为
        # 指引**（「请调用 search_transport 查交通」这种第二人称祈使句），
        # 只能留在顶层。2026-10-03 的审计发现它曾被额外塞进 ``items``
        # 的 ``next_step`` 字段，而前端把 items 当卡片数据渲染 ——
        # 于是用户看到一句对他的助手说的指令。**别再往 items 里放它**：
        # items 里的一切都会被渲染成用户可见的卡片字段。
        routed_by="system_rule",
        user_intent=parsed.value,
        instruction=hint,
    )


__all__ = [
    "ROUTE_TOOL_NAME",
    "aligo_route_intent",
]
