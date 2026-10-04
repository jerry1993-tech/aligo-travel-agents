# -*- coding: utf-8 -*-
"""基础系统提示词（``src/agents/prompts.py``）的测试。

═══ 为什么要给「一堆字符串」写测试 ═══

提示词看起来是纯文案，改错了顶多「说得不好听」。实际上它有四类**功能性**
缺陷，每一类都不会报错，只会让系统悄悄变差：

1. **开发口吻泄漏** —— ``⚠️``、``src/tools/travel.py``、``TODO`` 这些会
   原样进模型上下文。模型要么照着复述给用户，要么被内部路径干扰而跑偏。
2. **写死了具体数字** —— 差标是**按用户**配的，价格是实时查的。提示词里
   写「酒店不超过 600 元」，模型就会跳过工具调用直接引用它，而对一个
   差标 1200 元的用户，这句话是错的且毫无痕迹。
3. **漏了某个智能体** —— 那个智能体会拿到空提示词，以「没有身份设定」的
   状态运行，表现无法预测。
4. **兜底提示词泄漏了异常状态** —— 「我似乎没有正确的配置」这种话，
   用户看到只会困惑。

本文件守住这四类。
"""

from __future__ import annotations

import re

import pytest

from src.domain import AgentName, Intent
from src.agents.prompts import (
    MAIN_AGENT_NAME,
    PROMPTS,
    prompt_for,
)

#: 不许出现在提示词里的开发口吻标记。
#:
#: ⚠️ 与 ``test_orchestration_prompt.py`` 的同类断言保持一致。两处都要有：
#: 那一处守的是**动态**段落（运行时拼出来的），这一处守的是**静态**提示词。
#: 拼接链上任何一段泄漏，结果都一样糟。
DEVELOPER_MARKERS = (
    "⚠️",
    "src/",
    "tests/",
    "TODO",
    "FIXME",
    "XXX",
    "docstring",
    "pytest",
    "import ",
)

#: 所有提示词（含兜底）的取值来源，供全量遍历。
#:
#: ⚠️ 把兜底也纳进来：它是**唯一**一段可能永远不被人工审阅的提示词 ——
#: 只在配置出错时才生效，而那时没人会去看它写了什么。
ALL_PROMPTS = {**PROMPTS, "<fallback>": prompt_for("__never_registered__")}


# ---------------------------------------------------------------------------
# 一、覆盖完整性
# ---------------------------------------------------------------------------
def test_every_agent_name_has_a_prompt() -> None:
    """★★ 每个 :class:`AgentName` 成员都必须有登记。

    ⚠️ 遍历**枚举**而不是遍历字典。遍历字典只能证明「表里的项自己没问题」，
    证明不了「没有漏项」—— 而漏项是这里唯一的风险，且后果不小：
    漏掉的智能体会拿到兜底提示词，它不是坏掉，只是**变得通用**，
    于是它的专业约束（比如政策问答的「必须有出处」）全部消失，
    而它在测试里仍然「能正常回复」。

    ⚠️ 顺带守住反向：表里不该有已不存在的智能体名（改名后忘了清）。
    多余的项不会有害，但它是「有人改了枚举却忘了改这里」的信号。
    """
    declared = {name.value for name in AgentName}

    missing = declared - set(PROMPTS)
    assert not missing, f"这些智能体没有登记提示词：{sorted(missing)}"

    stale = set(PROMPTS) - declared
    assert not stale, f"PROMPTS 里有已不存在的智能体名：{sorted(stale)}"


def test_main_agent_name_is_registered() -> None:
    """``MAIN_AGENT_NAME`` 指向一个真实登记的智能体。

    ⚠️ 它被多处当作「哪个是面向用户的那个」的判据（动态 Prompt 只挂在它
    身上、快车道只对它生效）。指向一个不存在的名字，这些功能会静默失效 ——
    没有异常，只是动态 Prompt 再也不生效了。
    """
    assert MAIN_AGENT_NAME in PROMPTS
    assert MAIN_AGENT_NAME in {name.value for name in AgentName}


def test_prompts_are_not_shared_between_agents() -> None:
    """⚠️ 不同智能体**不得**共用同一段提示词。

    共用看着省事，但会让「给政策问答加一条『必须有出处』」的改动同时
    影响订单查询 —— 而订单查询本来就不需要那条约束，多出来的约束
    会让它变得啰嗦甚至拒绝回答。
    """
    seen: dict[str, str] = {}
    for name, text in PROMPTS.items():
        if text in seen:
            pytest.fail(f"{name} 与 {seen[text]} 共用了同一段提示词")
        seen[text] = name


# ---------------------------------------------------------------------------
# 二、开发口吻泄漏
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(ALL_PROMPTS))
def test_prompt_has_no_developer_markers(name: str) -> None:
    """★ 提示词里不得出现开发口吻标记、文件路径或代码片段。

    ⚠️ 这些文字会**原样**进模型上下文。写「见 src/tools/travel.py」，
    模型可能把它当成对用户说的话复述出来；写 ``import`` 或 ``pytest``
    这类词，模型可能开始模仿代码风格回答。

    ⚠️ 需要给后来的开发者留说明时，写在**常量/模块的 docstring** 里 ——
    那里不进上下文。本文件的存在就是为了让「顺手在提示词里加个注释」
    这个动作被拦下来。
    """
    text = ALL_PROMPTS[name]
    for marker in DEVELOPER_MARKERS:
        assert marker not in text, f"{name} 的提示词里出现了 {marker!r}"


def test_no_prompt_contains_a_file_path() -> None:
    """⚠️ 单独再查一遍路径形态。

    ``src/`` 只覆盖了本项目的相对路径写法。模型上下文里出现任何路径都是
    干扰，所以用一个更宽的模式兜一层。
    """
    pattern = re.compile(r"[\w.-]+/[\w./-]+\.(?:py|yaml|yml|json|md|toml)")
    for name, text in ALL_PROMPTS.items():
        found = pattern.search(text)
        assert not found, f"{name} 的提示词里出现了文件路径：{found.group(0)!r}"


def test_prompts_do_not_mention_internal_class_names() -> None:
    """⚠️ 不出现内部类名。

    ``PolicyLimit``、``AgentName``、``ToolChunk`` 这些词对模型没有意义，
    但它们看起来像「某个具体的东西」，模型可能会把它们当成业务名词
    写进回复里。
    """
    internal = ("PolicyLimit", "AgentName", "ToolChunk", "FunctionTool", "MiddlewareBase", "LaneName")
    for name, text in ALL_PROMPTS.items():
        for token in internal:
            assert token not in text, f"{name} 的提示词里出现了内部类名 {token}"


# ---------------------------------------------------------------------------
# 三、不许写死具体数字
# ---------------------------------------------------------------------------
def test_prompts_do_not_hardcode_policy_numbers() -> None:
    """★★ 提示词里**不得**出现具体的差标/价格数值。

    这是本文件最要紧的一条。差标是**按用户**配的（``PolicyLimit`` 每人一份），
    价格是实时查的。提示词里写「酒店不超过 600 元」，模型就有可能在没调
    工具的情况下直接引用它 —— 而对一个差标 1200 元的用户，这句话是**错的**，
    且错得毫无痕迹（用户会以为系统就是这么规定的）。

    ⚠️ 提示词要做的是**要求模型去查**，而不是替它记住答案。
    """
    # 匹配「不超过 600 元」「上限 2000」「经济舱」这类具体承诺。
    # 只查带货币单位或明确数值的写法，避免误伤「一次只问一个」这种普通用量词。
    money = re.compile(r"\d+(?:\.\d+)?\s*(?:元|块钱|人民币|RMB|CNY)")
    for name, text in ALL_PROMPTS.items():
        found = money.findall(text)
        assert not found, f"{name} 的提示词里出现了具体金额：{found}"


def test_prompts_do_not_name_a_specific_cabin_class() -> None:
    """⚠️ 不写死舱位标准。

    舱位标准同样按用户/职级配置（``CabinClass``），写死「只能坐经济舱」
    对高管用户就是错的。要求模型去查差标即可。
    """
    cabins = ("经济舱", "商务舱", "头等舱", "超级经济舱")
    for name, text in ALL_PROMPTS.items():
        for cabin in cabins:
            assert cabin not in text, f"{name} 的提示词里写死了舱位「{cabin}」"


def test_prompts_tell_the_model_to_query_facts() -> None:
    """★ 反过来，提示词必须**明确要求**去查事实。

    只禁止写死数字是不够的 —— 那样模型只是「没有可抄的数字」，
    它仍然可能凭常识编一个。必须有一条正向的要求。
    """
    for name in (MAIN_AGENT_NAME, "order_query"):
        text = PROMPTS[name]
        assert "工具" in text, f"{name} 的提示词没有要求模型使用工具"
        assert any(word in text for word in ("查", "查询")), f"{name} 的提示词没有要求模型去查"


def test_prompts_forbid_fabrication_explicitly() -> None:
    """★ 提示词必须**明确**禁止编造。

    ⚠️ 「不要编造」这类否定指令必须写出来。只写「请调用工具查询」时，
    模型在工具失败或结果为空的情况下仍会倾向于「给出一个有用的答案」——
    而一个编造的班次号会让用户白跑一趟机场。
    """
    for name in (MAIN_AGENT_NAME, "order_query"):
        text = PROMPTS[name]
        assert "编" in text or "凭记忆" in text or "凭印象" in text, \
            f"{name} 的提示词没有明确禁止编造"


def test_main_agent_prompt_binds_numbers_to_tool_returns() -> None:
    """★★ 否定式禁令之外，必须有一条**正向的溯源绑定**。

    2026-10-03 实测：主智能体的提示词只有「绝对不要编造标准数值」这一条
    **否定**禁令，没有「答复里的数只能来自工具返回」这层**正向**绑定。
    同一个用户、同一个问题「住宿标准」连问 8 次，工具每次都返回同一个数，
    但正文里有 3 轮写成了另一个数（500）—— 那正是中文差旅语境里最常见的
    先验值，来自模型的印象而不是工具。否定禁令压不住它，因为模型并不认为
    自己在「编造」，它认为自己在「用常识」。

    ⚠️ 这条断言只保证措辞在场，**保证不了模型遵守**。行为侧的兜底在
    :mod:`src.orchestration.reply_guard` 的事实接地闸门里。
    """
    text = PROMPTS[MAIN_AGENT_NAME]
    assert "只能来自本轮工具的返回" in text, "主提示词缺少数值溯源绑定条款"
    assert "以工具为准" in text, "主提示词没有说明冲突时以工具为准"


def test_main_agent_prompt_spells_out_the_tool_round_mechanism() -> None:
    """★★ 提示词必须写清「工具轮的文字用户看不到」这个**机制**。

    ⚠️ 2026-10-04 之前，这一节写的是一句与实现**相反**的话 ——
    「你输出的每一段文字都会原样出现在用户屏幕上，没有草稿区这东西」。
    真实的链路（``reply_guard._settle_round``）是：**调用了工具的那一轮，
    文字整轮丢弃**（指标 ``dropped_tool_round_*``）。提示词按旧文写，
    模型会以为「我先查一下差标」用户看得见，于是继续写 —— 实测 16 轮
    真实对话里就有 1 轮是这么被丢的（那次用户白等了一轮）。

    ⚠️ 钉死措辞是**有意**的：这两句是这次修复的落点。改措辞必须同步改
    测试，否则「提示词被悄悄改回旧口径」不会再被任何东西拦住 ——
    旧口径不会让任何测试变红，只会让 ``dropped_tool_round_narration``
    缓慢抬头，而那条曲线在看板上，不在 CI 里。
    """
    text = PROMPTS[MAIN_AGENT_NAME]

    assert "调了工具的那一轮，你写下的文字不会发给用户" in text, (
        "主提示词没有说明「工具轮的文字会被丢掉」这个机制"
    )
    assert "要查就只调工具，一个字都别配" in text, (
        "主提示词没有给出与机制配套的处方（只调工具、不配文字）"
    )
    assert "你输出的每一段文字都会原样出现在用户屏幕上" not in text, (
        "旧口径回来了 —— 那句话与守卫的真实行为相反"
    )


def test_main_agent_prompt_forbids_rewriting_numbers() -> None:
    """★★ 「溯源」之外还要有「不许改写」—— 两者是不同的问题。

    ⚠️ 溯源条款管的是「这个数**从哪来**」，管不住「这个数**被写成什么样**」。
    2026-10-04 的对抗审计把这一条单列出来：模型可以引用一个**有依据**的数，
    却把它四舍五入成另一个值（工具说 5,980，正文写「约 6 000」）——
    溯源判据在源头上是满足的，而用户拿到的仍是错的数。

    ⚠️ 这条断言只保证措辞在场。行为侧的兜底在
    :func:`~src.orchestration.reply_guard._ungrounded_limit_claims`，
    以及 ``_is_derived_total`` 对「合计 vs 上限」的区分。
    """
    text = PROMPTS[MAIN_AGENT_NAME]
    assert "不改写、不凑整" in text, "主提示词缺少「照原样引用」的条款"
    assert "四舍五入" in text, "主提示词没有点名「四舍五入」这种改写形态"


def test_main_agent_prompt_requires_the_arithmetic_to_be_shown() -> None:
    """★ 做合计时必须写出算式，且每个因子都要有来源。

    ⚠️ 只写「不要凭空给一个数」是不够的：合计（几晚 × 每晚多少）在模型看来
    是「算出来的」，不算凭空 —— 而算错、算成别的数、用印象里的单价去乘，
    都是实测过的形态。要求它把算式写出来，用户才有可能一眼核对，
    守卫的「合计」判据也才有东西可对。
    """
    text = PROMPTS[MAIN_AGENT_NAME]
    assert "算式" in text, "主提示词没有要求写出算式"
    assert "不要用自己的估算把它补上" in text, "主提示词没有禁止用估算补齐因子"


def test_main_agent_prompt_keeps_tool_errors_internal() -> None:
    """★ 工具的错误原文不许抄给用户。

    ⚠️ 工具的失败返回里带着 ``detail``（异常类型与原文，可能是英文、
    可能带参数名）。审计发现过两处：一处是工具把内部细节拼进了给用户看的
    摘要，另一处是模型把工具返回原样复述出来。前者已从工具侧修掉，
    这条守的是后者 —— **提示词侧**的那一半。
    """
    text = PROMPTS[MAIN_AGENT_NAME]
    assert "不要原样抄给用户" in text, "主提示词没有约束工具错误原文的转述方式"


def test_main_agent_prompt_hands_compliance_verdicts_to_the_tool() -> None:
    """★★ 「能不能报」必须由工具判，模型不许自己拿价格和标准比大小。

    ⚠️ 这是本项目「能用代码判定的绝不交给模型」原则在提示词里的落点。
    模型自己比较「800 与 600 谁大」时，错法有无数种：比反了、拿错档、
    把单晚价当总价。而核对工具里的判定是确定性代码，且它的结论会**回填**
    到 ``policy_verdict`` 卡片上 —— 模型口头算一个、卡片上是另一个的话，
    用户会看到自相矛盾的两处结论。
    """
    text = PROMPTS[MAIN_AGENT_NAME]
    assert "不要自己拿价格和标准做比较" in text, "主提示词没有把合规判定交给工具"


def test_policy_prompt_forbids_rewriting_the_source_numbers() -> None:
    """★★ 政策问答引用制度原文时，数字必须照抄。

    ⚠️ 与主提示词那条同源，但**风险更高**：这里的数字是**制度原文**，
    用户会拿它去报销、去和审批人理论。把「不超过 5,980」改写成「不超过
    6,000」听起来无害，实际是把一条硬性规定放宽了 —— 而且放宽的方向
    通常是「更好看、更好记」，也就是更容易被模型顺手做出来。
    """
    text = PROMPTS["policy_rag"]
    assert "照抄" in text, "政策提示词没有要求照抄原文数字"
    assert "四舍五入" in text, "政策提示词没有点名「四舍五入」这种改写形态"


def test_approval_prompt_forbids_rounding_the_amount() -> None:
    """★ 申请单上的金额不许凑整。

    ⚠️ 申请单的金额是审批人的对账依据：取整之后的数会和行程、发票对不上。
    而「凑个整数好看」恰恰是模型处理数字时最自然的倾向。
    """
    text = PROMPTS["approval"]
    assert "凑成整数" in text, "申请提示词没有禁止把金额凑整"
    assert "算式" in text, "申请提示词没有要求把推算过程写给用户看"


def test_main_agent_prompt_bounds_citation_and_scope() -> None:
    """★ 出处与适用范围的边界也要写出来。

    - 出处：工具没给的来源，模型不该替它编一个文件名/条款号。实测抓到过
      「依据：差旅制度「住宿标准」条款」这种编造（那轮只调了查标准的工具，
      上下文里根本没有条款原文）。
    - 适用范围：工具没有分档，不等于制度没有分档。实测抓到过「目前不按城市
      或职级分档」—— 而知识库里的政策文档是**有**城市分级的。把「我没查到」
      说成「不存在」是另一类不实陈述，比编数字更隐蔽。
    """
    text = PROMPTS[MAIN_AGENT_NAME]
    assert "不要替它编一个文件名或条款号" in text, "主提示词缺少出处来源约束"
    assert "没有细分" in text, "主提示词缺少适用范围/边界的表述纪律"


def test_main_agent_prompt_restrains_team_building() -> None:
    """★ 一句话能答的问题不该被升级成多智能体编排。

    2026-10-03 实测：「住宿标准」连问 8 次，前 7 次 4–7 秒直接作答，
    第 8 次走了完整的建团队→拉成员→派任务链路，用了 **50.6 秒**，答案还是
    同一句话。项目侧提示词里当时**没有任何**团队使用边界（唯一相关的一句
    只是「别把委派过程说给用户听」，它甚至预设了会委派）。

    ⚠️ 这里断言的是「有没有这条纪律」，不是「模型会不会建团队」——
    确定性闸门在 :class:`src.orchestration.lane.LaneRouterMiddleware`
    的快车道工具收窄里。
    """
    text = PROMPTS[MAIN_AGENT_NAME]
    assert "能用一句话答的问题，就自己答完" in text, "主提示词缺少团队使用边界"
    assert "别拆" in text, "主提示词没有给出「该不该拆」的判断标准"


# ---------------------------------------------------------------------------
# 四、面向用户的智能体要有「说话方式」的指引
# ---------------------------------------------------------------------------
def test_main_agent_prompt_guides_the_tone() -> None:
    """★ 主智能体是唯一**直接**面向用户的，必须有语气与表达方式的指引。

    ⚠️ 缺了这段，模型的默认行为是客服腔（「您好，我很乐意为您服务」）
    加上长篇罗列。而差旅场景里用户是在办正事，想尽快拿到能用的信息 ——
    语气不是锦上添花，它直接决定这个助手好不好用。
    """
    text = PROMPTS[MAIN_AGENT_NAME]

    assert "中文" in text, "没有要求用中文"
    assert any(word in text for word in ("简洁", "结论先行")), "没有对表达方式提出要求"


def test_main_agent_prompt_covers_the_business_flow() -> None:
    """★ 主智能体的提示词要覆盖四个业务环节。

    ⚠️ 逐个查关键词而不是只查「有没有内容」：漏掉「核对标准」这一环的后果
    是模型给出「这个酒店可以报」这类**未经核对**的结论 —— 而差标是按用户
    配的，它不可能知道。这是四环里最容易漏、后果最直接的一个。
    """
    text = PROMPTS[MAIN_AGENT_NAME]
    for stage, keywords in (
        ("收集要素", ("要素",)),
        ("查询方案", ("交通", "住宿")),
        ("核对标准", ("标准", "差标")),
        ("确认后执行", ("确认",)),
    ):
        assert any(word in text for word in keywords), f"主智能体提示词没有提到「{stage}」"


def test_policy_prompt_demands_citations() -> None:
    """★ 政策问答的提示词必须要求**注明出处**。

    ⚠️ 这是这个智能体存在的全部理由。没有出处的政策回答和模型自己编的
    没有区别，而差旅标准直接关系到用户能不能报销。
    """
    text = PROMPTS["policy_rag"]

    assert any(word in text for word in ("出处", "来源", "依据")), "没有要求注明出处"
    assert any(word in text for word in ("没有查到", "查不到")), "没有说明「查不到」时该怎么办"


def test_approval_prompt_requires_confirmation() -> None:
    """★ 出差申请的提示词必须要求**提交前复述确认**。

    ⚠️ 这个智能体经手的操作会真的生成一张申请单。系统层面有 HITL 确认框，
    但那个框只显示标题和金额 —— 用户真正用来判断的依据是模型的复述。
    """
    text = PROMPTS["approval"]

    assert "确认" in text, "没有要求确认"
    assert any(word in text for word in ("复述", "列出来")), "没有要求把内容复述给用户看"
    assert "事由" in text, "没有要求填写事由"


def test_intent_prompt_states_it_is_not_user_facing() -> None:
    """★ 意图识别的提示词要说明它**不直接回复用户**。

    ⚠️ 不说的话，它可能用客服腔组织一段面向用户的回答，而它的输出是要被
    主智能体消费的结构化结果 —— 掺进客服腔会增加主智能体解析的负担，
    甚至让它把那段话直接透传给用户。
    """
    text = PROMPTS["intent"]

    assert any(word in text for word in ("不直接回答", "不面向用户", "只输出")), \
        "没有说明这个智能体不直接面向用户"


def test_intent_prompt_mentions_multi_intent() -> None:
    """★ 意图识别必须处理**多意图**输入。

    ⚠️ 这是 P3 验收项之一（「多意图输入返回结构化拆解」）。提示词不提，
    模型就会默认「一句话只有一个意图」，返回单个意图 ——
    而「订票并查一下报销标准」这类输入在真实使用里很常见。
    """
    text = PROMPTS["intent"]

    assert any(word in text for word in ("不止一个", "多意图", "同时包含")), \
        "没有提示模型一句话可能有多个意图"


# ---------------------------------------------------------------------------
# 五、兜底提示词
# ---------------------------------------------------------------------------
def test_unknown_agent_gets_a_usable_fallback() -> None:
    """★★ 未登记的智能体拿到的是**可用的通用助手**提示词，而不是空串。

    ⚠️ 返回空串或 ``None`` 会让智能体以「没有身份设定」的状态运行，
    行为完全无法预测 —— 它可能自称是别的助手，可能拒绝回答。
    而返回一段通用的差旅助手提示词，用户完全不会察觉到异常。

    这条的重要性在于「它只在配置出错时生效」—— 也就是说它**永远不会**
    在日常开发中被任何人看到，只会在生产上出错的那一刻起作用。
    """
    text = prompt_for("__never_registered__")

    assert text.strip(), "兜底提示词不能为空"
    assert len(text) > 50, "兜底提示词太短，不足以给模型一个身份设定"
    assert "差旅" in text, "兜底提示词应当仍然把它定位成差旅助手"


def test_fallback_does_not_reveal_the_misconfiguration() -> None:
    """★★ 兜底提示词**不得**让模型意识到自己处于异常状态。

    ⚠️ 写「你没有配置提示词，请告知用户」这类话，模型会在回复里表现出来
    （「我似乎没有正确的配置，建议联系管理员」）。用户看到这句话只会困惑，
    而且他没有任何办法处理它。

    ⚠️ 这是一个**面向用户**的取舍：提示词是给模型看的，但它的效果最终
    体现在用户看到的话上。开发者的告警应该走日志，不走提示词。
    """
    text = prompt_for("__never_registered__").lower()

    for leak in ("配置", "错误", "异常", "未登记", "不存在", "管理员", "管理员", "fallback"):
        assert leak not in text, f"兜底提示词泄漏了异常状态：{leak!r}"


def test_fallback_also_respects_the_no_marker_rule() -> None:
    """兜底提示词同样不许有开发标记 —— 它是最少被人工审阅的一段。"""
    text = prompt_for("__never_registered__")
    for marker in DEVELOPER_MARKERS:
        assert marker not in text


def test_prompt_for_returns_the_registered_text() -> None:
    """登记过的名字拿到的就是那一段（而不是兜底）。"""
    for name in AgentName:
        assert prompt_for(name.value) == PROMPTS[name.value]


def test_prompt_for_logs_a_warning_on_fallback(caplog: pytest.LogCaptureFixture) -> None:
    """⚠️ 走兜底时**必须**记一条 warning。

    兜底提示词的设计目标是「让用户察觉不到」—— 这正是它危险的地方：
    没有日志的话，一个智能体长期跑在通用提示词上而没人知道，
    它的专业约束（政策问答的「必须有出处」）已经悄悄消失了。
    """
    import logging

    with caplog.at_level(logging.WARNING):
        prompt_for("__never_registered__")

    assert any("提示词" in r.message for r in caplog.records), "回退到兜底时没有记日志"


# ---------------------------------------------------------------------------
# 六、语言
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(ALL_PROMPTS))
def test_prompt_is_written_in_chinese(name: str) -> None:
    """★ 提示词是中文的，且中文字符占多数。

    ⚠️ 查比例而不是查「有没有中文」：一段英文提示词里混一个「的」字
    也能通过后者。而中英混排的提示词会让模型在回复里也混英文 ——
    对中文用户是明显的体验缺陷。
    """
    text = ALL_PROMPTS[name]
    chinese = sum(1 for ch in text if "一" <= ch <= "鿿")
    letters = sum(1 for ch in text if ch.isalpha())
    assert letters > 0, f"{name} 的提示词没有任何字母"
    assert chinese / letters > 0.8, f"{name} 的提示词中文字符占比只有 {chinese / letters:.0%}"


@pytest.mark.parametrize("name", sorted(ALL_PROMPTS))
def test_prompt_has_no_unfilled_placeholders(name: str) -> None:
    """⚠️ 没有忘记填的占位符。

    ``{stage}``、``%s``、``<TODO>`` 这类东西留在提示词里，模型会把它当成
    字面内容 —— 回复里出现一个 ``{stage}`` 是很显眼的缺陷。
    """
    text = ALL_PROMPTS[name]
    for pattern in (r"\{[a-z_]+\}", r"%[sd]", r"<[A-Z_]+>"):
        found = re.search(pattern, text)
        assert not found, f"{name} 的提示词里有未填的占位符：{found.group(0)!r}"


def test_prompts_do_not_reference_intents_by_raw_value() -> None:
    """⚠️ 提示词里不出现 ``PLAN_TRIP`` 这类内部枚举值。

    它们是给代码看的标识符，不是给人看的词。模型看到会照抄进回复。
    需要提到某个意图时，用中文说法（「规划行程」）。
    """
    for name, text in ALL_PROMPTS.items():
        for intent in Intent:
            assert intent.value not in text, f"{name} 的提示词里出现了内部意图值 {intent.value}"
