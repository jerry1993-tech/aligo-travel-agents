# -*- coding: utf-8 -*-
"""**动态 Prompt 组装** —— 按当前对话阶段改写 system prompt。

文件职责：
    实现博客第 361-373 行描述的「动态 Prompt 状态机」：识别用户所处的
    对话阶段，**只把该阶段需要的信息写进 system prompt**，从而「将模型
    注意力聚焦于当前主链路」。

    本模块只负责**拼装字符串**这一件事。真正把它挂进 agent 的是
    ``src/orchestration/context.py`` 里的 ``ContextInjectionMiddleware``
    （走 ``on_system_prompt`` 钩子）。

上下游依赖：
    - 上游：:mod:`src.domain`（:class:`TripStage` / :class:`TravelRequest`）。
    - 下游：``src/orchestration/context.py``。

═══ ⚠️ 本模块**刻意不 import** ``agentscope`` ═══

与 :mod:`src.domain`、:mod:`src.orchestration.classifier` 同样的纯度约定，
但这里还有一条**性能上的硬理由**，见下。

═══ 这个函数每轮模型调用都会跑一遍 ═══

已核实：``on_system_prompt`` 钩子在**每一个推理轮次**都会被调用一次
（``agent/_agent.py:1728 -> 3216 -> 3251``），且**框架不缓存**返回值。
一次差的回复里模型可能推理十几轮，也就是说本函数会被调用十几次。

因此它必须**廉价、确定、无副作用**：

- **不做 I/O**：不查库、不调模型、不读文件。要用的数据由调用方查好了传进来
  （``request`` / ``profile_summary``）。
- **确定性**：同样的入参必须产出同样的字符串。否则「同一轮对话里两处日志
  对不上」会极难排查。⚠️ 特别注意**不要遍历 set/dict 后直接拼串** ——
  Python 的 set 迭代顺序在同一进程内稳定但跨进程不保证，会让 prompt
  在不同副本间不一致，进而让灰度对比失效。本模块所有拼接都走**固定顺序
  的元组**。
- **不抛异常**：它跑在 agent 的主链路上，抛异常会让整轮回复失败。所有
  边界情况都就地降级。

═══ 幂等性：本函数可以被安全地重复调用 ═══

动态段落用一对显式标记包起来（:data:`MARKER_BEGIN` / :data:`MARKER_END`），
:func:`build_system_prompt` 在拼装前**先剥掉旧的动态段落**。这样下面两种
写法都得到同一个结果::

    build_system_prompt(base, stage=A)                       # 直接组装
    build_system_prompt(build_system_prompt(base, stage=A), stage=A)   # 幂等

⚠️ 这不是多余的防御。中间件链上**可能不止一个中间件改写 prompt**
（框架自己的 skills / offloader 就在做），而 ``on_system_prompt`` 的
``current_prompt`` 是**前序中间件的输出**。一旦有人把本函数接到自己的
输出上，没有剥离逻辑就会让动态段落**每一轮翻一倍** —— 几轮之后 prompt
被撑爆，症状是「聊几句就报上下文超限」，而根因极其难猜。
"""

from __future__ import annotations

import re

from src.domain import CabinClass, TransportMode, TravelRequest, TripStage
from src.orchestration.amounts import render_amount

#: 动态段落的起始标记。
#:
#: ⚠️ 用 HTML 注释而不是「===== 动态段落 =====」这类可见文字：注释在
#: markdown 渲染里不可见，即使剥离逻辑失效、段落漏进了给用户看的输出，
#: 也不会污染界面。而方括号/等号包裹的标题会明晃晃地出现在回复里。
MARKER_BEGIN = "<!-- ALIGO:DYNAMIC:BEGIN -->"

#: 动态段落的结束标记。
MARKER_END = "<!-- ALIGO:DYNAMIC:END -->"

#: 匹配一对完整动态段落的正则（非贪婪，跨行）。
_MANAGED_PAIR = re.compile(
    re.escape(MARKER_BEGIN) + r".*?" + re.escape(MARKER_END),
    re.DOTALL,
)

#: 匹配**未闭合**的起始标记（一直到串尾）。
#:
#: ⚠️ 单独处理未闭合的情况，而不是指望上面的成对正则兜住。未闭合意味着
#: 有人的拼接逻辑写漏了结束标记；此时若只按成对匹配剥离，残留的半个段落
#: 会**静默累积**，而且因为它不含结束标记，后续每一次剥离都拿它没办法。
_MANAGED_UNCLOSED = re.compile(
    re.escape(MARKER_BEGIN) + r".*\Z",
    re.DOTALL,
)


# --------------------------------------------------------------------------
# 阶段 → 主链路指令
# --------------------------------------------------------------------------
#: 每个对话阶段写进 prompt 的**主链路指令**。
#:
#: ⚠️ 做成**全量表**而不是「一串 if」：新增阶段若忘了登记，下面的
#: :func:`test_every_stage_has_a_directive` 会立刻失败。若写成 if 链，
#: 漏掉的阶段会退化成「没有任何动态指令」，表现为该阶段下模型行为莫名其妙
#: 地回到通用状态 —— 而这种「某些情况下不太对劲」是最难定位的一类问题。
#:
#: ⚠️ 指令里**不重复**基础 prompt 已经说过的通用规则（身份、语气、工具使用
#: 纪律）。重复的后果不是「强调」，而是让模型在不同位置看到措辞略有出入的
#: 同一规则，进而对哪条为准产生困惑。这里只写**本阶段独有**的东西。
_STAGE_DIRECTIVES: dict[TripStage, str] = {
    TripStage.IDLE: (
        "用户尚未表达明确的出差意图。"
        "用一两句话回应，并自然地把话题引向「去哪、什么时候、从哪出发」。"
        "不要主动罗列工具能力，也不要一次抛出多个问题。"
    ),
    TripStage.COLLECTING: (
        # ⚠️ 措辞刻意**不引用小节标题**（「见上方『已知要素』」这类）。
        # 那些小节是条件输出的：要素全齐时没有「待补齐」段、一个要素都没
        # 收集到时没有「已知」段。指令若点名了不存在的小节，模型就会去找
        # 一个没有的东西 —— 轻则忽略整条指令，重则凭空编一个字段来追问。
        # 改成描述**语义**（「用户已经提供过的内容」）就没有这个问题。
        #
        # ⚠️ 2026-10-04 改了时态：原文是「你正在补全出差要素」——一句**状态
        # 描述**。实测模型会把它当旁白复述出来（「我先确认一下要素」），
        # 而工具轮的文字会被守卫丢掉、正文轮里这类话又答不了用户。改成
        # **规则句**（本轮只做什么、不许写什么）就没有可复述的空间。
        "本轮只做一件事：把还没定下来的要素问出来。"
        "**只追问本次列出的待补齐项**，一次最多两项，能只问一项就只问一项，"
        "优先问最关键的。"
        "用户已经提供过的内容**绝对不要再问一遍**——"
        "重复追问用户已经回答过的问题是对话式收集最招人烦的失败模式。"
        "如果用户这一轮提供的信息不完整或有歧义，先按最合理的解释填入，"
        "需要用户确认就把确认写进追问本身（比如「按周三出发，对吗」），"
        "不要停下来要求用户重新表述，也不要写「我先确认一下要素」这类过程叙述。"
    ),
    TripStage.CONFIRMING: (
        "要素已齐全。请**完整复述**你理解到的行程要素，"
        "然后询问用户是否确认。"
        "在用户明确表示确认之前，**不要调用任何有副作用的下单类工具**——"
        "复述的目的正是让用户有机会纠正误解。"
    ),
    TripStage.DONE: (
        "本次出差安排已经完成。后续对话中若用户提出修改，"
        "视为一次**新的**需求，重新走收集流程，不要直接改动已完成的方案。"
    ),
    TripStage.CANCELLED: (
        "本次出差安排已取消，这是终态。"
        "如果用户想重新安排，把它当作一次全新的需求从头开始，"
        "不要试图恢复被取消的方案。"
    ),
}

#: 阶段 → 中文展示名（写进 prompt 供模型理解「现在在哪一步」）。
_STAGE_LABELS: dict[TripStage, str] = {
    TripStage.IDLE: "尚未开始",
    TripStage.COLLECTING: "收集中",
    TripStage.CONFIRMING: "待用户确认",
    TripStage.DONE: "已完成",
    TripStage.CANCELLED: "已取消",
}

#: 交通方式 → 中文展示名。
#:
#: ⚠️ 这些标签**故意放在本模块**而不是 :mod:`src.domain.enums`：它们是
#: **prompt 用词**，随时可能因为措辞调整而变化（比如「飞机」改成「航班」），
#: 而 domain 层的中文名是给界面和日志用的、需要稳定。两者的变更节奏不同，
#: 放一起会互相牵制。
_TRANSPORT_LABELS: dict[TransportMode, str] = {
    TransportMode.FLIGHT: "飞机",
    TransportMode.TRAIN: "火车",
    TransportMode.CAR: "汽车",
}

#: 舱位等级 → 中文展示名。
_CABIN_LABELS: dict[CabinClass, str] = {
    CabinClass.ECONOMY: "经济舱",
    CabinClass.PREMIUM_ECONOMY: "超级经济舱",
    CabinClass.BUSINESS: "商务舱",
    CabinClass.FIRST: "头等舱",
}

#: 一轮最多让模型追问几项。
#:
#: ⚠️ 定成 2 而不是「全部列出」：一次抛出四个问题，用户大概率只回答其中
#: 一两个，剩下的在下下轮又得重问，总轮数反而更多；而且每次都看到一长串
#: 待填项，体验上像在填表而不是对话。定成 1 又太慢。这是产品取舍，不是
#: 技术限制，所以写成常量并在这里说明。
MAX_MISSING_PROMPTS = 2


def strip_managed_sections(prompt: str) -> str:
    """剥掉 prompt 里所有由本模块生成的动态段落。

    Args:
        prompt (`str`): 任意 prompt 文本。

    Returns:
        `str`: 去掉动态段落并**去掉首尾空白**的结果。

    ⚠️ 返回值会 ``strip()``。理由：动态段落总是被附加在末尾，剥离后会留下
    一串空行；不清理的话，反复“剥离—附加”会让 prompt 末尾的空行越积越多，
    最终在按行做 diff 的日志里表现为「每轮都有变化」的噪音。

    ⚠️ 未闭合的 ``MARKER_BEGIN``（有头无尾）会被一直剥到串尾。见
    :data:`_MANAGED_UNCLOSED` 的说明。
    """
    cleaned = _MANAGED_PAIR.sub("", prompt)
    cleaned = _MANAGED_UNCLOSED.sub("", cleaned)
    return cleaned.strip()


def describe_known_slots(request: TravelRequest) -> list[str]:
    """把已收集到的要素整理成「字段名：值」的中文行。

    Args:
        request (`TravelRequest`): 当前收集状态。

    Returns:
        `list[str]`: 形如 ``["出发城市：杭州", "出差天数：3 天"]`` 的行；
            一个都没收集到时返回空列表。

    ⚠️ **无法区分「用户明确说了」与「用的默认值」**，所以默认值一律不列。
    例如 ``travelers`` 默认 1，无法判断用户是真说了「我一个人」还是根本没提。
    两个方向都试过：若把默认值也列成已知，模型会以为「人数已确认」而不再
    追问，一旦用户其实是两个人就成了**静默的错误**；若不列，最坏情况是
    「几个人」被多问一次。**宁可多问一句，不可少问** —— 与
    ``src/orchestration/classifier.py`` 里「歧义一律落到安全一侧」是同一条原则。

    真要把两者区分开，得靠 pydantic 的 ``model_fields_set`` 或自定义哨兵值，
    但那要求 ``TravelRequest`` 的构造路径全程保持「未提供的字段不传」——
    而它现在是由模型的结构化输出整体填充的，做不到。这里选择诚实降级。

    ⚠️ 顺序是**固定的元组**而非遍历 dict，理由见模块文档的「确定性」一节。
    """
    lines: list[str] = []

    if request.origin.strip():
        lines.append(f"出发城市：{request.origin.strip()}")
    if request.destination.strip():
        lines.append(f"目的城市：{request.destination.strip()}")
    if request.depart_date.strip():
        lines.append(f"出发日期：{request.depart_date.strip()}")
    if request.return_date.strip():
        lines.append(f"返程日期：{request.return_date.strip()}")
    # ⚠️ days 与 depart_date 是**互补**关系（用户要么给天数要么给日期），
    # 所以两者可以同时出现，不互斥。
    if request.days > 0:
        lines.append(f"出差天数：{request.days} 天")
    # ANY 表示「未指定」，不是「用户说了不限」—— 见上面的说明。
    if request.transport_mode is not TransportMode.ANY:
        lines.append(f"交通方式：{_TRANSPORT_LABELS.get(request.transport_mode, request.transport_mode.value)}")
    if request.cabin_class is not CabinClass.ANY:
        lines.append(f"舱位等级：{_CABIN_LABELS.get(request.cabin_class, request.cabin_class.value)}")
    if request.hotel_required:
        lines.append("需要预订酒店：是")
    if request.hotel_area.strip():
        lines.append(f"酒店区域偏好：{request.hotel_area.strip()}")
    if request.travelers > 1:
        lines.append(f"同行人数：{request.travelers} 人")
    if request.purpose.strip():
        lines.append(f"出差事由：{request.purpose.strip()}")
    if request.budget > 0:
        # ⚠️ 不能写成 ``{request.budget:g}``（2026-10-04 缺陷 P2）：``:g`` 只留
        # 6 位有效数字，1200000 会变成 ``1.2e+06``。这一行是**给模型照抄的**
        # 素材（提示词里明写「原样引用、不凑整」），模型照抄就会把用户没说过
        # 的科学计数法发给用户；反过来，模型若写成正常人写法的 ``1200000``，
        # 回复守卫的接地闸门又会因为工具/提示词侧记的是 ``1.2e+06`` 而判它
        # 「无依据」—— 管道自己违反了它对模型提出的数字纪律。
        lines.append(f"预算上限：{render_amount(request.budget)} 元")

    return lines


def user_data_excerpt(
    request: TravelRequest | None,
    profile_summary: str = "",
) -> str:
    """拼出动态段落里**承载用户数据**的那部分文本（供回复守卫做数字溯源）。

    回复守卫有一条「事实接地闸门」：正文里断言了一个差标上限、而这个数不在
    本轮工具返回里时，判定为编造并拦下重说。它的**合法来源**之一是「用户自己
    说过的数」—— 因为「我订了 500 的能报吗」里的 500 不是编造（见
    ``src/orchestration/reply_guard.py`` 的 :func:`_ungrounded_limit_claims`）。

    ⚠️ 但「用户说过的数」不止出现在**本轮**输入里：用户前一嘴说了「预算
    15000」，后续每一轮的动态 Prompt 都会把「预算上限：15000 元」写进去，
    模型照抄这句**完全正确**的话，却会因为「本轮用户文本里没有 15000」被判成
    编造 —— 2026-10-04 实测（缺陷 P1）：拦下 → 重说 → 模型坚持引用 → 最终
    用户看到的是一句道歉兜底。

    所以本函数把「我们**自己写进 prompt** 的那部分用户数据」单独摘出来，供
    ``ContextInjectionMiddleware`` 登记、再由守卫当作第四个合法来源。摘的
    范围刻意**只要用户数据**（已知要素 + 长期画像），不含阶段指令这类静态
    文案 —— 静态文案里的数字（例如「一次最多追问两项」）不是用户说过的，
    放进豁免集合只会白白放宽闸门。

    ⚠️ 与 :func:`_slots_section` 的同步责任：那边渲染的「已知要素」行必须
    逐条来自 :func:`describe_known_slots`（本函数也用它）。若将来往动态段落里
    加了别的**用户数据**来源而忘了在这里补，症状是「一段正确答复被拦下」——
    是个响亮的误报（会被日志与指标记到），不是静默错放。

    Args:
        request (`TravelRequest | None`): 已收集的出差要素；``None`` 表示没有。
        profile_summary (`str`): 长期画像摘要；空串表示没有。

    Returns:
        `str`: 摘出来的文本（各段以换行相连）；没有用户数据时返回空串。
    """
    parts: list[str] = []
    if request is not None:
        parts.extend(describe_known_slots(request))
    summary = profile_summary.strip()
    if summary:
        parts.append(summary)
    return "\n".join(parts)


def _stage_section(stage: TripStage) -> str:
    """生成「当前阶段」一节。

    Args:
        stage (`TripStage`): 当前阶段。

    Returns:
        `str`: 该节的 markdown 文本。

    ⚠️ 未知阶段（枚举里没登记的）退化成只报阶段名、不给指令，而不是抛异常。
    本函数跑在 agent 主链路上，抛异常会让整轮回复失败 —— 代价与「prompt
    少了一段」完全不成比例。

    ⚠️ 也接受**裸字符串**。``TravelRequest.stage`` 由 pydantic 保证是枚举，
    但本函数也可能被中间件直接调用（阶段可能来自 session 元数据、请求体、
    甚至某处 JSON 反序列化后的字符串）。既然签名声明了「未知阶段降级」，
    就必须真的对所有输入降级 —— 只处理枚举、遇到 str 就 ``stage.value``
    抛 ``AttributeError``，等于把「降级」写在了文档里而没写在代码里。
    这类文档与实现不符最危险：读代码的人会相信它不会抛。
    """
    # ``TripStage`` 是 StrEnum，所以下面两个 ``.get`` 在传入等值字符串时
    # 也能命中；传未知值时返回 None，走到兜底分支。
    raw = stage.value if isinstance(stage, TripStage) else str(stage)
    label = _STAGE_LABELS.get(stage) or raw
    directive = _STAGE_DIRECTIVES.get(stage, "")
    lines = ["## 当前阶段", f"用户正处于：**{label}**（{raw}）"]
    if directive:
        lines.append(directive)
    return "\n".join(lines)


def _slots_section(request: TravelRequest, stage: TripStage) -> str:
    """生成「已知要素 / 待补齐要素」一节。

    Args:
        request (`TravelRequest`): 当前收集状态。
        stage (`TripStage`): 当前阶段。

    Returns:
        `str`: 该节的 markdown 文本；无内容时返回空串。

    ⚠️ 「待补齐要素」**只在 COLLECTING 阶段列出**。在 CONFIRMING 阶段，
    必填项理论上已齐全（缺了就不该进这个阶段），即便因为用户中途改口而
    真的缺了，把「待补齐」摆出来会让模型倾向于**退回追问**，而不是先
    完成用户当下要的确认动作。此时该由阶段指令（复述 + 请确认）主导。
    """
    known = describe_known_slots(request)
    lines: list[str] = []

    if known:
        lines.append("## 已知要素")
        lines.append("以下内容用户已经提供，**不要重复询问**：")
        lines.extend(f"- {line}" for line in known)

    # ⚠️ 用 ``==`` 而不是 ``is``：``stage`` 可能是等值的裸字符串
    # （见 :func:`_stage_section` 的说明），此时 ``is`` 恒为假，
    # 「待补齐要素」这一节会在 COLLECTING 阶段**静默消失**。
    if stage == TripStage.COLLECTING:
        missing = request.missing_required()
        if missing:
            lines.append("")
            lines.append("## 待补齐要素")
            asked = missing[:MAX_MISSING_PROMPTS]
            lines.append(f"本轮请优先追问（最多 {MAX_MISSING_PROMPTS} 项）：{'、'.join(asked)}")
            rest = missing[MAX_MISSING_PROMPTS:]
            if rest:
                # ⚠️ 把「还有但本轮不问」的项也写出来，是为了**避免模型自己
                # 重新推导缺什么**。若不写，模型看到「本轮只问两项」却不知道
                # 剩余项是什么，可能自行脑补出并不缺的字段来追问。
                lines.append(f"其余待补齐项（本轮先不追问）：{'、'.join(rest)}")

    return "\n".join(lines)


def build_system_prompt(
    base_prompt: str,
    *,
    stage: TripStage = TripStage.IDLE,
    request: TravelRequest | None = None,
    profile_summary: str = "",
    enabled: bool = True,
) -> str:
    """组装带动态段落的 system prompt。

    Args:
        base_prompt (`str`): 基础 prompt（角色设定 + 通用纪律）。
            ⚠️ 应当是**原始基础 prompt**；传上一轮的输出也能work（见
            模块文档的幂等性一节），但那会浪费一次剥离。
        stage (`TripStage`): 当前对话阶段，决定主链路指令。
        request (`TravelRequest | None`): 已收集的出差要素；``None`` 表示
            本轮拿不到（例如快车道直接路由，没走收集链路），此时不输出
            要素相关的小节。
        profile_summary (`str`): 用户长期画像摘要（来自 P4 的记忆模块）；
           空串表示没有或未启用。
        enabled (`bool`): 是否启用动态 Prompt。``False`` 时**原样返回
            剥离过的基础 prompt**。

    Returns:
        `str`: 组装好的完整 system prompt。

    ⚠️ ``enabled=False`` 时仍然执行剥离。理由：关闭开关的语义是「不要再
    动态改写」，若此时把上一轮的动态段落原样留着，开关就只挡住了「新增」
    而没挡住「既有的」，表现为「关掉之后 prompt 里还挂着旧阶段的指令」。
    这种半开半关的状态比全开或全关都难排查。这个开关存在的意义是排障
    （怀疑动态 Prompt 导致行为异常时一键关掉），所以它必须关得干净。
    """
    stripped = strip_managed_sections(base_prompt)
    if not enabled:
        return stripped

    sections: list[str] = [_stage_section(stage)]

    if request is not None:
        slots = _slots_section(request, stage)
        if slots:
            sections.append(slots)

    if profile_summary.strip():
        # ⚠️ 画像来自用户的长期历史，**可能与本次出差无关**（比如去年常去
        # 上海，这次去北京）。所以措辞是「供参考，不得据此推断本次需求」，
        # 而不是「用户偏好如下」。后者会让模型把历史偏好当成已确认的本次
        # 要素，直接跳过追问 —— 这正是「个性化」最容易翻车的地方。
        sections.append(
            "## 用户历史偏好（仅供参考）\n"
            "以下是该用户的历史出差习惯，**仅作参考**，"
            "不得据此推断本次出差的具体要素；本次需求以用户当前所述为准。\n"
            f"{profile_summary.strip()}"
        )

    dynamic = f"{MARKER_BEGIN}\n" + "\n\n".join(sections) + f"\n{MARKER_END}"

    # ⚠️ 基础 prompt 为空时不要留下前导空行。
    if not stripped:
        return dynamic
    return f"{stripped}\n\n{dynamic}"


__all__ = [
    "MARKER_BEGIN",
    "MARKER_END",
    "MAX_MISSING_PROMPTS",
    "build_system_prompt",
    "describe_known_slots",
    "strip_managed_sections",
    "user_data_excerpt",
]
