# -*- coding: utf-8 -*-
"""**快慢车道**规则引擎：一句话进来，判定走快车道还是慢车道。

文件职责：
    实现 ``docs/博客原文-Alibaba-Business-Travel.md`` 第 137-160 行描述的
    分流机制 —— 「为我规划行程」这类**界面按钮文案或简短指令**不需要大模型
    分析，直接用规则命中并路由；其余走慢车道交给意图识别智能体做语义理解。

上下游依赖：
    - 上游：:mod:`src.domain`（意图、车道、智能体名册、路由模型）。
    - 下游：``src/orchestration/lane.py`` 的 ``LaneRouterMiddleware``
      （把本模块的判定变成真正的短路）、``src/server/`` 的接线。

═══ 快车道省下的到底是什么 ═══

不是「省一次网络往返」这么简单。慢车道的完整代价是：

1. 一次意图识别模型调用（输入含系统提示词 + 全部意图定义 + 用户输入）；
2. 一次结构化输出的**强制工具调用**往返（框架要注册一个工具逼模型按 schema 输出，
   见 :mod:`src.domain.schemas` 的模块文档）；
3. 这段往返里的**延迟** —— 用户点了个按钮，却要等两秒。

而「为我规划行程」这句话里**没有需要理解的信息**：意图是确定的，槽位是空的，
多意图不可能。用大模型去分析它，是拿一个概率模型去解一个确定性方程。
快车道的意义就在这：把确定性从概率模型手里拿回来。

═══ 三条设计决定，以及为什么 ═══

**一、只做精确匹配（归一化后），不做包含匹配。**
    「包含」看起来能提高命中率，实际是把误判风险引进来：一句
    「上次那个订单为什么取消了」包含「取消订单」的语素，但它是个**询问**，
    不是取消指令 —— 快车道若命中它，就会跳过语义理解直接去执行取消。
    快车道的错误比慢车道贵得多：慢车道最多多花一次模型调用，快车道会
    **做错事**。宁可漏，不可错。

**二、归一化只做「去客套与语气词」，不做语义化简。**
    剥掉「帮我 / 请 / 吧 / 一下」这类**不携带信息**的成分，让
    「请帮我规划行程吧」能与「规划行程」对上；但绝不剥「然后 / 顺便」这类
    可能改变语义的连接词 —— 剥了之后「规划行程然后顺便订个酒店」会退化成
    「规划行程」，而那句话其实是多意图，正该走慢车道。

**三、问句一律走慢车道。**
    见 :func:`_is_question`。这是**安全方向**上的保守：把问句误判为命令，
    会导致系统执行用户只是在打听的操作。

⚠️ 快车道 **不等于** 免确认。危险操作（取消、提交申请）在快车道上**更**需要
   人工确认 —— 因为快车道少了意图识别那一层语义校验，规则表是唯一的把关。
   确认由框架的 HITL / 权限链路负责，不在本模块。
"""

from __future__ import annotations

from dataclasses import dataclass

from src.domain import AgentName, Intent, LaneName, RouteDecision, TripStage

# ==============================================================================
# 一、意图 → 该调哪些智能体
# ==============================================================================
#: 意图到目标智能体的**唯一**映射表。
#:
#: ⚠️ 这张表是快慢两条车道**共用**的，这正是它存在的意义。快车道命中规则后
#: 用它的意图查这张表拿目标；慢车道则由意图识别智能体给出意图、再查同一张表。
#: 若各写一份，两条车道对同一个意图就会给出不同的目标 —— 而这种不一致极难
#: 发现，因为两条车道各自测起来都是对的。
#:
#: ⚠️ 值用元组而字段类型是 ``list[str]``：这里要的是「不可变的定义」，
#: 建 :class:`RouteDecision` 时才转成 list。定义可变的话，
#: 某个调用方「顺手 append 一下」就会永久污染全局路由表。
_INTENT_TARGET_AGENTS: dict[Intent, tuple[AgentName, ...]] = {
    Intent.PLAN_TRIP: (AgentName.MAIN_PLAN,),
    Intent.APPLY_APPROVAL: (AgentName.APPROVAL,),
    Intent.QUERY_POLICY: (AgentName.POLICY_RAG,),
    Intent.QUERY_ORDER: (AgentName.ORDER_QUERY,),
    Intent.MODIFY_TRIP: (AgentName.MAIN_PLAN,),
    Intent.CANCEL: (AgentName.MAIN_PLAN,),
    # 闲聊不调任何专门智能体 —— 由主智能体直接回应。
    Intent.CHITCHAT: (AgentName.MAIN_PLAN,),
    # ⚠️ OTHER 刻意也指向主规划智能体：它是「兜底」而非「无人应答」。
    # 指向空列表的话，用户会收到一条没有任何智能体处理的回复。
    Intent.OTHER: (AgentName.MAIN_PLAN,),
}


# ==============================================================================
# 二、快车道规则表
# ==============================================================================
@dataclass(frozen=True)
class FastLaneRule:
    """一条快车道规则。

    Attributes:
        name: 规则名。会写进 :attr:`RouteDecision.matched_rule` 并进日志 ——
            「为什么这轮走了快车道」这个问题的答案就是它。
        intent: 命中后判定的意图。
        phrases: 触发短语（**归一化之后**做精确相等比较，见 :func:`_normalize`）。
        reason: 面向用户的一句话说明（进思考链展示）。
    """

    name: str
    intent: Intent
    phrases: tuple[str, ...]
    reason: str


#: 快车道规则表 —— **本项目唯一的快车道调优面**。
#:
#: ⚠️ 每条短语都必须是**已经归一化过**的形态（无标点、无客套前缀、无语气词），
#: 并且 ``_normalize(phrase) == phrase``。例如写「规划行程」而不是
#: 「请帮我规划行程吧」—— 后者永远匹配不上，因为输入归一化之后
#: 「请帮我」和「吧」都已经不在了。
#:
#: ⚠️ 这是新增规则时**最容易犯**的错，而且症状极具误导性：规则明明写了却不
#: 生效，人会以为是匹配逻辑坏了，去翻 :func:`classify`；而真正的问题在于
#: 那条短语是**死的**——它看起来像覆盖率，实际一次也命中不了。
#: ``tests/test_orchestration_classifier.py`` 用一条遍历断言把这个不变式钉死。
#:
#: ⚠️ 刻意**不**收录「为我规划行程」这类未归一化的博客按钮原文：它会归一化成
#: 「规划行程」而与本表重复，成为一条永远走不到的冗余项。博客按钮文案确实
#: 必须走快车道，但那件事由**用例**保证（见测试文件里的
#: ``test_blog_button_texts_route_to_the_fast_lane``）—— 用一条可执行的断言
#: 来记录，比用一条匹配不上的冗余短语来「展示」要可靠得多。
FAST_LANE_RULES: tuple[FastLaneRule, ...] = (
    FastLaneRule(
        name="plan_trip",
        intent=Intent.PLAN_TRIP,
        phrases=(
            "规划行程",
            "开始规划",
            "规划出差行程",
            "制定行程",
            "安排行程",
            "出差规划",
            "规划一下行程",
            # 「我要出差」归一化后就是「出差」—— 写前者会是一条死短语。
            "出差",
        ),
        reason="识别为规划行程指令，直接进入事项收集",
    ),
    FastLaneRule(
        name="apply_approval",
        intent=Intent.APPLY_APPROVAL,
        phrases=(
            "提申请",
            "提交申请",
            "发起申请",
            "出差申请",
            "提交出差申请",
            "申请出差",
            "走审批",
            "提交审批",
        ),
        reason="识别为提交出差申请指令，直接进入申请流程",
    ),
    FastLaneRule(
        name="query_policy",
        intent=Intent.QUERY_POLICY,
        phrases=(
            "查政策",
            "查询政策",
            "差旅政策",
            "差旅标准",
            "报销标准",
            "报销规则",
            "报销制度",
            "住宿标准",
            # ⚠️ 这里**刻意没有**「能报多少」「能报吗」这类问句短语。
            #
            # 它们曾经被写在这张表里，而那是错的 —— 问句判定在规则匹配
            # **之前**执行（见 :func:`classify`），任何含疑问词的输入都会
            # 先被判成「问句」直接走慢车道。所以这类短语是**死的**：
            # 归一化稳定性检查放它过去（「能报多少」归一化后确实还是
            # 「能报多少」），却一次也命中不了。
            #
            # 这正是 FAST_LANE_RULES 上方警告的那种错误，而且它**真的发生了
            # 一次**。现在由 tests/test_orchestration_classifier.py 的
            # ``test_every_rule_phrase_actually_fires`` 守着。
            #
            # 而「能报多少」走慢车道本身也是**正确**的行为：它是一个需要
            # 结合上下文（哪个城市？什么舱位？）才能回答的问题，本就该交给
            # 意图识别智能体去理解，而不是靠短语精确匹配。
        ),
        reason="识别为差旅政策查询，交给政策问答智能体",
    ),
    FastLaneRule(
        name="query_order",
        intent=Intent.QUERY_ORDER,
        phrases=(
            "查订单",
            "查询订单",
            "订单",
            "订单状态",
            "订单详情",
            # 「我的订单」「我的申请」归一化后分别是「订单」「申请」。
            "申请",
            "申请进度",
            "审批进度",
        ),
        reason="识别为订单查询指令，交给订单查询智能体",
    ),
    FastLaneRule(
        name="modify_trip",
        intent=Intent.MODIFY_TRIP,
        phrases=(
            "修改行程",
            "改行程",
            "调整行程",
            "改签",
            "换酒店",
            "改时间",
            "修改方案",
        ),
        reason="识别为行程修改指令，进入方案调整",
    ),
    FastLaneRule(
        name="cancel_trip",
        intent=Intent.CANCEL,
        # ⚠️ 取消类操作即使是快车道，**也必须经过人工确认**（见模块文档末段）。
        # 这里只负责「不浪费一次模型调用去理解『取消订单』四个字」。
        phrases=(
            "取消",
            "取消行程",
            "取消订单",
            "取消申请",
            "取消出差",
            "退票",
        ),
        reason="识别为取消指令，将先与您确认后再执行",
    ),
)


# ==============================================================================
# 三、文本归一化
# ==============================================================================
#: 需要剥掉的**前置客套语**，按长度从长到短排列。
#:
#: ⚠️ 顺序不能乱：必须长的在前。否则「帮我规划行程」会先被「我」匹配，
#: 剥成「帮规划行程」—— 剩下的「帮」再也没机会被剥掉，规则就永远命不中。
#: 这类 bug 的表现是「有些说法能命中、有些不能」，很容易被当成「用户说法太怪」。
_POLITE_PREFIXES: tuple[str, ...] = (
    "请帮我",
    "麻烦",
    "帮我",
    "帮忙",
    "给我",
    "我的",
    "我要",
    "我想",
    "为我",
    "请",
    "我",
)

#: 需要剥掉的**后置语气词**，同样按长度从长到短排列。
#:
#: ⚠️ 刻意**不含**单个「下」与「的」：
#:   * 「下」会毁掉「下单」「下载」；
#:   * 「的」会毁掉「目的」。
#: 只收「一下」这个完整的词。收单字看似提高命中率，实际是在**改写用户的话**，
#: 而后缀剥离一旦过度，长句子可能被削成一条无关的触发短语。
_PARTICLES: tuple[str, ...] = (
    "一下",
    "吧",
    "呢",
    "啊",
    "哦",
    "呀",
    "嘛",
    "了",
    "呗",
)

#: 需要**整体删除**的标点（中英文）。
#:
#: ⚠️ 问号也在删除列表里 —— 因为「这是不是一个问句」在
#: :func:`_is_question` 里**先于**归一化判定，用的是原始文本。
#: 若把问号判定放到归一化之后，问号已经被删掉，所有问句都会被当成命令。
_PUNCTUATION = "，。！？、；：\"'（）《》【】,.!?;:\"'()<>[]{}~·—-_/\\%*#@&+=|"

#: 问句标记（在**原始文本**上检查）。
#:
#: ⚠️ 收得比较宽（含「呢」这种语气助词）是**刻意**的：这里误判成问句只会
#: 让一个本来能走快车道的请求改走慢车道 —— 代价是多一次模型调用；
#: 而漏判的代价是**把询问当成命令执行**。两侧代价不对称，所以往安全侧偏。
_QUESTION_MARKERS: tuple[str, ...] = (
    "？",
    "?",
    "怎么",
    "如何",
    "为什么",
    "为何",
    "是不是",
    "是否",
    "能不能",
    "可不可以",
    "多少",
    "哪",
    "吗",
    "呢",
    "什么",
)


def _strip_affixes(text: str) -> str:
    """反复剥掉前置客套语与后置语气词，直到不再变化。

    Args:
        text (`str`): 已去标点与空白的文本。

    Returns:
        `str`: 剥完的结果。

    ⚠️ 必须**循环**而不是各剥一次：「请帮我规划行程吧」一次循环只能剥掉
    「请帮我」与「吧」其中之一（取决于先剥哪边），剥完一边后另一边才暴露
    出来。循环到不动点，才能处理「我的订单呢」这种前后都带成分的输入。
    """
    changed = True
    while changed:
        changed = False
        for prefix in _POLITE_PREFIXES:
            if text.startswith(prefix) and len(text) > len(prefix):
                text = text[len(prefix) :]
                changed = True
                break
        for particle in _PARTICLES:
            if text.endswith(particle) and len(text) > len(particle):
                text = text[: -len(particle)]
                changed = True
                break
    return text


def _normalize(text: str) -> str:
    """把用户输入归一化成可与 :data:`FAST_LANE_RULES` 直接比较的形态。

    处理顺序（顺序本身是逻辑的一部分）：

    1. 转小写、去掉全部空白 —— 让「提交 申请」与「提交申请」等价；
    2. 删掉全部标点 —— 让「提交申请！」与「提交申请」等价；
    3. 循环剥掉客套前后缀（见 :func:`_strip_affixes`）。

    Args:
        text (`str`): 用户原始输入。

    Returns:
        `str`: 归一化结果。**不做**任何语义化简 —— 见模块文档第二条设计决定。

    ⚠️ 本函数**可能返回空串** —— 当输入里没有任何内容字符时（``"   "``、
    ``"。。。"`` 都是）。这是如实描述，不是缺陷：返回值忠实反映了「这串输入
    里没有任何可比较的内容」。

    ⚠️ 但「空串不得命中规则」这条保证**不在这里**，而在 :func:`classify`
    里有一道显式拦截。原因：把保证放在这里，就得让本函数返回某种哨兵值或
    抛异常，从而把「如何表示空」这个决定强加给所有调用方；而拦截
    「内容为空 → 绝不走快车道」本来就是**路由决策**的一部分。
    两条互补的边界：
      * :func:`_strip_affixes` 的 ``len(text) > len(prefix)`` 保住的是
        **剥客套词**不会把「帮我」剥成空；
      * :func:`classify` 的空串判定挡住的是**剥标点空白**之后为空。
    我写第一版时只想到前者，于是断言「永不返回空串」，被用例抓了个正着。
    """
    cleaned = "".join(ch for ch in text.lower() if ch not in _PUNCTUATION and not ch.isspace())
    return _strip_affixes(cleaned)


def _is_question(text: str) -> bool:
    """判断原始输入是否为问句。

    Args:
        text (`str`): 用户原始输入（**未归一化**）。

    Returns:
        `bool`: 含任一问句标记返回 True。

    ⚠️ 必须在原始文本上判定：归一化会删掉问号，那之后就再也分不出
    「取消订单」与「取消订单？」了 —— 而后者是在**问**能不能取消。
    """
    return any(marker in text for marker in _QUESTION_MARKERS)


# ==============================================================================
# 四、对外的两个入口
# ==============================================================================
def classify(
    text: str,
    *,
    enabled: bool = True,
    stage: TripStage | None = None,
    max_chars: int = 20,
) -> RouteDecision:
    """**第一跳**：判定本轮走快车道还是慢车道。

    Args:
        text (`str`): 用户原始输入。
        enabled (`bool`): 快车道**总开关**。为 False 时无条件走慢车道。
            对应配置项 ``orchestration.fast_lane_enabled``。
        stage (`TripStage | None`): 当前对话阶段。⚠️ 只用于**丰富 reason
            文案**，不参与判定 —— 见下方说明。
        max_chars (`int`): 归一化后允许走快车道的最大长度。对应配置项
            ``orchestration.fast_lane_max_chars``。

    Returns:
        `RouteDecision`: 快车道命中时含 ``matched_rule`` 与目标智能体；
        未命中时 ``lane=SLOW``、``target_agents=[intent]``。

    ⚠️ 三个开关都做成**参数**而不是在这里读配置：本模块是纯函数
    （见模块文档末段「快车道不等于免确认」上方的性质说明），
    「哪个配置项驱动哪个参数」这个决定放在**装配处**，而不是散在这里。
    好处是这一层可以完全脱离配置被穷举测试，而配置的默认值有它自己的测试。

    ⚠️ ``enabled=False`` 会跳过**全部**判定，包括问句判定与空输入判定 ——
    不是因为那些判定不重要，而是因为关了快车道之后「走哪条车道」这个问题
    已经没有悬念，再算一遍只是浪费。返回的 ``reason`` 会写明是被开关关掉的，
    免得日志里看起来像「规则没命中」。

    ⚠️ ``stage`` 为什么不参与判定：阶段影响的是**慢车道内部怎么组织提示词**
    （见 ``src/orchestration/prompt.py``），而不是「这句话该不该用大模型读」。
    把阶段塞进快车道判定，会让同一句话在不同阶段走不同车道 —— 这种「上下文
    相关的路由」是排查噩梦：用户复现不了，日志里也看不出规律。
    留作参数是为了把它写进 ``reason``，让日志能回答「这轮是什么阶段」。

    ⚠️ ``max_chars`` 是**第二道网**，不是主要机制。主要机制是精确匹配本身：
    归一化只删前后缀、不压缩句子，所以长句不会退化成一短串。
    它的真正用途是挡住**未来给归一化加规则**时引入的意外塌缩
    （比如有人加了「截断到第一个逗号」）—— 那时这条限制会立刻拦住长句。
    默认 20 远大于现有全部触发短语（最长 8 字），不会误伤。
    """
    if not enabled:
        return _slow_lane(reason="快车道已被配置关闭（orchestration.fast_lane_enabled）", stage=stage)

    if _is_question(text):
        return _slow_lane(
            reason="输入是问句，交由意图识别智能体理解",
            stage=stage,
        )

    normalized = _normalize(text)
    if not normalized:
        # ⚠️ 空输入**必须**显式拦在这里，不能指望「规则表里没有空短语」这个巧合
        # 来兜住它。一旦有人从配置里读进一条空短语（配置里一个空字符串很容易
        # 出现），空输入就会命中它 —— 而「用户什么都没说，系统却判定出意图并
        # 去执行」是最荒谬的一类故障。见 _normalize 的 docstring。
        return _slow_lane(reason="输入为空，交由意图识别智能体处理", stage=stage)

    if len(normalized) > max_chars:
        return _slow_lane(
            reason=f"输入较长（归一化后 {len(normalized)} 字），交由意图识别智能体理解",
            stage=stage,
        )

    for rule in FAST_LANE_RULES:
        if normalized in rule.phrases:
            return RouteDecision(
                lane=LaneName.FAST,
                intent=rule.intent,
                matched_rule=rule.name,
                target_agents=[a.value for a in _INTENT_TARGET_AGENTS[rule.intent]],
                reason=_with_stage(rule.reason, stage),
            )

    return _slow_lane(
        reason=f"未命中快车道规则（归一化后为「{normalized}」）",
        stage=stage,
    )


def route_for_intent(
    intent: Intent,
    *,
    matched_rule: str = "",
    reason: str = "",
    stage: TripStage | None = None,
) -> RouteDecision:
    """**第二跳**：慢车道拿到意图识别结果后，决定调哪些智能体。

    这是 :func:`classify` 的配对入口。慢车道的流程是两跳：

    ``classify() → [SLOW, intent]`` → 意图识别智能体 → ``route_for_intent()``

    Args:
        intent (`Intent`): 意图识别智能体判定的主意图。
        matched_rule (`str`): 规则名；慢车道通常为空。保留该参数是为了让
            调用方在「慢车道但依据某条业务规则改判」时也能留下痕迹。
        reason (`str`): 决策依据。
        stage (`TripStage | None`): 当前阶段，只用于丰富文案。

    Returns:
        `RouteDecision`: ``lane=SLOW``，目标智能体由
        :data:`_INTENT_TARGET_AGENTS` 决定。

    ⚠️ 返回的车道**恒为 SLOW**，即使意图看起来很简单。这是刻意的：
    「这一轮走没走过大模型」是**事实**，不该因为结果看起来简单就改写它。
    若在这里返回 FAST，指标与日志里的快车道占比就会失真 —— 而那个数字
    正是评估快车道规则覆盖率时唯一的依据。
    """
    return RouteDecision(
        lane=LaneName.SLOW,
        intent=intent,
        matched_rule=matched_rule,
        target_agents=[a.value for a in _INTENT_TARGET_AGENTS[intent]],
        reason=_with_stage(reason or f"由意图识别判定为「{intent.display_name}」", stage),
    )


def target_agents_for(intent: Intent) -> list[str]:
    """查某个意图对应的目标智能体。

    Args:
        intent (`Intent`): 意图。

    Returns:
        `list[str]`: 智能体名列表（**新**列表，可安全修改）。

    ⚠️ 每次返回新列表：直接返回 :data:`_INTENT_TARGET_AGENTS` 里的元组
    转出来的共享对象，会让调用方的 append 污染全局路由表。
    """
    return [a.value for a in _INTENT_TARGET_AGENTS[intent]]


# ==============================================================================
# 五、内部工具
# ==============================================================================
def _slow_lane(reason: str, stage: TripStage | None) -> RouteDecision:
    """构造「走慢车道，先叫意图识别智能体」的判定。

    ⚠️ 慢车道的目标智能体是 :attr:`AgentName.INTENT` 而不是主规划智能体：
    这是**诚实的**描述 —— 慢车道上第一个被调用的确实是意图识别智能体，
    之后调谁要看它的输出。若这里直接写主规划智能体，思考链上就会显示
    「正在规划行程」，而实际上系统还在识别意图。
    """
    return RouteDecision(
        lane=LaneName.SLOW,
        # ⚠️ 慢车道的意图此刻**还不知道**，先用 OTHER 占位，由
        # route_for_intent() 用真实结果覆盖。不写 None 是因为
        # RouteDecision.intent 是必填字段 —— 让「还没判定」有个显式取值，
        # 比让字段可选、下游到处判空要好。
        intent=Intent.OTHER,
        matched_rule="",
        target_agents=[AgentName.INTENT.value],
        reason=_with_stage(reason, stage),
    )


def _with_stage(reason: str, stage: TripStage | None) -> str:
    """把当前阶段附在 reason 末尾（便于日志定位）。

    Args:
        reason (`str`): 原始说明。
        stage (`TripStage | str | None`): 当前阶段。⚠️ 允许传字符串 ——
            调用方从 ``middle_context`` 里读出来的往往是序列化过的
            ``stage.value``（见 :mod:`src.orchestration.lane`）。

    Returns:
        `str`: 附了阶段的说明；阶段无法识别时原样返回。

    ⚠️ 只在**有值**时才追加：给 None 也拼一句「阶段：None」，会让日志里
    出现大量无意义的字串，反而淹没了真正有阶段信息的那几条。

    ⚠️ 本函数**永不抛异常**。它是 ``classify`` 的一部分，而 ``classify``
    在每一次模型调用的路径上 —— 为了让日志多一句备注而让整轮对话挂掉，
    这个交换在任何情况下都不划算。所以这里用 ``getattr`` 取 ``value``
    而不是写 ``stage.value``：裸字符串走 ``str(stage)`` 分支，未知对象
    也只会退化成一个普通字符串。
    """
    if stage is None:
        return reason
    value = getattr(stage, "value", None)
    if value is None:
        # 裸字符串（或别的什么）——直接用它自己，同样能给人看。
        value = str(stage)
    return f"{reason}（当前阶段：{value}）"


__all__ = [
    "FAST_LANE_RULES",
    "FastLaneRule",
    "classify",
    "route_for_intent",
    "target_agents_for",
]
