# -*- coding: utf-8 -*-
"""把**无状态专家智能体**包装成 ``FunctionTool``。

文件职责：
    主智能体需要「先想清楚用户要什么」的时候，能像调一个普通工具一样调
    意图识别 —— 而不是绕 Team 机制建一个 worker、等它异步回报。
    设计取舍见 :mod:`src.agents.intent` 的模块文档。

上下游依赖：
    - 上游：:mod:`src.agents.intent`、:mod:`src.tools._result`。
    - 下游：``src/server/agents_factory.py`` 把它加进每个用户的工具集。

═══ ⚠️ 为什么放进 ``src/tools/`` 而不是 ``src/agents/`` ═══

判据是**它是什么**，不是**它包着谁**。这个函数返回的是 ``ToolBase``，
遵守的是 ``src/tools/`` 那一整套约定（返回 ``ToolChunk``、摘要用中文、
成功与失败的状态区分、卡片字段的形状）。放在 ``src/agents/`` 里，
它就成了一条从 agents 包伸进 tools 约定的边，而 tools 层的测试
（``tests/test_tools_*.py``）也就照不到它了。

═══ ⚠️ 别把专家的输出「翻译」成自然语言 ═══

工具结果是给**模型**看的，不是给用户看的。这里刻意返回结构化的 JSON，
而不是「我判断你想规划行程」这样一句中文 —— 后者会诱使主智能体
直接把它复述给用户，于是用户看到一句「系统判断你的意图是…」，
那是内部实现泄漏。

结构化结果还有一个好处：主智能体可以据此**继续决策**（比如意图是
``QUERY_POLICY`` 就先去检索政策），而不是把一段散文当结论。
"""

from __future__ import annotations

from agentscope.tool import FunctionTool, ToolBase, ToolChunk

from src.agents.intent import IntentRecognizer
from src.tools._result import error_chunk, ok_chunk

#: 意图识别工具的注册名。
#:
#: ⚠️ 抽成常量，因为「函数名」（``FunctionTool`` 默认拿它当工具名）、
#: 本常量、以及 ``tests/test_tools_expert.py`` 的断言三处必须一致。
#: 硬写字面量的后果是改名时漏一处，而漏掉的那处不报错，
#: 只是模型在需要它的时候「找不到这个工具」。
#:
#: ⚠️ 这段注释曾经还写着「主智能体的提示词里也提到了这个名字，是三处硬写」——
#: 那是**错的**：``src/agents/prompts.py`` 的 ``MAIN_PLAN_PROMPT`` 从头到尾
#: 没有出现过 ``recognize_intent``（全仓库 grep 只有本文件的三处）。
#: 记在这里是因为「提示词里有」这个说法会把人引去做一次注定无果的 grep，
#: 而真正需要同步的只有函数名与常量。
INTENT_TOOL_NAME = "recognize_intent"


def build_intent_tool(recognizer: IntentRecognizer) -> ToolBase:
    """把意图识别器包装成一个工具。

    ⚠️ 工具**是只读的**（``is_read_only=True``）。

    这不是「因为它不写数据库所以顺手标上」，而是权限引擎的硬性要求：
    每次调用都弹一次人工确认的话，用户点「规划行程」要先确认一次
    「正在识别你的意图」，产品没法用。只读工具走引擎的快速通道
    （``agentscope/permission/_engine.py:659-687``，在**所有** ``PermissionMode`` 下
    自动放行），详见 ``src/tools/travel.py`` 的说明。

    Args:
        recognizer (`IntentRecognizer`): 意图识别器。
            ⚠️ 它内部**每次调用都新建 Agent**（见模块文档），所以这里
            持有单个实例是安全的。

    Returns:
        `ToolBase`: 包装好的工具。
    """

    async def recognize_intent(text: str) -> ToolChunk:
        """识别用户这句话想做哪些事，返回结构化拆解。

        当你需要判断用户到底想干什么（尤其是他一句话里说了好几件事）
        的时候调用本工具。不要在每轮对话都调用它 —— 只有当你对用户的
        意图不确定，或者需要把一句话拆成几件事分头处理时才用。

        Args:
            text (str): 用户的原话。原样传入，不要改写、不要省略 ——
                改写会让识别器看不到用户真实的措辞与错别字，
                而那恰恰是判断意图的关键信息。

        Returns:
            ToolChunk: 结构化识别结果，含意图列表、要素、以及是否需要追问。
                需要追问时 ``needs_clarification`` 为 true，
                追问的话题在 ``clarification_question`` 里。
        """
        clean = (text or "").strip()
        if not clean:
            # ⚠️ 空输入是**调用方**的问题（模型没传 text），不是用户的问题。
            # 用 error 而不是 needs_input：needs_input 的语义是「用户还没说
            # 清楚，去问用户」，而这里该问的是模型自己。
            return error_chunk(
                "没有收到要识别的文本，请把用户的原话传进来。",
                detail="empty text",
            )

        result = await recognizer.recognize(clean)

        summary = _summarize(result)
        return ok_chunk(
            summary,
            card="",
            items=[_dump_decision(item) for item in result.intents],
            **{
                # ⚠️ 这几个额外字段是**前端契约**的一部分（前端读它们渲染
                # 「显示推理」区块），字段名改动要前后端一起动。
                "reasoning": result.reasoning,
                "rewritten_query": result.rewritten_query,
                "needs_clarification": result.needs_clarification,
                "clarification_question": result.clarification_question,
            },
        )


    tool = FunctionTool(recognize_intent, is_read_only=True)
    # ⚠️ 断言注册名与常量一致。``FunctionTool`` 默认用**函数名**当工具名，
    # 于是「改函数名」与「改常量」两件事必须同步 —— 而它们之间没有任何
    # 编译期联系。这里在构造时立刻检查，把一个静默的「模型找不到工具」
    # 变成一次启动即失败的报错。
    if tool.name != INTENT_TOOL_NAME:
        raise ValueError(
            f"意图识别工具注册名是 {tool.name!r}，与常量 INTENT_TOOL_NAME "
            f"({INTENT_TOOL_NAME!r}) 不一致。请同步修改函数名或常量。",
        )
    return tool


def _summarize(result: object) -> str:
    """给识别结果写一句话摘要。

    ⚠️ 摘要是**给模型看**的，不是给用户看的，所以它写成
    「识别到 2 个意图：PLAN_TRIP、QUERY_POLICY」这种**工程化**的措辞，
    而不是「你想规划行程并查询政策」。后者会被模型当成成品直接复述出去
    —— 见模块文档。

    ⚠️ 空意图那条分支**在当前的识别器下走不到**，这是刻意的防御而
    不是没想清楚：``IntentRecognizer.recognize`` 的 ``_normalize`` 在
    意图为空时会走 ``_degrade``，而 ``_degrade`` 恒定返回**一条**
    ``OTHER`` 意图（``src/agents/intent.py:305-310`` 与 ``:368-378``）。
    所以生产路径上 ``intents`` 至少有一条。

    留着它，是因为本函数的参数是**鸭子类型**的 ``object``
    （见签名）—— 换个识别器实现、或将来 ``_degrade`` 改成返回空列表，
    这里就是唯一的兜底。写这段是为了让读到的人知道：
    **为它写的那条用例测的是接口契约，不是当前实现的某个真实分支**，
    别因为它「测不出东西」就把这条分支删掉。

    Args:
        result (`object`): :class:`~src.domain.schemas.IntentRecognitionResult`。

    Returns:
        `str`: 摘要文本。
    """
    intents = getattr(result, "intents", []) or []
    if not intents:
        return "没有识别出任何意图。"
    listed = "、".join(
        f"{getattr(item, 'intent').value}"
        f"({getattr(item, 'confidence'):.2f})"
        for item in intents
    )
    suffix = "；需要向用户追问" if getattr(result, "needs_clarification", False) else ""
    return f"识别到 {len(intents)} 个意图：{listed}{suffix}"


def _dump_decision(item: object) -> dict[str, object]:
    """把一条意图判定拍平成字典。

    ⚠️ 不用 ``model_dump()``。pydantic 的 ``model_dump()`` 会把
    :class:`~src.domain.enums.Intent` 序列化成**枚举成员**（在 Python 里
    是 ``str`` 的子类，能直接进 JSON），但一旦将来有人给这些模型加上
    自定义序列化器，这里就会跟着变。手写四个字段，输出形状就是**本文件
    说了算**，前端契约不会因为别处的改动而漂移。

    Args:
        item (`object`): :class:`~src.domain.schemas.IntentDecision`。

    Returns:
        `dict[str, object]`: 拍平后的字典。
    """
    return {
        "intent": getattr(item, "intent").value,
        "confidence": float(getattr(item, "confidence")),
        "slots": dict(getattr(item, "slots", {}) or {}),
        "reason": getattr(item, "reason", "") or "",
    }


__all__ = [
    "INTENT_TOOL_NAME",
    "build_intent_tool",
]
