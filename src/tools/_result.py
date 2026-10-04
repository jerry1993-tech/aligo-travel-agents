# -*- coding: utf-8 -*-
"""工具**返回值契约** —— 所有差旅工具统一用这里构造返回值。

文件职责：
    把「工具返回什么」这件事收敛到一处，让工具的实现各自只关心业务。

上下游依赖：
    - 上游：``agentscope.tool.ToolChunk``、``agentscope.message``。
    - 下游：``src/tools/`` 的每个工具模块、``web/`` 前端（按同一份契约解析）。

═══ 为什么工具不能随便返回字符串 ═══

已核实的框架行为（``tool/_adapters.py:176-192``）：工具函数的返回值会被
**归一化**——

    ToolChunk      → 原样使用
    str            → 包成一个 TextBlock
    dict / list    → ``json.dumps(..., ensure_ascii=False)`` 后塞进
                     **一个 TextBlock**（不是 DataBlock！）
    其它           → ``str(result)``

也就是说「返回 dict 就能得到结构化数据」是**错的**，只会得到一段 JSON
文本。这本身不致命，但它意味着**没有任何地方强制统一格式**：每个工具各自
拼字符串，很快就会出现「有的返回 ``{"ok": true}``、有的返回 ``成功``、
有的返回中文散文」。前端要按工具名写渲染逻辑，格式不统一就得逐个适配，
而新增工具时没有任何机制提醒作者该遵循什么。

本模块就是那个机制。

═══ 统一格式 ═══

所有工具返回一个 :class:`ToolChunk`，其内容是一个 TextBlock，文本是这段
JSON（``ensure_ascii=False``）::

    {
      "ok": true,
      "summary": "找到 5 个从杭州到北京的选项",   # 给人看的一句话
      "card": "transport_options",              # 前端卡片类型；无卡片时为空串
      "items": [ ... ],                          # 卡片数据；无卡片时为 null
      "needs": ["出发城市"]                       # 仅 needs_input 时出现
    }

**``summary`` 与 ``card`` 的分工**是这份契约的核心：

- ``summary`` 是给**模型**和**用户**看的自然语言。模型靠它组织最终答复，
  用户在纯文本降级（比如命令行工具）里靠它读懂结果。
- ``card`` 是给**前端**看的机器可读标记。前端见到 ``transport_options``
  就渲染行程卡片，见不到就退回默认渲染。

⚠️ 两者**都要有**，不能只留一个。只留 ``card``，模型拿到一串没有语义的
JSON，会开始瞎猜字段含义；只留 ``summary``，前端只能渲染纯文本，
「自定义行程卡片」这个验收项就无从谈起。

═══ 为什么错误也走同一个格式 ═══

已核实：工具内抛出的异常会被 ``call_tool`` 吞掉，转成
``ToolChunk(state=ERROR, text=str(e))`` 交给模型（``tool/_toolkit.py:356-372``）。
这意味着**抛异常不等于失败得更严重**，只是拿到一段没有格式的英文异常文本。
用户会看到「Error: division by zero」这种东西。

所以工具应当**自己捕获可预期的失败**，用 :func:`error_chunk` 给一句中文
说明。真正无法预期的问题（比如仓储实现崩了）才让它抛出去 —— 那时框架的
兜底至少还能保住「一轮回复不中断」。
"""

from __future__ import annotations

import json
import logging

from agentscope.message import TextBlock, ToolResultState
from agentscope.tool import ToolChunk

logger = logging.getLogger(__name__)

#: JSON 载荷里标识前端卡片类型的字段名。
#:
#: ⚠️ 前端按**值**（``transport_options`` 等）而不是按工具名来选渲染器。
#: 按工具名选的问题：同一个工具在不同阶段可能想给不同的卡片（比如
#: ``search_transport`` 在「还没选」时给列表卡片、在「已选定」时给确认卡片），
#: 绑死在工具名上就表达不了。用独立的 ``card`` 字段，工具想给什么卡片
#: 就填什么。
CARD_KEY = "card"

#: 各卡片类型的名称常量。
#:
#: ⚠️ 集中定义、且前端**只认这些值**。写错一个字（``transportOptions``）
#: 的后果是前端静默退回默认渲染 —— 不报错、不告警，只是卡片没了，
#: 排查起来要从「前端为什么没渲染」一路倒推到这个字符串。
CARD_TRANSPORT = "transport_options"
CARD_HOTEL = "hotel_options"
CARD_POLICY = "policy_verdict"
CARD_ORDERS = "order_list"
CARD_APPROVAL = "approval_result"
CARD_ROUTE = "route_decision"


def _dump(payload: dict[str, object]) -> str:
    """把载荷序列化成工具结果文本。

    Args:
        payload (`dict[str, object]`): 要序列化的内容。

    Returns:
        `str`: JSON 文本。

    ⚠️ ``ensure_ascii=False``。默认的 ``True`` 会把中文转成 ``\\uXXXX``——
    合法的 JSON，但模型读到的是转义串不是汉字，而 ``summary`` 那段文案
    本来就是写给模型看的。转义之后它的作用大打折扣。
    """
    return json.dumps(payload, ensure_ascii=False, default=str)


def _text_chunk(text: str, state: ToolResultState) -> ToolChunk:
    """构造单块文本的 ``ToolChunk``。

    ⚠️ ``content`` 必须是**列表**（``tool/_response.py`` 的类型要求），
    传裸字符串会报错。这一条很容易踩：``ToolChunk(content="hi")`` 看起来
    理所当然，实际是类型错误。

    Args:
        text (`str`): 文本内容。
        state (`ToolResultState`): 结果状态。

    Returns:
        `ToolChunk`: 构造好的结果块。
    """
    return ToolChunk(content=[TextBlock(text=text)], state=state)


def ok_chunk(
    summary: str,
    *,
    card: str = "",
    items: list[object] | None = None,
    **extra: object,
) -> ToolChunk:
    """构造一个**成功**的工具结果。

    Args:
        summary (`str`): 一句话中文摘要。**必须是有信息量的**——「查询成功」
            这类没有内容的摘要等于没写，模型只能靠自己编。
        card (`str`): 前端卡片类型；``""`` 表示不需要卡片。
        items (`list[object] | None`): 卡片数据；``None`` 会序列化成 ``null``。
        **extra: 额外的顶层字段，供特定工具补充上下文。

    Returns:
        `ToolChunk`: 状态为 ``SUCCESS`` 的结果块。

    ⚠️ ``items`` 默认为 ``None`` 而不是 ``[]``：两者在 JSON 里是
    ``null`` 与 ``[]``，前端可以据此区分「这个工具不产出列表」和
    「产出了但结果是空的」。都写成 ``[]`` 的话，前端拿到一个空卡片，
    还要额外判断该不该显示 —— 而它无法判断。
    """
    payload: dict[str, object] = {
        "ok": True,
        "summary": summary,
        CARD_KEY: card,
        "items": items,
    }
    payload.update(extra)
    return _text_chunk(_dump(payload), ToolResultState.SUCCESS)


def needs_input_chunk(summary: str, *, missing: list[str]) -> ToolChunk:
    """构造一个「**信息不足，需要用户补充**」的结果。

    ⚠️ 它**不是错误**，状态仍是 ``SUCCESS``。这一点必须说清楚，否则很容易
    被当成失败处理：用户说「帮我订张票」而没说去哪，这不是系统出错，
    而是对话的正常一步。标成 ERROR 会让界面上出现红色报错样式，
    而正确的表现是继续追问。

    ⚠️ 但也不是纯粹的成功 —— 所以要带 ``needs`` 字段。模型据此知道该追问
    什么；前端可以据此高亮输入框。若只给成功不给 ``needs``，模型会看到
    ``ok: true`` 却不知道查询其实没执行，很可能编一个结果出来。

    Args:
        summary (`str`): 面向用户的中文说明（通常是追问话术）。
        missing (`list[str]`): 缺少的要素名。

    Returns:
        `ToolChunk`: 状态为 ``SUCCESS``、带 ``needs`` 字段的结果块。
    """
    return _text_chunk(
        _dump(
            {
                "ok": True,
                "summary": summary,
                CARD_KEY: "",
                "items": None,
                "needs": list(missing),
            },
        ),
        ToolResultState.SUCCESS,
    )


def error_chunk(summary: str, *, detail: str = "") -> ToolChunk:
    """构造一个**失败**的工具结果。

    Args:
        summary (`str`): 面向用户的中文说明（要能指导下一步，如「稍后重试」）。
        detail (`str`): 面向排障的补充信息（写日志 + 进载荷，**不可渲染给用户**）。

    Returns:
        `ToolChunk`: 状态为 ``ERROR`` 的结果块。

    ⚠️ ``summary`` 与 ``detail`` 分开，且**都要**：``summary`` 进用户可见的
    回复，不能包含堆栈或内部标识；``detail`` 用于排查。合成一个字段的话，
    要么用户看到一堆技术细节，要么排查时什么都没有。

    ⚠️ ``detail`` 是**内部标识与异常原文**（``unknown kind='hotel'``、
    ``TimeoutError: ...``），调用方一律写成英文/枚举值，不要写中文句子 ——
    它没有经过任何面向用户的加工。它随载荷一起进模型上下文（模型据此
    判断能不能重试），也随载荷一起进浏览器（工具卡片的数据源）。

    2026-10-03 的对抗审计发现：本函数的 docstring 声称 ``detail``
    「会进日志」，**而实现从来没写过日志** —— 于是唯一能读到 ``detail``
    的地方变成了前端（``shell.tsx`` 用等宽字体把异常原文展示给用户）。
    现在两件事一起修：这里真的写一条 WARNING 日志，
    前端那处渲染也删掉。**改这里时不要只改一半**：详情只该出现在服务端
    日志里，用户可见的位置一律只有 ``summary``。
    """
    logger.warning("工具返回失败：%s（详情：%s）", summary, detail or "无")
    return _text_chunk(
        _dump(
            {
                "ok": False,
                "summary": summary,
                CARD_KEY: "",
                "items": None,
                "detail": detail,
            },
        ),
        ToolResultState.ERROR,
    )


__all__ = [
    "CARD_APPROVAL",
    "CARD_HOTEL",
    "CARD_KEY",
    "CARD_ORDERS",
    "CARD_POLICY",
    "CARD_ROUTE",
    "CARD_TRANSPORT",
    "error_chunk",
    "needs_input_chunk",
    "ok_chunk",
]
