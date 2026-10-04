# -*- coding: utf-8 -*-
"""零密钥降级模型（``src/llm/mock.py``）的测试。

==============================================================================
为什么 MockLLM 值得认真测
==============================================================================
    很容易把它当成「测试用的假东西，随便写写」。但本项目的定位恰恰相反：
    **MockLLM 是零密钥机器上的默认运行时**。也就是说，一个 clone 下仓库
    直接 ``make up`` 的人，看到的所有回答都来自它。它要是坏了：

      · 前端联调时事件流是空的 ⇒ 会被当成「前端写错了」；
      · ``make smoke`` 会挂在「收不到回复」⇒ 会被当成「服务起不来」；
      · 每个排查方向都指向错误的地方，而真正的缺陷在 mock 里。

    因此这里测的不是「mock 有没有返回文字」，而是它**是否忠实模拟了真模型的
    契约** —— 流式分片、块 id 归并、工具调用的结构、消息角色的对应。
    这些契约一旦被违反，问题不会在 mock 上暴露，而会在**换成真模型之后**暴露 ——
    那时排查成本高得多。
"""

from __future__ import annotations

import pytest

from agentscope.message import Msg, TextBlock
from agentscope.model import ChatResponse

from src.llm.mock import (
    REPLY_PREFIX,
    THINK_DIRECTIVE,
    TOOL_DIRECTIVE,
    MockChatModel,
)


def _user_msg(text: str) -> Msg:
    """构造一条 user 消息。

    ⚠️ ``Msg.content`` 必须是**内容块列表**，不能直接传字符串 ——
    传字符串会被 pydantic 拦下（``Input should be a valid list``）。
    这条约束值得在这里显式封装一次，免得每个用例各写一遍、
    也就各错一遍。

    Args:
        text (`str`): 用户文本。

    Returns:
        `Msg`: 含单个 ``TextBlock`` 的 user 消息。
    """
    return Msg(
        name="user",
        role="user",
        content=[TextBlock(type="text", text=text)],
    )


def _model(**kwargs: object) -> MockChatModel:
    """构造一个默认参数下的 Mock 模型。

    Args:
        **kwargs (`object`): 覆盖默认参数（如 ``stream=False``）。

    Returns:
        `MockChatModel`: 被测模型。
    """
    params: dict[str, object] = {"model": "mock-model"}
    params.update(kwargs)
    return MockChatModel(**params)  # type: ignore[arg-type]


async def _collect(model: MockChatModel, text: str, **kwargs: object) -> list[ChatResponse]:
    """跑一次调用并把流式分片全部收集起来。

    Args:
        model (`MockChatModel`): 模型。
        text (`str`): 用户文本。
        **kwargs (`object`): 传给模型的额外参数（如 ``tools``）。

    Returns:
        `list[ChatResponse]`: 收到的分片（非流式时只有一个）。
    """
    messages = [_user_msg(text)]
    result = await model(messages, **kwargs)  # type: ignore[arg-type]
    if isinstance(result, ChatResponse):
        return [result]
    return [chunk async for chunk in result]


# ==============================================================================
# 一、流式契约
# ==============================================================================
async def test_stream_yields_multiple_chunks_ending_with_is_last() -> None:
    """流式调用必须产出**多个**分片，且最后一个标了 ``is_last=True``。

    ⚠️ 「多个分片」不是可有可无的：只发一个分片的流式实现能通过任何
    「最终内容对不对」的断言，但它**测不出前端的流式渲染**——
    而前端流式渲染正是本项目要联调的东西之一。若 mock 只发一片，
    一个「只在收到第一片时渲染、后续分片被忽略」的前端 bug 会完全测不出来。

    而 ``is_last`` 是流结束的信号，框架的累积器与前端都依赖它。
    缺了它，最后一个分片永远不会被提交。
    """
    chunks = await _collect(_model(stream=True), "帮我订一张去上海的机票")

    assert len(chunks) > 1, f"流式只产出了 {len(chunks)} 个分片，无法验证流式渲染"

    assert chunks[-1].is_last is True, "最后一个分片没有标 is_last —— 流不会结束"
    assert all(chunk.is_last is False for chunk in chunks[:-1]), (
        "中间分片误标了 is_last —— 流会提前结束，后面的内容全部丢失"
    )


async def test_stream_chunks_share_one_block_id() -> None:
    """一个文本块的所有分片必须共用**同一个** block id。

    ⚠️ 这是 MockLLM 里最容易写错、且后果最明显的一处。
    框架的 ``_StreamAccumulator`` 是**按 id 归并**分片的
    （``model/_utils.py::_AccTextBlock.append``）。
    若每个分片各拿一个新 id，累积器会把它们当成 N 个独立的文本块 ——
    最终响应里出现 N 段互相截断的文字，看起来像模型「说话断断续续」。
    而单独看每一个分片，内容都是对的，因此极难定位。
    """
    chunks = await _collect(_model(stream=True), "出差去北京三天")
    text_blocks = [chunk.content[0] for chunk in chunks]

    block_ids = {block.id for block in text_blocks}  # type: ignore[attr-defined]
    assert len(block_ids) == 1, (
        f"一次回复里出现了 {len(block_ids)} 个不同的 block id：{block_ids}。"
        f"累积器会把它们当成多个独立文本块，输出变成 N 段截断的文字。"
    )


async def test_stream_deltas_reassemble_into_the_final_text() -> None:
    """所有分片的 ``delta`` 拼起来，必须等于最终分片的完整内容。

    这条同时验证两件事：
      · 分片确实是**增量**（而不是每片都发全文，那会让累积器输出重复内容）；
      · 最后一片给出的完整内容与增量拼出的结果一致（两者不一致意味着
        前端看到的（增量拼的）与服务端存的（最终内容）是两份不同的文本）。

    ⚠️ 断言用「包含」而不是「相等」是有意的：Mock 的最终分片里可能带上
    额外的收尾（例如提示可用工具），而增量拼出的正文是它的前缀。
    这里要钉的是「增量拼出来的东西**确实在**最终内容里」。
    """
    chunks = await _collect(_model(stream=True), "帮我看看上海的酒店")

    deltas = "".join(
        block.text for chunk in chunks[:-1] for block in chunk.content
    )
    final_text = "".join(block.text for block in chunks[-1].content)

    assert deltas, "增量拼出来是空的 —— 分片里没有 delta 内容"
    assert deltas in final_text, (
        f"增量拼出的内容不在最终内容里：\n增量={deltas!r}\n最终={final_text!r}"
    )


async def test_non_streaming_returns_a_single_response() -> None:
    """非流式调用返回**单个** ``ChatResponse``（而不是一个异步生成器）。

    ⚠️ 返回类型区分不是风格问题：``ChatModelBase.__call__`` 的调用方
    按 ``stream`` 参数决定是 ``async for`` 还是直接取属性。
    返回错了类型，调用方会拿到一个生成器对象当成响应体用 ——
    报错点离根因非常远。
    """
    result = await _model(stream=False)([_user_msg("你好")])

    assert isinstance(result, ChatResponse), f"非流式返回了 {type(result)}"


# ==============================================================================
# 二、确定性 —— 「可复现」是 Mock 存在的意义
# ==============================================================================
async def test_same_input_yields_same_text() -> None:
    """同样的输入 ⇒ 同样的输出。

    若 Mock 的输出带随机性，所有依赖它的用例都会变成偶发失败 ——
    而偶发失败会被当成「环境问题」忽略掉，测试的防线就没了。
    """
    first = await _collect(_model(stream=False), "报销标准是多少")
    second = await _collect(_model(stream=False), "报销标准是多少")

    assert first[0].content[0].text == second[0].content[0].text  # type: ignore[attr-defined]


async def test_reply_is_visibly_marked_as_mock() -> None:
    """回复正文必须带 ``REPLY_PREFIX`` 标记。

    ⚠️ 这不是装饰。排障时最容易走错的一条路是「模型回答得不对」——
    而如果实际跑的是 MockLLM（比如机器上没配 key），
    那么「调 prompt」「换模型」全都白费。
    一个显眼的标记能在第一眼就排除掉整条错误路径。
    """
    chunks = await _collect(_model(stream=False), "帮我订机票")

    assert REPLY_PREFIX in chunks[-1].content[0].text  # type: ignore[attr-defined]


# ==============================================================================
# 三、指令协议（让 Mock 能驱动完整链路）
# ==============================================================================
async def test_think_directive_produces_a_thinking_block() -> None:
    """``#mock-think:`` 指令 ⇒ 产出 ``ThinkingBlock``。

    用途：前端要渲染「思考链」，而思考链只在真模型真的思考时才出现。
    没有这条指令，开发者就无法在零密钥环境下验证思考链渲染 ——
    只能等到配上真 key 之后才发现问题。
    """
    chunks = await _collect(
        _model(stream=False),
        f"{THINK_DIRECTIVE} 先确认一下差旅标准\n帮我订机票",
    )

    block_types = {block.type for block in chunks[-1].content}
    assert "thinking" in block_types, f"没有产出 ThinkingBlock：{block_types}"


async def test_tool_directive_produces_a_tool_call_block() -> None:
    """``#mock-tool:`` 指令 ⇒ 产出 ``ToolCallBlock``，且 ``input`` 是 JSON 字符串。

    ⚠️ ``input`` 必须是**字符串**（JSON 文本）而不是 dict：
    框架下游的 ``FunctionTool`` 会直接 ``json.loads`` 它。
    传 dict 会在工具调用时抛 ``TypeError``，而报错栈指向框架内部，
    与本文件相距甚远。
    """
    import json

    chunks = await _collect(
        _model(stream=False),
        f'{TOOL_DIRECTIVE} search_flights {{"city": "上海"}}',
    )

    tool_blocks = [b for b in chunks[-1].content if b.type == "tool_call"]
    assert len(tool_blocks) == 1, f"没有产出 ToolCallBlock：{chunks[-1].content}"

    payload = tool_blocks[0].input  # type: ignore[attr-defined]
    assert isinstance(payload, str), f"input 应当是 JSON 字符串，得到 {type(payload)}"
    assert json.loads(payload) == {"city": "上海"}


async def test_directive_is_only_recognised_at_line_start() -> None:
    """指令只在**行首**生效，正文里引用它不会误触发。

    ⚠️ 这条防的是一类很现实的误触发：用户在对话里粘贴一段文档，
    而文档里恰好有一行以 ``#mock-tool:`` 开头（比如一篇讲解本项目的文章）。
    若实现用 ``in`` 或非锚定正则匹配，这一轮就会凭空发出一次工具调用 ——
    用户会看到「系统莫名其妙去查了航班」。
    """
    chunks = await _collect(
        _model(stream=False),
        f"这段话里提到了 {TOOL_DIRECTIVE} search_flights，但它不在行首。",
    )

    block_types = {block.type for block in chunks[-1].content}
    assert "tool_call" not in block_types, (
        f"行内的指令被误当成了真指令，凭空发起了一次工具调用：{block_types}"
    )


async def test_unknown_tool_still_produces_a_wellformed_call() -> None:
    """即使工具名不在本次可用工具列表里，也要产出结构合法的调用。

    为什么不让 Mock 去校验工具是否存在：那会让「工具不存在时怎么办」
    这件事由 **Mock** 决定，而真实场景下这个决定应该由框架/业务层做
    （可能是报错、可能是回退）。Mock 只负责产出结构正确的调用，
    让链路的其余部分自己去处理 —— 这样才测得到真实的错误路径。
    """
    chunks = await _collect(
        _model(stream=False),
        f"{TOOL_DIRECTIVE} totally_unknown_tool",
    )

    tool_blocks = [b for b in chunks[-1].content if b.type == "tool_call"]
    assert len(tool_blocks) == 1
    # 没给入参时必须是合法 JSON 而不是空串 —— 下游会直接 json.loads。
    assert tool_blocks[0].input == "{}"  # type: ignore[attr-defined]


async def test_directives_are_read_from_the_last_user_message() -> None:
    """指令只从**最后一条** user 消息里解析。

    ⚠️ 多轮对话下这条很关键：若从所有消息里找，那么第一轮用户发的
    ``#mock-tool:`` 会在后续每一轮里**重复触发**同一个工具调用，
    且行为看起来像「模型卡在某个动作上」。
    """
    messages = [
        _user_msg(f"{TOOL_DIRECTIVE} search_flights {{}}"),
        Msg(
            name="assistant",
            role="assistant",
            content=[TextBlock(type="text", text="好的")],
        ),
        _user_msg("那酒店呢"),
    ]

    result = await _model(stream=False)(messages)
    block_types = {block.type for block in result.content}  # type: ignore[union-attr]

    assert "tool_call" not in block_types, (
        "历史消息里的指令被重复触发了 —— 多轮对话会卡在同一个工具调用上"
    )


# ==============================================================================
# 四、用量统计
# ==============================================================================
async def test_usage_is_reported_and_non_negative() -> None:
    """必须回报 usage，且数值非负。

    ⚠️ 为什么 Mock 也要认真报 usage：token 统计是成本面板的数据源。
    若 Mock 回报 0，那么用 Mock 联调时面板上「每分钟 token 数」恒为 0 ——
    于是面板的接线错误（字段名写错、回调没挂上）在联调阶段完全测不出来，
    要等到接上真模型、开始真实计费时才发现，而那时每一条错数据都是钱。
    """
    chunks = await _collect(_model(stream=False), "帮我订一张去广州的机票")

    usage = chunks[-1].usage
    assert usage is not None, "没有回报 usage —— 成本面板在联调阶段会静默失效"
    assert usage.input_tokens >= 0
    assert usage.output_tokens >= 0


async def test_streaming_reports_usage_in_the_final_chunk() -> None:
    """流式调用必须在**最后一片**给出 usage。

    ⚠️ 中间分片不必带 usage（那时还不知道最终有多少输出），
    但最后一片必须带 —— 否则流式路径下 token 统计永远是空的，
    而**本项目的主链路正是流式的**。这是一个「只在流式下发生」的缺口，
    用非流式的用例测不出来。
    """
    chunks = await _collect(_model(stream=True), "订一张去深圳的票")

    assert chunks[-1].usage is not None, (
        "流式调用的最后一片没有 usage —— 主链路的 token 统计会是空的"
    )


# ==============================================================================
# 五、空输入与边界
# ==============================================================================
@pytest.mark.parametrize("text", ["", "   ", "\n\n"])
async def test_empty_user_text_does_not_crash(text: str) -> None:
    """空的用户输入不能把模型搞崩。

    ⚠️ 这是真实会发生的输入（用户直接按了回车，或前端发了个空消息）。
    崩在这里的代价不只是这一条请求失败：异常会沿着 SSE 链路往上传，
    很可能表现为一个断开的连接 —— 前端看到的是「页面卡住」，
    而不是「你发的消息是空的」。
    """
    chunks = await _collect(_model(stream=False), text)

    assert chunks, "空输入下没有返回任何分片"
    assert chunks[-1].content, "空输入下返回了空内容"


# ==============================================================================
# 六、构造参数校验
# ==============================================================================
def test_zero_chunk_size_is_rejected_at_construction() -> None:
    """``chunk_size=0`` 必须在**构造时**报错。

    ⚠️ 为什么这条值得一个用例：分片循环若按 ``chunk_size`` 步进，
    而它等于 0，循环的推进量就是 0 ⇒ **死循环**。
    症状是「这个请求永远不返回」，连接被挂住直到超时 ——
    和「参数写错了」这件事看起来毫无关系，排查会从网络层开始绕一大圈。

    构造期报错把它变成一条清晰的 ``ValueError``，代价是几行代码。
    """
    with pytest.raises(ValueError):
        MockChatModel(chunk_size=0)


# ==============================================================================
# 五、formatter（P1 的一个真 bug 的回归测试）
# ==============================================================================
# 背景：这一段用例守住的是一个**曾经真实发生过**的缺陷。
#
#   ``MockChatModel`` 最初没有 ``self.formatter``。直接调用模型做任何事都
#   完全正常（``_call_api`` 自己拼消息，不碰 formatter），但一旦被 **Agent
#   驱动**，整条链路就以 ``REPLY_END finished_reason="error"`` 收场 ——
#   没有任何栈、没有任何提示，只有一句「setup 失败」。
#
#   根因在两处，都是「读一下 model.formatter」这种不起眼的动作：
#
#     1. ``agentscope/app/_service/_model.py:67``
#            ``model.formatter.input_types = card.input_types``
#        —— 它包在 ``try/except Exception`` 里，异常只记 **DEBUG**。
#           也就是说缺失的 formatter 在这里被**静默吞掉**，日志默认级别下
#           一个字都看不到。
#
#     2. ``agentscope/agent/_agent.py:2062``
#            ``supported = self.model.formatter.supported_input_media_types``
#        —— 这一次没有被吞，于是 AttributeError 跑到 setup 阶段，
#           变成一句语义完全不同的「智能体初始化失败」。
#
#   两处叠加的效果是：**故障点与表现点隔着两层，中间还有一层静默吞异常**。
#   这正是下面这些断言存在的理由 —— 它们把「formatter 存在且可用」这件事
#   在**没有 Agent、没有服务层**的前提下单独钉住，失败信息直指根因。
def test_model_exposes_a_formatter() -> None:
    """★ ``MockChatModel`` 实例必须带 ``formatter``。

    ``formatter`` **不是** ``ChatModelBase`` 的基类属性 —— 它由每个具体模型
    在自己的 ``__init__`` 里赋值（框架里其余 10 个模型都是
    ``self.formatter = formatter or XxxChatFormatter()``）。
    漏掉它不会在构造时报错，只会在 Agent 驱动时炸，所以必须单独断言。
    """
    from agentscope.formatter import FormatterBase

    model = _model()

    assert hasattr(model, "formatter"), (
        "MockChatModel 没有 formatter —— Agent 驱动时会在 "
        "agent/_agent.py 的 setup 阶段以 AttributeError 失败，"
        "而错误信息看起来像「智能体初始化失败」，与 formatter 毫无关系。"
    )
    assert isinstance(model.formatter, FormatterBase)


def test_formatter_input_types_are_readable_and_writable() -> None:
    """``formatter.input_types`` 可读**也可写**。

    ★ 可写这一半很关键：``app/_service/_model.py`` 在建会话时会执行
      ``model.formatter.input_types = card.input_types``，
      把模型卡里的输入类型盖到实例上。若 ``input_types`` 是只读的
      （比如 formatter 用了 ``frozen=True``），这一句会抛异常 ——
      而它被那段的 ``try/except`` 吞成一条 DEBUG 日志，
      表现为「配置没生效」而不是「配置没生效且报错」。
    """
    model = _model()
    original = model.formatter.input_types

    model.formatter.input_types = ["text/plain", "image/png"]

    assert model.formatter.input_types == ["text/plain", "image/png"]
    assert original != model.formatter.input_types


def test_formatter_reports_supported_input_media_types() -> None:
    """``supported_input_media_types`` 可读 —— 这是 Agent 读的那个属性。

    对齐 ``agent/_agent.py`` 的实际访问路径：它读的**不是**
    ``input_types``，而是这个属性（由 ``FormatterBase`` 提供，
    会把 ``text/plain`` 之类「不算媒体」的类型过滤掉）。
    断言前者通过、后者缺失，是很容易发生的一种半吊子实现。
    """
    model = _model()
    model.formatter.input_types = ["text/plain", "image/png"]

    supported = model.formatter.supported_input_media_types

    assert isinstance(supported, list)
    # text/plain 不是「媒体类型」，必须被过滤掉；image/png 应当保留。
    assert "image/png" in supported
    assert "text/plain" not in supported, (
        "supported_input_media_types 没有过滤掉 text/plain —— "
        "Agent 会据此认为模型能接收文本「作为媒体」，行为可能与预期不符。"
    )


def test_each_model_instance_gets_its_own_formatter() -> None:
    """★ 每个模型实例各有**独立**的 formatter。

    这条防的是一种很自然的写法：把 formatter 做成**类属性**
    （``formatter = MockChatFormatter()`` 写在 class 体里）。
    那样每个实例的 ``input_types`` 覆盖都会写到**同一个对象**上 ——
    于是 A 会话把 input_types 改成 ``["image/png"]`` 之后，
    B 会话的模型也跟着变，且没有任何报错。

    单进程多租户下这是跨会话的状态污染，而且它在单会话测试里**永远测不出来**。
    """
    first = _model()
    second = _model()

    assert first.formatter is not second.formatter, (
        "两个模型实例共用了同一个 formatter 对象 —— "
        "服务层对 input_types 的覆盖会跨会话泄漏。"
    )

    first.formatter.input_types = ["image/png"]

    assert second.formatter.input_types != ["image/png"], (
        "改动一个实例的 input_types 影响到了另一个实例。"
    )


async def test_formatter_format_returns_plain_dicts() -> None:
    """``format()`` 能把 ``Msg`` 列表转成普通字典列表。

    ⚠️ 框架本身并不调用 Mock 的 ``format()``（每个具体模型在自己的
        ``_call_api`` 里拼消息，如 ``_openai_chat/_model.py``），
        所以这条用例不是「在测框架要用的东西」，而是在测**这个类自己
        是否成立** —— 它继承的是抽象基类，一个只满足「属性可读」
        而 ``format`` 直接抛 NotImplementedError 的实现同样能通过上面几条。
    """
    model = _model()

    formatted = await model.formatter.format([_user_msg("你好")])

    assert isinstance(formatted, list)
    assert formatted, "format() 返回了空列表 —— 消息被吃掉了。"
    assert all(isinstance(item, dict) for item in formatted)
    assert formatted[0]["role"] == "user"
