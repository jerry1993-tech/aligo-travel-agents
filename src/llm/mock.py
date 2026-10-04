# -*- coding: utf-8 -*-
"""MockLLM —— 无密钥环境下的**确定性**模型替身。

文件职责：
    在没有配置任何真实模型密钥时（``DASHSCOPE_API_KEY`` 为空）提供一条**能跑通
    全链路**的模型实现，使下面这些场景都不需要一把真 key：

        · 新人 clone 下来首次 ``make up``：容器要能进 healthy，``make smoke`` 要绿；
        · CI 里的 ``make test``：绝不能因为「今天没配额」而红；
        · 前端联调：需要稳定的、可预期的流式输出与思考链事件；
        · 演示与录屏：输出内容固定，不会因为模型抽风而变化。

    「确定性」是它的核心契约：**同样的输入必须得到同样的输出**。因此本模块
    不使用 ``random``、不读系统时间来决定内容（时间戳只出现在框架自动填充的
    ``created_at`` 字段里，那是元数据不是内容）。

上下游依赖：
    - 上游：由 ``src/llm/factory.py::build_chat_model`` 在
      「``use_mock_when_no_key`` 为真 **且** ``api_key`` 为空」时返回。
    - 下游：对上层完全透明 —— 它继承 :class:`agentscope.model.ChatModelBase`，
      对调用方而言与 ``OpenAIChatModel`` 没有任何 API 差异。

------------------------------------------------------------------------------
为什么只实现 ``_call_api``
------------------------------------------------------------------------------
    框架的 ``ChatModelBase.__call__`` 已经把「重试 + 流式累积」做完了
    （``agentscope/model/_base.py:182-290``）：
        · 它按 ``max_retries`` 重试 **可重试异常**；
        · 它用一个 ``_StreamAccumulator`` 把 ``is_last=False`` 的分片
          按 block id 归并成完整响应，并在需要时补一个收尾响应。

    这些都是「模型无关」的公共逻辑，重写一遍只会引入不一致。子类唯一的
    契约就是 ``_call_api(model_name, messages, tools, tool_choice, **kwargs)``。

------------------------------------------------------------------------------
测试钩子（指令式，行首匹配）
------------------------------------------------------------------------------
    为了能用它测「工具调用」「思考链」这些分支，在**最后一条 user 消息**里
    写下面这两种行，即可确定性触发对应行为：

        #mock-think: 我先核对一下差旅标准……
        #mock-tool: search_flights {"departure": "PEK", "arrival": "SHA"}

        · ``#mock-think:`` —— 先产出一段 ``ThinkingBlock``（思考链），
          用来验证前端与 ``agentscope.event`` 的 THINKING 事件通路；
        · ``#mock-tool:``  —— 产出一个 ``ToolCallBlock``，其后的第一个
          ``{`` 开始的内容到行尾是工具入参 JSON（必须是**合法 JSON 字符串**，
          ``ToolCallBlock.input`` 要求 ``str`` 而不是 ``dict``）。

    两个指令可以同时出现，思考块先于工具调用（与真实模型的输出顺序一致）。
    没有指令时按普通文本回答。
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Literal

import shortuuid
from pydantic import ConfigDict

from agentscope.credential import CredentialBase
from agentscope.formatter import FormatterBase
from agentscope.message import Msg, TextBlock, ThinkingBlock, ToolCallBlock
from agentscope.model import ChatModelBase, ChatResponse, ChatUsage, FinishedReason
from agentscope.tool import ToolChoice

# ==============================================================================
# 指令前缀（对外可见，测试与文档都引用这两个常量，不要在别处硬编码字面量）
# ==============================================================================

#: 思考链指令前缀，例如 ``#mock-think: 我先查一下公司差旅标准``。
THINK_DIRECTIVE = "#mock-think:"

#: 工具调用指令前缀，例如 ``#mock-tool: search_flights {"city": "上海"}``。
TOOL_DIRECTIVE = "#mock-tool:"

#: Mock 回复的统一前缀。存在的意义是**可辨识**：在日志或前端截图里看到它，
#: 立刻能判断「这一轮走的是 MockLLM，不是真模型」，而不是去怀疑模型变笨了。
REPLY_PREFIX = "【MockLLM】"

#: 凭据类型标识。与 :class:`MockCredential` 的 ``type`` 字段**必须**恒等 ——
#: 这个字符串同时出现在三处（类字段、``extra_credentials`` 注册、
#: 写进 storage 的凭据记录），散成字面量就会出现「注册的名字与存的名字不一致」
#: 这种只在运行期才炸的错。
MOCK_CREDENTIAL_TYPE = "aligo_mock_credential"

#: 降级凭据在用户凭据列表里的显示名。带“降级”二字是刻意的：用户在
#: 「凭据」页看到它时应当立刻明白这不是他配的、也不产生任何费用。
MOCK_CREDENTIAL_NAME = "MockLLM 降级模型（零密钥）"

#: Mock 模型名。它会出现在 ``ChatModelConfig.model`` 里、被
#: ``/model/?provider=aligo_mock_credential`` 列出来、并被前端选中后原样回传，
#: 因此必须是一个**稳定**的标识（``MockChatModel`` 不校验它，但改名会让
#: 已存在的会话配置指向一个列表里不存在的模型）。
MOCK_MODEL_NAME = "aligo-mock-chat"

#: Mock 模型的展示名（前端下拉框里显示的那一行）。
MOCK_MODEL_LABEL = "MockLLM（零密钥降级）"

#: 确定性凭据 id 的固定前缀。见 :func:`mock_credential_id`。
_MOCK_CREDENTIAL_ID_PREFIX = "aligo-mock-"

#: storage 的 ``credentials.id`` 列是 ``String(255)``，超长会被数据库截断或报错。
#: 留出前缀与哈希的余量。
_MAX_CREDENTIAL_ID_LEN = 255


def mock_credential_id(user_id: str) -> str:
    """由 ``user_id`` 派生**确定性**的 Mock 凭据 id。

    为什么必须确定性：播种是**按需重复执行**的（每个新用户第一次请求时补一条，
    见 ``src/llm/degradation.py``）。id 随机的话，「补一条」会在每次调用时
    变成「再加一条」—— 用户的凭据列表里会出现 N 条一模一样的降级模型，
    而 ``upsert_credential`` 的幂等性完全依赖这个 id。

    ⚠️ 为什么不直接用 ``user_id`` 拼进 id 就完事：``credentials.id`` 是
    **全局**主键（``storage/_sql/_tables.py``），而 ``upsert_credential``
    对「不属于我的预设 id」是直接 INSERT（撞主键即 IntegrityError）。user_id
    里若含超长或奇异字符，截断后可能与另一个用户撞上 —— 于是第二个用户的
    首次播种会以一个 500 收场。超过长度上限时改用一个带命名空间前缀的
    哈希：仍然确定性，但不再可能被别人撞上。

    Args:
        user_id (`str`): 框架认定的用户标识。

    Returns:
        `str`: 形如 ``aligo-mock-<user_id>`` 的确定性凭据 id。
    """
    candidate = f"{_MOCK_CREDENTIAL_ID_PREFIX}{user_id}"
    if len(candidate) <= _MAX_CREDENTIAL_ID_LEN:
        return candidate

    import hashlib

    digest = hashlib.sha256(user_id.encode("utf-8")).hexdigest()[:32]
    return f"{_MOCK_CREDENTIAL_ID_PREFIX}{digest}"


def mock_model_card() -> Any:
    """构造 Mock 模型的 ``ModelCard``。

    为什么 Mock 需要一个模型卡片（而不是像原来那样 ``list_models()`` 返回空）：

        前端「可用模型」列表是 ``GET /credential/`` × ``GET /model/?provider=``
        两个接口拼出来的（``web/frontend/src/hooks/useAvailableModels.ts``）。
        零密钥时用户唯一拥有的凭据就是降级凭据，而 ``list_models()`` 返回
        空列表会让这个分组**元素为空** ⇒ 前端 ``getFirstAvailableModel()``
        取不到任何模型 ⇒ ``selectedModel`` 恒为 ``null`` ⇒ **发送按钮永久禁用**。
        服务端其实完全能跑（MockChatModel 是确定性的），但用户一个字都发不出去。

        症状与原因的落差极大：探针全绿、页面正常渲染，只有「发送」是灰的。
        补上卡片，这条链路就自然通了 —— 不需要给前端开特例分支。

    Returns:
        `Any`: 一个 ``ModelCard`` 实例（``agentscope.model.ModelCard``）。
    """
    from agentscope.model import ModelCard

    return ModelCard(
        name=MOCK_MODEL_NAME,
        label=MOCK_MODEL_LABEL,
        status="active",
        context_size=32768,
        output_size=4096,
        # 空 schema：Mock 没有任何可调参数（``MockChatModel.Parameters`` 也是空的）。
        # 传 ``{}`` 而不是省略，是因为前端会拿它渲染「模型参数」弹层 ——
        # 字段缺失与空对象的区别是「渲染崩掉」与「渲染出一个空弹层」。
        parameter_schema={},
        parameters_overrides={},
    )


def _new_block_id() -> str:
    """生成一个 block id。

    为什么要自己生成而不是让 ``TextBlock`` 的 ``default_factory`` 去生成：
    流式输出时，**同一个文本块的所有分片必须共用同一个 id** ——
    框架的 ``_StreamAccumulator`` 正是按 id 归并分片的
    （``model/_utils.py`` 的 ``_AccTextBlock.append``）。若每个分片各拿一个新 id，
    累积器会把它们当成 N 个独立的文本块，最终响应里出现 N 段被截断的文字。

    Returns:
        `str`: 形如 ``txt-3f2a...`` 的短 id。
    """
    return f"txt-{shortuuid.uuid()}"


def _last_user_text(messages: list[Msg]) -> str:
    """取出最后一条 user 消息的纯文本内容（没有则返回空串）。

    取「最后一条」而不是「第一条」：多轮对话里最新的用户输入才决定这一轮该
    回什么，Mock 的指令解析同样遵循这个语义。

    Args:
        messages (`list[Msg]`): 模型看到的完整消息列表。

    Returns:
        `str`: 最后一条 user 消息的文本（多块用换行拼接）；找不到则空串。
    """
    for msg in reversed(messages):
        if msg.role == "user":
            return msg.get_text_content() or ""
    return ""


def _parse_directives(text: str) -> tuple[str | None, tuple[str, str] | None]:
    """从用户文本里解析 ``#mock-think:`` 与 ``#mock-tool:`` 指令。

    逐行扫描而不是用正则一次性匹配：指令只认**行首**（允许前导空白），
    这样用户正文里偶然出现的 ``#mock-tool:``（比如在引用一段文档）不会被误触发。

    Args:
        text (`str`): 最后一条 user 消息的文本。

    Returns:
        `tuple[str | None, tuple[str, str] | None]`:
            - 思考内容（``#mock-think:`` 之后的部分），没有则为 ``None``；
            - ``(工具名, 入参 JSON 字符串)``，没有则为 ``None``。
    """
    thinking: str | None = None
    tool_call: tuple[str, str] | None = None

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if line.startswith(THINK_DIRECTIVE):
            thinking = line[len(THINK_DIRECTIVE):].strip()
            continue

        if line.startswith(TOOL_DIRECTIVE):
            payload = line[len(TOOL_DIRECTIVE):].strip()
            # 工具名取到第一个空白/花括号为止，其余整体作为入参原文。
            # ⚠️ 刻意**不**在这里 json.loads 再 dumps：那样会改写用户的原文
            # （键序、空格、数字格式都会变），而测试往往要断言「传给工具的参数
            # 就是这一串」，改写会让断言对不上。
            split_at = len(payload)
            for idx, char in enumerate(payload):
                if char.isspace() or char == "{":
                    split_at = idx
                    break
            name = payload[:split_at].strip()
            args = payload[split_at:].strip()
            if name:
                # 入参缺省时给一个空对象，保证 ToolCallBlock.input 永远是
                # 合法 JSON（下游的 FunctionTool 会直接解析它）。
                tool_call = (name, args or "{}")

    return thinking, tool_call


def _build_text_reply(user_text: str, *, has_tools: bool) -> str:
    """按固定模板生成 Mock 的文本回复。

    模板刻意做成「能看出收到了什么 + 明确标注自己是 Mock」：
    它要够真实到能驱动前端渲染，又必须一眼看出不是真模型。

    Args:
        user_text (`str`): 最后一条用户消息的文本。
        has_tools (`bool`): 本次调用是否带了工具（决定是否提示可用工具）。

    Returns:
        `str`: 回复正文。
    """
    # 摘要只取首行的前 40 个字符：Mock 的回复要保持短且稳定，
    # 全文回显会让多轮对话的上下文迅速膨胀，反而拖慢本地联调。
    first_line = user_text.strip().splitlines()[0] if user_text.strip() else ""
    digest = first_line[:40] + ("…" if len(first_line) > 40 else "")

    lines = [
        f"{REPLY_PREFIX} 已收到你的差旅需求。",
        "",
    ]
    if digest:
        lines.append(f"我理解到的内容是：{digest}")
    else:
        lines.append("我没有收到具体的文本内容。")

    if has_tools:
        lines.append(
            "本次调用携带了可用工具，但未收到触发指令，"
            "因此只做文本回复。",
        )
    lines.append(
        "当前未配置真实模型密钥，本回复由 MockLLM 确定性生成，"
        "仅用于打通链路。",
    )
    return "\n".join(lines)


def _estimate_tokens(char_count: int) -> int:
    """按字符数粗略估算 token 数（约 1.5 字符 / token，取上整）。

    存在的意义是让 ``usage`` 字段**非空且稳定** —— 成本与用量面板、以及
    ``ChatUsage`` 的必填字段都依赖它。这里的数值不追求准确：Mock 的用量本来
    就没有计费含义，写一个固定的假数字（比如恒等于 1）反而会让「模型切换后
    用量没变化」这种真实问题被掩盖。

    ⚠️ 入参是**字符数**而不是字符串本身：调用方统计的是「多条消息的文本总长」，
    若先拼成一个字符串再传进来，为了拿一个长度就把全部上下文复制了一遍 ——
    长会话下这是一次毫无意义的 O(n) 内存分配。

    Args:
        char_count (`int`): 字符数。

    Returns:
        `int`: 估算的 token 数，至少为 1。
    """
    return max(1, int(char_count / 1.5 + 0.5))


class MockChatFormatter(FormatterBase):
    """MockLLM 的格式化器。

    ⚠️ 它看起来是纯样板，实则是**必需的**：框架的
    ``Agent._handle_incoming_messages`` 会读
    ``self.model.formatter.supported_input_media_types``
    （``agentscope/agent/_agent.py:2062``）来决定要不要把多模态内容块转成文本提示。

    ``formatter`` **不是** ``ChatModelBase`` 的基类属性，而是由每个具体模型
    在 ``__init__`` 里自己赋的（``model/_openai_chat/_model.py`` 等
    都是 ``self.formatter = formatter or XxxChatFormatter()``）。
    因此一个忘了设 formatter 的模型类，在**单纯调用模型**时表现完全正常
    （空密钥下也能出文本），只有在**被 Agent 驱动**时才炸
    ``AttributeError: 'MockChatModel' object has no attribute 'formatter'``,
    而那时错误已被框架兜成一个笼统的
    ``{"type": "setup", "message": "The session could not be prepared"}``——
    真正的原因只出现在日志里。这正是本类被补上的原因。

    ``input_types`` 只声明 ``text/plain``：MockLLM 确实**处理不了**图片/音频，
    声明支持只会让管线放行一张图、然后得到一段文本回显 —— 一个比报错更糟的
    「静默降级」。声明不支持，框架会在入口就把这个问题挡下来。
    """

    input_types: list[str] = ["text/plain"]

    async def format(self, msgs: list[Msg]) -> list[dict[str, Any]]:
        """把 ``Msg`` 列表投影成最简的 ``{name, role, text}`` 结构。

        MockLLM 自己的 ``_call_api`` **不会**调用本方法（它直接读 ``Msg``），
        但 ``FormatterBase.format`` 是抽象方法，必须实现；而且把它实现成
        可用的（而不是 ``raise NotImplementedError``）更安全 ——
        将来框架若有别的代码路径调用 ``formatter.format``，
        抛异常会让整条链路失败，而返回一个合理的投影最多是内容不够丰富。

        Args:
            msgs (`list[Msg]`): 待格式化的消息列表。

        Returns:
            `list[dict[str, Any]]`: 每条消息一个字典，含 ``name`` / ``role`` /
            ``text`` 三个键，文本取自消息里所有 ``TextBlock`` 的拼接。
        """
        self.assert_list_of_msgs(msgs)
        return [
            {
                "name": msg.name,
                "role": msg.role,
                # 用 getattr 兜底而不是直接取 block.text：content 里可能混有
                # 非 TextBlock 的块（思考块/工具调用块），它们没有 text 字段。
                "text": "".join(
                    getattr(block, "text", "")
                    for block in (msg.content or [])
                ),
            }
            for msg in msgs
        ]


class MockCredential(CredentialBase):
    """MockLLM 的凭据类型。

    看起来多余（Mock 不需要任何密钥），但它承担一个**架构职责**：
    ``agentscope.app`` 的聊天链路是从 storage 里的 credential 记录
    ``CredentialFactory.from_dict(...)`` 反查模型类的
    （``agentscope/app/_service/_model.py:12-63``）。注册一个 credential 类型，
    就能让「app 链路」与「我们自己的 build_chat_model」选到同一个模型实现，
    从而在零密钥环境下整个服务天然可用，不需要给 app 链路开特例分支。
    """

    # CredentialBase 是 pydantic 模型，默认不允许额外字段。
    # 这里保持默认（禁止额外字段）而不放开 extra：credential 是从 storage 里
    # 读出来的，多出一个字段往往意味着「存进去的结构与当前代码版本不一致」，
    # 应当报错而不是静默忽略。
    model_config = ConfigDict(title="AliGo MockLLM")

    type: Literal["aligo_mock_credential"] = "aligo_mock_credential"
    """凭据类型标识，用于 CredentialFactory 的反查注册。"""

    @classmethod
    def get_chat_model_class(cls) -> type[ChatModelBase]:
        """返回本凭据对应的模型类。

        Returns:
            `type[ChatModelBase]`: 恒为 :class:`MockChatModel`。
        """
        return MockChatModel

    @classmethod
    def get_embedding_model_class(cls) -> type[Any]:
        """返回本凭据对应的 **embedding** 模型类。

        本类同时承担「零密钥的向量化」这一职责，理由是**知识库链路走得是
        另一条路**，它压根不看 ``src/web_embedding`` 的三合一降级链：

            框架 ``KnowledgeBaseManagerBase.get_knowledge``
              → ``app/_service/_embedding.py::build_embedding_model``
              → ``CredentialFactory.get_credential_class(config.type)``
              → ``credential_cls.get_embedding_model_class()``

        也就是说，配置里写的 ``EmbeddingModelConfig.type`` 必须是一个**已注册的
        凭据类型**，且它得能给出 embedding 类。零密钥下框架自带的 4 种
        （dashscope/openai/gemini/ollama）要么必须有 key、要么需要一个
        Ollama 服务端，于是「知识库建得出来、检索永远没有结果」。

        复用同一个凭据类型（而不是再注册一个 ``..._embedding_credential``）
        是为了少一个注册点：注册点只有 ``extra_credentials`` 一处
        （``src/server/app.py``），而**漏注册的后果是静默的** ——
        ``src/knowledge/rag.py`` 对单条 KB 的解析失败是逐条吞掉的，
        症状退化成「知识库列表看得见、问答就是不带出处」。

        Returns:
            `type[Any]`: :class:`~src.web_embedding.mock.MockEmbeddingModel`。
        """
        # 局部导入：``src.web_embedding`` 会在导入期读配置，而本模块
        # 是 ``build_chat_model`` 的依赖 —— 顶层导入会让「只想用 Mock 聊天模型」
        # 的地方也被迫初始化一遍 embedding 配置。
        from src.web_embedding.mock import MockEmbeddingModel

        return MockEmbeddingModel


class MockChatModel(ChatModelBase):
    """确定性的 Mock 模型。

    只实现 ``_call_api``：重试与流式累积由框架的 ``ChatModelBase.__call__``
    负责（见模块文档字符串）。

    Attributes:
        stream (`bool`): 为 True 时返回异步生成器（逐分片），否则返回单个完整响应。
    """

    class Parameters(ChatModelBase.Parameters):
        """Mock 模型没有可调参数。

        保留这个空的内嵌类是为了**签名兼容**：框架与我们的装配函数都按
        ``parameters: BaseModel`` 传入（``agentscope/model/_base.py:66``），
        真实模型传的是各自的 ``Parameters`` 实例。若这里直接沿用基类而不显式
        继承，将来框架给基类加字段时，Mock 会悄悄继承到一堆对它有语义的配置项。
        """

    @classmethod
    def list_models(cls, custom_yaml_dir: str | None = None) -> list[Any]:
        """列出本模型可选的模型卡片。

        ⚠️ **必须**覆写基类的实现（而不是留空）。基类是按「子类所在目录下的
        ``_models/*.yaml``」来找卡片的（``agentscope/model/_base.py:150-160``），
        ``src/llm/`` 下没有这个目录 ⇒ 返回空列表 ⇒ 前端「可用模型」为空 ⇒
        发送按钮永久禁用。完整因果链见 :func:`mock_model_card`。

        Args:
            custom_yaml_dir (`str | None`): 基类签名要求的位置参数，
                本实现**不使用**（卡片是代码里写死的，不从磁盘读）。

        Returns:
            `list[Any]`: 只含一个元素的 ``ModelCard`` 列表。
        """
        del custom_yaml_dir  # 基类签名要求；Mock 的卡片不来自 YAML
        return [mock_model_card()]

    def __init__(
        self,
        credential: MockCredential | None = None,
        model: str = "aligo-mock",
        parameters: ChatModelBase.Parameters | None = None,
        stream: bool = True,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        context_size: int = 32768,
        chunk_size: int = 12,
    ) -> None:
        """初始化 Mock 模型。

        Args:
            credential (`MockCredential | None`): 凭据；``None`` 时自动建一个。
            model (`str`): 模型名，会出现在 ``ChatResponse`` 的元数据里。
            parameters (`ChatModelBase.Parameters | None`): 模型参数；``None`` 时用默认。
            stream (`bool`): 是否走流式（默认 True，与真实链路一致）。
            max_retries (`int`): 最大重试次数，透传给基类。
            retry_delay (`float`): 重试间隔秒数，透传给基类。
            context_size (`int`): 上下文长度，透传给基类（影响框架的上下文压缩）。
            chunk_size (`int`): 流式分片大小（字符数）。默认 12 是为了让
                「分片 → 累积 → 前端渲染」这条路径上真正产生**多个**分片；
                若一次吐完，流式相关的 bug（id 不共用、累积丢字）就测不出来。

        Raises:
            ValueError: ``chunk_size`` 小于 1 时。取 0 会让分片循环无法推进，
                表现为「请求卡死」而不是「参数错误」，所以在这里就拦下。
        """
        if chunk_size < 1:
            raise ValueError(
                f"chunk_size 必须 >= 1，实际为 {chunk_size}；"
                f"取 0 会让流式分片循环无法推进，请求会挂住而不是报错。",
            )

        super().__init__(
            credential=credential or MockCredential(),
            model=model,
            parameters=parameters or self.Parameters(),
            stream=stream,
            max_retries=max_retries,
            retry_delay=retry_delay,
            context_size=context_size,
        )
        self.chunk_size = chunk_size
        # formatter **必须**在这里赋值：基类不管这件事，每个具体模型各自负责
        # （见 MockChatFormatter 的文档字符串）。漏掉它的症状是
        # 「直接调模型一切正常、被 Agent 驱动就 setup 失败」。
        self.formatter = MockChatFormatter()

    async def _call_api(
        self,
        model_name: str,
        messages: list[Msg],
        tools: list[dict] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        """生成模型响应（框架唯一要求实现的抽象方法）。

        Args:
            model_name (`str`): 模型名（框架传入，Mock 不使用）。
            messages (`list[Msg]`): 完整消息列表。
            tools (`list[dict] | None`): 可用工具的 JSON Schema 列表。
            tool_choice (`ToolChoice | None`): 工具选择策略（Mock 不使用）。
            **kwargs: 其它透传参数（Mock 忽略）。

        Returns:
            `ChatResponse | AsyncGenerator[ChatResponse, None]`:
                ``self.stream`` 为 True 时返回异步生成器，
                否则返回单个 ``is_last=True`` 的完整响应。
        """
        started = time.perf_counter()

        user_text = _last_user_text(messages)
        thinking_text, tool_call = _parse_directives(user_text)

        # ---- 组装「最终响应」的内容块 ----------------------------------------
        # 先在内存里把完整内容拼好，再由流式分支切片吐出去。
        # 这样做的好处是「流式」与「非流式」两条路径共用同一份内容生成逻辑，
        # 不会出现「非流式回答了 A、流式回答了 B」这种只在一种模式下复现的偏差。
        blocks: list[TextBlock | ThinkingBlock | ToolCallBlock] = []

        if thinking_text:
            blocks.append(ThinkingBlock(thinking=thinking_text))

        if tool_call is not None:
            name, args_json = tool_call
            blocks.append(
                ToolCallBlock(
                    id=f"call-{shortuuid.uuid()}",
                    name=name,
                    input=args_json,
                ),
            )
        else:
            blocks.append(
                TextBlock(
                    text=_build_text_reply(user_text, has_tools=bool(tools)),
                    id=_new_block_id(),
                ),
            )

        # 文本块的分片是**累积式**的（框架按 id 归并），所以这里要记下
        # 「要依次吐出的前缀」。思考块同理，但思考块一般很短，不值得再切片。
        reply_text = (
            blocks[-1].text if isinstance(blocks[-1], TextBlock) else ""
        )

        # ---- 用量估算 --------------------------------------------------------
        # input 侧按「所有消息的文本长度」估，output 侧按本次产出的文本估。
        # 工具调用的入参也算作输出，否则带工具的那一轮用量会明显偏低。
        input_chars = sum(len(m.get_text_content() or "") for m in messages)
        output_chars = len(reply_text) + sum(
            len(b.input) for b in blocks if isinstance(b, ToolCallBlock)
        )
        usage = ChatUsage(
            input_tokens=_estimate_tokens(input_chars),
            output_tokens=_estimate_tokens(output_chars),
            time=time.perf_counter() - started,
        )

        response_id = f"mock-{shortuuid.uuid()}"

        # ---- 非流式：一次给全 ------------------------------------------------
        if not self.stream:
            return ChatResponse(
                id=response_id,
                content=blocks,
                is_last=True,
                usage=usage,
                finished_reason=FinishedReason.COMPLETED,
                metadata={"mock": True, "model": model_name},
            )

        # ---- 流式：先分片，再补一个含完整内容的收尾响应 ----------------------
        async def _stream() -> AsyncGenerator[ChatResponse, None]:
            """按块产出分片，最后补一个 ``is_last=True`` 的完整响应。

            顺序上**必须先分片、后收尾**：框架的 ``__call__`` 在遇到
            ``is_last=True`` 时会停止累积并把它原样交给消费方
            （``agentscope/model/_base.py:280-282``）。若先给收尾响应，后续分片会被丢弃。
            """
            # 思考块与工具调用块都是「整体」语义，不切片：
            # 工具调用的 input 是 JSON 字符串，切开会让下游看到半截 JSON。
            for block in blocks:
                if isinstance(block, TextBlock):
                    continue
                yield ChatResponse(
                    id=response_id,
                    content=[block],
                    is_last=False,
                    metadata={"mock": True},
                )

            # 文本块按 chunk_size 切开，所有分片共用同一个 block id（关键）。
            if reply_text:
                block_id = blocks[-1].id
                for start in range(0, len(reply_text), self.chunk_size):
                    yield ChatResponse(
                        id=response_id,
                        content=[
                            TextBlock(
                                text=reply_text[start : start + self.chunk_size],
                                id=block_id,
                            ),
                        ],
                        is_last=False,
                        metadata={"mock": True},
                    )

            # 收尾：带上**完整**内容与用量。消费方用它来构建最终的消息对象。
            yield ChatResponse(
                id=response_id,
                content=blocks,
                is_last=True,
                usage=usage,
                finished_reason=FinishedReason.COMPLETED,
                metadata={"mock": True, "model": model_name},
            )

        return _stream()

    @classmethod
    def _get_retryable_exceptions(cls) -> tuple[type[Exception], ...]:
        """Mock 不联网，没有「可重试」的异常。

        Returns:
            `tuple[type[Exception], ...]`: 空元组，表示任何异常都直接抛出。
        """
        return ()


__all__ = [
    "MockChatModel",
    "MockCredential",
    "REPLY_PREFIX",
    "THINK_DIRECTIVE",
    "TOOL_DIRECTIVE",
]
