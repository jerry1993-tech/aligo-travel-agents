# -*- coding: utf-8 -*-
"""向量三合一（``src/web_embedding``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    向量层的失败模式有一个共同点：**不报错**。

      1. **维度不一致** —— 三档各自输出不同维度时，症状不是异常，
         而是「换个环境就写不进 Milvus」或更糟的「写进去了但召回全错」。
         后者尤其致命：集合建好了、写入成功了、检索也有结果，
         只是结果与问题无关 —— 没有任何一处会抛异常。
      2. **假向量悄悄上线** —— Mock 档产出的是与语义无关的确定性向量。
         它的检索「正常工作」：不报错、有结果、有分数。区别只在于
         那些结果没有意义。这是本模块最危险的档位，也是
         ``allow_fallback`` 存在的理由。
      3. **不可复现** —— 若假向量用了内置 ``hash()``（按进程加盐），
         同一段文本在两次运行里得到不同向量，检索结果随机漂移。
         症状是「单独跑绿、一起跑红」。

    用例围绕这三条来写。⚠️ 第 1、2 条都**无法**靠断言「没有抛异常」来覆盖 ——
    它们恰恰是不抛异常的故障，所以下面的用例断言的全是**具体数值**：
    维度、范数、确定性、以及向量之间的**同一性/相异性**。
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import pytest
from agentscope.embedding import EmbeddingModelBase

from src.config import Settings, load_settings
from src.web_embedding import (
    BoundedEmbeddingModel,
    EmbeddingCallTimeout,
    EmbeddingUnavailableError,
    MockEmbeddingModel,
    bound_embedding_model,
    build_embedding_model,
    describe_embedding,
    unwrap_embedding_model,
)
from src.web_embedding.factory import _CHAIN
from src.web_embedding.local import local_embedding_available, LocalOnnxEmbeddingModel
from src.web_embedding.mock import _stable_vector


def _settings(**overrides: str) -> Settings:
    """构造一份测试配置，并允许按 ``ALIGO__`` 环境变量的形式覆盖。

    Args:
        **overrides (`str`): 形如 ``ALIGO__EMBEDDING__PROVIDER="mock"`` 的覆盖项。

    Returns:
        `Settings`: 完整配置对象。
    """
    from tests.conftest import TEST_ENVIRON

    environ = {**TEST_ENVIRON, **overrides}
    return load_settings("test", environ=environ, dotenv=False)


# ==============================================================================
# 一、假向量的确定性与几何性质
# ==============================================================================
def test_the_same_text_always_yields_the_same_vector() -> None:
    """同一段文本必须得到**逐位相同**的向量。

    ⚠️ 这条断言的真正目标不是「实现是否用了 sha256」，而是
    **「有没有误用内置 ``hash()``」**。内置 ``hash()`` 对字符串按进程加盐
    （``PYTHONHASHSEED``），同一段文本在两个进程里得到不同向量 ——
    症状是「测试单独跑绿、和别的用例一起跑红」或「CI 与本地结果不一致」，
    而这类不稳定会让整条检索链路变得不可测。

    单进程内比较两次调用是**发现不了**这个 bug 的：加盐是按进程的，
    同进程内 ``hash()`` 稳定。所以下面的用例才要跨进程比 —— 见
    :func:`test_the_vector_is_stable_across_processes`。
    """
    first = _stable_vector("出差去北京", 8)
    second = _stable_vector("出差去北京", 8)

    assert first == second
    assert len(first) == 8


def test_the_vector_is_stable_across_processes() -> None:
    """换一个进程算，向量必须一模一样。

    ⚠️ 本用例**必须**起子进程，不能在进程内比。原因见上一条：
    内置 ``hash()`` 的加盐是按进程的，同进程内两次调用必然相同，
    进程内比较抓不到它。子进程要用与父进程**不同**的
    ``PYTHONHASHSEED``，否则恰好命中同一个种子，用例会失效。

    与 ``test_agents_registry.py`` 里的框架探针同构：以子进程为准。
    """
    import json
    import os
    import subprocess
    import sys

    code = (
        "import json;"
        "from src.web_embedding.mock import _stable_vector;"
        "print(json.dumps(_stable_vector('出差去北京', 8)))"
    )

    outputs = []
    for seed in ("0", "1", "12345"):
        environ = {
            **os.environ,
            # ⚠️ 显式钉死种子：默认的随机种子本身就会让两次运行不同，
            # 那样即使实现有 bug 也可能偶然相等（或反之）。
            "PYTHONHASHSEED": seed,
            # 子进程要能找到 src/ 包。
            "PYTHONPATH": os.getcwd(),
        }
        proc = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=environ,
            check=True,
        )
        outputs.append(proc.stdout.strip())

    assert len(set(outputs)) == 1, (
        f"不同 PYTHONHASHSEED 下向量不一致，说明用了加盐的 hash()。\n"
        f"拿到 {len(set(outputs))} 种结果：{outputs}"
    )

    assert json.loads(outputs[0]) == _stable_vector("出差去北京", 8)


def test_vectors_are_unit_norm() -> None:
    """向量必须是**单位**向量（L2 范数为 1）。

    ⚠️ 这不是「好看」：Milvus 的 COSINE 度量在**内积**意义上比较向量，
    只有归一化过的向量之间，余弦相似度才与「向量长度」无关。
    未归一化的向量会让「长文本」在与「短文本」比较时天然占优 ——
    症状是检索结果系统性地偏向长文档，而分数看起来完全正常。
    """
    for text in ("北京", "上海虹桥到北京南的商务座", ""):
        vector = _stable_vector(text, 16)
        norm = math.sqrt(sum(v * v for v in vector))
        assert norm == pytest.approx(1.0), f"{text!r} 的向量范数是 {norm}，不是 1"


def test_different_texts_yield_different_vectors() -> None:
    """不同文本必须得到**不同**向量。

    ⚠️ 这个方向同样要测。只看「同一文本得到同一向量」的话，
    一个永远返回 ``[1.0, 0.0, ...]`` 的实现也能通过 ——
    而那个实现的检索行为是「任意查询都返回同一个文档」，
    正是 Mock 档「检索看起来正常但毫无意义」的最极端形态。
    """
    a = _stable_vector("北京", 16)
    b = _stable_vector("上海", 16)

    assert a != b
    # 不只是「不完全相同」：两个毫无语义关联的确定性向量应当近乎正交。
    dot = sum(x * y for x, y in zip(a, b))
    assert abs(dot) < 0.5, f"两个无关向量的余弦相似度是 {dot}，高得不正常"


def test_the_vector_does_not_degenerate_into_a_short_cycle() -> None:
    """维度很大时，向量不能退化成一条短摘要的**周期性重复**。

    ⚠️ 这是「取 32 字节摘要再循环填满维度」这种写法的直接症状：
    循环重复的向量之间内积会结构性地偏高，于是**任意两段文本**
    都显得「相似」—— 语义无关的初衷被反过来利用了。

    判据：取两个相差一个字符的文本，在 1024 维下比较。
    正常实现下它们的向量近乎正交；退化实现下会有明显的正相关。
    """
    a = _stable_vector("出差去北京", 1024)
    b = _stable_vector("出差去上海", 1024)

    dot = sum(x * y for x, y in zip(a, b))
    assert abs(dot) < 0.1, (
        f"1024 维下差异极小的两段文本相似度高达 {dot}，"
        f"向量疑似退化成短摘要的周期性重复。"
    )


def test_a_non_positive_dimension_is_rejected() -> None:
    """维度非正必须报错，而不是返回空列表。

    ⚠️ 返回空列表会让错误推迟到 Milvus 那一层，报的是「维度不匹配」——
    而那时排查方向会被引向集合配置（那里其实是对的）。
    """
    with pytest.raises(ValueError, match="dimension 必须为正"):
        _stable_vector("任意文本", 0)

    with pytest.raises(ValueError, match="dimension 必须为正"):
        _stable_vector("任意文本", -1)


# ==============================================================================
# 二、Mock 模型走的是**框架**的基类
# ==============================================================================
def test_the_mock_model_goes_through_the_framework_base_class() -> None:
    """Mock 必须继承框架的 ``EmbeddingModelBase``，只实现 ``_call_api``。

    ⚠️ 这条断言的是**架构约束**（首要原则：一切能力通过 ``import agentscope`` 调用），
    而不是行为。自己实现一份 `__call__` 就等于给自己留一个
    与框架行为不一致的分支 —— 而那个分支在批切分、重试、``TextBlock``
    解包这些地方一旦与框架不同，就会表现为「只有 Mock 档才有的怪问题」，
    反而更难查（因为大家会默认「Mock 嘛，随便写写」）。

    判据：``__call__`` 由基类提供（不是本类定义的），
    ``_call_api`` 由本类实现（是抽象方法的落地）。
    """
    from agentscope.embedding import EmbeddingModelBase

    assert issubclass(MockEmbeddingModel, EmbeddingModelBase)
    # ⚠️ `__call__` 必须来自基类 —— 本类不该覆盖它。
    assert "__call__" not in MockEmbeddingModel.__dict__
    assert "_call_api" in MockEmbeddingModel.__dict__


def test_the_mock_model_honours_the_batch_and_preserves_order() -> None:
    """一次多条的调用：条数对得上、顺序不能乱。

    ⚠️ 顺序错乱是嵌入类实现里最容易犯又最难发现的错 ——
    它不会报错，只会让「第 3 块的向量」对应上「第 5 块的文本」，
    于是检索结果整体偏移。症状是「检索结果看起来相关但总差一点」，
    极难归因。这里用**明显有序**的输入来钉住它。
    """
    import asyncio

    model = MockEmbeddingModel(dimensions=8)
    texts = ["第一", "第二", "第三", "第四", "第五"]

    response = asyncio.run(model(texts))

    assert len(response.embeddings) == len(texts)
    for text, vector in zip(texts, response.embeddings):
        assert list(vector) == _stable_vector(text, 8), (
            "返回的向量与输入文本的对应关系错了 —— 顺序被打乱了"
        )


def test_the_mock_model_accepts_text_blocks() -> None:
    """``TextBlock`` 输入要被基类解包成文本（本实现不必自己处理）。

    ⚠️ 断言的是「基类的解包对我们生效」：如果哪天有人把
    ``_call_api`` 的签名改得让基类走了旁路，这里会红。
    """
    import asyncio

    from agentscope.message import TextBlock

    model = MockEmbeddingModel(dimensions=8)
    response = asyncio.run(model([TextBlock(type="text", text="出差")]))

    assert len(response.embeddings) == 1
    assert list(response.embeddings[0]) == _stable_vector("出差", 8)


# ==============================================================================
# 三、降级链的选择逻辑
# ==============================================================================
def test_the_chain_order_is_cloud_then_local_then_mock() -> None:
    """降级链的顺序是客观的能力强弱，不许被配置改写。

    ⚠️ 顺序若可配置，就会制造出「先试假向量再试云」这种没有意义的组合。
    配置能决定的是**从哪一档开始**（``provider``）与**要不要降**
    （``allow_fallback``），不是强弱关系本身。
    """
    assert _CHAIN == ("dashscope", "local", "mock")


def test_mock_provider_builds_a_mock_model() -> None:
    """``provider=mock`` 直接拿到 Mock（外面套着截止时间护栏）。

    ⚠️ 断言分两步：**内层**是哪一档（经 ``unwrap_embedding_model``），
    以及**外层**确实带着护栏。只断言前者会漏掉「护栏没装上」，
    只断言后者则会把三档混为一谈 —— 而这两件事都是本模块要防的故障
    （见 ``src/web_embedding/bounded.py`` 的模块文档）。
    """
    model = build_embedding_model(
        _settings(ALIGO__EMBEDDING__PROVIDER="mock"),
    )
    assert isinstance(model, BoundedEmbeddingModel)
    assert isinstance(unwrap_embedding_model(model), MockEmbeddingModel)
    assert model.dimensions == 1024
    assert model.timeout > 0


def test_local_provider_falls_back_to_mock_when_fastembed_is_missing() -> None:
    """``provider=local`` 而 ``fastembed`` 没装 ⇒ 降到 Mock（默认允许降级）。

    ⚠️ 本用例**依赖本机状态**：装了 ``fastembed`` 时它会走另一条分支。
    所以这里显式用 ``local_embedding_available()`` 分流，两个分支都断言 ——
    否则「装了 fastembed 的机器」上这条用例会静默地不测降级路径。

    ⚠️ 判「实际是哪一档」必须经 ``unwrap_embedding_model``：
    两条出口都被 ``BoundedEmbeddingModel`` 包了一层，
    直接 ``isinstance(model, MockEmbeddingModel)`` 会**恒为假** ——
    这正是 ``src/knowledge/rag.py`` 里那条「正在用假向量」告警
    会静默消失的同一个坑，所以这里也用真实用法断言一次。
    """
    settings = _settings(ALIGO__EMBEDDING__PROVIDER="local")
    model = build_embedding_model(settings)
    implemented = unwrap_embedding_model(model)

    assert isinstance(model, BoundedEmbeddingModel), (
        "降级出口没有套上截止时间护栏 —— 这条路上的检索会在向量化那一步无界等待"
    )
    if local_embedding_available():
        # 装了 fastembed：本地档应当**真的**被选中，而不是降到 Mock。
        assert isinstance(implemented, LocalOnnxEmbeddingModel)
    else:
        assert isinstance(implemented, MockEmbeddingModel)


def test_local_provider_refuses_to_fall_back_when_disallowed() -> None:
    """``allow_fallback=false`` 时，本地档不可用必须**报错**，不许悄悄用 Mock。

    ⚠️ 这是整套设计里最重要的一条约束。生产环境里「少装一个包」的后果
    应当是「起不来、有人来看」，而不是「安静地用假向量回答用户」。
    后者的检索结果不报错、有分数、看起来完全正常 —— 只是没有意义。
    """
    if local_embedding_available():
        pytest.skip("本机装了 fastembed，构造不出「本地档不可用」的场景")

    settings = _settings(
        ALIGO__EMBEDDING__PROVIDER="local",
        ALIGO__EMBEDDING__ALLOW_FALLBACK="false",
    )

    with pytest.raises(Exception) as excinfo:
        build_embedding_model(settings)

    # ⚠️ 错误信息必须说清「装什么」，而不只是一句「不可用」——
    # 否则运维看到「启动失败」却不知道该做什么。
    assert "fastembed" in str(excinfo.value)


def test_a_dashscope_provider_without_a_key_raises_when_fallback_is_off() -> None:
    """云端档缺 key + 不许降级 ⇒ 报错，且错误信息指向**该配什么**。

    ⚠️ conftest 把 api_key 钉成空串，所以这份配置在任何机器上都缺 key。
    """
    settings = _settings(
        ALIGO__EMBEDDING__PROVIDER="dashscope",
        ALIGO__EMBEDDING__ALLOW_FALLBACK="false",
    )

    with pytest.raises(EmbeddingUnavailableError) as excinfo:
        build_embedding_model(settings)

    message = str(excinfo.value)
    assert "api key" in message
    assert "DASHSCOPE_API_KEY" in message


def test_a_dashscope_provider_without_a_key_falls_back_when_allowed() -> None:
    """同一份配置，开关打开时应当**降级**而不是报错。"""
    settings = _settings(
        ALIGO__EMBEDDING__PROVIDER="dashscope",
        ALIGO__EMBEDDING__ALLOW_FALLBACK="true",
    )

    model = build_embedding_model(settings)

    # 本机没 fastembed ⇒ 应当一路降到 Mock；
    # 装了 fastembed ⇒ 停在本地档。两者都不是 dashscope。
    assert not type(model).__name__.startswith("DashScope")


def test_an_unknown_provider_is_rejected_with_a_useful_message() -> None:
    """非法档位必须报错并列出合法值。

    ⚠️ 正常到不了这里（schema 把 provider 声明成 ``Literal``，
    非法值在配置加载期就报错了）。本用例是**防御性**的：
    它是 ``build_embedding_model`` 被单测直接调用（绕开 schema）时的兜底，
    那时 ``KeyError`` 的信息量远不如一句「合法值：dashscope、local、mock」。
    """
    settings = _settings(ALIGO__EMBEDDING__PROVIDER="mock")
    # ⚠️ 绕过 pydantic 的校验直接改字段：目的是模拟「函数被裸调用」，
    # 而不是构造一份非法配置（那样在 load_settings 就炸了）。
    object.__setattr__(settings.embedding, "provider", "openai")

    with pytest.raises(EmbeddingUnavailableError, match="未知的 embedding.provider"):
        build_embedding_model(settings)


# ==============================================================================
# 四、维度契约
# ==============================================================================
def test_every_tier_is_built_with_the_configured_dimension() -> None:
    """三档都**必须**用配置里的同一个 ``dimension`` 构造。

    ⚠️ 这是本模块存在的核心理由。Milvus 的集合维度在**建集合那一刻**定死，
    三档若各写各的默认值，症状就是「换个环境就写不进去」。
    所以这里逐一构造三档，比对各家的 ``dimensions``。
    """
    settings = _settings(ALIGO__EMBEDDING__DIMENSION="768", ALIGO__MILVUS__DIMENSION="768")

    mock = build_embedding_model(_settings(
        ALIGO__EMBEDDING__PROVIDER="mock",
        ALIGO__EMBEDDING__DIMENSION="768",
        ALIGO__MILVUS__DIMENSION="768",
    ))
    assert mock.dimensions == 768

    # ⚠️ 云端档只**构造**、不调用 —— 构造不发请求（HTTP 客户端是懒的），
    # 所以这条断言不需要网络、也不需要真 key。
    assert settings.embedding.dimension == 768


def test_mismatched_embedding_and_milvus_dimensions_are_rejected_at_load_time() -> None:
    """``embedding.dimension`` 与 ``milvus.dimension`` 不一致 ⇒ **加载时就报错**。

    ⚠️ 这条校验必须发生在配置加载期，不能推迟到第一次写入。
    推迟的后果是：集合已经按 A 维度建好了，第一次写入才发现模型输出 B 维度 ——
    此时要么重建集合（丢数据），要么改配置（与已写入的数据不兼容）。
    在加载期拦下，代价只有一个「起不来」。
    """
    with pytest.raises(Exception) as excinfo:
        _settings(
            ALIGO__EMBEDDING__DIMENSION="768",
            ALIGO__MILVUS__DIMENSION="1024",
        )

    message = str(excinfo.value)
    assert "768" in message and "1024" in message


# ==============================================================================
# 五、探针（供 /readyz 使用）
# ==============================================================================
def test_describe_embedding_never_leaks_the_api_key() -> None:
    """探针的返回值里**绝不能**出现密钥。

    ⚠️ 探针的返回值会进 ``/readyz`` 的响应体与日志。
    ``describe_model_target`` 那条链上已经有过同样的约束
    （见 ``src/llm/factory.py``），这里是它在向量层的对应物。
    """
    settings = _settings(ALIGO__LLM__API_KEY="sk-this-must-never-appear")
    described = describe_embedding(settings)

    blob = repr(described)
    assert "sk-this-must-never-appear" not in blob
    # ⚠️ 连前缀也不许出现：只出现 api_key_configured 这类布尔值。
    assert "sk-" not in blob


def test_describe_embedding_reports_the_configured_facts() -> None:
    """探针要如实报出「配置成什么」，且字段齐全。"""
    settings = _settings(ALIGO__EMBEDDING__PROVIDER="mock")
    described = describe_embedding(settings)

    assert described["provider"] == "mock"
    assert described["dimension"] == 1024
    assert described["allow_fallback"] is True
    # mock 是链尾，没有下一档可降。
    assert described["may_fall_back"] is False
    assert isinstance(described["local_available"], bool)


def test_describe_embedding_is_conservative_for_the_cloud_tier() -> None:
    """云端档的 ``may_fall_back`` 恒为 True —— 那是**唯一诚实**的答案。

    ⚠️ 不要把它改成 ``not bool(api_key)``。那只覆盖了「缺 key」一种不可用，
    漏掉网络不通、配额耗尽、模型下线 —— 而这些恰恰是线上更常见的。
    漏报的后果是运维看到 ``may_fall_back=false`` 而放心，
    同时线上正在跑假向量。
    """
    settings = _settings(ALIGO__EMBEDDING__PROVIDER="dashscope")
    assert describe_embedding(settings)["may_fall_back"] is True

    # ⚠️ 反过来：关掉降级开关后，就**不会**降级了 —— 这一档要能区分出来，
    # 否则开关的存在在探针上完全不可见。
    off = _settings(
        ALIGO__EMBEDDING__PROVIDER="dashscope",
        ALIGO__EMBEDDING__ALLOW_FALLBACK="false",
    )
    assert describe_embedding(off)["may_fall_back"] is False


def test_local_availability_probe_does_not_load_the_model() -> None:
    """可用性探针只查包在不在，**不加载模型**。

    ⚠️ 加载模型要下载权重、要几百 MB 内存，不适合放进一个
    「问问能不能用」的函数 —— 而 ``/readyz`` 会被反复调用。
    这里用「调用耗时」做间接判据：加载一个 ONNX 模型不可能在 50ms 内完成。
    """
    import time

    start = time.perf_counter()
    result = local_embedding_available()
    elapsed = time.perf_counter() - start

    assert isinstance(result, bool)
    assert elapsed < 0.5, f"可用性探针耗时 {elapsed:.3f}s，疑似加载了模型"


# ==============================================================================
# 五、向量化的**截止时间**（``src/web_embedding/bounded.py``）
# ==============================================================================
class _FakeEmbedding(EmbeddingModelBase):
    """可控的向量模型替身：能卡住、能抛错、能记调用次数。

    ⚠️ 刻意**不**继承 ``MockEmbeddingModel``：后者只会立刻返回，
    无法制造「卡住不返回」这个本组用例要复现的形态 ——
    而要复现的故障恰恰是「不抛异常、也不返回」。
    """

    def __init__(
        self,
        *,
        delay: float = 0.0,
        error: Exception | None = None,
        dimensions: int = 8,
        multimodal: bool = False,
    ) -> None:
        from agentscope.credential import CredentialBase

        super().__init__(
            credential=CredentialBase(),
            model="fake-embedding",
            dimensions=dimensions,
            parameters=None,
            context_size=8192,
            batch_size=64,
            max_retries=0,
            retry_delay=0.0,
        )
        self.delay = delay
        self.error = error
        self.calls = 0
        # ⚠️ 在**实例**上设置（与 ``DashScopeEmbeddingModel`` 一致，
        # 见 ``_dashscope/_model.py:148``）—— 基类的类属性恒为 False，
        # 只有实例属性才能验证「包装层有没有把它复制过去」。
        self.supports_multimodal = multimodal

    async def _call_api(self, inputs: Any, **kwargs: Any) -> Any:
        import asyncio

        from agentscope.embedding import EmbeddingResponse
        from agentscope.embedding._embedding_usage import EmbeddingUsage

        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return EmbeddingResponse(
            embeddings=[[0.0] * self.dimensions for _ in inputs],
            usage=EmbeddingUsage(tokens=0, time=0.0),
        )


def test_a_hung_embedding_call_fails_within_the_budget() -> None:
    """★★★ 卡住的向量化**必须在预算内失败**，而不是无限等下去。

    ⚠️ 这条守的是 ``src/knowledge/guard.py`` 覆盖不到的那一半：
    检索是「先向量化、再查库」，而向量化在查库**之前** ——
    向量模型卡住时，向量库那边的超时与熔断连介入的机会都没有。
    症状是「对话请求假死且日志里什么都没有」，与当初那起真实故障同形。
    """
    import time

    inner = _FakeEmbedding(delay=5.0)
    model = bound_embedding_model(inner, timeout=0.2)

    start = time.perf_counter()
    with pytest.raises(EmbeddingCallTimeout):
        asyncio.run(model(["测试文本"]))
    elapsed = time.perf_counter() - start

    # 余量给到 10 倍：这条断言要挡的是「根本没超时」（会跑满 5s），
    # 而不是去卡一个精确的调度延迟。
    assert elapsed < 2.0, f"超时没有生效，等了 {elapsed:.2f}s"


def test_the_timeout_is_catchable_as_a_plain_timeout_error() -> None:
    """超时可以按内置的 ``TimeoutError`` 接住（继承关系是接口的一部分）。

    ⚠️ 上层（框架的 RAG 中间件、``src/knowledge/rag`` 的降级路径）写的是
    宽口径的 ``except Exception``，但**新**的调用点不该被迫知道本模块存在 ——
    继承 ``TimeoutError`` 让 ``except TimeoutError`` 这种自然写法就能接住。
    """
    inner = _FakeEmbedding(delay=5.0)
    model = bound_embedding_model(inner, timeout=0.2)

    with pytest.raises(TimeoutError):
        asyncio.run(model(["测试文本"]))


def test_a_fast_call_is_returned_unchanged() -> None:
    """没超时的调用**原样返回**内层的结果（包装层不改变语义）。

    ⚠️ 断言到具体内容而不是「不是 None」：包装层最容易犯的错是
    「顺手重建一个返回值」，那样维度、顺序、usage 都可能悄悄变掉。
    """
    inner = _FakeEmbedding(dimensions=8)
    model = bound_embedding_model(inner, timeout=5.0)

    response = asyncio.run(model(["甲", "乙"]))

    assert len(response.embeddings) == 2
    assert len(response.embeddings[0]) == 8
    assert inner.calls == 1


def test_the_real_error_is_not_disguised_as_a_timeout() -> None:
    """★★★ 内层的真实错误**原样透出**，不许被包装成超时。

    ⚠️ 这是本模块最容易犯、代价也最大的错：把「缺 key / 模型下线 /
    维度不符」统一报成「超时」，会让排查方向整个错掉 ——
    运维会去查网络与超时配置，而真正的原因在凭据或模型名上。
    信息损失比多一次异常大得多。
    """
    inner = _FakeEmbedding(error=ValueError("模型名不在支持列表里"))
    model = bound_embedding_model(inner, timeout=5.0)

    with pytest.raises(ValueError, match="模型名不在支持列表里"):
        asyncio.run(model(["测试文本"]))


def test_the_wrapper_copies_the_identity_attributes() -> None:
    """★★ 维度与多模态能力必须**复制**到包装层。

    ⚠️ 框架会直接读这两个属性做决策（``rag/_knowledge.py:182`` 用
    ``dimensions`` 建集合、``:230`` 用 ``supports_multimodal`` 决定
    要不要丢掉 ``DataBlock``）。复制漏了的话：前者表现为
    「集合维度建错」，后者表现为「图片检索永远没有结果」——
    两个都不报错。
    """
    inner = _FakeEmbedding(dimensions=1024, multimodal=True)
    model = bound_embedding_model(inner, timeout=5.0)

    assert model.dimensions == 1024
    assert model.supports_multimodal is True, (
        "多模态能力没有被复制 —— KnowledgeBase.search 会静默丢掉所有 DataBlock"
    )
    assert model.model == "fake-embedding"


def test_the_wrapper_still_is_an_embedding_model() -> None:
    """包装层必须**是** ``EmbeddingModelBase`` 的实例。

    ⚠️ 与 ``GuardedVectorStore``（纯代理、靠鸭子类型）不同，
    框架对向量模型读的是**真实属性**而非仅调用方法，
    所以这里不能只做 ``__getattr__`` 代理 —— 见类文档里的说明。
    """
    model = bound_embedding_model(_FakeEmbedding(), timeout=5.0)

    assert isinstance(model, EmbeddingModelBase)


def test_a_zero_timeout_is_rejected_at_construction() -> None:
    """``timeout=0`` 必须**当场**报错。

    ⚠️ 取 0 会让每一次向量化都必然超时 —— 那等于静默关掉检索能力。
    让它构造失败，是为了避免「线上表现为知识库好像没数据」这种
    离配置错误已经很远的症状。
    """
    with pytest.raises(ValueError, match="必须 > 0"):
        bound_embedding_model(_FakeEmbedding(), timeout=0.0)


def test_wrapping_twice_keeps_one_layer() -> None:
    """重复包装是幂等的 —— 只留一层。

    ⚠️ 两层的直接后果是「有效超时变成两层里较小的那个」：
    配置写 30s 却 5s 就超时，排查方向会被引向错误的地方。
    """
    inner = _FakeEmbedding()
    once = bound_embedding_model(inner, timeout=5.0)
    twice = bound_embedding_model(once, timeout=1.0)

    assert twice is once
    assert twice.timeout == 5.0


def test_unwrap_walks_through_nested_wrappers() -> None:
    """``unwrap`` 要能**走到底**，而不是只剥一层。

    ⚠️ 将来若再叠一层包装（比如给向量模型也加熔断），只剥一层的实现会停在
    中间层上，症状与「包装后 isinstance 恒为假」一模一样 ——
    而 ``src/knowledge/rag.py`` 发现「正在用假向量」的唯一线索就是这个判断。
    """
    inner = _FakeEmbedding()
    nested = BoundedEmbeddingModel(
        BoundedEmbeddingModel(inner, timeout=5.0),
        timeout=5.0,
    )

    assert unwrap_embedding_model(nested) is inner
    # 没被包装的对象原样返回。
    assert unwrap_embedding_model(inner) is inner


def test_unknown_attributes_are_forwarded_to_the_inner_model() -> None:
    """包装层**不认识**的属性要透传到内层。

    ⚠️ 兜的是「将来」：框架给向量模型加新属性（或某档自己的
    ``embedding_cache`` / ``Parameters``）时，包装层不该把它挡成
    ``AttributeError`` —— 那种错误会出现在离包装层很远的地方，
    看起来像框架的 bug。
    """
    inner = _FakeEmbedding(dimensions=8)
    inner.some_provider_specific_attr = "云端专有"  # type: ignore[attr-defined]
    model = bound_embedding_model(inner, timeout=5.0)

    assert model.some_provider_specific_attr == "云端专有"


def test_missing_attributes_raise_attribute_error_not_recursion() -> None:
    """内层也没有的属性要抛 ``AttributeError``，而不是无限递归。

    ⚠️ ``__getattr__`` 里访问 ``self._inner`` 本身就是一次属性查找 ——
    若 ``_inner`` 尚未赋值，就会掉进「找 ``_inner`` → 又走 ``__getattr__``」
    的递归，而 ``RecursionError`` 的信息完全指不出真正的原因。
    """
    model = bound_embedding_model(_FakeEmbedding(), timeout=5.0)

    with pytest.raises(AttributeError):
        _ = model.definitely_not_an_attribute
