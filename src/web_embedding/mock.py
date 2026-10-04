# -*- coding: utf-8 -*-
"""确定性假向量 —— 三合一降级链的**最后一档**。

⚠️ 读这个模块前请先读 :mod:`src.web_embedding` 顶部关于「Mock 档很危险」
的那一段。这里只重复一句：**它产出的向量没有语义**，
两个意思完全相反的句子在它看来与两个随机句子一样远。
它存在的理由是让测试与离线冒烟**不依赖网络**，不是为了让生产少配一个 key。
"""

from __future__ import annotations

import hashlib
import math

from agentscope.embedding import EmbeddingModelBase, EmbeddingResponse
from agentscope.embedding._embedding_usage import EmbeddingUsage

#: 假向量的批量大小。
#:
#: ⚠️ 定成一个大数（而不是照抄云端那些 10 / 512）是**有意的**：
#: 基类的 ``__call__`` 会按 ``batch_size`` 把输入切片、再 ``asyncio.gather``
#: 并发调用 ``_call_api``。Mock 档没有网络、没有速率限制，
#: 切成多批只会让「一次调用」变成「N 次并发调用」，
#: 而每一次调用在测试里都是一个可观察的副作用点 ——
#: 于是「Mock 把输入切成了几批」这件与业务无关的事，
#: 会渗透进用例的断言里。取一个大数让这条路只有一批，把噪音消掉。
_BATCH_SIZE = 1_000_000


def _stable_vector(text: str, dimension: int) -> list[float]:
    """由 ``text`` 确定性地生成一个 ``dimension`` 维的单位向量。

    ⚠️ 用 ``hashlib`` 而不是内置的 ``hash()``：后者对字符串**默认加盐**
    （``PYTHONHASHSEED``），同一段文本在两个进程里会得到不同的向量。
    症状是「测试单独跑绿、和别的用例一起跑红」或者
    「CI 与本地结果不一致」—— 而这类不稳定会让整条检索链路不可测。

    ⚠️ 生成方式是「按需扩展摘要」而不是「取一个大随机数再切段」：
    维度的合法范围是 1..任意，把整个向量压在一条 32 字节的摘要里
    最多只能取到 64 个 float（每 4 bit 一个），维度一大就退化成循环重复，
    那样任意两段文本的向量都会高度相似（与「语义无关」的初衷相反）。
    这里按 ``dimension`` 分批摘要，要多少生成多少。

    Args:
        text (`str`): 待向量化的文本。
        dimension (`int`): 目标维度，必须为正。

    Returns:
        `list[float]`: 长度为 ``dimension`` 的单位向量（L2 范数为 1）。

    Raises:
        ValueError: ``dimension`` 非正时。
    """
    if dimension <= 0:
        raise ValueError(f"dimension 必须为正，收到 {dimension}")

    # 每 16 字节摘要提供 16 个 float（每字节一个），按需拼够 dimension 个。
    values: list[float] = []
    block = 0
    while len(values) < dimension:
        digest = hashlib.sha256(
            # ⚠️ 把 block 混进摘要输入，否则每一块都是同一条摘要的重复，
            # 于是向量呈现周期性 —— 任意两段文本的向量距离会被结构性地拉近。
            f"{text}\x00{block}".encode(),
        ).digest()
        # 每字节映射到 [-1, 1)：128 个取值，足够让向量之间互不相关。
        values.extend((byte - 127.5) / 127.5 for byte in digest)
        block += 1

    values = values[:dimension]

    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0.0:  # pragma: no cover —— 32 字节全为 127.5 的概率可忽略
        # ⚠️ 仍然要接住：零向量在 COSINE 度量下是**未定义**的
        # （分母为零），Milvus 会拒绝写入或返回 NaN 距离。
        # 与其让它在很远的地方以一种看不懂的方式炸掉，不如在这里给一个确定值。
        return [1.0] + [0.0] * (dimension - 1)
    return [v / norm for v in values]


class MockEmbeddingModel(EmbeddingModelBase):
    """不联网、可复现的假向量模型。

    ⚠️ 继承框架的 :class:`~agentscope.embedding.EmbeddingModelBase`，
    只实现 ``_call_api`` 一个方法。批切分、并发、重试、``TextBlock`` 解包
    全部由基类的 ``__call__`` 完成（``agentscope/embedding/_embedding_base.py:197-260``）——
    自己再写一遍就等于给自己留一个与框架行为不一致的分支。

    ⚠️ 构造参数里的 ``credential`` 是基类签名要求的，本类**不使用**。
    传 :class:`agentscope.credential.CredentialBase` 的空实例即可；
    但**不要**为了让签名好看就传一个真实的 DashScope 凭据 ——
    那会让「这个模型到底打不打电话」这件事在读代码时变得可疑。
    """

    class Parameters(EmbeddingModelBase.Parameters):
        """Mock 向量模型没有可调参数。

        保留这个空的内嵌类是为了**签名兼容**：框架的
        ``app/_service/_embedding.py::build_embedding_model`` 在
        ``config.parameters`` 非空时会调用 ``embedding_cls.Parameters(**...)``
        （``agentscope/app/_service/_embedding.py:77-81``）。没有它，任何带参数的
        ``EmbeddingModelConfig`` 都会以 ``AttributeError`` 失败 ——
        而失败点离「少写三行类定义」很远。
        """

    def __init__(
        self,
        *,
        dimensions: int,
        model: str = "aligo-mock-embedding",
        credential: object | None = None,
        parameters: object | None = None,
        context_size: int | None = None,
    ) -> None:
        """初始化。

        Args:
            dimensions (`int`): 输出维度。**必须**与 ``milvus.dimension`` 一致。
            model (`str`): 模型名，只用于日志与 ``EmbeddingModelConfig`` 的展示。
            credential (`object | None`): 基类签名要求的凭据；``None`` 时
                现场构造一个空的 ``CredentialBase``。
            parameters (`object | None`): 框架按配置传入的参数对象。本实现
                **接受但不使用**（Mock 没有可调参数）。⚠️ 这个形参本身是必需的：
                框架的 ``build_embedding_model`` **恒定**传
                ``parameters=``（``agentscope/app/_service/_embedding.py:83-88``），少一个
                形参就是 ``TypeError`` —— 而走这条路径的只有知识库链路，
                症状是「知识库建好了、检索报 500」。
            context_size (`int | None`): 框架在模型卡片里查到后才传（本项目
                的 Mock 卡片由 ``src/llm/mock.py`` 提供，因此实际总会传）。
                ``None`` 时回落到默认值。
        """
        from agentscope.credential import CredentialBase

        super().__init__(
            credential=credential if credential is not None else CredentialBase(),
            model=model,
            dimensions=dimensions,
            parameters=parameters,
            # ⚠️ context_size 对 Mock 没有意义（不走到任何真实接口），
            # 但**不能**随手填 0：基类把它存成字段，将来若有代码用
            # 「输入长度 / context_size」估算截断比例，0 会让它除零。
            context_size=context_size or 8192,
            batch_size=_BATCH_SIZE,
            # 不重试：Mock 不会因为网络失败，重试只会在真的出错时
            # 把同一个 TypeError 重复三遍，让日志更难读。
            max_retries=1,
            retry_delay=0.0,
        )

    async def _call_api(self, inputs: list[str], **kwargs: object) -> EmbeddingResponse:
        """把一批文本映射成向量。

        ⚠️ 签名必须与基类的抽象方法一致
        （``agentscope/embedding/_embedding_base.py:364-368``：``_call_api(self, inputs: list[Any], **kwargs)``）。
        基类保证传进来的 ``inputs`` **已经是** ``list[str]``
        （``TextBlock`` 在 ``__call__`` 里就被解包了），所以这里不必再做类型分支。

        Args:
            inputs (`list[str]`): 一批文本。
            **kwargs: 基类透传的额外参数，本实现忽略。

        Returns:
            `EmbeddingResponse`: 与 ``inputs`` **等长且同序**的向量列表。
        """
        del kwargs
        return EmbeddingResponse(
            embeddings=[
                _stable_vector(text, self.dimensions) for text in inputs
            ],
            usage=EmbeddingUsage(tokens=0, time=0.0),
        )


__all__ = ["MockEmbeddingModel"]
