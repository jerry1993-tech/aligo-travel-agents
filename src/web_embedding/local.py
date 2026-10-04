# -*- coding: utf-8 -*-
"""本地向量模型（ONNX）—— 三合一降级链的**中间档**。

═══ ⚠️ 为什么是 ONNX 而不是 sentence-transformers ═══

``sentence-transformers`` 依赖 PyTorch，而 torch 的 Linux wheel 是 800 MB 起步。
本项目的镜像构建环境里 pip 只有 180–240 kB/s（见 README 的前置说明），
装一次 torch 就是**一小时**级别的构建时间 —— 而它换来的能力，
只是把这一个档从「能用」变成「能用」。

``fastembed`` 走 ONNX Runtime，同一个模型只要 ~150 MB，且不需要 CUDA 那一套。
**代价是它只支持 ONNX 格式的模型**（fastembed 自己维护一份支持列表），
所以 ``local_model`` 只能填那份列表里的名字。

═══ ⚠️ 这一档在本机**当前不可用**（依赖未安装） ═══

``fastembed`` / ``onnxruntime`` 都不在 ``requirements.txt`` 里 —— 这是刻意的：

- 它们是**可选**能力，不进依赖清单，镜像因此不为此付费（约 150 MB）；
- 装了它们镜像才能多一档，不装就是「云端 → Mock」两档。

所以 :meth:`LocalOnnxEmbeddingModel.__init__` **在构造时就**尝试导入，
失败即抛 :class:`LocalEmbeddingUnavailableError`，由
:func:`~src.web_embedding.factory.build_embedding_model` 接住并降级。

⚠️ **在构造时失败，而不是在第一次 ``_call_api`` 时失败**，这一点很重要：
前者让降级链在**启动时**就选定了实现；后者会让应用带着一个坏掉的模型起来，
直到第一个用户提问才炸 —— 而那时错误会以「检索失败」的形式出现在业务层，
离真正的原因（少装一个包）已经很远了。
"""

from __future__ import annotations

from typing import Any

from agentscope.embedding import EmbeddingModelBase, EmbeddingResponse
from agentscope.embedding._embedding_usage import EmbeddingUsage

#: 本机能否跑本地档的**唯一**判据所需的两样东西。
_REQUIRED_PACKAGES = ("fastembed",)


class LocalEmbeddingUnavailableError(RuntimeError):
    """本地向量档不可用（依赖缺失或模型加载失败）。

    ⚠️ 单独定义一个异常类型，而不是让 ``ImportError`` 直接冒到调用方：
    降级链需要**区分**「这一档不可用，试下一档」与
    「这一档可用但调用出错，应当向上报」。两者都是异常，
    混在一起会让降级链在真正的故障上悄悄降级 —— 那是最坏的组合：
    线上静默地用了假向量，而日志里只有一行 warning。
    """


class LocalOnnxEmbeddingModel(EmbeddingModelBase):
    """基于 ``fastembed``（ONNX Runtime）的本地向量模型。

    ⚠️ 只实现 ``_call_api``：批切分、并发、重试、``TextBlock`` 解包
    全部由框架基类的 ``__call__`` 完成，与
    :class:`~src.web_embedding.mock.MockEmbeddingModel` 同构。
    """

    def __init__(
        self,
        *,
        dimensions: int,
        model: str,
        credential: object | None = None,
        batch_size: int = 64,
    ) -> None:
        """加载本地模型。

        ⚠️ 这里**真的会加载模型**（首次使用还会下载权重），
        所以构造开销不小。调用方应当在启动期构造一次并复用，
        不要放进请求路径 —— 见 :func:`~src.web_embedding.factory.build_embedding_model`
        的说明。

        Args:
            dimensions (`int`): 期望的输出维度。**必须**与
                ``milvus.dimension`` 一致。
            model (`str`): fastembed 支持的模型名（如 ``BAAI/bge-small-zh-v1.5``）。
            credential (`object | None`): 基类签名要求的凭据；本档不使用。
            batch_size (`int`): 单次 ``_call_api`` 处理多少条。

        Raises:
            LocalEmbeddingUnavailableError: ``fastembed`` 未安装，
                或模型加载失败，或**模型实际维度与 ``dimensions`` 不符**。
        """
        from agentscope.credential import CredentialBase

        super().__init__(
            credential=credential if credential is not None else CredentialBase(),
            model=model,
            dimensions=dimensions,
            parameters=None,
            context_size=8192,
            batch_size=batch_size,
            max_retries=1,
            retry_delay=0.0,
        )

        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise LocalEmbeddingUnavailableError(
                f"本地向量档需要 fastembed，但它没有安装（{exc}）。\n"
                f"安装：pip install fastembed\n"
                f"⚠️ 它同时会带上 onnxruntime（约 150 MB）—— 这就是本档"
                f"不进 requirements.txt 的原因：不装它镜像只少一档能力，"
                f"装它则是每个镜像都多背 150 MB。",
            ) from exc

        try:
            self._backend: Any = TextEmbedding(model_name=model)
        except Exception as exc:  # noqa: BLE001 —— 见下面关于「为什么这么宽」的说明
            # ⚠️ 这里必须捕获**宽**异常。fastembed 在下面几种情况下抛的异常类型
            # 各不相同：模型名不在支持列表（ValueError）、
            # 首次下载权重失败（网络层的各种异常）、
            # 缓存目录不可写（PermissionError）。它们的共同点是
            # 「**这一档用不了**」，而调用方要区分的是这个，不是异常的类型。
            #
            # 与上一段 `except ImportError` 分开写，是为了让两者的错误信息
            # 各自指向真正的原因（少装包 vs 模型名写错），而不是合并成一句
            # 什么都说不清的「本地档不可用」。
            raise LocalEmbeddingUnavailableError(
                f"本地向量模型 {model!r} 加载失败：{exc}\n"
                f"常见原因：① 模型名不在 fastembed 的支持列表里；"
                f"② 首次使用需要下载权重而网络不通；"
                f"③ 缓存目录不可写。",
            ) from exc

        # ⚠️ 维度校验放在**构造期**，不能省。
        # 配置里写 1024、而模型实际吐 512 维，是这个档最容易犯的错 ——
        # 若拖到第一次写入，Milvus 报的是「维度不匹配」，
        # 而排查方向会被引向集合配置（那里其实是对的）。
        actual = self._probe_dimension()
        if actual != dimensions:
            raise LocalEmbeddingUnavailableError(
                f"本地模型 {model!r} 实际输出 {actual} 维，"
                f"而配置要求 {dimensions} 维。\n"
                f"修法二选一：把 embedding.dimension 与 milvus.dimension "
                f"一起改成 {actual}（并**重建集合**），或换一个 {dimensions} 维的模型。",
            )

    def _probe_dimension(self) -> int:
        """跑一条真实输入，量出模型的输出维度。

        不查 ``fastembed`` 的模型表是因为那张表的字段名随版本变过，
        而「跑一条看看多长」这件事在任何版本上都成立。

        Returns:
            `int`: 实际输出维度。
        """
        vector = next(iter(self._backend.embed(["维度探测"])))
        return int(len(vector))

    async def _call_api(self, inputs: list[str], **kwargs: object) -> EmbeddingResponse:
        """把一批文本映射成向量。

        ⚠️ ``fastembed`` 是**同步**的且会阻塞在 ONNX Runtime 的推理上。
        直接在这里调用会把整个事件循环卡住 —— 而本项目是单进程 asyncio
        服务（``WORKERS=1``），一次嵌入就足以让所有并发请求停顿。
        所以丢进线程池执行。

        ⚠️ 用 ``asyncio.to_thread`` 而不是 ``run_in_executor``：
        前者是 3.9+ 的标准写法，且**会把当前 contextvars 复制进线程**
        （``contextvars`` 在 ``to_thread`` 中被传播）。
        这一点对本项目是实的：``trace_id`` 就在 ContextVar 里
        （``src/observability/context.py``），用 ``run_in_executor``
        的话这段耗时的嵌入调用在 trace 里会是**断的**。

        Args:
            inputs (`list[str]`): 一批文本。
            **kwargs: 基类透传的额外参数，本实现忽略。

        Returns:
            `EmbeddingResponse`: 与 ``inputs`` 等长且同序的向量列表。
        """
        import asyncio

        del kwargs

        def _embed() -> list[list[float]]:
            return [list(map(float, vec)) for vec in self._backend.embed(inputs)]

        embeddings = await asyncio.to_thread(_embed)
        return EmbeddingResponse(
            embeddings=embeddings,
            usage=EmbeddingUsage(tokens=0, time=0.0),
        )


def local_embedding_available() -> bool:
    """本机能否使用本地向量档（只探测**依赖是否安装**）。

    ⚠️ 只探测依赖，**不加载模型**：加载要下载权重、要几百 MB 内存，
    不适合放进一个「问问能不能用」的函数里。
    真正能不能用，以 :class:`LocalOnnxEmbeddingModel` 构造成功为准。

    Returns:
        `bool`: ``fastembed`` 可导入返回 True。
    """
    import importlib.util

    return all(
        importlib.util.find_spec(name) is not None for name in _REQUIRED_PACKAGES
    )


__all__ = [
    "LocalEmbeddingUnavailableError",
    "LocalOnnxEmbeddingModel",
    "local_embedding_available",
]
