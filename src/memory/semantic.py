# -*- coding: utf-8 -*-
"""差旅画像的**语义**部分：用户随口提过、无法结构化的话。

═══ 它管什么、不管什么 ═══

用户说「我上次去深圳住的那个酒店，楼下就有地铁，挺好」。这句话里
没有一个字段能收得住它 —— 但它在下一次订酒店时是有用的。

⚠️ 所以本模块的产出**只作提示**（prompt hints），**绝不**当作事实源：

  · 它不参与任何**校验**（不会因为它就拒绝一次下单）；
  · 它**不覆盖** :mod:`src.memory.profile` 里的结构化字段；
  · 它带着 ``score``，调用方可以选择丢掉低分的。

这条界线是刻意的。把语义召回的结果当事实用，会得到一类很难查的 bug：
用户换了常旅客号，向量库里旧号还留着，检索时两条都回来，
而模型选了**相似度更高**的那条（往往是旧的，因为被重复提到过）。

═══ ⚠️ 为什么用**独立集合**，而不是复用政策知识库那个 ═══

契约里的 ``ALIGO__MILVUS__COLLECTION`` 是政策知识库的集合名，
``scripts/milvus_init.py`` 的**恢复路径**就是「删掉这个集合重跑脚本」
（那是修复「维度建错了」的唯一办法）。如果画像笔记也住在里面，
那么一次为了修政策库而做的删除，会**顺手抹掉所有员工的长期记忆** ——
而这件事不会有任何报错，只会在几天后表现为「助手怎么变笨了」。

所以本模块用**派生**出来的集合名：``{契约集合名}_memory``。

⚠️ 派生而不是新增一个 ``ALIGO__MEMORY__COLLECTION`` 配置键，是想让
「运维需要记住的集合名」**只有一个**。多一个键就多一处可能配错、
且配错后（比如两个集合维度不一致）症状只是「召回质量差」的地方。

⚠️ 维度仍然取 ``settings.milvus.dimension`` —— 与政策库**同一个维度**，
因为向量模型是三合一的同一个（``src/web_embedding``），
给出两个不同的维度只会造成「模型 A 写的、模型 B 读不了」。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from agentscope.message import TextBlock
from agentscope.rag import Chunk, VectorRecord

from src.config.schema import Settings

# ⚠️ 从 ``src.knowledge.store`` 取这个函数而不是在记忆这边另写一份：
# 「把集合的一致性请求成 Strong」是**两个集合**（政策知识库与记忆笔记）
# 共同的需求，实测数据、失败语义与「请求≠生效」的坑都写在那份文档里。
# 抄一份的下场是改一处漏一处。
from src.knowledge.store import request_strong_consistency

from .profile import TravelerProfile

if TYPE_CHECKING:
    from agentscope.embedding import EmbeddingModelBase
    from agentscope.rag import VectorStoreBase

logger = logging.getLogger(__name__)


#: 画像笔记的 metadata 键前缀。
#:
#: ⚠️ 与 :mod:`src.knowledge.manager` 的 ``aligo_kb_id`` / ``aligo_user_id``
#: **刻意不同名**。虽然今天两者不共用一个集合（见模块文档），但
#: 「不共用集合」是一个可以被将来的优化改掉的决定，而键名撞车
#: 一旦发生就是静默的越权读取。多打几个字符换掉这个风险，值。
MEMORY_USER_KEY = "aligo_memory_user"
MEMORY_KIND_KEY = "aligo_memory_kind"


def memory_collection(settings: Settings) -> str:
    """算出画像笔记的集合名。

    ⚠️ 派生规则写在这里、且只有这里，是为了让「集合名从哪来」
    只有一个答案。散在多处的话，某天改了一处就会变成
    「写入进 A、检索查 B」—— 而那种 bug 的表现是「记忆不生效」，
    没有任何异常。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `str`: ``{settings.milvus.collection}_memory``。
    """
    return f"{settings.milvus.collection}_memory"


class MemoryKind:
    """笔记的类别。

    ⚠️ 用 ``str`` 常量而不是 ``Enum``：它的值直接写进 Milvus 的
    metadata JSON，再经 ``metadata_filter`` 的**扁平等值**过滤回来
    （见 ``_milvus_lite.py:504-515``，只支持 ``key == value``）。
    ``Enum`` 会在序列化那一层多出一次转换，而 Milvus 的 filter
    是拿字符串拼表达式的 —— 转换漏一次就变成「永远查不到」。
    """

    #: 偏好类（「我一般坐靠窗」「我不吃辣」）。
    PREFERENCE = "preference"
    #: 观察类（「上次去深圳住的那家楼下有地铁」）。
    OBSERVATION = "observation"
    #: 行程历史（「上季度去了三次上海」）。
    TRIP = "trip"


_VALID_KINDS: frozenset[str] = frozenset(
    {MemoryKind.PREFERENCE, MemoryKind.OBSERVATION, MemoryKind.TRIP},
)


def note_id(user_id: str, text: str) -> str:
    """由 ``(user_id, text)`` 算出一个**确定性**的笔记 id。

    ⚠️ 确定性的意义是**幂等**：同一句话被记住两次，第二次应该
    **覆盖**第一次而不是产生两条。理由不是省空间 —— 而是
    Top-K 检索里两条一模一样的笔记会**挤掉**别的笔记的位置。
    用户越是反复提到某件事，它就越会把其他记忆挤出召回窗口，
    这恰好与直觉相反。

    ⚠️ 确定性还让「同一句话在不同进程/不同副本上算出同一个 id」
    成立 —— 这是能靠 ``upsert`` 语义去重的前提
    （``MilvusLiteStore.insert`` 用 ``sha256(document_id\\0chunk_index)``
    作为主键，见 ``_milvus_lite.py:239-256``）。

    Args:
        user_id (`str`): 员工 id。
        text (`str`): 笔记正文。

    Returns:
        `str`: 64 位十六进制摘要，正好是 Milvus 主键 ``VARCHAR(64)`` 的宽度。
    """
    raw = f"{user_id}\0{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MemoryNote:
    """一条被召回的笔记。

    Attributes:
        text: 正文。
        kind: 类别，见 :class:`MemoryKind`。
        score: 相似度，**越高越相关**。COSINE 度量下取值大致在
            ``[-1, 1]``；``0`` 左右表示正交（即「不相关」）。
        note_id: 确定性 id，见 :func:`note_id`。
    """

    text: str
    kind: str
    score: float
    note_id: str


@dataclass(frozen=True)
class SemanticRecall:
    """一次语义召回的结果。

    ⚠️ 把 ``notes`` 与 ``error`` 放进**同一个**返回值，而不是让失败抛异常。
    理由见 :func:`SemanticMemory.recall` 的文档。
    """

    notes: list[MemoryNote] = field(default_factory=list)
    #: 失败原因；成功时为 ``None``。
    error: str | None = None

    @property
    def ok(self) -> bool:
        """是否成功（哪怕成功但一条没召回）。"""
        return self.error is None


class SemanticMemory:
    """把用户的自由文本记进向量库，并在需要时召回。

    ⚠️ 本类**不自己建向量库客户端，也不自己建 embedding 模型** ——
    两者都从外面传进来。这样做的直接好处是它可以在**没有 Milvus 的
    环境里被测**（传一个 fake store），而更重要的好处是：
    画像的语义部分与政策知识库**共用**同一个客户端与同一个模型，
    于是「模型换了、维度变了」这类问题只会在一个地方发生。
    """

    def __init__(
        self,
        vector_store: "VectorStoreBase",
        embedding_model: "EmbeddingModelBase",
        settings: Settings,
    ) -> None:
        """初始化。

        Args:
            vector_store (`VectorStoreBase`): 向量库客户端（与知识库共用）。
            embedding_model (`EmbeddingModelBase`): 向量模型（与知识库共用）。
            settings (`Settings`): 配置，提供集合名、维度与 ``top_k``。
        """
        self._store = vector_store
        self._embedding = embedding_model
        self._settings = settings

        #: 本进程是否**尝试过**把笔记集合的一致性改成 Strong（见
        #: :func:`src.knowledge.store.request_strong_consistency`）。
        #:
        #: ⚠️ 为什么是「尝试过」而不是「已生效」、又为什么**每个进程只做一次**：
        #: 改一致性属于 DDL（服务端要落一次元数据），而 ``remember`` 是
        #: 用户直接等着的写路径 —— 每次写入都发一条 DDL 既浪费又会与
        #: 其他写者抢同一把 DDL 锁。集合是**本条写路径**惰性创建的
        #: （见 ``remember`` 里的长注释），所以第一笔写入就是请求它的
        #: 时机；此后本进程内不再重复。部署期的正式入口是
        #: ``scripts/milvus_init.py``（那边每次都会请求）。
        self._consistency_attempted = False

    @property
    def collection(self) -> str:
        """画像笔记的集合名。

        Returns:
            `str`: 见 :func:`memory_collection`。
        """
        return memory_collection(self._settings)

    # ------------------------------------------------------------------
    # 写
    # ------------------------------------------------------------------
    async def remember(
        self,
        user_id: str,
        text: str,
        kind: str = MemoryKind.PREFERENCE,
    ) -> MemoryNote:
        """记住一句话。

        ⚠️ 写入前先 ``delete``。这一条**只**解决一件事：同一句原文
        被记两次时，第二次**覆盖**第一次而不是留下两条。
        为什么在意这个 —— Top-K 检索里两条一模一样的笔记会挤掉
        别的笔记的位置，于是「用户越是反复提到某件事，它就越把
        其他记忆挤出召回窗口」，恰好与直觉相反。

        ⚠️⚠️ 它**不解决**「用户改口了」这件事，这一点必须说清楚：
        用户说「我不吃辣」，三个月后说「我现在能吃辣了」——那是
        两句**不同**的原文，两个不同的 ``note_id``，**旧的那条会
        一直留在库里**，检索时和新的一条一起被召回，
        模型于是看到自相矛盾的两句话。

        这个限制是真实的，不是待办。要消掉旧的那条，调用方必须
        **显式**先 :meth:`recall` 找到它、再 :meth:`forget` ——
        也就是需要一次模型参与的判断（「用户是在补充还是在改口」），
        而那属于对话层的职责，不该由存储层猜。
        「按 kind 整体覆盖」看着像是解法，实际上更糟：座位偏好与
        饮食偏好同属 ``PREFERENCE``，覆盖会连带删掉不相干的那条。

        ⚠️ 集合不存在时本方法会**先把它建出来**（幂等），理由与实测见
        方法体内的注释。别把这步挪到装配期或读路径 —— 那里各有各的代价。

        Args:
            user_id (`str`): 员工 id。
            text (`str`): 笔记正文。
            kind (`str`, optional): 类别，见 :class:`MemoryKind`。

        Returns:
            `MemoryNote`: 写入的笔记（``score`` 为 1.0 —— 自己跟自己最像）。

        Raises:
            ValueError: ``kind`` 不是合法类别；或 ``text`` 为空。
            Exception: 向量库/模型不可用时的原始异常（见类文档与
                :meth:`recall` 的对比）。
        """
        cleaned = text.strip()
        if not cleaned:
            raise ValueError("要记住的内容不能为空。")
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"笔记类别 {kind!r} 不合法。合法值："
                f"{'、'.join(sorted(_VALID_KINDS))}。",
            )

        identifier = note_id(user_id, cleaned)

        # ⚠️⚠️ 写之前先确保集合存在。**这不是防御性编程**，是修一个
        # 2026-10-03 在容器里实测到的真实故障：``scripts/milvus_init.py``
        # 当时只建政策知识库那个集合，于是本方法的第一条 RPC（下面的
        # ``delete``）就以 ``MilvusException: collection not found[..._memory]``
        # 失败 —— 「记住这个」这条**用户明确要求**的路径在全新部署上
        # 一次都没成功过，而同一时刻每次对话的 :meth:`recall` 也在
        # 各打一条 ERROR 级日志。
        #
        # 为什么放在**这里**（写路径），而不是另外两个看起来更自然的地方：
        #
        #   · 装配期（``src/memory/__init__.py::build_memory``）不行 ——
        #     那边承诺「不做任何 I/O」，建集合会把「Milvus 没起来」
        #     变成「应用起不来」；
        #   · 读路径不行 —— ``recall`` 每次对话都跑，白花一次 RPC，
        #     而读失败本来就已经降级成「没有记忆」（见 :meth:`recall`），
        #     不需要集合存在；
        #   · 只靠 ``milvus_init`` 不行 —— 它覆盖的是「运维记得跑」，
        #     而实测漏掉的恰恰是这一步。
        #
        # 代价说清楚：框架的 ``create_collection`` 在集合已存在时是
        # **no-op**（``rag/_vdb/_milvus_lite.py:129-131``），成本是里面
        # 那次 ``has_collection`` RPC（毫秒级）；而本方法紧接着必然要发
        # 一次 embedding 请求（下一行的 ``self._embedding``），网络往返
        # 比它大一个数量级。收益是「集合被删了 / 从没建过」都能自愈。
        #
        # ⚠️ 它经 :class:`~src.knowledge.guard.GuardedVectorStore` 被箍住
        # （``create_collection`` 在 ``_GUARDED_OPERATIONS`` 里）⇒ 有超时
        # 与熔断：Milvus 挂了这里是**快速失败**，不是挂住。
        # 失败照常抛 —— 写路径的语义就是「不让用户以为记住了」。
        await self._store.create_collection(
            name=self.collection,
            dimensions=self._settings.milvus.dimension,
        )

        # ⚠️ 集合刚建出来时，一致性级别是 Milvus 的默认值 Bounded ——
        # 「刚记住的这句话，下一次召回可能还查不到」（实测窗口最长 1.37s，
        # 数据见 src/knowledge/store.py::CONSISTENCY_LEVEL）。这不是理论
        # 问题：POST /api/v1/memory/notes 返回 200 之后紧接着的
        # GET 就可能读不到它。所以建完立刻请求 Strong。
        #
        # ⚠️ 只在**本进程第一次写入**时请求（见 ``_consistency_attempted``
        # 的文档）。失败只告警不抛：退化掉的是「立刻可见」这个保证，
        # 而不是这次写入本身 —— 让一个可见性优化把「记住」变成不可用，
        # 方向是反的（``request_strong_consistency`` 的文档里写了同样的理由）。
        if not self._consistency_attempted:
            self._consistency_attempted = True
            await request_strong_consistency(self._store, self.collection)

        # ⚠️ 先删后写。``delete`` 按 ``document_id`` 过滤，
        # 匹配不到任何记录时是一次 no-op（不抛）—— 见
        # ``_milvus_lite.py:257-266``，它走的是 ``client.delete(filter=...)``。
        await self._store.delete(self.collection, identifier)

        vectors = await self._embedding([cleaned])
        await self._store.insert(
            self.collection,
            [
                VectorRecord(
                    vector=vectors.embeddings[0],
                    document_id=identifier,
                    chunk=Chunk(
                        content=TextBlock(type="text", text=cleaned),
                        source=f"memory:{kind}",
                        # ⚠️ 一条笔记一个 chunk。``chunk_index`` 参与
                        # 主键的计算，固定 0 才能真正做到「同一句话
                        # 覆盖同一句话」。
                        chunk_index=0,
                        total_chunks=1,
                        metadata=self._metadata(user_id, kind),
                    ),
                ),
            ],
        )
        return MemoryNote(
            text=cleaned,
            kind=kind,
            score=1.0,
            note_id=identifier,
        )

    async def forget(self, user_id: str, text: str) -> None:
        """忘掉一句话（按原文精确匹配）。

        ⚠️ 只能按**原文**忘。用户说「忘掉我不吃辣那件事」时，
        调用方需要先 :meth:`recall` 拿到原文再调这里 —— 这是刻意的：
        让「忘记」有一个必须显式给出目标的形状，而不是
        「按相似度删」那种会误删的语义。

        Args:
            user_id (`str`): 员工 id。
            text (`str`): 要忘掉的原文。
        """
        await self._store.delete(self.collection, note_id(user_id, text.strip()))

    async def forget_all(self, user_id: str) -> int:
        """忘掉某个用户的**全部**笔记。

        ⚠️ 绝不 ``delete_collection`` —— 那是 :mod:`src.knowledge.manager`
        文档里记着的那类数据丢失事故的另一副面孔：这个集合里
        住着**所有**用户的记忆。

        Args:
            user_id (`str`): 员工 id。

        Returns:
            `int`: 删掉的笔记条数。
        """
        summaries = await self._store.list_documents(
            self.collection,
            metadata_filter={MEMORY_USER_KEY: user_id},
        )
        for summary in summaries:
            await self._store.delete(self.collection, summary.document_id)
        return len(summaries)

    # ------------------------------------------------------------------
    # 读
    # ------------------------------------------------------------------
    async def recall(
        self,
        user_id: str,
        query: str,
        top_k: int | None = None,
        min_score: float = 0.0,
    ) -> SemanticRecall:
        """召回与 ``query`` 相关的笔记。

        ⚠️⚠️ **本方法永不抛异常**，失败时返回 ``error`` 非空的
        :class:`SemanticRecall`。这一点与 :meth:`remember` 相反，
        是刻意的不对称：

          · **写**失败必须让调用方知道 —— 用户明确说了「记住这个」，
            静默丢掉是一次欺骗；
          · **读**失败只影响「锦上添花」。召回发生在每一次对话的
            上下文注入路径上（``ContextInjectionMiddleware``），
            如果它抛异常，**整个对话**会因为「画像笔记暂时读不出来」
            而失败。用户会看到「助手挂了」，而实际上订票链路完全正常。

        这正是 P4 验收里「Milvus 不可用不得拖垮 /readyz」的同一条原则，
        只是发生在请求路径上而不是探测路径上。

        ⚠️ ``metadata_filter`` 是**隔离**而不是优化：``{MEMORY_USER_KEY: user_id}``
        让每个人只能召回自己的笔记。漏掉它的后果不是「多召回几条」，
        而是 A 员工读到 B 员工的差旅记录 —— 一次数据泄露。

        Args:
            user_id (`str`): 员工 id。
            query (`str`): 查询文本（通常是本轮用户输入）。
            top_k (`int | None`, optional): 返回条数；默认取
                ``settings.memory.top_k``。
            min_score (`float`, optional): 低于此分数的丢弃。
                默认 ``0.0`` —— COSINE 下 0 表示「正交」，
                也就是「毫无关系」。定得更高（比如 0.5）会让
                召回率骤降且很难解释为什么，所以默认不设门槛。

        Returns:
            `SemanticRecall`: 召回结果；失败时 ``notes`` 为空、
            ``error`` 有值。
        """
        cleaned = query.strip()
        if not cleaned:
            # ⚠️ 空查询**不是**错误：它只是没什么可召回的。
            # 返回 error 会让调用方在日志里看到一堆「失败」，
            # 而那些其实是空对话。
            return SemanticRecall()

        limit = top_k if top_k is not None else self._settings.memory.top_k

        try:
            vectors = await self._embedding([cleaned])
            results = await self._store.search(
                self.collection,
                query_vector=vectors.embeddings[0],
                top_k=limit,
                metadata_filter={MEMORY_USER_KEY: user_id},
            )
        except Exception as exc:  # noqa: BLE001 —— 见本方法的文档
            # ⚠️ 异常**不吞**：它被放进返回值，调用方可以选择记日志、
            # 打指标或者在 /readyz 里暴露出来。吞掉才是真的危险。
            from src.observability.redaction import safe_error

            logger.warning("画像语义召回失败（已降级为「没有记忆」）：%s", safe_error(exc))
            return SemanticRecall(error=safe_error(exc))

        notes: list[MemoryNote] = []
        for result in results:
            if result.score < min_score:
                continue
            notes.append(
                MemoryNote(
                    text=_text_of(result.chunk),
                    kind=str(result.chunk.metadata.get(MEMORY_KIND_KEY, "")),
                    score=result.score,
                    note_id=result.document_id,
                ),
            )
        return SemanticRecall(notes=notes)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    @staticmethod
    def _metadata(user_id: str, kind: str) -> dict[str, str]:
        """构造写进向量库的 metadata。

        ⚠️ 只有**扁平的两个字符串键**。``metadata_filter`` 走到
        Milvus 那边是 ``metadata["k"] == "v"`` 拼出来的表达式
        （``_milvus_lite.py:504-515``），只支持等值 —— 塞进去一个
        列表或嵌套 dict 不会报错，但那个键永远匹配不上，
        表现为「隔离失效」（filter 拼出来的表达式在 Milvus 侧语法错误，
        搜索直接失败）或者「永远查不到」。

        Args:
            user_id (`str`): 员工 id。
            kind (`str`): 笔记类别。

        Returns:
            `dict[str, str]`: 两个扁平键。
        """
        return {MEMORY_USER_KEY: user_id, MEMORY_KIND_KEY: kind}


def _text_of(chunk: Chunk) -> str:
    """从 chunk 里取出文本。

    ⚠️ ``Chunk.content`` 是 ``TextBlock | DataBlock``，**不是** ``str``
    （``rag/_document.py:80-82``）。直接 ``str(chunk.content)`` 会得到
    ``"text='...' type='text'"`` 这种 pydantic repr —— 它看起来像内容，
    于是这个错误能一路活到 prompt 里，只是把一堆引号和字段名喂给了模型。

    Args:
        chunk (`Chunk`): 待取文本的 chunk。

    Returns:
        `str`: 文本内容；``DataBlock``（图片/音视频）返回空串 ——
        画像笔记只可能是文本，遇到别的说明有人往这个集合里写了脏数据，
        与其把二进制描述塞进 prompt，不如丢掉。
    """
    content = chunk.content
    if isinstance(content, TextBlock):
        return content.text
    return ""


def render_profile_section(
    profile: TravelerProfile | None,
    recall: SemanticRecall,
    max_notes: int = 3,
) -> str:
    """把结构化画像与语义召回渲染成一段可以塞进 Prompt 的文本。

    ⚠️ 输出的这段文本会被 ``ContextInjectionMiddleware`` 放进
    **系统提示**里，因此它必须满足两个条件：

      1. **没有信息就不要输出**（返回空串，而不是「用户画像：无」）——
         空话占 token 且稀释指令；
      2. **结构化优先于语义**。两者冲突时以结构化为准，
         并且要把这件事**写出来** —— 见下面的实现。

    ⚠️ 语义笔记**限制条数**（``max_notes``）。召回的 ``top_k`` 与
    塞进 prompt 的条数是两个数：多召回几条是为了让调用方有得挑，
    但全塞进去会让提示词被历史细节淹没，而当前这一轮真正该做的事
    （订一张去北京的票）反而被挤到后面。

    Args:
        profile (`TravelerProfile | None`): 结构化画像；没有则 None。
        recall (`SemanticRecall`): 语义召回结果（可能失败）。
        max_notes (`int`, optional): 最多渲染几条笔记。

    Returns:
        `str`: 可直接拼接的文本段；没有任何信息时是**空串**。
    """
    lines: list[str] = []

    if profile is not None and not profile.is_empty():
        lines.append("【已确认的画像（权威，优先级高于下面的备注）】")
        if profile.preferred_airlines:
            # ⚠️ 只在**有**偏好的时候才提「按优先级排序」——
            # 没有列表时这句话是一句噪声。
            lines.append(
                f"· 偏好航司（按优先级）：{' > '.join(profile.preferred_airlines)}",
            )
        if profile.preferred_cabin:
            lines.append(f"· 偏好舱位：{profile.preferred_cabin}")
        if profile.seat_preference:
            lines.append(f"· 座位偏好：{profile.seat_preference}")
        if profile.preferred_hotel_brands:
            lines.append(f"· 偏好酒店品牌：{'、'.join(profile.preferred_hotel_brands)}")
        if profile.dietary_needs:
            lines.append(f"· 饮食需求：{'、'.join(profile.dietary_needs)}")
        if profile.frequent_flyer_numbers:
            # ⚠️ 常旅客号是**硬事实**，必须精确。原样输出、不做任何
            # 截断或格式化 —— 一个被「美化」过的卡号是订不上里程的。
            pairs = "、".join(
                f"{airline} {number}"
                for airline, number in sorted(profile.frequent_flyer_numbers.items())
            )
            lines.append(f"· 常旅客号（须原样使用，不得改写）：{pairs}")
        if profile.cost_center:
            lines.append(f"· 成本中心：{profile.cost_center}")
        if profile.default_approver:
            lines.append(f"· 默认审批人：{profile.default_approver}")
        if profile.accessibility_needs:
            lines.append(f"· 无障碍需求：{profile.accessibility_needs}")

    usable = [note for note in recall.notes if note.text][:max_notes]
    if usable:
        lines.append("")
        lines.append("【用户过往零散提及（**仅作参考**，与上面冲突时以上面为准）】")
        for note in usable:
            lines.append(f"· {note.text}")

    if not lines:
        # ⚠️ 返回空串，不是 None、不是「无」。调用方多半会做
        # ``"\n".join([base_prompt, section])``，返回 None 会炸，
        # 返回「无」会污染 prompt。
        return ""

    return "\n".join(lines)


__all__ = [
    "MEMORY_KIND_KEY",
    "MEMORY_USER_KEY",
    "MemoryKind",
    "MemoryNote",
    "SemanticMemory",
    "SemanticRecall",
    "memory_collection",
    "note_id",
    "render_profile_section",
]
