# -*- coding: utf-8 -*-
"""Milvus 向量库的构造、幂等建集合与**实际形态**核验。

═══ ⚠️ 这里没有自建 store ═══

框架的 :class:`agentscope.rag.MilvusLiteStore` 内部就是
``pymilvus.MilvusClient(uri=...)``（``agentscope/rag/_vdb/_milvus_lite.py:78-84``）。
它的名字有误导性 —— 决定「走内嵌 Lite 还是连服务端」的是
``_is_local_db_uri``（``:538-543``）::

    not uri.startswith(("http://", "https://")) and splitext(uri)[1] == ".db"

对我们的 ``http://milvus:19530`` 这个判据返回 **False**（既以 http 开头、
结尾也不是 ``.db``）⇒ 走**远端服务**分支，连的就是 compose 里的 Milvus。
所以「类名带 Lite」不代表只能跑内嵌模式，本模块直接复用即可。

═══ ⚠️ 幂等建集合的真正难点不是「建」，是「建错了怎么办」 ═══

``create_collection`` 在集合已存在时是 **no-op**（``:130-132``）。这意味着
「重复执行初始化脚本」天然幂等 —— 但也意味着**一个用错维度建出来的集合
永远不会被修正**。

后果分两种，第二种更坏：

  1. 维度偏小 —— 第一次写入就报「维度不匹配」，属于**好**的失败。
  2. 维度相同但**度量/索引**与配置不同（比如建的时候是 L2、配置写着 COSINE）
     —— 写入成功、检索成功、有分数。只是分数的大小关系没有意义。
     这是**坏**的失败：没有任何一处会报错。

所以本模块除了「建」，还必须能「**读回来对**」——
:func:`describe_collection` 就是干这个的。初始化脚本用它做建后自检。
"""

from __future__ import annotations

import logging
from typing import Any

from agentscope.rag import MilvusLiteStore

from src.config.schema import Settings
from src.knowledge.guard import GuardedVectorStore, guard_vector_store
from src.observability.redaction import redact, safe_error

logger = logging.getLogger(__name__)

#: Milvus 的一致性级别：整数 ↔ 人话。
#:
#: ⚠️ 这几个数字来自 Milvus 的 proto 定义（``ConsistencyLevel`` 枚举），
#: 客户端只做透传。写在这里是为了让 ``describe_collection`` 的返回值
#: 与 ``/readyz`` 的响应**可读** —— 一个裸的 ``2`` 没有人能看出它是
#: 「Bounded」，而正是这个默认值造成了下面这条真实故障。
_CONSISTENCY_NAMES: dict[int, str] = {
    0: "Strong",
    1: "Session",
    2: "Bounded",
    3: "Eventually",
}

#: 本项目请求的一致性级别。
#:
#: ⚠️ 2026-10-03 在容器里实测出来的：Milvus 的默认一致性是 **Bounded**
#: （``describe_collection`` 返回 2），它意味着**写入对检索不是立刻可见**。
#: 同一台机器上量到的窗口：插入后 0.39s / 0.40s / 1.37s 才被召回，
#: 删除后 0.56s / 0.57s / 0.58s 才不再被召回 —— 也就是说
#: ``DELETE /api/v1/memory/notes`` 返回 200「已忘记」之后，紧接着的
#: 召回**仍可能命中那条已被删掉的笔记**。对长期记忆这个功能来说这是
#: 不可接受的：用户刚说完「忘掉这个」，助手下一句又提起它。
#:
#: 改成 Strong 之后复测 3 轮：插入后立刻可见、删除后立刻不可见，
#: **全部即时**，召回耗时 101–186ms（主要是 embedding 的网络往返，
#: 一致性带来的增量在噪声里）。这就是这里选 Strong 的全部理由。
CONSISTENCY_LEVEL = "Strong"


def build_vector_store(settings: Settings) -> GuardedVectorStore:
    """按配置构造 Milvus 向量库客户端，并**包上检索护栏**。

    ⚠️ 构造**不发任何网络请求** —— ``MilvusLiteStore`` 的 ``__init__``
    只存参数，真正的 ``pymilvus.MilvusClient`` 是在 ``get_client()``
    里惰性创建的（``:78-84``，``pymilvus`` 甚至只在 ``TYPE_CHECKING``
    下导入）。这一点对本项目是实的：``/readyz`` 与启动日志都构造它，
    而它们不该因为 Milvus 没起来就崩。护栏同样不发请求（它只存一个超时
    与一个熔断器引用），所以这条约束在加护栏之后依然成立。

    ⚠️ 护栏加在**这个工厂**里，而不是各个调用点：调用点有两处
    （``src/knowledge/__init__.py`` 的知识库、``src/memory/__init__.py``
    的长期记忆画像），将来还可能有第三处。加在这里，**拿不到没护栏的
    store** —— 把「记得加超时」从人的自觉变成结构上的必然。

    Args:
        settings (`Settings`): 配置。超时取自
            ``settings.milvus.search_timeout_seconds``。

    Returns:
        `GuardedVectorStore`: 未连接的向量库客户端（读操作有超时与熔断，
        写操作原样透传）。
    """
    return guard_vector_store(
        _build_raw_store(settings),
        timeout=settings.milvus.search_timeout_seconds,
    )


def _build_raw_store(settings: Settings) -> MilvusLiteStore:
    """构造**不带护栏**的框架客户端。

    ⚠️ 独立成一个私有函数，是为了让「护栏」与「框架客户端的构造参数」
    两件事各占一个函数：改动其中之一时不必在同一段代码里分辨哪些参数
    属于谁。对外请用 :func:`build_vector_store`。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `MilvusLiteStore`: 未连接的向量库客户端。
    """
    return MilvusLiteStore(
        uri=settings.milvus.uri,
        metric_type=settings.milvus.metric_type,
        # ⚠️ 框架的默认值是 ``AUTOINDEX``，而本项目要的是 HNSW
        # （``MilvusSettings.index_type`` 的默认值，也是契约值）。
        # 必须显式传 —— 不传就会静默地建成 AUTOINDEX。
        #
        # ⚠️ 另注：``HNSW`` 这个字符串在 Milvus 后端里**到处都没被特殊处理**，
        # 只是原样透传给 ``index_params.add_index(index_type=...)``
        # （``:161-165``）。所以它能不能生效完全取决于服务端支持不支持 ——
        # 这也是 :func:`describe_collection` 要回读核验的原因之一。
        index_type=settings.milvus.index_type,
    )


async def request_strong_consistency(
    store: MilvusLiteStore,
    collection: str,
) -> bool:
    """请求把集合的一致性级别设为 :data:`CONSISTENCY_LEVEL`（幂等）。

    ⚠️ 为什么需要它：框架的 ``create_collection`` 不传
    ``consistency_level``（``agentscope/rag/_vdb/_milvus_lite.py:108-166``），
    服务端于是用默认值 Bounded —— 写进去的东西**要过一会儿才能被检索到**。
    实测数据与后果见 :data:`CONSISTENCY_LEVEL` 的注释。这里是我们唯一
    能纠正它的地方：框架的接口里没有这个旋钮，而集合一旦建好，
    一致性级别不会跟着配置走。

    ⚠️ **「返回成功」不等于「已生效」** —— 这是实测踩过的坑，也是本函数
    返回值只敢叫「请求已发出」的原因：

      · ``alter_collection_properties`` 对**不认识的属性名**不报错
        （实测传 ``{"collection.nonexistent.property": "x"}`` 返回 None，
        且该键真的出现在了 ``properties`` 里）—— 服务端不校验键名；
      · 改完之后 ``describe_collection`` 会**先返回旧值**，过一会儿才变。

    所以这里既不能用返回值证明生效，也不能拿紧随其后的 ``describe``
    去断言。真正的核验在 ``/readyz`` 与 ``scripts/milvus_init.py`` 的输出里
    （它们每次都回读），而不是在这里。

    ⚠️ 失败**只告警、不抛**。这不是「静默降级」：写路径上真正的失败
    （``insert`` / ``delete`` 抛错）照常抛、照常让接口返回 503，
    这里退化掉的只是**可见性保证**（从「立刻可见」退到「一秒内可见」），
    数据本身一条不少。反过来，如果在这里抛，一个不支持该属性的
    Milvus 版本会让**所有**记忆写入直接失败 —— 用一个优化项
    换取功能不可用，方向是反的。

    Args:
        store (`MilvusLiteStore`): 向量库客户端。
        collection (`str`): 集合名。

    Returns:
        `bool`: 请求是否**发出**（``True``）或被跳过/失败（``False``）。
        ⚠️ 不代表已经生效，见上。
    """
    import asyncio

    try:
        # ⚠️ 与 ``describe_collection`` 同一写法、同一理由：``get_client()``
        # 可能触发一次**同步**建连，直接调用会按住事件循环。
        client = await asyncio.to_thread(store.get_client)
        await asyncio.to_thread(
            client.alter_collection_properties,
            collection_name=collection,
            properties={"collection.consistency_level": CONSISTENCY_LEVEL},
        )
    except Exception as exc:  # noqa: BLE001 —— 见上面「失败只告警、不抛」
        logger.warning(
            "集合 %s 的一致性级别未能请求为 %s（检索可见性可能延迟约 1 秒）：%s",
            collection,
            CONSISTENCY_LEVEL,
            safe_error(exc),
        )
        return False
    return True


async def ensure_collection(
    store: MilvusLiteStore,
    settings: Settings,
    *,
    collection: str | None = None,
) -> bool:
    """幂等地建出契约集合。

    ⚠️ 「幂等」= 集合已存在就不动它，**不是**「确保它是对的」——
    见模块文档。要核验形态请用 :func:`describe_collection`。

    Args:
        store (`MilvusLiteStore`): 向量库客户端。
        settings (`Settings`): 配置。
        collection (`str | None`): 要建的集合名；``None`` 时是政策知识库的
            契约集合（``settings.milvus.collection``）。

            ⚠️ 这个参数存在的唯一理由是**长期记忆画像**那个集合：它的名字是
            ``{契约集合}_memory``（``src/memory/semantic.py::memory_collection``），
            而它的**形态约束与政策库逐项相同** —— 同一个 Milvus、同一个向量模型、
            同一个维度。所以这里只换名字，不换期望值；给记忆另立一份期望值
            就等于把「维度必须一致」这件事抄了两遍，改一处漏一处。

    Returns:
        `bool`: 本次调用**是否真的创建了**集合（``False`` 表示它本来就存在）。

    Raises:
        Exception: Milvus 不可达、鉴权失败等 —— 由调用方决定如何处理。
            ⚠️ 本函数**不吞异常**：初始化脚本要让上层以非零码退出，
            而 ``/readyz`` 走的是 :func:`probe_collection`，不是这里。
    """
    name = collection or settings.milvus.collection
    existed = await store.has_collection(name)
    await store.create_collection(
        name=name,
        dimensions=settings.milvus.dimension,
    )
    # ⚠️ 一致性级别只在**建集合那一刻**能定（框架的建集合接口没有这个
    # 参数，服务端默认给 Bounded）。所以这一步放在这里，而且**不区分**
    # 集合是不是刚建的：老的集合同样需要被纠正一次 —— 这正是「请求」
    # 而非「设置」的语义（见 request_strong_consistency 的文档）。
    await request_strong_consistency(store, name)
    if existed:
        logger.info("集合 %s 已存在，未改动。", name)
    else:
        logger.info(
            "集合 %s 已创建（dim=%d，index=%s，metric=%s）。",
            name,
            settings.milvus.dimension,
            settings.milvus.index_type,
            settings.milvus.metric_type,
        )
    return not existed


async def describe_collection(
    store: MilvusLiteStore,
    collection: str,
) -> dict[str, Any]:
    """把集合在 Milvus 里的**实际形态**读回来。

    ⚠️ 这个函数存在的唯一理由是「``create_collection`` 会静默 no-op」。
    配置说 1024/HNSW/COSINE，而线上那个集合可能是半年前用 768/L2 建的 ——
    此时一切照跑，只是检索质量不对。**只有回读能发现它。**

    做法是绕过框架直接问 ``pymilvus``：框架的 ``VectorStoreBase`` 接口里
    没有「描述集合」这个方法，而这是运维侧的需求，不属于检索抽象。

    Args:
        store (`MilvusLiteStore`): 向量库客户端（会触发一次真连接）。
        collection (`str`): 集合名。

    Returns:
        `dict[str, Any]`: 含 ``exists`` / ``dimension`` / ``metric_type`` /
        ``index_type`` / ``index_name`` 的字典。
        ⚠️ 无法读到的字段给 ``None`` 而不是猜测 ——
        一个假的值比一个缺失的值危险得多（见下面的实现说明）。
    """
    import asyncio

    result: dict[str, Any] = {
        "collection": collection,
        "exists": False,
        "dimension": None,
        "metric_type": None,
        "index_type": None,
        "index_name": None,
        # ⚠️ 一致性级别也回读。它不是「形态」的一部分（不影响正确性），
        # 但决定了**刚写进去的东西多久能被检索到** —— 默认的 Bounded
        # 会造成约 1 秒的可见性窗口（实测见 CONSISTENCY_LEVEL 的注释），
        # 而这是唯一能看见它的地方。
        "consistency_level": None,
    }

    # ⚠️ ``get_client()`` 可能触发一次**同步**建连（框架惰性创建，见
    # ``agentscope/rag/_vdb/_milvus_lite.py:78-84``）。在这里直接调用会把事件循环按住，
    # 于是本函数外层的 ``asyncio.wait_for``（:func:`probe_collection`）
    # 根本等不到超时 —— 循环被占着，定时器回调没机会跑。丢进线程里，
    # 超时才真的算数。正常路径上客户端已被护栏预热过，这里只是一次
    # 便宜的属性读取。
    client = await asyncio.to_thread(store.get_client)

    def _read() -> dict[str, Any]:
        info: dict[str, Any] = {}
        if not client.has_collection(collection_name=collection):
            return info

        info["exists"] = True

        described = client.describe_collection(collection_name=collection)

        # ⚠️ 优先用服务端给的 ``consistency_level_name``：不同 pymilvus
        # 版本对 ``consistency_level`` 的编码见过 int / str 两种写法，
        # 而名字是稳定的。名字也没有时才退回映射表，最后才把原值透出
        # —— 读不到就留 None，不猜（与维度、索引同一取舍）。
        raw_level = described.get("consistency_level")
        info["consistency_level"] = (
            described.get("consistency_level_name")
            or _CONSISTENCY_NAMES.get(raw_level)
            or raw_level
        )

        for field in described.get("fields", []):
            if field.get("name") != "vector":
                continue
            # ⚠️ pymilvus 把维度放在 ``params`` 里，而 ``params`` 的键名
            # 在不同版本间见过 ``dim`` 与 ``dimension`` 两种写法。
            # 所以两种都试，取不到就留 None —— 不猜。
            params = field.get("params") or {}
            dim = params.get("dim", params.get("dimension"))
            info["dimension"] = int(dim) if dim is not None else None

        # ⚠️ 索引信息要分两步问：先列名字，再按名字描述。
        # 集合可能一个索引都没有（维度校验失败而中断的建集合），
        # 那时两个调用都不该被走到。
        indexes = client.list_indexes(collection_name=collection)
        if indexes:
            index_name = indexes[0]
            info["index_name"] = index_name
            try:
                described_index = client.describe_index(
                    collection_name=collection,
                    index_name=index_name,
                )
                info["metric_type"] = described_index.get("metric_type")
                info["index_type"] = described_index.get("index_type")
            except Exception as exc:  # noqa: BLE001
                # ⚠️ 索引描述失败**不**让整个核验失败：维度是更要紧的信息，
                # 而「索引读不出来」本身就是一条值得报出去的发现。
                info["index_error"] = str(exc)
        return info

    result.update(await asyncio.to_thread(_read))
    return result


def verify_collection_shape(
    actual: dict[str, Any],
    settings: Settings,
    *,
    collection: str | None = None,
) -> list[str]:
    """把实际形态与配置逐项比对，返回**人话**的差异清单。

    ⚠️ 返回清单而不是抛异常，是为了让调用方能决定严重性：
    初始化脚本应当因此以非零码退出（这是发布前的最后一道闸），
    而 ``/readyz`` 只应把它记成一条警告（集合不合规不该让整个服务
    被 LB 摘掉 —— 此时对话、订单、审批都还是好的）。

    ⚠️ 字段为 ``None``（读不到）时**不算不符**。理由是分辨不了
    「值错了」与「pymilvus 版本不同读不出来」，而把后者报成「配置错误」
    会让人去改一份本来正确的配置 —— 比不报更糟。

    Args:
        actual (`dict[str, Any]`): :func:`describe_collection` 的返回值。
        settings (`Settings`): 配置。
        collection (`str | None`): 被核验的集合名（只影响报告里的措辞）；
            ``None`` 时取契约集合。与 :func:`ensure_collection` 的同名参数
            成对使用 —— 期望值两者共用（见那边的说明）。

    Returns:
        `list[str]`: 每条差异一句中文说明；完全一致时返回空列表。
    """
    problems: list[str] = []
    name = collection or settings.milvus.collection

    if not actual.get("exists"):
        return [f"集合 {name!r} 不存在。"]

    checks = (
        ("dimension", settings.milvus.dimension, "维度"),
        ("metric_type", settings.milvus.metric_type, "度量"),
        ("index_type", settings.milvus.index_type, "索引类型"),
    )
    for key, expected, label in checks:
        found = actual.get(key)
        if found is None:
            continue
        if found != expected:
            problems.append(
                f"{label}不符：集合里是 {found!r}，配置要求 {expected!r}。",
            )

    if problems:
        problems.append(
            "⚠️ 集合的维度/度量/索引在建集合那一刻定死，改配置不会改变已存在的集合。"
            f"修法是删掉集合 {name!r} 重建 "
            f"（会丢已索引的向量，需要重新灌数据），"
            f"或用 ALIGO__MILVUS__* 改成与现有集合一致。",
        )
    return problems


async def probe_collection(
    settings: Settings,
    *,
    timeout: float = 3.0,
) -> dict[str, Any]:
    """给 ``/readyz`` 用：**永不抛异常**地报告 Milvus 是否可用。

    ⚠️ 三条设计约束，每条都是被真实故障逼出来的：

    1. **永不抛**。``/readyz`` 是一个探针，探针炸掉等于服务不可探。
       Milvus 没起、网络不通、鉴权失败、pymilvus 没装 —— 全部收敛成
       ``{"ok": False, "error": "..."}``。

    2. **有超时**。``MilvusClient`` 的默认连接超时很长，若它卡住，
       ``/readyz`` 会一直挂着 —— 而 LB 会把这个「没响应」读成
       「进程死了」，从而摘掉一个**完全健康**的实例。
       向量库不可用**不该**拖垮整个服务：对话、订单、审批、鉴权都不经过它。

    3. **不复用长连接**。每次探针现建一个客户端、建完就关。
       复用的话，一个已经半死的连接会被反复返回「可用」，
       直到某次真实检索才暴露 —— 探针就失去了意义。

    Args:
        settings (`Settings`): 配置。
        timeout (`float`): 单次探测的总超时（秒）。

    Returns:
        `dict[str, Any]`: 至少含 ``ok``；失败时含 ``error``；
        成功时含 ``collection_exists`` 与 :func:`describe_collection` 的字段。
    """
    import asyncio

    async def _probe() -> dict[str, Any]:
        store = build_vector_store(settings)
        try:
            exists = await store.has_collection(settings.milvus.collection)
            if not exists:
                # ⚠️ 集合不存在**不算** Milvus 不可用：
                # 连接是通的，只是还没初始化。运维需要能区分这两件事 ——
                # 前者要改代码/配置，后者只要跑一次初始化脚本。
                return {
                    "ok": True,
                    "collection_exists": False,
                    "note": "Milvus 可达，但集合尚未创建（跑 scripts/milvus_init.py）。",
                }
            return {
                "ok": True,
                "collection_exists": True,
                **(await describe_collection(store, settings.milvus.collection)),
            }
        finally:
            # ⚠️ 必须关：每次探针都建新客户端，不关的话连接会一直堆积。
            # 用 ``__aexit__`` 而不是 ``close()``：框架把关闭逻辑
            # （含内嵌 Lite 的 server 释放）都放在 ``__aexit__`` 里。
            try:
                await store.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                # ⚠️ 关闭失败不能盖掉探测结果 —— 它只是一次清理。
                logger.debug("关闭向量库探针客户端失败", exc_info=True)

    try:
        return await asyncio.wait_for(_probe(), timeout=timeout)
    except asyncio.TimeoutError:
        return {
            "ok": False,
            "error": f"Milvus 探测超时（{timeout}s）：{settings.milvus.uri}",
        }
    except Exception as exc:  # noqa: BLE001 —— 探针永不抛，见 docstring 第 1 条
        # ⚠️ 错误信息要脱敏：Milvus 的 URI 可能带 ``user:pass@``，
        # 而探针结果会进 ``/readyz`` 的响应体。
        return {"ok": False, "error": safe_error(exc)}


def describe_vector_store(settings: Settings) -> dict[str, Any]:
    """把向量库的**配置意图**说清楚（纯配置，零 I/O）。

    ⚠️ 与 :func:`probe_collection` 的分工：这个函数回答「我们想连哪」，
    那个回答「连上了没、集合对不对」。启动日志里两个都要有 ——
    只看前者会把「配置写错了」看成「一切正常」。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `dict[str, Any]`: 目标 URI、集合名、维度、索引、度量、top_k。
        ⚠️ URI 已脱敏，不含任何凭据。
    """
    return {
        "uri": redact(settings.milvus.uri),
        "collection": settings.milvus.collection,
        "dimension": settings.milvus.dimension,
        "index_type": settings.milvus.index_type,
        "metric_type": settings.milvus.metric_type,
        "top_k": settings.milvus.top_k,
    }


__all__ = [
    "CONSISTENCY_LEVEL",
    "build_vector_store",
    "describe_collection",
    "describe_vector_store",
    "ensure_collection",
    "probe_collection",
    "request_strong_consistency",
    "verify_collection_shape",
]
