# -*- coding: utf-8 -*-
"""把知识库检索接进智能体中间件链的**桥接层**。

对外只有一个入口::

    from src.knowledge.rag import build_rag_middlewares

    middlewares = await build_rag_middlewares(user_id, settings, kb_manager)

它做三件事，且**只**做这三件事：

    1. 解析出「**这个用户**的知识库」运行时句柄（多租户隔离的第一道闸）；
    2. 把它们包成一个框架的 :class:`~agentscope.middleware.RAGMiddleware`；
    3. 缓存句柄与它的向量模型，避免每个回合重建（重建是**秒级**的重操作）。

═══ 为什么放在 knowledge 层，而不是 server 层 ═══

装配中间件的代码现在住在 ``src/server/agents_factory.py``。**本模块刻意不写
在那里**，理由是它要碰的东西全都属于知识库层，而不属于「把请求接进框架」
这一层：

    · 它调 ``list_knowledge_bases`` / ``get_knowledge`` —— 这两个是
      :class:`~src.knowledge.manager.SingleCollectionKbManager` 的能力，
      而 ``get_knowledge`` 内部会去存储里读凭据、再构造向量模型。
      server 层的工厂不该知道「向量模型是怎么来的」；
    · 它要在**进程级**缓存向量模型与句柄。这类生命周期管理只应有一处，
      而它天然属于「拥有向量库与存储的那一层」（见 :mod:`src.knowledge` 的
      模块文档：Milvus 的惰性连接、``/readyz`` 探针都收在这一层）；
    · 它必须能被**非 HTTP 入口**复用 —— 离线评测、CLI 灌数据脚本、
      定时任务都要「按用户挂上检索」。这些调用方不经过 ``create_app``，
      所以桥接逻辑不能藏在 server 的工厂闭包里。

一句话：server 层负责「把中间件挂上去」，knowledge 层负责「中间件是什么」。
把后者放进前者，就等于让装配层同时背上向量模型的生命周期 ——
而那正是本项目反复避免的一类耦合（对比 ``manager.py``：管理器不该知道
HTTP 的存在，因为它还要被 CLI 与离线评测复用）。

═══ ⚠️ 多租户：只解析**本人**的知识库 ═══

``user_id`` 是**唯一**可信的租户来源，且本模块只做两件事：以它为参数
``list_knowledge_bases(user_id)``，再以它为参数 ``get_knowledge(user_id, kb_id)``。
**绝不**接受调用方传进来的 kb_id 列表 —— 那等于把「查谁的知识库」这个决定
交回给上层，而上一层的入参可能来自请求体。隔离的第二道闸在
``KnowledgeBase.metadata_filter`` 里（见 ``manager.py`` 的「隔离靠什么」），
两道闸同时成立才算数。

═══ ⚠️ 缓存：为什么必须做，以及失效策略 ═══

``manager.get_knowledge`` 每次都会调 ``build_embedding_model``
（``_service/_embedding.py``），而那个构造是**昂贵**的：``dashscope`` 档要
新建一个 HTTP 连接池，``local`` 档要**加载模型权重**（秒级）。
``web_embedding/factory.py:122-128`` 已经把这条警告写死了：「**只在启动期
调用一次**，把结果存起来复用……放进请求路径等于按请求付这两笔开销」。

但 ``extra_agent_middlewares`` 工厂是**每个请求**都会被 await 的
（``agents_factory.py`` 的模块文档：框架每次装配 agent 都调它）。
所以「每回合重建句柄」不是理论风险，而是既成事实 —— 不缓存就等于
每个用户每句话都重建一次连接池。

失效策略（缓存键 = 管理器 + 用户；缓存值 = 记录签名 + 句柄元组）：

    · **按管理器隔离**：缓存挂在 ``WeakKeyDictionary`` 上，键是管理器对象
      本身。用弱引用而不是强引用，是为了不把管理器（进而它持有的存储与
      向量库）钉在进程里 —— 测试里每个用例都会造一个新管理器，强引用会让
      缓存跨用例存活并互相污染。同一个管理器下再按 ``user_id`` 分桶。
    · **按用户隔离**：``user_id`` 是分桶键的一部分。A 的句柄绝不会被
      返回给 B —— 即使 A、B 恰好有同名的知识库。
    · **按记录签名失效**：签名是每个 KB 的 ``(id, name, description)``
      元组。列表变了（新建/删除 KB）或名字/描述被改了，签名就不匹配，
      缓存整体作废并重建。**名字与描述必须进签名**，因为它们在 agentic
      模式下会出现在 ``search_knowledge`` 工具的描述里（``_rag.py`` 的
      ``_build_description``）—— 不失效的话，用户改了知识库名字，模型
      会一直按旧名字决定要不要检索、检索哪个。
    · **向量模型配置不进签名**：``KnowledgeBaseRecord.data.embedding_model_config``
      在记录创建时就固定、终生不变（框架文档写明：改了会让已入库的向量
      全部失效）。所以它不可能在 id 不变的情况下变化，没必要进签名。
    · **按时间兜底（TTL，默认 300 秒）**：签名只看得见**记录**的变化，
      看不见**凭据**的变化 —— 而句柄是从凭据构造出来的。管理员轮换或
      撤销向量凭据时，``KnowledgeBaseRecord`` 一个字段都不变（记录里存的是
      ``credential_id``，不是 key），所以签名照样相等，缓存会**一直命中**，
      那个已经失效的凭据能被用到进程重启。TTL 到点后强制重解析一次，
      真去调 ``get_knowledge``：凭据没了就抛异常、被逐条接住、该库被摘掉。
      取值理由与「为什么必须用单调钟」见 :data:`_HANDLE_TTL_SECONDS`
      与 :func:`_is_fresh`。
    · **部分失败不写缓存**：只要有任何一个 KB 解析失败（例如它的凭据被
      单独删了），本次结果就**不**入缓存，下一回合重新解析 —— 否则一次
      瞬时故障会被缓存成一个「永远少一个知识库」的稳定状态。

进程重启即清空：这张表是**进程内**的，不跨副本、不落盘。本项目当前
``workers=1`` 单进程（见 ``config/base.py`` 的说明），所以是安全的；
扩到多副本时每个副本各缓存一份，行为仍然正确（只是各付一次构造开销），
不需要额外处理。

═══ ⚠️ 并发：同一个用户的两次装配只解析一次 ═══

上面那套缓存只在**解析完成之后**才起作用。而解析中途有一个真实的挂起点
（``await kb_manager.get_knowledge(...)`` —— 它内部还要读凭据、建向量模型），
于是同一个用户的**两个并发请求**会让彼此刻意看不见对方：两边各自查到
「缓存里没有」，各自把每个 KB 解析一遍。每个用户开两个标签页、或者前端同时
发一条消息与一次会话刷新，就能凑出这个交错。

这不是**正确性**问题 —— 两份句柄等价，谁赢都对。它是**可以避免的重复开销**，
而「避免重复开销」正是本模块存在的理由：一次多余的解析 = 一次多余的
连接池构造或模型权重加载（秒级）。

处理方式是一张「正在解析」占位表（:data:`_INFLIGHT`）：第一个协程登记，
后来者等它做完，然后**重查一次缓存**。等待者刻意**不**直接复用解析者的
结果对象（那样能少查一次表），理由是等待者的记录签名可能已经与解析者的
不同 —— 缓存项是调用方拿着**自己的**签名写进去的，复用会让签名与句柄
对不上（之后一次命中就会按错的 id 去取句柄）。重查缓存则复用的全是既有
判据：签名不匹配自然不命中，等待者自己再解析一遍。详见
:data:`_INFLIGHT` 与 :func:`_resolve_handles`。

═══ ⚠️ 可观测性：向量模型落到 Mock 时不能让检索「看起来正常」 ═══

``web_embedding/local.py`` 的模块文档写了一个很难发现的故障：本机没装
``fastembed`` 时，本地向量档构造失败，降级链会**静默地**降到
:class:`~src.web_embedding.mock.MockEmbeddingModel`。Mock 向量之间没有
语义 —— 检索会「成功」（不报错、有结果、有分数），但结果**没有意义**。
从检索结果本身根本看不出来，这也正是它危险的地方。

所以本模块把「实际用了哪些向量模型」这件事**显式**记在返回的中间件上
（:class:`RagBridgeStatus`，挂在 ``middleware.rag_bridge_status``），
并在检测到 Mock 时打一条 warning。注意：句柄的向量模型当前是从**凭据**
构造的（``build_embedding_model`` 走 ``CredentialFactory``），不经过
``src.web_embedding`` 的降级链 —— 但把「是不是 Mock」和「本机有没有
本地档」一起记下来，是唯一能让人事后把「检索结果莫名不对」与
「这台机器少了什么」对上号的线索。它是**只读痕迹**，不影响检索行为。
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from agentscope.middleware import MiddlewareBase, RAGMiddleware

from src.web_embedding import MockEmbeddingModel, unwrap_embedding_model
# ⚠️ ``local_embedding_available`` 没有从 ``src.web_embedding`` 的 ``__init__``
# 导出（那里只导出模型类与工厂），所以从子模块直接取。它只查依赖是否安装，
# 不加载模型，可以放进每请求路径。
from src.web_embedding.local import local_embedding_available

if TYPE_CHECKING:
    from agentscope.app.rag.knowledge_base_manager import (
        KnowledgeBaseManagerBase,
    )
    from agentscope.model import ChatModelBase
    from agentscope.rag import KnowledgeBase

    from src.config.schema import Settings

logger = logging.getLogger(__name__)

#: ``RAGMiddleware.Parameters.top_k`` 的上下界，与框架的声明保持一致
#: （``_rag.py`` 的 ``Parameters.top_k``：``ge=1, le=50``）。
#:
#: ⚠️ 单独抄一份常量看似冗余，但 ``settings.milvus.top_k`` 的上界是 100
#: （见 ``MilvusSettings``），比中间件允许的 50 大。不夹取的话，一个把
#: ``ALIGO__MILVUS__TOP_K=60`` 写下去的人，会在**每次请求**里撞上 pydantic
#: 的校验错误 —— 一次「召回条数配置得大了点」把整条对话链路打挂。
#: 夹取并告警，把影响限制在「召回少了几条」这一档。
_MIN_TOP_K = 1
_MAX_TOP_K = 50


@dataclass(frozen=True)
class RagBridgeStatus:
    """一次桥接的**只读痕迹**，挂在返回的 :class:`RAGMiddleware` 上。

    ⚠️ 它存在的理由只有一个：让「检索结果看起来正常、其实没有意义」
    这种故障**有据可查**。详见模块文档最后一节。它不参与任何检索逻辑，
    改它不会改变行为。

    Attributes:
        user_id: 本次桥接为哪个用户解析。
        knowledge_base_ids: 解析成功的知识库 id 元组（顺序与句柄一致）。
        embedding_models: 每个句柄实际使用的向量模型**类名**，顺序同上。
            ⚠️ 只记类名，不记任何密钥或端点 —— 痕迹要能进日志。
        degraded: 是否有句柄用了确定性假向量（Mock）。为真意味着
            「检索会成功但结果没有语义」。
        local_embedding_available: 本机本地向量档（``fastembed``）是否可用。
            它解释的是**降级链**那一侧的隐患（``web_embedding/local.py``）；
            与 ``degraded`` 并列记录，便于事后归因。
        detail: 给人看的一句话说明。
    """

    user_id: str
    knowledge_base_ids: tuple[str, ...]
    embedding_models: tuple[str, ...]
    degraded: bool
    local_embedding_available: bool
    detail: str


@dataclass(frozen=True)
class _CachedHandles:
    """一个用户的句柄缓存项。

    Attributes:
        signature: 每个 KB 的 ``(id, name, description)`` 元组。列表或名字/
            描述一变，它就不相等，缓存整体作废（见模块文档的失效策略）。
        handles: 与 ``signature`` 一一对应的运行时句柄元组。
        stored_at: 写入时刻（``time.monotonic()`` 秒）。用于 TTL 失效 ——
            见 :func:`_is_fresh` 与模块文档的「按时间兜底」。
    """

    signature: tuple[tuple[str, str, str], ...]
    handles: tuple["KnowledgeBase", ...]
    stored_at: float


#: 进程级句柄缓存：``管理器 -> {user_id: 缓存项}``。
#:
#: ⚠️ 用 ``WeakKeyDictionary`` 而不是普通 dict：键是管理器**对象**，
#: 弱引用保证它被回收时缓存项一起消失。用强引用的话，测试里每个用例
#: 造的新管理器都会在进程里留一份 —— 缓存跨用例存活，症状是
#: 「单独跑绿、一起跑红」，而且很难看出是谁污染的谁。
#:
#: ⚠️ 不加锁：本项目 ``workers=1``、单事件循环，且本模块的字典读写在
#: 协程之间没有 ``await`` 夹在中间，因此是原子的。真引入多线程时这里
#: 需要一把锁 —— 那与 GIL 无关，与「有没有第二个线程」有关。
#:
#: ⚠️ 原子的是**单个**字典操作，不是「查了再写」这一串 —— 中间夹着
#: ``await`` 的判定（「没命中 ⇒ 去解析 ⇒ 写回来」）本来就会被并发穿过。
#: 它不靠锁解决，靠的是 :data:`_INFLIGHT` 那张单飞占位表：同一个用户
#: 同时只有一个协程真正在解析。缓存写在协程被唤醒前就已经完成（解析返回
#: 到写缓存之间没有 ``await``），所以等待者醒来重查必然能看见。
_HANDLE_CACHE: "weakref.WeakKeyDictionary[Any, dict[str, _CachedHandles]]" = (
    weakref.WeakKeyDictionary()
)

#: 「正在解析」的占位表：``管理器 -> {user_id: 完成信号}``。
#:
#: ⚠️ 它防的是**并发重复解析**，不是「读到写了一半的缓存」：解析中途有真实
#: 的挂起点，同一个用户的两个并发请求会双双判定「缓存里没有」、各自把每个
#: KB 解析一遍（各建一次连接池 / 各加载一次权重）。详细论证见模块文档的
#: 「并发」一节；等待者**不**复用解析者的结果、只等信号再重查缓存的理由
#: 也在那里。
#:
#: ⚠️ 值为什么是一个 ``Future`` 而不是 ``asyncio.Lock``：锁的语义是「排队，
#: 然后自己再判一次」，而这里希望的是「等它做完，然后**替我做完了**」——
#: 等待者醒来重查缓存即可命中，不必再解析。用 ``Lock`` 也能写对（多查一次
#: 缓存而已），但每次并发都要多一次「明知无用」的解析路径判定；``Future``
#: 让「谁在解析」这件事看得见，也让用例能把等待者精确地卡在那个点上。
#:
#: ⚠️ 键同样是**管理器对象**、同样用弱引用，理由与 :data:`_HANDLE_CACHE`
#: 逐字相同（不把管理器钉在进程里，测试之间不互相污染）。管理器不可弱引用
#: 时单飞闸门静默失效（退回「并发各解析一次」），由 :func:`_inflight_for`
#: 负责 —— 闸门是优化，不是正确性的一部分，不该因为它挂掉。
#:
#: ⚠️ 等待者手里攥着 ``Future`` **对象本身**（不是靠这张表找回来），所以
#: :func:`clear_handle_cache` 把它从表里摘掉也不会让等待者永远等下去 ——
#: 解析者的 ``finally`` 照样会兑现那个信号。
_INFLIGHT: "weakref.WeakKeyDictionary[Any, dict[str, asyncio.Future[None]]]" = (
    weakref.WeakKeyDictionary()
)

#: 句柄缓存的**兜底存活时间**（秒）。
#:
#: ⚠️ 为什么除了签名之外还需要一个 TTL —— 签名只看得到**记录**的变化，
#: 看不到**凭据**的变化，而句柄正是从凭据构造出来的（``get_knowledge`` →
#: ``build_embedding_model`` → ``CredentialFactory``）：
#:
#:     · 管理员**轮换**了向量凭据的 key：``KnowledgeBaseRecord`` 一个字段
#:       都没变（记录里存的是 ``credential_id``，不是 key 本身），签名不变；
#:     · 管理员**撤销/删除**了向量凭据：记录同样不变。
#:
#: 而 TTL 到点之后走的正是「重新解析」那条路 —— 它会真的去 ``get_knowledge``，
#: 于是凭据没了就抛异常，被 :func:`_resolve_handles` 逐条接住
#: （``complete=False`` ⇒ **不写缓存** ⇒ 下一回合继续重试）。
#: 换句话说，TTL 是唯一能让「凭据被撤销」**迟早**变成「那个库被摘掉」的机制；
#: 没有它，一个已经失效的凭据会跟着缓存一直用到进程重启。
#:
#: ⚠️ 取值是两头的折中：太小就退化成「每 N 秒重建一次连接池/加载一次权重」，
#: 把本模块存在的理由（见模块文档）抵消掉；太大则凭据撤销后要等很久才生效。
#: 300 秒 = 每 5 分钟重解析一次，相对「每个请求解析一次」仍是两个数量级的
#: 削减，而撤销的最坏生效延迟是一顿午饭之内 —— 对一个知识库检索来说足够。
#:
#: ⚠️ 这个 TTL 是**从写入时刻算起**的绝对窗口，不是「空闲窗口」：命中缓存
#: **不得**刷新 ``stored_at``。刷了的话，每个请求都会把截止时间往后推，
#: TTL 就只在「用户 300 秒没说话」时才到点 —— 而 ``agents_factory.py``
#: 每请求都调用本模块，持续对话的用户于是永远不重解析，上面那条
#: 「撤销迟早生效」的保证当场落空。执行点在
#: :func:`build_rag_middlewares` 的 ``if complete and not cached_hit:``。
_HANDLE_TTL_SECONDS = 300.0


def _is_fresh(entry: _CachedHandles, *, now: float) -> bool:
    """缓存项是否还在 TTL 内。

    ⚠️ 用 ``time.monotonic()`` 而不是 ``time.time()``：后者会被系统时钟
    校准（NTP 回拨、手动改时间）**向后跳**，那会让缓存项在 TTL 之内就
    被判成过期（只是白重建一次，可接受），或者更糟 —— 向前跳之后
    **永远**新鲜（凭据撤销再也生效不了）。单调钟只会向前走。

    Args:
        entry (`_CachedHandles`): 缓存项。
        now (`float`): 当前单调时刻（由调用方取一次，避免循环里反复取）。

    Returns:
        `bool`: 未过期返回 True。
    """
    return (now - entry.stored_at) < _HANDLE_TTL_SECONDS


def _signature(records: "list[Any]") -> tuple[tuple[str, str, str], ...]:
    """把知识库记录列表压成一个可比较的签名。

    ⚠️ 顺序在这里**被规范化**（按 id 排序）：存储返回记录的顺序不保证
    稳定，而签名只用于「变没变」的判断 —— 顺序抖动不该被误判成一次变更
    并触发重建。

    Args:
        records: :class:`~agentscope.app.storage.KnowledgeBaseRecord` 列表。

    Returns:
        `tuple`: ``((id, name, description), ...)``，按 id 升序。
    """
    return tuple(
        sorted(
            (
                record.id,
                record.data.name,
                record.data.description,
            )
            for record in records
        )
    )


def _hit_pairs(
    cached: "_CachedHandles | None",
    records: "list[Any]",
    signature: tuple[tuple[str, str, str], ...],
) -> "list[tuple[Any, KnowledgeBase]] | None":
    """命中缓存时返回「记录与句柄」的配对；未命中返回 ``None``。

    ⚠️ 返回 ``None`` 而不是空列表来表「未命中」：空列表在类型上是个合法的
    配对结果，用它当哨兵迟早会有人写 ``if not hit`` 而把「缓存里一个库都没有」
    与「缓存没命中」混为一谈。这个函数只回答「能不能用」，不回答「有几个」。

    ⚠️ 两个判据缺一不可：签名一致（记录没变）**且** 未过期（TTL）。
    只看签名的话，一条被撤销的凭据会跟着缓存一直用到进程重启 ——
    记录里存的是 ``credential_id``，凭据本身被删/被换 key 时记录一个字段都不变。
    见 :data:`_HANDLE_TTL_SECONDS` 与 :func:`_is_fresh`。

    ⚠️ 配对是**按 id 重建**的（缓存里存的是 id 元组，不是记录对象）：
    调用方手里这份 ``records`` 才是最新的（它刚查过存储），句柄才是从缓存里
    原样取回来的。两边都取新的那一份，才不会出现「记录是旧的、句柄是新的」
    这种半新半旧的状态。``zip`` 在这里是安全的：命中判据要求签名**逐字相等**，
    两者长度必然一致。

    Args:
        cached (`_CachedHandles | None`): 缓存里取出来的那一项。
        records: 本次查到的记录列表（顺序不保证，故先按 id 建索引）。
        signature: 本次的记录签名。

    Returns:
        `list[tuple[Any, KnowledgeBase]] | None`: 可复用时是配对列表，
        否则 ``None``。
    """
    if cached is None or cached.signature != signature:
        return None
    if not _is_fresh(cached, now=time.monotonic()):
        return None
    by_id = {record.id: record for record in records}
    return [
        (by_id[kb_id], handle)
        for (kb_id, _name, _desc), handle in zip(
            cached.signature,
            cached.handles,
        )
    ]


def _resolve_top_k(settings: "Settings", override: int | None) -> int:
    """算出实际使用的 ``top_k``，并夹进中间件允许的区间。

    Args:
        settings (`Settings`): 配置，默认值取 ``settings.milvus.top_k``。
        override (`int | None`): 调用方显式覆盖值；``None`` 时用配置值。

    Returns:
        `int`: 落在 ``[1, 50]`` 内的召回条数。

    Raises:
        ValueError: 显式传入的 ``override`` 小于 1 时。
    """
    if override is not None and override < _MIN_TOP_K:
        # 显式传参是调用方写错，不是「大了点」那种可以容忍的溢出 ——
        # 0 条召回不是检索，是静默失效，必须在装配期就报出来。
        raise ValueError(
            f"build_rag_middlewares 的 top_k 必须 >= {_MIN_TOP_K}，"
            f"收到 {override}。",
        )

    value = settings.milvus.top_k if override is None else override
    if value > _MAX_TOP_K:
        # ⚠️ 夹取而不是抛错：能走到这里的只有**配置值**（或显式传入的大值），
        # 而配置溢出只该削弱召回，不该打挂对话 —— 见上面 _MAX_TOP_K 的说明。
        logger.warning(
            "top_k=%d 超过中间件上限 %d，已夹取。"
            "把 ALIGO__MILVUS__TOP_K 改到这个区间内可消除此告警。",
            value,
            _MAX_TOP_K,
        )
        value = _MAX_TOP_K
    return value


def _build_status(
    user_id: str,
    records: "list[Any]",
    handles: tuple["KnowledgeBase", ...],
) -> RagBridgeStatus:
    """构造挂在中间件上的只读痕迹。

    ⚠️ ``local_embedding_available()`` 只探测**依赖是否安装**（查
    ``importlib.util.find_spec``），不加载模型、不发请求，所以可以放进
    这条每请求都会走的路径上（见 ``web_embedding/local.py`` 的同名函数）。

    Args:
        user_id (`str`): 当前用户。
        records: 解析成功的记录（与 ``handles`` 一一对应）。
        handles: 运行时句柄。

    Returns:
        `RagBridgeStatus`: 本次桥接的痕迹。
    """
    # ⚠️ 一律经过 ``unwrap_embedding_model``：句柄上的向量模型被
    # ``BoundedEmbeddingModel`` 包了一层（加调用截止时间，见
    # :mod:`src.web_embedding.bounded`），直接取 ``type(...).__name__``
    # 会得到包装类的名字、``isinstance(..., MockEmbeddingModel)`` 会**恒为假**。
    # 后者尤其危险：它会让「正在用没有语义的假向量」这条告警**静默消失**，
    # 而这条告警是发现该故障的唯一线索。
    implemented = [unwrap_embedding_model(kb.embedding_model) for kb in handles]
    model_names = tuple(type(model).__name__ for model in implemented)
    mock_kbs = [
        record.data.name
        for record, model in zip(records, implemented)
        if isinstance(model, MockEmbeddingModel)
    ]
    local_ok = local_embedding_available()

    if mock_kbs:
        detail = (
            f"知识库 {mock_kbs} 使用了确定性假向量（Mock）—— "
            f"检索会返回结果，但结果没有语义。"
        )
    elif not local_ok:
        detail = (
            "本次未使用假向量；但本机未安装 fastembed，本地向量档不可用"
            "（降级链可能悄悄落到 Mock，见 web_embedding/local.py）。"
        )
    else:
        detail = "本次解析到的向量模型均为可用实现。"

    return RagBridgeStatus(
        user_id=user_id,
        knowledge_base_ids=tuple(record.id for record in records),
        embedding_models=model_names,
        degraded=bool(mock_kbs),
        local_embedding_available=local_ok,
        detail=detail,
    )


async def build_rag_middlewares(
    user_id: str,
    settings: "Settings",
    kb_manager: "KnowledgeBaseManagerBase",
    *,
    rerank_model: "ChatModelBase | None" = None,
    mode: Literal["static", "agentic"] = "agentic",
    top_k: int | None = None,
    score_threshold: float | None = None,
) -> list[MiddlewareBase]:
    """为某个用户装配 RAG 中间件。

    ⚠️ **没有知识库时返回空列表**，而不是一个「检索永远为空」的中间件。
    差别不是性能，是**语义**：agentic 模式下挂着一个空的搜索工具，
    模型会看到「已装备 0 个知识库」的描述，却仍然可能去调它，
    然后把「没查出来」当成「知识库里没有」—— 而真实情况是「这个用户
    根本没有知识库，我们压根没接检索」。这两种情况在用户看来的处置
    完全不同（一个要补充知识库，一个只需换个说法再问），
    用一个空中间件把它们抹平，等于制造一个**看起来查过、其实没查**的假象。

    Args:
        user_id (`str`): 当前用户标识。⚠️ **唯一可信**的租户来源，见模块
            文档「多租户」一节。
        settings (`Settings`): 配置。默认值取自它：``top_k`` 来自
            ``settings.milvus.top_k``。``mode`` 目前**没有**对应的配置段
            （本轮刻意不加，避免动 ``extra="forbid"`` 的 schema），
            所以走函数入参。
        kb_manager (`KnowledgeBaseManagerBase`): 知识库管理器。只要求鸭子类型
            （``list_knowledge_bases`` 与 ``get_knowledge`` 两个方法），
            便于测试替身。
        rerank_model (`ChatModelBase | None`, optional): 可选的重排模型，
            原样透传给 :class:`RAGMiddleware`。⚠️ 重排是**尽力而为**的：
            它失败时中间件会退回向量序（``agentscope/middleware/_rag.py:439-445``），
            不会让检索整体失败。
        mode (`Literal["static", "agentic"]`, optional): 检索模式，默认
            ``"agentic"``（把 ``search_knowledge`` 工具交给模型，由它决定
            何时检索）。``"static"`` 会在每个回复的第一个推理步自动检索并
            注入提示（``_rag.py`` 的 ``on_reasoning``）。
        top_k (`int | None`, optional): 召回条数；``None`` 时取
            ``settings.milvus.top_k``。
        score_threshold (`float | None`, optional): 相似度阈值，透传给框架。

    Returns:
        `list[MiddlewareBase]`: 有知识库时是 ``[RAGMiddleware(...)]``
        （其上带 ``rag_bridge_status`` 痕迹）；没有时是 ``[]``。

    Raises:
        ValueError: 显式传入的 ``top_k`` 小于 1 时。⚠️ **除此之外本函数
            不抛异常** —— 知识库侧的任何故障都被吞掉并降级为空列表，
            理由见 :func:`_list_records` 与 :func:`_resolve_handles`。
    """
    records = await _list_records(user_id, kb_manager)
    if not records:
        return []

    signature = _signature(records)
    pairs, complete, cached_hit = await _resolve_handles(
        user_id,
        kb_manager,
        records,
        signature,
    )
    if not pairs:
        # 全都没解析出来（例如所有凭据都被删了）—— 与「没有知识库」同样
        # 处理：不挂中间件。这里刻意**不**把「全坏了」当成「检索为空」，
        # 因为对调用方而言两者都意味着「本次没有可用的知识库」；
        # 具体原因已经逐条记在日志里了。
        return []

    resolved_records = [record for record, _handle in pairs]
    handles = tuple(handle for _record, handle in pairs)

    parameters = RAGMiddleware.Parameters(
        mode=mode,
        top_k=_resolve_top_k(settings, top_k),
        score_threshold=score_threshold,
    )
    middleware = RAGMiddleware(
        knowledge_bases=list(handles),
        parameters=parameters,
        rerank_model=rerank_model,
    )

    # 挂上只读痕迹。⚠️ 这里用 setattr 而不是构造参数：RAGMiddleware 是
    # 框架的类，不该为我们的可观测需求改它的签名 —— 痕迹是本模块的产物，
    # 就由本模块贴上（MiddlewareBase 没有 __slots__，可以贴）。
    status = _build_status(user_id, resolved_records, handles)
    middleware.rag_bridge_status = status  # type: ignore[attr-defined]
    if status.degraded:
        logger.warning(
            "用户 %s 的 RAG 中间件已装配，但 %s",
            user_id,
            status.detail,
        )

    # ⚠️ 只有「全部解析成功」**且**「这一轮真的去解析了」才写缓存。
    # 部分失败时留空，下一回合重试 —— 见模块文档「部分失败不写缓存」。
    #
    # ⚠️ ``not cached_hit`` 这一半是**必须**的，漏掉会让 TTL 变成滑动窗口：
    # 命中缓存时句柄没变，但 ``time.monotonic()`` 是新的，于是每次调用都
    # 把 ``stored_at`` 往后推 —— TTL 退化成「空闲 300 秒才重解析」。而
    # ``agents_factory.py`` 是**每个请求**都调本函数的，所以一个持续对话的
    # 用户永远够不到那个空闲窗口：他被撤销的凭据会一直用到进程重启，
    # 正是 :data:`_HANDLE_TTL_SECONDS` 那整段注释要防的事。
    if complete and not cached_hit:
        _store_cache(
            kb_manager,
            user_id,
            _CachedHandles(signature, handles, stored_at=time.monotonic()),
        )

    return [middleware]


# ---------------------------------------------------------------------
# 内部步骤 —— 拆开是为了让「吞异常的位置」与「缓存命中的位置」一眼可见。
# ---------------------------------------------------------------------
async def _list_records(
    user_id: str,
    kb_manager: "KnowledgeBaseManagerBase",
) -> "list[Any]":
    """列出该用户的知识库记录；**失败时返回空列表**。

    ⚠️ 这里吞掉异常是**刻意的**，而且这是本模块最重要的一条决策：
    ``src/knowledge/__init__.py`` 的模块文档写明「Milvus 不可用不能拖垮
    服务 —— 对话、订单、审批、鉴权全都不经过向量库」。若这里把存储/向量库
    的异常抛出去，一次向量库抖动就会让**所有人**的对话 500 ——
    而他们说的很可能只是一句「你好」。检索是**增强**，不是主链路。

    ⚠️ 吞掉的是异常，不是信息：用 ``logger.exception`` 把栈完整记下来，
    否则「检索突然没了」会变成一条没有任何线索的现象。

    Args:
        user_id (`str`): 当前用户。
        kb_manager (`KnowledgeBaseManagerBase`): 知识库管理器。

    Returns:
        `list`: 记录列表；出错时为空列表。
    """
    try:
        return list(await kb_manager.list_knowledge_bases(user_id))
    except Exception:  # noqa: BLE001 —— 见 docstring：检索不能拖垮对话
        logger.exception(
            "列出用户 %s 的知识库失败；本轮不挂 RAG 中间件，对话照常。",
            user_id,
        )
        return []


async def _resolve_handles(
    user_id: str,
    kb_manager: "KnowledgeBaseManagerBase",
    records: "list[Any]",
    signature: tuple[tuple[str, str, str], ...],
) -> "tuple[list[tuple[Any, KnowledgeBase]], bool]":
    """解析句柄，命中缓存则直接复用；返回「记录与句柄」的配对。

    ⚠️ 命中判据用**签名**而不是「id 集合相等」：名字/描述变了也必须
    重建，理由见模块文档的失效策略。

    ⚠️ 逐条解析、逐条降级：某一个 KB 解析失败（典型的：它的凭据被单独
    删了 —— ``manager.get_knowledge`` 会抛 ``KnowledgeBaseNotFoundError``）
    只丢它自己，不影响其余 KB。整批放弃的代价是「一个坏库让所有库都不
    可用」，没有任何理由这么做。

    ⚠️ 返回的是 ``(记录, 句柄)`` **配对**，而不是裸句柄：句柄不携带记录
    id，只有配对才能让上层的痕迹把「哪个库的名字」与「哪个向量模型」
    对得上（见 :func:`_build_status`）。

    ⚠️ 并发：同一个 ``(管理器, 用户)`` 上**同时只有一个**协程真正在解析
    （:data:`_INFLIGHT` 那张占位表）。后来者等它做完，再走一遍上面那段
    命中判定 —— 于是绝大多数情况下它拿到的就是刚写好的缓存。

    Args:
        user_id (`str`): 当前用户。
        kb_manager (`KnowledgeBaseManagerBase`): 知识库管理器。
        records: 该用户的记录列表。
        signature: 当前记录签名。

    Returns:
        `tuple[list[tuple[Any, KnowledgeBase]], bool, bool]`: 配对列表、
        「是否**全部**解析成功」、以及「本轮是**命中缓存**还是真的重新解析了」。

        ⚠️ 第三个元素不是锦上添花：调用方要靠它区分「解析完刚拿到的句柄」
        与「从缓存里原样取回来的句柄」，只有前者才该刷新 ``stored_at``。
        没有它就别无办法分辨（两种情况下 ``pairs`` 长得一模一样），
        TTL 会退化成滑动窗口 —— 见调用处 ``if complete and not cached_hit``。
    """
    cache = _cache_for(kb_manager)

    # ⚠️ 命中：直接复用句柄，**不再**调 get_knowledge —— 这正是避免每回合
    # 重建向量模型的地方。第三个返回值 ``True`` 是「命中」的**唯一**证据：
    # 这里拿到的 ``handle`` 是缓存里的旧对象，它的构造凭据可能已经被撤销了 ——
    # 调用方必须**不**因此刷新 ``stored_at``，否则 TTL 永远够不到点
    # （见 :data:`_HANDLE_TTL_SECONDS`）。``complete=True`` 与命中无关，
    # 它是说「缓存里这份是齐的」。
    hit = _hit_pairs(cache.get(user_id), records, signature)
    if hit is not None:
        return hit, True, True

    # ── 单飞闸门：有人正在为这个用户解析就等它，别重复付那笔开销 ──
    inflight = _inflight_for(kb_manager)
    pending = inflight.get(user_id)
    if pending is not None:
        # ⚠️ ``shield`` 是必须的：等待者被取消（客户端断开、请求超时）时，
        # ``await`` 一个裸 Future 会把**它**也一起取消掉 —— 那样剩下的等待者
        # 全部落空，而解析者本人还在照常干活，白等一场。shield 让取消只影响
        # 等待者自己。
        #
        # ⚠️ 这里刻意不 ``try``：等待者被取消就该往上抛（取消是调用方的意图），
        # 被盾牌挡住的解析者不受影响。
        await asyncio.shield(pending)
        # 解析者做完了 —— 重查缓存。⚠️ 走的是**既有判据**（含签名比对），
        # 所以「解析者在解析途中记录变了」这种情况会自然落空，等待者自己
        # 再解析一遍，而不是拿一份签不上名的结果去写缓存。
        hit = _hit_pairs(cache.get(user_id), records, signature)
        if hit is not None:
            return hit, True, True

    # 走到这里说明：缓存没命中，且（现在）没有别人在解析 —— 由自己来解析。
    # ⚠️ 登记必须发生在**第一个 await 之前**：晚一步，与自己在同一轮事件循环
    # 里启动的兄弟协程就会和它一起成为解析者，闸门形同虚设。
    future: "asyncio.Future[None]" = asyncio.get_running_loop().create_future()
    inflight[user_id] = future

    by_id = {record.id: record for record in records}
    pairs: list[tuple[Any, "KnowledgeBase"]] = []
    complete = True
    try:
        # ⚠️ 按签名里的 id 顺序解析（与 ``_signature`` 的排序一致），
        # 这样句柄顺序稳定，缓存与痕迹里「id ↔ 句柄」才对得上。
        for kb_id, _name, _desc in signature:
            record = by_id[kb_id]
            try:
                handle = await kb_manager.get_knowledge(user_id, record.id)
            except Exception:  # noqa: BLE001 —— 单条失败只丢这一条
                complete = False
                logger.exception(
                    "用户 %s 的知识库 %s（%s）解析失败；跳过它，本轮不缓存。",
                    user_id,
                    record.id,
                    record.data.name,
                )
                continue
            pairs.append((record, handle))
    finally:
        # ⚠️ 无论成功、失败还是被取消，都必须兑现信号并摘掉占位 —— 漏掉任何
        # 一条路径，等在那儿的协程就会永远卡在 ``await`` 上。症状是「偶尔有
        # 请求一直不返回」，而且它只在那一次并发里出现，几乎无法归因。
        #
        # ⚠️ 用 ``set_result(None)`` 而不是 ``set_exception``：等待者只把完成
        # 本身当作信号（它随后重查缓存，真实结果以缓存为准）。设异常的话，
        # 一个「没人 await 它」的 future 会在回收时打出
        # 「Future exception was never retrieved」—— 噪音，且误导。
        if not future.done():
            future.set_result(None)
        # ⚠️ 按**身份**摘，而不是无脑 ``pop``：``clear_handle_cache`` 可能
        # 在这中间把整张表清掉并让另一个协程登记了新的占位，无脑 pop 会把
        # 别人的占位摘走，那之后第三个协程就会以为没人解析而重复一次。
        if inflight.get(user_id) is future:
            inflight.pop(user_id, None)

    # 第三个返回值恒为 ``False``：走到这里就意味着**没有**命中缓存，
    # 上面每个 ``handle`` 都是本次现解析出来的，``stored_at`` 该刷新。
    return pairs, complete, False


def _cache_for(kb_manager: Any) -> dict[str, _CachedHandles]:
    """取出某个管理器对应的用户桶（不存在则建）。

    ⚠️ 管理器若不可弱引用（例如用了 ``__slots__`` 且没声明
    ``__weakref__``），退化为「每次都重建」而不是报错 —— 缓存是优化，
    不是正确性的一部分，不该因为它挂掉。

    Args:
        kb_manager (`Any`): 知识库管理器。

    Returns:
        `dict[str, _CachedHandles]`: ``user_id -> 缓存项`` 的可变字典。
    """
    try:
        return _HANDLE_CACHE.setdefault(kb_manager, {})
    except TypeError:
        return {}


def _inflight_for(kb_manager: Any) -> dict[str, "asyncio.Future[None]"]:
    """取出某个管理器对应的「正在解析」占位桶（不存在则建）。

    ⚠️ 与 :func:`_cache_for` 逐字同款的降级：管理器不可弱引用时返回一个
    **临时空字典**。登记进去的 ``Future`` 别人看不见，于是闸门静默失效、
    退回「并发各解析一次」—— 那是这个模块本来的行为，不是错误。
    反过来，若这里抛异常，一次「管理器恰好不可弱引用」会让**装配中间件**
    整个失败，进而让对话链路挂掉 —— 用一个优化去换主链路，不划算。

    Args:
        kb_manager (`Any`): 知识库管理器。

    Returns:
        `dict[str, asyncio.Future[None]]`: ``user_id -> 完成信号`` 的可变字典。
    """
    try:
        return _INFLIGHT.setdefault(kb_manager, {})
    except TypeError:
        return {}


def _store_cache(
    kb_manager: Any,
    user_id: str,
    entry: _CachedHandles,
) -> None:
    """写入缓存（不可弱引用时静默跳过）。"""
    try:
        _HANDLE_CACHE.setdefault(kb_manager, {})[user_id] = entry
    except TypeError:
        pass


def clear_handle_cache() -> None:
    """清空进程级句柄缓存与「正在解析」占位表。

    ⚠️ 生产路径**不需要**调它 —— 失效是自动的（记录签名变化）。它存在是
    为了**测试**：用例之间共享同一个管理器实例时（例如会话级夹具），
    可以显式清掉缓存，避免上一个用例留下的句柄影响下一个。

    ⚠️ 连 :data:`_INFLIGHT` 一起清是**安全**的：等待者手里攥着 ``Future``
    对象本身，解析者的 ``finally`` 照样会兑现它，不会被清表卡住。
    """
    _HANDLE_CACHE.clear()
    _INFLIGHT.clear()


__all__ = [
    "RagBridgeStatus",
    "build_rag_middlewares",
    "clear_handle_cache",
]
