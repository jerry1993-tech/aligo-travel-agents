# -*- coding: utf-8 -*-
"""配置的**结构与校验规则**（本项目所有可配置项的唯一定义处）。

文件职责：
    把「配置长什么样」这件事集中在一个地方声明 —— 有哪些段、每段有哪些键、
    每个键的类型与取值范围。`config/*.yaml` 提供**值**，本模块提供**形状**，
    二者由 :mod:`src.config.loader` 撮合。

上下游依赖：
    - 上游：无。本模块只依赖 pydantic，不读文件、不读环境变量。
    - 下游：
        · ``src/config/loader.py``  用 :class:`Settings` 校验合并后的配置字典；
        · ``src/server/app.py`` 等消费方通过 ``Settings`` 的字段访问配置
          （而不是散落地读 ``os.environ``）。

设计要点（三条，都是刻意的）：
    1. **每一层都是 ``extra="forbid"``**。拼错的键会立刻报错，而不是被静默忽略后
       取默认值。后者是本项目最忌讳的一类偏差：配置看起来生效了，实际没有。
       ⚠️ 代价是「新增配置项必须两步走」：先在这里加字段，再去 ``config/base.yaml``
       写默认值。漏了第一步，启动时会以 ``Extra inputs are not permitted`` 崩溃 ——
       这是设计行为，不是 bug。
    2. **本模块不认识 ``ALIGO__`` 前缀，也不认识 YAML**。环境变量到嵌套字典的
       降维与合并全在 loader 里完成；本模块只做纯数据校验。这样 schema 可以被
       单测直接构造，无需任何外部环境。
    3. **跨字段的约束写成本模块的 validator，而不是散在业务代码里**。
       例如「开了 JWT 就必须给 secret」——放在这里，启动即失败；
       放在业务代码里，则要等到第一次请求才暴露。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ------------------------------------------------------------------------------
# 公共基类：把 extra="forbid" 与「禁止未知键」的策略收敛到一处。
# ------------------------------------------------------------------------------
# 为什么继承一个自定义基类而不是每个模型各写一遍 model_config：
# 这是**安全相关**的默认值，不是风格偏好。继承保证了「新增一个配置段时
# 不可能忘记打开严格校验」，而逐个书写迟早会漏掉一个。
class _StrictModel(BaseModel):
    """所有配置模型的基类：禁止未知键。

    ``extra="forbid"`` 的语义是「字典里出现了模型没声明的键就报错」。
    pydantic 为此给出的错误类型是 ``extra_forbidden``，错误文案为
    ``Extra inputs are not permitted`` —— 这两处字样会被 loader 原样带进
    抛出的 ``ValueError`` 里，是排障时的主要线索（例如 .env 里误写了
    ``ALIGO__IMAGE_REGISTRY`` 就会看到 ``image_registry  Extra inputs are not permitted``）。
    """

    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------------------
# 一、应用本体
# ------------------------------------------------------------------------------
class AppSettings(_StrictModel):
    """应用运行环境、日志与监听参数。

    Attributes:
        env: 运行环境，决定加载 ``config/`` 下哪一份 YAML。
        log_level: 日志级别。
        port: 服务监听端口（容器内固定 8000）。
        workers: uvicorn 进程数。默认 1，理由见 ``config/base.yaml`` 的注释
            （调度器重复触发 + 连接池按进程倍增）。
        download_secret: 工作区文件下载 token 的签名密钥；**空串表示交给框架
            每进程随机生成**（单实例部署下的合理默认，多副本时必须显式设置）。
    """

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    port: int = Field(default=8000, ge=1, le=65535)
    # ge=1：0 个 worker 不是一个合法服务；上限 32 是保守的经验值，
    # 真要开更多，请先按 config/base.yaml 的说明把调度器拆成独立进程。
    workers: int = Field(default=1, ge=1, le=32)
    # 对应 create_app 的 download_secret 参数。框架的默认行为是「每个进程
    # 随机生成一个」，其文档明确警告：**放在负载均衡后面必须显式设置**，
    # 否则 A 副本签发的 token 到了 B 副本会被判为无效 —— 症状是
    # 「下载文件时随机失败」，重试一次可能就好了，极难归因。
    # 本项目默认 WORKERS=1、单实例（见上面的 workers 注释），
    # 因此留空是**正确**的而不是偷懒；真正扩到多副本时再填。
    download_secret: str = ""


# ------------------------------------------------------------------------------
# 二、鉴权
# ------------------------------------------------------------------------------
class AuthSettings(_StrictModel):
    """身份解析策略。

    本项目采用「``X-User-ID`` 为主 + 可选 JWT」的双通道：
    JWT 校验通过后，其中的 ``sub`` 会被**注入成 ``X-User-ID``**，
    从而复用 AgentScope 框架既有的鉴权与多租户隔离链路，而不是另起一套。

    Attributes:
        require_user_header: 是否强制要求带 ``X-User-ID``。
        jwt_enabled: 是否启用 ``Authorization: Bearer`` 解析。
        jwt_secret: HMAC 密钥；``jwt_enabled`` 为真时不得为空。
        jwt_algorithm: 签名算法，默认 HS256（对称）。
        jwt_audience: 期望的 ``aud``；**空串表示不校验**。
        jwt_issuer: 期望的 ``iss``；**空串表示不校验**。
    """

    require_user_header: bool = True
    jwt_enabled: bool = False
    jwt_secret: str = ""
    jwt_algorithm: str = "HS256"
    jwt_audience: str = ""
    jwt_issuer: str = ""

    @model_validator(mode="after")
    def _jwt_secret_required_when_enabled(self) -> "AuthSettings":
        """开启 JWT 却没给密钥时，在**启动期**就把配置矛盾暴露出来。

        为什么不用默认值兜底：一个空的 HMAC 密钥意味着任何人都能伪造合法 token
        —— 这是安全漏洞，不是配置疏漏，必须硬失败。

        Returns:
            `AuthSettings`: 校验通过的自身。

        Raises:
            `ValueError`: ``jwt_enabled`` 为真而 ``jwt_secret`` 为空时。
        """
        if self.jwt_enabled and not self.jwt_secret.strip():
            raise ValueError(
                "auth.jwt_enabled 为 true 时必须提供非空的 auth.jwt_secret；"
                "空密钥会让任何调用者都能伪造 token。"
            )
        return self


# ------------------------------------------------------------------------------
# 二·补、限流
# ------------------------------------------------------------------------------
class RateLimitSettings(_StrictModel):
    """按身份限流的参数（实现见 ``src/server/middleware/rate_limit.py``）。

    **计数器在进程内存里**，这是本项目 ``workers=1`` 单进程拓扑下的正确选择：
    零额外延迟、零新依赖。代价必须写清楚 —— 一旦扩到多副本，每个副本各限一份，
    实际放行的总量是配置值的 N 倍。扩副本前请把计数器换成 Redis 共享实现
    （复用 ``redis`` 段已有的连接参数），否则限流会**静默失效**：
    它不会报错，只是不再拦住任何东西。

    Attributes:
        enabled: 总开关。关闭时中间件直接透传，行为与没有限流完全一致。
        requests_per_window: 每个窗口允许的请求数，同时是令牌桶的**容量**
            （即允许的瞬时突发上限）。
        window_seconds: 窗口长度（秒）。令牌以
            ``requests_per_window / window_seconds`` 的速率匀速补充 ——
            用令牌桶而不是固定窗口计数，是因为后者的计数在窗口边界会清零，
            允许在边界两侧各打满一次、瞬时放行 2 倍流量。
        max_keys: 计数器字典的容量上限（超限时淘汰已回满的条目）。
            **必须有上限**：身份来自请求头，不设限时一个伪造身份的脚本
            就能让这个字典无限增长，把一次外部扫描变成一次内存耗尽。
        exempt_paths: 不计入限流、也**不会**被限流的路径前缀。
            探针必须在这里 —— 被限流的 ``/healthz`` 会让编排系统误判容器已死，
            进而把它重启，而它其实是健康的（经典的「健康检查把自己打挂」）。
    """

    enabled: bool = True
    requests_per_window: int = Field(default=120, ge=1)
    window_seconds: float = Field(default=60.0, gt=0)
    max_keys: int = Field(default=10_000, ge=1)
    exempt_paths: tuple[str, ...] = ("/healthz", "/readyz", "/metrics")


# ------------------------------------------------------------------------------
# 三、PostgreSQL
# ------------------------------------------------------------------------------
class DBSettings(_StrictModel):
    """业务库与 AgentScope 存储的连接参数。

    Attributes:
        url: SQLAlchemy 异步连接串（``postgresql+asyncpg://…``）。
        pool_size: 连接池常驻连接数。
        max_overflow: 池满后允许临时超出的连接数。
        pool_recycle_seconds: 连接被回收重建的秒数。
        pool_timeout_seconds: 池被占满时，等一条空闲连接的秒数上限。
        statement_timeout_seconds: **单条语句**的执行秒数上限。
        create_tables: 启动时是否自动建表。
        echo: 是否把 SQL 打到日志（**生产必须关**）。
    """

    url: str
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=20, ge=0)
    pool_recycle_seconds: int = Field(default=1800, ge=-1)
    # ⚠️ 必须 > 0。SQLAlchemy 的 QueuePool 默认值是 30s —— 也就是说
    #    「池被占满时用户要等半分钟才拿到失败」。取 0 会被 SQLAlchemy
    #    解释成「无限等」，那正是要消灭的无界等待，故在这里就拒绝。
    pool_timeout_seconds: float = Field(default=5.0, gt=0, le=300)
    # ⚠️ 同样必须 > 0。PostgreSQL 的 statement_timeout=0 意思是「不限制」，
    #    那会让一次全表扫描把连接一直占着，直到池被占满
    #    （而池满的后果见上一项）。所以这里不接受 0：要关只能在代码里改，
    #    不能靠改一个配置项悄悄关掉。
    statement_timeout_seconds: float = Field(default=10.0, gt=0, le=600)
    create_tables: bool = True
    echo: bool = False


# ------------------------------------------------------------------------------
# 四、Redis
# ------------------------------------------------------------------------------
class RedisSettings(_StrictModel):
    """缓存 / 会话 / 消息总线的连接参数。

    Attributes:
        url: ``redis://[:password@]host:port/db`` 形式的连接串。
        max_connections: 连接池上限。SSE 长连接会长期占用，故按并发会话数估。
        socket_timeout_seconds: 单次操作超时。**必须显式设置**，否则依赖不可达时
            表现为请求挂死而非快速失败。
    """

    url: str
    max_connections: int = Field(default=50, ge=1)
    socket_timeout_seconds: float = Field(default=5.0, gt=0)


# ------------------------------------------------------------------------------
# 五、Milvus
# ------------------------------------------------------------------------------
class MilvusSettings(_StrictModel):
    """向量库连接与集合参数。

    Attributes:
        uri: Milvus gRPC 地址。
        collection: 集合名。本项目为**单集合**策略，所有知识库共用，靠 metadata
            过滤做 KB / 租户隔离。
        dimension: 向量维度，必须与 Embedding 模型输出一致。
        index_type: 索引类型（HNSW / IVF_FLAT …）。
        metric_type: 距离度量（COSINE / L2 / IP）。
        top_k: 召回候选条数。
        search_timeout_seconds: 单次**读**操作（检索 / 列文档 / 列分块）的
            超时秒数。⚠️ 这不是性能调优项，是**可用性**项 —— 理由见
            :mod:`src.knowledge.guard` 的模块文档（真实故障：
            Milvus 卡住 ⇒ 检索永不返回 ⇒ 对话请求假死）。
            超过它，本次检索降级为「无结果」，连续超时还会熔断。
    """

    uri: str
    collection: str
    dimension: int = Field(default=1024, ge=1)
    index_type: Literal["HNSW", "IVF_FLAT", "IVF_SQ8", "FLAT", "AUTOINDEX"] = "HNSW"
    metric_type: Literal["COSINE", "L2", "IP"] = "COSINE"
    top_k: int = Field(default=5, ge=1, le=100)
    search_timeout_seconds: float = Field(default=5.0, gt=0, le=120)


# ------------------------------------------------------------------------------
# 五·B、Embedding（向量化模型，「三合一」降级链）
# ------------------------------------------------------------------------------
class EmbeddingSettings(_StrictModel):
    """把文本变成向量的模型参数。

    ⚠️ **为什么是一个「降级链」而不是一个模型名**：本项目的检索能力要能在
    三种环境下都跑得起来 —— 有云 key 的正式环境、没有 key 但装了本地模型的
    开发机、以及既没有 key 也没有模型的 CI。三者需要的不是三套代码，而是
    同一个接口的三个实现（见 :mod:`src.web_embedding`）。``provider`` 只声明
    **首选**实现，``allow_fallback`` 决定首选不可用时是降级还是**启动即失败**。

    ⚠️ ``dimension`` 必须与 :attr:`MilvusSettings.dimension` **相等**，
    这条约束由 :class:`Settings` 的 validator 强制（见那里的说明）——
    不相等时写入和检索都会失败，而失败发生在**第一次写数据**的时候，
    离配置错误已经很远了。

    Attributes:
        provider: 首选实现。``dashscope`` 走云；``local`` 走本地 ONNX 模型
            （**不引入 torch**，见 :mod:`src.web_embedding`）；``mock`` 是
            确定性假向量，只用于测试与离线冒烟。
        model: 云端模型名（``provider=dashscope`` 时使用）。
        dimension: 输出向量维度。**必须与 Milvus 集合维度一致。**
        local_model: 本地模型标识（``provider=local`` 时使用）。
        allow_fallback: 首选不可用时是否按 ``local`` → ``mock`` 降级。
            ⚠️ 置为 ``False`` 意味着「宁可起不来也不用假向量」——
            生产环境应当如此：Mock 向量之间没有语义，检索会「成功」但结果无意义，
            而那比检索失败更难发现。
        timeout_seconds: 单次向量化调用的超时秒数。
            ⚠️ 与 :attr:`MilvusSettings.search_timeout_seconds` 同一性质：
            不是性能调优项，是**可用性**项。检索路径上「先向量化、再查库」，
            两段都必须有界 —— 只箍住查库那一段的话，向量模型卡住时
            检索会**卡在向量化那一步**，而熔断器连账都记不上
            （详见 :mod:`src.web_embedding.bounded` 的模块文档）。
            默认 30s 比检索超时宽松一档：向量化是**外网 + 批量**调用，
            正常就该比一次库内检索慢，卡到 30s 已属故障。
    """

    provider: Literal["dashscope", "local", "mock"] = "dashscope"
    model: str = "text-embedding-v4"
    dimension: int = Field(default=1024, ge=1)
    local_model: str = "BAAI/bge-small-zh-v1.5"
    allow_fallback: bool = True
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)


# ------------------------------------------------------------------------------
# 五·C、检索重排（Rerank —— 可选阶段）
# ------------------------------------------------------------------------------
class RerankSettings(_StrictModel):
    """检索结果重排的开关与模型。

    ⚠️ **本框架的重排是 LLM-as-reranker，不是 DashScope 的 TextReRank 接口。**
    ``RAGMiddleware`` 收的是一个 :class:`~agentscope.model.ChatModelBase`，
    再用一段提示词让**对话模型**给候选打分排序
    （``middleware/_rag.py:85-94`` 是那段提示词，``:153`` 是构造参数）。
    所以 ``model`` 要填**对话模型**名；填 ``qwen3-rerank`` 这类
    **专用重排模型**名会在调用时报错 —— 框架对重排失败是尽力而为
    （``_rag.py`` 里吞掉异常、退回向量序），症状是「重排看起来配了但没有效果」。

    ⚠️ ``enabled`` **默认关闭**：重排会让每次检索多一次模型调用
    （延迟 + token 费用），而收益只体现在召回质量上。
    这类「花钱换质量」的开关应当是显式的，不是一个配了模型名就默认打开的副作用。

    Attributes:
        enabled: 是否启用重排阶段。
        model: 重排用的**对话**模型名。**留空 = 复用主对话模型**
            （``settings.llm.model``）。留空是默认值 —— 多数部署没有
            「专门拿来当裁判的模型」，复用主模型即可。
    """

    enabled: bool = False
    model: str = ""


# ------------------------------------------------------------------------------
# 五·D、长期记忆 / 用户画像
# ------------------------------------------------------------------------------
class MemorySettings(_StrictModel):
    """长期记忆（用户画像）的开关与参数。

    ⚠️ ``reme_enabled`` **默认关闭**，且这是一个刻意的默认值，不是「还没接」。
    框架侧有三套长期记忆中间件（``AgenticMemoryMiddleware`` /
    ``Mem0Middleware`` / ``ReMeMiddleware``，见
    :mod:`agentscope.middleware._longterm_memory`），本项目**以自研画像为主**
    （Postgres 结构化 + Milvus 语义召回，见 :mod:`src.memory`）：
    自研那套的存储、字段、生命周期都在我们手里，能进业务库、能写 SQL、
    能被 /api/v1 直接读写；而框架那三套各自带着自己的存储形态
    （ReMe 落文件、mem0 落它自己的向量库），一旦成为主路径，
    用户画像就不再是我们的数据了。

    Attributes:
        enabled: 自研画像总开关。关闭时不注入画像中间件，也不写画像表。
        reme_enabled: 是否**额外**挂上框架的 ``ReMeMiddleware``。
            默认关闭；打开前请确认 ``reme-ai`` 已安装（未安装时框架抛
            ``ImportError`` 并提示装 ``agentscope[memory-reme]``）。
        workspace_dir: ReMe 的工作目录（``reme_enabled`` 时才使用）。
        top_k: 语义召回返回条数。
    """

    enabled: bool = True
    reme_enabled: bool = False
    workspace_dir: str = ".reme"
    top_k: int = Field(default=5, ge=1, le=50)


# ------------------------------------------------------------------------------
# 六、LLM
# ------------------------------------------------------------------------------
class LLMSettings(_StrictModel):
    """大模型调用参数（含零密钥降级与熔断）。

    Attributes:
        provider: 协议族。决定用哪个框架模型类，见 ``src/llm/factory.py``：

            * ``dashscope`` —— 走 :class:`agentscope.model.DashScopeChatModel`。
              这是**当前默认**（甲方 2026-10-01 拍板），因为阿里云百炼是该模型
              ``qwen3.8-flash`` 的原生平台，框架对它有专门实现：除基础采样参数外
              还支持 ``thinking_enable`` / ``thinking_budget`` / ``top_k`` 与
              DashScope 特有的消息格式化（``DashScopeChatFormatter``）。
              把 ``thinking_*`` 用起来是 P3「显示推理」的前提 —— 走通用的
              OpenAI 兼容协议拿不到这些字段。
            * ``openai`` —— 走 :class:`agentscope.model.OpenAIChatModel`。
              保留它是因为**任何 OpenAI 兼容端点**（vLLM、SGLang、One-API
              网关、以及 DeepSeek 等厂商）都能用它接入，换服务商只需改
              ``base_url`` 而不用改代码。私有化部署场景会用到这一档。
        model: 模型名。默认 ``qwen3.8-flash`` —— **必须与 provider 匹配**：
            把 DashScope 的模型名配给 ``openai`` 档会得到 404，反之亦然。
        api_key: 密钥。为空且 ``use_mock_when_no_key`` 为真时降级为 MockLLM。
        base_url: API 端点。留空时由**凭据类自己的默认值**兜底
            （``OpenAICredential`` / ``DashScopeCredential`` 各有一个官方端点），
            因此不必在配置里硬写。
        timeout_seconds: 单次请求超时。
        max_retries: 框架层重试次数（固定间隔、无指数退避，只对可重试异常生效）。
        retry_backoff_seconds: 重试之间的固定等待。
        max_tokens: 单次生成上限。
        temperature: 采样温度。
        use_mock_when_no_key: 零密钥降级开关，见 ``config/base.yaml`` 的说明。
        circuit_breaker_failure_threshold: 熔断阈值（连续失败次数）。
        circuit_breaker_recovery_seconds: 熔断冷却时长（秒）。
    """

    #: 合法取值与 ``src/llm/factory.py`` 里的 ``PROVIDER_*`` 常量**一一对应**。
    #: 用 Literal 而不是 str，是为了让写错的 provider 在**配置加载期**就报错，
    #: 而不是等到第一次模型调用时才变成一个看不懂的 AttributeError。
    provider: Literal["dashscope", "openai"] = "dashscope"
    model: str = "qwen3.8-flash"
    api_key: str = ""
    base_url: str = ""
    timeout_seconds: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=10)
    retry_backoff_seconds: float = Field(default=0.5, ge=0)
    max_tokens: int = Field(default=4096, ge=1)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    use_mock_when_no_key: bool = True
    circuit_breaker_failure_threshold: int = Field(default=5, ge=1)
    circuit_breaker_recovery_seconds: float = Field(default=30.0, gt=0)


# ------------------------------------------------------------------------------
# 七、可观测
# ------------------------------------------------------------------------------
class ObservabilitySettings(_StrictModel):
    """指标与 trace 的参数。

    Attributes:
        metrics_enabled: 是否暴露 ``/metrics``。
        metrics_interval_seconds: 仅作人工核对用，**不驱动抓取**（见 base.yaml 注释）。
        service_name: trace 的 ``service.name`` 资源属性。
        trace_exporter: 导出通道；``otlp`` 需与 ``otlp_endpoint`` 同时给出。
        otlp_endpoint: OTLP 接收端 URL。
        langfuse_public_key / langfuse_secret_key: OTLP 的 Basic 认证凭据。
            为空时**跳过导出器装配并告警**，而不是装一个注定 401 的导出器。
        langfuse_enabled: Langfuse 接入开关（供日志与文档引用）。
        langfuse_host: Langfuse 控制台地址。
    """

    metrics_enabled: bool = True
    metrics_interval_seconds: int = Field(default=15, ge=1)
    service_name: str = "aligo-travel"
    trace_exporter: Literal["none", "console", "otlp"] = "none"
    otlp_endpoint: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_enabled: bool = False
    langfuse_host: str = ""

    @property
    def otlp_ready(self) -> bool:
        """OTLP 导出器是否**具备装配条件**。

        装配条件是「通道选的是 otlp」**且**「端点非空」—— 二者缺一不装。
        之所以把这条判断收成一个属性而不是散在两处（`tracing.py` 与调用方），
        是因为它决定了「trace 到底有没有在发」这件事；判断分散时，
        两边漂移的症状是「以为开着、其实一条没发」，而日志层面看不出任何异常。

        Returns:
            `bool`: 具备装配条件返回 True。
        """
        return self.trace_exporter == "otlp" and bool(self.otlp_endpoint.strip())


# ------------------------------------------------------------------------------
# 八、灰度
# ------------------------------------------------------------------------------
class GraySettings(_StrictModel):
    """灰度发布开关。

    Attributes:
        enabled: 总开关；关闭时全部流量走 ``default_version``。
        default_version: 未命中灰度规则时的兜底版本。
    """

    enabled: bool = False
    default_version: str = "v1"


# ------------------------------------------------------------------------------
# 九、编排（快慢车道 + 多智能体调度）
# ------------------------------------------------------------------------------
class OrchestrationSettings(_StrictModel):
    """快慢车道分流与子智能体调度的参数。

    形状定义在 :mod:`src.config.schema`，使用方是
    :mod:`src.orchestration.classifier`（读阈值）与
    ``src/orchestration/lane.py``（读开关）。

    Attributes:
        fast_lane_enabled: 快车道**总开关**。关闭后全部流量走慢车道。
            ⚠️ 留这个开关不是为了省事，而是为了**排查**：线上出现「回复
            内容不对」时，第一个要排除的假设就是「是不是快车道规则误命中了」。
            把它关掉就能一刀切开两种可能，而不必回滚代码。
        fast_lane_max_chars: 归一化后允许走快车道的最大长度。
            见 :func:`src.orchestration.classifier.classify` 的说明 ——
            它是第二道网，不是主要机制。
        intent_confidence_threshold: 意图置信度低于此值时**追问澄清**，
            而不是按最高分意图直接调度。
            ⚠️ 阈值存在的意义：模型在「都不太像」时会挑一个最接近的，
            此时它的 confidence 通常不高。低于阈值的正确处置是问用户，
            而不是替用户猜 —— 猜错的代价是整条链路白跑。
        dynamic_prompt_enabled: 是否按对话阶段动态组装 system prompt
            （博客的 ``get_prompt_main_plan``）。关闭后使用静态提示词，
            便于对比「动态提示词到底带来了多少收益」。
        max_subagent_calls: 单轮最多调度几个子智能体。
            ⚠️ 必须有上限：多意图输入（「订票并查一下报销标准」）会解析出
            多个意图，若无上限，一次请求可能触发一串子调用，把延迟和成本
            都放大到不可控。超限时的处置由编排层决定（通常是先做前 N 个
            并告知用户其余待办）。
        expose_reasoning: 是否把模型的推理过程（显示推理）呈现给用户。
            关闭后思考链只显示任务状态，不显示推理文本。
        reply_guard_enabled: 回复守卫总开关
            （见 :class:`src.orchestration.reply_guard.ReplyGuardMiddleware`）。
            ⚠️ 关掉它会**同时**失去两件事：草稿剥离，以及「整轮只剩草稿时
            要求模型重说」。后者是用户拿不到答案时的唯一补救手段，
            所以只在排障时短时关闭。
        reply_guard_max_retries: 回复不可用时最多要求模型重说几轮。
            ``0`` 表示不重试（仍会剥离草稿，并在彻底无内容时发兜底话术）。
            ⚠️ 每重试一次就多一次模型调用，直接叠加到用户的等待时间上；
            设成 3 以上会让「模型状态不好」的一天变成「每个请求都等十几秒」。
    """

    fast_lane_enabled: bool = True
    fast_lane_max_chars: int = Field(default=20, ge=1, le=200)
    intent_confidence_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    dynamic_prompt_enabled: bool = True
    max_subagent_calls: int = Field(default=5, ge=1, le=20)
    expose_reasoning: bool = True
    reply_guard_enabled: bool = True
    reply_guard_max_retries: int = Field(default=2, ge=0, le=5)


# ------------------------------------------------------------------------------
# 根配置
# ------------------------------------------------------------------------------
class Settings(_StrictModel):
    """配置树根节点 —— 全项目配置访问的唯一入口。

    消费方的标准用法是拿到一个 ``Settings`` 实例后按属性访问
    （``settings.llm.model``），**不要**再去读 ``os.environ``：
    环境变量只是配置的一种来源，绕过本对象就等于绕过了校验与默认值。
    """

    app: AppSettings = Field(default_factory=AppSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    ratelimit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    db: DBSettings
    redis: RedisSettings
    milvus: MilvusSettings
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    rerank: RerankSettings = Field(default_factory=RerankSettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    observability: ObservabilitySettings = Field(
        default_factory=ObservabilitySettings,
    )
    gray: GraySettings = Field(default_factory=GraySettings)
    orchestration: OrchestrationSettings = Field(
        default_factory=OrchestrationSettings,
    )

    @model_validator(mode="after")
    def _embedding_and_collection_dimensions_agree(self) -> Settings:
        """Embedding 的输出维度必须与 Milvus 集合的维度**相等**。

        ⚠️ 这条约束为什么值得放在这里（而不是等第一次写入时报错）：

        Milvus 的集合维度是**建集合那一刻**定死的。维度不一致的后果分两种，
        两种都比「启动失败」糟得多：

        · **建集合之前**配错 —— 集合按 1024 建出来，而模型吐 768 维向量。
          写入时 Milvus 报维度不匹配，**但报错发生在第一次灌数据的时候**，
          那时集合已经存在，改配置也救不回来，得先删集合。
        · **换模型** —— 把 embedding model 换成一个不同维度的，忘了改
          ``milvus.dimension``。此时**写入正常、检索正常、结果全错**：
          向量对不上号，却因为维度恰好被 Milvus 截断/拒绝而表现成
          「召回质量突然变差」。这是最难排查的一类故障。

        放在这里，两种都变成**启动即失败**，且错误信息直接指出是哪两个值不一致。
        配置错误的发现成本，从「上线后某天」降到「启动那一秒」。

        ⚠️ 注意本约束**只覆盖「我们不能容忍的」不一致**：``provider=mock``
        时维度是可配置的假向量，仍然要求相等 —— 假向量也必须与集合对齐，
        否则冒烟测试会以「写入失败」的形式失败，而那会被误读成 Milvus 的问题。

        Returns:
            `Settings`: 校验通过的自身。

        Raises:
            ValueError: 两个维度不相等时。
        """
        if self.embedding.dimension != self.milvus.dimension:
            raise ValueError(
                f"embedding.dimension={self.embedding.dimension} 与 "
                f"milvus.dimension={self.milvus.dimension} 不一致。\n"
                f"两者必须相等：Milvus 集合的维度在**建集合时**定死，"
                f"而向量由 Embedding 模型产出，不一致会让写入失败"
                f"（改配置也救不回来，得先删集合），"
                f"或者在换模型后表现为「写入检索都正常、召回质量莫名变差」。\n"
                f"改 ALIGO__EMBEDDING__DIMENSION 或 ALIGO__MILVUS__DIMENSION，"
                f"使二者一致（默认都是 1024）。",
            )
        return self


__all__ = [
    "AppSettings",
    "AuthSettings",
    "DBSettings",
    "EmbeddingSettings",
    "GraySettings",
    "LLMSettings",
    "MemorySettings",
    "MilvusSettings",
    "ObservabilitySettings",
    "OrchestrationSettings",
    "RateLimitSettings",
    "RedisSettings",
    "RerankSettings",
    "Settings",
]
