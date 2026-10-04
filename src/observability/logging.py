# -*- coding: utf-8 -*-
"""日志装配：把 ``app.log_level`` 真正落到日志系统上，并给每条日志带上 trace_id。

文件职责：
    :func:`configure_logging` 是**全项目唯一**配置日志的地方。它在应用启动时
    被调用一次，把 ``settings.app.log_level`` 应用到根 logger 与框架的
    logger 上，并装一个把 ``trace_id`` 注入格式化输出的过滤器。

上下游依赖：
    - 上游：``src/server/app.py`` —— **两处**调用，且第一处是必需的：

          1. **模块级**（``app = create_root_app()`` 之前）—— 装配在 import 期完成，
             而装配期打的每一条日志都是「哪个开关开了、降级了没有」的事实来源，
             不先配置日志它们会被直接丢弃（没有 handler 时 Python 只把 WARNING
             及以上交给 lastResort）；
          2. lifespan 进入段 —— 此时才拿到 ``settings``，仍是权威的一次；
             有了幂等标记，它不会再配置一遍。

      ⚠️ 顺序不能反过来：先装配、后配置 = 装配期的 INFO **永久丢失**，
      事后补配也补不回来（records 已经被丢掉了）。

    - 下游：全项目的 ``logging.getLogger(__name__)`` 都受益于它。

------------------------------------------------------------------------------
为什么必须显式配置（不配也不报错，但配置项会变成死键）
------------------------------------------------------------------------------
    ``settings.app.log_level`` 来自 ``ALIGO__APP__LOG_LEVEL`` / ``config/{env}.yaml``。
    如果不把它应用到日志系统，会发生两件都不报错、但都很难发现的事：

        1. ``config/dev.yaml`` 写的 ``DEBUG`` 完全不起作用，实际还是 Python 的
           默认级别（root 是 WARNING）—— 于是开发者按文档以为自己在看 DEBUG 日志，
           实际连 INFO 都被丢掉了，排障时「什么都没打印」的困惑正来源于此。
        2. **框架自己的日志被静默丢弃**。AgentScope 用
           ``agentscope._logging.logger`` 打日志，它沿用的是标准 logging 的
           有效级别。root 停在 WARNING 时，框架的 INFO 级进度信息（模型调用、
           工具选择、会话装配）一条都看不到，而这正是排查多智能体行为时
           最需要的信息。

------------------------------------------------------------------------------
关于 uvicorn 的 logger
------------------------------------------------------------------------------
    uvicorn 会为自己的 logger（``uvicorn`` / ``uvicorn.error`` /
    ``uvicorn.access``）**预先装好 handler 并设好 propagate=False**。
    因此只改 root 级别对它们是无效的，必须单独设。这里把它们统一到同一级别，
    目的是让「应用日志」与「访问日志」在同一个时间轴上可读 ——
    两者级别不一致时，你会看到访问日志有请求、应用日志却没有对应的处理记录。

------------------------------------------------------------------------------
关于会泄漏载荷的第三方 logger
------------------------------------------------------------------------------
    第三件事与本模块的前两件不同：它不是「级别没生效」，而是**级别生效得太彻底** ——
    ``DEBUG`` 一开，某些 SDK 会把**请求体与响应体原样打进日志**，其中包含
    用户原话与完整的嵌入向量（见 :data:`_QUIET_LOGGERS` 里的实测原文）。

    这类内容的处理方式是**显式静音一个名单**，而不是「dev 别开 DEBUG」：
    关掉 DEBUG 会把我们自己的排障信息一起关掉，等于用失去可观测性换隐私 ——
    而真正需要管的只是那几个库。名单只收**确认打过载荷**的库，
    钉法上只能往安静的方向调（``max(应用级别, WARNING)``），
    绝不反过来把别人的日志调响。
"""

from __future__ import annotations

import logging
import sys
from typing import Final

from ..config import Settings, get_settings
from .context import get_trace_id

#: 日志格式。刻意用**单行**（不换行、不缩进）：
#: 容器日志是逐行采集的（``docker logs`` / Loki / 各类 sidecar 都按行切分），
#: 多行日志会被拆成互不相关的记录，堆栈也会散开。
#: trace_id 放在最前面，是为了让人在终端里能直接用 ``grep tr-xxxx`` 拉出一次请求的全链路。
_LOG_FORMAT = (
    "%(asctime)s.%(msecs)03d %(levelname)-7s [%(trace_id)s] %(name)s: %(message)s"
)

#: 日期部分。**毫秒不在这个串里** —— ``logging`` 用的是 ``time.strftime``，
#: 它不认 ``%f``（写了会被原样打印出来）。
#: 毫秒由 ``_LOG_FORMAT`` 里的 ``%(msecs)03d`` 提供（``LogRecord`` 的现成属性）。
#
#: ⚠️ 这里曾经是「注释写着精确到毫秒、格式串却只有秒」——两者差了一千倍，
#: 而注释给出的理由（秒级精度排查并发先后顺序不够用）恰好是**用得上它的那个场景**。
#: 后果是实打实的：本文档下游的分析要靠「日志结束时刻 − span 耗时」反推开始时刻，
#: 秒级截断给每条区间引入 ±1s 的不确定度，于是「同一时刻几条流在开」这种
#: 小于 1 秒的差别根本判不了（实测因此只能给「≥6 条、无法区分 6 与 7」）。
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: 本模块是否已经配置过（幂等保护，见 :func:`configure_logging`）。
_configured = False

#: **会把请求/响应原样打进日志**的第三方 logger —— 必须钉在 WARNING 及以上。
#:
#: ⚠️ 这条不是因为「吵」，是因为**泄漏**。2026-10-03 实测（
#: ``ALIGO__APP__LOG_LEVEL=DEBUG`` 的容器里抓到的原文）::
#:
#:     22:19:17.107 DEBUG [tr-NMVx…] dashscope: Request body: {'model':
#:         'text-embedding-v4', …, 'input': {'texts': ['帮我规划下周去上海出差。']}}
#:     22:19:17.327 DEBUG [tr-NMVx…] dashscope: Response: {'output': {'embeddings':
#:         [{'embedding': [-0.03275418281555176, -0.057107437402009964, … 共 1024 维]}]}}
#:
#: 两行都进了日志：**用户原话**，以及**完整的嵌入向量**。后者常被当成
#: 「只是一串数字」，但向量是文本的可逆近似 —— 能拿它做相似度检索、聚类，
#: 部分模型上还能反推大意。也就是说它和原文一样属于个人数据。
#:
#: ⚠️ 钉法上有个容易写错的点：这里是
#: ``max(应用级别, WARNING)``（见 :func:`configure_logging`），
#: **不是**直接 ``setLevel(WARNING)``。直接设会让「应用级别=ERROR」的
#: 部署反而把 dashscope 的日志**调响**（WARNING < ERROR）—— 一个「静音」
#: 开关把别人的日志变多，是最难察觉的那类副作用。
#:
#: ⚠️ 只钉**确认会泄漏载荷**的库，实际只有 ``dashscope`` 一家。以下这些在
#: DEBUG 下也很吵，但**逐条看过原文**（同上那次实测），只打连接、状态码、
#: 方法名，不含载荷，所以**刻意不动** —— 排查「为什么这次模型调用慢/失败」
#: 时它们是有用的：
#:
#:     httpcore2.http11 / httpcore2.connection   连接与 TLS 生命周期
#:     openai._base_client                       method / status / request_id
#:     redis.asyncio.connection                  MAINT_NOTIFICATIONS 之类的探测
#:     urllib3.connectionpool                    请求行（无 body）
#:
#: 若将来这些库改成打 body（升级后请复看一遍），照这里的写法加进来即可。
_QUIET_LOGGERS: Final[tuple[str, ...]] = ("dashscope",)


class _TraceIdFilter(logging.Filter):
    """把当前上下文的 ``trace_id`` 注入每条日志记录的过滤器。

    ``logging.Formatter`` 只能读取 ``LogRecord`` 上已有的属性，它不会去
    调用任何函数。因此「动态取值」这件事只能靠 Filter 在格式化之前完成。
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """给记录补上 ``trace_id`` 字段。

        Args:
            record (`logging.LogRecord`): 待处理的日志记录。

        Returns:
            `bool`: 恒为 True —— 本过滤器只做增强，不过滤任何记录。
        """
        # 不在请求上下文里时（启动、定时任务、脚本）用 "-" 占位。
        # 用空串会让日志行里出现一段连续空白，肉眼很难判断「是没有 trace_id」
        # 还是「格式串写错了」。
        record.trace_id = get_trace_id() or "-"
        return True


def configure_logging(settings: Settings | None = None) -> None:
    """配置日志系统（幂等，重复调用直接返回）。

    Args:
        settings (`Settings | None`): 配置；``None`` 时取进程内单例。
    """
    global _configured
    if _configured:
        return

    settings = settings or get_settings()
    level_name = settings.app.log_level.upper()
    # getattr 而不是 logging.getLevelName 反查：字符串转级别没有公开的
    # 正向 API（getLevelName 是级别→名字方向的）。schema 里 log_level 已是
    # Literal 枚举，取值必然合法，故这里不需要额外校验。
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
    handler.addFilter(_TraceIdFilter())

    root = logging.getLogger()
    root.setLevel(level)
    # ⚠️ 先清空已有 handler 再添加。uvicorn 以 `--reload` 或程序化方式启动时
    # 可能已经装过 handler；不清空的话每条日志会被打印两次（一份带 trace_id、
    # 一份不带），看起来像两个不同的组件在打日志。
    # 注意：清空的是**本函数自己能看到**的 handler，uvicorn 装在它自己 logger 上的
    # handler 不受影响（那几个 logger 的 propagate=False）。
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)

    # 框架与服务器的 logger 单独设级别：它们的 propagate 多为 False，
    # 光设 root 对它们无效（见模块文档字符串）。
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "agentscope", "reme"):
        logging.getLogger(name).setLevel(level)

    # 会泄漏载荷的第三方 logger 钉在 WARNING 以上。理由与「为什么用 max」
    # 都在 :data:`_QUIET_LOGGERS` 的说明里。
    # ⚠️ 必须放在上面那次设级别**之后**：先钉后放会被 ``setLevel(level)``
    # 覆盖掉，而且没有任何报错 —— 症状是「DEBUG 日志里又出现向量了」。
    quiet_level = max(level, logging.WARNING)
    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(quiet_level)

    _configured = True
    # 用 root logger 自己打这条，顺带验证格式串与过滤器都已生效。
    logging.getLogger(__name__).info(
        "日志已配置：level=%s env=%s",
        level_name,
        settings.app.env,
    )


def reset_logging() -> None:
    """把 :func:`configure_logging` 的幂等标记清掉，但**不动**已装的 handler。

    **仅供测试使用**：用例之间需要能用不同的 ``log_level`` 重复调用
    :func:`configure_logging`，而模块级的 ``_configured`` 会让第二次调用变成空操作。
    """
    global _configured
    _configured = False


__all__ = [
    "configure_logging",
    "reset_logging",
]
