# -*- coding: utf-8 -*-
"""框架自带长期记忆（``ReMeMiddleware``）的**可选**适配层 —— 默认关闭。

═══ ⚠️ 为什么它是可选的，而不是主路径 ═══

框架提供了三套长期记忆中间件（``AgenticMemoryMiddleware`` /
``Mem0Middleware`` / ``ReMeMiddleware``）。本项目**以自研画像为主**
（:mod:`src.memory.profile` + :mod:`src.memory.semantic`），
理由写在 ``MemorySettings`` 的文档里，这里只重复最关键的一条：

  **ReMe 的记忆落在文件里**（``workspace_dir``，默认 ``.reme``）。
  它不在业务库里、不能写 SQL、不能被 ``/api/v1`` 直接读写，
  也没有多租户的表结构。一旦它成为主路径，用户画像就不再是
  我们的数据 —— 而这套系统的其余部分（审批流、成本中心、
  常旅客号）全都建立在「画像是我方可查可改的记录」之上。

但那不等于它没用。ReMe 的 ``agent_control`` 模式把记忆检索
**当成一个工具**暴露给模型，让模型自己决定「这一轮要不要翻记忆」——
这与我们那种「每轮都注入」的静态策略是两种不同的取舍，
在某些对话里确实更好。

═══ ⚠️ 三件必须说清楚的事 ═══

  1. **安装**。``ReMeMiddleware`` 的应用是**惰性构建**的
     （``agentscope/middleware/_longterm_memory/_reme/_middleware.py:225-231`` 的 ``_build_app``），第一次用才
     ``import reme``。所以「装没装」在装配期是**看不出来**的 ——
     这正是本模块要在装配期**主动探测**一次的原因。
     没装时抛的是 ``ImportError``，而它会在第一次对话时才炸。

  2. **默认关闭**（``ALIGO__MEMORY__REME_ENABLED=false``）。
     打开前请确认镜像里真的装了 ``reme-ai``。

  3. **工作目录是容器内的路径**。不挂卷的话，容器一重建记忆就没了。
     ``docker-compose.yaml`` 里若要长期使用，需要给它挂一个命名卷。
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from src.config.schema import Settings

if TYPE_CHECKING:
    from agentscope.middleware import MiddlewareBase
    from agentscope.tool import FunctionTool

logger = logging.getLogger(__name__)


#: ReMe 的**导入名**（不是发行包名 ``reme-ai``）。
#:
#: ⚠️ 这两个名字不一样：``pip install reme-ai`` 之后 ``import reme``。
#: 用发行包名去 ``find_spec`` 会永远返回 None，于是「装了却探测不到」——
#: 而那会让我们打出一条「请安装 reme-ai」的告警，一个已经装好的人
#: 看到这条只会去重装一遍，然后发现问题依旧。
REME_IMPORT_NAME = "reme"

#: 安装提示里给出的 extras 名（框架自己的报错信息里用的就是它）。
REME_EXTRAS_HINT = "agentscope[memory-reme]"


def reme_available() -> bool:
    """当前环境里能否 ``import reme``。

    ⚠️ 用 ``find_spec`` 而不是 ``import reme``：

      · ``find_spec`` **不会真的把模块加载进来**（只是查找）。
        ``reme`` 的导入会拉起它自己的一整套依赖与配置读取，
        而本函数会在**应用装配期**被调用 —— 那时候我们只是想
        回答一个是非题，不想为一个关闭着的功能付出启动代价。
      · ``find_spec`` 不执行模块级代码，因此不会被一个
        坏掉的第三方包在导入期炸掉整个应用。

    Returns:
        `bool`: ``reme`` 可导入返回 True。
    """
    try:
        return importlib.util.find_spec(REME_IMPORT_NAME) is not None
    except (ImportError, ValueError):
        # ⚠️ ``find_spec`` 在父包不存在或 ``__spec__`` 为 None 时会抛，
        # 而对本函数来说那与「没装」是同一件事。
        return False


@dataclass
class ReMeBundle:
    """ReMe 中间件与它自带的工具。

    ⚠️ 两者**必须一起用**。``ReMeMiddleware`` 的 ``agent_control`` /
    ``both`` 模式会把 ``memory_search`` 暴露成一个工具，
    而那个工具**不在** Toolkit 里 —— 它由中间件自己提供，
    要显式 ``await mw.list_tools()`` 取出来加进 Toolkit
    （见 ``agentscope/middleware/_longterm_memory/_reme/_middleware.py:115-128`` 的示例）。
    只挂中间件、不加工具，症状是「模型说它要搜记忆，但那个工具
    不存在」，而日志里一切正常。
    """

    #: 中间件实例，挂到 ``Agent(middlewares=[...])`。
    middleware: "MiddlewareBase"
    #: 中间件提供的工具，加进 ``Toolkit``。
    tools: list["FunctionTool"] = field(default_factory=list)
    #: 实际生效的模式（``static_control`` / ``agent_control`` / ``both``）。
    mode: str = "both"


async def build_reme(settings: Settings) -> ReMeBundle | None:
    """按配置构建 ReMe 中间件；关闭或不可用时返回 None。

    ⚠️ **不抛异常。** 这是一个可选增强，而它会挂在**每一次对话**
    的路径上。因为一个没装的可选包让整个服务起不来，
    代价与收益完全不成比例。

    ⚠️ 但**不抛**不等于**不说**。关闭时是静默的（那是操作者自己选的），
    **打开却不可用**时打 ``error`` 级日志，把「装了但探测不到」
    与「压根没装」这两种情况分开说 —— 见 :data:`REME_IMPORT_NAME`
    的说明，这正是最容易误判的一处。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `ReMeBundle | None`: 中间件与工具；未启用或不可用时为 None。
    """
    if not settings.memory.reme_enabled:
        # ⚠️ 这是**默认路径**，所以它必须安静。用 debug 而不是 info：
        # 每次启动打一行「ReMe 未启用」只会训练运维忽略启动日志。
        logger.debug("ReMe 长期记忆未启用（默认）。")
        return None

    if not reme_available():
        logger.error(
            "⚠️ ALIGO__MEMORY__REME_ENABLED=true，但当前环境里 "
            "import reme 不可用 —— ReMe 长期记忆**不会生效**，"
            "对话将只使用自研画像。\n"
            "   修法：pip install %s（发行包名是 reme-ai，"
            "导入名是 reme，两者不同）。\n"
            "   注意：本项为 true 时若不同时装好依赖，"
            "框架会在**第一次对话**时才抛 ImportError。",
            REME_EXTRAS_HINT,
        )
        return None

    # ⚠️ 延迟到确认可用之后才 import：模块级 import 会让
    # 「没装 reme 的环境」连本模块都 import 不了，而本模块
    # 要被 ``src.memory.__init__`` 导出。
    from agentscope.middleware import ReMeMiddleware

    chat_model = _build_chat_model(settings)

    middleware = ReMeMiddleware(
        workspace_dir=settings.memory.workspace_dir,
        parameters=ReMeMiddleware.Parameters(
            chat_model=chat_model,
            # ⚠️ 这里**不**传 ``embedding_model``：传了会启用 ReMe 自己的
            # 向量库，那是**第二套**向量存储（既不是 Milvus 也不是
            # 我们的集合），运维会莫名其妙多出一个要备份的东西。
            # 不传时 ReMe 的检索退化为关键词匹配 —— 对一个
            # **关闭着的**可选功能来说，这个降级是可以接受的。
            mode="both",
            top_k=settings.memory.top_k,
        ),
    )

    try:
        tools = await middleware.list_tools()
    except Exception as exc:  # noqa: BLE001 —— 顶层装配，见下面的说明
        # ⚠️ ``list_tools()`` 会**触发惰性构建**（``_build_app``），
        # 也就是真的把 ReMe 应用拉起来。这一步可能因为工作目录
        # 不可写、ReMe 配置缺失等原因失败 —— 而那些原因在
        # 一个已经装好包的机器上照样会发生。
        # 在装配期把它变成一句清晰的日志，比让它留到第一次对话
        # 变成一次 500 要好得多。
        from src.observability.redaction import safe_error

        logger.error(
            "⚠️ ReMe 中间件构建失败，长期记忆降级为「仅自研画像」：%s\n"
            "   常见原因：工作目录 %r 不可写、或 ReMe 自身的配置缺失。",
            safe_error(exc),
            settings.memory.workspace_dir,
        )
        return None

    logger.info(
        "ReMe 长期记忆已启用（mode=both，workspace_dir=%s，工具 %d 个）。"
        "⚠️ 该目录在容器内，不挂卷则随容器重建而丢失。",
        settings.memory.workspace_dir,
        len(tools),
    )
    return ReMeBundle(middleware=middleware, tools=list(tools), mode="both")


def _build_chat_model(settings: Settings) -> Any:
    """给 ReMe 用的 chat model。

    ⚠️ ReMe 的写回（把对话总结成记忆卡片）需要一次 LLM 调用。
    这里**复用本项目的** :func:`src.llm.factory.build_chat_model`，
    而不是让 ReMe 去读它自己的 ``LLM_*`` 环境变量 —— 否则会出现
    「主链路用 DashScope、记忆写回用另一个模型」这种情况，
    两边的密钥、配额、计费都分开，而运维只配了一处。

    ⚠️ 零密钥时 ``build_chat_model`` 会返回 MockLLM（见
    ``src/llm/factory.py``），ReMe 的写回于是产出一堆没有意义的
    记忆卡片。这是刻意的：与其让 ReMe 因为「没有 key」而报错，
    不如让它在一个明显降级的状态下运行 —— 而
    ``/readyz`` 与启动日志里都能看到 ``llm.mock`` 为真。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `Any`: 框架的 chat model 实例。
    """
    from src.llm.factory import build_chat_model

    return build_chat_model(settings)


__all__ = [
    "REME_EXTRAS_HINT",
    "REME_IMPORT_NAME",
    "ReMeBundle",
    "build_reme",
    "reme_available",
]
