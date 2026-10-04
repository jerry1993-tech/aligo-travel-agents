# -*- coding: utf-8 -*-
"""``app.state`` 上的属性名常量。

文件职责：
    收纳「一个模块写、另一个模块读」的那些字符串键。它们全都是
    ``app.state`` 上的属性名 —— 写错一个字母的症状是**运行时报错**，
    而且报在一个与错字毫不相干的地方（``AttributeError: 'State' object
    has no attribute 'boot_completed'``）。

上下游依赖：
    - 上游：**没有任何 import**。这是刻意的，见下。
    - 下游：:mod:`src.server.app`（写入）、:mod:`src.server.probes`（读取）、
      :mod:`src.storage.engine` 与单测（断言）。

═══ 为什么值一个单独的模块 ═══

``probes.py`` 要读 ``boot_completed``，而它由 ``app.py`` 写。看起来
``probes.py`` 直接 ``from .app import BOOT_COMPLETED_ATTR`` 就行了，
但那是**循环导入**，而且失败得很隐蔽：

    app.py:96   from .probes import router as probes_router   ← 先执行到这里
    app.py:104  BOOT_COMPLETED_ATTR = "boot_completed"        ← 还没执行到

``app.py`` 在第 96 行就把 ``probes`` 拉进来了，而常量定义在其后。
于是 ``probes.py`` 里那句 ``from .app import ...`` 拿到的是一个
**执行到一半**的模块对象 —— 里面还没有这个属性，直接 ImportError。

把名字放进一个**谁都不 import 的叶子模块**是唯一干净的出路：
本模块不 import 任何东西，所以任何模块在任何时刻 import 它都能拿到完整内容。

⚠️ 因此**不要**为了「顺手」往本模块加需要 import 的东西
（日志、类型、配置都不行）。一旦它有了依赖，它就不再是叶子，
上面那个循环就会从另一个方向长回来。
"""

from __future__ import annotations

#: ``app.state`` 上标记「启动是否完成」的属性名。
#:
#: 写入方：``src/server/app.py`` 包装 lifespan 时置位
#: （进入段全部就绪后置 ``True``，退出段开始前置回 ``False``）。
#: 读取方：``src/server/probes.py`` 的 ``/readyz`` 检查。
#:
#: 为什么需要它：``/healthz`` 与 ``/readyz`` 都是 FastAPI 路由，而
#: FastAPI 在 lifespan 进入段执行**之前**就开始接受连接了。存在一个真实
#: 窗口期 —— 进程在响应 HTTP，但 storage / message_bus / scheduler
#: 都还没进异步上下文。只查 PG/Redis/Milvus 会报「就绪」，
#: 而第一个真实请求会因为 ``app.state.chat_service`` 尚不存在而 500。
BOOT_COMPLETED_ATTR = "boot_completed"

#: ``app.state`` 上挂业务库引擎（``AsyncEngine``）的属性名。
#:
#: 写入方：``src/server/app.py`` 的 lifespan（在 ``BOOT_COMPLETED_ATTR``
#: 置位**之前**完成，否则会有「已就绪但没有引擎」的窗口）。
#: 读取方：``/api/v1/**`` 的业务仓储，以及单测。
BUSINESS_ENGINE_ATTR = "business_engine"

#: ``app.state`` 上挂长期记忆门面（``TravelerMemory``）的属性名。
#:
#: 写入方：``src/server/app.py::create_root_app``（装配期，早于 lifespan ——
#: 它要在 ``create_app`` 之前交给中间件工厂）。
#: 读取方：``create_root_app`` 的 lifespan（据此决定要不要建画像表）、
#: ``/api/v1/memory/**`` 的画像接口，以及单测。
MEMORY_ATTR = "traveler_memory"

#: ``app.state`` 上挂全量配置（``Settings``）的属性名。
#:
#: 写入方：``src/server/app.py::create_root_app``（装配期）。
#: 读取方：``/api/v1/default-model`` —— 它要按 ``settings.llm`` 判断零密钥
#: 降级是否生效。
#:
#: ⚠️ 框架**不会**帮我们写它：``agentscope.app`` 的 lifespan 只往
#: ``app.state`` 写 ``storage`` / ``message_bus`` / ``chat_service`` 等
#: 运行期对象（``app/_lifespan.py``），配置是它自己进程内的单例。
#: 于是「路由里怎么拿到 Settings」这件事必须由我们显式约定 ——
#: 不约定的话，最常见的写法是路由里再 ``get_settings()`` 一次，
#: 而那在单测里会读到**与本次装配不同的那份配置**（测试用显式传入的
#: ``settings`` 构造应用），症状是「测试里明明关掉了降级，接口却按开启处理」。
SETTINGS_ATTR = "settings"

#: ``app.state`` 上挂资源访问策略（``ResourceAccessPolicyBase``）的属性名。
#:
#: 写入方：**框架自己** —— ``create_app`` 在装配期就把
#: ``app.state.resource_access_policy`` 置好了（``app/_app.py``，
#: 我们传 ``None`` 时它兜一个 ``DenyAllResourceAccessPolicy``）。
#: 读取方：``/api/v1/default-model``（判断系统凭据对当前用户是否可用）、
#: ``create_root_app`` 的 lifespan（据此决定要不要播种那条系统凭据 ——
#: 没注册策略就不必播种，播了也没人看得见）。
#:
#: ⚠️ 为什么要把这个「框架属性名」抄成本项目的常量：
#: 读不到时 ``getattr(..., None)`` 会**静默**回落到「不共享」，
#: 也就是框架默认的 deny-all 语义 —— 功能看起来完全正常，
#: 只是所有用户都拿不到那条共享凭据，症状与「运营者根本没配密钥」
#: 一模一样。把它写成一个具名常量，至少让「改错拼写」这件事
#: 只有一处可改。**不要**为了对称去 ``setattr`` 它 ——
#: 框架只在 ``create_app`` 装配期写这个属性（``app/_app.py:303-304``），
#: 其 lifespan 仅**读取**（``app/_lifespan.py:56``，再传给
#: ``ResourceAccessService``，见 ``:108``）。所以自己写一次的结果是二选一：
#: 写在 ``create_app`` **之前**会被框架覆盖（白写），写在**之后**则不会被覆盖，
#: 从此 ``app.state`` 上就有两份策略，谁生效取决于谁去读 —— 两种都是两个真相。
RESOURCE_ACCESS_POLICY_ATTR = "resource_access_policy"

__all__ = [
    "BOOT_COMPLETED_ATTR",
    "BUSINESS_ENGINE_ATTR",
    "MEMORY_ATTR",
    "RESOURCE_ACCESS_POLICY_ATTR",
    "SETTINGS_ATTR",
]
