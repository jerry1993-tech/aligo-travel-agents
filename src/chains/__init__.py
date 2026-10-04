# -*- coding: utf-8 -*-
"""思考链包：把框架的事件流变成用户看得懂的过程说明。

对外只暴露**纯**的收集器部分::

    from src.chains import TaskCollector, TaskRecord

═══ ⚠️ 这里**只导出不依赖 agentscope 的部分** ═══

与 :mod:`src.orchestration` 同样的理由，但后果更直接：

``src/chains/events.py``（把 ``AgentEvent`` 翻译成 :class:`TaskCollector` 的
调用）必须 import 框架。若在这里把它也导出，那么下面这行看似无害的代码::

    from src.chains import TaskCollector   # 想写个单测

会先执行本文件、进而 import ``events``、进而 import ``agentscope`` ——
于是任务清单的纯逻辑测试**被迫依赖框架**，跑得慢、装不上框架就跑不了，
而且一旦框架的导入路径变了，整个测试文件收集阶段就报错（连不相关的用例
都跑不了）。

所以分工是：
    - 本包 ``__init__`` 与 :mod:`src.chains.collector`：**纯**，可用普通单测穷举。
    - :mod:`src.chains.events`：**必须** import 框架，由集成测试覆盖。

⚠️ 真需要框架侧的翻译层时，**直接 import 子模块**
（``from src.chains.events import ...``），不要加到这里来。
"""

from .collector import DEFAULT_MAX_TASKS, TaskCollector, TaskRecord

__all__ = [
    "DEFAULT_MAX_TASKS",
    "TaskCollector",
    "TaskRecord",
]
