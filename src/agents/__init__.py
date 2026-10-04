# -*- coding: utf-8 -*-
"""智能体层：提示词、注册表、各专家智能体。

对外只暴露**纯**的提示词部分::

    from src.agents import MAIN_AGENT_NAME, PROMPTS, prompt_for

═══ ⚠️ 这里**只导出不依赖 agentscope 的部分** ═══

与 :mod:`src.orchestration` / :mod:`src.chains` 同样的理由，而且本包更典型：

- :mod:`src.agents.prompts` 是**纯数据**（一组字符串 + 一张表），
  不 import 任何东西。它该能被普通单测直接遍历。
- :mod:`src.agents.registry` 与 :mod:`src.agents.intent` 都 import 了
  ``agentscope``（前者要 ``SubAgentTemplate``，后者要 ``Agent``）。

若在本文件里也导出后两者，那么下面这行看起来完全无害的代码::

    from src.agents import PROMPTS   # 想遍历提示词

会先执行本文件、进而 import ``registry``／``intent``、进而 import ``agentscope``
—— 于是**提示词的纯单测被迫依赖框架**：装不上框架就跑不了，框架改了导入
路径则整个测试文件在**收集阶段**就报错（连不相关的用例都跑不了）。

所以分工是：

    - 本包 ``__init__`` 与 :mod:`src.agents.prompts`：**纯**，可用普通单测穷举。
    - :mod:`src.agents.registry`：import 框架，但**只用了框架的一个数据类**
      （``SubAgentTemplate``），不构造 Agent，没有 I/O。
    - :mod:`src.agents.intent`：**必须** import 框架，由集成测试覆盖。

⚠️ 真需要框架侧的东西时，**直接 import 子模块**::

    from src.agents.registry import default_registry
    from src.agents.intent import IntentRecognizer

不要为了「少写几个字」把它们加到这里来 —— 加进来的那一刻，
上面那条「提示词单测不依赖框架」的性质就没了，而且**没有任何测试会因此变红**。
"""

from .prompts import (
    INTENT_PROMPT,
    MAIN_AGENT_NAME,
    MAIN_PLAN_PROMPT,
    PROMPTS,
    prompt_for,
)

__all__ = [
    "INTENT_PROMPT",
    "MAIN_AGENT_NAME",
    "MAIN_PLAN_PROMPT",
    "PROMPTS",
    "prompt_for",
]
