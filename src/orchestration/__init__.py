# -*- coding: utf-8 -*-
"""**编排层**：决定「这一轮走哪条车道、调哪些智能体、提示词怎么组织」。

文件职责：
    把用户输入变成**路由决策**，再把路由决策变成实际生效的框架副作用
    （短路模型调用、改写 system prompt、注入上下文）。

上下游依赖：
    - 上游：:mod:`src.domain`、``src/config``，以及框架的
      ``agentscope.middleware``。
    - 下游：``src/server/`` 把中间件装配进 ``create_app``。

═══ ⚠️ 本包的 ``__init__`` **刻意只导出纯逻辑** ═══

:mod:`src.orchestration.classifier`（快慢车道规则引擎）与
:mod:`src.orchestration.prompt`（动态 Prompt 组装）都是纯函数，不 import
``agentscope``；而 :mod:`src.orchestration.lane` / ``.context``
（把判定变成真正短路的中间件）必须 import 框架的 ``MiddlewareBase``。

若这里把两者一起 export，那么

    from src.orchestration.classifier import classify

会因为 Python 先执行包的 ``__init__`` 而**顺带**把 ``agentscope`` 拉进来。
后果是快慢车道与动态 Prompt 的单测从此依赖框架可导入 —— 而它们本来就是
本项目里最该能「脱离框架被验证」的部分（与 :mod:`src.domain` 同理）。

所以本文件**只**导出纯逻辑；框架耦合的模块由使用方按完整路径显式导入：

    from src.orchestration.lane import LaneRouterMiddleware      # 需要 agentscope
    from src.orchestration.context import ContextInjectionMiddleware  # 需要 agentscope
    from src.orchestration.classifier import classify             # 不需要
    from src.orchestration.prompt import build_system_prompt      # 不需要

⚠️ 这不是洁癖，是一条可被验证的性质：
``tests/test_orchestration_classifier.py`` 与 ``tests/test_orchestration_prompt.py``
都在**子进程**里断言 ``agentscope`` 不在 ``sys.modules`` 中 —— 若哪天有人
往这两个纯模块里加了一句框架 import，用例立刻会红。（必须在子进程里验，
因为 ``conftest.py`` 早就把框架导入了，进程内断言必然假通过。）
"""

from __future__ import annotations

from src.orchestration.classifier import (
    FAST_LANE_RULES,
    FastLaneRule,
    classify,
    route_for_intent,
    target_agents_for,
)
from src.orchestration.prompt import (
    MARKER_BEGIN,
    MARKER_END,
    MAX_MISSING_PROMPTS,
    build_system_prompt,
    describe_known_slots,
    strip_managed_sections,
    user_data_excerpt,
)

__all__ = [
    "FAST_LANE_RULES",
    "MARKER_BEGIN",
    "MARKER_END",
    "MAX_MISSING_PROMPTS",
    "FastLaneRule",
    "build_system_prompt",
    "classify",
    "describe_known_slots",
    "route_for_intent",
    "strip_managed_sections",
    "target_agents_for",
    "user_data_excerpt",
]
