# -*- coding: utf-8 -*-
"""评测数据集包：金标准用例的存放地。

文件职责：
    作为 ``tests/evaluation/`` 的包标记，并把「这是一个**数据**目录」这件事
    写清楚。本包**不含**任何可执行逻辑 —— 里面只有 ``golden_dataset.yaml``。

上下游依赖：
    - 上游：无。本包刻意保持无依赖（连 ``src`` 都不 import）。
    - 下游：``tests/test_evaluation_dataset.py`` 读取 ``golden_dataset.yaml``
      做自校验；后续的离线评测算子（``scripts/``）也会按同一份路径读它。

═══ ⚠️ 为什么数据集放在 ``tests/`` 而不是 ``config/`` 或 ``docs/`` ═══

因为**它要被测试读、要被断言守住**。一份放在 ``docs/`` 里的评测集，唯一的
守卫是「写它的人记得更新」；而放在 ``tests/`` 下、由
``tests/test_evaluation_dataset.py`` 逐条校验（意图在词表内、工具名真实注册、
快车道判定与 ``classifier.classify`` 一致）之后，任何一处漂移都会变成红灯。

换句话说：本目录的价值不在于「存了一份数据」，而在于**那份数据被可执行的
断言钉住了**。放在别处，同样的文件只是一段会过期的文档。
"""

from __future__ import annotations

#: 本包不对外暴露任何名字 —— 它是数据目录，不是 API。
__all__: list[str] = []
