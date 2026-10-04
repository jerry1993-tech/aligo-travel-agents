# -*- coding: utf-8 -*-
"""差旅**业务域**：本项目自己写的业务概念，不依赖 ``agentscope``。

文件职责：
    定义「差旅业务长什么样」—— 意图分类、行程要素、订单状态、申请单与审批流。
    这一层是**纯业务语义**，不含任何框架调用、不读配置、不碰网络，
    因此可以被单测、脚本、智能体三方同样地 import。

上下游依赖：
    - 上游：只依赖 pydantic 与标准库。
    - 下游：
        · ``src/orchestration/``  用 :class:`~src.domain.enums.Intent` 做路线决策；
        · ``src/tools/``          用 :mod:`src.domain.enums` 约束工具入参；
        · ``src/agents/``         用 :mod:`src.domain.schemas` 当结构化输出模型；
        · ``src/storage/``        业务表落库时用同一批枚举，避免「库里存中文、
          代码里比英文」这种一到对账就出错的错配。

═══ 为什么业务域要单独成层，而不是塞进 agents/ ═══

三个理由，按重要性排序：

1. **它是唯一能「脱离框架被验证」的部分**。智能体行为依赖模型输出，只能做
   评测（P5 的 Golden Dataset）；而「订单状态机不允许从 CANCELLED 回到 PAID」
   这类规则是**确定性**的，应当由普通单测覆盖。把规则混进 agent 模块，
   就再也写不出这种测试了。

2. **它要被多方复用**。同一套 :class:`~src.domain.enums.OrderStatus`
   要被「下单工具」「订单查询接口」「审批流」三处使用。放在任何一处都会
   造成另外两处反向依赖。

3. **博客的教训**（``docs/博客原文-Alibaba-Business-Travel.md`` 第 361 行起）：
   项目早期把业务流程与规则**全部写进 Prompt**，得到一个「线性的工作流程
   说明书」，模型要在全量规则上实时推理用户当前状态 —— 这是准确率停在 50%
   的直接原因。把规则从 Prompt 里**搬进代码**，正是本项目要复刻的改进。

⚠️ 本模块**刻意不** import ``agentscope``。这不是洁癖：一旦这里 import 了框架，
   上面第 1 条（可脱离框架验证）就没了，而且 P5 的评测脚本需要在不启动
   任何服务的情况下加载这批模型。
"""

from __future__ import annotations

from src.domain.enums import (
    AgentName,
    CabinClass,
    Intent,
    LaneName,
    OrderStatus,
    TaskState,
    TransportMode,
    TripStage,
)
from src.domain.rules import (
    PolicyLimit,
    PolicyVerdict,
    TransitionCheck,
    check_cabin,
    check_flight_price,
    check_hotel_price,
    check_transition,
    check_transport_mode,
)
from src.domain.schemas import (
    IntentDecision,
    IntentRecognitionResult,
    RouteDecision,
    TravelRequest,
)

__all__ = [
    # 枚举
    "AgentName",
    "CabinClass",
    "Intent",
    "LaneName",
    "OrderStatus",
    "TaskState",
    "TransportMode",
    "TripStage",
    # 结构化模型
    "IntentDecision",
    "IntentRecognitionResult",
    "RouteDecision",
    "TravelRequest",
    # 规则引擎
    "PolicyLimit",
    "PolicyVerdict",
    "TransitionCheck",
    "check_cabin",
    "check_flight_price",
    "check_hotel_price",
    "check_transition",
    "check_transport_mode",
]
