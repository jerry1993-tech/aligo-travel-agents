# -*- coding: utf-8 -*-
"""AliGo 差旅助手 —— 应用源码包。

本包是**我们自己的代码**。框架依赖（``agentscope`` / ``reme``）自 2026-10-04 起
是**普通的 pip 包**（钉在 ``requirements.txt`` 第 零 节），与 ``src`` 严格分开：

    · ``agentscope`` / ``reme`` 由**安装元数据**解析到当前 Python 环境的
      ``site-packages``（核验：``python -c "import agentscope; print(agentscope.__file__)"``）。
      仓库里不再有 vendored 源码树，也**不设**任何指向源码树的 ``PYTHONPATH`` ——
      一旦同时存在两条解析路径，"实际加载的是哪一份代码"就变得不确定。
    · ``src`` 是本项目的顶层包，靠 ``PYTHONPATH=<仓库根>`` 解析
      （容器里由 Dockerfile 的 ``ENV PYTHONPATH=/app`` 提供）。

包结构（与 docs/03-模块关系与调用逻辑.md 对应）::

    src/config/         配置：schema（形状）+ loader（来源与合并）
    src/llm/            模型装配：factory / breaker（自研熔断）/ mock
    src/observability/  tracing（OTel → Langfuse）/ metrics（Prometheus）
    src/server/         FastAPI 应用：app.py（装配入口）+ probes.py（探针）
    src/agents/         各智能体的系统提示与版本化注册表
    src/orchestration/  快慢车道分流、路由交接、动态提示词组装
    src/tools/          差旅业务工具（FunctionTool 包装）
    src/knowledge/      Milvus 接入、单集合 KB 管理、混合检索
    src/memory/         短期会话 + 自研长期画像（+ ReMe 可选适配）
    src/chains/         思考链（消费 agentscope.event 的事件流）
    src/domain/         业务领域模型与规则引擎
    src/storage/        业务库 engine 与 alembic 迁移

⚠️ 本文件**刻意保持为空**（只有文档字符串）：任何在这里的 import 都会在
「导入 ``src`` 下的任意子模块」时被连带执行，从而把启动路径变得不可控，
也会让一次循环导入以难以定位的方式炸开。
"""
