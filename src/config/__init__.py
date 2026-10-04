# -*- coding: utf-8 -*-
"""配置包：把 YAML 与环境变量收敛成唯一的 :class:`Settings` 对象。

对外只暴露四个名字，其余（``_deep_merge`` / ``_expand_env`` 等）都是实现细节：:

    from src.config import Settings, get_settings, load_settings, repo_root

    settings = get_settings()          # 进程内单例，应用启动时调用一次
    settings = load_settings("test")   # 显式加载某一档，测试里常用

约定：**业务代码只依赖 :class:`Settings` 这个类型，不依赖 loader 的加载过程**。
这样将来把配置源从「YAML + 环境变量」换成「配置中心」时，业务代码一行不用改。
"""

from .loader import get_settings, load_settings, repo_root, set_settings
from .schema import (
    AppSettings,
    AuthSettings,
    DBSettings,
    GraySettings,
    LLMSettings,
    MilvusSettings,
    ObservabilitySettings,
    OrchestrationSettings,
    RedisSettings,
    Settings,
)

__all__ = [
    "AppSettings",
    "AuthSettings",
    "DBSettings",
    "GraySettings",
    "LLMSettings",
    "MilvusSettings",
    "ObservabilitySettings",
    "OrchestrationSettings",
    "RedisSettings",
    "Settings",
    "get_settings",
    "load_settings",
    "repo_root",
    "set_settings",
]
