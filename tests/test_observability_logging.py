# -*- coding: utf-8 -*-
"""日志装配：级别真的落到了日志系统上，且**会泄漏载荷的库被静音**。

═══ 为什么这个文件存在 ═══

    ``configure_logging`` 做三件事，前两件有明确的失败症状（日志太少），
    第三件恰恰相反 —— 症状是**日志太多、多到把用户数据写进去**：

    1. 把 ``app.log_level`` 应用到 root 与框架 logger（不配就等于没配）；
    2. 给每条日志注入 ``trace_id``；
    3. 把**会打请求/响应原文**的第三方 logger 钉在 WARNING 以上。

    第 3 件是 2026-10-03 实测出来的：DEBUG 档下 ``dashscope`` 把
    「用户原话 + 1024 维嵌入向量」整段写进了日志（原文见
    ``src/observability/logging.py`` 的 ``_QUIET_LOGGERS``）。
    向量不是「一串无害的数字」—— 它是文本的可逆近似，属于个人数据。

═══ ⚠️ 这里只能测「级别」，测不到「有没有人真的这么打过」 ═══

    库将来改了日志内容（比如升级后不再打 body），本文件的用例**照样绿**。
    这不是缺陷，是分工：本文件守的是「我们的静音没被绕过、没被写反」；
    「这个库今天到底打不打 body」只能靠真的抓一次日志（那次实测的结论写在
    ``_QUIET_LOGGERS`` 的注释里，并在名单上留了「升级后复看」的提示）。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import pytest

from src.config import Settings
from src.observability.logging import (
    _QUIET_LOGGERS,
    configure_logging,
    reset_logging,
)

#: 本项目的 logger —— 静音名单**绝不能**波及它。
_OUR_LOGGER = "src.orchestration.reply_guard"


@pytest.fixture(autouse=True)
def _restore_logging_state() -> Iterator[None]:
    """用例前后把 logging 的全局状态还原。

    ⚠️ ``configure_logging`` 会**清空 root 的全部 handler** 并改各级别 ——
    那是它的正常职责，但落在单进程的 pytest 里就是跨用例污染：
    前一条用例把 root 设成 DEBUG 并装了自己的 handler，后一条用例里
    ``caplog`` 之类的工具就会看到另一套东西。所以这里快照 + 还原。

    Returns:
        `Iterator[None]`: 无值夹具。
    """
    root = logging.getLogger()
    saved_level = root.level
    saved_handlers = list(root.handlers)
    saved_ours = {
        name: logging.getLogger(name).level
        for name in (*_QUIET_LOGGERS, _OUR_LOGGER)
    }
    reset_logging()
    yield
    root.setLevel(saved_level)
    root.handlers[:] = saved_handlers
    for name, level in saved_ours.items():
        logging.getLogger(name).setLevel(level)
    reset_logging()


def _settings_with_level(settings: Settings, level: str) -> Settings:
    """复制一份配置并改掉日志级别。

    Args:
        settings (`Settings`): 基准配置（``settings`` 夹具给的测试档）。
        level (`str`): 目标级别名。

    Returns:
        `Settings`: 改过级别的副本。
    """
    return settings.model_copy(
        update={
            "app": settings.app.model_copy(update={"log_level": level}),
        },
    )


# ==============================================================================
# 一、应用级别真的生效
# ==============================================================================
def test_the_configured_level_reaches_the_root_logger(settings: Settings) -> None:
    """★ ``app.log_level`` 必须落到 root 上（否则配置项是个死键）。"""
    configure_logging(_settings_with_level(settings, "DEBUG"))

    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger(_OUR_LOGGER).getEffectiveLevel() == logging.DEBUG, (
        "本项目自己的 logger 没拿到配置的级别 —— 排障时什么都看不到"
    )


# ==============================================================================
# 二、会泄漏载荷的库被静音
# ==============================================================================
def test_payload_leaking_loggers_never_follow_debug(settings: Settings) -> None:
    """★★ DEBUG 档下，会打载荷的库也**不许**打到 DEBUG。

    ⚠️ 这条是本次修复的核心断言。名单里每一个库，实测都在 DEBUG 档打过
    请求/响应原文（含用户原话与嵌入向量），所以「跟着 DEBUG 走」就等于
    「把用户数据写进日志」。
    """
    configure_logging(_settings_with_level(settings, "DEBUG"))

    for name in _QUIET_LOGGERS:
        effective = logging.getLogger(name).getEffectiveLevel()
        assert effective >= logging.WARNING, (
            f"{name} 在 DEBUG 档下仍然低于 WARNING（实际 {effective}）—— "
            "它的 DEBUG 日志里是请求/响应原文。"
        )


def test_the_quiet_list_is_not_empty() -> None:
    """★ 名单被清空时上面那条用例会**空转通过**。

    ⚠️ 这是一个真实的假绿形态：``for name in ()`` 什么也不检查，用例全绿，
    而泄漏原样存在。所以必须单独钉一句「名单里有东西」。
    """
    assert _QUIET_LOGGERS, "静音名单空了 —— 上面那条用例会空转通过"
    assert "dashscope" in _QUIET_LOGGERS, (
        "实测泄漏过用户原话与嵌入向量的库不在名单里"
    )


def test_quieting_can_only_lower_verbosity(settings: Settings) -> None:
    """★★ 静音**只能往安静的方向调**，绝不能反过来把日志调响。

    ⚠️ 这是最容易写错的一处：直接 ``setLevel(WARNING)`` 在
    「应用级别 = ERROR」的部署里会把 ``dashscope`` 从 ERROR **调低**到
    WARNING —— 一个以「静音」为名的开关反而让日志变多。所以实现用的是
    ``max(应用级别, WARNING)``，这条用例把这个性质钉住。
    """
    configure_logging(_settings_with_level(settings, "ERROR"))

    for name in _QUIET_LOGGERS:
        assert logging.getLogger(name).getEffectiveLevel() == logging.ERROR, (
            f"{name} 的级别被静音逻辑改写成了别的值 —— "
            "静音只该让它更安静，不该让它更啰嗦。"
        )


def test_quieting_survives_the_framework_logger_pass(settings: Settings) -> None:
    """★ 静音必须在「统一框架 logger 级别」**之后**执行。

    ⚠️ 顺序写反的症状很隐蔽：静音先执行、随后被 ``setLevel(level)`` 覆盖，
    结果就是「代码看着做了静音，日志里照样有向量」——没有任何报错。
    这条用例用一个**同时**在框架名单与静音名单里的名字来验顺序：
    若顺序反了，它的级别会是 DEBUG 而不是 WARNING。
    """
    import src.observability.logging as logging_module

    original = logging_module._QUIET_LOGGERS  # noqa: SLF001
    try:
        logging_module._QUIET_LOGGERS = (*original, "agentscope")  # noqa: SLF001
        configure_logging(_settings_with_level(settings, "DEBUG"))

        assert logging.getLogger("agentscope").getEffectiveLevel() >= (
            logging.WARNING
        ), "静音被后面的 setLevel(level) 覆盖了 —— 顺序反了。"
    finally:
        logging_module._QUIET_LOGGERS = original  # noqa: SLF001


def test_the_trace_id_filter_is_installed_on_the_handler(
    settings: Settings,
) -> None:
    """★ 每条日志都要带上 ``trace_id`` —— 这是全链路排查的唯一线索。

    ⚠️ 断言落在 **handler 的 filter** 上，不是「日志里有 `tr-` 字样」：
    后者在 trace_id 为空时会退化成 ``-`` 占位，仍然「看起来正常」。
    """
    from src.observability.logging import _TraceIdFilter

    configure_logging(_settings_with_level(settings, "INFO"))

    handlers = logging.getLogger().handlers
    assert handlers, "root 上没有任何 handler —— 日志会被直接丢掉"
    assert any(
        isinstance(f, _TraceIdFilter) for h in handlers for f in h.filters
    ), "没有任何 handler 装了 trace_id 过滤器"
