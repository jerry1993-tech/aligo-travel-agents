# -*- coding: utf-8 -*-
"""``redact`` / ``safe_error``：探针响应体与日志的最后一道兜底。

═══ 为什么这两个小函数值得一个专门的用例文件 ═══

    ``/readyz`` 的响应体是**最容易被整段贴出去**的东西：它短、看起来人畜无害、
    又是出问题时大家第一个去拿的东西。它里面的 ``detail`` 字段直接来自依赖抛出的
    异常文本（经 :func:`~src.observability.redaction.safe_error`），
    而那个文本里是否有凭据**不由我们决定** —— 由驱动、框架、以及未来某次升级决定。

    所以这一层的正确性只能靠**穷举形态**来钉：每多一种真实出现过的连接串形态，
    就多一条用例。本文件里的每一条都对应一个**具体的、曾经错过的**形态
    （见 :func:`test_a_password_containing_a_slash_is_fully_redacted` 与
    :func:`test_a_password_containing_an_at_sign_is_fully_redacted`），
    而不是"顺手多测几个"。

═══ 与 ``tests/test_probes.py`` 的分工 ═══

    那边是**端到端**：真的起一个应用、打 ``/readyz``、断言响应体里没有密钥 ——
    它回答"这条链路整体安全吗"。
    这边是**单元**：直接喂字符串，回答"脱敏器本身在每种形态下都对吗"。
    端到端用例挡不住这个：它只会喂进**框架今天恰好会抛的**那几种连接串，
    而脱敏器的盲区恰恰在"今天还没抛过的"形态上。
"""

from __future__ import annotations

import pytest

from src.observability.redaction import redact, safe_error

#: 一个形似真实口令、但**不是**真实口令的值。它只用来观察"还在不在"。
_SECRET = "s3cr3t-not-a-real-password"


# ==============================================================================
# 一、常规形态
# ==============================================================================
def test_a_plain_connection_string_is_redacted() -> None:
    """最常见的那一种：``scheme://user:password@host``。"""
    assert (
        redact(f"postgresql+asyncpg://aligo:{_SECRET}@pg:5432/aligo")
        == "postgresql+asyncpg://***:***@pg:5432/aligo"
    )


def test_the_host_and_database_survive_redaction() -> None:
    """⚠️ 抹掉的不该比该抹的多：主机名与库名是排查时**唯一有用**的信息。

    一条只剩 ``***:***`` 的错误信息没有价值 —— 它连"连的是哪个库"都不说。
    """
    result = redact(f"redis://alice:{_SECRET}@redis:6379/0")
    assert _SECRET not in result
    assert "redis:6379" in result, f"主机与端口被一起抹掉了：{result}"
    assert "alice" not in result, "用户名属于凭据，应当一并抹掉"


def test_a_percent_encoded_password_is_redacted() -> None:
    """百分号编码是**正确**的形态（RFC 3986 要求特殊字符编码），当然要能抹。"""
    assert (
        redact("postgresql+asyncpg://aligo:s3cr3t%40x@pg:5432/aligo")
        == "postgresql+asyncpg://***:***@pg:5432/aligo"
    )


# ==============================================================================
# 二、曾经漏掉的两种形态（审计发现，非设想）
# ==============================================================================
@pytest.mark.parametrize(
    "url",
    [
        # 口令**内部**有斜杠。
        f"postgresql+asyncpg://aligo:pa/{_SECRET}@pg:5432/aligo",
        # 斜杠把口令切开，后半段在斜杠之后。
        f"postgresql+asyncpg://aligo:{_SECRET}/x@pg:5432/aligo",
        # 口令紧跟在斜杠之后（口令本身不含斜杠，但边界上有）。
        f"postgresql+asyncpg://aligo:pa/x{_SECRET}@pg:5432/aligo",
    ],
)
def test_a_password_containing_a_slash_is_fully_redacted(url: str) -> None:
    """★ 口令里带**原始** ``/`` 时，整段都必须被抹掉。

    ⚠️ 这条用例的存在本身就是一次失败的记录。旧实现把密码档写成 ``[^/\\s@]+``
    （"密码里不该有 ``/``"），而 ``/`` 一旦出现，正则**整条不匹配** ——
    脱敏器静默地什么都不做，把原文原样返回：

        postgresql+asyncpg://aligo:pa/s3cr3t@pg:5432/aligo
        → 原样返回（一个字符都没抹）

    这比没有脱敏器更危险：调用方以为已经兜住了。
    """
    result = redact(url)
    assert _SECRET not in result, f"口令没有被脱敏：{result}"
    assert result.startswith("postgresql+asyncpg://***:***@"), result


@pytest.mark.parametrize(
    "url",
    [
        f"postgresql+asyncpg://aligo:pa@{_SECRET}@pg:5432/aligo",
        f"postgresql+asyncpg://aligo:{_SECRET}@ss@pg:5432/aligo",
        f"postgresql+asyncpg://aligo:a@b@{_SECRET}@pg:5432/aligo",
    ],
)
def test_a_password_containing_an_at_sign_is_fully_redacted(url: str) -> None:
    """★ 口令里带**原始** ``@`` 时，不能只抹到第一个 ``@`` 就收工。

    旧实现的行为是"提前闭合"：``u:pa@ss@db`` 只匹配到 ``u:pa@``，
    于是 ``ss@db`` 原样留在输出里 —— 密码的**后半段**泄漏，
    而前半段被抹掉这件事会让人误以为整条已经安全。

    ⚠️ 两个字符类都是**贪婪**的，因此匹配延伸到该 token 里最后那个 ``@``，
    这正是本用例要钉住的性质。
    """
    result = redact(url)
    assert _SECRET not in result, f"口令没有被完整脱敏：{result}"
    assert result.endswith("@pg:5432/aligo"), result


# ==============================================================================
# 三、不该被误伤的东西
# ==============================================================================
@pytest.mark.parametrize(
    "text",
    [
        # 没有密码的连接串：用户名本身不构成凭据，抹掉反而丢了「连的哪个用户」。
        "redis://alice@redis:6379/0 连不上",
        # 压根没有 userinfo。
        "https://example.com/path 返回 502",
        # URL 里带查询串/锚点：不该跨过 ? # 去把它当成凭据。
        "https://user@host/path?a=b:c@d 是普通 URL",
        "见 https://example.com/doc#sec:tion@x",
        # 中文与空格的普通句子。
        "Milvus 探针超时，已重试 3 次",
        "",
    ],
)
def test_innocent_text_is_left_alone(text: str) -> None:
    """⚠️ 反向用例：脱敏器**不该**把普通文本吃掉。

    只测"该抹的抹了"是不够的 —— 一个把所有带 ``://`` 的文本都抹成
    ``***:***`` 的实现也能通过上面全部用例，而它会毁掉日志的可读性
    （排查时最需要的就是那句 URL 到底连去哪）。
    """
    assert redact(text) == text


# ==============================================================================
# 四、safe_error：异常 → 一行安全文本
# ==============================================================================
def test_safe_error_keeps_the_type_and_first_line() -> None:
    """"哪个依赖挂了"由异常类型与首行回答，不需要完整 traceback。"""

    class OperationalError(Exception):
        """替身：与 SQLAlchemy 的同名异常一样，只看类名。"""

    result = safe_error(OperationalError("connection refused\n详细栈在此"))
    assert result.startswith("OperationalError: connection refused")
    assert "详细栈" not in result, "只取首行 —— 多行栈会把响应体撑大且无信息量"


def test_safe_error_on_an_empty_message_is_just_the_type() -> None:
    """没有消息时不能退化成空串：那会让 ``/readyz`` 的 detail 变成空白。"""
    assert safe_error(ValueError()) == "ValueError"


def test_safe_error_redacts_a_connection_string_inside_the_message() -> None:
    """★ 端到端的因果链：异常消息里的连接串，必须在这一步被抹掉。

    这条是 :func:`~src.observability.redaction.safe_error` 存在的**全部理由** ——
    它的返回值直接就是 ``/readyz`` 响应体的 ``detail`` 字段。
    """
    result = safe_error(
        ConnectionError(f"could not connect to redis://alice:{_SECRET}@redis:6379/0"),
    )
    assert _SECRET not in result, result
    assert "redis:6379" in result, "主机信息要留下，否则这条 detail 没有排查价值"
