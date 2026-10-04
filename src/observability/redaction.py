# -*- coding: utf-8 -*-
"""把字符串里可能出现的**连接串凭据**抹掉。

═══ 为什么要单独一个模块 ═══

这两个函数原本写在 :mod:`src.server.probes` 里（那里是最早、也是最需要
它们的调用点）。到了 P4，向量库的探针也要用它们 —— 而
``src/knowledge/`` 依赖 ``src/server/`` 是**反的**：知识层不该知道有一个
HTTP 服务层存在。所以把它们挪到一个谁都可以依赖的中立位置。

═══ ⚠️ 为什么这件事必须做 ═══

探针响应体是**最容易被整段贴进工单、聊天窗口、AI 对话**的东西 ——
它短、它看起来人畜无害、它是「出问题时大家第一个去拿的东西」。
框架或驱动的异常文案里是否带凭据不由我们决定，但我们可以保证
**经过本模块的任何字符串都不带**。

⚠️ 注意这里只处理 ``scheme://user:password@`` 这一种形态。
它覆盖了 PostgreSQL / Redis / Milvus 的连接串（本项目里唯一会带凭据的地方），
但**不**覆盖「密钥被单独打印」的情况 —— 那个靠约定
（只打印 ``api_key_configured`` 这类布尔值），不靠本模块兜底。
"""

from __future__ import annotations

import re

#: 匹配 ``scheme://user:password@`` 这一段。
#:
#: 拆解：``://`` + 一段不含空白的字符 + ``:`` + 一段不含空白的字符 + ``@``。
#: 刻意**不**匹配没有密码的 ``scheme://user@`` —— 那种情况下
#: 用户名本身不构成凭据，抹掉反而会丢失「连的是哪个用户」这条有用的信息。
#:
#: ═══ ⚠️ 为什么字符类比「看起来该有的」宽得多 ═══
#:
#: 这里踩过一个坑，是审计时用真实字符串试出来的：早先的写法把密码档写成
#: ``[^/\s@]+``（"密码里不该有 ``/`` 和 ``@``"）。可这个类**一旦匹配不上就整段不脱敏**，
#: 而它的两个例外恰恰都是真实存在的形态::
#:
#:     postgresql+asyncpg://u:pa/ss@db:5432/aligo   →  原样返回（一个字符都没抹）
#:     postgresql+asyncpg://u:pa@ss@db:5432/aligo   →  只剩 "ss@db"，密码后半段仍在
#:
#: 前者的成因是回溯失败：正则要求 ``:`` 与 ``@`` 之间不含 ``/``，而那里有个 ``/``，
#: 于是整条不匹配 —— **脱敏器静默地不脱敏**，比没有脱敏器更危险（它让人以为已经兜住了）。
#: 后者是提前闭合：第一处 ``@`` 就被当成了 userinfo 的终点。
#:
#: 所以现在宽到「只排除空白与 ``?`` ``#``」：``/`` 的语义在 URL 里本就是路径分隔符，
#: 而一个**原始**（未百分号编码）的 ``/`` 出现在这里，说明这个字符串压根不是合法 URL，
#: 更该整个抹掉。排掉 ``?`` ``#`` 是为了不跨查询串/锚点去误伤普通 URL。
#: 两个字符类都是**贪婪**的，因此匹配会延伸到该 token 里**最后**一个 ``@`` ——
#: 这正是 ``u:pa@ss@db`` 能整个被吃掉的原因。
#:
#: ⚠️ 宁可**多抹**，不可少抹：抹掉一段无害文本只是难看，漏掉一段密码是事故。
_CREDENTIALS_RE = re.compile(r"://[^\s?#]*:[^\s?#]*@")


def redact(text: str) -> str:
    """抹掉文本里形如 ``scheme://user:password@`` 的凭据部分。

    Args:
        text (`str`): 原始文本（通常来自异常）。

    Returns:
        `str`: 已脱敏的文本。
    """
    return _CREDENTIALS_RE.sub("://***:***@", text)


def safe_error(exc: BaseException) -> str:
    """把异常压成一条**安全且可读**的单行说明。

    只取异常类型名与消息首行：完整 traceback 会包含文件路径与调用栈，
    对「哪个依赖挂了」这个问题毫无帮助，却会让响应体膨胀、日志刷屏。

    Args:
        exc (`BaseException`): 捕获到的异常。

    Returns:
        `str`: 形如 ``OperationalError: connection refused`` 的说明。
    """
    first_line = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    text = (
        f"{type(exc).__name__}: {first_line}"
        if first_line
        else type(exc).__name__
    )
    return redact(text)


__all__ = ["redact", "safe_error"]
