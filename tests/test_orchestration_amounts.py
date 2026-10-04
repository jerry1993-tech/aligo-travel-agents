# -*- coding: utf-8 -*-
"""数字共享模块（``src/orchestration/amounts.py``）的测试。

═══ 本文件守的是什么 ═══

这一个模块存在的**唯一理由**是「同一份数字规则被三处消费」：

    · 动态 Prompt 把预算渲染成给模型照抄的文字（:func:`render_amount`）；
    · 差旅工具把价格/差标渲染进工具返回（:func:`render_amount`）；
    · 回复守卫拿正文里的数去和工具返回、用户原话比对（切词 + 归一）。

所以本文件守的不是「这几个函数各自对不对」（那些在
``tests/test_orchestration_reply_guard.py`` 里已按真实缺陷逐条钉住），
而是三件**只有跨模块才成立**的事：

1. **渲染出来的数，闸门必须认**（:func:`test_rendered_amount_survives_the_gate_tokenizer`）。
   这是 2026-10-04 缺陷 P2 的内核：``f"{v:g}"`` 渲染出 ``1.2e+06``，
   而闸门从正文里抽到的是 ``1200000``，两边对不上 → 模型照抄反而被判编造。
2. **三处消费同一份实现**（:func:`test_all_consumers_share_one_implementation`）。
   只要有人在某个模块里手写一份 ``:g`` 或自建正则，用例立刻红 ——
   这条是「规则只能有一份」的机械保证。
3. **渲染是逐字还原，不是改写**（:func:`test_render_amount_never_rewrites_the_value`）。
   ``:g`` 的 6 位有效数字会在 1e6 附近静默改写数值本身。
"""

from __future__ import annotations

import pytest

from src.orchestration import amounts
from src.orchestration.amounts import (
    fold_width,
    normalize_amount,
    numbers_in,
    render_amount,
)

# ---------------------------------------------------------------------------
# 一、渲染：逐字还原，不改写、不切科学计数法
# ---------------------------------------------------------------------------
#: ``(输入, 期望渲染)``。**含 1e6 及以上的用例是有意的** ——
#: 那一档正是 ``:g`` 开始改写（``1200000.0 -> 1.2e+06``）的地方。
RENDER_CASES: list[tuple[float, str]] = [
    (0.0, "0"),
    (500.0, "500"),
    (15000.0, "15000"),
    (0.5, "0.5"),
    (0.005, "0.005"),
    (12345.678, "12345.678"),
    (999999.5, "999999.5"),
    (1200000.0, "1200000"),
    (1234567.0, "1234567"),
    (85000.5, "85000.5"),
]


@pytest.mark.parametrize(("value", "expected"), RENDER_CASES)
def test_render_amount_never_rewrites_the_value(value: float, expected: str) -> None:
    """渲染结果必须是对原值的**十进制逐字还原**。

    ⚠️ 反面样例写在下面那条断言里：``f"{value:g}"`` 在同一批输入上会给出
    ``1.2e+06`` / ``1e+06`` / ``12345.7`` 三种**不同性质的改写**
    （科学计数法、四舍五入、丢有效数字）。它们都会被模型「原样引用」，
    于是用户看到 ``你的预算上限是 1.2e+06 元`` —— 那是管道自己违反了
    它在提示词里对模型提出的数字纪律（缺陷 P2，2026-10-04 实测）。
    """
    rendered = render_amount(value)

    assert rendered == expected
    # ⚠️ 科学计数法是**最强的那条禁令**：它既不是原值、也不是用户能读的写法。
    assert "e" not in rendered.lower(), f"{value} 渲染成了科学计数法：{rendered}"
    # 整数值不该带 ".0" 尾巴（模型会照抄进用户可见的正文）。
    if rendered.endswith(".0"):
        raise AssertionError(f"{value} 渲染出了多余的 .0 尾巴：{rendered}")


def test_the_naive_format_would_have_failed() -> None:
    """把缺陷 P2 的现场钉在用例里（也说明这条测试为什么存在）。

    ⚠️ 这条**不是**在测 ``:g`` —— 它是在证明「替代方案确实是坏的」。
    若哪天有人把 ``render_amount`` 换回 ``f"{value:g}"``，上面那条参数化
    用例会红，但红的原因（6 位有效数字）在这一条里才写得清楚。
    """
    assert f"{1200000.0:g}" == "1.2e+06"
    assert f"{999999.5:g}" == "1e+06"
    assert f"{12345.678:g}" == "12345.7"

    assert render_amount(1200000.0) == "1200000"
    assert render_amount(999999.5) == "999999.5"
    assert render_amount(12345.678) == "12345.678"


@pytest.mark.parametrize(("value", "_expected"), RENDER_CASES)
def test_rendered_amount_survives_the_gate_tokenizer(
    value: float,
    _expected: str,
) -> None:
    """渲染出来的字符串，闸门必须抽出**同一个数**。

    ⚠️ 这是 P2 的判据本身，而不是它的表象。闸门做的事是「从正文里抽数字，
    再和来源文本里抽出的数字比」；两边抽出来的必须是同一个串。所以这条
    用例比对的是**渲染结果经闸门自己的切词归一之后**的形态，
    而不是渲染结果的字面。

    反例（修复前的实测）：来源侧渲染 ``1.2e+06``，抽出来是 ``{'1', '2', '06'}``
    （科学计数法里的 ``e``/``+`` 都是分隔符）；正文侧模型写 ``1200000``，
    抽出来是 ``{'1200000'}`` —— 交集为空，一段正确的答复被判成编造。
    """
    rendered = render_amount(value)
    # 模型照抄渲染结果写进正文（这就是提示词要求的「原样引用」）。
    answer_numbers = numbers_in(f"你的预算上限是 {rendered} 元。")
    # 来源侧（动态 Prompt / 工具返回）登记的也是这个渲染结果。
    source_numbers = numbers_in(rendered)

    assert source_numbers, f"{value} 渲染后一个数字都抽不出来：{rendered!r}"
    assert answer_numbers == source_numbers, (
        f"渲染 {rendered!r} 的正文数字 {answer_numbers} "
        f"与来源数字 {source_numbers} 对不上"
    )
    # 再收一道：模型可能把它写成带千分位的「1,200,000」，归一后仍须相等。
    grouped = f"{int(value):,}" if float(value).is_integer() else rendered
    assert numbers_in(grouped) == source_numbers


# ---------------------------------------------------------------------------
# 二、一份规则：三处消费必须指向同一个实现
# ---------------------------------------------------------------------------
def test_all_consumers_share_one_implementation() -> None:
    """三个消费方拿到的必须是**同一个函数对象**。

    ⚠️ 这条用例是「规则只能有一份」的机械保证。历史缺陷 P2 的成因就是
    渲染侧（动态 Prompt）与比对侧（回复守卫）各写各的：一边 ``:g``、
    一边正则归一。谁都可以在本地把测试改绿，但改不掉对象身份 ——
    只要有人在某个模块里抄了一份实现，这里立刻红。

    ⚠️ ``reply_guard`` 里的下划线别名同样要指向同一个对象：它保留私有名
    只是为了不让本模块的迁移在 diff 里变成「全文件重命名」，不是允许分叉。
    """
    from src.tools import travel
    from src.orchestration import prompt, reply_guard

    assert prompt.render_amount is amounts.render_amount
    assert travel.render_amount is amounts.render_amount

    assert reply_guard._fold_width is fold_width
    assert reply_guard._normalize_amount is normalize_amount
    assert reply_guard._numbers_in is numbers_in
    # reply_guard 的金额正则仍留在本地（它只服务上报与上限识别），
    # 但它的数字段必须与共享模式串**同源**：两者对 `\d[\d,]*(?:\.\d+)?`
    # 的写法一致，这条断言挡住「只有一处改了字符类」的半吊子修改。
    assert reply_guard._NUMBER_TOKEN_PATTERN.pattern == amounts.NUMBER_TOKEN_PATTERN.pattern


def test_prompt_module_still_avoids_the_framework() -> None:
    """``amounts`` 必须保持纯 stdlib —— 它是被纯模块 import 的。

    ⚠️ 这条不是洁癖：``src/orchestration/__init__.py`` 的纯度设计保证
    ``test_orchestration_prompt.py`` 里的子进程用例能断言「框架没被导入」。
    ``prompt.py`` 现在 import 了 ``amounts``，所以 ``amounts`` 一旦引入框架
    （或任何重依赖），那条更重要的保证会**从侧面**被破坏 ——
    而失败现场会指向 prompt，根因却在 amounts。
    """
    import subprocess
    import sys

    code = (
        "import sys; import src.orchestration.amounts; "
        "assert 'agentscope' not in sys.modules, 'amounts 把框架拉进来了'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


# ---------------------------------------------------------------------------
# 三、切词与归一的接口契约（实现细节的用例在 reply_guard 那边）
# ---------------------------------------------------------------------------
def test_fold_width_is_one_to_one_in_length() -> None:
    """折全角必须**逐字符 1:1**，调用方才能拿折后串的偏移去切窗口。

    ⚠️ 这条保证是 ``_mentions_a_limit`` 的前提：它用 ``match.start()`` 在
    折过的串上按偏移取窗口，一旦折的过程改变长度（NFKC 就会，例如
    ``㍿`` → ``株式会社``），窗口会飘到别的字符上，判据随机化。
    """
    text = "酒店差标是 １，２００ 元／晚。①600元"
    folded = fold_width(text)

    assert len(folded) == len(text)
    assert folded == "酒店差标是 1,200 元/晚。①600元"
    # 圈号**不受影响**（它不是全角形式区，NFKC 才会把它折成 1）。
    assert "①" in folded


def test_fold_width_does_not_touch_ascii() -> None:
    """纯 ASCII 输入必须原样返回（切词路径上的绝大多数输入）。"""
    ascii_text = "hotel limit: 600 CNY/night, 1200 total."

    assert fold_width(ascii_text) == ascii_text


def test_normalize_amount_is_idempotent() -> None:
    """归一必须幂等 —— 登记进 ``middle_context`` 的是归一后的串。

    ⚠️ 幂等性是「守卫把登记的数字再归一一次」这条兼容路径的前提
    （见 ``reply_guard._ungrounded_limit_claims`` 里对 ``prompt_text`` 的处理）：
    登记是**持久化**数据，可能是旧版本代码写下的，再过一遍归一必须不变。
    """
    for raw in ("1,200", "1200.0", "１，２００", "600", "999999.5", "0"):
        once = normalize_amount(raw)
        assert normalize_amount(once) == once, f"{raw!r} 归一两次结果不同"
