# -*- coding: utf-8 -*-
"""前后端**契约测试**：TypeScript 侧里写死的名字与字段，必须与 Python 侧对得上。

为什么要这个文件
================

前端有两处**明明写着「必须与后端逐字一致」的约定，却没有任何东西守着**：

1. ``web/frontend/src/components/chat/tool-renderers/aligo/parse.ts`` 的
   ``CARD_*`` 常量（注释原话：「必须与后端 ``src/tools/_result.py`` 里的
   ``CARD_*`` 常量**逐字**一致」）；
2. ``.../aligo/index.tsx`` 的 ``ALIGO_TOOLS`` 工具名表（注释原话：
   「必须与后端 ``src/tools/`` 里 ``FunctionTool`` 生成的注册名**逐字**
   一致」）。

两处都自己写明了后果：「**静默的** —— 不报错、不告警，只是卡片没了 /
这个工具永远走默认渲染」。一份只写在注释里的约定等于没有约定：
2026-10-03 的对抗审计就是这么发现 ``policy_verdict`` 卡片逻辑与后端
岔开的（查标准的负载没有 ``compliant``，前端把它判成「不符合」）。

本文件把那两条注释变成可执行断言。

测试怎么读 TS 源码
==================

刻意**不引入 node / tsc**：CI 与容器里跑 pytest 时未必有前端工具链，
而为了几条字符串比对去装一整套 Node 依赖不划算。

做法是按行正则抠出 ``export const CARD_XXX = '...'`` 与
``{ name: '...', i18n: ... }`` 这两种**固定写法**。这比解析 TS 脆弱，
所以每条断言都先钉住「抠出来了 N 条」—— 抠法一旦失效（前端换了写法），
用例会以「只解析出 0 条」变红，而不是静默地全绿。**这一点是这个文件
能不能当守卫的关键**：没有数量下限的解析式断言，在解析失败时会全部通过。
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

import pytest

from src.config import load_settings
from src.llm.mock import MockChatModel
from src.server.agents_factory import build_repositories, build_tools_factory
from src.tools._result import (
    CARD_APPROVAL,
    CARD_HOTEL,
    CARD_ORDERS,
    CARD_POLICY,
    CARD_ROUTE,
    CARD_TRANSPORT,
)
from src.tools.expert import INTENT_TOOL_NAME
from src.tools.route import ROUTE_TOOL_NAME
from tests.conftest import TEST_ENVIRON

#: 前端工程根（相对仓库根）。
_FRONTEND = Path("web/frontend/src/components/chat/tool-renderers/aligo")
_PARSE_TS = _FRONTEND / "parse.ts"
_INDEX_TSX = _FRONTEND / "index.tsx"
_POLICY_TSX = _FRONTEND / "PolicyVerdictCard.tsx"
_SHELL_TSX = _FRONTEND / "shell.tsx"
_ROUTE_TSX = _FRONTEND / "RouteDecisionCard.tsx"

#: 差旅卡片渲染器所在的目录 —— 「排障字段不得进 UI」那条用例要扫全目录。
_ALIGO_DIR = _FRONTEND

#: ``export const CARD_ROUTE = 'route_decision';``
_CARD_CONST_RE = re.compile(
    r"^export const (CARD_[A-Z_]+)\s*=\s*'([^']*)'\s*;",
    re.MULTILINE,
)

#: ``{ name: 'search_transport', i18n: 'searchTransport', ... }``
_TOOL_ENTRY_RE = re.compile(
    r"\{\s*name:\s*'([^']+)'\s*,\s*i18n:\s*'([^']+)'",
)


def _strip_comments(source: str) -> str:
    """去掉 TS/TSX 源码里的注释，只留下会被执行的代码。

    Args:
        source (`str`): 源文件内容。

    Returns:
        `str`: 抹掉注释后的内容（行结构保留，方便读报错信息）。

    ⚠️ 为什么需要它：本文件有两条用例是**扫源码里有没有某个取值表达式**
    （``data.detail`` / ``record.next_step``）。而「为什么不渲染这个字段」
    的解释恰好也要写出字段名 —— 于是注释会把用例自己弄红，作者的第一反应
    会是「把注释删了」，而那正好毁掉最该留下的东西。抹注释比改注释对。

    ⚠️ 刻意做成**够用就好**的实现，不追求词法级正确：已核实本目录的写法是
    ``/** ... */`` 块注释 + 整行 ``//`` 注释，没有行尾 ``//`` 注释、
    没有正则字面量、没有模板串里带 ``/*``。真遇到这些写法时它会多抹或少抹，
    但**两条用例的判据是「有没有 ``x.detail`` 这样的取值」** ——
    少抹只会误报（人一眼能看出是注释），多抹只会漏报（但那要求代码写在
    注释后面）。写一个完整 TS 词法分析器换不来这点精度。
    """
    without_block = re.sub(r"/\*.*?\*/", "", source, flags=re.DOTALL)
    return "\n".join(
        line for line in without_block.splitlines() if not line.lstrip().startswith("//")
    )


def _read(path: Path) -> str:
    """读一个前端源文件。

    Args:
        path (`Path`): 相对仓库根的路径。

    Returns:
        `str`: 文件内容。

    ⚠️ 找不到就 ``pytest.fail`` 而不是抛 ``FileNotFoundError``：这条用例
    在纯后端环境里也应该给出**说得清**的结论（「前端源码不见了，契约
    无从校验」），而不是一个像是环境坏了的 traceback。
    """
    if not path.exists():
        import pytest

        pytest.fail(
            f"前端源文件不存在：{path} —— 契约测试无从校验。"
            "若前端被移出仓库，本文件应一并删除，而不是让它静默跳过。"
        )
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def registered_tools() -> dict[str, Any]:
    """跑一次**生产装配路径**，返回 ``工具名 -> 工具``。

    Returns:
        `dict[str, Any]`: 注册名到 ``FunctionTool`` 的映射。

    ⚠️ 走 :func:`src.server.agents_factory.build_tools_factory`，而不是
    本文件自己 ``build_business_tools(...)`` 拼一遍。理由不是「省事」：
    手工拼的那份**少了意图识别工具**（它是服务层作为 ``extra`` 追加的），
    于是「前端登记了 7 个、后端只有 6 个」这条差异会被那条断言本身
    掩盖成「前端多登记了一个」—— 而真相是装配入口不同。

    ⚠️ 用 ``MockChatModel``：装配只需要一个 ``ChatModelBase`` 实例
    （它被意图识别器捕获，构造期**不发起任何调用**）。用真实模型会让
    这条纯字符串校验依赖凭据与网络。

    ⚠️ module 作用域：装配一次给两条用例共用。``settings`` 夹具是
    function 作用域的，两者不兼容，所以这里自己 ``load_settings`` 一次
    （与 ``tests/test_evaluation_dataset.py`` 的同名夹具同一做法）。
    """
    definition = load_settings("test", environ=TEST_ENVIRON, dotenv=False)
    factory = build_tools_factory(
        settings=definition,
        repositories=build_repositories(),
        model=MockChatModel(),
    )
    tools = asyncio.run(factory("contract-test-user", "agent-1", "session-1"))
    return {tool.name: tool for tool in tools}


def cards_from_typescript() -> dict[str, str]:
    """从 ``parse.ts`` 抠出 ``CARD_*`` 常量。

    Returns:
        `dict[str, str]`: 常量名 → 字面量值。
    """
    found = dict(_CARD_CONST_RE.findall(_read(_PARSE_TS)))
    # ⚠️ 下限是 5 而不是 6，这是**故意的**：真值就是 6 个常量，写成 6
    # 会让「其中一个常量被改名」这种缺陷在**下限**上就变红，下面那条
    # 「常量名对不上」的精确断言永远轮不到 —— 用例红了，但报的是
    # 「正则失效」，把排查方向带偏。下限只负责「正则还工作吗」，
    # 少一个多一个交给集合比对。
    assert len(found) >= 5, (
        f"只从 parse.ts 解析出 {len(found)} 个 CARD_* 常量（预期 ≥5）—— "
        "前端的写法变了，正则失效。请同步更新本文件的正则，"
        "否则下面的断言会因为「没解析到」而全部通过。"
    )
    return found


def test_card_constants_match_the_backend() -> None:
    """★ 前端 ``CARD_*`` 必须与后端 ``CARD_*`` **逐字**一致。

    对不上的后果是静默降级：后端照常返回 ``card``，前端查不到渲染器就
    退回一段裸 JSON，不报错、不告警。用户看到的是「卡片没了」。

    ⚠️ 双向比对，不是单向包含：少一个（后端加了卡片前端没加）与
    多一个（前端留了已删除的卡片）都是缺陷。
    """
    backend = {
        "CARD_ROUTE": CARD_ROUTE,
        "CARD_TRANSPORT": CARD_TRANSPORT,
        "CARD_HOTEL": CARD_HOTEL,
        "CARD_POLICY": CARD_POLICY,
        "CARD_ORDERS": CARD_ORDERS,
        "CARD_APPROVAL": CARD_APPROVAL,
    }
    frontend = cards_from_typescript()

    assert set(frontend) == set(backend), (
        f"卡片常量名对不上：前端多 {sorted(set(frontend) - set(backend))}，"
        f"后端多 {sorted(set(backend) - set(frontend))}"
    )
    mismatched = {
        name: (frontend[name], backend[name])
        for name in backend
        if frontend[name] != backend[name]
    }
    assert not mismatched, f"卡片常量的**值**对不上（名字 → 前端/后端）：{mismatched}"


def test_every_card_constant_has_a_renderer() -> None:
    """★ 每个 ``CARD_*`` 都要在 ``CARD_VIEWS`` 里有对应组件。

    有这个常量、却没有登记组件，等于这张卡片永远走默认渲染 ——
    与「常量写错」是同一个症状（卡片不出来），但更难发现：
    常量是对的，只是没人用它。
    """
    source = _read(_INDEX_TSX)
    registered = set(re.findall(r"\[(CARD_[A-Z_]+)\]:", source))

    # ⚠️ 下限 5 而不是 6：真值 6。写成 6 的话，「漏登记一个渲染组件」
    # 正好是这个用例**要抓的缺陷**，却会先在**下限**上变红 ——
    # 报的是「正则失效」，与真实原因无关。下限只负责「正则还工作吗」。
    assert len(registered) >= 5, (
        f"只从 index.tsx 解析出 {len(registered)} 个 CARD_VIEWS 条目 —— "
        "写法变了，正则失效。"
    )
    missing = set(cards_from_typescript()) - registered
    assert not missing, f"这些卡片常量没有登记渲染组件：{sorted(missing)}"


def test_frontend_tool_names_match_the_registered_tools(
    registered_tools: dict[str, Any],
) -> None:
    """★ ``ALIGO_TOOLS`` 里的工具名必须与后端真实注册的工具名一致。

    ``FunctionTool`` 默认用**函数名**当注册名，前端这份表是手写的。
    拼错一个字不会报错，只是这个工具**永远**走默认渲染（裸 JSON）。

    ⚠️ 双向比对。少一个（后端加了工具前端没登记）与多一个（前端留了
    拼错或已删除的名字）是**两种**缺陷，症状一样但修法相反。
    """
    registered = set(registered_tools)

    # ⚠️ 先钉住「后端真的产出了工具」：装配失败返回空列表时，
    # 下面的断言会因为空集而**通过** —— 用例从守卫变成摆设。
    # 下限取 6（真值 7），单个工具消失时让下面那条**点名**的断言去报。
    assert len(registered) >= 6, f"后端只装配出 {sorted(registered)} —— 装配路径变了。"

    frontend = {name for name, _ in _TOOL_ENTRY_RE.findall(_read(_INDEX_TSX))}
    # ⚠️ 同上：真值 7，下限取 6。写成 7 会让「一个工具名拼错」先撞下限。
    assert len(frontend) >= 6, (
        f"只从 index.tsx 解析出 {len(frontend)} 个工具条目 —— 写法变了，正则失效。"
    )

    # ⚠️ ``recognize_intent`` 的名字取自 ``src/tools/expert.py`` 的
    # ``INTENT_TOOL_NAME`` 常量，而不是在这里手抄一遍 —— 手抄的话，
    # 那个常量改名了这条豁免照样悄悄放行，豁免就从「已知的例外」
    # 退化成「白名单」。
    assert INTENT_TOOL_NAME in registered, (
        f"意图识别工具 {INTENT_TOOL_NAME!r} 不在装配结果里 —— "
        "前端登记了它，这条比对才有意义。"
    )
    assert INTENT_TOOL_NAME in frontend, (
        f"前端没有登记 {INTENT_TOOL_NAME!r} —— 意图识别的结果会以裸 JSON 展示。"
    )

    missing_in_frontend = registered - frontend
    stale_in_frontend = frontend - registered

    assert not missing_in_frontend, (
        f"后端注册了 {sorted(missing_in_frontend)}，前端没有登记渲染器 —— "
        "这些工具的结果会以裸 JSON 的形式给用户看。"
    )
    assert not stale_in_frontend, (
        f"前端登记了 {sorted(stale_in_frontend)}，后端没有同名工具 —— "
        "这份表里有拼错或已删除的名字，对应的渲染器永远不会被用到。"
    )


def test_policy_card_reads_the_fields_the_tool_actually_writes(
    registered_tools: dict[str, Any],
) -> None:
    """★★ ``PolicyVerdictCard`` 读的字段必须是 ``check_travel_policy`` 真写的。

    ⚠️ 这条守的是 2026-10-03 那个真实缺陷的形状：**字段语义**在前端被
    用错（查标准的负载没有 ``compliant``，前端把 ``undefined`` 判成
    false，于是渲染出一个红色「不符合」徽标）。名字对得上、类型也在，
    错的只是「这个字段有没有值」的假设 —— 纯比对名字抓不住它，
    所以这里**跑一次真实的工具**，拿真负载的键来比对。

    ⚠️ 两个分支都要取：查标准与核对走的是同一张卡片，字段集**不同**。
    只测核对分支的话，``lookup`` 这个判别字段漏掉也发现不了。
    """
    tool = registered_tools["check_travel_policy"]

    def payload(**kwargs: object) -> dict:
        """调用工具并解回 JSON 载荷。"""
        chunk = asyncio.run(tool.call(**kwargs))
        text = "".join(getattr(b, "text", "") for b in chunk.content)
        return json.loads(text)

    lookup_item = payload(kind="hotel")["items"][0]
    check_item = payload(kind="hotel", price=300)["items"][0]

    # ⚠️ 判别字段必须在**两个**分支里都存在，且值相反。缺一个的后果是
    # 卡片又退回「猜 compliant 有没有值」那条老路。
    assert lookup_item.get("lookup") is True, lookup_item
    assert check_item.get("lookup") is False, check_item

    card = _read(_POLICY_TSX)
    for field, sample in (
        ("lookup", lookup_item),
        ("compliant", check_item),
        ("max_cabin_text", lookup_item),
        ("policy_note", lookup_item),
    ):
        assert f"record.{field}" in card, (
            f"卡片没有读 {field} —— 要么前端漏了，要么字段改名了。"
        )
        assert field in sample, (
            f"工具返回的负载里没有 {field}，而卡片在读它：{sorted(sample)}"
        )


#: 载荷里**只给排障看、不许进 UI** 的字段。
#:
#: ⚠️ 这张表是白名单式的「点名」而不是「凡不在表里的都能渲染」：新增字段
#: 默认是可渲染的，这里只钉住**已经确认属于内部信息**的那几个。
_FORBIDDEN_IN_UI = ("detail", "next_step")


def test_diagnostic_fields_are_never_rendered() -> None:
    """★ 排障字段（``detail``）与模型指引（``next_step``）不得出现在卡片里。

    这些字段**确实**在载荷里（模型要读，前端类型里也声明了），所以它们
    一路进到浏览器是设计使然；能不能出现在**界面**上才是这条守的东西。

    ⚠️ 守的是 2026-10-03 审计发现的形状：``shell.tsx`` 用等宽字体把
    ``detail``（``TimeoutError: ...``）渲染给用户，``RouteDecisionCard.tsx``
    把 ``next_step``（「请先调用 check_travel_policy…」）渲染成「下一步」。
    两处都不是「字段算错了」，而是**把内部字段当成了展示字段** ——
    名字对、值也在，纯比对名字抓不住，所以这条直接扫源码。

    ⚠️ 扫描方式是**读 JSX 里的取值表达式**（``data.detail`` / ``record.next_step``）
    而不是搜字符串：注释里当然要提到这些字段名（解释为什么不渲染它们），
    搜字符串会把注释也算成违规。
    """
    offenders: list[str] = []
    for tsx in sorted(_ALIGO_DIR.glob("*.tsx")):
        source = _strip_comments(_read(tsx))
        for field in _FORBIDDEN_IN_UI:
            # data.detail / record.detail / item.detail —— 取值即渲染
            if re.search(rf"\b\w+\.{field}\b", source):
                offenders.append(f"{tsx.name} 读取了 {field}")
    assert not offenders, (
        f"这些文件把内部字段当展示字段用了：{offenders} —— "
        "detail 是异常原文，next_step 是写给模型的祈使句，都不该给用户看。"
    )


def test_route_payload_keeps_the_hint_out_of_items(
    registered_tools: dict[str, Any],
) -> None:
    """★★ ``aligo_route_intent`` 的 ``items`` 里不得有 ``next_step``。

    ⚠️ 这条不是「字段改名了」的守卫，而是**职责划分**的守卫：
    ``items`` 是卡片的数据源，里面的一切都会被渲染成用户可见的字段；
    ``_NEXT_STEP_HINTS`` 是给模型的第二人称祈使句（含工具名与 ``**`` 标记）。
    两者混在一起时，用户会在卡片上读到一句对他的助手说的指令。

    ⚠️ 同时钉住 ``instruction`` **必须**还在顶层：只删不加的话，模型就
    真的收不到指引了 —— 那是把「泄漏」修成「功能缺失」，比原来更糟。
    """
    tool = registered_tools[ROUTE_TOOL_NAME]
    chunk = asyncio.run(
        tool.call(intent="QUERY_POLICY", matched_rule="policy_kw", reason="含「差标」"),
    )
    text = "".join(getattr(b, "text", "") for b in chunk.content)
    payload = json.loads(text)

    assert payload["items"], f"路由工具没有产出 items：{payload}"
    item = payload["items"][0]
    assert "next_step" not in item, (
        f"items 里又出现了 next_step（{item.get('next_step')!r}）—— "
        "它是写给模型的指引，会被卡片渲染给用户。"
    )
    assert payload.get("instruction"), (
        "顶层 instruction 不见了 —— 删 next_step 时把模型的指引也删掉了。"
    )
    # ⚠️ summary 同样要干净：它是「给模型和用户看的自然语言」，实测模型会把
    # 它整段抄进正文。里面出现工具名，用户就会读到「请先调用 check_travel_policy」。
    assert "check_travel_policy" not in payload["summary"], (
        f"summary 里混进了工具名：{payload['summary']!r}"
    )


def test_detail_is_logged_not_just_returned(caplog: pytest.LogCaptureFixture) -> None:
    """★ ``error_chunk`` 的 ``detail`` 必须真的写日志。

    docstring 从第一天起就写着 detail「会进日志与 items」，而实现从来没
    写过日志 —— 于是唯一能读到它的地方是前端。**注释里的承诺不算承诺**，
    这条把它变成可执行断言：删掉 ``logger.warning`` 这行，用例就红。

    ⚠️ 用 ``caplog`` 而不是打桩 ``logger.warning``：打桩只能证明「有人调了」，
    caplog 连 ``%s`` 占位符有没有填对一起验了 —— 而占位符写错（把 detail
    漏掉）正是那种「调用发生了、信息却没落盘」的缺陷。
    """
    from src.tools._result import error_chunk

    with caplog.at_level("WARNING", logger="src.tools._result"):
        error_chunk("查询失败了，请稍后重试。", detail="TimeoutError: milvus timeout")

    assert any(
        "TimeoutError: milvus timeout" in rec.getMessage() for rec in caplog.records
    ), (
        "error_chunk 没有把 detail 写进日志 —— "
        f"捕获到的记录：{[r.getMessage() for r in caplog.records]}"
    )
