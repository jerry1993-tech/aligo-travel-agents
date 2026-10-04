#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文档计数校验：**写死的数字 == 机器实测值**。

==============================================================================
它为什么存在
==============================================================================
    本项目的注释里散落着大量**计数断言**，例如::

        Makefile           「13 个服务里只有 2 个失败」
        Makefile           「docker-compose.yaml 里共 13 个 `image:` 行」
        .env.example       「把 9 个 Docker Hub 镜像写成 ${ALIGO__IMAGE_REGISTRY:-}…」
        docker-compose.yaml「则上面 9 个镜像全部变成 docker.m.daocloud.io/...」

    这些数字**不会随代码漂移而报错**。往 compose 里加一个服务，
    「共 13 个 image: 行」就变成了假话 —— 而它读起来仍然像一个认真的结论，
    甚至比不加这句更糟：读者会**据此推断**范围（"只有 13 个，我全看过了"）。

    所以本脚本把「数字」与「实测」接上电：数字变了，闸门就红。

==============================================================================
它与 check_doc_refs.py 的分工（互不越界）
==============================================================================
    check_doc_refs.py    查「文件在不在 + 行号超没超界」，**不看数字**。
    本脚本               查「数字对不对」，**不看行号**。

    刻意不合并：两者的触发时机不同。改源码会同时影响两者，但**改文档措辞**
    只影响计数，而**换 pin 的第三方版本**只影响行号。分开之后，一次失败
    就能直接指出是哪一类问题。

==============================================================================
设计：只有一份真值
==============================================================================
    一个常见的坏做法是在脚本里再写一遍数字::

        assert len(services) == 13   # ← 数字被抄了第二遍

    这样脚本自己就成了第二份真值：改了 compose 忘了改这里，闸门**依然绿**。
    本脚本不这样写 —— 数字**只从被校验的文本里读**（:class:`Site` 的正则捕获组），
    再与 :func:`_measure_*` 的实测值比对。脚本里没有 13、没有 9、没有 28。

==============================================================================
「断言消失了」也要报错
==============================================================================
    如果一个锚点在某文件里**一次都匹配不上**，本脚本报**失败**，而不是跳过。
    理由：把一句带数字的注释整段删掉，是让闸门变绿最省事的办法 ——
    而它同时也把那句注释**提供的保障**一起删掉了。
    （例外：声明为 :attr:`Claim.optional` 的断言允许缺席，
    用于「这个数字要等后续阶段才写进文档」的计数。）

==============================================================================
用法
==============================================================================
        python scripts/check_doc_counts.py            # 或 make check-docs
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

# ==============================================================================
# 配置
# ==============================================================================

#: 仓库根目录（本文件位于 scripts/ 下）。
REPO_ROOT = Path(__file__).resolve().parent.parent

#: docker-compose.yaml 的路径（本脚本的多数实测值来自它）。
COMPOSE_PATH = REPO_ROOT / "docker-compose.yaml"

#: 已安装的 AgentScope 包根（框架自 2026-10-04 起由 pip 包提供，不再 vendored）。
def _agentscope_src() -> Path | None:
    """用 ``agentscope.__file__`` 定位包根；未安装时为 ``None``。

    刻意不硬编码 ``site-packages`` 路径 —— 换 Python 版本 / 虚拟环境后
    硬编码路径会静默指向不存在的文件，这里则解析到"当前解释器真正会 import 的那份"。
    """
    try:
        import agentscope
    except ImportError:
        return None
    return Path(agentscope.__file__).resolve().parent


AGENTSCOPE_SRC = _agentscope_src()

#: 全仓库扫描时跳过的目录。
#:
#: ⚠️ ``docs`` 里的 ``博客原文-*.md`` 是**抓取来的第三方文本**，里面的数字
#:    （"50% 提升到 90%"之类）是阿里自己的口径，不是本项目的断言，
#:    更不由本项目维护。把它们卷进来只会产生一堆无法修复的报错。
#: ``third_party``：仓库历史上 vendored 过框架源码（2026-10-04 起改为 pip 包安装，
#:    该目录已移除）；保留这条是为了将来若再引入任何本地源码树时，
#:    把上游文字里的数字挡在扫描集之外。
SKIP_DIRS = frozenset(
    {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
     ".venv", "venv", "node_modules", "dist", "build", ".idea", ".vscode",
     ".claude", "third_party", "docs"},
)

#: 全仓库扫描时纳入的文件后缀（以及少数无后缀文件）。
SCAN_SUFFIXES = frozenset({".md", ".py", ".yaml", ".yml", ".sh", ".toml", ".ini"})
SCAN_FILENAMES = frozenset({"Makefile", "Dockerfile", ".env.example", ".dockerignore"})

#: 本脚本自身不参与扫描 —— 它的常量与 docstring 里有大量**示例数字**
#: （上面的 "13" "9" "28"），那些不是断言。
SKIP_FILES = frozenset({"scripts/check_doc_counts.py", "scripts/check_doc_refs.py"})


# ==============================================================================
# 数据模型
# ==============================================================================
@dataclass(frozen=True)
class Site:
    """一条断言**写在哪儿**，以及怎么从那一行里把数字抠出来。

    Attributes:
        path (`str | None`): 相对仓库根的文件路径；``None`` = 全仓库扫描。
        pattern (`re.Pattern[str]`): 锚点正则。**必须**用捕获组包住那个数字。
        group (`int`): 取第几个捕获组。默认 1。
            一条断言可以同时含两个不同的数（例如「core 6 个、其余 6 个」），
            这时用两个 :class:`Site` 指向同一个 pattern 的不同组。
        every (`bool`): 该文件里**所有**匹配都必须等于实测值。
            用于同一句话被重述多次的情形 —— 那正是最容易只改一处的场景。
        note (`str`): 这条锚点指向什么，出错时打印。
    """

    path: str | None
    pattern: re.Pattern[str]
    group: int = 1
    every: bool = False
    note: str = ""


@dataclass(frozen=True)
class Claim:
    """一个计数断言：一个实测函数 + 若干处写它的地方。

    Attributes:
        name (`str`): 人类可读的名字。
        unit (`str`): 单位（「个」「种」），只用于打印。
        measure (`Callable[[], int | None]`): 实测函数。
            返回 ``None`` 表示**本机测不了**（例如 agentscope 未安装），
            此时整条断言被记为「跳过」并打印原因 —— 而不是失败。
            刻意不用 0 表示"测不了"：0 是一个合法的计数结果，
            把它与"读不到"混为一谈会制造假失败。
        sites (`tuple[Site, ...]`): 断言写在哪些地方。
        optional (`bool`): True = 允许一处都找不到（断言尚未写进文档）。
    """

    name: str
    unit: str
    measure: Callable[[], int | None]
    sites: tuple[Site, ...]
    optional: bool = False
    skip_reason: str = ""


# ==============================================================================
# 实测：docker-compose.yaml
# ==============================================================================
def _load_compose() -> dict | None:
    """读入 docker-compose.yaml。

    ⚠️ 刻意用 ``safe_load`` 而不是 ``compose config``：后者要拉起 Docker
    守护进程才能跑，而本脚本被 ``make check-docs`` 调用 —— 那个目标必须能在
    **没有 Docker 的 CI 容器里**跑通（与 ``make test`` 同一条原则）。
    这里只需要静态 YAML，静态解析足够且更快。

    Returns:
        `dict | None`: compose 文档；读不到或不是合法 YAML 时为 ``None``。
    """
    try:
        text = COMPOSE_PATH.read_text(encoding="utf-8")
    except OSError:
        return None

    # ⚠️ 用 safe_load_all 而不是 safe_load：万一文件将来变成多文档
    #    （`---` 分隔），safe_load 会直接抛错，而我们要的是第一个文档。
    try:
        documents = list(yaml.safe_load_all(text))
    except yaml.YAMLError:
        return None
    for document in documents:
        if isinstance(document, dict) and "services" in document:
            return document
    return None


def _services() -> dict | None:
    """取 compose 的 services 段。"""
    document = _load_compose()
    if document is None:
        return None
    services = document.get("services")
    return services if isinstance(services, dict) else None


def _measure_total_services() -> int | None:
    """compose 里定义的服务总数（含全部 profile，含仅本地构建的 app）。

    Returns:
        `int | None`: 服务数；读不到 compose 时为 ``None``。
    """
    services = _services()
    return None if services is None else len(services)


def _measure_services_with_any_profile(profiles: frozenset[str], *, exclude: str = "") -> int | None:
    """统计「profiles 与给定集合有交集」的服务数。

    ⚠️ 这里有三个**不同**的数，注释里极易混用，所以口径必须写死在调用点上::

        core 档                    6   {core}
        core 之外、仍属默认全栈     6   {observability, tracing}
        默认全栈（非 optional）     12   {core, observability, tracing}

    Makefile 的 down-core 注释里那句话同时写了前两个数
    （"这 **6 个**，其余 6 个"），所以这两个断言指向**同一行**的两个捕获组 ——
    这也是 :class:`Site.group` 存在的原因。

    Args:
        profiles (`frozenset[str]`): 目标档位集合。
        exclude (`str`): 非空时，排除属于该档的服务。
            用于表达「core 之外、但仍属默认全栈的那 6 个」这种**相对**计数：
            它的补集是相对于 core 说的，不能写成"所有含 observability/tracing 的"
            （那样也能得出 6，但含义不同 —— 一旦将来有服务同时挂 core 与 tracing，
            两个写法就会分叉）。

    Returns:
        `int | None`: 服务数；读不到 compose 时为 ``None``。
    """
    services = _services()
    if services is None:
        return None

    counted = 0
    for service in services.values():
        service_profiles = set(service.get("profiles") or [])
        if not service_profiles & profiles:
            continue
        if exclude and exclude in service_profiles:
            continue
        counted += 1
    return counted


def _image_entries() -> list[tuple[str, bool]] | None:
    """列出每个服务的镜像：``(镜像规格, 是否本地构建)``。

    ⚠️ **不去重、不筛选**，这是有意的：本项目的注释里同时存在三个口径，
    它们各自需要不同的加工，把加工下沉到调用点比挤进一个布尔参数表清楚得多::

        13  compose 里的 `image:` **行**数            = len(entries)
        11  可拉取的去重规格数                        = 去掉 built 再去重
         9  受 ALIGO__IMAGE_REGISTRY 影响的去重规格数   = 只留含前缀的再去重

    「13 行 → 12 个可拉取 → 11 个去重规格」这条链子在 Makefile 的 pull
    目标注释里被完整写出来了，三个数都在那里，所以三个都得对得上 ——
    只对其中一个（"反正数量差不多"）会让那串推理变成噪音。

    Returns:
        `list[tuple[str, bool]] | None`: 每服务一条；读不到 compose 时为 ``None``。
    """
    services = _services()
    if services is None:
        return None

    entries: list[tuple[str, bool]] = []
    for service in services.values():
        image = service.get("image")
        if not isinstance(image, str) or not image:
            continue
        entries.append((image, "build" in service))
    return entries


def _measure_image_lines() -> int | None:
    """compose 里 ``image:`` 的**行数**（同一个规格出现两次算两行）。

    Returns:
        `int | None`: 行数；读不到 compose 时为 ``None``。
    """
    entries = _image_entries()
    return None if entries is None else len(entries)


def _measure_pullable_distinct_images() -> int | None:
    """需要从 registry 拉取的**去重规格**数（排除本地构建的 app）。

    Returns:
        `int | None`: 规格数；读不到 compose 时为 ``None``。
    """
    entries = _image_entries()
    if entries is None:
        return None
    return len({spec for spec, built in entries if not built})


def _measure_registry_prefixed_distinct_images() -> int | None:
    """含 ``${ALIGO__IMAGE_REGISTRY`` 的**去重规格**数。

    ⚠️ 按**规格**去重而非按服务计数：``redis:7-alpine`` 被 ``redis`` 与
    ``langfuse-redis`` 两个服务引用，填了前缀之后它只变成**一个**镜像名，
    所以是 9 而不是 10。这个 1 之差正是注释里"9 个镜像"与"10 个服务"
    看起来互相矛盾的原因。

    Returns:
        `int | None`: 规格数；读不到 compose 时为 ``None``。
    """
    entries = _image_entries()
    if entries is None:
        return None
    return len({spec for spec, _ in entries if "ALIGO__IMAGE_REGISTRY" in spec})


# ==============================================================================
# 实测：已安装的 AgentScope
# ==============================================================================
def _read_agentscope(relative: str) -> str | None:
    """读已安装 ``agentscope`` 包里的一个文件。

    Args:
        relative (`str`): 相对 ``agentscope`` 包根（``site-packages/agentscope``）的路径。

    Returns:
        `str | None`: 文件内容；未安装或文件不存在时为 ``None``。
    """
    if AGENTSCOPE_SRC is None:
        return None
    try:
        return (AGENTSCOPE_SRC / relative).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _measure_event_types() -> int | None:
    """``agentscope.event.EventType`` 的枚举成员数（实测 = 28）。

    ⚠️ 这个数**必须实测而不是照抄**：本项目前期据博客文档写过「27 种」，
    而本地安装的 agentscope==2.0.9 里是 28 种。差的那个是 ``CUSTOM`` ——
    一个"兜底事件类型"，看博客时代的文档根本不会知道它存在。
    前端按 27 种写 switch 时，第 28 种会**静默走进 default 分支**。

    Returns:
        `int | None`: 成员数；读不到源码时为 ``None``。
    """
    text = _read_agentscope("event/_event.py")
    if text is None:
        return None

    # 只取 class EventType(...) 这一个类体，避免把同文件里别的枚举算进来。
    match = re.search(r"^class EventType\b.*?(?=^class |\Z)", text, re.S | re.M)
    if match is None:
        return None

    # 枚举成员 = 类体里缩进 4 空格、名字全大写、跟着 `=` 的行。
    # 用 `{4}` 而不是 `\s+`：嵌套类/函数体内的赋值缩进更深，不该被算成成员。
    return len(re.findall(r"^ {4}([A-Z][A-Z0-9_]*)\s*=", match.group(0), re.M))


def _measure_middleware_hooks() -> int | None:
    """``MiddlewareBase`` 上可覆写的钩子数（实测 = 7）。

    口径：``middleware/_base.py`` 里以 ``on_`` 开头的 ``async def``。
    刻意**不**把 ``list_tools`` 与 ``get_middleware_key`` 算进来 ——
    它们是普通方法，不是「每次执行到某个阶段就会被调用」的钩子。
    把口径写成「所有 async def」会得到 9，那是错的数。

    Returns:
        `int | None`: 钩子数；读不到源码时为 ``None``。
    """
    text = _read_agentscope("middleware/_base.py")
    if text is None:
        return None
    return len(re.findall(r"^    async def on_\w+\(", text, re.M))


def _measure_collected_tests() -> int | None:
    """pytest 实际**收集到**的用例数。

    ⚠️ 为什么要真的跑一次 pytest 而不是数 ``def test_``：
    静态计数**数不对**。``@pytest.mark.parametrize`` 会让一个函数变成 N 个用例，
    ``skipif`` 收集时仍然计入，而 fixture 参数化又会让同一个函数再翻倍。
    静态数出来会比实际少，而"少一点"恰好是那种看起来合理、于是没人核对的错。

    代价：这一步要起一个 pytest 进程（本机实测约 7 秒），是 ``make check-docs``
    里最慢的一环。刻意接受 —— 因为**「N 个用例」正是 Makefile 的 check-docs
    注释点名要守的那一类数字**（"加一个测试用例就静默变错"），
    而它没有比"真跑一次"更可靠的取数方式。

    Returns:
        `int | None`: 收集到的用例数；pytest 不可用或输出无法解析时为 ``None``。
    """
    import subprocess

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    # pytest -q --collect-only 的最后一行形如 `90 tests collected in 6.79s`。
    # ⚠️ 数字可能是 `1 test`（单数），故两种写法都要接。
    match = re.search(r"^(\d+) tests? collected", result.stdout, re.MULTILINE)
    if match is None:
        return None
    return int(match.group(1))


def _measure_env_example_aligo_keys() -> int | None:
    """``.env.example`` 里**生效的** ``ALIGO__`` 键个数（写这段时实测 = 40）。

    「生效」的判据是**行首**就是 ``ALIGO__`` 且带 ``=``。被 ``#`` 注释掉的示例
    （例如已停用的 ``# LLM_MODEL=…``）不算 —— 它们不会进入容器环境，
    因此不是模板的一部分，也不受本断言约束。

    ⚠️ 这个数为什么值得实测，而不是"估一个"：README 用它来说明
    「这些键必须与 schema 一一对应」，而**每加一个键它就会变** ——
    本项目切 DashScope 时加了 ``ALIGO__LLM__MODEL``（24 → 25），
    补检索护栏时又加了 ``ALIGO__MILVUS__SEARCH_TIMEOUT_SECONDS``（39 → 40）。
    数字漂掉不会让任何测试变红，只会让读者拿着一份错的清单去逐条核对。
    ⚠️ 上面那个「实测 = 40」只是**写这段文字时的快照**：真正的真值由本函数
    现算，任何一处 docstring 里的数字都不参与比对（比对的只有文档正文里的数）。

    ⚠️ 与用例 ``tests/test_config_loader.py::
    test_env_example_keys_are_all_accepted_by_the_loader`` **分工不同、互不替代**：
    那条守的是「每个键都合法」（会挡住启动即崩），本条守的是
    「文档里的数字与实物一致」。只有前者会让一个**错的数字**长期留在文档里；
    只有后者会让一个**非法的键**留在模板里。

    Returns:
        `int | None`: 键个数；文件读不到时为 ``None``。
    """
    try:
        text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return len(re.findall(r"^ALIGO__[A-Z0-9_]+=", text, re.MULTILINE))


# ==============================================================================
# 断言表
# ==============================================================================
#: down-core 注释里那句话 —— 两个捕获组分别是「core 6 个」与「其余 6 个」。
#: 抽成常量是因为它与下面的 `execution order` 注释必须**逐字一致**：
#: 两处指向不同的捕获组，正则一旦分叉就会有一组永远匹配不上。
_DOWN_CORE_LINE = re.compile(r"这 \*\*(\d+) 个\*\*，其余 (\d+) 个")

CLAIMS: tuple[Claim, ...] = (
    Claim(
        name="compose 中定义的服务总数",
        unit="个服务",
        measure=_measure_total_services,
        sites=(
            Site(
                "Makefile",
                re.compile(r"(\d+) 个服务里\*\*只有"),
                note="down 目标注释：逐个服务试 `docker compose up -d` 的那些断言",
            ),
            Site(
                "README.md",
                re.compile(r"compose 里共 \*\*(\d+) 个服务\*\*"),
                note="README「端口与档位」段",
            ),
        ),
    ),
    Claim(
        name="compose 中 core 档的服务数",
        unit="个服务",
        measure=lambda: _measure_services_with_any_profile(frozenset({"core"})),
        sites=(
            Site(
                "Makefile",
                _DOWN_CORE_LINE,
                group=1,
                note="down-core 目标注释：`--profile core stop` 会停掉的那几个",
            ),
            Site(
                "README.md",
                re.compile(r"^\| `core` \| (\d+) \|"),
                note="README「端口与档位」的档位表",
            ),
        ),
    ),
    Claim(
        name="core 之外、仍属默认全栈的服务数",
        unit="个服务",
        measure=lambda: _measure_services_with_any_profile(
            frozenset({"observability", "tracing"}),
            exclude="core",
        ),
        sites=(
            Site(
                "Makefile",
                _DOWN_CORE_LINE,
                group=2,
                note="同上那一句的后半：「其余 6 个（clickhouse、langfuse…）」",
            ),
        ),
    ),
    Claim(
        name="compose 中默认全栈（非 optional 档）的服务数",
        unit="个服务",
        measure=lambda: _measure_services_with_any_profile(
            frozenset({"core", "observability", "tracing"}),
        ),
        sites=(
            Site(
                "Makefile",
                re.compile(r"(\d+) 个容器(?:全在跑| \+ 网络|，2026-09-24 实测)"),
                every=True,
                note="down 相关注释里反复出现的那句「12 个容器全在跑」",
            ),
            Site(
                "README.md",
                re.compile(r"默认全栈（core \+ tracing \+ observability）= \*\*(\d+) 个\*\*"),
                note="README「端口与档位」的档位表下方",
            ),
        ),
    ),
    Claim(
        name="compose 中 tracing 档的服务数",
        unit="个服务",
        measure=lambda: _measure_services_with_any_profile(frozenset({"tracing"})),
        sites=(
            Site(
                "README.md",
                re.compile(r"^\| `tracing` \| (\d+) \|"),
                note="README「端口与档位」的档位表",
            ),
        ),
    ),
    Claim(
        name="compose 中 observability 档的服务数",
        unit="个服务",
        measure=lambda: _measure_services_with_any_profile(frozenset({"observability"})),
        sites=(
            Site(
                "README.md",
                re.compile(r"^\| `observability` \| (\d+) \|"),
                note="README「端口与档位」的档位表",
            ),
        ),
    ),
    Claim(
        name="compose 中 optional 档的服务数",
        unit="个服务",
        measure=lambda: _measure_services_with_any_profile(frozenset({"optional"})),
        sites=(
            Site(
                "README.md",
                re.compile(r"^\| `optional` \| (\d+) \|"),
                note="README「端口与档位」的档位表",
            ),
        ),
    ),
    Claim(
        name="compose 里的 image: 行数",
        unit="行",
        measure=_measure_image_lines,
        sites=(
            Site(
                "Makefile",
                re.compile(r"里共 (\d+) 个 `image:` 行"),
                note="pull 目标注释：算「去重后有几个镜像」的起点",
            ),
        ),
    ),
    Claim(
        name="compose 中可拉取的去重镜像规格数",
        unit="个镜像",
        measure=_measure_pullable_distinct_images,
        sites=(
            Site(
                "Makefile",
                re.compile(r"（是 (\d+) 个：docker-compose"),
                note="pull 目标注释：`--ignore-buildable` 之后真正要拉的数量",
            ),
        ),
    ),
    Claim(
        name="受 ALIGO__IMAGE_REGISTRY 影响的 Docker Hub 镜像规格数",
        unit="个镜像",
        measure=_measure_registry_prefixed_distinct_images,
        sites=(
            Site(
                ".env.example",
                re.compile(r"把 (\d+) 个 Docker Hub 镜像写成"),
                note="IMAGE_REGISTRY 段的开头断言",
            ),
            Site(
                ".env.example",
                re.compile(r"填前缀 → 这 (\d+) 个\*\*整体\*\*改走该站"),
                note="IMAGE_REGISTRY 段：填了前缀之后受影响的镜像数",
            ),
            Site(
                "docker-compose.yaml",
                re.compile(r"则上面 (\d+) 个镜像全部变成"),
                note="compose 顶部 D8 段：镜像前缀的说明",
            ),
            Site(
                "README.md",
                re.compile(r"作用范围与 (\d+) 个受影响镜像的完整说明"),
                note="README「配置体系」的镜像前缀小节",
            ),
        ),
    ),
    Claim(
        name="agentscope.event.EventType 的成员数",
        unit="种",
        measure=_measure_event_types,
        sites=(
            Site(
                None,
                # ⚠️ 中间那段 ``\**`` 是必需的：README 里写的是 ``**28** 种 `EventType```
                #    （markdown 加粗），而源码注释里通常是裸的 ``28 种 EventType``。
                #    不吸收加粗标记的话，README 里的那处断言会**匹配不上** ——
                #    而它恰恰是最该被守住的一处（前端按错的种数写 switch 会静默漏事件）。
                re.compile(r"(\d+)\**\s*种\s*\**\s*`?EventType`?"),
                every=True,
                note="全仓库范围：任何地方写「N 种 EventType」都要等于实测值",
            ),
        ),
    ),
    Claim(
        name="agentscope.middleware.MiddlewareBase 的钩子数",
        unit="个",
        measure=_measure_middleware_hooks,
        sites=(
            Site(
                None,
                re.compile(r"(\d+) 个\s*(?:middleware\s*)?hook\b"),
                every=True,
                note="全仓库范围：任何地方写「N 个 hook」都要等于实测值",
            ),
        ),
    ),
    Claim(
        name="pytest 实际收集到的用例数",
        unit="个用例",
        measure=_measure_collected_tests,
        sites=(
            Site(
                None,
                re.compile(r"\*\*(\d+) 个用例"),
                every=True,
                note="全仓库范围：README 等处的「N 个用例」",
            ),
        ),
    ),
    Claim(
        name=".env.example 里生效的 ALIGO__ 键数",
        unit="个键",
        measure=_measure_env_example_aligo_keys,
        sites=(
            Site(
                "README.md",
                # ⚠️「个」与 `ALIGO__` 之间允许夹一个「生效的」：README 的**标题**
                #    写的是「25 个 `ALIGO__` 键」，而**正文**那句为了讲清
                #    "注释掉的不算"写成了「**25** 个生效的 `ALIGO__` 键」。
                #    只锚一种写法的话，另一种就会悄悄漂掉 —— 而同一句话的
                #    两种写法不一致时，读者不知道哪个才是真的。
                re.compile(r"(\d+)\**\s*个(?:生效的)?\s*\**\s*`?ALIGO__"),
                every=True,
                note="README「N 个 ALIGO__ 键」一节：标题与正文各一处",
            ),
        ),
    ),
)


# ==============================================================================
# 扫描
# ==============================================================================
@dataclass
class Finding:
    """一条不通过的断言。

    Attributes:
        claim (`str`): 断言名。
        where (`str`): 写在哪儿（``文件:行号``），这是要修的地方。
        text (`str`): 那一行的原文（去掉首尾空白），便于定位。
        claimed (`int`): 文档里写的数字。
        measured (`int`): 机器实测值。
    """

    claim: str
    where: str
    text: str
    claimed: int
    measured: int


@dataclass
class Outcome:
    """校验汇总。

    Attributes:
        findings (`list[Finding]`): 不通过的断言。
        checked (`int`): 通过校验的**数字处数**（不是断言条数）。
        skipped (`list[str]`): 被跳过的断言及原因。
        measured (`dict[str, int]`): 断言名 → 实测值，用于成功后打印。
    """

    findings: list[Finding] = field(default_factory=list)
    checked: int = 0
    skipped: list[str] = field(default_factory=list)
    measured: dict[str, int] = field(default_factory=dict)


def _repo_files() -> list[Path]:
    """收集参与全仓库扫描的文件（相对仓库根）。

    Returns:
        `list[Path]`: 待扫描文件。
    """
    found: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        if relative.as_posix() in SKIP_FILES:
            continue
        if path.suffix in SCAN_SUFFIXES or path.name in SCAN_FILENAMES:
            found.append(Path(relative.as_posix()))
    return found


def _scan_site(site: Site, repo_files: list[Path]) -> list[tuple[str, int, str]]:
    """在一个 Site 上找出全部匹配。

    Args:
        site (`Site`): 锚点定义。
        repo_files (`list[Path]`): 全仓库文件索引（``site.path is None`` 时用）。

    Returns:
        `list[tuple[str, int, str]]`: ``(文件, 行号, 那一行原文)`` 三元组。
        ⚠️ 这里刻意**只匹配单行**（``finditer`` 逐行跑，不做 DOTALL）——
        锚点写的是「一句话里紧挨着的数字」，跨行匹配会把不相关的两段拼起来。
    """
    targets = [Path(site.path)] if site.path else repo_files
    found: list[tuple[str, int, str]] = []

    for target in targets:
        try:
            text = (REPO_ROOT / target).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if site.pattern.search(line):
                found.append((target.as_posix(), lineno, line.strip()))
    return found


def _extract(site: Site, line: str) -> int | None:
    """从一行里抠出 :class:`Site` 指定的那个数字。

    Args:
        site (`Site`): 锚点定义（决定用哪个捕获组）。
        line (`str`): 那一行的原文。

    Returns:
        `int | None`: 数字；正则不匹配或那一组没参与匹配时为 ``None``。
    """
    match = site.pattern.search(line)
    if match is None:
        return None
    try:
        raw = match.group(site.group)
    except IndexError:
        return None
    return None if raw is None else int(raw)


def check_claims() -> Outcome:
    """执行全部断言校验。

    Returns:
        `Outcome`: 校验汇总。
    """
    outcome = Outcome()
    repo_files = _repo_files()

    for claim in CLAIMS:
        measured = claim.measure()
        if measured is None:
            outcome.skipped.append(f"{claim.name}：本机读不到实测来源（agentscope 未安装或 compose 缺失）")
            continue

        # ⚠️ 逐个 Site 检查，**不是**把所有 Site 合起来看有没有匹配。
        #    第一版是合起来看的，于是「某个 Site 的锚点已经失效、但兄弟 Site 还在」
        #    这种情况会**静默通过** —— 实测踩到了：`.env.example` 里那条
        #    "填前缀 → 这 9 个**整体**改走该站" 因为正则里多写了一个反引号而匹配不上，
        #    而同一断言在 compose 里的锚点仍然有效，闸门就没响。
        #    那正是本脚本最该抓的失效模式：**断言悄悄少了一条**。
        any_site_matched = False
        for site in claim.sites:
            matches = _scan_site(site, repo_files)

            if not matches:
                if claim.optional:
                    continue
                any_site_matched = True  # 下面会为它报一条 Finding
                outcome.findings.append(
                    Finding(
                        claim=claim.name,
                        where=f"{site.path or '（全仓库）'}  ← 锚点失效",
                        text=site.note or site.pattern.pattern,
                        claimed=-1,
                        measured=measured,
                    ),
                )
                continue

            any_site_matched = True
            for path, lineno, line in matches:
                claimed = _extract(site, line)
                if claimed is None:
                    # 匹配上了却抠不出数字（多组正则的某一组未参与匹配）—— 不判失败，
                    # 因为另一条 Site 会用同一个正则的另一个组来覆盖这个数。
                    continue
                if claimed != measured:
                    outcome.findings.append(
                        Finding(
                            claim=claim.name,
                            where=f"{path}:{lineno}",
                            text=line,
                            claimed=claimed,
                            measured=measured,
                        ),
                    )
                else:
                    outcome.checked += 1

        if not any_site_matched and claim.optional:
            outcome.skipped.append(f"{claim.name}：{claim.skip_reason or '尚未写入任何文档'}")
            continue

        outcome.measured[claim.name] = measured

    return outcome


def main() -> int:
    """脚本入口。

    Returns:
        `int`: 0 = 全部通过；1 = 有漂移。
    """
    print(f"扫描目录：{REPO_ROOT}")
    outcome = check_claims()

    if outcome.findings:
        print(f"❌ 发现 {len(outcome.findings)} 处计数漂移：")
        print()
        for finding in outcome.findings:
            print(f"  · {finding.claim}")
            print(f"      写在：{finding.where}")
            print(f"      原文：{finding.text}")
            if finding.claimed < 0:
                print(f"      实测：{finding.measured}  ← 这句断言已经不在任何文件里了")
            else:
                print(f"      写着 {finding.claimed}，实测 {finding.measured}")
            print()

    if outcome.skipped:
        print("⏭️  跳过的断言：")
        for reason in outcome.skipped:
            print(f"  · {reason}")
        print()

    if outcome.findings:
        print("修法：把文档里的数字改成实测值（上表已给出），")
        print("      ⚠️ 不要靠删掉那句注释来让本检查通过 —— 那会连同它提供的保障一起删掉。")
        return 1

    print(f"✅ 全部 {outcome.checked} 处计数与实测一致：")
    print()
    for name, value in outcome.measured.items():
        print(f"  · {value:>3}  {name}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
