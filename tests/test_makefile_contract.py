# -*- coding: utf-8 -*-
"""Makefile 契约测试：`make help` 与 Makefile 的真实内容必须互相自洽。

==============================================================================
为什么这件事值得单独一组用例
==============================================================================
    ``help`` 是 ``make`` 的**默认目标**（``.DEFAULT_GOAL := help``），
    因此它是使用者接触这个项目的**唯一入口**。一个没被列进 help 的目标，
    对使用者而言等于不存在 —— 哪怕它写得好好的。

    本项目已经真实踩过一次：``help`` 里漏列了 **7 个**真实存在的目标。
    漏列不会报任何错，``make help`` 照样以退出码 0 结束、输出漂漂亮亮。

    所以这里断言的不是"help 好看"，而是三条**双向**的一致性：

        真实目标  ⊆  help 输出      每个目标都在入口里被介绍
        help 输出 ⊆  真实目标      入口不承诺不存在的东西（会误导人去找）
        .PHONY   == 真实目标集合    见 :func:`test_phony_matches_real_targets`

==============================================================================
本组用例**只做静态解析**，唯一一次外部调用是 `make help`
==============================================================================
    「真实目标」是从 Makefile 文本里解析出来的，而不是从 ``make help``
    反推 —— 若用后者当基准，就变成了"help 与自己比"，永远通过。
    基准必须独立于被测物。

    只有 :func:`test_help_runs` 与 :func:`test_help_covers_every_real_target`
    真的调用 ``make``（因为「输出里有没有这句」这件事只有跑一遍才知道）。
    没有 ``make`` 的机器上这两条会 skip 并说明原因 —— 其余用例仍然生效。
"""

from __future__ import annotations

import ast
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

# ==============================================================================
# 解析工具
# ==============================================================================

#: 匹配**列 0 处**的目标定义，例如 ``up: check-env preflight``。
#:
#: 三段说明：
#:   1. ``^`` 不加 ``re.M`` 之外的东西，且**不允许前置空白** ——
#:      这一条就把配方行（以 Tab 缩进）和 ``.PHONY`` 的续行（以空格缩进）
#:      全部排除掉了，不必再单独过滤。
#:   2. 名字以字母/数字/下划线开头，故 ``.PHONY`` ``.DEFAULT_GOAL``
#:      这类特殊目标不会被当成普通目标。
#:   3. ``(?!=)`` 排除 ``NAME := value`` 这种变量赋值。
#:      只写 ``:`` 的话，``PYTHON := python`` 会被认成一个叫 PYTHON 的目标。
TARGET_RE = re.compile(r"^([A-Za-z0-9_][A-Za-z0-9_.-]*):(?!=)", re.MULTILINE)

#: ``.PHONY: a b c`` 的声明（含用 ``\`` 续行的多行写法）。
PHONY_RE = re.compile(r"^\.PHONY:(.*?)(?=^[^\s#]|\Z)", re.MULTILINE | re.DOTALL)

#: help 输出里对目标的提及，例如 ``make up-obs`` / ``make seed_data``。
#:
#: ⚠️ 字符类里**必须**含下划线：本项目确有 ``milvus_init`` ``seed_data`` 这类
#: 带下划线的目标名，而早期版本只允许 ``[a-z0-9-]``，于是 ``make seed_data``
#: 会被截成 ``seed`` —— 一个并不存在的目标名，让
#: :func:`test_help_does_not_promise_missing_targets` 冤枉一个合法的 help 行。
#: 目标名允许字母/数字/下划线/点/连字符（见 :data:`TARGET_RE`），本正则与它对齐。
HELP_MENTION_RE = re.compile(r"\bmake\s+([a-z][a-z0-9_-]*)")

#: ``ALL_PROFILES := --profile core --profile observability ...``
ALL_PROFILES_RE = re.compile(r"^ALL_PROFILES\s*:?=\s*(.*)$", re.MULTILINE)
PROFILE_FLAG_RE = re.compile(r"--profile\s+(\S+)")


def _makefile_text(repo_dir: Path) -> str:
    """读取 Makefile 原文。

    Args:
        repo_dir (`Path`): 仓库根目录。

    Returns:
        `str`: Makefile 内容。
    """
    return (repo_dir / "Makefile").read_text(encoding="utf-8")


def _real_targets(text: str) -> set[str]:
    """从 Makefile 文本里解析出**真实定义**的目标名。

    Args:
        text (`str`): Makefile 内容。

    Returns:
        `set[str]`: 目标名集合。
    """
    return set(TARGET_RE.findall(text))


def _phony_targets(text: str) -> set[str]:
    """解析 ``.PHONY`` 声明的目标名（支持 ``\\`` 续行）。

    Args:
        text (`str`): Makefile 内容。

    Returns:
        `set[str]`: 声明的伪目标名集合。
    """
    names: set[str] = set()
    for body in PHONY_RE.findall(text):
        # 续行符与注释都要去掉：`a b \` + 换行 + `c` 是一份声明。
        body = body.replace("\\\n", " ")
        body = body.split("#", 1)[0]
        names.update(body.split())
    return names


def _help_section(text: str) -> str:
    """取出 ``help`` 目标的配方（也就是 `make help` 会 echo 的那些行）。

    Args:
        text (`str`): Makefile 内容。

    Returns:
        `str`: help 配方原文。找不到 help 目标时为空串。
    """
    lines = text.splitlines()
    collected: list[str] = []
    inside = False
    for line in lines:
        if re.match(r"^help:", line):
            inside = True
            continue
        if inside:
            # 配方行以 Tab 开头；遇到下一个列 0 的非空行即结束。
            if line.startswith("\t") or not line.strip():
                collected.append(line)
            else:
                break
    return "\n".join(collected)


@pytest.fixture(scope="module")
def makefile(repo_dir: Path) -> str:
    """Makefile 原文（模块级缓存，避免每条用例重复读盘）。

    Args:
        repo_dir (`Path`): 仓库根目录。

    Returns:
        `str`: Makefile 内容。
    """
    return _makefile_text(repo_dir)


# ==============================================================================
# 目标集合的一致性
# ==============================================================================
def test_makefile_has_targets(makefile: str) -> None:
    """解析器本身要能工作 —— 一个都没解析出来说明正则坏了，不是"没有目标"。"""
    targets = _real_targets(makefile)
    assert "help" in targets, "连 help 目标都没解析出来，说明 TARGET_RE 已经与 Makefile 的写法脱节"
    assert len(targets) >= 15, f"只解析出 {len(targets)} 个目标，明显少于实际，正则可能失效"


def test_phony_matches_real_targets(makefile: str) -> None:
    """``.PHONY`` 与真实目标集合必须**互相**包含。

    两个方向各自对应一种真实故障：

    · ``真实目标 ⊄ .PHONY``
        该目标与同名文件（或目录）撞名时，make 会因为「文件已存在且比依赖新」
        而**跳过**它。症状是 `make <目标>` 什么都不做、也不报错 ——
        本项目里 ``test`` ``build`` ``clean`` 这些名字都有当目录的潜力。

    · ``.PHONY ⊄ 真实目标``
        声明了一个不存在的伪目标。看着无害，但它会让
        「目标被误删」这件事**看起来仍然正常** —— 因为 .PHONY 里还有它。
    """
    real = _real_targets(makefile)
    phony = _phony_targets(makefile)

    assert real - phony == set(), (
        f"这些目标没有声明为 .PHONY，撞上同名文件时会被静默跳过：{sorted(real - phony)}"
    )
    assert phony - real == set(), (
        f"这些名字声明了 .PHONY 但并没有对应的目标（目标被删了？）：{sorted(phony - real)}"
    )


def test_help_is_default_goal(makefile: str) -> None:
    """``help`` 必须是默认目标 —— 本组用例的整个前提都建立在这一点上。

    若默认目标被改成别的（比如 ``up``），那么敲一个裸 ``make`` 就会去拉
    10 GB 镜像建容器；而「help 是唯一入口」这个论断也就不再成立，
    本文件里那些"必须列进 help"的断言会随之变成过度约束。
    """
    assert re.search(r"^\.DEFAULT_GOAL\s*:?=\s*help\s*$", makefile, re.MULTILINE), (
        ".DEFAULT_GOAL 不是 help —— 请同时修订本文件的 docstring（它假定 help 是默认目标）"
    )


# ==============================================================================
# help 输出与真实目标的对应关系
# ==============================================================================
def test_help_section_lists_every_real_target(makefile: str) -> None:
    """静态检查：每个真实目标的名字都在 help 配方里出现过（``help`` 自身除外）。

    与 :func:`test_help_covers_every_real_target` 的区别：这条**不需要 make**，
    因此哪怕在没装 make 的环境里也能守住这个契约。
    """
    recipe = _help_section(makefile)
    assert recipe, "没找到 help 目标的配方 —— help 目标是不是被改名了？"

    missing = sorted(name for name in _real_targets(makefile) if name != "help" and name not in recipe)
    assert not missing, (
        f"这些目标存在于 Makefile 但 help 一句都没提：{missing}\n"
        f"help 是 `make` 的默认动作，也就是使用者的唯一入口 —— 漏列等于它们不存在。"
    )


def test_help_does_not_promise_missing_targets(makefile: str) -> None:
    """反向：help 里提到的每个 ``make <目标>`` 都必须真实存在。

    这条防的是另一种错误方向 —— help 里留着一个**已经被删掉**的目标名，
    使用者照着敲会得到 ``No rule to make target``。
    """
    real = _real_targets(makefile)
    mentioned = set(HELP_MENTION_RE.findall(_help_section(makefile)))

    # help 里会出现 `make up WAIT_TIMEOUT=600` 这种带参数的写法，目标名已由正则切掉。
    assert mentioned, "help 配方里一个 `make xxx` 都没提到，正则可能失效"
    unknown = sorted(mentioned - real)
    assert not unknown, f"help 里提到的这些目标并不存在，照敲会报 No rule to make target：{unknown}"


@pytest.mark.skipif(shutil.which("make") is None, reason="本机没有 make，无法验证 help 的真实输出")
def test_help_covers_every_real_target(repo_dir: Path) -> None:
    """真正跑一遍 ``make help``，断言输出里逐个列出全部真实目标。

    为什么还要真跑一次（而不是只做上面的静态检查）：静态检查只能证明
    "那个名字出现在了配方文本里"。若配方被 ``set -e`` 短路、被条件分支跳过、
    或 ``@echo`` 被误删，静态检查照样通过，而 ``make help`` 实际什么都不打印。
    契约的对象是**输出**，所以必须看输出。
    """
    text = _makefile_text(repo_dir)
    result = subprocess.run(
        ["make", "help"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, f"make help 退出码 {result.returncode}：\n{result.stderr}"
    output = result.stdout

    missing = sorted(name for name in _real_targets(text) if name != "help" and name not in output)
    assert not missing, f"`make help` 的输出里没有这些目标：{missing}\n\n实际输出：\n{output}"


# ==============================================================================
# 档位名与 compose 的一致性
# ==============================================================================
def test_all_profiles_matches_compose(makefile: str, repo_dir: Path) -> None:
    """``ALL_PROFILES`` 里的档名必须与 docker-compose.yaml 的 ``profiles:`` 集合**完全一致**。

    Makefile 里对这条有明确警告（"写错一个字母不会报任何错，只会让 down/clean
    静默漏掉一批容器与卷"），并给出了手工校验方法。这里把它变成自动的。

    为什么值得自动化：``down`` / ``clean`` / ``pull`` 都靠 ``$(ALL_PROFILES)``
    决定"看得见哪些服务"，而这三条命令的失败方式是**静默的** ——
    命令照样退出 0、照样打印「✅ 已清理」，实际上漏掉的那批容器与数据卷
    原封不动。这种失败只在下次启动看到旧数据时才被发现。
    """
    match = ALL_PROFILES_RE.search(makefile)
    assert match, "Makefile 里找不到 ALL_PROFILES 的定义"
    declared = set(PROFILE_FLAG_RE.findall(match.group(1)))
    assert declared, f"ALL_PROFILES 里没解析出任何 --profile：{match.group(1)!r}"

    compose_text = (repo_dir / "docker-compose.yaml").read_text(encoding="utf-8")
    document = yaml.safe_load(compose_text)
    used: set[str] = set()
    for service in (document.get("services") or {}).values():
        used.update(service.get("profiles") or [])

    assert declared == used, (
        "ALL_PROFILES 与 compose 的 profiles 集合不一致：\n"
        f"  只在 Makefile 里：{sorted(declared - used)}\n"
        f"  只在 compose 里：{sorted(used - declared)}\n"
        "漏掉一个档位不会报错，只会让 down/clean 静默漏掉那批容器与数据卷。"
    )


def test_every_compose_service_has_a_profile(repo_dir: Path) -> None:
    """每个 compose 服务都必须至少属于一个档位。

    ⚠️ 没有 ``profiles:`` 的服务是**默认档**，任何带 ``--profile X`` 的
    命令都会**包含**它。于是 ``ALL_PROFILES`` 的语义会从「全部档位」
    悄悄变成「全部档位 + 若干不在任何档里的服务」，
    上面那条一致性断言也会随之失去意义。
    本项目当前 13 个服务全部显式挂档 —— 这条用例把该约定钉住。
    """
    document = yaml.safe_load((repo_dir / "docker-compose.yaml").read_text(encoding="utf-8"))
    without = sorted(
        name
        for name, service in (document.get("services") or {}).items()
        if not service.get("profiles")
    )
    assert not without, (
        f"这些服务没有 profiles:（会落进默认档，使 ALL_PROFILES 的语义失效）：{without}"
    )


# ==============================================================================
# check-docs：两道闸门都要真的跑
# ==============================================================================
def test_check_docs_runs_both_gates(makefile: str) -> None:
    """``make check-docs`` 必须同时调用两个脚本。

    这条断言的存在理由：两个脚本管的是**互不相干**的两类问题
    （行号超界 / 计数漂移）。只调用其中一个，另一类问题会**完全没有闸门**，
    而 ``make check-docs`` 照样绿 —— 一个比没有闸门更危险的假绿灯，
    因为它会让人以为两类都查过了。
    """
    match = re.search(r"^check-docs:(.*?)(?=^[^\s#]|\Z)", makefile, re.MULTILINE | re.DOTALL)
    assert match, "Makefile 里找不到 check-docs 目标"
    body = match.group(1)

    for script in ("scripts/check_doc_refs.py", "scripts/check_doc_counts.py"):
        assert script in body, f"check-docs 没有调用 {script}"


def test_doc_check_scripts_exist(repo_dir: Path) -> None:
    """两个校验脚本必须真的在仓库里。

    上一条只证明 Makefile *提到了*它们。文件被删/改名时，
    ``make check-docs`` 会以 ``can't open file`` 失败 —— 那是好事，
    但失败信息说的是"找不到文件"，与"文档有问题"看起来完全不同。
    这条让 pytest 直接给出前者，省一次误判。
    """
    for script in ("scripts/check_doc_refs.py", "scripts/check_doc_counts.py"):
        assert (repo_dir / script).is_file(), f"{script} 不存在 —— make check-docs 会失败"


# ==============================================================================
# make 的另一个入口：面向运维的**提示文本**里的命令行
# ==============================================================================
#: ``make logs <服务名>`` 这种写法（**不带** ``SVC=``）。
#:
#: ``logs`` 目标取的是 ``$(or $(SVC),app)``，即服务名由变量传入；位置参数
#: 不是被忽略，而是被 make 当成**第二个目标**。仓库里没有名为 ``app``
#: （或任何服务名）的目标，于是 ``make logs app`` 的结果是
#: ``No rule to make target 'app'`` —— 一行都不打日志。
_LOG_WITH_SERVICE_RE = re.compile(r"\bmake logs\s+(?!SVC=)[A-Za-z0-9_.-]+")

#: 会被扫的目录。``docs/`` 不在其中：文档里写错命令行只是文档问题，
#: 而这里的每一条都是**报错时才会被读到**的字符串 —— 出错时按提示敲，
#: 再吃一句 make 报错，是把人往错误方向推。
_HINT_SCAN_DIRS = ("src", "scripts")


def _hint_strings(path: Path) -> list[tuple[int, str]]:
    """取出一个 ``.py`` 里所有字符串字面量（含 f-string 的常量片段）。

    Args:
        path (`Path`): 目标文件。

    Returns:
        `list[tuple[int, str]]`: ``(行号, 字面量)`` 列表。

    Notes:
        ⚠️ 这里走 **AST** 而不是整文件正则：解释"为什么不能写
        ``make logs app`` 的那段注释本身就含有这个字符串，整文件正则会
        把解释性注释当违规 —— 一条只能靠删掉注释来通过的用例。
        注释不出现在 AST 的字符串节点里，这个洞自然被堵上。
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.append((node.lineno, node.value))
    return found


def test_no_hint_tells_the_operator_to_run_make_logs_with_a_service(
    repo_dir: Path,
) -> None:
    """★ 提示文本里不得出现 ``make logs <服务名>``。

    这条是**真实修过的 bug** 的回归用例：``scripts/smoke.py``、
    ``src/llm/degradation.py``、``src/server/middleware/mock_credential.py``
    里有 5 处提示写的是 ``make logs app``，而 ``make logs app`` 在 make 眼里
    是"同时构建 logs 和 app 两个目标"，项目里并没有 ``app`` 目标 ⇒
    运维拿到的是一句 ``No rule to make target 'app'``，而不是日志。

    ⚠️ 正确写法是 ``make logs``（默认就是 app）或 ``make logs SVC=<服务名>``。
    本用例只拦前者之外的错法，不要求所有提示都必须提 logs。
    """
    offenders: list[str] = []
    for directory in _HINT_SCAN_DIRS:
        for path in sorted((repo_dir / directory).rglob("*.py")):
            for lineno, text in _hint_strings(path):
                for match in _LOG_WITH_SERVICE_RE.finditer(text):
                    rel = path.relative_to(repo_dir)
                    offenders.append(f"{rel}:{lineno} → {match.group(0)!r}")

    assert not offenders, (
        "以下提示把服务名当成了位置参数 —— 照着敲会得到 "
        "`No rule to make target`，一条日志都看不到：\n  "
        + "\n  ".join(offenders)
        + "\n改成 `make logs`（默认 app）或 `make logs SVC=<服务名>`。"
    )


__all__: list[str] = []
