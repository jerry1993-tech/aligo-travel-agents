#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""文档行号引用校验：**文件在不在 + 行号超没超界**。

==============================================================================
它为什么存在
==============================================================================
    本项目的注释与文档里大量写着形如 ``src/server/probes.py:262`` 的**行号引用**。
    这类引用有一个恶毒的性质：**失效时看不出任何异常**。

    往某个被引用的文件里加一行注释，后面所有引用的行号就整体偏移 1 ——
    文字仍然通顺，链接仍然"看起来"有效，只有真正翻过去的人才会发现
    指向的是另一段代码。而那个人多半正在排障，他会先怀疑自己看错了文件。

    最坏的情况不是"引错了行"，而是**误以为已经看过了**那一段。

==============================================================================
它与 check_doc_counts.py 的分工（互不越界）
==============================================================================
    本脚本    查「文件在不在 + 行号超没超界」，**不看数字对不对**。
    另一个    查「数字对不对（== 机器实测值）」，**不看行号**。

    刻意不合并成一个脚本：两者的触发时机不同。改源码会同时影响两者，
    但改动文档措辞只影响计数，而**换 pin 的第三方版本**只影响指向已安装包
    （``site-packages``）的行号。分开之后，一次失败就能直接指出是哪一类问题。

==============================================================================
判定范围
==============================================================================
    扫描（到哪里找引用）：**自有文件** —— 仓库里的 md / py / yaml / ini 等。
        （第三方代码不扫：它们的注释不是我们写的，也不归我们维护。
        历史上 vendored 的 ``third_party/`` 已于 2026-10-04 移除 ——
        框架改为 pip 包安装，见 requirements.txt 第 零 节。）

    判定（引用指向谁）：**自有文件与已安装的第三方包都判**。
        项目里大量引用指向框架/依赖的源码行号（``agentscope/...``、
        ``starlette/routing.py:695`` 等），它们由第 3 级解析落到当前解释器的
        ``site-packages``。理由：**换一个 pin 的版本**就会让这些行号漂掉，
        而文档不会自己跟着变；"这条引用指向一个真实存在的行"这件事，
        与那个文件是谁写的无关。

==============================================================================
用法
==============================================================================
        python scripts/check_doc_refs.py            # 或 make check-docs
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ==============================================================================
# 配置
# ==============================================================================

#: 仓库根目录（本文件位于 scripts/ 下）。
REPO_ROOT = Path(__file__).resolve().parent.parent

#: 扫描哪些扩展名的文件（找引用）。
SCAN_SUFFIXES = frozenset(
    {".md", ".py", ".yaml", ".yml", ".ini", ".toml", ".sh"},
)

#: 没有扩展名、或扩展名不在 :data:`SCAN_SUFFIXES` 里，但同样要扫的文件（按文件名精确匹配）。
#:
#: ``requirements.txt`` 是**显式列进来**的，而不是把 ``.txt`` 加进 :data:`SCAN_SUFFIXES`：
#: 仓库里承载行号引用的 ``.txt`` 只有它一个 —— 它的注释要用
#: ``agentscope/.../_model.py:NN`` 这类引用去解释「某条依赖为什么这样钉」，
#: 而那些引用一旦写错，闸门就必须拦下来。不给 ``.txt`` 开后缀，是为了避免把将来可能
#: 出现的测试夹具文本一并拖进扫描集。
#:
#: 这条是 2026-10-01 补上的真实缺口：修 ``fastmcp`` 版本时在该文件里写了两条
#: ``pyproject.toml:NN`` 引用，而当时的扫描集不含本文件 —— 两条引用**从未被校验过**，
#: 闸门却照绿。补进扫描集后，它们才真正进入被检查范围。
SCAN_FILENAMES = frozenset(
    {"Makefile", "Dockerfile", ".env.example", ".dockerignore", "requirements.txt"}
)

#: 遍历时跳过的目录名（构建产物、缓存、虚拟环境）。
#:
#: ⚠️ 「不扫某个目录」与「不把它算进引用目标索引」是两件事，分别由
#:    :data:`SKIP_DIRS` 与 :data:`NO_SCAN_DIRS` 控制。历史教训：本脚本第一版
#:    把两者混在一起，让所有指向 vendored agentscope（当时在 ``third_party/``）
#:    的引用都报「找不到文件」，而那些报错看起来像是文档写错了 ——
#:    排查方向完全被带偏。如今框架是 pip 包，相关引用由第 3 级解析落到
#:    ``site-packages``；机制保留不变。
SKIP_DIRS = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        "build",
        ".idea",
        ".vscode",
        ".claude",
    },
)

#: **不扫描**（不从中找引用）的目录 —— 但仍在引用目标索引里。
#:
#: 历史条目：``third_party`` 是 2026-10-04 之前 vendored 框架源码的位置
#: （此后框架改为 pip 包安装，该目录已从仓库移除）。保留这条是防御性的 ——
#: 将来若再引入任何本地源码树，它的上游注释同样不该被当成
#: 本项目的引用来源参与扫描。
NO_SCAN_DIRS = frozenset({"third_party"})

#: 扫描时跳过的具体文件（相对仓库根的 POSIX 路径）。
#:
#: ⚠️ 本脚本与 check_doc_counts.py 必须在列：它们的代码与注释里会出现
#:    形如 ``foo.py:123`` 的**示例**与**正则片段**，那些不是真引用。
#:    不排除的话，本脚本会把自己的示例当成引用去校验，然后报出一个
#:    指向它自己的、无法修复的错误。
SKIP_FILES = frozenset(
    {
        "scripts/check_doc_refs.py",
        "scripts/check_doc_counts.py",
    },
)

#: 引用形如 ``path/to/file.py:123`` 或 ``path/to/file.py:123-456``。
#:
#: 三段说明：
#:   1. ``(?<![\w./-])`` —— 左边界。防的是「匹配到一个更长 token 的尾巴」，
#:      例如 URL ``https://x.com/a.py:5`` 里不该把 ``a.py:5`` 当成引用。
#:      因为前面的 ``/`` 与 ``.`` 都在排除集里，正则从 ``com`` 开始就匹配不上，
#:      于是整条 URL 被自然跳过 —— 不需要单独写一段 URL 识别逻辑。
#:   2. 扩展名**必须**列出来。没有它的话，``postgres:5432`` 这种端口写法、
#:      ``langfuse:3.225.10`` 这种镜像 tag 全会变成"引用"。
#:   3. 行号段允许 ``-`` 连写（``:137-142``），因为本项目用它表达范围。
REFERENCE_RE = re.compile(
    r"(?<![\w./-])"
    r"((?:[\w.-]+/)*[\w.-]+\.(?:py|ts|tsx|js|mjs|json|yaml|yml|sh|sql|toml|ini|cfg|conf|html|css|md|txt))"
    r":(\d+)(?:-(\d+))?",
)


# ==============================================================================
# 数据模型
# ==============================================================================
@dataclass(frozen=True)
class Problem:
    """一条引用问题。

    Attributes:
        where (`str`): 引用**写在哪**（``文件:行号``），这是要修的地方。
        reference (`str`): 引用的原文，例如 ``src/server/probes.py:262``。
        reason (`str`): 不通过的原因。
    """

    where: str
    reference: str
    reason: str


# ==============================================================================
# 文件索引
# ==============================================================================
def _walk_files() -> list[Path]:
    """收集仓库里**所有**可作为引用目标的文件。

    这里刻意用一次完整遍历建索引，而不是对每条引用做 ``rglob``：
    后者是 O(引用数 × 目录树)，在引用上百条时明显变慢，
    而本脚本会被反复调用（每次改文档）。

    Returns:
        `list[Path]`: 相对仓库根的路径（POSIX 风格，便于按 ``/`` 切分）。
    """
    found: list[Path] = []
    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(REPO_ROOT)
        if any(part in SKIP_DIRS for part in relative.parts):
            continue
        found.append(Path(relative.as_posix()))
    return found


def _scan_sources() -> list[Path]:
    """收集要**扫描**的文件（找引用）。

    Returns:
        `list[Path]`: 相对仓库根的路径。
    """
    sources: list[Path] = []
    for path in _walk_files():
        if path.as_posix() in SKIP_FILES:
            continue
        if any(part in NO_SCAN_DIRS for part in path.parts):
            continue
        if path.suffix in SCAN_SUFFIXES or path.name in SCAN_FILENAMES:
            sources.append(path)
    return sources


def _resolve_installed(reference_path: str) -> Path | None:
    """尝试把引用解析成**已安装第三方包**里的文件。

    本项目有一部分引用指向的不是仓库里的代码，而是**装进环境里**的依赖，
    例如 ``starlette/routing.py:695``（引用它说明中间件栈的构建顺序）、
    ``agentscope/model/_base.py:207``（框架改 pip 包后，这类引用是主力）。

    ⚠️ 为什么这类引用也值得校验，而不是「外部的不查」：
    它们恰恰是最容易漂的一类 —— 换一个 starlette 版本，那一行就变了，
    而文档照样说着「见 starlette/routing.py:695」。
    这种失效比找不到文件更隐蔽：文件在、行号在，只是指向了别的东西。

    实现上只取路径的第一段当模块名去 ``find_spec``，而不是遍历整个
    site-packages：前者是 O(1) 且精确（``starlette`` → 它自己的目录），
    后者会把所有依赖的上万个文件都卷进来。

    Args:
        reference_path (`str`): 引用里写的路径。

    Returns:
        `Path | None`: 解析到的绝对路径；解析不到则为 ``None``。
    """
    parts = [part for part in reference_path.split("/") if part]
    if len(parts) < 2:
        # 至少要「包名/文件」两段才可能是包内路径。
        return None

    import importlib.util

    try:
        spec = importlib.util.find_spec(parts[0])
    except (ImportError, ValueError, ModuleNotFoundError):
        return None

    if spec is None or not spec.submodule_search_locations:
        # 不是包（可能是单文件模块，或压根没装）。
        return None

    for location in spec.submodule_search_locations:
        candidate = Path(location).joinpath(*parts[1:])
        if candidate.is_file():
            return candidate
    return None


def _resolve(reference_path: str, all_files: list[Path]) -> list[Path]:
    """把引用里的路径解析成真实存在的文件（绝对路径）。

    分三级尝试，顺序固定 —— 每一级都对应文档里真实存在的一种写法::

        1. 仓库内精确路径      src/server/probes.py:262
        2. 仓库内**路径段后缀**  loader.py:456
                              rules.py:163-164
        3. 已安装的第三方包      agentscope/model/_base.py:207
                              starlette/routing.py:695

    为什么第 2 级不能退化成「按文件名匹配」：``model/_base.py``
    按文件名会同时命中 agentscope 包里的 ``model/_base.py`` 与
    ``app/storage/_model/_base.py``
    （两者 basename 都是 ``_base.py``）—— 于是引用会被校验到一个**完全无关的
    文件**上，而它的行数恰好也可能够长，校验就"通过"了。
    这种假通过比报错更糟：它把一条已经失效的引用标注成了有效。
    按路径段比对时，``model/_base.py`` 与 ``_model/_base.py`` 是不同的段序列，
    不会混淆。

    ⚠️ 第 2 级要求「引用里写的路径段」是「真实路径的**后缀**」。
    对框架源码的引用因此**必须写成带包名的形式**（``agentscope/model/_base.py:207``、
    ``reme/...``）：不带包名的 ``model/_base.py:207`` 自 2026-10-04（框架改
    pip 包、vendored 源码树移除）起，第 2 级找不到仓库内目标，
    第 3 级又拿 ``model`` 去 ``find_spec`` 也找不到 —— 会被报成找不到文件。
    这是**正确行为**，不是误报：那条引用在任何环境下都无法精确指向一份代码。
    发现这种报错时应当改引用，而不是放宽匹配规则。

    Args:
        reference_path (`str`): 引用里写的路径。
        all_files (`list[Path]`): 全仓库文件索引（相对仓库根）。

    Returns:
        `list[Path]`: 匹配到的文件（**绝对路径**；可能为空，也可能多于一个 = 歧义）。
    """
    # ---- 第 1 级：精确路径 --------------------------------------------------
    # 绝大多数引用是这种，能直接命中，省掉一次全表扫描。
    exact = REPO_ROOT / reference_path
    if exact.is_file():
        return [exact]

    # ---- 第 2 级：仓库内路径段后缀 ------------------------------------------
    wanted = tuple(part for part in reference_path.split("/") if part)
    if wanted:
        matched = [
            REPO_ROOT / candidate
            for candidate in all_files
            if len(candidate.parts) >= len(wanted)
            and tuple(candidate.parts[-len(wanted):]) == wanted
        ]
        if matched:
            return matched

    # ---- 第 3 级：已安装的第三方包 ------------------------------------------
    installed = _resolve_installed(reference_path)
    if installed is not None:
        return [installed]

    return []


def _line_count(path: Path) -> int:
    """读取文件的行数。

    ⚠️ 用**二进制**读而不是文本模式：被引用的文件（尤其是第三方包）里
    可能有非 UTF-8 字节，文本模式会抛 ``UnicodeDecodeError`` ——
    那会被报成「校验脚本崩了」，一个与「行号对不对」毫无关系的问题。
    这里只数换行符，编码无关紧要。

    Args:
        path (`Path`): 文件的**绝对路径**。

    Returns:
        `int`: 行数。读取失败时返回 0（调用方会把它报成「引用超界」）。
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return 0
    if not raw:
        return 0
    # 末尾没有换行的文件，最后一行也算一行 —— 所以要 +1。
    return raw.count(b"\n") + (0 if raw.endswith(b"\n") else 1)


# ==============================================================================
# 校验
# ==============================================================================
def check_references() -> tuple[list[Problem], int]:
    """扫描全部源文件并校验其中的行号引用。

    Returns:
        `tuple[list[Problem], int]`: 问题列表与「检查过的引用条数」。
    """
    all_files = _walk_files()
    problems: list[Problem] = []
    checked = 0

    for source in _scan_sources():
        try:
            text = (REPO_ROOT / source).read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            problems.append(Problem(str(source), "<读取失败>", f"{type(exc).__name__}: {exc}"))
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            for match in REFERENCE_RE.finditer(line):
                reference_path, start_raw, end_raw = match.group(1), match.group(2), match.group(3)
                start = int(start_raw)
                end = int(end_raw) if end_raw else start

                # 引用自己所在的这一行 = 自引用？不可能出现，跳过以防万一。
                resolved = _resolve(reference_path, all_files)
                where = f"{source}:{lineno}"
                reference = match.group(0)

                if not resolved:
                    problems.append(
                        Problem(where, reference, "找不到这个文件（路径写错？文件已删除/改名？）"),
                    )
                    continue

                checked += 1

                if end < start:
                    problems.append(
                        Problem(where, reference, f"行号区间倒置：{start}-{end}"),
                    )
                    continue

                # 有歧义时要求**每一个**候选都满足区间。
                # 理由：一条指代不明的引用，只有在所有可能的解读下都成立时才算安全。
                # 这条规则会自然地把作者推向写更具体的路径，而不是"能过就行"。
                for target in resolved:
                    total = _line_count(target)
                    if total == 0:
                        problems.append(Problem(where, reference, f"{target} 是空文件或读不到内容"))
                        continue
                    if start > total:
                        problems.append(
                            Problem(
                                where,
                                reference,
                                f"{target} 只有 {total} 行，引用的第 {start} 行不存在",
                            ),
                        )
                    elif end > total:
                        problems.append(
                            Problem(
                                where,
                                reference,
                                f"{target} 只有 {total} 行，引用区间上界 {end} 超界",
                            ),
                        )

    return problems, checked


def main() -> int:
    """脚本入口。

    Returns:
        `int`: 0 = 全部通过；1 = 有问题。
    """
    print(f"扫描目录：{REPO_ROOT}")
    problems, checked = check_references()

    if not problems:
        print(f"✅ 全部 {checked} 条行号引用均有效（文件存在、行号未超界）。")
        return 0

    print(f"❌ 发现 {len(problems)} 条无效的行号引用（共检查 {checked} 条）：")
    print()
    # 按「写在哪」排序，便于一次改完一个文件里的所有问题。
    for problem in sorted(problems, key=lambda p: p.where):
        print(f"  · {problem.where}")
        print(f"      引用：{problem.reference}")
        print(f"      原因：{problem.reason}")
    print()
    print("修法：把行号改成实际的行号，或把引用改成不依赖行号的说法")
    print("      （例如「见 config/base.yaml 的 db.pool_recycle_seconds 注释」）。")
    print("      ⚠️ 不要为了让本检查通过而删掉引用 —— 一条指不准的引用仍有价值，")
    print("         而一条被删掉的引用会把读者引向『这里没什么可看的』。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
