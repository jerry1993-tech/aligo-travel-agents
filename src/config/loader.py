# -*- coding: utf-8 -*-
"""配置加载器：把「YAML 文件 + 环境变量」合成为唯一的 :class:`Settings` 对象。

文件职责：
    定义配置的**来源顺序**与**合并语义**，并在最后一步交给
    :class:`src.config.schema.Settings` 做严格校验。全项目只有本模块读配置文件、
    只有本模块解析 ``ALIGO__`` 前缀 —— 其它任何地方都不应再去碰 ``os.environ``。

上下游依赖：
    - 上游：读 ``config/base.yaml``、``config/{env}.yaml``、进程环境变量
      （可选：仓库根 ``.env``）；校验规则来自 :mod:`src.config.schema`。
    - 下游：``src/server/app.py`` 的模块级装配调用 :func:`get_settings`；
      ``src/llm/factory.py``、``src/observability/tracing.py`` 等从入参
      :class:`Settings` 取配置。

------------------------------------------------------------------------------
加载顺序（后者覆盖前者，逐层**深合并**）
------------------------------------------------------------------------------
    1. ``config/base.yaml``          所有环境共享的默认值
    2. ``config/{env}.yaml``         环境增量（env 由 ``ALIGO__APP__ENV`` 决定，默认 dev）
    3. ``ALIGO__<段>__<键>`` 环境变量  精确到键的覆盖

「深合并」的含义：只覆盖出现的那一个键，同一段里没出现的键保留上一层的值。
例如 ``config/dev.yaml`` 只写了 ``app.log_level``，那么 ``app.port`` 仍取
``base.yaml`` 的 8000。若改成整段替换，环境增量文件就必须把整段抄全 ——
一旦上游加了新键，抄漏的那个环境就会静默缺键。这是本项目刻意避开的坑。

------------------------------------------------------------------------------
两条容易踩的边界（都有对应的单测）
------------------------------------------------------------------------------
    · **字符串值里的 ``${VAR}`` 会被展开**（用进程环境变量）。
      这不是 YAML 语法（PyYAML 不认），是本模块 ``_expand_env`` 的行为。
      目的是让密钥只有一处真值：写在 ``.env`` 里，配置文件只做引用。
      支持 ``${VAR:-默认值}`` 形式；``VAR`` 未设置且没给默认值时展开为空串，
      并汇总成一条 WARNING（不报错 —— 缺密钥的部署要能起来，见「零密钥原则」）。
    · **``ALIGO__`` 前缀的未知键会让启动直接失败**（schema 的 ``extra="forbid"``）。
      这是刻意的：拼错的键静默取默认值，是最难发现的一类偏差。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from .schema import Settings

# ==============================================================================
# 常量
# ==============================================================================

#: 环境变量前缀。双下划线是**层级分隔符**：``ALIGO__DB__POOL_SIZE`` → ``db.pool_size``。
ENV_PREFIX = "ALIGO__"

#: 层级分隔符。
_LEVEL_SEP = "__"

#: 匹配 ``${NAME}`` 与 ``${NAME:-默认值}``。
#:   - 第一组：变量名（字母数字下划线，且不以数字开头）
#:   - 第二组：可选默认值（``:-`` 之后直到 ``}`` 的全部内容）
_PLACEHOLDER_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


# ==============================================================================
# 路径
# ==============================================================================
def repo_root() -> Path:
    """返回仓库根目录（本文件上溯三层：``src/config/loader.py`` → 仓库根）。

    为什么不从 ``os.getcwd()`` 推导：``make -f /abs/path/Makefile``、IDE 的
    make 集成、以及以绝对路径启动的 uvicorn，都会让 CWD 不是仓库根。一旦如此，
    相对路径找 ``config/`` 就会失败，而失败方式是「配置文件不见了」这种
    指向错误方向的报错。锚定到 ``__file__`` 之后，位置不再依赖调用者的 CWD。

    Returns:
        `Path`: 仓库根目录的绝对路径。
    """
    return Path(__file__).resolve().parents[2]


# ==============================================================================
# 内部工具：深合并 / 占位符展开 / 环境变量降维
# ==============================================================================
def _deep_merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """把 ``overlay`` 深合并进 ``base`` 的**副本**，返回新字典。

    规则：两边同一个键对应的都是字典 ⇒ 递归合并；否则 ``overlay`` 的值整体胜出
    （**包括用列表/标量覆盖字典** —— 那种情况下语义是「显式替换」，不做元素级拼接）。

    ⚠️ 不修改任何入参。配置在进程内是被多处共享的，就地修改会让「谁改的」
    变得无法追踪 —— 一次误改会污染后续所有读取者。

    Args:
        base (`dict`): 底层配置（如 base.yaml 的内容）。
        overlay (`Mapping`): 上层配置（如 dev.yaml 或环境变量降维后的字典）。

    Returns:
        `dict`: 合并后的新字典。
    """
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _expand_env(
    node: Any,
    environ: Mapping[str, str],
    missing: set[str],
) -> Any:
    """递归展开结构里所有字符串中的 ``${VAR}`` 占位符。

    只处理字符串叶子 —— 字典与列表会被递归下去，数字/布尔原样返回
    （YAML 里已经是 int/bool 的值不需要展开，也没有 ``${}`` 可展开）。

    Args:
        node (`Any`): 待展开的子树（字典 / 列表 / 标量）。
        environ (`Mapping[str, str]`): 用于替换的环境变量视图。
        missing (`set[str]`): **出参**。未设置且无默认值的变量名会被加进这个集合，
            由调用方汇总成一条告警。用出参而不是返回值，是为了不让递归的
            返回结构变复杂。

    Returns:
        `Any`: 展开后的同构结构。
    """
    if isinstance(node, str):
        def _sub(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            if name in environ:
                return environ[name]
            if default is not None:
                # ${VAR:-默认值}：变量缺失时用默认值，**不记进 missing**
                # —— 调用方已经显式表达了「缺失也没关系」。
                return default
            missing.add(name)
            # 展开为空串而不是保留原样：保留 `${REDIS_PASSWORD}` 会让一个
            # 明显的配置缺失伪装成「看起来正常」的连接串，故障点被推迟到
            # 真正建连时才暴露，且报错内容与根因相距很远。
            return ""

        return _PLACEHOLDER_RE.sub(_sub, node)

    if isinstance(node, dict):
        return {k: _expand_env(v, environ, missing) for k, v in node.items()}

    if isinstance(node, list):
        return [_expand_env(item, environ, missing) for item in node]

    return node


def _decode_container(value: str) -> Any:
    """把 JSON 容器字面量解码成 Python 对象，其余原样返回字符串。

    ★ 为什么需要这一步（这是本项目实际踩到的坑）：

        ``exempt_paths`` 之类的字段是 ``tuple[str, ...]``，只能从 YAML
        拿到真正的列表。而环境变量的值**永远是字符串**，pydantic 会拒绝
        把 ``'["/healthz"]'`` 当成 tuple —— 报错是
        ``Input should be a valid tuple [type=tuple_type]``。

        于是 ``ALIGO__RATELIMIT__EXEMPT_PATHS=...`` 这个键
        看似可配、实则一设就崩，而崩溃信息完全指不出「这个字段
        压根不支持从环境变量配置」。症状是容器**启动即崩循环**，
        且报错看起来像是值写错了。

        本函数让这类字段真的可配：写 JSON 即可。

    ⚠️ 只在**首字符是 ``[`` 或 ``{``** 时才尝试解析：
        配置里绝大多数值是 URL、密码、路径，逐个 ``json.loads`` 既慢
        又危险（例如一个恰好是合法 JSON 的密码会被悄悄换成别的类型）。
        以 ``[``/``{`` 开头是一个足够廉价且足够明确的信号。

    ⚠️ 解析失败或结果不是容器时，**原样返回字符串**：
        让 pydantic 去报「类型不对」。若在这里就抛异常，报错会变成
        「JSON 解析失败」，而真正的问题往往是「这个字段本来就不接受列表」。
        让类型错误由 schema 报，信息才落在正确的位置。

    ⚠️ 只接受 ``list`` / ``dict``：``json.loads`` 也认 ``null`` ``true`` ``1``，
        但那些**标量**本来就由 pydantic 负责转换，从这里走等于开了第二条
        转换路径 —— 两条路径迟早对同一个值给出不同结论。

    Args:
        value (`str`): 环境变量的原始值。

    Returns:
        `Any`: 解析出的 list/dict，或原字符串。
    """
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        return value
    return decoded if isinstance(decoded, (list, dict)) else value


def _env_overrides(environ: Mapping[str, str]) -> tuple[dict[str, Any], list[str]]:
    """把 ``ALIGO__<段>__<键>`` 形式的环境变量降维成嵌套字典。

    ``ALIGO__DB__POOL_SIZE=10`` → ``{"db": {"pool_size": "10"}}``。
    值的**类型转换交给 pydantic**（schema 里声明的是 int，传字符串 "10" 会被
    pydantic 转成 10）。本函数只负责「键的层级」这一件事。

    ⚠️ 「交给 pydantic」只对**标量**成立。``tuple`` / ``list`` / ``dict``
        这类字段拿字符串是转不出来的，必须先由 :func:`_decode_container`
        解码成 JSON 容器 —— 否则那些字段会变成「能从 YAML 配、不能从
        环境变量配」，而本项目的部署方式正是以环境变量为主。

    Args:
        environ (`Mapping[str, str]`): 进程环境变量视图。

    Returns:
        `tuple[dict, list[str]]`:
            - 降维后的嵌套字典；
            - 被跳过的变量名列表（形如 ``ALIGO__`` 之后为空的畸形键）。
              它们**不静默丢弃**，由调用方告警 —— 静默丢弃会让一个写错的键
              看起来「已经在生效」。
    """
    overrides: dict[str, Any] = {}
    malformed: list[str] = []

    for raw_name, value in environ.items():
        if not raw_name.startswith(ENV_PREFIX):
            continue

        # 去掉前缀后按 `__` 切分。注意 `ALIGO__IMAGE_REGISTRY`
        # 切出来只有一层 `["image_registry"]` ⇒ 落在根上，
        # 而 Settings 没有这个字段 ⇒ pydantic 报 extra_forbidden。
        # 这正是 .env.example 反复警告「别把 ALIGO__IMAGE_REGISTRY 写进 .env」的机制。
        parts = [p.lower() for p in raw_name[len(ENV_PREFIX):].split(_LEVEL_SEP)]
        if not parts or any(not p for p in parts):
            malformed.append(raw_name)
            continue

        cursor = overrides
        for part in parts[:-1]:
            child = cursor.get(part)
            if not isinstance(child, dict):
                # 上一层已经被标量占用（例如同时给了 ALIGO__DB 与 ALIGO__DB__URL）：
                # 建一个空字典顶掉它，让后续的层级键落进去。真正的冲突会在
                # pydantic 校验时以「期望字典、得到标量」的形式暴露出来。
                child = {}
                cursor[part] = child
            cursor = child
        cursor[parts[-1]] = _decode_container(value)

    return overrides, malformed


def _read_yaml(path: Path) -> dict[str, Any]:
    """读取一个 YAML 文件并保证顶层是字典。

    空文件与「只有注释的文件」都返回空字典（YAML 把它们解析成 ``None``）。
    这允许环境增量文件先建出来占位、之后再逐项填值。

    Args:
        path (`Path`): YAML 文件路径。

    Returns:
        `dict`: 文件内容；空文件返回 ``{}``。

    Raises:
        ValueError: 文件不存在，或顶层不是字典（后者几乎总是缩进写错导致的）。
    """
    if not path.exists():
        raise ValueError(
            f"配置文件不存在：{path}\n"
            f"（config/ 目录是项目的一部分，请确认它没有被误删或误挂载覆盖。）",
        )

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(
            f"配置文件顶层必须是「键: 值」的字典，实际得到 {type(loaded).__name__}：{path}\n"
            f"（最常见的成因是缩进不一致，把整段变成了列表或标量。）",
        )
    return loaded


def _load_dotenv_values(path: Path) -> dict[str, str]:
    """读取仓库根的 ``.env``（如果存在），返回其键值对。

    **这是一层便利，不是一层真值**：容器里根本没有 ``.env``（被 .dockerignore 排除），
    配置完全由 compose 的 ``env_file`` 注入。本函数存在的唯一目的，是让
    「本机直接跑 python 脚本 / 跑 pytest」时也能拿到 POSTGRES_*、DASHSCOPE_* 这些
    非 ``ALIGO__`` 前缀的变量，从而与容器行为一致。

    优先级上它**低于**进程环境变量（见 :func:`load_settings` 的合并顺序）：
    显式 ``export`` 的值永远赢过文件里的值。

    依赖 ``python-dotenv`` 是可选的 —— 没装就静默跳过，不影响主流程。

    Args:
        path (`Path`): ``.env`` 的路径。

    Returns:
        `dict[str, str]`: 文件里的键值对；文件不存在或 python-dotenv 不可用时为空字典。
    """
    if not path.exists():
        return {}

    try:
        from dotenv import dotenv_values
    except ImportError:  # pragma: no cover - 仅在没有 python-dotenv 的环境触发
        return {}

    # dotenv_values 返回 dict[str, str | None]；值为 None 的条目表示
    # 「文件里出现了这个键但没有等号右边」，按空串处理。
    return {k: (v if v is not None else "") for k, v in dotenv_values(path).items()}


# ==============================================================================
# 对外入口
# ==============================================================================
def load_settings(
    env: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    config_dir: str | Path | None = None,
    dotenv: bool = True,
) -> Settings:
    """按「base.yaml → {env}.yaml → ALIGO__ 环境变量」的顺序装配并校验配置。

    Args:
        env (`str | None`):
            运行环境名（dev / test / prod）。``None`` 时取环境变量
            ``ALIGO__APP__ENV``，仍为空则用 ``"dev"``。
            显式传入的 ``env`` **优先于** ``ALIGO__APP__ENV``，并且加载完成后
            ``settings.app.env`` 一定等于这里选定的值（见第 4.5 步）——
            「读的是哪份 YAML」与「app.env 写着什么」不允许出现分歧。
        environ (`Mapping[str, str] | None`):
            环境变量视图。``None``（默认）表示「读真实进程环境（并按需叠加 .env）」；
            显式传入时**完全以它为准**，既不读 ``os.environ`` 也不读 ``.env`` ——
            这是让配置相关的单测可重复的关键。
        config_dir (`str | Path | None`): config 目录；``None`` 时用仓库根的 ``config/``。
        dotenv (`bool`): 是否读取仓库根 ``.env`` 作为低优先级的补充。默认 True。

    Returns:
        `Settings`: 通过严格校验的配置对象。

    Raises:
        ValueError:
            - 配置文件缺失或格式非法；
            - 合并后的配置未通过 schema 校验（**含未知键**）。
              异常文案以 ``配置校验失败（环境 'xxx'）`` 开头，后面跟 pydantic
              的逐条明细 —— 排障时先看这里。
    """
    # ---- 第 0 步：确定环境变量的最终视图 -------------------------------------
    if environ is None:
        base_dir = repo_root()
        merged_env: dict[str, str] = {}
        if dotenv:
            merged_env.update(_load_dotenv_values(base_dir / ".env"))
        # 进程环境变量后合并 ⇒ 显式 export 的值赢过 .env 文件里的值。
        merged_env.update(os.environ)
        effective_env: Mapping[str, str] = merged_env
    else:
        effective_env = environ

    # ---- 第 1 步：决定加载哪一份环境增量 -------------------------------------
    resolved_env = env or effective_env.get(f"{ENV_PREFIX}APP{_LEVEL_SEP}ENV") or "dev"
    if resolved_env not in ("dev", "test", "prod"):
        # 提前拦一道，给出一条比 pydantic 更贴近场景的提示：
        # 走到 schema 校验再报错的话，用户看到的会是「app.env 不是合法枚举」，
        # 而真正的问题是「config/{这个名字}.yaml 根本不存在」。
        raise ValueError(
            f"未知的运行环境 {resolved_env!r}：只支持 dev / test / prod。\n"
            f"（它由环境变量 ALIGO__APP__ENV 决定，默认 dev。）",
        )

    directory = Path(config_dir) if config_dir is not None else repo_root() / "config"

    # ---- 第 2 步：base.yaml，再深合并 {env}.yaml -----------------------------
    raw = _read_yaml(directory / "base.yaml")
    raw = _deep_merge(raw, _read_yaml(directory / f"{resolved_env}.yaml"))

    # ---- 第 3 步：展开 ${VAR} 占位符 -----------------------------------------
    # 放在「环境变量覆盖」**之前**：占位符只出现在 YAML 里，
    # 环境变量给的是字面值，展开它既无意义也可能误伤（值里恰好含 ${} 时）。
    missing: set[str] = set()
    raw = _expand_env(raw, effective_env, missing)
    if missing:
        # 只告警、不失败 —— 这就是「零密钥原则」：缺 LLM key 也要能起来
        # （api_key 展开成空串 ⇒ 触发 MockLLM 降级）。
        import logging  # 局部导入：本模块在极早期被加载，避免在模块级引入日志副作用

        logging.getLogger(__name__).warning(
            "配置里引用了 %d 个未设置的环境变量，已展开为空串：%s",
            len(missing),
            ", ".join(sorted(missing)),
        )

    # ---- 第 4 步：ALIGO__ 环境变量覆盖（优先级最高）--------------------------
    overrides, malformed = _env_overrides(effective_env)
    if malformed:
        import logging

        logging.getLogger(__name__).warning(
            "忽略了 %d 个畸形的 ALIGO__ 环境变量（前缀之后为空）：%s",
            len(malformed),
            ", ".join(sorted(malformed)),
        )
    raw = _deep_merge(raw, overrides)

    # ---- 第 4.5 步：把 app.env 钉成**实际加载的那一档** ----------------------
    # 为什么必须显式钉：app.env 在 base.yaml 里的字面值是 `dev`。
    # 若这次加载的是 test 档（显式传参、或 ALIGO__APP__ENV=test），而没人去改
    # 这个字段，就会得到「YAML 读的是 test.yaml、app.env 却写着 dev」的自相矛盾状态。
    # 它的危害不是好看不好看：下游按 `settings.app.env` 决定集合后缀、日志级别、
    # 是否建表 —— 一个说 dev、实际跑在 test 数据上的进程，会把数据写错地方。
    # 这里以 resolved_env 为准（它本来就是「显式传参 → ALIGO__APP__ENV → dev」
    # 这条链路的唯一结果），从而保证「名字」与「内容」永远一致。
    app_section = raw.get("app")
    if not isinstance(app_section, dict):
        app_section = {}
        raw["app"] = app_section
    app_section["env"] = resolved_env

    # ---- 第 5 步：严格校验 ---------------------------------------------------
    try:
        return Settings.model_validate(raw)
    except ValidationError as exc:
        # 保留 pydantic 的完整明细（含 extra_forbidden 的
        # 「Extra inputs are not permitted」字样与出错字段名），
        # 只加一层中文前缀便于在容器日志里一眼定位。
        raise ValueError(
            f"配置校验失败（环境 {resolved_env!r}）：\n{exc}\n"
            f"（配置文件：{directory}/base.yaml + {resolved_env}.yaml；"
            f"环境变量覆盖前缀：{ENV_PREFIX}）",
        ) from exc


# ------------------------------------------------------------------------------
# 进程内单例
# ------------------------------------------------------------------------------
# 配置在一次进程生命周期内是不变的，重复解析 YAML 纯属浪费；
# 更重要的是：**保证全进程拿到同一个对象**，否则「A 模块改了配置、B 模块没看见」
# 这类问题会在多线程下变成偶发故障。
_settings_cache: Settings | None = None


def get_settings() -> Settings:
    """返回进程内共享的配置对象（首次调用时加载并缓存）。

    Returns:
        `Settings`: 全局唯一的配置实例。
    """
    global _settings_cache
    if _settings_cache is None:
        _settings_cache = load_settings()
    return _settings_cache


def set_settings(settings: Settings | None) -> None:
    """覆盖进程内的配置缓存（``None`` 表示清空、下次访问重新加载）。

    **仅供测试使用**：生产代码不应在运行期替换配置 —— 那会让「启动时校验过的配置」
    与「实际生效的配置」不再一致。存在这个函数是为了让用例能注入一份
    sqlite / Mock 的配置，而不必去改进程环境变量（后者会影响同进程的其它用例）。

    Args:
        settings (`Settings | None`): 要注入的配置，或 ``None`` 以清空缓存。
    """
    global _settings_cache
    _settings_cache = settings


__all__ = [
    "ENV_PREFIX",
    "get_settings",
    "load_settings",
    "repo_root",
    "set_settings",
]
