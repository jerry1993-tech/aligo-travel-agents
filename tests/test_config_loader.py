# -*- coding: utf-8 -*-
"""配置加载器（``src/config/loader.py``）的契约测试。

==============================================================================
这些用例在防什么
==============================================================================
    配置是本项目**唯一**的「改一个字就全线崩」的地方，而且它的失败方式很隐蔽：

      · 多写一个键  → 启动即崩（好），但只有**恰好**那份 YAML 被加载时才崩（坏）；
      · 少写一个键  → 静默取默认值，服务照常跑，**没有任何症状**；
      · 键拼错      → 与「少写一个键」完全相同 —— 静默、无症状。

    第二、三类才是真正致命的，因为它们不会被任何人工验证发现。
    本文件的用例围绕「让静默的东西变得可断言」来写：
    每一条 ``ALIGO__`` 覆盖路径都有一条用例钉住它，改坏立刻红。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.config import Settings, load_settings


# ==============================================================================
# 一、环境档位的选择
# ==============================================================================
def test_default_env_is_dev_when_nothing_specified(tmp_path: Path) -> None:
    """什么环境变量都不给时，默认落在 dev 档。

    这条是「clone 下来直接能跑」的基础：不要求开发者预先 export 任何东西。
    """
    settings = load_settings(environ={}, dotenv=False)

    assert settings.app.env == "dev"


def test_explicit_env_beats_env_var(tmp_path: Path) -> None:
    """显式传参的 ``env`` 优先于 ``ALIGO__APP__ENV``。

    ⚠️ 这条不是可有可无的细节：``make test`` 依赖「显式传 ``test`` 就一定
    加载 test.yaml」，若环境变量能盖过它，那么开发者的 shell 里只要残留一个
    ``ALIGO__APP__ENV=prod``，测试就会去读生产档配置 —— 而它看起来是完全正常的。
    """
    settings = load_settings(
        "test",
        environ={"ALIGO__APP__ENV": "dev"},
        dotenv=False,
    )

    assert settings.app.env == "test"


def test_app_env_field_matches_loaded_profile() -> None:
    """``settings.app.env`` 必须等于**实际加载的那一档**，不能是 YAML 里的字面值。

    base.yaml 里 ``app.env`` 写死的是 ``dev``。若加载 test 档时不去改这个字段，
    就会出现「读的是 test.yaml、app.env 却写着 dev」的自相矛盾状态。
    它的危害不是好看不好看：下游按 ``app.env`` 决定 Milvus 集合后缀与是否建表 ——
    一个「说 dev、实际跑在 test 数据上」的进程会把数据写错地方。
    """
    for env in ("dev", "test", "prod"):
        settings = load_settings(env, environ={}, dotenv=False)
        assert settings.app.env == env, f"加载 {env} 档却得到 app.env={settings.app.env}"


def test_unknown_env_is_rejected_with_actionable_message() -> None:
    """未知的环境名要报错，且错误文案要指向**真正的原因**。

    ``ALIGO__APP__ENV=staging`` 时，如果只在 schema 层报错，用户看到的是
    「app.env 不是合法枚举」—— 而真正的问题是「``config/staging.yaml`` 不存在」。
    用例断言的是**后者**被说出来。
    """
    with pytest.raises(ValueError) as excinfo:
        load_settings("staging", environ={}, dotenv=False)

    message = str(excinfo.value)
    assert "staging" in message
    assert "dev / test / prod" in message


# ==============================================================================
# 二、环境变量覆盖（ALIGO__<段>__<键>）
# ==============================================================================
def test_env_var_overrides_yaml_value() -> None:
    """``ALIGO__LLM__MODEL`` 能顶掉 YAML 里的 ``llm.model``。

    ⚠️ 这是排障时最常被误解的一条：.env 里的 ``LLM_MODEL=xxx`` 是**死键**
    （loader 只认 ``ALIGO__`` 前缀），写了完全不生效。用例把「哪个前缀有效」
    这件事钉死，免得有人「修好」了 loader 去兼容裸键名。
    """
    settings = load_settings(
        "test",
        environ={"ALIGO__LLM__MODEL": "deepseek-reasoner"},
        dotenv=False,
    )

    assert settings.llm.model == "deepseek-reasoner"


def test_env_var_value_is_type_coerced_by_schema() -> None:
    """环境变量给的一律是字符串，类型转换由 pydantic 负责。

    ``ALIGO__LLM__TEMPERATURE=0.9`` 必须变成 ``float`` 而不是字符串 ``"0.9"``，
    否则后续传给模型的参数会是一段文本，且**报错发生在调用下游时**，
    与本处的配置相距甚远。
    """
    settings = load_settings(
        "test",
        environ={
            "ALIGO__LLM__TEMPERATURE": "0.9",
            "ALIGO__DB__POOL_SIZE": "7",
            "ALIGO__OBSERVABILITY__METRICS_ENABLED": "false",
        },
        dotenv=False,
    )

    assert settings.llm.temperature == pytest.approx(0.9)
    assert isinstance(settings.llm.temperature, float)
    assert settings.db.pool_size == 7
    assert settings.observability.metrics_enabled is False


# ==============================================================================
# 二·补、跨字段校验（单看一个字段看不出问题的那些）
# ==============================================================================
def test_jwt_enabled_without_secret_is_rejected() -> None:
    """开了 JWT 却没给密钥 ⇒ 启动即失败。

    ⚠️ 为什么这条**必须**是硬失败，而不是「自动生成一个随机密钥」：
    ``jwt_secret`` 为空时，任何调用者都能自己签一个 token 通过校验 ——
    而且服务会**照常运行**，没有任何异常、没有任何日志。
    这不是「配置有点问题」，这是一次静默的、完全的鉴权绕过。
    因此宁可让进程起不来。

    用例同时钉住「谁能发现这个问题」：不是代码评审，是启动本身。
    """
    with pytest.raises(ValueError) as excinfo:
        load_settings(
            "test",
            environ={"ALIGO__AUTH__JWT_ENABLED": "true"},
            dotenv=False,
        )

    assert "jwt_secret" in str(excinfo.value)


def test_jwt_enabled_with_secret_is_accepted() -> None:
    """反过来：给了密钥就必须能起来。

    少了这条，一个「无脑拒绝所有 jwt 配置」的实现也能通过上一条用例。
    """
    settings = load_settings(
        "test",
        environ={
            "ALIGO__AUTH__JWT_ENABLED": "true",
            "ALIGO__AUTH__JWT_SECRET": "a-test-secret-not-used-in-production",
        },
        dotenv=False,
    )

    assert settings.auth.jwt_enabled is True
    assert settings.auth.jwt_secret == "a-test-secret-not-used-in-production"


# ==============================================================================
# 二·补二、限流配置（P2 新增）
# ==============================================================================
def test_ratelimit_window_must_be_positive() -> None:
    """窗口长度必须 > 0 —— 否则限流中间件**在构造函数里**就除零。

    ``RateLimitMiddleware.__init__`` 会算
    ``requests_per_window / window_seconds`` 来得到回填速率。
    窗口为 0 时那是一句 ``ZeroDivisionError``，发生在**装配期**：
    症状是应用完全起不来，而栈里指向的是一行除法，看不出它与配置有关。

    ⚠️ 断言的是 ``ValueError``（配置层干预）而不是让它走到
        ``ZeroDivisionError``。两道防线的差别在于**报错说人话**：
        前者会说「window_seconds 必须大于 0」，后者只有一个零除。
    """
    with pytest.raises(ValueError) as excinfo:
        load_settings(
            "test",
            environ={"ALIGO__RATELIMIT__WINDOW_SECONDS": "0"},
            dotenv=False,
        )

    assert "window_seconds" in str(excinfo.value)


def test_ratelimit_zero_capacity_is_rejected() -> None:
    """``requests_per_window=0`` 必须被拒 —— 那等于「一个请求都不放行」。

    这个值不会让任何东西崩溃，它只会让**所有**请求收到 429。
    运维看到的现象是「服务被限流打挂了」，于是去查攻击，
    而真相是配置里少打了一个数字。静默的配置错误比崩溃更难查。
    """
    with pytest.raises(ValueError):
        load_settings(
            "test",
            environ={"ALIGO__RATELIMIT__REQUESTS_PER_WINDOW": "0"},
            dotenv=False,
        )


def test_ratelimit_zero_max_keys_is_rejected() -> None:
    """``max_keys=0`` 必须被拒 —— 桶字典的容量上限不能为零。

    上限的存在意义是防内存耗尽（身份来自请求头，不设限时一个伪造身份的
    脚本就能让字典无限增长）。取 0 会让每次请求都触发一次淘汰，
    把一个「防护」变成一个「每次请求都跑的额外开销」，而且**看起来配了限流**。
    """
    with pytest.raises(ValueError):
        load_settings(
            "test",
            environ={"ALIGO__RATELIMIT__MAX_KEYS": "0"},
            dotenv=False,
        )


def test_ratelimit_can_be_disabled_and_retuned_from_env() -> None:
    """开关与数值都能被环境变量覆盖，且被正确转型。

    ⚠️ 这条与上面三条是一对：只断言「非法值被拒」的话，一个把所有
        ratelimit 配置都拒掉的实现同样能全绿。必须有「合法值真的生效」。
    """
    settings = load_settings(
        "test",
        environ={
            "ALIGO__RATELIMIT__ENABLED": "false",
            "ALIGO__RATELIMIT__REQUESTS_PER_WINDOW": "7",
            "ALIGO__RATELIMIT__WINDOW_SECONDS": "3.5",
        },
        dotenv=False,
    )

    assert settings.ratelimit.enabled is False
    assert settings.ratelimit.requests_per_window == 7
    assert settings.ratelimit.window_seconds == 3.5


def test_json_container_value_can_set_a_tuple_field() -> None:
    """★ 列表/元组字段可以由环境变量以 **JSON 字面量**覆盖。

    钉的是这个坑：环境变量的值永远是字符串，而 ``exempt_paths`` 是
    ``tuple[str, ...]`` —— 在 ``_decode_container`` 之前，
    写 ``ALIGO__RATELIMIT__EXEMPT_PATHS='["/healthz"]'`` 会得到
    ``Input should be a valid tuple [type=tuple_type]``，容器**启动即崩循环**，
    而报错完全指不出「这个字段压根不支持从环境变量配置」。

    本项目的部署方式以 ``.env`` 为主（见 docker-compose.yaml），
    所以「只能从 YAML 配的字段」实际上等于「运维改不动」。
    """
    settings = load_settings(
        "test",
        environ={
            "ALIGO__RATELIMIT__EXEMPT_PATHS": '["/healthz", "/api/v1/health"]',
        },
        dotenv=False,
    )

    assert settings.ratelimit.exempt_paths == ("/healthz", "/api/v1/health")


def test_bracket_value_that_is_not_json_falls_back_to_a_type_error() -> None:
    """以 ``[`` 开头但**不是**合法 JSON 时，报的应当是**类型**错误。

    ⚠️ 这条在钉「错误信息落在正确的位置」。一个偷懒的实现会在
        ``json.loads`` 抛异常时把 JSONDecodeError 透出去，于是运维看到的
        是「JSON 解析失败」—— 而真正的问题往往是「这个字段本来就不接受
        列表」（比如往一个字符串字段里填了看起来像数组的值）。
        这里断言报错里出现字段名，即由 schema 报的类型错误。
    """
    with pytest.raises(ValueError) as excinfo:
        load_settings(
            "test",
            # 逗号分隔是很多人第一反应会写的形式；它不是 JSON，因此
            # 按原字符串交给 pydantic，得到一个明确的类型错误。
            environ={"ALIGO__RATELIMIT__EXEMPT_PATHS": "/healthz,/readyz"},
            dotenv=False,
        )

    message = str(excinfo.value)
    assert "exempt_paths" in message
    assert "tuple" in message


def test_bracket_string_values_are_not_touched_for_other_fields() -> None:
    """普通标量字段的取值不受容器解码影响。

    防止的是这种回归：把 ``_decode_container`` 写成「先 json.loads，
    失败就用原值」，于是任何**恰好是合法 JSON** 的标量都会被悄悄改型。
    这里用一个以 ``[`` 开头的连接串验证它原样保留。
    """
    settings = load_settings(
        "test",
        environ={"ALIGO__REDIS__URL": "redis://:p@[::1]:6379/0"},
        dotenv=False,
    )

    assert settings.redis.url == "redis://:p@[::1]:6379/0"


def test_unknown_key_inside_known_section_is_rejected() -> None:
    """已知段落里的未知键必须报错（``extra_forbidden``）。

    这是本项目最重要的一条配置约束：拼错的键若不报错，就会静默取默认值，
    而那是最难发现的一类偏差 —— 配置看起来「写了」，实际一步都没生效。
    """
    with pytest.raises(ValueError) as excinfo:
        load_settings(
            "test",
            environ={"ALIGO__LLM__MODEL_NAME": "deepseek-chat"},  # 应为 MODEL
            dotenv=False,
        )

    message = str(excinfo.value)
    assert "配置校验失败" in message
    assert "model_name" in message


def test_root_level_unknown_key_is_rejected() -> None:
    """根层级的未知键同样报错。

    ⚠️ 用例单选 ``ALIGO__IMAGE_REGISTRY`` 不是随手挑的：它是**镜像仓库**地址，
    看起来完全像一条正当配置，很多人会顺手写进 .env。而 loader 会把它降维成
    根层级的 ``image_registry`` 键 ⇒ 容器**启动即崩循环**。
    .env.example 里为此专门写了一段警告，本用例是那段警告的机器化版本。
    """
    with pytest.raises(ValueError) as excinfo:
        load_settings(
            "test",
            environ={"ALIGO__IMAGE_REGISTRY": "registry.example.com"},
            dotenv=False,
        )

    assert "image_registry" in str(excinfo.value)


def test_malformed_env_var_is_ignored_not_fatal(capsys: pytest.CaptureFixture[str]) -> None:
    """形如 ``ALIGO__`` 之后为空的畸形键被**跳过并告警**，而不是让进程崩溃。

    为什么它与其他未知键不同待遇：``ALIGO__DB__``（末尾多两个下划线）这类
    几乎都是手滑，且它不含任何有效信息，报错只会把一次可恢复的启动
    变成一次服务中断。而告警保证了它不会被**静默**丢弃 ——
    否则用户会以为这个键「已经在生效」。
    """
    settings = load_settings("test", environ={"ALIGO__DB__": "oops"}, dotenv=False)

    # 服务照常起来，配置取自 YAML 默认值。
    assert isinstance(settings, Settings)
    assert settings.app.env == "test"


# ==============================================================================
# 三、${VAR} 占位符展开
# ==============================================================================
def test_placeholder_expands_from_environment() -> None:
    """``${VAR}`` 用**进程环境变量**替换（不是 YAML 自己的语法）。"""
    settings = load_settings(
        "test",
        environ={
            "POSTGRES_USER": "aligo",
            "POSTGRES_PASSWORD": "s3cret",
            "POSTGRES_DB": "aligo_db",
        },
        dotenv=False,
    )

    assert settings.db.url == "postgresql+asyncpg://aligo:s3cret@postgres:5432/aligo_db"


def test_missing_placeholder_expands_to_empty_string() -> None:
    """未设置且无默认值的占位符展开为**空串**，不保留 ``${...}`` 原文。

    ⚠️ 这条直接支撑「零密钥可运行」：``llm.api_key`` 引用 ``${DASHSCOPE_API_KEY}``，
    机器上没有这个变量时它必须变成空串 —— 只有空串才能触发
    ``src/llm/factory.py`` 的 MockLLM 降级。

    若哪天有人「改进」成保留原文（看起来更友好），后果是：
    api_key 变成字符串 ``"${DASHSCOPE_API_KEY}"`` ⇒ 既不空、也不对 ⇒
    降级不触发 ⇒ 服务拿着这个假 key 去请求 ⇒ 401。
    故障点从「启动时看得见」被推迟到「第一次对话时看不懂」。
    """
    settings = load_settings("test", environ={}, dotenv=False)

    assert settings.llm.api_key == ""
    assert "${" not in settings.llm.api_key
    assert "${" not in settings.db.url


def test_placeholder_with_default_uses_default_when_missing() -> None:
    """``${VAR:-默认值}`` 在变量缺失时用默认值。

    与上一条的区别：这里调用方**显式表达了「缺失是可以的」**，
    因此不该产生告警噪声。本项目用它来表达 ``download_secret`` ——
    留空是预期状态（框架会随机生成），不是配置缺失。
    """
    settings = load_settings("test", environ={}, dotenv=False)

    assert settings.app.download_secret == ""


def test_placeholder_with_default_prefers_environment_when_present() -> None:
    """``${VAR:-默认值}`` 在变量**存在**时必须用变量的值，而不是默认值。

    少了这条，一个「默认值写对了、变量却没生效」的实现也能通过上一条用例。
    """
    settings = load_settings(
        "test",
        environ={"ALIGO_DOWNLOAD_SECRET": "fixed-secret-for-multi-replica"},
        dotenv=False,
    )

    assert settings.app.download_secret == "fixed-secret-for-multi-replica"


# ==============================================================================
# 三·补、编排段（快慢车道）
# ==============================================================================
def test_orchestration_section_is_populated_from_base_yaml() -> None:
    """``orchestration`` 段的默认值必须真的来自 ``base.yaml``。

    ⚠️ 这条挡的是 schema 里那个「两步走」约定的**第二步遗漏**：schema 加了
    字段但忘了去 ``base.yaml`` 写值。此时校验不会失败（字段有 schema 默认值），
    一切照常运行 —— 于是「配置文档里写着阈值可调」与「实际生效的是代码里
    写死的默认值」这两件事就悄悄分家了。

    ⚠️ 断言的是**具体数值**而不是「字段存在」：本用例要能发现「有人把
    base.yaml 里的 0.6 改成 0.3 却没更新文档」，所以期望值必须写死。
    数值本身是产品决定，改动它就该改动这条用例。
    """
    settings = load_settings("test", environ={}, dotenv=False)

    assert settings.orchestration.fast_lane_enabled is True
    assert settings.orchestration.fast_lane_max_chars == 20
    assert settings.orchestration.intent_confidence_threshold == 0.6
    assert settings.orchestration.dynamic_prompt_enabled is True
    assert settings.orchestration.max_subagent_calls == 5
    assert settings.orchestration.expose_reasoning is True


def test_orchestration_is_overridable_from_env() -> None:
    """编排段可由 ``ALIGO__ORCHESTRATION__*`` 覆盖（用于线上排查）。

    ⚠️ 这条是「关掉快车道排查问题」这个运维手段的**可行性验证**。若环境变量
    覆盖不到这个段，`fast_lane_enabled` 这个开关就只是个摆设 —— 而运维会
    在真正需要它的那一刻才发现，那时已经在处理事故了。

    ⚠️ ``"false"`` 字符串被转成 ``False`` 是 loader 的类型强转在起作用
    （环境变量一律是字符串）。这条同时验证了布尔强转对该段生效。
    """
    settings = load_settings(
        "test",
        environ={
            "ALIGO__ORCHESTRATION__FAST_LANE_ENABLED": "false",
            "ALIGO__ORCHESTRATION__INTENT_CONFIDENCE_THRESHOLD": "0.85",
        },
        dotenv=False,
    )

    assert settings.orchestration.fast_lane_enabled is False
    assert settings.orchestration.intent_confidence_threshold == 0.85


@pytest.mark.parametrize(
    ("key", "value"),
    [
        # 置信度是概率，不能超过 1 —— 超出后「低于阈值就追问」这条规则
        # 会变成「永远追问」，用户每句话都被反问一次。
        ("INTENT_CONFIDENCE_THRESHOLD", "1.5"),
        ("INTENT_CONFIDENCE_THRESHOLD", "-0.1"),
        # 长度为 0 会让快车道彻底失效（任何输入都超限），而它看起来
        # 只是一个「把限制调严一点」的无害改动。
        ("FAST_LANE_MAX_CHARS", "0"),
        # 单轮子调用上限为 0 等于「一个子智能体都不许调」。
        ("MAX_SUBAGENT_CALLS", "0"),
    ],
)
def test_orchestration_out_of_range_values_are_rejected(key: str, value: str) -> None:
    """越界的编排参数必须**启动即失败**，而不是被接受后改变控制流。

    ⚠️ 这四条的危害都是「静默改变行为」而非「报错」：阈值 1.5 让系统句句
    追问，max_chars=0 让快车道彻底失效，max_subagent_calls=0 让多意图输入
    一个子智能体都调不动。它们全都不会崩，只会让系统表现得「有点怪」——
    而「有点怪」是最难被归因的一类线上问题。
    """
    with pytest.raises(Exception):
        load_settings(
            "test",
            environ={f"ALIGO__ORCHESTRATION__{key}": value},
            dotenv=False,
        )


# ==============================================================================
# 四、深合并语义
# ==============================================================================
def test_profile_yaml_only_overrides_its_own_keys() -> None:
    """``{env}.yaml`` 只覆盖它写了的那几项，其余取自 ``base.yaml``。

    没有这条，某个环境档里少写一行就会把整个段落重置成 schema 默认值 ——
    而 schema 默认值与 base.yaml 的字面值**未必相同**，于是「配置漂移」
    以最难察觉的方式发生：只影响那一个环境。
    """
    settings = load_settings("test", environ={}, dotenv=False)

    # test.yaml 显式覆盖了这两项。
    assert settings.llm.use_mock_when_no_key is True
    assert settings.milvus.collection == "aligo_travel_policy_test"
    # 而 test.yaml 没碰的项必须仍来自 base.yaml。
    assert settings.milvus.dimension == 1024
    assert settings.milvus.index_type == "HNSW"
    assert settings.llm.circuit_breaker_failure_threshold == 5


def test_every_profile_loads_successfully() -> None:
    """三档配置都必须能通过严格校验。

    这是一条**廉价的全局闸门**：任何一次「给 schema 加了必填字段、
    却只更新了 base.yaml」的改动，都会在这里被拦住，而不是等到
    ``make up`` 起了 13 个容器之后才发现 prod 档崩了。
    """
    for env in ("dev", "test", "prod"):
        settings = load_settings(env, environ={}, dotenv=False)
        assert settings.app.env == env


# ==============================================================================
# 五、.env.example 自身必须是**可用模板**
# ==============================================================================
#: ``.env.example`` 里「处于生效状态」的 ``ALIGO__`` 键。
#: 行首的 ``#`` 与 ``^ALIGO__`` 都不放行 —— 被注释掉的示例**不会**进入容器环境，
#: 所以它们不是模板的一部分，也不该被本用例要求"能通过校验"。
_ACTIVE_ALIGO_LINE = re.compile(r"^(ALIGO__[A-Z0-9_]+)=(.*)$")


def test_env_example_keys_are_all_accepted_by_the_loader() -> None:
    """``.env.example`` 里**每一个**生效的 ``ALIGO__`` 键都必须被 loader 接受。

    ⚠️ 这条用例挡的是本项目最贵的一类事故：**容器启动即崩循环**。
    loader 对未知键是 ``extra_forbidden``（刻意的，见
    :func:`test_unknown_key_inside_known_section_is_rejected`），
    因此只要 ``.env.example`` 里有一个 schema 不认识的键，
    任何照它抄一份 ``.env`` 的人都会得到一个**起不来**的服务 ——
    而报出来的只是一条 pydantic 校验信息，
    没人会想到「官方示例文件自己写错了」。

    ``.env.example`` 是 ``.gitignore`` 里**唯一被放行**的 ``.env*`` 文件，
    即注定被当作模板使用的那一份。模板里的键必须逐个有效。

    ⚠️ 实现上刻意**不自己解析 schema**，而是把整份文件喂给 :func:`load_settings`
    走真实代码路径。自己写一套「键 → 字段」的解析等于把 loader 的规则再实现一遍，
    两份规则迟早分叉 —— 而分叉的那天，这条用例会**继续通过**，
    只是它守的东西已经不是真的了。这类"守着一个已经不成立的断言的绿灯"
    比没有用例更糟。

    Returns:
        无。不抛异常即通过。
    """
    from tests.conftest import TEST_ENVIRON

    example = Path(__file__).resolve().parents[1] / ".env.example"
    assert example.is_file(), f"找不到 {example} —— 本用例的前提是它存在"

    environ = dict(TEST_ENVIRON)
    found: list[str] = []
    for line in example.read_text(encoding="utf-8").splitlines():
        match = _ACTIVE_ALIGO_LINE.match(line)
        if match is not None:
            environ[match.group(1)] = match.group(2).strip()
            found.append(match.group(1))

    # 先确认"确实读到了东西"：正则写错时会一条都匹配不到，
    # 于是 load_settings 用原始 environ 顺利通过 —— 一个**假绿**。
    # 这条断言把"没读到"与"读了且都对"区分开。
    assert len(found) >= 20, (
        f"只从 .env.example 里认出 {len(found)} 个 ALIGO__ 键，"
        "正则或文件结构可能已变 —— 本用例正在退化成假绿。"
    )

    settings = load_settings("test", environ=environ, dotenv=False)
    assert isinstance(settings, Settings)
