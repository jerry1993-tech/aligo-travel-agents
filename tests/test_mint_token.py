# -*- coding: utf-8 -*-
"""``scripts/mint_token.py`` 的用例。

==============================================================================
为什么一个「开发脚本」也值得写测试
==============================================================================
    它不是普通的辅助脚本，而是**两条安全边界的守卫**：

        · 生产环境拒签 —— 这条边界一旦失效，仓库里就多了一台
          「签发任意身份 token」的机器；
        · 绝不打印密钥 —— 这条边界失效的症状是密钥进了 shell history、
          CI 日志和别人的截图，且**当时不会有任何异常**。

    这两条都属于「失效时悄无声息」的类型，正是测试该覆盖的对象。

==============================================================================
最重要的一条：签发方必须与服务端的校验方一致
==============================================================================
    :func:`~scripts.mint_token.verify_with_middleware` 是本脚本自带的回验，
    它在真实运行里保证「签出来的一定能验过」。但**自带的**回验也可能被改坏，
    例如有人把它换成一句 `return args.user`。

    因此这里不信任脚本自己的回验，而是**另起一份**
    :class:`~src.server.middleware.AuthMiddleware`，用它的 ``_decode``
    再解一次。这样「脚本自检通过」与「中间件真的认」就是两个独立的事实。

    ⚠️ 这也是本项目那个 PyJWT ``aud`` 坑的回归测试：
        中间件在不配置受众时若不显式 ``verify_aud=False``，
        任何**带 aud** 的 token 都会被拒 —— 而现实里的 IdP 都签 aud。
        见 :func:`test_minted_token_with_audience_passes_the_real_middleware`。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import pytest

from scripts import mint_token

#: 测试用密钥。必须 ≥32 字节 —— 更短会让 PyJWT 发
#: ``InsecureKeyLengthWarning``，而在本文件里那条告警很容易被误读成
#: 「密钥错了」。长度足够时，任何告警都一定是真问题。
TEST_SECRET = "mint-token-unit-test-secret-0123456789abcdef"

#: 另一个密钥，用于验证「指纹能区分不同密钥」。
OTHER_SECRET = "a-completely-different-secret-0123456789abcdef"


# ==============================================================================
# 替身配置
# ==============================================================================
@dataclass
class _Auth:
    """``settings.auth`` 的替身。

    刻意用 dataclass 而不是 ``Mock``：``Mock`` 对**任何**属性都返回一个
    子对象，于是 ``auth.jwt_secret`` 拿到的是 ``Mock`` 而不是字符串，
    签名时会以一句莫名其妙的 ``TypeError`` 失败。真实字段名的替身
    能让类型错误在正确的位置暴露出来。
    """

    require_user_header: bool = True
    jwt_enabled: bool = True
    jwt_secret: str = TEST_SECRET
    jwt_algorithm: str = "HS256"
    jwt_audience: str = ""
    jwt_issuer: str = ""


@dataclass
class _App:
    """``settings.app`` 的替身。"""

    env: str = "dev"


@dataclass
class _Settings:
    """``Settings`` 的替身。"""

    auth: _Auth = field(default_factory=_Auth)
    app: _App = field(default_factory=_App)


@pytest.fixture
def fake_settings(monkeypatch: pytest.MonkeyPatch):
    """把 ``mint_token.load_settings`` 换成可配置的替身。

    ⚠️ 补丁打在 ``mint_token`` 的命名空间上，而不是 ``src.config``：
        脚本用的是 ``from src.config import load_settings``，名字在
        **导入时**就绑定到了脚本模块里。去补丁 ``src.config`` 是没用的 ——
        那种写法会静默地不生效，用例照跑，只是用的还是真配置。

    Returns:
        `Callable[..., _Settings]`: 调用它并传入 ``auth`` / ``app`` 覆写，
        返回构造好的替身配置。
    """

    def _install(auth: _Auth | None = None, app: _App | None = None) -> _Settings:
        settings = _Settings(auth=auth or _Auth(), app=app or _App())
        monkeypatch.setattr(mint_token, "load_settings", lambda *a, **k: settings)
        return settings

    return _install


# ==============================================================================
# 一、纯函数：指纹与载荷
# ==============================================================================
def test_fingerprint_is_deterministic_and_never_leaks_the_secret() -> None:
    """指纹可复现，且**不含**密钥原文。

    指纹的用途是「跨机器比对两边用的是不是同一个密钥」。它必须单向 ——
    一旦有人图省事改成返回 ``secret[:12]``，功能看起来完全一样
    （可比对、长度接近），但每次运行都在往终端里吐半个密钥。

    这条用例就是拦住那个改动的。
    """
    first = mint_token.fingerprint(TEST_SECRET)
    second = mint_token.fingerprint(TEST_SECRET)

    assert first == second, "同一个密钥两次算出的指纹必须一致，否则无法比对。"
    assert len(first) == mint_token.FINGERPRINT_CHARS
    assert TEST_SECRET[: len(first)] != first, "指纹不得是密钥的前缀截断。"
    assert TEST_SECRET not in first
    assert first not in TEST_SECRET


def test_fingerprint_differs_for_different_secrets() -> None:
    """不同密钥的指纹必须不同 —— 否则这条比对毫无鉴别力。"""
    assert mint_token.fingerprint(TEST_SECRET) != mint_token.fingerprint(OTHER_SECRET)


def test_build_claims_always_sets_exp_and_sub() -> None:
    """``exp`` 与 ``sub`` 是必填的，脚本不提供「不设过期」的选项。

    中间件的 ``_decode`` 把 ``options["require"] = ["exp", "sub"]`` 写死了：
    没有 ``exp`` 的 token 永不过期（一次泄漏就是永久后门），
    没有 ``sub`` 的 token 没有身份可言。
    """
    claims = mint_token.build_claims("alice", ttl_seconds=600, now=1_000_000)

    assert claims["sub"] == "alice"
    assert claims["exp"] == 1_000_000 + 600
    assert claims["iat"] == 1_000_000


def test_build_claims_omits_aud_and_iss_when_empty() -> None:
    """受众/签发者为空时**不写入**载荷。

    ⚠️ 与中间件的规则保持一致：中间件在配置为空时把校验整个关掉
    （``verify_aud=False``）。此时若这边仍写入 ``aud``，签出来的 token
    就与生产环境的行为不一致 —— 而本脚本存在的意义恰恰是复现生产行为。
    """
    claims = mint_token.build_claims("alice", ttl_seconds=60, now=0)

    assert "aud" not in claims
    assert "iss" not in claims


def test_build_claims_includes_aud_and_iss_when_configured() -> None:
    """配置了受众/签发者时写入载荷。"""
    claims = mint_token.build_claims(
        "alice",
        ttl_seconds=60,
        now=0,
        audience="aligo-api",
        issuer="aligo-dev",
    )

    assert claims["aud"] == "aligo-api"
    assert claims["iss"] == "aligo-dev"


# ==============================================================================
# 二、安全边界：生产拒签、密钥不外泄
# ==============================================================================
def test_prod_environment_is_refused(fake_settings, capsys: pytest.CaptureFixture) -> None:
    """``env == prod`` 时**拒签**，并且不产生任何 token。

    ★ 刻意**没有** ``--force`` 逃生口。一个「能签发任意身份 token」的脚本
    被拷到生产机器上跑一下，就是从内部开了一道后门；而这类操作往往
    披着「我就联调一下」的外衣。没有逃生口，就没有「顺手」的可能。
    """
    fake_settings(app=_App(env="prod"))

    exit_code = mint_token.main(["--user", "alice"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "拒绝" in captured.err
    # 关键：**标准输出**里不能有 token。哪怕退出码是对的，
    # 只要 token 被打印出来了，在多机部署 / CI 里它就已经被记录下来了。
    assert captured.out.strip() == ""


def test_missing_secret_is_refused(fake_settings, capsys: pytest.CaptureFixture) -> None:
    """没有密钥就拒绝，而不是签一个空密钥的 token。

    空 HMAC 密钥意味着任何人都能伪造合法 token —— 这是漏洞而非疏漏，
    必须硬失败。配置层（``AuthSettings`` 的交叉校验）在 ``jwt_enabled=true``
    时已经拦了一道，这里覆盖的是 ``jwt_enabled=false`` 且密钥也为空的情形。
    """
    fake_settings(auth=_Auth(jwt_enabled=False, jwt_secret=""))

    exit_code = mint_token.main(["--user", "alice"])

    assert exit_code == 1
    assert "jwt_secret" in capsys.readouterr().err


def test_missing_pyjwt_is_refused(
    fake_settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """未安装 PyJWT 时给出可执行的补救指令，而不是一句 ImportError。

    ⚠️ ``sys.modules["jwt"] = None`` 是让 ``import jwt`` 抛 ImportError
        的标准做法：导入系统遇到 ``None`` 会直接判定为「不可导入」，
        比 mock 掉 ``builtins.__import__`` 干净得多。
        用 ``monkeypatch.setitem`` 保证用例结束后自动还原。
    """
    fake_settings()
    monkeypatch.setitem(sys.modules, "jwt", None)

    exit_code = mint_token.main(["--user", "alice"])

    assert exit_code == 1
    assert "PyJWT" in capsys.readouterr().err


def test_non_positive_ttl_is_refused(fake_settings, capsys: pytest.CaptureFixture) -> None:
    """``--ttl`` 必须为正 —— 否则签出来的是「一诞生就过期」的 token。"""
    fake_settings()

    exit_code = mint_token.main(["--user", "alice", "--ttl", "0"])

    assert exit_code == 2
    assert "--ttl" in capsys.readouterr().err


def test_the_reserved_identity_is_refused(fake_settings, capsys: pytest.CaptureFixture) -> None:
    """``--user aligo-system`` 拒签，且不产生任何 token。

    ★ 它签出来的 token **签名是有效的** —— 中间件拦它靠的不是签名，
    而是解码之后那道身份判断（见 ``auth.py::_reject_reserved_identity``），
    所以本脚本的自检（只跑 ``_decode``）**抓不到**这种情况，会照常打印
    「回验：通过」。一个注定 403 的 token 配上一条注定 403 的 curl，
    比不给提示更糟：它会让人先怀疑中间件写错了。
    """
    fake_settings()

    exit_code = mint_token.main(["--user", "aligo-system"])

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "保留身份" in captured.err
    # 与生产拒签同一条纪律：标准输出里不能有 token。
    assert captured.out.strip() == ""


def test_a_padded_reserved_identity_is_refused_too(
    fake_settings, capsys: pytest.CaptureFixture
) -> None:
    """``" aligo-system "`` 同样拒签 —— 判据必须在 ``strip`` **之后**。

    ⚠️ 这条补的是上面那条抓不到的缝：``is_reserved_user_id`` 的判据是
    **逐字节相等**（见 ``src/llm/identity.py``，它要求调用方先 strip），
    而服务端 ``_decode`` 会把 ``sub`` 去掉首尾空白。若脚本先比再 strip
    （或者干脆不 strip），``" aligo-system"`` 会绕过判据签出去，
    到服务端却被解成 ``aligo-system`` 而命中保留身份 —— 判据与执行
    错开了一个 ``strip`` 的距离，两边各自看起来都对。
    """
    fake_settings()

    exit_code = mint_token.main(["--user", "  aligo-system  "])

    assert exit_code == 2, "带空白的保留身份绕过了判据 —— strip 的顺序反了"
    assert "保留身份" in capsys.readouterr().err


def test_an_empty_user_is_refused(fake_settings, capsys: pytest.CaptureFixture) -> None:
    """``--user ""`` / 只有空白：拒签，而不是签一个空 sub 的 token。

    ⚠️ 空 ``sub`` 不是「保留身份」，所以它逃过上面那道判据 —— 但不该
    走到签名那一步：服务端 ``_decode`` 对空 ``sub`` 一律返回 ``None``，
    本脚本的自检会失败并报「签发的 sub=''，但解出来是 None」，
    而这句话把人引向算法/密钥，与真正的原因（忘了填 --user）无关。
    """
    fake_settings()

    exit_code = mint_token.main(["--user", "   "])

    assert exit_code == 2
    assert "--user" in capsys.readouterr().err


def test_surrounding_whitespace_is_stripped_from_the_signed_sub(
    fake_settings, capsys: pytest.CaptureFixture
) -> None:
    """普通身份带空白时，签进去的是 strip **之后**的值。

    ⚠️ 与上面那条是同一件事的两半：判据用了 strip 后的值，签名也必须用
    同一个值。否则 ``--user " alice "`` 签出 ``sub=" alice "``，而自检拿
    ``_decode`` 的结果（已 strip）去比，报「签发的 sub 与解出来的不一致」——
    又是一句把人引向算法/密钥的错话。
    """
    fake_settings()

    exit_code = mint_token.main(["--user", " alice ", "--raw"])
    token = capsys.readouterr().out.strip()

    assert exit_code == 0
    import jwt as pyjwt

    claims = pyjwt.decode(
        token,
        TEST_SECRET,
        algorithms=["HS256"],
        # 与中间件一致：没配受众就不校验受众（见本文件模块文档的 aud 坑）。
        options={"verify_aud": False},
    )
    assert claims["sub"] == "alice"


def test_secret_never_appears_in_any_output(fake_settings, capsys: pytest.CaptureFixture) -> None:
    """★ 完整输出（stdout + stderr）里都不得出现密钥原文。

    这是最容易失效的一条 —— 加一行 `print(f"用密钥 {secret} 签名")` 就能
    把它破坏掉，而那一行在 code review 里看起来像是「方便排障」。
    """
    fake_settings(auth=_Auth(jwt_secret=TEST_SECRET))

    assert mint_token.main(["--user", "alice"]) == 0

    captured = capsys.readouterr()
    assert TEST_SECRET not in captured.out
    assert TEST_SECRET not in captured.err
    # 指纹应当在，否则上面那条断言可能只是「什么都没打印」。
    assert mint_token.fingerprint(TEST_SECRET) in captured.out


# ==============================================================================
# 三、签出来的 token 必须被**真的**中间件认
# ==============================================================================
def _real_decode(settings: _Settings, token: str) -> str | None:
    """用真实的 ``AuthMiddleware._decode`` 解一次 token。

    ⚠️ 刻意不复用 ``mint_token.verify_with_middleware``：那是**脚本自带的**
        回验，脚本自己改了它，用它来验证等于自证。这里另起一份中间件，
        这样「脚本自检通过」与「中间件真的认」是两个独立的事实。

    Args:
        settings (`_Settings`): 替身配置（字段与真实 ``Settings.auth`` 一致）。
        token (`str`): 待解析的 token。

    Returns:
        `str | None`: 解析出的 ``sub``；失败为 ``None``。
    """
    from src.server.middleware import AuthMiddleware

    return AuthMiddleware(None, settings=settings)._decode(token)  # noqa: SLF001


def test_minted_token_passes_the_real_middleware(fake_settings, capsys: pytest.CaptureFixture) -> None:
    """``--raw`` 签出的 token 能被真实中间件解出正确的 ``sub``。

    这是「签发方 == 校验方」的端到端证明。
    """
    settings = fake_settings(auth=_Auth(jwt_secret=TEST_SECRET))

    assert mint_token.main(["--user", "alice", "--raw"]) == 0
    token = capsys.readouterr().out.strip()

    assert _real_decode(settings, token) == "alice"


def test_minted_token_with_audience_passes_the_real_middleware(
    fake_settings,
    capsys: pytest.CaptureFixture,
) -> None:
    """★ 回归测试：**带了 ``aud`` 的 token 也必须能通过**。

    钉的是这个坑：PyJWT 在「token 里有 ``aud`` 而调用方没传 ``audience``
    参数」时抛 ``InvalidAudienceError`` —— 而不是「跳过校验」。
    于是「不配置受众」的默认配置会与字面意思相反：它拒绝了**所有**
    带 aud 的 token，而现实里的 IdP（Auth0 / Keycloak / 阿里云 IDaaS）
    几乎都签 aud。默认配置下**一个真实 token 都过不了**。

    隐蔽之处在于：自签的不带 aud 的测试 token 完全正常
    （见上一条用例），只有接上真实 IdP 才炸 —— 而那时已经在联调环境了。

    这里走的是「配置里没有受众、但 token 里有 aud」这条路径，
    正是修复前会失败的那一条。
    """
    settings = fake_settings(auth=_Auth(jwt_secret=TEST_SECRET, jwt_audience=""))

    # 命令行 --aud 强制写入 aud，而配置里的 jwt_audience 是空串。
    assert mint_token.main(["--user", "bob", "--aud", "some-idp-audience", "--raw"]) == 0
    token = capsys.readouterr().out.strip()

    assert _real_decode(settings, token) == "bob", (
        "带 aud 的 token 被拒了 —— 中间件在不配置受众时没有关掉 aud 校验。"
    )


def test_configured_audience_round_trips(fake_settings, capsys: pytest.CaptureFixture) -> None:
    """配置了受众时，签发与校验用的是同一个值，能对上。"""
    settings = fake_settings(auth=_Auth(jwt_secret=TEST_SECRET, jwt_audience="aligo-api"))

    assert mint_token.main(["--user", "carol", "--raw"]) == 0
    token = capsys.readouterr().out.strip()

    assert _real_decode(settings, token) == "carol"


def test_wrong_secret_is_rejected_by_the_real_middleware(
    fake_settings,
    capsys: pytest.CaptureFixture,
) -> None:
    """用 A 密钥签的 token 过不了 B 密钥的中间件。

    ⚠️ 必须有这条「反向」用例：上两条只断言「能通过」，
        一个 `_decode` 里直接 `return payload["sub"]` 而不验签的实现
        同样能让它们全绿。
    """
    fake_settings(auth=_Auth(jwt_secret=TEST_SECRET))

    assert mint_token.main(["--user", "dave", "--raw"]) == 0
    token = capsys.readouterr().out.strip()

    assert _real_decode(fake_settings(auth=_Auth(jwt_secret=OTHER_SECRET)), token) is None


# ==============================================================================
# 四、配置不开时的提示
# ==============================================================================
def test_jwt_disabled_still_warns_but_signs(
    fake_settings,
    capsys: pytest.CaptureFixture,
) -> None:
    """``jwt_enabled=false`` 时警告，但仍签发（退出码 0）。

    为什么不直接拒绝：这个组合是**合法**的 —— 先在本地拿一枚 token，
    再去改配置开启 JWT，是正常的联调顺序。脚本的职责是把
    「这枚 token 现在会被当成没带凭据」这件事说清楚，
    而不是替调用方决定。

    警告本身很重要：否则症状是「token 明明是对的却 401」，
    而原因（中间件压根不看 Authorization 头）在配置里，不在 token 上。
    """
    fake_settings(auth=_Auth(jwt_enabled=False, jwt_secret=TEST_SECRET))

    exit_code = mint_token.main(["--user", "alice", "--raw"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "jwt_enabled" in captured.err
    assert captured.out.strip(), "--raw 仍然要输出 token。"
