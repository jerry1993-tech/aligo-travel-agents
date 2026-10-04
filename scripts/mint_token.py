#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""签发一枚**开发用**的 JWT，供本地联调 ``Authorization: Bearer`` 通道。

==============================================================================
它解决的是什么问题
==============================================================================
    本项目鉴权有两条通道（见 ``src/server/middleware/auth.py``）：

        · ``X-User-ID`` 直连 —— 本地开发默认走这条，随便填谁都行；
        · ``Authorization: Bearer <JWT>`` —— 校验签名/有效期/受众/签发者。

    第二条通道在真实环境里由 IdP（Auth0 / Keycloak / 阿里云 IDaaS…）签发 token，
    本机没有 IdP。于是「本地根本没法验证 JWT 通道是否真的能用」——
    而那正是最需要在本机验的一段代码：它一旦写错，**自签的测试 token
    完全正常，只有接上真实 IdP 才炸**。

    本项目就踩过这样一个坑：``_decode`` 里没显式关掉 aud 校验，
    于是任何**带 aud** 的 token 一律被拒 —— 而现实里的 IdP 几乎都签 aud。
    本脚本存在的意义之一，就是让这类问题在**本机**就能被发现。

==============================================================================
它凭什么值得信任：用「验证方自己的代码」回验
==============================================================================
    一个自己写签名、自己写校验的脚本是**循环论证** —— 两边一起错，
    它照样打印「验证通过」。

    所以本脚本签发之后，会拿**:class:`AuthMiddleware` 的真实 ``_decode``**
    把 token 解回来（见 :func:`verify_with_middleware`）。走的就是运行时
    校验请求的那段代码。签名方与验证方在**本机**对不上，脚本立刻失败 ——
    而不是等到联调时才以一个 401 的形式出现。

==============================================================================
四条安全约束（都是硬性的）
==============================================================================
    1. **生产环境拒签**。``app.env == "prod"`` 直接退出，没有 ``--force``。
       一个能签发任意身份 token 的脚本放在生产环境里就是一台后门生成器，
       而它通常是被「顺手拷到服务器上跑一下」带过去的。
    2. **绝不打印密钥**。输出里只出现密钥的 **sha256 指纹前缀**，
       用于跨机器比对「两边用的是不是同一个密钥」。
       （``jwt_secret`` 可能同时是生产密钥，打进终端就等于泄漏到
       shell history、CI 日志与截图里。）
    3. **``jwt_enabled=false`` 时明确警告**。此时中间件根本不看
       ``Authorization`` 头，签出来的 token 会被当成没带凭据 ——
       症状是「token 明明是对的却 401」，而原因在配置里。
    4. **保留身份拒签**。``--user aligo-system`` 直接退出，同样没有
       ``--force``。它是系统凭据的属主（``src/llm/identity.py:42``），
       中间件对**声明它**的请求一律 403（三条通道都拦）。
       ⚠️ 本脚本若不拦，签出来的是一枚「签名有效但注定被拒」的 token ——
       而脚本的自检只跑 ``_decode``（只管解码与签名），跑不到解码**之后**
       那道身份判断，于是它会打印「回验：通过」再附一条必然 403 的 curl。
       那正是本脚本最想消灭的那类假绿灯，所以这里必须自己拦一道。

==============================================================================
用法
==============================================================================
        # 签发 alice 的 token（默认 1 小时）
        python scripts/mint_token.py --user alice

        # 指定时长与受众，并打印一条可直接粘贴的 curl
        python scripts/mint_token.py --user bob --ttl 300 --aud aligo-api

        # 只想要裸 token（给 shell 变量赋值）
        TOKEN=$(python scripts/mint_token.py --user alice --raw)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

# 允许以 `python scripts/mint_token.py` 直接运行（此时 sys.path[0] 是 scripts/，
# 而不是仓库根，`import src` 会失败）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Settings, load_settings  # noqa: E402
from src.llm.identity import is_reserved_user_id  # noqa: E402

#: 默认有效期（秒）。1 小时够一次联调，又不至于让一枚泄漏的 token 长期可用。
DEFAULT_TTL_SECONDS = 3600

#: 指纹取前多少位十六进制。12 位 = 48 bit，碰撞概率在本用途下可忽略，
#: 而长度短到能一眼比对。
FINGERPRINT_CHARS = 12

#: 拒绝签发的环境。刻意写成常量而不是内联字符串：它是一条安全边界，
#: 应当只有一处定义、且能被测试引用。
FORBIDDEN_ENVS = frozenset({"prod"})


def fingerprint(secret: str) -> str:
    """返回密钥的 sha256 指纹前缀。

    ⚠️ **绝不**改成返回密钥本身或它的前缀/后缀。指纹的单向性正是它的用途：
    可以在两台机器之间比对「是不是同一个密钥」，而任何一方都不必交出密钥。

    Args:
        secret (`str`): 密钥原文。

    Returns:
        `str`: 形如 ``a1b2c3d4e5f6`` 的指纹前缀。
    """
    return hashlib.sha256(secret.encode("utf-8")).hexdigest()[:FINGERPRINT_CHARS]


def build_claims(
    user_id: str,
    *,
    ttl_seconds: int,
    now: int,
    audience: str = "",
    issuer: str = "",
) -> dict[str, Any]:
    """构造 JWT 的载荷。

    ``exp`` 与 ``sub`` 是**必填**的 —— 中间件的 ``_decode`` 把
    ``options["require"] = ["exp", "sub"]`` 写死了。少了 ``exp``，
    token 永不过期（一次泄漏就是永久后门）；少了 ``sub``，放行等于匿名。
    本脚本因此不提供「不设过期时间」的选项。

    ``aud`` / ``iss`` 只在配置非空时才写进载荷：中间件的规则是
    「配置为空 ⇒ 不校验」，那么签一个带 aud 的 token 虽然也能通过
    （因为校验被显式关掉了），但会让这枚 token 与生产环境的行为不一致 ——
    本脚本的目标恰恰是**复现生产的行为**。

    Args:
        user_id (`str`): 要签发的身份，会写进 ``sub``。
        ttl_seconds (`int`): 有效期秒数。
        now (`int`): 当前 Unix 时间戳（由调用方传入，便于测试）。
        audience (`str`): ``aud``，空串则不写入。
        issuer (`str`): ``iss``，空串则不写入。

    Returns:
        `dict`: 载荷。
    """
    claims: dict[str, Any] = {
        "sub": user_id,
        # iat（签发时间）不是校验项，但排障时「这枚 token 是什么时候签的」
        # 往往比 exp 更有用 —— 一枚 exp 在未来却 iat 在昨天的 token
        # 通常意味着机器的时钟有问题。
        "iat": now,
        "exp": now + ttl_seconds,
    }
    if audience:
        claims["aud"] = audience
    if issuer:
        claims["iss"] = issuer
    return claims


class _JwtEnabledAuth:
    """``settings.auth`` 的只读视图：``jwt_enabled`` 恒为 ``True``。

    存在的理由是一件很容易忽略的事：

        :class:`~src.server.middleware.AuthMiddleware` 在
        ``jwt_enabled=False`` 时**根本不会导入 PyJWT**（``self._jwt`` 保持
        ``None``，见它的 ``__init__``）。于是「关了 JWT 却还想回验一枚
        token」会以 ``AttributeError: 'NoneType' object has no attribute
        'decode'`` 的形式失败 —— 而那看起来像是密钥或算法不对。

    而本脚本要回答的问题恰恰是：**「开了 JWT 之后，这枚 token 能不能过？」**
    所以这里强制把它当成开着来验。

    ⚠️ 用 ``__getattr__`` 转发而不是复制配置对象：
        转发对 pydantic 模型与普通 dataclass 都成立，也不必关心目标对象
        是不是 frozen（``model_copy`` / ``dataclasses.replace`` 各是一套写法，
        两套都要写才能同时兼容）。``_auth`` 存在 ``__dict__`` 里，
        因此 ``__getattr__`` 不会递归。
    """

    def __init__(self, auth: Any) -> None:
        """记录真实配置。

        Args:
            auth (`Any`): 真实的 ``settings.auth``。
        """
        self._auth = auth

    def __getattr__(self, name: str) -> Any:
        """除 ``jwt_enabled`` 外一律转发给真实配置。

        Args:
            name (`str`): 属性名。

        Returns:
            `Any`: ``jwt_enabled`` 恒为 True，其余取真实值。
        """
        if name == "jwt_enabled":
            return True
        return getattr(self._auth, name)


class _JwtEnabledSettings:
    """``settings`` 的只读视图，只满足 ``AuthMiddleware.__init__`` 的需求。"""

    def __init__(self, settings: Settings) -> None:
        """包装真实配置。

        Args:
            settings (`Settings`): 真实配置。
        """
        self.auth = _JwtEnabledAuth(settings.auth)


def verify_with_middleware(settings: Settings, token: str) -> str | None:
    """用**运行时那个** ``AuthMiddleware._decode`` 回验 token。

    ★ 这是本脚本唯一有价值的断言。自己签、自己验是循环论证；
    只有走运行时的校验代码，才能证明「签出来的东西服务端真的收」。

    ⚠️ 这里直接调用一个私有方法 ``_decode``。这是**刻意的**：
        用公开路径（发一个真请求）需要先起服务、建凭据、建智能体……
        为了一次校验拉起一整套环境不现实。而 ``_decode`` 恰恰是
        所有 JWT 校验逻辑的唯一入口，调用它的收益远大于「不碰私有成员」
        这条形式上的洁癖。代价是：该方法若改名，本脚本会以
        ``AttributeError`` 失败 —— 而不是静默地不校验，这是可以接受的。

    Args:
        settings (`Settings`): 用于构造中间件的配置。
        token (`str`): 待回验的 token。

    Returns:
        `str | None`: 解析出的 ``sub``；校验失败为 ``None``。
    """
    # 第一个参数是下游 ASGI 应用。``AuthMiddleware.__init__`` 只存配置、
    # 不碰 app，因此传 None 是安全的 —— 而且这里也永远不会发起请求。
    from src.server.middleware import AuthMiddleware

    middleware = AuthMiddleware(None, settings=_JwtEnabledSettings(settings))
    return middleware._decode(token)  # noqa: SLF001 - 见上面的说明


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv (`list[str] | None`): 参数列表；``None`` 时取 ``sys.argv[1:]``。

    Returns:
        `argparse.Namespace`: 解析结果。
    """
    parser = argparse.ArgumentParser(
        prog="mint_token.py",
        description="签发一枚开发用的 JWT（生产环境拒签）。",
    )
    parser.add_argument(
        "--user",
        required=True,
        help="身份标识，会写进 sub。例如 alice。",
    )
    parser.add_argument(
        "--ttl",
        type=int,
        default=DEFAULT_TTL_SECONDS,
        help=f"有效期秒数（默认 {DEFAULT_TTL_SECONDS}）。",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="要读取的配置档（dev / test / prod）。默认取 ALIGO__APP__ENV，再退回 dev。",
    )
    parser.add_argument(
        "--aud",
        default=None,
        help="覆盖 config 里的 jwt_audience。默认用配置值（空串则不写入 aud）。",
    )
    parser.add_argument(
        "--iss",
        default=None,
        help="覆盖 config 里的 jwt_issuer。默认用配置值。",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="只打印 token 本身，便于 `TOKEN=$(...)` 赋值。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """脚本入口。

    Args:
        argv (`list[str] | None`): 命令行参数。

    Returns:
        `int`: 进程退出码。0 成功，1 配置不满足，2 参数不合法。
    """
    args = _parse_args(argv)

    if args.ttl <= 0:
        print("--ttl 必须为正数。", file=sys.stderr)
        return 2

    # ---- 约束四：保留身份拒签 ------------------------------------------------
    # ⚠️ 编号按模块文档里的清单，而**执行**排在「约束一」之前 —— 它和
    #    ``--ttl`` 一样不需要读配置，于是越早失败越好：身份本身就不合法时，
    #    没有理由先把 .env 读一遍、再拿一句「配置加载失败」把真正的原因盖住。
    #
    # ⚠️ 先 strip 再判，而且**沿用 strip 后的值**去签。两个理由：
    #     · ``is_reserved_user_id`` 的判据是**逐字节相等**（见
    #       src/llm/identity.py 的说明），而服务端 ``_decode`` 会把 ``sub``
    #       去掉首尾空白 —— 不 strip 就会出现「签发时是 ' aligo-system'，
    #       判据没命中，服务端解出来却是 'aligo-system'」这条缝；
    #     · 不 strip 的话，``--user " alice "`` 会签出一个带空格的 sub，
    #       而下面的自检拿它去比 ``_decode`` 的结果（已 strip），报的是
    #       「签发的 sub 与解出来的不一致」—— 一句把人引向算法/密钥的错话。
    user_id = args.user.strip()
    if not user_id:
        print("--user 不能为空。", file=sys.stderr)
        return 2
    if is_reserved_user_id(user_id):
        print(
            f"拒绝为 {user_id!r} 签发 token：它是**系统保留身份**"
            "（系统凭据的属主）。\n"
            "中间件对声明它的请求一律 403（X-User-ID / JWT 的 sub / 匿名"
            "三条通道都拦），\n"
            "所以这枚 token 签出来也只会被拒 —— 它从来不是给客户端用的身份。\n"
            "换一个名字（例如 alice）即可。",
            file=sys.stderr,
        )
        return 2

    try:
        settings = load_settings(args.env)
    except ValueError as exc:
        print(f"配置加载失败：{exc}", file=sys.stderr)
        return 1

    auth = settings.auth

    # ---- 约束一：生产环境拒签 ------------------------------------------------
    if settings.app.env in FORBIDDEN_ENVS:
        print(
            f"拒绝在 env={settings.app.env!r} 下签发 token。\n"
            f"一个能签发任意身份 token 的脚本放在生产环境里就是后门生成器。\n"
            f"如果确实要联调生产，请在可信机器上用真实的 IdP。",
            file=sys.stderr,
        )
        return 1

    # ---- 约束三：通道没开就明确警告 ------------------------------------------
    if not auth.jwt_enabled:
        print(
            "⚠️  当前配置 auth.jwt_enabled=false，中间件**不会**解析 Authorization 头。\n"
            "    这枚 token 发出去会被当成「没带凭据」而返回 401。\n"
            "    要联调 JWT 通道，请先设置 ALIGO__AUTH__JWT_ENABLED=true"
            "（并确保 ALIGO__AUTH__JWT_SECRET 非空）。",
            file=sys.stderr,
        )

    if not auth.jwt_secret.strip():
        # 走到这里说明 jwt_enabled 为假（为真时配置层的交叉校验会先拦下，
        # 见 schema.py 的 _jwt_secret_required_when_enabled）。
        print(
            "auth.jwt_secret 为空，没有可用于签名的密钥。\n"
            "请设置 ALIGO__AUTH__JWT_SECRET（至少 32 字节，"
            "短密钥会被 PyJWT 判为不安全长度）。",
            file=sys.stderr,
        )
        return 1

    try:
        import jwt as pyjwt
    except ImportError:
        print(
            "未安装 PyJWT。请执行 `pip install PyJWT`（版本已在 requirements.txt 固定）。",
            file=sys.stderr,
        )
        return 1

    # 受众/签发者的取值优先级：命令行 > 配置。
    # 命令行覆盖是为了能拿**同一份密钥**去试不同的 aud 配置，
    # 而不必改 .env —— 那正是排查「aud 校验写错」时要做的事。
    audience = auth.jwt_audience if args.aud is None else args.aud
    issuer = auth.jwt_issuer if args.iss is None else args.iss

    # ⚠️ 用 time.time() 而不是 time.monotonic()：JWT 的 exp/iat 是
    #    **绝对时间戳**，必须与实际时钟对齐，否则服务端算出来是「已过期」。
    import time

    claims = build_claims(
        user_id,
        ttl_seconds=args.ttl,
        now=int(time.time()),
        audience=audience.strip(),
        issuer=issuer.strip(),
    )

    token = pyjwt.encode(claims, auth.jwt_secret, algorithm=auth.jwt_algorithm)

    # ---- 回验：用运行时的校验代码解开它 --------------------------------------
    resolved = verify_with_middleware(settings, token)
    if resolved != user_id:
        print(
            f"自检失败：签发的 sub={user_id!r}，但 AuthMiddleware._decode 解出来是 "
            f"{resolved!r}。\n"
            f"这说明签发方与校验方不一致，token 发出去必然被拒。"
            f"请检查 jwt_algorithm（当前 {auth.jwt_algorithm!r}）等配置。",
            file=sys.stderr,
        )
        return 1

    if args.raw:
        print(token)
        return 0

    # ---- 输出 ----------------------------------------------------------------
    # ⚠️ 这里**没有**、也永远不该有 secret 原文。只有指纹。
    print(f"env          : {settings.app.env}")
    print(f"算法          : {auth.jwt_algorithm}")
    print(f"密钥指纹      : {fingerprint(auth.jwt_secret)}  (sha256 前 {FINGERPRINT_CHARS} 位)")
    print(f"sub          : {claims['sub']}")
    print(f"aud          : {claims.get('aud', '(未写入)')}")
    print(f"iss          : {claims.get('iss', '(未写入)')}")
    print(f"有效期        : {args.ttl}s")
    print(f"回验          : 通过（用 AuthMiddleware._decode 实测）")
    print()
    print("token:")
    print(token)
    print()
    # 直接给出可粘贴的命令：省掉一次「Bearer 与 token 之间要不要空格」
    # 之类的低级试错，也让读者一眼看到它是**请求头**而不是查询参数。
    print("示例：")
    print(f'  curl -H "Authorization: Bearer {token}" \\')
    print('       -H "Content-Type: application/json" \\')
    print("       http://127.0.0.1:8000/api/v1/me")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
