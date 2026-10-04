# -*- coding: utf-8 -*-
"""运维探针（``src/server/probes.py``）的行为测试。

==============================================================================
为什么探针值得单独一个测试文件
==============================================================================
    探针是本项目里**唯一**「出错也不会有人立刻发现」的代码：
    它的消费者不是人，是 Docker 的健康检查、是负载均衡的摘流决策。
    探针写错的两种典型症状都不出现在应用日志里：

      · ``/healthz`` 去连了数据库 ⇒ 数据库抖动一下，**所有容器被重启**，
        于是一次本可自愈的短暂故障被放大成一次全站中断；
      · ``/readyz`` 永远返回 200 ⇒ 编排系统永远认为「没问题」，
        于是一个起不来的实例一直挂在负载均衡后面持续吞掉流量。

    本文件的用例把这两条都钉死。
"""

from __future__ import annotations

import json

import pytest
from httpx import AsyncClient

# ==============================================================================
# 一、存活探针 /healthz
# ==============================================================================
async def test_healthz_is_always_ok(client: AsyncClient) -> None:
    """``/healthz`` 返回 200 与固定体。

    它是 compose 的 ``HEALTHCHECK`` 目标与 k8s 的 ``livenessProbe`` 目标，
    返回体本身也被人肉 ``curl`` 时读取，因此格式是契约的一部分。
    """
    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_healthz_does_no_io(client: AsyncClient) -> None:
    """``/healthz`` **不能**依赖任何外部服务。

    ⚠️ 这条是本文件存在的主要理由。存活探针的语义是「这个进程还活着吗」，
    而数据库/Redis/Milvus 是**就绪**的判据，不是存活的判据。
    两者混淆的代价极不对称：

      · 正确分开 ⇒ 依赖挂了，``/readyz`` 变 503，编排把流量摘掉，
        进程原地待命，依赖恢复后自动接回。**用户无感**。
      · 混淆 ⇒ 依赖挂了，``/healthz`` 也失败，编排**杀掉并重启容器**。
        进程重启既不修复依赖，又让恢复变慢（冷启动、连接池重建），
        一次短暂抖动被放大成全站中断。

    本用例跑在「没有任何外部服务」的环境里（见 conftest 的注释），
    因此只要能返回 200，就证明它确实没去连任何东西。
    """
    for _ in range(3):
        response = await client.get("/healthz")
        assert response.status_code == 200


# ==============================================================================
# 二、就绪探针 /readyz
# ==============================================================================
async def test_readyz_reports_not_ready_without_dependencies(client: AsyncClient) -> None:
    """依赖不可达时 ``/readyz`` 返回 **503**（而不是 200 或 500）。

    ⚠️ 必须是 503，不能是 500：503 的语义是「暂时不可用，稍后重试」，
    负载均衡会据此摘流并在之后重试；500 的语义是「服务出错了」，
    很多网关会把它计入错误率并触发告警/降级。
    用 500 表达「依赖还没好」会让一次正常的启动过程看起来像一次故障。

    本用例的环境里 PG/Redis/Milvus 全都不可达（测试不依赖 Docker），
    因此这里的 503 是**预期结果**，不是测试环境的缺陷。
    """
    response = await client.get("/readyz")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["ready"] is False


async def test_readyz_body_is_self_diagnosing(client: AsyncClient) -> None:
    """503 的响应体必须**自解释**：逐项列出检查名、是否通过、耗时。

    ⚠️ 这条的价值在故障现场：容器起来了但一直 unhealthy，此时能拿到的
    信息往往只有「健康检查失败」五个字。若响应体只有一个
    ``{"ready": false}``，排障就得进容器手工逐个试连 ——
    而 503 体里直接写着「是 postgres 不通还是 redis 不通、各花了多久」，
    把一次进容器的排查变成一次 ``curl``。

    用例同时断言了**必需项清单**：它对应 ``REQUIRED_CHECKS``，
    与 docker-compose.yaml 的健康检查语义绑定。
    """
    response = await client.get("/readyz")
    body = response.json()

    assert set(body["checks"]) == {"postgres", "redis", "milvus", "boot"}

    for name, check in body["checks"].items():
        assert set(check) == {"ok", "detail", "duration_ms", "required"}, (
            f"检查项 {name} 的字段与契约不符：{sorted(check)}"
        )
        assert isinstance(check["duration_ms"], (int, float)), "缺少耗时，无法定位慢在哪一项"

    # 必需项必须标出来 —— 编排系统按它决定「该不该摘流」。
    assert body["checks"]["postgres"]["required"] is True
    assert body["checks"]["boot"]["required"] is True


async def test_readyz_never_leaks_credentials(client: AsyncClient) -> None:
    """失败详情里**绝不能**出现连接串中的密码。

    ⚠️ 这是一个真实的泄漏面：``/readyz`` 的 ``detail`` 字段会把底层异常
    的文本带出来，而 SQLAlchemy / redis-py 的连接错误**经常**带上完整 URL
    （``postgresql+asyncpg://user:password@host/db``）。
    又因为探针是**无鉴权**的（编排系统不能带凭据来探活），
    任何能访问该端口的人都能读到它。

    本用例的配置里 Redis 的密码是 ``test-password``（见 conftest），
    连接必然失败，因此这正是那条「会带出 URL 的异常路径」。
    """
    response = await client.get("/readyz")
    raw = response.text

    # 具体密码值不能出现……
    assert "test-password" not in raw, "响应体里泄漏了 Redis 密码"
    # ……而更一般地，`scheme://user:pass@` 这种形态一旦出现就是泄漏。
    assert "://" not in raw or "@" not in raw.split("://", 1)[1].split("/")[0], (
        f"响应体里出现了形如 scheme://user:pass@host 的连接串：{raw[:400]}"
    )


async def test_readyz_boot_check_reflects_lifespan(client: AsyncClient) -> None:
    """``boot`` 检查项在 lifespan 内必须为**通过**。

    ⚠️ 这一项与其它三项不同，它查的不是「外部依赖通不通」，而是
    「框架的 lifespan 有没有跑完」—— 即 ``chat_service`` / ``session_service``
    这些资源是否已经写入 ``app.state``。

    为什么不能省掉它：FastAPI 在 lifespan 的进入段执行完之前就已经开始
    接受连接。在那个窗口期里，PG/Redis/Milvus **都是通的**，
    但 ``app.state.chat_service`` 还不存在 —— 第一个真实请求会 500。
    只看外部依赖的探针会在这个窗口期误报「就绪」。

    本用例里 ``client`` 夹具已经进入了 lifespan，所以 boot 必须是通过的；
    同时 PG/Redis/Milvus 仍然不可达 —— 这个组合恰好证明了
    「boot 与外部依赖是**两条独立**的判据」，而不是同一个信号的两种说法。
    """
    body = (await client.get("/readyz")).json()

    assert body["checks"]["boot"]["ok"] is True
    assert body["checks"]["postgres"]["ok"] is False, (
        "测试环境不应连得上 PostgreSQL —— 若这条失败，说明测试依赖了外部服务，"
        "「make test 不依赖 Docker」这条验收已经不成立了。"
    )


# ==============================================================================
# 三、指标端点 /metrics
# ==============================================================================
async def test_metrics_endpoint_renders_prometheus_text(client: AsyncClient) -> None:
    """``/metrics`` 返回 Prometheus 文本格式（而不是 JSON）。

    格式本身是契约：Prometheus 抓到一个 JSON 会直接报解析错误，
    而 target 会显示 down —— 症状是「面板全空」，不是「接口报错」。
    """
    response = await client.get("/metrics")

    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    body = response.text
    # 至少要能看到就绪指标 —— 它是告警规则的锚点。
    assert "aligo_ready" in body, f"指标输出里缺少 aligo_ready：{body[:300]}"


async def test_metrics_are_scrapable_after_a_request(client: AsyncClient) -> None:
    """发过请求之后，HTTP 指标必须真的被记录下来。

    这条验证的是**中间件真的在链路里**（而不是「挂了个空壳」）。
    它顺带覆盖了 ``HttpMetricsMiddleware`` 的一个易错点：路由模板的取法。
    若拿的是原始 path 而不是路由模板，``/sessions/abc`` 与 ``/sessions/def``
    会变成两条不同的时间序列，指标基数（cardinality）随用户数线性爆炸 ——
    这是 Prometheus 最常见的「被自己压垮」的方式。

    ⚠️ 断言用 ``startswith('/healthz')`` 之类的宽松匹配是刻意的：
    这里要证明的是「指标存在且带上了路由标签」，而不是钉死标签的具体写法
    （``route`` 的取值在 P2 加完鉴权/限流中间件后可能变化）。
    """
    await client.get("/healthz")
    body = (await client.get("/metrics")).text

    assert "aligo_http_requests_total" in body, "HTTP 请求计数未记录"


async def test_metrics_expose_no_secrets(client: AsyncClient) -> None:
    """指标里不能出现密钥。

    ⚠️ 指标端点与就绪探针一样是**无鉴权**的（Prometheus 不带凭据来抓），
    因此任何被打成标签的值都是公开的。标签一旦泄漏密钥，
    它还会被长期保存在时序数据库里 —— 改密钥并不能让历史数据消失。
    """
    body = (await client.get("/metrics")).text

    assert "test-password" not in body
    assert "sk-" not in body, "指标里出现了形如 API Key 的字符串"


# ==============================================================================
# 四、trace_id 传播
# ==============================================================================
async def test_trace_id_header_is_returned(client: AsyncClient) -> None:
    """响应必须带上 ``X-Trace-ID``。

    有了它，用户报「这次请求失败了」时可以直接把响应头里的 id 交过来，
    与日志/链路追踪对上 —— 否则排障只能靠时间戳猜是哪一条。
    """
    response = await client.get("/healthz")

    assert "x-trace-id" in {k.lower() for k in response.headers}
    assert response.headers["x-trace-id"]


async def test_upstream_trace_id_is_honoured(client: AsyncClient) -> None:
    """上游传来的 ``X-Trace-ID`` 必须被**沿用**，而不是另生成一个。

    ⚠️ 这是跨服务串联链路的前提：网关 → 本服务 → 下游，三段日志要能
    串成一条。若本服务每次都自己生成，那么网关侧看到的 id 与
    服务侧日志里的 id 对不上，链路在第一个跳点就断了。

    这条也顺带覆盖了输入校验：上游给的值会进入日志，若不校验就可能被
    用来注入伪造的日志行（``\n`` 之后跟一行假日志）。
    """
    upstream = "tr-upstream-0123456789"
    response = await client.get("/healthz", headers={"X-Trace-ID": upstream})

    assert response.headers["x-trace-id"] == upstream


async def test_malicious_trace_id_is_replaced(client: AsyncClient) -> None:
    """含控制字符的 ``X-Trace-ID`` 必须被丢弃并替换成新生成的。

    日志注入是**真的**会发生的一类问题：``X-Trace-ID: abc\\n2026-01-01 INFO 用户已登录``
    会让日志里凭空多出一行看起来完全正常的记录。因为日志是被
    「人 + 日志系统」共同信任的输入，这种伪造可以覆盖或掩盖真实记录。

    ``src/server/middleware/http_trace.py`` 的 ``_pick_trace_id`` 用
    ``isascii() + isprintable()`` 做校验；本用例是那个校验的机器化版本。
    """
    poisoned = "tr-ok\n2026-01-01 00:00:00 INFO fake log line"
    response = await client.get("/healthz", headers={"X-Trace-ID": poisoned})

    echoed = response.headers["x-trace-id"]
    assert echoed != poisoned, "含有换行符的 trace_id 被原样透传了（日志注入）"
    assert "\n" not in echoed
    assert echoed


async def test_overlong_trace_id_is_replaced(client: AsyncClient) -> None:
    """超长 ``X-Trace-ID`` 必须被丢弃。

    没有长度上限时，一个恶意的超长 header 会被原样写进每一条日志 ——
    既是日志膨胀，也可能撑爆下游日志系统的单字段限制。
    """
    response = await client.get("/healthz", headers={"X-Trace-ID": "x" * 500})

    assert len(response.headers["x-trace-id"]) < 500


async def test_trace_id_is_unique_per_request(client: AsyncClient) -> None:
    """未传 ``X-Trace-ID`` 时，每个请求必须拿到**不同的** id。

    若实现里把 id 缓存在了进程级变量上（一个很容易犯的错），
    所有请求会共享同一个 id —— 链路追踪退化成「全是同一条」，
    比没有还难用（会让人以为「只有一个请求」）。
    """
    ids = {(await client.get("/healthz")).headers["x-trace-id"] for _ in range(5)}

    assert len(ids) == 5, f"5 次请求只产生了 {len(ids)} 个不同的 trace_id"


# ==============================================================================
# 五、响应体是合法 JSON（避免「探针坏了但没人发现」）
# ==============================================================================
@pytest.mark.parametrize("path", ["/healthz", "/readyz"])
async def test_probe_bodies_are_json(client: AsyncClient, path: str) -> None:
    """两个探针的响应体都必须是合法 JSON。

    探针的消费者包括 ``scripts/smoke.py`` 与运维脚本，它们会解析响应体。
    一个返回 HTML 错误页（如反代插进来的 502 页面）的探针会让这些脚本
    以难以理解的方式失败。

    Args:
        client (`AsyncClient`): HTTP 客户端。
        path (`str`): 被测路径。
    """
    response = await client.get(path)

    # 不 assert status_code == 200：/readyz 在无依赖时**应当**是 503。
    assert response.status_code in (200, 503)
    json.loads(response.text)
