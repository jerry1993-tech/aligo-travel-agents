#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""幂等地初始化 Milvus 集合，并**回读核验**它的形态。

==============================================================================
建的是**两个**集合，不是一个
==============================================================================
    本项目在 Milvus 里有两条互不相干的支路，各用各的集合：

      · ``{ALIGO__MILVUS__COLLECTION}``                —— 政策知识库（KB 链路）
      · ``{ALIGO__MILVUS__COLLECTION}_memory``         —— 长期记忆画像（语义召回）

    派生名由 ``src/memory/semantic.py::memory_collection`` 算出，运维只需要
    记住一个名字。两者**形态约束逐项相同**（同一个 Milvus、同一个向量模型、
    同一个维度），所以本脚本对它们用同一套期望值。

    ⚠️ 第二个集合曾经**不属于本脚本的职责**，代价是 2026-10-03 实测到的
    一次真实故障：全新部署上没人建它，于是「记住这个」这条写路径直接以
    ``MilvusException: collection not found`` 失败，而每次对话的语义召回
    也各打一条 ERROR 级日志。现在两条路都堵上了 —— 本脚本（运维路径）
    建它，``SemanticMemory.remember``（运行路径）在第一次写入前幂等地
    自愈（那一步的完整理由写在 ``src/memory/semantic.py`` 里）。

    ``ALIGO__MEMORY__ENABLED=false`` 时跳过第二个集合，并**打印一行说明**：
    跳过一个东西却不说，与忘了建它，在日志上长得一模一样。

==============================================================================
它与 `make test` / `make smoke` 的分工
==============================================================================
    `make test`     进程内，不连 Milvus，测「管理器的逻辑对不对」
    `make smoke`    对已启动的服务发请求，测「这一套部署能不能用」
    本脚本          直接连 Milvus，测「那个集合到底按什么参数建出来的」

    本脚本回答的是**前两者都答不了**的一个问题：集合在 Milvus 里的
    真实形态（维度/索引/度量）是否与配置一致。

==============================================================================
⚠️ 为什么「幂等」还不足以解决问题
==============================================================================
``MilvusLiteStore.create_collection`` 在集合已存在时是 **no-op**
（``agentscope/rag/_vdb/_milvus_lite.py:129-131``）。所以重复跑本脚本是安全的 ——
但也意味着**一个用错维度建出来的集合永远不会被修正**。

于是本脚本做两件事，而不是一件：

  1. 建集合（幂等）；
  2. **把建好的集合读回来，逐项与配置比对** —— 见 :func:`verify_collection_shape`。

第 2 步不是「锦上添花」。维度相同但度量不同（比如建的时候是 L2、
配置写着 COSINE）的集合，写入成功、检索成功、有分数 ——
**没有任何一处会报错**，只是分数的大小关系没有意义。
只有回读能发现它。

退出码：**0 = 全部集合已就绪且形态正确；1 = 有集合形态不符或连接失败**。
    ⚠️ 形态不符时**不**自动重建。删集合会丢光已索引的向量，
    而那可能是一小时的重新灌数据 —— 这种决定不该由一个初始化脚本
    替运维做。脚本只负责**把问题说清楚并给出一条可复制的修法**。
"""

from __future__ import annotations

import argparse
import asyncio
import sys

# 允许以 `python scripts/milvus_init.py` 直接运行（此时 sys.path[0] 是 scripts/）。
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from src.config import Settings, get_settings, load_settings  # noqa: E402
from src.knowledge.store import (  # noqa: E402
    build_vector_store,
    describe_collection,
    describe_vector_store,
    ensure_collection,
    verify_collection_shape,
)

# ⚠️ 从 ``src.memory.semantic`` 直接取派生函数，而不是自己拼 ``f"{...}_memory"``：
# 派生规则只有一处真值（那个函数的文档里写着为什么）。这里抄一遍的后果是
# 「脚本建了 A，应用写进 B」—— 而那种错的表现是「记忆不生效」，不报任何异常。
from src.memory.semantic import memory_collection  # noqa: E402


def _targets(settings: Settings) -> list[tuple[str, str]]:
    """本次要建/核验的集合清单：``(集合名, 人话标签)``。

    ⚠️ 长期记忆那个集合**也在这里**。它曾经不在 —— 于是全新部署上
    「记住这个」直接失败（详见模块文档里的实测记录）。
    ``ALIGO__MEMORY__ENABLED=false`` 时跳过它，并说明是**刻意跳过**。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `list[tuple[str, str]]`: 至少含政策知识库那一个。
    """
    targets: list[tuple[str, str]] = [
        (settings.milvus.collection, "政策知识库"),
    ]
    if settings.memory.enabled:
        targets.append((memory_collection(settings), "长期记忆画像"))
    else:
        print(
            "▶ 长期记忆已关闭（ALIGO__MEMORY__ENABLED=false），"
            f"跳过集合 {memory_collection(settings)!r}。",
        )
    return targets


def _print_target(settings: Settings, targets: list[tuple[str, str]]) -> None:
    """打印本次要连的目标与期望形态。

    ⚠️ 必须**先**打印目标再连。连不上时，日志里有没有这一行，
    决定了排查是「看一眼就知道连错地址了」还是「得去翻配置」。
    这也是 URI 要脱敏的原因：这行日志会被贴出去。
    """
    described = describe_vector_store(settings)
    print("▶ 目标：")
    for key, value in described.items():
        print(f"    {key:<12} {value}")
    # ⚠️ 单独列出「本次要动哪些集合」：上面那行的 ``collection`` 是**基名**
    # （记忆集合由它派生），只看它会以为本次只碰一个集合。
    print("▶ 本次要核验的集合：")
    for name, label in targets:
        print(f"    {label:<8} {name}")


def _print_actual(actual: dict, label: str) -> None:
    """打印某个集合的**实际**形态。

    Args:
        actual (`dict`): :func:`describe_collection` 的返回值。
        label (`str`): 人话标签（``政策知识库`` / ``长期记忆画像``）——
            两个集合的输出连着打印，没有标签就分不清哪段是哪段。
    """
    print(f"▶ {label}的实际形态：")
    for key in (
        "collection",
        "exists",
        "dimension",
        "index_type",
        "metric_type",
        # ⚠️ 一致性级别出现在这里，是因为它是**唯一**能看见「写入对检索
        # 是否立刻可见」的地方：服务端默认的 Bounded 会造成约 1 秒的
        # 可见性窗口（实测数据见 src/knowledge/store.py::CONSISTENCY_LEVEL）。
        # 它**不参与** verify_collection_shape 的判定 —— 那不是形态问题，
        # 而且刚请求完的一致性是**延迟生效**的（回读会先返回旧值），
        # 拿它当发布闸门会误报。
        "consistency_level",
    ):
        print(f"    {key:<12} {actual.get(key)}")
    if actual.get("index_error"):
        # ⚠️ 读不出索引信息本身是一条发现，不该被静默吞掉。
        print(f"    index_error  {actual['index_error']}")


async def _run(settings: Settings) -> int:
    """执行初始化并核验**全部**目标集合。

    ⚠️ 两个集合**各自独立判定**，但只要有一个不合格，退出码就是 1 ——
    本脚本是发布前的最后一道闸，它不该出现「一个坏了但整体还绿」的形状。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `int`: 进程退出码（0 通过 / 1 失败）。
    """
    targets = _targets(settings)
    _print_target(settings, targets)

    store = build_vector_store(settings)
    broken: list[tuple[str, str]] = []
    try:
        for name, label in targets:
            print(f"\n──── {label} ────")
            created = await ensure_collection(store, settings, collection=name)
            print(
                f"▶ 集合 {name!r} "
                f"{'已创建' if created else '已存在（未改动）'}。",
            )

            actual = await describe_collection(store, name)
            _print_actual(actual, label)

            problems = verify_collection_shape(actual, settings, collection=name)
            if problems:
                broken.append((name, label))
                print("❌ 形态与配置不符：")
                for line in problems:
                    print(f"  · {line}")

        if not broken:
            print(
                f"\n✅ {len(targets)} 个集合均已就绪，"
                "且维度 / 索引 / 度量与配置一致。",
            )
            return 0

        print("\n⚠️ 本脚本**不会**自动重建集合 —— 删集合会丢光已索引的向量。")
        for name, label in broken:
            print(
                f"   {label}：确认可以重灌数据时，手工执行：\n"
                f"       python -c \"import pymilvus;"
                f"pymilvus.MilvusClient('{settings.milvus.uri}')"
                f".drop_collection('{name}')\"",
            )
        print("   然后重跑本脚本。")
        return 1
    finally:
        # ⚠️ 必须关：``MilvusClient`` 会起后台线程，进程不退的话脚本会挂住。
        try:
            await store.__aexit__(None, None, None)
        except Exception:  # noqa: BLE001
            # 关闭失败不改变结论 —— 它只是一次清理。
            pass


def main() -> int:
    """命令行入口。

    Returns:
        `int`: 进程退出码。
    """
    parser = argparse.ArgumentParser(
        description=(
            "幂等地初始化 Milvus 集合（政策知识库 + 长期记忆画像），"
            "并回读核验其形态。"
        ),
    )
    parser.add_argument(
        "--env",
        default=None,
        help="配置档（dev / test / prod）。默认取 ALIGO__APP__ENV 或 dev。",
    )
    args = parser.parse_args()

    # ⚠️ ``get_settings()`` **没有** ``env`` 参数（loader.py:456）——
    # 它读的是进程内的 ``ALIGO__APP__ENV``。要指定别的档位得走
    # ``load_settings(env_name)``（它会自己读 .env）。
    # 写成 ``get_settings(env=...)`` 会是一个 TypeError，
    # 且只在带 --env 时才触发 —— 属于「上线前最后一次手工验证才发现」的那类错。
    settings = load_settings(args.env) if args.env else get_settings()

    try:
        return asyncio.run(_run(settings))
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 —— 顶层入口，见下面的说明
        # ⚠️ 顶层捕获**宽**异常是刻意的：这个脚本的失败信息是给运维看的，
        # 而「连不上」「鉴权失败」「pymilvus 没装」「集合名非法」
        # 抛出的异常类型各不相同。让 traceback 直接打出来，
        # 比包一层只剩「初始化失败」的自定义异常有用得多 ——
        # 因为真正的线索（拒绝连接 / 名字不合法）就在那条消息里。
        from src.observability.redaction import safe_error

        print(f"\n❌ 初始化失败：{safe_error(exc)}", file=sys.stderr)
        if isinstance(exc, ImportError) or "pymilvus" in str(exc):
            print(
                "   提示：pymilvus 没装。本项目的 requirements.txt 已含它；"
                "若在容器外跑，请 pip install pymilvus。",
                file=sys.stderr,
            )
        else:
            print(
                f"   提示：确认 Milvus 已启动且 {settings.milvus.uri} 可达"
                "（core 档含它：make up）。",
                file=sys.stderr,
            )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
