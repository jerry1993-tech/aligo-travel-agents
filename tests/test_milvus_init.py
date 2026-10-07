# -*- coding: utf-8 -*-
"""Milvus 初始化脚本（``scripts/milvus_init.py``）的测试。

==============================================================================
这些用例在防什么
==============================================================================
    ``scripts/milvus_init.py`` 是**部署路径上唯一**会去建 Milvus 集合的脚本，
    而它的失败方式全都很难看：

      · 少建一个集合 —— 2026-10-03 实测过：长期记忆那个集合没人建，
        「记住这个」在全新部署上直接以 ``collection not found`` 失败，
        而每次对话的语义召回各打一条 ERROR。漏建的代价不是「少个功能」，
        是「一条业务路径整体不可用，且只在用户用到它时才暴露」。
      · 集合建错形态 —— 维度/度量不符同样不报错，只是分数没有意义
        （见 ``src/knowledge/store.py`` 的模块文档）。
      · 没关客户端 —— ``pymilvus.MilvusClient`` 会起后台线程，
        脚本跑完不退，CI 里表现为「命令挂住」。

    所以本文件用**假向量库**把 ``_run`` 整条走一遍：建了什么、核验了什么、
    以什么码退出、有没有关掉客户端，逐条钉住。真连 Milvus 的验证在
    ``make milvus_init`` 里（那是运维路径），单测不该依赖一个跑着的库。
"""

from __future__ import annotations

import asyncio

import pytest

from scripts import milvus_init
from src.config import Settings
from src.memory.semantic import memory_collection


# ==============================================================================
# 测试替身：一个「记得住自己被怎么用」的假向量库
# ==============================================================================
class _FakeClient:
    """假 ``pymilvus.MilvusClient``，只实现 ``describe_collection`` 用到的四个方法。"""

    def __init__(self, dimension: int, metric_type: str, index_type: str) -> None:
        self.dimension = dimension
        self.metric_type = metric_type
        self.index_type = index_type

    def has_collection(self, collection_name: str) -> bool:
        return True

    def describe_collection(self, collection_name: str) -> dict:
        # ⚠️ 键名与真 pymilvus 一致（``fields[].params.dim``）——
        # 换成本项目自造的键名，用例会在真库上失效而自己不知道。
        return {"fields": [{"name": "vector", "params": {"dim": self.dimension}}]}

    def list_indexes(self, collection_name: str) -> list[str]:
        return ["vector"]

    def describe_index(self, collection_name: str, index_name: str) -> dict:
        return {"metric_type": self.metric_type, "index_type": self.index_type}


class _FakeStore:
    """假向量库：记录建过哪些集合，并按给定形态回答回读。"""

    def __init__(
        self,
        *,
        dimension: int = 1024,
        metric_type: str = "COSINE",
        index_type: str = "HNSW",
    ) -> None:
        self.created: list[tuple[str, int]] = []
        self.described: list[str] = []
        self.closed = False
        self._client = _FakeClient(dimension, metric_type, index_type)

    async def has_collection(self, name: str) -> bool:
        return False

    async def create_collection(self, name: str, dimensions: int) -> None:
        self.created.append((name, dimensions))

    def get_client(self) -> _FakeClient:
        return self._client

    async def __aexit__(self, *args: object) -> None:
        self.closed = True


@pytest.fixture
def fake_store() -> _FakeStore:
    """形态与默认配置一致的假向量库。"""
    return _FakeStore()


@pytest.fixture
def patched(monkeypatch: pytest.MonkeyPatch, fake_store: _FakeStore) -> _FakeStore:
    """把脚本里的 ``build_vector_store`` 换成假库。"""
    monkeypatch.setattr(milvus_init, "build_vector_store", lambda settings: fake_store)
    return fake_store


# ==============================================================================
# 一、目标清单
# ==============================================================================
def test_both_collections_are_targets(settings: Settings) -> None:
    """★★★ 目标清单里**必须**同时有政策库与长期记忆两个集合。

    ⚠️ 这条钉住的是一个真实故障（2026-10-03 实测）：当时脚本只建政策库
    那个集合，长期记忆的集合（``{契约集合}_memory``）没有任何人建 ——
    「记住这个」这条写路径在全新部署上一次都没成功过。
    """
    targets = milvus_init._targets(settings)

    names = [name for name, _ in targets]
    assert settings.milvus.collection in names
    assert memory_collection(settings) in names, (
        "长期记忆集合不在初始化目标里 —— 写路径会在缺集合时失败"
    )


def test_the_memory_collection_is_skipped_when_memory_is_disabled(
    settings: Settings,
    capsys: pytest.CaptureFixture,
) -> None:
    """★ ``ALIGO__MEMORY__ENABLED=false`` 时跳过它，且**打印说明**。

    ⚠️ 「跳过一个东西却不说」与「忘了建它」在日志上长得一模一样 ——
    所以跳过必须有输出。用例断言的就是这行输出存在。
    """
    disabled = settings.model_copy(deep=True)
    disabled.memory.enabled = False

    targets = milvus_init._targets(disabled)

    assert [name for name, _ in targets] == [disabled.milvus.collection]
    assert "跳过" in capsys.readouterr().out


# ==============================================================================
# 二、整条 _run：建 + 回读核验 + 退出码 + 收尾
# ==============================================================================
def test_run_creates_and_verifies_both_collections(
    settings: Settings,
    patched: _FakeStore,
    capsys: pytest.CaptureFixture,
) -> None:
    """★★★ 正常路径：两个集合都建出来、都回读核验，退出码 0，客户端关闭。

    ⚠️ 断言的是「**两个**都建」而不是「建了东西」：这正是本次修复的内容，
    只断言「非空」的用例在只建一个的实现上照样绿。
    """
    code = asyncio.run(milvus_init._run(settings))

    assert code == 0
    assert patched.created == [
        (settings.milvus.collection, settings.milvus.dimension),
        (memory_collection(settings), settings.milvus.dimension),
    ]
    output = capsys.readouterr().out
    assert memory_collection(settings) in output, "输出里看不到记忆集合的名字"
    assert patched.closed, "客户端没关 —— pymilvus 的后台线程会让脚本挂住"


def test_run_fails_when_any_collection_has_the_wrong_shape(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """★★ 任一集合形态不符 ⇒ 退出码 1，且修法里写的是**那个**集合。

    ⚠️ 修法那行是给运维复制粘贴执行的（``drop_collection('<名字>')``）。
    名字写错的后果不是措辞不准，是**删错集合** —— 所以连名字一起断言。
    """
    store = _FakeStore(dimension=512)  # 与配置的 1024 不符
    monkeypatch.setattr(milvus_init, "build_vector_store", lambda settings: store)

    code = asyncio.run(milvus_init._run(settings))

    assert code == 1
    output = capsys.readouterr().out
    assert f"drop_collection('{settings.milvus.collection}')" in output
    assert f"drop_collection('{memory_collection(settings)}')" in output


def test_a_missing_collection_is_reported_not_created_silently(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """★ 回读发现「不存在」时，脚本必须**报错退出**，而不是当成功。

    ⚠️ 假库这里让回读永远答「不存在」—— 对应真库上「建了但没建出来」
    这类诡异状态（权限不足、地址指到了另一个 Milvus）。
    静默成功是最坏的：运维以为建好了。
    """

    class _Vanished(_FakeClient):
        def has_collection(self, collection_name: str) -> bool:
            return False

    class _VanishingStore(_FakeStore):
        def get_client(self) -> _FakeClient:
            return _Vanished(1024, "COSINE", "HNSW")

    store = _VanishingStore()
    monkeypatch.setattr(milvus_init, "build_vector_store", lambda settings: store)

    assert asyncio.run(milvus_init._run(settings)) == 1


def test_the_repair_command_does_not_leak_credentials(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """★★ 手工修复命令里**绝不能**出现配置中的密码。

    ⚠️ 这行输出是给运维复制粘贴执行的，因此它会出现在 CI 日志、工单和
    聊天窗口的截图里 —— 读者范围比「能读到配置文件的人」广得多。

    ⚠️ 三条断言缺一不可，各堵一个方向：

    · 「密码不在」—— 直接的那条。
    · 「占位符在」—— 只钉上一条的话，把整条命令删掉也能让用例变绿，
      而那条命令本身是有用的（见上一个用例）。
    · 「说清了占位符要替换」—— 脱敏让命令**不可直接执行**，脚本必须
      明说。否则运维会以为命令写错了，而一条「看起来对、跑起来错」的
      提示比没有提示更浪费时间。
    """
    from tests.conftest import TEST_ENVIRON

    from src.config import load_settings

    leaky = load_settings(
        "test",
        environ={
            **TEST_ENVIRON,
            "ALIGO__MILVUS__URI": "http://aligo:s3cret-pw@milvus:19530",
        },
        dotenv=False,
    )
    store = _FakeStore(dimension=512)  # 与配置的 1024 不符
    monkeypatch.setattr(milvus_init, "build_vector_store", lambda settings: store)

    assert asyncio.run(milvus_init._run(leaky)) == 1

    output = capsys.readouterr().out
    assert "s3cret-pw" not in output, f"修复命令里泄漏了密码：{output}"
    assert "***:***@" in output, "占位符不见了 —— 但修复命令本身不该被删掉"
    assert "ALIGO__MILVUS__URI" in output, "没告诉运维占位符要换成真实值"
