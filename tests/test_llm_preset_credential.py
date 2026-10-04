# -*- coding: utf-8 -*-
"""预设凭据写入的单测（``src/llm/preset_credential.py``）。

==============================================================================
这一层在守什么
==============================================================================
    框架的 ``upsert_credential`` 对**预设 id** 的写入是「先查后写」：
    该 id 不属于调用者（含查不到）时走直接 ``INSERT``，撞主键就抛
    ``IntegrityError``（见 ``app/storage/_sql/_storage.py::upsert_credential``
    的长注释 —— 那是框架**有意**的设计，用来阻止跨租户覆盖）。

    后果是一个纯粹的并发形状：两个执行者同时给同一个用户首次播种，
    后到的那个必然撞主键。**它不是失败**，但若不认，就会在日志里
    留下一条假警报，把真故障埋掉。

    与 :mod:`src.llm.degradation` 的 Mock 播种、:mod:`src.llm.system_credential`
    的系统凭据播种共用一个实现，因此下面既测这个实现本身，
    也测两个调用方**确实走到了它**。

==============================================================================
为什么异常替身要叫 ``IntegrityError``
==============================================================================
    判定用的是 ``type(exc).__name__``（理由见源模块：不让单测与 Redis
    实现被拖进 SQLAlchemy 的依赖里）。也就是说**类名就是契约** ——
    替身必须真的叫这个名字，改名会让这些用例失去判别力。
    这里刻意不用 ``sqlalchemy.exc.IntegrityError``：那会把本文件绑到
    一个具体的驱动实现上，而这条规则要表达的是「存储层用某个类名
    表示主键冲突」，与是哪个库无关。
"""

from __future__ import annotations

from typing import Any

import pytest

from src.llm.degradation import ensure_mock_credential
from src.llm.mock import MOCK_CREDENTIAL_NAME, mock_credential_id
from src.llm.preset_credential import (
    PRESET_CONFLICT_ERROR_NAME,
    upsert_preset_credential,
)


class IntegrityError(Exception):
    """与 SQLAlchemy 的 ``IntegrityError`` **同名**的异常替身。

    ⚠️ 类名是契约的一部分（判定走 ``type(exc).__name__``），
    所以这个类不能改名、也不能继承 SQLAlchemy 的实现。
    """


class _FlakyStorage:
    """前 N 次 ``upsert_credential`` 抛指定异常，之后正常返回的存储替身。"""

    def __init__(
        self,
        *,
        failures: int = 0,
        exc: type[BaseException] = IntegrityError,
    ) -> None:
        """初始化。

        Args:
            failures (`int`): 前几次调用抛异常。
            exc (`type[BaseException]`): 抛出的异常类型。
        """
        self._remaining = failures
        self._exc = exc
        self.calls: list[tuple[str, Any]] = []

    async def upsert_credential(self, user_id: str, credential_data: Any) -> str:
        """记录调用；未耗尽失败次数时抛异常。

        Args:
            user_id (`str`): 属主。
            credential_data (`Any`): 凭据对象。

        Returns:
            `str`: 凭据 id。

        Raises:
            BaseException: 由 ``exc`` 指定的异常。
        """
        self.calls.append((user_id, credential_data))
        if self._remaining > 0:
            self._remaining -= 1
            raise self._exc("模拟存储层异常")
        return credential_data.id


class _Credential:
    """最小凭据替身：只带 ``upsert`` 会读回去的 ``id``。"""

    def __init__(self, credential_id: str = "preset-1") -> None:
        """记录 id。

        Args:
            credential_id (`str`): 预设 id。
        """
        self.id = credential_id


# ==============================================================================
# 一、实现本身
# ==============================================================================
async def test_a_clean_write_returns_the_id_and_calls_storage_once() -> None:
    """没有冲突时就是一次普通写入 —— 重试逻辑不该改变常态。"""
    storage = _FlakyStorage()

    returned = await upsert_preset_credential(storage, "u1", _Credential())

    assert returned == "preset-1"
    assert len(storage.calls) == 1


async def test_the_first_conflict_is_retried_and_not_reported() -> None:
    """★ 首次主键冲突 ⇒ 重试一次并成功，调用方看不到任何异常。

    这就是「另一个执行者刚刚做完了同一件事」的形状：第一次撞上，
    第二次走「记录存在且属于我 ⇒ 就地更新」那条分支，必然成功。
    """
    storage = _FlakyStorage(failures=1)

    returned = await upsert_preset_credential(storage, "u1", _Credential())

    assert returned == "preset-1"
    assert len(storage.calls) == 2, "没有重试"


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
async def test_any_other_error_propagates_untouched(
    error_type: type[BaseException],
) -> None:
    """★ 只有主键冲突网开一面 —— 其余异常**原样抛出、且不重试**。

    这一条比上面那条更重要：若把 ``except`` 写宽了（例如漏掉
    类名判定），一次数据库连不上会被吞成「播种成功」，
    而 ``/api/v1/default-model`` 会告诉用户「可以用」——
    真正的故障被翻译成了一句假话。
    """
    storage = _FlakyStorage(failures=99, exc=error_type)

    with pytest.raises(error_type):
        await upsert_preset_credential(storage, "u1", _Credential())

    assert len(storage.calls) == 1, "不该重试非主键冲突的异常"


async def test_a_second_conflict_does_propagate() -> None:
    """重试之后**仍然**冲突 ⇒ 让它冒泡。

    两次都撞主键，说明本函数赖以成立的前提（同一个 id、同一个属主）
    不成立 —— 那是真故障，不该被第二层 ``except`` 掩盖成
    「反正重试过了」。用例同时钉住「不会无限重试」。
    """
    storage = _FlakyStorage(failures=99)

    with pytest.raises(IntegrityError):
        await upsert_preset_credential(storage, "u1", _Credential())

    assert len(storage.calls) == 2, "重试次数不是恰好一次"


def test_the_conflict_error_name_is_the_one_we_retry() -> None:
    """把契约里的那个名字钉死。

    ``PRESET_CONFLICT_ERROR_NAME`` 是判定的全部依据；它被改成别的字符串
    （或某个后端换了个类名）时，表现是「播种失败」，而不是「重试没生效」——
    两者都只在日志里留下一行，肉眼区分不开。
    """
    assert PRESET_CONFLICT_ERROR_NAME == "IntegrityError"
    assert IntegrityError.__name__ == PRESET_CONFLICT_ERROR_NAME


# ==============================================================================
# 二、调用方确实走到了这条路上
# ==============================================================================
async def test_mock_seeding_survives_a_concurrent_first_write() -> None:
    """★ Mock 播种：并发首次写入不得表现为失败。

    ``MockCredentialSeedMiddleware`` 每个请求都要问一次「这个用户播种了没」，
    而 ``/api/v1/default-model`` 与它可能被同一次页面加载同时打中 ——
    这是**同一进程内**就能触发的竞态，不需要多 worker。
    """
    storage = _FlakyStorage(failures=1)

    returned = await ensure_mock_credential(storage, "alice")

    assert returned == mock_credential_id("alice")
    assert len(storage.calls) == 2
    owner, credential = storage.calls[0]
    assert owner == "alice"
    assert credential.name == MOCK_CREDENTIAL_NAME


# ==============================================================================
# 三、把「类名」这条契约钉在**真实的框架存储**上
# ==============================================================================
# ⚠️ 上面那些用例的异常是替身 —— 替身的类名由**我们**决定，所以它们证明了
# 「判据写对了」，但证明不了「框架真的抛这个名字」。这两件事必须分开验：
#
#   前者是我们的代码，改坏了会红；
#   后者是别人的代码，**升级框架/换驱动时可能悄悄变** —— 而它一旦变了，
#   表现不是"重试没生效"，而是"播种失败"，只在日志里留一行，
#   与"核弹级故障"共用同一条日志通道。
#
# 而这一条只能靠真实存储来钉：sqlite+aiosqlite 与 postgresql+asyncpg
# 两个后端都已实测抛 ``IntegrityError``（后者是在真容器里跑的，见
# docs/06-部署与运维.md 的排障条目）。下面这条把它变成回归用例。


async def test_the_real_framework_raises_the_exact_name_we_judge_by() -> None:
    """★ 真实框架存储撞主键时抛出的类名，必须就是 :data:`PRESET_CONFLICT_ERROR_NAME`。

    ⚠️ 构造的是**跨属主**冲突（bob 抢 alice 已占用的预设 id），因为那正是
    框架注释里写明的、有意抛错的那条路径：``INSERT`` 撞全局主键。
    同一属主的重复写入走的是就地 UPDATE，压根不抛 —— 那也一并钉住，
    否则这条用例可能在某次框架改动后变成"什么都没测"。

    ⚠️ 断言用的是**类名**而不是 ``isinstance``：这不是图省事，
    而是与 :func:`~src.llm.preset_credential.upsert_preset_credential`
    的判据**逐字一致**。用 ``isinstance`` 去断言的话，判据改成别的写法时
    这条用例反而可能继续绿。
    """
    from agentscope.app.storage._sql import AsyncSQLAlchemyStorage
    from agentscope.credential import DashScopeCredential

    storage = AsyncSQLAlchemyStorage(
        url="sqlite+aiosqlite:///:memory:",
        create_tables=True,
        auto_migrate=False,
    )
    async with storage:
        credential = DashScopeCredential(
            # ⚠️ 形似真实密钥的假值，只为在异常文本里能认出它。
            api_key="sk-not-a-real-key-only-for-this-test",
            id="preset-race-id",
            name="竞态用例",
        )

        # 属主自己写两次：第二次必须走就地 UPDATE，**不抛**。
        first = await storage.upsert_credential("alice", credential)
        again = await storage.upsert_credential("alice", credential)
        assert first == again == "preset-race-id", (
            "同一属主的重复写入不再是就地更新了 —— 下面那条「跨属主必冲突」"
            "就不再能证明任何事。"
        )

        # 跨属主抢占同一个预设 id：框架**必须**抛，且类名就是那个契约名。
        with pytest.raises(Exception) as caught:  # noqa: B017 —— 判据在下面
            await storage.upsert_credential("bob", credential)

        assert type(caught.value).__name__ == PRESET_CONFLICT_ERROR_NAME, (
            f"框架抛的是 {type(caught.value).__name__!r}，而判据是 "
            f"{PRESET_CONFLICT_ERROR_NAME!r} —— 重试会因此**永远不触发**，"
            "播种直接失败。此时要么改判据，要么承认这个版本的框架不兼容。"
        )

        # ⚠️ 顺带钉住框架的**反覆盖**语义：bob 抢失败之后，
        # 这条记录必须还在 alice 名下（否则"保护"就成了"静默搬家"）。
        owner_record = await storage.get_credential("alice", "preset-race-id")
        assert owner_record is not None, "alice 的记录被 bob 的失败写入弄没了"
