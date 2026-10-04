# -*- coding: utf-8 -*-
"""预设 id 凭据的幂等写入：一次写不进去，不等于失败了。

文件职责：
    收纳「往 storage 写一条**预设 id** 的凭据」时唯一一处容易写错的细节 ——
    **主键冲突不是失败**。两个调用方共用它：

        · ``src/llm/degradation.py::ensure_mock_credential``
          —— 每个用户一条 Mock 凭据（零密钥降级）；
        · ``src/llm/system_credential.py::ensure_system_credential``
          —— 全局一条系统凭据（真实密钥只读共享）。

上下游依赖：
    - 上游：只依赖调用方传进来的 ``storage``（框架的 ``StorageBase`` 实现）。
    - 下游：上面两个模块。

==============================================================================
为什么会有主键冲突：框架 upsert 的两条分支
==============================================================================
    ``AsyncSQLAlchemyStorage.upsert_credential(user_id, credential)`` 对
    **带预设 id** 的写入不是一句 ``INSERT ... ON CONFLICT``，而是先查后写
    （``app/storage/_sql/_storage.py::upsert_credential``）：

        · 该 id **属于** ``user_id`` ⇒ 就地更新（幂等：改了密钥重启即生效）；
        · 该 id **不属于** ``user_id``（含「查不到」）⇒ 直接 ``INSERT``，
          撞全局主键即抛 ``IntegrityError``。

    框架这么写是**对的**：SQL 表的主键是全局的，若「id 冲突就更新」，
    任何调用者都能拿一个预设 id 覆盖别人的凭据。它把一个安全问题
    （跨租户覆盖）换成了一个并发问题（首次写入撞车）——
    而并发问题可以用「重试一次」在调用侧干净地解决。

    于是本模块要处理的形状是 —— **只有「首次写入」那一刻**会冲突：

        进程 A: 查不到 → INSERT
        进程 B: 查不到 → INSERT   ← IntegrityError

    （``WORKERS=1`` 时同一进程内也可能发生：两条并发路径同时给同一个
    用户首次播种，例如「中间件按需播种」与「``/api/v1/default-model``
    被同一次页面加载同时打中」。）

==============================================================================
为什么必须认它，而不是让它冒泡
==============================================================================
    此刻两个执行者在写**同一条**记录，内容也相同（id 固定、密钥同源），
    语义上「目标已经达成」。让它冒泡的代价是把一次无害的并发变成一条
    刺眼的错误日志 —— 而**真正的故障**（连接断了、权限不够、表不存在、
    主键之外的约束不满足）会被埋进同一股日志流里，排查的人从这条假警报
    开始查，方向就错了。

    所以策略是：**只给主键冲突网开一面，其余异常原样抛出**。
    判定用异常**类名**而不是 ``isinstance``：``IntegrityError`` 来自
    ``sqlalchemy``，而本模块刻意不在 import 期绑定任何 SQLAlchemy 符号 ——
    ``storage`` 也可能是 Redis 实现或单测里的替身，它们不该因此被拖进
    SQLAlchemy 的依赖里，更不该在「零密钥启动」这条路径上多背一个导入面。
"""

from __future__ import annotations

from typing import Any

__all__ = ["PRESET_CONFLICT_ERROR_NAME", "upsert_preset_credential"]

#: 被当作「另一个执行者抢先完成」而不是故障的异常类名。
#:
#: 提成常量是为了让单测能直接引用它（而不是在测试里手抄一遍字符串），
#: 也让「本项目只认这一个名字」这件事有一处可读的出处。
PRESET_CONFLICT_ERROR_NAME = "IntegrityError"


async def upsert_preset_credential(
    storage: Any,
    owner_id: str,
    credential: Any,
) -> str:
    """把 ``credential`` 写到 ``owner_id`` 名下，容忍首次写入的主键冲突。

    Args:
        storage (`Any`): 框架的存储实现（``AsyncSQLAlchemyStorage`` 等）。
        owner_id (`str`): 凭据属主。
        credential (`Any`): 带**预设 id** 的凭据对象。

    Returns:
        `str`: 凭据记录 id。

    Raises:
        Exception: 除主键冲突之外的任何异常**原样抛出**。调用方决定是
            吞掉还是上报，本函数不替它们做决定，也不把异常翻译成一个
            看起来正常的返回值 —— 那会让「播种失败了」在两端都不可见。
    """
    try:
        return await storage.upsert_credential(owner_id, credential)
    except Exception as exc:  # noqa: BLE001 —— 只给主键冲突网开一面
        if type(exc).__name__ != PRESET_CONFLICT_ERROR_NAME:
            raise
        # 另一个执行者刚刚把同一条记录写好了 ⇒ 再走一次 upsert。
        # 这一次「查得到，且属于我」，命中框架的就地更新分支，因此会成功。
        #
        # ⚠️ 第二次**不再捕获**。重试之后还冲突，说明本函数赖以成立的前提
        # （同一个 id、同一个属主）不成立 —— 那是真故障，
        # 不该被第二层 except 掩盖成「反正重试过了」。
        return await storage.upsert_credential(owner_id, credential)
