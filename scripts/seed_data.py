#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构造**企业差旅政策文档**与**业务演示数据**，并幂等地灌进 Milvus / 业务库。

==============================================================================
它与 milvus_init.py / make test 的分工
==============================================================================
    scripts/milvus_init.py   只**建集合**（并回读核验形态），不写任何数据
    本脚本                   往已建好的集合里**灌数据**，并往业务库写演示数据
    make test                进程内、不连任何外部服务，只测本脚本的**纯逻辑**

    三者回答三个不同的问题：「集合建对了吗」「库里有没有内容」「构造逻辑对不对」。
    本脚本**不建集合** —— 那是 milvus_init.py 的职责，重复一份只会让「集合参数
    以谁为准」变得模糊。灌数据前请先跑 ``make milvus_init``。

==============================================================================
⚠️ 三条硬约束（都是被真实故障逼出来的）
==============================================================================
    1. **确定性**。所有数据都是**字面量**，不用 ``random``、不用 ``hash()``
       （理由见 ``src/storage/memory.py`` 的模块文档：``hash("北京")`` 在不同
       进程里结果不同，而价格会进快照、进录屏、进评测集）。同一天跑两次本脚本，
       写进去的字节应当**完全一样**。
    2. **幂等**。重复跑**不产生重复数据**。幂等键：
         · Milvus 侧：``document_id = "seed-policy-<doc_id>"``。框架写入走
           ``upsert``（``_vdb/_milvus_lite.py`` 的 ``insert`` 以
           ``(document_id, chunk_index)`` 派生主键），我们**再多做一步**
           「先 delete 再 insert」，见 :func:`seed_policy_documents` 的说明。
         · 业务库侧：各表的主键（``user_id`` / ``trip_id`` / ``order_id`` /
           ``request_id``）。写入前先查出已存在的键，只补**缺的**那些。
    3. **绝不打印任何密钥**。本脚本只打印**目标**（URI 已脱敏）与**条数**，
       ``.env`` 的值一个都不读、不打印。

==============================================================================
⚠️ ``--dry-run`` 是这条链路上唯一能离线验证的部分
==============================================================================
    真写路径要连 Milvus 与业务库，前者在没有 Docker 的机器上不可达，后者在没有
    ``make up`` 时也一样。因此 ``--dry-run`` **不建立任何连接**，只把「将要写入
    的东西」打印出来 —— 它既是给人看的手稿，也是 ``tests/test_seed_data.py``
    的主要抓手（离线、确定、可断言）。改动数据构造逻辑时先看 dry-run 的输出。

退出码：**0 = 成功（或 dry-run 正常打印）；1 = 连接失败 / 写入失败**。
    连接失败时打印**清晰的中文错误 + 下一步建议**，而不是一条裸 traceback ——
    理由与 milvus_init.py 相同：这个脚本的失败信息是给运维看的，而「连不上」
    本身就是最常见、最需要一句话说清的那类失败。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import sys
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

# 允许以 `python scripts/seed_data.py` 直接运行（此时 sys.path[0] 是 scripts/）。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agentscope.app.storage import (  # noqa: E402
    EmbeddingModelConfig,
    KnowledgeBaseData,
    KnowledgeBaseRecord,
)
from agentscope.credential import CredentialFactory  # noqa: E402
from agentscope.message import TextBlock  # noqa: E402
from agentscope.rag import Chunk  # noqa: E402
from sqlalchemy import (  # noqa: E402
    JSON,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    select,
    text,
)
from sqlalchemy.ext.asyncio import AsyncEngine  # noqa: E402

from src.config import Settings, load_settings  # noqa: E402
from src.domain.entities import ApprovalRequest, TravelOrder  # noqa: E402
from src.domain.enums import CabinClass, OrderStatus  # noqa: E402
from src.knowledge.manager import SingleCollectionKbManager  # noqa: E402
from src.knowledge.store import build_vector_store  # noqa: E402
from src.llm.degradation import ensure_mock_credential  # noqa: E402
from src.llm.factory import build_credential, has_api_key, should_use_mock  # noqa: E402
from src.llm.mock import MOCK_CREDENTIAL_TYPE, MockCredential  # noqa: E402
from src.llm.preset_credential import upsert_preset_credential  # noqa: E402
from src.observability.redaction import redact, safe_error  # noqa: E402
from src.storage.engine import (  # noqa: E402
    build_business_engine,
    business_schema_for,
    ensure_business_schema,
)
from src.web_embedding import unwrap_embedding_model  # noqa: E402

# ==============================================================================
# 一、连接失败时抛的异常
# ==============================================================================


class SeedConnectionError(RuntimeError):
    """连不上外部服务（Milvus / 业务库）时抛出。

    ⚠️ 存在的理由：调用方（CLI 与测试）需要一种**稳定的信号**来区分
    「服务没起来」与「代码写错了」。若一律抛裸异常，运维看到的是一条
    traceback，得读完整个调用栈才知道是连不上；而消息体本就该是一句
    中文 + 一条下一步建议。

    Attributes:
        message: 面向人的中文说明（含脱敏后的目标与下一步建议）。
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


# ==============================================================================
# 二、纯数据层 —— 全部是字面量，可离线构造与断言
# ==============================================================================


@dataclass(frozen=True)
class PolicySection:
    """政策文档里的一个小节。

    ⚠️ 文档按**小节**而不是按固定字数切分：一节就是检索的一个粒度。
    按字数切会让「一线城市住宿限额 700 元」被从中间截断，检索到半句；
    按小节切则保证每个 chunk 说的是**一件完整的事**。

    Attributes:
        heading: 小节标题。
        body: 小节正文（数句具体条款）。
    """

    heading: str
    body: str


@dataclass(frozen=True)
class PolicyDocument:
    """一篇企业差旅政策文档（灌进 Milvus，供 RAG 检索）。

    Attributes:
        doc_id: 文档标识（稳定，参与幂等键）。
        title: 文档标题。
        category: 分类（交通 / 住宿 / 报销 …）。
        sections: 小节列表，每节成为 Milvus 里的一个 chunk。
    """

    doc_id: str
    title: str
    category: str
    sections: tuple[PolicySection, ...]

    @property
    def chunk_count(self) -> int:
        """本文件将产生的 chunk 数（= 小节数）。

        Returns:
            `int`: 小节条数。
        """
        return len(self.sections)


@dataclass(frozen=True)
class UserRecord:
    """一个演示用户（写进业务库 ``users`` 表）。

    Attributes:
        user_id: 用户 id（主键，幂等键）。
        display_name: 姓名。
        department: 部门。
        employee_level: 职级（决定审批权限与差标档位）。
        cost_center: 成本中心编码。
    """

    user_id: str
    display_name: str
    department: str
    employee_level: str
    cost_center: str


@dataclass(frozen=True)
class TripRecord:
    """一次出差行程（写进业务库 ``trips`` 表）。

    ⚠️ 行程**不是** :class:`~src.domain.entities.TravelOrder`：行程是「打算去哪」，
    订单是「已经买了什么」。两者生命周期不同，故不合并（与 entities.py 的
    设计理由一致）。

    Attributes:
        trip_id: 行程 id（主键，幂等键）。
        user_id: 归属用户。
        origin: 出发城市。
        destination: 目的城市。
        depart_date: 出发日期，``YYYY-MM-DD``。
        days: 天数。
        purpose: 出差事由。
        status: 行程阶段（``PLANNED`` / ``ONGOING`` / ``DONE``）。
    """

    trip_id: str
    user_id: str
    origin: str
    destination: str
    depart_date: str
    days: int
    purpose: str
    status: str


@dataclass(frozen=True)
class SeedPlan:
    """一次播种的**完整内容**（纯数据，构造它不碰任何外部服务）。

    ⚠️ 幂等键由各条记录自身携带（``doc_id`` / ``user_id`` / ``trip_id`` /
    订单与申请单的 id）。把它们集中在一个对象里，是为了让 dry-run 与真写
    走**同一份**数据 —— 若 dry-run 自己再拼一遍，两者迟早不一致。

    Attributes:
        policy_documents: 政策文档。
        users: 演示用户。
        trips: 演示行程。
        orders: 演示订单（复用 domain 实体，状态取自状态机）。
        approvals: 演示申请单。
    """

    policy_documents: tuple[PolicyDocument, ...]
    users: tuple[UserRecord, ...]
    trips: tuple[TripRecord, ...]
    orders: tuple[TravelOrder, ...]
    approvals: tuple[ApprovalRequest, ...]


#: Milvus 侧 ``document_id`` 的前缀。
#:
#: ⚠️ 加前缀不只是为了好看：它让「由本脚本播种的文档」在一眼之间可辨认，
#: 从而能在排查时把「种子数据」与「用户上传的知识」区分开 —— 两者在同一个
#: 单集合里（见 ``src/knowledge/manager.py`` 的单集合策略）。
POLICY_DOC_ID_PREFIX = "seed-"


def policy_document_id(doc_id: str) -> str:
    """算出某篇政策文档在向量库里的 ``document_id``（幂等键的一部分）。

    Args:
        doc_id (`str`): 文档标识（:attr:`PolicyDocument.doc_id`）。

    Returns:
        `str`: 形如 ``seed-policy-overview``。
    """
    return f"{POLICY_DOC_ID_PREFIX}{doc_id}"


# ==============================================================================
# 政策知识库的**归属身份**与记录标识（都是契约值，不是随手写的字面量）
# ==============================================================================

#: 政策知识库归属的**用户 id**。
#:
#: ⚠️⚠️ 必须与演示身份一致（README.md 与 docs/01-功能接口.md 里的演示用户是
#: ``alice``）。管理器的检索路径是 ``get_knowledge(user_id, kb_id)``，它要求
#: KB 记录**属于该 user_id** 才会返回句柄；而前端列知识库时也只列
#: ``list_knowledge_bases(<当前登录用户>)``。
#:
#: 若这里写成别的 id（例如业务表里的 ``u1001``），播种出来的 KB 记录就落在
#: 一个**永远不会被登录用户访问到**的名下 —— 症状是「库里明明有 chunk，
#: 页面上却列不出任何知识库，问答永远没有出处」。用户 id 与演示身份脱节，
#: 正是这一类「数据在、就是看不见」故障的根因。
#:
#: 可用 CLI ``--kb-user`` 覆盖，但覆盖值与前端实际登录的身份必须一致。
KB_USER_ID = "alice"

#: 政策知识库记录的**固定 id**。
#:
#: ⚠️ 固定而不是让框架生成（``KnowledgeBaseRecord.id`` 是 ``default_factory``）：
#: 管理器的 ``create_knowledge_base`` 每次都会新造一个 id，反复播种就会在
#: storage 里堆出多条同名 KB 记录、并把同一批向量重复灌进同一个集合（单集合
#: 策略下它们共享 ``metadata_filter`` 之外的物理空间）。固定 id + 就地 upsert
#: 让「重复播种」在记录层与向量层**都是**幂等的 —— 这也是我们绕过
#: ``create_knowledge_base`` 而直接 ``storage.upsert_knowledge_base`` 的原因。
KB_ID = "aligo-travel-policy"

#: 知识库展示名（前端「知识库」页显示的标题）。
KB_NAME = "企业差旅政策库"

#: 知识库描述。会随句柄交给 RAG 中间件，用于告诉模型「这里有什么、何时查」。
KB_DESCRIPTION = "公司差旅标准、报销与审批规则，供政策问答检索。"

#: 播种用的 Mock 向量模型名。
#:
#: ⚠️ 与 :class:`~src.web_embedding.mock.MockEmbeddingModel` 的默认 ``model``
#: 同名。它会被写进 KB 记录的 ``embedding_model_config.model``，后续检索
#: 时由 ``build_embedding_model`` 按这个名字回查模型卡片（Mock 无卡片，
#: 因此 ``context_size`` 回落到默认值）。名字对不上不会报错，只会让检索时的
#: 向量与写入时的向量来自不同的「模型身份」—— 而 Mock 本身无语义，这种错配
#: 在结果上完全看不出来。
MOCK_EMBEDDING_MODEL_NAME = "aligo-mock-embedding"

#: 政策知识库**真实**向量凭据的固定 id（配了密钥时使用）。
#:
#: ⚠️ 固定而不是让框架生成：``upsert_credential`` 对「属于本用户的预设 id」
#: 是就地更新，于是「改了密钥 / 换了向量模型后重灌」会**更新同一条记录**，
#: 而不是每跑一次就多一条凭据。属主是**知识库归属者**（不是系统身份）——
#: 框架的知识库链路按 owner-internal 方式解析凭据，指到别人名下会解析不到，
#: 而那条失败路径在 ``src/knowledge/rag.py`` 里是**逐条吞掉**的，
#: 症状会退化成「知识库列表看得见、问答就是不带出处」。
KB_EMBEDDING_CREDENTIAL_ID = "aligo-kb-embedding"

#: 上面那条凭据的显示名（凭据页可见）。写明用途，免得被当成用户自建条目删掉。
KB_EMBEDDING_CREDENTIAL_NAME = "政策知识库向量凭据（由 seed 脚本按配置写入）"


def policy_documents() -> tuple[PolicyDocument, ...]:
    """返回全部政策文档（字面量，确定性）。

    ⚠️ 内容刻意写得**细**（城市分档限额、提前期天数、审批金额门槛都给了
    具体数字）：检索要能区分「一线城市住宿上限」与「提前期要求」这两类问题，
    靠的是条款之间**实质不同**，而不是标题不同。细则太少会让所有 query 都
    召回同一段泛泛而谈的总则，RAG 的引用来源也就失去了意义。

    Returns:
        `tuple[PolicyDocument, ...]`: 政策文档。
    """
    return (
        PolicyDocument(
            doc_id="policy-overview",
            title="差旅政策总则",
            category="总则",
            sections=(
                PolicySection(
                    heading="适用范围",
                    body=(
                        "本政策适用于公司全体正式员工因公出差所产生的交通、住宿"
                        "及相关费用。实习生、外包人员出差须由所属部门负责人另行"
                        "审批，标准不高于同职级正式员工的 80%。"
                    ),
                ),
                PolicySection(
                    heading="差旅基本原则",
                    body=(
                        "差旅须同时满足三个条件方可报销：与工作直接相关、经济合理、"
                        "凭证齐全。「经济合理」指在满足工作需要的前提下选择价格更优"
                        "的方案；能线上完成的会议不得安排出差。"
                    ),
                ),
                PolicySection(
                    heading="职级与差标档位",
                    body=(
                        "职级 P4-P6 适用 A 档差标，P7-P8 适用 B 档差标，P9 及以上适用 "
                        "C 档差标。档位决定可乘舱位与住宿限额，不影响报销的合规要求。"
                    ),
                ),
                PolicySection(
                    heading="生效与解释",
                    body=(
                        "本政策由财务部负责解释，自发布之日起生效。政策更新后，"
                        "已完成审批但尚未执行的行程按申请提交时的版本执行。"
                    ),
                ),
            ),
        ),
        PolicyDocument(
            doc_id="policy-transport",
            title="交通工具与舱位标准",
            category="交通",
            sections=(
                PolicySection(
                    heading="交通工具选择",
                    body=(
                        "两地直线距离在 1200 公里以内，原则上优先选择高铁；超出 1200 "
                        "公里或高铁总耗时超过 8 小时，可乘飞机。夜间到达无合适车次的，"
                        "可乘飞机并由系统记录理由，交由审批判断。"
                    ),
                ),
                PolicySection(
                    heading="飞机舱位等级",
                    body=(
                        "A 档差标限经济舱；B 档差标可乘经济舱与超级经济舱；C 档差标可"
                        "乘公务舱。头等舱一律不予报销，任何职级均无例外，包括总经理。"
                    ),
                ),
                PolicySection(
                    heading="火车座席等级",
                    body=(
                        "A 档差标限二等座；B 档差标可乘一等座；C 档差标可乘商务座。"
                        "车程超过 5 小时的，A 档也可乘一等座。"
                    ),
                ),
                PolicySection(
                    heading="市内交通",
                    body=(
                        "出差期间的市内交通据实报销，单次 200 元以上的须附事由说明。"
                        "机场与火车站往返优先选择公共交通；携带大件设备或深夜到达的，"
                        "可乘出租车或网约车。"
                    ),
                ),
            ),
        ),
        PolicyDocument(
            doc_id="policy-hotel",
            title="酒店住宿标准与限额",
            category="住宿",
            sections=(
                PolicySection(
                    heading="城市分级",
                    body=(
                        "按城市消费水平分为三类：一线城市（北京、上海、广州、深圳）为"
                        "一类；省会城市及苏州、杭州、南京、成都、武汉为二类；其余为三类。"
                    ),
                ),
                PolicySection(
                    heading="一类城市限额",
                    body=(
                        "一线城市住宿限额：A 档 600 元/晚，B 档 800 元/晚，C 档 1200 "
                        "元/晚。超出限额的部分由员工自理，确因会议指定酒店而超标的，"
                        "须在申请单中说明并由上一级审批。"
                    ),
                ),
                PolicySection(
                    heading="二类城市限额",
                    body=(
                        "二类城市住宿限额：A 档 450 元/晚，B 档 600 元/晚，C 档 900 "
                        "元/晚。"
                    ),
                ),
                PolicySection(
                    heading="三类城市限额",
                    body=(
                        "三类城市住宿限额：A 档 350 元/晚，B 档 500 元/晚，C 档 700 "
                        "元/晚。"
                    ),
                ),
                PolicySection(
                    heading="住宿特殊情形",
                    body=(
                        "同一性别同事同行，原则上应合住标准间以节约开支；员工明确"
                        "拒绝合住的，按各自限额单独结算。因航班取消等不可抗力产生的"
                        "临时住宿，凭航司证明据实报销，不计入限额。"
                    ),
                ),
            ),
        ),
        PolicyDocument(
            doc_id="policy-booking",
            title="预订提前期与改签退改",
            category="预订",
            sections=(
                PolicySection(
                    heading="国内航班提前期",
                    body=(
                        "国内航班须至少提前 7 天预订；提前不足 7 天的，须在申请单中"
                        "注明紧急原因。提前 14 天以上预订可享受更优票价，鼓励尽早规划。"
                    ),
                ),
                PolicySection(
                    heading="国际航班提前期",
                    body=(
                        "国际及港澳台航班须至少提前 21 天预订，并预留签证办理时间。"
                        "涉及多个国家的行程须逐段说明必要性。"
                    ),
                ),
                PolicySection(
                    heading="酒店预订提前期",
                    body=(
                        "酒店须至少提前 3 天预订。展会期间房价上涨的，可在限额内选择"
                        "距离会场较远的酒店，通勤费用据实报销。"
                    ),
                ),
                PolicySection(
                    heading="改签与退票",
                    body=(
                        "因公改签产生的费用据实报销；因个人原因改签的，差价由个人承担。"
                        "行程取消应第一时间办理退票，未及时退票造成的损失由责任人承担。"
                    ),
                ),
            ),
        ),
        PolicyDocument(
            doc_id="policy-reimbursement",
            title="报销流程与凭证要求",
            category="报销",
            sections=(
                PolicySection(
                    heading="报销时限",
                    body=(
                        "出差结束后 15 个自然日内提交报销单；跨月出差的，最迟于次月 10 "
                        "日前提交。逾期未报销的，须由部门负责人书面说明原因。"
                    ),
                ),
                PolicySection(
                    heading="必备凭证",
                    body=(
                        "报销须附：出差申请单编号、交通票据、住宿发票、支付凭证。"
                        "电子发票须为原件 PDF，截图与翻拍件不予受理。"
                    ),
                ),
                PolicySection(
                    heading="不合规票据",
                    body=(
                        "以下票据不予报销：抬头非公司全称的发票、个人消费混开的发票、"
                        "超过 180 天未认证的增值税专用发票、以及任何形式的虚开票据。"
                    ),
                ),
                PolicySection(
                    heading="支付与结算",
                    body=(
                        "差旅费用原则上使用公司差旅卡或公务卡支付，以便对账。使用个人"
                        "垫付的，报销款于审批通过后 5 个工作日内到账。"
                    ),
                ),
            ),
        ),
        PolicyDocument(
            doc_id="policy-approval",
            title="审批权限与流程",
            category="审批",
            sections=(
                PolicySection(
                    heading="申请单必填要素",
                    body=(
                        "出差申请单须填写：出发地与目的地、起止日期、事由、预估费用、"
                        "同行人员。要素不全的申请单系统不予提交。"
                    ),
                ),
                PolicySection(
                    heading="分级审批权限",
                    body=(
                        "预估费用 5000 元以下由部门负责人审批；5000 元至 20000 元由"
                        "部门负责人与财务负责人双签；20000 元以上须报分管副总审批。"
                    ),
                ),
                PolicySection(
                    heading="特批情形",
                    body=(
                        "超出差标但业务必需的，可提交特批申请，说明超标的必要性，"
                        "由上一级负责人审批。同一人年度特批超过 3 次的，纳入合规抽查。"
                    ),
                ),
                PolicySection(
                    heading="审批时限",
                    body=(
                        "审批人应在收到申请后 2 个工作日内处理。超过 2 个工作日未处理"
                        "的，系统自动提醒；超过 5 个工作日的，可升级至上一级审批人。"
                    ),
                ),
            ),
        ),
        PolicyDocument(
            doc_id="policy-violations",
            title="违规情形与处理",
            category="合规",
            sections=(
                PolicySection(
                    heading="常见违规情形",
                    body=(
                        "常见违规包括：虚报差旅费用、超标准消费后拆单报销、将私人行程"
                        "混入公务行程、重复报销同一笔费用、以及代他人报销。"
                    ),
                ),
                PolicySection(
                    heading="处理措施",
                    body=(
                        "首次轻微违规的，责令退回并给予书面警告；金额较大或再次违规的，"
                        "视情节给予通报批评、暂停差旅权限，直至移交纪律处理。"
                    ),
                ),
                PolicySection(
                    heading="申诉",
                    body=(
                        "对违规认定有异议的，可在收到通知后 10 个工作日内向财务部提交"
                        "申诉材料，由合规小组复核。复核期间不影响已认定的处理措施执行。"
                    ),
                ),
            ),
        ),
        PolicyDocument(
            doc_id="policy-special",
            title="特殊情形与例外",
            category="例外",
            sections=(
                PolicySection(
                    heading="会议与培训",
                    body=(
                        "参加外部会议或培训，主办方统一安排住宿的，按主办方标准执行，"
                        "不受本政策限额约束，但须在申请单中注明会议名称与主办方。"
                    ),
                ),
                PolicySection(
                    heading="紧急出差",
                    body=(
                        "突发故障、客户现场事故等紧急出差可事后补办申请单，但须在返回"
                        "后 3 个工作日内补齐，并由部门负责人确认紧急性。"
                    ),
                ),
                PolicySection(
                    heading="延长停留",
                    body=(
                        "出差结束后因个人原因延长停留的，延长期间的住宿与交通费用自理，"
                        "返程票据仍按原标准报销。"
                    ),
                ),
                PolicySection(
                    heading="随行与陪同",
                    body=(
                        "客户或合作方随行的，公司人员按其本人标准执行，不得代客户"
                        "支付应酬之外的差旅费用。"
                    ),
                ),
            ),
        ),
    )


#: 演示数据的统一时间戳前缀（**固定值**，不用系统时间）。
#:
#: ⚠️ 时间戳写死而不是取 ``datetime.now()``：后者会让每次运行写入的字节不同，
#: 于是「幂等」的断言只能比对主键、比对不了整行，漂移也就发现不了
#: （见 src/storage/memory.py 的「不许用系统时间」）。
_CREATED_AT = "2026-10-02T09:00:00+08:00"
_UPDATED_AT = "2026-10-02T10:30:00+08:00"


def users() -> tuple[UserRecord, ...]:
    """返回演示用户（字面量）。

    Returns:
        `tuple[UserRecord, ...]`: 五个不同部门与职级的用户。
    """
    return (
        UserRecord("u1001", "张伟", "研发中心", "P7", "CC-1001"),
        UserRecord("u1002", "李娜", "市场部", "P6", "CC-2001"),
        UserRecord("u1003", "王强", "销售部", "P8", "CC-3001"),
        UserRecord("u1004", "陈静", "财务部", "P6", "CC-4001"),
        UserRecord("u1005", "刘洋", "人力资源部", "P5", "CC-5001"),
    )


def trips() -> tuple[TripRecord, ...]:
    """返回演示行程（字面量）。

    Returns:
        `tuple[TripRecord, ...]`: 四条不同用户、不同阶段的行程。
    """
    return (
        TripRecord("t2001", "u1001", "杭州", "北京", "2026-10-12", 2, "客户技术交流", "PLANNED"),
        TripRecord("t2002", "u1002", "上海", "广州", "2026-10-15", 3, "市场调研", "PLANNED"),
        TripRecord("t2003", "u1003", "深圳", "成都", "2026-10-20", 4, "渠道洽谈", "PLANNED"),
        TripRecord("t2004", "u1004", "北京", "上海", "2026-11-03", 1, "季度对账", "DONE"),
    )


def orders() -> tuple[TravelOrder, ...]:
    """返回演示订单（复用 :class:`~src.domain.entities.TravelOrder`）。

    ⚠️ 状态用 :class:`~src.domain.enums.OrderStatus` 的**成员**而不是裸字符串：
    订单状态是要落库、要对账的业务事实，用枚举能让「写错一个状态名」在构造期
    就暴露，而不是等到查询时才发现某行状态谁也匹配不上。

    Returns:
        `tuple[TravelOrder, ...]`: 四种状态各一条（已通过 / 已支付 / 已取消 / 已完成）。
    """
    return (
        TravelOrder(
            order_id="o3001",
            user_id="u1001",
            kind="flight",
            status=OrderStatus.APPROVED,
            title="杭州 → 北京 CA1701",
            amount=1280.0,
            detail={
                "carrier": "CA1701",
                "cabin": CabinClass.ECONOMY.value,
                "depart_date": "2026-10-12",
            },
            created_at=_CREATED_AT,
            updated_at=_UPDATED_AT,
        ),
        TravelOrder(
            order_id="o3002",
            user_id="u1002",
            kind="hotel",
            status=OrderStatus.PAID,
            title="广州珠江新城亚朵酒店 2 晚",
            amount=1160.0,
            detail={
                "city": "广州",
                "nights": "2",
                "price_per_night": "580",
            },
            created_at=_CREATED_AT,
            updated_at=_UPDATED_AT,
        ),
        TravelOrder(
            order_id="o3003",
            user_id="u1003",
            kind="train",
            status=OrderStatus.CANCELLED,
            title="深圳 → 成都 G2964",
            amount=860.0,
            detail={
                "carrier": "G2964",
                "cabin": CabinClass.ECONOMY.value,
                "depart_date": "2026-10-20",
            },
            created_at=_CREATED_AT,
            updated_at=_UPDATED_AT,
        ),
        TravelOrder(
            order_id="o3004",
            user_id="u1004",
            kind="flight",
            status=OrderStatus.COMPLETED,
            title="北京 → 上海 MU5138",
            amount=990.0,
            detail={
                "carrier": "MU5138",
                "cabin": CabinClass.ECONOMY.value,
                "depart_date": "2026-11-03",
            },
            created_at=_CREATED_AT,
            updated_at=_UPDATED_AT,
        ),
    )


def approvals() -> tuple[ApprovalRequest, ...]:
    """返回演示申请单（复用 :class:`~src.domain.entities.ApprovalRequest`）。

    Returns:
        `tuple[ApprovalRequest, ...]`: 待审批 / 已通过 / 已驳回各一条。
    """
    return (
        ApprovalRequest(
            request_id="a4001",
            user_id="u1003",
            title="成都渠道洽谈差旅申请",
            status=OrderStatus.PENDING_APPROVAL,
            amount=4200.0,
            destination="成都",
            depart_date="2026-10-20",
            days=4,
            created_at=_CREATED_AT,
            updated_at=_UPDATED_AT,
        ),
        ApprovalRequest(
            request_id="a4002",
            user_id="u1002",
            title="广州市场调研申请",
            status=OrderStatus.APPROVED,
            amount=3600.0,
            destination="广州",
            depart_date="2026-10-15",
            days=3,
            created_at=_CREATED_AT,
            updated_at=_UPDATED_AT,
        ),
        ApprovalRequest(
            request_id="a4003",
            user_id="u1005",
            title="上海行业峰会参会申请",
            status=OrderStatus.REJECTED,
            amount=2800.0,
            destination="上海",
            depart_date="2026-11-05",
            days=2,
            created_at=_CREATED_AT,
            updated_at=_UPDATED_AT,
            approver_note="非年度计划内，建议线上参会。",
        ),
    )


def build_plan() -> SeedPlan:
    """把全部数据装配成一个 :class:`SeedPlan`（纯函数，零 I/O）。

    ⚠️ dry-run 与真写都从这里取数据，保证「打印出来的」就是「会写进去的」。

    Returns:
        `SeedPlan`: 本次要写入的全部内容。
    """
    return SeedPlan(
        policy_documents=policy_documents(),
        users=users(),
        trips=trips(),
        orders=orders(),
        approvals=approvals(),
    )


# ==============================================================================
# 三、dry-run 渲染 —— 不连任何服务，只把「将要写入的东西」打印出来
# ==============================================================================


def render_dry_run(plan: SeedPlan, *, kb_user: str = KB_USER_ID) -> str:
    """把一份 :class:`SeedPlan` 渲染成给人看的中文文本。

    ⚠️ 渲染结果**只依赖入参**（不含时间、不含随机数），因此它是可断言的：
    ``render_dry_run(build_plan()) == render_dry_run(build_plan())`` 必然成立，
    而这条等式正是「确定性」最直接的证据。

    Args:
        plan (`SeedPlan`): 待渲染的数据。
        kb_user (`str`): 知识库归属的用户 id（默认 :data:`KB_USER_ID`，
            与 CLI ``--kb-user`` 一致）。它必须出现在 dry-run 输出里 ——
            归属写错是「数据在、就是看不见」的头号原因，先看这一行能省掉
            一整轮「去 Milvus 里翻数据」的排查。

    Returns:
        `str`: 多行中文说明（含幂等键说明）。
    """
    lines: list[str] = []
    total_chunks = sum(doc.chunk_count for doc in plan.policy_documents)

    lines.append("▶ 政策文档（将写入 Milvus）")
    lines.append(
        f"    共 {len(plan.policy_documents)} 篇，{total_chunks} 个 chunk"
        f"（一篇一节一个 chunk，幂等键 document_id = '{POLICY_DOC_ID_PREFIX}<doc_id>'）"
    )
    lines.append(
        f"    归属知识库：{KB_NAME}（id={KB_ID}，user_id={kb_user}）"
        "—— 必须与该 user 的登录身份一致，否则用户列不到这个 KB"
    )
    for doc in plan.policy_documents:
        lines.append(
            f"    - [{doc.doc_id}] {doc.title}"
            f"（分类：{doc.category}，{doc.chunk_count} 节）"
        )

    lines.append("")
    lines.append("▶ 业务数据（将写入业务库 business schema）")
    lines.append(f"    用户   {len(plan.users)} 条（幂等键 user_id）")
    for user in plan.users:
        lines.append(
            f"    - [{user.user_id}] {user.display_name}"
            f"（{user.department} / {user.employee_level}）"
        )
    lines.append(f"    行程   {len(plan.trips)} 条（幂等键 trip_id）")
    for trip in plan.trips:
        lines.append(
            f"    - [{trip.trip_id}] {trip.origin}→{trip.destination}"
            f" {trip.depart_date}（{trip.days} 天，{trip.status}）"
        )
    lines.append(f"    订单   {len(plan.orders)} 条（幂等键 order_id）")
    for order in plan.orders:
        lines.append(
            f"    - [{order.order_id}] {order.title}"
            f"（{order.status.value}，{order.amount:.0f} 元）"
        )
    lines.append(f"    申请单 {len(plan.approvals)} 条（幂等键 request_id）")
    for request in plan.approvals:
        lines.append(
            f"    - [{request.request_id}] {request.title}"
            f"（{request.status.value}，{request.amount:.0f} 元）"
        )

    lines.append("")
    lines.append(
        "（dry-run：未连接任何外部服务。这次改动数据构造逻辑后，先看这份输出。）"
    )
    return "\n".join(lines)


# ==============================================================================
# 四、业务库写入
# ==============================================================================


def _business_tables(schema: str | None) -> tuple[MetaData, dict[str, Table]]:
    """声明业务库表的**最小结构**。

    ⚠️ 这些 ``CREATE TABLE`` 定义住在种子脚本里，是**当前阶段**的权宜：
    ``src/storage/postgres.py``（P4 的正式落库实现）尚未落地，本脚本要能先把
    演示数据写进去。等正式的仓储实现出现后，这里的表定义应当搬去那边，本脚本
    改为调用它 —— 在那之前，表结构以此处为准。

    ⚠️ ``schema`` 由 :func:`src.storage.engine.business_schema_for` 决定：
    PostgreSQL 下是 ``business``，sqlite（测试）下是 ``None``。**不能**在这里
    写死 ``"business"``，否则 sqlite 会生成 ``CREATE TABLE business.users``
    直接语法错误 —— 而本地测试正是跑在 sqlite 上的（同 engine.py 的说明）。

    Args:
        schema (`str | None`): 表所在的 schema；``None`` 表示不限定。

    Returns:
        `tuple[MetaData, dict[str, Table]]`: metadata 与「表名 → Table」字典。
    """
    metadata = MetaData(schema=schema)
    tables = {
        "users": Table(
            "users",
            metadata,
            Column("user_id", String(64), primary_key=True),
            Column("display_name", String(64), nullable=False),
            Column("department", String(64), nullable=False),
            Column("employee_level", String(32), nullable=False),
            Column("cost_center", String(32), nullable=False),
        ),
        "trips": Table(
            "trips",
            metadata,
            Column("trip_id", String(64), primary_key=True),
            Column("user_id", String(64), nullable=False),
            Column("origin", String(64), nullable=False),
            Column("destination", String(64), nullable=False),
            Column("depart_date", String(10), nullable=False),
            Column("days", Integer, nullable=False),
            Column("purpose", String(128), nullable=False),
            Column("status", String(32), nullable=False),
        ),
        "orders": Table(
            "orders",
            metadata,
            Column("order_id", String(64), primary_key=True),
            Column("user_id", String(64), nullable=False),
            Column("kind", String(16), nullable=False),
            Column("status", String(32), nullable=False),
            Column("title", String(128), nullable=False),
            Column("amount", Float, nullable=False),
            # 明细结构随 kind 而变，用 JSON 而不是拆列 —— 与
            # TravelOrder.detail: dict[str, str] 的类型一致。
            Column("detail", JSON, nullable=False),
            Column("created_at", String(32), nullable=False),
            Column("updated_at", String(32), nullable=False),
        ),
        "approvals": Table(
            "approvals",
            metadata,
            Column("request_id", String(64), primary_key=True),
            Column("user_id", String(64), nullable=False),
            Column("title", String(128), nullable=False),
            Column("status", String(32), nullable=False),
            Column("amount", Float, nullable=False),
            Column("destination", String(64), nullable=False),
            Column("depart_date", String(10), nullable=False),
            Column("days", Integer, nullable=False),
            Column("created_at", String(32), nullable=False),
            Column("updated_at", String(32), nullable=False),
            Column("approver_note", String(255), nullable=False),
        ),
    }
    return metadata, tables


def _to_rows(plan: SeedPlan) -> dict[str, list[dict]]:
    """把 :class:`SeedPlan` 摊平成「表名 → 行字典列表」。

    ⚠️ 枚举值一律取 ``.value``（``OrderStatus.PAID`` → ``"PAID"``）：写进数据库
    的应当是**值**而不是枚举对象 —— 后者在 sqlite 上会被存成一个谁也不认得的
    字符串，而问题只在查询时暴露。

    Args:
        plan (`SeedPlan`): 待摊平的数据。

    Returns:
        `dict[str, list[dict]]`: 四张表的行。
    """
    return {
        "users": [
            {
                "user_id": user.user_id,
                "display_name": user.display_name,
                "department": user.department,
                "employee_level": user.employee_level,
                "cost_center": user.cost_center,
            }
            for user in plan.users
        ],
        "trips": [
            {
                "trip_id": trip.trip_id,
                "user_id": trip.user_id,
                "origin": trip.origin,
                "destination": trip.destination,
                "depart_date": trip.depart_date,
                "days": trip.days,
                "purpose": trip.purpose,
                "status": trip.status,
            }
            for trip in plan.trips
        ],
        "orders": [
            {
                "order_id": order.order_id,
                "user_id": order.user_id,
                "kind": order.kind,
                "status": order.status.value,
                "title": order.title,
                "amount": order.amount,
                "detail": dict(order.detail),
                "created_at": order.created_at,
                "updated_at": order.updated_at,
            }
            for order in plan.orders
        ],
        "approvals": [
            {
                "request_id": request.request_id,
                "user_id": request.user_id,
                "title": request.title,
                "status": request.status.value,
                "amount": request.amount,
                "destination": request.destination,
                "depart_date": request.depart_date,
                "days": request.days,
                "created_at": request.created_at,
                "updated_at": request.updated_at,
                "approver_note": request.approver_note,
            }
            for request in plan.approvals
        ],
    }


async def _probe_business(engine: AsyncEngine) -> None:
    """探一次业务库连通性（一条 ``SELECT 1``）。

    ⚠️ 单独成一个函数，是为了让测试能**只替换这一步**来模拟「连不上」，
    而不必伪造整个 engine。它必须是第一件与数据库交互的事 —— 见
    :func:`seed_business_data` 的异常包装范围。

    Args:
        engine (`AsyncEngine`): 业务库引擎。

    Raises:
        Exception: 数据库不可达、认证失败等，原样抛出给上层包装。
    """
    async with engine.connect() as conn:
        await conn.execute(text("SELECT 1"))


async def _open_business_engine(settings: Settings) -> AsyncEngine:
    """构造业务库引擎并**先探一次连通性**。

    ⚠️ 「构造」本身不连接（``create_async_engine`` 是惰性的），所以必须在
    这里显式探一次 —— 否则「连不上」会推迟到建表时才暴露，而那时的报错
    会被混进「SQL 写错了」那一类，指向错误的方向。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `AsyncEngine`: 已确认可达的业务库引擎。

    Raises:
        SeedConnectionError: 引擎构造或连通性探测失败。
    """
    engine: AsyncEngine | None = None
    try:
        engine = build_business_engine(settings)
        await _probe_business(engine)
    except Exception as exc:  # noqa: BLE001 —— 收敛成一条中文错误，见类文档
        if engine is not None:
            # ⚠️ 探测失败也要释放引擎：``create_async_engine`` 可能已经起了
            # 连接池线程，不 dispose 会让脚本在退出时挂住（同 milvus_init）。
            try:
                await engine.dispose()
            except Exception:  # noqa: BLE001
                pass
        raise SeedConnectionError(
            f"无法连接业务库（{redact(settings.db.url)}）：{safe_error(exc)}\n"
            "下一步：确认 PostgreSQL 已启动且可达 —— core 档含它（make up）；\n"
            "       或先用 `python scripts/seed_data.py --dry-run` 检查将要写入的内容。"
        ) from exc
    return engine


async def _insert_missing(
    conn: object,
    table: Table,
    rows: list[dict],
    key: str,
) -> int:
    """只插入主键尚不存在的行，返回**本次新增**的条数。

    ⚠️ 用「先查后插」而不是数据库方言专属的 ``ON CONFLICT DO NOTHING``：
    后者在 sqlite 与 PostgreSQL 上写法不同，会把这个脚本绑死在某一方言上；
    而本项目本地测试跑 sqlite、线上跑 PostgreSQL，两者都要能跑。
    代价是多一次 SELECT —— 对一个演示种子脚本完全可接受。

    Args:
        conn (`object`): SQLAlchemy 的 ``AsyncConnection``（用 ``object`` 标注
            以免在类型上依赖具体实现）。
        table (`Table`): 目标表。
        rows (`list[dict]`): 待写入的行（每个 dict 的键是列名）。
        key (`str`): 主键列名，也是幂等键。

    Returns:
        `int`: 实际新增的行数（重复的键被跳过，不计入）。
    """
    if not rows:
        return 0
    existing = set((await conn.execute(select(table.c[key]))).scalars().all())  # type: ignore[attr-defined]
    missing = [row for row in rows if row[key] not in existing]
    if missing:
        await conn.execute(table.insert(), missing)  # type: ignore[attr-defined]
    return len(missing)


async def seed_business_data(
    settings: Settings,
    plan: SeedPlan,
    *,
    engine: AsyncEngine | None = None,
) -> dict[str, int]:
    """把业务数据幂等地写进业务库。

    ⚠️ ``engine`` 参数存在的唯一理由是**测试**：传入一个 sqlite 内存引擎，
    就能在完全离线的情况下跑两遍本函数、断言第二遍新增为 0 —— 这正是
    「幂等」最有力的证据。生产路径不传它，由本函数自建并释放。

    Args:
        settings (`Settings`): 配置。
        plan (`SeedPlan`): 待写入的数据。
        engine (`AsyncEngine | None`, optional): 复用的引擎；``None`` 时自建。

    Returns:
        `dict[str, int]`: 每张表本次**新增**的行数。

    Raises:
        SeedConnectionError: 自建引擎时连不上业务库。
    """
    owns_engine = engine is None
    active = engine if engine is not None else await _open_business_engine(settings)
    try:
        schema = business_schema_for(settings)
        # 幂等：schema 与表都用 IF NOT EXISTS 语义（create_all 的默认行为）。
        await ensure_business_schema(active, schema)
        metadata, tables = _business_tables(schema)
        rows_by_table = _to_rows(plan)

        counts: dict[str, int] = {}
        async with active.begin() as conn:
            await conn.run_sync(metadata.create_all)
            for name, key in (
                ("users", "user_id"),
                ("trips", "trip_id"),
                ("orders", "order_id"),
                ("approvals", "request_id"),
            ):
                counts[name] = await _insert_missing(
                    conn,
                    tables[name],
                    rows_by_table[name],
                    key,
                )
        return counts
    finally:
        if owns_engine:
            await active.dispose()


# ==============================================================================
# 五、Milvus 写入
# ==============================================================================


def _framework_chunks(doc: PolicyDocument) -> list[Chunk]:
    """把一篇政策文档转成框架的 :class:`Chunk` 列表。

    ⚠️ 每个 chunk 的正文带上「文档标题｜小节标题」前缀。理由：检索命中一个
    孤零零的小节时，模型分不清它出自哪份政策的哪一节；而引用来源必须能回溯到
    具体条款。前缀就是这条追溯链在**向量内容**里留下的锚点。

    Args:
        doc (`PolicyDocument`): 政策文档。

    Returns:
        `list[Chunk]`: 小节数 == ``doc.chunk_count``。
    """
    chunks: list[Chunk] = []
    for index, section in enumerate(doc.sections):
        chunks.append(
            Chunk(
                content=TextBlock(
                    text=f"{doc.title}｜{section.heading}\n{section.body}",
                ),
                source=doc.doc_id,
                chunk_index=index,
                total_chunks=len(doc.sections),
                metadata={
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "category": doc.category,
                    "heading": section.heading,
                },
            ),
        )
    return chunks


async def _close_vector_store(store: object) -> None:
    """尽力关闭向量库客户端。

    ⚠️ 失败只吞掉、不抛出：关闭发生在 ``finally`` 里，此时通常已经有一个
    更值得上报的异常在传播 —— 让一次清理失败盖掉真正的线索，是比泄漏一个
    连接更坏的结果。用 ``__aexit__`` 而不是 ``close()``：框架把关闭逻辑
    （含内嵌 Lite 的 server 释放）都放在 ``__aexit__`` 里。

    Args:
        store (`object`): 已构造的向量库客户端。
    """
    try:
        await store.__aexit__(None, None, None)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


async def _open_vector_store(settings: Settings) -> object:
    """构造向量库客户端并**先探一次连通性**。

    ⚠️ 与业务库同理：``MilvusLiteStore`` 的连接是惰性的，不在这里探一次，
    「连不上」会推迟到第一次写入时才炸。

    ⚠️⚠️ 「集合不存在」在这里被**显式**认定为播种失败（``SeedConnectionError``），
    而不是留到写入时由框架懒建。理由有三：

      1. 本脚本的契约就是**不建集合** —— 集合的维度/度量/索引由
         ``milvus_init.py`` 负责核验（见模块 docstring）。若这里默许懒建，
         就等于用一套没人核验过的参数偷偷建了一个集合，把「形态不对」
         这种最坏失败藏了起来。
      2. 尽早失败：集合缺失是一条**可执行**的运维动作（``make milvus_init``），
         越早给出这句话，越不用等到灌到一半才看见一条看不懂的维报错。
      3. 顺序契约：测试把 ``build_vector_store`` 换成桩，要求「先碰向量库、
         再碰 storage」；把集合检查放在这一步，缺集合时根本不会去连业务库。

    Args:
        settings (`Settings`): 配置。

    Returns:
        `object`: 已确认可达**且集合已存在**的向量库客户端。

    Raises:
        SeedConnectionError: 连不上 Milvus，或集合尚未创建。
    """
    store = build_vector_store(settings)
    try:
        exists = await store.has_collection(settings.milvus.collection)
    except Exception as exc:  # noqa: BLE001 —— 收敛成中文错误
        await _close_vector_store(store)
        raise SeedConnectionError(
            f"无法连接 Milvus（{redact(settings.milvus.uri)}）：{safe_error(exc)}\n"
            "下一步：确认 Milvus 已启动且可达 —— core 档含它（make up）；\n"
            "       若集合尚未创建，先跑 `make milvus_init`，再重跑本脚本。"
        ) from exc

    if not exists:
        # ⚠️ 这一条以前**不会**抛错（旧实现只是把「先跑 milvus_init」写进了
        # 「连不上」的文案里，集合缺失时反而一路走到写入才爆）。现在它是
        # 一条独立、明确的失败：连接是通的，只是还没初始化。
        await _close_vector_store(store)
        raise SeedConnectionError(
            f"Milvus 可达，但集合 {settings.milvus.collection!r} 尚未创建"
            f"（{redact(settings.milvus.uri)}）。\n"
            "下一步：先跑 `make milvus_init` 建集合，再重跑本脚本 —— "
            "本脚本按契约**不建集合**（集合的形态由 milvus_init.py 负责核验）。"
        )
    return store


@contextlib.asynccontextmanager
async def _open_storage(
    settings: Settings,
    storage: object | None,
) -> AsyncIterator[object]:
    """进入 storage 的异步上下文（生产时自建，测试时可注入替身）。

    ⚠️ ``storage`` 参数存在的唯一理由是**测试**（与 :func:`seed_business_data`
    的 ``engine`` 参数同构）：传一个内存替身就能在完全离线的情况下跑通
    「建 KB 记录 → 经管理器写 chunk」这条链路，并断言写进去的 metadata。

    ⚠️ 生产路径的 ``build_storage`` 用**局部导入**：``src.server.app`` 在
    import 期就会执行 ``app = create_root_app()``（装配整套服务）。把它放在
    顶部的 import 里，会让任何 import 本脚本的地方（包括 ``make test`` 的
    收集阶段）都被迫装配一次连真实外部服务的应用。局部导入把这一副作用
    推迟到「真的要写 storage」的那一刻。

    Args:
        settings (`Settings`): 配置。
        storage (`object | None`): 复用的 storage；``None`` 时按配置自建。

    Yields:
        `object`: 已进入上下文的 storage（``__aenter__`` 返回自身）。
    """
    if storage is not None:
        async with storage as entered:  # type: ignore[attr-defined]
            yield entered if entered is not None else storage
        return

    from src.server.app import build_storage

    active = build_storage(settings)
    async with active as entered:
        yield entered if entered is not None else active


async def ensure_kb_embedding_credential(
    storage: object,
    settings: Settings,
    user_id: str,
) -> tuple[str, str, str]:
    """为政策知识库钉一条**当前配置下真正可用**的向量凭据。

    框架的知识库链路**不看** :mod:`src.web_embedding` 那条三合一降级链，
    它只认 KB 记录里 ``embedding_model_config`` 指着的**凭据类型 + 凭据 id**
    （``app/_service/_embedding.py::build_embedding_model`` →
    ``CredentialFactory.get_credential_class``）。也就是说：KB 用哪一档向量
    模型，是在**写 KB 记录那一刻**由这里决定的，之后不会再变。

    因此本函数只有两条路，判据与 ``src/llm/factory.py`` 的
    ``has_api_key`` / ``should_use_mock`` **同源**（不另立一套）：

        · 有密钥 ⇒ 用配置的提供方凭据（DashScope/OpenAI…），模型名取
          ``settings.embedding.model`` —— 与 :mod:`src.web_embedding` 的
          云端档**同一个名字**，两边的向量因此可比；
        · 没有密钥（或显式降级）⇒ 沿用 Mock 凭据。

    ⚠️⚠️ 这里曾经**无条件**写 Mock 凭据，后果是「配置了密钥也依然用假向量」：
    末尾那句告警让运维「配置 DASHSCOPE_API_KEY 后重灌」，而重灌只会再写一次
    Mock —— 检测到了降级，却给了一条**永远修不好它**的 remediation。
    现在同一句告警里的「重灌」是真的有用的：重灌会重新钉记录 + 先删后插
    重算每个 chunk 的向量。

    ⚠️ ``local``（本地 ONNX）档**故意不在这里支持**：框架侧没有对应的凭据
    类型（:class:`~src.web_embedding.local.LocalOnnxEmbeddingModel` 是本项目
    自己的类，不挂在任何 ``CredentialBase`` 上）。真要用本地模型跑知识库，
    得先给它配一个能返回该模型类的凭据类型 —— 那是另一件事，不该在这里
    用「悄悄退回 Mock」来假装支持。

    ⚠️ 凭据写进 storage 时**带上了 API key 明文**（与
    ``src/llm/system_credential.py::ensure_system_credential`` 完全同形）。
    这是框架的存储约定；全项目 engine 都开了 ``hide_parameters``
    （``src/storage/engine.py``），因此日志里绑定参数位置是 ``(...)``。
    本函数**绝不**把密钥本身打印或放进返回值。

    Args:
        storage (`object`): 框架的存储实现（或测试替身）。
        settings (`Settings`): 配置。
        user_id (`str`): 凭据属主 —— **必须**是知识库的归属者，因为框架的
            KB 链路是「owner-internal」解析（``get_credential(kb.user_id, ...)``），
            指向别人名下的凭据会解析不到，症状是检索静默无出处。

    Returns:
        `tuple[str, str, str]`: ``(凭据类型, 凭据 id, 模型名)``，直接喂给
        ``EmbeddingModelConfig``。

    Raises:
        Exception: 存储写入失败时原样抛出（主键冲突除外，见
            :func:`~src.llm.preset_credential.upsert_preset_credential`）。
    """
    if should_use_mock(settings) or not has_api_key(settings):
        credential_id = await ensure_mock_credential(storage, user_id)
        return MOCK_CREDENTIAL_TYPE, credential_id, MOCK_EMBEDDING_MODEL_NAME

    credential = build_credential(settings)
    # 固定 id + 固定显示名：``upsert_credential`` 对「属于本用户的预设 id」
    # 是就地更新，所以改了密钥/换了模型重跑一次就生效，不会堆出第二条记录。
    credential.id = KB_EMBEDDING_CREDENTIAL_ID
    credential.name = KB_EMBEDDING_CREDENTIAL_NAME
    credential_id = await upsert_preset_credential(storage, user_id, credential)
    # ``credential.type`` 是 pydantic 的 Literal 字段（如
    # ``"dashscope_credential"``）—— 用**类字段**而不是猜字符串，
    # 与 ``system_credential_type`` 的做法一致。
    credential_type = str(type(credential).model_fields["type"].default)
    return credential_type, credential_id, settings.embedding.model


async def seed_policy_documents(
    settings: Settings,
    plan: SeedPlan,
    *,
    storage: object | None = None,
    user_id: str = KB_USER_ID,
    kb_id: str = KB_ID,
) -> dict[str, int]:
    """把政策文档幂等地灌进 Milvus —— **经由 KB 管理器**，而不是裸建集合。

    ⚠️⚠️ 为什么必须经管理器（这是本函数此前最大的缺陷）：
    管理器的检索路径 :meth:`SingleCollectionKbManager.get_knowledge` 会给句柄
    绑一个 ``metadata_filter = {"aligo_kb_id": ..., "aligo_user_id": ...}``。
    框架在**写入**方向也会把这些键强制合并进每个 chunk 的 metadata
    （``rag/_knowledge.py::insert_document`` 的优先级最高项）。因此
    「经管理器写」与「被管理器检索到」是同一件事的两面：不经管理器写进去的
    chunk，metadata 里没有这两个键，服务端一过滤就全部落空 ——
    表现为「知识库问答永远没有出处」，而数据其实好端端躺在库里。

    ⚠️ 为什么用**固定 id** 的 ``storage.upsert_knowledge_base`` 而不是
    ``manager.create_knowledge_base``：后者的 ``KnowledgeBaseRecord.id`` 是
    ``default_factory``，每调用一次就生成一个新 id，反复播种会堆出多条 KB 记录
    并重复灌向量。固定 id + upsert 让记录与向量两层同时幂等。我们因此绕过了
    管理器里那道 ``DimensionPolicy`` 校验，但维度直接用
    ``settings.milvus.dimension``，与单集合策略**按构造相等** —— 校验本就
    不可能失败（且 ``Settings`` 的 validator 已保证 embedding 与 milvus 维度一致）。

    ⚠️ **向量模型在写 KB 记录时钉死**（见 :func:`ensure_kb_embedding_credential`）：
    配了密钥就用真模型、没有就用 Mock，两档都以
    ``settings.embedding.dimension`` 为准（schema 已保证它等于
    ``settings.milvus.dimension``）。因此「换向量模型」的正确操作是
    **改配置 + 重跑本函数**，而不是改 KB 记录 —— 且必须重跑，
    因为已写入的向量是用旧模型算的。

    ⚠️ 幂等键 = ``document_id``（``seed-<doc_id>``）。框架的 ``insert`` 已按
    ``(document_id, chunk_index)`` 走 ``upsert``，重复插入同内容**不会**产生
    重复行。但我们仍然**先删后插**，理由是 upsert 覆盖不了的边界：若某篇文档的
    小节数被改动（少了几节），旧的多余 chunk 会被留下并且**仍可被检索到** ——
    一条已删除的条款继续被引用，比重复数据更危险。先删整个 document_id 再插，
    得到的才是「这一篇现在的样子」。

    Args:
        settings (`Settings`): 配置。
        plan (`SeedPlan`): 待写入的数据。
        storage (`object | None`, optional): 复用的 storage；``None`` 时自建
            （见 :func:`_open_storage`）。生产路径不传它。
        user_id (`str`): 知识库归属用户（默认 :data:`KB_USER_ID`）。
        kb_id (`str`): 知识库记录 id（默认 :data:`KB_ID`）。

    Returns:
        `dict[str, int]`: ``documents`` / ``chunks`` 两个计数，实际使用的
        ``embedding_model``（类名，便于发现悄悄降级到假向量），以及知识库
        ``knowledge_base_id`` 与 ``knowledge_base_created``（本次是否新建）。

    Raises:
        SeedConnectionError: 连不上 Milvus 或集合不存在。
    """
    # ⚠️⚠️ 顺序契约：**先探向量库连通，再碰 storage**。
    # tests/test_seed_data.py 把 ``build_vector_store`` 换成一调用即炸的桩来
    # 证明 dry-run 不连任何服务；同时 ``test_milvus_connection_error_shape``
    # 依赖「向量库连不上时，根本没走到 storage」这条顺序。改动本函数时不要
    # 把 storage 的构造提到这一行之前。
    store = await _open_vector_store(settings)

    # ⚠️ seed 进程不建 app，因此不会走 ``create_app`` 里那次凭据类型注册。
    # 不在这里手动注册，``get_knowledge`` 内层的 ``CredentialFactory.from_dict``
    # 遇到未注册的 ``aligo_mock_credential`` 会直接 ValidationError ——
    # 而本脚本正是要经 ``get_knowledge`` 拿句柄。
    CredentialFactory.register_credential(MockCredential)

    try:
        async with _open_storage(settings, storage) as active:
            # 幂等地为本用户补上向量凭据，拿到 embedding 用的 credential_id。
            # KB 记录的 embedding_model_config 必须指向一条**真实存在**的凭据记录，
            # 否则 get_knowledge 会以「凭据不存在」为由拒绝解析句柄。
            #
            # ⚠️ 有密钥就用真模型（见 ensure_kb_embedding_credential 的
            # 「这里曾经无条件写 Mock」那段）：否则「配了密钥也仍是假向量」，
            # 而告警给出的补救办法恰好无效。
            (
                embedding_credential_type,
                credential_id,
                embedding_model_name,
            ) = await ensure_kb_embedding_credential(active, settings, user_id)

            record = KnowledgeBaseRecord(
                id=kb_id,
                user_id=user_id,
                data=KnowledgeBaseData(
                    name=KB_NAME,
                    description=KB_DESCRIPTION,
                    embedding_model_config=EmbeddingModelConfig(
                        type=embedding_credential_type,
                        credential_id=credential_id,
                        model=embedding_model_name,
                        dimensions=settings.milvus.dimension,
                    ),
                    # ⚠️ 单集合策略下**所有** KB 指向同一个集合 —— 这里必须
                    # 是契约集合名，而不是像框架那样拼一个 ``kb_<uuid>``。
                    collection_name=settings.milvus.collection,
                ),
            )
            existing = await active.get_knowledge_base(user_id, kb_id)  # type: ignore[attr-defined]
            await active.upsert_knowledge_base(user_id, record)  # type: ignore[attr-defined]

            # 经管理器拿运行时句柄：它内部会解析凭据、构造 embedding 模型，
            # 并绑上本 KB 的 metadata 作用域（写入与检索双向生效）。
            manager = SingleCollectionKbManager(
                storage=active,  # type: ignore[arg-type]
                vector_store=store,  # type: ignore[arg-type]
                settings=settings,
            )
            handle = await manager.get_knowledge(user_id, kb_id)

            total_chunks = 0
            for doc in plan.policy_documents:
                document_id = policy_document_id(doc.doc_id)
                chunks = _framework_chunks(doc)
                # 见 docstring：先删后插，避免改小节后残留旧 chunk。
                await handle.delete_document(document_id)
                await handle.insert_document(
                    chunks,
                    document_id=document_id,
                    document_metadata={
                        "doc_id": doc.doc_id,
                        "title": doc.title,
                        "category": doc.category,
                        "seeded_by": "scripts/seed_data.py",
                    },
                )
                total_chunks += len(chunks)

            return {
                "documents": len(plan.policy_documents),
                "chunks": total_chunks,
                # ⚠️ **必须**先剥掉 ``BoundedEmbeddingModel`` 这层包装再取类名。
                # 句柄上的向量模型被截止时间包装套着，直接取 ``type(...).__name__``
                # 得到的是 "BoundedEmbeddingModel" —— 于是下面那句
                # 「本次用的是假向量」的告警**永远不会触发**，而它恰恰是
                # 唯一能发现「悄悄降级到 Mock」的地方（从检索输出看不出来）。
                # 同样的剥壳在 ``src/knowledge/rag.py`` 里也做了一遍。
                "embedding_model": type(
                    unwrap_embedding_model(handle.embedding_model)
                ).__name__,
                "knowledge_base_id": kb_id,
                "knowledge_base_created": existing is None,
            }
    finally:
        # ⚠️ 必须关：``MilvusClient`` 会起后台线程，不退的话脚本会挂住。
        await _close_vector_store(store)


# ==============================================================================
# 六、命令行入口
# ==============================================================================


async def _run(
    settings: Settings,
    *,
    only: str,
    dry_run: bool,
    kb_user: str = KB_USER_ID,
) -> int:
    """执行播种。

    Args:
        settings (`Settings`): 配置。
        only (`str`): ``all`` / ``policy`` / ``business``。
        dry_run (`bool`): 为真时只打印、不连接。
        kb_user (`str`): 知识库归属的用户 id（默认 :data:`KB_USER_ID`）。

    Returns:
        `int`: 进程退出码。
    """
    plan = build_plan()

    if dry_run:
        print(render_dry_run(plan, kb_user=kb_user))
        print(f"\n✅ dry-run 完成（env={settings.app.env}，未连接任何外部服务）。")
        return 0

    if only in ("all", "policy"):
        print("▶ 正在写入政策文档到 Milvus...")
        result = await seed_policy_documents(settings, plan, user_id=kb_user)
        print(
            f"✅ 政策文档：{result['documents']} 篇 / {result['chunks']} 个 chunk "
            f"（向量模型：{result['embedding_model']}）。"
        )
        print(
            f"   知识库记录：{result['knowledge_base_id']}"
            f"（user_id={kb_user}，"
            + ("本次新建" if result["knowledge_base_created"] else "复用已有")
            + "）。"
        )
        if result["embedding_model"] == "MockEmbeddingModel":
            # ⚠️ Mock 之间没有语义，检索会「成功」但结果无意义 —— 必须显式提醒，
            # 因为从检索的输出上看不出这一点（见 src/web_embedding 的模块文档）。
            #
            # ⚠️ 这段 remediation 必须与 ``ensure_kb_embedding_credential``
            # 的实际行为一致：KB 记录在**写入那一刻**就把向量模型钉死了，
            # 所以「重灌」是**真的**能把 Mock 换掉的（重跑会重新钉记录 +
            # 先删后插重算每个 chunk 的向量）—— 前提是有密钥。
            # 曾经这里写着「安装 fastembed 后重灌」，那对本脚本的链路
            # **完全无效**：框架侧没有本地 ONNX 档的凭据类型，
            # 装了 fastembed 也只能让 src/web_embedding 那条链受益。
            print(
                "   ⚠️ 本次用的是确定性假向量（Mock）—— 检索能返回结果但**没有语义**。\n"
                "      原因：本次运行时 DASHSCOPE_API_KEY 为空（或已显式降级）。\n"
                "      补救：配好密钥后**重跑本脚本** —— 知识库的向量模型是在写 KB\n"
                "      记录时钉住的，重跑会重新钉住并重算全部分块向量。\n"
                "      （注意：本链路不支持本地 ONNX 档，装 fastembed 对本脚本无效。）"
            )

    if only in ("all", "business"):
        print("▶ 正在写入业务数据到业务库...")
        counts = await seed_business_data(settings, plan)
        detail = "、".join(f"{name} 新增 {count}" for name, count in counts.items())
        print(f"✅ 业务数据：{detail}（重复运行时均为 0）。")

    return 0


def main() -> int:
    """命令行入口。

    Returns:
        `int`: 进程退出码。
    """
    parser = argparse.ArgumentParser(
        description="构造差旅政策文档与业务演示数据，并幂等地灌进 Milvus / 业务库。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要写入的内容，不连接任何外部服务（离线可跑）。",
    )
    parser.add_argument(
        "--only",
        choices=("all", "policy", "business"),
        default="all",
        help="只播种其中一类数据（默认 all）。",
    )
    parser.add_argument(
        "--env",
        default=None,
        help="配置档（dev / test / prod）。默认取 ALIGO__APP__ENV 或 dev。",
    )
    parser.add_argument(
        "--kb-user",
        default=KB_USER_ID,
        help=(
            "政策知识库归属的用户 id（默认 alice，与 README 演示身份一致）。"
            "⚠️ 必须与前端实际登录的身份一致，否则用户列不到这个知识库。"
        ),
    )
    args = parser.parse_args()

    # ⚠️ 与 milvus_init.py 同一处坑：``get_settings()`` 没有 ``env`` 参数，
    # 要指定别的档位必须走 ``load_settings(env_name)``（它会自己读 .env）。
    #
    # ⚠️ 也一样必须给 ``host_side=True``：本脚本由 ``make seed_data`` 在宿主机直跑，
    # 而它要连的 Milvus 与 PostgreSQL 在 base.yaml 里都是容器服务名。
    # 少了它，全新克隆上这一步会以「连不上 Milvus」告终 —— 但真正的原因在地址，
    # 不在服务是否启动。见 loader 的 ``_host_reachable_endpoint``。
    settings = load_settings(args.env, host_side=True)

    try:
        return asyncio.run(
            _run(
                settings,
                only=args.only,
                dry_run=args.dry_run,
                kb_user=args.kb_user,
            )
        )
    except KeyboardInterrupt:
        _flush_stdout()
        print("\n已中断。", file=sys.stderr)
        return 130
    except SeedConnectionError as exc:
        # 连接失败已经是一条**写好的人类可读消息**，不再包一层。
        _flush_stdout()
        print(f"\n❌ 播种失败：{exc.message}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 —— 顶层入口
        # ⚠️ 顶层捕获宽异常是刻意的（同 milvus_init.py）：写入失败的原因五花八门
        # （表结构不符、向量维度不匹配、embedding 没配），让真实的线索留在
        # 消息里比包一层「播种失败」更有用。只做脱敏，不吞线索。
        _flush_stdout()
        print(f"\n❌ 播种失败：{safe_error(exc)}", file=sys.stderr)
        return 1


def _flush_stdout() -> None:
    """把 stdout 刷出去，保证错误信息**出现在进度信息之后**。

    ⚠️ 这不是洁癖：输出被重定向到管道/文件时 stdout 是块缓冲、stderr 是行缓冲，
    于是「▶ 正在写入…」会排到「❌ 播种失败…」**后面**才打印 —— 读日志的人会
    先看到失败、再看到「正在写入」，像是脚本先失败了一次又重来。在写 stderr
    之前显式刷 stdout，顺序就与真实发生的一致。
    """
    try:
        sys.stdout.flush()
    except Exception:  # noqa: BLE001 —— 刷不出去也不该盖掉真正的错误
        pass


if __name__ == "__main__":
    raise SystemExit(main())
