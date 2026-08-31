"""默认知识库语料与启动 seeding。

M1 的 `InMemoryBM25Backend` 默认空语料，导致应用启动后知识问答端到端不可用
（README 第一条 curl 拿不到答案）。本模块在应用启动时注入一份示例语料，
使 `/api/chat` 的知识问答真正可用；生产可替换为 v1 的稠密混合检索 + ingestion。
"""
from __future__ import annotations

from app.rag.backend import InMemoryBM25Backend, KBHit, set_backend


DEFAULT_KB: list[KBHit] = [
    KBHit(
        chunk_id="c1",
        flow_code="QS_SRM_RG_0001",
        flow_name="供应商注册",
        tenant_id="public",
        text=(
            "供应商注册流程：进入 SRM 门户点击供应商注册，填写企业基本信息，"
            "上传营业执照、税务登记证、组织机构代码证，提交后等待采购专员审核，"
            "审核周期一般为 3 个工作日。"
        ),
    ),
    KBHit(
        chunk_id="c2",
        flow_code="QS_SRM_QM_0001",
        flow_name="资质准入",
        tenant_id="public",
        text=(
            "资质准入需提交：阳光协议、保密协议、质量协议、营业执照、"
            "法定代表人身份证、开户许可证。资质到期前 60 天系统会发送预警通知。"
        ),
    ),
    KBHit(
        chunk_id="c3",
        flow_code="QS_SRM_RM_0003",
        flow_name="对账管理",
        tenant_id="public",
        text=(
            "对账流程：供应商在每月 1-5 日发起对账并上传发票，"
            "采购专员核对无误后提交财务，付款周期为发票入账后 30 天。"
        ),
    ),
    KBHit(
        chunk_id="c4",
        flow_code="QS_SRM_QM_0002",
        flow_name="整改管理",
        tenant_id="public",
        text=(
            "收到整改通知后，供应商需在 7 个工作日内提交整改报告，"
            "说明原因、纠正措施与预防措施，逾期未处理将影响绩效评分。"
        ),
    ),
    KBHit(
        chunk_id="c5",
        flow_code="QS_SRM_PM_0001",
        flow_name="绩效管理",
        tenant_id="public",
        text=(
            "供应商绩效报表可在 SRM 系统【报表中心】-【供应商绩效】查看，"
            "包含交付及时率、合格率、服务响应评分等维度。"
        ),
    ),
]


def seed_kb() -> None:
    """用默认语料填充检索后端（覆盖空语料）。"""
    backend = InMemoryBM25Backend()
    backend.add(DEFAULT_KB)
    set_backend(backend)
