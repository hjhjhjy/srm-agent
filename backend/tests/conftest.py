"""测试基线：每个用例前重置 LLM 与检索后端，保证**确定性可复现**。

这是 LLM 应用测试的第一原则 —— 绝不允许真实模型调用进入单元测试。
"""
from __future__ import annotations

import pytest

from app.agent.session import reset_memory_store
from app.llm.gateway import ScriptedLLM, set_llm
from app.rag.backend import InMemoryBM25Backend, KBHit, set_backend
from app.tools.builtin.ticket_create import reset_idempotency_store

DEMO_CHUNKS: list[KBHit] = [
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
    # 其他租户私有语料 —— 用于验证租户隔离
    KBHit(
        chunk_id="t1",
        flow_code="QS_SRM_X_0001",
        flow_name="其他租户私有政策",
        tenant_id="other_corp",
        text="这是其他租户的私有知识：内部折扣政策与结算细则，绝不应被本租户检索到。",
    ),
]


@pytest.fixture(autouse=True)
def reset_runtime():
    """每个用例都拿到干净的 LLM 与知识库，避免用例间互相污染。"""
    set_llm(ScriptedLLM())
    backend = InMemoryBM25Backend()
    backend.add(DEMO_CHUNKS)
    set_backend(backend)
    reset_idempotency_store()
    reset_memory_store()
    yield
