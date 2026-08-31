"""知识库检索工具（只读）。

对标 v1 的核心能力，但增加了租户隔离过滤。
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from app.rag.backend import get_backend
from app.tools.base import ToolContext
from app.tools.registry import registry


class KBSearchArgs(BaseModel):
    query: str = Field(
        ...,
        description="用于检索的规范表述，保留流程码、表单名、资质/对账/订单等关键实体",
    )
    top_k: int = Field(4, ge=1, le=10, description="返回片段数量")
    flow_code: str | None = Field(
        None, description="限定流程码，如 QS_SRM_QM_0001；不确定时留空"
    )


@registry.tool(
    description=(
        "在《SRM 业务蓝图》知识库中检索与问题相关的原文片段。"
        "适用于：系统使用方法、业务流程规范、资质材料要求、报表位置、接口说明类问题。"
        "返回带流程码出处的原文片段，可用于引用溯源。"
    ),
    args_model=KBSearchArgs,
    required_scopes=("kb:read",),
)
async def kb_search(args: KBSearchArgs, ctx: ToolContext) -> dict:
    hits = get_backend().search(
        args.query,
        top_k=args.top_k,
        flow_code=args.flow_code,
        tenant_id=ctx.tenant_id,  # 租户隔离在后端层强制生效
    )
    return {
        "hits": [
            {
                "chunk_id": h.chunk_id,
                "score": round(h.score, 4),
                "flow_code": h.flow_code,
                "flow_name": h.flow_name,
                "text": h.text[:800],
            }
            for h in hits
        ],
        "total": len(hits),
    }
