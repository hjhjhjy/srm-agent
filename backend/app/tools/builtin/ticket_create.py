"""工单创建工具（**写操作**）。

这是 M1 唯一的写工具，用来完整演示三条治理链路：

1. **HITL 审批门**：`side_effect=True` → 编排层 interrupt，等待人工确认后才真正执行。
2. **幂等性**：相同 `idempotency_key` 直接回放首次结果，杜绝网络重试/用户重复点击导致的重复建单。
3. **审计留痕**：由编排层在执行前后写入 `AgentState.audit`（追加、不可变）。

生产对接点
----------
- `_IDEMPOTENCY_STORE` → Redis（`SET key value NX EX 86400`），保证分布式下幂等。
- `_TICKETS` → SRM 工单系统 API / 数据库表。
"""
from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

from app.tools.base import ToolContext, ToolError
from app.tools.registry import registry


class TicketCreateArgs(BaseModel):
    title: str = Field(..., min_length=4, max_length=100, description="工单标题")
    detail: str = Field(..., min_length=4, max_length=2000, description="问题详细描述")
    priority: Literal["low", "normal", "high"] = Field(
        "normal", description="优先级：low/normal/high"
    )


# ── 对接点：生产替换为 Redis（幂等）与工单系统（持久化） ──────────────────
_IDEMPOTENCY_STORE: dict[str, dict] = {}
_TICKETS: list[dict] = []


def reset_idempotency_store() -> None:
    """仅用于测试隔离。"""
    _IDEMPOTENCY_STORE.clear()
    _TICKETS.clear()


@registry.tool(
    description=(
        "创建人工工单，转交采购专员处理（写操作，需人工审批后执行）。"
        "仅在用户问题无法自助解决、明确要求人工介入时使用。"
        "不要用于知识问答类问题。"
    ),
    args_model=TicketCreateArgs,
    required_scopes=("ticket:write",),
    side_effect=True,  # ← 触发 HITL 审批门
    idempotent=True,
)
async def ticket_create(args: TicketCreateArgs, ctx: ToolContext) -> dict:
    # 写操作必须携带幂等键，否则拒绝执行 —— 这是硬约束，不是建议
    if not ctx.idempotency_key:
        raise ToolError("写操作必须携带 idempotency_key，否则无法保证幂等")

    # 幂等：同 key + 同租户/用户 才回放首次结果，杜绝跨租户碰撞（P0-3）
    if ctx.idempotency_key in _IDEMPOTENCY_STORE:
        cached = _IDEMPOTENCY_STORE[ctx.idempotency_key]
        if cached.get("tenant_id") == ctx.tenant_id and cached.get("supplier_id") == ctx.user_id:
            return {**cached, "replayed": True}
        # 幂等键冲突但租户不一致：视为不同请求，用派生键避免覆盖他人记录
        eff_key = f"{ctx.idempotency_key}:{ctx.tenant_id}:{ctx.user_id}"
    else:
        eff_key = ctx.idempotency_key

    ticket = {
        "ticket_no": "TK" + uuid.uuid4().hex[:10].upper(),
        "tenant_id": ctx.tenant_id,
        "supplier_id": ctx.user_id,
        "title": args.title,
        "detail": args.detail,
        "priority": args.priority,
        "status": "待分配",
        "created_at": int(time.time()),
    }
    _IDEMPOTENCY_STORE[eff_key] = ticket
    _TICKETS.append(ticket)
    return {**ticket, "replayed": False}
