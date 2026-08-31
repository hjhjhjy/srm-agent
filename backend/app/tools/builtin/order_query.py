"""采购订单查询工具（只读）。

M1 用内存数据集，标注了清晰的**对接点**：把 `_DEMO_ORDERS` 换成真实 SRM 库查询即可。
行级隔离（只返回本租户本供应商的数据）在工具内强制实现。
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

from app.tools.base import ToolContext
from app.tools.registry import registry


class OrderQueryArgs(BaseModel):
    order_no: Optional[str] = Field(
        None, description="采购订单号，如 PO20260001；传此参数时忽略 status"
    )
    status: Optional[Literal["待确认", "已确认", "已发货", "已完成"]] = Field(
        None, description="按订单状态过滤"
    )
    limit: int = Field(5, ge=1, le=20, description="最多返回条数")


# ── 对接点 ────────────────────────────────────────────────────────────
# 生产环境替换为：SELECT ... FROM srm_purchase_order
#                 WHERE tenant_id = :tenant AND supplier_id = :user
# 行级过滤必须下推到 SQL，不能在全量结果上做内存过滤。
_DEMO_ORDERS: list[dict] = [
    {
        "order_no": "PO20260001",
        "tenant_id": "qlk",
        "supplier_id": "SUP001",
        "status": "待确认",
        "amount": 128600.00,
        "eta": "2026-09-15",
        "items": "一次性输液器 5000 支",
    },
    {
        "order_no": "PO20260002",
        "tenant_id": "qlk",
        "supplier_id": "SUP001",
        "status": "已发货",
        "amount": 45200.00,
        "eta": "2026-09-02",
        "items": "医用敷料 1200 包",
    },
    {
        "order_no": "PO20260003",
        "tenant_id": "qlk",
        "supplier_id": "SUP002",
        "status": "已完成",
        "amount": 8800.00,
        "eta": "2026-08-20",
        "items": "检验试剂 30 盒",
    },
]


@registry.tool(
    description=(
        "查询当前供应商名下的采购订单（只读）。"
        "可传订单号精确查询单笔，或按状态过滤列表。"
        "返回订单号、状态、金额、预计到货日期与物料明细。"
    ),
    args_model=OrderQueryArgs,
    required_scopes=("order:read",),
)
async def order_query(args: OrderQueryArgs, ctx: ToolContext) -> dict:
    rows = [
        o
        for o in _DEMO_ORDERS
        # 行级隔离：只暴露本租户 + 本供应商的数据
        if o["tenant_id"] == ctx.tenant_id and o["supplier_id"] == ctx.user_id
    ]
    if args.order_no:
        rows = [o for o in rows if o["order_no"] == args.order_no]
        if not rows:
            return {"orders": [], "total": 0, "note": "未找到该订单号或无访问权限"}
    elif args.status:
        rows = [o for o in rows if o["status"] == args.status]

    rows = rows[: args.limit]
    return {
        "orders": [
            {
                "order_no": o["order_no"],
                "status": o["status"],
                "amount": o["amount"],
                "eta": o["eta"],
                "items": o["items"],
            }
            for o in rows
        ],
        "total": len(rows),
    }
