"""LangGraph 状态图编排。

图结构
------
```
START → router ─┬─────────────────────────────→ responder → END
                └→ planner → executor ─┬→ approval ─┬→ executor（批准后执行）
                                       │            └→ responder（拒绝/未决）
                                       └→ reflector ─┬→ planner（信息不足，重规划）
                                                     └→ responder（充分或护栏触发）
```

为什么用状态图而不是自由 ReAct 循环
----------------------------------
- **可控**：每一步的跳转条件是显式函数，不是 LLM 自由发挥
- **可测**：可单独 invoke 任一节点，无需跑全图
- **可观测**：每个节点天然是一个 span
- **可治理**：护栏、审批门作为节点/条件边存在，不会被绕过

⚠️ HITL 前提：`interrupt` 需要 checkpointer。因此 `build_app()` 默认挂
`MemorySaver`；生产替换为 `AsyncPostgresSaver`。
"""
from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    approval,
    executor,
    planner,
    reflector,
    responder,
    router,
)
from app.agent.state import AgentState

logger = logging.getLogger("srm.agent.graph")


# ── 条件边 ────────────────────────────────────────────────────────────


def route_after_router(state: dict) -> str:
    """需要检索/查数据的问题走规划；闲聊与转人工直接应答。"""
    return "planner" if state.get("intent") in ("rag_qa", "tool_task") else "responder"


def route_after_executor(state: dict) -> str:
    """存在待审批的写操作 → 进入审批门；否则进入反思。"""
    return "approval" if state.get("pending_approval") is not None else "reflector"


def route_after_approval(state: dict) -> str:
    """批准 → 回执行器真正执行；拒绝或未决 → 收敛应答。"""
    return "executor" if state.get("approval_decision") is True else "responder"


def route_after_reflector(state: dict) -> str:
    """信息不足且未触护栏 → 重规划；否则收敛应答。"""
    return "responder" if state.get("sufficient") else "planner"


# ── 构图 ──────────────────────────────────────────────────────────────


def build_graph(checkpointer=None):
    """构建并编译状态图。

    Args:
        checkpointer: LangGraph checkpointer。HITL（interrupt）必需；
            传入 None 时写操作会走"拒绝/未决"分支而不会挂起。
    """
    g = StateGraph(AgentState)

    g.add_node("router", router)
    g.add_node("planner", planner)
    g.add_node("executor", executor)
    g.add_node("approval", approval)
    g.add_node("reflector", reflector)
    g.add_node("responder", responder)

    g.set_entry_point("router")

    g.add_conditional_edges(
        "router",
        route_after_router,
        {"planner": "planner", "responder": "responder"},
    )
    g.add_edge("planner", "executor")
    g.add_conditional_edges(
        "executor",
        route_after_executor,
        {"approval": "approval", "reflector": "reflector"},
    )
    g.add_conditional_edges(
        "approval",
        route_after_approval,
        {"executor": "executor", "responder": "responder"},
    )
    g.add_conditional_edges(
        "reflector",
        route_after_reflector,
        {"planner": "planner", "responder": "responder"},
    )
    g.add_edge("responder", END)

    return g.compile(checkpointer=checkpointer)


def build_app(checkpointer=None):
    """默认应用：无 checkpointer 时自动挂 MemorySaver，保证 HITL 可用。

    生产替换为 `langgraph.checkpoint.postgres.AsyncPostgresSaver`，
    以获得跨进程、可持久的检查点能力。

    状态中包含 pydantic 模型（PlanStep / PendingApproval / Budget /
    AuditEntry / ToolCallRecord）。LangGraph 1.x 在 checkpointer 序列化时
    会对这些未注册类型打印弃用告警，但当前版本可正确往返；升级 LangGraph
    大版本时需注意为这些类型注册 msgpack 白名单（见 ADR-0002）。
    """
    if checkpointer is None:
        from langgraph.checkpoint.memory import MemorySaver

        checkpointer = MemorySaver()
    return build_graph(checkpointer=checkpointer)


async def run_agent(
    question: str,
    *,
    session_id: str = "",
    tenant_id: str = "",
    user_id: str = "",
    user_scopes: list[str] | None = None,
    app=None,
) -> dict:
    """便捷入口：跑完一次对话并返回终态。"""
    from app.agent.state import Budget, initial_state

    graph = app or build_app()
    state = initial_state(
        question,
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        user_scopes=user_scopes,
        budget=Budget(),
    )
    config = {"configurable": {"thread_id": session_id or "default"}}
    return await graph.ainvoke(state, config=config)
