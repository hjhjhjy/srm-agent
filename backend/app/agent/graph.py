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

from app.agent.checkpoint import CheckpointStore
from app.agent.memory_layers import MemoryManager
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
    memory_manager: MemoryManager | None = None,
    checkpoint_store: CheckpointStore | None = None,
    thread_id: str | None = None,
) -> dict:
    """便捷入口：跑完一次对话并返回终态。

    多轮记忆接线（Phase 4 + M4）：
    1. 调用前从会话记忆取该 session 的历史上下文，注入 ``dialogue_context``，
       让单轮模型也能理解「它/这个/怎么申请」等跨轮指代。
    2. 若传入 ``memory_manager``（M4 合规记忆），本轮问答会写入四层记忆的
       情景层（落地前已完成 PII 脱敏）；若传入 ``checkpoint_store``（M4 检查点），
       运行中每个节点边界都会被显式快照，支持审计 / 回放 / 断点续跑。

    Args:
        thread_id: 服务端派生的 thread_id（与 ``main.derive_thread_id`` 同源）。
            留空则由 ``session_id`` 推导，保持与 demo 脚本的向后兼容。
    """
    from app.agent.session import get_memory_store
    from app.agent.state import Budget, initial_state

    store = get_memory_store()
    ctx = store.context(session_id)
    graph = app or build_app()
    state = initial_state(
        question,
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        user_scopes=user_scopes,
        budget=Budget(),
        dialogue_context=ctx,
    )
    tid = thread_id or (session_id or "default")
    config = {"configurable": {"thread_id": tid}}

    if checkpoint_store is not None:
        result = await _run_with_checkpoints(graph, state, config, checkpoint_store, tid)
    else:
        result = await graph.ainvoke(state, config=config)

    answer = result.get("answer") or "" if isinstance(result, dict) else ""
    # 写回 Phase 4 会话记忆（驱动多轮指代）
    store.append(session_id, "user", question)
    store.append(session_id, "assistant", answer)
    # 写入 M4 四层合规记忆（含 PII 脱敏）
    if memory_manager is not None:
        memory_manager.record_turn(tenant_id, user_id, session_id, "user", question)
        if answer:
            memory_manager.record_turn(tenant_id, user_id, session_id, "assistant", answer)
    return result


async def _run_with_checkpoints(
    graph,
    state: AgentState,
    config: dict,
    store: CheckpointStore,
    thread_id: str,
) -> dict:
    """带逐节点检查点的运行：每完成一个节点即快照当前全量状态。

    使用 ``stream_mode="updates"`` 逐个节点推进，并在每个节点后通过
    ``graph.aget_state`` 抓取完整状态做快照。遇到 HITL 中断（``__interrupt__``）
    时自然停止，已捕获中断前的全部节点检查点。
    """
    async for item in graph.astream(state, config=config, stream_mode="updates"):
        for node in item:
            if node == "__interrupt__":
                continue
            try:
                snap = await graph.aget_state(config)
            except Exception:
                snap = None
            if snap is not None and getattr(snap, "values", None):
                store.save(snap.values, thread_id=thread_id, node=node)
    last = await graph.aget_state(config)
    if last is None or not getattr(last, "values", None):
        return {}
    return dict(last.values)
