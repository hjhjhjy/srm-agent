"""M4 状态检查点单元测试（确定性、零依赖、CI 常绿）。"""
from __future__ import annotations

from app.agent.checkpoint import (
    InMemoryCheckpointStore,
    restore_state,
    resume_from_checkpoint,
    snapshot_state,
)
from app.agent.graph import build_app, run_agent
from app.agent.state import (
    Budget,
    PendingApproval,
    PlanStep,
    ToolCallRecord,
    initial_state,
)
from app.llm.gateway import LLMResponse, ToolCall, get_llm


def test_snapshot_roundtrip_preserves_fields():
    s = initial_state("q", tenant_id="qlk", user_id="u1", user_scopes=["kb:read"])
    snap = snapshot_state(s)
    # 快照必须是 JSON 可序列化
    import json

    json.dumps(snap)
    restored = restore_state(snap)
    assert restored["question"] == "q"
    assert restored["tenant_id"] == "qlk"
    assert isinstance(restored["budget"], Budget)
    assert restored["plan"] == []
    assert restored["pending_approval"] is None


def test_snapshot_roundtrip_with_pending_plan_and_toolcalls():
    s = initial_state("q")
    s["plan"] = [PlanStep(step_id=1, description="x", tool="kb_search", args={"query": "a"})]
    s["pending_approval"] = PendingApproval(
        step_id=1, tool="ticket_create", args={}, rationale="r", idempotency_key="k"
    )
    s["tool_calls"] = [ToolCallRecord(step_id=1, tool="kb_search", args={}, ok=True, result={"hits": []})]
    restored = restore_state(snapshot_state(s))
    assert restored["plan"][0].tool == "kb_search"
    assert restored["pending_approval"].tool == "ticket_create"
    assert restored["tool_calls"][0].ok is True


def test_checkpoint_store_save_list_load_delete():
    store = InMemoryCheckpointStore()
    cid = store.save({"question": "q"}, thread_id="t1", node="router")
    assert store.load(cid).node == "router"
    assert len(store.list("t1")) == 1
    assert store.delete(cid) is True
    assert store.list("t1") == []
    assert store.delete(cid) is False  # 已删除


async def test_run_agent_captures_per_node_checkpoints():
    llm = get_llm()
    assert hasattr(llm, "push")
    llm.push(LLMResponse(content='{"intent":"rag_qa"}'))
    llm.push(LLMResponse(tool_calls=[ToolCall(name="kb_search", args={"query": "如何注册"})]))
    llm.push(LLMResponse(content='{"sufficient":true}'))
    llm.push(LLMResponse(content="注册流程：进门户提交审核。"))

    graph = build_app()
    store = InMemoryCheckpointStore()
    result = await run_agent(
        "如何注册成为青山利康供应商？",
        session_id="rc",
        tenant_id="qlk",
        user_id="u1",
        user_scopes=["kb:read", "order:read", "ticket:write"],
        app=graph,
        checkpoint_store=store,
        thread_id="rc",
    )
    assert result["answer"]
    cps = store.list("rc")
    # 至少覆盖 router / planner / executor / reflector / responder 中的多个节点
    assert len(cps) >= 3
    nodes = {c.node for c in cps}
    assert "router" in nodes and "responder" in nodes


async def test_resume_from_checkpoint_reruns():
    llm = get_llm()
    assert hasattr(llm, "push")
    llm.push(LLMResponse(content='{"intent":"rag_qa"}'))
    llm.push(LLMResponse(tool_calls=[ToolCall(name="kb_search", args={"query": "如何注册"})]))
    llm.push(LLMResponse(content='{"sufficient":true}'))
    llm.push(LLMResponse(content="注册流程：进门户提交审核。"))

    graph = build_app()
    store = InMemoryCheckpointStore()
    await run_agent(
        "如何注册成为供应商？",
        session_id="rc2",
        tenant_id="qlk",
        user_id="u1",
        user_scopes=["kb:read", "order:read", "ticket:write"],
        app=graph,
        checkpoint_store=store,
        thread_id="rc2",
    )
    cps = store.list("rc2")
    assert cps
    restored = await resume_from_checkpoint(graph, store, "rc2", cps[0].id)
    assert restored and restored.get("answer")
