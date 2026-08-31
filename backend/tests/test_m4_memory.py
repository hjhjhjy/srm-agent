"""M4 四层记忆单元测试（确定性、零依赖、CI 常绿）。"""
from __future__ import annotations

from app.agent.memory_layers import (
    MemoryManager,
    WorkingMemory,
)


def _mgr() -> MemoryManager:
    return MemoryManager()


def test_episodic_record_and_history():
    m = _mgr()
    m.record_turn("qlk", "u1", "s1", "user", "如何注册")
    m.record_turn("qlk", "u1", "s1", "assistant", "进门户")
    hist = m.episodic.history("qlk", "u1", "s1")
    assert len(hist) == 2
    assert hist[0].role == "user"
    assert hist[1].content == "进门户"


def test_episodic_tenant_isolation_and_forget():
    m = _mgr()
    m.record_turn("qlk", "u1", "s", "user", "q")
    m.record_turn("other", "u9", "s", "user", "secret")
    # 只删除 qlk/u1
    removed = m.forget_identity("qlk", "u1")
    assert removed["episodic"] == 1
    # 其他租户数据不受影响
    assert m.episodic.history("other", "u9", "s")


def test_build_dialogue_context_returns_recent():
    m = _mgr()
    m.record_turn("qlk", "u1", "s", "user", "如何注册供应商")
    m.record_turn("qlk", "u1", "s", "assistant", "进SRM门户")
    m.record_turn("qlk", "u1", "s", "user", "它需要哪些材料")
    ctx = m.build_dialogue_context("qlk", "u1", "s")
    assert "如何注册供应商" in ctx
    assert "它需要哪些材料" in ctx


def test_semantic_add_get_search():
    m = _mgr()
    m.add_fact("qlk", "u1", "account_period", "对账周期为30天", "kb")
    f = m.get_fact("qlk", "u1", "account_period")
    assert f is not None and f.value == "对账周期为30天"
    res = m.search_facts("qlk", "u1", "对账周期是多少")
    assert res and res[0].key == "account_period"


def test_procedural_save_get_list():
    m = _mgr()
    steps = [
        {"description": "查订单", "tool": "order_query", "args": {}},
        {"description": "建工单", "tool": "ticket_create", "args": {}},
    ]
    m.save_procedure("qlk", "u1", "order_dispute", steps)
    assert m.get_procedure("qlk", "u1", "order_dispute") == steps
    assert "order_dispute" in m.list_procedures("qlk", "u1")


def test_forget_identity_clears_all_layers():
    m = _mgr()
    m.record_turn("qlk", "u1", "s", "user", "q")
    m.add_fact("qlk", "u1", "k", "v")
    m.save_procedure("qlk", "u1", "p", [{"tool": "kb_search", "args": {}}])
    counts = m.forget_identity("qlk", "u1")
    assert counts["episodic"] >= 1
    assert counts["semantic"] == 1
    assert counts["procedural"] == 1
    # 导出应为空
    exp = m.export_identity("qlk", "u1")
    assert exp["episodic"] == [] and exp["semantic"] == [] and exp["procedural"] == []


def test_export_identity_shape():
    m = _mgr()
    m.record_turn("qlk", "u1", "s", "user", "hi")
    m.add_fact("qlk", "u1", "k", "v")
    m.save_procedure("qlk", "u1", "p", [{"tool": "x", "args": {}}])
    exp = m.export_identity("qlk", "u1")
    assert set(exp.keys()) == {"episodic", "semantic", "procedural"}
    assert len(exp["episodic"]) == 1
    assert exp["semantic"][0]["key"] == "k"
    assert exp["procedural"][0]["name"] == "p"


def test_sweep_respects_ttl():
    m = _mgr()
    m.record_turn("qlk", "u1", "s", "user", "old", now=1000.0)
    m.record_turn("qlk", "u1", "s", "user", "new", now=2000.0)
    removed = m.sweep(
        now=2000.0, ttl_episodic=500, ttl_semantic=10**9, ttl_procedural=10**9
    )
    assert removed["episodic"] == 1  # 仅 age=1000s > 500s 的 "old" 被清理
    hist = m.episodic.history("qlk", "u1", "s")
    assert len(hist) == 1 and hist[0].content == "new"


def test_working_memory_roundtrip():
    state = {"reflection": "done", "iteration": 2, "plan": [], "tool_calls": [], "pending_approval": None}
    wm = WorkingMemory.from_state(state)
    assert wm.iteration == 2 and wm.reflection == "done"
    out = wm.apply_to({"iteration": 0, "reflection": ""})
    assert out["iteration"] == 2
