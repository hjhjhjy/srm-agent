"""M4 合规记忆层单元测试（PII 脱敏 / 被遗忘权 / 保留期 / 导出）+ API 端到端。"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.agent.compliance import ComplianceManager, RetentionPolicy
from app.core import config as app_config
from app.main import app


def test_pii_scrubbed_on_episodic_store():
    cm = ComplianceManager()
    cm.record_turn("qlk", "u1", "s", "user", "联系我 13800138000 邮箱 a@b.com")
    turns = cm.episodic.all_for_identity("qlk", "u1")
    assert turns[0].content == "联系我 1**[MASKED]** 邮箱 [EMAIL_MASKED]"


def test_add_fact_scrubbed():
    cm = ComplianceManager()
    cm.add_fact("qlk", "u1", "contact", "手机 13912345678")
    f = cm.get_fact("qlk", "u1", "contact")
    assert f is not None and "13912345678" not in f.value


def test_save_procedure_scrubbed_args():
    cm = ComplianceManager()
    cm.save_procedure(
        "qlk",
        "u1",
        "onboard",
        [{"description": "联系 13800138000 安排培训", "tool": "ticket_create", "args": {"contact": "13800138000"}}],
    )
    steps = cm.get_procedure("qlk", "u1", "onboard")
    assert steps is not None
    assert "13800138000" not in steps[0]["description"]
    assert "13800138000" not in steps[0]["args"]["contact"]


def test_forget_identity_purges_all_layers():
    cm = ComplianceManager()
    cm.record_turn("qlk", "u1", "s", "user", "q")
    cm.add_fact("qlk", "u1", "k", "v")
    cm.save_procedure("qlk", "u1", "p", [{"tool": "x", "args": {}}])
    counts = cm.forget_identity("qlk", "u1")
    assert sum(counts.values()) >= 3
    assert cm.export_identity("qlk", "u1")["episodic"] == []


def test_export_identity_audited():
    cm = ComplianceManager()
    cm.record_turn("qlk", "u1", "s", "user", "q")
    cm.export_identity("qlk", "u1")
    assert cm.compliance_audit[-1]["action"] == "export_identity"


def test_sweep_expires_old_episodic_by_retention():
    cm = ComplianceManager(retention=RetentionPolicy(episodic_ttl_s=99))
    cm.record_turn("qlk", "u1", "s", "user", "old", now=1000.0)
    cm.record_turn("qlk", "u1", "s", "user", "new", now=1050.0)
    removed = cm.sweep(now=1100.0)
    # old: age=100 > 99 → 被清理；new: age=50 ≤ 99 → 保留
    assert removed["episodic"] == 1
    assert len(cm.episodic.all_for_identity("qlk", "u1")) == 1


# ── API 端到端：记忆 + 检查点管理 ────────────────────────────────────────


def test_memory_and_checkpoint_endpoints():
    with TestClient(app) as client:
        headers = {"X-API-Key": app_config.DEV_API_KEY}
        # 通过 /api/chat 写入情景记忆 + 检查点
        r = client.post(
            "/api/chat",
            headers=headers,
            json={"question": "如何注册成为供应商？", "session_id": "m4-e2e"},
        )
        assert r.status_code == 200, r.text

        # 导出自身身份记忆（含本轮对话）
        r2 = client.get("/api/memory/identities/qlk/SUP001/export", headers=headers)
        assert r2.status_code == 200, r2.text
        data = r2.json()["data"]
        assert data["episodic"], "至少应记录一轮对话"

        # 检查点列表（按 thread_id 派生）
        r3 = client.get("/api/checkpoints/m4-e2e", headers=headers)
        assert r3.status_code == 200, r3.text
        assert r3.json()["checkpoints"], "应至少有一个节点检查点"

        # 被遗忘权：删除后导出为空
        r4 = client.delete("/api/memory/identities/qlk/SUP001", headers=headers)
        assert r4.status_code == 200, r4.text
        assert r4.json()["removed"]["episodic"] >= 1
        r5 = client.get("/api/memory/identities/qlk/SUP001/export", headers=headers)
        assert r5.json()["data"]["episodic"] == []


def test_forget_requires_self_or_scope():
    with TestClient(app) as client:
        # 默认 dev 身份是 qlk/SUP001，且不含 compliance:manage
        headers = {"X-API-Key": app_config.DEV_API_KEY}
        r = client.delete("/api/memory/identities/other_corp/SUP999", headers=headers)
        assert r.status_code == 403, "跨身份删除需 compliance:manage 权限"


def test_checkpoint_delete_requires_own_thread():
    with TestClient(app) as client:
        headers = {"X-API-Key": app_config.DEV_API_KEY}
        client.post(
            "/api/chat",
            headers=headers,
            json={"question": "资质准入要哪些材料？", "session_id": "m4-cp"},
        )
        lst = client.get("/api/checkpoints/m4-cp", headers=headers).json()["checkpoints"]
        assert lst
        cid = lst[0]["id"]
        r = client.delete(f"/api/checkpoints/m4-cp/{cid}", headers=headers)
        assert r.status_code == 200, r.text
        lst2 = client.get("/api/checkpoints/m4-cp", headers=headers).json()["checkpoints"]
        assert all(c["id"] != cid for c in lst2)
