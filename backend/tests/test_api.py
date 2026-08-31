"""API 层测试：覆盖 M2 Phase 1 修复的 P0 治理击穿点。

此前这些点没有任何 API 测试，导致 P0-1（审批回调无鉴权）直到 review 才被发现。
本文件用 FastAPI TestClient 做确定性验证，不依赖真实 LLM。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from app.core import config as app_config
from app.main import Identity, app, derive_thread_id
from app.observability.audit import audit_store
from app.tools.builtin.ticket_create import _TICKETS, reset_idempotency_store
from app.tools.base import ToolContext
from app.tools.registry import registry


@pytest.fixture(autouse=True)
def _reset():
    """每个用例前清空全局审计与幂等存储，避免互相污染。"""
    audit_store.clear()
    reset_idempotency_store()
    yield


# ── 工具：签发 JWT（仅测试用，镜像 core/jwt.py 的 HS256 逻辑） ──────────────


def _sign_jwt(payload: dict, secret: str) -> str:
    def b64(d: bytes) -> str:
        return base64.urlsafe_b64encode(d).decode().rstrip("=")

    header = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    body = b64(json.dumps(payload).encode())
    signing_input = f"{header}.{body}".encode()
    sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    return f"{header}.{body}.{b64(sig)}"


# ── P0-1：审批回调必须鉴权，且需 approval:review scope ───────────────────


def test_resume_requires_authentication():
    with TestClient(app) as client:
        # 无凭据 → 401（修复前任何人可调用）
        r = client.post("/api/approvals/resume", json={"session_id": "x", "approved": True})
        assert r.status_code == 401


def test_resume_rejects_scope_without_approval_review():
    original = app_config.JWT_SECRET
    app_config.JWT_SECRET = "test-secret"
    try:
        token = _sign_jwt(
            {
                "tenant_id": "qlk",
                "user_id": "U2",
                "scopes": ["kb:read"],  # 无 approval:review
                "iss": "srm-agent",
                "exp": int(time.time()) + 3600,
            },
            "test-secret",
        )
        with TestClient(app) as client:
            r = client.post(
                "/api/approvals/resume",
                headers={"Authorization": f"Bearer {token}"},
                json={"session_id": "x", "approved": True},
            )
            assert r.status_code == 403, "缺少 approval:review 必须被拒"
    finally:
        app_config.JWT_SECRET = original


def test_resume_accepts_valid_approval_reviewer():
    original = app_config.JWT_SECRET
    app_config.JWT_SECRET = "test-secret"
    try:
        token = _sign_jwt(
            {
                "tenant_id": "qlk",
                "user_id": "ADMIN",
                "scopes": ["approval:review"],
                "iss": "srm-agent",
                "exp": int(time.time()) + 3600,
            },
            "test-secret",
        )
        with TestClient(app) as client:
            r = client.post(
                "/api/approvals/resume",
                headers={"Authorization": f"Bearer {token}"},
                json={"session_id": "x", "approved": True},
            )
            assert r.status_code != 403, "持有 approval:review 应通过鉴权"
    finally:
        app_config.JWT_SECRET = original


# ── P0-2：跨租户会话隔离 + 输出白名单 ───────────────────────────────────


def test_thread_id_derivation_is_tenant_scoped():
    a = Identity(tenant_id="qlk", user_id="SUP001", scopes=[])
    b = Identity(tenant_id="other_corp", user_id="SUP999", scopes=[])
    assert derive_thread_id(a, "same-session") != derive_thread_id(b, "same-session")


def test_chat_response_is_whitelisted_no_full_state_leak():
    with TestClient(app) as client:
        r = client.post(
            "/api/chat",
            headers={"X-API-Key": app_config.DEV_API_KEY},
            json={"question": "如何注册成为青山利康供应商？", "session_id": "demo-1"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # 不允许出现原始消息/全量工具结果等敏感字段
        assert "messages" not in body
        allowed = {"answer", "intent", "citations", "trace", "tool_calls", "pending_approval", "budget"}
        assert set(body.keys()) <= allowed
        for tc in body["tool_calls"]:
            assert "args" not in tc, "工具参数不应出现在响应中"
            assert "result" not in tc, "原始工具结果不应出现在响应中"


# ── P0-3：幂等键跨租户不碰撞 ───────────────────────────────────────────


async def test_idempotency_key_does_not_collide_across_tenants():
    args = {"title": "对账金额不一致", "detail": "需要人工核对", "priority": "high"}
    ctx_a = ToolContext(
        tenant_id="qlk", user_id="SUP001", scopes=["ticket:write"], idempotency_key="same-key"
    )
    ctx_b = ToolContext(
        tenant_id="other_corp", user_id="SUP999", scopes=["ticket:write"], idempotency_key="same-key"
    )
    ra = await registry.invoke("ticket_create", args, ctx_a)
    rb = await registry.invoke("ticket_create", args, ctx_b)
    # 跨租户相同幂等键 → 必须生成不同工单，绝不能复用他人结果
    assert ra["ticket_no"] != rb["ticket_no"]
    assert ra["replayed"] is False and rb["replayed"] is False
    assert len(_TICKETS) == 2


async def test_idempotency_replay_within_same_tenant():
    args = {"title": "重复提交测试", "detail": "验证幂等回放", "priority": "normal"}
    ctx = ToolContext(
        tenant_id="qlk", user_id="SUP001", scopes=["ticket:write"], idempotency_key="key-xyz"
    )
    first = await registry.invoke("ticket_create", args, ctx)
    second = await registry.invoke("ticket_create", args, ctx)
    assert second["replayed"] is True
    assert first["ticket_no"] == second["ticket_no"]
    assert len(_TICKETS) == 1
