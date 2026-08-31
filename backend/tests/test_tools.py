"""工具层单元测试：治理能力的正确性验证。"""
from __future__ import annotations

import pytest

from app.rag.backend import get_backend
from app.tools.base import ToolContext, ToolNotPermitted
from app.tools.builtin.calculator import safe_eval
from app.tools.builtin.ticket_create import reset_idempotency_store
from app.tools.registry import registry

FULL_SCOPES = ["kb:read", "order:read", "ticket:write", "calc:use"]
READONLY_SCOPES = ["kb:read", "order:read"]


def ctx(scopes=None, tenant="qlk", user="SUP001", idem=""):
    return ToolContext(
        tenant_id=tenant, user_id=user, scopes=scopes or FULL_SCOPES, idempotency_key=idem
    )


# ── 工具级授权 ────────────────────────────────────────────────────────


def test_registry_filters_tools_by_scope():
    """只读用户看不到也不该看到写工具。"""
    names_full = {t["name"] for t in registry.schemas_for(FULL_SCOPES)}
    names_ro = {t["name"] for t in registry.schemas_for(READONLY_SCOPES)}

    assert "ticket_create" in names_full
    assert "ticket_create" not in names_ro
    assert "kb_search" in names_ro


async def test_invoke_denied_without_scope():
    with pytest.raises(ToolNotPermitted):
        await registry.invoke(
            "ticket_create",
            {"title": "测试工单", "detail": "无权限调用应被拦截"},
            ctx(scopes=READONLY_SCOPES),
        )


# ── 租户隔离（补 v1 P0 缺口）────────────────────────────────────────────


def test_tenant_isolation_in_retrieval():
    """本租户检索不得命中其他租户私有语料。"""
    hits = get_backend().search("内部折扣政策", tenant_id="qlk", top_k=10)
    tenants = {h.tenant_id for h in hits}
    assert "other_corp" not in tenants
    assert hits == [] or all(h.tenant_id in ("public", "qlk") for h in hits)


def test_own_tenant_private_chunk_is_visible():
    """反过来：本租户自己的私有语料必须能检索到。"""
    hits = get_backend().search("内部折扣政策", tenant_id="other_corp", top_k=10)
    assert any(h.chunk_id == "t1" for h in hits)


# ── 行级隔离 ──────────────────────────────────────────────────────────


async def test_order_query_only_returns_own_rows():
    res = await registry.invoke("order_query", {}, ctx())
    assert res["total"] > 0
    # SUP001 名下只有两笔，SUP002 的 PO20260003 不得出现
    assert all(o["order_no"] != "PO20260003" for o in res["orders"])


# ── 幂等性（写操作核心保障）─────────────────────────────────────────────


async def test_ticket_create_is_idempotent():
    args = {"title": "对账金额不一致", "detail": "8月对账单与系统金额不符，需人工核对"}
    first = await registry.invoke("ticket_create", args, ctx(idem="key-abc"))
    second = await registry.invoke("ticket_create", args, ctx(idem="key-abc"))

    assert first["replayed"] is False
    assert second["replayed"] is True
    assert first["ticket_no"] == second["ticket_no"]  # 未重复建单


async def test_ticket_create_requires_idempotency_key():
    """缺幂等键必须拒绝执行 —— 这是硬约束。"""
    from app.tools.base import ToolError

    with pytest.raises(ToolError, match="idempotency_key"):
        await registry.invoke(
            "ticket_create",
            {"title": "无幂等键工单", "detail": "应被拒绝执行"},
            ctx(idem=""),
        )


# ── 计算工具安全（防 Prompt 注入导致的 RCE）────────────────────────────


def test_calculator_rejects_code_execution():
    for expr in [
        "__import__('os').system('echo pwned')",
        "open('/etc/passwd').read()",
        "(1).__class__.__bases__",
        "eval('1+1')",
    ]:
        with pytest.raises((ValueError, SyntaxError)):
            safe_eval(expr)


def test_calculator_rejects_huge_exponent():
    with pytest.raises(ValueError):
        safe_eval("9**9**9")


async def test_calculator_ok():
    res = await registry.invoke("calculator", {"expression": "(128600 + 45200) * 2"}, ctx())
    assert res["ok"] is True
    assert res["value"] == 347600


async def test_calculator_returns_error_not_exception():
    """非法表达式应返回结构化错误，而不是把异常抛进图状态。"""
    res = await registry.invoke("calculator", {"expression": "1/0"}, ctx())
    assert res["ok"] is False
    assert "error" in res
