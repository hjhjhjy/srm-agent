"""编排层集成测试。

覆盖四条主链路：
1. 只读知识问答（含引用溯源）
2. 写操作 → 挂起审批 → 批准后执行（HITL）
3. 写操作 → 审批拒绝 → 不执行
4. 护栏触发 → 强制收敛但仍给出答案（优雅降级）
"""
from __future__ import annotations

import pytest

from app.agent.graph import build_app
from app.agent.nodes import executor
from app.agent.state import Budget, PlanStep, initial_state
from app.tools.registry import registry
from app.llm.gateway import LLMResponse, ToolCall, get_llm
from app.tools.builtin.ticket_create import _TICKETS

try:
    from langgraph.types import Command
except ImportError:  # pragma: no cover
    Command = None

FULL_SCOPES = ["kb:read", "order:read", "ticket:write", "calc:use"]
READONLY_SCOPES = ["kb:read", "order:read"]


def _llm():
    llm = get_llm()
    assert hasattr(llm, "push"), "测试必须使用 ScriptedLLM 以保证确定性"
    return llm


def _state(q: str, *, scopes=None, session="s1", budget=None, tenant="qlk", user="SUP001"):
    return initial_state(
        q,
        session_id=session,
        tenant_id=tenant,
        user_id=user,
        user_scopes=scopes or FULL_SCOPES,
        budget=budget or Budget(),
    )


async def _run(graph, state, config, approve: bool = True, max_resume: int = 3):
    """跑到终态；遇到 HITL interrupt 时按 `approve` 决策恢复。"""
    result = await graph.ainvoke(state, config=config)
    n = 0
    while result.get("__interrupt__") and n < max_resume:
        if Command is None:
            pytest.skip("当前 langgraph 版本不提供 Command，无法测试 interrupt 恢复")
        result = await graph.ainvoke(Command(resume={"approved": approve}), config=config)
        n += 1
    return result


# ── 链路 1：只读知识问答 ───────────────────────────────────────────────


async def test_rag_path_returns_answer_and_citations():
    llm = _llm()
    llm.push(LLMResponse(content='{"intent":"rag_qa"}'))
    llm.push(
        LLMResponse(
            tool_calls=[ToolCall(name="kb_search", args={"query": "如何注册成为供应商"})]
        )
    )
    llm.push(LLMResponse(content='{"sufficient":true}'))
    llm.push(LLMResponse(content="注册流程：进入 SRM 门户 → 填写企业信息 → 上传三证 → 提交审核。"))

    graph = build_app()
    state = _state("如何注册成为青山利康供应商？", session="s-rag")
    result = await _run(graph, state, {"configurable": {"thread_id": "s-rag"}})

    assert result["answer"]
    assert result["citations"], "知识问答必须带引用出处"
    assert any(c["flow_code"] == "QS_SRM_RG_0001" for c in result["citations"])
    assert result["tool_calls"], "应记录工具调用留痕"


async def test_chitchat_skips_tools():
    llm = _llm()
    llm.push(LLMResponse(content='{"intent":"chitchat"}'))
    llm.push(LLMResponse(content="您好，我是青山利康 SRM 供应商助手。"))

    graph = build_app()
    state = _state("你好", session="s-hi")
    result = await _run(graph, state, {"configurable": {"thread_id": "s-hi"}})

    assert result["answer"]
    assert result["tool_calls"] == [], "闲聊不应触发任何工具调用"


# ── 链路 2/3：写操作 HITL 审批 ─────────────────────────────────────────


async def test_write_operation_executes_after_approval():
    llm = _llm()
    llm.push(LLMResponse(content='{"intent":"tool_task"}'))
    llm.push(
        LLMResponse(
            tool_calls=[
                ToolCall(
                    name="ticket_create",
                    args={
                        "title": "对账金额不一致",
                        "detail": "2026年8月对账单金额与系统不一致，需要人工核对",
                        "priority": "high",
                    },
                )
            ]
        )
    )
    llm.push(LLMResponse(content='{"sufficient":true}'))
    llm.push(LLMResponse(content="已为您创建工单，采购专员会尽快联系您。"))

    graph = build_app()
    state = _state("帮我建个工单，对账金额对不上", session="s-write")
    result = await _run(graph, state, {"configurable": {"thread_id": "s-write"}}, approve=True)

    # 工单确实被创建
    assert len(_TICKETS) == 1, f"审批通过后应创建 1 个工单，实际 {len(_TICKETS)}"
    write_calls = [c for c in result["tool_calls"] if c.tool == "ticket_create"]
    assert write_calls and write_calls[0].ok

    # 审计链路完整：请求审批 → 批准 → 执行
    actions = [a.action for a in result["audit"]]
    assert "approval_requested" in actions
    assert any(a in actions for a in ("human_decision", "auto_approved", "preset_decision"))
    assert "tool_execute_approved" in actions


async def test_write_operation_rejected_does_not_execute():
    llm = _llm()
    llm.push(LLMResponse(content='{"intent":"tool_task"}'))
    llm.push(
        LLMResponse(
            tool_calls=[
                ToolCall(
                    name="ticket_create",
                    args={
                        "title": "重复建单测试",
                        "detail": "该工单在审批被拒绝的情况下不应被创建",
                    },
                )
            ]
        )
    )
    llm.push(LLMResponse(content='{"sufficient":true}'))
    llm.push(LLMResponse(content="已取消工单创建。"))

    graph = build_app()
    state = _state("帮我建个工单", session="s-reject")
    result = await _run(graph, state, {"configurable": {"thread_id": "s-reject"}}, approve=False)

    assert _TICKETS == [], "审批被拒绝后绝不能创建工单"
    rejected = [a for a in result["audit"] if a.action in ("tool_rejected", "human_decision", "denied_by_default")]
    assert rejected, "拒绝决策必须留痕"


async def test_write_tool_invisible_and_unplanned_without_scope():
    """无 ticket:write 权限时：写工具不可见，且不会被规划进计划（源头防御）。

    注意：这里不会走到执行器拦截 —— 因为防御在更前面就生效了。
    这比"调用后再报错"更好：LLM 压根不知道有这个工具存在。
    """
    llm = _llm()
    llm.push(LLMResponse(content='{"intent":"tool_task"}'))
    # 模拟 LLM 幻觉出越权工具（现实中因工具不可见很少发生，但必须防御）
    llm.push(
        LLMResponse(
            tool_calls=[
                ToolCall(
                    name="ticket_create",
                    args={"title": "越权工单", "detail": "只读用户不应能创建"},
                )
            ]
        )
    )
    llm.push(LLMResponse(content='{"sufficient":true}'))
    llm.push(LLMResponse(content="您没有创建工单的权限。"))

    # 源头防御一：可见工具清单里没有写工具
    assert "ticket_create" not in {t["name"] for t in registry.schemas_for(READONLY_SCOPES)}

    graph = build_app()
    state = _state("帮我建个工单", scopes=READONLY_SCOPES, session="s-noscope")
    result = await _run(graph, state, {"configurable": {"thread_id": "s-noscope"}})

    # 源头防御二：即使幻觉出来，规划器也会剔除，绝不进入执行
    assert _TICKETS == [], "越权用户不应创建出工单"
    assert all(c.tool != "ticket_create" for c in result["tool_calls"]), "越权工具不应被执行"


async def test_executor_denies_out_of_scope_tool_even_if_planned():
    """纵深防御：绕过规划器直接把越权工具塞进计划，执行器仍必须拦截并留痕。

    这是第二道闸 —— 防止未来有人改动规划逻辑后防线失守。
    """
    state = _state("越权测试", scopes=READONLY_SCOPES, session="s-deny")
    state["plan"] = [
        PlanStep(
            step_id=1,
            description="越权建单",
            tool="ticket_create",
            args={"title": "越权工单", "detail": "应被执行器拦截并记入审计"},
        )
    ]

    out = await executor(state)

    assert _TICKETS == [], "越权调用绝不能创建出工单"
    assert out["tool_calls"], "应留下失败记录"
    assert out["tool_calls"][0].ok is False
    denied = [a for a in out["audit"] if a.action == "tool_denied"]
    assert denied, "执行器必须拦截越权工具并留痕"


# ── 链路 4：护栏 ───────────────────────────────────────────────────────


async def test_budget_exhausted_still_answers():
    """预算耗尽时强制收敛，但仍给出答案（优雅降级，不抛异常）。"""
    llm = _llm()
    llm.push(LLMResponse(content='{"intent":"rag_qa"}'))
    llm.push(LLMResponse(content='{"sufficient":true}'))
    llm.push(LLMResponse(content="降级回答。"))

    graph = build_app()
    state = _state(
        "如何注册成为供应商？",
        session="s-budget",
        budget=Budget(max_steps=0),  # 一步都不允许
    )
    result = await _run(graph, state, {"configurable": {"thread_id": "s-budget"}})

    assert result["answer"], "护栏触发后仍应有答案"
    assert all(not c.ok for c in result["tool_calls"]), "预算耗尽后不应有成功调用"


def test_budget_detects_repeat_calls():
    b = Budget(max_repeat_calls=2)
    assert not b.is_repeating("kb_search", {"query": "对账"})
    assert not b.is_repeating("kb_search", {"query": "对账"})
    assert b.is_repeating("kb_search", {"query": "对账"}), "第三次相同调用应判定为死循环"


def test_budget_step_limit():
    b = Budget(max_steps=2)
    b.consume_step()
    b.consume_step()
    assert b.exhausted
    assert b.exhausted_reason == "step_budget"
