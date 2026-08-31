"""Phase 4 集成测试：多轮指代注入 + Token 计量硬化。

核心断言：
1. ``dialogue_context`` 在 router / planner / responder 被**真正注入**到 LLM 提示
   （而非仅定义了模块却未接线）；
2. 每次 LLM 调用都消耗 token 预算；
3. ``run_agent`` 跨轮次自动写回并读取会话记忆，第二轮能携带首轮上下文。
"""
from __future__ import annotations

from app.agent.graph import run_agent
from app.agent.nodes import planner, responder, router
from app.agent.state import Budget, initial_state
from app.llm.gateway import LLMResponse, ScriptedLLM, set_llm


async def test_router_injects_dialogue_context():
    llm = ScriptedLLM([LLMResponse(content='{"intent":"rag_qa"}', tokens=5)])
    set_llm(llm)
    state = initial_state(
        "它怎么申请？",
        dialogue_context="用户此前问：如何注册成为供应商",
        budget=Budget(),
    )
    result = await router(state)
    user_msg = llm.calls[0]["messages"][-1]["content"]
    assert "如何注册成为供应商" in user_msg
    assert result["budget"].tokens_used >= 5


async def test_planner_injects_dialogue_context():
    llm = ScriptedLLM([LLMResponse(content="", tool_calls=[], tokens=7)])
    set_llm(llm)
    state = initial_state(
        "它需要哪些材料？",
        dialogue_context="用户此前问：如何注册供应商",
        budget=Budget(),
    )
    result = await planner(state)
    user_msg = llm.calls[0]["messages"][-1]["content"]
    assert "如何注册供应商" in user_msg
    assert result["budget"].tokens_used >= 7


async def test_responder_injects_dialogue_context():
    llm = ScriptedLLM([LLMResponse(content="请按流程办理。", tokens=9)])
    set_llm(llm)
    state = initial_state(
        "它要多久？",
        dialogue_context="用户此前问：如何注册供应商",
        budget=Budget(),
    )
    result = await responder(state)
    user_msg = llm.calls[0]["messages"][-1]["content"]
    assert "如何注册供应商" in user_msg
    assert "请按流程办理。" in result["answer"]
    assert result["budget"].tokens_used >= 9


async def test_run_agent_carries_context_across_turns():
    set_llm(ScriptedLLM())  # 空队列 → 全程规则降级，但记忆接线仍生效
    # 第一轮：建立会话记忆
    r1 = await run_agent(
        "如何注册成为青山利康供应商？",
        session_id="mt-phase4",
        tenant_id="qlk",
        user_id="SUP001",
        user_scopes=["kb:read"],
    )
    assert r1.get("answer")

    # 第二轮：省略/指代「它」——依赖首轮记忆注入的 dialogue_context
    r2 = await run_agent(
        "它具体需要准备哪些材料？",
        session_id="mt-phase4",
        tenant_id="qlk",
        user_id="SUP001",
        user_scopes=["kb:read"],
    )
    # 第二轮注入的 dialogue_context 来自首轮记忆，必含首轮问题片段
    assert "注册" in (r2.get("dialogue_context") or "")
