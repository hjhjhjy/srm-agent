"""M3 成本归因（FinOps）测试。

核心断言：
1. 模型定价表对未知模型有 default 兜底；
2. ``record_llm`` 把 token 换算成美元并按**当前租户上下文**归因（而非写死 unknown）；
3. 错误路径（无 token 消耗）不产生成本；
4. ``ScriptedLLM`` 也会产出 prompt/completion 的确定性拆分，供离线成本估算；
5. 端到端：经真实 Agent 图跑一轮，成本落到 set_identity 指定的租户维度。
"""
from __future__ import annotations

import asyncio

from app.agent.graph import run_agent
from app.llm.gateway import LLMResponse, ScriptedLLM, set_llm
from app.observability import metrics
from app.observability.tracing import reset_identity, set_identity


def _cost_for(tenant: str, model: str) -> float:
    return metrics.cost_summary().get(f"{tenant}/{model}", 0.0)


def test_pricing_fallback_for_unknown_model():
    # 未知模型走 default 定价，绝不抛错
    cost = metrics.record_llm_cost("t1", "mystery-model", 1000, 1000)
    table = metrics._pricing_table()
    expected = table["default"]["in"] + table["default"]["out"]
    assert abs(cost - expected) < 1e-9


def test_record_llm_attaches_to_current_tenant():
    set_identity("acme")
    try:
        before = _cost_for("acme", "deepseek-chat")
        metrics.record_llm(
            "deepseek-chat", 3000, 0.1, prompt_tokens=2000, completion_tokens=1000
        )
        after = _cost_for("acme", "deepseek-chat")
        assert after > before
        price = metrics._pricing_for("deepseek-chat")
        expected = (2000 / 1000) * price["in"] + (1000 / 1000) * price["out"]
        assert abs((after - before) - expected) < 1e-9
    finally:
        reset_identity()


def test_no_cost_when_no_tokens():
    set_identity("ghost")
    try:
        before = _cost_for("ghost", "deepseek-chat")
        # 错误路径：调用失败、0 token
        metrics.record_llm("deepseek-chat", 0, 0.0)
        assert _cost_for("ghost", "deepseek-chat") == before
    finally:
        reset_identity()


def test_scripted_llm_emits_token_split():
    set_identity("tenantX")
    try:
        llm = ScriptedLLM([LLMResponse(content="你好世界", tokens=10, model="scripted")])
        resp = asyncio.run(llm.achat([{"role": "user", "content": "hi there hello world"}]))
        assert resp.prompt_tokens > 0
        assert resp.completion_tokens == 10
        assert _cost_for("tenantX", "scripted") > 0
    finally:
        reset_identity()


def test_cost_summary_aggregates_by_tenant_model():
    set_identity("agg")
    try:
        metrics.record_llm_cost("agg", "gpt-4o", 1000, 1000)
        summary = metrics.cost_summary()
        assert "agg/gpt-4o" in summary
        assert summary["agg/gpt-4o"] > 0
    finally:
        reset_identity()


async def test_end_to_end_cost_attributed_to_tenant():
    """经真实 Agent 图跑一轮，成本应落到 set_identity 指定的租户维度（FinOps 端到端）。"""
    set_identity("qlk", "SUP001")
    try:
        set_llm(ScriptedLLM())  # 空队列 → 全程规则降级，但 LLM 调用仍计成本
        before = _cost_for("qlk", "scripted")
        await run_agent(
            "如何注册成为青山利康供应商？",
            session_id="cost-e2e",
            tenant_id="qlk",
            user_id="SUP001",
            user_scopes=["kb:read"],
        )
        after = _cost_for("qlk", "scripted")
        assert after > before, "Agent 跑一轮后，租户 qlk 的 LLM 成本应增加"
    finally:
        reset_identity()
