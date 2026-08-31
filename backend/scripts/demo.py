"""离线演示：无需 API Key、不联网，跑通三条主链路并打印执行轨迹。

运行：

    cd backend && python scripts/demo.py

未配置 LLM 时，编排层全程走规则/模板降级路径 —— 这本身就是"优雅降级"能力的演示。
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent.graph import build_app  # noqa: E402
from app.agent.state import Budget, initial_state  # noqa: E402
from app.rag.seed import seed_kb  # noqa: E402

import app.tools.builtin  # noqa: E402,F401  触发工具注册
from app.llm.gateway import ScriptedLLM, set_llm  # noqa: E402
from app.tools.builtin.ticket_create import reset_idempotency_store  # noqa: E402

try:
    from langgraph.types import Command
except ImportError:  # pragma: no cover
    Command = None

SCOPES_FULL = ["kb:read", "order:read", "ticket:write", "calc:use"]
SCOPES_RO = ["kb:read", "order:read"]

SEPARATOR = "=" * 72


def _banner(title: str) -> None:
    print(f"\n{SEPARATOR}\n{title}\n{SEPARATOR}")


def _show(result: dict) -> None:
    print(f"\n意图      : {result.get('intent', '')}")
    print(f"工具调用  : {[(c.tool, 'OK' if c.ok else 'FAIL:' + str(c.error)) for c in result.get('tool_calls') or []]}")
    print(f"执行轨迹  : {[t.get('node') for t in result.get('trace') or []]}")
    cites = result.get("citations") or []
    if cites:
        print(f"引用出处  : {[(c['flow_code'], c['flow_name']) for c in cites]}")
    audit = result.get("audit") or []
    if audit:
        print(f"审计留痕  : {[(a.action, a.tool) for a in audit]}")
    budget = result.get("budget")
    if budget:
        print(f"预算消耗  : steps={budget.steps_used}/{budget.max_steps} tokens={budget.tokens_used}")
    print(f"\n{result.get('answer', '')}\n")


async def _run(graph, state, config, approve: bool = True):
    result = await graph.ainvoke(state, config=config)
    n = 0
    while result.get("__interrupt__") and n < 3 and Command is not None:
        result = await graph.ainvoke(Command(resume={"approved": approve}), config=config)
        n += 1
    return result


async def main() -> None:
    seed_kb()  # 混合检索后端（BM25 + 离线稠密 + RRF）+ blueprint 语料
    set_llm(ScriptedLLM())  # 空队列 → 全程走规则降级路径
    reset_idempotency_store()

    graph = build_app()

    # ── 场景 1：知识问答（只读） ──────────────────────────────────────
    _banner("场景 1 · 知识问答（只读链路，带引用溯源）")
    state = initial_state(
        "如何注册成为青山利康供应商？",
        session_id="demo-1",
        tenant_id="qlk",
        user_id="SUP001",
        user_scopes=SCOPES_FULL,
        budget=Budget(),
    )
    _show(await _run(graph, state, {"configurable": {"thread_id": "demo-1"}}))

    # ── 场景 2：写操作 → 审批 → 批准 ─────────────────────────────────
    _banner("场景 2 · 写操作（HITL 审批 → 批准 → 执行）")
    state = initial_state(
        "帮我建个工单，对账金额对不上",
        session_id="demo-2",
        tenant_id="qlk",
        user_id="SUP001",
        user_scopes=SCOPES_FULL,
        budget=Budget(),
    )
    result = await _run(graph, state, {"configurable": {"thread_id": "demo-2"}}, approve=True)
    _show(result)

    # ── 场景 3：越权拦截 ─────────────────────────────────────────────
    _banner("场景 3 · 越权拦截（只读用户尝试写操作）")
    from app.tools.registry import registry

    print(f"写权限用户可见工具 : {[t['name'] for t in registry.schemas_for(SCOPES_FULL)]}")
    print(f"只读用户可见工具   : {[t['name'] for t in registry.schemas_for(SCOPES_RO)]}")
    print("\n结论：只读用户在规划阶段就看不到 ticket_create，从源头杜绝越权。\n")

    _banner("演示结束")
    print("提示：配置 LLM_API_KEY 环境变量后，路由/规划/应答将切换为真实模型。")


if __name__ == "__main__":
    asyncio.run(main())
