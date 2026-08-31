"""编排层节点实现。

设计原则
--------
- **单一职责**：每个节点只做一件事，可独立测试。
- **失败可降级**：LLM 不可用时全部回落到规则/模板，绝不硬失败。
- **治理内置**：护栏、授权、审批、审计在节点内落地，不依赖外部约定。

HITL 审批门（本项目治理核心）
-----------------------------
写操作的审批决策按以下优先级确定（**fail-closed 默认拒绝**）：

1. 调用方持有 `approval:auto` scope → 策略自动放行（内部可信账号）
2. LangGraph `interrupt` 可用 → 挂起等待人工，恢复时读取决策与审批人
3. 以上都不满足（无 interrupt 能力且非可信账号）→ **默认拒绝**

注意：不存在「外部审批系统回调预设」这一独立分支——无 interrupt 能力时直接 fail-closed，
因为此时我们无法可靠地挂起等待人工，放行反而更危险。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from typing import Any, Optional

from langchain_core.messages import AIMessage

import app.tools.builtin  # noqa: F401  导入即注册全部内置工具
from app.agent.state import (
    AuditEntry,
    Budget,
    PendingApproval,
    PlanStep,
    ToolCallRecord,
)
from app.llm.gateway import get_llm
from app.observability.audit import audit_store
from app.tools.base import ToolContext, ToolError
from app.tools.registry import registry

try:  # langgraph 各版本 interrupt 导出位置不同
    from langgraph.types import interrupt as _interrupt
except ImportError:  # pragma: no cover
    try:
        from langgraph.prebuilt import interrupt as _interrupt  # type: ignore
    except ImportError:
        _interrupt = None


logger = logging.getLogger("srm.agent")

MAX_ITERATIONS = 3
AUTO_APPROVE_SCOPE = "approval:auto"

ROUTER_SYSTEM = (
    "你是意图分类器。根据用户问题判断意图，只输出 JSON："
    '{"intent":"chitchat|rag_qa|tool_task|human_handoff"}。'
    "chitchat=问候闲聊；rag_qa=询问流程规范/系统用法/材料要求等知识；"
    "tool_task=需要查询实时业务数据（订单、发票、对账、金额计算）；"
    "human_handoff=要求转人工/投诉。"
)

PLANNER_SYSTEM = (
    "你是任务规划器。根据问题与可用工具，输出需要调用的工具调用（可多个，无依赖时可并行）。"
    "只调用确实需要的工具，不要过度调用。知识类问题用 kb_search。"
)

REFLECTOR_SYSTEM = (
    "你是结果校验器。判断已获得的工具结果是否足以回答用户问题。"
    '只输出 JSON：{"sufficient":true|false,"reason":"..."}。'
)

RESPONDER_SYSTEM = (
    "你是青山利康 SRM 供应商助手。基于给定的参考资料回答用户问题。"
    "要求：1) 只依据参考资料，不要臆造；2) 引用时标注流程码出处；"
    "3) 参考资料不足以回答时，明确说明并建议转人工；4) 简洁、分点、可执行。"
)

HUMAN_KEYWORDS = ["人工", "转人工", "投诉", "找人", "客服", "电话联系"]
GREETING_KEYWORDS = ["你好", "您好", "hi", "hello", "在吗", "谢谢", "再见"]
BIZ_KEYWORDS = ["订单", "发票", "对账", "金额", "合计", "多少钱", "算"]


# ── 工具函数 ──────────────────────────────────────────────────────────


def _safe_json(text: str) -> dict:
    """从 LLM 输出中稳健提取 JSON（容忍 ```json 围栏与前后废话）。"""
    if not text:
        return {}
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def _idem_key(state: dict, step: PlanStep) -> str:
    """生成幂等键：租户 + 用户 + 工具 + 参数。

    纳入 tenant/user 才能避免跨租户碰撞（P0-3）：供应商 A、B 相同参数
    必须得到不同幂等键，否则会复用彼此的工单结果。
    """
    payload = json.dumps(
        {
            "t": step.tool,
            "tid": state.get("tenant_id", ""),
            "uid": state.get("user_id", ""),
            "a": step.args,
        },
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def _rule_route(q: str) -> str:
    """规则路由（v1 能力保留），作为 LLM 路由失败时的兜底。"""
    ql = q.strip().lower()
    if any(k in ql for k in HUMAN_KEYWORDS):
        return "human_handoff"
    if any(k in ql for k in GREETING_KEYWORDS) and len(ql) <= 20:
        return "chitchat"
    if any(k in ql for k in BIZ_KEYWORDS):
        return "tool_task"
    return "rag_qa"


def _fallback_plan(state: dict) -> list[PlanStep]:
    """LLM 规划失败时的启发式计划 —— 同样遵守工具级授权。

    关键词判定顺序很关键：必须先匹配"工单"这类强意图，再匹配"金额/订单"，
    否则「对账金额对不上，帮我建个工单」会被误判成订单查询。
    """
    q = state.get("question", "")
    scopes = state.get("user_scopes", [])
    if state.get("intent") == "tool_task":
        # 工单诉求优先识别；无 ticket:write 权限则落到知识检索
        if any(k in q for k in ("工单", "报修", "人工处理")) and "ticket:write" in scopes:
            return [
                PlanStep(
                    step_id=1,
                    description="创建人工工单",
                    tool="ticket_create",
                    args={"title": q[:40], "detail": q[:500], "priority": "normal"},
                )
            ]
        if "订单" in q:
            return [PlanStep(step_id=1, description="查询采购订单", tool="order_query", args={})]
        if any(k in q for k in ("合计", "金额", "算")):
            return [
                PlanStep(step_id=1, description="查询订单金额", tool="order_query", args={}),
                PlanStep(step_id=2, description="汇总计算", tool="calculator", args={"expression": "0"}),
            ]
    return [PlanStep(step_id=1, description="检索知识库", tool="kb_search", args={"query": q})]


async def _run_tool(
    name: str,
    args: dict[str, Any],
    state: dict,
    step_id: int,
    idem_key: Optional[str],
) -> ToolCallRecord:
    """执行单个工具：超时 + 指数退避重试。

    注意：**仅幂等工具才重试** —— 对非幂等写操作重试可能造成重复副作用。
    """
    spec = registry.get(name)
    if spec is None:
        return ToolCallRecord(step_id=step_id, tool=name, args=args, ok=False, error="工具未注册")

    ctx = ToolContext(
        tenant_id=state.get("tenant_id", ""),
        user_id=state.get("user_id", ""),
        scopes=state.get("user_scopes", []),
        session_id=state.get("session_id", ""),
        idempotency_key=idem_key or "",
    )
    t0 = time.time()
    last_err: Optional[str] = None
    for attempt in range(spec.max_retries + 1):
        try:
            result = await asyncio.wait_for(
                registry.invoke(name, args, ctx), timeout=spec.timeout_ms / 1000
            )
            return ToolCallRecord(
                step_id=step_id,
                tool=name,
                args=args,
                ok=True,
                result=result,
                latency_ms=int((time.time() - t0) * 1000),
                idempotency_key=idem_key,
            )
        except asyncio.TimeoutError:
            last_err = f"工具超时({spec.timeout_ms}ms)"
        except ToolError as exc:
            last_err = str(exc)
        except Exception as exc:  # noqa: BLE001
            last_err = f"工具异常: {exc}"

        # 非幂等工具不重试，避免重复副作用
        if not spec.idempotent:
            break
        if attempt < spec.max_retries:
            await asyncio.sleep(0.05 * (2**attempt))
    return ToolCallRecord(
        step_id=step_id,
        tool=name,
        args=args,
        ok=False,
        error=last_err,
        latency_ms=int((time.time() - t0) * 1000),
        idempotency_key=idem_key,
    )


def _audit(state: dict, action: str, approver: Optional[str] = None, **kw) -> AuditEntry:
    entry = AuditEntry(
        tenant_id=state.get("tenant_id", ""),
        user_id=state.get("user_id", ""),
        session_id=state.get("session_id", ""),
        action=action,
        approver=approver,
        **kw,
    )
    # 同时写入进程级 append-only 审计存储（可导出、可落盘），不再只活在 state 里
    audit_store.append(entry)
    return entry


# ── 节点 ──────────────────────────────────────────────────────────────


async def router(state: dict) -> dict:
    """意图路由：LLM 语义分类优先，规则兜底。"""
    q = state.get("question", "")
    intent: Optional[str] = None
    try:
        resp = await get_llm().achat(
            [{"role": "system", "content": ROUTER_SYSTEM}, {"role": "user", "content": q}]
        )
        intent = _safe_json(resp.content).get("intent")
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM 路由失败，回落规则路由: %s", exc)

    if intent not in ("chitchat", "rag_qa", "tool_task", "human_handoff"):
        intent = _rule_route(q)

    return {
        "intent": intent,
        "trace": state.get("trace", []) + [{"node": "router", "intent": intent}],
    }


async def planner(state: dict) -> dict:
    """规划：把问题拆解为工具调用步骤。只暴露调用方有权使用的工具。"""
    budget: Budget = state.get("budget") or Budget()
    if budget.exhausted:
        return {
            "plan": [],
            "trace": state.get("trace", []) + [{"node": "planner", "skipped": "budget_exhausted"}],
        }

    scopes = state.get("user_scopes", [])
    # 工具级授权第一道闸：LLM 只看得到有权调用的工具
    tools = registry.schemas_for(scopes)
    plan: list[PlanStep] = []
    try:
        resp = await get_llm().achat(
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": state.get("question", "")},
            ],
            tools=tools,
        )
        if resp.tool_calls:
            plan = [
                PlanStep(
                    step_id=i + 1,
                    description=f"调用 {tc.name}",
                    tool=tc.name,
                    args=tc.args,
                )
                for i, tc in enumerate(resp.tool_calls)
            ]
        else:
            parsed = _safe_json(resp.content)
            raw = parsed.get("steps") if isinstance(parsed, dict) else None
            if isinstance(raw, list):
                plan = [
                    PlanStep(
                        step_id=i + 1,
                        description=s.get("description", ""),
                        tool=s.get("tool"),
                        args=s.get("args", {}),
                    )
                    for i, s in enumerate(raw)
                    if s.get("tool")
                ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM 规划失败，回落启发式计划: %s", exc)

    if not plan:
        plan = _fallback_plan(state)

    # 二次过滤：LLM 可能幻觉出越权工具，这里再拦一道
    plan = [p for p in plan if p.tool and registry.get(p.tool) and registry.get(p.tool).allows(scopes)]

    return {
        "plan": plan,
        "trace": state.get("trace", []) + [{"node": "planner", "steps": [p.tool for p in plan]}],
    }


async def executor(state: dict) -> dict:
    """执行：只读步骤并发执行；写操作挂起等待审批。

    实现 `depends_on` 的「无依赖可并行」诉求：只读步骤彼此无副作用，
    用 `asyncio.gather` 并发执行；写操作（side_effect）仍逐一挂起审批，
    保证 HITL 不被绕过。
    """
    budget: Budget = state.get("budget") or Budget()
    scopes = state.get("user_scopes", [])
    calls: list[ToolCallRecord] = list(state.get("tool_calls") or [])
    audit: list[AuditEntry] = list(state.get("audit") or [])
    trace: list[dict] = list(state.get("trace") or [])
    pending: Optional[PendingApproval] = state.get("pending_approval")

    # 分支 1：审批已通过 → 真正执行写操作
    if pending is not None and state.get("approval_decision") is True:
        rec = await _run_tool(pending.tool, pending.args, state, pending.step_id, pending.idempotency_key)
        calls.append(rec)
        audit.append(
            _audit(
                state,
                "tool_execute_approved",
                tool=pending.tool,
                args=pending.args,
                outcome="ok" if rec.ok else "error",
                approved=True,
                idempotency_key=pending.idempotency_key,
            )
        )
        budget.consume_step()
        return {
            "tool_calls": calls,
            "audit": audit,
            "budget": budget,
            "pending_approval": None,
            "approval_decision": None,
            "trace": trace + [{"node": "executor", "executed_write": pending.tool, "ok": rec.ok}],
        }

    # 分支 2：审批被拒绝 → 放弃该写操作
    if pending is not None and state.get("approval_decision") is False:
        audit.append(
            _audit(state, "tool_rejected", tool=pending.tool, args=pending.args, approved=False)
        )
        return {
            "audit": audit,
            "pending_approval": None,
            "approval_decision": None,
            "trace": trace + [{"node": "executor", "rejected": pending.tool}],
        }

    # 分支 3：按计划执行（只读并发 + 首个写操作挂起）
    read_steps: list[PlanStep] = []
    write_steps: list[PlanStep] = []
    for step in state.get("plan") or []:
        if not step.tool:
            continue
        spec = registry.get(step.tool)
        if spec is None:
            calls.append(
                ToolCallRecord(step_id=step.step_id, tool=step.tool, ok=False, error="工具未注册")
            )
            continue
        # 工具级授权第二道闸：执行前再校验一次
        if not spec.allows(scopes):
            calls.append(
                ToolCallRecord(
                    step_id=step.step_id, tool=step.tool, ok=False, error="权限不足，已拦截"
                )
            )
            audit.append(
                _audit(state, "tool_denied", tool=step.tool, args=step.args, outcome="denied")
            )
            continue
        if spec.requires_approval:
            write_steps.append(step)
        else:
            read_steps.append(step)

    async def _do_read(step: PlanStep) -> ToolCallRecord:
        # 护栏：预算耗尽 → 强制收敛
        if budget.exhausted:
            return ToolCallRecord(
                step_id=step.step_id,
                tool=step.tool,
                ok=False,
                error=f"预算耗尽({budget.exhausted_reason})，已中止",
            )
        # 护栏：循环检测
        if budget.is_repeating(step.tool, step.args):
            return ToolCallRecord(
                step_id=step.step_id,
                tool=step.tool,
                ok=False,
                error="检测到重复调用，判定为死循环已中止",
            )
        rec = await _run_tool(step.tool, step.args, state, step.step_id, None)
        budget.consume_step()
        return rec

    if read_steps:
        read_results = await asyncio.gather(*[_do_read(s) for s in read_steps])
        calls.extend(read_results)
        trace.append({"node": "executor", "read_calls": len(read_results)})

    # 写操作：取第一个挂起，等待人工审批（多写操作场景后续步骤留待下一轮，已知限制）
    if write_steps:
        step = write_steps[0]
        pa = PendingApproval(
            step_id=step.step_id,
            tool=step.tool,
            args=step.args,
            rationale=step.description,
            idempotency_key=_idem_key(state, step),
        )
        audit.append(
            _audit(
                state,
                "approval_requested",
                tool=step.tool,
                args=step.args,
                outcome="pending",
                idempotency_key=pa.idempotency_key,
            )
        )
        return {
            "pending_approval": pa,
            "tool_calls": calls,
            "audit": audit,
            "budget": budget,
            "trace": trace + [{"node": "executor", "awaiting_approval": step.tool}],
        }

    return {
        "tool_calls": calls,
        "audit": audit,
        "budget": budget,
        "pending_approval": None,
        "trace": trace + [{"node": "executor", "calls": len(calls)}],
    }


def approval(state: dict) -> dict:
    """审批门：决定写操作是否放行（见模块顶部三级决策）。"""
    pending: Optional[PendingApproval] = state.get("pending_approval")
    if pending is None:
        return {"approval_decision": None}

    scopes = set(state.get("user_scopes") or [])
    audit: list[AuditEntry] = list(state.get("audit") or [])

    # 1) 策略自动放行（内部可信账号）
    if AUTO_APPROVE_SCOPE in scopes:
        audit.append(
            _audit(
                state,
                "auto_approved",
                approver="system:auto",
                tool=pending.tool,
                args=pending.args,
                approved=True,
            )
        )
        return {
            "approval_decision": True,
            "audit": audit,
            "trace": state.get("trace", []) + [{"node": "approval", "mode": "auto"}],
        }

    # 2) LangGraph interrupt：挂起等待人工，恢复时读取决策与审批人
    #    注意：`interrupt()` 通过抛出控制异常（GraphInterrupt）实现挂起，
    #    该异常**必须自然上浮**由 LangGraph 捕获，绝不能包在 try/except 里吞掉，
    #    否则会被误判为「无法挂起」而直接 fail-closed。
    if _interrupt is not None:
        decision = _interrupt(
            {
                "type": "tool_approval",
                "tool": pending.tool,
                "args": pending.args,
                "rationale": pending.rationale,
                "idempotency_key": pending.idempotency_key,
            }
        )
        if not isinstance(decision, dict):
            decision = {"approved": bool(decision)}
        approved = bool(decision.get("approved"))
        reviewer = str(decision.get("reviewer", "") or "")
        audit.append(
            _audit(
                state,
                "human_decision",
                approver=reviewer,
                tool=pending.tool,
                args=pending.args,
                approved=approved,
            )
        )
        return {
            "approval_decision": approved,
            "audit": audit,
            "trace": state.get("trace", [])
            + [{"node": "approval", "mode": "interrupt", "approved": approved, "reviewer": reviewer}],
        }

    # 3) 安全默认：拒绝（无 interrupt 能力且非可信账号，绝不自动放行）
    audit.append(
        _audit(
            state,
            "denied_by_default",
            approver="system:fail-closed",
            tool=pending.tool,
            args=pending.args,
            approved=False,
        )
    )
    return {"approval_decision": False, "audit": audit}


async def reflector(state: dict) -> dict:
    """反思：判断已得信息是否足以回答；不足则触发重规划（受迭代上限约束）。"""
    iteration = int(state.get("iteration") or 0) + 1
    calls = state.get("tool_calls") or []
    ok_calls = [c for c in calls if c.ok]
    budget: Budget = state.get("budget") or Budget()

    sufficient = bool(ok_calls)
    reason = "已获得有效工具结果"

    if not ok_calls:
        sufficient, reason = True, "未获得有效工具结果，直接应答并建议转人工"
    elif budget.exhausted:
        sufficient, reason = True, f"预算耗尽({budget.exhausted_reason})，强制收敛"
    elif iteration >= MAX_ITERATIONS:
        sufficient, reason = True, "达到迭代上限，强制收敛"
    else:
        try:
            resp = await get_llm().achat(
                [
                    {"role": "system", "content": REFLECTOR_SYSTEM},
                    {
                        "role": "user",
                        "content": f"问题：{state.get('question','')}\n"
                        f"已得结果：{json.dumps([c.result for c in ok_calls], ensure_ascii=False)[:2000]}",
                    },
                ]
            )
            parsed = _safe_json(resp.content)
            if "sufficient" in parsed:
                sufficient = bool(parsed["sufficient"])
                reason = str(parsed.get("reason", "")) or reason
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM 反思失败，沿用启发式判断: %s", exc)

    return {
        "iteration": iteration,
        "sufficient": sufficient,
        "reflection": reason,
        "trace": state.get("trace", [])
        + [{"node": "reflector", "sufficient": sufficient, "iteration": iteration}],
    }


def _build_citations(calls: list[ToolCallRecord]) -> list[dict]:
    """从工具结果中提取引用出处（引用溯源，可解释性的基础）。"""
    cites: list[dict] = []
    for c in calls:
        if not c.ok or not isinstance(c.result, dict):
            continue
        for h in (c.result.get("hits") or []) if c.tool == "kb_search" else []:
            cites.append(
                {
                    "chunk_id": h.get("chunk_id", ""),
                    "flow_code": h.get("flow_code", ""),
                    "flow_name": h.get("flow_name", ""),
                    "score": h.get("score", 0.0),
                    "snippet": (h.get("text") or "")[:200],
                }
            )
    return cites


def _format_context(calls: list[ToolCallRecord]) -> str:
    parts: list[str] = []
    for c in calls:
        if not c.ok:
            parts.append(f"[工具 {c.tool} 失败] {c.error}")
            continue
        body = json.dumps(c.result, ensure_ascii=False) if not isinstance(c.result, str) else c.result
        parts.append(f"[工具 {c.tool} 结果]\n{body[:1500]}")
    return "\n\n".join(parts)


def _fallback_answer(state: dict) -> str:
    """LLM 不可用时的检索增强直答（沿用 v1 的降级策略）。"""
    calls = [c for c in (state.get("tool_calls") or []) if c.ok]
    if not calls:
        return (
            "当前未检索到与您问题直接相关的内容，也未获取到相关业务数据。"
            "建议您联系青山利康对接采购专员，或在系统内提交工单转人工处理。"
        )
    lines = ["为您整理以下信息：\n"]
    for i, c in enumerate(calls, 1):
        body = json.dumps(c.result, ensure_ascii=False) if not isinstance(c.result, str) else c.result
        lines.append(f"{i}. 来源：{c.tool}\n   {body[:700]}")
    lines.append("\n（离线降级直答；配置 LLM_API_KEY 后可获得更自然的归纳回答。）")
    return "\n".join(lines)


async def responder(state: dict) -> dict:
    """应答：汇总工具结果生成最终答案，附带引用与执行轨迹。"""
    budget: Budget = state.get("budget") or Budget()
    citations = _build_citations(state.get("tool_calls") or [])
    pending: Optional[PendingApproval] = state.get("pending_approval")

    if pending is not None:
        answer = (
            f"已提交写操作「{pending.tool}」，**等待人工审批后才会执行**。\n"
            f"操作内容：{json.dumps(pending.args, ensure_ascii=False)}\n"
            f"幂等键：{pending.idempotency_key}"
        )
        return {
            "answer": answer,
            "citations": citations,
            "trace": state.get("trace", []) + [{"node": "responder", "mode": "awaiting_approval"}],
            "messages": [AIMessage(content=answer)],
        }

    context = _format_context(state.get("tool_calls") or [])
    answer = ""
    try:
        resp = await get_llm().achat(
            [
                {"role": "system", "content": RESPONDER_SYSTEM},
                {
                    "role": "user",
                    "content": f"参考资料：\n{context}\n\n用户问题：{state.get('question','')}",
                },
            ]
        )
        answer = (resp.content or "").strip()
        budget.consume_tokens(resp.tokens)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM 应答失败，降级为检索直答: %s", exc)

    if not answer:
        answer = _fallback_answer(state)

    return {
        "answer": answer,
        "citations": citations,
        "budget": budget,
        "trace": state.get("trace", []) + [{"node": "responder", "mode": "answer"}],
        "messages": [AIMessage(content=answer)],
    }
