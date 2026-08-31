"""FastAPI 入口。

M1 提供能力
-----------
- `POST /api/chat`        非流式问答，返回答案 + 引用 + **Agent 执行轨迹**
- `POST /api/chat/stream` SSE 流式，逐节点推送执行轨迹（可解释性）
- `POST /api/approvals/resume` 人工审批回调，恢复被 interrupt 挂起的会话
- `GET  /api/tools`       当前身份可见的工具清单（工具级授权的外显）
- `GET  /api/health`      健康检查

⚠️ 鉴权说明（M1 现状）
----------------------
M1 使用**静态 API Key → 身份映射**，仅限本地开发与演示。
**M2 必须替换为 JWT**：身份与 scopes 由服务端从签名 token 解析，
绝不能信任客户端传入的 scopes —— 否则越权只是改个请求头的事。
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app.agent.graph import build_app
from app.agent.state import Budget, initial_state

try:
    from langgraph.types import Command
except ImportError:  # pragma: no cover
    Command = None

import app.tools.builtin  # noqa: F401  导入即注册工具
from app.llm.gateway import use_real_llm_if_configured
from app.tools.registry import registry

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("srm.main")

# ── M1 演示用静态凭据（M2 替换为 JWT） ─────────────────────────────────
DEV_KEYS: dict[str, dict[str, Any]] = {
    "dev-supplier-key": {
        "tenant_id": "qlk",
        "user_id": "SUP001",
        "scopes": ["kb:read", "order:read", "ticket:write", "calc:use"],
    },
    "dev-readonly-key": {
        "tenant_id": "qlk",
        "user_id": "SUP001",
        "scopes": ["kb:read", "order:read"],  # 无写权限
    },
    "dev-admin-key": {
        "tenant_id": "qlk",
        "user_id": "ADMIN",
        "scopes": ["kb:read", "order:read", "ticket:write", "calc:use", "approval:auto"],
    },
}


class Identity(BaseModel):
    tenant_id: str
    user_id: str
    scopes: list[str]


async def current_identity(x_api_key: Optional[str] = Header(None)) -> Identity:
    if not x_api_key or x_api_key not in DEV_KEYS:
        raise HTTPException(status_code=401, detail="无效或缺失的 X-API-Key")
    return Identity(**DEV_KEYS[x_api_key])


# ── Schemas ───────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field("default", description="会话 ID，同时作为 LangGraph thread_id")


class ChatResponse(BaseModel):
    answer: str
    intent: str = ""
    citations: list[dict] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
    pending_approval: Optional[dict] = None
    budget: dict = Field(default_factory=dict)


class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    reviewer: str = ""


# ── App ───────────────────────────────────────────────────────────────

_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_app()
    return _app


app = FastAPI(
    title="SRM 企业级 Agent",
    version="0.1.0",
    description="基于 LangGraph 的供应商服务 Agent：工具调用 + HITL 审批 + 护栏治理。",
)


@app.on_event("startup")
async def _startup():
    use_real_llm_if_configured()


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.1.0", "tools": len(registry.all_tools())}


@app.get("/api/tools")
async def list_tools(identity: Identity = Depends(current_identity)):
    """只返回当前身份有权调用的工具 —— 工具级授权的第一道闸。"""
    return {
        "identity": identity.model_dump(),
        "tools": [t["name"] for t in registry.schemas_for(identity.scopes)],
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, identity: Identity = Depends(current_identity)):
    graph = get_app()
    state = initial_state(
        req.question,
        session_id=req.session_id,
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
        user_scopes=identity.scopes,
        budget=Budget(),
    )
    result = await graph.ainvoke(state, config={"configurable": {"thread_id": req.session_id}})
    return _to_response(result)


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, identity: Identity = Depends(current_identity)):
    """SSE 流式：逐节点推送执行轨迹，前端可实时展示「Agent 在做什么」。"""
    from fastapi.responses import StreamingResponse

    graph = get_app()
    state = initial_state(
        req.question,
        session_id=req.session_id,
        tenant_id=identity.tenant_id,
        user_id=identity.user_id,
        user_scopes=identity.scopes,
        budget=Budget(),
    )
    config = {"configurable": {"thread_id": req.session_id}}

    async def gen():
        async for event in graph.astream(state, config=config, stream_mode="updates"):
            for node, update in event.items():
                if node == "__interrupt__":
                    yield _sse("interrupt", {"detail": "等待人工审批"})
                    continue
                yield _sse("trace", {"node": node, "update": _clean(update)})
        final = await graph.aget_state(config)
        yield _sse("done", _clean(dict(final.values)) if final else {})

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/approvals/resume")
async def resume_approval(req: ApprovalRequest):
    """人工审批回调：恢复被挂起的会话。

    这是 HITL 闭环的另一端 —— 前端/审批系统在此提交批准或拒绝。
    """
    if Command is None:
        raise HTTPException(status_code=501, detail="当前 langgraph 版本不支持 interrupt 恢复")
    graph = get_app()
    config = {"configurable": {"thread_id": req.thread_id}}
    result = await graph.ainvoke(Command(resume={"approved": req.approved}), config=config)
    logger.info("审批恢复 thread=%s approved=%s reviewer=%s", req.thread_id, req.approved, req.reviewer)
    return _to_response(result)


# ── 辅助 ──────────────────────────────────────────────────────────────


def _clean(obj: Any) -> Any:
    """把 pydantic / 非 JSON 类型转成可序列化结构。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return _clean(obj.model_dump())
    return str(obj)


def _sse(event: str, data: Any) -> str:
    import json

    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _to_response(result: dict) -> ChatResponse:
    pending = result.get("pending_approval")
    budget = result.get("budget")
    return ChatResponse(
        answer=result.get("answer", ""),
        intent=result.get("intent", ""),
        citations=_clean(result.get("citations") or []),
        trace=_clean(result.get("trace") or []),
        tool_calls=[
            {
                "tool": c.tool,
                "ok": c.ok,
                "error": c.error,
                "latency_ms": c.latency_ms,
            }
            for c in (result.get("tool_calls") or [])
        ],
        pending_approval=pending.model_dump() if pending else None,
        budget=budget.model_dump() if budget else {},
    )
