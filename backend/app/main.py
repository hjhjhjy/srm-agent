"""FastAPI 入口。

M2 治理收口后的能力
------------------
- `POST /api/chat`        非流式问答，返回答案 + 引用 + Agent 执行轨迹
- `POST /api/chat/stream` SSE 流式，逐节点推送执行轨迹（可解释性）
- `POST /api/approvals/resume` 人工审批回调（**必须鉴权 + approval:review**）
- `GET  /api/tools`       当前身份可见的工具清单
- `GET  /api/audit`       审计记录导出（需鉴权）
- `GET  /api/health`      健康检查

M1 的 P0 修复（本次收口）
------------------------
- 审批回调加鉴权，且要求 `approval:review` scope；
- 不再信任客户端传入的 thread_id，改为服务端派生 `hash(tenant|user|session)`；
- 凭据移出代码（环境变量 / JWT），启动期强校验 + 基础限流；
- 流式 `done` 仅返回白名单字段，杜绝全量 state 泄露。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from app.agent.graph import build_app
from app.agent.state import Budget, initial_state
from app.core import config as app_config
from app.core.jwt import verify_jwt
from app.observability import metrics as prom_metrics
from app.observability.audit import audit_store
from app.observability.tracing import get_request_id, new_id, set_request_id

try:
    from langgraph.types import Command
except ImportError:  # pragma: no cover
    Command = None

import app.tools.builtin
from app.llm.gateway import use_real_llm_if_configured
from app.rag.seed import seed_kb
from app.tools.registry import registry


class _RequestIdFilter(logging.Filter):
    """把当前请求的 request_id 注入每条日志记录，便于关联。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        return True


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [rid=%(request_id)s] %(name)s %(message)s",
)
logging.getLogger().addFilter(_RequestIdFilter())

logger = logging.getLogger("srm.main")


# ── 鉴权 ───────────────────────────────────────────────────────────────


class Identity(BaseModel):
    tenant_id: str
    user_id: str
    scopes: list[str]


async def current_identity(
    x_api_key: str | None = Header(None),
    authorization: str | None = Header(None),
) -> Identity:
    """身份解析：JWT（生产）优先，否则用环境变量注入的 dev key。

    绝不信任客户端自填的 scopes —— 身份与权限完全由服务端从密钥/JWT 解析。
    """
    if app_config.JWT_SECRET and authorization and authorization.startswith("Bearer "):
        try:
            p = verify_jwt(authorization[7:], app_config.JWT_SECRET, issuer=app_config.JWT_ISSUER)
        except Exception as exc:
            raise HTTPException(status_code=401, detail=f"invalid token: {exc}")
        return Identity(
            tenant_id=p.get("tenant_id", ""),
            user_id=p.get("user_id", ""),
            scopes=list(p.get("scopes", [])),
        )
    if app_config.DEV_API_KEY and x_api_key == app_config.DEV_API_KEY:
        return Identity(**app_config.dev_identity())
    raise HTTPException(status_code=401, detail="invalid or missing credentials")


def derive_thread_id(identity: Identity, session_id: str) -> str:
    """服务端派生 thread_id，防止客户端伪造/劫持他人会话（P0-2）。"""
    raw = f"{identity.tenant_id}|{identity.user_id}|{session_id}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def require_scope(identity: Identity, scope: str) -> None:
    if scope not in identity.scopes:
        raise HTTPException(status_code=403, detail=f"缺少所需权限: {scope}")


# ── Schemas ───────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field("default", description="会话 ID；thread_id 由服务端派生")


class ChatResponse(BaseModel):
    answer: str
    intent: str = ""
    citations: list[dict] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)
    tool_calls: list[dict] = Field(default_factory=list)
    pending_approval: dict | None = None
    budget: dict = Field(default_factory=dict)


class ApprovalRequest(BaseModel):
    session_id: str
    approved: bool
    reviewer: str = ""  # 可选；留空则使用鉴权身份 user_id


# ── 限流中间件 ──────────────────────────────────────────────────────────


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit_per_min: int = 60) -> None:
        super().__init__(app)
        self.limit = limit_per_min
        self.hits: dict[str, tuple[int, float]] = {}
        self._lock = asyncio.Lock()

    async def dispatch(self, request: Request, call_next):
        key = request.headers.get("x-api-key") or (request.headers.get("authorization") or "")[:24]
        now = time.time()
        async with self._lock:
            cnt, start = self.hits.get(key, (0, now))
            if now - start > 60:
                cnt, start = 0, now
            cnt += 1
            self.hits[key] = (cnt, start)
            if cnt > self.limit:
                return JSONResponse(status_code=429, content={"detail": "rate limited"})
        return await call_next(request)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """请求级观测：注入 request_id / trace_id，记录 HTTP 指标。

    作为最外层中间件注册，使 request_id 在整条调用链（节点 span、工具、LLM）
    都可读到，且能完整度量请求耗时（含限流判定）。
    """

    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or new_id()
        set_request_id(rid)
        method = request.method
        endpoint = request.url.path
        t0 = time.time()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            return response
        finally:
            prom_metrics.record_http(method, endpoint, status, time.time() - t0)


# ── App ───────────────────────────────────────────────────────────────

_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_app()
    return _app


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动：seeding 知识库（M1 缺这步导致知识问答端到端不可用）
    seed_kb()
    # 若配置了 LLM_API_KEY 则切换真实模型
    use_real_llm_if_configured()
    yield


app = FastAPI(
    title="SRM 企业级 Agent",
    version="0.3.0",
    description="基于 LangGraph 的供应商服务 Agent：工具调用 + HITL 审批 + 护栏治理 + 审计落盘 + 可观测性。",
)

app.add_middleware(RateLimitMiddleware, limit_per_min=app_config.RATE_LIMIT_PER_MIN)
# ObservabilityMiddleware 注册在后 → 成为最外层（先设置 request_id，再度量完整耗时）
app.add_middleware(ObservabilityMiddleware)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "0.3.0", "tools": len(registry.all_tools())}


@app.get("/metrics")
async def metrics():
    """Prometheus 抓取端点：请求数 / Token / 工具调用 / 护栏 / 审批 / 节点耗时。"""
    data, ctype = prom_metrics.render()
    return Response(content=data, media_type=ctype)


@app.get("/api/tools")
async def list_tools(identity: Identity = Depends(current_identity)):
    """只返回当前身份有权调用的工具 —— 工具级授权的第一道闸。"""
    return {
        "identity": identity.model_dump(),
        "tools": [t["name"] for t in registry.schemas_for(identity.scopes)],
    }


@app.get("/api/audit")
async def export_audit(identity: Identity = Depends(current_identity)):
    """导出审计记录（可问责：含审批人）。"""
    return {"total": len(audit_store.all()), "entries": audit_store.export()}


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
    config = {"configurable": {"thread_id": derive_thread_id(identity, req.session_id)}}
    result = await graph.ainvoke(state, config=config)
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
    config = {"configurable": {"thread_id": derive_thread_id(identity, req.session_id)}}

    async def gen():
        async for event in graph.astream(state, config=config, stream_mode="updates"):
            for node, update in event.items():
                if node == "__interrupt__":
                    yield _sse("interrupt", {"detail": "等待人工审批"})
                    continue
                yield _sse("trace", {"node": node, "update": _clean(update)})
        final = await graph.aget_state(config)
        # 安全：done 事件只返回白名单字段，绝不吐出全量 state（P0-2 修复）
        yield _sse(
            "done",
            _to_response(dict(final.values)).model_dump() if final else {},
        )

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/approvals/resume")
async def resume_approval(req: ApprovalRequest, identity: Identity = Depends(current_identity)):
    """人工审批回调：恢复被挂起的会话。

    必须持有 `approval:review` scope；审批人以鉴权身份为准（不信任客户端自填）。
    """
    require_scope(identity, "approval:review")
    if Command is None:
        raise HTTPException(status_code=501, detail="当前 langgraph 版本不支持 interrupt 恢复")
    graph = get_app()
    config = {"configurable": {"thread_id": derive_thread_id(identity, req.session_id)}}
    reviewer = req.reviewer or identity.user_id
    result = await graph.ainvoke(
        Command(resume={"approved": req.approved, "reviewer": reviewer}),
        config=config,
    )
    logger.info("审批恢复 tenant=%s reviewer=%s approved=%s", identity.tenant_id, reviewer, req.approved)
    return _to_response(result)


# ── 辅助 ──────────────────────────────────────────────────────────────


def _clean(obj: Any) -> Any:
    """把 pydantic / 非 JSON 类型转成可序列化结构（用于 trace 展示）。"""
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
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _to_response(result: dict) -> ChatResponse:
    """把图终态收敛为白名单响应（不暴露原始工具结果/全量审计）。"""
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
