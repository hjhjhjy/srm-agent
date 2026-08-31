"""轻量级链路追踪（span） + 可选 OpenTelemetry 桥接。

设计
----
- 每个编排节点天然是一个 span（router/planner/executor/approval/reflector/responder）。
- 通过 contextvars 传播 ``trace_id`` / ``request_id``，使一次请求内的所有 span
  与 HTTP 中间件关联，便于跨服务串联与问题回放。
- **可选 OTel 桥接**：若运行时安装了 ``opentelemetry-sdk``，span 会同步导出到
  OTel tracer（生产可接 Jaeger / Tempo / OTLP collector）。未安装时退化为
  进程内 span 记录 + 结构化日志，保证**离线零额外依赖、CI 常绿**。

生产启用 OTel 仅需额外：

    pip install opentelemetry-sdk opentelemetry-exporter-otlp
    # 启动时设置全局 TracerProvider（含 SpanProcessor/Exporter）
"""

from __future__ import annotations

import contextvars
import logging
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("srm.tracing")

trace_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("srm_trace_id", default=None)
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("srm_request_id", default=None)
_current_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("srm_span", default=None)

# 租户 / 用户上下文：用于成本归因（FinOps）与审计。由请求中间件在鉴权后注入，
# 使整条调用链（节点 span、工具、LLM）都能读到当前租户，无需层层透传参数。
tenant_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("srm_tenant", default=None)
user_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("srm_user", default=None)

_MAX_SPANS = 500
_spans: list[Span] = []

try:  # OpenTelemetry 为可选依赖：未安装时退化为进程内记录
    from opentelemetry import trace as _otel_trace  # type: ignore

    _HAS_OTEL = True
except Exception:
    _otel_trace = None
    _HAS_OTEL = False

_otel_tracer = _otel_trace.get_tracer("srm-agent") if _HAS_OTEL else None


def new_id() -> str:
    return uuid.uuid4().hex[:16]


def set_request_id(rid: str | None = None) -> str:
    """设置（或派生）request_id，并作为 trace 根。"""
    rid = rid or new_id()
    request_id_var.set(rid)
    if trace_id_var.get() is None:
        trace_id_var.set(rid)
    return rid


def get_request_id() -> str:
    return request_id_var.get() or "-"


def get_trace_id() -> str:
    return trace_id_var.get() or "-"


def ensure_trace_id() -> str:
    """返回当前 trace_id；没有则按 request_id 派生或新建。"""
    tid = trace_id_var.get()
    if not tid:
        tid = request_id_var.get() or new_id()
        trace_id_var.set(tid)
    return tid


def set_identity(tenant: str | None, user: str | None = None) -> None:
    """注入当前请求的身份（租户 / 用户），供成本归因与审计读取。"""
    tenant_var.set(tenant)
    user_var.set(user)


def get_tenant() -> str:
    """当前租户；未注入时返回 'unknown'（成本归因的兜底维度）。"""
    return tenant_var.get() or "unknown"


def get_user() -> str:
    return user_var.get() or "unknown"


def reset_identity() -> None:
    """测试 / 请求结束时清理身份上下文。"""
    tenant_var.set(None)
    user_var.set(None)


@dataclass
class Span:
    name: str
    trace_id: str = ""
    span_id: str = ""
    parent_id: str | None = None
    start_ms: float = 0.0
    end_ms: float = 0.0
    duration_ms: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "duration_ms": self.duration_ms,
            "attributes": self.attributes,
            "ok": self.ok,
            "error": self.error,
        }


def _record(s: Span) -> None:
    _spans.append(s)
    if len(_spans) > _MAX_SPANS:
        del _spans[: len(_spans) - _MAX_SPANS]
    logger.info("span name=%s trace=%s dur=%dms ok=%s", s.name, s.trace_id, s.duration_ms, s.ok)


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """上下文管理器：记录一个 span，并自动维护父子关系（基于 _current_span_id 栈）。"""
    parent_id = _current_span_id.get()
    span_id = new_id()
    token = _current_span_id.set(span_id)
    s = Span(
        name=name,
        trace_id=ensure_trace_id(),
        span_id=span_id,
        parent_id=parent_id,
        start_ms=time.time() * 1000,
        attributes=dict(attributes),
    )
    ot_cm = None
    if _HAS_OTEL and _otel_tracer is not None:
        try:
            ot_cm = _otel_tracer.start_as_current_span(name, attributes=dict(attributes))
            ot_cm.__enter__()
        except Exception:
            ot_cm = None
    try:
        yield
        s.ok = True
    except Exception as exc:
        s.ok = False
        s.error = str(exc)
        raise
    finally:
        s.end_ms = time.time() * 1000
        s.duration_ms = int(s.end_ms - s.start_ms)
        _record(s)
        _current_span_id.reset(token)
        if ot_cm is not None:
            try:
                ot_cm.__exit__(None, None, None)
            except Exception:
                pass


def spans() -> list[Span]:
    """返回进程内已记录 span 的快照（自观测 / 测试断言用）。"""
    return list(_spans)


def clear_spans() -> None:
    _spans.clear()
