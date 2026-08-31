"""Prometheus 指标（M3 可观测性）。

让「请求数 / Token 消耗 / 工具调用 / 护栏触发 / 审批事件」真正有数可看——
此前这些信号只活在日志里，无法被监控面板拉取。所有指标在模块加载时注册到
默认 REGISTRY，由 ``/metrics`` 端点暴露为 Prometheus 抓取格式。

指标打点位置：
- HTTP 中间件：请求数、请求耗时（``main.ObservabilityMiddleware``）
- LLM 网关：调用数、token 消耗、错误数、耗时（``app.llm.gateway``）
- 工具执行：调用数、耗时、被权限拦截数（``app.agent.nodes``）
- 治理护栏：预算耗尽 / 死循环检测 / 工具越权（``app.agent.nodes``）
- 审批门：请求 / 自动放行 / 人工批准 / 人工拒绝 / 默认拒绝（``app.agent.nodes``）
- 编排节点：各节点耗时直方图（``app.agent.nodes``）
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

# ── HTTP ──────────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "srm_http_requests_total", "HTTP 请求数", ["method", "endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "srm_http_request_duration_seconds", "HTTP 请求耗时(秒)", ["endpoint"]
)

# ── LLM ──────────────────────────────────────────────────────────────
# phase 标签用于成本归因：区分 router / planner / reflector / responder 各阶段的
# 调用与 token 消耗，便于定位「是哪个环节在烧 token」。
LLM_CALLS = Counter("srm_llm_calls_total", "LLM 调用数", ["model", "phase"])
LLM_TOKENS = Counter("srm_llm_tokens_total", "LLM 消耗 token 数", ["model", "phase"])
LLM_ERRORS = Counter("srm_llm_errors_total", "LLM 调用错误数", ["model", "phase"])
LLM_LATENCY = Histogram("srm_llm_duration_seconds", "LLM 调用耗时(秒)", ["model", "phase"])

# ── 工具 ─────────────────────────────────────────────────────────────
TOOL_CALLS = Counter("srm_tool_calls_total", "工具调用数", ["tool", "ok"])
TOOL_LATENCY = Histogram("srm_tool_duration_seconds", "工具调用耗时(秒)", ["tool"])
TOOL_DENIED = Counter("srm_tool_denied_total", "工具因权限被拦截数", ["tool"])

# ── 治理护栏 ─────────────────────────────────────────────────────────
GUARDRAIL_TRIGGERS = Counter(
    "srm_guardrail_triggers_total", "护栏触发次数", ["reason"]
)
APPROVAL_EVENTS = Counter("srm_approval_events_total", "审批事件数", ["event"])
NODE_LATENCY = Histogram("srm_node_duration_seconds", "编排节点耗时(秒)", ["node"])

# ── 安全纵深 ─────────────────────────────────────────────────────────
# 提示注入拦截次数（按命中规则聚合）、PII 脱敏项数（按类型聚合）。
# 这两类信号是 M5 安全纵深的核心可观测项：注入拦截暴涨 = 可能遭受对抗，
# PII 脱敏激增 = 知识库/工具可能泄露了敏感字段，均需告警。
SECURITY_INJECTION_BLOCKED = Counter(
    "srm_security_injection_blocked_total", "检出的提示注入次数", ["reason"]
)
SECURITY_PII_MASKED = Counter(
    "srm_security_pii_masked_total", "被脱敏的 PII 项数", ["type"]
)


# ── 打点辅助 ──────────────────────────────────────────────────────────


def record_http(method: str, endpoint: str, status: int, duration_s: float) -> None:
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration_s)


def record_llm(
    model: str, tokens: int, duration_s: float, error: bool = False, phase: str | None = None
) -> None:
    model = model or "unknown"
    phase = phase or "unknown"
    LLM_CALLS.labels(model=model, phase=phase).inc()
    if tokens:
        LLM_TOKENS.labels(model=model, phase=phase).inc(tokens)
    if duration_s:
        LLM_LATENCY.labels(model=model, phase=phase).observe(duration_s)
    if error:
        LLM_ERRORS.labels(model=model, phase=phase).inc()


def record_tool(name: str, ok: bool, duration_s: float) -> None:
    TOOL_CALLS.labels(tool=name, ok="true" if ok else "false").inc()
    TOOL_LATENCY.labels(tool=name).observe(duration_s)


def record_tool_denied(name: str) -> None:
    TOOL_DENIED.labels(tool=name).inc()


def record_guardrail(reason: str) -> None:
    GUARDRAIL_TRIGGERS.labels(reason=reason).inc()


def record_approval(event: str) -> None:
    APPROVAL_EVENTS.labels(event=event).inc()


def record_node(name: str, duration_s: float) -> None:
    NODE_LATENCY.labels(node=name).observe(duration_s)


def record_security_injection(n: int) -> None:
    """记录一次提示注入检出（n = 命中的规则条数）。"""
    SECURITY_INJECTION_BLOCKED.labels(reason="detected").inc(n)


def record_security_pii(n: int) -> None:
    """记录被脱敏的 PII 项数。"""
    SECURITY_PII_MASKED.labels(type="all").inc(n)


def render() -> tuple[bytes, str]:
    """返回 Prometheus 抓取格式的指标文本与 content-type。"""
    return generate_latest(), CONTENT_TYPE_LATEST
