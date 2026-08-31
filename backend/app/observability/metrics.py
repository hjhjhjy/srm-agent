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

import json
import os
from functools import lru_cache

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.observability.tracing import get_tenant

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

# ── 合规（M4）────────────────────────────────────────────────────────────
# 合规记忆层的健康度信号：PII 是否在落盘前被脱敏、被遗忘权触发多少次、DSAR 导出次数、
# 保留期清理了多少条。这些是「记忆是否合规」的可观测证据，缺了它们合规只是口头声明。
COMPLIANCE_PII_SCRUBBED = Counter(
    "srm_compliance_pii_scrubbed_total", "合规层落地存储前被脱敏的 PII 项数"
)
COMPLIANCE_FORGET_TOTAL = Counter(
    "srm_compliance_forget_total", "被遗忘权触发后删除的记忆记录数", ["layer"]
)
COMPLIANCE_EXPORT_TOTAL = Counter(
    "srm_compliance_export_total", "数据导出（DSAR / 数据可携）请求次数"
)
COMPLIANCE_RETENTION_EXPIRED = Counter(
    "srm_compliance_retention_expired_total", "因超过保留期被自动清理的记忆记录数", ["layer"]
)

# ── 成本归因（M3 FinOps）─────────────────────────────────────────────────
# 企业最先追问的就是「一个月烧多少钱」。本组指标把 LLM token 消耗换算成美元，并
# 按 (租户, 模型) 归因，便于定位「是哪个租户 / 哪个模型在烧钱」，支撑配额与预算熔断。
# 注意：这是成本**归因展示**，不是计费系统；定价为常见公开价，可经环境变量覆盖。
LLM_COST = Counter(
    "srm_llm_cost_total_usd", "LLM 累计成本(美元)，按租户/模型归因", ["tenant", "model"]
)

# 默认模型定价：美元 / 1K token（in=输入, out=输出）。覆盖常见国产/海外模型；
# 命中不到的模型走 "default"。可用 SRM_MODEL_PRICING 以 JSON 覆盖，例如：
#   {"deepseek-chat": {"in": 0.00027, "out": 0.0011}}
_DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "deepseek-chat": {"in": 0.00027, "out": 0.0011},
    "deepseek-reasoner": {"in": 0.00055, "out": 0.00219},
    "gpt-4o-mini": {"in": 0.00015, "out": 0.0006},
    "gpt-4o": {"in": 0.005, "out": 0.015},
    "qwen-plus": {"in": 0.0004, "out": 0.0004},
    "qwen-max": {"in": 0.0016, "out": 0.004},
    "default": {"in": 0.001, "out": 0.002},
}


@lru_cache(maxsize=1)
def _pricing_table() -> dict[str, dict[str, float]]:
    """返回定价表；优先读取环境变量 SRM_MODEL_PRICING 覆盖默认值。"""
    table = dict(_DEFAULT_PRICING)
    raw = os.getenv("SRM_MODEL_PRICING")
    if raw:
        try:
            override = json.loads(raw)
            if isinstance(override, dict):
                for k, v in override.items():
                    if isinstance(v, dict) and "in" in v and "out" in v:
                        table[k] = {"in": float(v["in"]), "out": float(v["out"])}
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return table


def _pricing_for(model: str) -> dict[str, float]:
    table = _pricing_table()
    return table.get(model) or table["default"]


def record_llm_cost(
    tenant: str, model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """按 (租户, 模型) 累加 LLM 成本（美元）。返回本次成本，供调用方回显。"""
    price = _pricing_for(model)
    cost = (prompt_tokens / 1000.0) * price["in"] + (completion_tokens / 1000.0) * price["out"]
    if cost:
        LLM_COST.labels(tenant=tenant or "unknown", model=model or "unknown").inc(cost)
    return cost


def cost_summary() -> dict[str, float]:
    """返回各 (tenant,model) 维度的累计成本（测试 / 调试 / 管理面板用）。

    注意：prometheus_client 的 Counter 在 ``collect()`` 时除了真实计数值，
    还会附带一个 ``<name>_created`` 的时间戳样本，需排除，否则会混入创建时间戳。
    """
    out: dict[str, float] = {}
    for metric in LLM_COST.collect():
        for sample in metric.samples:
            if sample.name.endswith("_created"):
                continue
            key = f"{sample.labels.get('tenant', 'unknown')}/{sample.labels.get('model', 'unknown')}"
            out[key] = out.get(key, 0.0) + sample.value
    return out


# ── 打点辅助 ──────────────────────────────────────────────────────────


def record_http(method: str, endpoint: str, status: int, duration_s: float) -> None:
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status).inc()
    REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration_s)


def record_llm(
    model: str,
    tokens: int,
    duration_s: float,
    error: bool = False,
    phase: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
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
    # 成本归因：按当前租户上下文（中间件注入）把 token 换算为美元。
    # 错误调用通常无 token 消耗，这里仍记录以便排查，但金额为 0。
    if prompt_tokens or completion_tokens:
        record_llm_cost(get_tenant(), model, prompt_tokens, completion_tokens)


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


# ── 合规打点 ──────────────────────────────────────────────────────────────


def record_compliance_pii(n: int) -> None:
    """记录合规记忆层落地前被脱敏的 PII 项数。"""
    if n:
        COMPLIANCE_PII_SCRUBBED.inc(n)


def record_compliance_forget(layer: str, count: int) -> None:
    """记录某层因被遗忘权删除的记录数。"""
    if count:
        COMPLIANCE_FORGET_TOTAL.labels(layer=layer).inc(count)


def record_compliance_export() -> None:
    """记录一次数据导出（DSAR）请求。"""
    COMPLIANCE_EXPORT_TOTAL.inc()


def record_compliance_retention(layer: str, count: int) -> None:
    """记录某层因保留期过期被清理的记录数。"""
    if count:
        COMPLIANCE_RETENTION_EXPIRED.labels(layer=layer).inc(count)


def render() -> tuple[bytes, str]:
    """返回 Prometheus 抓取格式的指标文本与 content-type。"""
    return generate_latest(), CONTENT_TYPE_LATEST
