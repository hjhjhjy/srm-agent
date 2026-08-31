"""M3 可观测性测试：/metrics 端点 + 节点 span 记录 + request_id 关联。

全部离线运行：ScriptedLLM + 内存知识库，不需要 API Key。
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.core import config as app_config
from app.main import app
from app.observability import tracing


def test_metrics_endpoint_exposes_core_metrics():
    tracing.clear_spans()
    with TestClient(app) as client:
        r = client.post(
            "/api/chat",
            headers={"X-API-Key": app_config.DEV_API_KEY},
            json={"question": "如何注册成为青山利康供应商？", "session_id": "obs-1"},
        )
        assert r.status_code == 200, r.text
        m = client.get("/metrics")
        assert m.status_code == 200
        body = m.text
        # 核心指标族必须出现在 Prometheus 输出中
        assert "srm_http_requests_total" in body
        assert "srm_llm_calls_total" in body
        assert "srm_node_duration_seconds" in body
        assert "srm_tool_calls_total" in body


def test_node_spans_recorded_for_a_run():
    tracing.clear_spans()
    with TestClient(app) as client:
        client.post(
            "/api/chat",
            headers={"X-API-Key": app_config.DEV_API_KEY},
            json={"question": "供应商资质需要哪些材料？", "session_id": "obs-2"},
        )
    names = {s.name for s in tracing.spans()}
    # 一次完整 rag_qa 至少经过 router/planner/executor/reflector/responder
    assert {"router", "planner", "executor", "reflector", "responder"}.issubset(names)


def test_request_id_propagated_in_response_header():
    with TestClient(app) as client:
        r = client.get("/api/health", headers={"X-Request-ID": "trace-xyz"})
        assert r.status_code == 200
        assert r.headers.get("X-Request-ID") == "trace-xyz"


def test_approval_flow_emits_approval_metrics():
    with TestClient(app) as client:
        # 写操作触发审批门：planner 走启发式计划 → executor 挂起待审批（requested）
        client.post(
            "/api/chat",
            headers={"X-API-Key": app_config.DEV_API_KEY},
            json={"question": "帮我建个工单：对账金额对不上", "session_id": "obs-3"},
        )
        m = client.get("/metrics")
        assert m.status_code == 200
        # 审批事件指标必须包含 requested（审批门被触发）
        assert 'srm_approval_events_total{event="requested"}' in m.text
