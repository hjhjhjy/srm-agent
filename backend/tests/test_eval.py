"""离线评测门禁：跑通 v1 迁移过来的 20 题检索/回答/引用评测。

不依赖真实 LLM / 向量库：使用混合检索后端（BM25 + 离线稠密 + RRF）+ blueprint 语料，
断言检索/回答/引用/附录四项准确率达标（与 CI 中 `python -m app.rag.eval` 一致）。
"""
from __future__ import annotations

from app.rag.eval import run_eval
from app.rag.seed import seed_kb


def test_eval_gate_passes_thresholds():
    seed_kb()  # 用 blueprint 语料构建混合检索后端
    result = run_eval()
    s = result["summary"]
    assert result["ok"], f"评测未达标: {s}"
    assert (s["retrieval_acc"] or 0) >= 0.90
    assert (s["answer_acc"] or 0) >= 0.85
    assert (s["citation_acc"] or 0) >= 0.90
    if s["appendix_acc"] is not None:
        assert s["appendix_acc"] >= 0.90
