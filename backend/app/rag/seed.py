"""默认知识库语料与启动 seeding。

M2 Phase 2 起：seeding 使用 **混合检索后端**（BM25 + 稠密 + RRF），
语料来自 `data/srm_blueprint.json`（覆盖 v1 的 20 题评测所需的全部流程码与附录类型）。
生产可替换为真实 ingestion 产物（见 `scripts/ingest.py`）。
"""
from __future__ import annotations

from pathlib import Path

from app.rag.backend import KBHit, set_backend
from app.rag.hybrid import HybridBackend

_BLUEPRINT_PATH = Path(__file__).resolve().parent / "data" / "srm_blueprint.json"


def load_blueprint(tenant_id: str = "public") -> list[KBHit]:
    """从 blueprint JSON 载入知识库片段。"""
    import json

    raw = json.loads(_BLUEPRINT_PATH.read_text(encoding="utf-8"))
    return [KBHit(**item) for item in raw]


def seed_kb() -> None:
    """用 blueprint 语料填充混合检索后端（覆盖空语料）。"""
    backend = HybridBackend()
    backend.add(load_blueprint())
    set_backend(backend)
