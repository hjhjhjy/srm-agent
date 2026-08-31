"""嵌入器（Embedder）单测：OfflineHashEmbedder 确定性 + get_embedder 工厂回退 + BGE 真实路径。

设计要点
--------
- OfflineHashEmbedder 必须确定性、L2 归一、query/doc 一致（embed_query == embed([t])[0]）。
- get_embedder() 在 auto 模式下，BGE 加载失败必须**静默回退**到 OfflineHashEmbedder，
  保证离线 / CI 永远可跑（用模拟异常覆盖该分支，避免触发真实模型下载）。
- BGE 真实向量化（维度 / 形状 / 确定性）默认跳过，需 `SRM_RUN_BGE_TESTS=1` 且本地有模型才跑，
  以免 CI 联网下载大模型。
"""
from __future__ import annotations

import os

import pytest

import app.rag.embeddings as emb_mod
from app.rag.embeddings import BGEEmbedder, OfflineHashEmbedder, get_embedder


# ── OfflineHashEmbedder ────────────────────────────────────────────────────
def test_offline_hash_is_deterministic():
    e = OfflineHashEmbedder()
    v1 = e.embed(["供应商注册流程"])
    v2 = e.embed(["供应商注册流程"])
    assert v1 == v2
    assert len(v1[0]) == e.dim


def test_offline_hash_differs_for_different_texts():
    e = OfflineHashEmbedder()
    a = e.embed(["供应商注册流程"])[0]
    b = e.embed(["资质准入要求"])[0]
    assert a != b


def test_offline_hash_is_l2_normalized():
    import math

    e = OfflineHashEmbedder()
    v = e.embed(["营业执照 身份证 开户许可证"])[0]
    norm = math.sqrt(sum(x * x for x in v))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_offline_hash_embed_query_equals_embed_single():
    e = OfflineHashEmbedder()
    assert e.embed_query("如何注册供应商") == e.embed(["如何注册供应商"])[0]


# ── get_embedder 工厂 ──────────────────────────────────────────────────────
def test_get_embedder_hash_mode_returns_offline(monkeypatch):
    emb_mod._EMBEDDER_CACHE.clear()
    monkeypatch.setenv("SRM_EMBEDDER", "hash")
    monkeypatch.delenv("SRM_BGE_MODEL", raising=False)
    emb = get_embedder()
    assert isinstance(emb, OfflineHashEmbedder)


def test_get_embedder_auto_falls_back_when_bge_fails(monkeypatch):
    """auto 模式下 BGE 加载失败必须回退到 OfflineHashEmbedder（且不抛异常）。"""
    emb_mod._EMBEDDER_CACHE.clear()
    monkeypatch.setenv("SRM_EMBEDDER", "auto")
    monkeypatch.delenv("SRM_BGE_MODEL", raising=False)

    class _Boom:
        def __init__(self, *a, **k):
            raise RuntimeError("simulated BGE load failure")

    monkeypatch.setattr(emb_mod, "BGEEmbedder", _Boom)
    emb = get_embedder()
    assert isinstance(emb, OfflineHashEmbedder)


def test_get_embedder_caches_by_mode(monkeypatch):
    emb_mod._EMBEDDER_CACHE.clear()
    monkeypatch.setenv("SRM_EMBEDDER", "hash")
    monkeypatch.delenv("SRM_BGE_MODEL", raising=False)
    a = get_embedder()
    b = get_embedder()
    assert a is b


# ── BGEEmbedder query 指令前缀逻辑（注入假 sentence_transformers，无需真实依赖/联网）──
def _install_fake_sentence_transformers(monkeypatch, captured):
    import sys
    import types

    class _FakeST:
        def __init__(self, model_path):
            captured["model_path"] = model_path

        def get_sentence_embedding_dimension(self):
            return 4

        def encode(self, texts, **kwargs):
            captured.setdefault("calls", []).append(list(texts))
            # 确定性伪向量（仅用于形态 / 形状校验）
            return [[float(len(t)), 0.0, 0.0, 0.0] for t in texts]

    fake = types.ModuleType("sentence_transformers")
    fake.SentenceTransformer = _FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)


def test_bge_embed_query_applies_zh_instruction_prefix(monkeypatch):
    captured: dict = {}
    _install_fake_sentence_transformers(monkeypatch, captured)
    emb = BGEEmbedder("BAAI/bge-large-zh-v1.5")
    assert emb.dim == 4
    # 文档编码不加前缀
    emb.embed(["供应商注册"])
    assert captured["calls"][-1] == ["供应商注册"]
    # 检索 query 必须自动补 BGE v1.5 中文指令前缀
    emb.embed_query("如何注册")
    assert captured["calls"][-1] == ["为这个句子生成表示以用于检索相关文章：如何注册"]


def test_bge_no_prefix_for_non_v15_model(monkeypatch):
    captured: dict = {}
    _install_fake_sentence_transformers(monkeypatch, captured)
    emb = BGEEmbedder("BAAI/bge-m3")  # 非 v1.5，不应加前缀
    emb.embed_query("如何注册")
    assert captured["calls"][-1] == ["如何注册"]


# ── BGEEmbedder 真实路径（默认跳过，避免 CI 联网下载大模型）─────────────────
@pytest.mark.skipif(
    os.getenv("SRM_RUN_BGE_TESTS") != "1",
    reason="需本地 BGE 模型与 sentence-transformers，默认跳过以保 CI 离线",
)
def test_bge_embedder_real():
    st = pytest.importorskip("sentence_transformers")
    assert st is not None
    emb = BGEEmbedder()  # 读 SRM_BGE_MODEL 或默认 BAAI/bge-large-zh-v1.5
    assert emb.dim > 0

    docs = emb.embed(["供应商注册流程", "资质准入要求"])
    assert len(docs) == 2
    assert all(len(v) == emb.dim for v in docs)

    q = emb.embed_query("如何注册供应商")
    assert len(q) == emb.dim

    # 确定性：同输入同输出
    again = emb.embed(["供应商注册流程"])
    assert docs[0] == again[0]

    # BGE v1.5 默认补中文 query 指令前缀 → query 向量应与「无前缀文档编码」不同
    doc_eq = emb.embed(["如何注册供应商"])[0]
    assert q != doc_eq
