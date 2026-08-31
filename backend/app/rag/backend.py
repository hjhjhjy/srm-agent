"""检索后端抽象层。

设计取舍
--------
M1 目标：离线可跑、可测试、可替换。默认实现是**内存 BM25**（中文二元切分，
无需 jieba / 无需模型权重），保证 `pytest` 在任何环境都能跑通。

M2 起替换为 v1 的稠密混合检索（BGE + BM25 + RRF + CrossEncoder 重排），
**接口保持不变** —— 这正是抽象层存在的意义。

⚠️ 关键：本层强制做**租户隔离**过滤，补 v1 `retriever.py` 无租户过滤的 P0 缺口。
"""
from __future__ import annotations

import math
import re
from typing import Iterable, Optional, Protocol

from pydantic import BaseModel, Field


class KBHit(BaseModel):
    """知识库片段。"""

    chunk_id: str
    text: str = ""
    flow_code: str = ""
    flow_name: str = ""
    tenant_id: str = "public"  # public = 全租户可见的公共知识
    # 以下字段为 v1 语料结构对齐（用于检索过滤与引用溯源 / 附录意图提权）
    chunk_type: str = ""  # subflow | module_overview | appendix
    appendix_type: str = ""  # form | message | warning | report | interface
    module: str = ""
    score: float = 0.0


class RetrievalBackend(Protocol):
    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        flow_code: Optional[str] = None,
        tenant_id: str = "",
    ) -> list[KBHit]: ...

    def add(self, chunks: Iterable[KBHit]) -> None: ...


def tokenize(text: str) -> list[str]:
    """中文二元切分 + 英文数字整词。

    不依赖 jieba，离线可用。二元切分对中文检索的召回效果接近词级切分，
    且不存在新词未登录问题。
    """
    toks: list[str] = []
    for seg in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z0-9_]+", text):
        if "\u4e00" <= seg[0] <= "\u9fff":
            if len(seg) == 1:
                toks.append(seg)
            else:
                toks.extend(seg[i : i + 2] for i in range(len(seg) - 1))
        else:
            toks.append(seg.lower())
    return toks


class InMemoryBM25Backend:
    """BM25 Okapi，纯内存。用于 M1 与单元测试。"""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._hits: list[KBHit] = []
        self._tf: list[dict[str, int]] = []
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0

    def add(self, chunks: Iterable[KBHit]) -> None:
        for c in chunks:
            self._hits.append(c)
        self._rebuild()

    def _rebuild(self) -> None:
        self._tf, self._df = [], {}
        total = 0
        for c in self._hits:
            toks = tokenize(c.text)
            tf: dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            for t in tf:
                self._df[t] = self._df.get(t, 0) + 1
            total += len(toks)
        self._avgdl = total / len(self._hits) if self._hits else 0.0

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        flow_code: Optional[str] = None,
        tenant_id: str = "",
    ) -> list[KBHit]:
        n = len(self._hits)
        if n == 0:
            return []
        q = tokenize(query)
        scored: list[KBHit] = []
        for idx, tf in enumerate(self._tf):
            hit = self._hits[idx]
            # 租户隔离：只能命中本租户语料 + public 公共语料
            if hit.tenant_id not in ("public", tenant_id):
                continue
            if flow_code and hit.flow_code.upper() != flow_code.upper():
                continue
            dl = sum(tf.values()) or 1
            score = 0.0
            for t in q:
                f = tf.get(t, 0)
                if f == 0:
                    continue
                df = self._df.get(t, 0)
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                score += (
                    idf
                    * (f * (self.k1 + 1))
                    / (f + self.k1 * (1 - self.b + self.b * dl / (self._avgdl or 1.0)))
                )
            if score > 0:
                scored.append(KBHit(**{**hit.model_dump(), "score": score}))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


_backend: RetrievalBackend = InMemoryBM25Backend()


def get_backend() -> RetrievalBackend:
    return _backend


def set_backend(b: RetrievalBackend) -> None:
    """替换检索实现（如接入 v1 稠密混合检索 / pgvector）。"""
    global _backend
    _backend = b
