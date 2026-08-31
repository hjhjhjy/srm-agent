"""文本向量化（Embedder）抽象层。

设计目标（与 v1 对齐）
----------------------
- 生产用稠密向量（BGE / M3E 等 sentence-transformers 本地模型），语义召回远强于关键词。
- 但本仓库默认 **离线可跑、确定性、零额外依赖**，因此提供一个 `OfflineHashEmbedder`
  （字符二元切分 + 哈希抽桶的固定维向量，纯 Python，无 jieba / 无 numpy），
  作为「本地开发 / CI / 单测」的稠密向量替身。它与 BM25 经 RRF 融合后，
  仍能稳定覆盖 v1 的 20 题评测。

切换生产稠密向量：把 `OfflineHashEmbedder` 换成 `BGEEmbedder(model_path)`，
接口一致（`embed` / `ensure_ready` / `dim`），检索编排层无需改动。
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


def tokenize(text: str) -> list[str]:
    """中文二元切分 + 英文数字整词（与 backend.py 一致，保证可复现）。"""
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


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def ensure_ready(self, corpus: list[str] | None = None) -> None: ...


class OfflineHashEmbedder:
    """确定性哈希向量（开发 / CI 兜底），不依赖任何模型权重。

    把 token 哈希到固定维度桶并累加词频，再做 L2 归一化。与 BM25 经 RRF 融合后，
    语义近似召回稳定，且跨环境 100% 可复现（同一文本永远得到同一向量）。
    """

    dim: int = 512

    def __init__(self, dim: int = 512, normalize: bool = True) -> None:
        self.dim = dim
        self._normalize = normalize

    def ensure_ready(self, corpus: list[str] | None = None) -> None:
        return

    def _vector(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self.dim] += 1.0
        if self._normalize:
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
        return vec

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class BGEEmbedder:
    """生产稠密向量（BGE / M3E 本地模型）。仅在生产环境按需启用（lazy import）。

    用法：
        from app.rag.embeddings import BGEEmbedder
        emb = BGEEmbedder("/path/to/bge-small-zh")
        backend = HybridBackend(embedder=emb)
    """

    dim: int = 0

    def __init__(self, model_path: str) -> None:
        # 国内网络可走镜像：os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path)
        self.dim = self.model.get_sentence_embedding_dimension()

    def ensure_ready(self, corpus: list[str] | None = None) -> None:
        return

    def embed(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]
