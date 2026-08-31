"""文本向量化（Embedder）抽象层。

设计目标（与 v1 对齐）
----------------------
- 生产用稠密向量（BGE / M3E 等 sentence-transformers 本地模型），语义召回远强于关键词。
- 但本仓库默认 **离线可跑、确定性、零额外依赖**，因此提供一个 `OfflineHashEmbedder`
  （字符二元切分 + 哈希抽桶的固定维向量，纯 Python，无 jieba / 无 numpy），
  作为「本地开发 / CI / 单测」的稠密向量替身。它与 BM25 经 RRF 融合后，
  仍能稳定覆盖 v1 的 20 题评测。

生产化切换
----------
通过 `get_embedder()` 工厂按环境变量选择嵌入器，调用方（检索编排层）零改动：

- `SRM_EMBEDDER=hash`  强制 `OfflineHashEmbedder`（确定性、零依赖）
- `SRM_EMBEDDER=bge`   强制 `BGEEmbedder`（依赖缺失 / 模型缺失即报错）
- 其他（默认 `auto`） 优先 `BGEEmbedder`，加载失败则回退 `OfflineHashEmbedder` 并告警

模型路径经 `SRM_BGE_MODEL` 指定，默认 `BAAI/bge-large-zh-v1.5`。
`BGEEmbedder` 对检索 query 自动补 BGE v1.5 中文指令前缀（文档侧不加），
该细节封装在 `embed_query()` / `embed()` 内，调用方无需关心。
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from typing import Any, Protocol

logger = logging.getLogger("srm.embeddings")


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

    def embed_query(self, text: str) -> list[float]: ...

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

    def embed_query(self, text: str) -> list[float]:
        """哈希嵌入器对 query / 文档一视同仁，直接返回单条向量。"""
        return self._vector(text)


class BGEEmbedder:
    """生产稠密向量：BGE 本地模型（sentence-transformers，lazy import）。

    相比 `OfflineHashEmbedder`，语义召回质量大幅提升。BGE v1.5 系列在检索场景下
    对 query 需要加指令前缀（中文："为这个句子生成表示以用于检索相关文章："），
    文档侧不加。本实现通过 `embed_query` / `embed` 分别处理，调用方无需关心前缀。

    依赖（可选，离线 / CI 默认不装）：sentence-transformers（会拉取 torch 等）。
    首次运行会按需从 HuggingFace 下载权重，国内可设 `HF_ENDPOINT=https://hf-mirror.com`。
    """

    # BGE v1.5 中文检索 query 指令前缀（官方推荐）
    _ZH_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    dim: int = 0
    model: Any

    def __init__(
        self,
        model_path: str | None = None,
        *,
        query_instruction: str | None = None,
        normalize: bool = True,
    ) -> None:
        # 延迟导入：未安装 sentence-transformers 时不影响包导入与离线运行
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model_path = (
            model_path or os.getenv("SRM_BGE_MODEL") or "BAAI/bge-large-zh-v1.5"
        )
        self._normalize = normalize

        # 自动为 BGE v1.5 中文模型补 query 指令前缀；可由 query_instruction 显式覆盖 / 置空关闭
        if query_instruction is not None:
            self._query_instruction = query_instruction
        elif "bge" in self._model_path.lower() and "v1.5" in self._model_path.lower():
            self._query_instruction = self._ZH_QUERY_INSTRUCTION
        else:
            self._query_instruction = ""

        self.model = SentenceTransformer(self._model_path)
        self.dim = self.model.get_sentence_embedding_dimension()

    def ensure_ready(self, corpus: list[str] | None = None) -> None:
        return

    def embed(self, texts: list[str]) -> list[list[float]]:
        """编码文档 / 语料（不加 query 指令前缀）。"""
        vecs = self.model.encode(
            texts,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            convert_to_tensor=False,
        )
        return [list(map(float, v)) for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        """编码检索 query（自动补 BGE v1.5 指令前缀，提升召回质量）。"""
        q = (self._query_instruction + text) if self._query_instruction else text
        return self.embed([q])[0]


# ── 工厂：按环境变量选择嵌入器，生产用 BGE，CI / 离线自动回退 hash ──────────
_EMBEDDER_CACHE: dict[str, Embedder] = {}


def get_embedder() -> Embedder:
    """按 `SRM_EMBEDDER` 选择稠密向量器，结果按 (mode, model) 缓存避免重复加载。

    - hash  强制 OfflineHashEmbedder（确定性、零依赖）
    - bge   强制 BGEEmbedder（缺失依赖 / 模型即报错）
    - auto（默认）优先 BGEEmbedder，加载失败则回退 OfflineHashEmbedder 并告警
    """
    mode = (os.getenv("SRM_EMBEDDER") or "auto").lower()
    model = os.getenv("SRM_BGE_MODEL") or "BAAI/bge-large-zh-v1.5"
    cache_key = f"{mode}:{model}"
    if cache_key in _EMBEDDER_CACHE:
        return _EMBEDDER_CACHE[cache_key]

    emb: Embedder
    if mode == "hash":
        emb = OfflineHashEmbedder()
    elif mode == "bge":
        emb = BGEEmbedder(model)  # 强制：缺失即报错
    else:  # auto
        try:
            emb = BGEEmbedder(model)
        except Exception as e:
            logger.warning(
                "BGE 嵌入器不可用（%s），回退到离线确定性 OfflineHashEmbedder", e
            )
            emb = OfflineHashEmbedder()

    _EMBEDDER_CACHE[cache_key] = emb
    return emb
