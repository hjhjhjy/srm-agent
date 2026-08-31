"""混合检索后端：BM25 关键词召回 + 稠密向量召回 + RRF 融合 + 附录意图提权。

这是 v1 `Retriever` 的离线可跑移植版：
- 去掉了对 Milvus / pgvector / reranker 的硬依赖，稠密召回改由本地 `Embedder` 完成；
- **完整保留租户隔离过滤**与「附录意图提权 / 流程感知附录注入」逻辑（v1 的核心召回精度手段）；
- 通过 `RetrievalBackend` 协议接入编排层，替换 `set_backend` 即可，零改动调用方。

融合方式：RRF（Reciprocal Rank Fusion），对 BM25 与稠密两路各自排序后按 1/(rank+k) 累加，
避免两套打分尺度不可比的问题。附录提权解决「正文 subflow 压过关键附录」导致的召回不精准。
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, List, Optional

from app.rag.backend import (
    InMemoryBM25Backend,
    KBHit,
    RetrievalBackend,
    tokenize,
)
from app.rag.embeddings import Embedder, OfflineHashEmbedder

logger = logging.getLogger("srm.hybrid")

FLOW_RE = re.compile(r"QS_SRM_[A-Z0-9_]+", re.IGNORECASE)

# 问题意图关键词 → 附录类型 → 提权权重（与 v1 一致）。
# 蓝图附录按 form/message/warning/report/interface 分类；当问题意图明确指向某类附录时，
# 对已进入候选的该类附录显著提权，解决融合下「正文压过关键附录」问题。
# 注意：意图规则只负责「是否启用某类附录提权」，真正的提权幅度由 search() 中
# 按附录自身 BM25 排名动态计算（越相关越强），避免「只要命中意图就给所有同类附录
# 统一加固定权重」而淹没正文 subflow（v1 的 +0.12/+0.22 在 RRF 小分值尺度下会失控）。
# 这里的正则刻意只保留「文档/通知」本身的强信号词，去掉竞价/招标/确认/订单/样/收到等
# 过于宽泛的流程词——否则「参与竞价」「接收订单」这类问题会误触发 form/message 提权。
_INTENT_RULES: List[tuple[re.Pattern, str, float]] = [
    (re.compile(r"文件|材料|资料|表单|清单|证件|证书|执照|身份证|营业执照|首营|协议"), "form", 0.18),
    (re.compile(r"通知|消息|短信|查收|提醒|告知|函|发送|收取|发票"), "message", 0.15),
    (re.compile(r"预警|到期|超时|逾期|临期|提前|过期"), "warning", 0.15),
    (re.compile(r"报表|统计|绩效|合格率|进度|查询|分析|看板|指标|明细|查看|哪里"), "report", 0.18),
    (re.compile(r"接口|对接|集成|同步|API|报文|SRM\d+"), "interface", 0.18),
]


class HybridBackend(RetrievalBackend):
    """BM25 + 稠密 + RRF 融合，带租户隔离与附录意图提权。"""

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        rrf_k: int = 60,
        dense_weight: float = 0.35,
        boost_appendix_intent: float = 0.02,
        boost_appendix_flow: float = 0.02,
        mmr_lambda: float = 0.55,
    ) -> None:
        self._bm25 = InMemoryBM25Backend(k1=k1, b=b)
        self._embedder = embedder or OfflineHashEmbedder()
        self._hits: List[KBHit] = []
        self._vecs: List[List[float]] = []
        self._by_id: dict[str, int] = {}
        self._rrf_k = rrf_k
        # 稠密路权重：离线 hash embedder 语义噪声较大，作为「语义纠偏」而非主信号，
        # 主信号交给 BM25（中文关键词精准），避免泛化 chunk 在稠密路霸榜。
        self._dense_weight = dense_weight
        self._boost_app_intent = boost_appendix_intent
        self._boost_app_flow = boost_appendix_flow
        # MMR 多样性：抑制与已选 chunk 高度相似的冗余项，给特定 subflow 留出位置。
        self._mmr_lambda = mmr_lambda

    def add(self, chunks: Iterable[KBHit]) -> None:
        self._hits = list(chunks)
        self._by_id = {h.chunk_id: i for i, h in enumerate(self._hits)}
        # 稠密向量：离线确定性 embedder 无需训练，直接编码
        corpus = [h.text for h in self._hits]
        self._embedder.ensure_ready(corpus)
        self._vecs = self._embedder.embed(corpus)
        self._bm25.add(self._hits)

    def _tenant_ok(self, hit: KBHit, tenant_id: str) -> bool:
        return hit.tenant_id in ("public", tenant_id)

    def search(
        self,
        query: str,
        *,
        top_k: int = 4,
        flow_code: Optional[str] = None,
        tenant_id: str = "",
    ) -> List[KBHit]:
        if not self._hits:
            return []

        flow_code = flow_code.upper() if flow_code else None

        # ── BM25 路 ──
        bm25_hits = self._bm25.search(
            query, top_k=len(self._hits), flow_code=flow_code, tenant_id=tenant_id
        )
        bm25_rank = {h.chunk_id: r for r, h in enumerate(bm25_hits)}

        # ── 稠密路 ──
        q_vec = self._embedder.embed([query])[0]
        dense: List[tuple[int, float]] = []
        for i, hit in enumerate(self._hits):
            if not self._tenant_ok(hit, tenant_id):
                continue
            if flow_code and hit.flow_code.upper() != flow_code:
                continue
            cos = sum(a * b for a, b in zip(q_vec, self._vecs[i]))
            dense.append((i, cos))
        dense.sort(key=lambda x: x[1], reverse=True)
        dense_rank = {self._hits[i].chunk_id: r for r, (i, _) in enumerate(dense)}

        # ── 加权 RRF 融合 ──
        # BM25 为主信号（中文关键词精准），稠密路降权（离线 hash embedder 语义噪声大，
        # 仅作语义纠偏），避免泛化 chunk 在稠密路霸榜而把特定 subflow 挤出 top-K。
        fused: dict[str, float] = {}
        for cid, r in bm25_rank.items():
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (r + self._rrf_k)
        for cid, r in dense_rank.items():
            fused[cid] = fused.get(cid, 0.0) + self._dense_weight / (r + self._rrf_k)

        # 候选（取前 40，留给附录注入空间）
        cand_ids = sorted(fused, key=lambda c: fused[c], reverse=True)[:40]

        # ── 附录提权（两路均为「比例提权」，仅抬升「自身确实相关」的附录，
        #    绝不给整类附录统一加固定权重——否则会把不相关附录也抬进 top-K 淹没正文 subflow）──
        # 1) 流程感知：若某 subflow 正文已召回，其同流程附录按比例提权（相关度越高提得越多）
        cand_flows = {
            self._hits[self._by_id[c]].flow_code
            for c in cand_ids
            if self._hits[self._by_id[c]].chunk_type == "subflow"
            and self._hits[self._by_id[c]].flow_code
        }
        for cid in list(fused):
            h = self._hits[self._by_id[cid]]
            if h.chunk_type == "appendix" and h.flow_code in cand_flows:
                r = bm25_rank.get(cid)
                if r is None:
                    continue
                rel = max(0.0, 1.0 - r / 20.0)
                if rel <= 0:
                    continue
                fused[cid] += self._boost_app_flow * rel

        # 2) 意图类型提权：问题意图明确指向某类附录时，对候选集中该类附录按比例提权
        mw_types = {atype for pat, atype, _ in _INTENT_RULES if pat.search(query)}
        if mw_types:
            for cid in list(fused):
                h = self._hits[self._by_id[cid]]
                if h.chunk_type != "appendix" or h.appendix_type not in mw_types:
                    continue
                r = bm25_rank.get(cid)
                if r is None:
                    continue
                rel = max(0.0, 1.0 - r / 20.0)  # 仅 BM25 前 20 名的附录参与提权
                if rel <= 0:
                    continue
                fused[cid] += self._boost_app_intent * rel

        # ── MMR 多样性重排 ──
        # 抑制与已选 chunk 高度相似的冗余项（泛化 subflow 彼此相似），把位置留给
        # 真正相关的特定 subflow（如 PO_0008/0010、RM_0002、SM_0004/0005、RD_0001）。
        # 惩罚采用乘性（fused[c] * (1 - λ*max_sim)），使相似度惩罚落在 fused 同一量纲，
        # 既去冗余又不会把高相关项一刀切掉。
        selected: List[str] = []
        pool = sorted(fused, key=lambda c: -fused[c])
        while len(selected) < top_k and pool:
            if not selected:
                best = pool[0]
            else:
                sel_idx = [self._by_id[s] for s in selected]
                best = None
                best_score = None
                for c in pool:
                    if c in selected:
                        continue
                    ci = self._by_id[c]
                    max_sim = 0.0
                    for si in sel_idx:
                        dot = sum(
                            a * b for a, b in zip(self._vecs[ci], self._vecs[si])
                        )
                        if dot > max_sim:
                            max_sim = dot
                    s = fused[c] * (1.0 - self._mmr_lambda * max_sim)
                    if best_score is None or s > best_score:
                        best_score = s
                        best = c
            selected.append(best)
            pool.remove(best)

        out: List[KBHit] = []
        for cid in selected:
            h = self._hits[self._by_id[cid]]
            out.append(KBHit(**{**h.model_dump(), "score": round(fused[cid], 4)}))
        return out
