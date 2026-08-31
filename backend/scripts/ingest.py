"""语料 ingestion：原始文档 → 切片 →（离线）向量化元数据 → 版本化语料 JSON。

设计
----
- 切片：按「标题行（# 起头）」切分章节，章节内再按段落聚合到目标长度，避免把一段语义
  切碎；同时用正则抽取流程码 QS_SRM_* 作为该 chunk 的 flow_code（供检索过滤与引用溯源）。
- 向量化：本仓库离线优先，稠密向量在检索后端 `add()` 时由 `Embedder` 实时计算，因此
  ingestion 只落盘「文本 + 元数据」，不在此处写死向量。生产若要落盘向量（BGE + pgvector），
  把 `OfflineHashEmbedder` 换成 `BGEEmbedder` 并在后端持久化即可，本脚本无需改动。
- 版本化：输出带 `version` 与 `tenant_id` 的语料清单，便于按租户 / 版本管理知识库。

用法
----
    python scripts/ingest.py --input docs/blueprint.md --tenant public --version 2026.08 \
        --out app/rag/data/custom_corpus.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.backend import KBHit

FLOW_RE = re.compile(r"QS_SRM_[A-Za-z0-9_]+")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.*)$")
SEP_RE = re.compile(r"\n\s*\n")


def _split_sections(text: str) -> list[str]:
    """按标题切分章节，保留标题作为首段。"""
    lines = text.splitlines()
    sections: list[list[str]] = []
    buf: list[str] = []
    for ln in lines:
        if HEADING_RE.match(ln) and buf:
            sections.append(buf)
            buf = []
        buf.append(ln)
    if buf:
        sections.append(buf)
    if not sections:
        sections = [lines]
    return ["\n".join(s).strip() for s in sections if "\n".join(s).strip()]


def _chunk_section(section: str, max_chars: int = 400) -> list[str]:
    paras = [p.strip() for p in SEP_RE.split(section) if p.strip()]
    chunks: list[str] = []
    cur = ""
    for p in paras:
        if not cur:
            cur = p
        elif len(cur) + len(p) + 1 <= max_chars:
            cur = cur + "\n" + p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)
    return chunks


def chunk_document(text: str, max_chars: int = 400) -> list[dict]:
    out: list[dict] = []
    for sec in _split_sections(text):
        for body in _chunk_section(sec, max_chars):
            m = FLOW_RE.search(body)
            flow_code = m.group(0).upper() if m else ""
            out.append(
                {
                    "text": body,
                    "flow_code": flow_code,
                    "flow_name": "",
                    "chunk_type": "subflow",
                }
            )
    return out


def ingest(input_path: Path, tenant_id: str, version: str, out_path: Path) -> dict:
    text = input_path.read_text(encoding="utf-8")
    chunks = chunk_document(text)

    hits: list[dict] = []
    for i, c in enumerate(chunks):
        cid = f"{input_path.stem}:{i:03d}"
        hit = {
            "chunk_id": cid,
            "text": c["text"],
            "flow_code": c["flow_code"],
            "flow_name": c["flow_name"],
            "tenant_id": tenant_id,
            "chunk_type": c["chunk_type"],
            "appendix_type": "",
            "module": "",
            "version": version,
        }
        # 用 KBHit 做结构校验，确保与检索后端契约一致
        KBHit(**{k: v for k, v in hit.items() if k != "version"})
        hits.append(hit)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": version,
        "tenant_id": tenant_id,
        "source": str(input_path),
        "count": len(hits),
        "chunks": hits,
        "sha256": hashlib.sha256(
            json.dumps(hits, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:16],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path, help="原始文档（.md/.txt）")
    ap.add_argument("--tenant", default="public")
    ap.add_argument("--version", default="0.1.0")
    ap.add_argument("--out", required=True, type=Path, help="输出语料 JSON")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[ERR] 输入文件不存在: {args.input}")
        return 2
    stats = ingest(args.input, args.tenant, args.version, args.out)
    print(
        f"[OK] 切片 {stats['count']} 块 → {args.out} "
        f"(tenant={stats['tenant_id']} version={stats['version']} sha={stats['sha256']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
