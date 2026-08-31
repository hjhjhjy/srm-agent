"""SRM 智能问答 Agent —— 检索/回答/引用 离线评测。

与线上行为保持一致：混合检索 top_k → 无 LLM 时走检索增强直答（fallback）→ 构建引用。
对 `data/eval_questions.json` 的题目计算：

  - retrieval_acc : 期望流程码（QS_SRM_*）是否全部出现在 top-K 召回中
  - answer_acc    : 期望关键词是否全部出现在（兜底）回答文本中
  - citation_acc  : 引用集合对期望流程码的覆盖率（= retrieval_acc）
  - appendix_acc  : 若题目指定 expected_appendix，top-K 中是否命中该附录类型

可作为模块被 pytest 调用，也可直接执行：
    python -m app.rag.eval [--fail-under 0.85]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

from app.rag.backend import RetrievalBackend, get_backend
from app.rag.backend import KBHit

FLOW_RE = re.compile(r"QS_SRM_[A-Z0-9_]+", re.IGNORECASE)
_Q_DIR = Path(__file__).resolve().parent / "data"
_Q_FILE = _Q_DIR / "eval_questions.json"

TOP_K = 8


def load_questions() -> list[dict]:
    return json.loads(_Q_FILE.read_text(encoding="utf-8"))


def _flow_codes_from_hits(hits: list[KBHit]) -> set:
    codes = set()
    for h in hits:
        if h.flow_code:
            codes.add(h.flow_code.upper())
        codes.update(m.upper() for m in FLOW_RE.findall(h.text or ""))
    return codes


def _appendix_types_from_hits(hits: list[KBHit]) -> set:
    return {h.appendix_type for h in hits if h.appendix_type}


def _build_answer(hits: list[KBHit], question: str) -> str:
    """复刻线上无 LLM 路径：检索增强直答。"""
    if not hits:
        return (
            "当前帮助文档（业务蓝图）中未检索到与您问题直接相关的内容。"
            "建议您联系青山利康对接采购专员，或在 SRM 系统内转人工处理。"
        )
    lines = ["根据《青山利康 SRM 业务蓝图》，为您整理以下相关信息：\n"]
    for i, h in enumerate(hits, 1):
        label = h.flow_code or h.flow_name or f"片段{i}"
        name = h.flow_name or ""
        header = f"【{label}】{name}".rstrip()
        doc = (h.text or "").strip().replace("\n", " ")
        snippet = doc[:700] + ("…" if len(doc) > 700 else "")
        lines.append(f"{i}. {header}\n   {snippet}\n")
    lines.append(
        "\n（以上为蓝图原文片段的检索直答；配置 LLM API Key 后可获得更自然的归纳回答。）"
    )
    return "\n".join(lines)


def _build_citations(hits: list[KBHit]) -> list[dict]:
    return [
        {
            "flow_code": h.flow_code,
            "flow_name": h.flow_name,
            "appendix_type": h.appendix_type,
            "code": "",
            "module": h.module,
            "source_snippet": (h.text or "")[:220],
        }
        for h in hits
    ]


def eval_question(item: dict, backend: RetrievalBackend, top_k: int = TOP_K) -> dict:
    q = item["question"]
    exp_codes = [c.upper() for c in item.get("expected_flow_codes", [])]
    exp_kw = item.get("expected_keywords", [])
    exp_app = item.get("expected_appendix", "")

    hits = backend.search(q, top_k=top_k, tenant_id="public")
    answer = _build_answer(hits, q)
    _build_citations(hits)  # 引用与检索同源

    found_codes = _flow_codes_from_hits(hits)
    found_app = _appendix_types_from_hits(hits)
    answer_text = answer or ""

    if exp_codes:
        missing_codes = [c for c in exp_codes if c not in found_codes]
        retrieval_pass = len(missing_codes) == 0
        retrieval_detail = "OK" if retrieval_pass else f"missing {missing_codes}"
    else:
        docs = "\n".join((h.text or "") for h in hits)
        miss = [k for k in exp_kw if k not in docs]
        retrieval_pass = len(miss) == 0
        retrieval_detail = "no-code(chk kw)" if retrieval_pass else f"kw miss {miss}"

    missing_kw = [k for k in exp_kw if k not in answer_text]
    answer_pass = len(missing_kw) == 0
    answer_detail = "OK" if answer_pass else f"kw miss {missing_kw}"

    if exp_codes:
        missing_cite = [c for c in exp_codes if c not in found_codes]
        citation_pass = len(missing_cite) == 0
    else:
        citation_pass = retrieval_pass

    if exp_app:
        appendix_pass = exp_app in found_app
        appendix_detail = "OK" if appendix_pass else f"no {exp_app} in {sorted(found_app)}"
    else:
        appendix_pass = True
        appendix_detail = "n/a"

    return {
        "question": q,
        "n_results": len(hits),
        "retrieval_pass": retrieval_pass,
        "retrieval_detail": retrieval_detail,
        "answer_pass": answer_pass,
        "answer_detail": answer_detail,
        "citation_pass": citation_pass,
        "appendix_pass": appendix_pass,
        "appendix_detail": appendix_detail,
        "found_codes": sorted(found_codes),
        "found_appendix": sorted(found_app),
    }


def run_eval(backend: Optional[RetrievalBackend] = None, top_k: int = TOP_K, fail_under: Optional[float] = None) -> dict:
    backend = backend or get_backend()
    questions = load_questions()
    rows = [eval_question(it, backend, top_k) for it in questions]

    def rate(key, pred=lambda _: True):
        sub = [r for r in rows if pred(r)]
        if not sub:
            return None, 0
        return sum(r[key] for r in sub) / len(sub), len(sub)

    retrieval_acc, n_ret = rate("retrieval_pass")
    answer_acc, n_ans = rate("answer_pass")
    citation_acc, n_cite = rate("citation_pass")
    appendix_acc, n_app = rate("appendix_pass", lambda r: r["appendix_detail"] != "n/a")

    print("\n=== 逐题评测 ===")
    print(f"{'Q':>2} | {'检索':<4} | {'回答':<4} | {'引用':<4} | {'附录':<4} | 问题 / 备注")
    print("-" * 90)
    for i, r in enumerate(rows, 1):
        print(
            f"{i:>2} | "
            f"{'✓' if r['retrieval_pass'] else '✗':<4} | "
            f"{'✓' if r['answer_pass'] else '✗':<4} | "
            f"{'✓' if r['citation_pass'] else '✗':<4} | "
            f"{'✓' if r['appendix_pass'] else '·':<4} | "
            f"{r['question']}  [{r['retrieval_detail']}]"
        )

    print("\n=== 汇总指标 ===")
    print(f"题目总数            : {len(rows)}")
    print(f"检索准确率(retrieval): {retrieval_acc*100:5.1f}%  (n={n_ret})   目标 ≥90%")
    print(f"回答准确率(answer)  : {answer_acc*100:5.1f}%  (n={n_ans})   目标 ≥85%")
    print(f"引用准确率(citation): {citation_acc*100:5.1f}%  (n={n_cite})   目标 ≥90%")
    if appendix_acc is not None:
        print(f"附录命中率(appendix): {appendix_acc*100:5.1f}%  (n={n_app})   目标 ≥90%")

    fails = [r for r in rows if not (r["retrieval_pass"] and r["answer_pass"] and r["citation_pass"])]
    if fails:
        print("\n--- 未达标项 ---")
        for r in fails:
            print(f"  · {r['question']}")
            print(f"      retrieval: {r['retrieval_detail']}")
            print(f"      answer   : {r['answer_detail']}")
            print(f"      found_codes={r['found_codes']} found_appendix={r['found_appendix']}")

    summary = {
        "total": len(rows),
        "retrieval_acc": retrieval_acc,
        "answer_acc": answer_acc,
        "citation_acc": citation_acc,
        "appendix_acc": appendix_acc,
    }

    ok = (
        (retrieval_acc or 0) >= 0.90
        and (answer_acc or 0) >= 0.85
        and (citation_acc or 0) >= 0.90
        and (appendix_acc if appendix_acc is not None else 1.0) >= 0.90
    )
    if fail_under is not None:
        ok = ok and (min(filter(None, [retrieval_acc, answer_acc, citation_acc, appendix_acc or 1.0])) >= fail_under)

    print("结果:", "达标 ✅" if ok else "未达标 ⚠️")
    return {"summary": summary, "rows": rows, "ok": ok}


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-under", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    from app.rag.seed import seed_kb

    seed_kb()
    result = run_eval(top_k=args.top_k, fail_under=args.fail_under)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
