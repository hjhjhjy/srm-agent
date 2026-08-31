"""Phase 4 记忆模块单元测试（确定性、零依赖、CI 常绿）。"""
from __future__ import annotations

from app.agent.memory import (
    build_coref_context,
    compress_messages,
    has_coref,
)
from app.agent.session import get_memory_store, reset_memory_store


def test_has_coref_detects_pronouns():
    assert has_coref("它怎么申请？")
    assert has_coref("这个流程需要多久")
    assert has_coref("上面说的规定")
    assert not has_coref("如何注册成为供应商")
    assert not has_coref("")


def test_build_coref_context_formats_recent_turns():
    history = [
        {"role": "user", "content": "如何注册供应商\n第二行噪声"},
        {"role": "assistant", "content": "进入 SRM 门户点击注册\n更多细节"},
        {"role": "user", "content": "它需要哪些材料"},
    ]
    ctx = build_coref_context(history)
    # 助手答案只保留首行，避免冗长回答的噪声带进上下文
    assert "如何注册供应商" in ctx
    assert "它需要哪些材料" in ctx
    assert "第二行噪声" not in ctx
    assert "更多细节" not in ctx


def test_build_coref_context_truncates_and_limits():
    history = [{"role": "user", "content": "q" * 500} for _ in range(10)]
    ctx = build_coref_context(history, max_turns=4, max_len=20)
    # 只取最近 4 条，且单条截断到 20 字符
    assert ctx.count("q") <= 4 * 20
    assert "q" in ctx


def test_compress_messages_keeps_head_and_tail():
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    out = compress_messages(msgs, max_keep=6)
    assert len(out) == 6
    assert out[0]["content"] == "m0"  # 首条保留
    assert out[-1]["content"] == "m19"  # 最新保留
    assert "m1" not in [m["content"] for m in out]


def test_memory_store_folds_old_messages_into_summary():
    reset_memory_store()
    store = get_memory_store()
    sid = "sess1"
    for i in range(10):
        store.append(sid, "user", f"第{i}轮问题")
    ctx = store.context(sid)
    # 超出 max_recent(默认6) 的早期消息被折叠进摘要
    assert "第0轮问题" in ctx and "历史摘要" in ctx
    assert "第9轮问题" in ctx  # 近期明细保留


def test_memory_store_isolated_by_session():
    reset_memory_store()
    store = get_memory_store()
    store.append("a", "user", "A 的问题")
    store.append("b", "user", "B 的问题")
    assert "A 的问题" in store.context("a")
    assert "B 的问题" not in store.context("a")
    assert "B 的问题" in store.context("b")


def test_reset_memory_store_clears():
    store = get_memory_store()
    store.append("x", "user", "temp")
    reset_memory_store()
    assert get_memory_store().context("x") == ""
