"""跨轮次会话记忆（进程级单例）。

按 ``session_id`` 维护一个 :class:`MemoryStore`，供编排层（``run_agent``）在每轮对话前
抽取多轮指代上下文注入 ``state["dialogue_context"]``，并在每轮结束后把本轮问答写回，
从而让「它怎么申请？」这类省略/指代在第二轮也能被正确理解。

设计为**确定性的进程内单例**：不依赖数据库、不联网、可复现，便于单测与 CI。
生产环境若需跨进程共享会话记忆，可在此替换为 Redis/Postgres 后端，对外接口保持不变。
"""
from __future__ import annotations

from app.agent.memory import MemoryStore

_store: MemoryStore = MemoryStore()


def get_memory_store() -> MemoryStore:
    """返回进程级会话记忆单例（按 session 隔离）。"""
    return _store


def reset_memory_store() -> None:
    """清空所有会话记忆 —— 测试 fixture 用于隔离用例，防止跨用例污染。"""
    global _store
    _store = MemoryStore()
