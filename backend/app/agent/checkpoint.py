"""状态检查点（M4）。

为什么需要它
------------
Phase 5 / M3 已具备 LangGraph 的 ``MemorySaver``（HITL 中断恢复依赖它），但它：
1. 与具体图实例绑定，外部不可直接 inspect / 导出 / 回放；
2. 不提供"按节点命名"的显式快照，排查"跑到哪一步、当时的完整状态是什么"很麻烦；
3. 跨进程 / 跨图实例不共享，无法做审计式轨迹留存。

本模块提供**与图实现解耦的显式检查点**：把 ``AgentState`` 序列化为纯 dict 快照，
按 ``(thread_id, node)`` 命名存入 ``CheckpointStore``，支持列出 / 读取 / 删除 / 恢复。
默认后端为内存实现，生产可替换为数据库 / 对象存储，对外接口不变。

快照必须确定性、可往返
---------------------
``snapshot_state`` 把 pydantic 模型与 langchain 消息转成 JSON 友好的 dict；
``restore_state`` 反向重建。两者互为逆操作，保证"存进去"与"拿出来"的语义一致，
从而支撑检查点回放（resume）与轨迹审计。

设计约束（与 M1~M3 一致）
------------------------
- 离线确定性：纯 stdlib 序列化，无随机数参与快照内容（ID 仅用于索引）。
- 零额外依赖：不引入新三方包。
- CI 常绿：``pytest`` / ``ruff`` / ``mypy`` 全绿。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from app.agent.state import (
    AuditEntry,
    Budget,
    PendingApproval,
    PlanStep,
    ToolCallRecord,
)

_MSG_TYPES = {"human": HumanMessage, "ai": AIMessage, "system": SystemMessage}


def _content_to_str(content: Any) -> str:
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, default=str)


def _msg_to_dict(msg: Any) -> dict[str, str]:
    """langchain 消息 → {type, content} 纯字典。"""
    if isinstance(msg, BaseMessage):
        return {"type": getattr(msg, "type", "human"), "content": _content_to_str(msg.content)}
    if isinstance(msg, dict):
        return {"type": msg.get("type", "human"), "content": _content_to_str(msg.get("content", ""))}
    return {"type": "human", "content": str(msg)}


def _dumpable(v: Any) -> Any:
    """pydantic / 容器 → JSON 友好结构（model_dump 优先）。"""
    if hasattr(v, "model_dump"):
        return v.model_dump()
    if isinstance(v, list):
        return [_dumpable(x) for x in v]
    if isinstance(v, dict):
        return {k: _dumpable(x) for k, x in v.items()}
    return v


def snapshot_state(state: dict[str, Any]) -> dict[str, Any]:
    """把 ``AgentState`` 序列化为可 JSON 化的纯 dict 快照。"""
    out: dict[str, Any] = {}
    for k, v in state.items():
        if k == "messages":
            out[k] = [_msg_to_dict(m) for m in (v or [])]
        else:
            out[k] = _dumpable(v)
    return out


def restore_state(snap: dict[str, Any]) -> dict[str, Any]:
    """``snapshot_state`` 的逆操作：把纯 dict 重建为 ``AgentState`` 可用结构。"""
    out: dict[str, Any] = dict(snap)
    msgs: list[BaseMessage] = []
    for m in snap.get("messages", []):
        mtype = m.get("type", "human")
        cls = _MSG_TYPES.get(mtype, HumanMessage)
        msgs.append(cls(content=m.get("content", "")))
    out["messages"] = msgs

    out["budget"] = Budget(**snap["budget"]) if snap.get("budget") else Budget()
    out["audit"] = [AuditEntry(**a) for a in snap.get("audit", [])]
    out["plan"] = [PlanStep(**p) for p in snap.get("plan", [])]
    out["tool_calls"] = [ToolCallRecord(**t) for t in snap.get("tool_calls", [])]
    pa = snap.get("pending_approval")
    out["pending_approval"] = PendingApproval(**pa) if pa else None
    return out


@dataclass
class Checkpoint:
    """一次状态快照（按 thread_id + node 命名）。"""

    id: str
    thread_id: str
    node: str
    ts: float
    snapshot: dict[str, Any]


class CheckpointStore:
    """检查点存储接口（默认内存实现，可替换为持久化后端）。"""

    def save(self, state: dict[str, Any], *, thread_id: str, node: str, now: float | None = None) -> str:
        raise NotImplementedError  # pragma: no cover - 接口

    def load(self, checkpoint_id: str) -> Checkpoint | None:
        raise NotImplementedError  # pragma: no cover - 接口

    def list(self, thread_id: str) -> list[Checkpoint]:
        raise NotImplementedError  # pragma: no cover - 接口

    def delete(self, checkpoint_id: str) -> bool:
        raise NotImplementedError  # pragma: no cover - 接口

    def delete_thread(self, thread_id: str) -> int:
        raise NotImplementedError  # pragma: no cover - 接口


class InMemoryCheckpointStore(CheckpointStore):
    """进程内检查点存储。"""

    def __init__(self) -> None:
        self._store: dict[str, Checkpoint] = {}
        self._seq: int = 0

    def save(self, state: dict[str, Any], *, thread_id: str, node: str, now: float | None = None) -> str:
        self._seq += 1
        ts = now if now is not None else time.time()
        raw = f"{thread_id}|{node}|{ts}|{self._seq}"
        cid = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
        self._store[cid] = Checkpoint(
            id=cid, thread_id=thread_id, node=node, ts=ts, snapshot=snapshot_state(state)
        )
        return cid

    def load(self, checkpoint_id: str) -> Checkpoint | None:
        return self._store.get(checkpoint_id)

    def list(self, thread_id: str) -> list[Checkpoint]:
        return sorted(
            (c for c in self._store.values() if c.thread_id == thread_id),
            key=lambda c: c.ts,
        )

    def delete(self, checkpoint_id: str) -> bool:
        return self._store.pop(checkpoint_id, None) is not None

    def delete_thread(self, thread_id: str) -> int:
        ids = [cid for cid, c in self._store.items() if c.thread_id == thread_id]
        for cid in ids:
            del self._store[cid]
        return len(ids)


async def resume_from_checkpoint(
    graph: Any,
    store: CheckpointStore,
    thread_id: str,
    checkpoint_id: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """从指定检查点恢复并重新跑完该图（回放 / 断点续跑）。

    Returns:
        恢复后图的最终状态；若检查点不存在返回 ``None``。
    """
    cp = store.load(checkpoint_id)
    if cp is None or cp.thread_id != thread_id:
        return None
    state = restore_state(cp.snapshot)
    # 用独立的 replay 线程承载回放，避免与原始线程在 MemorySaver 中的状态相互合并，
    # 保证"从检查点重建并重新执行"语义确定、可复现（审计/回放用法）。
    cfg = config or {"configurable": {"thread_id": f"{thread_id}__replay"}}
    result = await graph.ainvoke(state, config=cfg)
    return dict(result) if isinstance(result, dict) else result


# ── 进程级单例 ──────────────────────────────────────────────────────────

_store: CheckpointStore | None = None


def get_checkpoint_store() -> CheckpointStore:
    """返回进程级检查点存储单例（默认内存）。"""
    global _store
    if _store is None:
        _store = InMemoryCheckpointStore()
    return _store


def reset_checkpoint_store() -> None:
    """清空检查点存储（测试 fixture 用于隔离用例）。"""
    global _store
    _store = None
