"""四层记忆架构（M4）。

把"记忆"从一个 Phase 4 的对话缓冲，升级为企业级 Agent 应有的**四层模型**。
全部**离线确定性、零额外依赖（仅标准库）**，默认后端均为内存实现，生产可替换为
Redis / Postgres 而不改调用方接口。

四层定义
--------
1. **工作记忆 (Working)**：单次 Agent 运行期间的临时草稿——当前计划、工具调用留痕、
   反思、迭代次数。本质是 ``AgentState`` 的运行态，天然随一次请求生灭，不跨请求持久化。
2. **情景记忆 (Episodic)**：按 (租户, 用户, 会话) 组织的对话轮次。解决多轮指代
   （"它怎么申请"），并作为可审计的长期对话留痕。
3. **语义记忆 (Semantic)**：从交互中沉淀的"事实 / 偏好 / 已解决问答对"，按 key 存取，
   支持关键词检索。跨会话复用，让 Agent 记住"这个供应商的对账周期是 30 天"之类知识。
4. **程序记忆 (Procedural)**：可复用的流程模板（命名步骤序列）。把"常见工单处理 SOP"
   存为 procedure，下次直接作为起始计划召回，省去重复规划。

合规要点（见 compliance.py）
-------------------------
本模块只负责**存储与检索**；PII 脱敏、保留期清理、被遗忘权、数据导出由
``ComplianceManager`` 在写入/清理时统一施加，保证"合规"是架构内的强制闸而非可选项。

设计约束（与 M1~M3 一致，不可妥协）
--------------------------------
- 离线确定性：默认 ``ScriptedLLM`` + 内存后端，CI 下全程规则降级，无 flaky。
- 零额外依赖：纯 stdlib（dataclasses / hashlib / re / time），不引入新三方包。
- CI 常绿：``pytest`` / ``ruff`` / ``mypy`` 全绿。
"""
from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryLayer(str, Enum):
    """四层记忆的枚举，供指标标签 / 日志 / 清理使用。"""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"


# ── 数据载体 ──────────────────────────────────────────────────────────────


@dataclass
class Turn:
    """情景记忆中的一条对话轮次。"""

    tenant_id: str
    user_id: str
    session_id: str
    role: str  # "user" / "assistant"
    content: str
    ts: float
    content_hash: str  # 原文 hash：用于去重 / 关联，但不存明文


@dataclass
class Fact:
    """语义记忆中的一条事实。"""

    tenant_id: str
    user_id: str
    key: str
    value: str
    source: str
    ts: float
    value_hash: str


@dataclass
class Procedure:
    """程序记忆中的一条可复用流程模板。"""

    tenant_id: str
    user_id: str
    name: str
    steps: list[dict[str, Any]]
    ts: float


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── 工作记忆（单次运行态的轻量提取）──────────────────────────────────────


@dataclass
class WorkingMemory:
    """从 ``AgentState`` 抽取出的"当前工作集"，可反写回 state。

    工作记忆不跨请求持久化；它的价值在于让检查点 / 调试能一眼看到"这次运行做到哪了"。
    """

    plan: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reflection: str = ""
    iteration: int = 0
    pending_tool: str | None = None

    @classmethod
    def from_state(cls, state: dict) -> WorkingMemory:
        pa = state.get("pending_approval")
        return cls(
            plan=[p.model_dump() if hasattr(p, "model_dump") else p for p in (state.get("plan") or [])],
            tool_calls=[
                t.model_dump() if hasattr(t, "model_dump") else t for t in (state.get("tool_calls") or [])
            ],
            reflection=state.get("reflection") or "",
            iteration=int(state.get("iteration") or 0),
            pending_tool=pa.tool if pa else None,
        )

    def apply_to(self, state: dict) -> dict:
        """把工作记忆写回 state（用于从检查点恢复运行态）。"""
        state = dict(state)
        state["reflection"] = self.reflection
        state["iteration"] = self.iteration
        return state


# ── 后端接口（可插拔，默认内存实现）────────────────────────────────────


class EpisodicBackend:
    """情景记忆后端接口。"""

    def record(self, turn: Turn) -> None:  # pragma: no cover - 接口
        raise NotImplementedError

    def history(self, tenant: str, user: str, session: str, limit: int = 0) -> list[Turn]:
        raise NotImplementedError

    def recent_context(self, tenant: str, user: str, session: str, max_turns: int = 4, max_len: int = 160) -> str:
        raise NotImplementedError

    def all_for_identity(self, tenant: str, user: str) -> list[Turn]:
        raise NotImplementedError

    def forget(self, tenant: str, user: str) -> int:
        raise NotImplementedError

    def sweep(self, now: float, ttl: float) -> int:
        raise NotImplementedError


class InMemoryEpisodicBackend(EpisodicBackend):
    """进程内情景记忆：按 (租户, 用户, 会话) 分桶存对话轮次。"""

    def __init__(self) -> None:
        self._sessions: dict[tuple[str, str, str], list[Turn]] = {}

    def record(self, turn: Turn) -> None:
        self._sessions.setdefault((turn.tenant_id, turn.user_id, turn.session_id), []).append(turn)

    def history(self, tenant: str, user: str, session: str, limit: int = 0) -> list[Turn]:
        turns = list(self._sessions.get((tenant, user, session), []))
        if limit:
            turns = turns[-limit:]
        return turns

    def recent_context(self, tenant: str, user: str, session: str, max_turns: int = 4, max_len: int = 160) -> str:
        # 复用 Phase 4 的指代消解上下文提炼逻辑，保证多轮行为一致
        from app.agent.memory import build_coref_context

        history = [
            {"role": t.role, "content": t.content}
            for t in self.history(tenant, user, session, limit=max_turns)
        ]
        return build_coref_context(history, max_turns=max_turns, max_len=max_len)

    def all_for_identity(self, tenant: str, user: str) -> list[Turn]:
        out: list[Turn] = []
        for (t, u, _sid), turns in self._sessions.items():
            if t == tenant and u == user:
                out.extend(turns)
        return out

    def forget(self, tenant: str, user: str) -> int:
        removed = 0
        for key in [k for k in self._sessions if k[0] == tenant and k[1] == user]:
            removed += len(self._sessions.pop(key))
        return removed

    def sweep(self, now: float, ttl: float) -> int:
        removed = 0
        for turns in self._sessions.values():
            kept = [t for t in turns if now - t.ts <= ttl]
            removed += len(turns) - len(kept)
            turns[:] = kept
        return removed


class SemanticBackend:
    """语义记忆后端接口。"""

    def add_fact(self, fact: Fact) -> None:  # pragma: no cover - 接口
        raise NotImplementedError

    def get_fact(self, tenant: str, user: str, key: str) -> Fact | None:
        raise NotImplementedError

    def search(self, tenant: str, user: str, query: str, limit: int = 5) -> list[Fact]:
        raise NotImplementedError

    def all_for_identity(self, tenant: str, user: str) -> list[Fact]:
        raise NotImplementedError

    def forget(self, tenant: str, user: str) -> int:
        raise NotImplementedError

    def sweep(self, now: float, ttl: float) -> int:
        raise NotImplementedError


def _tokenize(text: str) -> set[str]:
    """确定性分词：英文/数字按词，中文按单字 + 二元组，便于中英混合检索。

    不引入 jieba 等依赖，用字符级 n-gram 做到零依赖下的可用召回。
    """
    text = (text or "").lower()
    tokens: set[str] = set(re.findall(r"[a-z0-9]+", text))
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    tokens.update(cjk)
    for i in range(len(cjk) - 1):
        tokens.add(cjk[i] + cjk[i + 1])
    return tokens


class InMemorySemanticBackend(SemanticBackend):
    """进程内语义记忆：按 (租户, 用户) 存 key→Fact。"""

    def __init__(self) -> None:
        self._facts: dict[tuple[str, str], dict[str, Fact]] = {}

    def add_fact(self, fact: Fact) -> None:
        self._facts.setdefault((fact.tenant_id, fact.user_id), {})[fact.key] = fact

    def get_fact(self, tenant: str, user: str, key: str) -> Fact | None:
        return self._facts.get((tenant, user), {}).get(key)

    def search(self, tenant: str, user: str, query: str, limit: int = 5) -> list[Fact]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scored: list[tuple[int, Fact]] = []
        for fact in self._facts.get((tenant, user), {}).values():
            blob = f"{fact.key} {fact.value} {fact.source}"
            f_tokens = _tokenize(blob)
            score = len(q_tokens & f_tokens)
            if score:
                scored.append((score, fact))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:limit]]

    def all_for_identity(self, tenant: str, user: str) -> list[Fact]:
        return list(self._facts.get((tenant, user), {}).values())

    def forget(self, tenant: str, user: str) -> int:
        store = self._facts.get((tenant, user))
        if not store:
            return 0
        n = len(store)
        del self._facts[(tenant, user)]
        return n

    def sweep(self, now: float, ttl: float) -> int:
        removed = 0
        for store in self._facts.values():
            keys = [k for k, f in store.items() if now - f.ts > ttl]
            for k in keys:
                del store[k]
            removed += len(keys)
        return removed


class ProceduralBackend:
    """程序记忆后端接口。"""

    def save_procedure(self, proc: Procedure) -> None:  # pragma: no cover - 接口
        raise NotImplementedError

    def get_procedure(self, tenant: str, user: str, name: str) -> list[dict[str, Any]] | None:
        raise NotImplementedError

    def list_procedures(self, tenant: str, user: str) -> list[str]:
        raise NotImplementedError

    def all_for_identity(self, tenant: str, user: str) -> list[Procedure]:
        raise NotImplementedError

    def forget(self, tenant: str, user: str) -> int:
        raise NotImplementedError

    def sweep(self, now: float, ttl: float) -> int:
        raise NotImplementedError


class InMemoryProceduralBackend(ProceduralBackend):
    """进程内程序记忆：按 (租户, 用户) 存 name→Procedure。"""

    def __init__(self) -> None:
        self._procs: dict[tuple[str, str], dict[str, Procedure]] = {}

    def save_procedure(self, proc: Procedure) -> None:
        self._procs.setdefault((proc.tenant_id, proc.user_id), {})[proc.name] = proc

    def get_procedure(self, tenant: str, user: str, name: str) -> list[dict[str, Any]] | None:
        proc = self._procs.get((tenant, user), {}).get(name)
        return list(proc.steps) if proc else None

    def list_procedures(self, tenant: str, user: str) -> list[str]:
        return sorted(self._procs.get((tenant, user), {}).keys())

    def all_for_identity(self, tenant: str, user: str) -> list[Procedure]:
        return list(self._procs.get((tenant, user), {}).values())

    def forget(self, tenant: str, user: str) -> int:
        store = self._procs.get((tenant, user))
        if not store:
            return 0
        n = len(store)
        del self._procs[(tenant, user)]
        return n

    def sweep(self, now: float, ttl: float) -> int:
        removed = 0
        for store in self._procs.values():
            keys = [k for k, p in store.items() if now - p.ts > ttl]
            for k in keys:
                del store[k]
            removed += len(keys)
        return removed


# ── 统一记忆管理器 ───────────────────────────────────────────────────────


class MemoryManager:
    """聚合四层记忆，向编排层 / API 暴露统一接口。

    后端可插拔（默认内存）；``ComplianceManager`` 继承此类，在写入/清理时施加合规闸。
    """

    def __init__(
        self,
        episodic: EpisodicBackend | None = None,
        semantic: SemanticBackend | None = None,
        procedural: ProceduralBackend | None = None,
    ) -> None:
        self.episodic = episodic or InMemoryEpisodicBackend()
        self.semantic = semantic or InMemorySemanticBackend()
        self.procedural = procedural or InMemoryProceduralBackend()

    # —— 情景记忆 ——
    def record_turn(self, tenant: str, user: str, session: str, role: str, content: str, now: float | None = None) -> None:
        self.episodic.record(
            Turn(
                tenant_id=tenant,
                user_id=user,
                session_id=session,
                role=role,
                content=content,
                ts=now if now is not None else time.time(),
                content_hash=_hash(content),
            )
        )

    def build_dialogue_context(self, tenant: str, user: str, session: str, max_turns: int = 4, max_len: int = 160) -> str:
        """为编排层生成多轮指代消解上下文（与 Phase 4 行为一致）。"""
        return self.episodic.recent_context(tenant, user, session, max_turns=max_turns, max_len=max_len)

    # —— 语义记忆 ——
    def add_fact(self, tenant: str, user: str, key: str, value: str, source: str = "", now: float | None = None) -> None:
        self.semantic.add_fact(
            Fact(
                tenant_id=tenant,
                user_id=user,
                key=key,
                value=value,
                source=source,
                ts=now if now is not None else time.time(),
                value_hash=_hash(value),
            )
        )

    def get_fact(self, tenant: str, user: str, key: str) -> Fact | None:
        return self.semantic.get_fact(tenant, user, key)

    def search_facts(self, tenant: str, user: str, query: str, limit: int = 5) -> list[Fact]:
        return self.semantic.search(tenant, user, query, limit=limit)

    # —— 程序记忆 ——
    def save_procedure(self, tenant: str, user: str, name: str, steps: list[dict[str, Any]], now: float | None = None) -> None:
        self.procedural.save_procedure(
            Procedure(
                tenant_id=tenant,
                user_id=user,
                name=name,
                steps=list(steps),
                ts=now if now is not None else time.time(),
            )
        )

    def get_procedure(self, tenant: str, user: str, name: str) -> list[dict[str, Any]] | None:
        return self.procedural.get_procedure(tenant, user, name)

    def list_procedures(self, tenant: str, user: str) -> list[str]:
        return self.procedural.list_procedures(tenant, user)

    # —— 工作记忆 ——
    @staticmethod
    def snapshot_working(state: dict) -> WorkingMemory:
        return WorkingMemory.from_state(state)

    def restore_working(self, state: dict, snap: WorkingMemory) -> dict:
        return snap.apply_to(state)

    # —— 合规运维 ——
    def forget_identity(self, tenant: str, user: str) -> dict[str, int]:
        """删除某身份在全部记忆层的全部数据（被遗忘权）。返回各层删除条数。"""
        return {
            MemoryLayer.EPISODIC.value: self.episodic.forget(tenant, user),
            MemoryLayer.SEMANTIC.value: self.semantic.forget(tenant, user),
            MemoryLayer.PROCEDURAL.value: self.procedural.forget(tenant, user),
        }

    def export_identity(self, tenant: str, user: str) -> dict[str, Any]:
        """导出某身份的全部记忆数据（数据可携 / DSAR）。"""
        return {
            MemoryLayer.EPISODIC.value: [
                {"session_id": t.session_id, "role": t.role, "content": t.content, "ts": t.ts}
                for t in self.episodic.all_for_identity(tenant, user)
            ],
            MemoryLayer.SEMANTIC.value: [
                {"key": f.key, "value": f.value, "source": f.source, "ts": f.ts}
                for f in self.semantic.all_for_identity(tenant, user)
            ],
            MemoryLayer.PROCEDURAL.value: [
                {"name": p.name, "steps": p.steps, "ts": p.ts}
                for p in self.procedural.all_for_identity(tenant, user)
            ],
        }

    def sweep(self, now: float | None = None, ttl_episodic: float = 0, ttl_semantic: float = 0, ttl_procedural: float = 0) -> dict[str, int]:
        """按各层保留期清理过期数据（保留期由合规层传入）。"""
        now = now if now is not None else time.time()
        return {
            MemoryLayer.EPISODIC.value: self.episodic.sweep(now, ttl_episodic),
            MemoryLayer.SEMANTIC.value: self.semantic.sweep(now, ttl_semantic),
            MemoryLayer.PROCEDURAL.value: self.procedural.sweep(now, ttl_procedural),
        }


# ── 进程级单例 ──────────────────────────────────────────────────────────

_manager: MemoryManager | None = None


def get_memory_manager() -> MemoryManager:
    """返回合规包装后的进程级记忆管理器单例（见 compliance.ComplianceManager）。"""
    global _manager
    if _manager is None:
        from app.agent.compliance import ComplianceManager

        _manager = ComplianceManager()
    return _manager


def reset_memory_manager() -> None:
    """清空记忆管理器（测试 fixture 用于隔离用例，防止跨用例污染）。"""
    global _manager
    _manager = None
