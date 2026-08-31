"""审计落盘：线程安全的 append-only 审计存储。

M1 的审计只活在 `AgentState` + `MemorySaver` 里，进程重启即丢、不可导出、
跨轮被清零，无法满足「不可变、可追溯、可问责」的治理要求。

本模块提供进程级 append-only 存储，并支持导出。后续可替换为
数据库表 / 独立的审计日志流（如 OpenTelemetry logs / Kafka）。
"""
from __future__ import annotations

import threading
from typing import Any

from app.agent.state import AuditEntry


class AuditStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: list[AuditEntry] = []

    def append(self, entry: AuditEntry) -> None:
        with self._lock:
            self._entries.append(entry)

    def all(self) -> list[AuditEntry]:
        with self._lock:
            return list(self._entries)

    def export(self, limit: int = 0) -> list[dict[str, Any]]:
        with self._lock:
            entries = self._entries if not limit else self._entries[-limit:]
            return [e.model_dump() for e in entries]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


audit_store = AuditStore()
