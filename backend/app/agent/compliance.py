"""合规记忆层（M4）。

把"合规"做成架构内的**强制闸**，而不是各业务代码里散落的 if。
``ComplianceManager`` 继承 ``MemoryManager``，在写入与清理时统一施加：

1. **PII 不落盘明文**：写入情景/语义/程序记忆前，对内容进行 PII 脱敏
   （复用 Phase 5 的 ``mask_pii``）。原文只在 ``*_hash`` 字段留指纹，用于去重 / 关联，
   绝不存储明文敏感信息。
2. **保留期（Retention）**：各层有不同的数据保留期（情景 30 天、语义 180 天、
   程序 365 天），到期由 ``sweep`` 自动清理——这是 GDPR「存储限制」原则的工程落地。
3. **被遗忘权（Right to be Forgotten）**：``forget_identity`` 删除某身份在全部分层
   的全部数据，并打点计数。这是对「用户要求删除其数据」的法定义务的支撑。
4. **数据可携 / 导出（DSAR）**：``export_identity`` 返回该身份的全部（已脱敏）记忆，
   供数据主体访问请求。
5. **访问审计**：所有合规操作（删除 / 导出 / 清理）追加到 `compliance_audit`，
   并打 Prometheus 指标，使"合规"可观测、可问责。

设计约束（与 M1~M3 一致）：离线确定性、零额外依赖、CI 常绿。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from app.agent.memory_layers import MemoryManager
from app.security.sanitize import mask_pii


@dataclass
class RetentionPolicy:
    """各记忆层的数据保留期（秒）。默认遵循"最短必要"原则。"""

    episodic_ttl_s: float = 30 * 24 * 3600  # 情景记忆：30 天
    semantic_ttl_s: float = 180 * 24 * 3600  # 语义记忆：180 天
    procedural_ttl_s: float = 365 * 24 * 3600  # 程序记忆：365 天


def _scrub_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """递归对程序步骤里的字符串值做 PII 脱敏（步骤参数可能含联系方式等）。"""
    out: list[dict[str, Any]] = []
    for step in steps:
        cleaned = dict(step)
        if isinstance(step.get("args"), dict):
            cleaned["args"] = _scrub_value(step["args"])
        out.append(cleaned)
    return out


def _scrub_value(v: Any) -> Any:
    if isinstance(v, str):
        return mask_pii(v)[0]
    if isinstance(v, dict):
        return {k: _scrub_value(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_scrub_value(x) for x in v]
    return v


class ComplianceManager(MemoryManager):
    """在 ``MemoryManager`` 之上施加合规闸的记忆管理器。"""

    def __init__(
        self,
        retention: RetentionPolicy | None = None,
        audit_log: list[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__()
        self.retention = retention or RetentionPolicy()
        self.compliance_audit: list[dict[str, Any]] = audit_log if audit_log is not None else []

    # —— 写入：PII 不落明文 ——
    def record_turn(self, tenant: str, user: str, session: str, role: str, content: str, now: float | None = None) -> None:
        masked, n = mask_pii(content)
        if n:
            from app.observability import metrics as _m

            _m.record_compliance_pii(n)
        super().record_turn(tenant, user, session, role, masked, now=now)

    def add_fact(self, tenant: str, user: str, key: str, value: str, source: str = "", now: float | None = None) -> None:
        masked, n = mask_pii(value)
        if n:
            from app.observability import metrics as _m

            _m.record_compliance_pii(n)
        super().add_fact(tenant, user, key, masked, source=source, now=now)

    def save_procedure(self, tenant: str, user: str, name: str, steps: list[dict[str, Any]], now: float | None = None) -> None:
        scrubbed = _scrub_steps(steps)
        # 步骤描述也可能含 PII
        for s in scrubbed:
            if isinstance(s.get("description"), str):
                s["description"], n = mask_pii(s["description"])
                if n:
                    from app.observability import metrics as _m

                    _m.record_compliance_pii(n)
        super().save_procedure(tenant, user, name, scrubbed, now=now)

    # —— 被遗忘权 ——
    def forget_identity(self, tenant: str, user: str) -> dict[str, int]:
        counts = super().forget_identity(tenant, user)
        self._audit("forget_identity", tenant, user, {"layers": counts})
        from app.observability import metrics as _m

        for layer, c in counts.items():
            _m.record_compliance_forget(layer, c)
        return counts

    # —— 数据导出（DSAR）——
    def export_identity(self, tenant: str, user: str) -> dict[str, Any]:
        self._audit("export_identity", tenant, user, {})
        from app.observability import metrics as _m

        _m.record_compliance_export()
        return super().export_identity(tenant, user)

    # —— 保留期清理 ——
    def sweep(
        self,
        now: float | None = None,
        ttl_episodic: float = 0,
        ttl_semantic: float = 0,
        ttl_procedural: float = 0,
    ) -> dict[str, int]:
        now = now if now is not None else time.time()
        # 保留期以合规策略为准，忽略调用方传入的 ttl 参数（保持接口兼容）
        counts = super().sweep(
            now,
            ttl_episodic=self.retention.episodic_ttl_s,
            ttl_semantic=self.retention.semantic_ttl_s,
            ttl_procedural=self.retention.procedural_ttl_s,
        )
        self._audit("sweep", "", "", {"removed": counts})
        from app.observability import metrics as _m

        for layer, c in counts.items():
            _m.record_compliance_retention(layer, c)
        return counts

    def _audit(self, action: str, tenant: str, user: str, detail: dict[str, Any]) -> None:
        self.compliance_audit.append(
            {"ts": time.time(), "action": action, "tenant_id": tenant, "user_id": user, "detail": detail}
        )
