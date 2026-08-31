"""AgentState：LangGraph 图的全局状态契约。

设计要点
--------
1. **全字段可 JSON 序列化** —— 支持检查点持久化、断点续跑、线上问题轨迹回放。
2. `messages` 采用 `add_messages` 归约器，天然支持多轮对话追加而非覆盖。
3. **治理字段**（`budget` / `audit` / `pending_approval`）与业务字段分离，
   便于编排层统一拦截，而不必散落到各业务节点里。
"""
from __future__ import annotations

import hashlib
import json
import time
from typing import Annotated, Any, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

try:  # langgraph 各版本导出位置不一致，做兼容
    from langgraph.graph.message import add_messages
except ImportError:  # pragma: no cover
    from langgraph.graph import add_messages  # type: ignore[no-redef]


Intent = Literal["chitchat", "rag_qa", "tool_task", "human_handoff"]


class PlanStep(BaseModel):
    """规划器产出的单步计划。"""

    step_id: int
    description: str
    tool: Optional[str] = None
    args: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[int] = Field(default_factory=list)


class ToolCallRecord(BaseModel):
    """一次工具调用的完整留痕 —— 审计与可观测的单一数据源。"""

    step_id: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = False
    result: Any = None
    error: Optional[str] = None
    latency_ms: int = 0
    requires_approval: bool = False
    approved: Optional[bool] = None
    idempotency_key: Optional[str] = None


class Budget(BaseModel):
    """护栏：三预算 + 循环检测。

    任一超限即强制收敛，防止 Agent 失控。这是"能跑的 Demo"与"敢上生产"的分水岭：
    LLM 可能规划出自我循环的步骤，没有预算约束就会无限烧钱。
    """

    max_steps: int = 6
    steps_used: int = 0
    max_tokens: int = 8000
    tokens_used: int = 0
    max_wall_clock_ms: int = 30_000
    started_at: float = Field(default_factory=time.time)
    # 循环检测：(工具, 参数指纹) -> 调用次数
    call_fingerprints: dict[str, int] = Field(default_factory=dict)
    max_repeat_calls: int = 3

    @property
    def elapsed_ms(self) -> int:
        return int((time.time() - self.started_at) * 1000)

    @property
    def exhausted_reason(self) -> Optional[str]:
        if self.steps_used >= self.max_steps:
            return "step_budget"
        if self.tokens_used >= self.max_tokens:
            return "token_budget"
        if self.elapsed_ms >= self.max_wall_clock_ms:
            return "wall_clock"
        return None

    @property
    def exhausted(self) -> bool:
        return self.exhausted_reason is not None

    @staticmethod
    def fingerprint(tool: str, args: dict[str, Any]) -> str:
        payload = json.dumps(
            {"t": tool, "a": args}, sort_keys=True, ensure_ascii=False, default=str
        )
        return hashlib.md5(payload.encode()).hexdigest()[:12]

    def is_repeating(self, tool: str, args: dict[str, Any]) -> bool:
        """同一工具+同一参数被调用超过阈值 → 判定为死循环。"""
        fp = self.fingerprint(tool, args)
        n = self.call_fingerprints.get(fp, 0) + 1
        self.call_fingerprints[fp] = n
        return n > self.max_repeat_calls

    def consume_step(self) -> None:
        self.steps_used += 1

    def consume_tokens(self, n: int) -> None:
        self.tokens_used += n


class AuditEntry(BaseModel):
    """审计留痕：谁在何时以何参数调用了何工具，结果如何。

    追加写、不可变。写操作（side_effect=True）必须留痕。
    """

    ts: float = Field(default_factory=time.time)
    tenant_id: str = ""
    user_id: str = ""
    session_id: str = ""
    action: str = ""
    tool: Optional[str] = None
    args: dict[str, Any] = Field(default_factory=dict)
    outcome: str = ""
    approved: Optional[bool] = None
    idempotency_key: Optional[str] = None


class PendingApproval(BaseModel):
    """待人工审批的写操作 —— HITL 审批门的数据载体。"""

    step_id: int
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    idempotency_key: str = ""


class AgentState(TypedDict):
    # ---- 输入 ----
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    session_id: str
    tenant_id: str
    user_id: str
    user_scopes: list[str]
    # ---- 路由 ----
    intent: str
    # ---- 规划 ----
    plan: list[PlanStep]
    # ---- 执行 ----
    tool_calls: list[ToolCallRecord]
    pending_approval: Optional[PendingApproval]
    # HITL 三态：None=待审批 / True=已批准 / False=已拒绝
    approval_decision: Optional[bool]
    # ---- 反思 ----
    iteration: int
    sufficient: bool
    reflection: str
    # ---- 治理 ----
    budget: Budget
    audit: list[AuditEntry]
    # ---- 输出 ----
    answer: str
    citations: list[dict[str, Any]]
    trace: list[dict[str, Any]]


def initial_state(
    question: str,
    *,
    session_id: str = "",
    tenant_id: str = "",
    user_id: str = "",
    user_scopes: Optional[list[str]] = None,
    budget: Optional[Budget] = None,
) -> AgentState:
    """构造初始状态。集中在这里是为了保证每个字段都有默认值，
    避免各调用方漏字段导致图运行时 KeyError。"""
    from langchain_core.messages import HumanMessage

    return AgentState(
        messages=[HumanMessage(content=question)],
        question=question,
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        user_scopes=user_scopes or [],
        intent="",
        plan=[],
        tool_calls=[],
        pending_approval=None,
        approval_decision=None,
        iteration=0,
        sufficient=False,
        reflection="",
        budget=budget or Budget(),
        audit=[],
        answer="",
        citations=[],
        trace=[],
    )
