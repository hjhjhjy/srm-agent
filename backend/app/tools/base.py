"""工具契约：所有工具必须声明治理所需的元数据。

为什么不用裸函数？
------------------
企业级场景下，编排层必须在**执行前**就知道：

- 这个工具要不要权限？（`required_scopes`）
- 它会不会改数据？（`side_effect` → 决定是否走 HITL 审批门）
- 重试安全吗？（`idempotent` + `idempotency_key`）
- 多久算超时？（`timeout_ms`）

裸函数无法回答这些问题，治理逻辑就只能散落进业务代码，最终失控。
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


class ToolError(Exception):
    """工具执行失败的统一基类。"""


class ToolNotFound(ToolError):
    pass


class ToolNotPermitted(ToolError):
    """调用方 scope 不足以调用该工具（工具级授权拦截）。"""


class ToolTimeout(ToolError):
    pass


@dataclass
class ToolContext:
    """工具执行时的运行时上下文。

    由编排层注入，工具**不可**自行从全局读取身份信息 ——
    否则越权与租户串数据只是时间问题。
    """

    tenant_id: str
    user_id: str
    scopes: list[str] = field(default_factory=list)
    session_id: str = ""
    idempotency_key: str = ""


@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    required_scopes: set[str] = field(default_factory=set)
    side_effect: bool = False
    idempotent: bool = True
    timeout_ms: int = 10_000
    max_retries: int = 2
    fn: Callable[..., Any] | None = None

    @property
    def requires_approval(self) -> bool:
        """有副作用的工具一律需人工审批（HITL）。

        这是 M1 就落地的硬规则：写操作默认不信任自动执行。
        """
        return self.side_effect

    def allows(self, scopes: list[str]) -> bool:
        """工具级授权：调用方 scopes 必须覆盖工具要求的全部 scope。"""
        return self.required_scopes.issubset(set(scopes))

    def json_schema(self) -> dict[str, Any]:
        """导出 OpenAI function calling 兼容 schema，供 LLM 使用。"""
        schema = self.args_model.model_json_schema()
        # 去掉 pydantic 自动生成的 title，减少 token 且避免 LLM 混淆
        schema.pop("title", None)
        return {
            "name": self.name,
            "description": self.description,
            "parameters": schema,
        }
