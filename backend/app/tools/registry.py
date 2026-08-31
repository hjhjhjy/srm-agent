"""工具注册中心：统一管理工具的注册、授权过滤、schema 导出与调用。

关键设计
--------
- **按 scopes 过滤可见工具**：LLM 只看得到调用方有权调用的工具，
  从源头减少越权调用的可能，而不是等调用后再报错。
- **调用前二次校验**：即使 LLM 幻觉出不存在的工具或越权工具，
  注册中心也会拦截并抛错，编排层据此降级。
"""
from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Optional

from app.tools.base import (
    ToolContext,
    ToolError,
    ToolNotPermitted,
    ToolNotFound,
    ToolSpec,
)

logger = logging.getLogger("srm.tools")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._tools:
            raise ValueError(f"工具重复注册: {spec.name}")
        if spec.fn is None:
            raise ValueError(f"工具 {spec.name} 未绑定实现")
        self._tools[spec.name] = spec
        logger.info(
            "注册工具 %s | side_effect=%s | scopes=%s | idempotent=%s",
            spec.name,
            spec.side_effect,
            sorted(spec.required_scopes),
            spec.idempotent,
        )
        return spec

    def tool(
        self,
        *,
        description: str,
        args_model: type,
        name: Optional[str] = None,
        required_scopes: tuple[str, ...] = (),
        side_effect: bool = False,
        idempotent: bool = True,
        timeout_ms: int = 10_000,
        max_retries: int = 2,
    ):
        """装饰器形式注册工具。

        示例::

            @registry.tool(
                description="查询采购订单",
                args_model=OrderQueryArgs,
                required_scopes=("order:read",),
            )
            async def order_query(args: OrderQueryArgs, ctx: ToolContext):
                ...
        """

        def deco(fn):
            spec = ToolSpec(
                name=name or fn.__name__,
                description=description,
                args_model=args_model,
                required_scopes=set(required_scopes),
                side_effect=side_effect,
                idempotent=idempotent,
                timeout_ms=timeout_ms,
                max_retries=max_retries,
                fn=fn,
            )
            self.register(spec)
            # 返回原函数而非 spec：保持模块级名称可直接调用，便于单测
            return fn

        return deco

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def all_tools(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def visible_to(self, scopes: list[str]) -> list[ToolSpec]:
        """按调用方权限过滤可见工具 —— 工具级授权的第一道闸。"""
        return [t for t in self._tools.values() if t.allows(scopes)]

    def schemas_for(self, scopes: list[str]) -> list[dict[str, Any]]:
        """导出给 LLM 的工具 schema（已按权限过滤）。"""
        return [t.json_schema() for t in self.visible_to(scopes)]

    async def invoke(
        self,
        name: str,
        args: dict[str, Any],
        ctx: ToolContext,
    ) -> Any:
        """调用工具：授权校验 → 参数校验 → 执行 → 统一异常。

        注意：这里**不**做重试与超时，那是编排层（Executor）的职责 ——
        重试需要配合预算与循环检测，放在工具层会绕过护栏。
        """
        spec = self._tools.get(name)
        if spec is None:
            raise ToolNotFound(f"未注册的工具: {name}")
        if not spec.allows(ctx.scopes):
            raise ToolNotPermitted(
                f"权限不足: 调用工具 {name} 需要 {sorted(spec.required_scopes)}"
            )

        validated = spec.args_model(**args)
        try:
            result = spec.fn(validated, ctx)
            if inspect.isawaitable(result):
                result = await result
            return result
        except ToolError:
            raise
        except Exception as exc:  # 统一包装，避免工具内部异常污染图状态
            raise ToolError(f"工具 {name} 执行失败: {exc}") from exc


registry = ToolRegistry()
