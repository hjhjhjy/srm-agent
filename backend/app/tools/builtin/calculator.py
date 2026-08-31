"""计算工具（只读、纯函数）。

安全说明
--------
**不使用 `eval` / `exec`**。工具参数是 LLM 生成的，而 LLM 输出可能受检索内容里的
Prompt 注入影响 —— 用 `eval` 等于给攻击者开了一个 RCE 口子。

这里用 AST 白名单求值：只允许数字常量与加减乘除等运算，
禁止变量引用、属性访问、函数调用、下标访问。
"""
from __future__ import annotations

import ast
import operator

from pydantic import BaseModel, Field

from app.tools.base import ToolContext
from app.tools.registry import registry


class CalculatorArgs(BaseModel):
    expression: str = Field(
        ...,
        description="算术表达式，仅支持数字与 + - * / // % ** ( )，如 (128600 + 45200) * 0.13",
    )


_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        # 防止 9**9**9 这类算力耗尽攻击
        if isinstance(node.op, ast.Pow) and (abs(right) > 64 or abs(left) > 1e6):
            raise ValueError("幂运算超出安全范围")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    raise ValueError("表达式包含不允许的语法（仅支持数字与四则运算）")


def safe_eval(expression: str) -> float:
    tree = ast.parse(expression, mode="eval")
    return _eval(tree)


@registry.tool(
    description=(
        "执行算术计算。用于订单金额合计、税率/折扣计算、日期天数差等需要精确数值的场景。"
        "仅支持数字与 + - * / // % ** 和括号。"
    ),
    args_model=CalculatorArgs,
    required_scopes=("calc:use",),
)
async def calculator(args: CalculatorArgs, ctx: ToolContext) -> dict:
    try:
        value = safe_eval(args.expression)
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError) as exc:
        return {"ok": False, "expression": args.expression, "error": str(exc)}
    return {"ok": True, "expression": args.expression, "value": value}
