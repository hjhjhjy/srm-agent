"""内置工具集合。

导入本模块即完成全部工具注册（装饰器副作用式注册）。
编排层通过 `from app.tools.registry import registry` 取用。
"""
from app.tools.builtin import (
    calculator,
    kb_search,
    order_query,
    ticket_create,
)

__all__ = ["calculator", "kb_search", "order_query", "ticket_create"]
