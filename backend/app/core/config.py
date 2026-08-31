"""运行时配置：从环境变量读取，绝不硬编码凭据。

M1 把 API Key 硬编码在代码里并在 README 公开（P0-5）。M2 改为：
- `SRM_JWT_SECRET` 设置后启用 JWT 鉴权（HS256，生产推荐）；
- 否则使用 `SRM_DEV_API_KEY`（由部署环境注入，不在代码里）；
- 两者都未设置时，启动期生成一个**仅本进程有效**的临时 dev key 并打印到日志，
  保证应用可本地运行，同时不把任何密钥写死在源码中。
"""
from __future__ import annotations

import os
import secrets
from typing import Any


def _get_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [s.strip() for s in raw.split(",") if s.strip()]


JWT_SECRET: str = os.getenv("SRM_JWT_SECRET", "")
JWT_ISSUER: str = os.getenv("SRM_JWT_ISSUER", "srm-agent")

# 本地演示用 dev key（由环境变量注入；未设置则启动期临时生成）
DEV_API_KEY: str = os.getenv("SRM_DEV_API_KEY", "")
if not DEV_API_KEY and not JWT_SECRET:
    DEV_API_KEY = secrets.token_hex(16)
    import logging

    logging.getLogger("srm.main").warning(
        "未配置 SRM_DEV_API_KEY / SRM_JWT_SECRET，已生成临时 dev key（仅本进程有效）: %s",
        DEV_API_KEY,
    )

DEV_TENANT_ID: str = os.getenv("SRM_DEV_TENANT_ID", "qlk")
DEV_USER_ID: str = os.getenv("SRM_DEV_USER_ID", "SUP001")
DEV_SCOPES: list[str] = _get_list(
    "SRM_DEV_SCOPES", "kb:read,order:read,ticket:write,calc:use,approval:review"
)

RATE_LIMIT_PER_MIN: int = int(os.getenv("SRM_RATE_LIMIT_PER_MIN", "60"))


def dev_identity() -> dict[str, Any]:
    """环境变量注入的本地演示身份（dev key 映射到该身份）。"""
    return {
        "tenant_id": DEV_TENANT_ID,
        "user_id": DEV_USER_ID,
        "scopes": list(DEV_SCOPES),
    }
