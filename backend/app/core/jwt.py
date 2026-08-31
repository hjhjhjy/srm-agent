"""极简 JWT（HS256）校验，仅依赖标准库，避免引入额外依赖。

用于生产环境的 Bearer 令牌鉴权：claims 至少包含
`tenant_id` / `user_id` / `scopes`，并带 `iss` 与 `exp`。
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any


def _b64decode(seg: str) -> bytes:
    seg += "=" * (-len(seg) % 4)
    return base64.urlsafe_b64decode(seg)


def verify_jwt(token: str, secret: str, issuer: str = "srm-agent", max_skew: int = 60) -> dict[str, Any]:
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("malformed token")
    signing_input = f"{parts[0]}.{parts[1]}".encode("utf-8")
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    provided = _b64decode(parts[2])
    if not hmac.compare_digest(expected, provided):
        raise ValueError("bad signature")
    payload = json.loads(_b64decode(parts[1]))
    if payload.get("iss") != issuer:
        raise ValueError("bad issuer")
    if "exp" in payload and payload["exp"] < time.time() - max_skew:
        raise ValueError("token expired")
    return payload
