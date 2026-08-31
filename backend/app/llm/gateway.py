"""LLM 网关。

设计要点
--------
1. **可注入**：`set_llm()` 可替换为脚本化实现，让 Agent 测试**确定性可复现** ——
   否则 CI 里全是 flaky test，这是 LLM 应用工程化的第一道门槛。
2. **统一返回结构化结果**：`LLMResponse` 带 `tool_calls` 与 `tokens`，
   token 直接喂给预算护栏做成本控制。
3. **降级链**：主模型失败 → 备模型 → 兜底空响应（由编排层降级为检索直答）。

M1 默认 `ScriptedLLM`（离线可跑）；配置 API Key 后自动切换 `OpenAICompatLLM`。
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional, Protocol

from pydantic import BaseModel, Field

logger = logging.getLogger("srm.llm")


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tokens: int = 0
    model: str = "scripted"


class BaseLLM(Protocol):
    async def achat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> LLMResponse: ...


class ScriptedLLM:
    """按预置脚本返回响应 —— 单元测试与 CI 的确定性保障。"""

    def __init__(self, responses: Optional[list[LLMResponse | str]] = None):
        self._queue: list[LLMResponse | str] = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def push(self, resp: LLMResponse | str) -> None:
        self._queue.append(resp)

    async def achat(self, messages, tools=None) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        if not self._queue:
            return LLMResponse(content="", model="scripted")
        item = self._queue.pop(0)
        if isinstance(item, str):
            return LLMResponse(content=item, tokens=max(1, len(item) // 2), model="scripted")
        return item


class OpenAICompatLLM:
    """OpenAI 兼容接口（DeepSeek / 通义 / 智谱 / 本地 vLLM 均可）。

    未配置 API Key 时 `available=False`，编排层会自动降级，不会硬失败。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def achat(self, messages, tools=None) -> LLMResponse:
        if not self.available:
            raise RuntimeError("未配置 LLM_API_KEY")

        import httpx  # 延迟导入，避免离线环境强制依赖

        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = [
                {"type": "function", "function": t} for t in tools
            ]
        async with httpx.AsyncClient(base_url=self.base_url, timeout=60) as client:
            r = await client.post(
                "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            r.raise_for_status()
            data = r.json()

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        calls = [
            ToolCall(
                name=tc["function"]["name"],
                args=json.loads(tc["function"].get("arguments") or "{}"),
            )
            for tc in (msg.get("tool_calls") or [])
        ]
        usage = data.get("usage") or {}
        return LLMResponse(
            content=msg.get("content") or "",
            tool_calls=calls,
            tokens=int(usage.get("total_tokens") or 0),
            model=self.model,
        )


_llm: BaseLLM = ScriptedLLM()


def get_llm() -> BaseLLM:
    return _llm


def set_llm(llm: BaseLLM) -> None:
    """替换 LLM 实现（测试注入 / 生产切换）。"""
    global _llm
    _llm = llm


def use_real_llm_if_configured() -> bool:
    """若环境变量配置了 API Key，则切换到真实模型。返回是否切换成功。"""
    candidate = OpenAICompatLLM()
    if candidate.available:
        set_llm(candidate)
        logger.info("已启用真实 LLM: %s @ %s", candidate.model, candidate.base_url)
        return True
    logger.info("未检测到 LLM_API_KEY，使用离线脚本化 LLM（降级模式）")
    return False
