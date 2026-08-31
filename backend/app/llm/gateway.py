"""LLM 网关。

设计要点
--------
1. **可注入**：`set_llm()` 可替换为脚本化实现，让 Agent 测试**确定性可复现** ——
   否则 CI 里全是 flaky test，这是 LLM 应用工程化的第一道门槛。
2. **统一返回结构化结果**：`LLMResponse` 带 `tool_calls` 与 `tokens`，
   token 直接喂给预算护栏做成本控制。
3. **降级链**：主模型失败 → 备用模型（配置 `LLM_*_SECONDARY`）→ 仍失败则由编排层降级为检索直答。

M1 默认 `ScriptedLLM`（离线可跑）；配置 API Key 后自动切换 `OpenAICompatLLM`。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Protocol

from pydantic import BaseModel, Field

from app.observability import metrics

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
        tools: list[dict[str, Any]] | None = None,
        phase: str = "unknown",
    ) -> LLMResponse: ...


class ScriptedLLM:
    """按预置脚本返回响应 —— 单元测试与 CI 的确定性保障。"""

    def __init__(self, responses: list[LLMResponse | str] | None = None):
        self._queue: list[LLMResponse | str] = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    def push(self, resp: LLMResponse | str) -> None:
        self._queue.append(resp)

    async def achat(
        self, messages, tools=None, phase: str = "unknown"
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "tools": tools})
        t0 = time.time()
        if not self._queue:
            resp = LLMResponse(content="", model="scripted")
        else:
            item = self._queue.pop(0)
            if isinstance(item, str):
                resp = LLMResponse(content=item, tokens=max(1, len(item) // 2), model="scripted")
            else:
                resp = item
        metrics.record_llm(
            resp.model or "scripted", resp.tokens, time.time() - t0, error=False, phase=phase
        )
        return resp


class OpenAICompatLLM:
    """OpenAI 兼容接口（DeepSeek / 通义 / 智谱 / 本地 vLLM 均可）。

    未配置 API Key 时 `available=False`，编排层会自动降级，不会硬失败。
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        secondary_base_url: str | None = None,
        secondary_api_key: str | None = None,
        secondary_model: str | None = None,
        max_retries: int = 2,
        retry_base: float = 1.0,
    ):
        self.base_url = base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "")
        self.model = model or os.getenv("LLM_MODEL", "deepseek-chat")
        # 备用模型（可选）：主模型失败时的降级目标
        self.secondary_base_url = secondary_base_url or os.getenv("LLM_BASE_URL_SECONDARY", "")
        self.secondary_api_key = secondary_api_key or os.getenv("LLM_API_KEY_SECONDARY", "")
        self.secondary_model = secondary_model or os.getenv("LLM_MODEL_SECONDARY", "")
        # 重试策略：单次调用最多重试 max_retries 次，退避 = retry_base * 2**attempt
        self.max_retries = max_retries
        self.retry_base = retry_base

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    async def achat(
        self, messages, tools=None, phase: str = "unknown"
    ) -> LLMResponse:
        if not self.available:
            metrics.record_llm("unavailable", 0, 0.0, error=True, phase=phase)
            raise RuntimeError("未配置 LLM_API_KEY")
        t0 = time.time()
        try:
            resp = await self._call_with_retry(
                messages, tools, self.base_url, self.api_key, self.model, phase=phase
            )
        except Exception as exc:
            metrics.record_llm(self.model or "unknown", 0, time.time() - t0, error=True, phase=phase)
            logger.warning("主模型调用失败，尝试备用模型: %s", exc)
            if self.secondary_base_url and self.secondary_api_key and self.secondary_model:
                t1 = time.time()
                try:
                    resp = await self._call_with_retry(
                        messages,
                        tools,
                        self.secondary_base_url,
                        self.secondary_api_key,
                        self.secondary_model,
                        phase=phase,
                    )
                except Exception:
                    metrics.record_llm(
                        self.secondary_model, 0, time.time() - t1, error=True, phase=phase
                    )
                    raise
            else:
                raise
        metrics.record_llm(
            resp.model or self.model or "unknown", resp.tokens, time.time() - t0, error=False, phase=phase
        )
        return resp

    async def _call_with_retry(
        self, messages, tools, base_url, api_key, model, phase: str = "unknown"
    ) -> LLMResponse:
        """带指数退避重试的调用封装（仅用于真实网络调用）。"""
        return await _retry_async(
            lambda: self._call(messages, tools, base_url, api_key, model),
            max_retries=self.max_retries,
            retry_base=self.retry_base,
        )

    async def _call(self, messages, tools, base_url, api_key, model) -> LLMResponse:
        import httpx  # 延迟导入，避免离线环境强制依赖

        payload: dict[str, Any] = {"model": model, "messages": messages}
        if tools:
            payload["tools"] = [{"type": "function", "function": t} for t in tools]
        # 单调用超时 25s，低于 30s 墙钟预算，避免单次 LLM 调用就耗尽整个请求预算
        async with httpx.AsyncClient(base_url=base_url, timeout=25) as client:
            r = await client.post(
                "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {api_key}"},
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
            model=model,
        )


async def _retry_async(
    func,
    *,
    max_retries: int = 2,
    retry_base: float = 1.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """指数退避重试。

    对 ``func()``（异步可调用，无参数）执行最多 ``max_retries + 1`` 次尝试。
    遇到 ``exceptions`` 中声明的异常时，按 ``retry_base * 2**attempt`` 秒退避后
    重试；全部失败后抛出最后一次捕获的异常。成功则立即返回结果。
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            backoff = retry_base * (2 ** attempt)
            logger.warning("LLM 调用第 %d 次失败，%.2fs 后重试: %s", attempt + 1, backoff, exc)
            await asyncio.sleep(backoff)
    # last_exc 在此分支必不为 None（至少尝试了一次且捕获了异常）
    assert last_exc is not None
    raise last_exc


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
