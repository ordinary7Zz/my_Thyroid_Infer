"""
OpenAI 兼容 LLM 客户端，支持重试与指数退避。

合并自两个 Agent 仓库中分散的 LLM 调用逻辑，统一为一处。
API key 通过构造函数参数传入（由 config 从环境变量读取），不硬编码。
"""

from __future__ import annotations

import time
from typing import Any, Optional

from openai import OpenAI, APIError, APITimeoutError, RateLimitError

BRIEF_ROUTING_REASONING = "-"


def build_llm_client(llm_cfg: dict, api_key: str) -> "LLMClient":
    """从 shared.agent_llm 配置构造客户端（含可选 enable_thinking / extra_body）。"""
    extra_body: dict[str, Any] = dict(llm_cfg.get("extra_body") or {})
    if llm_cfg.get("enable_thinking") is False:
        extra_body.setdefault("enable_thinking", False)
    return LLMClient(
        api_key=api_key,
        base_url=llm_cfg["base_url"],
        model_name=llm_cfg["model_name"],
        temperature=llm_cfg["temperature"],
        max_tokens=llm_cfg["max_tokens"],
        extra_body=extra_body if extra_body else None,
    )


class LLMClient:
    """OpenAI 兼容 API 客户端（支持 DashScope/Qwen 等）。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name: str = "qwen2.5-32b-instruct",
        temperature: float = 0.3,
        max_tokens: int = 1024,
        extra_body: Optional[dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._extra_body = extra_body
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
    ) -> str:
        """
        发送 chat completion 请求，返回模型文本响应。

        失败时按指数退避重试（1s, 2s, 4s, ...），最多 max_retries 次。
        重试覆盖的异常：APIError, APITimeoutError, RateLimitError。

        Args:
            system_prompt: 系统提示词。
            user_prompt: 用户提示词。
            max_retries: 最大重试次数。
            retry_base_delay: 首次重试延迟（秒），后续翻倍。

        Returns:
            模型响应文本 (choices[0].message.content)。

        Raises:
            RuntimeError: 所有重试均失败后抛出。
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if self._extra_body:
                    kwargs["extra_body"] = self._extra_body
                response = self._client.chat.completions.create(**kwargs)
                choice = response.choices[0]
                content = choice.message.content or ""
                finish = getattr(choice, "finish_reason", None)
                if finish == "length":
                    print(
                        f"  ⚠️ LLM 输出因 max_tokens={self.max_tokens} 被截断 "
                        f"(finish_reason=length)，可增大 shared.agent_llm.max_tokens",
                        flush=True,
                    )
                return content

            except (APIError, APITimeoutError, RateLimitError) as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue

        raise RuntimeError(
            f"LLM 调用失败（重试 {max_retries} 次后仍报错）: {last_error}"
        )
