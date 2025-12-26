# src/uap/llm_client.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import os


class LLMClient:
    def generate(
        self,
        prompt: str,
        n: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        raise NotImplementedError


@dataclass
class QianfanOpenAIClient(LLMClient):
    """
    Qianfan OpenAI-compatible client.
    Reads key from env: QIANFAN_API_KEY
    Default base_url: https://qianfan.baidubce.com/v2
    """
    model: str = "ernie-4.5-turbo-128k"
    base_url: str = "https://qianfan.baidubce.com/v2"

    def __post_init__(self):
        from openai import OpenAI  # lazy import

        api_key = os.getenv("QIANFAN_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing env var QIANFAN_API_KEY (do NOT hardcode it in repo).")

        self._client = OpenAI(api_key=api_key, base_url=self.base_url)

    def generate(
        self,
        prompt: str,
        n: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        messages = [{"role": "user", "content": prompt}]
        outs: List[str] = []

        # 有些实现对 n 支持不稳定；这里循环 n 次最稳
        for _ in range(max(1, n)):
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop=stop,
                )
            except Exception as e:
                raise RuntimeError(f"Qianfan request failed: {e}") from e

            content = (completion.choices[0].message.content or "").strip()
            if not content:
                raise RuntimeError("Qianfan returned empty content (check model name / quota / key).")
            outs.append(content)

        return outs


@dataclass
class VLLMChatClient(LLMClient):
    """vLLM backend. Lazy-load model in __post_init__ (not at import-time)."""
    model_path: str
    max_model_len: int = 2048
    gpu_memory_utilization: float = 0.90
    tensor_parallel_size: int = 1
    trust_remote_code: bool = True
    enable_prefix_caching: bool = True

    def __post_init__(self):
        from vllm import LLM, SamplingParams  # lazy import

        self._SamplingParams = SamplingParams
        self._engine = LLM(
            model=self.model_path,
            trust_remote_code=self.trust_remote_code,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            tensor_parallel_size=self.tensor_parallel_size,
            enable_prefix_caching=self.enable_prefix_caching,
        )

    def generate(
        self,
        prompt: str,
        n: int = 3,
        temperature: float = 0.2,
        max_tokens: int = 256,
        stop: Optional[List[str]] = None,
    ) -> List[str]:
        sp = self._SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            n=max(1, n),
            stop=stop or [],
        )
        conversation = [{"role": "user", "content": prompt}]
        outputs = self._engine.chat(conversation, sampling_params=sp, use_tqdm=False)
        return [o.text for o in outputs[0].outputs]
