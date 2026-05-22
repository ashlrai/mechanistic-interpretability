from __future__ import annotations

from typing import Any

import httpx

from mech_interp.types import GenerationRequest, GenerationResponse


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
            },
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
        raw = response.json()
        return GenerationResponse(
            text=str(raw.get("response", "")),
            provider=self.name,
            model=request.model,
            raw=raw,
        )


class LMStudioProvider:
    name = "lm_studio"

    def __init__(self, base_url: str = "http://localhost:1234/v1") -> None:
        self.base_url = base_url.rstrip("/")

    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
        raw = response.json()
        choices = raw.get("choices", [])
        text = ""
        if choices:
            text = str(choices[0].get("message", {}).get("content", ""))
        return GenerationResponse(text=text, provider=self.name, model=request.model, raw=raw)
