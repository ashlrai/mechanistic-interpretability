from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from mech_interp.config import AppConfig
from mech_interp.types import GenerationRequest, GenerationResponse


@dataclass(frozen=True)
class ProviderHealth:
    provider: str
    base_url: str
    reachable: bool
    models: tuple[str, ...] = ()
    error: str | None = None


class OllamaProvider:
    name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

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

    async def list_models(self) -> tuple[str, ...]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
        return self._parse_models(response.json())

    def list_models_sync(self) -> tuple[str, ...]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
        return self._parse_models(response.json())

    async def health(self) -> ProviderHealth:
        try:
            models = await self.list_models()
        except (httpx.HTTPError, ValueError) as exc:
            return ProviderHealth(
                provider=self.name,
                base_url=self.base_url,
                reachable=False,
                error=str(exc),
            )
        return ProviderHealth(
            provider=self.name,
            base_url=self.base_url,
            reachable=True,
            models=models,
        )

    def health_sync(self) -> ProviderHealth:
        try:
            models = self.list_models_sync()
        except (httpx.HTTPError, ValueError) as exc:
            return ProviderHealth(
                provider=self.name,
                base_url=self.base_url,
                reachable=False,
                error=str(exc),
            )
        return ProviderHealth(
            provider=self.name,
            base_url=self.base_url,
            reachable=True,
            models=models,
        )

    @staticmethod
    def _parse_models(raw: Any) -> tuple[str, ...]:
        if not isinstance(raw, dict):
            raise ValueError("Expected Ollama model response to be an object")

        models = raw.get("models", [])
        if not isinstance(models, list):
            raise ValueError("Expected Ollama models to be a list")

        names: list[str] = []
        for model in models:
            if isinstance(model, dict) and isinstance(model.get("name"), str):
                names.append(model["name"])
        return tuple(names)


class LMStudioProvider:
    name = "lm_studio"

    def __init__(self, base_url: str = "http://localhost:1234/v1", timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

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

    async def list_models(self) -> tuple[str, ...]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/models")
            response.raise_for_status()
        return self._parse_models(response.json())

    def list_models_sync(self) -> tuple[str, ...]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/models")
            response.raise_for_status()
        return self._parse_models(response.json())

    async def health(self) -> ProviderHealth:
        try:
            models = await self.list_models()
        except (httpx.HTTPError, ValueError) as exc:
            return ProviderHealth(
                provider=self.name,
                base_url=self.base_url,
                reachable=False,
                error=str(exc),
            )
        return ProviderHealth(
            provider=self.name,
            base_url=self.base_url,
            reachable=True,
            models=models,
        )

    def health_sync(self) -> ProviderHealth:
        try:
            models = self.list_models_sync()
        except (httpx.HTTPError, ValueError) as exc:
            return ProviderHealth(
                provider=self.name,
                base_url=self.base_url,
                reachable=False,
                error=str(exc),
            )
        return ProviderHealth(
            provider=self.name,
            base_url=self.base_url,
            reachable=True,
            models=models,
        )

    @staticmethod
    def _parse_models(raw: Any) -> tuple[str, ...]:
        if not isinstance(raw, dict):
            raise ValueError("Expected LM Studio model response to be an object")

        data = raw.get("data", [])
        if not isinstance(data, list):
            raise ValueError("Expected LM Studio model data to be a list")

        names: list[str] = []
        for model in data:
            if isinstance(model, dict) and isinstance(model.get("id"), str):
                names.append(model["id"])
        return tuple(names)


def configured_providers(
    config: AppConfig,
    timeout: float = 5.0,
) -> tuple[OllamaProvider | LMStudioProvider, ...]:
    providers: list[OllamaProvider | LMStudioProvider] = []

    ollama = config.providers.get(OllamaProvider.name)
    if ollama is not None:
        providers.append(OllamaProvider(base_url=ollama.base_url, timeout=timeout))

    lm_studio = config.providers.get(LMStudioProvider.name)
    if lm_studio is not None:
        providers.append(LMStudioProvider(base_url=lm_studio.base_url, timeout=timeout))

    return tuple(providers)
