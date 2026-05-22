from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from mech_interp import cli
from mech_interp.config.loader import AppConfig, ProviderConfig
from mech_interp.providers import (
    LMStudioProvider,
    OllamaProvider,
    ProviderHealth,
    configured_providers,
)


class DummyResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class DummyAsyncClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.requests: list[str] = []

    async def __aenter__(self) -> DummyAsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def get(self, url: str) -> DummyResponse:
        self.requests.append(url)
        if url.endswith("/api/tags"):
            return DummyResponse({"models": [{"name": "llama3.1"}, {"name": "mistral"}]})
        return DummyResponse({"data": [{"id": "local-model"}, {"id": "qwen2.5"}]})


def test_ollama_lists_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)

    models = asyncio.run(OllamaProvider(base_url="http://ollama.test").list_models())

    assert models == ("llama3.1", "mistral")


def test_lm_studio_lists_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(httpx, "AsyncClient", DummyAsyncClient)

    models = asyncio.run(LMStudioProvider(base_url="http://lm-studio.test/v1").list_models())

    assert models == ("local-model", "qwen2.5")


def test_provider_health_reports_unreachable(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider(base_url="http://ollama.test")

    def raise_connect_error() -> tuple[str, ...]:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(provider, "list_models_sync", raise_connect_error)

    health = provider.health_sync()

    assert health.reachable is False
    assert health.models == ()
    assert health.error == "connection refused"


def test_configured_providers_uses_configured_endpoints() -> None:
    config = AppConfig(
        providers={
            "ollama": ProviderConfig(base_url="http://ollama.test", default_model="llama3.1"),
            "lm_studio": ProviderConfig(base_url="http://lm.test/v1", default_model="local-model"),
        }
    )

    providers = configured_providers(config)

    assert [provider.name for provider in providers] == ["ollama", "lm_studio"]
    assert [provider.base_url for provider in providers] == [
        "http://ollama.test",
        "http://lm.test/v1",
    ]


def test_providers_cli_reports_reachability(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyProvider:
        name = "ollama"

        def health_sync(self) -> ProviderHealth:
            return ProviderHealth(
                provider="ollama",
                base_url="http://ollama.test",
                reachable=True,
                models=("llama3.1",),
            )

    monkeypatch.setattr(cli, "load_config", lambda: AppConfig())
    monkeypatch.setattr(cli, "configured_providers", lambda config, timeout: (DummyProvider(),))

    result = CliRunner().invoke(cli.app, ["providers"])

    assert result.exit_code == 0
    assert "ollama" in result.output
    assert "http://ollama.test" in result.output
    assert "llama3.1" in result.output
