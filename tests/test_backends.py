from mech_interp.backends import TransformerLensBackend
from mech_interp.providers import LMStudioProvider, OllamaProvider


def test_generation_providers_construct() -> None:
    assert OllamaProvider().name == "ollama"
    assert LMStudioProvider().name == "lm_studio"


def test_transformerlens_backend_constructs_without_loading() -> None:
    backend = TransformerLensBackend(model_name="gpt2-small")

    assert backend.name == "transformerlens"
    assert backend.model_name == "gpt2-small"
