from mech_interp.config import load_config


def test_load_default_config() -> None:
    config = load_config("configs/default.yaml")

    assert config.project.name == "local-mech-interp"
    assert config.backends.default_instrumented == "transformerlens"
    assert "ollama" in config.providers
