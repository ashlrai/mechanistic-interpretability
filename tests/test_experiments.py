from mech_interp.experiments import load_experiment_specs


def test_load_experiment_specs() -> None:
    registry = load_experiment_specs("experiments")
    names = {spec.name for spec in registry.list()}

    assert "polysemanticity-smoke" in names
    assert "superposition-smoke" in names
    assert "circuit-patching-smoke" in names
