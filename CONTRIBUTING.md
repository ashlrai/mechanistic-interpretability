# Contributing

Thank you for your interest in contributing to the Mechanistic Interpretability Platform.

## Dev Environment Setup

```bash
# Install uv if not already present
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dev + interp dependencies
uv sync --group dev --extra interp
```

## Running the Gate

Before opening a PR, ensure the full local CI gate passes:

```bash
bash scripts/check.sh
```

This runs ruff, mypy, pytest (unit), `mech validate` (experiment manifest validation), and a strict mkdocs build. For the full integration suite (requires model weights):

```bash
RUN_INTEGRATION_TESTS=1 bash scripts/check.sh --full
```

## Commit Convention

Follow the existing log style: `<type>: <summary>` (imperative, lowercase summary).

Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`.

## Adding a New Experiment Family

Use `src/mech_interp/experiments/circuit_patching.py` as the reference implementation.

1. Add a new entry to the `ExperimentFamily` StrEnum in `src/mech_interp/experiments/families.py`.
2. Create your experiment class (subclassing `Experiment`) in a new module under `src/mech_interp/experiments/`.
3. Register the mapping in `src/mech_interp/orchestration/runner.py::experiment_for_spec`.
4. Write a YAML spec under `experiments/` and run `mech validate` to confirm it parses.
5. Add unit tests; integration tests go under `tests/integration/`.

## Adding a New Spec YAML

Drop a YAML file under `experiments/` following the schema of existing specs. Run `mech validate` to catch schema errors early.

## Investigations and Publications

Preliminary findings and audit write-ups live under `docs/investigations/`. Add a new `.md` there for any notable result you want to preserve.

## Filing Issues

Please use the provided issue templates (bug report, experiment request, validation request) — they ensure we have the context needed to act quickly.
