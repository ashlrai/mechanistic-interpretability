"""Unit tests for the SAE cross-model feature comparison.

These tests use random-weight mini SAEs (n_features=8) and synthetic activations.
No real model is loaded; no network calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Availability gate
# ---------------------------------------------------------------------------

_TORCH_AVAILABLE = True
try:
    import torch  # noqa: F401
except ImportError:
    _TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TORCH_AVAILABLE, reason="torch not installed"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sae(input_dim: int = 16, n_features: int = 8, k: int = 2) -> Any:
    """Return a freshly initialised TopKSAE on CPU with random weights."""
    import torch

    from mech_interp.sae.model import TopKSAE

    torch.manual_seed(0)
    sae = TopKSAE(input_dim=input_dim, n_features=n_features, k=k)
    sae.eval()
    return sae


def _make_activations(n_tokens: int = 32, d_model: int = 16, seed: int = 42) -> Any:
    import torch

    torch.manual_seed(seed)
    return torch.randn(n_tokens, d_model)


# ---------------------------------------------------------------------------
# Tests: cosine similarity matrix
# ---------------------------------------------------------------------------


def test_cosine_similarity_matrix_diagonal_is_one_for_same_input() -> None:
    from mech_interp.experiments.sae_cross_model import (
        _cosine_similarity_matrix,
        _decoder_directions,
    )

    sae = _make_sae(input_dim=16, n_features=8)
    dirs = _decoder_directions(sae)  # already normalised
    sim = _cosine_similarity_matrix(dirs, dirs)

    # Diagonal should be 1.0 (self-similarity of normalised vectors)
    for i in range(sae.n_features):
        assert abs(float(sim[i, i].item()) - 1.0) < 1e-4, (
            f"Self-cosine for feature {i} is {sim[i, i].item():.6f}, expected ~1.0"
        )


def test_cosine_similarity_matrix_shape() -> None:
    from mech_interp.experiments.sae_cross_model import (
        _cosine_similarity_matrix,
        _decoder_directions,
    )

    src_sae = _make_sae(input_dim=16, n_features=8)
    tgt_sae = _make_sae(input_dim=16, n_features=6)
    src_dirs = _decoder_directions(src_sae)
    tgt_dirs = _decoder_directions(tgt_sae)
    sim = _cosine_similarity_matrix(src_dirs, tgt_dirs)
    assert tuple(sim.shape) == (8, 6)


def test_cosine_similarity_values_in_range() -> None:
    from mech_interp.experiments.sae_cross_model import (
        _cosine_similarity_matrix,
        _decoder_directions,
    )

    src_sae = _make_sae(input_dim=16, n_features=8)
    tgt_sae = _make_sae(input_dim=16, n_features=8)
    src_dirs = _decoder_directions(src_sae)
    tgt_dirs = _decoder_directions(tgt_sae)
    sim = _cosine_similarity_matrix(src_dirs, tgt_dirs)

    assert float(sim.min().item()) >= -1.01
    assert float(sim.max().item()) <= 1.01


# ---------------------------------------------------------------------------
# Tests: decoder_directions normalisation
# ---------------------------------------------------------------------------


def test_decoder_directions_are_unit_vectors() -> None:
    from mech_interp.experiments.sae_cross_model import _decoder_directions

    sae = _make_sae(input_dim=16, n_features=8)
    dirs = _decoder_directions(sae)
    norms = dirs.norm(dim=1)
    for i in range(sae.n_features):
        assert abs(float(norms[i].item()) - 1.0) < 1e-5, (
            f"Feature {i} norm is {norms[i].item():.6f}, expected 1.0"
        )


# ---------------------------------------------------------------------------
# Tests: greedy bipartite matching
# ---------------------------------------------------------------------------


def test_greedy_match_returns_correct_count() -> None:
    import torch

    from mech_interp.experiments.sae_cross_model import _greedy_bipartite_match

    sim = torch.rand(8, 8)
    matched = _greedy_bipartite_match(sim)
    assert len(matched) == 8


def test_greedy_match_no_duplicate_indices() -> None:
    import torch

    from mech_interp.experiments.sae_cross_model import _greedy_bipartite_match

    sim = torch.rand(8, 6)
    matched = _greedy_bipartite_match(sim)
    src_indices = [pair[0] for pair in matched]
    tgt_indices = [pair[1] for pair in matched]
    assert len(src_indices) == len(set(src_indices)), "Duplicate source indices in matching"
    assert len(tgt_indices) == len(set(tgt_indices)), "Duplicate target indices in matching"


def test_greedy_match_identity_matrix_returns_diagonal() -> None:
    """A perfect identity similarity matrix should match feature i to i."""
    import torch

    from mech_interp.experiments.sae_cross_model import _greedy_bipartite_match

    n = 6
    sim = torch.eye(n)
    matched = _greedy_bipartite_match(sim)
    # Every matched pair should have src == tgt (diagonal match)
    for src, tgt in matched:
        assert src == tgt, f"Expected diagonal match, got ({src}, {tgt})"


def test_greedy_match_non_square_matrix() -> None:
    """When n_src > n_tgt, number of matches = n_tgt."""
    import torch

    from mech_interp.experiments.sae_cross_model import _greedy_bipartite_match

    sim = torch.rand(10, 5)
    matched = _greedy_bipartite_match(sim)
    assert len(matched) == 5


# ---------------------------------------------------------------------------
# Tests: _median helper
# ---------------------------------------------------------------------------


def test_median_odd_list() -> None:
    from mech_interp.experiments.sae_cross_model import _median

    assert _median([1.0, 3.0, 2.0]) == pytest.approx(2.0)


def test_median_even_list() -> None:
    from mech_interp.experiments.sae_cross_model import _median

    assert _median([1.0, 2.0, 3.0, 4.0]) == pytest.approx(2.5)


def test_median_single() -> None:
    from mech_interp.experiments.sae_cross_model import _median

    assert _median([7.5]) == pytest.approx(7.5)


def test_median_empty() -> None:
    from mech_interp.experiments.sae_cross_model import _median

    assert _median([]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Tests: full experiment with synthetic backends
# ---------------------------------------------------------------------------


class _SyntheticBackend:
    """Minimal fake backend that returns random activations for any hook site."""

    def __init__(self, d_model: int = 16, n_prompts_out: int = 4, seed: int = 0) -> None:
        self.d_model = d_model
        self.n_prompts_out = n_prompts_out
        self.seed = seed

    def capture_activations(
        self, prompts: list[str], hook_sites: list[str]
    ) -> dict[str, Any]:
        import torch

        torch.manual_seed(self.seed)
        n = min(len(prompts), self.n_prompts_out)
        result = {}
        for site in hook_sites:
            # (batch, seq=4, d_model)
            result[site] = torch.randn(n, 4, self.d_model)
        return result


def test_sae_cross_model_experiment_smoke(tmp_path: Path) -> None:
    """Full pipeline with two synthetic backends and tiny SAEs."""
    from mech_interp.experiments.sae_cross_model import SAECrossModelExperiment
    from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

    D_MODEL = 16
    src_backend = _SyntheticBackend(d_model=D_MODEL, seed=0)
    tgt_backend = _SyntheticBackend(d_model=D_MODEL, seed=1)

    spec = ExperimentSpec(
        name="test-sae-cross",
        family="sae_cross_model",
        backend="transformerlens",
        description="unit test",
        parameters={
            "source_model": "gpt2",
            "target_model": "gpt2",
            "hook_site": "blocks.0.hook_resid_pre",
            "n_features": 8,
            "k": 2,
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "seed": 42,
            "device": "cpu",
            "high_similarity_threshold": 0.5,
            "top_prompts_per_feature": 2,
            "prompts": [
                "The cat sat on the mat.",
                "Dogs are man's best friend.",
                "The quick brown fox.",
                "Hello world is a classic.",
            ],
        },
    )
    run = ExperimentRun(
        id=1,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )

    experiment = SAECrossModelExperiment(
        source_backend=src_backend,  # type: ignore[arg-type]
        target_backend=tgt_backend,  # type: ignore[arg-type]
    )
    result = experiment.run(spec, run)

    assert result.status == RunStatus.SUCCEEDED
    assert result.metrics["n_matched_pairs"] > 0

    # Artifacts must exist and parse
    matched_path = Path(result.artifacts["matched_features"])
    summary_path = Path(result.artifacts["match_summary"])
    assert matched_path.is_file()
    assert summary_path.is_file()

    matched = json.loads(matched_path.read_text())
    summary = json.loads(summary_path.read_text())

    # Each matched entry has required keys
    for entry in matched:
        assert "source_feature" in entry
        assert "target_feature" in entry
        assert "cosine" in entry
        assert -1.1 <= float(entry["cosine"]) <= 1.1

    # Matched features are sorted by cosine descending
    cosines = [float(e["cosine"]) for e in matched]
    assert cosines == sorted(cosines, reverse=True)

    # Summary has expected fields
    assert "median_cosine" in summary
    assert "n_matched_pairs" in summary
    assert "high_similarity_pairs" in summary


def test_sae_cross_model_d_model_mismatch_raises(tmp_path: Path) -> None:
    """Source d_model != target d_model must raise ValueError."""
    from mech_interp.experiments.sae_cross_model import SAECrossModelExperiment
    from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

    src_backend = _SyntheticBackend(d_model=16, seed=0)
    tgt_backend = _SyntheticBackend(d_model=32, seed=1)  # different d_model

    spec = ExperimentSpec(
        name="test-mismatch",
        family="sae_cross_model",
        backend="transformerlens",
        description="",
        parameters={
            "source_model": "gpt2",
            "target_model": "gpt2-medium",
            "hook_site": "blocks.0.hook_resid_pre",
            "n_features": 8,
            "k": 2,
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "seed": 42,
            "device": "cpu",
            "prompts": ["Hello world.", "Test prompt."],
        },
    )
    run = ExperimentRun(
        id=2,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )

    with pytest.raises(ValueError, match="d_model"):
        SAECrossModelExperiment(
            source_backend=src_backend,  # type: ignore[arg-type]
            target_backend=tgt_backend,  # type: ignore[arg-type]
        ).run(spec, run)


def test_sae_cross_model_k_exceeds_n_features_raises(tmp_path: Path) -> None:
    from mech_interp.experiments.sae_cross_model import SAECrossModelExperiment
    from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

    src_backend = _SyntheticBackend(d_model=16, seed=0)

    spec = ExperimentSpec(
        name="test-k-too-large",
        family="sae_cross_model",
        backend="transformerlens",
        description="",
        parameters={
            "source_model": "gpt2",
            "target_model": "gpt2",
            "hook_site": "blocks.0.hook_resid_pre",
            "n_features": 4,
            "k": 8,  # k > n_features → should raise
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "seed": 42,
            "device": "cpu",
            "prompts": ["Hello world."],
        },
    )
    run = ExperimentRun(
        id=3,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )

    with pytest.raises(ValueError, match="k="):
        SAECrossModelExperiment(
            source_backend=src_backend,  # type: ignore[arg-type]
            target_backend=src_backend,  # type: ignore[arg-type]
        ).run(spec, run)


# ---------------------------------------------------------------------------
# Tests: SAECrossModelProposalGenerator
# ---------------------------------------------------------------------------


def test_proposal_generator_emits_circuit_patching_specs(tmp_path: Path) -> None:
    from mech_interp.orchestration.proposal_generators import SAECrossModelProposalGenerator

    # Write fake artifacts
    matched = [
        {
            "source_feature": 0,
            "target_feature": 1,
            "cosine": 0.92,
            "source_top_prompts": ["The cat sat.", "A dog ran.", "Birds fly."],
            "target_top_prompts": ["A cat rested.", "Dog barked.", "Fish swim."],
        },
        {
            "source_feature": 2,
            "target_feature": 3,
            "cosine": 0.85,
            "source_top_prompts": ["Paris is nice.", "Berlin is cold."],
            "target_top_prompts": ["Rome has ruins.", "Athens is old."],
        },
    ]
    summary = {
        "hook_site": "blocks.4.hook_resid_pre",
        "source_model": "gpt2",
        "high_similarity_threshold": 0.8,
        "n_matched_pairs": 2,
    }
    spec = {
        "parameters": {"source_model": "gpt2", "hook_site": "blocks.4.hook_resid_pre"},
    }

    (tmp_path / "matched_features.json").write_text(json.dumps(matched))
    (tmp_path / "match_summary.json").write_text(json.dumps(summary))
    (tmp_path / "spec.json").write_text(json.dumps(spec))

    generator = SAECrossModelProposalGenerator()
    proposals = generator.generate(tmp_path, limit=5)

    assert len(proposals) == 2
    for prop in proposals:
        assert prop["family"] == "circuit_patching"
        assert prop["backend"] == "transformerlens"
        assert "source_feature_pair" in prop["parameters"]
        assert prop["parameters"]["hook_sites"] == ["blocks.4.hook_resid_pre"]


def test_proposal_generator_returns_empty_on_missing_artifacts(tmp_path: Path) -> None:
    from mech_interp.orchestration.proposal_generators import SAECrossModelProposalGenerator

    generator = SAECrossModelProposalGenerator()
    proposals = generator.generate(tmp_path, limit=5)
    assert proposals == []


def test_proposal_generator_skips_pairs_with_too_few_prompts(tmp_path: Path) -> None:
    from mech_interp.orchestration.proposal_generators import SAECrossModelProposalGenerator

    matched = [
        {
            "source_feature": 0,
            "target_feature": 1,
            "cosine": 0.95,
            "source_top_prompts": ["Only one prompt."],  # < 2 prompts → skip
            "target_top_prompts": [],
        },
    ]
    summary = {
        "hook_site": "blocks.4.hook_resid_pre",
        "source_model": "gpt2",
        "high_similarity_threshold": 0.8,
    }
    spec: dict[str, Any] = {}

    (tmp_path / "matched_features.json").write_text(json.dumps(matched))
    (tmp_path / "match_summary.json").write_text(json.dumps(summary))
    (tmp_path / "spec.json").write_text(json.dumps(spec))

    generator = SAECrossModelProposalGenerator()
    proposals = generator.generate(tmp_path, limit=5)
    assert proposals == []
