"""IOI validation integration test for acdc_edge.

Runs the acdc_edge experiment on a small IOI variant and validates that the
surviving edges contain nodes from at least 2 canonical IOI head groups as
identified by Wang et al., 2022:

  "Interpretability in the Wild: a Circuit for Indirect Object Identification
   in GPT-2 small", Wang et al., NeurIPS 2022.
   https://arxiv.org/abs/2211.00593

Canonical IOI head groups in GPT-2 small (layer.head notation):
  - Name-mover heads:    9.6, 9.9, 10.0    — directly output the IO token
  - S-inhibition heads:  7.3, 8.6          — inhibit subject token output
  - Backup name-movers:  10.7, 11.10       — secondary IO movers
  - Induction heads:     5.5, 5.8, 5.9    — copy past contexts
  - Duplicate token:     0.1, 3.0         — detect repeated names

This test uses small max_edges=80 and 2-3 prompt pairs so it runs in under 2 minutes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mech_interp.analysis.ioi_validation import compare_to_canonical_ioi
from mech_interp.experiments.acdc_edge import ACDCEdgeExperiment
from mech_interp.types import ExperimentRun, ExperimentSpec, RunStatus, utc_now

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Spec helpers
# ---------------------------------------------------------------------------


def _ioi_spec(*, max_edges: int = 80, layers: list[int] | None = None) -> ExperimentSpec:
    """Small IOI spec for fast test execution.

    Uses 2 IOI prompt pairs with the standard structure:
      "Then [S1] and [IO] went to [place]. [S1] gave [object] to" → IO token
    """
    return ExperimentSpec(
        name="e2e-acdc-edge-ioi",
        family="acdc_edge",
        backend="transformerlens",
        description="IOI validation test",
        parameters={
            "model": "gpt2-small",
            "prompt_pairs": [
                {
                    "id": "ioi-john-mary",
                    "clean_prompt": (
                        "Then John and Mary went to the store."
                        " John gave a bottle of milk to"
                    ),
                    "corrupted_prompt": (
                        "Then John and Mary went to the store."
                        " Mary gave a bottle of milk to"
                    ),
                    "correct_token": " Mary",
                    "incorrect_token": " John",
                },
                {
                    "id": "ioi-tom-alice",
                    "clean_prompt": (
                        "Then Tom and Alice went to the park."
                        " Tom gave a flower to"
                    ),
                    "corrupted_prompt": (
                        "Then Tom and Alice went to the park."
                        " Alice gave a flower to"
                    ),
                    "correct_token": " Alice",
                    "incorrect_token": " Tom",
                },
                {
                    "id": "ioi-peter-emma",
                    "clean_prompt": (
                        "Then Peter and Emma went to the library."
                        " Peter gave a book to"
                    ),
                    "corrupted_prompt": (
                        "Then Peter and Emma went to the library."
                        " Emma gave a book to"
                    ),
                    "correct_token": " Emma",
                    "incorrect_token": " Peter",
                },
            ],
            # Layers 7-11 cover all canonical IOI heads.
            "layers": layers or [7, 8, 9, 10, 11],
            "include_attention": True,
            "include_mlps": False,
            "max_edges": max_edges,
            "tau": 0.01,
            "max_iterations": 5,
            "ablation_type": "mean",
            "seed": 42,
            "device": "cpu",
        },
    )


def _make_run(spec: ExperimentSpec, tmp_path: Path) -> ExperimentRun:
    return ExperimentRun(
        id=1,
        spec_name=spec.name,
        family=spec.family,
        backend=spec.backend,
        status=RunStatus.RUNNING,
        artifact_dir=tmp_path,
        created_at=utc_now(),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_edges_json(result: Any) -> dict[str, Any]:
    edges_path = Path(result.artifacts["edges_json"])
    data: dict[str, Any] = json.loads(edges_path.read_text(encoding="utf-8"))
    return data


def _top_k_surviving_nodes(
    edges_data: dict[str, Any],
    k: int = 20,
) -> list[str]:
    """Return top-K node IDs appearing in surviving edges, ranked by edge importance."""
    surviving = [e for e in edges_data.get("edges", []) if not e.get("pruned", True)]
    surviving.sort(key=lambda e: e.get("importance", 0.0), reverse=True)
    seen: list[str] = []
    seen_set: set[str] = set()
    for edge in surviving[:k]:
        for field in ("src_id", "dst_id"):
            nid = edge.get(field, "")
            if nid and nid not in seen_set:
                seen_set.add(nid)
                seen.append(nid)
    return seen[:k]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_acdc_edge_ioi_runs_successfully(gpt2_backend: Any, tmp_path: Path) -> None:
    """acdc_edge completes on IOI prompts without errors."""
    spec = _ioi_spec(max_edges=80)
    result = ACDCEdgeExperiment(backend=gpt2_backend).run(spec, _make_run(spec, tmp_path))

    assert result.status == RunStatus.SUCCEEDED
    assert "edges_json" in result.artifacts
    assert 0.0 <= result.metrics["faithfulness"] <= 1.0


def test_acdc_edge_ioi_edges_json_is_well_formed(gpt2_backend: Any, tmp_path: Path) -> None:
    """edges.json artifact has required structure."""
    spec = _ioi_spec(max_edges=80)
    result = ACDCEdgeExperiment(backend=gpt2_backend).run(spec, _make_run(spec, tmp_path))

    edges_data = _parse_edges_json(result)
    assert edges_data["model"] == "gpt2-small"
    assert isinstance(edges_data["edges"], list)
    assert isinstance(edges_data.get("pruning_history"), list)

    required_edge_keys = {
        "edge_id", "src_id", "dst_id", "src_layer", "dst_layer", "importance", "pruned"
    }
    for edge in edges_data["edges"]:
        assert required_edge_keys.issubset(edge.keys())


def test_acdc_edge_ioi_canonical_head_groups_present(gpt2_backend: Any, tmp_path: Path) -> None:
    """Surviving circuit edges contain nodes from at least 2 canonical IOI head groups.

    Canonical groups per Wang et al. 2022:
      - name_mover:      9.6, 9.9, 10.0
      - s_inhibition:    7.3, 8.6
      - backup_name_mover: 10.7, 11.10
      - induction:       5.5, 5.8, 5.9  (not in layers 7-11, so may not appear)
      - duplicate_token: 0.1, 3.0       (not in layers 7-11, so may not appear)

    With layers=[7,8,9,10,11], the reachable canonical groups are:
      name_mover, s_inhibition, backup_name_mover.
    We require at least 2 of these 3 to have at least one surviving edge endpoint.
    """
    spec = _ioi_spec(max_edges=80)
    result = ACDCEdgeExperiment(backend=gpt2_backend).run(spec, _make_run(spec, tmp_path))

    edges_data = _parse_edges_json(result)
    validation = compare_to_canonical_ioi(edges_data)

    # Report top-5 surviving nodes for diagnostic visibility in CI output.
    top_nodes = _top_k_surviving_nodes(edges_data, k=5)
    print(f"\nTop-5 surviving IOI nodes: {top_nodes}")
    print(f"Groups hit: {validation['canonical_groups_hit']}")
    print(f"Faithfulness: {validation['faithfulness']:.4f}")
    print(f"Recall: {validation['recall']:.4f}")

    # Primary assertion: at least 2 canonical groups represented in circuit.
    # This validates that acdc_edge is finding circuit structure, not noise.
    # We check only the groups reachable in layers 7-11.
    reachable_groups = {"name_mover", "s_inhibition", "backup_name_mover"}
    reachable_hits = sum(
        1
        for g, hits in validation["canonical_groups_hit"].items()
        if g in reachable_groups and hits
    )
    assert reachable_hits >= 2, (
        f"Expected ≥2 canonical IOI head groups to be represented in surviving edges "
        f"(of the reachable groups: {reachable_groups}), "
        f"but only {reachable_hits} found.\n"
        f"Groups hit: {validation['canonical_groups_hit']}\n"
        f"Top surviving nodes: {top_nodes}\n"
        f"This may indicate the KL-weighted approximation is not finding IOI circuit "
        f"structure. Check tau ({spec.parameters['tau']}) and "
        f"max_edges ({spec.parameters['max_edges']})."
    )


def test_acdc_edge_ioi_faithfulness_above_threshold(gpt2_backend: Any, tmp_path: Path) -> None:
    """Circuit faithfulness > 0.5 on the held-out IOI eval prompt.

    Faithfulness of 0.5 means the pruned circuit reproduces at least half of
    the full model's logit difference.  With the KL-weighted approximation on
    IOI, we expect higher faithfulness since the surviving edges capture the
    core circuit heads.
    """
    spec = _ioi_spec(max_edges=80)
    result = ACDCEdgeExperiment(backend=gpt2_backend).run(spec, _make_run(spec, tmp_path))

    faithfulness = result.metrics["faithfulness"]
    print(f"\nIOI faithfulness: {faithfulness:.4f}")

    assert faithfulness > 0.5, (
        f"Expected faithfulness > 0.5 on IOI task, got {faithfulness:.4f}. "
        f"The KL-weighted approximation may be over-pruning the circuit. "
        f"Surviving edges: {result.metrics.get('surviving_edge_count', '?')} / "
        f"{result.metrics.get('candidate_edge_count', '?')}"
    )


def test_acdc_edge_ioi_compare_to_canonical_returns_valid_structure(
    gpt2_backend: Any, tmp_path: Path
) -> None:
    """compare_to_canonical_ioi returns all expected keys with valid types."""
    spec = _ioi_spec(max_edges=80)
    result = ACDCEdgeExperiment(backend=gpt2_backend).run(spec, _make_run(spec, tmp_path))
    edges_data = _parse_edges_json(result)

    validation = compare_to_canonical_ioi(edges_data)

    assert isinstance(validation["surviving_nodes"], list)
    assert isinstance(validation["canonical_groups_hit"], dict)
    assert isinstance(validation["groups_with_any_hit"], int)
    assert isinstance(validation["total_canonical_heads"], int)
    assert 0.0 <= validation["recall"] <= 1.0
    assert 0.0 <= validation["precision"] <= 1.0
    assert 0.0 <= validation["faithfulness"] <= 1.0
    assert set(validation["canonical_groups_hit"].keys()) == {
        "name_mover",
        "s_inhibition",
        "backup_name_mover",
        "induction",
        "duplicate_token",
    }
