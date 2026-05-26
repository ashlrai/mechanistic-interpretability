"""Unit tests for the per-family proposal generators."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from mech_interp.experiments.registry import load_experiment_spec
from mech_interp.orchestration.proposal_generators import (
    PROPOSAL_GENERATORS,
    ACDCEdgeProposalGenerator,
    ACDCLiteProposalGenerator,
    PolysemanticitySAEProposalGenerator,
)
from mech_interp.orchestration.proposals import propose_from_run


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_polysemanticity_sae_generator_emits_circuit_patching_proposals(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "spec.json",
        {
            "name": "sae-run",
            "family": "polysemanticity_sae",
            "backend": "transformerlens",
            "parameters": {
                "model": "gpt2-small",
                "hook_site": "blocks.0.hook_resid_pre",
            },
        },
    )
    _write(tmp_path / "sae_weights.safetensors.json", {"n_features": 64, "k": 8})
    _write(
        tmp_path / "feature_analysis.json",
        {
            "n_features": 4,
            "features": [
                {
                    "feature_index": 0,
                    "dead": False,
                    "max_activation": 3.0,
                    "coherence_score": 0.6,
                    "top_prompts": [
                        {"prompt": "Paris is the capital of France."},
                        {"prompt": "Rome is the capital of Italy."},
                    ],
                },
                {
                    "feature_index": 1,
                    "dead": False,
                    "max_activation": 2.0,
                    "coherence_score": 0.5,
                    "top_prompts": [
                        {"prompt": "Dogs are loyal."},
                        {"prompt": "Cats are quiet."},
                    ],
                },
                {"feature_index": 2, "dead": True},
            ],
        },
    )
    proposals = PolysemanticitySAEProposalGenerator().generate(tmp_path, limit=10)
    assert len(proposals) == 2
    assert all(p["family"] == "circuit_patching" for p in proposals)
    # Top feature (highest max_activation) ranks first.
    assert proposals[0]["parameters"]["source_feature_index"] == 0
    assert proposals[0]["parameters"]["hook_sites"] == ["blocks.0.hook_resid_pre"]
    # Control site is the same-layer MLP.
    assert proposals[0]["parameters"]["control_hook_sites"] == ["blocks.0.mlp.hook_post"]


def test_acdc_lite_generator_emits_activation_capture_proposal(tmp_path: Path) -> None:
    _write(
        tmp_path / "spec.json",
        {
            "name": "acdc-run",
            "family": "acdc_lite",
            "backend": "transformerlens",
            "parameters": {
                "model": "gpt2-small",
                "prompt_pairs": [
                    {
                        "clean_prompt": "The capital of France is",
                        "corrupted_prompt": "The capital of Italy is",
                    }
                ],
            },
        },
    )
    _write(
        tmp_path / "circuit.json",
        {
            "model": "gpt2-small",
            "faithfulness": 0.93,
            "nodes": [
                {"node_id": "L0.H3", "layer": 0, "component": "attn", "head": 3,
                 "importance": 0.7, "pruned": False},
                {"node_id": "L1.MLP", "layer": 1, "component": "mlp", "head": None,
                 "importance": 0.5, "pruned": False},
                {"node_id": "L2.H0", "layer": 2, "component": "attn", "head": 0,
                 "importance": 0.001, "pruned": True},
            ],
        },
    )
    proposals = ACDCLiteProposalGenerator().generate(tmp_path, limit=10)
    assert len(proposals) == 1
    spec = proposals[0]
    # activation_capture is wired via runner=activation_capture under a registered family.
    assert spec["family"] == "polysemanticity"
    assert spec["parameters"]["runner"] == "activation_capture"
    assert spec["parameters"]["sites"] == [
        "blocks.0.attn.hook_z",
        "blocks.1.hook_mlp_out",
    ]
    assert spec["parameters"]["prompts"] == ["The capital of France is"]


def test_propose_from_run_writes_validated_specs_and_manifest(tmp_path: Path) -> None:
    # Reuse the SAE artifact layout from the unit test above.
    artifacts = tmp_path / "run"
    artifacts.mkdir()
    _write(
        artifacts / "spec.json",
        {
            "name": "sae",
            "family": "polysemanticity_sae",
            "backend": "transformerlens",
            "parameters": {
                "model": "gpt2-small",
                "hook_site": "blocks.0.hook_resid_pre",
            },
        },
    )
    _write(artifacts / "sae_weights.safetensors.json", {"n_features": 64, "k": 8})
    _write(
        artifacts / "feature_analysis.json",
        {
            "n_features": 1,
            "features": [
                {
                    "feature_index": 0,
                    "dead": False,
                    "max_activation": 1.0,
                    "coherence_score": 0.5,
                    "top_prompts": [
                        {"prompt": "The capital of France is Paris."},
                        {"prompt": "The capital of Italy is Rome."},
                    ],
                }
            ],
        },
    )

    result = propose_from_run(
        "polysemanticity_sae",
        artifacts,
        tmp_path / "proposed",
        limit=5,
    )
    assert len(result.spec_paths) == 1
    spec = load_experiment_spec(result.spec_paths[0])
    assert spec.family == "circuit_patching"
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["family"] == "polysemanticity_sae"
    assert manifest["proposal_count"] == 1


def test_propose_from_run_raises_for_unsupported_family(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(ValueError, match="No per-run proposal generator"):
        propose_from_run("nonexistent_family", tmp_path, tmp_path / "out")


def test_registry_covers_expected_families() -> None:
    assert set(PROPOSAL_GENERATORS) >= {
        "polysemanticity_sae",
        "acdc_lite",
        "acdc_edge",
        "refusal_direction",
        "sae_cross_model",
        "attribution_patching",
        "crosscoder",
    }


def test_acdc_edge_generator_emits_acdc_lite_followup(tmp_path: Path) -> None:
    spec_payload: dict[str, Any] = {
        "name": "acdc-edge-run",
        "family": "acdc_edge",
        "backend": "transformerlens",
        "parameters": {
            "model": "gpt2-small",
            "prompt_pairs": [
                {
                    "clean_prompt": "The capital of France is",
                    "corrupted_prompt": "The capital of Italy is",
                    "correct_token": " Paris",
                    "incorrect_token": " Rome",
                }
            ],
        },
    }
    edges_payload: dict[str, Any] = {
        "model": "gpt2-small",
        "faithfulness": 0.88,
        "edges": [
            {
                "edge_id": "L0.H3->L2.H5",
                "src_id": "L0.H3",
                "dst_id": "L2.H5",
                "src_layer": 0,
                "dst_layer": 2,
                "importance": 0.8,
                "pruned": False,
            },
            {
                "edge_id": "L1.MLP->L3.H0",
                "src_id": "L1.MLP",
                "dst_id": "L3.H0",
                "src_layer": 1,
                "dst_layer": 3,
                "importance": 0.5,
                "pruned": False,
            },
            {
                "edge_id": "L0.H0->L1.H0",
                "src_id": "L0.H0",
                "dst_id": "L1.H0",
                "src_layer": 0,
                "dst_layer": 1,
                "importance": 0.01,
                "pruned": True,
            },
        ],
    }
    (tmp_path / "spec.json").write_text(json.dumps(spec_payload), encoding="utf-8")
    (tmp_path / "edges.json").write_text(json.dumps(edges_payload), encoding="utf-8")

    proposals = ACDCEdgeProposalGenerator().generate(tmp_path, limit=5)
    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal["family"] == "acdc_lite"
    # Destination layers of surviving edges: 2, 3 (in rank order).
    assert proposal["parameters"]["layers"] == [2, 3]
    assert proposal["parameters"]["model"] == "gpt2-small"
    assert proposal["parameters"]["source_acdc_edge_faithfulness"] == pytest.approx(0.88)
