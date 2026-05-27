"""Unit tests for src/mech_interp/analysis/refusal_audit.py.

Tests use a fully mocked store — no real database or model required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from mech_interp.analysis.refusal_audit import (
    RefusalAuditReport,
    _parse_hook_site,
    compile_refusal_audit,
)
from mech_interp.types import ExperimentResult, RunStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(
    run_id: int,
    metrics: dict[str, float],
    artifacts: dict[str, str] | None = None,
    notes: str = "",
) -> ExperimentResult:
    return ExperimentResult(
        run_id=run_id,
        status=RunStatus.SUCCEEDED,
        metrics=metrics,
        artifacts=artifacts or {},
        notes=notes,
    )


def _make_store(
    refusal_result: ExperimentResult,
    caa_result: ExperimentResult,
    circuit_result: ExperimentResult,
    scrub_result: ExperimentResult,
    model_name: str = "test-model",
) -> Any:
    store = MagicMock()
    store.get_result.side_effect = lambda rid: {
        refusal_result.run_id: refusal_result,
        caa_result.run_id: caa_result,
        circuit_result.run_id: circuit_result,
        scrub_result.run_id: scrub_result,
    }.get(rid)
    store.get_run_spec.return_value = {"parameters": {"model": model_name}}
    return store


# ---------------------------------------------------------------------------
# Tests: _parse_hook_site
# ---------------------------------------------------------------------------


def test_parse_hook_site_resid_post() -> None:
    layer, head = _parse_hook_site("blocks.10.hook_resid_post")
    assert layer == 10
    assert head == -1


def test_parse_hook_site_hook_z() -> None:
    layer, head = _parse_hook_site("blocks.9.attn.hook_z")
    assert layer == 9
    assert head == -1


def test_parse_hook_site_with_head_index() -> None:
    # Synthetic site that encodes a head in the name.
    layer, head = _parse_hook_site("blocks.10.attn.hook_result.head5")
    assert layer == 10
    assert head == 5


def test_parse_hook_site_unknown() -> None:
    layer, head = _parse_hook_site("unknown_site")
    assert layer == -1
    assert head == -1


# ---------------------------------------------------------------------------
# Tests: compile_refusal_audit — basic fields
# ---------------------------------------------------------------------------


def test_compile_refusal_audit_basic() -> None:
    refusal = _make_result(
        1,
        {"extraction_quality": 2.5, "baseline_refusal_rate": 1.0},
        notes="refusal note",
    )
    caa = _make_result(
        2,
        {"best_layer": 10.0, "best_refusal_rate_shift": 0.8},
        notes="caa note",
    )
    circuit = _make_result(3, {}, artifacts={}, notes="circuit note")
    scrub = _make_result(4, {"scrubbed_faithfulness": 0.75}, notes="scrub note")

    store = _make_store(refusal, caa, circuit, scrub, "Qwen/Qwen2.5-1.5B-Instruct")
    report = compile_refusal_audit(1, 2, 3, 4, store)

    assert report.model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert report.best_steering_layer == 10
    assert report.extraction_quality == pytest.approx(2.5)
    assert report.baseline_refusal_rate == pytest.approx(1.0)
    assert report.refusal_rate_shift_at_best == pytest.approx(0.8)
    assert report.circuit_faithfulness == pytest.approx(0.75)
    assert report.refusal_run_id == 1
    assert report.caa_run_id == 2
    assert report.circuit_run_id == 3
    assert report.scrub_run_id == 4
    assert "refusal note" in report.notes
    assert "caa note" in report.notes


def test_compile_refusal_audit_missing_run_raises() -> None:
    store = MagicMock()
    store.get_result.return_value = None
    with pytest.raises(ValueError, match="No result found for refusal run"):
        compile_refusal_audit(99, 2, 3, 4, store)


def test_compile_refusal_audit_missing_caa_raises() -> None:
    refusal = _make_result(1, {"extraction_quality": 1.0, "baseline_refusal_rate": 1.0})
    store = MagicMock()

    def _get_result(rid: int) -> ExperimentResult | None:
        if rid == 1:
            return refusal
        return None

    store.get_result.side_effect = _get_result
    store.get_run_spec.return_value = {"parameters": {"model": "m"}}
    with pytest.raises(ValueError, match="No result found for CAA"):
        compile_refusal_audit(1, 2, 3, 4, store)


# ---------------------------------------------------------------------------
# Tests: circuit patching head extraction with artifact
# ---------------------------------------------------------------------------


def test_compile_audit_with_circuit_artifact(tmp_path: Path) -> None:
    ranked = [
        {
            "rank": 1,
            "pair_id": "pair-0001",
            "hook_site": "blocks.10.attn.hook_z",
            "recovery_fraction": 0.85,
            "evidence_label": "causal evidence",
            "clean_logit_diff": 1.0,
            "corrupted_logit_diff": 0.1,
            "patched_logit_diff": 0.9,
            "activation_norm": None,
        },
        {
            "rank": 2,
            "pair_id": "pair-0001",
            "hook_site": "blocks.9.attn.hook_z",
            "recovery_fraction": 0.72,
            "evidence_label": "causal evidence",
            "clean_logit_diff": 1.0,
            "corrupted_logit_diff": 0.1,
            "patched_logit_diff": 0.8,
            "activation_norm": None,
        },
        {
            "rank": 3,
            "pair_id": "pair-0001",
            "hook_site": "blocks.10.hook_resid_post",
            "recovery_fraction": 0.65,
            "evidence_label": "causal evidence",
            "clean_logit_diff": 1.0,
            "corrupted_logit_diff": 0.1,
            "patched_logit_diff": 0.75,
            "activation_norm": None,
        },
    ]
    ranked_path = tmp_path / "patching_ranked_results.json"
    ranked_path.write_text(json.dumps(ranked), encoding="utf-8")

    refusal = _make_result(1, {"extraction_quality": 2.0, "baseline_refusal_rate": 1.0})
    caa = _make_result(2, {"best_layer": 10.0, "best_refusal_rate_shift": 0.7})
    circuit = _make_result(
        3, {}, artifacts={"patching_ranked_json": str(ranked_path)}
    )
    scrub = _make_result(4, {"scrubbed_faithfulness": 0.78})

    store = _make_store(refusal, caa, circuit, scrub)
    report = compile_refusal_audit(1, 2, 3, 4, store)

    # Deduplication by (layer, head): ranks 1+3 share (10, -1), rank 2 is (9, -1).
    # So only 2 unique (layer, head) pairs in top_n=3.
    assert len(report.top_causal_heads) == 2
    layers = [h[0] for h in report.top_causal_heads]
    assert 10 in layers
    assert 9 in layers
    # Recovery fractions are non-negative and the first is the highest
    assert report.top_causal_heads[0][2] >= report.top_causal_heads[1][2]


# ---------------------------------------------------------------------------
# Tests: RefusalAuditReport serialisation
# ---------------------------------------------------------------------------


def test_report_to_dict_roundtrip() -> None:
    report = RefusalAuditReport(
        model="test-model",
        best_steering_layer=10,
        best_coefficient=-3.0,
        refusal_rate_shift_at_best=0.8,
        top_causal_heads=[(10, -1, 0.85), (9, -1, 0.72), (10, -1, 0.65)],
        circuit_faithfulness=0.75,
        extraction_quality=2.5,
        baseline_refusal_rate=1.0,
        refusal_run_id=1,
        caa_run_id=2,
        circuit_run_id=3,
        scrub_run_id=4,
        notes="test notes",
    )
    d = report.to_dict()
    assert d["model"] == "test-model"
    assert d["best_steering_layer"] == 10
    assert d["circuit_faithfulness"] == pytest.approx(0.75)
    # top_causal_heads should be lists (not tuples) after to_dict
    assert isinstance(d["top_causal_heads"][0], list)


def test_report_to_json_is_valid_json() -> None:
    report = RefusalAuditReport(
        model="m",
        best_steering_layer=8,
        best_coefficient=-2.0,
        refusal_rate_shift_at_best=0.5,
        top_causal_heads=[],
        circuit_faithfulness=0.6,
        extraction_quality=1.2,
        baseline_refusal_rate=0.9,
        refusal_run_id=10,
        caa_run_id=11,
        circuit_run_id=12,
        scrub_run_id=13,
        notes="",
    )
    parsed = json.loads(report.to_json())
    assert parsed["model"] == "m"


def test_report_to_markdown_contains_headline() -> None:
    report = RefusalAuditReport(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        best_steering_layer=10,
        best_coefficient=-3.0,
        refusal_rate_shift_at_best=0.8,
        top_causal_heads=[(10, -1, 0.85)],
        circuit_faithfulness=0.75,
        extraction_quality=2.5,
        baseline_refusal_rate=1.0,
        refusal_run_id=1,
        caa_run_id=2,
        circuit_run_id=3,
        scrub_run_id=4,
        notes="",
    )
    md = report.to_markdown()
    assert "Qwen/Qwen2.5-1.5B-Instruct" in md
    assert "layer 10" in md
    assert "SUPPORTED" in md


def test_report_to_markdown_partial_faithfulness() -> None:
    report = RefusalAuditReport(
        model="m",
        best_steering_layer=8,
        best_coefficient=-2.0,
        refusal_rate_shift_at_best=0.5,
        top_causal_heads=[],
        circuit_faithfulness=0.6,
        extraction_quality=1.2,
        baseline_refusal_rate=0.9,
        refusal_run_id=1,
        caa_run_id=2,
        circuit_run_id=3,
        scrub_run_id=4,
        notes="",
    )
    md = report.to_markdown()
    assert "PARTIAL" in md


def test_report_to_markdown_rejected_faithfulness() -> None:
    report = RefusalAuditReport(
        model="m",
        best_steering_layer=8,
        best_coefficient=-2.0,
        refusal_rate_shift_at_best=0.5,
        top_causal_heads=[],
        circuit_faithfulness=0.3,
        extraction_quality=1.2,
        baseline_refusal_rate=0.9,
        refusal_run_id=1,
        caa_run_id=2,
        circuit_run_id=3,
        scrub_run_id=4,
        notes="",
    )
    md = report.to_markdown()
    assert "REJECTED" in md
