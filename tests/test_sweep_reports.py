"""Tests for sweep_reports.summarize_sweep and write_sweep_report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mech_interp.analysis.sweep_reports import SweepReport, summarize_sweep, write_sweep_report
from mech_interp.types import ExperimentResult, ExperimentSpec, RunStatus


def _make_spec(name: str, axis_values: dict[str, object]) -> ExperimentSpec:
    return ExperimentSpec(
        name=name,
        family="circuit_patching",
        backend="transformerlens",
        parameters={
            "matrix_axes": axis_values,
            "matrix": "base-spec",
        },
    )


def _make_result(run_id: int, metrics: dict[str, float]) -> ExperimentResult:
    return ExperimentResult(
        run_id=run_id,
        status=RunStatus.SUCCEEDED,
        metrics=metrics,
    )


class TestSummarizeSweep:
    def test_axis_recovery(self) -> None:
        specs = [
            _make_spec("base-aaa", {"n_features": 64, "k": 4}),
            _make_spec("base-bbb", {"n_features": 128, "k": 4}),
            _make_spec("base-ccc", {"n_features": 256, "k": 8}),
        ]
        results = [
            _make_result(1, {"loss": 0.5}),
            _make_result(2, {"loss": 0.4}),
            _make_result(3, {"loss": 0.3}),
        ]
        report = summarize_sweep(specs, results)

        assert "n_features" in report.axes
        assert "k" in report.axes
        assert report.axes["n_features"] == [64, 128, 256]
        assert set(report.axes["k"]) == {4, 8}

    def test_runs_count_matches_specs(self) -> None:
        specs = [
            _make_spec("base-aaa", {"seed": 1}),
            _make_spec("base-bbb", {"seed": 2}),
        ]
        results = [
            _make_result(1, {}),
            _make_result(2, {}),
        ]
        report = summarize_sweep(specs, results)
        assert len(report.runs) == 2

    def test_best_by_metric_max_and_min(self) -> None:
        specs = [
            _make_spec("base-aaa", {"lr": 0.001}),
            _make_spec("base-bbb", {"lr": 0.01}),
            _make_spec("base-ccc", {"lr": 0.1}),
        ]
        results = [
            _make_result(1, {"accuracy": 0.6}),
            _make_result(2, {"accuracy": 0.9}),
            _make_result(3, {"accuracy": 0.7}),
        ]
        report = summarize_sweep(specs, results)

        assert "accuracy_max" in report.best_by_metric
        assert "accuracy_min" in report.best_by_metric
        assert report.best_by_metric["accuracy_max"]["value"] == pytest.approx(0.9)
        assert report.best_by_metric["accuracy_max"]["axis_values"]["lr"] == 0.01
        assert report.best_by_metric["accuracy_min"]["value"] == pytest.approx(0.6)
        assert report.best_by_metric["accuracy_min"]["axis_values"]["lr"] == 0.001

    def test_no_results_gives_planned_status(self) -> None:
        specs = [_make_spec("base-aaa", {"seed": 1})]
        report = summarize_sweep(specs, [])
        assert report.runs[0]["status"] == "planned"

    def test_seed_axis_two_values(self) -> None:
        """Integration assertion: a seed sweep with 2 values yields 2 axis values."""
        specs = [
            _make_spec("smoke-aaa", {"seed": 1}),
            _make_spec("smoke-bbb", {"seed": 2}),
        ]
        results = [
            _make_result(10, {"reconstruction_mse": 0.05}),
            _make_result(11, {"reconstruction_mse": 0.04}),
        ]
        report = summarize_sweep(specs, results)
        assert len(report.axes["seed"]) == 2
        assert set(report.axes["seed"]) == {1, 2}

    def test_list_axis_values_preserved(self) -> None:
        """Axis values that are lists (e.g. layers=[0]) should round-trip."""
        specs = [
            _make_spec("patch-aaa", {"layers": [0]}),
            _make_spec("patch-bbb", {"layers": [1]}),
        ]
        results = [_make_result(1, {}), _make_result(2, {})]
        report = summarize_sweep(specs, results)
        assert [0] in report.axes["layers"]
        assert [1] in report.axes["layers"]

    def test_multiple_metrics_all_tracked(self) -> None:
        specs = [
            _make_spec("base-aaa", {"lr": 0.001}),
            _make_spec("base-bbb", {"lr": 0.01}),
        ]
        results = [
            _make_result(1, {"loss": 0.5, "sparsity": 0.8}),
            _make_result(2, {"loss": 0.3, "sparsity": 0.6}),
        ]
        report = summarize_sweep(specs, results)
        assert "loss_max" in report.best_by_metric
        assert "loss_min" in report.best_by_metric
        assert "sparsity_max" in report.best_by_metric
        assert "sparsity_min" in report.best_by_metric


class TestWriteSweepReport:
    def test_writes_json_and_md(self, tmp_path: Path) -> None:
        specs = [_make_spec("base-aaa", {"n": 64})]
        results = [_make_result(1, {"loss": 0.5})]
        report = summarize_sweep(specs, results)
        json_path, md_path = write_sweep_report(report, tmp_path)

        assert json_path.exists()
        assert md_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "axes" in data
        assert "runs" in data
        assert "best_by_metric" in data

    def test_markdown_contains_axis_names(self, tmp_path: Path) -> None:
        specs = [_make_spec("base-aaa", {"n_features": 128})]
        results = [_make_result(1, {})]
        report = summarize_sweep(specs, results)
        _, md_path = write_sweep_report(report, tmp_path)
        md = md_path.read_text(encoding="utf-8")
        assert "n_features" in md

    def test_to_dict_round_trips(self) -> None:
        report = SweepReport(
            axes={"k": [4, 8]},
            runs=[{"name": "x", "status": "succeeded", "axis_values": {"k": 4}, "metrics": {}}],
            best_by_metric={},
        )
        d = report.to_dict()
        assert d["axes"]["k"] == [4, 8]
