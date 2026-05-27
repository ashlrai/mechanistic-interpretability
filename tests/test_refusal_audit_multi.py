"""Unit tests for src/mech_interp/analysis/refusal_audit_multi.py.

All tests use synthetic in-memory data — no database or model required.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from mech_interp.analysis.refusal_audit_multi import (
    ModelAuditRow,
    MultiModelReport,
    compile_multi_report,
    load_audit_row,
    load_rows_from_dir,
    load_rows_from_slugs,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_row(
    slug: str = "test_model",
    model: str = "org/test-model",
    extraction_quality: float = 3.0,
    best_layer: int = 12,
    refusal_rate_shift: float = 0.1,
    faithfulness: float = 0.04,
    refusal_run_id: int = 1,
    caa_run_id: int = 2,
    circuit_run_id: int = 3,
    scrub_run_id: int = 4,
) -> ModelAuditRow:
    return ModelAuditRow(
        slug=slug,
        model=model,
        extraction_quality=extraction_quality,
        best_layer=best_layer,
        refusal_rate_shift=refusal_rate_shift,
        faithfulness=faithfulness,
        refusal_run_id=refusal_run_id,
        caa_run_id=caa_run_id,
        circuit_run_id=circuit_run_id,
        scrub_run_id=scrub_run_id,
    )


def _write_audit_json(directory: Path, slug: str, **kwargs: object) -> Path:
    """Write a minimal refusal_audit_<slug>.json into directory."""
    row = _make_row(slug=slug, **kwargs)  # type: ignore[arg-type]
    data = {
        "model": row.model,
        "extraction_quality": row.extraction_quality,
        "best_steering_layer": row.best_layer,
        "refusal_rate_shift_at_best": row.refusal_rate_shift,
        "circuit_faithfulness": row.faithfulness,
        "refusal_run_id": row.refusal_run_id,
        "caa_run_id": row.caa_run_id,
        "circuit_run_id": row.circuit_run_id,
        "scrub_run_id": row.scrub_run_id,
    }
    path = directory / f"refusal_audit_{slug}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# ModelAuditRow verdicts
# ---------------------------------------------------------------------------


class TestModelAuditRowVerdicts:
    def test_fails_when_extraction_quality_low(self) -> None:
        # extraction_quality < 1.0 → not direction_extractable → FAILS (not even DIRECTION-ONLY)
        row = _make_row(extraction_quality=0.5, refusal_rate_shift=0.1, faithfulness=0.04)
        assert row.abliteration_verdict == "FAILS"

    def test_direction_only_when_extractable_but_shift_low(self) -> None:
        row = _make_row(extraction_quality=2.0, refusal_rate_shift=0.1, faithfulness=0.04)
        assert row.abliteration_verdict == "DIRECTION-ONLY"

    def test_works_when_all_pass(self) -> None:
        row = _make_row(extraction_quality=2.0, refusal_rate_shift=0.5, faithfulness=0.75)
        assert row.abliteration_verdict == "WORKS"

    def test_direction_extractable_threshold(self) -> None:
        assert _make_row(extraction_quality=0.9).direction_extractable is False
        assert _make_row(extraction_quality=1.0).direction_extractable is True
        assert _make_row(extraction_quality=4.1).direction_extractable is True

    def test_steering_effective_threshold(self) -> None:
        assert _make_row(refusal_rate_shift=0.29).steering_effective is False
        assert _make_row(refusal_rate_shift=0.30).steering_effective is True

    def test_circuit_verdict_supported(self) -> None:
        assert _make_row(faithfulness=0.71).circuit_verdict == "SUPPORTED"

    def test_circuit_verdict_partial(self) -> None:
        assert _make_row(faithfulness=0.60).circuit_verdict == "PARTIAL"

    def test_circuit_verdict_rejected(self) -> None:
        assert _make_row(faithfulness=0.04).circuit_verdict == "REJECTED"

    def test_short_name_known_model(self) -> None:
        row = _make_row(model="Qwen/Qwen2.5-1.5B-Instruct")
        assert row.short_name == "Qwen2.5-1.5B-I"

    def test_short_name_unknown_model(self) -> None:
        row = _make_row(model="org/some-unknown-model-7b")
        assert row.short_name == "some-unknown-model-7b"


# ---------------------------------------------------------------------------
# MultiModelReport aggregate stats
# ---------------------------------------------------------------------------


class TestMultiModelReport:
    def _all_fail_report(self) -> MultiModelReport:
        rows = [
            _make_row(slug="m1", extraction_quality=4.0, refusal_rate_shift=0.05,
                      faithfulness=0.04),
            _make_row(slug="m2", extraction_quality=3.5, refusal_rate_shift=0.10,
                      faithfulness=0.06),
            _make_row(slug="m3", extraction_quality=2.1, refusal_rate_shift=0.08,
                      faithfulness=0.03),
        ]
        return compile_multi_report(rows)

    def _mixed_report(self) -> MultiModelReport:
        rows = [
            _make_row(slug="m1", extraction_quality=4.0, refusal_rate_shift=0.05,
                      faithfulness=0.04),
            _make_row(slug="m2", extraction_quality=2.0, refusal_rate_shift=0.55,
                      faithfulness=0.80),
        ]
        return compile_multi_report(rows)

    def test_all_fail(self) -> None:
        report = self._all_fail_report()
        assert report.all_fail is True
        assert report.n_fails + report.n_direction_only == 3
        assert report.n_works == 0

    def test_mixed(self) -> None:
        report = self._mixed_report()
        assert report.all_fail is False
        assert report.n_works == 1

    def test_mean_extraction_quality(self) -> None:
        rows = [
            _make_row(extraction_quality=2.0),
            _make_row(extraction_quality=4.0),
        ]
        report = compile_multi_report(rows)
        assert abs(report.mean_extraction_quality() - 3.0) < 1e-9

    def test_mean_faithfulness(self) -> None:
        rows = [
            _make_row(faithfulness=0.04),
            _make_row(faithfulness=0.08),
        ]
        report = compile_multi_report(rows)
        assert abs(report.mean_faithfulness() - 0.06) < 1e-9

    def test_empty_report(self) -> None:
        report = compile_multi_report([])
        assert report.n_models == 0
        assert report.mean_extraction_quality() == 0.0
        assert report.mean_faithfulness() == 0.0


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


class TestMarkdownRendering:
    def test_all_fail_headline_in_markdown(self) -> None:
        rows = [
            _make_row(slug="m1", extraction_quality=4.0, refusal_rate_shift=0.05,
                      faithfulness=0.04),
            _make_row(slug="m2", extraction_quality=3.5, refusal_rate_shift=0.10,
                      faithfulness=0.06),
        ]
        report = compile_multi_report(rows)
        md = report.to_markdown()
        assert "systematically fails" in md
        assert "## Comparison Table" in md
        assert "## Per-Model Caveats" in md
        assert "## Interpretation" in md

    def test_works_headline_in_markdown(self) -> None:
        rows = [
            _make_row(slug="m1", extraction_quality=2.0, refusal_rate_shift=0.5, faithfulness=0.80),
        ]
        report = compile_multi_report(rows)
        md = report.to_markdown()
        assert "WORKS" in md

    def test_table_contains_all_rows(self) -> None:
        rows = [_make_row(slug=f"m{i}", model=f"org/model-{i}") for i in range(3)]
        report = compile_multi_report(rows)
        md = report.to_markdown()
        for r in rows:
            assert r.short_name in md

    def test_run_reference_table_present(self) -> None:
        rows = [
            _make_row(slug="m1", refusal_run_id=70, caa_run_id=71,
                      circuit_run_id=72, scrub_run_id=73),
        ]
        report = compile_multi_report(rows)
        md = report.to_markdown()
        assert "70" in md
        assert "71" in md
        assert "## Run Reference" in md

    def test_json_serialisable(self) -> None:
        rows = [_make_row(slug="m1")]
        report = compile_multi_report(rows)
        data = json.loads(report.to_json())
        assert data["n_models"] == 1
        assert "rows" in data
        assert data["rows"][0]["slug"] == "m1"


# ---------------------------------------------------------------------------
# load_audit_row — file I/O
# ---------------------------------------------------------------------------


class TestLoadAuditRow:
    def test_round_trip(self, tmp_path: Path) -> None:
        path = _write_audit_json(
            tmp_path,
            "qwen25_1_5b",
            model="Qwen/Qwen2.5-1.5B-Instruct",
            extraction_quality=4.1,
            best_layer=10,
            refusal_rate_shift=0.05,
            faithfulness=0.04,
            refusal_run_id=70,
            caa_run_id=71,
            circuit_run_id=72,
            scrub_run_id=73,
        )
        row = load_audit_row(path)
        assert row.slug == "qwen25_1_5b"
        assert row.model == "Qwen/Qwen2.5-1.5B-Instruct"
        assert abs(row.extraction_quality - 4.1) < 1e-6
        assert row.best_layer == 10
        assert row.refusal_run_id == 70

    def test_missing_fields_default_to_zero(self, tmp_path: Path) -> None:
        path = tmp_path / "refusal_audit_sparse.json"
        path.write_text(json.dumps({"model": "org/m"}), encoding="utf-8")
        row = load_audit_row(path)
        assert row.extraction_quality == 0.0
        assert row.faithfulness == 0.0
        assert row.slug == "sparse"

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "refusal_audit_bad.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises((json.JSONDecodeError, ValueError)):
            load_audit_row(path)


# ---------------------------------------------------------------------------
# load_rows_from_dir / load_rows_from_slugs
# ---------------------------------------------------------------------------


class TestLoadRows:
    def test_load_rows_from_dir(self, tmp_path: Path) -> None:
        _write_audit_json(tmp_path, "model_a")
        _write_audit_json(tmp_path, "model_b")
        # Baseline file should be skipped
        baseline = tmp_path / "refusal_audit.json"
        baseline.write_text(json.dumps({"model": "baseline"}), encoding="utf-8")

        rows = load_rows_from_dir(tmp_path)
        slugs = {r.slug for r in rows}
        assert "model_a" in slugs
        assert "model_b" in slugs
        # baseline file (refusal_audit.json) excluded
        assert len(rows) == 2

    def test_load_rows_from_slugs(self, tmp_path: Path) -> None:
        _write_audit_json(tmp_path, "qwen2_0_5b")
        _write_audit_json(tmp_path, "qwen25_0_5b")

        rows = load_rows_from_slugs(["qwen2_0_5b"], tmp_path)
        assert len(rows) == 1
        assert rows[0].slug == "qwen2_0_5b"

    def test_missing_slug_skipped(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        rows = load_rows_from_slugs(["nonexistent_model"], tmp_path)
        assert rows == []
        captured = capsys.readouterr()
        assert "not found" in captured.out

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        rows = load_rows_from_dir(tmp_path)
        assert rows == []


# ---------------------------------------------------------------------------
# CLI smoke test (subprocess-free: call _main via monkeypatch)
# ---------------------------------------------------------------------------


class TestCLISmoke:
    def test_cli_writes_markdown(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_audit_json(tmp_path, "qwen2_0_5b", model="Qwen/Qwen2-0.5B-Instruct")
        _write_audit_json(tmp_path, "qwen25_0_5b", model="Qwen/Qwen2.5-0.5B-Instruct")

        out_path = tmp_path / "out.md"
        monkeypatch.setattr(
            "sys.argv",
            [
                "refusal_audit_multi",
                "--audit-dir", str(tmp_path),
                "--output", str(out_path),
            ],
        )
        from mech_interp.analysis.refusal_audit_multi import _main
        _main()

        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert "## Comparison Table" in content

    def test_cli_with_slugs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _write_audit_json(tmp_path, "qwen2_0_5b")
        out_path = tmp_path / "out.md"
        monkeypatch.setattr(
            "sys.argv",
            [
                "refusal_audit_multi",
                "--slugs", "qwen2_0_5b",
                "--audit-dir", str(tmp_path),
                "--output", str(out_path),
            ],
        )
        from mech_interp.analysis.refusal_audit_multi import _main
        _main()
        assert out_path.exists()

    def test_cli_no_files_exits_gracefully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out_path = tmp_path / "out.md"
        monkeypatch.setattr(
            "sys.argv",
            [
                "refusal_audit_multi",
                "--audit-dir", str(tmp_path),
                "--output", str(out_path),
            ],
        )
        from mech_interp.analysis.refusal_audit_multi import _main
        _main()
        captured = capsys.readouterr()
        assert "No audit JSON" in captured.out
        assert not out_path.exists()
