from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mech_interp.backends import create_instrumented_backend
from mech_interp.experiments.base import Experiment
from mech_interp.storage.artifacts import resolve_run_artifact_dir
from mech_interp.types import (
    CrossModelProbeRecord,
    CrossModelProbeRequest,
    CrossModelProbeResult,
    ExperimentResult,
    ExperimentRun,
    ExperimentSpec,
    InstrumentedModelBackend,
    RunStatus,
)

CORRELATIONAL_ALIGNMENT_LABEL = "correlational alignment"
HYPOTHESIS_LABEL = "hypothesis"


class CrossModelProbeRecordSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | None = None
    split: str
    prompt: str
    correct_token: str | None = None
    incorrect_token: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("split", "prompt")
    @classmethod
    def strip_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class ArtifactPolicy(BaseModel):
    model_config = ConfigDict(extra="allow")

    retain_probe_weights: bool = False
    write_report: bool = True
    activation_verbalization: bool = False
    max_verbalized_records: int = 5


class CrossModelProbeSpec(BaseModel):
    model_config = ConfigDict(extra="allow")

    source_model: str = "roneneldan/TinyStories-1M"
    source_model_name: str | None = None
    target_model: str = "roneneldan/TinyStories-3M"
    target_model_name: str | None = None
    source_hook_site: str
    target_hook_site: str
    records: list[CrossModelProbeRecordSpec] | None = None
    dataset_path: str | None = None
    dataset_sha256: str | None = None
    ridge_alpha: float = Field(default=1.0, ge=0.0)
    dtype: str = "float32"
    artifact_policy: ArtifactPolicy = Field(default_factory=ArtifactPolicy)

    @property
    def resolved_source_model_name(self) -> str:
        return self.source_model_name or self.source_model

    @property
    def resolved_target_model_name(self) -> str:
        return self.target_model_name or self.target_model


class CrossModelRepresentationProbeExperiment(Experiment):
    family = "cross_model_representation_probe"

    def __init__(self, backend: InstrumentedModelBackend | None = None) -> None:
        self.backend = backend

    def run(self, spec: ExperimentSpec, run: ExperimentRun) -> ExperimentResult:
        config = CrossModelProbeSpec.model_validate(spec.parameters)
        records, dataset_metadata = _resolve_records(config)
        _validate_splits(records)

        backend = self.backend or create_instrumented_backend(
            spec.backend,
            {
                "model_name": config.resolved_source_model_name,
                "device": spec.parameters.get("device", "auto"),
            },
        )
        request = CrossModelProbeRequest(
            source_model_name=config.resolved_source_model_name,
            target_model_name=config.resolved_target_model_name,
            records=tuple(records),
            source_hook_site=config.source_hook_site,
            target_hook_site=config.target_hook_site,
            ridge_alpha=config.ridge_alpha,
            dtype=config.dtype,
            retain_probe_weights=config.artifact_policy.retain_probe_weights,
            max_verbalized_records=(
                config.artifact_policy.max_verbalized_records
                if config.artifact_policy.activation_verbalization
                else 0
            ),
        )
        probe_results = backend.run_cross_model_probe(request)
        if not probe_results:
            raise ValueError("Cross-model representation probe produced no results.")

        artifact_dir = resolve_run_artifact_dir(run)
        artifact_dir.mkdir(parents=True, exist_ok=True)
        rows = [_result_row(result) for result in probe_results]
        sorted_rows = sorted(
            rows,
            key=lambda row: float(row["mean_cosine_similarity"]),
            reverse=True,
        )
        ranked_rows = [
            {"rank": rank, **row}
            for rank, row in enumerate(sorted_rows, start=1)
        ]

        ranked_json = artifact_dir / "cross_model_probe_results.json"
        ranked_csv = artifact_dir / "cross_model_probe_results.csv"
        summary_path = artifact_dir / "cross_model_probe_summary.json"
        report_path = artifact_dir / "research_note.md"
        verbalization_path = artifact_dir / "activation_verbalization.json"

        ranked_json.write_text(
            json.dumps(ranked_rows, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_results_csv(ranked_csv, ranked_rows)

        summary = {
            "source_model": config.resolved_source_model_name,
            "target_model": config.resolved_target_model_name,
            "source_hook_site": config.source_hook_site,
            "target_hook_site": config.target_hook_site,
            "record_count": len(records),
            "train_record_count": sum(record.split == "train" for record in records),
            "eval_record_count": sum(record.split == "eval" for record in records),
            "ridge_alpha": config.ridge_alpha,
            "dtype": config.dtype,
            "dataset": dataset_metadata,
            "artifact_policy": config.artifact_policy.model_dump(),
            "evidence_label": CORRELATIONAL_ALIGNMENT_LABEL,
            "top_results": ranked_rows[:10],
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        artifacts = {
            "cross_model_probe_summary": str(summary_path.resolve()),
            "cross_model_probe_results_json": str(ranked_json.resolve()),
            "cross_model_probe_results_csv": str(ranked_csv.resolve()),
        }

        if config.artifact_policy.activation_verbalization:
            verbalizations = _activation_verbalizations(records, config.artifact_policy)
            verbalization_path.write_text(
                json.dumps(verbalizations, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            artifacts["activation_verbalization"] = str(verbalization_path.resolve())

        if config.artifact_policy.write_report:
            report_path.write_text(
                _render_report(spec, summary, ranked_rows),
                encoding="utf-8",
            )
            artifacts["research_note"] = str(report_path.resolve())

        if config.artifact_policy.retain_probe_weights:
            weights_path = getattr(backend, "last_probe_weights_path", None)
            if isinstance(weights_path, str | Path) and Path(weights_path).is_file():
                artifacts["probe_weights"] = str(Path(weights_path).resolve())
            else:
                weights = getattr(backend, "last_probe_weights", None)
                if weights is not None:
                    weights_output = artifact_dir / "probe_weights.npz"
                    np.savez(
                        weights_output,
                        weights=np.asarray(weights),
                        metadata=np.array(
                            json.dumps(
                                {
                                    "source_hook_site": config.source_hook_site,
                                    "target_hook_site": config.target_hook_site,
                                    "ridge_alpha": config.ridge_alpha,
                                },
                                sort_keys=True,
                            )
                        ),
                    )
                    artifacts["probe_weights"] = str(weights_output.resolve())

        return ExperimentResult(
            run_id=run.id,
            status=RunStatus.SUCCEEDED,
            metrics=_metrics(ranked_rows),
            artifacts=artifacts,
            notes=_result_notes(ranked_rows),
        )


def _resolve_records(
    config: CrossModelProbeSpec,
) -> tuple[list[CrossModelProbeRecord], dict[str, Any] | None]:
    if config.records:
        return [
            _record_from_spec(index, record)
            for index, record in enumerate(config.records)
        ], None

    if config.dataset_path is None:
        raise ValueError("Cross-model probe requires records or dataset_path.")

    dataset_path = Path(config.dataset_path)
    content = dataset_path.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()
    if config.dataset_sha256 and config.dataset_sha256 != sha256:
        raise ValueError(
            "Cross-model probe dataset hash mismatch: "
            f"expected {config.dataset_sha256}, got {sha256}."
        )

    records: list[CrossModelProbeRecord] = []
    with dataset_path.open("r", encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Invalid cross-model probe record on line {line_number}: expected object."
                )
            records.append(
                _record_from_spec(
                    len(records),
                    CrossModelProbeRecordSpec.model_validate(raw),
                )
            )

    return records, {
        "path": str(dataset_path),
        "sha256": sha256,
        "record_count": len(records),
    }


def _record_from_spec(index: int, record: CrossModelProbeRecordSpec) -> CrossModelProbeRecord:
    return CrossModelProbeRecord(
        id=record.id or f"record-{index + 1:04d}",
        split=record.split,
        prompt=record.prompt,
        correct_token=record.correct_token,
        incorrect_token=record.incorrect_token,
        metadata=dict(record.metadata),
    )


def _validate_splits(records: list[CrossModelProbeRecord]) -> None:
    splits = {record.split for record in records}
    if "train" not in splits:
        raise ValueError("Cross-model probe requires at least one train record.")
    if "eval" not in splits:
        raise ValueError("Cross-model probe requires at least one eval record.")


def _result_row(result: CrossModelProbeResult) -> dict[str, Any]:
    return {**asdict(result), "evidence_label": CORRELATIONAL_ALIGNMENT_LABEL}


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    eval_rows = [row for row in rows if row["split"] == "eval"] or rows
    return {
        "probe_result_count": float(len(rows)),
        "eval_mean_cosine_similarity": _mean(eval_rows, "mean_cosine_similarity"),
        "eval_normalized_mse": _mean(eval_rows, "normalized_mse"),
        "eval_variance_explained": _mean(eval_rows, "variance_explained"),
    }


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "source_hook_site",
        "target_hook_site",
        "split",
        "record_count",
        "mean_cosine_similarity",
        "normalized_mse",
        "variance_explained",
        "mean_logit_diff_error",
        "evidence_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _activation_verbalizations(
    records: list[CrossModelProbeRecord],
    policy: ArtifactPolicy,
) -> list[dict[str, Any]]:
    selected = records[: max(policy.max_verbalized_records, 0)]
    return [
        {
            "record_id": record.id,
            "split": record.split,
            "prompt": record.prompt,
            "evidence_label": HYPOTHESIS_LABEL,
            "nearest_neighbor_contexts": [
                {
                    "record_id": other.id,
                    "prompt": other.prompt,
                    "reason": "Same split context used as a local nearest-neighbor placeholder.",
                }
                for other in records
                if other.id != record.id and other.split == record.split
            ][:3],
            "hypothesis": (
                "Hypothesis placeholder: inspect these contexts and run causal tests before "
                "treating the representation description as evidence."
            ),
        }
        for record in selected
    ]


def _render_report(
    spec: ExperimentSpec,
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> str:
    lines = [
        f"# Cross-Model Representation Probe: {spec.name}",
        "",
        f"- Source model: {summary['source_model']}",
        f"- Target model: {summary['target_model']}",
        f"- Source site: `{summary['source_hook_site']}`",
        f"- Target site: `{summary['target_hook_site']}`",
        f"- Train records: {summary['train_record_count']}",
        f"- Eval records: {summary['eval_record_count']}",
        f"- Evidence label: {summary.get('evidence_label', CORRELATIONAL_ALIGNMENT_LABEL)}",
        "- Interpretation: this probe reports alignment statistics only; it does not "
        "establish causal interchangeability.",
        "",
        "## Results",
        "",
        "| Rank | Label | Split | Records | Mean cosine | Normalized MSE | Variance explained |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.get('rank', '-')} | "
            f"{row.get('evidence_label', CORRELATIONAL_ALIGNMENT_LABEL)} | "
            f"{row['split']} | {row['record_count']} | "
            f"{float(row['mean_cosine_similarity']):.4f} | "
            f"{float(row['normalized_mse']):.4f} | "
            f"{float(row['variance_explained']):.4f} |"
        )
    lines.extend(
        [
            "",
            "Activation verbalization entries, when present, are labeled as hypothesis "
            "rather than evidence; use them only to propose causal follow-up tests.",
            "",
        ]
    )
    return "\n".join(lines)


def _result_notes(rows: list[dict[str, Any]]) -> str:
    eval_rows = [row for row in rows if row["split"] == "eval"] or rows
    top = eval_rows[0]
    return (
        "Cross-model probe completed. "
        f"Eval mean cosine {float(top['mean_cosine_similarity']):.3f}; "
        f"variance explained {float(top['variance_explained']):.3f}."
    )



