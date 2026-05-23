from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mech_interp.datasets import PromptDatasetError, load_prompt_dataset
from mech_interp.types import ExperimentSpec


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    message: str


@dataclass(frozen=True)
class PreflightReport:
    spec_name: str
    checks: list[PreflightCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(check.status != "error" for check in self.checks)


OPTIONAL_BACKEND_IMPORTS = {
    "transformerlens": "transformer_lens",
    "nnsight": "nnsight",
    "mlx": "mlx",
}


def preflight_spec(spec: ExperimentSpec) -> PreflightReport:
    checks = [
        _dependency_check(spec),
        _dataset_hash_check(spec),
        _prompt_pair_check(spec),
        _answer_token_check(spec),
        _hook_site_check(spec),
        _activation_memory_check(spec),
        _artifact_policy_check(spec),
    ]
    return PreflightReport(spec_name=spec.name, checks=checks)


def inspect_dataset(path: str | Path) -> dict[str, Any]:
    dataset_path = Path(path)
    content = dataset_path.read_bytes()
    rows = dataset_path.read_text(encoding="utf-8").splitlines()
    parsed_rows: list[Any] = []
    invalid_json_lines: list[int] = []
    for index, line in enumerate(rows, start=1):
        if not line.strip():
            continue
        row = _parse_json_line(line)
        parsed_rows.append(row)
        if (
            dataset_path.suffix.lower() in {".jsonl", ".json"}
            and isinstance(row, dict)
            and row.get("_invalid_json_line") is True
        ):
            invalid_json_lines.append(index)
    raw_sha256 = hashlib.sha256(content).hexdigest()
    normalized_sha256 = _normalized_dataset_hash(dataset_path)
    return {
        "path": str(dataset_path),
        "sha256": normalized_sha256 or raw_sha256,
        "raw_sha256": raw_sha256,
        "normalized_sha256": normalized_sha256,
        "bytes": len(content),
        "rows": len(parsed_rows),
        "invalid_json_lines": invalid_json_lines,
        "fields": sorted(
            {
                key
                for row in parsed_rows
                if isinstance(row, dict)
                for key in row
                if not key.startswith("_")
            }
        ),
    }


def validate_answer_tokens(path: str | Path, model: str) -> dict[str, Any]:
    dataset = inspect_dataset(path)
    invalid: list[dict[str, str]] = []
    for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        row = _parse_json_line(line)
        if not isinstance(row, dict):
            continue
        for field_name, token in _answer_token_fields(row):
            reason = _answer_token_issue(token)
            if reason is not None:
                invalid.append(
                    {
                        "line": str(index),
                        "field": field_name,
                        "token": "" if token is None else str(token),
                        "reason": reason,
                    }
                )
    return {
        **dataset,
        "model": model,
        "valid": not invalid,
        "invalid_tokens": invalid,
    }


def _dependency_check(spec: ExperimentSpec) -> PreflightCheck:
    module = OPTIONAL_BACKEND_IMPORTS.get(spec.backend)
    if module is None:
        return PreflightCheck(
            "optional_dependency",
            "ok",
            f"No optional import for {spec.backend}.",
        )
    if importlib.util.find_spec(module) is None:
        return PreflightCheck(
            "optional_dependency",
            "warning",
            f"Missing optional dependency {module}.",
        )
    return PreflightCheck("optional_dependency", "ok", f"Found {module}.")


def _dataset_hash_check(spec: ExperimentSpec) -> PreflightCheck:
    dataset_specs = _dataset_specs(spec)
    if not dataset_specs:
        return PreflightCheck("dataset_hash", "ok", "No dataset hash requirement.")
    unpinned: list[str] = []
    mismatches: list[str] = []
    for path, expected_value in dataset_specs:
        if not path.exists():
            mismatches.append(f"{path}: missing")
            continue
        inspected = inspect_dataset(path)
        accepted_hashes = {
            str(value)
            for value in (
                inspected.get("sha256"),
                inspected.get("raw_sha256"),
                inspected.get("normalized_sha256"),
            )
            if value
        }
        if expected_value is None:
            unpinned.append(str(path))
        elif expected_value not in accepted_hashes:
            actual = inspected["sha256"]
            raw = inspected["raw_sha256"]
            mismatches.append(f"{path}: expected {expected_value}, got {actual} (raw {raw})")
    if mismatches:
        return PreflightCheck("dataset_hash", "error", "; ".join(mismatches))
    if unpinned:
        return PreflightCheck(
            "dataset_hash",
            "warning",
            f"Dataset hash not pinned: {', '.join(unpinned)}",
        )
    return PreflightCheck("dataset_hash", "ok", "Dataset hashes match or were not pinned.")


def _prompt_pair_check(spec: ExperimentSpec) -> PreflightCheck:
    pairs = spec.parameters.get("prompt_pairs", [])
    if not pairs:
        return PreflightCheck("prompt_pairs", "ok", "No clean/corrupt pairs required.")
    missing = [
        str(pair.get("id", index))
        for index, pair in enumerate(pairs, start=1)
        if isinstance(pair, dict)
        and not _required_pair_keys() <= set(pair)
    ]
    if missing:
        return PreflightCheck(
            "prompt_pairs",
            "error",
            f"Incomplete prompt pairs: {', '.join(missing)}",
        )
    return PreflightCheck("prompt_pairs", "ok", f"{len(pairs)} prompt pair(s) complete.")


def _answer_token_check(spec: ExperimentSpec) -> PreflightCheck:
    pairs = spec.parameters.get("prompt_pairs", [])
    invalid = [
        str(pair.get("id", index))
        for index, pair in enumerate(pairs, start=1)
        if isinstance(pair, dict)
        for token in (pair.get("correct_token"), pair.get("incorrect_token"))
        if _answer_token_issue(token) is not None
    ]
    dataset_invalid: list[str] = []
    for path, _expected in _dataset_specs(spec):
        if not path.exists():
            continue
        result = validate_answer_tokens(path, str(spec.parameters.get("model", "")))
        dataset_invalid.extend(
            f"{path}:{item['line']}:{item['field']}"
            for item in result["invalid_tokens"]
        )
    if invalid:
        return PreflightCheck(
            "answer_tokens",
            "warning",
            f"Likely multi-token answers: {', '.join(invalid)}",
        )
    if dataset_invalid:
        return PreflightCheck(
            "answer_tokens",
            "warning",
            f"Dataset answer-token issues: {', '.join(dataset_invalid)}",
        )
    return PreflightCheck("answer_tokens", "ok", "Answer tokens look single-token by whitespace.")


def _hook_site_check(spec: ExperimentSpec) -> PreflightCheck:
    sites = spec.parameters.get("hook_sites", [])
    if not sites:
        return PreflightCheck("hook_sites", "ok", "No hook sites declared.")
    invalid = [str(site) for site in sites if not isinstance(site, str) or not site.strip()]
    if invalid:
        return PreflightCheck("hook_sites", "error", f"Invalid hook sites: {', '.join(invalid)}")
    return PreflightCheck("hook_sites", "ok", f"{len(sites)} hook site(s) declared.")


def _activation_memory_check(spec: ExperimentSpec) -> PreflightCheck:
    batch = int(spec.parameters.get("batch_size", 1))
    seq = int(spec.parameters.get("sequence_length", 128))
    hidden = int(spec.parameters.get("hidden_size", 768))
    hooks = len(spec.parameters.get("hook_sites", [])) or int(spec.parameters.get("hook_count", 1))
    bytes_per = 2 if str(spec.parameters.get("dtype", "float16")) in {"float16", "bfloat16"} else 4
    gib = batch * seq * hidden * hooks * bytes_per / (1024**3)
    status = "warning" if gib > 16 else "ok"
    return PreflightCheck(
        "activation_memory",
        status,
        f"Estimated activation cache: {gib:.3f} GiB.",
    )


def _artifact_policy_check(spec: ExperimentSpec) -> PreflightCheck:
    policy = spec.parameters.get("artifact_policy", {})
    if isinstance(policy, dict) and policy.get("retain_activation_tensors"):
        return PreflightCheck("artifact_policy", "warning", "Tensor retention is enabled.")
    if (isinstance(policy, dict) and policy.get("retain_probe_weights")) or spec.parameters.get(
        "retain_probe_weights"
    ):
        return PreflightCheck("artifact_policy", "warning", "Probe weight retention is enabled.")
    return PreflightCheck("artifact_policy", "ok", "Risky tensor retention is disabled.")


def _parse_json_line(line: str) -> Any:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"text": line, "_invalid_json_line": True}


def _required_pair_keys() -> set[str]:
    return {"clean_prompt", "corrupted_prompt", "correct_token", "incorrect_token"}


def _dataset_specs(spec: ExperimentSpec) -> list[tuple[Path, str | None]]:
    parameters = spec.parameters
    specs: list[tuple[Path, str | None]] = []

    dataset_path = parameters.get("dataset_path")
    if dataset_path:
        expected = parameters.get("dataset_sha256")
        specs.append((Path(str(dataset_path)), str(expected) if expected else None))

    expected_hashes = parameters.get("dataset_hashes", {})
    datasets = parameters.get("datasets") or parameters.get("dataset")
    paths = datasets if isinstance(datasets, list) else [datasets] if datasets else []
    for value in paths:
        path = Path(str(value))
        expected = (
            expected_hashes.get(str(path))
            if isinstance(expected_hashes, dict)
            else None
        )
        specs.append((path, str(expected) if expected else None))

    seen: set[str] = set()
    unique: list[tuple[Path, str | None]] = []
    for path, expected in specs:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append((path, expected))
    return unique


def _normalized_dataset_hash(path: Path) -> str | None:
    try:
        return load_prompt_dataset(path).sha256
    except (OSError, PromptDatasetError):
        return None


def _answer_token_fields(row: dict[str, Any]) -> list[tuple[str, Any]]:
    fields: list[tuple[str, Any]] = [
        (key, row.get(key))
        for key in ("correct_token", "incorrect_token")
        if key in row
    ]
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        for key in ("correct_token", "incorrect_token"):
            if key in metadata:
                fields.append((f"metadata.{key}", metadata.get(key)))
        kind = metadata.get("kind")
        if kind == "clean" and "answer" in metadata:
            fields.append(("metadata.answer", metadata.get("answer")))
        if kind == "corrupted" and "answer" in metadata:
            fields.append(("metadata.answer", metadata.get("answer")))
    return fields


def _answer_token_issue(token: Any) -> str | None:
    if not isinstance(token, str):
        return "missing_or_non_string"
    normalized = token.strip()
    if not normalized:
        return "empty"
    if len(normalized.split()) != 1:
        return "multiple_whitespace_tokens"
    return None
