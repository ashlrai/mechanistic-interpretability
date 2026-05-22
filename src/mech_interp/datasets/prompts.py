from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias, cast

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class PromptDatasetError(ValueError):
    """Raised when a prompt dataset cannot be loaded or normalized."""


@dataclass(frozen=True)
class PromptRecord:
    """A single prompt and optional JSON-serializable metadata."""

    id: str
    prompt: str
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        record_id = self.id.strip()
        prompt = _normalize_prompt_text(self.prompt)
        if not record_id:
            raise PromptDatasetError("prompt record id must not be empty")
        if not prompt:
            raise PromptDatasetError(f"prompt record '{record_id}' must not be empty")

        metadata = _normalize_json_mapping(
            self.metadata,
            field_name=f"record '{record_id}' metadata",
        )
        object.__setattr__(self, "id", record_id)
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "metadata", metadata)

    @property
    def sha256(self) -> str:
        return _sha256_json(self.normalized())

    def normalized(self) -> dict[str, JsonValue]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class PromptDataset:
    """A loaded prompt dataset with stable record and dataset hashes."""

    name: str
    records: tuple[PromptRecord, ...]
    source_path: Path | None = None

    def __post_init__(self) -> None:
        name = self.name.strip()
        if not name:
            raise PromptDatasetError("prompt dataset name must not be empty")
        if not self.records:
            raise PromptDatasetError(f"prompt dataset '{name}' must contain at least one record")
        ids = [record.id for record in self.records]
        duplicates = sorted({record_id for record_id in ids if ids.count(record_id) > 1})
        if duplicates:
            joined = ", ".join(duplicates)
            raise PromptDatasetError(
                f"prompt dataset '{name}' has duplicate record id(s): {joined}"
            )

        object.__setattr__(self, "name", name)

    @property
    def prompts(self) -> list[str]:
        return [record.prompt for record in self.records]

    @property
    def sha256(self) -> str:
        return _sha256_json([record.normalized() for record in self.records])

    def record_hashes(self) -> dict[str, str]:
        return {record.id: record.sha256 for record in self.records}


def load_prompt_dataset(path: str | Path, *, name: str | None = None) -> PromptDataset:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise PromptDatasetError(f"prompt dataset does not exist: {dataset_path}")
    if not dataset_path.is_file():
        raise PromptDatasetError(f"prompt dataset path is not a file: {dataset_path}")

    suffix = dataset_path.suffix.lower()
    if suffix == ".jsonl":
        records = _load_jsonl_records(dataset_path)
    elif suffix in {".txt", ".text"}:
        records = _load_text_records(dataset_path)
    else:
        raise PromptDatasetError(
            f"unsupported prompt dataset format '{suffix}' for {dataset_path}; "
            "expected .jsonl, .txt, or .text"
        )

    return PromptDataset(
        name=name or dataset_path.stem,
        records=tuple(records),
        source_path=dataset_path,
    )


def _load_jsonl_records(path: Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with path.open("r", encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromptDatasetError(
                    f"invalid JSONL in {path} on line {line_number}: {exc.msg}"
                ) from exc
            if not isinstance(raw, dict):
                raise PromptDatasetError(
                    f"invalid JSONL in {path} on line {line_number}: expected an object"
                )
            records.append(_record_from_json_object(raw, path=path, line_number=line_number))
    return records


def _record_from_json_object(
    raw: dict[str, Any],
    *,
    path: Path,
    line_number: int,
) -> PromptRecord:
    prompt = raw.get("prompt")
    if not isinstance(prompt, str):
        raise PromptDatasetError(
            f"invalid prompt record in {path} on line {line_number}: "
            "field 'prompt' must be a string"
        )

    record_id = raw.get("id", f"{path.stem}-{line_number:04d}")
    if not isinstance(record_id, str):
        raise PromptDatasetError(
            f"invalid prompt record in {path} on line {line_number}: field 'id' must be a string"
        )

    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise PromptDatasetError(
            f"invalid prompt record in {path} on line {line_number}: "
            "field 'metadata' must be an object"
        )

    extra_fields = {
        key: value for key, value in raw.items() if key not in {"id", "prompt", "metadata"}
    }
    if extra_fields:
        metadata = {**metadata, **extra_fields}

    return PromptRecord(id=record_id, prompt=prompt, metadata=metadata)


def _load_text_records(path: Path) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    with path.open("r", encoding="utf-8") as dataset_file:
        for line in dataset_file:
            prompt = _normalize_prompt_text(line)
            if not prompt or prompt.startswith("#"):
                continue
            records.append(
                PromptRecord(
                    id=f"{path.stem}-{len(records) + 1:04d}",
                    prompt=prompt,
                )
            )
    return records


def _normalize_prompt_text(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_json_mapping(value: dict[str, Any], *, field_name: str) -> dict[str, JsonValue]:
    normalized = _freeze_json_value(_ensure_json_value(value, field_name=field_name))
    return cast(dict[str, JsonValue], normalized)


def _ensure_json_value(value: Any, *, field_name: str) -> JsonValue:
    if isinstance(value, float) and not math.isfinite(value):
        raise PromptDatasetError(f"{field_name} contains a non-finite float")
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, list):
        return [_ensure_json_value(item, field_name=field_name) for item in value]
    if isinstance(value, dict):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PromptDatasetError(f"{field_name} contains a non-string key")
            normalized[key] = _ensure_json_value(item, field_name=field_name)
        return normalized
    raise PromptDatasetError(f"{field_name} contains a non-JSON value: {type(value).__name__}")


def _freeze_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, list):
        return [_freeze_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _freeze_json_value(value[key]) for key in sorted(value)}
    return value


def _sha256_json(value: JsonValue | list[dict[str, JsonValue]]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
