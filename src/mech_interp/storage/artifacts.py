from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from mech_interp.types import ArtifactRecord

METADATA_ARRAY_NAME = "__metadata__"


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def run_dir(self, run_id: int) -> Path:
        path = self._run_path(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, run_id: int, name: str, payload: dict[str, Any]) -> ArtifactRecord:
        path = self.run_dir(run_id) / name
        tmp_path = path.with_name(f".{path.name}.tmp")
        with tmp_path.open("w", encoding="utf-8") as artifact_file:
            json.dump(payload, artifact_file, indent=2, sort_keys=True)
            artifact_file.write("\n")
        tmp_path.replace(path)
        return self._record(name=name, path=path, media_type="application/json")

    def write_text(self, run_id: int, name: str, text: str) -> ArtifactRecord:
        path = self.run_dir(run_id) / name
        tmp_path = path.with_name(f".{path.name}.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
        return self._record(name=name, path=path, media_type="text/plain")

    def write_npz(
        self,
        run_id: int,
        name: str,
        arrays: Mapping[str, npt.ArrayLike],
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        if not arrays:
            raise ValueError("NPZ artifacts must contain at least one array.")
        if METADATA_ARRAY_NAME in arrays:
            raise ValueError(f"'{METADATA_ARRAY_NAME}' is reserved for artifact metadata.")

        materialized = {
            array_name: np.asarray(array)
            for array_name, array in arrays.items()
        }
        tensor_metadata = {
            array_name: self._array_metadata(array)
            for array_name, array in materialized.items()
        }
        artifact_metadata: dict[str, Any] = {
            "metadata": dict(metadata or {}),
            "tensors": tensor_metadata,
        }
        payload = {
            **materialized,
            METADATA_ARRAY_NAME: np.array(
                json.dumps(artifact_metadata, default=str, sort_keys=True),
            ),
        }

        path = self.run_dir(run_id) / name
        tmp_path = path.with_name(f".{path.name}.tmp.npz")
        np.savez(tmp_path, **cast(Any, payload))
        tmp_path.replace(path)
        return self._record(
            name=name,
            path=path,
            media_type="application/x-numpy-npz",
            metadata=artifact_metadata,
        )

    def read_json(self, run_id: int, name: str) -> dict[str, Any]:
        path = self._run_path(run_id) / name
        with path.open("r", encoding="utf-8") as artifact_file:
            payload = json.load(artifact_file)
        if not isinstance(payload, dict):
            raise ValueError(f"Artifact {path} did not contain a JSON object.")
        return payload

    def read_npz(self, run_id: int, name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        path = self._run_path(run_id) / name
        with np.load(path, allow_pickle=False) as archive:
            arrays = {
                array_name: archive[array_name]
                for array_name in archive.files
                if array_name != METADATA_ARRAY_NAME
            }
            metadata = self._decode_npz_metadata(archive[METADATA_ARRAY_NAME])
        return arrays, metadata

    def write_manifest(self, run_id: int, records: list[ArtifactRecord]) -> ArtifactRecord:
        payload = {
            "run_id": run_id,
            "artifacts": [
                {
                    "name": record.name,
                    "path": str(record.path),
                    "media_type": record.media_type,
                    "sha256": record.sha256,
                    "size_bytes": record.size_bytes,
                    "metadata": record.metadata,
                }
                for record in records
            ],
        }
        return self.write_json(run_id, "manifest.json", payload)

    def read_manifest(self, run_id: int) -> dict[str, Any]:
        return self.read_json(run_id, "manifest.json")

    def _run_path(self, run_id: int) -> Path:
        return self.root / f"run-{run_id:06d}"

    def _record(
        self,
        name: str,
        path: Path,
        media_type: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        content = path.read_bytes()
        return ArtifactRecord(
            name=name,
            path=path,
            media_type=media_type,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            metadata=dict(metadata or {}),
        )

    def _array_metadata(self, array: np.ndarray) -> dict[str, Any]:
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "sha256": self._array_hash(array),
        }

    def _array_hash(self, array: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(array)
        digest = hashlib.sha256()
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(json.dumps(list(contiguous.shape)).encode("utf-8"))
        digest.update(contiguous.view(np.uint8).tobytes())
        return digest.hexdigest()

    def _decode_npz_metadata(self, metadata_array: np.ndarray) -> dict[str, Any]:
        payload = str(metadata_array.item())
        decoded = json.loads(payload)
        if not isinstance(decoded, dict):
            raise ValueError("NPZ metadata did not contain a JSON object.")
        return decoded
