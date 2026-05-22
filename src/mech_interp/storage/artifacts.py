from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from mech_interp.types import ArtifactRecord


class ArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def run_dir(self, run_id: int) -> Path:
        path = self._run_path(run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, run_id: int, name: str, payload: dict[str, Any]) -> ArtifactRecord:
        path = self.run_dir(run_id) / name
        with path.open("w", encoding="utf-8") as artifact_file:
            json.dump(payload, artifact_file, indent=2, sort_keys=True)
            artifact_file.write("\n")
        return self._record(name=name, path=path, media_type="application/json")

    def write_text(self, run_id: int, name: str, text: str) -> ArtifactRecord:
        path = self.run_dir(run_id) / name
        path.write_text(text, encoding="utf-8")
        return self._record(name=name, path=path, media_type="text/plain")

    def read_json(self, run_id: int, name: str) -> dict[str, Any]:
        path = self._run_path(run_id) / name
        with path.open("r", encoding="utf-8") as artifact_file:
            payload = json.load(artifact_file)
        if not isinstance(payload, dict):
            raise ValueError(f"Artifact {path} did not contain a JSON object.")
        return payload

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
                }
                for record in records
            ],
        }
        return self.write_json(run_id, "manifest.json", payload)

    def read_manifest(self, run_id: int) -> dict[str, Any]:
        return self.read_json(run_id, "manifest.json")

    def _run_path(self, run_id: int) -> Path:
        return self.root / f"run-{run_id:06d}"

    def _record(self, name: str, path: Path, media_type: str) -> ArtifactRecord:
        content = path.read_bytes()
        return ArtifactRecord(
            name=name,
            path=path,
            media_type=media_type,
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        )
