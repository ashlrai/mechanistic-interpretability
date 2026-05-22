import json
from pathlib import Path

from mech_interp.storage import ArtifactStore


def test_artifact_store_writes_json_and_manifest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    record = store.write_json(1, "spec.json", {"name": "demo"})
    manifest = store.write_manifest(1, [record])

    assert record.path.exists()
    assert manifest.path.exists()
    assert record.sha256
    assert json.loads(manifest.path.read_text())["artifacts"][0]["name"] == "spec.json"
