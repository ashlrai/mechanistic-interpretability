import json
from pathlib import Path

import numpy as np

from mech_interp.storage import ArtifactStore


def test_artifact_store_writes_json_and_manifest(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)

    record = store.write_json(1, "spec.json", {"name": "demo"})
    manifest = store.write_manifest(1, [record])

    assert record.path.exists()
    assert manifest.path.exists()
    assert record.sha256
    assert store.read_json(1, "spec.json") == {"name": "demo"}
    assert store.read_manifest(1)["artifacts"][0]["name"] == "spec.json"
    assert json.loads(manifest.path.read_text())["artifacts"][0]["name"] == "spec.json"


def test_artifact_store_writes_npz_tensors_and_manifest_metadata(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    activations = np.arange(6, dtype=np.float32).reshape(2, 3)
    mask = np.array([True, False, True])

    record = store.write_npz(
        2,
        "activations.npz",
        {"activations": activations, "mask": mask},
        metadata={"layer": 4},
    )
    manifest = store.write_manifest(2, [record])
    arrays, metadata = store.read_npz(2, "activations.npz")

    np.testing.assert_array_equal(arrays["activations"], activations)
    np.testing.assert_array_equal(arrays["mask"], mask)
    assert metadata["metadata"] == {"layer": 4}
    assert metadata["tensors"]["activations"]["shape"] == [2, 3]
    assert metadata["tensors"]["activations"]["dtype"] == "float32"
    assert len(metadata["tensors"]["activations"]["sha256"]) == 64

    manifest_artifact = store.read_manifest(2)["artifacts"][0]
    assert manifest_artifact["name"] == "activations.npz"
    assert manifest_artifact["metadata"]["tensors"]["mask"]["shape"] == [3]
    assert json.loads(manifest.path.read_text())["artifacts"][0]["media_type"] == (
        "application/x-numpy-npz"
    )
