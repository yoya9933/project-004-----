from __future__ import annotations

from pathlib import Path

import pytest

from legal_risk_modeling.manifest import build_manifest, validate_manifest, write_manifest


def test_manifest_v2_detects_artifact_tampering(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.csv"
    artifact = tmp_path / "artifact.csv"
    dataset.write_text("x\n1\n", encoding="utf-8")
    artifact.write_text("y\n2\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = build_manifest(
        project_root=tmp_path,
        dataset_paths=[dataset],
        artifact_paths=[artifact],
        model_spec={"target": "demo"},
        run_id="test",
    )
    write_manifest(manifest_path, manifest)
    assert validate_manifest(project_root=tmp_path, manifest_path=manifest_path)["schema_version"] == 2

    artifact.write_text("y\n3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_manifest(project_root=tmp_path, manifest_path=manifest_path)
