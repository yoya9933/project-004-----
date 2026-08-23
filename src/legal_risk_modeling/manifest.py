from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_sha(project_root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_snapshot() -> dict[str, str]:
    packages: dict[str, str] = {}
    try:
        from importlib.metadata import distributions

        for dist in distributions():
            name = dist.metadata.get("Name")
            if name:
                packages[name] = dist.version
    except Exception:  # pragma: no cover - diagnostic only
        pass
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def build_manifest(
    *,
    project_root: Path,
    dataset_paths: list[Path],
    artifact_paths: list[Path],
    model_spec: dict[str, Any],
    run_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(project_root),
        "python": sys.version,
        "platform": platform.platform(),
        "datasets": [
            {
                "path": str(path.relative_to(project_root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in dataset_paths
        ],
        "artifacts": [
            {
                "path": str(path.relative_to(project_root)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for path in artifact_paths
            if path.exists()
        ],
        "model_spec": model_spec,
        "packages": package_snapshot(),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
