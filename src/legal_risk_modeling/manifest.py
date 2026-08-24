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


def _git_output(project_root: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=project_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state(project_root: Path) -> dict[str, Any]:
    sha = _git_output(project_root, "rev-parse", "HEAD")
    status = _git_output(project_root, "status", "--porcelain", "--untracked-files=normal")
    return {"sha": sha, "dirty": None if status is None else bool(status)}


def package_snapshot() -> dict[str, str]:
    from importlib.metadata import distributions

    packages: dict[str, str] = {}
    for dist in distributions():
        name = dist.metadata.get("Name")
        if name:
            packages[name] = dist.version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


def _relative(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    root = project_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"governed path must be inside project root: {resolved}") from exc


def _file_record(path: Path, project_root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"governed file does not exist: {path}")
    return {
        "path": _relative(path, project_root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_manifest(
    *,
    project_root: Path,
    dataset_paths: list[Path],
    artifact_paths: list[Path],
    model_spec: dict[str, Any],
    run_id: str,
    invocation: dict[str, Any] | None = None,
    temporal_split_policy: dict[str, Any] | None = None,
    promotion_policy: dict[str, Any] | None = None,
    strict_git: bool = False,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    git = git_state(project_root)
    if strict_git and not git["sha"]:
        raise RuntimeError("governed run requires an available Git commit SHA")
    if strict_git and git["dirty"] is not False:
        raise RuntimeError("governed run requires a clean Git worktree")

    lock_path = project_root / "requirements.lock"
    lock_record = _file_record(lock_path, project_root) if lock_path.is_file() else None
    return {
        "schema_version": 2,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "packages": package_snapshot(),
            "requirements_lock": lock_record,
        },
        "invocation": invocation or {},
        "temporal_split_policy": temporal_split_policy,
        "promotion_policy": promotion_policy,
        "datasets": [_file_record(path, project_root) for path in dataset_paths],
        "artifacts": [_file_record(path, project_root) for path in artifact_paths],
        "model_spec": model_spec,
    }


def _validate_record(project_root: Path, record: dict[str, Any]) -> None:
    raw_path = Path(str(record["path"]))
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise ValueError(f"unsafe manifest path: {raw_path}")
    path = (project_root / raw_path).resolve()
    try:
        path.relative_to(project_root.resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes project root: {raw_path}") from exc
    if not path.is_file():
        raise FileNotFoundError(f"manifest file is missing: {raw_path}")
    if int(record.get("bytes", -1)) != path.stat().st_size:
        raise ValueError(f"manifest byte-size mismatch: {raw_path}")
    if str(record.get("sha256", "")) != sha256_file(path):
        raise ValueError(f"manifest SHA256 mismatch: {raw_path}")


def validate_manifest(
    *,
    project_root: Path,
    manifest_path: Path,
    expected_git_sha: str | None = None,
    require_clean_git: bool = False,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest is required: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2:
        raise ValueError("manifest schema_version must be 2")

    git = manifest.get("git") or {}
    if expected_git_sha is not None and git.get("sha") != expected_git_sha:
        raise ValueError(
            f"manifest Git SHA mismatch: {git.get('sha')!r} != {expected_git_sha!r}"
        )
    if require_clean_git and git.get("dirty") is not False:
        raise ValueError("manifest was created from a dirty or unknown Git worktree")

    datasets = manifest.get("datasets")
    artifacts = manifest.get("artifacts")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("manifest datasets must be a non-empty list")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("manifest artifacts must be a non-empty list")
    for record in [*datasets, *artifacts]:
        _validate_record(project_root, record)

    runtime = manifest.get("runtime") or {}
    lock_record = runtime.get("requirements_lock")
    if lock_record is not None:
        _validate_record(project_root, lock_record)
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
