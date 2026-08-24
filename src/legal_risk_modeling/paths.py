from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_ROOT_ENV = "LEGAL_RISK_ARTIFACT_ROOT"


def artifact_root(project_root: Path) -> Path:
    configured = os.environ.get(ARTIFACT_ROOT_ENV, "").strip()
    if not configured:
        return project_root / ".artifacts"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else project_root / path


def ratio_output_dir(project_root: Path) -> Path:
    return artifact_root(project_root) / "reduction_ratio_model_expanded_824_sklearn"


def classification_output_dir(project_root: Path) -> Path:
    return artifact_root(project_root) / "is_reduced_classification"


def rag_benchmark_output_dir(project_root: Path) -> Path:
    return artifact_root(project_root) / "rag_benchmark"


def legacy_ratio_output_dir(project_root: Path) -> Path:
    return project_root / "06_交付物" / "reduction_ratio_model_expanded_824_sklearn"


def resolve_ratio_release_dir(project_root: Path) -> Path:
    active = ratio_output_dir(project_root)
    if (active / "approved_release.json").is_file():
        return active
    return legacy_ratio_output_dir(project_root)
