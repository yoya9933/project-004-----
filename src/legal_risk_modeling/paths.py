from __future__ import annotations

import os
from pathlib import Path

ARTIFACT_ROOT_ENV = "LEGAL_RISK_ARTIFACT_ROOT"


def artifact_root(project_root: Path) -> Path:
    configured = os.environ.get(ARTIFACT_ROOT_ENV)
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else project_root / path
    return project_root / ".artifacts"


def ratio_output_dir(project_root: Path) -> Path:
    return artifact_root(project_root) / "reduction_ratio_model_expanded_824_sklearn"


def classification_output_dir(project_root: Path) -> Path:
    return artifact_root(project_root) / "is_reduced_classification"


def rag_benchmark_output_dir(project_root: Path) -> Path:
    return artifact_root(project_root) / "rag_benchmark"


def data_quality_output_dir(project_root: Path) -> Path:
    return artifact_root(project_root) / "data_quality"


def rag_human_gold_output_dir(project_root: Path) -> Path:
    return artifact_root(project_root) / "rag_human_gold"


def resolve_ratio_release_dir(project_root: Path) -> Path:
    release_dir = ratio_output_dir(project_root)
    required = [
        release_dir / "approved_release.json",
        release_dir / "promotion_report.json",
        release_dir / "manifest.json",
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        joined = ", ".join(path.relative_to(project_root).as_posix() for path in missing)
        raise FileNotFoundError(
            "Governed release is unavailable; legacy artifacts are not permitted. Missing: "
            + joined
        )
    return release_dir
