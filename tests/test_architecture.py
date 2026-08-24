from __future__ import annotations

import ast
from pathlib import Path

import pytest

from legal_risk_modeling.paths import resolve_ratio_release_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def contains_fit_call(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "fit"
        for node in ast.walk(tree)
    )


def test_streamlit_root_is_thin_and_ui_has_no_training() -> None:
    root = PROJECT_ROOT / "streamlit_app.py"
    ui = PROJECT_ROOT / "src" / "legal_risk_modeling" / "ui.py"
    assert len(root.read_text(encoding="utf-8").splitlines()) < 50
    assert "sklearn" not in imported_roots(root)
    assert "sklearn" not in imported_roots(ui)
    assert not contains_fit_call(root)
    assert not contains_fit_call(ui)


def test_legacy_powershell_model_scripts_are_only_wrappers() -> None:
    for relative in [
        "04_執行稿/run_reduction_ratio_model.ps1",
        "04_執行稿/run_is_reduced_classification.ps1",
    ]:
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8-sig")
        assert "$FeatureNames" not in text
        assert "Train-LogisticRegression" not in text
        assert len(text.splitlines()) < 60


def test_duplicate_powershell_keyword_preprocessor_is_removed() -> None:
    assert not (PROJECT_ROOT / "04_執行稿" / "build_keyword_candidate_pools.ps1").exists()


def test_ratio_release_resolution_never_falls_back_to_legacy(tmp_path: Path) -> None:
    legacy = tmp_path / "06_交付物" / "reduction_ratio_model_expanded_824_sklearn"
    legacy.mkdir(parents=True)
    (legacy / "approved_release.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="legacy artifacts are not permitted"):
        resolve_ratio_release_dir(tmp_path)
