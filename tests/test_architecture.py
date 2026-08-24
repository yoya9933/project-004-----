from __future__ import annotations

import ast
from pathlib import Path

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
