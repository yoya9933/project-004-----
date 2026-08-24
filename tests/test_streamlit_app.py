from __future__ import annotations

import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_fails_closed_without_governed_release(tmp_path: Path) -> None:
    previous = os.environ.get("LEGAL_RISK_ARTIFACT_ROOT")
    os.environ["LEGAL_RISK_ARTIFACT_ROOT"] = str(tmp_path / "empty-artifacts")
    try:
        app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=20)
    finally:
        if previous is None:
            os.environ.pop("LEGAL_RISK_ARTIFACT_ROOT", None)
        else:
            os.environ["LEGAL_RISK_ARTIFACT_ROOT"] = previous
    assert len(app.exception) == 0
    assert app.title[0].value == "工程違約金風險模型"
    assert len(app.error) >= 1
    assert "legacy artifacts are not permitted" in app.error[0].value
    assert len(app.selectbox) == 0
