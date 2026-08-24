from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_renders_without_exception() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py")).run(timeout=20)
    assert len(app.exception) == 0
    assert app.title[0].value == "工程違約金風險模型"
    assert len(app.selectbox) >= 1
