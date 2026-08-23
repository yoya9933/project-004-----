from __future__ import annotations

import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GOVERNED_TRAINER = PROJECT_ROOT / "04_執行稿" / "run_reduction_ratio_model_sklearn.py"


def main() -> None:
    """Compatibility entrypoint; all training is delegated to the governed trainer."""
    if not GOVERNED_TRAINER.exists():
        raise FileNotFoundError(f"governed trainer not found: {GOVERNED_TRAINER}")
    runpy.run_path(str(GOVERNED_TRAINER), run_name="__main__")


if __name__ == "__main__":
    main()
