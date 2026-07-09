from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "streamlit_app.py"


def load_app_module():
    spec = importlib.util.spec_from_file_location("feature_correlation_target", APP_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {APP_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = load_app_module()


def make_row(**overrides: str) -> dict[str, str]:
    row = {
        "正式特徵_業主可歸責": "0",
        "正式特徵_承包商可歸責": "1",
        "正式特徵_展延免計工期爭議": "0",
        "正式特徵_實際損害不明偏低": "1",
        "正式特徵_部分完成部分驗收": "1",
        "正式特徵_業主已使用受益": "0",
        "是否酌減": "否",
        "酌減率": "0",
        "主張違約金": "100",
        "法院准許違約金": "100",
    }
    row.update(overrides)
    return row


class FeatureCorrelationDataTests(unittest.TestCase):
    def test_prepare_frame_converts_features_and_targets(self) -> None:
        frame = app.prepare_feature_correlation_frame(
            [
                make_row(),
                make_row(
                    **{
                        "正式特徵_業主可歸責": "1",
                        "正式特徵_承包商可歸責": "0",
                        "是否酌減": "是",
                        "酌減率": "0.6",
                        "主張違約金": "100",
                        "法院准許違約金": "40",
                    }
                ),
            ]
        )
        self.assertEqual(frame.loc[0, "業主可歸責"], 0.0)
        self.assertEqual(frame.loc[1, "是否酌減"], 1.0)
        self.assertAlmostEqual(frame.loc[1, "酌減率"], 0.6)

    def test_prepare_frame_rejects_missing_required_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "缺少必要欄位"):
            app.prepare_feature_correlation_frame([{"是否酌減": "是"}])

    def test_invalid_penalty_amounts_exclude_reduction_rate(self) -> None:
        frame = app.prepare_feature_correlation_frame(
            [
                make_row(
                    **{
                        "酌減率": "0.2",
                        "主張違約金": "0",
                        "法院准許違約金": "0",
                    }
                )
            ]
        )
        self.assertTrue(math.isnan(frame.loc[0, "酌減率"]))

    def test_compute_correlations_keeps_constant_feature_as_nan(self) -> None:
        rows = [
            make_row(
                **{
                    "正式特徵_業主可歸責": str(index % 2),
                    "正式特徵_承包商可歸責": str((index + 1) % 2),
                    "是否酌減": "是" if index % 2 else "否",
                    "酌減率": str(index / 3),
                    "主張違約金": "100",
                    "法院准許違約金": str(100 - (100 * index / 3)),
                }
            )
            for index in range(4)
        ]
        frame = app.prepare_feature_correlation_frame(rows)
        result = app.compute_feature_correlation(frame)
        self.assertEqual(result["matrix"].shape, (6, 6))
        partial = result["matrix"].loc["部分完成／部分驗收", "業主可歸責"]
        self.assertTrue(math.isnan(partial))
        self.assertEqual(len(result["is_reduced"]), 6)
        self.assertEqual(len(result["reduction_rate"]), 6)


class FeatureCorrelationUiTests(unittest.TestCase):
    def test_streamlit_app_mounts_feature_correlation_tab(self) -> None:
        source = APP_PATH.read_text(encoding="utf-8")
        self.assertIn("def render_feature_correlation()", source)
        self.assertIn("tab_correlation", source)
        self.assertIn('"特徵相關性"', source)


if __name__ == "__main__":
    unittest.main()
