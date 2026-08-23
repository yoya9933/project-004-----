from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ratio_candidates(random_state: int = 42) -> dict[str, list[tuple[str, Any]]]:
    """Single source of truth for all ratio-model candidate definitions."""
    return {
        "mean_baseline": [("mean_baseline", DummyRegressor(strategy="mean"))],
        "ridge_regression_l2": [
            (
                f"ridge_alpha_{alpha:g}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", Ridge(alpha=alpha)),
                    ]
                ),
            )
            for alpha in [0.001, 0.01, 0.05, 0.1, 1.0, 10.0, 100.0]
        ],
        "lasso_regression_l1": [
            (
                f"lasso_alpha_{alpha:g}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", Lasso(alpha=alpha, max_iter=10000, random_state=random_state)),
                    ]
                ),
            )
            for alpha in [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1]
        ],
        "elastic_net": [
            (
                f"elastic_alpha_{alpha:g}_l1_{l1_ratio:g}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "model",
                            ElasticNet(
                                alpha=alpha,
                                l1_ratio=l1_ratio,
                                max_iter=10000,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            )
            for alpha in [0.0001, 0.001, 0.005, 0.01, 0.05]
            for l1_ratio in [0.2, 0.5, 0.8]
        ],
    }


def model_spec() -> dict[str, Any]:
    """Serializable model policy used by manifests and tests."""
    return {
        "target": "remaining_ratio",
        "prediction_clip": [0.0, 1.0],
        "selection_metric": "validation_rmse",
        "families": {
            name: [candidate_name for candidate_name, _ in candidates]
            for name, candidates in build_ratio_candidates().items()
        },
    }
