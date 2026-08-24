from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ratio_candidates(random_state: int = 42) -> dict[str, list[tuple[str, Any]]]:
    """Single source of truth for all ratio-model candidate definitions."""
    return {
        "mean_baseline": [("mean_baseline", DummyRegressor(strategy="mean"))],
        "ridge_regression_l2": [
            (
                f"ridge_alpha_{alpha:g}",
                Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=alpha))]),
            )
            for alpha in [0.001, 0.01, 0.05, 0.1, 1.0, 10.0, 100.0]
        ],
        "lasso_regression_l1": [
            (
                f"lasso_alpha_{alpha:g}",
                Pipeline(
                    [
                        ("scale", StandardScaler()),
                        ("model", Lasso(alpha=alpha, max_iter=50000, random_state=random_state)),
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
                                max_iter=50000,
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            )
            for alpha in [0.0001, 0.001, 0.005, 0.01, 0.05]
            for l1_ratio in [0.2, 0.5, 0.8]
        ],
        "random_forest": [
            (
                f"rf_depth_{depth}_leaf_{leaf}",
                RandomForestRegressor(
                    n_estimators=500,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
            for depth in [3, 5, 8, None]
            for leaf in [5, 10, 20]
        ],
        "extra_trees": [
            (
                f"extra_depth_{depth}_leaf_{leaf}",
                ExtraTreesRegressor(
                    n_estimators=500,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
            for depth in [3, 5, 8, None]
            for leaf in [5, 10, 20]
        ],
        "gradient_boosting": [
            (
                f"gbr_depth_{depth}_lr_{learning_rate:g}_leaf_{leaf}",
                GradientBoostingRegressor(
                    n_estimators=300,
                    learning_rate=learning_rate,
                    max_depth=depth,
                    min_samples_leaf=leaf,
                    random_state=random_state,
                ),
            )
            for depth in [1, 2, 3]
            for learning_rate in [0.01, 0.03, 0.05]
            for leaf in [5, 10, 20]
        ],
        "hist_gradient_boosting": [
            (
                f"hist_lr_{learning_rate:g}_leaf_{leaf}_l2_{l2:g}",
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=learning_rate,
                    min_samples_leaf=leaf,
                    l2_regularization=l2,
                    random_state=random_state,
                ),
            )
            for learning_rate in [0.01, 0.03, 0.05]
            for leaf in [10, 20, 30]
            for l2 in [0.0, 0.1, 1.0]
        ],
    }


def build_classification_model(l2: float = 0.01, random_state: int = 42) -> Pipeline:
    """Single governed classifier definition shared by all classification entrypoints."""
    if l2 <= 0:
        raise ValueError("l2 must be positive")
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0 / l2,
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def model_spec() -> dict[str, Any]:
    return {
        "target": "remaining_ratio",
        "prediction_clip": [0.0, 1.0],
        "selection_metric": "training_period_rolling_cv_rmse",
        "families": {
            name: [candidate_name for candidate_name, _ in candidates]
            for name, candidates in build_ratio_candidates().items()
        },
    }


def classification_model_spec(l2: float = 0.01) -> dict[str, Any]:
    return {
        "target": "is_reduced",
        "threshold": 0.5,
        "features": "legal_risk_modeling.features.FEATURE_NAMES",
        "models": {
            "majority_baseline": {"type": "training_prevalence"},
            "keyword_rule_baseline": {"type": "deterministic_rule"},
            "logistic_regression_l2": {
                "type": "sklearn.pipeline.Pipeline",
                "estimator": "sklearn.linear_model.LogisticRegression",
                "l2": l2,
                "C": 1.0 / l2,
                "class_weight": "balanced",
            },
        },
    }
