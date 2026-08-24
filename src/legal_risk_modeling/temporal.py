from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class TemporalSplitPolicy:
    train_start_year: int = 2021
    train_end_year: int = 2023
    validation_year: int = 2024
    test_year: int = 2025
    latest_check_year: int = 2026

    def __post_init__(self) -> None:
        if not (
            self.train_start_year
            <= self.train_end_year
            < self.validation_year
            < self.test_year
            < self.latest_check_year
        ):
            raise ValueError(
                "invalid temporal split: require "
                "train_start_year <= train_end_year < validation_year < test_year < latest_check_year"
            )

    @property
    def train_name(self) -> str:
        return f"train_{self.train_start_year}_{self.train_end_year}"

    @property
    def validation_name(self) -> str:
        return f"validation_{self.validation_year}"

    @property
    def test_name(self) -> str:
        return f"test_{self.test_year}"

    @property
    def latest_name(self) -> str:
        return f"latest_{self.latest_check_year}"

    def as_dict(self) -> dict[str, int]:
        return asdict(self)

    def split_frame(
        self,
        frame: pd.DataFrame,
        *,
        year_column: str = "decision_year",
    ) -> list[tuple[str, pd.DataFrame]]:
        years = pd.to_numeric(frame[year_column], errors="coerce")
        return [
            (
                self.train_name,
                frame[
                    (years >= self.train_start_year) & (years <= self.train_end_year)
                ].copy(),
            ),
            (self.validation_name, frame[years == self.validation_year].copy()),
            (self.test_name, frame[years == self.test_year].copy()),
            (self.latest_name, frame[years == self.latest_check_year].copy()),
        ]

    def rolling_cv_splits(
        self,
        frame: pd.DataFrame,
        *,
        year_column: str = "decision_year",
    ) -> list[tuple[str, pd.DataFrame, pd.DataFrame]]:
        years = pd.to_numeric(frame[year_column], errors="coerce")
        train_mask = (years >= self.train_start_year) & (years <= self.train_end_year)
        train_years = sorted({int(year) for year in years[train_mask].dropna().tolist()})
        folds: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
        for holdout_year in train_years[1:]:
            fit = frame[train_mask & (years < holdout_year)].copy()
            validate = frame[train_mask & (years == holdout_year)].copy()
            if fit.empty or validate.empty:
                continue
            folds.append((f"rolling_{holdout_year}", fit, validate))
        return folds
