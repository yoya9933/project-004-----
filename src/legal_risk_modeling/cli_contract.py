from __future__ import annotations

import argparse
from pathlib import Path


def existing_csv_path(value: str | Path) -> Path:
    path = Path(value)
    if path.suffix.lower() != ".csv":
        raise argparse.ArgumentTypeError(f"expected a .csv file, got: {path}")
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"CSV file does not exist: {path}")
    return path


def ensure_existing_csv(path: Path) -> Path:
    try:
        return existing_csv_path(path)
    except argparse.ArgumentTypeError as exc:
        raise ValueError(str(exc)) from exc


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return number


def validate_temporal_split(
    train_start_year: int,
    train_end_year: int,
    validation_year: int,
    test_year: int,
    latest_check_year: int,
) -> None:
    if not (
        train_start_year
        <= train_end_year
        < validation_year
        < test_year
        < latest_check_year
    ):
        raise ValueError(
            "invalid temporal split: require "
            "train_start_year <= train_end_year < validation_year < test_year < latest_check_year"
        )
