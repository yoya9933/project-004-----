"""Shared source-of-truth modules for legal penalty risk modeling."""

from .features import FEATURE_NAMES, ISSUE_FIELDS, build_feature_frame, case_splits
from .promotion import build_promotion_report

__all__ = [
    "FEATURE_NAMES",
    "ISSUE_FIELDS",
    "case_splits",
    "build_feature_frame",
    "build_promotion_report",
]
