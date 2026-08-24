from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from legal_risk_modeling.cli_contract import ensure_existing_csv, positive_int  # noqa: E402
from legal_risk_modeling.paths import rag_human_gold_output_dir  # noqa: E402
from legal_risk_modeling.rag_human_gold import write_review_queue  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a review queue for human RAG relevance judgments.")
    parser.add_argument(
        "--retrieval-csv",
        type=Path,
        default=PROJECT_ROOT / ".artifacts" / "ai_rag_annotation" / "rag_similar_cases.csv",
    )
    parser.add_argument(
        "--gold-csv",
        type=Path,
        default=PROJECT_ROOT / "06_交付物" / "ai_rag_annotation" / "human_rag_gold.csv",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=rag_human_gold_output_dir(PROJECT_ROOT) / "review_queue.csv",
    )
    parser.add_argument("--top-n", type=positive_int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieval_csv = ensure_existing_csv(args.retrieval_csv)
    gold_csv = ensure_existing_csv(args.gold_csv)
    report = write_review_queue(
        retrieval_csv,
        gold_csv,
        args.output_csv,
        top_n=args.top_n,
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
