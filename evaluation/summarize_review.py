"""Aggregate a completed human review CSV.

Reviewers fill `human_correct` and `human_hallucination` with y/n.

Usage:
    uv run python -m evaluation.summarize_review reports/human_review_rag.csv
"""

import csv
import json
import sys
from pathlib import Path


def summarize(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def rate(col: str) -> float | None:
        vals = [r[col].strip().lower() for r in rows if r.get(col, "").strip()]
        if not vals:
            return None
        return round(sum(v in {"y", "yes", "1", "true"} for v in vals) / len(vals), 4)

    return {
        "n_rows": len(rows),
        "n_reviewed": sum(1 for r in rows if r.get("human_correct", "").strip()),
        "human_accuracy": rate("human_correct"),
        "human_hallucination_rate": rate("human_hallucination"),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(json.dumps(summarize(Path(sys.argv[1])), indent=2))
