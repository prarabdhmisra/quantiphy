"""Pre-flight check for a QuantiPhy submission CSV.

The daily quota is 3 *scored* submissions per team per UTC day, and there is no server-side dry
run. Every failure mode below silently costs points rather than erroring, so check locally first.

Usage:
    py -3.12 scripts/validate_submission.py submission.csv [--expect-rows 3289]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from quantiphy.scoring import CATEGORIES, category_labels  # noqa: E402

MAX_BYTES = 2 * 1024 * 1024

#: Units the questions ask for. The scorer performs no unit conversion whatsoever -- the answer
#: must already be in the unit named at the end of the question.
UNIT_PATTERN = re.compile(r"\bin\s+([a-zA-Z/^0-9²]+)\s*\??\s*$")


def check(path: Path, expect_rows: int | None) -> list[str]:
    problems: list[str] = []
    size = path.stat().st_size
    if size > MAX_BYTES:
        problems.append(f"file is {size / 1e6:.2f} MB; the portal rejects anything over 2 MB")

    frame = pd.read_csv(path, encoding="utf-8-sig")
    print(f"{path.name}: {len(frame)} rows, {len(frame.columns)} columns")
    print(f"  columns: {list(frame.columns)}")

    if "parsed_value" not in frame.columns:
        problems.append("missing the 'parsed_value' column that holds the prediction")
        return problems

    if expect_rows is not None and len(frame) != expect_rows:
        problems.append(f"expected {expect_rows} rows, found {len(frame)}")

    values = pd.to_numeric(frame["parsed_value"], errors="coerce")
    blank = int(values.isna().sum())
    zero = int((values == 0).sum())
    negative = int((values < 0).sum())
    if blank:
        problems.append(f"{blank} blank/unparseable predictions -- each scores a hard 0")
    if zero:
        problems.append(f"{zero} zero predictions -- each scores a hard 0")
    if negative:
        problems.append(f"{negative} negative predictions -- scored by magnitude, likely a bug")

    if {"inference_type", "video_type"} <= set(frame.columns):
        counts = category_labels(frame).value_counts()
        print(f"  categories: {counts.to_dict()}")
        absent = [name for name in CATEGORIES if counts.get(name, 0) == 0]
        if absent:
            problems.append(f"categories {absent} are empty -- the evaluator reports no average")
    else:
        problems.append("missing 'inference_type'/'video_type'; the evaluator needs both to "
                        "categorize, and crashes on a null inference_type")

    if "question" in frame.columns:
        units = frame["question"].astype(str).str.extract(UNIT_PATTERN, expand=False)
        print(f"  requested units: {units.value_counts(dropna=False).to_dict()}")
        unknown = int(units.isna().sum())
        if unknown:
            problems.append(f"{unknown} questions with no parseable trailing unit -- confirm by "
                            "hand that these predictions use the intended unit")

    finite = values[values.notna() & (values > 0)]
    if not finite.empty:
        print(f"  prediction range: {finite.min():.4g} to {finite.max():.4g} "
              f"(median {finite.median():.4g})")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--expect-rows", type=int, default=None,
                        help="3289 for the test split, 159 for validation")
    args = parser.parse_args()

    problems = check(args.csv, args.expect_rows)
    if problems:
        print(f"\nFAIL -- {len(problems)} problem(s), do not spend a submission on this:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nOK -- safe to submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
