"""Pre-flight check for a QuantiPhy submission CSV.

The daily quota is 3 *scored* submissions per team per UTC day, and there is no server-side dry
run. Every failure mode below silently costs points rather than erroring, so check locally first.

Usage:
    py -3.12 scripts/validate_submission.py submission.csv [--expect-rows 3289]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy.parsing import parse_output_unit  # noqa: E402
from quantiphy.scoring import CATEGORIES, category_labels  # noqa: E402

MAX_BYTES = 2 * 1024 * 1024

TEMPLATE = ROOT / "data" / "fixtures" / "quantiphy_submission_template.csv"

#: Metadata the evaluator joins on. A submission that reorders or rewrites any of these still looks
#: well-formed but scores near zero, so it is checked against the official template row-for-row.
TEMPLATE_COLUMNS = ("id", "video_id", "video_source", "video_type", "fps", "inference_type",
                    "question", "ground_truth_prior", "depth_info")


def as_text(column: pd.Series) -> pd.Series:
    """Compare text the same way regardless of how pandas read the file.

    An empty CSV cell arrives as NaN with default parsing and as ``""`` with ``keep_default_na``
    off; both mean "blank", and conflating them is the difference between a real misalignment and
    1,741 phantom ones.
    """
    return column.fillna("").astype(str).str.strip()


def check_ids(frame: pd.DataFrame, expect_rows: int | None) -> list[str]:
    """`id` must be 1..N, every value present exactly once, and already in ascending order.

    The organizers' template ships it that way and the upload page says to leave it unchanged, so
    anything else is a bug on our side rather than a format we should tolerate.
    """
    if "id" not in frame.columns:
        return ["missing the 'id' column; the template ships it and the portal joins on it"]

    problems: list[str] = []
    ids = pd.to_numeric(frame["id"], errors="coerce")
    if ids.isna().any():
        problems.append(f"{int(ids.isna().sum())} non-numeric ids")
        return problems
    if (ids % 1 != 0).any():
        problems.append("ids must be whole numbers")
    duplicated = int(ids.duplicated().sum())
    if duplicated:
        problems.append(f"{duplicated} duplicated ids")
    if not ids.is_monotonic_increasing:
        problems.append("ids are not in ascending order; keep the template's row order")

    expected = expect_rows if expect_rows is not None else len(frame)
    missing = set(range(1, expected + 1)) - set(ids.astype(int))
    if missing:
        sample = sorted(missing)[:5]
        problems.append(f"{len(missing)} ids missing from 1..{expected}, e.g. {sample}")
    return problems


def check_against_template(frame: pd.DataFrame, template_path: Path) -> list[str]:
    """Every metadata column must match the official template row-for-row."""
    if not template_path.exists():
        return [f"template fixture not found at {template_path}; cannot verify row alignment"]

    template = pd.read_csv(template_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if len(frame) != len(template):
        return [f"{len(frame)} rows against the template's {len(template)}; cannot compare rows"]

    problems: list[str] = []
    for column in TEMPLATE_COLUMNS:
        if column not in frame.columns:
            problems.append(f"missing template column '{column}'")
            continue
        ours, theirs = as_text(frame[column]), as_text(template[column])
        differing = int((ours != theirs).sum())
        if differing:
            first = int((ours != theirs).idxmax())
            problems.append(f"column '{column}' differs from the template in {differing} rows "
                            f"(first at row {first}); rows may have been reordered or dropped")
    return problems


def check(path: Path, expect_rows: int | None, template: Path | None = TEMPLATE) -> list[str]:
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

    problems.extend(check_ids(frame, expect_rows))
    if template is not None:
        problems.extend(check_against_template(frame, template))

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
        # The scorer performs no unit conversion, so the answer must already be in the unit the
        # question names. A handful of questions name none at all -- that is the benchmark's own
        # ambiguity, not a defect in our file, so it is reported and not treated as a failure.
        units = as_text(frame["question"]).map(parse_output_unit)
        print(f"  requested units: {units.value_counts(dropna=False).to_dict()}")
        unstated = frame.loc[units.isna(), "id"] if "id" in frame.columns else units[units.isna()]
        if len(unstated):
            print(f"  WARNING: {len(unstated)} questions state no unit; the solver assumes SI. "
                  f"ids: {unstated.tolist()[:10]}")

    finite = values[values.notna() & (values > 0)]
    if not finite.empty:
        print(f"  prediction range: {finite.min():.4g} to {finite.max():.4g} "
              f"(median {finite.median():.4g})")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--expect-rows", type=int, default=3289,
                        help="3289 for the test split, 159 for validation")
    parser.add_argument("--no-template", action="store_true",
                        help="skip the row-for-row template comparison (validation split)")
    args = parser.parse_args()

    problems = check(args.csv, args.expect_rows, None if args.no_template else TEMPLATE)
    if problems:
        print(f"\nFAIL -- {len(problems)} problem(s), do not spend a submission on this:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nOK -- safe to submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
