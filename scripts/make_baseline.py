"""Build a zero-vision baseline prediction set from the validation split's ground truth.

No GPU, no detection, no solver. Every test row is answered with the median *validation* answer for
its scored category and requested unit -- a "what does a typical answer to this kind of question
look like" guess and nothing more.

The point is not the score. It is that the submission portal scores on upload and returns a
per-category MRA breakdown, so this establishes three things for free before any GPU spend:

* the upload path works end to end, months before the deadline rather than at crunch time;
* a floor that every vision-backed submission has to beat to be worth its compute;
* the per-category structure of the test split, which the 159-row validation set only hints at.

Grouping is by ``(category, unit)`` for the same reason ``make_submission.fallback_values`` does it:
metres, m/s and m/s^2 live on completely different scales, so one global number would be wrong for
most rows. ``ground_truth_posterior`` is already expressed in the unit the question asks for -- the
official scorer does no unit conversion -- so a median taken within a unit group needs no rescaling.

Usage:
    py -3.12 scripts/make_baseline.py --out baseline_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy.parsing import parse_output_unit  # noqa: E402
from quantiphy.scoring import category_labels  # noqa: E402

TEMPLATE = ROOT / "data" / "fixtures" / "quantiphy_submission_template.csv"
VALIDATION = ROOT / "data" / "fixtures" / "gpt-5.1_validation.csv"


def _keys(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """The two grouping keys: scored category (S2/D2/S3/D3) and the question's output unit."""
    units = frame["question"].map(parse_output_unit).fillna("?")
    return category_labels(frame), units


def baseline_values(template: pd.DataFrame, validation: pd.DataFrame) -> pd.Series:
    """A per-row median validation answer, widening the group until one has data.

    The ladder is ``(category, unit)`` -> ``unit`` -> ``category`` -> everything. Unit comes before
    category because it is the scale-defining key: a category median mixing metres with m/s^2 is
    meaningless, while a unit median pooled across categories is at least dimensionally right.
    """
    truth = pd.to_numeric(validation["ground_truth_posterior"], errors="coerce")
    usable = truth[truth.notna() & (truth > 0)]
    if usable.empty:
        raise ValueError(f"no usable ground truth in {VALIDATION}")

    val_category, val_unit = _keys(validation)
    by_pair = usable.groupby([val_category[usable.index], val_unit[usable.index]]).median()
    by_unit = usable.groupby(val_unit[usable.index]).median()
    by_category = usable.groupby(val_category[usable.index]).median()
    overall = float(usable.median())

    category, unit = _keys(template)
    pair_index = pd.MultiIndex.from_arrays([category, unit])
    return (pd.Series(by_pair.reindex(pair_index).to_numpy(), index=template.index)
            .fillna(unit.map(by_unit))
            .fillna(category.map(by_category))
            .fillna(overall)
            .astype(float))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--validation", type=Path, default=VALIDATION)
    parser.add_argument("--shrink", type=float, default=1.0,
                       help="multiplier on every baseline value; <1 trades overshoots for "
                            "undershoots, which MRA charges asymmetrically")
    args = parser.parse_args()

    template = pd.read_csv(args.template, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    validation = pd.read_csv(args.validation, encoding="utf-8-sig")

    # Note `make_submission.py --shrink` cannot do this job: it scales only *fallback* rows, and
    # every row here is a prediction, so it would be a silent no-op.
    values = baseline_values(template, validation) * args.shrink
    predictions = pd.DataFrame({"id": template["id"], "parsed_value": values})
    predictions.to_csv(args.out, index=False, encoding="utf-8")

    category, unit = _keys(template)
    print(f"{len(predictions)} baseline predictions -> {args.out}")
    for name, group in values.groupby(category):
        print(f"  {name}: n={len(group)} median {group.median():.4g}")
    print(f"  distinct values: {values.nunique()} across {len(set(zip(category, unit)))} groups")
    print(f"\nnow run: py -3.12 scripts/make_submission.py {args.out} --out sub_baseline.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
