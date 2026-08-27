"""Revert solver answers that disagree wildly with the constant, per category.

Under MRA a prediction at 1.9x truth scores exactly zero, so an answer 10x away from a reference
that itself scores ~0.37 is not a noisy answer -- it is a guaranteed zero, and the constant it
displaced would have earned partial credit. Reverting those rows costs the solver nothing it was
going to collect.

**This is not the confidence gate, which was refuted four times.** That tried to predict correctness
from the detector's own confidence, and detector confidence carries no such signal. This uses an
external reference -- the zero-vision constant, a submission scored at 0.365 -- and asks only
"do these two disagree by more than a factor a correct answer could survive". No prediction of
correctness is involved, which is the same reason per-category *selection* works where prediction
failed.

Rows are declined rather than overwritten: ``parsed_value`` is cleared and ``method`` set to
``none``, so ``make_submission.py --fallback-from`` fills them through the ladder every other
declined row already uses. Nothing here invents a number.

Thresholds are per category because the two instruments do not rank the same way everywhere. Where
the solver is the better instrument (S2, D2, S3 on the portal) a tight clip throws away rows that
were earning; where the constant is better (D3, 0.396 against the solver's 0.315) clipping is closer
to free. A clip at 1.0 degenerates to the pure constant, which is exactly the champion's D3 channel
-- so D3's clip curve is bounded below by a number already measured.

Usage:
    py -3.12 scripts/clip_disagreement.py replay-band30inf.csv --out clipped.csv \\
        --clip S2=20 --clip D2=10 --clip S3=10 --clip D3=5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy.scoring import CATEGORIES, category_labels  # noqa: E402

DEFAULT_BASELINE = ROOT / "baseline_predictions.csv"
DECLINED = "none"
CLIP_REASON = "disagrees with the constant by more than {threshold:g}x"


def parse_clip(spec: str) -> tuple[str, float]:
    """``"D3=5"`` -> ``("D3", 5.0)``, with the guards that keep a reading interpretable."""
    category, separator, raw = spec.partition("=")
    if not separator:
        raise SystemExit(f"{spec!r}: expected CATEGORY=threshold")
    category = category.strip()
    if category not in CATEGORIES:
        raise SystemExit(f"{spec!r}: {category!r} is not one of {list(CATEGORIES)}")
    try:
        threshold = float(raw)
    except ValueError:
        raise SystemExit(f"{spec!r}: {raw!r} is not a number") from None
    if threshold <= 1.0:
        # At exactly 1.0 every answered row is reverted and the result is the pure constant, which
        # is already measured. Below 1.0 the comparison is meaningless. Either way the slot buys
        # nothing, so refuse rather than emit a submission that looks new.
        raise SystemExit(f"{spec!r}: threshold must exceed 1.0; at 1.0 this is just the constant")
    return category, threshold


def clip(predictions: pd.DataFrame, constants: pd.Series,
         thresholds: dict[str, float]) -> tuple[pd.DataFrame, list[dict]]:
    """Decline every answered row whose fold-disagreement with the constant exceeds its threshold.

    Fold distance is symmetric -- ``max(pred/const, const/pred)`` -- because a 10x undershoot and a
    10x overshoot are both zeros, even though the metric punishes them differently on the way there.
    """
    out = predictions.copy()
    out["method"] = out["method"].fillna(DECLINED).astype(str).str.strip().replace("", DECLINED)
    labels = category_labels(out)
    value = pd.to_numeric(out["parsed_value"], errors="coerce")
    const = out["row_index"].astype(int).add(1).map(constants)

    answered = out["method"].ne(DECLINED) & value.notna() & const.gt(0)
    fold = pd.Series(np.nan, index=out.index)
    fold.loc[answered] = np.maximum(value[answered] / const[answered],
                                    const[answered] / value[answered])

    manifest = []
    for category, threshold in thresholds.items():
        selected = answered & (labels == category) & (fold > threshold)
        rows = int(selected.sum())
        if not rows:
            raise SystemExit(f"{category}={threshold:g} clips no rows; the reading would be a "
                             f"byte-identical copy and the slot would be wasted")
        out.loc[selected, "parsed_value"] = np.nan
        out.loc[selected, "method"] = DECLINED
        out.loc[selected, "reason"] = CLIP_REASON.format(threshold=threshold)
        manifest.append({
            "category": category, "threshold": threshold, "rows": rows,
            "category_rows": int((labels == category).sum()),
            "answered_before": int((answered & (labels == category)).sum()),
            "ids": out.loc[selected, "row_index"].astype(int).add(1).tolist(),
        })
    return out, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("predictions", type=Path, help="replay_cache.py --out predictions CSV")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--clip", action="append", default=[], metavar="CATEGORY=threshold",
                        help="repeatable; categories left out are not clipped at all")
    args = parser.parse_args()

    if not args.clip:
        parser.error("at least one --clip is required")
    thresholds = dict(parse_clip(spec) for spec in args.clip)

    predictions = pd.read_csv(args.predictions)
    constants = pd.read_csv(args.baseline).set_index("id")["parsed_value"].astype(float)
    out, manifest = clip(predictions, constants, thresholds)
    out.to_csv(args.out, index=False, encoding="utf-8")

    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(
        {"predictions": str(args.predictions), "out": str(args.out), "clips": manifest},
        indent=2), encoding="utf-8")

    print(f"{args.out}  ({len(out)} rows)")
    for entry in manifest:
        print(f"  {entry['category']:>4}  clip >{entry['threshold']:g}x  "
              f"{entry['rows']:4d} reverted of {entry['answered_before']:4d} answered "
              f"({entry['rows'] / entry['category_rows']:5.1%} of the category)")
    print(f"manifest -> {manifest_path}")
    print(f"\nnow run: py -3.12 scripts/make_submission.py {args.out} --out <sub>.csv "
          f"--fallback-from {args.baseline.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
