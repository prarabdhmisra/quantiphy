"""How far a replay's answers sit from the zero-vision constant, sliced by cause.

The test split has no ground truth, so nothing offline can score a replay. What *can* be measured is
each answer against the constant that row would otherwise receive -- and that constant is not
arbitrary, it is a submission we have scored (``baseline-v3``, macro 0.365). So an answer near it is
a coin flip worth a slot, and one 100x away is a guaranteed zero, strictly worse than declining.

This is the instrument that chose the trusted band on 2026-08-25, diagnosed D3 as a ~3x overshoot
rising with the prior's derivative order, and localized the 3D defect to LENGTH questions. All three
were computed ad hoc in a throwaway session and could not be re-run. They are the basis of every
lever now on the board, so they belong in a script.

**The reference is not truth.** Where the solver already beats the constant in a category (S2, D2,
S3) a large ratio is weak evidence; where the constant beats the solver (D3, 0.396 against 0.315) it
is strong. Read the ``ratio`` column with the category's measured standing in mind, which is why
that standing is printed beside it.

Usage:
    py -3.12 scripts/disagreement.py replay-band30inf.csv
    py -3.12 scripts/disagreement.py replay-band30inf.csv --by category,prior
    py -3.12 scripts/disagreement.py replay-band30inf.csv --by category,dimension \\
        --method-prefix geometric-3d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy import units  # noqa: E402
from quantiphy.parsing import parse_output_unit, parse_question  # noqa: E402
from quantiphy.scoring import CATEGORIES, category_labels  # noqa: E402

DEFAULT_BASELINE = ROOT / "baseline_predictions.csv"
DECLINED = "none"

#: Measured per-category MRA of the two sources, so the printout can say which reference is the
#: better instrument in each category rather than leaving the reader to remember. From the portal:
#: ``baseline-v3`` 2026-08-23 and ``solver-v2`` 2026-08-25.
CONSTANT_MRA = {"S2": 0.337, "D2": 0.315, "S3": 0.410, "D3": 0.396}
SOLVER_V2_MRA = {"S2": 0.430, "D2": 0.461, "S3": 0.465, "D3": 0.315}

#: Slice keys. ``prior`` is ``video_type[0]`` -- S = Size, V = Velocity, A = Acceleration, which the
#: dataset card confirms *is* the prior's derivative order, and that order predicts the error.
KEYS = ("category", "prior", "method", "dimension", "unit", "route")


def add_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach every slice key as a column. Cheap enough to always do all of them."""
    out = frame.copy()
    out["category"] = category_labels(out)
    out["prior"] = out["video_type"].astype(str).str[0]
    out["method"] = out["method"].fillna(DECLINED).astype(str).str.strip().replace("", DECLINED)
    # The route is the method without its modifier suffixes: `geometric-3d+radial+target-from-prior`
    # and `geometric-3d` are the same code path, and splitting them fragments a slice the way
    # counting raw decline *strings* once hid the trusted band inside ~300 distinct messages.
    out["route"] = out["method"].str.split("+").str[0]
    out["unit"] = out["question"].map(parse_output_unit).fillna("?")
    out["dimension"] = out["unit"].map(
        lambda unit: (units.lookup(unit) or (None, None))[1]) .fillna("?")
    # A question whose unit is unparseable still has keyword evidence; without this those rows all
    # collapse into "?" and a dimension slice silently loses them.
    missing = out["dimension"] == "?"
    if missing.any():
        out.loc[missing, "dimension"] = (
            out.loc[missing, "question"].map(lambda q: parse_question(q)[1]).fillna("?"))
    return out


def ratios(replay: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Answered rows only, with ``ratio = answer / constant`` joined on ``id``.

    ``id == row_index + 1`` throughout this project. Declined rows carry no answer, so including
    them would compare the constant against itself and pull every median toward 1.0.
    """
    if "row_index" not in replay.columns:
        raise SystemExit("replay has no 'row_index' column; it is not a replay_cache output")
    frame = add_keys(replay)
    frame["id"] = frame["row_index"].astype(int) + 1

    constants = baseline.set_index("id")["parsed_value"].astype(float)
    unknown = ~frame["id"].isin(constants.index)
    if unknown.any():
        raise SystemExit(f"{int(unknown.sum())} replay ids are absent from the baseline; the two "
                         f"files describe different splits")

    answered = frame.loc[frame["method"].ne(DECLINED) & frame["parsed_value"].notna()].copy()
    answered["const"] = answered["id"].map(constants)
    usable = answered["const"].gt(0) & np.isfinite(answered["parsed_value"].astype(float))
    if not usable.all():
        print(f"note: {int((~usable).sum())} answered rows dropped "
              f"(non-finite answer or non-positive constant)")
    answered = answered.loc[usable]
    answered["ratio"] = answered["parsed_value"].astype(float) / answered["const"]
    return answered


def table(answered: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Per-slice median ratio and the tail fractions that decide whether a slice is shippable."""
    def summarise(group: pd.DataFrame) -> pd.Series:
        ratio = group["ratio"].to_numpy()
        fold = np.maximum(ratio, 1.0 / ratio)          # distance from 1.0 in either direction
        return pd.Series({
            "n": len(ratio),
            "median": float(np.median(ratio)),
            "within2x": float((fold <= 2).mean()),
            "within10x": float((fold <= 10).mean()),
            "over100x": float((fold > 100).mean()),
        })

    return (answered.groupby(by, dropna=False, observed=True)
            .apply(summarise, include_groups=False)
            .reset_index()
            .sort_values("n", ascending=False))


def render(result: pd.DataFrame, by: list[str]) -> None:
    widths = {key: max(len(key), int(result[key].astype(str).str.len().max())) for key in by}
    header = "  ".join(f"{key:>{widths[key]}}" for key in by)
    print(f"\n{header}  {'n':>5} {'median':>8} {'<2x':>7} {'<10x':>7} {'>100x':>7}")
    for _, row in result.iterrows():
        cells = "  ".join(f"{str(row[key]):>{widths[key]}}" for key in by)
        print(f"{cells}  {int(row['n']):5d} {row['median']:8.3f} "
              f"{row['within2x']:6.1%} {row['within10x']:6.1%} {row['over100x']:6.1%}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("replay", type=Path, help="replay_cache.py --out predictions CSV")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE,
                        help="zero-vision constants from make_baseline.py")
    parser.add_argument("--by", default="category,prior",
                        help=f"comma-separated slice keys from {list(KEYS)}")
    parser.add_argument("--method-prefix", default=None,
                        help="restrict to methods starting with this, e.g. geometric-3d")
    parser.add_argument("--min-n", type=int, default=1,
                        help="hide slices thinner than this; a median over 5 rows is noise")
    args = parser.parse_args()

    by = [key.strip() for key in args.by.split(",") if key.strip()]
    unknown = [key for key in by if key not in KEYS]
    if unknown:
        raise SystemExit(f"unknown slice key(s) {unknown}; choose from {list(KEYS)}")

    answered = ratios(pd.read_csv(args.replay), pd.read_csv(args.baseline))
    if args.method_prefix:
        answered = answered.loc[answered["method"].str.startswith(args.method_prefix)]
        if answered.empty:
            raise SystemExit(f"no answered rows with method starting {args.method_prefix!r}")

    print(f"{len(answered)} answered rows against {args.baseline.name}")
    print("\nreference standing (measured on the portal), per category:")
    print(f"  {'cat':>4} {'constant':>9} {'solver-v2':>10}   better instrument")
    for category in CATEGORIES:
        constant, solver = CONSTANT_MRA[category], SOLVER_V2_MRA[category]
        better = "constant" if constant > solver else "solver"
        print(f"  {category:>4} {constant:9.3f} {solver:10.3f}   {better}")

    result = table(answered, by)
    render(result.loc[result["n"] >= args.min_n], by)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
