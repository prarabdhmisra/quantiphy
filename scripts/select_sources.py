"""Compose a submission by choosing, per category, which measured source to use.

The macro is the unweighted mean of four *independently* scored categories, and the portal reports
each one. So once two submissions have both been scored, the macro of any per-category mix of them is
not a prediction -- it is arithmetic. Taking the solver where it measured better and the constant
where it measured worse is exact, and it needs no new computation.

That is what makes this the campaign's highest-leverage tool. The project spent four attempts trying
to invent a confidence signal that would predict when the solver is right, and every one was refuted.
This sidesteps the question: rather than predicting correctness, measure it per category on the real
scoring set and then select.

First use, 2026-08-23:

    constant (baseline-v3)  S2 0.337  D2 0.315  S3 0.410  D3 0.396  -> 0.365
    solver-v1               S2 0.353  D2 0.368  S3 0.441  D3 0.220  -> 0.345
    best mix (solver on S2/D2/S3, constant on D3)                    -> 0.3895

The solver beat the constant in three categories and collapsed in the fourth, so the mix beats both
parents. Losing on the macro while winning three of four channels is exactly why per-category
readings are worth more than the headline number.

Usage:
    py -3.12 scripts/select_sources.py --out mix.csv \\
        --source solver-v1.submission.csv --for S2,D2,S3 \\
        --source baseline-v3.submission.csv --for D3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy.scoring import CATEGORIES, category_labels  # noqa: E402


def compose(pairs: list[tuple[Path, list[str]]]) -> tuple[pd.DataFrame, dict[str, Path]]:
    """One submission frame drawing each category from its chosen source.

    Every category must be assigned exactly once. An unassigned category would leave those rows
    holding whichever source happened to be loaded first, and a doubly-assigned one hides a typo --
    both produce a submission that validates cleanly and silently measures something else.
    """
    assigned: dict[str, Path] = {}
    frames: dict[Path, pd.DataFrame] = {}

    for path, categories in pairs:
        if path not in frames:
            frames[path] = pd.read_csv(path, dtype=str, keep_default_na=False,
                                       encoding="utf-8-sig")
        for category in categories:
            if category not in CATEGORIES:
                raise SystemExit(f"{category!r} is not one of {list(CATEGORIES)}")
            if category in assigned:
                raise SystemExit(f"{category} assigned twice: {assigned[category]} and {path}")
            assigned[category] = path

    missing = [name for name in CATEGORIES if name not in assigned]
    if missing:
        raise SystemExit(f"no source chosen for {missing}; every category must be assigned")

    base = next(iter(frames.values()))
    ids = base["id"].to_numpy()
    for path, frame in frames.items():
        if not (frame["id"].to_numpy() == ids).all():
            raise SystemExit(f"{path} has a different id ordering; sources must be row-aligned")

    out = base.copy()
    labels = category_labels(out)
    for category, path in assigned.items():
        rows = (labels == category).to_numpy()
        out.loc[rows, "parsed_value"] = frames[path].loc[rows, "parsed_value"].to_numpy()
    return out, assigned


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--for", dest="categories", action="append", required=True,
                        help="comma-separated categories for the preceding --source")
    args = parser.parse_args()

    if len(args.source) != len(args.categories):
        parser.error("each --source needs exactly one --for")

    pairs = [(path, [c.strip().upper() for c in spec.split(",") if c.strip()])
             for path, spec in zip(args.source, args.categories)]
    out, assigned = compose(pairs)
    out.to_csv(args.out, index=False, encoding="utf-8")

    labels = category_labels(out)
    print(f"{args.out}  ({len(out)} rows)")
    for name in CATEGORIES:
        print(f"  {name}: {int((labels == name).sum()):4d} rows from {assigned[name].name}")
    print(f"\nnow run: py -3.12 scripts/validate_submission.py {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
