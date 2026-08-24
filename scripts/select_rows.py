"""Compose a submission by choosing, per *row*, which measured source to use.

``select_sources.py`` selects per category, and that is exact: the portal scores the four categories
independently and reports each, so the macro of a per-category mix of two scored submissions is
arithmetic. This script selects per row, which is *not* derivable from any reading -- it is a new
source and it costs a slot to measure. Use it only when the rows being separated are known to differ
in kind, not to chase a score.

The case it was built for, 2026-08-24: ``make_submission.py`` fills declined rows from the median of
the predictions file's own solved values, so a solver submission's "fallback" is the solver's own
overshoot re-applied to every row it declined -- 1,951 of 3,289 rows, at 2.74 where the
validation-derived constant says 1.25 for ``D2|meters``. Overlaying the solver only where it actually
measured, on top of the constant baseline, separates the solver's worth from its fallback's harm.

Usage:
    py -3.12 scripts/select_rows.py \\
        --base baseline-v3.submission.csv \\
        --overlay solver-v1.submission.csv \\
        --overlay-ids data/probes/solved-ids-test-solver-v1.csv \\
        --out mix-v2.submission.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy.scoring import CATEGORIES, category_labels  # noqa: E402


def read_submission(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def read_ids(path: Path) -> list[int]:
    """The id column of an id list, validated.

    An empty list is refused rather than passed through: it would produce a byte-identical copy of
    the base, which validates cleanly, uploads cleanly, and spends a slot re-measuring something
    already known.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "id" not in frame.columns:
        raise SystemExit(f"{path} has no 'id' column; columns are {list(frame.columns)}")
    ids = pd.to_numeric(frame["id"], errors="coerce")
    if ids.isna().any():
        raise SystemExit(f"{path} has {int(ids.isna().sum())} non-numeric ids")
    duplicated = int(ids.duplicated().sum())
    if duplicated:
        raise SystemExit(f"{path} lists {duplicated} duplicate ids")
    if ids.empty:
        raise SystemExit(f"{path} is empty; the result would be a copy of --base")
    return [int(value) for value in ids]


def compose(base: pd.DataFrame, overlay: pd.DataFrame, ids: list[int]) -> pd.DataFrame:
    """``base`` everywhere, with ``overlay``'s ``parsed_value`` on the listed ids.

    Only ``parsed_value`` moves. Every metadata column comes from the base, so the result stays
    row-for-row identical to the organizers' template no matter what the overlay carries.
    """
    if not (base["id"].to_numpy() == overlay["id"].to_numpy()).all():
        raise SystemExit("base and overlay have different id orderings; sources must be row-aligned")

    known = set(base["id"].astype(int))
    unknown = sorted(set(ids) - known)
    if unknown:
        raise SystemExit(f"{len(unknown)} ids are not in the base submission, first few: "
                         f"{unknown[:5]}")

    out = base.copy()
    rows = base["id"].astype(int).isin(ids).to_numpy()
    out.loc[rows, "parsed_value"] = overlay.loc[rows, "parsed_value"].to_numpy()
    return out


def manifest(base: Path, overlay: Path, out_path: Path, frame: pd.DataFrame,
             ids: list[int]) -> dict:
    labels = category_labels(frame)
    selected = frame["id"].astype(int).isin(ids).to_numpy()
    return {
        "base": str(base),
        "overlay": str(overlay),
        "submission": str(out_path),
        "rows": len(ids),
        "by_category": {
            name: {
                "from_overlay": int((selected & (labels == name).to_numpy()).sum()),
                "category_rows": int((labels == name).sum()),
            }
            for name in CATEGORIES
        },
        "ids": ids,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--overlay-ids", type=Path, required=True,
                        help="CSV with an 'id' column: the rows to take from --overlay")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    base = read_submission(args.base)
    overlay = read_submission(args.overlay)
    ids = read_ids(args.overlay_ids)

    out = compose(base, overlay, ids)
    out.to_csv(args.out, index=False, encoding="utf-8")

    record = manifest(args.base, args.overlay, args.out, out, ids)
    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"{args.out}  ({len(out)} rows)")
    print(f"  {len(ids)} rows from {args.overlay.name}, "
          f"{len(out) - len(ids)} from {args.base.name}")
    for name in CATEGORIES:
        counts = record["by_category"][name]
        print(f"  {name}: {counts['from_overlay']:4d} of {counts['category_rows']:4d} "
              f"from {args.overlay.name}")
    print(f"  -> {manifest_path}")
    print(f"\nnow run: py -3.12 scripts/validate_submission.py {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
