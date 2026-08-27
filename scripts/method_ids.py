"""The ids of a replay's rows whose solve took a named code path, for a per-row overlay.

``solved_ids.py`` answers "which rows did the solver answer at all", which is what the 2026-08-24
fallback experiment needed. This answers the finer question every experiment since has needed:
*which rows took this particular route*. The ``method`` string already records it -- the solver
appends a marker per branch it took (``+radial``, ``+separation``, ``+prior-tangential``) -- so a
route is selectable without re-deriving it from the dataset.

Why that matters more than it sounds. On 2026-08-26 D3 was declared closed to the solver because no
threshold on *disagreement magnitude* let it beat the constant. That conclusion is sound and still
holds, and it says nothing about selecting by *route*, which is a different partition of the same
rows: in D3 ``geometric-3d+radial`` sits at 0.75x the constant with 51% of rows within 2x, while
plain ``geometric-3d`` sits at 2.29x with 27%. Feeding those to one threshold averages a good
population into a bad one.

Emits ``id`` (== ``row_index + 1``) so the output drops straight into ``select_rows.py
--overlay-ids``. Refuses an empty selection: it would compose to a byte-identical copy of the base
and spend a submission slot re-measuring something already known.

Usage:
    py -3.12 scripts/method_ids.py replay.csv --contains prior-tangential --out ids.csv
    py -3.12 scripts/method_ids.py replay.csv --method geometric-2d+separation \\
        --method geometric-3d+separation --category S3 --out ids.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy.scoring import CATEGORIES, category_labels  # noqa: E402

DECLINED = "none"


def select(replay: pd.DataFrame, *, contains: str | None = None,
           methods: tuple[str, ...] = (), categories: tuple[str, ...] = ()) -> pd.DataFrame:
    """The ``id``/``category``/``method`` rows matching the filters, ordered by id.

    ``--method`` is exact and ``--contains`` is a substring, and they are deliberately separate:
    ``geometric-3d`` as a substring also matches ``geometric-3d+radial``, which is a *different*
    population with three times the within-2x rate. Asking for one and silently getting both is how
    a route selection turns into the average it was built to avoid.
    """
    for column in ("row_index", "method"):
        if column not in replay.columns:
            raise SystemExit(f"{column!r} is missing; this is not a replay_cache output")

    frame = replay.copy()
    frame["method"] = frame["method"].fillna(DECLINED).astype(str).str.strip()
    frame["category"] = category_labels(frame)
    frame["id"] = frame["row_index"].astype(int) + 1

    # A declined row carries no answer, so overlaying it would copy the base's own value back --
    # invisible in the diff and indistinguishable from a bug in the composition.
    keep = frame["method"].ne(DECLINED) & frame["method"].ne("")
    if contains:
        keep &= frame["method"].str.contains(contains, regex=False)
    if methods:
        keep &= frame["method"].isin(methods)
    if categories:
        unknown = set(categories) - set(CATEGORIES)
        if unknown:
            raise SystemExit(f"{sorted(unknown)} are not categories; expected {list(CATEGORIES)}")
        keep &= frame["category"].isin(categories)

    selected = frame.loc[keep, ["id", "category", "method"]].sort_values("id")
    if selected.empty:
        raise SystemExit("no rows matched; the overlay would be a copy of the base and the slot "
                         "would measure nothing")
    return selected.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, help="a replay_cache.py --out predictions CSV")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--contains", help="substring of the method string")
    parser.add_argument("--method", action="append", default=[], help="exact method; repeatable")
    parser.add_argument("--category", action="append", default=[],
                        help="restrict to these categories; repeatable")
    args = parser.parse_args()

    if not args.contains and not args.method:
        parser.error("give --contains or at least one --method, or every solved row is selected")

    replay = pd.read_csv(args.replay)
    selected = select(replay, contains=args.contains, methods=tuple(args.method),
                      categories=tuple(c.strip().upper() for c in args.category))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    selected[["id"]].to_csv(args.out, index=False, encoding="utf-8")

    print(f"{len(selected)} of {len(replay)} rows -> {args.out}")
    for (category, method), count in selected.groupby(["category", "method"]).size().items():
        print(f"  {category}  {count:5d}  {method}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
