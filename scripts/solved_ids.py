"""Record which rows a solver run actually answered, as a committed artifact.

A solver submission carries two populations that look identical in the CSV: rows the solver measured,
and rows it declined and `make_submission.py` filled from a fallback constant. Telling them apart
after the fact is guesswork -- the fallback is a single repeated value, so a solved row that happens
to land on it is indistinguishable. The run's own ``predictions.csv`` knows: ``method`` is ``none``
exactly when the solver declined.

That distinction is worth a file rather than a lambda, because every per-row experiment from here
selects on it, and the shard predictions live in a Hub cache that holds more than one revision --
globbing it returns rows from every snapshot at once.

Ids are emitted, not row indices: ``id == row_index + 1`` throughout this project, and the submission
template is keyed by ``id``.

Usage:
    py -3.12 scripts/solved_ids.py --run test-solver-v1 --shards 4 \\
        --out data/probes/solved-ids-test-solver-v1.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from merge_shards import DEFAULT_REPO, EXPECTED_ROWS, load_shards  # noqa: E402

DECLINED = "none"


def solved_ids(merged: pd.DataFrame) -> pd.DataFrame:
    """The ``id``/``method`` pairs for rows the solver measured, ordered by id.

    Refuses a frame the shards did not reassemble cleanly. A duplicate row index means two shards
    overlapped, which invalidates the partition rather than one row.
    """
    for column in ("row_index", "method"):
        if column not in merged.columns:
            raise SystemExit(f"predictions carry no {column!r} column; cannot tell solved from "
                             f"declined. Re-run the job with a build that records it.")

    duplicated = int(merged["row_index"].duplicated().sum())
    if duplicated:
        raise SystemExit(f"{duplicated} duplicate row_index values -- the shards overlap, so the "
                         f"SHARD arithmetic is wrong. Refusing to emit an id list.")

    method = merged["method"].fillna(DECLINED).astype(str).str.strip()
    solved = merged.loc[method.ne(DECLINED) & method.ne(""), ["row_index"]].copy()
    solved["id"] = solved["row_index"].astype(int) + 1
    solved["method"] = method.loc[solved.index]
    return solved[["id", "method"]].sort_values("id").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run name without the -shard<k> suffix")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--expect-rows", type=int, default=EXPECTED_ROWS)
    args = parser.parse_args()

    merged = load_shards(args.repo, args.run, args.shards)
    if len(merged) != args.expect_rows:
        # Loud rather than fatal: a short run still yields a usable id list, but every row it never
        # reached is silently a declined row, and that is a different experiment.
        print(f"WARNING: {len(merged)} rows, expected {args.expect_rows} -- "
              f"{args.expect_rows - len(merged)} rows were never attempted and will read as "
              f"declined.")

    ids = solved_ids(merged)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    ids.to_csv(args.out, index=False, encoding="utf-8")

    print(f"\n{len(ids)} solved of {len(merged)} rows ({len(ids) / max(len(merged), 1):.1%})")
    for name, count in ids["method"].value_counts().items():
        print(f"  {count:5d}  {name}")
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
