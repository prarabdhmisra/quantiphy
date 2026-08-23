"""Concatenate a sharded run's predictions into one file for ``make_submission.py``.

A sharded run leaves ``<run>-shard<k>/predictions.csv`` per shard. Each row carries its original
``row_index`` from before the split, so the shards reassemble by concatenation and the ordering of
the pieces does not matter.

The checks here exist because the failure mode is silent. A shard that died at 80% still uploads a
``partial.csv`` and a truncated ``predictions.csv``, and concatenating those yields a submission that
validates cleanly, uploads cleanly, and quietly scores hundreds of rows on the fallback constant
instead of the solver. So a missing row index is reported as a gap, and any duplicate -- which would
mean two shards overlapped and the whole sharding arithmetic is wrong -- is an error.

Unsolved rows are left blank on purpose. ``make_submission.py`` fills them from the zero-vision
fallback; emitting a zero here would score a hard zero instead.

Usage:
    py -3.12 scripts/merge_shards.py --run test-solver-v1 --shards 4 --out predictions_test.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_REPO = "prarabdhmisra/quantiphy-runs"
EXPECTED_ROWS = 3289


def load_shards(repo: str, run: str, shards: int) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    frames = []
    for index in range(1, shards + 1):
        name = f"{run}-shard{index}"
        try:
            path = hf_hub_download(repo, repo_type="dataset",
                                   filename=f"{name}/predictions.csv")
        except Exception as error:
            print(f"  shard {index}: MISSING predictions.csv ({type(error).__name__}). "
                  f"It is probably still running -- a partial.csv exists mid-run.")
            continue
        frame = pd.read_csv(path)
        print(f"  shard {index}: {len(frame)} rows, "
              f"{int(pd.to_numeric(frame['parsed_value'], errors='coerce').notna().sum())} solved")
        frames.append(frame)
    if not frames:
        raise SystemExit("no shard produced a predictions.csv")
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run name without the -shard<k> suffix")
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--expect-rows", type=int, default=EXPECTED_ROWS)
    args = parser.parse_args()

    merged = load_shards(args.repo, args.run, args.shards)

    if "row_index" not in merged.columns:
        raise SystemExit("shards carry no row_index; they cannot be reassembled safely")

    duplicated = merged["row_index"].duplicated().sum()
    if duplicated:
        raise SystemExit(f"{duplicated} duplicate row_index values -- the shards overlap, which "
                         f"means the SHARD arithmetic is wrong. Refusing to merge.")

    merged = merged.sort_values("row_index").reset_index(drop=True)
    present = set(merged["row_index"].astype(int))
    missing = sorted(set(range(args.expect_rows)) - present)

    solved = int(pd.to_numeric(merged["parsed_value"], errors="coerce").notna().sum())
    print(f"\n{len(merged)} of {args.expect_rows} rows, {solved} solved "
          f"({solved / max(len(merged), 1):.1%} of those present)")

    if missing:
        # Not fatal: make_submission fills every absent row from the fallback. But it must be loud,
        # because the result is a submission that looks complete and silently measures the constant.
        runs = []
        start = previous = missing[0]
        for index in missing[1:]:
            if index != previous + 1:
                runs.append((start, previous))
                start = index
            previous = index
        runs.append((start, previous))
        print(f"WARNING: {len(missing)} rows absent -- they will fall back to the constant, "
              f"not the solver.")
        for low, high in runs[:10]:
            print(f"    rows {low}..{high} ({high - low + 1})")
        if len(runs) > 10:
            print(f"    ... and {len(runs) - 10} more gaps")

    merged.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\n-> {args.out}")
    print(f"now run: py -3.12 scripts/make_submission.py {args.out} --out solver.submission.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
