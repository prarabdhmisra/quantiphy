"""Find the confidence threshold above which the geometric solver beats the fallback constant.

This is the highest-value knob on the board, and the reason is measured rather than argued. A
zero-vision constant -- the median validation answer per (category, unit) -- scored **0.364** on the
real test split. The solver, on the rows it actually solves, scored **0.312** on the last replay,
with 7 of 16 rows at exactly zero. So firing it everywhere *loses* points.

But the solver is not uniformly bad, it is bimodal: its good rows score 0.70-1.00, far above the
constant, and its bad rows score 0.00. All of the value is therefore in knowing which is which, and
on that 16-row sample an oracle that fired only where the solver wins scored **0.496**. That gap,
~+13 points, is worth more than every other pending item combined.

``solve_row`` already takes ``min_confidence`` and still defaults it to 0.0. This script picks the
number to put there, by replaying a finished validation run -- which has ground truth -- and scoring
every threshold on the actual metric.

Two guards against fooling ourselves, both demanded by past sessions' mistakes:

* **Leave-one-out for the constant.** No row may inform the fallback it is compared against, or the
  constant looks better than it is and the gate looks worse.
* **A paired bootstrap on the per-row difference.** Per-category shrinkage once measured +0.03
  in-sample and -0.02 leave-one-out on this very split. A gain whose CI includes zero is not a gain.

Usage:
    py -3.12 scripts/fit_confidence_gate.py [--run validation-distance-fix1]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy.parsing import build_request, parse_output_unit  # noqa: E402
from quantiphy.scoring import CATEGORIES, category_labels  # noqa: E402
from quantiphy.solver import solve_row  # noqa: E402

#: The official threshold set, per the code (the paper's Appendix A.2 set is a typo -- see the
#: "Do not re-derive" list).
THRESHOLDS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])


def _load_replay_module():
    """Reuse ``replay_cache.CachedBackend`` rather than reimplementing the cache reader."""
    path = ROOT / "scripts" / "replay_cache.py"
    spec = importlib.util.spec_from_file_location("replay_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def mra_per_row(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Per-row MRA. A non-finite or non-positive prediction scores a hard zero, as the scorer does."""
    pred = np.asarray(pred, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative = np.abs(pred - truth) / np.abs(truth)
        scored = (relative[:, None] < (1.0 - THRESHOLDS)[None, :]).mean(axis=1)
    return np.where(np.isfinite(pred) & (pred > 0), scored, 0.0)


def macro_mra(scores: np.ndarray, categories: np.ndarray) -> float:
    """The official score: mean of the four per-category means, not the row mean."""
    present = [categories == name for name in CATEGORIES]
    return float(np.mean([scores[mask].mean() for mask in present if mask.any()]))


def loo_constant(truth: np.ndarray, categories: np.ndarray, units: np.ndarray) -> np.ndarray:
    """The zero-vision fallback each row would have received, fitted without that row.

    Same widening ladder as ``scripts/make_baseline.py``: (category, unit) -> unit -> category ->
    everything. Leave-one-out because this is the thing the gate is judged against.
    """
    pair = np.array([f"{c}|{u}" for c, u in zip(categories, units)])
    out = np.empty(len(truth))
    for i in range(len(truth)):
        others = np.ones(len(truth), dtype=bool)
        others[i] = False
        for selector in (pair == pair[i], units == units[i], categories == categories[i],
                         np.ones(len(truth), dtype=bool)):
            pool = truth[others & selector]
            pool = pool[pool > 0]
            if pool.size:
                out[i] = np.median(pool)
                break
    return out


def paired_bootstrap(gain: np.ndarray, draws: int = 10000, seed: int = 0) -> tuple[float, float]:
    """95% CI on the mean per-row gain. Resamples rows, keeping the pairing intact."""
    rng = np.random.default_rng(seed)
    means = np.array([gain[rng.integers(0, len(gain), len(gain))].mean() for _ in range(draws)])
    return tuple(np.percentile(means, [2.5, 97.5]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="validation-distance-fix1")
    parser.add_argument("--repo", default="prarabdhmisra/quantiphy-runs")
    parser.add_argument("--limit", type=int, default=159)
    args = parser.parse_args()

    replay = _load_replay_module()
    frame, cache = replay.load(args.repo, args.run, args.limit)
    backend = replay.CachedBackend(cache)

    truth = frame["ground_truth_posterior"].astype(float).to_numpy()
    categories = category_labels(frame).to_numpy()
    units = frame["question"].map(parse_output_unit).fillna("?").to_numpy()

    solver = np.full(len(frame), np.nan)
    confidence = np.zeros(len(frame))
    for position, (_, row) in enumerate(frame.iterrows()):
        answer = solve_row(build_request(row), backend, f"cache/{row['video_id']}.mp4")
        if answer.solved:
            solver[position] = answer.value
            confidence[position] = answer.confidence

    constant = loo_constant(truth, categories, units)
    solved = np.isfinite(solver)
    constant_scores = mra_per_row(constant, truth)

    print(f"{solved.sum()}/{len(frame)} rows solved by the solver; "
          f"{len(backend.misses)} cache misses\n")
    print(f"{'gate':>8} {'fires':>6} {'macro MRA':>10} {'vs constant':>12}   verdict")
    print("-" * 62)

    base = macro_mra(constant_scores, categories)
    print(f"{'never':>8} {0:6d} {base:10.4f} {0.0:+12.4f}   the 0.364 baseline, reproduced offline")

    best = (None, -1.0)
    for gate in [0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        fire = solved & (confidence >= gate)
        blended = np.where(fire, solver, constant)
        scores = mra_per_row(blended, truth)
        score = macro_mra(scores, categories)
        low, high = paired_bootstrap(scores - constant_scores)
        verdict = ("BEATS the constant" if low > 0 else
                   "worse" if high < 0 else "not established (CI spans 0)")
        print(f"{gate:8.2f} {int(fire.sum()):6d} {score:10.4f} {score - base:+12.4f}   {verdict}")
        if score > best[1]:
            best = (gate, score)

    # The ceiling: fire only where the solver actually wins. Unreachable (it peeks at the truth),
    # but it bounds what any gate can buy and says whether confidence is the right signal at all.
    solver_scores = mra_per_row(np.where(solved, solver, constant), truth)
    oracle = macro_mra(np.maximum(solver_scores, constant_scores), categories)
    print(f"\n{'oracle':>8} {'':6} {oracle:10.4f} {oracle - base:+12.4f}   ceiling for any gate")
    print(f"\nbest gate {best[0]} -> {best[1]:.4f} (constant {base:.4f}, oracle {oracle:.4f})")
    print(f"gate captures {100 * (best[1] - base) / (oracle - base):.0f}% of the available headroom"
          if oracle > base else "no headroom: the solver never beats the constant here")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
