"""Faithful reimplementation of the QuantiPhy MRA metric, plus paired significance testing.

Mirrors the official ``evaluator.py`` from github.com/Paulineli/QuantiPhy. Validated against the
organizers' published GPT-5.1 validation score (see tests/test_scoring.py).

Two things about the official metric are easy to get wrong and are load-bearing here:

* The threshold set is ``{0.1, ..., 0.9, 0.95}``. The paper's Appendix A.2 states
  ``{0.5, 0.55, ..., 0.95}`` instead; that variant does *not* reproduce the published numbers
  (0.376 vs 0.486), so the code is authoritative and the paper's appendix is a typo.
* A missing, unparseable, or zero prediction scores a hard 0 and still counts toward the
  category mean. It is never dropped. Only a missing *ground truth* excludes a row.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

#: Confidence thresholds from the official evaluator. A prediction is credited at threshold
#: ``theta`` when its relative error is strictly below ``1 - theta``.
THRESHOLDS: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)

#: The four scored categories, ``inference_type[0] + video_type[1]``. The final score is their
#: unweighted mean, so a category left unpopulated makes the total undefined.
CATEGORIES: tuple[str, ...] = ("S2", "D2", "S3", "D3")


def item_scores(predictions, ground_truth) -> np.ndarray:
    """Per-item MRA in [0, 1], in 0.1 steps.

    Predictions are taken by magnitude (the official parser applies ``abs``). NaN and zero
    predictions score 0. Items whose ground truth is missing or zero score NaN and are excluded
    downstream.
    """
    pred = np.abs(pd.to_numeric(pd.Series(predictions), errors="coerce").to_numpy(dtype=float))
    truth = pd.to_numeric(pd.Series(ground_truth), errors="coerce").to_numpy(dtype=float)

    scorable = np.isfinite(truth) & (truth != 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        relative_error = np.abs(pred - truth) / truth

    tolerances = 1.0 - np.asarray(THRESHOLDS)
    credited = (relative_error[:, None] < tolerances[None, :]).sum(axis=1) / len(THRESHOLDS)

    # NaN relative error (unparseable prediction) fails every comparison, which is already 0 --
    # but make that explicit rather than relying on NaN comparison semantics.
    credited = np.where(np.isnan(pred), 0.0, credited)
    return np.where(scorable, credited, np.nan)


def category_labels(frame: pd.DataFrame) -> pd.Series:
    """Build the ``S2``/``D2``/``S3``/``D3`` label the official evaluator scores on."""
    inference = frame["inference_type"].astype(str).str[0]
    dimension = frame["video_type"].astype(str).str[1]
    return inference + dimension


@dataclass(frozen=True)
class Result:
    """Scored submission: the headline macro average plus its per-category breakdown."""

    macro_mra: float
    per_category: dict[str, float]
    counts: dict[str, int]
    invalid_fraction: float
    scores: np.ndarray

    def __str__(self) -> str:
        cells = "  ".join(
            f"{name} {self.per_category.get(name, float('nan')):.4f} (n={self.counts.get(name, 0)})"
            for name in CATEGORIES
        )
        return f"macro MRA {self.macro_mra:.4f} | {cells} | invalid {self.invalid_fraction:.1%}"


def score(frame: pd.DataFrame, prediction_column: str = "parsed_value",
          truth_column: str = "ground_truth_posterior") -> Result:
    """Score a submission frame the way the official evaluator does.

    Requires ``inference_type`` and ``video_type`` alongside the prediction and truth columns.
    Raises if any of the four categories is absent, matching the official behaviour of reporting
    no average at all rather than a partial one.
    """
    missing = {"inference_type", "video_type", prediction_column, truth_column} - set(frame.columns)
    if missing:
        raise KeyError(f"submission is missing required columns: {sorted(missing)}")

    scores = item_scores(frame[prediction_column], frame[truth_column])
    labels = category_labels(frame)

    per_category: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name in CATEGORIES:
        selected = scores[(labels == name).to_numpy() & np.isfinite(scores)]
        if selected.size:
            per_category[name] = float(selected.mean())
            counts[name] = int(selected.size)

    absent = [name for name in CATEGORIES if name not in per_category]
    if absent:
        raise ValueError(
            f"categories {absent} have no scorable rows; the official evaluator reports no "
            "average in this case, so the submission would score nothing"
        )

    predictions = pd.to_numeric(frame[prediction_column], errors="coerce")
    invalid = (predictions.isna() | (predictions == 0)).mean()

    return Result(
        macro_mra=float(np.mean([per_category[name] for name in CATEGORIES])),
        per_category=per_category,
        counts=counts,
        invalid_fraction=float(invalid),
        scores=scores,
    )


def _macro_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    values = []
    for name in CATEGORIES:
        selected = scores[(labels == name) & np.isfinite(scores)]
        if selected.size:
            values.append(selected.mean())
    return float(np.mean(values)) if values else float("nan")


def paired_bootstrap(baseline: Result, candidate: Result, labels: pd.Series,
                     resamples: int = 10_000, seed: int = 0) -> dict[str, float]:
    """Compare two scored submissions on the *same* items.

    The validation split is only 159 rows, so an unpaired comparison cannot resolve anything
    below roughly 8 MRA points. Resampling the per-item score *differences* removes the shared
    item-difficulty variance and is far more sensitive, which is the only way to tell a real
    improvement from noise on a set this small.

    Returns the observed delta and a 95% confidence interval. Accept a change only when the
    interval excludes zero.
    """
    if baseline.scores.shape != candidate.scores.shape:
        raise ValueError("paired comparison requires identical item sets")

    label_values = np.asarray(labels)
    observed = candidate.macro_mra - baseline.macro_mra

    rng = np.random.default_rng(seed)
    count = baseline.scores.size
    deltas = np.empty(resamples)
    for index in range(resamples):
        draw = rng.integers(0, count, count)
        deltas[index] = (
            _macro_from_scores(candidate.scores[draw], label_values[draw])
            - _macro_from_scores(baseline.scores[draw], label_values[draw])
        )

    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_no_improvement": float((deltas <= 0).mean()),
        "significant": bool(low > 0 or high < 0),
    }
