"""Anchors our scorer to the organizers' published numbers.

If these fail, every downstream measurement is untrustworthy -- fix this before anything else.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantiphy.scoring import (
    CATEGORIES,
    THRESHOLDS,
    Result,
    item_scores,
    category_labels,
    paired_bootstrap,
    score,
)

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"

#: From the organizers' own mra_results/all_model_results.csv.
PUBLISHED_GPT51_MACRO = 0.4856119486941172
PUBLISHED_GPT51_INVALID = 0.06289308176100629


@pytest.fixture(scope="module")
def gpt51() -> pd.DataFrame:
    return pd.read_csv(FIXTURES / "gpt-5.1_validation.csv", encoding="utf-8-sig")


def test_reproduces_published_baseline(gpt51: pd.DataFrame) -> None:
    """Our macro MRA must match the published GPT-5.1 validation score.

    Tolerance is 0.005: the official script drops rows whose ground truth is missing before
    computing invalid counts in a slightly different order, which moves the last digit or two.
    """
    result = score(gpt51)
    assert result.macro_mra == pytest.approx(PUBLISHED_GPT51_MACRO, abs=0.005)


def test_reproduces_published_invalid_fraction(gpt51: pd.DataFrame) -> None:
    result = score(gpt51)
    assert result.invalid_fraction == pytest.approx(PUBLISHED_GPT51_INVALID, abs=0.005)


def test_all_four_categories_present(gpt51: pd.DataFrame) -> None:
    result = score(gpt51)
    assert set(result.per_category) == set(CATEGORIES)
    assert sum(result.counts.values()) == 159


def test_paper_appendix_threshold_set_does_not_reproduce(gpt51: pd.DataFrame, monkeypatch) -> None:
    """The paper's Appendix A.2 threshold set is a typo, and this is how we know.

    Substituting {0.5, 0.55, ..., 0.95} yields ~0.376 rather than the published ~0.486.
    """
    import quantiphy.scoring as scoring

    monkeypatch.setattr(scoring, "THRESHOLDS", tuple(np.arange(0.50, 0.96, 0.05)))
    assert scoring.score(gpt51).macro_mra == pytest.approx(0.376, abs=0.01)


@pytest.mark.parametrize(
    "ratio, expected",
    [
        (1.00, 1.0),   # exact
        (0.90, 0.8),
        (1.10, 0.8),
        (0.50, 0.4),   # 2x undershoot still earns partial credit ...
        (2.00, 0.0),   # ... but 2x overshoot earns nothing. This asymmetry drives our shrinkage.
        (10.0, 0.0),
    ],
)
def test_score_is_asymmetric_in_log_space(ratio: float, expected: float) -> None:
    assert item_scores([ratio], [1.0])[0] == pytest.approx(expected)


@pytest.mark.parametrize("prediction", [np.nan, 0.0, None, "not a number", ""])
def test_invalid_predictions_score_zero_and_still_count(prediction) -> None:
    """A blank cell is a hard zero, not an exclusion. Always emit a fallback instead."""
    assert item_scores([prediction], [5.0])[0] == 0.0


def test_negative_predictions_are_scored_by_magnitude() -> None:
    """The official parser applies abs() before scoring, so sign is discarded."""
    assert item_scores([-4.9], [4.9])[0] == pytest.approx(1.0)


def test_missing_ground_truth_is_excluded_not_zeroed() -> None:
    scores = item_scores([1.0, 1.0], [np.nan, 0.0])
    assert np.isnan(scores).all()


def test_thresholds_match_official_evaluator() -> None:
    assert THRESHOLDS == (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    assert len(THRESHOLDS) == 10


def test_missing_category_raises_rather_than_scoring_partially(gpt51: pd.DataFrame) -> None:
    """The official evaluator reports no average at all if a category is absent."""
    labels = category_labels(gpt51)
    with pytest.raises(ValueError, match="D3"):
        score(gpt51[labels != "D3"])


def test_paired_bootstrap_detects_a_real_improvement(gpt51: pd.DataFrame) -> None:
    baseline = score(gpt51)
    perfect = gpt51.copy()
    perfect["parsed_value"] = perfect["ground_truth_posterior"]
    verdict = paired_bootstrap(baseline, score(perfect), category_labels(gpt51), resamples=2000)
    assert verdict["significant"] and verdict["delta"] > 0.4


def test_paired_bootstrap_rejects_a_null_change(gpt51: pd.DataFrame) -> None:
    """Identical predictions must not register as an improvement."""
    baseline = score(gpt51)
    verdict = paired_bootstrap(baseline, score(gpt51.copy()), category_labels(gpt51), resamples=2000)
    assert verdict["delta"] == pytest.approx(0.0) and not verdict["significant"]
