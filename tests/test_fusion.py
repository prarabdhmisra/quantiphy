"""Tests for combining the two arms.

The property that matters most is the one the module is named for: the combination happens in **log
space**. An arithmetic mean of 1 and 100 is 50.5, which under a relative metric is not between them
in any useful sense, and it would also destroy the counterfactual-prior guarantee the geometric arm
is built around.

The second property is that fusing is *gated*, because the measured premise is only half true: truth
lies between the two arms on 46.7% of the rows where both answered, so an ungated mean converts one
winner and one loser into two losers.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import fuse_predictions  # noqa: E402

from quantiphy import fusion


# --------------------------------------------------------------------------- log space

def test_the_mean_is_geometric_not_arithmetic() -> None:
    """1 and 100 average to 10. An arithmetic mean would say 50.5, which is 50x from one input."""
    assert fusion.log_mean(1.0, 100.0) == pytest.approx(10.0)
    assert fusion.log_mean(2.0, 8.0) == pytest.approx(4.0)


def test_the_mean_is_scale_equivariant() -> None:
    """Scale both arms by k and the answer scales by exactly k.

    This is why the arithmetic mean is not merely a worse estimator here but a wrong one: the
    geometric arm's headline robustness claim is that multiplying the given prior by 1000 multiplies
    the answer by 1000, and fusing must not smuggle in a fixed real-world magnitude.
    """
    base = fusion.log_mean(3.0, 7.0)
    for k in (0.001, 0.1, 10.0, 1000.0):
        assert fusion.log_mean(3.0 * k, 7.0 * k) == pytest.approx(base * k, rel=1e-12)


def test_the_unweighted_mean_recovers_either_arm_from_the_other() -> None:
    """At the default weight the fused value is invertible: ``second == fused ** 2 / first``.

    This is not a curiosity. ``replay-tangential.csv`` is a gitignored run output and was gone by
    2026-08-29, so the solver arm behind ``fuse-v1.predictions.csv`` could not be re-read -- but every
    ``fuse-fused`` row is the geometric mean of two arms and one of them (the VLM's) was still on
    disk. Inverting recovered the solver's 1,307 answers exactly, which is how the S2 and D3 probes
    of that day were built without a fresh replay. The check that proved it: the recovered S2/D2/S3
    values matched the surviving pre-tangential-fix replay on 997 of 997 rows, and D3 differed on
    exactly the rows that fix touches.

    It holds only at ``weight == 0.5``. A weighted mean is still invertible but by a different
    exponent, so anything relying on this must not silently inherit a changed default.
    """
    assert fusion.WEIGHT == 0.5
    for first, second in ((3.0, 7.0), (0.004, 12000.0), (1.0, 1.0)):
        fused = fusion.log_mean(first, second)
        assert fused ** 2 / first == pytest.approx(second, rel=1e-9)
        assert fused ** 2 / second == pytest.approx(first, rel=1e-9)


@pytest.mark.parametrize("weight, expected", [(0.0, 3.0), (1.0, 12.0), (0.5, 6.0)])
def test_weight_interpolates_between_the_two_arms(weight: float, expected: float) -> None:
    assert fusion.log_mean(3.0, 12.0, weight) == pytest.approx(expected)


@pytest.mark.parametrize("weight", [-0.1, 1.1, 2.0])
def test_a_weight_outside_the_unit_interval_is_refused(weight: float) -> None:
    """Extrapolating past either arm is not averaging, and it is silently catastrophic here."""
    with pytest.raises(ValueError):
        fusion.log_mean(3.0, 12.0, weight)


# --------------------------------------------------------------------------- usability

@pytest.mark.parametrize("bad", [None, 0.0, -1.0, float("nan"), float("inf")])
def test_zero_and_negative_and_nonfinite_are_not_usable(bad) -> None:
    """845 of the VLM's 3,289 test replies are literally 0 -- "the ball is stationary, so its speed
    is zero". A zero is a hard zero under this metric and log(0) is not a number to average, so this
    branch is the common case rather than an edge case."""
    assert not fusion.usable(bad)


def test_disagreement_is_symmetric_and_at_least_one() -> None:
    assert fusion.disagreement(2.0, 8.0) == pytest.approx(4.0)
    assert fusion.disagreement(8.0, 2.0) == pytest.approx(4.0)
    assert fusion.disagreement(5.0, 5.0) == pytest.approx(1.0)
    assert fusion.disagreement(5.0, None) is None


# --------------------------------------------------------------------------- fuse

def test_fuse_averages_two_close_answers() -> None:
    value, route = fusion.fuse(4.0, 9.0)
    assert route == fusion.FUSED
    assert value == pytest.approx(6.0)


def test_fuse_keeps_the_primary_when_the_two_are_too_far_apart() -> None:
    """The measured reason: truth is between the arms on only 46.7% of rows, so averaging a right
    answer with a wildly wrong one produces two losers instead of one."""
    value, route = fusion.fuse(1.0, 100.0, max_disagreement=5.0)
    assert route == fusion.TOO_FAR
    assert value == pytest.approx(1.0)


def test_fuse_falls_through_to_whichever_arm_answered() -> None:
    assert fusion.fuse(4.0, None) == (4.0, fusion.PRIMARY_ONLY)
    assert fusion.fuse(None, 9.0) == (9.0, fusion.SECONDARY_ONLY)
    assert fusion.fuse(None, 0.0) == (None, fusion.NEITHER)


def test_fuse_declines_only_when_neither_arm_is_usable() -> None:
    """Both arms answering is precisely the case where declining would throw away two measurements
    to avoid choosing between them."""
    assert fusion.fuse(None, None)[0] is None
    assert fusion.fuse(0.0, -3.0)[0] is None


def test_prefer_lower_takes_the_smaller_arm() -> None:
    """The one variant with a mechanism instead of a fitted parameter: fatal overshoots run at 20%
    for the solver and 15% for the VLM, and 5% for the minimum, because overshooting requires both
    arms to overshoot. Overshoot is fatal under MRA and undershoot is cheap."""
    value, route = fusion.fuse(4.0, 9.0, prefer_lower=True)
    assert route == fusion.FUSED
    assert value == pytest.approx(4.0)


def test_prefer_lower_still_respects_the_disagreement_gate() -> None:
    assert fusion.fuse(1.0, 100.0, prefer_lower=True) == (1.0, fusion.TOO_FAR)


@pytest.mark.parametrize("cap", [0.0, 0.5, 0.99])
def test_a_disagreement_cap_below_one_is_refused(cap: float) -> None:
    """It is a fold factor, so anything under 1 can never be satisfied and would silently disable
    fusion on every row while looking like a configured threshold."""
    with pytest.raises(ValueError):
        fusion.fuse(4.0, 9.0, max_disagreement=cap)


def test_fusing_never_lands_outside_the_two_arms() -> None:
    """A mean that can exceed both inputs is not a mean, and here it would be an overshoot the
    metric charges at full price."""
    for primary, secondary in [(1.0, 3.0), (3.0, 1.0), (0.05, 0.2), (700.0, 900.0)]:
        value, _ = fusion.fuse(primary, secondary)
        assert min(primary, secondary) <= value <= max(primary, secondary)


# --------------------------------------------------------------------------- the script

def _frames():
    """Four rows, one per category, with both arms answering at a 4x disagreement."""
    solver = pd.DataFrame({
        "row_index": [0, 1, 2, 3],
        "video_id": ["v"] * 4,
        "question": ["What is the width of the box in meters?"] * 4,
        "video_type": ["S2SX", "V2SX", "S3SX", "A3SX"],
        "inference_type": ["SS", "DS", "SS", "DS"],
        "parsed_value": [2.0, 2.0, 2.0, 2.0],
        "method": ["geometric-2d"] * 4,
    })
    vlm = solver[["row_index"]].copy()
    vlm["parsed_value"] = [8.0, 8.0, 8.0, 8.0]
    vlm["method"] = "vlm-sentinel"
    return solver, vlm


def _spec(value):
    return [f"{c}={value}" for c in ("S2", "D2", "S3", "D3")]


def test_the_script_fuses_within_the_cap_and_keeps_the_primary_outside_it():
    solver, vlm = _frames()
    caps = fuse_predictions.parse_pairs(_spec(5), "cap", float)
    primaries = fuse_predictions.parse_pairs(_spec("vlm"), "primary", fuse_predictions._arm)
    out = fuse_predictions.fuse_frames(solver, vlm, caps, primaries)
    assert list(out["parsed_value"]) == [pytest.approx(4.0)] * 4     # geometric mean of 2 and 8
    assert set(out["method"]) == {"fuse-fused"}

    tight = fuse_predictions.parse_pairs(_spec(2), "cap", float)
    out = fuse_predictions.fuse_frames(solver, vlm, tight, primaries)
    assert list(out["parsed_value"]) == [8.0] * 4                    # the VLM, as primary
    assert set(out["method"]) == {"fuse-disagreement"}


def test_a_cap_of_one_is_the_no_fusion_control():
    """One submission carries four channels, so the control is expressed as a cap rather than as a
    separate flag: at a fold factor of 1 the arms must be exactly equal to fuse."""
    solver, vlm = _frames()
    caps = fuse_predictions.parse_pairs(["S2=1", "D2=5", "S3=1", "D3=3"], "cap", float)
    primaries = fuse_predictions.parse_pairs(_spec("solver"), "primary", fuse_predictions._arm)
    out = fuse_predictions.fuse_frames(solver, vlm, caps, primaries)
    assert out.loc[0, "parsed_value"] == 2.0 and out.loc[0, "method"] == "fuse-disagreement"
    assert out.loc[1, "parsed_value"] == pytest.approx(4.0)
    assert out.loc[3, "parsed_value"] == 2.0                          # 4x disagreement, cap 3


def test_the_script_requires_every_category_to_be_named():
    """A category left out would fall through to a default and produce a channel nobody chose --
    and the reading would then be attributed to the wrong change."""
    with pytest.raises(SystemExit, match="must name every category"):
        fuse_predictions.parse_pairs(["S2=5", "D2=5"], "cap", float)


def test_the_script_refuses_a_duplicate_or_unknown_category():
    with pytest.raises(SystemExit, match="twice"):
        fuse_predictions.parse_pairs(["S2=5", "S2=3", "D2=5", "S3=5", "D3=5"], "cap", float)
    with pytest.raises(SystemExit, match="not one of"):
        fuse_predictions.parse_pairs(["S9=5"], "cap", float)


def test_the_script_refuses_an_unknown_arm():
    with pytest.raises(SystemExit, match="primary must be one of"):
        fuse_predictions.parse_pairs(_spec("gpt"), "primary", fuse_predictions._arm)


def test_a_row_the_vlm_never_reached_keeps_the_solver_answer():
    solver, vlm = _frames()
    out = fuse_predictions.fuse_frames(
        solver, vlm.iloc[:2],
        fuse_predictions.parse_pairs(_spec(5), "cap", float),
        fuse_predictions.parse_pairs(_spec("solver"), "primary", fuse_predictions._arm))
    assert out.loc[2, "parsed_value"] == 2.0
    assert out.loc[2, "method"] == "fuse-primary-only"
