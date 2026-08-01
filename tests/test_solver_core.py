"""Tests for the CPU half of the geometric solver: units, parsing, and scale recovery."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quantiphy import geometry, units
from quantiphy.parsing import (
    DepthReading,
    build_request,
    parse_depth,
    parse_output_unit,
    parse_prior,
)

FIXTURES = Path(__file__).resolve().parent.parent / "data" / "fixtures"


@pytest.fixture(scope="module")
def test_split() -> pd.DataFrame:
    return pd.read_parquet(FIXTURES / "test_dataset.parquet")


# --------------------------------------------------------------------------- units

@pytest.mark.parametrize("text, expected_si", [
    ("6.7cm", 0.067), ("40mm", 0.040), ("0.65m", 0.65),
    ("2.5104m/s", 2.5104), ("-2.86m/s^2", -2.86),
])
def test_quantity_round_trips_to_si(text: str, expected_si: float) -> None:
    value, unit = units.parse_quantity(text)
    assert units.to_si(value, unit) == pytest.approx(expected_si)


def test_superscript_and_spacing_variants_normalise() -> None:
    for spelling in ("m/s²", "m/s^2", "m/s2", "M/S^2", " m / s ^ 2 "):
        assert units.dimension_of(spelling) == "acceleration"


def test_bare_number_uses_assumed_dimension() -> None:
    """1,034 of 3,289 test priors give a number with no unit."""
    assert units.parse_quantity("5.21", assume="speed") == (5.21, "m/s")
    assert units.parse_quantity("5.21") is None


def test_unit_conversion_is_exact_both_ways() -> None:
    assert units.from_si(1.0, "cm") == pytest.approx(100.0)
    assert units.from_si(units.to_si(9.9, "cm"), "cm") == pytest.approx(9.9)


# --------------------------------------------------------------------------- parsing

@pytest.mark.parametrize("question, expected", [
    ("What is the length of the wood block in cm?", "cm"),
    ("What is the velocity of the ball at 1.00s in m/s?", "m/s"),
    ("What is the height, in meters, of the white pillar?", "meters"),
    ("What is the acceleration of the car at 1.0s in m/s²?", "m/s^2"),
    ("What is the diameter of the ball in mm？", "mm"),          # fullwidth question mark
    ("What is the width of the object in the center?", None),    # "in the" is not a unit
])
def test_output_unit_extraction(question: str, expected: str | None) -> None:
    assert parse_output_unit(question) == expected


def test_prior_parses_object_quantity_and_si_value() -> None:
    (prior,) = parse_prior("diameter of the ping pong ball = 40mm")
    assert prior.quantity == "diameter"
    assert prior.dimension == "length"
    assert prior.value_si == pytest.approx(0.040)
    assert "ping pong ball" in prior.object_name


def test_prior_handles_possessive_and_parenthetical() -> None:
    (prior,) = parse_prior("Callisto's model speed(the outermost planet) = 0.627 m/s")
    assert prior.dimension == "speed"
    assert prior.value_si == pytest.approx(0.627)


def test_prior_keeps_negative_acceleration_magnitude() -> None:
    (prior,) = parse_prior("acceleration of the orange car = -2.86m/s^2")
    assert prior.value_si == pytest.approx(-2.86)


def test_gravity_prior_is_flagged_and_deprioritised_as_scale_anchor() -> None:
    """Gravity is a constant, not a scene measurement, so it cannot set pixel scale."""
    request = build_request({
        "video_id": "x", "fps": 30, "video_type": "A2MC", "inference_type": "DD",
        "question": "What is the velocity of the ball at 1.0s in m/s?",
        "prior": "gravity acc = 9.8m/s^2\nlength of the car = 4.5m",
        "depth_info": None,
    })
    assert len(request.priors) == 2
    assert not request.scale_prior.is_gravity
    assert request.scale_prior.value_si == pytest.approx(4.5)


def test_depth_parses_multiline_underscored_keys() -> None:
    readings = parse_depth(
        "t=1.2s, distance_right_tennis_ball_camera = 0.7760m\n"
        "distance_pencil_bag_camera = 1.0500m\n"
        "t=0.60s, distance_jar_camera =1.9735m"          # note: no space after '='
    )
    assert len(readings) == 3
    assert readings[0].object_name == "right tennis ball"
    assert readings[0].timestamp == pytest.approx(1.2)
    assert readings[1].timestamp is None
    assert readings[2].distance_m == pytest.approx(1.9735)


@pytest.mark.parametrize("question, timestamp, interval", [
    ("What is the velocity of the ball at 1.20s in m/s?", 1.2, None),
    ("What is the displacement between 1.5s and 1.6s in meters?", None, (1.5, 1.6)),
    ("What is the diameter of the ball in cm?", None, None),
])
def test_timing_extraction(question: str, timestamp, interval) -> None:
    request = build_request({
        "video_id": "x", "fps": 24, "video_type": "S2SC", "inference_type": "SS",
        "question": question, "prior": "diameter of the ball = 6.7cm", "depth_info": None,
    })
    assert request.timestamp == timestamp and request.interval == interval


def test_unit_overrides_keyword_when_they_disagree() -> None:
    """The stated unit is the reliable signal; a stray keyword must not flip the dimension."""
    request = build_request({
        "video_id": "x", "fps": 24, "video_type": "V2SC", "inference_type": "DS",
        "question": "What is the distance travelled by the speeding car in meters?",
        "prior": "speed of the car = 3m/s", "depth_info": None,
    })
    assert request.dimension == "length"
    assert any("keyword implies" in w for w in request.warnings)


def test_missing_unit_falls_back_rather_than_failing() -> None:
    """A row must never fail to produce a unit -- a blank answer scores a hard zero."""
    request = build_request({
        "video_id": "x", "fps": 24, "video_type": "S2SC", "inference_type": "SS",
        "question": "How wide is the block?", "prior": "length of the rod = 1m", "depth_info": None,
    })
    assert request.output_unit == "m"
    assert "no explicit unit in question; assumed SI" in request.warnings


# ------------------------------------------------------- coverage on the real test split

def test_parser_coverage_on_full_test_split(test_split: pd.DataFrame) -> None:
    """Regression guard: parsing must stay near-total across all 3,289 real rows."""
    requests = [build_request(row) for _, row in test_split.iterrows()]
    total = len(requests)
    assert total == 3289

    assert sum(1 for r in requests if r.priors) / total > 0.99
    assert sum(1 for r in requests if r.target_object) / total > 0.95
    assert sum(1 for r in requests
               if "no explicit unit" not in " ".join(r.warnings)) / total > 0.99

    three_d = [r for r in requests if r.is_3d]
    assert sum(1 for r in three_d if r.depths) / len(three_d) > 0.98

    # Every row must yield a usable unit, or we would submit a blank and score zero.
    assert all(r.output_unit and units.lookup(r.output_unit) for r in requests)


# --------------------------------------------------------------------------- geometry

def test_2d_scale_transfer_between_objects() -> None:
    """Tennis ball 6.7 cm across 40 px; another object 120 px is therefore 20.1 cm."""
    gamma = geometry.gamma_from_prior(0.067, 40)
    assert geometry.world_from_pixels(120, gamma=gamma) == pytest.approx(0.201)


def test_pixel_speed_to_world_speed() -> None:
    """200 px/s under a 6.7 cm / 40 px scale is 0.335 m/s."""
    gamma = geometry.gamma_from_prior(0.067, 40)
    assert geometry.world_from_pixels(200, gamma=gamma) == pytest.approx(0.335)


def test_perspective_correction_scales_with_depth_ratio() -> None:
    """An object twice as far that subtends the same pixels is twice as large."""
    focal = geometry.focal_length_px(prior_world_si=1.0, prior_pixels=100, prior_depth_m=2.0)
    assert focal == pytest.approx(200.0)
    assert geometry.world_from_pixels(100, focal_px=focal, depth_m=4.0) == pytest.approx(2.0)


def test_depth_ratio_matches_the_pinhole_route() -> None:
    focal = geometry.focal_length_px(1.0, 100, 2.0)
    pinhole = geometry.world_from_pixels(250, focal_px=focal, depth_m=5.0)
    corrected = geometry.depth_ratio_correction(
        geometry.world_from_pixels(250, gamma=geometry.gamma_from_prior(1.0, 100)), 5.0, 2.0)
    assert pinhole == pytest.approx(corrected)


def test_radial_speed_from_timestamped_depths() -> None:
    depths = (
        DepthReading("boat", 130.0, 1.0),
        DepthReading("boat", 133.0, 2.0),
    )
    assert geometry.radial_speed(depths, "boat") == pytest.approx(3.0)
    assert geometry.radial_speed(depths, "human") is None


def test_speeds_combine_in_quadrature() -> None:
    assert geometry.combine_speeds(3.0, 4.0) == pytest.approx(5.0)
    assert geometry.combine_speeds(3.0, None) == pytest.approx(3.0)


def test_depth_lookup_prefers_nearest_timestamp() -> None:
    depths = (
        DepthReading("yellow car", 25.7, 0.58),
        DepthReading("yellow car", 20.1, 1.58),
        DepthReading("human", 18.7, 0.58),
    )
    assert geometry.depth_for(depths, "the yellow car", timestamp=1.5) == pytest.approx(20.1)
    assert geometry.depth_for(depths, "human") == pytest.approx(18.7)
    assert geometry.depth_for(depths, "bicycle") is None


def test_solve_is_exactly_linear_in_the_prior() -> None:
    """The counterfactual test the paper shows every VLM failing.

    Scaling the given prior by alpha must scale the answer by exactly alpha. This is the headline
    robustness claim for the writeup, so it is pinned here.
    """
    base = geometry.solve(target_pixels=120, prior_world_si=0.067, prior_pixels=40)
    for alpha in (0.001, 0.01, 0.1, 10, 100, 1000):
        scaled = geometry.solve(target_pixels=120, prior_world_si=0.067 * alpha, prior_pixels=40)
        assert scaled == pytest.approx(base * alpha, rel=1e-12)


def test_solve_applies_depth_and_radial_terms_together() -> None:
    tangential = geometry.solve(
        target_pixels=100, prior_world_si=1.0, prior_pixels=100,
        target_depth_m=4.0, prior_depth_m=2.0)
    assert tangential == pytest.approx(2.0)

    combined = geometry.solve(
        target_pixels=100, prior_world_si=1.0, prior_pixels=100,
        target_depth_m=4.0, prior_depth_m=2.0, radial_si=1.5, is_speed=True)
    assert combined == pytest.approx((2.0 ** 2 + 1.5 ** 2) ** 0.5)


@pytest.mark.parametrize("bad", [0, float("nan"), float("inf")])
def test_unusable_pixel_measurements_raise_rather_than_returning_nonsense(bad) -> None:
    with pytest.raises(geometry.ScaleError):
        geometry.gamma_from_prior(1.0, bad)
