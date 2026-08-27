"""Tests for the CPU half of the geometric solver: units, parsing, and scale recovery."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from quantiphy import geometry, units
from quantiphy.geometry import DepthReading
from quantiphy.parsing import (
    DepthReading,
    build_request,
    parse_depth,
    parse_output_unit,
    parse_prior,
    parse_question,
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


@pytest.mark.parametrize("text, expected_object", [
    # The phrase is fed straight to Grounding-DINO, so the quantity word must not survive:
    # asking it to find "billiard ball diameter" returns a box round the wrong, larger thing,
    # which makes gamma too small and shrinks every downstream answer.
    ("billiard ball diameter = 57.2mm", "billiard ball"),
    ("walking velocity = 1.25m/s", "walking"),
    ("lane width = 3.66m", "lane"),
    ("center circle radius = 9.15m", "center circle"),
    ("credit card width = 5.4cm", "credit card"),
    ("ruler calibre = 1cm", "ruler"),
    ("sailboat length = 12m", "sailboat"),
    # ...while the already-clean "<quantity> of the <object>" form must be left alone.
    ("diameter of the tennis ball = 6.7cm", "tennis ball"),
    ("acceleration of the typewriter before 0.45s = 9.8m/s^2", "typewriter"),
    ("diameter of the tire of the blue car = 40cm", "tire of the blue car"),
])
def test_prior_object_phrase_is_groundable(text: str, expected_object: str) -> None:
    (prior,) = parse_prior(text)
    assert prior.object_name == expected_object


def test_prior_recovers_a_leading_timestamp_instead_of_burying_it_in_the_phrase() -> None:
    """124 test rows are written ``t=0.6, ball acceleration = ...``.

    The timestamp was silently dropped and the ``t=0.6`` glued onto the object phrase, so the
    prior was measured at the wrong instant from a phrase nothing can ground.
    """
    (prior,) = parse_prior("t=0.6, ball acceleration = 4.6m/s^2")
    assert prior.object_name == "ball"
    assert prior.timestamp == pytest.approx(0.6)
    assert prior.value_si == pytest.approx(4.6)

    (with_unit,) = parse_prior("t=1.6s, ball acceleration = 3.0m/s^2")
    assert with_unit.timestamp == pytest.approx(1.6)


def test_prior_accepts_a_tilde_in_place_of_equals() -> None:
    """``pedestrian walking speed ~1.1 m/s`` -- 0 test rows, but 6 of the first 20 validation."""
    (prior,) = parse_prior("pedestrian walking speed ~1.1 m/s")
    assert prior.dimension == "speed"
    assert prior.value_si == pytest.approx(1.1)
    assert prior.object_name == "pedestrian walking"


@pytest.mark.parametrize("text", [
    "acceleration = 9.8 m/s^2",
    "gravity acc = 9.8m/s^2",
    "gravity_acceleration = 9.8m/s^2",
])
def test_a_bare_constant_prior_names_no_object_to_ground(text: str) -> None:
    """A physical constant is not a scene measurement: there is nothing in frame to measure.

    Leaving a phrase here is worse than leaving none -- the solver would ground an object called
    "acceleration" and scale the whole answer off it.
    """
    (prior,) = parse_prior(text)
    assert prior.object_name == ""


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


def test_no_scale_prior_phrase_is_contaminated_on_the_full_test_split(
        test_split: pd.DataFrame) -> None:
    """The assertion that actually proves the grounding fix, at 3,289-row scale.

    Before the fix, 412 rows handed Grounding-DINO a phrase containing the quantity word
    ("billiard ball diameter") and 124 handed it a ``t=0.6`` prefix. Both make the detector return
    a box round the wrong thing, and because the prior sets ``gamma`` the error is multiplicative
    on every answer in the row. Zero is the only acceptable count.
    """
    quantity_word = re.compile(
        r"\b(speeds?|velocit\w*|acceler\w*|acc|gravity|diameters?|radius|lengths?|widths?|"
        r"heights?|calibre|caliber|sizes?|distances?)\b", re.IGNORECASE)

    contaminated, groundable = [], 0
    for _, row in test_split.iterrows():
        prior = build_request(row).scale_prior
        if prior is None or not prior.object_name:
            continue                       # a constant with no object is handled by the solver
        groundable += 1
        if quantity_word.search(prior.object_name) or prior.object_name.startswith("t="):
            contaminated.append((str(row["prior"])[:60], prior.object_name))

    assert contaminated == []
    # Guard the other way too: over-stripping would empty phrases and silently lose rows.
    assert groundable > 0.88 * len(test_split)


# ------------------------------------------------------------------- separations ("A to B")

@pytest.mark.parametrize("question, measurement, object_a, object_b", [
    ("What is the distance between the white car and the bicycle at 1.0s in meters?",
     "distance-pair", "white car", "bicycle"),
    ("What is the distance from the bird to the roof at 1s, in meters?",
     "distance-pair", "bird", "roof"),
    ("What is the distance of the ball from the desk at 0.40s in meters?",
     "distance-pair", "ball", "desk"),
    # The descriptor is the only thing telling the two objects apart, so it has to survive.
    ("What is the distance between the person in the black shirt and the person in the white "
     "shirt at 1.0s in meters?",
     "distance-pair", "person in the black shirt", "person in the white shirt"),
    # B is the camera: a depth query, not a second object in the scene.
    ("What is the distance in meters between the black cat and the camera at 0.5 s?",
     "distance-camera", "black cat", None),
    ("What is the basketball's  distance from the camera at 1.2s in meters?",
     "distance-camera", "basketball", None),
    # One phrase, two instances of it -- not answerable from a single best box per frame.
    ("What is the distance between the two black lamps in meters?",
     "distance-twin", "black lamps", None),
    # A single object's own span really is an extent; it must not become a pair.
    ("What is the height of the cabin floor above the ground in meters?",
     "height", "cabin floor above the ground", None),
])
def test_separation_questions_recover_both_objects(
        question: str, measurement: str, object_a: str, object_b: str | None) -> None:
    parsed_measurement, _, parsed_a, parsed_b, _ = parse_question(question)
    assert (parsed_measurement, parsed_a, parsed_b) == (measurement, object_a, object_b)


def test_a_time_interval_is_not_mistaken_for_a_pair_of_objects() -> None:
    """"between 1s and 2s" is an interval. Reading it as two objects would ground the numbers."""
    measurement, _, target, target_b, timing = parse_question(
        "What is the speed of the car between 1s and 2s in m/s?")
    assert (target, target_b) == ("car", None)
    assert timing["interval"] == (1.0, 2.0)
    assert not measurement.startswith("distance")


def test_every_separation_row_on_the_full_test_split_is_routed_away_from_max_extent(
        test_split: pd.DataFrame) -> None:
    """The assertion that proves the distance fix, at 3,289-row scale.

    278 rows ask for a distance. Before the fix every one of them fell through ``extent_for`` to
    ``max(width, height)`` -- one object's own box, where a separation between two objects was
    asked. On the billiard rows that measured 60 px where the geometry needed 149 px and 182 px.

    Each row must now land in exactly one of four buckets, and no ``distance`` row may keep the
    bare single-object routing while naming two objects.
    """
    requests = [build_request(row) for _, row in test_split.iterrows()]
    buckets = Counter(r.measurement for r in requests if r.measurement.startswith("distance"))

    assert sum(buckets.values()) == 278
    assert buckets["distance-pair"] == 183       # two phrases -> centroid separation
    assert buckets["distance-twin"] == 44        # one phrase twice -> declines, does not guess
    assert buckets["distance-camera"] == 10      # answered from depth_info, no vision needed
    assert buckets["distance"] == 41             # genuinely one object's own span

    # A pair row without both phrases would silently fall back to a single extent again.
    pairs = [r for r in requests if r.measurement == "distance-pair"]
    assert all(r.target_object and r.target_object_b for r in pairs)
    assert all(r.target_object != r.target_object_b for r in pairs)
    # Every camera row must have something for depth_for to match on.
    assert all(r.target_object for r in requests if r.measurement == "distance-camera")


def test_camera_distance_rows_are_answered_from_depth_info_without_vision(
        test_split: pd.DataFrame) -> None:
    """These 10 rows state their own answer in ``depth_info``; the backend is never consulted."""
    from quantiphy.solver import solve_row
    from quantiphy.vision import NullBackend

    solved = 0
    for _, row in test_split.iterrows():
        request = build_request(row)
        if request.measurement != "distance-camera":
            continue
        # NullBackend measures nothing, so a solved row proves no vision was involved.
        answer = solve_row(request, NullBackend(), video_path="")
        if answer.solved:
            assert answer.method == "depth-info-direct"
            assert answer.value > 0
            solved += 1
    assert solved >= 8


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

# --------------------------------------------------------------- 2026-08-22 defect regressions


def test_partial_name_overlap_never_outranks_an_exact_match() -> None:
    """A loose compound key must not beat a precise one. It used to, by 8x.

    ``distance_ball_camera = 5.0`` against ``distance_basketball_football_volleyball_camera = 40.0``:
    the containment score was summed over the full cross product, so three partial hits scored 1.5
    and beat the exact token's 1.0.
    """
    exact = geometry._name_overlap({"ball"}, {"ball"})
    loose = geometry._name_overlap({"ball"}, {"basketball", "football", "volleyball"})
    assert loose < exact


def test_depth_ratio_declines_rather_than_inflating_without_bound() -> None:
    """Overshoot is fatal under MRA, so an implausible ratio must decline, not multiply."""
    assert geometry.depth_ratio_correction(1.0, 2.0, 1.0) == pytest.approx(2.0)
    with pytest.raises(geometry.ScaleError):
        geometry.depth_ratio_correction(1.0, 100.0, 1.0)
    with pytest.raises(geometry.ScaleError):
        geometry.depth_ratio_correction(1.0, 1.0, 100.0)


def test_radial_speed_ignores_capitalisation() -> None:
    """Target phrases keep their original case, so a case-sensitive match silently dropped radial."""
    depths = (
        DepthReading(object_name="car", distance_m=10.0, timestamp=0.0),
        DepthReading(object_name="car", distance_m=20.0, timestamp=1.0),
    )
    assert geometry.radial_speed(depths, "car") == pytest.approx(10.0)
    assert geometry.radial_speed(depths, "Car") == pytest.approx(10.0)


def test_prior_depth_is_read_at_the_instant_its_pixels_were_measured() -> None:
    """A prior and target that are the same object must get a ratio of exactly 1.0.

    The prior's pixels are measured at the request's timestamp whenever the prior names no instant
    of its own, but its *depth* used to be looked up with ``None`` -- which falls through to the
    first reading in file order. Same object, two different instants, fabricated ratio.
    """
    depths = (
        DepthReading(object_name="car", distance_m=17.8, timestamp=1.0),
        DepthReading(object_name="car", distance_m=23.2, timestamp=2.0),
    )
    # What the fix does: both lookups use the request's instant, so the ratio is exactly 1.0.
    prior_depth = geometry.depth_for(depths, "car", 2.0)
    target_depth = geometry.depth_for(depths, "car", 2.0)
    assert target_depth / prior_depth == pytest.approx(1.0)

    # What the bug did, and why it mattered: a None timestamp takes the first reading in file
    # order, so the same object against itself came out 30% larger.
    stale_prior_depth = geometry.depth_for(depths, "car", None)
    assert stale_prior_depth == pytest.approx(17.8)
    assert target_depth / stale_prior_depth == pytest.approx(23.2 / 17.8)



# ------------------------------------------- 2026-08-27: the prior's own radial motion

def test_tangential_component_removes_the_along_axis_part() -> None:
    assert geometry.tangential_component(5.0, 3.0) == pytest.approx(4.0)
    assert geometry.tangential_component(5.0, -3.0) == pytest.approx(4.0)
    assert geometry.tangential_component(5.0, 0.0) == pytest.approx(5.0)
    assert geometry.tangential_component(5.0, None) == pytest.approx(5.0)


@pytest.mark.parametrize("radial", [5.0, 6.0, -7.5, float("nan"), float("inf")])
def test_tangential_component_declines_an_impossible_decomposition(radial: float) -> None:
    """A radial part at or above the stated total means the two disagree, not that the object
    is moving straight at the camera. Returning ~0 there would emit a hard zero."""
    assert geometry.tangential_component(5.0, radial) is None


def test_solve_uses_a_smaller_scale_when_the_prior_moves_toward_the_camera() -> None:
    """The prior states a 3D speed; the pixels only ever saw its in-plane part. Dividing the full
    3D speed by the in-plane pixel speed inflates gamma, and D3 is 100% dynamic-prior rows."""
    flat = geometry.solve(target_pixels=100, prior_world_si=5.0, prior_pixels=100)
    corrected = geometry.solve(target_pixels=100, prior_world_si=5.0, prior_pixels=100,
                               prior_radial_si=3.0)
    assert flat == pytest.approx(5.0)
    assert corrected == pytest.approx(4.0)


def test_solve_falls_back_to_the_full_prior_when_the_radial_part_is_impossible() -> None:
    assert geometry.solve(target_pixels=100, prior_world_si=5.0, prior_pixels=100,
                          prior_radial_si=9.0) == pytest.approx(5.0)
