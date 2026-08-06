"""Tests for the vision-independent logic: measurement selection, kinematics, and end-to-end solve.

A fake backend stands in for Grounding-DINO so the whole pipeline is exercised on CPU. The pure
functions in ``backends.grounding`` are tested directly on synthetic trajectories, where the true
speed and acceleration are known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantiphy.backends.grounding import (
    DetectionSeries,
    displacement_px,
    extent_for,
    kinematics,
)
from quantiphy.parsing import build_request
from quantiphy.solver import solve_row
from quantiphy.vision import NullBackend, PixelMeasurement, VisionBackend


def make_series(times, cx, cy, width=20.0, height=10.0, score=0.9,
                frames_total=None, frames_sampled=None) -> DetectionSeries:
    times = np.asarray(times, dtype=float)
    return DetectionSeries(
        times=times,
        cx=np.asarray(cx, dtype=float),
        cy=np.asarray(cy, dtype=float),
        width=np.full(times.shape, width),
        height=np.full(times.shape, height),
        scores=np.full(times.shape, score),
        frames_total=len(times) if frames_total is None else frames_total,
        frames_sampled=len(times) if frames_sampled is None else frames_sampled,
    )


class FakeBackend:
    """Returns preset pixel measurements keyed by object phrase."""

    def __init__(self, table: dict[str, PixelMeasurement]) -> None:
        self.table = table
        self.calls: list[tuple[str, str]] = []

    def measure(self, request, video_path, object_name, dimension) -> PixelMeasurement:
        self.calls.append((object_name, dimension))
        return self.table.get(object_name,
                              PixelMeasurement(object_name=object_name, note="not in table"))


def row(**overrides) -> dict:
    base = {
        "video_id": "captured_0001", "fps": 24, "video_type": "S2MC", "inference_type": "SS",
        "question": "What is the diameter of the soccer ball in cm?",
        "prior": "diameter of the tennis ball = 6.7cm",
        "depth_info": None,
    }
    base.update(overrides)
    return base


# ------------------------------------------------------------------ extent selection

@pytest.mark.parametrize("measurement, expected", [
    ("height", 10.0), ("width", 20.0), ("diameter", 15.0), ("radius", 7.5), ("length", 20.0),
])
def test_extent_matches_the_measurement_asked_for(measurement: str, expected: float) -> None:
    series = make_series([0, 0.1, 0.2], [0, 0, 0], [0, 0, 0], width=20.0, height=10.0)
    assert extent_for(series, measurement) == pytest.approx(expected)


def test_extent_uses_median_so_a_bad_frame_is_ignored() -> None:
    series = make_series([0, 0.1, 0.2, 0.3, 0.4], [0] * 5, [0] * 5)
    series.width[2] = 500.0                      # one frame snapped to the wrong object
    assert extent_for(series, "width") == pytest.approx(20.0)


def test_calibre_is_the_small_axis_not_the_large_one() -> None:
    """A calibre is a bore, not a length. Falling through to max() measured the whole ruler."""
    series = make_series([0, 0.1, 0.2], [0] * 3, [0] * 3, width=500.0, height=12.0)
    assert extent_for(series, "calibre") == pytest.approx(12.0)
    assert extent_for(series, "caliber") == pytest.approx(12.0)


def test_detection_rate_measures_hit_rate_not_clip_length() -> None:
    """It divides by the frames we actually looked at, not by the whole clip.

    With ``max_frames=48`` the old form capped a 428-frame clip's confidence at 0.112, so
    confidence tracked video duration rather than detection quality -- useless for gating.
    """
    hit_every_sampled_frame = make_series(np.linspace(0, 2, 48), [0] * 48, [0] * 48,
                                          frames_total=428, frames_sampled=48)
    assert hit_every_sampled_frame.detection_rate == pytest.approx(1.0)

    missed_half = make_series(np.linspace(0, 2, 24), [0] * 24, [0] * 24,
                              frames_total=428, frames_sampled=48)
    assert missed_half.detection_rate == pytest.approx(0.5)


# ------------------------------------------------------------------ kinematics

def test_constant_velocity_recovered_exactly() -> None:
    times = np.linspace(0, 2, 25)
    series = make_series(times, 100 * times, np.zeros_like(times))
    speed, accel, quality = kinematics(series, at_time=1.0)
    assert speed == pytest.approx(100.0, rel=1e-6)
    assert accel == pytest.approx(0.0, abs=1e-6)
    assert quality > 0.99


def test_constant_acceleration_recovered_exactly() -> None:
    """Free fall at 200 px/s^2: speed at t=1 is 200 px/s."""
    times = np.linspace(0, 2, 25)
    series = make_series(times, np.zeros_like(times), 100 * times ** 2)
    speed, accel, _ = kinematics(series, at_time=1.0)
    assert accel == pytest.approx(200.0, rel=1e-6)
    assert speed == pytest.approx(200.0, rel=1e-6)


def test_quadratic_fit_survives_per_frame_jitter() -> None:
    """The reason we fit instead of differencing: 1 px of jitter at 24 fps is 24 px/s of noise."""
    rng = np.random.default_rng(0)
    times = np.linspace(0, 2, 49)
    clean = 100 * times
    series = make_series(times, clean + rng.normal(0, 1.0, times.size), np.zeros_like(times))

    fitted, _, _ = kinematics(series, at_time=1.0)
    differenced = float(np.mean(np.abs(np.diff(series.cx) / np.diff(times))))

    assert abs(fitted - 100.0) < 3.0
    assert abs(fitted - 100.0) < abs(differenced - 100.0)


def test_kinematics_needs_at_least_three_points() -> None:
    assert kinematics(make_series([0, 0.1], [0, 10], [0, 0]), at_time=0.0) == (0.0, 0.0, 0.0)


def test_displacement_between_two_timestamps() -> None:
    times = np.linspace(0, 2, 25)
    series = make_series(times, 30 * times, 40 * times)
    assert displacement_px(series, 1.0, 2.0) == pytest.approx(50.0, rel=1e-6)


# ------------------------------------------------------------------ end-to-end solve

def test_solves_a_2d_size_question() -> None:
    """Tennis ball 6.7 cm spans 40 px; the soccer ball spans 104 px, so it is 17.4 cm."""
    request = build_request(row())
    backend = FakeBackend({
        "tennis ball": PixelMeasurement("tennis ball", extent_px=40.0, confidence=0.9),
        "soccer ball": PixelMeasurement("soccer ball", extent_px=104.0, confidence=0.8),
    })
    answer = solve_row(request, backend, "clip.mp4")
    assert answer.solved and answer.unit == "cm"
    assert answer.value == pytest.approx(6.7 * 104 / 40)
    assert answer.method == "geometric-2d"
    assert answer.confidence == pytest.approx(0.8)


def test_answer_is_converted_into_the_questions_unit() -> None:
    """A metres-vs-centimetres slip is a silent 100x error, so pin the conversion."""
    request = build_request(row(question="What is the diameter of the soccer ball in meters?"))
    backend = FakeBackend({
        "tennis ball": PixelMeasurement("tennis ball", extent_px=40.0, confidence=0.9),
        "soccer ball": PixelMeasurement("soccer ball", extent_px=104.0, confidence=0.9),
    })
    answer = solve_row(request, backend, "clip.mp4")
    assert answer.unit == "meters"
    assert answer.value == pytest.approx(0.067 * 104 / 40)


def test_depth_correction_applies_only_for_3d_rows() -> None:
    depth = ("t=1.0s, distance_tennis_ball_camera = 2.0m\n"
             "t=1.0s, distance_soccer_ball_camera = 4.0m")
    backend = FakeBackend({
        "tennis ball": PixelMeasurement("tennis ball", extent_px=40.0, confidence=0.9),
        "soccer ball": PixelMeasurement("soccer ball", extent_px=40.0, confidence=0.9),
    })

    flat = solve_row(build_request(row(video_type="S2MC", depth_info=depth)), backend, "c.mp4")
    assert flat.method == "geometric-2d" and flat.value == pytest.approx(6.7)

    perspective = solve_row(build_request(row(video_type="S3MC", depth_info=depth)),
                            backend, "c.mp4")
    assert perspective.method == "geometric-3d"
    assert perspective.value == pytest.approx(6.7 * 4.0 / 2.0)


def test_speed_question_uses_pixel_speed_not_extent() -> None:
    request = build_request(row(
        question="What is the velocity of the soccer ball at 1.0s in m/s?",
        video_type="S2MC", inference_type="SD"))
    backend = FakeBackend({
        "tennis ball": PixelMeasurement("tennis ball", extent_px=40.0, confidence=0.9),
        "soccer ball": PixelMeasurement("soccer ball", speed_px_per_s=200.0, confidence=0.7),
    })
    answer = solve_row(request, backend, "clip.mp4")
    assert answer.value == pytest.approx(0.067 * 200 / 40)
    assert ("tennis ball", "length") in backend.calls      # prior measured as a length
    assert ("soccer ball", "speed") in backend.calls       # target measured as a speed


def test_gravity_only_prior_cannot_scale_a_length_question() -> None:
    """Gravity is a constant, not a scene measurement -- it fixes no pixel scale."""
    request = build_request(row(prior="gravity acc = 9.8m/s^2"))
    answer = solve_row(request, FakeBackend({}), "clip.mp4")
    assert not answer.solved and "gravity" in answer.reason


def test_prior_naming_no_object_declines_instead_of_grounding_a_constant() -> None:
    """``acceleration = 9.8 m/s^2`` -- 40 test rows. There is nothing in frame to measure.

    The old path grounded an object literally called "acceleration" and scaled the answer off
    whatever box came back. Declining sends the row to the fallback, which is strictly better.
    """
    request = build_request(row(prior="acceleration = 9.8 m/s^2"))
    answer = solve_row(request, FakeBackend({}), "clip.mp4")
    assert not answer.solved and "no groundable object" in answer.reason


def test_target_falling_back_to_the_prior_object_is_marked_in_the_method() -> None:
    """When the question yields no target phrase we reuse the prior's, which makes different
    questions collapse onto one answer. Still emit it -- a blank scores zero -- but say so."""
    request = build_request(row(question="How wide is it, in cm?"))
    assert request.target_object is None
    backend = FakeBackend({
        "tennis ball": PixelMeasurement("tennis ball", extent_px=40.0, confidence=0.9),
    })
    answer = solve_row(request, backend, "clip.mp4")
    assert answer.solved
    assert "target-from-prior" in answer.method


def test_unmeasurable_target_fails_softly_with_a_reason() -> None:
    request = build_request(row())
    backend = FakeBackend({
        "tennis ball": PixelMeasurement("tennis ball", extent_px=40.0, confidence=0.9),
    })
    answer = solve_row(request, backend, "clip.mp4")
    assert not answer.solved and "target not measured" in answer.reason


def test_low_confidence_rows_are_withheld_for_fallback() -> None:
    request = build_request(row())
    backend = FakeBackend({
        "tennis ball": PixelMeasurement("tennis ball", extent_px=40.0, confidence=0.9),
        "soccer ball": PixelMeasurement("soccer ball", extent_px=104.0, confidence=0.05),
    })
    answer = solve_row(request, backend, "clip.mp4", min_confidence=0.2)
    assert not answer.solved and "below" in answer.reason


def test_null_backend_yields_no_answer_but_never_raises() -> None:
    answer = solve_row(build_request(row()), NullBackend(), "clip.mp4")
    assert not answer.solved and answer.value is None


def test_end_to_end_answer_is_linear_in_the_prior() -> None:
    """The counterfactual robustness claim, checked through the full solve path."""
    backend = FakeBackend({
        "tennis ball": PixelMeasurement("tennis ball", extent_px=40.0, confidence=0.9),
        "soccer ball": PixelMeasurement("soccer ball", extent_px=104.0, confidence=0.9),
    })
    base = solve_row(build_request(row()), backend, "c.mp4").value
    for alpha in (0.001, 0.1, 10, 1000):
        scaled = solve_row(
            build_request(row(prior=f"diameter of the tennis ball = {6.7 * alpha}cm")),
            backend, "c.mp4").value
        assert scaled == pytest.approx(base * alpha, rel=1e-9)


def test_backends_satisfy_the_declared_protocol() -> None:
    assert isinstance(NullBackend(), VisionBackend)
    assert isinstance(FakeBackend({}), VisionBackend)
