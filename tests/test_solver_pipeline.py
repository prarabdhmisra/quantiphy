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
    centroid_at,
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


def test_local_fit_survives_per_frame_jitter() -> None:
    """The reason we fit at all: 1 px of jitter at 24 fps is 24 px/s of noise.

    Averaged over many draws rather than pinned to one seed, because a single draw of this cannot
    tell a real regression from luck -- the per-draw error swings by more than the effect size.

    Note what is deliberately *not* asserted: that the fit beats naive differencing. It no longer
    does on this input. ``mean(|dx/dt|)`` is unbeatable on a monotone ramp with symmetric noise,
    and a fit over the whole clip used to beat it too (mean error 0.20 px/s, winning 250 draws in
    300). :data:`FIT_WINDOW_S` gives that up on purpose -- see the next test for what it buys.
    """
    times = np.linspace(0, 2, 49)
    jitter_scale = 1.0 / float(times[1] - times[0])          # 24 px/s of per-sample noise

    errors = []
    for seed in range(200):
        noise = np.random.default_rng(seed).normal(0, 1.0, times.size)
        series = make_series(times, 100 * times + noise, np.zeros_like(times))
        errors.append(abs(kinematics(series, at_time=1.0)[0] - 100.0))

    # The guarantee is suppression of the noise scale, not perfection: ~1 px/s out of 24 px/s in.
    assert float(np.mean(errors)) < jitter_scale / 8.0
    assert float(np.percentile(errors, 95)) < jitter_scale / 4.0


def test_local_fit_recovers_speed_a_global_fit_averages_away() -> None:
    """Why the fit window exists, and the single largest lever measured on this benchmark.

    A saw cutting back and forth, a pencil drawing, a person pacing: motion that is not one clean
    parabola over the whole clip. One quadratic spanning all of it fits a nearly flat line through
    the oscillation and reports an object at rest. Velocity is 900 of the 3,289 test rows, so that
    failure was costing more than any other single defect.
    """
    times = np.linspace(0, 8, 193)                  # 8 s at ~24 fps
    speed = 300.0
    period = 2.0
    # A triangle wave: constant |speed|, reversing direction every half period.
    phase = (times % period) / period
    cx = np.where(phase < 0.5, speed * period * phase, speed * period * (1.0 - phase))
    series = make_series(times, cx, np.zeros_like(times))

    local, _, _ = kinematics(series, at_time=0.5)            # mid-stroke, moving at full speed
    globally, _, _ = kinematics(series, at_time=0.5, half_width=1e9)

    assert local == pytest.approx(speed, rel=0.05)
    assert globally < speed / 10.0                            # the global fit sees a stationary object


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


# ------------------------------------------------------------------- separations

def test_separation_uses_the_gap_between_two_centroids_not_one_box() -> None:
    """The distance fix, checked through the full solve path.

    A tennis ball 6.7 cm across 40 px fixes gamma at 1.675 mm/px. Two objects whose centroids sit
    200 px apart are therefore 33.5 cm apart -- and crucially *not* the 80 px / 13.4 cm that either
    one's own box extent would have reported.
    """
    backend = FakeBackend({
        "tennis ball": PixelMeasurement("tennis ball", extent_px=40.0, confidence=0.9),
        "white car": PixelMeasurement("white car", extent_px=80.0, centroid_px=(100.0, 100.0),
                                      confidence=0.9),
        "bicycle": PixelMeasurement("bicycle", extent_px=80.0, centroid_px=(220.0, 260.0),
                                    confidence=0.9),
    })
    answer = solve_row(build_request(row(
        question="What is the distance between the white car and the bicycle in meters?")),
        backend, "c.mp4")

    assert answer.solved
    assert "+separation" in answer.method
    assert answer.value == pytest.approx(0.335, rel=1e-6)      # 200 px * 1.675 mm/px
    assert ("bicycle", "length") in backend.calls               # both objects were measured


def test_separation_declines_when_the_second_object_is_never_located() -> None:
    """Falling back to one object's extent here is the confidently-wrong answer to avoid."""
    backend = FakeBackend({
        "tennis ball": PixelMeasurement("tennis ball", extent_px=40.0, confidence=0.9),
        "white car": PixelMeasurement("white car", extent_px=80.0, centroid_px=(100.0, 100.0),
                                      confidence=0.9),
    })
    answer = solve_row(build_request(row(
        question="What is the distance between the white car and the bicycle in meters?")),
        backend, "c.mp4")

    assert not answer.solved
    assert "separation needs both objects located" in answer.reason


def test_twin_separation_declines_rather_than_measuring_one_instance() -> None:
    """"between the two cars" cannot be answered from a single best box per frame."""
    backend = FakeBackend({
        "tennis ball": PixelMeasurement("tennis ball", extent_px=40.0, confidence=0.9),
        "cars": PixelMeasurement("cars", extent_px=80.0, centroid_px=(100.0, 100.0),
                                 confidence=0.9),
    })
    answer = solve_row(build_request(row(
        question="What is the distance between the two cars in meters?")), backend, "c.mp4")

    assert not answer.solved
    assert "top-2 detections" in answer.reason


def test_centroid_at_evaluates_the_trajectory_at_the_requested_instant() -> None:
    """Both objects of a separation must be located at the same moment to be comparable."""
    times = np.linspace(0.0, 2.0, 9)
    series = DetectionSeries(
        times=times, cx=100.0 + 50.0 * times, cy=np.full_like(times, 30.0),
        width=np.full_like(times, 10.0), height=np.full_like(times, 10.0),
        scores=np.full_like(times, 0.9), frames_total=9, frames_sampled=9,
    )
    assert centroid_at(series, 1.0) == pytest.approx((150.0, 30.0), abs=1e-6)
    assert centroid_at(series, 2.0) == pytest.approx((200.0, 30.0), abs=1e-6)
    # With no instant named, the median position is the best available estimate.
    assert centroid_at(series, None) == pytest.approx((150.0, 30.0), abs=1e-6)


# ------------------------------- 2026-08-27: the prior's own motion toward the camera

def _speed_prior_row(**overrides) -> dict:
    """A 3D row whose prior is a speed and whose prior object also moves along the camera axis.

    This is the D3 population exactly: `D3` is `[dynamic prior][3D]`, so every one of its 972 rows
    carries a velocity or acceleration prior and a depth track.
    """
    fields = {
        "question": "What is the length of the boat in meters?",
        "prior": "speed of the ferry = 5.0m/s",
        "video_type": "V3MC", "inference_type": "DS",
        "depth_info": ("t=0.0s, distance_ferry_camera = 10.0m\n"
                       "t=1.0s, distance_ferry_camera = 13.0m\n"
                       "t=1.0s, distance_boat_camera = 10.0m"),
    }
    return row(**{**fields, **overrides})


def _boat_backend() -> FakeBackend:
    return FakeBackend({
        "ferry": PixelMeasurement("ferry", speed_px_per_s=100.0, confidence=0.9),
        "boat": PixelMeasurement("boat", extent_px=100.0, confidence=0.9),
    })


def test_a_3d_speed_prior_is_scaled_by_its_in_plane_component_only() -> None:
    """gamma must come from the 4.0 m/s the pixels could see, not the 5.0 m/s the prior states."""
    answer = solve_row(build_request(_speed_prior_row()), _boat_backend(), "c.mp4")
    assert "+prior-tangential" in answer.method
    assert answer.value == pytest.approx(4.0)          # was 5.0: a 25% overshoot on every such row


def test_a_2d_speed_prior_is_left_alone() -> None:
    """No perspective, so the pixel speed already is the whole speed."""
    answer = solve_row(build_request(_speed_prior_row(video_type="V2MC")),
                       _boat_backend(), "c.mp4")
    assert "+prior-tangential" not in answer.method
    assert answer.value == pytest.approx(5.0)


def test_an_impossible_prior_radial_leaves_the_scale_untouched() -> None:
    """Radial 9 m/s against a stated 5 m/s means the readings disagree with the prior. Emitting
    the tiny tangential part would drive gamma to ~0 and the answer to a hard zero."""
    answer = solve_row(build_request(_speed_prior_row(
        depth_info=("t=0.0s, distance_ferry_camera = 10.0m\n"
                    "t=1.0s, distance_ferry_camera = 19.0m\n"
                    "t=1.0s, distance_boat_camera = 10.0m"))), _boat_backend(), "c.mp4")
    assert "+prior-tangential" not in answer.method
    assert answer.value == pytest.approx(5.0)


def test_an_acceleration_prior_is_never_radially_corrected() -> None:
    """REFUTED 2026-08-27, and pinned so it is not re-added. The same decomposition on the 172 D3
    acceleration-prior rows with three or more depth readings measures a radial share of **1.50** --
    physically impossible, because d2Z/dt2 over a 2-3 s clip with 2-4 readings is noise. Applying it
    moved the median answer from 1.07x the constant to 1.90x and doubled the >1.9x share to 50%.
    """
    answer = solve_row(build_request(_speed_prior_row(
        prior="acceleration of the ferry = 5.0m/s^2",
        question="What is the acceleration of the boat in m/s^2?")),
        FakeBackend({
            "ferry": PixelMeasurement("ferry", accel_px_per_s2=100.0, confidence=0.9),
            "boat": PixelMeasurement("boat", accel_px_per_s2=100.0, confidence=0.9),
        }), "c.mp4")
    assert "+prior-tangential" not in answer.method
    assert answer.value == pytest.approx(5.0)
