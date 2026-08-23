"""End-to-end solve: one dataset row plus a vision backend to one number in the requested unit.

The order of operations is deliberate. Scale is recovered from the prior's object, not the
target's, because the prior is the only thing in the row tied to real-world units. Everything else
is a conversion.

Failure is always soft. Any row that cannot be solved geometrically returns ``value=None`` with a
reason attached, and the caller substitutes a VLM or prior-based estimate. Emitting nothing scores
a hard zero under this metric, so there is no such thing as an acceptable gap.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from quantiphy import geometry
from quantiphy.parsing import SolveRequest
from quantiphy.units import from_si
from quantiphy.vision import PixelMeasurement, VisionBackend


@dataclass(frozen=True)
class Answer:
    """A solved row, in the unit the question asked for."""

    value: float | None
    unit: str
    confidence: float = 0.0
    method: str = "none"
    reason: str = ""
    prior_pixels: float | None = None
    target_pixels: float | None = None

    @property
    def solved(self) -> bool:
        return self.value is not None


def _measure_prior(request: SolveRequest, backend: VisionBackend,
                   video_path: str) -> tuple[PixelMeasurement | None, object]:
    """Measure the prior's object in pixels, using the prior's own quantity and timestamp."""
    prior = request.scale_prior
    if prior is None:
        return None, None
    if prior.is_gravity and request.dimension != "acceleration":
        # Gravity fixes no pixel scale for lengths or speeds; there is nothing to measure.
        return None, prior

    prior_request = replace(
        request,
        measurement=prior.quantity,
        timestamp=prior.timestamp if prior.timestamp is not None else request.timestamp,
        interval=None,
    )
    return backend.measure(prior_request, video_path, prior.object_name, prior.dimension), prior


#: Pixel span a scale prior must occupy for its answer to be trusted, as ``(low, high)``.
#:
#: This replaces gating on detector confidence, which was measured and refuted: every threshold from
#: 0.0 to 0.5 scored *below* a constant predictor, because ``mean_score x detection_rate`` says
#: nothing about whether the prior measured the right thing. ``prior_pixels`` does. Across 147
#: replayed validation rows ``log(pred/truth) ~= -0.87 * log(prior_pixels)`` with correlation
#: -0.725: almost the entire error is the prior's own pixel measurement. Median pred/truth runs 7.04
#: for priors under 25 px and 0.08 for priors over 400 px, and both of those score a hard zero.
#:
#: One wart, recorded rather than hidden: ``prior_pixels`` is a pixel *extent* for a length prior but
#: a pixel *speed* for a speed prior, so this single band is applied to two different quantities. It
#: earns its keep empirically at both ends -- the 16 rows declined at "0.1-0.2 px" are speed priors
#: whose object was measured as effectively stationary, which is just as unusable as a 2-pixel box --
#: but the right fix is eventually two bands, not one. Do not read the edges as physically meaningful
#: until that is separated.
#:
#: Inside the band the solver is worth firing; outside it the caller's fallback beats it. Treat the
#: exact edges as provisional -- they were chosen from six variants on 159 rows, and while the
#: underlying correlation is strong the resulting macro gain has a 95% CI of [-0.028, +0.087], which
#: does not exclude zero. Confirm against the test split before building anything on the numbers.
TRUSTED_PRIOR_PIXELS = (30.0, 300.0)


def _separation_px(first: PixelMeasurement, second: PixelMeasurement) -> float | None:
    """Pixel gap between two located objects, or None when either could not be localised.

    None rather than a guess on purpose: a missing second object silently falling back to the first
    object's own extent is precisely the "confidently wrong" answer this path exists to remove.
    """
    if first.centroid_px is None or second.centroid_px is None:
        return None
    (x1, y1), (x2, y2) = first.centroid_px, second.centroid_px
    return math.hypot(x2 - x1, y2 - y1) or None


def solve_row(request: SolveRequest, backend: VisionBackend, video_path: str,
              min_confidence: float = 0.0,
              trusted_prior_pixels: tuple[float, float] = TRUSTED_PRIOR_PIXELS) -> Answer:
    """Solve one row geometrically, or explain why it could not be solved.

    Widen ``trusted_prior_pixels`` to ``(0.0, float("inf"))`` to measure the solver ungated -- which
    is what the replay harness does when it needs the raw distribution rather than the gated answer.
    """
    unit = request.output_unit

    if request.measurement == "distance-camera" and request.dimension == "length":
        # "the distance between the cat and the camera" is a depth query, and depth_info states the
        # answer outright. No prior, no scale transfer, no detection -- and no error either. The
        # dimension guard matters: a depth is metres, so emitting it into a question whose unit
        # parsed as m/s would be a silent unit error of exactly the kind this codebase avoids.
        depth = geometry.depth_for(request.depths, request.target_object, request.timestamp)
        if depth is None:
            return Answer(None, unit, reason="distance-to-camera with no matching depth reading")
        return Answer(value=from_si(depth, unit), unit=unit, confidence=1.0,
                      method="depth-info-direct")

    if request.measurement == "distance-twin":
        # "the distance between the two cars": one phrase, two instances of it. The detector keeps
        # only the single best box per frame, so the pair cannot be recovered. Declining costs a
        # fallback; measuring one car's box would confidently report ~1/3 of the real separation,
        # and a 3x overshoot or undershoot scores zero either way.
        return Answer(None, unit, reason="separation between two instances of one phrase "
                                         "needs top-2 detections")

    prior_measurement, prior = _measure_prior(request, backend, video_path)
    if prior is None:
        return Answer(None, unit, reason="no usable scale prior")
    if prior_measurement is None:
        return Answer(None, unit, reason="gravity prior cannot set pixel scale")
    if not prior.object_name:
        # A constant with no object in frame ("acceleration = 9.8 m/s^2"). Nothing to measure, so
        # nothing can set the scale -- decline and let the fallback fill it.
        return Answer(None, unit, reason="prior names no groundable object")

    prior_pixels = prior_measurement.value_for(prior.dimension)
    if not prior_pixels:
        return Answer(None, unit, reason=f"prior object not measured: {prior_measurement.note}")

    target_name = request.target_object or prior.object_name
    target_measurement = backend.measure(request, video_path, target_name, request.dimension)

    separation = None
    if request.measurement == "distance-pair" and request.target_object_b:
        # The question asks how far apart two objects are, which is the gap between their centroids
        # -- not either one's own box extent. Measuring the extent here is what made these rows
        # report ~60 px where the geometry needed ~150-180 px.
        second = backend.measure(request, video_path, request.target_object_b, request.dimension)
        separation = _separation_px(target_measurement, second)
        if separation is None:
            return Answer(None, unit,
                          reason=f"separation needs both objects located: "
                                 f"{target_measurement.note or 'ok'} / {second.note or 'ok'}")
        target_measurement = replace(
            second, object_name=f"{target_name} <-> {request.target_object_b}",
            confidence=min(target_measurement.confidence, second.confidence))

    target_pixels = (separation if separation is not None
                     else target_measurement.value_for(request.dimension))
    if not target_pixels:
        return Answer(None, unit, reason=f"target not measured: {target_measurement.note}")

    # Read the prior's depth at the instant its *pixels* were measured, which is what
    # `prior_request` above actually used -- not at `prior.timestamp`, which is frequently None.
    # When it is None, `depth_for` falls through to the first reading in file order, so a prior and
    # a target that are the same object at different instants get a ratio that should be exactly 1.0
    # and is instead whatever two arbitrary readings divide to: measured at 1.30 and 0.68 on real
    # rows. 1,284 of the 1,548 3D test rows carry more than one timed reading, and 310 of them have
    # a same-object depth spread of 1.25x or worse, so this is the largest defect in the 3D path.
    prior_depth_at = prior.timestamp if prior.timestamp is not None else request.timestamp
    prior_depth = geometry.depth_for(request.depths, prior.object_name, prior_depth_at)
    target_depth = geometry.depth_for(request.depths, target_name, request.timestamp)
    use_depth = request.is_3d and prior_depth is not None and target_depth is not None

    radial = None
    if request.dimension == "speed" and request.is_3d:
        radial = geometry.radial_speed(request.depths, target_name)

    try:
        value_si = geometry.solve(
            target_pixels=target_pixels,
            prior_world_si=prior.value_si,
            prior_pixels=prior_pixels,
            target_depth_m=target_depth if use_depth else None,
            prior_depth_m=prior_depth if use_depth else None,
            radial_si=radial,
            is_speed=(request.dimension == "speed"),
        )
    except geometry.ScaleError as error:
        return Answer(None, unit, reason=str(error))

    confidence = min(prior_measurement.confidence, target_measurement.confidence)
    if confidence < min_confidence:
        return Answer(None, unit, confidence=confidence,
                      reason=f"confidence {confidence:.3f} below {min_confidence:.3f}")

    low, high = trusted_prior_pixels
    if not low <= prior_pixels <= high:
        # Declining is the point. The answer we would emit here is not merely noisy, it is biased by
        # a known factor in a known direction -- and under MRA a 7x overshoot and a 0.08x undershoot
        # both score exactly zero, so there is nothing to salvage by emitting it anyway.
        return Answer(None, unit, confidence=confidence, prior_pixels=prior_pixels,
                      reason=f"prior measured {prior_pixels:.1f} px, outside the trusted "
                             f"{low:.0f}-{high:.0f} px band")

    method = "geometric-3d" if use_depth else "geometric-2d"
    if separation is not None:
        method += "+separation"
    if radial:
        method += "+radial"
    if not request.target_object:
        # We measured the prior's object for want of a target phrase, so every question about this
        # video collapses onto one answer. Still emit it -- a blank scores a hard zero -- but leave
        # a mark, or a collapsed row is indistinguishable from a real solve in the output.
        method += "+target-from-prior"

    return Answer(
        value=from_si(value_si, unit),
        unit=unit,
        confidence=confidence,
        method=method,
        prior_pixels=prior_pixels,
        target_pixels=target_pixels,
    )
