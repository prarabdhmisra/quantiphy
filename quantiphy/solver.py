"""End-to-end solve: one dataset row plus a vision backend to one number in the requested unit.

The order of operations is deliberate. Scale is recovered from the prior's object, not the
target's, because the prior is the only thing in the row tied to real-world units. Everything else
is a conversion.

Failure is always soft. Any row that cannot be solved geometrically returns ``value=None`` with a
reason attached, and the caller substitutes a VLM or prior-based estimate. Emitting nothing scores
a hard zero under this metric, so there is no such thing as an acceptable gap.
"""

from __future__ import annotations

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


def solve_row(request: SolveRequest, backend: VisionBackend, video_path: str,
              min_confidence: float = 0.0) -> Answer:
    """Solve one row geometrically, or explain why it could not be solved."""
    unit = request.output_unit

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
    target_pixels = target_measurement.value_for(request.dimension)
    if not target_pixels:
        return Answer(None, unit, reason=f"target not measured: {target_measurement.note}")

    prior_depth = geometry.depth_for(request.depths, prior.object_name, prior.timestamp)
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

    method = "geometric-3d" if use_depth else "geometric-2d"
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
