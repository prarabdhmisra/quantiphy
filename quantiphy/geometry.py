"""Pixel-to-world scale recovery.

This is the physics core of the solver. Given a pixel-space measurement and one real-world prior,
recover the requested quantity. No world knowledge and no learned model is involved, which is
precisely why this survives the counterfactual-prior test that every VLM in the benchmark paper
fails: multiply the prior by 1000 and the output scales by exactly 1000.

**2D (planar motion, constant depth).** One scale factor gamma [metres per pixel] links pixels to
the world, and it transfers directly between objects and between quantities::

    gamma = prior_world / prior_pixels          # m/px, from any length, speed, or acceleration
    target_world = gamma * target_pixels

Speeds work identically because gamma cancels the same way for px/s as for px, and accelerations
for px/s^2 -- provided pixel rates use real seconds, which is why ``fps`` matters.

**3D (perspective).** Under a pinhole camera a world length ``L`` at depth ``Z`` subtends
``L_px = f * L / Z``, so gamma is no longer constant -- it is ``Z / f``. The prior fixes the focal
length once, and every other object is then read off at its own depth::

    f = Z_prior * prior_pixels / prior_world    # pixels
    target_world = target_pixels * Z_target / f

which is the depth ratio ``Z_target / Z_prior`` applied to the 2D answer. This is the step VLMs
have no way to perform, and it is why the ``depth_info`` column exists.

**Radial motion.** ``depth_info`` gives distance-to-camera at several timestamps, so motion toward
or away from the camera is recoverable as ``dZ/dt`` and combined with in-plane motion in
quadrature. Ignoring it under-reports speed -- which the metric punishes far less than
over-reporting, so the fallback is deliberately the conservative one.
"""

from __future__ import annotations

import math

from quantiphy.parsing import DepthReading


class ScaleError(ValueError):
    """Raised when a scale factor cannot be recovered from the available measurements."""


def gamma_from_prior(prior_world_si: float, prior_pixels: float) -> float:
    """Metres per pixel (or m/s per px/s, etc.) implied by a prior and its pixel measurement."""
    if prior_pixels == 0 or not math.isfinite(prior_pixels):
        raise ScaleError(f"prior pixel measurement is unusable: {prior_pixels!r}")
    if not math.isfinite(prior_world_si):
        raise ScaleError(f"prior world value is unusable: {prior_world_si!r}")
    return abs(prior_world_si) / abs(prior_pixels)


def focal_length_px(prior_world_si: float, prior_pixels: float, prior_depth_m: float) -> float:
    """Recover focal length in pixels from one metric prior at a known depth."""
    if prior_depth_m <= 0 or not math.isfinite(prior_depth_m):
        raise ScaleError(f"prior depth is unusable: {prior_depth_m!r}")
    return abs(prior_pixels) * prior_depth_m / abs(prior_world_si)


def world_from_pixels(target_pixels: float, *, gamma: float | None = None,
                      focal_px: float | None = None, depth_m: float | None = None) -> float:
    """Convert a pixel measurement to SI, either by a flat 2D scale or by pinhole projection."""
    if gamma is not None:
        return abs(target_pixels) * gamma
    if focal_px is not None and depth_m is not None:
        if focal_px <= 0:
            raise ScaleError(f"focal length is unusable: {focal_px!r}")
        return abs(target_pixels) * depth_m / focal_px
    raise ScaleError("need either gamma, or both focal_px and depth_m")


#: Widest depth ratio we will apply, as a factor either way.
#:
#: The correction's *direction* is right (a farther object subtending the same pixels really is
#: larger), but nothing downstream bounds its magnitude, and the 3D path has three independent
#: one-way inflation mechanisms feeding it: an arbitrary reading picked by file order, a left/right
#: tie broken by position, and ``combine_speeds``'s ``hypot``, which can only ever push a speed up.
#: Under MRA a prediction at 1.9x truth scores exactly zero, so an unbounded multiplier is not a
#: rounding risk, it is a total loss. Beyond this factor the reading pair is far likelier to be a
#: mismatch than a real perspective change, so we decline instead of emitting it.
MAX_DEPTH_RATIO = 4.0


def depth_ratio_correction(value_2d: float, target_depth_m: float, prior_depth_m: float) -> float:
    """Apply the ``Z_target / Z_prior`` correction to a value computed with a flat 2D scale.

    Equivalent to the pinhole route but expressed as a correction, which is convenient when the
    2D estimate already exists and only the depths differ.
    """
    if prior_depth_m <= 0:
        raise ScaleError(f"prior depth is unusable: {prior_depth_m!r}")
    if target_depth_m <= 0:
        raise ScaleError(f"target depth is unusable: {target_depth_m!r}")
    ratio = target_depth_m / prior_depth_m
    if not 1.0 / MAX_DEPTH_RATIO <= ratio <= MAX_DEPTH_RATIO:
        raise ScaleError(
            f"depth ratio {ratio:.2f} exceeds {MAX_DEPTH_RATIO:g}x; the two readings "
            f"({prior_depth_m:.3g} m and {target_depth_m:.3g} m) are more likely mismatched "
            f"than that far apart")
    return value_2d * ratio


def radial_speed(depths: tuple[DepthReading, ...], object_name: str) -> float | None:
    """Speed along the camera axis from timestamped depth readings, or None if underdetermined.

    Matches the object the same way :func:`depth_for` does. It used to use a raw case-sensitive
    ``in`` against an un-lowercased key, and target phrases arrive from ``_strip_object`` with their
    original capitalisation -- so "Car" silently found nothing where "car" found a reading, and any
    capitalised phrase dropped the radial component without a word. Reusing ``_name_overlap`` also
    stops "left tennis ball" matching a reading for the *right* one by bare substring.
    """
    wanted = set(object_name.lower().split()) if object_name else set()
    if not wanted:
        return None
    timed = sorted(
        (d for d in depths
         if d.timestamp is not None
         and _name_overlap(wanted, set(d.object_name.lower().split()))),
        key=lambda d: d.timestamp,
    )
    if len(timed) < 2:
        return None
    span = timed[-1].timestamp - timed[0].timestamp
    if span <= 0:
        return None
    return (timed[-1].distance_m - timed[0].distance_m) / span


def combine_speeds(tangential_si: float, radial_si: float | None) -> float:
    """Total speed from in-plane and along-axis components."""
    if radial_si is None:
        return abs(tangential_si)
    return math.hypot(tangential_si, radial_si)


#: Shortest token that may match by containment. Below this, fragments collide ("car" in "cart").
_MIN_PARTIAL_TOKEN = 4


def _name_overlap(wanted: set[str], tokens: set[str]) -> float:
    """How well two object names agree, counting exact token hits above partial ones.

    A compound noun is often one word in the question and two in ``depth_info``, or the other way
    round: the question asks about the "basketball" while the key is ``distance_ball_camera``. Whole
    token equality alone scores that zero, so the row silently loses a depth it actually has.
    Containment hits are worth half in total -- not half *each*. Summing them uncapped broke the
    guarantee this docstring makes: the cross product is unbounded, so a key naming three objects
    whose names all contain the wanted token scored 1.5 and beat the exact key's 1.0. Against real
    ``depth_info`` that turned ``distance_ball_camera = 5.0m`` into
    ``distance_basketball_football_volleyball_camera = 40.0m``, an 8x error. No test-split block
    currently triggers it, so this is a latent bug rather than a live one -- but it is one line.
    """
    exact = float(len(wanted & tokens))
    partial = any(
        len(left) >= _MIN_PARTIAL_TOKEN and len(right) >= _MIN_PARTIAL_TOKEN
        and (left in right or right in left)
        for left in wanted - tokens
        for right in tokens - wanted
    )
    return exact + (0.5 if partial else 0.0)


def depth_for(depths: tuple[DepthReading, ...], object_name: str | None,
              timestamp: float | None = None) -> float | None:
    """Best available distance-to-camera for an object, preferring the nearest timestamp.

    Object names are matched loosely because ``depth_info`` uses underscored keys
    (``distance_right_tennis_ball_camera``) while questions use prose ("the right tennis ball").
    """
    if not depths:
        return None
    if not object_name:
        # With no name to match, a single unambiguous reading is still usable.
        unique = {d.distance_m for d in depths}
        return depths[0].distance_m if len(unique) == 1 else None

    wanted = set(object_name.lower().split())
    scored = []
    for reading in depths:
        overlap = _name_overlap(wanted, set(reading.object_name.split()))
        if overlap:
            scored.append((overlap, reading))
    if not scored:
        return None

    best = max(score for score, _ in scored)
    candidates = [reading for score, reading in scored if score == best]

    if timestamp is not None:
        timed = [r for r in candidates if r.timestamp is not None]
        if timed:
            return min(timed, key=lambda r: abs(r.timestamp - timestamp)).distance_m
    return candidates[0].distance_m


def solve(*, target_pixels: float, prior_world_si: float, prior_pixels: float,
          target_depth_m: float | None = None, prior_depth_m: float | None = None,
          radial_si: float | None = None, is_speed: bool = False) -> float:
    """Full scale transfer: pixel measurement plus one prior to an SI answer.

    Applies the perspective correction whenever both depths are known, and folds in radial motion
    for speeds. Falls back to the flat 2D scale when depth is unavailable.
    """
    gamma = gamma_from_prior(prior_world_si, prior_pixels)
    value = world_from_pixels(target_pixels, gamma=gamma)

    if target_depth_m is not None and prior_depth_m is not None:
        value = depth_ratio_correction(value, target_depth_m, prior_depth_m)

    if is_speed:
        value = combine_speeds(value, radial_si)
    return value
