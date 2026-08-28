"""Combine the geometric solver's answer and the VLM's on rows both of them measured.

Referenced from :mod:`quantiphy.backends.vlm` since that module was written and unbuilt until
2026-08-28, on the reasoning that the two arms return different things -- pixels scaled by a prior
versus an answer in the question's own unit -- and should be *combined* rather than substituted.

**Log space, never an arithmetic mean.** These are physical magnitudes spanning orders of magnitude
and the metric is relative, so the mean of 1 and 100 is 10, not 50.5. That rule predates this module
and is the one thing here that is not in doubt.

What *is* in doubt is whether fusing beats picking, and the honest answer from the only data with
ground truth is **not established**. Measured on the 60 validation rows where the solver answered and
the VLM answered via its sentinel route:

| estimator | MRA | vs VLM alone |
|---|---|---|
| solver alone | 0.4667 | |
| VLM alone | **0.5033** | |
| geometric mean | 0.4767 | **-0.027, CI [-0.125, +0.063]** |
| min of the two | 0.5250 | +0.022, CI [-0.038, +0.082] |
| geometric mean, capped at 5x disagreement | 0.5617 | +0.058, CI [+0.007, +0.110] |

Three things follow, and the module's shape comes from them.

**The premise of averaging is half true.** Truth lies *between* the two arms on only 46.7% of those
rows. When one arm is right and the other is wildly wrong, the mean converts one winner into two
losers, which is why the plain geometric mean *loses* to the better arm.

**So agreement is the gate.** Fusing only where the arms agree within a factor is the one variant
whose interval excludes zero -- but **that cap was the argmax of a six-point sweep on those same 60
rows, so its significance is contaminated by selection.** This project has been burned by exactly
that three times; treat 5.0 as a starting point to bracket on the test split, not as a measurement.

**The mechanism that is real is the overshoot rate.** Fatal overshoots (>1.9x truth, a hard zero)
run at 20% for the solver and 15% for the VLM, and **5% for the minimum of the two** -- you can only
overshoot if both arms do. Undershoot is cheap and overshoot is fatal, so ``weight`` below 0.5 and
``prefer_lower`` exist for that reason and not as free parameters to tune.
"""

from __future__ import annotations

import math

#: Log-space weight on the *secondary* arm. 0.5 is the geometric mean; 0.0 keeps the primary.
#:
#: Not fitted. The 60-row overlap ranks 0.3 above 0.5 (0.5100 against 0.4767) and the direction has a
#: mechanism -- weighting toward the arm you trust more loses less when the other one is broken -- but
#: no weight's interval excludes zero there, so the default stays at the unweighted mean rather than
#: at the sweep's best value.
WEIGHT = 0.5

#: Widest factor between the two arms that still allows fusing, as a fold factor either way.
#:
#: See the module docstring: this is the sweep's argmax on 60 rows and is **not** confirmed. Bracket
#: it per category on the test split -- one submission carries four independent channels, so 3, 5 and
#: 10 can be measured against a no-fusion control in a single slot.
MAX_DISAGREEMENT = 5.0

#: Routes :func:`fuse` can report, so a probe can select rows by how they were combined.
FUSED = "fused"
PRIMARY_ONLY = "primary-only"
SECONDARY_ONLY = "secondary-only"
TOO_FAR = "disagreement"
NEITHER = "none"


def usable(value: float | None) -> bool:
    """Whether an arm's answer can be combined at all.

    Zero and negative are rejected rather than clamped: the scorer takes a magnitude and gives a hard
    zero to a zero prediction, and ``log(0)`` is not a number this module should be asked to average.
    845 of the VLM's 3,289 test replies are literally ``0``, so this branch is the common case, not
    an edge case.
    """
    return (value is not None and math.isfinite(value) and value > 0)


def disagreement(first: float | None, second: float | None) -> float | None:
    """How far apart two answers are, as a factor >= 1 either way, or None if not comparable."""
    if not usable(first) or not usable(second):
        return None
    ratio = first / second
    return max(ratio, 1.0 / ratio)


def log_mean(first: float, second: float, weight: float = WEIGHT) -> float:
    """Weighted geometric mean: ``first**(1-weight) * second**weight``.

    Exactly scale-equivariant, which is why it and not the arithmetic mean belongs here: scale both
    inputs by ``k`` and the result scales by ``k``. That preserves the counterfactual-prior property
    the geometric arm is built around, so fusing cannot smuggle in a fixed real-world size.
    """
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight must be in [0, 1], got {weight!r}")
    return math.exp((1.0 - weight) * math.log(first) + weight * math.log(second))


def fuse(primary: float | None, secondary: float | None, *, weight: float = WEIGHT,
         max_disagreement: float = MAX_DISAGREEMENT,
         prefer_lower: bool = False) -> tuple[float | None, str]:
    """One answer from two, with the route that produced it.

    ``primary`` is the arm kept when the two disagree too far to average -- so it should be whichever
    arm measured better *in this category*, which the portal reports directly. Falling back to the
    primary rather than declining is deliberate: both arms answering is exactly the case where a
    decline would throw away two measurements to avoid choosing between them.

    ``prefer_lower`` takes the smaller of the two instead of averaging, which is the one variant with
    a mechanism rather than a fitted parameter: it cuts the fatal-overshoot rate from 15-20% to 5%
    because overshooting requires *both* arms to overshoot, and under MRA a 1.9x overshoot scores
    zero while a 0.5x undershoot still scores 0.4.
    """
    if max_disagreement < 1.0:
        raise ValueError(f"max_disagreement is a fold factor >= 1, got {max_disagreement!r}")

    if not usable(primary) and not usable(secondary):
        return None, NEITHER
    if not usable(secondary):
        return primary, PRIMARY_ONLY
    if not usable(primary):
        return secondary, SECONDARY_ONLY

    if disagreement(primary, secondary) > max_disagreement:
        return primary, TOO_FAR
    if prefer_lower:
        return min(primary, secondary), FUSED
    return log_mean(primary, secondary, weight), FUSED
