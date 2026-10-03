"""Pins the track-quality measures that decide whether a mask tracker is worth buying."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))

from audit_tracks import track_quality  # noqa: E402
from quantiphy.backends.grounding import DetectionSeries  # noqa: E402


def _series(cx: list[float], width: list[float] | None = None) -> DetectionSeries:
    n = len(cx)
    width = width or [20.0] * n
    return DetectionSeries(times=np.linspace(0, 1, n), cx=np.array(cx, dtype=float),
                           cy=np.zeros(n), width=np.array(width, dtype=float),
                           height=np.full(n, 20.0), scores=np.full(n, 0.5),
                           frames_total=n, frames_sampled=n)


def test_a_smooth_track_has_no_jumps_and_a_steady_box() -> None:
    quality = track_quality(_series([float(x) for x in range(0, 100, 5)]))

    assert quality["jump_rate"] == 0.0
    assert quality["area_cv"] == 0.0


def test_an_identity_swap_registers_as_a_jump() -> None:
    # Steady 5 px steps, then one 200 px leap to another object and back.
    cx = [0, 5, 10, 15, 215, 20, 25, 30, 35, 40]

    quality = track_quality(_series([float(x) for x in cx]))

    assert quality["jump_rate"] == 2 / 9


def test_a_breathing_box_raises_area_cv() -> None:
    quality = track_quality(_series([0.0] * 6, width=[10, 30, 10, 30, 10, 30]))

    assert quality["area_cv"] > 0.4


def test_too_few_samples_returns_nan_rather_than_a_false_clean_bill() -> None:
    quality = track_quality(_series([0.0, 1.0]))

    assert np.isnan(quality["jump_rate"])
