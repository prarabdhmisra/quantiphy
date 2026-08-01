"""The contract between the CPU solver and the GPU vision stack.

Everything above this line (units, parsing, geometry) is deterministic and runs anywhere.
Everything below it needs a GPU and runs on Kaggle or Colab. Keeping the seam explicit means the
scale math can be tested exhaustively without ever loading a model, and the vision stack can be
swapped (SAM2 vs. a cheaper detector, CoTracker vs. RAFT) without touching the physics.

Planned implementation, all open-weight so it stays Track-B legal:

* **Grounding-DINO** for open-vocabulary detection from the parsed ``target_object`` phrase.
* **SAM2** for a per-frame mask, giving a far cleaner pixel extent than a bounding box.
* **CoTracker3** for centroid trajectories; velocity and acceleration come from a robust
  polynomial fit over the trajectory rather than raw finite differences, which are dominated by
  tracking jitter at 24 fps.

The measurement returns a ``confidence`` because the blowup detector depends on it: low-confidence
pixel measurements are exactly the rows where the answer should be shrunk toward the VLM estimate
rather than trusted, and overshooting is what costs whole points under this metric.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from quantiphy.parsing import SolveRequest


@dataclass(frozen=True)
class PixelMeasurement:
    """A measurement in pixel space, before any scale is applied.

    Exactly one of the value fields is meaningful per request, selected by the request's
    dimension: ``extent_px`` for lengths, ``speed_px_per_s`` for speeds, ``accel_px_per_s2`` for
    accelerations.
    """

    object_name: str
    extent_px: float | None = None
    speed_px_per_s: float | None = None
    accel_px_per_s2: float | None = None
    confidence: float = 0.0
    frames_tracked: int = 0
    note: str = ""

    def value_for(self, dimension: str) -> float | None:
        return {
            "length": self.extent_px,
            "speed": self.speed_px_per_s,
            "acceleration": self.accel_px_per_s2,
        }.get(dimension)


@runtime_checkable
class VisionBackend(Protocol):
    """What the GPU stack must provide for the solver to run."""

    def measure(self, request: SolveRequest, video_path: str,
                object_name: str, dimension: str) -> PixelMeasurement:
        """Measure one named object in pixel space.

        Called at least twice per row -- once for the prior's object (to fix the scale) and once
        for the target. Must not raise: return a zero-confidence measurement instead, so the
        solver can fall back rather than dropping the row and scoring a hard zero.
        """
        ...


class NullBackend:
    """Placeholder that measures nothing, so the CPU pipeline is runnable end to end.

    Every row falls back, which is the correct behaviour to exercise before the GPU stack lands.
    """

    def measure(self, request: SolveRequest, video_path: str,
                object_name: str, dimension: str) -> PixelMeasurement:
        return PixelMeasurement(object_name=object_name, confidence=0.0,
                                note="NullBackend: no vision stack attached")
