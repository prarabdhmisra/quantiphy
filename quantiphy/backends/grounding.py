"""Grounding-DINO vision backend: open-vocabulary detection turned into pixel measurements.

This is the first GPU implementation of :class:`quantiphy.vision.VisionBackend`. It deliberately
uses detection alone -- no SAM2, no CoTracker -- because that is the smallest thing that produces
an end-to-end score we can measure. Mask-based extents and dedicated point tracking are worth real
accuracy, but only once we know what detection alone buys.

All weights are open, so this stays Track-B legal.

Two decisions that matter more than the model choice:

* **Detections are cached per (video, phrase), not per row.** 3,289 questions share only 568
  videos, and the same object is asked about repeatedly, so a naive per-row loop would redo the
  same inference roughly six times over. The cache persists to disk so reruns after a crash are
  nearly free.
* **Kinematics come from a robust quadratic fit over the whole trajectory**, not frame-to-frame
  differences. At 24 fps (2,478 of 3,289 rows) a one-pixel jitter between adjacent frames is
  24 px/s of phantom speed, and differencing twice for acceleration squares that problem. Fitting
  a quadratic and reading its derivatives is far steadier and gives a fit quality we can use as a
  confidence signal.

Nothing here raises. A failure returns a zero-confidence measurement so the solver falls back
instead of dropping the row -- a dropped row scores a hard zero.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quantiphy.parsing import SolveRequest
from quantiphy.vision import PixelMeasurement

DEFAULT_MODEL = "IDEA-Research/grounding-dino-base"


@dataclass
class DetectionSeries:
    """Per-frame best detection of one phrase in one video."""

    times: np.ndarray          # seconds
    cx: np.ndarray             # centroid x, pixels
    cy: np.ndarray             # centroid y, pixels
    width: np.ndarray          # bbox width, pixels
    height: np.ndarray         # bbox height, pixels
    scores: np.ndarray
    frames_total: int           # frames in the clip
    frames_sampled: int = 0     # frames we actually ran the detector on

    @property
    def detection_rate(self) -> float:
        """Share of the frames we *looked at* that produced a detection.

        Deliberately not a share of the clip: with ``max_frames`` capped at 48, dividing by a
        428-frame clip pinned confidence at 0.112 no matter how clean the detections were, so
        confidence tracked video duration instead of detection quality. Defaults to
        ``frames_total`` so caches pickled before this field existed still load.
        """
        denominator = self.frames_sampled or self.frames_total
        return len(self.times) / denominator if denominator else 0.0

    @property
    def mean_score(self) -> float:
        return float(self.scores.mean()) if self.scores.size else 0.0


def extent_for(series: DetectionSeries, measurement: str) -> float | None:
    """Pick the bbox dimension the question is actually asking about.

    The median across frames is used rather than the mean: detection boxes occasionally snap to a
    larger co-occurring object for a frame or two, and the median ignores those outright.
    """
    if series.width.size == 0:
        return None
    width = float(np.median(series.width))
    height = float(np.median(series.height))

    if measurement == "height":
        return height
    if measurement == "width":
        return width
    if measurement == "diameter":
        return (width + height) / 2.0          # round objects: average the two axes
    if measurement == "radius":
        return (width + height) / 4.0
    if measurement in {"calibre", "caliber"}:
        return min(width, height)              # a bore, not a length: the small axis
    if measurement in {"length", "size"}:
        return max(width, height)
    return max(width, height)


def _robust_quadratic(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit ``values ~ a*t^2 + b*t + c``, discarding outliers once. Returns (coeffs, r_squared)."""
    if times.size < 3:
        return np.zeros(3), 0.0
    degree = 2 if times.size >= 5 else 1
    coeffs = np.polyfit(times, values, degree)
    residuals = values - np.polyval(coeffs, times)
    spread = np.std(residuals)
    if spread > 0 and times.size >= 6:
        keep = np.abs(residuals) < 2.5 * spread
        if keep.sum() >= 3:
            coeffs = np.polyfit(times[keep], values[keep], degree)
            residuals = values - np.polyval(coeffs, times)

    total = np.sum((values - values.mean()) ** 2)
    if total <= 0:
        # A stationary axis has no variance to explain, so R^2 is undefined. It is perfectly
        # fit, not badly fit -- and most objects here move along one axis only (a ball falls, a
        # car drives horizontally), so scoring this 0 would depress confidence on most rows.
        r_squared = 1.0 if np.allclose(residuals, 0.0) else 0.0
    else:
        r_squared = 1.0 - np.sum(residuals ** 2) / total
    if degree == 1:
        coeffs = np.concatenate([[0.0], coeffs])
    return coeffs, float(max(0.0, r_squared))


def kinematics(series: DetectionSeries, at_time: float | None) -> tuple[float, float, float]:
    """Return ``(speed_px_per_s, accel_px_per_s2, fit_quality)`` from the centroid trajectory.

    Evaluated at ``at_time`` when the question names an instant, otherwise at the trajectory's
    midpoint, which is the most stable point of a quadratic fit.
    """
    if series.times.size < 3:
        return 0.0, 0.0, 0.0

    coeff_x, quality_x = _robust_quadratic(series.times, series.cx)
    coeff_y, quality_y = _robust_quadratic(series.times, series.cy)
    moment = at_time if at_time is not None else float(np.median(series.times))

    velocity_x = 2 * coeff_x[0] * moment + coeff_x[1]
    velocity_y = 2 * coeff_y[0] * moment + coeff_y[1]
    speed = float(np.hypot(velocity_x, velocity_y))
    accel = float(np.hypot(2 * coeff_x[0], 2 * coeff_y[0]))
    return speed, accel, (quality_x + quality_y) / 2.0


def displacement_px(series: DetectionSeries, start: float, end: float) -> float | None:
    """Straight-line centroid displacement between two timestamps."""
    if series.times.size < 2:
        return None
    coeff_x, _ = _robust_quadratic(series.times, series.cx)
    coeff_y, _ = _robust_quadratic(series.times, series.cy)
    delta_x = np.polyval(coeff_x, end) - np.polyval(coeff_x, start)
    delta_y = np.polyval(coeff_y, end) - np.polyval(coeff_y, start)
    return float(np.hypot(delta_x, delta_y))


class GroundingDinoBackend:
    """Detect a named object across a clip and turn it into pixel-space measurements."""

    def __init__(self, model_id: str = DEFAULT_MODEL, device: str | None = None,
                 box_threshold: float = 0.25, text_threshold: float = 0.20,
                 max_frames: int = 48, cache_path: str | Path | None = None,
                 batch_size: int = 8) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(self.device)
        self.model.eval()

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.max_frames = max_frames
        self.batch_size = batch_size

        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[tuple[str, str], DetectionSeries] = {}
        if self.cache_path and self.cache_path.exists():
            with self.cache_path.open("rb") as handle:
                self._cache = pickle.load(handle)

    # ---------------------------------------------------------------- frames

    def _read_frames(self, video_path: str) -> tuple[list, np.ndarray, int]:
        """Uniformly sample up to ``max_frames`` frames, returning them with their timestamps.

        The benchmark authors found that dropping frames hurts velocity and acceleration far more
        than dropping resolution, so sampling is uniform across the whole clip rather than
        truncated, preserving the full time span the fit is made over.
        """
        import cv2
        from PIL import Image

        capture = cv2.VideoCapture(str(video_path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frames, times = [], []
        wanted = (set(np.linspace(0, total - 1, min(self.max_frames, total)).astype(int))
                  if total > 0 else None)
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if wanted is None or index in wanted:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                times.append(index / fps)
            index += 1
        capture.release()
        return frames, np.asarray(times, dtype=float), index

    # ---------------------------------------------------------------- detection

    def _detect(self, video_path: str, phrase: str) -> DetectionSeries:
        key = (str(video_path), phrase.lower())
        if key in self._cache:
            return self._cache[key]

        frames, times, total = self._read_frames(video_path)
        rows: list[tuple[float, float, float, float, float, float]] = []

        if frames:
            # Grounding-DINO expects lowercase, period-terminated prompts.
            prompt = phrase.lower().strip().rstrip(".") + "."
            for start in range(0, len(frames), self.batch_size):
                chunk = frames[start:start + self.batch_size]
                chunk_times = times[start:start + self.batch_size]
                inputs = self.processor(images=chunk, text=[prompt] * len(chunk),
                                        return_tensors="pt", padding=True).to(self.device)
                with self.torch.inference_mode():
                    outputs = self.model(**inputs)
                results = self.processor.post_process_grounded_object_detection(
                    outputs, inputs.input_ids,
                    threshold=self.box_threshold, text_threshold=self.text_threshold,
                    target_sizes=[image.size[::-1] for image in chunk],
                )
                for moment, result in zip(chunk_times, results):
                    if len(result["scores"]) == 0:
                        continue
                    best = int(result["scores"].argmax())
                    x0, y0, x1, y1 = result["boxes"][best].tolist()
                    rows.append((moment, (x0 + x1) / 2, (y0 + y1) / 2,
                                 abs(x1 - x0), abs(y1 - y0),
                                 float(result["scores"][best])))

        array = np.asarray(rows, dtype=float).reshape(-1, 6)
        series = DetectionSeries(
            times=array[:, 0], cx=array[:, 1], cy=array[:, 2],
            width=array[:, 3], height=array[:, 4], scores=array[:, 5],
            frames_total=max(total, len(frames)),
            frames_sampled=len(frames),
        )
        self._cache[key] = series
        return series

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("wb") as handle:
            pickle.dump(self._cache, handle)

    # ---------------------------------------------------------------- protocol

    def measure(self, request: SolveRequest, video_path: str,
                object_name: str, dimension: str) -> PixelMeasurement:
        if not object_name:
            return PixelMeasurement(object_name="", note="no object phrase to ground")
        try:
            series = self._detect(video_path, object_name)
        except Exception as error:                                # noqa: BLE001
            return PixelMeasurement(object_name=object_name,
                                    note=f"detection failed: {type(error).__name__}: {error}")

        if series.times.size == 0:
            return PixelMeasurement(object_name=object_name, frames_tracked=0,
                                    note="object never detected")

        base_confidence = series.mean_score * series.detection_rate

        if dimension == "length":
            if request.interval is not None:
                value = displacement_px(series, *request.interval)
                confidence = base_confidence
            else:
                value = extent_for(series, request.measurement)
                confidence = base_confidence
            return PixelMeasurement(object_name=object_name, extent_px=value,
                                    confidence=confidence, frames_tracked=int(series.times.size))

        speed, accel, quality = kinematics(series, request.timestamp)
        # A kinematic answer is only as good as the trajectory fit, so fold it into confidence.
        confidence = base_confidence * quality
        if dimension == "speed":
            return PixelMeasurement(object_name=object_name, speed_px_per_s=speed,
                                    confidence=confidence, frames_tracked=int(series.times.size))
        return PixelMeasurement(object_name=object_name, accel_px_per_s2=accel,
                                confidence=confidence, frames_tracked=int(series.times.size))
