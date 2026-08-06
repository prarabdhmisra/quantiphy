"""Re-solve a finished vision run offline, from its cached detections. No GPU, no video.

The expensive part of a run is detection, and ``run_vision_job.py`` already pushes it to the Hub as
``<run>/detections.pkl`` -- 35 KB for the 20-row smoke run. Everything downstream of detection
(parsing, extent selection, kinematics, scale transfer) is deterministic CPU code, so any change to
it can be measured against detections we have already paid for, in about two seconds.

That is what found the 2.4x undershoot. The per-row decomposition below prints, beside each answer,
the ``prior_pixels`` value that *would* have produced the truth. When that column is a near-constant
multiple of what we measured, the error is in the prior's pixel measurement and nothing else --
which is a very different bug from a noisy detector, and is fixed in a different file.

Usage:
    py -3.12 scripts/replay_cache.py [--limit 20] [--run validation-grounding]

Detections are cached per ``(video path, phrase)``. **Changing an object phrase changes the key**,
so a parsing fix that renames phrases shows up here as cache misses, not as new numbers. That is the
correct signal: those rows need a fresh (cheap) detection pass before they can be measured again.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy.backends.grounding import displacement_px, extent_for, kinematics  # noqa: E402
from quantiphy.parsing import build_request  # noqa: E402
from quantiphy.solver import solve_row  # noqa: E402
from quantiphy.vision import PixelMeasurement  # noqa: E402

DEFAULT_REPO = "prarabdhmisra/quantiphy-runs"
VALIDATION = "PaulineLi/QuantiPhy-validation"


class CachedBackend:
    """A :class:`~quantiphy.vision.VisionBackend` fed from a pickled detection cache.

    Deliberately reuses ``extent_for``/``kinematics``/``displacement_px`` from the real backend
    rather than reimplementing them -- those are pure numpy and import without torch, so a replay
    exercises the same measurement code the GPU run did. Only the frames come from a different
    place.
    """

    def __init__(self, cache: dict) -> None:
        # The cache is keyed by the job's absolute path (/root/.cache/...), which never exists
        # locally, so re-key by basename.
        self.series = {(os.path.basename(path), phrase): value
                       for (path, phrase), value in cache.items()}
        self.misses: list[tuple[str, str]] = []

    def measure(self, request, video_path: str, object_name: str,
                dimension: str) -> PixelMeasurement:
        if not object_name:
            return PixelMeasurement(object_name="", note="no object phrase to ground")
        key = (os.path.basename(video_path), object_name.lower())
        series = self.series.get(key)
        if series is None:
            self.misses.append(key)
            return PixelMeasurement(object_name=object_name, note="not in the detection cache")

        confidence = series.mean_score * series.detection_rate
        tracked = int(series.times.size)
        if dimension == "length":
            extent = (displacement_px(series, *request.interval) if request.interval is not None
                      else extent_for(series, request.measurement))
            return PixelMeasurement(object_name, extent_px=extent, confidence=confidence,
                                    frames_tracked=tracked)

        speed, accel, quality = kinematics(series, request.timestamp)
        return PixelMeasurement(
            object_name,
            speed_px_per_s=speed if dimension == "speed" else None,
            accel_px_per_s2=accel if dimension == "acceleration" else None,
            confidence=confidence * quality, frames_tracked=tracked,
        )


def load(repo: str, run: str, limit: int) -> tuple[pd.DataFrame, dict]:
    from huggingface_hub import hf_hub_download

    csv = hf_hub_download(VALIDATION, repo_type="dataset", filename="validation_dataset.csv")
    frame = pd.read_csv(csv, encoding="utf-8-sig")
    frame = frame[frame["ground_truth_posterior"].notna()].reset_index(drop=True).head(limit)

    pkl = hf_hub_download(repo, repo_type="dataset", filename=f"{run}/detections.pkl")
    with open(pkl, "rb") as handle:
        return frame, pickle.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=20, help="validation rows to replay")
    parser.add_argument("--run", default="validation-grounding", help="run folder in the repo")
    parser.add_argument("--repo", default=os.environ.get("OUTPUT_REPO", DEFAULT_REPO))
    args = parser.parse_args()

    frame, cache = load(args.repo, args.run, args.limit)
    backend = CachedBackend(cache)
    print(f"{len(cache)} cached (video, phrase) pairs; replaying {len(frame)} rows\n")

    header = (f"{'#':>2} {'ratio':>7} {'pred':>10} {'truth':>10} {'prior_px':>9} "
              f"{'needed':>9} {'x':>6} {'target_px':>10}  prior -> target")
    print(header)
    print("-" * len(header))

    ratios: list[float] = []
    for index, row in frame.iterrows():
        request = build_request(row)
        answer = solve_row(request, backend, f"cache/{row['video_id']}.mp4")
        truth = float(row["ground_truth_posterior"])
        prior = request.scale_prior

        if not answer.solved:
            print(f"{index:2d} {'-':>7} {'-':>10} {truth:10.4g} {'':9} {'':9} {'':6} {'':10}  "
                  f"UNSOLVED: {answer.reason}")
            continue

        ratio = answer.value / truth if truth else math.nan
        needed = answer.prior_pixels * ratio          # prior_px that would have hit the truth
        ratios.append(ratio)
        print(f"{index:2d} {ratio:7.3f} {answer.value:10.4g} {truth:10.4g} "
              f"{answer.prior_pixels:9.1f} {needed:9.1f} {ratio:6.2f} {answer.target_pixels:10.1f}  "
              f"{prior.object_name!r} ({prior.quantity}) -> "
              f"{request.target_object!r} ({request.measurement})")

    print(f"\nsolved {len(ratios)}/{len(frame)}")
    if ratios:
        array = np.asarray(ratios)
        print(f"median pred/truth      {np.median(array):.3f}      (1.0 is unbiased)")
        print(f"geometric mean         {math.exp(np.mean(np.log(array))):.3f}")
        print(f"within 2x either way   {int(np.sum((array > 0.5) & (array < 2.0)))}/{len(array)}")
        print(f"overshoots             {int(np.sum(array > 1.0))}/{len(array)}"
              f"        (overshoot is fatal under MRA, undershoot is cheap)")
        print(f"if the whole bias sat in the prior, prior_pixels would be "
              f"{1 / np.median(array):.2f}x too large")

    if backend.misses:
        unique = sorted(set(backend.misses))
        print(f"\n{len(backend.misses)} cache misses over {len(unique)} distinct keys - these rows "
              f"were NOT measured:")
        for video, phrase in unique:
            print(f"    {video}  {phrase!r}")
        print("Expected after a change to object phrases: the key moved, so these need a fresh\n"
              "detection pass before they can be scored again.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
