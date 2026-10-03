"""Are the solver's pixel tracks the reason it overshoots? Free, offline, from cached detections.

The solver overshoots velocity ~2x and acceleration ~3.5x (median, against the constant), though
``kinematics()`` already fits a robust quadratic rather than differencing. The remaining suspect is
the track itself: Grounding-DINO runs per frame and keeps the top box, so in a multi-object scene it
can hop between objects (an identity swap), and its box can breathe frame to frame. A mask tracker
(SAM 2) fixes exactly those two failures and nothing else, so it is worth buying only if they are
where the error lives.

The test split has no ground truth, so the VLM stands in for it: it is the stronger arm in D2/D3
(+0.163 and +0.137 per row over the solver), and on rows where both arms answer, a large
``log(solver / vlm)`` marks a likely solver error. A proxy, not truth -- it is read only as a
*contrast* between clean and jumpy tracks, never as an absolute error.

Usage:
    py -3.12 scripts/audit_tracks.py [--replay replay-full.csv] [--run test-solver-v1 --shards 4]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from quantiphy.backends.grounding import DetectionSeries  # noqa: E402
from quantiphy.parsing import build_request  # noqa: E402
from quantiphy.scoring import category_labels  # noqa: E402

#: A step longer than this many median steps is a jump. Generous on purpose: real accelerating
#: motion changes step length smoothly and by far less than 5x between adjacent samples.
JUMP_FACTOR = 5.0
#: ...and also longer than this share of the box diagonal, so a near-static object's sub-pixel
#: jitter (median step ~0) cannot turn every wobble into a "jump".
JUMP_MIN_DIAGONAL_SHARE = 0.5
MIN_SAMPLES = 4
#: Above this a track is "jumpy". One bad step in ten is enough to bend a quadratic fit.
JUMPY_RATE = 0.10
OVERSHOOT_RATIO = 1.9


def track_quality(series: DetectionSeries) -> dict[str, float]:
    """Jump rate (share of steps that leap) and box-area coefficient of variation."""
    nan = {"jump_rate": math.nan, "area_cv": math.nan, "samples": float(series.times.size)}
    if series.times.size < MIN_SAMPLES:
        return nan
    steps = np.hypot(np.diff(series.cx), np.diff(series.cy))
    diagonal = float(np.median(np.hypot(series.width, series.height)))
    threshold = max(JUMP_FACTOR * float(np.median(steps)), JUMP_MIN_DIAGONAL_SHARE * diagonal)
    area = series.width * series.height
    area_cv = float(area.std() / area.mean()) if area.mean() > 0 else math.nan
    return {"jump_rate": float((steps > threshold).mean()), "area_cv": area_cv,
            "samples": float(series.times.size)}


def vlm_reference(v1: Path, v2: Path) -> pd.Series:
    """The VLM's usable answer per row_index: the original arm, else the strict re-run."""
    def sentinel(path: Path) -> pd.Series:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame = frame[frame["method"] == "vlm-sentinel"]
        return pd.to_numeric(frame.set_index("row_index")["parsed_value"], errors="coerce")
    first, second = sentinel(v1), sentinel(v2)
    return first.combine_first(second)


def row_tracks(frame: pd.DataFrame, series: dict) -> pd.DataFrame:
    """Track quality of each row's target object and scale-prior object."""
    records = []
    for row_index, row in frame.iterrows():
        try:
            request = build_request(row)
        except Exception:                                              # noqa: BLE001
            continue
        prior = request.scale_prior
        record = {"row_index": row_index, "dimension": request.dimension}
        for role, name in (("target", request.target_object),
                           ("prior", prior.object_name if prior else None)):
            found = series.get((f"{row['video_id']}.mp4", (name or "").lower()))
            quality = (track_quality(found) if found is not None and found.times.size
                       else {"jump_rate": math.nan, "area_cv": math.nan, "samples": 0.0})
            record.update({f"{role}_{key}": value for key, value in quality.items()})
        records.append(record)
    return pd.DataFrame(records).set_index("row_index")


def summarise(table: pd.DataFrame) -> pd.DataFrame:
    """Per (dimension, track class): rows, solver/VLM disagreement, and overshoot share."""
    both = table.dropna(subset=["log_ratio"])
    worst = both[["target_jump_rate", "prior_jump_rate"]].max(axis=1)
    both = both.assign(track=np.where(worst.isna(), "unknown",
                                      np.where(worst > JUMPY_RATE, "jumpy", "clean")))
    return both.groupby(["dimension", "track"]).agg(
        rows=("log_ratio", "size"),
        median_ratio=("log_ratio", lambda s: float(np.exp(s.median()))),
        median_abs_log=("log_ratio", lambda s: float(s.abs().median())),
        overshoot_share=("log_ratio", lambda s: float((s > math.log(OVERSHOOT_RATIO)).mean())),
        target_area_cv=("target_area_cv", "median"),
    ).round(3)


def main() -> int:
    from replay_cache import DEFAULT_REPO, TEST_PARQUET, load_detections

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, default=ROOT / "replay-full.csv")
    parser.add_argument("--run", default="test-solver-v1")
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--out", type=Path, default=None, help="per-row table, for follow-up")
    args = parser.parse_args()

    frame = pd.read_parquet(TEST_PARQUET).reset_index(drop=True)
    cache = load_detections(args.repo, args.run, args.shards)
    series = {(os.path.basename(path), phrase): value for (path, phrase), value in cache.items()}

    table = row_tracks(frame, series)
    solver = pd.read_csv(args.replay).set_index("row_index")["parsed_value"]
    vlm = vlm_reference(ROOT / "vlm-v1.predictions.csv", ROOT / "vlm-v2.predictions.csv")
    table["category"] = category_labels(frame).reindex(table.index)
    table["solver"], table["vlm"] = solver, vlm
    valid = (table["solver"] > 0) & (table["vlm"] > 0)
    table["log_ratio"] = np.where(valid, np.log(table["solver"] / table["vlm"]), np.nan)

    tracked = table.dropna(subset=["target_jump_rate"])
    print(f"{len(table)} rows parsed | {len(tracked)} with a cached target track | "
          f"{int(table['log_ratio'].notna().sum())} answered by both arms")
    jumpy = (tracked[["target_jump_rate", "prior_jump_rate"]].max(axis=1) > JUMPY_RATE)
    print(f"jumpy tracks (> {JUMPY_RATE:.0%} of steps leap): {int(jumpy.sum())} of {len(tracked)} "
          f"({jumpy.mean():.1%})\n")
    print("solver vs VLM, by dimension and track class (ratio = solver / vlm):")
    print(summarise(table).to_string())
    print("\njumpy share by category x dimension:")
    print(tracked.assign(jumpy=jumpy).groupby(["category", "dimension"])["jumpy"]
          .agg(["size", "mean"]).round(3).to_string())
    if args.out:
        table.to_csv(args.out)
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
