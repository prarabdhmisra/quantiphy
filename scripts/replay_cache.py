"""Re-solve a finished vision run offline, from its cached detections. No GPU, no video.

The expensive part of a run is detection, and ``run_vision_job.py`` already pushes it to the Hub as
``<run>/detections.pkl`` -- 35 KB for the 20-row smoke run. Everything downstream of detection
(parsing, extent selection, kinematics, scale transfer) is deterministic CPU code, so any change to
it can be measured against detections we have already paid for, in about two seconds.

That is what found the 2.4x undershoot. The per-row decomposition below prints, beside each answer,
the ``prior_pixels`` value that *would* have produced the truth. When that column is a near-constant
multiple of what we measured, the error is in the prior's pixel measurement and nothing else --
which is a very different bug from a noisy detector, and is fixed in a different file.

Two splits, and they answer different questions. Validation has ground truth, so it prints the
per-row ratio table above. The test split has none, so it prints solve rates instead -- and the
solve rate is a real instrument, because a solved row was measured at **+0.218 in D2** against the
fallback constant. Judge a change on the test split by whether it solves strictly more rows, then
buy one submission slot to find out what they were worth.

A test replay always checks itself against ``SOLVER_V1`` first. That gate is the harness's only
claim to authority -- same code, same detections, so a disagreement means the harness is broken, not
the solver -- and it exits non-zero rather than printing solve rates that could justify a slot.

Usage:
    py -3.12 scripts/replay_cache.py [--limit 20] [--run validation-grounding]
    py -3.12 scripts/replay_cache.py --split test --run test-solver-v1 --shards 4 \
        [--trusted-prior-pixels 30,inf] [--out predictions.csv]

Detections are cached per ``(video path, phrase)``. **Changing an object phrase changes the key**,
so a parsing fix that renames phrases shows up here as cache misses, not as new numbers. That is the
correct signal: those rows need a fresh (cheap) detection pass before they can be measured again.
An *empty* cached series is not a miss -- the detector ran and found nothing, which is an answer.
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
from quantiphy.backends.grounding import (  # noqa: E402
    centroid_at,
    displacement_px,
    extent_for,
    kinematics,
)
from quantiphy.parsing import build_request  # noqa: E402
from quantiphy.solver import TRUSTED_PRIOR_PIXELS, solve_row  # noqa: E402
from quantiphy.vision import PixelMeasurement  # noqa: E402

DEFAULT_REPO = "prarabdhmisra/quantiphy-runs"
VALIDATION = "PaulineLi/QuantiPhy-validation"
TEST_PARQUET = ROOT / "data" / "fixtures" / "test_dataset.parquet"

#: What ``test-solver-v1`` measured on the GPU, read off its own ``predictions.csv``. A test replay
#: runs the identical measurement code against the identical detections, so it must reproduce these
#: exactly before any change to the solver is believed. A mismatch means the harness is wrong, not
#: the solver -- so this is asserted loudly rather than printed for eyeballing.
#:
#: ``band`` is the ``TRUSTED_PRIOR_PIXELS`` the run itself used, and the gate always replays at it
#: explicitly rather than at whatever the solver's default has since become. The default has already
#: moved once -- to ``(30, inf)``, which is what this run's own numbers argued for -- and a gate that
#: read the live default would have started failing at exactly the moment its finding was adopted.
SOLVER_V1 = {"solved": 1338, "S2": 0.42513, "D2": 0.28448, "S3": 0.76562, "D3": 0.32922,
             "band": (30.0, 300.0)}


class CachedBackend:
    """A :class:`~quantiphy.vision.VisionBackend` fed from a pickled detection cache.

    Deliberately reuses ``extent_for``/``kinematics``/``displacement_px``/``centroid_at`` from the
    real backend rather than reimplementing them -- those are pure numpy and import without torch,
    so a replay exercises the same measurement code the GPU run did. Only the frames come from a
    different place.
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

        if series.times.size == 0:
            # An empty series is a *cached* answer -- the detector ran and found nothing -- so it is
            # not a miss. Mirroring the real backend's note matters because the decline-reason
            # histogram is what the next fix is chosen from, and without this the 41 "object never
            # detected" rows arrive as an empty reason and vanish into the tail.
            return PixelMeasurement(object_name=object_name, frames_tracked=0,
                                    note="object never detected")

        confidence = series.mean_score * series.detection_rate
        tracked = int(series.times.size)
        if dimension == "length":
            extent = (displacement_px(series, *request.interval) if request.interval is not None
                      else extent_for(series, request.measurement))
            return PixelMeasurement(object_name, extent_px=extent,
                                    centroid_px=centroid_at(series, request.timestamp),
                                    confidence=confidence, frames_tracked=tracked)

        speed, accel, quality = kinematics(series, request.timestamp)
        return PixelMeasurement(
            object_name,
            speed_px_per_s=speed if dimension == "speed" else None,
            accel_px_per_s2=accel if dimension == "acceleration" else None,
            confidence=confidence * quality, frames_tracked=tracked,
        )


def load_detections(repo: str, run: str, shards: int) -> dict:
    """The run's detection cache, unioned across its shards.

    A sharded run partitions *rows*, so each shard's cache holds only the ``(video, phrase)`` pairs
    its own slice needed. The keys are the pair itself, so the dicts union cleanly -- and where two
    shards saw the same video the entries are identical measurements of identical frames, so which
    one wins does not matter. ``shards=0`` means an unsharded run with one cache at
    ``<run>/detections.pkl``.
    """
    from huggingface_hub import hf_hub_download

    names = [run] if shards == 0 else [f"{run}-shard{index}" for index in range(1, shards + 1)]
    cache: dict = {}
    for name in names:
        path = hf_hub_download(repo, repo_type="dataset", filename=f"{name}/detections.pkl")
        with open(path, "rb") as handle:
            piece = pickle.load(handle)
        overlap = len(set(piece) & set(cache))
        print(f"  {name}: {len(piece)} pairs" + (f" ({overlap} already seen)" if overlap else ""))
        cache.update(piece)
    return cache


def load(repo: str, run: str, limit: int, split: str = "validation",
         shards: int = 0) -> tuple[pd.DataFrame, dict]:
    """The rows to replay, and the detections to replay them against.

    The test split is read from the *local* pinned parquet rather than through
    ``snapshot_download("PaulineLi/QuantiPhy")``, which would pull every video -- gigabytes -- to
    answer a question that needs no pixels at all. No video is opened here; only their basenames,
    as cache keys.
    """
    if split == "test":
        if not TEST_PARQUET.exists():
            raise SystemExit(f"{TEST_PARQUET} is missing; the test split cannot be replayed")
        frame = pd.read_parquet(TEST_PARQUET).reset_index(drop=True)
        if limit:
            frame = frame.head(limit)
        return frame, load_detections(repo, run, shards)

    from huggingface_hub import hf_hub_download

    csv = hf_hub_download(VALIDATION, repo_type="dataset", filename="validation_dataset.csv")
    frame = pd.read_csv(csv, encoding="utf-8-sig")
    frame = frame[frame["ground_truth_posterior"].notna()].reset_index(drop=True).head(limit)
    return frame, load_detections(repo, run, shards)


def replay(frame: pd.DataFrame, backend: CachedBackend, **kwargs) -> pd.DataFrame:
    """Re-solve every row, one record each, in ``run_vision_job.py``'s prediction schema.

    A row that raises is recorded as declined with the exception as its reason, exactly as the GPU
    job does. That matters more here than there: the point of a replay is a *complete* comparison
    against the run it reproduces, and one raised row aborting the pass looks like a harness bug
    when it is a solver bug on a single input.
    """
    records = []
    for row_index, row in frame.iterrows():
        try:
            request = build_request(row)
            # Only the basename is used, as a cache key -- no file is opened.
            answer = solve_row(request, backend, f"cache/{row['video_id']}.mp4", **kwargs)
            value = answer.value if answer.solved else None
            method, reason = answer.method, answer.reason
            confidence = answer.confidence
            prior_px, target_px = answer.prior_pixels, answer.target_pixels
        except Exception as error:                                     # noqa: BLE001
            value, method, reason = None, "none", f"{type(error).__name__}: {error}"
            confidence = prior_px = target_px = float("nan")

        records.append({
            "row_index": row_index,
            "video_id": row["video_id"],
            "question": row["question"],
            "video_type": row["video_type"],
            "inference_type": row["inference_type"],
            "parsed_value": value,
            "method": method,
            "reason": reason,
            "confidence": confidence,
            # Not in the GPU schema, and the whole reason a replay beats reading predictions.csv:
            # these are what the TRUSTED_PRIOR_PIXELS band gets re-fitted against.
            "prior_pixels": prior_px,
            "target_pixels": target_px,
        })
    return pd.DataFrame(records)


def report(results: pd.DataFrame) -> None:
    """Solve rate overall and per scored category, then the decline reasons that dominate.

    There is no ground truth for the test split, so a replay cannot be scored -- the solve rate and
    the *identity* of newly solved rows is the entire signal. Judge a change by whether it solves
    strictly more rows, then spend one submission slot to find out what they were worth.
    """
    from quantiphy.scoring import CATEGORIES, category_labels

    solved = results["parsed_value"].notna()
    print(f"\nsolved {int(solved.sum())}/{len(results)} ({solved.mean():.1%})")

    labels = category_labels(results)
    print(f"\n{'cat':>4} {'rows':>6} {'solved':>7} {'rate':>9}  {'solver-v1':>9} {'delta':>8}")
    for category in CATEGORIES:
        rows = labels == category
        if not rows.any():
            continue
        rate = solved[rows].mean()
        was = SOLVER_V1[category]
        print(f"{category:>4} {int(rows.sum()):6d} {int(solved[rows].sum()):7d} {rate:9.3%}  "
              f"{was:9.3%} {rate - was:+8.3%}")

    declines = results.loc[~solved, "reason"].value_counts()
    print(f"\ntop decline reasons ({len(declines)} distinct over {int((~solved).sum())} rows):")
    for reason, count in declines.head(10).items():
        print(f"  {count:5d}  {reason}")


def check_reproduction(results: pd.DataFrame) -> bool:
    """Did the replay reproduce ``solver-v1``? Print the verdict, and return it.

    Deliberately a hard gate rather than a note. The replay's only claim to authority is that it is
    the same code on the same detections, so if it disagrees with the run it reproduces then every
    number it prints is unfounded -- including the ones that would justify spending a slot.
    """
    from quantiphy.scoring import CATEGORIES, category_labels

    solved = results["parsed_value"].notna()
    labels = category_labels(results)
    problems = []
    if int(solved.sum()) != SOLVER_V1["solved"]:
        problems.append(f"solved {int(solved.sum())}, expected {SOLVER_V1['solved']}")
    for category in CATEGORIES:
        rows = labels == category
        # 0.0005 is half of the last recorded digit, not an allowance for drift: the replay is
        # deterministic, so anything past rounding is a real difference and must fail.
        if rows.any() and abs(solved[rows].mean() - SOLVER_V1[category]) > 0.0005:
            problems.append(f"{category} {solved[rows].mean():.3%} vs {SOLVER_V1[category]:.3%}")

    if problems:
        print("\nDOES NOT REPRODUCE solver-v1: " + "; ".join(problems))
        print("The harness is wrong, not the solver -- fix that before believing anything above.")
        return False
    print(f"\nreproduces solver-v1 exactly ({SOLVER_V1['solved']} solved, all four rates match)")
    return True


def band(text: str) -> tuple[float, float]:
    """Parse ``--trusted-prior-pixels lo,hi``; ``inf`` is accepted, and so is ``0,inf``."""
    try:
        low, high = (float(part) for part in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected 'low,high', got {text!r}") from None
    if not low <= high:
        raise argparse.ArgumentTypeError(f"low must not exceed high, got {low} > {high}")
    return low, high


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="rows to replay; 0 means all. Defaults to 20 on validation (the smoke "
                             "run's size) and to all 3,289 on test")
    parser.add_argument("--run", default="validation-grounding", help="run folder in the repo")
    parser.add_argument("--repo", default=os.environ.get("OUTPUT_REPO", DEFAULT_REPO))
    parser.add_argument("--split", choices=("validation", "test"), default="validation",
                        help="validation has ground truth and prints per-row ratios; test has "
                             "none, so it prints solve rates and decline reasons instead")
    parser.add_argument("--shards", type=int, default=0,
                        help="shard count of a sharded run; 0 for a single detections.pkl")
    parser.add_argument("--trusted-prior-pixels", type=band, default=None, metavar="LOW,HIGH",
                        help="override solver.TRUSTED_PRIOR_PIXELS for this replay -- this is how "
                             "the band is re-fitted without editing the solver")
    parser.add_argument("--out", type=Path, default=None,
                        help="write the replayed predictions here, in run_vision_job.py's schema, "
                             "so make_submission.py can consume them directly")
    args = parser.parse_args()

    # A test replay defaults to every row: an arbitrary first-20 slice says nothing about a solve
    # rate -- and it silently *looks* like an answer, which is worse than a slow pass. All 3,289
    # cost seconds.
    limit = args.limit if args.limit is not None else (20 if args.split == "validation" else 0)
    kwargs = ({"trusted_prior_pixels": args.trusted_prior_pixels}
              if args.trusted_prior_pixels is not None else {})

    frame, cache = load(args.repo, args.run, limit, args.split, args.shards)
    backend = CachedBackend(cache)
    print(f"{len(cache)} cached (video, phrase) pairs; replaying {len(frame)} rows\n")

    if args.split == "test":
        # The gate is a statement about the harness, not about the current configuration, so it
        # replays at the band `solver-v1` itself ran with -- never at the live default, which has
        # already moved to (30, inf). Skipped only under --limit, where a partial pass cannot
        # reproduce a whole-run solve rate.
        requested = kwargs.get("trusted_prior_pixels", TRUSTED_PRIOR_PIXELS)
        gate = None
        if limit:
            print(f"(--limit {limit}: reproduction gate skipped, it needs all 3,289 rows)\n")
            results = replay(frame, backend, **kwargs)
        else:
            gate = replay(frame, backend, trusted_prior_pixels=SOLVER_V1["band"])
            results = (gate if requested == SOLVER_V1["band"]
                       else replay(frame, backend, **kwargs))

        report(results)
        if gate is not None and results is not gate:
            low, high = requested
            print(f"\n(reported at band ({low:g}, {high:g}); the gate below is a separate replay "
                  f"at solver-v1's own ({SOLVER_V1['band'][0]:g}, {SOLVER_V1['band'][1]:g}))")
        reproduced = check_reproduction(gate) if gate is not None else None

        if backend.misses:
            unique = sorted(set(backend.misses))
            print(f"\n{len(backend.misses)} cache misses over {len(unique)} distinct keys -- "
                  f"those rows were NOT measured. A renamed phrase lands here, not in the numbers "
                  f"above, and needs a fresh detection pass.")
            for video, phrase in unique[:10]:
                print(f"    {video}  {phrase!r}")
            if len(unique) > 10:
                print(f"    ... and {len(unique) - 10} more")

        if args.out:
            results.to_csv(args.out, index=False, encoding="utf-8")
            print(f"\n-> {args.out}")
            print(f"now: py -3.12 scripts/make_submission.py {args.out} "
                  f"--out solver-v2.submission.csv --fallback-from baseline_predictions.csv")
        # A replay that does not reproduce its own run is a broken harness, and exiting 0 would let
        # it feed a submission.
        return 0 if reproduced is not False else 1

    header = (f"{'#':>2} {'ratio':>7} {'pred':>10} {'truth':>10} {'prior_px':>9} "
              f"{'needed':>9} {'x':>6} {'target_px':>10}  prior -> target")
    print(header)
    print("-" * len(header))

    ratios: list[float] = []
    records: list[dict] = []
    for index, row in frame.iterrows():
        request = build_request(row)
        answer = solve_row(request, backend, f"cache/{row['video_id']}.mp4")
        truth = float(row["ground_truth_posterior"])
        prior = request.scale_prior

        # Same schema the test branch writes, so a validation replay feeds the same instruments --
        # and this is the *only* replay with ground truth, which makes it the only one that can
        # settle a question like "does fusing two arms beat picking one" without buying a slot.
        records.append({
            "row_index": index, "video_id": row["video_id"], "question": row["question"],
            "video_type": row["video_type"], "inference_type": row["inference_type"],
            "parsed_value": answer.value if answer.solved else None,
            "method": answer.method, "reason": answer.reason,
            "confidence": answer.confidence,
            "prior_pixels": answer.prior_pixels, "target_pixels": answer.target_pixels,
            "ground_truth_posterior": truth,
        })

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

    if args.out:
        pd.DataFrame(records).to_csv(args.out, index=False, encoding="utf-8")
        print(f"\n-> {args.out}  (carries ground_truth_posterior, unlike a test replay)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
