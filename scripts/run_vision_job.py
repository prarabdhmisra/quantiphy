#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=4.41",
#   "accelerate",
#   "opencv-python-headless",
#   "pillow",
#   "pandas>=2.0",
#   "numpy>=1.24",
#   "pyarrow",
#   "huggingface-hub>=0.25",
#   "tqdm",
# ]
# ///
"""Run the geometric solver over a QuantiPhy split on Hugging Face Jobs.

Why a Job rather than a notebook: the full test split is 3,289 rows over 568 videos, which is
hours of detection. A Job runs detached, so a closed laptop or a dropped tab doesn't kill it.

    hf jobs uv run --flavor l4x1 --timeout 6h --secrets HF_TOKEN \
      --env QUANTIPHY_GIT=git+https://github.com/<you>/quantiphy.git \
      --env OUTPUT_REPO=<you>/quantiphy-runs \
      --env SPLIT=validation \
      scripts/run_vision_job.py

Designed around the two things that actually go wrong on long jobs:

* **It checkpoints and resumes.** Partial predictions and the detection cache are pushed to a Hub
  dataset repo every ``CHECKPOINT_EVERY`` rows. A job that dies at row 3,000 restarts from 3,000,
  not from zero. Since detections are cached per (video, phrase) and 3,289 questions share only
  568 videos, a warm cache makes a re-run minutes rather than hours.
* **It never emits a blank.** Unsolved rows are written with an empty ``parsed_value`` and a
  reason, so the fallback arm can fill them later. Submitting a blank scores a hard zero, so the
  CSV this produces is an intermediate artefact, not a submission -- run it through
  ``scripts/validate_submission.py`` after fallbacks are applied.

Environment:
    QUANTIPHY_GIT     pip-installable source for the solver package (or QUANTIPHY_HF_REPO)
    QUANTIPHY_HF_REPO dataset repo holding the package source, if not using git
    OUTPUT_REPO       dataset repo to push predictions and the detection cache to (required)
    SPLIT             "validation" (159 rows, has ground truth) or "test" (3,289 rows)
    LIMIT             optional row cap, for a smoke test
    BOX_THRESHOLD     detection threshold, default 0.25
    MAX_FRAMES        frames sampled per clip, default 48
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT_EVERY = 100
WORK = Path("/tmp/quantiphy")
WORK.mkdir(parents=True, exist_ok=True)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def install_solver() -> None:
    """Make the ``quantiphy`` package importable inside the job."""
    git_source = os.environ.get("QUANTIPHY_GIT")
    hub_source = os.environ.get("QUANTIPHY_HF_REPO")

    if git_source:
        log(f"installing solver from {git_source}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", git_source])
        return
    if hub_source:
        from huggingface_hub import snapshot_download
        log(f"fetching solver source from hub repo {hub_source}")
        sys.path.insert(0, snapshot_download(repo_id=hub_source, repo_type="dataset"))
        return
    raise SystemExit("set QUANTIPHY_GIT or QUANTIPHY_HF_REPO so the solver can be imported")


def load_split(split: str):
    """Download the split's metadata and videos, returning (frame, video_id -> path)."""
    import pandas as pd
    from huggingface_hub import snapshot_download

    if split == "validation":
        root = Path(snapshot_download("PaulineLi/QuantiPhy-validation", repo_type="dataset"))
        frame = pd.read_csv(root / "validation_dataset.csv", encoding="utf-8-sig")
        frame = frame[frame["ground_truth_posterior"].notna()].reset_index(drop=True)
        video_dir = root / "validation_videos"
    else:
        root = Path(snapshot_download("PaulineLi/QuantiPhy", repo_type="dataset"))
        frame = pd.read_parquet(root / "test_dataset.parquet").reset_index(drop=True)
        video_dir = root

    # Filenames are not reliably clean: at least one validation clip has a leading space, and the
    # id column omits the extension. Index by normalised name and look up leniently.
    index = {path.name.strip().lower(): path for path in video_dir.rglob("*.mp4")}

    def find(video_id) -> Path | None:
        stem = str(video_id).strip().lower()
        return index.get(f"{stem}.mp4") or index.get(stem)

    frame["video_path"] = frame["video_id"].map(find)
    missing = frame["video_path"].isna().sum()
    log(f"{split}: {len(frame)} rows, {len(index)} videos, {missing} rows without a video file")
    return frame


def load_checkpoint(output_repo: str, name: str):
    """Restore partial predictions and the detection cache from a previous run, if any."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    done, cache_path = {}, WORK / "detections.pkl"
    for filename, target in ((f"{name}/partial.csv", None), (f"{name}/detections.pkl", cache_path)):
        try:
            path = hf_hub_download(repo_id=output_repo, repo_type="dataset", filename=filename)
        except Exception:                                          # noqa: BLE001
            continue
        if target is None:
            partial = pd.read_csv(path)
            done = dict(zip(partial["row_index"], zip(partial["parsed_value"],
                                                      partial["method"], partial["reason"])))
            log(f"resuming: {len(done)} rows already solved")
        else:
            target.write_bytes(Path(path).read_bytes())
            log("restored detection cache")
    return done, cache_path


def push(output_repo: str, name: str, frame, cache_path: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    partial = WORK / "partial.csv"
    frame.to_csv(partial, index=False)
    api.upload_file(path_or_fileobj=partial, path_in_repo=f"{name}/partial.csv",
                    repo_id=output_repo, repo_type="dataset")
    if cache_path.exists():
        api.upload_file(path_or_fileobj=cache_path, path_in_repo=f"{name}/detections.pkl",
                        repo_id=output_repo, repo_type="dataset")


def main() -> int:
    split = os.environ.get("SPLIT", "validation")
    output_repo = os.environ.get("OUTPUT_REPO")
    if not output_repo:
        raise SystemExit("OUTPUT_REPO is required so results survive the job")

    install_solver()

    import pandas as pd
    from huggingface_hub import HfApi
    from tqdm import tqdm

    from quantiphy.backends.grounding import GroundingDinoBackend
    from quantiphy.parsing import build_request
    from quantiphy.solver import solve_row

    HfApi().create_repo(output_repo, repo_type="dataset", exist_ok=True, private=True)

    frame = load_split(split)
    limit = os.environ.get("LIMIT")
    if limit:
        frame = frame.head(int(limit)).copy()
        log(f"LIMIT set: solving only {len(frame)} rows")

    name = f"{split}-grounding"
    done, cache_path = load_checkpoint(output_repo, name)

    backend = GroundingDinoBackend(
        box_threshold=float(os.environ.get("BOX_THRESHOLD", 0.25)),
        max_frames=int(os.environ.get("MAX_FRAMES", 48)),
        cache_path=cache_path,
    )
    log(f"backend ready on {backend.device}")

    records = []
    for position, (row_index, row) in enumerate(tqdm(frame.iterrows(), total=len(frame))):
        if row_index in done:
            value, method, reason = done[row_index]
        elif row["video_path"] is None:
            value, method, reason = None, "none", "no video file"
        else:
            try:
                request = build_request(row)
                answer = solve_row(request, backend, str(row["video_path"]))
                value = answer.value if answer.solved else None
                method, reason = answer.method, answer.reason
            except Exception as error:                             # noqa: BLE001
                # One bad row must not lose hours of work.
                value, method, reason = None, "none", f"{type(error).__name__}: {error}"

        records.append({
            "row_index": row_index,
            "video_id": row["video_id"],
            "question": row["question"],
            "video_type": row["video_type"],
            "inference_type": row["inference_type"],
            "parsed_value": value,
            "method": method,
            "reason": reason,
        })

        if (position + 1) % CHECKPOINT_EVERY == 0:
            backend.save_cache()
            push(output_repo, name, pd.DataFrame(records), cache_path)
            solved = sum(1 for r in records if r["parsed_value"] is not None)
            log(f"checkpoint {position + 1}/{len(frame)} — solved {solved}")

    backend.save_cache()
    results = pd.DataFrame(records)
    push(output_repo, name, results, cache_path)

    HfApi().upload_file(
        path_or_fileobj=results.to_csv(index=False).encode(),
        path_in_repo=f"{name}/predictions.csv",
        repo_id=output_repo, repo_type="dataset",
    )

    solved = results["parsed_value"].notna().sum()
    log(f"done: solved {solved}/{len(results)} ({100 * solved / len(results):.1f}%)")
    log("top failure reasons:")
    for reason, count in results.loc[results["parsed_value"].isna(), "reason"] \
            .value_counts().head(8).items():
        log(f"    {count:5d}  {reason}")

    # Scoring only means something where ground truth exists.
    if split == "validation":
        from quantiphy.scoring import score
        merged = frame.reset_index(drop=True).copy()
        merged["parsed_value"] = results["parsed_value"].to_numpy()
        try:
            log(f"coverage-limited: {score(merged[merged.parsed_value.notna()])}")
        except ValueError as error:
            log(f"coverage-limited: not scorable — {error}")
        merged["parsed_value"] = merged["parsed_value"].fillna(0.0)
        log(f"as-submitted (unsolved=0): {score(merged)}")
        log("GPT-5.1 reference on this split: macro 0.4856")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
