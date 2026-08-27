#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   "transformers>=4.45",
#   "accelerate",
#   "bitsandbytes",
#   "opencv-python-headless",
#   "pillow",
#   "pandas>=2.0",
#   "numpy>=1.24",
#   "pyarrow",
#   "huggingface-hub>=0.25",
#   "tqdm",
# ]
# ///
"""Run the open-weight VLM arm over a QuantiPhy split. Portable across HF Jobs, Kaggle and Colab.

Runs anywhere with a GPU and an ``HF_TOKEN``, because everything is environment-driven and all state
lives on the Hub. That portability is the point: the same script is a paid detached HF Job for the
one big test pass, and a free Kaggle batch run for the twenty prompt iterations that precede it.

    # HF Jobs -- --detach is NOT optional, see run_vision_job.py
    hf jobs uv run --detach --flavor a100-large --timeout 8h --secrets HF_TOKEN \\
      --env QUANTIPHY_GIT=git+https://github.com/<you>/quantiphy.git \\
      --env OUTPUT_REPO=<you>/quantiphy-runs --env SPLIT=validation \\
      --env VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct \\
      scripts/run_vlm_job.py

    # Kaggle / Colab: same script, three lines of notebook
    !pip install -q transformers accelerate bitsandbytes opencv-python-headless huggingface_hub
    %env OUTPUT_REPO=<you>/quantiphy-runs
    !python scripts/run_vlm_job.py

**Raw text is the artefact.** Each row appends a line to ``<run>/vlm_raw.jsonl`` holding the model's
reply *unparsed*, plus the prompt hash and frame times. Parsing and fusion then become free to
re-measure offline, exactly as ``replay_cache.py`` does for detections -- and on this project that
discipline has repeatedly turned a paid experiment into an unpaid one.

**It checkpoints and resumes by row index.** A Kaggle session dies at 12 hours and a Colab tab dies
whenever it likes, so resumption is not a nicety here. A re-run with the same ``RUN_NAME`` skips rows
already answered; change ``RUN_NAME`` whenever the prompt changes, or the checkpoint will replay
stale replies and measure nothing.

Environment:
    OUTPUT_REPO       dataset repo for results (required)
    SPLIT             "validation" (159 rows, has truth) or "test" (3,289 rows)
    VLM_MODEL         HF model id, default Qwen/Qwen3-VL-8B-Instruct
    RUN_NAME          output folder, default "<split>-vlm-<model tail>"
    SHARD             "k/n" contiguous slice, as in run_vision_job.py
    LIMIT             row cap, for a smoke test
    VLM_PROMPT        "brief" (default, mild CoT) or "direct" (no reasoning at all)
    VLM_FRAMES        frames per question, default 12
    VLM_4BIT          "1" to load 4-bit -- fits a 32B on a 40 GB A100 or Kaggle's 2x16 GB
    QUANTIPHY_GIT     pip-installable source, when the package is not already importable
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT_EVERY = 25
WORK = Path("/tmp/quantiphy-vlm")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def install_solver() -> None:
    """Make ``quantiphy`` importable, installing from git only if it is not already there.

    Kaggle and Colab clone the repo and run from inside it, so the package is usually already on the
    path and this is a no-op. HF Jobs starts from an empty uv environment and needs the install.
    """
    try:
        import quantiphy  # noqa: F401
        return
    except ImportError:
        pass
    source = os.environ.get("QUANTIPHY_GIT")
    if not source:
        raise SystemExit("quantiphy is not importable and QUANTIPHY_GIT is unset")
    log(f"installing {source}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", source], check=True)


def load_split(split: str):
    """Split metadata plus a video-id to path map. Mirrors run_vision_job.load_split."""
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

    # Filenames are not reliably clean: at least one validation clip has a leading space, and the id
    # column omits the extension. Index by normalised name and look up leniently.
    index = {path.name.strip().lower(): path for path in video_dir.rglob("*.mp4")}

    def find(video_id):
        stem = str(video_id).strip().lower()
        return index.get(f"{stem}.mp4") or index.get(stem)

    frame["video_path"] = frame["video_id"].map(find)
    log(f"{split}: {len(frame)} rows, {len(index)} videos, "
        f"{int(frame['video_path'].isna().sum())} rows without a video file")
    return frame


def apply_shard(frame):
    """Contiguous ``SHARD=k/n`` slice, preserving each row's original index.

    Contiguous rather than strided because rows are ordered by video: a contiguous shard re-uses each
    decoded clip across the ~5.8 questions that share it.
    """
    frame = frame.reset_index(drop=True)
    frame["row_index"] = frame.index
    shard = os.environ.get("SHARD")
    if shard:
        which, count = (int(part) for part in shard.split("/"))
        if not 1 <= which <= count:
            raise SystemExit(f"SHARD={shard} is out of range; expected 1/n .. n/n")
        bounds = [round(len(frame) * i / count) for i in range(count + 1)]
        frame = frame.iloc[bounds[which - 1]:bounds[which]].copy()
        log(f"SHARD {which}/{count}: rows {bounds[which - 1]}..{bounds[which] - 1} "
            f"({len(frame)} of {bounds[-1]})")
    limit = os.environ.get("LIMIT")
    if limit:
        frame = frame.head(int(limit)).copy()
        log(f"LIMIT set: {len(frame)} rows")
    return frame


def load_checkpoint(output_repo: str, name: str) -> tuple[dict, Path]:
    """Replies already collected for this run, keyed by row index."""
    from huggingface_hub import hf_hub_download

    WORK.mkdir(parents=True, exist_ok=True)
    local = WORK / "vlm_raw.jsonl"
    done: dict[int, dict] = {}
    try:
        remote = hf_hub_download(output_repo, repo_type="dataset",
                                 filename=f"{name}/vlm_raw.jsonl")
    except Exception as error:
        log(f"no checkpoint for {name} ({type(error).__name__}); starting fresh")
        local.write_text("", encoding="utf-8")
        return done, local

    text = Path(remote).read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip():
            record = json.loads(line)
            done[int(record["row_index"])] = record
    local.write_text(text if text.endswith("\n") or not text else text + "\n", encoding="utf-8")
    log(f"resuming {name}: {len(done)} rows already answered")
    return done, local


def push(output_repo: str, name: str, local: Path) -> None:
    from huggingface_hub import HfApi
    HfApi().upload_file(path_or_fileobj=str(local), path_in_repo=f"{name}/vlm_raw.jsonl",
                        repo_id=output_repo, repo_type="dataset")


def main() -> int:
    output_repo = os.environ.get("OUTPUT_REPO")
    if not output_repo:
        raise SystemExit("OUTPUT_REPO is required so results survive the session")
    split = os.environ.get("SPLIT", "validation")
    model_id = os.environ.get("VLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    style = os.environ.get("VLM_PROMPT", "brief")

    install_solver()

    from huggingface_hub import HfApi
    from tqdm import tqdm

    from quantiphy.backends.vlm import VlmBackend
    from quantiphy.parsing import build_request
    from quantiphy.prompting import build_prompt, parse_answer, system_prompt

    HfApi().create_repo(output_repo, repo_type="dataset", exist_ok=True, private=True)

    frame = apply_shard(load_split(split))
    name = os.environ.get("RUN_NAME") or f"{split}-vlm-{model_id.split('/')[-1]}"
    done, local = load_checkpoint(output_repo, name)

    # Fail before the model loads, not on row 1 of 159, if the style name is a typo.
    system = system_prompt(style)

    backend = VlmBackend(model_id,
                         frames=int(os.environ.get("VLM_FRAMES", 12)),
                         load_in_4bit=os.environ.get("VLM_4BIT") == "1")
    log(f"model {model_id}  4bit={backend.load_in_4bit}  frames={backend.frames}  "
        f"prompt={style}")
    log(f"device {backend.device}")

    prior_column = "ground_truth_prior" if "ground_truth_prior" in frame.columns else "prior"
    answered = 0
    with local.open("a", encoding="utf-8") as handle:
        for _, row in tqdm(list(frame.iterrows()), total=len(frame)):
            index = int(row["row_index"])
            if index in done:
                continue
            if row["video_path"] is None:
                # Never write a blank: a missing prediction scores a hard zero and still counts, so
                # the row must reach the fallback. Recording the reason is how we tell the two apart.
                record = {"row_index": index, "video_id": row["video_id"], "raw_text": "",
                          "note": "no video file", "model": model_id}
            else:
                request = build_request(row)
                depth = row.get("depth_info")
                prompt = build_prompt(request, str(row[prior_column]),
                                      None if depth is None or str(depth) == "nan" else str(depth),
                                      style=style)
                reply = backend.answer(index, str(row["video_id"]), str(row["video_path"]),
                                       system, prompt, request.timestamp)
                parsed = parse_answer(reply.raw_text, request.output_unit)
                record = {
                    "row_index": index, "video_id": reply.video_id, "raw_text": reply.raw_text,
                    "model": model_id, "note": reply.note, "prompt_style": style,
                    "prompt_sha": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12],
                    "frame_times": [round(t, 3) for t in reply.frame_times],
                    "unit": request.output_unit,
                    # Parsed values are recorded for convenience only. The raw text is the artefact:
                    # a parser change re-reads this file for free rather than paying for the run.
                    "parsed_value": parsed.value, "parse_route": parsed.route,
                }
            handle.write(json.dumps(record) + "\n")
            handle.flush()
            answered += 1
            if answered % CHECKPOINT_EVERY == 0:
                push(output_repo, name, local)
                log(f"checkpoint at {answered} new rows")

    push(output_repo, name, local)
    log(f"done: {answered} new rows, {len(done) + answered} total -> {output_repo}/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
