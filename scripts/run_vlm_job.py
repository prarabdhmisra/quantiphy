#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch",
#   # Not optional and not obvious: `AutoProcessor.from_pretrained` builds a *video* processor for
#   # Qwen3-VL even though this backend only ever hands it PIL frames, and Qwen3VLVideoProcessor is
#   # torchvision-gated. Without it the job dies on `requires the Torchvision library` after the
#   # whole 62-package install and the model download -- about three minutes of paid GPU per launch.
#   "torchvision",
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
    ROW_IDS           comma-separated row indices to run, for a stratified subset. Exits rather
                      than running short if any requested row is absent from the split.
    PRIOR_SCALE       Experiment B: multiply the stated prior's magnitude by this factor before
                      building the prompt (default 1.0 = untouched). Units, object names and any
                      `t=` timestamp are preserved; only the number moves. Change RUN_NAME with it,
                      or the checkpoint replays the unscaled replies.
    VLM_PROMPT        "brief" (default, mild CoT), "direct" (no reasoning), or "strict"
                      (one sentence, no declining, no zero, 2 sig figs -- see prompting.py)
    VLM_FRAMES        frames per question, default 12
    VLM_MAX_NEW_TOKENS  generation budget per reply, default 320 -- was 128 and truncated 841 of
                      3,289 test replies before the ANSWER: marker; see vlm.py
    VLM_MAX_SIDE      longest frame side in px, default 768 -- a memory bound, see vlm.py
    VLM_4BIT          "1" to load 4-bit -- fits a 32B on a 40 GB A100 or Kaggle's 2x16 GB
    QUANTIPHY_GIT     pip-installable source, when the package is not already importable
    GITHUB_TOKEN      PAT for a PRIVATE source repo. Pass as a SECRET, never --env:
                      `hf jobs inspect` echoes the environment dict in plaintext. Needs only
                      read-only Contents scope on the one repo. Unset = anonymous clone.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT_EVERY = 25
WORK = Path("/tmp/quantiphy-vlm")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _with_credentials(source: str) -> str:
    """Inject a GitHub PAT into an https source URL, for a private repo.

    The token arrives as a *secret* (``--secrets GITHUB_TOKEN=...``), never as ``--env``: ``hf jobs
    inspect`` echoes the whole environment dict back in plaintext, so a token baked into
    ``QUANTIPHY_GIT`` would be readable from job metadata by anyone who can see the job.

    Injecting after the scheme rather than string-formatting the whole URL matters, because
    ``git+https://host/o/r.git@ref`` already uses ``@`` for the ref. Credentials add a *second* one,
    and the fallback clone below splits the ref off with ``partition("@")`` -- first match wins. So
    the ref is always parsed from the clean URL, and credentials are added afterwards.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token or "://" not in source:
        return source
    scheme, _, rest = source.partition("://")
    host = rest.split("/", 1)[0]
    if host != "github.com" or "@" in host:      # already carries credentials, or not GitHub
        return source
    return f"{scheme}://x-access-token:{token}@{rest}"


def _redact(text: str) -> str:
    """Never let a token reach a log line or an exception message."""
    for name in ("GITHUB_TOKEN", "GH_TOKEN"):
        token = os.environ.get(name)
        if token:
            text = text.replace(token, "***")
    return text


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
    authed = _with_credentials(source)
    log(f"installing {source}" + ("  (with GitHub credentials)" if authed != source else ""))

    # `hf jobs uv run` executes inside an ephemeral uv environment with **no pip in it**, so
    # `python -m pip` fails outright -- which is exactly how this script's first two launches died,
    # in seconds, on `No module named pip`. run_vision_job.py already carried this ladder; the VLM
    # arm had the naive version because it had never actually run. uv first, then pip for anywhere
    # else this runs (Colab, Kaggle, a plain venv), then a bare clone onto sys.path.
    attempts = (["uv", "pip", "install", "-q", "--python", sys.executable, authed],
                [sys.executable, "-m", "pip", "install", "-q", authed])
    for command in attempts:
        try:
            subprocess.check_call(command)
            return
        except (subprocess.CalledProcessError, FileNotFoundError, OSError) as error:
            # `error` stringifies the whole command, tokenised URL included -- redact it.
            log(_redact(f"  {command[0]} install failed ({error}); trying the next option"))

    # `git+<url>@<ref>` is pip syntax, not git's: `git clone` would take the whole thing as a URL.
    url = source.removeprefix("git+")
    url, _, ref = url.partition("@")          # parsed from the CLEAN url, before credentials
    checkout = WORK / "src"
    log(f"  falling back to a plain clone of {url}" + (f" at {ref}" if ref else ""))
    if not checkout.exists():
        clone = ["git", "clone", "--depth", "1"] + (["--branch", ref] if ref else [])
        try:
            subprocess.check_call(clone + [_with_credentials(url), str(checkout)])
        except subprocess.CalledProcessError as error:
            raise SystemExit(_redact(f"clone failed: {error}")) from None
    sys.path.insert(0, str(checkout))
    import quantiphy  # noqa: F401  -- fail here, not 200 rows into the run


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
    row_ids = os.environ.get("ROW_IDS")
    if row_ids:
        wanted = [int(part) for part in row_ids.replace("\n", ",").split(",") if part.strip()]
        frame = frame[frame["row_index"].isin(wanted)].copy()
        missing = sorted(set(wanted) - set(frame["row_index"]))
        log(f"ROW_IDS: {len(frame)} of {len(wanted)} requested rows selected"
            + (f"; {len(missing)} not in split: {missing[:10]}" if missing else ""))
        if len(frame) != len(wanted):
            # A silently short row set would make a slope fit look converged on the wrong rows.
            raise SystemExit(f"ROW_IDS selected {len(frame)} rows, expected {len(wanted)}")
    limit = os.environ.get("LIMIT")
    if limit:
        frame = frame.head(int(limit)).copy()
        log(f"LIMIT set: {len(frame)} rows")
    return frame


# --- Experiment B: prior perturbation -------------------------------------------------------
# Canonical copy lives in papers/quantiphy-physworld/analysis/perturb_prior.py, where a free CPU
# gate asserts it moves every parsed SI value by exactly the factor over all 3,289 rows. Inlined
# here because `hf jobs uv run` uploads this file alone -- the package comes from QUANTIPHY_GIT,
# so importing it would mean pushing before every launch. Keep the two in sync by hand.
_PERTURB_NUMBER = re.compile(r"(?<![A-Za-z0-9^./])(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
_PERTURB_CLAUSE = re.compile(r"([\n;]+)")
_PERTURB_TIME_PREFIX = re.compile(r"^\s*t\s*=\s*-?\d+(?:\.\d+)?\s*,", re.IGNORECASE)


def _perturb_format(value: float) -> str:
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def scale_prior_text(text: str, factor: float) -> str:
    """Multiply every magnitude in a prior string by ``factor``, preserving all other characters.

    A leading ``t=0.6,`` names the instant the prior holds at. That is a TIME, not a magnitude --
    scaling it would change the question's physics rather than its scale -- so it is protected.
    """
    if not text or not str(text).strip():
        return text

    out_parts = []
    for part in _PERTURB_CLAUSE.split(str(text)):
        if _PERTURB_CLAUSE.fullmatch(part) or not part.strip():
            out_parts.append(part)
            continue
        prefix = ""
        prefix_match = _PERTURB_TIME_PREFIX.match(part)
        if prefix_match:
            prefix = part[: prefix_match.end()]
            part = part[prefix_match.end():]
        cut = max(part.rfind("="), part.rfind("~"))
        if cut < 0:
            out_parts.append(prefix + part)
            continue
        label, right = part[: cut + 1], part[cut + 1:]
        right = _PERTURB_NUMBER.sub(
            lambda m: _perturb_format(float(m.group(1)) * factor), right, count=1)
        out_parts.append(prefix + label + right)
    return "".join(out_parts)


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

    from quantiphy.backends.vlm import DEFAULT_MAX_NEW_TOKENS, MAX_FRAME_SIDE, VlmBackend
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
                         max_new_tokens=int(os.environ.get("VLM_MAX_NEW_TOKENS",
                                                           DEFAULT_MAX_NEW_TOKENS)),
                         load_in_4bit=os.environ.get("VLM_4BIT") == "1",
                         max_frame_side=int(os.environ.get("VLM_MAX_SIDE", MAX_FRAME_SIDE)))
    log(f"model {model_id}  4bit={backend.load_in_4bit}  frames={backend.frames}  "
        f"prompt={style}  max_side={backend.max_frame_side}  "
        f"max_new_tokens={backend.max_new_tokens}")
    log(f"device {backend.device}")

    prior_column = "ground_truth_prior" if "ground_truth_prior" in frame.columns else "prior"

    # Experiment B: rewrite the prior's stated magnitude, leaving the question, video, unit and
    # any timestamp byte-identical. A scale-recovering method's answer must move linearly with
    # this; a method emitting a category-typical magnitude will not move at all.
    prior_scale = float(os.environ.get("PRIOR_SCALE", 1.0))
    if prior_scale != 1.0:
        before = str(frame[prior_column].iloc[0])
        frame[prior_column] = frame[prior_column].map(
            lambda text: scale_prior_text(text, prior_scale))
        log(f"PRIOR_SCALE={prior_scale}: {before!r} -> {frame[prior_column].iloc[0]!r}")
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
