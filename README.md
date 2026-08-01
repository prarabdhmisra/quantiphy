# QuantiPhy Challenge — NeurIPS 2026

Entry for the [QuantiPhy Challenge](https://quantiphy.stanford.edu/competition/index.html):
estimating physical quantities (size, velocity, acceleration) from short video.
**Primary target: Track B (Open-Weight). Deadline: plan for Oct 1, 2026.**

Full strategy: `~/.claude/plans/inherited-doodling-sun.md`

## Thesis

Every question supplies one physical prior in real units and asks for another quantity. The camera
is static, clips are 2–3 s, motion is rigid and translational, `fps` is given, and metric depth is
given for all 3D items. So the task is a **geometry problem**: measure in pixels, recover the
pixel→world scale from the prior, convert.

The benchmark paper's own finding is that VLMs don't do this — they score 39–51 MRA *blind*
(no video at all) versus 56 with video, and their answers barely change when the given prior is
multiplied by 1000. A tool-based measurement pipeline is the open lane.

## Status

- [x] Scoring harness, validated against the organizers' published baseline
- [x] Submission pre-flight validator
- [ ] Register team; email organizers (`ORGANIZER-EMAIL.md`) — **blocker**
- [ ] Pull test videos; VLM control arm
- [ ] Geometric solver (2D → 3D)
- [ ] Fusion + blowup detector

## Layout

    quantiphy/scoring.py           MRA metric + paired bootstrap
    scripts/validate_submission.py pre-flight check before spending daily quota
    tests/test_scoring.py          anchors the scorer to published numbers
    data/fixtures/                 GPT-5.1 validation outputs, published results

## Usage

```bash
py -3.12 -m pytest tests/ -q                                    # 21 tests
py -3.12 scripts/validate_submission.py sub.csv --expect-rows 3289
```

```python
from quantiphy.scoring import score, category_labels, paired_bootstrap
result = score(frame)                       # macro MRA 0.4836 | S2 0.5687  D2 0.3459 ...
verdict = paired_bootstrap(baseline, candidate, category_labels(frame))
```

## Where to run the vision stack

| Job | Where | Why |
|---|---|---|
| Iterating on the backend | **Colab Pro** (`notebooks/colab_vision.ipynb`) | Interactive; T4 is enough for Grounding-DINO. |
| Batch over all 568 videos | **HF Jobs** (Pro account) | Runs detached, so a dropped browser tab doesn't kill it. |
| Long unattended sweeps | **Kaggle** (30 h/week free) | Free, but 9 h/session and no detach. |

HF Jobs is the right home for the full test run — `l4x1` is the sweet spot (more VRAM than a T4,
far cheaper than an A100, and Grounding-DINO is not compute-bound):

```bash
# Smoke test first -- 20 rows, ~10 minutes, confirms the whole path works before spending hours.
hf jobs uv run --flavor l4x1 --timeout 1h --secrets HF_TOKEN \
  --env QUANTIPHY_GIT=git+https://github.com/<you>/quantiphy.git \
  --env OUTPUT_REPO=<you>/quantiphy-runs \
  --env SPLIT=validation --env LIMIT=20 \
  scripts/run_vision_job.py

# Then the real validation run, which prints a scored result at the end.
hf jobs uv run --flavor l4x1 --timeout 3h --secrets HF_TOKEN \
  --env QUANTIPHY_GIT=... --env OUTPUT_REPO=... --env SPLIT=validation \
  scripts/run_vision_job.py

hf jobs logs <job_id> --follow
```

Only run `SPLIT=test` (3,289 rows) once the validation number is good.

The job **checkpoints every 100 rows** to `OUTPUT_REPO`, pushing both partial predictions and the
detection cache, and resumes from them on restart. Because detections are keyed by (video, phrase)
and 3,289 questions share only 568 videos, a warm cache turns a re-run into minutes.

`QUANTIPHY_GIT` needs this repo pushed to GitHub. Without it, upload the package to a Hub dataset
repo and pass `QUANTIPHY_HF_REPO` instead.

Note the job downloads videos from the Hub *on HF infrastructure*, so there is no need to have
them locally — a local copy is only useful for eyeballing failures.

## Things that will silently cost you points

Measured, not assumed — see the plan for the numbers behind each.

- **Overshooting is fatal, undershooting is cheap.** `pred >= 1.9x` truth scores 0; a `0.5x`
  undershoot still scores 0.4. Shrink when uncertain. 25% of GPT-5.1's rows score exactly 0, and
  57% of those are overshoots — the largest recoverable pool of points in the benchmark.
- **The scorer performs no unit conversion, anywhere.** Answer in the unit the question names.
- **Blank, NaN, and zero are hard zeros**, still counted in the category mean. Always fall back.
- **Any empty category makes the whole average undefined**, not partial.
- **Aggregate ensembles in log space**, never an arithmetic mean.
- **Don't calibrate on the 159-row validation split.** Per-category shrinkage measured
  in-sample 0.493 but leave-one-out 0.463 — worse than the 0.484 baseline. It overfits.
- **Validation has a ±5.7 point 95% CI.** Use `paired_bootstrap`, and accept a change only when
  its interval excludes zero.
- The paper's Appendix A.2 threshold set is a typo; the code's `{0.1..0.9, 0.95}` is authoritative.
