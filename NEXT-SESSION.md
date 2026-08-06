# Resume here

Last worked: **2026-08-05**. Branch `fix/prior-grounding-phrase` is pushed and green (114 tests);
`main` is deliberately still at the last *measured* state.

## Paste this to start the next session

> Resuming the QuantiPhy Challenge (NeurIPS 2026) entry. Read
> `C:\Users\prara_\quantiphy\NEXT-SESSION.md` first. Repo is public at
> https://github.com/prarabdhmisra/quantiphy. The 2.4x undershoot is diagnosed and fixed on branch
> `fix/prior-grounding-phrase` — read "What the undershoot actually was" before touching anything.
> Don't re-derive anything in "Do not re-derive"; it cost real money. Ask me before spending more
> than ~$20 in a session.

## The undershoot is NOT one bug — read this before believing any single-cause story

Two hypotheses have now been tested and **both are wrong**:

1. *Boxes overestimate non-rectangular objects* (the old SAM2 story). No — `'billiard ball'` is
   detected as a clean, tight, stable box: 56.5 x 59.1 px, score 0.661, detection rate 1.00, width
   spread only 51–58 px across frames. 57.2 mm over 57.8 px is **0.990 mm/px**, and that scale is
   right.
2. *The prior's grounding phrase contains the quantity word* (tested 2026-08-05, measured, wrong).
   The phrase defect is real — 412 test rows — but Grounding-DINO turned out to be robust to it.
   Stripping it moved `prior_pixels` from 55.5 to 57.8 px on the billiard rows and 53.1 to 54.4 on
   the walking rows. **The bias did not move: median pred/truth 0.419 -> 0.403.**

The "uniform 2.4x multiplicative undershoot" was an **artifact of averaging across question types**.
Split by what the question asks for, the 19 solved rows look nothing like a single bias:

| measurement | n | median ratio | ratios |
|---|---|---|---|
| **distance** | 4 | **0.226** | 0.06 0.14 0.31 0.40 — *every one undershoots* |
| velocity | 6 | **0.094** | 0.00 0.03 0.09 0.10 0.74 1.54 |
| **width** | 2 | **7.409** | 7.14 7.68 — *overshoot, which is fatal* |
| length | 3 | 1.708 | 0.21 1.71 9.51 |
| height / size / diameter / accel | 4 | ~0.7 | 1.01 1.09 0.37 0.44 |

Excluding `distance` rows the median is **0.736 with 7 of 15 overshooting** — not an undershoot at
all. There is no global correction factor to find here, and fitting one would be fitting noise.

### The one clean, verified bug: "distance between A and B"

`_TARGET_KEYWORDS` maps `distance` to measurement `"distance"`, and `extent_for()` has no branch for
it, so it falls through to `max(width, height)` — **one object's own box size, where a separation
between two objects was asked for.**

Verified numerically on the billiard rows, where the prior's scale is known good at 0.990 mm/px:

```
row 14  truth 14.74 cm  needs 148.9 px of separation   we measured 59.9 px (one ball's box)
row 17  truth 18.04 cm  needs 182.3 px of separation   we measured 57.0 px (one ball's box)
```

**203 of 3,289 test rows (6.2%) ask "distance between A and B"**, plus 75 more asking for a distance
some other way. All of them are currently measuring the wrong quantity. This needs two centroids,
not one extent — and the detector already returns centroids, so it is a solver fix, not a vision one.

### Counted over all 3,289 real test rows — these are the numbers that justify the fix

| Rows | Defect |
|---|---|
| **412** | prior phrase contained the quantity word |
| **205** | `acceleration = 9.8 m/s^2` grounded an object literally called "acceleration" |
| **124** | `t=0.6, ball acceleration` lost the timestamp *and* glued `t=0.6` onto the phrase |
| 4 | `calibre` fell through to `max(w,h)` and measured the whole ruler |
| 0 | zero-valued priors — checked and ruled out |

`tests/test_no_scale_prior_phrase_is_contaminated_on_the_full_test_split` now asserts all of these
are zero, offline, in ~2 s. That assertion, not the 20-row MRA, is the real evidence.

## What changed on the branch

* **`quantiphy/parsing.py`** — `_prior_object()` strips quantity words unconditionally; a leading
  `t=0.6,` lifts into `timestamp`; `~` accepted as a separator alongside `=`. A prior naming no
  object returns `""`.
* **`quantiphy/solver.py`** — an empty prior phrase declines with "prior names no groundable
  object" instead of scaling off a nonsense box. A row whose target phrase is missing gets
  `+target-from-prior` on its method, so the collapse is visible in output.
* **`quantiphy/backends/grounding.py`** — `detection_rate` divides by frames *sampled*, not clip
  length (a 428-frame clip used to cap confidence at 0.112 no matter how clean the detections were);
  `calibre`/`caliber` use the small axis.
* **`scripts/replay_cache.py`** — new. The offline harness this diagnosis came from.
* **`scripts/run_vision_job.py`** — `RUN_NAME` env var. Checkpoints resume by row index, so without
  it a re-run replays stale answers and measures nothing.

**Expect the solved count on test to drop.** 325 rows move from "confidently wrong" to "unsolved →
fallback". That is the right side of a metric where overshoot is fatal, not a regression.

## `scripts/replay_cache.py` — use this before spending anything

```bash
py -3.12 scripts/replay_cache.py [--limit 20] [--run validation-grounding]
```

Detection is the expensive part and it is already pushed to the Hub per run. Everything downstream
is deterministic CPU code, so any parsing or geometry change can be measured against detections we
have already paid for, in about two seconds.

The one catch: detections are cached per `(video, phrase)`, so **a change that renames phrases shows
up as cache misses, not new numbers**. That is correct and the script says so loudly. Those rows
need a fresh detection pass — which for the smoke set is 4 videos and a few cents.

## Pending

In priority order, by measured evidence rather than by hunch:

| # | Item | Rows | Cost | Notes |
|---|---|---|---|---|
| 1 | **"distance between A and B" must use two centroids, not one box extent** | **278** | free | The only cleanly verified bug left. Needs a second target phrase out of `parse_question`, then a centroid separation in `grounding.py`. Start here. |
| 2 | **Diagnose the velocity rows** — median ratio 0.094, four of six near zero | ~1,000 | free | Biggest category and the worst performing. Replay the cache; the quadratic fit or the `fps`/timestamp path is suspect. No GPU needed. |
| 3 | Kill the fatal overshoots: gate on prior confidence | — | free | The `pedestrian walking` prior scored 0.341 with box width jittering 13–59 px and produced 7x overshoots. `min_confidence` already exists and is unused (`solve_row` defaults it to 0.0). Now that `detection_rate` is fixed, confidence is finally meaningful. |
| 4 | Decide on `fix/prior-grounding-phrase` | 412 | free | Real defects, 114 tests green, but it did **not** move the metric. Merge on correctness grounds, not on a score claim. |
| 5 | Full 159-row validation run on `l4x1` | — | ~1–3 h, $5–15 | Only after 1–3. Running it now would measure known-broken behaviour. |
| 6 | Fallback arm (VLM estimate), fused in log space | 325 | ~$5 GPU | More urgent than before: 325 rows now decline by design. |
| 7 | SAM2 masks / CoTracker3 | — | ~$10 GPU | **Deprioritised twice over.** The billiard box is already tight and correct; masks would not have helped. |
| 8 | Follow-up email; `--shrink` measurement | — | free | Parked / needs (5). |

## Confirmation run — 2026-08-05

Job `6a73ddeda00abefd4b294c9b`, `l4x1`, `LIMIT=20`, `RUN_NAME=validation-grounding-fix1`, against
branch `fix/prior-grounding-phrase`. Cold cache: all four prior phrases changed.

**Success criterion, fixed in advance:** median pred/truth moves from 0.419 materially toward 1.0
and solved rises above 13/20 (the `~` fix alone should add up to 6). If it does not move, the phrase
hypothesis is wrong — stop and re-plan rather than pushing on.

**Result — half met, and the half that failed is the important one.**

| | before | after |
|---|---|---|
| solved | 13/20 | **19/20** |
| median pred/truth | 0.419 | **0.403** |
| within 2x either way | 4/13 | 5/19 |
| overshoots | 1/13 | **7/19** |

Solved rose as predicted (the `~` fix, worth ~0 points on test — it buys measurement power, not
score). **The bias did not move**, so the phrase hypothesis is refuted as the cause. The one unsolved
row is the gravity prior, correctly declined.

Note the overshoot count went 1/13 -> 7/19: the six rows the `~` fix unlocked include two 7x
overshoots, which score exactly zero. Unlocking rows is not automatically progress.

Re-read any run offline with `py -3.12 scripts/replay_cache.py --run <name>` — no GPU needed.

## Baseline to beat, and the caveat on the old number

| | |
|---|---|
| Smoke test 2026-08-02, 20 rows, pre-fix | 13/20 solved, MRA 0.300 over solved, **0.195** as-submitted |
| Median pred/truth, pre-fix | 0.419 |
| GPT-5.1 on full validation | **0.4856** |
| Human average / top humans | 0.556 / 0.72 |
| Best open-weight (Qwen3-VL-32B) | 46.0 |

**20 rows, all S2/D2.** No macro average exists for that sample and the CI is enormous. It ranks as
"the pipeline runs", nothing finer.

## State

| | |
|---|---|
| Repo | https://github.com/prarabdhmisra/quantiphy (public, MIT) |
| Tests | **114 passing** (`py -3.12 -m pytest tests/ -q`) |
| Plan | `~/.claude/plans/snappy-launching-candy.md` |
| Track | **B (Open-Weight)** primary, A secondary |
| Deadline | **Plan for Oct 1, 2026** (site advertises Nov 5, but its own timeline finalizes rankings mid-October) |

**Built:** `scoring.py` (MRA + paired bootstrap), `units.py`, `parsing.py`, `geometry.py`,
`vision.py` (backend Protocol), `backends/grounding.py` (Grounding-DINO), `solver.py`,
`scripts/run_vision_job.py` (HF Jobs, checkpoints + resumes), `scripts/replay_cache.py`,
`scripts/make_submission.py`, `scripts/validate_submission.py`, `notebooks/colab_vision.ipynb`.

The HF Jobs path is proven end to end: install, video download, GPU detection, checkpoint, resume,
scoring, exit 0.

## Do not re-derive — these are measured, and they cost real money

* MRA thresholds are `{0.1..0.9, 0.95}` **per the code**. The paper's Appendix A.2 set
  `{0.5..0.95}` is a typo — it yields 0.376 against the published 0.486.
* **Overshoot is fatal, undershoot is cheap.** `pred >= 1.9x` truth scores 0; `0.5x` still scores
  0.4. 25% of GPT-5.1 rows score exactly 0 and **57% of those are overshoots** — the largest
  recoverable pool. Oracle fix = +19 pts; a realistic detector = +7.5 pts.
* **Do not calibrate on the 159-row validation set.** Per-category shrinkage: in-sample 0.493,
  leave-one-out **0.463** — worse than the 0.484 baseline. Global recalibration is +0.002.
* **The unit-error hypothesis is refuted** — only ~4% of GPT-5.1 errors sit near powers of ten.
* Validation has a **±5.7 pt 95% CI**. Use `paired_bootstrap`; accept only if the CI excludes zero.
* The scorer does **no unit conversion**; blank/NaN/zero are hard zeros; any empty category makes
  the whole average undefined.
* Aggregate ensembles in **log space**, never an arithmetic mean.
* The 20-row detection cache lives in `prarabdhmisra/quantiphy-runs`. Replaying it is free — do that
  before proposing any GPU spend on a solver-logic question.

## Team

**Vector Syndicate** — 2 people. Organizer email **sent 2026-08-01** (contents in
`ORGANIZER-EMAIL.md`).

## Organizers — nothing new as of 2026-08-05

They wrote again saying the webpage has "a new template" and that the leaderboard will not go up
"any time soon". Both were checked against the live site:

* The template is **byte-identical** to the pinned copy — same SHA-256, same 739,113 bytes, and
  both it and `competition/index.html` still report `Last-Modified: Sun, 02 Aug 2026 13:29:16 GMT`.
  Their message refers to the update we already absorbed on 2026-08-02. **No code work follows.**
* `leaderboard.html`, `submit.html`, `upload.html`, `rules.html` all still 404.

Consequence, unchanged: **no external feedback signal until roughly October.** `quantiphy/scoring.py`
on the 159-row validation split is the only evidence we get, at ±5.7 pt. The 3-per-day quota is
irrelevant for now. The upload path stays untested until crunch time.

**Still unanswered** (parked until ~September, none of it blocking): which deadline is
authoritative, fine-tuning / external data / ensemble rules, whether gated weights count as
open-weight, and team eligibility across tracks.

## Submitting

```bash
py -3.12 scripts/make_submission.py predictions.csv --out sub.csv
py -3.12 scripts/validate_submission.py sub.csv        # must exit 0
```

`make_submission.py` fills `parsed_value` in the pinned official template and copies every other
column through untouched; `validate_submission.py` diffs the result against that template
row-for-row. Re-check the template's SHA-256 against the live file first — see
`data/fixtures/README.md`. `--shrink` is available and **unmeasured**; leave it at 1.0 until it is
tested with `paired_bootstrap`.

## Notes

* Compute: **Colab Pro** to iterate, **HF Jobs `l4x1`** for batch (detached, survives a closed
  laptop), Kaggle for free sweeps. Do not buy a GPU — the whole competition is tens of GPU-hours.
* Videos are **not** stored locally. Colab and HF Jobs fetch them on the remote machine.
* `README.md` publicly documents the competitive analysis. Trim it into a gitignored `NOTES.md` if
  that becomes a concern.
* Session of 2026-08-01 cost ~$100, mostly research. 2026-08-02: ~$37 plus three `l4x1` smoke runs.
* Session of 2026-08-05: ~$52 of Claude time plus one 7-minute `l4x1` run (cents). Two lessons worth
  keeping. **The detection cache makes solver-logic questions free** — replaying it beats paying for
  a run. And **decompose by question type before believing a global bias**: "we undershoot 2.4x
  uniformly" survived two sessions and dissolved the moment the 19 rows were split by measurement,
  into a distance bug, collapsed velocities, and 7x overshoots pulling the other way.
