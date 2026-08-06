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

## What the undershoot actually was

**Not** boxes overestimating non-rectangular objects (the SAM2 hypothesis this file used to carry).
It was `parse_prior` handing the grounding model a phrase containing the quantity word.

Grounding-DINO was being asked to detect **"billiard ball diameter"** and **"walking velocity"**
instead of "billiard ball" and the walking person. It returns its best box for a loosely related,
larger region, so `gamma = prior_world / prior_pixels` comes out too small — and because the prior
sets the scale, **every** answer in the row shrinks by the same factor. That is exactly the uniform
multiplicative signature we measured.

The old code did have a fallback that stripped quantity words, but it only ran `if not object_name`,
and the `of|for` regex always returned a non-empty *wrong* string, so it never fired.

This was found for **zero GPU spend**, by replaying the 20-row smoke run's cached detections
offline. The decomposition:

```
 #   ratio  prior_px  needed  prior phrase              -> target phrase
 0   0.099     501.2    49.7  'ruler calibre'           -> 'wood block'
14   0.419      55.5    23.3  'billiard ball diameter'  -> 'balck ball'
16   0.766      55.5    42.5  'billiard ball diameter'  -> 'orange ball'
 4   0.143      53.1     7.6  'walking velocity'        -> 'two black road signs'
```

`prior_pixels` would have to be **0.42x** what we measured for the answers to land. The error is
entirely in the prior's pixel measurement.

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

| # | Item | Cost | Notes |
|---|---|---|---|
| 1 | Merge `fix/prior-grounding-phrase` once the confirm run is read | free | see "Confirmation run" below |
| 2 | Full 159-row validation run on `l4x1` | ~1–3 h, $5–15 | **now worth doing** — do it on the fixed solver, not before |
| 3 | Fallback arm (VLM estimate), fused in log space | ~$5 GPU | more urgent than it was: 325 test rows now land here by design |
| 4 | Better phrase for bare gerunds — `walking speed` strips to `'walking'` (17 test rows) | free | measure before bothering; "walking" may ground the walker fine |
| 5 | SAM2 masks / CoTracker3 | ~$10 GPU | **deprioritised** — the evidence says phrase, not box shape. Re-evaluate after (2) |
| 6 | Follow-up email: 4 unanswered questions | free | parked by choice until ~September |
| 7 | Measure `--shrink` | free once (2) exists | needs (2) |

## Confirmation run — 2026-08-05

Job `6a73ddeda00abefd4b294c9b`, `l4x1`, `LIMIT=20`, `RUN_NAME=validation-grounding-fix1`, against
branch `fix/prior-grounding-phrase`. Cold cache: all four prior phrases changed.

**Success criterion, fixed in advance:** median pred/truth moves from 0.419 materially toward 1.0
and solved rises above 13/20 (the `~` fix alone should add up to 6). If it does not move, the phrase
hypothesis is wrong — stop and re-plan rather than pushing on.

> Result: see the session notes below / re-read with
> `py -3.12 scripts/replay_cache.py --run validation-grounding-fix1`.

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
* Session of 2026-08-05: diagnosis done entirely offline against the cached detections, then one
  20-row confirm run. The lesson worth keeping: **the cache made a $5–15 blind run unnecessary.**
