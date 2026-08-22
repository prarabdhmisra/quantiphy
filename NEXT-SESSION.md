# Resume here

Last worked: **2026-08-22**.

> **THE PORTAL IS LIVE.** The single most important thing on this page. Submissions are scored on
> upload and return a per-category MRA immediately, 3 per UTC day. Everything below that says
> "no feedback until October" was wrong and has been corrected — see "Submitting". Get a submission
> on the board before spending anything on GPU.

> **Where things live.** This document is on both branches and is accurate on both. The *code* from
> 2026-08-05 and 2026-08-22 is only on **`fix/prior-grounding-phrase`** (pushed, 129 tests green) —
> `main` is deliberately still at the last *measured* state, 94 tests. Anything below that names
> `scripts/replay_cache.py` or 129 tests needs that branch checked out:
>
> ```bash
> git checkout fix/prior-grounding-phrase
> ```

## Paste this to start the next session

> Resuming the QuantiPhy Challenge (NeurIPS 2026) entry. Read
> `C:\Users\prara_\quantiphy\NEXT-SESSION.md` first. Repo is public at
> https://github.com/prarabdhmisra/quantiphy. Work on branch `fix/prior-grounding-phrase` (129 tests
> green). **The submission portal is live and scores on upload, 3/day — read "Submitting" first.**
> The distance bug is fixed and verified; three single-cause theories of the bias have now been
> refuted, so read "The depth hypothesis is also refuted" and "Do not re-derive" before proposing a
> lever. Don't re-derive anything in those; it cost real money. Ask me before spending more than
> ~$20 in a session.

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

### The one clean, verified bug: "distance between A and B" — **FIXED 2026-08-22**

> Kept for the diagnosis; the fix and its verification are in "The distance fix" below. The row
> counts here (203 + 75) were re-counted more precisely as 192 + 86 = 278.

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

Added 2026-08-22:

* **`quantiphy/parsing.py`** — `parse_question` now returns a **5-tuple** with a second object;
  `SolveRequest.target_object_b`. `_strip_pair_object` keeps the "in/on" descriptor that
  `_strip_object` destroys, so "the person in the black shirt" survives as a distinguishable phrase.
* **`quantiphy/vision.py`** — `PixelMeasurement.centroid_px`. The `measure()` Protocol is unchanged,
  so no backend needed rewriting and `NullBackend` still satisfies it.
* **`quantiphy/backends/grounding.py`** — `centroid_at()`, reusing the same robust quadratic fit as
  `kinematics`/`displacement_px`.
* **`quantiphy/geometry.py`** — `_name_overlap()`: compound-noun containment matching in `depth_for`.
* **`quantiphy/solver.py`** — the four distance routes; `_separation_px()`; `+separation` on method.
* **`scripts/make_baseline.py`** — new. Zero-vision baseline from validation-truth medians.
* **`scripts/replay_cache.py`** — `CachedBackend` fills `centroid_px` too, or separations silently
  fail to replay.

**Expect the solved count on test to drop.** 325 rows move from "confidently wrong" to "unsolved →
fallback". That is the right side of a metric where overshoot is fatal, not a regression.

## The distance fix — done 2026-08-22, and it works

The 278 distance rows split into four mechanisms, not one. `parse_question` now returns a second
object and the measurement name routes the row:

| measurement | rows | what happens |
|---|---|---|
| `distance-pair` | **183** | two phrases → gap between their centroids, both evaluated at the requested instant |
| `distance-twin` | 44 | "the two cars": one phrase twice → **declines** (needs top-2 detections) |
| `distance` | 41 | genuinely one object's own span (wingspan) → unchanged |
| `distance-camera` | 10 | `depth_info` states the answer → **exact, no vision at all** |

Verified against cached detections, on the row the previous session diagnosed:

```
row 14   truth 14.74 cm   needed 148.9 px of separation
  before:  59.9 px  (one ball's box)   pred/truth 0.40   -> scores 0
  after:  161.2 px  (centroid gap)     pred/truth 1.083  -> near-full credit
```

Replay overall went 15/20 → 16/20 solved, median pred/truth 0.736 → 0.874. Note the sample is tiny
and mostly *not* distance rows; the row-14 decomposition is the real evidence, not the median.

**Declining is deliberate in two places.** A `distance-pair` row whose second object is never
detected returns nothing rather than falling back to the first object's extent — that fallback is
exactly the confidently-wrong 2.5x answer the fix exists to remove, and it scores zero either way.
Same for `distance-twin`.

Free side effect: `geometry.depth_for` now matches compound nouns by containment ("basketball"
against the key `distance_ball_camera`), scored *below* an exact token hit so a precise key still
wins. That lifted 3D depth engagement from 50.8% → **56.0%** of the 1,548 3D rows.

## The depth hypothesis is also refuted — do not re-derive this

It looked like the biggest lever on the board: the depth correction engages on only 786 of 1,548 3D
rows, and S3+D3 are **half** the macro average. Measuring the ratio on the 786 rows where matching
*does* work killed it:

```
target/prior depth ratio:  p5 0.743   p25 0.963   p50 1.000   p75 1.106   p95 1.477
within ±10% of 1.0 ............ 56.4%
ratio=1 is a >1.5x error ....... 7.8%
ratio=1 scores ~0 .............. 1.4%
```

When depth is missing the solver falls back to the flat 2D scale, which *implicitly assumes ratio =
1* — and that assumption is centred exactly on the truth. So the 762 "unmatched" rows are not
broken. The prize is the ~60-row tail where ratio=1 is a >1.5x error, and any proxy policy must be
conservative, because a proxy larger than the prior's depth inflates the answer and overshoot is
fatal.

That is now **three** single-cause stories tested and refuted (SAM2 boxes, the prior phrase, depth).
The lesson keeps holding: decompose and measure the sub-population before believing a lever.

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
| ~~1~~ | ~~**"distance between A and B" must use two centroids**~~ | ~~278~~ | done | **DONE 2026-08-22.** Verified on cached detections: the billiard row went from 59.9 px (one ball's box) to 161.2 px of separation, pred/truth 0.40 → **1.083**. See "The distance fix" below. |
| 1a | **Upload the zero-vision baseline** (`baseline.submission.csv`, built and validated) | 3,289 | free | Register the team, upload, read the per-category MRA. Do this before any GPU spend — it is the floor everything else must beat. |
| 1b | **Fresh detection pass for the new phrases** | ~183 | cents | The pair fix renames phrases, so those rows are cache misses. Needed before the separation numbers can be scored at scale. |
| 1c | `distance-twin` — "between the **two cars**", one phrase twice | **44** | free-ish | Declines by design today. Needs the detector to keep its **top-2** boxes per frame, not just the best; `DetectionSeries` holds one. Small, well-defined change to `grounding.py`. |
| 2 | **Diagnose the velocity rows** — median ratio 0.094, four of six near zero | ~1,000 | free | Biggest category and the worst performing. Replay the cache; the quadratic fit or the `fps`/timestamp path is suspect. No GPU needed. |
| 3 | Kill the fatal overshoots: gate on prior confidence | — | free | The `pedestrian walking` prior scored 0.341 with box width jittering 13–59 px and produced 7x overshoots. `min_confidence` already exists and is unused (`solve_row` defaults it to 0.0). Now that `detection_rate` is fixed, confidence is finally meaningful. |
| 4 | Decide on `fix/prior-grounding-phrase` | 412 | free | Real defects, 114 tests green, but it did **not** move the metric. Merge on correctness grounds, not on a score claim. |
| 5 | Full 159-row validation run on `l4x1` | — | ~1–3 h, $5–15 | Only after 1–3. Running it now would measure known-broken behaviour. |
| 6 | Fallback arm (VLM estimate), fused in log space | 325 | ~$5 GPU | More urgent than before: 325 rows now decline by design. |
| 7 | SAM2 masks / CoTracker3 | — | ~$10 GPU | **Deprioritised twice over.** The billiard box is already tight and correct; masks would not have helped. |
| 7b | Depth-proxy for rows whose target has no `depth_info` entry | ~60 | free | **Do not oversell this one** — see "The depth hypothesis is also refuted" below. Worth ~60 tail rows, not the 762 it first appeared to be. |
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
| Tests | **94 on `main`, 129 on the branch** (`py -3.12 -m pytest tests/ -q`) |
| Plan | `~/.claude/plans/snappy-launching-candy.md` |
| Track | **B (Open-Weight)** primary, A secondary |
| Deadline | **Plan for Oct 1, 2026** (site advertises Nov 5, but its own timeline finalizes rankings mid-October) |

**Built:** `scoring.py` (MRA + paired bootstrap), `units.py`, `parsing.py`, `geometry.py`,
`vision.py` (backend Protocol), `backends/grounding.py` (Grounding-DINO), `solver.py`,
`scripts/run_vision_job.py` (HF Jobs, checkpoints + resumes), `scripts/make_submission.py`,
`scripts/make_baseline.py` (zero-vision floor),
`scripts/validate_submission.py`, `notebooks/colab_vision.ipynb`, plus
`scripts/replay_cache.py` **on the branch only**.

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
* `leaderboard.html`, `submit.html`, `upload.html`, `rules.html` all still 404 — **but those were
  never the real paths.** See "Submitting": the portal lives under `competition/auth/` and works.

Re-verified again **2026-08-22**: template still byte-identical (739,113 bytes, same SHA-256, same
`Last-Modified`). Deadline on the site is **Nov 5, 2026 23:59 AOE**; the timeline's "final
submission phase" is early October, so keep planning for October.

Consequence, **corrected**: we have a real feedback signal now — 3 scored submissions per UTC day
against the hidden 3,289-row test set, with a per-category breakdown. The earlier conclusion ("no
external feedback until roughly October") was wrong, and it was wrong because the check probed
guessed URLs instead of reading the page's own nav links. The 3-per-day quota is now the main
budget to manage.

**Still unanswered** (parked until ~September, none of it blocking): which deadline is
authoritative, fine-tuning / external data / ensemble rules, whether gated weights count as
open-weight, and team eligibility across tracks.

## Submitting

**The portal is live and scores on upload.** Found 2026-08-22. The earlier "no submission page"
conclusion came from probing invented paths (`submit.html`, `upload.html`, `leaderboard.html` — all
still 404). The real ones are under `competition/auth/`, linked from the page's own nav:

| | |
|---|---|
| Register | `https://quantiphy.stanford.edu/competition/auth/register.html` (team name + ≤5 members) |
| Log in | `.../auth/login.html` |
| Upload + score | `.../auth/account.html` |
| Backend | Supabase (configured, real project), in-DB RPC `score_submission` |
| Returns | macro MRA, **per-category S2/D2/S3/D3 bars**, invalid rate, counts — immediately |
| Quota | **3 scored submissions per UTC day**; only `pass` submissions count. Resets 00:00 UTC |
| Limits | fill `parsed_value`, leave `id` unchanged, ≤ 2 MB |
| Not live | `leaderboard.html` — our score is visible, our *rank* is not |

This replaces the 159-row validation set (±5.7 pt CI) as the primary instrument. Use it.

Blank cells are sent as *missing*, not zero — the client skips them. Whether the server denominates
over all 3,289 rows or only rows submitted is unknown, and we never need to find out: always submit
all 3,289 filled, which `make_submission.py` guarantees via fallbacks.

```bash
py -3.12 scripts/make_baseline.py --out baseline_predictions.csv   # zero-vision floor, no GPU
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
* Session of 2026-08-22: **$0 of compute** — the distance fix, the baseline submission, and the
  portal discovery were all offline. Two lessons. **Read the site's own nav links instead of
  guessing URLs**: three sessions concluded "no submission portal" from 404s on invented paths while
  `auth/register.html` sat linked on the page, costing ~3 weeks of available feedback signal. And
  **size a lever before committing to it** — the depth hypothesis looked like half the metric and
  measured out at ~60 rows, in one free query against rows where it already worked.
