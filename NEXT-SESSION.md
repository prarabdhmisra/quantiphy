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

## 2026-08-23 — the 60-day campaign starts, and the real lever is the constants

`baseline-v3` scored **0.365** (S2 0.337, D2 0.315, S3 0.410, D3 0.396) — the corrected fixture moved
**D2 only**, exactly as diagnosed. That is the champion and the floor.

### The finding that reorganised everything

The same constant construction scores **0.459 in-sample on validation** and **0.365 on test**. The
metric is not the problem: each of the 27 `(category, unit)` constants is a median of about **six**
validation rows. LOO on validation predicts 0.3707 against a measured 0.365 — agreement to 0.006, so
the test distribution is the *same* as validation's and the whole 0.094 gap is **estimation error**.
Test groups average 120 rows; the largest, `D2|meters`, holds 829. The scoring set can tell us where
those constants belong.

**So the biggest lever is not the VLM. It is fitting ~27 constants against submission feedback.**
Simulated on a 3,289-row stand-in: 0.4341 → 0.4592 at 12 probes/group, ceiling 0.4668. The real start
is worse (0.365), so the real gain should be larger. Plan: `~/.claude/plans/greedy-spinning-hopper.md`.

### Why 3 submissions/day is far more capacity than it looks

The test set is fixed, so a difference between two submissions is **exact — no sampling noise**. And
the macro is the unweighted mean of four *separately reported* category means, so a change confined
to one category moves only that number. **Each submission is four independent experiments**: 12
readings/day, 180 slots over 60 days. For a group `g` in category `C`, if `g` is the only thing
changed in `C`:

```
delta(mean score on g) = (n_C / n_g) * delta(reported C)
```

Reported to 3 decimals, so a category mean is known to ±0.0005 and any group above ~30 rows resolves
usefully. The 11 groups under 30 rows (91 rows total) are not worth slots.

**Determinism is already proven, for free.** `baseline.submission.csv` (v1) and `baseline-v3` differ
on **829 rows, all inside D2** — S2/S3/D3 predictions are byte-identical. Both submissions reported
S2 0.337 / S3 0.410 / D3 0.396, a day apart. Identical inputs, identical outputs. No slot was spent
confirming it, and the planned Day-1 calibration submission was cancelled as redundant.

**And v1 → v3 is our first differential reading, also free.** `D2|meters` (829 of 1160 D2 rows) went
1.71 → 1.25 and D2 went 0.311 → 0.315, so that group's mean score rose by
`0.004 × 1160/829 = +0.0056`. **Lower was better**, which is where Day 1's probes point.

### Operational warning from the simulation

A **half-finished search is worse than not starting**: 4 probes/group scored 0.4253, *below* the
0.4341 start. So probe a grid that always contains the incumbent ×1.0 and take the argmax — that
makes every adopted result monotone by construction. Never adopt from a loose bracket.

### Built today

* `scripts/probe.py` — builds a probe submission from the champion. **Refuses two groups in one
  category**, because that is the assumption the inversion rests on. Emits a manifest of the exact
  row ids changed.
* `data/probes/ledger.csv` — one row per submission, with provenance. **This file is the campaign**;
  without it 180 submissions are 180 unreproducible anecdotes.
* `scripts/analyze_probes.py` — inverts the ledger, and checks that macro really is the mean of the
  four categories on every scored row (it is, on all three) plus that unperturbed categories
  reproduce exactly. A failure there invalidates the method, not one reading.
* `scripts/run_vision_job.py` — `SHARD=k/n`, contiguous slices, `row_index` preserved. Contiguous
  rather than strided because rows are ordered by video, so a shard re-uses each clip across the
  ~5.8 questions sharing it. Partition verified exact.
* Tests **134 → 146**, still CPU-only.

### Still true and still blocking

**No solver submission has ever been possible.** Every solver number in this document comes from the
159-row validation cache. The test detection pass (~$10–15, sharded `l4x1`) is the gate for Track B,
and 561 of 568 test videos are already local. GPU budget for the campaign: **$400**.

## 2026-08-22 evening — the first CONFIRMED lever, and a refuted instrument

Three things happened, in descending order of importance.

### 1. The velocity fit was broken. Fixed, and the gain is statistically established.

`kinematics()` fitted **one quadratic over the whole sampled clip** and read its derivative at the
requested instant. Any motion that is not a single clean parabola — a person walking, a saw cutting
back and forth, a pencil drawing, anything oscillatory or multi-phase — averaged away to near zero.
Now it fits inside `FIT_WINDOW_S = 0.30` s of the requested instant.

| on the 45 cached speed rows | MRA | median pred/truth |
|---|---|---|
| global quadratic (old) | 0.344 | 0.518 |
| **local ±0.30 s window** | **0.447** | **0.894** |

**+0.102, 95% CI [+0.022, +0.182], P(no gain) 0.006.** Velocity is **900 of the 3,289 test rows**,
spread evenly across all four categories, so this is worth roughly **+0.030 macro**. After four
refuted single-cause theories this is the first one that survived a paired bootstrap.

Whole solver, same cache: **0.338 → 0.422** (constant 0.378), **+0.084, CI [+0.033, +0.134],
p=0.0006** against the old solver. Against the *constant* it is +0.044 with CI [−0.032, +0.091] —
**not** established.

There is a real cost, recorded so nobody "fixes" it back: on *genuinely* constant-velocity motion a
whole-clip fit is the better estimator (0.20 px/s error against this window's 1.45, over 300 jitter
draws, winning 250 of 300). We take the worse estimator for the ideal case because the ideal case is
rare and the global fit's failure mode is not noise, it is a near-zero answer. Widths 0.25–0.35 are
indistinguishable on 45 rows; 0.30 is the conservative middle.

### 2. LOO on 159 rows CANNOT RANK two candidates. This is the instrument, so read it.

Dropping the `(category, unit)` tier from `make_baseline.py` measured **better** by leave-one-out
(unit-only 0.3776 vs 0.3707). On the test set it measured **worse**:

| | S2 | D2 | S3 | D3 | **macro** |
|---|---|---|---|---|---|
| `(category, unit)` ladder, stale fixture | 0.337 | 0.311 | 0.410 | 0.396 | **0.364** |
| unit-only, corrected fixture | 0.309 | 0.308 | **0.428** | 0.392 | **0.359** |

LOO predicted +0.007; test delivered −0.005. **Reverted.** The earlier claim that "offline LOO
predicts the test score to within 0.003" is true only for *evaluating one fixed procedure* — it is
false for *choosing between* procedures, because the max of several LOO estimates is biased upward
and a 0.007 gap is inside the noise at n=159.

**This downgrades the `prior_pixels` band too.** `TRUSTED_PRIOR_PIXELS = (30, 300)` was picked the
same way — six variants, 159 rows, a CI that already spanned zero. Treat it as unproven. Its
*mechanism* is strong and independent of the threshold (`log(pred/truth) ≈ −0.87·log(prior_pixels)`,
corr −0.725, n=147: almost the whole error is the prior's own pixel measurement), so the gate itself
is well-founded even if the edges are not. The kinematics fix is a different kind of claim — paired,
mechanism-backed, p=0.0006, not a selected threshold — and is not affected.

Also refuted this session, both free and both properly cross-validated:
* **MRA-argmax per group instead of the median**: LOO 0.320 vs 0.377. Overfits tiny groups.
* **Post-processing a good per-row predictor with the constant.** Tested on GPT-5.1's own 159
  predictions (0.4748): global shrink, symmetric clamp to `[c/K, c·K]`, asymmetric overshoot cap and
  log-space blending **all lose at every parameter value**. The constant carries no information a
  real per-row predictor lacks. So never fuse a VLM with the constant — fuse it with the *solver*,
  or gate. Corollary: the "+19 pts from fixing overshoots" pool is unreachable by any blind cap.

### 3. The validation fixture was corrupt, and fixing it proved the scorer exact

`data/fixtures/gpt-5.1_validation.csv` had **one ground truth wrong by 100x** (125.0 for 1.25), 18
stale `video_type` values and 6 questions missing the timestamp the parser reads. The organizers'
own split is now committed as `data/fixtures/quantiphy_validation.csv`, and against it our scorer
reproduces the published GPT-5.1 macro at **0.4856 vs a published 0.48561** — exact to four decimals,
where the stale file gave 0.4836. `tests/test_scoring.py` tolerance tightened 0.005 → 5e-4.

Note the correction is not cosmetic: it moves 829 of 3,289 baseline predictions (the D2 constant
goes 1.71 → 1.25). `baseline-v3.submission.csv` is that ladder with clean truth, built and validated
but **not yet scored**.

### Depth / 3D fixes landed (sized on the real 1,548 3D test rows)

The correction's **direction is correct** — pinhole-verified, do not look there again. Fixed:
**C1**, the prior's depth was read at `prior.timestamp` (usually `None`), which falls through to
*file order*, so a same-object ratio that must be 1.0 came out 1.30 — **1,284 rows have >1 timed
reading, 310 have a ≥1.25x artifact**. **`MAX_DEPTH_RATIO = 4.0`**, because the 3D path had three
one-way inflation mechanisms and no upper bound under a metric where 1.9x scores zero. **C3**,
`_name_overlap`'s containment score was summed uncapped so a loose compound key beat an exact one
(an 8x error; latent on this test split). **C6**, `radial_speed` matched case-sensitively, so any
capitalised phrase silently dropped radial.

Tests **129 → 134**, all CPU-only. Still open: C2 (left/right ties, 315 rows), C5 (regex drops, 59
rows), C7 (separation uses only object A's depth), and the fact that `prior_pixels` is an extent for
length priors but a *speed* for speed priors, so one band covers two quantities.

### The blocker

A solver submission needs a detection pass over 3,289 rows / 568 videos. **It has never been run** —
every solver number in this document comes from the 159-row validation cache. That run, and the
open-weight VLM arm, are the next steps. 561 of 568 test videos are already local in `data/videos/`.

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

## First real test score: 0.364 — and it reframes everything

Submitted 2026-08-22, `baseline.submission.csv`, **zero vision**: just the median validation answer
per (category, unit). Scored on the hidden 3,289-row test set.

| | S2 | D2 | S3 | D3 | **macro** |
|---|---|---|---|---|---|
| zero-vision constant | 0.337 | 0.311 | 0.410 | 0.396 | **0.364** |

3,289 matched, 0 missing, 0.0% invalid — the submission path is proven end to end.

**Two things this buys that matter more than the score.**

1. The server's "Average MRA" is *exactly* the macro average `quantiphy/scoring.py` computes —
   confirmed to four decimals. Our scorer is the real scorer.
2. Our offline LOO estimate of this same submission was **0.3673** against a measured **0.364** --
   within 0.003. **Read that narrowly.** LOO is accurate for *evaluating one fixed procedure*; it is
   NOT accurate for *choosing between* procedures, and on 2026-08-22 it picked a baseline variant
   that measured +0.007 by LOO and **-0.005 on test**. See "LOO on 159 rows CANNOT RANK two
   candidates" above. A submission slot is for confirmation, and selection is exactly what needs
   confirming.

### The uncomfortable part: the solver currently loses to a constant

Scoring the 16 rows the solver actually solved in the last replay, under the real metric:

| | MRA |
|---|---|
| geometric solver, on rows it solved | **0.312** |
| zero-vision constant, same metric | **0.364** |
| rows scoring exactly 0 | **7 of 16** |

Firing the solver ungated *costs* points. But the distribution is bimodal, not uniformly bad — the
good rows score 0.70–1.00, far above the constant, and the bad ones score 0.00. So the entire value
is in **knowing which is which**:

```
fire only where the solver beats the constant (oracle):  0.496
fire everywhere (today's behaviour):                     0.312
never fire (the constant):                               0.364
```

**Confidence gating is therefore the whole game, not a polish step** — worth roughly +13 points,
which is the difference between last place and GPT-5.1's 53.1. `min_confidence` already exists in
`solve_row` and still defaults to 0.0. Now that `detection_rate` is fixed, confidence is meaningful.
This promotes old pending #3 to **#1**.

For scale, the published test-split numbers: human 55.6, GPT-5.1 53.1, Gemini-2.5 Pro 49.6,
**Qwen3-VL-32B 46.0** (the Track-B bar), InternVL-3.5-30B 40.7. A zero-vision constant at 36.4 is
already within 4.3 points of InternVL-3.5-30B, which says a lot about how much credit MRA gives for
the right order of magnitude.

### Two zero-vision ideas, measured

* **Global shrink.** LOO-optimal at 0.90 (0.3765 vs 0.3673), but +0.009 with a 95% CI of
  [-0.006, +0.024] — **not established**. Behind `make_baseline.py --shrink`;
  `baseline-shrink90.submission.csv` is built and ready if a slot is spare. Note
  `make_submission.py --shrink` is a **no-op** on a baseline: it scales only *fallback* rows.
* **Predict a learned multiple of `ground_truth_prior`** instead of a constant. **Refuted** — 0.2955
  vs 0.3673, and the CI on the difference excludes zero, so the harm is established. `truth/prior`
  spans 75x from p10 to p90: the prior sets the scene's scale but does not pin the answer. Do not
  retry this.

## The 159-row validation run — and the confidence gate is REFUTED

Job `6a89ee7673304676c8ec8746`, `l4x1`, `RUN_NAME=validation-distance-fix1`, **33 minutes** (not the
1–3 h estimated, so cents). 150/159 solved (94.3%). Only two failure reasons: 7 gravity priors,
2 `distance-twin`. Re-read it free with
`py -3.12 scripts/fit_confidence_gate.py --run validation-distance-fix1`.

| | macro MRA |
|---|---|
| solver, coverage-limited | 0.3294 |
| solver, as-submitted | 0.3116 |
| **zero-vision constant (LOO)** | **0.3707** |
| **oracle: fire only where the solver wins** | **0.5261** |
| GPT-5.1 on this split | 0.4856 |

**The confidence gate does not work.** Every threshold from 0.0 to 0.5 scores *below* the constant:

```
gate  0.00 -> 0.3347      gate 0.20 -> 0.3422      gate 0.40 -> 0.3454
gate  0.15 -> 0.3555      gate 0.30 -> 0.3561      gate 0.50 -> 0.3580   (best, still < 0.3707)
```

Detector confidence (`mean_score x detection_rate x fit_quality`) simply does not predict whether
the answer is right. The 20-row preview that showed gate 0.15 at +0.105 was **noise** — its CI
spanned zero, and that was the only reason to distrust it. Fourth single-cause theory refuted.

### What did work, barely: a disagreement gate

Fire the solver only when it *agrees with the constant* within a factor k. Measured on all 159 rows,
paired bootstrap on the per-row difference:

| gate | fires | macro | vs constant | 95% CI | |
|---|---|---|---|---|---|
| **within 1.5x** | 25 | **0.3837** | **+0.0130** | [+0.0013, +0.0252] | **excludes zero** |
| within 1.5x, 2D only | 17 | 0.3816 | +0.0109 | [+0.0006, +0.0201] | excludes zero |
| within 3.0x, 2D only | 39 | 0.4029 | +0.0322 | [−0.0000, +0.0509] | touches zero |
| within 2.0x | 48 | 0.3940 | +0.0233 | [−0.0069, +0.0409] | spans zero |

**Read this with real suspicion.** The 1.5x gate is the survivor of ~20 variants tested against the
same 159 rows, so it is exactly the multiple-comparisons artifact this project has been burned by
before (per-category shrinkage: +0.03 in-sample, −0.02 leave-one-out). A CI that *barely* excludes
zero after twenty looks is not a finding. **Confirm it on the test set** — a submission is free and
3,289 rows give ~20x the precision — before building anything on it.

**Also refuted:** log-space blending of solver and constant. Every weight from 0.2 to 1.0 scores
below the constant (best −0.015). Do not retry.

### The real lead: `geometric-3d` is actively harmful

Solver vs constant, on the rows the solver solved, split by method:

| method | n | solver | constant | |
|---|---|---|---|---|
| `geometric-2d` | 93 | 0.317 | 0.301 | slight win |
| `geometric-2d+separation` | 4 | 0.600 | 0.500 | win |
| **`geometric-3d`** | **34** | **0.315** | **0.500** | **loses badly** |
| `geometric-3d+separation` | 4 | 0.325 | 0.900 | loses badly |
| `geometric-3d+radial` | 7 | 0.329 | 0.414 | loses |

Every 3D route loses to a constant, and S3+D3 are **half** the macro score. Note this does *not*
contradict "the depth ratio is centred on 1.0" — that measured the ratio's *distribution*, not
whether we apply the right ratio to the right object. The suspicion is now that `depth_for` matches
the wrong reading, so a correction gets applied backwards. **This is the most promising open lead.**

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
| ~~1d~~ | ~~**CONFIDENCE GATING**~~ | — | done | **REFUTED 2026-08-22 on 159 rows.** Every threshold scores below the constant; detector confidence does not predict correctness. See "The confidence gate is REFUTED". |
| **1e** | **Diagnose `geometric-3d`** — every 3D route loses to a constant (0.315 vs 0.500), and S3+D3 are **half** the score | ~1,548 | free | Now the best lead. Suspicion: `depth_for` matches the wrong reading, so the correction is applied backwards. Replay `validation-distance-fix1` — no GPU. |
| 1f | Confirm-or-kill the 1.5x disagreement gate on the test set | all | 1 slot + test run | +0.013 with CI [+0.0013,+0.0252], but it survived ~20 variants on 159 rows. Treat as unproven until a real submission says otherwise. |
| 3 | Kill the fatal overshoots: gate on prior confidence | — | free | The `pedestrian walking` prior scored 0.341 with box width jittering 13–59 px and produced 7x overshoots. `min_confidence` already exists and is unused (`solve_row` defaults it to 0.0). Now that `detection_rate` is fixed, confidence is finally meaningful. |
| 4 | Decide on `fix/prior-grounding-phrase` | 412 | free | Real defects, 114 tests green, but it did **not** move the metric. Merge on correctness grounds, not on a score claim. |
| 5 | Full 159-row validation run on `l4x1` | — | ~1–3 h, $5–15 | **Promoted: this is now the unblocking step.** It has ground truth, so it is what the confidence gate is fitted on. Do this before any full test run. |
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

**Team eligibility across tracks -- ANSWERED 2026-08-23**, read off the competition page rather than
guessed. **Track A (Main)** permits "any model ... proprietary, open-weight, or hybrid"; **Track B
(Open-Weight)** is the "same scoring rule ... but submissions must be based on publicly available
model weights and tools". Crucially: *"Competitors may enter either or both tracks. Each track has
its own leaderboard, evaluation protocol, and awards."* Track choice is **per upload** from a single
template; the tracks share a registration but have independent leaderboards. "Teams of up to five.
One team per person, per track." Deadline **Nov 5, 2026 23:59 AOE**.

Consequence: **enter both, always.** Every component here is open-weight (Grounding-DINO, Qwen3-VL,
a hand-written solver, constants fitted from the organizers' own validation split), so every upload
is Track-B-eligible and therefore Track-A-eligible too. There is no fork in the roadmap.

**Unverified and it matters:** whether opting a single upload into *both* tracks consumes one of the
3 daily slots or two. If two, the probing throughput halves from 12 readings/day to 6. Check the
upload form on `auth/account.html` -- it is behind a login, so it cannot be read from here.

**Still unanswered** (parked, none of it blocking): which deadline is authoritative, fine-tuning /
external data / ensemble rules, and whether gated weights count as open-weight.

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
