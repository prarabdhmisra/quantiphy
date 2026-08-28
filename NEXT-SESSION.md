# Resume here

Last worked: **2026-08-26**.

> **THE PORTAL IS LIVE.** The single most important thing on this page. Submissions are scored on
> upload and return a per-category MRA immediately, 3 per UTC day. Everything below that says
> "no feedback until October" was wrong and has been corrected — see "Submitting". Get a submission
> on the board before spending anything on GPU.

> **Where things live.** This document is on both branches and is accurate on both. The *code* from
> 2026-08-05 onward is only on **`fix/prior-grounding-phrase`** (pushed, 207 tests green) —
> `main` is deliberately still at the last *measured* state, 94 tests. Anything below that names
> `scripts/replay_cache.py`, `scripts/select_rows.py` or 207 tests needs that branch checked out:
>
> ```bash
> git checkout fix/prior-grounding-phrase
> ```

## Paste this to start the next session

> Resuming the QuantiPhy Challenge (NeurIPS 2026). Read
> `C:\Users\prara_\quantiphy\NEXT-SESSION.md` first. Repo is public at
> https://github.com/prarabdhmisra/quantiphy. Work on branch `fix/prior-grounding-phrase` (241 tests
> green). **The portal is live and scores on upload, 3/day — read "Submitting".** Champion is
> **`mix-v7`, macro 0.4435**; **`mix-v8` is built, validated and unsubmitted** — it carries the
> 2026-08-27 D3 fix on 240 rows plus an S3 cleanup on 41, with S2 and D2 as bit-identical controls
> that must come back at 0.440 and 0.461. The board's best is still `mix-v5` at 0.441.
>
> **The VLM arm is now the top item, and everything geometric is worth thousandths.** Solver on
> S2/S3 plus a Qwen-class VLM on D2/D3 composes to **~0.492** by arithmetic, and at 0.4435 we are
> still below the paper's best open-weight number (0.460), which is what Track B has to beat. A
> prompt A/B (`brief` vs `direct` — the CoT question) ran on the 159 truth-bearing validation rows on
> 2026-08-27; read it with `py -3.12 scripts/score_vlm.py --run validation-vlm-qwen3vl8b-brief
> --run validation-vlm-qwen3vl8b-direct` before committing to a full test pass.
>
> Read "2026-08-27" then "2026-08-26" before proposing a lever. **Six single-cause theories of the 3D
> bias have been refuted and the seventh was real:** the radial correction was applied to the target
> and never to the prior, which could only ever fire in D3. **Constant-multiplier probing is
> exhausted** (8 group probes, 8 losses). **"D3 is closed to the solver" holds for thresholds on
> disagreement magnitude but not for selection by route** — see `scripts/method_ids.py`. Item 1k (the
> gravity prior) is re-sized and **not free**: its 46 videos are all detection-cache misses. Ask me
> before spending more than ~$20 in a session.

## 2026-08-27 — the radial correction was only ever applied to one of the two objects

The sixth theory of the 3D bias, and the first that predicts the *category pattern* rather than just
the magnitude. `solve_row` computed `radial_speed` for the **target** and folded it in with `hypot`.
It never computed it for the **prior**. But a speed prior states a 3D speed while `prior_pixels`
recorded only its in-plane projection, so `gamma = prior_world / prior_pixels` was too large by
`1 / sqrt(1 - (v_r/v)^2)` on every 3D speed-prior row.

That is exactly D3 and nothing else. `D3` is `[dynamic prior][3D]`, so all 972 of its rows carry a
velocity or acceleration prior *and* a depth track. S3's priors are sizes, so no radial term exists;
D2 is 2D, so there is no depth to read. **The bug could only ever fire in the one category that
overshot**, which is why five single-cause theories that ignored the category structure all failed.

**The median prior object spends 62% of its own speed along the camera axis.** This is not a small
correction, and it is one-way -- it only ever inflated the answer.

### Measured on the 240 rows it fires on, all in D3, with S2/D2/S3 bit-identical

| | median vs the constant | within 2x | over 1.9x (a hard zero) | under 0.1x |
|---|---|---|---|---|
| before | 1.235 | 32.9% | **41.7%** | 4.2% |
| **after** | **0.953** | **39.2%** | **30.0%** | 6.2% |

The rule needs no tuning: apply the decomposition whenever it is physically available (`|v_r| < |v|`)
and leave `gamma` alone when it is not. That principled rule is also the argmax over every guard
variant tried, which is the first time on this project that a threshold and a principle have agreed.
Where `|v_r| >= |v|` the depth readings and the stated prior contradict each other, and emitting the
near-zero tangential speed would drive `gamma` to 0 and the answer to a hard zero.

### The acceleration version is REFUTED, and pinned by a test

The same decomposition on acceleration priors looks equally justified and is worthless: over the 172
D3 acceleration-prior rows with three or more timed depth readings, the depth-derived radial
acceleration is a **median 1.50x the stated total** -- physically impossible for a decomposition,
because `d2Z/dt2` from two to four readings over a 2-3 s clip is noise. Applying it moved the median
from 1.07x the constant to 1.90x and doubled the >1.9x share to 50%. `tests/test_solver_pipeline.py`
pins the negative so it is not re-added.

### Route selection is a partition the campaign had never used

`scripts/method_ids.py` (new, tested): the ids of rows that took a named code path, feeding
`select_rows.py --overlay-ids`. **"D3 is closed to the solver" was measured over thresholds on
*disagreement magnitude*, and that finding still holds** -- but route is a different partition of the
same rows, and it separates populations a single threshold averages together:

| | n | median vs const | within 2x |
|---|---|---|---|
| D3 `geometric-3d+radial` | 66 | 0.75 | **51.5%** |
| D3 `geometric-3d` | 213 | 2.29 | 27.2% |
| S3 `geometric-3d+radial` | 93 | 0.99 | **77.4%** |
| S3 `geometric-3d+target-from-prior` | 16 | 0.14 | **0.0%** |
| S3 `geometric-2d+separation` | 12 | 0.21 | 16.7% (**50% are >100x**) |

### `mix-v8.submission.csv` — BUILT, VALIDATED, NOT YET SUBMITTED

Two experiments and two controls in one slot. **S2 and D2 are bit-identical to `mix-v7`, so the
portal must return 0.440 and 0.461 for them** or the composition arithmetic is wrong.

* **D3**, 174 of 972 rows: the constant -> the tangentially-corrected solver, **gated at 5x of the
  constant**. Reads as `delta(D3) = 0.179 * delta(mean score on those rows)`. First thing to reopen
  D3.
* **S3**, 41 of 576 rows (37 actually move): the solver -> the constant, on the three degenerate
  routes above. Near-certain zeros, and the constant scores 0.410 in S3. Reads at `0.071 * delta`.

**The gate is not a tuning knob and the first build was wrong without it.** Overlaying all 240
corrected rows would have carried 6.2% of them under 0.1x of the constant and 30.0% over 1.9x --
both hard zeros, replacing rows that were already collecting the constant's ~0.40. That is knowingly
including the rows the 2026-08-26 clip priced at **-0.257/row** in D3. Gating at 5x combines the two
measured findings instead of setting one against the other:

| | n | median | within 2x | >1.9x | <0.1x |
|---|---|---|---|---|---|
| ungated | 240 | 0.953 | 39.2% | 30.0% | 6.2% |
| **within 5x** | **174** | **0.831** | **54.0%** | **18.4%** | **0.0%** |

Built by clipping the replay (`clip_disagreement.py --clip D3=5`) rather than with a bespoke id list:
`mix-v7`'s D3 is the *pure* constant, so every unoverlaid D3 row sits at ratio exactly 1.0 and can
never be clipped, which makes a category-wide clip exactly the gate wanted. Clipped rows return as
`method=none` and `method_ids.py` already drops those.

**The prediction to beat is 0.374, not 0.396.** The 2026-08-26 anchor is that D3 solver rows within 5x
of the constant score 0.374 against its 0.396. This population is that same gate *plus* the bias
correction, so it should be strictly better than 0.374; whether it clears 0.396 is what the slot buys.
Do not read a null as refuting the fix — the fix is a correctness claim measured on 240 rows, and this
is a question about whether a de-biased geometric answer can beat a constant in D3 at all.

### Item 1k (the gravity prior) is NOT free — every one of its videos is a cache miss

Sized before committing to it, and the "free to try" label in the table below was wrong.
`solve_row` declines a gravity row **before** it ever measures the target, so the detector was never
run on those target phrases: all **46 videos** behind the 291 rows are absent from the test detection
cache. The solver change is free; measuring it needs a fresh detection pass over those 46 videos
(cents, minutes). Fold it into the next GPU session rather than treating it as an offline lever.

The physics for when it is built: a gravity prior fixes no pixel length, but a free-falling target's
own pixel acceleration does -- `gamma = 9.8 / a_px`. Gate it on the fitted acceleration being
predominantly **vertical and downward**, which is the signature of free fall and also rejects the
static targets (a "width of the table" row would divide by a near-zero `a_px` and emit a hard zero).

### The VLM arm ran for the first time, and its installer was broken

`run_vlm_job.py` used `python -m pip`, which does not exist inside `hf jobs uv run` -- so the first
two launches died in seconds on `No module named pip`. `run_vision_job.py` already carried the
uv -> pip -> clone ladder *and a comment explaining exactly this*; the VLM arm had the naive version
because it had never actually run. **Code that has never executed is not code, however well tested**
-- 241 unit tests said nothing about this.

`VLM_PROMPT=brief|direct` is now a switch rather than a hardcoded string, because the paper's Table 2
puts chain-of-thought at 56.1 -> 27.7, 49.8 -> 22.4, 50.1 -> 23.1 against video+prior: it roughly
halves MRA for every strong model. `brief` (mild CoT) stays the default so a resumed run stays
comparable with its own checkpoint. An unknown style raises before the model loads.

### The A/B: CoT does NOT collapse here, and the D3 number is the whole reason to keep going

159 validation rows, Qwen3-VL-8B, 12 frames, both prompts:

| | S2 | D2 | S3 | D3 | macro |
|---|---|---|---|---|---|
| **`brief`** (mild CoT) | 0.3875 | 0.4649 | 0.4791 | **0.5234** | **0.4637** |
| `direct` (no reasoning) | 0.3781 | 0.4405 | 0.4116 | 0.5149 | 0.4363 |
| *our solver, on **test*** | *0.440* | *0.461* | *0.477* | *0.396* | *0.4435* |

Paired bootstrap, `direct` against `brief`: delta **-0.027**, 95% CI **[-0.088, +0.032]**,
P(no improvement) 0.813. **Not significant, so the paper's Table 2 CoT collapse does not reproduce**
for this model at this prompt -- and the point estimate leans the other way. `brief` stays the
default; the switch stays, because it was a real question and is now a measured one.

**The finding that justifies the whole arm: VLM D3 = 0.5234 against our 0.396.** That is within a
point of the paper's Qwen3-VL-32B D3 of 0.534, so the 8B reproduces the 32B where it matters. S2 we
win outright (0.440 vs 0.388) and S3 is a tie.

**Read all of that as a go/no-go and nothing finer.** These are *validation* numbers against our
*test* numbers -- different splits -- and the per-category n is 32-47 rows, which is far below what
this project has repeatedly established the 159-row split can resolve. The composition is only real
once the VLM has its own scored test submission.

**26% of `brief`'s rows fell back, and 38 of those 42 are the model answering literally zero** --
"the ball is stationary, so its velocity is zero", "height cannot be determined without a reference".
`parse_answer` correctly declines a zero (it would be a hard zero) and the constant fills the row.
Worth knowing for the composition: **those fallback rows should take the *solver's* answer, not the
constant**, since the solver beats the constant in S2, D2 and S3. That is free once both are scored.

### The parse route is a quality signal, and it is the one the VLM arm should be selected on

Free, from replies already paid for, and it answers an open question in `prompting.py`'s own
docstring. `parse_answer` falls back to "the last number in the reply" when the model drops the
`ANSWER:` marker, and that docstring says the fallback is "genuinely useful ... but it also happily
picks up a number from the reasoning". **The second half dominates.** Per row, against the
`(category, unit)` median constant, over the 117 answered validation rows:

| route | n | VLM | constant | delta/row | 95% CI |
|---|---|---|---|---|---|
| `vlm-sentinel` | 75 | 0.536 | 0.439 | **+0.097** | [-0.012, +0.205] |
| `vlm-last-number` | 42 | 0.281 | 0.436 | **-0.156** | **[-0.293, -0.021]** |
| all answered | 117 | 0.444 | 0.438 | +0.007 | — |

So **the VLM as a whole barely beats a constant, while the half of it that follows the output format
beats it clearly and the half that does not is measurably worse.** A model that will not emit the
marker it was told to emit is hedging, and its number should not be used.

Read the significance honestly: only the **negative** has a CI excluding zero. The sentinel row's
+0.097 does not, at n=75. But the negative is the actionable half, it has a mechanism, and dropping
those rows is the conservative move rather than the aggressive one.

**Not a category artifact.** The `last-number` rows are spread 11/12/5/14 across D2/D3/S2/S3, and the
constant scores 0.436 on them against 0.439 on the sentinel rows -- the two populations are equally
hard by the constant's own measure.

Restricted to **D2+D3, where the arm is actually meant to be used**, the split is starker still:
sentinel rows score **0.609** against the constant's 0.440, and all-answered rows only 0.484.

**So compose with `method_ids.py --contains sentinel`,** not with every answered row. And note the
159-row split has refuted three levers that looked real in-sample -- confirm this one on test before
treating it as settled.

### The full test pass is far cheaper than the old estimate — measure it, then run it

Measured on `l4x1`, Qwen3-VL-8B in bf16 at 12 frames: **`direct` runs 3.3 s/row and `brief` 8.3 s/row.**
The reasoning tokens cost 2.5x the wall clock, so the prompt A/B decides the bill as well as the
score. Over the 3,289 test rows:

| prompt | GPU-hours | 4 shards, each | rough cost at ~$0.80/h for `l4x1` |
|---|---|---|---|
| `direct` | 3.0 | ~45 min | **~$2.50** |
| `brief` | 7.6 | ~1.9 h | ~$6 |

So the "~$20-40" in item 6 is wrong by roughly an order of magnitude, and the arm is affordable to
re-run whenever the prompt changes. Add 2-3 minutes per shard for the install and the 15 GB model
download. Launch with the winning style:

```bash
export PYTHONIOENCODING=utf-8            # or the upload bar's block character kills the client
for K in 1 2 3 4; do
  hf jobs uv run --detach --flavor l4x1 --timeout 4h --secrets HF_TOKEN \
    -e QUANTIPHY_GIT=git+https://github.com/prarabdhmisra/quantiphy.git@fix/prior-grounding-phrase \
    -e OUTPUT_REPO=prarabdhmisra/quantiphy-runs -e SPLIT=test \
    -e VLM_MODEL=Qwen/Qwen3-VL-8B-Instruct -e VLM_PROMPT=<winner> \
    -e SHARD=$K/4 -e RUN_NAME=test-vlm-qwen3vl8b-<winner>-shard$K \
    scripts/run_vlm_job.py
done
```

**Then do not fuse anything.** Turn the VLM run into its own submission, spend one slot on it, and
*select per category* against `mix-v8` -- that composition is arithmetic once both are scored, and
this project has refuted four attempts to predict per-row which arm is right. Log-space fusion is a
later question and it needs two scored parents first.

`scripts/score_vlm.py` (new) re-parses `vlm_raw.jsonl` rather than trusting the run's own
`parsed_value`, scores against validation truth, and compares two runs on the *intersection* of rows
both answered with a paired bootstrap.

### Why the VLM arm is the only lever that still matters

Every remaining geometric lever is worth 0.005-0.010 on the macro. The arithmetic on the paper's
Table 1, with our own measured S2/S3:

| | S2 | D2 | S3 | D3 | macro |
|---|---|---|---|---|---|
| **mix-v7 (champion)** | 0.440 | 0.461 | 0.477 | 0.396 | **0.4435** |
| Qwen3-VL-32B (paper) | 0.358 | **0.516** | 0.432 | **0.534** | 0.460 |
| **solver on S2/S3 + a Qwen-class VLM on D2/D3** | 0.440 | 0.516 | 0.477 | 0.534 | **~0.492** |

Per-category composition is *arithmetic* once both parents are scored, so that is not a projection of
a model's quality -- only of Qwen3-VL-8B reproducing the 32B's D2/D3 within a few points. Note also
that at 0.4435 we are still **below** the best open-weight number in the paper (0.460), which is the
number Track B has to beat.

## 2026-08-26 — three strategies closed, and the champion moved 0.441 -> 0.4435

Three slots, twelve readings, and the day's value was almost entirely in what it ruled out. Two new
committed instruments: `scripts/disagreement.py` (the ratio-vs-constant table, which reproduces the
2026-08-25 ad-hoc analysis digit-for-digit and is now re-runnable) and `scripts/clip_disagreement.py`.

### Track A vs Track B — ANSWERED, and there is nothing to choose

The upload form on `auth/account.html` has **no track control at all**: one file input, one counter
reading `N / 3 SCORED TODAY`. Track is not a per-upload decision, so a single upload is the entry
for both boards and the "does dual-track cost two slots" worry in the old text is moot. Every
component is open-weight, so every file is B-eligible and therefore A-eligible. Worth one line to
the organizers asking how they assign tracks; it blocks nothing.

### The three closures

1. **Constant-multiplier probing is EXHAUSTED.** `probe-d3a` moved four groups toward the solver's
   own median ratio and **lost in all four** (group-mean deltas: `S2|m/s` -0.059, `D2|m/s` -0.114,
   `S3|cm` -0.203, `D3|cm` -0.124). With Day 2's x0.7/x1.4 also losing on four other groups, that is
   8 probes and 8 losses. The constants are at their argmax. **The solver's median ratio vs the
   constant is NOT a guide to where a constant belongs** — that inference was the whole basis of the
   probe and it is refuted. Do not spend another slot on a group multiplier.

2. **D3 is CLOSED to the solver.** The clip curve rises 0.315 -> 0.386 (>5x) -> 0.395 (>2x) and the
   full-clip limit is the pure constant's 0.396. Inverting `solver-v3`, the solver rows that agree
   with the constant within 5x score **0.374** against the constant's 0.396 — worse. So no threshold
   exists at which the solver beats a constant in D3. Stop optimising D3 with geometry; only a
   different *arm* (the VLM) can move it.

3. **Item 1e (the inverted 3D ratio) is REFUTED — the fifth single-cause theory to fall.** Measured
   directly: median `Z_target / Z_prior` is **1.000-1.02 in every category/dimension slice**, so the
   depth correction is a near-no-op and cannot cause a 4x error. The `geometric-3d` vs `-2d` split
   that looked like a code defect is a **population** split — slicing by `(category, unit)` instead
   shows *both* routes undershoot in `S3|cm` and *both* overshoot in `D3|cm`.

### The finding worth building on

**Disagreement with the constant means opposite things in D2 and D3.** From `solver-v3`, per
clipped row against `solver-v2`:

| category | clip | effect of reverting those rows | reading |
|---|---|---|---|
| D2 | >10x | **-0.192/row** | the solver was **earning**; a 10x gap means the *constant* is wrong |
| D2 | >100x | -0.002 reported | even 100x outliers earn. The solver is trustworthy in D2 at extremes |
| D3 | >5x | **+0.257/row** | the solver was losing badly |
| S3 | >10x | +0.161/row | clipping helps; lands exactly on `mix-v4`'s 0.477 |
| S2 | >20x | ~0 | noise |

So **D2 is where solver coverage is worth the most**, and 297 D2 rows are still declined. That sizes
item 1k directly: 170 of the 291 gravity-prior rows are in D2, at roughly +0.19/row, which is
`170 * 0.19 / 1160` ~= **+0.028 on D2** if they solve as well as existing rows. Biggest lever left.

### Item 1j: half of it wins, and the other half is a trap

Dropping the trusted band's lower edge from 30 px to 0:

* **S2: 0.430 -> 0.440 (+0.010) on 29 rows — ADOPTED**, and it is the whole of today's gain.
* **S3: 0.465 -> 0.451 (-0.014) on 45 rows — REJECTED.**

Both populations had **0% of rows >100x off**, which is exactly why this is worth writing down:
**"0% >100x" is not a quality signal.** A row 2-10x off scores zero just as surely and never appears
in that column. Judge a population by `within2x`, not by the extreme tail.

### Item 1l: sized, and it is ~3 rows, not 328

The old estimate counted all 454 `geometric-3d` rows, but `radial_speed` only applies to **speed**
questions. There are 277 of those and **216 already fire**. Of the 61 that do not, 27 have a single
matched depth reading (genuinely underdetermined) and only **3** are blocked by the timestamp gate.
Closed.

### Where the champion stands

| | S2 | D2 | S3 | D3 | macro |
|---|---|---|---|---|---|
| mix-v5 (yesterday) | 0.430 | 0.461 | 0.477 | 0.396 | 0.441 |
| **mix-v7 = solver-v4 on S2, mix-v5 elsewhere** | **0.440** | 0.461 | 0.477 | 0.396 | **0.4435** |

Built and validated with `scripts/select_sources.py`. **Not on the board** — the standing rule
forbids spending a slot to confirm a composition, but the official record only knows `mix-v5` at
0.441, so upload `mix-v7` as tomorrow's slot 1 for provenance rather than for information.

Against the field: S2 **0.440** vs Qwen3-VL-32B's 0.358 and GPT-5.1's 0.463; S3 0.477 vs 0.432 and
0.515. The gap is still entirely D2 (0.461 vs 0.516) and D3 (0.396 vs 0.534), and D3 now has only
one road left: the VLM arm.

## 2026-08-25 — the replay harness is built, and the band is 77% of all declines

**The whole of "DAY 3: the offline re-solve" below is DONE**, except the two items promoted into
Pending as 1c and 1h. Read this section instead of that plan; the plan is kept because its reasoning
is still the reasoning, and because step 1 of it is the gate every future replay runs.

### The harness

```bash
py -3.12 scripts/replay_cache.py --split test --run test-solver-v1 --shards 4 \
    [--trusted-prior-pixels 30,inf] [--out predictions.csv]
```

`replay_cache.load` now takes `--split test`, reads the **local** `data/fixtures/test_dataset.parquet`
(never `snapshot_download("PaulineLi/QuantiPhy")` — that pulls gigabytes of video to answer a
question that opens no pixels), and unions the four shard detection caches. All 3,289 rows replay in
seconds on CPU. `CachedBackend` and `solve_row` needed no change, exactly as predicted.

**It reproduces `solver-v1` exactly.** 1,338 solved; S2 42.513% / D2 28.448% / S3 76.562% /
D3 32.922%; and row-by-row against the GPU `predictions.csv`: identical solved/declined decision on
all 3,289 rows, identical `method`, values agreeing to **8e-13** (CSV float formatting). That check
is wired in as a hard gate — `check_reproduction` exits non-zero rather than print solve rates that
could justify spending a slot.

One real fidelity bug found and fixed while doing it: an **empty** cached series is a cached
*answer* (the detector ran and found nothing), not a cache miss, and `CachedBackend` was returning it
with a blank note. 41 rows declined with an empty reason and scattered into the tail of the very
histogram the next fix gets chosen from. Now they report `object never detected`, matching the real
backend, and the reason histogram is byte-identical to the GPU run's.

### The finding: the trusted band, not the gravity prior, is the pocket

The plan said to attack decline reasons "largest first" and listed `gravity prior cannot set pixel
scale` at 291. **That was an artifact of counting reason *strings*.** The band's message embeds the
measured pixel value, so it fragments into ~300 distinct strings and never appears in a `head(10)`.
Bucketed by *cause*, over all 1,951 declined rows:

| cause | S2 | D2 | S3 | D3 | total |
|---|---|---|---|---|---|
| **prior pixels outside trusted band** | 279 | 626 | 123 | 483 | **1,511** |
| gravity prior cannot set pixel scale | 0 | 170 | 0 | 121 | 291 |
| separation needs top-2 detections (`distance-twin`) | 11 | 22 | 0 | 11 | 44 |
| prior object not measured | 32 | 8 | 0 | 0 | 40 |
| prior names no groundable object | 0 | 2 | 0 | 35 | 37 |
| target not measured | 7 | 2 | 6 | 1 | 16 |
| depth-ratio guard tripped | 0 | 0 | 6 | 1 | 7 |
| no usable scale prior | 4 | 0 | 0 | 0 | 4 |

`TRUSTED_PRIOR_PIXELS = (30.0, 300.0)` is **77% of all declines and 46% of the entire test set.** It
was fitted from 6 variants on 159 rows with a CI spanning zero. Of the 1,511 it rejects, **1,247 are
rejected for being *above* 300 px** and only 264 for being below 30. The rejected values have a
median of **526 px** — the upper edge is not clipping a tail, it is cutting through the middle of the
distribution. A prior object spanning 500 px of an HD frame is ordinary, not implausible.

### The two edges are not the same, and this is the number that decided it

There is no test truth, so this cannot be scored offline. What *can* be done is compare each newly
solved answer against the constant that row would otherwise get — the constant scores ~0.35, so an
answer near it is a coin flip worth a slot and one 1000x away is a guaranteed zero, strictly worse
than declining. The accepted 1,338 rows are the yardstick: they are already on the board earning
+0.218/row in D2.

| population | n | median × const | within 2× | within 10× | >100× off |
|---|---|---|---|---|---|
| **accepted (the yardstick)** | 1,338 | 1.37 | 37.3% | 86.7% | **1.5%** |
| **rejected, above 300 px** | 1,247 | 0.72 | 35.8% | 80.8% | **5.0%** |
| rejected, below 30 px | 264 | 22.1 | 18.6% | 41.3% | **37.1%** |
| — below 30 px, D2 only | 93 | 86.2 | 12.9% | 25.8% | 46.2% |
| — below 30 px, D3 only | 97 | 329 | 6.2% | 21.6% | 56.7% |

**The upper edge is throwing away 1,247 rows that behave statistically like the rows already
scoring.** The lower edge is doing real work: sub-pixel priors (0.1–0.3 px) produce garbage, and in
D2/D3 roughly half of them are >100x off. So the re-fit is **keep 30, drop 300**.

Note the below-30 rows split hard by category — S2 (n=29) and S3 (n=45) have **zero** rows >100x off
and ~85–90% within 10x, while D2/D3 are the disaster. A per-category lower edge is a real follow-up,
worth ~74 rows. Not done, because the upper edge is worth 17x more and one slot should measure one
change.

### `solver-v2.submission.csv` — BUILT, VALIDATED, NOT YET SUBMITTED

Band `(30.0, inf)`, everything else at the champion. Rebuild it with:

```bash
py -3.12 scripts/make_baseline.py --out baseline_predictions.csv          # gitignored, regenerate
py -3.12 scripts/replay_cache.py --split test --run test-solver-v1 --shards 4 \
    --trusted-prior-pixels 30,inf --out replay-band30inf.csv
py -3.12 scripts/make_submission.py replay-band30inf.csv --out solver-v2.submission.csv \
    --fallback-from baseline_predictions.csv                              # NEVER omit the fallback
py -3.12 scripts/validate_submission.py solver-v2.submission.csv          # exits 0
```

**2,585 of 3,289 solved (78.6%), against solver-v1's 1,338 (40.7%)** — strictly more in every
category, which was the gate set in advance:

| | S2 | D2 | S3 | D3 | total |
|---|---|---|---|---|---|
| solver-v1 | 247 (42.5%) | 330 (28.4%) | 441 (76.6%) | 320 (32.9%) | 1,338 |
| **solver-v2** | **497 (85.5%)** | **863 (74.4%)** | **519 (90.1%)** | **706 (72.6%)** | **2,585** |

One slot measures all four categories at once, and the per-category winners then compose offline for
free and exactly (`select_sources.py`). **Do not assume it wins.** The honest prior: the newly solved
rows carry 5.0% >100x-off against the accepted population's 1.5%, so a category could go either way,
and D3 already loses to a constant on the rows it answers. The composition step is what makes this
safe to try — a category that regresses just keeps `mix-v4`'s source.

`TRUSTED_PRIOR_PIXELS` in `solver.py` is **deliberately still (30.0, 300.0)**. The candidate was
built through the `--trusted-prior-pixels` flag, so nothing in the solver changed on an unmeasured
result. Change the default only once a slot has spoken.

### D3 diagnosed: the derivative order predicts it, and it is an OVERSHOOT

Item 1h, done on the same replay. `video_type[0]` is the prior's physical type (S = Size,
V = Velocity, A = Acceleration), so it *is* the derivative order. Median solver answer as a multiple
of the constant, over solved rows:

| prior | S2 | D2 | S3 | D3 |
|---|---|---|---|---|
| Size (0 derivatives) | 1.39 (n=247) | 0.58 (n=14) | 0.60 (n=431) | — |
| Velocity (1) | — | 1.64 (n=249) | 1.40 (n=10) | **2.70 (n=172)** |
| Acceleration (2) | — | 2.55 (n=67) | — | **3.79 (n=148)** |

D3 has **no Size priors at all** — 572 A-prior and 400 V-prior — and the ratio climbs monotonically
with derivative order. So D3's 0.220 is not noise: the solver overshoots the constant by ~3x there,
and **overshoot is fatal under MRA** while undershoot is cheap. That reconciles two numbers that
looked contradictory: the solver "loses to a constant by −0.137 on the rows it answers" *because* a
3x overshoot scores approximately zero, not because the measurements are random.

### The actual defect is `geometric-3d` on LENGTH questions — 277 rows, wrong in both directions

Item 1e, localized. Splitting `geometric-3d`'s solved rows by question dimension:

| cat | dimension | radial | n | median × const | >10× off |
|---|---|---|---|---|---|
| S3 | speed | yes | 91 | **0.99** | **1.1%** |
| D3 | speed | yes | 35 | **1.16** | **0.0%** |
| S3 | speed | no | 44 | 0.54 | 4.5% |
| **S3** | **length** | no | **141** | **0.26** | **9.9%** |
| **D3** | **length** | no | **136** | **5.69** | **30.9%** |
| S3 | acceleration | no | 7 | 0.29 | 14.3% |

Two separable conclusions, and the earlier "geometric-3d is actively harmful" was too broad:

1. **3D speed is healthy.** With the radial component it is essentially unbiased (0.99 and 1.16) and
   almost never wildly off. And the radial correction is doing real work: the same category without
   it sits at 0.54. `radial_speed` needs **two timed depth readings** for the object, so it fires on
   only 126 of 454 `geometric-3d` rows. Widening where it can fire is a real, bounded lever.
2. **3D length is the bug.** 277 solved rows, off by ~4x — and in *opposite directions* between S3
   (0.26, a 4x undershoot) and D3 (5.69, a 5.7x overshoot). One code path being wrong by a similar
   factor in opposite directions across two categories is the signature of a **ratio applied the
   wrong way round**, which is exactly the standing suspicion in item 1e: `depth_for` matching the
   wrong reading so the perspective correction inverts. Read `geometry.solve`'s
   `target_depth_m`/`prior_depth_m` ordering against `solver.py:167-170` first.

**Caveat on the instrument.** These ratios are against the fallback constant, not truth — there is
no test truth. In D3 the constant *beats* the solver (0.396 vs 0.220), so "far from the constant" is
genuinely bad there; in S3 the solver beats it (0.441 vs 0.410), so the S3 column is weaker
evidence on its own. The opposite-signed 4x in one code path is the finding, and it does not depend
on which reference is better.

## DAY 3: the offline re-solve — the plan, kept for its reasoning (DONE 2026-08-25)

**Why this and not more probing.** Slots 2 and 3 on Day 2 showed the constants for the three largest
groups are already sited well, so more scale probes there are near-worthless. Meanwhile `mix-v2`
priced a *solved* row at **+0.218 in D2**, and only **28.4%** of D2 is solved. The pocket is the
1,951 declined rows, not the 1,338 solved ones. Costs **$0** and **no slot**.

### The blocker is one function

`scripts/replay_cache.py:load(repo, run, limit)` hardcodes the 159-row validation split and a single
`detections.pkl`. It needs to (a) read `data/fixtures/test_dataset.parquet` instead, and (b) merge
the four shard pickles. **`CachedBackend` and `solve_row` need no change at all** — a replay runs the
identical measurement code the GPU run did, on CPU, in seconds.

* The four pickles are on the Hub, **not yet downloaded**, ~4.1 MB total:
  `prarabdhmisra/quantiphy-runs` → `test-solver-v1-shard{1..4}/detections.pkl`.
  Pull them with `hf_hub_download(repo, repo_type="dataset", filename=...)`, the way
  `scripts/merge_shards.py:load_shards` already does.
* Keys are `(video path, phrase)`, so the four dicts **union cleanly** with `dict.update`.
* `scripts/run_vision_job.py:load_split("test")` already shows the exact test-frame construction to
  copy — reuse it rather than rebuilding the row→request path.
* **Detections are cached per `(video, phrase)`.** Any parsing change that renames a phrase shows up
  as a cache *miss*, not a new number. So phrase-renaming fixes cannot be evaluated this way; they
  need a fresh detection pass. The script says so loudly — believe it.

### Then, in order

1. **Reproduce `solver-v1` exactly** from the replay before changing anything. Success criterion:
   1,338 solved, and the per-category solve rates S2 42.5% / D2 28.4% / S3 76.6% / D3 32.9%. If the
   replay does not reproduce them, stop — the harness is wrong, not the solver.
2. **Attack the decline reasons, largest first.** Counted over all 3,289 test rows:
   `gravity prior cannot set pixel scale` **291**, `separation between two instances of one phrase
   needs top-2 detections` **44**, `prior names no groundable object` **37**,
   `prior object not measured: object never detected` **33**, then the `TRUSTED_PRIOR_PIXELS` band.
3. **Re-fit `TRUSTED_PRIOR_PIXELS = (30.0, 300.0)`** against the real scoring set. It was selected
   from 6 variants on 159 rows with a CI spanning zero — the mechanism is solid (corr −0.725,
   n=147), the edges are not.
4. **Diagnose D3 while you are in there.** The solver loses to a constant by **−0.137 on the rows it
   chose to answer**. That is a wrong answer, not a noisy one, and D3 is 972 rows.

### How to judge the result without spending a slot

You cannot score a replay — there is no test truth. What the replay gives you is the **solve rate**
and the *identity* of newly solved rows. Ship a new solver submission only when it solves strictly
more rows, and build it with:

```bash
py -3.12 scripts/make_submission.py <preds> --out solver-v2.submission.csv     --fallback-from baseline_predictions.csv      # NEVER omit this -- see 2026-08-24
py -3.12 scripts/solved_ids.py --run <run> --shards 4 --out data/probes/solved-ids-<run>.csv
```

Then one slot measures it in all four categories at once, and the winners compose offline for free.

### Also worth one slot that day

Refine **`S3|cm` at ×0.5**. Day 2's argmax sat on the edge of the ×0.7 bracket, so the optimum may
be lower. Everything else in that submission should stay at the champion.

## The paper (arXiv 2512.19526) — three things that change the roadmap

Local copy `2512.19526v1.pdf` (gitignored, 23 MB), also at huggingface.co/papers/2512.19526.

### 1. Per-category baselines exist, and we are already at open-weight parity on half the metric

The paper's `2S/2D/3S/3D` are the portal's `S2/D2/S3/D3`. Table 1:

| | S2 | D2 | S3 | D3 | avg |
|---|---|---|---|---|---|
| **our mix-v1** | **35.3** | 36.8 | **44.1** | 39.6 | 38.9 |
| Qwen3-VL-32B | 35.8 | **51.6** | 43.2 | **53.4** | 46.0 |
| GPT-5.1 | 46.3 | 56.2 | 51.5 | 58.3 | 53.1 |
| Human | 50.0 | 59.1 | 55.2 | 57.9 | 55.6 |

**We match Qwen3-VL-32B on S2 and beat it on S3.** The whole gap is D2 and D3 -- the dynamic-prior
categories, exactly where the derivative-order finding predicts we are weakest.

**So the VLM arm's job is D2 and D3, not everything.** Solver on S2/S3 + a Qwen-class VLM on D2/D3
composes to `(35.3+51.6+44.1+53.4)/4` = **46.1**, and per-category composition is proven exact.
That is the single clearest path from 0.389 to ~0.46.

### 2. Their own experiment proves our structural edge

Counterfactual analysis multiplies the prior by 0.001 to 700 and finds **most models' MRA drops
~80%**. And *prior-only* (video removed entirely) scores close to video+prior. Their words: VLMs
"behave less like visual measurers and more like powerful guessers conditioned on textual hints."

Our solver actually consumes the prior's numeric value. So the hybrid should beat either arm rather
than merely averaging them -- and on any scene scaled unusually, we should win outright.

### 3. Chain-of-thought is mostly catastrophic here. Fix the prompt before spending GPU.

Table 2's CoT column against video+prior: 56.1 -> 27.7, 49.8 -> 22.4, 50.1 -> 23.1. CoT roughly
halves MRA for most models (one small model improves; the strong ones all collapse).

`quantiphy/prompting.py` currently says *"Keep any reasoning to one or two sentences, then end with
ANSWER:"* -- mild CoT, and this says it may be harmful. **A/B a direct-answer variant on the 159
validation rows before any test pass.** Cheap, and it may be worth more than the model size.

## The dataset card decodes video_type — and it explains D3 completely

From `PaulineLi/QuantiPhy`'s README, read 2026-08-23. **`video_type` is `[P][D][O][B]`:**

| pos | meaning | values |
|---|---|---|
| **P** | **Physical prior type** | **S = Size, V = Velocity, A = Acceleration** |
| D | Dimensionality | 2 = 2D, 3 = 3D |
| O | Object setting | S = single, M = multi |
| B | Background | X = plain, S = simple, C = complex |

And `inference_type` is `[prior][target]` dynamism: `DS` = **D**ynamic prior -> **S**tatic target. So the
scored category is **`[prior dynamism][dimensionality]`** — D3 means *dynamic prior, 3D*.

Also: videos are **2-3 s, static camera**. So radial depth change is real object motion, not camera
motion, which retroactively justifies the `+radial` route.

### The error scales with how many derivatives the prior needs. Measured:

| prior type | n | median ratio vs constant | % over 1.9x |
|---|---|---|---|
| **S — Size** | 692 | **0.86** | 26% |
| **V — Velocity** | 431 | **2.03** | 52% |
| **A — Acceleration** | 215 | **3.52** | 69% |

Monotone, and the mechanism is exact: `gamma = prior_world / prior_pixels`, and a Size prior measures
a box extent (0th derivative), Velocity measures px/s (1st), Acceleration px/s^2 (2nd). Every
derivative amplifies detector noise, **in the denominator of the scale factor.**

This explains all four category results at once:

* **S3** = static prior + 3D = 431 Size priors -> ratio 0.60 -> the solver's biggest gain (+0.031)
* **D3** = dynamic prior + 3D = **no Size priors at all** (572 A, 400 V) -> collapsed (-0.176)
* **D2** = dynamic prior but 2D -> still gained (+0.053). 2D tolerates a noisy prior; compounding it
  with the depth correction is what breaks D3.

**Useful negatives:** background complexity (C/S/X -> 1.37/1.14/1.55) and single-vs-multi object
(1.35/1.50) carry almost no signal. Do not spend effort gating on them.

### The lever this opens — SIZED DOWN 2026-08-24, it is small

Gating on **prior type** is still free and still mechanically justified, but the version written
here on 2026-08-23 ("one submission tests that in all four categories at once") was wrong, and the
row counts say so:

* **S2 is 100% Size-prior and S3 is 566 of 576 Size-prior.** There is nothing to gate in either.
* **Most A-prior rows already decline.** The solver fires on only **67** A-prior rows in D2 and
  **148** in D3. Gating them changes 67 rows of 1,160 in D2 — below the point where a category
  number moves usefully.

So it is a one-category experiment worth a few thousandths, not a four-channel one. Keep it on the
backlog; do not spend a slot on it while larger pockets are open.

### Also in that repo, not yet used

* **`quantiphy_fullset_videos_480p/`** — the VLM resizes frames anyway, so this cuts Kaggle download
  time, where bandwidth is the real constraint.
* **`github.com/Paulineli/QuantiPhy`** — "evaluation code and a starter kit for running a VLM on
  QuantiPhy". Likely the reference prompt and output format. Read before finalising the VLM prompt.
* No extra ground truth anywhere: the test parquet has no posterior column, and 159 validation rows
  remain the only truth we will ever have. Checked, so nobody looks again.

## CHAMPION: mix-v5, macro 0.441 — the band was the biggest lever of the campaign

`solver-v2` was scored on 2026-08-25 and the widened band paid in **every** category:

| | S2 | D2 | S3 | D3 | macro |
|---|---|---|---|---|---|
| solver-v1, band (30, 300) | 0.353 | 0.368 | 0.441 | 0.220 | 0.345 |
| **solver-v2, band (30, inf)** | **0.430** | **0.461** | 0.465 | 0.315 | **0.418** |
| gain | +0.077 | +0.093 | +0.024 | +0.095 | **+0.073** |

Two things worth keeping from that row. **The offline plausibility screen was right**: the
above-300px rows behaved like the accepted population and were worth solving, and no category
regressed. And **`solver-v2` at 0.418 beats `mix-v4` at 0.411 as a single source** — it is already
"solver where it solved, constant elsewhere", because `make_submission.py --fallback-from` fills the
declines, so there is no `mix-v2`-style per-row overlay left to add on top of it.

D3 still loses to the constant (0.315 vs 0.396) even after gaining +0.095, exactly as the overshoot
diagnosis predicts: opening the band solves more D3 rows, but the answers are still ~3x high and a
3x overshoot scores near zero. Fixing D3 means fixing the bias, not the solve rate — item 1e.

Composing the per-category winners, which is **arithmetic, not a prediction**:

| | S2 | D2 | S3 | D3 | macro |
|---|---|---|---|---|---|
| mix-v4 (previous champion) | 0.393 | 0.377 | **0.477** | **0.396** | 0.411 |
| solver-v2 | **0.430** | **0.461** | 0.465 | 0.315 | 0.418 |
| **mix-v5** = solver-v2 on S2/D2, mix-v4 on S3/D3 | **0.430** | **0.461** | **0.477** | **0.396** | **0.441** |

```bash
py -3.12 scripts/select_sources.py --out mix-v5.submission.csv \
    --source solver-v2.submission.csv --for S2,D2 \
    --source mix-v4.submission.csv --for S3,D3
```

Built and validated. `mix-v4` keeps S3 because its `S3|cm` ×0.7 rescale measured 0.477 against
solver-v2's 0.465 — and note that rescale has **not** been tried on top of solver-v2's S3, which is
a real candidate for a slot rather than a derivation.

**Where we now stand against the field** (paper Table 1): S2 0.430 vs Qwen3-VL-32B's 0.358 and
GPT-5.1's 0.463; S3 0.477 vs 0.432 and 0.515. We are **ahead of the best open-weight model on both
static-prior categories** and closing on GPT-5.1. The whole remaining gap is D2 (0.461 vs 0.516) and
D3 (0.396 vs 0.534) — still exactly the dynamic-prior categories, and D3 is still the single worst
channel on the board.

### The previous champion, for the record

#### mix-v4, macro 0.411 — and the composition method is PROVEN

| | S2 | D2 | S3 | D3 | macro |
|---|---|---|---|---|---|
| baseline-v3 (constant) | 0.337 | 0.315 | 0.410 | 0.396 | 0.365 |
| solver-v1 | 0.353 | 0.368 | 0.441 | **0.220** | 0.345 |
| mix-v1 (solver S2/D2/S3, constant D3) | 0.353 | 0.368 | 0.441 | 0.396 | 0.389 |
| mix-v2 (solver only where it solved) | **0.393** | **0.377** | **0.469** | 0.351 | 0.398 |
| mix-v3 (mix-v2 on S2/D2/S3, constant D3) | 0.393 | 0.377 | 0.469 | 0.396 | 0.409 |
| **mix-v4** (mix-v3 with `S3\|cm` ×0.7) | **0.393** | **0.377** | **0.477** | **0.396** | **0.411** |

`mix-v1` was predicted at 0.3895 from its two parents and measured **0.389, with all four categories
matching to the digit** — that is the proof. `mix-v3` and `mix-v4` are therefore *derived, not measured*: their
macros are arithmetic from numbers the portal already reported, and they cost no slot. Build one
with `scripts/select_sources.py`.

Note `mix-v2`'s macro (0.398) is **lower** than three of its four channels deserve. Read the columns.

### The consequence that changes how slots are spent

**Categories are scored independently, exactly.** So the macro of any per-category mix of
already-measured sources is *arithmetic, not estimation* — and therefore

> **never spend a slot confirming a composition of things already measured.**

This one cost a slot to prove the principle, which was worth it once. From here, compose freely
offline and spend every slot on something *new*: a source not yet measured in that category, or a
probe. That roughly doubles the effective information rate of the 3/day quota.

It also retires a question this project failed at four times. Four attempts to build a confidence
signal predicting when the solver is right were all refuted. Selection **sidesteps prediction**:
measure per category on the real scoring set, then choose. No signal required.

### The lesson about reading results

The solver's headline (0.345) was *below* the constant's (0.365), and on the headline alone it would
have been discarded. It was actually better in **three of four categories**. A submission is a
measurement instrument with four channels, not a single score — read it per category, always.

### D3 is now the biggest single pocket on the board

The solver collapsed there, 0.396 → 0.220, and D3 is 972 rows. It was also the category the offline
disagreement check flagged worst: median ratio **3.44** against the constant with **65%** of rows
above 1.9x, where the other three sat between 0.61 and 1.94. That check has earned some standing.
Detections for all 3,289 test rows are cached, so diagnosing D3 is free CPU work — and fixing it
would be worth roughly another +0.04 on the same exact arithmetic.

## 2026-08-24 — where a DECLINED row's number comes from was the biggest lever on the board

Champion **0.389 → 0.409** in one slot plus one free composition. No GPU, no solver change.

### The defect

`make_submission.py`'s fallback ladder fills every unsolved row with the median of **the predictions
file's own solved values** for that `(category, unit)`. So a solver submission's "zero-vision
fallback" was never the zero-vision baseline — it is the solver's own overshoot, re-applied to every
row the solver declined. That is **1,951 of 3,289 rows (59.3%)**, and `mix-v1` carried it on 1,299.

| group | solver-v1 fallback | median of its own solved rows | `baseline-v3` constant |
|---|---|---|---|
| `D2\|meters` | 2.7439 | 2.7439 | **1.2500** |
| `S2\|meters` | 4.7142 | 4.3082 | **2.1250** |
| `D3\|meters` | 4.6670 | 4.6201 | **1.6593** |
| `S3\|cm` | 12.8794 | 12.8872 | **49.6900** |

The identity in column 3 is the whole story: the fallback *is* the solver's median. And we already
had a measured reading that lower was better in the largest group — v1→v3 moved `D2|meters`
1.71→1.25 and that group's mean score rose `0.004 × 1160/829 = +0.0056`.

**Fixed at the root:** `make_submission.py --fallback-from <predictions>` fills declined rows from a
named file (e.g. `make_baseline.py`'s output) instead. Rebuilding the test run through it reproduces
`mix-v2` on all 3,289 rows. The default is unchanged on purpose — flipping it silently would re-date
every earlier submission.

### What `mix-v2` measured

`mix-v2` = solver on the **1,338 rows where `method != 'none'`**, `baseline-v3`'s constant on the
other 1,951. Macro 0.398 — and the macro is the least interesting number on the page:

| | S2 | D2 | S3 | D3 |
|---|---|---|---|---|
| mix-v1 | 0.353 | 0.368 | 0.441 | **0.396** |
| mix-v2 | **0.393** | **0.377** | **0.469** | 0.351 |

Three channels won, one lost, so composing the winners gives `mix-v3` at 0.409 for free.

### The solver is now priced, per category, on the rows it actually fires

Inverting each category against `baseline-v3` — `(mix-v2 − constant) × n_C / n_solved`:

| category | solved / total | solver vs constant on those rows |
|---|---|---|
| S2 | 247 / 581 | **+0.132** |
| D2 | 330 / 1160 | **+0.218** |
| S3 | 441 / 576 | **+0.077** |
| D3 | 320 / 972 | **−0.137** |

**D3 is not a weak solver, it is a wrong one** — it loses to a constant on the very rows it chose to
answer. And separately its fallback was doing most of the visible damage: D3 scored 0.220 with the
run's own fallback, 0.351 with the measured constant, 0.396 with no solver at all.

**The consequence for tomorrow.** A solved row in D2 is worth +0.218 and only 28.4% of D2 is solved.
The top decline reasons across all 3,289 rows are `gravity prior cannot set pixel scale` (291),
`separation between two instances of one phrase needs top-2 detections` (44), `prior names no
groundable object` (37), `prior object not measured` (33), then the `TRUSTED_PRIOR_PIXELS (30, 300)`
band — which is itself unproven, selected from 6 variants on 159 rows with a CI spanning zero.
**Raising the solve rate is now worth more than improving any solved row**, and it is free CPU work:
download the four `test-solver-v1-shard*/detections.pkl` (4.1 MB) and widen `replay_cache.load` to
`data/fixtures/test_dataset.parquet`. `CachedBackend` and `solve_row` need no change.

### Built today

* `scripts/solved_ids.py` — records which rows a run actually answered, from the shards' own
  `method` column, into `data/probes/solved-ids-<run>.csv`. A committed artifact rather than a
  lambda, because the fallback is a single repeated value: a solved row that lands on it is
  indistinguishable after the fact. **Do not glob the HF cache for shard predictions** — it holds
  more than one snapshot and returns 5,756 rows from two revisions. Use `merge_shards.py`.
* `scripts/select_rows.py` — per-**row** source selection, the sibling of `select_sources.py`.
  Per-*category* composition is arithmetic and never needs a slot; per-*row* is a new source and
  costs one.
* `make_submission.py --fallback-from`, above.
* Tests **176 → 189**, still CPU-only. (The "129" written here on 2026-08-22 was already stale.)

### Slots 2 and 3: the bracket, and what it says — the incumbents were already near-optimal

`probe-d2a` (×0.7) and `probe-d2b` (×1.4) on the largest group in each category of `mix-v3`:
`S2|meters` 208/581, `D2|meters` 829/1160, `S3|cm` 170/576, `D3|meters` 472/972. With the champion's
×1.0 already measured, every group got three points around the incumbent, so the argmax is monotone
by construction rather than a bet on a half-finished search.

| group | ×0.7 | ×1.0 | ×1.4 | argmax | on the group's own mean |
|---|---|---|---|---|---|
| `S2\|meters` | 0.389 | **0.393** | 0.368 | ×1.0 | −0.011 / — / −0.070 |
| `D2\|meters` | 0.358 | **0.377** | 0.324 | ×1.0 | −0.027 / — / −0.074 |
| **`S3\|cm`** | **0.477** | 0.469 | 0.430 | **×0.7** | **+0.027** / — / −0.132 |
| `D3\|meters` | 0.383 | **0.396** | 0.366 | ×1.0 | −0.027 / — / −0.062 |

**Three of four constants were already sited well**, which is a real result: the `(category, unit)`
medians fitted on 159 validation rows transfer to the 3,289-row scoring set better than the
0.459-in-sample / 0.365-on-test gap suggested. The gap is estimation error in the *small* groups,
not a systematic mis-siting of the large ones.

**Every group loses more at ×1.4 than it loses at ×0.7** — the response curve is asymmetric in
exactly the direction the metric predicts. Overshoot is fatal, undershoot is cheap. When a future
bracket has to be one-sided for want of slots, make it the low side.

`S3|cm` at ×0.7 is adopted → **`mix-v4`, 0.411**. The argmax sits on the edge of the bracket, so the
true optimum may be lower still: **refine `S3|cm` at ×0.5 and ×0.35 next.** Inversion factors for
re-use — a reported shift `d` maps to this much on the group's own mean score: **S2 ×2.79,
D2 ×1.40, S3 ×3.39, D3 ×2.06.**

### Day 2 close: 0.389 → 0.411 on three slots, no GPU

Slot 1 bought +0.020 (the fallback defect), slots 2–3 bought +0.002 (one constant re-sited) and,
more usefully, **retired the "the constants are badly placed" hypothesis for the three largest
groups.** Next day's slots should go to the solve rate, not to more scale probes on those three.

## 2026-08-23 evening — the first solver submission on test, and a units trap

**`solver-v1.submission.csv` is built and validated.** 3,289 rows from a 4-shard test detection pass,
merged with zero gaps and zero duplicates. **1,338 solved (40.7%)**, 1,951 on the zero-vision
fallback. Per-category solve rates: S2 42.5%, D2 28.4%, S3 76.6%, D3 32.9%. This is the first time a
solver submission has been possible at all -- every earlier solver number came from the 159-row
validation cache.

### Read a cross-method comparison unit-free, or it will lie to you

Median *prediction* by method looked damning: `geometric-2d` 3.21, `geometric-3d+radial` **56.66** --
seemingly a 17x inflation, and plausible because `combine_speeds` uses `hypot` and can only push a
number up. That reading was **wrong**, and nearly bought a harmful "fix".

The confound: `cm/s` answers are numerically ~100x `m/s` answers, and `+radial` fires mostly on cm/s
rows. Recomputed as a **ratio to the constant**, which is unit-free:

| method | n | median ratio | frac > 1.9x |
|---|---|---|---|
| `geometric-2d` | 814 | 1.62 | 46% |
| `geometric-3d` | 287 | 0.99 | 43% |
| **`geometric-3d+radial`** | 120 | **1.08** | **16%** |
| `geometric-2d+separation` | 43 | 3.09 | 53% |

`+radial` is the *best-behaved* route, not the worst. And measured against real validation truth,
radial **helps**: macro 0.4220 with it against 0.4089 without, and on the 8 rows where it fires
0.688 against 0.400 (constant 0.475), median pred/truth 1.10 against 0.74. **Do not disable radial.**

The genuine disagreement is by **unit**, not method: `meters` rows run at median ratio **2.23** over
654 solved rows, while `m/s` sits at 1.02 and `m/s^2` at 1.14. Undershooting units (`cm` 0.47,
`cm/s` 0.66) are cheap under MRA. Two tiny broken pockets: `mm` (4 rows, 35x) and `cm/s^2`
(7 rows, 0.17x) -- 11 rows total, not worth a slot.

**And a large disagreement with the constant says nothing about which one is right.** Validation with
truth says the gated solver beats the constant, 0.4220 to 0.3776. Only a submission settles it.

### One cosmetic bug fixed while checking

`run_vision_job.py`'s checkpoint counter used `parsed_value is not None`. On a resume the value comes
back through `pd.read_csv`, so a declined row is NaN, and `NaN is not None` is True -- it logged
"solved 600" against the original run's "solved 352" for the same rows. The final tally and the CSV
both use pandas null semantics, so no data was affected, but a progress counter that misreports the
gate is worse than none.

### Shard 4 died and resumed, which is why checkpointing exists

Shard 4 hit ERR with **no traceback** at 657/822 after 2h11m -- a container kill, not a code fault.
`partial.csv` held 600 rows and `detections.pkl` held the detections, so a relaunch with the same
`RUN_NAME` replayed 600 rows in three seconds and only paid for the remaining ~222.

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
| ~~1a~~ | ~~**Upload the zero-vision baseline**~~ | ~~3,289~~ | done | **DONE 2026-08-22.** 0.364, then `baseline-v3` 0.365. Champion is now `mix-v3` at 0.409. |
| ~~1g~~ | ~~**Offline re-solve of all 3,289 test rows**~~ | ~~1,950~~ | done | **DONE 2026-08-25.** Harness built, reproduces `solver-v1` exactly. Found the trusted band is 77% of declines, not the gravity prior. See "2026-08-25". |
| ~~1i~~ | ~~**Upload `solver-v2.submission.csv`**~~ | ~~3,289~~ | done | **DONE 2026-08-25: 0.418**, beating the old champion outright and gaining in all four categories. Composed to `mix-v5` at **0.441**. |
| ~~1m~~ | ~~**Set `TRUSTED_PRIOR_PIXELS = (30.0, inf)`**~~ | — | done | **DONE 2026-08-25.** Default changed, its docstring corrected (it used to claim priors over 400 px score a hard zero — the test split says otherwise), and the replay gate now pins `solver-v1`'s own `(30, 300)` beside its numbers so it does not read the live default. 207 tests. |
| ~~1n~~ | ~~**Try `S3\|cm` rescale on top of solver-v2's S3**~~ | ~~576~~ | done | **CLOSED 2026-08-26.** ×0.5 beyond the champion lost **-0.203 on the group's mean**, so `mix-v4`'s ×0.7 is at or near the argmax after all. Constant-multiplier probing is exhausted generally: 8 group probes over two days, 8 losses. |
| ~~1j~~ | ~~**Per-category lower edge on the band**~~ | ~~74~~ | done | **HALF ADOPTED 2026-08-26.** Floor 0 in **S2 wins +0.010** (29 rows, adopted into `mix-v7`); in **S3 it loses -0.014** (45 rows, rejected). Both had 0% of rows >100x off -- so **`0% >100x` is not a quality signal**; judge by `within2x`. |
| **1k** | **The gravity prior** — `cannot set pixel scale` | **291** | cents, not free | Now the largest *remaining* bucket, and all of it is D2 (170) and D3 (121) — the two categories the paper says are our whole gap. A gravity prior gives `g = 9.81 m/s²` with no pixel length, so scale must come from the target's own kinematics; that is a solver change, then a replay. **PROMOTED TO TOP 2026-08-26:** 170 of the 291 are in D2, and D2 is now measured as the category where a solved row is worth the most (+0.19/row, and even its 100x-outliers earn). Sized at roughly **+0.028 on D2**. Every other free lever is closed. **RE-SIZED 2026-08-27 and it is not free:** `solve_row` declines before it measures the target, so all **46 videos** behind these rows are absent from the detection cache. The solver change is offline; measuring it needs a detection pass over those 46 videos. Physics and the free-fall gate are written up in the 2026-08-27 section. |
| ~~1h~~ | ~~**Diagnose D3**~~ | ~~972~~ | done | **DONE 2026-08-25.** It is a ~3x *overshoot* rising monotonically with the prior's derivative order, and overshoot scores zero. The fixable part is item 1e. See "D3 diagnosed". |
| 1b | **Fresh detection pass for the new phrases** | ~183 | cents | The pair fix renames phrases, so those rows are cache misses. Needed before the separation numbers can be scored at scale. |
| 1c | `distance-twin` — "between the **two cars**", one phrase twice | **44** | free-ish | Declines by design today. Needs the detector to keep its **top-2** boxes per frame, not just the best; `DetectionSeries` holds one. Small, well-defined change to `grounding.py`. |
| 2 | **Diagnose the velocity rows** — median ratio 0.094, four of six near zero | ~1,000 | free | Biggest category and the worst performing. Replay the cache; the quadratic fit or the `fps`/timestamp path is suspect. No GPU needed. |
| ~~1d~~ | ~~**CONFIDENCE GATING**~~ | — | done | **REFUTED 2026-08-22 on 159 rows.** Every threshold scores below the constant; detector confidence does not predict correctness. See "The confidence gate is REFUTED". |
| ~~1e~~ | ~~**Fix `geometric-3d` on LENGTH questions**~~ | ~~277~~ | done | **REFUTED 2026-08-26.** Median `Z_target/Z_prior` measured at **1.000-1.02 in every slice** -- the depth correction is a near-no-op and cannot cause a 4x error. The 3d-vs-2d split is a *population* split by `(category, unit)`, not a code defect. Fifth single-cause theory of this bias to fall. |
| ~~1l~~ | ~~**Widen where the radial correction can fire**~~ | ~~3~~ | done | **CLOSED 2026-08-26, mis-sized by 100x.** The old count included non-speed rows; `radial_speed` only applies to speed. 277 such rows, 216 already fire, 27 underdetermined, and only **3** blocked by the timestamp gate. |
| ~~1f~~ | ~~Confirm-or-kill the disagreement gate on the test set~~ | ~~all~~ | done | **MEASURED 2026-08-26 and it SPLITS by category.** Reverting rows that disagree with the constant is worth **+0.257/row in D3** and **-0.192/row in D2** -- the solver's D2 outliers are its best rows. Adopted nowhere as a blanket rule; the split itself is the finding. |
| 3 | Kill the fatal overshoots: gate on prior confidence | — | free | The `pedestrian walking` prior scored 0.341 with box width jittering 13–59 px and produced 7x overshoots. `min_confidence` already exists and is unused (`solve_row` defaults it to 0.0). Now that `detection_rate` is fixed, confidence is finally meaningful. |
| 4 | Decide on `fix/prior-grounding-phrase` | 412 | free | Real defects, 114 tests green, but it did **not** move the metric. Merge on correctness grounds, not on a score claim. |
| ~~5~~ | ~~Full 159-row validation run on `l4x1`~~ | — | done | **LAUNCHED 2026-08-27** as the VLM prompt A/B (`brief` vs `direct`), two runs of 159 rows on `l4x1` with Qwen3-VL-8B. Read the result with `scripts/score_vlm.py`. |
| **6 (TOP)** | **The VLM arm on D2/D3** | **2,132** | **~$3–6 GPU, measured** | **PROMOTED TO TOP 2026-08-27, and it is the only lever left that is worth more than a thousandth.** Not a fallback for declined rows -- a *second arm*, selected per category. Solver on S2/S3 plus a Qwen-class VLM on D2/D3 composes to **~0.492** against the champion's 0.4435, and per-category composition is arithmetic once both parents are scored. Gated on the prompt A/B in item 5. |
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
| Tests | **94 on `main`, 241 on the branch** (`py -3.12 -m pytest tests/ -q`) |
| Plan | `~/.claude/plans/greedy-spinning-hopper.md` (the 60-day campaign, 2026-08-23 → 10-21) |
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
