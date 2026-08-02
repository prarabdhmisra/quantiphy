# Resume here

Last worked: **2026-08-02**. Nothing is in flight — no running jobs, no uncommitted work.

## Paste this to start the next session

> Resuming the QuantiPhy Challenge (NeurIPS 2026) entry. Read
> `C:\Users\prara_\quantiphy\NEXT-SESSION.md` first — full state as of 2026-08-02, nothing in
> flight. Repo is public at https://github.com/prarabdhmisra/quantiphy, 94 tests pass, the
> submission format is settled, and the solver has now run on real video once (20 rows).
>
> The headline problem is in "Smoke test" below: we **systematically undershoot by about 2.4×**.
> Start there — diagnose the bias before adding any new component. Don't re-derive anything in
> "Do not re-derive"; it cost real money. Ask me before spending more than ~$20 in a session.

## Pending — nothing here is started

| # | Item | Cost | Blocked by |
|---|---|---|---|
| 1 | Full 159-row validation run on `l4x1` | ~1–3 h, $5–15 GPU | nothing — needs my go-ahead on spend |
| 2 | Diagnose the 2.4× undershoot (see "Next actions") | free, local | needs (1) to confirm the bias is real |
| 3 | `parse_prior` fix for the `name ~ value unit` form | free, ~30 min | nothing. Worth ~0 points — hygiene only |
| 4 | Fallback arm (VLM estimate), fused in log space | ~$5 GPU | nothing |
| 5 | SAM2 masks for extent; CoTracker3 for weak fits | ~$10 GPU | best done after (2) says where the error is |
| 6 | Follow-up email: 4 unanswered questions | free | parked by choice until ~September |
| 7 | Measure `--shrink` on fallback rows | free once (1) exists | needs (1) |

Nothing is half-finished and no branch is open. Any of these can be picked up cold.

## Smoke test, 2026-08-02 — first contact with real video

`LIMIT=20` on validation, `l4x1`, job `6a6fcdff…`. Solved **13/20**. Measured, not estimated:

| | |
|---|---|
| MRA over the 13 solved rows | **0.300** |
| MRA as-submitted (unsolved → 0) | **0.195** |
| GPT-5.1 on full validation | 0.4856 |

**Caveat: 20 rows, all S2/D2.** No macro average exists for this sample and the CI is enormous.
These numbers rank as "the pipeline runs and is currently bad", nothing finer.

What the 13 solved rows actually look like:

* **Median pred/truth = 0.419.** We undershoot by ~2.4× and we do it *consistently* — only 1 of 13
  overshoots, 4 of 13 land within 2×, 3 of 13 are off by more than 10×.
* A uniform multiplicative undershoot is a **structural bug, not measurement noise**. The likely
  mechanism: the box drawn round the *prior's* object overestimates its pixel extent, so
  `gamma = prior_world / prior_pixels` comes out too small and every downstream answer shrinks by
  the same factor. This is the SAM2-masks hypothesis, and it now has evidence behind it.
* 2 of 13 rows returned identical predictions for different questions — grounding collapsing onto
  one box. Real, but a smaller effect than the scale bias.

The 7 unsolved rows are **not** a test-set problem:

* 6 × "no usable scale prior" are all the same string, `pedestrian walking speed ~1.1 m/s`. The
  parser wants `name = value unit`, and this has `~` and no `=`. **This form appears 0 times in the
  3,289 test rows** — only 4 test priors fail to parse at all, all of them the typo
  `lenth of the credit card = 8.56 cm`. So this failure inflates our apparent error rate on the
  split we measure with, but would cost roughly nothing on test. Fix it for measurement hygiene,
  not for score.
* 1 × "gravity prior cannot set pixel scale" is the solver correctly declining a row it cannot
  anchor. That one needs the fallback arm.

## State

| | |
|---|---|
| Repo | https://github.com/prarabdhmisra/quantiphy (public, MIT) |
| Tests | 94 passing (`py -3.12 -m pytest tests/ -q`) |
| Plan | `~/.claude/plans/inherited-doodling-sun.md` |
| Track | **B (Open-Weight)** primary, A secondary |
| Deadline | **Plan for Oct 1, 2026** (site advertises Nov 5, but its own timeline finalizes rankings mid-October) |
| Bar to beat | GPT-5.1 **0.4856** on validation; human avg 0.556, top humans 0.72; best open-weight (Qwen3-VL-32B) 46.0 |

**Built:** `scoring.py` (MRA + paired bootstrap), `units.py`, `parsing.py`, `geometry.py`,
`vision.py` (backend Protocol), `backends/grounding.py` (Grounding-DINO), `solver.py`,
`scripts/run_vision_job.py` (HF Jobs, checkpoints + resumes), `scripts/make_submission.py`,
`scripts/validate_submission.py`, `notebooks/colab_vision.ipynb`.

**Detection accuracy is now measured, on 20 rows only** — see "Smoke test" above. Everything
beyond those 20 rows is still synthetic-fixture territory.

The HF Jobs path is proven end to end: install, video download, GPU detection, checkpoint, resume
(a warm re-run finished 20 rows in 18 s), scoring, exit 0.

## Do not re-derive — these are measured, and they cost real money

* MRA thresholds are `{0.1..0.9, 0.95}` **per the code**. The paper's Appendix A.2 set
  `{0.5..0.95}` is a typo — it yields 0.376 against the published 0.486.
* **Overshoot is fatal, undershoot is cheap.** `pred ≥ 1.9×` truth scores 0; `0.5×` still scores
  0.4. 25% of GPT-5.1 rows score exactly 0 and **57% of those are overshoots** — the largest
  recoverable pool. Oracle fix = +19 pts; a realistic detector = +7.5 pts.
* **Do not calibrate on the 159-row validation set.** Per-category shrinkage: in-sample 0.493,
  leave-one-out **0.463** — worse than the 0.484 baseline. Global recalibration is +0.002.
* **The unit-error hypothesis is refuted** — only ~4% of GPT-5.1 errors sit near powers of ten.
* Validation has a **±5.7 pt 95% CI**. Use `paired_bootstrap`; accept only if the CI excludes zero.
* The scorer does **no unit conversion**; blank/NaN/zero are hard zeros; any empty category makes
  the whole average undefined.
* Aggregate ensembles in **log space**, never an arithmetic mean.

## Team

**Vector Syndicate** — 2 people. Organizer email **sent 2026-08-01** (contents in
`ORGANIZER-EMAIL.md`).

## Organizers' reply — 2026-08-02

They answered two of the six questions. **Neither blocker remains.**

1. **Submission format: solved.** They posted a real template at
   `https://quantiphy.stanford.edu/competition/eval/quantiphy_submission_template.csv`, now pinned
   at `data/fixtures/quantiphy_submission_template.csv` with its SHA-256 in that folder's README.
   3,289 rows; `id` is 1-based and contiguous, so **`id == parquet row index + 1`** — the guess in
   the email was right. Every shared column matches `test_dataset.parquet` row-for-row with zero
   mismatches. `scripts/make_submission.py` builds against it; `scripts/validate_submission.py`
   diffs against it. Re-check the hash before any real submission — they have changed the linked
   "template" once already.

2. **No leaderboard "any time soon"; it appears "when the deadline approaches."**
   `leaderboard.html`, `submit.html` and `upload.html` all still 404. The consequence matters more
   than the fact: **there is no external feedback signal until roughly October.** So —
   * `quantiphy/scoring.py` on the 159-row validation split is the *only* evidence we get, at
     ±5.7 pt. Treat sub-6-point "improvements" as noise unless `paired_bootstrap` says otherwise.
   * The 3-per-day submission quota is irrelevant for now; nothing is gained by rationing.
   * The upload path stays untested until crunch time. Mitigated by building and freezing the
     submission writer now, so only one column is left to fill at the deadline.

**Still unanswered** (parked until ~September, none of it blocking): which deadline is
authoritative, fine-tuning / external data / ensemble rules, whether gated weights count as
open-weight, and team eligibility across tracks. Draft is still in `ORGANIZER-EMAIL.md`.

## Next actions, in order

1. **Full 159-row validation run** (`l4x1`, ~1–3 h detached, roughly $5–15). 20 rows cannot tell a
   systematic bias from a small sample, and every decision below depends on knowing which it is.
   Read **coverage-limited** (is the geometry right?) separately from **as-submitted**
   (unsolved = 0, what the leaderboard would say).
2. **Diagnose the 2.4× undershoot before building anything new.** If the median pred/truth ratio
   holds near 0.42 across 159 rows it is a bug with a single cause, and fixing it is worth more
   than any new component. Check in this order:
   * Is the *prior* object's box systematically larger than the object? That alone produces exactly
     this signature. Dump a few annotated frames and look.
   * Does the target measurement use the right axis — width vs height vs diagonal — for the
     quantity the question asks about?
   * Is the depth-ratio correction firing on 2D rows where it should not?

   Resist the urge to fit a global correction factor. `paired_bootstrap` first, and note the
   standing warning against calibrating on this split.
3. Fix `parse_prior` for the `name ~ value unit` form. Cheap, and it stops 6 validation rows from
   masquerading as solver failures. **Worth ~0 points on test** — do not confuse it with progress.
4. Build the **fallback arm** (VLM estimate) and fuse in log space. Until then every unsolved row
   is a hard zero. `scripts/make_submission.py` already applies a per-(category, unit) median as a
   floor, so this is an upgrade to that, not a new hole.
5. Then: SAM2 masks for extent (boxes overestimate non-rectangular objects — and step 2 suggests
   that is already costing us), CoTracker3 where the quadratic fit quality is low.

## Submitting

```bash
py -3.12 scripts/make_submission.py predictions.csv --out sub.csv
py -3.12 scripts/validate_submission.py sub.csv        # must exit 0
```

`make_submission.py` fills `parsed_value` in the pinned official template and copies every other
column through untouched; `validate_submission.py` diffs the result against that template
row-for-row. Re-check the template's SHA-256 against the live file first — see
`data/fixtures/README.md`. `--shrink` is available and **unmeasured**: overshoot is fatal and
undershoot is cheap, so a value below 1.0 should help the fallback rows, but nobody has measured
how far. Leave it at 1.0 until it is tested with `paired_bootstrap`.

## Notes

* Compute: **Colab Pro** to iterate, **HF Jobs `l4x1`** for batch (detached, survives a closed
  laptop), Kaggle for free sweeps. Do not buy a GPU — the whole competition is tens of GPU-hours.
* Videos are **not** stored locally (download killed at 30/568, deliberately). Colab and HF Jobs
  fetch them on the remote machine.
* `README.md` publicly documents the competitive analysis above. Trim it into a gitignored
  `NOTES.md` if that becomes a concern.
* Session of 2026-08-01 cost ~$100, mostly research. Set a budget next time.
* Session of 2026-08-02: ~$37 of Claude time, plus three `l4x1` smoke runs (well under $5 of GPU).
  Job artifacts land in `prarabdhmisra/quantiphy-runs` (dataset repo, private) — the detection
  cache there is warm for the first 20 validation rows, so re-running them is free and instant.
  Two of the three runs were spent on bugs, not results: `uv run` environments ship no `pip`, and
  a `LIMIT` run that misses a category used to crash rather than report. Both are fixed.
