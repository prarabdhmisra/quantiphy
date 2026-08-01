# Resume here

Last worked: **2026-08-01**. Nothing is in flight — no running jobs, no uncommitted work.

## Paste this to start the next session

> Resuming the QuantiPhy Challenge (NeurIPS 2026) entry. Read
> `C:\Users\prara_\quantiphy\NEXT-SESSION.md` first — it has the full state as of 2026-08-01 and
> nothing is in flight. Repo is public at https://github.com/prarabdhmisra/quantiphy, 79 tests
> pass, and the whole pipeline exists but **has never touched a real video**.
>
> Do these in order:
> 1. Run the HF Jobs smoke test (`LIMIT=20`, command in the README) and read the output honestly —
>    I expect it to be poor.
> 2. Triage the failures by category: order-of-magnitude errors are scale/unit bugs and are cheap;
>    2× errors are genuine measurement error and are not.
> 3. Then the full 159-row validation run.
>
> Don't re-derive the measured facts in the "Do not re-derive" section below — they cost real
> money to establish. Ask me before spending more than ~$20 in a session.

## State

| | |
|---|---|
| Repo | https://github.com/prarabdhmisra/quantiphy (public, MIT) |
| Tests | 79 passing (`py -3.12 -m pytest tests/ -q`) |
| Plan | `~/.claude/plans/inherited-doodling-sun.md` |
| Track | **B (Open-Weight)** primary, A secondary |
| Deadline | **Plan for Oct 1, 2026** (site advertises Nov 5, but its own timeline finalizes rankings mid-October) |
| Bar to beat | GPT-5.1 **0.4856** on validation; human avg 0.556, top humans 0.72; best open-weight (Qwen3-VL-32B) 46.0 |

**Built:** `scoring.py` (MRA + paired bootstrap), `units.py`, `parsing.py`, `geometry.py`,
`vision.py` (backend Protocol), `backends/grounding.py` (Grounding-DINO), `solver.py`,
`scripts/run_vision_job.py` (HF Jobs, checkpoints + resumes), `scripts/validate_submission.py`,
`notebooks/colab_vision.ipynb`.

**⚠️ Everything vision-related is verified against synthetic fixtures only.** Parser coverage
numbers are real (measured on all 3,289 test rows); detection accuracy is completely unknown.

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

## Blockers

1. **Awaiting the organizers' reply.** Three answers gate a valid submission: how test rows are
   identified (the parquet has **no ID column** and the linked "template" is a validation *output*
   file), whether server-side scoring is actually live (`leaderboard.html` 404s), and which
   deadline is real. **Chase by email if there's no reply within ~a week** — the development phase
   ends early October and this cannot be left unresolved.
2. Log the answers here when they arrive, then build the submission writer against the real schema.

**Not blocked meanwhile:** the smoke test and validation run need none of this. Keep going.

## Next actions, in order

1. HF Jobs smoke test, `LIMIT=20` (~10 min). Command in README.
2. Full validation run. Read **coverage-limited** (is the geometry right?) separately from
   **as-submitted** (unsolved = 0, what the leaderboard would say).
3. Triage failures. Expect Grounding-DINO to mis-ground the cluttered "complex background" clips —
   103/159 of validation.
4. Build the **fallback arm** (VLM estimate) and fuse in log space. Until then every unsolved row
   is a hard zero.
5. Then: SAM2 masks for extent (boxes overestimate non-rectangular objects, and overestimates are
   metric-expensive), CoTracker3 where the quadratic fit quality is low.

## Notes

* Compute: **Colab Pro** to iterate, **HF Jobs `l4x1`** for batch (detached, survives a closed
  laptop), Kaggle for free sweeps. Do not buy a GPU — the whole competition is tens of GPU-hours.
* Videos are **not** stored locally (download killed at 30/568, deliberately). Colab and HF Jobs
  fetch them on the remote machine.
* `README.md` publicly documents the competitive analysis above. Trim it into a gitignored
  `NOTES.md` if that becomes a concern.
* Session of 2026-08-01 cost ~$100, mostly research. Set a budget next time.
