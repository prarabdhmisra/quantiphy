# Draft email to the organizers

**To:** quantiphybench@gmail.com
**Subject:** QuantiPhy Challenge — submission template, scoring status, and deadline clarification

Send this in week 1. Questions 1–3 are blockers: we cannot make a valid first submission without
answers, and one of them may mean scoring is not live yet.

---

Hi,

I'm registering a two-person team for the QuantiPhy Challenge and working through the starter kit.
A few things I couldn't resolve from the site, the README, or the dataset cards.

**1. Test-set submission template and row IDs.** The competition page links
`model_outputs/gpt-5.1.csv` as the reference template, but that file is a *validation* output — it
has 159 rows and includes a `ground_truth_posterior` column. Meanwhile `test_dataset.parquet` has
no ID column at all, and the upload page says to "leave `id` unchanged" while its accepted ID
aliases (`id`, `item_id`, `sample_id`, `uid`, `qid`, `question_id`, `index`) don't include
`video_id`. Could you publish the actual test-set submission template, or confirm how test rows
should be identified — is it the 0-based row index of `test_dataset.parquet`?

**2. Is server-side scoring live?** The competition page describes a real-time leaderboard, but
`/competition/leaderboard.html` returns 404 and I wanted to confirm that uploads are currently
being scored end to end before we build against the portal.

**3. Which deadline is authoritative?** The countdown timer, the GitHub README, and the Hugging
Face card all state November 5, 2026, 23:59 AOE. The timeline on the same page puts "Verification
and final ranking" in mid-to-late October, and the NeurIPS call for competitions asks that
competitions complete by end of October. We'd like to plan against the right date.

**4. Rules we couldn't find.** The published materials don't address these either way, and we'd
rather ask than assume:
- Is **fine-tuning** on the provided validation split permitted?
- Is **external training data** (public datasets, self-collected video) permitted?
- Are **ensembles** of multiple models permitted in either track?
- For the Open-Weight Track, do **gated** public weights (e.g. HF models behind a license
  click-through) count as "publicly available"?
- The full test set — videos and questions — is a public CC-BY-4.0 download with only the answer
  column withheld. Is there a rule against **manual human annotation** of test items being
  submitted as predictions? We're not planning to do this and think it would undermine the
  benchmark's long-term value, but the rules are silent and it seemed worth raising.

**5. Team eligibility.** The rules say "one team per person, per track," which reads as allowing
different teams across the two tracks, while the registration page says each person may belong to
only one active team. Which applies?

**6. Report and workshop.** Are top teams invited to co-author the competition report, or only to
submit short technical summaries? And is there any travel support or a virtual presentation option
for the December workshop?

Thanks very much — happy to help test the submission portal if that's useful.

Best,
Prarabdh Misra
