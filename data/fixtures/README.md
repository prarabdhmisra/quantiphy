# Test fixtures

Small files committed so the test suite runs without network access. Both are redistributed here
under their original licences, with attribution to the QuantiPhy authors (Stanford University).

| File | Source | Licence |
|---|---|---|
| `gpt-5.1_validation.csv` | [`Paulineli/QuantiPhy`](https://github.com/Paulineli/QuantiPhy) `model_outputs/gpt-5.1.csv` | MIT |
| `published_results.csv` | same repo, `mra_results/all_model_results.csv` | MIT |
| `test_dataset.parquet` | [`PaulineLi/QuantiPhy`](https://huggingface.co/datasets/PaulineLi/QuantiPhy) | CC-BY-4.0 |
| `quantiphy_submission_template.csv` | `quantiphy.stanford.edu/competition/eval/` | CC-BY-4.0 |

`test_dataset.parquet` holds **question metadata only** — the answer column is withheld by the
organizers and is not present here.

## The submission template is the contract

Retrieved **2026-08-02**, 739,113 bytes,
SHA-256 `1d46a24a3c97723f24ac5051282f67e915a75f16214c8f3ff5a0b4d7bd23a93b`.

**Re-verified 2026-08-05** after the organizers wrote that they had "updated the webpage with a new
template": same hash, same 739,113 bytes, and both the template and `competition/index.html` still
report `Last-Modified: Sun, 02 Aug 2026 13:29:16 GMT`. Their message refers to the update already
pinned here. The contract has not moved.

**Re-verified again 2026-08-22**: unchanged on all three counts. Checked because the submission
portal was found to be live (`competition/auth/account.html`), which makes the template a live
contract rather than a future one — the uploader parses `id` and `parsed_value` and rejects nothing
else, so a column drift here would be silent.

This replaces the `model_outputs/gpt-5.1.csv` the competition page used to link, which was a
*validation* output (159 rows, with a `ground_truth_posterior` column) and not a submission format
at all. The organizers posted this after we asked, on 2026-08-02.

Verified, not assumed — `tests/test_submission.py` re-checks all of it:

* 3,289 rows; `id` is **1-based, contiguous and ascending**, so `id == parquet row index + 1`.
* Every shared column matches `test_dataset.parquet` **row-for-row with zero mismatches**;
  `ground_truth_prior` is identical to the parquet's `prior`.
* `parsed_value` is empty in all 3,289 rows.

Pin it, because they have already changed the linked "template" once. Re-check with:

```bash
py -3.12 -c "
import hashlib, urllib.request
url='https://quantiphy.stanford.edu/competition/eval/quantiphy_submission_template.csv'
print(hashlib.sha256(urllib.request.urlopen(url, timeout=120).read()).hexdigest())
"
```

A different hash means the contract moved — diff it before trusting any existing submission.

These pin three things we rely on:

* `gpt-5.1_validation.csv` + `published_results.csv` let `tests/test_scoring.py` prove our scorer
  reproduces the organizers' published macro MRA of 0.4856. Without that anchor, every measured
  improvement is unverifiable.
* `test_dataset.parquet` lets `tests/test_solver_core.py` assert parser coverage across all 3,289
  real test rows, so a parsing regression fails locally rather than silently costing points.
* `quantiphy_submission_template.csv` is what `scripts/make_submission.py` fills and what
  `scripts/validate_submission.py` compares against row-for-row. There is no server-side dry run,
  so this local comparison is the only thing that catches a misaligned submission.

Refresh with:

```bash
py -3.12 -c "
import urllib.request
B='https://raw.githubusercontent.com/Paulineli/QuantiPhy/main/'
for src,dst in [('model_outputs/gpt-5.1.csv','data/fixtures/gpt-5.1_validation.csv'),
                ('mra_results/all_model_results.csv','data/fixtures/published_results.csv')]:
    open(dst,'wb').write(urllib.request.urlopen(B+src,timeout=120).read())
"
hf download PaulineLi/QuantiPhy --type dataset --include "test_dataset.parquet" --local-dir data/fixtures
```

Citation:

> Li, P., Xiang, T., Mao, E., Wei, S., Chen, X., Masood, A., Fei-Fei, L., Adeli, E.
> *QuantiPhy: A Quantitative Benchmark Evaluating Physical Reasoning Abilities of Vision-Language
> Models.* arXiv:2512.19526.
