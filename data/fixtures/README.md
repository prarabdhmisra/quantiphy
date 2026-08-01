# Test fixtures

Small files committed so the test suite runs without network access. Both are redistributed here
under their original licences, with attribution to the QuantiPhy authors (Stanford University).

| File | Source | Licence |
|---|---|---|
| `gpt-5.1_validation.csv` | [`Paulineli/QuantiPhy`](https://github.com/Paulineli/QuantiPhy) `model_outputs/gpt-5.1.csv` | MIT |
| `published_results.csv` | same repo, `mra_results/all_model_results.csv` | MIT |
| `test_dataset.parquet` | [`PaulineLi/QuantiPhy`](https://huggingface.co/datasets/PaulineLi/QuantiPhy) | CC-BY-4.0 |

`test_dataset.parquet` holds **question metadata only** — the answer column is withheld by the
organizers and is not present here.

These pin two things we rely on:

* `gpt-5.1_validation.csv` + `published_results.csv` let `tests/test_scoring.py` prove our scorer
  reproduces the organizers' published macro MRA of 0.4856. Without that anchor, every measured
  improvement is unverifiable.
* `test_dataset.parquet` lets `tests/test_solver_core.py` assert parser coverage across all 3,289
  real test rows, so a parsing regression fails locally rather than silently costing points.

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
