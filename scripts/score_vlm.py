"""Score a VLM run offline from its raw replies, and compare two runs on the same rows.

``run_vlm_job.py`` writes the model's reply *unparsed* to ``<run>/vlm_raw.jsonl``. That is the whole
point of the file: parsing, unit handling and fusion are pure functions, so every question about them
is answerable for free against replies already paid for -- the same discipline that makes
``replay_cache.py`` the cheapest instrument on this project.

So this re-parses rather than trusting the run's own ``parsed_value`` column. A parser fix must be
measurable without a second GPU pass, and if the two disagree the raw text is authoritative.

On the validation split there is ground truth, so this prints real MRA, and a second ``--run`` is
scored on the *intersection* of rows both answered and compared with a paired bootstrap. Reading two
runs over different row sets was how a 0.007 difference once looked like a lever.

The declined-row policy matters as much as the score. A blank, a NaN and a zero are hard zeros under
this metric and still count in the category mean, so ``--fallback`` fills unparseable rows with the
zero-vision constant, which is what a real submission would do. Without it the number on screen
measures the parser as much as the model.

Usage:
    py -3.12 scripts/score_vlm.py --run validation-vlm-qwen3vl8b-brief
    py -3.12 scripts/score_vlm.py --run validation-vlm-qwen3vl8b-brief \\
        --run validation-vlm-qwen3vl8b-direct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy.parsing import parse_output_unit  # noqa: E402
from quantiphy.prompting import parse_answer  # noqa: E402
from quantiphy.scoring import CATEGORIES, category_labels, paired_bootstrap, score  # noqa: E402

DEFAULT_REPO = "prarabdhmisra/quantiphy-runs"
VALIDATION = ROOT / "data" / "fixtures" / "quantiphy_validation.csv"


def load_raw(repo: str, run: str) -> pd.DataFrame:
    """One row per reply, from the run's ``vlm_raw.jsonl`` on the Hub."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo, repo_type="dataset", filename=f"{run}/vlm_raw.jsonl")
    records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        raise SystemExit(f"{run}/vlm_raw.jsonl is empty; the job wrote no replies")
    frame = pd.DataFrame(records)
    duplicated = int(frame["row_index"].duplicated().sum())
    if duplicated:
        # A resumed run appends, so a duplicate means the checkpoint replayed rows. Last wins, and
        # saying so is better than a silently doubled denominator.
        print(f"  note: {duplicated} duplicate row_index values; keeping the last of each")
        frame = frame.drop_duplicates("row_index", keep="last")
    return frame.set_index("row_index").sort_index()


def reparse(raw: pd.DataFrame, units: pd.Series) -> pd.DataFrame:
    """Re-read every reply with the current parser, into ``value``/``route``/``note``."""
    rows = []
    for index, record in raw.iterrows():
        unit = record.get("unit") or units.get(index) or ""
        parsed = parse_answer(str(record.get("raw_text") or ""), str(unit))
        rows.append({"row_index": index, "value": parsed.value, "route": parsed.route,
                     "parse_note": parsed.note, "unit": unit})
    return pd.DataFrame(rows).set_index("row_index")


def evaluate(truth: pd.DataFrame, parsed: pd.DataFrame,
             fallback: pd.Series | None) -> tuple[pd.DataFrame, int]:
    """The scorable frame for the rows the run reached, and how many were filled.

    Restricted to rows the run actually answered, which matters for a run still in flight: scoring a
    partial run over the full denominator would read every unreached row as a hard zero and measure
    the crash rather than the model. Scoring is left to the caller because a partial run can be
    legitimately *unscorable* -- the official evaluator reports no average at all when a category is
    empty, and saying which category is missing is more useful than a bare traceback.
    """
    frame = truth.loc[truth.index.intersection(parsed.index)].copy()
    frame["parsed_value"] = parsed["value"].reindex(frame.index)
    filled = 0
    if fallback is not None:
        blank = frame["parsed_value"].isna()
        filled = int(blank.sum())
        frame.loc[blank, "parsed_value"] = fallback.reindex(frame.index[blank]).to_numpy()
    return frame, filled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="run name; give twice to compare two runs")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--validation", type=Path, default=VALIDATION)
    parser.add_argument("--no-fallback", action="store_true",
                        help="score unparseable rows as hard zeros instead of filling them")
    args = parser.parse_args()

    if len(args.run) > 2:
        parser.error("at most two runs; a paired bootstrap compares exactly two")

    truth = pd.read_csv(args.validation, encoding="utf-8-sig")
    truth = truth[truth["ground_truth_posterior"].notna()].reset_index(drop=True)
    units = truth["question"].map(parse_output_unit).fillna("")

    fallback = None
    if not args.no_fallback:
        # The median truth for the row's own (category, unit) group -- the same ladder
        # make_baseline.py uses, and what a real submission would put in a declined row.
        values = pd.to_numeric(truth["ground_truth_posterior"], errors="coerce")
        keys = [category_labels(truth), units]
        fallback = values.groupby(keys).transform("median")

    results, frames = {}, {}
    for run in args.run:
        print(f"\n=== {run} ===")
        raw = load_raw(args.repo, run)
        parsed = reparse(raw, units)
        frame, filled = evaluate(truth, parsed, fallback)
        print(f"  {len(frame)} of {len(truth)} rows answered; "
              f"{int(parsed['value'].notna().sum())} parsed"
              + (f", {filled} filled from the constant" if filled else ""))
        print(f"  routes: {dict(parsed['route'].value_counts())}")
        try:
            result = score(frame)
        except ValueError as error:
            # A run still in flight, most likely. Not an error worth a traceback: the rows it has
            # are real, there is just no macro average until all four categories are present.
            print(f"  NOT SCORABLE YET: {error}")
            continue
        results[run], frames[run] = result, frame
        print(f"  macro MRA {result.macro_mra:.4f}   invalid {result.invalid_fraction:.1%}")
        print("  " + "  ".join(f"{name} {result.per_category[name]:.4f} "
                               f"(n={result.counts[name]})" for name in CATEGORIES))

    if len(args.run) == 2 and len(frames) == 2:
        first, second = args.run
        shared = frames[first].index.intersection(frames[second].index)
        print(f"\n=== {second} against {first}, on the {len(shared)} rows both answered ===")
        if len(shared) < len(frames[first]) or len(shared) < len(frames[second]):
            print("  scored on the intersection only, so these numbers differ from the two above")
        rescored = {run: score(frames[run].loc[shared]) for run in args.run}
        for run in args.run:
            print(f"  {run}: {rescored[run].macro_mra:.4f}")
        verdict = paired_bootstrap(rescored[first], rescored[second],
                                   category_labels(frames[first].loc[shared]))
        print(f"\n  {verdict}")
        print("\n  Accept only if the 95% interval excludes zero -- 159 rows carry a +-5.7 point CI,"
              "\n  and this split has already refuted three levers that looked real in-sample.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
