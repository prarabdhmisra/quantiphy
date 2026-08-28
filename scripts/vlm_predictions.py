"""Turn a VLM run's raw replies into a replay-shaped predictions CSV.

Deliberately emits the *same* schema ``replay_cache.py`` does -- ``row_index``, ``parsed_value``,
``method``, ``reason``, plus the dataset columns the slice keys need. That single decision means the
VLM arm inherits every instrument the geometric arm already has, unchanged:

* ``make_submission.py`` fills the official template from it and backfills declines from a constant;
* ``disagreement.py`` slices it by category, prior type, unit and route;
* ``method_ids.py`` selects the rows that took a given parse route;
* ``clip_disagreement.py`` reverts the rows that disagree wildly with the constant;
* ``select_rows.py`` overlays it onto a champion per row.

``method`` carries the parse route (``vlm-sentinel``, ``vlm-last-number``) and is ``none`` for a
declined row, which is exactly the convention ``method_ids.py`` and ``solved_ids.py`` already read.
That is worth more than it looks: on the 159 validation rows the two routes are not interchangeable,
and being able to overlay only the rows the model answered *in the demanded format* is a selection
this project can make without inventing a new confidence signal.

**Re-parses the raw text rather than copying the run's ``parsed_value``.** The raw reply is the
artefact; a parser fix has to be measurable without paying for a second pass.

Usage:
    py -3.12 scripts/vlm_predictions.py --run test-vlm-qwen3vl8b-brief --shards 4 \\
        --out vlm-v1.predictions.csv
    py -3.12 scripts/make_submission.py vlm-v1.predictions.csv --out vlm-v1.submission.csv \\
        --fallback-from baseline_predictions.csv
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

DEFAULT_REPO = "prarabdhmisra/quantiphy-runs"
TEST_PARQUET = ROOT / "data" / "fixtures" / "test_dataset.parquet"
DECLINED = "none"

#: Columns a replay carries so the slice keys work. Kept here rather than assumed, because a missing
#: `inference_type` makes `category_labels` silently mislabel every row rather than raise.
CARRIED = ("video_id", "question", "video_type", "inference_type")


def load_raw(repo: str, run: str, shards: int) -> pd.DataFrame:
    """Every reply the run produced, unioned across its shards and keyed by ``row_index``.

    A sharded run partitions *rows*, so the pieces are disjoint by construction and an overlap means
    the ``SHARD`` arithmetic is wrong -- refused rather than silently deduplicated, because the two
    replies would be answers to the same question and picking one is a coin flip. Duplicates *within*
    one shard are a resumed run appending, where last-wins is correct.
    """
    from huggingface_hub import hf_hub_download

    names = [run] if shards == 0 else [f"{run}-shard{k}" for k in range(1, shards + 1)]
    pieces = []
    for name in names:
        try:
            path = hf_hub_download(repo, repo_type="dataset", filename=f"{name}/vlm_raw.jsonl")
        except Exception as error:                                     # noqa: BLE001
            print(f"  {name}: MISSING vlm_raw.jsonl ({type(error).__name__}); its rows will read "
                  f"as declined and take the fallback")
            continue
        records = [json.loads(line) for line in
                   Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
        frame = pd.DataFrame(records).drop_duplicates("row_index", keep="last")
        print(f"  {name}: {len(frame)} replies")
        pieces.append(frame)

    if not pieces:
        raise SystemExit(f"no shard of {run} produced a vlm_raw.jsonl")

    merged = pd.concat(pieces, ignore_index=True)
    overlap = int(merged["row_index"].duplicated().sum())
    if overlap:
        raise SystemExit(f"{overlap} row_index values appear in more than one shard; the SHARD "
                         f"partition is wrong, so refusing to guess which reply to keep")
    return merged.set_index("row_index").sort_index()


def predictions(raw: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    """One record per dataset row, in ``replay_cache.py``'s schema.

    Every row of the split appears, not just the ones the run reached: a row the job never got to is
    a *declined* row that must reach the fallback, and dropping it here would leave a hole that
    ``make_submission.py`` fills silently and that no count would ever reveal.
    """
    units = dataset["question"].map(parse_output_unit).fillna("")
    records = []
    for row_index in range(len(dataset)):
        row = dataset.iloc[row_index]
        record = {"row_index": row_index,
                  **{name: row[name] for name in CARRIED},
                  "parsed_value": None, "method": DECLINED, "reason": "row never attempted",
                  "unit": units.iloc[row_index], "raw_text": ""}
        if row_index in raw.index:
            reply = raw.loc[row_index]
            text = str(reply.get("raw_text") or "")
            unit = str(reply.get("unit") or units.iloc[row_index])
            answer = parse_answer(text, unit)
            record.update(unit=unit, raw_text=text)
            if answer.ok:
                record.update(parsed_value=answer.value, method=f"vlm-{answer.route}", reason="")
            else:
                record.update(reason=answer.note or f"unparseable ({answer.route})")
        records.append(record)
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run name without the -shard<k> suffix")
    parser.add_argument("--shards", type=int, default=0, help="0 for an unsharded run")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--dataset", type=Path, default=TEST_PARQUET)
    args = parser.parse_args()

    dataset = pd.read_parquet(args.dataset).reset_index(drop=True)
    raw = load_raw(args.repo, args.run, args.shards)
    frame = predictions(raw, dataset)
    frame.drop(columns=["raw_text"]).to_csv(args.out, index=False, encoding="utf-8")

    answered = frame["method"].ne(DECLINED)
    print(f"\n{len(frame)} rows | answered {int(answered.sum())} ({answered.mean():.1%}) -> "
          f"{args.out}")
    for name, count in frame.loc[answered, "method"].value_counts().items():
        print(f"  {count:5d}  {name}")
    declines = frame.loc[~answered, "reason"].value_counts()
    print(f"declined {int((~answered).sum())}, top reasons:")
    for reason, count in declines.head(6).items():
        print(f"  {count:5d}  {reason}")
    print(f"\nnow run: py -3.12 scripts/make_submission.py {args.out} --out <sub>.csv "
          f"--fallback-from baseline_predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
