"""Build the final submission CSV from a solver run.

The organizers' template (`data/fixtures/quantiphy_submission_template.csv`) is the contract: 3,289
rows, ten columns, `id` running 1..3289 in order, and `parsed_value` empty. This script fills that
one column and changes nothing else -- the metadata columns are carried through as raw strings so
the file we upload is byte-identical to theirs everywhere except the predictions.

`id` is the 0-based parquet row index plus one, confirmed column-by-column against
`data/fixtures/test_dataset.parquet`. `scripts/run_vision_job.py` keys its output on `row_index`,
so the join is `id == row_index + 1`.

`--fallback-from` decides what an unsolved row gets. Without it the fallback is the median of
*this run's own solved values*, which quietly re-applies the solver's bias to every row the solver
declined -- on 2026-08-23 that put 2.7439 on 830 D2 rows where the validation-derived constant says
1.2500, and the measured cost was large. Point it at `make_baseline.py`'s output to fill declined
rows from ground truth instead.

Usage:
    py -3.12 scripts/make_submission.py predictions.csv --out submission.csv \
        --fallback-from baseline_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy.parsing import parse_output_unit  # noqa: E402
from quantiphy.scoring import category_labels  # noqa: E402

TEMPLATE = ROOT / "data" / "fixtures" / "quantiphy_submission_template.csv"

#: Written into `parsed_value`. Six significant figures is far below the metric's resolution -- the
#: loosest MRA threshold credits a 90% relative error -- and keeps the file well under the 2 MB cap.
VALUE_FORMAT = "%.6g"


def load_template(path: Path) -> pd.DataFrame:
    """Read the template with every column as a raw string, so nothing is silently reformatted."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def prediction_ids(predictions: pd.DataFrame) -> pd.Series:
    """The template `id` each prediction row belongs to.

    Accepts either an explicit `id` column or the `row_index` emitted by the vision job, which is
    the 0-based parquet position and therefore one less than `id`.
    """
    if "id" in predictions.columns:
        return pd.to_numeric(predictions["id"], errors="coerce")
    if "row_index" in predictions.columns:
        return pd.to_numeric(predictions["row_index"], errors="coerce") + 1
    raise KeyError("predictions need an 'id' or a 'row_index' column to join on")


def fallback_values(template: pd.DataFrame, solved: pd.Series, shrink: float) -> pd.Series:
    """A per-row stand-in for every unsolved question.

    Blank, NaN and zero are all hard zeros in the official metric, so *any* plausible number beats
    an empty cell. Rows are grouped by scored category and by the unit the question asks for --
    metres, m/s and m/s^2 live on completely different scales, so a single global number would be
    wrong for most of them -- and each group takes the median of whatever that group actually
    solved, widening to the category and then to the whole run when a group solved nothing.

    The median needs no log transform: it is invariant under any monotone rescaling, so the
    geometric and ordinary medians of a positive sample are the same number. (The log-space rule in
    the README bites on *means*, which are not.)

    `shrink` scales the result down. Overshooting is fatal and undershooting is cheap -- 1.9x truth
    scores 0 while 0.5x still scores 0.4 -- so a shrunk fallback should beat a centred one. How far
    to shrink is an empirical question we have not measured yet, so the default is 1.0 (no shrink)
    rather than a guess baked in silently.
    """
    usable = solved[solved.notna() & (solved > 0)]
    if usable.empty:
        raise ValueError("no solved rows to build a fallback from; every prediction would be 0")

    units = template["question"].map(parse_output_unit)
    groups = category_labels(template) + "|" + units.fillna("?")

    by_group = usable.groupby(groups[usable.index]).median()
    by_category = usable.groupby(category_labels(template)[usable.index]).median()
    overall = float(usable.median())

    filled = (groups.map(by_group)
              .fillna(category_labels(template).map(by_category))
              .fillna(overall))
    return filled.astype(float) * shrink


def external_fallback(template: pd.DataFrame, path: Path) -> pd.Series:
    """Per-row fallback values read from a separate predictions file, aligned to the template.

    The point is provenance, not arithmetic. A fallback derived from the run being submitted is a
    function of that run's own errors; one read from a file fitted on the organizers' validation
    truth is not. Rows the file leaves blank, zero or negative fall through to the median ladder.
    """
    frame = pd.read_csv(path, encoding="utf-8-sig")
    if "parsed_value" not in frame.columns:
        raise KeyError(f"{path} is missing the 'parsed_value' column")
    values = pd.to_numeric(frame["parsed_value"], errors="coerce")
    values = values.where(values.notna() & (values > 0))
    by_id = pd.Series(values.to_numpy(), index=prediction_ids(frame).to_numpy())
    return pd.Series(pd.to_numeric(template["id"]).map(by_id).to_numpy(dtype=float),
                     index=template.index)


def build(template_path: Path, predictions_path: Path, shrink: float,
          fallback_from: Path | None = None) -> pd.DataFrame:
    template = load_template(template_path)
    predictions = pd.read_csv(predictions_path, encoding="utf-8-sig")
    if "parsed_value" not in predictions.columns:
        raise KeyError("predictions are missing the 'parsed_value' column")

    ids = prediction_ids(predictions)
    valid_ids = pd.to_numeric(template["id"])
    unknown = ids[~ids.isin(valid_ids)]
    if not unknown.empty:
        raise ValueError(f"{len(unknown)} prediction ids are outside 1..{len(template)}, "
                         f"e.g. {unknown.head(3).tolist()}")
    if ids.duplicated().any():
        raise ValueError(f"{int(ids.duplicated().sum())} duplicated prediction ids")

    values = pd.to_numeric(predictions["parsed_value"], errors="coerce")
    # Zero scores exactly as badly as a blank, so treat it as unsolved and let it take a fallback.
    values = values.where(values.notna() & (values != 0))

    # Positional order in `predictions` is irrelevant; only the id decides where a value lands.
    by_id = pd.Series(values.to_numpy(), index=ids.to_numpy())
    solved = pd.Series(valid_ids.map(by_id).to_numpy(dtype=float), index=template.index)

    ladder = fallback_values(template, solved, shrink)
    if fallback_from is None:
        fallback = ladder
    else:
        external = external_fallback(template, fallback_from) * shrink
        fallback = external.fillna(ladder)
    filled = solved.fillna(fallback)

    submission = template.copy()
    submission["parsed_value"] = [VALUE_FORMAT % value for value in filled]

    covered = int(solved.notna().sum())
    source = "this run's own solved median" if fallback_from is None else fallback_from.name
    print(f"{len(submission)} rows | solved {covered} ({100 * covered / len(submission):.1f}%) | "
          f"fallback {len(submission) - covered} from {source}")
    for name, group in solved.groupby(category_labels(template)):
        print(f"  {name}: {int(group.notna().sum())}/{len(group)} solved")
    return submission


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="solver output with parsed_value + row_index")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=TEMPLATE)
    parser.add_argument("--shrink", type=float, default=1.0,
                        help="multiplier on fallback values; <1 trades overshoots for undershoots")
    parser.add_argument("--fallback-from", type=Path, default=None,
                        help="predictions file to fill declined rows from, e.g. make_baseline.py's "
                             "output; without it the fallback is this run's own solved median")
    args = parser.parse_args()

    submission = build(args.template, args.predictions, args.shrink, args.fallback_from)
    submission.to_csv(args.out, index=False, encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB)")
    print(f"now run: py -3.12 scripts/validate_submission.py {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
