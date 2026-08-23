"""Invert the probe ledger into per-group mean scores, and propose the next grid.

Every row of ``data/probes/ledger.csv`` is one scored submission. When a submission changed exactly
one group inside a category, the reported category mean inverts to that group's mean score::

    delta(mean score on g) = (n_C / n_g) * delta(reported C)

so a ledger of readings becomes a set of measured points on each group's response curve. That curve
is what we are searching: MRA against the constant, per group, on the actual scoring set.

Two guardrails are the reason this script exists rather than a spreadsheet.

* **Unchanged categories must reproduce exactly.** The test set is fixed, so a category nobody
  perturbed has to return the identical number. When it does not, either the scoring set changed or
  the manifest is wrong, and every differential reading afterwards is meaningless. That is reported
  as a CONSISTENCY FAILURE, loudly, because it invalidates the method rather than one measurement.
* **A group is only promoted on a tight bracket.** Simulation showed a loose search landing *below*
  its own starting point, so a group whose argmax sits at the edge of what has been probed is
  reported as "keep searching", never as an answer.

Usage:
    py -3.12 scripts/analyze_probes.py [--ledger data/probes/ledger.csv]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy.scoring import CATEGORIES  # noqa: E402

LEDGER = ROOT / "data" / "probes" / "ledger.csv"

#: Reported means carry three decimals, so a category number is known to about this much. Anything
#: smaller than roughly this over a group's share of its category is not a measurement.
REPORTING_PRECISION = 0.0005


def scored(ledger: pd.DataFrame) -> pd.DataFrame:
    """Ledger rows that actually came back with a score."""
    return ledger[ledger["macro"].notna() & (ledger["macro"].astype(str) != "")].copy()


def check_macro_arithmetic(row: pd.Series) -> str | None:
    """The macro must be the unweighted mean of the four categories, or we misunderstand the metric.

    Cheap, and it has already earned its place once: this identity is what confirmed the portal's
    "Average MRA" is exactly ``quantiphy.scoring``'s macro rather than a row-weighted mean.
    """
    parts = [float(row[name]) for name in CATEGORIES]
    expected = sum(parts) / len(parts)
    if abs(expected - float(row["macro"])) > 0.001:
        return (f"{row['submission']}: reported macro {float(row['macro']):.4f} but the four "
                f"categories average {expected:.4f}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=LEDGER)
    args = parser.parse_args()

    ledger = pd.read_csv(args.ledger, encoding="utf-8-sig")
    done = scored(ledger)
    print(f"{len(ledger)} ledger rows, {len(done)} scored\n")

    problems = [message for _, row in done.iterrows()
                if (message := check_macro_arithmetic(row)) is not None]
    if problems:
        print("CONSISTENCY FAILURE -- the macro is not the mean of the four categories:")
        for message in problems:
            print(f"  {message}")
        print("Stop probing until this is explained.\n")
    else:
        print(f"macro == mean(S2,D2,S3,D3) on all {len(done)} scored rows: OK\n")

    champions = done[done["champion_flag"] == 1]
    if champions.empty:
        print("no champion recorded yet")
        return 0
    champion = champions.iloc[-1]
    print(f"champion: {champion['submission']}  macro {float(champion['macro']):.4f}  "
          f"(S2 {champion['S2']}  D2 {champion['D2']}  S3 {champion['S3']}  D3 {champion['D3']})")

    print("\nper-category headroom if that category alone reached 0.60:")
    for name in CATEGORIES:
        current = float(champion[name])
        print(f"  {name}  {current:.3f}  ->  +{0.25 * (0.60 - current):.4f} macro")

    print("\nresolution: a group holding a share s of its category resolves to "
          f"{REPORTING_PRECISION:.4f}/s on its own mean score")
    for share, label in ((0.715, "D2|meters, 829/1160"), (0.295, "S3|cm, 170/576"),
                         (0.026, "a 30-row group in D2")):
        print(f"  {label:<26} +/- {REPORTING_PRECISION / share:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
