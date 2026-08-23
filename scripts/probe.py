"""Build a probe submission: the reigning champion with one group per category perturbed.

The campaign this serves rests on two properties of the scoring portal. The test set is fixed, so
two submissions differing on a known set of rows produce an *exactly* determined score difference --
there is no sampling noise to bootstrap away. And the macro is the unweighted mean of four
separately reported category means, so a change confined to one category moves only that category's
number. One submission therefore carries four independent experiments.

For a group ``g`` inside category ``C`` that is the only thing changed in ``C``::

    delta(reported C) = (n_g / n_C) * delta(mean score on g)

which inverts to give the group's mean-score change directly. ``analyze_probes.py`` does that
inversion; this script's whole job is to make the left-hand side interpretable, which means
**refusing to perturb two groups in the same category**. That guard is the reason the arithmetic
above holds, so it is an error rather than a warning.

Usage:
    py -3.12 scripts/probe.py --champion baseline-v3.submission.csv --out probe.csv \
        --probe "D2|meters=0.5" --probe "S2|meters=2.0"

A probe is ``CATEGORY|unit=multiplier`` (relative to the champion) or ``CATEGORY|unit@value`` for an
absolute constant. Always include the incumbent multiplier 1.0 somewhere in a grid: a grid that
brackets the champion can never adopt something worse than the champion, and a search that lacks
that property measured *below* its own starting point in simulation.
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
from quantiphy.scoring import CATEGORIES, category_labels  # noqa: E402


def group_keys(frame: pd.DataFrame) -> pd.Series:
    """The ``CATEGORY|unit`` label a probe addresses, one per row.

    Same two keys ``make_baseline`` groups on, joined into one string so a probe can name a group on
    the command line. Rows whose question states no unit collapse into ``?`` and are addressable
    like any other group -- there are 18 of them and they are not worth a slot, but they must not
    silently vanish either.
    """
    units = frame["question"].map(parse_output_unit).fillna("?")
    return category_labels(frame).astype(str) + "|" + units.astype(str)


class ProbeError(Exception):
    """A probe spec that would produce an uninterpretable reading."""


def parse_probe(spec: str) -> tuple[str, str, float]:
    """``"D2|meters=0.5"`` -> ``("D2|meters", "multiplier", 0.5)``."""
    for separator, kind in (("@", "absolute"), ("=", "multiplier")):
        if separator in spec:
            group, _, raw = spec.partition(separator)
            try:
                value = float(raw)
            except ValueError as error:
                raise ProbeError(f"{spec!r}: {raw!r} is not a number") from error
            if value <= 0:
                raise ProbeError(f"{spec!r}: value must be positive (a zero prediction scores 0)")
            return group.strip(), kind, value
    raise ProbeError(f"{spec!r}: expected CATEGORY|unit=multiplier or CATEGORY|unit@value")


def apply_probes(champion: pd.DataFrame, specs: list[str]) -> tuple[pd.Series, list[dict]]:
    """Perturbed ``parsed_value`` plus a manifest describing every change.

    The manifest is what makes a reading reproducible three weeks later, so it records the row count
    and the before/after value per group rather than just the spec that was asked for.
    """
    keys = group_keys(champion)
    values = pd.to_numeric(champion["parsed_value"], errors="coerce").astype(float)

    seen: dict[str, str] = {}
    manifest: list[dict] = []
    for spec in specs:
        group, kind, value = parse_probe(spec)
        category, _, _ = group.partition("|")
        if category not in CATEGORIES:
            raise ProbeError(f"{spec!r}: {category!r} is not one of {list(CATEGORIES)}")
        if category in seen:
            raise ProbeError(
                f"{spec!r} and {seen[category]!r} both perturb {category}. Two groups in one "
                f"category cannot be separated from a single reported number -- probe one per "
                f"category per submission.")
        seen[category] = spec

        selected = (keys == group).to_numpy()
        if not selected.any():
            raise ProbeError(f"{spec!r}: no rows match. Groups present: {sorted(keys.unique())}")

        before = float(values[selected].iloc[0]) if selected.sum() else float("nan")
        values.loc[selected] = value if kind == "absolute" else values.loc[selected] * value
        after = float(values[selected].iloc[0])
        manifest.append({
            "spec": spec, "group": group, "category": category, "kind": kind, "value": value,
            "rows": int(selected.sum()),
            "category_rows": int((category_labels(champion) == category).sum()),
            "before": before, "after": after,
            "ids": champion.loc[selected, "id"].astype(int).tolist(),
        })
    return values, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--champion", type=Path, required=True,
                        help="the submission this probe is measured against")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--probe", action="append", default=[], metavar="CATEGORY|unit=mult",
                        help="repeatable; at most one per category")
    args = parser.parse_args()

    if not args.probe:
        parser.error("at least one --probe is required")

    champion = pd.read_csv(args.champion, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    try:
        values, manifest = apply_probes(champion, args.probe)
    except ProbeError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    out = champion.copy()
    out["parsed_value"] = values.to_numpy()
    out.to_csv(args.out, index=False, encoding="utf-8")

    manifest_path = args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(
        {"champion": str(args.champion), "submission": str(args.out), "probes": manifest},
        indent=2), encoding="utf-8")

    print(f"{args.out}  ({len(out)} rows)")
    for entry in manifest:
        share = entry["rows"] / entry["category_rows"]
        print(f"  {entry['group']:<16} {entry['rows']:4d}/{entry['category_rows']:<4d} rows "
              f"({share:.1%} of {entry['category']})  {entry['before']:.4g} -> {entry['after']:.4g}")
        print(f"    a reported {entry['category']} shift of d maps to "
              f"{1 / share:.2f}*d on this group's mean score")
    print(f"manifest -> {manifest_path}")
    print(f"\nnow run: py -3.12 scripts/validate_submission.py {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
