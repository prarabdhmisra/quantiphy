"""Tests for the probe builder.

The campaign's arithmetic assumes each submission perturbs at most one group per category, and that
the perturbed rows are exactly the intended group. Both are properties of ``probe.py``, and if
either silently breaks then every subsequent reading is a difference against an unknown baseline --
so they are pinned here rather than trusted.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from probe import ProbeError, apply_probes, group_keys, parse_probe  # noqa: E402

CHAMPION = ROOT / "baseline-v3.submission.csv"


@pytest.fixture(scope="module")
def champion() -> pd.DataFrame:
    if not CHAMPION.exists():
        pytest.skip(f"{CHAMPION.name} not built")
    return pd.read_csv(CHAMPION, dtype=str, keep_default_na=False, encoding="utf-8-sig")


@pytest.mark.parametrize("spec,expected", [
    ("D2|meters=0.5", ("D2|meters", "multiplier", 0.5)),
    ("S2|m/s@3.25", ("S2|m/s", "absolute", 3.25)),
])
def test_parses_both_probe_forms(spec, expected) -> None:
    assert parse_probe(spec) == expected


@pytest.mark.parametrize("spec", ["D2|meters", "D2|meters=nope", "D2|meters=0", "D2|meters=-1"])
def test_rejects_unusable_specs(spec) -> None:
    with pytest.raises(ProbeError):
        parse_probe(spec)


def test_refuses_two_groups_in_one_category(champion) -> None:
    """The guard the whole inversion depends on."""
    with pytest.raises(ProbeError, match="both perturb D2"):
        apply_probes(champion, ["D2|meters=0.5", "D2|m/s=2.0"])


def test_rejects_a_group_nobody_matches(champion) -> None:
    with pytest.raises(ProbeError, match="no rows match"):
        apply_probes(champion, ["D2|furlongs=0.5"])


def test_changes_exactly_the_named_group(champion) -> None:
    """A manifest that disagrees with the file would corrupt every later reading."""
    before = pd.to_numeric(champion["parsed_value"], errors="coerce").to_numpy()
    values, manifest = apply_probes(champion, ["D2|meters=0.5"])
    changed = ~pd.Series(values.to_numpy() == before)

    assert len(manifest) == 1
    assert int(changed.sum()) == manifest[0]["rows"]
    expected = (group_keys(champion) == "D2|meters").to_numpy()
    assert (changed.to_numpy() == expected).all()
    assert set(champion.loc[expected, "id"].astype(int)) == set(manifest[0]["ids"])


def test_multiplier_is_applied_exactly(champion) -> None:
    values, manifest = apply_probes(champion, ["S3|cm=0.5"])
    entry = manifest[0]
    assert entry["after"] == pytest.approx(entry["before"] * 0.5)


def test_absolute_form_overwrites_rather_than_scaling(champion) -> None:
    values, manifest = apply_probes(champion, ["S3|cm@7.0"])
    assert manifest[0]["after"] == pytest.approx(7.0)


def test_four_categories_can_be_probed_at_once(champion) -> None:
    """Four independent experiments per submission is the campaign's whole throughput argument."""
    _, manifest = apply_probes(champion, [
        "S2|meters=0.5", "D2|meters=0.5", "S3|cm=0.5", "D3|meters=0.5"])
    assert {entry["category"] for entry in manifest} == {"S2", "D2", "S3", "D3"}
