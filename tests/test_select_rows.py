"""Tests for the per-row source selector and the solved-id list it consumes.

Per-row selection is the one composition the campaign cannot check arithmetically: a per-category
mix is exact and needs no slot, but a per-row mix is a new source that costs a real submission to
measure. So the properties that make it readable afterwards -- the overlay lands on exactly the
listed ids, nothing else moves, and the metadata stays template-identical -- are pinned here rather
than eyeballed once at build time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from select_rows import compose, manifest, read_ids  # noqa: E402
from solved_ids import solved_ids  # noqa: E402

COLUMNS = ["id", "video_id", "video_source", "video_type", "fps", "inference_type", "question",
           "ground_truth_prior", "depth_info", "parsed_value"]


def submission(values: list[str], tag: str) -> pd.DataFrame:
    """Four rows, one per category, so category bookkeeping is exercised."""
    rows = [
        ("S2MC", "SS", "width in meters"),
        ("V2SX", "DS", "speed in m/s"),
        ("S3MC", "SS", "height in meters"),
        ("A3SX", "DS", "acceleration in m/s^2"),
    ]
    return pd.DataFrame({
        "id": [str(index + 1) for index in range(4)],
        "video_id": [f"{tag}_clip" for _ in rows],
        "video_source": ["simulation"] * 4,
        "video_type": [row[0] for row in rows],
        "fps": ["24"] * 4,
        "inference_type": [row[1] for row in rows],
        "question": [row[2] for row in rows],
        "ground_truth_prior": ["length of boat = 3.62m"] * 4,
        "depth_info": [""] * 4,
        "parsed_value": values,
    })[COLUMNS]


@pytest.fixture
def base() -> pd.DataFrame:
    return submission(["1.0", "2.0", "3.0", "4.0"], "base")


@pytest.fixture
def overlay() -> pd.DataFrame:
    return submission(["10.0", "20.0", "30.0", "40.0"], "overlay")


def test_overlays_exactly_the_listed_ids(base, overlay) -> None:
    out = compose(base, overlay, [2, 4])
    assert list(out["parsed_value"]) == ["1.0", "20.0", "3.0", "40.0"]


def test_metadata_always_comes_from_the_base(base, overlay) -> None:
    """The submission must stay row-for-row identical to the organizers' template."""
    out = compose(base, overlay, [1, 2, 3, 4])
    metadata = [column for column in COLUMNS if column != "parsed_value"]
    assert out[metadata].equals(base[metadata])


def test_rejects_a_misaligned_overlay(base, overlay) -> None:
    shuffled = overlay.iloc[::-1].reset_index(drop=True)
    with pytest.raises(SystemExit, match="row-aligned"):
        compose(base, shuffled, [1])


def test_rejects_ids_the_base_does_not_have(base, overlay) -> None:
    with pytest.raises(SystemExit, match="not in the base submission"):
        compose(base, overlay, [1, 99])


def test_manifest_counts_match_the_file(base, overlay, tmp_path) -> None:
    out = compose(base, overlay, [1, 2])
    record = manifest(Path("base.csv"), Path("overlay.csv"), tmp_path / "out.csv", out, [1, 2])
    assert record["rows"] == 2
    assert record["by_category"]["S2"]["from_overlay"] == 1
    assert record["by_category"]["D2"]["from_overlay"] == 1
    assert record["by_category"]["S3"]["from_overlay"] == 0
    assert sum(entry["category_rows"] for entry in record["by_category"].values()) == len(out)


def test_an_empty_id_list_is_refused(tmp_path) -> None:
    """It would copy the base byte for byte and spend a slot re-measuring a known number."""
    path = tmp_path / "ids.csv"
    path.write_text("id\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="empty"):
        read_ids(path)


def test_duplicate_ids_are_refused(tmp_path) -> None:
    path = tmp_path / "ids.csv"
    path.write_text("id\n1\n1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicate"):
        read_ids(path)


def test_solved_ids_shifts_row_index_to_id_and_drops_declines() -> None:
    """``id == row_index + 1``; ``method == 'none'`` is exactly a decline."""
    merged = pd.DataFrame({
        "row_index": [0, 1, 2, 3],
        "method": ["geometric-2d", "none", "geometric-3d+radial", None],
        "parsed_value": [1.0, 2.0, 3.0, 4.0],
    })
    result = solved_ids(merged)
    assert list(result["id"]) == [1, 3]
    assert list(result["method"]) == ["geometric-2d", "geometric-3d+radial"]


def test_solved_ids_refuses_overlapping_shards() -> None:
    merged = pd.DataFrame({"row_index": [0, 0], "method": ["geometric-2d", "geometric-2d"]})
    with pytest.raises(SystemExit, match="overlap"):
        solved_ids(merged)
