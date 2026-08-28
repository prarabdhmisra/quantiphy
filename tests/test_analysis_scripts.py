"""Tests for the two instruments built on 2026-08-26: the disagreement table and the clip.

Both exist because a throwaway analysis chose a lever once and could not be re-run. So the property
that matters most here is not that they compute a median -- it is that they *refuse* the inputs that
would produce a confident, wrong, unreproducible reading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from clip_disagreement import clip, parse_clip  # noqa: E402
from disagreement import add_keys, ratios, table  # noqa: E402
import method_ids  # noqa: E402
import score_vlm  # noqa: E402
import vlm_predictions  # noqa: E402

from quantiphy import scoring  # noqa: E402


def predictions(**overrides) -> pd.DataFrame:
    """Four rows, one per scored category, all answered. `inference_type[0]` + `video_type[1]`
    is what `category_labels` builds the S2/D2/S3/D3 label from."""
    frame = pd.DataFrame({
        "row_index": [0, 1, 2, 3],
        "question": ["What is the width of the box in meters?"] * 4,
        "video_type": ["S2SX", "V2SX", "S3SX", "A3SX"],
        "inference_type": ["SS", "DS", "SS", "DS"],
        "parsed_value": [1.0, 1.0, 1.0, 1.0],
        "method": ["geometric-2d", "geometric-2d", "geometric-3d+radial", "geometric-3d"],
        "reason": ["", "", "", ""],
    })
    return frame.assign(**overrides)


def constants(values=(1.0, 1.0, 1.0, 1.0)) -> pd.Series:
    return pd.Series(list(values), index=pd.Index([1, 2, 3, 4], name="id"),
                     name="parsed_value")


# --- disagreement -------------------------------------------------------------------------------

def test_keys_split_the_prior_type_off_video_type():
    keyed = add_keys(predictions())
    assert list(keyed["category"]) == ["S2", "D2", "S3", "D3"]
    assert list(keyed["prior"]) == ["S", "V", "S", "A"]


def test_route_collapses_method_modifiers():
    # `geometric-3d+radial` and `geometric-3d` are one code path. Splitting them fragments a slice
    # the same way counting raw decline strings once hid the trusted band in ~300 messages.
    assert list(add_keys(predictions())["route"]) == [
        "geometric-2d", "geometric-2d", "geometric-3d", "geometric-3d"]


def test_declined_rows_are_excluded_from_the_ratio():
    frame = predictions(method=["none", "geometric-2d", "geometric-2d", "geometric-2d"],
                        parsed_value=[np.nan, 2.0, 2.0, 2.0])
    answered = ratios(frame, constants().reset_index())
    # Including a declined row would compare the constant with itself and drag the median to 1.0.
    assert len(answered) == 3
    assert answered["ratio"].tolist() == [2.0, 2.0, 2.0]


def test_ratio_refuses_ids_absent_from_the_baseline():
    with pytest.raises(SystemExit, match="different splits"):
        ratios(predictions(), constants().iloc[:2].reset_index())


def test_fold_distance_is_symmetric():
    # A 10x undershoot and a 10x overshoot are both hard zeros under MRA, so the tail columns must
    # count them alike even though the metric punishes them differently on the way there.
    frame = predictions(parsed_value=[10.0, 0.1, 1.0, 1.0])
    result = table(ratios(frame, constants().reset_index()), ["route"])
    row = result.loc[result["route"] == "geometric-2d"].iloc[0]
    assert row["n"] == 2
    assert row["within2x"] == 0.0
    assert row["within10x"] == 1.0


# --- clip ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("spec", ["D3=1.0", "D3=0.5"])
def test_clip_refuses_a_threshold_at_or_below_one(spec):
    # At 1.0 every answered row reverts and the result is the pure constant -- already measured, so
    # the slot would buy nothing while looking like a new source.
    with pytest.raises(SystemExit, match="must exceed 1.0"):
        parse_clip(spec)


def test_clip_refuses_an_unknown_category():
    with pytest.raises(SystemExit, match="not one of"):
        parse_clip("D4=5")


def test_clip_reverts_only_rows_past_the_threshold():
    frame = predictions(parsed_value=[1.0, 20.0, 1.0, 1.0])
    out, manifest = clip(frame, constants(), {"D2": 10.0})
    assert out.loc[1, "method"] == "none"
    assert pd.isna(out.loc[1, "parsed_value"])
    # Every other row, including the other categories, is untouched.
    assert out.loc[[0, 2, 3], "method"].tolist() == [
        "geometric-2d", "geometric-3d+radial", "geometric-3d"]
    assert manifest[0]["rows"] == 1 and manifest[0]["ids"] == [2]


def test_clip_catches_an_undershoot_as_well_as_an_overshoot():
    frame = predictions(parsed_value=[1.0, 0.02, 1.0, 1.0])
    out, _ = clip(frame, constants(), {"D2": 10.0})
    assert out.loc[1, "method"] == "none"


def test_clip_refuses_a_spec_that_changes_nothing():
    # A no-op clip validates cleanly, uploads cleanly, and spends a daily slot re-measuring a
    # submission we already have a score for.
    with pytest.raises(SystemExit, match="clips no rows"):
        clip(predictions(), constants(), {"D2": 10.0})


def test_clipped_rows_carry_a_reason_naming_the_threshold():
    frame = predictions(parsed_value=[1.0, 20.0, 1.0, 1.0])
    out, _ = clip(frame, constants(), {"D2": 10.0})
    assert "10" in out.loc[1, "reason"]


# --- method_ids ---------------------------------------------------------------------------------

def test_method_ids_selects_by_substring_and_emits_ids_not_row_indices():
    selected = method_ids.select(predictions(), contains="radial")
    assert list(selected["id"]) == [3]                      # row_index 2 -> id 3
    assert list(selected["category"]) == ["S3"]


def test_method_ids_exact_match_does_not_pick_up_the_modified_variants():
    """`geometric-3d` as a substring also matches `geometric-3d+radial`, and in D3 those two sit at
    2.29x and 0.75x the constant. Selecting one and silently getting both is the whole failure mode
    this script exists to avoid."""
    assert list(method_ids.select(predictions(), methods=("geometric-3d",))["id"]) == [4]
    assert list(method_ids.select(predictions(), contains="geometric-3d")["id"]) == [3, 4]


def test_method_ids_excludes_declined_rows():
    frame = predictions(method=["none", "", "geometric-2d", "geometric-2d"])
    assert list(method_ids.select(frame, contains="geometric")["id"]) == [3, 4]


def test_method_ids_restricts_to_the_named_categories():
    assert list(method_ids.select(predictions(), contains="geometric-2d",
                                 categories=("D2",))["id"]) == [2]


def test_method_ids_refuses_an_empty_selection():
    """An empty overlay composes to a byte-identical copy of the base: it validates, uploads, and
    spends one of three daily slots measuring nothing."""
    with pytest.raises(SystemExit):
        method_ids.select(predictions(), contains="no-such-route")


def test_method_ids_refuses_an_unknown_category():
    with pytest.raises(SystemExit):
        method_ids.select(predictions(), contains="geometric", categories=("D4",))


# --- score_vlm ----------------------------------------------------------------------------------

def vlm_truth() -> pd.DataFrame:
    """Four validation-shaped rows, one per scored category."""
    return pd.DataFrame({
        "question": ["What is the width of the box in meters?"] * 4,
        "video_type": ["S2SX", "V2SX", "S3SX", "A3SX"],
        "inference_type": ["SS", "DS", "SS", "DS"],
        "ground_truth_posterior": [2.0, 2.0, 2.0, 2.0],
    })


def vlm_raw(**overrides) -> pd.DataFrame:
    frame = pd.DataFrame({
        "row_index": [0, 1, 2, 3],
        "raw_text": ["ANSWER: 2.0", "ANSWER: 2.0", "ANSWER: 2.0", "ANSWER: 2.0"],
        "unit": ["meters"] * 4,
    }).assign(**overrides)
    return frame.set_index("row_index")


def test_score_vlm_reparses_the_raw_text_rather_than_trusting_parsed_value():
    """The raw reply is the artefact. A parser fix has to be measurable without a second GPU pass,
    so a stale `parsed_value` in the jsonl must not be able to shadow it."""
    raw = vlm_raw(raw_text=["ANSWER: 2.0"] * 4, parsed_value=[99.0] * 4)
    parsed = score_vlm.reparse(raw, pd.Series(["meters"] * 4))
    assert list(parsed["value"]) == [2.0] * 4
    assert set(parsed["route"]) == {"sentinel"}


def test_score_vlm_scores_only_the_rows_the_run_reached():
    """A half-finished run must not be scored over the full denominator: every unreached row would
    read as a hard zero and the number would measure the crash, not the model."""
    frame, filled = score_vlm.evaluate(
        vlm_truth(), score_vlm.reparse(vlm_raw().iloc[:2], pd.Series(["meters"] * 2)), None)
    assert len(frame) == 2
    assert filled == 0
    # ...and two categories are then empty, which the official evaluator treats as no average at
    # all rather than a partial one. Left to the caller so it can name the missing category.
    with pytest.raises(ValueError, match="no scorable rows"):
        scoring.score(frame)


def test_score_vlm_fills_an_unparseable_row_from_the_constant():
    """A blank scores a hard zero and still counts, so a real submission would fall back. Scoring
    without the fallback measures the parser as much as the model."""
    raw = vlm_raw(raw_text=["ANSWER: 2.0", "I cannot tell", "ANSWER: 2.0", "ANSWER: 2.0"])
    parsed = score_vlm.reparse(raw, pd.Series(["meters"] * 4))
    fallback = pd.Series([5.0] * 4)
    frame, filled = score_vlm.evaluate(vlm_truth(), parsed, fallback)
    result = scoring.score(frame)
    assert filled == 1
    assert frame["parsed_value"].tolist() == [2.0, 5.0, 2.0, 2.0]
    assert result.invalid_fraction == 0.0            # nothing blank reaches the scorer


def test_score_vlm_keeps_the_last_reply_when_a_resumed_run_duplicated_a_row(tmp_path, monkeypatch):
    """A resumed run appends, so a duplicate row_index means the checkpoint replayed rows. Counting
    both would double that row in the denominator."""
    path = tmp_path / "vlm_raw.jsonl"
    path.write_text('{"row_index": 0, "raw_text": "ANSWER: 1.0", "unit": "meters"}\n'
                    '{"row_index": 0, "raw_text": "ANSWER: 7.0", "unit": "meters"}\n',
                    encoding="utf-8")
    monkeypatch.setattr(score_vlm, "hf_hub_download", lambda *a, **k: str(path), raising=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        type("M", (), {"hf_hub_download": staticmethod(lambda *a, **k: str(path))}))
    raw = score_vlm.load_raw("repo", "run")
    assert len(raw) == 1 and raw.iloc[0]["raw_text"] == "ANSWER: 7.0"


def test_score_vlm_refuses_an_empty_run(tmp_path, monkeypatch):
    path = tmp_path / "vlm_raw.jsonl"
    path.write_text("", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        type("M", (), {"hf_hub_download": staticmethod(lambda *a, **k: str(path))}))
    with pytest.raises(SystemExit):
        score_vlm.load_raw("repo", "run")


# --- vlm_predictions ----------------------------------------------------------------------------

def vlm_dataset() -> pd.DataFrame:
    return pd.DataFrame({
        "video_id": ["v0", "v1", "v2", "v3"],
        "question": ["What is the width of the box in meters?"] * 4,
        "video_type": ["S2SX", "V2SX", "S3SX", "A3SX"],
        "inference_type": ["SS", "DS", "SS", "DS"],
    })


def vlm_replies(records) -> pd.DataFrame:
    return pd.DataFrame(records).set_index("row_index").sort_index()


def test_vlm_predictions_emits_a_row_for_every_dataset_row_not_just_answered_ones():
    """A row the job never reached is a *declined* row that must reach the fallback. Dropping it
    leaves a hole make_submission fills silently and no count ever reveals."""
    frame = vlm_predictions.predictions(
        vlm_replies([{"row_index": 0, "raw_text": "ANSWER: 2.0", "unit": "meters"}]),
        vlm_dataset())
    assert len(frame) == 4
    assert list(frame["method"]) == ["vlm-sentinel", "none", "none", "none"]
    assert list(frame.loc[1:, "reason"]) == ["row never attempted"] * 3


def test_vlm_predictions_records_the_parse_route_as_the_method():
    """So method_ids.py can overlay only the rows answered in the demanded format, without anyone
    inventing a new confidence signal."""
    frame = vlm_predictions.predictions(
        vlm_replies([{"row_index": 0, "raw_text": "ANSWER: 2.0", "unit": "meters"},
                     {"row_index": 1, "raw_text": "about 3.5", "unit": "meters"}]),
        vlm_dataset())
    assert list(frame.loc[:1, "method"]) == ["vlm-sentinel", "vlm-last-number"]
    assert list(method_ids.select(frame, contains="sentinel")["id"]) == [1]


def test_vlm_predictions_declines_a_zero_and_says_so():
    """A zero is a hard zero that still counts, so it must reach the fallback rather than be
    emitted. 38 of 42 fallbacks on the 2026-08-27 validation run were exactly this."""
    frame = vlm_predictions.predictions(
        vlm_replies([{"row_index": 0, "raw_text": "It is stationary.\nANSWER: 0",
                      "unit": "meters"}]), vlm_dataset())
    assert frame.loc[0, "method"] == "none"
    assert frame.loc[0, "parsed_value"] is None
    assert "zero" in frame.loc[0, "reason"]


def test_vlm_predictions_reparses_rather_than_copying_the_runs_own_value():
    frame = vlm_predictions.predictions(
        vlm_replies([{"row_index": 0, "raw_text": "ANSWER: 2.0", "unit": "meters",
                      "parsed_value": 99.0}]), vlm_dataset())
    assert frame.loc[0, "parsed_value"] == pytest.approx(2.0)


def test_vlm_predictions_refuses_shards_that_overlap(tmp_path, monkeypatch):
    """A sharded run partitions rows, so an overlap means the SHARD arithmetic is wrong. The two
    replies answer the same question and picking one is a coin flip."""
    def fake_download(repo, repo_type, filename):
        path = tmp_path / filename.replace("/", "_")
        path.write_text('{"row_index": 0, "raw_text": "ANSWER: 1", "unit": "m"}\n', encoding="utf-8")
        return str(path)
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        type("M", (), {"hf_hub_download": staticmethod(fake_download)}))
    with pytest.raises(SystemExit, match="more than one shard"):
        vlm_predictions.load_raw("repo", "run", shards=2)
