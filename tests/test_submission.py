"""Guards the submission file itself.

A submission can be perfectly modelled and still score nothing if its rows drift out of alignment
with the organizers' template, and there is no server-side dry run to catch that. These tests are
the only thing standing between a good run and a wasted one.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
TEMPLATE = ROOT / "data" / "fixtures" / "quantiphy_submission_template.csv"

TEST_ROWS = 3289


def _load(name: str):
    """Import a `scripts/` module by path -- they are entry points, not an installed package."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


make_submission = _load("make_submission")
validate_submission = _load("validate_submission")


@pytest.fixture(scope="module")
def template() -> pd.DataFrame:
    return make_submission.load_template(TEMPLATE)


@pytest.fixture(scope="module")
def predictions(template: pd.DataFrame) -> pd.DataFrame:
    """A run that solved every third row, keyed the way run_vision_job.py keys its output."""
    solved = template.index[::3]
    return pd.DataFrame({
        "row_index": solved,
        "parsed_value": [1.0 + index for index in range(len(solved))],
    })


def _write(frame: pd.DataFrame, path: Path) -> Path:
    frame.to_csv(path, index=False, encoding="utf-8")
    return path


# --- the template itself -------------------------------------------------------------------

def test_template_is_the_official_test_split(template: pd.DataFrame) -> None:
    """id is 1..3289 in order, and parsed_value is empty -- not a validation output."""
    assert len(template) == TEST_ROWS
    assert list(template.columns) == list(validate_submission.TEMPLATE_COLUMNS) + ["parsed_value"]
    assert template["id"].astype(int).tolist() == list(range(1, TEST_ROWS + 1))
    assert (template["parsed_value"] == "").all()


def test_template_matches_the_test_parquet_row_for_row(template: pd.DataFrame) -> None:
    """`id` is the parquet row index plus one. Everything downstream assumes this."""
    parquet = pd.read_parquet(ROOT / "data" / "fixtures" / "test_dataset.parquet")

    assert len(parquet) == len(template)
    for column in ("video_id", "video_source", "video_type", "inference_type", "question"):
        assert template[column].tolist() == parquet[column].astype(str).tolist()
    assert template["ground_truth_prior"].tolist() == parquet["prior"].astype(str).tolist()


# --- building a submission -----------------------------------------------------------------

def test_build_fills_every_row_and_preserves_the_template(tmp_path: Path,
                                                          predictions: pd.DataFrame,
                                                          template: pd.DataFrame) -> None:
    # Arrange
    path = _write(predictions, tmp_path / "preds.csv")

    # Act
    submission = make_submission.build(TEMPLATE, path, shrink=1.0)

    # Assert
    assert len(submission) == TEST_ROWS
    assert list(submission.columns) == list(template.columns)
    for column in validate_submission.TEMPLATE_COLUMNS:
        assert submission[column].tolist() == template[column].tolist()
    values = pd.to_numeric(submission["parsed_value"])
    assert values.notna().all() and (values > 0).all(), "a blank or zero is a hard zero"


def test_predictions_land_on_their_own_id_regardless_of_order(tmp_path: Path,
                                                              predictions: pd.DataFrame) -> None:
    """Row order in the predictions file must not matter; only the id decides placement."""
    # Arrange
    shuffled = predictions.sample(frac=1.0, random_state=0)

    # Act
    ordered = make_submission.build(TEMPLATE, _write(predictions, tmp_path / "a.csv"), 1.0)
    scrambled = make_submission.build(TEMPLATE, _write(shuffled, tmp_path / "b.csv"), 1.0)

    # Assert
    assert ordered["parsed_value"].tolist() == scrambled["parsed_value"].tolist()


def test_shrink_scales_only_the_fallback_rows(tmp_path: Path, predictions: pd.DataFrame) -> None:
    """Solved rows are our measurement and must pass through untouched."""
    # Arrange
    path = _write(predictions, tmp_path / "preds.csv")
    solved_ids = set(predictions["row_index"] + 1)

    # Act
    full = pd.to_numeric(make_submission.build(TEMPLATE, path, 1.0)["parsed_value"])
    half = pd.to_numeric(make_submission.build(TEMPLATE, path, 0.5)["parsed_value"])

    # Assert
    template_ids = pd.to_numeric(make_submission.load_template(TEMPLATE)["id"])
    is_solved = template_ids.isin(solved_ids).to_numpy()
    assert full[is_solved].equals(half[is_solved])
    assert (half[~is_solved] < full[~is_solved]).all()


def test_zero_predictions_are_treated_as_unsolved(tmp_path: Path) -> None:
    """Zero scores exactly as badly as a blank, so it must be replaced, not carried through."""
    # Arrange
    frame = pd.DataFrame({"row_index": [0, 1, 2], "parsed_value": [0.0, 5.0, None]})
    path = _write(frame, tmp_path / "preds.csv")

    # Act
    submission = make_submission.build(TEMPLATE, path, shrink=1.0)

    # Assert
    values = pd.to_numeric(submission["parsed_value"])
    assert (values > 0).all()
    assert values.iloc[1] == 5.0


def test_build_rejects_out_of_range_ids(tmp_path: Path) -> None:
    # Arrange
    frame = pd.DataFrame({"id": [1, TEST_ROWS + 1], "parsed_value": [1.0, 2.0]})
    path = _write(frame, tmp_path / "preds.csv")

    # Act / Assert
    with pytest.raises(ValueError, match="outside"):
        make_submission.build(TEMPLATE, path, shrink=1.0)


def test_build_refuses_a_run_that_solved_nothing(tmp_path: Path) -> None:
    """Without a single solved row there is no basis for a fallback, and every row would score 0."""
    # Arrange
    frame = pd.DataFrame({"row_index": [0, 1], "parsed_value": [None, 0.0]})
    path = _write(frame, tmp_path / "preds.csv")

    # Act / Assert
    with pytest.raises(ValueError, match="no solved rows"):
        make_submission.build(TEMPLATE, path, shrink=1.0)


# --- validating a submission ---------------------------------------------------------------

@pytest.fixture(scope="module")
def good_submission(tmp_path_factory, predictions: pd.DataFrame) -> Path:
    directory = tmp_path_factory.mktemp("submission")
    built = make_submission.build(TEMPLATE, _write(predictions, directory / "preds.csv"), 1.0)
    return _write(built, directory / "submission.csv")


def test_validator_accepts_a_well_formed_submission(good_submission: Path) -> None:
    assert validate_submission.check(good_submission, TEST_ROWS, TEMPLATE) == []


@pytest.mark.parametrize("corrupt, expected", [
    (lambda f: f.drop(columns=["id"]), "id"),
    (lambda f: f.assign(id=[2] + f["id"].tolist()[1:]), "duplicated"),
    (lambda f: f.iloc[::-1], "ascending"),
    (lambda f: f.iloc[:-1], "rows"),
    (lambda f: f.assign(video_id="wrong"), "video_id"),
    (lambda f: f.assign(parsed_value=0.0), "zero"),
])
def test_validator_rejects_corrupted_submissions(good_submission: Path, tmp_path: Path,
                                                 corrupt, expected: str) -> None:
    """Each of these would upload cleanly and score near nothing. Prove the validator catches them."""
    # Arrange
    frame = pd.read_csv(good_submission, dtype=str, keep_default_na=False)
    path = _write(corrupt(frame), tmp_path / "broken.csv")

    # Act
    problems = validate_submission.check(path, TEST_ROWS, TEMPLATE)

    # Assert
    assert problems, "corruption slipped through the validator"
    assert any(expected in problem for problem in problems), problems


# --- where a declined row's number comes from ------------------------------------------------

def test_fallback_defaults_to_the_run_s_own_solved_median(tmp_path: Path,
                                                          predictions: pd.DataFrame) -> None:
    """The 2026-08-23 defect, pinned: without --fallback-from the run fills its own gaps.

    That is not a neutral choice. It re-applies whatever bias the solver has to every row the
    solver declined -- 59% of the set on the first real solver submission.
    """
    # Arrange
    path = _write(predictions, tmp_path / "preds.csv")

    # Act
    values = pd.to_numeric(make_submission.build(TEMPLATE, path, 1.0)["parsed_value"])

    # Assert
    own = pd.to_numeric(predictions["parsed_value"])
    solved = set(predictions["row_index"])
    declined = [index for index in range(len(values)) if index not in solved]
    filled = values.iloc[declined]
    # Every fallback is a median of some subset of this run's own values, so it cannot escape their
    # range, and there is one per (category, unit) group rather than one per row.
    assert filled.min() >= own.min() and filled.max() <= own.max()
    assert filled.nunique() <= 40


def test_fallback_from_fills_declined_rows_from_the_named_file(tmp_path: Path,
                                                               template: pd.DataFrame,
                                                               predictions: pd.DataFrame) -> None:
    # Arrange
    path = _write(predictions, tmp_path / "preds.csv")
    constant = _write(pd.DataFrame({"id": pd.to_numeric(template["id"]), "parsed_value": 7.5}),
                      tmp_path / "constant.csv")

    # Act
    values = pd.to_numeric(make_submission.build(TEMPLATE, path, 1.0, constant)["parsed_value"])

    # Assert
    solved = set(predictions["row_index"])
    declined = [index for index in range(len(values)) if index not in solved]
    assert (values.iloc[declined] == 7.5).all()
    assert (values.iloc[sorted(solved)] != 7.5).all(), "solved rows must keep the solver's value"


def test_fallback_from_falls_through_to_the_ladder_where_the_file_is_blank(
        tmp_path: Path, template: pd.DataFrame, predictions: pd.DataFrame) -> None:
    """A partial constants file must not leave a hard zero behind."""
    # Arrange
    partial = pd.DataFrame({"id": pd.to_numeric(template["id"]), "parsed_value": 7.5})
    partial.loc[partial.index[1:], "parsed_value"] = 0.0
    path = _write(predictions, tmp_path / "preds.csv")

    # Act
    values = pd.to_numeric(make_submission.build(
        TEMPLATE, path, 1.0, _write(partial, tmp_path / "partial.csv"))["parsed_value"])

    # Assert
    assert (values > 0).all()
    assert values.nunique() > 1


def test_shrink_applies_to_an_external_fallback_too(tmp_path: Path, template: pd.DataFrame,
                                                    predictions: pd.DataFrame) -> None:
    # Arrange
    path = _write(predictions, tmp_path / "preds.csv")
    constant = _write(pd.DataFrame({"id": pd.to_numeric(template["id"]), "parsed_value": 7.5}),
                      tmp_path / "constant.csv")

    # Act
    values = pd.to_numeric(make_submission.build(TEMPLATE, path, 0.5, constant)["parsed_value"])

    # Assert
    solved = set(predictions["row_index"])
    declined = [index for index in range(len(values)) if index not in solved]
    assert values.iloc[declined].eq(3.75).all()
