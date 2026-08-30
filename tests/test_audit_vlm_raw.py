"""Tests for the VLM raw-reply audit.

The audit is read-only, but its classification decides *which* fix to buy -- a truncation count
argues for one constant, a refusal count argues for a prompt rewrite, and they cost different
amounts. So the two classifiers get pinned, including the negatives, in the same spirit as the
solver's refuted-variant tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from audit_vlm_raw import (  # noqa: E402
    PUBLISHED,
    PUBLISHED_ZEROS,
    classify_zero,
    ends_cleanly,
    gate_counts,
    gate_replay,
)


class TestEndsCleanly:
    """The truncation discriminator."""

    @pytest.mark.parametrize("text", [
        "The pier is about 2.4 m wide.",
        "ANSWER: 2.4",
        "It measures 3 metres!",
        'The answer is "2.4"',
        "roughly 1.5 m (estimated)",
    ])
    def test_finished_replies_end_cleanly(self, text: str) -> None:
        assert ends_cleanly(text)

    @pytest.mark.parametrize("text", [
        "the distance is approximately 1.85 m and the boat's width appears to be roughly",
        "Estimating from the shadow, the pier is about 1.5 times the",
        "the lamps are spaced about 1.5 meters apart",
        "comparing against the door frame,",
    ])
    def test_truncated_replies_do_not(self, text: str) -> None:
        assert not ends_cleanly(text)

    def test_trailing_whitespace_is_ignored(self) -> None:
        assert ends_cleanly("It is 2.4 m.   \n")

    def test_empty_is_not_clean(self) -> None:
        # An empty reply is its own decline route; calling it "finished" would hide it in the
        # truncation bucket and overstate the case for raising max_new_tokens.
        assert not ends_cleanly("")
        assert not ends_cleanly("   \n ")

    def test_a_reply_cut_mid_number_is_the_costly_case(self) -> None:
        """This is the mechanism behind `mix-v11`'s -0.106 to -0.253/row.

        The parser's `last-number` fallback scrapes the truncated fragment, so a reply cut after
        "approximately 1." yields 1 where the sentence was going to say 1.85. Under MRA that is not
        a small error, it is most of a hard zero -- and it is why the route measured bad in all four
        categories rather than merely noisy.
        """
        cut = "Comparing with the car, the distance is approximately 1."
        assert ends_cleanly(cut), "ends on '.', so length alone cannot flag it"
        assert "ANSWER:" not in cut, "but the marker never arrived, which is what the audit counts"


class TestClassifyZero:
    """Why a reply that parsed to zero said zero. Most specific cause wins."""

    def test_refusal_beats_everything(self) -> None:
        text = "The height cannot be determined without a reference object. ANSWER: 0"
        assert classify_zero(text).startswith("refusal")

    def test_stationary_is_a_physics_claim_not_a_bug(self) -> None:
        text = "The ball is stationary throughout the clip, so its velocity is zero. ANSWER: 0"
        assert "genuinely zero" in classify_zero(text)

    def test_a_nonzero_in_the_reasoning_is_a_sig_figs_bug(self) -> None:
        text = "The gap is about 0.4 m across, which rounds to 0. ANSWER: 0"
        assert "sig-figs" in classify_zero(text)

    def test_bare_zero_has_no_stated_cause(self) -> None:
        assert "no stated cause" in classify_zero("ANSWER: 0")

    def test_refusal_wins_over_a_nonzero_number(self) -> None:
        """Ordering matters, and this is the case that sets it.

        A refusal that happens to quote the reference measurement ("the boat is 3.62 m") contains a
        non-zero number, so a sig-figs-first ordering would file 457 prompt-comprehension failures
        as rounding bugs and point the fix at the wrong thing.
        """
        text = "The boat is 3.62 m but the pier's width cannot be determined. ANSWER: 0"
        assert classify_zero(text).startswith("refusal")

    def test_stationary_wins_over_a_nonzero_number(self) -> None:
        text = "Over the 3.0 s clip the crate is at rest, so the speed is zero. ANSWER: 0"
        assert "genuinely zero" in classify_zero(text)

    def test_classification_is_total(self) -> None:
        """Every zero row lands in exactly one bucket, so the Q4 table sums to the zero count."""
        for text in ("", "ANSWER: 0", "cannot determine", "stationary", "0.4 rounds to 0"):
            assert isinstance(classify_zero(text), str)
            assert classify_zero(text)


def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestGates:
    """The gates must refuse, not warn. A spend gets justified off these numbers."""

    def test_replay_gate_passes_when_the_reparse_agrees(self) -> None:
        frame = _frame([
            {"row_index": 0, "parse_route": "sentinel", "parsed_value": 2.4,
             "route": "sentinel", "value": 2.4},
            {"row_index": 1, "parse_route": "sentinel", "parsed_value": None,
             "route": "sentinel", "value": None},
        ])
        gate_replay(frame)  # must not raise

    def test_replay_gate_fails_on_a_single_diverging_row(self) -> None:
        frame = _frame([
            {"row_index": 0, "parse_route": "sentinel", "parsed_value": 2.4,
             "route": "sentinel", "value": 2.4},
            {"row_index": 1, "parse_route": "sentinel", "parsed_value": 9.9,
             "route": "last-number", "value": 1.0},
        ])
        with pytest.raises(SystemExit):
            gate_replay(frame)

    def test_count_gate_fails_when_totals_drift(self) -> None:
        frame = _frame([{"row_index": 0, "route": "sentinel", "value": 2.4, "reason": ""}])
        with pytest.raises(SystemExit):
            gate_counts(frame)

    def test_a_zero_keeps_its_route_but_is_not_a_usable_answer(self) -> None:
        """The arithmetic the count gate exists to protect.

        `parse_answer` leaves `route == "sentinel"` on a zero and only clears the value, while the
        predictions CSV files it under `none`. Counting by route alone would report 2,464 sentinel
        answers instead of 1,619 and price the coverage hole at half its real size -- in the
        optimistic direction, which is the one that gets money spent.
        """
        rows = [{"row_index": i, "route": "sentinel", "value": None,
                 "reason": "model answered zero; declining"} for i in range(PUBLISHED_ZEROS)]
        frame = _frame(rows)
        usable = frame["value"].notna() & frame["route"].str.startswith("sentinel")
        assert int(usable.sum()) == 0
        assert PUBLISHED["vlm-sentinel"] == 1619


class TestSplitMismatchGuard:
    """The footgun tomorrow's prompt A/B would have walked into.

    A validation run has `row_index` 0..158 and so do the first 159 rows of the test predictions, so
    joining one against the other succeeds, validates one-to-one, and labels every row with an
    unrelated category. The A/B would then have been read off noise.
    """

    @staticmethod
    def _replies(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "row_index": range(n),
            "raw_text": [f"ANSWER: {i + 1}" for i in range(n)],
            "unit": ["meters"] * n,
        })

    @staticmethod
    def _predictions(tmp_path, n: int, with_row_index: bool = True):
        frame = pd.DataFrame({
            "video_type": ["S2MC"] * n,
            "inference_type": ["SS"] * n,
        })
        if with_row_index:
            frame["row_index"] = range(n)
        path = tmp_path / "predictions.csv"
        frame.to_csv(path, index=False)
        return path

    def test_a_row_count_mismatch_is_refused(self, tmp_path) -> None:
        from audit_vlm_raw import reparse

        with pytest.raises(SystemExit, match="mislabels every category"):
            reparse(self._replies(159), self._predictions(tmp_path, 3289))

    def test_partial_allows_a_deliberate_smoke_run(self, tmp_path) -> None:
        from audit_vlm_raw import reparse

        out = reparse(self._replies(20), self._predictions(tmp_path, 3289), partial=True)
        assert len(out) == 20
        assert out["video_type"].notna().all()

    def test_a_run_reaching_past_the_predictions_file_is_refused(self, tmp_path) -> None:
        from audit_vlm_raw import reparse

        replies = self._replies(10)
        replies["row_index"] = range(3280, 3290)
        with pytest.raises(SystemExit, match="wrong --predictions"):
            reparse(replies, self._predictions(tmp_path, 3289), partial=True)

    def test_a_fixture_without_row_index_gets_one_from_split_order(self, tmp_path) -> None:
        """The validation fixture carries no row_index. Same convention as make_submission.py."""
        from audit_vlm_raw import reparse

        out = reparse(self._replies(159), self._predictions(tmp_path, 159, with_row_index=False))
        assert len(out) == 159
        assert out["video_type"].notna().all()

    def test_missing_category_columns_are_refused(self, tmp_path) -> None:
        from audit_vlm_raw import reparse

        path = tmp_path / "bad.csv"
        pd.DataFrame({"row_index": range(5), "parsed_value": [1.0] * 5}).to_csv(path, index=False)
        with pytest.raises(SystemExit, match="cannot label categories"):
            reparse(self._replies(5), path)
