"""Tests for the offline replay harness.

The replay's entire claim to authority is that it is *the same code on the same detections* as the
GPU run it reproduces. So the properties pinned here are the ones that, if they broke silently,
would let a wrong number justify a real submission slot: the shard caches union rather than
overwrite, an empty cached series declines with the reason the real backend gives it rather than a
blank, a raised row is recorded instead of aborting the pass, and the reproduction gate fails loudly
on any difference beyond rounding.

`load` itself is not tested: it is two lines of `hf_hub_download` and a `read_parquet`, and mocking
the Hub would test the mock. The reproduction gate is what actually guards that path, and it runs
against the real cache every time the harness is used.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from quantiphy.backends.grounding import DetectionSeries  # noqa: E402
from replay_cache import (  # noqa: E402
    SOLVER_V1,
    CachedBackend,
    band,
    check_reproduction,
    replay,
)


def series(count: int = 6) -> DetectionSeries:
    """A detection series that measures cleanly: 6 frames, a 100 px box drifting right."""
    times = np.linspace(0.0, 0.25, count)
    return DetectionSeries(
        times=times, cx=np.linspace(200.0, 300.0, count), cy=np.full(count, 150.0),
        width=np.full(count, 100.0), height=np.full(count, 60.0),
        scores=np.full(count, 0.8), frames_total=count, frames_sampled=count,
    )


def empty_series() -> DetectionSeries:
    """What the detector caches when it ran and found nothing -- not the same as a cache miss."""
    blank = np.array([])
    return DetectionSeries(times=blank, cx=blank, cy=blank, width=blank, height=blank,
                          scores=blank, frames_total=48, frames_sampled=48)


def rows(**overrides) -> pd.DataFrame:
    """One test-split row, in the parquet's column shape."""
    row = {
        "video_id": "simulation_0007", "video_source": "simulation", "video_type": "S2MC",
        "fps": 24, "inference_type": "SS", "question": "What is the width of the pier in meters?",
        "prior": "length of boat = 3.62m", "depth_info": None,
    }
    row.update(overrides)
    return pd.DataFrame([row])


class TestBand:
    """`--trusted-prior-pixels` is how the band gets re-fitted, so its parsing is load-bearing."""

    def test_parses_a_pair(self):
        assert band("30,300") == (30.0, 300.0)

    def test_accepts_an_open_upper_edge(self):
        low, high = band("30,inf")
        assert (low, high) == (30.0, float("inf"))

    @pytest.mark.parametrize("text", ["30", "30,300,400", "a,b", ""])
    def test_rejects_malformed_input(self, text):
        with pytest.raises(argparse.ArgumentTypeError):
            band(text)

    def test_rejects_an_inverted_band(self):
        # Silently swapping the edges would produce a band that rejects everything, and a replay
        # that solves zero rows reads as a solver collapse rather than a typo in the flag.
        with pytest.raises(argparse.ArgumentTypeError):
            band("300,30")


class TestCachedBackend:
    def test_rekeys_the_jobs_absolute_paths_by_basename(self):
        # The cache is keyed by /root/.cache/... paths that never exist on this machine.
        backend = CachedBackend({("/root/.cache/hub/xyz/simulation_0007.mp4", "boat"): series()})
        assert ("simulation_0007.mp4", "boat") in backend.series

    def test_an_empty_series_is_a_decline_not_a_miss(self):
        """The 41-row bug: an empty cached series is a *cached answer*, not a missing one.

        Without the explicit note these rows declined with a blank reason and scattered across the
        tail of the histogram the next fix is chosen from -- while the real backend reports them as
        one 33-row bucket.
        """
        backend = CachedBackend({("/j/simulation_0007.mp4", "boat"): empty_series()})
        request = _request(rows().iloc[0])
        measured = backend.measure(request, "cache/simulation_0007.mp4", "boat", "length")

        assert measured.note == "object never detected"
        assert measured.frames_tracked == 0
        assert backend.misses == []

    def test_a_genuine_miss_is_recorded(self):
        backend = CachedBackend({})
        request = _request(rows().iloc[0])
        measured = backend.measure(request, "cache/simulation_0007.mp4", "boat", "length")

        assert measured.note == "not in the detection cache"
        assert backend.misses == [("simulation_0007.mp4", "boat")]


class TestReplay:
    def test_emits_the_gpu_prediction_schema(self):
        """`make_submission.py` and `solved_ids.py` both key off these columns."""
        results = replay(rows(), CachedBackend(_cache()))

        for column in ("row_index", "video_id", "question", "video_type", "inference_type",
                       "parsed_value", "method", "reason", "confidence"):
            assert column in results.columns
        assert results["row_index"].tolist() == [0]

    def test_records_the_prior_and_target_pixels(self):
        # These are absent from the GPU schema and are what the band is re-fitted against, so a
        # replay that dropped them would look fine and be useless.
        results = replay(rows(), CachedBackend(_cache()))
        assert {"prior_pixels", "target_pixels"} <= set(results.columns)

    def test_solves_a_well_formed_row(self):
        results = replay(rows(), CachedBackend(_cache()))
        assert results["parsed_value"].notna().all()
        assert results["method"].iloc[0] != "none"

    def test_the_band_is_forwarded_to_the_solver(self):
        """Without this the `--trusted-prior-pixels` sweep would silently measure one band."""
        wide = replay(rows(), CachedBackend(_cache()), trusted_prior_pixels=(0.0, float("inf")))
        shut = replay(rows(), CachedBackend(_cache()), trusted_prior_pixels=(1e9, 2e9))

        assert wide["parsed_value"].notna().all()
        assert shut["parsed_value"].isna().all()
        assert "outside the trusted" in shut["reason"].iloc[0]

    def test_a_raised_row_is_recorded_rather_than_losing_the_pass(self):
        class Exploding(CachedBackend):
            def measure(self, *args, **kwargs):
                raise RuntimeError("detector exploded")

        results = replay(rows(), Exploding(_cache()))

        assert len(results) == 1
        assert results["method"].iloc[0] == "none"
        assert "RuntimeError: detector exploded" in results["reason"].iloc[0]


class TestCheckReproduction:
    def test_the_recorded_run_carries_its_own_band(self):
        """The gate must never read the solver's live default.

        `TRUSTED_PRIOR_PIXELS` has already moved once, to (30, inf), *because* of what this run
        measured. A gate that compared against the live default would have begun failing at the
        moment its own finding was adopted -- so the band the run actually used is recorded beside
        its numbers, and the gate replays at that.
        """
        low, high = SOLVER_V1["band"]
        assert low < high
        assert (low, high) == (30.0, 300.0)

    def test_accepts_the_recorded_solver_v1_numbers(self, capsys):
        assert check_reproduction(_solver_v1_shaped()) is True
        assert "reproduces solver-v1 exactly" in capsys.readouterr().out

    def test_fails_on_a_single_row_difference(self, capsys):
        """A hard gate, not a note.

        One row is well inside anything that looks like noise, and there is no noise here -- the
        replay is deterministic. A harness that quietly disagreed with its own run would go on to
        print solve rates that justify spending a slot.
        """
        results = _solver_v1_shaped()
        first = results.index[results["parsed_value"].notna()][0]
        results.loc[first, "parsed_value"] = None

        assert check_reproduction(results) is False
        captured = capsys.readouterr().out
        assert "DOES NOT REPRODUCE" in captured
        assert "The harness is wrong, not the solver" in captured


def _request(row):
    from quantiphy.parsing import build_request
    return build_request(row)


def _cache() -> dict:
    prefix = "/root/.cache/huggingface/hub/snap/simulation_0007.mp4"
    return {(prefix, "boat"): series(), (prefix, "pier"): series()}


def _solver_v1_shaped() -> pd.DataFrame:
    """A results frame with solver-v1's exact per-category counts, and nothing else real.

    Built from the recorded rates rather than restating them, so the fixture cannot drift away from
    the constant the gate checks against.
    """
    counts = {"S2": (581, 247), "D2": (1160, 330), "S3": (576, 441), "D3": (972, 320)}
    assert sum(solved for _, solved in counts.values()) == SOLVER_V1["solved"]

    records = []
    for category, (total, solved) in counts.items():
        inference, dimension = category[0], category[1]
        for index in range(total):
            records.append({
                "row_index": len(records),
                # category_labels reads inference_type[0] and video_type[1].
                "inference_type": f"{inference}S",
                "video_type": f"S{dimension}MC",
                "parsed_value": 1.0 if index < solved else None,
                "reason": "" if index < solved else "gravity prior cannot set pixel scale",
            })
    return pd.DataFrame(records)
