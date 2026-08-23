"""Tests for the VLM arm's CPU-testable surface.

Two properties are load-bearing and cheap to lose.

The module must import **without torch**. The whole suite runs on a laptop, and a top-level
``import torch`` in a backend would take that away -- which matters more than it sounds, because the
CPU suite is the only thing standing between a prompt change and a paid GPU run.

And frame selection must concentrate on the instant a question names. A question about t=1.4 s
answered from frames spanning 0-10 s is the same class of error as the global kinematics fit that
cost this project its largest measured defect.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from quantiphy.backends.vlm import (
    DEFAULT_FRAMES,
    INSTANT_WINDOW_S,
    VlmAnswer,
    choose_frame_times,
)

ROOT = Path(__file__).resolve().parent.parent
TIMES = np.arange(0, 10, 1 / 24)          # a 10 s clip at 24 fps


def test_module_imports_without_torch() -> None:
    """Run in a subprocess with torch blocked, so a stray top-level import cannot hide."""
    code = (
        "import sys;"
        "sys.modules['torch'] = None;"
        "import importlib;"
        "importlib.import_module('quantiphy.backends.vlm');"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_uniform_when_no_instant_is_named() -> None:
    """A size question is answered from whichever frame shows the object best: spread out."""
    chosen = TIMES[choose_frame_times(TIMES, None, DEFAULT_FRAMES)]
    assert len(chosen) == DEFAULT_FRAMES
    assert chosen[0] == pytest.approx(0.0)
    assert chosen[-1] == pytest.approx(TIMES[-1])


def test_concentrates_on_the_instant_when_one_is_named() -> None:
    chosen = TIMES[choose_frame_times(TIMES, 1.4, DEFAULT_FRAMES)]
    assert len(chosen) == DEFAULT_FRAMES
    assert np.abs(chosen - 1.4).max() <= INSTANT_WINDOW_S + 1e-9
    # And it is genuinely narrower than the uniform spread it replaces.
    assert np.ptp(chosen) < np.ptp(TIMES) / 4


def test_falls_back_to_uniform_when_the_window_cannot_fill_the_budget() -> None:
    """Better twelve frames from the wrong places than three from the right ones."""
    chosen = choose_frame_times(TIMES, 99.0, DEFAULT_FRAMES)
    assert len(chosen) == DEFAULT_FRAMES
    assert TIMES[chosen][-1] == pytest.approx(TIMES[-1])


def test_never_asks_for_more_frames_than_the_clip_has() -> None:
    short = TIMES[:5]
    chosen = choose_frame_times(short, None, DEFAULT_FRAMES)
    assert len(chosen) == len(short)
    assert len(set(chosen.tolist())) == len(chosen)          # and no duplicates


def test_handles_an_unreadable_clip() -> None:
    assert choose_frame_times(np.empty(0), None, DEFAULT_FRAMES).size == 0


def test_indices_are_in_range_and_sorted() -> None:
    for at_time in (None, 0.0, 1.4, 9.9, 99.0):
        chosen = choose_frame_times(TIMES, at_time, DEFAULT_FRAMES)
        assert chosen.min() >= 0 and chosen.max() < TIMES.size
        assert (np.diff(chosen) > 0).all()


def test_answer_keeps_the_reply_unparsed() -> None:
    """The raw text is the artefact; parsing it here would make prompt changes cost a GPU run."""
    answer = VlmAnswer(row_index=3, video_id="v", raw_text="ANSWER: 4.2")
    assert answer.raw_text == "ANSWER: 4.2"
    assert not hasattr(answer, "value")
