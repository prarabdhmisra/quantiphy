"""Tests for VLM prompt construction and answer extraction.

The parser is the highest-risk pure code in the VLM arm, because its failure mode is silent. An
answer misread by 100x still validates, still uploads, and scores a hard zero on every affected row.
So the unit-handling cases here are the point of this file, not an afterthought.
"""

from __future__ import annotations

import pytest

from quantiphy.parsing import build_request
from quantiphy.prompting import ANSWER_SENTINEL, build_prompt, parse_answer


def row(question: str, prior: str = "length of boat = 3.62m", **over):
    base = {"video_id": "v", "video_type": "S2MC", "fps": 24, "inference_type": "SS",
            "question": question, "ground_truth_prior": prior, "depth_info": None}
    base.update(over)
    return base


# --------------------------------------------------------------------------- parsing

@pytest.mark.parametrize("text,expected", [
    (f"{ANSWER_SENTINEL} 3.5", 3.5),
    (f"The block spans about 9 px per cm. {ANSWER_SENTINEL} 3.5", 3.5),
    (f"{ANSWER_SENTINEL}3.5", 3.5),
    (f"{ANSWER_SENTINEL} 12,000", 12000.0),
    (f"{ANSWER_SENTINEL} 2e3", 2000.0),
    (f"{ANSWER_SENTINEL} .75", 0.75),
    (f"{ANSWER_SENTINEL} -4", 4.0),           # scorer takes the magnitude
])
def test_reads_the_sentinel(text, expected) -> None:
    assert parse_answer(text, "meters").value == pytest.approx(expected)


def test_falls_back_to_the_last_number_but_says_so() -> None:
    """Models often comply in substance while dropping the marker. Usable, but flagged."""
    parsed = parse_answer("It looks like roughly 4.8 metres across.", "meters")
    assert parsed.value == pytest.approx(4.8)
    assert parsed.route == "last-number"


@pytest.mark.parametrize("text", ["", "   ", "I cannot determine this from the frames.",
                                  "unable to measure"])
def test_declines_when_there_is_no_number(text) -> None:
    assert not parse_answer(text, "meters").ok


def test_declines_on_zero_rather_than_emitting_it() -> None:
    """A zero prediction scores a hard zero and still counts, so the fallback should get the row."""
    parsed = parse_answer(f"{ANSWER_SENTINEL} 0", "meters")
    assert not parsed.ok
    assert "zero" in parsed.note


def test_converts_a_compatible_unit_the_model_volunteered() -> None:
    """The silent-100x case. The scorer does no unit conversion, so we must."""
    parsed = parse_answer(f"{ANSWER_SENTINEL} 4.2 cm", "meters")
    assert parsed.value == pytest.approx(0.042)
    assert "converted" in parsed.route


def test_refuses_a_unit_of_the_wrong_dimension() -> None:
    """A speed answered into a length question is not convertible and must not be guessed at."""
    parsed = parse_answer(f"{ANSWER_SENTINEL} 5 m/s", "meters")
    assert not parsed.ok
    assert "not a length" in parsed.note


def test_leaves_a_matching_unit_alone() -> None:
    parsed = parse_answer(f"{ANSWER_SENTINEL} 7.5 meters", "meters")
    assert parsed.value == pytest.approx(7.5)
    assert "converted" not in parsed.route


def test_takes_the_last_sentinel_when_the_model_repeats_itself() -> None:
    text = f"{ANSWER_SENTINEL} 1.0\nOn reflection that is wrong.\n{ANSWER_SENTINEL} 2.0"
    assert parse_answer(text, "meters").value == pytest.approx(2.0)


# --------------------------------------------------------------------------- prompting

def test_prompt_states_the_prior_the_unit_and_the_envelope() -> None:
    """The three things the paper says VLMs get wrong, made explicit."""
    request = build_request(row("What is the width of the pier in meters?"))
    prompt = build_prompt(request, "length of boat = 3.62m", None)
    assert "length of boat = 3.62m" in prompt
    assert "meters" in prompt
    assert ANSWER_SENTINEL in prompt
    assert "What is the width of the pier in meters?" in prompt


def test_prompt_passes_the_prior_through_verbatim() -> None:
    """Not re-rendered from the parsed objects: that parser is lossy on purpose."""
    request = build_request(row("What is the length of the wood block in cm?",
                               prior="ruler calibre = 1 cm"))
    assert "ruler calibre = 1 cm" in build_prompt(request, "ruler calibre = 1 cm", None)


def test_prompt_names_the_instant_when_the_question_does() -> None:
    request = build_request(row("What is the speed of the boat at 1.0s in m/s?"))
    assert request.timestamp is not None
    assert "t = 1 s" in build_prompt(request, "length of boat = 3.62m", None)


def test_prompt_includes_depth_info_only_when_present() -> None:
    request = build_request(row("What is the height of the person in meters?"))
    assert "Camera distances" not in build_prompt(request, "p = 1m", None)
    assert "Camera distances" in build_prompt(request, "p = 1m", "distance_human_camera = 12.26m")


def test_prompt_does_not_ask_the_model_to_answer_low() -> None:
    """Measured on GPT-5.1's answers: shrinking a good predictor loses at every factor."""
    prompt = build_prompt(build_request(row("What is the width of the pier in meters?")),
                          "length of boat = 3.62m", None).lower()
    for nudge in ("underestimate", "err low", "smaller", "conservative"):
        assert nudge not in prompt
