"""Tests for VLM prompt construction and answer extraction.

The parser is the highest-risk pure code in the VLM arm, because its failure mode is silent. An
answer misread by 100x still validates, still uploads, and scores a hard zero on every affected row.
So the unit-handling cases here are the point of this file, not an afterthought.
"""

from __future__ import annotations

import pytest

from quantiphy.parsing import build_request
from quantiphy.prompting import ANSWER_SENTINEL, build_prompt, parse_answer, system_prompt


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


# --------------------------------------------------- 2026-08-27: the CoT switch

def test_the_default_style_still_asks_for_brief_reasoning() -> None:
    """`brief` is what every prompt_sha recorded so far used. Changing the default silently would
    make a new run non-comparable with an old one while the run name stayed the same."""
    request = build_request(row("What is the width of the pier in meters?"))
    assert "reasoning" in build_prompt(request, "length of boat = 3.62m", None)
    assert build_prompt(request, "length of boat = 3.62m", None) == build_prompt(
        request, "length of boat = 3.62m", None, style="brief")


def test_the_direct_style_asks_for_no_reasoning_at_all() -> None:
    """The paper's Table 2 puts CoT at 56.1 -> 27.7, 49.8 -> 22.4, 50.1 -> 23.1 against
    video+prior: it roughly halves MRA for every strong model. So the no-reasoning variant is the
    challenger, not a micro-optimisation."""
    request = build_request(row("What is the width of the pier in meters?"))
    direct = build_prompt(request, "length of boat = 3.62m", None, style="direct")
    assert "reasoning" not in direct and "Reason" not in direct
    assert ANSWER_SENTINEL in direct                    # the sentinel is the parser's only anchor
    assert "Reason" not in system_prompt("direct")
    assert "Reason briefly" in system_prompt("brief")


def test_both_styles_keep_the_prior_the_unit_and_the_instant() -> None:
    """Whatever else changes, the three things the benchmark turns on must survive both styles."""
    request = build_request(row("What is the speed of the car at 1.0s in m/s?"))
    for style in ("brief", "direct"):
        prompt = build_prompt(request, "length of boat = 3.62m", "distance_car_camera = 9m",
                              style=style)
        assert "3.62m" in prompt and "m/s" in prompt
        assert "t = 1 s" in prompt and "Camera distances" in prompt


def test_an_unknown_style_is_refused_rather_than_silently_defaulted() -> None:
    """A typo in the env var must not quietly produce a run that measures the incumbent again and
    reports it as the challenger."""
    request = build_request(row("What is the width of the pier in meters?"))
    with pytest.raises(ValueError):
        build_prompt(request, "length of boat = 3.62m", None, style="chain-of-thought")
    with pytest.raises(ValueError):
        system_prompt("verbose")


# ------------------------------- 2026-08-30: the `strict` style, from measured defects

def test_strict_forbids_declining_because_457_replies_declined() -> None:
    """The single biggest addressable defect found in the 8B's 3,289 test replies.

    457 of them said the quantity "cannot be determined without a reference" while the prompt was
    supplying one, concentrated in D2 (218) and D3 (171) -- the two categories the VLM otherwise
    wins. So `strict` has to close that door in the system turn, where the reference is introduced.
    """
    system = system_prompt("strict").lower()
    assert "always give a number" in system
    assert "never reply that the quantity cannot be determined" in system
    assert "never answer zero" in system


def test_strict_asks_for_two_significant_figures_because_118_rounded_to_zero() -> None:
    """118 replies computed a real value and answered 0 -- a sub-metre magnitude rounded to an int.

    Under MRA a zero is not a small answer, it is a hard zero, scoring the same as a 100x overshoot.
    """
    prompt = build_prompt(build_request(row("What is the width of the pier in meters?")),
                          "length of boat = 3.62m", None, style="strict")
    assert "two significant figures" in prompt
    assert "never 0" in prompt
    assert ANSWER_SENTINEL in prompt


def test_strict_is_not_the_downward_nudge_the_module_refuses() -> None:
    """Forbidding a zero constrains the format; it must not bias the estimate.

    `prompting`'s own docstring rules out asking the model to answer low, because shrinking a good
    per-row predictor was measured on GPT-5.1's 159 validation answers and loses monotonically at
    every factor from 0.95 down. A format rule is allowed; a direction is not.
    """
    text = (build_prompt(build_request(row("What is the width of the pier in meters?")),
                         "length of boat = 3.62m", None, style="strict")
            + " " + system_prompt("strict")).lower()
    for nudge in ("underestimate", "err low", "smaller", "conservative", "round down"):
        assert nudge not in text


def test_brief_is_byte_identical_to_the_measured_baseline() -> None:
    """`strict` is a third style rather than an edit to `brief`, and this is what pins that.

    Every reading on the board came from `brief`. If adding a challenger silently moved the
    incumbent, the A/B would compare two unmeasured prompts and the eight recorded `prompt_sha`
    values would no longer identify what produced them.
    """
    request = build_request(row("What is the speed of the car at 1.0s in m/s?"))
    assert system_prompt("brief") == (
        "You are a careful physical measurement assistant. You are given frames from a video, a "
        "real-world reference measurement taken from that same scene, and a question. Use the "
        "reference measurement to set the scale -- do not answer from typical real-world sizes, "
        "because the scene may be scaled differently from what you expect. Reason briefly, then "
        "give a single number.")
    assert build_prompt(request, "length of boat = 3.62m", "distance_car_camera = 9m") == (
        "Reference measurement from this scene: length of boat = 3.62m\n"
        "Camera distances in this scene: distance_car_camera = 9m\n"
        "The question asks about the instant t = 1 s.\n"
        "\n"
        "Question: What is the speed of the car at 1.0s in m/s?\n"
        "\n"
        "Answer with a single number in m/s. Do not convert to any other unit. Keep any reasoning "
        "to one or two sentences, then end with exactly:\n"
        f"{ANSWER_SENTINEL} <number>")


def test_strict_keeps_the_prior_the_unit_and_the_instant() -> None:
    """The three things the benchmark turns on must survive the new style too."""
    request = build_request(row("What is the speed of the car at 1.0s in m/s?"))
    prompt = build_prompt(request, "length of boat = 3.62m", "distance_car_camera = 9m",
                          style="strict")
    assert "3.62m" in prompt and "m/s" in prompt
    assert "t = 1 s" in prompt and "Camera distances" in prompt


def test_strict_still_asks_for_less_reasoning_than_brief() -> None:
    """Tighter than `brief`'s "one or two sentences", which helps the truncation fix.

    It does not substitute for it: the cap is what actually cut 841 replies off, and a shorter
    instruction only reduces how often a reply reaches the cap.
    """
    request = build_request(row("What is the width of the pier in meters?"))
    strict = build_prompt(request, "length of boat = 3.62m", None, style="strict")
    assert "one short sentence" in strict
    assert "one or two sentences" not in strict
