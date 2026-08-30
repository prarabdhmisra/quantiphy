"""Prompt construction and answer extraction for the VLM arm.

Pure functions, no torch, no network. That is deliberate: the model call is the expensive and
untestable part, so everything that decides *what we ask* and *what we read back* lives here where
it can be exercised on a laptop. The backend in :mod:`quantiphy.backends.vlm` is a thin shell around
these.

Two facts about this benchmark shape the prompt.

**The prior is the whole point.** Every row carries a real-world measurement of something in the
scene (``"length of boat = 3.62m"``) and 3D rows add camera distances. The benchmark paper's headline
finding is that VLMs *ignore* these and answer from memorised object sizes, which is exactly what the
counterfactual-prior test in ``tests/test_solver_core.py`` demonstrates. So the prior is stated
prominently, as a measurement to use rather than a hint, and the question is asked after it.

**The scorer does no unit conversion.** An answer in cm where metres were asked for scores a hard
zero, and that is a silent 100x, the single most dangerous failure mode here. So the unit is restated
outside the question text and the parser refuses to guess: a reply that names a *different* unit is
converted when the dimension matches and rejected when it does not.

Deliberately *not* done: asking the model to answer low. MRA punishes overshoot far harder than
undershoot (1.9x scores zero, 0.5x still scores 0.4), which makes a downward nudge tempting -- but
shrinking a genuinely good per-row predictor was measured on GPT-5.1's 159 validation answers and
loses monotonically at every factor from 0.95 down. Bias belongs in the fusion policy, if anywhere,
not in the prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from quantiphy import units
from quantiphy.parsing import SolveRequest

#: The envelope we demand. A fixed sentinel is far easier to parse than prose, and its absence is
#: itself a signal -- a reply without it usually means the model refused or hedged.
ANSWER_SENTINEL = "ANSWER:"

_SYSTEM_BASE = (
    "You are a careful physical measurement assistant. You are given frames from a video, a "
    "real-world reference measurement taken from that same scene, and a question. Use the reference "
    "measurement to set the scale -- do not answer from typical real-world sizes, because the scene "
    "may be scaled differently from what you expect."
)

#: The two prompt styles the arm can be run at, and the reason there are exactly two.
#:
#: The paper's Table 2 measures chain-of-thought against video+prior at 56.1 -> 27.7, 49.8 -> 22.4
#: and 50.1 -> 23.1: it roughly *halves* MRA for every strong model, and only one small model gains.
#: This module shipped asking for "one or two sentences" of reasoning, which is mild CoT and sits on
#: the wrong side of that finding -- so the variant is a measurement to make, not a knob to tune.
#:
#: ``brief`` is the incumbent and stays the default: every ``prompt_sha`` recorded so far is a brief
#: prompt, and moving the default would make a resumed run non-comparable with its own checkpoint.
#:
#: ``strict`` is added 2026-08-30 to answer two *measured* defects in ``brief``'s 3,289 test replies
#: (``scripts/audit_vlm_raw.py``), and it is a third style rather than an edit to ``brief`` for the
#: reason above -- ``brief`` is the measured baseline and the A/B is the point.
#:
#: * **457 replies refuse**, saying the quantity "cannot be determined without a reference" while the
#:   prompt is supplying one. Concentrated in D2 (218) and D3 (171), the two categories the VLM
#:   otherwise wins. So ``strict`` states that declining is not an available answer and that the
#:   reference given *is* the reference.
#: * **118 replies compute a real value and answer ``0``**, i.e. a sub-metre magnitude rounded to an
#:   integer. So ``strict`` asks for two significant figures and rules out zero explicitly.
#:
#: **This is not the downward nudge the module docstring refuses.** Forbidding a zero and requiring
#: two significant figures constrains the *format*; it does not move the estimate in either
#: direction, and a zero is not a small answer under MRA -- it is a hard zero, scoring exactly the
#: same as a 100x overshoot. The one-sentence limit is also tighter than ``brief``'s "one or two",
#: which helps the truncation fix rather than substituting for it.
_STYLES = {
    "brief": ("Reason briefly, then give a single number.",
              "Keep any reasoning to one or two sentences, then end with exactly:"),
    "direct": ("Give a single number and nothing else.",
               "Give the number only, with no explanation, as exactly:"),
    "strict": ("Reason in one short sentence, then give a single number. Always give a number: the "
               "reference measurement you are given is the reference you need, so never reply that "
               "the quantity cannot be determined, and never answer zero. If you are unsure, "
               "estimate.",
               "Give at least two significant figures, and never 0. Keep reasoning to one short "
               "sentence, then end with exactly:"),
}


def _style(name: str) -> tuple[str, str]:
    if name not in _STYLES:
        raise ValueError(f"unknown prompt style {name!r}; expected one of {sorted(_STYLES)}")
    return _STYLES[name]

_NUMBER = re.compile(r"[-+]?(?:\d+(?:[ ,]\d{3})*(?:\.\d+)?|\.\d+)(?:\s*[eE][-+]?\d+)?")


@dataclass(frozen=True)
class ParsedAnswer:
    """What we managed to read out of the model's reply."""

    value: float | None
    unit: str | None
    #: How the number was found, for auditing a run without re-reading every reply.
    route: str
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.value is not None


def build_prompt(request: SolveRequest, prior_text: str, depth_text: str | None,
                 style: str = "brief") -> str:
    """The user-turn text accompanying the frames.

    ``prior_text`` and ``depth_text`` are passed through verbatim from the dataset rather than
    re-rendered from the parsed objects. The parser exists to serve the geometric solver and is lossy
    on purpose -- it strips quantity words, drops readings it cannot key -- and none of that loss
    helps a model that can read the original string perfectly well.
    """
    lines = [f"Reference measurement from this scene: {prior_text.strip()}"]
    if depth_text and depth_text.strip():
        lines.append(f"Camera distances in this scene: {depth_text.strip()}")
    if request.timestamp is not None:
        lines.append(f"The question asks about the instant t = {request.timestamp:g} s.")
    lines.append("")
    lines.append(f"Question: {request.question.strip()}")
    lines.append("")
    lines.append(
        f"Answer with a single number in {request.output_unit}. Do not convert to any other unit. "
        f"{_style(style)[1]}")
    lines.append(f"{ANSWER_SENTINEL} <number>")
    return "\n".join(lines)


def system_prompt(style: str = "brief") -> str:
    return f"{_SYSTEM_BASE} {_style(style)[0]}"


def _to_float(raw: str) -> float | None:
    try:
        return float(raw.replace(",", "").replace(" ", ""))
    except ValueError:
        return None


def _find_unit(text: str) -> str | None:
    """The unit named next to the answer, if the model volunteered one."""
    match = re.search(rf"{re.escape(ANSWER_SENTINEL)}\s*{_NUMBER.pattern}\s*([A-Za-z/^*\d]+)", text)
    if match and units.is_unit(match.group(1).strip()):
        return match.group(1).strip()
    return None


def parse_answer(text: str, expected_unit: str) -> ParsedAnswer:
    """Read a number out of a model reply, in the unit the question asked for.

    Order matters. The sentinel is tried first because it is the only unambiguous route; falling back
    to "the last number in the reply" is genuinely useful (models often comply in substance while
    dropping the marker) but it also happily picks up a number from the reasoning, so it is recorded
    as a distinct ``route`` and can be audited or discarded per run.

    A zero or a negative is treated as a failure rather than passed through: the scorer takes the
    magnitude and gives a hard zero to a zero prediction, so emitting one is strictly worse than
    declining and letting the fallback fill the row.
    """
    if not text or not text.strip():
        return ParsedAnswer(None, None, "empty", "model returned nothing")

    stated_unit = _find_unit(text)
    marker = text.rfind(ANSWER_SENTINEL)
    if marker >= 0:
        found = _NUMBER.search(text[marker + len(ANSWER_SENTINEL):])
        route = "sentinel"
    else:
        matches = list(_NUMBER.finditer(text))
        found = matches[-1] if matches else None
        route = "last-number"

    if found is None:
        return ParsedAnswer(None, None, "no-number", "no number in the reply")

    value = _to_float(found.group(0))
    if value is None:
        return ParsedAnswer(None, None, "unparseable", f"{found.group(0)!r} is not a number")

    value = abs(value)
    if value == 0.0:
        # A zero scores a hard zero and still counts, so declining is strictly better: the caller's
        # fallback gets the row instead.
        return ParsedAnswer(None, stated_unit, route, "model answered zero; declining")

    if stated_unit and stated_unit != expected_unit:
        try:
            if units.dimension_of(stated_unit) != units.dimension_of(expected_unit):
                return ParsedAnswer(None, stated_unit, route,
                                    f"answered in {stated_unit}, which is not a "
                                    f"{units.dimension_of(expected_unit)}")
            converted = units.from_si(units.to_si(value, stated_unit), expected_unit)
        except Exception as error:                     # unknown unit: refuse rather than guess
            return ParsedAnswer(None, stated_unit, route, f"unit {stated_unit!r}: {error}")
        return ParsedAnswer(converted, expected_unit, route + "+converted",
                            f"converted {value:g} {stated_unit} -> {converted:g} {expected_unit}")

    return ParsedAnswer(value, expected_unit, route)
