"""Turn each row's ``question``/``prior``/``depth_info`` text into a structured solve request.

This is the deterministic front half of the geometric solver and runs on CPU. Nothing here calls a
model: every field is recovered by pattern, so it is fast, auditable, and testable against all
3,289 test rows without a GPU.

Quirks in the real data that this handles, all confirmed by profiling the released test split:

* Units appear mid-sentence (``"What is the height, in meters, of the pillar?"``) as well as at
  the end, and some questions terminate with a fullwidth ``？`` (U+FF1F).
* 1,034 of 3,289 priors give a bare number with no unit, so the unit is inferred from the quantity.
* Accelerations may be negative (``acceleration of the orange car = -2.86m/s^2``).
* Priors may carry possessives and parentheticals (``Callisto's model speed(the outermost planet)``)
  and 15 are multi-line.
* ``depth_info`` is multi-line, keyed by underscored object name, sometimes with a timestamp
  prefix and sometimes without, and the ``=`` may have no following space.
* At least one question misspells "displacement" as "diasplacement".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from quantiphy.units import Dimension, is_unit, lookup, parse_quantity, to_si

#: Words that identify what is being asked for, mapped to the dimension they imply. Order matters:
#: the first match wins, so "acceleration" is tested before "speed".
_TARGET_KEYWORDS: tuple[tuple[str, Dimension, str], ...] = (
    (r"acceler", "acceleration", "acceleration"),
    (r"veloc|speed", "speed", "velocity"),
    (r"displac|diasplac", "length", "displacement"),
    (r"\bdistance\b", "length", "distance"),
    (r"diameter", "length", "diameter"),
    (r"radius", "length", "radius"),
    (r"height|tall", "length", "height"),
    (r"width|wide", "length", "width"),
    (r"\blength\b|\blong\b", "length", "length"),
    (r"how long does|what time|duration", "time", "time"),
)

#: Quantity words on the left-hand side of a prior, mapped to their dimension.
_PRIOR_KEYWORDS: tuple[tuple[str, Dimension], ...] = (
    (r"acceler|gravity", "acceleration"),
    (r"veloc|speed", "speed"),
    (r"diameter|radius|length|width|height|calibre|caliber|size|distance", "length"),
)

_TIME_AT = re.compile(r"\bat\s+(?:t\s*=\s*)?(\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE)
_TIME_BETWEEN = re.compile(
    r"between\s+(\d+(?:\.\d+)?)\s*s?\s+and\s+(\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE
)
_TIME_AFTER = re.compile(r"\bafter\s+(\d+(?:\.\d+)?)\s*s\b", re.IGNORECASE)

#: ``in <unit>`` anywhere in the question. Guarded against "in the center" by requiring the
#: captured token to resolve in the unit registry.
_UNIT_IN = re.compile(r"\bin\s+([A-Za-z][A-Za-z/^0-9²]*)", re.IGNORECASE)

_DEPTH_LINE = re.compile(
    r"(?:t\s*=\s*(?P<time>\d+(?:\.\d+)?)\s*s\s*[,;]?\s*)?"
    r"distance_(?P<object>[A-Za-z0-9_]+?)_camera\s*=\s*"
    r"(?P<value>[-+]?\d*\.?\d+)\s*(?P<unit>mm|cm|m|km)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Prior:
    """One ``name = value unit`` clause from the ``prior`` column, normalised to SI."""

    object_name: str
    quantity: str
    dimension: Dimension
    value_si: float
    unit: str
    timestamp: float | None = None

    @property
    def is_gravity(self) -> bool:
        return "gravity" in self.quantity


@dataclass(frozen=True)
class DepthReading:
    object_name: str
    distance_m: float
    timestamp: float | None = None


@dataclass(frozen=True)
class SolveRequest:
    """Everything the geometric solver needs, recovered from one dataset row."""

    video_id: str
    fps: float
    question: str
    output_unit: str
    dimension: Dimension
    measurement: str
    target_object: str | None
    timestamp: float | None = None
    interval: tuple[float, float] | None = None
    priors: tuple[Prior, ...] = ()
    depths: tuple[DepthReading, ...] = ()
    is_3d: bool = False
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def scale_prior(self) -> Prior | None:
        """The prior that actually anchors pixel->world scale.

        Gravity is a physics constant rather than a scene measurement, so it can only set scale
        for acceleration; prefer any other prior when one exists.
        """
        if not self.priors:
            return None
        non_gravity = [p for p in self.priors if not p.is_gravity]
        return non_gravity[0] if non_gravity else self.priors[0]


def _strip_object(text: str) -> str:
    """Tidy an extracted object phrase into something a grounding model can consume."""
    cleaned = re.sub(r"\(.*?\)", " ", text)
    cleaned = re.sub(r"\b(at|after|between|from|in|on|when|during)\b.*$", " ", cleaned,
                     flags=re.IGNORECASE)
    cleaned = cleaned.replace("’s", "").replace("'s", "")
    cleaned = re.sub(r"^\s*(the|a|an)\s+", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"[\s,]+", " ", cleaned).strip(" ,.?？")
    return cleaned


def parse_output_unit(question: str) -> str | None:
    """The unit the answer must be expressed in, or None if the question doesn't say."""
    folded = question.replace("²", "^2")
    for candidate in _UNIT_IN.findall(folded):
        token = candidate.rstrip(".,;:?？")
        if is_unit(token):
            return token.lower()
    return None


def parse_prior(text: str) -> tuple[Prior, ...]:
    """Parse the ``prior`` column into zero or more normalised clauses."""
    if not text or not str(text).strip():
        return ()

    parsed: list[Prior] = []
    for line in re.split(r"[\n;]+", str(text)):
        if "=" not in line:
            continue
        left, _, right = line.rpartition("=")
        label = left.strip().lower()

        dimension: Dimension | None = None
        quantity = "unknown"
        for pattern, dim in _PRIOR_KEYWORDS:
            match = re.search(pattern, label)
            if match:
                dimension, quantity = dim, match.group(0)
                break
        if dimension is None:
            continue

        quantified = parse_quantity(right, assume=dimension)
        if quantified is None:
            continue
        value, unit = quantified

        # Reject a unit whose dimension contradicts the quantity word (e.g. "speed ... = 5 m"),
        # trusting the word and falling back to the SI default.
        resolved = lookup(unit)
        if resolved is not None and resolved[1] != dimension:
            quantified = parse_quantity(re.sub(r"[A-Za-z/^]+\s*$", "", right), assume=dimension)
            if quantified is None:
                continue
            value, unit = quantified

        timestamp = None
        time_match = _TIME_AT.search(label) or _TIME_AFTER.search(label)
        if time_match:
            timestamp = float(time_match.group(1))

        object_name = _strip_object(re.sub(r"^.*?\b(?:of|for)\b", "", label) or label)
        if not object_name:
            object_name = _strip_object(re.sub(r"\b(speed|velocity|acceleration|diameter|radius|"
                                               r"length|width|height|calibre|caliber)\b", " ",
                                               label))

        parsed.append(Prior(
            object_name=object_name,
            quantity="gravity" if "gravity" in label else quantity,
            dimension=dimension,
            value_si=to_si(value, unit),
            unit=unit,
            timestamp=timestamp,
        ))
    return tuple(parsed)


def parse_depth(text: str) -> tuple[DepthReading, ...]:
    """Parse ``depth_info`` into per-object metric distances from the camera."""
    if not text or not str(text).strip():
        return ()
    readings: list[DepthReading] = []
    for match in _DEPTH_LINE.finditer(str(text)):
        unit = match.group("unit") or "m"
        readings.append(DepthReading(
            object_name=match.group("object").replace("_", " ").strip().lower(),
            distance_m=to_si(float(match.group("value")), unit),
            timestamp=float(match.group("time")) if match.group("time") else None,
        ))
    return tuple(readings)


def parse_question(question: str) -> tuple[str, Dimension | None, str | None, dict]:
    """Recover ``(measurement, dimension, target_object, timing)`` from the question text."""
    lowered = question.lower()

    measurement, dimension = "size", None
    for pattern, dim, name in _TARGET_KEYWORDS:
        if re.search(pattern, lowered):
            measurement, dimension = name, dim
            break

    timing: dict = {}
    between = _TIME_BETWEEN.search(question)
    if between:
        timing["interval"] = (float(between.group(1)), float(between.group(2)))
    else:
        at = _TIME_AT.search(question) or _TIME_AFTER.search(question)
        if at:
            timing["timestamp"] = float(at.group(1))

    target = None
    phrase = re.search(r"\bof\s+(?:the\s+)?(.+?)(?:\s*[\?？]|$)", question, re.IGNORECASE)
    if phrase:
        target = _strip_object(phrase.group(1)) or None
    if not target:
        # "What is the distance between the bird and the wall?" has no "of the ...".
        fallback = re.search(r"\bbetween\s+(?:the\s+)?(.+?)(?:\s+and\b|[\?？]|$)", question,
                             re.IGNORECASE)
        if fallback:
            target = _strip_object(fallback.group(1)) or None
    return measurement, dimension, target, timing


def build_request(row) -> SolveRequest:
    """Assemble a :class:`SolveRequest` from a dataset row (dict or pandas Series)."""
    get = row.get
    question = str(get("question") or "")
    video_type = str(get("video_type") or "")

    measurement, keyword_dimension, target, timing = parse_question(question)
    output_unit = parse_output_unit(question)

    warnings: list[str] = []
    if output_unit is None:
        # Never leave this unset -- a wrong unit is a silent 100x error. Fall back to the SI unit
        # for the dimension the question words imply, and flag the row for review.
        from quantiphy.units import DEFAULT_UNIT
        output_unit = DEFAULT_UNIT[keyword_dimension or "length"]
        warnings.append("no explicit unit in question; assumed SI")

    # The unit is a more reliable dimension signal than the keyword, so let it win on conflict.
    dimension = lookup(output_unit)[1]
    if keyword_dimension is not None and keyword_dimension != dimension:
        warnings.append(f"keyword implies {keyword_dimension} but unit implies {dimension}")
    if target is None:
        warnings.append("could not identify target object")

    return SolveRequest(
        video_id=str(get("video_id") or ""),
        fps=float(get("fps") or 30),
        question=question,
        output_unit=output_unit,
        dimension=dimension,
        measurement=measurement,
        target_object=target,
        timestamp=timing.get("timestamp"),
        interval=timing.get("interval"),
        priors=parse_prior(get("prior") if get("prior") is not None
                           else get("ground_truth_prior")),
        depths=parse_depth(get("depth_info")),
        is_3d=len(video_type) > 1 and video_type[1] == "3",
        warnings=tuple(warnings),
    )
