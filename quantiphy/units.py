"""Unit registry and conversion.

The official scorer performs **no unit conversion whatsoever** -- it compares the bare number in
our submission against a ground truth stored in whatever unit the question names. So every answer
has to be converted to the question's unit before it is written out, and getting this wrong is a
silent 100x error rather than a crash.

Everything is normalised to SI internally (metres, m/s, m/s^2, seconds) and converted back exactly
once, at the boundary, in ``answer.py``.
"""

from __future__ import annotations

import re
from typing import Literal

Dimension = Literal["length", "speed", "acceleration", "time"]

#: Unit symbol -> (multiplier to SI, dimension). Keys are already normalised by ``_normalise``.
_UNITS: dict[str, tuple[float, Dimension]] = {
    # length
    "m": (1.0, "length"), "meter": (1.0, "length"), "meters": (1.0, "length"),
    "metre": (1.0, "length"), "metres": (1.0, "length"),
    "cm": (0.01, "length"), "centimeter": (0.01, "length"), "centimeters": (0.01, "length"),
    "mm": (0.001, "length"), "millimeter": (0.001, "length"), "millimeters": (0.001, "length"),
    "km": (1000.0, "length"), "kilometer": (1000.0, "length"), "kilometers": (1000.0, "length"),
    "in": (0.0254, "length"), "inch": (0.0254, "length"), "inches": (0.0254, "length"),
    "ft": (0.3048, "length"), "foot": (0.3048, "length"), "feet": (0.3048, "length"),
    # speed
    "m/s": (1.0, "speed"), "cm/s": (0.01, "speed"), "mm/s": (0.001, "speed"),
    "km/h": (1 / 3.6, "speed"), "kph": (1 / 3.6, "speed"), "mph": (0.44704, "speed"),
    "ft/s": (0.3048, "speed"),
    # acceleration
    "m/s^2": (1.0, "acceleration"), "cm/s^2": (0.01, "acceleration"),
    "mm/s^2": (0.001, "acceleration"), "ft/s^2": (0.3048, "acceleration"),
    "g": (9.80665, "acceleration"),
    # time
    "s": (1.0, "time"), "sec": (1.0, "time"), "secs": (1.0, "time"),
    "second": (1.0, "time"), "seconds": (1.0, "time"),
    "ms": (0.001, "time"), "min": (60.0, "time"), "h": (3600.0, "time"),
}

#: Default unit assumed when a prior states a bare number with no unit. 1,034 of the 3,289 test
#: priors do exactly this, and they follow the benchmark's SI convention.
DEFAULT_UNIT: dict[Dimension, str] = {
    "length": "m", "speed": "m/s", "acceleration": "m/s^2", "time": "s",
}


def _normalise(text: str) -> str:
    """Fold the spelling variants that appear in the data into registry keys."""
    cleaned = text.strip().lower()
    cleaned = cleaned.replace("²", "^2").replace("³", "^3")
    cleaned = cleaned.replace("/s/s", "/s^2").replace("/s2", "/s^2").replace("/s**2", "/s^2")
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned.rstrip(".,;:")


def lookup(text: str) -> tuple[float, Dimension] | None:
    """Resolve a unit string, or None if it isn't a unit we recognise."""
    return _UNITS.get(_normalise(text))


def is_unit(text: str) -> bool:
    return lookup(text) is not None


def dimension_of(unit: str) -> Dimension:
    entry = lookup(unit)
    if entry is None:
        raise ValueError(f"unrecognised unit: {unit!r}")
    return entry[1]


def to_si(value: float, unit: str) -> float:
    entry = lookup(unit)
    if entry is None:
        raise ValueError(f"unrecognised unit: {unit!r}")
    return value * entry[0]


def from_si(value_si: float, unit: str) -> float:
    """Convert an SI magnitude into the unit the question asked for."""
    entry = lookup(unit)
    if entry is None:
        raise ValueError(f"unrecognised unit: {unit!r}")
    return value_si / entry[0]


#: Regex alternation over every known unit spelling, longest first so that ``m/s^2`` wins over
#: ``m`` and ``cm/s`` wins over ``cm``.
_UNIT_ALTERNATION = "|".join(
    re.escape(symbol) for symbol in sorted(_UNITS, key=len, reverse=True)
)

#: Matches a number immediately followed by an optional unit, e.g. ``-2.86m/s^2``, ``0.627 m/s``,
#: ``40mm``, ``5.21``. Superscript two is folded before matching.
NUMBER_WITH_UNIT = re.compile(
    rf"(?P<value>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*(?P<unit>{_UNIT_ALTERNATION})?\b",
    re.IGNORECASE,
)


def parse_quantity(text: str, assume: Dimension | None = None) -> tuple[float, str] | None:
    """Pull a ``(value, unit)`` pair out of free text such as ``'= -2.86m/s^2'``.

    When no unit is present, falls back to the SI default for ``assume``. Returns None if there is
    no number at all, or if there is no unit and no dimension to assume.
    """
    folded = text.replace("²", "^2")
    match = NUMBER_WITH_UNIT.search(folded)
    if match is None:
        return None

    value = float(match.group("value"))
    raw_unit = match.group("unit")
    if raw_unit and lookup(raw_unit):
        return value, _normalise(raw_unit)
    if assume is not None:
        return value, DEFAULT_UNIT[assume]
    return None
