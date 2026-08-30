"""Fuse two prediction files into one, per category, in the replay schema.

Takes the geometric arm's predictions and the VLM arm's and applies :mod:`quantiphy.fusion` row by
row. Emits the same schema ``replay_cache.py`` does, so the result feeds ``make_submission.py``,
``method_ids.py``, ``select_rows.py`` and ``disagreement.py`` unchanged.

**The disagreement cap is per category and that is the point.** One submission carries four
independently reported categories, so a single slot can bracket the cap -- 3x in one category, 5x in
another, 10x in a third, and a no-fusion control in the fourth -- instead of spending three slots on
a one-dimensional sweep. The 5.0 default is the argmax of a six-point sweep on 60 validation rows and
is explicitly *not* confirmed; see the module docstring in ``quantiphy/fusion.py``.

``--primary`` is the arm kept when the two disagree past the cap, and it should be whichever arm the
portal measured better *in that category*. As of `mix-v10`, over all sentinel rows: the VLM won D2
(+0.163/row) and D3 (+0.137/row) and S2 (+0.013/row); the solver won S3 (-0.031/row for the VLM).

**S2's entry there is stale and the sign is wrong on the population that matters.** Measured
2026-08-30 on the 204 S2 sentinel rows where the two arms agree within 5x -- the rows a cap of 5
actually fuses -- the VLM alone scores 0.447, the solver alone 0.470 (+0.066/row) and the blend at
weight 0.7 scores 0.475 (+0.080/row). So in S2 the *solver* is the better arm and the blend beats both
of them, which is the first direct evidence on test that fusing adds something neither arm has. The
+0.013/row above averaged the agreeing rows together with the 59 that disagree past 5x; whether the
solver is also the right arm on *those* is what ``ids-s2-sentinel-disagree.csv`` was built to probe.

``method`` records the route -- ``fuse-fused``, ``fuse-disagreement``, ``fuse-primary-only``,
``fuse-secondary-only`` -- so a probe can select rows by how they were combined, and is ``none``
where neither arm answered.

Usage:
    py -3.12 scripts/fuse_predictions.py \\
        --solver replay-tangential.csv --vlm vlm-v1.predictions.csv \\
        --cap S2=5 --cap D2=5 --cap S3=1 --cap D3=3 \\
        --primary S2=vlm --primary D2=vlm --primary S3=solver --primary D3=vlm \\
        --out fuse-v1.predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy import fusion  # noqa: E402
from quantiphy.scoring import CATEGORIES, category_labels  # noqa: E402

DECLINED = "none"
ARMS = ("solver", "vlm")
#: Columns the slice keys need, carried through from the solver replay.
CARRIED = ("video_id", "question", "video_type", "inference_type")


def parse_pairs(specs: list[str], name: str, cast) -> dict[str, object]:
    """``["D3=3", "S3=1"]`` -> ``{"D3": 3.0, "S3": 1.0}``, with every category required.

    Every category must be named. A category left out would silently fall through to some default and
    produce a channel nobody chose, which is the one failure mode a four-channel probe cannot survive
    -- the reading would be attributed to the wrong change.
    """
    out: dict[str, object] = {}
    for spec in specs:
        if "=" not in spec:
            raise SystemExit(f"--{name} wants CATEGORY=value, got {spec!r}")
        key, _, raw = spec.partition("=")
        key = key.strip().upper()
        if key not in CATEGORIES:
            raise SystemExit(f"{key!r} is not one of {list(CATEGORIES)}")
        if key in out:
            raise SystemExit(f"--{name} names {key} twice")
        out[key] = cast(raw.strip())
    missing = [c for c in CATEGORIES if c not in out]
    if missing:
        raise SystemExit(f"--{name} must name every category; missing {missing}")
    return out


def _arm(raw: str) -> str:
    if raw.lower() not in ARMS:
        raise SystemExit(f"primary must be one of {list(ARMS)}, got {raw!r}")
    return raw.lower()


def fuse_frames(solver: pd.DataFrame, vlm: pd.DataFrame, caps: dict[str, float],
                primaries: dict[str, str], *, weight: float = fusion.WEIGHT,
                prefer_lower: bool = False) -> pd.DataFrame:
    """One record per row, fusing where both arms answered.

    A cap of 1.0 disables fusion for that category -- the arms can only ever be *exactly* equal at
    that fold factor -- which is how the control channel is expressed without a separate flag.
    """
    if "row_index" not in solver.columns or "row_index" not in vlm.columns:
        raise SystemExit("both inputs must carry 'row_index'; they are not replay outputs")

    left = solver.set_index("row_index")
    right = vlm.set_index("row_index")
    categories = category_labels(solver).to_numpy()

    records = []
    for position, row_index in enumerate(solver["row_index"].to_numpy()):
        category = categories[position]
        cap, primary = caps[category], primaries[category]
        source = left.loc[row_index]
        other = right.loc[row_index] if row_index in right.index else None

        sol = source.get("parsed_value")
        vlm_value = None if other is None else other.get("parsed_value")
        first, second = ((sol, vlm_value) if primary == "solver" else (vlm_value, sol))
        value, route = fusion.fuse(first, second, weight=weight, max_disagreement=cap,
                                   prefer_lower=prefer_lower)

        records.append({
            "row_index": int(row_index),
            **{name: source[name] for name in CARRIED},
            "parsed_value": value,
            "method": DECLINED if route == fusion.NEITHER else f"fuse-{route}",
            "reason": "neither arm answered" if route == fusion.NEITHER else "",
            "category": category, "cap": cap, "primary": primary,
        })
    return pd.DataFrame(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver", type=Path, required=True, help="a replay_cache.py --out CSV")
    parser.add_argument("--vlm", type=Path, required=True, help="a vlm_predictions.py --out CSV")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cap", action="append", default=[], metavar="CATEGORY=fold",
                        help="max disagreement per category; 1 disables fusion there")
    parser.add_argument("--primary", action="append", default=[], metavar="CATEGORY=solver|vlm",
                        help="arm kept when the two disagree past the cap")
    parser.add_argument("--weight", type=float, default=fusion.WEIGHT,
                        help="log-space weight on the secondary arm")
    parser.add_argument("--prefer-lower", action="store_true",
                        help="take the smaller arm instead of averaging; halves fatal overshoots")
    args = parser.parse_args()

    caps = parse_pairs(args.cap, "cap", float)
    primaries = parse_pairs(args.primary, "primary", _arm)

    solver = pd.read_csv(args.solver)
    vlm = pd.read_csv(args.vlm)
    frame = fuse_frames(solver, vlm, caps, primaries, weight=args.weight,
                        prefer_lower=args.prefer_lower)
    frame.drop(columns=["category", "cap", "primary"]).to_csv(args.out, index=False,
                                                              encoding="utf-8")

    answered = frame["method"].ne(DECLINED)
    print(f"{len(frame)} rows | answered {int(answered.sum())} ({answered.mean():.1%}) -> {args.out}")
    print(f"\n{'cat':>4} {'cap':>6} {'primary':>8} " +
          " ".join(f"{r:>16}" for r in ("fused", "disagreement", "primary-only")))
    for category in CATEGORIES:
        rows = frame["category"] == category
        counts = frame.loc[rows, "method"].value_counts()
        print(f"{category:>4} {caps[category]:6g} {primaries[category]:>8} " +
              " ".join(f"{int(counts.get(f'fuse-{r}', 0)):16d}"
                       for r in (fusion.FUSED, fusion.TOO_FAR, fusion.PRIMARY_ONLY)))
    print(f"\nnow run: py -3.12 scripts/make_submission.py {args.out} --out <sub>.csv "
          f"--fallback-from baseline_predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
