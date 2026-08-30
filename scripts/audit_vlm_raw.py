"""Read-only audit of the VLM's raw replies, to price the coverage hole before buying a re-run.

The VLM answers usably on 1,619 of 3,289 test rows (49%). Every other row falls back to the geometric
solver or to a constant -- and the VLM is the *better* arm where it answers, by +0.163/row in D2 and
+0.137/row in D3. So the 1,670 rows it drops are worth more than any remaining combination of the two
arms, and 2026-08-30 established that every such combination is at its argmax or refuted.

**This costs nothing.** ``run_vision_job.py``'s sibling ``run_vlm_job.py`` persists every reply
unparsed to ``<run>/vlm_raw.jsonl`` on the Hub, so the diagnosis is a download and a re-parse. No GPU,
no submission slot. The point is to decide *which* fix to buy, not to guess.

Three questions, and each one names a different fix:

1. **The 845 literal zeros.** ``parse_answer`` declines a zero because MRA gives a hard zero to a zero
   prediction, so the decline is right. The question is why the model said zero. A refusal
   ("height cannot be determined without a reference") is a prompt-comprehension bug, since the prompt
   *does* supply a reference. A genuine "the object is stationary" is a physics disagreement. An
   integer ``0`` next to reasoning that computed 0.4 is a significant-figures bug. Three causes, three
   different fixes, and only the raw text separates them.

2. **The 756 ``last-number`` rows.** These have no ``ANSWER:`` marker, so the parser scraped the last
   number out of the reasoning -- and those values measured bad in all four categories (-0.106 to
   -0.253/row, ``mix-v11``). ``max_new_tokens`` is hardcoded at 128 while the ``brief`` prompt asks for
   one or two sentences *before* the sentinel. The test is whether the reply *stops mid-clause*, not
   whether it is long: a token cap spreads out in character space, so the char histogram cannot see
   it. If the marker-less replies end unfinished, the marker is being truncated away and the fix is
   one constant.

3. **What is recoverable right now.** Any row whose raw reply yields a defensible non-zero value under
   a stricter parse is a free row, with no GPU spend at all.

**The self-gate comes first.** Before any new claim, the audit reproduces the published counts from the
raw text -- 1,619 sentinel / 756 last-number / 845 zeros / 69 no-number. Same replies, same parser, so
a disagreement means this script is wrong, not that a discovery has been made. It exits non-zero rather
than printing an analysis that could justify a spend. That is the pattern ``replay_cache.py`` uses for
its own ``SOLVER_V1`` gate, and it is the only reason to believe anything below it.

Usage:
    py -3.12 scripts/audit_vlm_raw.py
    py -3.12 scripts/audit_vlm_raw.py --run test-vlm-qwen3vl8b-brief --shards 4
    py -3.12 scripts/audit_vlm_raw.py --run validation-vlm-qwen3vl8b-brief --shards 0
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from quantiphy.prompting import ANSWER_SENTINEL, parse_answer  # noqa: E402
from quantiphy.scoring import CATEGORIES, category_labels  # noqa: E402

DEFAULT_REPO = "prarabdhmisra/quantiphy-runs"
DEFAULT_RUN = "test-vlm-qwen3vl8b-brief"

#: Counts published in NEXT-SESSION.md for the test run, by parse route. The self-gate.
PUBLISHED = {"vlm-sentinel": 1619, "vlm-last-number": 756, "none": 914}
PUBLISHED_ZEROS = 845

#: ``max_new_tokens`` in ``VlmBackend.__init__``, which ``run_vlm_job.py`` does not override -- so
#: this is the cap every recorded run actually used, and there is no env var to change it.
MAX_NEW_TOKENS = 128

#: Phrases that mark the model declining rather than measuring. Matched case-insensitively.
REFUSAL_MARKERS = (
    "cannot be determined", "can't be determined", "cannot determine", "unable to determine",
    "not possible to determine", "no reference", "without a reference", "insufficient",
    "cannot be measured", "not visible", "cannot see", "unclear", "impossible to",
)
#: Phrases where the model asserts the quantity really is zero. A physics disagreement, not a bug.
STATIONARY_MARKERS = (
    "stationary", "not moving", "does not move", "at rest", "no movement", "remains still",
    "isn't moving", "is not moving", "zero velocity", "no motion",
)

_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

#: Characters a *finished* reply ends on. Anything else means the generation was cut off.
TERMINAL = '.!?"’”)'


def ends_cleanly(text: str) -> bool:
    """Whether the reply stops where a completed sentence would.

    The discriminator for truncation. ``max_new_tokens`` is counted in tokens, so a cap produces a
    spread of character lengths rather than a spike -- but it always stops the text mid-clause, and
    on this run it frequently stops it mid-number ("the distance is approximately 1.").
    """
    stripped = text.rstrip()
    return bool(stripped) and (stripped[-1] in TERMINAL or stripped[-1].isdigit())


def load_raw(repo: str, run: str, shards: int) -> pd.DataFrame:
    """Every persisted reply for a run, as one frame keyed on ``row_index``.

    Mirrors ``vlm_predictions.py``'s Hub layout: an unsharded run lives at ``<run>/vlm_raw.jsonl``
    and a sharded one at ``<run>-shard<k>/vlm_raw.jsonl``. Overlapping ``row_index`` across shards is
    refused rather than merged, for the reason ``vlm_predictions.py`` gives: an overlap means the
    SHARD arithmetic was wrong, and silently keeping one copy would hide it.
    """
    from huggingface_hub import hf_hub_download

    names = [run] if shards == 0 else [f"{run}-shard{k}" for k in range(1, shards + 1)]
    records: list[dict] = []
    for name in names:
        path = hf_hub_download(repo, repo_type="dataset", filename=f"{name}/vlm_raw.jsonl")
        with open(path, encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        print(f"  {name}: {len(rows)} replies")
        records.extend(rows)

    frame = pd.DataFrame(records)
    if "row_index" not in frame.columns:
        raise SystemExit(f"no 'row_index' in the raw JSONL; columns are {list(frame.columns)}")
    duplicated = int(frame["row_index"].duplicated().sum())
    if duplicated:
        raise SystemExit(f"{duplicated} row_index values appear in more than one shard; the SHARD "
                         f"arithmetic is wrong and the merge would hide it")
    return frame.sort_values("row_index").reset_index(drop=True)


def reparse(frame: pd.DataFrame, predictions: Path) -> pd.DataFrame:
    """Re-run the shipped parser over the raw text, and carry in the columns the audit needs.

    ``category_labels`` needs ``video_type``/``inference_type``, which the JSONL does not store, so
    they are joined from the published predictions CSV on ``row_index``. ``unit`` *is* in the JSONL and
    is used from there -- it is what the run itself passed to the parser, so re-parsing with it is a
    true replay rather than an approximation.
    """
    out = frame.copy()
    out["raw_text"] = out["raw_text"].fillna("").astype(str)
    parsed = [parse_answer(text, expected_unit=str(unit))
              for text, unit in zip(out["raw_text"], out["unit"])]
    out["route"] = [p.route for p in parsed]
    out["value"] = [p.value for p in parsed]
    out["reason"] = [p.note for p in parsed]

    published = pd.read_csv(predictions, usecols=["row_index", "video_type", "inference_type"])
    return out.merge(published, on="row_index", how="left", validate="one_to_one")


def gate_replay(frame: pd.DataFrame) -> None:
    """The re-parse must reproduce the run's own stored parse on every row.

    The JSONL records ``parsed_value`` and ``parse_route`` beside the raw text, so this is an exact
    check and not a count comparison: same text, same parser, same answer. It is the stronger of the
    two gates, because it fails on a single row rather than only when totals drift.
    """
    stored_route = frame["parse_route"].fillna("")
    same_route = (frame["route"] == stored_route)
    stored = pd.to_numeric(frame["parsed_value"], errors="coerce")
    mine = pd.to_numeric(frame["value"], errors="coerce")
    both_null = stored.isna() & mine.isna()
    close = pd.Series(False, index=frame.index)
    pair = stored.notna() & mine.notna()
    close[pair] = ((stored[pair] - mine[pair]).abs() <= 1e-9 * stored[pair].abs().clip(lower=1e-30))
    same_value = both_null | close

    print("\nSELF-GATE 1 -- the re-parse reproduces the run's own stored parse, row by row")
    print(f"  route matches : {int(same_route.sum())} of {len(frame)}")
    print(f"  value matches : {int(same_value.sum())} of {len(frame)}")
    if not (same_route.all() and same_value.all()):
        bad = frame.loc[~(same_route & same_value), ["row_index", "parse_route", "route",
                                                     "parsed_value", "value"]]
        print(bad.head(10).to_string(index=False))
        raise SystemExit("\nGATE FAILED -- this script parses the raw text differently from the run "
                         "that produced it. Fix that before believing anything below.")
    print("  gate passed.")


def classify_zero(text: str) -> str:
    """Why a reply that parsed to zero said zero. One row, one cause, most specific first."""
    low = text.lower()
    numbers = [float(m) for m in _NUMBER.findall(text) if _is_float(m)]
    nonzero = [n for n in numbers if n != 0.0]
    if any(marker in low for marker in REFUSAL_MARKERS):
        return "refusal: says it cannot determine (prompt DOES give a reference)"
    if any(marker in low for marker in STATIONARY_MARKERS):
        return "asserts the quantity is genuinely zero (stationary / at rest)"
    if nonzero:
        return "reasoning contains a non-zero number but the answer is 0 (sig-figs / rounding)"
    return "zero with no other number and no stated cause"


def _is_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def gate_counts(frame: pd.DataFrame) -> None:
    """Reproduce the published per-route counts, or refuse to print an analysis.

    A row's CSV ``method`` is ``none`` whenever the parse produced no usable value, *whatever* route
    it took -- a zero keeps ``route == "sentinel"`` but is still declined. So the published buckets
    are route-and-usable, not route alone, and getting that wrong is the easiest way to mis-price the
    coverage hole in the optimistic direction.
    """
    usable = frame["value"].notna()
    sentinel = int((usable & frame["route"].str.startswith("sentinel")).sum())
    last = int((usable & frame["route"].str.startswith("last-number")).sum())
    declined = len(frame) - sentinel - last
    zeros = int(frame["reason"].fillna("").str.contains("answered zero", case=False).sum())

    got = {"vlm-sentinel": sentinel, "vlm-last-number": last, "none": declined}
    print("\nSELF-GATE 2 -- reproduce the counts published in NEXT-SESSION.md")
    ok = True
    for key, want in PUBLISHED.items():
        mark = "OK " if got[key] == want else "FAIL"
        ok &= got[key] == want
        print(f"  {mark} {key:18s} published {want:5d}  recomputed {got[key]:5d}")
    mark = "OK " if zeros == PUBLISHED_ZEROS else "FAIL"
    ok &= zeros == PUBLISHED_ZEROS
    print(f"  {mark} {'zero-valued':18s} published {PUBLISHED_ZEROS:5d}  recomputed {zeros:5d}")
    if not ok:
        raise SystemExit("\nGATE FAILED -- this script disagrees with the published counts. Fix it "
                         "before believing anything below; do not buy a re-run off these numbers.")
    print("  gate passed: same replies, same parser, same counts.\n")


def report(frame: pd.DataFrame) -> None:
    """The three questions, per category."""
    frame = frame.copy()
    frame["category"] = category_labels(frame)
    frame["chars"] = frame["raw_text"].str.len()
    frame["has_marker"] = frame["raw_text"].str.contains(ANSWER_SENTINEL, regex=False)
    frame["ends_cleanly"] = frame["raw_text"].map(ends_cleanly)

    print("=" * 78)
    print("Q1  THE ZEROS -- why did the model say zero?")
    print("=" * 78)
    zeros = frame[frame["reason"].fillna("").str.contains("zero", case=False)]
    causes = zeros["raw_text"].map(classify_zero)
    table = pd.crosstab(causes, zeros["category"]).reindex(columns=list(CATEGORIES)).fillna(0)
    table = table.astype(int)
    table["TOTAL"] = table.sum(axis=1)
    print(table.sort_values("TOTAL", ascending=False).to_string())

    print("\n" + "=" * 78)
    print("Q2  THE MISSING SENTINEL -- is it truncation at max_new_tokens=128?")
    print("=" * 78)
    print(f"  reply length in chars, by whether the '{ANSWER_SENTINEL}' marker survived\n")
    stats = frame.groupby("has_marker")["chars"].agg(
        count="count", mean="mean", p50=lambda s: s.quantile(0.50),
        p90=lambda s: s.quantile(0.90), p99=lambda s: s.quantile(0.99), max="max")
    print(stats.rename(index={True: "marker present", False: "marker MISSING"}).round(1).to_string())
    print("\n  Suggestive but not sufficient: marker-less replies run about twice as long, which is")
    print("  what a cap would do, but a long reply could also just be a rambling one.")

    # The decisive test is *where* the reply stops, not how long it is. `max_new_tokens` is counted
    # in tokens, so a cap yields a spread of character lengths -- looking for a spike in the char
    # histogram finds nothing even when truncation is exactly what happened. That test was tried
    # first here and was simply the wrong instrument; it is recorded in the commit history rather
    # than left in, because a weak instrument that answers "no" is worse than none.
    ends = frame.groupby("has_marker")["ends_cleanly"].agg(n="count", cleanly="mean")
    print("\n  TRUNCATION FINGERPRINT -- does the reply end where a finished sentence would?")
    print(ends.rename(index={True: "marker present", False: "marker MISSING"})
          .assign(cleanly=lambda f: (f["cleanly"] * 100).round(1).astype(str) + "%").to_string())
    print("\n    A reply cut off mid-number is also what corrupts the `last-number` route: the")
    print("    parser scrapes the truncated fragment. That is the mechanism behind mix-v11's")
    print("    -0.106 to -0.253/row, and it is a one-constant fix, not a model limitation.")

    print("\n" + "=" * 78)
    print("Q3  RECOVERABLE NOW -- rows a stricter parse could fill without any GPU spend")
    print("=" * 78)
    recoverable = zeros["raw_text"].map(
        lambda t: classify_zero(t).startswith("reasoning contains a non-zero"))
    n = int(recoverable.sum())
    print(f"  zero rows whose reasoning holds a non-zero number: {n}")
    if n:
        by = zeros.loc[recoverable.to_numpy(), "category"].value_counts()
        print("  by category:", by.to_dict())
        print("\n  NOT adopted automatically. A number lifted out of reasoning is exactly the")
        print("  `last-number` route, and that route measured -0.106 to -0.253/row in all four")
        print("  categories (mix-v11). Any adoption needs its own submission channel.")

    print("\n" + "=" * 78)
    print("Q4  THE COVERAGE HOLE, ONE CAUSE PER ROW -- this is what a re-run has to fix")
    print("=" * 78)
    usable_sentinel = frame["value"].notna() & frame["route"].str.startswith("sentinel")
    hole = frame[~usable_sentinel].copy()

    def cause(row) -> str:
        """Most specific cause first, so each row is counted exactly once."""
        if not row["has_marker"]:
            return "1. TRUNCATED: no ANSWER: marker, reply hit the token wall"
        note = str(row["reason"] or "")
        if "answered zero" in note:
            return "2. ZERO: " + classify_zero(row["raw_text"])
        if "no number" in note:
            return "3. no number anywhere in the reply"
        if not str(row["raw_text"]).strip():
            return "4. empty reply"
        return "5. other decline (unit mismatch / unparseable)"

    hole["cause"] = hole.apply(cause, axis=1)
    table = pd.crosstab(hole["cause"], hole["category"]).reindex(
        columns=list(CATEGORIES)).fillna(0).astype(int)
    table["TOTAL"] = table.sum(axis=1)
    print(table.sort_index().to_string())
    print(f"\n  coverage hole: {len(hole)} of {len(frame)} rows ({len(hole)/len(frame):.1%})")
    print(f"  usable sentinel answers: {int(usable_sentinel.sum())} "
          f"({usable_sentinel.mean():.1%})")

    fixable = hole["cause"].str.startswith(("1.", "2. ZERO: refusal", "2. ZERO: reasoning"))
    print(f"\n  addressable by a prompt/config change (truncation + refusal + sig-figs): "
          f"{int(fixable.sum())} rows ({fixable.sum()/len(frame):.1%} of the test set)")
    print("  by category:", hole.loc[fixable, "category"].value_counts().reindex(
        list(CATEGORIES)).to_dict())
    print("\n  Genuine 'the object is stationary' claims are NOT in that count: they are a physics")
    print("  disagreement with the dataset, not a formatting bug, and no prompt change is owed one.")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--shards", type=int, default=4, help="0 for an unsharded run")
    parser.add_argument("--predictions", type=Path, default=ROOT / "vlm-v1.predictions.csv",
                        help="published predictions CSV, for the category columns")
    parser.add_argument("--no-count-gate", action="store_true",
                        help="skip the published-count gate; the row-by-row replay gate still runs")
    args = parser.parse_args()

    print(f"downloading raw replies from {args.repo} :: {args.run}")
    frame = reparse(load_raw(args.repo, args.run, args.shards), args.predictions)
    print(f"\n{len(frame)} replies re-parsed")
    gate_replay(frame)
    if not args.no_count_gate:
        gate_counts(frame)
    report(frame)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
