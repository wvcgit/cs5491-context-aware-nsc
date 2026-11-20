"""Size-control check: same-width pairs of near-equal parameter count.

Rule (proposal, "Data and evaluation"): two architectures of the same split
form a qualifying pair if they have the same hidden width and

    min(params_a, params_b) / max(params_a, params_b) >= 0.90.

The ratio is compared exactly (fractions), so a pair at exactly 0.90 counts.
The architecture with the lower recorded perplexity is the better one. A score
orders a pair correctly if it gives the better architecture the strictly
higher score.

Ties
----
* truth tie: both recorded perplexities are identical. No order is correct, so
  the pair is excluded from correct / wrong / score-tie counts and reported
  separately. No tolerance is applied: the archive's recorded precision
  defines the label.
* score tie: the two scores agree to a relative tolerance (default 1e-12),
  i.e. the score cannot order the pair.
"""

from __future__ import annotations

import math
from collections import Counter
from fractions import Fraction
from typing import Dict, List, Mapping, Sequence, Tuple

from .records import ArchRecord

MIN_PARAM_RATIO = Fraction(9, 10)
SCORE_TIE_REL_TOL = 1e-12

VERDICTS = ("correct", "wrong", "score_tie", "truth_tie")


def _exact(x) -> Fraction:
    """Exact rational value of a parameter count as it is written."""
    if isinstance(x, bool):
        raise TypeError("parameter count cannot be a bool")
    if isinstance(x, Fraction):
        return x
    if isinstance(x, int):
        return Fraction(x)
    if isinstance(x, float):
        if not math.isfinite(x):
            raise ValueError(f"non-finite parameter count {x!r}")
        return Fraction(repr(x))  # shortest decimal that round-trips
    return Fraction(str(x))


def param_ratio(params_a, params_b) -> Fraction:
    """Smaller parameter count divided by the larger one (exact)."""
    a, b = _exact(params_a), _exact(params_b)
    if a <= 0 or b <= 0:
        raise ValueError("parameter counts must be positive")
    lo, hi = (a, b) if a <= b else (b, a)
    return lo / hi


def qualifying_pairs(
    records: Sequence[ArchRecord], min_ratio: Fraction = MIN_PARAM_RATIO
) -> List[Tuple[ArchRecord, ArchRecord, Fraction]]:
    """All unordered pairs with equal width and parameter ratio >= min_ratio.

    Records are processed in arch_id order so the pair list is deterministic.
    Mixing splits is an error.
    """
    splits = {r.split for r in records}
    if len(splits) > 1:
        raise ValueError(f"pairs must come from one split, got {sorted(splits)}")
    ordered = sorted(records, key=lambda r: r.arch_id)
    out = []
    for i, a in enumerate(ordered):
        for b in ordered[i + 1:]:
            if a.width != b.width:
                continue
            ratio = param_ratio(a.params, b.params)
            if ratio >= min_ratio:
                out.append((a, b, ratio))
    return out


def judge(score_a: float, score_b: float, ppl_a: float, ppl_b: float,
          rel_tol: float = SCORE_TIE_REL_TOL) -> str:
    """Verdict of one score on one pair: correct, wrong, score_tie or truth_tie."""
    if ppl_a == ppl_b:
        return "truth_tie"
    if math.isclose(score_a, score_b, rel_tol=rel_tol, abs_tol=0.0):
        return "score_tie"
    a_is_better = ppl_a < ppl_b
    score_prefers_a = score_a > score_b
    return "correct" if a_is_better == score_prefers_a else "wrong"


def compare(
    pairs: Sequence[Tuple[ArchRecord, ArchRecord, Fraction]],
    scores: Mapping[str, float],
    rel_tol: float = SCORE_TIE_REL_TOL,
) -> Dict[str, object]:
    """Judge every pair with one score.

    Returns a dict with the verdict counts and the per-pair verdict list
    (``(arch_id_a, arch_id_b, verdict)``, in pair order).
    """
    verdicts = []
    for a, b, _ in pairs:
        v = judge(scores[a.arch_id], scores[b.arch_id], a.perplexity, b.perplexity, rel_tol)
        verdicts.append((a.arch_id, b.arch_id, v))
    counts = Counter(v for _, _, v in verdicts)
    return {"counts": {k: counts.get(k, 0) for k in VERDICTS}, "verdicts": verdicts}


def transitions(before: Sequence[Tuple[str, str, str]],
                after: Sequence[Tuple[str, str, str]]) -> Dict[str, int]:
    """Count verdict changes pair by pair, keyed ``"before->after"``."""
    if [(a, b) for a, b, _ in before] != [(a, b) for a, b, _ in after]:
        raise ValueError("verdict lists refer to different pairs")
    counter = Counter(f"{vb}->{va}" for (_, _, vb), (_, _, va) in zip(before, after))
    return dict(sorted(counter.items()))
