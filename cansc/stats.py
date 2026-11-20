"""Rank statistics used by the evaluation (standard library only).

average_ranks   1-based ranks; tied values get the mean of their positions
spearman        Spearman rho = Pearson correlation of average ranks
kendall_tau_b   tie-corrected Kendall tau-b by direct pair counting, O(n^2)
rank_change     how much two score vectors reorder the same items
"""

from __future__ import annotations

import math
from typing import Dict, List, Sequence


def _check_same_length(x: Sequence[float], y: Sequence[float]) -> None:
    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    if len(x) < 2:
        raise ValueError("need at least two items")


def _check_finite(values: Sequence[float]) -> None:
    for v in values:
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"non-finite value {v!r}")


def average_ranks(values: Sequence[float]) -> List[float]:
    """Rank values in ascending order (1 = smallest); ties share the mean rank.

    Ties are exact equality, so the caller decides the precision of the input.
    """
    _check_finite(values)
    n = len(values)
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    start = 0
    while start < n:
        end = start
        while end + 1 < n and values[order[end + 1]] == values[order[start]]:
            end += 1
        mean_rank = (start + end) / 2 + 1  # positions start..end are 0-based
        for k in range(start, end + 1):
            ranks[order[k]] = mean_rank
        start = end + 1
    return ranks


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation; NaN if either input is constant."""
    _check_same_length(x, y)
    n = len(x)
    mx = math.fsum(x) / n
    my = math.fsum(y) / n
    sxy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = math.fsum((a - mx) ** 2 for a in x)
    syy = math.fsum((b - my) ** 2 for b in y)
    if sxx == 0 or syy == 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""
    _check_same_length(x, y)
    return pearson(average_ranks(x), average_ranks(y))


def _sign(v: float) -> int:
    return (v > 0) - (v < 0)


def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> float:
    """Kendall tau-b = (P - Q) / sqrt((P + Q + Tx)(P + Q + Ty)).

    P / Q count concordant / discordant pairs, Tx pairs tied only in x and
    Ty pairs tied only in y; pairs tied in both are ignored. NaN if every pair
    is tied in x or in y.
    """
    _check_same_length(x, y)
    _check_finite(x)
    _check_finite(y)
    concordant = discordant = tied_x = tied_y = 0
    n = len(x)
    for i in range(n):
        for j in range(i + 1, n):
            dx = _sign(x[i] - x[j])
            dy = _sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                tied_x += 1
            elif dy == 0:
                tied_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + tied_x) * (concordant + discordant + tied_y))
    if denom == 0:
        return float("nan")
    return (concordant - discordant) / denom


def rank_change(before: Sequence[float], after: Sequence[float]) -> Dict[str, float]:
    """Summarise how ``after`` reorders the items ranked by ``before``.

    Ranks are average ranks (higher score = higher rank number). Returns the
    Spearman correlation between the two score vectors, the number of items
    whose rank changed, the mean and maximum absolute rank shift, and the
    number of item pairs whose strict order is reversed.
    """
    _check_same_length(before, after)
    rb = average_ranks(before)
    ra = average_ranks(after)
    shifts = [abs(a - b) for a, b in zip(ra, rb)]
    reversed_pairs = 0
    n = len(before)
    for i in range(n):
        for j in range(i + 1, n):
            if _sign(before[i] - before[j]) * _sign(after[i] - after[j]) < 0:
                reversed_pairs += 1
    return {
        "spearman_between": spearman(before, after),
        "items_rank_changed": sum(1 for s in shifts if s != 0),
        "mean_abs_rank_shift": math.fsum(shifts) / n,
        "max_abs_rank_shift": max(shifts),
        "reversed_pairs": reversed_pairs,
        "total_pairs": n * (n - 1) // 2,
    }
