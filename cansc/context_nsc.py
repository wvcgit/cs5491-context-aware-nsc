"""Context-aware reweighting of the NSC feed-forward (FFN) terms.

This module implements only the scoring change proposed for the project. It
never recomputes NSC itself: the per-layer attention contributions ``A_l`` and
the raw FFN capacities ``F_l`` must come from the course NSC implementation
(see ``course_adapter.py``), so the course initialisation and matrix-grouping
conventions are inherited unchanged.

Definition for one architecture with L layers, hidden width d and alpha >= 0::

    C_l = (F_1 + ... + F_{l-1}) / d        (C_1 = 0, an empty sum)
    w_l = 1 / (1 + alpha * C_l)
    S   = G + sum_l (A_l + w_l * F_l)

``G`` collects every term of the course score that is not attached to a layer
(for example embedding or output-head terms, if the course score has any) and
is left unchanged. Only raw capacities enter ``C_l``: discounting an FFN does
not shrink the context seen by later layers. With alpha = 0 every weight is
exactly 1, so the score equals the plain per-layer sum, i.e. original NSC.

The functions use only ``+``, ``*`` and ``/`` on their inputs, so they work
with ``float`` and with ``fractions.Fraction`` (used for exact hand checks).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class LayerTerms:
    """Course NSC terms of one Transformer block.

    Attributes:
        attention: attention contribution A_l exactly as the course scorer
            computes it (kept unchanged by CA-NSC).
        ffn: raw, undiscounted FFN capacity F_l exactly as the course scorer
            computes it.
    """

    attention: Real
    ffn: Real


@dataclass(frozen=True)
class ArchTerms:
    """Per-layer decomposition of the course NSC score of one architecture.

    Attributes:
        arch_id: identifier of the architecture in the course archive.
        width: hidden (residual-stream) width d of the architecture.
        layers: one LayerTerms per Transformer block, in forward order.
        other: sum of all course NSC terms not attached to a block.
    """

    arch_id: str
    width: Real
    layers: Tuple[LayerTerms, ...]
    other: Real = 0

    @property
    def depth(self) -> int:
        """Number of Transformer blocks L."""
        return len(self.layers)

    @property
    def ffn(self) -> List[Real]:
        """Raw FFN capacities [F_1, ..., F_L]."""
        return [layer.ffn for layer in self.layers]

    @property
    def attention(self) -> List[Real]:
        """Attention contributions [A_1, ..., A_L]."""
        return [layer.attention for layer in self.layers]


def _is_bad_float(x: Real) -> bool:
    return isinstance(x, float) and not math.isfinite(x)


def _check_width(width: Real) -> None:
    if _is_bad_float(width) or width <= 0:
        raise ValueError(f"hidden width must be positive and finite, got {width!r}")


def _check_alpha(alpha: Real) -> None:
    if _is_bad_float(alpha) or alpha < 0:
        raise ValueError(f"alpha must be non-negative and finite, got {alpha!r}")


def _check_capacity(f: Real, layer_index: int) -> None:
    if _is_bad_float(f):
        raise ValueError(f"FFN capacity of layer {layer_index} is not finite: {f!r}")
    if f < 0:
        # 1 / (1 + alpha * C) can change sign or explode when C < 0, so the
        # heuristic is only defined for non-negative capacities.
        raise ValueError(
            f"FFN capacity of layer {layer_index} is negative ({f!r}); "
            "the discount assumes F_l >= 0"
        )


def cumulative_capacity(ffn: Sequence[Real], width: Real) -> List[Real]:
    """Return [C_1, ..., C_L] with C_l = (sum of raw F_j for j < l) / d.

    Args:
        ffn: raw FFN capacities in forward order.
        width: hidden width d of the same architecture.

    Raises:
        ValueError: non-positive width, or a negative / non-finite capacity.
    """
    _check_width(width)
    out = []
    running = 0  # raw capacity of the layers already passed
    for index, f in enumerate(ffn, start=1):
        _check_capacity(f, index)
        out.append(running / width)
        running = running + f
    return out


def ffn_weights(ffn: Sequence[Real], width: Real, alpha: Real) -> List[Real]:
    """Return [w_1, ..., w_L] with w_l = 1 / (1 + alpha * C_l).

    w_1 is exactly 1 for every alpha because C_1 is an empty sum.
    """
    _check_alpha(alpha)
    return [1 / (1 + alpha * c) for c in cumulative_capacity(ffn, width)]


def _total(values: Sequence[Real]) -> Real:
    """Sum per-layer values: math.fsum for floats (as the course original_nsc
    does), exact left-to-right addition for Fractions."""
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        return math.fsum(values)
    total = 0
    for v in values:
        total = total + v
    return total


def context_aware_score(terms: ArchTerms, alpha: Real) -> Real:
    """CA-NSC score S = G + sum_l (A_l + w_l * F_l).

    Each layer value A_l + w_l F_l is formed first and the values are added
    with math.fsum, like the course's ``original_nsc``. With alpha = 0, w_l is
    exactly 1, so the result equals :func:`plain_sum` bit for bit, and equals
    the course score bit for bit when A_l + F_l is the course layer value.
    """
    weights = ffn_weights(terms.ffn, terms.width, alpha)
    return _total([terms.other] + [layer.attention + w * layer.ffn
                                   for layer, w in zip(terms.layers, weights)])


def plain_sum(terms: ArchTerms) -> Real:
    """Undiscounted decomposition sum G + sum_l (A_l + F_l)."""
    return _total([terms.other] + [layer.attention + layer.ffn for layer in terms.layers])


def layer_profile(terms: ArchTerms, alpha: Real) -> List[Dict[str, Real]]:
    """Per-layer diagnostics used in reports and pair analyses.

    Returns one dict per layer with keys ``layer`` (1-based), ``A``, ``F``,
    ``C``, ``w`` and ``wF``.
    """
    caps = cumulative_capacity(terms.ffn, terms.width)
    weights = ffn_weights(terms.ffn, terms.width, alpha)
    rows = []
    for index, (layer, c, w) in enumerate(zip(terms.layers, caps, weights), start=1):
        rows.append(
            {"layer": index, "A": layer.attention, "F": layer.ffn, "C": c, "w": w, "wF": w * layer.ffn}
        )
    return rows


def ffn_loss(terms: ArchTerms, alpha: Real) -> Real:
    """Total FFN capacity removed by the discount, sum_l (1 - w_l) F_l.

    plain_sum(terms) - ffn_loss(terms, alpha) == context_aware_score(terms, alpha)
    up to rounding.
    """
    weights = ffn_weights(terms.ffn, terms.width, alpha)
    return _total([(1 - w) * f for f, w in zip(terms.ffn, weights)])
