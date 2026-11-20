"""Hand-computable FFN reordering example (synthetic capacities, no labels).

Setting: three Transformer blocks, hidden width d = 4, attention terms
A_l = 1, no non-layer terms (G = 0), alpha = 1/2. The FFN capacities are the
multiset {8, 4, 4} placed in three orders; a fourth row keeps order (8, 4, 4)
but doubles the width to d = 8. The capacity units are synthetic: the example
checks the order sensitivity of the formula, not values of the course NSC.

Hand calculation (exact fractions)::

    order (F1,F2,F3)  d   C = (C1,C2,C3)   w = (w1,w2,w3)      sum w*F   S_0   S_1/2
    (8,4,4)           4   (0, 2, 3)        (1, 1/2, 2/5)       58/5      19    73/5
    (4,8,4)           4   (0, 1, 3)        (1, 2/3, 2/5)       164/15    19    209/15
    (4,4,8)           4   (0, 1, 2)        (1, 2/3, 1/2)       32/3      19    41/3
    (8,4,4)           8   (0, 1, 3/2)      (1, 2/3, 4/7)       272/21    19    335/21

with C_l = (F_1 + ... + F_{l-1}) / d, w_l = 1 / (1 + C_l / 2) and
S = sum_l A_l + sum_l w_l F_l = 3 + sum_l w_l F_l.
"""

from __future__ import annotations

from fractions import Fraction as Fr
from typing import Dict, List

from .context_nsc import (ArchTerms, LayerTerms, context_aware_score,
                          cumulative_capacity, ffn_loss, ffn_weights, plain_sum)

ALPHA = Fr(1, 2)
ATTENTION = (1, 1, 1)

# (order, width) -> hand-computed values, entered by hand from the docstring.
HAND: Dict[tuple, Dict[str, object]] = {
    ((8, 4, 4), 4): {"C": (Fr(0), Fr(2), Fr(3)), "w": (Fr(1), Fr(1, 2), Fr(2, 5)),
                     "wF": Fr(58, 5), "S0": Fr(19), "S": Fr(73, 5)},
    ((4, 8, 4), 4): {"C": (Fr(0), Fr(1), Fr(3)), "w": (Fr(1), Fr(2, 3), Fr(2, 5)),
                     "wF": Fr(164, 15), "S0": Fr(19), "S": Fr(209, 15)},
    ((4, 4, 8), 4): {"C": (Fr(0), Fr(1), Fr(2)), "w": (Fr(1), Fr(2, 3), Fr(1, 2)),
                     "wF": Fr(32, 3), "S0": Fr(19), "S": Fr(41, 3)},
    ((8, 4, 4), 8): {"C": (Fr(0), Fr(1), Fr(3, 2)), "w": (Fr(1), Fr(2, 3), Fr(4, 7)),
                     "wF": Fr(272, 21), "S0": Fr(19), "S": Fr(335, 21)},
}


def _terms(order, width, numeric):
    """Build ArchTerms with Fraction (exact) or float inputs."""
    layers = tuple(LayerTerms(attention=numeric(a), ffn=numeric(f)) for a, f in zip(ATTENTION, order))
    return ArchTerms(arch_id=f"toy{order}-d{width}", width=numeric(width), layers=layers, other=numeric(0))


def _fmt(x: Fr) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def run() -> Dict[str, object]:
    """Compute every toy row with exact and float arithmetic and compare to HAND."""
    rows: List[Dict[str, object]] = []
    all_exact = True
    max_float_err = 0.0
    for (order, width), hand in HAND.items():
        exact = _terms(order, width, Fr)
        c = cumulative_capacity(exact.ffn, exact.width)
        w = ffn_weights(exact.ffn, exact.width, ALPHA)
        s0 = plain_sum(exact)
        s = context_aware_score(exact, ALPHA)
        wf = s - sum(exact.attention)
        loss = ffn_loss(exact, ALPHA)
        exact_match = (tuple(c) == hand["C"] and tuple(w) == hand["w"] and wf == hand["wF"]
                       and s0 == hand["S0"] and s == hand["S"] and s0 - loss == s)
        all_exact = all_exact and exact_match

        flt = _terms(order, width, float)
        s_float = context_aware_score(flt, float(ALPHA))
        s0_float = context_aware_score(flt, 0.0)
        err = max(abs(s_float - float(hand["S"])), abs(s0_float - float(hand["S0"])))
        max_float_err = max(max_float_err, err)
        rows.append({
            "order": list(order), "width": width,
            "C": [_fmt(x) for x in c], "w": [_fmt(x) for x in w],
            "sum_wF": _fmt(wf), "loss": _fmt(loss),
            "S_alpha0": _fmt(s0), "S_alpha": _fmt(s), "S_alpha_float": s_float,
            "hand_S_alpha": _fmt(hand["S"]), "exact_match": exact_match, "float_abs_err": err,
        })
    return {"alpha": _fmt(ALPHA), "attention": list(ATTENTION), "rows": rows,
            "all_exact_match": all_exact, "max_float_abs_err": max_float_err,
            "units": "synthetic capacities (not course NSC values)"}


def to_markdown(result: Dict[str, object]) -> str:
    """Markdown table of the toy example (used in results/toy_reordering.md)."""
    lines = [
        "# Toy FFN reordering example (synthetic capacities)",
        "",
        f"alpha = {result['alpha']}, A_l = 1 for every layer, G = 0. "
        "C_l = (F_1+...+F_{l-1})/d, w_l = 1/(1+alpha*C_l), S = sum A_l + sum w_l F_l.",
        "",
        "| order (F1,F2,F3) | d | C | w | sum w*F | loss sum (1-w)F | S (alpha=0) | S (program) | S (hand) | exact match |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        lines.append(
            f"| ({', '.join(map(str, r['order']))}) | {r['width']} | ({', '.join(r['C'])}) | "
            f"({', '.join(r['w'])}) | {r['sum_wF']} | {r['loss']} | {r['S_alpha0']} | "
            f"{r['S_alpha']} = {r['S_alpha_float']:.4f} | {r['hand_S_alpha']} | {r['exact_match']} |"
        )
    lines += [
        "",
        f"All rows match the hand calculation exactly: {result['all_exact_match']}; "
        f"largest float error {result['max_float_abs_err']:.1e}.",
        "",
        "With alpha = 0 every order scores 19, because a sum ignores order. With alpha = 1/2 the score",
        "falls as the large FFN moves later: its own weight drops (more capacity precedes it), and the",
        "small FFNs that move forward gain less than it loses. C_3 = 3 for both (8,4,4) and (4,8,4),",
        "because C_l depends only on the multiset of earlier capacities; those two orders differ only",
        "through w_2. Doubling d halves every C_l, so the same capacities are discounted less.",
    ]
    return "\n".join(lines) + "\n"
