"""Randomized property checks of the CA-NSC reweighting.

These checks exercise the scoring change on synthetic per-layer terms (random
depths, widths and capacities) with a fixed seed. They verify the limiting
cases of the proposal (alpha = 0, first layer, single layer) and structural
properties of the formula. They do not involve the course NSC values, so the
corresponding checks on the real development architectures are run separately
in evaluate.py once the course starter is bound.
"""

from __future__ import annotations

import random
from fractions import Fraction
from typing import Dict

from .context_nsc import (ArchTerms, LayerTerms, context_aware_score,
                          ffn_loss, ffn_weights, plain_sum)

ALPHAS = (0.0, 1e-4, 0.01, 0.1, 1.0, 10.0, 1e3)
WIDTHS = (64, 128, 192, 256, 384, 512, 768, 1024)


def random_terms(rng: random.Random, index: int, max_layers: int = 24) -> ArchTerms:
    """Random float decomposition: some zero FFNs, varied depth and width."""
    depth = rng.randint(1, max_layers)
    width = rng.choice(WIDTHS)
    layers = tuple(
        LayerTerms(attention=rng.uniform(0.0, 50.0),
                   ffn=0.0 if rng.random() < 0.1 else rng.uniform(0.0, 8.0 * width))
        for _ in range(depth)
    )
    return ArchTerms(f"synthetic-{index}", width, layers, other=rng.uniform(-20.0, 20.0))


def random_exact_terms(rng: random.Random, index: int, max_layers: int = 12) -> ArchTerms:
    """Random integer decomposition, converted to Fractions for exact checks."""
    depth = rng.randint(1, max_layers)
    width = rng.choice(WIDTHS)
    layers = tuple(LayerTerms(Fraction(rng.randint(0, 50)), Fraction(rng.randint(0, 8 * width)))
                   for _ in range(depth))
    return ArchTerms(f"exact-{index}", Fraction(width), layers, other=Fraction(rng.randint(-20, 20)))


def run_synthetic_checks(seed: int = 5491, cases: int = 2000, exact_cases: int = 300) -> Dict[str, object]:
    """Run every property on random cases and return counts of violations."""
    rng = random.Random(seed)
    v = {name: 0 for name in (
        "alpha0_not_bitwise_equal_plain_sum", "first_weight_not_1", "single_layer_changed",
        "score_increased_with_alpha", "weight_out_of_(0,1]", "weight_increased_with_depth",
        "prefix_weights_changed", "alpha0_not_order_invariant", "loss_identity_failed",
        "width_scaling_changed_weights", "exact_alpha0_mismatch", "exact_single_layer_mismatch",
        "exact_first_weight_not_1")}
    max_err = {"alpha0_vs_plain_sum": 0.0, "single_layer": 0.0, "loss_identity_rel": 0.0,
               "alpha0_order_rel": 0.0, "width_scaling": 0.0}
    single_layer_cases = 0

    for i in range(cases):
        t = random_terms(rng, i)
        base = plain_sum(t)
        s0 = context_aware_score(t, 0.0)
        max_err["alpha0_vs_plain_sum"] = max(max_err["alpha0_vs_plain_sum"], abs(s0 - base))
        if s0 != base:
            v["alpha0_not_bitwise_equal_plain_sum"] += 1

        prev_score = s0
        for a in ALPHAS:
            w = ffn_weights(t.ffn, t.width, a)
            if w[0] != 1:
                v["first_weight_not_1"] += 1
            if any(not (0 < x <= 1) for x in w):
                v["weight_out_of_(0,1]"] += 1
            if any(w[k + 1] > w[k] for k in range(len(w) - 1)):
                v["weight_increased_with_depth"] += 1
            s = context_aware_score(t, a)
            if s > prev_score:
                v["score_increased_with_alpha"] += 1
            prev_score = s
            rel = abs((base - ffn_loss(t, a)) - s) / max(1.0, abs(s))
            max_err["loss_identity_rel"] = max(max_err["loss_identity_rel"], rel)
            if rel > 1e-9:
                v["loss_identity_failed"] += 1
            # prefix property: truncating later layers leaves earlier weights unchanged
            k = rng.randint(1, t.depth)
            if ffn_weights(t.ffn[:k], t.width, a) != w[:k]:
                v["prefix_weights_changed"] += 1
            # width scaling: multiplying all F and d by the same factor keeps C_l and w_l
            scale = 2.0
            w_scaled = ffn_weights([f * scale for f in t.ffn], t.width * scale, a)
            err = max(abs(x - y) for x, y in zip(w, w_scaled))
            max_err["width_scaling"] = max(max_err["width_scaling"], err)
            if err > 1e-12:
                v["width_scaling_changed_weights"] += 1

        # alpha = 0 ignores layer order (up to summation rounding)
        rev = ArchTerms(t.arch_id, t.width, tuple(reversed(t.layers)), t.other)
        rel = abs(context_aware_score(rev, 0.0) - s0) / max(1.0, abs(s0))
        max_err["alpha0_order_rel"] = max(max_err["alpha0_order_rel"], rel)
        if rel > 1e-12:
            v["alpha0_not_order_invariant"] += 1

        # single-layer version of the same architecture: alpha has no effect
        one = ArchTerms(t.arch_id, t.width, t.layers[:1], t.other)
        single_layer_cases += 1
        s_one = context_aware_score(one, 0.0)
        for a in ALPHAS:
            err = abs(context_aware_score(one, a) - s_one)
            max_err["single_layer"] = max(max_err["single_layer"], err)
            if err != 0:
                v["single_layer_changed"] += 1

    for i in range(exact_cases):
        t = random_exact_terms(rng, i)
        if context_aware_score(t, Fraction(0)) != plain_sum(t):
            v["exact_alpha0_mismatch"] += 1
        for a in (Fraction(1, 1000), Fraction(1, 7), Fraction(3), Fraction(50)):
            if ffn_weights(t.ffn, t.width, a)[0] != 1:
                v["exact_first_weight_not_1"] += 1
            one = ArchTerms(t.arch_id, t.width, t.layers[:1], t.other)
            if context_aware_score(one, a) != plain_sum(one):
                v["exact_single_layer_mismatch"] += 1

    return {
        "seed": seed, "float_cases": cases, "exact_cases": exact_cases,
        "single_layer_cases": single_layer_cases, "alphas": list(ALPHAS),
        "violations": v, "total_violations": sum(v.values()), "max_errors": max_err,
        "note": "synthetic per-layer terms; real-archive checks run in evaluate.py",
    }
