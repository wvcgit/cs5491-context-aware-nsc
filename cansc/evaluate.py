"""Development-split evaluation of course NSC and CA-NSC.

Inputs, bound to the course starter by course_adapter.py:

* ``records``: ArchRecord list of the official development split;
* ``original_fn(record)``: course NSC, from a direct call to the course scorer;
* ``terms_fn(record)``: ArchTerms with the course per-layer attention
  contributions and raw FFN capacities;
* ``harness``: the course evaluate module. Its ``canonical`` (12 significant
  digits), ``ranks`` (average ranks), ``correlation`` and ``pair_summary``
  produce the official numbers; this package's own statistics recompute them
  as an independent cross-check.

Optional hooks add the checks that need the course code: ``reset_fn`` empties
the MP cache before every timed pass, ``layer_capacities_fn`` returns the
course per-layer values, and ``single_layer_fn`` returns the course score and
terms of a one-block copy. The final split is never touched.
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .context_nsc import ArchTerms, context_aware_score, ffn_loss, ffn_weights, layer_profile, plain_sum
from .pairs import compare, qualifying_pairs, transitions
from .protocol import select_alpha, weight_summary
from .records import DEVELOPMENT, ArchRecord
from .stats import average_ranks, kendall_tau_b, pearson, rank_change, spearman

IDENTITY_REL_TOL = 1e-9  # |S_CA(0) - S_course| <= 1e-9 * max(1, |S_course|)
LARGE_ALPHAS = (1e3, 1e6)  # extra values for the single-block check ("any alpha >= 0")


class LocalHarness:
    """Stand-in for the course evaluate module (used by the synthetic tests)."""

    @staticmethod
    def canonical(value: float) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("All scores must be finite")
        return float(format(value, ".12g"))

    @staticmethod
    def ranks(values: Sequence[float]) -> List[float]:
        return average_ranks(values)

    @staticmethod
    def correlation(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
        r = pearson(x, y)
        return None if math.isnan(r) else r

    @staticmethod
    def pair_summary(scores, quality, pairs) -> Dict[str, object]:
        c = d = tx = ty = both = 0
        for i, j in pairs:
            dx = (scores[i] > scores[j]) - (scores[i] < scores[j])
            dy = (quality[i] > quality[j]) - (quality[i] < quality[j])
            if dx == 0 and dy == 0:
                both += 1
            elif dx == 0:
                tx += 1
            elif dy == 0:
                ty += 1
            elif dx == dy:
                c += 1
            else:
                d += 1
        denom = math.sqrt((c + d + tx) * (c + d + ty))
        return {"pairs": c + d + tx + ty + both, "concordant": c, "discordant": d,
                "score_only_ties": tx, "quality_only_ties": ty, "both_ties": both,
                "tau_b": (c - d) / denom if denom else None}


def time_scoring(score_fn: Callable[[ArchRecord], float], records: Sequence[ArchRecord], repeats: int,
                 reset_fn: Optional[Callable[[], None]] = None) -> Tuple[List[float], float]:
    """Score every record ``repeats`` times; return the scores and the median total ms.

    ``reset_fn`` (if given) runs before each pass, outside the timed region.
    """
    timings = []
    scores: List[float] = []
    for _ in range(max(1, repeats)):
        if reset_fn is not None:
            reset_fn()
        start = time.perf_counter()
        scores = [float(score_fn(r)) for r in records]
        timings.append((time.perf_counter() - start) * 1000.0)
    return scores, statistics.median(timings)


def _check_inputs(records: Sequence[ArchRecord], terms: Sequence[ArchTerms]) -> None:
    ids = [r.arch_id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate architecture ids in the development split")
    bad = sorted({r.split for r in records} - {DEVELOPMENT})
    if bad:
        raise ValueError(f"evaluation must use the development split only, found {bad}")
    for r, t in zip(records, terms):
        if t.arch_id != r.arch_id:
            raise ValueError(f"terms for {t.arch_id} returned for record {r.arch_id}")
        if t.width != r.width:
            raise ValueError(f"{r.arch_id}: width {t.width} in terms vs {r.width} in record")
        if t.depth != r.depth:
            raise ValueError(f"{r.arch_id}: {t.depth} layer terms vs depth {r.depth}")


def identity_check(original: Sequence[float], ca_alpha0: Sequence[float],
                   rel_tol: float = IDENTITY_REL_TOL) -> Dict[str, object]:
    """Compare course NSC with CA-NSC at alpha = 0, architecture by architecture."""
    abs_errs = [abs(a - b) for a, b in zip(original, ca_alpha0)]
    fails = [i for i, (o, e) in enumerate(zip(original, abs_errs)) if e > rel_tol * max(1.0, abs(o))]
    return {
        "n": len(abs_errs),
        "max_abs_err": max(abs_errs),
        "max_rel_err": max(e / max(1.0, abs(o)) for o, e in zip(original, abs_errs)),
        "n_bitwise_equal": sum(1 for e in abs_errs if e == 0),
        "tolerance": f"|S_CA(0) - S_course| <= {rel_tol:g} * max(1, |S_course|)",
        "n_fail": len(fails),
        "passed": not fails,
    }


def evaluate_development(records: Sequence[ArchRecord],
                         original_fn: Callable[[ArchRecord], float],
                         terms_fn: Callable[[ArchRecord], ArchTerms],
                         grid: Sequence[float],
                         repeats: int = 5,
                         harness=None,
                         reset_fn: Optional[Callable[[], None]] = None,
                         layer_capacities_fn: Optional[Callable[[ArchRecord], List[float]]] = None,
                         single_layer_fn: Optional[Callable[[ArchRecord], Tuple[float, ArchTerms]]] = None
                         ) -> Dict[str, object]:
    """Run the milestone evaluation on the development split.

    Returns a JSON-serialisable dict with the main table rows, the selected
    alpha, the correctness checks on real architectures, the rank change and
    the size-control pair comparison. Ranking statistics use canonical
    (12-significant-digit) scores, as the course evaluator does; score ranges
    use the unrounded scores.
    """
    h = harness if harness is not None else LocalHarness()
    records = list(records)
    terms = [terms_fn(r) for r in records]
    _check_inputs(records, terms)
    ids = [r.arch_id for r in records]
    index = {k: i for i, k in enumerate(ids)}
    quality = [-float(r.perplexity) for r in records]
    rank_quality = h.ranks(quality)
    all_pairs = [(i, j) for i in range(len(ids)) for j in range(i)]

    # Course NSC, called directly; timing starts from a cold MP cache.
    original, original_ms = time_scoring(original_fn, records, repeats, reset_fn)

    # Correctness checks on the real architectures.
    checks: Dict[str, object] = {
        "alpha0_vs_course": identity_check(original, [float(context_aware_score(t, 0.0)) for t in terms])}
    if layer_capacities_fn is not None:
        mismatches, max_err, n_layers = 0, 0.0, 0
        for r, t in zip(records, terms):
            course_layers = layer_capacities_fn(r)
            mine = [layer.attention + layer.ffn for layer in t.layers]
            n_layers += len(mine)
            for a, b in zip(course_layers, mine):
                max_err = max(max_err, abs(a - b))
                mismatches += a != b
            mismatches += len(course_layers) != len(mine)
        checks["per_layer_decomposition"] = {"n_layers": n_layers, "n_not_bitwise_equal": mismatches,
                                             "max_abs_err": max_err}
    first = [ffn_weights(t.ffn, t.width, a)[0] for t in terms for a in grid]
    checks["first_layer_weight"] = {"n_checked": len(first), "all_exactly_one": all(w == 1 for w in first)}
    single_alphas = list(grid) + [a for a in LARGE_ALPHAS if a not in grid]
    archived_single = [(t, o) for t, o in zip(terms, original) if t.depth == 1]
    checks["single_layer_archived"] = {
        "n_single_layer_architectures": len(archived_single),
        "max_abs_err_any_alpha": max((abs(float(context_aware_score(t, a)) - o)
                                      for t, o in archived_single for a in single_alphas), default=None)}
    if single_layer_fn is not None:
        errs = []
        for r in records:
            course_one, terms_one = single_layer_fn(r)
            errs += [abs(float(context_aware_score(terms_one, a)) - course_one) for a in single_alphas]
        checks["single_layer_constructed"] = {
            "n_configs": len(records), "alphas": single_alphas, "max_abs_err": max(errs),
            "all_bitwise_equal": all(e == 0 for e in errs),
            "construction": "block 1 of each development architecture, scored by the course code"}

    def summarise(name: str, alpha, raw: List[float], ms: float) -> Dict[str, object]:
        canon = [h.canonical(v) for v in raw]
        rho = h.correlation(h.ranks(canon), rank_quality)
        rho_check = spearman(canon, quality)
        return {"method": name, "alpha": alpha, "n": len(raw),
                "spearman": rho, "spearman_crosscheck_abs_diff": abs(rho - rho_check) if rho is not None else None,
                "kendall_tau_b": h.pair_summary(canon, quality, all_pairs)["tau_b"],
                "kendall_crosscheck": kendall_tau_b(canon, quality),
                "score_min": min(raw), "score_max": max(raw), "time_ms": ms}

    rows = [summarise("NSC (course)", None, original, original_ms)]
    raw_by_alpha: Dict[float, List[float]] = {}
    for alpha in grid:
        fn = (lambda r, a=alpha: context_aware_score(terms_fn(r), a))
        raw, ms = time_scoring(fn, records, repeats, reset_fn)
        raw_by_alpha[alpha] = raw
        row = summarise("CA-NSC", alpha, raw, ms)
        row.update(weight_summary(terms, alpha))
        rows.append(row)

    selected = select_alpha([(row["alpha"], row["spearman"] if row["spearman"] is not None else math.nan)
                             for row in rows[1:]])
    canon_orig = [h.canonical(v) for v in original]
    canon_sel = [h.canonical(v) for v in raw_by_alpha[selected]]

    # Size control: same hidden width, total-parameter ratio >= 0.90.
    pairs = qualifying_pairs(records)
    index_pairs = [(index[a.arch_id], index[b.arch_id]) for a, b, _ in pairs]
    float_rule = {frozenset((i, j)) for i, j in all_pairs
                  if records[i].width == records[j].width
                  and min(records[i].params, records[j].params) / max(records[i].params, records[j].params) >= 0.90}
    comp_orig = compare(pairs, dict(zip(ids, canon_orig)), rel_tol=0.0)
    comp_sel = compare(pairs, dict(zip(ids, canon_sel)), rel_tol=0.0)
    official_orig = h.pair_summary(canon_orig, quality, index_pairs)
    official_sel = h.pair_summary(canon_sel, quality, index_pairs)

    def agrees(counts, official) -> bool:
        return (counts["correct"] == official["concordant"] and counts["wrong"] == official["discordant"]
                and counts["score_tie"] == official["score_only_ties"]
                and counts["truth_tie"] == official["quality_only_ties"] + official["both_ties"])

    by_id = {r.arch_id: r for r in records}
    terms_by_id = {t.arch_id: t for t in terms}
    reversals = [(a, b) for (a, b, vo), (_, _, vs) in zip(comp_orig["verdicts"], comp_sel["verdicts"])
                 if vo == "correct" and vs == "wrong"]
    examples = []
    for a, b in sorted(reversals, key=lambda p: -abs(by_id[p[0]].perplexity - by_id[p[1]].perplexity)):
        examples.append({"pair": [a, b], "architectures": [
            _describe(by_id[x], terms_by_id[x], selected, original[index[x]]) for x in (a, b)]})

    # Structure of the pairs whose verdict changed (better = lower perplexity).
    changed = []
    for (a, b, vo), (_, _, vs) in zip(comp_orig["verdicts"], comp_sel["verdicts"]):
        if vo == vs or "truth_tie" in (vo, vs):
            continue
        better, worse = (a, b) if by_id[a].perplexity < by_id[b].perplexity else (b, a)
        tb, tw = terms_by_id[better], terms_by_id[worse]
        att_b, att_w = math.fsum(map(float, tb.attention)), math.fsum(map(float, tw.attention))
        ffn_b, ffn_w = math.fsum(map(float, tb.ffn)), math.fsum(map(float, tw.ffn))
        changed.append({
            "transition": f"{vo}->{vs}", "better": better, "worse": worse,
            "depth_better": tb.depth, "depth_worse": tw.depth,
            "attention_better": att_b, "attention_worse": att_w, "ffn_better": ffn_b, "ffn_worse": ffn_w,
            "ppl_better": by_id[better].perplexity, "ppl_worse": by_id[worse].perplexity,
            # True when the better model has more attention capacity and less raw FFN capacity
            "better_is_attention_rich": att_b > att_w and ffn_b < ffn_w,
            "worse_is_attention_rich": att_w > att_b and ffn_w < ffn_b,
        })

    # Capacity units (label-free): median F_l / d per hidden width.
    per_width: Dict[float, List[float]] = {}
    for t in terms:
        per_width.setdefault(t.width, []).extend(float(f) / float(t.width) for f in t.ffn)
    units = {str(w): statistics.median(v) for w, v in sorted(per_width.items())}

    return {
        "split": DEVELOPMENT, "n": len(records), "grid": list(grid), "repeats": repeats,
        "ffn_capacity_per_width_median": units,
        "timing": "median of the repeats; total for the split; MP cache emptied before each pass"
                  if reset_fn is not None else "median of the repeats; total for the split",
        "rows": rows, "selected_alpha": selected, "checks": checks,
        "rank_change_course_vs_selected": rank_change(canon_orig, canon_sel),
        "size_control": {
            "rule": "same d_model; min(total)/max(total) >= 0.90 (exact; equals the course float rule: "
                    f"{float_rule == {frozenset(p) for p in index_pairs}})",
            "n_pairs": len(pairs),
            "course": comp_orig["counts"], "selected": comp_sel["counts"],
            "official_course": official_orig, "official_selected": official_sel,
            "counts_match_official": agrees(comp_orig["counts"], official_orig)
                                     and agrees(comp_sel["counts"], official_sel),
            "transitions": transitions(comp_orig["verdicts"], comp_sel["verdicts"]),
            "n_reversed_correct_to_wrong": len(reversals),
            "reversal_example": examples[0] if examples else None,
            "reversals": [e["pair"] for e in examples],
            "changed_pairs": changed,
        },
        "scores": {"ids": ids, "course": original, **{f"alpha={a:g}": raw_by_alpha[a] for a in grid}},
    }


def _describe(record: ArchRecord, terms: ArchTerms, alpha: float, course_score: float) -> Dict[str, object]:
    """Per-layer breakdown of one architecture under the selected alpha."""
    config = record.config if isinstance(record.config, dict) else {}
    return {
        "arch_id": record.arch_id, "width": record.width, "depth": record.depth,
        "params": record.params, "perplexity": record.perplexity,
        "n_head": config.get("n_head"), "d_inner": config.get("d_inner"),
        "course_nsc": course_score, "ca_nsc": float(context_aware_score(terms, alpha)),
        "plain_sum": float(plain_sum(terms)), "ffn_loss": float(ffn_loss(terms, alpha)),
        "attention_total": math.fsum(float(a) for a in terms.attention),
        "ffn_total": math.fsum(float(f) for f in terms.ffn), "other": float(terms.other),
        "layers": [{k: float(v) for k, v in row.items()} for row in layer_profile(terms, alpha)],
    }
