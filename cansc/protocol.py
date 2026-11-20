"""Pre-registered alpha grid and selection rule.

Everything here is fixed before any development perplexity is read; see
PROTOCOL.md for the full text. The numeric grid is instantiated once from the
*unlabelled* development layer terms and frozen in results/protocol_frozen.json.

Grid rule
---------
1. For every development architecture compute C_l from the course FFN
   capacities and the hidden width (no perplexity involved). Pool C_l over all
   layers l >= 2 (C_1 = 0 by definition).
2. C_ref = median of the pooled values.
3. Non-zero candidates alpha_k = round_sig1(m_k / C_ref) with
   m = (0.1, 0.3, 1, 3), rounded to one significant figure, half up, on the
   decimal representation. Before rounding these give w = 1 / (1 + m) at
   C = C_ref, i.e. about 0.91, 0.77, 0.50 and 0.25: light, mild, medium and
   strong discounting of a typical layer.
4. Grid = (0, alpha_1, ..., alpha_4). alpha = 0 is the original-NSC ablation.

Selection rule
--------------
Primary metric: Spearman rho between the score and negative perplexity on the
official development split, average ranks for ties. The alpha with the highest
*unrounded* rho is selected; exact ties go to the smaller alpha; alpha = 0 may
win. The final split is not used.
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from .context_nsc import ArchTerms, cumulative_capacity, ffn_weights

RULE_VERSION = "ca-nsc-grid-v1"
ALPHA_MULTIPLIERS: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0)
PRIMARY_METRIC = "Spearman rho(score, -perplexity), average ranks for ties"
TIE_RULE = "highest unrounded rho; exact ties -> smaller alpha; alpha = 0 may win"


def pooled_capacities(terms_list: Iterable[ArchTerms]) -> List[float]:
    """C_l of every layer l >= 2 of every architecture (label-free)."""
    pooled = []
    for terms in terms_list:
        pooled.extend(float(c) for c in cumulative_capacity(terms.ffn, terms.width)[1:])
    return pooled


def quantile(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolation quantile (same as numpy's default) of sorted data."""
    if not sorted_values:
        raise ValueError("empty data")
    pos = (len(sorted_values) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def round_sig1(x: float) -> float:
    """Round a positive number to one significant figure, half up.

    Rounding acts on the shortest decimal representation of ``x`` (its repr),
    so 0.15 -> 0.2 and 0.0473 -> 0.05 regardless of binary representation.
    """
    if x <= 0 or not math.isfinite(x):
        raise ValueError(f"expected a positive finite number, got {x!r}")
    d = Decimal(repr(x))
    return float(d.quantize(Decimal(1).scaleb(d.adjusted()), rounding=ROUND_HALF_UP))


def instantiate_grid(terms_list: Sequence[ArchTerms]) -> Dict[str, object]:
    """Apply the grid rule to unlabelled development layer terms.

    Returns a JSON-serialisable dict with the grid, C_ref, the pooled C_l
    quantiles and the weight each alpha gives at C_ref.
    """
    pooled = sorted(pooled_capacities(terms_list))
    if not pooled:
        raise ValueError("no layer l >= 2 in the development split: C_ref undefined")
    c_ref = statistics.median(pooled)
    if c_ref <= 0:
        raise ValueError(f"C_ref must be positive, got {c_ref}")
    nonzero = []
    for m in ALPHA_MULTIPLIERS:
        a = round_sig1(m / c_ref)
        if a not in nonzero:
            nonzero.append(a)
    grid = [0.0] + nonzero
    depths = [t.depth for t in terms_list]
    return {
        "rule_version": RULE_VERSION,
        "alpha_multipliers": list(ALPHA_MULTIPLIERS),
        "primary_metric": PRIMARY_METRIC,
        "tie_rule": TIE_RULE,
        "n_architectures": len(terms_list),
        "n_pooled_layers": len(pooled),
        "depth_min": min(depths),
        "depth_max": max(depths),
        "c_ref": c_ref,
        "c_quantiles": {
            "min": pooled[0],
            "q10": quantile(pooled, 0.10),
            "q25": quantile(pooled, 0.25),
            "q50": quantile(pooled, 0.50),
            "q75": quantile(pooled, 0.75),
            "q90": quantile(pooled, 0.90),
            "max": pooled[-1],
        },
        "grid": grid,
        "weight_at_c_ref": {str(a): 1.0 / (1.0 + a * c_ref) for a in grid},
        "weight_at_c_max": {str(a): 1.0 / (1.0 + a * pooled[-1]) for a in grid},
        "labels_used": False,
    }


def freeze_or_load(terms_list: Sequence[ArchTerms], path: Path) -> Dict[str, object]:
    """Load the frozen protocol if it exists, otherwise instantiate and write it.

    An existing file is never overwritten; delete it deliberately to re-freeze.
    Raises if the stored rule differs from the current code.
    """
    path = Path(path)
    if path.exists():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        if frozen.get("rule_version") != RULE_VERSION or \
                tuple(frozen.get("alpha_multipliers", ())) != ALPHA_MULTIPLIERS:
            raise RuntimeError(f"{path} was frozen under a different rule; refusing to continue")
        frozen["loaded_existing"] = True
        return frozen
    frozen = instantiate_grid(terms_list)
    frozen["frozen_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    frozen["loaded_existing"] = False
    return frozen


def select_alpha(rho_by_alpha: Sequence[Tuple[float, float]]) -> float:
    """Selection rule: max unrounded rho, exact ties to the smaller alpha.

    Args:
        rho_by_alpha: (alpha, rho) pairs; NaN rho values are ignored.
    """
    valid = [(a, r) for a, r in rho_by_alpha if not math.isnan(r)]
    if not valid:
        raise ValueError("no valid Spearman value to select from")
    best = max(r for _, r in valid)
    return min(a for a, r in valid if r == best)


def weight_summary(terms_list: Sequence[ArchTerms], alpha: float) -> Dict[str, float]:
    """Min and median FFN weight over all (architecture, layer >= 2) pairs.

    Also returns the median per-architecture retained FFN fraction
    sum_l w_l F_l / sum_l F_l (architectures with zero FFN capacity skipped)
    and the median FFN share of the score, sum_l w_l F_l / S_alpha.
    """
    pooled = []
    retained = []
    shares = []
    for terms in terms_list:
        weights = ffn_weights(terms.ffn, terms.width, alpha)
        pooled.extend(float(w) for w in weights[1:])
        total_f = math.fsum(float(f) for f in terms.ffn)
        kept = math.fsum(float(w) * float(f) for w, f in zip(weights, terms.ffn))
        if total_f > 0:
            retained.append(kept / total_f)
        score = float(terms.other) + math.fsum(float(a) for a in terms.attention) + kept
        if score > 0:
            shares.append(kept / score)
    out = {"scope": "all (architecture, layer l >= 2) pairs of the split"}
    if pooled:
        out["w_min"] = min(pooled)
        out["w_median"] = statistics.median(pooled)
        out["n_layers"] = len(pooled)
    if retained:
        out["retained_median"] = statistics.median(retained)
    if shares:
        out["ffn_share_median"] = statistics.median(shares)
    return out
