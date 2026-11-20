"""CA-NSC packaged as a course ``student_score`` module.

Run with the unmodified course evaluator, from the project folder:

    CANSC_ALPHA=<alpha> python3 starter/evaluate.py --split development \
        --student student_ca_nsc.py --output <file>.json

score(config)     CA-NSC with alpha from the CANSC_ALPHA environment variable,
                  or SELECTED_ALPHA (the value selected on the development
                  split under PROTOCOL.md) when the variable is unset.
ablation(config)  the same code path with alpha = 0: no context discount, which
                  is exactly original NSC.

Both functions see the configuration only (no labels, no architecture id).
The attention / FFN terms come from the course nsc.py via cansc.course_adapter.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cansc.context_nsc import context_aware_score  # noqa: E402
from cansc.course_adapter import terms_from_config  # noqa: E402

# Development choice frozen on 2026-11-19 by the PROTOCOL.md selection rule
# (highest development Spearman over the grid 0, 0.02, 0.07, 0.2, 0.7; see
# results/main_table.md). Do not change it after looking at the final split.
SELECTED_ALPHA = 0.7


def _alpha() -> float:
    value = os.environ.get("CANSC_ALPHA")
    if value is not None:
        return float(value)
    if SELECTED_ALPHA is None:
        raise RuntimeError("set CANSC_ALPHA or SELECTED_ALPHA")
    return float(SELECTED_ALPHA)


def score(config):
    """CA-NSC: course attention terms plus context-discounted FFN terms."""
    return context_aware_score(terms_from_config("config", config), _alpha())


def ablation(config):
    """Mechanism removed (alpha = 0): every FFN weight is 1, i.e. original NSC."""
    return context_aware_score(terms_from_config("config", config), 0.0)
