"""Binding between this project and the course NSC starter (``starter/``).

STATUS: BOUND to the course starter ``CS5491-202627A-v1`` (nsc-starter.zip,
unmodified). Nothing here re-implements NSC:

* ``nsc.layers`` parses a configuration into hidden width d, per-layer head
  counts H_l and FFN widths f_l;
* ``nsc.psi_mp`` is the course MP capacity of one weight matrix (sigma = 0.02,
  512 Simpson steps);
* ``nsc.original_nsc`` is the reference score.

The course per-layer capacity is ``3*H_l*psi(d, d/H_l) + psi(d, d) + 2*psi(d, f_l)``
(Q/K/V per head, one output projection, two FFN projections; README "Baseline
conventions"). The course code only returns the per-layer sum, so
``layer_terms`` evaluates the same expression in two parts:

    A_l = 3*H_l*psi(d, d/H_l) + psi(d, d)      attention contribution
    F_l = 2*psi(d, f_l)                         raw FFN capacity
    G   = 0                                     (embeddings, biases, norms and
                                                 the LM head are not in the score)

``A_l + F_l`` is the course expression term for term, so it equals
``nsc.layer_capacities`` exactly; the runner checks this, and that
``G + sum_l (A_l + F_l)`` equals ``nsc.original_nsc``, on every development
architecture.

Data (starter README "Files and schema"): ``d_model``, ``n_layer``, ``n_head``
and ``d_inner`` from data/architectures.json; the parameter count is
``resources[id]["total"]``; perplexity is ``valid_ppl`` from
data/development_labels.json; the split is data/splits.json. The final label
file is never opened here.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from .context_nsc import ArchTerms, LayerTerms
from .records import DEVELOPMENT, FINAL, ArchRecord

PROJECT_DIR = Path(__file__).resolve().parent.parent
STARTER_DIR = Path(os.environ.get("CANSC_STARTER_DIR", str(PROJECT_DIR / "starter")))

# True: the functions below call the course starter.
BOUND = True

REQUIRED_FILES = ("nsc.py", "evaluate.py", "data/architectures.json", "data/resources.json",
                  "data/splits.json", "data/development_labels.json")
LABEL_FIELD = {DEVELOPMENT: "valid_ppl"}

_MODULES: Dict[str, object] = {}


class StarterUnavailable(RuntimeError):
    """Raised when the course starter is missing or the adapter is not bound."""


def _display(path: Path) -> str:
    """Project-relative path for messages (avoids printing home directories)."""
    try:
        return str(path.resolve().relative_to(PROJECT_DIR))
    except ValueError:
        return str(path)


def missing_items() -> List[str]:
    """What is missing before the development evaluation can run."""
    items = []
    if not STARTER_DIR.is_dir():
        items.append(f"course starter folder not found at {_display(STARTER_DIR)}/ (or set CANSC_STARTER_DIR)")
    else:
        items += [f"{_display(STARTER_DIR / name)} is missing" for name in REQUIRED_FILES
                  if not (STARTER_DIR / name).is_file()]
    if not BOUND:
        items.append("cansc/course_adapter.py is not bound to the starter (BOUND = False)")
    return items


def _require() -> None:
    missing = missing_items()
    if missing:
        raise StarterUnavailable("; ".join(missing))


def _load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, STARTER_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def course_nsc():
    """The course ``nsc`` module (registered as ``nsc``, which evaluate.py imports).

    If the course evaluator has already imported the starter's nsc.py (when
    student_ca_nsc.py runs under starter/evaluate.py), that module is reused.
    """
    _require()
    if "nsc" not in _MODULES:
        existing = sys.modules.get("nsc")
        path = getattr(existing, "__file__", None)
        if path and Path(path).resolve() == (STARTER_DIR / "nsc.py").resolve():
            _MODULES["nsc"] = existing
        else:
            _MODULES["nsc"] = _load_module("nsc", "nsc.py")
    return _MODULES["nsc"]


def harness():
    """The course ``evaluate`` module: canonical, ranks, correlation, pair_summary."""
    course_nsc()
    if "evaluate" not in _MODULES:
        _MODULES["evaluate"] = _load_module("nsc_starter_evaluate", "evaluate.py")
    return _MODULES["evaluate"]


def clear_cache() -> None:
    """Empty the course MP cache, so each timed pass starts cold."""
    course_nsc().psi_mp.cache_clear()


def _read(name: str):
    return json.loads((STARTER_DIR / "data" / name).read_text(encoding="utf-8"))


def _split_keys(name: str, allow_final: bool) -> List[str]:
    if name == FINAL and not allow_final:
        raise PermissionError("the final split is held out until the final evaluation")
    if name not in (DEVELOPMENT, FINAL):
        raise ValueError(f"unknown split {name!r}")
    _require()
    return list(_read("splits.json")[name])


def terms_from_config(arch_id: str, config: dict) -> ArchTerms:
    """Course attention / FFN terms of one configuration (see module docstring)."""
    nsc = course_nsc()
    d, heads, inner = nsc.layers(config)
    layers = tuple(
        LayerTerms(attention=3 * h * nsc.psi_mp(d, d // h) + nsc.psi_mp(d, d), ffn=2 * nsc.psi_mp(d, f))
        for h, f in zip(heads, inner))
    return ArchTerms(arch_id=arch_id, width=d, layers=layers, other=0.0)


def unlabelled_terms(name: str = DEVELOPMENT, *, allow_final: bool = False) -> List[ArchTerms]:
    """Layer terms of one split from configurations only (no label file is read).

    Used to instantiate the alpha grid before any perplexity is loaded.
    """
    configs = _read("architectures.json")
    return [terms_from_config(k, configs[k]) for k in _split_keys(name, allow_final)]


def load_split(name: str, *, allow_final: bool = False) -> List[ArchRecord]:
    """ArchRecords of one official split, in splits.json order.

    The final split stays held out: asking for it without ``allow_final=True``
    raises, and the milestone runner never passes that flag.
    """
    keys = _split_keys(name, allow_final)
    configs = _read("architectures.json")
    resources = _read("resources.json")
    labels = _read(f"{name}_labels.json")
    field = LABEL_FIELD.get(name, "test_ppl")
    records = []
    for k in keys:
        cfg = configs[k]
        records.append(ArchRecord(arch_id=k, split=name, width=cfg["d_model"], depth=cfg["n_layer"],
                                  params=resources[k]["total"], perplexity=labels[k][field], config=cfg))
    return records


def original_nsc(record: ArchRecord) -> float:
    """Course NSC score of one architecture (direct call, unchanged)."""
    return course_nsc().original_nsc(record.config)


def layer_terms(record: ArchRecord) -> ArchTerms:
    """Course per-block attention contributions and raw FFN capacities."""
    return terms_from_config(record.arch_id, record.config)


def layer_capacities(record: ArchRecord) -> List[float]:
    """Course per-layer capacities (``nsc.layer_capacities``), for the decomposition check."""
    return course_nsc().layer_capacities(record.config)


def single_block(record: ArchRecord) -> Tuple[float, ArchTerms]:
    """Course NSC and layer terms of a one-block copy of the architecture.

    The archive has no single-block architecture (depth 2-16), so the
    single-layer check keeps block 1 (its head count and FFN width) and scores
    the copy with the course code.
    """
    nsc = course_nsc()
    cfg = copy.deepcopy(record.config)
    _, heads, inner = nsc.layers(cfg)
    cfg.update(n_layer=1, n_head=[heads[0]], d_inner=[inner[0]])
    return nsc.original_nsc(cfg), terms_from_config(f"{record.arch_id}#block1", cfg)
