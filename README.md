# Context-Aware NSC (CA-NSC) — CS5491 project milestone

Milestone code for the CS5491 course project *Context-Aware NSC Scoring for
Transformer Architecture Selection*. Neural Spectral Capacity
(NSC) scores a Transformer by adding Marchenko–Pastur capacity estimates of its
weight matrices. CA-NSC changes one thing: the FFN term of block `l` is
multiplied by

```
w_l = 1 / (1 + α C_l),    C_l = (F_1 + ... + F_{l-1}) / d
```

where, in the course scorer's terms, `F_j = 2·ψ(d, f_j)` is the raw FFN
capacity of block `j` (`ψ = nsc.psi_mp`, `f_j` from `d_inner`) and
`d = d_model`. The attention term `A_l = 3·H_l·ψ(d, d/H_l) + ψ(d, d)` is
unchanged; `α = 0` is exactly the course NSC.

## Milestone results (official development split, n = 100)

| method | α | Spearman ρ (score, −PPL) | Kendall τ_b | size-control pairs correct / wrong |
|---|---|---|---|---|
| NSC (course) | – | 0.969745 | 0.861414 | 146 / 34 |
| CA-NSC (ablation) | 0 | 0.969745 | 0.861414 | 146 / 34 |
| CA-NSC (selected) | 0.7 | 0.975674 | 0.889697 | 149 / 31 |

- α grid `{0, 0.02, 0.07, 0.2, 0.7}` frozen from unlabelled `C_l` before any
  label was read; ρ grows with α over the whole grid, so the selected value is
  the grid edge. Full table: `results/main_table.md`.
- α = 0 equals `nsc.original_nsc` bit for bit on all 100 architectures; the
  unmodified course evaluator reproduces every number above exactly
  (`results/crosscheck.md`).
- At α = 0.7 the median FFN weight is 0.244 (minimum 0.096). In the 180
  size-control pairs CA-NSC fixes 5 and breaks 2; the changes mostly favour
  models with more attention and less FFN capacity (`results/size_control.md`).
- The final split has not been used.

## Course starter (not included)

The course starter is course material and is not redistributed in this
repository. The results were produced with the unmodified starter
`CS5491-202627A-v1` (`nsc-starter.zip`, SHA-256
`0ec560225cbac0acbc84be1cee656aed7bb6f7ee7078545d5c422e873fedee12`). To
reproduce them, unzip it next to this README so that its files sit in
`starter/` (or point `CANSC_STARTER_DIR` to another folder):

```bash
unzip nsc-starter.zip && mv nsc-starter starter
```

```bash
cd starter && python3 check_starter.py && cd ..
```

The binding is `cansc/course_adapter.py`: it calls `nsc.layers`,
`nsc.psi_mp`, `nsc.original_nsc` and the evaluator's `canonical`, `ranks`,
`correlation` and `pair_summary`; it never re-implements NSC and never opens
`data/final_labels.json`. Without `starter/`, the tests skip the 5 starter
tests and `run_milestone.py` runs only the synthetic checks, leaving the
committed development results unchanged.

The milestone report draft is submitted through Canvas and is not part of this
repository; `run_milestone.py` still writes the numbers it uses to
`report/generated/`.

## Environment

- Python 3.9 or later, standard library only (`requirements.txt` is empty on
  purpose). Developed with Python 3.11.2 on macOS (Apple laptop), CPU only.

## Reproduce

Run from the repository root, after placing the course starter in `starter/`:

```bash
python3 -m unittest discover -s tests -t . -v
```

Expected: `Ran 46 tests ... OK` (about 0.1 s; `OK (skipped=5)` without
`starter/`).

```bash
python3 run_milestone.py
```

Expected (about 4 s; ρ values are deterministic, timings are not):

```
[1] synthetic property checks: 2000 float + 300 exact cases, 0 violations
[2] toy reordering: exact match with hand calculation = True
      ...
[3] course starter: bound (starter/)
[4a] alpha grid loaded: [0.0, 0.02, 0.07, 0.2, 0.7] (C_ref = 4.4221; C_l from 0.385 to 13.508)
[4b] n = 100; alpha = 0 vs course NSC: max abs error 0.0e+00 (passed = True); selected alpha = 0.7
      NSC (course)  alpha=    --  rho=0.969745 ...
      CA-NSC        alpha=   0.7  rho=0.975674  w_min=0.096
      size control: 180 pairs; course {'correct': 146, 'wrong': 34, ...}; selected {'correct': 149, 'wrong': 31, ...}
[4c] course evaluator cross-check: max abs difference 0.0e+00; pairs 180; original NSC equals reference output = True
[4d] starter reorder pair: NSC tie True, ablation = NSC True, CA-NSC prefers larger FFN first True
[5] report numbers written to report/generated/
```

`[4a]` says `loaded` because `results/protocol_frozen.json` is part of the
repository; deleting it re-freezes the grid from the configurations alone and
gives the same values.

The course evaluator can also be run directly on CA-NSC (score = CA-NSC at the
frozen α = 0.7, ablation = α = 0):

```bash
python3 starter/evaluate.py --split development --student student_ca_nsc.py
```

## Method details and protocol

- `PROTOCOL.md`: the selection protocol, fixed before any development result
  (metric, tie rule, α-grid rule, size-control rule), its pre-result amendment
  for the starter's conventions, and the instantiated grid.
- Ranking statistics follow the course evaluator: scores are rounded to 12
  significant digits before ranking, ties get average ranks, and Spearman is
  computed against negative perplexity. Selection uses the unrounded
  correlation coefficient.
- Size control: same `d_model` and `min(total)/max(total) ≥ 0.90`, with `total`
  from `data/resources.json` (180 development pairs).

## Files

| path | content |
|---|---|
| `cansc/context_nsc.py` | the reweighting: `cumulative_capacity`, `ffn_weights`, `context_aware_score`, `plain_sum`, `ffn_loss`, `layer_profile` |
| `cansc/course_adapter.py` | binding to the course starter (terms, records, single-block copies, evaluator functions) |
| `cansc/evaluate.py` | development evaluation: checks, main table, rank change, size control, reversal analysis |
| `cansc/protocol.py` | α-grid rule, freezing, selection rule, FFN-weight summary |
| `cansc/stats.py`, `cansc/pairs.py` | own rank statistics and pair rule (cross-checks of the course functions) |
| `cansc/checks.py`, `cansc/toy.py` | synthetic property checks; hand-computable reordering example |
| `cansc/report_tex.py` | writes `report/generated/*.tex` from `results/` |
| `student_ca_nsc.py` | CA-NSC in the course `student_score` interface (α frozen at 0.7) |
| `run_milestone.py` | runs everything above |
| `results/main_table.md` | main development table and the checks on real architectures |
| `results/size_control.md`, `results/reversal_example.md` | pair verdicts, changed pairs, per-layer analysis of a reversed pair |
| `results/crosscheck.md`, `results/course_evaluator_development.json` | comparison with the unmodified course evaluator (its budgeted-selection output is not analysed in the milestone) |
| `results/protocol_frozen.json`, `results/dev_results.json`, `results/dev_scores.csv` | frozen grid; full machine-readable results; per-architecture scores |
| `results/toy_reordering.md`, `results/course_reorder_pair.md` | hand-computable example; the starter's reorder pair with course capacities |
| `results/checks_synthetic.json`, `results/status.json` | synthetic property checks; run status |
| `tests/` | unit tests, a synthetic pipeline test and starter integration tests |
| `starter/` (not included) | place the course starter here to reproduce the development results |

The synthetic architectures in `tests/` exercise the code only; their numbers
never reach `results/` or the report.
