# Experimental protocol — CA-NSC milestone

Fixed on 2026-11-19, before any development-split perplexity or score was seen
(none was available: the course archive was not yet in the repository). The
code that applies each rule is named in brackets.

## 1. Data and split

- Use the course archive and its **official** split: 100 development and 100
  final architectures. The milestone uses the development split only.
- The final split is held out until the final evaluation. `load_split("final")`
  raises unless `allow_final=True`, which `run_milestone.py` never passes
  [`cansc/course_adapter.py`].
- Target: recorded perplexity; a good score is high when perplexity is low, so
  every correlation is computed against **negative** perplexity.

## 2. Methods compared in the milestone

- **NSC (course)**: a direct call to the course scorer, unchanged.
- **CA-NSC(α)**: with the course per-block attention contribution `A_l`, raw FFN
  capacity `F_l`, non-block terms `G` and hidden width `d`,

  ```
  C_l = (F_1 + ... + F_{l-1}) / d        (C_1 = 0)
  w_l = 1 / (1 + α C_l)
  S_α = G + Σ_l (A_l + w_l F_l)
  ```

  Only raw capacities enter `C_l` [`cansc/context_nsc.py`].
- **α = 0** is the ablation and must reproduce the course NSC (Section 5).

## 3. α grid (rule fixed now; numbers instantiated from unlabelled data)

1. For every development architecture compute `C_l` from the course FFN
   capacities and hidden width only (no perplexity), and pool the values of
   all layers `l ≥ 2` (`C_1 = 0` by definition).
2. `C_ref` = median of the pooled values.
3. Non-zero candidates: `α_k = round_sig1(m_k / C_ref)` for
   `m = (0.1, 0.3, 1, 3)`, rounded to one significant figure, half up, on the
   decimal value. Grid = `{0, α_1, α_2, α_3, α_4}`.
4. Why: at `C = C_ref` the unrounded values give `w = 1/(1+m)` ≈ 0.91, 0.77,
   0.50 and 0.25, i.e. light, mild, medium and strong discounting of a typical
   block; the deepest blocks (largest `C_l`) are discounted more, so the
   strongest value also shows whether over-discounting hurts. Four non-zero
   values and one significant figure keep the search small relative to 100
   development architectures.
5. The grid is written once to `results/protocol_frozen.json` (with `C_ref`,
   the pooled `C_l` quantiles, the weight each α gives at `C_ref` and at the
   largest `C_l`, and `labels_used: false`) and never overwritten; loading a
   file frozen under a different rule raises [`cansc/protocol.py`].

**Instantiated grid** (`results/protocol_frozen.json`, frozen at
2026-11-20T07:50:53+00:00 from the development configurations alone, before the
label file was opened): 826 pooled values of `C_l`
(layers `l ≥ 2` of 100 architectures, depth 2–16),
min 0.385, q25 2.219, median `C_ref` = 4.4221, q75 7.165,
max 13.508. Grid = `{0, 0.02, 0.07, 0.2, 0.7}`; weights at
`C_ref`: 0.919, 0.764, 0.531, 0.244; at the largest `C_l`:
0.787, 0.514, 0.270, 0.096.

## 4. Primary metric and selection rule

- Primary metric: Spearman ρ between the score and negative perplexity on the
  development split, with average ranks for tied values (score ties and
  perplexity ties alike).
- Select the α with the highest **unrounded** ρ. Exact ties go to the smaller
  α. α = 0 is allowed to win; if it does, the conclusion is that the
  discount did not help on the development split.
- Every candidate is reported, not only the winner. Kendall τ_b is computed as
  a secondary statistic and is not used for selection [`cansc/stats.py`].

## 5. Correctness checks that must pass before results are used

- α = 0 versus course NSC, architecture by architecture on the whole
  development split: `|S_CA(0) − S_course| ≤ 1e-9 · max(1, |S_course|)`; the
  maximum absolute error is reported.
- First-block weight exactly 1 for every development architecture and every
  grid value.
- Single-block architectures: score unchanged for every α ≥ 0 (archived ones if
  the archive has any; exact synthetic cases otherwise).
- Hand-computable FFN reordering example: program output equals the hand
  calculation exactly (fractions) [`cansc/toy.py`].

## 6. Size-control check

- Pairs: two development architectures with the same hidden width and
  `min(params)/max(params) ≥ 0.90`, using the archive's reported parameter
  count; the ratio is compared exactly and the bound is inclusive
  [`cansc/pairs.py`].
- Verdict of a score on a pair: *correct* if the lower-perplexity architecture
  has the strictly higher score, *wrong* if the reverse, *score tie* if the
  scores agree to a relative tolerance of 1e-12.
- Pairs with identical recorded perplexity are *truth ties*: no order is
  correct, so they are excluded from the three counts and reported separately.
- Compare course NSC with the selected α, report the verdict transitions, and
  analyse one real pair reversed from correct to wrong (the one with the
  largest perplexity gap), if any.

## 7. Reporting

- Main table: method, α, n, Spearman ρ, score minimum and maximum, scoring time
  (median of 5 runs, total over the split, including the course decomposition
  for CA-NSC), FFN weight minimum and median over all (architecture, block
  `l ≥ 2`) pairs, and the median retained FFN fraction `Σ w_l F_l / Σ F_l`.
- Rank change between course NSC and the selected α: Spearman between the two
  score vectors, number of architectures whose rank changes, mean and maximum
  rank shift, number of reversed architecture pairs.

## 8. Deferred to the final stage

Full baseline comparison (course NSC, positional example, ablation, reported
parameter count, reference FLOPs), Kendall τ_b and ties, budgeted selection
under 10M / 20M / 30M / 50M parameter caps with one or three candidates per
method, best selected perplexity and regret against the supplied 200-trial
uniform random selection, and the final-split evaluation with every choice
above frozen.

## 9. Amendment A — binding to the course starter

Written on 2026-11-19 after the course starter (`CS5491-202627A-v1`) arrived
and before any development score, perplexity or reference output was
computed or opened. It fixes what the starter determines; Sections 1–8 stand.

1. **Terms.** From the starter README and `nsc.py`, the course layer value is
   `3·H_l·ψ(d, d/H_l) + ψ(d, d) + 2·ψ(d, f_l)` with `ψ = nsc.psi_mp` (σ = 0.02).
   CA-NSC uses `A_l = 3·H_l·ψ(d, d/H_l) + ψ(d, d)`, `F_l = 2·ψ(d, f_l)`,
   `G = 0`, `d = d_model`; `H_l`, `f_l` come from `n_head`, `d_inner` via
   `nsc.layers`. Parameter count: `resources[id]["total"]`. Label:
   `valid_ppl` from `development_labels.json`.
2. **Ranking statistics follow the course evaluator.** Scores are canonicalised
   to 12 significant digits (`evaluate.canonical`) before ranks, ties and pair
   verdicts. Spearman ρ is `evaluate.correlation(evaluate.ranks(score),
   evaluate.ranks(−ppl))`; Kendall τ_b comes from `evaluate.pair_summary`. The
   coefficient itself is never rounded, and selection (Section 4) uses it
   unrounded. This package's own statistics recompute both as a cross-check.
3. **Size control uses the course rule**: same `d_model` and
   `min(total)/max(total) ≥ 0.90` (the starter README expects 180 development
   pairs). Verdicts use canonical scores. The official counts are those of
   `evaluate.pair_summary`: concordant = correct, discordant = wrong,
   score-only ties = score ties, quality-only plus both ties = truth ties.
4. **Single-block check.** The archive has depth 2–16 and no single-block
   architecture, so a one-block copy (block 1) of every development
   architecture is scored by the course code and compared with CA-NSC for
   every grid value and for α = 10³ and 10⁶.
5. **Per-layer check.** `A_l + F_l` must equal `nsc.layer_capacities` exactly
   for every development layer, in addition to the total check of Section 5.
6. **Timing.** `nsc.psi_mp` is cached, so the cache is emptied before every
   timed pass and the reported time includes the MP integrals; the median of
   5 passes is reported. Times remain machine-dependent diagnostics.
7. **Order of operations.** (a) Instantiate and freeze the α grid from the
   development configurations alone (`course_adapter.unlabelled_terms` never
   opens a label file). (b) Load development labels and run the evaluation.
   (c) Only then cross-check with the unmodified course evaluator
   (`starter/evaluate.py --student student_ca_nsc.py`, where `score` is CA-NSC
   at the selected α and `ablation` is α = 0) and compare the original-NSC
   numbers with `expected_development.json`. The evaluator's budgeted-selection
   output on the development split is kept as a raw file but is not analysed in
   the milestone (Section 8). `final_labels.json` is never opened; the course's
   `check_starter.py` only hashes its bytes.

## Change log

- 2026-11-19: protocol written. The rounding convention in 3.3 (half up on the
  decimal value) was fixed while writing unit tests, before any data existed.
- 2026-11-19: Amendment A (Section 9) added when the course starter arrived,
  before any development score, label or reference output was seen.
- 2026-11-19: grid instantiated (Section 3) and the development evaluation run
  once; the selected α (0.7, the largest grid value) is frozen in
  `student_ca_nsc.py`. No rule was changed after the results were seen.
