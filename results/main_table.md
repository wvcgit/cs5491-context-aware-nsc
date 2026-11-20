# Development split results (n = 100, official split; final split not used)

Frozen grid: 0, 0.02, 0.07, 0.2, 0.7 (rule ca-nsc-grid-v1; C_ref = 4.4221, the median of 826 unlabelled C_l values, layers l >= 2; C_l quantiles min 0.385, q25 2.219, q75 7.165, max 13.508).
Timing: median of the repeats; total for the split; MP cache emptied before each pass.
Spearman and Kendall use scores canonicalised to 12 significant digits (course evaluator); score ranges are unrounded.

| method | alpha | n | Spearman (score, -ppl) | Kendall tau-b | score min | score max | time (ms) | FFN w min | FFN w median | retained FFN (median) | FFN share of score (median) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NSC (course) | -- | 100 | 0.969745 | 0.861414 | 116.105 | 13046.8 | 105.5 | -- | -- | -- | -- |
| CA-NSC | 0 | 100 | 0.969745 | 0.861414 | 116.105 | 13046.8 | 105.9 | 1.0000 | 1.0000 | 1.0000 | 0.7421 |
| CA-NSC | 0.02 | 100 | 0.971305 | 0.866667 | 115.347 | 12250.6 | 106.4 | 0.7873 | 0.9187 | 0.9408 | 0.7319 |
| CA-NSC | 0.07 | 100 | 0.972877 | 0.873535 | 113.579 | 10993 | 108.0 | 0.5140 | 0.7636 | 0.8278 | 0.7101 |
| CA-NSC | 0.2 | 100 | 0.975302 | 0.882020 | 109.684 | 9474.88 | 111.0 | 0.2702 | 0.5307 | 0.6530 | 0.6715 |
| CA-NSC (selected) | 0.7 | 100 | 0.975674 | 0.889697 | 100.326 | 7865.18 | 106.9 | 0.0956 | 0.2442 | 0.4103 | 0.5540 |

FFN weight statistics: all (architecture, layer l >= 2) pairs of the development split; retained FFN = sum_l w_l F_l / sum_l F_l per architecture.

Selected alpha (max unrounded Spearman, ties to smaller alpha): **0.7**.
Rank change course -> selected (canonical scores): Spearman between scores 0.980102; 75 architectures changed rank; mean |shift| 4.020; max |shift| 15.0; 236 of 4950 pairs reversed.

## Correctness checks on the development architectures
- alpha = 0 vs course NSC (`nsc.original_nsc`): max abs error 0.000e+00 (bitwise equal for 100 of 100); tolerance |S_CA(0) - S_course| <= 1e-09 * max(1, |S_course|); failures 0.
- A_l + F_l vs `nsc.layer_capacities`: 926 layers, 0 not bitwise equal, max abs error 0.000e+00.
- first-layer weight exactly 1 in all 500 (architecture, alpha) cases: True.
- archived single-layer architectures: 0.
- one-block copies (block 1 of each development architecture, scored by the course code): 100 configurations x alphas 0, 0.02, 0.07, 0.2, 0.7, 1000, 1e+06: max abs error 0.000e+00; all bitwise equal: True.
- own Spearman implementation vs course `ranks`/`correlation`: max abs difference 0.0e+00.
