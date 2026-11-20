# Size-control check (development split)

Rule: same d_model; min(total)/max(total) >= 0.90 (exact; equals the course float rule: True). Qualifying pairs: **180**; pairs with identical recorded perplexity (truth ties, excluded from the counts below): 0.
Scores canonicalised to 12 significant digits, as in the course evaluator.

| method | correct | wrong | score tie | tau-b (course pair_summary) |
|---|---|---|---|---|
| NSC (course) | 146 | 34 | 0 | 0.622222 |
| CA-NSC (alpha = 0.7) | 149 | 31 | 0 | 0.655556 |

Counts equal the course `pair_summary` counts: True.

Verdict transitions course -> selected: correct->correct: 144, correct->wrong: 2, wrong->correct: 5, wrong->wrong: 29

Pairs that the selected alpha reverses from correct to wrong: 2 (config_11_j1 vs config_19_j6; config_14_j3 vs config_2_j9).

## Pairs whose verdict changed (better = lower perplexity)

| transition | better | worse | depth b/w | attention b/w | raw FFN b/w | ppl b/w |
|---|---|---|---|---|---|---|
| correct->wrong | config_19_j6 | config_11_j1 | 9/10 | 57.84/64.38 | 467.72/453.22 | 55.454/56.245 |
| wrong->correct | config_2_j8 | config_13_j5 | 12/11 | 302.99/278.47 | 1189.82/1289.92 | 38.965/41.357 |
| wrong->correct | config_9_j5 | config_13_j5 | 11/11 | 278.25/278.47 | 1224.36/1289.92 | 39.204/41.357 |
| correct->wrong | config_2_j9 | config_14_j3 | 15/15 | 5516.63/5532.59 | 6703.52/6670.12 | 25.794/25.880 |
| wrong->correct | config_16_j0 | config_7_j0 | 13/12 | 1269.97/1174.41 | 2399.71/2595.71 | 29.954/31.599 |
| wrong->correct | config_16_j2 | config_3_j3 | 14/13 | 353.71/328.30 | 1288.33/1384.33 | 36.854/37.041 |
| wrong->correct | config_2_j8 | config_9_j5 | 12/11 | 302.99/278.25 | 1189.82/1224.36 | 38.965/39.204 |
