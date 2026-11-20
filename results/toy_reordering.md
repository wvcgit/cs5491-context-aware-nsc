# Toy FFN reordering example (synthetic capacities)

alpha = 1/2, A_l = 1 for every layer, G = 0. C_l = (F_1+...+F_{l-1})/d, w_l = 1/(1+alpha*C_l), S = sum A_l + sum w_l F_l.

| order (F1,F2,F3) | d | C | w | sum w*F | loss sum (1-w)F | S (alpha=0) | S (program) | S (hand) | exact match |
|---|---|---|---|---|---|---|---|---|---|
| (8, 4, 4) | 4 | (0, 2, 3) | (1, 1/2, 2/5) | 58/5 | 22/5 | 19 | 73/5 = 14.6000 | 73/5 | True |
| (4, 8, 4) | 4 | (0, 1, 3) | (1, 2/3, 2/5) | 164/15 | 76/15 | 19 | 209/15 = 13.9333 | 209/15 | True |
| (4, 4, 8) | 4 | (0, 1, 2) | (1, 2/3, 1/2) | 32/3 | 16/3 | 19 | 41/3 = 13.6667 | 41/3 | True |
| (8, 4, 4) | 8 | (0, 1, 3/2) | (1, 2/3, 4/7) | 272/21 | 64/21 | 19 | 335/21 = 15.9524 | 335/21 | True |

All rows match the hand calculation exactly: True; largest float error 0.0e+00.

With alpha = 0 every order scores 19, because a sum ignores order. With alpha = 1/2 the score
falls as the large FFN moves later: its own weight drops (more capacity precedes it), and the
small FFNs that move forward gain less than it loses. C_3 = 3 for both (8,4,4) and (4,8,4),
because C_l depends only on the multiset of earlier capacities; those two orders differ only
through w_2. Doubling d halves every C_l, so the same capacities are discounted less.
