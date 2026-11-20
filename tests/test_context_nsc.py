"""Unit tests of the CA-NSC reweighting (cansc.context_nsc and cansc.toy)."""

import unittest
from fractions import Fraction as Fr

from cansc import toy
from cansc.context_nsc import (ArchTerms, LayerTerms, context_aware_score, cumulative_capacity,
                               ffn_loss, ffn_weights, plain_sum)


def terms(ffn, width=4, attention=None, other=0):
    attention = attention or [1] * len(ffn)
    return ArchTerms("t", width, tuple(LayerTerms(a, f) for a, f in zip(attention, ffn)), other)


class CumulativeCapacityTest(unittest.TestCase):
    def test_values(self):
        self.assertEqual(cumulative_capacity([8, 4, 4], 4), [0, 2, 3])

    def test_uses_raw_capacity_whatever_alpha(self):
        # C_3 counts F_1 + F_2 = 12 raw units, not the discounted w_2 * F_2.
        self.assertEqual(cumulative_capacity([Fr(8), Fr(4), Fr(4)], Fr(4))[2], 3)

    def test_rejects_bad_inputs(self):
        with self.assertRaises(ValueError):
            cumulative_capacity([1, -0.5], 4)
        with self.assertRaises(ValueError):
            cumulative_capacity([1, 2], 0)
        with self.assertRaises(ValueError):
            cumulative_capacity([1, float("nan")], 4)


class WeightsAndScoreTest(unittest.TestCase):
    def test_weights_example(self):
        self.assertEqual(ffn_weights([8, 4, 4], 4, 0.5), [1.0, 0.5, 0.4])

    def test_negative_alpha_rejected(self):
        with self.assertRaises(ValueError):
            ffn_weights([1, 2], 4, -0.1)

    def test_first_weight_is_one(self):
        for alpha in (0, 0.3, 7.0, 1e6):
            self.assertEqual(ffn_weights([5, 1, 9], 3, alpha)[0], 1)

    def test_alpha_zero_equals_plain_sum_bitwise(self):
        t = terms([0.1, 0.7, 3.3, 2.9], width=3, attention=[0.2, 1.1, 0.4, 5.0], other=-0.3)
        self.assertEqual(context_aware_score(t, 0.0), plain_sum(t))

    def test_single_layer_unchanged(self):
        t = terms([123.0], width=64, attention=[4.5], other=2.0)
        for alpha in (0.0, 0.01, 1.0, 1e9):
            self.assertEqual(context_aware_score(t, alpha), plain_sum(t))

    def test_attention_and_other_untouched(self):
        t = terms([Fr(8), Fr(4), Fr(4)], width=Fr(4), attention=[10, 20, 30], other=5)
        # Only the FFN part changes: 5 + 60 + (8 + 4/2 + 4*2/5), exactly.
        self.assertEqual(context_aware_score(t, Fr(1, 2)), 5 + 60 + Fr(58, 5))

    def test_loss_identity(self):
        t = terms([Fr(8), Fr(4), Fr(4)])
        self.assertEqual(plain_sum(t) - ffn_loss(t, Fr(1, 2)), context_aware_score(t, Fr(1, 2)))

    def test_order_sensitivity(self):
        front = context_aware_score(terms([8, 4, 4]), 0.5)
        back = context_aware_score(terms([4, 4, 8]), 0.5)
        self.assertGreater(front, back)
        self.assertEqual(context_aware_score(terms([8, 4, 4]), 0.0), context_aware_score(terms([4, 4, 8]), 0.0))

    def test_two_layer_front_loading_is_preferred(self):
        # For F_a > F_b, order (a, b) beats (b, a) for every alpha > 0 (proof in the report).
        for alpha in (0.01, 0.5, 4.0):
            self.assertGreater(context_aware_score(terms([9.0, 2.0]), alpha),
                               context_aware_score(terms([2.0, 9.0]), alpha))


class ToyExampleTest(unittest.TestCase):
    def test_matches_hand_calculation(self):
        result = toy.run()
        self.assertTrue(result["all_exact_match"])
        self.assertLess(result["max_float_abs_err"], 1e-12)
        scores = [r["S_alpha"] for r in result["rows"]]
        self.assertEqual(scores, ["73/5", "209/15", "41/3", "335/21"])


if __name__ == "__main__":
    unittest.main()
