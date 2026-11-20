"""Unit tests of the rank statistics against hand-computed values."""

import math
import unittest

from cansc.stats import average_ranks, kendall_tau_b, rank_change, spearman


class RankStatisticsTest(unittest.TestCase):
    def test_average_ranks_with_ties(self):
        self.assertEqual(average_ranks([10, 20, 20, 30]), [1.0, 2.5, 2.5, 4.0])
        self.assertEqual(average_ranks([3, 3, 3]), [2.0, 2.0, 2.0])

    def test_spearman_hand_value(self):
        # ranks y = [1, 2, 3.5, 5, 3.5]; sum dxdy = 8, sum dx^2 = 10, sum dy^2 = 9.5
        self.assertAlmostEqual(spearman([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]), 8 / math.sqrt(95), places=15)

    def test_spearman_extremes(self):
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3], [30, 20, 10]), -1.0)
        self.assertTrue(math.isnan(spearman([1, 2, 3], [5, 5, 5])))

    def test_kendall_tau_b_hand_value(self):
        # P = 8, Q = 1, one pair tied in y only: (8 - 1) / sqrt(9 * 10)
        self.assertAlmostEqual(kendall_tau_b([1, 2, 3, 4, 5], [5, 6, 7, 8, 7]), 7 / math.sqrt(90), places=15)

    def test_negative_perplexity_direction(self):
        # A score that is high exactly when perplexity is low has rho = +1 against -ppl.
        ppl = [30.0, 20.0, 10.0]
        score = [1.0, 2.0, 3.0]
        self.assertAlmostEqual(spearman(score, [-p for p in ppl]), 1.0)

    def test_rank_change(self):
        same = rank_change([1, 2, 3, 4], [10, 20, 30, 40])
        self.assertEqual((same["items_rank_changed"], same["reversed_pairs"]), (0, 0))
        swap = rank_change([1, 2, 3, 4], [1, 3, 2, 4])
        self.assertEqual((swap["items_rank_changed"], swap["reversed_pairs"], swap["max_abs_rank_shift"]),
                         (2, 1, 1.0))

    def test_rejects_nan(self):
        with self.assertRaises(ValueError):
            spearman([1.0, float("nan")], [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
