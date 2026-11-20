"""Unit tests of the size-control pair rule and verdicts."""

import unittest
from fractions import Fraction

from cansc.pairs import compare, judge, param_ratio, qualifying_pairs, transitions
from cansc.records import ArchRecord


def rec(arch_id, width, params, ppl, split="development"):
    return ArchRecord(arch_id, split, width, 2, params, ppl)


class PairRuleTest(unittest.TestCase):
    def test_ratio_boundary_is_inclusive_and_exact(self):
        self.assertEqual(param_ratio(9_000_000, 10_000_000), Fraction(9, 10))
        recs = [rec("a", 256, 9_000_000, 10.0), rec("b", 256, 10_000_000, 11.0),
                rec("c", 256, 8_999_999, 12.0)]
        ids = [(a.arch_id, b.arch_id) for a, b, _ in qualifying_pairs(recs)]
        # a-b at exactly 0.90 qualifies; c-b is just below 0.90; a-c is ~0.99999989.
        self.assertEqual(ids, [("a", "b"), ("a", "c")])

    def test_float_params_use_written_decimal(self):
        self.assertEqual(param_ratio(4.5, 5.0), Fraction(9, 10))

    def test_width_must_match(self):
        recs = [rec("a", 256, 100, 10.0), rec("b", 512, 100, 11.0)]
        self.assertEqual(qualifying_pairs(recs), [])

    def test_mixed_splits_rejected(self):
        with self.assertRaises(ValueError):
            qualifying_pairs([rec("a", 256, 100, 10.0), rec("b", 256, 100, 11.0, split="final")])


class VerdictTest(unittest.TestCase):
    def test_judge(self):
        self.assertEqual(judge(2.0, 1.0, 10.0, 12.0), "correct")   # a better, a scored higher
        self.assertEqual(judge(1.0, 2.0, 10.0, 12.0), "wrong")
        self.assertEqual(judge(1.0, 1.0 + 1e-15, 10.0, 12.0), "score_tie")
        self.assertEqual(judge(2.0, 1.0, 11.5, 11.5), "truth_tie")  # truth tie wins over score order

    def test_compare_and_transitions(self):
        recs = [rec("a", 256, 100, 10.0), rec("b", 256, 95, 12.0), rec("c", 256, 92, 12.0)]
        pairs = qualifying_pairs(recs)
        before = compare(pairs, {"a": 3.0, "b": 2.0, "c": 1.0})
        after = compare(pairs, {"a": 1.0, "b": 2.0, "c": 3.0})
        self.assertEqual(before["counts"], {"correct": 2, "wrong": 0, "score_tie": 0, "truth_tie": 1})
        self.assertEqual(after["counts"], {"correct": 0, "wrong": 2, "score_tie": 0, "truth_tie": 1})
        self.assertEqual(transitions(before["verdicts"], after["verdicts"]),
                         {"correct->wrong": 2, "truth_tie->truth_tie": 1})


if __name__ == "__main__":
    unittest.main()
