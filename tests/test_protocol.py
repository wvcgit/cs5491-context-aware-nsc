"""Unit tests of the pre-registered grid rule and selection rule."""

import json
import statistics
import tempfile
import unittest
from pathlib import Path

from cansc import protocol
from cansc.context_nsc import ArchTerms, LayerTerms


def uniform_terms(depth, width=8, ffn_per_layer=8):
    """Architecture whose C_l = (l - 1) * ffn_per_layer / width."""
    return ArchTerms(f"u{depth}", width, tuple(LayerTerms(1.0, float(ffn_per_layer)) for _ in range(depth)))


class GridRuleTest(unittest.TestCase):
    def test_round_sig1(self):
        self.assertEqual(protocol.round_sig1(0.0473), 0.05)
        self.assertEqual(protocol.round_sig1(1.37), 1.0)
        self.assertEqual(protocol.round_sig1(0.000271), 0.0003)
        self.assertEqual(protocol.round_sig1(0.15), 0.2)   # half up on the decimal value
        self.assertEqual(protocol.round_sig1(9.6), 10.0)
        self.assertEqual(protocol.round_sig1(37.0), 40.0)

    def test_grid_from_known_capacities(self):
        # Depths 3 and 5 with F = d: pooled C (l >= 2) = [1, 2] + [1, 2, 3, 4]; median 2.
        terms = [uniform_terms(3), uniform_terms(5)]
        self.assertEqual(sorted(protocol.pooled_capacities(terms)), [1, 1, 2, 2, 3, 4])
        frozen = protocol.instantiate_grid(terms)
        self.assertEqual(frozen["c_ref"], 2.0)
        # 0.1/2, 0.3/2, 1/2, 3/2 rounded to one significant figure
        self.assertEqual(frozen["grid"], [0.0, 0.05, 0.2, 0.5, 2.0])
        self.assertFalse(frozen["labels_used"])

    def test_single_layer_only_split_has_no_reference(self):
        with self.assertRaises(ValueError):
            protocol.instantiate_grid([uniform_terms(1)])

    def test_freeze_writes_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "protocol_frozen.json"
            first = protocol.freeze_or_load([uniform_terms(3), uniform_terms(5)], path)
            self.assertFalse(first["loaded_existing"])
            # Different terms must not change the frozen grid.
            second = protocol.freeze_or_load([uniform_terms(9, ffn_per_layer=80)], path)
            self.assertTrue(second["loaded_existing"])
            self.assertEqual(second["grid"], first["grid"])
            stored = json.loads(path.read_text())
            stored["alpha_multipliers"] = [1, 2]
            path.write_text(json.dumps(stored))
            with self.assertRaises(RuntimeError):
                protocol.freeze_or_load([uniform_terms(3)], path)


class SelectionRuleTest(unittest.TestCase):
    def test_max_rho(self):
        self.assertEqual(protocol.select_alpha([(0.0, 0.51), (0.1, 0.55), (1.0, 0.40)]), 0.1)

    def test_exact_tie_goes_to_smaller_alpha(self):
        self.assertEqual(protocol.select_alpha([(1.0, 0.6), (0.1, 0.6), (0.0, 0.5)]), 0.1)

    def test_alpha_zero_can_win(self):
        self.assertEqual(protocol.select_alpha([(0.0, 0.7), (0.1, 0.6999999999)]), 0.0)

    def test_nan_ignored(self):
        self.assertEqual(protocol.select_alpha([(0.0, float("nan")), (0.1, 0.2)]), 0.1)


class WeightSummaryTest(unittest.TestCase):
    def test_scope_excludes_first_layer(self):
        summary = protocol.weight_summary([uniform_terms(3)], alpha=1.0)
        # C = (0, 1, 2) -> w = (1, 1/2, 1/3); layers l >= 2 only.
        self.assertAlmostEqual(summary["w_min"], 1 / 3)
        self.assertAlmostEqual(summary["w_median"], statistics.median([1 / 2, 1 / 3]))
        self.assertAlmostEqual(summary["retained_median"], (1 + 1 / 2 + 1 / 3) / 3)


if __name__ == "__main__":
    unittest.main()
