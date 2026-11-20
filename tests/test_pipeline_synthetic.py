"""End-to-end test of the development pipeline on a SYNTHETIC archive.

The synthetic architectures only exercise the plumbing (scoring, checks,
selection, pair comparison). Their numbers are not results and never appear in
results/ or in the report.
"""

import random
import unittest

from cansc.context_nsc import ArchTerms, LayerTerms, context_aware_score, plain_sum
from cansc.evaluate import evaluate_development
from cansc.records import ArchRecord


def make_archive(n=30, seed=7):
    rng = random.Random(seed)
    records, terms = [], {}
    for i in range(n):
        width = rng.choice([128, 256])
        depth = rng.randint(1, 8)
        layers = tuple(LayerTerms(rng.uniform(1, 5), rng.uniform(0, 4 * width)) for _ in range(depth))
        t = ArchTerms(f"s{i:02d}", width, layers, other=rng.uniform(0, 3))
        params = 1_000_000 + 10_000 * rng.randint(0, 20)
        ppl = round(60 - plain_sum(t) / 200 + rng.uniform(-3, 3), 2)
        records.append(ArchRecord(t.arch_id, "development", width, depth, params, ppl))
        terms[t.arch_id] = t
    return records, terms


def toy_archive():
    """Four same-width architectures with a known correct -> wrong reversal (Q, R)."""
    spec = {"P": ((8, 4, 4), 100, 10.0), "Q": ((4, 4, 8), 98, 11.0),
            "R": ((8, 4, 3.5), 96, 12.0), "T": ((2, 2, 2), 95, 20.0)}
    records, terms = [], {}
    for arch_id, (ffn, params, ppl) in spec.items():
        terms[arch_id] = ArchTerms(arch_id, 4, tuple(LayerTerms(1.0, float(f)) for f in ffn))
        records.append(ArchRecord(arch_id, "development", 4, 3, params, ppl))
    return records, terms


class PipelineTest(unittest.TestCase):
    def test_random_archive(self):
        records, terms = make_archive()
        grid = [0.0, 0.01, 0.1, 1.0]
        res = evaluate_development(records, lambda r: plain_sum(terms[r.arch_id]),
                                   lambda r: terms[r.arch_id], grid, repeats=1)
        rows = res["rows"]
        self.assertEqual(len(rows), 1 + len(grid))
        self.assertTrue(res["checks"]["alpha0_vs_course"]["passed"])
        self.assertTrue(res["checks"]["first_layer_weight"]["all_exactly_one"])
        self.assertEqual(rows[1]["spearman"], rows[0]["spearman"])  # alpha = 0 is the course score
        best = max(r["spearman"] for r in rows[1:])
        self.assertEqual(res["selected_alpha"], min(r["alpha"] for r in rows[1:] if r["spearman"] == best))
        sc = res["size_control"]
        self.assertEqual(sum(sc["course"].values()), sc["n_pairs"])
        self.assertEqual(sum(sc["selected"].values()), sc["n_pairs"])
        self.assertEqual(sum(sc["transitions"].values()), sc["n_pairs"])
        single = res["checks"]["single_layer_archived"]
        if single["n_single_layer_architectures"]:
            self.assertEqual(single["max_abs_err_any_alpha"], 0.0)

    def test_identity_failure_is_reported(self):
        records, terms = make_archive(n=10)
        broken = lambda r: plain_sum(terms[r.arch_id]) + (1e-3 if r.arch_id == "s03" else 0.0)
        res = evaluate_development(records, broken, lambda r: terms[r.arch_id], [0.0], repeats=1)
        self.assertFalse(res["checks"]["alpha0_vs_course"]["passed"])
        self.assertEqual(res["checks"]["alpha0_vs_course"]["n_fail"], 1)

    def test_known_reversal(self):
        records, terms = toy_archive()
        res = evaluate_development(records, lambda r: plain_sum(terms[r.arch_id]),
                                   lambda r: terms[r.arch_id], [0.5], repeats=1)
        sc = res["size_control"]
        self.assertEqual(sc["n_pairs"], 6)
        self.assertEqual(sc["course"], {"correct": 5, "wrong": 0, "score_tie": 1, "truth_tie": 0})
        self.assertEqual(sc["selected"], {"correct": 5, "wrong": 1, "score_tie": 0, "truth_tie": 0})
        self.assertEqual(sc["reversal_example"]["pair"], ["Q", "R"])
        q = sc["reversal_example"]["architectures"][0]
        self.assertAlmostEqual(q["ca_nsc"], float(context_aware_score(terms["Q"], 0.5)))

    def test_wrong_split_rejected(self):
        records, terms = toy_archive()
        bad = [ArchRecord(r.arch_id, "final", r.width, r.depth, r.params, r.perplexity) for r in records]
        with self.assertRaises(ValueError):
            evaluate_development(bad, lambda r: 0.0, lambda r: terms[r.arch_id], [0.0], repeats=1)


if __name__ == "__main__":
    unittest.main()
