"""Integration tests against the course starter (skipped if starter/ is absent).

They use configurations only: no label file is read here.
"""

import copy
import unittest

from cansc import course_adapter
from cansc.context_nsc import context_aware_score, plain_sum

HAVE_STARTER = not course_adapter.missing_items()


@unittest.skipUnless(HAVE_STARTER, "course starter not available")
class StarterBindingTest(unittest.TestCase):
    def setUp(self):
        self.nsc = course_adapter.course_nsc()
        self.toy = {"d_model": 128, "n_layer": 2, "n_head": [4, 4], "d_inner": [256, 512]}

    def test_readme_reorder_pair(self):
        reverse = copy.deepcopy(self.toy)
        reverse["d_inner"].reverse()
        back = course_adapter.terms_from_config("back", self.toy)
        front = course_adapter.terms_from_config("front", reverse)
        # Starter README: original NSC 121.820169804 for both orders.
        self.assertAlmostEqual(self.nsc.original_nsc(self.toy), 121.820169804, places=9)
        self.assertEqual(context_aware_score(back, 0.0), self.nsc.original_nsc(self.toy))
        self.assertEqual(context_aware_score(front, 0.0), self.nsc.original_nsc(reverse))
        self.assertGreater(context_aware_score(front, 0.7), context_aware_score(back, 0.7))

    def test_decomposition_is_exact_on_development_configs(self):
        configs = {t.arch_id: t for t in course_adapter.unlabelled_terms("development")}
        raw = course_adapter._read("architectures.json")
        self.assertEqual(len(configs), 100)
        for arch_id, terms in configs.items():
            course_layers = self.nsc.layer_capacities(raw[arch_id])
            self.assertEqual([a + f for a, f in zip(terms.attention, terms.ffn)], course_layers)
            self.assertEqual(plain_sum(terms), self.nsc.original_nsc(raw[arch_id]))
            self.assertEqual(terms.width, raw[arch_id]["d_model"])

    def test_unlabelled_terms_read_no_label_file(self):
        seen = []
        original_read = course_adapter._read
        course_adapter._read = lambda name: (seen.append(name), original_read(name))[1]
        try:
            course_adapter.unlabelled_terms("development")
        finally:
            course_adapter._read = original_read
        self.assertEqual(sorted(set(seen)), ["architectures.json", "splits.json"])

    def test_final_split_is_refused(self):
        with self.assertRaises(PermissionError):
            course_adapter.load_split("final")
        with self.assertRaises(PermissionError):
            course_adapter.unlabelled_terms("final")

    def test_student_module(self):
        import student_ca_nsc
        raw = course_adapter._read("architectures.json")
        for arch_id in ("config_0_j0", "config_0_j1"):
            cfg = raw[arch_id]
            terms = course_adapter.terms_from_config(arch_id, cfg)
            self.assertEqual(student_ca_nsc.ablation(cfg), self.nsc.original_nsc(cfg))
            self.assertEqual(student_ca_nsc.score(cfg), context_aware_score(terms, student_ca_nsc.SELECTED_ALPHA))


if __name__ == "__main__":
    unittest.main()
