"""Tests that the report numbers come only from current results."""

import json
import tempfile
import unittest
from pathlib import Path

from cansc import report_tex


class ReportNumbersTest(unittest.TestCase):
    def test_blocked_run_ignores_stale_dev_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            results, out = Path(tmp) / "results", Path(tmp) / "generated"
            results.mkdir()
            (results / "status.json").write_text(json.dumps({"dev_evaluation": "blocked", "missing": ["x"]}))
            # A stale file from some earlier run must not reach the report.
            (results / "dev_results.json").write_text("{not even valid json")
            report_tex.write_generated(results, out)
            macros = (out / "macros.tex").read_text()
            self.assertIn(r"\devresultsfalse", macros)
            self.assertIn(r"\newcommand{\RhoCourse}{\Pending}", macros)
            self.assertIn(r"\Pending[requires the course archive]", (out / "main_rows.tex").read_text())

    def test_number_formatting(self):
        self.assertEqual(report_tex._sig4(24940.5), "24{,}940")
        self.assertEqual(report_tex._sig4(309.03), "309.0")
        self.assertEqual(report_tex._sig4(256.0), "256")
        self.assertEqual(report_tex._tex("arch_01&x"), r"arch\_01\&x")


if __name__ == "__main__":
    unittest.main()
