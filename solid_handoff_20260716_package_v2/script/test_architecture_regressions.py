#!/usr/bin/env python3
"""Deterministic tests for the architecture vulnerability registry."""
import unittest

from script.check_architecture_regressions import evaluate, load_registry


class ArchitectureRegressionTests(unittest.TestCase):
    def setUp(self):
        self.registry, _ = load_registry()

    def test_online_locked_cases(self):
        answers = {
            "res_a_004": "ABC",
            "res_a_006": "A",
            "res_a_011": "ABC",
            "fc_a_014": "ABC",
        }
        rows, blocking = evaluate(self.registry, answers, {"P0"})
        states = {row["qid"]: row["state"] for row in rows}
        self.assertEqual(states["res_a_004"], "PASS")
        self.assertEqual(states["res_a_006"], "PASS")
        self.assertEqual(states["res_a_011"], "PASS")
        self.assertEqual(states["fc_a_014"], "PASS")
        self.assertEqual(blocking, 0)

    def test_known_wrong_blocks(self):
        answers = {
            "res_a_004": "AB",
            "res_a_006": "B",
            "res_a_011": "BC",
            "fc_a_014": "AB",
        }
        rows, blocking = evaluate(self.registry, answers, {"P0"})
        self.assertEqual(blocking, 4)
        self.assertTrue(all(row["state"] in {"FAIL", "KNOWN-WRONG"}
                            for row in rows))

    def test_newly_resolved_fc4_ins14_constraint(self):
        answers = {
            "fc_a_005": "ABD",
            "fc_a_015": "D",
            "fc_a_004": "ABCD",
            "ins_a_014": "AB",
            "reg_a_011": "ACD",
            "res_a_002": "ABC",
            "fc_a_018": "A",
        }
        rows, blocking = evaluate(self.registry, answers, {"P1"})
        states = {row["qid"]: row["state"] for row in rows}
        self.assertEqual(states["ins_a_014"], "PASS")
        self.assertEqual(states["fc_a_004"], "CANDIDATE")
        self.assertEqual(blocking, 0)

        answers["ins_a_014"] = "A"
        answers["fc_a_004"] = "ABC"
        rows, blocking = evaluate(self.registry, answers, {"P1"})
        states = {row["qid"]: row["state"] for row in rows}
        self.assertEqual(states["ins_a_014"], "FAIL")
        self.assertEqual(states["fc_a_004"], "KNOWN-WRONG")
        self.assertEqual(blocking, 2)


if __name__ == "__main__":
    unittest.main()
