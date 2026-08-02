"""Tests for domain-semantic fast document-selection gates."""
import os
import unittest
from unittest import mock

from agent.run_b2 import _docsel_batch_key, _fast_docsel_enabled


class FastDocselGateTest(unittest.TestCase):
    def test_choices_remain_enabled_in_every_domain(self):
        with mock.patch.dict(os.environ, {
                "AFAC_FAST_DOCSEL": "1",
                "AFAC_FAST_DOCSEL_CALC": "1",
                "AFAC_FAST_DOCSEL_CALC_DOMAINS": "financial_contracts",
        }, clear=False):
            self.assertTrue(_fast_docsel_enabled({
                "domain": "regulatory", "answer_format": "multi"}))

    def test_calculation_gate_depends_on_domain_not_qid(self):
        env = {
            "AFAC_FAST_DOCSEL": "1",
            "AFAC_FAST_DOCSEL_CALC": "1",
            "AFAC_FAST_DOCSEL_CALC_DOMAINS": "financial_contracts",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            for qid in ("opaque_a", "opaque_b"):
                self.assertTrue(_fast_docsel_enabled({
                    "qid": qid, "domain": "financial_contracts",
                    "answer_format": "calc"}))
                self.assertFalse(_fast_docsel_enabled({
                    "qid": qid, "domain": "regulatory",
                    "answer_format": "calc"}))

    def test_research_docsel_separates_choice_and_calculation_workflows(self):
        self.assertEqual(_docsel_batch_key({
            "domain": "research", "answer_format": "multi"}),
            ("research", "choice"))
        self.assertEqual(_docsel_batch_key({
            "domain": "research", "answer_format": "calc"}),
            ("research", "calc"))
        self.assertEqual(_docsel_batch_key({
            "domain": "regulatory", "answer_format": "calc"}),
            ("regulatory", "all"))


if __name__ == "__main__":
    unittest.main()
