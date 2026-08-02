"""Tests for the conservative, zero-token document router."""
from copy import deepcopy
import pathlib
import sys
import unittest
from unittest.mock import patch


WORK = pathlib.Path(__file__).resolve().parents[1]
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

from agent.b_schema import load_questions  # noqa: E402
from agent.docsel_fast import (  # noqa: E402
    _strict_bm25_contract_pick,
    select_docs_fast,
)


class DocselFastTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        qdir = WORK.parent / "upload_b" / "question_b"
        cls.questions = {q["qid"]: q for q in load_questions(qdir)}

    def test_qid_and_answer_fields_do_not_affect_decision(self):
        original = deepcopy(self.questions["fin_b_004"])
        mutated = deepcopy(original)
        mutated["qid"] = "deliberately_unrelated_identifier"
        mutated["answer"] = "THIS MUST NEVER BE READ"
        mutated["gold"] = ["also ignored"]
        self.assertEqual(select_docs_fast(original), select_docs_fast(mutated))

    def test_financial_report_company_years_are_fully_covered(self):
        picked, diag = select_docs_fast(self.questions["fin_b_004"])
        self.assertEqual(set(picked), {
            "annual_byd_2024_report", "annual_byd_2025_report",
            "annual_catl_2024_report", "annual_catl_2025_report",
            "annual_midea_2024_report", "annual_midea_2025_report",
        })
        self.assertEqual(diag["method"], "explicit_company_aliases")

    def test_insurance_shared_aliases_respect_product_brands(self):
        picked, _ = select_docs_fast(self.questions["ins_b_008"])
        # Ping An/ZhongAn special vehicle and food-safety products are all
        # explicitly named.  No unrelated insurance document may be added.
        self.assertEqual(set(picked), {"9", "10", "13", "14"})

    def test_contract_comparison_has_every_named_issuer(self):
        picked, _ = select_docs_fast(self.questions["fc_b_006"])
        self.assertEqual(set(picked), {"text04", "text05", "text11"})
        self.assertGreaterEqual(len(picked), 2)

    def test_contract_calc_can_use_unique_glossary_company_identity(self):
        # "长安银行" is explicitly defined as the target company in text12's
        # glossary, despite not being that document's cover-page issuer.
        picked, diag = select_docs_fast(self.questions["fc_b_020"])
        self.assertEqual(picked, ["text12"])
        self.assertEqual(diag["method"], "explicit_company_aliases")
        self.assertIn("长安银行", diag["aliases"])

    def test_narrative_fact_bundle_routes_contract_calculations(self):
        expected = {
            "fc_b_001": "text01",
            "fc_b_005": "text08",
            "fc_b_014": "text14",
        }
        for qid, doc_id in expected.items():
            with self.subTest(qid=qid):
                picked, diag = select_docs_fast(self.questions[qid])
                self.assertEqual(picked, [doc_id])
                self.assertEqual(
                    diag["method"],
                    "unique_source_bound_narrative_facts")

    def test_narrative_fact_bundle_prevents_wrong_regulation_card(self):
        picked, diag = select_docs_fast(self.questions["reg_b_003"])
        self.assertEqual(picked, ["csrc_0005_att1"])
        self.assertEqual(diag["method"],
                         "unique_source_bound_narrative_facts")
        self.assertIn("progress_report_offset", diag["metrics"])

    def test_ambiguous_contract_calc_fails_closed(self):
        vague = {
            "qid": "must_not_influence_routing",
            "domain": "financial_contracts",
            "answer_format": "calc",
            "question": "根据相关募集说明书计算上述指标，并保留两位小数。",
            "options": {},
        }
        self.assertIsNone(select_docs_fast(vague))

    def test_contract_bm25_fallback_has_strict_absolute_and_relative_floor(self):
        class FakeIndex:
            def __init__(self, scores):
                self.scores = scores

            def search(self, _query, k=3):
                return [({"doc_id": doc_id}, score)
                        for doc_id, score in self.scores[:k]]

        q = {"domain": "financial_contracts", "question": "抽取并计算指标",
             "options": {}}
        accepted = FakeIndex([
            ("text12", 165.0), ("text03", 100.0), ("text02", 90.0),
        ])
        with patch("agent.docsel_fast._bm25", return_value=accepted):
            picked, diag = _strict_bm25_contract_pick(q)
        self.assertEqual(picked, ["text12"])
        self.assertEqual(diag["bm25_top_score"], 165.0)
        self.assertEqual(diag["bm25_margin"], 1.65)

        weak_absolute = FakeIndex([
            ("text12", 149.99), ("text03", 50.0), ("text02", 40.0),
        ])
        weak_margin = FakeIndex([
            ("text12", 164.99), ("text03", 100.0), ("text02", 90.0),
        ])
        for index in (weak_absolute, weak_margin):
            with self.subTest(scores=index.scores), \
                    patch("agent.docsel_fast._bm25", return_value=index):
                self.assertIsNone(_strict_bm25_contract_pick(q))

    def test_regulatory_ambiguity_falls_back(self):
        self.assertIsNone(select_docs_fast(self.questions["reg_b_001"]))

    def test_unique_source_bound_regulation_can_skip_docsel_qwen(self):
        picked, diag = select_docs_fast(self.questions["reg_b_004"])
        self.assertEqual(len(picked), 1)
        self.assertIn("银行卡清算机构管理办法", picked[0])
        self.assertEqual(diag["method"],
                         "unique_source_bound_narrative_facts")

    def test_thematic_research_stays_on_qwen_path(self):
        self.assertIsNone(select_docs_fast(self.questions["res_b_001"]))

    def test_single_source_research_calculation_can_use_strict_bm25(self):
        picked, diag = select_docs_fast(self.questions["res_b_012"])
        self.assertEqual(picked, ["pack2_text05"])
        self.assertEqual(diag["method"], "strict_single_source_calc_bm25")


if __name__ == "__main__":
    unittest.main()
