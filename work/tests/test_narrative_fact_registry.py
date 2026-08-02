"""Regression tests for source-backed narrative calculation facts."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
sys.path.insert(0, str(WORK))

from agent.b_schema import load_questions, load_schema, effective_kinds  # noqa: E402
from agent.deterministic_calc import try_solve  # noqa: E402
from agent.narrative_fact_registry import (  # noqa: E402
    calculation_facts,
    extract_facts,
    facts_block,
)


QUESTIONS = load_questions(ROOT / "upload_b" / "question_b")
SCHEMA = load_schema(ROOT / "upload_b" / "submit.csv")
BY_ID = {q["qid"]: q for q in QUESTIONS}


class RealSourceExtractionTests(unittest.TestCase):
    def test_supported_shapes_and_exact_values(self) -> None:
        expected = {
            "fc_b_001": ["5.55", "5.36", "7.68", "7.35"],
            "fc_b_005": ["1468.47", "740.58"],
            "fc_b_014": ["116570.42", "140326.08", "175365.27"],
            "reg_b_003": ["60", "1", "30"],
            "reg_b_004": ["90"],
            "reg_b_007": ["60", "1", "30"],
            "reg_b_014": ["100", "0.5", "0.5"],
            "reg_b_016": ["5000"],
            "reg_b_018": ["30"],
            "res_b_005": ["1300.4", "596.0", "45.8"],
            "res_b_007": ["59.60"],
        }
        for qid, values in expected.items():
            with self.subTest(qid=qid):
                facts = extract_facts(BY_ID[qid])
                self.assertEqual([str(f.value) for f in facts], values)

    def test_qid_is_irrelevant(self) -> None:
        base = {k: v for k, v in BY_ID["res_b_007"].items() if k != "qid"}
        left = extract_facts(dict(base, qid="arbitrary_one"))
        right = extract_facts(dict(base, qid="unrelated_identifier"))
        self.assertEqual(left, right)

    def test_every_verbatim_is_in_declared_document_and_page(self) -> None:
        for qid in (
            "fc_b_001",
            "fc_b_005",
            "fc_b_014",
            "reg_b_003",
            "reg_b_004",
            "reg_b_014",
            "reg_b_016",
            "reg_b_018",
            "res_b_005",
            "res_b_007",
        ):
            domain = BY_ID[qid]["domain"]
            for fact in extract_facts(BY_ID[qid]):
                path = WORK / "processed_data" / domain / f"{fact.doc_id}.txt"
                text = path.read_text(encoding="utf-8")
                self.assertIn(fact.verbatim, text, f"{qid}: {fact.source_label}")
                if fact.page is not None:
                    marker = f"[P{fact.page}]"
                    self.assertIn(marker, text)
                    start = text.index(marker) + len(marker)
                    next_page = text.find("\n[P", start)
                    page_text = text[start : next_page if next_page >= 0 else None]
                    self.assertIn(fact.verbatim, page_text, fact.source_label)

    def test_facts_block_keeps_source_and_verbatim(self) -> None:
        facts = extract_facts(BY_ID["res_b_007"])
        block = facts_block(facts)
        self.assertIn("metric=new_orders", block)
        self.assertIn("source=pack2_text09 P30", block)
        self.assertIn("2025 年全年，公司新签订单金额59.60 亿元", block)


class CalculatorAdapterTests(unittest.TestCase):
    def _solve(self, qid: str):
        q = BY_ID[qid]
        facts = extract_facts(q)
        kinds = effective_kinds(q, SCHEMA[qid])
        return try_solve(
            q["question"],
            "",
            kinds,
            facts=calculation_facts(q["question"], facts),
        )

    def test_mean_appreciation_score_and_research_shapes_solve(self) -> None:
        expected = {
            "fc_b_001": "5.55%；5.36%；7.68%；7.35%",
            "fc_b_005": "1468.47%；740.58%",
            "fc_b_014": "14.41",
            "reg_b_014": "99.25",
            "reg_b_016": "2.00",
            "res_b_005": "22.27%",
            "res_b_007": "61.98",
        }
        for qid, answer in expected.items():
            with self.subTest(qid=qid):
                result = self._solve(qid)
                self.assertIsNotNone(result)
                self.assertEqual(result.raw_answer, answer)

    def test_mining_series_is_bound_to_business_row_not_issuer_total(self) -> None:
        q = BY_ID["fc_b_001"]
        facts = extract_facts(q)
        self.assertEqual({f.scope for f in facts}, {"矿业板块主营业务"})
        result = try_solve(
            q["question"],
            "",
            ["percent"] * 4,
            facts=calculation_facts(q["question"], facts),
        )
        self.assertIsNotNone(result)
        self.assertEqual(result.raw_answer, "5.55%；5.36%；7.68%；7.35%")

    def test_threshold_relation_is_inclusive(self) -> None:
        facts = extract_facts(BY_ID["reg_b_016"])
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].metric, "trigger_threshold")
        self.assertEqual(facts[0].relation, "greater_or_equal")


class FailClosedTests(unittest.TestCase):
    def test_conflicting_research_sources_return_no_facts(self) -> None:
        question = {
            "domain": "research",
            "question": (
                "一家IP设计公司2025年新签订单中AI相关订单占比超七成。"
                "若2026年新签订单增速放缓至30%，AI订单占比进一步提升至80%，"
                "则2026年AI新签订单约为多少亿元？"
            ),
            "options": {},
        }
        template = (
            "[P1]\n2025 年全年，公司新签订单金额{value} 亿元"
            "（YoY+103.41%），其中AI 算力相关订单占比超73%"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            folder = root / "research"
            folder.mkdir(parents=True)
            (folder / "one.txt").write_text(template.format(value="59.60"), encoding="utf-8")
            (folder / "two.txt").write_text(template.format(value="60.00"), encoding="utf-8")
            self.assertEqual(extract_facts(question, processed_dir=root), ())

    def test_selected_doc_ids_do_not_escape_to_other_documents(self) -> None:
        q = dict(BY_ID["res_b_007"])
        q["doc_ids"] = ["pack2_text01"]
        self.assertEqual(extract_facts(q), ())

    def test_unknown_shape_returns_empty(self) -> None:
        self.assertEqual(
            extract_facts({"domain": "research", "question": "请概括行业观点。"}),
            (),
        )


if __name__ == "__main__":
    unittest.main()
