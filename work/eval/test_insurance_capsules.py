"""Offline regression tests for deterministic insurance typed memory."""
import json
import pathlib
import re
import sys
import unittest

WORK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from agent.insurance_capsules import (  # noqa: E402
    SCHEMA_VERSION, insurance_capsule_block, insurance_option_evidence_block,
    insurance_lexical_coverage_block, insurance_question_route,
    insurance_question_theme, load_capsules, option_document_map, select_capsules,
)


ARTIFACT = WORK / "processed_data" / "insurance_capsules.json"


class InsuranceCapsuleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = load_capsules(ARTIFACT, force=True)

    def test_schema_and_all_documents(self):
        self.assertEqual(self.data["schema_version"], SCHEMA_VERSION)
        self.assertEqual(set(self.data["documents"]), {str(i) for i in range(1, 17)})
        self.assertNotIn('"qid"', json.dumps(self.data, ensure_ascii=False))
        self.assertNotIn('"answer"', json.dumps(self.data, ensure_ascii=False))

    def test_every_capsule_is_typed_and_short(self):
        required = {"id", "doc_id", "page", "clause", "clause_title",
                    "topic", "numbers", "verbatim", "sources"}
        for doc in self.data["documents"].values():
            self.assertTrue(doc["capsules"])
            for card in doc["capsules"]:
                self.assertTrue(required <= set(card))
                self.assertLessEqual(len(card["verbatim"]), 520)
                self.assertTrue(card["verbatim"].strip())

    def test_raw_verbatim_is_traceable_to_its_page(self):
        page_re = re.compile(r"^\[P(\d+)\]\s*$")
        for doc_id, doc in self.data["documents"].items():
            pages, page, lines = {}, 0, []
            for raw in (WORK / "processed_data" / "insurance" /
                        f"{doc_id}.txt").read_text(encoding="utf-8").splitlines():
                match = page_re.match(raw.strip())
                if match:
                    if page:
                        pages[page] = "".join(lines)
                    page, lines = int(match.group(1)), []
                else:
                    lines.append(raw)
            if page:
                pages[page] = "".join(lines)
            norm_pages = {p: re.sub(r"\s+", "", t).replace("％", "%")
                          for p, t in pages.items()}
            for card in doc["capsules"]:
                if "raw_text" not in card["sources"]:
                    continue
                verbatim = re.sub(r"\s+", "", card["verbatim"]).replace("％", "%")
                self.assertIn(verbatim, norm_pages[card["page"]], card["id"])

    def test_known_death_ratio_is_retrieved(self):
        q = {
            "qid": "ignored_a",
            "domain": "insurance",
            "doc_ids": ["2"],
            "question": "身故给付比例中，18至41周岁对应多少？",
            "options": {"A": "160%", "B": "140%"},
        }
        block = insurance_capsule_block(q, path=ARTIFACT)
        self.assertIn("160%", block)
        self.assertIn("P3", block)
        self.assertIn("身故", block)

    def test_qid_is_irrelevant(self):
        base = {
            "domain": "insurance",
            "doc_ids": ["1", "2"],
            "question": "比较两款产品犹豫期和退费规则",
            "options": {},
        }
        left = select_capsules(dict(base, qid="one"), path=ARTIFACT)
        right = select_capsules(dict(base, qid="completely_different"), path=ARTIFACT)
        self.assertEqual([x["id"] for x in left], [x["id"] for x in right])

    def test_cross_document_balance_and_budget(self):
        q = {
            "domain": "insurance",
            "doc_ids": ["11", "12", "14"],
            "question": "哪些条款明确规定施救费用、免赔额和赔偿限额？",
            "options": {},
        }
        rows = select_capsules(q, path=ARTIFACT, per_doc=1, max_cards=12,
                               char_budget=4000)
        self.assertEqual({"11", "12", "14"}, {x["doc_id"] for x in rows})
        block = insurance_capsule_block(q, path=ARTIFACT, per_doc=1,
                                        max_cards=12, char_budget=4000)
        self.assertLessEqual(len(block), 4000)
        self.assertRegex(block, "施救|免赔|赔偿限额")

    def test_option_evidence_binds_products_without_deciding_answers(self):
        q = {
            "qid": "must_not_matter",
            "domain": "insurance",
            "doc_ids": ["2", "4", "6", "8"],
            "question": "年龄错误少付保费后如何处理？",
            "options": {
                "A": "国寿增益宝按实付与应付比例给付",
                "B": "平安安佑福按实付与应付比例给付",
                "C": "众安营运交通工具保险按比例给付",
                "D": "太保团体百万医疗按实付与应付比例给付",
            },
        }
        self.assertEqual(option_document_map(q, path=ARTIFACT),
                         {"A": "2", "B": "4", "C": "8", "D": "6"})
        block = insurance_option_evidence_block([q], path=ARTIFACT,
                                                char_budget=4000)
        self.assertLessEqual(len(block), 4000)
        self.assertIn("题1选项B｜doc=4", block)
        self.assertIn("题1选项D｜doc=6", block)
        self.assertIn("实付保险费和应付保险费的比例", block)
        self.assertNotIn("入选", block)

    def test_single_policy_options_share_the_named_document(self):
        q = {
            "domain": "insurance", "doc_ids": ["16"],
            "question": "关于减额交清，下列说法正确的是？",
            "options": {"A": "可申请", "B": "仍须缴费", "C": "保额减少"},
        }
        self.assertEqual(option_document_map(q, path=ARTIFACT),
                         {"A": "16", "B": "16", "C": "16"})

    def test_theme_is_qid_invariant_and_shape_based(self):
        base = {
            "domain": "insurance", "doc_ids": ["2", "4"],
            "question": "关于本合同成立后2年内自杀的责任免除",
            "options": {"A": "产品甲", "B": "产品乙"},
        }
        self.assertEqual(insurance_question_theme(dict(base, qid="x")),
                         "legal_procedure")
        self.assertEqual(insurance_question_theme(dict(base, qid="y")),
                         "legal_procedure")

    def test_dedicated_legal_routes_are_semantic_and_qid_invariant(self):
        minor = {
            "domain": "insurance", "doc_ids": ["2", "3", "4", "16"],
            "question": "关于未成年人身故保险金限制，哪些产品明确提及该限制？",
            "options": {
                "A": "国寿增益宝", "B": "平安安佑福重疾险",
                "C": "众安个人急性白血病复发医疗保险",
                "D": "平安富鸿金生养老年金保险",
            },
        }
        suicide = {
            "domain": "insurance", "doc_ids": ["2", "4", "8", "16"],
            "question": "哪些产品明确包含本合同成立起2年内自杀，"
                        "但无民事行为能力人除外的规则？",
            "options": {
                "A": "国寿增益宝", "B": "平安安佑福重疾险",
                "C": "众安营运交通工具团体意外伤害保险",
                "D": "平安富鸿金生养老年金保险",
            },
        }
        for qid in ("opaque", "completely_other"):
            self.assertEqual(
                insurance_question_route(dict(minor, qid=qid), path=ARTIFACT),
                "minor_death_limit_exhaustive")
            self.assertEqual(
                insurance_question_route(dict(suicide, qid=qid), path=ARTIFACT),
                "suicide_exception")

        audit = insurance_lexical_coverage_block(minor, path=ARTIFACT)
        self.assertIn("扫描", audit)
        self.assertIn("选项A doc=2", audit)
        self.assertIn("选项B doc=4", audit)
        self.assertIn("选项C doc=3", audit)
        self.assertRegex(audit, r"选项A.*命中[1-9]\d*条")
        self.assertRegex(audit, r"选项B.*命中[1-9]\d*条")
        self.assertRegex(audit, r"选项C.*命中0条")

    def test_option_excerpt_keeps_decisive_literal_terms(self):
        quake = {
            "domain": "insurance", "doc_ids": ["11", "12"],
            "question": "哪些责任免除明确列明地震？",
            "options": {"A": "平安家庭财产保险",
                        "B": "众安家庭财产综合保险"},
        }
        nuclear = {
            "domain": "insurance", "doc_ids": ["4", "5"],
            "question": "哪些责任免除明确列明核爆炸、核辐射或核污染？",
            "options": {"A": "平安安佑福", "B": "平安e生保"},
        }
        quake_block = insurance_option_evidence_block(
            [quake], path=ARTIFACT, char_budget=900)
        nuclear_block = insurance_option_evidence_block(
            [nuclear], path=ARTIFACT, char_budget=900)
        self.assertLessEqual(len(quake_block), 900)
        self.assertLessEqual(len(nuclear_block), 900)
        self.assertIn("地震", quake_block)
        self.assertIn("核爆炸", nuclear_block)


if __name__ == "__main__":
    unittest.main()
