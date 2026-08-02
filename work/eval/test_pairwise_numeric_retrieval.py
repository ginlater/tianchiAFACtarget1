"""Lexical regression tests for two-entity numeric comparison evidence."""
import json
import unittest

from agent import retrieval


class PairwiseNumericRetrievalTests(unittest.TestCase):
    def test_classifier_is_question_driven_and_fails_closed(self):
        self.assertEqual(
            retrieval.pairwise_numeric_entities(
                "2026年3月湖北省光模块出口环比增速显著高于江苏省"),
            ("湖北省", "江苏省"),
        )
        self.assertEqual(
            retrieval.pairwise_numeric_entities("前者表现显著高于后者"),
            (),
        )
        self.assertEqual(
            retrieval.pairwise_numeric_entities(
                "甲方科技公司利润增速高于乙方制造公司"),
            ("甲方科技公司", "乙方制造公司"),
        )

    def test_exact_pair_row_is_ranked_and_projected(self):
        generic = {
            "id": "d#c0", "doc_id": "d", "page": 1,
            "text": "湖北省光模块出口环比增速。" * 30,
        }
        exact = {
            "id": "d#c1", "doc_id": "d", "page": 2,
            "text": ("无关图表说明。" * 80 +
                     "2026年3月，江苏省出口额13.27亿元，环比-1.2%；"
                     "湖北省出口额4.68亿元，环比+44.2%。" +
                     "无关图表说明。" * 80),
        }
        query = "2026年3月湖北省光模块出口环比增速显著高于江苏省"
        hit, _score = retrieval.BM25([generic, exact]).search(query, k=1)[0]
        self.assertEqual(hit["id"], "d#c1")
        self.assertLessEqual(len(hit["text"]), 422)
        self.assertIn("江苏省…环比-1.2%", hit["text"])
        self.assertIn("湖北省…环比+44.2%", hit["text"])

    def test_real_research_clause_is_bound_to_page_12(self):
        with open("upload_b/question_b/research_b_question.jsonl",
                  encoding="utf-8-sig") as stream:
            question = next(
                json.loads(line) for line in stream
                if json.loads(line)["qid"] == "res_b_015"
            )
        query = question["options"]["D"]
        hit, _score = retrieval.doc_index("pack2_text11").search(query, k=1)[0]
        self.assertEqual(hit["id"], "pack2_text11#c20")
        self.assertEqual(hit["page"], 12)
        self.assertIn("江苏省…环比-1.2%", hit["text"])
        self.assertIn("湖北省…环比+44.2%", hit["text"])


if __name__ == "__main__":
    unittest.main()
