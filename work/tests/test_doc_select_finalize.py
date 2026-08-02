"""Regression tests for source-count fallbacks in Qwen document selection."""
from pathlib import Path
import unittest

from agent.b_schema import load_questions
from agent.doc_select import _finalize_picks, _requires_multiple_sources


ROOT = Path(__file__).resolve().parents[2]


class DocSelectFinalizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = {
            q["qid"]: q
            for q in load_questions(ROOT / "upload_b" / "question_b")
        }

    def test_single_source_calculation_is_not_padded_with_runner_up(self):
        q = self.questions["fc_b_020"]
        self.assertFalse(_requires_multiple_sources(q))
        self.assertEqual(
            _finalize_picks(q, ["text12"], ["text12", "text02"], 4),
            ["text12"],
        )

    def test_explicit_multi_document_question_keeps_second_source_floor(self):
        q = self.questions["fc_b_008"]
        self.assertTrue(_requires_multiple_sources(q))
        picked = _finalize_picks(
            q, ["text04"], ["text04", "text05", "text11"], 4)
        self.assertGreaterEqual(len(picked), 2)


if __name__ == "__main__":
    unittest.main()
