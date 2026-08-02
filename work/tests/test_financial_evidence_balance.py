"""Network-free tests for source-balanced annual-report evidence."""
from pathlib import Path
import unittest

from agent import answerer, batch


ROOT = Path(__file__).resolve().parents[2]


class FinancialEvidenceBalanceTest(unittest.TestCase):
    def test_latest_year_sources_are_interleaved_and_rotated(self):
        docs = [
            "annual_alpha_2024_report", "annual_alpha_2025_report",
            "annual_beta_2024_report", "annual_beta_2025_report",
            "annual_gamma_2024_report", "annual_gamma_2025_report",
        ]
        ids = [f"chunk-{i}" for i in range(len(docs))]
        chunks = {cid: {"id": cid, "doc_id": doc}
                  for cid, doc in zip(ids, docs)}
        starts = []
        for rotation in range(3):
            ordered = batch._financial_source_round_robin(
                ids, chunks, set(ids), docs, rotation=rotation)
            first_docs = [chunks[cid]["doc_id"] for cid in ordered[:3]]
            self.assertTrue(all("_2025_" in doc for doc in first_docs))
            self.assertEqual(len(set(first_docs)), 3)
            starts.append(first_docs[0])
        self.assertEqual(len(set(starts)), 3)

    def test_exact_metric_chunk_leads_inside_each_source(self):
        docs = ["annual_alpha_2025_report", "annual_beta_2025_report"]
        chunks = {
            "a-noise": {"doc_id": docs[0], "text": "行业销量保持增长"},
            "a-facts": {"doc_id": docs[0], "text":
                        "营业收入 423,701,834 经营活动产生的现金流量净额"},
            "b-noise": {"doc_id": docs[1], "text": "公司持续推进全球化"},
            "b-facts": {"doc_id": docs[1], "text":
                        "营业收入 456,451,731 基本每股收益 5.80"},
        }
        ids = list(chunks)
        question = {
            "question": "比较营业收入、经营活动产生的现金流量净额和基本每股收益",
            "options": {"A": "两家公司营业收入增长"},
        }
        ordered = batch._financial_source_round_robin(
            ids, chunks, set(ids), docs, question=question)
        self.assertEqual(ordered[:2], ["a-facts", "b-facts"])

    def test_ratio_snapshot_uses_latest_report_with_prior_column(self):
        q = {
            "domain": "financial_reports",
            "question": "比较三家公司2024年、2025年资产负债率、流动比率和速动比率",
            "options": {"A": "三家公司资产负债率下降"},
            "doc_ids": [
                "annual_byd_2024_report", "annual_byd_2025_report",
                "annual_catl_2024_report", "annual_catl_2025_report",
                "annual_midea_2024_report", "annual_midea_2025_report",
            ],
        }
        block = answerer._financial_ratio_snapshot_block(q)
        self.assertEqual(block.count("本期末/上年末/增减:"), 3)
        self.assertNotIn("annual_byd_2024_report", block)
        self.assertIn("资产负债率=70.74%/74.64%/-3.90%", block)
        self.assertIn("资产负债率=61.94%/65.24%/-3.30%", block)
        self.assertIn("资产负债率=61.17%/62.33%/-1.16%", block)
        self.assertIn("速动比率=94.60%/85.94%/8.66%", block)

    def test_ratio_snapshot_adds_older_report_for_three_year_span(self):
        q = {
            "domain": "financial_reports",
            "question": "比较比亚迪与宁德时代2023—2025年偿债指标",
            "options": {
                "A": "宁德时代资产负债率连续下降，利息保障倍数连续上升",
                "B": "比亚迪现金利息保障倍数由2023年降至2025年",
            },
            "doc_ids": [
                "annual_byd_2024_report", "annual_byd_2025_report",
                "annual_catl_2024_report", "annual_catl_2025_report",
            ],
        }
        block = answerer._financial_ratio_snapshot_block(q)
        self.assertIn("annual_catl_2024_report", block)
        self.assertIn("annual_catl_2025_report", block)
        self.assertIn("资产负债率=65.24%/69.34%/-4.10%", block)
        self.assertIn("利息保障倍数=16.16/15.35/5.28%", block)
        self.assertIn("利息保障倍数=31.95/16.16/97.71%", block)

    def test_summary_snapshot_binds_current_and_prior_columns(self):
        q = {
            "domain": "financial_reports",
            "question": "比较2024年、2025年营业收入、经营活动产生的现金流量净额、基本每股收益",
            "options": {"A": "两家公司营业收入均增长"},
            "doc_ids": [
                "annual_catl_2024_report", "annual_catl_2025_report",
                "annual_midea_2024_report", "annual_midea_2025_report",
            ],
        }
        block = answerer._financial_summary_snapshot_block(q)
        self.assertIn("营业收入=423,701,834/362,012,554", block)
        self.assertIn("基本每股收益=16.14/11.58", block)
        self.assertIn("营业收入=456,451,731/407,149,600", block)
        self.assertIn("基本每股收益=5.80/5.44", block)
        self.assertNotIn("annual_catl_2024_report", block)

    def test_summary_snapshot_recovers_research_expense_pair(self):
        q = {
            "domain": "financial_reports",
            "question": "比较2024年、2025年研发费用及研发费用占营业收入比例",
            "options": {"A": "两家公司研发费用率均上升"},
            "doc_ids": [
                "annual_catl_2025_report", "annual_midea_2025_report",
            ],
        }
        block = answerer._financial_summary_snapshot_block(q)
        self.assertIn("研发费用=22,146,581/18,606,756", block)
        self.assertIn("研发费用=17,787,624/16,232,771", block)

    def test_dividend_ratio_snapshot_binds_each_report_year(self):
        q = {
            "domain": "financial_reports",
            "question": "根据中国建筑2024年和2025年年度报告",
            "options": {"A": "2025年现金分红占归母净利润比例较2024年提高"},
            "doc_ids": [
                "annual_cscec_2024_report", "annual_cscec_2025_report",
            ],
        }
        block = answerer._financial_dividend_ratio_snapshot_block(q)
        self.assertIn("2024年度原报告明确比例=24.29%", block)
        self.assertIn("2025年度原报告明确比例=28.75%", block)
        self.assertNotIn("20.82%", block)

    def test_four_company_dividend_ranking_gets_qwen_scope_audit(self):
        q = {
            "domain": "financial_reports", "answer_format": "multi",
            "question": "根据四家公司2025年年度报告中的现金分红数据，以下哪些排序正确",
            "options": {
                "A": "全年每10股现金分红由高到低排序正确",
                "B": "某公司末期分红等于全年分红",
                "C": "每股现金分红换算为每10股正确",
                "D": "前两名差额正确",
            },
            "doc_ids": ["a", "b", "c", "d"],
        }
        profile = batch._choice_scope_audit_profile(q)
        self.assertEqual(profile["profile"],
                         "financial_full_year_dividend_ranking")


if __name__ == "__main__":
    unittest.main()
