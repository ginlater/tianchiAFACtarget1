"""Semantic routing tests for targeted choice-question Qwen audits."""
from pathlib import Path
import unittest
from unittest import mock

from agent import answerer, batch
from agent.b_schema import load_questions


ROOT = Path(__file__).resolve().parents[2]


class ChoiceScopeAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.questions = {
            q["qid"]: q
            for q in load_questions(ROOT / "upload_b" / "question_b")
        }

    def test_contract_structures_route_without_identifier_dependency(self):
        expected = {
            "fc_b_003": "contract_compound_named_sections",
            "fc_b_007": "contract_cross_document_literal_presence",
            "fc_b_011": "contract_multi_period_metric_table",
            "fc_b_019": "contract_risk_governance_inventory",
        }
        for qid, profile in expected.items():
            with self.subTest(qid=qid):
                q = dict(self.questions[qid], qid="opaque")
                if qid == "fc_b_007":
                    q["doc_ids"] = ["source-a", "source-b", "source-c"]
                self.assertEqual(
                    batch._choice_scope_audit_profile(q)["profile"], profile)
        quantitative = dict(self.questions["fc_b_009"], qid="opaque",
                            doc_ids=["source-a", "source-b", "source-c"])
        self.assertIsNone(batch._choice_scope_audit_profile(quantitative))

    def test_regulatory_structures_route_from_visible_rule_shape(self):
        q1 = dict(self.questions["reg_b_001"], qid="opaque",
                  doc_ids=["rule-a", "rule-b", "rule-c"])
        effective_date = batch._choice_scope_audit_profile(q1)
        self.assertEqual(effective_date["profile"],
                         "regulatory_cross_rule_effective_date")
        self.assertEqual(effective_date["evidence_chars"], 500)
        for qid, profile in (
                ("reg_b_015", "regulatory_registry_discrepancy_exception"),
                ("reg_b_021", "regulatory_uncertain_simplification_guard")):
            q = dict(self.questions[qid], qid="opaque")
            self.assertEqual(
                batch._choice_scope_audit_profile(q)["profile"], profile)

    def test_compact_numeric_rule_makes_percentage_point_scope_hard(self):
        q = dict(self.questions["fin_b_006"], qid="opaque")
        with mock.patch.dict("os.environ", {"AFAC_COMPACT_JUDGE": "1"}):
            rule = answerer.judge_std_for(q)
        self.assertIn("口径题硬约束", rule)
        self.assertIn("二者不可互换", rule)
        self.assertIn("不适用轻度转述", rule)

    def test_first_pass_budget_uses_visible_domain_route(self):
        insurance = [dict(self.questions[qid], qid=f"opaque-{pos}")
                     for pos, qid in enumerate(
                         ("ins_b_002", "ins_b_004", "ins_b_014"))]
        self.assertEqual(batch._batch_b1_thinking_budget(insurance), 2400)
        financial = [dict(self.questions["fin_b_006"], qid="opaque")]
        self.assertEqual(batch._batch_b1_thinking_budget(financial), 2500)
        contract = [dict(self.questions["fc_b_013"], qid="opaque")]
        self.assertEqual(batch._batch_b1_thinking_budget(contract), 2600)

    def test_sparse_contract_batch_is_source_structural(self):
        qs = [
            {"domain": "financial_contracts", "doc_ids": [doc]}
            for doc in ("a", "b", "c")
        ]
        self.assertTrue(batch._contract_sparse_three_source_batch(qs))
        qs[2]["doc_ids"] = ["a"]
        self.assertFalse(batch._contract_sparse_three_source_batch(qs))

    def test_scope_window_protects_each_option_source(self):
        evidence = "\n\n".join((
            "【text03 P174】发行人无法按时还本付息时，给予自原约定给付日起"
            "90个自然日宽限期。",
            "【text03 P216】债券持有人会议规则产生的纠纷，向厦门仲裁委员会"
            "提起仲裁。",
            "【text03 P173】出现交叉保护情形，应在10个交易日内恢复承诺；"
            "未恢复的，持有人可要求采取负面事项救济措施。",
            "【text03 P176】募集说明书约定争议协商不成，向位于发行人住所"
            "所在地有管辖权的法院提请诉讼。",
        ))
        q = dict(self.questions["fc_b_003"], qid="opaque")
        window = batch._balanced_scope_audit_window(evidence, q, 700)
        self.assertLessEqual(len(window), 700)
        self.assertIn("10个交易日", window)
        self.assertIn("90个自然日", window)
        self.assertIn("法院提请诉讼", window)
        self.assertIn("负面事项救济", window)

    def test_capacity_presence_route_protects_literal_counterevidence(self):
        q = dict(self.questions["fc_b_007"], qid="opaque",
                 doc_ids=["text04", "text05", "text11"])
        evidence, kept, protected = answerer.gather_evidence(q, cap=9000)
        hits = [chunk for chunk in kept
                if chunk["doc_id"] == "text04" and chunk.get("page") == 95]
        self.assertTrue(hits)
        self.assertTrue(any("不涉及产能" in chunk["text"] for chunk in hits))
        self.assertTrue(any(chunk["id"] in protected for chunk in hits))
        self.assertIn("不涉及产能", evidence)
        window = batch._balanced_scope_audit_window(evidence, q, 800)
        self.assertIn("不涉及产能", window)
        scan = batch._literal_presence_scan_block(q, 320)
        self.assertIn("目标=产能消化", scan)
        self.assertIn("A|POS_MENTION|安克创新/text04|hit=0", scan)
        self.assertIn("B|DIRECT_TEXT|本川智能/text05", scan)
        self.assertIn("预计年销售额合计约40,500万元", scan)
        self.assertIn("相关性较高", scan)
        self.assertIn("C|NEG_MENTION|普联软件/text11|hit=0", scan)
        self.assertIn("软件和信息技术服务行业", scan)

    def test_scope_protocol_accepts_only_complete_consistent_decision(self):
        q = dict(self.questions["fc_b_003"], qid="opaque")
        complete = """<QWEN_DECISION_V1>
答案: ABCD
A|IN|P173明确10个交易日
B|IN|P174明确90个自然日
C|IN|P176明确住所地法院诉讼
D|IN|P173明确负面事项救济
STATUS|COMPLETE
</QWEN_DECISION_V1>"""
        parsed = batch._parse_scope_decision(complete, q)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["answer"], "ABCD")

        truncated = complete.replace("</QWEN_DECISION_V1>", "")
        self.assertEqual(
            batch._parse_scope_decision(truncated, q)["error"],
            "TRUNCATED_OR_UNDELIMITED")

        mismatch = complete.replace("答案: ABCD", "答案: ABC")
        self.assertEqual(
            batch._parse_scope_decision(mismatch, q)["error"],
            "FINAL_SET_MISMATCH")

        bare = complete.replace("<QWEN_DECISION_V1>\n", "").replace(
            "\n</QWEN_DECISION_V1>", "")
        self.assertTrue(batch._parse_scope_decision(bare, q)["valid"])

    def test_scope_protocol_need_evidence_never_supplies_letters(self):
        q = dict(self.questions["fc_b_003"], qid="opaque")
        content = """<QWEN_DECISION_V1>
答案: -
A|IN|P173明确10个交易日
B|IN|P174明确90个自然日
C|UNKNOWN|窗口未含争议解决原句
D|IN|P173明确负面事项救济
STATUS|NEED_EVIDENCE
</QWEN_DECISION_V1>"""
        parsed = batch._parse_scope_decision(content, q)
        self.assertTrue(parsed["valid"])
        self.assertEqual(parsed["status"], "NEED_EVIDENCE")
        self.assertIsNone(parsed["answer"])

        provisional = content.replace("答案: -", "答案: ACD")
        parsed = batch._parse_scope_decision(provisional, q)
        self.assertTrue(parsed["valid"])
        self.assertIsNone(parsed["answer"])
        self.assertEqual(parsed["provisional_answer_ignored"], "ACD")


if __name__ == "__main__":
    unittest.main()
