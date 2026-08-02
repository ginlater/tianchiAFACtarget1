"""Network-free integration test for deterministic-calc + Qwen verification."""
import io
import json
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work"))

from agent import calc
from agent.deterministic_calc import Fact, Intent, SolveResult


class DeterministicCalcIntegrationTests(unittest.TestCase):
    @staticmethod
    def _solved(intent, facts):
        return SolveResult(intent, ("1.00",), "确定性算式", tuple(facts))

    def test_financial_and_research_budgets_are_semantic_and_qid_free(self):
        cross_facts = [
            Fact("revenue", 100, "hundred_million_yuan", entity="甲公司",
                 period="2025", source="甲P1"),
            Fact("operating_cash_flow", 20, "hundred_million_yuan",
                 entity="甲公司", period="2025", source="甲P1"),
            Fact("revenue", 80, "hundred_million_yuan", entity="乙公司",
                 period="2025", source="乙P2"),
            Fact("operating_cash_flow", 8, "hundred_million_yuan",
                 entity="乙公司", period="2025", source="乙P2"),
        ]
        solved = self._solved(Intent.CASHFLOW_RATE_RANK, cross_facts)
        budgets = []
        audits = []
        for qid in ("opaque_one", "unrelated_replay_id"):
            q = {"qid": qid, "domain": "financial_reports",
                 "question": "根据甲公司和乙公司年报计算经营现金流率并排序"}
            budgets.append(calc.deterministic_verifier_budget(q, solved))
            audits.append(calc.deterministic_scope_audit_budget(q, solved))
        self.assertEqual(budgets[0], budgets[1])
        self.assertEqual(budgets[0].profile,
                         "financial_cross_entity_rate_rank")
        self.assertEqual(audits[0], audits[1])
        self.assertEqual(audits[0].profile,
                         "financial_cross_entity_scope_audit")
        self.assertNotIn("qid", budgets[0].audit_dict())
        self.assertNotIn("qid", audits[0].audit_dict())

        demand = self._solved(
            Intent.DEMAND_GROWTH,
            [Fact("battery_per_vehicle", 45.8, "kwh", period="2025",
                  source="研报P10")],
        )
        q = {"qid": "anything", "domain": "research",
             "question": "销量持平、单车带电量提升时计算全年需求增速"}
        self.assertEqual(
            calc.deterministic_verifier_budget(q, demand).profile,
            "research_demand_population",
        )
        self.assertEqual(
            calc.deterministic_scope_audit_budget(q, demand).profile,
            "research_population_scope_audit",
        )

    def test_named_semantic_budget_is_not_overwritten_by_legacy_globals(self):
        solved = self._solved(
            Intent.DUPONT,
            [Fact("debt_ratio", 60, "percent", entity="甲公司"),
             Fact("roe", 12, "percent", entity="甲公司")],
        )
        q = {"qid": "opaque", "domain": "financial_reports",
             "question": "计算权益乘数和近似资产收益率"}
        budget = calc.deterministic_verifier_budget(q, solved)
        with mock.patch.dict(os.environ, {
                "AFAC_DET_CALC_EVIDENCE_CHARS": "999",
                "AFAC_DET_CALC_THINKING_BUDGET": "111",
        }, clear=False):
            routed = calc._det_budget_with_env_overrides(budget)
        self.assertEqual(routed.profile, "financial_dupont_closed")
        self.assertEqual(routed.evidence_chars, 5200)
        self.assertEqual(routed.thinking_budget, 1050)

    def test_margin_scope_audit_is_compact_and_semantic(self):
        facts = [
            Fact(metric, value, "yuan", entity="甲公司", period=period,
                 source=f"年报{period}")
            for period in ("2024", "2025")
            for metric, value in (("revenue", 100),
                                  ("parent_net_profit", 10),
                                  ("operating_cash_flow", 20))
        ]
        solved = self._solved(Intent.MARGIN_DROP, facts)
        q = {"qid": "opaque", "domain": "financial_reports",
             "question": "计算两年归母净利率和经营现金流率下降百分点。"}
        routed = calc.deterministic_scope_audit_budget(q, solved)
        self.assertEqual(routed.profile,
                         "financial_multi_metric_scope_audit")
        self.assertEqual((routed.evidence_chars, routed.thinking_budget,
                          routed.max_tokens), (2200, 500, 850))
        self.assertIn("每个年度只用一条", routed.instruction)
        self.assertNotIn("opaque", routed.instruction)

    def test_multi_valuation_budget_depends_on_text_structure_not_qid(self):
        facts = [
            Fact("appreciation_rate", 1468.47, "percent",
                 period="20230630", source="交易报告书P153"),
            Fact("appreciation_rate", 740.58, "percent",
                 period="20231231", source="交易报告书P220"),
        ]
        solved = SolveResult(
            Intent.DIRECT_SERIES, ("1468.47%", "740.58%"),
            "两个基准日分别绑定原文披露的评估增值率。", tuple(facts))
        budgets = []
        audits = []
        for qid in ("opaque_a", "opaque_b"):
            q = {
                "qid": qid,
                "domain": "financial_contracts",
                "question": (
                    "标的资产评估增值率在两次评估（基准日2023年6月30日"
                    "和2023年12月31日）中分别约为多少？"),
            }
            budgets.append(calc.deterministic_verifier_budget(q, solved))
            audits.append(calc.deterministic_scope_audit_budget(q, solved))
        self.assertEqual(budgets[0], budgets[1])
        self.assertEqual(audits[0], audits[1])
        self.assertEqual(budgets[0].profile,
                         "contract_multi_valuation_appreciation")
        self.assertEqual((budgets[0].evidence_chars,
                          budgets[0].thinking_budget,
                          budgets[0].max_tokens), (9500, 1450, 2300))
        self.assertEqual(audits[0].profile,
                         "contract_multi_valuation_scope_audit")
        self.assertEqual((audits[0].evidence_chars,
                          audits[0].thinking_budget,
                          audits[0].max_tokens), (1600, 200, 600))
        self.assertNotIn("qid", budgets[0].audit_dict())
        self.assertNotIn("qid", audits[0].audit_dict())

        single_q = {
            "qid": "does_not_matter",
            "domain": "financial_contracts",
            "question": "截至2023年12月31日，评估增值率约为多少？",
        }
        self.assertEqual(
            calc.deterministic_verifier_budget(single_q, solved).profile,
            "deterministic_default")
        self.assertIsNone(
            calc.deterministic_scope_audit_budget(single_q, solved))

    def test_contract_series_and_mean_budgets_follow_fact_structure(self):
        series_facts = [
            Fact("gross_margin", value, "percent", period=period,
                 source=f"募集说明书P{idx}")
            for idx, (period, value) in enumerate(
                (("2022", 10), ("2023", 11), ("2024", 12),
                 ("2025年1-6月", 13)), start=1)
        ]
        mean_facts = [
            Fact("parent_net_profit", value, "ten_thousand_yuan",
                 period=period, source=f"募集说明书P{idx}")
            for idx, (period, value) in enumerate(
                (("2023", 100), ("2024", 200), ("2025", 300)), start=1)
        ]
        cases = (
            (Intent.DIRECT_SERIES, series_facts,
             "2022-2024年及2025年1-6月主营业务毛利率分别为多少？",
             "contract_direct_series_dense", (5200, 1250, 1900)),
            (Intent.MEAN, mean_facts,
             "计算2023年至2025年归母净利润平均值。",
             "contract_multi_period_mean", (5200, 1600, 2300)),
        )
        for intent, facts, question, profile, expected in cases:
            solved = self._solved(intent, facts)
            routed = []
            for qid in ("opaque_a", "opaque_b"):
                routed.append(calc.deterministic_verifier_budget(
                    {"qid": qid, "domain": "financial_contracts",
                     "question": question}, solved))
            self.assertEqual(routed[0], routed[1])
            self.assertEqual(routed[0].profile, profile)
            self.assertEqual((routed[0].evidence_chars,
                              routed[0].thinking_budget,
                              routed[0].max_tokens), expected)

    def test_regulatory_verifier_budgets_follow_intent(self):
        cases = (
            (Intent.CALENDAR_OFFSET, "受理当日不计入，按90日计算何时决定？",
             "regulatory_calendar_offset", (5200, 2100, 3000)),
            (Intent.PERIODIC_INTERVAL, "后续每30日公告一次，间隔多少日？",
             "regulatory_periodic_interval", (4000, 1100, 1900)),
            (Intent.THRESHOLD_COUNT, "按单笔核实门槛需核实几笔？",
             "regulatory_threshold_count", (4200, 1350, 2200)),
            (Intent.ADVANCE_NOTICE, "至少提前30个自然日，应从何时公示？",
             "regulatory_advance_notice", (4200, 1500, 2400)),
        )
        for intent, question, profile, expected in cases:
            solved = self._solved(intent, ())
            budgets = [
                calc.deterministic_verifier_budget(
                    {"qid": qid, "domain": "regulatory",
                     "question": question}, solved)
                for qid in ("opaque_a", "opaque_b")
            ]
            self.assertEqual(budgets[0], budgets[1])
            self.assertEqual(budgets[0].profile, profile)
            self.assertEqual((budgets[0].evidence_chars,
                              budgets[0].thinking_budget,
                              budgets[0].max_tokens), expected)

    def test_next_business_day_budget_requires_regulatory_text_and_no_calendar(self):
        budgets = []
        for qid in ("opaque_a", "opaque_b"):
            q = {
                "qid": qid,
                "domain": "regulatory",
                "question": (
                    "若第60日为2026年3月27日，期满后次一工作日是哪一天？"),
            }
            budgets.append(calc.deterministic_scope_audit_budget(
                q, business_calendar_complete=False))
        self.assertEqual(budgets[0], budgets[1])
        self.assertEqual(budgets[0].profile,
                         "regulatory_next_business_day_no_calendar")
        self.assertEqual((budgets[0].evidence_chars,
                          budgets[0].thinking_budget,
                          budgets[0].max_tokens), (5000, 1350, 2200))
        self.assertIsNone(calc.deterministic_scope_audit_budget(
            {**q, "qid": "complete_calendar"},
            business_calendar_complete=True))
        self.assertIsNone(calc.deterministic_scope_audit_budget(
            {**q, "qid": "wrong_domain", "domain": "research"},
            business_calendar_complete=False))

    def test_multi_valuation_runs_a_real_scope_audit_before_verification(self):
        q = {
            "qid": "opaque_contract",
            "domain": "financial_contracts",
            "answer_format": "calc",
            "doc_ids": ["contract-document"],
            "question": (
                "标的资产评估增值率在两次评估（基准日2023年6月30日"
                "和2023年12月31日）中分别约为多少？"),
            "options": {},
        }
        solved = SolveResult(
            Intent.DIRECT_SERIES, ("1468.47%", "740.58%"),
            "分别绑定两个评估基准日。",
            (Fact("appreciation_rate", 1468.47, "percent",
                  period="20230630", source="P153"),
             Fact("appreciation_rate", 740.58, "percent",
                  period="20231231", source="P220")))
        calls = []

        def fake_chat(messages, **kwargs):
            calls.append((kwargs, messages[0]["content"]))
            if kwargs["tag"] == "calc_scope_audit":
                self.assertEqual(kwargs["thinking_budget"], 200)
                self.assertEqual(kwargs["max_tokens"], 600)
                self.assertIn("STATUS=OK或CONFLICT", messages[0]["content"])
                return ("【口径审计】两个基准日及其增值率已逐一绑定，无跨期错配。",
                        "", {"prompt_tokens": 100,
                             "completion_tokens": 50})
            self.assertEqual(kwargs["tag"], "calc_det")
            self.assertEqual(kwargs["thinking_budget"], 1450)
            self.assertEqual(kwargs["max_tokens"], 2300)
            self.assertIn("独立口径审计", messages[0]["content"])
            return ("【取数】两期原文增值率。\n【计算】按基准日顺序列示。\n"
                    "【核验】不存在跨期错配。\n答案: 1468.47%；740.58%",
                    "", {"prompt_tokens": 100,
                         "completion_tokens": 50})

        with mock.patch.dict(os.environ, {
                "AFAC_DET_CALC": "1", "AFAC_NARRATIVE_REGISTRY": "0",
        }, clear=False), \
                mock.patch.object(calc, "calc_evidence",
                                  return_value=("两期评估原文证据", ["d#c1"])), \
                mock.patch("agent.deterministic_calc.try_solve",
                           return_value=solved), \
                mock.patch.object(calc, "chat", side_effect=fake_chat):
            raw, info = calc.answer_calc(
                q, ["percent", "percent"], return_info=True)
        self.assertEqual(raw, "1468.47%；740.58%")
        self.assertEqual([kwargs["tag"] for kwargs, _prompt in calls],
                         ["calc_scope_audit", "calc_det"])
        self.assertEqual(info["scope_audit_budget_profile"]["profile"],
                         "contract_multi_valuation_scope_audit")

    def test_next_business_day_fallback_uses_semantic_budget(self):
        q = {
            "qid": "opaque_regulation",
            "domain": "regulatory",
            "answer_format": "calc",
            "doc_ids": ["regulation-document"],
            "question": (
                "若第60日为2026年3月27日，期满后次一工作日是哪一天？"),
            "options": {},
        }
        calls = []

        def fake_chat(messages, **kwargs):
            calls.append(kwargs)
            self.assertEqual(kwargs["tag"], "calc1")
            self.assertEqual(kwargs["thinking_budget"], 1350)
            self.assertEqual(kwargs["max_tokens"], 2200)
            self.assertIn("当前没有完整法定节假日日历",
                          messages[0]["content"])
            return ("【取数】期满日为2026年3月27日。\n"
                    "【计算】核对星期与周末后顺延。\n"
                    "答案: 2026年3月30日", "",
                    {"prompt_tokens": 100, "completion_tokens": 50})

        with mock.patch.dict(os.environ, {
                "AFAC_DET_CALC": "1", "AFAC_CALC_SINGLE": "1",
                "AFAC_NARRATIVE_REGISTRY": "0",
        }, clear=False), \
                mock.patch("pathlib.Path.exists", return_value=False), \
                mock.patch.object(calc, "calc_evidence",
                                  return_value=("法规原文证据", ["d#c1"])), \
                mock.patch.object(calc, "chat", side_effect=fake_chat):
            raw, info = calc.answer_calc(q, ["date"], return_info=True)
        self.assertEqual(raw, "2026年3月30日")
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            info["fallback_scope_budget_profile"]["profile"],
            "regulatory_next_business_day_no_calendar")

    def test_strict_result_is_returned_only_after_qwen_visible_verification(self):
        q = {
            "qid": "arbitrary_id",
            "domain": "financial_contracts",
            "answer_format": "calc",
            "doc_ids": ["doc-any"],
            "question": (
                "每股净资产的评估值为多少？已知股东全部权益评估值为"
                "1,298,036.10万元，注册资本为564,141.7298万元"
                "（即总股本数，单位万股）。请保留两位小数。"),
            "options": {},
        }

        def fake_chat(_messages, **kwargs):
            self.assertEqual(kwargs["tag"], "calc_det")
            self.assertTrue(kwargs["thinking"])
            return ("【取数】权益评估值与总股本来自题干。\n"
                    "【计算】1298036.10÷564141.7298=2.30。\n"
                    "【核验】单位换算相消，仅最终保留两位。\n答案: 2.30",
                    "", {"prompt_tokens": 100, "completion_tokens": 50})

        with mock.patch.dict(os.environ, {"AFAC_DET_CALC": "1"},
                             clear=False), \
                mock.patch.object(calc, "calc_evidence",
                                  return_value=("题干取数证据", ["d#c1"])), \
                mock.patch.object(calc, "chat", side_effect=fake_chat):
            raw, info = calc.answer_calc(q, ["number"], return_info=True)
        self.assertEqual(raw, "2.30")
        self.assertEqual(info["reasoning_stage"], "calc_det")
        self.assertEqual(info["deterministic_intent"], "per_share")
        self.assertIn("答案: 2.30", info["reasoning"])

    def test_calendar_offset_mismatch_uses_compact_qwen_reconcile(self):
        q = {
            "qid": "opaque_calendar",
            "domain": "regulatory",
            "answer_format": "calc",
            "doc_ids": ["rule-document"],
            "question": (
                "申请在2026年4月1日被受理，若受理当日不计入，且按90日计算，"
                "最迟应在何时作出决定？"),
            "options": {},
        }
        calls = []

        def fake_chat(messages, **kwargs):
            calls.append(kwargs["tag"])
            if kwargs["tag"] == "calc_det":
                return ("【计算】误把4月2日至30日算成30日。\n"
                        "答案: 2026年6月29日", "",
                        {"prompt_tokens": 100, "completion_tokens": 50})
            self.assertEqual(kwargs["tag"], "calc_det_reconcile")
            self.assertEqual(kwargs["thinking_budget"], 250)
            self.assertEqual(kwargs["max_tokens"], 600)
            self.assertIn("日期差必须恰好等于题定日数",
                          messages[0]["content"])
            return ("【边界】4月2日为第1日。\n"
                    "【日期差】6月30日与4月1日相差90日。\n"
                    "【结论】第90日为6月30日。\n答案: 2026年6月30日", "",
                    {"prompt_tokens": 100, "completion_tokens": 50})

        with mock.patch.dict(os.environ, {
                "AFAC_DET_CALC": "1", "AFAC_NARRATIVE_REGISTRY": "0",
        }, clear=False), \
                mock.patch.object(calc, "calc_evidence",
                                  return_value=("法规原文", ["d#c1"])), \
                mock.patch.object(calc, "chat", side_effect=fake_chat):
            raw, info = calc.answer_calc(q, ["date"], return_info=True)
        self.assertEqual(raw, "2026年6月30日")
        self.assertEqual(calls, ["calc_det", "calc_det_reconcile"])
        self.assertEqual(info["reasoning_stage"], "calc_det_reconcile")
        self.assertEqual([trace["stage"] for trace in info["traces"]],
                         ["calc_det", "calc_det_reconcile"])

    def test_closed_semantic_mismatch_requires_qwen_reconcile(self):
        q = {
            "qid": "opaque_mixture",
            "domain": "research",
            "answer_format": "calc",
            "doc_ids": ["research-document"],
            "question": (
                "APP自营GMV为21亿元且不含会员费。会员24.01万人，"
                "年均商品消费2960元，普通用户消费2072元，求普通用户人数。"),
            "options": {},
        }
        solved = SolveResult(
            Intent.MIXTURE_COUNT, ("67.1",),
            "GMV不含会员费；(2100000000-710696000)/2072=67.05135万人。",
            ())
        calls = []

        def fake_chat(messages, **kwargs):
            calls.append(kwargs["tag"])
            if kwargs["tag"] == "calc_det":
                return ("【核验】把会员费计入会员GMV贡献。\n答案: 64.7", "",
                        {"prompt_tokens": 100, "completion_tokens": 50})
            self.assertEqual(kwargs["tag"], "calc_det_closed_reconcile")
            self.assertFalse(kwargs["thinking"])
            self.assertNotIn("thinking_budget", kwargs)
            self.assertEqual(kwargs["max_tokens"], 240)
            self.assertIn("下列甲乙方案都不预设正确", messages[0]["content"])
            self.assertIn("GMV不含会员费", messages[0]["content"])
            self.assertLessEqual(len(messages[0]["content"]), 2400)
            return ("【口径】会员费在GMV之外，只扣会员商品消费。\n"
                    "【复算】(2100000000-710696000)÷2072=67.05135万人。\n"
                    "答案: 67.1", "",
                    {"prompt_tokens": 100, "completion_tokens": 50})

        with mock.patch.dict(os.environ, {
                "AFAC_DET_CALC": "1", "AFAC_NARRATIVE_REGISTRY": "0",
        }, clear=False), \
                mock.patch.object(calc, "calc_evidence",
                                  return_value=("闭式事实证据", ["d#c1"])), \
                mock.patch("agent.deterministic_calc.try_solve",
                           return_value=solved), \
                mock.patch.object(calc, "chat", side_effect=fake_chat):
            raw, info = calc.answer_calc(q, ["number"], return_info=True)
        self.assertEqual(raw, "67.10")
        self.assertEqual(calls, ["calc_det", "calc_det_closed_reconcile"])
        self.assertEqual(info["reasoning_stage"],
                         "calc_det_closed_reconcile")
        self.assertEqual([trace["stage"] for trace in info["traces"]],
                         ["calc_det", "calc_det_closed_reconcile"])

    def test_matched_deterministic_trace_is_schema_canonical(self):
        q = {
            "qid": "opaque_mixture",
            "domain": "research",
            "answer_format": "calc",
            "doc_ids": ["research-document"],
            "question": "APP自营GMV不含会员费，计算普通用户人数。",
            "options": {},
        }
        solved = SolveResult(
            Intent.MIXTURE_COUNT, ("67.1",), "闭式结果为67.1万人。", ())

        def fake_chat(_messages, **kwargs):
            self.assertEqual(kwargs["tag"], "calc_det")
            return ("【核验】口径与闭式算式一致。\n答案: 67.1万人", "",
                    {"prompt_tokens": 100, "completion_tokens": 50})

        with mock.patch.dict(os.environ, {
                "AFAC_DET_CALC": "1", "AFAC_NARRATIVE_REGISTRY": "0",
        }, clear=False), \
                mock.patch.object(calc, "calc_evidence",
                                  return_value=("闭式事实证据", ["d#c1"])), \
                mock.patch("agent.deterministic_calc.try_solve",
                           return_value=solved), \
                mock.patch.object(calc, "chat", side_effect=fake_chat):
            raw, info = calc.answer_calc(q, ["number"], return_info=True)
        self.assertEqual(raw, "67.10")
        self.assertEqual(info["raw_answer"], "67.10")
        self.assertEqual(info["traces"][0]["answer"], "67.10")
        self.assertIn("67.1万人", info["reasoning"])

    def test_closed_semantic_ambiguous_reconcile_falls_back(self):
        q = {
            "qid": "opaque_mixture",
            "domain": "research",
            "answer_format": "calc",
            "doc_ids": ["research-document"],
            "question": (
                "APP自营GMV为21亿元且不含会员费。会员24.01万人，"
                "年均商品消费2960元，普通用户消费2072元，求普通用户人数。"),
            "options": {},
        }
        solved = SolveResult(
            Intent.MIXTURE_COUNT, ("67.1",),
            "GMV不含会员费；(2100000000-710696000)/2072=67.05135万人。",
            ())
        calls = []

        def fake_chat(_messages, **kwargs):
            calls.append(kwargs["tag"])
            if kwargs["tag"] == "calc_det":
                content = "【核验】把会员费计入贡献。\n答案: 64.7"
            elif kwargs["tag"] == "calc_det_closed_reconcile":
                content = "【口径】两种均可能。\n答案: 67.1 或 64.7"
            else:
                self.assertEqual(kwargs["tag"], "calc1")
                content = "【完整证据复核】采用会员商品消费口径。\n答案: 64.7"
            return content, "", {"prompt_tokens": 100,
                                    "completion_tokens": 50}

        with mock.patch.dict(os.environ, {
                "AFAC_DET_CALC": "1", "AFAC_NARRATIVE_REGISTRY": "0",
                "AFAC_CALC_SINGLE": "1",
        }, clear=False), \
                mock.patch.object(calc, "calc_evidence",
                                  return_value=("闭式事实证据", ["d#c1"])), \
                mock.patch("agent.deterministic_calc.try_solve",
                           return_value=solved), \
                mock.patch.object(calc, "chat", side_effect=fake_chat):
            raw, info = calc.answer_calc(q, ["number"], return_info=True)
        self.assertEqual(raw, "64.7")
        self.assertEqual(calls,
                         ["calc_det", "calc_det_closed_reconcile", "calc1"])
        self.assertEqual(info["reasoning_stage"], "calc1")
        self.assertEqual([trace["stage"] for trace in info["traces"]],
                         ["calc_det", "calc_det_closed_reconcile", "calc1"])

    def test_closed_semantic_nonstrict_result_skips_reconcile(self):
        q = {
            "qid": "opaque_mixture",
            "domain": "research",
            "answer_format": "calc",
            "doc_ids": ["research-document"],
            "question": "APP自营GMV不含会员费，求普通用户人数。",
            "options": {},
        }
        solved = SolveResult(
            Intent.MIXTURE_COUNT, ("67.1",), "候选算式", (),
            confidence="tentative")
        calls = []

        def fake_chat(_messages, **kwargs):
            calls.append(kwargs["tag"])
            content = ("【首轮】答案有分歧。\n答案: 64.7"
                       if kwargs["tag"] == "calc_det" else
                       "【完整复核】保留首轮。\n答案: 64.7")
            return content, "", {"prompt_tokens": 100,
                                    "completion_tokens": 50}

        with mock.patch.dict(os.environ, {
                "AFAC_DET_CALC": "1", "AFAC_NARRATIVE_REGISTRY": "0",
                "AFAC_CALC_SINGLE": "1",
        }, clear=False), \
                mock.patch.object(calc, "calc_evidence",
                                  return_value=("闭式事实证据", ["d#c1"])), \
                mock.patch("agent.deterministic_calc.try_solve",
                           return_value=solved), \
                mock.patch.object(calc, "chat", side_effect=fake_chat):
            raw, _info = calc.answer_calc(q, ["number"], return_info=True)
        self.assertEqual(raw, "64.7")
        self.assertEqual(calls, ["calc_det", "calc1"])

    def test_closed_semantic_scope_conflict_skips_reconcile(self):
        q = {
            "qid": "opaque_growth",
            "domain": "research",
            "answer_format": "calc",
            "doc_ids": ["research-document"],
            "question": "销量持平，单车带电量从45.8提升至56，求需求增速。",
            "options": {},
        }
        solved = SolveResult(
            Intent.DEMAND_GROWTH, ("22.27%",),
            "(56/45.8-1)*100%=22.27%",
            (Fact("battery_per_vehicle", 45.8, "kwh", period="2025"),))
        calls = []

        def fake_chat(_messages, **kwargs):
            calls.append(kwargs["tag"])
            responses = {
                "calc_scope_audit":
                    "CONFLICT|披露总需求与因子比例口径竞争\nSTATUS=CONFLICT",
                "calc_det": "【核验】使用披露总需求。\n答案: 22.19%",
                "calc1": "【完整复核】使用披露总需求。\n答案: 22.19%",
            }
            return responses[kwargs["tag"]], "", {
                "prompt_tokens": 100, "completion_tokens": 50}

        with mock.patch.dict(os.environ, {
                "AFAC_DET_CALC": "1", "AFAC_NARRATIVE_REGISTRY": "0",
                "AFAC_CALC_SINGLE": "1",
        }, clear=False), \
                mock.patch.object(calc, "calc_evidence",
                                  return_value=("闭式事实证据", ["d#c1"])), \
                mock.patch("agent.deterministic_calc.try_solve",
                           return_value=solved), \
                mock.patch.object(calc, "chat", side_effect=fake_chat):
            raw, _info = calc.answer_calc(q, ["percent"], return_info=True)
        self.assertEqual(raw, "22.19%")
        self.assertEqual(calls, ["calc_scope_audit", "calc_det", "calc1"])

    def test_yoy_share_budget_has_token_headroom(self):
        solved = SolveResult(
            Intent.YOY_AND_SHARE_PP, ("40.05", "10.10"), "确定性算式", ())
        q = {"qid": "opaque_yoy", "domain": "financial_reports",
             "question": "计算两年境外收入同比及占比百分点。"}
        budget = calc.deterministic_verifier_budget(q, solved)
        self.assertEqual(budget.profile, "financial_yoy_share")
        self.assertEqual(budget.thinking_budget, 900)
        self.assertEqual(budget.max_tokens, 1800)

    def test_dupont_no_percent_sign_uses_compact_semantic_reconcile(self):
        q = {
            "qid": "opaque_dupont",
            "domain": "financial_reports",
            "answer_format": "calc",
            "doc_ids": ["annual-report"],
            "question": (
                "资产负债率61.17%，ROE19.70%，计算权益乘数和近似资产收益率，"
                "后者保留两位小数且不带百分号。"),
            "options": {},
        }
        solved = SolveResult(
            Intent.DUPONT, ("2.58", "7.65"),
            "权益乘数=1/(1-0.6117)=2.575328；"
            "近似资产收益率=0.197/2.575328=7.64951%。",
            (Fact("debt_ratio", 61.17, "percent"),
             Fact("roe", 19.70, "percent")))
        calls = []

        def fake_chat(messages, **kwargs):
            calls.append(kwargs["tag"])
            if kwargs["tag"] == "calc_det":
                return ("【核验】把不带百分号理解成小数。\n答案: 2.58；0.08",
                        "", {"prompt_tokens": 100, "completion_tokens": 50})
            self.assertEqual(kwargs["tag"], "calc_det_closed_reconcile")
            self.assertFalse(kwargs["thinking"])
            self.assertEqual(kwargs["max_tokens"], 240)
            self.assertIn("不得把7.65%改写为小数0.08", messages[0]["content"])
            return ("【口径】百分数数值保留，仅去掉%符号。\n"
                    "【复算】0.197/2.575328=7.64951%，舍入为7.65。\n"
                    "答案: 2.58；7.65", "",
                    {"prompt_tokens": 100, "completion_tokens": 50})

        with mock.patch.dict(os.environ, {
                "AFAC_DET_CALC": "1", "AFAC_NARRATIVE_REGISTRY": "0",
        }, clear=False), \
                mock.patch.object(calc, "calc_evidence",
                                  return_value=("闭式事实证据", ["d#c1"])), \
                mock.patch("agent.deterministic_calc.try_solve",
                           return_value=solved), \
                mock.patch.object(calc, "chat", side_effect=fake_chat):
            raw, info = calc.answer_calc(
                q, ["number", "number"], return_info=True)
        self.assertEqual(raw, "2.58；7.65")
        self.assertEqual(calls, ["calc_det", "calc_det_closed_reconcile"])
        self.assertEqual(info["reasoning_stage"],
                         "calc_det_closed_reconcile")

    def test_equity_multiplier_rounding_mismatch_uses_compact_reconcile(self):
        q = {
            "qid": "opaque_leverage_rank",
            "domain": "financial_reports",
            "answer_format": "calc",
            "doc_ids": ["annual-a", "annual-b", "annual-c"],
            "question": (
                "根据三家公司的资产负债率计算权益乘数，按权益乘数从高到低"
                "排序，并计算最高值与最低值之差，保留两位小数。"),
            "options": {},
        }
        solved = SolveResult(
            Intent.EQUITY_MULTIPLIER_RANK,
            ("春华集团>秋实集团>远山集团", "0.84"),
            ("春华集团=1/(1-0.7073)=3.417...；"
             "秋实集团=1/(1-0.6841)=3.165...；"
             "远山集团=1/(1-0.6117)=2.575...；"
             "完整精度差值=0.842...，最终舍入为0.84。"),
            (Fact("debt_ratio", 70.73, "percent", entity="春华集团",
                  period="2025"),
             Fact("debt_ratio", 68.41, "percent", entity="秋实集团",
                  period="2025"),
             Fact("debt_ratio", 61.17, "percent", entity="远山集团",
                  period="2025")),
        )
        calls = []

        def fake_chat(messages, **kwargs):
            calls.append(kwargs["tag"])
            if kwargs["tag"] == "calc_scope_audit":
                return ("BIND|三家公司|2025年|合并口径\nSTATUS=OK", "", {
                    "prompt_tokens": 100, "completion_tokens": 20})
            if kwargs["tag"] == "calc_det":
                return ("【核验】手算时改写了春华集团的分母。\n"
                        "答案: 春华集团>秋实集团>远山集团；0.85", "", {
                            "prompt_tokens": 100, "completion_tokens": 50})
            self.assertEqual(kwargs["tag"],
                             "calc_det_closed_reconcile")
            self.assertFalse(kwargs["thinking"])
            self.assertEqual(kwargs["max_tokens"], 240)
            prompt = messages[0]["content"]
            self.assertIn("1/(1-资产负债率)", prompt)
            self.assertIn("只在最终数值槽", prompt)
            return ("【口径】三家集团均按合并口径资产负债率。\n"
                    "【复算】使用完整精度计算、排序并求极差，最终舍入。\n"
                    "答案: 春华集团>秋实集团>远山集团；0.84", "", {
                        "prompt_tokens": 100, "completion_tokens": 50})

        with mock.patch.dict(os.environ, {
                "AFAC_DET_CALC": "1", "AFAC_NARRATIVE_REGISTRY": "0",
        }, clear=False), \
                mock.patch.object(calc, "calc_evidence",
                                  return_value=("闭式事实证据", ["d#c1"])), \
                mock.patch("agent.deterministic_calc.try_solve",
                           return_value=solved), \
                mock.patch.object(calc, "chat", side_effect=fake_chat):
            raw, info = calc.answer_calc(
                q, ["ranking", "number"], return_info=True)
        self.assertEqual(raw, "春华集团>秋实集团>远山集团；0.84")
        self.assertEqual(calls, ["calc_scope_audit", "calc_det",
                                 "calc_det_closed_reconcile"])
        self.assertEqual(info["reasoning_stage"],
                         "calc_det_closed_reconcile")
        self.assertEqual([trace["stage"] for trace in info["traces"]],
                         ["calc_det", "calc_det_closed_reconcile"])

    def test_unique_registry_rescue_is_cached_and_suppresses_redundant_retry(self):
        q = {
            "qid": "opaque_contract",
            "domain": "financial_contracts",
            "answer_format": "calc",
            "doc_ids": ["wrong_selected_doc"],
            "question": "计算2023年至2025年归母净利润平均值。",
            "options": {},
        }
        rescued_facts = (
            SimpleNamespace(doc_id="right_source_doc", page=8,
                            metric="parent_net_profit"),
        )
        structured = (
            Fact("parent_net_profit", 100, "ten_thousand_yuan",
                 period="2023", source="right_source_doc P8"),
        )
        extract_calls = []
        evidence_calls = []
        chat_tags = []

        def fake_extract(item, **kwargs):
            extract_calls.append((tuple(item.get("doc_ids") or ()), kwargs))
            return rescued_facts if kwargs.get("doc_ids") == () else ()

        def fake_evidence(item, **kwargs):
            evidence_calls.append((item, kwargs))
            self.assertEqual(item["doc_ids"],
                             ["wrong_selected_doc", "right_source_doc"])
            self.assertEqual(kwargs["narrative_facts"], rescued_facts)
            return "唯一事实束及原文证据", ["right_source_doc#p8"]

        def fake_chat(_messages, **kwargs):
            chat_tags.append(kwargs["tag"])
            return (
                "【取数】三期事实已由唯一来源束绑定。\n"
                "【计算】平均值为2.00。\n补充检索: 无需，事实已完整\n"
                "答案: 2.00",
                "", {"prompt_tokens": 100, "completion_tokens": 50})

        log = io.StringIO()
        env = {"AFAC_NARRATIVE_REGISTRY": "1", "AFAC_DET_CALC": "1",
               "AFAC_CALC_SINGLE": "1"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch("agent.narrative_fact_registry.extract_facts",
                           side_effect=fake_extract), \
                mock.patch("agent.narrative_fact_registry.calculation_facts",
                           return_value=structured) as convert, \
                mock.patch.object(calc, "calc_evidence",
                                  side_effect=fake_evidence), \
                mock.patch("pathlib.Path.exists", return_value=False), \
                mock.patch("agent.deterministic_calc.try_solve",
                           return_value=None) as solve, \
                mock.patch.object(calc, "chat", side_effect=fake_chat):
            raw, info = calc.answer_calc(
                q, ["number"], log=log, return_info=True)

        self.assertEqual(raw, "2.00")
        self.assertEqual(chat_tags, ["calc1"])
        self.assertEqual(len(evidence_calls), 1)
        self.assertEqual(len(extract_calls), 2)
        self.assertEqual(extract_calls[1][1]["doc_ids"], ())
        convert.assert_called_once_with(q["question"], rescued_facts)
        self.assertIs(solve.call_args.kwargs["facts"], structured)
        audit = info["narrative_source_audit"]
        self.assertEqual(audit["status"], "rescued_unique_bundle")
        self.assertEqual(audit["added_doc_ids"], ["right_source_doc"])
        self.assertTrue(audit["unique_complete_bundle"])
        suppression = info["calc1b_retry_suppression"]
        self.assertEqual(
            suppression["reason"],
            "valid_calc_with_unique_complete_narrative_fact_bundle")
        record = json.loads(log.getvalue())
        self.assertEqual(record["narrative_source_audit"], audit)
        self.assertEqual(record["calc1b_retry_suppression"], suppression)


if __name__ == "__main__":
    unittest.main()
