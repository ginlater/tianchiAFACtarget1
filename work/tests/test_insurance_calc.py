"""Regression tests for deterministic, source-bound insurance calculations."""

from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work"))

from agent.insurance_calc import (  # noqa: E402
    parse_age_death_schedule,
    parse_net_premium_death_rule,
    parse_policy_year_schedule,
    parse_post_annuity_death_rule,
    parse_surrender_fee_schedule,
    try_solve_insurance,
    try_solve_insurance_cash_value,
)
from agent import calc as calc_module  # noqa: E402
from agent.calc import deterministic_verifier_budget  # noqa: E402


QUESTION = (
    "平安智盈金生在养老保险金开始领取日前解除合同。若累计所交保险费为100万元、"
    "保单账户累计收益为20万元，分别在第3、第5、第8、第12个保单年度解除合同时，"
    "第8个保单年度与第3个保单年度的现金价值差额为多少万元？答案只填写数字。"
)

SURRENDER_FOUR = (
    "四款养老/寿险产品在犹豫期后解除合同时适用不同规则。某投保人分别持有以下合同并"
    "在相应保单年度末解除：平安智盈金生第8个保单年度末，累计所交保险费80万元、"
    "保单账户累计收益12万元；国寿增益宝第3个保单年度末，个人账户价值90万元；"
    "国寿鑫享添盈现金价值70万元；平安富鸿金生现金价值86万元。"
    "四份合同合计可退还多少万元？答案只填写数字。"
)

DEATH_FOUR = (
    "某人分别持有四份合同：国寿增益宝被保险人40周岁，基本保险金额90万元、"
    "个人账户价值100万元；平安智盈金生已开始领取养老保险金，养老保险金开始领取日的"
    "保单账户价值120万元，累计已产生养老保险金45万元；平安富鸿金生累计已交保险费"
    "100万元、累计已给付养老保险金35万元、现金价值72万元；国寿鑫享添盈所交保险费"
    "100万元、累计已给付养老年金25万元、现金价值68万元。"
    "四份合同合计身故保险金为多少万元？答案只填写数字。"
)

POST_ANNUITY_TWO = (
    "平安智盈金生在养老保险金开始领取日及之后身故。若养老保险金开始领取日的保单"
    "账户价值为150万元，累计已产生养老保险金分别为160万元和90万元两种情形，"
    "对应身故保险金合计为多少万元？答案只填写数字。"
)

SURRENDER_FOUR_LATER = (
    "四份合同分别发生犹豫期后退保或解除：平安智盈金生在第12个保单年度解除，"
    "累计所交保险费60万元、保单账户累计收益10万元；国寿增益宝在第5个保单年度解除，"
    "个人账户价值50万元；国寿鑫享添盈退还现金价值45万元；"
    "平安富鸿金生退还现金价值48万元。四份合同合计可退还多少万元？答案只填写数字。"
)


class InsurancePolicyYearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "work" / "processed_data" / "insurance_capsules.json"
        cls.data = json.loads(path.read_text(encoding="utf-8"))
        cls.doc = cls.data["documents"]["1"]
        cls.card = next(card for card in cls.doc["capsules"]
                        if card.get("id") == "1:p13:0040")

    def test_flat_table_alignment_and_boundaries(self) -> None:
        schedule = parse_policy_year_schedule(self.card, self.doc["identity"])
        self.assertIsNotNone(schedule)
        self.assertEqual(dict(schedule.exact_rates), {
            1: 95, 2: 97, 3: 99, 4: 100, 5: 100,
        })
        expected = {
            1: "95", 3: "99", 5: "100", 6: "115",
            10: "115", 11: "118", 12: "118",
        }
        for year, value in expected.items():
            with self.subTest(year=year):
                self.assertEqual(str(schedule.cash_value(year, 100, 20)), value)

    def test_real_shape_solves_to_sixteen(self) -> None:
        q = {"domain": "insurance", "question": QUESTION,
             "options": {}, "doc_ids": ["1", "10"]}
        got = try_solve_insurance_cash_value(q)
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("16.00",))
        self.assertIn("第8个保单年度现金价值=115.00万元", got.reasoning)
        self.assertIn("第3个保单年度现金价值=99.00万元", got.reasoning)

    def test_qid_independent_and_irrelevant_document_invariant(self) -> None:
        for qid, docs in (("anything", ["1"]), ("totally_new", ["10", "1"])):
            q = {"qid": qid, "domain": "insurance", "question": QUESTION,
                 "options": {}, "doc_ids": docs}
            got = try_solve_insurance_cash_value(q)
            self.assertIsNotNone(got)
            self.assertEqual(got.slots, ("16.00",))

    def test_missing_named_product_document_fails_closed(self) -> None:
        q = {"domain": "insurance", "question": QUESTION,
             "options": {}, "doc_ids": ["10"]}
        self.assertIsNone(try_solve_insurance_cash_value(q))

    def test_misaligned_table_fails_closed(self) -> None:
        bad = dict(self.card)
        bad["verbatim"] = bad["verbatim"].replace("\n99%", "", 1)
        self.assertIsNone(parse_policy_year_schedule(bad, self.doc["identity"]))

        missing_year = dict(self.card)
        missing_year["verbatim"] = missing_year["verbatim"].replace(
            "第3 个\n保单年度\n", "", 1)
        self.assertIsNone(parse_policy_year_schedule(
            missing_year, self.doc["identity"]))


class InsuranceClauseParserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "work" / "processed_data" / "insurance_capsules.json"
        cls.data = json.loads(path.read_text(encoding="utf-8"))["documents"]

    def _card(self, doc_id: str, card_id: str) -> tuple[dict, dict]:
        doc = self.data[doc_id]
        card = next(card for card in doc["capsules"] if card["id"] == card_id)
        return card, doc["identity"]

    def test_surrender_fee_table_is_source_aligned(self) -> None:
        card, identity = self._card("2", "2:p9:0036")
        schedule = parse_surrender_fee_schedule(card, identity)
        self.assertIsNotNone(schedule)
        self.assertEqual(dict(schedule.exact_rates), {
            1: 4, 2: 3, 3: 2, 4: 1, 5: 1,
        })
        self.assertEqual(schedule.onward, (6, 0))
        self.assertEqual(str(schedule.cash_value(3, 90)), "88.2")
        self.assertEqual(schedule.cash_value(6, 90), 90)

    def test_surrender_fee_overlap_fails_closed(self) -> None:
        card, identity = self._card("2", "2:p9:0036")
        bad = dict(card)
        bad["verbatim"] = bad["verbatim"].replace("第六年及以后", "第五年及以后")
        self.assertIsNone(parse_surrender_fee_schedule(bad, identity))

    def test_age_death_bands_are_continuous_and_source_bound(self) -> None:
        card, identity = self._card("2", "2:p3:0005")
        schedule = parse_age_death_schedule(card, identity)
        self.assertIsNotNone(schedule)
        self.assertEqual(
            [schedule.rate_for_age(age) for age in (17, 18, 40, 41, 60, 61)],
            [100, 160, 160, 140, 140, 120],
        )
        self.assertEqual(schedule.death_benefit(40, 90, 100), 144)
        self.assertEqual(schedule.card_id, "2:p3:0005")

    def test_age_death_gap_fails_closed(self) -> None:
        card, identity = self._card("2", "2:p3:0005")
        bad = dict(card)
        bad["verbatim"] = bad["verbatim"].replace(
            "年满41周岁的年生效对应日起至年满61周岁",
            "年满42周岁的年生效对应日起至年满61周岁",
        )
        self.assertIsNone(parse_age_death_schedule(bad, identity))

    def test_post_annuity_and_net_premium_rules_come_from_verbatim(self) -> None:
        card, identity = self._card("1", "1:p4:0007")
        after_start = parse_post_annuity_death_rule(card, identity)
        self.assertIsNotNone(after_start)
        self.assertEqual(after_start.death_benefit(150, 160), 0)
        self.assertEqual(after_start.death_benefit(150, 90), 60)

        for doc_id, card_id in (("15", "15:p3:0005"),
                                ("16", "16:p5:0011")):
            with self.subTest(doc_id=doc_id):
                card, identity = self._card(doc_id, card_id)
                rule = parse_net_premium_death_rule(card, identity)
                self.assertIsNotNone(rule)
                self.assertEqual(rule.death_benefit(100, 35, 72), 72)
                self.assertEqual(rule.doc_id, doc_id)


class InsuranceAggregateSolverTests(unittest.TestCase):
    @staticmethod
    def _q(question: str, docs: list[str], qid: str = "arbitrary") -> dict:
        return {"qid": qid, "domain": "insurance", "question": question,
                "options": {}, "doc_ids": docs}

    def test_four_product_surrender_uses_two_tables_and_direct_cash_operands(self) -> None:
        got = try_solve_insurance(self._q(
            SURRENDER_FOUR, ["16", "2", "1", "15", "10"], "new_case"))
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("333.20",))
        sources = {fact.source.split(";", 1)[0] for fact in got.facts}
        self.assertEqual(sources, {"doc=1", "doc=2", "doc=15", "doc=16"})
        self.assertIn("2:p9:0036", got.reasoning)
        self.assertIn("题干给定现金价值作为退还金额", got.reasoning)

    def test_later_year_surrender_selects_other_source_rows(self) -> None:
        got = try_solve_insurance(self._q(
            SURRENDER_FOUR_LATER, ["1", "2", "15", "16"]))
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("211.50",))
        self.assertIn("第12保单年度", got.reasoning)
        self.assertIn("第5保单年度", got.reasoning)

    def test_four_product_death_uses_all_three_clause_shapes(self) -> None:
        got = try_solve_insurance(self._q(
            DEATH_FOUR, ["15", "1", "16", "2"], "not_a_competition_id"))
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("366.00",))
        self.assertEqual(len(got.facts), 4)
        self.assertIn("2:p3:0005", got.reasoning)
        self.assertIn("1:p4:0007", got.reasoning)
        self.assertIn("15:p3:0005", got.reasoning)
        self.assertIn("16:p5:0011", got.reasoning)
        self.assertIn("40周岁满足18≤40<41", got.reasoning)
        self.assertIn("半开年龄档[18,41)", got.reasoning)
        self.assertIn("40属于[0,41)", got.reasoning)
        self.assertIn("身故给付比例160%", got.reasoning)

    def test_two_post_annuity_scenarios_and_changed_operands(self) -> None:
        got = try_solve_insurance(self._q(POST_ANNUITY_TWO, ["10", "1"]))
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("60.00",))
        self.assertEqual([fact.period for fact in got.facts], ["情形1", "情形2"])

        changed = (POST_ANNUITY_TWO.replace("150万元", "200万元")
                   .replace("160万元", "220万元")
                   .replace("90万元", "100万元"))
        changed_result = try_solve_insurance(self._q(changed, ["1"], "changed"))
        self.assertIsNotNone(changed_result)
        self.assertEqual(changed_result.slots, ("100.00",))

    def test_qid_is_irrelevant_but_source_and_units_are_mandatory(self) -> None:
        answers = []
        for qid in ("x", "y", "ins_b_999_does_not_exist"):
            got = try_solve_insurance(self._q(POST_ANNUITY_TWO, ["1"], qid))
            self.assertIsNotNone(got)
            answers.append(got.raw_answer)
        self.assertEqual(answers, ["60.00"] * 3)

        self.assertIsNone(try_solve_insurance(
            self._q(POST_ANNUITY_TWO, ["10"], "missing_source")))
        self.assertIsNone(try_solve_insurance(
            self._q(SURRENDER_FOUR, ["1", "2", "15"], "one_source_missing")))
        broken_unit = POST_ANNUITY_TWO.replace("160万元", "160美元")
        self.assertIsNone(try_solve_insurance(
            self._q(broken_unit, ["1"], "unclosed_unit")))

    def test_semantic_verifier_budgets_ignore_qid(self) -> None:
        cases = (
            (QUESTION, ["1", "10"], "insurance_single_closed",
             (2800, 480, 1000)),
            (POST_ANNUITY_TWO, ["1"], "insurance_single_closed",
             (2800, 480, 1000)),
            (SURRENDER_FOUR, ["1", "2", "15", "16"],
             "insurance_multi_surrender", (5200, 900, 1800)),
            (DEATH_FOUR, ["1", "2", "15", "16"],
             "insurance_multi_age_death", (7200, 1500, 2400)),
        )
        for question, docs, profile, expected in cases:
            with self.subTest(profile=profile):
                budgets = []
                for qid in ("first_id", "unrelated_replay_id"):
                    q = self._q(question, docs, qid)
                    solved = try_solve_insurance(q)
                    self.assertIsNotNone(solved)
                    budgets.append(deterministic_verifier_budget(q, solved))
                self.assertEqual(budgets[0], budgets[1])
                self.assertEqual(budgets[0].profile, profile)
                self.assertEqual(
                    (budgets[0].evidence_chars,
                     budgets[0].thinking_budget,
                     budgets[0].max_tokens),
                    expected,
                )
                self.assertNotIn("qid", budgets[0].audit_dict())

    def test_budget_profile_is_used_and_written_to_audit_log(self) -> None:
        q = self._q(DEATH_FOUR, ["1", "2", "15", "16"], "opaque_id")
        seen = []

        def fake_chat(_messages, **kwargs):
            seen.append(kwargs)
            return ("【取数】四项均已核对。\n【计算】四项相加为366.00。\n"
                    "【核验】年龄档与单位一致。\n答案: 366.00", 0, {})

        log = io.StringIO()
        with mock.patch.dict(os.environ, {"AFAC_DET_CALC": "1"}, clear=True), \
                mock.patch.object(calc_module, "calc_evidence",
                                  return_value=("逐条来源证据", ["source#1"])), \
                mock.patch.object(calc_module, "chat", side_effect=fake_chat):
            answer, info = calc_module.answer_calc(
                q, ["number"], log=log, return_info=True)

        self.assertEqual(answer, "366.00")
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["thinking_budget"], 1500)
        self.assertEqual(seen[0]["max_tokens"], 2400)
        profile = info["deterministic_budget_profile"]
        self.assertEqual(profile["profile"], "insurance_multi_age_death")
        self.assertEqual(profile["evidence_chars"], 7200)
        record = json.loads(log.getvalue())
        self.assertEqual(record["deterministic_budget_profile"], profile)
        self.assertIn("半开年龄档[18,41)",
                      record["deterministic_reasoning"])

    def test_verifier_disagreement_falls_back_to_full_calc_path(self) -> None:
        q = self._q(DEATH_FOUR, ["1", "2", "15", "16"], "another_id")
        outputs = iter((
            "【取数】年龄档有分歧。\n【计算】误算。\n【核验】不同意工具。\n答案: 348.00",
            "【取数】重新逐项核对原文。\n【计算】四项合计366.00。\n"
            "【核验】40周岁属于[18,41)，比例160%。\n答案: 366.00",
        ))
        tags = []

        def fake_chat(_messages, **kwargs):
            tags.append(kwargs["tag"])
            return next(outputs), 0, {}

        log = io.StringIO()
        env = {"AFAC_DET_CALC": "1", "AFAC_CALC_SINGLE": "1"}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(calc_module, "calc_evidence",
                                  return_value=("完整原始证据", ["source#1"])), \
                mock.patch.object(calc_module, "chat", side_effect=fake_chat):
            answer, info = calc_module.answer_calc(
                q, ["number"], log=log, return_info=True)

        self.assertEqual(answer, "366.00")
        self.assertEqual(tags, ["calc_det", "calc1"])
        self.assertEqual(info["reasoning_stage"], "calc1")
        self.assertEqual(
            info["deterministic_budget_profile"]["profile"],
            "insurance_multi_age_death",
        )
        self.assertEqual(
            json.loads(log.getvalue())["reasoning_stage"], "calc1")


if __name__ == "__main__":
    unittest.main()
