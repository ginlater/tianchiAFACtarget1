"""Offline regression tests for the financial-report fact registry.

Expected arithmetic below is recomputed from independently transcribed report
values.  This test never imports a label file, submission, answer artifact, or
model-generated solution.  The only historical run artifact it reads is the
document-selection log, and only its selected document identifiers are passed
to the registry.
"""

from __future__ import annotations

import decimal
import json
import re
import sys
import unittest
from pathlib import Path

D = decimal.Decimal
ROUND = decimal.ROUND_HALF_UP
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work"))

from agent.deterministic_calc import (  # noqa: E402
    Intent, classify_intent, parse_facts, try_solve,
)
from agent.financial_fact_registry import FinancialFactRegistry  # noqa: E402

KINDS = {
    Intent.YOY_AND_SHARE_PP: ("number", "number"),
    Intent.MARGIN_DROP: ("number", "number", "number"),
    Intent.CASHFLOW_RATE_RANK: ("ranking", "number"),
    Intent.DIVIDEND_RANK: ("ranking", "number"),
    Intent.IMPLIED_REVENUE: ("number", "number"),
    Intent.DUPONT: ("number", "number"),
    Intent.EQUITY_MULTIPLIER_RANK: ("ranking", "number"),
    Intent.DIVIDEND_RECONCILE: ("number", "number", "number"),
}


def fmt(value: D) -> str:
    return format(value.quantize(D("0.01"), rounding=ROUND), ".2f")


class FinancialFactRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        question_path = ROOT / "upload_b/question_b/financial_reports_b_questions.jsonl"
        questions = []
        with question_path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                item = json.loads(line)
                if item.get("type") == "计算题" and classify_intent(item["question"]) in KINDS:
                    questions.append(item)

        # This is used only as an input-document fixture.  Answers, reasoning,
        # evidence selections, and score-derived artifacts are never read.
        selection_path = ROOT / "work/output/b_honest_full4/docsel_log.jsonl"
        selected = {}
        with selection_path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                selected[record["qid"]] = tuple(record["picked"])

        cls.registry = FinancialFactRegistry(ROOT / "work/processed_data")
        cls.cases = {}
        cls.case_items = {}
        for item in questions:
            intent = classify_intent(item["question"])
            # Deliberately omit the identifier from the registry input.
            request = {
                "domain": item["domain"],
                "question": item["question"],
                "doc_ids": selected[item["qid"]],
            }
            cls.case_items[intent] = request
            cls.cases[intent] = cls.registry.extract(request)

    def solve(self, intent: Intent):
        request = self.case_items[intent]
        result = self.cases[intent]
        return try_solve(
            request["question"], "", KINDS[intent], facts=result.facts
        )

    def fact(self, intent: Intent, metric: str, entity: str, period: str) -> D:
        matches = [
            fact.value for fact in self.cases[intent].facts
            if (fact.metric, fact.entity, fact.period) == (metric, entity, period)
        ]
        self.assertEqual(len(matches), 1, (intent, metric, entity, period, matches))
        return matches[0]

    def test_all_eight_historical_doc_selections_solve_strictly(self) -> None:
        self.assertEqual(len(self.cases), 8)
        solved = 0
        for intent, result in self.cases.items():
            self.assertTrue(result.complete, (intent, result.missing, result.conflicts))
            answer = self.solve(intent)
            self.assertIsNotNone(answer, intent)
            solved += answer is not None
        self.assertEqual(solved, 8)

    def test_transcribed_source_values_and_units(self) -> None:
        expected = {
            Intent.YOY_AND_SHARE_PP: {
                ("foreign_revenue", "比亚迪", "2024"): D("221884773000"),
                ("foreign_revenue", "比亚迪", "2025"): D("310740988000"),
                ("revenue", "比亚迪", "2024"): D("777102455000"),
                ("revenue", "比亚迪", "2025"): D("803964958000"),
            },
            Intent.MARGIN_DROP: {
                ("parent_net_profit", "比亚迪", "2024"): D("40254346000"),
                ("parent_net_profit", "比亚迪", "2025"): D("32619022000"),
                ("operating_cash_flow", "比亚迪", "2024"): D("133453873000"),
                ("operating_cash_flow", "比亚迪", "2025"): D("59135544000"),
            },
            Intent.CASHFLOW_RATE_RANK: {
                ("revenue", "宁德时代", "2025"): D("423701834000"),
                ("operating_cash_flow", "宁德时代", "2025"): D("133219982000"),
                ("revenue", "美的集团", "2025"): D("456451731000"),
                ("operating_cash_flow", "美的集团", "2025"): D("53345930000"),
            },
            Intent.DIVIDEND_RANK: {
                ("dividend_per_10", "宁德时代", "2025"): D("79.64"),
                ("dividend_per_10", "美的集团", "2025"): D("43"),
                ("dividend_per_10", "招商银行", "2025"): D("20.16"),
                ("dividend_per_10", "中国建筑", "2025"): D("2.718"),
            },
            Intent.IMPLIED_REVENUE: {
                ("ebitda", "中国移动", "2025"): D("338931000000"),
                ("ebitda_rate", "中国移动", "2025"): D("32.3"),
                ("revenue", "中国移动", "2025"): D("1050187000000"),
            },
            Intent.DUPONT: {
                ("debt_ratio", "美的集团", "2025"): D("61.17"),
                ("roe", "美的集团", "2025"): D("19.70"),
            },
            Intent.EQUITY_MULTIPLIER_RANK: {
                ("debt_ratio", "比亚迪", "2025"): D("70.74"),
                ("debt_ratio", "宁德时代", "2025"): D("61.94"),
                ("debt_ratio", "美的集团", "2025"): D("61.17"),
            },
            Intent.DIVIDEND_RECONCILE: {
                ("parent_net_profit", "中国建筑", "2025"): D("39069002000"),
                ("dividend_ratio", "中国建筑", "2025"): D("28.75"),
                ("dividend_per_10", "中国建筑", "2025"): D("2.718"),
                ("base_shares", "中国建筑", "2025"): D("41320390444"),
            },
        }
        non_money = {"ebitda_rate", "debt_ratio", "roe", "dividend_per_10", "dividend_ratio", "base_shares"}
        for intent, values in expected.items():
            for key, value in values.items():
                self.assertEqual(self.fact(intent, *key), value)
                facts = [fact for fact in self.cases[intent].facts
                         if (fact.metric, fact.entity, fact.period) == key]
                self.assertRegex(facts[0].source, r":P\d+")
                if key[0] not in non_money:
                    self.assertEqual(facts[0].unit, "yuan")

    def test_yoy_share_formula_is_independent(self) -> None:
        foreign_2024 = D("221884773000")
        foreign_2025 = D("310740988000")
        revenue_2024 = D("777102455000")
        revenue_2025 = D("803964958000")
        yoy = (foreign_2025 - foreign_2024) / foreign_2024 * D(100)
        share_pp = (foreign_2025 / revenue_2025 - foreign_2024 / revenue_2024) * D(100)
        self.assertEqual(self.solve(Intent.YOY_AND_SHARE_PP).slots, (fmt(yoy), fmt(share_pp)))

    def test_margin_drop_formula_is_independent(self) -> None:
        revenue_2024, revenue_2025 = D("777102455000"), D("803964958000")
        profit_2024, profit_2025 = D("40254346000"), D("32619022000")
        cash_2024, cash_2025 = D("133453873000"), D("59135544000")
        net_drop = (profit_2024 / revenue_2024 - profit_2025 / revenue_2025) * D(100)
        cash_drop = (cash_2024 / revenue_2024 - cash_2025 / revenue_2025) * D(100)
        self.assertEqual(
            self.solve(Intent.MARGIN_DROP).slots,
            (fmt(net_drop), fmt(cash_drop), fmt(cash_drop - net_drop)),
        )

    def test_cashflow_and_dividend_rank_formulas_are_independent(self) -> None:
        cashflow_rates = {
            "宁德时代": D("133219982000") / D("423701834000") * D(100),
            "美的集团": D("53345930000") / D("456451731000") * D(100),
        }
        ordered = sorted(cashflow_rates, key=cashflow_rates.get, reverse=True)
        gap = cashflow_rates[ordered[0]] - cashflow_rates[ordered[-1]]
        self.assertEqual(
            self.solve(Intent.CASHFLOW_RATE_RANK).slots,
            (">".join(ordered), fmt(gap)),
        )

        dividends = {
            "宁德时代": D("69.57") + D("10.07"),
            "美的集团": D("43"),
            "招商银行": D("2.016") * D(10),
            "中国建筑": D("2.718"),
        }
        ordered = sorted(dividends, key=dividends.get, reverse=True)
        gap = dividends[ordered[0]] - dividends[ordered[-1]]
        self.assertEqual(
            self.solve(Intent.DIVIDEND_RANK).slots,
            (">".join(ordered), fmt(gap)),
        )

    def test_ebitda_and_dupont_formulas_are_independent(self) -> None:
        ebitda = D("338931000000")
        rate = D("32.3") / D(100)
        revenue = D("1050187000000")
        implied = ebitda / rate
        deviation = abs(implied - revenue) / revenue * D(100)
        self.assertEqual(
            self.solve(Intent.IMPLIED_REVENUE).slots,
            (fmt(implied / D(1_000_000)), fmt(deviation)),
        )

        debt = D("61.17") / D(100)
        roe = D("19.70") / D(100)
        multiplier = D(1) / (D(1) - debt)
        roa_points = roe / multiplier * D(100)
        self.assertEqual(
            self.solve(Intent.DUPONT).slots,
            (fmt(multiplier), fmt(roa_points)),
        )

    def test_equity_rank_formula_is_independent(self) -> None:
        debt_ratios = {
            "比亚迪": D("70.74") / D(100),
            "宁德时代": D("61.94") / D(100),
            "美的集团": D("61.17") / D(100),
        }
        multipliers = {name: D(1) / (D(1) - ratio) for name, ratio in debt_ratios.items()}
        ordered = sorted(multipliers, key=multipliers.get, reverse=True)
        gap = multipliers[ordered[0]] - multipliers[ordered[-1]]
        self.assertEqual(
            self.solve(Intent.EQUITY_MULTIPLIER_RANK).slots,
            (">".join(ordered), fmt(gap)),
        )

    def test_dividend_reconciliation_formula_is_independent(self) -> None:
        profit = D("39069002000")
        ratio = D("28.75") / D(100)
        per_ten = D("2.718")
        shares = D("41320390444")
        inverse = profit * ratio / D(100_000_000)
        plan = per_ten * shares / D(10) / D(100_000_000)
        self.assertEqual(
            self.solve(Intent.DIVIDEND_RECONCILE).slots,
            (fmt(inverse), fmt(plan), fmt(abs(inverse - plan))),
        )

    def test_fact_block_round_trips(self) -> None:
        for result in self.cases.values():
            parsed = parse_facts(result.fact_block)
            before = {(f.metric, f.entity, f.period, f.value, f.unit) for f in result.facts}
            after = {(f.metric, f.entity, f.period, f.value, f.unit) for f in parsed}
            self.assertEqual(after, before)

    def test_incomplete_selection_fails_closed(self) -> None:
        request = dict(self.case_items[Intent.DIVIDEND_RANK])
        request["doc_ids"] = tuple(
            identifier for identifier in request["doc_ids"]
            if "cscec" not in identifier
        )
        result = self.registry.extract(request)
        self.assertFalse(result.complete)
        self.assertTrue(any("中国建筑" in item for item in result.missing))
        self.assertEqual(self.registry.facts_for(request), [])
        self.assertEqual(self.registry.fact_block(request), "")

    def test_choice_claim_shape_exposes_the_same_source_facts(self) -> None:
        question_path = ROOT / "upload_b/question_b/financial_reports_b_questions.jsonl"
        choice = None
        with question_path.open(encoding="utf-8-sig") as handle:
            for line in handle:
                item = json.loads(line)
                if ("分地区营业收入" in item.get("question", "") and
                        item.get("options")):
                    choice = item
                    break
        self.assertIsNotNone(choice)
        request = dict(self.case_items[Intent.YOY_AND_SHARE_PP])
        request["question"] = (choice["question"] + " " +
                               " ".join(choice["options"].values()))
        result = self.registry.extract(request)
        self.assertTrue(result.complete, (result.missing, result.conflicts))
        self.assertEqual({fact.metric for fact in result.facts},
                         {"foreign_revenue", "revenue"})

    def test_module_has_no_identifier_or_answer_artifact_dependency(self) -> None:
        source = (ROOT / "work/agent/financial_fact_registry.py").read_text(encoding="utf-8")
        for forbidden in ("qid", "known_labels", "validation_labels", "answers.json", "b_v4"):
            self.assertNotIn(forbidden, source)
        self.assertIsNone(re.search(r"fin_[ab]_\d{3}", source))


if __name__ == "__main__":
    unittest.main()
