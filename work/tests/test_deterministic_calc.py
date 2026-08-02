"""Stdlib-only tests for the fail-closed deterministic calculation path."""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work"))

from agent.deterministic_calc import (  # noqa: E402
    BusinessCalendar,
    Fact,
    FactBook,
    Intent,
    classify_intent,
    coverage_stats,
    parse_facts,
    try_solve,
)


def fact_line(**parts: object) -> str:
    return "FACT|" + "|".join(f"{key}={value}" for key, value in parts.items())


class FactParsingTests(unittest.TestCase):
    def test_structured_fact_record(self) -> None:
        text = fact_line(entity="比亚迪", period=2025, metric="营业收入",
                         scope="合并", value="777,102,000,000", unit="元",
                         source="P12")
        facts = parse_facts(text)
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].metric, "revenue")
        self.assertEqual(str(facts[0].value), "777102000000")
        self.assertEqual(facts[0].source, "P12")

    def test_loose_parser_rejects_approximation_and_numeric_soup(self) -> None:
        evidence = "2025年营业收入约为100亿元。\n营业收入数据如下：100、90、80亿元。"
        self.assertEqual(parse_facts(evidence), [])

    def test_loose_parser_accepts_one_exact_labelled_value(self) -> None:
        facts = parse_facts("【P12】比亚迪2025年营业收入为7771亿元。")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].metric, "revenue")
        self.assertEqual(facts[0].period, "2025")
        self.assertEqual(facts[0].value, 7771)

    def test_footnote_number_is_not_parsed_as_revenue(self) -> None:
        for note in (
            "注2：营运利润=营业收入-营运支出",
            "附注 2：营运利润=营业收入-营运支出",
            "（注2）营运利润=营业收入-营运支出",
        ):
            with self.subTest(note=note):
                self.assertEqual(parse_facts(note), [])

    def test_factbook_rejects_conflicting_values(self) -> None:
        book = FactBook([
            Fact("revenue", 100, "hundred_million_yuan", period="2025"),
            Fact("revenue", 101, "hundred_million_yuan", period="2025"),
        ])
        self.assertIsNone(book.unique("营业收入", period="2025"))


class DateAndQuestionOnlyTests(unittest.TestCase):
    def test_business_day_requires_complete_calendar(self) -> None:
        q = "若第60日为2026年3月27日，期满后次一工作日是哪一天？"
        self.assertIsNone(try_solve(q, "", ["date"]))
        got = try_solve(q, "", ["date"],
                        business_calendar=BusinessCalendar.weekdays_only(
                            confirmed_no_holidays=True))
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("2026年3月30日",))

    def test_business_day_holiday_is_skipped(self) -> None:
        q = "若第60日为2026年3月27日，期满后次一工作日是哪一天？"
        cal = BusinessCalendar(holidays=frozenset({dt.date(2026, 3, 30)}),
                               complete=True)
        got = try_solve(q, "", ["date"], business_calendar=cal)
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("2026年3月31日",))

    def test_calendar_offset_excludes_acceptance_day(self) -> None:
        q = ("申请在2026年4月1日被受理，若受理当日不计入，且按90日计算，"
             "最迟应在何时作出决定？")
        got = try_solve(q, "", ["date"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("2026年6月30日",))
        self.assertIn("次日2026年4月2日为第1日", got.reasoning)
        self.assertIn("相差90个自然日", got.reasoning)

    def test_advance_notice(self) -> None:
        q = "收费调整拟于2026年5月1日施行，按至少提前30个自然日持续公示，最晚应从何时开始？"
        got = try_solve(q, "", ["date"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("2026年4月1日",))

    def test_periodic_interval_ignores_background_workday(self) -> None:
        q = ("首次期满报告在到期后次一工作日已发出，后续公告按每30日一次，"
             "则第2次公告与第1次公告之间间隔多少日？")
        self.assertEqual(classify_intent(q), Intent.PERIODIC_INTERVAL)
        got = try_solve(q, "", ["number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("30.00",))

    def test_explicit_per_share(self) -> None:
        q = ("每股净资产的评估值为多少？已知股东全部权益评估值为1,298,036.10万元，"
             "注册资本为564,141.7298万元（即总股本数，单位万股）。请保留两位小数。")
        got = try_solve(q, "", ["number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("2.30",))

    def test_member_mixture(self) -> None:
        q = ("FY2026全年总GMV为100亿元，自营品占比60%，APP自营占自营总GMV比重35%。"
             "会员年均商品消费为2960元。普通用户年均消费额为会员年均消费额的70%。"
             "若会员人数为24.01万人，则普通用户人数约为多少万人？保留一位小数。")
        got = try_solve(q, "", ["number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("67.1",))


class EvidenceFormulaTests(unittest.TestCase):
    def test_mean_with_unit_conversion(self) -> None:
        q = ("计算2023年至2025年发行人归属于母公司所有者的净利润的平均值"
             "（单位：亿元，保留两位小数）为多少？")
        ev = "\n".join([
            fact_line(period=2023, metric="归属于母公司所有者的净利润", value=1000000000, unit="元"),
            fact_line(period=2024, metric="归属于母公司所有者的净利润", value=2000000000, unit="元"),
            fact_line(period=2025, metric="归属于母公司所有者的净利润", value=3000000000, unit="元"),
        ])
        got = try_solve(q, ev, ["number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("20.00",))

    def test_mean_conflict_falls_back(self) -> None:
        q = "计算2023年至2025年归属于母公司所有者的净利润平均值（单位：亿元）"
        ev = "\n".join([
            fact_line(period=2023, metric="归母净利润", value=10, unit="亿元"),
            fact_line(period=2023, metric="归母净利润", value=11, unit="亿元"),
            fact_line(period=2024, metric="归母净利润", value=20, unit="亿元"),
            fact_line(period=2025, metric="归母净利润", value=30, unit="亿元"),
        ])
        self.assertIsNone(try_solve(q, ev, ["number"]))

    def test_direct_percent_series(self) -> None:
        q = "2022-2024年及2025年，发行人矿业板块的主营业务毛利率分别是多少？"
        ev = "\n".join(
            fact_line(period=year, metric="主营业务毛利率", value=value,
                      unit="%", source=f"P{year}")
            for year, value in [(2022, "5.55"), (2023, "5.36"),
                                (2024, "7.68"), ("2025年1-6月", "7.35")]
        )
        got = try_solve(q, ev, ["percent"] * 4)
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("5.55%", "5.36%", "7.68%", "7.35%"))

    def test_direct_series_preserves_two_dates_in_one_year(self) -> None:
        q = ("标的资产的评估增值率在两次评估（基准日2023年6月30日和"
             "2023年12月31日）中分别约为多少？")
        ev = "\n".join([
            fact_line(period="2023-06-30", metric="评估增值率", value="20.10", unit="%"),
            fact_line(period="2023-12-31", metric="评估增值率", value="21.20", unit="%"),
        ])
        got = try_solve(q, ev, ["percent", "percent"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("20.10%", "21.20%"))

    def test_yoy_and_share_points(self) -> None:
        q = ("查阅2024年和2025年营业收入及分地区收入。计算2025年境外收入同比增幅，"
             "以及境外收入占营业收入比重较2024年提高的百分点，均保留两位小数。")
        ev = "\n".join([
            fact_line(period=2024, metric="境外收入", value=200, unit="亿元"),
            fact_line(period=2025, metric="境外收入", value=300, unit="亿元"),
            fact_line(period=2024, metric="营业收入", value=1000, unit="亿元"),
            fact_line(period=2025, metric="营业收入", value=1200, unit="亿元"),
        ])
        got = try_solve(q, ev, ["number", "number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("50.00", "5.00"))

    def test_margin_drops(self) -> None:
        q = ("查阅2024年和2025年的营业收入、归属于上市公司股东的净利润和"
             "经营活动产生的现金流量净额。分别计算归母净利率下降的百分点、"
             "经营现金流率下降的百分点，并计算后者比前者多下降多少个百分点。")
        ev = "\n".join([
            fact_line(period=2024, metric="营业收入", value=1000, unit="亿元"),
            fact_line(period=2024, metric="归母净利润", value=100, unit="亿元"),
            fact_line(period=2024, metric="经营现金流", value=200, unit="亿元"),
            fact_line(period=2025, metric="营业收入", value=1000, unit="亿元"),
            fact_line(period=2025, metric="归母净利润", value=80, unit="亿元"),
            fact_line(period=2025, metric="经营现金流", value=150, unit="亿元"),
        ])
        got = try_solve(q, ev, ["number", "number", "number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("2.00", "5.00", "3.00"))

    def test_cashflow_rate_ranking(self) -> None:
        q = ("根据甲公司与乙公司2025年年度报告中的营业收入和经营活动产生的现金流量净额，"
             "计算两家公司的经营现金流率，按经营现金流率从高到低排序，并计算差值。")
        ev = "\n".join([
            fact_line(entity="甲公司", period=2025, metric="营业收入", value=100, unit="亿元"),
            fact_line(entity="甲公司", period=2025, metric="经营现金流", value=20, unit="亿元"),
            fact_line(entity="乙公司", period=2025, metric="营业收入", value=200, unit="亿元"),
            fact_line(entity="乙公司", period=2025, metric="经营现金流", value=20, unit="亿元"),
        ])
        got = try_solve(q, ev, ["ranking", "number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("甲公司>乙公司", "10.00"))

    def test_ranking_tie_falls_back(self) -> None:
        q = "根据甲公司与乙公司的经营现金流率，按经营现金流率从高到低排序并计算差值。"
        ev = "\n".join([
            fact_line(entity="甲公司", metric="营业收入", value=100, unit="亿元"),
            fact_line(entity="甲公司", metric="经营现金流", value=10, unit="亿元"),
            fact_line(entity="乙公司", metric="营业收入", value=200, unit="亿元"),
            fact_line(entity="乙公司", metric="经营现金流", value=20, unit="亿元"),
        ])
        self.assertIsNone(try_solve(q, ev, ["ranking", "number"]))

    def test_implied_revenue(self) -> None:
        q = ("查阅EBITDA、披露的EBITDA率和营业收入。先按隐含营业收入=EBITDA÷披露的"
             "EBITDA率反向推算营业收入，再计算绝对相对偏差。前者以百万元计。")
        ev = "\n".join([
            fact_line(metric="EBITDA", value=2000000000, unit="元"),
            fact_line(metric="EBITDA率", value=20, unit="%"),
            fact_line(metric="营业收入", value=9800000000, unit="元"),
        ])
        got = try_solve(q, ev, ["number", "number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("10000.00", "2.04"))

    def test_implied_revenue_ignores_formula_footnote_number(self) -> None:
        q = ("查阅EBITDA、披露的EBITDA率和营业收入。先按隐含营业收入="
             "EBITDA÷披露的EBITDA率反向推算营业收入，再计算绝对相对偏差。"
             "前者以百万元计。")
        ev = "\n".join([
            fact_line(entity="中国移动", metric="EBITDA",
                      value=338931000000, unit="元", source="P22"),
            fact_line(entity="中国移动", metric="EBITDA率",
                      value="32.3", unit="%", source="P22"),
            fact_line(entity="中国移动", metric="营业收入",
                      value=1050187000000, unit="元", source="P22"),
            "注2：营运利润=营业收入-营运支出",
        ])
        got = try_solve(q, ev, ["number", "number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("1049321.98", "0.08"))

    def test_dupont(self) -> None:
        q = ("按权益乘数=1÷(1-资产负债率)计算权益乘数，再按近似资产收益率="
             "净资产收益率÷权益乘数反推近似资产收益率。")
        ev = "\n".join([
            fact_line(metric="资产负债率", value=60, unit="%"),
            fact_line(metric="加权平均净资产收益率", value=20, unit="%"),
        ])
        got = try_solve(q, ev, ["number", "number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("2.50", "8.00"))

    def test_equity_multiplier_ranking(self) -> None:
        q = "根据甲公司、乙公司的资产负债率计算权益乘数，按权益乘数从高到低排序并计算差值。"
        ev = "\n".join([
            fact_line(entity="甲公司", metric="资产负债率", value=75, unit="%"),
            fact_line(entity="乙公司", metric="资产负债率", value=50, unit="%"),
        ])
        got = try_solve(q, ev, ["ranking", "number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("甲公司>乙公司", "2.00"))

    def test_dividend_ranking(self) -> None:
        q = "查阅甲公司和乙公司的每10股全年现金分红，按金额从高到低排序并计算差额。"
        ev = "\n".join([
            fact_line(entity="甲公司", metric="每10股全年现金分红", value=10, unit="元"),
            fact_line(entity="乙公司", metric="每10股全年现金分红", value=3, unit="元"),
        ])
        got = try_solve(q, ev, ["ranking", "number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("甲公司>乙公司", "7.00"))

    def test_dividend_reconciliation(self) -> None:
        q = ("先用归母净利润×分红比例反推现金分红总额，再用每10股派息金额×"
             "基准股本÷10计算方案现金分红总额，并计算绝对差额。")
        ev = "\n".join([
            fact_line(metric="归母净利润", value=100, unit="亿元"),
            fact_line(metric="现金分红比例", value=20, unit="%"),
            fact_line(metric="每10股派息金额", value=2, unit="元"),
            fact_line(metric="基准股本", value=10000000000, unit="股"),
        ])
        got = try_solve(q, ev, ["number", "number", "number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("20.00", "20.00", "0.00"))

    def test_threshold_inclusive_boundary(self) -> None:
        q = "三笔金额分别为4999元、5000元和8000元。按单笔核实门槛，需核实几笔？"
        ev = "单笔金额达到5000元的境外汇款应当核实。"
        got = try_solve(q, ev, ["number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("2.00",))

    def test_score_adjustment_requires_signed_unique_facts(self) -> None:
        q = ("正常经营基准分为100分。证券公司因事项甲被采取警示函措施，"
             "分支机构因事项乙也被采取警示函措施，最终评价计分是多少分？")
        ev = "\n".join([
            fact_line(entity="证券公司", metric="评价扣分", value="-0.5", unit="number"),
            fact_line(entity="分支机构", metric="评价扣分", value="-0.25", unit="number"),
        ])
        got = try_solve(q, ev, ["number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("99.25",))
        bad = fact_line(entity="证券公司", metric="评价扣分", value="0.5", unit="number")
        self.assertIsNone(try_solve(q, bad, ["number"]))

    def test_demand_growth_from_per_vehicle_fact(self) -> None:
        q = ("若2026年国内新能源乘用车销量与2025年持平，但单车带电量因技术升级"
             "从2025年的水平提升至56kWh，则全年动力电池需求同比增速最接近多少？")
        ev = fact_line(period=2025, metric="单车带电量", value=50, unit="kWh")
        got = try_solve(q, ev, ["percent"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("12.00%",))

    def test_growth_and_ai_share(self) -> None:
        q = ("2026年新签订单增速放缓至30%，AI订单占比进一步提升至80%，"
             "则2026年AI新签订单约为多少亿元？")
        ev = fact_line(period=2025, metric="新签订单金额", value=25, unit="亿元")
        got = try_solve(q, ev, ["number"])
        self.assertIsNotNone(got)
        self.assertEqual(got.slots, ("26.00",))

    def test_output_kind_mismatch_falls_back(self) -> None:
        q = "收费调整拟于2026年5月1日施行，按至少提前30个自然日公示。"
        self.assertIsNone(try_solve(q, "", ["ranking"]))


class CorpusCoverageTests(unittest.TestCase):
    def test_all_calculation_questions_have_a_structural_route(self) -> None:
        question_dir = ROOT / "upload_b" / "question_b"
        questions: list[dict[str, object]] = []
        for path in sorted(question_dir.iterdir()):
            if path.suffix == ".json":
                questions.extend(json.loads(path.read_text(encoding="utf-8-sig")))
            elif path.suffix == ".jsonl":
                questions.extend(json.loads(line) for line in
                                 path.read_text(encoding="utf-8-sig").splitlines()
                                 if line.strip())
        calculations = [q for q in questions if not q.get("options")]
        stats = coverage_stats(calculations)
        self.assertEqual(stats["total"], 26)
        self.assertEqual(stats["recognized"], 26, stats)
        self.assertEqual(stats["unknown"], 0, stats)
        self.assertEqual(stats["implemented_family"], 21, stats)
        self.assertEqual(stats["fallback_only"], 5, stats)


if __name__ == "__main__":
    unittest.main()
