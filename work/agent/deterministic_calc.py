"""Strict, deterministic helpers for B-style extraction/calculation questions.

This module deliberately has no knowledge of question identifiers or answer
files.  It operates on three inputs only:

* the natural-language question;
* evidence text (preferably ``FACT`` records produced by deterministic
  extractors); and
* the required output slot kinds.

The public :func:`try_solve` function is fail-closed.  It returns ``None``
unless an intent is unambiguous, every required operand is uniquely resolved,
units are compatible, and the produced slots pass format validation.  A caller
can therefore use a non-``None`` result directly and retain the existing Qwen
path as the fallback.

The preferred evidence interchange is one record per line::

    FACT|entity=比亚迪|period=2025|metric=营业收入|scope=合并|
         value=777102000000|unit=元|source=P12

This is intentionally simple enough for the existing PDF/table/term-clause
extractors to emit without another model call.  A conservative parser for
ordinary labelled prose is also provided, but it only accepts a metric and a
single exact value in the same clause.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import decimal
import enum
import re
from collections import Counter
from typing import Iterable, Mapping, Sequence


D = decimal.Decimal
ROUND = decimal.ROUND_HALF_UP


class Intent(str, enum.Enum):
    """Question structures recognised without consulting any answer data."""

    DIRECT_SERIES = "direct_series"
    MEAN = "mean"
    PER_SHARE = "per_share"
    YOY_AND_SHARE_PP = "yoy_and_share_pp"
    MARGIN_DROP = "margin_drop"
    CASHFLOW_RATE_RANK = "cashflow_rate_rank"
    DIVIDEND_RANK = "dividend_rank"
    IMPLIED_REVENUE = "implied_revenue"
    DUPONT = "dupont"
    EQUITY_MULTIPLIER_RANK = "equity_multiplier_rank"
    DIVIDEND_RECONCILE = "dividend_reconcile"
    CONTRACT_RULE_ARITHMETIC = "contract_rule_arithmetic"
    NEXT_BUSINESS_DAY = "next_business_day"
    CALENDAR_OFFSET = "calendar_offset"
    PERIODIC_INTERVAL = "periodic_interval"
    SCORE_ADJUSTMENT = "score_adjustment"
    THRESHOLD_COUNT = "threshold_count"
    ADVANCE_NOTICE = "advance_notice"
    DEMAND_GROWTH = "demand_growth"
    GROWTH_AND_SHARE = "growth_and_share"
    MIXTURE_COUNT = "mixture_count"
    UNKNOWN = "unknown"


_FALLBACK_ONLY_INTENTS = {
    Intent.CONTRACT_RULE_ARITHMETIC,
}


@dataclasses.dataclass(frozen=True)
class Fact:
    """One exact, attributable value.

    Percentage values are stored in percentage-point form: ``68.5%`` is
    ``Decimal("68.5")`` with unit ``"percent"``.  Monetary and share units are
    converted only when a formula explicitly requires it.
    """

    metric: str
    value: D
    unit: str = "number"
    entity: str = ""
    period: str = ""
    scope: str = ""
    source: str = ""
    exact: bool = True

    def __post_init__(self) -> None:
        # Keep the public constructor ergonomic while making all downstream
        # arithmetic Decimal-only and therefore reproducible.
        if not isinstance(self.value, D):
            object.__setattr__(self, "value", D(str(self.value).replace(",", "")))


@dataclasses.dataclass(frozen=True)
class BusinessCalendar:
    """Calendar used by workday calculations.

    ``complete`` must be true before :func:`try_solve` will emit a workday
    answer.  This prevents silently treating a statutory holiday as a workday.
    A reproduction runner may build a complete calendar from an official,
    packaged calendar file.  Weekend-only behaviour is still useful for unit
    tests but has to be opted into explicitly.
    """

    holidays: frozenset[_dt.date] = frozenset()
    weekend: frozenset[int] = frozenset({5, 6})
    complete: bool = False

    @classmethod
    def weekdays_only(cls, *, confirmed_no_holidays: bool = False) -> "BusinessCalendar":
        """Build a weekend-only calendar with an explicit completeness claim."""

        return cls(complete=confirmed_no_holidays)

    def is_workday(self, day: _dt.date) -> bool:
        return day.weekday() not in self.weekend and day not in self.holidays


@dataclasses.dataclass(frozen=True)
class SolveResult:
    intent: Intent
    slots: tuple[str, ...]
    reasoning: str
    facts: tuple[Fact, ...]
    confidence: str = "strict"

    @property
    def raw_answer(self) -> str:
        return "；".join(self.slots)


_UNIT_ALIASES = {
    "": "number",
    "个": "number",
    "笔": "number",
    "项": "number",
    "日": "day",
    "天": "day",
    "%": "percent",
    "％": "percent",
    "个百分点": "percentage_point",
    "元": "yuan",
    "万元": "ten_thousand_yuan",
    "亿元": "hundred_million_yuan",
    "百万元": "million_yuan",
    "股": "share",
    "万股": "ten_thousand_share",
    "人": "person",
    "万人": "ten_thousand_person",
    "辆": "vehicle",
    "万辆": "ten_thousand_vehicle",
    "亿元/年": "hundred_million_yuan",
    "GWh": "gwh",
    "kWh": "kwh",
    "倍": "multiple",
}

_SCALE = {
    "number": D(1),
    "day": D(1),
    "yuan": D(1),
    "ten_thousand_yuan": D(10_000),
    "million_yuan": D(1_000_000),
    "hundred_million_yuan": D(100_000_000),
    "share": D(1),
    "ten_thousand_share": D(10_000),
    "person": D(1),
    "ten_thousand_person": D(10_000),
    "vehicle": D(1),
    "ten_thousand_vehicle": D(10_000),
    "gwh": D(1),
    "kwh": D(1),
    "multiple": D(1),
}


# Longest aliases must win.  The vocabulary describes metrics, not individual
# questions or expected answers, and is reusable across any documents.
_METRIC_ALIASES: Mapping[str, tuple[str, ...]] = {
    "foreign_revenue": ("境外营业收入", "境外收入", "海外收入"),
    "revenue": ("营业总收入", "营业收入", "报告营业收入"),
    "parent_net_profit": (
        "归属于上市公司股东的净利润",
        "归属于母公司所有者的净利润",
        "归属于母公司股东的净利润",
        "归母净利润",
        "归母净利",
    ),
    "operating_cash_flow": (
        "经营活动产生的现金流量净额",
        "经营现金流量净额",
        "经营现金流",
    ),
    "ebitda_rate": ("EBITDA率", "EBITDA 率"),
    "ebitda": ("EBITDA",),
    "debt_ratio": ("资产负债率",),
    "roe": ("加权平均净资产收益率", "净资产收益率", "ROE"),
    "dividend_ratio": ("现金分红占归母净利润比例", "现金分红比例", "分红比例"),
    "dividend_per_10": (
        "每10股全年现金分红",
        "每 10 股全年现金分红",
        "每10股派息金额",
        "每 10 股派息金额",
        "每10股派现",
        "每10股派",
    ),
    "base_shares": ("利润分配基准股本", "基准股本", "总股本数", "注册资本"),
    "equity_value": ("股东全部权益评估值", "股东全部权益价值", "股东权益评估值"),
    "net_asset_per_share": ("每股净资产的评估值", "每股净资产评估值", "每股净资产"),
    "gross_margin": ("主营业务毛利率", "毛利率"),
    "appreciation_rate": ("评估增值率", "增值率"),
    "battery_demand": ("动力电池需求", "电池需求", "装机量"),
    "vehicle_sales": ("新能源乘用车销量", "乘用车销量", "汽车销量", "销量"),
    "battery_per_vehicle": ("单车带电量",),
    "new_orders": ("新签订单金额", "新签订单"),
    "ai_order_share": ("AI订单占比", "AI相关订单占比", "AI订单比例"),
    "gmv": ("总GMV", "GMV"),
    "self_operated_share": ("自营品占比", "自营占比"),
    "app_self_share": ("APP自营占自营总GMV比重", "APP自营占比"),
    "member_spend": ("会员年均商品消费", "会员年均消费"),
    "ordinary_spend": ("普通用户年均消费额", "普通用户年均消费"),
    "member_count": ("会员人数",),
    "trigger_threshold": ("核实门槛", "单笔核实门槛", "门槛"),
    "score_adjustment": ("评价扣分", "扣分", "减分", "加分"),
    "contract_result": ("合同计算结果", "合同给付金额", "合同退还金额", "合同现金价值"),
}

_ALIASES_FLAT = sorted(
    ((alias.upper(), key) for key, aliases in _METRIC_ALIASES.items() for alias in aliases),
    key=lambda x: len(x[0]),
    reverse=True,
)

_APPROX = re.compile(r"约|大约|超过|超出|至少|至多|不超过|不少于|不高于|不低于")
_NUM = re.compile(r"-?[\d,]+(?:\.\d+)?")
_DATE = re.compile(r"((?:19|20)\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_SOURCE = re.compile(r"(?:\(?(P\d+)\)?|【([^】]*P\d+[^】]*)】)")


def canonical_metric(label: str) -> str:
    """Return a stable metric name, preserving unknown labels conservatively."""

    if label in _METRIC_ALIASES:
        return label
    compact = re.sub(r"[\s()（）·:：]", "", label or "").upper()
    for alias, key in _ALIASES_FLAT:
        if alias.replace(" ", "") in compact:
            return key
    return re.sub(r"\s+", "", label or "").strip("：:；;。")


def _unit(raw: str | None) -> str:
    return _UNIT_ALIASES.get((raw or "").strip(), (raw or "number").strip())


def _dec(raw: str) -> D:
    return D(raw.replace(",", "").strip())


def _norm_text(text: str) -> str:
    return re.sub(r"[\s（）()《》【】·,，。:：;；]", "", text or "").upper()


def _same_entity(a: str, b: str) -> bool:
    na, nb = _norm_text(a), _norm_text(b)
    return bool(na and nb and (na == nb or na in nb or nb in na))


def _find_metric(text: str) -> str:
    compact = re.sub(r"\s+", "", text).upper()
    for alias, key in _ALIASES_FLAT:
        if alias.replace(" ", "") in compact:
            return key
    return ""


def _parse_fact_record(line: str) -> Fact | None:
    if not line.lstrip().startswith("FACT|"):
        return None
    fields: dict[str, str] = {}
    for piece in line.strip().split("|")[1:]:
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        fields[key.strip().lower()] = value.strip()
    required = {"metric", "value"}
    if not required <= fields.keys() or not _NUM.fullmatch(fields["value"].replace(" ", "")):
        return None
    exact = fields.get("exact", "1").lower() not in {"0", "false", "no"}
    return Fact(
        metric=canonical_metric(fields["metric"]),
        value=_dec(fields["value"]),
        unit=_unit(fields.get("unit")),
        entity=fields.get("entity", ""),
        period=fields.get("period", ""),
        scope=fields.get("scope", ""),
        source=fields.get("source", ""),
        exact=exact,
    )


def parse_facts(evidence: str) -> list[Fact]:
    """Parse exact facts from deterministic records or tightly labelled prose.

    Loose prose is accepted only when one clause has a known metric, exactly
    one numeric value, an explicit equality verb, and no approximation marker.
    This intentionally leaves recall on the table in exchange for safe gating.
    """

    facts: list[Fact] = []
    for raw_line in (evidence or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rec = _parse_fact_record(line)
        if rec is not None:
            facts.append(rec)
            continue
        for clause in re.split(r"[；;。]", line):
            metric = _find_metric(clause)
            if not metric or _APPROX.search(clause):
                continue
            # Avoid interpreting years and page numbers as values.
            body = _DATE.sub("", clause)
            body = re.sub(r"(?:19|20)\d{2}年(?:1-6月)?", "", body)
            body = re.sub(r"\bP\d+\b", "", body, flags=re.I)
            # Footnote labels such as ``注2：营运利润=营业收入-营运支出``
            # are formulas, not scalar observations.  Without stripping the
            # leading note number, the loose parser can invent ``revenue=2``
            # and make an otherwise exact FACT registry appear ambiguous.
            body = re.sub(
                r"^\s*[（(]?(?:附)?注\s*\d+\s*[）)]?\s*[:：]?",
                "", body,
            )
            matches = list(re.finditer(
                r"(?<!\d)(-?[\d,]+(?:\.\d+)?)\s*"
                r"(亿元/年|百万元|万元|亿元|万股|万人|万辆|个百分点|GWh|kWh|[%％元股人辆倍日天]?)",
                body,
            ))
            if len(matches) != 1 or not re.search(r"=|为|是|达到|报为|披露", clause):
                continue
            m = matches[0]
            # A bare year that survived date removal is not a fact value.
            if not m.group(2) and re.fullmatch(r"(?:19|20)\d{2}", m.group(1).replace(",", "")):
                continue
            periods = re.findall(r"(?:19|20)\d{2}(?:年(?:1-6月)?)?", clause)
            source_match = _SOURCE.search(raw_line)
            source = ""
            if source_match:
                source = source_match.group(1) or source_match.group(2) or ""
            positions = [clause.upper().find(alias)
                         for alias, key in _ALIASES_FLAT
                         if key == metric and clause.upper().find(alias) >= 0]
            metric_pos = min(positions) if positions else 0
            before = clause[:metric_pos]
            before = re.sub(r"【[^】]*】|\[[^]]*]|(?:19|20)\d{2}年?", "", before)
            entity = before.strip(" ：:,，-—")[-30:]
            facts.append(Fact(
                metric=metric,
                value=_dec(m.group(1)),
                unit=_unit(m.group(2)),
                entity=entity,
                period=periods[-1].replace("年", "") if periods else "",
                source=source,
            ))
    return facts


class FactBook:
    """Unique-resolution facade used by all solvers."""

    def __init__(self, facts: Iterable[Fact]):
        self.facts = tuple(f for f in facts if f.exact and f.metric and f.value.is_finite())

    def find(
        self,
        metric: str,
        *,
        entity: str = "",
        period: str = "",
        scope: str = "",
    ) -> list[Fact]:
        metric = canonical_metric(metric)
        out = [f for f in self.facts if canonical_metric(f.metric) == metric]
        if entity:
            out = [f for f in out if _same_entity(f.entity, entity)]
        if period:
            np = re.sub(r"\D", "", period)
            if len(np) == 4:
                # A question may say only "2025年" while the table labels its
                # sole observation "2025年1-6月".  Prefix matching is safe here
                # because ``unique`` still rejects multiple conflicting
                # subperiod values in that year.
                out = [f for f in out if re.sub(r"\D", "", f.period).startswith(np)]
            else:
                out = [f for f in out if re.sub(r"\D", "", f.period) == np]
        if scope:
            ns = _norm_text(scope)
            out = [f for f in out if ns and ns in _norm_text(f.scope)]
        return out

    def unique(self, metric: str, **filters: str) -> Fact | None:
        vals = self.find(metric, **filters)
        if not vals:
            return None
        # Duplicate extraction of the same value/unit is corroboration, not
        # ambiguity.  Conflicting scopes or units fail closed.
        keys = {(f.value, f.unit) for f in vals}
        if len(keys) != 1:
            return None
        return vals[0]

    def entities_in_question(self, question: str, metrics: Sequence[str]) -> list[str]:
        found: list[str] = []
        for f in self.facts:
            if f.metric not in metrics or not f.entity or f.entity not in question:
                continue
            if not any(_same_entity(f.entity, x) for x in found):
                found.append(f.entity)
        return found


def classify_intent(question: str) -> Intent:
    """Classify structure using language only; never reads an identifier."""

    q = re.sub(r"\s+", "", question or "")
    # Ask-target beats incidental background wording.  Some interval questions
    # mention that the first report was sent on the next workday but ask only
    # about the cadence between later reports.
    if re.search(r"第\d+次.*第\d+次.*间隔多少日", q) and re.search(r"每\d+日", q):
        return Intent.PERIODIC_INTERVAL
    if re.search(r"期满后次一工作日|下一个工作日|次一工作日", q):
        return Intent.NEXT_BUSINESS_DAY
    if re.search(r"至少提前\d+个?自然日", q):
        return Intent.ADVANCE_NOTICE
    if "受理当日不计入" in q and re.search(r"按\d+日计算", q):
        return Intent.CALENDAR_OFFSET
    if re.search(r"需(?:核实|审核|报告).*几(?:笔|项|个)", q) and "门槛" in q:
        return Intent.THRESHOLD_COUNT
    if "基准分" in q and re.search(r"最终.*(?:计分|得分)", q):
        return Intent.SCORE_ADJUSTMENT
    if "隐含营业收入" in q and "EBITDA" in q:
        return Intent.IMPLIED_REVENUE
    if "权益乘数" in q and "近似资产收益率" in q:
        return Intent.DUPONT
    if "权益乘数" in q and "排序" in q:
        return Intent.EQUITY_MULTIPLIER_RANK
    if "经营现金流率" in q and "排序" in q:
        return Intent.CASHFLOW_RATE_RANK
    if re.search(r"每10股全年现金分红|每10股全年派现", q) and "排序" in q:
        return Intent.DIVIDEND_RANK
    if "反推现金分红总额" in q and "方案现金分红总额" in q:
        return Intent.DIVIDEND_RECONCILE
    if "归母净利率" in q and "经营现金流率" in q and "百分点" in q:
        return Intent.MARGIN_DROP
    if (("同比增幅" in q and "占营业收入比重" in q and "百分点" in q)
            or ("境外收入" in q and "营业收入" in q
                and ("百分点" in q or "相对增幅" in q))):
        return Intent.YOY_AND_SHARE_PP
    if "平均值" in q:
        return Intent.MEAN
    if "每股净资产" in q and "股东全部权益评估值" in q and "注册资本" in q:
        return Intent.PER_SHARE
    if "普通用户人数" in q and "会员人数" in q and "GMV" in q:
        return Intent.MIXTURE_COUNT
    if "AI" in q and "新签订单" in q and "增速" in q and "占比" in q:
        return Intent.GROWTH_AND_SHARE
    if "单车带电量" in q and "动力电池需求同比增速" in q:
        return Intent.DEMAND_GROWTH
    if (re.search(r"四份合同|分别持有四份合同|对应身故保险金合计", q)
            or ("合同" in q and re.search(r"现金价值|身故保险金|退还|解除", q))):
        return Intent.CONTRACT_RULE_ARITHMETIC
    if re.search(r"分别(?:是|为|约为)?多少", q):
        return Intent.DIRECT_SERIES
    return Intent.UNKNOWN


def coverage_stats(questions: Iterable[str | Mapping[str, object]]) -> dict[str, object]:
    """Return recogniser and implemented-family coverage without using answers."""

    counts: Counter[str] = Counter()
    total = 0
    for item in questions:
        text = str(item.get("question", "")) if isinstance(item, Mapping) else str(item)
        counts[classify_intent(text).value] += 1
        total += 1
    unknown = counts.get(Intent.UNKNOWN.value, 0)
    fallback_only = sum(counts.get(intent.value, 0) for intent in _FALLBACK_ONLY_INTENTS)
    return {
        "total": total,
        "recognized": total - unknown,
        "unknown": unknown,
        "implemented_family": total - unknown - fallback_only,
        "fallback_only": fallback_only,
        "by_intent": dict(sorted(counts.items())),
    }


def _periods(question: str) -> list[str]:
    years = [str(y) for y in re.findall(r"(?:19|20)\d{2}", question)]
    # ``2023年至2025年`` implies the inclusive sequence.
    m = re.search(r"((?:19|20)\d{2})\s*年至\s*((?:19|20)\d{2})年", question)
    if m:
        a, b = map(int, m.groups())
        if 0 <= b - a <= 10:
            return [str(y) for y in range(a, b + 1)]
    # Financial tables often compact the same notation to ``2022-2024年``.
    m = re.search(r"((?:19|20)\d{2})\s*[-—至]\s*((?:19|20)\d{2})年", question)
    if m:
        a, b = map(int, m.groups())
        if 0 <= b - a <= 10:
            expanded = [str(y) for y in range(a, b + 1)]
            return list(dict.fromkeys(expanded + years))
    return list(dict.fromkeys(years))


def _series_periods(question: str, count: int) -> list[str]:
    """Periods for direct extraction, preserving two dates in the same year."""

    dates = [f"{y}{int(m):02d}{int(d):02d}" for y, m, d in _DATE.findall(question)]
    if len(dates) == count:
        return dates
    # Years inside a cited document title (``《...2026年...》``) identify the
    # source, not a requested series observation.  Removing book titles keeps
    # the period axis driven by the actual ask.
    semantic_question = re.sub(r"《[^》]*》", "", question)
    return _periods(semantic_question)


def _places(question: str) -> int:
    m = re.search(r"保留([一二两三四]|\d+)位小数", question)
    if not m:
        return 2
    raw = m.group(1)
    return {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4}.get(raw, int(raw) if raw.isdigit() else 2)


def _quant(value: D, places: int) -> D:
    return value.quantize(D(1).scaleb(-places), rounding=ROUND)


def _fmt_number(value: D, places: int) -> str:
    if not value.is_finite():
        raise ValueError("non-finite value")
    return f"{_quant(value, places):.{places}f}"


def _fmt_date(day: _dt.date) -> str:
    return f"{day.year}年{day.month}月{day.day}日"


def _to_base(f: Fact) -> D | None:
    scale = _SCALE.get(f.unit)
    return f.value * scale if scale is not None else None


def _percent_fraction(f: Fact) -> D | None:
    if f.unit in {"percent", "percentage_point"}:
        return f.value / D(100)
    if f.unit == "number" and D(0) <= f.value <= D(1):
        return f.value
    return None


def _question_entities(question: str, book: FactBook, metrics: Sequence[str]) -> list[str]:
    entities = book.entities_in_question(question, metrics)
    # Longest first avoids treating a short alias and a full name as two firms.
    entities.sort(key=len, reverse=True)
    ded: list[str] = []
    for ent in entities:
        if not any(_same_entity(ent, x) for x in ded):
            ded.append(ent)
    return ded


def _result(intent: Intent, slots: Sequence[str], reasoning: str, facts: Sequence[Fact]) -> SolveResult:
    return SolveResult(intent, tuple(slots), reasoning, tuple(facts))


def _solve_dates(
    question: str,
    intent: Intent,
    kinds: Sequence[str],
    business_calendar: BusinessCalendar | None,
) -> SolveResult | None:
    dates = [_dt.date(*map(int, m.groups())) for m in _DATE.finditer(question)]
    if len(dates) != 1:
        return None
    day = dates[0]
    if intent == Intent.NEXT_BUSINESS_DAY:
        if business_calendar is None or not business_calendar.complete:
            return None
        nxt = day + _dt.timedelta(days=1)
        while not business_calendar.is_workday(nxt):
            nxt += _dt.timedelta(days=1)
        return _result(intent, [_fmt_date(nxt)], f"{_fmt_date(day)}期满，按完整工作日历取次一工作日{_fmt_date(nxt)}。", [])
    if intent == Intent.CALENDAR_OFFSET:
        ns = re.findall(r"按(\d+)日计算", question)
        if len(ns) != 1 or "受理当日不计入" not in question:
            return None
        n = int(ns[0])
        out = day + _dt.timedelta(days=n)
        first = day + _dt.timedelta(days=1)
        delta = (out - day).days
        return _result(
            intent, [_fmt_date(out)],
            f"受理当日不计入，次日{_fmt_date(first)}为第1日；"
            f"{_fmt_date(out)}与受理日{_fmt_date(day)}相差{delta}个自然日，"
            f"故{_fmt_date(out)}正是第{n}日。", [])
    if intent == Intent.ADVANCE_NOTICE:
        ns = re.findall(r"至少提前(\d+)个?自然日", question)
        if len(ns) != 1:
            return None
        n = int(ns[0])
        out = day - _dt.timedelta(days=n)
        return _result(intent, [_fmt_date(out)], f"施行日{_fmt_date(day)}向前推{n}个自然日，最晚为{_fmt_date(out)}。", [])
    return None


def _solve_periodic_interval(question: str, places: int) -> SolveResult | None:
    values = re.findall(r"每(\d+)日(?:一次)?", question)
    if len(values) != 1:
        return None
    n = D(values[0])
    return _result(Intent.PERIODIC_INTERVAL, [_fmt_number(n, places)], f"后续事项固定每{values[0]}日一次，相邻两次间隔为{values[0]}日。", [])


def _solve_per_share(question: str, places: int) -> SolveResult | None:
    eq = re.search(r"股东全部权益评估值为\s*([\d,.]+)\s*(亿元|万元|元)", question)
    cap = re.search(r"注册资本为\s*([\d,.]+)\s*(万股|万元|股|元)", question)
    if not eq or not cap:
        return None
    ef = Fact("equity_value", _dec(eq.group(1)), _unit(eq.group(2)), source="题干")
    cf = Fact("base_shares", _dec(cap.group(1)), _unit(cap.group(2)), source="题干")
    # In financial disclosures, 万元 of registered capital equals 万股 only
    # when the question explicitly equates it with total shares.
    if cf.unit == "ten_thousand_yuan":
        if not re.search(r"即总股本数|即股本总数", question):
            return None
        cf = dataclasses.replace(cf, unit="ten_thousand_share")
    eb, cb = _to_base(ef), _to_base(cf)
    if eb is None or cb is None or cb <= 0:
        return None
    value = eb / cb
    return _result(Intent.PER_SHARE, [_fmt_number(value, places)], f"每股评估值={ef.value}/{cf.value}={value}元/股。", [ef, cf])


def _solve_mean(question: str, book: FactBook, places: int) -> SolveResult | None:
    periods = _periods(question)
    metric = _find_metric(question)
    if len(periods) < 2 or not metric:
        return None
    chosen: list[Fact] = []
    for period in periods:
        f = book.unique(metric, period=period)
        if f is None:
            return None
        chosen.append(f)
    bases = [_to_base(f) for f in chosen]
    if any(v is None for v in bases):
        return None
    mean_base = sum((v for v in bases if v is not None), D(0)) / D(len(chosen))
    target = "hundred_million_yuan" if "单位：亿元" in question or "单位:亿元" in question else chosen[0].unit
    scale = _SCALE.get(target)
    if scale is None:
        return None
    out = mean_base / scale
    expr = "+".join(str(f.value) for f in chosen)
    return _result(Intent.MEAN, [_fmt_number(out, places)], f"平均值=({expr})/{len(chosen)}={out}。", chosen)


def _fact_ratio(numer: Fact, denom: Fact) -> D | None:
    n, d = _to_base(numer), _to_base(denom)
    if n is None or d is None or d == 0:
        return None
    return n / d


def _year_fact(book: FactBook, metric: str, year: str, entity: str = "") -> Fact | None:
    return book.unique(metric, period=year, entity=entity)


def _solve_yoy_share(question: str, book: FactBook, places: int) -> SolveResult | None:
    years = _periods(question)
    if len(years) != 2:
        return None
    prev, cur = sorted(years)
    foreign0 = _year_fact(book, "foreign_revenue", prev)
    foreign1 = _year_fact(book, "foreign_revenue", cur)
    revenue0 = _year_fact(book, "revenue", prev)
    revenue1 = _year_fact(book, "revenue", cur)
    chosen = [foreign0, foreign1, revenue0, revenue1]
    if any(f is None for f in chosen):
        return None
    assert all(f is not None for f in chosen)
    f0, f1, r0, r1 = chosen  # type: ignore[misc]
    b0, b1 = _to_base(f0), _to_base(f1)
    if b0 is None or b1 is None or b0 == 0:
        return None
    yoy = (b1 - b0) / abs(b0) * D(100)
    share0, share1 = _fact_ratio(f0, r0), _fact_ratio(f1, r1)
    if share0 is None or share1 is None:
        return None
    pp = (share1 - share0) * D(100)
    return _result(Intent.YOY_AND_SHARE_PP,
                   [_fmt_number(yoy, places), _fmt_number(pp, places)],
                   f"境外收入同比={(b1-b0)}/{abs(b0)}；占营收比重提高={(share1-share0)*100}个百分点。",
                   [f0, f1, r0, r1])


def _solve_margin_drop(question: str, book: FactBook, places: int) -> SolveResult | None:
    years = _periods(question)
    if len(years) != 2:
        return None
    prev, cur = sorted(years)
    fs: list[Fact] = []
    ratios: dict[tuple[str, str], D] = {}
    for year in (prev, cur):
        rev = _year_fact(book, "revenue", year)
        np = _year_fact(book, "parent_net_profit", year)
        cf = _year_fact(book, "operating_cash_flow", year)
        if not rev or not np or not cf:
            return None
        nr, cr = _fact_ratio(np, rev), _fact_ratio(cf, rev)
        if nr is None or cr is None:
            return None
        ratios[("net", year)] = nr
        ratios[("cash", year)] = cr
        fs += [rev, np, cf]
    net_drop = (ratios[("net", prev)] - ratios[("net", cur)]) * D(100)
    cash_drop = (ratios[("cash", prev)] - ratios[("cash", cur)]) * D(100)
    diff = cash_drop - net_drop
    return _result(Intent.MARGIN_DROP,
                   [_fmt_number(net_drop, places), _fmt_number(cash_drop, places), _fmt_number(diff, places)],
                   f"归母净利率下降{net_drop}个百分点，经营现金流率下降{cash_drop}个百分点，差额{diff}。", fs)


def _solve_implied_revenue(question: str, book: FactBook, places: int) -> SolveResult | None:
    ebitda = book.unique("ebitda")
    rate = book.unique("ebitda_rate")
    revenue = book.unique("revenue")
    if not ebitda or not rate or not revenue:
        return None
    rf = _percent_fraction(rate)
    eb, rv = _to_base(ebitda), _to_base(revenue)
    if rf is None or rf <= 0 or eb is None or rv is None or rv == 0:
        return None
    implied_base = eb / rf
    # The requested first slot is explicitly in million yuan for this intent.
    implied_million = implied_base / _SCALE["million_yuan"]
    deviation = abs(implied_base - rv) / abs(rv) * D(100)
    return _result(Intent.IMPLIED_REVENUE,
                   [_fmt_number(implied_million, places), _fmt_number(deviation, places)],
                   f"隐含营收=EBITDA/EBITDA率={implied_million}百万元；相对偏差={deviation}%。",
                   [ebitda, rate, revenue])


def _solve_dupont(book: FactBook, places: int) -> SolveResult | None:
    debt = book.unique("debt_ratio")
    roe = book.unique("roe")
    if not debt or not roe:
        return None
    dr, rr = _percent_fraction(debt), _percent_fraction(roe)
    if dr is None or rr is None or not (D(0) <= dr < D(1)):
        return None
    multiplier = D(1) / (D(1) - dr)
    roa_points = rr / multiplier * D(100)
    return _result(Intent.DUPONT,
                   [_fmt_number(multiplier, places), _fmt_number(roa_points, places)],
                   f"权益乘数=1/(1-{dr})={multiplier}；近似资产收益率={rr}/{multiplier}={roa_points}%。",
                   [debt, roe])


def _rank_result(
    intent: Intent,
    values: Mapping[str, D],
    places: int,
    facts: Sequence[Fact],
    label: str,
) -> SolveResult | None:
    if len(values) < 2 or len(set(values.values())) != len(values):
        return None  # a tie makes the required ordering ambiguous
    ordered = sorted(values, key=lambda e: values[e], reverse=True)
    gap = values[ordered[0]] - values[ordered[-1]]
    ranking = ">".join(ordered)
    return _result(intent, [ranking, _fmt_number(gap, places)],
                   f"{label}从高到低为{ranking}；最高与最低差额为{gap}。", facts)


def _solve_cashflow_rank(question: str, book: FactBook, places: int) -> SolveResult | None:
    entities = _question_entities(question, book, ("revenue", "operating_cash_flow"))
    if len(entities) < 2:
        return None
    values: dict[str, D] = {}
    facts: list[Fact] = []
    for ent in entities:
        rev = book.unique("revenue", entity=ent)
        cf = book.unique("operating_cash_flow", entity=ent)
        if not rev or not cf:
            return None
        ratio = _fact_ratio(cf, rev)
        if ratio is None:
            return None
        values[ent] = ratio * D(100)
        facts += [rev, cf]
    return _rank_result(Intent.CASHFLOW_RATE_RANK, values, places, facts, "经营现金流率")


def _solve_equity_rank(question: str, book: FactBook, places: int) -> SolveResult | None:
    entities = _question_entities(question, book, ("debt_ratio",))
    if len(entities) < 2:
        return None
    values: dict[str, D] = {}
    facts: list[Fact] = []
    for ent in entities:
        debt = book.unique("debt_ratio", entity=ent)
        if not debt:
            return None
        dr = _percent_fraction(debt)
        if dr is None or not (D(0) <= dr < D(1)):
            return None
        values[ent] = D(1) / (D(1) - dr)
        facts.append(debt)
    return _rank_result(Intent.EQUITY_MULTIPLIER_RANK, values, places, facts, "权益乘数")


def _solve_dividend_rank(question: str, book: FactBook, places: int) -> SolveResult | None:
    entities = _question_entities(question, book, ("dividend_per_10",))
    if len(entities) < 2:
        return None
    values: dict[str, D] = {}
    facts: list[Fact] = []
    for ent in entities:
        f = book.unique("dividend_per_10", entity=ent)
        if not f or f.unit not in {"yuan", "number"}:
            return None
        values[ent] = f.value
        facts.append(f)
    return _rank_result(Intent.DIVIDEND_RANK, values, places, facts, "每10股全年现金分红")


def _solve_dividend_reconcile(book: FactBook, places: int) -> SolveResult | None:
    profit = book.unique("parent_net_profit")
    ratio = book.unique("dividend_ratio")
    per10 = book.unique("dividend_per_10")
    shares = book.unique("base_shares")
    if not all((profit, ratio, per10, shares)):
        return None
    assert profit and ratio and per10 and shares
    pb, sb = _to_base(profit), _to_base(shares)
    rf = _percent_fraction(ratio)
    if pb is None or sb is None or rf is None or per10.unit not in {"yuan", "number"}:
        return None
    inverse = pb * rf / _SCALE["hundred_million_yuan"]
    plan = per10.value * sb / D(10) / _SCALE["hundred_million_yuan"]
    diff = abs(inverse - plan)
    return _result(Intent.DIVIDEND_RECONCILE,
                   [_fmt_number(inverse, places), _fmt_number(plan, places), _fmt_number(diff, places)],
                   f"反推总额=归母净利润×分红比例={inverse}亿元；方案总额=每10股派息×股本/10={plan}亿元；差额={diff}亿元。",
                   [profit, ratio, per10, shares])


def _solve_threshold(question: str, evidence: str, book: FactBook,
                     places: int) -> SolveResult | None:
    amounts_m = re.search(r"金额分别为([^。]+)", question)
    if not amounts_m:
        return None
    amounts = [D(x.replace(",", "")) for x in re.findall(r"([\d,.]+)\s*元", amounts_m.group(1))]
    if not amounts:
        return None
    # The trigger relation and threshold must occur in the same evidence clause.
    clauses = re.split(r"[。；;\n]", evidence)
    triggers: list[tuple[str, D, str]] = []
    for clause in clauses:
        if not re.search(r"核实|审核|审查", clause):
            continue
        m = re.search(r"(不低于|大于等于|达到|超过|高于|大于)\s*([\d,.]+)\s*元", clause)
        if m:
            op = ">=" if m.group(1) in {"不低于", "大于等于", "达到"} else ">"
            triggers.append((op, _dec(m.group(2)), clause.strip()))
    # A deterministic source registry can carry the inclusive/exclusive
    # relation explicitly in ``scope``.  This avoids forcing a second prose
    # parser to rediscover that ``5000元以上`` includes the boundary.
    for fact in book.find("trigger_threshold"):
        if fact.unit != "yuan":
            continue
        if "relation=greater_or_equal" in fact.scope:
            triggers.append((">=", fact.value, fact.source))
        elif "relation=greater" in fact.scope:
            triggers.append((">", fact.value, fact.source))
    unique = {(op, threshold) for op, threshold, _source in triggers}
    if len(unique) != 1:
        return None
    op, threshold = next(iter(unique))
    source = next(source for candidate_op, candidate_threshold, source in triggers
                  if (candidate_op, candidate_threshold) == (op, threshold))
    count = sum(1 for x in amounts if x >= threshold) if op == ">=" else sum(1 for x in amounts if x > threshold)
    f = Fact("trigger_threshold", threshold, "yuan", source=source)
    return _result(Intent.THRESHOLD_COUNT, [_fmt_number(D(count), places)],
                   f"逐笔按{op}{threshold}元核实，{len(amounts)}笔中有{count}笔达到门槛。", [f])


def _solve_score_adjustment(question: str, book: FactBook, places: int) -> SolveResult | None:
    baselines = re.findall(r"基准分为\s*([\d.]+)\s*分", question)
    if len(baselines) != 1:
        return None
    baseline = D(baselines[0])
    adjustments = [f for f in book.find("score_adjustment")
                   if f.entity and f.entity in question and f.unit == "number"]
    entities = {_norm_text(f.entity) for f in adjustments}
    if len(adjustments) < 1 or len(entities) != len(adjustments):
        return None
    # For an adverse measure the extractor must emit signed deductions.  Do
    # not silently reinterpret a positive number as negative.
    if re.search(r"警示函|处罚|扣分|减分", question) and any(f.value >= 0 for f in adjustments):
        return None
    score = baseline + sum((f.value for f in adjustments), D(0))
    if score < 0:
        return None
    expr = "+".join(str(f.value) for f in adjustments)
    return _result(Intent.SCORE_ADJUSTMENT, [_fmt_number(score, places)],
                   f"最终计分={baseline}+({expr})={score}。", adjustments)


def _solve_demand_growth(question: str, book: FactBook, places: int) -> SolveResult | None:
    new_values = re.findall(r"提升至\s*([\d.]+)\s*kWh", question, flags=re.I)
    if len(new_values) != 1 or not re.search(r"销量.*(?:持平|相同)", question):
        return None
    new_per_vehicle = D(new_values[0])
    old = book.unique("battery_per_vehicle", period="2025") or book.unique("battery_per_vehicle")
    used: list[Fact] = []
    if old and old.unit == "kwh" and old.value > 0:
        old_per_vehicle = old.value
        used.append(old)
    else:
        demand = book.unique("battery_demand", period="2025")
        sales = book.unique("vehicle_sales", period="2025")
        if not demand or not sales or demand.unit != "gwh":
            return None
        sales_count = _to_base(sales)
        if sales_count is None or sales_count <= 0:
            return None
        old_per_vehicle = demand.value * D(1_000_000) / sales_count
        used += [demand, sales]
    if old_per_vehicle <= 0:
        return None
    growth = (new_per_vehicle / old_per_vehicle - D(1)) * D(100)
    return _result(Intent.DEMAND_GROWTH, [_fmt_number(growth, places) + "%"],
                   f"销量持平时需求增速=({new_per_vehicle}/{old_per_vehicle}-1)×100%={growth}%。", used)


def _solve_direct_series(question: str, book: FactBook, kinds: Sequence[str], places: int) -> SolveResult | None:
    metric = _find_metric(question)
    periods = _series_periods(question, len(kinds))
    if not metric or not periods or len(kinds) != len(periods):
        return None
    facts: list[Fact] = []
    slots: list[str] = []
    for year, kind in zip(periods, kinds):
        f = book.unique(metric, period=year)
        if not f:
            return None
        facts.append(f)
        if kind == "percent":
            if f.unit not in {"percent", "percentage_point"}:
                return None
            slots.append(_fmt_number(f.value, places) + "%")
        else:
            slots.append(_fmt_number(f.value, places))
    return _result(Intent.DIRECT_SERIES, slots,
                   "按题目期间顺序直接提取：" + "；".join(slots) + "。", facts)


def _solve_growth_share(question: str, book: FactBook, places: int) -> SolveResult | None:
    base = book.unique("new_orders", period="2025") or book.unique("new_orders")
    if not base:
        return None
    growths = re.findall(r"增速(?:放缓)?至\s*([\d.]+)%", question)
    shares = re.findall(r"AI订单占比(?:进一步)?提升至\s*([\d.]+)%", question)
    if len(growths) != 1 or len(shares) != 1:
        return None
    bb = _to_base(base)
    if bb is None:
        return None
    result_base = bb * (D(1) + D(growths[0]) / D(100)) * D(shares[0]) / D(100)
    target_scale = _SCALE["hundred_million_yuan"] if "亿元" in question else _SCALE.get(base.unit)
    if target_scale is None:
        return None
    out = result_base / target_scale
    return _result(Intent.GROWTH_AND_SHARE, [_fmt_number(out, places)],
                   f"AI新签订单={base.value}×(1+{growths[0]}%)×{shares[0]}%={out}。", [base])


def _solve_mixture(question: str, places: int) -> SolveResult | None:
    def one(pattern: str) -> D | None:
        vals = re.findall(pattern, question)
        return D(vals[0].replace(",", "")) if len(vals) == 1 else None

    total = one(r"总GMV为\s*([\d.]+)亿元")
    self_share = one(r"自营品占比\s*([\d.]+)%")
    app_share = one(r"APP自营占自营总GMV比重\s*([\d.]+)%")
    member_spend = one(r"会员年均商品消费为\s*([\d,.]+)元")
    ordinary_ratio = one(r"普通用户年均消费额为会员年均消费额的\s*([\d.]+)%")
    members = one(r"会员人数为\s*([\d.]+)万人")
    if any(v is None for v in (total, self_share, app_share, member_spend, ordinary_ratio, members)):
        return None
    assert total is not None and self_share is not None and app_share is not None
    assert member_spend is not None and ordinary_ratio is not None and members is not None
    app_gmv_yuan = total * D(100_000_000) * self_share / D(100) * app_share / D(100)
    member_gmv = members * D(10_000) * member_spend
    ordinary_spend = member_spend * ordinary_ratio / D(100)
    if ordinary_spend <= 0 or app_gmv_yuan < member_gmv:
        return None
    ordinary_people_wan = (app_gmv_yuan - member_gmv) / ordinary_spend / D(10_000)
    return _result(Intent.MIXTURE_COUNT, [_fmt_number(ordinary_people_wan, places)],
                   f"APP自营GMV={app_gmv_yuan}元，扣除会员消费{member_gmv}元后除以普通用户年均消费{ordinary_spend}元，得{ordinary_people_wan}万人。", [])


_EXPECTED_SLOTS: Mapping[Intent, tuple[str, ...] | None] = {
    Intent.NEXT_BUSINESS_DAY: ("date",),
    Intent.CALENDAR_OFFSET: ("date",),
    Intent.ADVANCE_NOTICE: ("date",),
    Intent.PERIODIC_INTERVAL: ("number",),
    Intent.PER_SHARE: ("number",),
    Intent.MEAN: ("number",),
    Intent.YOY_AND_SHARE_PP: ("number", "number"),
    Intent.MARGIN_DROP: ("number", "number", "number"),
    Intent.CASHFLOW_RATE_RANK: ("ranking", "number"),
    Intent.DIVIDEND_RANK: ("ranking", "number"),
    Intent.IMPLIED_REVENUE: ("number", "number"),
    Intent.DUPONT: ("number", "number"),
    Intent.EQUITY_MULTIPLIER_RANK: ("ranking", "number"),
    Intent.DIVIDEND_RECONCILE: ("number", "number", "number"),
    Intent.THRESHOLD_COUNT: ("number",),
    Intent.SCORE_ADJUSTMENT: ("number",),
    Intent.DEMAND_GROWTH: ("percent",),
    Intent.GROWTH_AND_SHARE: ("number",),
    Intent.MIXTURE_COUNT: ("number",),
    Intent.DIRECT_SERIES: None,
}


def _kinds_compatible(intent: Intent, kinds: Sequence[str]) -> bool:
    expected = _EXPECTED_SLOTS.get(intent)
    if expected is None:
        return intent == Intent.DIRECT_SERIES
    # The official template sometimes represents a date with a number slot;
    # date conversion remains the caller's schema responsibility.
    if intent in {Intent.NEXT_BUSINESS_DAY, Intent.CALENDAR_OFFSET, Intent.ADVANCE_NOTICE}:
        return tuple(kinds) in {("date",), ("number",)}
    return tuple(kinds) == expected


def _slots_valid(slots: Sequence[str], kinds: Sequence[str], intent: Intent) -> bool:
    if len(slots) != len(kinds):
        return False
    for value, kind in zip(slots, kinds):
        if kind == "ranking" and not re.fullmatch(r"[^>；;]+(?:>[^>；;]+)+", value):
            return False
        if kind == "percent" and not re.fullmatch(r"-?\d+(?:\.\d+)?[%％]", value):
            return False
        if kind == "number" and intent not in {
            Intent.NEXT_BUSINESS_DAY, Intent.CALENDAR_OFFSET, Intent.ADVANCE_NOTICE
        } and not re.fullmatch(r"-?\d+(?:\.\d+)?", value):
            return False
        if kind == "date" and not _DATE.fullmatch(value):
            return False
        if not value:
            return False
    return True


def try_solve(
    question: str,
    evidence: str,
    kinds: Sequence[str],
    *,
    facts: Iterable[Fact] = (),
    business_calendar: BusinessCalendar | None = None,
) -> SolveResult | None:
    """Attempt a deterministic solution and return ``None`` on uncertainty.

    This is the sole integration entry point.  It catches arithmetic/parser
    failures because a deterministic fast path must never take down the Qwen
    fallback.
    """

    try:
        intent = classify_intent(question)
        if intent == Intent.UNKNOWN or not _kinds_compatible(intent, kinds):
            return None
        parsed = parse_facts(evidence)
        book = FactBook([*parsed, *facts])
        places = _places(question)
        result: SolveResult | None = None
        if intent in {Intent.NEXT_BUSINESS_DAY, Intent.CALENDAR_OFFSET, Intent.ADVANCE_NOTICE}:
            result = _solve_dates(question, intent, kinds, business_calendar)
        elif intent == Intent.PERIODIC_INTERVAL:
            result = _solve_periodic_interval(question, places)
        elif intent == Intent.PER_SHARE:
            result = _solve_per_share(question, places)
        elif intent == Intent.MEAN:
            result = _solve_mean(question, book, places)
        elif intent == Intent.YOY_AND_SHARE_PP:
            result = _solve_yoy_share(question, book, places)
        elif intent == Intent.MARGIN_DROP:
            result = _solve_margin_drop(question, book, places)
        elif intent == Intent.CASHFLOW_RATE_RANK:
            result = _solve_cashflow_rank(question, book, places)
        elif intent == Intent.DIVIDEND_RANK:
            result = _solve_dividend_rank(question, book, places)
        elif intent == Intent.IMPLIED_REVENUE:
            result = _solve_implied_revenue(question, book, places)
        elif intent == Intent.DUPONT:
            result = _solve_dupont(book, places)
        elif intent == Intent.EQUITY_MULTIPLIER_RANK:
            result = _solve_equity_rank(question, book, places)
        elif intent == Intent.DIVIDEND_RECONCILE:
            result = _solve_dividend_reconcile(book, places)
        elif intent == Intent.THRESHOLD_COUNT:
            result = _solve_threshold(question, evidence, book, places)
        elif intent == Intent.SCORE_ADJUSTMENT:
            result = _solve_score_adjustment(question, book, places)
        elif intent == Intent.DEMAND_GROWTH:
            result = _solve_demand_growth(question, book, places)
        elif intent == Intent.DIRECT_SERIES:
            result = _solve_direct_series(question, book, kinds, places)
        elif intent == Intent.GROWTH_AND_SHARE:
            result = _solve_growth_share(question, book, places)
        elif intent == Intent.MIXTURE_COUNT:
            result = _solve_mixture(question, places)
        # Contract-rule arithmetic is recognised for routing but intentionally
        # remains a Qwen fallback until the clause extractor can emit uniquely
        # bound per-contract formula records.
        if result is None or not _slots_valid(result.slots, kinds, intent):
            return None
        return result
    except (ArithmeticError, ValueError, TypeError, OverflowError, decimal.InvalidOperation):
        return None


__all__ = [
    "BusinessCalendar",
    "Fact",
    "FactBook",
    "Intent",
    "SolveResult",
    "canonical_metric",
    "classify_intent",
    "coverage_stats",
    "parse_facts",
    "try_solve",
]
