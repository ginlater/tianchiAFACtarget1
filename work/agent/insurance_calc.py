"""Fail-closed deterministic calculations for typed insurance clauses.

The insurance corpus contains many compact rate tables whose row/column
relationship is easy for a language model to transpose after PDF text has
been flattened.  This module treats those tables as executable memory: it
binds a question to one explicitly named product, parses the source capsule's
ordered policy-year table and range formulas, then performs Decimal arithmetic.

There is deliberately no question-id or answer-file input.  If product
identity, table alignment, ranges, units, or the requested operation are not
unique, the public solver returns ``None`` and the normal Qwen path remains in
control.
"""

from __future__ import annotations

import dataclasses
import decimal
import pathlib
import re

from .deterministic_calc import Fact, Intent, SolveResult
from .insurance_capsules import DEFAULT_PATH, load_capsules


D = decimal.Decimal
ROUND = decimal.ROUND_HALF_UP

_UNIT_SCALE = {
    "元": D("1"),
    "万元": D("10000"),
    "亿元": D("100000000"),
}
_FACT_UNIT = {
    "元": "yuan",
    "万元": "ten_thousand_yuan",
    "亿元": "hundred_million_yuan",
}


@dataclasses.dataclass(frozen=True)
class PolicyYearSchedule:
    """One source-bound cash-value schedule.

    ``exact_rates`` maps individual early policy years to a percentage of
    cumulative premiums.  ``range_rates`` maps inclusive year bands to
    ``premium + cumulative_return * rate``.  The final ``None`` upper bound
    means "that year and later".
    """

    doc_id: str
    page: int
    card_id: str
    product: str
    exact_rates: tuple[tuple[int, D], ...]
    range_rates: tuple[tuple[int, int | None, D], ...]

    def rate_for_exact_year(self, year: int) -> D | None:
        matches = [rate for y, rate in self.exact_rates if y == year]
        return matches[0] if len(matches) == 1 else None

    def return_rate_for_year(self, year: int) -> D | None:
        matches = [rate for lo, hi, rate in self.range_rates
                   if year >= lo and (hi is None or year <= hi)]
        return matches[0] if len(matches) == 1 else None

    def cash_value(self, year: int, premium: D, cumulative_return: D) -> D | None:
        exact = self.rate_for_exact_year(year)
        ranged = self.return_rate_for_year(year)
        if exact is not None and ranged is not None:
            return None
        if exact is not None:
            return premium * exact / D("100")
        if ranged is not None:
            return premium + cumulative_return * ranged / D("100")
        return None


@dataclasses.dataclass(frozen=True)
class SurrenderFeeSchedule:
    """Cash value equals account value less a source-table surrender fee."""

    doc_id: str
    page: int
    card_id: str
    product: str
    exact_rates: tuple[tuple[int, D], ...]
    onward: tuple[int, D]

    def fee_rate(self, year: int) -> D | None:
        matches = [rate for candidate, rate in self.exact_rates
                   if candidate == year]
        if year >= self.onward[0]:
            matches.append(self.onward[1])
        return matches[0] if len(matches) == 1 else None

    def cash_value(self, year: int, account_value: D) -> D | None:
        rate = self.fee_rate(year)
        if rate is None:
            return None
        return account_value * (D("100") - rate) / D("100")


@dataclasses.dataclass(frozen=True)
class AgeDeathSchedule:
    """Death benefit is max(age-rate * base sum insured, account value)."""

    doc_id: str
    page: int
    card_id: str
    product: str
    ranges: tuple[tuple[int, int | None, D], ...]

    def band_for_age(self, age: int) -> tuple[int, int | None, D] | None:
        """Return the unique matching half-open ``[lo, hi)`` source band."""

        matches = [(lo, hi, rate) for lo, hi, rate in self.ranges
                   if age >= lo and (hi is None or age < hi)]
        return matches[0] if len(matches) == 1 else None

    def rate_for_age(self, age: int) -> D | None:
        band = self.band_for_age(age)
        return band[2] if band is not None else None

    def death_benefit(self, age: int, base_sum: D,
                      account_value: D) -> D | None:
        rate = self.rate_for_age(age)
        if rate is None:
            return None
        return max(base_sum * rate / D("100"), account_value)


@dataclasses.dataclass(frozen=True)
class PostAnnuityDeathRule:
    """After annuity commencement, pay the positive unconsumed balance."""

    doc_id: str
    page: int
    card_id: str
    product: str

    def death_benefit(self, commencement_value: D,
                      produced_annuity: D) -> D:
        return max(commencement_value - produced_annuity, D("0"))


@dataclasses.dataclass(frozen=True)
class NetPremiumDeathRule:
    """Pay max(premiums less paid annuity, cash value)."""

    doc_id: str
    page: int
    card_id: str
    product: str

    def death_benefit(self, premium: D, paid_annuity: D,
                      cash_value: D) -> D:
        return max(premium - paid_annuity, cash_value)


@dataclasses.dataclass(frozen=True)
class BoundProduct:
    """One question fragment uniquely bound to one selected source identity."""

    doc_id: str
    product: str
    identity: dict
    document: dict
    segment: str


@dataclasses.dataclass(frozen=True)
class _Component:
    product: str
    value_base: D
    source: str
    detail: str
    period: str = ""


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _unique_decimal_values(raw: list[str]) -> tuple[D, ...] | None:
    try:
        values = tuple(D(value.replace(",", "")) for value in raw)
    except decimal.InvalidOperation:
        return None
    return values if all(value.is_finite() for value in values) else None


def parse_policy_year_schedule(card: dict, identity: dict) -> PolicyYearSchedule | None:
    """Parse one policy-year cash-value capsule, rejecting ambiguous tables."""

    if card.get("topic") != "cash_value":
        return None
    doc_id = str(card.get("doc_id") or "")
    card_id = str(card.get("id") or "")
    product = str((identity or {}).get("product") or "").strip()
    try:
        page = int(card.get("page"))
    except (TypeError, ValueError):
        return None
    if not doc_id or not card_id or not product or page <= 0:
        return None

    text = card.get("verbatim") or ""
    compact = _compact(text)
    if "养老保险金开始领取日前" not in compact:
        return None

    # Only zip a flattened table when its header and value vectors are both
    # explicit and exactly aligned.  The section boundary prevents rates from
    # later range formulas being mistaken for table cells.
    table_match = re.search(
        r"具体比例如下表所示[:：]?(.*?)"
        r"(?=2[\.．、]您在本合同第\d+个保单年度)",
        compact,
    )
    if not table_match:
        return None
    table = table_match.group(1)
    years = [int(x) for x in re.findall(r"第(\d+)个保单年度", table)]
    rate_tail = re.split(r"比例", table, maxsplit=1)
    if len(rate_tail) != 2:
        return None
    rates = _unique_decimal_values(re.findall(
        r"(-?\d+(?:\.\d+)?)\s*[%％]", rate_tail[1]))
    if (not years or rates is None or len(years) != len(rates) or
            years != sorted(years) or len(set(years)) != len(years)):
        return None
    exact_rates = tuple(zip(years, rates))

    # Parse the two common source formulas rather than infer them from nearby
    # numbers.  A footnote marker may appear between 累计收益 and 的.
    finite = re.search(
        r"第(\d+)个保单年度至第(\d+)个保单年度.*?"
        r"现金价值等于以下两项金额之和.*?保单账户累计收益\d*的"
        r"(-?\d+(?:\.\d+)?)%",
        compact,
    )
    onward = re.search(
        r"第(\d+)个保单年度及以后.*?"
        r"现金价值等于以下两项金额之和.*?保单账户累计收益\d*的"
        r"(-?\d+(?:\.\d+)?)%",
        compact,
    )
    if not finite or not onward:
        return None
    lo, hi = int(finite.group(1)), int(finite.group(2))
    later = int(onward.group(1))
    try:
        finite_rate = D(finite.group(3))
        onward_rate = D(onward.group(2))
    except decimal.InvalidOperation:
        return None

    # Require a continuous, non-overlapping partition.  This prevents a
    # malformed or partially captured table from silently selecting a band.
    if (lo > hi or years != list(range(1, lo)) or hi + 1 != later or
            any(rate < 0 for rate in rates) or
            finite_rate < 0 or onward_rate < 0):
        return None
    return PolicyYearSchedule(
        doc_id=doc_id,
        page=page,
        card_id=card_id,
        product=product,
        exact_rates=exact_rates,
        range_rates=((lo, hi, finite_rate), (later, None, onward_rate)),
    )


_CN_DIGITS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3,
    "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def _small_cn_int(raw: str) -> int | None:
    """Parse the small integers used in insurance policy-year tables."""

    if raw.isdigit():
        return int(raw)
    if raw in _CN_DIGITS:
        return _CN_DIGITS[raw]
    if raw == "十":
        return 10
    if "十" in raw:
        left, right = raw.split("十", 1)
        if left and left not in _CN_DIGITS:
            return None
        if right and right not in _CN_DIGITS:
            return None
        return (_CN_DIGITS.get(left, 1) * 10 + _CN_DIGITS.get(right, 0))
    return None


def _card_header(card: dict, identity: dict) -> tuple[str, int, str, str] | None:
    doc_id = str(card.get("doc_id") or "")
    card_id = str(card.get("id") or "")
    product = str((identity or {}).get("product") or "").strip()
    try:
        page = int(card.get("page"))
    except (TypeError, ValueError):
        return None
    if not doc_id or not card_id or not product or page <= 0:
        return None
    return doc_id, page, card_id, product


def parse_surrender_fee_schedule(
    card: dict, identity: dict,
) -> SurrenderFeeSchedule | None:
    """Parse a flattened year-to-surrender-fee table without guessing rows."""

    if card.get("topic") != "cash_value":
        return None
    header = _card_header(card, identity)
    if header is None:
        return None
    compact = _compact(card.get("verbatim") or "")
    marker = "现金价值等于个人账户价值扣除相应的退保费用后的余额"
    table_marker = "退保费用占个人账户价值的比例为下表中各保单年度对应的数值"
    if marker not in compact or table_marker not in compact:
        return None
    table = compact.split(table_marker, 1)[1]
    rows = re.findall(
        r"第([零〇一二两三四五六七八九十\d]+)年(及以后)?"
        r"(-?\d+(?:\.\d+)?)\s*[%％]",
        table,
    )
    if not rows:
        return None
    exact: list[tuple[int, D]] = []
    onward: list[tuple[int, D]] = []
    try:
        for raw_year, suffix, raw_rate in rows:
            year = _small_cn_int(raw_year)
            rate = D(raw_rate)
            if year is None or year < 1 or rate < 0 or rate > 100:
                return None
            (onward if suffix else exact).append((year, rate))
    except decimal.InvalidOperation:
        return None
    if len(onward) != 1 or not exact:
        return None
    exact.sort()
    years = [year for year, _rate in exact]
    onward_year, onward_rate = onward[0]
    # A missing, duplicate or overlapping row must not be repaired by
    # proximity: the table itself has to define one continuous partition.
    if (len(set(years)) != len(years) or years != list(range(1, onward_year)) or
            onward_year <= years[-1]):
        return None
    doc_id, page, card_id, product = header
    return SurrenderFeeSchedule(
        doc_id, page, card_id, product, tuple(exact),
        (onward_year, onward_rate),
    )


def parse_age_death_schedule(card: dict, identity: dict) -> AgeDeathSchedule | None:
    """Parse age bands and the max(rate*base, account-value) death rule."""

    header = _card_header(card, identity)
    if header is None:
        return None
    compact = _compact(card.get("verbatim") or "")
    if not all(marker in compact for marker in (
        "身故保险金额为下列两者的较大值",
        "身故给付比例与基本保险金额的乘积",
        "个人账户价值",
    )):
        return None
    first = re.search(
        r"本合同生效之日起至年满(\d+)周岁的年生效对应日前身故.*?"
        r"身故给付比例为(-?\d+(?:\.\d+)?)%",
        compact,
    )
    middles = re.findall(
        r"年满(\d+)周岁的年生效对应日起至年满(\d+)周岁的年生效对应日前身故.*?"
        r"身故给付比例为(-?\d+(?:\.\d+)?)%",
        compact,
    )
    last = re.search(
        r"年满(\d+)周岁的年生效对应日起身故.*?"
        r"身故给付比例为(-?\d+(?:\.\d+)?)%",
        compact,
    )
    if not first or not middles or not last:
        return None
    try:
        ranges: list[tuple[int, int | None, D]] = [
            (0, int(first.group(1)), D(first.group(2))),
            *((int(lo), int(hi), D(rate)) for lo, hi, rate in middles),
            (int(last.group(1)), None, D(last.group(2))),
        ]
    except decimal.InvalidOperation:
        return None
    if any(lo < 0 or (hi is not None and lo >= hi) or rate < 0 or rate > 1000
           for lo, hi, rate in ranges):
        return None
    for left, right in zip(ranges, ranges[1:]):
        if left[1] != right[0]:
            return None
    doc_id, page, card_id, product = header
    return AgeDeathSchedule(doc_id, page, card_id, product, tuple(ranges))


def parse_post_annuity_death_rule(
    card: dict, identity: dict,
) -> PostAnnuityDeathRule | None:
    """Recognise the two exhaustive after-commencement death branches."""

    if card.get("topic") != "death_benefit":
        return None
    header = _card_header(card, identity)
    if header is None:
        return None
    compact = _compact(card.get("verbatim") or "")
    required = (
        "养老保险金开始领取日及之后身故",
        "已产生的养老保险金之和小于养老保险金开始领取日的保单账户价值",
        "养老保险金开始领取日的保单账户价值减去累计已产生的养老保险金的差额",
        "已产生的养老保险金之和大于或等于养老保险金开始领取日的保单账户价值",
        "身故保险金为零",
    )
    if not all(marker in compact for marker in required):
        return None
    return PostAnnuityDeathRule(*header)


def parse_net_premium_death_rule(
    card: dict, identity: dict,
) -> NetPremiumDeathRule | None:
    """Recognise max(premium-paid annuity, cash value) from either syntax."""

    header = _card_header(card, identity)
    if header is None:
        return None
    compact = _compact(card.get("verbatim") or "")
    if "身故保险金" not in compact or "现金价值" not in compact or \
            "较大" not in compact:
        return None
    enumerated = (
        re.search(
            r"所交保险费(?:（不计利息）)?减去本合同累计已给付的养老年金",
            compact,
        ) is not None and
        re.search(r"(?:2[.．、])?本合同现金价值", compact) is not None
    )
    inline = re.search(
        r"累计已交保险费扣除累计已给付养老保险金后的余额与本合同的现金价值的较大者",
        compact,
    ) is not None
    if not (enumerated or inline):
        return None
    return NetPremiumDeathRule(*header)


def _identity_score(question: str, identity: dict) -> tuple[int, str]:
    compact_q = _compact(question)
    labels = [identity.get("product") or "", *(identity.get("aliases") or [])]
    matches = [label for label in labels
               if len(_compact(label)) >= 3 and _compact(label) in compact_q]
    if not matches:
        return 0, ""
    best = max(matches, key=lambda value: (len(_compact(value)), value))
    return len(_compact(best)), best


def _select_schedule(q: dict, path: str | pathlib.Path | None) -> PolicyYearSchedule | None:
    data = load_capsules(path or DEFAULT_PATH)
    ranked: list[tuple[int, str, dict, dict]] = []
    for raw_doc_id in q.get("doc_ids") or []:
        doc_id = str(raw_doc_id)
        doc = data.get("documents", {}).get(doc_id)
        if not doc:
            continue
        identity = doc.get("identity") or {}
        score, _matched = _identity_score(q.get("question") or "", identity)
        if score:
            ranked.append((score, doc_id, doc, identity))
    if not ranked:
        return None
    best_score = max(row[0] for row in ranked)
    best_docs = [row for row in ranked if row[0] == best_score]
    if len(best_docs) != 1:
        return None
    _score, _doc_id, doc, identity = best_docs[0]
    schedules = [schedule for card in doc.get("capsules") or []
                 if (schedule := parse_policy_year_schedule(card, identity)) is not None]
    return schedules[0] if len(schedules) == 1 else None


def _identity_labels(identity: dict) -> tuple[str, ...]:
    values = [identity.get("product") or "", *(identity.get("aliases") or [])]
    labels = {_compact(str(value)) for value in values
              if len(_compact(str(value))) >= 3}
    return tuple(sorted(labels, key=lambda value: (-len(value), value)))


def _bind_product_segments(
    q: dict, path: str | pathlib.Path | None,
) -> list[BoundProduct] | None:
    """Bind every named-product fragment to exactly one selected document.

    The question is split only at statement boundaries.  A fragment that
    matches two selected source identities, or a source identity that appears
    in two fragments, is ambiguous and therefore rejected.
    """

    data = load_capsules(path or DEFAULT_PATH)
    selected_ids = set()
    for raw_doc_id in q.get("doc_ids") or []:
        doc_id = str(raw_doc_id)
        if doc_id in selected_ids:
            return None
        selected_ids.add(doc_id)
    # Scan the whole capsule identity catalog, not merely the selected subset.
    # Otherwise a missing selected source could make one named product vanish
    # from a multi-contract sum instead of failing closed.
    catalog = [
        (str(doc_id), doc, doc.get("identity") or {})
        for doc_id, doc in data.get("documents", {}).items()
    ]

    question = q.get("question") or ""
    fragments = [part.strip() for part in re.split(r"[；。]", question)
                 if part.strip()]
    groups: list[tuple[str, str, dict, dict, list[str]]] = []
    bound_docs = set()
    for fragment in fragments:
        compact_fragment = _compact(fragment)
        matches: list[tuple[int, str, dict, dict]] = []
        for doc_id, doc, identity in catalog:
            labels = [label for label in _identity_labels(identity)
                      if label in compact_fragment]
            if labels:
                matches.append((len(labels[0]), doc_id, doc, identity))
        if not matches:
            # A following sentence commonly supplies the operands after a
            # first sentence names the product and event.  It belongs to the
            # preceding unique product until another product is named.
            if groups:
                groups[-1][4].append(fragment)
            continue
        # A product fragment itself must identify one source.  Longest-match
        # scoring is not used to resolve two different documents.
        docs = {match[1] for match in matches}
        if len(docs) != 1:
            return None
        _score, doc_id, doc, identity = matches[0]
        if doc_id not in selected_ids:
            return None
        if doc_id in bound_docs:
            return None
        bound_docs.add(doc_id)
        groups.append((doc_id, str(identity.get("product") or ""),
                       identity, doc, [fragment]))
    bindings = [BoundProduct(
            doc_id=doc_id,
            product=product,
            identity=identity,
            document=doc,
            segment="。".join(parts),
        ) for doc_id, product, identity, doc, parts in groups]
    return bindings or None


def _source(rule) -> str:
    return f"doc={rule.doc_id};P{rule.page};{rule.card_id}"


def _unique_rule(binding: BoundProduct, parser):
    rules = [rule for card in binding.document.get("capsules") or []
             if (rule := parser(card, binding.identity)) is not None]
    return rules[0] if len(rules) == 1 else None


def _direct_surrender_source(binding: BoundProduct) -> str | None:
    """Return source clauses which explicitly refund cash value on surrender.

    Some source packs contain two policy editions with the same semantic
    clause.  Multiple identical *direct cash-value* clauses are harmless: the
    cash-value amount itself is supplied by the question.  Unlike a rate table,
    no row or boundary is selected from these clauses.
    """

    refs = []
    for card in binding.document.get("capsules") or []:
        if card.get("topic") != "surrender":
            continue
        compact = _compact(card.get("verbatim") or "")
        if ("解除本合同" in compact and
                re.search(r"退还(?:本合同的)?现金价值", compact)):
            header = _card_header(card, binding.identity)
            if header is not None:
                doc_id, page, card_id, _product = header
                refs.append(f"doc={doc_id};P{page};{card_id}")
    refs = sorted(set(refs))
    return "/".join(refs) if refs else None


def _unique_year(text: str) -> int | None:
    years = [int(raw) for raw in re.findall(r"第(\d+)个?保单年度", _compact(text))]
    return years[0] if len(years) == 1 and years[0] > 0 else None


def _unique_age(text: str) -> int | None:
    ages = [int(raw) for raw in re.findall(
        r"(?:被保险人|年龄)(?:为)?(\d+)周岁", _compact(text))]
    return ages[0] if len(ages) == 1 and ages[0] >= 0 else None


def _unique_amount(text: str, labels: tuple[str, ...]) -> tuple[D, str] | None:
    compact = _compact(text)
    alternatives = "|".join(re.escape(_compact(label))
                              for label in sorted(labels, key=len, reverse=True))
    matches = re.findall(
        rf"(?:{alternatives})(?:为|是|共计|合计|[:：=])?"
        r"(-?[\d,]+(?:\.\d+)?)(亿元|万元|元)",
        compact,
    )
    if len(matches) != 1:
        return None
    raw_value, unit = matches[0]
    try:
        value = D(raw_value.replace(",", ""))
    except decimal.InvalidOperation:
        return None
    return (value, unit) if value.is_finite() and value >= 0 else None


def _two_scenario_amounts(
    text: str, labels: tuple[str, ...],
) -> tuple[tuple[D, str], tuple[D, str]] | None:
    compact = _compact(text)
    alternatives = "|".join(re.escape(_compact(label))
                              for label in sorted(labels, key=len, reverse=True))
    match = re.search(
        rf"(?:{alternatives})分别为"
        r"(-?[\d,]+(?:\.\d+)?)(亿元|万元|元)(?:和|与|、)"
        r"(-?[\d,]+(?:\.\d+)?)(亿元|万元|元)两种情形",
        compact,
    )
    if not match:
        return None
    try:
        first = D(match.group(1).replace(",", ""))
        second = D(match.group(3).replace(",", ""))
    except decimal.InvalidOperation:
        return None
    if not first.is_finite() or not second.is_finite() or first < 0 or second < 0:
        return None
    return (first, match.group(2)), (second, match.group(4))


def _output_unit(question: str) -> str | None:
    matches = re.findall(r"多少(亿元|万元|元)", _compact(question))
    return matches[0] if len(matches) == 1 and matches[0] in _UNIT_SCALE else None


def _to_base(amount: tuple[D, str] | None) -> D | None:
    if amount is None or amount[1] not in _UNIT_SCALE:
        return None
    return amount[0] * _UNIT_SCALE[amount[1]]


def _component_fact(component: _Component, output_unit: str) -> Fact:
    return Fact(
        "contract_result",
        component.value_base / _UNIT_SCALE[output_unit],
        _FACT_UNIT[output_unit],
        entity=component.product,
        period=component.period,
        source=component.source,
    )


def _question_amount(question: str, label: str) -> tuple[D, str] | None:
    compact = _compact(question)
    match = re.search(
        re.escape(label) + r"(?:为|是|共计|合计)?"
        r"(-?[\d,]+(?:\.\d+)?)(亿元|万元|元)", compact)
    if not match:
        return None
    try:
        value = D(match.group(1).replace(",", ""))
    except decimal.InvalidOperation:
        return None
    return (value, match.group(2)) if value.is_finite() and value >= 0 else None


def _target_years(question: str) -> tuple[int, int] | None:
    compact = _compact(question)
    match = re.search(
        r"第(\d+)个保单年度(?:与|和)第(\d+)个保单年度的?"
        r"现金价值(?:之间的)?差额", compact)
    if not match:
        return None
    first, second = int(match.group(1)), int(match.group(2))
    return (first, second) if first > 0 and second > 0 and first != second else None


def _format(value: D) -> str:
    return format(value.quantize(D("0.01"), rounding=ROUND), "f")


def try_solve_insurance_cash_value(
    q: dict,
    *,
    path: str | pathlib.Path | None = None,
) -> SolveResult | None:
    """Solve a uniquely source-bound policy-year cash-value difference."""

    question = q.get("question") or ""
    if (q.get("domain") != "insurance" or q.get("options") or
            "现金价值" not in question or "差额" not in question or
            "养老保险金开始领取日前" not in _compact(question)):
        return None
    schedule = _select_schedule(q, path)
    premium = _question_amount(question, "累计所交保险费")
    cumulative_return = _question_amount(question, "保单账户累计收益")
    years = _target_years(question)
    if schedule is None or premium is None or cumulative_return is None or years is None:
        return None

    premium_value, premium_unit = premium
    return_value, return_unit = cumulative_return
    output_match = re.search(r"差额为多少(亿元|万元|元)", _compact(question))
    output_unit = output_match.group(1) if output_match else premium_unit
    if premium_unit not in _UNIT_SCALE or return_unit not in _UNIT_SCALE or \
            output_unit not in _UNIT_SCALE:
        return None
    premium_base = premium_value * _UNIT_SCALE[premium_unit]
    return_base = return_value * _UNIT_SCALE[return_unit]
    first_year, second_year = years
    first_value = schedule.cash_value(first_year, premium_base, return_base)
    second_value = schedule.cash_value(second_year, premium_base, return_base)
    if first_value is None or second_value is None:
        return None
    divisor = _UNIT_SCALE[output_unit]
    first_out, second_out = first_value / divisor, second_value / divisor
    difference = abs(first_out - second_out)
    source = f"doc={schedule.doc_id};P{schedule.page};{schedule.card_id}"
    reasoning = (
        f"依据{schedule.product}现金价值原表（{source}），"
        f"第{first_year}个保单年度现金价值={_format(first_out)}{output_unit}，"
        f"第{second_year}个保单年度现金价值={_format(second_out)}{output_unit}；"
        f"绝对差额=|{_format(first_out)}-{_format(second_out)}|="
        f"{_format(difference)}{output_unit}。"
    )
    facts = (
        Fact("contract_result", first_out, _FACT_UNIT[output_unit],
             entity=schedule.product, period=f"第{first_year}个保单年度", source=source),
        Fact("contract_result", second_out, _FACT_UNIT[output_unit],
             entity=schedule.product, period=f"第{second_year}个保单年度", source=source),
    )
    return SolveResult(
        intent=Intent.CONTRACT_RULE_ARITHMETIC,
        slots=(_format(difference),),
        reasoning=reasoning,
        facts=facts,
    )


def _component_text(component: _Component, output_unit: str) -> str:
    value = component.value_base / _UNIT_SCALE[output_unit]
    return (f"{component.product}：{component.detail}="
            f"{_format(value)}{output_unit}（{component.source}）")


def _sum_result(components: list[_Component], output_unit: str,
                operation: str) -> SolveResult | None:
    if not components or output_unit not in _UNIT_SCALE:
        return None
    total_base = sum((component.value_base for component in components), D("0"))
    total = total_base / _UNIT_SCALE[output_unit]
    reasoning = "；".join(_component_text(component, output_unit)
                           for component in components)
    reasoning += f"；{operation}={_format(total)}{output_unit}。"
    return SolveResult(
        intent=Intent.CONTRACT_RULE_ARITHMETIC,
        slots=(_format(total),),
        reasoning=reasoning,
        facts=tuple(_component_fact(component, output_unit)
                    for component in components),
    )


def _surrender_component(binding: BoundProduct) -> _Component | None:
    segment = binding.segment
    year = _unique_year(segment)
    premium = _to_base(_unique_amount(
        segment, ("累计所交保险费", "累计已交保险费", "所交保险费")))
    cumulative_return = _to_base(_unique_amount(
        segment, ("保单账户累计收益", "账户累计收益")))
    account_value = _to_base(_unique_amount(segment, ("个人账户价值",)))
    direct_cash = _to_base(_unique_amount(segment, ("现金价值",)))

    candidates: list[_Component] = []
    if year is not None and premium is not None and cumulative_return is not None:
        rule = _unique_rule(binding, parse_policy_year_schedule)
        value = rule.cash_value(year, premium, cumulative_return) if rule else None
        if rule is not None and value is not None:
            candidates.append(_Component(
                binding.product, value, _source(rule),
                f"第{year}保单年度按累计保费与账户累计收益档位计算",
                f"第{year}个保单年度",
            ))
    if year is not None and account_value is not None:
        rule = _unique_rule(binding, parse_surrender_fee_schedule)
        value = rule.cash_value(year, account_value) if rule else None
        if rule is not None and value is not None:
            candidates.append(_Component(
                binding.product, value, _source(rule),
                f"第{year}保单年度个人账户价值扣除退保费用",
                f"第{year}个保单年度",
            ))
    if direct_cash is not None:
        source = _direct_surrender_source(binding)
        if source:
            candidates.append(_Component(
                binding.product, direct_cash, source,
                "题干给定现金价值作为退还金额", "解除合同",
            ))
    return candidates[0] if len(candidates) == 1 else None


def _try_solve_surrender_total(
    q: dict, path: str | pathlib.Path | None,
) -> SolveResult | None:
    question = q.get("question") or ""
    compact = _compact(question)
    if "合计可退还" not in compact or not ("解除" in compact or "退保" in compact):
        return None
    output_unit = _output_unit(question)
    bindings = _bind_product_segments(q, path)
    if output_unit is None or bindings is None or len(bindings) < 2:
        return None
    components = [_surrender_component(binding) for binding in bindings]
    if any(component is None for component in components):
        return None
    return _sum_result(components, output_unit, "各合同合计可退还金额")


def _death_component(binding: BoundProduct) -> _Component | None:
    segment = binding.segment
    age = _unique_age(segment)
    base_sum = _to_base(_unique_amount(
        segment, ("基本保险金额", "基本保额")))
    personal_account = _to_base(_unique_amount(segment, ("个人账户价值",)))
    commencement_value = _to_base(_unique_amount(
        segment, ("养老保险金开始领取日的保单账户价值",)))
    produced_annuity = _to_base(_unique_amount(
        segment, ("累计已产生养老保险金", "已产生养老保险金")))
    premium = _to_base(_unique_amount(
        segment, ("累计已交保险费", "累计所交保险费", "所交保险费")))
    paid_annuity = _to_base(_unique_amount(
        segment, ("累计已给付养老保险金", "累计已给付养老年金")))
    cash_value = _to_base(_unique_amount(segment, ("现金价值",)))

    candidates: list[_Component] = []
    if age is not None and base_sum is not None and personal_account is not None:
        rule = _unique_rule(binding, parse_age_death_schedule)
        band = rule.band_for_age(age) if rule else None
        value = rule.death_benefit(age, base_sum, personal_account) if rule else None
        if rule is not None and band is not None and value is not None:
            lo, hi, rate = band
            rate_text = format(rate, "f")
            if hi is None:
                boundary = (f"{age}周岁满足{age}≥{lo}，命中年龄档"
                            f"[{lo},+∞)")
            else:
                boundary = (
                    f"{age}周岁满足{lo}≤{age}<{hi}，命中半开年龄档"
                    f"[{lo},{hi})（上界核验：{age}属于[0,{hi})，"
                    f"不属于[{hi},+∞)）"
                )
            candidates.append(_Component(
                binding.product, value, _source(rule),
                f"{boundary}，身故给付比例{rate_text}%；"
                "比例×基本保险金额与个人账户价值取大",
                f"{age}周岁身故",
            ))
    if commencement_value is not None and produced_annuity is not None:
        rule = _unique_rule(binding, parse_post_annuity_death_rule)
        if rule is not None:
            value = rule.death_benefit(commencement_value, produced_annuity)
            candidates.append(_Component(
                binding.product, value, _source(rule),
                "开始领取日账户价值扣除累计已产生养老保险金，不低于零",
                "养老保险金开始领取日后身故",
            ))
    if premium is not None and paid_annuity is not None and cash_value is not None:
        rule = _unique_rule(binding, parse_net_premium_death_rule)
        if rule is not None:
            value = rule.death_benefit(premium, paid_annuity, cash_value)
            candidates.append(_Component(
                binding.product, value, _source(rule),
                "已交保费扣除累计已给付养老金与现金价值取大",
                "身故",
            ))
    return candidates[0] if len(candidates) == 1 else None


def _try_solve_death_total(
    q: dict, path: str | pathlib.Path | None,
) -> SolveResult | None:
    question = q.get("question") or ""
    if "合计身故保险金" not in _compact(question):
        return None
    output_unit = _output_unit(question)
    bindings = _bind_product_segments(q, path)
    if output_unit is None or bindings is None or len(bindings) < 2:
        return None
    components = [_death_component(binding) for binding in bindings]
    if any(component is None for component in components):
        return None
    return _sum_result(components, output_unit, "各合同身故保险金合计")


def _try_solve_post_annuity_scenarios(
    q: dict, path: str | pathlib.Path | None,
) -> SolveResult | None:
    question = q.get("question") or ""
    compact = _compact(question)
    if ("养老保险金开始领取日及之后身故" not in compact or
            "两种情形" not in compact or "身故保险金合计" not in compact):
        return None
    output_unit = _output_unit(question)
    bindings = _bind_product_segments(q, path)
    if output_unit is None or bindings is None or len(bindings) != 1:
        return None
    binding = bindings[0]
    rule = _unique_rule(binding, parse_post_annuity_death_rule)
    commencement = _to_base(_unique_amount(
        binding.segment, ("养老保险金开始领取日的保单账户价值",)))
    scenarios = _two_scenario_amounts(
        binding.segment, ("累计已产生养老保险金", "已产生养老保险金"))
    if rule is None or commencement is None or scenarios is None:
        return None
    scenario_bases = [_to_base(amount) for amount in scenarios]
    if any(amount is None for amount in scenario_bases):
        return None
    components = []
    for index, produced in enumerate(scenario_bases, 1):
        value = rule.death_benefit(commencement, produced)
        components.append(_Component(
            binding.product, value, _source(rule),
            "开始领取日账户价值扣除该情形累计已产生养老保险金，不低于零",
            f"情形{index}",
        ))
    return _sum_result(components, output_unit, "两种情形对应身故保险金合计")


def try_solve_insurance(
    q: dict,
    *,
    path: str | pathlib.Path | None = None,
) -> SolveResult | None:
    """Dispatch strict source-bound insurance arithmetic by question shape.

    This function intentionally never reads ``qid``.  It accepts a result only
    when exactly one structural solver succeeds, every named product binds to
    one selected capsule identity, and every monetary operand has a closed
    unit conversion.
    """

    if q.get("domain") != "insurance" or q.get("options"):
        return None
    solvers = (
        lambda: try_solve_insurance_cash_value(q, path=path),
        lambda: _try_solve_surrender_total(q, path),
        lambda: _try_solve_death_total(q, path),
        lambda: _try_solve_post_annuity_scenarios(q, path),
    )
    results = [result for solve in solvers if (result := solve()) is not None]
    return results[0] if len(results) == 1 else None
