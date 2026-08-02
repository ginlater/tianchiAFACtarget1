"""Deterministic, fail-closed facts for financial-report calculations.

The registry has no knowledge of question identifiers, answer files, scored
runs, or historical outputs.  Its complete input boundary is:

* the natural-language question and domain;
* an explicit list of document identifiers selected by the caller; and
* reproducible files under ``processed_data``.

It converts exact report disclosures into :class:`deterministic_calc.Fact`
objects.  Duplicate disclosures with the same value corroborate one another;
conflicting values, an unknown monetary unit, or a missing page attribution
fail closed.  The existing Qwen calculation path can therefore remain the
fallback whenever :meth:`FinancialFactRegistry.extract` reports a missing or
conflicting operand.

The implementation deliberately uses table headings, report years, column
roles, company names, and metric language.  It does not contain per-question
branches or expected answers.
"""

from __future__ import annotations

import dataclasses
import decimal
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .deterministic_calc import Fact, Intent, canonical_metric, classify_intent


D = decimal.Decimal

_MONEY_SCALE = {
    "元": D(1),
    "千元": D(1_000),
    "百万元": D(1_000_000),
    "万元": D(10_000),
    "亿元": D(100_000_000),
}
_NUMBER = re.compile(r"-?[\d,]+(?:\.\d+)?")
_PERCENT = re.compile(r"-?[\d,]+(?:\.\d+)?\s*[%％]")
_REPORT_YEAR = re.compile(r"_((?:19|20)\d{2})_report$")
_PAGE_MARKER = re.compile(r"^\[P(\d+)\]\s*$", re.MULTILINE)

_METRIC_LABELS: Mapping[str, tuple[str, ...]] = {
    "revenue": ("营业收入",),
    "parent_net_profit": (
        "归属于上市公司股东的净利润",
        "归属于母公司所有者的净利润",
        "归属于母公司股东的净利润",
        "归属于本行股东的净利润",
    ),
    "operating_cash_flow": ("经营活动产生的现金流量净额",),
    "roe": ("加权平均净资产收益率",),
}

_STATEMENT_METRICS = frozenset({
    "revenue", "parent_net_profit", "operating_cash_flow"
})

_INTENT_METRICS: Mapping[Intent, tuple[str, ...]] = {
    Intent.YOY_AND_SHARE_PP: ("foreign_revenue", "revenue"),
    Intent.MARGIN_DROP: ("revenue", "parent_net_profit", "operating_cash_flow"),
    Intent.CASHFLOW_RATE_RANK: ("revenue", "operating_cash_flow"),
    Intent.DIVIDEND_RANK: ("dividend_per_10",),
    Intent.IMPLIED_REVENUE: ("ebitda", "ebitda_rate", "revenue"),
    Intent.DUPONT: ("debt_ratio", "roe"),
    Intent.EQUITY_MULTIPLIER_RANK: ("debt_ratio",),
    Intent.DIVIDEND_RECONCILE: (
        "parent_net_profit", "dividend_ratio", "dividend_per_10", "base_shares"
    ),
}


@dataclasses.dataclass(frozen=True)
class RegistryResult:
    """Facts plus fail-closed diagnostics.

    ``missing`` contains required operands that could not be uniquely
    extracted.  ``conflicts`` names operands for which exact source passages
    disagreed.  A caller should normally attempt the deterministic solver only
    with ``complete`` results, although the solver repeats its own uniqueness
    checks as a second safety boundary.
    """

    facts: tuple[Fact, ...]
    missing: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    ignored_documents: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return bool(self.facts) and not self.missing and not self.conflicts

    @property
    def fact_block(self) -> str:
        return facts_to_block(self.facts)


@dataclasses.dataclass(frozen=True)
class _Document:
    identifier: str
    entity: str
    year: str
    raw: str
    pages: Mapping[int, str]
    fact_rows: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class _Candidate:
    fact: Fact
    document: str


def _decimal_text(value: D) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _field(text: object) -> str:
    return re.sub(r"[|\r\n]+", " ", str(text or "")).strip()


def facts_to_block(facts: Iterable[Fact]) -> str:
    """Serialize facts into the interchange accepted by ``try_solve``."""

    lines: list[str] = []
    for fact in facts:
        lines.append(
            "FACT|entity={}|period={}|metric={}|scope={}|value={}|unit={}|source={}".format(
                _field(fact.entity),
                _field(fact.period),
                _field(fact.metric),
                _field(fact.scope),
                _decimal_text(fact.value),
                _field(fact.unit),
                _field(fact.source),
            )
        )
    return "\n".join(lines)


def _pages(raw: str) -> dict[int, str]:
    parts = _PAGE_MARKER.split(raw or "")
    return {
        int(parts[index]): parts[index + 1]
        for index in range(1, len(parts) - 1, 2)
    }


def _plain(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _traditional_to_simple(text: str) -> str:
    table = str.maketrans({
        "銀": "银", "國": "国", "動": "动", "築": "筑", "時": "时",
        "團": "团", "業": "业", "東": "东", "華": "华", "報": "报",
    })
    return (text or "").translate(table)


def _clean_company_name(value: str) -> str:
    value = _traditional_to_simple(value)
    value = re.sub(r"[（(].*$", "", value)
    value = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", value)
    for suffix in ("股份有限公司", "集团有限公司", "有限公司"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    return value


def _company_candidates(raw: str, title: str) -> set[str]:
    """Derive reusable short company-name candidates from report metadata."""

    sample = raw[:20_000]
    candidates: set[str] = set()
    for pattern in (
        r"公司的中文简称\s*\n\s*([^\n]+)",
        r"公司中文简称\s*\n\s*([^\n]+)",
        r"(?:A\s*股)?股票简称\s*\n\s*([^\n]+)",
    ):
        for match in re.finditer(pattern, sample, flags=re.I):
            candidates.add(_clean_company_name(match.group(1)))
    for text in (title, sample[:4_000]):
        for match in re.finditer(r"([\u4e00-\u9fff]{2,24}(?:股份有限公司|有限公司))", text):
            candidates.add(_clean_company_name(match.group(1)))
    return {c for c in candidates if 2 <= len(c) <= 16}


def _bind_entity(question: str, candidates: Iterable[str]) -> str:
    normalized_question = _traditional_to_simple(question).replace(" ", "")
    present = [candidate for candidate in candidates if candidate in normalized_question]
    if not present:
        return ""
    # Prefer the longest explicit name in the question.  This keeps e.g.
    # ``美的集团`` instead of a shorter incidental token.
    return sorted(present, key=lambda item: (-len(item), item))[0]


def _entity_for_question(question: str, raw: str, title: str) -> str:
    """Derive a report's short company name and bind it to the question."""

    return _bind_entity(question, _company_candidates(raw, title))


def _loose_label(label: str) -> str:
    return r"\s*".join(re.escape(ch) for ch in label)


def _source(document: str, pages: Sequence[int]) -> str:
    unique = sorted(set(int(page) for page in pages if int(page) > 0))
    return document + ":" + "+".join(f"P{page}" for page in unique)


def _page_unit(pages: Mapping[int, str], page: int, *, near: str = "") -> str:
    """Return a unique monetary unit at or immediately before a source page."""

    if near:
        match = re.search(r"[（(](?:人民币)?(百万元|千元|万元|亿元|元)(?:[/／]股)?[）)]", near)
        if match:
            return match.group(1)
    for distance in range(0, 5):
        body = pages.get(page - distance, "")
        for pattern in (
            r"单位[：:]\s*(?:人民币)?\s*(百万元|千元|万元|亿元|元)",
            r"金额单位为人民币(百万元|千元|万元|亿元|元)",
            r"均以人民币(百万元|千元|万元|亿元|元)为单位",
            r"财务附注中报表的单位为[：:]\s*(百万元|千元|万元|亿元|元)",
            r"（人民币(百万元|千元|万元|亿元|元)[，,）)]",
            r"^\s*人民币(百万元|千元|万元|亿元|元)\s*$",
        ):
            units = set(re.findall(pattern, body, flags=re.MULTILINE))
            if len(units) == 1:
                return next(iter(units))
            if len(units) > 1:
                return ""
    return ""


def _number(raw: str) -> D | None:
    text = (raw or "").strip().replace(",", "").replace("％", "%")
    text = text.rstrip("%")
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return None
    value = D(text)
    return value if value.is_finite() else None


def _make_money(
    metric: str,
    raw_value: str,
    unit: str,
    document: _Document,
    period: str,
    pages: Sequence[int],
) -> _Candidate | None:
    value = _number(raw_value)
    scale = _MONEY_SCALE.get(unit)
    if value is None or scale is None or not pages:
        return None
    return _Candidate(Fact(
        metric=metric,
        value=value * scale,
        unit="yuan",
        entity=document.entity,
        period=period,
        scope="合并",
        source=_source(document.identifier, pages),
    ), document.identifier)


def _make_plain(
    metric: str,
    raw_value: str,
    unit: str,
    document: _Document,
    period: str,
    pages: Sequence[int],
) -> _Candidate | None:
    value = _number(raw_value)
    if value is None or not pages:
        return None
    return _Candidate(Fact(
        metric=metric,
        value=value,
        unit=unit,
        entity=document.entity,
        period=period,
        scope="合并",
        source=_source(document.identifier, pages),
    ), document.identifier)


def _statement_facts(document: _Document, metrics: set[str]) -> list[_Candidate]:
    """Read column-bound consolidated facts emitted by ``fin_facts2``."""

    parsed: list[tuple[str, bool, str, str, int]] = []
    # The label itself can contain punctuation (notably ``其中：营业收入``),
    # so the greedy label group intentionally splits on the final colon.
    row_pattern = re.compile(r"^\[([^]]+)]\s*(.+)[：:]\s*(.*?)\s*\(P(\d+)\)$")
    column_pattern = re.compile(
        r"(合并本期|合并上期)\s*=\s*(-?[\d,]+(?:\.\d+)?|-)"
    )
    for row in document.fact_rows:
        match = row_pattern.match(row.strip())
        if not match:
            continue
        table, label, body, page_text = match.groups()
        if "合并" not in table:
            continue
        metric = canonical_metric(label)
        if metric not in metrics or metric not in _STATEMENT_METRICS:
            continue
        compact_label = re.sub(r"[\s一二三四五六七八九十、：:]", "", label)
        exact_revenue = metric != "revenue" or "营业总收入" not in compact_label
        for role, value in column_pattern.findall(body):
            if value == "-":
                continue
            period = document.year if role == "合并本期" else str(int(document.year) - 1)
            parsed.append((metric, exact_revenue, period, value, int(page_text)))

    # ``营业收入`` is the required accounting row.  ``营业总收入`` is only a
    # fallback for institutions whose report does not expose the narrower row.
    exact_periods = {
        period for metric, exact, period, _, _ in parsed if metric == "revenue" and exact
    }
    out: list[_Candidate] = []
    for metric, exact, period, value, page in parsed:
        if metric == "revenue" and not exact and period in exact_periods:
            continue
        unit = _page_unit(document.pages, page)
        candidate = _make_money(metric, value, unit, document, period, [page])
        if candidate:
            out.append(candidate)
    return out


def _summary_windows(document: _Document) -> list[tuple[int, str]]:
    """Return early report pages headed as audited multi-year summaries."""

    selected: list[tuple[int, str]] = []
    ordered = sorted(document.pages)
    for page in ordered:
        if page > 35:
            break
        body = document.pages[page]
        previous = document.pages.get(page - 1, "")
        own_header = "主要会计数据" in body
        continuation = "主要会计数据" in previous
        # If the previous page already opened the quarterly table, this page
        # is not a continuation of the annual summary even though it follows
        # the summary heading physically.
        if continuation and "分季度主要财务指标" in previous:
            continue
        if own_header or continuation:
            # Do not let quarterly or non-recurring-loss tables on the same
            # physical page masquerade as the annual summary row above them.
            cut = len(body)
            for marker in ("分季度", "非经常性损益项目"):
                position = body.find(marker)
                if position >= 0:
                    cut = min(cut, position)
            selected.append((page, body[:cut]))
    return selected


def _values_after_label(body: str, labels: Sequence[str], *, percent: bool = False) -> tuple[str, str] | None:
    token = r"(-?[\d,]+(?:\.\d+)?\s*[%％])" if percent else r"(-?[\d,]+(?:\.\d+)?)"
    for label in labels:
        pattern = (
            _loose_label(label)
            + r"\s*(?:[（(][^）)]{0,30}[）)])?\s*"
            + token + r"\s*" + token
        )
        match = re.search(pattern, body)
        if match:
            return match.group(1), match.group(2)
    return None


def _summary_facts(document: _Document, metrics: set[str]) -> list[_Candidate]:
    out: list[_Candidate] = []
    for page, body in _summary_windows(document):
        for metric in metrics & set(_METRIC_LABELS):
            is_percent = metric == "roe"
            pair = _values_after_label(body, _METRIC_LABELS[metric], percent=is_percent)
            if not pair:
                continue
            if is_percent:
                for value, period in zip(pair, (document.year, str(int(document.year) - 1))):
                    candidate = _make_plain(metric, value, "percent", document, period, [page])
                    if candidate:
                        out.append(candidate)
                continue
            # A unit attached directly to the row is stronger than a page-wide
            # unit, especially where a summary mixes yuan/share and money rows.
            unit = ""
            for label in _METRIC_LABELS[metric]:
                match = re.search(_loose_label(label) + r"\s*[（(](?:人民币)?(百万元|千元|万元|亿元|元)", body)
                if match:
                    unit = match.group(1)
                    break
            unit = unit or _page_unit(document.pages, page)
            for value, period in zip(pair, (document.year, str(int(document.year) - 1))):
                candidate = _make_money(metric, value, unit, document, period, [page])
                if candidate:
                    out.append(candidate)
    return out


def _foreign_revenue_facts(document: _Document) -> list[_Candidate]:
    out: list[_Candidate] = []
    for page, body in document.pages.items():
        if not all(marker in body for marker in ("营业收入构成", "分地区", "境外", "营业收入合计")):
            continue
        # Revenue-composition tables interleave each amount with its share:
        # current amount, current share, previous amount, previous share.
        def amount_pair(label: str) -> tuple[str, str] | None:
            number = r"(-?[\d,]+(?:\.\d+)?)"
            match = re.search(
                _loose_label(label) + r"\s*" + number
                + r"\s*-?[\d,.]+\s*[%％]\s*" + number,
                body,
            )
            return (match.group(1), match.group(2)) if match else None

        revenues = amount_pair("营业收入合计")
        foreign = amount_pair("境外")
        unit = _page_unit(document.pages, page)
        if not revenues or not foreign or not unit:
            continue
        periods = (document.year, str(int(document.year) - 1))
        for metric, pair in (("revenue", revenues), ("foreign_revenue", foreign)):
            for value, period in zip(pair, periods):
                candidate = _make_money(metric, value, unit, document, period, [page])
                if candidate:
                    out.append(candidate)
        # The first complete revenue-composition table is authoritative; later
        # segment notes can contain narrower geographic scopes.
        break
    return out


def _debt_ratio_facts(document: _Document) -> list[_Candidate]:
    out: list[_Candidate] = []
    for page, body in document.pages.items():
        lines = [line.strip() for line in body.splitlines()]
        for index, line in enumerate(lines):
            if re.sub(r"[（）()%％\s]", "", line) != "资产负债率":
                continue
            values: list[str] = []
            for following in lines[index + 1:index + 12]:
                values.extend(match.group(0) for match in _PERCENT.finditer(following))
                if len(values) >= 2:
                    break
            if len(values) < 2:
                continue
            periods = (document.year, str(int(document.year) - 1))
            for value, period in zip(values[:2], periods):
                candidate = _make_plain("debt_ratio", value, "percent", document, period, [page])
                if candidate:
                    out.append(candidate)
            return out
        sentence = re.search(
            r"资产负债率为\s*(-?[\d.]+)%\s*[（(]\s*(?:19|20)\d{2}[^：:]{0,20}[：:]\s*(-?[\d.]+)%",
            body,
        )
        if sentence:
            for value, period in zip(sentence.groups(), (document.year, str(int(document.year) - 1))):
                candidate = _make_plain("debt_ratio", value, "percent", document, period, [page])
                if candidate:
                    out.append(candidate)
            return out
    return out


def _ebitda_facts(document: _Document) -> list[_Candidate]:
    out: list[_Candidate] = []
    for page, body in document.pages.items():
        if "主要财务指标" not in body or "EBITDA率" not in _plain(body):
            continue
        lines = [line.strip() for line in body.splitlines()]

        def exact_line_pair(label: str, pattern: re.Pattern[str]) -> tuple[str, str] | None:
            for index, line in enumerate(lines):
                if _plain(line).upper() != _plain(label).upper():
                    continue
                values: list[str] = []
                for following in lines[index + 1:index + 10]:
                    if re.fullmatch(r"注\d+", _plain(following)):
                        continue
                    values.extend(match.group(0) for match in pattern.finditer(following))
                    if len(values) >= 2:
                        return values[0], values[1]
            return None

        amount = exact_line_pair("EBITDA", _NUMBER)
        rate = exact_line_pair("EBITDA率", _PERCENT)
        unit = _page_unit(document.pages, page)
        if not amount or not rate or not unit:
            continue
        periods = (document.year, str(int(document.year) - 1))
        for value, period in zip(amount, periods):
            candidate = _make_money("ebitda", value, unit, document, period, [page])
            if candidate:
                out.append(candidate)
        for value, period in zip(rate, periods):
            candidate = _make_plain("ebitda_rate", value, "percent", document, period, [page])
            if candidate:
                out.append(candidate)
        break
    return out


def _find_page_value(
    document: _Document,
    pattern: str,
    *, flags: int = 0,
) -> tuple[D, int] | None:
    matches: list[tuple[D, int]] = []
    for page, body in document.pages.items():
        for match in re.finditer(pattern, body, flags=flags):
            value = _number(match.group(1))
            if value is not None:
                matches.append((value, page))
    unique = {value for value, _ in matches}
    if len(unique) != 1:
        return None
    value = next(iter(unique))
    page = min(page for candidate, page in matches if candidate == value)
    return value, page


def _dividend_fact(document: _Document, align_rows: Sequence[str]) -> _Candidate | None:
    """Resolve full-year cash dividend per ten shares by disclosure role."""

    raw_pages = list(document.pages.items())

    def unique_matches(pattern: str) -> list[tuple[D, int]]:
        found: list[tuple[D, int]] = []
        for page, body in raw_pages:
            compact = _plain(body)
            for match in re.finditer(pattern, compact):
                value = _number(match.group(1))
                if value is not None:
                    found.append((value, page))
        return found

    # Strongest form: the report explicitly calls a figure the full-year
    # amount.  Per-share bank disclosures are converted to per-ten-share.
    per_share = unique_matches(r"全年每股现金分红([\d,.]+)元")
    if per_share:
        values = {value for value, _ in per_share}
        if len(values) == 1:
            value = next(iter(values)) * D(10)
            pages = [page for _, page in per_share]
            return _make_plain("dividend_per_10", str(value), "yuan", document, document.year, pages)
        return None

    annual_total = unique_matches(
        rf"(?:公司)?{re.escape(document.year)}年度利润分配方案为[：:]?每10股(?:派发)?现金(?:分红)?(?:人民币)?([\d,.]+)元"
    )
    if annual_total:
        values = {value for value, _ in annual_total}
        if len(values) == 1:
            value = next(iter(values))
            pages = [page for _, page in annual_total]
            return _make_plain("dividend_per_10", str(value), "yuan", document, document.year, pages)
        return None

    annual = unique_matches(
        rf"{re.escape(document.year)}年度利润分配预案[^。；]{{0,220}}?每10股(?:派发|派送)?(?:现金分红|现金红利|现金股息)?(?:人民币)?([\d,.]+)元"
    )
    interim = unique_matches(
        rf"{re.escape(document.year)}年(?:度)?中期(?:利润分配|分红)[^。；]{{0,180}}?每\s*10股(?:派发)?现金(?:分红|股息)?(?:人民币)?([\d,.]+)元"
    )
    annual_values = {value for value, _ in annual}
    interim_values = {value for value, _ in interim}
    if len(annual_values) == 1 and len(interim_values) <= 1:
        value = next(iter(annual_values))
        pages = [page for _, page in annual]
        if interim_values:
            value += next(iter(interim_values))
            pages += [page for _, page in interim]
        return _make_plain("dividend_per_10", str(value), "yuan", document, document.year, pages)

    # Some reports phrase the annual proposal without the leading year.  This
    # form is accepted only if there is no separately disclosed current-year
    # interim amount, so it cannot silently undercount a full-year request.
    generic = unique_matches(
        r"(?:本报告期|本年度)?利润分配(?:预案|方案)[^。；]{0,220}?每10股(?:派发|派送)?(?:现金分红|现金红利|现金股息)?(?:人民币)?([\d,.]+)元"
    )
    generic_values = {value for value, _ in generic}
    if len(generic_values) == 1 and not interim_values:
        value = next(iter(generic_values))
        return _make_plain(
            "dividend_per_10", str(value), "yuan", document, document.year,
            [page for _, page in generic],
        )

    # ``align_matrix`` is a deterministic excerpt index.  It is a final
    # recovery source only; the same role/uniqueness rules still apply.
    aligned: list[tuple[D, int]] = []
    for row in align_rows:
        page_match = re.match(r"\[P(\d+)]", row)
        value_match = re.search(r"全年每股现金分红([\d,.]+)元", _plain(row))
        if page_match and value_match:
            value = _number(value_match.group(1))
            if value is not None:
                aligned.append((value * D(10), int(page_match.group(1))))
    aligned_values = {value for value, _ in aligned}
    if len(aligned_values) == 1:
        value = next(iter(aligned_values))
        return _make_plain(
            "dividend_per_10", str(value), "yuan", document, document.year,
            [page for _, page in aligned],
        )
    return None


def _dividend_reconcile_facts(document: _Document) -> list[_Candidate]:
    out: list[_Candidate] = []
    # All three operands must occur in a page that describes the current-year
    # distribution proposal.  This excludes historical distribution tables.
    for page, body in document.pages.items():
        compact = _plain(body)
        if "利润分配" not in compact or document.year not in compact:
            continue
        ratio = re.search(r"现金分红占(?:合并报表)?归属于(?:上市公司|母公司)(?:普通股)?股东净利润的比例为([\d.]+)%", compact)
        per10 = re.search(r"每10股(?:派发|派送)?(?:现金分红|现金红利|现金股息)?(?:人民币)?([\d.]+)元", compact)
        shares = re.search(r"(?:总股本|股本)([\d,]+)股为基数", compact)
        if not (ratio and per10 and shares):
            continue
        for metric, match, unit in (
            ("dividend_ratio", ratio, "percent"),
            ("dividend_per_10", per10, "yuan"),
            ("base_shares", shares, "share"),
        ):
            candidate = _make_plain(metric, match.group(1), unit, document, document.year, [page])
            if candidate:
                out.append(candidate)
        # Prefer the earliest complete proposal disclosure; repeated detailed
        # sections are corroborating but not needed to establish uniqueness.
        break
    return out


def _candidate_key(fact: Fact) -> tuple[str, str, str, str]:
    return fact.metric, fact.entity, fact.period, fact.scope


def _resolve(candidates: Iterable[_Candidate]) -> tuple[list[Fact], list[str]]:
    grouped: dict[tuple[str, str, str, str], list[Fact]] = defaultdict(list)
    for candidate in candidates:
        fact = candidate.fact
        if fact.source and re.search(r":P\d+", fact.source):
            grouped[_candidate_key(fact)].append(fact)

    facts: list[Fact] = []
    conflicts: list[str] = []
    for key in sorted(grouped):
        values = {(fact.value, fact.unit) for fact in grouped[key]}
        descriptor = "/".join(part or "-" for part in key[:3])
        if len(values) != 1:
            conflicts.append(descriptor)
            continue
        first = grouped[key][0]
        sources = sorted({fact.source for fact in grouped[key]})
        facts.append(dataclasses.replace(first, source=";".join(sources)))
    return facts, conflicts


def _years_in_question(question: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"(?:19|20)\d{2}", question)))


def _named_entities_in_question(question: str) -> list[str]:
    """Read the company enumeration before the first report year.

    Calculation questions conventionally introduce their source companies as
    ``查阅甲、乙和丙 2025 年年度报告``.  Parsing this enumeration lets the
    registry notice an incomplete document selection without opening any
    unselected report.
    """

    compact = _traditional_to_simple(question)
    match = re.search(
        r"(?:查阅|根据|结合)\s*(.+?)\s*(?=(?:19|20)\d{2}\s*年)", compact
    )
    if not match:
        return []
    parts = re.split(r"[、,，]|(?:\s*[与和]\s*)", match.group(1))
    names: list[str] = []
    for part in parts:
        name = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", "", part)
        name = re.sub(r"^(?:相关|各|上述)", "", name)
        if 2 <= len(name) <= 16 and name not in names:
            names.append(name)
    return names


def _missing_operands(
    question: str,
    intent: Intent,
    metrics: Sequence[str],
    facts: Sequence[Fact],
    expected_entities: Sequence[str],
) -> list[str]:
    available = {(fact.metric, fact.entity, fact.period) for fact in facts}
    entities = list(dict.fromkeys(entity for entity in expected_entities if entity))
    question_years = _years_in_question(question)
    missing: list[str] = []

    if intent in {Intent.YOY_AND_SHARE_PP, Intent.MARGIN_DROP}:
        targets = [(metric, entities[0] if len(entities) == 1 else "", year)
                   for metric in metrics for year in question_years]
    elif intent in {
        Intent.CASHFLOW_RATE_RANK, Intent.DIVIDEND_RANK,
        Intent.EQUITY_MULTIPLIER_RANK,
    }:
        targets = [(metric, entity, question_years[-1] if question_years else "")
                   for entity in entities for metric in metrics]
    else:
        targets = [(metric, entities[0] if len(entities) == 1 else "",
                    question_years[-1] if question_years else "") for metric in metrics]

    if not entities:
        return [f"{metric}/-/" + (question_years[-1] if question_years else "-") for metric in metrics]
    for metric, entity, period in targets:
        matches = [item for item in available
                   if item[0] == metric and item[1] == entity
                   and (not period or item[2] == period)]
        if not matches:
            missing.append("/".join((metric, entity or "-", period or "-")))
    return sorted(set(missing))


class FinancialFactRegistry:
    """Extract exact financial calculation operands from selected reports."""

    def __init__(self, processed_dir: str | Path | None = None):
        if processed_dir is None:
            processed_dir = Path(__file__).resolve().parents[1] / "processed_data"
        self.processed_dir = Path(processed_dir)
        self._meta: Mapping[str, Mapping[str, object]] | None = None
        self._fact_rows: Mapping[str, Sequence[str]] | None = None
        self._align: Mapping[str, object] | None = None

    def _load(self) -> None:
        if self._meta is not None:
            return
        with (self.processed_dir / "docs_meta.json").open(encoding="utf-8") as handle:
            self._meta = json.load(handle)
        with (self.processed_dir / "fin_facts2.json").open(encoding="utf-8") as handle:
            self._fact_rows = json.load(handle)
        with (self.processed_dir / "align_matrix.json").open(encoding="utf-8") as handle:
            self._align = json.load(handle)

    def _document(self, identifier: str, question: str) -> _Document | None:
        assert self._meta is not None and self._fact_rows is not None
        meta = self._meta.get(identifier)
        year_match = _REPORT_YEAR.search(identifier)
        if not meta or meta.get("domain") != "financial_reports" or not year_match:
            return None
        path = self.processed_dir / "financial_reports" / f"{identifier}.txt"
        if not path.is_file():
            return None
        raw = path.read_text(encoding="utf-8")
        entity = _entity_for_question(question, raw, str(meta.get("title", "")))
        if not entity:
            return None
        return _Document(
            identifier=identifier,
            entity=entity,
            year=year_match.group(1),
            raw=raw,
            pages=_pages(raw),
            fact_rows=tuple(self._fact_rows.get(identifier, ())),
        )

    def extract(
        self,
        item: str | Mapping[str, object],
        document_ids: Sequence[str] | None = None,
    ) -> RegistryResult:
        """Extract facts for one question using only explicitly supplied docs.

        ``item`` may be the question string itself or a mapping containing
        ``question``, ``domain``, and optionally ``doc_ids``.  Extra mapping
        fields are ignored and never copied into output.
        """

        self._load()
        if isinstance(item, Mapping):
            question = str(item.get("question", ""))
            domain = str(item.get("domain", "financial_reports"))
            supplied = item.get("doc_ids", ()) if document_ids is None else document_ids
        else:
            question = str(item)
            domain = "financial_reports"
            supplied = () if document_ids is None else document_ids
        if isinstance(supplied, str):
            supplied = (supplied,)
        identifiers = tuple(dict.fromkeys(str(value) for value in supplied or ()))
        intent = classify_intent(question)
        metrics = _INTENT_METRICS.get(intent, ())
        if domain != "financial_reports" or not question or not identifiers or not metrics:
            return RegistryResult((), tuple(metrics))

        requested_years = set(_years_in_question(question))
        documents: list[_Document] = []
        ignored: list[str] = []
        for identifier in identifiers:
            document = self._document(identifier, question)
            if document is None or (requested_years and document.year not in requested_years):
                ignored.append(identifier)
                continue
            documents.append(document)

        candidates: list[_Candidate] = []
        metric_set = set(metrics)
        assert self._align is not None
        dividends = self._align.get("fin_dividends", {})
        dividend_map = dividends if isinstance(dividends, Mapping) else {}
        for document in documents:
            candidates.extend(_statement_facts(document, metric_set))
            candidates.extend(_summary_facts(document, metric_set))
            if "foreign_revenue" in metric_set:
                candidates.extend(_foreign_revenue_facts(document))
            if "debt_ratio" in metric_set:
                candidates.extend(_debt_ratio_facts(document))
            if {"ebitda", "ebitda_rate"} & metric_set:
                candidates.extend(_ebitda_facts(document))
            if "dividend_per_10" in metric_set:
                align_key = re.sub(r"^annual_|_report$", "", document.identifier)
                rows = dividend_map.get(align_key, ())
                dividend = _dividend_fact(document, rows if isinstance(rows, Sequence) else ())
                if dividend:
                    candidates.append(dividend)
            if intent == Intent.DIVIDEND_RECONCILE:
                candidates.extend(_dividend_reconcile_facts(document))

        # Discard irrelevant periods before conflict resolution.  A malformed
        # comparative value outside the requested years must not veto an exact
        # current-period operand.
        candidates = [candidate for candidate in candidates
                      if candidate.fact.metric in metric_set
                      and (not requested_years or candidate.fact.period in requested_years)]
        facts, conflicts = _resolve(candidates)
        expected_entities = _named_entities_in_question(question)
        if not expected_entities:
            expected_entities = list(dict.fromkeys(document.entity for document in documents))
        missing = _missing_operands(
            question, intent, metrics, facts, expected_entities
        )
        return RegistryResult(
            tuple(facts), tuple(missing), tuple(conflicts), tuple(ignored)
        )

    def facts_for(
        self,
        item: str | Mapping[str, object],
        document_ids: Sequence[str] | None = None,
    ) -> list[Fact]:
        """Return facts only when every operand is uniquely available."""

        result = self.extract(item, document_ids)
        return list(result.facts) if result.complete else []

    def fact_block(
        self,
        item: str | Mapping[str, object],
        document_ids: Sequence[str] | None = None,
    ) -> str:
        """Return ``FACT`` records, or an empty string on any uncertainty."""

        result = self.extract(item, document_ids)
        return result.fact_block if result.complete else ""


__all__ = [
    "FinancialFactRegistry",
    "RegistryResult",
    "facts_to_block",
]
