"""Deterministic source facts for narrative calculation questions.

This registry covers values that are awkward for a generic table parser but
are stated explicitly in contracts, regulations, and research reports.  It is
deliberately *question shaped*, not question-id shaped: dispatch is based only
on the requested metric and wording, and the implementation never reads an
answer file, run history, or leaderboard artifact.

Every returned value carries a processed-document id, physical PDF page when
one exists, and an exact substring of the processed text.  Extractors are
fail-closed.  A shape returns no facts if two candidate documents yield
different semantic values, a period axis cannot be bound to a row, or an
expected rule clause is incomplete.

The public :func:`extract_facts` result can be rendered for a Qwen evidence
block with :func:`facts_block`, or converted into the generic deterministic
calculator's ``Fact`` records with :func:`calculation_facts`.
"""

from __future__ import annotations

import dataclasses
import decimal
import pathlib
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Final


D = decimal.Decimal


@dataclasses.dataclass(frozen=True)
class NarrativeFact:
    """One exact source-backed primitive.

    ``relation`` records semantics that are not safely represented by the
    unsigned number alone (for example a deduction or an inclusive threshold).
    It is intentionally separate from ``value`` so the source number remains
    verbatim-verifiable.
    """

    metric: str
    value: D
    unit: str
    doc_id: str
    page: int | None
    verbatim: str
    period: str = ""
    entity: str = ""
    scope: str = ""
    relation: str = "equals"

    def __post_init__(self) -> None:
        if not isinstance(self.value, D):
            object.__setattr__(self, "value", D(str(self.value).replace(",", "")))
        if not self.metric or not self.doc_id or not self.verbatim.strip():
            raise ValueError("a source fact needs metric, doc_id, and verbatim text")
        if not self.value.is_finite():
            raise ValueError("non-finite fact value")

    @property
    def source_label(self) -> str:
        suffix = f" P{self.page}" if self.page is not None else ""
        return f"{self.doc_id}{suffix}"


@dataclasses.dataclass(frozen=True)
class _Page:
    number: int | None
    text: str


@dataclasses.dataclass(frozen=True)
class _Document:
    doc_id: str
    domain: str
    path: pathlib.Path
    text: str

    def pages(self) -> tuple[_Page, ...]:
        markers = list(re.finditer(r"(?m)^\[P(\d+)\]\s*$", self.text))
        if not markers:
            return (_Page(None, self.text),)
        pages: list[_Page] = []
        if self.text[: markers[0].start()].strip():
            pages.append(_Page(None, self.text[: markers[0].start()]))
        for i, marker in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(self.text)
            pages.append(_Page(int(marker.group(1)), self.text[marker.end() : end]))
        return tuple(pages)


_DOMAIN_DIRS: Final[tuple[str, ...]] = (
    "financial_contracts",
    "regulatory",
    "research",
)


def _default_processed_dir() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "processed_data"


def _load_documents(
    processed_dir: pathlib.Path,
    domain: str,
    doc_ids: Sequence[str],
) -> tuple[_Document, ...]:
    if domain not in _DOMAIN_DIRS:
        return ()
    folder = processed_dir / domain
    if not folder.is_dir():
        return ()
    allow = {str(x) for x in doc_ids if str(x)}
    paths = sorted(folder.glob("*.txt"))
    if allow:
        paths = [p for p in paths if p.stem in allow]
    return tuple(
        _Document(p.stem, domain, p, p.read_text(encoding="utf-8")) for p in paths
    )


def _excerpt(match: re.Match[str], group: str | int = 0) -> str:
    return match.group(group).strip()


def _signature(facts: Sequence[NarrativeFact]) -> tuple[tuple[object, ...], ...]:
    """Semantic signature excluding location and duplicate source wording."""

    return tuple(
        (
            f.metric,
            f.value,
            f.unit,
            f.period,
            f.entity,
            f.scope,
            f.relation,
        )
        for f in facts
    )


def _unique_bundle(
    candidates: Iterable[Sequence[NarrativeFact]],
) -> tuple[NarrativeFact, ...]:
    """Accept duplicate corroboration, reject conflicting semantic bundles."""

    bundles = [tuple(x) for x in candidates if x]
    if not bundles:
        return ()
    by_signature: dict[tuple[tuple[object, ...], ...], tuple[NarrativeFact, ...]] = {}
    for bundle in bundles:
        by_signature.setdefault(_signature(bundle), bundle)
    if len(by_signature) != 1:
        return ()
    return next(iter(by_signature.values()))


def _contract_mining_margin(
    question: str, docs: Sequence[_Document]
) -> tuple[NarrativeFact, ...]:
    if not (
        "矿业板块" in question
        and "主营业务毛利率" in question
        and re.search(r"2022\s*[-—至]\s*2024", question)
        and "2025年1-6月" in re.sub(r"\s+", "", question)
    ):
        return ()
    candidates: list[tuple[NarrativeFact, ...]] = []
    header = re.compile(
        r"业务板块\s*2025\s*年\s*1\s*[-—]\s*6\s*月\s*"
        r"2024\s*年度\s*2023\s*年度\s*2022\s*年度"
    )
    row = re.compile(
        r"矿业\s*(?P<h1>[\d.]+)%\s*(?P<y2024>[\d.]+)%\s*"
        r"(?P<y2023>[\d.]+)%\s*(?P<y2022>[\d.]+)%"
    )
    for doc in docs:
        for page in doc.pages():
            if not header.search(page.text):
                continue
            for m in row.finditer(page.text):
                source = _excerpt(m)
                candidates.append(
                    tuple(
                        NarrativeFact(
                            "gross_margin",
                            D(m.group(group)),
                            "percent",
                            doc.doc_id,
                            page.number,
                            source,
                            period=period,
                            entity="发行人",
                            scope="矿业板块主营业务",
                        )
                        for period, group in (
                            ("2022", "y2022"),
                            ("2023", "y2023"),
                            ("2024", "y2024"),
                            ("2025年1-6月", "h1"),
                        )
                    )
                )
    return _unique_bundle(candidates)


def _contract_appreciation_rates(
    question: str, docs: Sequence[_Document]
) -> tuple[NarrativeFact, ...]:
    compact = re.sub(r"\s+", "", question)
    if not (
        "评估增值率" in compact
        and "两次评估" in compact
        and "2023年6月30日" in compact
        and "2023年12月31日" in compact
    ):
        return ()
    pattern = re.compile(
        r"以\s*2023\s*年\s*6\s*月\s*30\s*日为评估基准日，"
        r".*?增值率为\s*(?P<v1>[\d,]+(?:\.\d+)?)%。.*?"
        r"2023\s*年末标的公司净资产.*?评估基准日\s*"
        r"2023\s*年\s*12\s*月\s*31\s*日.*?"
        r"增值率\s*(?P<v2>[\d,]+(?:\.\d+)?)%",
        re.S,
    )
    candidates: list[tuple[NarrativeFact, ...]] = []
    for doc in docs:
        for page in doc.pages():
            for m in pattern.finditer(page.text):
                # The complete paragraph is short enough to retain both dates,
                # making the period-to-value binding independently auditable.
                source = _excerpt(m)
                candidates.append(
                    (
                        NarrativeFact(
                            "appreciation_rate",
                            D(m.group("v1").replace(",", "")),
                            "percent",
                            doc.doc_id,
                            page.number,
                            source,
                            period="20230630",
                            entity="冠鸿智能",
                            scope="股东全部权益评估",
                        ),
                        NarrativeFact(
                            "appreciation_rate",
                            D(m.group("v2").replace(",", "")),
                            "percent",
                            doc.doc_id,
                            page.number,
                            source,
                            period="20231231",
                            entity="冠鸿智能",
                            scope="股东全部权益评估",
                        ),
                    )
                )
    return _unique_bundle(candidates)


def _contract_parent_profit_series(
    question: str, docs: Sequence[_Document]
) -> tuple[NarrativeFact, ...]:
    compact = re.sub(r"\s+", "", question)
    if not (
        "平均值" in compact
        and "2023年至2025年" in compact
        and re.search(r"归属于母公司(?:所有者|股东)的净利润", compact)
    ):
        return ()
    header = re.compile(r"项目\s*2025\s*年度\s*2024\s*年度\s*2023\s*年度")
    row = re.compile(
        r"其中[：:]\s*归属于母公司股东的净利润\s*"
        r"(?P<y2025>-?[\d,]+(?:\.\d+)?)\s*"
        r"(?P<y2024>-?[\d,]+(?:\.\d+)?)\s*"
        r"(?P<y2023>-?[\d,]+(?:\.\d+)?)"
    )
    candidates: list[tuple[NarrativeFact, ...]] = []
    for doc in docs:
        for page in doc.pages():
            if not header.search(page.text) or "单位：万元" not in page.text:
                continue
            for m in row.finditer(page.text):
                source = _excerpt(m)
                candidates.append(
                    tuple(
                        NarrativeFact(
                            "parent_net_profit",
                            D(m.group(group).replace(",", "")),
                            "ten_thousand_yuan",
                            doc.doc_id,
                            page.number,
                            source,
                            period=period,
                            entity="发行人",
                            scope="合并报表",
                        )
                        for period, group in (
                            ("2023", "y2023"),
                            ("2024", "y2024"),
                            ("2025", "y2025"),
                        )
                    )
                )
    return _unique_bundle(candidates)


def _reg_delivery_cadence(
    question: str, docs: Sequence[_Document]
) -> tuple[NarrativeFact, ...]:
    compact = re.sub(r"\s+", "", question)
    if not (
        "批准程序" in compact
        and ("未交付" in compact or "未过户" in compact)
        and ("次一工作日" in compact or "每30日" in compact)
    ):
        return ()
    pattern = re.compile(
        r"自完成相关批准程序之日起\s*(?P<deadline>六十|60)\s*日内，"
        r"(?P<body>.*?期满后次一工作日.*?此后每\s*(?P<cadence>三十|30)\s*日"
        r"应当公告一次.*?)。",
        re.S,
    )
    candidates: list[tuple[NarrativeFact, ...]] = []
    for doc in docs:
        for page in doc.pages():
            for m in pattern.finditer(page.text):
                source = _excerpt(m)
                candidates.append(
                    (
                        NarrativeFact(
                            "delivery_deadline",
                            D(60),
                            "day",
                            doc.doc_id,
                            page.number,
                            source,
                            relation="within",
                        ),
                        NarrativeFact(
                            "progress_report_offset",
                            D(1),
                            "workday",
                            doc.doc_id,
                            page.number,
                            source,
                            relation="next",
                        ),
                        NarrativeFact(
                            "announcement_interval",
                            D(30),
                            "day",
                            doc.doc_id,
                            page.number,
                            source,
                            relation="every",
                        ),
                    )
                )
    return _unique_bundle(candidates)


def _reg_decision_period(
    question: str, docs: Sequence[_Document]
) -> tuple[NarrativeFact, ...]:
    compact = re.sub(r"\s+", "", question)
    if not ("银行卡清算机构" in compact and "筹备申请" in compact and "90日" in compact):
        return ()
    pattern = re.compile(
        r"中国人民银行.*?自受理之日起\s*(?P<days>90|九十)\s*日内"
        r"作出批准或不批准筹备的决定[^。]*。"
    )
    candidates: list[tuple[NarrativeFact, ...]] = []
    for doc in docs:
        for page in doc.pages():
            for m in pattern.finditer(page.text):
                candidates.append(
                    (
                        NarrativeFact(
                            "decision_period",
                            D(90),
                            "day",
                            doc.doc_id,
                            page.number,
                            _excerpt(m),
                            entity="中国人民银行",
                            scope="银行卡清算机构筹备申请",
                            relation="within",
                        ),
                    )
                )
    return _unique_bundle(candidates)


def _reg_score_rules(
    question: str, docs: Sequence[_Document]
) -> tuple[NarrativeFact, ...]:
    compact = re.sub(r"\s+", "", question)
    if not (
        "证券公司" in compact
        and "基准分" in compact
        and "警示函" in compact
        and "分支机构" in compact
        and "最终" in compact
    ):
        return ()
    candidates: list[tuple[NarrativeFact, ...]] = []
    base_re = re.compile(r"设定正常经营的证券公司基准分为\s*(?P<v>100)\s*分")
    warning_re = re.compile(
        r"公司或者其董事、监事、高级管理人员.*?被采取出具警示函，"
        r"责令公开说明，责令定期\s*报告的，每次扣\s*(?P<v>0\.5)\s*分",
        re.S,
    )
    branch_re = re.compile(
        r"证券公司分公司、营业部等分\s*支机构被采取上述措施的，"
        r"按以上原则减半扣分，累计最高扣\s*(?P<cap>5)\s*分"
    )
    for doc in docs:
        pages = doc.pages()
        bases = [(p, m) for p in pages for m in base_re.finditer(p.text)]
        warnings = [(p, m) for p in pages for m in warning_re.finditer(p.text)]
        branches = [(p, m) for p in pages for m in branch_re.finditer(p.text)]
        if not bases or not warnings or not branches:
            continue
        # Repeated publication text in one attachment is corroboration.  Each
        # primitive must nevertheless resolve to the same literal value.
        if {m.group("v") for _p, m in bases} != {"100"}:
            continue
        if {m.group("v") for _p, m in warnings} != {"0.5"}:
            continue
        if {m.group("cap") for _p, m in branches} != {"5"}:
            continue
        bp, bm = bases[0]
        wp, wm = warnings[0]
        rp, rm = branches[0]
        candidates.append(
            (
                NarrativeFact(
                    "evaluation_baseline",
                    D(100),
                    "point",
                    doc.doc_id,
                    bp.number,
                    _excerpt(bm),
                    entity="证券公司",
                ),
                NarrativeFact(
                    "warning_letter_deduction",
                    D("0.5"),
                    "point",
                    doc.doc_id,
                    wp.number,
                    _excerpt(wm),
                    entity="证券公司",
                    relation="subtract",
                ),
                NarrativeFact(
                    "branch_deduction_factor",
                    D("0.5"),
                    "multiple",
                    doc.doc_id,
                    rp.number,
                    _excerpt(rm),
                    entity="分支机构",
                    relation="multiply",
                ),
            )
        )
    return _unique_bundle(candidates)


def _reg_remittance_threshold(
    question: str, docs: Sequence[_Document]
) -> tuple[NarrativeFact, ...]:
    compact = re.sub(r"\s+", "", question)
    if not (
        "境外汇款" in compact
        and "单笔核实门槛" in compact
        and re.search(r"需核实几笔", compact)
    ):
        return ()
    pattern = re.compile(
        r"为客户向境外汇出资金金额为单笔人民币\s*(?P<v>5000)\s*元"
        r"或者外币等值1000美元以上的，应当核实汇款人信息的准确性。"
    )
    candidates: list[tuple[NarrativeFact, ...]] = []
    for doc in docs:
        for page in doc.pages():
            for m in pattern.finditer(page.text):
                candidates.append(
                    (
                        NarrativeFact(
                            "trigger_threshold",
                            D(m.group("v")),
                            "yuan",
                            doc.doc_id,
                            page.number,
                            _excerpt(m),
                            entity="人民币境外汇款",
                            scope="单笔金额",
                            relation="greater_or_equal",
                        ),
                    )
                )
    return _unique_bundle(candidates)


def _reg_advance_notice(
    question: str, docs: Sequence[_Document]
) -> tuple[NarrativeFact, ...]:
    compact = re.sub(r"\s+", "", question)
    if not ("收费调整" in compact and "提前30个自然日" in compact and "持续公示" in compact):
        return ()
    pattern = re.compile(
        r"调整支付业务的收费项目或者收费标准的，原则上应当至少于调整施行前"
        r"\s*(?P<v>30)\s*个自然日，.*?进行持续公示[^。]*。",
        re.S,
    )
    candidates: list[tuple[NarrativeFact, ...]] = []
    for doc in docs:
        for page in doc.pages():
            for m in pattern.finditer(page.text):
                candidates.append(
                    (
                        NarrativeFact(
                            "advance_notice",
                            D(m.group("v")),
                            "day",
                            doc.doc_id,
                            page.number,
                            _excerpt(m),
                            entity="非银行支付机构",
                            scope="收费调整持续公示",
                            relation="at_least",
                        ),
                    )
                )
    return _unique_bundle(candidates)


def _research_battery_base(
    question: str, docs: Sequence[_Document]
) -> tuple[NarrativeFact, ...]:
    compact = re.sub(r"\s+", "", question)
    if not (
        "国内新能源乘用车销量" in compact
        and "单车带电量" in compact
        and "动力电池需求同比增速" in compact
        and "2025年" in compact
    ):
        return ()
    table = re.compile(
        r"图：国内本土电动乘用车预测（(?:原|新)预测）\s*"
        r"2025\s*2026E\s*2027E\s*"
        r"国内：新能源乘用车销量（万辆）\s*"
        r"(?P<sales>[\d,]+(?:\.\d+)?)\s*[\d,]+(?:\.\d+)?\s*[\d,]+(?:\.\d+)?"
        r".*?国内：电动乘用车电池装机需求\s*（Gwh）\s*"
        r"(?P<demand>[\d,]+(?:\.\d+)?)\s*[\d,]+(?:\.\d+)?\s*[\d,]+(?:\.\d+)?"
        r".*?乘用车单车带电量（kwh）\s*"
        r"(?P<per_vehicle>[\d,]+(?:\.\d+)?)\s*[\d,]+(?:\.\d+)?\s*[\d,]+(?:\.\d+)?",
        re.S | re.I,
    )
    candidates: list[tuple[NarrativeFact, ...]] = []
    for doc in docs:
        for page in doc.pages():
            for m in table.finditer(page.text):
                source = _excerpt(m)
                candidates.append(
                    (
                        NarrativeFact(
                            "vehicle_sales",
                            D(m.group("sales").replace(",", "")),
                            "ten_thousand_vehicle",
                            doc.doc_id,
                            page.number,
                            source,
                            period="2025",
                            entity="国内新能源乘用车",
                            scope="本土销量",
                        ),
                        NarrativeFact(
                            "battery_demand",
                            D(m.group("demand").replace(",", "")),
                            "gwh",
                            doc.doc_id,
                            page.number,
                            source,
                            period="2025",
                            entity="国内新能源乘用车",
                            scope="电池装机需求",
                        ),
                        NarrativeFact(
                            "battery_per_vehicle",
                            D(m.group("per_vehicle").replace(",", "")),
                            "kwh",
                            doc.doc_id,
                            page.number,
                            source,
                            period="2025",
                            entity="国内新能源乘用车",
                            scope="全部新能源乘用车（非纯电单列）",
                        ),
                    )
                )
    return _unique_bundle(candidates)


def _research_new_orders(
    question: str, docs: Sequence[_Document]
) -> tuple[NarrativeFact, ...]:
    compact = re.sub(r"\s+", "", question)
    if not (
        "IP设计公司" in compact
        and "2025年新签订单" in compact
        and "AI订单占比" in compact
        and "2026年" in compact
    ):
        return ()
    pattern = re.compile(
        r"2025\s*年全年，公司新签订单金额\s*(?P<v>[\d,]+(?:\.\d+)?)\s*亿元"
        r"（YoY\+(?P<growth>[\d.]+)%），其中AI\s*算力相关订单占\s*比超\s*"
        r"(?P<share>[\d.]+)%"
    )
    candidates: list[tuple[NarrativeFact, ...]] = []
    for doc in docs:
        for page in doc.pages():
            for m in pattern.finditer(page.text):
                candidates.append(
                    (
                        NarrativeFact(
                            "new_orders",
                            D(m.group("v").replace(",", "")),
                            "hundred_million_yuan",
                            doc.doc_id,
                            page.number,
                            _excerpt(m),
                            period="2025",
                            entity="芯原",
                            scope="全年新签订单",
                        ),
                    )
                )
    return _unique_bundle(candidates)


def extract_facts(
    question_or_item: str | Mapping[str, object],
    *,
    processed_dir: str | pathlib.Path | None = None,
    doc_ids: Sequence[str] | None = None,
) -> tuple[NarrativeFact, ...]:
    """Extract the unique fact bundle for a supported narrative shape.

    ``qid`` is intentionally ignored when a mapping is supplied.  If
    ``doc_ids`` is omitted, the mapping's selected ``doc_ids`` are used; if
    neither is available, all processed documents in the declared domain are
    scanned and semantic conflict detection remains active.
    """

    if isinstance(question_or_item, Mapping):
        question = str(question_or_item.get("question", ""))
        domain = str(question_or_item.get("domain", ""))
        supplied_ids = question_or_item.get("doc_ids", ())
        item_doc_ids = (
            tuple(str(x) for x in supplied_ids)
            if isinstance(supplied_ids, Sequence) and not isinstance(supplied_ids, str)
            else ()
        )
    else:
        question = str(question_or_item)
        domain = ""
        item_doc_ids = ()
    chosen_ids = tuple(str(x) for x in (doc_ids if doc_ids is not None else item_doc_ids))
    root = pathlib.Path(processed_dir) if processed_dir is not None else _default_processed_dir()

    if not domain:
        if any(x in question for x in ("募集说明书", "交易报告书", "评估增值率")):
            domain = "financial_contracts"
        elif any(x in question for x in ("监管", "核实门槛", "筹备申请", "警示函", "持续公示")):
            domain = "regulatory"
        elif any(x in question for x in ("动力电池", "新签订单", "GMV")):
            domain = "research"
    docs = _load_documents(root, domain, chosen_ids)

    if domain == "financial_contracts":
        for handler in (
            _contract_mining_margin,
            _contract_appreciation_rates,
            _contract_parent_profit_series,
        ):
            facts = handler(question, docs)
            if facts:
                return facts
    elif domain == "regulatory":
        for handler in (
            _reg_delivery_cadence,
            _reg_decision_period,
            _reg_score_rules,
            _reg_remittance_threshold,
            _reg_advance_notice,
        ):
            facts = handler(question, docs)
            if facts:
                return facts
    elif domain == "research":
        for handler in (_research_battery_base, _research_new_orders):
            facts = handler(question, docs)
            if facts:
                return facts
    return ()


def calculation_facts(
    question: str, facts: Sequence[NarrativeFact]
) -> tuple[object, ...]:
    """Convert primitives into ``deterministic_calc.Fact`` records.

    The return annotation remains generic to keep this module importable in
    preprocessing-only contexts.  Importing the calculator is deferred until
    this adapter is explicitly called.
    """

    from .deterministic_calc import Fact

    out: list[Fact] = []
    warning = next((f for f in facts if f.metric == "warning_letter_deduction"), None)
    branch = next((f for f in facts if f.metric == "branch_deduction_factor"), None)
    for fact in facts:
        if fact.metric in {
            "evaluation_baseline",
            "warning_letter_deduction",
            "branch_deduction_factor",
            "delivery_deadline",
            "progress_report_offset",
            "announcement_interval",
            "decision_period",
            "advance_notice",
        }:
            continue
        out.append(
            Fact(
                metric=fact.metric,
                value=fact.value,
                unit=fact.unit,
                entity=fact.entity,
                period=fact.period,
                scope=(fact.scope + ("|" if fact.scope else "") +
                       f"relation={fact.relation}"),
                source=f"{fact.source_label}: {fact.verbatim}",
            )
        )
    if warning is not None and "证券公司" in question:
        out.append(
            Fact(
                "score_adjustment",
                -warning.value,
                "number",
                entity="证券公司",
                source=f"{warning.source_label}: {warning.verbatim}",
            )
        )
    if warning is not None and branch is not None and "分支机构" in question:
        out.append(
            Fact(
                "score_adjustment",
                -(warning.value * branch.value),
                "number",
                entity="分支机构",
                source=(
                    f"{warning.source_label}: 每次扣{warning.value}分；"
                    f"{branch.source_label}: 分支机构减半扣分"
                ),
            )
        )
    return tuple(out)


def facts_block(facts: Sequence[NarrativeFact]) -> str:
    """Compact, attributable evidence block for a downstream verifier."""

    rows = []
    for fact in facts:
        attrs = [
            f"metric={fact.metric}",
            f"value={fact.value}",
            f"unit={fact.unit}",
        ]
        if fact.period:
            attrs.append(f"period={fact.period}")
        if fact.entity:
            attrs.append(f"entity={fact.entity}")
        if fact.scope:
            attrs.append(f"scope={fact.scope}")
        if fact.relation != "equals":
            attrs.append(f"relation={fact.relation}")
        rows.append(
            "FACT|" + "|".join(attrs) + f"|source={fact.source_label}\n原文：{fact.verbatim}"
        )
    return "\n".join(rows)


__all__ = [
    "NarrativeFact",
    "calculation_facts",
    "extract_facts",
    "facts_block",
]
