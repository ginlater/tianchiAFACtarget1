"""Zero-token, high-confidence document routing.

The selector is deliberately conservative.  It only looks at the current
question and its options, plus corpus metadata/text-derived identities and a
lexical BM25 index.  It never looks at qid, answers, previous runs or score
feedback.  ``None`` means that the normal Qwen document selector must run.

Public entry point::

    decision = select_docs_fast(question)
    if decision is not None:
        picked, diagnostics = decision

This module is separate from :mod:`agent.doc_select` so it can be evaluated
before being enabled in the production pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import re
import unicodedata
from typing import Iterable, Optional

from . import retrieval
from .paths import PROCESSED_DIR


# A small orthographic fold is needed because the CMB report title is
# traditional Chinese while questions use simplified Chinese.  This is text
# normalisation, not semantic inference.
_CHAR_FOLD = str.maketrans({
    "銀": "银", "國": "国", "際": "际", "科": "科", "技": "技",
    "股": "股", "份": "份", "有": "有", "限": "限", "公": "公", "司": "司",
})
_PUNCT_RE = re.compile(r"[^0-9a-zA-Z一-鿿%]+")


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").translate(_CHAR_FOLD)
    return _PUNCT_RE.sub("", text).lower()


def _qtext(q: dict) -> str:
    """Question surface only.  Deliberately ignores qid and every answer key."""
    opts = q.get("options") or {}
    values = opts.values() if isinstance(opts, dict) else opts
    return (q.get("question") or "") + "\n" + "\n".join(str(v) for v in values)


_CORP_RE = re.compile(
    r"([一-鿿A-Za-z0-9（）()]{2,42}?"
    r"(?:股份有限公司|集团有限公司|有限责任公司|有限公司))")
_SHORT_RE = re.compile(
    r"(?:股票|证券|公司)简称\s*[:：]\s*([A-Za-z0-9一-鿿]{2,16})")
_CORP_SUFFIX_RE = re.compile(r"(?:股份有限公司|集团有限公司|有限责任公司|有限公司)$")
_BAD_ALIAS = {
    "股份有限公司", "有限公司", "集团有限公司", "上市公司",
    "公司", "本公司", "发行人", "上市公司", "标的公司", "目标公司",
    "交易对方", "标的资产", "交易标的", "债务人", "债权人",
    "主承销商", "独立财务顾问", "年度报告", "募集说明书",
}

# Contract documents commonly identify the actual target company in their
# glossary rather than in the cover-page issuer name, for example::
#
#     标的公司、长安银行
#     指 长安银行股份有限公司
#
# Such a definition is a stronger identity signal than a mere body-text
# mention.  Keep this deliberately line-oriented so ordinary prose cannot
# accidentally manufacture aliases.
_DEFINED_COMPANY_RE = re.compile(
    r"(?m)^\s*([^\n]{2,48})\s*\n\s*指\s+"
    r"([^\n]{2,80}?(?:股份有限公司|集团有限公司|有限责任公司|有限公司))"
)


def _clean_company(raw: str) -> str:
    # Regexes can start at a nearby label because Chinese has no word boundary.
    raw = re.sub(r"^(?:标题|股票简称|证券简称|公司简称)[:：]?", "", raw)
    return raw.strip("。；;,: ")


def _company_aliases(company: str, short_names: Iterable[str]) -> tuple[str, ...]:
    company = _clean_company(company)
    base = _CORP_SUFFIX_RE.sub("", company)
    aliases = {company, base}
    for short in short_names:
        short = short.strip()
        # A-share suffixes such as "陕国投A" are not normally used in prose.
        aliases.add(re.sub(r"(?<=[一-鿿])[A-Z]$", "", short, flags=re.I))
        aliases.add(short)
    # A few industry tails follow the actual brand in formal company names.
    # Removing them is generic and lets "宁德时代" match
    # "宁德时代新能源科技股份有限公司".
    for tail in ("新能源科技", "科技", "控股"):
        if base.endswith(tail) and len(base) - len(tail) >= 3:
            aliases.add(base[:-len(tail)])
    return tuple(sorted({_norm(a) for a in aliases
                         if len(_norm(a)) >= 3 and _norm(a) not in _BAD_ALIAS},
                        key=lambda x: (-len(x), x)))


def _defined_company_aliases(text: str) -> tuple[str, ...]:
    """Extract explicit glossary aliases for legally named companies.

    Only aliases coupled to a full legal company name by a visible ``指``
    definition are accepted.  Generic glossary labels are discarded.
    """
    aliases: set[str] = set()
    for lhs, legal_name in _DEFINED_COMPANY_RE.findall(text):
        legal_base = _norm(_CORP_SUFFIX_RE.sub("", legal_name))
        # The left side also contains role labels such as "董事会" and
        # "标的公司".  Accept a short form only when it is literally part of
        # the legal company name; this retains "长安银行" while rejecting
        # generic governance/transaction vocabulary.
        left_aliases = []
        for candidate in re.split(r"[、,，/；;]", lhs):
            norm = _norm(candidate.strip())
            if norm and (norm in legal_base or legal_base in norm):
                left_aliases.append(candidate)
        candidates = left_aliases + [legal_name, _CORP_SUFFIX_RE.sub("", legal_name)]
        for candidate in candidates:
            norm = _norm(candidate.strip())
            if len(norm) >= 3 and norm not in _BAD_ALIAS:
                aliases.add(norm)
    return tuple(sorted(aliases, key=lambda x: (-len(x), x)))


@dataclass(frozen=True)
class _Identity:
    doc_id: str
    domain: str
    company_key: str
    aliases: tuple[str, ...]
    year: Optional[str]
    surface: str


@lru_cache(maxsize=None)
def _identity(doc_id: str) -> _Identity:
    meta = retrieval.docs_meta()[doc_id]
    domain = meta["domain"]
    # Identity fields occur near the front.  The larger report window also
    # handles cover pages made only of artwork (notably China State
    # Construction's reports).
    if domain == "financial_reports":
        head_limit = 50_000
    elif domain == "financial_contracts":
        # Contract glossaries can begin after several cover/notice pages.
        head_limit = 20_000
    else:
        head_limit = 8_000
    head = retrieval.doc_path(doc_id).read_text(encoding="utf-8")[:head_limit]
    title = str(meta.get("title") or "")
    combined = title + "\n" + head
    companies = [_clean_company(x) for x in _CORP_RE.findall(combined)]
    primary = companies[0] if companies else ""
    shorts = _SHORT_RE.findall(combined[:12_000])
    aliases = set(_company_aliases(primary, shorts) if primary else ())
    # A target/counterparty defined in the contract glossary is an explicit
    # document identity too.  Limit this to financial contracts: reports have
    # long abbreviation tables whose incidental company references are not a
    # safe single-report routing signal.
    if domain == "financial_contracts":
        aliases.update(_defined_company_aliases(head[:20_000]))
    company_key = _norm(_CORP_SUFFIX_RE.sub("", primary))
    ym = re.search(r"(20\d{2})\s*年", title + "\n" + head[:800])
    if not ym:
        ym = re.search(r"_(20\d{2})_", doc_id)
    return _Identity(
        doc_id=doc_id,
        domain=domain,
        company_key=company_key or doc_id,
        aliases=tuple(sorted(aliases, key=lambda x: (-len(x), x))),
        year=ym.group(1) if ym else None,
        surface=_norm(combined[:12_000]),
    )


@lru_cache(maxsize=None)
def _domain_ids(domain: str) -> tuple[str, ...]:
    return tuple(d for d, m in retrieval.docs_meta().items()
                 if m["domain"] == domain)


@lru_cache(maxsize=None)
def _bm25(domain: str) -> retrieval.BM25:
    chunks = []
    for doc_id in _domain_ids(domain):
        meta = retrieval.docs_meta()[doc_id]
        text = retrieval.doc_path(doc_id).read_text(encoding="utf-8")[:30_000]
        title = str(meta.get("title") or "")
        chunks.append({"id": doc_id, "doc_id": doc_id, "page": None,
                       "text": (title + "\n") * 3 + text})
    return retrieval.BM25(chunks)


def _quoted_title_pick(q: dict, domain: str) -> Optional[tuple[list[str], dict]]:
    anchors = [a for a in re.findall(r"《([^》]{8,160})》", q.get("question") or "")
               if len(_norm(a)) >= 10]
    if not anchors:
        return None
    picked = []
    matches = {}
    for anchor in anchors:
        na = _norm(anchor)
        hits = [d for d in _domain_ids(domain)
                if na in _identity(d).surface]
        if len(hits) != 1:
            return None
        picked.append(hits[0])
        matches[anchor] = hits[0]
    picked = list(dict.fromkeys(picked))
    return picked, {
        "method": "exact_quoted_title",
        "confidence": 1.0,
        "anchors": matches,
        "fallback": False,
    }


def _matched_company_groups(q: dict, domain: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Return matched company groups and the aliases that triggered them."""
    nq = _norm(_qtext(q))
    groups: dict[str, list[_Identity]] = {}
    for doc_id in _domain_ids(domain):
        ident = _identity(doc_id)
        groups.setdefault(ident.company_key, []).append(ident)
    triggers: dict[str, str] = {}
    docs: dict[str, list[str]] = {}
    for key, identities in groups.items():
        aliases = sorted({a for ident in identities for a in ident.aliases},
                         key=lambda a: (-len(a), a))
        hit = next((a for a in aliases if a in nq), None)
        if hit:
            triggers[key] = hit
            docs[key] = [x.doc_id for x in identities]
    return triggers, docs


_MULTI_DOC_RE = re.compile(
    r"(?:两|三|四|几|多)(?:家|份|款|类|个)[^\n。？?]{0,18}"
    r"(?:公司|产品|文件|报告|机构|募集说明书|合同)")


def _requires_multiple(q: dict) -> bool:
    text = _qtext(q)
    return bool(_MULTI_DOC_RE.search(text) or
                re.search(r"以下三份|各文件|与各文件|两家公司|四家", text))


def _company_pick(q: dict, domain: str) -> Optional[tuple[list[str], dict]]:
    triggers, group_docs = _matched_company_groups(q, domain)
    if not triggers:
        return None
    # A no-option extraction/calculation is assumed to have one source unless
    # its wording explicitly requests multiple documents.  Several matched
    # company groups therefore indicate ambiguity, not permission to guess.
    if not q.get("options") and not _requires_multiple(q) and len(triggers) != 1:
        return None
    years = set(re.findall(r"20\d{2}", _qtext(q)))
    picked = []
    for key in triggers:
        docs = group_docs[key]
        if domain == "financial_reports" and years:
            dated = [d for d in docs if _identity(d).year in years]
            docs = dated or docs
        picked.extend(docs)
    picked = list(dict.fromkeys(picked))
    if _requires_multiple(q) and len(picked) < 2:
        return None
    # More than eight identities is a sign that a generic alias slipped
    # through.  Let Qwen resolve it rather than bloating the evidence context.
    if not picked or len(picked) > 8:
        return None
    return picked, {
        "method": "explicit_company_aliases",
        "confidence": 0.99,
        "matched_groups": len(triggers),
        "aliases": sorted(triggers.values()),
        "years": sorted(years),
        "fallback": False,
    }


@lru_cache(maxsize=1)
def _insurance_catalog() -> dict:
    path = PROCESSED_DIR / "insurance_titles.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _insurance_brand(info: dict) -> str:
    company = info.get("company") or ""
    product = info.get("product") or ""
    if "众安" in company or "众安" in product:
        return "众安"
    if "平安" in company or "平安" in product:
        return "平安"
    if "中国人寿" in company or "国寿" in product:
        return "国寿"
    if "太平洋" in company or "太保" in product:
        return "太保"
    return ""


def _insurance_pick(q: dict) -> Optional[tuple[list[str], dict]]:
    nq = _norm(_qtext(q))
    catalog = _insurance_catalog()
    alias_owners: dict[str, list[str]] = {}
    for doc_id, info in catalog.items():
        for alias in info.get("alias", []):
            alias_owners.setdefault(_norm(alias), []).append(doc_id)

    picked, matches = [], {}
    for doc_id, info in catalog.items():
        brand = _norm(_insurance_brand(info))
        product = _norm(info.get("product") or "")
        aliases = [_norm(x) for x in info.get("alias", []) if len(_norm(x)) >= 2]
        hit = None
        if product and product in nq:
            hit = product
        else:
            for alias in sorted(aliases, key=lambda x: -len(x)):
                owners = alias_owners.get(alias, [])
                branded = brand + alias if brand else ""
                if branded and branded in nq:
                    hit = branded
                    break
                # A unique product alias is safe without a company qualifier.
                if len(owners) == 1 and alias in nq:
                    hit = alias
                    break
                # If the shared alias is genuinely bare (none of its owners'
                # brands occurs next to it), all matching products are needed.
                owner_brands = [_norm(_insurance_brand(catalog[x])) for x in owners]
                if alias in nq and not any((b + alias) in nq for b in owner_brands if b):
                    hit = alias
                    break
        if hit:
            picked.append(doc_id)
            matches[doc_id] = hit
    picked = list(dict.fromkeys(picked))
    if _requires_multiple(q) and len(picked) < 2:
        return None
    if not picked or len(picked) > 10:
        return None
    return picked, {
        "method": "insurance_product_aliases",
        "confidence": 0.995,
        "matched_products": matches,
        "fallback": False,
    }


def _family(doc_id: str) -> str:
    return re.sub(r"_att\d+$", "", doc_id)


def _reg_core_aliases(doc_id: str) -> tuple[str, ...]:
    meta = retrieval.docs_meta()[doc_id]
    title = str(meta.get("title") or "")
    cores = re.findall(r"《([^》]{4,80})》", title)
    if not cores and doc_id.startswith("strict_v3_"):
        cores = re.findall(r"[（(]([^）)]{4,80})[）)]", title)
        if not cores and "中华人民共和国" in title:
            cores = [title.split("_")[-1]]
    aliases = set()
    for core in cores:
        n = _norm(core)
        if len(n) >= 6:
            aliases.add(n)
        short = re.sub(r"(?:监督管理条例实施细则|管理办法|实施细则|暂行规定|规定|规则)$", "", n)
        if len(short) >= 6:
            aliases.add(short)
        for prefix in ("金融机构客户",):
            if short.startswith(prefix) and len(short) - len(prefix) >= 6:
                aliases.add(short[len(prefix):])
    return tuple(sorted(aliases, key=lambda x: (-len(x), x)))


def _reg_representative(family: str) -> str:
    ids = set(_domain_ids("regulatory"))
    att1 = family + "_att1"
    return att1 if att1 in ids else family


def _regulatory_pick(q: dict) -> Optional[tuple[list[str], dict]]:
    nq = _norm(_qtext(q))
    matched: dict[str, list[str]] = {}
    for doc_id in _domain_ids("regulatory"):
        aliases = [a for a in _reg_core_aliases(doc_id) if a in nq]
        if aliases:
            matched.setdefault(_family(doc_id), []).extend(aliases)
    # Regulations are the high-risk domain: more than one plausible title
    # family always goes to Qwen.
    if len(matched) != 1:
        return None
    target_family = next(iter(matched))
    ranked = _bm25("regulatory").search(_qtext(q), k=12)
    family_scores: dict[str, float] = {}
    for chunk, score in ranked:
        fam = _family(chunk["doc_id"])
        family_scores[fam] = max(family_scores.get(fam, 0.0), score)
    order = sorted(family_scores.items(), key=lambda x: -x[1])
    if not order or order[0][0] != target_family:
        return None
    ratio = order[0][1] / max(order[1][1], 1e-9) if len(order) > 1 else float("inf")
    if ratio < 1.40:
        return None
    picked = [_reg_representative(target_family)]
    return picked, {
        "method": "unique_regulation_title_plus_bm25",
        "confidence": round(min(0.99, 0.80 + 0.10 * ratio), 4),
        "matched_aliases": sorted(set(matched[target_family])),
        "bm25_family_margin": round(ratio, 4),
        "fallback": False,
    }


def _strict_bm25_contract_pick(q: dict) -> Optional[tuple[list[str], dict]]:
    """Last-resort route for a single contract with a very large lexical lead."""
    if _requires_multiple(q):
        return None
    ranked = _bm25("financial_contracts").search(_qtext(q), k=3)
    if len(ranked) < 2:
        return None
    top_doc, top_score = ranked[0][0]["doc_id"], ranked[0][1]
    ratio = top_score / max(ranked[1][1], 1e-9)
    # Below this absolute score, an impressive ratio can be caused by a few
    # generic finance terms in an otherwise weak query.  Conversely the 1.65
    # relative margin is only accepted together with this high lexical floor.
    if top_score < 150 or ratio < 1.65:
        return None
    return [top_doc], {
        "method": "strict_contract_bm25_margin",
        "confidence": round(min(0.97, 0.78 + 0.08 * ratio), 4),
        "bm25_top_score": round(top_score, 4),
        "bm25_second_score": round(ranked[1][1], 4),
        "bm25_margin": round(ratio, 4),
        "fallback": False,
    }


def _strict_bm25_single_source_calc(q: dict, domain: str) -> Optional[tuple[list[str], dict]]:
    """Route a no-option calculation only when one source dominates lexically.

    This is useful for research calculations whose narrative deliberately
    omits the report title.  It is not enabled for thematic choice questions,
    where several reports often contribute even when one BM25 score leads.
    """
    # In this dataset, a calculation/extraction question is identified by the
    # absence of options; no type label or answer schema is consulted here.
    if q.get("options"):
        return None
    ranked = _bm25(domain).search(_qtext(q), k=3)
    if len(ranked) < 2:
        return None
    top_doc, top_score = ranked[0][0]["doc_id"], ranked[0][1]
    ratio = top_score / max(ranked[1][1], 1e-9)
    if top_score < 70 or ratio < 1.75:
        return None
    return [top_doc], {
        "method": "strict_single_source_calc_bm25",
        "confidence": round(min(0.97, 0.78 + 0.08 * ratio), 4),
        "bm25_margin": round(ratio, 4),
        "fallback": False,
    }


def _narrative_registry_pick(q: dict) -> Optional[tuple[list[str], dict]]:
    """Use a unique source-bound fact bundle as a fail-closed calc router.

    The narrative registry scans the current domain and accepts a bundle only
    when its visible wording, values and source locations are semantically
    unique.  Routing to those source documents avoids asking Qwen to select a
    regulation from a misleading title card, while the final answer remains a
    charged Qwen calculation/verification.  No qid, answer schema value,
    prior run or reference output is available to the registry.
    """
    if q.get("options") or q.get("answer_format") != "calc" or \
            q.get("domain") not in {
                "financial_contracts", "regulatory", "research"}:
        return None
    from .narrative_fact_registry import extract_facts
    probe = {
        "domain": q.get("domain"),
        "question": q.get("question", ""),
        "options": {},
    }
    facts = extract_facts(probe, doc_ids=())
    if not facts:
        return None
    allowed = set(_domain_ids(str(q.get("domain"))))
    picked = list(dict.fromkeys(str(f.doc_id) for f in facts))
    if not picked or any(doc_id not in allowed for doc_id in picked):
        return None
    return picked, {
        "method": "unique_source_bound_narrative_facts",
        "confidence": 0.995,
        "fact_count": len(facts),
        "metrics": sorted({str(f.metric) for f in facts}),
        "fallback": False,
    }


def select_docs_fast(q: dict) -> Optional[tuple[list[str], dict]]:
    """Return a zero-token document decision, or ``None`` for Qwen fallback.

    The result is invariant to ``q['qid']``: only domain, question and options
    are inspected.  Picks preserve corpus order for reproducibility.
    """
    domain = q.get("domain")
    if domain not in {"financial_contracts", "financial_reports", "insurance",
                      "regulatory", "research"}:
        return None
    narrative = _narrative_registry_pick(q)
    if narrative is not None:
        return narrative
    if domain == "insurance":
        return _insurance_pick(q)
    if domain == "regulatory":
        return _regulatory_pick(q)
    # Cross-report research choice questions intentionally remain with Qwen:
    # their document references are thematic rather than explicit identities.
    if domain == "research":
        return _strict_bm25_single_source_calc(q, domain)
    exact = _quoted_title_pick(q, domain)
    if exact is not None:
        return exact
    company = _company_pick(q, domain)
    if company is not None:
        return company
    if domain == "financial_contracts":
        return _strict_bm25_contract_pick(q)
    return None


__all__ = ["select_docs_fast"]
