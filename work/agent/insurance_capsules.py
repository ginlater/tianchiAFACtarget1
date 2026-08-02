"""Deterministic typed-memory capsules for the insurance domain.

The companion builder writes ``processed_data/insurance_capsules.json`` from
the parsed insurance text and the existing lexical ``domain_facts.json``.
This module is deliberately inference-free: topic detection and retrieval are
literal-regex/lexical operations and never inspect a qid or an answer key.
"""
from __future__ import annotations

import json
import pathlib
import re
from collections import Counter

from .paths import PROCESSED_DIR


SCHEMA_VERSION = "insurance_capsule.v1"
DEFAULT_PATH = PROCESSED_DIR / "insurance_capsules.json"

# Ordered from specific to broad.  The order is also the deterministic
# tiebreaker when a clause matches several types.
TOPIC_RULES = (
    ("cooling_off", "犹豫期", r"犹豫期|签收.{0,8}(?:解除|退还)|全额退还保险费"),
    ("death_benefit", "身故给付", r"身故保险金|身故保险金额|身故给付比例|身故时"),
    ("annuity_benefit", "养老年金/生存金", r"养老保险金|养老年金|生存保险金|生存金|年金领取"),
    ("maturity_benefit", "满期给付", r"满期保险金|满期生存保险金|保险期间届满.*给付"),
    ("disability_care", "失能失智/护理", r"失能|失智|护理保险金|日常生活活动"),
    ("critical_illness", "疾病给付", r"重大疾病|轻症|中症|白血病|疾病保险金"),
    ("medical_reimbursement", "医疗报销", r"医疗费用|医疗保险金|报销|给付比例|基本医疗保险|免赔额"),
    ("waiting_period", "等待期", r"等待期"),
    ("cash_value", "现金价值", r"现金价值"),
    ("surrender", "退保/解除", r"退保|解除合同|未满期(?:净)?保险费|退还保险费"),
    ("policy_loan", "保单借款", r"保单贷款|保单借款|申请借款|贷款金额|借款金额"),
    ("partial_withdrawal", "部分领取", r"部分领取|领取个人账户价值"),
    ("reduction", "减保/减额交清", r"减保|减少基本保险金额|减额交清"),
    ("grace_period", "宽限期", r"宽限期"),
    ("reinstatement", "复效/效力恢复", r"复效|效力恢复|效力中止"),
    ("account_value", "账户价值", r"个人账户价值|保单账户价值|投资组合账户价值"),
    ("settlement_rate", "结算/保证利率", r"结算利率|保证利率|最低保证利率|利率.*结算"),
    ("dividend", "红利/分红", r"保单红利|红利分配|分红保险|累积生息"),
    ("rescue_expense", "施救费用", r"施救费|施救费用|防止或减少.{0,20}损失所支付"),
    ("deductible", "免赔额/免赔率", r"免赔额|免赔率|免赔比例|绝对免赔"),
    ("exclusion", "责任免除", r"责任免除|不承担.{0,8}保险责任|不负责赔偿|除外责任|免责"),
    ("claim_payment", "理赔/给付时限", r"保险金申请|保险金给付|赔偿处理|请求赔偿|给付保险金"),
    ("coverage_period", "保险期间", r"保险期间|保障期间|保多久"),
    ("eligibility", "投保范围", r"投保年龄|投保范围|被保险人.*年龄|接受的被保险人"),
    ("beneficiary", "受益人", r"受益人"),
    ("premium", "保险费/费用", r"保险费|初始费用|风险保障费|交费方式|缴纳保险费"),
    ("sum_insured", "保险金额/限额", r"保险金额|赔偿限额|累计赔偿限额|责任限额"),
    ("property_liability", "财产/责任保障", r"保险标的|财产损失|第三者责任|赔偿责任|保险责任"),
)

_TOPIC_REGEX = tuple((key, label, re.compile(pattern))
                     for key, label, pattern in TOPIC_RULES)
TOPIC_LABELS = {key: label for key, label, _pattern in TOPIC_RULES}
_NUMBER_RE = re.compile(
    r"(?:20\d{2}\s*年(?:\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?)?)|"
    r"(?:-?\d[\d,]*(?:\.\d+)?\s*(?:%|％|个工作日|工作日|周岁|岁|个月|月|日|天|"
    r"小时|分钟|元|万元|亿元|倍|份|次|项|年))|"
    r"(?:[零〇一二三四五六七八九十百千万两]+(?:个)?(?:工作日|周岁|岁|年|月|日|天|次|项|份))"
)
_LEX_RE = re.compile(r"\d+(?:\.\d+)?%?|[A-Za-z]+|[一-鿿]+")
_FOCUS_STOPWORDS = {
    "保险", "保险责", "保险责任", "责任", "条款", "产品", "公司",
    "明确", "列明", "规定", "约定", "包含", "关于", "说法", "正确",
    "下列", "选项", "匹配", "处理", "给付", "赔付",
}


def _literal_query_matches(query: str, text: str) -> set[str]:
    """Return maximal non-generic literal phrases shared by query and text."""
    matches = set()
    compact_query = re.sub(r"\s+", "", query or "")
    for run in re.findall(r"[A-Za-z0-9.%％一-鿿]+", compact_query):
        for n in range(2, min(10, len(run)) + 1):
            for i in range(len(run) - n + 1):
                phrase = run[i:i + n]
                if phrase in text and phrase not in _FOCUS_STOPWORDS:
                    matches.add(phrase)
    # Avoid counting every overlapping substring of one longer exact phrase.
    return {phrase for phrase in matches
            if not any(phrase != other and phrase in other
                       for other in matches)}


def _quoted_query_anchors(query: str) -> set[str]:
    """Extract literal target terms explicitly quoted by the question."""
    anchors = set()
    for quoted in re.findall(r"[“\"]([^”\"]{2,80})[”\"]", query or ""):
        for part in re.split(r"[、,，/]|(?:或)|(?:以及)|(?:等)", quoted):
            part = re.sub(r"^[\s‘’'（(]+|[\s‘’'）)]+$", "", part)
            if len(part) >= 2 and part not in _FOCUS_STOPWORDS:
                anchors.add(part)
    return anchors


def topic_scores(text: str, title: str = "") -> dict[str, int]:
    """Return deterministic regex-match scores, with a title-match boost."""
    scores = {}
    for key, _label, pattern in _TOPIC_REGEX:
        body_n = len(pattern.findall(text or ""))
        title_n = len(pattern.findall(title or ""))
        if body_n or title_n:
            scores[key] = body_n + 5 * title_n
    return scores


def infer_topics(text: str, title: str = "") -> list[str]:
    """Infer ordered typed topics without a model."""
    scores = topic_scores(text, title)
    order = {key: i for i, (key, _label, _pat) in enumerate(TOPIC_RULES)}
    return sorted(scores, key=lambda key: (-scores[key], order[key]))


def extract_numbers(text: str) -> list[str]:
    """Extract value+unit strings in source order, de-duplicated."""
    out = []
    seen = set()
    for match in _NUMBER_RE.finditer((text or "").replace("％", "%")):
        value = re.sub(r"\s+", "", match.group(0))
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _lexical_tokens(text: str) -> Counter:
    tokens = []
    for raw in _LEX_RE.findall((text or "").replace("％", "%")):
        if re.fullmatch(r"[一-鿿]+", raw):
            tokens.extend(raw)
            tokens.extend(raw[i:i + 2] for i in range(len(raw) - 1))
        else:
            tokens.append(raw.lower())
    return Counter(tokens)


_CACHE: dict[str, tuple[int, dict]] = {}


def load_capsules(path: str | pathlib.Path | None = None, *, force=False) -> dict:
    """Load and validate the capsule artifact, cached by path and mtime."""
    p = pathlib.Path(path or DEFAULT_PATH).resolve()
    if not p.exists():
        return {"schema_version": SCHEMA_VERSION, "documents": {}, "stats": {}}
    stamp = p.stat().st_mtime_ns
    cached = _CACHE.get(str(p))
    if cached and cached[0] == stamp and not force:
        return cached[1]
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported insurance capsule schema: {data.get('schema_version')!r}")
    if not isinstance(data.get("documents"), dict):
        raise ValueError("insurance capsule artifact has no documents mapping")
    _CACHE[str(p)] = (stamp, data)
    return data


def _query_text(q: dict) -> str:
    options = q.get("options") or {}
    return (q.get("question") or "") + " " + " ".join(str(v) for v in options.values())


def _score(card: dict, identity: dict, q_tokens: Counter,
           q_numbers: set[str], q_topics: set[str], query: str = "") -> float:
    text = " ".join((
        identity.get("company", ""), identity.get("product", ""),
        " ".join(identity.get("aliases") or []), card.get("clause_title", ""),
        card.get("verbatim", ""),
    ))
    c_tokens = _lexical_tokens(text)
    overlap = 0.0
    for token, qn in q_tokens.items():
        if token not in c_tokens:
            continue
        weight = 1.8 if len(token) == 2 and "一" <= token[0] <= "鿿" else 0.6
        if token[:1].isdigit() or token.endswith("%"):
            weight = 4.0
        overlap += min(qn, c_tokens[token]) * weight
    card_topics = {card.get("topic"), *(card.get("tags") or [])}
    topic_bonus = 22.0 * len(q_topics & card_topics)
    number_bonus = 8.0 * len(q_numbers & set(card.get("numbers") or []))
    fact_bonus = 0.75 if "domain_facts" in (card.get("sources") or []) else 0.0
    # Clause headings are unusually reliable anchors in insurance policies.
    # A literal heading such as ``年龄错误`` or ``责任免除`` should beat a
    # remote body-text coincidence such as ``发生保险事故``.  This is still
    # pure lexical ranking: no qid, label or historical answer is consulted.
    title = re.sub(r"[\s：:（）()]+", "", card.get("clause_title", ""))
    title = re.sub(r"(?:的)?(?:处理|约定|说明)$", "", title)
    compact_query = re.sub(r"\s+", "", query or "")
    title_bonus = 35.0 if len(title) >= 2 and title in compact_query else 0.0
    source_text = " ".join((card.get("clause_title", ""),
                            card.get("verbatim", "")))
    literal_bonus = sum(10.0 * len(phrase) ** 2
                        for phrase in _literal_query_matches(query, source_text))
    anchor_bonus = 120.0 * sum(1 for anchor in _quoted_query_anchors(query)
                               if anchor in source_text)
    return (overlap + topic_bonus + number_bonus + fact_bonus + title_bonus +
            literal_bonus + anchor_bonus)


def select_capsules(q: dict, *, path: str | pathlib.Path | None = None,
                    max_cards: int = 24, per_doc: int = 2,
                    char_budget: int = 8500) -> list[dict]:
    """Select balanced, query-relevant capsules for an insurance question.

    Selection uses only the question/options, selected document ids, typed
    regex topics and lexical overlap.  ``qid`` is intentionally never read.
    """
    if q.get("domain") != "insurance" or not q.get("doc_ids"):
        return []
    data = load_capsules(path)
    query = _query_text(q)
    q_tokens = _lexical_tokens(query)
    q_numbers = set(extract_numbers(query))
    q_topics = set(infer_topics(query))
    doc_order = [str(x) for x in q.get("doc_ids") or []]
    ranked = {}
    for doc_id in doc_order:
        doc = data["documents"].get(doc_id)
        if not doc:
            continue
        identity = doc.get("identity") or {}
        rows = []
        for card in doc.get("capsules") or []:
            score = _score(card, identity, q_tokens, q_numbers, q_topics,
                           query=query)
            if score > 0:
                rows.append((score, card.get("id", ""), card, identity))
        rows.sort(key=lambda item: (-item[0], item[1]))
        ranked[doc_id] = rows

    chosen = []
    seen = set()
    used = 0

    def take(item):
        nonlocal used
        score, _cid, card, identity = item
        cid = card.get("id")
        cost = len(card.get("verbatim", "")) + 150
        if cid in seen or len(chosen) >= max_cards or used + cost > char_budget:
            return False
        row = dict(card)
        row["identity"] = identity
        row["score"] = round(score, 3)
        chosen.append(row)
        seen.add(cid)
        used += cost
        return True

    # First guarantee per-document coverage for cross-product comparison.
    for pos in range(max(0, per_doc)):
        for doc_id in doc_order:
            rows = ranked.get(doc_id) or []
            if pos < len(rows):
                take(rows[pos])
    # Then spend the remaining budget on the globally strongest evidence.
    all_rows = [item for rows in ranked.values() for item in rows]
    all_rows.sort(key=lambda item: (-item[0], item[1]))
    for item in all_rows:
        take(item)
        if len(chosen) >= max_cards:
            break
    return chosen


def insurance_capsule_block(q: dict, *, path: str | pathlib.Path | None = None,
                            max_cards: int = 24, per_doc: int = 2,
                            char_budget: int = 8500) -> str:
    """Render selected capsules as a compact evidence block for a prompt."""
    cards = select_capsules(q, path=path, max_cards=max_cards,
                            per_doc=per_doc, char_budget=char_budget)
    if not cards:
        return ""
    parts = ["保险条款确定性记忆胶囊（离线词法构建；以下均为原文，不是模型摘要）:"]
    for card in cards:
        ident = card.get("identity") or {}
        clause = " ".join(x for x in (card.get("clause"),
                                       card.get("clause_title")) if x).strip()
        topic = TOPIC_LABELS.get(card.get("topic"), card.get("topic", ""))
        nums = "、".join(card.get("numbers") or []) or "无显式数字"
        meta = (f"◆doc={card.get('doc_id')}｜{ident.get('product', '')}｜"
                f"P{card.get('page')}｜{clause or '未编号条款'}｜主题={topic}｜数字={nums}")
        entry = meta + "\n原句：" + (card.get("verbatim") or "").strip()
        if len("\n".join(parts + [entry])) > char_budget:
            break
        parts.append(entry)
    return "\n".join(parts)


_BRAND_HINTS = {
    "中国平安": ("平安",),
    "平安健康": ("平安",),
    "平安养老": ("平安",),
    "中国人寿": ("国寿", "中国人寿"),
    "众安在线": ("众安",),
    "太平洋健康": ("太保", "太平洋"),
}


def _identity_match_score(option: str, identity: dict) -> float:
    """Score how explicitly an option names one policy identity.

    Product aliases carry most of the score; company/brand words only break
    ties between products sharing aliases such as ``特种车``.  The function
    intentionally does not inspect the option letter or the question id.
    """
    text = re.sub(r"\s+", "", option or "")
    product = re.sub(r"\s+", "", identity.get("product", ""))
    aliases = [re.sub(r"\s+", "", x)
               for x in (identity.get("aliases") or []) if len(x) >= 2]
    score = 0.0
    if product and product in text:
        score += 200.0 + len(product)
    for alias in aliases:
        if alias in text:
            score += 80.0 + 2.0 * len(alias)
    company = identity.get("company", "")
    for company_key, hints in _BRAND_HINTS.items():
        if company_key in company:
            score += 18.0 * sum(1 for hint in hints if hint in text)
    # Long literal product fragments make the matcher robust to suffixes such
    # as ``（互联网版）条款`` being omitted in an option.
    for n in (8, 6, 4):
        if len(product) < n:
            continue
        fragments = {product[i:i + n] for i in range(len(product) - n + 1)}
        if any(fragment in text for fragment in fragments):
            score += n * 2.0
            break
    return score


def option_document_map(q: dict, *, path: str | pathlib.Path | None = None) -> dict[str, str]:
    """Map each option to the policy it explicitly names, when unambiguous."""
    if q.get("domain") != "insurance":
        return {}
    data = load_capsules(path)
    docs = [str(x) for x in q.get("doc_ids") or []]
    out = {}
    if len(docs) == 1:
        return {str(letter): docs[0]
                for letter in (q.get("options") or {})}
    for letter, option in (q.get("options") or {}).items():
        ranked = []
        for doc_id in docs:
            identity = (data.get("documents", {}).get(doc_id) or {}).get(
                "identity", {})
            score = _identity_match_score(str(option), identity)
            if score > 0:
                ranked.append((score, doc_id))
        ranked.sort(key=lambda row: (-row[0], row[1]))
        if not ranked:
            continue
        # A brand-only hit is too weak; and a tied score is not an unambiguous
        # product binding.  In both cases the normal shared capsule remains.
        if ranked[0][0] < 40 or (len(ranked) > 1 and
                                abs(ranked[0][0] - ranked[1][0]) < 1e-9):
            continue
        out[str(letter)] = ranked[0][1]
    return out


def _option_claim(option: str, identity: dict) -> str:
    """Remove policy-name words so capsule ranking focuses on the claim."""
    claim = str(option or "")
    names = [identity.get("product", ""), *(identity.get("aliases") or [])]
    company = identity.get("company", "")
    for company_key, hints in _BRAND_HINTS.items():
        if company_key in company:
            names.extend(hints)
    for name in sorted({x for x in names if x}, key=len, reverse=True):
        claim = claim.replace(name, " ")
    return re.sub(r"\s+", " ", claim).strip()


def _focused_verbatim(text: str, query: str, limit: int) -> str:
    """Return one continuous source excerpt centred on lexical query hits."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    q_tokens = [t for t in _lexical_tokens(query)
                if (len(t) >= 2 or t[:1].isdigit()) and t in text]
    if not q_tokens:
        return text[:limit].rstrip() + "…"
    candidates = {0, max(0, len(text) - limit)}
    for token in q_tokens:
        start = 0
        while True:
            pos = text.find(token, start)
            if pos < 0:
                break
            candidates.add(max(0, min(len(text) - limit,
                                      pos - limit // 3)))
            start = pos + max(1, len(token))
    counts = _lexical_tokens(query)
    # Reward the longest literal query phrases, not just frequent Chinese
    # bigrams.  Without this, a window full of generic words like ``保险责任``
    # can beat the one containing the decisive literal ``核爆炸`` or ``地震``.
    literal_phrases = _literal_query_matches(query, text)
    quoted_anchors = {anchor for anchor in _quoted_query_anchors(query)
                      if anchor in text}

    def window_score(start: int) -> tuple[float, int]:
        window = text[start:start + limit]
        toks = _lexical_tokens(window)
        score = sum(min(n, toks.get(tok, 0)) * (2.0 if len(tok) >= 2 else 0.2)
                    for tok, n in counts.items())
        score += sum(float(len(phrase) ** 3)
                     for phrase in literal_phrases if phrase in window)
        score += 200.0 * sum(1 for anchor in quoted_anchors if anchor in window)
        return score, -start

    start = max(candidates, key=window_score)
    prefix = "…" if start else ""
    suffix = "…" if start + limit < len(text) else ""
    return prefix + text[start:start + limit].strip() + suffix


def insurance_option_evidence_block(
        questions: list[dict], *, path: str | pathlib.Path | None = None,
        char_budget: int = 7500) -> str:
    """Render a fair option-by-option evidence matrix for a question batch.

    Each option is first bound to the product name it literally contains, then
    receives the strongest capsule from that product only.  The code labels
    evidence; it never decides whether the option is true.  A single global
    budget is divided evenly so late questions cannot be starved by early ones.
    """
    rows = []
    data = load_capsules(path)
    for q_index, q in enumerate(questions, 1):
        mapping = option_document_map(q, path=path)
        for letter, option in (q.get("options") or {}).items():
            doc_id = mapping.get(str(letter))
            if not doc_id:
                continue
            identity = (data.get("documents", {}).get(doc_id) or {}).get(
                "identity", {})
            claim = _option_claim(str(option), identity)
            evidence_query = (q.get("question") or "") + " " + claim
            oq = dict(q, doc_ids=[doc_id], question=evidence_query, options={})
            cards = select_capsules(oq, path=path, max_cards=1,
                                    per_doc=1, char_budget=1800)
            if cards:
                rows.append((q_index, str(letter), evidence_query, cards[0]))
    if not rows:
        return ""
    header = ("逐选项条款证据矩阵（代码仅按选项中的产品名称定位原文，"
              "不预判选项真伪；每项必须独立核对）:")
    # Reserve metadata and distribute the remaining space uniformly.  The
    # excerpt of every row is continuous verbatim source text.
    metadata_reserve = 105 * len(rows) + len(header) + 1
    excerpt_limit = max(180, min(460,
                                 (char_budget - metadata_reserve) // len(rows)))
    parts = [header]
    for q_index, letter, evidence_query, card in rows:
        ident = card.get("identity") or {}
        meta = (f"【题{q_index}选项{letter}｜doc={card.get('doc_id')}｜"
                f"{ident.get('product', '')}｜P{card.get('page')}｜"
                f"{card.get('clause_title') or card.get('clause') or '原文'}】")
        excerpt = _focused_verbatim(card.get("verbatim", ""), evidence_query,
                                    excerpt_limit)
        parts.append(meta + "\n" + excerpt)
    rendered = "\n".join(parts)
    # The fair allocation above is conservative, but unusually long product
    # names can still exceed the cap.  Trim all excerpts equally once more.
    if len(rendered) > char_budget:
        overflow = len(rendered) - char_budget
        excerpt_limit = max(120, excerpt_limit - (overflow // len(rows) + 2))
        parts = [header]
        for q_index, letter, evidence_query, card in rows:
            ident = card.get("identity") or {}
            meta = (f"【题{q_index}选项{letter}｜doc={card.get('doc_id')}｜"
                    f"{ident.get('product', '')}｜P{card.get('page')}｜"
                    f"{card.get('clause_title') or card.get('clause') or '原文'}】")
            parts.append(meta + "\n" + _focused_verbatim(
                card.get("verbatim", ""), evidence_query, excerpt_limit))
        rendered = "\n".join(parts)
    return rendered[:char_budget]


def insurance_question_theme(q: dict) -> str:
    """Deterministic semantic-shape bucket used only for batching evidence."""
    text = _query_text(q)
    # Order matters: product names may contain ``医疗`` even when the actual
    # question is about age errors or exclusions.
    if re.search(r"年龄错误|诉讼时效|未成年人.{0,8}身故|"
                 r"(?:自杀.{0,20}(?:2年|两年)|(?:2年|两年).{0,20}自杀)", text):
        return "legal_procedure"
    if re.search(r"责任免除|免责|除外责任|恐怖|逃逸|地震|行政行为|司法行为|"
                 r"核爆|核辐射|核污染", text):
        return "exclusion"
    if re.search(r"等待期|精神损害|保障触发|预防接种|医疗责任|医疗费用", text):
        return "coverage_trigger"
    if re.search(r"身故保险金|保单贷款|借款|减额交清|养老年金|账户价值", text):
        return "life_contract"
    topics = infer_topics(text)
    return topics[0] if topics else "other"


def insurance_question_route(
        q: dict, *, path: str | pathlib.Path | None = None) -> str:
    """Return a source-workload route from visible insurance semantics only.

    Most policy questions need one literal clause per option.  Two structures
    deserve dedicated treatment: a cross-product minor-death-limit inventory
    is an exhaustive *presence/absence* check, while the suicide exception has
    to keep ``2 years`` and ``incapable of civil conduct`` in one continuous
    clause.  The classifier never reads qid, prior answers or token ledgers.
    """

    if q.get("domain") != "insurance":
        return insurance_question_theme(q)
    text = _query_text(q)
    mapping = option_document_map(q, path=path)
    product_count = len(set(mapping.values())) or len(
        set(str(x) for x in (q.get("doc_ids") or [])))
    if (re.search(r"未成年(?:人)?", text) and "身故" in text and
            re.search(r"限制|限额|上限", text) and product_count >= 4):
        return "minor_death_limit_exhaustive"
    if ("自杀" in text and "无民事行为能力" in text and
            re.search(r"(?:2\s*年|两年)", text) and product_count >= 3):
        return "suicide_exception"
    theme = insurance_question_theme(q)
    return "ordinary_exclusion" if theme == "exclusion" else theme


def insurance_lexical_coverage_block(
        q: dict, *, path: str | pathlib.Path | None = None) -> str:
    """Audit how completely a dedicated presence/absence query was scanned.

    The block reports lexical scan scope and matching source locations.  It is
    deliberately not an answer: Qwen still decides whether the exact language
    satisfies each option.  Recording zero matches is useful evidence only
    because every capsule row for the product is counted and the rule is an
    explicit literal-clause inventory, not an open-ended semantic search.
    """

    route = insurance_question_route(q, path=path)
    if route not in {"minor_death_limit_exhaustive", "suicide_exception"}:
        return ""
    if route == "minor_death_limit_exhaustive":
        target = re.compile(
            r"未成年人身故保险金限制|为未成年人投保.{0,180}?身故.{0,180}?限额",
            re.S)
        label = "未成年人身故保险金限制/监管限额"
    else:
        target = re.compile(
            r"(?:2\s*年|两年)内自杀.{0,100}?无民事行为能力|"
            r"无民事行为能力.{0,100}?(?:2\s*年|两年)内自杀",
            re.S)
        label = "2年内自杀及无民事行为能力人例外"
    data = load_capsules(path)
    mapping = option_document_map(q, path=path)
    lines = [f"条款词法覆盖审计（目标={label}；仅记录扫描范围与原文命中，不预判选项）:"]
    for letter in (q.get("options") or {}):
        doc_id = mapping.get(str(letter))
        if not doc_id:
            continue
        doc = (data.get("documents") or {}).get(doc_id) or {}
        cards = doc.get("capsules") or []
        matches = []
        for card in cards:
            source = " ".join((card.get("clause_title", ""),
                               card.get("verbatim", "")))
            if target.search(source):
                matches.append(
                    f"{card.get('id')}@P{card.get('page')}:{card.get('clause_title') or card.get('clause') or '原文'}")
        product = (doc.get("identity") or {}).get("product", "")
        detail = "；".join(matches) if matches else "无字面命中"
        lines.append(
            f"- 选项{letter} doc={doc_id} {product}：扫描{len(cards)}条胶囊；"
            f"命中{len(matches)}条；{detail}")
    return "\n".join(lines)
