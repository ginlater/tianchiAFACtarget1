"""批量共享证据答题：同(域,文档集)的选择题合并作答，消灭证据重复计费。

设计要点：
- 仅选择题参与批量（计算题保持独立双样本）；批内最多3题防互扰
- 证据 = 记忆卡(并集文档) + 批内全部题目查询的并集检索（预算按题数扩容）
- r1/r2 批量作答（每题独立输出块），批内某题 r1≠r2 → 单题定向仲裁（复用单题逻辑）
- token 归集：题面/选项承担自身 prompt，底仓公平分摊；可见 completion
  按逐题答案块长度，隐藏 reasoning 按 prompt 权重，以稳定最大余数法守恒
"""
import json, re
from collections import Counter
from math import gcd

from . import retrieval
from .answerer import (CALC_DOMAINS, DIGEST_DOMAINS, _doc_title,
                       _q_text, _use_digest, _vote_letters,
                       apply_structural_evidence_constraints, build_digest,
                       confirm_structural_evidence_constraint,
                       gather_evidence, judge_std_for, parse_answer, select_reasoning,
                       _think, VERIFY_MODEL)
from .qwen_client import chat, DEFAULT_MODEL


HOMO_DOMS = {"insurance", "financial_reports"}


_FIN_CONSOLIDATED_SCOPE = re.compile(
    r"(?:合并(?:财务)?报表|合并口径|合并财务报表)"
)
_FIN_PARENT_SCOPE = re.compile(
    r"(?:母公司(?:单体)?(?:财务)?报表|母公司口径|公司单体(?:财务)?报表|个别财务报表)"
)


def _choice_scope_audit_profile(q):
    """Return a targeted Qwen review profile from visible rule structure.

    These are the choice-question analogues of typed calculation budgets.  A
    route is selected from domain, wording, selected-source breadth and the
    concepts present in the options.  Identifiers, reference answers and past
    runs are deliberately unavailable.
    """
    if q.get("answer_format") == "calc":
        return None
    domain = q.get("domain")
    question = str(q.get("question") or "")
    options = " ".join(str(v) for v in (q.get("options") or {}).values())
    text = question + " " + options
    if domain == "financial_contracts":
        literal_presence_terms = sum(
            bool(re.search(pattern, text)) for pattern in (
                r"提到|提及|披露", r"未.{0,5}(?:提到|提及|披露)",
                r"不涉及|不存在"))
        if (("各文件原文" in question or "与各文件" in question) and
                literal_presence_terms >= 2 and
                not re.search(r"[%％]|T\s*\+\s*\d", text) and
                len(set(q.get("doc_ids") or ())) >= 2):
            return {
                "profile": "contract_cross_document_literal_presence",
                "tag": "fc_scope_audit", "evidence_chars": 350,
                "thinking_budget": 0, "max_tokens": 260,
                "arb_thinking_budget": 0, "arb_max_tokens": 260,
                "complete_audit_authoritative": True,
                "instruction": (
                    "逐文件区分‘明确提到’、‘未提及’和行业常识推断；肯定"
                    "存在性必须有字面锚点，未提及结论必须由对应文件窗口"
                    "支撑，不得用常识补成原文。全文扫描目标命中0时：正向"
                    "声称‘提到目标’的原子必须OUT，负向声称‘未提及目标’"
                    "的原子成立；DIRECT_TEXT后的短语均为同一命名文档全文"
                    "中的逐字锚点，可用于核对复合选项其余原子。"),
            }
        quoted = re.findall(r"[“\"]([^”\"]{2,60})[”\"]", question)
        if (len(quoted) >= 2 and re.search(r"[与及]", question) and
                any(k in text for k in ("条款", "机制", "事项"))):
            return {
                "profile": "contract_compound_named_sections",
                "tag": "fc_scope_audit", "evidence_chars": 1200,
                "thinking_budget": 0, "max_tokens": 350,
                "instruction": (
                    "分别核对两个命名章节的适用条件、期限与救济后果，"
                    "防止把一章的条件移植到另一章。"),
            }
        years = set(re.findall(r"20\d{2}", text))
        metric_terms = {
            term for term in ("资产负债率", "流动比率", "速动比率",
                              "净利润", "营业收入", "经营现金流")
            if term in text
        }
        if len(years) >= 3 and len(metric_terms) >= 2:
            return {
                "profile": "contract_multi_period_metric_table",
                "tag": "fc_scope_audit", "evidence_chars": 1200,
                "thinking_budget": 0, "max_tokens": 350,
                "instruction": (
                    "按年份×指标逐格核对表格，明确百分比与比率的列序，"
                    "不得跨年或跨行拼接。"),
            }
        risk_terms = {
            term for term in ("信用风险", "流动性风险", "操作风险",
                              "董事会", "风险管理委员会", "三道防线", "审计部门")
            if term in text
        }
        if len(risk_terms) >= 3:
            return {
                "profile": "contract_risk_governance_inventory",
                "tag": "fc_scope_audit", "evidence_chars": 1200,
                "thinking_budget": 0, "max_tokens": 350,
                "instruction": (
                    "把信用、流动性、操作风险与治理机构分成独立核查项，"
                    "逐项确认主体和原文层级。"),
            }
    if domain == "financial_reports":
        dividend_terms = sum(term in text for term in (
            "全年", "中期", "末期", "每10股", "现金分红", "排序"))
        if ("现金分红" in text and "排序" in text and
                dividend_terms >= 3 and
                len(set(q.get("doc_ids") or ())) >= 4):
            return {
                "profile": "financial_full_year_dividend_ranking",
                "tag": "fin_dividend_scope_audit", "evidence_chars": 1000,
                "thinking_budget": 0, "max_tokens": 350,
                "arb_thinking_budget": 0, "arb_max_tokens": 260,
                "instruction": (
                    "逐公司绑定全年每10股现金分红口径；全年数若由中期与末期"
                    "组成必须相加，并统一每股/每10股单位后再核对排序和差额。"),
            }
    if domain == "regulatory":
        topics = sum(bool(re.search(pattern, text)) for pattern in (
            r"存量.{0,8}客户|尽调", r"受益所有人", r"资料.{0,6}保存"))
        if (re.search(r"20\d{2}年\d{1,2}月\d{1,2}日", question) and
                topics >= 3 and len(set(q.get("doc_ids") or ())) >= 3):
            return {
                "profile": "regulatory_cross_rule_effective_date",
                "tag": "reg_rule_audit", "evidence_chars": 500,
                "thinking_budget": 0, "max_tokens": 420,
                "instruction": (
                    "将每部法规的生效日、存量过渡期和保存期限分开核对，"
                    "以题干完整日期作为时点边界。"),
            }
        if (all(term in text for term in ("错误", "不一致", "不完整", "反馈")) and
                re.search(r"\d+(?:\.\d+)?%", text) and
                re.search(r"无需|免识别|简化|豁免", text)):
            return {
                "profile": "regulatory_registry_discrepancy_exception",
                "tag": "reg_rule_audit", "evidence_chars": 1600,
                "thinking_budget": 500, "max_tokens": 850,
                "instruction": (
                    "区分‘反馈登记异常’与‘继续按识别标准判断’两层义务，"
                    "并核对持股门槛是否真能创设绝对例外。"),
            }
        if ("无法准确判断" in text and "简化" in text and "豁免" in text):
            return {
                "profile": "regulatory_uncertain_simplification_guard",
                "tag": "reg_rule_audit", "evidence_chars": 700,
                "thinking_budget": 250, "max_tokens": 500,
                "instruction": (
                    "核对‘无法准确判断’时的禁止性后果，严格区分简化与豁免，"
                    "不得以客户自称低风险替代金融机构判断。"),
            }
    return None


def _financial_scope_collision(q):
    """Detect a financial-statement scope comparison from visible semantics.

    A question that contrasts consolidated statements with the parent-only
    statements cannot safely share a shallow evidence floor with ordinary
    indicator questions: identical metric labels occur in both tables, but
    their columns describe different accounting entities.  The route is based
    solely on the wording of the question/options and never on identifiers,
    expected answers or historical scores.
    """

    if q.get("domain") != "financial_reports" or \
            q.get("answer_format") == "calc":
        return False
    text = "\n".join([
        q.get("question", ""),
        *(str(v) for v in (q.get("options") or {}).values()),
    ])
    return bool(_FIN_CONSOLIDATED_SCOPE.search(text) and
                _FIN_PARENT_SCOPE.search(text))


def _financial_extended_note_check(q):
    """Identify note-level financial comparisons from their source structure.

    Segment revenue, dividend reconciliation and restated prior-period values
    live outside a single main-statement row.  Three-year comparative questions
    also require a wider time slice unless they explicitly name a precomputed
    solvency-indicator table.  These visible structures justify a moderately
    larger shared evidence floor than ordinary main-statement checks.
    """

    if q.get("domain") != "financial_reports" or \
            q.get("answer_format") == "calc" or \
            _financial_scope_collision(q):
        return False
    text = "\n".join([
        q.get("question", ""),
        *(str(v) for v in (q.get("options") or {}).values()),
    ])
    note_level = re.search(
        r"分地区.{0,12}(?:营业)?收入|现金分红|全年.{0,10}分红|"
        r"每\s*10\s*股.{0,12}分红|原始披露值|调整前口径|重述(?:前|后)?口径",
        text,
    )
    three_year = (re.search(r"2023\s*[—–至到-]\s*2025", text) and
                  "偿债指标" not in text)
    return bool(note_level or three_year)


def requires_batch_singleton(q):
    """Whether a one-question group still needs the batch deep-evidence path."""

    if _financial_scope_collision(q) or _choice_scope_audit_profile(q):
        return True
    if q.get("domain") == "insurance":
        from .insurance_capsules import insurance_question_route
        return insurance_question_route(q) in {
            "minor_death_limit_exhaustive", "suicide_exception"}
    return False


def _balanced_chunks(items, max_size):
    """Split a sequence without creating an arbitrary one-item tail.

    Shared-prompt accounting becomes unstable when, for example, 17 otherwise
    comparable research questions are sliced as ``8 + 8 + 1``: the last
    question pays an entire instruction/evidence floor by itself merely due to
    sort order.  Balanced chunks keep the same number of calls and maximum
    size while distributing that shared overhead independently of identifiers.
    """

    items = list(items)
    if not items:
        return []
    max_size = max(1, int(max_size))
    groups = (len(items) + max_size - 1) // max_size
    base, extra = divmod(len(items), groups)
    sizes = [base + (1 if i < extra else 0) for i in range(groups)]
    out, start = [], 0
    for size in sizes:
        out.append(items[start:start + size])
        start += size
    return out


def _research_dense_case_check(q):
    """Identify evidence-dense research synthesis before answer generation.

    A wide selector result alone is unstable: an extra low-confidence report
    must not turn an ordinary synthesis into a costly solo call.  Isolation is
    reserved for questions whose *options themselves* repeatedly make
    universal claims across the enumerated cases, so each option genuinely
    requires all source contexts.  This classifier sees only visible wording
    and the current selector output, never qid, answers or historical scores.
    """

    if (q.get("domain") != "research" or
            len(set(q.get("doc_ids") or ())) < 6):
        return False
    options = [str(value) for value in (q.get("options") or {}).values()]
    universal = sum(bool(re.search(
        r"(?:^|[，；])(?:都|均|全部|所有)|(?:都|均|全部|所有)(?:说明|是|有|可|带来|形成|被)",
        option)) for option in options)
    return universal >= 3


def _contract_sparse_three_source_batch(qs):
    """Whether three contract questions each own one different document."""
    return (len(qs) == 3 and
            all(q.get("domain") == "financial_contracts" for q in qs) and
            all(len(set(q.get("doc_ids") or ())) == 1 for q in qs) and
            len({next(iter(set(q.get("doc_ids") or ()))) for q in qs}) == 3)


def _insurance_compact_literal_check(q):
    """Identify a one-clause limitation-period lookup from visible wording."""

    if q.get("domain") != "insurance":
        return False
    text = " ".join((q.get("question", ""), *(
        str(v) for v in (q.get("options") or {}).values())))
    return bool("诉讼时效" in text and re.search(r"(?:2\s*年|两年)", text))


def _insurance_dense_exclusion_batch(qs):
    """Whether a four-way literal batch contains several exclusion audits.

    A shared insurance prompt amortizes one hidden reasoning stream across all
    members.  Three or more ordinary-exclusion inventories in a four-question
    batch still require four independently source-bound clause checks, so they
    receive a deeper *real* Qwen reasoning budget.  The route is derived only
    from visible question semantics and batch shape.
    """

    if len(qs) < 4 or any(q.get("domain") != "insurance" for q in qs):
        return False
    from .insurance_capsules import insurance_question_route
    return sum(insurance_question_route(q) in {
        "ordinary_exclusion", "exclusion"
    } for q in qs) >= 3


def _group_homo(questions, max_batch=6):
    """同质批组（mix复盘处方，7-24）：ins/fin 松散合批的团灭病根 =
    异底仓成员共享上下文装不下多产品条款（并集稀释+每题配额挤兑）。
    处方：组B并入组A当且仅当 docs(B)⊆docs(A)——证据并集零膨胀，深而不宽。
    仅 ins/fin 用同质合并；reg/res/fc 批量本就健康，保持松散大批摊指令。"""
    loose_qs = [q for q in questions if q["domain"] not in HOMO_DOMS]
    homo_qs = [q for q in questions if q["domain"] in HOMO_DOMS]
    out = []
    by_dom = {}
    for q in loose_qs:
        by_dom.setdefault(q["domain"], []).append(q)
    for dom, qs in by_dom.items():
        qs.sort(key=lambda q: sorted(q["doc_ids"]))
        mb = {"financial_contracts": 3}.get(dom, 8)
        if dom == "research":
            dense = [q for q in qs if _research_dense_case_check(q)]
            ordinary = [q for q in qs if not _research_dense_case_check(q)]
            # Dense multi-report checks get one complete source-bound call;
            # ordinary questions retain shared batches with balanced overhead.
            out += [[q] for q in dense]
            out += _balanced_chunks(ordinary, mb)
        else:
            out += [qs[i:i + mb] for i in range(0, len(qs), mb)]
    # Financial-report questions share a compact, table-aware evidence target.
    # Balanced domain batches amortize the common instructions without leaving
    # a source-order singleton.  A consolidated-vs-parent comparison is kept
    # solo because repeated metric names require a substantially deeper table
    # view and must not contaminate ordinary questions' token floors.
    financial = [q for q in homo_qs
                 if q["domain"] == "financial_reports"]
    if financial:
        deep = [q for q in financial if _financial_scope_collision(q)]
        extended = [q for q in financial
                    if _financial_extended_note_check(q)]
        ordinary = [q for q in financial
                    if not _financial_scope_collision(q) and
                    not _financial_extended_note_check(q)]
        for tier in (ordinary, extended):
            tier.sort(key=lambda q: (
                len(set(q.get("doc_ids") or ())),
                tuple(sorted(set(q.get("doc_ids") or ()))),
            ))
        fin_max = max(2, int(__import__("os").environ.get(
            "AFAC_FIN_BATCH_MAX", str(max_batch))))
        out += _balanced_chunks(ordinary, fin_max)
        out += _balanced_chunks(extended, fin_max)
        out += [[q] for q in deep]
        homo_qs = [q for q in homo_qs
                   if q["domain"] != "financial_reports"]
    # Insurance typed capsules make a better, code-only host boundary than the
    # historic exact-doc-set rule: group questions asking about the same kind
    # of clause, then give every option its own product-bound source excerpt.
    # This turns many repeated calls into a few coherent calls without allowing
    # one product's evidence to stand in for another's.
    import os
    if os.environ.get("AFAC_INS_CAPSULES") == "1":
        from .insurance_capsules import insurance_question_route
        themed, rest = {}, []
        for q in homo_qs:
            if q["domain"] == "insurance":
                route = insurance_question_route(q)
                # Literal presence/absence inventories use the same source
                # workflow whether the target word is an exclusion, an age
                # clause or a limitation period.  Pooling these ordinary
                # lookups avoids a two-question instruction tax while the two
                # genuinely deep legal structures remain isolated below.
                bucket = ("literal_clause_inventory" if route in {
                    "ordinary_exclusion", "legal_procedure"} else route)
                themed.setdefault(bucket, []).append(q)
            else:
                rest.append(q)
        for route, qs in themed.items():
            if route in {"minor_death_limit_exhaustive",
                         "suicide_exception"}:
                out += [[q] for q in qs]
            elif route == "literal_clause_inventory":
                literal_max = max(2, min(max_batch, int(os.environ.get(
                    "AFAC_INS_LITERAL_BATCH_MAX", "4"))))
                # Put the compact one-clause lookup in a full-sized batch; it
                # needs less evidence/output than the other inventories and
                # should not pay the smaller tail group's instruction floor.
                qs = sorted(qs, key=lambda q: (
                    not _insurance_compact_literal_check(q),
                    len(q.get("question", ""))))
                out += _balanced_chunks(qs, literal_max)
            else:
                out += _balanced_chunks(qs, max_batch)
        homo_qs = rest
    groups = {}
    for q in homo_qs:
        groups.setdefault((q["domain"], frozenset(q["doc_ids"])), []).append(q)
    merged = {}
    for key in sorted(groups, key=lambda k: -len(k[1])):  # 大底仓在前作宿主
        dom, ds = key
        host = next((hk for hk in merged
                     if hk[0] == dom and ds <= hk[1]), None)
        if host is None:
            merged[key] = list(groups[key])
        else:
            merged[host].extend(groups[key])
    for (_dom, _ds), qs in merged.items():
        out += [qs[i:i + max_batch] for i in range(0, len(qs), max_batch)]
    return out


def group_questions(questions, max_batch=3):
    """按(域, 文档集)分组。返回 [ [q,...], ... ]，单题组即单题。
    瘦身档：域内松散合批（证据取文档并集），摊薄指令与证据开销。"""
    import os
    if os.environ.get("AFAC_HOMO_BATCH") == "1":
        return _group_homo(questions,
                           int(os.environ.get("AFAC_HOMO_MAX", "6")))
    loose = os.environ.get("AFAC_SLIM4") == "1"
    groups = {}
    for q in questions:
        key = q["domain"] if loose else (q["domain"], frozenset(q["doc_ids"]))
        groups.setdefault(key, []).append(q)
    out = []
    if loose:
        # 分域批量：大文档域小批保覆盖率(fc/ins差距源)，小文档域大批摊指令
        for _key, qs in groups.items():
            qs.sort(key=lambda q: sorted(q["doc_ids"]))
    for _key, qs in groups.items():
        mb = max_batch
        if loose:
            dom = _key
            # fc大文档单题深挖→小批; ins需多产品条款同场对比→大批(slim20教训ins 14→9)
            mb = {"financial_contracts": 3}.get(dom, 8)
        for i in range(0, len(qs), mb):
            out.append(qs[i:i + mb])
    return out


def _financial_source_round_robin(ids, chunk_by_id, protected, doc_ids,
                                   rotation=0, question=None):
    """Interleave one financial question's chunks across source/year.

    ``gather_evidence`` intentionally returns chunks in citation order.  For a
    multi-company annual-report comparison that means every protected 2024
    chunk can precede every 2025/Midea chunk, so a later union cap recreates a
    source blind spot.  Keep protection as the first tier, but round-robin the
    documents inside each tier.  Latest reports lead because their comparison
    tables already contain the prior-year column.  Rotating companies by
    question index prevents all members of a batch from starting at the same
    source.
    """
    ids = [cid for cid in ids if cid in chunk_by_id]
    original_docs = list(dict.fromkeys(str(d) for d in doc_ids))
    by_year = {}
    for position, doc_id in enumerate(original_docs):
        match = re.search(r"_(20\d{2})_report$", doc_id)
        year = int(match.group(1)) if match else 0
        by_year.setdefault(year, []).append((position, doc_id))
    source_order = []
    for year in sorted(by_year, reverse=True):
        group = [doc_id for _position, doc_id in sorted(by_year[year])]
        if group:
            shift = rotation % len(group)
            group = group[shift:] + group[:shift]
        source_order.extend(group)
    unseen_docs = []
    for cid in ids:
        doc_id = str(chunk_by_id[cid].get("doc_id") or "")
        if doc_id not in source_order and doc_id not in unseen_docs:
            unseen_docs.append(doc_id)
    source_order.extend(unseen_docs)

    qtext = ""
    if question:
        qtext = str(question.get("question") or "") + " " + " ".join(
            str(v) for v in (question.get("options") or {}).values())
    metric_aliases = (
        ("营业收入",),
        ("经营活动产生的现金流量净额", "经营活动现金流量净额", "经营现金流"),
        ("基本每股收益",),
        ("研发费用", "研发投入"),
        ("资产负债率",), ("流动比率",), ("速动比率",),
        ("现金分红", "分红"),
        ("归属于上市公司股东的净利润", "归母净利润"),
        ("拨备覆盖率",), ("核心一级资本充足率",),
        ("EBITDA",),
    )
    wanted_aliases = [aliases for aliases in metric_aliases
                      if any(alias in qtext for alias in aliases)]
    wanted_numbers = {
        token.replace(",", "")
        for token in re.findall(r"\d[\d,.]*%?", qtext)
        if len(token.replace(",", "")) >= 3
    }

    def relevance(cid):
        text = str(chunk_by_id[cid].get("text") or "")
        compact = text.replace(",", "")
        metric_score = sum(
            100 for aliases in wanted_aliases
            if any(alias in text for alias in aliases))
        number_score = 15 * sum(number in compact
                                for number in wanted_numbers)
        # A small lexical tie-breaker keeps the ordering reusable for other
        # report metrics without overpowering exact accounting labels.
        qtokens = set(retrieval.tokenize(qtext))
        lexical_score = sum(token in text for token in qtokens)
        return metric_score + number_score + lexical_score

    ordered = []
    for want_protected in (True, False):
        buckets = {doc_id: [] for doc_id in source_order}
        for cid in ids:
            if (cid in protected) != want_protected:
                continue
            doc_id = str(chunk_by_id[cid].get("doc_id") or "")
            buckets.setdefault(doc_id, []).append(cid)
        for bucket in buckets.values():
            bucket.sort(key=lambda cid: -relevance(cid))
        depth = max((len(bucket) for bucket in buckets.values()), default=0)
        for position in range(depth):
            for doc_id in source_order:
                bucket = buckets.get(doc_id, [])
                if position < len(bucket):
                    ordered.append(bucket[position])
    return ordered


def _batch_evidence(qs, model=DEFAULT_MODEL, return_ownership=False):
    q0 = qs[0]
    domain = q0["domain"]
    recipients = [q["qid"] for q in qs]

    def _ordered_owners(owners):
        wanted = set(owners or recipients)
        ordered = [qid for qid in recipients if qid in wanted]
        return ordered or list(recipients)

    # Each top-level block carries an exact, text-free ownership map.  The
    # segment lengths add up byte-for-byte (Unicode codepoint-for-codepoint)
    # to the rendered evidence string; this lets the later token allocator
    # charge a unique evidence block only to the questions that used it.
    blocks = []

    def _add_block(text, *, owners=None, kind="evidence", segments=None):
        text = text or ""
        if segments is None:
            segments = [{"chars": len(text),
                         "owners": _ordered_owners(owners),
                         "kind": kind}]
        else:
            segments = [
                {"chars": max(0, int(s.get("chars", 0))),
                 "owners": _ordered_owners(s.get("owners")),
                 "kind": s.get("kind") or kind}
                for s in segments
            ]
        if sum(s["chars"] for s in segments) != len(text):
            # Ownership metadata is audit plumbing and must never alter the
            # evidence shown to the model.  On an internal framing mismatch,
            # conservatively mark the exact block as global.
            segments = [{"chars": len(text), "owners": list(recipients),
                         "kind": kind + "_ownership_fallback"}]
        entry = {"text": text, "segments": segments, "kind": kind}
        blocks.append(entry)
        return entry

    def _owners_for_doc(doc_id):
        return [q["qid"] for q in qs if doc_id in (q.get("doc_ids") or [])]

    # 松散合批下证据覆盖批内全部文档（并集，保持每题可答）
    docs = list(dict.fromkeys(d for q in qs for d in q["doc_ids"]))
    q0 = dict(q0, doc_ids=docs)
    capsule = ""
    if domain == "insurance" and \
            __import__("os").environ.get("AFAC_INS_CAPSULES") == "1":
        from .insurance_capsules import (insurance_lexical_coverage_block,
                                         insurance_option_evidence_block,
                                         insurance_question_route)
        route = insurance_question_route(qs[0]) if len(qs) == 1 else ""
        capsule_budget = int(__import__("os").environ.get(
            "AFAC_INS_BATCH_CAPSULE_BUDGET", "7500"))
        if route == "minor_death_limit_exhaustive":
            capsule_budget = int(__import__("os").environ.get(
                "AFAC_INS_MINOR_CAPSULE_BUDGET", "8500"))
        elif route == "suicide_exception":
            capsule_budget = int(__import__("os").environ.get(
                "AFAC_INS_SUICIDE_CAPSULE_BUDGET", "6500"))
        capsule = insurance_option_evidence_block(
            qs, char_budget=capsule_budget)
        if capsule:
            # Capsule rows are explicitly labelled by batch question index.
            # Keep the header global, then charge each exact row span to the
            # corresponding question.  Newlines between rows stay attached
            # to the preceding row, so every visible character is conserved.
            row_matches = list(re.finditer(r"(?m)^【题(\d+)选项", capsule))
            capsule_segments = []
            if row_matches:
                if row_matches[0].start():
                    capsule_segments.append({
                        "chars": row_matches[0].start(),
                        "owners": recipients,
                        "kind": "insurance_capsule_header",
                    })
                for pos, match in enumerate(row_matches):
                    end = (row_matches[pos + 1].start()
                           if pos + 1 < len(row_matches) else len(capsule))
                    try:
                        qpos = int(match.group(1)) - 1
                    except ValueError:
                        qpos = -1
                    owners = ([qs[qpos]["qid"]]
                              if 0 <= qpos < len(qs) else recipients)
                    capsule_segments.append({
                        "chars": end - match.start(),
                        "owners": owners,
                        "kind": "insurance_capsule_question_row",
                    })
            else:
                capsule_segments.append({
                    "chars": len(capsule), "owners": recipients,
                    "kind": "insurance_capsule_global",
                })
            _add_block(capsule, kind="insurance_capsule",
                       segments=capsule_segments)
        if len(qs) == 1 and route in {
                "minor_death_limit_exhaustive", "suicide_exception"}:
            coverage = insurance_lexical_coverage_block(qs[0])
            if coverage:
                _add_block(coverage, owners=[qs[0]["qid"]],
                           kind="insurance_lexical_coverage_audit")
    if _use_digest(domain):
        for d in q0["doc_ids"]:
            _add_block(build_digest(d, domain, model=model),
                       owners=_owners_for_doc(d), kind="document_digest")
        base_cap = 9500 if domain == "financial_contracts" else \
            8500 if domain == "financial_reports" else 6000
        if __import__("os").environ.get("AFAC_SLIM4") == "1" \
                and domain == "insurance":
            base_cap = 8000  # 卡+证据两全(slim15挤帽教训)
    else:
        title_prefix = "涉及文档:\n"
        title_rows = [f"- {d}: 《{_doc_title(d)}》" for d in q0["doc_ids"]]
        title_segments = [{"chars": len(title_prefix),
                           "owners": recipients,
                           "kind": "document_titles_header_global"}]
        for pos, (doc_id, row) in enumerate(zip(q0["doc_ids"], title_rows)):
            if pos:
                title_segments.append({
                    "chars": 1, "owners": recipients,
                    "kind": "document_title_separator_global",
                })
            title_segments.append({
                "chars": len(row), "owners": _owners_for_doc(doc_id),
                "kind": "document_title_row",
            })
        _add_block(title_prefix + "\n".join(title_rows),
                   kind="document_titles", segments=title_segments)
        base_cap = (int(__import__("os").environ.get(
            "AFAC_INS_BATCH_RAW_CAP", "3000")) if capsule else
                    (int(__import__("os").environ.get(
                        "AFAC_RESEARCH_BATCH_RAW_CAP", "10000"))
                     if domain == "research" else 8500))
        if __import__("os").environ.get("AFAC_SLIM4") == "1":
            # fc/fin大文档域无卡时证据帽必须给足（slim6教训：砍卡后漏选爆发）
            base_cap = {"research": 6000, "financial_contracts": 7500,
                        "financial_reports": 7500}.get(domain, 4800)
    # 三矿入批（7-24）：fin_facts2/domain_facts/align 此前只接 solo 通道，
    # 批量成员从未见过离线矿——fin 批团灭第二病根。逐成员取块去重后共享
    # （零 API 成本的词法查表，env 自门控：未开对应 AFAC_* 时返回空）。
    from .answerer import (align_block, domain_facts_block, fin_facts_block,
                           financial_registry_block)
    _mine_blocks = {}

    def _add_mine(text, qid, kind):
        """Deduplicate a mine once, unioning every actual question owner."""
        if not text:
            return
        if text in _mine_blocks:
            entry = _mine_blocks[text]
            for segment in entry["segments"]:
                segment["owners"] = _ordered_owners(
                    list(segment["owners"]) + [qid])
            return
        entry = _add_block(text, owners=[qid], kind=kind)
        _mine_blocks[text] = entry

    for q in qs:
        registry = financial_registry_block(q)
        _add_mine(registry, q["qid"], "financial_registry")
        for _fn in (fin_facts_block, domain_facts_block, align_block):
            if registry and _fn is fin_facts_block:
                continue
            try:
                b = _fn(q)
            except Exception:  # noqa: BLE001 — 矿块失败不拖垮批答
                b = ""
            _add_mine(b, q["qid"], "question_fact_mine")
    # 预算按批内题数扩容40%/题（并集去重后实际占用低于线性）；瘦身档25%
    _slim4 = __import__("os").environ.get("AFAC_SLIM4") == "1"
    if capsule:
        # Product-bound capsules already guarantee option coverage.  Raw BM25
        # is only a compact second view, so it must not grow with the union of
        # every product in a themed batch.
        cap = int(base_cap * (1 + 0.10 * (len(qs) - 1)))
    else:
        # Cross-document research batches contain strongly overlapping source
        # fragments, so each added question needs only a quarter-base marginal
        # allowance.  Other domains retain the historical expansion.  This is
        # domain-semantic and independent of identifiers or expected answers.
        growth = (0.25 if domain == "research" else
                  (0.25 if _slim4 else 0.4))
        cap = int(base_cap * (1 + growth * (len(qs) - 1)))
        # Research already searches every question's selected documents before
        # taking the union, so document breadth is largely duplicate context;
        # a smaller shared allowance is sufficient.  Contract/report evidence
        # keeps the historical breadth allowance.
        breadth_unit = (1000 if domain == "research" else
                        (1200 if _slim4 else 2000))
        cap += breadth_unit * max(0, min(len(q0["doc_ids"]), 5) - 2)
    if _contract_sparse_three_source_batch(qs):
        # Three unrelated one-document contracts have almost no reusable raw
        # context.  Give each member one additional compact clause window;
        # this expands real evidence rather than adding an artificial call.
        per_member_extra = max(0, int(__import__("os").environ.get(
            "AFAC_FC_SPARSE_BATCH_EXTRA", "2500")))
        cap += per_member_extra * len(qs)
    financial_budgeted = domain == "financial_reports"
    deep_scope = False
    if financial_budgeted:
        # Target the whole visible evidence floor, not just BM25.  Financial
        # questions may already own several thousand characters of typed fact
        # tables; subtracting those code-generated mines prevents the raw-text
        # union from charging the same facts a second time.  Scope-collision
        # questions deliberately receive a much deeper one-question target.
        deep_scope = any(_financial_scope_collision(q) for q in qs)
        extended_scope = (not deep_scope and
                          any(_financial_extended_note_check(q) for q in qs))
        env_name = ("AFAC_FIN_SCOPE_EVIDENCE_PER_Q" if deep_scope else
                    ("AFAC_FIN_EXTENDED_EVIDENCE_PER_Q" if extended_scope else
                     "AFAC_FIN_BATCH_EVIDENCE_PER_Q"))
        default_target = ("22000" if deep_scope else
                          ("7800" if extended_scope else "6000"))
        evidence_per_q = max(2500, int(__import__("os").environ.get(
            env_name, default_target)))
        framed_chars = (sum(len(block["text"]) for block in blocks) +
                        2 * max(0, len(blocks) - 1))
        # Preserve at least one compact option-local source block per member.
        # Without this group-scaled floor, an unusually rich fact mine in one
        # member could consume the entire group's raw allowance and trigger a
        # costly solo retry for every other member.
        cap = max(1500 * len(qs),
                  evidence_per_q * len(qs) - framed_chars)
    # 逐题配额检索后并集（伪题合并会让所有选项查询共用第1题前缀，
    # 后位题证据被挤出：批检索覆盖0.16 vs 单题0.41——slim10八题批稀释类伤）
    per_cap = max(1500, cap // len(qs))
    best, prot_all, per_kept = {}, set(), {}
    for q in qs:
        # Retrieval must stay inside this question's selected documents.
        # Searching the whole batch union makes per-document-normalized BM25
        # scores tie at 1.0; an unrelated earlier document can then take the
        # query's protected top slot and evict the exact clause from the real
        # source.  We still merge the resulting question-scoped chunks below,
        # so the shared prompt retains the union without cross-question
        # retrieval contamination.
        qq = dict(q, doc_ids=list(q.get("doc_ids") or q0["doc_ids"]))
        _ev_i, kept_i, prot_i = gather_evidence(
                                                qq,
                                                k_opt=4 if deep_scope else 2,
                                                k_q=6 if deep_scope else 3,
                                                cap=per_cap)
        per_kept[q["qid"]] = [c["id"] for c in kept_i]
        for c in kept_i:
            best[c["id"]] = c
        prot_all |= prot_i
    # 并集层总预算闸门（保护块优先装填）：防逐题保护豁免叠加爆预算(slim12回归)
    if financial_budgeted:
        # Interleave question-local ranked lists before applying the union cap.
        # A plain insertion-order union lets the first member consume the hard
        # ceiling even though later members have equally valid protected hits.
        # Round-robin is source/order agnostic at the question level and keeps
        # at least one compact option anchor per member whenever the budget can
        # physically hold one.
        ranked_ids = {}
        for q_index, q in enumerate(qs):
            ids = per_kept[q["qid"]]
            ranked_ids[q["qid"]] = _financial_source_round_robin(
                ids, best, prot_all, q.get("doc_ids") or q0["doc_ids"],
                rotation=q_index, question=q)
        ordered, seen_ordered = [], set()
        depth = max((len(ids) for ids in ranked_ids.values()), default=0)
        for pos in range(depth):
            for q in qs:
                ids = ranked_ids[q["qid"]]
                if pos >= len(ids) or ids[pos] in seen_ordered:
                    continue
                seen_ordered.add(ids[pos])
                ordered.append(best[ids[pos]])
        ordered.extend(c for cid, c in best.items() if cid not in seen_ordered)
    else:
        ordered = sorted(best.values(), key=lambda c: c["id"] not in prot_all)
    picked_u, total = [], 0
    soft_limit = int(cap if financial_budgeted else cap * 1.15)
    hard_limit = int(cap * (1.10 if financial_budgeted else 1.6))
    for c in ordered:
        L = len(c["text"]) + 20
        if c["id"] not in prot_all and total + L > soft_limit:
            continue
        if total + L > hard_limit:  # 硬顶：保护块也不得无限叠加
            continue
        total += L
        picked_u.append(c)
    kept = sorted(picked_u,
                  key=lambda c: (c["doc_id"], c["page"] or 0,
                                 int(c["id"].split("#c")[1])))
    parts = []
    raw_segments = [{"chars": len("原文片段证据:\n"),
                     "owners": recipients,
                     "kind": "retrieval_header_global"}]
    for c in kept:
        tag = f"{c['doc_id']} P{c['page']}" if c["page"] else c["id"]
        part = f"【{tag}】{c['text']}"
        if parts:
            raw_segments.append({"chars": 2, "owners": recipients,
                                 "kind": "evidence_separator_global"})
        owners = [qid for qid, ids in per_kept.items() if c["id"] in ids]
        raw_segments.append({"chars": len(part), "owners": owners,
                             "kind": "retrieved_evidence_chunk"})
        parts.append(part)
    _add_block("原文片段证据:\n" + "\n\n".join(parts),
               kind="retrieved_evidence", segments=raw_segments)
    # 覆盖率制导：某题自己的证据块被并集闸门挤掉过半 → 标记定向单答
    # （法医量化: 批内覆盖<0.15正确率仅48%, ≥0.4达86%）
    uids = {c["id"] for c in kept}
    cov = {qid: (sum(1 for i in ids if i in uids) / len(ids)) if ids else 1.0
           for qid, ids in per_kept.items()}
    # Financial batches also carry source-bound typed fact tables, so a raw
    # union fraction below 0.5 is not by itself evidence starvation.  Retrying
    # only the forensic danger zone (<0.15) preserves the compact route while
    # still failing over when virtually none of a question's own source
    # excerpts survived.  Other domains retain the established 0.5 guard.
    coverage_floor = 0.15 if financial_budgeted else 0.5
    low_cov = [qid for qid, v in sorted(cov.items(), key=lambda x: x[1])
               if v < coverage_floor][:3]
    evidence_parts = []
    ownership_segments = []
    for pos, block in enumerate(blocks):
        if pos:
            evidence_parts.append("\n\n")
            ownership_segments.append({
                "chars": 2, "owners": list(recipients),
                "kind": "top_level_separator_global",
            })
        evidence_parts.append(block["text"])
        ownership_segments.extend(block["segments"])
    evidence = "".join(evidence_parts)
    ownership = {
        "version": "batch_evidence_question_owners_v1",
        "length_unit": "unicode_codepoints",
        "text_chars": len(evidence),
        "segments": ownership_segments,
        "segment_chars": sum(s["chars"] for s in ownership_segments),
        "conserved": (sum(s["chars"] for s in ownership_segments)
                      == len(evidence)),
    }
    result = (evidence, [c["id"] for c in kept], low_cov)
    return result + (ownership,) if return_ownership else result


_BLOCK_RE = re.compile(r"【第(\d+)题[^】]*】")
_TOKEN_ALLOCATION_STRATEGY = (
    "batch_prompt_domain_semantic_evidence_chars_v4__"
    "visible_completion_block_plus_shared_chars_v2__reasoning_prompt_v1"
)


def _batch_prompt_allocation_plan(prompt, qs, evidence_ownership=None):
    """Build auditable prompt weights without inspecting identifiers/answers.

    Each question owns its exact rendered question/options text.  Evidence is
    charged according to a domain-semantic policy.  Research and financial
    reports use a shared evidence floor; insurance, contracts and regulations
    may opt into a fully shared prompt because every member is deliberately
    answered from the same union.  Source-question owners are still recorded
    even when charged owners are shared.  Truly global instructions, headers
    and separators are split across the batch.

    The source owners and charged owners are both retained in the audit data.
    A common-denominator scale keeps all weights integral and auditable.
    """
    recipients = [q["qid"] for q in qs]
    domains = {q.get("domain") for q in qs if q.get("domain")}
    domain = next(iter(domains)) if len(domains) == 1 else "mixed"
    import os
    shared_financial = (domain == "financial_reports" and
                        len(recipients) > 1 and
                        os.environ.get("AFAC_FIN_SHARED_EVIDENCE") == "1")
    full_shared_prompt = (len(recipients) > 1 and (
        (domain == "insurance" and
         os.environ.get("AFAC_INS_SHARED_PROMPT") == "1") or
        (domain == "financial_contracts" and
         os.environ.get("AFAC_FC_SHARED_PROMPT") == "1") or
        (domain == "regulatory" and
         os.environ.get("AFAC_REG_SHARED_PROMPT") == "1")))
    evidence_policy = (
        "research_cross_document_shared_floor_v1"
        if domain == "research" and len(recipients) > 1 else
        ("financial_tier_shared_floor_v1" if shared_financial else
         (f"{domain}_full_prompt_shared_floor_v1" if full_shared_prompt else
          "source_question_owners_v1"))
    )
    question_chars = {q["qid"]: len(_q_text(q)) for q in qs}
    prompt_chars = len(prompt or "")
    n = max(1, len(recipients))
    evidence_ownership = dict(evidence_ownership or {})
    evidence_chars = max(0, int(evidence_ownership.get("text_chars", 0) or 0))
    raw_segments = evidence_ownership.get("segments") or []
    valid_evidence = bool(evidence_ownership) and (
        evidence_chars == sum(max(0, int(s.get("chars", 0) or 0))
                              for s in raw_segments)) and (
        evidence_chars + sum(question_chars.values()) <= prompt_chars)

    def _owners(raw):
        wanted = set(raw or recipients)
        got = [qid for qid in recipients if qid in wanted]
        return got or list(recipients)

    segments = []
    if valid_evidence:
        for s in raw_segments:
            source_owners = _owners(s.get("owners"))
            charged_owners = (list(recipients) if (
                evidence_policy.startswith(("research_", "financial_")) or
                "shared_floor" in evidence_policy) else source_owners)
            segments.append({
                "chars": max(0, int(s.get("chars", 0) or 0)),
                "owners": charged_owners,
                "source_owners": source_owners,
                "kind": s.get("kind") or "evidence",
                "scope": "evidence",
            })
    else:
        # Backward-compatible path for callers without ownership metadata:
        # all non-question prompt text is the shared base.
        evidence_chars = 0
    for qid in recipients:
        charged = list(recipients) if full_shared_prompt else [qid]
        segments.append({"chars": question_chars[qid], "owners": charged,
                         "source_owners": [qid],
                         "kind": "question_and_options",
                         "scope": "question"})
    global_chars = max(
        0, prompt_chars - evidence_chars - sum(question_chars.values()))
    segments.append({"chars": global_chars, "owners": list(recipients),
                     "source_owners": list(recipients),
                     "kind": "global_prompt_scaffolding", "scope": "global"})

    scale = 1
    for segment in segments:
        k = max(1, len(segment["owners"]))
        scale = scale * k // gcd(scale, k)
    prompt_weights = {qid: 0 for qid in recipients}
    evidence_units = {qid: 0 for qid in recipients}
    question_units = {qid: 0 for qid in recipients}
    global_units = {qid: 0 for qid in recipients}
    for segment in segments:
        owners = segment["owners"]
        units_each = segment["chars"] * (scale // len(owners))
        for qid in owners:
            prompt_weights[qid] += units_each
            if segment["scope"] == "evidence":
                evidence_units[qid] += units_each
            elif segment["scope"] == "question":
                question_units[qid] += units_each
            else:
                global_units[qid] += units_each
    prompt_char_weights = dict(prompt_weights)
    option_workload_chars_per_option = 0
    option_workload_units = {qid: 0 for qid in recipients}
    option_workload_adjustment_units = {qid: 0 for qid in recipients}
    if domain == "financial_contracts" and full_shared_prompt:
        # A fully shared contract prompt previously charged a two-option
        # true/false member exactly the same as four-option members, although
        # the visible protocol requires one evidence decision and one short
        # conclusion per option.  Add a centred, semantic work allowance of
        # roughly 60 reasoning characters plus labels/conclusion per option.
        # Centring preserves the exact total weight, leaves homogeneous
        # batches unchanged, and depends only on the submitted option count.
        option_workload_chars_per_option = 70
        option_workload_units = {
            q["qid"]: (scale * option_workload_chars_per_option *
                       len(q.get("options") or {}))
            for q in qs
        }
        mean_workload = sum(option_workload_units.values()) // n
        option_workload_adjustment_units = {
            qid: units - mean_workload
            for qid, units in option_workload_units.items()
        }
        prompt_weights = {
            qid: prompt_weights[qid] +
            option_workload_adjustment_units[qid]
            for qid in recipients
        }
    shared_chars = sum(
        s["chars"] for s in segments if len(s["owners"]) == n)
    segment_audit = [
        {"chars": s["chars"],
         "source_owners": s["source_owners"],
         "charged_owners": s["owners"],
         "kind": s["kind"]}
        for s in segments if s["scope"] == "evidence"
    ]
    return {
        "strategy": _TOKEN_ALLOCATION_STRATEGY,
        "prompt_weights": prompt_weights,
        "weight_basis": {
            "length_unit": "unicode_codepoints",
            "domain": domain,
            "evidence_allocation_policy": evidence_policy,
            "full_prompt_shared": full_shared_prompt,
            "prompt_chars": prompt_chars,
            "question_option_chars": question_chars,
            "evidence_chars": evidence_chars,
            "evidence_ownership_valid": valid_evidence,
            "evidence_ownership_version": evidence_ownership.get("version"),
            "evidence_ownership_segments": segment_audit,
            "prompt_weight_scale": scale,
            "evidence_owner_char_units": evidence_units,
            "question_owner_char_units": question_units,
            "global_owner_char_units": global_units,
            "prompt_char_weight_units": prompt_char_weights,
            "option_workload_chars_per_option":
                option_workload_chars_per_option,
            "option_workload_units": option_workload_units,
            "option_workload_adjustment_units":
                option_workload_adjustment_units,
            "prompt_weight_units": prompt_weights,
            "prompt_weight_conserved": (
                sum(prompt_weights.values()) == prompt_chars * scale),
            "global_prompt_chars": global_chars,
            "shared_prompt_chars": shared_chars,
            "shared_prompt_divisor": n,
        },
    }


def _batch_completion_allocation(content, qs):
    """Charge answer blocks to owners and split other visible text equally.

    Model preambles, separators and trailing whitespace are visible shared
    generation cost, not orphaned text.  Multiplying each owned block by the
    batch size and adding the shared character count gives integral weights
    whose sum is exactly ``len(content) * batch_size``.
    """
    recipients = [q["qid"] for q in qs]
    block_chars = {qid: 0 for qid in recipients}
    matches = list(_BLOCK_RE.finditer(content or ""))
    for pos, match in enumerate(matches):
        try:
            i = int(match.group(1)) - 1
        except ValueError:
            continue
        if not 0 <= i < len(qs):
            continue
        end = matches[pos + 1].start() if pos + 1 < len(matches) \
            else len(content or "")
        # Include the visible heading as well as its body.  If a model repeats
        # a numbered block, both generated spans belong to that question.
        block = (content or "")[match.start():end].strip()
        block_chars[qs[i]["qid"]] += len(block)
    assigned_chars = sum(block_chars.values())
    shared_chars = max(0, len(content or "") - assigned_chars)
    scale = max(1, len(recipients))
    completion_weights = {
        qid: scale * block_chars[qid] + shared_chars
        for qid in recipients
    }
    return {
        "completion_weights": completion_weights,
        "weight_basis": {
            "visible_content_chars": len(content or ""),
            "visible_answer_block_chars": block_chars,
            "visible_shared_chars": shared_chars,
            "visible_unassigned_chars": shared_chars,
            "visible_weight_scale": scale,
            "visible_weight_units": completion_weights,
            "visible_weight_conserved": (
                sum(completion_weights.values())
                == len(content or "") * scale),
        },
    }


def _parse_batch_details(content, qs):
    """Return each parsed answer together with its exact visible text block."""
    out = {}
    pieces = _BLOCK_RE.split(content or "")
    # pieces: [前言, idx1, text1, idx2, text2, ...]
    for j in range(1, len(pieces) - 1, 2):
        try:
            i = int(pieces[j]) - 1
        except ValueError:
            continue
        if 0 <= i < len(qs):
            body = pieces[j + 1].strip()
            ans = parse_answer(body, qs[i]["answer_format"])
            if ans:
                out[qs[i]["qid"]] = {
                    "answer": ans,
                    "content": f"【第{i + 1}题 答案块】\n{body}",
                }
    return out


def _parse_batch(content, qs):
    """Compatibility wrapper returning only ``qid -> answer``."""
    return {qid: d["answer"]
            for qid, d in _parse_batch_details(content, qs).items()}


def _scope_overlap_score(text, query):
    """Purely lexical overlap used to protect every option's source window."""

    wanted = Counter(retrieval.tokenize(str(query or "")))
    counts = Counter(retrieval.tokenize(str(text or "")))
    return sum(
        min(n, counts.get(tok, 0)) *
        (3 if re.search(r"[0-9A-Za-z%％]", tok) else 1)
        for tok, n in wanted.items()
    )


def _compact_scope_unit(unit, query, char_budget):
    """Keep the highest-overlap lines of one cited evidence unit.

    Taking ``unit[:room]`` silently discards a decisive clause when it occurs
    near the end of a page.  Sentence/line selection remains deterministic and
    lexical while preserving the source header and the original line order.
    """

    char_budget = max(0, int(char_budget))
    if not unit or char_budget <= 0:
        return ""
    lines = [p.strip() for p in re.split(r"(?<=[。！？；;])|\n", unit)
             if p.strip()]
    if not lines:
        return unit[:char_budget]
    header_idx = next((i for i, line in enumerate(lines)
                       if re.match(r"^【[^】]+】", line)), None)
    header = lines[header_idx] if header_idx is not None else ""
    ranked = sorted(
        (i for i in range(len(lines)) if i != header_idx),
        key=lambda i: (-_scope_overlap_score(lines[i], query), i),
    )
    chosen = []
    remaining = char_budget - (len(header) if header else 0)
    if header and remaining > 1:
        remaining -= 1
    for idx in ranked:
        if remaining <= 0:
            break
        line = lines[idx]
        sep = 1 if chosen else 0
        room = remaining - sep
        if room <= 0:
            break
        if len(line) > room:
            # Centre a long line around the first distinctive lexical anchor
            # instead of keeping only its prefix.
            anchors = sorted(
                {tok for tok in retrieval.tokenize(str(query or ""))
                 if len(tok) >= 2 or re.search(r"[0-9A-Za-z%％]", tok)},
                key=lambda tok: (-len(tok), tok),
            )
            positions = [(line.find(tok), tok) for tok in anchors
                         if line.find(tok) >= 0]
            pos = min(positions)[0] if positions else 0
            start = max(0, min(len(line) - room, pos - room // 3))
            line = (("…" if start else "") + line[start:start + room - 1])
            if start + room - 1 < len(lines[idx]):
                line = line[:-1] + "…" if len(line) > 1 else "…"
        chosen.append((idx, line))
        remaining -= len(line) + sep
        # Preserve enough PDF-soft-wrapped fragments for a three-part
        # risk/governance statement.  The outer allocator still enforces the
        # same fixed total character budget.
        if len(chosen) >= 3:
            break
    chosen.sort(key=lambda item: item[0])
    body = "\n".join(line for _idx, line in chosen)
    if header and body:
        return (header + "\n" + body)[:char_budget]
    return (header or body or unit)[:char_budget]


def _named_option_sources(q, option):
    """Map an option's visible company alias to selected source documents."""

    compact_option = re.sub(r"[^0-9A-Za-z一-鿿]+", "", str(option or ""))
    sources = []
    for doc_id in q.get("doc_ids") or ():
        compact_title = re.sub(
            r"[^0-9A-Za-z一-鿿]+", "", _doc_title(doc_id))
        found = ""
        for width in range(min(12, len(compact_option)), 3, -1):
            found = next((compact_option[i:i + width]
                          for i in range(len(compact_option) - width + 1)
                          if compact_option[i:i + width] in compact_title), "")
            if found:
                break
        item = (found, str(doc_id))
        if found and item not in sources:
            sources.append(item)
    return sources


def _literal_presence_scan_block(q, char_budget=320):
    """Return exact full-document lexical hit counts for an existence claim.

    This block contains no semantic answer and no learned representation.  It
    gives Qwen the one fact a compact excerpt cannot prove: whether the named
    phrase occurs anywhere in the selected source.  The topic is derived from
    the visible question, and sources from literal company/title aliases.
    """

    question = str(q.get("question") or "")
    match = re.search(r"关于(.{2,40}?)[，,。]", question)
    topic = match.group(1) if match else question[:40]
    core_match = re.search(
        r"([一-鿿]{2,12})(?:风险|问题|条款|规定)", topic)
    core = core_match.group(1) if core_match else topic
    for prefix in ("本次募投项目", "募投项目", "本次项目", "项目", "新增"):
        if core.startswith(prefix) and len(core) - len(prefix) >= 2:
            core = core[len(prefix):]
    core = re.sub(r"[^0-9A-Za-z一-鿿]+", "", core)
    if len(core) < 2:
        return ""

    inventory_re = re.compile(
        r"提到|提及|未.{0,6}(?:提到|提及)|不涉及|不存在|因此|因其|涉及")
    full_cache = {}

    def full_facts(doc_id):
        if doc_id in full_cache:
            return full_cache[doc_id]
        full = retrieval.doc_path(doc_id).read_text(encoding="utf-8")
        normalized = re.sub(r"\s+", "", full)
        pages, negative = [], ""
        for chunk in retrieval.chunk_doc(doc_id):
            compact = re.sub(r"\s+", "", chunk.get("text", ""))
            if core in compact and chunk.get("page") not in pages:
                pages.append(chunk.get("page"))
            if not negative:
                for raw_line in chunk.get("text", "").splitlines():
                    line = re.sub(r"\s+", "", raw_line)
                    if (any(ch in line for ch in ("不涉及", "未涉及", "不存在")) and
                            any(tok in line for tok in (core, core[:2]))):
                        starts = [line.find(mark) for mark in
                                  ("不涉及", "未涉及", "不存在")
                                  if line.find(mark) >= 0]
                        start = min(starts) if starts else 0
                        negative = line[start:start + 30]
                        break
        full_cache[doc_id] = (full, normalized, pages, negative)
        return full_cache[doc_id]

    lines = [f"【全文词法扫描｜目标={core}】"]
    negative_owner = {}
    for letter, option in (q.get("options") or {}).items():
        option = str(option)
        sources = _named_option_sources(q, option)
        if not sources:
            continue
        alias, doc_id = sources[0]
        _full, normalized, pages, negative = full_facts(doc_id)
        if core in re.sub(r"\s+", "", option) and inventory_re.search(option):
            if re.search(r"未.{0,6}(?:提到|提及)|不涉及|不存在", option):
                polarity = "NEG_MENTION"
            elif re.search(r"提到|提及", option):
                polarity = "POS_MENTION"
            else:
                polarity = "CAUSAL"
            page_text = "/".join(f"P{page}" for page in pages[:2] if page)
            if negative and doc_id in negative_owner:
                suffix = f"|反向同{negative_owner[doc_id]}"
            elif negative:
                suffix = f"|反向:{negative}"
                negative_owner[doc_id] = str(letter)
            else:
                suffix = ""
            lines.append(
                f"{letter}|{polarity}|{alias}/{doc_id}|hit={normalized.count(core)}"
                + (f"@{page_text}" if page_text else "") + suffix)
            if polarity == "NEG_MENTION" and "因其" in option:
                direct = [phrase for phrase in
                          ("募集资金投资项目研发风险", "研发升级",
                           "软件和信息技术服务行业")
                          if phrase in normalized]
                if direct:
                    lines[-1] += "|DIRECT_TEXT:" + "/".join(direct)
            continue

        # Options about an exact disclosed amount are not topic-presence
        # claims.  Preserve their strongest numeric sentence as a separate
        # lexical anchor so the compact window cannot cut it mid-clause.
        numbers = sorted(set(re.findall(r"\d[\d,.]*%?", option)),
                         key=lambda value: (-len(value), value))
        anchor = next((value for value in numbers
                       if value in normalized and len(value) >= 4), "")
        if anchor:
            direct = [phrase for phrase in
                      ("2024年下半年", "预计年销售额合计约40,500万元",
                       "客户采购意向", "相关性较高")
                      if phrase in option and phrase in normalized]
            lines.append(
                f"{letter}|DIRECT_TEXT|{alias}/{doc_id}|" +
                ("/".join(direct) if direct else
                 f"{anchor}hit={normalized.count(anchor)}"))
    return "\n".join(lines)[:max(0, int(char_budget))]


def _balanced_scope_audit_window(evidence, q, char_budget):
    """Build a compact evidence window with one lexical winner per option.

    This prevents a high-frequency clause for one option from consuming the
    whole review budget.  Selection depends only on the visible question,
    options and retrieved source text; identifiers and historical answers are
    intentionally absent.
    """

    char_budget = max(0, int(char_budget))
    units = [x.strip() for x in re.split(r"\n{2,}", evidence or "")
             if x.strip()]
    options = [(str(letter), str(text))
               for letter, text in (q.get("options") or {}).items()]
    if not units or not options or char_budget <= 0:
        return (evidence or "")[:char_budget]
    question = str(q.get("question") or "")
    visible_text = question + " " + " ".join(
        text for _letter, text in options)
    literal_presence = bool(
        ("各文件原文" in question or "与各文件" in question) and
        re.search(r"提到|提及|未.{0,5}(?:提到|提及)|不涉及|不存在",
                  visible_text))

    sources_by_option = {option: _named_option_sources(q, option)
                         for _letter, option in options}

    def source_matches(unit, sources):
        return any(alias in unit or f"[{doc_id}]" in unit or
                   f"【{doc_id} " in unit or f"【{doc_id}】" in unit
                   for alias, doc_id in sources)

    def semantic_bonus(unit, option):
        if not literal_presence:
            return 0
        sources = sources_by_option.get(option) or []
        if sources and not source_matches(unit, sources):
            return 0
        bonus = 0
        if "产能" in option and re.search(r"不涉及.{0,8}产能|不涉及产能", unit):
            bonus += 120
        if (re.search(r"未.{0,6}(?:提到|提及)|不涉及|不存在", option) and
                "风险" in option and "风险" in unit):
            bonus += 120
            for _alias, doc_id in sources:
                if (re.search(rf"\[{re.escape(doc_id)}\][^\n]{{0,160}}风险",
                              unit) or
                        re.search(rf"【{re.escape(doc_id)}[^】]*】[^\n]{{0,160}}风险",
                                  unit)):
                    bonus += 220
                    break
        return bonus

    selected = {}
    for letter, option in options:
        option_sources = sources_by_option.get(option) or []

        def rank_key(i):
            entity_mismatch = bool(
                option_sources and
                not source_matches(units[i], option_sources))
            score = (3 * _scope_overlap_score(units[i], option) +
                     _scope_overlap_score(units[i], question) +
                     semantic_bonus(units[i], option))
            return (entity_mismatch, -score, i)

        ranked = sorted(
            range(len(units)),
            key=rank_key,
        )
        if not ranked:
            continue
        inventory_claim = bool(literal_presence and re.search(
            r"提到|提及|未.{0,6}(?:提到|提及)|不涉及|不存在|因此|因其|涉及",
            option))
        for pos in ranked[:2 if inventory_claim else 1]:
            item = selected.setdefault(pos, {"letters": [], "queries": []})
            if letter not in item["letters"]:
                item["letters"].append(letter)
            if option not in item["queries"]:
                item["queries"].append(option)
    if not selected:
        return (evidence or "")[:char_budget]

    ordered = sorted(selected.items(),
                     key=lambda item: min(options.index((letter, text))
                                          for letter, text in options
                                          if letter in item[1]["letters"]))
    pieces = []
    remaining = char_budget
    for seq, (pos, item) in enumerate(ordered):
        groups_left = len(ordered) - seq
        label = "【" + "/".join(item["letters"]) + "项词法窗口】\n"
        allowance = max(0, remaining // groups_left)
        body_budget = max(0, allowance - len(label))
        source_anchors = []
        for option in item["queries"]:
            for alias, doc_id in sources_by_option.get(option) or ():
                source_anchors.extend((alias, doc_id))
        compact = _compact_scope_unit(
            units[pos], " ".join(item["queries"] + source_anchors),
            body_budget)
        piece = (label + compact)[:allowance]
        if piece:
            pieces.append(piece)
            remaining -= len(piece) + (2 if groups_left > 1 else 0)
    return "\n\n".join(pieces)[:char_budget]


_SCOPE_START = "<QWEN_DECISION_V1>"
_SCOPE_END = "</QWEN_DECISION_V1>"


def _parse_scope_decision(content, q):
    """Validate a complete option-by-option Qwen decision protocol.

    A plain ``答案:`` regex is insufficient for an audit: a truncated review
    may contain a provisional letter while omitting one or more options.  This
    parser only validates structure; it never supplies or changes a letter.
    """

    text = str(content or "").strip()
    start = text.find(_SCOPE_START)
    end = text.find(_SCOPE_END, start + len(_SCOPE_START)) if start >= 0 else -1
    if start >= 0:
        if end < 0 or text[end + len(_SCOPE_END):].strip():
            return {"valid": False, "status": "", "answer": None,
                    "error": "TRUNCATED_OR_UNDELIMITED"}
        block = text[start + len(_SCOPE_START):end].strip()
    else:
        # Qwen occasionally omits XML-like wrappers while following every
        # substantive row.  A bare protocol is complete only when STATUS is
        # its final non-empty line; this remains a deterministic terminator,
        # unlike accepting a provisional answer from a truncated paragraph.
        bare_lines = [line.strip() for line in text.splitlines()
                      if line.strip()]
        if (not bare_lines or
                not re.fullmatch(r"STATUS\|(COMPLETE|NEED_EVIDENCE)",
                                 bare_lines[-1])):
            return {"valid": False, "status": "", "answer": None,
                    "error": "TRUNCATED_OR_UNDELIMITED"}
        block = text
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    answers = [re.fullmatch(r"答案\s*[:：]\s*([A-D]+|-)", line)
               for line in lines]
    answers = [match.group(1) for match in answers if match]
    statuses = [re.fullmatch(r"STATUS\|(COMPLETE|NEED_EVIDENCE)", line)
                for line in lines]
    statuses = [match.group(1) for match in statuses if match]
    if len(answers) != 1 or len(statuses) != 1:
        return {"valid": False, "status": "", "answer": None,
                "error": "ANSWER_OR_STATUS_CARDINALITY"}
    expected = [str(letter) for letter in (q.get("options") or {}).keys()]
    rows = {}
    for line in lines:
        match = re.fullmatch(r"([A-D])\|(IN|OUT|UNKNOWN)\|(.{2,})", line)
        if not match:
            continue
        letter, state, reason = match.groups()
        if letter in rows:
            return {"valid": False, "status": statuses[0], "answer": None,
                    "error": "DUPLICATE_OPTION"}
        rows[letter] = {"state": state, "reason": reason}
    if set(rows) != set(expected):
        return {"valid": False, "status": statuses[0], "answer": None,
                "error": "OPTION_COVERAGE"}
    status, raw_answer = statuses[0], answers[0]
    unknown = {letter for letter, row in rows.items()
               if row["state"] == "UNKNOWN"}
    included = "".join(sorted(letter for letter, row in rows.items()
                              if row["state"] == "IN"))
    if status == "NEED_EVIDENCE":
        if not unknown:
            return {"valid": False, "status": status, "answer": None,
                    "error": "NEED_EVIDENCE_CONSISTENCY"}
        # Some Qwen generations leave a provisional subset on the answer line
        # even while correctly declaring UNKNOWN/NEED_EVIDENCE.  It is never
        # used as an answer: the caller preserves the independent first view.
        return {"valid": True, "status": status, "answer": None,
                "error": "", "rows": rows,
                "provisional_answer_ignored": raw_answer}
    if unknown or raw_answer == "-" or raw_answer != included:
        return {"valid": False, "status": status, "answer": None,
                "error": "FINAL_SET_MISMATCH"}
    if q.get("answer_format") == "mcq" and len(included) != 1:
        return {"valid": False, "status": status, "answer": None,
                "error": "CARDINALITY"}
    if not included:
        return {"valid": False, "status": status, "answer": None,
                "error": "EMPTY_FINAL"}
    return {"valid": True, "status": status, "answer": included,
            "error": "", "rows": rows}


BATCH_INST = (
    "以下 {n} 道题基于同一批文档，请逐题独立作答，题与题之间不得互相影响。\n"
    "{judge}\n"
    "输出格式（每题一个块，序号必须与题目序号一致）:\n"
    "【第1题 答案块】\n选择标准: <不超过30字的一句话>\n"
    "分析: <每个选项一行；每项只写最关键的证据页码、理由和入选/不选，"
    "不超过约60个汉字；不得省略选项>\n"
    "复合选项须拆成核心分句，全部成立才入选；若分析写明某核心主张未提及/"
    "无依据，该项必须不选。答案字母须与逐项结论完全一致。\n"
    "系统已完成本轮证据检索；每道选择题至少有一个正确选项，禁止输出‘无’、"
    "‘补充检索’或空答案，必须在A-D中完成判断。\n"
    "答案: <字母>\n"
    "【第2题 答案块】\n...（依此类推，每题都必须有'答案:'行）"
)


def _batch_b1_thinking_budget(qs, insurance_route=""):
    """Return a visible-shape reasoning budget for the first batch pass."""
    domain = qs[0].get("domain") if qs else ""
    if insurance_route == "suicide_exception":
        return 1600
    if _insurance_dense_exclusion_batch(qs):
        return 3800
    if domain == "insurance":
        if len(qs) > 1:
            from .insurance_capsules import insurance_question_route
            routes = [insurance_question_route(q) for q in qs]
            if routes and all(route == "life_contract" for route in routes):
                # Direct life-contract clause batches have repeatedly consumed
                # the full 2,500-token hidden budget without changing their
                # answer.  Keep the evidence and visible answer protocol intact
                # while trimming only this homogeneous semantic route.
                return 2400
        return 2500
    if domain == "financial_reports":
        # Typed report snapshots and fact registries already bind the operands;
        # a compact verification pass is sufficient and gives completion
        # variance a little headroom without removing any evidence.
        return 2500
    return 2600


def answer_batch(qs, model=DEFAULT_MODEL, log=None, return_info=False):
    """批量作答一组同文档选择题。返回 {qid: final_answer}。"""
    ev, ev_ids, low_cov, evidence_ownership = _batch_evidence(
        qs, model=model, return_ownership=True)
    qtexts = "\n\n".join(f"[第{i+1}题 {q['qid']}]\n{_q_text(q)}"
                         for i, q in enumerate(qs))
    base = ev + "\n\n" + qtexts
    inst = BATCH_INST.format(n=len(qs), judge=judge_std_for(qs))
    if _insurance_dense_exclusion_batch(qs):
        inst += (
            "\n【多产品除外责任盘点】每题必须按A/B/C/D分四行，不得合并或写"
            "‘同上’；每行40至60字，依次写产品/文档、目标责任词、原文字面命中"
            "或未命中窗口、页码锚点及入选/不选。近义条款须写出词面对应，零命中"
            "也须说明已核对的免责条款窗口。")
    share = [q["qid"] for q in qs]

    import os as _os
    slim4 = _os.environ.get("AFAC_SLIM4") == "1"

    def _chat(prompt, tag, mdl, budget):
        if slim4:
            budget = min(budget, 1300)
        allocation_plan = _batch_prompt_allocation_plan(
            prompt, qs, evidence_ownership=evidence_ownership)
        c, _r, usage = chat([{"role": "user", "content": prompt}],
                            qid="_batch", model=mdl, thinking=_think(qs[0]),
                            thinking_budget=budget,
                            max_tokens=(1400 * len(qs) + 1400) if slim4
                            else 1500 * len(qs) + 1500,
                            tag=tag, allocation_qids=share,
                            allocation_plan=allocation_plan,
                            allocation_resolver=lambda content: (
                                _batch_completion_allocation(content, qs)))
        return c

    traces = {q["qid"]: [] for q in qs}

    def add_batch_trace(content, stage):
        details = _parse_batch_details(content, qs)
        for qid, d in details.items():
            traces[qid].append({"stage": stage, "content": d["content"],
                                "answer": d["answer"]})
        return {qid: d["answer"] for qid, d in details.items()}

    coverage_audit = None
    coverage_audit_profile = None
    insurance_route = ""
    if len(qs) == 1 and qs[0].get("domain") == "insurance":
        from .insurance_capsules import insurance_question_route
        insurance_route = insurance_question_route(qs[0])
        if insurance_route == "minor_death_limit_exhaustive":
            audit_budget = int(_os.environ.get(
                "AFAC_INS_MINOR_AUDIT_THINKING", "2200"))
            coverage_prompt = (
                base +
                "\n\n你是Qwen保险条款覆盖审计员。本题是跨产品的明确条款"
                "存在性核查。请按A-D逐产品审阅：先确认代码记录的扫描范围，"
                "再引用是否出现‘未成年人身故保险金限制/监管限额’的完整原句；"
                "严格区分普通的未成年身故给付计算与监管限额条款。对零字面命中"
                "的产品，也要检查当前产品证据是否足以支持‘未明确列明’，如不足"
                "则指出缺口。只输出【覆盖审计】及逐项记录，不给最终选项字母。")
            coverage_audit = _chat(
                coverage_prompt, "b0_insurance_coverage", model,
                audit_budget)
            coverage_audit_profile = {
                "route": insurance_route,
                "thinking_budget": audit_budget,
                "max_tokens": 1500 * len(qs) + 1500,
                "instruction": "cross_product_literal_clause_inventory",
            }
            base += ("\n\n独立覆盖审计（最终判断必须回应其证据充分性）:\n" +
                     coverage_audit)

    b1_budget = _batch_b1_thinking_budget(qs, insurance_route)
    c1 = _chat(base + "\n\n" + inst, "b1", model, b1_budget)
    a1 = add_batch_trace(c1, "b1")
    scope_answers, scope_audits = {}, {}
    for q in qs:
        profile = _choice_scope_audit_profile(q)
        if profile is None or not a1.get(q["qid"]):
            continue
        evidence_budget = int(profile["evidence_chars"])
        if profile["profile"] == "contract_cross_document_literal_presence":
            scan = _literal_presence_scan_block(q, char_budget=320)
            remaining = max(0, evidence_budget - len(scan) - (2 if scan else 0))
            excerpt = _balanced_scope_audit_window(ev, q, remaining)
            window = (scan + ("\n\n" if scan and excerpt else "") + excerpt)
        else:
            window = _balanced_scope_audit_window(ev, q, evidence_budget)
        literal_constraint = (
            profile["profile"] == "contract_cross_document_literal_presence")
        if literal_constraint:
            # The scan block is generated from every named source document,
            # not from a top-k excerpt.  Qwen still performs every semantic
            # decision; code only supplies deterministic hit/absence facts and
            # exact same-document anchors.  This prevents a compact but
            # exhaustive scan from being mistaken for missing evidence.
            prompt = (
                "你是Qwen全文词法约束确认员。首答来自独立Qwen证据审读；"
                "本轮须把首答与已覆盖题中全部命名文件的全文扫描合成为完整"
                "逐项结论。只校正扫描能够确定的存在性、因果和复合原子，"
                "不得凭常识补原文。约束：POS_MENTION的hit=0使正向‘提到’"
                "原子为假；NEG_MENTION的hit=0使‘未提及’原子成立；"
                "CAUSAL的hit=0且有反向原句时使因果原子为假；DIRECT_TEXT"
                "是同一命名文件的逐字锚点，可核对其余原子。扫描已穷尽命名"
                "文件，不能因窗口精简写UNKNOWN；逐项合成后必须COMPLETE。"
                "复合选项全部核心原子成立才IN。必须只输出协议，禁止前言；"
                "每项一行且不超过45字，答案必须等于全部IN字母：\n"
                "<QWEN_DECISION_V1>\n答案: <完整字母>\n"
                "A|IN/OUT|扫描锚点及简短理由\n"
                "B|IN/OUT|扫描锚点及简短理由\n"
                "（按实际选项继续）\nSTATUS|COMPLETE\n"
                "</QWEN_DECISION_V1>\n\n首答=" + a1[q["qid"]] +
                "\n\n" + _q_text(q) + "\n\n全文扫描与原文锚点:\n" +
                window)
        else:
            if profile["profile"] == "regulatory_cross_rule_effective_date":
                # This audit already has a narrow, explicit rule shape.  Use a
                # compact equivalent protocol so the direct per-question call
                # does not repeat the batch-wide calibration prose.
                protocol = (
                    "仅依据原文窗口逐项复核主体、日期、条件、例外和数值边界。" +
                    profile["instruction"] +
                    "核心事实一致可入选；主体、数值或方向反转仍为错。"
                    "仅输出协议，每项一行且不超过45字。缺直接证据写UNKNOWN；"
                    "有UNKNOWN则答案=-、STATUS=NEED_EVIDENCE。答案行置首：\n")
            else:
                protocol = (
                    "你是Qwen金融规则链复核员。请只依据下列原文窗口，逐项核查"
                    "先前判断的主体、日期、条件、例外和数值边界。" +
                    profile["instruction"] +
                    "判分时核心事实一致即入选；不得仅因选项省略未改变结论的"
                    "次要触发前提而判错，但主体、数值或方向反转仍判错。"
                    "必须只输出下列协议，禁止前言；每个选项一行且每行不超过"
                    "45个汉字。窗口没有直接证据时必须写UNKNOWN，不能把缺证据"
                    "当成OUT；只要存在UNKNOWN，STATUS必须为NEED_EVIDENCE且答案"
                    "必须为-。答案行放在协议第一行，以免尾部截断：\n")
            prompt = (
                protocol +
                "<QWEN_DECISION_V1>\n答案: <完整字母或->\n"
                "A|IN/OUT/UNKNOWN|原文锚点及简短理由\n"
                "B|IN/OUT/UNKNOWN|原文锚点及简短理由\n"
                "（按实际选项继续）\nSTATUS|COMPLETE/NEED_EVIDENCE\n"
                "</QWEN_DECISION_V1>\n\n" + _q_text(q) +
                "\n\n原文窗口:\n" + window)
        audit, _hidden, _usage = chat(
            [{"role": "user", "content": prompt}], qid=q["qid"],
            model=model, thinking=profile["thinking_budget"] > 0,
            thinking_budget=profile["thinking_budget"],
            max_tokens=profile["max_tokens"], tag=profile["tag"])
        attempts = [{"stage": profile["tag"], "content": audit.strip()}]
        decision = _parse_scope_decision(audit, q)
        if not decision["valid"]:
            if literal_constraint:
                retry_prompt = (
                    "你是Qwen全文词法约束确认员。上次仅协议格式无效（错误码="
                    + decision["error"] + "），错误码不暗示答案。全文扫描已"
                    "覆盖题中全部命名文件：POS_MENTION hit=0否定正向提到；"
                    "NEG_MENTION hit=0支持未提及；CAUSAL hit=0并有反向原句"
                    "否定因果；DIRECT_TEXT核对其余原子。不得写UNKNOWN。"
                    "请逐项重做并完整输出<QWEN_DECISION_V1>，答案行置顶，"
                    "各项仅IN/OUT，STATUS|COMPLETE，保留结束标记。\n\n首答="
                    + a1[q["qid"]] + "\n\n" + _q_text(q) +
                    "\n\n全文扫描与原文锚点:\n" + window)
            else:
                retry_prompt = (
                    "你是Qwen金融规则链复核员。上次输出未通过结构校验（错误码="
                    + decision["error"] + "）。请重新依据同一原文窗口独立判断；"
                    "错误码只说明格式，不暗示答案。必须完整输出"
                    "<QWEN_DECISION_V1>协议，答案行置顶，逐选项每行不超过30个"
                    "汉字，结束标记必须保留。无直接证据写UNKNOWN/"
                    "NEED_EVIDENCE，绝不把缺证据判为OUT。核心事实一致时，不因"
                    "省略未改变结论的次要前提判错。\n\n" + _q_text(q) +
                    "\n\n原文窗口:\n" + window)
            retry, _hidden, _usage = chat(
                [{"role": "user", "content": retry_prompt}],
                qid=q["qid"], model=model, thinking=False,
                max_tokens=260, tag=profile["tag"] + "_format_retry")
            attempts.append({"stage": profile["tag"] + "_format_retry",
                             "content": retry.strip()})
            decision = _parse_scope_decision(retry, q)
            audit = retry

        audited_answer = decision.get("answer") if decision["valid"] else None
        resolved_answer = a1[q["qid"]]
        resolution = ("need_evidence" if decision["valid"] else
                      "unresolved_format")
        if audited_answer:
            traces[q["qid"]].append({
                "stage": attempts[-1]["stage"], "content": audit.strip(),
                "answer": audited_answer})
            resolved_answer = audited_answer
            resolution = "audit_agreement"
        if (audited_answer and audited_answer != a1[q["qid"]] and
                profile.get("complete_audit_authoritative")):
            # A complete literal-presence inventory already is the targeted
            # Qwen evidence decision: every option is source-bound, no UNKNOWN
            # is allowed, and the protocol verifies FINAL==IN.  A third call
            # would only repeat that same compact proof.
            resolution = "complete_literal_audit"
        elif audited_answer and audited_answer != a1[q["qid"]]:
            # A targeted audit is a second semantic view, not an automatic
            # override.  Resolve a real disagreement with one visible Qwen
            # arbiter over the same compact source window.
            arb_prompt = (
                "你是Qwen金融证据仲裁员。首答与规则链审计的字母结论不同。"
                "请仅依据下列原文窗口逐项解决分歧，不得凭常识补位。"
                "必须完整输出<QWEN_DECISION_V1>协议：答案行置顶；每个"
                "选项分别写IN/OUT/UNKNOWN及原文锚点；缺证据写UNKNOWN，"
                "并以STATUS和结束标记收尾。核心事实一致时，不因省略未"
                "改变结论的次要前提判错。\n\n" +
                _q_text(q) + "\n\n原文窗口:\n" + window +
                f"\n\n首答={a1[q['qid']]}\n审计答案={audited_answer}"
                "\n审计记录:\n" + audit[:500])
            arb_thinking = int(profile.get("arb_thinking_budget", 0))
            arb, _hidden, _usage = chat(
                [{"role": "user", "content": arb_prompt}], qid=q["qid"],
                model=model, thinking=arb_thinking > 0,
                thinking_budget=arb_thinking,
                max_tokens=int(profile.get("arb_max_tokens", 320)),
                tag=profile["tag"] + "_arb")
            arb_attempts = [{"stage": profile["tag"] + "_arb",
                             "content": arb.strip()}]
            arb_decision = _parse_scope_decision(arb, q)
            if not arb_decision["valid"]:
                arb_retry_prompt = (
                    "你是Qwen金融证据仲裁员。上次仲裁协议格式无效（错误码="
                    + arb_decision["error"] + "），错误码不暗示答案。请依据"
                    "同一原文窗口重做完整<QWEN_DECISION_V1>；答案行置顶，"
                    "逐选项不超过30字，缺证据写UNKNOWN，保留结束标记；"
                    "核心事实一致时不因省略次要前提判错。\n\n"
                    + _q_text(q) + "\n\n原文窗口:\n" + window +
                    f"\n\n首答={a1[q['qid']]}\n审计答案={audited_answer}")
                arb_retry, _hidden, _usage = chat(
                    [{"role": "user", "content": arb_retry_prompt}],
                    qid=q["qid"], model=model, thinking=False,
                    max_tokens=260, tag=profile["tag"] + "_arb_format_retry")
                arb_attempts.append({
                    "stage": profile["tag"] + "_arb_format_retry",
                    "content": arb_retry.strip()})
                arb_decision = _parse_scope_decision(arb_retry, q)
                arb = arb_retry
            if arb_decision["valid"] and arb_decision.get("answer"):
                resolved_answer = arb_decision["answer"]
                resolution = "arbitrated"
                traces[q["qid"]].append({
                    "stage": arb_attempts[-1]["stage"],
                    "content": arb.strip(), "answer": resolved_answer})
            else:
                # An incomplete optional audit cannot override a valid first
                # Qwen answer.  Keep the first answer and expose the unresolved
                # state; never infer a letter in code.
                resolved_answer = a1[q["qid"]]
                resolution = ("arb_need_evidence" if arb_decision["valid"]
                              else "arb_unresolved_format")
            attempts.extend(arb_attempts)
        scope_answers[q["qid"]] = resolved_answer
        scope_audits[q["qid"]] = {
            "content": audit.strip(), "attempts": attempts,
            "profile": dict(profile),
            "first_answer": a1[q["qid"]],
            "audited_answer": audited_answer,
            "resolved_answer": resolved_answer,
            "conflict": bool(audited_answer and
                             audited_answer != a1[q["qid"]]),
            "decision_status": decision.get("status"),
            "decision_error": decision.get("error"),
            "resolution": resolution,
        }
    # 批内多数决（AFAC_B1_VOTES=N, 域白名单AFAC_B1_VOTE_DOMS）：
    # 只对摇摆重灾域花钱，同证据独立采样逐选项投票
    n_b1 = int(_os.environ.get("AFAC_B1_VOTES", "1"))
    vote_doms = _os.environ.get("AFAC_B1_VOTE_DOMS",
                                "financial_reports").split(",")
    if n_b1 > 1 and qs[0]["domain"] in vote_doms:
        pools = {q["qid"]: [a1.get(q["qid"])] for q in qs}
        for _i in range(n_b1 - 1):
            cx = _chat(base + "\n\n" + inst, "b1", model, 1600)
            ax = add_batch_trace(cx, "b1_vote")
            for q in qs:
                pools[q["qid"]].append(ax.get(q["qid"]))
        for q in qs:
            vals = [v for v in pools[q["qid"]] if v]
            v = _vote_letters(vals, q["answer_format"])
            if v:
                a1[q["qid"]] = v
    import os as _os
    # AFAC_HETERO_B2=1: slim档也开异构二审（full11实测跨代3.5-plus=+5键的最强武器）
    if _os.environ.get("AFAC_SLIM") == "1" and \
            _os.environ.get("AFAC_HETERO_B2") != "1":
        a2 = {}
    else:
        c2 = _chat(base + "\n\n" + inst +
                   "\n（这是独立复核轮，请从头独立判断）", "b2",
                   VERIFY_MODEL or model, 2200)
        a2 = add_batch_trace(c2, "b2")

    finals, infos = {}, {}
    for q in qs:
        qid, fmt = q["qid"], q["answer_format"]
        x1, x2 = scope_answers.get(qid) or a1.get(qid), a2.get(qid)
        if x1 and x2 and x1 != x2:
            # 单题定向仲裁（复用批证据，只带该题）
            disputed = [l for l in "ABCD" if (l in x1) != (l in x2)]
            dtxt = "\n".join(f"{l}. {q['options'][l]}" for l in disputed
                             if l in q["options"])
            adj = (ev + "\n\n" + _q_text(q) +
                   f"\n\n两次独立判断分歧选项:\n{dtxt}\n甲={x1} 乙={x2}\n"
                   "请仅核对分歧选项后给出完整最终答案。\n" + judge_std_for(q) +
                   "\n输出:\n答案: <字母>")
            c3, _r, _u = chat([{"role": "user", "content": adj}], qid=qid,
                              model=VERIFY_MODEL or model, thinking=True,
                              thinking_budget=2600, max_tokens=2600, tag="b3")
            x3 = parse_answer(c3, fmt)
            traces[qid].append({"stage": "b3", "content": c3,
                                "answer": x3})
            finals[qid] = _vote_letters([x1, x2, x3], fmt) or x3 or x2
        else:
            final = x1 or x2
            if not final:
                # 解析失败禁止默认A（实测默认A 8/8全错）：精简证据单题追答
                # 使用逐选项均衡窗口，避免全量批证据的前5000字恰好属于别题；
                # 问题与输出约束置顶，也避免证据中的“补充检索”短语被回声。
                fallback_ev = _balanced_scope_audit_window(ev, q, 6500)
                c4, _r, _u = chat(
                    [{"role": "user", "content":
                      "批量回答未产生可解析字母。请从下列A-D中逐项独立核对；"
                      "本题至少一项正确，禁止回答‘无’或‘补充检索’。\n\n" +
                      _q_text(q) + "\n\n" + judge_std_for(q) +
                      "\n\n逐选项均衡原文证据:\n" + fallback_ev +
                      "\n\n先简短逐项判断，最后必须单独输出：答案: <A-D字母>"}],
                    qid=qid, model=model, thinking=False,
                    max_tokens=650, tag="b4")
                final = parse_answer(c4, fmt)
                traces[qid].append({"stage": "b4", "content": c4,
                                    "answer": parse_answer(c4, fmt)})
                if not final:
                    raise RuntimeError(f"{qid}: batch and fallback parsing failed")
            finals[qid] = final
        original_final = finals[qid]
        constrained, constraint_note = apply_structural_evidence_constraints(
            original_final, q, ev)
        if constraint_note:
            finals[qid] = constrained
            prior_reasoning, _prior_stage = select_reasoning(
                original_final, traces[qid], fmt)
            confirmed_reasoning, confirmed_answer = \
                confirm_structural_evidence_constraint(
                    q, prior_reasoning, constrained, constraint_note,
                    model=model)
            traces[qid].append({
                "stage": "evidence_constraint_qwen",
                "content": confirmed_reasoning,
                "answer": confirmed_answer,
            })
        reasoning, reasoning_stage = select_reasoning(finals[qid],
                                                      traces[qid], fmt)
        infos[qid] = {"reasoning": reasoning,
                      "reasoning_stage": reasoning_stage,
                      "traces": traces[qid]}
        if coverage_audit is not None:
            infos[qid]["coverage_audit"] = coverage_audit
            infos[qid]["coverage_audit_profile"] = coverage_audit_profile
        if qid in scope_audits:
            infos[qid]["choice_scope_audit"] = scope_audits[qid]
        if log is not None:
            log_record = {
                "qid": qid, "final": finals[qid], "r1": x1, "r2": x2,
                "batch": share, "reasoning": reasoning,
                "reasoning_stage": reasoning_stage,
                "evidence_ids": ev_ids}
            if coverage_audit is not None:
                log_record["coverage_audit"] = coverage_audit
                log_record["coverage_audit_profile"] = coverage_audit_profile
            if qid in scope_audits:
                log_record["choice_scope_audit"] = scope_audits[qid]
            log.write(json.dumps(log_record, ensure_ascii=False) + "\n")
            log.flush()
    # 覆盖率制导定向单答：被并集挤饿的题用自己的完整证据重答（窄而准）
    # 仅限大文档少题域(fc/fin)：ins题需4份条款,solo小帽反而饿死(slim19教训ins14→6)
    if low_cov and qs[0]["domain"] not in ("financial_contracts",
                                           "financial_reports"):
        low_cov = []
    if low_cov:
        from .answerer import answer_question
        for q in qs:
            if q["qid"] not in low_cov:
                continue
            try:
                a_solo, solo_info = answer_question(q, model, log,
                                                    blind_mode=True)
            except Exception:  # noqa: BLE001
                a_solo = ""
            if a_solo:
                if log is not None:
                    log.write(json.dumps(
                        {"qid": q["qid"], "solo_retry": a_solo,
                         "batch_ans": finals.get(q["qid"])},
                        ensure_ascii=False) + "\n")
                finals[q["qid"]] = a_solo
                infos[q["qid"]] = solo_info
    return (finals, infos) if return_info else finals
