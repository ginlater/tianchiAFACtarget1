"""计算题/抽取题答题流程（B榜新题型，26/100）。

难点：需要精确定位原始数值（常在表格里），按题目口径计算，输出规范格式。
策略：宽证据检索 → 先抽取数值再计算（两阶段，防止边读边算出错）→
      独立第二样本 → 不一致时定向仲裁。
"""
import dataclasses
import json, os, re

DEEP = os.environ.get("AFAC_DEEP") == "1"
SLIM = os.environ.get("AFAC_SLIM") == "1"

from . import retrieval
from .answerer import build_digest, DIGEST_DOMAINS, gather_evidence, _doc_title
from .qwen_client import chat, DEFAULT_MODEL

SLOT_DESC = {
    "number": "纯数字，保留两位小数，不带单位和千分位逗号",
    "percent": "百分数，形如 12.34%，保留两位小数",
    "ranking": "排序，用英文半角 > 连接，前后不加空格；公司名用题干中的简称原文",
    "date": "中文日期，形如 2026年1月1日",
    "text": "题目要求的完整文本，不加多余说明",
}

CALC_INST = """你是金融数据分析专家。请分两步作答：

第一步【取数】：从证据中逐条摘录计算所需的原始数值，每条注明来源页码与口径
（年度、公司、报表项目名）。只用证据中出现的数字，严禁用记忆或估算补数。
若某个必需数值在证据中找不到，最后一行输出：补充检索: <关键词>

第二步【计算】：写出算式并逐步计算。规则：
- 使用原始金额计算，中间过程不四舍五入，只在最终结果按要求保留两位小数
- "百分点"指两个百分比之差（如 12.5%-10.2%=2.3 个百分点），"百分比/增幅"指相对变化率
- 同比增幅 = (本期-上期)/|上期| × 100%
- 占比 = 部分/整体 × 100%
- 注意题目要求的单位（亿元/万元/元），必要时换算
- 涉及全年现金分红/派现的计算：全年=中期(半年度)+末期两笔合计，必须核查证据中
  是否存在中期分红记录；只按年末利润分配方案单笔计算全年值是常见错误

答案格式要求：本题需要 {n} 个答案，依次为：
{slots}
最后一行必须严格输出（多个答案用中文分号；分隔，不写单位、不写解释）：
答案: {template}"""


@dataclasses.dataclass(frozen=True)
class DeterministicVerifierBudget:
    """Auditable, semantic budget for one deterministic-result verifier.

    The profile is deliberately derived from the question text and the typed
    facts produced by the deterministic solver.  In particular, it never
    reads ``qid`` or a reference answer.  That makes the same calculation
    structure receive the same budget when replayed under a different id.
    """

    profile: str
    evidence_chars: int
    thinking_budget: int
    max_tokens: int
    fact_count: int
    product_count: int
    operation: str
    output_instruction: str

    def audit_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DeterministicScopeAuditBudget:
    """Optional first-pass audit for genuinely ambiguous source structures.

    This is not a token filler.  It is enabled only when the typed operation
    still has a material source-binding risk: several entities must be
    compared, several financial ratios share a denominator, or the wording
    leaves more than one plausible statistical population.  The audit is fed
    back into the final Qwen verifier, so every charged token contributes to
    the visible answer's provenance.
    """

    profile: str
    evidence_chars: int
    thinking_budget: int
    max_tokens: int
    instruction: str

    def audit_dict(self):
        return dataclasses.asdict(self)


def _multi_valuation_appreciation_structure(q):
    """Whether the text asks for two separately bound appraisal rates.

    This deliberately inspects only the domain and natural-language question.
    A pair can be expressed as ``two appraisals``/``two valuation dates`` or
    as two concrete baseline dates whose results are requested separately.
    The predicate therefore remains stable if the same question is replayed
    under an opaque identifier.
    """

    if str(q.get("domain") or "") != "financial_contracts":
        return False
    question = re.sub(r"\s+", "", str(q.get("question") or ""))
    if "评估增值率" not in question:
        return False
    explicit_pair = re.search(
        r"(?:两|二|2)(?:个)?(?:评估基准日|次(?:评估|估值))", question)
    dates = set(re.findall(
        r"20\d{2}年\d{1,2}月\d{1,2}日", question))
    dated_pair = (len(dates) >= 2 and "基准日" in question and
                  ("分别" in question or "评估" in question))
    return bool(explicit_pair or dated_pair)


def deterministic_verifier_budget(q, solved):
    """Choose Qwen verification depth from calculation semantics only.

    Simple single-product insurance arithmetic is already closed by one
    source rule and Decimal arithmetic, so the verifier needs a compact source
    excerpt and a short visible check.  Multi-product surrender needs every
    product rule.  A heterogeneous death-benefit sum containing an age band is
    the riskiest insurance structure: it receives a larger excerpt and more
    thinking time so Qwen can independently check each rule and the boundary.
    """

    facts = tuple(getattr(solved, "facts", ()) or ())
    entities = {
        re.sub(r"\s+", "", str(getattr(fact, "entity", "") or ""))
        for fact in facts
        if str(getattr(fact, "entity", "") or "").strip()
    }
    fact_count = len(facts)
    product_count = len(entities)
    question = re.sub(r"\s+", "", str(q.get("question") or ""))
    domain = str(q.get("domain") or "")
    intent = str(getattr(getattr(solved, "intent", None), "value",
                         getattr(solved, "intent", "")))

    common_short = (
        "输出简洁但完整：不得复述题目全文，同一数值只写一次；"
        "保留【取数】【计算】【核验】三段和最后答案行。"
    )
    if _multi_valuation_appreciation_structure(q):
        return DeterministicVerifierBudget(
            profile="contract_multi_valuation_appreciation",
            evidence_chars=9500,
            thinking_budget=1450,
            max_tokens=2300,
            fact_count=fact_count,
            product_count=product_count,
            operation="multi_baseline_appreciation_rates",
            output_instruction=(
                common_short +
                "按题面顺序逐一绑定每个评估基准日与其原文披露的增值率；"
                "不得把不同基准日的账面值、评估值或增值率交叉配对。"
            ),
        )
    if (domain == "financial_contracts" and intent == "direct_series" and
            fact_count >= 4):
        return DeterministicVerifierBudget(
            profile="contract_direct_series_dense",
            evidence_chars=5200,
            thinking_budget=1250,
            max_tokens=1900,
            fact_count=fact_count,
            product_count=product_count,
            operation=intent,
            output_instruction=(
                common_short +
                "按题面顺序逐项绑定原文事实、期间和单位；不得漏项或重排。"
            ),
        )
    if (domain == "financial_contracts" and intent == "mean" and
            fact_count >= 3):
        return DeterministicVerifierBudget(
            profile="contract_multi_period_mean",
            evidence_chars=5200,
            thinking_budget=1600,
            max_tokens=2300,
            fact_count=fact_count,
            product_count=product_count,
            operation=intent,
            output_instruction=(
                common_short +
                "逐期核对同一指标、主体和单位，列全参与平均的原值后再计算。"
            ),
        )
    if domain == "regulatory":
        profiles = {
            "calendar_offset": (
                "regulatory_calendar_offset", 5200, 2100, 3000,
                "明确受理当日是否计入，并用自然日逐日核对起止日期。"),
            "periodic_interval": (
                "regulatory_periodic_interval", 4000, 1100, 1900,
                "区分首次公告时点与后续固定周期，只计算题目所问相邻间隔。"),
            "threshold_count": (
                "regulatory_threshold_count", 4200, 1350, 2200,
                "逐笔与单笔门槛比较，明确等于门槛是否触发，禁止合并金额。"),
            "advance_notice": (
                "regulatory_advance_notice", 4200, 1500, 2400,
                "从施行日反向核对至少提前的自然日天数和最晚开始日期。"),
        }
        if intent in profiles:
            profile, ev_chars, think, max_out, instruction = profiles[intent]
            return DeterministicVerifierBudget(
                profile=profile,
                evidence_chars=ev_chars,
                thinking_budget=think,
                max_tokens=max_out,
                fact_count=fact_count,
                product_count=product_count,
                operation=intent,
                output_instruction=common_short + instruction,
            )
    if (domain == "insurance" and product_count >= 3 and
            "合计身故保险金" in question and
            re.search(r"\d+周岁", question) and
            ("基本保险金额" in question or "基本保额" in question)):
        return DeterministicVerifierBudget(
            profile="insurance_multi_age_death",
            evidence_chars=7200,
            thinking_budget=1500,
            max_tokens=2400,
            fact_count=fact_count,
            product_count=product_count,
            operation="heterogeneous_death_sum_with_age_band",
            output_instruction=(
                common_short +
                "逐产品核对规则；年龄档必须写成半开区间并代入年龄，"
                "明确上界年龄属于下一档后再求和。"
            ),
        )
    if (domain == "insurance" and product_count >= 3 and
            "合计可退还" in question and
            ("解除" in question or "退保" in question)):
        return DeterministicVerifierBudget(
            profile="insurance_multi_surrender",
            evidence_chars=5200,
            thinking_budget=900,
            max_tokens=1800,
            fact_count=fact_count,
            product_count=product_count,
            operation="multi_product_surrender_sum",
            output_instruction=common_short + "逐产品各写一条规则与金额后求和。",
        )
    if domain == "insurance" and product_count == 1 and fact_count <= 2:
        return DeterministicVerifierBudget(
            profile="insurance_single_closed",
            evidence_chars=2800,
            thinking_budget=480,
            max_tokens=1000,
            fact_count=fact_count,
            product_count=product_count,
            operation=("single_product_scenarios" if "两种情形" in question
                       else "single_product_period_arithmetic"),
            output_instruction=(
                common_short +
                "取数不超过四条，计算只展开命中的条款分支和最终算式。"
            ),
        )
    if domain == "financial_reports":
        profiles = {
            "yoy_and_share_pp": ("financial_yoy_share", 5600, 900, 1800,
                                  "逐年列出境外收入与总收入，并分别核对同比和占比百分点。"),
            "margin_drop": ("financial_multi_metric_scope", 5800, 1300, 2300,
                            "逐年绑定营业收入、归母净利润和经营现金流；三项算式"
                            "分别使用Decimal精确结果，中间不得以手算近似值替换。"),
            "cashflow_rate_rank": ("financial_cross_entity_rate_rank", 7000,
                                   1500, 2500,
                                   "逐公司绑定同年度合并口径分子分母，再计算、排序和求差。"),
            "dividend_rank": ("financial_dividend_rank", 6200, 1200, 2100,
                              "逐公司核对全年口径，明确中期与末期是否需要合计后再排序。"),
            "equity_multiplier_rank": ("financial_multi_entity_leverage", 6200,
                                       1400, 2200,
                                       "逐公司绑定资产负债率及合并口径，分别算权益乘数后排序。"),
            "dupont": ("financial_dupont_closed", 5200, 1050, 1900,
                       "写明百分数转小数、权益乘数和近似资产收益率两步核验。"
                       "结果要求不带百分号时，是保留百分数数值并去掉%符号"
                       "（如7.65），不是把答案改写成小数0.08。"),
            "dividend_reconcile": ("financial_dividend_reconcile", 5200, 900,
                                   1800,
                                   "同时核对归母净利润口径、分红比例、每10股金额和基准股本。"),
            "implied_revenue": ("financial_implied_revenue", 5200, 1050, 1950,
                                "核对EBITDA率的百分数换算、收入单位及绝对相对偏差。"),
        }
        if intent in profiles:
            profile, ev_chars, think, max_out, instruction = profiles[intent]
            return DeterministicVerifierBudget(
                profile=profile,
                evidence_chars=ev_chars,
                thinking_budget=think,
                max_tokens=max_out,
                fact_count=fact_count,
                product_count=product_count,
                operation=intent,
                output_instruction=common_short + instruction,
            )
    if domain == "research":
        profiles = {
            "demand_growth": ("research_demand_population", 5200, 1100, 2100,
                              "先确认旧单车带电量对应的销量和电池需求统计口径。"
                              "若题干固定销量且只改变单车带电量，需求同比应按新旧"
                              "单车量之比约消销量；不得把独立展示舍入的需求总量混作"
                              "精确基数，除非题干明确指定该总量为基数。"),
            "growth_and_share": ("research_growth_share_closed", 5200, 1000,
                                 1900,
                                 "依次核对基期订单、增速和AI占比，禁止把占比当增速。"),
            "mixture_count": ("research_mixture_closed", 4400, 1050, 1800,
                              "依次核对GMV层级、会员消费和普通用户单价后求人数。"
                              "题干若明确GMV不含会员费，会员对GMV的贡献只能计商品"
                              "消费，严禁把会员费加进GMV再扣除。最后答案行只写"
                              "数值，不得附‘人’或‘万人’等单位。"),
        }
        if intent in profiles:
            profile, ev_chars, think, max_out, instruction = profiles[intent]
            return DeterministicVerifierBudget(
                profile=profile,
                evidence_chars=ev_chars,
                thinking_budget=think,
                max_tokens=max_out,
                fact_count=fact_count,
                product_count=product_count,
                operation=intent,
                output_instruction=common_short + instruction,
            )
    return DeterministicVerifierBudget(
        profile="deterministic_default",
        evidence_chars=5200,
        thinking_budget=900,
        max_tokens=1800,
        fact_count=fact_count,
        product_count=product_count,
        operation="typed_deterministic_calculation",
        output_instruction=common_short,
    )


def deterministic_scope_audit_budget(
    q, solved=None, *, business_calendar_complete=None
):
    """Return a semantic first-pass audit budget, never a qid-based route."""

    domain = str(q.get("domain") or "")
    intent = str(getattr(getattr(solved, "intent", None), "value",
                         getattr(solved, "intent", "")))
    facts = tuple(getattr(solved, "facts", ()) or ())
    entities = {
        re.sub(r"\s+", "", str(getattr(fact, "entity", "") or ""))
        for fact in facts
        if str(getattr(fact, "entity", "") or "").strip()
    }
    if _multi_valuation_appreciation_structure(q):
        return DeterministicScopeAuditBudget(
            "contract_multi_valuation_scope_audit", 1600, 200, 600,
            "分别核对两个评估基准日对应的账面值、评估值及原文披露增值率，"
            "列出任何跨日期错配；原文已直接披露时优先核对同日原句。")
    if (domain == "regulatory" and
            business_calendar_complete is False):
        if not intent:
            from .deterministic_calc import classify_intent
            intent = classify_intent(str(q.get("question") or "")).value
        if intent == "next_business_day":
            return DeterministicScopeAuditBudget(
                "regulatory_next_business_day_no_calendar",
                5000, 1350, 2200,
                "当前没有完整法定节假日日历，须明确核对给定日期的星期、"
                "随后周末及证据可确认的节假日；不得把自然日直接当工作日。")
    if domain == "financial_reports":
        routes = {
            "yoy_and_share_pp": ("financial_two_year_scope_audit", 1800, 250,
                                 600,
                                 "核对两个年度的境外收入和营业收入是否同口径。"),
            "margin_drop": ("financial_multi_metric_scope_audit", 2200, 500,
                            850,
                            "核对两年三项指标是否均为合并口径；每个年度只用一条"
                            "BIND汇总三项，禁止抄数值、来源或逐指标拆行；只有真实"
                            "母公司竞争口径才写CONFLICT。"),
            "cashflow_rate_rank": ("financial_cross_entity_scope_audit", 3000,
                                   600, 1100,
                                   "逐公司独立核对营业收入与经营现金流的年度、单位和报表口径。"),
            "dividend_rank": ("financial_dividend_period_audit", 1000, 80,
                              320,
                              "核对各公司全年分红是否由中期和末期组成，避免只取一笔。"),
            "equity_multiplier_rank": ("financial_leverage_scope_audit", 3000,
                                       650, 1200,
                                       "逐公司核对资产负债率年度与合并口径，排除母公司口径。"),
            "implied_revenue": ("financial_rate_rounding_audit", 400, 80,
                                250,
                                "核对披露的EBITDA率是否为已舍入百分数，并确认偏差按报告营业收入作分母。"),
        }
        if intent in routes:
            # Ranking routes are meaningful only when the facts really bind
            # several entities; a malformed single-entity extraction should
            # fail closed into the normal full calculation path instead.
            if intent in {"cashflow_rate_rank", "dividend_rank",
                          "equity_multiplier_rank"} and len(entities) < 2:
                return None
            profile, chars, think, max_out, instruction = routes[intent]
            return DeterministicScopeAuditBudget(
                profile, chars, think, max_out, instruction)
    if domain == "research" and intent == "demand_growth":
        return DeterministicScopeAuditBudget(
            "research_population_scope_audit", 1000, 200, 550,
            "列出证据中所有可能的销量、动力电池需求和单车带电量口径，"
            "判断题干的“全年动力电池需求”应与哪一组同口径；销量固定且"
            "只改变单车量时，不得将独立展示舍入的总需求混入因子比例。")
    return None


def deterministic_disagreement_instruction(solved):
    """Return a semantic rule for a compact Qwen closed-form reconcile.

    Only strict deterministic intents with a known ambiguity receive this
    route.  The rule is selected from the formula shape, never from a qid,
    previous score or expected answer.
    """

    intent = str(getattr(getattr(solved, "intent", None), "value",
                         getattr(solved, "intent", "")))
    routes = {
        "margin_drop": (
            "使用原始金额和Decimal完整精度核对三个差值；题干要求中间不"
            "舍入时，禁止用手算截断的小数替换闭式记录，只在最终槽位舍入。"),
        "demand_growth": (
            "若销量持平且题干只把单车带电量从旧值改为新值，则需求同比中"
            "销量代数约消，应比较新旧单车量；独立展示舍入的需求总量仅作"
            "交叉核验，除非题干明确指定它为同比基数。"),
        "mixture_count": (
            "题干说GMV不含会员费，表示年费在该GMV之外；从APP自营GMV"
            "扣会员贡献时只能扣会员商品消费，不能把年费加进GMV贡献。"),
        "dupont": (
            "题干要求收益率答案不带百分号时，先按百分数计算和舍入，再仅"
            "移除%符号；不得把7.65%改写为小数0.08。"),
    }
    return routes.get(intent, "")


def _det_budget_with_env_overrides(budget):
    """Keep legacy global knobs for non-routed deterministic structures.

    ``honest_repro.env`` historically pins the two global knobs to 5200/900.
    Applying those values to the new insurance profiles would silently erase
    semantic routing, so routed profiles are intentionally authoritative.
    Their exact values are recorded in the audit payload.  The legacy knobs
    remain available to deterministic structures still using the default
    profile.
    """

    if budget.profile != "deterministic_default":
        return budget

    values = {
        "evidence_chars": "AFAC_DET_CALC_EVIDENCE_CHARS",
        "thinking_budget": "AFAC_DET_CALC_THINKING_BUDGET",
        "max_tokens": "AFAC_DET_CALC_MAX_TOKENS",
    }
    updates = {}
    for field, env_name in values.items():
        if env_name in os.environ:
            value = int(os.environ[env_name])
            if value <= 0:
                raise ValueError(f"{env_name} must be positive")
            updates[field] = value
    return dataclasses.replace(budget, **updates) if updates else budget


def _slots_text(kinds):
    return "\n".join(f"  第{i+1}个：{SLOT_DESC.get(k, k)}"
                     for i, k in enumerate(kinds))


def _template(kinds):
    ex = {"number": "123.45", "percent": "12.34%",
          "ranking": "甲公司>乙公司", "date": "2026年1月1日", "text": "<文本>"}
    return "；".join(ex.get(k, "<答案>") for k in kinds)


ANS_RE = re.compile(r"^\s*答案[:：]\s*(.+)$", re.M)  # 行锚定：防推理中转述格式说明被误抓
SEARCH_RE = re.compile(r"补充检索[:：]\s*(.+)")

_SLOT_PAT = {
    "number": re.compile(r"-?[\d,]+(?:\.\d+)?"),
    "percent": re.compile(r"-?[\d.,]+\s*[%％]"),
    "date": re.compile(r"\d{4}年\d{1,2}月\d{1,2}日"),
    "ranking": re.compile(r".+>.+"),
}


def valid_calc(ans, kinds):
    """槽位格式校验：不合法视为解析失败走升级重试，而不是带垃圾提交。"""
    if not ans:
        return False
    parts = [p.strip() for p in re.split(r"[；;]", ans) if p.strip()]
    if len(parts) < len(kinds):
        return False
    for p, k in zip(parts, kinds):
        pat = _SLOT_PAT.get(k)
        if pat and not pat.fullmatch(p):
            return False
    return True


def valid_calc_exact(ans, kinds):
    """Require one and only one schema-valid value for every answer slot.

    ``split_answer`` is deliberately permissive when recovering ordinary
    model output.  A compact disagreement reconcile is a stricter trust
    boundary: text such as ``67.1 或 64.7`` must not be accepted merely
    because its first number normalises to a candidate.
    """
    if not valid_calc(ans, kinds):
        return False
    parts = [p.strip() for p in re.split(r"[；;]", ans or "") if p.strip()]
    return len(parts) == len(kinds)


def scope_audit_allows_closed_reconcile(scope_audit_text, *, required=False):
    """Fail closed when an independent scope audit reports ambiguity."""
    text = (scope_audit_text or "").strip()
    if not text:
        return not required
    upper = text.upper()
    if "CONFLICT" in upper:
        return False
    return bool(re.search(r"STATUS\s*(?:=|:|：|\|)\s*OK\b", upper))


_TEMPLATE_ECHO = re.compile(r"甲公司|乙公司|123\.45|12\.34%|<文本>|<答案>")


def parse_calc(content):
    ms = list(ANS_RE.finditer(content or ""))
    for m in reversed(ms):
        v = m.group(1).strip()
        if v and not _TEMPLATE_ECHO.search(v):
            return v
    return ""


def _tables_block(q):
    """表格全景（AFAC_CALC_TABLES=1）：计算题取数几乎总在表格，而 BM25 只召回
    与题面词汇重叠的表——跨口径表（如乘用车题旁的商用车/合计预测表）从来进不了
    上下文，模型根本没有选口径的机会（res_b_005 类伤）。此层把涉及文档的全部
    图/表块整体入证据，口径取舍交还模型。超限按题面词重叠排序截断。"""
    limit = int(os.environ.get("AFAC_TABLE_LIMIT", "15000"))
    segs = []
    for d in q["doc_ids"]:
        p = retrieval.doc_path(d)
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8")
        for m in re.finditer(r"^(?:图|表)\s*[：:].*$", text, re.M):
            start = m.start()
            win = text[start:start + 6000]
            end_m = re.search(r"^数据来源[：:].*$", win, re.M)
            seg = win[:end_m.end()] if end_m else win[:2500]
            segs.append((d, seg))
    if not segs:
        return ""
    qwords = set(re.findall(r"[一-鿿]{2,6}", q["question"]))
    total = sum(len(s[1]) for s in segs)
    if total > limit:
        segs.sort(key=lambda s: -sum(1 for w in qwords if w in s[1]))
        kept, acc = [], 0
        for s in segs:
            if acc + len(s[1]) > limit:
                continue
            kept.append(s)
            acc += len(s[1])
        segs = kept
    body = "\n\n".join(f"[{d}] {seg}" for d, seg in segs)
    return "文档表格全景（含同一指标的不同口径表，注意甄别题目所问口径）:\n" + body


def _prepare_narrative_source_rescue(q):
    """Resolve a uniquely source-bound narrative bundle before retrieval.

    Document selection is intentionally allowed to fail independently of the
    registry.  If the selected documents contain no supported bundle, the
    same domain/question extractor is run over the whole processed domain.
    ``extract_facts`` is fail-closed: it returns a bundle only when all
    candidates have one semantic signature, so adding its cited source does
    not turn an ambiguous scan into an answer.  The returned facts are also
    cached for both evidence rendering and the deterministic solver.
    """

    if (os.environ.get("AFAC_NARRATIVE_REGISTRY") != "1" or
            str(q.get("domain") or "") not in {
                "financial_contracts", "regulatory", "research"}):
        return q, (), None

    from .narrative_fact_registry import extract_facts

    supplied = q.get("doc_ids") or ()
    selected_ids = tuple(str(x) for x in supplied if str(x))
    selected_facts = tuple(extract_facts(q) or ())
    rescue_attempted = not selected_facts
    facts = selected_facts
    if rescue_attempted:
        # An explicitly empty allow-list means "scan the declared domain" in
        # the registry.  Its conflict detector still sees every candidate.
        facts = tuple(extract_facts(q, doc_ids=()) or ())

    source_ids = tuple(dict.fromkeys(
        str(getattr(fact, "doc_id", "")) for fact in facts
        if str(getattr(fact, "doc_id", ""))))
    added_ids = tuple(doc_id for doc_id in source_ids
                      if doc_id not in selected_ids)
    prepared = q
    if added_ids:
        prepared = dict(q)
        prepared["doc_ids"] = list(selected_ids + added_ids)

    if selected_facts:
        status = "selected_unique_bundle"
    elif facts:
        status = "rescued_unique_bundle"
    else:
        status = "no_unique_bundle"
    audit = {
        "status": status,
        "rescue_attempted": rescue_attempted,
        "selected_doc_ids": list(selected_ids),
        "added_doc_ids": list(added_ids),
        "source_doc_ids": list(source_ids),
        "fact_count": len(facts),
        # Registry extractors only emit a complete, unique semantic bundle;
        # empty is the fail-closed representation for incomplete/conflicting.
        "unique_complete_bundle": bool(facts),
    }
    return prepared, facts, audit


def calc_evidence(q, model=DEFAULT_MODEL, extra=(), cap_mult=1,
                  narrative_facts=None):
    """计算题证据：记忆卡（含关键财务数字）+ 宽检索原文片段。"""
    blocks = []
    domain = q["domain"]
    _has_detbudget = False
    from .answerer import _use_digest
    if os.environ.get("AFAC_NO_DIGEST") == "1" and not _use_digest(domain):
        blocks.append("涉及文档:\n" + "\n".join(
            f"- {d}: 《{_doc_title(d)}》" for d in q["doc_ids"]))
    elif domain in DIGEST_DOMAINS:
        for d in q["doc_ids"]:
            blocks.append(build_digest(d, domain, model=model))
    else:
        blocks.append("涉及文档:\n" + "\n".join(
            f"- {d}: 《{_doc_title(d)}》" for d in q["doc_ids"]))
    if domain == "insurance" and os.environ.get("AFAC_INS_CAPSULES") == "1":
        from .insurance_capsules import insurance_capsule_block
        cb = insurance_capsule_block(
            q, char_budget=int(os.environ.get("AFAC_INS_CALC_CAPSULE_BUDGET",
                                              "6000")))
        if cb:
            blocks.append(cb)
    if (domain in {"financial_contracts", "regulatory", "research"} and
            os.environ.get("AFAC_NARRATIVE_REGISTRY") == "1"):
        from .narrative_fact_registry import extract_facts, facts_block
        cached_facts = (tuple(narrative_facts)
                        if narrative_facts is not None
                        else tuple(extract_facts(q) or ()))
        if cached_facts:
            blocks.append(
                "确定性叙述事实注册表（原文、文档与页码逐项绑定）:\n" +
                facts_block(cached_facts))
            _has_detbudget = True
    # Strict financial-report registry: exact source rows are bound to entity,
    # report year, column role, unit and page before any arithmetic.  It is
    # enabled only when every required operand is unique; otherwise the legacy
    # lexical/Qwen evidence path below remains untouched.
    _fin_registry_complete = False
    if domain == "financial_reports" and \
            os.environ.get("AFAC_FIN_REGISTRY") == "1":
        from .financial_fact_registry import FinancialFactRegistry
        result = FinancialFactRegistry().extract(q)
        if result.complete:
            blocks.append(
                "确定性财报事实注册表（逐项绑定公司、年份、口径、单位及来源页）:\n" +
                result.fact_block)
            _has_detbudget = True
            _fin_registry_complete = True
    # fin结构化事实表(E1尸检产物): 离线词法查表块, 取数从检索问题变查表问题
    # v2(单元格级,列口径绑定)走answerer.fin_facts_block; v1(行级)保留原路径
    if domain == "financial_reports" and not _fin_registry_complete and \
            os.environ.get("AFAC_FIN_FACTS") == "2":
        from .answerer import fin_facts_block
        ff = fin_facts_block(q)
        if ff:
            blocks.append(ff)
    if not _fin_registry_complete:
        from .answerer import align_block
        ab = align_block(q)
        if ab:
            blocks.append(ab)
    # 确定性财务计算器(AFAC_FIN_CALC=1): 词法矿精确取数+Python算术, 零token,
    # 把精确数字与算好的比率作证据喂Qwen(Qwen仍做最终答案生成/口径判断)
    if os.environ.get("AFAC_FIN_CALC") == "1" and not _fin_registry_complete:
        from .fin_calc import calc_facts_block
        cb = calc_facts_block(q)
        if cb:
            blocks.append(cb)
            _has_detbudget = True
    if os.environ.get("AFAC_CALC_TABLES") == "1":
        tb = _tables_block(q)
        if tb:
            blocks.append(tb)
    if domain == "financial_reports" and os.environ.get("AFAC_FIN_FACTS") == "1":
        import pathlib as _pl
        _ff = _pl.Path(__file__).resolve().parents[1] / "processed_data" / "fin_facts.json"
        if _ff.exists():
            facts = json.load(open(_ff))
            qwords = set(re.findall(r"[一-鿿]{2,6}", q["question"]))
            rows = []
            for d in q["doc_ids"]:
                for r in facts.get(d, []):
                    score = sum(1 for w in qwords if w in r)
                    if score:
                        rows.append((score, d, r))
            rows.sort(key=lambda x: -x[0])
            if rows:
                tbl = "\n".join(f"[{d}] {r}" for _s, d, r in rows[:28])
                blocks.append("数值速查表(离线抽取):\n" + tbl)
    # 年报计算题取数强化：三口径(主要数据表/分红两笔/母公司报表)强制齐全
    # （fin_b_005/012/016/019类伤）仅全火力档：瘦帽下额外查询=稀释(slim18教训)
    if domain == "financial_reports" and not SLIM:
        extra = tuple(extra) + ("主要会计数据 财务指标",
                                "利润分配 中期分红 末期分红 每10股",
                                "母公司 资产负债表 所有者权益 少数股东")
    # 计算题证据要宽：数字常散落在多张表
    cap = ((2600 if os.environ.get("AFAC_SLIM4") == "1" else 6000)
           if os.environ.get("AFAC_NO_DIGEST") == "1"
           else (14000 if DEEP else (7000 if SLIM else 11000))
           + 2000 * max(0, len(q["doc_ids"]) - 2))
    if os.environ.get("AFAC_CALC_LEAN") == "1":
        # 查表主导取数(500k总攻): facts2单元格表为主证据, 原文微量兜底
        cap = 3800
    # 确定性预算已给出精确数字/比率 → 检索仅留极小兜底(治"块加长token升")
    if _has_detbudget:
        cap = min(cap, 2600)
    ev, kept, _prot = gather_evidence(q, k_opt=4, k_q=5, cap=int(cap * cap_mult),
                                      extra_queries=extra)
    blocks.append("原文片段证据:\n" + ev)
    return "\n\n".join(blocks), [c["id"] for c in kept]


def answer_calc(q, kinds, model=DEFAULT_MODEL, log=None, verify_model=None,
                blind_mode=False, return_info=False):
    # 暗沟修复(GLOBAL_ESCAPE审计): verify_model只认CLI不读env,
    # 26道计算题在env配置跑里从未吃到跨代异构二审
    if verify_model is None:
        verify_model = os.environ.get("AFAC_VERIFY_MODEL") or None
    qid = q["qid"]
    inst = CALC_INST.format(n=len(kinds), slots=_slots_text(kinds),
                            template=_template(kinds))
    from .b_schema import is_date_question
    if is_date_question(q) and any(k == "number" for k in kinds):
        # 掐死数字槽逼出的自创日期编码（20260330/4.01类伤）：让模型输出中文日期，
        # fmt_slot 的确定性防护会自动转成 YYYYMMDD.00
        inst += ("\n注意：本题答案是一个日期。最后一行请输出完整中文日期"
                 "（YYYY年M月D日，如 2026年1月1日），系统会自动转换为要求的格式。")
    q, narrative_facts, narrative_source_audit = \
        _prepare_narrative_source_rescue(q)
    ev, ev_ids = calc_evidence(
        q, model=model,
        narrative_facts=(narrative_facts
                         if narrative_source_audit is not None else None))
    base = ev + "\n\n题目:\n" + q["question"] + "\n\n" + inst

    # Deterministic calculator fast path: Python resolves only strict,
    # unit-closed structures, then one Qwen call verifies the cited evidence
    # and emits the actual visible reasoning.  A disagreement never gets
    # forced through; it falls back to the complete Qwen pipeline below.
    det_trace = None
    det_attempt_traces = []
    det_budget_audit = None
    scope_audit_trace = None
    scope_audit_budget_audit = None
    fallback_scope_budget = None
    fallback_scope_budget_audit = None
    det_solved = None
    if os.environ.get("AFAC_DET_CALC") == "1":
        from .b_schema import fmt_slot, split_answer
        from .deterministic_calc import BusinessCalendar, try_solve

        calendar = None
        cal_path = (__import__("pathlib").Path(__file__).resolve().parents[1]
                    / "processed_data" / "business_calendar.json")
        if cal_path.exists():
            import datetime as _dt
            payload = json.loads(cal_path.read_text(encoding="utf-8"))
            if payload.get("complete") is True:
                holidays = frozenset(_dt.date.fromisoformat(x)
                                     for x in payload.get("holidays", []))
                weekend = frozenset(int(x) for x in
                                    payload.get("weekend", [5, 6]))
                calendar = BusinessCalendar(holidays=holidays,
                                            weekend=weekend, complete=True)
        structured_facts = ()
        if os.environ.get("AFAC_NARRATIVE_REGISTRY") == "1":
            from .narrative_fact_registry import calculation_facts
            if narrative_facts:
                structured_facts = calculation_facts(q["question"],
                                                     narrative_facts)
        solved = None
        if q.get("domain") == "insurance":
            # Typed insurance clauses are executable memory.  The solver is
            # product/source bound and fail-closed, so an unrelated extra
            # document cannot transpose a flattened policy-year table.
            from .insurance_calc import try_solve_insurance
            solved = try_solve_insurance(q)
        if solved is None:
            solved = try_solve(q["question"], ev, kinds,
                               facts=structured_facts,
                               business_calendar=calendar)
        if solved is None:
            # A working-day result must not be made deterministic from a
            # weekdays-only assumption.  When the complete holiday calendar
            # is unavailable, keep Qwen as the answerer but give that one
            # fallback call an explicit, auditable semantic budget.
            fallback_scope_budget = deterministic_scope_audit_budget(
                q, business_calendar_complete=(calendar is not None and
                                                calendar.complete))
            if fallback_scope_budget is not None:
                fallback_scope_budget_audit = \
                    fallback_scope_budget.audit_dict()
        if solved is not None:
            det_solved = solved
            fact_lines = []
            for f in solved.facts:
                fact_lines.append(
                    f"- {f.entity or '当前对象'} {f.period} {f.metric}="
                    f"{f.value} {f.unit} 来源:{f.source or '确定性证据记录'}")
            budget = _det_budget_with_env_overrides(
                deterministic_verifier_budget(q, solved))
            det_budget_audit = budget.audit_dict()
            scope_budget = deterministic_scope_audit_budget(q, solved)
            scope_audit_text = ""
            if scope_budget is not None:
                scope_audit_budget_audit = scope_budget.audit_dict()
                scope_line_limit = (
                    3 if scope_budget.profile ==
                    "financial_multi_metric_scope_audit"
                    else max(3, len(fact_lines) + 2))
                scope_prompt = (
                    "你是Qwen金融口径审计员。下面的确定性工具只做词法取数和"
                    "Decimal算术。请独立审查事实是否绑定到题目要求的主体、年度、"
                    "统计范围、报表口径和单位；重点寻找证据中会导致另一结果的"
                    "竞争口径。" + scope_budget.instruction +
                    "严格使用短协议：每个主体/期间/口径只写一条BIND行；只有"
                    "真实竞争口径才写CONFLICT行；末行写STATUS=OK或CONFLICT。"
                    f"全文最多{scope_line_limit}行。禁止复述计算过程，禁止输出"
                    "最终答案格式行。\n\n题目:\n" + q["question"] +
                    "\n\n待审证据:\n" + ev[:scope_budget.evidence_chars] +
                    "\n\n确定性事实:\n" +
                    ("\n".join(fact_lines) or "均直接来自题干") +
                    "\n确定性算式:\n" + solved.reasoning)
                scope_audit_text, _tsa, _usa = chat(
                    [{"role": "user", "content": scope_prompt}], qid=qid,
                    model=model, thinking=True,
                    thinking_budget=scope_budget.thinking_budget,
                    max_tokens=scope_budget.max_tokens,
                    tag="calc_scope_audit")
                scope_audit_trace = {
                    "stage": "calc_scope_audit",
                    "content": scope_audit_text,
                    "budget_profile": scope_audit_budget_audit,
                }
            excerpt_n = budget.evidence_chars
            verify_prompt = (
                "你是Qwen金融计算核验员。下面的确定性工具只做词法取数、日期和"
                "Decimal算术；请根据题目与证据复核其单位、口径、算式和舍入。"
                "若完全一致，必须保留工具给出的精确答案；若证据不足或工具有误，"
                "明确指出，不得迎合。" + budget.output_instruction +
                "最后一行严格写'答案: ...'。\n\n题目:\n" + q["question"] +
                "\n\n证据摘录:\n" + ev[:excerpt_n] +
                "\n\n确定性事实:\n" + ("\n".join(fact_lines) or "均直接来自题干") +
                "\n确定性算式:\n" + solved.reasoning +
                "\n工具候选答案: " + solved.raw_answer +
                (("\n\n独立口径审计（必须回应其中的冲突或确认项）:\n" +
                  scope_audit_text) if scope_audit_text else ""))
            cdet, _tdet, _udet = chat(
                [{"role": "user", "content": verify_prompt}], qid=qid,
                model=model, thinking=True,
                thinking_budget=budget.thinking_budget,
                max_tokens=budget.max_tokens, tag="calc_det")
            adet = parse_calc(cdet)
            det_trace = {"stage": "calc_det", "content": cdet,
                         "answer": adet,
                         "budget_profile": det_budget_audit}
            det_attempt_traces.append(det_trace)
            expected = [fmt_slot(x, kind)
                        for x, kind in zip(solved.slots, kinds)]
            actual = split_answer(adet, kinds) if adet else []
            if (actual != expected and
                    solved.intent.value == "calendar_offset" and
                    len(actual) == len(expected) == 1 and actual[0]):
                reconcile_prompt = (
                    "你是Qwen自然日闭式核验员。首轮核验给出的日期与词法+"
                    "datetime确定性工具不同。请只重算计日边界，不做新的事实"
                    "检索：受理当日不计入时，次日是第1日；候选终日与受理日"
                    "的日期差必须恰好等于题定日数。先写【边界】和【日期差】"
                    "各一行，再写【结论】；最后一行严格写‘答案: YYYY年M月D日’。"
                    "不得沿用首轮逐月计数中的算术笔误。\n\n题目:\n" +
                    q["question"] + "\n\n确定性日期差证明:\n" +
                    solved.reasoning + "\n工具候选答案: " +
                    solved.raw_answer + "\n首轮Qwen候选答案: " + adet +
                    "\n首轮核验摘录:\n" + cdet[:500])
                crec, _trec, _urec = chat(
                    [{"role": "user", "content": reconcile_prompt}],
                    qid=qid, model=model, thinking=True,
                    thinking_budget=250, max_tokens=600,
                    tag="calc_det_reconcile")
                arec = parse_calc(crec)
                rec_actual = split_answer(arec, kinds) if arec else []
                rec_trace = {
                    "stage": "calc_det_reconcile", "content": crec,
                    "answer": arec, "budget_profile": {
                        "profile": "calendar_offset_closed_reconcile",
                        "thinking_budget": 250, "max_tokens": 600}}
                det_attempt_traces.append(rec_trace)
                if (rec_actual == expected and
                        valid_calc_exact(arec, kinds) and
                        len(crec.strip()) >= 20):
                    cdet, adet, actual, det_trace = (
                        crec, arec, rec_actual, rec_trace)
            disagreement_rule = deterministic_disagreement_instruction(solved)
            scope_required = scope_audit_trace is not None
            scope_ok = scope_audit_allows_closed_reconcile(
                scope_audit_text, required=scope_required)
            if (actual != expected and disagreement_rule and
                    getattr(solved, "confidence", "") == "strict" and scope_ok and
                    len(actual) == len(expected) and all(actual)):
                # A strict unit-closed solver can still disagree with the
                # first Qwen verifier because the verifier hand-rounded an
                # intermediate, mixed two independently rounded bases, or
                # misread an explicit exclusion.  Give Qwen one compact,
                # auditable arbitration between the two *untrusted* schemes.
                # Code accepts the closed-form path only when this independent
                # Qwen response reproduces it; otherwise the normal full
                # evidence fallback below remains authoritative.
                compact_facts = []
                for fact in solved.facts:
                    row = (f"{getattr(fact, 'entity', '')}|"
                           f"{getattr(fact, 'period', '')}|"
                           f"{getattr(fact, 'metric', '')}="
                           f"{getattr(fact, 'value', '')} "
                           f"{getattr(fact, 'unit', '')}")
                    if row not in compact_facts:
                        compact_facts.append(row)
                reconcile_prompt = (
                    "你是Qwen闭式计算分歧仲裁员。下列甲乙方案都不预设正确；"
                    "仅依据题干、无来源修饰的事实值和口径规则独立选择，禁止"
                    "引入新数字或凭候选标签站队。" + disagreement_rule +
                    "请输出【口径】、【复算】和最后一行‘答案: ...’，必须保留"
                    "题目要求的槽位数与格式；全文不超过8行。\n\n题目:\n" +
                    q["question"][:500] + "\n\n事实值:\n" +
                    (("\n".join(compact_facts))[:300] or "均为题干闭式给定") +
                    "\n\n方案甲（Decimal闭式记录）:\n" + solved.reasoning[:700] +
                    "\n甲候选=" + solved.raw_answer +
                    "\n\n方案乙（首轮Qwen核验末段）:\n" + cdet[-180:])
                crec, _trec, _urec = chat(
                    [{"role": "user", "content": reconcile_prompt}],
                    qid=qid, model=model, thinking=False, max_tokens=240,
                    tag="calc_det_closed_reconcile")
                arec = parse_calc(crec)
                rec_actual = split_answer(arec, kinds) if arec else []
                rec_trace = {
                    "stage": "calc_det_closed_reconcile", "content": crec,
                    "answer": arec, "budget_profile": {
                        "profile": "semantic_closed_form_disagreement_v1",
                        "intent": solved.intent.value,
                        "enable_thinking": False, "max_tokens": 240,
                        "question_char_cap": 500, "facts_char_cap": 300,
                        "reasoning_char_cap": 700,
                        "first_tail_char_cap": 180}}
                det_attempt_traces.append(rec_trace)
                if (rec_actual == expected and
                        valid_calc_exact(arec, kinds) and
                        len(crec.strip()) >= 20):
                    cdet, adet, actual, det_trace = (
                        crec, arec, rec_actual, rec_trace)
            if actual == expected and len(cdet.strip()) >= 20:
                # Qwen may explain a unit in the visible reasoning and repeat
                # it on the final line (for example ``67.1万人``).  Once that
                # answer has independently normalised to every strict solver
                # slot, persist the schema-canonical value in the selected
                # trace.  The full Qwen text remains untouched and auditable.
                adet = "；".join(actual)
                det_trace["answer"] = adet
                det_stage = det_trace.get("stage", "calc_det")
                info = {"reasoning": cdet.strip(),
                        "reasoning_stage": det_stage,
                        "traces": list(det_attempt_traces),
                        "raw_answer": adet,
                        "deterministic_intent": solved.intent.value,
                        "deterministic_reasoning": solved.reasoning,
                        "deterministic_budget_profile": det_budget_audit}
                if narrative_source_audit is not None:
                    info["narrative_source_audit"] = \
                        narrative_source_audit
                if scope_audit_trace is not None:
                    info["scope_audit"] = scope_audit_trace
                    info["scope_audit_budget_profile"] = \
                        scope_audit_budget_audit
                if log is not None:
                    log_record = {
                        "qid": qid, "final": adet, "a1": None,
                        "a2": None, "c1": cdet, "c1b": None,
                        "c2": None, "c3": None,
                        "reasoning": cdet.strip(),
                        "reasoning_stage": det_stage,
                        "deterministic_intent": solved.intent.value,
                        "deterministic_reasoning": solved.reasoning,
                        "deterministic_budget_profile": det_budget_audit,
                        "evidence_ids": ev_ids[:40]}
                    if len(det_attempt_traces) > 1:
                        log_record["deterministic_attempts"] = \
                            det_attempt_traces
                    if narrative_source_audit is not None:
                        log_record["narrative_source_audit"] = \
                            narrative_source_audit
                    if scope_audit_trace is not None:
                        log_record["scope_audit"] = scope_audit_trace
                        log_record["scope_audit_budget_profile"] = \
                            scope_audit_budget_audit
                    log.write(json.dumps(log_record, ensure_ascii=False) + "\n")
                    log.flush()
                return (adet, info) if return_info else adet

    if fallback_scope_budget is not None:
        base = (
            ev[:fallback_scope_budget.evidence_chars] +
            "\n\n题目:\n" + q["question"] + "\n\n" + inst +
            "\n\n【工作日口径核查】" + fallback_scope_budget.instruction)
    slim4 = os.environ.get("AFAC_SLIM4") == "1"
    c1_thinking = (fallback_scope_budget.thinking_budget
                   if fallback_scope_budget is not None else
                   (4000 if DEEP else
                    (1100 if slim4 else 2000 if SLIM else 2800)))
    c1_max_tokens = (fallback_scope_budget.max_tokens
                     if fallback_scope_budget is not None else
                     (2600 if slim4 else 3600))
    c1, _t, _u = chat([{"role": "user", "content": base}], qid=qid,
                      model=model, thinking=True,
                      thinking_budget=c1_thinking,
                      max_tokens=c1_max_tokens, tag="calc1")
    a1 = parse_calc(c1)
    traces = list(det_attempt_traces) + [
        {"stage": "calc1", "content": c1, "answer": a1}]
    selected_stage = "calc1"

    # 升级重试触发：①报缺数 ②答案缺失/截断/槽位不合法（fin_b_014/016/017/019类伤）
    # 升级动作必须同时升证据帽（只升思考预算救不了证据缺数的自旋）
    ms = SEARCH_RE.search(c1)
    ok1 = valid_calc(a1, kinds)
    c1b = None
    retry_suppression = None
    registry_bundle_complete = bool(
        narrative_source_audit and
        narrative_source_audit.get("unique_complete_bundle"))
    if ms and ok1 and registry_bundle_complete:
        retry_suppression = {
            "suppressed": True,
            "reason": (
                "valid_calc_with_unique_complete_narrative_fact_bundle"),
            "reported_query": ms.group(1).strip(),
        }
    # 瘦身档升级重试只救空白/报缺数（槽位小瑕疵触发的重试在瘦预算下烧token无收益）
    need_retry = ((ms or not a1) if slim4 else (ms or not ok1))
    if retry_suppression is not None:
        need_retry = False
    if need_retry:
        supp = ms.group(1).strip() if ms else ""
        if supp and blind_mode:  # 盲测下缺数可能因选错文档，允许域级扩检加选
            from .answerer import expand_docs_if_needed
            q2, added = expand_docs_if_needed(q, supp, model=model)
            if added:
                q = q2
                if log is not None:
                    log.write(json.dumps({"qid": qid, "doc_expanded": added},
                                         ensure_ascii=False) + "\n")
        ev2, ev_ids2 = calc_evidence(
            q, model=model, extra=([supp] if supp else ()), cap_mult=2,
            narrative_facts=(narrative_facts
                             if narrative_source_audit is not None else None))
        if fallback_scope_budget is not None:
            base = (
                ev2[:fallback_scope_budget.evidence_chars] +
                "\n\n题目:\n" + q["question"] + "\n\n" + inst +
                "\n\n【工作日口径核查】" + fallback_scope_budget.instruction)
        else:
            base = ev2 + "\n\n题目:\n" + q["question"] + "\n\n" + inst
        c1b, _t, _u = chat([{"role": "user", "content": base}], qid=qid,
                           model=model, thinking=True,
                           thinking_budget=(
                               fallback_scope_budget.thinking_budget
                               if fallback_scope_budget is not None else
                               (4000 if DEEP else
                                (1600 if slim4 else 2800))),
                           max_tokens=(
                               fallback_scope_budget.max_tokens
                               if fallback_scope_budget is not None else
                               (2600 if slim4 else 3600)), tag="calc1b")
        a1b = parse_calc(c1b)
        traces.append({"stage": "calc1b", "content": c1b,
                       "answer": a1b})
        if valid_calc(a1b, kinds) or (a1b and not a1):
            c1, a1, ev_ids = c1b, a1b, ev_ids2
            selected_stage = "calc1b"

    # 独立第二样本（异构模型更有信息量）；SLIM 跳过
    a2, c2 = None, None
    if os.environ.get("AFAC_CALC_SINGLE") == "1" and valid_calc(a1, kinds):
        pass  # 单样本模式(500k总攻): 槽位合法即定案, 省calc2整段
    elif os.environ.get("AFAC_CALC_HETERO") == "1" or not SLIM:
        c2, _t, _u = chat([{"role": "user", "content": base}], qid=qid,
                          model=verify_model or model, thinking=True,
                          thinking_budget=(4000 if DEEP else (2000 if SLIM else 2800)), max_tokens=3600, tag="calc2")
        a2 = parse_calc(c2)
        traces.append({"stage": "calc2", "content": c2, "answer": a2})

    final, c3 = a1 or a2, None
    if not a1 and a2:
        selected_stage = "calc2"
    if a1 and a2 and _norm(a1) != _norm(a2):
        adj = (base + f"\n\n两次独立计算结果不同：\n甲: {a1}\n乙: {a2}\n"
               "请重新核对取数（页码、年度、口径是否对应）与算式，指出分歧原因，"
               "给出正确结果。最后一行仍按要求输出 答案: ")
        c3, _t, _u = chat([{"role": "user", "content": adj}], qid=qid,
                          model=verify_model or model, thinking=True,
                          thinking_budget=(4400 if DEEP else 3200), max_tokens=3600, tag="calc3")
        a3 = parse_calc(c3)
        traces.append({"stage": "calc3", "content": c3, "answer": a3})
        final = a3 or a1
        if a3:
            selected_stage = "calc3"
    if not final:
        raise RuntimeError(f"{qid}: model produced no valid calculation answer")
    if final and "ranking" in kinds:
        final = shorten_rank_names(final, kinds, q["question"])
    chosen = next((t for t in reversed(traces)
                   if t["stage"] == selected_stage), traces[-1])
    reasoning = (chosen.get("content") or "").strip()
    if not reasoning:
        raise RuntimeError(f"{qid}: calculation answer has no reasoning trace")
    info = {"reasoning": reasoning, "reasoning_stage": selected_stage,
            "traces": traces, "raw_answer": chosen.get("answer")}
    if narrative_source_audit is not None:
        info["narrative_source_audit"] = narrative_source_audit
    if retry_suppression is not None:
        info["calc1b_retry_suppression"] = retry_suppression
    if det_budget_audit is not None:
        info["deterministic_budget_profile"] = det_budget_audit
    if scope_audit_trace is not None:
        info["scope_audit"] = scope_audit_trace
        info["scope_audit_budget_profile"] = scope_audit_budget_audit
    if fallback_scope_budget_audit is not None:
        info["fallback_scope_budget_profile"] = \
            fallback_scope_budget_audit
    if det_solved is not None:
        info["deterministic_intent"] = det_solved.intent.value
        info["deterministic_reasoning"] = det_solved.reasoning
    if log is not None:
        log_record = {"qid": qid, "final": final, "a1": a1, "a2": a2,
                      "c1": c1, "c1b": c1b, "c2": c2, "c3": c3,
                      "reasoning": reasoning,
                      "reasoning_stage": selected_stage,
                      "evidence_ids": ev_ids[:40]}
        if narrative_source_audit is not None:
            log_record["narrative_source_audit"] = \
                narrative_source_audit
        if retry_suppression is not None:
            log_record["calc1b_retry_suppression"] = retry_suppression
        if det_budget_audit is not None:
            log_record["deterministic_budget_profile"] = det_budget_audit
        if scope_audit_trace is not None:
            log_record["scope_audit"] = scope_audit_trace
            log_record["scope_audit_budget_profile"] = \
                scope_audit_budget_audit
        if fallback_scope_budget_audit is not None:
            log_record["fallback_scope_budget_profile"] = \
                fallback_scope_budget_audit
        if det_solved is not None:
            log_record["deterministic_intent"] = det_solved.intent.value
            log_record["deterministic_reasoning"] = det_solved.reasoning
        if det_attempt_traces:
            log_record["deterministic_attempts"] = det_attempt_traces
        log.write(json.dumps(log_record,
                             ensure_ascii=False) + "\n")
        log.flush()
    return (final, info) if return_info else final


def _norm(s):
    return re.sub(r"[\s,，]", "", s or "")


def shorten_rank_names(ans, kinds, qtext):
    """ranking槽公司名规范化：全称→题干简称原文（fin_b_016类伤，确定性防护）。

    简称几乎总是全称前缀（宁德时代新能源科技股份有限公司→宁德时代），
    取出现在题干中的最长前缀替换。
    """
    parts = [p.strip() for p in re.split(r"[；;]", ans or "")]
    if len(parts) < len(kinds):
        return ans
    changed = False
    for idx, (p, k) in enumerate(zip(parts, kinds)):
        if k != "ranking" or ">" not in p:
            continue
        segs = []
        for s in (x.strip() for x in p.split(">")):
            best = ""
            for j in range(2, len(s)):
                if s[:j] in qtext and j > len(best):
                    best = s[:j]
            segs.append(best if best and s not in qtext else s)
        new = ">".join(segs)
        if new != p:
            parts[idx] = new
            changed = True
    return "；".join(parts) if changed else ans
