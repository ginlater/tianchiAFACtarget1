"""逐题答题流程：记忆卡 + 定向检索 → 逐项判断 → 复核 → 答案规范化。

AFAC_STABLE=1 环境变量启用稳定模式：非计算域关闭思维链+低温采样（降方差降token）。
"""
import json, os, pathlib, re, threading

from . import retrieval
from .qwen_client import chat, DEFAULT_MODEL

ROOT = pathlib.Path(__file__).resolve().parents[2]

# ---- 领域配置：记忆卡标准查询（词法检索用，纯关键词，合规） ----
DIGEST_QUERIES = {
    "insurance": [
        "身故保险金 赔付 已交保费 现金价值 保单账户价值 基本保额",
        "身故给付比例 周岁 年生效对应日 个人账户价值 160% 140% 120%",
        "退保 犹豫期 现金价值 退还 保单价值",
        "养老年金 领取 开始 方式 年龄 保证领取",
        "保险责任 失能 护理 满期 生存金 减保",
        "投保范围 保险期间 交费方式 宽限期 保单贷款",
        "责任免除 不承担 酒后驾驶 艾滋病 遗传性疾病 既往症 自杀 故意",
        "特定药品 指定药店 处方 审核 直接结算 院外",
        "免赔额 抵扣 基本医疗保险 统筹 个人账户 其他商业保险 补偿",
        "住院医疗 门诊 报销比例 社保 未经社会基本医疗保险结算",
    ],
    "financial_reports": [
        "营业收入 归属于上市公司股东的净利润 同比 主要会计数据",
        "经营活动产生的现金流量净额 每股收益 加权平均净资产收益率",
        "研发投入 研发费用 占营业收入比例 研发人员",
        "利润分配 分红 派息 每10股 现金股利 回购",
        "分红政策 净利润的 比例 股东回报规划 中期分红 特别分红",
        "总资产 归属于上市公司股东的净资产 营业成本 毛利率",
    ],
    "financial_contracts": [
        "发行人 发行金额 发行规模 票面利率 期限 品种",
        "主体信用评级 债项评级 评级机构 展望",
        "主承销商 簿记管理人 受托管理人 联席",
        "募集资金用途 偿还 补充流动资金",
        "违约 逾期利息 违约金 兑付日 回售 赎回 付息",
        "违约金 计算方式 本金 利息 票面利率 延迟支付 惩罚",
        "发行公告 上市 日期 网上 申购 缴款",
    ],
}
DIGEST_DOMAINS = set(DIGEST_QUERIES)


def _use_digest(domain):
    """瘦身档全局关记忆卡，但保险域例外保留：16个小文档条款卡摊到20题，
    是准确率主杠杆（slim4保险14错法医结论：11道证据饿死）。"""
    if domain not in DIGEST_DOMAINS:
        return False
    if os.environ.get("AFAC_NO_DIGEST") != "1":
        return True
    keep = os.environ.get("AFAC_DIGEST_KEEP", "insurance")
    return domain in keep.split(",")

_digest_cache = {}
_digest_lock = threading.Lock()
_digest_locks = {}

FMT_NAME = {"mcq": "单选题(唯一正确答案)", "multi": "多选题(一个或多个正确)",
            "tf": "判断题(A/B其一)"}


_INS_TITLES = None


def _doc_title(doc_id):
    # 保险文档身份锚：PDF标题层失真/雷同（doc9与10同名、doc16乱码），
    # 用离线词法抽取的 公司+产品 映射（ins_b_004/008/017类伤）
    global _INS_TITLES
    meta = retrieval.docs_meta()[doc_id]
    if meta["domain"] == "insurance":
        if _INS_TITLES is None:
            p = ROOT / "work" / "processed_data" / "insurance_titles.json"
            _INS_TITLES = json.load(open(p)) if p.exists() else {}
        t = _INS_TITLES.get(doc_id)
        if t:
            return f"{t['company']}{t['product']}"
    return meta["title"]


DIGEST_INST = {
    "insurance": (
        "请提取该保险条款的【事实卡】，必须完整包含（该产品条款没有的项写'无'）："
        "(1)产品全名与类型;(2)身故保险金计算规则——含所有年龄段/情形的给付比例或公式，逐档列出;"
        "(3)现金价值与退保规则;(4)年金/生存金/满期金领取规则(开始时间、方式、保证领取);"
        "(5)账户价值与结算利率规则;(6)医疗险报销规则——免赔额数值、免赔额可用什么抵扣/"
        "不可用什么抵扣、报销比例(经社保结算与未经社保结算)、特定药品要求;"
        "(7)责任免除要点(逐条列出免责情形);(8)其他特殊责任(失能护理/减保/贷款);"
        "(9)犹豫期天数与退费规则(主规则及例外原句);(10)施救费用/特殊费用的赔偿上限。"
        "保留原文精确数字与比例，标注来源页码如(P12)。只写文档明确存在的内容。900字以内。"),
    "financial_reports": (
        "请提取该年报的【事实卡】：(1)公司名与报告年度;(2)主要会计数据——营业收入、"
        "归母净利润、扣非净利润、经营活动现金流量净额、每股收益、总资产、归母净资产，"
        "本年与上年数值及同比增减率都要;(3)研发投入金额与占营业收入比例、研发人员数;"
        "(4)利润分配——分别列出本年度利润分配预案原句(每10股派X元)、已实施的中期/特别"
        "分红、全年合计，逐项标明口径，勿合并;(5)分季度或分业务重要数据。"
        "保留原文精确数字与单位，标注页码如(P12)。900字以内。"),
    "financial_contracts": (
        "请提取该文档（债券募集说明书/重组报告书等）的【事实卡】：(1)发行人全称;"
        "(2)债券名称、品种、发行金额/上限、期限、票面利率或定价方式;(3)主体评级、"
        "债项评级、评级机构、展望;(4)主承销商、簿记管理人、受托管理人等中介机构"
        "——逐机构标明角色;(5)募集资金用途;(6)付息日与各品种兑付日（含回售/赎回"
        "情形下的兑付日，逐品种列出）;(7)违约情形、逾期利息/违约金计算方式原句;"
        "(8)重要日期（发行日、公告日、上市日等）。"
        "保留原文精确数字与公式，标注页码如(P12)。1200字以内。"),
}

WHOLE_DOC_LIMIT = 40000  # 字符；小文档直接全文构卡，防条款漏检


def build_digest(doc_id, domain, qid="_digest", model=DEFAULT_MODEL):
    """懒构建文档事实卡（Qwen 生成，token 计入台账，跨题复用=Agent记忆）。"""
    with _digest_lock:
        if doc_id in _digest_cache:
            return _digest_cache[doc_id]
        lock = _digest_locks.setdefault(doc_id, threading.Lock())
    with lock:
        with _digest_lock:
            if doc_id in _digest_cache:
                return _digest_cache[doc_id]
        full = retrieval.doc_path(doc_id).read_text(encoding="utf-8")
        if len(full) <= WHOLE_DOC_LIMIT:
            raw = full
        else:
            idx = retrieval.doc_index(doc_id)
            # 轮询交错：每个查询轮流出一块，防止预算被前面的查询挤占
            per_q = [idx.search(q, k=5) for q in DIGEST_QUERIES[domain]]
            seen, parts, total = set(), [], 0
            for rank in range(5):
                for hits in per_q:
                    if rank >= len(hits):
                        continue
                    c, _s = hits[rank]
                    if c["id"] in seen:
                        continue
                    seen.add(c["id"])
                    tag = f"P{c['page']}" if c["page"] else c["id"].split("#")[1]
                    piece = f"[{tag}] {c['text']}"
                    if total + len(piece) > (8000 if __import__("os").environ.get("AFAC_SLIM")=="1" else 11500):
                        continue
                    total += len(piece)
                    parts.append(piece)
            raw = "\n".join(parts)
        prompt = (
            f"文档《{_doc_title(doc_id)}》(编号 {doc_id}) 内容如下"
            + ("（注意：以下仅为按主题选取的片段，不是全文）" if len(full) > WHOLE_DOC_LIMIT else "")
            + "。\n" + DIGEST_INST[domain]
            + "\n重要：事实卡只记录确实存在的内容，严禁写'文档未提供/未提及某内容'"
              "之类的否定性断言（你看到的可能只是片段）。\n\n" + raw)
        content, _r, _u = chat([{"role": "user", "content": prompt}],
                               qid=qid, model=model, thinking=False,
                               max_tokens=(1200 if os.environ.get("AFAC_SLIM")=="1" else 1800), tag=f"digest:{doc_id}")
        card = f"《{_doc_title(doc_id)}》({doc_id}) 事实卡:\n{content}"
        with _digest_lock:
            _digest_cache[doc_id] = card
        return card


def save_digests(path):
    with _digest_lock:
        json.dump(_digest_cache, open(path, "w"), ensure_ascii=False, indent=1)


def load_digests(path):
    p = pathlib.Path(path)
    if p.exists():
        with _digest_lock:
            _digest_cache.update(json.load(open(p)))


# ---------------- 证据组装 ----------------

def gather_evidence(q, k_opt=2, k_q=3, cap=9000, extra_queries=()):
    doc_ids = q["doc_ids"]
    queries = [q["question"]] + [f"{q['question'][:40]} {t}" for t in q["options"].values()]
    # 数字微调陷阱对策：含数字的选项补一条去数字查询（防选项数字与原文不同时匹配失败）
    for t in q["options"].values():
        stripped = re.sub(r"[0-9.,%％]+", " ", t)
        if stripped != t and len(stripped.strip()) >= 8:
            queries.append(stripped)
    # 术语同义扩展（小表，纯词法）：题目措辞与文档法律用语的常见鸿沟
    SYN = [("违约利息", "逾期利息 违约金"), ("公告日期", "公告 发布日"),
           ("手动", "人工"), ("兑付日", "兑付日 到期日 回售"),
           # 金融文本写法鸿沟（题目用语→文档常见写法）
           ("一季度", "1-3月 Q1"), ("二季度", "4-6月 Q2"),
           ("三季度", "7-9月 Q3"), ("四季度", "10-12月 Q4"),
           ("上半年", "1-6月 H1"), ("下降", "同减 减少 下滑 同比-"),
           ("增长", "同增 增加 提升"), ("销量", "销量 销")]
    for t in list(queries):
        for a, b in SYN:
            if a in t:
                queries.append(t.replace(a, b))
    queries += list(extra_queries)
    # 关键词逐文档强制检索：短金融术语在每份文档单独取top-1并保护
    # （修复类缺口：担保人/母公司列/地震免责/资产负债率/募集资金用途 等关键句被长查询稀释）
    LEXICON = ["担保人", "担保", "母公司", "募集资金用途", "资产负债率", "流动比率",
               "速动比率", "责任免除", "免赔额", "犹豫期", "诉讼时效", "转股价格",
               "锁定期", "评级", "受托管理人", "兑付", "分红", "研发投入",
               "每股收益", "现金流量净额", "施行", "工作日", "自然日"]
    qtext = q["question"] + " " + " ".join(q["options"].values())
    hard_kws = [kw for kw in LEXICON if kw in qtext][:6]
    for m in re.finditer(r"[“\"《]([^”\"》]{2,12})[”\"》]", qtext):
        if len(hard_kws) < 8:
            hard_kws.append(m.group(1))
    forced = []
    for kw in hard_kws:
        for d in doc_ids:
            hits_kw = retrieval.doc_index(d).search(kw, k=2)
            cands_kw = [c for c, _s in hits_kw if kw in c["text"]]
            if not cands_kw:
                continue
            # 假阳性防护：优先取关键词所在行含数字/百分号的块（样板条款句常无取值）
            with_num = [c for c in cands_kw if any(
                kw in ln and re.search(r"[\d％%]", ln)
                for ln in c["text"].split("\n"))]
            forced.append((with_num or cands_kw)[0])
    # 跨查询同块取最高分（低分先占坑会挤掉后续强命中——已修复的召回bug）
    # 每条查询的top-1受保护，预算截断时优先保留（防单选项关键证据被全局高分挤掉）
    best, chunk_by_id, protected = {}, {}, set()
    n_core = 1 + len(q["options"])  # 题干+原始选项查询才享受top-1保护
    for i, query in enumerate(queries):
        k = k_q if i == 0 else k_opt
        hits = retrieval.search_docs(doc_ids, query, k_per_doc=k)
        if hits and i < n_core:
            protected.add(hits[0][0]["id"])
        for c, s in hits:
            cid = c["id"]
            chunk_by_id[cid] = c
            if s > best.get(cid, 0):
                best[cid] = s
    for c in forced:
        cid = c["id"]
        chunk_by_id[cid] = c
        protected.add(cid)
        best[cid] = max(best.get(cid, 0), 1e9)  # 强制块置顶
    out = [(chunk_by_id[cid], s) for cid, s in best.items()]
    # 目录/图表索引块降权（占坑但无正文信息量）
    def _is_toc(c):
        t = c["text"]
        return t.count("……") >= 3 or t.count("...") >= 6 or \
            len(re.findall(r"^[图表]：", t, re.M)) >= 4
    out.sort(key=lambda x: (x[0]["id"] not in protected, _is_toc(x[0]), -x[1]))
    kept, total = [], 0
    for c, s in out:
        piece_len = len(c["text"]) + 20
        # 保护块（各查询top-1+强制关键词块）不受帽截断——fc_b_003类伤：
        # A/C选项支持块已召回且受保护，仍被2200字帽挤出
        if c["id"] not in protected and total + piece_len > cap:
            continue
        total += piece_len
        kept.append(c)
    # 每份文档保底1块正文证据（防证据帽把第二来源整份挤掉——res_b_002/015类伤）
    have = {c["doc_id"] for c in kept}
    for d in doc_ids:
        if d in have:
            continue
        cand = next((c for c, _s in out
                     if c["doc_id"] == d and not _is_toc(c)), None)
        if cand is None:
            continue
        while kept and sum(len(c["text"]) + 20 for c in kept) \
                + len(cand["text"]) + 20 > cap:
            victims = [c for c in kept
                       if c["id"] not in protected and c["doc_id"] != d]
            if not victims:
                break
            kept.remove(victims[-1])
        kept.append(cand)
    kept.sort(key=lambda c: (c["doc_id"], c["page"] or 0,
                             int(c["id"].split("#c")[1])))
    parts = []
    for c in kept:
        tag = f"{c['doc_id']} P{c['page']}" if c["page"] else c["id"]
        parts.append(f"【{tag}】{c['text']}")
    return "\n\n".join(parts), kept, protected


def _render(kept):
    parts = []
    for c in kept:
        tag = f"{c['doc_id']} P{c['page']}" if c["page"] else c["id"]
        parts.append(f"【{tag}】{c['text']}")
    return "\n\n".join(parts)


def evidence_block(q, model=DEFAULT_MODEL, extra_queries=()):
    """返回 (证据文本, chunk列表, 受保护id集合, 记忆卡文本)。"""
    domain = q["domain"]
    blocks, digests = [], ""
    if os.environ.get("AFAC_NO_DIGEST") == "1" and not _use_digest(domain):
        digests = "涉及文档:\n" + "\n".join(
            f"- {d}: 《{_doc_title(d)}》" for d in q["doc_ids"])
        blocks.append(digests)
        cap = (2200 if os.environ.get("AFAC_SLIM4") == "1" else 3600) \
            + 1000 * max(0, len(q["doc_ids"]) - 2)
        ev, kept, prot = gather_evidence(q, k_opt=2, k_q=2, cap=cap,
                                         extra_queries=extra_queries)
        blocks.append("原文片段证据:\n" + ev)
        return "\n\n".join(blocks), kept, prot, digests
    if domain in DIGEST_DOMAINS:
        digests = "\n\n".join(build_digest(d, domain, model=model)
                              for d in q["doc_ids"])
        blocks.append(digests)
        # 大文档域(合同/年报,单文档30万字符)证据基数更大；多文档题按文档数增配
        base_cap = 9500 if domain == "financial_contracts" else \
            8500 if domain == "financial_reports" else 6000
        if os.environ.get("AFAC_DEEP") == "1":
            base_cap = int(base_cap * 1.6)
        if SLIM:
            base_cap = int(base_cap * 0.6)
        cap = base_cap + 2000 * max(0, len(q["doc_ids"]) - 2)
        ev, kept, prot = gather_evidence(q, k_opt=3, k_q=2, cap=cap,
                                         extra_queries=extra_queries)
    else:
        titles = "\n".join(f"- {d}: 《{_doc_title(d)}》" for d in q["doc_ids"])
        digests = "涉及文档:\n" + titles
        blocks.append(digests)
        # research 选项数字散布多文档，覆盖优先给较大预算
        k_opt, cap = (4, 10000) if domain == "research" else (3, 8500)
        if os.environ.get("AFAC_DEEP") == "1":
            cap = int(cap * 1.6)
        ev, kept, prot = gather_evidence(q, k_opt=k_opt, k_q=4, cap=cap,
                                         extra_queries=extra_queries)
    blocks.append("原文片段证据:\n" + ev)
    return "\n\n".join(blocks), kept, prot, digests


# ---------------- 作答与解析 ----------------

ANSWER_RE = re.compile(r"答案[:：]\s*([A-D]{1,4})")
SEARCH_RE = re.compile(r"补充检索[:：]\s*(.+)")


def normalize(ans, fmt):
    letters = [c for c in ans.upper() if c in "ABCD"]
    if not letters:
        return ""
    if fmt in ("mcq", "tf"):
        return letters[0]
    return "".join(sorted(set(letters)))


_FALLBACK_BAD = re.compile(r"不选|无法|判断[:：]|入选|分析|证据|复核|标准")


def parse_answer(content, fmt):
    m = list(ANSWER_RE.finditer(content))
    if m:
        return normalize(m[-1].group(1), fmt)
    # 兜底：仅接受短纯字母行；含判定过程词的行不得作答案源
    # （防"判断: A不选 B不选 C不选 D不选"被反转成ABCD——fin_b_007/012类伤）
    for line in reversed(content.strip().splitlines()):
        s = line.strip()
        if len(s) > 12 or _FALLBACK_BAD.search(s):
            continue
        cand = normalize(s, fmt)
        if cand:
            return cand
    return ""


def _q_text(q):
    opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    return f"题目({FMT_NAME[q['answer_format']]}):\n{q['question']}\n\n选项:\n{opts}"


JUDGE_STD = (
    "判分标准（重要）:\n"
    "-1. 【证据优先于常识】文档是唯一事实来源。本评测题目会故意设置与常识相近但被"
    "文档修改过的数值/日期/条件（或文档为2025-2026年新规，晚于你的知识）。凡与你"
    "记忆中的'常识'冲突的，一律以证据原文为准；凡选项声称的具体数值/日期/时限在"
    "证据中找不到对应原文的，视为证据缺失，必须发起补充检索，严禁凭印象补位。\n"
    "0. 先明确题干的【选择标准】——题目要选的是什么（如'可以获得赔付的产品'、"
    "'条款给出了公式的产品'、'描述正确的选项'）。选项常为'对象（括注说明）'形式，"
    "判断的是该对象是否满足题干选择标准，而不是括注内容是否正确！括注若本身说明"
    "该对象不满足标准（如'不赔'、'未给出公式'），则该选项不应入选。\n"
    "1. 描述判断类选项通常由文档原句轻度转述而来。若选项能对应到证据中的某句话"
    "（数值、主体、年份、趋势方向一致），即判'对'——即使选项省略了原句的限定词"
    "（如指标名前缀'除…'、'剔除…'）、措辞不同或表述不完整。出题人只会在可核对的"
    "具体元素上做手脚：数值、机构/主体名、年份、方向、自动/手动这类关键词。"
    "特别地：若证据中存在与选项字面一致的原句（数值+主体对应），直接判'对'，"
    "不要用你换算出的其他口径数字去推翻字面对应的原句。\n"
    "2. 仅当选项与证据存在实质矛盾时判'错'：数值/日期/主体错误、条件或因果颠倒、"
    "张冠李戴、程度或趋势方向相反、无中生有。判'错'必须指出可核对的具体事实错误；"
    "选项末尾的模糊评价性表述（如'支撑了…发展/体系'、'反映了…趋势'）不构成判错依据。\n"
    "3. 计算题严格按条款规则计算，注意年龄分档、免赔条件、已领扣减等细节。\n"
    "4. 比较类选项（谁高谁低、早于晚于）必须找到双方数值逐一核对。\n"
    "4b.【存在性判读】选项中的术语在文档中对应多个相近条款时（如'违约利息'既可能"
    "对应'逾期利息'条款也可能对应'违约金'条款；'兑付日'既有名义兑付日也有回售/赎回"
    "情形下的兑付日），必须把所有相关条款全部核对；只要其中任一条款支持选项表述，"
    "即判'对'。不得只用最先想到的那个条款否定选项。同一事项存在多个披露口径的数字"
    "时（全年合计/年末预案、含中期/不含中期），选项与其中任一口径的原句数字一致即"
    "判'对'，不得用另一口径推翻。\n"
    "4c.【卡片非全集】文档事实卡是摘要，卡上没有 ≠ 文档没有。判某选项'无中生有'前，"
    "必须先确认原文片段证据中确实检索不到，且已发起过补充检索。\n"
    "5. 场景题注意事故情形与保障范围的匹配（如营运交通工具意外险只保乘坐营运交通"
    "工具期间的意外；题干未说明场景符合时不得假定符合）。\n"
    "6.【有无类题】题干问'哪些产品的条款中明确规定了X'时：某产品经补充检索后证据中"
    "仍无X相关条款的，判为'未规定'即不选。此类题考察的就是条款有无，缺失即否定，"
    "不适用'证据不足不判'原则。\n"
    "7.【例外不触发】条款主规则附带例外情形（'但…的除外/需扣除…'）时，题干给定场景"
    "未触发例外的，按主规则判断，例外分句不影响结论。\n"
    "8.【槽位绑定】句子含多个数额对应多个用途/主体时（'其中X亿用于A，Y亿用于B'），"
    "必须逐一配对，明确题目问的是哪个用途，严禁取最大或最先出现的数额。\n"
    "9.【列头绑定】财务报表多列并排（合并本期/合并上期/母公司本期/母公司上期）时，"
    "先数清列头再取数；题目问母公司口径必须取母公司列，严禁用合并列充当。\n"
    "10.【字面高于常识】判断题表述与文档某句逐字或近逐字一致的，直接判'正确'，"
    "即使与你的行业常识相悖（如政府性基金作担保人）——文档是唯一事实标准。\n"
    "11.【数值容差】选项数值与证据数值仅差四舍五入（如32.27与32.3）视为一致。\n"
    "12.【有无类逐文档】'哪些产品/文件明确规定X'必须对每个选项的文档分别检索X及其"
    "同义词（含汉字数字写法），逐文档记录有/无后再作答。\n"
    "13.【口径词严格核算】相对/绝对、百分点/百分比、降幅/增幅、差值/比值、"
    "环比/同比属于口径词，不适用宽容转述判分：选项数字与其口径定义核算结果"
    "不符即判错（如指标从0.95%降至0.94%，'降幅0.01%'为错——相对降幅≈1.05%，"
    "绝对变化是0.01个百分点）。\n"
    "校准示例（务必对齐此口径，示例为通用同构案例）:\n"
    "- 原文'银行剔除表外理财杠杆率从2.1倍升至3.8倍'，选项'银行理财杠杆率从2.1倍"
    "升至3.8倍' → 判对。数值与趋势一致，省略指标前缀/限定词不算错。\n"
    "- 原文'线上渠道占比提升至47%，带动整体销量较快增长'，选项'线上渠道占比超过"
    "40%，支撑了公司业务体系发展' → 判对。'超过40%'与47%相容，结尾评价性表述"
    "不作判错依据。\n"
    "- 原文'部署812条风控规则实现自动拦截'，选项'部署812条风控规则实现人工拦截'"
    " → 判错。自动/人工关键词反转，实质矛盾。\n"
    "- 原文'经审核确认属于重大缺陷且由系统原因导致的，应当自确认之日起20个工作日"
    "内提交整改报告'，选项'确认重大缺陷的应在20个工作日内提交整改报告' → 判对。"
    "省略次要前提不算错，时限与主体一致。\n"
    "- 原文利润分配段'拟每10股派发现金12元(含税)'（全年含已实施中期合计15元），"
    "选项'拟每10股派12元' → 判对。与任一披露口径的原句一致即可，勿用合计口径推翻。\n"
    "- 多份保险合同赔付计算题：各选项按'每份合同独立按各自条款公式计算'列出金额时，"
    "按该口径逐份计算并求和判断，不引入题干未要求的多合同赔付协调/损失补偿封顶。\n"
    "- 题干'李某因意外事故骨折住院'（未说明乘机），选项'航空意外险可赔付' → 不入选。"
    "该险种仅保障乘机期间意外，题干未说明场景即视为不符合。"
)

if os.environ.get("AFAC_SLIM4") == "1":  # 瘦身档：规则全保留，示例压缩为一对天平砝码
    JUDGE_STD = JUDGE_STD.split("校准示例")[0].rstrip() + (
        "\n校准示例(口径天平): 原文'银行剔除表外理财杠杆率从2.1倍升至3.8倍'，"
        "选项'银行理财杠杆率从2.1倍升至3.8倍'→判对（原句轻度转述/省略限定词仍判对）。"
        "选项'公司明确提出2027年产能翻番目标'而全部证据无任何产能目标表述→无中生有判错。\n"
        "多选题中，选项是文档观点/事实的概括转述或合理归纳且与证据无矛盾时应入选；"
        "'无中生有'仅指选项核心事实(数字/主体/方向)在证据中无对应，"
        "不得因证据片段未覆盖个别措辞而弃选整个选项。")

R1_INST = (
    "你是金融文档审读专家。严格依据上述证据逐项判断每个选项的真伪，"
    "证据不足的选项不得臆断。涉及计算的题先列出各产品/公司的规则与数值再计算。\n"
    + JUDGE_STD + "\n"
    "输出格式:\n"
    "选择标准: <一句话复述题干要求选出什么>\n"
    "分析: <每个选项一行,引用证据页码,判断该选项是否满足选择标准及理由>\n"
    "判断: A入选/不选 B入选/不选 C入选/不选 D入选/不选\n"
    "答案: <字母>\n"
    "若关键证据缺失导致无法判断某选项，最后一行输出: 补充检索: <用于查找证据的关键词>"
)

R2_INST = (
    "你是复核专家。上面是题目、证据与初判答案。请忽略初判的结论，"
    "独立地按题干的选择标准逐项复核。\n" + JUDGE_STD + "\n"
    "复核重点: ①初判是否搞错了选择标准（把'括注分析正确'当成了'该选项应入选'）；"
    "②数字/日期/主体是否与证据相符；③是否漏选了实质满足标准的选项；"
    "④是否因过度严苛把概括性正确的选项误判为错；⑤是否把证据中不存在的内容当成了依据。\n"
    "输出格式:\n选择标准: <一句话>\n复核: <每个选项一行>\n答案: <字母>\n"
    "若关键证据缺失导致无法判断某选项，最后一行输出: 补充检索: <关键词>"
)


def expand_docs_if_needed(q, query, model=DEFAULT_MODEL):
    """B模式：补检查询在已选文档中命中弱时，扩展到全域语料动态加选文档。"""
    from . import doc_select  # 延迟导入避免环
    cur = set(q["doc_ids"])
    in_doc = retrieval.search_docs(q["doc_ids"], query, k_per_doc=1)
    best_in = in_doc[0][1] if in_doc else 0.0
    idx = doc_select.domain_doc_index(q["domain"])
    ext = [(c, s) for c, s in idx.search(query, k=3) if c["doc_id"] not in cur]
    if ext and ext[0][1] > best_in * 1.5:
        new_doc = ext[0][0]["doc_id"]
        return dict(q, doc_ids=q["doc_ids"] + [new_doc]), new_doc
    return q, None


CALC_DOMAINS = ("insurance", "financial_reports")
STABLE = os.environ.get("AFAC_STABLE") == "1"
DEEP = os.environ.get("AFAC_DEEP") == "1"  # 深挖模式：低置信题复核用
SLIM = os.environ.get("AFAC_SLIM") == "1"   # 瘦身模式：单样本+紧证据(终跑省token)
STABLE_DOMAINS = ("regulatory",) if not STABLE else \
    ("regulatory", "financial_contracts", "research")
VERIFY_MODEL = os.environ.get("AFAC_VERIFY_MODEL", "")


def _think(q):
    """法规域实测无思维链20/20且省35%token；stable模式扩展到更多域。"""
    return q["domain"] not in STABLE_DOMAINS


def _vote_letters(answers, fmt):
    """逐选项多数决：字母在≥半数答案中出现即入选。"""
    answers = [a for a in answers if a]
    if not answers:
        return ""
    if fmt in ("mcq", "tf"):
        from collections import Counter
        return Counter(answers).most_common(1)[0][0]
    need = len(answers) / 2
    letters = [l for l in "ABCD"
               if sum(l in a for a in answers) > need - 1e-9]
    return "".join(letters)


LEAN_R2 = os.environ.get("AFAC_LEAN_R2") == "1"


def answer_question(q, model=DEFAULT_MODEL, log=None, blind_mode=False):
    qid, fmt = q["qid"], q["answer_format"]
    think_r1 = 2200 if q["domain"] in CALC_DOMAINS else 1900
    if DEEP:
        think_r1 = 3400
    if SLIM:
        think_r1 = 1600
    if os.environ.get("AFAC_SLIM4") == "1":
        think_r1 = 1100
    ev, kept, prot, digests = evidence_block(q, model=model)
    ev_ids = [c["id"] for c in kept]
    base = ev + "\n\n" + _q_text(q)

    c1, r1think, _ = chat(
        [{"role": "user", "content": base + "\n\n" + R1_INST}],
        qid=qid, model=model, thinking=_think(q), thinking_budget=think_r1,
        max_tokens=4000, tag="r1")
    ans1 = parse_answer(c1, fmt)

    # 补充检索一轮
    ms = SEARCH_RE.search(c1)
    if ms:
        supp_q = ms.group(1).strip()
        if blind_mode:  # B模式：证据缺口可能因选错文档，允许域级扩检加选
            q, added = expand_docs_if_needed(q, supp_q, model=model)
            if added and log is not None:
                log.write(json.dumps({"qid": qid, "doc_expanded": added},
                                     ensure_ascii=False) + "\n")
        ev2, kept, prot, digests = evidence_block(q, model=model,
                                                  extra_queries=[supp_q])
        ev_ids = [c["id"] for c in kept]
        base = ev2 + "\n\n" + _q_text(q)
        c1b, _t, _ = chat(
            [{"role": "user", "content": base + "\n\n" + R1_INST.rsplit("\n", 1)[0]}],
            qid=qid, model=model, thinking=_think(q), thinking_budget=think_r1,
            max_tokens=4000, tag="r1b")
        if parse_answer(c1b, fmt):
            c1, ans1 = c1b, parse_answer(c1b, fmt)

    final, c2, ans2 = ans1, None, None
    # tf 复核实测 0/20 翻转，跳过省 token；multi/mcq 保留盲复核；SLIM 全部单样本
    if (not SLIM and fmt in ("multi", "mcq")) or not ans1:
        if LEAN_R2:
            # 精简复核证据：记忆卡 + r1引用页的块 + 每选项受保护块（独立性保留）
            cited = set(re.findall(r"P(\d+)", c1))
            sub = [c for c in kept
                   if c["id"] in prot or (c["page"] and str(c["page"]) in cited)]
            r2_base = (digests + "\n\n原文片段证据:\n" + _render(sub)
                       + "\n\n" + _q_text(q))
        else:
            r2_base = base
        c2, _t, _ = chat(
            [{"role": "user", "content": r2_base + "\n\n" + R2_INST}],
            qid=qid, model=VERIFY_MODEL or model, thinking=_think(q),
            thinking_budget=1500, max_tokens=2600, tag="r2")
        ans2 = parse_answer(c2, fmt)
        if ans2 and ans2 != ans1:
            # 定向仲裁：只带分歧选项的针对性证据，三样本逐选项多数决
            disputed = [l for l in "ABCD"
                        if (l in (ans1 or "")) != (l in (ans2 or ""))]
            dq = [f"{q['question'][:30]} {q['options'][l]}" for l in disputed
                  if l in q["options"]]
            ev3, _k3, _p3 = gather_evidence(q, k_opt=3, k_q=2, cap=5500,
                                            extra_queries=dq)
            dtxt = "\n".join(f"{l}. {q['options'][l]}" for l in disputed
                             if l in q["options"])
            adj = ("原文片段证据:\n" + ev3 + "\n\n" + _q_text(q) +
                   f"\n\n两次独立判断在以下选项上有分歧:\n{dtxt}\n"
                   "请仅针对这些分歧选项逐项核对证据并给出该选项是否入选的结论。\n"
                   + JUDGE_STD + "\n输出格式:\n仲裁: <分歧选项逐项>\n"
                   "答案: <完整最终答案字母>")
            c3, _t, _ = chat([{"role": "user", "content": adj}],
                             qid=qid, model=VERIFY_MODEL or model, thinking=True,
                             thinking_budget=2600, max_tokens=3000, tag="r3")
            ans3 = parse_answer(c3, fmt)
            final = _vote_letters([ans1, ans2, ans3], fmt) or ans3 or ans2
        elif ans2:
            final = ans2
    if not final:
        final = "A"  # 绝不留空（空=判错）

    if log is not None:
        log.write(json.dumps({
            "qid": qid, "final": final, "r1": ans1, "r2": ans2,
            "c1": c1, "c2": c2, "evidence_ids": ev_ids[:40]},
            ensure_ascii=False) + "\n")
        log.flush()
    return final, {"r1": ans1, "r2": ans2, "c1": c1}
