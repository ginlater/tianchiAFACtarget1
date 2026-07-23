"""B榜文档盲检：文档级词法BM25粗召回 + Qwen从候选卡片精选。

合规：粗召回为纯词法统计；语义选择由 Qwen 完成（token 计入台账）。
"""
import json, pathlib, re

from . import retrieval
from .qwen_client import chat, DEFAULT_MODEL, LEDGER

ROOT = pathlib.Path(__file__).resolve().parents[2]

_DOC_BM25 = {}   # domain -> BM25 over doc-level pseudo-chunks
INDEX_HEAD = 30000


def _doc_card(doc_id):
    m = retrieval.docs_meta()[doc_id]
    title = _display_title(doc_id)
    bits = [f"{doc_id}: 《{title}》"]
    if m.get("column"):
        bits.append(f"栏目:{m['column']}")
    if m.get("pub_date"):
        bits.append(f"日期:{m['pub_date'][:10]}")
    return " ".join(bits)


def domain_doc_index(domain):
    if domain not in _DOC_BM25:
        chunks = []
        for doc_id, m in retrieval.docs_meta().items():
            if m["domain"] != domain:
                continue
            text = retrieval.doc_path(doc_id).read_text(encoding="utf-8")
            # 标题加权：重复3次并入索引文本
            idx_text = (m["title"] + "\n") * 3 + text[:INDEX_HEAD]
            chunks.append({"id": doc_id, "doc_id": doc_id, "page": None,
                           "text": idx_text})
        _DOC_BM25[domain] = retrieval.BM25(chunks)
    return _DOC_BM25[domain]


def coarse_candidates(q, k=18):
    """regulatory(513篇)用BM25并集召回；其余领域文档少，直接全量进候选。
    瘦身档：非regulatory也走BM25预筛top-12（纯词法，合规）。"""
    if q["domain"] != "regulatory":
        all_ids = [d for d, m in retrieval.docs_meta().items()
                   if m["domain"] == q["domain"]]
        if __import__("os").environ.get("AFAC_SLIM4") == "1" and len(all_ids) > 9:
            idx = domain_doc_index(q["domain"])
            query = q["question"] + " " + " ".join(q["options"].values())
            ids, seen = [], set()
            for c, _s in idx.search(query, k=9):
                if c["doc_id"] not in seen:
                    seen.add(c["doc_id"])
                    ids.append(c["doc_id"])
            return ids or all_ids
        return all_ids
    idx = domain_doc_index(q["domain"])
    query = q["question"] + " " + " ".join(q["options"].values())
    ids, seen = [], set()
    for c, _s in idx.search(query, k=k):
        if c["doc_id"] not in seen:
            seen.add(c["doc_id"])
            ids.append(c["doc_id"])
    subqueries = [q["question"]] + [f"{q['question'][:30]} {v}"
                                    for v in q["options"].values()]
    for sq in subqueries:
        for c, _s in idx.search(sq, k=4):
            if c["doc_id"] not in seen:
                seen.add(c["doc_id"])
                ids.append(c["doc_id"])
    return ids


SEL_RE = re.compile(r"\[.*?\]", re.S)

# 样板噪声：研报免责声明/分析师信息、页眉页码、募集说明书声明段
BOILER = re.compile(
    r"请务必阅读|免责条款|证券研究报告|执业证书编号|SAC|分析师|联系人|"
    r"^\d+$|^P\d+$|研究助理|@|电话|邮箱|"
    r"声明及提示|发行人声明|虚假记载|误导性陈述|重大遗漏|真实性、准确性")

_BAD_TITLE = re.compile(r"声明|提示|指引|目录|凡欲认购|信息披露义务|发行人及其董事")


def _display_title(doc_id):
    """标题为样板句时（text13类伤：卡片0次出现'铁路'），
    纯词法回退：取正文头部高频《…》短语或 XX债券/说明书 模式串。"""
    m = retrieval.docs_meta()[doc_id]
    t = re.sub(r"^标题：", "", m["title"])
    if not _BAD_TITLE.search(t):
        return t
    raw = retrieval.doc_path(doc_id).read_text(encoding="utf-8")[:8000]
    names = re.findall(r"《([^》]{4,30})》", raw) + \
        re.findall(r"([一-龥A-Za-z0-9]{2,18}(?:债券|票据)(?:募集说明书)?)", raw)
    _GENERIC = re.compile(r"^(本次|本期|该|上述|次级|中的)")
    if names:
        from collections import Counter
        best = Counter(n for n in names if not _BAD_TITLE.search(n)
                       and not _GENERIC.match(n) and len(n) >= 6).most_common(1)
        if best:
            return f"{best[0][0]}（{t[:10]}…）"
    return t


def _content_head(doc_id, n=90):
    """取正文中前若干条有信息量的行（跳过样板与页码）。"""
    raw = retrieval.doc_path(doc_id).read_text(encoding="utf-8")[:2500]
    out = []
    for ln in raw.split("\n"):
        s = ln.strip()
        if len(s) < 6 or s.startswith("[P") or BOILER.search(s):
            continue
        out.append(s)
        if sum(len(x) for x in out) >= n:
            break
    return re.sub(r"\s+", " ", " ".join(out))[:n]


def select_docs(q, qid=None, model=DEFAULT_MODEL, k_coarse=12, max_docs=4):
    """返回该题应阅读的 doc_ids。"""
    qid = qid or q["qid"]
    cands = coarse_candidates(q, k=k_coarse)
    if not cands:
        return []
    if len(cands) <= 2:
        return cands
    meta_all = retrieval.docs_meta()
    head_n = 55 if __import__("os").environ.get("AFAC_SLIM4") == "1" else 130
    cards = []
    for d in cands:
        # csrc网页用 meta 摘要（含当事人/文号，标题雷同时唯一有区分度）；其余用正文开头
        summary = meta_all[d].get("summary")
        head = (summary or _content_head(d))[:head_n]
        cards.append(f"[{d}] {_doc_card(d)} | {head}")
    opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    ex = json.dumps(cands[:2], ensure_ascii=False)
    if q["domain"] == "regulatory":
        max_docs = max(max_docs, 5)
        extra_hint = "相近规章（如治理准则/股东会规则/章程指引）拿不准哪部适用时，都选上。"
    elif q["domain"] == "research":
        max_docs = max(max_docs, 5)
        extra_hint = ("题目或选项涉及几个行业/公司，就为每个行业/公司选一份对应研报，"
                      "不得合并省略。")
    else:
        extra_hint = ""
    prompt = (
        "题目:\n" + q["question"] + "\n选项:\n" + opts +
        "\n\n候选文档列表(方括号内为文档ID):\n" + "\n".join(cards) +
        f"\n\n请判断回答此题必须阅读哪些文档（题目/选项涉及几个主体、产品或法规就选几份，"
        f"通常2-4份，最多{max_docs}份；比较类题至少2份）。{extra_hint}"
        f"只输出文档ID的 JSON 数组，ID必须与方括号内完全一致，如 {ex}。")
    content, _r, _u = chat([{"role": "user", "content": prompt}],
                           qid=qid, model=model, thinking=False,
                           max_tokens=160 if head_n == 80 else 250,
                           tag="docsel")
    m = SEL_RE.search(content)
    picked = []
    if m:
        try:
            arr = json.loads(m.group(0))
            cset = set(cands)
            for x in arr:
                x = str(x)
                if x in cset:
                    picked.append(x)
                else:  # 端匹配仅在唯一时接受
                    ends = [d for d in cands if d.endswith(x)]
                    if len(ends) == 1:
                        picked.append(ends[0])
            picked = list(dict.fromkeys(picked))[:max_docs]
        except (json.JSONDecodeError, TypeError):
            pass
    return _finalize_picks(q, picked, cands, max_docs)


def _finalize_picks(q, picked, cands, max_docs):
    """选择后的零token兜底：材料类别补齐/csrc附件耦合/保险别名覆盖/
    BM25 top1并入/比较题≥2份。单题与批量共用，保证兜底语义一致。"""
    if not picked:
        picked = cands[:2]
    # 材料类别补齐：题目要求"结合A与B两类材料"时，每类至少一份
    meta = retrieval.docs_meta()
    qtext = q["question"] + " ".join(q["options"].values())
    if q["domain"] == "regulatory" and re.search(r"处罚|案例|决定书|违规查处", qtext):
        has_case = any(meta[d].get("column") == "行政处罚" for d in picked)
        if not has_case:
            case = next((d for d in cands
                         if meta[d].get("column") == "行政处罚"), None)
            if case:
                picked.append(case)
    for d in list(picked):
        if re.fullmatch(r"csrc_\d{4}", d):
            for att_i in (1, 2):
                att = f"{d}_att{att_i}"
                if att in meta and att not in picked:
                    picked.append(att)
    # 保险域：选项点名产品必须全覆盖（ins_b_012类伤：题问4产品只选3份文档）
    if q["domain"] == "insurance":
        try:
            tit = json.loads((ROOT / "work" / "processed_data" /
                              "insurance_titles.json").read_text())
            for d, info in tit.items():
                if d in picked or d not in set(cands):
                    continue
                if any(a in qtext for a in info.get("alias", [])):
                    picked.append(d)
        except FileNotFoundError:
            pass
    # 词法兜底：文档级BM25 top-1 未被选中则并入
    # （fc_b_013/016类伤：正确文档text13候选卡全是样板句，Qwen选卡失手）
    if q["domain"] != "regulatory" and len(cands) > 2:
        idx = domain_doc_index(q["domain"])
        top = idx.search(q["question"] + " " + " ".join(q["options"].values()), k=1)
        if top and top[0][0]["doc_id"] not in picked:
            picked.append(top[0][0]["doc_id"])
    # 研报域逐选项文档兜底：选项点名行业的报告必须在场
    # （res_b_008类伤：题干枚举4行业但寿险报告text15从未被选中）
    if q["domain"] == "research":
        idx = domain_doc_index("research")
        extra_n = 0
        for v in q["options"].values():
            if extra_n >= 2:
                break
            top = idx.search(v, k=1)
            if top and top[0][0]["doc_id"] not in picked:
                picked.append(top[0][0]["doc_id"])
                extra_n += 1
    # 比较/结合类题至少2份文档
    if len(picked) == 1:
        nxt = next((c for c in cands if c not in picked), None)
        if nxt:
            picked.append(nxt)
    return picked[:max_docs + 2]


BATCH_OBJ_RE = re.compile(r"\{.*\}", re.S)
BATCH_ENTRY_RE = re.compile(r'"([^"]+)"\s*:\s*\[([^\]]*)\]')


def select_docs_batch(qs, model=DEFAULT_MODEL, k_coarse=12):
    """同域 N 题共享候选卡，一次调用返回 {qid: [doc_ids]}。

    候选卡只发一次；逐题兜底（_finalize_picks）与单题版完全一致。
    某题条目缺失/解析失败 → 该题回退单题 select_docs（不劣于现状）。
    token 均摊到批内各 qid（残差留在 _docsel_{domain}，总量诚实）。
    """
    if not qs:
        return {}
    if len(qs) == 1:
        return {qs[0]["qid"]: select_docs(qs[0], model=model)}
    domain = qs[0]["domain"]
    per_cands = {q["qid"]: coarse_candidates(q, k=k_coarse) for q in qs}
    if domain == "regulatory":
        shared = list(dict.fromkeys(
            d for q in qs for d in per_cands[q["qid"]]))
        max_docs = 5
        extra_hint = ("每题末尾的'词法候选'是初筛提示，一般应从中选择；"
                      "相近规章（如治理准则/股东会规则/章程指引）拿不准哪部适用时，都选上。")
    else:
        shared = [d for d, m in retrieval.docs_meta().items()
                  if m["domain"] == domain]
        if domain == "research":
            max_docs = 5
            extra_hint = ("题目或选项涉及几个行业/公司，就为每个行业/公司选一份对应研报，"
                          "不得合并省略。")
        else:
            max_docs = 4
            extra_hint = ""
    meta_all = retrieval.docs_meta()
    head_n = 55 if __import__("os").environ.get("AFAC_SLIM4") == "1" else 130
    cards = []
    for d in shared:
        summary = meta_all[d].get("summary")
        head = (summary or _content_head(d))[:head_n]
        cards.append(f"[{d}] {_doc_card(d)} | {head}")
    qblocks = []
    for q in qs:
        opts = "".join(f"\n{k}. {v}" for k, v in q["options"].items())
        hint = ""
        if domain == "regulatory":
            hint = "\n词法候选: " + ",".join(per_cands[q["qid"]])
        qblocks.append(f"【{q['qid']}】{q['question']}{opts}{hint}")
    ex = json.dumps({qs[0]["qid"]: shared[:2], qs[1]["qid"]: shared[1:3]},
                    ensure_ascii=False)
    prompt = (
        "候选文档列表(方括号内为文档ID):\n" + "\n".join(cards) +
        f"\n\n以下{len(qs)}道题都从上面同一批候选文档中选阅读材料，"
        "请逐题独立判断该题必须阅读哪些文档，题与题互不影响"
        f"（题目/选项涉及几个主体、产品或法规就选几份，通常2-4份，"
        f"最多{max_docs}份；比较类题至少2份）。{extra_hint}\n\n" +
        "\n\n".join(qblocks) +
        "\n\n只输出一个JSON对象：键为题目ID(【】内)，值为该题文档ID数组，"
        f"文档ID必须与方括号内完全一致，不得输出其他内容。示例: {ex}")
    content, _r, usage = chat([{"role": "user", "content": prompt}],
                              qid=f"_docsel_{domain}", model=model,
                              thinking=False,
                              max_tokens=80 * len(qs) + 200, tag="docsel")
    p = usage.get("prompt_tokens", 0) // len(qs)
    c = usage.get("completion_tokens", 0) // len(qs)
    with LEDGER._lock:
        for q in qs:
            slot = LEDGER.per_qid.setdefault(q["qid"], [0, 0])
            slot[0] += p
            slot[1] += c
        b = LEDGER.per_qid.get(f"_docsel_{domain}")
        if b:
            b[0] = max(0, b[0] - p * len(qs))
            b[1] = max(0, b[1] - c * len(qs))
    sel = {}
    m = BATCH_OBJ_RE.search(content or "")
    obj = None
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    entries = (obj.items() if isinstance(obj, dict) else
               [(k, re.findall(r'"([^"]+)"', inner))
                for k, inner in BATCH_ENTRY_RE.findall(content or "")])
    sset = set(shared)
    for qid, arr in entries:
        if not isinstance(arr, list):
            continue
        picked = []
        for x in arr:
            x = str(x)
            if x in sset:
                picked.append(x)
            else:  # 端匹配仅在唯一时接受
                ends = [d for d in shared if d.endswith(x)]
                if len(ends) == 1:
                    picked.append(ends[0])
        sel[str(qid)] = list(dict.fromkeys(picked))[:max_docs]
    out = {}
    for q in qs:
        picked = sel.get(q["qid"], [])
        if not picked:
            out[q["qid"]] = select_docs(q, model=model)
            continue
        # 非reg域兜底候选用全域列表（top-9预筛会把别名/top1兜底需要的文档挡在外面）
        cands = per_cands[q["qid"]] if domain == "regulatory" else shared
        out[q["qid"]] = _finalize_picks(q, picked, cands, max_docs)
    return out
