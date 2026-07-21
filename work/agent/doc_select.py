"""B榜文档盲检：文档级词法BM25粗召回 + Qwen从候选卡片精选。

合规：粗召回为纯词法统计；语义选择由 Qwen 完成（token 计入台账）。
"""
import json, pathlib, re

from . import retrieval
from .qwen_client import chat, DEFAULT_MODEL

ROOT = pathlib.Path(__file__).resolve().parents[2]

_DOC_BM25 = {}   # domain -> BM25 over doc-level pseudo-chunks
INDEX_HEAD = 30000


def _doc_card(doc_id):
    m = retrieval.docs_meta()[doc_id]
    title = re.sub(r"^标题：", "", m["title"])
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


def coarse_candidates(q, k=25):
    """regulatory(513篇)用BM25并集召回；其余领域文档少，直接全量进候选。"""
    if q["domain"] != "regulatory":
        return [d for d, m in retrieval.docs_meta().items()
                if m["domain"] == q["domain"]]
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

# 样板噪声：研报免责声明/分析师信息、页眉页码
BOILER = re.compile(
    r"请务必阅读|免责条款|证券研究报告|执业证书编号|SAC|分析师|联系人|"
    r"^\d+$|^P\d+$|研究助理|@|电话|邮箱")


def _content_head(doc_id, n=200):
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
    cards = []
    for d in cands:
        # csrc网页用 meta 摘要（含当事人/文号，标题雷同时唯一有区分度）；其余用正文开头
        summary = meta_all[d].get("summary")
        head = summary or _content_head(d)
        cards.append(f"[{d}] {_doc_card(d)} | {head}")
    opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    ex = json.dumps(cands[:2], ensure_ascii=False)
    if q["domain"] == "regulatory":
        max_docs = max(max_docs, 5)
        extra_hint = "相近规章（如治理准则/股东会规则/章程指引）拿不准哪部适用时，都选上。"
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
                           max_tokens=250, tag="docsel")
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
    # 比较/结合类题至少2份文档
    if len(picked) == 1:
        nxt = next((c for c in cands if c not in picked), None)
        if nxt:
            picked.append(nxt)
    return picked[:max_docs + 2]
