"""批量共享证据答题：同(域,文档集)的选择题合并作答，消灭证据重复计费。

设计要点：
- 仅选择题参与批量（计算题保持独立双样本）；批内最多3题防互扰
- 证据 = 记忆卡(并集文档) + 批内全部题目查询的并集检索（预算按题数扩容）
- r1/r2 批量作答（每题独立输出块），批内某题 r1≠r2 → 单题定向仲裁（复用单题逻辑）
- token 归集：调用量按批内题数均摊到各 qid
"""
import json, re

from . import retrieval
from .answerer import (CALC_DOMAINS, DIGEST_DOMAINS, JUDGE_STD, _doc_title,
                       _q_text, _use_digest, _vote_letters, build_digest,
                       gather_evidence, parse_answer, _think, VERIFY_MODEL)
from .qwen_client import chat, LEDGER, DEFAULT_MODEL


def group_questions(questions, max_batch=3):
    """按(域, 文档集)分组。返回 [ [q,...], ... ]，单题组即单题。
    瘦身档：域内松散合批（证据取文档并集），摊薄指令与证据开销。"""
    import os
    loose = os.environ.get("AFAC_SLIM4") == "1"
    groups = {}
    for q in questions:
        key = q["domain"] if loose else (q["domain"], frozenset(q["doc_ids"]))
        groups.setdefault(key, []).append(q)
    out = []
    if loose:
        max_batch = 4
        # 文档集相近的排在一起，减少并集膨胀
        for _key, qs in groups.items():
            qs.sort(key=lambda q: sorted(q["doc_ids"]))
    for _key, qs in groups.items():
        for i in range(0, len(qs), max_batch):
            out.append(qs[i:i + max_batch])
    return out


def _batch_evidence(qs, model=DEFAULT_MODEL):
    q0 = qs[0]
    domain = q0["domain"]
    # 松散合批下证据覆盖批内全部文档（并集，保持每题可答）
    docs = list(dict.fromkeys(d for q in qs for d in q["doc_ids"]))
    q0 = dict(q0, doc_ids=docs)
    blocks = []
    if _use_digest(domain):
        for d in q0["doc_ids"]:
            blocks.append(build_digest(d, domain, model=model))
        base_cap = 9500 if domain == "financial_contracts" else \
            8500 if domain == "financial_reports" else 6000
    else:
        blocks.append("涉及文档:\n" + "\n".join(
            f"- {d}: 《{_doc_title(d)}》" for d in q0["doc_ids"]))
        base_cap = 10000 if domain == "research" else 8500
        if __import__("os").environ.get("AFAC_SLIM4") == "1":
            # fc/fin大文档域无卡时证据帽必须给足（slim6教训：砍卡后漏选爆发）
            base_cap = {"research": 6000, "financial_contracts": 7500,
                        "financial_reports": 7500}.get(domain, 4800)
    # 预算按批内题数扩容40%/题（并集去重后实际占用低于线性）；瘦身档25%
    _slim4 = __import__("os").environ.get("AFAC_SLIM4") == "1"
    cap = int(base_cap * (1 + (0.25 if _slim4 else 0.4) * (len(qs) - 1)))
    cap += (1200 if _slim4 else 2000) * max(0, min(len(q0["doc_ids"]), 5) - 2)
    # 合成一个"联合题"喂给 gather_evidence：并集 options 驱动逐选项检索
    merged_opts = {}
    for i, q in enumerate(qs):
        for k, v in q["options"].items():
            merged_opts[f"{i}{k}"] = v
    pseudo = {"question": " ".join(q["question"][:60] for q in qs),
              "options": merged_opts, "doc_ids": q0["doc_ids"]}
    ev, kept, prot = gather_evidence(pseudo, k_opt=2, k_q=3, cap=cap)
    blocks.append("原文片段证据:\n" + ev)
    return "\n\n".join(blocks), [c["id"] for c in kept]


_BLOCK_RE = re.compile(r"【第(\d+)题[^】]*】")


def _parse_batch(content, qs):
    """按【第i题…】切块，逐题解析答案。返回 {qid: ans}"""
    out = {}
    pieces = _BLOCK_RE.split(content or "")
    # pieces: [前言, idx1, text1, idx2, text2, ...]
    for j in range(1, len(pieces) - 1, 2):
        try:
            i = int(pieces[j]) - 1
        except ValueError:
            continue
        if 0 <= i < len(qs):
            ans = parse_answer(pieces[j + 1], qs[i]["answer_format"])
            if ans:
                out[qs[i]["qid"]] = ans
    return out


BATCH_INST = (
    "以下 {n} 道题基于同一批文档，请逐题独立作答，题与题之间不得互相影响。\n"
    + JUDGE_STD + "\n"
    "输出格式（每题一个块，序号必须与题目序号一致）:\n"
    "【第1题 答案块】\n选择标准: <一句话>\n分析: <每选项一行，引证据页码>\n"
    "判断: A入选/不选 ...\n答案: <字母>\n"
    "【第2题 答案块】\n...（依此类推，每题都必须有'答案:'行）"
)


def answer_batch(qs, model=DEFAULT_MODEL, log=None):
    """批量作答一组同文档选择题。返回 {qid: final_answer}。"""
    ev, ev_ids = _batch_evidence(qs, model=model)
    qtexts = "\n\n".join(f"[第{i+1}题 {q['qid']}]\n{_q_text(q)}"
                         for i, q in enumerate(qs))
    base = ev + "\n\n" + qtexts
    inst = BATCH_INST.format(n=len(qs))
    share = [q["qid"] for q in qs]

    import os as _os
    slim4 = _os.environ.get("AFAC_SLIM4") == "1"

    def _chat(prompt, tag, mdl, budget):
        if slim4:
            budget = min(budget, 1800)
        c, _r, usage = chat([{"role": "user", "content": prompt}],
                            qid="_batch", model=mdl, thinking=_think(qs[0]),
                            thinking_budget=budget,
                            max_tokens=(1200 * len(qs) + 1200) if slim4
                            else 1500 * len(qs) + 1500,
                            tag=tag)
        # 均摊 token 到批内各题（_batch 槽位随后清零）
        p = usage.get("prompt_tokens", 0) // len(share)
        cc = usage.get("completion_tokens", 0) // len(share)
        with LEDGER._lock:
            for qid in share:
                slot = LEDGER.per_qid.setdefault(qid, [0, 0])
                slot[0] += p
                slot[1] += cc
            b = LEDGER.per_qid.get("_batch")
            if b:
                b[0] = max(0, b[0] - p * len(share))
                b[1] = max(0, b[1] - cc * len(share))
        return c

    c1 = _chat(base + "\n\n" + inst, "b1", model, 2600)
    a1 = _parse_batch(c1, qs)
    import os as _os
    if _os.environ.get("AFAC_SLIM") == "1":
        a2 = {}
    else:
        c2 = _chat(base + "\n\n" + inst +
                   "\n（这是独立复核轮，请从头独立判断）", "b2",
                   VERIFY_MODEL or model, 2200)
        a2 = _parse_batch(c2, qs)

    finals = {}
    for q in qs:
        qid, fmt = q["qid"], q["answer_format"]
        x1, x2 = a1.get(qid), a2.get(qid)
        if x1 and x2 and x1 != x2:
            # 单题定向仲裁（复用批证据，只带该题）
            disputed = [l for l in "ABCD" if (l in x1) != (l in x2)]
            dtxt = "\n".join(f"{l}. {q['options'][l]}" for l in disputed
                             if l in q["options"])
            adj = (ev + "\n\n" + _q_text(q) +
                   f"\n\n两次独立判断分歧选项:\n{dtxt}\n甲={x1} 乙={x2}\n"
                   "请仅核对分歧选项后给出完整最终答案。\n" + JUDGE_STD +
                   "\n输出:\n答案: <字母>")
            c3, _r, _u = chat([{"role": "user", "content": adj}], qid=qid,
                              model=VERIFY_MODEL or model, thinking=True,
                              thinking_budget=2600, max_tokens=2600, tag="b3")
            x3 = parse_answer(c3, fmt)
            finals[qid] = _vote_letters([x1, x2, x3], fmt) or x3 or x2
        else:
            finals[qid] = x1 or x2 or "A"
        if log is not None:
            log.write(json.dumps({
                "qid": qid, "final": finals[qid], "r1": x1, "r2": x2,
                "batch": share, "c1": (c1 or "")[:1500],
                "evidence_ids": ev_ids[:30]}, ensure_ascii=False) + "\n")
            log.flush()
    return finals
