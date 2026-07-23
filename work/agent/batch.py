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
    # 逐题配额检索后并集（伪题合并会让所有选项查询共用第1题前缀，
    # 后位题证据被挤出：批检索覆盖0.16 vs 单题0.41——slim10八题批稀释类伤）
    per_cap = max(1500, cap // len(qs))
    best, prot_all, per_kept = {}, set(), {}
    for q in qs:
        qq = dict(q, doc_ids=q0["doc_ids"])
        _ev_i, kept_i, prot_i = gather_evidence(qq, k_opt=2, k_q=3,
                                                cap=per_cap)
        per_kept[q["qid"]] = [c["id"] for c in kept_i]
        for c in kept_i:
            best[c["id"]] = c
        prot_all |= prot_i
    # 并集层总预算闸门（保护块优先装填）：防逐题保护豁免叠加爆预算(slim12回归)
    ordered = sorted(best.values(), key=lambda c: c["id"] not in prot_all)
    picked_u, total = [], 0
    for c in ordered:
        L = len(c["text"]) + 20
        if c["id"] not in prot_all and total + L > int(cap * 1.15):
            continue
        if total + L > int(cap * 1.6):  # 硬顶：保护块也不得无限叠加
            continue
        total += L
        picked_u.append(c)
    kept = sorted(picked_u,
                  key=lambda c: (c["doc_id"], c["page"] or 0,
                                 int(c["id"].split("#c")[1])))
    parts = []
    for c in kept:
        tag = f"{c['doc_id']} P{c['page']}" if c["page"] else c["id"]
        parts.append(f"【{tag}】{c['text']}")
    blocks.append("原文片段证据:\n" + "\n\n".join(parts))
    # 覆盖率制导：某题自己的证据块被并集闸门挤掉过半 → 标记定向单答
    # （法医量化: 批内覆盖<0.15正确率仅48%, ≥0.4达86%）
    uids = {c["id"] for c in kept}
    cov = {qid: (sum(1 for i in ids if i in uids) / len(ids)) if ids else 1.0
           for qid, ids in per_kept.items()}
    low_cov = [qid for qid, v in sorted(cov.items(), key=lambda x: x[1])
               if v < 0.5][:3]
    return "\n\n".join(blocks), [c["id"] for c in kept], low_cov


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
    ev, ev_ids, low_cov = _batch_evidence(qs, model=model)
    qtexts = "\n\n".join(f"[第{i+1}题 {q['qid']}]\n{_q_text(q)}"
                         for i, q in enumerate(qs))
    base = ev + "\n\n" + qtexts
    inst = BATCH_INST.format(n=len(qs))
    share = [q["qid"] for q in qs]

    import os as _os
    slim4 = _os.environ.get("AFAC_SLIM4") == "1"

    def _chat(prompt, tag, mdl, budget):
        if slim4:
            budget = min(budget, 1300)
        c, _r, usage = chat([{"role": "user", "content": prompt}],
                            qid="_batch", model=mdl, thinking=_think(qs[0]),
                            thinking_budget=budget,
                            max_tokens=(1400 * len(qs) + 1400) if slim4
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
    # 批内多数决（AFAC_B1_VOTES=N, 域白名单AFAC_B1_VOTE_DOMS）：
    # 只对摇摆重灾域花钱，同证据独立采样逐选项投票
    n_b1 = int(_os.environ.get("AFAC_B1_VOTES", "1"))
    vote_doms = _os.environ.get("AFAC_B1_VOTE_DOMS",
                                "financial_reports").split(",")
    if n_b1 > 1 and qs[0]["domain"] in vote_doms:
        pools = {q["qid"]: [a1.get(q["qid"])] for q in qs}
        for _i in range(n_b1 - 1):
            cx = _chat(base + "\n\n" + inst, "b1", model, 1600)
            ax = _parse_batch(cx, qs)
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
            final = x1 or x2
            if not final:
                # 解析失败禁止默认A（实测默认A 8/8全错）：精简证据单题追答
                # （全量批证据追答实测78k/跑，截到5000字后成本降2/3信息量足够）
                c4, _r, _u = chat(
                    [{"role": "user", "content": ev[:5000] + "\n\n" + _q_text(q) +
                      "\n\n" + JUDGE_STD + "\n只输出最后一行：答案: <字母>"}],
                    qid=qid, model=model, thinking=False,
                    max_tokens=400, tag="b4")
                final = parse_answer(c4, fmt) or "A"
            finals[qid] = final
        if log is not None:
            log.write(json.dumps({
                "qid": qid, "final": finals[qid], "r1": x1, "r2": x2,
                "batch": share, "c1": (c1 or "")[:1500],
                "evidence_ids": ev_ids[:30]}, ensure_ascii=False) + "\n")
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
                a_solo, _info = answer_question(q, model, log,
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
    return finals
