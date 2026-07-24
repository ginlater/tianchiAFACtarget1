#!/usr/bin/env python3
"""routerM 记忆驻留架构：批量压缩(读共享) + 逐题作答(注意力独享)。

第一性原理: 单发贵在同一批卡被每题重复阅读; 批量便宜但8题共享输出注意力。
拆开: StageA 一次调用为一组题各产记忆包(压缩÷N); StageB 每题独享小调用+思考。
用法: router_m.py <qids_file> <tag>   (calc题走既有answer_calc管线)
"""
import json, pathlib, re, sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema, calc, doc_select  # noqa: E402
from agent.qwen_client import chat, LEDGER  # noqa: E402

WORK = pathlib.Path(__file__).resolve().parents[1]
QIDS = open(sys.argv[1]).read().strip().split(",") if len(sys.argv) > 1 else []
TAG = sys.argv[2] if len(sys.argv) > 2 else "b_routerM"

qs_all = {q["qid"]: q for q in b_schema.load_questions(
    str(WORK.parent / "upload_b" / "question_b"))}
schema = b_schema.load_schema(str(WORK.parent / "upload_b" / "submit.csv"))
qids = [q for q in QIDS if q in qs_all]

outdir = WORK / "output" / TAG
outdir.mkdir(exist_ok=True)
log = open(outdir / "run_log.jsonl", "w")

# docsel(盲测选文档, 逐题计账)
picks = {}
dlog = open(outdir / "docsel_log.jsonl", "w")
for qid in qids:
    q = qs_all[qid]
    picked = doc_select.select_docs(q)
    picks[qid] = picked
    dlog.write(json.dumps({"qid": qid, "picked": picked},
                          ensure_ascii=False) + "\n")
dlog.flush()

answers = {}
ANS = re.compile(r"答案[:：]\s*([A-D]+)")

# 按域分组; 组内按共享文档打包(每包≤6题)
groups = {}
for qid in qids:
    q = qs_all[qid]
    q["doc_ids"] = picks[qid]
    if q["answer_format"] == "calc":
        groups.setdefault("_calc", []).append(qid)
    else:
        groups.setdefault(q["domain"], []).append(qid)

for dom, qlist in groups.items():
    if dom == "_calc":
        continue
    for i in range(0, len(qlist), 6):
        pack_q = qlist[i:i + 6]
        # StageA: 组证据 = 并集文档的卡/矿 + 每题定向检索top块
        blocks, seen_docs = [], set()
        for qid in pack_q:
            q = qs_all[qid]
            ff = answerer.fin_facts_block(q)
            df = answerer.domain_facts_block(q)
            ab = answerer.align_block(q)
            for b in (ff, df, ab):
                if b:
                    blocks.append(b[:3000])
            ev, kept, _p = answerer.gather_evidence(
                q, k_opt=2, k_q=2, cap=3200)
            blocks.append(f"◎{qid}相关原文:\n" + ev)
        qtexts = []
        for qid in pack_q:
            q = qs_all[qid]
            opts = "\n".join(f"{k}. {v}"
                             for k, v in (q.get("options") or {}).items())
            qtexts.append(f"【{qid}】{q['question']}\n{opts}")
        ca, _t, _u = chat([{"role": "user", "content":
            "\n\n".join(blocks)[:60000] + "\n\n===题目===\n"
            + "\n\n".join(qtexts) + "\n\n"
            "请为每道题产出【记忆包】(每题≤300字): 逐选项列出证据中的相关数值/"
            "条款原句要点+页码; 证据未见的选项写'未见对应表述'。不判断对错。"
            "格式: 【qid】开头, 逐题输出。"}],
            qid=pack_q[0], model="qwen3.6-plus", thinking=False,
            max_tokens=2600, tag="mA")
        packs = {}
        for m in re.finditer(r"【(\w+)】([^【]+)", ca or ""):
            packs[m.group(1)] = m.group(2).strip()
        # StageB: 逐题独享作答
        for qid in pack_q:
            q = qs_all[qid]
            opts = "\n".join(f"{k}. {v}"
                             for k, v in (q.get("options") or {}).items())
            pk = packs.get(qid, "")
            fmt = q["answer_format"]
            cb, _t, _u = chat([{"role": "user", "content":
                f"记忆包:\n{pk}\n\n题目({answerer.FMT_NAME.get(fmt, fmt)}):"
                f"{q['question']}\n{opts}\n\n严格依据记忆包逐项判断"
                "(证据不足以支持的选项不选)。最后一行输出 答案:<字母>"}],
                qid=qid, model="qwen3.6-plus", thinking=True,
                thinking_budget=900, max_tokens=1400, tag="mB")
            m = ANS.search(cb or "")
            ans = "".join(sorted(set(m.group(1)))) if m else ""
            if fmt == "mcq" and len(ans) > 1:
                ans = ans[0]
            answers[qid] = [ans]
            log.write(json.dumps({"qid": qid, "c1": cb, "final": ans,
                                  "pack": pk[:400]},
                                 ensure_ascii=False) + "\n")
            log.flush()
            print(f"{qid} -> {ans}", flush=True)

# calc题: 既有管线(三矿+跨代二审已由env控制)
for qid in groups.get("_calc", []):
    q = qs_all[qid]
    kinds = b_schema.effective_kinds(q, schema.get(qid, ["number"]))
    final = calc.answer_calc(q, kinds, log=log, blind_mode=True)
    answers[qid] = b_schema.split_answer(final or "", kinds)
    print(f"{qid} -> {answers[qid]}", flush=True)

json.dump(answers, open(outdir / "answers.json", "w"),
          ensure_ascii=False, indent=1)
LEDGER.dump(outdir / "token_ledger.json")
p, c, t = LEDGER.totals()
print(f"routerM done: {len(answers)}题 tokens {t:,}")
