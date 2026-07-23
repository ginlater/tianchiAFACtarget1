#!/usr/bin/env python3
"""极限预算打包器：5题合批生成推理摘要，总账压进500k分界线内。
用法: .venv/bin/python script/package_run_lean.py <run_tag> <out_name.csv>
"""
import csv, json, pathlib, re, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import b_schema  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "output"
tag, out_name = sys.argv[1], sys.argv[2]

answers = json.load(open(OUT / tag / "answers.json"))
base_led = json.load(open(OUT / tag / "token_ledger.json"))
qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
schema = b_schema.load_schema(str(ROOT / "upload_b" / "submit.csv"))
order = [q["qid"] for q in qs_all]
qmap = {q["qid"]: q for q in qs_all}

runlog = {}
for l in open(OUT / tag / "run_log.jsonl"):
    d = json.loads(l)
    if d.get("qid") and d.get("c1"):
        runlog[d["qid"]] = d.get("c1", "")

INST = ("为以上每道题各写一段推理摘要（100-160字），按'定位：…提取：…推导：…"
        "结论：…'四段连排；定位含文档页码，提取含关键数值，计算题推导写算式，"
        "结论与该题作答一致。输出格式：每题一行，以【qid】开头。")

BLOCK_RE = re.compile(r"【([a-z_0-9]+)】\s*([^【]+)", re.S)

t0 = time.time()
reasonings = {}
for i in range(0, len(order), 5):
    grp = order[i:i + 5]
    parts = []
    for qid in grp:
        q = qmap[qid]
        ans_txt = "；".join(str(a) for a in answers.get(qid, []) if a)
        src = (runlog.get(qid) or "")[:700]
        parts.append(f"【{qid}】题目:{q['question'][:120]}\n作答:{ans_txt}\n"
                     f"解题记录:{src}")
    prompt = "\n\n".join(parts) + "\n\n" + INST
    c1, _r, usage = chat([{"role": "user", "content": prompt}], qid="_rbatch",
                         model=DEFAULT_MODEL, thinking=False,
                         max_tokens=280 * len(grp), tag="reason_lean")
    got = {m.group(1): m.group(2).strip().replace("\n", " ")
           for m in BLOCK_RE.finditer(c1 or "")}
    # token 均摊
    p = usage.get("prompt_tokens", 0) // len(grp)
    c = usage.get("completion_tokens", 0) // len(grp)
    with LEDGER._lock:
        for qid in grp:
            slot = LEDGER.per_qid.setdefault(qid, [0, 0])
            slot[0] += p
            slot[1] += c
        b = LEDGER.per_qid.get("_rbatch")
        if b:
            b[0] = max(0, b[0] - p * len(grp))
            b[1] = max(0, b[1] - c * len(grp))
    for qid in grp:
        txt = got.get(qid, "")
        if len(txt) < 20:  # 单题补救
            q = qmap[qid]
            ans_txt = "；".join(str(a) for a in answers.get(qid, []) if a)
            c2, _r2, _u2 = chat(
                [{"role": "user", "content":
                  f"题目:{q['question'][:150]}\n作答:{ans_txt}\n"
                  f"解题记录:{(runlog.get(qid) or '')[:800]}\n\n"
                  "写一段120字左右推理摘要，含定位/提取/推导/结论与页码数值，"
                  "结论与作答一致，只输出正文。"}],
                qid=qid, model=DEFAULT_MODEL, thinking=False, max_tokens=260,
                tag="reason_fix")
            txt = (c2 or "").strip().replace("\n", " ")
        reasonings[qid] = txt
    print(f"[{len(reasonings)}/100] {LEDGER.totals()[2]:,}", flush=True)

json.dump(reasonings, open(OUT / f"reasonings_lean_{tag}.json", "w"),
          ensure_ascii=False, indent=1)
per_qid = {}
for src in (base_led["per_qid"], LEDGER.per_qid):
    for k, v in src.items():
        slot = per_qid.setdefault(k, [0, 0])
        slot[0] += v[0]
        slot[1] += v[1]
p = sum(v[0] for v in per_qid.values())
c = sum(v[1] for v in per_qid.values())
t = p + c
b_schema.write_submission(ROOT / out_name, answers, schema, order, per_qid,
                          (p, c, t), reasonings=reasonings)
short = [q for q, v in reasonings.items() if len(v) < 20]
tok_score = t / 500_000 * 100 if t < 500_000 else (5_000_000 - t) / 5_000_000 * 100
print(f"done {time.time()-t0:.0f}s | 总账 {t:,} | token分 {tok_score:.1f} | "
      f"<20字 {short or '无'}")
print("→", ROOT / out_name)
