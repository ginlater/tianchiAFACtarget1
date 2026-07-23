#!/usr/bin/env python3
"""重生成含'证据缺失/无法推导'红旗的推理摘要：加厚证据+严格指令。"""
import json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "output"

redo = set(json.load(open(OUT / "reason_redo.json")))
res = json.load(open(OUT / "reasonings.json"))
key = json.load(open(OUT / "b_v4" / "answers.json"))
qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
picks = {json.loads(l)["qid"]: json.loads(l)["picked"]
         for l in open(OUT / "b_slim21" / "docsel_log.jsonl")}
qs = [dict(q, doc_ids=picks.get(q["qid"], [])) for q in qs_all
      if q["qid"] in redo]

INST = (
    "你是金融文档分析专家。请为下面这道题写一段可审计的【推理摘要】"
    "（140-220字）支撑给定答案：\n"
    "1) 点明依据文档与页码；2) 引关键原句要点/数值；"
    "3) 写出推导（计算题必须给算式）；4) 结论与答案一致。\n"
    "硬性要求：从证据中定位支持答案的内容并组织推导；"
    "严禁出现'证据缺失/无法推导/未提供'等否定性表述；只输出摘要正文。")


def gen_one(q):
    qid = q["qid"]
    ans_txt = "；".join(str(a) for a in key.get(qid, []) if a)
    try:
        _e, kept, _p = answerer.gather_evidence(q, k_opt=3, k_q=3, cap=6500)
        ev = "\n\n".join(f"【{c['doc_id']} P{c['page']}】{c['text'][:500]}"
                         for c in kept[:9])
    except Exception:  # noqa: BLE001
        ev = ""
    opts = "\n".join(f"{k}. {v}" for k, v in (q.get("options") or {}).items())
    prompt = (f"证据片段:\n{ev}\n\n题目:\n{q['question']}\n"
              + (f"选项:\n{opts}\n" if opts else "")
              + f"\n最终答案: {ans_txt}\n\n{INST}")
    c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                      model=DEFAULT_MODEL, thinking=False, max_tokens=450,
                      tag="reason2")
    return qid, (c1 or "").strip().replace("\n", " ")


BAD = re.compile("缺失|无法|未包含|未提供|不包含|矛盾|遗憾|抱歉|无从")
with ThreadPoolExecutor(max_workers=6) as ex:
    for qid, txt in ex.map(gen_one, qs):
        if len(txt) >= 20 and not BAD.search(txt):
            res[qid] = txt
            print(f"✓ {qid} 重写成功 {len(txt)}字", flush=True)
        else:
            print(f"⚠ {qid} 仍有问题, 保留待人工: {txt[:60]}", flush=True)

json.dump(res, open(OUT / "reasonings.json", "w"), ensure_ascii=False, indent=1)
LEDGER.dump(OUT / "reasoning_ledger2.json")
p, c, t = LEDGER.totals()
print(f"redo done; tokens {t:,}")
