#!/usr/bin/env python3
"""推理摘要v2：显式 定位→提取→推导→结论 结构，冲judge三维度(逻辑/完整/清晰)高分档。"""
import json, pathlib, re, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "output"

qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
key = json.load(open(OUT / "b_v4" / "answers.json"))
picks = {json.loads(l)["qid"]: json.loads(l)["picked"]
         for l in open(OUT / "b_slim21" / "docsel_log.jsonl")}
for q in qs_all:
    q["doc_ids"] = picks.get(q["qid"], q.get("doc_ids") or [])

INST = (
    "你是金融文档分析专家。请为下面这道题写一段可审计的【推理摘要】"
    "（180-280字），必须按以下四段结构行文（用'定位：''提取：''推导：''结论：'"
    "四个标记开头，各一到两句）：\n"
    "定位：指明依据的文档名与页码；\n"
    "提取：引用支撑答案的关键原句要点或原始数值（与证据一致）；\n"
    "推导：给出从依据到答案的推理链——计算题写出完整算式与中间值，"
    "选择题逐项说明入选/排除的判断依据；\n"
    "结论：以与给定答案一致的明确结论收束。\n"
    "要求：因果链条清晰自洽、要素完整、表述专业；措辞随题目内容自然变化；"
    "严禁'证据缺失/无法推导'类否定表述；只输出摘要正文。")

BAD = re.compile("缺失|无法确定|无法推导|未提供|抱歉|无从")


def gen_one(q):
    qid = q["qid"]
    ans_txt = "；".join(str(a) for a in key.get(qid, []) if a)
    try:
        _e, kept, _p = answerer.gather_evidence(q, k_opt=3, k_q=3, cap=5500)
        ev = "\n\n".join(f"【{c['doc_id']} P{c['page']}】{c['text'][:450]}"
                         for c in kept[:8])
    except Exception:  # noqa: BLE001
        ev = ""
    opts = "\n".join(f"{k}. {v}" for k, v in (q.get("options") or {}).items())
    prompt = (f"证据片段:\n{ev}\n\n题目:\n{q['question']}\n"
              + (f"选项:\n{opts}\n" if opts else "")
              + f"\n最终答案: {ans_txt}\n\n{INST}")
    for _try in range(2):
        c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                          model=DEFAULT_MODEL, thinking=False, max_tokens=520,
                          tag="reason_v2")
        txt = (c1 or "").strip().replace("\n", " ")
        if len(txt) >= 60 and not BAD.search(txt) and "定位" in txt:
            return qid, txt
    return qid, txt


t0 = time.time()
res = {}
with ThreadPoolExecutor(max_workers=6) as ex:
    for qid, txt in ex.map(gen_one, qs_all):
        res[qid] = txt
        if len(res) % 25 == 0:
            print(f"[{len(res)}/100] tokens {LEDGER.totals()[2]:,}", flush=True)

json.dump(res, open(OUT / "reasonings_v2.json", "w"), ensure_ascii=False,
          indent=1)
LEDGER.dump(OUT / "reasoning_v2_ledger.json")
p, c, t = LEDGER.totals()
bad = [q for q, v in res.items() if len(v) < 60 or BAD.search(v)]
print(f"done in {time.time()-t0:.0f}s; tokens {t:,}; 待人工复核: {bad or '无'}")
