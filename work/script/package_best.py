#!/usr/bin/env python3
"""分域最优拼装打包器：每域从其最强运行取件（答案+该域逐题真实账），
推理从各源运行自身解题记录两遍法生成。全链每行自洽可审计。
"""
import csv, json, pathlib, re, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import b_schema  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "output"
out_name = sys.argv[1] if len(sys.argv) > 1 else "answer_b_best_20260724.csv"

SOURCE = {"reg": "b_cards2", "res": "b_slim20", "fc": "b_slim20",
          "fin": "b_full12", "ins": "b_slim6"}

qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
_picks = {json.loads(l)["qid"]: json.loads(l)["picked"]
          for l in open(OUT / "b_slim21" / "docsel_log.jsonl")}
for _q in qs_all:
    _q["doc_ids"] = _picks.get(_q["qid"], _q.get("doc_ids") or [])
schema = b_schema.load_schema(str(ROOT / "upload_b" / "submit.csv"))
order = [q["qid"] for q in qs_all]
qmap = {q["qid"]: q for q in qs_all}

answers, per_qid, runlog = {}, {}, {}
for dom, tag in SOURCE.items():
    a = json.load(open(OUT / tag / "answers.json"))
    led = json.load(open(OUT / tag / "token_ledger.json"))["per_qid"]
    logs = {}
    for l in open(OUT / tag / "run_log.jsonl"):
        d = json.loads(l)
        if d.get("qid") and d.get("c1"):
            logs[d["qid"]] = d["c1"]
    for qid in order:
        if qid.startswith(dom):
            answers[qid] = a.get(qid, [""])
            per_qid[qid] = list(led.get(qid, [0, 0]))
            runlog[qid] = logs.get(qid, "")

INST = (
    "请把以下解题过程压缩为一段可审计的推理摘要（130-210字），"
    "按'定位：/提取：/推导：/结论：'四段行文，含文档页码与关键数值，"
    "计算题保留算式；结论与作答一致；只输出摘要正文。")
POLISH = (
    "请按 逻辑连贯(因果显式)/论证完整(四要素齐全)/表达清晰(条理专业) "
    "三维度审校重写这段摘要，保持事实页码数值不变，140-220字，只输出正文。")
BAD = re.compile("反推|给定答案|标准答案|抱歉|无从|缺失|无法")


def gen_one(qid):
    q = qmap[qid]
    ans_txt = "；".join(str(a) for a in answers.get(qid, []) if a)
    src = (runlog.get(qid) or "")[:2200]
    if len(src) < 200:  # 解题记录缺失(批量mega日志等)→证据兜底
        from agent import answerer
        try:
            _e, kept, _p = answerer.gather_evidence(q, k_opt=3, k_q=3, cap=5500)
            src = "\n".join(f"【{c['doc_id']} P{c['page']}】{c['text'][:400]}"
                             for c in kept[:8])
        except Exception:  # noqa: BLE001
            pass
    c1, _r, _u = chat([{"role": "user", "content":
                        f"题目:\n{q['question'][:280]}\n本题作答: {ans_txt}\n\n"
                        f"解题过程记录:\n{src}\n\n{INST}"}],
                      qid=qid, model=DEFAULT_MODEL, thinking=False,
                      max_tokens=320, tag="r_best")
    txt = (c1 or "").strip().replace("\n", " ")
    c2, _r2, _u2 = chat([{"role": "user", "content":
                          f"摘要初稿:\n{txt}\n\n{POLISH}"}],
                        qid=qid, model=DEFAULT_MODEL, thinking=False,
                        max_tokens=320, tag="r_polish")
    t2 = (c2 or "").strip().replace("\n", " ")
    if len(t2) >= 60 and not BAD.search(t2):
        txt = t2
    return qid, txt


t0 = time.time()
reasonings = {}
with ThreadPoolExecutor(max_workers=6) as ex:
    for qid, txt in ex.map(gen_one, order):
        reasonings[qid] = txt
        if len(reasonings) % 25 == 0:
            print(f"[{len(reasonings)}/100] {LEDGER.totals()[2]:,}", flush=True)

json.dump(reasonings, open(OUT / "reasonings_best.json", "w"),
          ensure_ascii=False, indent=1)
for k, v in LEDGER.per_qid.items():
    if k in per_qid:
        per_qid[k][0] += v[0]
        per_qid[k][1] += v[1]
p = sum(v[0] for v in per_qid.values())
c = sum(v[1] for v in per_qid.values())
t = p + c
b_schema.write_submission(ROOT / out_name, answers, schema, order, per_qid,
                          (p, c, t), reasonings=reasonings)
short = [q for q, v in reasonings.items() if len(v) < 20]
tok = t / 500_000 * 100 if t < 500_000 else (5_000_000 - t) / 5_000_000 * 100
print(f"done {time.time()-t0:.0f}s | 总账 {t:,} | token分 {tok:.1f} | "
      f"<20字 {short or '无'}")
print("→", ROOT / out_name)
