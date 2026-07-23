#!/usr/bin/env python3
"""合规打包器：单一运行端到端 → 新规格式提交。

答案=该运行自己的输出；reasoning=压缩该运行自己的解题记录(c1)；
token=运行台账+推理生成台账逐题合并。全链自洽可审计。
用法: .venv/bin/python script/package_run.py <run_tag> <out_name.csv>
"""
import csv, json, pathlib, re, sys, time
from concurrent.futures import ThreadPoolExecutor

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

INST = (
    "请把以下解题过程压缩为一段可审计的推理摘要（120-200字），"
    "按'定位：/提取：/推导：/结论：'四段行文：定位=依据文档与页码；"
    "提取=关键原句要点或数值；推导=推理链（计算题保留算式，"
    "选择题逐项取舍要点）；结论=与作答一致的收束。"
    "保留页码与数字；只输出摘要正文。")
BAD = re.compile("反推|给定答案|标准答案|抱歉|无从")


def gen_one(qid):
    q = qmap[qid]
    ans_txt = "；".join(str(a) for a in answers.get(qid, []) if a)
    src = (runlog.get(qid) or "")[:2400]
    prompt = (f"题目:\n{q['question'][:300]}\n本题作答: {ans_txt}\n\n"
              f"解题过程记录:\n{src}\n\n{INST}")
    txt = ""
    for _try in range(2):
        c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                          model=DEFAULT_MODEL, thinking=False, max_tokens=330,
                          tag="reason_pkg")
        txt = (c1 or "").strip().replace("\n", " ")
        if len(txt) >= 60 and not BAD.search(txt):
            break
    return qid, txt


t0 = time.time()
reasonings = {}
with ThreadPoolExecutor(max_workers=6) as ex:
    for qid, txt in ex.map(gen_one, order):
        reasonings[qid] = txt
        if len(reasonings) % 25 == 0:
            print(f"[{len(reasonings)}/100] {LEDGER.totals()[2]:,}", flush=True)

json.dump(reasonings, open(OUT / f"reasonings_{tag}.json", "w"),
          ensure_ascii=False, indent=1)
LEDGER.dump(OUT / f"reasoning_{tag}_ledger.json")

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

problems = []
rows = list(csv.DictReader(open(ROOT / out_name, encoding="utf-8-sig")))
sp = sum(int(r["prompt_tokens"]) for r in rows[1:])
sc = sum(int(r["completion_tokens"]) for r in rows[1:])
if (sp, sc) != (int(rows[0]["prompt_tokens"]), int(rows[0]["completion_tokens"])):
    problems.append("对账不平")
short = [r["qid"] for r in rows[1:] if len(r["reasoning"]) < 20]
if short:
    problems.append(f"reasoning<20字:{short}")
if not (500_000 <= t <= 5_000_000):
    problems.append(f"token {t:,} 出满分区间")
print(f"done {time.time()-t0:.0f}s | tokens合计 {t:,} | 问题: {problems or '无'}")
print("→", ROOT / out_name)
