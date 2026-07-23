#!/usr/bin/env python3
"""B榜新规组装器：答案键 + 基座运行台账 + reasoning列（含生成消耗合并计账）。

新规公式: 总分 = acc×0.6 + 推理过程分×0.2 + Token效率分×0.2
Token效率: [500k,5M]=100分; <500k线性递减(actual/500000×100)。
用法: .venv/bin/python script/assemble_final2.py <base_tag> <out_name.csv>
"""
import csv, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import b_schema  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
OUT = WORK / "output"

base_tag = sys.argv[1]
out_name = sys.argv[2]

key = json.load(open(OUT / "b_v4" / "answers.json"))
reasonings = json.load(open(OUT / "reasonings.json"))
base_led = json.load(open(OUT / base_tag / "token_ledger.json"))
rea_leds = [json.load(open(p)) for p in sorted(OUT.glob("reasoning_ledger*.json"))]

qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
schema = b_schema.load_schema(str(ROOT / "upload_b" / "submit.csv"))
order = [q["qid"] for q in qs_all]

# 合并逐题台账（生成reasoning的消耗按新规计入对应题）
per_qid = {}
for src in [base_led["per_qid"]] + [rl["per_qid"] for rl in rea_leds]:
    for k, v in src.items():
        slot = per_qid.setdefault(k, [0, 0])
        slot[0] += v[0]
        slot[1] += v[1]
p = sum(v[0] for v in per_qid.values())
c = sum(v[1] for v in per_qid.values())
t = p + c

b_schema.write_submission(ROOT / out_name, key, schema, order, per_qid,
                          (p, c, t), reasonings=reasonings)

# ---- 自检 ----
problems = []
rows = list(csv.DictReader(open(ROOT / out_name, encoding="utf-8-sig")))
assert rows[0]["qid"] == "summary"
sp = sc = st = 0
for r in rows[1:]:
    sp += int(r["prompt_tokens"]); sc += int(r["completion_tokens"])
    st += int(r["total_tokens"])
    if int(r["total_tokens"]) != int(r["prompt_tokens"]) + int(r["completion_tokens"]):
        problems.append(f"{r['qid']} total不等于p+c")
    if len(r["reasoning"]) < 20:
        problems.append(f"{r['qid']} reasoning<20字")
if (sp, sc, st) != (int(rows[0]["prompt_tokens"]),
                    int(rows[0]["completion_tokens"]),
                    int(rows[0]["total_tokens"])):
    problems.append("逐题合计≠summary")
if not (500_000 <= t <= 5_000_000):
    problems.append(f"token总量{t:,}不在满分区间[500k,5M]")

tok_score = 100 if 500_000 <= t <= 5_000_000 else (
    t / 500_000 * 100 if t < 500_000 else max(0, 100 * (1 - (t - 5e6) / 5e6)))
print(f"tokens: {t:,} (p={p:,} c={c:,}) → Token效率分 {tok_score:.1f}")
for acc in (90, 95, 97, 99):
    for rs in (70, 80, 90):
        pass
print("新公式分数预览(acc/100×60 + 推理分×0.2 + token分×0.2):")
for acc in (95, 97, 99):
    for rs in (70, 80):
        print(f"  {acc}题+推理{rs}分 → {acc*0.6 + rs*0.2 + tok_score*0.2:.1f}")
print("问题:", problems or "无")
print("→", ROOT / out_name)
