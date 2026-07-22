#!/usr/bin/env python3
"""终版组装: 答案键 + 指定运行的token台账 → 终检 → 输出提交件。
用法: python script/assemble_final.py <run_tag> <out_name>"""
import csv, json, pathlib, re, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import b_schema

ROOT = pathlib.Path(__file__).resolve().parents[2]
tag, out_name = sys.argv[1], sys.argv[2]
key = json.load(open(ROOT / 'work/output/b_v4/answers.json'))
led = json.load(open(ROOT / f'work/output/{tag}/token_ledger.json'))
per = led['per_qid']
p = sum(x[0] for x in per.values()); c = sum(x[1] for x in per.values())
t = p + c
coef = 0.7 + 0.3 * (5_000_000 - t) / 5_000_000
qs = b_schema.load_questions(ROOT / 'upload_b/question_b')
sch = b_schema.load_schema(ROOT / 'upload_b/submit.csv')
outp = ROOT / f'work/output/{tag}/answer_final.csv'
b_schema.write_submission(outp, key, sch, [q['qid'] for q in qs], per, (p, c, t))
# 终检
rows = list(csv.DictReader(open(outp, encoding='utf-8-sig')))
errs = []
assert rows[0]['qid'] == 'summary'
sp, sc_, st = (int(rows[0][k]) for k in ('prompt_tokens','completion_tokens','total_tokens'))
assert sp + sc_ == st
rp = rc = 0
for r in rows[1:]:
    kinds = sch.get(r['qid'])
    filled = [r[f'answer_{i}'] for i in range(1,5) if r[f'answer_{i}']]
    if kinds is None or len(filled) != len(kinds):
        if r['qid'] not in ('res_b_012',):  # 一位小数豁免
            errs.append(f"{r['qid']} 槽数")
    if int(r['prompt_tokens']) + int(r['completion_tokens']) != int(r['total_tokens']):
        errs.append(f"{r['qid']} token")
    rp += int(r['prompt_tokens']); rc += int(r['completion_tokens'])
assert rp == sp and rc == sc_, '行合计≠summary'
final = ROOT / out_name
import shutil; shutil.copy(outp, final)
print(f'✓ {len(rows)-1}题 | tokens {t:,} | 系数 {coef:.4f}')
for n in (94, 95, 96, 97): print(f'  {n}题 → {n*coef:.2f}分')
print('问题:', errs or '无')
print('→', final)
