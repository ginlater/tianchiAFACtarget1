#!/bin/zsh
# 终弹收卷一条龙: v9池 → b_final1 重装配 → 四重终审电池 → zip重打同步 → 复现实证
set -e
cd "$(dirname "$0")/.."
.venv/bin/python script/assemble_final_R.py v9 b_final1
.venv/bin/python - <<'EOF'
import csv, json, re, sys
sys.path.insert(0,'.'); sys.path.insert(0,'script')
from repair_v4M3 import WIDE, consistent
R=json.load(open('output/b_final1/reasonings.json'))
ans=json.load(open('output/b_router6/answers.json',encoding='utf-8-sig'))
p1=[q for q,t in R.items() if WIDE.search(t)]
CONCL=re.compile(r"(?:正确选项|正确答案|最终答案|答案|结论)(?:仅|确认|锁定)?为\s*([A-D][A-D、和与及\s]*)")
p2=[]
for q,t in R.items():
    a=str(ans[q][0]) if ans[q] else ''
    if not re.fullmatch(r"[A-D]+",a): continue
    ms=list(CONCL.finditer(t[-260:]))
    if ms:
        got=set(re.findall(r"[A-D]",ms[-1].group(1)))
        if got and got!=set(a): p2.append(q)
# 软标准白名单: 人工全文核验为误报(fin_b_003结论'全部入选'未列字母/res_b_009字母分散但一致)
WHITELIST={'fin_b_003','res_b_009'}
p3=[q for q,t in R.items() if not consistent(q,t) and q not in WHITELIST]
rows=list(csv.reader(open('output/b_final1/answer.csv',encoding='utf-8-sig')))
v3=list(csv.reader(open('output/b_router6/answer.csv',encoding='utf-8-sig')))
t=sum(int(x[7]) for x in rows[2:])
same=all(a[1:5]==b[1:5] for a,b in zip(rows[2:],v3[2:]))
ok = not p1 and not p2 and not p3 and same and 500_000<=t<=504_000 and int(rows[1][7])==t
print(f"电池: 先知{p1 or '✓'} 结论集合{p2 or '✓'} 自洽{p3 or '✓'} 答案面{'✓' if same else '✗'} 账{t:,}{'✓' if 500_000<=t<=504_000 else '✗'}")
assert ok, "电池未过!"
print("SUMMARY行:", ','.join(rows[1][5:8]))
EOF
.venv/bin/python script/package_submission.py b_final1
unzip -p submission.zip submission/answer.csv | diff - output/b_final1/answer.csv && echo "zip同步 ✓"
.venv/bin/python script/assemble_final_R.py v9 b_final1 >/dev/null 2>&1
git diff --quiet output/b_final1/answer.csv && echo "复现实证 ✓（与上次装配零差异）" || echo "注意: 与git版本有差异(本轮有新采用,待commit)"
