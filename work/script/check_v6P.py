#!/usr/bin/env python3
"""v6P 池全门禁体检：宽域先知/自洽/犹疑扩展/机械编号残留/收尾雷同/长度/事实保真抽检。
用法: .venv/bin/python script/check_v6P.py
"""
import json, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from repair_v4M3 import WIDE, consistent  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
HEDGE = re.compile(r"可能|或许|无法验证|证据缺失|未提供|未列示|未直接|暂不|难以确证")
SCAF = re.compile(r"第[一二三四]步")

R = json.load(open(OUT / "reasonings_v6P.json"))
base = json.load(open(OUT / "reasonings_probe.json"))
probs = {}
tails = {}
for q, t in R.items():
    c = []
    if WIDE.search(t):
        c.append("先知")
    if not consistent(q, t):
        c.append("自洽")
    if HEDGE.search(t):
        c.append("犹疑")
    if SCAF.search(t):
        c.append("编号残留")
    if len(t) < 200:
        c.append("过短")
    if not t.endswith("。"):
        c.append("无句号尾")
    # 事实保真粗检: 源文本中的数字应大体保留（丢失>40%判伤）
    src_nums = set(re.findall(r"\d[\d,\.]{2,}", base.get(q, "")))
    if src_nums:
        kept = sum(1 for n in src_nums if n in t)
        if kept / len(src_nums) < 0.6:
            c.append(f"数字丢失{kept}/{len(src_nums)}")
    tl = t[-12:]
    tails[tl] = tails.get(tl, 0) + 1
    if c:
        probs[q] = c
dup = {k: v for k, v in tails.items() if v > 3}
print(f"伤行 {len(probs)}: ", {q: c for q, c in list(probs.items())[:12]})
print(f"收尾雷同(>3): {dup or '无 ✓'}")
lens = [len(t) for t in R.values()]
print(f"长度: 均{sum(lens)//100} min{min(lens)}")
json.dump(sorted(probs), open(OUT / "v6P_bad_rows.json", "w"))
