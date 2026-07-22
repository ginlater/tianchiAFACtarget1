#!/usr/bin/env python3
"""数值归一假设下的分数方程可行性求解。"""
import csv, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[2]
FILES = [("answer_b_final_20260722.csv", 84), ("answer_b_v4_20260722.csv", 83),
         ("answer_b_v4p2_20260722.csv", 84), ("answer_b_v4p3_20260722.csv", 90),
         ("answer_b_v4p4_20260722.csv", 92), ("answer_b_v5_20260722.csv", 93),
         ("answer_b_v6_20260722.csv", 93), ("answer_b_v7_20260722.csv", 94),
         ("answer_b_slim3_20260723.csv", 95)]


def canon(v):
    out = []
    for p in v.split("|"):
        if re.fullmatch(r"-?\d+(\.\d+)?%?", p):
            num = p.rstrip("%")
            out.append(str(float(num)) + ("%" if p.endswith("%") else ""))
        else:
            out.append(p)
    return "|".join(out)


def load(f):
    return {r["qid"]: canon("|".join(r[f"answer_{i}"] for i in range(1, 5)).rstrip("|"))
            for r in csv.DictReader(open(ROOT / f, encoding="utf-8-sig"))
            if r["qid"] != "summary"}


subs = [(load(f), c) for f, c in FILES]
D = sorted(q for q in subs[0][0] if len({s[q] for s, _ in subs}) > 1)
target = tuple(c - subs[0][1] for _, c in subs[1:])
HARD = {"ins_b_005": "BD", "ins_b_006": "BCD", "res_b_020": "BD", "fc_b_016": "C",
        "fin_b_012": "AB", "res_b_008": "AC", "ins_b_012": "BCD", "fin_b_011": "ABC"}
TF = {"fc_b_013"}
moves_of = {}
for q in D:
    vals = sorted({s[q] for s, _ in subs})
    if q in HARD:
        opts = [HARD[q]]
    elif q in TF:
        opts = [v for v in vals if v in ("A", "B")]
    else:
        opts = vals + ["__OTHER__"]
    moves_of[q] = [(t, tuple((1 if subs[i + 1][0][q] == t else 0)
                             - (1 if subs[0][0][q] == t else 0)
                             for i in range(8))) for t in opts]
layers = [{tuple([0] * 8)}]
for q in D:
    nxt = set()
    for vec in layers[-1]:
        for _, mv in moves_of[q]:
            nxt.add(tuple(v + m for v, m in zip(vec, mv)))
    layers.append(nxt)
print("数值归一假设下可行:", target in layers[-1])
if target in layers[-1]:
    suffix = [set() for _ in range(len(D) + 1)]
    suffix[len(D)] = {target}
    for i in range(len(D) - 1, -1, -1):
        keep = set()
        for vec in layers[i]:
            for _, mv in moves_of[D[i]]:
                if tuple(v + m for v, m in zip(vec, mv)) in suffix[i + 1]:
                    keep.add(vec)
                    break
        suffix[i] = keep
    for i, q in enumerate(D):
        feas = [t for t, mv in moves_of[q]
                if any(tuple(v + m for v, m in zip(vec, mv)) in suffix[i + 1]
                       for vec in suffix[i])]
        print(f"  {q}: {feas}" + (" *钉死" if len(feas) == 1 else ""))
