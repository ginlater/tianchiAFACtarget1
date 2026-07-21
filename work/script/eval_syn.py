#!/usr/bin/env python3
"""对抗题库压测评估：对比流水线答案与题库标准答案。

用法: python work/script/eval_syn.py work/output/<tag>/answers.json
"""
import json, pathlib, sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
SYN = ROOT / "work" / "eval" / "synthetic"


def load_syn():
    qs = {}
    for f in sorted(SYN.glob("*_hard.json")):
        for q in json.load(open(f)):
            qs[q["qid"]] = q
    return qs


def main(path):
    ours = json.load(open(path))
    syn = load_syn()
    ok_d, tot_d = Counter(), Counter()
    trap_ok, trap_tot = Counter(), Counter()
    wrong = []
    for qid, ans in sorted(ours.items()):
        q = syn.get(qid)
        if not q:
            continue
        d, t = q["domain"], q.get("trap_type", "?")
        tot_d[d] += 1
        trap_tot[t] += 1
        if ans == q["answer"]:
            ok_d[d] += 1
            trap_ok[t] += 1
        else:
            wrong.append((qid, ans, q["answer"], t))
    print("分域:")
    for d in sorted(tot_d):
        print(f"  {d}: {ok_d[d]}/{tot_d[d]}")
    print(f"  总计: {sum(ok_d.values())}/{sum(tot_d.values())}")
    print("分陷阱类型:")
    for t in sorted(trap_tot):
        print(f"  {t}: {trap_ok[t]}/{trap_tot[t]}")
    print("\n错题:")
    for qid, a, g, t in wrong:
        print(f"  {qid} [{t}] ours={a} gold={g}")


if __name__ == "__main__":
    main(sys.argv[1])
