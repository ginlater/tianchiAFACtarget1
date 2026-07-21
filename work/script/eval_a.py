#!/usr/bin/env python3
"""对照验证标签集评估一次运行的答案。

用法: python work/script/eval_a.py work/output/<tag>/answers.json
"""
import json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
LABELS = json.load(open(ROOT / "work" / "eval" / "validation_labels.json"))


def main(path):
    ours = json.load(open(path))
    stats = {"locked": [0, 0], "master": [0, 0]}
    diffs, wrong_hits = [], []
    for qid, ans in sorted(ours.items()):
        lab = LABELS.get(qid)
        if not lab:
            continue
        conf = lab["confidence"]
        if ans in lab["known_wrong"]:
            wrong_hits.append((qid, ans))
        if conf == "master_wrong":
            continue  # 无可靠标签
        key = "locked" if conf == "locked" else "master"
        stats[key][1] += 1
        if ans == lab["answer"]:
            stats[key][0] += 1
        else:
            diffs.append((qid, conf, ans, lab["answer"]))

    lk, lm = stats["locked"], stats["master"]
    print(f"locked(硬标签): {lk[0]}/{lk[1]}")
    print(f"master(94母版,含4道未知错题): {lm[0]}/{lm[1]}")
    if lm[1]:
        est = lk[0] + lm[0] + max(0, min(4, lm[1] - lm[0]))
        print(f"→ 估计真实正确数区间: [{lk[0]+lm[0]}, {est}] / {lk[1]+lm[1]}")
    if wrong_hits:
        print(f"!! 命中已知错误答案: {wrong_hits}")
    print("\n与标签不一致的题:")
    for qid, conf, ours_a, lab_a in diffs:
        print(f"  {qid} [{conf}] ours={ours_a} label={lab_a}")


if __name__ == "__main__":
    main(sys.argv[1])
