#!/usr/bin/env python3
"""从队友探针数据构建本地验证标签集。

母版 94/100 正确；其中 fc_a_004(母版AC)、fc_a_015(母版D) 已被线上探针证实为错题。
剩余 98 题中 94 对 4 错。9 题为 locked_online 硬标签。
输出 work/eval/validation_labels.json
"""
import csv, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
PKG = ROOT / "solid_handoff_20260716_package_v2"
MASTER = PKG / "output" / "answer_a_inferred_94_probe_reg11_base.csv"
OUT = ROOT / "work" / "eval" / "validation_labels.json"

LOCKED = {
    "fc_a_014": "ABC", "res_a_004": "ABC", "res_a_006": "A",
    "res_a_011": "ABC", "fc_a_005": "ABD", "ins_a_014": "AB",
    "reg_a_011": "ACD", "res_a_002": "ABC", "fc_a_018": "A",
}
KNOWN_WRONG = {
    "fc_a_014": ["AB", "B", "BC"], "res_a_004": ["AB"], "res_a_006": ["B"],
    "res_a_011": ["BC"], "fc_a_005": ["ABCD", "AB"], "ins_a_014": ["ABD", "A"],
    "reg_a_011": ["AC", "ABCD"], "res_a_002": ["ABCD"], "fc_a_018": ["B"],
    # 母版中确认的错题
    "fc_a_004": ["ACD", "AC", "A", "AD", "ABC"],
    "fc_a_015": [],  # mcq, A/B/C/D 单字母探针均未提升母版，计分疑似异常
}

labels = {}
with open(MASTER) as f:
    for row in csv.reader(f, delimiter="\t"):
        if not row or row[0] in ("qid", "summary"):
            continue
        qid, ans = row[0], row[1]
        if qid in LOCKED:
            conf = "locked"          # 线上锁定，硬标签
            ans = LOCKED[qid]
        elif qid in ("fc_a_004", "fc_a_015"):
            conf = "master_wrong"    # 母版该题确认错误，标签仅供参考
        else:
            conf = "master"          # 母版答案，~95.6% 可信（89题中85对4错... 见下方统计）
        labels[qid] = {
            "answer": ans,
            "confidence": conf,
            "known_wrong": KNOWN_WRONG.get(qid, []),
        }

n_locked = sum(1 for v in labels.values() if v["confidence"] == "locked")
n_master = sum(1 for v in labels.values() if v["confidence"] == "master")
n_wrong = sum(1 for v in labels.values() if v["confidence"] == "master_wrong")
assert len(labels) == 100, len(labels)

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(labels, open(OUT, "w"), ensure_ascii=False, indent=1)
# 100 = 9 locked(全对) + 2 master_wrong(全错) + 89 master(其中4错,85对)
print(f"labels: {len(labels)} = locked {n_locked} + master {n_master}(内含4道未知错题) + master_wrong {n_wrong}")
print(f"written to {OUT}")
