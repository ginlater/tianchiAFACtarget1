#!/usr/bin/env python
"""为 mix 家族（混合/批量路由系 run）生成逐题排名档案 output/family_rank_mix.json。

规则：
- 每个 run 需同时具备 answers.json 与 token_ledger.json，缺任一则跳过并记录。
- 判对：与 b_v4 答案键 canon 归一后全等。
- 每题成本 = per_qid[qid](prompt+completion) + 共享开销摊派；
  共享开销 = calls 总 token − per_qid 中真实题目键（不含下划线伪键）总 token，
  非正则取 0，按该 run 覆盖题数（answers.json 题数）均摊。
- 只收录答对记录，rank 按成本升序。
"""
import json
import os
import re

WORK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(WORK, "output")

RUNS = ["b_mixA", "b_mixB", "b_mixC", "b_finsB", "b_scout", "b_e1fin", "b_finsC"]


def canon(v):
    parts = v if isinstance(v, list) else [v]
    out = []
    for x in parts:
        x = str(x).strip()
        out.append(str(float(x.rstrip('%'))) if re.fullmatch(r'-?\d+(\.\d+)?%?', x) else x)
    return '|'.join(out)


def main():
    key = json.load(open(os.path.join(OUT, "b_v4", "answers.json")))
    key_canon = {q: canon(v) for q, v in key.items()}

    used_runs = []
    skipped = []
    rank = {}

    for tag in RUNS:
        d = os.path.join(OUT, tag)
        ans_p = os.path.join(d, "answers.json")
        led_p = os.path.join(d, "token_ledger.json")
        missing = [os.path.basename(p) for p in (ans_p, led_p) if not os.path.exists(p)]
        if missing:
            skipped.append({"run": tag, "missing": missing})
            continue

        answers = json.load(open(ans_p))
        ledger = json.load(open(led_p))
        per_qid = ledger.get("per_qid", {})
        calls = ledger.get("calls", [])

        calls_total = sum(c.get("prompt_tokens", 0) + c.get("completion_tokens", 0) for c in calls)
        real_total = sum(sum(v) for k, v in per_qid.items() if not k.startswith("_"))
        covered = len(answers)
        shared = calls_total - real_total
        share = (shared / covered) if (shared > 0 and covered > 0) else 0.0

        n_correct = 0
        for qid, val in answers.items():
            if qid not in key_canon:
                continue
            if canon(val) != key_canon[qid]:
                continue
            n_correct += 1
            direct = sum(per_qid.get(qid, [0, 0]))
            cost = round(direct + share, 2)
            rank.setdefault(qid, []).append([cost, tag])

        used_runs.append({
            "run": tag,
            "covered": covered,
            "correct": n_correct,
            "calls_total_tokens": calls_total,
            "shared_overhead_tokens": max(0, shared),
            "shared_per_q": round(share, 2),
        })

    for qid in rank:
        rank[qid].sort(key=lambda x: (x[0], x[1]))

    all_qids = sorted(key_canon.keys())
    no_correct = [q for q in all_qids if q not in rank]

    result = {
        "family": "mix",
        "runs": used_runs,
        "skipped": skipped,
        "rank": {q: rank[q] for q in sorted(rank)},
        "no_correct": no_correct,
    }
    out_p = os.path.join(OUT, "family_rank_mix.json")
    with open(out_p, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    cheapest_sum = sum(v[0][0] for v in rank.values())
    print("runs_used:", len(used_runs))
    print("skipped:", skipped)
    print("qids_covered_correct:", len(rank))
    print("no_correct:", len(no_correct))
    print("cheapest_correct_cost_sum:", round(cheapest_sum, 2))
    for r in used_runs:
        print(" ", r)


if __name__ == "__main__":
    main()
