#!/usr/bin/env python3
"""对照验证键评分：python script/eval_vs_key.py <tag> [tag2 ...]

键面 = output/b_v4/answers.json（99 题可信，res_b_005 除外——该题不计分）。
口径 = 数值等价归一（67.10==67.1、22.19%==22.19），字母集合序无关。
输出：总键数 / 域分布 / 错题清单（与键不一致处）/ 总账。
"""
import json, pathlib, re, sys

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
SKIP = {"res_b_005"}  # 全场唯一无可信键的题


def norm_one(v):
    s = str(v).strip().rstrip("％%")
    try:
        return f"{float(s.replace(',', '')):.4f}"
    except ValueError:
        pass
    if re.fullmatch(r"[A-Da-d]+", s):
        return "".join(sorted(set(s.upper())))
    return re.sub(r"\s", "", s)


def norm(ans):
    if isinstance(ans, list):
        return tuple(norm_one(x) for x in ans if str(x).strip())
    return (norm_one(ans),)


def main(tags):
    key = json.load(open(OUT / "b_v4" / "answers.json", encoding="utf-8-sig"))
    for tag in tags:
        d = OUT / tag
        ans = json.load(open(d / "answers.json", encoding="utf-8-sig"))
        led = {}
        lp = d / "token_ledger.json"
        if lp.exists():
            led = json.load(open(lp)).get("per_qid", {})
        ok, bad, dom_ok, dom_n = 0, [], {}, {}
        for q, kv in sorted(key.items()):
            if q in SKIP or q not in ans:
                continue
            dom = q.split("_b_")[0]
            dom_n[dom] = dom_n.get(dom, 0) + 1
            if norm(ans[q]) == norm(kv):
                ok += 1
                dom_ok[dom] = dom_ok.get(dom, 0) + 1
            else:
                bad.append((q, ans[q], kv))
        tot = sum(sum(v) for v in led.values()) if led else 0
        n = sum(dom_n.values())
        doms = " ".join(f"{d}:{dom_ok.get(d,0)}/{dom_n[d]}"
                        for d in sorted(dom_n))
        print(f"\n== {tag} ==  键 {ok}/{n}  账 {tot:,}")
        print(f"   {doms}")
        for q, a, k in bad:
            print(f"   ✗ {q}: 答{a} 键{k}"
                  + (f"  账{sum(led[q]):,}" if q in led else ""))


if __name__ == "__main__":
    main(sys.argv[1:] or ["b_grandKing1"])
