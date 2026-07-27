#!/usr/bin/env python3
"""诚实单跑测量分析：按调用类型(tag)拆解 token，对键算准确率，外推全卷。
用法: .venv/bin/python script/honest_analyze.py <tag>
"""
import json, pathlib, re, sys
from collections import defaultdict

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
tag = sys.argv[1] if len(sys.argv) > 1 else "b_honest1"


def norm_one(v):
    s = str(v).strip().rstrip("％%")
    try:
        return f"{float(s.replace(',', '')):.4f}"
    except ValueError:
        pass
    if re.fullmatch(r"[A-Da-d]+", s):
        return "".join(sorted(set(s.upper())))
    return re.sub(r"\s", "", s)


def norm(a):
    a = a if isinstance(a, list) else [a]
    return tuple(norm_one(x) for x in a if str(x).strip())


def main():
    d = OUT / tag
    led = json.load(open(d / "token_ledger.json"))
    calls = led.get("calls", [])
    per = led["per_qid"]
    ans = json.load(open(d / "answers.json", encoding="utf-8-sig"))
    key = json.load(open(OUT / "b_v4" / "answers.json", encoding="utf-8-sig"))

    # 按 tag 拆解
    by_tag = defaultdict(lambda: [0, 0, 0])
    for c in calls:
        t = c.get("tag", "?")
        by_tag[t][0] += c.get("prompt_tokens", 0)
        by_tag[t][1] += c.get("completion_tokens", 0)
        by_tag[t][2] += 1
    tot = sum(sum(v) for v in per.values())
    n = len(ans)
    print(f"== {tag} ==  {n}题  总账 {tot:,}  均 {tot//max(n,1):,}/题")
    if by_tag:
        print("按调用类型:")
        for t, (p, c, k) in sorted(by_tag.items(), key=lambda x: -sum(x[1][:2])):
            print(f"  {t:12} {p+c:>8,}  ({k}次, {(p+c)//max(k,1):,}/次)")

    # 准确率(对键, 仅测量; res_b_005 跳过)
    ok, bad = 0, []
    for q in ans:
        if q == "res_b_005" or q not in key:
            continue
        if norm(ans[q]) == norm(key[q]):
            ok += 1
        else:
            bad.append(q)
    tested = sum(1 for q in ans if q != "res_b_005" and q in key)
    print(f"准确率(对键): {ok}/{tested}  错题: {bad}")

    # 外推全卷
    if n:
        per_q = tot / n
        full = per_q * 100
        acc_rate = ok / max(tested, 1)
        Tf = (5_000_000 - full) / 5_000_000 * 100 if full >= 500_000 \
            else 90 * (full / 500_000) ** 2
        est_acc = acc_rate * 100
        score = est_acc * 0.5 + 85 * 0.3 + Tf * 0.2
        print(f"外推100题: ~{full:,.0f} tokens | acc~{est_acc:.0f} | "
              f"T~{Tf:.1f} | 诚实总分~{score:.1f} (R按85)")


if __name__ == "__main__":
    main()
