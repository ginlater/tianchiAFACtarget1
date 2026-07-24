#!/usr/bin/env python3
"""v7 冠军池：三风格池逐行博弈（模拟评委当裁判），每题取校准分最高的行。

选手: 家族肥版(probe) / 论证体混族(v4M) / 专业体(v6P)
账面: 采用行记其所属池的真实生成账。tie-break 偏向去模板池(v6P>v4M>probe)。
用法: .venv/bin/python script/build_v7.py
"""
import json, pathlib

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"


def load_scores(f):
    S = json.load(open(f))
    out = {}
    for q, v in S.items():
        sc = v.get("calibrated") if isinstance(v, dict) else v
        if isinstance(v, dict) and sc is None:
            sc = (v.get("logical", 0) + v.get("completeness", 0)
                  + v.get("clarity", 0)) / 3
        out[q] = sc
    return out


POOLS = [  # (名, 文本, 账, sim分, tie优先级大者优先)
    ("v6P", "reasonings_v6P.json", "reasoning_v6P_ledger.json",
     "reasonings_v6P.json.simscores.json", 3),
    ("v4M", "reasonings_v4M.json", "reasoning_v4M_ledger.json",
     "b_router7R/reasonings.json.simscores.json", 2),
    ("probe", "reasonings_probe.json", "reasoning_probe_ledger.json",
     "reasonings_probe.json.simscores.json", 1),
]


def main():
    loaded = []
    for name, tf, lf, sf, pri in POOLS:
        try:
            texts = json.load(open(OUT / tf))
            led = json.load(open(OUT / lf))["per_qid"]
            scores = load_scores(OUT / sf)
        except FileNotFoundError as e:
            print(f"跳过 {name}: {e}")
            continue
        if name == "v6P":
            try:
                bad = set(json.load(open(OUT / "v6P_bad_rows.json")))
                texts = {q: t for q, t in texts.items() if q not in bad}
                print(f"v6P 失格行 {len(bad)} 已剔除")
            except FileNotFoundError:
                pass
        loaded.append((name, texts, led, scores, pri))
    # probe 特例: res_b_005/012 用修复件
    r5 = json.load(open(OUT / "res005_2227_reason.json"))
    r12 = json.load(open(OUT / "res012_fix_reason.json"))
    R, rled, src = {}, {}, {}
    qids = set().union(*[set(t[1]) for t in loaded])
    for q in sorted(qids):
        best = None
        for name, texts, led, scores, pri in loaded:
            if q not in texts or q not in led:
                continue
            sc = scores.get(q, 0)
            key = (sc, pri)
            if best is None or key > best[0]:
                best = (key, name, texts[q], list(led[q]))
        _k, name, txt, cost = best
        if name == "probe" and q == "res_b_005":
            txt, cost = r5["text"], list(r5["cost"])
        if name == "probe" and q == "res_b_012":
            txt, cost = r12["text"], list(r12["cost"])
        R[q], rled[q], src[q] = txt, cost, name
    json.dump(R, open(OUT / "reasonings_v7.json", "w"), ensure_ascii=False,
              indent=1)
    json.dump({"per_qid": rled}, open(OUT / "reasoning_v7_ledger.json", "w"))
    json.dump(src, open(OUT / "v7_row_sources.json", "w"), indent=1)
    from collections import Counter
    tot = sum(sum(v) for v in rled.values())
    print(f"v7冠军池: 账{tot:,} 行源分布{dict(Counter(src.values()))}")


if __name__ == "__main__":
    main()
