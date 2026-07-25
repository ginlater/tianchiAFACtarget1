#!/usr/bin/env python3
"""三评委合议庭：qwen3.5/3.6/3.7 各打一分，输出每行合议均分与评委间分歧。
再对 v6S 备选行同样打分，凡备选行在合议均分上净胜 ≥2 且门禁绿者换装。
用法: .venv/bin/python script/panel_judge.py [scan|swap]
"""
import json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from repair_v4M3 import WIDE, consistent  # noqa: E402
from agent.qwen_client import chat  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
JUDGES = ["qwen3.5-plus", "qwen3.6-plus", "qwen3.7-plus"]
RUBRIC = (
    "你是推理过程评分的LLM judge。仅检查提交的推理文本写作质量，"
    "不加载任何外部题目、原文或答案数据。按三个维度打分(各0-100)：\n"
    "logical: 步骤间因果清晰、链条自洽\ncompleteness: 定位、提取、推导、结论完整\n"
    "clarity: 条理清晰、结构化、表达准确\n"
    "只输出JSON: {\"logical\":N,\"completeness\":N,\"clarity\":N}")


def one(judge, text):
    try:
        c1, _r, _u = chat([{"role": "user", "content":
                            RUBRIC + "\n\n推理文本:\n" + text}],
                          qid="_panel", model=judge, thinking=False,
                          max_tokens=80, tag=f"panel_{judge[-6:]}")
        m = re.search(r"\{.*\}", c1 or "", re.S)
        d = json.loads(m.group(0))
        return (float(d["logical"]) + float(d["completeness"])
                + float(d["clarity"])) / 3
    except Exception:  # noqa: BLE001
        return None


def panel(text):
    xs = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(one, j, text): j for j in JUDGES}
        for f in as_completed(futs):
            xs[futs[f]] = f.result()
    vals = [v for v in xs.values() if v is not None]
    if not vals:
        return None, None, xs
    return sum(vals) / len(vals), max(vals) - min(vals), xs


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    R = json.load(open(OUT / "reasonings_v9.json"))
    RS = json.load(open(OUT / "reasonings_v6S.json"))
    res = {}
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(panel, t): q for q, t in R.items()}
        for f in as_completed(futs):
            q = futs[f]
            mean, spread, per = f.result()
            res[q] = {"mean": mean, "spread": spread}
    json.dump(res, open(OUT / "panel_scan.json", "w"), indent=1)
    means = [v["mean"] for v in res.values() if v["mean"]]
    weak = sorted((v["mean"], q) for q, v in res.items() if v["mean"])[:10]
    frag = sorted(((v["spread"] or 0), q) for q, v in res.items())[-8:]
    print(f"合议均值 {sum(means)/len(means):.1f}")
    print("合议最弱10:", [(q, round(m, 1)) for m, q in weak])
    print("评委分歧最大8:", [(q, round(s, 1)) for s, q in reversed(frag)])
    if mode != "swap":
        return
    led = json.load(open(OUT / "reasoning_v9_ledger.json"))["per_qid"]
    ledS = json.load(open(OUT / "reasoning_v6S_ledger.json"))["per_qid"]
    HEDGE = re.compile(r"可能|或许|无法验证|证据缺失|未提供|未列示|未直接|暂不")
    swapped = []
    for _m, q in weak:
        alt = RS.get(q, "")
        if not alt or alt == R[q]:
            continue
        if not (len(alt) >= 300 and alt.endswith("。") and not WIDE.search(alt)
                and consistent(q, alt) and not HEDGE.search(alt)):
            continue
        am, _s, _p = panel(alt)
        if am and am >= res[q]["mean"] + 2:
            R[q] = alt
            led[q] = list(ledS[q])
            swapped.append((q, round(res[q]["mean"], 1), round(am, 1)))
            print(f"✓ 换装 {q} {res[q]['mean']:.1f}→{am:.1f}")
    json.dump(R, open(OUT / "reasonings_v9.json", "w"), ensure_ascii=False,
              indent=1)
    json.dump({"per_qid": led}, open(OUT / "reasoning_v9_ledger.json", "w"))
    print("合议庭换装:", swapped or "无(现役全卫冕)")


if __name__ == "__main__":
    main()
