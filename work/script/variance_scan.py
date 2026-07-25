#!/usr/bin/env python3
"""评委席方差扫描：对每行以官方 rubric 独立采样 3 次，输出均值与极差。
极差大的行 = 真评委掷骰下的风险行（新信号轴：稳定性,不是均值）。
用法: .venv/bin/python script/variance_scan.py
"""
import json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent.qwen_client import chat, DEFAULT_MODEL  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
RUBRIC = (
    "你是推理过程评分的LLM judge。仅检查提交的推理文本写作质量，"
    "不加载任何外部题目、原文或答案数据。按三个维度打分(各0-100)：\n"
    "logical: 步骤间因果清晰、链条自洽\ncompleteness: 定位、提取、推导、结论完整\n"
    "clarity: 条理清晰、结构化、表达准确\n"
    "只输出JSON: {\"logical\":N,\"completeness\":N,\"clarity\":N}")


def score_once(text):
    c1, _r, _u = chat([{"role": "user", "content":
                        RUBRIC + "\n\n推理文本:\n" + text}],
                      qid="_var", model=DEFAULT_MODEL, thinking=False,
                      max_tokens=80, tag="varscan")
    m = re.search(r"\{.*\}", c1 or "", re.S)
    try:
        d = json.loads(m.group(0))
        return (float(d["logical"]) + float(d["completeness"])
                + float(d["clarity"])) / 3
    except Exception:  # noqa: BLE001
        return None


def scan(q, text):
    xs = [s for s in (score_once(text) for _ in range(3)) if s is not None]
    if not xs:
        return q, None, None
    return q, sum(xs) / len(xs), max(xs) - min(xs)


def main():
    R = json.load(open(OUT / "b_final1" / "reasonings.json"))
    res = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for f in as_completed([ex.submit(scan, q, t) for q, t in R.items()]):
            q, mean, spread = f.result()
            res[q] = {"mean": mean, "spread": spread}
    json.dump(res, open(OUT / "variance_scan.json", "w"), indent=1)
    risky = sorted((v["spread"], q) for q, v in res.items()
                   if v["spread"] is not None and v["spread"] >= 8)
    means = [v["mean"] for v in res.values() if v["mean"] is not None]
    print(f"全池均值 {sum(means)/len(means):.1f}; 高方差行(极差≥8) {len(risky)}:")
    for s, q in sorted(risky, reverse=True)[:12]:
        print(f"  {q} 极差{s:.0f} 均值{res[q]['mean']:.0f}")


if __name__ == "__main__":
    main()
