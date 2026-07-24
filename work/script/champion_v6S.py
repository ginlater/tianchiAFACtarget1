#!/usr/bin/env python3
"""新血锦标赛：v6S 结构化散文体 vs 现役 v9，逐行双盲对打。
采用条件 = v6S行全门禁绿 AND 双盲复评 ≥ v9行 +1.5。
用法: .venv/bin/python script/champion_v6S.py
"""
import json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from repair_v4M3 import WIDE, consistent  # noqa: E402
from uplift_bottom import dim_score  # noqa: E402
from agent.qwen_client import LEDGER  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
HEDGE = re.compile(r"可能|或许|无法验证|证据缺失|未提供|未列示|未直接|暂不")
SCAF = re.compile(r"第[一二三四]步")
META = re.compile(r"四要素|改写|本段|上述要求|指令|解题记录")

R9 = json.load(open(OUT / "reasonings_v9.json"))
led9 = json.load(open(OUT / "reasoning_v9_ledger.json"))["per_qid"]
RS = json.load(open(OUT / "reasonings_v6S.json"))
ledS = json.load(open(OUT / "reasoning_v6S_ledger.json"))["per_qid"]
base_nums = {q: set(re.findall(r"\d[\d,\.]{2,}", t)) for q, t in R9.items()}


def gates(q, t):
    nums = base_nums.get(q, set())
    numok = (not nums) or sum(1 for n in nums if n in t) / len(nums) >= 0.7
    return (len(t) >= 300 and t.endswith("。") and not WIDE.search(t)
            and consistent(q, t) and not HEDGE.search(t)
            and not SCAF.search(t) and not META.search(t) and numok)


def duel(q):
    new = RS.get(q, "")
    if not gates(q, new):
        return q, None, "门禁"
    d_old = dim_score(R9[q])
    d_new = dim_score(new)
    a_old = sum(d_old.values()) / 3
    a_new = sum(d_new.values()) / 3
    if a_new >= a_old + 1.5:
        return q, new, f"{a_old:.0f}→{a_new:.0f}"
    return q, None, f"复评{a_new:.0f}vs{a_old:.0f}无优势"


def main():
    adopted = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        for f in as_completed([ex.submit(duel, q) for q in R9]):
            q, new, note = f.result()
            if new:
                R9[q] = new
                led9[q] = list(ledS[q])
                adopted.append((q, note))
                print(f"✓ {q} {note}", flush=True)
    json.dump(R9, open(OUT / "reasonings_v9.json", "w"), ensure_ascii=False,
              indent=1)
    json.dump({"per_qid": led9}, open(OUT / "reasoning_v9_ledger.json", "w"))
    print(f"锦标赛采用 {len(adopted)}: {[q for q, _ in adopted]}")


if __name__ == "__main__":
    main()
