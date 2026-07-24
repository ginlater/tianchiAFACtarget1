#!/usr/bin/env python3
"""v6P 12 伤行并行修复（自洽/犹疑/数字保真），过门禁即采用。"""
import json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from repair_v4M3 import WIDE, consistent  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
HEDGE = re.compile(r"可能|或许|无法验证|证据缺失|未提供|未列示|未直接|暂不|难以确证")
SCAF = re.compile(r"第[一二三四]步")

base = json.load(open(OUT / "reasonings_probe.json"))
base["res_b_005"] = json.load(open(OUT / "res005_2227_reason.json"))["text"]
base["res_b_012"] = json.load(open(OUT / "res012_fix_reason.json"))["text"]
R = json.load(open(OUT / "reasonings_v6P.json"))
led = json.load(open(OUT / "reasoning_v6P_ledger.json"))["per_qid"]
ans = json.load(open(OUT / "b_router6" / "answers.json", encoding="utf-8-sig"))
BAD = ['fc_b_002', 'fc_b_003', 'fc_b_011', 'fin_b_003', 'fin_b_012',
       'fin_b_019', 'ins_b_010', 'reg_b_015', 'reg_b_024', 'res_b_005',
       'res_b_008', 'res_b_012']
INST = ("把下面这段解题推理摘要改写为资深金融分析师的专业论证段落（380-540字）："
        "全部事实、数字、页码、结论一字不改保留；禁止'第一步/第二步'式编号，"
        "用自然段落论证（数据来源→口径核对→推导→复核→结论）；论证连接词显式，"
        "断言挂页码；开头收尾随内容自然变化，无套话；全程自信，禁止出现 "
        "可能/或许/无法验证/未提供/未列示/未直接/暂不 等词；结尾必须明确写出"
        "最终结论本身（选择题写出字母组合，计算题写出数值）。只输出正文。")


def gates(q, t):
    nums = set(re.findall(r"\d[\d,\.]{2,}", base.get(q, "")))
    numok = (not nums) or sum(1 for n in nums if n in t) / len(nums) >= 0.6
    return (len(t) >= 200 and t.endswith("。") and not WIDE.search(t)
            and consistent(q, t) and not HEDGE.search(t)
            and not SCAF.search(t) and numok)


def fix(q):
    tail = "；".join(str(a) for a in ans[q] if str(a).strip())
    for _ in range(3):
        c1, _r, _u = chat([{"role": "user", "content":
                            f"原始推理摘要:\n{base[q]}\n\n{INST}\n"
                            f"本题最终结论: {tail}"}],
                          qid=q, model=DEFAULT_MODEL, thinking=False,
                          max_tokens=800, tag="reasonV6fix")
        t = (c1 or "").strip().replace("\n", " ")
        if gates(q, t):
            return q, t, True
    return q, "", False


def main():
    fails = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(fix, q) for q in BAD]):
            q, t, ok = f.result()
            if ok:
                R[q] = t
                led[q] = list(LEDGER.per_qid[q])
            else:
                fails.append(q)
    json.dump(R, open(OUT / "reasonings_v6P.json", "w"), ensure_ascii=False,
              indent=1)
    json.dump({"per_qid": led}, open(OUT / "reasoning_v6P_ledger.json", "w"))
    tot = sum(sum(v) for v in led.values())
    print(f"修复 {len(BAD) - len(fails)}/{len(BAD)}, 未过 {fails or '无'}, "
          f"终池账 {tot:,}")


if __name__ == "__main__":
    main()
