#!/usr/bin/env python3
"""章节全文计算架构（7-25 res_b_005 第二发弹药线）。

结构：计算题证据 = 命中文档的完整前部章节（含报告摘要与全部预测表），
零检索零卡片——检索类架构的口径盲区（BM25 只召回题面词汇命中的表）在此
被整章上下文根治，口径取舍完全交还模型。单掷认账，产出即弹药。

用法: .venv/bin/python script/run_calc_chapter.py
"""
import json, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent.calc import CALC_INST, parse_calc, valid_calc  # noqa: E402
from agent.qwen_client import chat, LEDGER  # noqa: E402

WORK = pathlib.Path(__file__).resolve().parents[1]
OUT = WORK / "output" / "b_calcChap1"
QID = "res_b_005"
DOC = "pack2_text04"
PAGE_CUT = 21  # P1-P20: 摘要+销量+带电量+乘用车/商用车/合计全部预测表

QUESTION = ("若2026年国内新能源乘用车销量与2025年持平，但单车带电量因技术升级"
            "从2025年的水平提升至56kWh，则全年动力电池需求同比增速最接近多少？")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    text = open(WORK / "processed_data" / "research" / f"{DOC}.txt",
                encoding="utf-8").read()
    pages = {int(m.group(1)): m.start()
             for m in re.finditer(r"\[P(\d+)\]", text)}
    chapter = text[:pages[PAGE_CUT]]
    inst = CALC_INST.format(
        n=1, slots="  第1个：百分数，形如 12.34%，保留两位小数",
        template="12.34%")
    prompt = (f"研究报告《{DOC}》完整章节（P1-P{PAGE_CUT-1}，含摘要与全部预测表）:\n"
              + chapter + "\n\n题目:\n" + QUESTION + "\n\n" + inst)
    c1, _r, usage = chat([{"role": "user", "content": prompt}], qid=QID,
                         model="qwen3.6-plus", thinking=True,
                         thinking_budget=3200, max_tokens=3600, tag="chap1")
    a1 = parse_calc(c1)
    ok = valid_calc(a1, ["percent"])
    with open(OUT / "run_log.jsonl", "a") as f:
        f.write(json.dumps({"qid": QID, "final": a1, "a1": a1, "valid": ok,
                            "arch": f"chapter_full P1-P{PAGE_CUT-1}",
                            "c1": c1}, ensure_ascii=False) + "\n")
    json.dump({QID: [a1]}, open(OUT / "answers.json", "w"),
              ensure_ascii=False, indent=1)
    LEDGER.dump(OUT / "token_ledger.json")
    led = json.load(open(OUT / "token_ledger.json"))["per_qid"][QID]
    print(f"final={a1!r} valid={ok} 账={sum(led):,} (p={led[0]:,} c={led[1]:,})")


if __name__ == "__main__":
    main()
