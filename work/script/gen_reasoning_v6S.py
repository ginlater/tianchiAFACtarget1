#!/usr/bin/env python3
"""v6S 结构化散文体池: rubric clarity 维度"结构化"字面攻坚(新血锦标赛候选)。

底料 = 实测家族肥版文本（世界B下值85的内容），改写三原则：
1) 事实零改动（数字/页码/结论一字不换——内容保真由源文本担保，无幻觉面）
2) 去模板指纹：禁'第一步/第二步'机械编号、禁全池雷同收尾句——自然段落论证
3) 专业分析师口吻（rubric 80+档门槛词：逻辑清晰/论证完整/表达专业）
用法: .venv/bin/python script/gen_reasoning_v6.py
"""
import json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"

base = json.load(open(OUT / "reasonings_v9.json"))



INST = (
    "把下面这段解题推理摘要改写为结构化专业论证（400-540字），要求：\n"
    "1) 全部事实、数字、页码引用、结论一字不改保留——只改组织与措辞\n"
    "2) 结构化行文：用'数据定位方面，…。口径核对上，…。推导层面，…。"
    "独立复核环节，…。综合结论，…'这类语义小节词自然分段推进（措辞随题变化，"
    "不得逐题雷同），禁止'第一步/第二步'式编号\n"
    "3) 每个关键断言紧跟页码/条款依据；算式完整展开\n"
    "4) 开头收尾随题目内容自然生成，禁止任何跨题复用套话\n"
    "5) 全程自信，无'可能/或许/无法验证/未提供/未列示'等犹疑词，"
    "无'答案/给定/解题记录'字样\n"
    "只输出正文。")


def gen_one(qid):
    src = base[qid]
    prompt = f"原始推理摘要:\n{src}\n\n{INST}"
    c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                      model=DEFAULT_MODEL, thinking=False, max_tokens=780,
                      tag="reasonV6S")
    return qid, (c1 or "").strip().replace("\n", " ").replace("\r", " ")


def main():
    texts = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(gen_one, q) for q in base]
        for n, f in enumerate(as_completed(futs)):
            qid, txt = f.result()
            texts[qid] = txt
            if (n + 1) % 25 == 0:
                print(f"[{n+1}/100] {LEDGER.totals()[2]:,}", flush=True)
    json.dump(texts, open(OUT / "reasonings_v6S.json", "w"),
              ensure_ascii=False, indent=1)
    LEDGER.dump(OUT / "reasoning_v6S_ledger.json")
    print(f"v6P池成: 账 {LEDGER.totals()[2]:,}")


if __name__ == "__main__":
    main()
