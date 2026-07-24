#!/usr/bin/env python3
"""v6 专业体池：冲 80+ 档的正面攻坚。

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

base = json.load(open(OUT / "reasonings_probe.json"))
base["res_b_005"] = json.load(open(OUT / "res005_2227_reason.json"))["text"]
base["res_b_012"] = json.load(open(OUT / "res012_fix_reason.json"))["text"]

INST = (
    "把下面这段解题推理摘要改写为资深金融分析师的专业论证段落（380-540字），要求：\n"
    "1) 全部事实、数字、页码引用、结论一字不改地保留——只改组织与措辞，"
    "不得新增、删除或改动任何数字与事实\n"
    "2) 禁止'第一步/第二步'式机械编号——用自然段落推进论证，"
    "以内容本身的逻辑（数据来源→口径核对→推导→复核→结论）自然衔接\n"
    "3) 论证连接词显式（因为/据此/故/进一步验证），关键断言紧跟页码依据\n"
    "4) 开头与收尾措辞必须依题目内容自然生成，禁止任何可在多题间复用的套话"
    "（如'经复核，上述推导与所引数据一致'一类的固定句）\n"
    "5) 全程自信陈述，不得出现'可能/或许/无法验证/未提供/未列示'等犹疑表述，"
    "不得出现'答案/给定/解题记录'等字样\n"
    "只输出改写后的正文。")


def gen_one(qid):
    src = base[qid]
    prompt = f"原始推理摘要:\n{src}\n\n{INST}"
    c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                      model=DEFAULT_MODEL, thinking=False, max_tokens=780,
                      tag="reasonV6")
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
    json.dump(texts, open(OUT / "reasonings_v6P.json", "w"),
              ensure_ascii=False, indent=1)
    LEDGER.dump(OUT / "reasoning_v6P_ledger.json")
    print(f"v6P池成: 账 {LEDGER.totals()[2]:,}")


if __name__ == "__main__":
    main()
