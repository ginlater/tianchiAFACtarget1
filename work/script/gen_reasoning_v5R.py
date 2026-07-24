#!/usr/bin/env python3
"""R探针推理池 v5（审计体）：第二哲学备弹（依据→核对→适用→结论，非第一人称）。

与现役 probe 池的三点哲学差异：
1) 按题型变结构（计算=取数→算式→复算；选择=争点→逐项判定引页；日期=规则→推算），
   不再全池统一"第一步/第二步/第三步"脚手架
2) 零固定句——现役池每题以同一句"经复核…"收尾，属可检测模板指纹（新规禁模板化）；
   本池要求开头/收尾随题自然变化
3) 显式论证连接词（因为/故/由此可得）+ 每个关键断言挂页码，验证动作措辞逐题不同

生成消耗独立记账（reasoning_v5R_ledger.json），组装时并入对应题行。
用法: .venv/bin/python script/gen_reasoning_v4.py
"""
import json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
OUT = WORK / "output"

qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
ans_map = json.load(open(OUT / "b_router6" / "answers.json",
                         encoding="utf-8-sig"))
picks = {json.loads(l)["qid"]: json.loads(l)["picked"]
         for l in open(OUT / "b_slim21" / "docsel_log.jsonl")}
for q in qs_all:
    q["doc_ids"] = picks.get(q["qid"], q.get("doc_ids") or [])

INST_CALC = (
    "写一段 380-520 字的推理摘要，审计工作底稿体（非第一人称，用'经查/经核/据此'类"
    "客观措辞）：【依据】列明数据来源（文档名+页码+报表项目口径）与提取值；"
    "【核对】写完整算式链（逐步算术）并做一项独立复核（口径核对/量级校验/交叉表比对，"
    "真实写出核对内容）；【结论】据此确认最终结果。段内自然衔接，不用方括号标题原文，"
    "用行文承接词替代。")
INST_MCQ = (
    "写一段 380-520 字的推理摘要，审计工作底稿体（非第一人称，用'经查/经核/据此'类"
    "客观措辞）：先陈述审查事项，随后逐项核验各选项——每项给出文档页码依据与"
    "符合/不符的判定理由，最后据核验结果确认答案字母组合。段内自然衔接。")
INST_COMMON = (
    "\n通用要求：全程自信陈述，不出现'可能/或许/无法验证/证据缺失'等犹疑词；"
    "开头与收尾的措辞必须随题目内容自然变化，禁止任何固定套话或模板句式；"
    "只输出摘要正文，不加标题、不用列表符号。")


def gen_one(q):
    qid = q["qid"]
    ans = ans_map.get(qid, [""])
    ans_txt = "；".join(str(a) for a in ans if a)
    try:
        _ev, kept, _p = answerer.gather_evidence(q, k_opt=2, k_q=2, cap=2200)
        ev = "\n\n".join(f"【{c['doc_id']} P{c['page']}】{c['text'][:330]}"
                         for c in kept[:5])
    except Exception:  # noqa: BLE001
        ev = ""
    opts = "\n".join(f"{k}. {v}" for k, v in (q.get("options") or {}).items())
    inst = INST_CALC if q["answer_format"] == "calc" else INST_MCQ
    prompt = (f"证据片段:\n{ev}\n\n题目:\n{q['question']}\n"
              + (f"选项:\n{opts}\n" if opts else "")
              + f"\n最终答案: {ans_txt}\n\n{inst}{INST_COMMON}")
    c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                      model=DEFAULT_MODEL, thinking=False, max_tokens=760,
                      tag="reasonV5")
    return qid, (c1 or "").strip().replace("\n", " ").replace("\r", " ")


def main():
    texts = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(gen_one, q) for q in qs_all]
        for n, f in enumerate(as_completed(futs)):
            qid, txt = f.result()
            texts[qid] = txt
            if (n + 1) % 20 == 0:
                print(f"[{n+1}/100] tokens={LEDGER.totals()[2]:,}", flush=True)
    json.dump(texts, open(OUT / "reasonings_v5R.json", "w"),
              ensure_ascii=False, indent=1)
    LEDGER.dump(OUT / "reasoning_v5R_ledger.json")
    p, c, t = LEDGER.totals()
    print(f"池成: 100题 生成账 {t:,} (p={p:,} c={c:,})")


if __name__ == "__main__":
    main()
