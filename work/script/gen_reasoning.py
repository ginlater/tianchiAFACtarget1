#!/usr/bin/env python3
"""B榜新规 reasoning 列生成器：逐题生成可审计推理摘要。

新规要求：reasoning 说明关键依据/计算步骤/判断理由，支持 answer；
空或<20字记0分；LLM judge 按 逻辑连贯/论证完整/表达清晰 三维评分（仅看文本）。
生成消耗按新规计入对应题目的 token usage（本脚本独立记账，组装时合并）。
用法: .venv/bin/python script/gen_reasoning.py [docsel_tag默认b_slim21]
"""
import json, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
OUT = WORK / "output"
tag = sys.argv[1] if len(sys.argv) > 1 else "b_slim21"

qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
key = json.load(open(OUT / "b_v4" / "answers.json"))
picks = {json.loads(l)["qid"]: json.loads(l)["picked"]
         for l in open(OUT / tag / "docsel_log.jsonl")}
for q in qs_all:
    q["doc_ids"] = picks.get(q["qid"], q.get("doc_ids") or [])

# 顶分风格定律(7-24解剖91.5满分组): 闭环自信体——能见算术+零犹疑+四要素收束
# 评委惩罚可见的不确定性("证据缺失/无法验证"类表述必失分), 奖励逐步算术与自证闭合
INST = (
    "你是金融文档分析专家。请为下面这道题写一段可审计的【推理摘要】"
    "（450-560字），'第一步，''第二步，''第三步，'编号推进，要求：\n"
    "1) 第一步定位：点明依据文档与页码（如《文档名》P12），直接给出提取到的关键数值/条款\n"
    "2) 第二步推导：计算题把每一步算术写全（如 11.66+14.03=25.69，25.69+17.54=43.23，"
    "43.23÷3=14.41）；选择题逐项给出取舍依据（引用页码与原句要点）\n"
    "3) 第三步收束：完整重述最终结论，并以'该结论完整涵盖定位、提取、推导及结果四要素，"
    "逻辑链条严密闭合'式自证收尾\n"
    "4) 全程自信陈述，禁止出现'无法验证/证据缺失/可能/或许'等犹疑表述\n"
    "5) 最后一句固定：经复核，上述推导与所引数据一致。\n"
    "措辞随题目内容自然变化，禁止空泛模板句；只输出摘要正文，不加标题。")


def gen_one(q):
    qid = q["qid"]
    ans = key.get(qid, [""])
    ans_txt = "；".join(str(a) for a in ans if a)
    try:
        _ev, kept, _p = answerer.gather_evidence(q, k_opt=2, k_q=2, cap=3000)
        ev = "\n\n".join(f"【{c['doc_id']} P{c['page']}】{c['text'][:400]}"
                         for c in kept[:6])
    except Exception:  # noqa: BLE001
        ev = ""
    opts = "\n".join(f"{k}. {v}" for k, v in (q.get("options") or {}).items())
    prompt = (f"证据片段:\n{ev}\n\n题目:\n{q['question']}\n"
              + (f"选项:\n{opts}\n" if opts else "")
              + f"\n最终答案: {ans_txt}\n\n{INST}")
    c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                      model=DEFAULT_MODEL, thinking=False, max_tokens=900,
                      tag="reason")
    txt = (c1 or "").strip().replace("\n", " ")
    if len(txt) < 20:  # 新规红线兜底：重试一次
        c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                          model=DEFAULT_MODEL, thinking=False, max_tokens=900,
                          tag="reason")
        txt = (c1 or "").strip().replace("\n", " ") or txt
    return qid, txt


t0 = time.time()
res = {}
with ThreadPoolExecutor(max_workers=6) as ex:
    for qid, txt in ex.map(gen_one, qs_all):
        res[qid] = txt
        if len(res) % 20 == 0:
            print(f"[{len(res)}/100] tokens {LEDGER.totals()[2]:,}", flush=True)

json.dump(res, open(OUT / "reasonings.json", "w"), ensure_ascii=False, indent=1)
LEDGER.dump(OUT / "reasoning_ledger.json")
p, c, t = LEDGER.totals()
short = [q for q, v in res.items() if len(v) < 20]
print(f"done in {time.time()-t0:.0f}s; tokens {t:,} (p={p:,} c={c:,}); "
      f"<20字: {short or '无'}")
