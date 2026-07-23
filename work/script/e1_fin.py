#!/usr/bin/env python3
"""E1试点：500k总攻架构的fin域验证。

设计（GLOBAL_ESCAPE的非线性动作🥇）：
- 满帽mega-context：4份年报事实卡(全脂)+20题一并入场, 联合作答
- 融合推理：每题输出 答案+推理摘要 同场生成(第四种风格, 零后置成本)
- 跨代复核：3.5-plus同context独立第二遍 + 分歧题定向仲裁
门槛: fin ≥16/20 且 ≤200k tokens → 通过则推广全卷(500k总攻)
"""
import json, pathlib, re, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "output"
tag = "b_e1fin"

qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
fins = [q for q in qs_all if q["qid"].startswith("fin")]
picks = {json.loads(l)["qid"]: json.loads(l)["picked"]
         for l in open(OUT / "b_slim21" / "docsel_log.jsonl")}
for q in fins:
    q["doc_ids"] = picks.get(q["qid"], [])
docs = sorted({d for q in fins for d in q["doc_ids"]})
print(f"[E1] fin 20题, 涉及{len(docs)}份年报: {docs}", flush=True)

# 1) 全脂事实卡（fresh, 计账）
cards = []
for d in docs:
    cards.append(answerer.build_digest(d, "financial_reports", qid="_e1card"))
print(f"[E1] 构卡完成 tokens {LEDGER.totals()[2]:,}", flush=True)

# 2) mega联合作答（融合推理）
schema = b_schema.load_schema(str(ROOT / "upload_b" / "submit.csv"))
qtexts = []
for i, q in enumerate(fins):
    kinds = b_schema.effective_kinds(q, schema.get(q["qid"], ["letter"]))
    opts = "\n".join(f"{k}. {v}" for k, v in (q.get("options") or {}).items())
    fmt = ("计算题,答案槽:" + "/".join(kinds)) if q["answer_format"] == "calc" \
        else q["answer_format"]
    qtexts.append(f"【{q['qid']}】({fmt}) {q['question']}"
                  + (f"\n{opts}" if opts else ""))
MEGA_INST = (
    "以上是全部相关年报的事实卡与20道题。请逐题独立作答。要求：\n"
    "1) 判断/选择题严格依据卡中数据逐项核验；计算题写出算式(全年分红=中期+末期两笔合计;"
    "口径词严格核算;数值主张必须现场重算)\n"
    "2) 每题输出两行：\n【qid】答案: <字母或数值(两位小数)或 排序;数值 格式>\n"
    "推理: <120-180字推理摘要：定位(卡内页码)→关键数值→算式/逐项判断→结论>\n"
    "3) 逐题输出, 一题不漏。")
base = "\n\n".join(cards) + "\n\n" + "\n\n".join(qtexts) + "\n\n" + MEGA_INST
c1, _r, _u = chat([{"role": "user", "content": base}], qid="_e1r1",
                  model=DEFAULT_MODEL, thinking=True, thinking_budget=6000,
                  max_tokens=9000, tag="e1_r1")
c2, _r2, _u2 = chat([{"role": "user", "content": base +
                      "\n（独立复核轮：从头独立作答，不要参考任何先前结论）"}],
                    qid="_e1r2", model="qwen3.5-plus", thinking=True,
                    thinking_budget=6000, max_tokens=9000, tag="e1_r2")
print(f"[E1] 双遍作答完成 tokens {LEDGER.totals()[2]:,}", flush=True)

BLOCK = re.compile(r"【(fin_b_\d+)】\s*答案[:：]\s*([^\n]+)\n+推理[:：]\s*([^【]+)", re.S)
def parse(content):
    out = {}
    for m in BLOCK.finditer(content or ""):
        out[m.group(1)] = (m.group(2).strip(), m.group(3).strip().replace("\n", " "))
    return out
a1, a2 = parse(c1), parse(c2)

# 3) 分歧仲裁（主模型）
answers, reasonings = {}, {}
for q in fins:
    qid = q["qid"]
    v1, v2 = a1.get(qid, ("", ""))[0], a2.get(qid, ("", ""))[0]
    if v1 and v2 and re.sub(r"[\s,，]", "", v1) != re.sub(r"[\s,，]", "", v2):
        adj = ("\n\n".join(cards) + f"\n\n题目:【{qid}】{q['question']}\n"
               + "\n".join(f"{k}. {v}" for k, v in (q.get("options") or {}).items())
               + f"\n\n两次独立作答分歧: 甲={v1} 乙={v2}\n"
               "请核对卡中数据仲裁，输出两行：\n答案: <最终>\n推理: <150字摘要>")
        c3, _r3, _u3 = chat([{"role": "user", "content": adj}], qid=qid,
                            model=DEFAULT_MODEL, thinking=True,
                            thinking_budget=3000, max_tokens=1200, tag="e1_arb")
        m = re.search(r"答案[:：]\s*([^\n]+)", c3 or "")
        mr = re.search(r"推理[:：]\s*(.+)", c3 or "", re.S)
        answers[qid] = (m.group(1).strip() if m else v1)
        reasonings[qid] = (mr.group(1).strip().replace("\n", " ") if mr
                           else a1.get(qid, ("", ""))[1])
    else:
        answers[qid] = v1 or v2
        reasonings[qid] = a1.get(qid, ("", ""))[1] or a2.get(qid, ("", ""))[1]

outdir = OUT / tag
outdir.mkdir(exist_ok=True)
# 规范化+落盘
final = {}
for q in fins:
    kinds = b_schema.effective_kinds(q, schema.get(q["qid"], ["letter"]))
    raw = answers.get(q["qid"], "")
    if q["answer_format"] == "calc":
        final[q["qid"]] = b_schema.split_answer(raw, kinds)
    else:
        final[q["qid"]] = [b_schema.fmt_slot(raw, "letter")]
json.dump(final, open(outdir / "answers.json", "w"), ensure_ascii=False, indent=1)
json.dump(reasonings, open(outdir / "reasonings.json", "w"), ensure_ascii=False,
          indent=1)
LEDGER.dump(outdir / "token_ledger.json")
p, c, t = LEDGER.totals()
print(f"[E1] done: tokens {t:,} (门槛≤200k)")
