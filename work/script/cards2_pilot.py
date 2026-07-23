#!/usr/bin/env python3
"""记忆压缩二代试点：选块器驱动的极小高保真卡 + 域级批量答题。

架构（赛题"动态记忆压缩"的终极形态原型）：
  1) 选块（零API）：全部题目的检索需求（题干/选项/实体反查/强制词表）在每份文档上
     并集出"关键块清单"——20轮迭代沉淀的选块器就是配方表
  2) 压卡（一文档一调用）：关键块 → ≤900字事实卡，数字/日期/公式/免责列举原样保留
  3) 答题（域级批）：选择题5题一批共享该批文档卡；计算题单答（卡+题目）
试点措施：docsel 复用 b_slim21 的选择结果（正式跑会重做，成本已知 ~32k）。
"""
import json, pathlib, sys, time
from collections import defaultdict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema, batch, calc  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
OUT = WORK / "output" / "b_cards2"
OUT.mkdir(parents=True, exist_ok=True)

t0 = time.time()
qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
schema = b_schema.load_schema(str(ROOT / "upload_b" / "submit.csv"))
picks = {json.loads(l)["qid"]: json.loads(l)["picked"]
         for l in open(WORK / "output" / "b_slim21" / "docsel_log.jsonl")}
for q in qs_all:
    q["doc_ids"] = picks.get(q["qid"], q.get("doc_ids") or [])

# ---- 1) 选块（纯词法/BM25，零API） ----
doc_blocks = defaultdict(dict)
for q in qs_all:
    if not q["doc_ids"]:
        continue
    _ev, kept, _prot = answerer.gather_evidence(q, k_opt=2, k_q=2, cap=4200)
    for c in kept:
        doc_blocks[c["doc_id"]][c["id"]] = c
print(f"选块完成: {len(doc_blocks)}份文档, "
      f"平均{sum(len(v) for v in doc_blocks.values())/max(1,len(doc_blocks)):.1f}块/档",
      flush=True)

# ---- 2) 压卡 ----
cards = {}
for d, blocks in sorted(doc_blocks.items()):
    bl = sorted(blocks.values(),
                key=lambda c: (c["page"] or 0, int(c["id"].split("#c")[1])))
    text = "\n\n".join(f"[P{c['page']}]{c['text'][:600]}" for c in bl)[:15000]
    prompt = (f"以下是《{answerer._doc_title(d)}》中回答一批金融考题所需的关键原文块。"
              "请压缩为一张【事实卡】（900字以内）：\n"
              "- 全部数字/日期/比例/公式/期限原样保留并注页码(P字样)\n"
              "- 免责/例外/分档类条款逐项列举，禁止用'等'字概括\n"
              "- 主体名(公司/产品)写全称\n- 与数字和条款无关的叙述删除\n\n" + text)
    c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=f"_card_{d}",
                      model=DEFAULT_MODEL, thinking=False, max_tokens=1100,
                      tag="card2")
    cards[d] = c1.strip()
json.dump(cards, open(OUT / "cards.json", "w"), ensure_ascii=False, indent=1)
p_, c_, t_ = LEDGER.totals()
print(f"压卡完成: {len(cards)}张, 累计tokens {t_:,}", flush=True)

# ---- 3) 答题 ----
log = open(OUT / "run_log.jsonl", "w")
results = {}
choice = [q for q in qs_all if q["answer_format"] != "calc"]
calcs = [q for q in qs_all if q["answer_format"] == "calc"]

by_dom = defaultdict(list)
for q in choice:
    by_dom[q["domain"]].append(q)
for dom, qs in by_dom.items():
    qs.sort(key=lambda q: sorted(q["doc_ids"]))
    for i in range(0, len(qs), 5):
        grp = qs[i:i + 5]
        docs = list(dict.fromkeys(d for q in grp for d in q["doc_ids"]))
        ev = "\n\n".join(f"【{d}卡】{cards.get(d, '(无卡)')}" for d in docs)
        qtexts = "\n\n".join(f"[第{j+1}题 {q['qid']}]\n{answerer._q_text(q)}"
                             for j, q in enumerate(grp))
        inst = batch.BATCH_INST.format(n=len(grp))
        c1, _r, u = chat([{"role": "user", "content": ev + "\n\n" + qtexts +
                           "\n\n" + inst}],
                         qid=grp[0]["qid"], model=DEFAULT_MODEL,
                         thinking=answerer._think(grp[0]),
                         thinking_budget=1600, max_tokens=1400 * len(grp),
                         tag="mega")
        got = batch._parse_batch(c1, grp)
        for q in grp:
            a = got.get(q["qid"], "")
            results[q["qid"]] = [b_schema.fmt_slot(a or "A", "letter")]
            log.write(json.dumps({"qid": q["qid"], "final": a,
                                  "mega": [x["qid"] for x in grp]},
                                 ensure_ascii=False) + "\n")
        print(f"[{len(results)}/100] {dom} +{len(grp)} "
              f"({LEDGER.totals()[2]:,} tok)", flush=True)

for q in calcs:
    kinds = b_schema.effective_kinds(q, schema.get(q["qid"], ["number"]))
    docs = q["doc_ids"]
    ev = "\n\n".join(f"【{d}卡】{cards.get(d, '(无卡)')}" for d in docs)
    inst = calc.CALC_INST.format(n=len(kinds), slots=calc._slots_text(kinds),
                                 template=calc._template(kinds))
    from agent.b_schema import is_date_question
    if is_date_question(q) and any(k == "number" for k in kinds):
        inst += "\n注意：本题答案是一个日期，最后一行输出完整中文日期。"
    # 计算题证据 = 卡 + 定向原文块（数字精度不能只靠卡）
    _ev2, kept, _p = answerer.gather_evidence(q, k_opt=3, k_q=3, cap=3800)
    raw_ev = "\n\n".join(f"【{c['doc_id']} P{c['page']}】{c['text']}"
                         for c in kept)
    c1, _r, _u = chat([{"role": "user", "content": ev + "\n\n原文片段:\n" +
                        raw_ev + "\n\n题目:\n" + q["question"] + "\n\n" + inst}],
                      qid=q["qid"], model=DEFAULT_MODEL, thinking=True,
                      thinking_budget=1600, max_tokens=2600, tag="mcalc")
    a1 = calc.parse_calc(c1)
    results[q["qid"]] = b_schema.split_answer(a1, kinds)
    log.write(json.dumps({"qid": q["qid"], "final": a1},
                         ensure_ascii=False) + "\n")
    print(f"[{len(results)}/100] calc {q['qid']} ({LEDGER.totals()[2]:,} tok)",
          flush=True)

log.close()
json.dump(results, open(OUT / "answers.json", "w"), ensure_ascii=False,
          indent=1)
LEDGER.dump(OUT / "token_ledger.json")
order = [q["qid"] for q in qs_all]
b_schema.write_submission(OUT / "answer.csv", results, schema, order,
                          LEDGER.per_qid, LEDGER.totals())
p, c, t = LEDGER.totals()
print(f"done in {time.time()-t0:.0f}s; tokens {t:,} (p={p:,} c={c:,})")
