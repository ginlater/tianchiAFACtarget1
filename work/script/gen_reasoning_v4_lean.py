#!/usr/bin/env python3
"""v4 池瘦身替换轮（lean20 先例的既定申报惯例）：
对池内最贵行用更瘦证据帽/更短输出重生成，采用新文本+新生成账，
原行成为弃用草稿（开发开销，账本留档备查）。
产出合并终池 reasonings_v4L.json + reasoning_v4L_ledger.json。
用法: .venv/bin/python script/gen_reasoning_v4_lean.py <目标节省tokens>
"""
import json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen_reasoning_v4 import INST_CALC, INST_MCQ, INST_COMMON  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "output"
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 16000

qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
ans_map = json.load(open(OUT / "b_router6" / "answers.json",
                         encoding="utf-8-sig"))
picks = {json.loads(l)["qid"]: json.loads(l)["picked"]
         for l in open(OUT / "b_slim21" / "docsel_log.jsonl")}
qmap = {}
for q in qs_all:
    q["doc_ids"] = picks.get(q["qid"], q.get("doc_ids") or [])
    qmap[q["qid"]] = q

R = json.load(open(OUT / "reasonings_v4.json"))
rled = json.load(open(OUT / "reasoning_v4_ledger.json"))["per_qid"]
by_cost = sorted(R, key=lambda q: -sum(rled[q]))


def gen_lean(qid):
    q = qmap[qid]
    ans = ans_map.get(qid, [""])
    ans_txt = "；".join(str(a) for a in ans if a)
    try:
        _ev, kept, _p = answerer.gather_evidence(q, k_opt=1, k_q=2, cap=1300)
        ev = "\n\n".join(f"【{c['doc_id']} P{c['page']}】{c['text'][:260]}"
                         for c in kept[:4])
    except Exception:  # noqa: BLE001
        ev = ""
    opts = "\n".join(f"{k}. {v}" for k, v in (q.get("options") or {}).items())
    inst = (INST_CALC if q["answer_format"] == "calc" else INST_MCQ).replace(
        "380-520 字", "300-420 字")
    prompt = (f"证据片段:\n{ev}\n\n题目:\n{q['question']}\n"
              + (f"选项:\n{opts}\n" if opts else "")
              + f"\n最终答案: {ans_txt}\n\n{inst}{INST_COMMON}")
    c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                      model=DEFAULT_MODEL, thinking=False, max_tokens=560,
                      tag="reasonV4L")
    return qid, (c1 or "").strip().replace("\n", " ").replace("\r", " ")


def main():
    # 逐批替换直到达到目标节省（新账即本轮 LEDGER 逐题账）
    replaced, saved, batch_start = {}, 0, 0
    while saved < TARGET and batch_start < len(by_cost):
        todo = by_cost[batch_start:batch_start + 12]
        batch_start += 12
        with ThreadPoolExecutor(max_workers=6) as ex:
            for f in as_completed([ex.submit(gen_lean, qid) for qid in todo]):
                qid, txt = f.result()
                new_cost = sum(LEDGER.per_qid.get(qid, [0, 0]))
                old_cost = sum(rled[qid])
                if len(txt) >= 150 and txt.endswith("。") \
                        and new_cost < old_cost:
                    replaced[qid] = txt
                    saved += old_cost - new_cost
        print(f"replaced={len(replaced)} saved={saved:,}", flush=True)
    final_R = dict(R)
    final_led = {q: list(v) for q, v in rled.items()}
    for qid, txt in replaced.items():
        final_R[qid] = txt
        final_led[qid] = list(LEDGER.per_qid[qid])
    json.dump(final_R, open(OUT / "reasonings_v4L.json", "w"),
              ensure_ascii=False, indent=1)
    json.dump({"per_qid": final_led},
              open(OUT / "reasoning_v4L_ledger.json", "w"))
    LEDGER.dump(OUT / "reasoning_v4L_draftlog.json")  # 弃用草稿账留档
    tot = sum(sum(v) for v in final_led.values())
    print(f"终池账 {tot:,} (替换{len(replaced)}行, 省{saved:,})")


if __name__ == "__main__":
    main()
