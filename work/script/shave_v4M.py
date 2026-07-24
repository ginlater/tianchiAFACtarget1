#!/usr/bin/env python3
"""v4M 池瘦身回压终轮：肥行从大到小 lean 重生成，仅当 全门禁绿 且 更便宜 才采用
（重试全额入账的保守口径不变）。压到 POOL_CAP 即停。
用法: .venv/bin/python script/shave_v4M.py [POOL_CAP=137400]
"""
import json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen_reasoning_v4 import INST_CALC, INST_MCQ, INST_COMMON  # noqa: E402
from repair_v4M import NO_PRECOG_INST  # noqa: E402
from repair_v4M3 import WIDE, consistent  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "output"
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 137_400

qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
picks = {json.loads(l)["qid"]: json.loads(l)["picked"]
         for l in open(OUT / "b_slim21" / "docsel_log.jsonl")}
qmap = {}
for q in qs_all:
    q["doc_ids"] = picks.get(q["qid"], q.get("doc_ids") or [])
    qmap[q["qid"]] = q
ans_map = json.load(open(OUT / "b_router6" / "answers.json",
                         encoding="utf-8-sig"))
R = json.load(open(OUT / "reasonings_v4M.json"))
led = json.load(open(OUT / "reasoning_v4M_ledger.json"))["per_qid"]


def gates(qid, txt):
    return (len(txt) >= 150 and txt.endswith("。") and "？" not in txt
            and not WIDE.search(txt) and consistent(qid, txt))


def gen_lean(qid):
    q = qmap[qid]
    ans_txt = "；".join(str(a) for a in ans_map.get(qid, [""]) if a)
    try:
        _ev, kept, _p = answerer.gather_evidence(q, k_opt=1, k_q=2, cap=1500)
        ev = "\n\n".join(f"【{c['doc_id']} P{c['page']}】{c['text'][:260]}"
                         for c in kept[:4])
    except Exception:  # noqa: BLE001
        ev = ""
    opts = "\n".join(f"{k}. {v}" for k, v in (q.get("options") or {}).items())
    inst = (INST_CALC if q["answer_format"] == "calc" else INST_MCQ).replace(
        "380-520 字", "300-400 字")
    prompt = (f"证据:\n{ev}\n\n题目:\n{q['question']}\n"
              + (f"选项:\n{opts}\n" if opts else "")
              + f"\n应论证到达的结论: {ans_txt}\n\n{inst}{INST_COMMON}"
              + NO_PRECOG_INST)
    c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                      model=DEFAULT_MODEL, thinking=False, max_tokens=540,
                      tag="reasonShave")
    txt = (c1 or "").strip().replace("\n", " ").replace("\r", " ")
    return qid, txt.replace("证据片段", "检索证据").replace("提供的证据", "检索证据")


def tot():
    return sum(sum(v) for v in led.values())


def main():
    cands = sorted(R, key=lambda q: -sum(led[q]))
    i, adopted = 0, 0
    while tot() > CAP and i < len(cands):
        batch = cands[i:i + 8]
        i += 8
        with ThreadPoolExecutor(max_workers=6) as ex:
            for f in as_completed([ex.submit(gen_lean, q) for q in batch]):
                qid, txt = f.result()
                cost = list(LEDGER.per_qid.get(qid, [0, 0]))
                if gates(qid, txt) and sum(cost) < sum(led[qid]):
                    R[qid] = txt
                    led[qid] = cost
                    adopted += 1
        print(f"批后池账 {tot():,} (已采用{adopted})", flush=True)
    json.dump(R, open(OUT / "reasonings_v4M.json", "w"),
              ensure_ascii=False, indent=1)
    json.dump({"per_qid": led}, open(OUT / "reasoning_v4M_ledger.json", "w"))
    LEDGER.dump(OUT / "reasoning_v4M_shavelog.json")
    left = [q for q, t in R.items() if WIDE.search(t) or not consistent(q, t)]
    print(f"终态池账 {tot():,} | 残留 {left or '零 ✓'}")


if __name__ == "__main__":
    main()
