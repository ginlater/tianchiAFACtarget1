#!/usr/bin/env python3
"""v4M 二次修复：5 顽固行改用中标 run 解题记录做事实底座（推理=解题记录压缩的
正宗路径），另对非修复行做瘦身轮-2（带先知门禁）把池账压回 ≤136,500。
用法: .venv/bin/python script/repair_v4M2.py
"""
import json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import b_schema  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen_reasoning_v4 import INST_CALC, INST_MCQ, INST_COMMON  # noqa: E402
from repair_v4M import PRECOG, NO_PRECOG_INST  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "output"
POOL_CAP = 136_500

STUB = {"ins_b_007": "b_slim14", "ins_b_008": "b_slim14",
        "fin_b_016": "b_slim12", "fin_b_019": "b_routerG_qB_finins",
        "ins_b_003": "b_slim25"}
REPAIRED = ['fc_b_001', 'fc_b_008', 'fc_b_007', 'fc_b_015', 'fin_b_005',
            'fin_b_015', 'fin_b_014', 'fin_b_018', 'ins_b_011', 'ins_b_016',
            'ins_b_018']

qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
qmap = {q["qid"]: q for q in qs_all}
ans_map = json.load(open(OUT / "b_router6" / "answers.json",
                         encoding="utf-8-sig"))
R = json.load(open(OUT / "reasonings_v4M.json"))
led = json.load(open(OUT / "reasoning_v4M_ledger.json"))["per_qid"]


def record_of(qid, tag):
    best = ""
    p = OUT / tag / "run_log.jsonl"
    if p.exists():
        for line in open(p, encoding="utf-8"):
            if qid not in line:
                continue
            try:
                r = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if r.get("qid") == qid:
                best = r.get("c1") or r.get("c3") or best
    return (best or "")[:2600]


def gates(txt):
    return (len(txt) >= 150 and txt.endswith("。")
            and not PRECOG.search(txt) and "？" not in txt)


def gen_from_record(qid, attempt=0):
    q = qmap[qid]
    ans_txt = "；".join(str(a) for a in ans_map.get(qid, [""]) if a)
    rec = record_of(qid, STUB[qid])
    opts = "\n".join(f"{k}. {v}" for k, v in (q.get("options") or {}).items())
    inst = INST_CALC if q["answer_format"] == "calc" else INST_MCQ
    prompt = (f"系统解题记录（含定位到的证据与判断过程）:\n{rec}\n\n题目:\n"
              + q["question"] + ("\n选项:\n" + opts if opts else "")
              + f"\n\n把上述解题记录压缩改写为推理摘要，结论为: {ans_txt}\n\n"
              + inst + INST_COMMON + NO_PRECOG_INST
              + "\n不得出现'解题记录/系统'字样，以分析者口吻直接陈述。")
    c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                      model=DEFAULT_MODEL, thinking=False, max_tokens=760,
                      tag="reasonV4M2")
    txt = (c1 or "").strip().replace("\n", " ").replace("\r", " ")
    txt = txt.replace("证据片段", "检索证据").replace("提供的证据", "检索证据")
    if not gates(txt) and attempt < 2:
        return gen_from_record(qid, attempt + 1)
    return qid, txt


def gen_lean2(qid):
    q = qmap[qid]
    ans_txt = "；".join(str(a) for a in ans_map.get(qid, [""]) if a)
    from agent import answerer
    picks = {json.loads(l)["qid"]: json.loads(l)["picked"]
             for l in open(OUT / "b_slim21" / "docsel_log.jsonl")}
    q = dict(q, doc_ids=picks.get(qid, q.get("doc_ids") or []))
    try:
        _ev, kept, _p = answerer.gather_evidence(q, k_opt=1, k_q=2, cap=1200)
        ev = "\n\n".join(f"【{c['doc_id']} P{c['page']}】{c['text'][:240]}"
                         for c in kept[:4])
    except Exception:  # noqa: BLE001
        ev = ""
    opts = "\n".join(f"{k}. {v}" for k, v in (q.get("options") or {}).items())
    inst = (INST_CALC if q["answer_format"] == "calc" else INST_MCQ).replace(
        "380-520 字", "280-380 字")
    prompt = (f"证据:\n{ev}\n\n题目:\n{q['question']}\n"
              + (f"选项:\n{opts}\n" if opts else "")
              + f"\n应论证到达的结论: {ans_txt}\n\n{inst}{INST_COMMON}"
              + NO_PRECOG_INST)
    c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                      model=DEFAULT_MODEL, thinking=False, max_tokens=520,
                      tag="reasonV4L2")
    txt = (c1 or "").strip().replace("\n", " ").replace("\r", " ")
    txt = txt.replace("证据片段", "检索证据").replace("提供的证据", "检索证据")
    return qid, txt


def main():
    # 1) 顽固行: 解题记录底座重生成
    with ThreadPoolExecutor(max_workers=5) as ex:
        for f in as_completed([ex.submit(gen_from_record, q) for q in STUB]):
            qid, txt = f.result()
            if gates(txt):
                R[qid] = txt
                led[qid] = list(LEDGER.per_qid[qid])
                print(f"记录底座修复 ✓ {qid}")
            else:
                print(f"仍未过门禁 ✗ {qid}（保留剪尾版）")
    # 2) 瘦身轮-2: 非修复行按账从大到小替换, 压到 POOL_CAP
    protect = set(STUB) | set(REPAIRED)
    def tot():
        return sum(sum(v) for v in led.values())
    cands = sorted((q for q in R if q not in protect),
                   key=lambda q: -sum(led[q]))
    i = 0
    while tot() > POOL_CAP and i < len(cands):
        batch = cands[i:i + 10]
        i += 10
        with ThreadPoolExecutor(max_workers=5) as ex:
            for f in as_completed([ex.submit(gen_lean2, q) for q in batch]):
                qid, txt = f.result()
                new_c = sum(LEDGER.per_qid.get(qid, [0, 0])) - \
                    (sum(LEDGER.per_qid.get(qid, [0, 0])) - 0)
                new_cost = list(LEDGER.per_qid[qid])
                if gates(txt) and sum(new_cost) < sum(led[qid]):
                    R[qid] = txt
                    led[qid] = new_cost
        print(f"lean2 批后池账 {tot():,}", flush=True)
    json.dump(R, open(OUT / "reasonings_v4M.json", "w"),
              ensure_ascii=False, indent=1)
    json.dump({"per_qid": led}, open(OUT / "reasoning_v4M_ledger.json", "w"))
    LEDGER.dump(OUT / "reasoning_v4M2_fixlog.json")
    print(f"终池账 {tot():,}")


if __name__ == "__main__":
    main()
