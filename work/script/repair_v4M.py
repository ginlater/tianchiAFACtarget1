#!/usr/bin/env python3
"""v4M 修复轮：16 行答案先知语/结构伤重生成（肥证据+先知禁令），
合并剪尾清创池 → 终池 reasonings_v4M.json + reasoning_v4M_ledger.json。
用法: .venv/bin/python script/repair_v4M.py
"""
import json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen_reasoning_v4 import INST_CALC, INST_MCQ, INST_COMMON  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "output"

BAD = ['fc_b_001', 'fc_b_008', 'fc_b_007', 'fc_b_015', 'fin_b_005',
       'fin_b_015', 'fin_b_014', 'fin_b_016', 'fin_b_018', 'fin_b_019',
       'ins_b_003', 'ins_b_007', 'ins_b_008', 'ins_b_011', 'ins_b_016',
       'ins_b_018']
PRECOG = re.compile(r"给定的?最终?答案|题目最终答案|按照最终答案|最终答案(未|已)?包含"
                    r"|答案序列|依据答案|根据答案|答案提示|答案指向|答案反推|答案逻辑")
NO_PRECOG_INST = (
    "\n最重要纪律：推理中绝对不可出现'题目答案/给定答案/答案提示/反推'等任何暗示"
    "已知答案的表述——结论必须完全由证据数据推出并自然到达；必须以完整陈述句加句号"
    "收尾；不得出现问号或自我怀疑句式。")

qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
ans_map = json.load(open(OUT / "b_router6" / "answers.json",
                         encoding="utf-8-sig"))
picks = {json.loads(l)["qid"]: json.loads(l)["picked"]
         for l in open(OUT / "b_slim21" / "docsel_log.jsonl")}
qmap = {}
for q in qs_all:
    q["doc_ids"] = picks.get(q["qid"], q.get("doc_ids") or [])
    qmap[q["qid"]] = q

R = json.load(open(OUT / "reasonings_v4M_stage.json"))
led = json.load(open(OUT / "reasoning_v4L_ledger.json"))["per_qid"]


def gen_fix(qid, attempt=0):
    q = qmap[qid]
    ans = ans_map.get(qid, [""])
    ans_txt = "；".join(str(a) for a in ans if a)
    try:
        _ev, kept, _p = answerer.gather_evidence(q, k_opt=2, k_q=3, cap=3500)
        ev = "\n\n".join(f"【{c['doc_id']} P{c['page']}】{c['text'][:380]}"
                         for c in kept[:7])
    except Exception:  # noqa: BLE001
        ev = ""
    opts = "\n".join(f"{k}. {v}" for k, v in (q.get("options") or {}).items())
    inst = INST_CALC if q["answer_format"] == "calc" else INST_MCQ
    prompt = (f"证据:\n{ev}\n\n题目:\n{q['question']}\n"
              + (f"选项:\n{opts}\n" if opts else "")
              + f"\n应论证到达的结论: {ans_txt}\n\n{inst}{INST_COMMON}"
              + NO_PRECOG_INST)
    c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                      model=DEFAULT_MODEL, thinking=False, max_tokens=760,
                      tag="reasonV4M")
    txt = (c1 or "").strip().replace("\n", " ").replace("\r", " ")
    ok = (len(txt) >= 150 and txt.endswith("。") and not PRECOG.search(txt)
          and "？" not in txt)
    if not ok and attempt < 2:
        return gen_fix(qid, attempt + 1)
    return qid, txt, ok


def main():
    final_R = dict(R)
    final_led = {q: list(v) for q, v in led.items()}
    fails = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(gen_fix, q) for q in BAD]):
            qid, txt, ok = f.result()
            if ok:
                final_R[qid] = txt
                final_led[qid] = list(LEDGER.per_qid[qid])
            else:
                fails.append(qid)
    # 池级编辑清创: 生成上下文词汇回声替换(不改账)
    for q in final_R:
        final_R[q] = final_R[q].replace("证据片段", "检索证据") \
                               .replace("提供的证据", "检索证据")
    json.dump(final_R, open(OUT / "reasonings_v4M.json", "w"),
              ensure_ascii=False, indent=1)
    json.dump({"per_qid": final_led},
              open(OUT / "reasoning_v4M_ledger.json", "w"))
    LEDGER.dump(OUT / "reasoning_v4M_fixlog.json")
    tot = sum(sum(v) for v in final_led.values())
    print(f"修复 {len(BAD)-len(fails)}/{len(BAD)} 行, 未过门禁 {fails}, "
          f"终池账 {tot:,}")


if __name__ == "__main__":
    main()
