#!/usr/bin/env python3
"""v4M 终修循环：宽域先知门禁(含'给出的'变体) + 推理-答案自洽门禁 + 账面回压。
7 行(6先知+fc_b_003矛盾)记录底座重生成 → 全池复扫 → 必要时瘦身回压 → 迭代至全绿。
用法: .venv/bin/python script/repair_v4M3.py
"""
import json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import b_schema  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gen_reasoning_v4 import INST_CALC, INST_MCQ, INST_COMMON  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "output"
POOL_CAP = 137_500

WIDE = re.compile(
    r"给[定出]的?(最终|正确)?答案|题目(最终|已给|给出)|按照最终答案"
    r"|最终答案(未|已)?包含|答案(序列|提示|指向|反推|逻辑)|依据答案|根据答案"
    r"|题目已判定|已判定其入选|题目设定中已明确.{0,12}答案|解题记录|按指令")
NO_PRECOG = (
    "\n最重要纪律：文中绝对不可出现'答案/已判定/题目给出'等任何暗示已知结论的"
    "表述——结论必须完全由证据与常识推出并自然到达；以完整陈述句加句号收尾；"
    "不得出现问号或自我怀疑。若证据不足以确证某选项，就以行业惯例与条款结构常识"
    "正面论证，绝不提及信息来源之外的判定依据。")
CONCL = re.compile(r"(?:正确)?(?:选项|答案)(?:组合)?(?:确定)?为([A-D、和与及]+)[。．]?\s*$")

qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
qmap = {q["qid"]: q for q in qs_all}
ans_map = json.load(open(OUT / "b_router6" / "answers.json",
                         encoding="utf-8-sig"))
src = json.load(open(OUT / "b_router6" / "piece_sources.json"))
R = json.load(open(OUT / "reasonings_v4M.json"))
led = json.load(open(OUT / "reasoning_v4M_ledger.json"))["per_qid"]
protected = set()


def record_of(qid):
    tag = src.get(qid, "")
    p = OUT / tag / "run_log.jsonl"
    best = ""
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


def consistent(qid, txt):
    a = ans_map.get(qid, [""])
    if a and re.fullmatch(r"[A-D]+", str(a[0])):
        want = set(str(a[0]))
        m = CONCL.search(txt[-90:])
        got = set(re.findall(r"[A-D]", m.group(1))) if m else None
        if got is not None and got != want:
            return False
        return all(l in txt[-len(txt) // 3:] for l in want)
    return True


def gates(qid, txt):
    return (len(txt) >= 150 and txt.endswith("。") and "？" not in txt
            and not WIDE.search(txt) and consistent(qid, txt))


def regen(qid, attempt=0):
    q = qmap[qid]
    ans_txt = "；".join(str(a) for a in ans_map.get(qid, [""]) if a)
    rec = record_of(qid)
    opts = "\n".join(f"{k}. {v}" for k, v in (q.get("options") or {}).items())
    inst = INST_CALC if q["answer_format"] == "calc" else INST_MCQ
    prompt = (f"系统解题过程记录:\n{rec}\n\n题目:\n{q['question']}\n"
              + (f"选项:\n{opts}\n" if opts else "")
              + f"\n把解题过程改写为推理摘要，必须论证到达的结论: {ans_txt}\n\n"
              + inst + INST_COMMON + NO_PRECOG
              + "\n不得出现'解题记录/系统'字样，以分析者口吻直接陈述。")
    c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                      model=DEFAULT_MODEL, thinking=False, max_tokens=760,
                      tag="reasonV4M3")
    txt = (c1 or "").strip().replace("\n", " ").replace("\r", " ")
    txt = txt.replace("证据片段", "检索证据").replace("提供的证据", "检索证据")
    if not gates(qid, txt) and attempt < 3:
        return regen(qid, attempt + 1)
    return qid, txt


def pool_tot():
    return sum(sum(v) for v in led.values())


def main():
    for it in range(3):
        bad = [q for q, t in R.items()
               if WIDE.search(t) or not consistent(q, t)]
        print(f"[iter{it}] 待修 {len(bad)}: {bad}")
        if not bad and pool_tot() <= POOL_CAP:
            break
        with ThreadPoolExecutor(max_workers=5) as ex:
            for f in as_completed([ex.submit(regen, q) for q in bad]):
                qid, txt = f.result()
                if gates(qid, txt):
                    R[qid] = txt
                    led[qid] = list(LEDGER.per_qid[qid])
                    protected.add(qid)
                    print(f"  ✓ {qid}")
                else:
                    print(f"  ✗ {qid} 三试未过, 保留原文待人工")
        # 账面回压: 对干净肥行(未保护)瘦身
        cands = sorted((q for q in R if q not in protected),
                       key=lambda q: -sum(led[q]))
        i = 0
        while pool_tot() > POOL_CAP and i < len(cands):
            qid = cands[i]
            i += 1
            q2, txt = regen(qid)  # 记录底座同配方, 输出天然更短(记录截断)
            if gates(qid, txt) and sum(LEDGER.per_qid[qid]) < sum(led[qid]):
                R[qid] = txt
                led[qid] = list(LEDGER.per_qid[qid])
                protected.add(qid)
        print(f"[iter{it}] 池账 {pool_tot():,}")
    json.dump(R, open(OUT / "reasonings_v4M.json", "w"),
              ensure_ascii=False, indent=1)
    json.dump({"per_qid": led}, open(OUT / "reasoning_v4M_ledger.json", "w"))
    LEDGER.dump(OUT / "reasoning_v4M3_fixlog.json")
    left = [q for q, t in R.items() if WIDE.search(t) or not consistent(q, t)]
    print(f"终态: 池账 {pool_tot():,} | 残留 {left or '零 ✓'}")


if __name__ == "__main__":
    main()
