#!/usr/bin/env python3
"""动态记忆压缩试点（赛题主题的字面兑现）：压缩一次、采样多次。

stage1: 宽证据(12k) → 题目条件化记忆包(≤700字, 含每选项相关数值/条款+页码)
stage2: 小包上3独立票(3.6×2+3.7×1跨代) → 多数决; 全歧→3.5仲裁
对照: 这些题在最优静态配置下的命中率(routerZ_plan)。
"""
import json, pathlib, re, sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema  # noqa: E402
from agent.qwen_client import chat  # noqa: E402

WORK = pathlib.Path(__file__).resolve().parents[1]
import sys as _sys
HARD = open(_sys.argv[1]).read().strip().split(",") if len(_sys.argv) > 1 else []
TAG = _sys.argv[2] if len(_sys.argv) > 2 else "b_dynheavy"
key = json.load(open(WORK / "output" / "b_v4" / "answers.json"))
qs = {q["qid"]: q for q in b_schema.load_questions(
    str(WORK.parent / "upload_b" / "question_b"))}
picks = {}
for ln in open(WORK / "output" / "b_router8_heavy" / "docsel_log.jsonl"):
    d = json.loads(ln)
    picks[d["qid"]] = d.get("picked", [])
if not picks:
    raise SystemExit("no docsel")

ANS = re.compile(r"答案[:：]\s*([A-D]+)")
res = {}
for qid in HARD:
    q = dict(qs[qid])
    q["doc_ids"] = picks.get(qid) or q.get("doc_ids") or []
    ev, kept, _p, _d = answerer.evidence_block(q)
    opts = "\n".join(f"{k}. {v}" for k, v in (q.get("options") or {}).items())
    # stage1 压缩: 题目条件化记忆包
    c1, _t, _u = chat([{"role": "user", "content":
        f"{ev[:26000]}\n\n题目：{q['question']}\n{opts}\n\n"
        "请把与本题四个选项相关的全部证据压缩成【记忆包】(≤800字)："
        "每个选项单列一行——该选项涉及的原文数值/条款原句要点+页码；"
        "若某选项在证据中找不到对应内容, 该行写'选项X: 证据中未见对应表述(检索范围: 各文档相关章节)'；"
        "不判断对错, 只忠实压缩。只输出记忆包。"}],
        qid=qid, model="qwen3.6-plus", thinking=False, max_tokens=1100,
        tag="dyn_c")
    pack = (c1 or "").strip()
    # 有损结构不垄断信道: 附瘦原文回退(每选项top1保护块)
    fallback = "\n".join(f"[{c['doc_id']} P{c['page']}]{c['text'][:220]}"
                          for c in kept[:6])
    pack = pack + "\n\n原文回退片段:\n" + fallback
    # stage2 三票跨代
    votes = []
    for mdl in ("qwen3.6-plus", "qwen3.6-plus", "qwen3.7-plus"):
        c2, _t, _u = chat([{"role": "user", "content":
            f"记忆包：\n{pack}\n\n题目：{q['question']}\n{opts}\n\n"
            "严格依据记忆包逐项判断每个选项(证据不足以支持的选项不选)。"
            "最后一行输出 答案:<字母>"}],
            qid=qid, model=mdl, thinking=True, thinking_budget=1200,
            max_tokens=1600, tag="dyn_v")
        m = ANS.search(c2 or "")
        if m:
            votes.append("".join(sorted(set(m.group(1)))))
    tally = Counter(votes).most_common()
    final = tally[0][0] if tally and tally[0][1] >= 2 else None
    if final is None and votes:
        c3, _t, _u = chat([{"role": "user", "content":
            f"记忆包：\n{pack}\n\n题目：{q['question']}\n{opts}\n\n"
            f"三次独立判断分歧: {votes}。请仲裁, 最后一行输出 答案:<字母>"}],
            qid=qid, model="qwen3.5-plus", thinking=True, thinking_budget=1600,
            max_tokens=1600, tag="dyn_a")
        m = ANS.search(c3 or "")
        final = "".join(sorted(set(m.group(1)))) if m else votes[0]
    ok = final == (key[qid][0] if isinstance(key[qid], list) else key[qid])
    res[qid] = {"votes": votes, "final": final, "key": key[qid], "ok": ok}
    print(f"{qid}: 票{votes} → {final} 键{key[qid]} {'✓' if ok else '✗'}",
          flush=True)
from agent.qwen_client import LEDGER
outdir = WORK / "output" / TAG
outdir.mkdir(exist_ok=True)
json.dump({q: [r["final"]] for q, r in res.items()},
          open(outdir / "answers.json", "w"), ensure_ascii=False, indent=1)
LEDGER.dump(outdir / "token_ledger.json")
n = sum(1 for r in res.values() if r["ok"])
print(f"\n动态压缩层: {n}/{len(HARD)}")
