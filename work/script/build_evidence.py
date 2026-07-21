#!/usr/bin/env python3
"""从 run_log.jsonl 生成提交要求的 evidence.json。

格式对齐赛规样例: [{qid, answer, evidence_retrieval:[{doc_id, quoted_clause, reasoning}]}]
用法: python work/script/build_evidence.py work/output/<tag>
"""
import json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work"))
from agent import retrieval  # noqa: E402


def analysis_lines(c1):
    """从模型分析文本中抽取每选项一行的理由。"""
    out = {}
    for line in (c1 or "").splitlines():
        m = re.match(r"^[分析复核仲裁]*[:：]?\s*\**([A-D])[.、::：]?\s*(.{10,})", line.strip())
        if m and m.group(1) not in out:
            out[m.group(1)] = m.group(2)[:160]
    return out


def main(outdir):
    outdir = pathlib.Path(outdir)
    answers = json.load(open(outdir / "answers.json"))
    logs = {}
    with open(outdir / "run_log.jsonl") as f:
        for line in f:
            r = json.loads(line)
            logs[r["qid"]] = r  # 后写覆盖先写（保留最新一次）

    chunk_cache = {}

    def chunk_text(cid):
        doc_id = cid.split("#")[0]
        if doc_id not in chunk_cache:
            chunk_cache[doc_id] = {c["id"]: c for c in retrieval.chunk_doc(doc_id)}
        return chunk_cache[doc_id].get(cid)

    ev_out = []
    for qid, ans in sorted(answers.items()):
        r = logs.get(qid, {})
        reasons = analysis_lines(r.get("c1", ""))
        picked_letters = [c for c in ans if c in "ABCD"]
        items = []
        seen_docs = set()
        for cid in r.get("evidence_ids", [])[:6]:
            c = chunk_text(cid)
            if not c:
                continue
            if c["doc_id"] in seen_docs and len(items) >= 3:
                continue
            seen_docs.add(c["doc_id"])
            items.append({
                "doc_id": c["doc_id"],
                "quoted_clause": c["text"][:220],
                "reasoning": reasons.get(picked_letters[0] if picked_letters else "A",
                                          "依据该片段与题目选项逐项比对得出答案。"),
            })
            if len(items) >= 4:
                break
        ev_out.append({"qid": qid, "answer": ans, "evidence_retrieval": items})
    json.dump(ev_out, open(outdir / "evidence.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"evidence.json written: {len(ev_out)} entries -> {outdir}")


if __name__ == "__main__":
    main(sys.argv[1])
