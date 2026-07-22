#!/usr/bin/env python3
"""从一次B运行的日志中提取低置信题目清单（仅用运行内部信号，自包含合规）。

信号：①选择题 r1≠r2（曾仲裁）②计算题双样本分歧 ③计算题单源(另一样本弃答)
     ④盲检曾域级扩检 ⑤答案槽有空值
用法: python script/flag_low_confidence.py work/output/<tag>
"""
import json, pathlib, re, sys

def main(outdir):
    outdir = pathlib.Path(outdir)
    norm = lambda s: re.sub(r"[\s,，]", "", s or "")
    flags = {}

    def add(qid, reason):
        flags.setdefault(qid, []).append(reason)

    for line in open(outdir / "run_log.jsonl"):
        r = json.loads(line)
        qid = r.get("qid")
        if "doc_expanded" in r:
            add(qid, f"盲检扩检+{r['doc_expanded']}")
            continue
        if "a1" in r:  # 计算题
            if r.get("a1") and r.get("a2") and norm(r["a1"]) != norm(r["a2"]):
                add(qid, f"计算分歧 a1={r['a1'][:20]} a2={r['a2'][:20]}")
            if not r.get("a2") or "补充" in (r.get("a2") or ""):
                add(qid, "计算单源")
            if not r.get("final"):
                add(qid, "计算无答案")
        elif "r1" in r:  # 选择题
            if r.get("r2") and r["r1"] != r["r2"]:
                add(qid, f"选择分歧 r1={r['r1']} r2={r['r2']} final={r['final']}")
            if not r.get("r1"):
                add(qid, "r1解析失败")

    ans = json.load(open(outdir / "answers.json"))
    for qid, slots in ans.items():
        if any(s == "" for s in slots):
            add(qid, "存在空答案槽")

    print(f"低置信题: {len(flags)}")
    for qid, rs in sorted(flags.items()):
        print(f"  {qid}: {'; '.join(rs)}")
    out = outdir / "low_confidence.json"
    json.dump(flags, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"→ {out}")


if __name__ == "__main__":
    main(sys.argv[1])
