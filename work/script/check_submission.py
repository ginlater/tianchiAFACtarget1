#!/usr/bin/env python3
"""answer.csv 提交前终检：行数/唯一性/答案合法性/token一致性/题型匹配。

用法: python work/script/check_submission.py <answer.csv> [题目json...]
不传题目文件时默认用 A 榜题目；B 榜请显式传入 B 题文件。
"""
import csv, json, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
QDIR = ROOT / "public_dataset_upload" / "questions" / "group_a"


def main(path, qfiles=()):
    qs = {}
    files = [pathlib.Path(f) for f in qfiles] or sorted(QDIR.glob("*_questions.json"))
    for f in files:
        data = json.load(open(f))
        for q in (data if isinstance(data, list) else data.get("questions", data)):
            qs[q["qid"]] = q["answer_format"]
    print(f"题目来源: {[f.name for f in files]} ({len(qs)}题)")

    rows = list(csv.reader(open(path), delimiter="\t"))
    errs, warns = [], []
    assert rows[0] == ["qid", "answer", "prompt_tokens", "completion_tokens",
                      "total_tokens"], f"header异常: {rows[0]}"
    summary = rows[1]
    assert summary[0] == "summary", "第二行必须是summary"
    p, c, t = int(summary[2]), int(summary[3]), int(summary[4])
    if p + c != t:
        errs.append(f"summary token不自洽: {p}+{c}!={t}")
    body = rows[2:]
    seen = set()
    per_sum = [0, 0]
    for r in body:
        qid, ans = r[0], r[1]
        if qid in seen:
            errs.append(f"重复qid {qid}")
        seen.add(qid)
        if qid not in qs:
            errs.append(f"未知qid {qid}")
            continue
        fmt = qs[qid]
        if not ans:
            errs.append(f"{qid} 答案为空")
        elif fmt in ("mcq", "tf") and (len(ans) != 1 or ans not in "ABCD"):
            errs.append(f"{qid} {fmt}答案非法: {ans!r}")
        elif fmt == "tf" and ans not in "AB":
            errs.append(f"{qid} tf答案非法: {ans!r}")
        elif fmt == "multi":
            if (not ans or any(ch not in "ABCD" for ch in ans)
                    or list(ans) != sorted(set(ans))):
                errs.append(f"{qid} multi答案非法/乱序: {ans!r}")
        if len(r) >= 5 and r[2]:
            per_sum[0] += int(r[2])
            per_sum[1] += int(r[3])
    missing = set(qs) - seen
    if missing:
        errs.append(f"缺少{len(missing)}题: {sorted(missing)[:5]}...")
    if per_sum[0] and per_sum[0] > p:
        warns.append(f"逐题prompt token合计({per_sum[0]})大于summary({p})")
    print(f"行数: {len(rows)} (header+summary+{len(body)}题)")
    print(f"summary tokens: prompt={p:,} completion={c:,} total={t:,}")
    from collections import Counter
    fmts = Counter(qs[r[0]] for r in body if r[0] in qs)
    print(f"题型分布: {dict(fmts)}")
    for w in warns:
        print("WARN:", w)
    if errs:
        print("\n!!! 发现问题:")
        for e in errs:
            print("  -", e)
        sys.exit(1)
    print("\n✓ 格式检查全部通过")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
