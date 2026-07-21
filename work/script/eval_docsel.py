#!/usr/bin/env python3
"""B模式演练：对A榜题隐藏doc_ids，测文档盲检命中率。

用法: python work/script/eval_docsel.py [--coarse-only] [--domains d1,d2]
"""
import argparse, json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work"))
from agent import doc_select  # noqa: E402

QDIR = ROOT / "public_dataset_upload" / "questions" / "group_a"
DOMAINS = ["insurance", "financial_contracts", "financial_reports",
           "regulatory", "research"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse-only", action="store_true")
    ap.add_argument("--coarse-k", type=int, default=12)
    ap.add_argument("--domains", default=",".join(DOMAINS))
    args = ap.parse_args()

    qs = []
    for d in args.domains.split(","):
        qs.extend(json.load(open(QDIR / f"{d}_questions.json")))

    def run(q):
        truth = set(q["doc_ids"])
        if args.coarse_only:
            got = set(doc_select.coarse_candidates(q, k=args.coarse_k))
        else:
            got = set(doc_select.select_docs(q, k_coarse=args.coarse_k))
        return q, truth, got

    stats = {}
    with ThreadPoolExecutor(max_workers=6) as ex:
        for q, truth, got in ex.map(run, qs):
            full = truth <= got
            s = stats.setdefault(q["domain"], [0, 0, 0, 0])  # full,total,missed,extra
            s[0] += full
            s[1] += 1
            s[2] += len(truth - got)
            s[3] += len(got - truth)
            if not full:
                print(f"MISS {q['qid']}: need {sorted(truth)} got {sorted(got)}")
    print()
    for d, (full, tot, miss, extra) in sorted(stats.items()):
        print(f"{d}: 全命中 {full}/{tot}  漏检文档数 {miss}  多选文档数 {extra}")


if __name__ == "__main__":
    main()
