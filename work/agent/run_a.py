"""A榜批量运行器：并行答题 → answer.csv(tab分隔) + evidence.json + 运行日志。

用法: python -m agent.run_a --tag dev1 [--domains insurance,...] [--qids ...]
      [--limit N] [--workers 6] [--model qwen3.6-plus]
"""
import argparse, json, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, retrieval  # noqa: E402
from agent.qwen_client import LEDGER, DEFAULT_MODEL  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
QDIR = ROOT / "public_dataset_upload" / "questions" / "group_a"
DOMAINS = ["insurance", "financial_contracts", "financial_reports",
           "regulatory", "research"]


def load_questions(domains, qids=None, limit=None):
    qs = []
    for d in domains:
        qs.extend(json.load(open(QDIR / f"{d}_questions.json")))
    if qids:
        qs = [q for q in qs if q["qid"] in qids]
    if limit:
        qs = qs[:limit]
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--domains", default=",".join(DOMAINS))
    ap.add_argument("--qids", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--fresh-digests", action="store_true",
                    help="不用共享缓存，正式跑分用（诚实计token）")
    args = ap.parse_args()

    outdir = ROOT / "work" / "output" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    digest_path = outdir / "digests.json"
    # 开发期共享缓存省钱；正式提交跑分时用 --fresh-digests 全量重建以诚实计token
    shared_cache = ROOT / "work" / "output" / "digest_cache.json"
    if not args.fresh_digests:
        answerer.load_digests(shared_cache)
        answerer.load_digests(digest_path)

    qs = load_questions([d for d in args.domains.split(",") if d],
                        set(args.qids.split(",")) if args.qids else None,
                        args.limit or None)
    print(f"questions: {len(qs)}, model={args.model}", flush=True)

    results = {}
    t0 = time.time()
    log = open(outdir / "run_log.jsonl", "a")
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(answerer.answer_question, q, args.model, log): q
                for q in qs}
        for n, fut in enumerate(as_completed(futs)):
            q = futs[fut]
            try:
                ans, _info = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {q['qid']}: {e}", flush=True)
                ans = "A"
            results[q["qid"]] = ans
            p, c, t = LEDGER.totals()
            print(f"[{n+1}/{len(qs)}] {q['qid']} -> {ans}  "
                  f"(tokens so far: {t:,})", flush=True)
    log.close()
    answerer.save_digests(digest_path)
    answerer.save_digests(shared_cache)
    LEDGER.dump(outdir / "token_ledger.json")

    # answer.csv：tab 分隔，与队友被平台接受的格式一致
    p, c, t = LEDGER.totals()
    order = [q["qid"] for q in load_questions(DOMAINS)]
    with open(outdir / "answer.csv", "w") as f:
        f.write("qid\tanswer\tprompt_tokens\tcompletion_tokens\ttotal_tokens\n")
        f.write(f"summary\t\t{p}\t{c}\t{t}\n")
        for qid in order:
            if qid in results:
                qp, qc = LEDGER.per_qid.get(qid, [0, 0])
                f.write(f"{qid}\t{results[qid]}\t{qp}\t{qc}\t{qp+qc}\n")
    json.dump(results, open(outdir / "answers.json", "w"),
              ensure_ascii=False, indent=1)
    print(f"done in {time.time()-t0:.0f}s; tokens: prompt={p:,} "
          f"completion={c:,} total={t:,}")
    print(f"output: {outdir}/answer.csv")


if __name__ == "__main__":
    main()
