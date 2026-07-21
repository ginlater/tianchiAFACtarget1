"""B榜运行器：文档盲检(doc_select) + 答题。题目文件无 doc_ids。

用法: python -m agent.run_b --tag b_final --questions path/to/b_questions.json
支持断点续跑：--resume 会跳过 answers.json 中已有的 qid。
"""
import argparse, json, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, doc_select  # noqa: E402
from agent.qwen_client import LEDGER, DEFAULT_MODEL  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--questions", required=True, nargs="+",
                    help="一个或多个题目json文件")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh-digests", action="store_true")
    args = ap.parse_args()

    outdir = ROOT / "work" / "output" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    digest_path = outdir / "digests.json"
    if not args.fresh_digests:
        answerer.load_digests(ROOT / "work" / "output" / "digest_cache.json")
        answerer.load_digests(digest_path)

    qs = []
    for f in args.questions:
        data = json.load(open(f))
        qs.extend(data if isinstance(data, list) else data.get("questions", data))
    results = {}
    if args.resume and (outdir / "answers.json").exists():
        results = json.load(open(outdir / "answers.json"))
        qs = [q for q in qs if q["qid"] not in results]
        prev_ledger = outdir / "token_ledger.json"
        if prev_ledger.exists():  # 续跑合并历史token，保证诚实计量
            prev = json.load(open(prev_ledger))
            for k, v in prev["per_qid"].items():
                slot = LEDGER.per_qid.setdefault(k, [0, 0])
                slot[0] += v[0]
                slot[1] += v[1]
            LEDGER.calls.extend(prev.get("calls", []))
    print(f"questions to run: {len(qs)}, model={args.model}", flush=True)

    docsel_log = open(outdir / "docsel_log.jsonl", "a")
    log = open(outdir / "run_log.jsonl", "a")

    def work(q):
        blind = not q.get("doc_ids")
        if blind:
            picked = doc_select.select_docs(q, model=args.model)
            q = dict(q, doc_ids=picked)
            docsel_log.write(json.dumps(
                {"qid": q["qid"], "picked": picked}, ensure_ascii=False) + "\n")
            docsel_log.flush()
        return answerer.answer_question(q, args.model, log, blind_mode=blind)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(work, q): q for q in qs}
        for n, fut in enumerate(as_completed(futs)):
            q = futs[fut]
            try:
                ans, _info = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"FAIL {q['qid']}: {e}", flush=True)
                ans = "A"
            results[q["qid"]] = ans
            _p, _c, t = LEDGER.totals()
            print(f"[{n+1}/{len(qs)}] {q['qid']} -> {ans} (tokens: {t:,})",
                  flush=True)
            json.dump(results, open(outdir / "answers.json", "w"),
                      ensure_ascii=False, indent=1)
    log.close()
    docsel_log.close()
    answerer.save_digests(digest_path)
    answerer.save_digests(ROOT / "work" / "output" / "digest_cache.json")
    LEDGER.dump(outdir / "token_ledger.json")

    p, c, t = LEDGER.totals()
    order = [q["qid"] for f in args.questions
             for q in json.load(open(f))]
    with open(outdir / "answer.csv", "w") as f:
        f.write("qid\tanswer\tprompt_tokens\tcompletion_tokens\ttotal_tokens\n")
        f.write(f"summary\t\t{p}\t{c}\t{t}\n")
        for qid in order:
            if qid in results:
                qp, qc = LEDGER.per_qid.get(qid, [0, 0])
                f.write(f"{qid}\t{results[qid]}\t{qp}\t{qc}\t{qp+qc}\n")
    print(f"done in {time.time()-t0:.0f}s; total tokens {t:,}")
    print(f"output: {outdir}/answer.csv")


if __name__ == "__main__":
    main()
