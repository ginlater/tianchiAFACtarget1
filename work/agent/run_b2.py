"""B榜正式运行器：题型自适应（选择题/计算题）+ 文档盲检 + B格式提交。

用法:
  python -m agent.run_b2 --tag b_final \
      --qdir ../upload_b/question_b --submit-template ../upload_b/submit.csv \
      [--resume] [--limit N] [--qids a,b] [--fresh-digests]
"""
import argparse, json, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema, batch, calc, doc_select  # noqa: E402
from agent.qwen_client import LEDGER, DEFAULT_MODEL  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--qdir", required=True)
    ap.add_argument("--submit-template", required=True)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--verify-model", default="")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fresh-digests", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--qids", default="")
    ap.add_argument("--batch", action="store_true",
                    help="同(域,文档集)选择题批量共享证据作答")
    args = ap.parse_args()

    outdir = ROOT / "work" / "output" / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    shared = ROOT / "work" / "output" / "digest_cache.json"
    if not args.fresh_digests:
        answerer.load_digests(shared)
        answerer.load_digests(outdir / "digests.json")

    qs_all = b_schema.load_questions(args.qdir)
    schema = b_schema.load_schema(args.submit_template)
    order = [q["qid"] for q in qs_all]

    results = {}
    if args.resume and (outdir / "answers.json").exists():
        results = json.load(open(outdir / "answers.json"))
        prev = outdir / "token_ledger.json"
        if prev.exists():  # 合并历史token，保证诚实计量
            d = json.load(open(prev))
            for k, v in d["per_qid"].items():
                slot = LEDGER.per_qid.setdefault(k, [0, 0])
                slot[0] += v[0]
                slot[1] += v[1]
            LEDGER.calls.extend(d.get("calls", []))
    qs = [q for q in qs_all if q["qid"] not in results]
    if args.qids:
        keep = set(args.qids.split(","))
        qs = [q for q in qs if q["qid"] in keep]
    if args.limit:
        qs = qs[:args.limit]
    print(f"待作答 {len(qs)} / 共 {len(qs_all)} 题；model={args.model} "
          f"verify={args.verify_model or args.model}", flush=True)

    log = open(outdir / "run_log.jsonl", "a")
    dlog = open(outdir / "docsel_log.jsonl", "a")

    def work(q):
        kinds = b_schema.effective_kinds(q, schema.get(q["qid"], ["letter"]))
        if not q.get("doc_ids"):
            picked = doc_select.select_docs(q, model=args.model)
            q = dict(q, doc_ids=picked)
            dlog.write(json.dumps({"qid": q["qid"], "picked": picked,
                                   "kinds": kinds}, ensure_ascii=False) + "\n")
            dlog.flush()
        if q["answer_format"] == "calc":
            raw = calc.answer_calc(q, kinds, model=args.model, log=log,
                                   verify_model=args.verify_model or None,
                                   blind_mode=True)
            return b_schema.split_answer(raw, kinds)
        ans, _info = answerer.answer_question(q, args.model, log,
                                              blind_mode=True)
        return [b_schema.fmt_slot(ans, "letter")]

    t0 = time.time()
    if args.batch:
        # 阶段1: 分域批量盲检（候选卡每域只发一次；reg 每10题一组）
        from collections import defaultdict
        pre = [q for q in qs if q.get("doc_ids")]
        need = [q for q in qs if not q.get("doc_ids")]
        by_dom = defaultdict(list)
        for q in need:
            by_dom[q["domain"]].append(q)
        chunks = []
        for dom, dqs in by_dom.items():
            gs = 10 if dom == "regulatory" else 20
            chunks += [dqs[i:i + gs] for i in range(0, len(dqs), gs)]

        def docsel_chunk(chunk):
            try:
                got = doc_select.select_docs_batch(chunk, model=args.model)
            except Exception as e:  # noqa: BLE001 — 整块失败退单题
                print(f"docsel batch fail ({chunk[0]['domain']}): {e}",
                      flush=True)
                got = {q["qid"]: doc_select.select_docs(q, model=args.model)
                       for q in chunk}
            out = []
            for q in chunk:
                picked = got[q["qid"]]
                dlog.write(json.dumps({"qid": q["qid"], "picked": picked},
                                      ensure_ascii=False) + "\n")
                dlog.flush()
                out.append(dict(q, doc_ids=picked))
            return out

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            qs = pre + [q for lst in ex.map(docsel_chunk, chunks)
                        for q in lst]
        print(f"docsel 完成; tokens {LEDGER.totals()[2]:,}", flush=True)
        # 阶段2: 选择题分组批答 + 计算题独立
        choice = [q for q in qs if q["answer_format"] != "calc"]
        calcs = [q for q in qs if q["answer_format"] == "calc"]
        groups = batch.group_questions(choice)
        print(f"选择题 {len(choice)} → {len(groups)} 组; 计算题 {len(calcs)}",
              flush=True)

        def run_group(g):
            if len(g) == 1:
                ans, _ = answerer.answer_question(g[0], args.model, log,
                                                  blind_mode=True)
                return {g[0]["qid"]: [b_schema.fmt_slot(ans, "letter")]}
            finals = batch.answer_batch(g, model=args.model, log=log)
            return {qid: [b_schema.fmt_slot(a, "letter")]
                    for qid, a in finals.items()}

        def run_calc(q):
            kinds = b_schema.effective_kinds(q, schema.get(q["qid"], ["number"]))
            raw = calc.answer_calc(q, kinds, model=args.model, log=log,
                                   verify_model=args.verify_model or None,
                                   blind_mode=True)
            return {q["qid"]: b_schema.split_answer(raw, kinds)}

        jobs = [(run_group, g) for g in groups] + [(run_calc, q) for q in calcs]
        done_n = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(fn, arg) for fn, arg in jobs]
            for fut in as_completed(futs):
                try:
                    got = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"FAIL batch/calc: {e}", flush=True)
                    got = {}
                results.update(got)
                done_n += len(got)
                _p, _c, t = LEDGER.totals()
                print(f"[{done_n}/{len(qs)}] +{list(got)} ({t:,} tok)",
                      flush=True)
                json.dump(results, open(outdir / "answers.json", "w"),
                          ensure_ascii=False, indent=1)
        # 补漏：任何未出答案的题回退单题
        missing = [q for q in qs if q["qid"] not in results]
        for q in missing:
            try:
                slots = work(q)
            except Exception:  # noqa: BLE001
                slots = ["A"] if q["answer_format"] != "calc" else ["0.00"]
            results[q["qid"]] = slots
        if missing:
            print(f"回退补漏 {len(missing)} 题", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, q): q for q in qs}
            for n, fut in enumerate(as_completed(futs)):
                q = futs[fut]
                try:
                    slots = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"FAIL {q['qid']}: {e}", flush=True)
                    slots = ["A"] if q["answer_format"] != "calc" else ["0.00"]
                results[q["qid"]] = slots
                _p, _c, t = LEDGER.totals()
                print(f"[{n+1}/{len(qs)}] {q['qid']} -> {slots} ({t:,} tok)",
                      flush=True)
                json.dump(results, open(outdir / "answers.json", "w"),
                          ensure_ascii=False, indent=1)
    log.close()
    dlog.close()
    answerer.save_digests(outdir / "digests.json")
    answerer.save_digests(shared)
    LEDGER.dump(outdir / "token_ledger.json")

    b_schema.write_submission(outdir / "answer.csv", results, schema, order,
                              LEDGER.per_qid, LEDGER.totals())
    p, c, t = LEDGER.totals()
    print(f"done in {time.time()-t0:.0f}s; tokens {t:,} (p={p:,} c={c:,})")
    print(f"output: {outdir}/answer.csv")


if __name__ == "__main__":
    main()
