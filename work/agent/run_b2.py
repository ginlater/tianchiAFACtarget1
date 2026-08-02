"""B榜正式运行器：题型自适应（选择题/计算题）+ 文档盲检 + B格式提交。

用法:
  python -m agent.run_b2 --tag b_final \
      --qdir ../upload_b/question_b --submit-template ../upload_b/submit.csv \
      [--resume] [--limit N] [--qids a,b] [--fresh-digests]
"""
import argparse, json, pathlib, sys, time
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema, batch, calc, doc_select  # noqa: E402
from agent.paths import OUTPUT_DIR  # noqa: E402
from agent.qwen_client import (LEDGER, DEFAULT_MODEL, close_audit,
                               configure_audit)  # noqa: E402
from agent.repro import (LockedJsonlWriter, collect_reasonings,
                         write_run_config)  # noqa: E402


def _fast_docsel_enabled(q):
    """Return whether the conservative code selector is enabled for ``q``.

    Calculation domains can be narrowed independently.  This preserves the
    authentic Qwen source-selection stage where the reference workflow used
    it, while allowing contract calculations with explicit titles/companies
    to avoid a redundant selector.  The gate reads only domain and question
    type, never identifiers or outputs.
    """
    if os.environ.get("AFAC_FAST_DOCSEL") != "1":
        return False
    if q.get("answer_format") != "calc":
        return True
    if os.environ.get("AFAC_FAST_DOCSEL_CALC") != "1":
        return False
    raw = os.environ.get("AFAC_FAST_DOCSEL_CALC_DOMAINS", "")
    domains = {x.strip() for x in raw.split(",") if x.strip()}
    return not domains or q.get("domain") in domains


def _docsel_batch_key(q):
    """Keep research source selection homogeneous by answer workflow.

    Research calculations need a compact numeric source set, whereas choice
    syntheses deliberately select several reports.  Mixing both in one Qwen
    selector lets a low-confidence narrative report change a choice question's
    later batch route and spreads the selector cost away from calculations.
    Other domains retain their established domain-wide selector batches.
    """

    domain = q.get("domain")
    if domain == "research":
        return (domain, "calc" if q.get("answer_format") == "calc"
                else "choice")
    return (domain, "all")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--output-dir", default="",
                    help="explicit output directory for one-click reproduction")
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

    for label, model_name in (("model", args.model),
                              ("verify model", args.verify_model)):
        if model_name and not model_name.lower().startswith("qwen"):
            ap.error(f"{label} must be a Qwen model, got {model_name!r}")
    if args.verify_model:
        # answerer/batch keep this as an import-time compatibility constant;
        # make the CLI value the explicit effective setting.
        answerer.VERIFY_MODEL = args.verify_model
        batch.VERIFY_MODEL = args.verify_model

    if not args.tag and not args.output_dir:
        ap.error("one of --tag or --output-dir is required")
    outdir = (pathlib.Path(args.output_dir).expanduser().resolve()
              if args.output_dir else OUTPUT_DIR / args.tag)
    if outdir.exists() and any(outdir.iterdir()) and not args.resume:
        raise RuntimeError(f"output directory is not empty: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    shared = outdir / "digest_cache.json"
    if not args.fresh_digests:
        answerer.load_digests(shared)
        answerer.load_digests(outdir / "digests.json")

    # Load the complete official question order before freezing run_config.
    # Schema 2 records this immutable order so partial/qid-filtered runs cannot
    # later be presented as a formal 100-question reproduction.
    qs_all = b_schema.load_questions(args.qdir)
    schema = b_schema.load_schema(args.submit_template)
    order = [q["qid"] for q in qs_all]

    configure_audit(outdir / "api_calls.jsonl", append=args.resume)
    write_run_config(outdir / "run_config.json", args, args.qdir,
                     args.submit_template, question_ids=order)

    results, reasonings, reasoning_sources = {}, {}, {}
    if args.resume and (outdir / "answers.json").exists():
        results = json.load(open(outdir / "answers.json"))
        rp = outdir / "reasonings.json"
        sp = outdir / "reasoning_sources.json"
        reasonings = json.load(open(rp)) if rp.exists() else {}
        reasoning_sources = json.load(open(sp)) if sp.exists() else {}
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

    mode = "a" if args.resume else "w"
    log = LockedJsonlWriter(outdir / "run_log.jsonl", mode)
    dlog = LockedJsonlWriter(outdir / "docsel_log.jsonl", mode)

    def checkpoint():
        for name, data in (("answers.json", results),
                           ("reasonings.json", reasonings),
                           ("reasoning_sources.json", reasoning_sources)):
            tmp = outdir / (name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
            tmp.replace(outdir / name)

    def accept(got):
        for qid, (slots, info) in got.items():
            reasoning = (info.get("reasoning") or "").strip()
            if len(reasoning) < 20:
                raise RuntimeError(f"{qid}: reasoning shorter than 20 chars")
            results[qid] = slots
            reasonings[qid] = reasoning
            reasoning_sources[qid] = info
        checkpoint()

    def work(q):
        kinds = b_schema.effective_kinds(q, schema.get(q["qid"], ["letter"]))
        if not q.get("doc_ids"):
            decision = None
            if _fast_docsel_enabled(q):
                from agent.docsel_fast import select_docs_fast
                decision = select_docs_fast(q)
            if decision is not None:
                picked, diag = decision
                dlog.write(json.dumps({"qid": q["qid"], "picked": picked,
                                       "kinds": kinds, "selector": "code",
                                       "diagnostics": diag},
                                      ensure_ascii=False) + "\n")
            else:
                picked = doc_select.select_docs(q, model=args.model)
                dlog.write(json.dumps({"qid": q["qid"], "picked": picked,
                                       "kinds": kinds, "selector": "qwen"},
                                      ensure_ascii=False) + "\n")
            q = dict(q, doc_ids=picked)
            dlog.flush()
        if q["answer_format"] == "calc":
            raw, info = calc.answer_calc(
                q, kinds, model=args.model, log=log,
                verify_model=args.verify_model or None,
                blind_mode=True, return_info=True)
            return b_schema.split_answer(raw, kinds), info
        ans, info = answerer.answer_question(q, args.model, log,
                                             blind_mode=True)
        return [b_schema.fmt_slot(ans, "letter")], info

    t0 = time.time()
    if args.batch:
        # 阶段1: 分域批量盲检（候选卡每域只发一次；reg 每10题一组）
        from collections import defaultdict
        pre = [q for q in qs if q.get("doc_ids")]
        need = [q for q in qs if not q.get("doc_ids")]
        fast = []
        unresolved = []
        if os.environ.get("AFAC_FAST_DOCSEL") == "1":
            from agent.docsel_fast import select_docs_fast
            for q in need:
                decision = (select_docs_fast(q)
                            if _fast_docsel_enabled(q) else None)
                if decision is None:
                    unresolved.append(q)
                    continue
                picked, diag = decision
                dlog.write(json.dumps({
                    "qid": q["qid"], "picked": picked,
                    "selector": "code", "diagnostics": diag},
                    ensure_ascii=False) + "\n")
                fast.append(dict(q, doc_ids=picked))
        else:
            unresolved = need
        by_dom = defaultdict(list)
        for q in unresolved:
            by_dom[_docsel_batch_key(q)].append(q)
        chunks = []
        for (dom, _workflow), dqs in by_dom.items():
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
                dlog.write(json.dumps({"qid": q["qid"], "picked": picked,
                                       "selector": "qwen"},
                                      ensure_ascii=False) + "\n")
                dlog.flush()
                out.append(dict(q, doc_ids=picked))
            return out

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            qs = pre + fast + [q for lst in ex.map(docsel_chunk, chunks)
                        for q in lst]
        print(f"docsel 完成; code={len(fast)} qwen={len(unresolved)}; "
              f"tokens {LEDGER.totals()[2]:,}", flush=True)
        # 阶段2: 选择题分组批答 + 计算题独立
        choice = [q for q in qs if q["answer_format"] != "calc"]
        calcs = [q for q in qs if q["answer_format"] == "calc"]
        groups = batch.group_questions(choice)
        print(f"选择题 {len(choice)} → {len(groups)} 组; 计算题 {len(calcs)}",
              flush=True)

        def run_group(g):
            if len(g) == 1 and not batch.requires_batch_singleton(g[0]):
                ans, info = answerer.answer_question(g[0], args.model, log,
                                                     blind_mode=True)
                return {g[0]["qid"]: (
                    [b_schema.fmt_slot(ans, "letter")], info)}
            finals, infos = batch.answer_batch(
                g, model=args.model, log=log, return_info=True)
            return {qid: ([b_schema.fmt_slot(a, "letter")], infos[qid])
                    for qid, a in finals.items()}

        def run_calc(q):
            kinds = b_schema.effective_kinds(q, schema.get(q["qid"], ["number"]))
            raw, info = calc.answer_calc(
                q, kinds, model=args.model, log=log,
                verify_model=args.verify_model or None,
                blind_mode=True, return_info=True)
            return {q["qid"]: (b_schema.split_answer(raw, kinds), info)}

        jobs = [(run_group, g) for g in groups] + [(run_calc, q) for q in calcs]
        done_n = 0
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(fn, arg) for fn, arg in jobs]
            for fut in as_completed(futs):
                got = fut.result()
                accept(got)
                done_n += len(got)
                _p, _c, t = LEDGER.totals()
                print(f"[{done_n}/{len(qs)}] +{list(got)} ({t:,} tok)",
                      flush=True)
        # 批解析遗漏可安全回退到同一正式单题流程；失败则整次运行失败。
        missing = [q for q in qs if q["qid"] not in results]
        for q in missing:
            slots, info = work(q)
            accept({q["qid"]: (slots, info)})
        if missing:
            print(f"回退补漏 {len(missing)} 题", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(work, q): q for q in qs}
            for n, fut in enumerate(as_completed(futs)):
                q = futs[fut]
                slots, info = fut.result()
                accept({q["qid"]: (slots, info)})
                _p, _c, t = LEDGER.totals()
                print(f"[{n+1}/{len(qs)}] {q['qid']} -> {slots} ({t:,} tok)",
                      flush=True)
    log.close()
    dlog.close()
    answerer.save_digests(outdir / "digests.json")
    answerer.save_digests(shared)
    LEDGER.dump(outdir / "token_ledger.json")

    # Deterministic fallback only reads visible outputs already paid for.
    recovered = collect_reasonings(outdir / "run_log.jsonl", results)
    for qid, text in recovered.items():
        reasonings.setdefault(qid, text)
    missing_answers = [qid for qid in order if qid not in results]
    missing_reasoning = [qid for qid in results
                         if len((reasonings.get(qid) or "").strip()) < 20]
    if not args.limit and not args.qids and missing_answers:
        raise RuntimeError(f"incomplete run, missing answers: {missing_answers}")
    if missing_reasoning:
        raise RuntimeError(f"missing reasoning: {missing_reasoning}")
    checkpoint()

    b_schema.write_submission(outdir / "answer.csv", results, schema, order,
                              LEDGER.per_qid, LEDGER.totals(),
                              reasonings=reasonings)
    close_audit()
    p, c, t = LEDGER.totals()
    print(f"done in {time.time()-t0:.0f}s; tokens {t:,} (p={p:,} c={c:,})")
    print(f"output: {outdir}/answer.csv")


if __name__ == "__main__":
    try:
        main()
    finally:
        # API and JSONL sinks are also registered with atexit, but closing here
        # makes partial failure logs immediately inspectable in long-lived
        # runners and is safe because all closes are idempotent.
        close_audit()
