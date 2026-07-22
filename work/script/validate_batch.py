#!/usr/bin/env python3
"""批量答题在A榜标签上的验证：fin+ins 40题，批量 vs 历史单题最佳。"""
import json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work"))
from agent import answerer, batch  # noqa: E402
from agent.qwen_client import LEDGER  # noqa: E402

qs = []
for d in ["financial_reports", "insurance"]:
    qs += json.load(open(ROOT / f"public_dataset_upload/questions/group_a/{d}_questions.json"))
answerer.load_digests(ROOT / "work/output/digest_cache.json")

groups = batch.group_questions(qs)
sizes = {}
for g in groups: sizes[len(g)] = sizes.get(len(g), 0) + 1
print(f"{len(qs)}题 → {len(groups)}组, 组大小分布: {sizes}")

results = {}
log = open(ROOT / "work/output/val_batch_log.jsonl", "w")

def run(g):
    if len(g) == 1:
        a, _ = answerer.answer_question(g[0], log=log)
        return {g[0]["qid"]: a}
    return batch.answer_batch(g, log=log)

with ThreadPoolExecutor(max_workers=5) as ex:
    for got in ex.map(run, groups):
        results.update(got)
        print(".", end="", flush=True)
print()
labels = json.load(open(ROOT / "work/eval/validation_labels.json"))
ok = tot = 0
for qid, a in results.items():
    lab = labels.get(qid)
    if not lab or lab["confidence"] == "master_wrong": continue
    tot += 1
    ok += (a == lab["answer"])
p, c, t = LEDGER.totals()
print(f"批量模式: {ok}/{tot} 正确; tokens={t:,} ({t//len(qs):,}/题)")
json.dump(results, open(ROOT / "work/output/val_batch_answers.json", "w"), ensure_ascii=False)
