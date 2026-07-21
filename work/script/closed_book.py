#!/usr/bin/env python3
"""闭卷基线诊断：不给任何证据直接让 qwen3.6-plus 答题。

用途（用户反先验洞察的操作化）：
- 闭卷对而流水线错 → 流水线把模型带偏（证据噪声/提示问题）
- 闭卷错而流水线对 → 检索发挥了作用（题目反先验设计生效）
- 闭卷==流水线 且 都错 → 最危险：检索失败+先验漏答，重点修检索
输出 work/eval/closed_book.json
"""
import json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "work"))
from agent.qwen_client import chat  # noqa: E402
from agent.answerer import parse_answer, FMT_NAME  # noqa: E402

QDIR = ROOT / "public_dataset_upload" / "questions" / "group_a"
DOMAINS = ["insurance", "financial_contracts", "financial_reports",
           "regulatory", "research"]


def ask(q):
    opts = "\n".join(f"{k}. {v}" for k, v in q["options"].items())
    prompt = (f"金融题({FMT_NAME[q['answer_format']]})，凭你的知识作答，"
              f"最后一行输出 答案: <字母>\n{q['question']}\n{opts}")
    try:
        c, _r, _u = chat([{"role": "user", "content": prompt}], qid="closed",
                         thinking=True, thinking_budget=1500, max_tokens=2200)
        return q["qid"], parse_answer(c, q["answer_format"])
    except Exception as e:  # noqa: BLE001
        return q["qid"], f"ERR:{e}"


def main():
    qs = []
    for d in DOMAINS:
        qs.extend(json.load(open(QDIR / f"{d}_questions.json")))
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for qid, ans in ex.map(ask, qs):
            out[qid] = ans
            print(qid, ans, flush=True)
    json.dump(out, open(ROOT / "work" / "eval" / "closed_book.json", "w"),
              ensure_ascii=False, indent=1)
    labels = json.load(open(ROOT / "work" / "eval" / "validation_labels.json"))
    ok = sum(1 for qid, a in out.items()
             if labels.get(qid, {}).get("confidence") != "master_wrong"
             and a == labels.get(qid, {}).get("answer"))
    print(f"\n闭卷与标签一致: {ok}/98 (母版2道错题除外)")


if __name__ == "__main__":
    main()
