#!/usr/bin/env python3
"""对抗性逐行审查军团：三镜头（逻辑/完整性/清晰度）过堂推理池每一行，
输出结构化缺陷清单。评委是确定性的→每个真实缺陷修复都稳定兑现。
用法: .venv/bin/python script/review_rows.py <池名 默认v8>
"""
import json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent.qwen_client import chat, DEFAULT_MODEL  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
POOL = sys.argv[1] if len(sys.argv) > 1 else "v8"

LENSES = {
    "logical": ("你是严苛的逻辑审查官。只看下面这段推理文本本身（不看任何外部资料），"
                "找出其中的逻辑缺陷：因果断裂、循环论证、前后矛盾、跳步、"
                "结论与论据脱节。"),
    "complete": ("你是论证完整性审查官。检查这段推理是否四要素齐全：数据定位"
                 "（来源/页码）、数值提取、推导过程（算式或逐项判定）、明确结论；"
                 "找出缺失或含糊的要素。"),
    "clarity": ("你是表达清晰度审查官。找出这段文本中表述含混、指代不明、"
                "句子过载、术语错用、读者无法跟上的地方。"),
}
TAIL = ("\n严格但公正：只报告真实存在的问题，无问题就说无。"
        "只输出JSON: {\"severity\": 0到3的整数(0=无伤,1=瑕疵,2=明显缺陷,3=严重), "
        "\"defects\": [\"缺陷简述\", ...]}")


def review(qid, text, lens, inst):
    c1, _r, _u = chat([{"role": "user", "content":
                        f"{inst}\n\n推理文本:\n{text}\n{TAIL}"}],
                      qid=f"_rev_{qid}", model=DEFAULT_MODEL, thinking=False,
                      max_tokens=300, tag=f"rev_{lens}")
    m = re.search(r"\{.*\}", c1 or "", re.S)
    try:
        d = json.loads(m.group(0))
        return int(d.get("severity", 0)), d.get("defects", [])
    except Exception:  # noqa: BLE001
        return 0, []


def main():
    R = json.load(open(OUT / f"reasonings_{POOL}.json"))
    report = {}
    jobs = [(q, t, lens, inst) for q, t in R.items()
            for lens, inst in LENSES.items()]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(review, q, t, lens, inst): (q, lens)
                for q, t, lens, inst in jobs}
        done = 0
        for f in as_completed(futs):
            q, lens = futs[f]
            sev, defects = f.result()
            if sev >= 2:
                report.setdefault(q, {})[lens] = {"sev": sev, "def": defects}
            done += 1
            if done % 60 == 0:
                print(f"[{done}/{len(jobs)}]", flush=True)
    json.dump(report, open(OUT / f"row_defects_{POOL}.json", "w"),
              ensure_ascii=False, indent=1)
    sevsum = {q: sum(v["sev"] for v in d.values()) for q, d in report.items()}
    worst = sorted(sevsum.items(), key=lambda x: -x[1])
    print(f"缺陷行 {len(report)}/100; 最重: {worst[:10]}")


if __name__ == "__main__":
    main()
