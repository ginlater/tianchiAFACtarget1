#!/usr/bin/env python3
"""模拟LLM judge：按官方三维rubric(逻辑/完整/清晰)给推理摘要打分。
纯赛前校准工具，不参与生成提交内容。
用法: .venv/bin/python script/judge_sim.py <reasonings.json> [抽样数,默认20]
"""
import json, pathlib, random, re, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent.qwen_client import chat, DEFAULT_MODEL  # noqa: E402

path = sys.argv[1]
n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 20
res = json.load(open(path))
random.seed(42)
picks = random.sample(sorted(res), min(n_sample, len(res)))

RUBRIC = (
    "你是推理文本质量评审。仅根据文本本身（不参考任何题目或答案数据），"
    "按三个维度各打0-100分：\n"
    "logical(逻辑连贯性): 推理步骤间是否有清晰因果关系，链条是否自洽；\n"
    "completeness(论证完整性): 是否具备完整分析过程（定位、提取、推导、结论）；\n"
    "clarity(表达清晰度): 是否条理清晰、结构化、表达准确。\n"
    "参考分档：80+=逻辑清晰论证完整表达专业；60-79=有明确分析步骤和推导；"
    "30-59=有部分分析但不完整；0-29=空泛、模板化或无实质内容。\n"
    "只输出JSON: {\"logical\":N,\"completeness\":N,\"clarity\":N}")


def judge(qid):
    c1, _r, _u = chat([{"role": "user", "content":
                        f"推理文本:\n{res[qid]}\n\n{RUBRIC}"}],
                      qid=f"_judge_{qid}", model=DEFAULT_MODEL,
                      thinking=False, max_tokens=60, tag="judge_sim")
    m = re.search(r"\{[^}]+\}", c1 or "")
    try:
        d = json.loads(m.group(0))
        return qid, (d["logical"] + d["completeness"] + d["clarity"]) / 3
    except Exception:  # noqa: BLE001
        return qid, None


scores = {}
with ThreadPoolExecutor(max_workers=6) as ex:
    for qid, s in ex.map(judge, picks):
        if s is not None:
            scores[qid] = s
vals = sorted(scores.values())
if vals:
    print(f"抽样{len(vals)}题 模拟推理分: 均值{sum(vals)/len(vals):.1f} "
          f"min{vals[0]:.0f} 中位{vals[len(vals)//2]:.0f} max{vals[-1]:.0f}")
    low = [q for q, s in scores.items() if s < 70]
    print("低分题:", low or "无")
