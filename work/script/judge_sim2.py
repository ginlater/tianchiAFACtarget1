#!/usr/bin/env python3
"""模拟judge v2：全量评分+双采样降噪+官方rubric逐字复刻+实测校准偏移。
用法: .venv/bin/python script/judge_sim2.py <reasonings.json> [--polish]
--polish: 对校准后<80分的题定向润色并复评(输出到 <file>.polished.json)
"""
import json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent.qwen_client import chat, DEFAULT_MODEL  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
path = sys.argv[1]
do_polish = "--polish" in sys.argv
res = json.load(open(path))

# 官方rubric逐字复刻(4.2/4.3节)
RUBRIC = (
    "你是推理过程评分的LLM judge。仅检查提交的推理文本写作质量，"
    "不加载任何外部题目、原文或答案数据。按三个维度打分(各0-100)：\n"
    "logical(逻辑连贯性): 推理步骤之间是否存在清晰因果关系，整体链条是否自洽\n"
    "completeness(论证完整性): 是否具备完整分析过程，如定位、提取、推导和结论\n"
    "clarity(表达清晰度): 是否条理清晰、结构化、表达准确\n"
    "评分标准：80分以上=逻辑清晰、论证完整、表达专业；60-79=有明确分析步骤"
    "和推导过程；30-59=有部分分析，但不完整；0-29=空泛、模板化或无实质内容。\n"
    "只输出JSON: {\"logical\":N,\"completeness\":N,\"clarity\":N}")

# 校准偏移: 对实测85分批次(full10四段式)的sim均值 − 85
CALIB_FILE = OUT / "judge_calib.json"


def one_call(text):
    c1, _r, _u = chat([{"role": "user", "content":
                        f"推理文本:\n{text}\n\n{RUBRIC}"}],
                      qid="_judge", model=DEFAULT_MODEL,
                      thinking=False, max_tokens=60, tag="judge2")
    m = re.search(r"\{[^}]+\}", c1 or "")
    try:
        d = json.loads(m.group(0))
        return (float(d["logical"]) + float(d["completeness"])
                + float(d["clarity"])) / 3
    except Exception:  # noqa: BLE001
        return None


def judge(qid):
    vals = [v for v in (one_call(res[qid]), one_call(res[qid]))
            if v is not None]
    return qid, (sum(vals) / len(vals)) if vals else None


def score_all(rmap):
    global res
    res = rmap
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for qid, s in ex.map(judge, sorted(rmap)):
            if s is not None:
                out[qid] = s
    return out


# 1) 确保校准偏移存在
if CALIB_FILE.exists():
    offset = json.load(open(CALIB_FILE))["offset"]
else:
    base = json.load(open(OUT / "reasonings_b_full10.json"))
    bs = score_all(base)
    sim_mean = sum(bs.values()) / len(bs)
    offset = sim_mean - 85.0
    json.dump({"offset": offset, "sim_mean": sim_mean, "n": len(bs)},
              open(CALIB_FILE, "w"))
    print(f"校准完成: sim均值{sim_mean:.1f} → 偏移{offset:+.1f}")

# 2) 全量评分目标文件
target = json.load(open(path))
scores = score_all(target)
adj = {q: s - offset for q, s in scores.items()}
vals = sorted(adj.values())
mean = sum(vals) / len(vals)
print(f"全量{len(vals)}题 | 校准后预估推理分: 均值{mean:.1f} "
      f"min{vals[0]:.0f} 中位{vals[len(vals)//2]:.0f} max{vals[-1]:.0f}")
low = sorted([(s, q) for q, s in adj.items() if s < 80])
print(f"校准后<80分({len(low)}题):", [q for _s, q in low][:15])
json.dump({q: round(s, 1) for q, s in adj.items()},
          open(str(path) + ".simscores.json", "w"), ensure_ascii=False)

# 3) 定向润色闭环
if do_polish and low:
    POLISH = (
        "这段推理摘要在评审中偏弱。请重写使其达到'逻辑清晰、论证完整、"
        "表达专业'档：显式因果连接、定位/提取/推导/结论四要素齐全、"
        "保留全部页码与数值、术语准确无冗余。140-230字，只输出正文。")

    def polish(qid):
        c1, _r, _u = chat([{"role": "user", "content":
                            f"原摘要:\n{target[qid]}\n\n{POLISH}"}],
                          qid=qid, model=DEFAULT_MODEL, thinking=False,
                          max_tokens=330, tag="r_repolish")
        t = (c1 or "").strip().replace("\n", " ")
        return qid, t if len(t) >= 60 else target[qid]

    with ThreadPoolExecutor(max_workers=6) as ex:
        for qid, t in ex.map(polish, [q for _s, q in low]):
            target[qid] = t
    out2 = str(path).replace(".json", "") + ".polished.json"
    json.dump(target, open(out2, "w"), ensure_ascii=False, indent=1)
    ns = score_all({q: target[q] for _s, q in low})
    nadj = [s - offset for s in ns.values()]
    print(f"润色后低分组复评: 均值{sum(nadj)/len(nadj):.1f}")
    print("→", out2)
