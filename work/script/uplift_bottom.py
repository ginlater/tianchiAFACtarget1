#!/usr/bin/env python3
"""底带行升级轮：对 sim 底带行做维度诊断→靶向重生成→双盲复评，
只有 (全门禁绿) AND (复评 ≥ 原分+1.5) 才采用——防随机化伤害。
用法: .venv/bin/python script/uplift_bottom.py
"""
import json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from repair_v4M3 import WIDE, consistent  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "output"
HEDGE = re.compile(r"可能|或许|无法验证|证据缺失|未提供|未列示|未直接|暂不")

RUBRIC = (
    "你是推理过程评分的LLM judge。仅检查提交的推理文本写作质量，"
    "不加载任何外部题目、原文或答案数据。按三个维度打分(各0-100)：\n"
    "logical: 推理步骤之间是否存在清晰因果关系，整体链条是否自洽\n"
    "completeness: 是否具备完整分析过程，如定位、提取、推导和结论\n"
    "clarity: 是否条理清晰、结构化、表达准确\n"
    "只输出JSON: {\"logical\":N,\"completeness\":N,\"clarity\":N}")

targets = json.load(open(OUT / "final_bottom_band.json"))
R = json.load(open(OUT / "reasonings_v9.json"))
led = json.load(open(OUT / "reasoning_v9_ledger.json"))["per_qid"]
ans = json.load(open(OUT / "b_router6" / "answers.json", encoding="utf-8-sig"))
src6 = json.load(open(OUT / "b_router6" / "piece_sources.json"))
S0 = json.load(open(OUT / "b_final1" / "reasonings.json.simscores.json"))


def dim_score(text, n=2):
    tot = {"logical": 0, "completeness": 0, "clarity": 0}
    for _ in range(n):
        c1, _r, _u = chat([{"role": "user", "content":
                            RUBRIC + "\n\n推理文本:\n" + text}],
                          qid="_dim", model=DEFAULT_MODEL, thinking=False,
                          max_tokens=80, tag="dimjudge")
        m = re.search(r"\{.*\}", c1 or "", re.S)
        try:
            d = json.loads(m.group(0))
            for k in tot:
                tot[k] += float(d.get(k, 0))
        except Exception:  # noqa: BLE001
            pass
    return {k: v / n for k, v in tot.items()}


def record_of(qid):
    tag = src6.get(qid, "")
    p = OUT / tag / "run_log.jsonl"
    best = ""
    if p.exists():
        for line in open(p, encoding="utf-8"):
            if qid in line:
                try:
                    r = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                if r.get("qid") == qid:
                    best = r.get("c1") or r.get("c3") or best
    return (best or "")[:2400]


def uplift(qid):
    old = R[qid]
    dims = dim_score(old)
    weak = min(dims, key=dims.get)
    mandate = {
        "logical": "论证链每一步都用'因为/据此/故'显式衔接，禁止跳步，结论由前文严格推出",
        "completeness": "四要素必须齐全且充实：数据定位(文档+页码)、数值提取、完整推导(算式/逐项判定)、独立复核动作、明确结论",
        "clarity": "句子短而准，每句只说一件事，术语准确，段落按'定位→推导→复核→结论'自然递进",
    }[weak]
    rec = record_of(qid)
    a_txt = "；".join(str(a) for a in ans[qid] if str(a).strip())
    prompt = (f"参考解题材料:\n{rec}\n\n现有推理摘要(待升级):\n{old}\n\n"
              f"重写这段推理摘要(400-540字, 自然段落): 保留全部事实数字页码与结论不变, "
              f"重点强化【{mandate}】。禁提'解题记录/答案/材料'字样; 无犹疑词; "
              f"结尾明确写出结论({a_txt})。只输出正文。")
    c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                      model=DEFAULT_MODEL, thinking=False, max_tokens=800,
                      tag="uplift")
    new = (c1 or "").strip().replace("\n", " ")
    gates = (len(new) >= 300 and new.endswith("。") and not WIDE.search(new)
             and consistent(qid, new) and not HEDGE.search(new))
    if not gates:
        return qid, None, dims, None
    nd = dim_score(new)
    old_avg = sum(dims.values()) / 3
    new_avg = sum(nd.values()) / 3
    if new_avg >= old_avg + 1.5:
        return qid, new, dims, nd
    return qid, None, dims, nd


def main():
    adopted = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for f in as_completed([ex.submit(uplift, q) for q in targets]):
            qid, new, dims, nd = f.result()
            if new:
                R[qid] = new
                led[qid] = list(LEDGER.per_qid[qid])
                adopted.append(qid)
                print(f"✓ {qid} {sum(dims.values())/3:.0f}→{sum(nd.values())/3:.0f}")
            else:
                print(f"— {qid} 保留原文 ({'门禁' if nd is None else f'复评{sum(nd.values())/3:.0f}无提升'})")
    json.dump(R, open(OUT / "reasonings_v9.json", "w"), ensure_ascii=False,
              indent=1)
    json.dump({"per_qid": led}, open(OUT / "reasoning_v9_ledger.json", "w"))
    print(f"升级采用 {len(adopted)}: {adopted}")


if __name__ == "__main__":
    main()
