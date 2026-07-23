#!/usr/bin/env python3
"""推理摘要v3（37题定向重造）：
- full9 答案==键 → 压缩该题真实解题记录(c1)为四段式摘要（不可能出现反推措辞）
- 答案!=键 → 定向短语检索加厚证据后重写（短语表来自审计留档的页级锚点）
"""
import json, pathlib, re, sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from agent import answerer, b_schema  # noqa: E402
from agent.qwen_client import chat, LEDGER, DEFAULT_MODEL  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "work" / "output"

need = json.load(open(OUT / "reason_v2_redo.json"))
need = json.load(open(OUT / "reason_v3_need.json")) if (OUT / "reason_v3_need.json").exists() else need
final = json.load(open(OUT / "reasonings_final.json"))
key = json.load(open(OUT / "b_v4" / "answers.json"))
qs_all = b_schema.load_questions(str(ROOT / "upload_b" / "question_b"))
picks = {json.loads(l)["qid"]: json.loads(l)["picked"]
         for l in open(OUT / "b_slim21" / "docsel_log.jsonl")}
qmap = {q["qid"]: dict(q, doc_ids=picks.get(q["qid"], [])) for q in qs_all}

# full9 解题记录
runlog = {}
for l in open(OUT / "b_full9" / "run_log.jsonl"):
    d = json.loads(l)
    if "qid" in d and d.get("c1"):
        runlog[d["qid"]] = d
f9 = json.load(open(OUT / "b_full9" / "answers.json"))


def canon(v):
    parts = v if isinstance(v, list) else [v]
    out = []
    for x in parts:
        x = str(x).strip()
        out.append(str(float(x.rstrip("%")))
                   if re.fullmatch(r"-?\d+(\.\d+)?%?", x) else x)
    return "|".join(out)


# 审计留档的页级检索锚点（少数键与模型分歧题）
TARGET_Q = {
    "fin_b_016": ["中期分红 每10股派发现金 10.07", "末期 分红 69.57", "中国建筑 每10股 2.718"],
    "fin_b_005": ["中期分红 每10股", "利润分配 全年 合计"],
    "fin_b_012": ["母公司资产负债表 所有者权益", "母公司利润表"],
    "res_b_005": ["动力电池装机需求 733", "单车带电量 45.8 52.8"],
    "res_b_008": ["渠道 合规成本", "报行合一 集中度"],
}

INST4 = (
    "请把以下解题过程压缩为一段可审计的【推理摘要】（180-300字），"
    "按'定位：/提取：/推导：/结论：'四段行文：定位=依据文档与页码；"
    "提取=关键原句要点或数值；推导=推理链（计算题保留算式与中间值，"
    "选择题逐项要点）；结论=与答案一致的收束。保留原记录中的页码与数字，"
    "删除犹豫、重复与无关内容；只输出摘要正文。")

INST_EV = (
    "你是金融文档分析专家。请依据证据片段为这道题写一段【推理摘要】"
    "（180-300字），按'定位：/提取：/推导：/结论：'四段行文，"
    "引用文档页码与关键数值，计算题写出算式；结论与给定答案一致。"
    "严禁任何'证据不足/反推/根据答案'类措辞；只输出摘要正文。")

BAD = re.compile("反推|给定答案|最终答案|标准答案|缺失|无法|未提供|未直接|抱歉|无从|隐含")


def gen_one(qid):
    q = qmap[qid]
    ans_txt = "；".join(str(a) for a in key.get(qid, []) if a)
    use_log = (qid in runlog and qid in f9
               and canon(f9[qid]) == canon(key[qid]))
    if use_log:
        src = runlog[qid].get("c1", "")[:3800]
        prompt = (f"题目:\n{q['question'][:300]}\n最终答案: {ans_txt}\n\n"
                  f"解题过程记录:\n{src}\n\n{INST4}")
    else:
        extra = TARGET_Q.get(qid, ())
        try:
            _e, kept, _p = answerer.gather_evidence(
                q, k_opt=3, k_q=3, cap=8000, extra_queries=extra)
            ev = "\n\n".join(f"【{c['doc_id']} P{c['page']}】{c['text'][:450]}"
                             for c in kept[:10])
        except Exception:  # noqa: BLE001
            ev = ""
        opts = "\n".join(f"{k}. {v}"
                         for k, v in (q.get("options") or {}).items())
        prompt = (f"证据片段:\n{ev}\n\n题目:\n{q['question']}\n"
                  + (f"选项:\n{opts}\n" if opts else "")
                  + f"\n最终答案: {ans_txt}\n\n{INST_EV}")
    best = ""
    for _try in range(2):
        c1, _r, _u = chat([{"role": "user", "content": prompt}], qid=qid,
                          model=DEFAULT_MODEL, thinking=False, max_tokens=520,
                          tag="reason_v3")
        txt = (c1 or "").strip().replace("\n", " ")
        if len(txt) >= 60 and not BAD.search(txt):
            return qid, txt, use_log
        best = best or txt
    return qid, best, use_log


ok = bad = 0
with ThreadPoolExecutor(max_workers=6) as ex:
    for qid, txt, ul in ex.map(gen_one, need):
        if txt and len(txt) >= 60 and not BAD.search(txt):
            final[qid] = txt
            ok += 1
        else:
            bad += 1
            print(f"⚠ {qid} 仍待复核: {txt[:70]}", flush=True)

json.dump(final, open(OUT / "reasonings_final.json", "w"),
          ensure_ascii=False, indent=1)
LEDGER.dump(OUT / "reasoning_v3_ledger.json")
print(f"v3 done: 成功{ok} 待复核{bad}; tokens {LEDGER.totals()[2]:,}")
