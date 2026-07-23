"""计算题/抽取题答题流程（B榜新题型，26/100）。

难点：需要精确定位原始数值（常在表格里），按题目口径计算，输出规范格式。
策略：宽证据检索 → 先抽取数值再计算（两阶段，防止边读边算出错）→
      独立第二样本 → 不一致时定向仲裁。
"""
import json, os, re

DEEP = os.environ.get("AFAC_DEEP") == "1"
SLIM = os.environ.get("AFAC_SLIM") == "1"

from . import retrieval
from .answerer import build_digest, DIGEST_DOMAINS, gather_evidence, _doc_title
from .qwen_client import chat, DEFAULT_MODEL

SLOT_DESC = {
    "number": "纯数字，保留两位小数，不带单位和千分位逗号",
    "percent": "百分数，形如 12.34%，保留两位小数",
    "ranking": "排序，用英文半角 > 连接，前后不加空格；公司名用题干中的简称原文",
    "date": "中文日期，形如 2026年1月1日",
    "text": "题目要求的完整文本，不加多余说明",
}

CALC_INST = """你是金融数据分析专家。请分两步作答：

第一步【取数】：从证据中逐条摘录计算所需的原始数值，每条注明来源页码与口径
（年度、公司、报表项目名）。只用证据中出现的数字，严禁用记忆或估算补数。
若某个必需数值在证据中找不到，最后一行输出：补充检索: <关键词>

第二步【计算】：写出算式并逐步计算。规则：
- 使用原始金额计算，中间过程不四舍五入，只在最终结果按要求保留两位小数
- "百分点"指两个百分比之差（如 12.5%-10.2%=2.3 个百分点），"百分比/增幅"指相对变化率
- 同比增幅 = (本期-上期)/|上期| × 100%
- 占比 = 部分/整体 × 100%
- 注意题目要求的单位（亿元/万元/元），必要时换算

答案格式要求：本题需要 {n} 个答案，依次为：
{slots}
最后一行必须严格输出（多个答案用中文分号；分隔，不写单位、不写解释）：
答案: {template}"""


def _slots_text(kinds):
    return "\n".join(f"  第{i+1}个：{SLOT_DESC.get(k, k)}"
                     for i, k in enumerate(kinds))


def _template(kinds):
    ex = {"number": "123.45", "percent": "12.34%",
          "ranking": "甲公司>乙公司", "date": "2026年1月1日", "text": "<文本>"}
    return "；".join(ex.get(k, "<答案>") for k in kinds)


ANS_RE = re.compile(r"^\s*答案[:：]\s*(.+)$", re.M)  # 行锚定：防推理中转述格式说明被误抓
SEARCH_RE = re.compile(r"补充检索[:：]\s*(.+)")

_SLOT_PAT = {
    "number": re.compile(r"-?[\d,]+(?:\.\d+)?"),
    "percent": re.compile(r"-?[\d.,]+\s*[%％]"),
    "date": re.compile(r"\d{4}年\d{1,2}月\d{1,2}日"),
    "ranking": re.compile(r".+>.+"),
}


def valid_calc(ans, kinds):
    """槽位格式校验：不合法视为解析失败走升级重试，而不是带垃圾提交。"""
    if not ans:
        return False
    parts = [p.strip() for p in re.split(r"[；;]", ans) if p.strip()]
    if len(parts) < len(kinds):
        return False
    for p, k in zip(parts, kinds):
        pat = _SLOT_PAT.get(k)
        if pat and not pat.fullmatch(p):
            return False
    return True


_TEMPLATE_ECHO = re.compile(r"甲公司|乙公司|123\.45|12\.34%|<文本>|<答案>")


def parse_calc(content):
    ms = list(ANS_RE.finditer(content or ""))
    for m in reversed(ms):
        v = m.group(1).strip()
        if v and not _TEMPLATE_ECHO.search(v):
            return v
    return ""


def calc_evidence(q, model=DEFAULT_MODEL, extra=(), cap_mult=1):
    """计算题证据：记忆卡（含关键财务数字）+ 宽检索原文片段。"""
    blocks = []
    domain = q["domain"]
    from .answerer import _use_digest
    if os.environ.get("AFAC_NO_DIGEST") == "1" and not _use_digest(domain):
        blocks.append("涉及文档:\n" + "\n".join(
            f"- {d}: 《{_doc_title(d)}》" for d in q["doc_ids"]))
    elif domain in DIGEST_DOMAINS:
        for d in q["doc_ids"]:
            blocks.append(build_digest(d, domain, model=model))
    else:
        blocks.append("涉及文档:\n" + "\n".join(
            f"- {d}: 《{_doc_title(d)}》" for d in q["doc_ids"]))
    # 年报计算题取数强化：三口径(主要数据表/分红两笔/母公司报表)强制齐全
    # （fin_b_005/012/016/019类伤：跨口径数值缺块）
    if domain == "financial_reports":
        extra = tuple(extra) + ("主要会计数据 财务指标",
                                "利润分配 中期分红 末期分红 每10股",
                                "母公司 资产负债表 所有者权益 少数股东")
    # 计算题证据要宽：数字常散落在多张表
    cap = ((2600 if os.environ.get("AFAC_SLIM4") == "1" else 6000)
           if os.environ.get("AFAC_NO_DIGEST") == "1"
           else (14000 if DEEP else (7000 if SLIM else 11000))
           + 2000 * max(0, len(q["doc_ids"]) - 2))
    ev, kept, _prot = gather_evidence(q, k_opt=4, k_q=5, cap=int(cap * cap_mult),
                                      extra_queries=extra)
    blocks.append("原文片段证据:\n" + ev)
    return "\n\n".join(blocks), [c["id"] for c in kept]


def answer_calc(q, kinds, model=DEFAULT_MODEL, log=None, verify_model=None,
                blind_mode=False):
    qid = q["qid"]
    inst = CALC_INST.format(n=len(kinds), slots=_slots_text(kinds),
                            template=_template(kinds))
    from .b_schema import is_date_question
    if is_date_question(q) and any(k == "number" for k in kinds):
        # 掐死数字槽逼出的自创日期编码（20260330/4.01类伤）：让模型输出中文日期，
        # fmt_slot 的确定性防护会自动转成 YYYYMMDD.00
        inst += ("\n注意：本题答案是一个日期。最后一行请输出完整中文日期"
                 "（YYYY年M月D日，如 2026年1月1日），系统会自动转换为要求的格式。")
    ev, ev_ids = calc_evidence(q, model=model)
    base = ev + "\n\n题目:\n" + q["question"] + "\n\n" + inst

    slim4 = os.environ.get("AFAC_SLIM4") == "1"
    c1, _t, _u = chat([{"role": "user", "content": base}], qid=qid,
                      model=model, thinking=True,
                      thinking_budget=(4000 if DEEP else (1100 if slim4 else 2000 if SLIM else 2800)),
                      max_tokens=(2600 if slim4 else 3600), tag="calc1")
    a1 = parse_calc(c1)

    # 升级重试触发：①报缺数 ②答案缺失/截断/槽位不合法（fin_b_014/016/017/019类伤）
    # 升级动作必须同时升证据帽（只升思考预算救不了证据缺数的自旋）
    ms = SEARCH_RE.search(c1)
    ok1 = valid_calc(a1, kinds)
    c1b = None
    # 瘦身档升级重试只救空白/报缺数（槽位小瑕疵触发的重试在瘦预算下烧token无收益）
    need_retry = (ms or not a1) if slim4 else (ms or not ok1)
    if need_retry:
        supp = ms.group(1).strip() if ms else ""
        if supp and blind_mode:  # 盲测下缺数可能因选错文档，允许域级扩检加选
            from .answerer import expand_docs_if_needed
            q2, added = expand_docs_if_needed(q, supp, model=model)
            if added:
                q = q2
                if log is not None:
                    log.write(json.dumps({"qid": qid, "doc_expanded": added},
                                         ensure_ascii=False) + "\n")
        ev2, ev_ids2 = calc_evidence(q, model=model,
                                     extra=([supp] if supp else ()), cap_mult=2)
        base = ev2 + "\n\n题目:\n" + q["question"] + "\n\n" + inst
        c1b, _t, _u = chat([{"role": "user", "content": base}], qid=qid,
                           model=model, thinking=True,
                           thinking_budget=(4000 if DEEP else (1600 if slim4 else 2800)),
                           max_tokens=(2600 if slim4 else 3600), tag="calc1b")
        a1b = parse_calc(c1b)
        if valid_calc(a1b, kinds) or (a1b and not a1):
            c1, a1, ev_ids = c1b, a1b, ev_ids2

    # 独立第二样本（异构模型更有信息量）；SLIM 跳过
    a2, c2 = None, None
    if not SLIM:
        c2, _t, _u = chat([{"role": "user", "content": base}], qid=qid,
                          model=verify_model or model, thinking=True,
                          thinking_budget=(4000 if DEEP else (2000 if SLIM else 2800)), max_tokens=3600, tag="calc2")
        a2 = parse_calc(c2)

    final, c3 = a1 or a2, None
    if a1 and a2 and _norm(a1) != _norm(a2):
        adj = (base + f"\n\n两次独立计算结果不同：\n甲: {a1}\n乙: {a2}\n"
               "请重新核对取数（页码、年度、口径是否对应）与算式，指出分歧原因，"
               "给出正确结果。最后一行仍按要求输出 答案: ")
        c3, _t, _u = chat([{"role": "user", "content": adj}], qid=qid,
                          model=verify_model or model, thinking=True,
                          thinking_budget=(4400 if DEEP else 3200), max_tokens=3600, tag="calc3")
        a3 = parse_calc(c3)
        final = a3 or a1
    if log is not None:
        log.write(json.dumps({"qid": qid, "final": final, "a1": a1, "a2": a2,
                              "c1": c1, "c1b": c1b, "c2": c2, "c3": c3,
                              "evidence_ids": ev_ids[:40]},
                             ensure_ascii=False) + "\n")
        log.flush()
    return final


def _norm(s):
    return re.sub(r"[\s,，]", "", s or "")
