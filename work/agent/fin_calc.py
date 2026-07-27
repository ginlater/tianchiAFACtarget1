"""确定性财务计算器（零 API token，词法取数 + Python 算术）。

合规定位：非模型、非 embedding 的确定性代码，从离线词法矿 fin_facts2 取数，
按标准财务公式精确计算，产出"确定性预算"证据块喂给 Qwen（Qwen 仍做最终答案
生成与口径判断）。目的：把 calc 的算术从 LLM（易错、耗 token）挪到代码（精确、
零成本），同时把精确数字确定性送入上下文（治愈取数漏检）。

覆盖：数据抽取干净的年报（比亚迪/宁德/中国移动等）；抽取有缺口的（美的合并表/
建筑扫描图）返回空，回退 Qwen 原路。开关 AFAC_FIN_CALC=1。
"""
import json
import pathlib
import re

_FF = None


def _mine():
    global _FF
    if _FF is None:
        p = (pathlib.Path(__file__).resolve().parents[1]
             / "processed_data" / "fin_facts2.json")
        _FF = json.load(open(p)) if p.exists() else {}
    return _FF


def mine_val(doc, item, period="本期"):
    """从 fin_facts2 词法取某指标某期的合并口径值（排除'流动'小计行）。"""
    for r in _mine().get(doc, []):
        head = r.split(":")[0]
        if item in r and "流动" not in head:
            for lab in (f"合并{period}", period):
                m = re.search(rf"{lab}=([\d,]+)", r)
                if m:
                    return float(m.group(1).replace(",", ""))
    return None


def debt_ratio(doc):
    tot = mine_val(doc, "资产总计")
    liab = mine_val(doc, "负债合计")
    if tot and liab and tot > 0:
        return liab / tot
    return None


def equity_multiplier(doc):
    dr = debt_ratio(doc)
    return 1 / (1 - dr) if dr is not None and dr < 1 else None


def yoy(doc_cur, doc_prev, item):
    """同比增速 = (本期-上期)/上期。优先单文档本期/上期，退双文档本期。"""
    cur = mine_val(doc_cur, item, "本期")
    prev = mine_val(doc_cur, item, "上期")
    if prev is None and doc_prev:
        prev = mine_val(doc_prev, item, "本期")
    if cur and prev and prev != 0:
        return (cur - prev) / abs(prev)
    return None


# 公司简称 → 年报 doc_id（本批固定语料的确定性映射，非答案键）
_DOC = {
    "比亚迪": "annual_byd_2025_report",
    "宁德时代": "annual_catl_2025_report",
    "美的集团": "annual_midea_2025_report",
    "招商银行": "annual_cmb_2025_report",
    "中国建筑": "annual_cscec_2025_report",
    "中国移动": "annual_chinamobile_2025_report",
}


def calc_facts_block(q):
    """为 fin 计算题产出'确定性预算'证据块：把矿里能精确取到的指标与代码算得的
    比率列出，标注口径与页码来源，供 Qwen 核对采用。取不到的项不列（不臆造）。"""
    if q.get("domain") != "financial_reports":
        return ""
    qtext = q.get("question", "")
    lines = []
    for name, doc in _DOC.items():
        if name not in qtext or doc not in (q.get("doc_ids") or []):
            continue
        parts = []
        for item, tag in [("营业收入", "营业收入"),
                          ("归属于母公司", "归母净利润"),
                          ("资产总计", "资产总计"),
                          ("负债合计", "负债合计"),
                          ("经营活动产生", "经营活动现金流净额")]:
            v = mine_val(doc, item)
            if v is not None:
                parts.append(f"{tag}(合并本期)={v:,.0f}")
        dr = debt_ratio(doc)
        if dr is not None:
            parts.append(f"资产负债率={dr * 100:.2f}%")
            parts.append(f"权益乘数=1/(1-资产负债率)={1 / (1 - dr):.4f}")
        if parts:
            lines.append(f"【{name}·确定性取数(词法矿fin_facts2)】" + "；".join(parts))
    if not lines:
        return ""
    return ("确定性预算(代码从报表精确取数与计算，请核对口径后采用；"
            "若与原文冲突以原文为准):\n" + "\n".join(lines))
