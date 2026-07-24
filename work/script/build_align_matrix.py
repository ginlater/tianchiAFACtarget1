#!/usr/bin/env python3
"""跨文档实体对齐矩阵 v1（瓶颈突破实验：慢性题全是跨文档比较题）。

方法论: 在失败点diff模型视野与原始文档 → 缺口=跨文档对齐+派生量现场计算
结构: ①fin 公司×年度分红矩阵(中期/末期/全年预计算) ②ins 产品×条款存在性矩阵
产物: processed_data/align_matrix.json（离线词法+确定性算术，零token合规）。
"""
import json, pathlib, re

WORK = pathlib.Path(__file__).resolve().parents[1]
PD = WORK / "processed_data"

# ---------- fin: 每10股分红矩阵(全年=中期+末期 预计算) ----------
DIV_PAT = re.compile(
    r"每\s*10\s*股[^0-9]{0,14}?([\d.]+)\s*元")
MID_HINT = re.compile(r"中期|半年度")
FIN_HINT = re.compile(r"末期|年末|年度利润分配(方案|预案)")
YEAR_HINT = re.compile(r"(20\d{2})\s*年")


def fin_dividends():
    """宽窗口候选证据包: 预计算只在无歧义时做, 歧义交给模型带完整口径上下文决策。"""
    import fitz
    RAW = WORK.parent / "public_dataset_upload" / "raw" / "financial_reports"
    out = {}
    for pdf in sorted(list(RAW.glob("*.PDF")) + list(RAW.glob("*.pdf"))):
        m = re.match(r"annual_(\w+?)_(\d{4})_report", pdf.stem)
        if not m:
            continue
        comp, yr = m.group(1), m.group(2)
        d = fitz.open(pdf)
        stmts, seen = [], set()
        for pno in range(len(d)):
            t = d[pno].get_text().replace("\n", " ")
            t2 = re.sub(r"\s+", "", t)
            for mm in re.finditer(r"每10股[^。]{0,80}?([\d.]+)元", t2):
                s = max(0, mm.start() - 90)
                ctx = t2[s:mm.end() + 40]
                k = mm.group(1)
                if (k, ctx[:50]) in seen:
                    continue
                seen.add((k, ctx[:50]))
                stmts.append(f"[P{pno+1}] …{ctx}…")
            for mm in re.finditer(r"每股[^。]{0,40}?([\d.]+)元", t2):
                if "每10股" in t2[max(0, mm.start()-6):mm.end()]:
                    continue
                s = max(0, mm.start() - 70)
                ctx = t2[s:mm.end() + 30]
                if re.search(r"股息|分红|派", ctx):
                    k = ("ps", mm.group(1), ctx[:40])
                    if k in seen:
                        continue
                    seen.add(k)
                    stmts.append(f"[P{pno+1}][每股口径] …{ctx}…")
        if stmts:
            out[f"{comp}_{yr}"] = stmts[:12]
    return out


# ---------- ins: 产品×条款存在性矩阵 ----------
CLAUSES = {
    "未成年人身故限制": r"未成年人身故",
    "自杀免责(2年)": r"2\s*年内自杀|二年内自杀",
    "犹豫期": r"犹豫期",
    "宽限期": r"宽限期",
    "保单借款": r"借款|贷款",
    "施救费用": r"施救费",
    "复效": r"复效",
    "减保/部分领取": r"部分领取|减保",
}


def ins_matrix():
    df = json.load(open(PD / "domain_facts.json"))
    titles = json.load(open(PD / "insurance_titles.json"))
    out = {}
    for doc, rows in df.items():
        if not doc.isdigit():
            continue
        name = titles.get(doc, doc)
        ent = {}
        for cl, pat in CLAUSES.items():
            hit = [r for r in rows if re.search(pat, r)]
            ent[cl] = (f"有({hit[0][:70]})" if hit else "未见")
        out[f"{doc}:{name}"] = ent
    return out


def main():
    result = {"fin_dividends": fin_dividends(), "ins_clauses": ins_matrix()}
    json.dump(result, open(PD / "align_matrix.json", "w"),
              ensure_ascii=False, indent=1)
    fd = result["fin_dividends"]
    print("fin分红证据包:", {k: len(v) for k, v in fd.items()})
    print("ins条款矩阵:", len(result["ins_clauses"]), "产品")


if __name__ == "__main__":
    main()
