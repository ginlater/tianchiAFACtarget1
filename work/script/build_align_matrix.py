#!/usr/bin/env python3
"""跨文档实体对齐矩阵 v1（瓶颈突破实验：慢性题全是跨文档比较题）。

方法论: 在失败点diff模型视野与原始文档 → 缺口=跨文档对齐+派生量现场计算
结构: ①fin 公司×年度分红矩阵(中期/末期/全年预计算) ②ins 产品×条款存在性矩阵
产物: processed_data/align_matrix.json（离线词法+确定性算术，零token合规）。
"""
import argparse
import json
import pathlib
import re

WORK = pathlib.Path(__file__).resolve().parents[1]
PD = WORK / "processed_data"

# ---------- fin: 每10股分红矩阵(全年=中期+末期 预计算) ----------
DIV_PAT = re.compile(
    r"每\s*10\s*股[^0-9]{0,14}?([\d.]+)\s*元")
MID_HINT = re.compile(r"中期|半年度")
FIN_HINT = re.compile(r"末期|年末|年度利润分配(方案|预案)")
YEAR_HINT = re.compile(r"(20\d{2})\s*年")


def fin_dividends(raw_root=None):
    """宽窗口候选证据包: 预计算只在无歧义时做, 歧义交给模型带完整口径上下文决策。"""
    import fitz
    raw_reports = pathlib.Path(raw_root or
                               WORK.parent / "public_dataset_upload" / "raw") / "financial_reports"
    out = {}
    pdfs = sorted(set(raw_reports.glob("*.PDF")) | set(raw_reports.glob("*.pdf")),
                  key=lambda path: path.as_posix())
    for pdf in pdfs:
        m = re.match(r"annual_(\w+?)_(\d{4})_report", pdf.stem)
        if not m:
            continue
        comp, yr = m.group(1), m.group(2)
        stmts, seen = [], set()
        with fitz.open(pdf) as document:
            for pno in range(len(document)):
                text = document[pno].get_text().replace("\n", " ")
                compact = re.sub(r"\s+", "", text)
                for match in re.finditer(r"每10股[^。]{0,80}?([\d.]+)元", compact):
                    start = max(0, match.start() - 90)
                    context = compact[start:match.end() + 40]
                    key = match.group(1)
                    if (key, context[:50]) in seen:
                        continue
                    seen.add((key, context[:50]))
                    stmts.append(f"[P{pno + 1}] …{context}…")
                for match in re.finditer(r"每股[^。]{0,40}?([\d.]+)元", compact):
                    if "每10股" in compact[max(0, match.start() - 6):match.end()]:
                        continue
                    start = max(0, match.start() - 70)
                    context = compact[start:match.end() + 30]
                    if re.search(r"股息|分红|派", context):
                        key = ("ps", match.group(1), context[:40])
                        if key in seen:
                            continue
                        seen.add(key)
                        stmts.append(f"[P{pno + 1}][每股口径] …{context}…")
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


def ins_matrix(processed_dir=PD):
    processed_dir = pathlib.Path(processed_dir)
    df = json.loads((processed_dir / "domain_facts.json").read_text(encoding="utf-8"))
    titles = json.loads((processed_dir / "insurance_titles.json").read_text(encoding="utf-8"))
    out = {}
    for doc, rows in df.items():
        if not doc.isdigit():
            continue
        identity = titles.get(doc, doc)
        if isinstance(identity, dict):
            identity_parts = [identity.get("company", ""),
                              identity.get("product", "")]
            aliases = identity.get("alias", identity.get("aliases", [])) or []
            if aliases:
                identity_parts.append("别名=" + "/".join(aliases))
            name = "｜".join(part for part in identity_parts if part) or doc
        else:
            name = identity
        ent = {}
        for cl, pat in CLAUSES.items():
            hit = [r for r in rows if re.search(pat, r)]
            ent[cl] = (f"有({hit[0][:70]})" if hit else "未见")
        out[f"{doc}:{name}"] = ent
    return out


def build(raw_root=None, processed_dir=PD, output_path=None):
    processed_dir = pathlib.Path(processed_dir)
    result = {"fin_dividends": fin_dividends(raw_root),
              "ins_clauses": ins_matrix(processed_dir)}
    target = pathlib.Path(output_path or processed_dir / "align_matrix.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n",
                      encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=pathlib.Path,
                        default=WORK.parent / "public_dataset_upload" / "raw")
    parser.add_argument("--processed", type=pathlib.Path, default=PD)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    result = build(args.input, args.processed, args.output)
    fd = result["fin_dividends"]
    print("fin分红证据包:", {k: len(v) for k, v in fd.items()})
    print("ins条款矩阵:", len(result["ins_clauses"]), "产品")


if __name__ == "__main__":
    main()
