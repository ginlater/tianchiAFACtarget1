#!/usr/bin/env python3
"""fin事实表v2：主报表单元格级抽取，恢复列身份（合并/公司 × 本期/上期）。

根因（fin_b_012尸检）：美的等报告用"合并及公司"四栏双口径报表，扁平解析后
列身份丢失 → 模型面对数字汤无法分辨 合并/母公司。
方法：fitz词坐标 → y聚类成行 → 数字单元格x聚类成列 → 表头绑定列身份。
产物：processed_data/fin_facts2.json {doc_id: [事实行...]}（离线词法，合规）。
"""
import json, pathlib, re, sys

import fitz

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "public_dataset_upload" / "raw" / "financial_reports"
OUT = ROOT / "work" / "processed_data" / "fin_facts2.json"

TITLE_RE = re.compile(
    r"(合并及公司|合并及母公司|合并|母公司|公司)(资产负债表|利润表|损益表|现金流量表)")
NUM_RE = re.compile(r"^\(?-?[\d,]+(?:\.\d+)?\)?$|^-$")
YEAR_RE = re.compile(r"20\d{2}")


def page_rows(page):
    """词坐标 → 行（y中心连续分组, 容差4.5pt——中西文基线不同y0会漂）。"""
    words = sorted(page.get_text("words"),
                   key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    out, cur, last_y = [], [], None
    for x0, y0, x1, y1, w, *_ in words:
        yc = (y0 + y1) / 2
        if last_y is not None and yc - last_y > 4.5:
            out.append(sorted(cur))
            cur = []
        cur.append((x0, x1, w))
        last_y = yc
    if cur:
        out.append(sorted(cur))
    return out


def cluster_columns(xs, gap=28):
    """数字单元格x坐标 → 列中心（间距聚类）。"""
    if not xs:
        return []
    xs = sorted(xs)
    centers, cur = [], [xs[0]]
    for x in xs[1:]:
        if x - cur[-1] > gap:
            centers.append(sum(cur) / len(cur))
            cur = [x]
        else:
            cur.append(x)
    centers.append(sum(cur) / len(cur))
    return centers


def extract_statement(page, title):
    """一页主报表 → 事实行列表。"""
    rows = page_rows(page)
    # 真表门槛：审计报告等散文页会误匹配表头字样, 数字单元格<15视为非表
    n_num = sum(1 for row in rows for x0, x1, w in row
                if NUM_RE.match(w) and any(c.isdigit() for c in w))
    if n_num < 15:
        return []
    # 收集全页数字单元格x
    numx = []
    for row in rows:
        for x0, x1, w in row:
            if NUM_RE.match(w) and any(c.isdigit() for c in w):
                numx.append(x1)  # 右缘聚类：财务表右对齐, x0随位数漂移
    cols = cluster_columns(numx)
    if len(cols) < 2:
        return []
    dual = "及" in title  # 合并及公司 → 4值列
    ncols = 4 if dual else 2
    # 取x最大的ncols列为值列（左侧可能有附注列混入数字）
    vcols = cols[-ncols:] if len(cols) >= ncols else cols
    if dual:
        heads = ["合并本期", "合并上期", "公司本期", "公司上期"]
    elif title.startswith("母公司") or title.startswith("公司"):
        heads = ["母公司本期", "母公司上期"]
    else:
        heads = ["合并本期", "合并上期"]
    heads = heads[:len(vcols)]
    facts = []
    for row in rows:
        label_parts, cells = [], {}
        for x0, x1, w in row:
            if (NUM_RE.match(w) and (any(c.isdigit() for c in w) or w == "-")
                    and x1 >= vcols[0] - 60):
                ci = min(range(len(vcols)), key=lambda i: abs(vcols[i] - x1))
                cells.setdefault(ci, w)
            elif not NUM_RE.match(w):
                label_parts.append(w)
        label = "".join(label_parts).strip()
        label = re.sub(r"附注|[一二三四五六七八九十]+\([\d一-鿿()a-z]*\)?", "", label)
        label = label.strip("|、 ")
        if not cells or not label or len(label) > 40:
            continue
        if not re.search(r"[一-鿿]{2}", label):
            continue
        if "。" in label or "审计" in label or "错报" in label:
            continue  # 散文行
        vals = " ".join(f"{heads[i]}={cells[i]}" for i in sorted(cells)
                        if i < len(heads))
        facts.append(f"[{title}] {label}: {vals}")
    return facts


def dividend_lines(doc):
    """利润分配/每10股分红 原文行（fin_b_005类：中期+末期两笔）。"""
    out = []
    for pno in range(len(doc)):
        t = doc[pno].get_text()
        if "每10股" not in t.replace(" ", "") and "每 10 股" not in t:
            continue
        for ln in t.splitlines():
            l2 = ln.replace(" ", "")
            if "每10股" in l2 and ("派" in l2 or "红利" in l2 or "股息" in l2):
                out.append(f"[P{pno+1}分红] {ln.strip()[:120]}")
    # 去重保序
    seen, ded = set(), []
    for l in out:
        k = l.split("]")[1]
        if k not in seen:
            seen.add(k)
            ded.append(l)
    return ded[:20]


def main():
    result = {}
    for pdf in sorted(list(RAW.glob("*.PDF")) + list(RAW.glob("*.pdf"))):
        doc_id = pdf.stem
        d = fitz.open(pdf)
        facts = []
        carry, carry_p = None, -9
        for pno in range(len(d)):
            head = d[pno].get_text()[:400]
            m = TITLE_RE.search(head.replace(" ", ""))
            if m:
                title = m.group(0)
                yrs = YEAR_RE.findall(head)
                ttl = f"{title}{'/'.join(yrs[:1])}" if yrs else title
                carry, carry_p = ttl, pno
            elif carry and pno - carry_p <= 3:
                # 报表续页无表头(CATL式)：3页内继承前表头
                ttl = carry + "(续)"
            else:
                continue
            got = [f + f" (P{pno+1})" for f in extract_statement(d[pno], ttl)]
            if m and len(got) < 4:
                carry, carry_p = None, -9  # 假表头(审计散文页): 不外溢
            facts += got
        facts += dividend_lines(d)
        result[doc_id] = facts
        print(f"{doc_id}: {len(facts)}行", flush=True)
    json.dump(result, open(OUT, "w"), ensure_ascii=False, indent=0)
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
