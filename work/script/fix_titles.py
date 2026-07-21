#!/usr/bin/env python3
"""重算文档标题并更新 docs_meta.json。

PDF: 版面分析——取前两页中字号最大的文本行（规则允许的预处理）。
TXT: 文件名即完整法规名。HTML: 保留 ArticleTitle。
"""
import json, pathlib, re

import fitz

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROC = ROOT / "work" / "processed_data"
fitz.TOOLS.mupdf_display_errors(False)

SUFFIX = re.compile(
    r"(条款|办法|规定|决定|报告书|说明书|年度报告|全文|摘要|指引|准则|细则|"
    r"保险|年金|寿险|研究报告|点评|深度报告)$")
NOISE = re.compile(r"^(第[一二三四五六七八九十百]+[条章]|目\s*录|[\d\s.]+$|"
                   r"注册号|阅读指引|总\s*则)")


def pdf_title(path):
    doc = fitz.open(path)
    best = []  # (size, text)
    for pno in range(min(2, doc.page_count)):
        d = doc[pno].get_text("dict")
        for block in d["blocks"]:
            for line in block.get("lines", []):
                txt = "".join(s["text"] for s in line["spans"]).strip()
                if not txt or len(txt) < 5 or len(txt) > 60 or NOISE.match(txt):
                    continue
                size = max(s["size"] for s in line["spans"])
                best.append((round(size, 1), pno, block["number"], txt))
        if best:
            break
    doc.close()
    if not best:
        return None
    maxsize = max(b[0] for b in best)
    # 同为最大字号的相邻行拼接（如 公司名 + 报告名 两行大标题）
    top = [b for b in best if b[0] >= maxsize - 0.6]
    title = "".join(b[3] for b in top[:3])
    return title[:70] if len(title) >= 5 else None


PRODUCT = re.compile(
    r"[一-鿿A-Za-z0-9]{2,18}(?:养老年金保险|年金保险|终身寿险|两全保险|养老保险|"
    r"人寿保险|财产保险|责任保险|医疗保险|疾病保险|意外伤害保险)"
    r"(?:（[^）]{1,14}）){0,2}")
DOCNAME = re.compile(r"^[^。；]{0,40}(募集说明书|重大资产重组报告书|报告书)[^。；]{0,10}$")


def insurance_product_name(text):
    from collections import Counter
    names = Counter(PRODUCT.findall(text))
    if not names:
        return None
    # 权重 = 频次 × 长度（偏好完整产品名），出现≥3次才可信
    best, score = None, 0
    for name, cnt in names.items():
        if cnt < 3 or "本合同" in name or "条款" in name:
            continue
        s = cnt * len(name)
        if s > score:
            best, score = name, s
    return best


meta = json.load(open(PROC / "docs_meta.json"))
changed = 0
for doc_id, m in meta.items():
    src = ROOT / m["src"]
    new_title = None
    if src.suffix.lower() == ".pdf":
        new_title = pdf_title(src)
        if not new_title:
            # 兜底：正文前14行中以标题后缀结尾的最长行
            text = (PROC / m["domain"] / f"{doc_id}.txt").read_text("utf-8")
            lines = [l.strip() for l in text[:1500].split("\n")
                     if l.strip() and not l.strip().startswith("[P")][:14]
            cands = [l for l in lines if SUFFIX.search(l) and not NOISE.match(l)
                     and 6 <= len(l) <= 60]
            new_title = max(cands, key=len) if cands else None
    elif src.suffix == ".txt":
        new_title = re.sub(r"^strict[_a-z0-9]*?_\d+_", "", src.stem)

    # 领域修正
    text = (PROC / m["domain"] / f"{doc_id}.txt").read_text("utf-8")
    if m["domain"] == "insurance":
        cur = new_title or m["title"]
        # 字号标题已含产品名则保留，否则用全文最高频产品名
        if not PRODUCT.search(cur):
            prod = insurance_product_name(text)
            if prod:
                new_title = prod + "条款"
    elif m["domain"] == "financial_reports":
        y = re.search(r"(20\d\d)", doc_id)
        base = (new_title or m["title"]).split("20")[0]
        if y:
            new_title = f"{base}{y.group(1)}年年度报告"
    elif m["domain"] == "financial_contracts" and new_title:
        if "说明书" not in new_title and "报告书" not in new_title:
            lines = [l.strip() for l in text[:2000].split("\n") if l.strip()][:20]
            extra = next((l for l in lines if DOCNAME.match(l)), None)
            if extra and extra not in new_title:
                new_title = (new_title + extra)[:80]

    if new_title and new_title != m["title"]:
        m["title"] = new_title
        changed += 1
json.dump(meta, open(PROC / "docs_meta.json", "w"), ensure_ascii=False, indent=1)
print("titles updated:", changed)
for d in ["1", "2", "11", "13", "15", "16", "text07", "text01",
          "csrc_0027_att1", "csrc_0271", "annual_byd_2024_report",
          "pack2_text01"]:
    print(f"  {d} -> {meta[d]['title']}")
