#!/usr/bin/env python3
"""解析全部原始文档 → work/processed_data/{domain}/{doc_id}.txt + docs_meta.json

- PDF: PyMuPDF 逐页抽取，页首插入 [P{n}] 标记（供证据定位与按页读取）
- HTML(csrc): bs4 抽取正文 + meta(ArticleTitle/PubDate/ColumnName)
- TXT: 原样复制
纯结构化处理，无任何语义模型参与（符合赛规预处理边界）。
"""
import json, pathlib, re, sys

import fitz  # pymupdf
from bs4 import BeautifulSoup

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "public_dataset_upload" / "raw"
OUT = ROOT / "work" / "processed_data"

fitz.TOOLS.mupdf_display_errors(False)


def clean_page(t: str) -> str:
    t = t.replace(" ", " ").replace("", "·")
    lines = [ln.rstrip() for ln in t.split("\n")]
    # 合并竖排侧栏：连续多行单字符（研报侧栏"股\n票\n研\n究"）压成一行
    merged, buf = [], []
    for ln in lines:
        s = ln.strip()
        if len(s) == 1 and not s.isdigit():
            buf.append(s)
            continue
        if buf:
            merged.append("".join(buf) if len(buf) >= 3 else "\n".join(buf))
            buf = []
        merged.append(ln)
    if buf:
        merged.append("".join(buf) if len(buf) >= 3 else "\n".join(buf))
    t = "\n".join(merged)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def parse_pdf(path: pathlib.Path):
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc):
        txt = clean_page(page.get_text())
        pages.append(f"[P{i+1}]\n{txt}")
    full = "\n\n".join(pages)
    # 标题：第一页前几行中最长的非空行
    first_lines = [l.strip() for l in doc[0].get_text().split("\n") if l.strip()][:8]
    title = max(first_lines, key=len) if first_lines else path.stem
    meta = {"n_pages": doc.page_count, "title": title}
    doc.close()
    return full, meta


def parse_html(path: pathlib.Path):
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    metas = {}
    for m in soup.find_all("meta"):
        name = (m.get("name") or "").strip()
        if name in ("ArticleTitle", "PubDate", "ColumnName", "description", "Description"):
            c = (m.get("content") or "").strip()
            if c and name not in metas:
                metas[name] = c
    for tag in soup(["script", "style", "head"]):
        tag.decompose()
    body = soup.get_text("\n", strip=True)
    body = re.sub(r"\n{3,}", "\n\n", body)
    title = metas.get("ArticleTitle", path.stem)
    header = f"标题：{title}\n栏目：{metas.get('ColumnName','')}\n发布日期：{metas.get('PubDate','')}\n"
    return header + "\n" + body, {"n_pages": None, "title": title,
                                   "pub_date": metas.get("PubDate", ""),
                                   "column": metas.get("ColumnName", "")}


def out_write(domain: str, doc_id: str, text: str):
    d = OUT / domain
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc_id}.txt").write_text(text, encoding="utf-8")


def main():
    all_meta = {}
    jobs = []  # (domain, doc_id, path, kind)
    for p in sorted((RAW / "insurance").glob("*.pdf")):
        jobs.append(("insurance", p.stem, p, "pdf"))
    for p in sorted((RAW / "financial_contracts").glob("*.pdf")):
        jobs.append(("financial_contracts", p.stem, p, "pdf"))
    for p in sorted(RAW.glob("financial_reports/*.[pP][dD][fF]")):
        jobs.append(("financial_reports", p.stem, p, "pdf"))
    for p in sorted((RAW / "research").glob("*.pdf")):
        jobs.append(("research", p.stem, p, "pdf"))
    reg = RAW / "regulatory"
    for p in sorted(reg.glob("txt/*.txt")):
        jobs.append(("regulatory", p.stem, p, "txt"))
    for p in sorted(reg.glob("html/*.html")):
        jobs.append(("regulatory", p.stem, p, "html"))
    for p in sorted(reg.glob("attachments/*.pdf")):
        jobs.append(("regulatory", p.stem, p, "pdf"))

    print(f"total jobs: {len(jobs)}", flush=True)
    for n, (domain, doc_id, path, kind) in enumerate(jobs):
        try:
            if kind == "pdf":
                text, meta = parse_pdf(path)
            elif kind == "html":
                text, meta = parse_html(path)
            else:
                text = path.read_text(encoding="utf-8", errors="ignore")
                meta = {"n_pages": None, "title": doc_id.split("_", 3)[-1][:80]}
            out_write(domain, doc_id, text)
            meta.update({"domain": domain, "doc_id": doc_id,
                         "src": str(path.relative_to(ROOT)), "n_chars": len(text)})
            all_meta[doc_id] = meta
        except Exception as e:
            print(f"ERROR {path}: {e}", flush=True)
        if (n + 1) % 50 == 0:
            print(f"  {n+1}/{len(jobs)} done", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(all_meta, open(OUT / "docs_meta.json", "w"), ensure_ascii=False, indent=1)
    by_domain = {}
    for m in all_meta.values():
        by_domain.setdefault(m["domain"], [0, 0])
        by_domain[m["domain"]][0] += 1
        by_domain[m["domain"]][1] += m["n_chars"]
    for d, (c, ch) in sorted(by_domain.items()):
        print(f"{d}: {c} docs, {ch/1e6:.1f}M chars")
    print("META OK", len(all_meta))


if __name__ == "__main__":
    main()
