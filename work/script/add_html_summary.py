#!/usr/bin/env python3
"""为 regulatory HTML 文档补充 meta description（含当事人/文号）到 docs_meta.json。

纯结构化字段抽取（网页自带 meta），无语义模型参与，符合预处理边界。
用于文档盲检卡片——处罚决定书标题高度雷同，当事人信息才有区分度。
"""
import json, pathlib, re

from bs4 import BeautifulSoup

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROC = ROOT / "work" / "processed_data"

meta = json.load(open(PROC / "docs_meta.json"))
n = 0
for doc_id, m in meta.items():
    src = ROOT / m["src"]
    if src.suffix != ".html":
        continue
    soup = BeautifulSoup(src.read_text(encoding="utf-8", errors="ignore"), "lxml")
    desc = ""
    for tag in soup.find_all("meta"):
        if (tag.get("name") or "").lower() == "description":
            c = (tag.get("content") or "").strip()
            if len(c) > len(desc):
                desc = c
    if desc:
        m["summary"] = re.sub(r"\s+", " ", desc)[:160]
        n += 1
json.dump(meta, open(PROC / "docs_meta.json", "w"), ensure_ascii=False, indent=1)
print(f"html summary added: {n}")
print("样例 csrc_0271:", meta["csrc_0271"].get("summary", "")[:120])
