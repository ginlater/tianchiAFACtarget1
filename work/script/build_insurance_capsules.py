#!/usr/bin/env python3
"""Build deterministic typed-memory capsules for parsed insurance clauses.

Inputs (no model calls):
  processed_data/insurance/*.txt
  processed_data/domain_facts.json
  processed_data/insurance_titles.json

Output:
  processed_data/insurance_capsules.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from collections import Counter

WORK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK))

from agent.insurance_capsules import (  # noqa: E402
    SCHEMA_VERSION, extract_numbers, infer_topics,
)

PD = WORK / "processed_data"
INS_DIR = PD / "insurance"
DEFAULT_OUT = PD / "insurance_capsules.json"

PAGE_RE = re.compile(r"^\[P(\d+)\]\s*$")
CN_CLAUSE_RE = re.compile(r"^(第[零〇一二三四五六七八九十百千万两\d]+条)\s*(.*)$")
DEC_CLAUSE_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){1,2})\s*(.*)$")
DOTS_RE = re.compile(r"[.。·…]{5,}")
SECTION_NAMES = {
    "保险责任", "责任免除", "赔偿处理", "保险金额", "责任限额与免赔额（率）",
    "投保人、被保险人义务", "保险人义务", "争议处理和法律适用", "其他事项",
    "总则", "释义", "保险期间", "保险费", "合同解除", "现金价值权益",
}
BODY_PREFIX = re.compile(
    r"^(?:本合同|本保险|保险期间|投保人|被保险人|保险人|我们|您|在|若|如|除|"
    r"发生|出现|下列|对于|订立|知道|因|自|按照|主车|赔款|合同)")
PAGE_HEADER_RE = re.compile(
    r"条款（第[零〇一二三四五六七八九十百千万两\d]+页）$|"
    r"^(?:请扫描以查询验证条款|险种简称[:：]|险种代码[:：])")


def _normal(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").replace("％", "%"))


def _split_pages(text: str) -> list[tuple[int, list[str]]]:
    pages = []
    page = 0
    lines = []
    for raw in text.splitlines():
        match = PAGE_RE.match(raw.strip())
        if match:
            if page or lines:
                pages.append((page, lines))
            page = int(match.group(1))
            lines = []
        else:
            lines.append(raw.rstrip())
    if page or lines:
        pages.append((page, lines))
    return pages


def _is_toc(lines: list[str]) -> bool:
    text = "\n".join(lines)
    headers = sum(bool(CN_CLAUSE_RE.match(line.strip()) or
                       DEC_CLAUSE_RE.match(line.strip())) for line in lines)
    return "条款目录" in text or (headers >= 7 and len(DOTS_RE.findall(text)) >= 3)


def _looks_like_title(rest: str, *, decimal=False) -> bool:
    rest = rest.strip()
    if not rest or len(rest) > (50 if decimal else 32):
        return False
    if re.search(r"[。；，：]", rest):
        return False
    return True


def _split_chunks(lines: list[str], max_chars: int) -> list[str]:
    atoms = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if len(line) <= max_chars:
            atoms.append(line)
            continue
        sentences = [x for x in re.split(r"(?<=[。；！？])", line) if x]
        for sentence in sentences:
            if len(sentence) <= max_chars:
                atoms.append(sentence)
            else:
                atoms.extend(sentence[i:i + max_chars]
                             for i in range(0, len(sentence), max_chars))
    chunks, buf = [], []
    size = 0
    for atom in atoms:
        extra = len(atom) + (1 if buf else 0)
        if buf and size + extra > max_chars:
            chunks.append("\n".join(buf))
            buf, size = [], 0
        buf.append(atom)
        size += len(atom) + (1 if len(buf) > 1 else 0)
    if buf:
        chunks.append("\n".join(buf))
    return chunks


def _parse_facts(rows: list[str]) -> list[dict]:
    out = []
    for row in rows:
        match = re.match(r"\[P(\d+)\]\s*(.*)", row)
        if not match:
            continue
        text = match.group(2).strip()
        # Drop reading-guide and TOC-only pointers; the raw substantive clause
        # remains available and is a stronger source.
        if (len(text) < 8 or "条款目录" in text or
                re.fullmatch(r"第.+条\s+.{0,30}", text)):
            continue
        out.append({"page": int(match.group(1)), "text": text,
                    "covered": False})
    return out


def _parse_sections(text: str) -> list[dict]:
    """Return page-bound clause bodies while propagating headings across pages."""
    groups = []
    clause = ""
    title = ""
    section = ""
    pending_code = ""
    title_open = False

    for page, lines in _split_pages(text):
        if _is_toc(lines):
            continue
        current = None
        seen_page_content = False

        def append_body(value: str):
            nonlocal current
            if not clause or not value:
                return
            key = (page, clause, title, section)
            if current is None or current["key"] != key:
                current = {"key": key, "page": page, "clause": clause,
                           "clause_title": title or section, "lines": []}
                groups.append(current)
            current["lines"].append(value)

        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            # Several PDF parsers emit the printed page number as the first
            # line.  Later one-digit lines can be footnote markers and must be
            # retained to keep ``verbatim`` traceable to the raw page.
            if not seen_page_content and re.fullmatch(r"\d+", line):
                seen_page_content = True
                continue
            seen_page_content = True
            if re.fullmatch(r"-\d+-", line):
                continue
            cn = CN_CLAUSE_RE.match(line)
            dec = DEC_CLAUSE_RE.match(line)
            if cn or dec:
                match = cn or dec
                code, rest = match.group(1), match.group(2).strip()
                clause = code
                current = None
                pending_code = ""
                title_open = False
                decimal = dec is not None
                if _looks_like_title(rest, decimal=decimal):
                    title = rest
                    title_open = True
                elif rest:
                    title = section
                    append_body(rest)
                else:
                    title = ""
                    pending_code = code
                continue

            if line in SECTION_NAMES or (len(line) <= 12 and
                                          line.endswith("保险责任")):
                section = line
                continue

            if pending_code:
                if len(line) <= 50 and not re.search(r"[。；，：]", line):
                    title = line
                    title_open = True
                    pending_code = ""
                    continue
                pending_code = ""
                title = section

            if (title_open and title and len(title) <= 10 and len(line) <= 10 and
                    len(title) + len(line) <= 36 and
                    not re.match(r"^[（(]?[0-9一二三四五六七八九十]+[）)．.]", line) and
                    not re.search(r"[。；，：]", line) and
                    not BODY_PREFIX.search(line)):
                title += line
                title_open = False
                current = None
                continue
            title_open = False
            if PAGE_HEADER_RE.search(line):
                continue
            append_body(line)
    return [g for g in groups if g["lines"]]


def _fact_hits(chunk: str, page: int, facts: list[dict]) -> list[dict]:
    norm_chunk = _normal(chunk)
    hits = []
    for fact in facts:
        if fact["page"] != page:
            continue
        norm_fact = _normal(fact["text"])
        probe = norm_fact[:min(50, len(norm_fact))]
        if (probe and probe in norm_chunk) or (len(norm_chunk) >= 30 and
                                               norm_chunk[:40] in norm_fact):
            fact["covered"] = True
            hits.append(fact)
    return hits


def _source_hash(raw: str, facts: list[str], identity: dict) -> str:
    h = hashlib.sha256()
    h.update(raw.encode("utf-8"))
    h.update(json.dumps(facts, ensure_ascii=False,
                        sort_keys=True).encode("utf-8"))
    h.update(json.dumps(identity, ensure_ascii=False,
                        sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def build_document(doc_id: str, raw: str, identity: dict,
                   fact_rows: list[str], max_chars: int) -> tuple[dict, dict]:
    facts = _parse_facts(fact_rows)
    cards = []
    seq = 0
    seen = set()

    def add_card(page, clause, clause_title, verbatim, topics, hits, sources):
        nonlocal seq
        if not verbatim.strip() or not topics:
            return
        key = (page, clause, topics[0], _normal(verbatim))
        if key in seen:
            return
        seen.add(key)
        seq += 1
        cards.append({
            "id": f"{doc_id}:p{page}:{seq:04d}",
            "doc_id": doc_id,
            "page": page,
            "clause": clause,
            "clause_title": clause_title,
            "topic": topics[0],
            "tags": topics[1:],
            "numbers": extract_numbers(verbatim),
            "verbatim": verbatim.strip(),
            "sources": sources,
            "domain_fact_refs": [f"P{x['page']}:{x['text']}" for x in hits[:3]],
        })

    for group in _parse_sections(raw):
        for chunk in _split_chunks(group["lines"], max_chars):
            hits = _fact_hits(chunk, group["page"], facts)
            topics = infer_topics(chunk, group["clause_title"])
            if not topics and hits:
                topics = ["property_liability"]
            if topics:
                sources = ["raw_text"] + (["domain_facts"] if hits else [])
                add_card(group["page"], group["clause"],
                         group["clause_title"], chunk, topics, hits, sources)

    # Facts that fell between parser boundaries remain traceable as exact
    # lexical fallback cards.  They are never summarized or semantically filled.
    for fact in facts:
        if fact["covered"]:
            continue
        topics = infer_topics(fact["text"])
        if not topics:
            continue
        add_card(fact["page"], "", "domain_facts词法行", fact["text"],
                 topics, [fact], ["domain_facts"])
        fact["covered"] = True

    doc = {
        "identity": {
            "company": identity.get("company", ""),
            "product": identity.get("product", ""),
            "aliases": identity.get("alias", identity.get("aliases", [])),
        },
        "source_sha256": _source_hash(raw, fact_rows, identity),
        "capsules": cards,
    }
    stats = {
        "cards": len(cards),
        "facts": len(facts),
        "facts_covered": sum(x["covered"] for x in facts),
    }
    return doc, stats


def build(max_chars: int = 520, processed_dir: pathlib.Path = PD) -> dict:
    processed_dir = pathlib.Path(processed_dir)
    insurance_dir = processed_dir / "insurance"
    facts_all = json.loads((processed_dir / "domain_facts.json").read_text(encoding="utf-8"))
    titles = json.loads((processed_dir / "insurance_titles.json").read_text(encoding="utf-8"))
    documents = {}
    doc_stats = {}
    topic_counts = Counter()
    topic_membership = Counter()
    lengths = []
    numeric = 0
    fact_backed = 0

    def sort_key(path):
        return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)

    for path in sorted(insurance_dir.glob("*.txt"), key=sort_key):
        doc_id = path.stem
        raw = path.read_text(encoding="utf-8", errors="ignore")
        doc, stats = build_document(doc_id, raw, titles.get(doc_id, {}),
                                    facts_all.get(doc_id, []), max_chars)
        documents[doc_id] = doc
        doc_stats[doc_id] = stats
        for card in doc["capsules"]:
            topic_counts[card["topic"]] += 1
            topic_membership.update([card["topic"], *(card.get("tags") or [])])
            lengths.append(len(card["verbatim"]))
            numeric += bool(card["numbers"])
            fact_backed += "domain_facts" in card["sources"]

    total = sum(x["cards"] for x in doc_stats.values())
    facts = sum(x["facts"] for x in doc_stats.values())
    facts_covered = sum(x["facts_covered"] for x in doc_stats.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "builder": "script/build_insurance_capsules.py",
        "built_from": [
            "processed_data/insurance/*.txt",
            "processed_data/domain_facts.json",
            "processed_data/insurance_titles.json",
        ],
        "documents": documents,
        "stats": {
            "documents": len(documents),
            "capsules": total,
            "with_numbers": numeric,
            "domain_fact_backed": fact_backed,
            "domain_facts": facts,
            "domain_facts_covered": facts_covered,
            "max_verbatim_chars": max(lengths, default=0),
            "average_verbatim_chars": round(sum(lengths) / max(len(lengths), 1), 1),
            "primary_topics": dict(sorted(topic_counts.items())),
            "topic_membership": dict(sorted(topic_membership.items())),
            "per_document": doc_stats,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", type=pathlib.Path, default=PD)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--max-chars", type=int, default=520)
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args()
    artifact = build(max_chars=args.max_chars, processed_dir=args.processed)
    if not args.stats_only:
        out = args.output or args.processed / "insurance_capsules.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(artifact, ensure_ascii=False, indent=1) + "\n",
                       encoding="utf-8")
    print(json.dumps(artifact["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
