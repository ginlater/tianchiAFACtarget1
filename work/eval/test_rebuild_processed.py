"""Portable end-to-end tests for the deterministic preprocessing rebuild."""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

import fitz

WORK = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORK / "script"))

from rebuild_processed import rebuild  # noqa: E402


def write_pdf(path: pathlib.Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as document:
        page = document.new_page()
        y = 72
        for line in lines:
            page.insert_text((72, y), line, fontname="china-s", fontsize=11)
            y += 20
        document.save(path)


def make_fixture(raw: pathlib.Path) -> None:
    write_pdf(raw / "insurance" / "1.pdf", [
        "中国平安人寿保险股份有限公司",
        "平安测试医疗保险",
        "第一条 保险责任",
        "本合同犹豫期为15日，身故保险金按160%给付。",
    ])
    write_pdf(raw / "financial_contracts" / "text01.pdf", [
        "测试债券募集说明书", "票面利率为3%，到期兑付。",
    ])
    write_pdf(raw / "financial_reports" / "annual_demo_2024_report.PDF", [
        "示例公司2024年年度报告", "合并资产负债表", "资产总计 100 90",
    ])
    write_pdf(raw / "research" / "pack2_text01.pdf", [
        "测试行业研究报告", "预计销量同比增长20%。",
    ])
    text_dir = raw / "regulatory" / "txt"
    text_dir.mkdir(parents=True)
    (text_dir / "strict_v3_demo.txt").write_text("测试监管规定\n第一条 测试。\n",
                                                   encoding="utf-8")
    html_dir = raw / "regulatory" / "html"
    html_dir.mkdir(parents=True)
    (html_dir / "csrc_0001.html").write_text(
        '<html><head><meta name="ArticleTitle" content="测试处罚决定">'
        '<meta name="description" content="当事人测试公司，决定给予警告。">'
        '<meta name="PubDate" content="2026-01-01"></head>'
        '<body><p>处罚决定正文</p></body></html>', encoding="utf-8")
    write_pdf(raw / "regulatory" / "attachments" / "csrc_0001_att1.pdf",
              ["测试附件", "附件正文。"])


class RebuildProcessedTests(unittest.TestCase):
    def test_check_only_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            raw = base / "raw"
            make_fixture(raw)
            output = base / "must_not_exist"
            result = rebuild(raw, output, check_only=True)
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["input"]["files"], 7)
            self.assertFalse(output.exists())

    def test_full_rebuild_is_portable_and_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            raw = base / "dataset" / "raw"
            make_fixture(raw)
            first, second = base / "processed-one", base / "processed-two"
            rebuild(base / "dataset", first)
            rebuild(raw, second)
            manifest_a = json.loads((first / "processed_manifest.json").read_text(
                encoding="utf-8"))
            manifest_b = json.loads((second / "processed_manifest.json").read_text(
                encoding="utf-8"))
            self.assertEqual(manifest_a, manifest_b)
            self.assertEqual(manifest_a["output"]["documents"], 7)
            self.assertEqual(manifest_a["semantic_models_used"], [])
            meta = json.loads((first / "docs_meta.json").read_text(encoding="utf-8"))
            self.assertTrue(all(not pathlib.PurePosixPath(item["src"]).is_absolute()
                                for item in meta.values()))
            identities = json.loads((first / "insurance_titles.json").read_text(
                encoding="utf-8"))
            self.assertIn("医疗保险", identities["1"]["product"])
            alignment = json.loads((first / "align_matrix.json").read_text(
                encoding="utf-8"))
            identity_key = next(iter(alignment["ins_clauses"]))
            self.assertIn(identities["1"]["company"], identity_key)
            self.assertIn(identities["1"]["product"], identity_key)
            capsules = json.loads((first / "insurance_capsules.json").read_text(
                encoding="utf-8"))
            self.assertTrue(capsules["documents"]["1"]["capsules"])

    def test_existing_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = pathlib.Path(temporary)
            raw = base / "raw"
            make_fixture(raw)
            output = base / "existing"
            output.mkdir()
            sentinel = output / "keep.txt"
            sentinel.write_text("untouched", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                rebuild(raw, output)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched")


if __name__ == "__main__":
    unittest.main()
