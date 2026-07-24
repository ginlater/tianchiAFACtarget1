#!/usr/bin/env python3
"""打包 submission.zip（B榜前15代码审核用）。用法: python script/package_submission.py b_final"""
import pathlib, sys, zipfile

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = ROOT / "work"


def main(tag="b_router6"):
    out = WORK / "submission.zip"
    outdir = WORK / "output" / tag
    zf = zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED)

    def add(path, arc):
        zf.write(path, arc)

    add(outdir / "answer.csv", "submission/answer.csv")
    add(outdir / "evidence.json", "submission/evidence.json")
    add(WORK / "requirements.txt", "submission/requirements.txt")
    add(WORK / "README.md", "submission/README.md")
    for sub, pat in [("agent", "*.py"), ("script", "*.py")]:
        for p in sorted((WORK / sub).glob(pat)):
            add(p, f"submission/{sub}/{p.name}")
    for p in sorted((WORK / "processed_data").rglob("*")):
        if p.is_file() and ".DS_Store" not in p.name:
            add(p, f"submission/processed_data/{p.relative_to(WORK / 'processed_data')}")
    for name in ["run_log.jsonl", "token_ledger.json", "docsel_log.jsonl",
                 "digests.json"]:
        p = outdir / name
        if p.exists():
            add(p, f"submission/logs/{name}")
    zf.close()
    size = out.stat().st_size / 1e6
    names = zipfile.ZipFile(out).namelist()
    assert not any(".env" in n or "sk-" in n for n in names), "密钥泄漏检查失败"
    print(f"submission.zip: {size:.0f}MB, {len(names)} files")
    assert size < 1000, "超过1GB限制"


if __name__ == "__main__":
    main(sys.argv[1])
