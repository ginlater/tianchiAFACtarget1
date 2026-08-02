import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "script" / "build_insurance_titles.py"
SPEC = importlib.util.spec_from_file_location("build_insurance_titles", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_insurance_titles_handles_split_and_joined_titles(tmp_path: Path) -> None:
    insurance = tmp_path / "processed" / "insurance"
    insurance.mkdir(parents=True)
    (insurance / "27.txt").write_text(
        """[P1]
某某健康保险股份有限公司
悦享百万医疗保险（互联网2026 版）条款
险种简称：悦享百万
""",
        encoding="utf-8",
    )
    (insurance / "105.txt").write_text(
        """[P1]
星河在线财产保险股份有限公司星河家庭财产综合保险（2025版）
第一条 本保险合同由保险条款和保险单组成。
""",
        encoding="utf-8",
    )
    (insurance / "split-name.txt").write_text(
        """[P1]
阅读指引
对“星河
．．安心住院
．．7.0医疗保险
．．A款”内容的解释以条款为准。
在本条款中，‘本公司’指星河健康保险股份有限公司。
""",
        encoding="utf-8",
    )

    output = tmp_path / "titles.json"
    first = MODULE.build(tmp_path / "processed", output)
    first_bytes = output.read_bytes()
    second = MODULE.build(tmp_path / "processed", output)

    assert first == second
    assert output.read_bytes() == first_bytes
    assert list(first) == ["27", "105", "split-name"]
    assert first["27"]["company"] == "某某健康保险股份有限公司"
    assert "悦享百万" in first["27"]["alias"]
    assert first["105"]["product"].startswith("星河家庭财产综合保险")
    assert "安心住院7.0医疗保险A款" in first["split-name"]["product"]

    serialized = json.dumps(first, ensure_ascii=False)
    assert "qid" not in serialized.lower()
    assert "answer" not in serialized.lower()
