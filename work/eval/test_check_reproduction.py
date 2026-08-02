"""Regression and tamper tests for script/check_reproduction.py.

These tests never invoke an API.  The historical full2 run is used as an
immutable tamper fixture.  It predates the literal API-response provenance
gate and intentionally demonstrates why code-appended reasoning is rejected.
"""
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import socket
import tempfile
import unittest
from unittest import mock


WORK = Path(__file__).resolve().parents[1]
SCRIPT = WORK / "script" / "check_reproduction.py"
FULL2 = WORK / "output" / "honest_repro_full2"

SPEC = importlib.util.spec_from_file_location("check_reproduction", SCRIPT)
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class ReproductionCheckerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        missing = [name for name in CHECKER.REQUIRED
                   if not (FULL2 / name).is_file()]
        if missing:
            raise unittest.SkipTest(
                f"old full2 fixture is unavailable: {', '.join(missing)}")

    def _run(self, path):
        with redirect_stdout(io.StringIO()):
            CHECKER.main(path)

    def _assert_fails(self, path, fragment):
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                CHECKER.main(path)
        self.assertIn(fragment, str(caught.exception))

    def _case(self, copied_name):
        """Create a fixture where only ``copied_name`` is writable."""
        temp = tempfile.TemporaryDirectory()
        case = Path(temp.name) / "case"
        case.mkdir()
        for name in CHECKER.REQUIRED:
            source = (FULL2 / name).resolve()
            target = case / name
            if name == copied_name:
                shutil.copy2(source, target)
            else:
                target.symlink_to(source)
        self.addCleanup(temp.cleanup)
        return case

    @staticmethod
    def _load(path):
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _dump(path, payload):
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                        encoding="utf-8")

    @staticmethod
    def _load_jsonl(path):
        return [json.loads(line) for line in path.read_text(
            encoding="utf-8").splitlines()]

    @staticmethod
    def _dump_jsonl(path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                for row in rows), encoding="utf-8")

    def test_old_full2_exposes_unreported_reasoning_postprocess(self):
        # Any accidental network path is a hard test failure.
        with mock.patch.object(socket, "socket",
                               side_effect=AssertionError("network attempted")):
            self._assert_fails(
                FULL2,
                "selected reasoning is not literal output of an attached")

    def test_reasoning_response_match_requires_success_and_qid_attachment(self):
        direct = {
            "status": "ok", "qid": "q1", "allocation_qids": [],
            "response": {"content": "定位原文并得出答案 A",
                         "reasoning_content": ""},
        }
        self.assertTrue(CHECKER._api_attached_to(direct, "q1"))
        self.assertTrue(CHECKER._response_contains_reasoning(
            direct, "定位原文并得出答案 A"))

        batch = {
            "status": "ok", "qid": "_batch",
            "allocation_qids": ["q1", "q2"],
            "response": {"content": "", "reasoning_content": "批量题解释"},
        }
        self.assertTrue(CHECKER._api_attached_to(batch, "q2"))
        self.assertFalse(CHECKER._api_attached_to(batch, "q3"))
        self.assertTrue(CHECKER._response_contains_reasoning(
            batch, "批量题解释"))
        batch["status"] = "error"
        self.assertFalse(CHECKER._response_contains_reasoning(
            batch, "批量题解释"))

    def test_answer_and_reasoning_cross_file_tampering_fails(self):
        case = self._case("answers.json")
        path = case / "answers.json"
        payload = self._load(path)
        payload["fc_b_011"] = ["D"]
        self._dump(path, payload)
        self._assert_fails(case, "answers.json/CSV answer mismatch")

        case = self._case("reasonings.json")
        path = case / "reasonings.json"
        payload = self._load(path)
        payload["fc_b_011"] += " 篡改"
        self._dump(path, payload)
        self._assert_fails(case, "reasonings.json/CSV reasoning mismatch")

    def test_reasoning_source_trace_tampering_fails(self):
        case = self._case("reasoning_sources.json")
        path = case / "reasoning_sources.json"
        payload = self._load(path)
        payload["fc_b_011"]["traces"] = []
        self._dump(path, payload)
        self._assert_fails(case, "no selected stage/traces")

        case = self._case("reasoning_sources.json")
        path = case / "reasoning_sources.json"
        payload = self._load(path)
        payload["fc_b_011"]["traces"][0]["answer"] = "D"
        self._dump(path, payload)
        self._assert_fails(case, "selected reasoning trace disagrees")

    def test_api_ledger_mapping_and_allocation_tampering_fails(self):
        case = self._case("token_ledger.json")
        path = case / "token_ledger.json"
        payload = self._load(path)
        first = payload["calls"][0]
        target = first["allocation_qids"][0]
        first["allocated_usage"][target][0] += 1
        self._dump(path, payload)
        self._assert_fails(case, "allocated_usage does not conserve")

        case = self._case("token_ledger.json")
        path = case / "token_ledger.json"
        payload = self._load(path)
        payload["calls"][0]["tag"] = "tampered_tag"
        self._dump(path, payload)
        self._assert_fails(case, "API/ledger per-call mismatch")

    def test_run_and_docsel_coverage_tampering_fails(self):
        for name in ("run_log.jsonl", "docsel_log.jsonl"):
            with self.subTest(name=name):
                case = self._case(name)
                path = case / name
                rows = self._load_jsonl(path)
                rows.pop(0)
                self._dump_jsonl(path, rows)
                self._assert_fails(case, f"{name} qid coverage mismatch")

    def test_evidence_token_and_reference_tampering_fails(self):
        def mutate_and_check(mutator, fragment):
            case = self._case("evidence.json")
            path = case / "evidence.json"
            payload = self._load(path)
            mutator(payload["questions"][0])
            self._dump(path, payload)
            self._assert_fails(case, fragment)

        mutate_and_check(
            lambda entry: entry["token_accounting"].__setitem__(
                "total_tokens", entry["token_accounting"]["total_tokens"] + 1),
            "evidence/CSV token mismatch")
        mutate_and_check(
            lambda entry: entry.__setitem__("api_attempts", []),
            "evidence has no API references")
        mutate_and_check(
            lambda entry: entry["token_accounting"].__setitem__(
                "ledger_calls", []),
            "evidence has no ledger references")

    def test_schema2_contract_api_evidence_and_reasoning_guards(self):
        qids = [f"q{i:03d}" for i in range(100)]
        config = {
            "schema_version": 2,
            "question_ids": qids,
            "question_count": 100,
            "model": "qwen3.6-plus",
            "verify_model": "qwen3.6-plus",
            "arguments": {
                "model": "qwen3.6-plus",
                "verify_model": "qwen3.6-plus",
                "resume": False,
                "fresh_digests": True,
                "limit": 0,
                "qids": "",
                "batch": True,
            },
        }
        self.assertIs(CHECKER.validate_complete_run_config(config, qids),
                      config)
        for key, bad in (("resume", True), ("fresh_digests", False),
                         ("limit", 1), ("qids", "q001"),
                         ("batch", False)):
            with self.subTest(config_key=key):
                changed = copy.deepcopy(config)
                changed["arguments"][key] = bad
                with self.assertRaisesRegex(RuntimeError, f"arguments.{key}"):
                    CHECKER.validate_complete_run_config(changed, qids)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            qdir = root / "question_b"
            qdir.mkdir()
            (qdir / "questions.json").write_text(json.dumps([
                {"qid": qid, "question": qid, "type": "单选题",
                 "options": {"A": "甲"}} for qid in qids
            ], ensure_ascii=False))
            submit = root / "submit.csv"
            submit.write_text("qid\n")
            config["input_paths"] = {
                "qdir": str(qdir), "submit_template": str(submit),
            }
            config["inputs"] = CHECKER.build_input_manifest(qdir, submit)
            CHECKER.validate_schema2_inputs(config, qids)
            submit.write_text("qid,changed\n")
            with self.assertRaisesRegex(SystemExit, "input manifest differs"):
                CHECKER.validate_schema2_inputs(config, qids)

        CHECKER.validate_schema2_api_record(
            {"status": "ok", "model": "qwen3.6-plus"}, 1)
        with self.assertRaisesRegex(SystemExit, "not successful"):
            CHECKER.validate_schema2_api_record(
                {"status": "error", "model": "qwen3.6-plus"}, 2)
        with self.assertRaisesRegex(SystemExit, "must be exactly"):
            CHECKER.validate_schema2_api_record(
                {"status": "ok", "model": "qwen3.6-flash"}, 3)

        evidence = {
            "retrieval": {
                "selected_doc_ids": ["doc1"],
                "evidence_ids": ["doc1#c1"],
                "evidence_retrieval": [{
                    "evidence_id": "doc1#c1", "doc_id": "doc1",
                    "page": 1, "quoted_clause": "原文事实",
                }],
            },
        }
        CHECKER.validate_complete_evidence(evidence, "q001")
        missing = copy.deepcopy(evidence)
        missing["retrieval"]["evidence_retrieval"][0]["missing"] = True
        with self.assertRaisesRegex(SystemExit, "contains missing chunk"):
            CHECKER.validate_complete_evidence(missing, "q001")

        payload = {"provenance": {
            "source_directory": ".", "run_config": config,
        }}
        CHECKER.validate_schema2_evidence_provenance(payload, config)
        absolute = copy.deepcopy(payload)
        absolute["provenance"]["source_directory"] = "/private/run"
        with self.assertRaisesRegex(SystemExit, "must be portable"):
            CHECKER.validate_schema2_evidence_provenance(absolute, config)

        self.assertEqual(CHECKER.suspicious_reasoning_marker(
            "依据原文逐项计算，因此答案为AB"), "")
        for text in ("参考答案为AB", "使用answer key校正", "对键后修改",
                     "leaderboard反馈显示错误", "router6答案"):
            with self.subTest(reasoning=text):
                self.assertTrue(CHECKER.suspicious_reasoning_marker(text))


if __name__ == "__main__":
    unittest.main()
