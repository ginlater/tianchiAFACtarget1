"""Network-free tests for the full3 source/data freeze gate."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest
import warnings
import zipfile
from types import SimpleNamespace
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[2]

from agent import repro  # noqa: E402
from script import build_evidence  # noqa: E402
from script import package_submission  # noqa: E402


class RuntimeFreezeTests(unittest.TestCase):
    def make_repo(self, root: pathlib.Path) -> pathlib.Path:
        work = root / "work"
        (work / "agent").mkdir(parents=True)
        (work / "script").mkdir()
        (work / "config").mkdir()
        (work / "processed_data" / "nested").mkdir(parents=True)
        (root / "upload_b").mkdir()

        for name in repro.RUNTIME_AGENT_MODULES:
            (work / "agent" / name).write_text("# frozen\n")
        (work / "agent" / "run_b2.py").write_text(
            "from agent import answerer, b_schema, batch, calc, doc_select\n"
            "from agent.repro import write_run_config\n")
        # Deliberately present legacy/experimental modules are not part of the
        # formal entrypoint's dependency closure and must never enter a package.
        (work / "agent" / "run_a.py").write_text("LEGACY = 'A fallback'\n")
        (work / "agent" / "run_b.py").write_text("LEGACY = True\n")
        for name in repro.RUNTIME_SCRIPT_NAMES:
            (work / "script" / name).write_text("# frozen\n")
        (work / "generate_answer.sh").write_text("#!/bin/sh\n")
        (work / "config" / "honest_repro.env").write_text("AFAC_SLIM=1\n")
        (work / "requirements.txt").write_text("openai==1.0\n")
        (work / "processed_data" / "facts.json").write_text('{"x":1}\n')
        (work / "processed_data" / "nested" / "rows.txt").write_text("row\n")
        (work / "README.md").write_text("runtime\n")
        (root / "比赛规则.md").write_text("比赛规则\n")
        (root / "b榜新增规则.txt").write_text("B new rules\n")
        (root / "b榜补充.md").write_text("B supplemental rules\n")
        (root / "upload_b" / "readme.md").write_text("B rules\n")
        return root

    def make_flat_package(self, root: pathlib.Path) -> pathlib.Path:
        """Create a minimal independently importable extracted package."""

        (root / "agent").mkdir(parents=True)
        (root / "script").mkdir()
        (root / "config").mkdir()
        (root / "processed_data").mkdir()
        (root / "official_rules").mkdir()

        # repro.py and paths.py are the relocation mechanism under test.  The
        # other whitelisted modules only need to exist and form a closed AST
        # import graph for this network-free manifest smoke test.
        for name in repro.RUNTIME_AGENT_MODULES:
            target = root / "agent" / name
            if name in {"__init__.py", "paths.py", "repro.py"}:
                shutil.copy2(ROOT / "work" / "agent" / name, target)
            else:
                target.write_text("# frozen\n")
        for name in repro.RUNTIME_SCRIPT_NAMES:
            (root / "script" / name).write_text("# frozen\n")
        shutil.copy2(ROOT / "work" / "generate_answer.sh",
                     root / "generate_answer.sh")
        (root / "config" / "honest_repro.env").write_text(
            "AFAC_DET_CALC_MAX_TOKENS=1800\n")
        (root / "processed_data" / "facts.json").write_text('{}\n')
        (root / "requirements.txt").write_text("openai==2.46.0\n")
        (root / "README.md").write_text("flat runtime\n")
        for name in repro.RUNTIME_OFFICIAL_RULES:
            (root / "official_rules" / name).write_text(name + "\n")
        (root / "official_rules" / "upload_b_readme.md").write_text(
            "B input rules\n")
        return root

    def test_manifest_is_relative_complete_and_verifiable(self):
        with tempfile.TemporaryDirectory() as td:
            repo = self.make_repo(pathlib.Path(td))
            manifest = repro.build_runtime_manifest(repo)
            paths = {entry["path"] for entry in manifest["files"]}
            self.assertIn("work/agent/run_b2.py", paths)
            self.assertNotIn("work/agent/run_a.py", paths)
            self.assertNotIn("work/agent/run_b.py", paths)
            agent_paths = {path for path in paths
                           if path.startswith("work/agent/")}
            self.assertEqual(
                agent_paths,
                {"work/agent/" + name
                 for name in repro.RUNTIME_AGENT_MODULES})
            self.assertIn("work/processed_data/nested/rows.txt", paths)
            self.assertIn("比赛规则.md", paths)
            self.assertIn("b榜新增规则.txt", paths)
            self.assertIn("b榜补充.md", paths)
            self.assertIn("upload_b/readme.md", paths)
            self.assertTrue(all(not pathlib.PurePosixPath(path).is_absolute()
                                for path in paths))
            for entry in manifest["files"]:
                data = (repo / entry["path"]).read_bytes()
                self.assertEqual(entry["bytes"], len(data))
                self.assertEqual(entry["sha256"],
                                 hashlib.sha256(data).hexdigest())
            self.assertEqual(repro.verify_runtime_manifest(manifest, repo),
                             manifest)

    def test_changed_added_and_unsafe_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            repo = self.make_repo(pathlib.Path(td))
            manifest = repro.build_runtime_manifest(repo)

            target = repo / "work" / "processed_data" / "facts.json"
            target.write_text('{"x":2}\n')
            with self.assertRaisesRegex(RuntimeError, "changed="):
                repro.verify_runtime_manifest(manifest, repo)
            target.write_text('{"x":1}\n')

            # An unused Python file outside the explicit dependency closure is
            # intentionally ignored and cannot alter the archive.
            (repo / "work" / "agent" / "added.py").write_text("NEW = 1\n")
            self.assertEqual(repro.verify_runtime_manifest(manifest, repo),
                             manifest)

            # Dynamic data/config trees remain closed-world: additions there
            # are runtime inputs and must invalidate the snapshot.
            (repo / "work" / "processed_data" / "added.json").write_text("{}\n")
            with self.assertRaisesRegex(RuntimeError, "added="):
                repro.verify_runtime_manifest(manifest, repo)

            unsafe = copy.deepcopy(manifest)
            unsafe["files"][0]["path"] = "../outside"
            with self.assertRaisesRegex(RuntimeError, "unsafe"):
                repro.validate_runtime_manifest(unsafe)

    def test_agent_whitelist_is_required_and_dependency_closed(self):
        with tempfile.TemporaryDirectory() as td:
            repo = self.make_repo(pathlib.Path(td))
            agent = repo / "work" / "agent"
            dependencies = repro.validate_runtime_agent_dependencies(repo)
            self.assertNotIn("run_a.py", dependencies)
            self.assertNotIn("run_b.py", dependencies)

            for name in repro.RUNTIME_AGENT_MODULES:
                path = agent / name
                original = path.read_bytes()
                path.unlink()
                with self.subTest(missing=name):
                    with self.assertRaises(FileNotFoundError):
                        repro.runtime_files(repo)
                path.write_bytes(original)

            (agent / "experimental.py").write_text("VALUE = 1\n")
            with (agent / "run_b2.py").open("a") as stream:
                stream.write("from agent import experimental\n")
            with self.assertRaisesRegex(RuntimeError, "outside whitelist"):
                repro.runtime_files(repo)

    def test_symlinks_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            repo = self.make_repo(pathlib.Path(td))
            link = repo / "work" / "processed_data" / "link.txt"
            try:
                link.symlink_to(repo / "work" / "processed_data" / "facts.json")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                repro.build_runtime_manifest(repo)

    def test_write_run_config_records_manifest_and_git_dirty(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            qdir = root / "questions"
            qdir.mkdir()
            (qdir / "q.json").write_text("[]\n")
            submit = root / "submit.csv"
            submit.write_text("qid\n")
            output = root / "run_config.json"
            frozen = {"schema_version": 1, "root": "repository", "files": []}
            args = SimpleNamespace(model="qwen3.6-plus", verify_model=None,
                                   output_dir=str(root / "out"))
            with mock.patch.object(repro, "build_runtime_manifest",
                                   return_value=frozen), \
                    mock.patch.object(repro, "_git_commit", return_value="abc"), \
                    mock.patch.object(repro, "_git_dirty", return_value=True):
                config = repro.write_run_config(output, args, qdir, submit)
            self.assertIs(config["git_dirty"], True)
            self.assertEqual(config["runtime_manifest"], frozen)
            self.assertEqual(config["schema_version"], 2)
            self.assertEqual(config["question_ids"], [])
            self.assertEqual(config["question_count"], 0)
            stored = json.loads(output.read_text())
            self.assertEqual(stored["runtime_manifest"], frozen)

    def test_schema2_run_config_is_portable_and_complete(self):
        with tempfile.TemporaryDirectory() as td:
            repo = pathlib.Path(td) / "repo"
            qdir = repo / "upload_b" / "question_b"
            qdir.mkdir(parents=True)
            (qdir / "questions.json").write_text("[]\n")
            submit = repo / "upload_b" / "submit.csv"
            submit.write_text("qid\n")
            output = repo / "work" / "output" / "run_config.json"
            output.parent.mkdir(parents=True)
            qids = [f"q{i:03d}" for i in range(100)]
            args = SimpleNamespace(
                model="qwen3.6-plus", verify_model="qwen3.6-plus",
                output_dir=str(repo / "work" / "output"), resume=False,
                fresh_digests=True, limit=0, qids="", batch=True)
            frozen = {"schema_version": 1, "root": "repository", "files": []}
            argv = [str(repo / "work" / "agent" / "run_b2.py"),
                    "--qdir", str(qdir), "--submit-template", str(submit)]
            with mock.patch.object(repro, "build_runtime_manifest",
                                   return_value=frozen), \
                    mock.patch.object(repro, "_runtime_layout",
                                      return_value=(repo, repo / "work", False)), \
                    mock.patch.object(repro, "_git_commit", return_value="abc"), \
                    mock.patch.object(repro, "_git_dirty", return_value=True), \
                    mock.patch("sys.argv", argv):
                config = repro.write_run_config(
                    output, args, qdir, submit, question_ids=qids)
            self.assertEqual(config["schema_version"], 2)
            self.assertEqual(config["question_ids"], qids)
            self.assertEqual(config["question_count"], 100)
            self.assertEqual(config["command"][0], "work/agent/run_b2.py")
            self.assertIn("upload_b/question_b", config["command"])
            self.assertEqual(config["input_paths"], {
                "qdir": "upload_b/question_b",
                "submit_template": "upload_b/submit.csv",
            })
            self.assertIs(repro.validate_complete_run_config(config, qids),
                          config)
            broken = copy.deepcopy(config)
            broken["arguments"]["resume"] = True
            with self.assertRaisesRegex(RuntimeError, "arguments.resume"):
                repro.validate_complete_run_config(broken, qids)

    def test_build_evidence_resolves_relative_input_and_uses_relative_source(self):
        with tempfile.TemporaryDirectory() as td:
            repo = pathlib.Path(td) / "repo"
            work = repo / "work"
            qdir = repo / "upload_b" / "question_b"
            work.mkdir(parents=True)
            qdir.mkdir(parents=True)
            (qdir / "q.json").write_text(json.dumps([{
                "qid": "q1", "question": "题目", "type": "单选题",
                "options": {"A": "甲"},
            }], ensure_ascii=False))
            with mock.patch.object(build_evidence, "WORK", work):
                questions = build_evidence._question_map({
                    "input_paths": {"qdir": "upload_b/question_b"},
                })
            self.assertEqual(list(questions), ["q1"])
            source = (ROOT / "work" / "script" /
                      "build_evidence.py").read_text(encoding="utf-8")
            self.assertIn('"source_directory": "."', source)

    def test_every_formal_afac_switch_is_fixed_or_explicitly_unset(self):
        pattern = re.compile(r"AFAC_[A-Z0-9_]+")
        used = set()
        for name in repro.RUNTIME_AGENT_MODULES:
            used.update(pattern.findall(
                (ROOT / "work" / "agent" / name).read_text(encoding="utf-8")))

        config_text = (ROOT / "work" / "config" /
                       "honest_repro.env").read_text(encoding="utf-8")
        fixed = set(re.findall(
            r"(?m)^\s*(?:export\s+)?(AFAC_[A-Z0-9_]+)=", config_text))
        entrypoint = (ROOT / "work" / "generate_answer.sh").read_text(
            encoding="utf-8")
        explicitly_unset = set()
        for line in entrypoint.splitlines():
            if line.startswith("unset "):
                explicitly_unset.update(pattern.findall(line))
        self.assertEqual(used - fixed - explicitly_unset, set())

    def test_packager_gate_rules_destinations_and_secret_scan(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            repo = self.make_repo(root / "repo")
            run = root / "run"
            run.mkdir()
            config = {
                "git_dirty": True,
                "runtime_manifest": repro.build_runtime_manifest(repo),
            }
            (run / "run_config.json").write_text(json.dumps(config))
            with mock.patch.object(package_submission, "REPO", repo):
                loaded = package_submission.load_frozen_runtime(run)
                self.assertEqual(loaded, config)

                (repo / "work" / "config" / "honest_repro.env").write_text(
                    "AFAC_SLIM=0\n")
                with self.assertRaisesRegex(RuntimeError, "changed="):
                    package_submission.load_frozen_runtime(run)

            self.assertEqual(
                package_submission.runtime_archive_destination("比赛规则.md"),
                "official_rules/比赛规则.md")
            self.assertEqual(
                package_submission.runtime_archive_destination(
                    "b榜新增规则.txt"),
                "official_rules/b榜新增规则.txt")
            self.assertEqual(
                package_submission.runtime_archive_destination(
                    "upload_b/readme.md"),
                "official_rules/upload_b_readme.md")

            with mock.patch.object(package_submission, "REPO", repo), \
                    mock.patch.object(package_submission, "WORK",
                                      repo / "work"):
                with self.assertRaisesRegex(RuntimeError, "frozen runtime"):
                    package_submission.safe_output_path(
                        repo / "work" / "processed_data" / "bad.zip")
                allowed = package_submission.safe_output_path(
                    repo / "work" / "output" / "submission.zip")
                self.assertEqual(allowed,
                                 (repo / "work" / "output" /
                                  "submission.zip").resolve())

            secret = root / "secret.txt"
            secret.write_text("DASHSCOPE_API_KEY=abcdefghijklmnop\n")
            with self.assertRaisesRegex(RuntimeError, "secret-like"):
                package_submission.checked_bytes(secret)

    def test_static_package_contents_exclude_legacy_and_workspace_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            repo = self.make_repo(root / "repo")
            run = root / "run"
            run.mkdir()
            for name in package_submission.RUN_FILES:
                (run / name).write_text("{}\n")

            banned_output = repo / "work" / "output" / "b_v4"
            banned_output.mkdir(parents=True)
            (banned_output / "piece_sources.json").write_text("{}\n")
            (repo / "work" / "output" / "assignment_final.json").write_text(
                "{}\n")
            (repo / "work" / ".env").write_text("DASHSCOPE_API_KEY=secret\n")
            (repo / "work" / "config" / ".env").write_text(
                "DASHSCOPE_API_KEY=also-secret\n")

            with mock.patch.object(package_submission, "REPO", repo), \
                    mock.patch.object(package_submission, "WORK",
                                      repo / "work"):
                entries = package_submission.collect(run)

            destinations = [destination for _path, destination in entries]
            sources = []
            for path, _destination in entries:
                try:
                    sources.append(path.resolve().relative_to(repo).as_posix())
                except ValueError:
                    sources.append(path.name)
            joined = "\n".join(sources + destinations)
            self.assertIn("answer.csv", destinations)
            self.assertIn("evidence.json", destinations)
            self.assertIn("agent/run_b2.py", destinations)
            self.assertIn("script/check_reproduction.py", destinations)
            self.assertIn("processed_data/facts.json", destinations)
            self.assertIn("logs/run_config.json", destinations)
            self.assertIn("requirements.txt", destinations)
            self.assertIn("README.md", destinations)
            self.assertNotIn("agent/run_a.py", destinations)
            self.assertNotIn("agent/run_b.py", destinations)
            self.assertFalse(any(name.startswith("submission/")
                                 for name in destinations))
            for forbidden in ("work/output", "b_v4", "piece_sources",
                              "assignment_final"):
                self.assertNotIn(forbidden, joined)
            self.assertFalse(any(pathlib.PurePosixPath(name).name == ".env"
                                 for name in sources + destinations))

    def test_official_input_gate_and_archive_closure_are_network_free(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            repo = root / "repo"
            qdir = repo / "upload_b" / "question_b"
            qdir.mkdir(parents=True)
            qids = [f"q{i:03d}" for i in range(100)]
            (qdir / "questions.json").write_text(json.dumps([
                {"qid": qid, "question": qid, "type": "单选题",
                 "options": {"A": "甲"}} for qid in qids
            ], ensure_ascii=False))
            submit = repo / "upload_b" / "submit.csv"
            submit.write_text("qid\n")
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
                "input_paths": {
                    "qdir": "upload_b/question_b",
                    "submit_template": "upload_b/submit.csv",
                },
                "inputs": repro.build_input_manifest(qdir, submit),
            }
            with mock.patch.object(package_submission, "REPO", repo):
                self.assertIs(package_submission.validate_official_run(config),
                              config)
                submit.write_text("qid,changed\n")
                with self.assertRaisesRegex(RuntimeError, "input manifest"):
                    package_submission.validate_official_run(config)

            def write_archive(path, members, declared=None):
                declared = declared if declared is not None else [
                    {"path": name, "bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data, _attr in members
                ]
                manifest = {"schema_version": 2, "files": declared}
                with zipfile.ZipFile(path, "w") as archive:
                    for name, data, attr in members:
                        info = zipfile.ZipInfo(name)
                        info.external_attr = attr
                        archive.writestr(info, data)
                    archive.writestr("package_manifest.json",
                                     json.dumps(manifest).encode())

            valid = root / "valid.zip"
            write_archive(valid, [("answer.csv", b"ok", 0o644 << 16)])
            package_submission.verify_archive(valid)

            corrupt = root / "corrupt-crc.zip"
            shutil.copy2(valid, corrupt)
            with zipfile.ZipFile(corrupt) as archive:
                info = archive.getinfo("answer.csv")
                offset = info.header_offset
            raw = bytearray(corrupt.read_bytes())
            name_len = int.from_bytes(raw[offset + 26:offset + 28], "little")
            extra_len = int.from_bytes(raw[offset + 28:offset + 30], "little")
            data_offset = offset + 30 + name_len + extra_len
            raw[data_offset] ^= 0x01
            corrupt.write_bytes(raw)
            with self.assertRaisesRegex(RuntimeError, "CRC failed"):
                package_submission.verify_archive(corrupt)

            bad_hash = root / "bad-hash.zip"
            write_archive(bad_hash, [("answer.csv", b"ok", 0o644 << 16)], [
                {"path": "answer.csv", "bytes": 2, "sha256": "0" * 64}
            ])
            with self.assertRaisesRegex(RuntimeError, "bytes/hash mismatch"):
                package_submission.verify_archive(bad_hash)

            unsafe = root / "unsafe.zip"
            write_archive(unsafe, [("../answer.csv", b"ok", 0o644 << 16)])
            with self.assertRaisesRegex(RuntimeError, "unsafe archive path"):
                package_submission.verify_archive(unsafe)

            symlink = root / "symlink.zip"
            write_archive(symlink, [
                ("answer.csv", b"target", (0o120777 & 0xFFFF) << 16)
            ])
            with self.assertRaisesRegex(RuntimeError, "symlink member"):
                package_submission.verify_archive(symlink)

            unclosed = root / "unclosed.zip"
            write_archive(unclosed, [
                ("answer.csv", b"ok", 0o644 << 16),
                ("undeclared.txt", b"extra", 0o644 << 16),
            ], [{"path": "answer.csv", "bytes": 2,
                 "sha256": hashlib.sha256(b"ok").hexdigest()}])
            with self.assertRaisesRegex(RuntimeError, "file set is not closed"):
                package_submission.verify_archive(unclosed)

            duplicate = root / "duplicate.zip"
            manifest = {"schema_version": 2, "files": [{
                "path": "answer.csv", "bytes": 2,
                "sha256": hashlib.sha256(b"ok").hexdigest(),
            }]}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(duplicate, "w") as archive:
                    archive.writestr("answer.csv", b"ok")
                    archive.writestr("answer.csv", b"ok")
                    archive.writestr("package_manifest.json",
                                     json.dumps(manifest).encode())
            with self.assertRaisesRegex(RuntimeError, "duplicate member path"):
                package_submission.verify_archive(duplicate)

    def test_flat_archive_extracts_and_runtime_self_check_never_needs_api(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            source = self.make_flat_package(root / "source")
            archive_path = root / "submission.zip"
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for path in sorted(source.rglob("*")):
                    if path.is_file():
                        zf.write(path, path.relative_to(source).as_posix())

            extracted = root / "extracted"
            with zipfile.ZipFile(archive_path) as zf:
                self.assertIsNone(zf.testzip())
                zf.extractall(extracted)

            env = dict(os.environ)
            env.pop("DASHSCOPE_API_KEY", None)
            env.pop("PYTHONPATH", None)
            completed = subprocess.run(
                ["bash", str(extracted / "generate_answer.sh"),
                 "--check-runtime"],
                cwd=extracted, env=env, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=30, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("runtime check OK", completed.stdout)

            manifest = repro.build_runtime_manifest(extracted)
            paths = {entry["path"] for entry in manifest["files"]}
            self.assertIn("agent/repro.py", paths)
            self.assertIn("processed_data/facts.json", paths)
            self.assertIn("official_rules/b榜补充.md", paths)
            self.assertTrue(all(not path.startswith("work/") for path in paths))


if __name__ == "__main__":
    unittest.main()
