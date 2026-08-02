#!/usr/bin/env python3
"""Build an auditable reproduction archive from one complete honest run.

The command has no historical/default run name: callers must explicitly name
the single run being packaged.  The run is checked against its API audit and
token ledger before any archive is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import tempfile
import zipfile


WORK = pathlib.Path(__file__).resolve().parents[1]
REPO = WORK.parent
CHECKER = WORK / "script" / "check_reproduction.py"
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

from agent import b_schema  # noqa: E402
from agent.repro import (build_input_manifest, runtime_files,  # noqa: E402
                         validate_complete_run_config,
                         validate_runtime_manifest,
                         verify_runtime_manifest)

RUN_FILES = (
    "answer.csv",
    "evidence.json",
    "answers.json",
    "reasonings.json",
    "reasoning_sources.json",
    "api_calls.jsonl",
    "token_ledger.json",
    "run_log.jsonl",
    "docsel_log.jsonl",
    "run_config.json",
)

_SECRET_PATTERNS = (
    re.compile(rb"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(rb"DASHSCOPE_API_KEY\s*=\s*[^\s\"']{8,}"),
    re.compile(rb"api[_-]?key\s*[=:]\s*[\"'][A-Za-z0-9_-]{16,}", re.I),
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def checked_bytes(path: pathlib.Path) -> bytes:
    if path.is_symlink():
        raise RuntimeError(f"refusing to package symlink: {path}")
    data = path.read_bytes()
    for pattern in _SECRET_PATTERNS:
        if pattern.search(data):
            raise RuntimeError(f"secret-like content found in {path}")
    return data


def runtime_archive_destination(relative: str) -> str:
    """Map a frozen runtime path to the rule-mandated flat ZIP layout."""

    path = pathlib.PurePosixPath(relative)
    parts = path.parts
    if parts[:2] == ("work", "agent") and len(parts) >= 3:
        tail = pathlib.PurePosixPath(*parts[2:]).as_posix()
        return "agent/" + tail
    if parts[:2] == ("work", "script") and len(parts) >= 3:
        tail = pathlib.PurePosixPath(*parts[2:]).as_posix()
        return "script/" + tail
    if parts[:2] == ("work", "processed_data") and len(parts) >= 3:
        tail = pathlib.PurePosixPath(*parts[2:]).as_posix()
        return "processed_data/" + tail
    if parts[:2] == ("work", "config") and len(parts) >= 3:
        tail = pathlib.PurePosixPath(*parts[2:]).as_posix()
        return "config/" + tail
    if relative == "work/requirements.txt":
        return "requirements.txt"
    if relative == "work/generate_answer.sh":
        return "generate_answer.sh"
    if relative == "work/README.md":
        return "README.md"
    if relative == "比赛规则.md":
        return "official_rules/比赛规则.md"
    if relative in {"b榜新增规则.txt", "b榜补充.md"}:
        return "official_rules/" + relative
    if relative == "upload_b/readme.md":
        return "official_rules/upload_b_readme.md"

    # When invoked from an already extracted package, runtime manifests are
    # rooted at that flat directory rather than at a development checkout.
    if parts and parts[0] in {"agent", "script", "processed_data", "config"}:
        return relative
    if relative in {"requirements.txt", "generate_answer.sh", "README.md"}:
        return relative
    if parts and parts[0] == "official_rules":
        return relative
    raise RuntimeError(f"no archive destination for runtime path: {relative}")


def load_frozen_runtime(run_dir: pathlib.Path) -> dict:
    """Load and validate the run-side snapshot without following a symlink."""

    path = run_dir / "run_config.json"
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"unsafe or missing run_config.json: {path}")
    data = checked_bytes(path)
    try:
        config = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid run_config.json: {path}") from exc
    manifest = validate_runtime_manifest(config.get("runtime_manifest"))
    if not isinstance(config.get("git_dirty"), bool):
        raise RuntimeError("run_config.json is missing boolean git_dirty")
    verify_runtime_manifest(manifest, REPO)
    return config


def validate_official_run(config: dict) -> dict:
    """Bind a package build to the repository's current official B inputs."""

    qdir = REPO / "upload_b" / "question_b"
    submit = REPO / "upload_b" / "submit.csv"
    question_ids = [q["qid"] for q in b_schema.load_questions(qdir)]
    validate_complete_run_config(config, question_ids)
    current_inputs = build_input_manifest(qdir, submit)
    if config.get("inputs") != current_inputs:
        raise RuntimeError(
            "run_config input manifest differs from current official inputs")
    paths = config.get("input_paths")
    expected_paths = {
        "qdir": "upload_b/question_b",
        "submit_template": "upload_b/submit.csv",
    }
    if paths != expected_paths:
        raise RuntimeError(
            "formal run_config input_paths must name current official inputs")
    return config


def safe_output_path(raw_output: pathlib.Path) -> pathlib.Path:
    """Resolve an archive target without allowing it to mutate frozen trees."""

    candidate = raw_output.expanduser()
    if candidate.is_symlink():
        raise RuntimeError(f"refusing symlink archive target: {candidate}")
    output = candidate.resolve()
    if output.suffix.lower() != ".zip":
        raise RuntimeError(f"archive target must end in .zip: {output}")
    protected_dirs = (
        WORK / "agent",
        WORK / "script",
        WORK / "config",
        WORK / "processed_data",
        REPO / "upload_b",
    )
    for directory in protected_dirs:
        resolved = directory.resolve()
        if output == resolved or resolved in output.parents:
            raise RuntimeError(
                f"archive target is inside frozen runtime tree: {output}")
    protected_files = {
        (WORK / "requirements.txt").resolve(),
        (WORK / "generate_answer.sh").resolve(),
        (WORK / "README.md").resolve(),
        (REPO / "比赛规则.md").resolve(),
    }
    if output in protected_files:
        raise RuntimeError(f"archive target overwrites frozen file: {output}")
    return output


def collect(run_dir: pathlib.Path) -> list[tuple[pathlib.Path, str]]:
    entries: list[tuple[pathlib.Path, str]] = []
    for name in RUN_FILES:
        path = run_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"complete run is missing {path}")
        destination = (name if name in {"answer.csv", "evidence.json"}
                       else "logs/" + name)
        entries.append((path, destination))

    for _category, relative, path in runtime_files(REPO):
        entries.append((path, runtime_archive_destination(relative)))
    return entries


def _validate_archive_path(name: str) -> None:
    """Reject ZIP-slip, platform-ambiguous and non-canonical member names."""

    if (not isinstance(name, str) or not name or "\\" in name or
            "\x00" in name or name.endswith("/") or name.startswith("/")):
        raise RuntimeError(f"unsafe archive path: {name!r}")
    path = pathlib.PurePosixPath(name)
    if (path.is_absolute() or path.as_posix() != name or
            any(part in {"", ".", ".."} for part in path.parts) or
            (path.parts and re.fullmatch(r"[A-Za-z]:", path.parts[0]))):
        raise RuntimeError(f"unsafe archive path: {name!r}")


def verify_archive(path: pathlib.Path) -> dict:
    """Re-open a completed ZIP and prove its manifest is a byte-level closure."""

    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise RuntimeError("duplicate member path in generated archive")
        for info in infos:
            _validate_archive_path(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_IFMT(mode) == stat.S_IFLNK:
                raise RuntimeError(
                    f"symlink member in generated archive: {info.filename}")
            if info.is_dir():
                raise RuntimeError(
                    f"unexpected directory member in archive: {info.filename}")
            if info.flag_bits & 0x1:
                raise RuntimeError(
                    f"encrypted member in generated archive: {info.filename}")
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"archive CRC failed for {bad}")
        if names.count("package_manifest.json") != 1:
            raise RuntimeError("archive must contain one package_manifest.json")
        try:
            manifest = json.loads(
                archive.read("package_manifest.json").decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("invalid package_manifest.json") from exc
        if (not isinstance(manifest, dict) or
                manifest.get("schema_version") != 2 or
                not isinstance(manifest.get("files"), list)):
            raise RuntimeError("unsupported package manifest")

        declared = {}
        for pos, entry in enumerate(manifest["files"], 1):
            if (not isinstance(entry, dict) or
                    set(entry) != {"path", "bytes", "sha256"}):
                raise RuntimeError(
                    f"invalid package manifest file entry {pos}")
            member = entry["path"]
            _validate_archive_path(member)
            if member == "package_manifest.json" or member in declared:
                raise RuntimeError(
                    f"duplicate/reserved package manifest path: {member}")
            if (not isinstance(entry["bytes"], int) or
                    isinstance(entry["bytes"], bool) or entry["bytes"] < 0):
                raise RuntimeError(
                    f"invalid package manifest byte count for {member}")
            if (not isinstance(entry["sha256"], str) or
                    not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
                raise RuntimeError(
                    f"invalid package manifest sha256 for {member}")
            declared[member] = entry

        actual_names = set(names) - {"package_manifest.json"}
        if set(declared) != actual_names:
            missing = sorted(actual_names - set(declared))
            extra = sorted(set(declared) - actual_names)
            raise RuntimeError(
                "package manifest file set is not closed: "
                f"missing={missing[:8]}, extra={extra[:8]}")
        by_name = {info.filename: info for info in infos}
        for member, entry in declared.items():
            data = archive.read(by_name[member])
            if (len(data) != entry["bytes"] or
                    sha256_bytes(data) != entry["sha256"]):
                raise RuntimeError(
                    f"package manifest bytes/hash mismatch for {member}")
    return manifest


def build(run_dir: pathlib.Path, output: pathlib.Path) -> pathlib.Path:
    raw_run_dir = run_dir.expanduser()
    if raw_run_dir.is_symlink():
        raise RuntimeError(f"refusing symlink run directory: {raw_run_dir}")
    run_dir = raw_run_dir.resolve()
    output = safe_output_path(output)
    run_config = load_frozen_runtime(run_dir)
    validate_official_run(run_config)
    subprocess.run([sys.executable, str(CHECKER), str(run_dir)], check=True)

    entries = collect(run_dir)
    archive_names = [name for _path, name in entries]
    if len(archive_names) != len(set(archive_names)):
        raise RuntimeError("duplicate archive path")
    if any(".env" in pathlib.PurePosixPath(name).name and
           not name.endswith("honest_repro.env") for name in archive_names):
        raise RuntimeError("unexpected environment file in archive")

    frozen = run_config["runtime_manifest"]
    frozen_by_path = {entry["path"]: entry for entry in frozen["files"]}
    frozen_by_source = {
        (REPO / relative).resolve(): entry
        for relative, entry in frozen_by_path.items()
    }
    collected_runtime = {
        path.resolve() for path, _destination in entries
        if path.resolve() in frozen_by_source
    }
    if collected_runtime != set(frozen_by_source):
        missing = sorted(str(path) for path in
                         set(frozen_by_source) - collected_runtime)
        raise RuntimeError("frozen runtime file omitted from archive: " +
                           ",".join(missing[:8]))

    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_files = []
    with tempfile.NamedTemporaryFile(
            prefix=".afac-package-", suffix=".zip", dir=output.parent,
            delete=False) as temporary:
        temporary_path = pathlib.Path(temporary.name)
    try:
        with zipfile.ZipFile(temporary_path, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=6) as archive:
            for path, destination in entries:
                data = checked_bytes(path)
                frozen_entry = frozen_by_source.get(path.resolve())
                if frozen_entry is not None:
                    if (len(data) != frozen_entry["bytes"] or
                            sha256_bytes(data) != frozen_entry["sha256"]):
                        raise RuntimeError(
                            "formal runtime changed during packaging: " +
                            frozen_entry["path"])
                info = zipfile.ZipInfo(destination)
                info.compress_type = zipfile.ZIP_DEFLATED
                # Stable archive metadata and executable bit for the entrypoint.
                info.date_time = (2026, 7, 25, 0, 0, 0)
                mode = 0o755 if destination.endswith("generate_answer.sh") else 0o644
                info.external_attr = (mode & 0xFFFF) << 16
                archive.writestr(info, data)
                manifest_files.append({
                    "path": destination,
                    "bytes": len(data),
                    "sha256": sha256_bytes(data),
                })
            manifest = {
                "schema_version": 2,
                "source_run": run_dir.name,
                "model": run_config.get("model"),
                "git_commit": run_config.get("git_commit"),
                "git_dirty": run_config.get("git_dirty"),
                "runtime_manifest_sha256": sha256_bytes(
                    json.dumps(frozen, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":")).encode("utf-8")),
                "input_manifest_sha256": sha256_bytes(
                    json.dumps(run_config["inputs"], ensure_ascii=False,
                               sort_keys=True, separators=(",", ":"))
                    .encode("utf-8")),
                "files": manifest_files,
            }
            payload = (json.dumps(manifest, ensure_ascii=False, indent=2,
                                  sort_keys=True) + "\n").encode("utf-8")
            info = zipfile.ZipInfo("package_manifest.json")
            info.date_time = (2026, 7, 25, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o644 & 0xFFFF) << 16
            archive.writestr(info, payload)

        # Independently prove the generated ZIP is a safe, manifest-closed
        # regular-file archive before replacing an older deliverable.
        verify_archive(temporary_path)
        os.replace(temporary_path, output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    print(f"reproduction archive: {output} "
          f"({output.stat().st_size / 1_000_000:.1f} MB, "
          f"{len(entries) + 1} files)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=pathlib.Path,
                        help="one complete output directory from generate_answer.sh")
    parser.add_argument("--output", required=True, type=pathlib.Path,
                        help="target .zip path")
    args = parser.parse_args()
    build(args.run_dir, args.output)


if __name__ == "__main__":
    main()
