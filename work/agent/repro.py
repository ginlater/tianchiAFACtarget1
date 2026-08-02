"""Reproduction utilities: locked logs, config snapshots and reasoning export."""
import ast
import hashlib
import atexit
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from .paths import REPO_DIR, WORK_DIR


RUNTIME_AGENT_MODULES = (
    "__init__.py",
    "paths.py",
    "qwen_client.py",
    "repro.py",
    "retrieval.py",
    "doc_select.py",
    "docsel_fast.py",
    "b_schema.py",
    "answerer.py",
    "batch.py",
    "calc.py",
    "deterministic_calc.py",
    "fin_calc.py",
    "financial_fact_registry.py",
    "insurance_capsules.py",
    "insurance_calc.py",
    "narrative_fact_registry.py",
    "run_b2.py",
)

RUNTIME_CONFIG_FILES = (
    "honest_repro.env",
)

RUNTIME_OFFICIAL_RULES = (
    "比赛规则.md",
    "b榜新增规则.txt",
    "b榜补充.md",
)

RUNTIME_SCRIPT_NAMES = (
    "add_html_summary.py",
    "build_align_matrix.py",
    "build_domain_facts.py",
    "build_evidence.py",
    "build_fin_facts2.py",
    "build_insurance_capsules.py",
    "build_insurance_titles.py",
    "check_reproduction.py",
    "fix_titles.py",
    "package_submission.py",
    "parse_docs.py",
    "rebuild_processed.py",
)

_MANIFEST_CATEGORIES = {
    "source",
    "config",
    "requirements",
    "processed_data",
    "documentation",
    "official_rule",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
FORMAL_MODEL = "qwen3.6-plus"
FORMAL_QUESTION_COUNT = 100


class LockedJsonlWriter:
    """Small file-like JSONL sink safe for concurrent worker threads."""

    def __init__(self, path, mode="a"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, mode, encoding="utf-8")
        self._lock = threading.Lock()
        self._closed = False
        atexit.register(self.close)

    def write(self, text):
        with self._lock:
            if self._closed:
                raise ValueError(f"write to closed log: {self.path}")
            n = self._file.write(text)
            self._file.flush()
            return n

    def flush(self):
        with self._lock:
            if not self._closed:
                self._file.flush()

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._file.flush()
            self._file.close()
            self._closed = True


def _sha256(path):
    _size, digest = _fingerprint(path)
    return digest


def _fingerprint(path):
    """Return byte count and hash from the same file read."""

    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            size += len(chunk)
            h.update(chunk)
    return size, h.hexdigest()


def _safe_repo_file(path, repo_dir):
    """Return ``(resolved_path, repository-relative POSIX path)``.

    Runtime snapshots must never follow a symlink or escape the repository.
    Keeping this check in the run-side implementation also means a malicious
    filename cannot be smuggled into ``run_config.json`` for the packager to
    resolve later.
    """

    repo = Path(repo_dir).expanduser().resolve()
    candidate = Path(path)
    if candidate.is_symlink():
        raise RuntimeError(f"runtime snapshot refuses symlink: {candidate}")
    if not candidate.is_file():
        raise FileNotFoundError(f"runtime snapshot file missing: {candidate}")
    resolved = candidate.resolve()
    try:
        relative = resolved.relative_to(repo)
    except ValueError as exc:
        raise RuntimeError(
            f"runtime snapshot path escapes repository: {candidate}") from exc
    if not relative.parts or any(part in {"", ".", ".."}
                                 for part in relative.parts):
        raise RuntimeError(f"unsafe runtime snapshot path: {relative}")
    return resolved, relative.as_posix()


def _tree_files(root, *, suffix=None):
    """List regular files without following symlinks.

    A symlink anywhere in a frozen tree is rejected, including a symlinked
    directory.  ``.DS_Store`` and Python bytecode are never runtime inputs.
    """

    root = Path(root)
    if root.is_symlink():
        raise RuntimeError(f"runtime snapshot refuses symlink tree: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"runtime snapshot directory missing: {root}")
    out = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"runtime snapshot refuses symlink: {path}")
        if not path.is_file() or path.name == ".DS_Store":
            continue
        if suffix is not None and path.suffix != suffix:
            continue
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        out.append(path)
    return out


def _agent_imports(path):
    """Return local ``agent`` modules imported by one Python source file."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise RuntimeError(f"cannot inspect runtime dependency: {path}") from exc
    names = set()
    for node in ast.walk(tree):
        candidates = []
        if isinstance(node, ast.ImportFrom):
            if node.level:
                if node.module:
                    candidates.append(node.module.split(".", 1)[0])
                else:
                    candidates.extend(alias.name.split(".", 1)[0]
                                      for alias in node.names)
            elif node.module == "agent":
                candidates.extend(alias.name.split(".", 1)[0]
                                  for alias in node.names)
            elif node.module and node.module.startswith("agent."):
                candidates.append(node.module.split(".", 2)[1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("agent."):
                    candidates.append(alias.name.split(".", 2)[1])
        for name in candidates:
            filename = name + ".py"
            names.add(filename)
    return names


def _runtime_layout(root=None):
    """Return ``(manifest_root, runtime_dir, packaged)``.

    The development checkout stores the formal runtime below ``work/`` while
    the submitted archive is intentionally flat: ``agent/``, ``script/`` and
    ``processed_data/`` live at the extraction root.  Reproduction snapshots
    must work in both locations; otherwise the packaged entrypoint would fail
    while trying to rediscover ``<parent>/work/agent``.

    Explicit roots accept either the repository root or an extracted package
    root.  The default uses this module's stable ``WORK_DIR`` and therefore
    remains relocatable even when the extraction directory happens to have an
    arbitrary name.
    """

    if root is None:
        runtime = Path(WORK_DIR).expanduser().resolve()
        parent = runtime.parent
        is_development = (
            runtime.name == "work" and
            (parent / "work").resolve() == runtime and
            (parent / "比赛规则.md").is_file()
        )
        if is_development:
            return parent, runtime, False
        return runtime, runtime, True

    candidate = Path(root).expanduser().resolve()
    if (candidate / "work" / "agent").is_dir() or (
            (candidate / "work").is_dir() and
            not (candidate / "agent").is_dir()):
        return candidate, candidate / "work", False
    return candidate, candidate, True


def validate_runtime_agent_dependencies(repo_dir=None):
    """Require the explicit Agent whitelist to be a closed import graph."""

    root, runtime, _packaged = _runtime_layout(repo_dir)
    agent_dir = runtime / "agent"
    allowed = set(RUNTIME_AGENT_MODULES)
    dependencies = set()
    for name in RUNTIME_AGENT_MODULES:
        path, _relative = _safe_repo_file(agent_dir / name, root)
        dependencies.update(_agent_imports(path))
    outside = sorted(dependencies - allowed)
    if outside:
        raise RuntimeError(
            "formal agent dependency is outside whitelist: " +
            ",".join(outside))
    return tuple(sorted(dependencies))


def runtime_files(repo_dir=None):
    """Return the complete formal-runtime file set.

    Paths are repository-relative in the resulting tuples.  The same function
    is consumed by the packager, so additions and deletions are checked just as
    strictly as byte changes.  The B-round format authorities are frozen next
    to executable inputs because they justify schema overrides such as the
    explicitly named percentage rows.
    """

    root, work, packaged = _runtime_layout(repo_dir)
    specs = []

    def add(category, path, *, optional=False):
        path = Path(path)
        if optional and not path.exists() and not path.is_symlink():
            return
        resolved, relative = _safe_repo_file(path, root)
        specs.append((category, relative, resolved))

    validate_runtime_agent_dependencies(root)
    for name in RUNTIME_AGENT_MODULES:
        add("source", work / "agent" / name)
    for name in RUNTIME_SCRIPT_NAMES:
        add("source", work / "script" / name)
    add("source", work / "generate_answer.sh")

    for name in RUNTIME_CONFIG_FILES:
        add("config", work / "config" / name)
    add("requirements", work / "requirements.txt")
    for path in _tree_files(work / "processed_data"):
        add("processed_data", path)
    add("documentation", work / "README.md")

    if packaged:
        rules = work / "official_rules"
        for name in RUNTIME_OFFICIAL_RULES:
            add("official_rule", rules / name)
        add("official_rule", rules / "upload_b_readme.md", optional=True)
    else:
        for name in RUNTIME_OFFICIAL_RULES:
            add("official_rule", root / name)
        add("official_rule", root / "upload_b" / "readme.md", optional=True)

    paths = [relative for _category, relative, _path in specs]
    if len(paths) != len(set(paths)):
        raise RuntimeError("duplicate path in formal runtime file set")
    return sorted(specs, key=lambda item: item[1])


def build_runtime_manifest(repo_dir=None):
    """Hash every formal runtime input using repository-relative paths."""

    files = []
    for category, relative, path in runtime_files(repo_dir):
        size, digest = _fingerprint(path)
        files.append({
            "category": category,
            "path": relative,
            "bytes": size,
            "sha256": digest,
        })
    return {"schema_version": 1, "root": "repository", "files": files}


def _validate_manifest_path(raw):
    if not isinstance(raw, str) or not raw or "\\" in raw or "\x00" in raw:
        raise RuntimeError(f"unsafe runtime manifest path: {raw!r}")
    path = Path(raw)
    if path.is_absolute() or raw.startswith("/") or any(
            part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"unsafe runtime manifest path: {raw!r}")
    if path.as_posix() != raw:
        raise RuntimeError(f"non-canonical runtime manifest path: {raw!r}")


def validate_runtime_manifest(manifest):
    """Validate the untrusted manifest shape before resolving any path."""

    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("missing or unsupported runtime manifest")
    if manifest.get("root") != "repository" or not isinstance(
            manifest.get("files"), list):
        raise RuntimeError("invalid runtime manifest root/files")
    seen = set()
    for entry in manifest["files"]:
        if not isinstance(entry, dict):
            raise RuntimeError("invalid runtime manifest entry")
        if set(entry) != {"category", "path", "bytes", "sha256"}:
            raise RuntimeError("invalid runtime manifest entry fields")
        if entry["category"] not in _MANIFEST_CATEGORIES:
            raise RuntimeError(
                f"invalid runtime manifest category: {entry['category']!r}")
        _validate_manifest_path(entry["path"])
        if entry["path"] in seen:
            raise RuntimeError(
                f"duplicate runtime manifest path: {entry['path']}")
        seen.add(entry["path"])
        if (not isinstance(entry["bytes"], int) or
                isinstance(entry["bytes"], bool) or entry["bytes"] < 0):
            raise RuntimeError(
                f"invalid byte count for {entry['path']}")
        if (not isinstance(entry["sha256"], str) or
                not _SHA256_RE.fullmatch(entry["sha256"])):
            raise RuntimeError(f"invalid sha256 for {entry['path']}")
    return manifest


def verify_runtime_manifest(manifest, repo_dir=None):
    """Require the current formal runtime tree to equal a frozen snapshot."""

    frozen = validate_runtime_manifest(manifest)
    current = build_runtime_manifest(repo_dir)
    old = {entry["path"]: entry for entry in frozen["files"]}
    new = {entry["path"]: entry for entry in current["files"]}
    missing = sorted(set(old) - set(new))
    added = sorted(set(new) - set(old))
    changed = sorted(path for path in set(old) & set(new)
                     if old[path] != new[path])
    if missing or added or changed:
        pieces = []
        if missing:
            pieces.append("missing=" + ",".join(missing[:8]))
        if added:
            pieces.append("added=" + ",".join(added[:8]))
        if changed:
            pieces.append("changed=" + ",".join(changed[:8]))
        raise RuntimeError("formal runtime differs from run_config: " +
                           "; ".join(pieces))
    return current


def build_input_manifest(qdir, submit_template):
    """Fingerprint the exact question files and submission template.

    Input manifests are intentionally independent from the runtime manifest:
    the latter freezes code/data while this one proves which official test
    bundle a run consumed.  Symlinks are rejected so a path cannot change its
    target between the run and package gates.
    """

    directory = Path(qdir)
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f"unsafe or missing question directory: {directory}")
    files = []
    for path in sorted(directory.iterdir()):
        if path.is_symlink():
            raise RuntimeError(f"input manifest refuses symlink: {path}")
        if path.is_file() and path.suffix in (".json", ".jsonl"):
            files.append(path)
    template = Path(submit_template)
    if template.is_symlink() or not template.is_file():
        raise RuntimeError(
            f"unsafe or missing submission template: {template}")
    files.append(template)
    names = [path.name for path in files]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate basename in input manifest")
    return [{"name": path.name, "bytes": path.stat().st_size,
             "sha256": _sha256(path)} for path in files]


# Backward-compatible private alias for callers/tests from pre-schema-2 runs.
_input_manifest = build_input_manifest


def _portable_repo_value(value, repo_dir=None):
    """Turn an absolute repository-internal path into a POSIX relative path."""

    if not isinstance(value, (str, os.PathLike)):
        return value
    text = os.fspath(value)
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        return text
    root = Path(repo_dir or _runtime_layout()[0]).expanduser().resolve()
    try:
        return candidate.resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        # External inputs are intentionally retained as absolute paths: they
        # cannot be made repository-relative without changing their meaning.
        return text


def validate_complete_run_config(config, expected_question_ids=None):
    """Validate the non-secret controls for one formal 100-question run.

    This pure validator is shared by the offline checker and packager.  It is
    deliberately strict for schema 2; legacy run configs remain readable by
    the checker but are never packageable as a new formal delivery.
    """

    if not isinstance(config, dict) or config.get("schema_version") != 2:
        raise RuntimeError("formal packaging requires run_config schema_version=2")
    qids = config.get("question_ids")
    if (not isinstance(qids, list) or
            any(not isinstance(qid, str) or not qid.strip() for qid in qids) or
            len(qids) != FORMAL_QUESTION_COUNT or
            len(set(qids)) != FORMAL_QUESTION_COUNT):
        raise RuntimeError(
            f"formal run must contain {FORMAL_QUESTION_COUNT} ordered unique qids")
    if config.get("question_count") != FORMAL_QUESTION_COUNT:
        raise RuntimeError("run_config question_count is not 100")
    if expected_question_ids is not None and qids != list(expected_question_ids):
        raise RuntimeError("run_config question_ids do not match official order")
    if config.get("model") != FORMAL_MODEL:
        raise RuntimeError(f"formal model must be exactly {FORMAL_MODEL}")
    if config.get("verify_model") != FORMAL_MODEL:
        raise RuntimeError(f"formal verify_model must be exactly {FORMAL_MODEL}")

    arguments = config.get("arguments")
    if not isinstance(arguments, dict):
        raise RuntimeError("run_config arguments must be an object")
    expected = {
        "model": FORMAL_MODEL,
        "verify_model": FORMAL_MODEL,
        "resume": False,
        "fresh_digests": True,
        "limit": 0,
        "qids": "",
        "batch": True,
    }
    for key, value in expected.items():
        if arguments.get(key) != value:
            raise RuntimeError(
                f"formal run_config arguments.{key} must be {value!r}")
    return config


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=WORK_DIR,
            stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _git_dirty():
    """Record repository dirtiness without serialising filenames or content."""

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=WORK_DIR, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=True)
        return bool(result.stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        # No Git metadata is itself an unaudited state, so fail conservative.
        return True


def write_run_config(path, args, qdir, submit_template, question_ids=None):
    """Persist every effective non-secret setting needed to replay a run."""
    packages = {}
    for name in ("openai", "httpx", "PyMuPDF", "pdfplumber", "lxml"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    qids = list(question_ids or [])
    manifest_root = _runtime_layout()[0]
    config = {
        "schema_version": 2,
        "created_at": time.time(),
        "command": [_portable_repo_value(value, manifest_root)
                    for value in sys.argv],
        "arguments": {k: str(v) if isinstance(v, Path) else v
                      for k, v in vars(args).items()},
        "environment": {k: v for k, v in sorted(os.environ.items())
                        if k.startswith("AFAC_") or k == "PYTHONHASHSEED"},
        "model": args.model,
        "verify_model": (args.verify_model
                         or os.environ.get("AFAC_VERIFY_MODEL")
                         or args.model),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "runtime_manifest": build_runtime_manifest(),
        "inputs": build_input_manifest(qdir, submit_template),
        "input_paths": {
            "qdir": _portable_repo_value(
                str(Path(qdir).expanduser().resolve()), manifest_root),
            "submit_template": _portable_repo_value(
                str(Path(submit_template).expanduser().resolve()),
                manifest_root),
        },
        "question_ids": qids,
        "question_count": len(qids),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    return config


def collect_reasonings(log_path, qids):
    """Collect the actual answer-call explanation for each qid, without API use."""
    wanted = set(qids)
    out = {}
    p = Path(log_path)
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = row.get("qid")
            if qid not in wanted:
                continue
            text = row.get("reasoning")
            if not text:
                for key in ("c3", "c1", "c1b", "c2"):
                    if row.get(key):
                        text = row[key]
                        break
            if text:
                out[qid] = str(text).strip()
    return out
