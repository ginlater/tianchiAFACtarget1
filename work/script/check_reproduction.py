#!/usr/bin/env python3
"""Strict, zero-API validation of one honest reproduction output directory.

The checker deliberately joins the independently written artifacts back
together.  A directory is accepted only when the submitted rows, final answer
objects, reasoning provenance, raw API audit, token ledger, execution logs and
evidence bundle describe the same run.
"""
import csv
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import json
import pathlib
import re
import sys


WORK = pathlib.Path(__file__).resolve().parents[1]
if str(WORK) not in sys.path:
    sys.path.insert(0, str(WORK))

from agent import b_schema  # noqa: E402
from agent.repro import (FORMAL_MODEL, build_input_manifest,  # noqa: E402
                         validate_complete_run_config)


REQUIRED = (
    "answer.csv", "answers.json", "reasonings.json", "reasoning_sources.json",
    "run_config.json", "api_calls.jsonl", "run_log.jsonl",
    "docsel_log.jsonl", "token_ledger.json", "evidence.json",
)

_SUSPICIOUS_REASONING_PATTERNS = (
    re.compile(r"答案(?:键|库)"),
    re.compile(r"(?:参考|标准|官方|预置|目标)答案"),
    re.compile(r"(?:榜单|排行榜)(?:答案|反馈|结果|分数)"),
    re.compile(r"对键"),
    re.compile(r"(?<![A-Za-z0-9_])(?:answer[-_ ]?key|ground[-_ ]?truth|"
               r"gold(?:en)?[-_ ]?answer|leaderboard|b_?router6|router6)"
               r"(?![A-Za-z0-9_])",
               re.I),
)


def fail(message):
    raise SystemExit(f"REPRO CHECK FAILED: {message}")


def load_json(path, label=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"invalid {label or pathlib.Path(path).name}: {exc}")


def jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for no, line in enumerate(f, 1):
            if not line.strip():
                fail(f"blank JSONL line {path.name}:{no}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"invalid JSONL {path.name}:{no}: {exc}")
            if not isinstance(row, dict):
                fail(f"JSONL row is not an object {path.name}:{no}")
            rows.append(row)
    return rows


def token_int(value, label):
    # bool is an int subclass but is never a valid raw token count.
    if isinstance(value, bool):
        fail(f"invalid integer for {label}: {value!r}")
    try:
        number = int(value)
    except (TypeError, ValueError):
        fail(f"invalid integer for {label}: {value!r}")
    if str(value).strip() not in {str(number), f"+{number}"}:
        fail(f"non-integral token value for {label}: {value!r}")
    if number < 0:
        fail(f"negative token value for {label}: {number}")
    return number


def token_pair(value, label):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        fail(f"invalid token pair for {label}: {value!r}")
    return (token_int(value[0], f"{label}.prompt_tokens"),
            token_int(value[1], f"{label}.completion_tokens"))


def normalize_reasoning(value):
    return str(value or "").replace("\n", " ").strip()


def suspicious_reasoning_marker(value):
    """Return the forbidden answer-key marker found in a reasoning, if any."""

    text = str(value or "")
    for pattern in _SUSPICIOUS_REASONING_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return ""


def answer_slots(value, label):
    if not isinstance(value, list) or not 1 <= len(value) <= 4:
        fail(f"{label} must be a list with 1-4 answer slots")
    slots = [str(x if x is not None else "").strip() for x in value]
    if not any(slots):
        fail(f"empty answer slots for {label}")
    return tuple(slots + [""] * (4 - len(slots)))


_NUMBER = re.compile(r"^[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(%?)$")


def _value_equivalent(left, right):
    """Compare a parsed raw answer with one final formatted slot.

    The final JSON/CSV comparisons are byte-level after trimming.  This looser
    comparison is only for an upstream trace or run-log answer: a parser may
    legitimately retain ``67.1`` while the schema formatter emits ``67.10``.
    """
    a = re.sub(r"\s+", "", str(left or "")).strip("'\"。")
    b = re.sub(r"\s+", "", str(right or "")).strip("'\"。")
    a = re.sub(r"^(?:答案|answer)[:：]", "", a, flags=re.I)
    b = re.sub(r"^(?:答案|answer)[:：]", "", b, flags=re.I)
    a = a.replace("＞", ">").replace("％", "%")
    b = b.replace("＞", ">").replace("％", "%")
    if a == b:
        return True

    # Multiple-choice ordering is semantically irrelevant, although the final
    # formatted answer is checked separately for canonical storage.
    if re.fullmatch(r"[A-D]+", a) and re.fullmatch(r"[A-D]+", b):
        return "".join(sorted(set(a))) == "".join(sorted(set(b)))

    ma, mb = _NUMBER.fullmatch(a), _NUMBER.fullmatch(b)
    if ma and mb and bool(ma.group(1)) == bool(mb.group(1)):
        try:
            return Decimal(a.rstrip("%").replace(",", "")) == \
                Decimal(b.rstrip("%").replace(",", ""))
        except InvalidOperation:
            pass

    def date_parts(text):
        match = re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text)
        if match:
            return tuple(map(int, match.groups()))
        match = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", text)
        return tuple(map(int, match.groups())) if match else None

    da, db = date_parts(a), date_parts(b)
    return da is not None and da == db


def answer_equivalent(raw, final_slots):
    slots = [slot for slot in final_slots if slot]
    if not slots:
        return False
    if isinstance(raw, list):
        pieces = [str(x).strip() for x in raw]
    else:
        text = str(raw or "").strip()
        pieces = re.split(r"[；;]", text) if len(slots) > 1 else [text]
    if len(pieces) != len(slots):
        return False
    return all(_value_equivalent(piece, slot)
               for piece, slot in zip(pieces, slots))


def evidence_entries(path):
    """Return qid-keyed entries from supported evidence layouts."""
    payload = load_json(path, "evidence.json")
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = None
        for key in ("questions", "items", "evidence"):
            if isinstance(payload.get(key), list):
                entries = payload[key]
                break
        if entries is None:
            entries = []
            for qid, value in payload.items():
                if not isinstance(value, dict):
                    fail("evidence keyed-object values must be objects")
                entry = dict(value)
                entry.setdefault("qid", qid)
                entries.append(entry)
    else:
        fail("evidence.json must contain a list or object")

    result = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict):
            fail(f"evidence entry {i} is not an object")
        qid = str(entry.get("qid") or "").strip()
        if not qid:
            fail(f"evidence entry {i} has no qid")
        if qid in result:
            fail(f"duplicate qid in evidence.json: {qid}")
        if not any(v not in (None, "", [], {}) for k, v in entry.items()
                   if k != "qid"):
            fail(f"evidence entry has no payload: {qid}")
        result[qid] = entry
    return payload, result


def validate_complete_evidence(entry, qid):
    """Require concrete, non-missing cited clauses for a schema-2 question."""

    retrieval = entry.get("retrieval")
    if not isinstance(retrieval, dict):
        fail(f"schema2 evidence has no retrieval object for {qid}")
    selected = retrieval.get("selected_doc_ids")
    evidence_ids = retrieval.get("evidence_ids")
    chunks = retrieval.get("evidence_retrieval")
    if (not isinstance(selected, list) or not selected or
            any(not str(value).strip() for value in selected)):
        fail(f"schema2 evidence has no selected documents for {qid}")
    if (not isinstance(evidence_ids, list) or not evidence_ids or
            any(not str(value).strip() for value in evidence_ids)):
        fail(f"schema2 evidence has no evidence_ids for {qid}")
    if not isinstance(chunks, list) or not chunks:
        fail(f"schema2 evidence has no cited chunks for {qid}")
    seen = []
    for pos, chunk in enumerate(chunks, 1):
        if not isinstance(chunk, dict):
            fail(f"schema2 evidence chunk is not an object for {qid}#{pos}")
        if chunk.get("missing"):
            fail(f"schema2 evidence contains missing chunk for {qid}#{pos}")
        for field in ("evidence_id", "doc_id", "quoted_clause"):
            if not str(chunk.get(field) or "").strip():
                fail(f"schema2 evidence chunk lacks {field} for {qid}#{pos}")
        seen.append(str(chunk["evidence_id"]))
    if seen != [str(value) for value in evidence_ids]:
        fail(f"schema2 evidence_ids/chunks order mismatch for {qid}")


def validate_schema2_api_record(record, line):
    """Require every recorded attempt in a formal run to be a target-model success."""

    if record.get("status") != "ok":
        fail(f"schema2 API attempt is not successful at line {line}")
    if record.get("model") != FORMAL_MODEL:
        fail(f"schema2 API model must be exactly {FORMAL_MODEL} at line {line}")


def validate_schema2_evidence_provenance(payload, config):
    """Bind portable evidence metadata to the exact checked run config."""

    if not isinstance(payload, dict):
        fail("schema2 evidence must use the object layout")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        fail("schema2 evidence has no provenance object")
    if provenance.get("source_directory") != ".":
        fail("schema2 evidence source_directory must be portable '.'")
    if provenance.get("run_config") != config:
        fail("schema2 evidence run_config differs from run_config.json")


def _resolve_config_input(raw):
    path = pathlib.Path(str(raw or "")).expanduser()
    if path.is_absolute():
        return path
    candidates = (WORK.parent / path, WORK / path)
    return next((candidate for candidate in candidates if candidate.exists()),
                path)


def validate_schema2_inputs(config, question_ids):
    """Re-read the configured inputs and verify their order and fingerprints."""

    paths = config.get("input_paths")
    if not isinstance(paths, dict):
        fail("schema2 run_config has no input_paths object")
    qdir = _resolve_config_input(paths.get("qdir"))
    submit = _resolve_config_input(paths.get("submit_template"))
    try:
        loaded_qids = [q["qid"] for q in b_schema.load_questions(qdir)]
        current_manifest = build_input_manifest(qdir, submit)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError,
            json.JSONDecodeError) as exc:
        fail(f"schema2 inputs cannot be verified: {exc}")
    if loaded_qids != list(question_ids):
        fail("schema2 configured question order differs from answer.csv")
    if config.get("inputs") != current_manifest:
        fail("schema2 input manifest differs from configured input files")


def _call_signature(qid, model, tag, prompt, completion, recipients):
    return (str(qid or ""), str(model or ""), str(tag or ""),
            prompt, completion, tuple(recipients))


def _api_attached_to(record, qid):
    """Whether one raw API attempt was charged/attached to ``qid``."""

    recipients = record.get("allocation_qids") or []
    return qid in recipients if recipients else record.get("qid") == qid


def _response_contains_reasoning(record, reasoning):
    """Require the submitted explanation to be literal visible API output."""

    if record.get("status") != "ok" or not reasoning:
        return False
    response = record.get("response")
    if not isinstance(response, dict):
        return False
    return any(reasoning in str(response.get(key) or "")
               for key in ("content", "reasoning_content"))


def _embedded_record(record, marker, label):
    if not isinstance(record, dict):
        fail(f"{label} reference is not an object")
    line = token_int(record.get(marker), f"{label}.{marker}")
    if line < 1:
        fail(f"{label}.{marker} must be 1-based")
    clean = dict(record)
    clean.pop(marker, None)
    return line, clean


def main(out):
    out = pathlib.Path(out)
    for name in REQUIRED:
        if not (out / name).is_file():
            fail(f"missing {name}")

    # Load the run contract first so subsequent gates can apply schema-2-only
    # requirements while preserving compatibility with historical fixtures.
    cfg_text = (out / "run_config.json").read_text(encoding="utf-8")
    if "DASHSCOPE_API_KEY" in cfg_text or "sk-" in cfg_text:
        fail("secret-like text found in run_config.json")
    try:
        config = json.loads(cfg_text)
    except json.JSONDecodeError as exc:
        fail(f"invalid run_config.json: {exc}")
    if not isinstance(config, dict):
        fail("run_config.json must contain an object")
    schema_version = config.get("schema_version", 1)
    if schema_version not in (1, 2):
        fail(f"unsupported run_config schema_version: {schema_version!r}")
    strict_schema2 = schema_version == 2

    # ------------------------------------------------------------------ CSV
    with open(out / "answer.csv", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames
    expected_header = ["qid", "answer_1", "answer_2", "answer_3", "answer_4",
                       "prompt_tokens", "completion_tokens", "total_tokens",
                       "reasoning"]
    if header != expected_header:
        fail("unexpected answer.csv header")
    if not rows or rows[0]["qid"] != "summary":
        fail("first data row must be summary")
    questions = rows[1:]
    question_ids = [str(row.get("qid") or "").strip() for row in questions]
    if (len(questions) != 100 or any(not qid for qid in question_ids)
            or len(set(question_ids)) != 100):
        fail(f"expected 100 unique nonempty questions, got {len(questions)}")
    qid_set = set(question_ids)
    if strict_schema2:
        try:
            validate_complete_run_config(config, question_ids)
        except RuntimeError as exc:
            fail(str(exc))
        validate_schema2_inputs(config, question_ids)

    csv_answers = {}
    csv_reasonings = {}
    csv_usage = {}
    for qid, row in zip(question_ids, questions):
        slots = tuple((row[f"answer_{i}"] or "").strip()
                      for i in range(1, 5))
        if not any(slots):
            fail(f"empty answer: {qid}")
        reasoning = (row["reasoning"] or "").strip()
        if len(reasoning) < 20:
            fail(f"short reasoning: {qid}")
        csv_answers[qid] = slots
        csv_reasonings[qid] = reasoning

        rp = token_int(row["prompt_tokens"], f"{qid}.prompt_tokens")
        rc = token_int(row["completion_tokens"], f"{qid}.completion_tokens")
        rt = token_int(row["total_tokens"], f"{qid}.total_tokens")
        if rt != rp + rc:
            fail(f"row token sum mismatch for {qid}: {rp}+{rc}!={rt}")
        csv_usage[qid] = (rp, rc, rt)

    qp = sum(v[0] for v in csv_usage.values())
    qc = sum(v[1] for v in csv_usage.values())
    summary = rows[0]
    declared = (
        token_int(summary["prompt_tokens"], "summary.prompt_tokens"),
        token_int(summary["completion_tokens"], "summary.completion_tokens"),
        token_int(summary["total_tokens"], "summary.total_tokens"),
    )
    if declared != (qp, qc, qp + qc):
        fail(f"CSV token sum mismatch: {declared} vs {(qp, qc, qp + qc)}")

    # ----------------------------------------- answers/reasonings/provenance
    answers = load_json(out / "answers.json")
    reasonings = load_json(out / "reasonings.json")
    sources = load_json(out / "reasoning_sources.json")
    for label, obj in (("answers.json", answers),
                       ("reasonings.json", reasonings),
                       ("reasoning_sources.json", sources)):
        if not isinstance(obj, dict):
            fail(f"{label} must contain an object")
        missing = sorted(qid_set - set(obj))
        extra = sorted(set(obj) - qid_set)
        if missing or extra:
            fail(f"{label} qid mismatch: missing={missing}, extra={extra}")

    normalized_answers = {}
    for qid in question_ids:
        normalized_answers[qid] = answer_slots(
            answers[qid], f"answers.json.{qid}")
        if normalized_answers[qid] != csv_answers[qid]:
            fail(f"answers.json/CSV answer mismatch for {qid}: "
                 f"JSON={normalized_answers[qid]} CSV={csv_answers[qid]}")
        if not isinstance(reasonings[qid], str):
            fail(f"reasonings.json value is not text for {qid}")
        if normalize_reasoning(reasonings[qid]) != csv_reasonings[qid]:
            fail(f"reasonings.json/CSV reasoning mismatch for {qid}")
        if strict_schema2:
            marker = suspicious_reasoning_marker(reasonings[qid])
            if marker:
                fail(f"reasoning contains suspicious answer-key marker for "
                     f"{qid}: {marker!r}")

        source = sources[qid]
        if not isinstance(source, dict):
            fail(f"reasoning source is not an object for {qid}")
        if source.get("reasoning") != reasonings[qid]:
            fail(f"reasoning_sources/reasonings mismatch for {qid}")
        stage = str(source.get("reasoning_stage") or "").strip()
        traces = source.get("traces")
        if not stage or not isinstance(traces, list) or not traces:
            fail(f"reasoning source has no selected stage/traces for {qid}")
        selected = []
        for pos, trace in enumerate(traces, 1):
            if not isinstance(trace, dict):
                fail(f"reasoning trace is not an object for {qid}#{pos}")
            if not str(trace.get("stage") or "").strip():
                fail(f"reasoning trace has no stage for {qid}#{pos}")
            if not str(trace.get("content") or "").strip():
                fail(f"reasoning trace has no content for {qid}#{pos}")
            if trace.get("answer") in (None, "", []):
                fail(f"reasoning trace has no answer for {qid}#{pos}")
            if (trace.get("stage") == stage
                    and trace.get("content") == reasonings[qid]):
                selected.append(trace)
        if not selected:
            fail(f"selected reasoning does not correspond to a trace for {qid}")
        if not any(answer_equivalent(t.get("answer"), csv_answers[qid])
                   for t in selected):
            fail(f"selected reasoning trace disagrees with final answer for {qid}")
        if source.get("raw_answer") not in (None, "", []):
            if not answer_equivalent(source["raw_answer"], csv_answers[qid]):
                fail(f"reasoning source raw_answer disagrees with final for {qid}")

    # ---------------------------------------------------------- API + ledger
    audit = jsonl(out / "api_calls.jsonl")
    successes = []
    api_signatures = []
    for line, row in enumerate(audit, 1):
        if strict_schema2:
            validate_schema2_api_record(row, line)
        if row.get("status") != "ok":
            continue
        successes.append(row)
        usage = row.get("usage")
        if not isinstance(usage, dict):
            fail(f"successful API call has no usage at line {line}")
        ap = token_int(usage.get("prompt_tokens"),
                       f"api_calls.jsonl:{line}.prompt_tokens")
        ac = token_int(usage.get("completion_tokens"),
                       f"api_calls.jsonl:{line}.completion_tokens")
        at = token_int(usage.get("total_tokens"),
                       f"api_calls.jsonl:{line}.total_tokens")
        if at != ap + ac:
            fail(f"API token sum mismatch at line {line}: {ap}+{ac}!={at}")
        request = row.get("request")
        if not isinstance(request, dict) or not isinstance(
                request.get("messages"), list) or not request["messages"]:
            fail(f"successful API call has no request messages at line {line}")
        response = row.get("response")
        if not isinstance(response, dict) or not any(
                str(response.get(key) or "").strip()
                for key in ("content", "reasoning_content")):
            fail(f"successful API call has no visible response at line {line}")
        recipients = row.get("allocation_qids") or []
        if not isinstance(recipients, list):
            fail(f"API allocation_qids is not a list at line {line}")
        api_signatures.append(_call_signature(
            row.get("qid"), row.get("model"), row.get("tag"), ap, ac,
            recipients))
        if not str(row.get("model") or "").lower().startswith("qwen"):
            fail(f"non-Qwen model found in API audit at line {line}")

    ledger = load_json(out / "token_ledger.json")
    if not isinstance(ledger, dict):
        fail("token_ledger.json must contain an object")
    per_qid = ledger.get("per_qid")
    calls = ledger.get("calls")
    if not isinstance(per_qid, dict) or not isinstance(calls, list):
        fail("token_ledger.json must contain per_qid object and calls list")
    clean_ledger = {str(qid): token_pair(usage, f"ledger.{qid}")
                    for qid, usage in per_qid.items()}

    rebuilt = defaultdict(lambda: [0, 0])
    ledger_signatures = []
    for line, call in enumerate(calls, 1):
        if not isinstance(call, dict):
            fail(f"ledger call {line} is not an object")
        cp = token_int(call.get("prompt_tokens"),
                       f"ledger.calls.{line}.prompt_tokens")
        cc = token_int(call.get("completion_tokens"),
                       f"ledger.calls.{line}.completion_tokens")
        qid = str(call.get("qid") or "").strip()
        if not qid:
            fail(f"ledger call {line} has no qid")
        recipients = call.get("allocation_qids") or []
        if not isinstance(recipients, list) or any(
                not str(x).strip() for x in recipients):
            fail(f"invalid allocation_qids in ledger call {line}")
        recipients = [str(x) for x in recipients]
        if len(recipients) != len(set(recipients)):
            fail(f"duplicate allocation_qids in ledger call {line}")
        allocated = call.get("allocated_usage") or {}
        if not isinstance(allocated, dict):
            fail(f"allocated_usage is not an object in ledger call {line}")
        if recipients:
            if set(allocated) != set(recipients):
                fail(f"allocated_usage recipients mismatch in ledger call {line}")
            alloc_p = alloc_c = 0
            for target in recipients:
                pair = token_pair(allocated[target],
                                  f"ledger.calls.{line}.allocated.{target}")
                alloc_p += pair[0]
                alloc_c += pair[1]
                rebuilt[target][0] += pair[0]
                rebuilt[target][1] += pair[1]
            if (alloc_p, alloc_c) != (cp, cc):
                fail(f"allocated_usage does not conserve call {line}: "
                     f"{(alloc_p, alloc_c)} vs {(cp, cc)}")
        else:
            if allocated:
                fail(f"direct ledger call {line} has allocated_usage")
            rebuilt[qid][0] += cp
            rebuilt[qid][1] += cc
        ledger_signatures.append(_call_signature(
            qid, call.get("model"), call.get("tag"), cp, cc, recipients))

    rebuilt_clean = {qid: tuple(value) for qid, value in rebuilt.items()}
    if rebuilt_clean != clean_ledger:
        keys = sorted(set(rebuilt_clean) | set(clean_ledger))
        bad = [(qid, rebuilt_clean.get(qid), clean_ledger.get(qid))
               for qid in keys if rebuilt_clean.get(qid) != clean_ledger.get(qid)]
        fail(f"ledger per_qid cannot be rebuilt from calls: {bad[:5]}")
    if Counter(api_signatures) != Counter(ledger_signatures):
        missing = Counter(api_signatures) - Counter(ledger_signatures)
        extra = Counter(ledger_signatures) - Counter(api_signatures)
        fail(f"API/ledger per-call mismatch: missing={list(missing.items())[:3]}, "
             f"extra={list(extra.items())[:3]}")

    lp = sum(v[0] for v in clean_ledger.values())
    lc = sum(v[1] for v in clean_ledger.values())
    if declared != (lp, lc, lp + lc):
        fail(f"ledger mismatch: {declared} vs {(lp, lc, lp + lc)}")
    ap = sum(sig[3] for sig in api_signatures)
    ac = sum(sig[4] for sig in api_signatures)
    if declared != (ap, ac, ap + ac):
        fail(f"API usage mismatch: {declared} vs {(ap, ac, ap + ac)}")

    # Mirror agent.b_schema.write_submission exactly for any direct shared
    # costs that are not already allocated to question qids by the ledger.
    shared_p = sum(v[0] for k, v in clean_ledger.items() if k not in qid_set)
    shared_c = sum(v[1] for k, v in clean_ledger.items() if k not in qid_set)
    add_p, rem_p = divmod(shared_p, len(question_ids))
    add_c, rem_c = divmod(shared_c, len(question_ids))
    expected_shared = {}
    for i, qid in enumerate(question_ids):
        direct_p, direct_c = clean_ledger.get(qid, (0, 0))
        share = (add_p + (1 if i < rem_p else 0),
                 add_c + (1 if i < rem_c else 0))
        expected_shared[qid] = share
        expected = (direct_p + share[0], direct_c + share[1],
                    direct_p + direct_c + share[0] + share[1])
        if csv_usage[qid] != expected:
            fail(f"per-qid ledger allocation mismatch for {qid}: "
                 f"CSV={csv_usage[qid]} expected={expected}")

    # ----------------------------------------------------- execution coverage
    run_rows = jsonl(out / "run_log.jsonl")
    docsel_rows = jsonl(out / "docsel_log.jsonl")
    by_run, by_docsel = defaultdict(list), defaultdict(list)
    for row in run_rows:
        by_run[str(row.get("qid") or "")].append(row)
    for row in docsel_rows:
        by_docsel[str(row.get("qid") or "")].append(row)
    for label, mapping in (("run_log.jsonl", by_run),
                           ("docsel_log.jsonl", by_docsel)):
        missing = sorted(qid_set - set(mapping))
        extra = sorted(set(mapping) - qid_set)
        if missing or extra:
            fail(f"{label} qid coverage mismatch: "
                 f"missing={missing}, extra={extra}")
    for qid in question_ids:
        finals = [row for row in by_run[qid]
                  if row.get("final") not in (None, "", [])]
        if not finals:
            fail(f"run_log has no final result for {qid}")
        if not answer_equivalent(finals[-1]["final"], csv_answers[qid]):
            fail(f"run_log final disagrees with submitted answer for {qid}")
        if not any(isinstance(row.get("picked"), list) and row["picked"]
                   for row in by_docsel[qid]):
            fail(f"docsel_log has no nonempty picked documents for {qid}")

    # ---------------------------------------------------------- evidence join
    evidence_payload, evidence = evidence_entries(out / "evidence.json")
    if strict_schema2:
        validate_schema2_evidence_provenance(evidence_payload, config)
    missing_ev = sorted(qid_set - set(evidence))
    extra_ev = sorted(set(evidence) - qid_set)
    if missing_ev or extra_ev:
        fail(f"evidence qid mismatch: missing={missing_ev}, extra={extra_ev}")

    # Index which global raw records belong to each question.  Evidence must
    # reference exactly these records by their 1-based source line/call index.
    expected_api_refs = defaultdict(set)
    for line, row in enumerate(audit, 1):
        recipients = row.get("allocation_qids") or []
        attached = recipients or ([row.get("qid")]
                                  if row.get("qid") in qid_set else [])
        for qid in attached:
            if qid in qid_set:
                expected_api_refs[qid].add(line)
    expected_ledger_refs = defaultdict(set)
    for line, call in enumerate(calls, 1):
        recipients = call.get("allocation_qids") or []
        attached = recipients or ([call.get("qid")]
                                  if call.get("qid") in qid_set else [])
        for qid in attached:
            if qid in qid_set:
                expected_ledger_refs[qid].add(line)

    for qid in question_ids:
        entry = evidence[qid]
        if strict_schema2:
            validate_complete_evidence(entry, qid)
        ev_slots = answer_slots(entry.get("answer_slots"),
                                f"evidence.{qid}.answer_slots")
        if ev_slots != csv_answers[qid]:
            fail(f"evidence answer_slots mismatch for {qid}")
        if not answer_equivalent(entry.get("answer"), csv_answers[qid]):
            fail(f"evidence answer mismatch for {qid}")
        if entry.get("reasoning") != reasonings[qid]:
            fail(f"evidence reasoning mismatch for {qid}")
        if entry.get("reasoning_source_stage") != \
                sources[qid].get("reasoning_stage"):
            fail(f"evidence reasoning stage mismatch for {qid}")

        accounting = entry.get("token_accounting")
        if not isinstance(accounting, dict):
            fail(f"evidence has no token_accounting for {qid}")
        ev_usage = (
            token_int(accounting.get("prompt_tokens"),
                      f"evidence.{qid}.prompt_tokens"),
            token_int(accounting.get("completion_tokens"),
                      f"evidence.{qid}.completion_tokens"),
            token_int(accounting.get("total_tokens"),
                      f"evidence.{qid}.total_tokens"),
        )
        if ev_usage != csv_usage[qid] or ev_usage[2] != ev_usage[0] + ev_usage[1]:
            fail(f"evidence/CSV token mismatch for {qid}: "
                 f"evidence={ev_usage} CSV={csv_usage[qid]}")
        direct = token_pair(accounting.get("direct_usage"),
                            f"evidence.{qid}.direct_usage")
        shared = token_pair(accounting.get("shared_usage"),
                            f"evidence.{qid}.shared_usage")
        if direct != clean_ledger.get(qid, (0, 0)):
            fail(f"evidence direct_usage mismatch for {qid}")
        if shared != expected_shared[qid]:
            fail(f"evidence shared_usage mismatch for {qid}")
        if (direct[0] + shared[0], direct[1] + shared[1]) != ev_usage[:2]:
            fail(f"evidence direct/shared usage does not sum for {qid}")

        api_refs = entry.get("api_attempts")
        if not isinstance(api_refs, list) or not api_refs:
            fail(f"evidence has no API references for {qid}")
        seen_api = set()
        has_success = False
        for ref in api_refs:
            line, embedded = _embedded_record(
                ref, "audit_line", f"evidence.{qid}.api")
            if line > len(audit):
                fail(f"evidence API reference out of range for {qid}: {line}")
            if line in seen_api:
                fail(f"duplicate evidence API reference for {qid}: {line}")
            seen_api.add(line)
            if embedded != audit[line - 1]:
                fail(f"evidence API record differs from audit line {line} "
                     f"for {qid}")
            has_success |= embedded.get("status") == "ok"
        if seen_api != expected_api_refs[qid]:
            fail(f"evidence API reference set mismatch for {qid}: "
                 f"evidence={sorted(seen_api)} "
                 f"expected={sorted(expected_api_refs[qid])}")
        if not has_success:
            fail(f"evidence has no successful API reference for {qid}")

        ledger_refs = accounting.get("ledger_calls")
        if not isinstance(ledger_refs, list) or not ledger_refs:
            fail(f"evidence has no ledger references for {qid}")
        seen_ledger = set()
        referenced_usage = [0, 0]
        for ref in ledger_refs:
            line, embedded = _embedded_record(
                ref, "ledger_call", f"evidence.{qid}.ledger")
            this_usage = embedded.pop("this_qid_usage", None)
            if line > len(calls):
                fail(f"evidence ledger reference out of range for {qid}: {line}")
            if line in seen_ledger:
                fail(f"duplicate evidence ledger reference for {qid}: {line}")
            seen_ledger.add(line)
            original = calls[line - 1]
            if embedded != original:
                fail(f"evidence ledger record differs from call {line} for {qid}")
            recipients = original.get("allocation_qids") or []
            if recipients:
                expected_pair = token_pair(
                    (original.get("allocated_usage") or {}).get(qid),
                    f"ledger.calls.{line}.allocated.{qid}")
                if token_pair(this_usage,
                              f"evidence.{qid}.ledger.{line}.this_qid_usage") \
                        != expected_pair:
                    fail(f"evidence this_qid_usage mismatch for {qid}, call {line}")
            else:
                if this_usage is not None:
                    fail(f"direct evidence ledger ref has this_qid_usage for {qid}")
                expected_pair = (
                    token_int(original.get("prompt_tokens"),
                              f"ledger.calls.{line}.prompt_tokens"),
                    token_int(original.get("completion_tokens"),
                              f"ledger.calls.{line}.completion_tokens"),
                )
            referenced_usage[0] += expected_pair[0]
            referenced_usage[1] += expected_pair[1]
        if seen_ledger != expected_ledger_refs[qid]:
            fail(f"evidence ledger reference set mismatch for {qid}: "
                 f"evidence={sorted(seen_ledger)} "
                 f"expected={sorted(expected_ledger_refs[qid])}")
        if tuple(referenced_usage) != direct:
            fail(f"evidence ledger references do not sum to direct_usage for {qid}")

        embedded_runs = entry.get("run_log_records")
        if not isinstance(embedded_runs, list) or not embedded_runs:
            fail(f"evidence has no run_log records for {qid}")
        if embedded_runs != by_run[qid]:
            fail(f"evidence run_log records differ from run_log.jsonl for {qid}")

    # Validate the top-level shared-cost evidence when using that layout.
    if isinstance(evidence_payload, dict) and "questions" in evidence_payload:
        shared_ev = evidence_payload.get("shared_costs")
        if not isinstance(shared_ev, dict):
            fail("evidence has no shared_costs object")
        if (token_int(shared_ev.get("prompt_tokens"),
                      "evidence.shared_costs.prompt_tokens"),
                token_int(shared_ev.get("completion_tokens"),
                          "evidence.shared_costs.completion_tokens")) != \
                (shared_p, shared_c):
            fail("evidence shared_costs token mismatch")

    # --------------------------------------------------------------- config
    model_values = [value for key, value in config.items()
                    if "model" in str(key).lower() and value]
    if not model_values or any(
            not str(value).lower().startswith("qwen")
            for value in model_values if isinstance(value, str)):
        fail("run config model is not Qwen")

    # A trace/run-log join alone is insufficient: deterministic postprocessing
    # could otherwise manufacture a new explanation after the paid call.  The
    # exact selected reasoning must occur in a successful response explicitly
    # attached to that question (directly or through batch allocation).
    for qid in question_ids:
        if not any(_api_attached_to(row, qid) and
                   _response_contains_reasoning(row, reasonings[qid])
                   for row in successes):
            fail("selected reasoning is not literal output of an attached "
                 f"successful API response for {qid}")

    print(f"reproduction check OK: 100 questions, {declared[2]:,} tokens, "
          f"{len(successes)} successful API calls")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: check_reproduction.py OUTPUT_DIR")
    main(sys.argv[1])
