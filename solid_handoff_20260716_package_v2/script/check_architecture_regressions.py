#!/usr/bin/env python3
"""Validate and score the online-derived architecture regression registry.

This script never calls a model. It checks that the registry is internally
consistent and optionally compares a CSV/JSONL result against locked answers
and online-proven wrong answers.
"""
import argparse
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "labels" / "architecture_regressions.json"
QDIR = ROOT / "public_dataset_upload" / "questions" / "group_a"
VALID = re.compile(r"^[ABCD]{1,4}$")


def load_questions():
    questions = {}
    for path in sorted(QDIR.glob("*_questions.json")):
        for question in json.loads(path.read_text(encoding="utf-8")):
            questions[question["qid"]] = question
    return questions


def legal_answer(answer, answer_format):
    if not isinstance(answer, str) or not VALID.fullmatch(answer):
        return False
    if answer != "".join(sorted(set(answer))):
        return False
    if answer_format in {"mcq", "tf"} and len(answer) != 1:
        return False
    return True


def load_registry(path=REGISTRY_PATH):
    registry = json.loads(Path(path).read_text(encoding="utf-8"))
    questions = load_questions()
    seen = set()
    errors = []
    for case in registry.get("cases", []):
        qid = case.get("qid")
        if qid in seen:
            errors.append(f"duplicate qid: {qid}")
            continue
        seen.add(qid)
        question = questions.get(qid)
        if not question:
            errors.append(f"unknown qid: {qid}")
            continue
        expected = case.get("expected")
        known_wrong = case.get("known_wrong", [])
        candidates = case.get("next_candidates", [])
        for label, values in (("known_wrong", known_wrong),
                              ("next_candidates", candidates)):
            if len(values) != len(set(values)):
                errors.append(f"{qid}: duplicate {label}")
            for answer in values:
                if not legal_answer(answer, question["answer_format"]):
                    errors.append(f"{qid}: illegal {label} answer {answer}")
        if expected is not None:
            if not legal_answer(expected, question["answer_format"]):
                errors.append(f"{qid}: illegal expected answer {expected}")
            if expected in known_wrong:
                errors.append(f"{qid}: expected answer is known_wrong")
        if case.get("status") == "locked_online" and expected is None:
            errors.append(f"{qid}: locked_online requires expected")
        if case.get("status") != "locked_online" and expected is not None:
            errors.append(f"{qid}: only locked_online may define expected")
        if not case.get("failure_modes") or not case.get("acceptance"):
            errors.append(f"{qid}: missing failure_modes or acceptance")
    if errors:
        raise ValueError("\n".join(errors))
    return registry, questions


def load_answers(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        answers = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("qid") and row.get("answer"):
                    answers[row["qid"]] = row["answer"]
        return answers
    first = path.read_text(encoding="utf-8-sig").splitlines()[0]
    delimiter = "\t" if "\t" in first else ","
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle, delimiter=delimiter)
        return {row["qid"]: row["answer"] for row in rows
                if row.get("qid") and row["qid"] != "summary"}


def evaluate(registry, answers, tiers=None):
    rows = []
    blocking = 0
    for case in registry["cases"]:
        if tiers and case["priority"] not in tiers:
            continue
        qid = case["qid"]
        answer = answers.get(qid)
        if answer is None:
            state = "MISSING"
            blocking += 1
        elif case["expected"] is not None:
            state = "PASS" if answer == case["expected"] else "FAIL"
            blocking += state == "FAIL"
        elif answer in case["known_wrong"]:
            state = "KNOWN-WRONG"
            blocking += 1
        elif answer in case["next_candidates"]:
            state = "CANDIDATE"
        else:
            state = "UNRESOLVED"
        rows.append({
            "qid": qid,
            "priority": case["priority"],
            "status": case["status"],
            "answer": answer,
            "expected": case["expected"],
            "state": state,
        })
    return rows, blocking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--tier", action="append", choices=["P0", "P1"])
    parser.add_argument("--strict-unresolved", action="store_true")
    args = parser.parse_args()

    registry, _ = load_registry(args.registry)
    counts = {}
    for case in registry["cases"]:
        counts[(case["priority"], case["status"])] = \
            counts.get((case["priority"], case["status"]), 0) + 1
    print(f"registry OK: {len(registry['cases'])} cases")
    for (priority, status), count in sorted(counts.items()):
        print(f"  {priority} {status}: {count}")
    if not args.results:
        return

    answers = load_answers(args.results)
    rows, blocking = evaluate(registry, answers, set(args.tier or []))
    for row in rows:
        expected = row["expected"] if row["expected"] is not None else "?"
        print(f"{row['state']:11} {row['priority']} {row['qid']:12} "
              f"got={row['answer']} expected={expected} status={row['status']}")
    if args.strict_unresolved:
        blocking += sum(row["state"] in {"CANDIDATE", "UNRESOLVED"}
                        for row in rows)
    summary = {}
    for row in rows:
        summary[row["state"]] = summary.get(row["state"], 0) + 1
    print("summary: " + ", ".join(f"{key}={value}"
                                  for key, value in sorted(summary.items())))
    if blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
