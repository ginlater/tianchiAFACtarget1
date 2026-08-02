#!/usr/bin/env bash
set -euo pipefail

RUNTIME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR=""
OUTPUT_DIR=""
SUBMIT_TEMPLATE=""
MODEL="qwen3.6-plus"
WORKERS="6"
PYTHON_BIN=""
CHECK_RUNTIME="0"

usage() {
  echo "Usage: $0 --input INPUT_DIR --output OUTPUT_DIR [--submit-template FILE] [--model qwen...] [--workers N] [--python PYTHON]" >&2
  echo "       $0 --check-runtime [--python PYTHON]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --input) INPUT_DIR="$2"; shift 2 ;;
    --output) OUTPUT_DIR="$2"; shift 2 ;;
    --submit-template) SUBMIT_TEMPLATE="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --workers) WORKERS="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    --check-runtime) CHECK_RUNTIME="1"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$RUNTIME_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$RUNTIME_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

# Offline relocation smoke test used by reviewers after extracting the ZIP.
# It hashes the complete formal runtime and exits before reading a key or
# touching the output directory.
if [[ "$CHECK_RUNTIME" == "1" ]]; then
  export PYTHONPATH="$RUNTIME_DIR"
  "$PYTHON_BIN" - <<'PY'
from agent.repro import build_runtime_manifest

manifest = build_runtime_manifest()
if not manifest.get("files"):
    raise SystemExit("formal runtime manifest is empty")
print(f"runtime check OK: {len(manifest['files'])} frozen files")
PY
  exit 0
fi

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
  usage
  exit 2
fi
if [[ "$MODEL" != qwen* ]]; then
  echo "Only Qwen models are allowed: $MODEL" >&2
  exit 2
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" && ! -f "$RUNTIME_DIR/.env" ]]; then
  echo "DASHSCOPE_API_KEY is required" >&2
  exit 2
fi
if [[ ! -d "$RUNTIME_DIR/processed_data" ]]; then
  echo "Missing processed_data under $RUNTIME_DIR" >&2
  exit 2
fi

if [[ -d "$INPUT_DIR/question_b" ]]; then
  QUESTION_DIR="$INPUT_DIR/question_b"
else
  QUESTION_DIR="$INPUT_DIR"
fi
if [[ ! -d "$QUESTION_DIR" ]]; then
  echo "Question directory not found: $QUESTION_DIR" >&2
  exit 2
fi

if [[ -z "$SUBMIT_TEMPLATE" ]]; then
  if [[ -f "$INPUT_DIR/submit.csv" ]]; then
    SUBMIT_TEMPLATE="$INPUT_DIR/submit.csv"
  elif [[ -f "$(dirname "$QUESTION_DIR")/submit.csv" ]]; then
    SUBMIT_TEMPLATE="$(dirname "$QUESTION_DIR")/submit.csv"
  else
    echo "submit.csv not found; pass --submit-template" >&2
    exit 2
  fi
fi

set -a
# shellcheck source=/dev/null
. "$RUNTIME_DIR/config/honest_repro.env"
set +a
unset AFAC_VOTE_QIDS AFAC_R2_DOMAINS AFAC_B1_VOTE_DOMS AFAC_VERIFY_MODEL
export PYTHONPATH="$RUNTIME_DIR"

"$PYTHON_BIN" -m agent.run_b2 \
  --output-dir "$OUTPUT_DIR" \
  --qdir "$QUESTION_DIR" \
  --submit-template "$SUBMIT_TEMPLATE" \
  --model "$MODEL" \
  --verify-model "$MODEL" \
  --workers "$WORKERS" \
  --batch \
  --fresh-digests

"$PYTHON_BIN" "$RUNTIME_DIR/script/build_evidence.py" "$OUTPUT_DIR"
"$PYTHON_BIN" "$RUNTIME_DIR/script/check_reproduction.py" "$OUTPUT_DIR"
