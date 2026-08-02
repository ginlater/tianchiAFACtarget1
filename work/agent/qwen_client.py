"""Qwen API 客户端：DashScope compatible-mode + 全局 token 台账 + 重试。

比赛合规：推理阶段仅调用 Qwen 系列模型（阿里云百炼）。
Token 统计覆盖所有调用（含检索辅助、复核），写入 answer.csv summary。
"""
import atexit, json, os, threading, time
from pathlib import Path

from openai import OpenAI

from .paths import ENV_FILE

BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-plus"


def _ordered_weights(recipients, weights=None):
    """Return non-negative integer weights in recipient order.

    Batch callers deliberately use character counts as an auditable proxy for
    token ownership.  Keeping the arithmetic integral lets the allocator use
    exact largest remainders rather than floating-point tie breaking.  Missing
    or all-zero weights fall back to an equal split, which preserves the
    historical behaviour for callers that only pass ``allocation_qids``.
    """
    src = weights if isinstance(weights, dict) else {}
    out = {}
    for qid in recipients:
        try:
            value = int(src.get(qid, 0))
        except (TypeError, ValueError):
            value = 0
        out[qid] = max(0, value)
    if recipients and not sum(out.values()):
        out = {qid: 1 for qid in recipients}
    return out


def _largest_remainder(total, recipients, weights=None):
    """Allocate every integer token with stable Hamilton apportionment.

    Floors are assigned first; leftover tokens go to the greatest exact
    remainder.  Ties use the original recipient order, never lexical qid
    order, so changing an identifier cannot change ownership.
    """
    recipients = list(recipients or [])
    if not recipients:
        return {}
    total = max(0, int(total or 0))
    ordered = _ordered_weights(recipients, weights)
    denominator = sum(ordered.values())
    allocated = {}
    remainders = []
    used = 0
    for pos, qid in enumerate(recipients):
        numerator = total * ordered[qid]
        whole, remainder = divmod(numerator, denominator)
        allocated[qid] = whole
        used += whole
        remainders.append((remainder, pos, qid))
    # At most len(recipients)-1 tokens remain after flooring.  Sorting by the
    # exact integer remainder avoids platform-dependent floating behaviour.
    remainders.sort(key=lambda row: (-row[0], row[1]))
    for _remainder, _pos, qid in remainders[:total - used]:
        allocated[qid] += 1
    return allocated


def _reported_reasoning_tokens(usage):
    """Read hidden reasoning usage from compatible-mode response metadata."""
    details = usage.get("completion_tokens_details") or {}
    value = details.get("reasoning_tokens")
    if value is None:
        value = usage.get("reasoning_tokens")
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _merge_allocation_plan(base, update):
    """Merge resolver output without losing prompt-side audit metadata."""
    result = dict(base or {})
    for key, value in (update or {}).items():
        if key == "weight_basis" and isinstance(value, dict):
            merged = dict(result.get(key) or {})
            merged.update(value)
            result[key] = merged
        else:
            result[key] = value
    return result


def _load_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        envf = ENV_FILE
        if envf.exists():
            for line in envf.read_text().splitlines():
                if line.startswith("DASHSCOPE_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY not found (env or .env)")
    return key


class TokenLedger:
    """线程安全的 token 台账，按 qid 归集。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.per_qid = {}          # qid -> [prompt, completion]
        self.calls = []            # 审计日志

    def add(self, qid, model, usage, tag="", allocation_qids=None,
            allocation_plan=None):
        p = max(0, int(usage.get("prompt_tokens", 0) or 0))
        c = max(0, int(usage.get("completion_tokens", 0) or 0))
        recipients = list(dict.fromkeys(allocation_qids or []))
        plan = dict(allocation_plan or {})
        with self._lock:
            allocated = {}
            allocation = {
                "strategy": "direct",
                "apportionment": "direct",
                "prompt_weights": {},
                "visible_completion_weights": {},
                "reasoning_weights": {},
                "reported_reasoning_tokens": None,
                "visible_completion_tokens": c,
                "allocated_prompt_tokens": {},
                "allocated_visible_completion_tokens": {},
                "allocated_reasoning_tokens": {},
                "weight_basis": {},
            }
            if recipients:
                prompt_weights = _ordered_weights(
                    recipients, plan.get("prompt_weights"))
                completion_weights = _ordered_weights(
                    recipients, plan.get("completion_weights"))
                prompt_alloc = _largest_remainder(
                    p, recipients, prompt_weights)
                reported_reasoning = _reported_reasoning_tokens(usage)
                reasoning_n = min(c, reported_reasoning or 0)
                visible_n = c - reasoning_n
                # Hidden thinking is caused by the full prompt, not by the
                # length of the subsequently rendered answer block.
                reasoning_alloc = _largest_remainder(
                    reasoning_n, recipients, prompt_weights)
                visible_alloc = _largest_remainder(
                    visible_n, recipients, completion_weights)
                for target in recipients:
                    ap = prompt_alloc[target]
                    ac = reasoning_alloc[target] + visible_alloc[target]
                    slot = self.per_qid.setdefault(target, [0, 0])
                    slot[0] += ap
                    slot[1] += ac
                    allocated[target] = [ap, ac]
                allocation = {
                    "strategy": plan.get("strategy") or "equal_split_v1",
                    "apportionment": (
                        "stable_largest_remainder_input_order_v1"),
                    "prompt_weights": prompt_weights,
                    "visible_completion_weights": completion_weights,
                    "reasoning_weights": prompt_weights,
                    "reported_reasoning_tokens": reported_reasoning,
                    "visible_completion_tokens": visible_n,
                    "allocated_prompt_tokens": prompt_alloc,
                    "allocated_visible_completion_tokens": visible_alloc,
                    "allocated_reasoning_tokens": reasoning_alloc,
                    "weight_basis": dict(plan.get("weight_basis") or {}),
                }
            else:
                slot = self.per_qid.setdefault(qid, [0, 0])
                slot[0] += p
                slot[1] += c
            if recipients:
                conserved_prompt = sum(
                    allocation["allocated_prompt_tokens"].values())
                conserved_completion = sum(
                    allocation["allocated_visible_completion_tokens"].values()
                ) + sum(allocation["allocated_reasoning_tokens"].values())
            else:
                conserved_prompt, conserved_completion = p, c
            allocation["conservation"] = {
                "reported_prompt_tokens": p,
                "allocated_prompt_tokens": conserved_prompt,
                "reported_completion_tokens": c,
                "allocated_completion_tokens": conserved_completion,
                "reported_total_tokens": p + c,
                "allocated_total_tokens": (
                    conserved_prompt + conserved_completion),
                "ok": (conserved_prompt == p and
                       conserved_completion == c),
            }
            call = {"qid": qid, "model": model, "tag": tag,
                    "prompt_tokens": p, "completion_tokens": c,
                    "allocation_qids": recipients,
                    "allocated_usage": allocated,
                    "allocation_strategy": allocation["strategy"],
                    "allocation_apportionment": allocation[
                        "apportionment"],
                    "allocation_weights": {
                        "prompt": allocation["prompt_weights"],
                        "visible_completion": allocation[
                            "visible_completion_weights"],
                        "reasoning": allocation["reasoning_weights"],
                    },
                    "allocation": allocation,
                    "ts": time.time()}
            self.calls.append(call)
            return allocation

    def totals(self):
        p = sum(v[0] for v in self.per_qid.values())
        c = sum(v[1] for v in self.per_qid.values())
        return p, c, p + c

    def dump(self, path):
        with self._lock:
            json.dump({"per_qid": self.per_qid, "calls": self.calls},
                      open(path, "w"), ensure_ascii=False, indent=1)


LEDGER = TokenLedger()
_client = None
_client_lock = threading.Lock()
_audit_lock = threading.Lock()
_audit_file = None


def configure_audit(path, *, append=False):
    """Write every API attempt to a full, key-free JSONL audit log."""
    global _audit_file
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _audit_lock:
        if _audit_file is not None:
            _audit_file.close()
        _audit_file = open(p, "a" if append else "w", encoding="utf-8")


def close_audit():
    global _audit_file
    with _audit_lock:
        if _audit_file is not None:
            _audit_file.flush()
            _audit_file.close()
            _audit_file = None


atexit.register(close_audit)


def _write_audit(record):
    with _audit_lock:
        if _audit_file is None:
            return
        _audit_file.write(json.dumps(record, ensure_ascii=False,
                                     default=str) + "\n")
        _audit_file.flush()


def client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(api_key=_load_key(), base_url=BASE_URL, timeout=300)
    return _client


def chat(messages, *, qid="_", model=DEFAULT_MODEL, thinking=False,
         thinking_budget=None, max_tokens=4096, temperature=None, tag="",
         max_retries=5, allocation_qids=None, allocation_plan=None,
         allocation_resolver=None):
    """返回 (content:str, reasoning:str, usage:dict)。所有用量记入 LEDGER。"""
    extra = {"enable_thinking": bool(thinking)}
    if thinking and thinking_budget:
        extra["thinking_budget"] = int(thinking_budget)
    if temperature is None:
        temperature = 0.6 if thinking else 0.1
    request = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "extra_body": extra,
    }
    last_err = None
    for attempt in range(max_retries):
        started = time.time()
        try:
            resp = client().chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=temperature, extra_body=extra)
            msg = resp.choices[0].message
            usage = resp.usage.model_dump() if resp.usage else {}
            reasoning = getattr(msg, "reasoning_content", None) or ""
            content = (msg.content or "").strip()
            resolved_plan = dict(allocation_plan or {})
            if allocation_resolver is not None:
                try:
                    resolved_plan = _merge_allocation_plan(
                        resolved_plan, allocation_resolver(content))
                except Exception as allocation_error:  # noqa: BLE001
                    # The API call has already consumed tokens.  A local block
                    # parser failure must never trigger another paid request or
                    # make this usage disappear; equal completion weights are
                    # the conservative fallback and the audit states why.
                    resolved_plan = _merge_allocation_plan(resolved_plan, {
                        "completion_weights": {},
                        "weight_basis": {
                            "allocation_resolver_error": {
                                "type": type(allocation_error).__name__,
                                "message": str(allocation_error),
                            },
                        },
                    })
            allocation = LEDGER.add(
                qid, model, usage, tag, allocation_qids,
                allocation_plan=resolved_plan)
            _write_audit({
                "status": "ok", "qid": qid, "tag": tag, "model": model,
                "attempt": attempt + 1, "started_at": started,
                "finished_at": time.time(), "request": request,
                "allocation_qids": list(allocation_qids or []),
                "allocation": allocation,
                "response": {"content": content,
                             "reasoning_content": reasoning},
                "usage": usage,
            })
            return content, reasoning, usage
        except Exception as e:  # noqa: BLE001 — 网络/限流统一重试
            last_err = e
            _write_audit({
                "status": "error", "qid": qid, "tag": tag,
                "model": model, "attempt": attempt + 1,
                "started_at": started, "finished_at": time.time(),
                "request": request,
                "allocation_qids": list(allocation_qids or []),
                "allocation_plan": dict(allocation_plan or {}),
                "error": {"type": type(e).__name__, "message": str(e)},
            })
            wait = min(2 ** attempt * 2, 30)
            time.sleep(wait)
    raise RuntimeError(f"chat failed after {max_retries} retries: {last_err}")
