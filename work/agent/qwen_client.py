"""Qwen API 客户端：DashScope compatible-mode + 全局 token 台账 + 重试。

比赛合规：推理阶段仅调用 Qwen 系列模型（阿里云百炼）。
Token 统计覆盖所有调用（含检索辅助、复核），写入 answer.csv summary。
"""
import json, os, pathlib, threading, time

from openai import OpenAI

ROOT = pathlib.Path(__file__).resolve().parents[2]
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen3.6-plus"


def _load_key() -> str:
    key = os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        envf = ROOT / "work" / ".env"
        if envf.exists():
            for line in envf.read_text().splitlines():
                if line.startswith("DASHSCOPE_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY not found (env or work/.env)")
    return key


class TokenLedger:
    """线程安全的 token 台账，按 qid 归集。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.per_qid = {}          # qid -> [prompt, completion]
        self.calls = []            # 审计日志

    def add(self, qid, model, usage, tag=""):
        p = usage.get("prompt_tokens", 0)
        c = usage.get("completion_tokens", 0)
        with self._lock:
            slot = self.per_qid.setdefault(qid, [0, 0])
            slot[0] += p
            slot[1] += c
            self.calls.append({"qid": qid, "model": model, "tag": tag,
                               "prompt_tokens": p, "completion_tokens": c,
                               "ts": time.time()})

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


def client() -> OpenAI:
    global _client
    with _client_lock:
        if _client is None:
            _client = OpenAI(api_key=_load_key(), base_url=BASE_URL, timeout=300)
    return _client


def chat(messages, *, qid="_", model=DEFAULT_MODEL, thinking=False,
         thinking_budget=None, max_tokens=4096, temperature=None, tag="",
         max_retries=5):
    """返回 (content:str, reasoning:str, usage:dict)。所有用量记入 LEDGER。"""
    extra = {"enable_thinking": bool(thinking)}
    if thinking and thinking_budget:
        extra["thinking_budget"] = int(thinking_budget)
    if temperature is None:
        temperature = 0.6 if thinking else 0.1
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client().chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens,
                temperature=temperature, extra_body=extra)
            msg = resp.choices[0].message
            usage = resp.usage.model_dump() if resp.usage else {}
            LEDGER.add(qid, model, usage, tag)
            reasoning = getattr(msg, "reasoning_content", None) or ""
            return (msg.content or "").strip(), reasoning, usage
        except Exception as e:  # noqa: BLE001 — 网络/限流统一重试
            last_err = e
            wait = min(2 ** attempt * 2, 30)
            time.sleep(wait)
    raise RuntimeError(f"chat failed after {max_retries} retries: {last_err}")
