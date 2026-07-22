"""B榜题目适配：题型归一 + 从 submit.csv 占位符推断答案 schema。

submit.csv 的占位符是官方给出的答案格式契约：
  999999.99      → 数值，保留两位小数
  999999.99%     → 百分数，带%保留两位小数
  公司名称>公司名称 → 排序类，半角 > 连接
  A              → 选项字母
占位符个数 = 该题需要填写的 answer_i 个数。
"""
import csv, json, pathlib, re

ROOT = pathlib.Path(__file__).resolve().parents[2]

TYPE_MAP = {"多选题": "multi", "单选题": "mcq", "判断题": "tf",
            "计算题": "calc", "抽取题": "calc"}


def load_questions(qdir):
    """读取 B 榜题目目录（.json 与 .jsonl 混合，可能带 BOM）。"""
    qs = []
    for f in sorted(pathlib.Path(qdir).iterdir()):
        if f.suffix not in (".json", ".jsonl"):
            continue
        txt = f.read_text(encoding="utf-8-sig")
        data = ([json.loads(l) for l in txt.splitlines() if l.strip()]
                if f.suffix == ".jsonl" else json.loads(txt))
        qs.extend(data)
    for q in qs:
        q["answer_format"] = TYPE_MAP.get(q.get("type", ""), "multi")
        if not q.get("options"):
            q["answer_format"] = "calc"
    return qs


def _slot_kind(ph):
    if ph.endswith("%"):
        return "percent"
    if ">" in ph:
        return "ranking"
    if re.fullmatch(r"[0-9.]+", ph):
        return "number"
    if re.search(r"\d{4}年", ph):
        return "date"
    if ph in ("A", "B", "C", "D") or re.fullmatch(r"[A-D]+", ph):
        return "letter"
    return "text"


# 日期型计算题的占位符常给成 999999.99，与 readme 的日期规格冲突。
# 按 readme 规格优先原则：题干问法指向日期的，答案一律输出中文日期格式
# （readme：日期答案必须使用 YYYY年M月D日），槽位据此升级为 date。
_DATE_KIND_ASK = re.compile(r"哪一天|哪天|哪一日|何时|何日|什么时候")


def effective_kinds(q, kinds):
    if q.get("answer_format") == "calc" and _DATE_KIND_ASK.search(q.get("question", "")):
        return ["date" if k == "number" else k for k in kinds]
    return kinds


def is_date_question(q):
    t = q.get("question", "")
    return bool(_DATE_KIND_ASK.search(t))


def load_schema(submit_csv):
    """qid -> [slot_kind, ...]（长度即需填写的答案个数）"""
    schema = {}
    with open(submit_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["qid"] == "summary":
                continue
            slots = [r[f"answer_{i}"].strip() for i in range(1, 5)]
            schema[r["qid"]] = [_slot_kind(s) for s in slots if s]
    return schema


# ---------------- 答案格式化 ----------------

_NUM = re.compile(r"-?[\d,]+(?:\.\d+)?")


def fmt_slot(value, kind):
    """把模型给出的答案片段规范成官方格式。"""
    v = (value or "").strip().strip("。;；,，")
    if kind == "letter":
        letters = [c for c in v.upper() if c in "ABCD"]
        return "".join(sorted(set(letters)))
    if kind == "ranking":
        # 统一为半角 > 且前后无空格；去掉可能的公司后缀空白
        parts = re.split(r"\s*[>＞]\s*", v)
        parts = [p.strip().strip("。；;,，") for p in parts if p.strip()]
        return ">".join(parts)
    if kind == "date":
        m = re.search(r"(\d{4})\D{1,2}(\d{1,2})\D{1,2}(\d{1,2})", v)
        if not m:  # 救裸 YYYYMMDD / YYYYMMDD.00（模型被数字槽逼出的自创编码）
            m = re.fullmatch(
                r"((?:19|20)\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?:\.0{1,2})?", v)
        if m:
            return f"{int(m.group(1))}年{int(m.group(2))}月{int(m.group(3))}日"
        return v
    if kind in ("percent", "number"):
        # 日期形答案在数字槽：确定性转为 YYYYMMDD.00（防 '4月1日'→'4.01' 之类乱切）
        dm = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", v)
        if dm and kind == "number":
            return f"{int(dm.group(1)):04d}{int(dm.group(2)):02d}{int(dm.group(3)):02d}.00"
        m = _NUM.search(v.replace("%", ""))
        if not m:
            return v
        num = float(m.group(0).replace(",", ""))
        s = f"{num:.2f}"
        return s + "%" if kind == "percent" else s
    return v


def split_answer(raw, kinds):
    """把模型的一行答案拆成 len(kinds) 个槽位。"""
    raw = (raw or "").strip()
    if len(kinds) == 1:
        return [fmt_slot(raw, kinds[0])]
    # 优先按分号/中文分号切（题目普遍用“A；B”描述答案格式）
    parts = re.split(r"[;；]", raw)
    if len(parts) < len(kinds):
        # 退化：按逗号切，但排序类内部不含逗号才安全
        alt = re.split(r"[,，]", raw)
        if len(alt) >= len(kinds):
            parts = alt
    parts = [p for p in (x.strip() for x in parts) if p]
    if len(parts) < len(kinds):
        # 兜底：模型未按分号分隔时，按顺序抽取全部数字填充数值/百分数槽
        nums = _NUM.findall(raw)
        if len(nums) >= len(kinds) and all(k in ("number", "percent")
                                           for k in kinds):
            parts = nums
    out = []
    for i, kind in enumerate(kinds):
        out.append(fmt_slot(parts[i] if i < len(parts) else "", kind))
    return out


def write_submission(path, results, schema, order, ledger_per_qid,
                     totals):
    """按 B 榜格式写 csv：逗号分隔，answer_1..answer_4，summary 行在表头后。

    共享成本（记忆卡等非题目 qid）均摊到各题行，保证 逐题合计==summary（诚实且自洽）。
    带 BOM 对齐官方模板。
    """
    p, c, t = totals
    shared_p = sum(v[0] for k, v in ledger_per_qid.items() if k not in order)
    shared_c = sum(v[1] for k, v in ledger_per_qid.items() if k not in order)
    n = max(len(order), 1)
    add_p, rem_p = divmod(shared_p, n)
    add_c, rem_c = divmod(shared_c, n)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["qid", "answer_1", "answer_2", "answer_3", "answer_4",
                    "prompt_tokens", "completion_tokens", "total_tokens"])
        w.writerow(["summary", "", "", "", "", p, c, t])
        for i, qid in enumerate(order):
            slots = results.get(qid) or [""]
            slots = list(slots) + [""] * (4 - len(slots))
            qp, qc = ledger_per_qid.get(qid, [0, 0])
            qp += add_p + (1 if i < rem_p else 0)
            qc += add_c + (1 if i < rem_c else 0)
            w.writerow([qid] + slots[:4] + [qp, qc, qp + qc])
