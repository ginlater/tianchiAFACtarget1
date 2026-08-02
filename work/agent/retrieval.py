"""词法检索：分块 + 字符 bigram BM25。

合规说明：纯词法统计（BM25），不使用任何 embedding 模型；
规则明确允许"按领域构建关键词索引和结构化字段索引"。
"""
import json, math, re
from collections import Counter

from .paths import PROCESSED_DIR

PROC = PROCESSED_DIR

_META = None


def docs_meta():
    global _META
    if _META is None:
        with open(PROC / "docs_meta.json", encoding="utf-8") as f:
            _META = json.load(f)
    return _META


def doc_path(doc_id):
    m = docs_meta()[doc_id]
    return PROC / m["domain"] / f"{doc_id}.txt"


# ---------------- 分块 ----------------

_PAGE_RE = re.compile(r"\[P(\d+)\]\n")
_CLAUSE_RE = re.compile(r"(?=\n第[一二三四五六七八九十百零\d]+条)")

MAX_CHUNK = 1400
MIN_CHUNK = 120


def _split_long(text):
    if len(text) <= MAX_CHUNK:
        return [text]
    parts, buf = [], ""
    for para in text.split("\n"):
        if len(buf) + len(para) > MAX_CHUNK and len(buf) >= MIN_CHUNK:
            parts.append(buf)
            buf = para
        else:
            buf = buf + "\n" + para if buf else para
    if buf:
        parts.append(buf)
    return parts


def chunk_doc(doc_id):
    """返回 [{id, doc_id, page, text}]"""
    text = doc_path(doc_id).read_text(encoding="utf-8")
    chunks = []
    if "[P" in text and _PAGE_RE.search(text):
        pieces = _PAGE_RE.split(text)
        # pieces: [pre, p1, text1, p2, text2, ...]
        it = iter(range(1, len(pieces), 2))
        for i in it:
            page, ptxt = int(pieces[i]), pieces[i + 1]
            for j, sub in enumerate(_split_long(ptxt)):
                if sub.strip():
                    chunks.append({"page": page, "text": sub.strip()})
    else:
        # 法规 txt / html：优先按条款切
        if len(_CLAUSE_RE.findall(text)) >= 5:
            pieces = _CLAUSE_RE.split(text)
        else:
            pieces = _split_long(text)
        for p in pieces:
            for sub in _split_long(p):
                if sub.strip():
                    chunks.append({"page": None, "text": sub.strip()})
    # 合并过小块
    merged = []
    for c in chunks:
        if merged and len(merged[-1]["text"]) < MIN_CHUNK and \
           merged[-1]["page"] == c["page"]:
            merged[-1]["text"] += "\n" + c["text"]
        else:
            merged.append(c)
    for i, c in enumerate(merged):
        c["id"] = f"{doc_id}#c{i}"
        c["doc_id"] = doc_id
    return merged


# ---------------- 分词与 BM25 ----------------

_TOKEN_RE = re.compile(r"[0-9]+(?:\.[0-9]+)?%?|[A-Za-z]+|[一-鿿]")

# Numeric comparisons are unusually vulnerable to evidence-budget truncation:
# the decisive source is often one short table commentary sentence embedded in
# a page-sized chart chunk.  Keep this entirely lexical and question-driven.
# Province-level names cover the common case without trying to infer entities
# from historical answers or document ids; the suffix expression extends the
# same mechanism to explicitly named companies, banks, insurers and groups.
_ADMIN_NAMES = (
    "北京", "天津", "上海", "重庆", "河北", "山西", "辽宁", "吉林",
    "黑龙江", "江苏", "浙江", "安徽", "福建", "江西", "山东", "河南",
    "湖北", "湖南", "广东", "海南", "四川", "贵州", "云南", "陕西",
    "甘肃", "青海", "台湾", "内蒙古", "广西", "西藏", "宁夏", "新疆",
    "香港", "澳门",
)
_ADMIN_RE = re.compile(
    "(?:" + "|".join(sorted(_ADMIN_NAMES, key=len, reverse=True)) +
    r")(?:省|市|自治区|壮族自治区|回族自治区|维吾尔自治区|特别行政区)?"
)
_ORG_RE = re.compile(
    r"[A-Za-z0-9一-鿿·]{2,18}(?:股份有限公司|有限责任公司|有限公司|"
    r"公司|银行|保险|集团)"
)
_PAIR_COMPARE_RE = re.compile(
    r"高于|低于|超过|不及|多于|少于|快于|慢于|领先|落后|相比|"
    r"较.{0,8}(?:高|低|多|少|快|慢|增|降)"
)
_NUMERIC_METRIC_RE = re.compile(
    r"\d|[%％]|同比|环比|增速|增幅|降幅|增长率|下降率|占比|比例|"
    r"金额|收入|利润|价格|均价|销量|数量|出口额|率"
)


def pairwise_numeric_entities(query):
    """Return explicit entities for a lexical numeric-comparison claim.

    An empty tuple means the query is not safely recognizable as a comparison
    between two named objects.  This deliberately fails closed: ordinary BM25
    remains unchanged for vague comparisons such as "前者高于后者".
    """

    query = str(query or "")
    if not (_PAIR_COMPARE_RE.search(query) and _NUMERIC_METRIC_RE.search(query)):
        return ()
    entities = []
    for match in _ADMIN_RE.finditer(query):
        name = match.group(0)
        if name not in entities:
            entities.append(name)
    comparison = _PAIR_COMPARE_RE.search(query)
    # Search the two sides separately.  Without this boundary, a permissive
    # Chinese organization regex can swallow "利润增速高于乙公司" as one name.
    sides = ((query[:comparison.start()], query[comparison.end():])
             if comparison else (query,))
    for side in sides:
        matches = list(_ORG_RE.finditer(side))
        if not matches:
            continue
        name = matches[-1].group(0)
        if name not in entities:
            entities.append(name)
    return tuple(entities) if len(entities) >= 2 else ()


def compact_pairwise_numeric_hit(chunk, query, width=420):
    """Project a page chunk to the source window binding both compared values.

    The returned mapping preserves ``id``, ``doc_id`` and ``page`` so existing
    provenance and de-duplication remain exact.  Projection occurs only if the
    same chunk contains every explicit query entity and a numeric marker.
    """

    entities = pairwise_numeric_entities(query)
    text = chunk.get("text", "")
    if not entities or not all(entity in text for entity in entities):
        return chunk
    positions = [text.find(entity) for entity in entities]
    lo_entity = min(positions)
    hi_entity = max(pos + len(entity)
                    for pos, entity in zip(positions, entities))
    # The compared objects must be locally co-present; otherwise a long report
    # merely mentioning both names on distant pages is not a bound comparison.
    if hi_entity - lo_entity > width:
        return chunk
    # Prefer an exact two-row/ two-clause projection.  It keeps both operands
    # while dropping an unrelated middle region (for example a third province
    # in the same source sentence), which is precisely what a comparison needs
    # and is small enough to survive a shared batch evidence cap.
    fragments = []
    for position, entity in sorted(zip(positions, entities)):
        end_candidates = [p for p in (text.find("；", position),
                                      text.find(";", position),
                                      text.find("。", position)) if p >= 0]
        end = min(end_candidates) + 1 if end_candidates else min(
            len(text), position + 100)
        fragment = text[position:end].replace("\n", "").strip()
        if (fragment and _NUMERIC_METRIC_RE.search(fragment) and
                re.search(r"\d+(?:\.\d+)?\s*[%％]?", fragment)):
            fragments.append(fragment)
    if len(fragments) == len(entities):
        metric_names = [name for name in (
            "同比", "环比", "增速", "增幅", "降幅", "增长率", "下降率",
            "占比", "比例", "金额", "收入", "利润", "价格", "均价",
            "销量", "数量", "出口额",
        ) if name in query]
        compact_rows = []
        for entity, fragment in zip(
                [e for _p, e in sorted(zip(positions, entities))], fragments):
            values = []
            for metric in metric_names:
                match = re.search(
                    re.escape(metric) +
                    r"[^\d+\-]{0,8}[+\-]?\d+(?:\.\d+)?\s*[%％]?",
                    fragment,
                )
                if match:
                    values.append(match.group(0).replace(" ", ""))
            if values:
                compact_rows.append(entity + "…" + values[0])
        if len(compact_rows) == len(entities):
            projected = dict(chunk)
            projected["text"] = "…" + "；".join(compact_rows) + "…"
            return projected
        # Preserve a nearby date once; it binds values from charts containing
        # multiple months without retaining the whole chart payload.
        prefix = text[max(0, lo_entity - 80):lo_entity]
        dates = re.findall(r"20\d{2}\s*年\s*\d{1,2}\s*月", prefix)
        bound = ((dates[-1] + "，") if dates else "") + "".join(fragments)
        projected = dict(chunk)
        projected["text"] = "…" + bound + "…"
        return projected
    lo = max(0, lo_entity - width // 3)
    hi = min(len(text), max(hi_entity + width // 3, lo + width))
    window = text[lo:hi]
    if not re.search(r"\d+(?:\.\d+)?\s*[%％]?", window):
        return chunk
    projected = dict(chunk)
    projected["text"] = (("…" if lo else "") + window +
                         ("…" if hi < len(text) else ""))
    return projected


def tokenize(text):
    """中文字符 bigram + 数字/英文整词。数字带百分号整体保留。
    年份归一：查询/文档中 20XX 追加 XX 形式（金融文本常写"26年"）。"""
    text = text.replace("％", "%").replace("，", ",")
    raw = _TOKEN_RE.findall(text)
    for t in list(raw):
        if len(t) == 4 and t.isdigit() and 2015 <= int(t) <= 2035:
            raw.append(t[2:])
    toks = []
    i = 0
    while i < len(raw):
        t = raw[i]
        if "一" <= t <= "鿿":
            j = i
            while j < len(raw) and "一" <= raw[j] <= "鿿":
                j += 1
            run = raw[i:j]
            toks.extend(run)  # unigram
            toks.extend(run[k] + run[k + 1] for k in range(len(run) - 1))
            i = j
        else:
            toks.append(t.lower())
            i += 1
    return toks


class BM25:
    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.tf = []
        self.df = Counter()
        self.dl = []
        for c in chunks:
            cnt = Counter(tokenize(c["text"]))
            self.tf.append(cnt)
            self.dl.append(sum(cnt.values()))
            for tok in cnt:
                self.df[tok] += 1
        self.N = len(chunks)
        self.avgdl = (sum(self.dl) / self.N) if self.N else 1.0

    def search(self, query, k=5):
        q = tokenize(query)
        # 去重但保留权重：查询词频
        qcnt = Counter(q)
        scores = [0.0] * self.N
        for tok, qw in qcnt.items():
            df = self.df.get(tok)
            if not df:
                continue
            idf = math.log(1 + (self.N - df + 0.5) / (df + 0.5))
            # bigram 权重更高（更有区分度）
            wt = 1.6 if len(tok) == 2 and "一" <= tok[0] <= "鿿" else 1.0
            if tok[-1:] == "%" or tok[:1].isdigit():
                wt = 2.2  # 数字精确匹配权重最高
            for i in range(self.N):
                f = self.tf[i].get(tok)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += qw * wt * idf * f * (self.k1 + 1) / denom
        pair_entities = pairwise_numeric_entities(query)

        def _rank(i):
            # An exact two-entity co-occurrence is the lexical equivalent of a
            # structured comparison row.  Prefer it over a higher-scoring page
            # that repeats only generic metric words such as "环比/增速".
            pair_bound = bool(pair_entities) and all(
                entity in self.chunks[i]["text"] for entity in pair_entities)
            return (not pair_bound, -scores[i])

        order = sorted(range(self.N), key=_rank)[:k]
        return [
            (compact_pairwise_numeric_hit(self.chunks[i], query), scores[i])
            for i in order if scores[i] > 0
        ]


_INDEX_CACHE = {}


def doc_index(doc_id) -> BM25:
    if doc_id not in _INDEX_CACHE:
        _INDEX_CACHE[doc_id] = BM25(chunk_doc(doc_id))
    return _INDEX_CACHE[doc_id]


def search_docs(doc_ids, query, k_per_doc=3):
    """在指定文档集合内检索，返回 [(chunk, score)]，按分数排序。

    跨文档分数按各文档该查询的top1归一（各索引IDF独立不可比：目标公司名在
    本档IDF≈0贡献为零，填充文档页眉样板反被逐块抬分——fin_b_011/017类伤）。"""
    hits = []
    for d in doc_ids:
        dh = doc_index(d).search(query, k=k_per_doc)
        if not dh:
            continue
        top = dh[0][1] or 1.0
        hits.extend((c, s / top) for c, s in dh)
    hits.sort(key=lambda x: -x[1])
    return hits
