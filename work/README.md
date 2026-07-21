# 金融长文档问答 Agent（AFAC 赛题）

仅使用 Qwen 系列模型 API（阿里云百炼，评测基准 qwen3.6-plus）完成推理；
检索为纯词法 BM25（字符 bigram + 数字/英文整词），**不使用任何 embedding 模型**。

## 系统架构

1. **文档预处理**（无语义模型参与，合规）
   - `script/parse_docs.py`：PyMuPDF 逐页抽取 PDF（页码标记 [Pn]）、bs4 解析 HTML（保留
     ArticleTitle/PubDate/栏目 元数据）、法规 TXT 按条款切分 → `processed_data/`
   - `script/fix_titles.py`：PDF 字号版面分析恢复文档标题（规则允许的版面分析范畴）
2. **词法检索** `agent/retrieval.py`
   - 页/条款级分块，BM25（中文字符 bigram + 数字/百分比整词加权）
3. **Agent 记忆** `agent/answerer.py`
   - 按文档懒构建【事实卡】（Qwen 生成、跨题复用，token 全额计入统计）：
     小文档全文构卡，大文档按领域标准查询选段构卡
4. **文档盲检（B榜）** `agent/doc_select.py`
   - regulatory(513篇)：BM25 并集粗召回；其余领域全量候选
   - Qwen 从候选卡片（标题/日期/栏目/开头）中精选 doc_id
5. **答题流程**：事实卡+逐选项定向证据 → r1 逐项判断（先复述题干选择标准）→
   多选题 r2 独立复核 → 分歧时 r3 扩大证据仲裁 → 答案字母规范化（多选去重升序）
6. **Token 台账** `agent/qwen_client.py`：所有调用（含事实卡、盲检、复核）线程安全归集，
   写入 answer.csv summary 行

## 复现

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
export DASHSCOPE_API_KEY=sk-xxx
# 1. 预处理
.venv/bin/python script/parse_docs.py && .venv/bin/python script/fix_titles.py
# 2. A榜（题目含 doc_ids）
.venv/bin/python -m agent.run_a --tag submit_a --fresh-digests
# 3. B榜（题目无 doc_ids，自动盲检）
.venv/bin/python -m agent.run_b --tag submit_b --questions <b_questions.json> --fresh-digests
# 4. 证据文件
.venv/bin/python script/build_evidence.py output/<tag>
```

输出：`output/<tag>/answer.csv`（tab 分隔，含 summary token 统计）、`evidence.json`、
`run_log.jsonl`（每题完整推理记录）、`token_ledger.json`（逐调用审计）。
