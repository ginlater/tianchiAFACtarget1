# AFAC2026 赛题四：金融长文本 Agent 诚实复现

本目录是一条单次、端到端的复现流水线。它从同一份题目与文档输入出发，在一次运行中
完成文档选择、词法检索、动态记忆压缩、Qwen 作答、reasoning 生成、Token 计量与证据
联结。正式流程不会读取旧 `answer.csv`、答案键、榜单反馈或历史运行结果，也不会逐题
挑选或改写答案。

## 1. 方法概览

系统用“代码处理确定性结构，Qwen 处理语义判断”的方式置换 Token：

1. **离线文档层**：PDF/HTML 解析、页码保留、标题修复、表格单元格恢复；不使用
   embedding。
2. **确定性记忆层**：
   - `insurance_capsules.json`：保险条款按页码、条款、主题、数字保存原句胶囊；
   - `fin_facts2.json`：财务表格绑定“合并/母公司 × 本期/上期”列身份；
   - `domain_facts.json` 与 `align_matrix.json`：条款数字行和跨文档对齐事实。
   - `financial_fact_registry.py`：把财报指标绑定到公司、年份、合并口径、单位和页码；
   - `narrative_fact_registry.py`：从合同、法规和研报原句中提取带页码的计算原子，冲突
     时整束拒绝。
3. **动态工作记忆**：BM25、题面同义归一、逐选项保护块和字符预算，只把当题所需
   原文送入上下文。
4. **文档路由**：公司、产品、年份或法规标题唯一时由确定性代码选择；有歧义时回退
   Qwen。选择只依赖题面、选项与语料元数据，不读取 qid 或答案。
5. **计算路由**：严格 AST、`Decimal`、单位与唯一性校验先计算；成功后仍由一次 Qwen
   根据题目和证据核验并生成可见 reasoning。事实冲突、单位不闭合、日期日历不完整或
   槽位不匹配时自动回退完整 Qwen 计算流程。
6. **语义作答**：同底仓选择题共享证据批答；提示词按题面语义编译必要规则。输出答案
   和 reasoning 来自同一次可见 Qwen 响应。

推理阶段只调用 Qwen 系列模型；所有 API 用量直接采用 DashScope 返回的 `usage`。

## 2. 环境

- Python 3.9.6（本机验证版本；建议使用 3.9–3.11）
- 安装依赖：`python -m pip install -r requirements.txt`
- 设置 `DASHSCOPE_API_KEY` 环境变量。正式包不含 `.env` 或任何密钥。

输入目录应包含 B 榜题目目录和官方 `submit.csv`：

```text
INPUT/
├── question_b/          # .json / .jsonl
└── submit.csv
```

`processed_data/` 已随复现材料提供。若需从原始 PDF/HTML 重新构建，使用：

```bash
python script/rebuild_processed.py \
  --input /path/to/raw_input \
  --output /path/to/new_processed_data
```

输出目录必须尚不存在；脚本先在临时 staging 目录完成并校验，再原子式复制结果，绝不
覆盖已有 `processed_data`。该步骤只做解析、OCR/版面恢复和确定性词法结构化，不产生
非 Qwen 语义摘要。使用 `--check-only` 可只检查 573 份输入的布局、依赖和哈希而不写文件。

## 3. 一键生成

在本目录执行：

```bash
./generate_answer.sh \
  --input /path/to/INPUT \
  --output /path/to/reproduction_run \
  --model qwen3.6-plus \
  --workers 6
```

配置固定在 `config/honest_repro.env`。输出目录必须为空，避免混入旧运行缓存。脚本依次
运行正式 Agent、构建审计证据并执行零 API 严格校验。

## 4. 单次运行产物

```text
reproduction_run/
├── answer.csv                 # 100 题答案、逐题 usage、reasoning
├── answers.json
├── reasonings.json
├── reasoning_sources.json     # 最终 reasoning 对应的真实调用阶段
├── api_calls.jsonl            # 完整请求、响应、重试、模型与 API usage
├── token_ledger.json          # 调用级和逐题分摊账本
├── run_log.jsonl              # 候选答案、后处理、证据块 ID
├── docsel_log.jsonl           # 文档选择方法、结果与置信诊断
├── run_config.json            # 参数、环境、版本、输入 SHA256
└── evidence.json              # 上述信息按 qid 联结的完整证据链
```

`script/check_reproduction.py` 会逐项验证：100 个 qid、答案与 reasoning 非空、逐题
CSV usage 与 ledger 分摊一致、summary 与所有题行一致、ledger 与 API 成功响应一致、
模型全部为 Qwen、证据覆盖完整且配置不含密钥。每题最终 reasoning 还必须逐字出现在
该题直接或批量分摊的某次成功 Qwen API 响应中；代码事后追加的解释不能通过。

校验通过后，可从这一份运行显式打包（脚本没有历史运行的默认值）：

```bash
python script/package_submission.py /path/to/reproduction_run \
  --output /path/to/submission.zip
```

归档包含正式入口、固定配置、Agent、预处理重建脚本、`processed_data`、完整 API 调用审计
和逐题证据链；打包前再次执行全量一致性校验并扫描密钥样式。ZIP 根目录直接是
`answer.csv`、`evidence.json`、`agent/`、`script/`、`processed_data/`、`logs/`、
`requirements.txt` 和 `README.md`，不存在额外的 `submission/` 套层。解压后可先做零 API
重定位验收：

```bash
./generate_answer.sh --check-runtime
```

该命令只检查源码、配置、规则文件和 `processed_data` 的完整性与哈希，不读取 API key、
不发起网络请求。归档同时保留 `比赛规则.md`、`b榜新增规则.txt`、`b榜补充.md` 和
`upload_b/readme.md`；对 reasoning 列和原始 usage 审计以 B 榜新增/补充规则为准。

## 5. 复现误差审计

`eval/reference_audit.py` 是开发期只读工具，仅报告答案、总 Token、逐题 Token ±15%
与 reasoning 完整性；它不修改答案或 Token，也没有被 `generate_answer.sh` 导入。正式
运行目录和正式 Agent 不读取参考提交文件。

## 6. 合规边界

允许并已使用：词法 BM25、确定性表格/条款抽取、Python 精确计算、同底仓批处理、
Qwen 回退和同次响应生成 reasoning。

正式流水线明确不包含：embedding 检索、非 Qwen 语义模型、qid 特判、答案键、榜单
反推、历史结果拼装、人工替换答案、Token 填充或泊车调用。
