# B 榜作战手册（7-22 00:00 – 7-24 17:00）

## 开闸第一小时（拿到 B 题文件后）
1. **先看不跑**：`python -m json.tool <b_file> | head -50` 检查字段（qid/domain/split/question/options/answer_format 是否与 A 一致；确认无 doc_ids）
2. **格式适配**：若字段有出入，改 run_b.load 部分；若 domain 缺失，用 qid 前缀推断（ins_/fc_/fin_/reg_/res_）
3. **docsel 冒烟**（不答题，只跑文档盲检 10 题）：确认候选卡与选择输出正常
4. **小样全链路**：每域 2 题共 10 题跑通，人工看 run_log 证据质量

## 正式跑
```bash
cd work && set -a && source .env && set +a
AFAC_VERIFY_MODEL=qwen3.6-max-preview .venv/bin/python -m agent.run_b \
  --tag b_final --questions <b_files...> --fresh-digests --workers 5
```
- 预算预期：~2.2-2.5M tokens（5M 上限，余量充足；若中途发现异常高，停下查）
- **断点续跑**：`--resume`（answers.json 已有的 qid 跳过，token 台账合并）
- 时间预算：全量约 30-40 分钟；窗口 2.7 天，至少留 24h 余量应对复跑

## 提交前
```bash
.venv/bin/python script/check_submission.py output/b_final/answer.csv   # 需先把QDIR指到B题文件!
.venv/bin/python script/build_evidence.py output/b_final
```
注意：check_submission.py 的 QDIR 指向 group_a，B 日要传 B 题文件路径（改脚本参数）。

## 已知风险与对策
- B 题干可能用 fc_text_00N 式别名指代文档 → doc_select 候选卡含 doc_id，Qwen 能对上；
  若出现"第一份文档"式指代（无名可检索），docsel 会退化——评估后可对该题多选文档（max 5）
- regulatory 513 篇大海捞针：粗召回 k=15 并集 + 相近规章都选 + csrc网页↔附件耦合已就位
- 若某域答题异常（全错样），先查 docsel_log 是否选错文档域
- API 限流/欠费：run_b 有重试；欠费联系用户充值后 --resume

## 提交包（B 榜前 15 需提交）
`python script/package_submission.py <tag>` → submission.zip
（answer.csv/evidence.json/processed_data/agent/script/logs/requirements.txt/README.md，≤1GB，
**确保 .env 和 API key 不入包**）
