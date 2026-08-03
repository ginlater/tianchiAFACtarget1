# Router6 单次端到端复现：提交件与验收材料

本目录冻结了 `codex/router6-single-run` 分支提交
`97f3eabdc69e11bbcd6e8b003237d6c41ade65cd` 的一次完整 100 题运行。
答案、reasoning、API 调用、Token 台账和证据链均来自同一次执行。

## 验收结论

- 题目数：100
- 成功 Qwen API 调用：73
- Prompt Token：405,498
- Completion Token：89,605
- Total Token：495,103
- 对 router6-v3 目标答案：100/100 一致
- 总 Token 相对误差：0.96%
- 逐题 Token：100/100 位于 ±15% 容差内
- reasoning：100/100 非空且可回溯到成功 API 响应
- 运行源码：`git_dirty=false`
- 强复现检查：通过
- 解包运行时检查：618 个冻结文件通过

## 文件说明

- `submission.zip`：可直接下载检查或提交的完整复现包。
- `answer.csv`：本次运行最终答案、逐题 Token 和 reasoning。
- `api_calls.jsonl`：每次 Qwen 请求、响应及服务商原始 usage。
- `token_ledger.json`：调用级和逐题 Token 守恒分摊。
- `run_config.json`：模型、参数、Git 提交、输入及运行时文件哈希。
- `run_log.jsonl`：逐题候选、最终阶段和证据块记录。
- `reasoning_sources.json`：最终 reasoning 对应的真实调用阶段。
- `evidence.json`：按 qid 联结的完整审计证据链。
- `reference_audit.json`：开发期只读容差验收报告；不参与正式答题。
- `SHA256SUMS`：本目录所有冻结文件的 SHA-256。

## 本地验收

从仓库根目录执行：

```bash
python work/script/check_reproduction.py deliverables/router6_single_run
shasum -a 256 -c deliverables/router6_single_run/SHA256SUMS
```

直接检查提交包的离线运行时：

```bash
tmpdir="$(mktemp -d)"
unzip -q deliverables/router6_single_run/submission.zip -d "$tmpdir"
bash "$tmpdir/generate_answer.sh" --check-runtime
```

`reference_audit.json` 保留验收时的原始绝对路径，以证明报告未经事后改写；正式运行和
`submission.zip` 均不读取该文件。
