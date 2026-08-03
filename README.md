# AFAC2026 赛题四：Router6 单次端到端复现

本分支冻结了一条从官方题目与文档输入出发、一次运行生成全部答案、reasoning、证据链和
真实 Token 台账的 Qwen Agent。

快速入口：

- [完整运行与提交 ZIP](deliverables/router6_single_run/README.md)
- [架构设计说明](docs/router6_single_run/ARCHITECTURE.md)
- [运行和打包说明](work/README.md)
- [正式一键入口](work/generate_answer.sh)
- [官方复现要求 DOCX](docs/router6_single_run/4-AFAC2026挑战组复现材料——金融长文本Agent的动态记忆压缩与高效问答挑战.docx)
- [复现方法报告 PDF](docs/router6_single_run/AFAC2026_复现方法报告_草稿_20260802.pdf)

冻结验收结果：100 题、73 次成功 Qwen 调用、495,103 Token；强复现检查、逐题 Token
容差检查、ZIP 哈希闭包和解包运行时检查均已通过。

```bash
python work/script/check_reproduction.py deliverables/router6_single_run
./work/generate_answer.sh --check-runtime
```
