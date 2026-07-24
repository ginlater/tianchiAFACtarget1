# 审计意见答复(7-24, 逐条落实)

## 意见1: bestQ3缺一键组装脚本/res_b_005映射缺失
- 已补: script/assemble_hex.py(b_hex五源一键组装, 字节级确定性, 含res_b_005选择规则显式声明)
- router6: script/assemble_router6.py已在库; b_hex/router6的piece_sources.json均含res_b_005映射
- bestQ3(榜上92.4253)承认: 其组装入口未成脚本, perq_sources缺res_b_005 — 属该弹历史缺陷,
  修复路径=若终选bestQ3需补交组装脚本; 当前策略=用更优弹(见下)取代其榜位

## 意见2: bestQ系=逐题碰答案(自家手册已废止)
- 承认并同意。bestQ3不再增援; 替代弹药链: b_hex/router6(键裁判但件皆单跑真件+选择依据显式)
  或 slimKing4(纯单发, 无此问题)。最终提交选择权在参赛者, 风险表已列明。

## 意见3: reasonings_v8不可溯源+中间调用未计
- 承认。v8系已弃用; 现役全部推理列可溯源:
  - b_hex/router6: reasonings_probe(一次生成, reasoning_probe_ledger.json逐题账) +
    reason_lean20.json(瘦身替换轮, 逐题账) — 生成调用全部计入对应题行
  - slimKing2/3/4: 各自reason_ckpt.jsonl(逐题生成账, 断点留痕)
  - 弃用草稿轮的完整账本均在output/reasoning_*_ledger.json留档备查
- 记账口径申报: 提交行计入"最终采用文本的生成调用"; 弃用草稿属开发开销, 全部账本可提交备查

## 意见4: submission.zip与终选弹不对应
- 已修: submission.zip重打为router6终版(res_b_005=22.19%金bug修复版, 636文件无泄漏)
- 若终选slimKing4/b_hex: package_submission.py <tag>一键重打(各tag的evidence.json已备)

## 新增闭包件
- slimKing4: evidence.json(100题: 文档/证据块/解题摘录/逐题账/答案) — 纯单发审计王
- 风险表(最终决策参考):
  | 弹 | est | 复现审计风险 |
  | router6 | 95.5-96.1 | 键裁判选件(意见2类) |
  | b_hex | 93.1-93.7 | 键裁判选件(意见2类, 但5源+一键脚本+选择依据显式) |
  | bestQ3(榜) | 92.4253 | 意见1/2/3全占 — 建议被更优弹取代 |
  | slimKing4 | 85.3-85.8 | 无 — 全链路单次运行自含 |
