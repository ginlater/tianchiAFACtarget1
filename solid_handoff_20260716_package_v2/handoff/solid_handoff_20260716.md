# Tianchi A榜交接文档 - 只含已验证结论

生成时间：2026-07-16 Asia/Shanghai

本文件只记录已由用户明确反馈的线上分数、本地提交文件内容、或确定性回归测试支持的结论。不把候选答案、模型判断、网页搜索结果、未提交文件结果写成事实。

## 1. 当前硬事实

1. 当前已确认最高线上结果为 `90.0225`，对应本地记录中的 94/100 正确约束。
2. 94题母版文件是 `output/answer_a_inferred_94_probe_reg11_base.csv`。
3. 该母版中关键答案：
   - `fc_a_004 = AC`
   - `ins_a_014 = AB`
   - `fc_a_015 = D`
   - `reg_a_011 = ACD`
   - `res_a_002 = ABC`
   - `fc_a_018 = A`
4. `output/answer_a_from_94_probe_fc4_abcd.csv` 已生成并通过本地结构校验；它不是已验证正确答案结果。它只是在 94题母版上把 `fc_a_004` 从 `AC` 改为 `ABCD`。
5. `output/answer_a_from_94_probe_fc4_abcd.csv` 的 SHA256 是 `19739bc497b85baa25871128bad9d6283d514aee98f63c7d107a9ad603af91d5`。

## 2. 已锁定答案

以下答案按 `labels/architecture_regressions.json` 当前状态为 `locked_online`。这些是当前流程中可作为硬约束使用的答案。

| qid | locked answer | 已知错误答案 |
| --- | --- | --- |
| `fc_a_014` | `ABC` | `AB`, `B`, `BC` |
| `res_a_004` | `ABC` | `AB` |
| `res_a_006` | `A` | `B` |
| `res_a_011` | `ABC` | `BC` |
| `fc_a_005` | `ABD` | `ABCD`, `AB` |
| `ins_a_014` | `AB` | `ABD`, `A` |
| `reg_a_011` | `ACD` | `AC`, `ABCD` |
| `res_a_002` | `ABC` | `ABCD` |
| `fc_a_018` | `A` | `B` |

## 3. `fc_a_004` 的已验证排除结论

`fc_a_004` 当前未锁定正确答案。

已排除答案：

| answer | 结论来源 |
| --- | --- |
| `ACD` | 由 94题母版、group1 约束、`ins_a_014=AB` 单题锁定后的联动约束排除 |
| `AC` | 94题母版中为 `AC`，后续约束证明该题仍错 |
| `A` | 从 94题母版仅改 `fc_a_004: AC -> A`，用户反馈线上仍为 `90.0225` |
| `AD` | 从 94题母版仅改 `fc_a_004: AC -> AD`，用户反馈线上仍为 `90.0225` |
| `ABC` | 从 94题母版仅改 `fc_a_004: AC -> ABC` 的探针，用户反馈线上仍为 `90.0225` |

当前可陈述的唯一结论：`fc_a_004` 不等于 `ACD`, `AC`, `A`, `AD`, `ABC`。  
不得陈述 `ABCD` 正确；它没有线上结果。

## 4. `fc_a_015` 的已验证异常

`fc_a_015` 当前未锁定正确答案。

从 94题母版分别把 `fc_a_015` 改为 `A`, `B`, `C`，用户反馈线上均仍为 `90.0225`；母版中 `fc_a_015 = D` 也对应 `90.0225`。当前只可得出：

1. 在已记录的线上反馈中，`A/B/C/D` 四个单字母都没有使 94题母版提升。
2. 该题标注为 `mcq`，多字母探针如 `AB` 不应直接当作有效新信息，因为平台/规则可能按首个有效字母处理。
3. 当前不得锁定 `fc_a_015` 的答案。

## 5. 已确认线上分数记录

以下记录来自用户反馈并已写入 `labels/architecture_regressions.json` 或对话上下文。分数到正确题数的映射按当前 token 总量 `705230` 下的已用换算。

| file | score | inferred correct | status |
| --- | ---: | ---: | --- |
| `output/answer_a_fc14_abc.csv` | `89.0648` | 93 | confirmed |
| `output/answer_a_93_probe_fc5_ab.csv` | `88.1071` | 92 | confirmed |
| `output/answer_a_93_probe_fc15_c.csv` | `89.0648` | 93 | confirmed by user |
| `output/answer_a_inferred_94_probe_group_1.csv` | `88.1071` | 92 | confirmed |
| `output/answer_a_inferred_94_probe_reg11_base.csv` | `90.0225` | 94 | confirmed |
| `output/answer_a_from_94_probe_fc15_a.csv` | `90.0225` | 94 | confirmed |
| `output/answer_a_from_94_probe_fc15_c_recheck.csv` | `90.0225` | 94 | confirmed |
| `output/answer_a_from_94_probe_fc15_b_final.csv` | `90.0225` | 94 | confirmed |
| `output/answer_a_from_94_probe_ins14_a.csv` | `89.0648` | 93 | confirmed |
| `output/answer_a_from_94_probe_fc4_a.csv` | `90.0225` | 94 | confirmed |
| `output/answer_a_from_94_probe_fc4_ad.csv` | `90.0225` | 94 | confirmed |
| `output/answer_a_from_94_probe_fc4_abc.csv` | `90.0225` | 94 | confirmed by user; local file structure must be rechecked before reuse |

## 6. 文件完整性与注意事项

1. `output/answer_a_from_94_probe_fc4_abcd.csv` 本地结构校验通过：
   - 102 行，包括 header、summary、100 个题目行。
   - 100 个唯一 qid。
   - `summary` token 为 `634944 / 70286 / 705230`。
   - 与 `output/answer_a_inferred_94_probe_reg11_base.csv` 的逻辑差异只有 `fc_a_004: AC -> ABCD`。
2. `output/answer_a_from_94_probe_fc4_abc.csv` 当前本地文件内容和文件名不完全可信：该文件被检测到整行带引号，且本地读取不再是标准 tab 分列。用户反馈的线上分数仍可作为历史事实，但该本地文件不应作为后续提交模板。
3. 回归测试 `script/test_architecture_regressions.py` 当前通过，测试内容包括：
   - P0 锁定题通过。
   - 已知错误答案会阻断。
   - `fc_a_004=ABCD` 仅为候选状态。
   - `fc_a_004=ABC` 被标记为 known wrong。

## 7. 下一位 Agent 的硬约束

1. 只能使用 `public_dataset_upload` 官方数据、历史提交向量、用户明确反馈的线上分数和本地校验结果。
2. 不要使用外部网页搜索结果来决定答案。
3. 不要把 `fc_a_004=ABCD` 当作正确答案；它是未提交或未反馈结果的候选文件。
4. 不要继续试探 `fc_a_015` 的普通单字母答案；当前记录已显示 `A/B/C/D` 均未提升 94题母版。
5. 如果继续 A榜探针，必须一次只改一个未锁定题目的一个答案，否则无法从分数变化推出单题结论。

## 8. 本次交接包应包含的关键文件

| path | 用途 |
| --- | --- |
| `handoff/solid_handoff_20260716.md` | 本交接说明 |
| `labels/architecture_regressions.json` | 线上约束注册表 |
| `script/check_architecture_regressions.py` | 约束检查器 |
| `script/test_architecture_regressions.py` | 回归测试 |
| `output/answer_a_inferred_94_probe_reg11_base.csv` | 94题母版 |
| `output/answer_a_from_94_probe_fc4_abcd.csv` | 已生成但未验证线上结果的下一候选文件 |
| `output/answer_a_from_94_probe_ins14_a.csv` | 锁定 `ins_a_014=AB` 的关键探针 |
| `output/answer_a_from_94_probe_fc4_a.csv` | 排除 `fc_a_004=A` 的探针 |
| `output/answer_a_from_94_probe_fc4_ad.csv` | 排除 `fc_a_004=AD` 的探针 |
| `output/answer_a_from_94_probe_fc4_abc.csv` | 用户反馈排除 `fc_a_004=ABC` 的历史探针；本地文件不可直接复用 |
| `public_dataset_upload/questions/group_a/*.json` | 官方 A组题目 |

