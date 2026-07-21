# AFAC2026挑战组-赛题四：金融长文本Agent 的动态记忆压缩与高效问答挑战 — 项目备忘（每次会话必读）

（天池平台。赛题关键词：**动态记忆压缩**、长文本、Token 效率。评审叙事要突出记忆框架。）

## 总目标
**拿一等奖。** 最终排名看 B 榜成绩，前 15 名需提交完整可复现代码（submission.zip：answer.csv / evidence.json / processed_data / agent / script / logs / requirements.txt / README）。

## 铁律
1. **不走队友的探针刷分路线**（`solid_handoff_20260716_package_v2` 是靠一次改一题提交反推答案，不可复现、有取消资格风险、对 B 榜盲测无效）。用户明确要求不要顺着他做。
2. 推理问答阶段**只能用 Qwen 系列模型 API**（阿里云百炼 DashScope 或魔搭），评测基准模型为 **Qwen3.6-plus**。
3. **禁止任何 embedding 模型**做检索/推理 → 检索必须是词法（关键词/BM25）或 Qwen 驱动的 agentic 检索。
4. 预处理（PDF解析/OCR/表格恢复）可用非 Qwen 工具（如 MinerU），但非 Qwen 产生的语义摘要/召回/排序/知识库**不得进入正式答题流程**。
5. 多选题无部分分；答案字母去重升序无分隔符（如 `ABC`）；单选/判断取首个有效字母。

## 评分公式
- `TokenScore = max(0, min(1, (5,000,000 - TotalTokens) / 5,000,000))`
- `FinalScore = 100 * Accuracy * (0.7 + 0.3 * TokenScore)`
- 队友 94/100 + 705,230 tokens → 90.0225（已验证换算）。100 题满分 + ~700k tokens ≈ 95.8。
- 同分依次比：准确率 → TotalTokens 低 → 提交早。

## 关键时间（2026年）
- A 榜截止：**7-21 20:00**（今天 7-18）
- B 榜窗口：**7-22 00:00 → 7-24 17:00**（只有 ~2.7 天！B 榜方案必须在 A 榜期间就绪）

## 战略核心（勿忘）
- **A 榜只是资格赛，队友 90.02 已锁定 B 榜资格；最终评奖只看 B 榜成绩。**
- A 榜剩余时间的价值 = 用验证标签把 Agent 打磨到位 + 演练 B 模式（对 A 题隐藏 doc_ids 测文档检索命中率）。
- B 榜预算目标：总 token ≤1M（系数 0.94），准确率最大化优先；+1 题正确 ≈ 值 165k tokens 的复核开销。
- 已发现的两大准确率杠杆：①保险等小文档全文记忆卡（防条款漏检，如"身故给付比例160%"分档表）；②判分标准校准（选项是概括转述，核心事实一致即判对，不因省略次要前提判错——reg_a_001 教训）。

## 数据布局（doc_id → 文件映射已验证）
`public_dataset_upload/`
- `questions/group_a/{domain}_questions.json` — 5 领域 × 20 题
- `raw/insurance/{1..16}.pdf` — doc_id 即数字
- `raw/financial_contracts/text01..14.pdf` — doc_id 即 textNN
- `raw/financial_reports/annual_{company}_{year}_report.PDF` — doc_id 同名
- `raw/research/pack2_text01..20.pdf` — doc_id 同名
- `raw/regulatory/`：`txt/strict_v3_*.txt`（6个，doc_id 为去扩展名文件名）、`html/csrc_0001..0377.html`（377个，doc_id 如 csrc_0262）、`attachments/csrc_NNNN_attN.pdf`（130个，doc_id 如 csrc_0009_att1）
- B 榜难点：regulatory 有 377 个 html + 130 个附件要盲测检索。

## 提交格式（重要：tab 分隔）
`answer.csv`：`qid\tanswer\tprompt_tokens\tcompletion_tokens\ttotal_tokens`，第二行为 `summary`（含三项 token 总计），之后 100 行每题一行（题行只填 qid+answer 也可，队友文件如此且被接受）。

## 队友数据的正确用法：本地验证集（不是答案来源）
- `labels/validation_labels.json`（本项目 `eval/` 下构建）：94题母版 `output/answer_a_inferred_94_probe_reg11_base.csv` 全部答案 + 锁定题 + 已知错误答案。
- 母版 94/100 正确 → 可离线评估我们的 Agent；与母版不一致的题重点复核（可能是母版错的 ~6 题）。
- 已锁定正确（locked_online）：fc_a_014=ABC, res_a_004=ABC, res_a_006=A, res_a_011=ABC, fc_a_005=ABD, ins_a_014=AB, reg_a_011=ACD, res_a_002=ABC, fc_a_018=A
- 已排除：fc_a_004 ∉ {ACD, AC, A, AD, ABC}（母版里 fc_a_004=AC 是错的 → 母版该题不可信，ABCD 是唯一未测的多选组合但未验证）；fc_a_015 单字母 A/B/C/D 全部不提升母版（异常，按 mcq 首字母规则 D 在母版即错，A/B/C 也不对？→ 注意：mcq 只取首字母，所以互斥探针已覆盖 4 个选项仍无提升，说明母版的 94 正确题里不含 fc_a_015，且 4 个字母都试过仍 94 → 该题可能计分异常或题目有问题，暂放弃纠结，用模型自身判断)
- 推论：母版 6 道错题中已确认 fc_a_004、fc_a_015 两道；其余 4 道未知分布在剩余 98 题中（98 题里 94 对 4 错）。

## API 已验证事实（2026-07-18）
- key 在 `work/.env`，DashScope compatible-mode：`https://dashscope.aliyuncs.com/compatible-mode/v1`
- `qwen3.6-plus` 可用（=评测基准模型）；`qwen3.6-flash` 可用于开发调试省钱；另有 `qwen3.6-max-preview`
- **qwen3.6-plus 默认开思维链**，reasoning tokens 计入 completion_tokens（1+1题烧了208个）
- 请求体加 `"enable_thinking": false` 可关闭思考（验证过：同题 completion=1）→ token 优化核心手段：抽取/检索关思考，难题开思考

## 工程结构（本项目根目录）
- `work/` — 我们的正式工程（agent/ script/ processed_data/ eval/ logs/ output/）
- `work/.venv` — Python 环境
- API key：需用户提供 DASHSCOPE_API_KEY（尚未拿到时先做离线部分）

## 状态速记（随时更新，末次 2026-07-18 晚）
- [x] 规则读透；队友包分析完（探针路线弃用，锁定答案作验证集）
- [x] 573 文档解析 → work/processed_data（标题用PDF字号版面分析修复）
- [x] 验证集 work/eval/validation_labels.json（9 locked + 89 master + 2 master_wrong）
- [x] Agent v1: 记忆卡(digest) + BM25 + r1判断/r2复核/r3仲裁 + token台账
- [x] 保险域迭代: 13/20 → 17/20（选择标准陷阱修复是关键）
- [x] 研报域: 13 → 17-18/20（dedup修复+判分口径）；法规 20/20；合同13/年报16(带bug跑的,待复测)
- [x] 闭卷基线 39/98 完成；对抗题库 75 题交付（eval/synthetic/，格式已归一化）
- [x] 判分口径少样本示例已入 JUDGE_STD（除字前缀/模糊尾巴/自动手动反转/省略前提）
- [x] dev_full1 全量100题：估84-88/98，tokens 2.08M（法规20/20 研报19/20 年报18/20 保险16/20 **合同13/20瓶颈**）
- [x] 合同域类级修复：证据基数10000、记忆卡1200字含逐品种兑付日/逾期利息原句/重要日期、术语同义扩展(违约利息→逾期利息)
- [x] 判分再补：字面原句优先（不许用换算口径推翻原句）；年报记忆卡分红逐口径列出
- [>] syn_test1 对抗压测75题跑中；之后 dev_fc_fin2 复测（注意先再次失效fc/fin记忆卡缓存——syn_test1结束会写回旧卡）
- [x] **7-18 首次线上提交 submit_a1：73.1002 = 85/100 正确 × 0.860系数(2.33M tokens)**
  - 反推结论：6道母版分歧题几乎全是我们错(master对)；master未知4错里3-4道我们犯同样错(硬陷阱题)
  - 明日抓手：①token 2.33M→1.2M(+6分)；②6道分歧题类级归因；③多选三样本多数决稳摇摆题
- [x] submit_a2（6项类级修复+全题型复核）：估86-90 @2.65M tokens → 预期分~71 低于a1的73.1，**未提交**
- [x] 发现并修**金融写法鸿沟**（检索类缺陷）：题目"2026年/一季度/下降" vs 文档"26年/1-3月/同比-"
  → tokenizer年份归一(20XX追加XX) + 季度/方向同义表；res_a_020目标块从>6名→第4
- [x] **submit_a3 已提交：72.7819 = 87/100 @2.72M**（+2题但+390k token，净-0.3分）
  - 两次线上校准一致证实：与母版分歧≈全是我们错 → 估分公式=锁定命中+母版一致−3
  - **构卡预算挤占bug已修**：原文片段按查询顺序填充后截断，第6位的"违约金"查询排名第1的块进不了卡
    （fc_a_008的150%公式、fin_a_002的净利润50%分红句都因此丢失）→ 改为轮询交错
  - fc_a_018 属真难题（两文档均无显式公告日期），指望投票不指望检索
- **核心矛盾：运行间方差±3盖过单点修复信号** → 明日必做：逐选项三样本多数决(r1/r2/r3全独立,
  按选项聚合投票) + token优化(2.6M→1.2M)，两者一起上
## 7-19 进展
- [x] 投票机制+瘦身：dev_vote1 锁定7/9历史最佳；tf复核0/20翻转→砍掉；答题成本-17%
- [x] submit_a4（轮询构卡生效版）：合同fc_a_008/013/018修复但整体est 84，未提交（线上最佳仍73.1）
- [x] **B模式彩排**：盲测est 78/100 @2.25M；docsel全命中84/100
  - docsel类修复：①csrc网页→附件自动耦合(reg_a_019)；②强制≥2文档(res 3失误)；③regulatory最多5份+相近规章都选
  - fc域7个docsel失误是A榜"第一份文档"指代造成的假信号，B不会有
- [x] stable模式结论：**法规域20/20且省35% token→法规默认关思维链**；研报掉2-3题→其余域保留思考
- [x] **异构二审**（AFAC_VERIFY_MODEL=qwen3.6-max-preview 做r2/r3）：合同域19/20收敛；同模型重采样错误相关，异构打破
- [x] 保险顽固题亲读条款定案→两条通用判分规则：
  - 规则6【有无类题】"哪些条款明确规定X"：补检后仍无→缺失即否定（ins_a_016教训: 医疗险没有施救费用概念）
  - 规则7【例外不触发】主规则带例外分句，题干未触发例外按主规则判（ins_a_019教训: 增益宝犹豫期全额退+部分领取例外）
  - 保险记忆卡补类目(9)犹豫期(10)施救费用上限
- [x] **submit_a5 线上 76.0556 = 88/100 @2.26M（新最佳，纯系统产出三连升85→87→88）**
- [x] submit_a6 已交用户提交中：口径同构示例修复顽固组，est 88-90 @2.32M 预期76.5-77.5
- [x] **B彩排2：盲测 est 83-85/100，docsel 85%**（fc的11/20是A榜指代假信号；B真实预期90%+）
- [x] B日作战手册 work/B_DAY_RUNBOOK.md + 打包脚本 script/package_submission.py（含密钥泄漏检查）
- 跨6次运行只剩6道摇摆题；估分公式偶尔低估2题（我们开始在母版错题上赢）
- [x] submit_a6 线上 75.7691 = 88/100 @2.32M（准确率平台=88确认；token是剩余主杠杆）
- [x] **盲测对抗题 71/75(94.7%)，盲检96%全命中** → B就绪度强；fc彩排低分确认是A榜指代假信号
- [x] lean-r2（AFAC_LEAN_R2=1 复核只带记忆卡+引用页+核心保护块）保险域18/20安全；保护收紧到题干+4选项
- [x] **submit_a7 线上 78.8957 = 90/100 @2.06M（A榜最终成绩，五连升85→87→88→88→90）**
## 7-20~7-21 B榜
- [x] B就绪：打包链路验证(8MB无泄漏)；docsel类修复(k=25/meta摘要卡/样板过滤/材料类别补齐)
- [x] **B题解析：格式大变**——26计算题(无选项) + 逗号CSV + answer_1..4多槽 + 占位符即答案schema
- [x] B适配三模块：b_schema.py(占位符反推schema+格式化) / calc.py(两阶段取数计算+异构双样本) / run_b2.py
- [x] 自检抓出关键bug：AFAC_VERIFY_MODEL env漏设(选择题异构二审失效)→止损重跑
- [x] **b_final 完成：100题 @2,805,025 tokens 系数0.832**；计算题双样本一致23/25；抽样验算全过；格式零问题
- [x] reg_b_003 规格冲突决策：readme日期格式(规格)优先于submit.csv数字占位符(样例)→提交2026年3月30日
- [x] 代码全量入组内私有仓库 github.com/ginlater/tianchiAFACtarget1 (main)；分析文档 work/B_RESULT_ANALYSIS.md
- **B窗口至7-24 17:00；提交后反推正确数更新分析**
- [x] B榜文档盲检 doc_select.py: 非regulatory全量候选卡 + regulatory BM25并集，Qwen按doc_id精选
- [ ] 其余四域调优 → A榜全量 → A榜提交（截止7-21 20:00）
- [ ] run_b.py 驱动器 + B榜演练

## 用户核心洞察（2026-07-18，必须贯彻）
- **题目按反先验设计**：不召回证据、纯靠 Qwen 常识必错。准确率天花板在检索召回。
- 应对：证据优先于常识指令 + 逐选项证据充分性检查 + 闭卷基线诊断 + 子agent构造对抗题库（work/eval/synthetic/）
- API 费用不设限（用户会充值），准确率优先。

## 已确认的实验结论
- r2复核翻转率仅~1/13（同模型同证据重问≈确认），要换视角复核才有信息增量
- 保险域 token ~16k/题 偏高，优化方向：r2只对不确定题、r2用精简证据
- docsel 教训: 让模型输出与卡片方括号完全一致的 doc_id（数字型doc_id与序号会歧义）
- qwen3.6-plus 走 compatible-mode 需 extra_body={'enable_thinking':bool}；SOCKS代理需 httpx[socks]
- **pack2_text14 == pack2_text20 是同一份研报**（字节级相同）；docsel 评测中二者互换不算错
- 闭卷基线 39/98（fin 2/20 最反先验）；检索净救回24题；判分口径=原句轻度转述即对（丢'除'字前缀官方也判对）
- 已修复召回大bug：多查询去重先低分占坑挤掉强命中→改跨查询取最高分
