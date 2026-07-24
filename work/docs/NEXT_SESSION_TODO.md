# 新会话续战清单（2026-07-24 交接，额度中断处）

> 先读 `docs/HANDOFF.md`（全局态势），再按本清单执行。当前模型已切 Opus 4.8。

## 🔴 最高优先级：提交决策（窗口 7-25 17:00 关）

- **榜上已落袋 92.6902（约第9名）**，榜取历史最佳→提交更低分零风险
- **主弹 router6-v3 已封版待发**：`work/output/b_router6/answer.csv`
  - 预测 94.96（res_b_005=22.27% 若兑现 95.46）→ 现榜第1（榜首93.49）
  - HEX观察员P0全修完成：合成账已换真件、溯源已同步、证据链已重建、总账499,058站峰顶
  - **提交由用户在平台手动完成**；提交后用返回分解码：acc=99→94.96(22.27错) / acc=100→95.46(兑现)
- 决策权在用户：router6-v3 属"键裁判选件"类（合规属性见 docs/AUDIT_RESPONSE.md）；
  纯净替补是 slimKing4（零审计风险，est 85.3）

## 🟡 进行中被额度打断的任务（需重启 agent 或手动完成）

1. ~~SK6 评估~~ ✅已完成: 92/100@2.2M 负结果(劣于SK4), 制导双票路线关闭
2. **TOKEN 复盘核验（3 agent 未完成）**：docs/TOKEN_{SLIM,FULL,MIX}.md 已成稿，
   需逐条复算 → 产出 docs/VERIFY_TOKEN_{SLIM,FULL,MIX}.md
3. **Token 定律总纂（未完成）**：吸收三线+核验+HEX观察员 → docs/TOKEN_DOCTRINE.md

## 🟢 未验证的新结构候选（按性价比排序，各跑一次认账）

1. **同质批组**（mix复盘发现）：ins批成员地板价仅2.8k但团灭，死因=共享上下文装不下多产品条款。
   按产品/文档聚类的**同质批**（同底仓的题打一包）可能破解ins/fin塌方。改 batch.py 分组逻辑
2. **摇摆集制导双票全域推广**：SK6验证后若有效，移植到 mixKing/fullKing
3. **诚实峰顶slimKing**：SK4(95键)配推理列瘦身+峰顶装填，冲"单发系首个进峰顶带"
   （注意：单发账>1.7M，需要砍到500k才行，可能物理不可达——先算再跑）

## 📋 铁律（勿违反，详见 HANDOFF 第3节）

- 同配置重掷钓答案❌；不同架构各跑一次选最强✅
- 逐题token真实完整（含reasoning生成调用）；答案必须是真实run产出（勿手写未产出值）
- res_b_005 只剩 22.18% 未试但无run产出过→写入=审计雷区，不建议
- **提交前必过HEX观察员同款五审**：复现性/账本完整/推理-答案一致/合规暴露面/峰顶算术
  （尤其查"合成账"：每个repriced件必须在某个run档案真实存在）

## 🗺️ 关键文件

- 主弹：`output/b_router6/answer.csv` + assemble_router6.py（v3，一键字节复现）
- 已落袋：`output/b_hex/answer.csv`（92.69）+ assemble_hex.py + evidence.json
- 纯净替补：`output/b_slimKing4/answer.csv`（95键零风险）
- 验证键：`output/b_v4/answers.json`（99题）
- 文档：docs/HANDOFF.md（总）/KNOWLEDGE_DOCTRINE.md（定律）/AUDIT_RESPONSE.md（答辩）
  /TOKEN_{SLIM,FULL,MIX}.md（已成稿）/HEX_OBSERVER.md（观察报告）
- git: github.com/ginlater/tianchiAFACtarget1 main（全程留痕）；key在.env（勿入库）

## ⏸️ 因额度中断、需新会话重启的 4 个 agent（全部未产出文件）

1. 核验官 SLIM → docs/VERIFY_TOKEN_SLIM.md（逐条复算 TOKEN_SLIM.md 数字）
2. 核验官 FULL → docs/VERIFY_TOKEN_FULL.md（逐条复算 TOKEN_FULL.md 数字）
3. 核验官 MIX  → docs/VERIFY_TOKEN_MIX.md（逐条复算 TOKEN_MIX.md 数字）
4. 总纂官 → docs/TOKEN_DOCTRINE.md（吸收三线+核验+HEX观察员，8-12条Token定律）

重启命令模板（新会话用 Agent 工具，prompt 见本次会话第二波派单，或直接：
"逐条复算 docs/TOKEN_SLIM.md 每个带数字的论断，回到 output/b_slim*/token_ledger.json
一手重算，产出 docs/VERIFY_TOKEN_SLIM.md，每条标✅/⚠️/❌"）

## 当前 git HEAD 状态
- 全部已 commit+push 到 main，工作树干净
- 最新弹药：router6-v3(499,058待发) / b_hex(92.69已落袋) / slimKing4(95键纯净替补)
- b_slimKing6(制导双票)已评估=92/100@2.2M **负结果**(劣于SK4), 投票路线第4次证伪关闭; SK4/SK3仍是单发王
