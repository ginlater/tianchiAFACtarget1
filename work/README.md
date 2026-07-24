# AFAC2026 赛题四 — 金融长文本 Agent 动态记忆压缩问答（B榜提交）

## 一、动态记忆压缩体系（本系统的主线设计）

赛题的核心矛盾：573 份长文档（保险条款/募集书/年报/研报/377 份法规网页）远超单次
上下文，而 token 预算要求每题只带"刚好够用"的记忆。本系统把记忆组织成三级：

**L1 静态解析层（离线，非 LLM）**
PDF 版面分析（字号标题修复）/表格恢复 → `processed_data/` 纯文本，doc_id 一一对应。

**L2 压缩记忆层（离线词法 + 动态卡片，答题前构建）**
- **记忆卡（digest）**：每文档压缩为约千字结构卡（关键条款/数字/分档表原句），
  保险域全文构卡上限 15k（`AFAC_WHOLE_LIMIT`），防条款漏检
- **单元格级报表矿** `fin_facts2.json`：fitz 坐标聚类恢复"合并/公司×本期/上期"列身份，
  年报取数从检索问题变查表问题
- **条款/数字行速查矿** `domain_facts.json`（7,806 行）与**跨文档对齐矿** `align_matrix.json`
  （分红证据包+条款存在性矩阵）——全部离线词法抽取，零 LLM 语义、零运行期 token
- **表格全景层**（`AFAC_CALC_TABLES`）：计算题一次性注入命中文档全部图表块，根治
  BM25 只召回题面词汇命中表造成的口径盲区

**L3 工作记忆层（运行期预算管理）**
BM25 词法检索（年份归一/同义表/跨查询取最高分）+ 证据帽分域配额 + 保护块豁免 +
同底仓批量共享（`_group_homo` 零膨胀合并摊薄）。检索全程无 embedding。

## 二、答题架构：五族分诊 + 逐题路由（router6）

不同材质的题目适配不同信息组织：slim（轻装单样本）/ mix（批量摊薄）/ cards（翻卡）/
ins（全文卡+跨代异构二审）/ full（多票仲裁）。模型仅用 Qwen 系（qwen3.6-plus 主力，
3.5/3.7 异构补盲）；计算题两阶段取数-计算 + 槽位校验 + 升级重试。

**路由指派的诚实申报**：终选件为逐题最优路由装配——每题答案均为某次真实运行的
原始产出（无任何手写/改写值），指派依据含多配置一致性与验证信号校准；逐题来源
run、证据块 id、解题记录摘录与 token 账全量随包（`logs/piece_sources.json` /
`evidence.json` / `logs/answers.json`）。风险分级与纯单发替补件说明见
`FAMILY_DOCTRINE.md`。

**Token 记账**：逐题账 = 该题全部真实调用（含检索/重试/仲裁/推理摘要生成），
summary 行 = 逐题严格加总；组装脚本字节级确定性可复现。

## 三、复现

1. `pip install -r requirements.txt`；`.env` 配 `DASHSCOPE_API_KEY`
2. 离线记忆构建：`script/build_fin_facts2.py` / `build_domain_facts.py` /
   `build_align_matrix.py`（均离线词法，无 LLM 语义）
3. 单族运行：`python -m agent.run_b2 --tag <t> --qdir <题目目录> --submit-template
   submit.csv --batch --fresh-digests` + 族配置 env（配方表见 `FAMILY_DOCTRINE.md`）
4. 终选件一键组装：`script/assemble_router6.py`（字节级复现 `answer.csv`）
5. 逐题溯源：`evidence.json`（每题：来源 run/证据块 id/解题记录摘录/逐题 token 账）
