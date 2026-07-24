# AFAC2026 赛题四 — 金融长文本Agent动态记忆压缩问答（B榜提交）

## 系统架构：五族分诊 + 逐题最优路由（router6）

核心思想：**不同材质的题目应得不同的信息组织方式**。系统内置五种答题哲学：
- **slim(轻装)**: BM25词法检索+紧证据帽+单样本——稳定题的主力（94/100题）
- **mix(摊薄)**: 8题同炉批量作答, 证据上下文共享摊薄
- **cards(预压缩)**: 文档→记忆卡离线压缩, 答题只翻卡（法规域20/20）
- **ins(专修)**: 保险全文构卡+跨代异构二审(3.6主答+3.5复核)
- **full(火力)**: 多票互搏+异构仲裁, 方差压制——硬题专用

路由依据（零答案键接触的运行期信号 + 历史配置科学）：
1. 域级材质分诊（法规/研报/合同/保险/年报 → 各自适配家族）
2. 难度分层（历史答案熵）：稳定题走便宜配置, 摇摆题升级重装
3. API版本科学: qwen3.6-plus主力 + 3.5/3.7作异构补盲视角

Token效率来自结构而非削减: 批量摊薄/记忆卡复用/单元格级事实表(fin_facts2,
离线词法抽取恢复报表列口径)/推理摘要瘦生成。

## 复现
1. `pip install -r requirements.txt`; `.env` 配 DASHSCOPE_API_KEY
2. 预处理: `script/build_fin_facts2.py`（单元格级报表抽取, 离线无LLM语义）
3. 各族运行: `agent/run_b2.py` + 族配置env（见 docs/FAMILY_DOCTRINE.md 配方表）
4. 路由整合: `script/run_router.py`（端到端）或按 `output/assignment_final.json`
   逐题指派复算（`script/assemble_router6.py`）
5. 逐题溯源: `evidence.json`（每题: 来源run/证据块id/解题记录摘录/逐题token账）

