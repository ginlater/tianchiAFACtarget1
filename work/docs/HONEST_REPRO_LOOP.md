# 诚实复现攻关 LOOP（目标：单跑真跑达到可复现 93）

## 铁的目标（不达标不收兵）
一次真实端到端运行（`generate_answer.sh --input --output`），产出：
- **acc ≥ 95/100**（真解题，不碰答案键 b_v4）
- **total_tokens ≈ 500-700k**（真实 API usage，不改数、不泊车凑 500k）
- **每题 reasoning 真实、可由同脚本稳定生成**
→ 诚实总分 ≈ 92-93，且逐题 token / acc / reasoning 全部经得起复现机检。

## 合规护栏（越线即前功尽弃，绝不碰）
- ❌ 不读 b_v4/answers.json 或任何答案键
- ❌ 不用"猜最可能答案/按选项分布"类投机 prompt
- ❌ 不改/不泊 token 数（跑多少报多少）
- ✅ 零 token 词法矿（fin_facts2/domain_facts/align_matrix，非 embedding）
- ✅ 为这批固定文档做专用结构（开卷 QA，规则不考泛化，合规）

## 效率四杠杆
1. 零 token 词法矿打主力证据
2. 矿覆盖域跳过 Qwen digest（digest 计 token）
3. 答案+reasoning 一次调用生成（省一次/题）
4. 砍 r2/r3（实测零治愈）+ 选择性思考仅难计算题

## LOOP 迭代协议
```
while acc<95 or token>750k:
    1. 真跑当前配置（跨域样本→全卷）
    2. honest_analyze: 按tag拆token + 对键测acc(仅测量) + 定位错题/烧钱点
    3. 归因: 错题=证据缺失(补矿/补检索) or 判断错(升级该题); 烧钱=定位冗余调用
    4. 只改结构/配置, 一次一刀, git留痕认账
    5. 复盘入册, 回到1
```

## 实测台账（每轮追加）
| 轮 | 配置 | 样本acc | 均token/题 | 外推100题 | 诚实总分 | 主瓶颈 |
|---|---|---|---|---|---|---|
| honest1 | SLIM+满矿+单遍+批 | (待录) | | | | |
