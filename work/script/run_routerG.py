#!/usr/bin/env python3
"""router2 级联路由：难题多花token、简单题少花——端到端单次执行。

阶段1(侦察): 全卷按域路由到便宜配方
阶段2(升级): 难度分诊表(历史答案熵,零答案键接触)标定的困难题 → 重装配方重答
合并: 升级结果覆盖侦察结果(运行内决策,不接触任何答案键)。全账入册。
用法: .venv/bin/python script/run_router2.py <tag>
"""
import json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
tag = sys.argv[1] if len(sys.argv) > 1 else "b_router2"

diff = json.load(open(WORK / "output" / "difficulty_map.json"))
hard = sorted(q for q in diff["hard"] if "_b_" in q)
print(f"[r2] 困难层{len(hard)}题: {hard}", flush=True)

SCOUT = {   # 阶段1: 全卷便宜路由(mixC三域配方 + slim式fin/ins)
    "qА_trio": {
        "qids_file": "output/qids_mixC.txt",
        "env": {"AFAC_SLIM": "1", "AFAC_SLIM4": "1", "AFAC_NO_DIGEST": "1",
                "AFAC_DIGEST_KEEP": "none", "AFAC_STABLE": "1"},
        "args": ["--batch", "--workers", "4"],
    },
    "qB_finins": {
        "qids_file": "output/qids_mixD.txt",
        "env": {"AFAC_SLIM": "1", "AFAC_SLIM4": "1", "AFAC_NO_DIGEST": "1",
                "AFAC_DIGEST_KEEP": "none", "AFAC_STABLE": "1",
                "AFAC_FIN_FACTS": "2"},
        "args": ["--batch", "--workers", "4"],
    },
}
HEAVY_ENV = {"AFAC_STABLE": "1", "AFAC_VERIFY_MODEL": "qwen3.5-plus",
             "AFAC_ARB_VOTES": "3", "AFAC_R1_VOTES": "2",
             "AFAC_LEAN_R2": "1",  # FULL手册翻案:单刀省127k且+2键(full12vs13对照)
             "AFAC_DIGEST_KEEP": "insurance,financial_reports",
             "AFAC_NO_DIGEST": "1",
             "AFAC_FIN_FACTS": "2",       # 单元格速查表(fin_b_012类伤解药)
             "AFAC_WHOLE_LIMIT": "15000"}  # ins节选构卡(insE 20/20配方)

t0 = time.time()
def launch(name, env_extra, args, qids):
    env = dict(os.environ); env.update(env_extra)
    sub_tag = f"{tag}_{name}"
    cmd = [str(WORK/".venv"/"bin"/"python"), "-m", "agent.run_b2",
           "--tag", sub_tag, "--qdir", "../upload_b/question_b",
           "--submit-template", "../upload_b/submit.csv", "--qids", qids] + args
    logf = open(WORK/"output"/f"{sub_tag}.out", "w")
    return sub_tag, subprocess.Popen(cmd, cwd=str(WORK), env=env,
                                     stdout=logf, stderr=subprocess.STDOUT)

# 阶段1
procs = []
for name, r in SCOUT.items():
    qids = open(WORK / r["qids_file"]).read().strip()
    procs.append(launch(name, r["env"], r["args"], qids))
    print(f"[r2] 侦察 {name} 发射", flush=True)
for st, p in procs:
    p.wait(); print(f"[r2] {st} 完成 ({time.time()-t0:.0f}s)", flush=True)

# 阶段2: 困难题重装升级（fresh卡入账; 跨代主攻: 确认偏差题命中全在3.5/3.7）
st2, p2 = launch("heavy", HEAVY_ENV,
                 ["--workers", "3", "--fresh-digests",
                  "--model", "qwen3.7-plus"], ",".join(hard))
print(f"[r2] 升级纵队发射({len(hard)}题)", flush=True)
p2.wait(); print(f"[r2] {st2} 完成 ({time.time()-t0:.0f}s)", flush=True)

# 合并: 升级覆盖侦察
answers, per_qid = {}, {}
for name in list(SCOUT) + ["heavy"]:
    sub = WORK / "output" / f"{tag}_{name}"
    a = json.load(open(sub / "answers.json"))
    led = json.load(open(sub / "token_ledger.json"))["per_qid"]
    for q, v in a.items():
        if name == "heavy" or q not in answers:
            answers[q] = v
        pq = per_qid.setdefault(q, [0, 0])
        lv = led.get(q, [0, 0])
        pq[0] += lv[0]; pq[1] += lv[1]   # 侦察+升级都计账(诚实)
outdir = WORK / "output" / tag
outdir.mkdir(exist_ok=True)
json.dump(answers, open(outdir/"answers.json", "w"), ensure_ascii=False, indent=1)
json.dump({"per_qid": per_qid, "calls": []}, open(outdir/"token_ledger.json", "w"))
tot = sum(sum(v) for v in per_qid.values())
print(f"[r2] 级联完成: {len(answers)}题 全账{tot:,} → {outdir}")
