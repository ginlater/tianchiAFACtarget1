#!/usr/bin/env python3
"""智能路由器：每域一次性路由到其最适配配置，端到端单次执行出完整提交。

路由表 = 57次实验矩阵蒸馏（每域冠军配方）。全程一遍跑完、全账入册。
用法: .venv/bin/python script/run_router.py <tag>
"""
import json, os, pathlib, subprocess, sys, time

ROOT = pathlib.Path(__file__).resolve().parents[2]
WORK = ROOT / "work"
tag = sys.argv[1] if len(sys.argv) > 1 else "b_routerF"

# 路由表: 域 → (环境配置, 额外CLI参数)
ROUTES = {
    "regfc": {      # reg+fc → 批量摊薄(mixC实证高命中域)
        "qids_file": "output/qids_regfc.txt",
        "env": {"AFAC_SLIM": "1", "AFAC_SLIM4": "1", "AFAC_NO_DIGEST": "1",
                "AFAC_DIGEST_KEEP": "none", "AFAC_STABLE": "1", "AFAC_ALIGN": "1"},
        "args": ["--batch", "--workers", "4"],
    },
    "resolo": {     # res → 逐题瘦+双票投票(病理: res塌方是骰子, 投票是解药)
        "qids_file": "output/qids_resonly.txt",
        "env": {"AFAC_SLIM": "1", "AFAC_NO_DIGEST": "1",
                "AFAC_DIGEST_KEEP": "none", "AFAC_STABLE": "1", "AFAC_ALIGN": "1",
                "AFAC_R1_VOTES": "2", "AFAC_R1_ESC": "1"},
        "args": ["--workers", "4"],
    },
    "insfull": {    # ins → insE冠军配方: 节选构卡+LEAN_R2(实证20/20@585k)
        "qids_file": "output/qids_ins.txt",
        "env": {"AFAC_STABLE": "1", "AFAC_ALIGN": "1", "AFAC_VERIFY_MODEL": "qwen3.5-plus",
                "AFAC_DIGEST_KEEP": "insurance", "AFAC_NO_DIGEST": "1",
                "AFAC_LEAN_R2": "1", "AFAC_WHOLE_LIMIT": "15000"},
        "args": ["--workers", "3", "--fresh-digests"],
    },
    "finfull": {    # fin → full12配方+单元格速查表(012稳定治愈2/2)
        "qids_file": "output/qids_fin.txt",
        "env": {"AFAC_STABLE": "1", "AFAC_ALIGN": "1", "AFAC_VERIFY_MODEL": "qwen3.5-plus",
                "AFAC_ARB_VOTES": "3", "AFAC_R1_VOTES": "2",
                "AFAC_FIN_FACTS": "2"},
        "args": ["--workers", "3", "--fresh-digests"],
    },
}

t0 = time.time()
procs = []
for name, r in ROUTES.items():
    env = dict(os.environ)
    env.update(r["env"])
    qids = open(WORK / r["qids_file"]).read().strip()
    sub_tag = f"{tag}_{name}"
    cmd = [str(WORK / ".venv" / "bin" / "python"), "-m", "agent.run_b2",
           "--tag", sub_tag, "--qdir", "../upload_b/question_b",
           "--submit-template", "../upload_b/submit.csv",
           "--qids", qids] + r["args"]
    logf = open(WORK / "output" / f"{sub_tag}.out", "w")
    p = subprocess.Popen(cmd, cwd=str(WORK), env=env, stdout=logf,
                         stderr=subprocess.STDOUT)
    procs.append((name, p))
    print(f"[router] {name} 已发射 (pid {p.pid})", flush=True)

for name, p in procs:
    rc = p.wait()
    print(f"[router] {name} 完成 rc={rc} ({time.time()-t0:.0f}s)", flush=True)

# 合并三支路由的答案与逐题账 → 单一提交结构
answers, per_qid = {}, {}
for name in ROUTES:
    sub = WORK / "output" / f"{tag}_{name}"
    a = json.load(open(sub / "answers.json"))
    led = json.load(open(sub / "token_ledger.json"))["per_qid"]
    for q, v in a.items():
        answers[q] = v
        per_qid[q] = list(led.get(q, [0, 0]))
outdir = WORK / "output" / tag
outdir.mkdir(exist_ok=True)
json.dump(answers, open(outdir / "answers.json", "w"), ensure_ascii=False,
          indent=1)
json.dump({"per_qid": per_qid, "calls": []},
          open(outdir / "token_ledger.json", "w"))
tot = sum(sum(v) for v in per_qid.values())
print(f"[router] 合并完成: {len(answers)}题 答题账{tot:,} → {outdir}")
