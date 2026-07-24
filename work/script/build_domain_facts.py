#!/usr/bin/env python3
"""全域离线事实表 v1（零token词法抽取——赛题'记忆压缩'主题的极致形态）。

ins: 条款关键行（身故/满期/犹豫期/免责/借款/给付比例/未成年人/自杀 等 + 金额百分比行）
fc:  募集书关键行（兑付/利率/担保/违约/评级/日期/金额 行）
产物: processed_data/domain_facts.json {doc_id: [行...]}，与 fin_facts2 并行使用。
"""
import json, pathlib, re

WORK = pathlib.Path(__file__).resolve().parents[1]
PD = WORK / "processed_data"

INS_KW = re.compile(
    r"身故|满期|犹豫期|免责|责任免除|借款|贷款|给付比例|未成年人|自杀|现金价值|"
    r"宽限期|复效|退保|减保|部分领取|生存金|红利|万能|结算利率|保证利率|施救|"
    r"等待期|重大疾病|轻症|中症|豁免")
FC_KW = re.compile(
    r"兑付|付息|利率|担保|违约|评级|回售|赎回|摘牌|上市|起息|到期|募集资金|"
    r"发行规模|票面|受托管理|债券持有人|交叉保护|偿债")
NUM = re.compile(r"\d")
PAGE = re.compile(r"\[P(\d+)\]")


def extract(domain, kw, min_num=False):
    out = {}
    for f in sorted((PD / domain).glob("*.txt")):
        doc = f.stem
        rows, page = [], 0
        for ln in open(f, encoding="utf-8", errors="ignore"):
            m = PAGE.search(ln)
            if m:
                page = int(m.group(1))
            t = ln.strip()
            if not (8 <= len(t) <= 160):
                continue
            if kw.search(t) and (not min_num or NUM.search(t)):
                rows.append(f"[P{page}] {t[:140]}")
        # 去重保序
        seen, ded = set(), []
        for r in rows:
            k = r.split("] ", 1)[-1][:60]
            if k not in seen:
                seen.add(k)
                ded.append(r)
        out[doc] = ded[:400]
        print(f"{domain}/{doc}: {len(ded)}行")
    return out


def main():
    result = {}
    result.update(extract("insurance", INS_KW))
    result.update(extract("financial_contracts", FC_KW))
    json.dump(result, open(PD / "domain_facts.json", "w"), ensure_ascii=False,
              indent=0)
    tot = sum(len(v) for v in result.values())
    print(f"→ domain_facts.json {len(result)}文档 {tot}行")


if __name__ == "__main__":
    main()
