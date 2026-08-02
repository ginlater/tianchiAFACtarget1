#!/usr/bin/env python3
"""全域离线事实表 v1（零token词法抽取——赛题'记忆压缩'主题的极致形态）。

ins: 条款关键行（身故/满期/犹豫期/免责/借款/给付比例/未成年人/自杀 等 + 金额百分比行）
fc:  募集书关键行（兑付/利率/担保/违约/评级/日期/金额 行）
产物: processed_data/domain_facts.json {doc_id: [行...]}，与 fin_facts2 并行使用。
"""
import argparse
import json
import pathlib
import re

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


def extract(domain, kw, min_num=False, *, processed_dir=None, verbose=False):
    processed_dir = pathlib.Path(processed_dir or PD)
    out = {}
    for f in sorted((processed_dir / domain).glob("*.txt")):
        doc = f.stem
        rows, page = [], 0
        with f.open(encoding="utf-8", errors="ignore") as source:
            for ln in source:
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
        if verbose:
            print(f"{domain}/{doc}: {len(ded)}行")
    return out


RES_KW = re.compile(
    r"同比|环比|增速|增长率|占比|渗透率|市占|预测|预计|目标价|评级|产能|出货|装机|"
    r"销量|均价|毛利率|净利率|CAGR|市场规模|需求|供给|盈利预测|估值")


def build(processed_dir=PD, output_path=None, *, verbose=False):
    processed_dir = pathlib.Path(processed_dir)
    result = {}
    result.update(extract("insurance", INS_KW, processed_dir=processed_dir,
                          verbose=verbose))
    result.update(extract("financial_contracts", FC_KW,
                          processed_dir=processed_dir, verbose=verbose))
    result.update(extract("research", RES_KW, min_num=True,
                          processed_dir=processed_dir, verbose=verbose))
    target = pathlib.Path(output_path or processed_dir / "domain_facts.json")
    target.write_text(json.dumps(result, ensure_ascii=False, indent=0) + "\n",
                      encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=pathlib.Path, default=PD)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    result = build(args.processed, args.output, verbose=args.verbose)
    tot = sum(len(v) for v in result.values())
    print(f"→ domain_facts.json {len(result)}文档 {tot}行")


if __name__ == "__main__":
    main()
