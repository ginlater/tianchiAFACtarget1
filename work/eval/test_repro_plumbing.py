"""Regression tests for the honest-reproduction audit plumbing.

These tests deliberately avoid network access.  They cover the invariants that
the official reproduction material must be able to demonstrate: exact shared
token allocation, complete API logging, per-question batch reasoning and
reasoning recovery from the current run only.
"""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agent import answerer, batch, qwen_client
from agent.repro import LockedJsonlWriter, collect_reasonings


class _Usage:
    def model_dump(self):
        return {"prompt_tokens": 11, "completion_tokens": 7,
                "total_tokens": 18}


class _Completions:
    def create(self, **kwargs):
        message = SimpleNamespace(content="分析完整。\n答案: AC",
                                  reasoning_content="内部思考")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)],
                               usage=_Usage())


class ReproPlumbingTests(unittest.TestCase):
    def setUp(self):
        qwen_client.close_audit()
        qwen_client.LEDGER = qwen_client.TokenLedger()

    def tearDown(self):
        qwen_client.close_audit()

    def test_shared_allocation_preserves_every_token(self):
        ledger = qwen_client.TokenLedger()
        ledger.add("_batch", "qwen-test",
                   {"prompt_tokens": 10, "completion_tokens": 5},
                   "b1", allocation_qids=["q1", "q2", "q3"])
        self.assertEqual(ledger.totals(), (10, 5, 15))
        self.assertEqual(ledger.per_qid,
                         {"q1": [4, 2], "q2": [3, 2], "q3": [3, 1]})
        call = ledger.calls[0]
        self.assertEqual(call["allocation_qids"], ["q1", "q2", "q3"])
        self.assertEqual(sum(v[0] for v in call["allocated_usage"].values()),
                         10)
        self.assertEqual(sum(v[1] for v in call["allocated_usage"].values()),
                         5)

    def test_weighted_allocation_splits_visible_and_hidden_completion(self):
        ledger = qwen_client.TokenLedger()
        usage = {
            "prompt_tokens": 11,
            "completion_tokens": 13,
            "total_tokens": 24,
            "completion_tokens_details": {"reasoning_tokens": 5},
        }
        plan = {
            "strategy": "test_weighted_v1",
            "prompt_weights": {"q1": 1, "q2": 3},
            "completion_weights": {"q1": 3, "q2": 1},
            "weight_basis": {"unit": "test_chars"},
        }
        allocation = ledger.add(
            "_batch", "qwen-test", usage, "b1",
            allocation_qids=["q1", "q2"], allocation_plan=plan)
        # Prompt: 3/8. Hidden reasoning follows prompt: 1/4.  The remaining
        # eight visible tokens follow answer-block weights: 6/2.
        self.assertEqual(ledger.per_qid,
                         {"q1": [3, 7], "q2": [8, 6]})
        self.assertEqual(ledger.totals(), (11, 13, 24))
        self.assertEqual(allocation["allocated_reasoning_tokens"],
                         {"q1": 1, "q2": 4})
        self.assertEqual(allocation["allocated_visible_completion_tokens"],
                         {"q1": 6, "q2": 2})
        self.assertEqual(sum(allocation["allocated_prompt_tokens"].values()),
                         usage["prompt_tokens"])
        self.assertEqual(sum(
            allocation["allocated_reasoning_tokens"].values()) + sum(
            allocation["allocated_visible_completion_tokens"].values()),
            usage["completion_tokens"])
        self.assertTrue(allocation["conservation"]["ok"])
        self.assertEqual(
            allocation["apportionment"],
            "stable_largest_remainder_input_order_v1")

    def test_largest_remainder_ties_follow_input_order_not_qid(self):
        got = qwen_client._largest_remainder(
            2, ["z-last-lexically", "a-first-lexically", "middle"],
            {"z-last-lexically": 1, "a-first-lexically": 1, "middle": 1})
        self.assertEqual(got, {
            "z-last-lexically": 1,
            "a-first-lexically": 1,
            "middle": 0,
        })

    def test_batch_prompt_weights_include_own_text_and_equal_shared_base(self):
        qs = [
            {"qid": "q-short", "answer_format": "mcq",
             "question": "短题？", "options": {"A": "甲", "B": "乙"}},
            {"qid": "q-long", "answer_format": "mcq",
             "question": "这是明显更长的第二道题目？",
             "options": {"A": "较长的甲选项", "B": "较长的乙选项"}},
        ]
        rendered = [answerer._q_text(q) for q in qs]
        prompt = "共享证据底仓和统一指令" + "\n\n".join(rendered)
        plan = batch._batch_prompt_allocation_plan(prompt, qs)
        shared = len(prompt) - sum(map(len, rendered))
        self.assertEqual(plan["prompt_weights"], {
            "q-short": 2 * len(rendered[0]) + shared,
            "q-long": 2 * len(rendered[1]) + shared,
        })
        self.assertGreater(plan["prompt_weights"]["q-long"],
                           plan["prompt_weights"]["q-short"])
        self.assertEqual(plan["weight_basis"]["shared_prompt_chars"],
                         shared)

    def test_batch_prompt_weights_charge_evidence_to_actual_owners(self):
        qs = [
            {"qid": "q-light", "domain": "financial_reports",
             "answer_format": "mcq",
             "question": "同样长度？", "options": {"A": "甲"}},
            {"qid": "q-heavy", "domain": "financial_reports",
             "answer_format": "mcq",
             "question": "同样长度？", "options": {"A": "乙"}},
        ]
        # Ten chars belong only to q-light, 100 only to q-heavy, and twenty
        # are a genuinely shared evidence block.  The final seven chars stand
        # for global instructions outside the evidence string.
        ownership = {
            "version": "test_owner_map_v1",
            "text_chars": 130,
            "segments": [
                {"chars": 10, "owners": ["q-light"], "kind": "mine"},
                {"chars": 100, "owners": ["q-heavy"], "kind": "mine"},
                {"chars": 20, "owners": ["q-light", "q-heavy"],
                 "kind": "shared_chunk"},
            ],
        }
        rendered = [answerer._q_text(q) for q in qs]
        prompt = "证" * 130 + "".join(rendered) + "全局指令七字!"
        plan = batch._batch_prompt_allocation_plan(
            prompt, qs, evidence_ownership=ownership)
        basis = plan["weight_basis"]
        self.assertTrue(basis["evidence_ownership_valid"])
        self.assertTrue(basis["prompt_weight_conserved"])
        self.assertEqual(basis["prompt_weight_scale"], 2)
        self.assertEqual(basis["evidence_owner_char_units"],
                         {"q-light": 40, "q-heavy": 220})
        self.assertEqual(
            plan["prompt_weights"]["q-heavy"] -
            plan["prompt_weights"]["q-light"],
            180)
        self.assertEqual(sum(plan["prompt_weights"].values()),
                         len(prompt) * basis["prompt_weight_scale"])
        self.assertEqual(basis["evidence_allocation_policy"],
                         "source_question_owners_v1")

    def test_research_evidence_is_auditable_shared_batch_floor(self):
        qs = [
            {"qid": "first", "domain": "research", "answer_format": "mcq",
             "question": "同样长度？", "options": {"A": "甲"}},
            {"qid": "second", "domain": "research", "answer_format": "mcq",
             "question": "同样长度？", "options": {"A": "乙"}},
        ]
        ownership = {
            "version": "source_map_v1",
            "text_chars": 110,
            "segments": [
                {"chars": 10, "owners": ["first"], "kind": "chunk"},
                {"chars": 100, "owners": ["second"], "kind": "chunk"},
            ],
        }
        rendered = [answerer._q_text(q) for q in qs]
        prompt = "证" * 110 + "".join(rendered) + "共享指令"
        plan = batch._batch_prompt_allocation_plan(
            prompt, qs, evidence_ownership=ownership)
        basis = plan["weight_basis"]
        self.assertEqual(plan["prompt_weights"]["first"],
                         plan["prompt_weights"]["second"])
        self.assertEqual(basis["evidence_allocation_policy"],
                         "research_cross_document_shared_floor_v1")
        segment = basis["evidence_ownership_segments"][0]
        self.assertEqual(segment["source_owners"], ["first"])
        self.assertEqual(segment["charged_owners"], ["first", "second"])
        self.assertTrue(basis["prompt_weight_conserved"])

    def test_domain_semantic_allocation_is_invariant_to_qid_spelling(self):
        def make_qs(a, b):
            return [
                {"qid": a, "domain": "research", "answer_format": "mcq",
                 "question": "短题？", "options": {"A": "甲"}},
                {"qid": b, "domain": "research", "answer_format": "mcq",
                 "question": "这是较长的问题？",
                 "options": {"A": "较长选项"}},
            ]

        def weights(qs):
            owners = [q["qid"] for q in qs]
            ownership = {
                "version": "source_map_v1", "text_chars": 40,
                "segments": [{"chars": 40, "owners": [owners[0]],
                              "kind": "chunk"}],
            }
            prompt = "证" * 40 + "".join(answerer._q_text(q) for q in qs)
            plan = batch._batch_prompt_allocation_plan(
                prompt, qs, evidence_ownership=ownership)
            return [plan["prompt_weights"][qid] for qid in owners]

        self.assertEqual(weights(make_qs("res_like_001", "res_like_002")),
                         weights(make_qs("nothing", "arbitrary")))

    def test_batch_evidence_reports_registry_and_chunk_owners(self):
        qs = [
            {"qid": "q-light", "domain": "research", "doc_ids": ["a"],
             "question": "甲？", "options": {"A": "甲"},
             "answer_format": "multi"},
            {"qid": "q-heavy", "domain": "research", "doc_ids": ["b"],
             "question": "乙？", "options": {"A": "乙"},
             "answer_format": "multi"},
        ]

        def fake_gather(q, **_kwargs):
            doc = q["doc_ids"][0]
            chunk = {"id": f"{doc}#c1", "doc_id": doc,
                     "page": 1, "text": doc * 9}
            return chunk["text"], [chunk], {chunk["id"]}

        def fake_registry(q):
            return ("L" * 10 if q["qid"] == "q-light" else "H" * 100)

        with mock.patch.object(batch, "_doc_title", side_effect=lambda d: d), \
                mock.patch.object(batch, "_use_digest", return_value=False), \
                mock.patch.object(batch, "gather_evidence",
                                  side_effect=fake_gather), \
                mock.patch.object(answerer, "financial_registry_block",
                                  side_effect=fake_registry), \
                mock.patch.object(answerer, "domain_facts_block",
                                  return_value=""), \
                mock.patch.object(answerer, "align_block", return_value=""):
            ev, _ids, _low, ownership = batch._batch_evidence(
                qs, return_ownership=True)
        self.assertTrue(ownership["conserved"])
        self.assertEqual(ownership["text_chars"], len(ev))
        registry = [s for s in ownership["segments"]
                    if s["kind"] == "financial_registry"]
        self.assertEqual(registry, [
            {"chars": 10, "owners": ["q-light"],
             "kind": "financial_registry"},
            {"chars": 100, "owners": ["q-heavy"],
             "kind": "financial_registry"},
        ])
        chunks = [s for s in ownership["segments"]
                  if s["kind"] == "retrieved_evidence_chunk"]
        self.assertEqual([s["owners"] for s in chunks],
                         [["q-light"], ["q-heavy"]])

    def test_batch_completion_weights_use_exact_visible_blocks(self):
        qs = [
            {"qid": "q1", "answer_format": "multi"},
            {"qid": "q2", "answer_format": "mcq"},
        ]
        text = ("批前说明\n【第1题 答案块】\n分析: 短\n答案: A\n"
                "【第2题 答案块】\n分析: 这是明显更长的答案块内容\n答案: B")
        plan = batch._batch_completion_allocation(text, qs)
        first = text[text.index("【第1题"):text.index("【第2题")].strip()
        second = text[text.index("【第2题"):].strip()
        shared = len(text) - len(first) - len(second)
        self.assertEqual(plan["completion_weights"],
                         {"q1": 2 * len(first) + shared,
                          "q2": 2 * len(second) + shared})
        self.assertGreater(plan["completion_weights"]["q2"],
                           plan["completion_weights"]["q1"])
        self.assertEqual(plan["weight_basis"]["visible_unassigned_chars"],
                         shared)
        self.assertTrue(plan["weight_basis"]["visible_weight_conserved"])
        self.assertEqual(sum(plan["completion_weights"].values()),
                         len(text) * 2)

    def test_research_batch_cap_is_env_driven_not_qid_driven(self):
        qs = [
            {"qid": f"arbitrary-{i}", "domain": "research",
             "doc_ids": [f"doc-{i}"], "question": "材料判断？",
             "options": {"A": "甲"}, "answer_format": "multi"}
            for i in range(8)
        ]
        seen_caps = []

        def fake_gather(_q, **kwargs):
            seen_caps.append(kwargs["cap"])
            return "", [], set()

        empty = mock.Mock(return_value="")
        with mock.patch.dict("os.environ", {
                "AFAC_RESEARCH_BATCH_RAW_CAP": "6000",
                "AFAC_SLIM4": "0"}), \
                mock.patch.object(batch, "_doc_title", side_effect=str), \
                mock.patch.object(batch, "_use_digest", return_value=False), \
                mock.patch.object(batch, "gather_evidence",
                                  side_effect=fake_gather), \
                mock.patch.object(answerer, "financial_registry_block",
                                  empty), \
                mock.patch.object(answerer, "fin_facts_block", empty), \
                mock.patch.object(answerer, "domain_facts_block", empty), \
                mock.patch.object(answerer, "align_block", empty):
            batch._batch_evidence(qs)
        # Cross-document research uses a 25% marginal allowance: 6000 *
        # (1 + .25*7), plus the research shared-breadth allowance, gives a
        # 19,500-character union cap split evenly across eight queries.
        self.assertEqual(seen_caps, [2437] * 8)

    def test_research_groups_are_balanced_and_dense_sources_are_solo(self):
        ordinary = [
            {"qid": f"arbitrary-{i}", "domain": "research",
             "doc_ids": [f"d{i % 5}", f"e{i % 3}"]}
            for i in range(17)
        ]
        with mock.patch.dict("os.environ", {
                "AFAC_HOMO_BATCH": "1", "AFAC_HOMO_MAX": "5",
                "AFAC_INS_CAPSULES": "0"}):
            groups = batch.group_questions(ordinary)
        self.assertEqual(sorted(map(len, groups)), [5, 6, 6])
        self.assertTrue(all(len(group) > 1 for group in groups))

        dense = dict(
            ordinary[0], qid="any-dense-id",
            doc_ids=[f"source-{i}" for i in range(6)],
            question="甲、乙、丙三个领域都出现了技术突破案例，哪些判断正确？",
            options={
                "A": "都说明关键技术能够降本",
                "B": "均形成了持续壁垒",
                "C": "全部带来了性能提升",
                "D": "所有路径都可快速模仿",
            }, answer_format="multi")
        mixed = ordinary[1:17] + [dense]
        with mock.patch.dict("os.environ", {
                "AFAC_HOMO_BATCH": "1", "AFAC_HOMO_MAX": "5",
                "AFAC_INS_CAPSULES": "0"}):
            groups = batch.group_questions(mixed)
        solo = [group for group in groups if len(group) == 1]
        self.assertEqual([[q["qid"] for q in group] for group in solo],
                         [["any-dense-id"]])
        self.assertEqual(sorted(len(group) for group in groups if len(group) > 1),
                         [8, 8])

        wide_but_ordinary = dict(
            dense, qid="generic-wide", question="多个行业都提到出海战略。",
            options={"A": "目的地集中东南亚", "B": "输出技术标准",
                     "C": "服务业模式不同", "D": "多区域布局产能"})
        self.assertFalse(batch._research_dense_case_check(wide_but_ordinary))

    def test_insurance_routes_balance_exclusions_and_isolate_deep_legal_checks(self):
        exclusions = [{
            "qid": f"opaque-exclusion-{i}", "domain": "insurance",
            "answer_format": "multi", "question": "哪些产品明确列明地震免责？",
            "options": {"A": "产品甲", "B": "产品乙"},
            "doc_ids": [f"doc-{i}-a", f"doc-{i}-b"],
        } for i in range(5)]
        minor = {
            "qid": "opaque-minor", "domain": "insurance",
            "answer_format": "multi",
            "question": "关于未成年人身故保险金限制，哪些产品明确提及该限制？",
            "options": {"A": "甲", "B": "乙", "C": "丙", "D": "丁"},
            "doc_ids": ["d1", "d2", "d3", "d4"],
        }
        suicide = {
            "qid": "opaque-suicide", "domain": "insurance",
            "answer_format": "multi",
            "question": "哪些产品有成立起2年内自杀但无民事行为能力人除外规则？",
            "options": {"A": "甲", "B": "乙", "C": "丙", "D": "丁"},
            "doc_ids": ["d1", "d2", "d3", "d4"],
        }
        compact = {
            "qid": "opaque-limit", "domain": "insurance",
            "answer_format": "multi",
            "question": "哪些产品明确约定保险金请求权诉讼时效为2年？",
            "options": {"A": "甲", "B": "乙"},
            "doc_ids": ["d1", "d2"],
        }
        age = {
            "qid": "opaque-age", "domain": "insurance",
            "answer_format": "multi", "question": "年龄错误后如何处理？",
            "options": {"A": "甲", "B": "乙"},
            "doc_ids": ["d1", "d2"],
        }
        with mock.patch.dict("os.environ", {
                "AFAC_HOMO_BATCH": "1", "AFAC_HOMO_MAX": "5",
                "AFAC_INS_CAPSULES": "1",
                "AFAC_INS_LITERAL_BATCH_MAX": "4"}):
            groups = batch.group_questions(
                exclusions + [compact, age, minor, suicide])
        self.assertEqual(sorted(len(g) for g in groups), [1, 1, 3, 4])
        solos = {g[0]["qid"] for g in groups if len(g) == 1}
        self.assertEqual(solos, {"opaque-minor", "opaque-suicide"})
        compact_group = next(g for g in groups
                             if any(q["qid"] == "opaque-limit" for q in g))
        self.assertEqual(len(compact_group), 4)
        self.assertTrue(batch._insurance_dense_exclusion_batch(
            exclusions[:3] + [compact]))
        self.assertFalse(batch._insurance_dense_exclusion_batch(
            compact_group[:3]))
        self.assertTrue(batch.requires_batch_singleton(minor))
        self.assertTrue(batch.requires_batch_singleton(suicide))

    def test_financial_scope_route_uses_visible_semantics_not_qid(self):
        def question(qid, text, options=None):
            return {
                "qid": qid, "domain": "financial_reports",
                "answer_format": "multi", "question": text,
                "options": options or {"A": "指标同比增长"},
                "doc_ids": ["annual_demo_2025_report"],
            }

        for qid in ("fin_like_012", "completely-arbitrary"):
            q = question(
                qid,
                "根据合并财务报表与母公司财务报表判断下列说法。")
            self.assertTrue(batch.requires_batch_singleton(q))
        self.assertTrue(batch.requires_batch_singleton(question(
            "option-only", "根据年度报告判断。",
            {"A": "合并口径为正", "B": "母公司口径为负"})))
        self.assertFalse(batch.requires_batch_singleton(question(
            "ordinary", "根据合并财务报表判断营业收入变化。")))
        self.assertFalse(batch.requires_batch_singleton({
            **question("calc", "比较合并口径和母公司口径。"),
            "answer_format": "calc",
        }))
        self.assertTrue(batch._financial_extended_note_check(question(
            "dividend", "比较多家公司全年现金分红。")))
        self.assertTrue(batch._financial_extended_note_check(question(
            "restated", "2024年数据采用年报调整前口径。")))
        self.assertTrue(batch._financial_extended_note_check(question(
            "three-year", "根据2023—2025年比较数据判断。")))
        self.assertFalse(batch._financial_extended_note_check(question(
            "precomputed", "根据2023—2025年偿债指标判断。")))

    def test_financial_shared_floor_keeps_source_owners_in_audit(self):
        qs = [{
            "qid": qid, "domain": "financial_reports",
            "answer_format": "multi", "question": "同样长度？",
            "options": {"A": "相同选项"},
        } for qid in ("owner-a", "owner-b")]
        ownership = {
            "version": "source_map_v1", "text_chars": 110,
            "segments": [
                {"chars": 10, "owners": ["owner-a"], "kind": "mine"},
                {"chars": 100, "owners": ["owner-b"], "kind": "mine"},
            ],
        }
        prompt = "证" * 110 + "".join(answerer._q_text(q) for q in qs)
        with mock.patch.dict("os.environ", {
                "AFAC_FIN_SHARED_EVIDENCE": "1"}):
            plan = batch._batch_prompt_allocation_plan(
                prompt, qs, evidence_ownership=ownership)
        basis = plan["weight_basis"]
        self.assertEqual(plan["prompt_weights"]["owner-a"],
                         plan["prompt_weights"]["owner-b"])
        self.assertEqual(basis["evidence_allocation_policy"],
                         "financial_tier_shared_floor_v1")
        segments = basis["evidence_ownership_segments"]
        self.assertEqual(segments[0]["source_owners"], ["owner-a"])
        self.assertEqual(segments[0]["charged_owners"],
                         ["owner-a", "owner-b"])

    def test_insurance_full_prompt_is_shared_but_source_owners_remain_auditable(self):
        qs = [
            {"qid": "short", "domain": "insurance",
             "answer_format": "multi", "question": "短题？",
             "options": {"A": "甲"}},
            {"qid": "long", "domain": "insurance",
             "answer_format": "multi", "question": "更长的保险条款问题？",
             "options": {"A": "更长的产品选项文字"}},
        ]
        ownership = {
            "version": "source_map_v1", "text_chars": 120,
            "segments": [
                {"chars": 20, "owners": ["short"], "kind": "capsule"},
                {"chars": 100, "owners": ["long"], "kind": "capsule"},
            ],
        }
        prompt = "证" * 120 + "脚手架" + "".join(
            answerer._q_text(q) for q in qs)
        with mock.patch.dict("os.environ", {"AFAC_INS_SHARED_PROMPT": "1"}):
            plan = batch._batch_prompt_allocation_plan(
                prompt, qs, evidence_ownership=ownership)
        self.assertEqual(plan["prompt_weights"]["short"],
                         plan["prompt_weights"]["long"])
        basis = plan["weight_basis"]
        self.assertTrue(basis["full_prompt_shared"])
        self.assertEqual(basis["evidence_allocation_policy"],
                         "insurance_full_prompt_shared_floor_v1")
        self.assertEqual(
            basis["evidence_ownership_segments"][0]["source_owners"],
            ["short"])
        self.assertEqual(
            basis["evidence_ownership_segments"][0]["charged_owners"],
            ["short", "long"])

    def test_contract_shared_prompt_centres_visible_option_workload(self):
        qs = [
            {"qid": "binary", "domain": "financial_contracts",
             "answer_format": "mcq", "question": "是否正确？",
             "options": {"A": "正确", "B": "错误"}},
            {"qid": "four-a", "domain": "financial_contracts",
             "answer_format": "multi", "question": "哪些正确？",
             "options": {x: x for x in "ABCD"}},
            {"qid": "four-b", "domain": "financial_contracts",
             "answer_format": "multi", "question": "哪些符合？",
             "options": {x: x for x in "ABCD"}},
        ]
        rendered = [answerer._q_text(q) for q in qs]
        prompt = "共享证据与统一协议" + "".join(rendered)
        with mock.patch.dict("os.environ", {"AFAC_FC_SHARED_PROMPT": "1"}):
            plan = batch._batch_prompt_allocation_plan(prompt, qs)
        basis = plan["weight_basis"]
        self.assertEqual(basis["option_workload_chars_per_option"], 70)
        self.assertLess(plan["prompt_weights"]["binary"],
                        plan["prompt_weights"]["four-a"])
        self.assertEqual(plan["prompt_weights"]["four-a"],
                         plan["prompt_weights"]["four-b"])
        self.assertEqual(sum(plan["prompt_weights"].values()),
                         len(prompt) * basis["prompt_weight_scale"])
        self.assertTrue(basis["prompt_weight_conserved"])

    def test_financial_groups_balance_ordinary_and_isolate_scope_collision(self):
        ordinary = [{
            "qid": f"unseen-{i}", "domain": "financial_reports",
            "answer_format": "multi", "question": "比较年度财务指标。",
            "options": {"A": "指标上升"},
            "doc_ids": [f"annual_company_{i % 4}_2025_report"],
        } for i in range(11)]
        deep = {
            "qid": "unseen-scope", "domain": "financial_reports",
            "answer_format": "multi",
            "question": "比较合并财务报表和母公司财务报表。",
            "options": {"A": "两种口径相同"},
            "doc_ids": ["annual_company_2025_report"],
        }
        with mock.patch.dict("os.environ", {
                "AFAC_HOMO_BATCH": "1", "AFAC_HOMO_MAX": "5",
                "AFAC_FIN_BATCH_MAX": "5", "AFAC_INS_CAPSULES": "0"}):
            groups = batch.group_questions(ordinary + [deep])
        solos = [[q["qid"] for q in group]
                 for group in groups if len(group) == 1]
        self.assertEqual(solos, [["unseen-scope"]])
        self.assertEqual(sorted(len(group) for group in groups
                                if len(group) > 1), [3, 4, 4])

    def test_financial_evidence_budget_subtracts_fact_mines_and_deepens_scope(self):
        ordinary = [{
            "qid": f"owner-{i}", "domain": "financial_reports",
            "answer_format": "multi", "question": "比较年度财务指标。",
            "options": {"A": "指标上升"}, "doc_ids": [f"doc-{i}"],
        } for i in range(2)]
        deep = [{
            "qid": "owner-deep", "domain": "financial_reports",
            "answer_format": "multi",
            "question": "比较合并财务报表与母公司财务报表。",
            "options": {"A": "合并口径为正，母公司口径为负"},
            "doc_ids": ["doc-deep"],
        }]
        seen = []

        def fake_gather(q, **kwargs):
            seen.append((q["qid"], kwargs["cap"], kwargs["k_opt"],
                         kwargs["k_q"]))
            return "", [], set()

        def fake_fin(q):
            return ("M" * (2000 if q["qid"] == "owner-0" else 4000))

        env = {
            "AFAC_NO_DIGEST": "1", "AFAC_SLIM4": "0",
            "AFAC_FIN_BATCH_EVIDENCE_PER_Q": "6500",
            "AFAC_FIN_SCOPE_EVIDENCE_PER_Q": "22000",
        }
        empty = mock.Mock(return_value="")
        with mock.patch.dict("os.environ", env, clear=False), \
                mock.patch.object(batch, "_doc_title", side_effect=str), \
                mock.patch.object(batch, "_use_digest", return_value=False), \
                mock.patch.object(batch, "gather_evidence",
                                  side_effect=fake_gather), \
                mock.patch.object(answerer, "financial_registry_block", empty), \
                mock.patch.object(answerer, "fin_facts_block",
                                  side_effect=fake_fin), \
                mock.patch.object(answerer, "domain_facts_block", empty), \
                mock.patch.object(answerer, "align_block", empty):
            batch._batch_evidence(ordinary)
            batch._batch_evidence(deep)

        ordinary_caps = [cap for qid, cap, _ko, _kq in seen
                         if qid.startswith("owner-") and qid != "owner-deep"]
        self.assertEqual(len(set(ordinary_caps)), 1)
        # 13,000 total visible evidence characters minus 6,000 characters of
        # owned fact mines and small title framing, then split over two owners.
        self.assertTrue(3000 <= ordinary_caps[0] <= 3500)
        deep_call = next(row for row in seen if row[0] == "owner-deep")
        # Deep scope uses a 22k visible floor and a wider source search.
        self.assertTrue(17500 <= deep_call[1] <= 18000)
        self.assertEqual(deep_call[2:], (4, 6))

    def test_research_dense_solo_raw_cap_is_source_breadth_driven(self):
        base = {
            "domain": "research", "answer_format": "multi",
            "question": "跨行业比较这些技术降本路径。",
            "options": {"A": "甲", "B": "乙"},
        }
        seen_caps = []

        def fake_gather(_q, **kwargs):
            seen_caps.append(kwargs["cap"])
            return "证据", [], set()

        env = {
            "AFAC_NO_DIGEST": "1",
            "AFAC_RESEARCH_DENSE_RAW_CAP": "6000",
            "AFAC_SLIM4": "0",
        }
        with mock.patch.dict("os.environ", env, clear=False), \
                mock.patch.object(answerer, "_doc_title",
                                  side_effect=lambda d: d), \
                mock.patch.object(answerer, "gather_evidence",
                                  side_effect=fake_gather):
            for qid in ("arbitrary-first", "unseen-other"):
                q = dict(base, qid=qid,
                         doc_ids=[f"report-{i}" for i in range(6)])
                answerer.evidence_block(q)
            # Five reports stay below the breadth threshold.  Its ordinary
            # formula is 3600 + 1000 * (5 - 2) = 6600.
            answerer.evidence_block(dict(
                base, qid="ordinary", doc_ids=[f"report-{i}"
                                                for i in range(5)]))

        self.assertEqual(seen_caps, [6000, 6000, 6600])

    def test_research_dense_solo_cap_has_a_safe_lower_bound(self):
        q = {
            "qid": "any-id", "domain": "research",
            "answer_format": "multi", "question": "比较六份材料。",
            "options": {"A": "甲"},
            "doc_ids": [f"report-{i}" for i in range(6)],
        }
        with mock.patch.dict("os.environ", {
                "AFAC_NO_DIGEST": "1",
                "AFAC_RESEARCH_DENSE_RAW_CAP": "1",
                "AFAC_SLIM4": "0",
        }, clear=False), mock.patch.object(
                answerer, "_doc_title", side_effect=lambda d: d), \
                mock.patch.object(
                answerer, "gather_evidence",
                return_value=("证据", [], set())) as gather:
            answerer.evidence_block(q)
        self.assertEqual(gather.call_args.kwargs["cap"], 1500)

    def test_research_dense_excerpt_honors_cap_and_keeps_every_source(self):
        q = {
            "qid": "arbitrary-id", "domain": "research",
            "question": "比较技术突破带来的降本和性能提升。",
            "options": {"A": "核心技术降低成本", "B": "性能提升"},
            "doc_ids": [f"d{i}" for i in range(6)],
        }
        kept = []
        for i, doc_id in enumerate(q["doc_ids"]):
            kept.extend([
                {"id": f"{doc_id}#c0", "doc_id": doc_id, "page": 1,
                 "text": ("无关背景资料" * 180)},
                {"id": f"{doc_id}#c1", "doc_id": doc_id, "page": 2,
                 "text": (("核心技术降低成本并促进性能提升" + str(i)) * 90)},
            ])
        first = answerer._dense_research_excerpt(q, kept, 6000)
        second = answerer._dense_research_excerpt(
            dict(q, qid="completely-unseen-id"), kept, 6000)

        self.assertLessEqual(len(answerer._render(first)), 6000)
        self.assertEqual({c["doc_id"] for c in first}, set(q["doc_ids"]))
        self.assertTrue(all(c["id"].endswith("#c1") for c in first))
        self.assertEqual([(c["id"], c["text"]) for c in first],
                         [(c["id"], c["text"]) for c in second])

    def test_batch_prompt_demands_concise_cited_option_reasons(self):
        self.assertIn("不超过约60个汉字", batch.BATCH_INST)
        self.assertIn("证据页码", batch.BATCH_INST)
        self.assertIn("不得省略选项", batch.BATCH_INST)

    def test_api_audit_keeps_full_request_response_and_usage(self):
        with tempfile.TemporaryDirectory() as td:
            audit = Path(td) / "api_calls.jsonl"
            qwen_client.configure_audit(audit)
            fake = SimpleNamespace(
                chat=SimpleNamespace(completions=_Completions()))
            with mock.patch.object(qwen_client, "client", return_value=fake):
                content, reasoning, usage = qwen_client.chat(
                    [{"role": "user", "content": "题目全文"}],
                    qid="q1", model="qwen-test", thinking=True,
                    thinking_budget=123, tag="r1", max_retries=1)
            qwen_client.close_audit()
            row = json.loads(audit.read_text(encoding="utf-8"))
        self.assertEqual(content, "分析完整。\n答案: AC")
        self.assertEqual(reasoning, "内部思考")
        self.assertEqual(usage["total_tokens"], 18)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["request"]["messages"][0]["content"], "题目全文")
        self.assertEqual(row["request"]["extra_body"]["thinking_budget"], 123)
        self.assertEqual(row["response"]["content"], content)
        self.assertEqual(row["usage"]["prompt_tokens"], 11)

    def test_api_audit_records_resolved_allocation_strategy_and_weights(self):
        with tempfile.TemporaryDirectory() as td:
            audit = Path(td) / "api_calls.jsonl"
            qwen_client.configure_audit(audit)
            fake = SimpleNamespace(
                chat=SimpleNamespace(completions=_Completions()))
            plan = {
                "strategy": "batch_test_v1",
                "prompt_weights": {"q1": 2, "q2": 1},
                "weight_basis": {"shared_prompt_chars": 10},
            }
            with mock.patch.object(qwen_client, "client", return_value=fake):
                qwen_client.chat(
                    [{"role": "user", "content": "共享题面"}],
                    qid="_batch", model="qwen-test", tag="b1",
                    max_retries=1, allocation_qids=["q1", "q2"],
                    allocation_plan=plan,
                    allocation_resolver=lambda _content: {
                        "completion_weights": {"q1": 1, "q2": 3},
                        "weight_basis": {
                            "visible_answer_block_chars": {"q1": 1,
                                                           "q2": 3}},
                    })
            qwen_client.close_audit()
            row = json.loads(audit.read_text(encoding="utf-8"))
        call = qwen_client.LEDGER.calls[-1]
        self.assertEqual(row["allocation"]["strategy"], "batch_test_v1")
        self.assertEqual(row["allocation"]["prompt_weights"],
                         {"q1": 2, "q2": 1})
        self.assertEqual(row["allocation"]["visible_completion_weights"],
                         {"q1": 1, "q2": 3})
        self.assertEqual(call["allocation_strategy"], "batch_test_v1")
        self.assertEqual(call["allocation_weights"]["prompt"],
                         {"q1": 2, "q2": 1})
        self.assertEqual(sum(v[0] for v in call["allocated_usage"].values()),
                         row["usage"]["prompt_tokens"])
        self.assertEqual(sum(v[1] for v in call["allocated_usage"].values()),
                         row["usage"]["completion_tokens"])

    def test_allocation_resolver_failure_never_retries_paid_api_call(self):
        create = mock.Mock(side_effect=_Completions().create)
        fake = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create)))

        def broken_resolver(_content):
            raise ValueError("bad block framing")

        with mock.patch.object(qwen_client, "client", return_value=fake):
            qwen_client.chat(
                [{"role": "user", "content": "共享题面"}],
                qid="_batch", model="qwen-test", max_retries=3,
                allocation_qids=["q1", "q2"],
                allocation_plan={
                    "strategy": "batch_test_v1",
                    "prompt_weights": {"q1": 2, "q2": 1},
                },
                allocation_resolver=broken_resolver)
        self.assertEqual(create.call_count, 1)
        self.assertEqual(qwen_client.LEDGER.totals(), (11, 7, 18))
        allocation = qwen_client.LEDGER.calls[-1]["allocation"]
        self.assertEqual(allocation["visible_completion_weights"],
                         {"q1": 1, "q2": 1})
        self.assertEqual(
            allocation["weight_basis"]["allocation_resolver_error"]["type"],
            "ValueError")

    def test_batch_parser_keeps_each_visible_question_block(self):
        qs = [
            {"qid": "q1", "answer_format": "multi"},
            {"qid": "q2", "answer_format": "mcq"},
        ]
        text = ("批前说明\n【第1题 答案块】\n分析: 甲\n答案: A, C\n"
                "【第2题 答案块】\n分析: 乙\n答案: B")
        got = batch._parse_batch_details(text, qs)
        self.assertEqual(got["q1"]["answer"], "AC")
        self.assertEqual(got["q2"]["answer"], "B")
        self.assertIn("分析: 甲", got["q1"]["content"])
        self.assertNotIn("分析: 乙", got["q1"]["content"])
        self.assertIn("分析: 乙", got["q2"]["content"])

    def test_batch_retrieval_stays_with_each_questions_documents(self):
        qs = [
            {"qid": "q1", "domain": "research", "doc_ids": ["doc_a"],
             "question": "甲材料说法？", "options": {"A": "甲"},
             "answer_format": "multi"},
            {"qid": "q2", "domain": "research", "doc_ids": ["doc_b"],
             "question": "乙材料说法？", "options": {"A": "乙"},
             "answer_format": "multi"},
        ]
        seen = []

        def fake_gather(q, **_kwargs):
            seen.append(list(q["doc_ids"]))
            doc = q["doc_ids"][0]
            chunk = {"id": f"{doc}#c1", "doc_id": doc,
                     "page": 1, "text": f"{doc} evidence"}
            return chunk["text"], [chunk], {chunk["id"]}

        with mock.patch.object(batch, "_doc_title", side_effect=lambda d: d), \
                mock.patch.object(batch, "_use_digest", return_value=False), \
                mock.patch.object(batch, "gather_evidence",
                                  side_effect=fake_gather):
            _ev, ids, _low = batch._batch_evidence(qs)
        self.assertEqual(seen, [["doc_a"], ["doc_b"]])
        self.assertEqual(set(ids), {"doc_a#c1", "doc_b#c1"})

    def test_option_local_anchor_protects_named_source_hit(self):
        wrong = {"id": "doc_a#c1", "doc_id": "doc_a", "page": 1,
                 "text": "通用募集说明书样板文字"}
        exact = {"id": "doc_b#c9", "doc_id": "doc_b", "page": 9,
                 "text": "本川智能完全达产前最高比例为3.49%。"}

        class FakeIndex:
            def __init__(self, doc_id):
                self.doc_id = doc_id

            def search(self, _query, k=1):
                chunk = exact if self.doc_id == "doc_b" else wrong
                return [(chunk, 20.0 if self.doc_id == "doc_b" else 5.0)][:k]

        q = {"qid": "shape_only", "domain": "financial_contracts",
             "doc_ids": ["doc_a", "doc_b"],
             "question": "以下汇总披露哪些正确？",
             "options": {"A": "本川智能完全达产前最高比例为3.49%"}}
        with mock.patch.object(answerer.retrieval, "search_docs",
                               return_value=[(wrong, 1.0)]), \
                mock.patch.object(answerer.retrieval, "doc_index",
                                  side_effect=lambda d: FakeIndex(d)), \
                mock.patch.object(answerer, "_doc_title",
                                  side_effect=lambda d: {
                                      "doc_a": "安克创新募集说明书",
                                      "doc_b": "本川智能募集说明书"}[d]), \
                mock.patch.object(answerer, "ENT_PROBE", False):
            _ev, kept, protected = answerer.gather_evidence(
                q, k_opt=1, k_q=1, cap=200)
        self.assertIn("doc_b#c9", {c["id"] for c in kept})
        self.assertIn("doc_b#c9", protected)

    def test_reasoning_prefers_trace_supporting_final_answer(self):
        traces = [
            {"stage": "r1", "content": "初判\n答案: A", "answer": "A"},
            {"stage": "r2", "content": "复核\n答案: AC", "answer": "AC"},
        ]
        text, stage = answerer.select_reasoning("AC", traces, "multi")
        self.assertEqual(stage, "r2")
        self.assertEqual(text, "复核\n答案: AC")

    def test_reasoning_discloses_deterministic_ensemble(self):
        traces = [
            {"stage": "vote1", "content": "候选一\n答案: A", "answer": "A"},
            {"stage": "vote2", "content": "候选二\n答案: BC", "answer": "BC"},
        ]
        text, stage = answerer.select_reasoning("AB", traces, "multi")
        self.assertEqual(stage, "ensemble")
        self.assertIn("候选一", text)
        self.assertIn("候选二", text)
        self.assertIn("确定性聚合结果：答案 AB", text)

    def test_compact_judge_routes_by_semantics_not_qid(self):
        base = {
            "domain": "insurance",
            "question": "哪些产品明确规定按同比增幅计算给付比例？",
            "options": {"A": "甲产品", "B": "乙产品"},
        }
        with mock.patch.dict("os.environ", {"AFAC_COMPACT_JUDGE": "1"}):
            a = answerer.judge_std_for(dict(base, qid="arbitrary_001"))
            b = answerer.judge_std_for(dict(base, qid="totally_other"))
        self.assertEqual(a, b)
        self.assertIn("【有无题】", a)
        self.assertIn("【口径题硬约束】", a)
        self.assertIn("【保险】", a)
        self.assertLess(len(a), len(answerer.JUDGE_STD) * 0.6)

    def test_compact_judge_requires_cross_object_evidence_for_extrema(self):
        q = {
            "domain": "research",
            "question": "四个行业的品牌化挑战，哪些判断符合材料？",
            "options": {
                "A": "甲行业难度最大，因为消费者认知较低",
                "B": "乙行业难度最小，因为已有销售渠道",
                "C": "丙行业需要同时改善内容与品质",
            },
        }
        with mock.patch.dict("os.environ", {"AFAC_COMPACT_JUDGE": "1"}):
            prompt = answerer.judge_std_for(q)
        self.assertIn("【极值比较】", prompt)
        self.assertIn("单个对象存在困难/优势", prompt)

    def test_compact_judge_treats_explicit_stem_property_as_given(self):
        q = {
            "domain": "research",
            "question": "三个领域都出现了通过技术突破实现结构性降本的案例，哪些分析正确？",
            "options": {
                "A": "都形成成本优势",
                "B": "降本的同时都带来产品性能提升",
            },
        }
        with mock.patch.dict("os.environ", {"AFAC_COMPACT_JUDGE": "1"}):
            prompt = answerer.judge_std_for(q)
        self.assertIn("【题干既定前提】", prompt)
        self.assertIn("只逐案例核验新增效果", prompt)

    def test_compact_judge_requires_evidence_for_concept_reclassification(self):
        q = {
            "domain": "research",
            "question": "不同机构的实践中哪些判断符合材料？",
            "options": {
                "A": "增配长久期债券实质上属于杠杆思维",
                "B": "原文明确建议降低波动后再提升杠杆",
            },
        }
        with mock.patch.dict("os.environ", {"AFAC_COMPACT_JUDGE": "1"}):
            prompt = answerer.judge_std_for(q)
        self.assertIn("【概念归类硬约束】", prompt)
        self.assertIn("ANALOGY一律不选", prompt)
        self.assertIn("久期配置不自动等同于加杠杆", prompt)

    def test_qualitative_extreme_is_removed_without_cross_object_evidence(self):
        q = {
            "domain": "research", "answer_format": "multi",
            "question": "四家分属不同行业的企业面临不同挑战。",
            "options": {
                "A": "甲行业品牌化难度最大，因为认知较低",
                "B": "乙行业需要稳定交付建立认可",
            },
        }
        got, note = answerer.apply_structural_evidence_constraints(
            "AB", q, "原文仅说明甲行业消费者认知较低；乙行业重视稳定交付。")
        self.assertEqual(got, "B")
        self.assertIn("极值比较", note)

    def test_explicit_extreme_evidence_preserves_option(self):
        q = {
            "domain": "research", "answer_format": "multi",
            "question": "三家公司中哪些判断正确？",
            "options": {"A": "甲公司的转型难度最大", "B": "乙公司稳定"},
        }
        got, note = answerer.apply_structural_evidence_constraints(
            "AB", q, "横向调查结论：甲公司的转型难度最大。")
        self.assertEqual(got, "AB")
        self.assertEqual(note, "")

    def test_structural_constraint_reasoning_is_visible_qwen_response(self):
        q = {
            "qid": "opaque_research_item",
            "domain": "research", "answer_format": "multi",
            "question": "四家企业的品牌化挑战，哪些判断符合材料？",
            "options": {
                "A": "甲企业品牌化难度最大",
                "B": "乙企业需要稳定交付建立认可",
            },
        }
        response = (
            "复核：材料只说明甲存在困难，没有同口径比较支持‘难度最大’；"
            "乙的稳定交付判断保留。\n答案: B")
        with mock.patch.object(
                answerer, "chat", return_value=(response, "", {})) as called:
            reasoning, confirmed = \
                answerer.confirm_structural_evidence_constraint(
                    q, "先前Qwen判断为AB。", "B",
                    "选项A缺少同口径横向极值证据。")
        self.assertEqual(reasoning, response)
        self.assertEqual(confirmed, "B")
        self.assertEqual(called.call_args.kwargs["tag"],
                         "evidence_constraint")
        self.assertFalse(called.call_args.kwargs["thinking"])

    def test_compact_judge_distinguishes_summary_from_universal_claim(self):
        q = {
            "domain": "financial_contracts", "answer_format": "multi",
            "question": "关于违约事项，哪些说法正确？",
            "options": {
                "A": "发行人发生违约时有90个自然日宽限期",
                "B": "任何违约情形一律适用90个自然日宽限期",
            },
        }
        with mock.patch.dict("os.environ", {"AFAC_COMPACT_JUDGE": "1"}):
            prompt = answerer.judge_std_for(q)
        self.assertIn("【概括范围】", prompt)
        self.assertIn("任何/全部/一律", prompt)

    def test_collect_reasonings_uses_latest_current_run_record(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "run.jsonl"
            rows = [
                {"qid": "q1", "reasoning": "第一次解释文本足够长"},
                {"qid": "q2", "c1": "第二题可见输出也足够长"},
                {"qid": "q1", "reasoning": "重试后的最终解释文本"},
            ]
            p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                 for r in rows), encoding="utf-8")
            got = collect_reasonings(p, ["q1", "q2"])
        self.assertEqual(got["q1"], "重试后的最终解释文本")
        self.assertEqual(got["q2"], "第二题可见输出也足够长")

    def test_locked_writer_close_is_idempotent(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "log.jsonl"
            writer = LockedJsonlWriter(p, "w")
            writer.write('{"ok": true}\n')
            writer.close()
            writer.close()
            with self.assertRaises(ValueError):
                writer.write("late\n")
            self.assertEqual(p.read_text(encoding="utf-8"),
                             '{"ok": true}\n')


if __name__ == "__main__":
    unittest.main()
