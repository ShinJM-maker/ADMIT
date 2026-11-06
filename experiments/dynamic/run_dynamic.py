#!/usr/bin/env python3
"""Dynamic M3DocVQA comparison (Table 1, Sec. 6.2-6.3): ReAct, PlanRAG and AutoGen with typed memory.

Policies (the paper's names; runtime labels in parentheses):
  fail_open      (cavec_paper)          check each write; after failed repair, store and return the pre-repair value
  hard_refusal   (cavec_h1_hard_refuse) check each write; after failed repair, refuse on both surfaces
  admit          (cavec_mfc)            ADMIT: MFC/OFO; after failed repair, NO_COMMIT in memory, answer UNVERIFIED
  final_support  (final_support_retry)  run Vanilla, then check and repair only the final output
  gate_only      component: lexical gate without RepairRead
  gate_repair    component: lexical gate + one RepairRead (= ADMIT's admission rule)
Every policy uses the strict lexical gate (no yes/no carve-out). Reader: Qwen3-VL-8B-Instruct, greedy decoding,
text-only, 16,384-token context, 14,000-token evidence budget, BM25 top-10 section cards.

Usage: python experiments/dynamic/run_dynamic.py --shard_id 0 --num_shards 1 [--policies admit fail_open]
Inputs: ADMIT_INDEX (M3DocVQA/chunks.jsonl, docs.jsonl, sec_index_md/), ADMIT_DATA (M3DocVQA/val.json),
ADMIT_READER_MODEL. Output: work/dynamic/shard<k>.jsonl, one compact trajectory per (policy, qid, agent).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "harness"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agents as p30  # noqa: E402
from final_support_contract import POLICY as FINAL_SUPPORT_MEMORY_POLICY  # noqa: E402

POLICIES = ("fail_open", "hard_refusal", "admit", "final_support", "gate_only", "gate_repair")
AGENTS = ("react", "planrag", "autogen")


def strict_cavec_admit(commit_value, evidence_pack_text, question=""):
    """Strict lexical gate used by every final policy: refusals are status records, otherwise substring support."""
    if p30.is_refusal(commit_value):
        return True, "refusal_recorded"
    if p30.support_in_pack(commit_value, evidence_pack_text):
        return True, "S_str_positive"
    return False, "S_str_zero"


def gate_only_hook(reader, candidate_value, question, sec_cands, pack_text, baseline_value=None,
                   top_k_cards=10, cached_repair=None):
    """Component arm: the lexical gate alone. A failed check writes NO_COMMIT and shows the original answer."""
    log = {"candidate": candidate_value, "n_repair_calls": 0, "repair_value": None}
    admit, reason = p30.cavec_admit(candidate_value, pack_text, question=question)
    log["admit_reason"] = reason
    if admit:
        log.update(final_source="candidate", memory_source="candidate")
        return candidate_value, candidate_value, log
    if baseline_value is not None:
        log.update(final_source="fallback_baseline_unsupported", memory_source="quarantine_no_commit")
        return p30.NO_COMMIT_SENTINEL, baseline_value, log
    refusal = "I do not have enough information to answer."
    log.update(final_source="fallback_refusal", memory_source="fallback_refusal")
    return refusal, refusal, log


def agent_function(agent):
    return {"react": (p30.run_react, {"max_turns": 5}),
            "planrag": (p30.run_planrag, {}),
            "autogen": (p30.run_autogen, {"max_turns": 5})}[agent]


def memory_records(result, agent):
    if agent == "react":
        rows = [r["memory_record"] for r in result.get("scratchpad", [])]
    elif agent == "planrag":
        rows = [r["memory_record"] for r in result.get("sub_logs", [])]
    else:
        rows = [r["_memory_record"] for r in result.get("conversation", []) if r.get("role") == "reader_commit"]
    return rows


def run_policy(reader, policy, agent, question, bm25, cards, chunk_text):
    fn, kw = agent_function(agent)
    if policy == "final_support":
        vanilla = fn(reader, question, bm25, cards, chunk_text, top_k=10, with_cavec=False, cavec_fail_closed=False,
                     with_router=False, with_cavec_mfc=False, **kw)
        _, result = p30._post_hoc_arms(reader, vanilla, question, bm25, cards, chunk_text, 10)
        result["base_memory_records"] = memory_records(vanilla, agent)
        result["n_reader_calls"] = vanilla["n_reader_calls"]
        result["n_repair_calls"] = result.get("n_repair_calls", 0)
        result["final_memory_record"] = dict(vanilla["final_memory_record"])  # final-only repair keeps Vanilla memory
        result["terminal_memory_contract"] = FINAL_SUPPORT_MEMORY_POLICY
        return result
    if policy in ("gate_only", "gate_repair"):
        original = p30.cavec_mfc_hook
        p30.cavec_mfc_hook = gate_only_hook if policy == "gate_only" else original
        try:
            return fn(reader, question, bm25, cards, chunk_text, top_k=10, with_cavec=False, cavec_fail_closed=False,
                      with_router=False, with_cavec_mfc=True, **kw)
        finally:
            p30.cavec_mfc_hook = original
    config = {"fail_open": dict(with_cavec=True, cavec_fail_closed=False, with_cavec_mfc=False),
              "hard_refusal": dict(with_cavec=True, cavec_fail_closed=True, with_cavec_mfc=False),
              "admit": dict(with_cavec=False, cavec_fail_closed=False, with_cavec_mfc=True)}[policy]
    return fn(reader, question, bm25, cards, chunk_text, top_k=10, with_router=False, **config, **kw)


def compact(result):
    result.pop("final_pack_text", None)
    for key in ("scratchpad", "sub_logs"):
        for row in result.get(key, []):
            row.pop("pack_text", None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=POLICIES)
    ap.add_argument("--agents", nargs="+", default=list(AGENTS), choices=AGENTS)
    ap.add_argument("--qids", type=Path, default=REPO / "configs/dynamic_qids.json")
    ap.add_argument("--out_dir", type=Path, default=Path(os.environ.get("ADMIT_WORK", REPO / "work")) / "dynamic")
    ap.add_argument("--gpu_mem_util", type=float, default=0.82)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    p30.cavec_admit = strict_cavec_admit
    qids = json.loads(a.qids.read_text())
    tasks = [(pol, q, ag) for q in qids for ag in a.agents for pol in a.policies]
    tasks = [t for i, t in enumerate(tasks) if i % a.num_shards == a.shard_id]
    if a.limit:
        tasks = tasks[:a.limit]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"shard{a.shard_id}.jsonl"
    done = {(r["policy"], r["qid"], r["agent"]) for r in map(json.loads, out.open())} if out.exists() else set()
    pending = [t for t in tasks if t not in done]
    print(f"shard {a.shard_id}/{a.num_shards}: {len(pending)} pending of {len(tasks)}", flush=True)
    if not pending:
        return
    qid2gold, qid2question = p30.load_qid_meta()
    chunk_text = p30.load_chunk_text()
    bm25, cards = p30.load_bm25_and_cards()
    model = os.environ.get("ADMIT_READER_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    reader = p30.ReaderVLLM(model, max_model_len=16384, token_budget=14000, gpu_mem_util=a.gpu_mem_util)
    with out.open("a", encoding="utf-8") as fh:
        for k, (policy, qid, agent) in enumerate(pending, 1):
            t0 = time.time()
            res = run_policy(reader, policy, agent, qid2question[qid], bm25, cards, chunk_text)
            res["em"] = p30.substr_em(res["final_committed_answer"], qid2gold[qid])
            res["calls_q"] = res.get("n_reader_calls", 0) + res.get("n_repair_calls", 0)
            compact(res)
            fh.write(json.dumps({"policy": policy, "qid": qid, "agent": agent, "gold": qid2gold[qid],
                                 "result": res, "wall_s": time.time() - t0}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{k}/{len(pending)}] {policy}:{agent}:{qid} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
