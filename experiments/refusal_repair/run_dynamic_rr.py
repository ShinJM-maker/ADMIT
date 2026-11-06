#!/usr/bin/env python3
"""Dynamic refusal-repair runs (Table 3b, App. B.15): ADMIT and fail-open, each with and without refusal repair.

Refusal repair gives an initial refusal one RepairRead on the same question and bundle. A repaired value that is
a non-refusal and passes the strict lexical gate is stored and returned; otherwise the refusal stays a STATUS
record (under fail-open the pre-repair value, here the refusal, is kept). Only the (qid, agent) pairs whose
trajectory records an initial refusal are run (configs/refusal_repair_pairs.json); every other pair is
identical under both arms by construction.

Same harness, reader, inputs and strict gate as experiments/dynamic/run_dynamic.py.

Usage: python experiments/refusal_repair/run_dynamic_rr.py --family admit --shard_id 0 --num_shards 4
Output: $ADMIT_WORK/refusal_repair/<family>_shard<k>.jsonl (one compact trajectory per arm, qid, agent).
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
sys.path.insert(0, str(REPO / "experiments" / "dynamic"))
import agents as p30  # noqa: E402
from run_dynamic import agent_function, compact, strict_cavec_admit  # noqa: E402

ARMS = {"admit": ("admit", "admit_rr"), "failopen": ("fail_open", "fail_open_rr")}
ORIGINAL_MFC_HOOK = p30.cavec_mfc_hook
ORIGINAL_FO_HOOK = p30.cavec_hook


def _repair_refusal(reader, candidate_value, question, sec_cands, pack_text, top_k_cards):
    repair_val, n_calls = p30.repair_read(reader, question, sec_cands, top_k_cards)
    log = {"candidate": candidate_value, "n_repair_calls": n_calls, "admit_reason": "refusal_repair",
           "repair_value": repair_val}
    ok = not p30.is_refusal(repair_val) and p30.support_in_pack(repair_val, pack_text)
    log["final_source"] = "repair_after_refusal" if ok else "candidate_refusal"
    return (repair_val if ok else candidate_value), log


def admit_rr_hook(reader, candidate_value, question, sec_cands, pack_text, baseline_value=None, top_k_cards=10,
                  cached_repair=None):
    if not p30.is_refusal(candidate_value):
        return ORIGINAL_MFC_HOOK(reader, candidate_value, question, sec_cands, pack_text,
                                 baseline_value=baseline_value, top_k_cards=top_k_cards, cached_repair=cached_repair)
    value, log = _repair_refusal(reader, candidate_value, question, sec_cands, pack_text, top_k_cards)
    log["memory_source"] = log["final_source"]
    return value, value, log


def failopen_rr_hook(reader, candidate_value, question, sec_cands, pack_text, baseline_value=None, top_k_cards=10,
                     fail_closed=True, cached_repair=None):
    if not p30.is_refusal(candidate_value):
        return ORIGINAL_FO_HOOK(reader, candidate_value, question, sec_cands, pack_text, baseline_value=baseline_value,
                                top_k_cards=top_k_cards, fail_closed=fail_closed, cached_repair=cached_repair)
    return _repair_refusal(reader, candidate_value, question, sec_cands, pack_text, top_k_cards)


def run(reader, family, arm, agent, question, bm25, cards, chunk_text):
    fn, kw = agent_function(agent)
    if family == "failopen":
        p30.cavec_hook = failopen_rr_hook if arm.endswith("_rr") else ORIGINAL_FO_HOOK
        try:
            return fn(reader, question, bm25, cards, chunk_text, top_k=10, with_cavec=True, cavec_fail_closed=False,
                      with_router=False, with_cavec_mfc=False, **kw)
        finally:
            p30.cavec_hook = ORIGINAL_FO_HOOK
    p30.cavec_mfc_hook = admit_rr_hook if arm.endswith("_rr") else ORIGINAL_MFC_HOOK
    try:
        return fn(reader, question, bm25, cards, chunk_text, top_k=10, with_cavec=False, cavec_fail_closed=False,
                  with_router=False, with_cavec_mfc=True, **kw)
    finally:
        p30.cavec_mfc_hook = ORIGINAL_MFC_HOOK


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=tuple(ARMS), default="admit")
    ap.add_argument("--shard_id", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--pairs", type=Path, default=REPO / "configs/refusal_repair_pairs.json")
    ap.add_argument("--out_dir", type=Path, default=Path(os.environ.get("ADMIT_WORK", REPO / "work")) / "refusal_repair")
    ap.add_argument("--gpu_mem_util", type=float, default=0.82)
    a = ap.parse_args()

    p30.cavec_admit = strict_cavec_admit
    pairs = json.loads(a.pairs.read_text())[f"dynamic_{a.family}"]
    tasks = [(arm, p["qid"], p["agent"]) for p in pairs for arm in ARMS[a.family]]
    tasks = [t for i, t in enumerate(tasks) if i % a.num_shards == a.shard_id]
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / f"{a.family}_shard{a.shard_id}.jsonl"
    done = {(r["arm"], r["qid"], r["agent"]) for r in map(json.loads, out.open())} if out.exists() else set()
    pending = [t for t in tasks if t not in done]
    print(f"{a.family} shard {a.shard_id}/{a.num_shards}: {len(pending)} pending of {len(tasks)}", flush=True)
    if not pending:
        return
    qid2gold, qid2question = p30.load_qid_meta()
    chunk_text = p30.load_chunk_text()
    bm25, cards = p30.load_bm25_and_cards()
    model = os.environ.get("ADMIT_READER_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    reader = p30.ReaderVLLM(model, max_model_len=16384, token_budget=14000, gpu_mem_util=a.gpu_mem_util)
    with out.open("a", encoding="utf-8") as fh:
        for k, (arm, qid, agent) in enumerate(pending, 1):
            t0 = time.time()
            res = run(reader, a.family, arm, agent, qid2question[qid], bm25, cards, chunk_text)
            res["em"] = p30.substr_em(res["final_committed_answer"], qid2gold[qid])
            res["calls_q"] = res.get("n_reader_calls", 0) + res.get("n_repair_calls", 0)
            compact(res)
            fh.write(json.dumps({"arm": arm, "qid": qid, "agent": agent, "gold": qid2gold[qid], "result": res,
                                 "wall_s": time.time() - t0}, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[{k}/{len(pending)}] {arm}:{agent}:{qid} {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
