#!/usr/bin/env python3
"""Portability refusal-repair runs (App. B.15): terminal replay and typed PlanRAG dataset transfer.

Terminal replay: every row of experiments/portability/run_terminal.py whose ADMIT terminal write is an initial
refusal (STATUS/REFUSAL) gets one RepairRead with the contract's repair prompt on the same frozen packet; the
repaired value is admitted and returned only if it is a non-refusal that passes the strict gate.
Dataset transfer: the qids in configs/refusal_repair_pairs.json["datasets"] (their ADMIT trajectory records an
initial refusal) are run under arms admit and admit_rr with a sequential reader (same prompts, greedy decoding
and seed as experiments/portability/run_datasets.py).

Inputs under $ADMIT_WORK/portability: results/terminal.shard*.jsonl (run_terminal.py output),
<dataset>_stage2.jsonl and <dataset>_selected.json (as for run_datasets.py).
Usage: python experiments/refusal_repair/run_portability_rr.py --shard_id 0 --num_shards 4
Output: $ADMIT_WORK/refusal_repair/portability/{terminal,dude,tatqa,mmlongbench}.shard<k>.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
P = REPO / "experiments" / "portability"
sys.path.insert(0, str(P)); sys.path.insert(0, str(P / "snapshots"))
import contract as c  # noqa: E402
from run_terminal import em  # noqa: E402
import run_dude_dynamic as base  # noqa: E402
import run_datasets as rd  # noqa: E402

W = Path(os.environ.get("ADMIT_WORK", REPO / "work")) / "portability"
TL = threading.local()


class SeqReader(rd.Reader):
    """One request at a time; same prompts, greedy decoding and seed as the threaded broker."""

    def __init__(self, runner, tokenizer, llm, SamplingParams):
        self.runner = runner; self.tokenizer = tokenizer; self.llm = llm; self.SP = SamplingParams
        self.token_budget = 12000; self.max_images_total = 0
        self._chunks_idx = {}; self._cards_idx = {}; self._docs_idx = {}; self._parent_to_cards = {}

    def _gen(self, prompts, max_tokens):
        outs = self.llm.generate([{"prompt": p} for p in prompts],
                                 self.SP(temperature=0, top_p=1, max_tokens=max_tokens, seed=20260928), use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]

    def generate_batch(self, prompts):
        return self._gen(prompts, 64)

    def reasoning_call(self, prompt_text, max_new_tokens=96):
        return self._gen([c.chat_prompt(self.tokenizer, prompt_text)], max_new_tokens)[0]


def terminal_refusals():
    rows = []
    for fn in sorted(glob.glob(str(W / "results/terminal.shard*.jsonl"))):
        for line in open(fn):
            r = json.loads(line)
            if (r["arms"]["admit"].get("memory_record") or {}).get("status") == "REFUSAL":
                rows.append({k: r[k] for k in ("qid", "pipeline", "question", "gold", "packet_text", "candidate")})
    return sorted(rows, key=lambda r: (r["pipeline"], r["qid"]))


def terminal(llm, SamplingParams, rows):
    tok = llm.get_tokenizer()
    prompts = [c.chat_prompt(tok, c.REPAIR_PROMPT_HEAD.format(question=r["question"]) + r["packet_text"] + c.PROMPT_TAIL)
               for r in rows]
    outs = llm.generate([{"prompt": p} for p in prompts],
                        SamplingParams(temperature=0., top_p=1., max_tokens=64, seed=20260928), use_tqdm=False)
    res = []
    for r, o in zip(rows, outs):
        rep = c.clean_answer(o.outputs[0].text.strip())
        ok = (not c.is_refusal(rep)) and bool(c.support_in_pack(rep, r["packet_text"]))
        res.append({"qid": r["qid"], "pipeline": r["pipeline"], "repair": rep, "admitted": int(ok),
                    "em_admit": em(r["candidate"], r["gold"]), "em_admit_rr": em(rep, r["gold"]) if ok else 0})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard_id", type=int, required=True); ap.add_argument("--num_shards", type=int, required=True)
    ap.add_argument("--gpu_mem_util", type=float, default=0.82)
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(model=c.MODEL, trust_remote_code=True, dtype="bfloat16", gpu_memory_utilization=a.gpu_mem_util,
              max_model_len=16384, limit_mm_per_prompt={"image": 1}, enforce_eager=True, tensor_parallel_size=1,
              max_num_batched_tokens=16384, seed=20260928)
    out = Path(os.environ.get("ADMIT_WORK", REPO / "work")) / "refusal_repair" / "portability"
    out.mkdir(parents=True, exist_ok=True)
    rows = [r for i, r in enumerate(terminal_refusals()) if i % a.num_shards == a.shard_id]
    dest = out / f"terminal.shard{a.shard_id}.jsonl"
    if not dest.exists():
        with dest.open("w") as f:
            for k in range(0, len(rows), 64):
                for x in terminal(llm, SamplingParams, rows[k:k + 64]):
                    f.write(json.dumps(x) + "\n")
                print(f"terminal {min(k + 64, len(rows))}/{len(rows)}", flush=True)

    runner = rd.load("strict_dataset_runner", REPO / "harness/agents.py")
    runner.cavec_admit = c.strict_cavec_admit
    runner.build_evidence_pack = base.inline_pack
    original = runner.cavec_mfc_hook

    def hook(reader, candidate_value, question, sec_cands, pack_text, baseline_value=None, top_k_cards=10,
             cached_repair=None):
        if getattr(TL, "arm", "admit") != "admit_rr" or not runner.is_refusal(candidate_value):
            return original(reader, candidate_value, question, sec_cands, pack_text, baseline_value=baseline_value,
                            top_k_cards=top_k_cards, cached_repair=cached_repair)
        rep, n = runner.repair_read(reader, question, sec_cands, top_k_cards)
        log = {"candidate": candidate_value, "n_repair_calls": n, "admit_reason": "refusal_repair", "repair_value": rep}
        if not runner.is_refusal(rep) and c.support_in_pack(rep, pack_text):
            log.update(final_source="repair_after_refusal", memory_source="repair_after_refusal")
            return rep, rep, log
        log.update(final_source="candidate_refusal", memory_source="candidate_refusal")
        return candidate_value, candidate_value, log

    runner.cavec_mfc_hook = hook
    tokenizer = llm.get_tokenizer()
    affected = json.loads((REPO / "configs/refusal_repair_pairs.json").read_text())["datasets"]
    for dataset in ("dude", "tatqa", "mmlongbench"):
        keep = set(affected[dataset])
        chosen = [r for r in json.loads((W / f"{dataset}_selected.json").read_text()) if r["qid"] in keep]
        chosen = [r for i, r in enumerate(chosen) if i % a.num_shards == a.shard_id]
        dest = out / f"{dataset}.shard{a.shard_id}.jsonl"
        if dest.exists():
            continue
        source = base.read_jsonl(W / f"{dataset}_stage2.jsonl")
        cards = base.cards_from_rows(source); card_by_id = {x["card_id"]: x for x in cards}
        bm25 = base.SimpleBM25([x["bm25_text"] for x in cards], runner.tok)
        reader = SeqReader(runner, tokenizer, llm, SamplingParams)
        with dest.open("w") as fh:
            for k, row in enumerate(chosen, 1):
                out_row = {"qid": row["qid"], "dataset": dataset, "gold": row["gold"], "arms": {}}
                for arm in ("admit", "admit_rr"):
                    TL.arm = arm
                    result = runner.run_planrag(reader, row["question"], bm25, cards, {}, top_k=10, with_cavec=False,
                                                cavec_fail_closed=False, with_cavec_mfc=True)
                    m = rd.annotate(runner, reader, result, card_by_id, "admit", row["question"])
                    m["raw_em"] = em(result["final_committed_answer"], row["gold"])
                    out_row["arms"][arm] = {"metrics": {x: v for x, v in m.items() if x != "writes"},
                                            "final_committed_answer": result["final_committed_answer"],
                                            "final_memory_record": result["final_memory_record"]}
                fh.write(json.dumps(out_row, ensure_ascii=False) + "\n"); fh.flush()
                print(f"{dataset} {k}/{len(chosen)}", flush=True)


if __name__ == "__main__":
    main()
