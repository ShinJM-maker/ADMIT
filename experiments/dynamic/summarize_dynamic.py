#!/usr/bin/env python3
"""Summarize run_dynamic.py outputs: refusal-aware substring EM (code/answer_metrics.py), reader calls/q, contamination and NO_COMMIT rate per policy.

Contamination: a trajectory is contaminated when any reusable record (intermediate writes and the terminal record)
is a FACT whose value is not a normalized substring of the exact packet the reader saw. EM differences use a paired
bootstrap over qids (10,000 resamples, seed 0), averaging the three agents within each qid.

Usage: python experiments/dynamic/summarize_dynamic.py [--in_dir work/dynamic]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
import sys  # noqa: E402
sys.path.insert(0, str(REPO / "code"))
from answer_metrics import substring_em  # noqa: E402  refusal-aware substring EM used in the paper
AGENTS = ("react", "planrag", "autogen")


def records(result, agent):
    if "base_memory_records" in result:
        rows = list(result["base_memory_records"])
    elif agent == "react":
        rows = [r.get("memory_record", {}) for r in result.get("scratchpad", [])]
    elif agent == "planrag":
        rows = [r.get("memory_record", {}) for r in result.get("sub_logs", [])]
    else:
        rows = [r.get("_memory_record", {}) for r in result.get("conversation", []) if r.get("role") == "reader_commit"]
    if result.get("final_memory_record"):
        rows.append(result["final_memory_record"])
    return rows


def contaminated(result, agent):
    return int(any(r.get("record_type") == "FACT" and r.get("lexically_supported") is False
                   for r in records(result, agent)))


def no_commit(result, agent):
    return int(any(r.get("status") == "NO_COMMIT" for r in records(result, agent)))


def paired_boot(a, b, B=10000, seed=0):
    per = defaultdict(list)
    for k in a.keys() & b.keys():
        per[k[0]].append(a[k] - b[k])
    v = [sum(x) / len(x) for x in per.values()]
    rng = random.Random(seed)
    bs = sorted(sum(v[rng.randrange(len(v))] for _ in v) / len(v) for _ in range(B))
    return [round(100 * sum(v) / len(v), 2), round(100 * bs[int(.025 * B)], 2), round(100 * bs[int(.975 * B)], 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, default=Path(os.environ.get("ADMIT_WORK", REPO / "work")) / "dynamic")
    a = ap.parse_args()
    rows = {}
    for f in sorted(a.in_dir.glob("shard*.jsonl")):
        for r in map(json.loads, f.open()):
            res = r["result"]
            res["em"] = substring_em(res["final_committed_answer"], r["gold"])
            rows[(r["policy"], r["qid"], r["agent"])] = res
    policies = sorted({k[0] for k in rows})
    out, em = {}, {}
    for pol in policies:
        keys = [k for k in rows if k[0] == pol]
        em[pol] = {(k[1], k[2]): rows[k]["em"] for k in keys}
        out[pol] = {"n": len(keys)}
        for agent in (*AGENTS, "pooled"):
            ks = [k for k in keys if agent == "pooled" or k[2] == agent]
            if not ks:
                continue
            n = len(ks)
            out[pol][agent] = {
                "em": round(100 * sum(rows[k]["em"] for k in ks) / n, 2),
                "calls_q": round(sum(rows[k]["calls_q"] for k in ks) / n, 3),
                "contamination": round(100 * sum(contaminated(rows[k], k[2]) for k in ks) / n, 2),
                "no_commit": round(100 * sum(no_commit(rows[k], k[2]) for k in ks) / n, 2)}
    if "admit" in em:
        out["admit_minus"] = {p: paired_boot(em["admit"], em[p]) for p in policies if p != "admit"}
    print(json.dumps(out, indent=1))
    (a.in_dir / "SUMMARY.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
