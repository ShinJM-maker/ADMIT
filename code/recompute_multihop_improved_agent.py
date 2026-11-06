#!/usr/bin/env python3
"""Multi-hop search agents with an improved harness, on questions no earlier run used.

Each released record is one question under one arm. Settings: 1,000 2WikiMultihopQA
validation questions outside the earlier 1,000-question sample and 1,000 MuSiQue-Ans
training questions; the corpus of each is all paragraphs of its 1,000 questions; reader
prompt with a refusal clause. Arms: none (one retrieval for the main question, no memory);
failopen_m and admit_m (the agent of recompute_multihop_final_check.py, memory-aware final
check); fo2 and admit2 (the improved harness: stored grounded values carry their card and a
220-character span to the planner and the final reader, a repeated sub-question is re-planned
once with the article titles found so far, and a search also returns the best card of up to
two corpus articles named in the query). Fields: final answer, substring EM, refusal, hops
(sub-question, retrieved titles, linked titles, reader output, stored value and whether it
occurs in the text it was read from), stored memory, re-plans, gold articles among the final
cards.

A final answer rests on an ungrounded value as in recompute_multihop_final_check.py. Per
setting the script reports each arm's EM, answered rate and resting share, and the paired
differences; the pooled row averages the two settings with equal weight. Intervals: 4,000
question resamples, seed 0, resampling within each setting. No model call is made. By
default the script checks the stored summary; --write regenerates it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from recompute_multihop_final_check import rests_on_ungrounded

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "multihop_improved_agent_20261103.jsonl"
OUTPUT = ROOT / "results" / "multihop_improved_agent_summary_20261103.json"
ARMS = ("none", "failopen_m", "admit_m", "fo2", "admit2")
PAIRS = (("fo2", "failopen_m"), ("admit2", "admit_m"), ("admit2", "fo2"), ("admit_m", "failopen_m"))


def boot(groups, n=4000):
    rng = random.Random(0)
    mean = lambda gs: sum(sum(g) / len(g) for g in gs) / len(gs)
    bs = sorted(mean([[rng.choice(g) for _ in g] for g in groups]) for _ in range(n))
    return [round(100 * mean(groups), 2), round(100 * bs[int(.025 * n)], 2), round(100 * bs[int(.975 * n) - 1], 2)]


def summarize():
    cell, order = {}, []
    for line in open(RECORDS):
        r = json.loads(line)
        if r["setting"] not in order:
            order.append(r["setting"])
        r["rests"] = rests_on_ungrounded(r) if r["arm"] != "none" else 0
        cell[(r["setting"], r["qid"], r["arm"])] = r
    qs = {s: sorted({q for x, q, _ in cell if x == s}, key=lambda q: int(q.split("_")[-1])) for s in order}
    diff = lambda s, x, y: [cell[(s, q, x)]["em"] - cell[(s, q, y)]["em"] for q in qs[s]]
    out = {}
    for s in order:
        n = len(qs[s])
        res = {"questions": n}
        for a in ARMS:
            rows = [cell[(s, q, a)] for q in qs[s]]
            res[a] = {"em": round(100 * sum(r["em"] for r in rows) / n, 2),
                      "answered": round(100 * sum(1 - r["refused"] for r in rows) / n, 2),
                      "rests_on_ungrounded": round(100 * sum(r["rests"] for r in rows) / n, 2)}
        for x, y in PAIRS:
            res[f"{x}_minus_{y}"] = boot([diff(s, x, y)])
        out[s] = res
    out["pooled"] = {"settings": order, **{a: round(sum(out[s][a]["em"] for s in order) / len(order), 2) for a in ARMS},
                     **{f"{x}_minus_{y}": boot([diff(s, x, y) for s in order]) for x, y in PAIRS}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    got = summarize()
    f = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
    for s, r in got.items():
        ems = " ".join(f"{x} {r[x]['em'] if isinstance(r[x], dict) else r[x]:.2f}" for x in ARMS)
        print(f"{s:16s} EM: {ems}")
        if s != "pooled":
            print(f"{'':16s} rests: " + " ".join(f"{x} {r[x]['rests_on_ungrounded']:.2f}" for x in ARMS[1:]))
        print(f"{'':16s} " + " | ".join(f"{x}-{y} {f(r[f'{x}_minus_{y}'])}" for x, y in PAIRS))
    if a.write:
        OUTPUT.write_text(json.dumps(got, indent=1) + "\n")
        print("wrote", OUTPUT)
    else:
        assert json.loads(OUTPUT.read_text()) == got, "summary mismatch"
        print("Multi-hop improved agent: OK")


if __name__ == "__main__":
    main()
