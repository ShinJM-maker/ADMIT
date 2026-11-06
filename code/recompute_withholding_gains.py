#!/usr/bin/env python3
"""Where withholding raises answer scores: the main-text table and the multi-hop figures of the appendix.

Injected rows (main-text table): the portability injected-value replay (DUDE, TAT-QA, MMLongBench-Doc,
pooled over trajectories) with sources blocked or available, at injection rates
0.1/0.2/0.4; substring EM of fail-open, random withholding of as many writes
(seeds averaged within a record), and ADMIT, with paired differences. Multi-hop rows (appendix):
for every multi-hop setting, the questions on which fail-open's final answer (memory-aware
check) rests on an ungrounded stored value, and the other questions; EM of fail-open and
ADMIT (memory-aware check) and their paired difference.

Intervals: 4,000 paired resamples with seed 0; injected rows resample trajectories
within each dataset, multi-hop rows resample questions. No model call is made. By
default the script checks the stored summary; --write regenerates it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from recompute_multihop_final_check import rests_on_ungrounded

ROOT = Path(__file__).resolve().parents[1]
INJECTED = ROOT / "results" / "injected_value_replay_portability_20261030.jsonl"
MULTIHOP = ROOT / "results" / "multihop_final_check_20261102.jsonl"
OUTPUT = ROOT / "results" / "withholding_gains_summary_20261102.json"


def mean(v):
    return sum(v) / len(v)


def boot(groups, n=4000):
    rng = random.Random(0)
    est = mean([x for g in groups for x in g])
    bs = sorted(mean([rng.choice(g) for g in groups for _ in g]) for _ in range(n))
    return [round(100 * est, 2), round(100 * bs[int(.025 * n)], 2), round(100 * bs[int(.975 * n) - 1], 2)]


def injected():
    rows = [json.loads(l) for l in open(INJECTED)]
    out = []
    for env in ("blocked", "available"):
        for p in (0.1, 0.2, 0.4):
            rs = [r for r in rows if r["env"] == env and r["inject_p"] == p]
            ds = sorted({r["dataset"] for r in rs})
            em = lambda r, a: mean(r["arms"][a]["em"])
            diff = lambda x, y: [[em(r, x) - em(r, y) for r in rs if r["dataset"] == d] for d in ds]
            out.append({"env": env, "inject_p": p, "trajectories": len(rs),
                        **{a: round(100 * mean([em(r, a) for r in rs]), 2) for a in ("failopen", "random", "admit")},
                        "admit_minus_failopen": boot(diff("admit", "failopen")),
                        "admit_minus_random": boot(diff("admit", "random"))})
    return out


def multihop():
    cell, order = {}, []
    for line in open(MULTIHOP):
        r = json.loads(line)
        if r["setting"] not in order:
            order.append(r["setting"])
        cell[(r["setting"], r["qid"], r["arm"])] = r
    out = []
    for s in order:
        qs = sorted({q for x, q, _ in cell if x == s}, key=lambda q: int(q.split("_")[1]))
        rest = {q for q in qs if rests_on_ungrounded(cell[(s, q, "failopen_memory_support")])}
        for name, L in (("failopen_rests", [q for q in qs if q in rest]), ("other", [q for q in qs if q not in rest])):
            if not L:
                continue
            f = [cell[(s, q, "failopen_memory_support")]["em"] for q in L]
            a = [cell[(s, q, "admit_memory_support")]["em"] for q in L]
            out.append({"setting": s, "reader_prompt": cell[(s, qs[0], "admit")]["reader_prompt"],
                        "questions": name, "n": len(L), "failopen": round(100 * mean(f), 2),
                        "admit": round(100 * mean(a), 2),
                        "admit_minus_failopen": boot([[x - y for x, y in zip(a, f)]])})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    got = {"injected": injected(), "multihop": multihop()}
    f = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
    for r in got["injected"]:
        print(f"injected {r['env']:9s} p={r['inject_p']} n {r['trajectories']:3d} FO {r['failopen']:.2f} "
              f"random {r['random']:.2f} ADMIT {r['admit']:.2f} | ADMIT-FO {f(r['admit_minus_failopen'])} "
              f"ADMIT-random {f(r['admit_minus_random'])}")
    for r in got["multihop"]:
        print(f"multihop {r['setting']:34s} {r['reader_prompt']:6s} {r['questions']:14s} n {r['n']:4d} "
              f"FO {r['failopen']:.2f} ADMIT {r['admit']:.2f} | ADMIT-FO {f(r['admit_minus_failopen'])}")
    if args.write:
        OUTPUT.write_text(json.dumps(got, indent=1) + "\n")
        print("wrote", OUTPUT)
    else:
        assert json.loads(OUTPUT.read_text()) == got, "summary mismatch"
        print("Withholding gains: OK")


if __name__ == "__main__":
    main()
