#!/usr/bin/env python3
"""Multi-hop search agents: admitted memory as support for the final-answer check.

Each released record is one question in one setting (FRAMES, development; 2WikiMultihopQA,
MuSiQue, MuSiQue fresh with a complete or an incomplete corpus, confirmation) under one arm:
rag (one retrieval for the main question, no memory); failopen and admit (the final check
takes the main-question cards as support); failopen_memory_support and admit_memory_support
(the stored values are added to the final check's support). Fields: final answer, substring
EM, refusal, which branch of the final check produced the output, the substring EM of the
synthesized answer before the check, and the stored memory (value, and whether it occurs in
the text it was read from).

A stored value is grounded if it occurs in the text it was read from or in an earlier
grounded stored value. A final answer rests on an ungrounded value if it is not a refusal
and it contains, or is contained in, an ungrounded stored value (whitespace and case
folded). Per setting the script reports each arm's EM and answered rate; ADMIT minus fail-open
in EM under each check; the share of final answers resting on an ungrounded value; EM of
both policies (memory-aware check) on the questions where fail-open's final answer rests on an
ungrounded value and on the other questions; the memory-aware minus default check for ADMIT;
and the RepairRead replacements under ADMIT's default check. Intervals: 4,000 question resamples, seed 0, questions in qid order; the
pooled row averages the four confirmation settings with equal weight, resampling within each.
No model call is made. By default the script checks the stored summary; --write regenerates it.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "multihop_final_check_20261102.jsonl"
OUTPUT = ROOT / "results" / "multihop_final_check_summary_20261102.json"
ARMS = ("rag", "failopen", "admit", "failopen_memory_support", "admit_memory_support")
REFUSAL = "do not have enough information"


def norm(s):
    return re.sub(r"\s+", " ", str(s or "").lower().strip())


def rests_on_ungrounded(r):
    a = norm(r["answer"])
    if not a or REFUSAL in a:
        return 0
    grounded, ungrounded = [], []
    for m in r["memory"]:
        v = norm(m["value"])
        if m["supported"] or any(v and v in g for g in grounded):
            grounded.append(v)
        else:
            ungrounded.append(v)
    return int(any(v and (v in a or a in v) for v in ungrounded))


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
        r["rests"] = rests_on_ungrounded(r) if r["arm"] != "rag" else 0
        cell[(r["setting"], r["qid"], r["arm"])] = r
    qs = {s: sorted({q for x, q, _ in cell if x == s}, key=lambda q: int(q.split("_")[1])) for s in order}
    diff = lambda s, x, y, k: [cell[(s, q, x)][k] - cell[(s, q, y)][k] for q in qs[s]]
    out = {}
    for s in order:
        rows = {a: [cell[(s, q, a)] for q in qs[s]] for a in ARMS}
        n = len(qs[s])
        res = {"split": rows["admit"][0]["split"], "reader_prompt": rows["admit"][0]["reader_prompt"], "questions": n}
        for a in ARMS:
            res[a] = {"em": round(100 * sum(r["em"] for r in rows[a]) / n, 2),
                      "answered": round(100 * sum(1 - r["refused"] for r in rows[a]) / n, 2),
                      "rests_on_ungrounded": round(100 * sum(r["rests"] for r in rows[a]) / n, 2)}
        res["admit_minus_failopen_em_default"] = boot([diff(s, "admit", "failopen", "em")])
        res["admit_minus_failopen_em_memory_support"] = boot([diff(s, "admit_memory_support", "failopen_memory_support", "em")])
        res["admit_minus_failopen_rests_memory_support"] = boot([diff(s, "admit_memory_support", "failopen_memory_support", "rests")])
        res["memory_support_minus_default_admit_em"] = boot([diff(s, "admit_memory_support", "admit", "em")])
        rep = [r for r in rows["admit"] if r["final_check"] == "repair"]
        res["admit_default_check"] = {
            "check_failed": round(100 * sum(r["final_check"] != "candidate" for r in rows["admit"]) / n, 2),
            "replaced_by_repair": round(100 * len(rep) / n, 2),
            "replaced_synthesized_correct": sum(r["synthesized_em"] for r in rep),
            "replaced_output_correct": sum(r["em"] for r in rep)}
        rq = [q for q in qs[s] if cell[(s, q, "failopen_memory_support")]["rests"]]
        oq = [q for q in qs[s] if not cell[(s, q, "failopen_memory_support")]["rests"]]
        acc = lambda L, arm: round(100 * sum(cell[(s, q, arm)]["em"] for q in L) / len(L), 2) if L else None
        res["by_failopen_resting"] = {
            "resting": {"questions": len(rq), "failopen_em": acc(rq, "failopen_memory_support"),
                        "admit_em": acc(rq, "admit_memory_support")},
            "other": {"questions": len(oq), "failopen_em": acc(oq, "failopen_memory_support"),
                      "admit_em": acc(oq, "admit_memory_support")}}
        res["admit_memory_support_check_failed"] = round(
            100 * sum(r["final_check"] != "candidate" for r in rows["admit_memory_support"]) / n, 2)
        out[s] = res
    conf = [s for s in order if out[s]["split"] == "confirmation"]
    out["pooled confirmation"] = {
        "settings": conf,
        "admit_minus_failopen_em_memory_support": boot([diff(s, "admit_memory_support", "failopen_memory_support", "em") for s in conf]),
        "admit_minus_failopen_em_default": boot([diff(s, "admit", "failopen", "em") for s in conf]),
        "memory_support_minus_default_admit_em": boot([diff(s, "admit_memory_support", "admit", "em") for s in conf])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    got = summarize()
    f = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
    for s, r in got.items():
        if s == "pooled confirmation":
            print(f"{s:34s} ADMIT-FO (memory-aware) {f(r['admit_minus_failopen_em_memory_support'])}  "
                  f"(default) {f(r['admit_minus_failopen_em_default'])}  memory-aware - default (ADMIT) {f(r['memory_support_minus_default_admit_em'])}")
            continue
        print(f"{s:34s} {r['split']:12s} n {r['questions']:4d} EM: RAG {r['rag']['em']:.2f} FO {r['failopen']['em']:.2f} "
              f"ADMIT {r['admit']['em']:.2f} FO+m {r['failopen_memory_support']['em']:.2f} ADMIT+m {r['admit_memory_support']['em']:.2f} | "
              f"ADMIT-FO (memory-aware) {f(r['admit_minus_failopen_em_memory_support'])} | rests on ungrounded: "
              f"FO+m {r['failopen_memory_support']['rests_on_ungrounded']:.2f} ADMIT+m {r['admit_memory_support']['rests_on_ungrounded']:.2f} | "
              f"memory-aware - default (ADMIT) {f(r['memory_support_minus_default_admit_em'])}")
    if a.write:
        OUTPUT.write_text(json.dumps(got, indent=1) + "\n")
        print("wrote", OUTPUT)
    else:
        assert json.loads(OUTPUT.read_text()) == got, "summary mismatch"
        print("Multi-hop final check: OK")


if __name__ == "__main__":
    main()
