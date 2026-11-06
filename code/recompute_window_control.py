#!/usr/bin/env python3
"""Window control for support spans: recompute the appendix contrasts.

Each released record is one question of DUDE, TAT-QA, or MMLongBench-Doc in one
anytime-document setting (agent: fixed reader, ReAct, AutoGen, PlanRAG) under one
policy: admit, admit_span (each admitted value stored with the text around its
first match), or admit_window (the same values, each with an excerpt of the same
length taken from the start of the text read at that step, without locating the
value). Primary: admit_span minus admit_window in final substring EM, each
question's difference averaged over the four settings, 95% bootstrap over
questions (4,000 draws, seed 0, questions sorted by dataset and qid). Per setting
the script reports EM, answered rate, answer accuracy (substring EM among
non-refused final answers; difference of ratios over resampled questions), and
the mean official metric. No model call is made. By default the script checks the
stored summary; pass --write to regenerate it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "anytime_window_control_20261101.jsonl"
OUTPUT = ROOT / "results" / "anytime_window_control_summary_20261101.json"
SETTINGS = ("fixed", "react", "autogen", "planrag")
ARMS = ("admit", "admit_span", "admit_window")
CONTRASTS = (("admit_span", "admit_window"), ("admit_window", "admit"), ("admit_span", "admit"))


def boot(vals: list[float], n: int = 4000) -> list[float]:
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(vals) for _ in vals) / len(vals) for _ in range(n))
    return [round(100 * sum(vals) / len(vals), 2), round(100 * bs[int(.025 * n)], 2),
            round(100 * bs[int(.975 * n) - 1], 2)]


def boot_ratio(cnt: list[tuple[int, int, int, int]], n: int = 4000) -> list[float]:
    def diff(sample):
        t = [sum(c[i] for c in sample) for i in range(4)]
        return t[0] / t[1] - t[2] / t[3]
    rng = random.Random(0)
    bs = sorted(diff([rng.choice(cnt) for _ in cnt]) for _ in range(n))
    return [round(100 * diff(cnt), 2), round(100 * bs[int(.025 * n)], 2), round(100 * bs[int(.975 * n) - 1], 2)]


def summarize() -> dict:
    rows = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    cell = {(r["agent"], r["dataset"], r["qid"], r["arm"]): r for r in rows}
    qs = sorted({(d, q) for (s, d, q, a) in cell})
    em = lambda s, d, q, a: cell[(s, d, q, a)]["em"]
    out = {"n_questions": len(qs), "n_records": len(rows),
           "primary_span_minus_window_em": boot([sum(em(s, d, q, "admit_span") - em(s, d, q, "admit_window")
                                                     for s in SETTINGS) / len(SETTINGS) for d, q in qs]),
           "settings": {}}
    for s in SETTINGS:
        g = {"arms": {}, "contrasts": {}}
        for arm in ARMS:
            rs = [cell[(s, d, q, arm)] for d, q in qs]
            ans = sum(1 - r["refused"] for r in rs)
            g["arms"][arm] = {"final_em": round(100 * sum(r["em"] for r in rs) / len(rs), 2),
                              "answered": round(100 * ans / len(rs), 2),
                              "answer_accuracy": round(100 * sum(r["em"] for r in rs) / ans, 2),
                              "native": round(100 * sum(r["native"] for r in rs) / len(rs), 2)}
        for x, y in CONTRASTS:
            g["contrasts"][f"{x}-{y}"] = {
                "em_pp": boot([em(s, d, q, x) - em(s, d, q, y) for d, q in qs]),
                "native_pp": boot([cell[(s, d, q, x)]["native"] - cell[(s, d, q, y)]["native"] for d, q in qs]),
                "accuracy_pp": boot_ratio([(cell[(s, d, q, x)]["em"], 1 - cell[(s, d, q, x)]["refused"],
                                            cell[(s, d, q, y)]["em"], 1 - cell[(s, d, q, y)]["refused"])
                                           for d, q in qs])}
        out["settings"][s] = g
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    summary = summarize()
    if args.write:
        OUTPUT.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {OUTPUT.name}")
    else:
        stored = json.loads(OUTPUT.read_text(encoding="utf-8"))
        assert stored == summary, "summary mismatch"
        print("Window control: OK")
    p = summary["primary_span_minus_window_em"]
    print(f"primary admit_span - admit_window EM: {p[0]:+.2f} [{p[1]:+.2f}, {p[2]:+.2f}]")


if __name__ == "__main__":
    main()
