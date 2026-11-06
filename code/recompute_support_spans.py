#!/usr/bin/env python3
"""Support spans as memory provenance: recompute the appendix decomposition.

Each released record is one fail-open PlanRAG trajectory of the portability
replay (DUDE, TAT-QA, MMLongBench-Doc) read a second time by the frozen reader
under four memories built from the same writes: fail-open, ADMIT, fail-open with
the support span of each admitted write, and ADMIT with those spans. A record
stores substring EM and refusal per memory. The script reports paired
differences in substring EM and in answered-but-wrong rate (not refused and EM 0)
with 95% bootstrap intervals over trajectories (4,000 draws, seed 0). No model
call is made. By default it checks the stored summary; pass --write to
regenerate it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "support_span_memory_20261031.jsonl"
OUTPUT = ROOT / "results" / "support_span_memory_summary_20261031.json"
PAIRS = {"total": ("admit_span", "failopen"), "spans": ("failopen_span", "failopen"),
         "withholding": ("admit_span", "failopen_span")}


def boot(d: list[float], n: int = 4000) -> list[float]:
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return [round(100 * sum(d) / len(d), 2), round(100 * bs[int(.025 * n)], 2),
            round(100 * bs[int(.975 * n) - 1], 2)]


def wrong(c: dict) -> int:
    return 1 - c["em"] - c["refused"]


def compute() -> dict:
    rows = [json.loads(l) for l in RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    cells = []
    for p in (0.0, 0.2):
        for env in ("available", "blocked"):
            for ds in ("pooled", "DUDE", "TAT-QA", "MMLongBench-Doc"):
                rs = [r for r in rows if r["inject_p"] == p and r["env"] == env
                      and (ds == "pooled" or r["dataset"] == ds)]
                cell = {"inject_p": p, "env": env, "dataset": ds, "n": len(rs)}
                for name, (x, y) in PAIRS.items():
                    cell[name] = {
                        "em_pp": boot([r["arms"][x]["em"] - r["arms"][y]["em"] for r in rs]),
                        "wrong_pp": boot([wrong(r["arms"][x]) - wrong(r["arms"][y]) for r in rs]),
                    }
                cells.append(cell)
    return {"description": "Paired differences (pp; mean, 95% bootstrap CI) between memories built "
                           "from the same fail-open PlanRAG writes.", "n_records": len(rows), "cells": cells}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the stored summary")
    args = parser.parse_args()
    result = compute()
    if args.write:
        OUTPUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    else:
        assert json.loads(OUTPUT.read_text(encoding="utf-8")) == result, "stored support-span summary differs"
    for c in result["cells"]:
        if c["inject_p"] == 0.0 or c["dataset"] == "pooled":
            print(f"p={c['inject_p']} {c['env']} {c['dataset']} n={c['n']}: " + "  ".join(
                f"{k} EM {c[k]['em_pp']} wrong {c[k]['wrong_pp']}" for k in PAIRS))
    print("Support-span memory: OK")


if __name__ == "__main__":
    main()
