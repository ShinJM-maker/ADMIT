#!/usr/bin/env python3
"""Injected-value replay: do withheld wrong values reach answers and later queries?

Recomputes the appendix rates from the released per-trajectory records
(M3DocVQA reader and consumer studies; DUDE, TAT-QA, and MMLongBench-Doc
portability replay, also pooled). Each record is one fail-open PlanRAG trajectory in which at least one admitted write
was replaced by the reader's closed-book answer to the same sub-question and that
value does not match the reference. For each memory policy (fail-open, ADMIT, and
random withholding of the same number of writes) a record stores, per run, the
substring EM of the final answer, whether it is the fixed refusal, whether it
repeats an injected wrong value, and, for the second consumer agent, whether such
a value enters its search queries. Random withholding has three seeds in the
reader study and one in the consumer study; seeds are averaged within a record.
No model call is made. By default the script checks the stored summary; pass
--write to regenerate it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = [ROOT / "results" / "injected_value_replay_20261030.jsonl",
           ROOT / "results" / "injected_value_replay_portability_20261030.jsonl"]
OUTPUT = ROOT / "results" / "injected_value_replay_summary_20261030.json"
POLICIES = ("failopen", "admit", "random")


def mean_pct(rows: list[dict], policy: str, field: str) -> float:
    vals = [sum(r["arms"][policy][field]) / len(r["arms"][policy][field]) for r in rows]
    return round(100 * sum(vals) / len(vals), 2)


def compute() -> dict:
    rows = [json.loads(line) for path in RECORDS
            for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["study"], r.get("dataset", "M3DocVQA"), r["inject_p"], r["env"])].append(r)
        if r["study"] == "portability":  # pooled over the three datasets
            groups[(r["study"], "pooled", r["inject_p"], r["env"])].append(r)
    cells = []
    for (study, dataset, p, env), rs in sorted(groups.items()):
        fields = ["em", "refused", "repeats_wrong"] + (["query_has_wrong"] if study == "consumer" else [])
        cells.append({
            "study": study, "dataset": dataset, "inject_p": p, "env": env, "n": len(rs),
            **{pol: {f: mean_pct(rs, pol, f) for f in fields} for pol in POLICIES},
        })
    return {
        "description": (
            "Fail-open PlanRAG trajectories with at least one injected value that does not match "
            "the reference. Rates are percentages over trajectories; random withholding removes "
            "as many writes as ADMIT withholds."
        ),
        "n_records": len(rows),
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the stored summary")
    args = parser.parse_args()
    result = compute()
    if args.write:
        OUTPUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    else:
        assert json.loads(OUTPUT.read_text(encoding="utf-8")) == result, "stored injected-value summary differs"
    for c in result["cells"]:
        line = "  ".join(f"{pol} EM {c[pol]['em']} repeat {c[pol]['repeats_wrong']}" for pol in POLICIES)
        print(f"{c['study']} {c['dataset']} p={c['inject_p']} {c['env']} n={c['n']}: {line}")
    print("Injected-value replay: OK")


if __name__ == "__main__":
    main()
