#!/usr/bin/env python3
"""Refusal-repair extension (Sec. 6.4, Table 3b, App. B.15) recomputed from released compact rows.

No model calls. Refusal repair gives an initial refusal one RepairRead; the repaired value is admitted only if it
is a non-refusal that passes the same lexical check, otherwise the refusal record stays.

Inputs (results/refusal_repair_20261106/):
  dynamic_refusal_repair_900.jsonl  one row per (qid, agent) of the 300-qid dynamic sample, with EM and reader
      calls under ADMIT, ADMIT+R, fail-open and fail-open+R. Pairs whose trajectory recorded an initial refusal
      were rerun with and without the rule (admit_rerun / failopen_rerun); the other pairs are identical under
      both arms by construction and carry the Table 1 values.
  terminal_refusal_repair.jsonl     one row per (upstream, qid) of the 18-upstream terminal replay.
  datasets_refusal_repair.jsonl     one row per qid of the typed PlanRAG transfer (DUDE, TAT-QA, MMLongBench-Doc)
      with substring EM, reader calls and the official per-question score under ADMIT and ADMIT+R.

Intervals: paired bootstrap over qids, 10,000 resamples, seed 0 (agents of a qid averaged first).
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "results" / "refusal_repair_20261106"


def boot(per_qid, B=10_000, seed=0):
    v = [sum(x) / len(x) for _, x in sorted(per_qid.items())]
    rng = random.Random(seed)
    bs = sorted(sum(v[rng.randrange(len(v))] for _ in v) / len(v) for _ in range(B))
    return [round(100 * sum(v) / len(v), 2), round(100 * bs[int(.025 * B)], 2), round(100 * bs[int(.975 * B)], 2)]


def rows(name):
    return [json.loads(line) for line in (D / name).open()]


def dynamic():
    R = rows("dynamic_refusal_repair_900.jsonl")
    out = {"admit_rerun_matches_released": f"{sum(r['admit_rerun_matches_released'] or 0 for r in R)}/"
                                           f"{sum(r['admit_rerun'] for r in R)}",
           "failopen_rerun_matches_released": f"{sum(r['failopen_rerun_matches_released'] or 0 for r in R)}/"
                                              f"{sum(r['failopen_rerun'] for r in R)}"}
    for agent in ("react", "planrag", "autogen", "pooled"):
        S = [r for r in R if agent == "pooled" or r["agent"] == agent]
        res = {}
        for pol in ("admit", "admit_rr", "failopen", "failopen_rr"):
            res[pol] = [round(100 * sum(r[f"{pol}_em"] for r in S) / len(S), 2),
                        round(sum(r[f"{pol}_calls"] for r in S) / len(S), 3)]
        for a, b in (("admit_rr", "admit"), ("failopen_rr", "failopen")):
            d = defaultdict(list)
            for r in S:
                d[r["qid"]].append(r[f"{a}_em"] - r[f"{b}_em"])
            res[f"{a} - {b}"] = boot(d)
        out[agent] = res
    return out


def terminal():
    R = rows("terminal_refusal_repair.jsonl")
    per = defaultdict(dict)
    out = {"initial_refusals": sum(r["initial_refusal"] for r in R),
           "wins": sum(r["admit_rr_em"] > r["admit_em"] for r in R),
           "losses": sum(r["admit_rr_em"] < r["admit_em"] for r in R)}
    for r in R:
        per[r["pipeline"]][r["qid"]] = r
    pipes = sorted(per)
    n = sum(len(per[p]) for p in pipes)
    out["admit"] = [round(100 * sum(r["admit_em"] for r in R) / n, 2), round(sum(r["admit_calls"] for r in R) / n, 3)]
    out["admit_rr"] = [round(100 * sum(r["admit_rr_em"] for r in R) / n, 2), round(sum(r["admit_rr_calls"] for r in R) / n, 3)]
    d = defaultdict(list)
    for p in pipes:
        for q, r in per[p].items():
            d[q].append(r["admit_rr_em"] - r["admit_em"])
    out["delta_mean_over_18"] = boot(d)
    return out


def datasets():
    R = rows("datasets_refusal_repair.jsonl")
    out = {}
    for ds in ("dude", "tatqa", "mmlongbench"):
        S = [r for r in R if r["dataset"] == ds]
        rerun = [r for r in S if r["initial_refusal_rerun"]]
        d = {r["qid"]: [r["admit_rr_native"] - r["admit_native"]] for r in S}
        down = [r for r in rerun if r["admit_rr_native"] < r["admit_native"]]
        out[ds] = {"n": len(S), "rerun": len(rerun),
                   "admit_rerun_matches_released": sum(r["admit_rerun_matches_released"] for r in rerun),
                   "native_delta": boot(d),
                   "up": sum(r["admit_rr_native"] > r["admit_native"] for r in rerun), "down": len(down),
                   "down_not_answerable": sum(r["not_answerable"] for r in down),
                   "contamination_rr": sum(r.get("admit_rr_contamination", 0) for r in rerun)}
    return out


def main():
    res = {"dynamic": dynamic(), "terminal": terminal(), "datasets": datasets()}
    print(json.dumps(res, indent=1))
    dyn, term, ds = res["dynamic"], res["terminal"], res["datasets"]
    assert dyn["admit_rerun_matches_released"] == "310/335" and dyn["failopen_rerun_matches_released"] == "312/334"
    assert dyn["pooled"]["admit"] == [33.0, 2.739] and dyn["pooled"]["admit_rr"] == [34.11, 3.438]
    assert dyn["pooled"]["failopen"] == [33.67, 2.72] and dyn["pooled"]["failopen_rr"] == [34.78, 3.419]
    assert dyn["pooled"]["admit_rr - admit"] == [1.11, 0.33, 2.22] == dyn["pooled"]["failopen_rr - failopen"]
    assert term["initial_refusals"] == 1766 and term["wins"] == 49 and term["losses"] == 0
    assert term["admit"] == [36.24, 1.109] and term["admit_rr"] == [37.15, 1.436]
    assert term["delta_mean_over_18"] == [0.91, 0.37, 1.61]
    assert ds["dude"]["native_delta"] == [2.21, 0.33, 4.22]
    assert ds["tatqa"]["native_delta"][0] == 0.05
    assert ds["mmlongbench"]["native_delta"] == [-1.67, -3.33, 0.0]
    assert ds["mmlongbench"]["down"] == ds["mmlongbench"]["down_not_answerable"] == 6
    assert all(ds[k]["contamination_rr"] == 0 for k in ds)
    print("OK")


if __name__ == "__main__":
    main()
