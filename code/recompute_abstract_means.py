#!/usr/bin/env python3
"""Recompute the abstract's mean score gains from released records, without model calls.

Injection: pool selected source-blocked cases within each injection rate, first
averaging random seeds within a record, then weight the three rates equally.
Spans: compare admit_span with value-only admit in the four matched window-
control settings, averaging questions within settings and settings equally.
Refusal rereading: compare admit_rr with admit in the three matched dynamic
agent runs, averaging questions within agents and agents equally.
All calculations use unrounded observations and exact rational arithmetic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "injection": "results/injected_value_replay_portability_20261030.jsonl",
    "spans": "results/anytime_window_control_20261101.jsonl",
    "refusal": "results/refusal_repair_20261106/dynamic_refusal_repair_900.jsonl",
}
OUTPUT = ROOT / "results/abstract_mean_effects_20261107.json"


def records(name):
    return [json.loads(line) for line in (ROOT / INPUTS[name]).read_text().splitlines() if line.strip()]


def mean(values):
    return sum(values, Fraction()) / len(values)


def fraction(value):
    return Fraction(str(value))


def point(value):
    return {"exact_pp": str(value), "unrounded_pp": float(value), "reported_pp": round(float(value), 2)}


def compute():
    rows = records("injection")
    per_rate = []
    for rate in (0.1, 0.2, 0.4):
        selected = [r for r in rows if r["study"] == "portability" and r["env"] == "blocked"
                    and r["inject_p"] == rate]
        assert selected
        scores = {pol: [mean([fraction(x) for x in r["arms"][pol]["em"]]) for r in selected]
                  for pol in ("admit", "failopen", "random")}
        diffs = {pol: 100 * mean([a - b for a, b in zip(scores["admit"], scores[pol])])
                 for pol in ("failopen", "random")}
        per_rate.append({"rate": rate, "n_cases": len(selected),
                         **{pol: point(value) for pol, value in diffs.items()}})
    injection = {
        "estimand": "Equal-weight mean of the three rate-specific pooled paired score differences; "
                    "random seeds are averaged within each selected record before rate aggregation.",
        "per_rate": per_rate,
        "admit_minus_failopen": point(mean([Fraction(r["failopen"]["exact_pp"]) for r in per_rate])),
        "admit_minus_random": point(mean([Fraction(r["random"]["exact_pp"]) for r in per_rate])),
    }

    rows = records("spans")
    cells = {(r["agent"], r["dataset"], r["qid"], r["arm"]): r for r in rows}
    assert len(cells) == len(rows)
    per_setting = []
    for setting in ("fixed", "react", "autogen", "planrag"):
        treated = [r for r in rows if r["agent"] == setting and r["arm"] == "admit_span"]
        assert len(treated) == 900
        diff = 100 * mean([fraction(r["em"]) - fraction(cells[
            (setting, r["dataset"], r["qid"], "admit")]["em"]) for r in treated])
        per_setting.append({"setting": setting, "n_questions": len(treated), **point(diff)})
    spans = {
        "estimand": "Equal-weight mean across four matched sequential-reading settings; "
                    "ADMIT with source excerpts minus value-only ADMIT.",
        "per_setting": per_setting,
        "mean": point(mean([Fraction(r["exact_pp"]) for r in per_setting])),
    }

    rows = records("refusal")
    per_agent = []
    for agent in ("react", "planrag", "autogen"):
        selected = [r for r in rows if r["agent"] == agent]
        assert len(selected) == 300
        diff = 100 * mean([fraction(r["admit_rr_em"]) - fraction(r["admit_em"]) for r in selected])
        per_agent.append({"agent": agent, "n_questions": len(selected), **point(diff)})
    refusal = {
        "estimand": "Equal-weight mean across three matched dynamic agent runs; "
                    "ADMIT with initial-refusal rereading minus ADMIT without this extension.",
        "per_agent": per_agent,
        "mean": point(mean([Fraction(r["exact_pp"]) for r in per_agent])),
    }
    return {"metric": "substring EM difference, in percentage points",
            "input_sha256": {rel: hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
                             for rel in INPUTS.values()},
            "injection": injection, "source_excerpts": spans, "refusal_rereading": refusal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the point-estimate summary")
    args = parser.parse_args()
    result = compute()
    if args.write:
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    else:
        assert json.loads(OUTPUT.read_text()) == result, "abstract mean summary differs"
    print("Injection minus continued storage:", result["injection"]["admit_minus_failopen"]["reported_pp"])
    print("Injection minus count-matched random withholding:", result["injection"]["admit_minus_random"]["reported_pp"])
    print("Source excerpts minus value-only ADMIT:", result["source_excerpts"]["mean"]["reported_pp"])
    print("Initial-refusal rereading minus base ADMIT:", result["refusal_rereading"]["mean"]["reported_pp"])
    print("Abstract means: OK")


if __name__ == "__main__":
    main()
