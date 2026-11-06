#!/usr/bin/env python3
"""Anytime document agent: recompute the appendix tables and contrasts.

Each released record is one question of DUDE, TAT-QA, or MMLongBench-Doc under
one memory policy (none, failopen, admit, failopen_span, admit_span) in the
four-step agent whose documents arrive one per step and cannot be reopened. A
record stores final substring EM, refusal, the dataset's native metric (DUDE
ANLS, TAT-QA F1, MMLongBench-Doc typed score), per-step EM, whether an unsupported
value was stored, and whether the final answer repeats one. Contrasts are paired
by question with 95% bootstrap intervals (4,000 draws, seed 0, questions sorted
by dataset and qid). Answer accuracy is substring EM among non-refused final
answers; its contrasts resample questions and take the difference of the two
ratios. The ReAct and AutoGen records add the agent; their pooled
contrasts average each question's paired difference over the two agents before
resampling questions. The PlanRAG records add the sub-question reads per step and
are summarized like the fixed reader. No model call is made. By default the script checks the
stored summary; pass --write to regenerate it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "anytime_document_agent_20261031.jsonl"
AGENT_RECORDS = ROOT / "results" / "anytime_react_autogen_20261031.jsonl"
PLANRAG_RECORDS = ROOT / "results" / "anytime_planrag_20261101.jsonl"
OUTPUT = ROOT / "results" / "anytime_document_agent_summary_20261031.json"
ARMS = ("none", "failopen", "admit", "failopen_span", "admit_span")
CONTRASTS = {"admit_span-failopen": ("admit_span", "failopen"), "admit-failopen": ("admit", "failopen"),
             "spans": ("failopen_span", "failopen"), "withholding": ("admit_span", "failopen_span"),
             "memory": ("failopen", "none")}
ACCURACY_CONTRASTS = ("admit_span-failopen", "withholding")


def boot(d: list[float], n: int = 4000) -> list[float]:
    rng = random.Random(0)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n))
    return [round(100 * sum(d) / len(d), 2), round(100 * bs[int(.025 * n)], 2),
            round(100 * bs[int(.975 * n) - 1], 2)]


def wrong(r: dict) -> int:
    return 1 - r["em"] - r["refused"]


def accuracy(rs: list[dict]) -> float:
    return round(100 * sum(r["em"] for r in rs) / sum(1 - r["refused"] for r in rs), 2)


def boot_accuracy(pairs: list[tuple[list[dict], list[dict]]], n: int = 4000) -> list[float]:
    # Each pair holds one question's records under the two policies (one per agent).
    cnt = [(sum(r["em"] for r in x), sum(1 - r["refused"] for r in x),
            sum(r["em"] for r in y), sum(1 - r["refused"] for r in y)) for x, y in pairs]

    def diff(sample: list[tuple[int, int, int, int]]) -> float:
        t = [sum(c[i] for c in sample) for i in range(4)]
        return t[0] / t[1] - t[2] / t[3]

    rng = random.Random(0)
    bs = sorted(diff([rng.choice(cnt) for _ in cnt]) for _ in range(n))
    return [round(100 * diff(cnt), 2), round(100 * bs[int(.025 * n)], 2), round(100 * bs[int(.975 * n) - 1], 2)]


def by_dataset(path: Path) -> tuple[int, dict]:
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    cell = {(r["dataset"], r["qid"], r["arm"]): r for r in rows}
    groups = {}
    for name in ("pooled", "dude", "tatqa", "mmlongbench"):
        qs = sorted({(d, q) for (d, q, a) in cell if name == "pooled" or d == name})
        g = {"n_questions": len(qs), "arms": {}, "contrasts": {}}
        for arm in ARMS:
            rs = [cell[(d, q, arm)] for d, q in qs]
            n = len(rs)
            g["arms"][arm] = {
                "final_em": round(100 * sum(r["em"] for r in rs) / n, 2),
                "answered_wrong": round(100 * sum(wrong(r) for r in rs) / n, 2),
                "answered": round(100 * sum(1 - r["refused"] for r in rs) / n, 2),
                "native": round(100 * sum(r["native"] for r in rs) / n, 2),
                "answer_accuracy": accuracy(rs),
                "step_em": [round(100 * sum(r["step_em"][k] for r in rs) / n, 2) for k in range(4)],
                "stored_unsupported": round(100 * sum(r["stored_unsupported"] for r in rs) / n, 2),
                "final_repeats_unsupported": round(100 * sum(r["final_repeats_unsupported"] for r in rs) / n, 2),
            }
        for key, (x, y) in CONTRASTS.items():
            g["contrasts"][key] = {
                "em_pp": boot([cell[(d, q, x)]["em"] - cell[(d, q, y)]["em"] for d, q in qs]),
                "wrong_pp": boot([wrong(cell[(d, q, x)]) - wrong(cell[(d, q, y)]) for d, q in qs]),
                "native_pp": boot([cell[(d, q, x)]["native"] - cell[(d, q, y)]["native"] for d, q in qs]),
            }
            if key in ACCURACY_CONTRASTS:
                g["contrasts"][key]["accuracy_pp"] = boot_accuracy(
                    [([cell[(d, q, x)]], [cell[(d, q, y)]]) for d, q in qs])
        groups[name] = g
    return len(rows), groups


def compute() -> dict:
    n, groups = by_dataset(RECORDS)
    out = {"n_records": n, "groups": groups}
    rows = [json.loads(l) for l in AGENT_RECORDS.read_text(encoding="utf-8").splitlines() if l.strip()]
    cell = {(r["dataset"], r["qid"], r["agent"], r["arm"]): r for r in rows}
    out["n_agent_records"] = len(rows)
    out["agents"] = {}
    for name, agents in (("pooled", ("autogen", "react")), ("react", ("react",)), ("autogen", ("autogen",))):
        qs = sorted({(d, q) for (d, q, ag, a) in cell})
        g = {"n_questions": len(qs), "agents": list(agents), "arms": {}, "contrasts": {}}
        for arm in ARMS:
            rs = [cell[(d, q, ag, arm)] for d, q in qs for ag in agents]
            n = len(rs)
            g["arms"][arm] = {
                "final_em": round(100 * sum(r["em"] for r in rs) / n, 2),
                "answered_wrong": round(100 * sum(wrong(r) for r in rs) / n, 2),
                "answered": round(100 * sum(1 - r["refused"] for r in rs) / n, 2),
                "native": round(100 * sum(r["native"] for r in rs) / n, 2),
                "answer_accuracy": accuracy(rs),
                "stored_unsupported": round(100 * sum(r["stored_unsupported"] for r in rs) / n, 2),
                "final_repeats_unsupported": round(100 * sum(r["final_repeats_unsupported"] for r in rs) / n, 2),
            }
        for key, (x, y) in CONTRASTS.items():
            g["contrasts"][key] = {
                m: boot([sum(f(cell[(d, q, ag, x)]) - f(cell[(d, q, ag, y)]) for ag in agents) / len(agents)
                         for d, q in qs])
                for m, f in (("em_pp", lambda r: r["em"]), ("wrong_pp", wrong), ("native_pp", lambda r: r["native"]))}
            if key in ACCURACY_CONTRASTS:
                g["contrasts"][key]["accuracy_pp"] = boot_accuracy(
                    [([cell[(d, q, ag, x)] for ag in agents], [cell[(d, q, ag, y)] for ag in agents]) for d, q in qs])
        out["agents"][name] = g
    out["n_planrag_records"], out["planrag"] = by_dataset(PLANRAG_RECORDS)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the stored summary")
    args = parser.parse_args()
    result = compute()
    if args.write:
        OUTPUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    else:
        assert json.loads(OUTPUT.read_text(encoding="utf-8")) == result, "stored anytime-agent summary differs"
    for name, g in result["groups"].items():
        print(name, {k: v["em_pp"] for k, v in g["contrasts"].items()})
    for name, g in result["agents"].items():
        print("agents", name, {k: v["em_pp"] for k, v in g["contrasts"].items()})
    for name, g in result["planrag"].items():
        print("planrag", name, {k: v["em_pp"] for k, v in g["contrasts"].items()})
    print("Anytime document agent: OK")


if __name__ == "__main__":
    main()
