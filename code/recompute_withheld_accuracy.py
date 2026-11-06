#!/usr/bin/env python3
"""Accuracy of answers withheld from memory versus admitted non-refusal answers.

Recomputes the comparison in the appendix from released compact rows without
model calls. Each setting resamples qids 10,000 times with its own fixed seed,
keeping every agent or upstream row of a sampled qid. By default the script
checks the stored result file; pass --write to regenerate it.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PORTABILITY = RESULTS / "strict_portability_20260928"
OUTPUT = RESULTS / "withheld_accuracy_20261025.json"
NOT_ANSWERABLE = RESULTS / "mmlongbench_not_answerable_qids_20261025.json"
DRAWS = 10_000
SEED = 20261025
TASK_METRIC = {
    "dude": "anls",
    "tatqa": "native_f1",
    "mmlongbench": "generalized_accuracy_short_output",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(groups: dict[str, list[tuple[int, float]]], seed: int) -> dict:
    qids = sorted(groups)

    def stat(sample: list[str]) -> tuple[float, int, int, float, float]:
        withheld = [s for q in sample for flag, s in groups[q] if flag]
        admitted = [s for q in sample for flag, s in groups[q] if not flag]
        w = sum(withheld) / len(withheld) * 100
        a = sum(admitted) / len(admitted) * 100
        return w - a, len(withheld), len(admitted), w, a

    diff, n_w, n_a, w, a = stat(qids)
    rng = random.Random(seed)
    draws = sorted(stat([rng.choice(qids) for _ in qids])[0] for _ in range(DRAWS))
    return {
        "withheld_score": round(w, 2),
        "withheld_n": n_w,
        "admitted_score": round(a, 2),
        "admitted_n": n_a,
        "difference_pp": round(diff, 2),
        "ci95": [round(draws[int(0.025 * DRAWS)], 2), round(draws[int(0.975 * DRAWS) - 1], 2)],
        "qids": len(qids),
    }


def compute() -> dict:
    writes = {
        (r["qid"], r["agent"]): r
        for r in read_jsonl(RESULTS / "final_write_diagnostics_per_qid_20261022.jsonl")
        if r["policy"] == "cavec_mfc"
    }
    finals = {
        (r["qid"], r["agent"]): r
        for r in read_jsonl(RESULTS / "strict_comparator_per_qid.jsonl")
        if r["policy"] == "cavec_mfc"
    }
    refusals = {
        (r["family"], r["setting"], r["qid"]): r["refusal"]["admit"]
        for r in read_jsonl(ROOT / "configs" / "portability_refusal_flags.jsonl")
    }
    not_answerable = set(json.loads(NOT_ANSWERABLE.read_text(encoding="utf-8"))["qids"])
    out: dict[str, dict] = {}
    offset = 0

    for metric in ("strict_em", "substring_em"):
        groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for key, row in writes.items():
            if row["terminal_no_commit"]:
                groups[key[0]].append((1, row[metric]))
            elif not finals[key]["final_refusal"]:
                groups[key[0]].append((0, row[metric]))
        out[f"m3docvqa_dynamic_terminal_{metric}"] = summarize(groups, SEED + offset)
        offset += 1

    groups = defaultdict(list)
    for row in read_jsonl(PORTABILITY / "terminal_per_qid.jsonl"):
        arm = row["arms"]["admit"]
        if arm["no_commit"]:
            groups[row["qid"]].append((1, arm["raw_em"]))
        elif not refusals[("terminal", row["pipeline"], row["qid"])]:
            groups[row["qid"]].append((0, arm["raw_em"]))
    out["m3docvqa_terminal_replay_substring_em"] = summarize(groups, SEED + offset)
    offset += 1

    rows = read_jsonl(PORTABILITY / "dataset_per_qid.jsonl")
    for dataset, exclude_na in (("dude", False), ("tatqa", False), ("mmlongbench", False), ("mmlongbench", True)):
        groups = defaultdict(list)
        for row in rows:
            if row["dataset"] != dataset:
                continue
            if exclude_na and row["qid"] in not_answerable:
                continue
            arm = row["arms"]["admit"]
            score = arm["native_metrics"][TASK_METRIC[dataset]]
            if arm["no_commit"]:
                groups[row["qid"]].append((1, score))
            elif not refusals[("dataset", dataset, row["qid"])]:
                groups[row["qid"]].append((0, score))
        name = f"{dataset}_any_no_commit_task_score" + ("_excluding_not_answerable" if exclude_na else "")
        out[name] = summarize(groups, SEED + offset)
        offset += 1

    return {
        "description": (
            "Withheld: ADMIT runs with a terminal NO_COMMIT (dynamic M3DocVQA) or any "
            "NO_COMMIT (portability files). Admitted: remaining ADMIT runs whose final "
            "output is not the fixed refusal. Groups follow the admission decision, so "
            "differences describe the evaluated populations, not a causal effect."
        ),
        "draws": DRAWS,
        "seed_base": SEED,
        "settings": out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the stored result file")
    args = parser.parse_args()
    result = compute()
    if args.write:
        OUTPUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    else:
        stored = json.loads(OUTPUT.read_text(encoding="utf-8"))
        assert stored == result, "stored withheld-accuracy results differ from recomputation"
    for name, row in result["settings"].items():
        print(
            f"{name}: withheld {row['withheld_score']:.1f} (n={row['withheld_n']}), "
            f"admitted {row['admitted_score']:.1f} (n={row['admitted_n']}), "
            f"diff {row['difference_pp']:+.1f} [{row['ci95'][0]:+.1f}, {row['ci95'][1]:+.1f}]"
        )
    print("Withheld-answer accuracy: OK")


if __name__ == "__main__":
    main()
