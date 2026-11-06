#!/usr/bin/env python3
"""Recompute the response-period three-turn and transcript-reuse diagnostics.

Both tests use saved reader outputs under the pre-correction contract and are
reported in the appendix as historical interaction checks. The script recomputes
condition rates and qid-cluster bootstrap intervals with the original seeds and
checks them against the stored summaries. No model or network access is used.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "historical_multiturn_20260726"
METRICS = (
    "recall_exact",
    "forgetting_failure",
    "valid_transition",
    "silent_change",
    "candidate_persistence",
    "final_lexical_support",
    "final_em",
    "final_abstention",
)
CONTRASTS = (("H1_M0", "H1_M1"), ("H1_M0", "H0_M0"), ("H0_M1", "H0_M0"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def close(observed: float, expected: float, label: str) -> None:
    assert math.isclose(observed, expected, rel_tol=0, abs_tol=1e-9), (label, observed, expected)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] * (1 - (index - low)) + ordered[high] * (index - low)


def turn_delta(rows: list[dict], a: str, b: str, metric: str, seed: int) -> tuple[float, list[float]]:
    def qid_means(condition: str) -> dict[str, float]:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            if row["condition"] == condition:
                grouped[row["qid"]].append(float(row["scores"][metric]))
        return {qid: mean(values) for qid, values in grouped.items()}

    left, right = qid_means(a), qid_means(b)
    qids = sorted(set(left) & set(right))
    deltas = {qid: left[qid] - right[qid] for qid in qids}
    rng = random.Random(seed)
    draws = [mean(deltas[qid] for qid in [rng.choice(qids) for _ in qids]) for _ in range(10_000)]
    return mean(deltas.values()), [percentile(draws, 0.025), percentile(draws, 0.975)]


def check_three_turn() -> dict:
    stored = json.loads((DATA / "three_turn_summary.json").read_text(encoding="utf-8"))
    primary = read_jsonl(DATA / "three_turn_primary.jsonl")
    tagged = read_jsonl(DATA / "three_turn_provenance_tag.jsonl")
    assert len(primary) == 284 and len({row["sample_id"] for row in primary}) == 71
    for condition, expected in stored["primary"]["condition"].items():
        subset = [row for row in primary if row["condition"] == condition]
        assert len(subset) == expected["n"] == 71
        for metric in METRICS:
            close(mean(float(row["scores"][metric]) for row in subset), expected[metric], f"{condition}:{metric}")
    for index, (a, b) in enumerate(CONTRASTS):
        for offset, metric in enumerate(METRICS):
            delta, ci = turn_delta(primary, a, b, metric, 20260726 + index * 100 + offset)
            expected = stored["primary"]["contrasts"][f"{a}_minus_{b}"][metric]
            close(delta, expected["delta"], f"{a}-{b}:{metric}")
            close(ci[0], expected["ci95"][0], f"{a}-{b}:{metric}:lo")
            close(ci[1], expected["ci95"][1], f"{a}-{b}:{metric}:hi")
    paired = [row for row in primary if row["condition"] == "H1_M0"] + tagged
    for index, metric in enumerate(METRICS):
        delta, ci = turn_delta(paired, "H1_M0_TAG", "H1_M0", metric, 20261726 + index)
        expected = stored["provenance_tag"]["tag_minus_untagged"][metric]
        close(delta, expected["delta"], f"tag:{metric}")
        close(ci[0], expected["ci95"][0], f"tag:{metric}:lo")
        close(ci[1], expected["ci95"][1], f"tag:{metric}:hi")
    return stored["primary"]


def transcript_delta(rows: list[dict], left: str, right: str, field: str, seed: int) -> tuple[float, list[float]]:
    by_qid: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if field == "em" and row["position"] != "terminal_memory_write":
            continue
        if row["condition"] in {left, right}:
            by_qid[row["qid"]][row["condition"]].append(float(row[field]))
    values = np.asarray([
        np.mean(group[left]) - np.mean(group[right])
        for _, group in sorted(by_qid.items())
        if left in group and right in group
    ])
    rng = np.random.default_rng(seed)
    draws = np.asarray([values[rng.integers(0, len(values), len(values))].mean() for _ in range(10_000)])
    return 100 * float(values.mean()), [100 * float(np.quantile(draws, q)) for q in (0.025, 0.975)]


def check_transcript() -> dict:
    stored = json.loads((DATA / "transcript_reuse_summary.json").read_text(encoding="utf-8"))
    rows = read_jsonl(DATA / "transcript_reuse.jsonl")
    assert len(rows) == 549 and len({row["audit_id"] for row in rows}) == 183
    for condition, expected in stored["transcript"].items():
        group = [row for row in rows if row["condition"] == condition]
        terminal = [row for row in group if row["position"] == "terminal_memory_write"]
        assert len(group) == expected["n"] and len(terminal) == expected["em_n"]
        close(100 * float(np.mean([row["repeats_candidate"] for row in group])), expected["repeat_candidate_pct"], condition)
        close(100 * float(np.mean([row["answer_supported"] for row in group])), expected["answer_supported_pct"], condition)
        close(100 * float(np.mean([row["em"] for row in terminal])), expected["em_pct"], condition)
    specs = {
        "transcript_naive_minus_terminal_repeat": ("naive_transcript_reuse", "terminal_contract", "repeats_candidate", 20260725),
        "transcript_provenance_minus_naive_repeat": ("provenance_aware_reuse", "naive_transcript_reuse", "repeats_candidate", 20260726),
        "transcript_provenance_minus_naive_em": ("provenance_aware_reuse", "naive_transcript_reuse", "em", 20260727),
    }
    for name, (left, right, field, seed) in specs.items():
        point, ci = transcript_delta(rows, left, right, field, seed)
        expected = stored["paired_deltas"][name]
        close(point, expected["point_pp"], name)
        close(ci[0], expected["ci95_pp"][0], f"{name}:lo")
        close(ci[1], expected["ci95_pp"][1], f"{name}:hi")
    return stored


def main() -> None:
    turn = check_three_turn()
    mfc, fo = turn["condition"]["H1_M0"], turn["condition"]["H1_M1"]
    contrast = turn["contrasts"]["H1_M0_minus_H1_M1"]
    for metric in ("recall_exact", "silent_change", "candidate_persistence", "final_em"):
        ci = contrast[metric]["ci95_pp"]
        print(
            f"three-turn {metric}: MFC/OFO {100 * mfc[metric]:.2f}, fail-open {100 * fo[metric]:.2f}, "
            f"diff {contrast[metric]['delta_pp']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]"
        )
    transcript = check_transcript()
    for condition, row in transcript["transcript"].items():
        print(f"transcript {condition}: repeats {row['repeat_candidate_pct']:.2f}, EM {row['em_pct']:.2f} (n={row['em_n']})")
    for name, row in transcript["paired_deltas"].items():
        print(f"{name}: {row['point_pp']:+.2f} [{row['ci95_pp'][0]:+.2f}, {row['ci95_pp'][1]:+.2f}]")
    print("Historical multi-turn diagnostics: OK")


if __name__ == "__main__":
    main()
