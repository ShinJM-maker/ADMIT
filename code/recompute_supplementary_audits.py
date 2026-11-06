#!/usr/bin/env python3
"""Recompute the supplementary OCR, derived-value, and audit-scope counts.

The OCR rates and qid-cluster intervals are recomputed from the per-row file
with the original seeds. The TAT-QA counts are recounted from the per-case
flags, and each official derivation is replayed with an arithmetic-only parser.
The correction, log-completeness, and decomposition counts are checked against
the appendix values. No model or network access is used.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import json
import math
import operator
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "supplementary_audits_20260725"
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def close(observed: float, expected: float, label: str) -> None:
    assert math.isclose(observed, expected, rel_tol=0, abs_tol=1e-9), (label, observed, expected)


def ocr_delta(rows: list[dict], left: str, right: str, field: str, seed: int) -> tuple[float, list[float]]:
    by_qid: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
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


def check_ocr() -> dict:
    stored = load("ocr_perturbation_summary.json")
    rows = read_jsonl(DATA / "ocr_perturbation.jsonl")
    assert len(rows) == 600 and len({(row["qid"], row["agent"]) for row in rows}) == 150
    for condition, expected in stored["ocr"].items():
        group = [row for row in rows if row["condition"] == condition]
        assert len(group) == expected["n"] == 150
        for field, key in (("answer_supported", "answer_supported_pct"),
                           ("candidate_still_supported", "candidate_still_supported_pct"),
                           ("em", "em_pct")):
            close(100 * float(np.mean([row[field] for row in group])), expected[key], f"{condition}:{field}")
    for index, condition in enumerate(("span_whitespace", "span_confusion", "global_cer_0.5pct")):
        for offset, field in enumerate(("candidate_still_supported", "answer_supported", "em")):
            point, ci = ocr_delta(rows, condition, "clean", field, 20260730 + index * 3 + offset)
            expected = stored["paired_deltas"][f"ocr_{condition}_minus_clean_{field}"]
            close(point, expected["point_pp"], f"{condition}:{field}")
            close(ci[0], expected["ci95_pp"][0], f"{condition}:{field}:lo")
            close(ci[1], expected["ci95_pp"][1], f"{condition}:{field}:hi")
    return stored


def evaluate(node: ast.AST) -> Decimal:
    if isinstance(node, ast.Expression):
        return evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return Decimal(str(node.value))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = evaluate(node.operand)
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](evaluate(node.left), evaluate(node.right))
    raise ValueError(ast.dump(node))


def replay(derivation: str, gold: str) -> bool:
    expression = re.sub(r"(\d[\d,]*(?:\.\d+)?)\s*%", r"(\1/100)", derivation)
    expression = re.sub(r"\s+", "", expression.replace(",", "").replace("$", ""))
    assert not re.search(r"[^0-9.+\-*/()]", expression), derivation
    value = evaluate(ast.parse(expression, mode="eval"))
    target = Decimal(gold)
    quantum = Decimal(1).scaleb(min(0, target.as_tuple().exponent))
    return any((value * scale).quantize(quantum, rounding=ROUND_HALF_UP) == target
               for scale in (Decimal(1), Decimal(100)))


def check_tatqa() -> dict:
    stored = load("tatqa_derived_value_summary.json")
    rows = read_jsonl(DATA / "tatqa_derived_value_cases.jsonl")
    assert len(rows) == stored["derivability"]["all_arithmetic"] == 145
    assert sum(row["execution"]["matched"] for row in rows) == 145
    assert sum(replay(row["derivation"], row["execution"]["gold_numeric"]) for row in rows) == 145
    derived = [row for row in rows if row["derived_only"]]
    target = [row for row in derived if row["strict_numeric_output_match"] and row["terminal_quarantine"]]
    assert len(derived) == stored["derivability"]["derived_only"] == 117
    assert len(target) == stored["structural_grounding"]["derived_only_strict_match_terminal_quarantine"] == 12
    assert sum(row["table_context_verified"] for row in target) == stored["structural_grounding"]["target_with_verified_table_context"] == 12
    return stored


def check_scope() -> None:
    correction = load("exact_packet_correction_summary.json")
    assert (correction["nonrefusal_admission_decisions"], correction["decision_changes"],
            correction["trajectories_with_decision_change"], correction["trajectories"]) == (1758, 107, 54, 900)
    assert correction["decision_changes_serialization_admit_packet_reject"] + correction["decision_changes_serialization_reject_packet_admit"] == 107
    log = load("audit_log_completeness_summary.json")
    completeness = log["audit_completeness"]
    assert completeness["writes"] == 2508 and all(
        completeness[key] == 2508 for key in ("candidate", "admit_reason", "final_source", "memory_source", "packet_hash"))
    assert completeness["terminal_writes"] == completeness["terminal_output_value"] == 900
    assert all(log["closure_contract"].values())
    assert log["schema_prevalence"]["heuristic_multi_value"] == 172
    structured = load("structured_decomposition_summary.json")["multi_value"]
    assert (structured["surface_flagged_writes"], structured["valid_constructions"], structured["valid_atomic"],
            structured["invalid_fail_closed"], structured["valid_multi_writes"]) == (172, 148, 100, 24, 48)
    assert structured["monolithic_pass_but_component_failure"] == 0
    final_only = load("response_period_final_only_transfer_summary.json")["datasets"]
    for dataset, (contamination, em) in {"DUDE": (-31.33, -0.33), "TAT-QA": (-67.33, 0.00), "MMLongBench-Doc": (-25.00, 1.00)}.items():
        contrast = final_only[dataset]["admit_minus_final_support_retry"]
        assert round(contrast["lexical_trajectory_contamination"]["point_pp"], 2) == contamination
        assert round(contrast["raw_em"]["point_pp"], 2) == em
        assert contrast["raw_em"]["lo95_pp"] <= 0 <= contrast["raw_em"]["hi95_pp"]


def main() -> None:
    ocr = check_ocr()
    for condition, row in ocr["ocr"].items():
        print(f"OCR {condition}: supported output {row['answer_supported_pct']:.2f}, EM {row['em_pct']:.2f}")
    delta = ocr["paired_deltas"]["ocr_span_whitespace_minus_clean_answer_supported"]
    print(f"OCR span_whitespace minus clean, supported output: {delta['point_pp']:+.2f} "
          f"[{delta['ci95_pp'][0]:+.2f}, {delta['ci95_pp'][1]:+.2f}]")
    check_tatqa()
    print("TAT-QA: 145/145 derivations replayed; 117 derived-only; 12 withheld exact matches, 12 with table context")
    check_scope()
    print("Correction 107/1,758 decisions in 54/900 trajectories; audit log 2,508/2,508 writes, 900/900 outputs; 172 flagged, 148 valid parses")
    print("Supplementary audits: OK")


if __name__ == "__main__":
    main()
