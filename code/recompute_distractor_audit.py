#!/usr/bin/env python3
"""Controlled distractor audit: lexical admission of planted-distractor copies.

Recomputes the appendix counts and Wilson 95% intervals from the released
per-question records. Each record holds the saved reader answer on a perturbed
packet, its counterfactual stratum, whether it copies the planted distractor,
and the lexical admission decision on the exact perturbed text. No model call
is made. By default the script checks the stored summary; pass --write to
regenerate it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORDS = ROOT / "results" / "controlled_distractor_audit_20261025.jsonl"
OUTPUT = ROOT / "results" / "controlled_distractor_audit_summary_20261025.json"
Z = 1.959963984540054


def wilson(k: int, n: int) -> list[float]:
    p = k / n
    denom = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / denom
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return [round((centre - half) * 100, 2), round((centre + half) * 100, 2)]


def rate(rows: list[dict]) -> dict:
    k = sum(r["lex_admit"] for r in rows)
    n = len(rows)
    return {"admitted": k, "n": n, "rate_pct": round(100 * k / n, 2), "wilson95_pct": wilson(k, n)}


def compute() -> dict:
    rows = [json.loads(line) for line in RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    stratum = [r for r in rows if r["primary_stratum"] == "COUNTERFACTUAL_FOLLOWING"]
    copies = [r for r in stratum if r["c4h_follows_distractor"] == 1]
    return {
        "description": (
            "Lexical admission on saved answers to packets with a planted typed distractor. "
            "Gold was used offline to construct and label the perturbation; the reader and "
            "the gate did not receive gold, and cases were not selected by gate outcome."
        ),
        "n_questions": len(rows),
        "counterfactual_following": rate(stratum),
        "explicit_distractor_copies": rate(copies),
        "refusals_in_stratum": sum(r["is_refusal"] for r in stratum),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="regenerate the stored summary")
    args = parser.parse_args()
    result = compute()
    if args.write:
        OUTPUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    else:
        assert json.loads(OUTPUT.read_text(encoding="utf-8")) == result, "stored distractor-audit summary differs"
    for key in ("counterfactual_following", "explicit_distractor_copies"):
        row = result[key]
        print(f"{key}: {row['admitted']}/{row['n']} ({row['rate_pct']}%), Wilson {row['wilson95_pct']}")
    print("Controlled distractor audit: OK")


if __name__ == "__main__":
    main()
