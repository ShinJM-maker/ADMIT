"""Minimal ADMIT example: one admitted write, one repaired write, one NO_COMMIT.

Run from the repository root: python examples/quickstart.py
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from admit_admission_engine import admit_write, is_factual_memory_record  # noqa: E402

EVIDENCE = "Invoice 2041. Total: $240. Ship date: 12 May. Plan: Basic; Billing: monthly."


def repair_read(answers):
    """Stand-in for one extra reader call on the same question and evidence."""
    def call(question, evidence):
        return answers[question]
    return call


REPAIRS = {
    "What is the ship date?": "12 May",
    "What is the monthly price?": "$90",
}

writes = [
    ("What is the total?", "$240"),
    ("What is the ship date?", "14 May"),
    ("What is the monthly price?", "$90"),
]

memory = []
for question, value in writes:
    d = admit_write(
        question=question,
        commit={"answer_value": value},
        evidence_packet_text=EVIDENCE,
        repair_read=repair_read(REPAIRS),
    )
    memory.append(d)
    shown = d.output_value + ("" if is_factual_memory_record(d) else "  [UNVERIFIED]")
    print(f"{question:30s} shown: {shown:22s} memory: {d.memory_status:10s} {d.memory_value}")

reusable = [d.memory_value for d in memory if is_factual_memory_record(d)]
print("ReuseEligible(M):", reusable)
