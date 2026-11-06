"""Run ADMIT's admission interface without a language model."""
from dataclasses import asdict
from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from admit_admission_engine import admit_write

for repaired_value in ("42 million", "not available"):
    result = admit_write(
        question="What was revenue?",
        commit={"answer_value": "41 million"},
        evidence_packet_text="Revenue was 42 million.",
        repair_read=lambda question, evidence, value=repaired_value: value,
    )
    print(json.dumps(asdict(result), indent=2))
