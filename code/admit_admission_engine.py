"""Reference implementation of ADMIT's definitive strict lexical boundary.

The module is deliberately independent of a model-serving stack. Callers pass
the exact reader-visible post-truncation OCR/text packet and, optionally, a
one-shot RepairRead callback. Under the default MFC/OFO policy, every admitted
non-refusal content value must pass ``lexical_support``. The diagnostic
fail-open control intentionally stores fallback values after failed admission.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import re
from typing import Any, Callable, Mapping, Optional


NO_COMMIT = "[NO_COMMIT]"
REFUSAL_PHRASE = "do not have enough information"
FACTUAL_MEMORY_STATUSES = frozenset({"COMMIT", "COMMIT_DIAGNOSTIC_FAIL_OPEN"})
RepairRead = Callable[[str, str], str]


def normalize(text: Any) -> str:
    """Apply the normalization specified in the paper."""

    return re.sub(r"\s+", " ", str(text or "").lower().strip())


def is_refusal(value: Any) -> bool:
    normalized = normalize(value)
    return not normalized or REFUSAL_PHRASE in normalized


def lexical_support(value: Any, evidence_packet_text: str) -> bool:
    """Return true only for a supported non-refusal value."""

    normalized = normalize(value)
    if not normalized or REFUSAL_PHRASE in normalized:
        return False
    return normalized in normalize(evidence_packet_text)


def extract_answer_value(commit: Mapping[str, Any]) -> str:
    """Extract the auditable value from a structured public commit."""

    if "answer_value" not in commit or not isinstance(commit["answer_value"], str):
        raise ValueError("commit must contain a string answer_value")
    return commit["answer_value"].strip()


@dataclass(frozen=True)
class AdmissionDecision:
    output_value: str
    memory_value: str
    grounding_status: str
    memory_status: str
    source: str
    repair_calls: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_factual_memory_record(decision: AdmissionDecision) -> bool:
    """Return whether factual-memory consumers may reuse this record."""

    return decision.memory_status in FACTUAL_MEMORY_STATUSES


def admit_write(
    *,
    question: str,
    commit: Mapping[str, Any],
    evidence_packet_text: str,
    repair_read: Optional[RepairRead] = None,
    repair_selected: bool = True,
    fallback_output: Optional[str] = None,
    repair_fail_policy: str = "mfc_ofo",
) -> AdmissionDecision:
    """Apply strict lexical admission at one public memory-write boundary.

    ``question`` is the task question attached to this write (for example,
    the current PlanRAG subquestion), not necessarily the overall task query.
    ``repair_read`` is called at most once and must read that question and
    evidence packet. Refusals are retained as non-factual status records.
    ``mfc_ofo`` emits a tentative fallback but writes ``NO_COMMIT``.
    ``hard_refuse`` records a refusal at both surfaces. ``fail_open`` is
    supplied only as the paper's diagnostic control.

    A missing or non-string answer value follows the repair path. For such
    malformed commits, callers should supply the original reader output in
    ``fallback_output``; otherwise the conversational fallback is empty.
    """

    if repair_fail_policy not in {"mfc_ofo", "hard_refuse", "fail_open"}:
        raise ValueError("unknown repair_fail_policy")

    try:
        candidate = extract_answer_value(commit)
        parse_failed = False
    except ValueError:
        candidate = ""
        parse_failed = True
    if not parse_failed and is_refusal(candidate):
        return AdmissionDecision(
            output_value=candidate,
            memory_value=candidate,
            grounding_status="REFUSAL",
            memory_status="REFUSAL",
            source="candidate_refusal",
            repair_calls=0,
        )
    if not parse_failed and lexical_support(candidate, evidence_packet_text):
        return AdmissionDecision(
            output_value=candidate,
            memory_value=candidate,
            grounding_status="VERIFIED_LEXICAL",
            memory_status="COMMIT",
            source="candidate",
            repair_calls=0,
        )

    repaired: Optional[str] = None
    repair_calls = 0
    if repair_selected and repair_read is not None:
        repaired = str(repair_read(question, evidence_packet_text) or "").strip()
        repair_calls = 1
        if not is_refusal(repaired) and lexical_support(repaired, evidence_packet_text):
            return AdmissionDecision(
                output_value=repaired,
                memory_value=repaired,
                grounding_status="VERIFIED_LEXICAL",
                memory_status="COMMIT",
                source="repair",
                repair_calls=repair_calls,
            )

    fallback = str(fallback_output if fallback_output is not None else candidate)
    if repair_fail_policy == "fail_open":
        return AdmissionDecision(
            output_value=fallback,
            memory_value=fallback,
            grounding_status="UNVERIFIED",
            memory_status="COMMIT_DIAGNOSTIC_FAIL_OPEN",
            source="fallback",
            repair_calls=repair_calls,
        )

    if repair_fail_policy == "hard_refuse":
        refusal = "I do not have enough information to answer"
        return AdmissionDecision(
            output_value=refusal,
            memory_value=refusal,
            grounding_status="REFUSAL",
            memory_status="REFUSAL",
            source="hard_refuse",
            repair_calls=repair_calls,
        )
    return AdmissionDecision(
        output_value=fallback,
        memory_value=NO_COMMIT,
        grounding_status="UNVERIFIED",
        memory_status="NO_COMMIT",
        source="repair_fail",
        repair_calls=repair_calls,
    )


def _self_test() -> None:
    packet = "The incident report identifies Brighton Avenue as the location."
    # Malformed structured writes must reach repair, never the initial-refusal
    # branch. Preserve a supplied fallback without promoting it under MFC/OFO.
    for malformed in ({}, {"answer_value": None}, {"answer_value": 1995}):
        for repair_value in ("Brighton Avenue", "North Road",
                             "I do not have enough information to answer"):
            for policy in ("mfc_ofo", "hard_refuse", "fail_open"):
                calls = []

                def repair(q: str, b: str) -> str:
                    calls.append((q, b))
                    return repair_value

                result = admit_write(
                    question="Where was the incident?", commit=malformed,
                    evidence_packet_text=packet, repair_read=repair,
                    fallback_output="North Road", repair_fail_policy=policy,
                )
                assert calls == [("Where was the incident?", packet)]
                assert result.repair_calls == 1
                if repair_value == "Brighton Avenue":
                    assert result.memory_status == "COMMIT"
                    assert result.memory_value == repair_value
                elif policy == "mfc_ofo":
                    assert result.memory_value == NO_COMMIT
                    assert result.output_value == "North Road"
                    assert not is_factual_memory_record(result)
                elif policy == "hard_refuse":
                    assert result.memory_status == "REFUSAL"
                    assert not is_factual_memory_record(result)
                else:
                    assert result.memory_status == "COMMIT_DIAGNOSTIC_FAIL_OPEN"
                    assert result.memory_value == "North Road"

        missing = admit_write(question="Where?", commit=malformed,
                             evidence_packet_text=packet)
        assert missing.memory_status == "NO_COMMIT"
        assert missing.output_value == ""
        assert missing.repair_calls == 0

    empty = admit_write(question="Where?", commit={"answer_value": " "},
                        evidence_packet_text=packet)
    assert empty.memory_status == "REFUSAL"
    assert empty.repair_calls == 0

    supported = admit_write(
        question="Where was the incident?",
        commit={"answer_value": "Brighton Avenue"},
        evidence_packet_text=packet,
    )
    assert supported.memory_value == "Brighton Avenue"
    assert is_factual_memory_record(supported)

    quarantined = admit_write(
        question="Where was the incident?",
        commit={"answer_value": "North Road"},
        evidence_packet_text=packet,
        fallback_output="North Road",
    )
    assert quarantined.memory_value == NO_COMMIT
    assert quarantined.output_value == "North Road"
    assert not is_factual_memory_record(quarantined)

    repaired = admit_write(
        question="Where was the incident?",
        commit={"answer_value": "North Road"},
        evidence_packet_text=packet,
        repair_read=lambda _q, _b: "Brighton Avenue",
    )
    assert repaired.memory_value == "Brighton Avenue"
    assert repaired.repair_calls == 1

    refusal = admit_write(
        question="Where was the incident?",
        commit={"answer_value": "I do not have enough information to answer"},
        evidence_packet_text=packet,
    )
    assert refusal.memory_status == "REFUSAL"
    assert refusal.memory_value != NO_COMMIT
    assert not is_factual_memory_record(refusal)

    hard_refusal = admit_write(
        question="Where was the incident?",
        commit={"answer_value": "North Road"},
        evidence_packet_text=packet,
        repair_fail_policy="hard_refuse",
    )
    assert hard_refusal.memory_status == "REFUSAL"
    assert is_refusal(hard_refusal.memory_value)
    assert not is_factual_memory_record(hard_refusal)

    repair_refusal = admit_write(
        question="Where was the incident?",
        commit={"answer_value": "North Road"},
        evidence_packet_text=packet,
        repair_read=lambda _q, _b: "I do not have enough information to answer",
        fallback_output="North Road",
    )
    assert repair_refusal.output_value == "North Road"
    assert repair_refusal.memory_status == "NO_COMMIT"
    assert repair_refusal.repair_calls == 1


if __name__ == "__main__":
    _self_test()
    print(json.dumps({"status": "ok", "contract": "strict_lexical"}))
