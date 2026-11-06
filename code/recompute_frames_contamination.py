#!/usr/bin/env python3
"""Recompute FRAMES memory contamination from recorded direct-text support flags.

The memory list is append-only and contains the non-refusal values actually
stored during intermediate hops. The final output is not appended to memory.
A run is contaminated if any stored value has supported=False, meaning that
it does not occur in the OCR/text of its own read. This is distinct from the
final-answer overlap diagnostic and from transitive support through older
memory. The main table uses the default, direct-text admission arms; the
memory-aware variants are included here to make the setting distinction clear.
The additional ADMIT hop trace supplies withheld-write events and logical
reader-call counts. A matched-output hard-refusal replay changes only final
failed-repair outputs: intermediate admitted values and unresolved questions
remain identical, so subsequent planner and reader inputs are unchanged.
No model calls are required. Use --write to regenerate the audit summary.
"""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/multihop_final_check_20261102.jsonl"
OUTPUT = ROOT / "results/frames_memory_contamination_20261106.json"
TRACE = ROOT / "results/frames_admit_trace_20261106.jsonl"
TRACE_MANIFEST = ROOT / "results/frames_admit_trace_manifest_20261106.json"
ARMS = ("failopen", "admit", "failopen_memory_support", "admit_memory_support")
REFUSAL = "I do not have enough information to answer."


def runtime_and_refusal_replay(compact_rows):
    manifest = json.loads(TRACE_MANIFEST.read_text())
    payload = TRACE.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == manifest["trace_sha256"]
    traces = [json.loads(line) for line in payload.splitlines()]
    original = {r["qid"]: r for r in compact_rows
                if r["setting"] == "FRAMES" and r["arm"] == "admit"}
    assert len(traces) == len({r["qid"] for r in traces}) == len(original) == 824
    assert {r["qid"] for r in traces} == set(original)
    withheld = initial_reads = hop_repairs = final_repairs = 0
    replay_correct = replay_changes = lost_correct = 0
    for trace in traces:
        recorded = original[trace["qid"]]
        for key in ("answer", "em", "refused", "n_hops"):
            assert trace[key] == recorded[key]
        assert trace["final_source"] == recorded["final_check"]
        assert [{"value": m["value"], "supported": m["supported"]}
                for m in trace["memory"]] == recorded["memory"]
        reads = [h for h in trace["hops"] if "repair_calls" in h]
        assert len(reads) == trace["n_hops"]
        assert all(h["repair_calls"] in (0, 1) for h in reads)
        expected_memory = [(m["sub_q"], m["value"]) for m in trace["memory"]]
        admit_memory, refusal_memory = [], []
        admit_unresolved, refusal_unresolved = [], []
        for hop in reads:
            if hop["stored"] is None:
                # Neither STATUS(NO_COMMIT) nor STATUS(REFUSAL) contributes
                # answer content; both adapters record the same unresolved q.
                admit_unresolved.append(hop["sub_q"])
                refusal_unresolved.append(hop["sub_q"])
            else:
                assert hop["stored_supported"] is True
                assert hop["memory_source"] in ("candidate", "repair")
                value = (hop["sub_q"], hop["stored"])
                admit_memory.append(value)
                refusal_memory.append(value)
            assert admit_memory == refusal_memory
            assert admit_unresolved == refusal_unresolved
        assert admit_memory == refusal_memory == expected_memory
        withheld += any(h["memory_source"] == "quarantine_no_commit" for h in reads)
        initial_reads += len(reads) + 1  # hop reads plus final synthesis
        hop_repairs += sum(h["repair_calls"] for h in reads)
        assert trace["final_source"] in ("candidate", "repair", "fallback_baseline_unsupported")
        final_repairs += trace["final_source"] != "candidate"
        replace = trace["final_source"] == "fallback_baseline_unsupported"
        replay_answer = REFUSAL if replace else trace["answer"]
        replay_em = 0 if replace else trace["em"]
        if not replace:
            assert replay_answer == trace["answer"]
        replay_correct += replay_em
        replay_changes += replace
        lost_correct += trace["em"] if replace else 0
    calls = initial_reads + hop_repairs + final_repairs
    n = len(traces)
    return {
        "trace_manifest": manifest,
        "admit_runtime": {
            "questions": n,
            "runs_with_no_commit": withheld,
            "no_commit_pct": round(100 * withheld / n, 2),
            "initial_reader_calls": initial_reads,
            "intermediate_repair_calls": hop_repairs,
            "final_repair_calls": final_repairs,
            "reader_calls": calls,
            "reader_calls_per_question": round(calls / n, 3),
        },
        "hard_refusal_replay": {
            "questions": n,
            "correct_answers": replay_correct,
            "substring_em": round(100 * replay_correct / n, 2),
            "contaminated_runs": 0,
            "contamination_pct": 0.0,
            "outputs_replaced_with_refusal": replay_changes,
            "correct_outputs_replaced": lost_correct,
            "reader_calls": calls,
            "reader_calls_per_question": round(calls / n, 3),
            "admitted_values_and_unresolved_questions_unchanged": True,
        },
    }


def summarize():
    source = SOURCE.read_bytes()
    rows = [json.loads(line) for line in source.splitlines()]
    result = {
        "source": SOURCE.name,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "scope": "FRAMES intermediate stored non-refusal values; direct OCR/text support",
        "main_table_arms": ["failopen", "admit"],
        "arms": {},
    }
    for arm in ARMS:
        subset = [r for r in rows if r["setting"] == "FRAMES" and r["arm"] == arm]
        assert len(subset) == len({r["qid"] for r in subset}) == 824
        memory = [m for r in subset for m in r["memory"]]
        assert all(isinstance(m["supported"], bool) for m in memory)
        assert all(isinstance(m["value"], str) and m["value"].strip() for m in memory)
        contaminated = sum(any(not m["supported"] for m in r["memory"]) for r in subset)
        correct = sum(r["em"] for r in subset)
        result["arms"][arm] = {
            "questions": len(subset),
            "correct_answers": correct,
            "substring_em": round(100 * correct / len(subset), 2),
            "contaminated_runs": contaminated,
            "contamination_pct": round(100 * contaminated / len(subset), 2),
            "stored_values": len(memory),
            "values_without_direct_text_support": sum(not m["supported"] for m in memory),
        }
    result.update(runtime_and_refusal_replay(rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    result = summarize()
    if args.write:
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    else:
        assert json.loads(OUTPUT.read_text()) == result, "FRAMES contamination summary mismatch"
    for arm, row in result["arms"].items():
        print(f"{arm}: EM {row['substring_em']:.2f}; contamination "
              f"{row['contaminated_runs']}/{row['questions']} = {row['contamination_pct']:.2f}%")
    print("ADMIT runtime:", result["admit_runtime"])
    print("Hard-refusal matched-output replay:", result["hard_refusal_replay"])
    print("FRAMES direct-text memory contamination: OK")


if __name__ == "__main__":
    main()
