"""Validate the public ADMIT supplementary bundle."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_close(observed: float, expected: float, label: str) -> None:
    assert abs(observed - expected) < 1e-12, (
        f"{label}: observed {observed}, expected {expected}"
    )


def validate_manifest() -> None:
    manifest = ROOT / "MANIFEST.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        observed = sha256(path)
        assert observed == digest, f"checksum mismatch: {relative}"


def validate_contract() -> None:
    contract = read_json(ROOT / "configs" / "execution_contract.json")
    static = contract["static"]
    assert static["released_validation_n"] == 2441
    assert static["candidate_complete_n"] == 2324
    assert static["excluded_incomplete_candidate_n"] == 117
    assert static["calibration_partition_n"] == 1162
    assert static["reporting_partition_n"] == 1162
    assert static["untouched_held_out_claim"] is False
    assert "dynamic-aligned" in static["metric"]
    assert static["support_router_hyperparameters"]["final_estimators"] == 2
    assert "typed status without RepairRead" in static["definitive_routing"]
    assert "no causal compute effect" in contract["dynamic"]["router_scope"]
    assert contract["dynamic"]["public_memory"] == "typed FACT/STATUS records"
    assert "FACT projection" in contract["dynamic"]["factual_consumer_rule"]
    admission = contract["admission"]
    assert admission["verifier"] == "strict normalized substring"
    assert admission["non_refusal_bypass_enabled"] is False
    assert admission["refusal_action"].startswith("record as non-factual")
    assert "ignore REFUSAL and NO_COMMIT" in admission["factual_consumer_rule"]
    assert admission["repair_calls_per_proposed_write_max"] == 1
    assert "terminal" in contract["metrics"]["contamination"]
    assert contract['dynamic']['final_support_memory_policy'] == 'preserve_all_vanilla_memory_output_only_repair'


def validate_registry() -> None:
    registry = read_json(ROOT / "results" / "results_registry_public.json")
    assert registry["schema_version"] == "1.1"
    assert registry["entries"]
    assert all("seed" not in entry for entry in registry["entries"])
    allowed = {"main", "main-ablation", "main_ablation", "diagnostic"}
    assert all(entry["status"] in allowed for entry in registry["entries"])

    static_contract = registry["contracts"]["static"]
    assert static_contract["released_validation_n"] == 2441
    assert static_contract["candidate_complete_n"] == 2324
    assert static_contract["excluded_incomplete_candidate_rows"] == 117
    assert static_contract["untouched_held_out_claim"] is False
    assert "dynamic-aligned" in static_contract["EM"]

    public_sources = registry["provenance"]["public_verification_sources"]
    assert set(public_sources) == {"static", "dynamic_comparator", "dynamic_component"}
    for source in public_sources.values():
        path = ROOT / "results" / source["artifact_id"]
        assert path.is_file(), f"missing public verification source: {path.name}"
        assert sha256(path) == source["sha256"], f"public source hash: {path.name}"

    aggregate = read_json(ROOT / "results" / "static_deployable_router_results.json")
    aggregate_contract = aggregate["contract"]
    assert aggregate_contract["released_validation_qids"] == 2441
    assert aggregate_contract["candidate_complete_qids"] == 2324
    assert aggregate_contract["excluded_incomplete_candidate_qids"] == 117
    aggregate_metrics = aggregate["metrics"]
    static_entries = [entry for entry in registry["entries"] if entry["agent"] == "static"]
    assert {entry["policy"] for entry in static_entries} == set(aggregate_metrics)
    for entry in static_entries:
        expected = aggregate_metrics[entry["policy"]]
        observed = entry["metrics"]
        assert observed["EM"] == expected["EM"]
        assert observed["F1"] == expected["F1"]
        assert observed["ANLS"] == expected["ANLS"]
        assert observed["calls_per_q"] == expected["calls"]
        assert observed["no_commit_rate"] == expected["no_commit"]

    dynamic_summary = read_json(ROOT / "results" / "strict_dynamic_summary_public.json")
    assert dynamic_summary["public_verification"]["comparator"] == public_sources["dynamic_comparator"]
    assert dynamic_summary["public_verification"]["component"] == public_sources["dynamic_component"]
    policy_to_arm = {
        "Gate-only": ("component", "gate_only"),
        "Gate + RepairRead": ("component", "gate_repair"),
        "Full ADMIT": ("component", "full_admit"),
        "Final-support + retry": ("comparator", "final_support_retry"),
        "ADMIT fail-open": ("comparator", "cavec_paper"),
        "Hard-refuse": ("comparator", "cavec_h1_hard_refuse"),
        "ADMIT MFC/OFO": ("comparator", "cavec_mfc"),
    }
    assert "typed FACT/STATUS" in dynamic_summary["contract"]
    assert dynamic_summary["runtime_audit"]["typed_trajectory_rows"] == 3041
    assert dynamic_summary["runtime_audit"]["contract_equivalent_reuse_pairs"] == {
        "comparator": 478,
        "component": 449,
    }
    for entry in registry["entries"]:
        if entry["agent"] == "static":
            continue
        family, arm = policy_to_arm[entry["policy"]]
        frozen = dynamic_summary[family]["arms"][arm]
        assert entry["metrics"] == frozen["metrics"][entry["agent"]]
        assert entry["typed_rerun_pairs"] == frozen["typed_rerun_pairs"]
        assert entry["source"]["artifact_id"] == frozen["artifact"]
        assert entry["source"]["sha256"] == frozen["sha256"]


def validate_static_per_qid() -> None:
    path = ROOT / "results" / "static_deployable_router_per_qid.jsonl"
    rows = read_jsonl(path)
    assert len(rows) == 1162
    assert len({row["qid"] for row in rows}) == 1162
    assert all("evidence" not in row and "packet_text" not in row for row in rows)
    assert all(set(row["policies"]) == {
        "Vanilla single-pass", "Gate-only", "ADMIT",
        "Learned-router diagnostic", "Final-support + retry", "Every-q repair"
    } for row in rows)

    aggregate = read_json(ROOT / "results" / "static_deployable_router_results.json")
    metric_map = {
        "EM": "EM", "F1": "F1", "ANLS": "ANLS",
        "unsupNR_strict": "unsupNR_strict",
        "value_supported_EM": "value_supported_EM",
        "calls": "calls", "no_commit": "no_commit",
    }
    for policy, expected in aggregate["metrics"].items():
        assert expected["n"] == len(rows)
        for source_metric, target_metric in metric_map.items():
            observed = sum(row["policies"][policy][source_metric] for row in rows) / len(rows)
            assert_close(observed, expected[target_metric], f"static {policy} {target_metric}")

    static_pairs = {
        "ADMIT_minus_Vanilla single-pass": ("ADMIT", "Vanilla single-pass"),
        "ADMIT_minus_Final-support + retry": ("ADMIT", "Final-support + retry"),
        "ADMIT_minus_Every-q repair": ("ADMIT", "Every-q repair"),
        "Learned-router_diagnostic_minus_ADMIT": ("Learned-router diagnostic", "ADMIT"),
    }
    paired_metrics = ("EM", "F1", "ANLS", "value_supported_EM", "calls", "no_commit")
    for contrast, (treatment, baseline) in static_pairs.items():
        expected = aggregate["paired_deltas"][contrast]
        for metric in paired_metrics:
            observed = sum(
                row["policies"][treatment][metric] - row["policies"][baseline][metric]
                for row in rows
            ) / len(rows)
            assert_close(observed, expected[metric]["mean"], f"static paired {contrast} {metric}")

    # The final-only control is an answer-surface identification control:
    # it has the same emitted answer and answer metrics as ADMIT for every qid.
    parity_fields = (
        "answer", "EM", "F1", "ANLS", "support", "unsupNR_strict",
        "value_supported_EM", "calls",
    )
    for row in rows:
        for metrics in row["policies"].values():
            if not metrics["answer"].strip() or "do not have enough information" in metrics["answer"].lower():
                assert metrics["EM"] == 0, "static refusal credited as correct"
        admit = row["policies"]["ADMIT"]
        final_only = row["policies"]["Final-support + retry"]
        assert all(admit[field] == final_only[field] for field in parity_fields), (
            f"static answer-surface mismatch: {row['qid']}"
        )


def validate_per_qid(path: Path, expected_count: int, require_zero: bool) -> None:
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        assert set(row) == {
            "qid", "agent", "policy", "raw_em", "contamination",
            "no_commit", "calls_q", "final_refusal"
        }
        assert not (row["final_refusal"] and row["raw_em"]), "refusal credited as correct"
        if require_zero:
            assert row["contamination"] == 0
        count += 1
    assert count == expected_count, f"expected {expected_count} rows, observed {count}"


def validate_dynamic_per_qid() -> None:
    summary = read_json(ROOT / "results" / "strict_dynamic_summary_public.json")
    specs = {
        "component": {
            "path": ROOT / "results" / "strict_component_per_qid.jsonl",
            "treatment": "full_admit",
            "baselines": ("gate_only", "gate_repair"),
            "paired_group": "full_minus_other",
        },
        "comparator": {
            "path": ROOT / "results" / "strict_comparator_per_qid.jsonl",
            "treatment": "cavec_mfc",
            "baselines": (
                "final_support_retry", "cavec_paper", "cavec_h1_hard_refuse"
            ),
            "paired_group": "admit_minus_control",
        },
    }
    metrics = ("raw_em", "contamination", "no_commit", "calls_q", "final_refusal")
    paired_metrics = ("raw_em", "contamination", "no_commit", "calls_q")

    for family, spec in specs.items():
        rows = read_jsonl(spec["path"])
        arms = summary[family]["arms"]
        assert {row["policy"] for row in rows} == set(arms)
        assert len({(row["qid"], row["agent"], row["policy"]) for row in rows}) == len(rows)

        by_arm_agent = defaultdict(list)
        for row in rows:
            by_arm_agent[(row["policy"], row["agent"])].append(row)

        for arm, frozen in arms.items():
            pooled = []
            for agent in ("react", "planrag", "autogen"):
                subset = by_arm_agent[(arm, agent)]
                assert len(subset) == 300
                assert len({row["qid"] for row in subset}) == 300
                pooled.extend(subset)
                expected = frozen["metrics"][agent]
                assert expected["n"] == 300
                for metric in metrics:
                    observed = sum(row[metric] for row in subset) / len(subset)
                    assert_close(observed, expected[metric], f"{family} {arm} {agent} {metric}")
            assert len(pooled) == 900
            expected = frozen["metrics"]["pooled"]
            assert expected["n"] == 900
            for metric in metrics:
                observed = sum(row[metric] for row in pooled) / len(pooled)
                assert_close(observed, expected[metric], f"{family} {arm} pooled {metric}")

        treatment = spec["treatment"]
        paired = summary["paired"][family][spec["paired_group"]]
        for baseline in spec["baselines"]:
            for agent in ("react", "planrag", "autogen", "pooled"):
                agents = ("react", "planrag", "autogen") if agent == "pooled" else (agent,)
                treatment_rows = {
                    (row["agent"], row["qid"]): row
                    for selected_agent in agents
                    for row in by_arm_agent[(treatment, selected_agent)]
                }
                baseline_rows = {
                    (row["agent"], row["qid"]): row
                    for selected_agent in agents
                    for row in by_arm_agent[(baseline, selected_agent)]
                }
                assert treatment_rows.keys() == baseline_rows.keys()
                n = len(treatment_rows)
                assert n == (900 if agent == "pooled" else 300)
                for metric in paired_metrics:
                    observed = sum(
                        treatment_rows[key][metric] - baseline_rows[key][metric]
                        for key in treatment_rows
                    ) / n
                    expected = paired[baseline][agent][metric]["point"]
                    assert_close(
                        observed, expected,
                        f"paired {family} {treatment}-{baseline} {agent} {metric}",
                    )


def validate_engine() -> None:
    path = ROOT / "code" / "admit_admission_engine.py"
    spec = importlib.util.spec_from_file_location("admit_engine", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._self_test()


def validate_write_scope() -> None:
    """Regress the omitted-final-write case against the archived compact audit."""
    path = ROOT / "results/write_scope_per_qid.jsonl"
    audit = read_jsonl(path)
    assert len(audit) == 6300
    indexed = {(x['family'], x['policy'], x['agent'], x['qid']): x for x in audit}
    assert len(indexed) == len(audit)
    final_revision = read_json(ROOT / 'results/final_support_contract_revision_20261020.json')
    final_path = ROOT / 'results' / final_revision['compact_artifact']
    assert sha256(final_path) == final_revision['compact_sha256']
    final_audit = {(x['agent'], x['qid']): x for x in read_jsonl(final_path)}
    assert len(final_audit) == 900
    assert {x['memory_policy'] for x in final_audit.values()} == {final_revision['policy']}
    assert all(x['terminal_value_sha256'] == x['pre_repair_candidate_sha256'] for x in final_audit.values())
    assert sum(x['source_format'] == 'typed' for x in final_audit.values()) == 422
    assert sum(x['source_format'] == 'legacy' for x in final_audit.values()) == 478
    changed = defaultdict(int)
    for family in ('comparator', 'component'):
        for row in read_jsonl(ROOT / 'results' / f'strict_{family}_per_qid.jsonl'):
            a = indexed[(family, row['policy'], row['agent'], row['qid'])]
            if family == 'comparator' and row['policy'] == 'final_support_retry':
                f = final_audit[(row['agent'], row['qid'])]
                assert f['memory_policy'] == 'preserve_all_vanilla_memory_output_only_repair'
                assert a['contamination'] == f['contamination'] == int(f['intermediate_contamination'] or f['terminal_contamination'])
                assert a['terminal_record_contamination'] == f['terminal_contamination']
            else:
                assert a['contamination'] == int(a['previous_contamination'] or a['terminal_record_contamination'])
            assert a['has_terminal_record'] or not a['terminal_record_contamination']
            assert row['contamination'] == a['contamination']
            assert row['raw_em'] == a['substring_em']
            assert 0 <= a['word_boundary_em'] <= a['substring_em'] <= 1
            changed[family + '/' + row['policy']] += int(a['contamination'] != a['previous_contamination'])
    # These actual terminal-only cases must never silently disappear again.
    assert changed['comparator/cavec_paper'] == 21
    assert changed['comparator/final_support_retry'] == 29
    summary = read_json(ROOT / 'results/strict_dynamic_summary_public.json')
    registry = read_json(ROOT / 'results/results_registry_public.json')
    assert summary['final_support_contract_revision'] == registry['final_support_contract_revision'] == final_revision
    assert sum(x['contamination'] for x in final_audit.values()) == 177
    # An unsupported pre-repair terminal value stays in this output-only
    # control even when the repair changes the user-visible answer.
    repaired = final_audit[('planrag', '0ffc318235d60a9934bfec7afd36c9a7')]
    assert repaired['terminal_value_sha256'] != repaired['posthoc_output_sha256']
    assert repaired['terminal_contamination'] == repaired['contamination'] == 1
    for agent, qid in [('react', 'b28d5efd4c17fb8b96986823543c1fc1'),
                       ('planrag', '540f2fc8563638b9212d69e22336094b'),
                       ('autogen', 'b28d5efd4c17fb8b96986823543c1fc1')]:
        assert final_audit[(agent, qid)]['terminal_contamination'] == 1
    revision = summary['write_scope_revision']
    assert sha256(path) == revision['audit_sha256']
    assert registry['write_scope_revision'] == revision
    assert dict(changed) == revision['changed_rows']
    sensitivity = read_json(ROOT / 'results/word_boundary_sensitivity_20261020.json')
    assert sha256(path) == sensitivity['source_sha256']
    for arm, scopes in sensitivity['arms'].items():
        for scope, expected in scopes.items():
            selected = [x for x in audit if x['family'] == 'comparator' and x['policy'] == arm
                        and (scope == 'pooled' or x['agent'] == scope)]
            assert len(selected) == expected['n']
            for metric in ('substring_em', 'word_boundary_em'):
                assert_close(sum(x[metric] for x in selected) / len(selected), expected[metric],
                             f'word-boundary {arm}/{scope}/{metric}')
    assert sensitivity['arms']['cavec_mfc']['pooled']['changed_matches'] == 3
    assert sensitivity['arms']['cavec_paper']['pooled']['changed_matches'] == 5


def validate_public_text() -> None:
    forbidden = (
        "2026" + "0504",
        "/mnt/" + "raid6",
        "tlswnd" + "als13",
        "/data" + "_",
    )
    suffixes = {".md", ".py", ".json", ".jsonl", ".sha256"}
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix not in suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        assert not any(token in text for token in forbidden), f"private token: {path}"


if __name__ == "__main__":
    validate_contract()
    validate_registry()
    validate_per_qid(ROOT / "results" / "strict_component_per_qid.jsonl", 2700, True)
    validate_per_qid(ROOT / "results" / "strict_comparator_per_qid.jsonl", 3600, False)
    validate_static_per_qid()
    validate_dynamic_per_qid()
    validate_write_scope()
    validate_engine()
    from answer_metrics import regression_checks
    regression_checks()
    import validate_prompt_contract
    validate_prompt_contract.main()
    validate_public_text()
    validate_manifest()
    print("ADMIT supplementary validation: OK")
