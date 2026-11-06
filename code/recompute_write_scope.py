"""Reproduce all-write contamination and the word-boundary sensitivity check.

The compact write audit includes intermediate and terminal indicators derived
from the saved trajectories. No model generation or human judgment is used.
Run without arguments to verify, or with --write to regenerate derived files.
"""
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ('react', 'planrag', 'autogen')
SEED = 20260807


def dumps(value):
    return json.dumps(value, indent=2) + '\n'


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def paired_bootstrap(delta):
    delta = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(SEED)
    means = delta[rng.integers(0, len(delta), (10000, len(delta)))].mean(axis=1)
    return {'point': float(delta.mean()),
            'ci95': list(map(float, np.quantile(means, [.025, .975]))),
            'unit': 'qid paired bootstrap', 'draws': 10000}


def apply_write_scope(files):
    path = ROOT / 'results/write_scope_per_qid.jsonl'
    audit = [json.loads(line) for line in path.open()]
    assert len(audit) == 6300
    by_key = {(x['family'], x['policy'], x['agent'], x['qid']): x for x in audit}
    assert len(by_key) == len(audit)
    summary = json.loads(files['strict_dynamic_summary_public.json'])
    registry = json.loads(files['results_registry_public.json'])
    final_contract = json.loads((ROOT / 'results/final_support_contract_revision_20261020.json').read_text())
    final_rows = [json.loads(line) for line in (ROOT / 'results' / final_contract['compact_artifact']).open()]
    final_by_key = {(x['agent'], x['qid']): x for x in final_rows}
    assert len(final_by_key) == 900
    revision = {'date': '2026-10-20', 'model_calls_added': 0,
                'scope': 'all reuse-eligible content writes, including the terminal memory write',
                'audit_artifact': path.name, 'audit_sha256': sha(path.read_text()),
                'primary_answer_scores_calls_and_admit_policy_unchanged': True,
                'final_support_terminal_metadata_normalized': True,
                'previous_aggregation': 'typed trajectories omitted terminal records; the final-only comparator additionally mixed post-repair and Vanilla terminal states',
                'changed_rows': {}}
    for family in ('comparator', 'component'):
        name = f'strict_{family}_per_qid.jsonl'
        values = [json.loads(line) for line in files[name].splitlines()]
        grouped = defaultdict(dict)
        for x in values:
            a = by_key[(family, x['policy'], x['agent'], x['qid'])]
            if family == 'comparator' and x['policy'] == 'final_support_retry':
                canonical = final_by_key[(x['agent'], x['qid'])]
                assert canonical['memory_policy'] == final_contract['policy']
                assert canonical['terminal_value_sha256'] == canonical['pre_repair_candidate_sha256']
                assert a['contamination'] == canonical['contamination'] == int(
                    canonical['intermediate_contamination'] or canonical['terminal_contamination'])
            else:
                assert a['contamination'] == int(a['previous_contamination'] or a['terminal_record_contamination'])
            assert x['raw_em'] == a['substring_em']
            label = family + '/' + x['policy']
            revision['changed_rows'].setdefault(label, 0)
            revision['changed_rows'][label] += int(x['contamination'] != a['contamination'])
            x['contamination'] = float(a['contamination'])
            grouped[x['policy']][(x['agent'], x['qid'])] = x
        for arm, obj in summary[family]['arms'].items():
            for scope in (*AGENTS, 'pooled'):
                selected = [x for (agent, qid), x in grouped[arm].items() if scope == 'pooled' or agent == scope]
                obj['metrics'][scope]['contamination'] = float(np.mean([x['contamination'] for x in selected]))
        treatment = 'cavec_mfc' if family == 'comparator' else 'full_admit'
        group = 'admit_minus_control' if family == 'comparator' else 'full_minus_other'
        for baseline, scopes in summary['paired'][family][group].items():
            for scope in (*AGENTS, 'pooled'):
                agents = AGENTS if scope == 'pooled' else (scope,)
                qids = sorted({qid for agent, qid in grouped[treatment]})
                delta = [np.mean([grouped[treatment][(agent, qid)]['contamination'] -
                                  grouped[baseline][(agent, qid)]['contamination'] for agent in agents])
                         for qid in qids]
                scopes[scope]['contamination'] = paired_bootstrap(delta)
        files[name] = ''.join(json.dumps(x) + '\n' for x in values)
        summary['public_verification'][family]['sha256'] = sha(files[name])
    summary['write_scope_revision'] = revision
    summary['contract'] += '; contamination includes intermediate and terminal memory writes'
    summary['final_support_contract_revision'] = final_contract
    final_arm = summary['comparator']['arms']['final_support_retry']
    final_arm['artifact'] = final_contract['normalized_artifact']
    final_arm['sha256'] = final_contract['normalized_sha256']
    final_arm['generation_source'] = final_contract['generation_source']
    final_arm['terminal_memory_policy'] = final_contract['policy']
    files['strict_dynamic_summary_public.json'] = dumps(summary)
    for family in ('comparator', 'component'):
        registry['provenance']['public_verification_sources']['dynamic_' + family] = summary['public_verification'][family]
    policy_map = {
        'Gate-only': ('component', 'gate_only'), 'Gate + RepairRead': ('component', 'gate_repair'),
        'Full ADMIT': ('component', 'full_admit'), 'Final-support + retry': ('comparator', 'final_support_retry'),
        'ADMIT fail-open': ('comparator', 'cavec_paper'), 'Hard-refuse': ('comparator', 'cavec_h1_hard_refuse'),
        'ADMIT MFC/OFO': ('comparator', 'cavec_mfc')}
    for entry in registry['entries']:
        if entry['agent'] != 'static':
            family, arm = policy_map[entry['policy']]
            entry['metrics'] = summary[family]['arms'][arm]['metrics'][entry['agent']]
            if arm == 'final_support_retry':
                entry['source']['artifact_id'] = final_contract['normalized_artifact']
                entry['source']['sha256'] = final_contract['normalized_sha256']
                entry['generation_source'] = final_contract['generation_source']
                entry['terminal_memory_policy'] = final_contract['policy']
    registry['metric_definitions']['contamination'] = revision['scope'] + ' contains at least one non-refusal value without lexical support in its exact reader-visible OCR/text'
    registry['contracts']['dynamic']['contamination_scope'] = revision['scope']
    registry['contracts']['dynamic']['final_support_memory_policy'] = final_contract['definition']
    delta = summary['paired']['comparator']['admit_minus_control']['cavec_paper']['pooled']['contamination']['point']
    registry['contracts']['dynamic']['utility_state_result'] = f'ADMIT vs fail-open: -0.67 pp substring EM (95% CI [-1.33,-0.11]) and {100*delta:.2f} pp all-write lexical trajectory contamination'
    registry['write_scope_revision'] = revision
    registry['final_support_contract_revision'] = final_contract
    files['results_registry_public.json'] = dumps(registry)
    # This sensitivity analysis preserves every primary EM value. Bootstrap
    # qids after averaging the three agent-level differences per qid.
    sensitivity = {'status': 'mechanical sensitivity analysis; not human semantic accuracy',
                   'definition': 'Primary normalization and bidirectional containment, with Unicode word boundaries at both ends of the contained string',
                   'population': 'all 300 matched query IDs and three agents; 900 trajectories per policy',
                   'bootstrap': {'draws': 10000, 'seed': SEED, 'unit': 'query ID, averaging three agents within query'},
                   'arms': {}, 'paired': {}, 'source_artifact': path.name, 'source_sha256': revision['audit_sha256']}
    comparator = {arm: {(x['agent'], x['qid']): x for x in audit if x['family'] == 'comparator' and x['policy'] == arm}
                  for arm in ('final_support_retry', 'cavec_paper', 'cavec_h1_hard_refuse', 'cavec_mfc')}
    for arm, values in comparator.items():
        sensitivity['arms'][arm] = {scope: {
            'n': len(selected := [x for (agent, qid), x in values.items() if scope == 'pooled' or agent == scope]),
            'substring_em': float(np.mean([x['substring_em'] for x in selected])),
            'word_boundary_em': float(np.mean([x['word_boundary_em'] for x in selected])),
            'changed_matches': sum(x['substring_em'] != x['word_boundary_em'] for x in selected)}
            for scope in (*AGENTS, 'pooled')}
    for baseline in ('cavec_paper', 'cavec_h1_hard_refuse', 'final_support_retry'):
        sensitivity['paired'][baseline] = {}
        for scope in (*AGENTS, 'pooled'):
            agents = AGENTS if scope == 'pooled' else (scope,)
            qids = sorted({q for a, q in comparator['cavec_mfc']})
            sensitivity['paired'][baseline][scope] = paired_bootstrap([
                np.mean([comparator['cavec_mfc'][(a, q)]['word_boundary_em'] - comparator[baseline][(a, q)]['word_boundary_em'] for a in agents])
                for q in qids])
    files['word_boundary_sensitivity_20261020.json'] = dumps(sensitivity)
    files['write_scope_correction_20261020.json'] = dumps(revision)
    return files


def main():
    from recompute_refusal_scores import generate
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    files = apply_write_scope(generate())
    for name, content in files.items():
        path = ROOT / 'results' / name
        if args.write:
            path.write_text(content)
        else:
            assert path.read_text() == content, 'recomputation mismatch: ' + name
    print(('WROTE' if args.write else 'PASS: reproduced') + ' all-write contamination and word-boundary sensitivity')


if __name__ == '__main__':
    main()
