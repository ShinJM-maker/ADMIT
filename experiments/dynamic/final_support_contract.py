"""Canonical final-only control: check output without editing Vanilla memory."""
from __future__ import annotations

import copy
import re
from typing import Any

POLICY = 'preserve_all_vanilla_memory_output_only_repair'


def normalize(value: Any) -> str:
    return re.sub(r'\s+', ' ', str(value or '').lower().strip())


def is_refusal(value: Any) -> bool:
    return not normalize(value) or 'do not have enough information' in normalize(value)


def make_record(value: str, supported: bool, packet_hash: str | None) -> dict:
    if is_refusal(value):
        return {'record_type': 'STATUS', 'status': 'REFUSAL', 'value': value}
    return {'record_type': 'FACT', 'status': 'COMMIT', 'value': value,
            'lexically_supported': bool(supported), 'evidence_packet_sha256': packet_hash}


def intermediate_values(result: dict, agent: str) -> list[str]:
    if agent == 'react':
        return [x['committed_answer'] for x in result.get('scratchpad', [])]
    if agent == 'planrag':
        return [x['committed_answer'] for x in result.get('sub_logs', [])]
    return [x['content'] for x in result.get('conversation', []) if x.get('role') == 'reader_commit']


def canonicalize(row: dict, vanilla: dict | None = None,
                 reader_writes: dict | None = None) -> dict:
    """Return a derived row; never mutate the saved generation artifact.

    Typed archives retain the exact original final candidate and its check
    branch. Legacy archives use their preexisting, per-write exact-reader
    annotations. The historical yes/no carve-out is never an evidence test.
    """
    out = copy.deepcopy(row)
    result = out['result']
    if result.get('terminal_memory_contract') == POLICY:
        return out
    hook = result['final_hook_log']
    if 'base_memory_records' in result:
        value = hook['candidate']
        source = hook['final_source']
        assert source in {'refusal', 'candidate_supported', 'repair_supported', 'fallback_vanilla_raw_unsupported'}
        assert is_refusal(value) == (source == 'refusal')
        result['archived_posthoc_terminal_record'] = result['final_memory_record']
        result['final_memory_record'] = make_record(
            value, source == 'candidate_supported', result['final_evidence_packet_sha256'])
    else:
        assert vanilla is not None and reader_writes is not None
        prefix = []
        for i, value in enumerate(intermediate_values(vanilla, row['agent'])):
            if is_refusal(value):
                prefix.append(make_record(value, False, None))
            else:
                w = reader_writes[i]
                assert w['admitted_value'] == value
                prefix.append(make_record(value, w['reader_substring_pass'], w['reader_pack_sha256']))
        value = vanilla['final_committed_answer']
        assert normalize(hook.get('candidate', value)) == normalize(value)
        if is_refusal(value):
            terminal = make_record(value, False, None)
        else:
            w = reader_writes['final']
            assert w['admitted_value'] == value
            terminal = make_record(value, w['reader_substring_pass'], w['reader_pack_sha256'])
        result['base_memory_records'] = prefix
        result['final_memory_record'] = terminal
    result['terminal_memory_contract'] = POLICY
    assert result['final_committed_answer'] == row['result']['final_committed_answer']
    assert result['calls_q'] == row['result']['calls_q']
    return out
