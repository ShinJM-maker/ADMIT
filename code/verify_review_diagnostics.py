"""Recompute the October 22 saved-output diagnostics from released compact records.

This verifies aggregate arithmetic and paired intervals. It does not recreate
model generations or certify the full evidence of records whose packet was
reconstructed without a stored hash. No model or network access is used.
"""
from pathlib import Path
from collections import Counter
import json
import re
import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'results'
AGENTS=('react','planrag','autogen')

def rows(name):return [json.loads(s) for s in (ROOT/name).open() if s.strip()]
def load(name):return json.loads((ROOT/name).read_text())
def close(a,b):assert abs(a-b)<1e-12,(a,b)

def bootstrap(v):
    v=np.asarray(v,dtype=float)
    idx=np.random.default_rng(20261022).integers(0,len(v),(10000,len(v)))
    return dict(point=float(v.mean()),ci95=[float(x) for x in np.quantile(v[idx].mean(axis=1),[.025,.975])])

def paired(v,a,b,field):
    d={(x['policy'],x['agent'],x['qid']):x for x in v}
    return bootstrap([np.mean([d[(a,g,q)][field]-d[(b,g,q)][field] for g in AGENTS]) for q in sorted({x['qid'] for x in v})])

def check_interval(a,b):
    close(a['point'],b['point'])
    for x,y in zip(a['ci95'],b['ci95']):close(x,y)

def main():
    v=rows('final_write_diagnostics_per_qid_20261022.jsonl')
    expected=load('final_write_diagnostics_20261022.json')
    old={(x['policy'],x['agent'],x['qid']):x for x in rows('strict_comparator_per_qid.jsonl')}
    assert len(v)==3600 and len({(x['policy'],x['agent'],x['qid']) for x in v})==3600
    for x in v:
        counts=sum(x[k] for k in ('original_admission','repaired_admission','initial_refusal','failed_repair_withholding','failed_repair_admission'))
        assert counts==x['proposed_writes']
        assert x['admitted_writes']==x['original_admission']+x['repaired_admission']+x['failed_repair_admission']
        y=old[(x['policy'],x['agent'],x['qid'])]
        assert x['substring_em']==y['raw_em']
        assert max(x['intermediate_contamination'],x['terminal_contamination'])==y['contamination']
        assert max(x['intermediate_no_commit'],x['terminal_no_commit'])==y['no_commit']
    for policy,scopes in expected['policies'].items():
        for scope,metrics in scopes.items():
            selected=[x for x in v if x['policy']==policy and (scope=='pooled' or x['agent']==scope)]
            assert len(selected)==metrics['n']
            for k,val in metrics.items():
                if k in ('n','admitted_fraction_of_writes','yesno_n','yesno_accuracy'):continue
                close(float(np.mean([x[k] for x in selected])),val)
            close(sum(x['admitted_writes'] for x in selected)/sum(x['proposed_writes'] for x in selected),metrics['admitted_fraction_of_writes'])
    for comparator,metrics in expected['paired'].items():
        for field,interval in metrics.items():check_interval(paired(v,'cavec_mfc',comparator,field),interval)
    disagreements=load('answer_disagreements_20261022.json')
    assert len(disagreements)==8
    assert sum(x['admit_substring_em']>x['fail_open_substring_em'] for x in disagreements)==1
    assert sum(x['admit_strict_em']!=x['fail_open_strict_em'] for x in disagreements)==1
    component=rows('strict_component_per_qid.jsonl')
    for field,interval in expected['component_paired'].items():check_interval(paired(component,'gate_repair','gate_only',field),interval)
    availability=rows('component_write_availability_per_qid_20261022.jsonl')
    availability_summary=load('component_write_availability_20261022.json')['policies']
    component_lookup={(x['policy'],x['agent'],x['qid']):x for x in component}
    assert len(availability)==len({(x['policy'],x['agent'],x['qid']) for x in availability})==1800
    for x in availability:
        y=component_lookup[(x['policy'],x['agent'],x['qid'])]
        assert x['admitted_writes']+x['no_commit_writes']+x['refusal_writes']==x['proposed_writes']
        assert int(x['no_commit_writes']>0)==y['no_commit']
        assert x['proposed_writes']<=x['reader_calls']==y['calls_q']<=2*x['proposed_writes']
        if x['policy']=='gate_only':assert x['reader_calls']==x['proposed_writes']
    for policy,metrics in availability_summary.items():
        selected=[x for x in availability if x['policy']==policy]
        assert len(selected)==metrics['n']==900
        for field in ('proposed_writes','admitted_writes','no_commit_writes','refusal_writes','reader_calls'):
            close(sum(x[field] for x in selected),metrics[field])
        assert sum(x['no_commit_writes']>0 for x in selected)==metrics['any_no_commit']
        close(metrics['admitted_fraction'],metrics['admitted_writes']/metrics['proposed_writes'])
    g=rows('gate_span_per_write_20261022.jsonl');summary=load('gate_span_summary_20261022.json')
    assert len(g)==summary['groups']['all']['n']
    assert len(g)+len(summary['unresolved_packet_records'])==summary['admitted_write_population']==1321
    for name,predicate in [('all',lambda x:True),('intermediate',lambda x:not x['terminal']),('terminal',lambda x:x['terminal']),('yesno',lambda x:x['yesno']),('numeric',lambda x:x['numeric']),('short_numeric',lambda x:x['short_numeric'])]:
        selected=[x for x in g if predicate(x)];observed=dict(n=len(selected),interior_only=sum(x['interior_only'] for x in selected),numeric_interior_only=sum(x['numeric_interior_only'] for x in selected))
        assert observed==summary['groups'][name]
    for x in load('interior_match_examples_20261022.json'):
        context=x['normalized_matching_span']['context'];value=x['value']
        assert value in context and not re.search(r'(?<!\w)'+re.escape(value)+r'(?!\w)',context)
    lineage=load('historical_pool_lineage_20261022.json')
    for ds,record in lineage.items():
        kept=set(record['retained_qids']);assert len(kept)==record['qids']
        test=record['test_qids_by_fold'];flattened=[q for qs in test.values() for q in qs]
        assert len(flattened)==len(set(flattened))==len(kept) and set(flattened)==kept
        for i,fit in enumerate(record['fit_qids_per_fold']):assert fit==len(kept)-len(test[str(i)])
    f=lineage['FRAMES'];assert f['qids']+f['excluded_count']==f['official_test_rows']==824
    units=load('candidate_row_unit_audit_20261022.json')['datasets']
    for ds,x in units.items():
        assert x['qids']==lineage[ds]['qids'] and x['rows']==lineage[ds]['candidate_rows']
        kinds=x['rows_by_kind'];hist={int(k):v for k,v in x['rows_per_qid_histogram'].items()}
        assert kinds['baseline_text']==kinds['baseline_image']==x['qids']
        assert kinds['swap_text']==kinds['swap_image']
        assert sum(kinds.values())==sum(k*v for k,v in hist.items())==x['rows']
        assert sum(hist.values())==x['qids'] and all(k%2==0 and k<=20 for k in hist)
        close(x['cached_generation_count_per_qid'],x['rows']/x['qids'])
    print('PASS: final write categories, availability, contamination scope, strict EM/F1, component CIs and write admission, gate-span counts, historical pool lineage, and candidate-row units')

if __name__=='__main__':main()
