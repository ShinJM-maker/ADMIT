"""Verify public strict-portability aggregates from compact per-qid metrics."""
from pathlib import Path
import json,math,collections
ROOT=Path(__file__).resolve().parents[1]/'results/strict_portability_20260928'
def main():
 flags={(x['family'],x['setting'],x['qid']):x['refusal'] for x in map(json.loads,(ROOT.parents[1]/'configs/portability_refusal_flags.jsonl').open())}
 for family,nsettings,nrows,key,metrics in [('terminal',8,2400,'pipeline',['raw_em','contamination','no_commit','calls_q']),('dataset',3,900,'dataset',['raw_em','contamination','no_commit','calls_q'])]:
  rows=[json.loads(l) for l in (ROOT/f'{family}_per_qid.jsonl').open()];assert len(rows)==nrows
  summary=json.loads((ROOT/f'{family.upper()}_SUMMARY.json').read_text());groups=collections.defaultdict(list)
  for r in rows:groups[r[key]].append(r)
  assert len(groups)==nsettings
  for k,rs in groups.items():
   assert len(rs)==300 and len({r['qid'] for r in rs})==300
   data=summary['pipelines' if family=='terminal' else 'datasets'][k]
   for arm in ['fail_open','admit','hard_refuse']:
    assert all(not (flags[(family,k,r['qid'])][arm] and r['arms'][arm]['raw_em']) for r in rs), 'portability refusal credited as correct'
    for metric in metrics:
     actual=sum(r['arms'][arm][metric] for r in rs)/300
     assert math.isclose(actual,data['arms'][arm][metric],abs_tol=1e-12)
    if arm!='fail_open':assert all(r['arms'][arm]['contamination']==0 for r in rs)
   if family=='terminal':assert all(r['arms']['admit']['raw_em']==r['arms']['fail_open']['raw_em'] and r['arms']['admit']['calls_q']==r['arms']['fail_open']['calls_q'] for r in rs)
   else:
    metric={'dude':'anls','tatqa':'native_f1','mmlongbench':'generalized_accuracy_short_output'}[k]
    for arm in ['fail_open','admit','hard_refuse']:
     assert math.isclose(sum(r['arms'][arm]['native_metrics'][metric] for r in rs)/300,data['arms'][arm]['task_score'],abs_tol=1e-12)
 print('Strict portability public aggregates: OK (2400 terminal pairs; 900 three-arm dataset qids).')

EXT=ROOT.parent/'strict_portability_ext_20261106'
def validate_extension():
 """Eight additional public upstream artifacts: same 300 qids, contract, reader, and decoding."""
 if not EXT.exists():return
 base={json.loads(l)['qid'] for l in (ROOT/'terminal_per_qid.jsonl').open()}
 rows=[json.loads(l) for l in (EXT/'terminal_per_qid.jsonl').open()];assert len(rows)==2400
 summary=json.loads((EXT/'TERMINAL_SUMMARY.json').read_text());manifest=json.loads((EXT/'MANIFEST.json').read_text())
 groups=collections.defaultdict(list)
 for r in rows:groups[r['pipeline']].append(r)
 assert sorted(groups)==sorted(manifest['pipelines']) and len(groups)==8
 for k,rs in groups.items():
  assert len(rs)==300 and {r['qid'] for r in rs}==base
  data=summary['pipelines'][k]
  for arm in ['fail_open','admit','hard_refuse']:
   assert all(not (r['arms'][arm]['refusal'] and r['arms'][arm]['raw_em']) for r in rs), 'refusal credited as correct'
   for metric in ['raw_em','contamination','no_commit','calls_q']:
    assert math.isclose(sum(r['arms'][arm][metric] for r in rs)/300,data['arms'][arm][metric],abs_tol=1e-12)
   if arm!='fail_open':assert all(r['arms'][arm]['contamination']==0 for r in rs)
  assert all(r['arms']['admit']['raw_em']==r['arms']['fail_open']['raw_em'] and r['arms']['admit']['calls_q']==r['arms']['fail_open']['calls_q'] for r in rs)
  assert all(r['arms']['admit']['no_commit']==r['arms']['fail_open']['contamination'] for r in rs)
 print('Strict portability extension: OK (2400 terminal pairs; 8 additional upstreams on the same 300 qids).')

GRAPH=ROOT.parent/'strict_portability_graph_20261106'
def validate_graph():
 """HopRAG and MoLoRAG local reproductions on the same 300 qids, contract, reader, and decoding."""
 if not GRAPH.exists():return
 base={json.loads(l)['qid'] for l in (ROOT/'terminal_per_qid.jsonl').open()}
 rows=[json.loads(l) for l in (GRAPH/'terminal_per_qid.jsonl').open()];assert len(rows)==600
 summary=json.loads((GRAPH/'TERMINAL_SUMMARY.json').read_text())
 groups=collections.defaultdict(list)
 for r in rows:groups[r['pipeline']].append(r)
 assert sorted(groups)==['hoprag','molorag']
 for k,rs in groups.items():
  assert len(rs)==300 and {r['qid'] for r in rs}==base
  for arm in ['fail_open','admit','hard_refuse']:
   assert all(not (r['arms'][arm]['refusal'] and r['arms'][arm]['raw_em']) for r in rs)
   for metric in ['raw_em','contamination','no_commit','calls_q']:
    assert math.isclose(sum(r['arms'][arm][metric] for r in rs)/300,summary['pipelines'][k]['arms'][arm][metric],abs_tol=1e-12)
   if arm!='fail_open':assert all(r['arms'][arm]['contamination']==0 for r in rs)
  assert all(r['arms']['admit']['raw_em']==r['arms']['fail_open']['raw_em'] and r['arms']['admit']['no_commit']==r['arms']['fail_open']['contamination'] for r in rs)
 print('Strict portability graph upstreams: OK (600 terminal pairs; HopRAG and MoLoRAG on the same 300 qids).')

if __name__=='__main__':
 main();validate_extension();validate_graph()
