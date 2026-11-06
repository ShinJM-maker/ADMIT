"""Recompute corrected metrics from the archived compact rows (NumPy required).

Run with --write to regenerate, or without arguments to verify current files.
Model outputs and decisions are frozen. The original stage changes only
refusal-aware substring EM; the released write-scope audit is then applied.
"""
from pathlib import Path
from collections import defaultdict
import argparse,copy,hashlib,json
import numpy as np
from answer_metrics import is_refusal
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'historical/pre_refusal_score_correction_20261020/results'
DEST=ROOT/'results'
DATE='2026-10-20'
def read(name):return json.loads((OLD/name).read_text())
def rows(name):return [json.loads(x) for x in (OLD/name).open() if x.strip()]
def dump(value):return json.dumps(value,indent=2)+'\n'
def jsonl(values):return ''.join(json.dumps(x)+'\n' for x in values)
def sha(data):return hashlib.sha256(data.encode()).hexdigest()
def boot(delta,seed,n=10000,pp=False,kind='dynamic'):
    delta=np.asarray(delta,dtype=float);rng=np.random.default_rng(seed)
    sampled=delta[rng.integers(0,len(delta),(n,len(delta)))].mean(axis=1)
    mean=float(delta.mean());lo,hi=map(float,np.quantile(sampled,[.025,.975]))
    if kind=='static':return {'n':len(delta),'mean':mean,'ci95':[lo,hi],'draws':n}
    if kind=='portability':return {'delta':100*mean if pp else mean,'lo95':100*lo if pp else lo,'hi95':100*hi if pp else hi}
    return {'point':mean,'ci95':[lo,hi],'unit':'qid paired bootstrap','draws':n}

def generate():
    files={};ledger={'date':DATE,'rule':'Fixed refusals and empty outputs receive zero substring EM; all other matching is unchanged. Dataset-native task scores are unchanged.','model_calls':0,'changed_rows':{},'source_scope':'Archived pre-correction compact metrics; immutable original outputs were separately checked locally.','bootstrap_seeds':{'static':42,'dynamic':20260807,'portability':20260928}}
    static=rows('static_deployable_router_per_qid.jsonl');aggregate=read('static_deployable_router_results.json')
    for policy in static[0]['policies']:
        changed=0
        for row in static:
            m=row['policies'][policy]
            if is_refusal(m['answer']):changed+=int(bool(m['EM']));m['EM']=0
            m['value_supported_EM']=m['EM']*m['support']
        ledger['changed_rows']['static/'+policy]=changed
        for metric in ('EM','F1','ANLS','unsupNR_strict','value_supported_EM','calls','no_commit'):
            aggregate['metrics'][policy][metric]=float(np.mean([x['policies'][policy][metric] for x in static]))
    pairs={'ADMIT_minus_Vanilla single-pass':('ADMIT','Vanilla single-pass'),'ADMIT_minus_Final-support + retry':('ADMIT','Final-support + retry'),'ADMIT_minus_Every-q repair':('ADMIT','Every-q repair'),'Learned-router_diagnostic_minus_ADMIT':('Learned-router diagnostic','ADMIT')}
    ordered=sorted(static,key=lambda x:x['qid'])
    for key,(a,b) in pairs.items():
        # Other metrics did not change; retain their original exact intervals.
        for metric in ['EM','value_supported_EM']:
            delta=[x['policies'][a][metric]-x['policies'][b][metric] for x in ordered]
            aggregate['paired_deltas'][key][metric]=boot(delta,42,kind='static')
    aggregate['contract']['metric']+='; scoring revision '+DATE+': refusals receive zero substring EM'
    aggregate['contract']['frozen_router_training']='Original fitted router and routes retained; only reporting scores are recomputed.'
    aggregate['scoring_revision']={'date':DATE,'model_calls':0,'archived_results':'historical/pre_refusal_score_correction_20261020/results'}
    files['static_deployable_router_per_qid.jsonl']=jsonl(static)
    aggregate['artifact_sha256']['per_qid']=sha(files['static_deployable_router_per_qid.jsonl'])
    files['static_deployable_router_results.json']=dump(aggregate)
    summary=read('strict_dynamic_summary_public.json')
    for family in ['comparator','component']:
        name=f'strict_{family}_per_qid.jsonl';values=rows(name);by=defaultdict(dict)
        for x in values:
            key=family+'/'+x['policy'];ledger['changed_rows'].setdefault(key,0)
            if x['final_refusal']:
                ledger['changed_rows'][key]+=int(bool(x['raw_em']));x['raw_em']=0.0
            by[x['policy']][(x['agent'],x['qid'])]=x
        for arm,obj in summary[family]['arms'].items():
            for agent in ['react','planrag','autogen','pooled']:
                subset=[x for (a,q),x in by[arm].items() if agent=='pooled' or a==agent]
                for metric in ['raw_em','contamination','no_commit','calls_q','final_refusal']:
                    obj['metrics'][agent][metric]=float(np.mean([x[metric] for x in subset]))
        a='cavec_mfc' if family=='comparator' else 'full_admit'
        pg='admit_minus_control' if family=='comparator' else 'full_minus_other'
        for b,scopes in summary['paired'][family][pg].items():
            qids=sorted({q for ag,q in by[a]})
            for agent in ['react','planrag','autogen','pooled']:
                agents=['react','planrag','autogen'] if agent=='pooled' else [agent]
                delta=[np.mean([by[a][(ag,q)]['raw_em']-by[b][(ag,q)]['raw_em'] for ag in agents]) for q in qids]
                scopes[agent]['raw_em']=boot(delta,20260807)
        files[name]=jsonl(values)
        summary['public_verification'][family]['sha256']=sha(files[name])
    summary['scoring_revision']={'date':DATE,'model_calls':0,'definition':'refusal-aware bidirectional substring EM','runtime_freeze_date':summary['freeze_date']}
    files['strict_dynamic_summary_public.json']=dump(summary)
    registry=read('results_registry_public.json')
    definitions=registry['metric_definitions']
    for key in ['raw_em','static_em']:definitions[key]+='; fixed refusals and empty outputs score zero'
    for key in ['EM']:
        registry['contracts']['static'][key]+='; fixed refusals receive zero'
    registry['contracts']['dynamic']['answer_metric']='refusal-aware normalized bidirectional substring EM'
    registry['static_paired_deltas']=aggregate['paired_deltas']
    sources=registry['provenance']['public_verification_sources']
    sources['static']['sha256']=sha(files['static_deployable_router_per_qid.jsonl'])
    for f in ['comparator','component']:sources['dynamic_'+f]=summary['public_verification'][f]
    mapping={'Gate-only':('component','gate_only'),'Gate + RepairRead':('component','gate_repair'),'Full ADMIT':('component','full_admit'),'Final-support + retry':('comparator','final_support_retry'),'ADMIT fail-open':('comparator','cavec_paper'),'Hard-refuse':('comparator','cavec_h1_hard_refuse'),'ADMIT MFC/OFO':('comparator','cavec_mfc')}
    for entry in registry['entries']:
        if entry['agent']=='static':
            entry['source']['sha256']=sha(files['static_deployable_router_results.json'])
            obj=aggregate['metrics'][entry['policy']];target=entry['metrics']
            for key in ['EM','F1','ANLS','unsupNR_strict','value_supported_EM']:target[key]=obj[key]
        else:
            family,arm=mapping[entry['policy']];entry['metrics']=summary[family]['arms'][arm]['metrics'][entry['agent']]
        entry['scoring_revision']=DATE
    registry['scoring_revision']={'date':DATE,'output_generation_unchanged':True,'original_full_trajectory_score_fields':'Historical; corrected metrics recomputed from saved output/refusal indicators.','ledger':'scoring_correction_20261020.json'}
    flags={(x['family'],x['setting'],x['qid']):x['refusal'] for x in [json.loads(s) for s in (ROOT/'configs/portability_refusal_flags.jsonl').open()]}
    for family in ['terminal','dataset']:
        folder='strict_portability_20260928/';name=folder+family+'_per_qid.jsonl';values=rows(name);s=read(folder+family.upper()+'_SUMMARY.json');groups=defaultdict(list);setting_key='pipeline' if family=='terminal' else 'dataset'
        for x in values:
            setting=x[setting_key];fr=flags[(family,setting,x['qid'])]
            for arm,m in x['arms'].items():
                key=family+'/'+setting+'/'+arm;ledger['changed_rows'].setdefault(key,0)
                if fr[arm]:ledger['changed_rows'][key]+=int(bool(m['raw_em']));m['raw_em']=0
            groups[setting].append(x)
        for key,rs in groups.items():
            group=s['pipelines' if family=='terminal' else 'datasets'][key]
            for arm in ['fail_open','admit','hard_refuse']:group['arms'][arm]['raw_em']=float(np.mean([x['arms'][arm]['raw_em'] for x in rs]))
            # Terminal paired summaries include raw EM. Dataset paired summaries
            # report native task score, which is intentionally unchanged.
            if family=='terminal':
                for arm in ['fail_open','hard_refuse']:
                    delta=[x['arms']['admit']['raw_em']-x['arms'][arm]['raw_em'] for x in rs]
                    group['paired'][arm]['raw_em']=boot(delta,20260928,pp=True,kind='portability')
        s['scoring_revision']={'date':DATE,'native_scores_unchanged':True,'model_calls':0}
        files[name]=jsonl(values);files[folder+family.upper()+'_SUMMARY.json']=dump(s)
    # The extension hashes identify the current compact verification files.
    extension=registry.get('strict_portability_extension',{})
    def replace_hashes(obj):
        if isinstance(obj,dict):
            for key,value in obj.items():
                if isinstance(value,str) and key.endswith('sha256'):
                    for fname in ['terminal_per_qid.jsonl','dataset_per_qid.jsonl']:
                        oldhash=hashlib.sha256((OLD/'strict_portability_20260928'/fname).read_bytes()).hexdigest()
                        if value==oldhash:obj[key]=sha(files['strict_portability_20260928/'+fname])
                elif isinstance(value,(dict,list)):replace_hashes(value)
        elif isinstance(obj,list):
            for value in obj:replace_hashes(value)
    replace_hashes(extension)
    files['results_registry_public.json']=dump(registry)
    ledger['input_sha256']={str(p.relative_to(OLD)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OLD.rglob('*')) if p.is_file()}
    files['scoring_correction_20261020.json']=dump(ledger)
    return files

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--write',action='store_true');args=parser.parse_args()
    files=generate()
    if (ROOT/'results/write_scope_per_qid.jsonl').exists():
        from recompute_write_scope import apply_write_scope
        files=apply_write_scope(files)
    for name,content in files.items():
        p=DEST/name
        if args.write:p.write_text(content)
        else:assert p.read_text()==content,'rescore mismatch: '+name
    print(('WROTE' if args.write else 'PASS: reproduced')+' refusal-aware aggregates and paired intervals from archived rows')
if __name__=='__main__':main()
