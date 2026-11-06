"""Fresh three-arm PlanRAG using the frozen final harness; batched identical-prefix reuse."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,importlib.util,json,queue,sys,threading,time
import os
H=Path(__file__).resolve().parent;REPO=H.parents[1];sys.path.insert(0,str(H/'snapshots'))
W=Path(os.environ.get('ADMIT_WORK',REPO/'work'))/'portability'  # <dataset>_stage2.jsonl, <dataset>_selected.json
import run_dude_dynamic as base
import contract as c
from run_terminal import em

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class Broker:
 def __init__(self):self.queue=queue.Queue();self.lock=threading.Lock();self.cache={};self.local=threading.local();self.records=[]
 def call(self,prompt,max_tokens):
  key=(self.local.qid,prompt,max_tokens)
  with self.lock:
   if key not in self.cache:
    future=concurrent.futures.Future();self.cache[key]=future;self.queue.put((key,future))
   future=self.cache[key]
  return future.result()
 def generate_pending(self,llm,SamplingParams):
  try:first=self.queue.get(timeout=.05)
  except queue.Empty:return
  batch=[first];time.sleep(.015)
  while len(batch)<16:
   try:batch.append(self.queue.get_nowait())
   except queue.Empty:break
  try:
   outputs=llm.generate([{'prompt':k[1]} for k,f in batch],[SamplingParams(temperature=0,top_p=1,max_tokens=k[2],seed=20260928) for k,f in batch],use_tqdm=False)
   assert len(outputs)==len(batch) and all(x.outputs for x in outputs)
   for (key,future),out in zip(batch,outputs):
    value=out.outputs[0].text.strip();self.records.append({'qid':key[0],'prompt':key[1],'prompt_sha256':hashlib.sha256(key[1].encode()).hexdigest(),'max_tokens':key[2],'output':value});future.set_result(value)
  except BaseException as e:
   for _,future in batch:
    if not future.done():future.set_exception(e)
   raise
class Reader(base.InlineReader):
 def __init__(self,runner,tokenizer,broker):
  self.runner=runner;self.tokenizer=tokenizer;self.broker=broker;self.token_budget=12000;self.max_images_total=0
  self._chunks_idx={};self._cards_idx={};self._docs_idx={};self._parent_to_cards={}
 def generate_batch(self,prompts):return [self.broker.call(p,64) for p in prompts]
 def reasoning_call(self,prompt_text,max_new_tokens=96):
  prompt=c.chat_prompt(self.tokenizer,prompt_text);return self.broker.call(prompt,max_new_tokens)

def annotate(runner,reader,result,card_by_id,arm,question):
 checks=[]
 for r in result['sub_logs']:
  packet=base.pack_from_ids(reader,r['pack_card_ids'],card_by_id)
  assert c.evidence_packet_sha256(packet)==r['evidence_packet_sha256']
  checks.append((r['memory_record'],packet,r['sub_q']))
 checks.append((result['final_memory_record'],result['final_pack_text'],question))
 for record,packet,q in checks:
  expected=c.public_memory_record(record['value'],packet);assert record==expected
  if record['record_type']=='STATUS':assert runner.factual_projection(record) is None
 contaminated=any(c.factual_projection(rec) is not None and not c.support_in_pack(rec['value'],packet) for rec,packet,q in checks)
 if arm in ['admit','hard_refuse']:assert not contaminated
 # Persist all exact delivered packets, including intermediate ones, for independent audit.
 return {'contamination':int(contaminated),'no_commit':int(any(rec.get('status')=='NO_COMMIT' for rec,p,q in checks)),'calls_q':result['n_reader_calls']+result['n_repair_calls'],'planner_calls_q':1,'writes':[{'memory_record':rec,'packet_text':p,'packet_sha256':c.evidence_packet_sha256(p),'question':q,'reuse_eligible_value':c.factual_projection(rec)} for rec,p,q in checks]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--shard-id',type=int,required=True);p.add_argument('--num-shards',type=int,default=2);p.add_argument('--preflight',action='store_true');p.add_argument('--limit',type=int,default=0);args=p.parse_args()
 mf=W/'DATASET_MANIFEST.json'
 manifest=json.loads(mf.read_text()) if mf.exists() else {'datasets':{k:{} for k in ('dude','tatqa','mmlongbench')}}
 for key,m in manifest['datasets'].items():
  if m:
   assert sha(W/f'{key}_stage2.jsonl')==m['stage2_sha256']
   assert sha(W/f'{key}_selected.json')==m['selected_sha256']
 if args.preflight:print('PASS: all three frozen stage2 pools and historical samples');return
 runner=load('strict_dataset_runner',REPO/'harness/agents.py');runner.cavec_admit=c.strict_cavec_admit;runner.build_evidence_pack=base.inline_pack
 from vllm import LLM,SamplingParams
 llm=LLM(model=c.MODEL,trust_remote_code=True,dtype='bfloat16',gpu_memory_utilization=.82,max_model_len=16384,limit_mm_per_prompt={'image':1},enforce_eager=True,tensor_parallel_size=1,max_num_batched_tokens=16384,seed=20260928)
 tokenizer=llm.get_tokenizer();out=W/('dataset_smoke' if args.limit else 'dataset_results');out.mkdir(parents=True,exist_ok=True)
 for dataset,m in manifest['datasets'].items():
  chosen=json.loads((W/f'{dataset}_selected.json').read_text());chosen=[r for i,r in enumerate(chosen) if i%args.num_shards==args.shard_id]
  if args.limit:chosen=chosen[:args.limit]
  dest=out/f'{dataset}.shard{args.shard_id}.jsonl';audit=out/f'{dataset}.shard{args.shard_id}.generations.jsonl'
  if dest.exists() or audit.exists():raise FileExistsError(dest)
  source=base.read_jsonl(W/f'{dataset}_stage2.jsonl');cards=base.cards_from_rows(source);card_by_id={x['card_id']:x for x in cards};bm25=base.SimpleBM25([x['bm25_text'] for x in cards],runner.tok)
  for offset in range(0,len(chosen),8):
   batch=chosen[offset:offset+8];broker=Broker();reader=Reader(runner,tokenizer,broker)
   def job(row,arm):
    broker.local.qid=row['qid'];config={'with_cavec':arm!='admit','cavec_fail_closed':arm=='hard_refuse','with_cavec_mfc':arm=='admit'}
    result=runner.run_planrag(reader,row['question'],bm25,cards,{},top_k=10,**config)
    metrics=annotate(runner,reader,result,card_by_id,arm,row['question']);metrics['raw_em']=em(result['final_committed_answer'],row['gold'])
    return row['qid'],arm,{'result':result,'metrics':metrics}
   executor=concurrent.futures.ThreadPoolExecutor(max_workers=24)
   futures=[executor.submit(job,row,arm) for row in batch for arm in ['fail_open','admit','hard_refuse']]
   try:
    while not all(f.done() for f in futures):
     for f in futures:
      if f.done() and f.exception():raise f.exception()
     broker.generate_pending(llm,SamplingParams)
    results={r['qid']:{**r,'dataset':dataset,'fresh_inference':True,'arms':{}} for r in batch}
    for f in futures:
     qid,arm,value=f.result();results[qid]['arms'][arm]=value
    for qid,row in results.items():
     generated=[g for g in broker.records if g['qid']==qid]
     row['physical_model_generations']=len(generated)
     for arm,payload in row['arms'].items():
      for write in payload['metrics']['writes']:
       assert any(write['packet_text'] in g['prompt'] for g in generated)
     with dest.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    with audit.open('a') as f:
     for g in broker.records:f.write(json.dumps(g,ensure_ascii=False)+'\n')
   finally:
    for future in broker.cache.values():
     if not future.done():future.set_exception(RuntimeError('batch aborted'))
    executor.shutdown(wait=True,cancel_futures=True)
   print(f'{dataset} shard {args.shard_id}: {min(offset+8,len(chosen))}/{len(chosen)}',flush=True)
  dest.with_suffix('.complete.json').write_text(json.dumps({'n':len(chosen),'result_sha256':sha(dest),'generation_sha256':sha(audit),'dataset_manifest_sha256':sha(H/'DATASET_MANIFEST.json')})+'\n')
if __name__=='__main__':main()
