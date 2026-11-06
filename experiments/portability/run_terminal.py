"""Fresh matched terminal inference; no historical answers are reused."""
from pathlib import Path
import argparse,hashlib,json
import contract as c
HERE=Path(__file__).resolve().parent

def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def em(a,golds):
 if c.is_refusal(a):return 0
 a=c.normalize(a)
 return int(any(a and (g:=c.normalize(gold)) and (a in g or g in a) for gold in golds))
def policies(candidate,repair,packet,question):
 triggered=not c.cavec_admit(candidate,packet,question)[0]
 assert (repair is not None)==triggered
 kw=dict(reader=None,candidate_value=candidate,question=question,sec_cands=[],pack_text=packet,baseline_value=candidate,cached_repair=(repair,int(triggered)))
 arms={}
 for name in ['fail_open','admit','hard_refuse']:
  if name=='admit':memory,output,log=c.cavec_mfc_hook(**kw)
  else:
   output,log=c.cavec_hook(**kw,fail_closed=name=='hard_refuse');memory=output
  record=c.public_memory_record(memory,packet);reuse=c.factual_projection(record)
  contam=bool(reuse is not None and not c.support_in_pack(reuse,packet))
  if name!='fail_open':assert not contam
  if record['record_type']=='STATUS':assert reuse is None
  arms[name]={'output':output,'memory_record':record,'reuse_eligible_value':reuse,'hook':log,'contamination':int(contam),'no_commit':int(record.get('status')=='NO_COMMIT'),'calls_q':1+log['n_repair_calls']}
 assert arms['admit']['output']==arms['fail_open']['output']
 return arms

def main():
 p=argparse.ArgumentParser();p.add_argument('--shard-id',type=int,required=True);p.add_argument('--num-shards',type=int,default=2);p.add_argument('--smoke',action='store_true');p.add_argument('--preflight',action='store_true');args=p.parse_args()
 import os
 work=Path(os.environ.get('ADMIT_WORK',HERE.parents[1]/'work'))/'portability'
 source=work/'frozen_packets.jsonl'  # written by prepare_packets.py
 m={'contract_sha256':sha((HERE/'contract.py').read_text())}
 if (work/'MANIFEST.json').exists():  # optional: pin the frozen packets and contract
  m=json.loads((work/'MANIFEST.json').read_text())
  assert sha((HERE/'contract.py').read_text())==m['contract_sha256']
  assert hashlib.sha256(source.read_bytes()).hexdigest()==m['sources']['frozen_packets.jsonl']['sha256']
 rows=[json.loads(l) for l in source.open() if l.strip()];rows=[r for r in rows if r['qid_index']%args.num_shards==args.shard_id]
 if args.smoke:rows=rows[:8]
 if args.preflight:print('PASS',len(rows),'rows; no inference');return
 out=work/('smoke' if args.smoke else 'results');out.mkdir(parents=True,exist_ok=True)
 dest=out/f'terminal.shard{args.shard_id}.jsonl'
 if dest.exists():raise FileExistsError(dest)
 from vllm import LLM,SamplingParams
 llm=LLM(model=c.MODEL,trust_remote_code=True,dtype='bfloat16',gpu_memory_utilization=.82,max_model_len=8192,enforce_eager=True,tensor_parallel_size=1,max_num_batched_tokens=16384,seed=20260928)
 tokenizer=llm.get_tokenizer();sampling=SamplingParams(temperature=0.,top_p=1.,max_tokens=64,seed=20260928)
 def generate(prompts):
  if not prompts:return []
  result=llm.generate([{'prompt':x} for x in prompts],sampling,use_tqdm=False)
  assert len(result)==len(prompts) and all(x.outputs for x in result),'generation failure'
  return [x.outputs[0].text.strip() for x in result]
 for offset in range(0,len(rows),32):
  batch=rows[offset:offset+32]
  for r in batch:
   raw=r.pop('raw_packet_text');ids=tokenizer.encode(raw,add_special_tokens=False);packet=tokenizer.decode(ids[:4500],skip_special_tokens=True)
   r.update(packet_text=packet,packet_sha256=sha(packet),packet_tokens=min(len(ids),4500),packet_truncated=len(ids)>4500)
   prompt=c.chat_prompt(tokenizer,c.BASE_PROMPT.format(question=r['question'],context=packet));r['initial_prompt']=prompt
   assert len(tokenizer.encode(prompt,add_special_tokens=False))+64<=8192
  raw_outputs=generate([r['initial_prompt'] for r in batch])
  repairs=[]
  for r,raw in zip(batch,raw_outputs):
   r['candidate_raw']=raw;r['candidate']=c.clean_answer(raw);r['initial_prompt_sha256']=sha(r['initial_prompt'])
   if not c.cavec_admit(r['candidate'],r['packet_text'],r['question'])[0]:
    r['repair_prompt']=c.chat_prompt(tokenizer,c.REPAIR_PROMPT_HEAD.format(question=r['question'])+r['packet_text']+c.PROMPT_TAIL)
    assert len(tokenizer.encode(r['repair_prompt'],add_special_tokens=False))+64<=8192
    repairs.append(r)
  for r,raw in zip(repairs,generate([r['repair_prompt'] for r in repairs])):
   r['repair_raw']=raw;r['repair']=c.clean_answer(raw);r['repair_prompt_sha256']=sha(r['repair_prompt'])
  with dest.open('a') as f:
   for r in batch:
    r['arms']=policies(r['candidate'],r.get('repair'),r['packet_text'],r['question'])
    for a in r['arms'].values():a['raw_em']=em(a['output'],r['gold'])
    r['contract_sha256']=m['contract_sha256'];r['fresh_inference']=True
    f.write(json.dumps(r,ensure_ascii=False)+'\n')
  print(f'shard {args.shard_id}: {min(offset+32,len(rows))}/{len(rows)}',flush=True)
 (out/f'terminal.shard{args.shard_id}.complete.json').write_text(json.dumps({'rows':len(rows),'result_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'manifest_sha256':hashlib.sha256((HERE/'MANIFEST.json').read_bytes()).hexdigest()})+'\n')
if __name__=='__main__':main()
