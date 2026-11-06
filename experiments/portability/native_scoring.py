"""Pinned official scorers plus explicit, prediction-only output adapters."""
from pathlib import Path
import ast,importlib.util,json,logging,re,sys
H=Path(__file__).resolve().parent;N=H/'native_metrics'
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
sys.path.insert(0,str(N/'tatqa'))
tatqa=load('native_tatqa',N/'tatqa/tatqa_metric.py')
mm=load('native_mmlong',N/'mmlongbench/eval_score.py')
# DUDE's selected sample is single-answer. Load its official NLS functions,
# excluding unrelated submission/remote-metric imports.
source=(N/'dude/evaluate_submission.py').read_text();tree=ast.parse(source)
ns={'logging':logging};nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in {'levenshtein_distance','parse_answers','get_NLS'}]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<official DUDE scalar NLS>','exec'),ns)
def is_refusal(s):return not str(s).strip() or 'do not have enough information' in str(s).lower()
def tatqa_prediction(s):
 s=s.strip();scale=''
 # Never use gold scale or answer in parsing the prediction.
 m=re.fullmatch(r'\s*([$€£]?\s*[-+]?\(?[\d,.]+\)?)\s*(%|percent|thousand|million|billion)\s*\.?\s*',s,re.I)
 if m:s=m.group(1).strip();scale='percent' if m.group(2) in ['%','percent'] else m.group(2).lower()
 return s,scale

def score(dataset,row,output):
 if dataset=='dude':
  prediction='' if is_refusal(output) else output
  return {'anls':float(ns['get_NLS'](row['gold'],prediction,.5))}
 if dataset=='tatqa':
  pred,scale=tatqa_prediction(output);metric=tatqa.TaTQAEmAndF1();metric(row['native_annotation'],pred,pred_scale=scale)
  detail=metric._details[-1];return {'native_em':float(detail['em']),'native_f1':float(detail['f1']),'predicted_scale':scale}
 if dataset=='mmlongbench':
  ann=row['native_annotation'];prediction='Not answerable' if is_refusal(output) else output
  assert ann['answer_format'] in ['Int','Float','Str','None']
  return {'generalized_accuracy_short_output':mm.eval_score(ann['answer'],prediction,ann['answer_format']),'scored_prediction':prediction}
 raise ValueError(dataset)
if __name__=='__main__':
 assert score('dude',{'gold':['Pump A']},'Pump A')['anls']==1
 assert score('dude',{'gold':['Pump A']},'I do not have enough information to answer.')['anls']==0
 assert tatqa_prediction('25%')==('25','percent')
 assert tatqa_prediction('25')==('25','')
 for key in ['dude','tatqa','mmlongbench']:
  rows=json.loads((H/'snapshots'/f'{key}_selected.json').read_text())
  for row in rows:
   for out in [row['gold'][0],'I do not have enough information to answer.']:score(key,row,out)
 print('PASS: 1800 annotation/scorer checks plus prediction-only scale checks')
