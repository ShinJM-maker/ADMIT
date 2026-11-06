from pathlib import Path
import urllib.request,json,hashlib
H=Path(__file__).resolve().parent/'native_metrics';H.mkdir(exist_ok=True)
configs={'tatqa':('NExTplusplus/TAT-QA','master',['tatqa_metric.py','tatqa_utils.py','LICENSE']), 'dude':('Jordy-VL/DUDEeval','main',['evaluate_submission.py','LICENSE']), 'mmlongbench':('mayubo2333/MMLongBench-Doc','main',['eval/eval_score.py','eval/extract_answer.py','LICENSE.md'])}
manifest={}
for key,(repo,branch,files) in configs.items():
 commit=json.load(urllib.request.urlopen(f'https://api.github.com/repos/{repo}/commits/{branch}',timeout=30))['sha']
 folder=H/key;folder.mkdir(exist_ok=True);manifest[key]={'repo':repo,'commit':commit,'files':{}}
 for name in files:
  url=f'https://raw.githubusercontent.com/{repo}/{commit}/{name}'
  try:data=urllib.request.urlopen(url,timeout=30).read()
  except urllib.error.HTTPError as e:
   if e.code==404 and 'LICENSE' in name:continue
   raise
  dst=folder/Path(name).name;dst.write_bytes(data);manifest[key]['files'][dst.name]={'url':url,'sha256':hashlib.sha256(data).hexdigest()}
(H/'SOURCES.json').write_text(json.dumps(manifest,indent=2)+'\n');print('Downloaded pinned official metric sources.')
