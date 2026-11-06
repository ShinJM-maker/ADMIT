"""Verify literal prompt strings, hashes, and the generation/admission distinction."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]/'configs'
def digest(data):return hashlib.sha256(data).hexdigest()
def main():
    raw=(ROOT/'frozen_prompt_templates.json').read_bytes()
    data=json.loads(raw);manifest=json.loads((ROOT/'frozen_prompt_manifest.json').read_text())
    assert digest(raw)==manifest['file_sha256']
    for key,expected in manifest['string_sha256'].items():
        if isinstance(expected,dict):
            for name,value in expected.items():assert digest(data[key][name].encode())==value
        else:assert digest(data[key].encode())==expected
    assert data['dynamic_repair']==data['terminal_repair']
    assert 'EXCEPTION' in data['dynamic_repair']
    assert 'best informed guess' in data['static_reread']
    contract=json.loads((ROOT/'execution_contract.json').read_text())
    assert not contract['admission']['non_refusal_bypass_enabled']
    assert 'receive zero' in contract['metrics']['dynamic_raw_em']
    print('Exact frozen prompt templates and strict-gate distinction: OK')
if __name__=='__main__':main()
