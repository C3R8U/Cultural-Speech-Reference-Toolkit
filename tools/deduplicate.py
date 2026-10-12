"""Detect exact normalized generation-reference leakage across data splits."""
import argparse,json,hashlib,unicodedata
from pathlib import Path

def norm(s):
    return ''.join(c for c in unicodedata.normalize('NFKC',s) if not c.isspace() and not unicodedata.category(c).startswith('P'))
def main():
    p=argparse.ArgumentParser();p.add_argument('--training-texts',required=True)
    p.add_argument('--paired-manifest',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    texts={norm(s) for s in Path(a.training_texts).read_text(encoding='utf-8').splitlines() if norm(s)}
    seen={};matches=[]
    for line in Path(a.paired_manifest).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():continue
        r=json.loads(line);text=norm(r.get('reference_text') or '')
        if not text:continue
        if r['split']!='train' and text in texts:matches.append({'sample_id':r['sample_id'],'kind':'pretraining_overlap'})
        if text in seen and seen[text][0]!=r['split']:
            matches.append({'sample_id':r['sample_id'],'other':seen[text][1],'kind':'paired_split_overlap'})
        seen[text]=(r['split'],r['sample_id'])
    Path(a.out).write_text(json.dumps({'passed':not matches,'matches':matches,
        'scope':'Exact normalized references only; source-work and near-duplicate audits still required'},indent=2),encoding='utf-8')
    raise SystemExit(2 if matches else 0)
if __name__=='__main__':main()
