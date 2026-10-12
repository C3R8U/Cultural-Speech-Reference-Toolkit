import argparse,json,unicodedata
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('manifests',nargs='+');p.add_argument('--out',required=True);a=p.parse_args()
    chars=set()
    for path in a.manifests:
        for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():continue
            r=json.loads(line)
            if r['split']!='train':continue
            for key in ['transcript','reference_text','text','prompt','target']:
                if isinstance(r.get(key),str):
                    chars.update(c for c in unicodedata.normalize('NFKC',r[key]) if not c.isspace() and not unicodedata.category(c).startswith('P'))
    if not chars:raise ValueError('No actual training texts')
    Path(a.out).write_text(json.dumps(['<blank>','<bos>','<eos>','<unk>']+sorted(chars),ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
