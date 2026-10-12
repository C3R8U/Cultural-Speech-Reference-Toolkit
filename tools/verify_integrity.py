"""Verify every distributed file against the package's SHA256SUMS.json."""
import hashlib,json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/'SHA256SUMS.json').read_text(encoding='utf-8'))
    failures=[]
    for relative,digest in manifest.items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():failures.append(relative);continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:failures.append(relative)
    if failures:raise SystemExit('Integrity failure: '+', '.join(failures))
    print(f'PASS: {len(manifest)} distributed files match SHA-256 manifest')
if __name__=='__main__':main()
