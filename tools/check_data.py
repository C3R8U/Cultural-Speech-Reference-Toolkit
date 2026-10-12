"""Two separate gates: structural validity and actual research readiness."""
import argparse,csv,json,wave,hashlib
from pathlib import Path
from collections import Counter,defaultdict

INTENTS={'文化咨询','情感表达','创意请求'}

def inspect(root,mode='scaffold'):
    root=Path(root).resolve(); errors=[];missing=Counter(); audio_hashes={}
    rows=[json.loads(s) for s in (root/'samples.jsonl').read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    with (root/'speakers.csv').open(encoding='utf-8-sig',newline='') as f: speakers=list(csv.DictReader(f))
    ids=[r['sample_id'] for r in rows];sids=[s['speaker_id'] for s in speakers]
    if len(set(ids))!=len(ids):errors.append('Duplicate sample IDs')
    if len(set(sids))!=len(sids):errors.append('Duplicate speaker IDs')
    if len(rows)!=manifest['planned_sample_count']:errors.append('Sample count mismatch')
    if len(speakers)!=manifest['planned_speaker_count']:errors.append('Speaker count mismatch')
    splits=defaultdict(set); speaker_map={s['speaker_id']:s for s in speakers}
    actual_audio=actual_transcripts=0
    for r in rows:
        sid=r['speaker_id'];splits[sid].add(r['split'])
        if sid not in speaker_map:errors.append(f'{r["sample_id"]}: unknown speaker')
        elif r['split']!=speaker_map[sid]['split']:errors.append(f'{r["sample_id"]}: speaker split mismatch')
        if r['split'] not in {'train','validation','test'}:errors.append('Unknown split')
        if r.get('task_category') not in INTENTS:errors.append('Unknown planned category')
        if manifest.get('test_only') and r['split']!='test':errors.append('Field data must be test only')
        for key,spec in [('raw_audio_path',(48000,3,1)),('model_audio_path',(16000,2,1))]:
            path=(root/r[key]).resolve()
            if not path.is_relative_to(root):errors.append(f'{r["sample_id"]}: unsafe path');continue
            if not path.is_file():missing[key]+=1;continue
            try:
                with wave.open(str(path),'rb') as w:
                    if (w.getframerate(),w.getsampwidth(),w.getnchannels())!=spec or w.getcomptype()!='NONE':
                        errors.append(f'{r["sample_id"]}: invalid PCM spec {key}')
                    duration=w.getnframes()/w.getframerate()
                    if not 1.5<=duration<=12:errors.append(f'{r["sample_id"]}: duration outside 1.5–12 s')
                    if key=='model_audio_path':
                        actual_audio+=1
                        digest=hashlib.sha256(w.readframes(w.getnframes())).hexdigest()
                        if digest in audio_hashes:errors.append(f'Duplicate waveform: {r["sample_id"]}, {audio_hashes[digest]}')
                        audio_hashes[digest]=r['sample_id']
            except (wave.Error,EOFError):errors.append(f'{r["sample_id"]}: invalid WAV')
        if isinstance(r.get('transcript'),str) and r['transcript'].strip():actual_transcripts+=1
        else:missing['transcript']+=1
        if r.get('cultural_intent_label') not in INTENTS:missing['actual_intent_label']+=1
        if r.get('status')=='planned_not_collected':missing['collected_status']+=1
        if r.get('audio_present') is not True:missing['audio_present_flag']+=1
        if r.get('task_category')=='创意请求':
            if not r.get('reference_text'):missing['verified_generation_reference']+=1
            if r.get('reference_verified') is not True:missing['reference_verification']+=1
    if any(len(v)>1 for v in splits.values()):errors.append('Speaker leakage across splits')
    categories=dict(Counter(r['task_category'] for r in rows))
    if categories!=manifest['category_counts']:errors.append('Planned category counts mismatch')
    if manifest.get('split_speakers'):
        counts=dict(Counter(s['split'] for s in speakers))
        if counts!=manifest['split_speakers']:errors.append('Planned speaker splits mismatch')
    if manifest.get('scene_category_counts'):
        counts={scene:dict(Counter(r['task_category'] for r in rows if r['scene']==scene))
                for scene in manifest['scene_category_counts']}
        if counts!=manifest['scene_category_counts']:errors.append('Scene category counts mismatch')
    for s in speakers:
        if s.get('adult_verified','').lower()!='true':missing['verified_adult_speakers']+=1
        if s.get('consent_status')!='obtained':missing['speaker_consent']+=1
        if s.get('identity_dedup_status')!='verified':missing['identity_verification']+=1
    ready=not errors and not missing
    return {'dataset':manifest['name'],'mode':mode,'structure_passed':not errors,
            'research_ready':ready,'passed':not errors and (mode=='scaffold' or ready),
            'samples':len(rows),'reserved_speakers':len(speakers),'actual_audio':actual_audio,
            'actual_transcripts':actual_transcripts,'split_counts':dict(Counter(r['split'] for r in rows)),
            'planned_categories':categories,'missing':dict(missing),'errors':errors,
            'note':'Structural validity does not certify actual participants or completed experiments'}

def main():
    p=argparse.ArgumentParser();p.add_argument('roots',nargs='+')
    p.add_argument('--mode',choices=['scaffold','research'],default='research');p.add_argument('--out')
    a=p.parse_args(); reports=[inspect(root,a.mode) for root in a.roots]
    speaker_sets=[]
    for root in a.roots:
        with open(Path(root)/'speakers.csv',encoding='utf-8-sig',newline='') as f:
            speaker_sets.append({r['speaker_id'] for r in csv.DictReader(f)})
    overlap=set()
    for i,s in enumerate(speaker_sets):
        for t in speaker_sets[i+1:]:overlap.update(s&t)
    report={'reports':reports,'cross_dataset_id_overlap':sorted(overlap),
            'passed':all(r['passed'] for r in reports) and not overlap,
            'identity_note':'Distinct reserved IDs cannot prove different real people'}
    text=json.dumps(report,ensure_ascii=False,indent=2);print(text)
    if a.out:Path(a.out).write_text(text,encoding='utf-8')
    raise SystemExit(0 if report['passed'] else 2)
if __name__=='__main__':main()
