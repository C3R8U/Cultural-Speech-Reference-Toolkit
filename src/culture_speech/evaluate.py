import argparse,csv,json
from pathlib import Path
import torch
from .model import load_checkpoint
from .data import SpeechDataset,collate
from .decoding import prefix_beam,rerank
from .metrics import error_counts,generation_metrics

def main():
    p=argparse.ArgumentParser()
    for name in ['checkpoint','manifest','root','out']: p.add_argument('--'+name,required=True)
    p.add_argument('--split',default='test');p.add_argument('--model-name',default='reference_dcsb')
    p.add_argument('--unit',choices=['character','word'],default='character')
    p.add_argument('--alpha',type=float,default=.1,help='Select using validation only; reference default')
    p.add_argument('--no-rerank',action='store_true');p.add_argument('--int8',action='store_true')
    args=p.parse_args(); model,checkpoint=load_checkpoint(args.checkpoint); model.eval()
    if args.int8:
        from .quantize import quantize
        model=quantize(model)
    data=SpeechDataset(args.manifest,args.root,args.split,checkpoint['vocab'])
    rows=[]; references=[];hypotheses=[]
    with torch.inference_mode():
        for item in data:
            batch=collate([item]); output=model(batch['features'],batch['lengths'])
            candidates=prefix_beam(output['ctc_logits'][0].log_softmax(-1).tolist(),10)
            sequence=(candidates[0][0] if args.no_rerank else
                      rerank(model,output['representation'],candidates,args.alpha,5)[0][0])
            transcript=''.join(checkpoint['vocab'][k] for k in sequence)
            errors,total=error_counts(item['record']['transcript'],transcript,args.unit)
            prediction=int(output['intent_logits'].argmax(-1)[0])
            row={'model':args.model_name+('_int8' if args.int8 else ''),
                 'seed':checkpoint['extra'].get('seed','unknown'),'sample_id':item['record']['sample_id'],
                 'speaker_id':item['record']['speaker_id'],'errors':errors,'reference_units':total,
                 'unit':args.unit,'reference':item['record']['transcript'],'hypothesis':transcript,
                 'correct':int(prediction==item['label']) if item['label']>=0 else '',
                 'generated_text':''}
            if item['generation'] is not None:
                length=64 if item['record'].get('generation_subtype')=='古诗续写' else 128
                generated=model.generate(output['representation'],length,5)
                row['generated_text']=''.join(checkpoint['vocab'][k] for k in generated)
                references.append(item['record']['reference_text']);hypotheses.append(row['generated_text'])
            rows.append(row)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'error_unit':args.unit,'error_rate':sum(r['errors'] for r in rows)/sum(r['reference_units'] for r in rows),
             'n':len(rows),'implementation':'new_reference_implementation'}
    if references: summary.update(generation_metrics(references,hypotheses))
    out.with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__':main()
