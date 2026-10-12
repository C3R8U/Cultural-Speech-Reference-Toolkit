import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import read_records, build_vocab, SpeechDataset, collate
from .model import Config, SpeechSemanticModel, joint_loss, save_checkpoint

def main():
    p=argparse.ArgumentParser(description='New reference training; not original experiment reproduction')
    p.add_argument('--manifest',required=True); p.add_argument('--root',required=True)
    p.add_argument('--out',required=True); p.add_argument('--seed',type=int,default=17)
    p.add_argument('--epochs',type=int,default=50); p.add_argument('--patience',type=int,default=10)
    p.add_argument('--batch-size',type=int,default=32); p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--bridge',choices=['dcsb','soft','hard'],default='dcsb')
    p.add_argument('--device',default='cpu');p.add_argument('--checkpoint');p.add_argument('--vocab'); args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False
    records=read_records(args.manifest)
    ids=[r['sample_id'] for r in records]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate sample IDs')
    speaker_splits={}
    for r in records:
        speaker_splits.setdefault(r['speaker_id'],set()).add(r['split'])
    if any(len(s)>1 for s in speaker_splits.values()): raise ValueError('Speaker split leakage')
    from .model import load_checkpoint
    if args.checkpoint:
        model,checkpoint=load_checkpoint(args.checkpoint);vocab=checkpoint['vocab']
        model.config.bridge=args.bridge
    else:
        vocab=json.loads(Path(args.vocab).read_text(encoding='utf-8')) if args.vocab else build_vocab(records)
        model=SpeechSemanticModel(Config(len(vocab),bridge=args.bridge))
    loaders={s:DataLoader(SpeechDataset(args.manifest,args.root,s,vocab),batch_size=args.batch_size,
                          shuffle=s=='train',collate_fn=collate) for s in ['train','validation']}
    model=model.to(args.device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    if (out/'history.jsonl').exists() or (out/'best.pt').exists():raise ValueError('Output run already exists; choose a new directory')
    best, stale=float('inf'),0
    for epoch in range(args.epochs):
        metrics={}
        for split,loader in loaders.items():
            model.train(split=='train'); loss_sum=0.; count=0
            with torch.set_grad_enabled(split=='train'):
                for batch in loader:
                    batch={k:v.to(args.device) if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
                    output=model(batch['features'],batch['lengths'],batch['decoder_input'])
                    loss,_=joint_loss(output,batch)
                    if split=='train':
                        optimizer.zero_grad(); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step()
                    loss_sum+=float(loss.detach())*len(batch['records']); count+=len(batch['records'])
            metrics[split]=loss_sum/count
        with open(out/'history.jsonl','a',encoding='utf-8') as f:
            f.write(json.dumps({'epoch':epoch+1,'seed':args.seed,**metrics})+'\n')
        if metrics['validation']<best:
            best=metrics['validation']; stale=0
            save_checkpoint(out/'best.pt',model,vocab,{'seed':args.seed,'epoch':epoch+1,'validation_loss':best})
        else: stale+=1
        print(epoch+1,metrics)
        if stale>=args.patience: break
if __name__=='__main__': main()
