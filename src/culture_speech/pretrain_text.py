"""Auxiliary semantic classification or generator pretraining on actual text JSONL."""
import argparse,json,random
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from .data import normalize
from .model import Config,SpeechSemanticModel,save_checkpoint,load_checkpoint

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--task',choices=['semantic','generation'],required=True);p.add_argument('--checkpoint');p.add_argument('--vocab')
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--batch-size',type=int,default=32);a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    rows=[json.loads(s) for s in Path(a.input).read_text(encoding='utf-8').splitlines() if s.strip()]
    train=[r for r in rows if r['split']=='train'];dev=[r for r in rows if r['split']=='validation']
    if not train or not dev:raise ValueError('Need explicit text train and validation splits')
    keys=['text'] if a.task=='semantic' else ['prompt','target']
    train_texts={normalize(r[k]) for r in train for k in keys}
    if any(normalize(r[k]) in train_texts for r in dev for k in keys):raise ValueError('Exact text split leakage')
    if a.checkpoint:model,c=load_checkpoint(a.checkpoint);vocab=c['vocab']
    else:
        vocab=json.loads(Path(a.vocab).read_text(encoding='utf-8')) if a.vocab else ['<blank>','<bos>','<eos>','<unk>']+sorted({ch for text in train_texts for ch in text})
        model=SpeechSemanticModel(Config(len(vocab)))
    mapping={ch:i for i,ch in enumerate(vocab)}
    def encode(s):
        ids=[mapping.get(ch,3) for ch in normalize(s)]
        if not ids:raise ValueError('Empty normalized text')
        return torch.tensor(ids,dtype=torch.long)
    classes=sorted({str(r['label']) for r in train}) if a.task=='semantic' else []
    auxiliary=torch.nn.Linear(model.config.hidden,len(classes)) if classes else None
    modules=[model.characters,model.semantic, auxiliary] if auxiliary is not None else [model.characters,model.semantic,model.decoder,model.generation_projection]
    optimizer=torch.optim.Adam([p for m in modules for p in m.parameters()],lr=.001)
    best=float('inf');out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(a.epochs):
        values={}
        for name,data in [('train',train),('validation',dev)]:
            model.train(name=='train')
            if auxiliary is not None:auxiliary.train(name=='train')
            data=list(data)
            if name=='train':random.shuffle(data)
            total=0.;count=0
            with torch.set_grad_enabled(name=='train'):
                for start in range(0,len(data),a.batch_size):
                    batch=data[start:start+a.batch_size]
                    sequences=[encode(r['text'] if a.task=='semantic' else r['prompt']) for r in batch]
                    lengths=torch.tensor([len(s) for s in sequences]);ids=pad_sequence(sequences,batch_first=True)
                    rep=model.semantic_representation(model.characters(ids),lengths)
                    if auxiliary is not None:
                        labels=torch.tensor([classes.index(str(r['label'])) for r in batch])
                        loss=F.cross_entropy(auxiliary(rep),labels)
                    else:
                        targets=[encode(r['target']) for r in batch]
                        di=pad_sequence([torch.cat([torch.tensor([1]),t]) for t in targets],batch_first=True)
                        dt=pad_sequence([torch.cat([t,torch.tensor([2])]) for t in targets],batch_first=True)
                        h=rep.unsqueeze(0).repeat(2,1,1).contiguous()
                        decoded,_=model.decoder(model.characters(di),(h,torch.zeros_like(h)))
                        logits=model.generation_projection(decoded)
                        loss=F.cross_entropy(logits.reshape(-1,len(vocab)),dt.reshape(-1),ignore_index=0)
                    if name=='train':
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([p for m in modules for p in m.parameters()],5);optimizer.step()
                    total+=float(loss.detach())*len(batch);count+=len(batch)
            values[name]=total/count
        if values['validation']<best:
            best=values['validation'];save_checkpoint(out,model,vocab,{'seed':a.seed,'text_task':a.task,'auxiliary_classes':classes})
            if auxiliary is not None:torch.save({'classes':classes,'state_dict':auxiliary.state_dict()},out.with_suffix('.auxiliary.pt'))
        print(epoch+1,values)
if __name__=='__main__':main()
