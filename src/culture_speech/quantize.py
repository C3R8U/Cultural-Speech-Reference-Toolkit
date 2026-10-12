"""CPU dynamic post-training INT8; no fabricated trained checkpoints."""
import argparse,json,time,io
from pathlib import Path
import torch
from torch import nn
from .model import load_checkpoint

def quantize(model,engine=None):
    supported=torch.backends.quantized.supported_engines
    engine=engine or ('qnnpack' if 'qnnpack' in supported and 'x86' not in supported else
                      'x86' if 'x86' in supported else 'fbgemm')
    if engine not in supported: raise ValueError(f'Unsupported INT8 engine {engine}; available {supported}')
    torch.backends.quantized.engine=engine
    return torch.ao.quantization.quantize_dynamic(model.cpu().eval(),{nn.LSTM,nn.Linear},dtype=torch.qint8)

def state_bytes(model):
    stream=io.BytesIO();torch.save(model.state_dict(),stream);return len(stream.getvalue())

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--out',required=True)
    p.add_argument('--engine');p.add_argument('--threads',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(args.threads)
    model,checkpoint=load_checkpoint(args.checkpoint);model.eval(); quantized=quantize(model,args.engine)
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    # Version-specific state; reconstruct topology and quantize before loading this state.
    torch.save({'config':checkpoint['config'],'vocab':checkpoint['vocab'],'extra':checkpoint['extra'],
                'state_dict':quantized.state_dict(),'torch_version':str(torch.__version__),
                'engine':torch.backends.quantized.engine,'format':'dynamic_int8_state'},out)
    report={'fp32_state_bytes':state_bytes(model),'int8_state_bytes':state_bytes(quantized),
            'engine':torch.backends.quantized.engine,'torch_version':str(torch.__version__),
            'note':'No accuracy or real-device performance claims; run paired evaluation on real data'}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

def load_int8(path):
    from .model import Config,SpeechSemanticModel
    # Packed dynamic LSTM parameters serialize as a PyTorch ScriptObject.
    # Allow only that native type, retaining weights-only loading.
    with torch.serialization.safe_globals([torch.ScriptObject]):
        c=torch.load(path,map_location='cpu',weights_only=True)
    if c['torch_version']!=str(torch.__version__): raise ValueError('INT8 state requires matching PyTorch version')
    model=quantize(SpeechSemanticModel(Config(**c['config'])),c['engine'])
    model.load_state_dict(c['state_dict']);return model,c
if __name__=='__main__':main()
