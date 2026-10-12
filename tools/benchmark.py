"""Measure local CPU model path; not a Raspberry Pi reproduction claim."""
import argparse,json,time,platform,threading,os
from pathlib import Path
import torch
import psutil
from culture_speech.model import load_checkpoint
from culture_speech.quantize import quantize
from culture_speech.data import mfcc
from culture_speech.decoding import prefix_beam,rerank

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--wav-list',required=True)
    p.add_argument('--out',required=True);p.add_argument('--int8',action='store_true')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--engine');p.add_argument('--alpha',type=float,default=.1)
    a=p.parse_args();torch.set_num_threads(a.threads);model,_=load_checkpoint(a.checkpoint);model.eval()
    if a.int8:model=quantize(model,a.engine)
    paths=[Path(s) for s in Path(a.wav_list).read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(paths)!=100 or len(set(paths))!=100:raise ValueError('Supply 100 distinct 3-second WAV paths')
    features=[mfcc(p)[None,:,:] for p in paths]
    import wave
    for path in paths:
        with wave.open(str(path),'rb') as w:
            if abs(w.getnframes()/w.getframerate()-3.)>.01:raise ValueError('Expected 3-second utterances')
    def run(x):
        output=model(x,torch.tensor([x.shape[1]]))
        candidates=prefix_beam(output['ctc_logits'][0].log_softmax(-1).tolist(),10)
        return rerank(model,output['representation'],candidates,a.alpha,5)
    process=psutil.Process();samples=[];stop=threading.Event()
    def monitor():
        process.cpu_percent(None)
        while not stop.wait(.5):samples.append((process.cpu_percent(None)/psutil.cpu_count(),process.memory_info().rss))
    with torch.inference_mode():
        for i in range(20):run(features[i%len(features)])
        thread=threading.Thread(target=monitor,daemon=True);thread.start();latencies=[]
        start=time.perf_counter()
        try:
            for x in features:
                t=time.perf_counter();run(x);latencies.append(time.perf_counter()-t)
        finally:stop.set();thread.join()
        elapsed=time.perf_counter()-start
    report={'platform':platform.platform(),'torch':str(torch.__version__),'threads':a.threads,
            'int8':a.int8,'scope':'MFCC feature input through CTC and semantic reranking; generation excluded',
            'warmup_runs':20,'utterances':100,'latency_ms':1000*sum(latencies)/100,
            'rtf':sum(latencies)/300,'throughput_per_second':100/elapsed,
            'sampled_peak_rss_bytes':max([s[1] for s in samples]+[process.memory_info().rss]),
            'mean_cpu_percent_total_capacity':sum(s[0] for s in samples)/len(samples) if samples else None,
            'power_watts':None,'note':'RSS is sampled, not exact peak. Power requires external meter. No reported manuscript values are inserted.'}
    Path(a.out).write_text(json.dumps(report,indent=2),encoding='utf-8')
if __name__=='__main__':main()
