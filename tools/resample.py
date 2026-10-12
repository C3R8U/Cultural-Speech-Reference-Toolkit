import argparse,wave
from pathlib import Path
import numpy as np
from scipy.signal import resample_poly

def main():
    p=argparse.ArgumentParser();p.add_argument('input');p.add_argument('output');a=p.parse_args()
    with wave.open(a.input,'rb') as w:
        if (w.getframerate(),w.getsampwidth(),w.getnchannels())!=(48000,3,1):raise ValueError('Expected PCM24 48k mono')
        raw=np.frombuffer(w.readframes(w.getnframes()),dtype=np.uint8).reshape(-1,3)
    values=raw[:,0].astype(np.int32)|(raw[:,1].astype(np.int32)<<8)|(raw[:,2].astype(np.int32)<<16)
    values=np.where(values & 0x800000,values-0x1000000,values)/8388608.
    output=np.clip(np.rint(resample_poly(values,1,3)*32768),-32768,32767).astype('<i2')
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    with wave.open(str(path),'wb') as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(output.tobytes())
if __name__=='__main__':main()
