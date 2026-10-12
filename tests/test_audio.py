import unittest,tempfile,wave,subprocess,sys,json
from pathlib import Path
import numpy as np
from culture_speech.data import mfcc,SpeechDataset,collate
from culture_speech.metrics import generation_metrics

class AudioTests(unittest.TestCase):
    def test_antialias_resample_and_mfcc(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory);raw=root/'raw.wav';output=root/'model.wav'
            samples=np.rint(np.sin(2*np.pi*440*np.arange(96000)/48000)*.1*8388607).astype(np.int32)
            b=np.column_stack([samples&255,(samples>>8)&255,(samples>>16)&255]).astype(np.uint8)
            with wave.open(str(raw),'wb') as w:
                w.setnchannels(1);w.setsampwidth(3);w.setframerate(48000);w.writeframes(b.tobytes())
            tool=Path(__file__).resolve().parents[1]/'tools'/'resample.py'
            subprocess.run([sys.executable,str(tool),str(raw),str(output)],check=True)
            with wave.open(str(output),'rb') as w:
                self.assertEqual((w.getframerate(),w.getsampwidth(),w.getnchannels(),w.getnframes()),(16000,2,1,32000))
            x=mfcc(output);self.assertEqual(x.shape[1],39);self.assertTrue(np.isfinite(x.numpy()).all())
            row={'sample_id':'S1','speaker_id':'P1','split':'train','status':'collected',
                 'model_audio_path':'model.wav','transcript':'甲乙','cultural_intent_label':'创意请求','reference_text':'乙甲','reference_verified':True}
            manifest=root/'records.jsonl';manifest.write_text(json.dumps(row,ensure_ascii=False)+'\n',encoding='utf-8')
            ds=SpeechDataset(manifest,root,'train',['<blank>','<bos>','<eos>','<unk>','甲','乙'])
            batch=collate([ds[0]]);self.assertEqual(batch['decoder_input'].tolist(),[[1,5,4]])
    def test_identical_generation_scores(self):
        result=generation_metrics(['甲乙丙丁'],['甲乙丙丁'])
        self.assertAlmostEqual(result['bleu2_character_0_100'],100)
        self.assertAlmostEqual(result['rougeL_character_0_100'],100)
if __name__=='__main__':unittest.main()
