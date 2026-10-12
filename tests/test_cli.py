"""Tiny artificial fixtures test execution only, never manuscript performance."""
import unittest,tempfile,sys,subprocess,json,csv,wave,os
from pathlib import Path
import numpy as np
from culture_speech.model import SpeechSemanticModel,Config,save_checkpoint

class CLITests(unittest.TestCase):
    def test_real_file_training_evaluation_quantization_interfaces(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root=Path(directory)
            vocab=['<blank>','<bos>','<eos>','<unk>','甲','乙']
            initial=root/'initial.pt';save_checkpoint(initial,SpeechSemanticModel(Config(6,hidden=8,embedding=6,acoustic_layers=1,semantic_layers=1,dropout=0)),vocab)
            records=[]
            for i,split in enumerate(['train','validation','test']):
                path=root/f'{i}.wav';signal=np.rint(np.sin(np.arange(24000)*2*np.pi*(400+i*100)/16000)*2000).astype('<i2')
                with wave.open(str(path),'wb') as w:
                    w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(signal.tobytes())
                records.append({'sample_id':str(i),'speaker_id':str(i),'split':split,'status':'collected',
                    'model_audio_path':path.name,'transcript':'甲乙','cultural_intent_label':'创意请求',
                    'generation_subtype':'古诗续写','reference_text':'乙甲','reference_verified':True})
            manifest=root/'manifest.jsonl';manifest.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records),encoding='utf-8')
            def run(module,args):
                env=os.environ.copy();env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
                result=subprocess.run([sys.executable,'-m',module,*map(str,args)],capture_output=True,text=True,encoding='utf-8',env=env)
                self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
            run('culture_speech.train',['--manifest',manifest,'--root',root,'--checkpoint',initial,'--out',root/'run','--epochs',1])
            checkpoint=root/'run'/'best.pt';self.assertTrue(checkpoint.is_file())
            for suffix,int8 in [('fp',False),('int8',True)]:
                run('culture_speech.evaluate',['--checkpoint',checkpoint,'--manifest',manifest,'--root',root,'--out',root/f'{suffix}.csv',*(['--int8'] if int8 else [])])
                with (root/f'{suffix}.csv').open(encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
                self.assertEqual(len(rows),1);self.assertEqual(rows[0]['unit'],'character')
            run('culture_speech.quantize',['--checkpoint',checkpoint,'--out',root/'int8.pt'])
            self.assertTrue((root/'int8.pt').is_file())
            news=root/'news.jsonl'
            news.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in [
                {'split':'train','text':'甲甲','label':'a'}, {'split':'train','text':'乙乙','label':'b'},
                {'split':'validation','text':'甲乙','label':'a'}, {'split':'validation','text':'乙甲','label':'b'}]),encoding='utf-8')
            run('culture_speech.pretrain_text',['--input',news,'--task','semantic','--checkpoint',initial,'--out',root/'news.pt','--epochs',1])
            poetry=root/'poetry.jsonl'
            poetry.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in [
                {'split':'train','prompt':'甲甲','target':'乙乙'},
                {'split':'validation','prompt':'甲乙','target':'乙甲'}]),encoding='utf-8')
            run('culture_speech.pretrain_text',['--input',poetry,'--task','generation','--checkpoint',root/'news.pt','--out',root/'text.pt','--epochs',1])
if __name__=='__main__':unittest.main()
