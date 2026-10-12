"""Real WAV/JSONL input only; planned collection prompts are never training text."""
import json, unicodedata
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

INTENTS = {'文化咨询':0, '情感表达':1, '创意请求':2}

def normalize(text):
    # Explicit reference policy; original manuscript numeric-expression policy is unavailable.
    text = unicodedata.normalize('NFKC', text)
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))

def read_records(path):
    with open(path, encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]

def build_vocab(records):
    characters = set()
    for row in records:
        if row['split'] != 'train': continue
        if row.get('transcript'): characters.update(normalize(row['transcript']))
        if row.get('reference_text'): characters.update(normalize(row['reference_text']))
    if not characters: raise ValueError('No actual training transcripts; prompts cannot substitute for speech')
    return ['<blank>', '<bos>', '<eos>', '<unk>']+sorted(characters)

def mfcc(path):
    import librosa, numpy as np, wave
    with wave.open(str(path),'rb') as wav:
        if (wav.getframerate(),wav.getsampwidth(),wav.getnchannels()) != (16000,2,1):
            raise ValueError(f'{path}: expected 16 kHz / PCM16 / mono')
    audio, rate = librosa.load(path, sr=None, mono=False)
    if audio.size == 0 or not np.isfinite(audio).all(): raise ValueError('Invalid waveform')
    static = librosa.feature.mfcc(y=audio, sr=rate, n_mfcc=13, n_fft=512, win_length=400,
                                 hop_length=160, window='hann', center=False, n_mels=40)
    if static.shape[1] < 9: raise ValueError('Utterance too short for width-9 deltas')
    features = np.concatenate([static,librosa.feature.delta(static),librosa.feature.delta(static,order=2)],axis=0)
    return torch.from_numpy(features.T.copy()).float()

class SpeechDataset(Dataset):
    def __init__(self, manifest, root, split, vocab):
        self.root = Path(root).resolve()
        self.vocab = {c:i for i,c in enumerate(vocab)}
        self.rows = [r for r in read_records(manifest) if r['split']==split]
        if not self.rows: raise ValueError(f'Empty split {split}')
        for r in self.rows:
            if r.get('status') != 'collected' or not r.get('transcript'):
                raise ValueError('Uncollected scaffold rejected; supply real verified records')
            if r.get('cultural_intent_label') is not None and r['cultural_intent_label'] not in INTENTS:
                raise ValueError('Unknown actual intent label')
            if r.get('reference_text') and r.get('reference_verified') is not True:
                raise ValueError('Generation target requires explicit reference_verified=true')
            path = (self.root/r['model_audio_path']).resolve()
            if not path.is_relative_to(self.root) or not path.is_file(): raise ValueError('Missing or unsafe audio path')
            r['_path'] = path
    def encode(self, text):
        return torch.tensor([self.vocab.get(c,3) for c in normalize(text)],dtype=torch.long)
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        features, target = mfcc(r['_path']), self.encode(r['transcript'])
        if not len(target): raise ValueError('Empty normalized transcript')
        required = len(target)+int((target[1:]==target[:-1]).sum())
        if len(features)<required: raise ValueError('CTC target requires more frames than available')
        reference = r.get('reference_text')
        label = INTENTS.get(r.get('cultural_intent_label'),-1)
        if label==2 and not reference: raise ValueError('Creative request needs verified generation reference')
        if reference and label != 2: raise ValueError('Generation reference must belong to creative request')
        generation = self.encode(reference) if reference else None
        return {'features':features,'target':target,'label':label,'generation':generation,'record':r}

def collate(rows):
    inputs, targets = [], []
    for row in rows:
        g = row['generation']
        inputs.append(torch.cat([torch.tensor([1]),g]) if g is not None else torch.tensor([0]))
        targets.append(torch.cat([g,torch.tensor([2])]) if g is not None else torch.tensor([0]))
    return {'features':pad_sequence([r['features'] for r in rows],batch_first=True),
            'lengths':torch.tensor([len(r['features']) for r in rows]),
            'ctc_targets':torch.cat([r['target'] for r in rows]),
            'ctc_lengths':torch.tensor([len(r['target']) for r in rows]),
            'intent_labels':torch.tensor([r['label'] for r in rows]),
            'decoder_input':pad_sequence(inputs,batch_first=True),
            'decoder_target':pad_sequence(targets,batch_first=True), 'records':[r['record'] for r in rows]}
